# A built-in chat client — evaluation

**Status: evaluation; the loop itself is not built.** Evaluated 2026-08-24
against `1d542391`; the in-process loop spike below was run 2026-08-25 and
settles the hand-roll-vs-vendor question.

The loop has since been built, and so has a second engine beside it: the pane
can instead host a harness the user already runs, over ACP
([chat-acp-engine.md](chat-acp-engine.md)). That answers the "complementary, not
a substitute" line under *Vendoring* — both are shipped, and which one a session
uses is `chat.engine`.

Three things have since been settled and are marked where they are argued:
chat runs in **local mode only** (*Where chat may run*); the loop must not take
`execute_code`'s promote window (*the one tool not to take as-is*); and
`_PROXY_TIMEOUT` is **not** a constraint, correcting an earlier draft of this
document. Two prerequisites have landed — `--view` session registration
(biopb/biopb#836) and the job-record groundwork the loop will fill,
`origin="chat"` plus a per-cell `intent` (biopb/biopb#843).

Today biopb reaches an AI agent one way: the user runs their own MCP client
(Claude Code, Cursor, …), which spawns `biopb-mcp` over stdio. That excludes
everyone unwilling to install and configure a coding-agent harness — which is
most microscopists. This asks what it would take to ship a chat window of our
own, and what it would cost.

## What already exists

Most of the runtime is in place, and three pieces matter more than the rest.

**An agentless session.** `biopb mcp view` (`src/main/python/biopb/cli.py:576`,
`biopb-mcp/src/biopb_mcp/mcp/__main__.py:181`) opens napari for a human and
serves `/mcp` anyway. No new process topology is needed to host a chat loop.

**A second writer into the kernel, already designed.** The user console
(`biopb-mcp/docs/user-console.md`, `mcp/_jobs.py:14-21`) put a human and the
agent on one job runner: one job at a time, no preemption, and a
`foreign_digest` so the agent learns its namespace moved under it. A chat agent
is a third writer of that same class and inherits the design rather than
reopening it — `origin="chat"` and the `intent` field are already in `_jobs`,
so the loop fills a record that exists rather than adding one.

**And it is a third writer, not a third *agent*.** The kernel now admits one
agent per lifetime: the first non-user submitter claims it, and a second client
is refused by everything that changes kernel state — `execute_code`,
`interrupt_kernel`, `restart_kernel` — keeping only the read-only tools. So a
chat loop and an attached MCP harness are **mutually exclusive**: whichever runs
first holds the session, and the other cannot take it. That is a deliberate
product call rather than a limitation to design around — two agents in one
namespace can be serialized but not reconciled, because neither can see the
other's model of what the variables and layers are.

Two things follow for the loop. It must supply a stable writer id of its own,
which in-process it trivially can. And **the recovery is the user's**: the
observe page's restart is never gated, so the chat UI is the natural place to
surface "another client holds this kernel — restart it?" rather than leaving a
refused agent to explain itself in prose.

**A working agent loop, in the test tree.** `_tests/agentbench/` drives a real
session with a real model: MCP client, schema translation, tool dispatch,
conversation loop, trace. It is benchmark-shaped, not product-shaped — see
"What the harness does not give us" below.

## Where the loop runs

**Not in a napari dock widget.** The kernel is a *grandchild* of the MCP server
process (`mcp/_kernel.py` spawns it via `jupyter_client`), so a widget docked in
the kernel's napari can reach `_jobs`, `_conn`, `ops` and the `viewer` proxy as
live objects — `mcp/_bootstrap.py:805-818` binds exactly those — but **not** the
tools, `find_skills`, or the `skill://` / `guide://` resources, which are defined
in the parent. Two further facts make it the wrong home:

- `restart_kernel` is a tool the agent can call, and closing the napari window
  tears the kernel to idle. A chat widget hosted there **can be destroyed by its
  own agent**, taking the conversation with it.
- It cannot discover its own `/mcp` port. The port is written to
  `$BIOPB_PORT_REPORT_FILE` by the session child (`mcp/__main__.py:382-392`) and
  only when the shim sets that variable, so under `biopb mcp view` no such file
  exists. The kernel is launched with `os.environ.copy()` plus dask/death-watch
  FDs (`mcp/_kernel.py:280-327`) and learns nothing else.

**The loop belongs in the session child**, for the same reason the dask cluster
does: it must outlive kernel restarts. That process already owns `KernelHost`,
already serves `/mcp` and `/api/*`, and is already where things that survive a
restart are put.

This also makes the UI disposable. With the conversation held server-side, a
kernel restart kills the *view* and not the thread — which is what makes the
embedding question below safe to get wrong.

## The tool surface: in-process, not over the wire

An in-process loop should call the tools as **Python functions** and take their
schemas from the FastMCP registry. Verified against the installed SDK
(`mcp` 1.27.2):

```
@mcp.tool() returns <class 'function'>, callable directly
mcp.list_tools() yields the same name + inputSchema the wire carries
```

So `_server.execute_code(...)` is an ordinary call, and the function-calling
payload is generated from `mcp._tool_manager` — one definition, no transport.

This matters because it is easy to conclude the opposite from `agentbench`, which
routes over streamable HTTP (`_tests/agentbench/_session.py:767`). It does that
**deliberately**: it is a test harness whose purpose is exercising the real
external-client path, and its own docstring argues that a hand-written tool
surface would stop tracking the runtime. A shipped in-process loop has no such
requirement. The drift risk comes from *re-declaring* tools, not from skipping
the wire.

Going in-process drops the `mcp` client SDK, streamable HTTP, `_bridge.py`'s
translation (bar a few lines stripping `$schema` / `title`, which some providers
reject), and the `CLIENT_TOOLS` resource shim
(`_tests/agentbench/_session.py:345-378`) — `skill://` and `guide://` become
direct `_skills` / `_resources` calls. `take_screenshot` improves too: it already
returns `ImageContent` (`mcp/_server.py:653`), so in-process we hold a real image
block instead of the harness's stringified placeholder.

An internal MCP *client* still earns its place in two narrow cases, neither
required now: letting the chat agent use third-party MCP servers a user has
configured, and moving the loop out of the session child later. Both are
additive; the tool surface is the same either way.

**The in-process dispatch returns one of two shapes, and a caller collapses
them.** `_tool_manager.call_tool(..., convert_result=True)` yields a bare
`list[ContentBlock]` for a tool with no structured output schema, and a
`(blocks, structured)` tuple for one that has it. The split follows the return
annotation exactly: the seven `-> str` tools carry an `outputSchema` and return
the tuple; `find_skills` and `take_screenshot` (`-> list`) do not.

This is the layer's declared contract, not a defect and not undocumented —
`lowlevel/server.py:105-106` names both halves (`UnstructuredContent`,
`CombinationContent`) and `FastMCP.call_tool` is annotated as the union. The
low-level server collapses them at `lowlevel/server.py:545`
(`isinstance(results, tuple) and len(results) == 2`) on its way to one
`CallToolResult`. Calling in-process puts us below that layer, so we do its job:
three lines, mirroring its check.

**Normalizing the nine tools instead is possible and is the wrong trade.**
`@mcp.tool(structured_output=False)` (`server.py:406`) would make every tool
return the bare shape — but seven tools currently advertise `outputSchema` in
`tools/list` and return `structuredContent` to real MCP clients, and dropping
that to save three lines in one in-process caller is a wire contract change
serving the wrong party. (Nothing is lost informationally: the JSON is also
serialized into text content, `lowlevel/server.py:501-502`.) The reverse is worse
— `take_screenshot` returns content blocks, which have no business in structured
output.

There *is* a real cleanup here, and it is a different one: the `-> str` / `-> list`
split is incidental rather than chosen. Making it deliberate is worth doing for
the contract external agents see, not for this. What is worth adding either way
is a test pinning the shape per tool, so a FastMCP bump or a casual annotation
change fails loudly instead of silently reshaping what the loop receives.

### `execute_code` is the one tool not to take as-is

It carries a wire-shaped parameter that the rule above would otherwise import by
accident. `kernel.promote_after` (10 s, `biopb-mcp/src/biopb_mcp/_config.py:260`)
makes `execute_code` wait that long for the job to finish before handing back a
handle. Over streamable HTTP that is a latency optimization — a poll is a whole
round trip, so inlining a fast result saves one. **In-process the saving is
zero**: a poll is a function call.

For a chat surface it is not neutral, it is a cost. The model is blocked on the
tool result, so the stream emits nothing for those seconds: a frozen cursor, then
a wall of text. And it is the common case rather than the rare one — most cells in
an image-analysis session (a segmentation, a `compute()`) run past 10 s and
promote anyway, having paid the wait first.

It cannot simply be turned down. `_promote_after` is a **process global**
(`mcp/_server.py:28`), set once by the launcher from config
(`mcp/__main__.py:386`) and shared with any external harness attached to the same
session — for which 10 s is the right value. Lowering it for chat changes the
contract the MCP agent is working under.

The shape that follows: the loop submits with **no** promote window and streams
the job's partial stdout as it accumulates. `_jobs` already buffers stdout per job
and `poll_job` already returns it as partial output, so this is polling a
`StringIO` in-process — free at a few hundred ms. It is strictly better than
waiting, because the user watches prints appear instead of a stalled cursor.

So either the in-process dispatch takes a per-call override, or the loop calls
kernel submit/poll directly and leaves `execute_code` to the MCP surface. Worth
writing down: it reads as an inconsistency with "call the tools as Python
functions" otherwise, and it is the only tool where that rule needs an exception.

## What the harness does not give us

Roughly 230 of `_session.py`'s 804 lines are reusable client mechanics; the rest
is scaffolding (guard tripwires, wheel-staging, forced config trees). And the
most visible machinery in `_conversation.py` exists *because no human is
present*: the `Respondent` persona hand-off, the `__BIOPB_TASK_COMPLETE__`
sentinel, and idle-stall detection are all replaced by a real user. What carries
over is narrower: tool-call dispatch, tool-result message construction, the
provider reasoning-field echo (`_models.py:306-320`), and the trace log.

Genuinely absent, and needed:

| Gap | Where it stands |
| --- | --- |
| Streaming | none — `_models.py` `complete()` blocks |
| Conversation history | unbounded; no summarization or context-length handling |
| Cancellation | `_jobs.interrupt_current` exists; the loop never exposes it |
| Vision | wiring only — see the spike; `ImageContent` reaches a model fine in-process |
| Key custody | nothing; keys come from env / `.env` (`_models.py:41-47`) — bounded by *Where chat may run* below: local mode only, so the keys never face a public origin |
| Cost guards | no token accounting, no retry/429 handling |
| Tool-output truncation | informal only, no general guard |
| `instructions` as system prompt | captured at `_session.py:718`, never used |

## Vendoring

The `mcp>=1.20,<2` pin (`biopb-mcp/pyproject.toml:118`) is **not** a blocker.
`mcp` 1.x is still actively released, and the ecosystem pins the same way —
`fastmcp-slim`, `langchain-mcp-adapters` and `strands-agents` all cap at `<2`.
Every candidate surveyed resolved onto `mcp==1.29.1`.

- **`openai-agents`** — +12 packages; streaming, `SQLiteSession`,
  `cancel(mode=…)`, MCP image mapping, and MCP resources (verified in source; its
  docs omit them). Hard `openai<4` core dep.
- **`strands-agents`** — core constraint `mcp<2.0.0,>=1.23.0`, exactly ours;
  richest MCP surface, real HITL interrupts. Costs `boto3` + OpenTelemetry.
- **`claude-agent-sdk`** — strongest feature set, but bundles a 342 MB
  Bun-compiled binary (no Node needed), is Claude-only, and its terms bar
  third-party claude.ai login, so every user needs their own API key. Optional
  extra at best; never vendored into an installer.
- **`agent-client-protocol`** — sole dep `pydantic>=2.7`; hosts whatever harness
  a user already runs. Complementary, not a substitute: it assumes they have one.
- **Dead:** `any-agent` (soft-deprecated by its README), `mcp-use` (no PyPI
  release since 2026-03 despite commits; now fronts itself as a TypeScript
  project).

**Verdict after the spike: hand-roll.** Much of the pitch for `openai-agents` is
its MCP-client surface, and calling tools in-process spends none of it. What
remains is streaming, sessions and cancellation — a thin case for a dependency
that reaches into the loop's control flow. Nothing found during the spike argues
the other way: the one candidate, the two-shaped tool return, turned out to be a
declared contract collapsed in three lines rather than a reason to take on a
client. Measured, the loop is **134
non-comment lines**, 55 of which are the two synthesized resource-tool schemas
(pure data), so the machinery is under 100 lines. It needs **no provider SDK**:
the spike's model call is raw `httpx`, about 12 lines, because `openai` is not in
the shared venv and adding it was not worth doing. Every hard thing the spike hit
was biopb-specific plumbing a framework would not have known about either.

Revisit this if the chat agent must consume third-party MCP servers, or if
durable mid-run resume becomes a requirement.

## UI

**Docked vs. separate window is not a build-time decision.**
`QtViewerDockWidget` subclasses `QDockWidget` and napari never strips
`DockWidgetFloatable` / `DockWidgetMovable`, so the user undocks it onto a second
monitor and back at will. `add_dock_widget` also takes `area=` and `tabify=`.

The real fork is Qt-in-napari vs. browser, and because the loop is server-side
and the UI is a web route, the Qt option is a `QWebEngineView` onto that same
route. One UI either way; the question is only whether we also ship the wrapper.

**It should be the observe page evolved, not a sibling.** That page already shows
job history, the code the agent ran, truncated output, interrupt and notebook
export. Chat is the missing *input* affordance. A separate route would render
overlapping state in two places and duplicate the job display.

The safety argument agrees. For users who cannot read the code, making the
executed code unavoidable — rather than tucked behind a transcript disclosure —
is the mitigation; if chat and job history are one surface, that comes free.

Results go back on the viewer (the server instructions already say so), so the
chat pane is **not** an image-rendering surface. It needs a message thread, the
agent's code inline and collapsible, a stop button, and clickable layer
references — a narrow column, which is what a napari right dock is.

For the web surface, `@ai-sdk/react`'s `useChat` (53 KB gz, Apache-2.0, headless)
against a Starlette endpoint implementing its published stream protocol is the
lightest fit for a plain-CSS SPA; the protocol is a spec we implement in ~60
lines, not a dependency. Both Python helper packages for it are abandoned
single-release projects. `assistant-ui` is richer but wants Tailwind and 181 KB.

Do **not** mount Chainlit / Gradio / Panel / NiceGUI. Four of them do mount
inside a Starlette app, so they pass the single-origin test — and it does not
help: each brings a second complete frontend (own bundle, own socket transport,
own state model and visual language) into an app that already has a React SPA.
Reflex is disqualified outright; it inverts the relationship and hosts *us*.

`PyQt6-WebEngine-Qt6` is `manylinux_2_34` — the same tag `PyQt6-Qt6` already
carries — so it costs ~116 MB of download and **no platform reach**; the glibc
floor is already set. Licensing adds no obligation (GPL-3.0-only, as `pyqt6`
already is via `napari[all]`). The risks are integration, not size: WebEngine
must be initialized before `QApplication` when it lives in a plugin (which a
napari dock widget is), and napari#2584 reports a VisPy OpenGL context conflict —
macOS-only, 2021, PyQt5 / napari 0.4.7, **unverified on napari 0.7 + PyQt6**.

## Where chat may run — local mode only

**Decided: chat is off when the control is `--remote`.** Not a new gate; the same
one the user console already argued for and built, reached by the same reasoning.

`_SESSION_ALLOWED_ROOTS = {"api"}` (`biopb-control/.../_control.py:123`) is an
allowlist whose purpose is to keep `/mcp` — "an RCE hole on the public origin",
per `session_proxy`'s own comment — off the proxied origin. A chat route is a
**second execute-capable surface**, so it inherits the console's shape: its own
path root, proxied only while the control is loopback-bound, with the switch that
makes it reachable being the same switch that guards it (`console_enabled`, and
`observe.console_enabled` as the opt-out precedent — a knob may narrow the
surface, never widen it past the public-bind refusal).

Reusing the data-plane token instead was already rejected for the console
(`biopb-mcp/docs/user-console.md`, *Why not "just gate it with the token"*): that
token authorizes reading pixels and restarting the plane, lives in a
same-uid-readable credential file, and rides the render WebSocket as `?token=`
into logs. Those properties are fine for "view my data" and wrong for "run code as
me". If a remote chat is ever wanted it needs a **distinct** credential.

Chat adds one thing the console did not, and it is worth stating because it is not
covered by any argument above: the loop **spends the user's model credits**. A
leaked viewing credential becoming a shell was already the objection; it becoming
a billable shell is a second one, and it is the only part of the surface where the
damage is not bounded by what the machine can reach.

The useful consequence is that **key custody becomes a local problem**. The keys
live on the machine the user is sitting at, alongside the napari window, and never
have to survive a public origin.

## Design constraint: do not make the model read pixels

**The number to use is the ablation, not the score.** `MicroVQA` (CVPR 2025,
1,042 expert microscopy questions) is *five-way multiple choice*: chance is 22.0,
the human expert baseline is 50.3, and the top model scores 52.8. Do not read
52.8 as an estimate of chat performance — the authors report that under
open-ended prompting the same models "rarely gave good responses" and "tend to
give very vague answers, and tend to depend strongly on the text part of the
input prompt." The honest figure for a chat box is unmeasured and lower.

What transfers is the no-image ablation (their Table 7): remove the picture, ask
anyway, and the best model drops only **52.8 -> 49.2**. On a benchmark built to
be vision-centric, against a 22.0 floor, the entire image-derived advantage is
**3.6 points**. Expert review of chain-of-thought attributes 50% of errors to
perception, 30% to knowledge, 13% to overgeneralization (n=30, one model — treat
as ordinal). And the automated extension found that supplying the image
*increases* perception errors while decreasing hallucination: pixels convert one
failure mode into another rather than removing failure.

Getting a screenshot to a model is, however, a solved wiring problem rather than
a hard one — the spike below did it end to end. The constraint is **model
capability, and it is hard**: a non-multimodal model does not degrade, it
hard-fails. The configured bench agent `deepseek-v4-flash` returns
`400 ... is not a multimodal model` from the gateway on any request carrying an
image. A chat window that screenshots must gate on capability or refuse
deliberately; it cannot simply send and hope.

So images should reach the model — the agent needs to see what it did — but the
design must not depend on it interpreting them. Feed measurements and
display-ready renders. `fiji-mcp`'s `get_thumbnail`, which returns a PNG with
LUT, contrast and overlays already baked in and tells the agent to call it
liberally as visual ground truth, is a better shape for `take_screenshot` than
raw canvas pixels.

### Silent wrongness is the design target, and it gets worse as models improve

Recomputed from `human-eval-bia`'s raw results (18 models, 57 unit-tested
bioimage tasks, 10 samples each): **25.2% of all runs produce code that executes
cleanly and returns the wrong answer.** The absolute rate stays roughly flat
across the whole competence range, but the *share of failures that are silent*
climbs with capability — 18.8% of codellama's failures, 63.3% of
claude-3-5-sonnet's. **Capability improvement converts loud crashes into quiet
wrong answers.** For a user who cannot read the code, a traceback was a safety
feature, and it is the one that disappears first.

Two supporting details. The pass distribution is **bimodal** — 54% of tasks are
all-or-nothing across ten samples — so "try again" helps only in a middle band
and is worthless on the hard zeros. And the tasks nobody solved are not hard;
they are **data-contract guesses**: 234 of 260 attempts got (Y,X) versus
(width,height) wrong on one task. Models also reach for `cv2` where the domain
uses `skimage`, and `cv2` assumes 8-bit BGR and detonates on 16-bit scientific
arrays. Both are eliminable here rather than merely mitigable: the kernel already
knows every object's shape, dtype and axis labels, so making that mandatory
context — not something the agent may choose to look up — removes the largest
single failure class. It lands directly on the axis-order traps `guide://data`
already documents.

### The largest available lever is plan-first, not a better model

BioDSBench (*Nature Biomed Eng* 2026, 293 biomedical data-science tasks):
standalone LLMs score **under 40%**; an agent that drafts an explicit analysis
plan, refines it, then generates and executes reaches **74%**. That is a bigger
delta than any model upgrade in this literature — and the plan is the one
artifact this audience can actually review. Reviewing an analysis plan in English
is something a microscopist can do; reviewing the Python is not.

### The wider risk: shipping this removes the last reviewer

Every biopb agent user today can read the code the agent ran.
`biopb-mcp/docs/bench-coverage.md` already records that confidently-wrong numeric
answers are the dominant real failure mode and that the suite structurally cannot
score a correct refusal. The prior art says the same thing in first-party terms.

Omega's own authors, in the closest comparable system:

> "LLMs hallucinate facts and occasionally make trivial reasoning mistakes...
> **This is a cause for caution as non-expert users might be led astray by an
> overly confident agent.** Moreover, it is incumbent on the user to explain the
> task clearly and unambiguously in natural language."

That last clause is the assumption that fails for the audience a chat window is
*for*. The BioImage.IO Chatbot authors name the same risk ("plausible but
incorrect information") and mitigate differently, with RAG grounding rather than
a UI affordance — and neither team measured whether their mitigation worked.

A concrete instance, from a Fiji core maintainer evaluating CopilotJ: *"Note that
GPT measured the wrong image — it measured the binary mask instead of the
original image — but hey, it's fundamentally working now."* A confident, plausible,
wrong quantification, caught only because an expert read it.

And the sharpest one, from Agentic-J: a workflow that missed half the true
positives (5/10 sensitivity, because 7 unresolved ROIs were silently defaulted to
negative) produced a plot separating the groups at **p = 0.016** against the
expert ground truth's **p = 0.015**. **"The plot looks right" is not evidence.**
Validation has to sit upstream of the summary statistic, with coverage counts
("38 of 45 assigned; 7 dropped") surfaced as prominently as the result, and
"could not resolve" kept as a distinct outcome rather than folded into a class.

Note also what is *not* evidence: there is no filed report anywhere in a readable
tracker of a scientist acting on a confidently-wrong agent result. Every
anti-overtrust affordance found in the survey was written prospectively by a
maintainer. The only recorded incidents come from a CHI 2025 interview study, and
only because the researchers read the participants' logs. **Empty issue trackers
are not reassurance** — this failure mode does not generate bug reports.

This is a product decision, not an engineering one, and it should be taken
deliberately.

## Prior art

Surveyed 2026-08-25 across napari and Fiji/ImageJ, plus general scientific chat
UIs (Jupyter AI, bia-bob, Positron/Posit Assistant, VS Code notebooks, MATLAB
Copilot). `forum.image.sc` was excluded deliberately — it blocks automated
fetching — and that exclusion is material: `fiji-llm` and `bia-bob` route **all**
bug reports there, so end-user complaints in exactly our domain are invisible
here.

**The field is small and we are already in it.** Of 662 plugins in the napari
hub, four are LLM/chat: `napari-chatgpt`, `napari-chat-assistant`, `napari-mcp`,
and `biopb-mcp`. QuPath, OMERO and Micro-Manager have none at all.

### The architecture finding, and the tension with ours

`fiji/fiji-llm` (official Fiji org, BSD-2, active) does something sharper than
"one registry, two consumers": it starts an MCP server on `:9090` from its tool
registry, then **builds an MCP client back to its own server** and feeds that to
the in-app chat. The chat is just another MCP client. There is no
direct-registry path, so parity with external agents is *structural* rather than
a thing maintainers must remember. The enforcing rule is its issue #7 — app
context was being injected into chat messages, and the fix was to make it a tool
*"so that external agents also have access to it."*

**This is a genuine argument against the in-process dispatch recommended above.**
Calling the tool functions directly is cheaper and simpler, and the spike shows
it works — but it reopens exactly the drift `fiji-llm` closed by construction:
nothing stops someone giving the chat a capability that never becomes a tool, and
external agents silently lose parity. The mitigation, if we stay in-process, is
to adopt their rule as a *test* rather than a habit: assert that the chat's
capability set is exactly the registry's. Cheap, and it fails loudly.

The counterweight is developed under *Recovery* below: only an in-process chat can
put the user's intent into the session record, because under MCP the prompts never
reach biopb at all.

### Permission: what everyone got wrong

**No project in either ecosystem asks per-action permission from inside its own
chat window.** More usefully, three built the mechanism and shipped without
wiring it:

- CopilotJ has `request_user_confirm` across five backend files; the frontend is
  `case "confirmation_request": // TODO: add a ui component`.
- Agentic-J implements `ask_user` as `input()`, registered on no agent.
- `fiji-mcp`'s `ExecutionLock` glass pane is a known no-op.

In all three the gate degraded into *a prompt instruction asking the model to
please ask permission* — **which reads exactly like a working gate in code
review.** The only gates that actually work came from removing the execution
channel (`fiji-llm` ships no execute tool) or reusing a gesture the human already
makes: bia-bob's Shift+Enter, `ijpb`'s Run button, or `fiji-llm` permitting only
commands ending in `...` — the ones that raise ImageJ's own parameter dialog, so
the user's normal OK *is* the approval. **If we build a gate, build the UI first
and the transport second.**

Two framings worth taking. Janelia's *AI in Microscopy* ch.3 (Ouyang & Zhang)
gives the taxonomy nobody else states — **assisted** (human approves each action)
-> **supervised** (routine automatic, consequential confirmed) -> **autonomous**
— and names the failure mode **"sleepwalking, where an agent takes confident but
incorrect actions."** And `fiji-llm` issue #22 resolves the gate question by
*scope* rather than by policy: gated for the in-app chat, open on the MCP surface,
making explicit *"the difference in trust model between 'assistant sitting next
to me' and 'agent I deliberately connected'."*

One caution that lands directly on us: an `execute_code` tool **subsumes every
other gate**. Where a project gates package installs or file writes but exposes
arbitrary Python, the gates are decorative. Omega is the pure case — its LLM
safety rating (A-E) exists, and its only caller is a manual button in a code
editor, never the execution path; its one per-action prompt is for `pip install`.

### Binding: 7 of 10 exposed

`fiji/fiji-llm` binds Jetty to all interfaces with no auth and no `Origin` check
(its maintainer opened #23 to fix it). `Agentic-J` ships noVNC on `0.0.0.0`
**unauthenticated**, in a compose file that contradicts its own adjacent comment
— a full desktop running arbitrary Groovy, open to the LAN. CopilotJ pairs
wildcard CORS with `allow_credentials=True` behind a GitHub-Pages-hosted
frontend, so any page the user visits could POST macros to their loopback server;
loopback is no defence there, and same-origin would have been.

Every one of these authors reasoned "it only runs locally, so it's fine." The
*official* Fiji project has the bug. Our two-mode security model already frames
this correctly — the finding is that nobody else got it right by accident, so it
should be checked deliberately for the chat surface rather than assumed.

### Showing the work

The convergent rule, cleanest in Jupyter AI's source: **expand before the
decision, collapse after** — pending-approval tool calls render `<details open>`,
completed ones collapsed.

Omega inverts it, and the inversion is instructive. Code blocks over ten lines
collapse, and auto-expand only on `/error|failed|exception/i`. **Work is revealed
only when it failed** — so the ran-clean-but-wrong case, which is the 25% of runs
this document is most worried about, is precisely the case that stays hidden.

The single most-filed bug across every tracker surveyed is the agent reporting
success it did not achieve. **Never render "done" from the fact that a call
returned.** `Fiji-Macro-Bridge`'s answer is worth copying outright: `run_macro`
returns a **change envelope** — log lines added, images opened, table rows — not
raw output, so "what changed" is data rather than narration.

And the hard finding for our audience: **for users who cannot read code, "show
the code" is not a countermeasure.** Posit argued this about their own product —
*"the only way to know whether Databot followed the right process is to read the
code it generated... without coding ability, it can be hard to verify what it
actually did"* — then deprecated Databot into a permission-gated assistant. What
survives is showing the *artifact*, small steps with forced regroup, and
QP-CAT-style affordances: a confidence column whose own legend says *"treat with
skepticism; high does not mean correct"*, refusal rendered as a first-class
outcome rather than an error, and "plausible-but-wrong" named in the user docs as
a thing to expect.

### Recovery: a wall, and we are already past most of it

**Not one tool surveyed checkpoints live session state — and for a Python
session it cannot be done deterministically.** Settled in this project earlier: a
kernel namespace holds Qt objects, open file handles, dask clients, gRPC channels
and C-extension state; `dill`/`cloudpickle` are best-effort and fail on exactly
those, often silently; and napari's viewer state does not live in the namespace at
all. So the survey result is not unclaimed ground — it is a wall the field already
hit, and the other tools' choices are the workaround rather than a shortfall. VS
Code checkpoints files because the file is the only checkpointable thing there.
Jupyter AI delegates undo to the host document's CRDT, and the mechanism built to
show agent edits live is what caused data loss serious enough to make RTC optional
four months after launch, with a follow-up issue titled *"Forbid notebook writes"*
arguing that agents should use structured, app-aware tools rather than a generic
write path. Fiji's one level of Ctrl-Z is a pixel buffer, not a state snapshot.

**Our equivalent of that document is the session record, and it is already
built.** `_notebook.py` exports the job-ordered `execute_code` history as a
Jupyter notebook with `origin` (agent vs user) recorded per cell — an audit record
with provenance, which is more than Jupyter AI has. Its own docstring states the
limits without varnish: *"an audit record first, a runnable script second"*,
external state not captured, tensor-server source ids and viewer layers gone on a
fresh kernel. That is the correct promise and the same one a notebook makes.

What it captures of intent is second-hand. `execute_code` now takes an optional
`intent` alongside the code, recorded on the job and rendered as the markdown
cell above it, so the export can answer *what was being attempted* and not only
*what ran*. That closes the shape of the gap but not its substance: under MCP the
field is filled by the **agent**, which is the party whose misreading the record
exists to catch. It is best-effort provenance, and an agent that has misunderstood
the task will write down the task it misunderstood.

The gap that matters for the failure mode this document is about therefore stays
open: reading the code cannot catch the agent solving the wrong problem, which is
exactly the GPT-measured-the-binary-mask case and the 25% clean-but-wrong class,
and neither can reading the agent's own account of it. Only the user's words
settle it. **An in-process chat is the only configuration in which the session
record can capture the user's intent at all** — under MCP the prompts live in the
external harness and biopb never sees them; the protocol hands a server tool calls
and nothing else. That is a real counterweight to the parity argument above.

The work the loop then inherits is small, because the record was built to receive
it: fill `intent` with the user's own turn instead of the agent's paraphrase, and
fuse the surrounding chat — the request, the plan, the answer — into the export as
further markdown cells around the code they produced.

**The destructive surface is bounded in the data plane and open everywhere
else.** Inside the data plane the guarantee is real: tensor-server sources are
never mutated in place, and the only write path is `upload`, which creates a *new*
source. But that holds exactly as long as the agent stays in the data plane.
`execute_code` runs arbitrary Python with the whole standard library, so
`open(..., "w")`, `os.remove` and `shutil.rmtree` are one line away and go nowhere
near the tensor server. The read-only property is a property of a *plane*, not of
the session.

`biopb-mcp/docs/agent-fs-guardrail.md` is the live design for closing that, and it
is worth reading before leaning on any FS claim: a PEP-578 audit hook installed at
the tail of bootstrap, denying writes outside an allowlist, with an actionable
`PermissionError` that steers the agent back to `client.upload_array`. It is
**proposed, not implemented**, and it is deliberately honest about being a
speed-bump rather than a boundary — deliberate evasion via `ctypes` or a
`subprocess` is out of scope, and dask workers are a *different process group*, so
without the matching `WorkerPlugin` any file I/O inside a `map_blocks` walks
straight past the kernel's hook. The real boundary is Landlock, and Linux-only.

The napari layer convention is softer still: the agent is *asked* to add new
layers rather than overwrite, and nothing enforces it — `viewer.layers.remove(...)`
or a reassignment of `layer.data` is unguarded. So the accurate summary for a chat
client is that a non-expert's data is protected by a norm and a proposal, not by a
mechanism. That is defensible for the audience biopb has today, because every one
of them can read the code that ran. **It is precisely the assumption a chat window
removes**, which makes the guardrail a dependency of shipping one rather than an
unrelated hardening task.

What stays genuinely open is **pre-execution validation** — the thing both
CopilotJ's and `fiji-llm`'s maintainers name and nobody ships. For them it means
checking parameters against the installed plugin version; here it means checking
shape, dtype and axis labels against the live descriptor *before* the code runs
rather than after it throws. It is also the lever against the largest measured
failure class in `human-eval-bia`, where 234 of 260 attempts on one task got
(Y,X) versus (width,height) wrong.

The framing to drop is "one agent action equals one Ctrl+Z": honest for layers,
dishonest for state, and largely unnecessary given the two controls above.

### Omega is our closest sibling, and we already beat it where it matters

Omega (`royerlab/napari-chatgpt`, BSD-3, 301 stars) is finished rather than
abandoned — its README now points at `napari-mcp` as the successor. Two of its
choices match ours by convergence: the chat is a **browser window** served by a
local FastAPI, with only a control panel in the napari dock, and it drove the
viewer with vision in 2024.

Where it is the worst case is threading: generated code runs on napari's Qt/GUI
thread, so a long segmentation freezes the viewer, and the chat blocks on a queue
`get()` with no timeout and **no Stop button** — the shape of its issue #52,
*"Omega never answers and seems to be stuck."* biopb already has the answer here:
jobs run off the Qt thread, `interrupt_current` exists, and `poll_job` is the
fire-and-poll pattern Jupyter AI independently converged on (*"a timeout does NOT
mean execution failed; the kernel continues running"*). That is a real, existing
advantage rather than something to build.

One more convergence worth noting: Omega **deleted ~650 lines of speculative
code-repair** in 2026 — *"retry loops counterproductive"* — and three days later
added an execution-grounded repair loop that reads the real traceback.
`napari-chat-assistant` concluded the opposite and validates imports against
installed modules with `ast`/`importlib`. Two projects, opposite answers; the
defensible synthesis is to drop static pre-checks and keep repair that is
grounded in what actually happened.

### Verification note

Repository facts (activity, licences, bindings, source quotes) were read
first-hand. Some cross-project audit claims — notably several Agentic-J internals
— are second-hand from a maintainer's audit thread and are flagged as such in the
research record rather than relied on here.

## Corrections to existing docs

`biopb-control/ARCHITECTURE.md:78` and `biopb-mcp/CLAUDE.md:539` state that observe uses SSE and
that the proxy is therefore a streaming passthrough. It polls
(`web/packages/app/src/pages/ObservePage.tsx:154-162`); no `text/event-stream`
exists anywhere in the tree. The passthrough capability is real
(`biopb-control/src/biopb_control/_control.py:1151-1156`, with a pure-ASGI auth
middleware at `:435-439` chosen so it does not buffer).

**`_PROXY_TIMEOUT` is not a constraint on chat — an earlier draft of this document
said it was, and that was wrong.** The claim was that a long tool call between
tokens would exceed the 300 s read bound and 502. It cannot: no biopb tool blocks
anywhere near that. `execute_code` promotes to a job handle at
`kernel.promote_after` (10 s); the longest-blocking tool is `start_kernel`,
bounded by `kernel.startup_timeout` (60 s POSIX, 120 s Windows); `execute_timeout`
is 120 s. The bound is per read-event rather than total, so it is a keep-alive
requirement, and a loop streaming a running job's partial stdout (see
*`execute_code` is the one tool not to take as-is*) has no inter-chunk gap at all.
The proxy already streams — `StreamingResponse(resp.aiter_raw(), …)` at both
sites. And with chat local-only, it may not traverse the proxy in the first place.

What *is* worth fixing is the comment's stated reason. `:552-559` justifies the
number by "no long-poll / SSE / chunked-with-gaps path, so a large slice/render
streams without inter-chunk stalls". That premise stops being true the moment
anything streams, and the next person tuning the value would be reasoning from
it. Correct the justification; leave the number, which has now been re-derived
against the numbers above.

## Prerequisite

`_sessions.register` had exactly one caller, `mcp/_shim.py:296`. The `--view`
path never registered, so agentless sessions were invisible to `/api/sessions`,
the dashboard, and all `/session/<id>/*` proxying.

**Done** — biopb/biopb#836 (`fix/view-session-registration`, against `dev`): the
viewer publishes itself once the kernel is up and drops the record in
`_shutdown`, with the id format moved to `biopb._sessions.new_session_id()` now
that the registry has two writers. It stands on its own regardless of what
happens to chat.

## Sizing

Hand-rolled in-process, with the harness's reusable parts lifted: on the order of
3–4 weeks for a v1 that could be handed to a non-programmer, dominated by
streaming, history management, key custody and cost guards rather than by the
loop. A demo — no streaming, env-var key, existing observe page plus an input box
— is a few days.

Two of those four have since been narrowed. Key custody is a local-mode problem
(*Where chat may run*), so an env-var key is not merely the demo's shortcut but a
defensible v1. And streaming is partly a job-runner question rather than a model
one: the loop must submit without the promote window and stream partial stdout,
which is most of what a user watching a cell run actually wants to see.

One spike remains: `QWebEngineView` in a napari 0.7 / PyQt6 dock widget on Linux,
which settles napari#2584 and the plugin initialization order before anything
depends on it. The in-process loop spike is done — see below.

## Spike: the in-process loop, run

Run 2026-08-25. The loop was placed **inside the real `_serve_http` bring-up** —
actual Xvfb, dask, `KernelHost`, napari — by substituting `_server.run`, the
uvicorn serve loop, for the conversation. That detail is load-bearing: a first
attempt hand-wired its own `KernelHost` and the bootstrap health probe failed
(`status='ok', stdout='False'`), which the model then spent eight turns
diagnosing as a server bug. Reimplementing the launcher's wiring is a trap;
letting the launcher do it is one line.

Model: `deepseek-v4-flash` (and `-vision-exp` where images were involved) over an
OpenAI-compatible gateway. Tools were called as Python functions, schemas read
from `mcp.list_tools()`, resources through `mcp.read_resource`.

**Tool use, cold path.** The agent started the kernel itself:

```
[turn 0] -> start_kernel({})     <- "Kernel ready. The napari viewer, dask, and tensor client are up"
[turn 1] -> execute_code({...})  <- "peak col (X): 128 ... added layer: True"
[turn 2] -> take_screenshot({})  <- 53 chars, 1 image(s)
[turn 3] final
```

**Resources resolve, and the handshake `instructions` work as a system prompt.**
Asked whether a drift workflow existed, the agent called `find_skills`,
dereferenced `skill://drift-correction`, quoted step 1 verbatim, and then asked
permission before acting — the behaviour those instructions were written to
produce. Feeding `mcp._mcp_server.instructions` in directly is all that took;
`agentbench` captures the same string at `_session.py:718` and never uses it.

**Vision, with a control.** In the run above the agent's own stdout printed
`peak col (X): 128`, so a correct answer proved only that the image was
*delivered*. The control placed a blob at (380, 400) before the loop started,
told the agent nothing about it, and forbade reading pixels with code:

```
[spike] preload status=ok
[turn 0] -> take_screenshot({'canvas_only': True})   <- 1 image(s)
[turn 1] final
-> "the bright spot is in the lower-right quadrant"
```

Correct, from a 26 KB data-URI image block, in one turn. Note the shape this
forces: chat-completions tool messages are strings, so the image cannot ride in
the tool result — it travels as its own follow-up user message, and the tool
message carries a placeholder.

**Not covered by the spike:** streaming, cancelling a running `execute_code`
mid-turn, a conversation long enough to reach a context limit, any model family
other than the one gateway's, and anything resembling a real analysis task. The
tasks were toys chosen to exercise plumbing.

Spike code is not in the repo — it was scratch, and the parts worth keeping are
the ~100 lines described under Vendoring.

## What was not verified

The loop itself has now been run against a live session (above), but the
*surveyed packages* have not: streaming, image and cancellation behaviour for
`openai-agents`, `strands-agents` and the rest still come from source signatures
and docs, not observed runtime — the spike was hand-rolled, so it exercised none
of them.

The benchmark figures under "do not make the model read pixels" are quoted from
the `MicroVQA` and `human-eval-bia` abstracts, not re-derived. Agentic-J's 50%
sensitivity is weaker still: it reached this doc through a summarizer over the
arXiv HTML rather than a line-by-line read, so treat it as indicative and check
the paper before it decides anything.

Whether Anthropic would grant a third-party subscription-login exception to an
academic OSS project has no public process.
