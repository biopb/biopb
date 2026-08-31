# The ACP chat engine

Status: **implemented**. The observe page's chat pane can be driven either by
biopb's own in-process loop (`mcp/_chat.py`, the default) or by a coding harness
the user already runs, hosted over the [Agent Client
Protocol](https://agentclientprotocol.com/). `chat.engine` picks; the pane
offers a switch when both are usable.

The built-in loop exists for a microscopist with no agent at all and is
deliberately thin — one OpenAI-compatible call, the user's own key, no plan
mode, no subagents. This is the other half of that audience: someone who already
has a harness gets *that* harness, on their own subscription, driving the viewer
in front of them. See [chat-client-evaluation.md](chat-client-evaluation.md) for
why the built-in loop was hand-rolled and why it stays.

## How biopb reaches the agent

`session/new` carries an `mcpServers` array, and biopb goes in it as **this
session's `/mcp` over http**:

```json
{"type": "http", "name": "biopb", "url": "http://127.0.0.1:<port>/mcp",
 "headers": []}
```

Two things about that are load-bearing.

**It is this session, not a new one.** The biopb entry the installer writes into
a user's client config runs `biopb-mcp --transport stdio`, which the shim turns
into a fresh session child and a second napari window. Registering biopb the
normal way would give the harness a viewer nobody is looking at. The URL comes
from the registry record the session published for itself
(`mcp/__main__.py`), so there is one answer to "where is this session".

**`headers` is required, not optional.** The spec's example omits it; both the
`agent-client-protocol` Python models and opencode's own schema reject the entry
without it. Sending `{"type": "http", "name", "url"}` fails validation with a
union error that names all three transports and explains none of them.

### Why http and not a stdio bridge

The alternative is a stdio `mcpServers` entry pointing at a bridge process
(`biopb-mcp` attaching to the running session instead of spawning one). It would
work, and it would reintroduce the exact process shape the http-only restructure
exists to prevent: a process whose fd 1 is a JSON-RPC channel, in a tree with
Qt/GL/dask native code that writes to fd 1 past Python. That is the "stdio leak"
class, fixed structurally in `f5055bf6` by deleting `run_stdio()` and leaving
stdio to a featherweight shim that imports only the mcp SDK.

Over http no biopb process has stdout as a protocol channel, so the class cannot
occur — and no new transport code is needed, since `/mcp` is already this
server's only transport and its loopback Host allowlist carries wildcard ports.

The bridge is worth building only for a harness that does not advertise
`mcpCapabilities.http`. None of the supported ones is such a harness.

## Why only opencode

Of the five MCP clients `biopb._agents` knows, four cannot fill this role:

| client | as an ACP agent |
|---|---|
| **opencode** | `opencode acp`, native. Advertises `mcpCapabilities {http, sse}` and connects what it is handed. |
| Claude Code | no native `acp` subcommand; needs `npx @zed-industries/claude-agent-acp`, so Node plus a first-run network fetch |
| Codex CLI | no `acp` subcommand (checked against 0.147.0); `codex mcp-server` is the inverse — Codex *as* an MCP server, not a client we can hand a server to |
| Cursor | `cursor-agent acp` exists, but is reported to ignore `session/new` `mcpServers` — which is the entire mechanism here |
| Claude Desktop | no headless agent at all |

Note the catalog here asks a different question from `biopb._agents.status()`:
that one reads a *client's config file* to see whether biopb is registered with
it, this one asks whether a CLI exists to run. A user can have Cursor registered
and no ACP harness, or the reverse.

## What biopb does and does not mediate

The harness is a real MCP client. It calls `execute_code` over `/mcp` like any
other, takes the kernel's one-agent claim under its own `clientInfo.name`, and
its cells appear in the job list as `origin="mcp"` — because that is what it is,
just one biopb launched. Nothing in `_chat_acp.py` touches the kernel.

**Permission questions are the agent's, and biopb has to ask for them.**
`session/request_permission` renders as a thread item carrying the agent's own
options, verbatim; `chat.acp_permission = "allow"` answers them automatically.

The catch is that the *harness* decides what to ask about, and opencode's
defaults are permissive: `read`, `edit`, `bash`, `webfetch` and `websearch` all
default to `allow`, so only `external_directory` and `doom_loop` prompt. Left
alone, `acp_permission = "ask"` would decide how to answer questions that are
never raised — measured, not assumed: a file write landed with no prompt at all.

So biopb pins the policy when it launches the harness, as an inline config in
the environment (`_ACP_AGENTS[...]["config_env"]`):

```
OPENCODE_CONFIG_CONTENT={"permission":{"edit":"ask","bash":"ask",
                                       "webfetch":"ask","websearch":"ask"}}
```

**In the environment, not a config file in the working directory** — a security
property, not a convenience. opencode would read a project config from its cwd,
but the agent can *write* anywhere under that cwd: a permission file living
there is one approved edit away from the agent turning its own prompts off. The
environment is read once at process start and fixed for the run.

opencode merges config sources rather than replacing them, and this one sits
above the global config, above `OPENCODE_CONFIG`, above the project config and
above `.opencode/` — only a machine-managed config outranks it. So the user's own
settings survive, and nothing the agent writes displaces ours. Verified with a
project `opencode.json` saying `{"permission": {"edit": "allow"}}`: the prompt
still fired.

### The harness's own biopb registration is switched off

The same inline config carries `{"mcp": {"biopb": {"enabled": false}}}`, and this
one is unconditional.

opencode **merges MCP servers from its config into an ACP session**, alongside
the ones handed to it in `session/new`. The installer registers biopb in that
config under exactly the key `biopb` (`biopb._agents`), over stdio — which the
shim turns into a second session child and a second napari window. So without
this the agent gets biopb twice: ours on the viewer the user is watching, and
theirs on a viewer nobody is.

It looks like it works without the suppression, because the two collide on the
name `biopb` and ours wins. That is an accident of naming, not a guarantee, and
it stops holding the moment anyone registers biopb under a different key —
measured: with our entry renamed to `biopb-http`, the agent listed eighteen
biopb tools and connected both servers.

Servers from `session/new` are a separate namespace, so disabling the config
entry does not touch ours — verified the same way. And only biopb's own entry is
suppressed: any other MCP server the user configured is theirs and stays. This is
about not being present twice, not about taking their tools away.

### What is pinned, and what is not

Only what must not move once the agent is running. **Permissions** are pinned
because they are a promise to the person watching, and a policy the agent could
change mid-session is not a policy.

The **model** is deliberately not pinned. It is a choice rather than a
guarantee, and one a user should be able to change without losing the
conversation — so it goes through ACP's `session/set_config_option` at runtime.
An eventual `/model` command belongs on that same seam: an env var would mean
respawning the agent to change it, which is exactly the wrong shape.

**Enumerated, not `{"*": "ask"}`** — and that difference is the design. The
wildcard also prompts for biopb's own MCP tools, so every cell the agent ran
would stop for a click. Verified both ways: under the enumeration
`biopb_server_status` ran unprompted while `apply_patch` raised a question.

What this does *not* do is gate `execute_code`. biopb's tools reach the kernel
without a prompt by design; the mitigation there is the one the observe page
already provides — every cell the agent runs is in the job list beside the
thread, with its code.

We decline ACP's `fs` and `terminal` client capabilities at `initialize`: a chat
pane is not an editor. That does not take the harness's own file access away and
is not meant to.

## Switching engines

`POST /chat/engine` refuses while a turn is running, and refuses when the kernel
is already claimed — one agent runs code in a session, the claim is released
only by a kernel restart, and a switch made anyway produces a pane that answers
questions and then refuses every cell. The refusal names the holder and points
at the restart control.

It is not persisted. The config file says what a session *starts* as; a click in
one window should not re-aim every future viewer.

The switch is session state, so the window that did not click has to find out:
the pane reads `GET /api/chat/engine` ahead of every history read and adopts
what it says, dropping the thread and cursor it was holding. Not a field on the
status probe, which happens once per page — and not inferred from which key the
history page carried, because the cursor spelling, the adapter and the slash
commands are all keyed to the engine, so it has to be known before the read
rather than after it.

## Shapes worth knowing

- **The thread is items with a revision.** ACP updates items in place, so the
  history read takes a `?since=<rev>` watermark rather than an `?after=<id>`
  cursor: a tool call moving to `completed` appends nothing, and an id cursor
  cannot express it. The ACP engine returns `items`, the built-in returns
  `messages`, and the pane picks its adapter by which key it got.
- **Message chunks coalesce by `messageId`.** opencode streams fragments as
  short as three characters; one reply is one item.
- **`tool_call_update` carries only what changed** — `title: null` at
  completion — so updates merge and never write a null over a field the view
  has.
- **Slash commands are two namespaces, and only one is offered.** The agent
  advertises its own by notification (`available_commands_update`); they can
  change mid-session, so they ride the polled history read. ACP has no method
  for invoking one — the command goes to the agent as prompt text and it parses
  its own prefix — and biopb accepts an advertised name that way, arguments and
  all. It does **not** list them: offering one is a promise the pane cannot
  keep, because the harness's working directory is empty and throwaway while a
  coding agent's commands are about a project. opencode's three are the
  illustration — `/review` has no repo, `/init` writes an AGENTS.md that dies
  with the temp dir, `/customize-opencode` edits config that biopb pins or that
  the temp dir takes with it. Revisit if the harness is ever given a real
  persistent workspace. Locally, `/new`, `/context` and `/model` survive under
  this engine; `/compact` does not, since it folds the built-in loop's
  projection of a thread the harness never reads — and ACP has no compaction
  method to forward it to. `/context` is *better* here: the agent reports its
  own `used`/`size` rather than the pane estimating from characters.
- **`chat.acp_model` overrides; it is not required.** When set, it is applied
  with `session/set_config_option` after `session/new`. When unset, the harness
  decides — and it decides from *its own config file*, which biopb does not
  shadow: `OPENCODE_CONFIG_CONTENT` merges rather than replaces, outranking the
  file only for the keys it declares (permissions, the MCP suppression), and
  `model` is not one of them. Measured, spawning `opencode acp` three ways: no
  env pin and no file model → `opencode/big-pickle`; biopb's pin over a file
  saying `openai/gpt-5.4` → `openai/gpt-5.4`; a pin that does carry `model` →
  that model.

  What a new session cannot inherit is the choice made in opencode's **TUI**,
  which lives in its session store rather than its config. So a user whose CLI
  works fine can still land on the built-in default here — observed:
  `opencode/big-pickle` failing with "Endpoint is unavailable" while the CLI
  worked. Setting a model in the harness's own config fixes that for both;
  `chat.acp_model` and `/model` are for pointing *this* session somewhere else.
- **The model moves at runtime, which is why it is not pinned.** `/model` reads
  `GET /api/chat/models` and writes `POST /chat/model`; under ACP that is
  another `session/set_config_option` on the *live* session, so changing model
  does not cost the conversation. Under the built-in loop it is `chat.model`,
  which `_model.make_model` reads per provider call — so the switch lands on the
  next call, with no restart. Refused mid-turn under both: a turn is several
  provider calls, and switching between them answers half a round in one voice
  and half in another. Not persisted, for the reason the engine is not.

  The list is never ours to keep. The harness states its own in
  `config_options`, groups flattened; the built-in loop reads the provider's
  `GET {base_url}/models`, which is optional in the OpenAI-compatible shape, so
  a provider that does not answer it is reported as publishing no list rather
  than as having one model.

  **The two are not symmetric, and the write path is shaped by it.** ACP has no
  session-less way to ask what exists: `config_options` rides `session/new`,
  `session/load`, `session/fork` and the set call itself, and nothing else. So
  `POST /chat/model` *starts the agent* under this engine before it validates.
  Without that, a name typed before the first turn is a name nobody checked —
  it is checked at spawn, found wanting, and silently replaced by the harness's
  default, leaving the pane naming a model it is not using. (Observed:
  `gpt-5.6-luna` accepted, where opencode offers `openai/gpt-5.6-luna`.)
  Starting the agent does not cost the engine switch: the one-agent claim is
  taken when a client *runs code*, not when one connects.

  The spawn-time fallback that remains — a bad `chat.acp_model` in the config
  file, which no keystroke can intercept — now lands in the thread as well as
  the log. The only other sign was the header quietly naming a model the user
  did not choose.
- **Threads move the pipes.** The harness is a plain `Popen` behind an
  `acp.Transport`, not `asyncio.create_subprocess_exec`: on Windows this server
  runs on the Selector loop, which implements neither subprocesses nor pipes.
