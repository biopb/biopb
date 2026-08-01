# The interaction layer

[`docs/skill-testing.md`](../../../../../../docs/skill-testing.md) §6. Put a
model in front of the **shipped** skill body, against a **real** session, and
score what comes out.

```sh
# needs a GL-capable display -- see "what this needs" below
xvfb-run -a -s '-screen 0 1024x768x24' \
  uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills/interaction -m interaction
```

Deselected by default (`-m interaction`), like `outcome` and `satisfiability`,
and never in CI (§10).

The hermetic half — the loop, the trace, the personas — is **not** marked and
runs with the ordinary suite, like `outcomes/test_outcome_protocol.py`. A break
in the machinery should surface as a normal red test rather than be found by
someone mid-diagnosis with a paid run.

## Running one with real models

Two keys, two families, and neither is optional (§6a):

```sh
export OPENAI_API_KEY=...        # or GEMINI_API_KEY / DEEPSEEK_API_KEY
export ANTHROPIC_API_KEY=...     # the respondent, and only ever the respondent
export BIOPB_SKILL_AGENT_MODEL=gpt-5          # optional, has a default
export OPENAI_BASE_URL=...                    # for a non-OpenAI compatible endpoint

xvfb-run -a -s '-screen 0 1024x768x24' \
  uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest .../interaction -m interaction
```

The provider SDKs are imported lazily and are **not** dependencies of this
package — one `--with` line, the same pattern the outcome layer uses for
`pystackreg`. Keys are read from the environment at call time and are never
written to a trace, an artifact or a log.

## What is here

| File | Holds |
|---|---|
| `_session.py` | Bring-up: a real shim-spawned session, a synchronous façade over the async MCP client, and the environment facts that are forced rather than inherited |
| `_bridge.py` | MCP tool schemas → the function-calling shape a chat model expects |
| `_agent.py` | `ChatAgent`; `ScriptedAgent`, `ReplayAgent`, and the live `OpenAICompatAgent` |
| `_respondent.py` | `Persona`, `Respondent`; `ScriptedRespondent`, `SilentRespondent`, and the live `ClaudeRespondent` |
| `_personas.py` | The respondent fixtures — who the agent talks to, and what only they know |
| `_conversation.py` | The two-model loop, the caps, the `Trace` |
| `test_session_smoke.py` | That the stack works, with **no model in it** |
| `test_conversation.py` | That the loop works, with no model *and* no session |
| `test_personas.py` | That a persona gives nothing away |

## What has been run, and what has not

Worth stating plainly, because "armed but never run" is how the contract layer
rotted for a release:

- **Run and green**: everything driven by `ScriptedAgent` / `ScriptedRespondent`
  (the loop, the trace, replay, the caps, scraping, the personas) and the nine
  session smoke tests. No key, no cost.
- **Written but never executed**: `OpenAICompatAgent` and `ClaudeRespondent` —
  the two provider adapters. They are a few dozen lines of plumbing each and
  the first real run will shake them out. Until that run has happened, do not
  read this README as saying they work.

The scoring pass that puts a real agent in front of `drift-correction` is not
here yet either.

## Why this tier is the one with teeth

Every other layer reads a skill file, or runs a procedure transcribed from one.
§5 was pulled out of the merge gate for exactly that reason: its subjects are a
hand transcription, so deleting `drift-correction.md` would leave it green
(§5c).

Here the body arrives through the real `biopb_mcp.mcp._skills` — `find_skills`
and `skill://<id>`, the same calls the runtime makes — and the run happens
against a real session: real kernel, real napari, real dask, the nine real
tools with their real schemas and the server's own `instructions`.

**Nothing is stood in for, and that was a deliberate choice.** A hand-written
tool surface would have been cheaper and would have put `execute_code`'s return
shape, `server_status`'s report and the `guide://` bodies back into a
transcription — the same disease one level up.

## What this needs

**A GL-capable display.** Not just Qt: `QT_QPA_PLATFORM=offscreen` gets you a
napari `Viewer`, and then `add_image` dies inside vispy's extension probe,
because the offscreen platform provides no GL context. Use a desktop session's
own display, or `xvfb-run`. Without one these tests **skip with instructions**
rather than run somewhere subtly different.

Nothing else: no API key for the smoke tests, and no network beyond loopback.

## Three things the harness forces

Each of these silently changes what a run tests, so none of them is inherited
from whatever machine is running.

**A real viewer.** `transport.display_mode` defaults to `auto`, which degrades
to a viewer-less kernel when no display is found. That is a legitimate
production mode and so nothing fails loudly — but a run that took it would be
scoring a session in which step 2's *"show the user the first and last frames"*
cannot happen at all. Bring-up probes for it and refuses.

**No tensor plane.** A developer box often has a data plane up, and then
`client` is live and the agent can wander into whatever catalog that machine
holds — so a finding might not reproduce anywhere else. The child is pointed at
an unreachable URL, `client` lands as `None`, and the fixture reaches the agent
as a napari layer and nothing else. Every skill's Parameters table already
accepts *"a layer on `viewer`"* as a source, and a session with no tensor plane
is a real configuration a user can be in, so step 1 still has something true to
resolve.

**A config tree of our own.** `XDG_CONFIG_HOME` points at a temp dir, so the run
reads neither the developer's `mcp-config.json` nor their personal
`~/.config/biopb/skills/*.md`. The catalog under test is the shipped one.

## Arrays cross by file, not by literal

`put_array` / `get_array` write `.npy` into a shared temp dir and have the
kernel load or save it. The session child is on this machine, a fixture movie is
several megabytes, and tool output is truncated for the agent's benefit — so
base64 inside a tool call would be both slow and lossy.

`get_array` returns `None` when the name does not evaluate to an array. A run
that left nothing behind is an ordinary outcome — the agent gave up, or never
got there — and it has to arrive as *"nothing to score"*, which `Outcome.passed`
already refuses to treat as a pass.

## Setup is not a turn

`session.call(...)` is the agent acting and is recorded with its turn number.
`session.setup(...)` is the harness talking to the kernel around the agent and
is recorded at turn `-1`. The structural assertions (§6) ask whether a blocking
question preceded the expensive call, so injecting the fixture must never read
as something the agent did.

## Why the smoke tests exist

§6 is the least isolable tier in the suite: a red run's cause space is the skill
body, the model, the tool schemas, the kernel, Qt, dask and the fixture. That is
the real cost of testing against the full environment rather than a stand-in,
and it was taken on with open eyes.

`test_session_smoke.py` is the mitigation. When the stack is what broke, *it*
fails — separately, deterministically, and before a single token is spent.
