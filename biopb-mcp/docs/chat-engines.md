# Chat: two engines

`mcp/_chat.py` and `mcp/_chat_acp.py` are the two `chat.engine` options
(`builtin` default, or `acp`); the pane offers a switch when both are usable.
The built-in loop is for a user with no agent of their own; ACP hands the
pane to a harness they already run -- Claude Code, opencode, etc. Neither
replaces the other.

**Chat is off whenever the control runs `--remote`**, for either engine. It
shares the user console's gating: reachable only while the control is
loopback-bound, on a separate proxy root, never on a public bind.

## Built-in loop

**Calls biopb's own tools as Python functions**, from the same FastMCP tool
registry an external MCP client uses -- not over the wire -- so it needs no
MCP client SDK.

**Does not reuse `execute_code`'s `promote_after` window.** That wait saves
an external client a round trip; in-process a poll is a function call, so
the loop submits with no promote window and streams a job's partial stdout
as it accumulates instead of blocking on it.

**Runs in the session child process, not the napari kernel.** The kernel is
a grandchild the loop must survive restarting.

**No third-party agent framework is vendored** -- the loop calls tools
in-process and needs no provider SDK.

**A screenshot is refused rather than silently sent to a non-multimodal
model** -- capability is checked before an image reaches the provider, with
automatic recovery for a model that can't take it.

## ACP engine

`mcp/_chat_acp.py` hands the pane to a coding harness the user already
runs, over the [Agent Client Protocol](https://agentclientprotocol.com/), so
it drives the same viewer on the user's own subscription.

### How biopb reaches the agent

`session/new` carries an `mcpServers` array; biopb goes in it as **this
session's own `/mcp` over http**:

```json
{"type": "http", "name": "biopb", "url": "http://127.0.0.1:<port>/mcp",
 "headers": []}
```

**It is this session, not a new one.** The URL comes from the registry record
this session published for itself, so the harness attaches to the viewer
already in front of the user rather than spawning a second one via the
installer's stdio entry.

**`headers` is required, not optional** -- both the `agent-client-protocol`
models and opencode's schema reject the entry without it.

opencode's `acp` subcommand also starts its own headless HTTP server on an
OS-assigned loopback port (`--hostname 127.0.0.1 --port 0`); biopb neither
discovers nor uses it -- ACP over stdio is the supported interface -- and that
listener is unauthenticated, so it isn't a surface to rely on.

### Why only opencode

| client | as an ACP agent |
|---|---|
| **opencode** | `opencode acp`, native; advertises `mcpCapabilities {http, sse}` |
| Claude Code | no native `acp` subcommand; needs `npx @zed-industries/claude-agent-acp` |
| Codex CLI | no `acp` subcommand; `codex mcp-server` is the inverse (Codex as an MCP server) |
| Cursor | `cursor-agent acp` exists but ignores `session/new`'s `mcpServers` |
| Claude Desktop | no headless agent |

This asks a different question from `biopb._agents.status()`, which checks
whether a client's config registers biopb; this asks whether a CLI exists to
run as an ACP agent at all.

### What biopb does and does not mediate

The harness is an ordinary MCP client: it calls `execute_code` over `/mcp`,
takes the kernel's one-agent claim under its own `clientInfo.name`, and its
cells appear in the job list as `origin="mcp"`. Nothing in `_chat_acp.py`
touches the kernel directly.

**Permission questions are the agent's; biopb only has to ask them.**
`session/request_permission` renders as a thread item with the agent's own
options; `chat.acp_permission = "allow"` answers automatically. Left at
`"ask"`, opencode's own defaults (`read`/`edit`/`bash`/`webfetch`/`websearch`
all `allow` by default) mean most actions are never asked about at all -- so
biopb pins policy at launch, as an **environment variable** rather than a
project config file (a config file living in the agent's own working
directory is one approved edit away from the agent disabling its own
prompts):

```
OPENCODE_CONFIG_CONTENT={"permission":{"edit":"ask","bash":"ask",
                                       "webfetch":"ask","websearch":"ask"},
                         "mcp":{"biopb":{"enabled":false}}}
```

opencode merges config sources with this one outranking everything but a
machine-managed config, so the user's own settings and any other MCP server
they've configured survive untouched.

The `mcp.biopb.enabled: false` half is unconditional: without it, opencode
would also merge in the *installer's* stdio `biopb` entry from its own config
(a second session, a second napari window nobody is looking at) alongside the
one handed to it in `session/new` -- two different namespaces, so suppressing
the config entry never touches the one this call passed in.

Permissions are enumerated rather than `{"*": "ask"}`, because the wildcard
would also prompt for biopb's own MCP tools on every cell. `execute_code`
itself is never gated by this -- biopb's tools reach the kernel with no
prompt by design, and the mitigation is the job list beside the thread, not a
permission dialog. biopb also declines ACP's `fs`/`terminal` client
capabilities at `initialize`: a chat pane is not an editor, and this does not
restrict the harness's own file access.

**The model is not pinned** -- it moves via ACP's `session/set_config_option`
at runtime instead, so a user can change it mid-conversation without losing
the thread; `chat.acp_model` sets a starting point, and unset, the harness's
own config decides.

### Switching engines

`POST /chat/engine` refuses while a turn is running, refuses if the target
engine isn't ready, and refuses if the kernel is already claimed by a writer
(naming the holder and pointing at kernel restart). The choice is **session
state, not persisted** -- the config file only picks the engine a session
*starts* as. `GET /api/chat/engine` is read ahead of every history load so a
second window adopts whatever the first one switched to, dropping its own
thread and cursor (the item/message shape, the adapter, and slash commands
are all keyed to the engine).

### Shapes worth knowing

- **The thread is items with a revision**, not a message list with an append
  cursor: ACP updates items in place (e.g. a tool call moving to
  `completed`), so history reads take `?since=<rev>`. The ACP engine's
  history responses key on `items`; the builtin's key on `messages`.
- **Message chunks coalesce by `messageId`** -- opencode streams fragments as
  short as three characters.
- **`tool_call_update` carries only what changed**, so a client must merge
  rather than overwrite.
- **Slash commands are two namespaces.** The agent advertises its own by
  notification and they ride the polled history read (ACP has no invoke
  method -- commands are sent as prompt text and the agent parses its own
  prefix). biopb does **not** list them itself: the harness's working
  directory is empty and throwaway, so a command about "the project"
  (`/review`, `/init`) has no project to act on here; `/model`, `/new` and
  `/context` are the ones that hold up in that setting. `/compact` has no ACP
  equivalent and is not offered.
- **The model resolves harness-config first, then biopb's pin.**
  `chat.acp_model` is applied via `session/set_config_option` after
  `session/new`; left unset, the harness uses its own config file, which
  biopb does not shadow (the env config above merges only the keys it
  declares). A model chosen in the harness's own interactive TUI, rather than
  its config file, is not visible to a fresh ACP session and so is not
  inherited.
- **`POST /chat/model` starts the agent before it validates**, because ACP
  has no session-less way to list or check models -- `config_options` only
  rides `session/new`/`session/load`/`session/fork`/the set call itself. A
  bad model name is caught at spawn and silently replaced by the harness's
  own default; that fallback now surfaces in the thread as well as the log.
- **The harness is a plain `Popen`, not `asyncio.create_subprocess_exec`** --
  on Windows this server runs the Selector loop, which supports neither.
