# Launching a client session from the dashboard

Status: **proposed** — nothing implemented.

The dashboard already knows which MCP clients are installed and whether biopb is
registered with each (`/api/agents`, backed by `biopb._agents`). It stops there:
the user still opens a terminal to actually talk to an agent. Three of the five
clients can be driven from a browser, so the dashboard can start that session
itself — one button, next to the registration state it already shows.

Most of what follows was settled by running the clients rather than reading their
docs, so the measurements are dated; see [Measurements](#measurements).

## Which clients, and three shapes

| client | mechanism | where the UI is | what the control owns |
|---|---|---|---|
| **opencode** | `opencode web --hostname 127.0.0.1 --port 0` | a loopback HTTP server; opencode opens the browser itself | a foreground child |
| **Claude Code** | `claude remote-control --spawn session` | claude.ai/code or the Claude mobile app, over an outbound relay | a foreground child |
| **Codex CLI** | `codex app-server daemon enable-remote-control` | ChatGPT, over an outbound relay, after a pairing code | nothing — a machine-global daemon |

Claude Desktop and Cursor have no web interface; their rows never show the
button.

Two notes on the Claude Code invocation. `remote-control` is not listed in
`claude --help`'s command table even though it exists; `--remote-control` / `--rc`
are the *interactive* variants and would need a PTY from the control, which the
subcommand does not. And `--spawn session` is not optional here — see
[below](#claude-code-needs---spawn-session).

### `can_launch` is one test, declared by the backend

Registration state does not answer this on its own: `_agents.status()` is a
config-file read, so Claude Desktop can be `registered` with no CLI present at
all. The launcher check must not be folded into `is_installed()` either —
`JsonConfigClient` keeps that one deliberately loose (the app's config
*directory*, not a binary) because "register is still an escape hatch that works
anyway", and requiring a binary would flip Claude Desktop and Cursor to
`not_installed` and take their Register button with them. Neither has anything on
PATH.

So the launcher becomes a second predicate on the backend, evaluated in the same
subprocess-free pass and reported beside `state`:

```python
class ClientBackend:
    def launch_argv(self) -> Optional[list[str]]:
        """The command that opens a web/remote session, or None if this client
        has no such surface."""
        return None

    def can_launch(self) -> bool:
        argv = self.launch_argv()
        return bool(argv) and shutil.which(argv[0]) is not None
```

Claude Desktop and Cursor inherit `None` and are excluded by construction, with
no client-id list in the SPA. Codex overrides `can_launch` to also require
`$CODEX_HOME/packages/standalone/current/codex` — a `Path.exists()`, no
subprocess — because `which("codex")` cannot distinguish an npm install from the
standalone one the daemon demands. `status()` carries `can_launch` next to
`state` / `drifted` / `config_path`, and the launch handler re-reads that same
predicate rather than testing again, so the button and the gate cannot drift
apart. A `registered` client with `can_launch: False` renders as Register /
Unregister without a session button — which is exactly right for Claude Desktop.

What stays out of reach of any cheap check: subscription, org policy, and the
workspace-trust dialog. Those surface only at launch; see
[Error surface](#error-surface).

## Lifecycle: one child, one page, one stop

The model is:

- **one client child per client**, kept alive;
- **`[start]` opens a browser page** against the child that already exists, never
  a second child;
- **`[x]` lives on the session pane**, because one client child yields exactly
  one biopb session.

That last equivalence is the load-bearing one, and it holds because MCP
connections are held at *server* scope in two of the three clients:

| client | MCP connection scope | one child ⇒ |
|---|---|---|
| opencode | server-scoped: `GET /mcp` has no session in the path, one MCP child whose parent is the server, unchanged across two sessions | one biopb session |
| Codex | process-wide on the app-server: `mcpServerStatus/list` with no `threadId` returns the tools | one biopb session |
| Claude Code | **per session**: two background sessions produced two MCP children with different parents | one *per session* |

### Claude Code needs `--spawn session`

`claude remote-control` defaults to a server hosting many concurrent sessions
(`--capacity 32`, `--spawn same-dir`), each its own process with its own MCP
children — so one daemon would give N biopb session children and N napari
windows. `--spawn session` is the classic single-session mode that exits when its
session ends. It restores the 1:1 mapping and gives `[x]` its natural meaning.

### Codex is not ours to keep alive

The Codex app-server daemon is machine-global and may already be running for
reasons that have nothing to do with biopb. "Keep one alive" is therefore a no-op
there, and `[x]` must mean `disable-remote-control`, not `stop` — the one place
the three-button model needs a different verb.

`codex remote-control start` and `codex app-server daemon start` both refuse
unless Codex was installed by its own installer:

```
Error: managed standalone Codex install not found at
  ~/.codex/packages/standalone/current/codex
```

An npm-installed Codex cannot run the daemon at all. `which("codex")` cannot tell
the two apart, but the path in that message can: its presence is a plain
`Path.exists()`, which is why `can_launch` [above](#can_launch-is-one-test-declared-by-the-backend)
can answer this before the user clicks rather than leaving it to a failed launch.
Bare `codex app-server` (stdio) works either way; only the daemon needs the
standalone install.

## Ownership: the client is an owned child

The repo has two lifecycle patterns and this is a choice between them, not a
platform detail:

- **Pattern O** (`_lifecycle/owned_child.py`) — the child dies with its parent.
  POSIX: no `start_new_session`, so the parent's group teardown reaches it, plus
  a parent-death pipe (`_lifecycle/deathwatch.py`) for the uncatchable deaths.
  Windows: a kill-on-close Job Object.
- **Detached** (`_lifecycle/daemon.detach_kwargs`) — the child survives.
  `_launch_viewer` uses it deliberately: "a viewer is the user's window, and a
  control restart must not close it."

**The client is Pattern O.** Every other link already works that way —
`control → client → biopb-mcp shim → session child → kernel`, each dying with its
parent — so making the client an owned child closes the chain, and the teardown
below stops being something we sequence by hand. Detaching it would break the
chain at exactly the link holding an unauthenticated listener.

The cost is real: **restarting the control drops live chat sessions**, including
a remote-control session someone is attached to from a phone. Put that in the UI
rather than leaving it to be discovered. (The dashboard is served by the control,
so it is down in that window regardless.)

## The session record needs the client's pid

`_sessions.register()` records `session_id, host, port, pid, create_time,
mcp_url, started_at`, where `pid` is the session child — the port-owning process.
Nothing names the client that spawned it, so a session pane cannot say who owns
it or refuse to act on a client the control did not start.

Add it in the shim, which needs no plumbing from the control because **its own
parent is the client**:

```python
_sessions.register(session_id, port=port, pid=child.pid, mcp_url=url,
                   client_pid=os.getppid(),
                   client_create_time=process_create_time(os.getppid()))
```

`register()` stores `**extra` verbatim as documented forward room, so this is
additive — but it has to land before the buttons ship, because a record's shape
does not retrofit. The `--view` path simply omits the fields; it has no client.

Three constraints on that value:

- **Capture at publish time, never re-read.** If the client dies the shim is
  reparented and a later `getppid()` names init.
- **It is the client only while the shim is a direct child.** True for both
  clients measured; a shell wrapper would break it, so record the parent's name
  too and have `[x]` refuse on a mismatch.
- **It is not a kill target.** The control holds the client as an `OwnedChild`,
  and a live `Popen` handle cannot be fooled by pid reuse. `client_pid` is for
  display and provenance: it is how `[x]` recognises a client the control did
  *not* launch — a user's own `opencode web` in a terminal, or the shared Codex
  daemon — and offers session-stop only.

## Teardown

1. `[x]` → is this a client the control launched (handle still held, matching
   `client_pid` / `client_create_time`)? If not, offer "stop this biopb session"
   and nothing more.
2. If it is, stop the **client**, through `OwnedChild.stop()`. The chain unwinds
   itself: `biopb-mcp` is the client's child, bridge close triggers
   `_reap_session`, which stops the session child and drops the registry record
   in one path.
3. Wait bounded for the record to disappear; escalate only against the client.

**Never stop the biopb session first.** Besides being the wrong order, opencode
will reconnect it (`POST /mcp/{name}/connect`) and spawn a fresh viewer behind
the user's back.

## Security

**The gate is not the same for the two shapes.** opencode's is a loopback
listener, so its button belongs behind the same `console_enabled` bit as the
viewer launch and the chat proxy. The relay clients are the opposite case: being
reachable from another device is the entire point, and gating them on
loopback-only forbids the case that makes them worth building. They need their
own opt-in — starting an agent with shell access on the owner's subscription is a
step up from viewing data, and it should be enabled deliberately rather than
riding a gate that means something else.

For Claude Code, pin `--permission-mode default`; never `bypassPermissions`.

**opencode's listener is unauthenticated by default, and we warn rather than
secure it.** `OPENCODE_SERVER_PASSWORD` (with `OPENCODE_SERVER_USERNAME`,
default `opencode`) turns on HTTP Basic across the whole listener. The control
passes its environment to the child the way `_launch_viewer` already does, so a
user who starts the control with that variable set gets a secured server and no
warning, with no code on our side. Left unset, opencode prints its own warning as
the first line of the launch log:

```
!  OPENCODE_SERVER_PASSWORD is not set; server is unsecured.
```

Surface *that* line rather than reimplementing the check — it stays truthful if
their default changes. A pre-launch `os.environ` check is still worth having so
the card can warn before the click.

**The ACP path should mint one instead.** `_chat_acp.py`'s listener is
incidental — nobody browses it — so there is no reason to leave it unsecured
once a password is available. That closes biopb#909 and makes
[chat-acp-engine.md](chat-acp-engine.md)'s claim that eliminating the exposure
"requires an opencode mode that provides authenticated transport" obsolete.
Independent of this feature.

Because the web launcher does not reuse the ACP launcher, it also does not
inherit its `OPENCODE_CONFIG_CONTENT` permission pinning: a web session runs
under the user's own opencode settings, where `bash` and `edit` default to
`allow`. That is the right answer for a harness the user chose to open, and the
deliberate difference between the two paths is recorded here so it does not read
as an oversight later.

**The relay clients expose no local listener.** Claude Code polls Anthropic's
backend over outbound HTTPS; Codex opens an outbound websocket to a chatgpt.com
relay and persists a device enrollment. Authentication is the user's own account
(Claude Code) or an enrolled device plus a short-lived pairing code (Codex) —
nothing for the control to mint, hold, or leak. The tradeoff is elsewhere:
Anthropic documents that a session transcript is stored server-side while Remote
Control is connected, execution and filesystem staying local. For a lab running
biopb locally for data-governance reasons that belongs next to the button, not in
a footnote. Codex's equivalent retention behaviour is unverified.

## Error surface

What `can_launch` cannot see is still invisible until launch: no subscription, an
org policy toggle, an unaccepted workspace-trust dialog, or a
`DO_NOT_TRACK`-family variable inherited from the control's environment. So the
launch contract is `_launch_viewer`'s, which already solves this shape:

- a per-launch log file with the same retention treatment;
- a bounded readiness wait, ceilinged under the client's own HTTP timeout;
- three states — `started` / `starting` / `failed` — where only `failed` carries
  `error`, a log tail and `log_path`;
- rendered in the inline `.note.launch` block the viewer card already uses, not
  an `alert()`.

**Show the tail on success too, not only on failure.** opencode prints its URL
there (`Web interface:      http://127.0.0.1:<port>/`), which is what the user
needs if the browser-open did not land — and it is where the unsecured-server
warning lives. Strip ANSI and the ASCII logo for display.

Readiness is weaker than the viewer's, which has an exact signal (a registry
record matching the child's pid). A client child has none, so "started" means the
process is alive and, for opencode, that the URL line appeared.

`claude agents --json` lists live Claude Code sessions (`pid`, `cwd`,
`sessionId`, `name`, `status`) without a TTY, and the Codex app-server emits
`remoteControl/status/changed`. Both are better reconciliation sources than
anything we would keep ourselves.

## Cross-platform

The record and the `getppid()` capture are portable. The kill is where platforms
diverge, and `process_create_time` (`_lifecycle/proc.py`) is the reason:

| | source | recycled-pid guard |
|---|---|---|
| Windows | `GetProcessTimes` creation FILETIME | yes — and most needed, since Windows reuses pids aggressively |
| Linux | `/proc/<pid>/stat` field 22 | yes |
| macOS | none | no — returns `None`, callers degrade to liveness-only |

Holding an `OwnedChild` rather than a recorded pid makes that gap irrelevant
while the control is up, since the handle is immune to reuse. It reappears only
after a control restart, when the handle is gone and the record is all that is
left. On macOS `[x]` should degrade to session-stop in that state rather than
aim a signal at a number.

Two Windows specifics: `OwnedChild.stop()` uses `TerminateJobObject` for a real
tree kill, which is *better* than the POSIX path — but `terminate()` there is
`TerminateProcess`, so the client gets no graceful shutdown and its MCP children
see EOF from closed handles instead of a clean exit. And `detach_kwargs()`'s own
documented limit applies to anything session-scoped: it does not survive logout
on Windows, or on systemd without `enable-linger`.

## Open questions

- **The button means "a new session", not "drive this viewer."** The registered
  entry is `biopb-mcp --transport stdio`, so the client spawns its own session
  child and its own napari window. Codex re-reads its config live, so
  register-then-start works in either order; Claude Code has no per-session MCP
  override on the remote-control path, so there is no way to hand it this
  session's `/mcp` the way the ACP engine does. Label the button honestly rather
  than implying attachment.
- Codex transcript retention while remote control is connected.
- Whether opencode and Codex's daemon path work on Windows at all; `_agents`
  already carries Windows config paths for all five clients, so registration is
  cross-platform, but launching is unverified there.

## Measurements

All 2026-09-05 on Linux, against opencode 1.18.28, Claude Code 2.1.261 and Codex
0.151.0 as installed on the dev box. Client configs were backed up and restored;
each result was checked against a byte-identical `diff` afterwards.

- **opencode MCP scope.** Registered a stub stdio MCP server, started
  `opencode serve --hostname 127.0.0.1 --port 0`, then created two sessions.
  `GET /mcp` returned `{"probe_a":{"status":"connected"}}` — server-scoped, no
  session in the path — and exactly one stub process existed throughout, its
  parent the opencode server. Killing the server took the stub with it.
- **Claude Code MCP scope.** Two `claude --bg` sessions with the stub as
  `--mcp-config --strict-mcp-config` produced two stub processes with different
  parents.
- **Codex config.** `codex mcp add` writes `[mcp_servers.X]` into the same
  `~/.codex/config.toml` that holds `[tui.*]` and `[projects.*]`; the app-server
  reports that `codexHome` on `initialize`, launches the servers (tools listed,
  not merely echoed), and picks up a server added while it is already running —
  before any `config/mcpServer/reload` call.
- **opencode Basic auth.** With `OPENCODE_SERVER_PASSWORD` set, `/` and `/api/*`
  return 401 unauthenticated and 200 with `-u opencode:<pw>`.
- **No TTY needed.** `codex app-server daemon` is documented for SSH-driven use
  and reached its control socket with stdin not a terminal; `claude
  remote-control --help` printed without one. Codex's TTY refusals are scoped to
  its interactive TUI and a delete confirmation.
