# Launching a client session from the dashboard

Status: **proposed** — nothing implemented.

The dashboard already knows which MCP clients are installed and whether biopb is
registered with each (`/api/agents`, backed by `biopb._control._agents`). It stops there:
the user still opens a terminal to actually talk to an agent. Three of the five
clients can be driven from a browser, so the dashboard can start that session
itself — one button, next to the registration state it already shows.

Most of what follows was settled by running the clients rather than reading their
docs, so the measurements are dated; see [Measurements](#measurements).

The client does not own a biopb session. Each client registers `biopb-shim`, which
starts unbound and, when the agent calls `attach`, takes the **lease** of a session
that is a detached process (the control launches one with `POST /api/sessions/new`;
it runs until stopped from the dashboard). Launching a client therefore starts a
*holder* of a session, not its parent.

## Which clients, and three shapes

| client | mechanism | where the UI is | what the control owns |
|---|---|---|---|
| **opencode 1** | `opencode web --hostname 127.0.0.1 --port 0` | a loopback HTTP server; opencode opens the browser itself | a foreground child |
| **opencode 2** | `opencode pair` (no `web` command) | a loopback service; a one-time link signs the browser in | nothing — a machine-global background service |
| **Claude Code** | `claude remote-control --spawn session` | claude.ai/code or the Claude mobile app, over an outbound relay | a foreground child |
| **Codex CLI** | `codex app-server daemon enable-remote-control` | ChatGPT, over an outbound relay, after a pairing code | nothing — a machine-global daemon |

Claude Desktop and Cursor have no web interface; their rows never show the
button.

### opencode 1 and 2

The version decides the invocation (`opencode --version`, the probe
`_agents` already uses to pick the config shape).

**opencode 1** runs `opencode web` as a foreground child that serves one loopback
listener and opens the browser itself.

**opencode 2** has no `web` command. Its CLI talks to a background service
(`opencode service start|status|stop`) that starts on demand, on a loopback port it
chooses. `opencode pair` prints a **one-time link** (single use, expires in 5
minutes) that signs a browser in to it:

```
Open a link to connect. Links work once and expire in 5 minutes.
http://127.0.0.1:<port>/auth/connect/<token>
```

- **Other devices.** `opencode service set hostname 0.0.0.0` makes the service
  listen beyond loopback. `opencode pair --url <external-url>` writes links against
  an external address. `opencode pair --remote` pairs through opencode's tunnel
  address, "enabling remote access if needed (exposes the server to the internet)";
  it asks for confirmation, and `--yes` is required without a terminal.
- **Standalone server.** `opencode serve --hostname H --port P` runs the same API
  and web server in the foreground, which is the shape that fits a child the
  control owns.
- **Ownership.** The background service is machine-global and may be running for
  other reasons, like the Codex daemon: a launch is `pair` plus showing the link,
  and `[x]` must not mean `service stop`.

Two notes on the Claude Code invocation. `remote-control` is not listed in
`claude --help`'s command table even though it exists; `--remote-control` / `--rc`
are the *interactive* variants and would need a PTY from the control, which the
subcommand does not. `--spawn session` is the choice of mode — see
[below](#claude-code-spawn-mode).

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
- **`[x]` lives on the session pane**, because one client child holds exactly
  one biopb session.

That last equivalence holds when a client child runs one `biopb-shim`. A session
has one lease holder, and a shim holds a session only after `attach`, so the
sessions a client holds are bounded by the shims it runs, which follow the
client's MCP connection scope. Measured as scope of a
stdio MCP server, which the shim is:

| client | MCP connection scope | one child ⇒ |
|---|---|---|
| opencode 1 | server-scoped: `GET /mcp` has no session in the path, one MCP child whose parent is the server, unchanged across two sessions | one shim, one biopb session |
| Codex | process-wide on the app-server: `mcpServerStatus/list` with no `threadId` returns the tools | one shim, one biopb session |
| Claude Code | **per session**: two background sessions produced two MCP children with different parents | one shim *per session* |

opencode 2's scope is unmeasured; its service hosts the servers, so the opencode 1
result is the expectation, not a fact.

### Claude Code: spawn mode

`claude remote-control` defaults to a server hosting many concurrent sessions
(`--capacity 32`, `--spawn same-dir`), each its own process with its own MCP
children, so one daemon runs N shims. That is cheap: a shim starts unbound and
creates no biopb session or napari window until its agent calls `attach`. Only
conversations that ask for a session get one.

`--spawn session` is the classic single-session mode that exits when its session
ends. It remains the better default, though not a requirement: one client child
is one conversation and holds at most one biopb session, so `[x]` has its natural
meaning and the lease's client pid names a single holder. The default mode works
too, but `[x]` then ends every conversation the daemon hosts.

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

- **Pattern O** (`biopb/_lifecycle/_owned_child.py`) — the child dies with its parent.
  POSIX: no `start_new_session`, so the parent's group teardown reaches it, plus
  a parent-death pipe (`_lifecycle/deathwatch.py`) for the uncatchable deaths.
  Windows: a kill-on-close Job Object.
- **Detached** (`_lifecycle/daemon.detach_kwargs`) — the child survives.
  `_launch_session` uses it deliberately: a session is the user's, and "a control
  restart must not end the session."

**The client is Pattern O; the session it holds is detached.** The client's chain
is `control → client → biopb-shim`: each link dies with its parent, and the shim
releases its lease on stdin EOF or SIGTERM (and a `kill -9`'d shim's lease lapses
after 30 s). So making the client an owned child closes the chain, and nothing
outlives the control but the sessions, which are meant to. Detaching the client
would break the chain at exactly the link holding a listener that can run code.

The cost is real: **restarting the control drops live client sessions**, including
a remote-control session someone is attached to from a phone. The biopb session
itself survives, free, and the next client can attach to it. Put that in the UI
rather than leaving it to be discovered. (The dashboard is served by the control,
so it is down in that window regardless.)

## The lease needs the client's pid

A session registers itself (`_sessions.register()`: `session_id, host, port, pid,
create_time, mcp_url, started_at`, with the port-owning session process as `pid`).
It records nothing about who is using it, and the shim does not register anything,
so a session pane cannot say which client holds it or refuse to act on a client
the control did not start.

The holder is the lease, so that is where the client goes. The shim already finds
its client for the Windows death watchdog (`_find_client_process`, which climbs
past launcher stubs that `os.getppid()` would name), and sends only `{token,
force}` to `/api/lease/acquire`. Add the client's `pid`, its `create_time`
(`process_create_time`) and its process name to that call; the session keeps them
beside the holder in `_lease.snapshot()`, and `/api/sessions` returns them in the
row's busy status.

Constraints on that value:

- **Capture at attach, never re-read.** It is `None` when psutil is missing or an
  ancestor cannot be read; such a holder is "unknown", never "not ours".
- **It is not a kill target.** The control holds the client as an `OwnedChild`,
  and a live `Popen` handle cannot be fooled by pid reuse. The pid is for display
  and provenance: it is how `[x]` recognises a client the control did *not*
  launch — a user's own `opencode web` in a terminal, or the shared Codex
  daemon — and offers session-stop only.
- **A match needs the name too.** Compare pid, `create_time` and process name with
  the handle the control holds, and refuse on a mismatch.

## Teardown

Two stops, kept separate because the session outlives its client.

1. `[x]` on a client → is this one the control launched (handle still held,
   matching the lease holder's pid and `create_time`)? If not, offer "stop this
   biopb session" and nothing more.
2. If it is, stop the **client**, through `OwnedChild.stop()`. Its shim exits on
   stdin EOF or SIGTERM and releases the lease; the session is untouched and free
   for the next agent. Wait bounded for the lease to show free; escalate only
   against the client.
3. Stopping the **session** is its own action (`/api/sessions`). A shim bound to a
   session that stops is unbound, and its next call says why.

**Stop the client first when both go.** A shim started with `--session auto`
(Codex) that is respawned after its session is gone attaches to the newest free
session or launches a new one, so stopping the session first can bring up a fresh
viewer behind the user's back. An opencode shim starts unbound and costs nothing
in that case.

## Security

**The gate is not the same for the shapes.** opencode's is a loopback listener,
so its button belongs behind the same condition as the chat proxy (the control's
`--auth` mode). The relay clients are the opposite case: being reachable from
another device is the entire point, and gating them on loopback-only forbids the
case that makes them worth building. They need their own opt-in — starting an
agent with shell access on the owner's subscription is a step up from viewing
data, and it should be enabled deliberately rather than riding a gate that means
something else.

opencode 2 sits between: it listens on loopback by default, and `pair --remote`
or `service set hostname 0.0.0.0` reach beyond it. The control starts the local
form only (`pair` with no flag) and never passes `--remote` or `--yes`; exposing
the service to the internet is the user's own step, the way an operator opts in
to a relay client.

For Claude Code, pin `--permission-mode default`; never `bypassPermissions`.

**opencode 1's listener is unauthenticated by default, and we warn rather than
secure it.** `OPENCODE_SERVER_PASSWORD` (with `OPENCODE_SERVER_USERNAME`,
default `opencode`) turns on HTTP Basic across the whole listener. The control
passes its environment to the child the way `_launch_session` already does, so a
user who starts the control with that variable set gets a secured server and no
warning, with no code on our side. Left unset, opencode prints its own warning as
the first line of the launch log:

```
!  OPENCODE_SERVER_PASSWORD is not set; server is unsecured.
```

Surface *that* line rather than reimplementing the check — it stays truthful if
their default changes. A pre-launch `os.environ` check is still worth having so
the card can warn before the click.

**opencode 2 is authenticated by default.** The service answers `/api/*` with 401
until a browser has used a pairing link, and `opencode serve` with no
`OPENCODE_SERVER_PASSWORD` generates a random password and prints it
(`server password <...>`) instead of running open; with the variable set, Basic
auth as `opencode:<password>` works as before. The warning above does not apply;
the link or password in the launch log is what the user needs, and it is a
credential, so show it to the one who clicked and do not keep it in a log the
dashboard serves to others.

**A web session runs under the user's own opencode settings**, where `bash`
and `edit` default to `allow`. Nothing is pinned on their behalf: this is a
harness the user chose to open, on a machine already theirs, and rewriting its
permissions from under it would be the surprising answer rather than the safe
one. Recorded here so it does not later read as an oversight.

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
launch contract is `_launch_session`'s, which already solves this shape:

- a per-launch log file with the same retention treatment;
- a bounded readiness wait, ceilinged under the client's own HTTP timeout;
- three states — `started` / `starting` / `failed` — where only `failed` carries
  `error`, a log tail and `log_path`;
- rendered in the inline `.note.launch` block the dashboard already uses for
  session launches, not an `alert()`.

**Show the tail on success too, not only on failure.** opencode 1 prints its URL
there (`Web interface:      http://127.0.0.1:<port>/`), which is what the user
needs if the browser-open did not land — and it is where the unsecured-server
warning lives. opencode 2's `pair` prints the one-time link and `serve` its
generated password; those are credentials (see [Security](#security)). Strip ANSI
and the ASCII logo for display.

Readiness is weaker than a session's, which has an exact signal (a registry
record carrying the per-launch token the control handed the child). A client child
has none, so "started" means the process is alive and, for opencode, that the URL
or link line appeared; for opencode 2, that `pair` exited 0.

`claude agents --json` lists live Claude Code sessions (`pid`, `cwd`,
`sessionId`, `name`, `status`) without a TTY, and the Codex app-server emits
`remoteControl/status/changed`. Both are better reconciliation sources than
anything we would keep ourselves.

## Cross-platform

The lease record and the client lookup are portable. The kill is where platforms
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

- **Which session does the launched client get?** The registered entry is
  `biopb-shim`, which starts unbound: the agent calls `attach` and chooses from the
  live sessions, and only Codex is registered with `--session auto` (newest free
  session, else a new one), because it does not follow `list_changed`. So a button
  on a session's pane cannot say "this one" unless the launch hands the shim the
  id, and the shim reads `$BIOPB_SESSION`. That reaches a client whose MCP child
  inherits the control's environment; whether opencode's and Claude Code's do is
  unmeasured, and Codex passes its MCP servers a short fixed environment, so it
  would need the id on the `--session` argument instead. Until then, label the
  button as launching a client that attaches, not one wired to that session.
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

### Re-checked 2026-10-15

Linux, opencode 1.18.34 and 2.0.29 (`@opencode/cli`, isolated `HOME` and XDG
paths), Claude Code 2.1.300, Codex 0.157.1. The 2026-09-05 results above were not
repeated; these were.

- **Commands still exist.** `claude remote-control` with `--spawn
  {same-dir,worktree,session}` (default `same-dir`) and `--capacity` (default 32);
  `claude agents` (`--json`, `--all`); `codex app-server daemon` with
  `enable-remote-control` / `disable-remote-control`; `codex remote-control
  start|stop|pair` (marked experimental).
- **opencode 2 has no `web`.** Its commands include `pair`, `serve`, `service`,
  `mcp`, `auth`, `acp`, `run`. `opencode pair` started the background service
  on a loopback port and printed a single-use link, valid 5 minutes, with SSH
  forwarding instructions; `opencode service status` printed the same address and
  `service stop` closed it.
- **opencode 2 authentication.** On the service and on `serve`, `/` answered 200
  and `/api/session` 401. `serve` with `OPENCODE_SERVER_PASSWORD=<pw>` answered
  200 to `-u opencode:<pw>`; with it unset, `serve` printed a generated `server
  password` and still answered 401 without it.
- **opencode 2 config.** A server registered as `mcp.biopb` (the v1 shape) was
  not seen (`opencode mcp list`: "No MCP servers configured"); the same server
  under `mcp.servers.biopb` was.
- **Not measured.** MCP connection scope under opencode 2's service, `pair
  --remote` (it exposes the server to the internet), Windows.
