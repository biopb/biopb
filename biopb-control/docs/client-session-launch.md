# Launching an agent client from the dashboard

The dashboard knows which MCP clients are installed and whether biopb is
registered with each (`GET /api/agents`, backed by `biopb._control._agents`). For
the clients that can be driven from a browser it also starts them: a **Start**
button on the client's row runs the client's web or remote command, opens the
link it prints in a new tab, and becomes **Stop**.

A client does not own a biopb session. It registers `biopb-shim`, which starts
unbound and, when the agent calls `attach`, takes the **lease** of a session (a
detached process the control launches with `POST /api/sessions/new`, which runs
until stopped from the dashboard). Launching a client starts a *holder* of a
session, not its parent.

## Clients

| client | command | where the UI is | the control holds |
|---|---|---|---|
| opencode 1 | `opencode serve --hostname 127.0.0.1 --port 0` | its loopback web UI | an owned child |
| opencode 2 | `opencode pair` | a loopback background service; the printed one-time link signs a browser in | nothing (a command that returns) |
| Claude Code | `claude remote-control --spawn session --permission-mode default` | claude.ai/code or the mobile app, over Anthropic's relay | an owned child |
| Codex CLI | `codex app-server daemon enable-remote-control` | ChatGPT, over OpenAI's relay, after a pairing code | nothing (a command that returns) |

Claude Desktop and Cursor have no web or remote surface and never get a button.

- **opencode** is probed with `opencode --version` (the probe that also picks the
  config shape). opencode 1 uses `serve`, the same server as `web` without the
  browser the dashboard opens itself. opencode 2's service is machine-wide
  (`opencode service start|status|stop`); `pair` prints a link that works once and
  expires in 5 minutes.
- **Claude Code** runs one conversation per process (`--spawn session`), so Stop
  ends that conversation. The default mode hosts many conversations in one daemon
  and works too, but Stop then ends them all. A shim per conversation is cheap: it
  creates no session until its agent attaches.
- **Codex's** daemon is machine-wide and runs only from Codex's own installer
  (`~/.codex/packages/standalone/current/codex`), not an npm install. Stop runs
  `disable-remote-control`.

## Backend contract

`ClientBackend` (in `biopb._control._agents`) declares, per client:

| | |
|---|---|
| `has_launch` | whether the client has a launch surface at all |
| `launch_argv(remote=)` / `stop_argv(remote=)` | the commands; `None` when there is none |
| `launch_kind` | `child` (a process the control owns) or `oneshot` (a command that returns) |
| `can_launch()` | `launch_argv()` names a program on PATH (Codex also needs its standalone install) |
| `launch_hint()` | what is missing, for a client that has a surface but cannot launch here |
| `relay` | reachable from other devices through a relay |
| `ready_pattern` | the output that means a `child` is usable (opencode's URL, Claude Code's `claude.ai` link) |

`can_launch` is a second predicate beside registration state, not part of
`is_installed()`: registration is a config read, so a client can be registered
with no CLI present, and `is_installed()` stays loose so Register keeps working.
`status()` reports `can_launch`, `launch_hint` and `relay` with `state`, and
`GET /api/agents` adds `running` and `launch_blocked`.

## The launcher

`ClientLauncher` (`biopb_control/_clients.py`), one per control:

- **`launch(id)`** refuses with `LaunchRefused` (403) if blocked, else runs the
  command. A `child` is spawned as an `OwnedChild` with output going to a
  per-launch log, and is **started** when `ready_pattern` shows (or, with none,
  after it has stayed up 3 s), **starting** if the pattern has not shown after 15
  s, **failed** if it exits first. A `oneshot` runs to completion and is
  remembered. One launch per client at a time; the lock is not held while waiting.
- **`stop(id)`** stops only a client this control launched: `OwnedChild.stop()`
  for a child, the backend's `stop_argv` for a `oneshot`. The decision is the
  handle the control holds, never a pid. A client the user started themselves
  (their own `opencode serve`, a Codex daemon already running) is refused.
- **Environment.** The child inherits the control's environment minus its exec
  token.

Every response carries `state`, the output so far as `log`, and `log_path`; only
`failed` carries `error`.

## Dashboard

- **Start / Stop** on a registered client's row. A client with a surface that
  cannot launch shows Start disabled, with `launch_hint` or `launch_blocked` as
  its tooltip. Claude Desktop and Cursor show neither.
- **A tab is opened on click** (a tab opened after the launch returns would be
  blocked) and pointed at the first `http(s)` link in the output. With no link,
  or on failure, it closes.
- The launch **output is not pasted into the card**. The card's *Logs →* link
  opens the shared log page at `?source=agents`, which tails the newest client
  log (`GET /api/agents/logs`). A `oneshot` has no log file, since its output
  holds a credential (opencode 2's link), and shows its output only on failure.
- The Data plane, Sessions and Agent clients cards carry their *Config →* and
  *Logs →* links in the heading. Sessions' *Logs →* is the newest session the
  dashboard launched (`?source=sessions`).

## Ownership and lifecycle

The repo has two child lifecycles. **Owned** (`biopb._lifecycle.OwnedChild`)
dies with its parent. **Detached** (`detach_kwargs`) survives it; a biopb session
uses this deliberately, so a control restart does not end it.

**A client is owned; the session it holds is detached.** The chain is
`control → client → biopb-shim`, each link dying with its parent; the shim
releases its lease on stdin EOF or SIGTERM, and a killed shim's lease lapses
after 30 s. Nothing outlives the control but sessions. The cost: **restarting the
control ends client sessions** (including a remote one someone is attached to
from a phone), while the biopb session survives and the next client can attach
to it.

A `oneshot` outlives the control by nature: its service is machine-wide, and
Stop never takes it down unless the launch exposed it (below).

**Stop the client first when both go.** A shim started with `--session auto`
(Codex) that is respawned after its session is gone attaches to the newest free
session or launches a new one.

## The lease names its client

On `attach` the shim sends the client's `pid`, `create_time` and process name
with `/api/lease/acquire` (the shim finds its client past launcher stubs with
`_find_client_process`). The session keeps them for as long as the lease lives,
and `/api/sessions` returns them, so a session row can say "held by opencode".

It is provenance for display. It is never a kill target, and Stop does not match
on it: the process the shim finds is the client's session process, which for
Claude Code is a descendant of the `remote-control` child the control holds.

## Gating and security

- **Every launch needs the `chat` execute root to be served** (so `--hide chat`
  closes it), because the client it starts is an agent with a shell. Launch and
  stop are exec-credential verbs in `split` auth mode.
- **Relay clients need the operator's opt-in**: `biopb control start
  --allow-relay-clients` (or `BIOPB_CONTROL_RELAY_CLIENTS=1`). Being reachable
  from another device is the point of a relay client, so it cannot ride a
  loopback-only gate, and starting a shell on the owner's subscription should be
  deliberate.
- **opencode 2 with the opt-in** runs `pair --remote --yes`, which exposes the
  server to the internet through opencode's tunnel (`--yes` because there is no
  terminal to confirm on), and Stop then runs `service stop`, the only way to take
  the exposure back off. Without it, only the local `pair` runs. `pair --remote`
  has not been exercised.
- `GET /api/agents` reports the reason as `launch_blocked`, from the same function
  the verb enforces.
- **Authentication per client.** opencode 1's listener is unauthenticated unless
  `OPENCODE_SERVER_PASSWORD` is set in the control's environment; it prints its
  own warning, which is in the log. opencode 2 is authenticated by default: its
  API answers 401 until a browser has used a pairing link. Claude Code and Codex
  expose no local listener: they poll or connect out to a vendor relay, and the
  user's own account (Claude Code) or an enrolled device plus a pairing code
  (Codex) is the authentication. Pairing links and passwords are credentials.
- Claude Code runs with `--permission-mode default`, never `bypassPermissions`.
  A web session runs under the user's own opencode settings, where `bash` and
  `edit` default to `allow`; nothing is pinned on their behalf.
- Anthropic stores a Remote Control session's transcript server-side while it is
  connected, which matters to a lab running biopb locally for data-governance
  reasons. Codex's retention is unverified.

## Open questions

- **Which biopb session a launched client gets.** The registered entry is
  `biopb-shim`: the agent calls `attach` and chooses, and only Codex is
  registered with `--session auto`. A launch cannot yet say "this session". The
  shim reads `$BIOPB_SESSION`, which reaches a client whose MCP child inherits the
  control's environment; whether opencode's and Claude Code's do is unmeasured,
  and Codex passes MCP servers a short fixed environment.
- **opencode 2's MCP connection scope.** opencode 1 and Codex run one shim per
  server, so one client is one biopb session; Claude Code runs one per
  conversation. opencode 2's service hosts the servers, so opencode 1's result is
  the expectation, not a measurement.
- Whether the launch paths work on Windows and macOS: `OwnedChild.stop()` there
  is `TerminateJobObject` (no graceful shutdown), and macOS has no
  `process_create_time`, so nothing there can guard against pid reuse.
- An owned client does not survive a graceful control shutdown, but a `kill -9`
  of the control leaves it running: the client cannot install the parent-death
  watch its own children do.
