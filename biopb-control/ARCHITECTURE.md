# biopb-control Architecture

## Role

The control plane is the **durable root** of a biopb deployment — a small
always-on Starlette/uvicorn app that does three things:

- **supervises the durable planes** as subprocesses (the data plane today, the
  algorithm plane pending),
- **is the single web origin** — it serves the browser SPA and reverse-proxies
  everything behind it,
- **holds the session registry**, so ephemeral MCP sessions on dynamic ports are
  discoverable and routable.

Everything else in the deployment hangs off it, and nothing above it: the planes
are its children, and MCP sessions are independent clients that merely register.

## Invariants

Two rules keep that tree correct, and every change here must preserve them.

- **I1 — the control never *owns* a session.** A session serving an MCP client is
  spawned by that client's shim and only **registers**, so the control routes to
  and lists it without holding it. The one session the control may *launch* is an
  agentless `biopb mcp view` viewer, whose only other spawner is a terminal; that
  child is detached and self-registering, so the registry still only observes and
  a control restart never closes the user's window. What both preserve is
  biopb/biopb-mcp#98: a session inherits its spawner's environment, and the wrong
  one puts the napari viewer where the user is not. So a launch is refused unless
  this control is loopback-bound **and** has a display of its own, and it launches
  `--view`, which exits rather than falling back to a virtual display nobody can
  see.
- **I2 — the control stays lean and subprocess-based.** It supervises components
  as subprocesses, never by importing them, so no Qt/napari/dask/kernel ever enters
  this process. Facts shared with those components — the control endpoint, the
  session-file contract, auth predicates, process/lifecycle helpers — live in
  **stdlib-only core-SDK modules** that neither side imports from the other.

## Data-plane supervision

`DataPlaneSupervisor` spawns the tensor server, polls liveness with a stdlib TCP
connect (no pyarrow/grpc imported — I2), and restarts it on crash with capped
backoff.

It is the **sole owner**: it always spawns its own child and **refuses a port
already held by a foreign process**, surfacing that as a conflict rather than
adopting it. So the tracked child is the whole state, and `control stop` is a
complete teardown with no "adopted, left running" case.

**The plane is bound to the control's lifetime.** An orphaned plane keeps holding
the gRPC port, which the next `control start` reads as a conflict it refuses — so a
crashed control would wedge every restart behind a plane nobody owns. The
supervisor closes that by tying the plane's life to its own: a **parent-death
pipe** on POSIX (the plane self-terminates as a contained group-kill), a
**kill-on-close Job Object** on Windows (the OS reaps it). Both fire when the
control dies **uncatchably** — SIGKILL, OOM, crash, logout.

That binding is **orthogonal to the graceful stop** path — SIGTERM on POSIX, a
sentinel file on Windows — which still runs the plane's orderly shutdown. The
binding is only the backstop for when it is not.

## The single web origin

A session's `observe` UI lives on a **dynamic port**, so N sessions would otherwise
mean N bookmarks — a single origin needs an owner that discovers sessions at request
time. And the data plane, the control, and each session all expose an `/api/*`
namespace, which would collide at the root. So the control serves
**one base-`/` SPA** and gives every target its own prefix:

| Path | Target | Hop |
|---|---|---|
| `/`, `/viewer`, `/admin`, `/assets/*` | control-served `web/` SPA | in-process |
| `/api/*` | control's own API (status, sessions, data-plane verbs, viewer launch) | in-process |
| `/health` | bare liveness | in-process |
| `/data_plane/api/*` | tensor sidecar (API-only) | loopback proxy |
| `/session/<id>/observe` | control-served SPA observe shell | in-process |
| `/session/<id>/api/*` | that session's observe API | loopback proxy |
| `/session/<id>/console/*` | that session's user console — **loopback-bound control only** | loopback proxy |
| `/mcp` | agent JSON-RPC — **not routed here**; shim → child, direct | — |

The SPA is built with base `/` so its assets resolve from the root under any shell
prefix. `/data_plane/*` is a pure prefix-stripping proxy into the sidecar, while
control *verbs about* the plane live under `/api/data_plane/*`, so proxy and verbs
never mix. Observe uses SSE, so its proxy is a streaming passthrough; explicit
prefix mounts — no root catch-all — keep the static `/`-fallback from swallowing
the session and data-plane prefixes.

Because the data plane is the control's child, clients ask the control to *ensure*
it rather than starting one themselves.

## Security model

Being the single origin makes the control the **one place** the browser is
authenticated, for itself and for everything it fronts.

- **Web-origin auth**, shared by control + tensor sidecar + session observe: a
  configured data-plane token is required; in all-loopback local mode a
  loopback-`Host` check is the DNS-rebinding backstop; and every state-changing
  verb refuses a forgeable cross-site request. Because the proxy hop strips a
  proxied child's own Host/Origin guard, this check is that child's **only** auth.
- **The `/session/<id>` proxy is an allowlist, not a denylist.** A session child's
  `/mcp` is arbitrary code execution sharing the same port as its observe API, and
  path normalization would let a denylist be walked around (`api/../mcp`
  collapsing onto `/mcp`). Only a first path segment of `api` — or `console`,
  under the rule below — is proxied; parent traversal is rejected. (`observe` is
  not proxied at all: the page is the control's own SPA shell, served in-process.)
- **The user console is a separate root, gated on this listener's bind.** A code
  cell on the observe page runs in that session's kernel, so it is an RCE on the
  same origin the allowlist above exists to keep RCE off. Folding it into `api`
  would leave that allowlist enforced but no longer true, so it gets its own root
  and is proxied only when the control is loopback-bound — `api` always, `console`
  local-mode only, `/mcp` never. The control decides because only it knows its own
  bind: the proxy hop strips Host and Origin, so the child cannot tell a browser
  from this trusted hop. Not gated by the token instead: that credential
  authorizes reading pixels, is readable from a local file by design, and rides
  the render WS as a query param — fine for viewing, not a thing to trade for a
  shell. Known limit: a loopback control published by a reverse proxy reads as
  local; that operator owns the exposure decision, as they already do for the
  data-plane token.
- **Supervised restart is control-routed, not blind-proxied.** The tensor
  sidecar's self-restart spawns a detached process — correct standalone, but under
  supervision it would race the supervisor for the port. So the control marks its
  child, the sidecar surfaces that and refuses self-restart, and the admin UI
  routes restart through the control instead. Config *edits* stay a blind proxy —
  the tensor process is the sole validator of its own config; only *restart*, an
  ownership action, is control-routed.

## Session registry

Each session writes a JSON record — host, port, pid, `/mcp` url — into a state dir
once it is reachable, and removes it on reap; the control reads that dir. The
contract is a stdlib-only core-SDK module (I2): the session side writes, the
control reads, and neither imports the other.

There are two writers, because there are two ways a session comes to exist. A
shim-owned child is published by its **shim**, which owns its reap and so its
de-registration. An agentless `biopb mcp view` session has no shim, so it
**publishes itself** and drops its record on the way out. Either way the control
only ever reads.

Lookups **self-heal**, pruning records whose owning pid is dead — or alive on a
recycled pid, caught by a create-time token — so a dead session expires to a clean
"session ended" rather than a hang.

`POST /api/sessions/new` is the third way a session comes to exist: the control
spawns `biopb mcp view` and waits for it to appear in this registry, matched on
the child's own pid. Registration is an exact readiness signal — `--view` opens
its window *before* it registers — so a record means a viewer really opened, and
a child that dies first never registers and comes back with its own log tail.
The verb is offered only where it can work (I1); the dashboard reads that from
`/api/status` and shows the refusal in the button's place.

**Stopping one is not the mirror image.** The control does not signal a pid — it
proxies `/session/<id>/api/shutdown`, and the session runs the same teardown
Ctrl-C does. So ownership never enters it: a viewer started from a terminal and
one started here are the same process ending itself, and the control keeps no
record of which it launched. The route rides `api` rather than the local-only
gate (it is not an execute surface, and `api` already carries the kernel
restart), and only a session that owns its own reap serves it — a shim-owned
child does not, since ending it would leave its shim bridging to a dead process.
