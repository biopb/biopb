# biopb-control Architecture

## Role

The control plane is the **durable root** of a biopb deployment — a small
always-on Starlette/uvicorn app that does exactly three things:

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

- **I1 — the control *observes* sessions, never *spawns* them.** A control-spawned
  session would inherit the control's frozen environment, putting the agent's
  napari viewer on the wrong display (biopb/biopb-mcp#98). Sessions stay
  shim-owned and env-inherited; they only **register** so the control can route to
  and list them.
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
sentinel file on Windows — which still runs the plane's orderly shutdown, releasing
its file-cache process lock, whenever the control is alive to ask. The binding is
only the backstop for when it is not. The cost is a brief data-serving gap across a
control restart; keeping the control lean and crash-only-restartable bounds it.

## The single web origin

Two problems make one origin the right shape. A session's `observe` UI lives on a
**dynamic port**, so N sessions would otherwise mean N bookmarks — a single origin
needs an owner that discovers sessions at request time. And the data plane, the
control, and each session all expose an `/api/*` namespace, which would collide at
the root.

So the control serves **one base-`/` SPA** and gives every target its own prefix:

| Path | Target | Hop |
|---|---|---|
| `/`, `/viewer`, `/admin`, `/assets/*` | control-served `web/` SPA | in-process |
| `/api/*` | control's own API (status, sessions, data-plane verbs) | in-process |
| `/health` | bare liveness | in-process |
| `/data_plane/api/*`, `/data_plane/ws/render` | tensor sidecar (API-only) | loopback proxy |
| `/session/<id>/observe` | control-served SPA observe shell | in-process |
| `/session/<id>/api/*` | that session's observe API | loopback proxy |
| `/mcp` | agent JSON-RPC — **not routed here**; shim → child, direct | — |

The SPA is built with base `/` so its assets resolve from the root under any shell
prefix. `/data_plane/*` is a pure prefix-stripping proxy into the sidecar, while
control *verbs about* the plane live under `/api/data_plane/*`, so proxy and verbs
never mix. Observe uses SSE, so its proxy is a streaming passthrough; explicit
prefix mounts — no root catch-all — keep the static `/`-fallback from swallowing
the session and data-plane prefixes.

Because the data plane is the control's child, clients ask the control to *ensure*
it rather than starting one themselves. This is what broke the old bootstrap cycle,
where a session depended on the data plane yet its own data layer started it — a
cycle with no clean owner.

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
  collapsing onto `/mcp`). Only a first path segment of `observe` or `api` is
  proxied; parent traversal is rejected.
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
contract is a stdlib-only core-SDK module (I2): the session's shim writes, the
control reads, and neither imports the other.

Lookups **self-heal**, pruning records whose owning pid is dead — or alive on a
recycled pid, caught by a create-time token — so a dead session expires to a clean
"session ended" rather than a hang.
