# MCP sessions & the control plane

How a biopb deployment is structured at runtime: a durable **control plane** at
the root, durable **data / algorithm planes** it supervises, and **ephemeral,
shim-owned MCP sessions** that *use* those planes but never start them. This
absorbs two earlier notes — why the shim/heavy split exists, and how sessions and
the control relate. Everything below is built except the algorithm plane under
the control (the `[pending]` node in the diagram), which is not yet done; when it
lands, the plane configs stay **federated, not merged** — one writer domain per
file (`biopb.json` / `control.json` / `mcp-config.json`), since only the process
that validates a file's schema can own writing it.

## Why this shape

1. **fd-1 corruption → the shim/heavy split.** Under the stdio MCP transport,
   **fd 1 *is* the JSON-RPC channel**. The heavy MCP process (uvicorn + Qt + dask
   + kernel) writing *anything* to stdout — a uvicorn WARNING, stray kernel output
   — corrupts the protocol stream and crashes the client. So the process the
   client spawns is a **featherweight shim** that owns fd 1 and imports only the
   mcp SDK; all heavy work runs in a *separate* child it bridges to over http. The
   fd-1 corruption class becomes structurally impossible.

2. **A bootstrap cycle.** The MCP session depends on the data plane, yet its own
   data-access layer used to *start* the data plane (`_connection.auto_connect →
   start_local_server`). A consumer starting its dependency is a cycle, not a DAG,
   with no clean answer to "who owns the tensor server." A control plane owns that;
   `_connection` becomes a pure client.

3. **N ephemeral web surfaces.** Each session's `observe` UI lives on a dynamic
   per-connection port, so no static bookmark or reverse-proxy upstream can reach
   it. A single web origin needs an owner that discovers sessions dynamically.

Plus the daemon model's env/orphan bugs: **#98** (a login-time daemon freezes
`DISPLAY`, so the agent's viewer lands on the wrong display) and **#403** (Windows
kernel orphans).

## The architecture

```
   control plane   (durable ROOT — lean: supervise + route + serve the web UI)
        ├── supervises ─► data plane      (tensor Flight server + HTTP sidecar)
        ├── supervises ─► algorithm plane (algorithm servers)          [pending]
        └── observes   ◄─ MCP sessions    (ephemeral, SHIM-owned; self-register)
                            env inherited from the shim  (the #98 fix)
                            USE the planes; never START them
   dependency graph: a tree rooted at the control. No cycles.
```

**Two invariants keep it correct:**

- **I1 — the control *observes* sessions, never *spawns* them.** A control-spawned
  session would inherit the control's frozen env, re-breaking #98. Sessions stay
  shim-owned and env-inherited; they only **register** so the control can route to
  and list them.
- **I2 — the control stays lean and subprocess-based.** It supervises components as
  subprocesses (`python -m biopb_tensor_server …`, `python -m biopb_mcp.mcp …`),
  never by importing them — so no Qt/napari/dask/kernel and no cross-package
  imports enter it. Shared facts (the control endpoint, the session-file contract,
  auth predicates) live in **stdlib-only core-SDK modules** (`biopb._config_control`,
  `biopb._config_sessions`, `biopb._web_auth`) that neither side imports from the
  other.

| Component | Lifetime | Owned by | Role |
|---|---|---|---|
| Control plane | durable (root) | OS service / `biopb` launcher | supervise, route, single-origin web front, session registry, auth/TLS termination |
| Data plane | durable | control (subprocess) | pixels, cache, remote-data proxy |
| Algorithm plane | durable | control (subprocess) | compute ops *(pending)* |
| MCP session | ephemeral | the **shim** | kernel + dask + viewer; env-inherited; registers with control |
| Shim | per client connection | the MCP client | stdio↔http bridge; spawns & reaps its session child |

## The control plane

A small always-on process (`biopb-control` package; `biopb control
start/stop/status/run`) — a Starlette/uvicorn app on `127.0.0.1:8813` that:

- **supervises the data plane.** `DataPlaneSupervisor` spawns the tensor server
  (same argv as `biopb server start`), polls liveness, restarts on crash with
  capped backoff. It is the **sole owner**: it always spawns its own child and
  refuses a port already in use rather than adopting a foreign server — so
  `self._proc` is the whole state and `control stop` is a *complete* teardown (no
  "adopted, left running" case). It imports neither heavy package (I2); the only
  shared fact, its loopback endpoint, lives in `biopb._config_control`. `biopb
  control start` brings the plane up by default (`--no-data-plane` runs the control
  alone).
- **is the single web origin** — serves the built `web/` SPA at base `/` and
  reverse-proxies each plane under its own prefix (below).
- **holds the session registry** and terminates auth/TLS.

**Failure semantics:** the control is a single point of failure for *supervision
and routing only* — already-spawned planes keep serving through a control blip and
re-register when it returns. Keeping it lean and crash-only-restartable bounds the
blast radius (contrast a process-merge, where a fault would take down data serving
itself).

## Shim-owned sessions

Each stdio shim spawns and owns its own MCP session child — FastAPI/uvicorn + the
FastMCP app + the kernel host — on a **dynamic OS-assigned port** (`--port 0`),
bridges stdio ↔ that child's `/mcp`, and reaps it (and its kernel grandchild) when
the client disconnects. The child:

- **inherits the shim's live environment** (`DISPLAY` / `XAUTHORITY` /
  `WAYLAND_DISPLAY` are the user's current session — the #98 fix), so the agent's
  viewer lands on the human's real display and the CLAUDE.md §2.6 shared-canvas
  thesis holds.
- **reports its assigned port** to the shim over an inherited pipe fd
  (`BIOPB_PORT_REPORT_FD`).
- **registers with the control** on startup and deregisters on reap.
- is **reaped as a tree**: POSIX via process-group / parent-death pipe; Windows via
  a **Job Object** the shim creates and assigns (`_winjob.py`, #403), so a
  force-killed shim takes the whole subtree down with it.

The kernel itself stays **lazy** — `start_kernel` (an explicit tool the agent
calls) brings up the napari viewer + dask cluster; kernel-dependent tools guard
with "call `start_kernel`" when it is down. So the shim-spawn cost is only the
http-stack import; the heavy GUI start is deferred to first use.

**Accepted trade-offs:** a session ends when its client quits (durable state lives
in the data plane, which the control keeps up); two agents get two independent
viewers (isolation) at the cost of N dask clusters / kernels / viewers.

### The standalone `biopb mcp view`

The no-agent path: a **foreground, blocking, Ctrl-C** viewer that opens napari
immediately, binds a dynamic port, prints its `/mcp` URL for optional agent attach,
and writes no PID file. It stays **fully standalone** — it does *not* register with
the control (an agentless foreground session has nothing to observe and no remote
lifecycle to control), and works whether or not the control is running. The old
shared `biopb mcp start/stop/status` daemon is **deprecated** (still works, emits a
notice).

## Single-origin web front

The control serves **one base-`/` SPA** and gives every downstream plane its own
path prefix, so the three `/api/*` namespaces that used to collide at the root
never meet:

| Path | Target | Hop |
|---|---|---|
| `/`, `/viewer`, `/admin`, `/assets/*` | control-served `web/` SPA | in-process |
| `/api/*` | control's own API (`status`, `sessions`, `data_plane/{ensure,stop,restart}`) | in-process |
| `/health` | bare liveness (installer / `_control_client`) | in-process |
| `/data_plane/api/*`, `/data_plane/ws/render` | tensor sidecar (API-only) | loopback proxy |
| `/session/<id>/observe` | control-served SPA observe shell | in-process |
| `/session/<id>/api/*` | that session's observe API | loopback proxy |
| `/mcp` | agent JSON-RPC — **not routed here**; shim → child, direct | — |

The SPA is built with **base `/`** (`VITE_TENSOR_API=/data_plane`, no
`VITE_BASE_PATH`), so its `/assets/*` resolve from the absolute root whatever prefix
a shell was served under. `/data_plane/*` is a pure prefix-stripping proxy into the
sidecar (now **API-only** — the control owns the whole UI, so there is no
`/data_plane/viewer`); control *verbs about* the plane live under
`/api/data_plane/*`, so proxy and verbs never mix.

**Session registry.** Each session writes
`~/.local/share/biopb/sessions/<id>.json` (host + port + pid + `/mcp` url) once its
child is reachable and removes it on reap; the control reads that dir. The contract
is stdlib-only `biopb._config_sessions` (shim writes, control reads, neither imports
the other). `resolve()` / `list_sessions()` **self-heal** by pruning records whose
owning pid is dead — or alive but a different process on a recycled pid (create-time
token mismatch, the biopb#138 PID-reuse guard); liveness/identity go through the
dependency-free `biopb._proc`.

**Origin auth.** The web API is gated at the origin (`biopb._web_auth`, shared by
control + sidecar + observe): when a data-plane token is configured it is required
(`Bearer` / `X-Biopb-Token`); in local mode (all-loopback) a **loopback-Host** check
is the DNS-rebinding backstop; and every state-changing verb refuses a forgeable
cross-site request (CSRF). Because the `/session/<id>/*` proxy hop strips the child's
own Host/Origin guard, this control-side check is the child's *only* auth.

## Gotchas

- **Supervised restart must not go through the blind proxy** (biopb#418). The
  sidecar's `/api/admin/restart` self-restarts by spawning a detached `biopb server
  restart` — correct standalone, but under supervision it SIGTERMs the control's
  tracked child and races the supervisor for the gRPC port. So the control marks
  its child (`BIOPB_DATA_PLANE_SUPERVISED`), the sidecar surfaces `supervised` on
  `/api/admin/status` and **refuses** self-restart (409), and the admin UI routes a
  supervised restart to the in-process supervisor verb `/api/data_plane/restart`.
  Config *edits* stay a blind proxy (the tensor process is the sole validator/writer
  of `biopb.json`); only *restart* — an ownership action — is control-routed.
- **The `/session/<id>` proxy is an allowlist, not a denylist.** The child's `/mcp`
  (an `execute_code` RCE) shares the child port, and the Host/Origin strip is its
  entire auth. httpx normalizes dot-segments, so a denylist would let `api/../mcp`
  (or its `%2e%2e` form) collapse onto `/mcp`. Only a first path segment of
  `observe` or `api` is proxied; parent-traversal is rejected.
- **Observe uses SSE — the proxy must stream** (a streaming ASGI passthrough, not a
  buffering fetch). With explicit prefix `Mount`s (no root catch-all) the static
  `/`-fallback cannot swallow `/session/<id>/*` or `/data_plane/*`.
- **Stale upstreams / reap consistency.** A dead session must expire from the
  registry (file removed on reap + self-heal prune) so routing returns a clean
  "session ended", not a hang; cleanup is tied to the shim's single reap path, so
  even the `os._exit` signal handler de-registers — no routing ghost.
- **Resource multiplication & latency.** N sessions = N clusters/kernels/viewers
  (was one shared); startup latency is per-session (no daemon to amortize the
  http-stack import). Both accepted for session-scoped clients; a shared-cluster
  opt-in is the door left open.

## Code anchors

- `biopb-mcp/src/biopb_mcp/mcp/_shim.py` — `spawn_session` (dynamic-port child, env
  inheritance, port pipe, tree reap) + the stdio↔http bridge.
- `biopb-mcp/src/biopb_mcp/mcp/{_server,_kernel}.py` — the FastMCP app + `KernelHost`
  (lazy `start_kernel`, the serializing `RLock`).
- `biopb-mcp/src/biopb_mcp/mcp/_observe.py` — the per-session observe UI (derives its
  base path at runtime, so the control never rewrites the HTML — keeps I2).
- `biopb-mcp/src/biopb_mcp/_connection.py` — the pure client (asks the control to
  ensure the plane; never shells out a server).
- `biopb-control/src/biopb_control/` — `_control.py` (the ASGI origin: SPA + plane
  proxies) and `_supervisor.py` (`DataPlaneSupervisor`).
- `biopb/_config_control.py`, `biopb/_config_sessions.py`, `biopb/_web_auth.py`,
  `biopb/_proc.py`, `biopb/_algorithms.py` — the stdlib-only cross-process seams.
