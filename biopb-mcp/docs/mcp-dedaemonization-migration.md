# Control-plane architecture: a central control endpoint

**Status:** design / planned. Supersedes and absorbs the earlier
"de-daemonization migration" note — MCP de-daemonization is now **Layer 1** of a
larger move to a central control plane. Retains the shim/heavy *split* rationale
from `daemon-migration.md` (that split stays; only the daemon's detachment and
its role as lifecycle root are undone).

> Filename is historical (`mcp-dedaemonization-migration.md`); scope is now the
> whole-system control plane. Rename on first commit if desired.

---

## 1. The problem: three tangles in the current shape

### 1.1 A bootstrap loop — the dependent starts its dependency

The MCP session depends on the data plane, yet the MCP session's own
data-access layer *starts* the data plane:
`MCP kernel → _connection.auto_connect → start_local_server`
(`biopb-mcp/src/biopb_mcp/_connection.py:594–689`:
`can_autostart_server()` / `start_local_server()` / `auto_connect()`). The tensor
server is thus subordinate to the thing that consumes it — a cycle, not a DAG,
and the reason "who brings up the tensor server" has no clean answer today.

### 1.2 Fragmented web surfaces

Two independent web origins exist: the tensor server's FastAPI sidecar (webapp +
slice API) and each MCP session's `observe` UI. After de-daemonization (Layer 1)
the observe surface is worse than two — it's *ephemeral and on a dynamic port
per session*, so no static bookmark or static reverse-proxy `upstream` can reach
it. A single origin now *requires* dynamic discovery, which needs an owner.

### 1.3 Lifecycle and configuration scattered across components

Each component starts its own dependencies; the algorithm-plane configuration
lives inside MCP; and the daemon model carries the env/orphan bug class (#98
frozen `DISPLAY`, #403 Windows kernel orphan). There is no single place that
knows "what is running" or that owns "what should be running."

---

## 2. Target architecture

A lean, always-on **control plane** becomes the durable root. It
supervises the durable planes, observes the ephemeral sessions, and is the single
web origin.

```
   biopb control plane      (durable ROOT — lean: supervise + route + static UI)
        │
        ├── supervises ──►  data plane      (tensor Flight server)           ┐ durable,
        ├── supervises ──►  algorithm plane (algorithm servers + their config)┘ control-owned
        │
        └── observes   ◄──  MCP sessions    (ephemeral, SHIM-owned)  ── self-register on start
                              │  env inherited from the shim   (this is the #98 fix)
                              └─ USE the data / algorithm planes; never START them

   dependency graph: a tree rooted at the control. No cycles.
```

### The two invariants that make this correct

**I1 — The control supervises durable planes but only *observes* ephemeral
sessions; it must never *spawn* a session.** If the control spawned MCP sessions,
each session would inherit the *control's* frozen environment instead of the
client's live one — re-breaking #98 (headless viewer). Sessions stay **shim-owned
and env-inherited**; they merely **register** with the control so it can route to
their observe page and list them. *Control owns the durable planes; shims own the
sessions; sessions register for observability.*

**I2 — The control stays lean and subprocess-based.** Its only jobs are
supervision, routing, and serving a static UI. It launches components as
**subprocesses** (`python -m biopb_tensor_server …`, `python -m biopb_mcp.mcp …`),
never by importing them — so no napari/Qt/dask/kernel and no cross-package
imports enter the control plane. This is the discipline that keeps the control from
becoming the coupling monster the rejected process-merge would have been. It is
not a new pattern: it is the existing `biopb` CLI subprocess supervision
(`biopb server|mcp start/stop`, which already spawns both without importing
either) **promoted from a one-shot command into a persistent process with a
registry, health monitoring, and a web front.**

### Component roles

| Component | Lifetime | Owned by | Role |
|---|---|---|---|
| **Control plane** | durable (root) | the OS service / `biopb` launcher | supervise, route, single-origin web front, session registry, auth/TLS termination |
| **Data plane** (tensor Flight) | durable | control (supervised subprocess) | pixels, cache, remote-data proxy |
| **Algorithm plane** | durable | control (supervised subprocess/container) | compute ops; config lives with control |
| **MCP session** | ephemeral | the **shim** (not the control) | kernel + dask + viewer; env-inherited; registers with control |
| **Shim** | per client connection | the MCP client | stdio bridge; spawns & reaps its session child |

---

## 3. The control endpoint

**Is:** a small always-on process that (a) supervises the durable planes
(start / liveness-monitor / restart-on-request or crash-with-backoff / status),
(b) holds an in-memory **registry** of live MCP sessions that self-register, (c)
serves the **single origin** — static control UI + dataviewer webapp + reverse-proxy
to each session's observe page — and terminates auth/TLS there.

**Is not:** a compute host. No Qt, no napari, no dask cluster, no kernel, no
import of `biopb_tensor_server` or `biopb_mcp`. Everything heavy is a supervised
subprocess.

**Supervision model:** deliberately simple — start, poll liveness, expose status,
restart on request or on crash with backoff. Not an orchestrator; no scheduling,
no scaling. (The tensor server's progressive-discovery `health` fields are the
readiness signal it polls.)

**Bootstrapping:** the control is the root, started by the `biopb` launcher / an OS
service (systemd / launchd / Windows service). Everything else is downstream, so
the graph has a single entry and no cycle.

**Failure semantics:** the control is a single point of failure for
*supervision and routing only* — already-spawned planes keep serving through an
control blip and **re-register** when it returns. Keeping the control lean and
crash-only-restartable is what bounds that blast radius. (Contrast the process
merge, where a fault took down data serving itself.)

---

## 4. Layer 1 — MCP de-daemonization (shim-owned, ephemeral, control-observed)

The tactical layer. Fixes #98/#403 now and makes sessions clean clients.

### 4.1 Decision

- **Each stdio shim spawns and owns its own MCP session child** — FastAPI/uvicorn
  + the FastMCP app — on a **dynamic (OS-assigned) port**, and bridges stdio ↔
  that child. The child dies when the shim's client disconnects.
- The child **inherits the shim's live environment** → `DISPLAY` / `XAUTHORITY` /
  `WAYLAND_DISPLAY` are always the user's current session. This is the root fix
  for #98; the wire-threading/shim-stamping design becomes unnecessary.
- The child **registers with the control** on startup (session id + dynamic port +
  pid) and **deregisters on reap**, so the control can route `/observe` to it and
  list it. Registration/observability is the *only* control touchpoint — the control
  does not own the child's lifecycle (invariant I1).

### 4.2 The §2.6 shared-session thesis is preserved

The agent's child opens napari on the human's *inherited* `DISPLAY`, so the
scientist still watches the agent's live canvas. The perceive→act→verify loop is
intact. `biopb mcp` (below) is the separate no-agent convenience.

### 4.3 What it gives up (accepted)

- **Persistence across a client restart** — a session ends when its client quits.
  Acceptable because durable state lives in the data plane (cache, proxy), which
  the control keeps up; teardown only happens at app-quit anyway. (Public record:
  Claude Code, opencode, and Claude Desktop all keep the stdio server for the
  whole app session and tear down only on quit.)
- **A single session shared across clients** — two agents get two viewers now,
  intentionally (isolation), at the cost of resource multiplication (§9).

### 4.4 Implementation

- **Serve-core refactor (enabler).** Replace `mcp.run(transport="streamable-http")`
  (`_server.py:830`, which spins its own `anyio.run` loop + `uvicorn.Server`) with
  an explicit `uvicorn.Server` over `app = mcp.streamable_http_app()`. Bind port
  `0`; read the assigned port from the socket; report it to the shim over an
  inherited pipe fd (reuse the report-pipe pattern — `BIOPB_TOKEN_REPORT_FD`,
  `_bootstrap.py:277`; add `BIOPB_PORT_REPORT_FD`). Compose the mandatory
  `session_manager.run()` lifespan explicitly. **No** PID file / sentinel / signal
  handlers / `os._exit` in the core — those move to §6's standalone wrapper.
  Register observe routes (`mcp.custom_route`) *before* the app is built
  (`_observe.py:307`).
- **Shim owns the child.** Rewrite `ensure_daemon` (`_shim.py:100`) →
  `spawn_session`: drop the "already listening?" probe, the fixed port, and the
  detachment; spawn a tracked child with **inherited env** + the port pipe; read
  the port; bridge to `http://127.0.0.1:<port>/mcp`; then **register the session
  with the control**. On stdin-EOF / bridge close / shim exit, reap the child *and
  its kernel grandchild* and **deregister**:
  - POSIX — child in the shim's process group / parent-death pipe → group kill.
  - Windows — the **shim** creates a Job Object (`_winjob.py`, #403) and assigns
    the child, so a force-killed shim reaps the whole tree. *Ownership shifts from
    the daemon (holding the kernel) to the shim (holding the child, which contains
    the kernel).*
  - Optional small **linger window** for robustness against a client that briefly
    cycles stdin; likely unnecessary in the fully-owned model.

---

## 5. Layer 2 — Control owns the data plane; break the `_connection` loop

The structural fix. Highest value; unblocks single origin.

> **Status: core landed.** The control skeleton + data-plane supervision +
> `_connection` de-loop shipped as the Layer-2 *core* (phasing step 1). What
> remains for full Layer 2 is trimming the tensor server's public web surface
> (the third bullet), which pairs with the Layer-3 single-origin front.

- **Move the auto-start responsibility up.** *(done)* `_connection`'s
  `can_autostart_server` / `launch_local_server` / `start_local_server` are
  **removed**; `auto_connect` no longer `Popen`s a server. `_connection` is now a
  **pure client**: connect, wait through a `STARTING` data-folder scan, and — when
  the local plane is down — ask the control to ensure it (`biopb_mcp/_control_client.py`),
  never shelling out. With no control reachable it records an actionable
  "run `biopb control start`" status.
- **A lean control supervises the data plane.** *(done)* New **`biopb-control`**
  workspace package (a product component, versioned off the `release-v*` tag),
  managed by `biopb control
  start/stop/status/run` in the core CLI (which owns the pidfile/detach/stop
  plumbing, reused from the server/mcp daemons). `DataPlaneSupervisor` spawns the
  tensor server (the same `python -m biopb_tensor_server.cli launch` argv `biopb
  server start` uses), polls TCP liveness, and restarts on crash with capped
  backoff. **The control is the *sole owner* of the data plane** — it always spawns
  and manages its own child and never adopts a server it did not start (decision:
  the earlier dual spawn/adopt mode was dropped as needless complexity, see §9).
  A gRPC port already in use is a *conflict* it refuses (`control start` errors,
  mirroring `biopb server start`'s port guard), not something to attach to. This
  single-owner rule is what keeps `self._proc` the whole state and makes `biopb
  control stop` a **complete data-plane teardown** (no "adopted, left running"
  case). Per invariant **I2** it imports neither `biopb_tensor_server` nor
  `biopb_mcp`; the only shared fact — the control's loopback endpoint — lives in the
  core SDK (`biopb/_config_control.py`). `biopb control start` brings the plane up by
  default (`--no-data-plane` starts the control without it, on-demand).
- **Control API.** *(done)* A stdlib `ThreadingHTTPServer` on `127.0.0.1:8813`
  exposes `GET /health` and `POST /data_plane/ensure` — the endpoint `_connection`
  calls in place of the old shell-out. This is the seed of the Layer-3 origin; it
  stays stdlib until the front replaces it with an ASGI app on the same port.
- **The tensor server's HTTP sidecar becomes a private upstream fronted by the
  control**, no longer its own public origin. The Layer-3 front reverse-proxies
  the data API + `/ws/render` through 8813 under a `/data_plane/*` namespace,
  forwarding the Bearer token. In the **finalized** Layer-3 model the sidecar is
  **API-only** and serves no browser asset: the control owns the whole UI as one
  base-`/` SPA (dashboard `/`, dataviewer `/viewer`, observe
  `/session/<id>/observe`), so there is **no `/data_plane/viewer`** mount and the
  sidecar has no public web role left (see §6.1).

Result: the cycle in §1.1 becomes a tree rooted at the control.

---

## 6. Layer 3 — Single-origin web front (a facet of the control)

The control *is* the durable web origin, so single origin falls out — it is no
longer a reverse-proxy bolted onto the ephemeral sidecar. The control serves its
**own root dashboard** and reverse-proxies each downstream plane under its own
namespace; no upstream owns the root, and no two upstreams share a path prefix.

> **Status: complete.** Namespaced origin + data-plane proxy + session registry +
> per-session observe + the control-served SPA are all built (§6.1). The
> control API is a Starlette/uvicorn app on `127.0.0.1:8813` (replacing the stdlib
> `ThreadingHTTPServer`). It answers bare `/health`, the control API under `/api/*`
> (`/api/status`, `/api/sessions`, `/api/data_plane/{ensure,stop,restart}`), and
> **serves the built `web/` SPA bundle at its root** (from `--static-dir`) — one
> base-`/` app whose routes are the dashboard (`/`), the dataviewer (`/viewer`),
> and each session's observe shell (`/session/<id>/observe`). It reverse-proxies
> the tensor sidecar under the `/data_plane/*` namespace — **API-only**: data API
> at `/data_plane/api/*` and the `/data_plane/ws/render` WebSocket, with **no
> `/data_plane/viewer`** — via explicit prefix `Mount`s that strip the namespace
> (`biopb_control/_control.py`). The catch-all is retired. The SPA is built with
> base `/` (`VITE_TENSOR_API=/data_plane`; no `VITE_BASE_PATH`).

### 6.1 Decided 2026-07-11 — namespace discipline + a control-owned root

Three decisions fix the routing shape:

1. **Root `/` is the control's own dashboard, not the dataviewer. (BUILT.)** A
   **self-contained, buildless page** — a single embedded HTML+vanilla-JS
   document served in-process, exactly the pattern `_observe.py` already uses (no
   Vite/npm build, so the lean control stays buildless; a real SPA build is
   deferred until the dashboard outgrows status+links). It polls `/api/status`
   (the data-plane snapshot + a live-session count) and `/api/sessions`, exposes
   the data-plane `ensure`/`stop`/`restart` controls (POSTing the
   `/api/data_plane/*` verbs), links to the dataviewer (enabled only when the
   data plane is `serving`), and lists each live session with its
   `/session/<id>/observe` link. *(Finalized: this buildless vanilla-JS page was
   subsequently rewritten into a React route (`DashboardPage.tsx`) of the one
   base-`/` `web/` SPA the control serves — the deferred "real SPA build" — with
   its behavior carried over unchanged.)*

2. **Every downstream plane owns a path prefix; the root catch-all is retired.**
   The flat "proxy everything else" mount is replaced by explicit `Mount`s. Each
   plane's API lives under its own prefix, so the three `/api/*` namespaces that
   used to collide at the root never meet:

   | Whose API | Prefix |
   |---|---|
   | control (the root dashboard) | `/api/*` |
   | data plane (tensor sidecar) | `/data_plane/api/*` |
   | each MCP session (observe) | `/session/<id>/api/*` |

3. **`/observe` stops being a magic singleton.** In the shim-owned, N-session
   model "the current session" is under-defined, so the canonical route is
   per-session — `/session/<id>/observe` (+ `/session/<id>/api/*`). A control-owned
   `/sessions` list page (rendered from the registry) replaces the "current
   session" guess; `/observe` at most becomes a convenience redirect to the sole
   session when exactly one is live. `/mcp` is **not** reused as a session prefix —
   it is the agent JSON-RPC transport, deliberately kept off this origin.

**Routing (finalized target):**

| Path | Target | Hop |
|---|---|---|
| `/`, `/viewer`, `/admin`, `/assets/*` | control-served base-`/` `web/` SPA (dashboard + dataviewer/admin) | in-process (static + SPA fallback) |
| `/api/status` | all sub-components' state | in-process |
| `/api/sessions` | live-session list (from the registry) | in-process |
| `/api/data_plane/ensure` \| `stop` \| `restart` | supervisor verbs | in-process |
| `/health` | bare liveness (installer / `_control_client`) | in-process |
| `/data_plane/api/*` | tensor sidecar data API | loopback proxy |
| `/data_plane/ws/render` | tensor sidecar render WebSocket | loopback proxy |
| `/session/<id>/observe` | control-served SPA shell (observe route) | in-process (static + SPA fallback) |
| `/session/<id>/api/*` | that session's observe API | loopback proxy |
| `/mcp` (agent JSON-RPC) | **not routed here** — shim → child, direct/private | — |

`/data_plane/*` is a **pure proxy** into the tensor sidecar (prefix stripped
before forwarding, since the sidecar serves `/api/*` at its own root); all
control *verbs about* the plane live under `/api/data_plane/*`, so the two never
mix. During the rewrite `/data_plane/ensure` moves to `/api/data_plane/ensure`
(pre-release, so no legacy alias is kept); bare `/health` stays put as the
load-bearing liveness probe.

**Frontend serving (finalized).** The planned per-surface base paths were
dropped. Rather than build the dataviewer under `base: '/data_plane/viewer/'`,
the finalized model builds **one SPA with base `/`** that the control serves at
its root, and every surface — dashboard `/`, dataviewer `/viewer`, observe
`/session/<id>/observe` — is a client route of it. Because assets are requested
from the absolute root (`/assets/*`), they resolve no matter which prefix the
shell was served under, so there is **no `VITE_BASE_PATH`**;
`VITE_TENSOR_API="/data_plane"` still points the viewer at the control-proxied
data plane. CI/release build that single bundle accordingly
(`tensor-server-ci.yaml`, `release.yaml`).

*Implemented for observe (per-session routing):* the control resolves
`/session/<id>/*` to the child's dynamic loopback port per-request via
`biopb._config_sessions.resolve()` (unknown/dead → clean 404, ghost pruned), and
proxies with **Host + Origin dropped** so the child's own loopback Host/Origin
guard passes on the trusted hop whatever external host reached the control.
Only an **allowlist** of the observe surface is proxied — the first path segment
must be `observe` or `api`, and any parent-traversal is rejected. The child's
`/mcp` agent transport (RCE) is on the same port and this Host/Origin strip is its
*entire* auth, so nothing else may through; a *denylist* would be unsafe because
httpx normalizes dot-segments, collapsing `api/../mcp` (or its ASGI-decoded
`%2e%2e` form) to `/mcp`. No agent needs `/mcp` routed here anyway — they reach the
child's `/mcp` directly (§6.1 decision 3). The
`_observe.py` base-path fix needs **no build step and no child env**: the page
derives `BASE = location.pathname.replace(/\/observe\/?$/, '')` at runtime and
prefixes every `/api/*` call, so the same static page works served directly
(`BASE = ""`) or behind `/session/<id>/`. The control never rewrites the HTML
(keeps I2 — it stays oblivious to observe's contents). Origin-wide token auth is
now implemented for the control's own `/api/*` (see "Origin auth — step 1" below);
`/session/<id>/*` gating is the remaining piece.

**Session discovery — a filesystem registry (BUILT).** Each session writes
`~/.local/share/biopb/sessions/<id>.json` (host + port + pid + `/mcp` url) once
its child is fully reachable, and removes it on reap; the control reads that dir.
Matches the existing sentinel/PID-file idioms; no new endpoint. (HTTP
self-registration is the alternative if the front and children may later cross a
machine boundary.)

*Implemented:* the on-disk contract lives in a shared, **stdlib-only**
`biopb._config_sessions` (core SDK — the shim writes it, the control reads it, and
neither can import the other, exactly like `_config_control`). `register()` writes
atomically (temp + `os.replace`) and records a **create-time identity token** for
the child pid; `unregister()` is tied to the shim's single reap path
(`_shim._reap_session`, so even the SIGTERM/SIGHUP signal handler that `os._exit`s
past `serve`'s `finally` still de-registers — no routing ghost); `list_sessions()`
/ `resolve()` **self-heal** by pruning + unlinking any record whose owning process
is gone — the pid is dead, *or* alive but a different process on a recycled pid
(create-time mismatch). That PID-reuse guard is the same one the daemon PID files
use (biopb#138): without it a recycled pid would keep a ghost "alive" and could
route `/session/<id>` traffic to an unrelated process. Liveness/identity is
delegated to the shared, dependency-free `biopb._proc` (`is_process_running` +
`process_create_time` — the latter handles the Windows signal-0 hazard and the
macOS "no token → degrade to liveness-only" case), fail-open on an undecidable
pid. This is the backstop for gotchas 2/3 above.

**Why it is worth more than one bookmark:**
- **Security win.** The observe UI currently fronts an `execute_code` RCE behind
  loopback + Host/Origin only. Behind the control origin, the tensor server's Bearer
  token gates it *uniformly*, and the loopback hop to the child is trusted. This
  is the surface you put TLS + authn in front of for the §1 "trusted intranet"
  hardening — **one origin to front, not N dynamic ports.**
- **No CORS** — same origin for dataviewer + observe → shared token/cookie, no
  cross-origin config, no per-session-port CORS churn.

**Origin auth — implemented.** The control's web API is gated at the origin —
both the control's **own** `/api/*` and each session's proxied
`/session/<id>/api/*`: when a data-plane token is configured it is required
(`Authorization: Bearer` / `X-Biopb-Token`); when it isn't (local mode, all
listeners loopback-bound) a **loopback Host** check is the rebinding backstop; and
every state-changing verb additionally refuses a forgeable cross-site request
(CSRF), mirroring the sidecar's `_require_same_origin`. The decisions live in a
shared, **stdlib-only `biopb._web_auth`** — predicates the control, the tensor
sidecar, and observe all import (none can import another, I2); the sidecar's
`check_token` / `_require_same_origin` were folded onto it (#425). This closes the
stop/restart CSRF-DoS, the `/api/sessions` enumeration the dashboard (#14) opened,
and — with the session-API gate (biopb/biopb#424) — the guessable-session-id CSRF
/ DNS-rebind against the observe kernel verbs; the proxy hop strips the child's own
Host/Origin guard, so this control-side check is the only one. The `/observe` SPA
shell stays open (a plain GET serving the app bundle). `/api/data_plane/ensure`
stays open (idempotent; biopb-mcp's token-less `_control_client` posts it): under
the two-mode model biopb-mcp is local-only and a local control is tokenless, so
gating it would buy nothing (the only residual is that in remote mode it is an
unauthenticated but idempotent public state-change — biopb/biopb#424 item 2). TLS
termination stays a front-proxy concern (§1).

**Gotchas (same class as the merge audit):**
1. **Streaming + explicit mounts.** Observe uses SSE; its proxy must stream (no
   buffering). With the root catch-all retired for explicit prefix `Mount`s
   (§6.1), the old "exempt observe from the StaticFiles 404→`index.html`
   fallback" hazard goes away — the control dashboard's static fallback is scoped
   to `/` and cannot swallow `/session/<id>/*` or `/data_plane/*`. Still use a
   streaming ASGI passthrough, not a buffering fetch.
2. **Stale upstreams.** A dead session must expire from the registry (file removed
   on reap / TTL) so the front returns a clean "session ended", not a hang.
3. **Reap/registry consistency.** Tie registry cleanup to the shim's child-reap
   path (§4.4) so a force-killed child leaves no routing ghost.

### The downgraded `biopb mcp` CLI (the standalone, no-agent path)

**Superseded (decided during Layer 1).** The standalone no-agent path is now
`biopb mcp view` — a **foreground, blocking, Ctrl-C** viewer that opens the napari
window immediately (eager `host.ensure_started()`), binds a dynamic port, prints
its `/mcp` URL for optional agent attach, and writes no PID file. It fills the
role this section reserved for a repurposed `biopb mcp start`, so **the shared
background daemon is retired rather than reshaped**:

- `biopb mcp view` — foreground agentless viewer (shipped in Layer 1). *Not*
  deprecated; this is the standalone path going forward.
- `biopb mcp start` / `stop` / `status` / `restart` / `logs` — **deprecated**
  (each emits a runtime notice; `cli._warn_mcp_daemon_deprecated`). The stdio
  shim owns its own per-client session, so a shared PID-tracked daemon has little
  purpose. They still work; removal is a later step.
- The two config keys the daemon path touched (`mcp.transport.port`,
  `mcp.transport.kernel_log`) are **kept, not deprecated** — both still drive live
  non-daemon paths (a direct `--transport http` / the observe mount; the
  per-session-log "single shared file" override). Their comments note the reduced
  role.

**Decided 2026-07-11 — `biopb mcp view` does NOT register with the control.** L3
shipped the session registry + control-fronted `/session/<id>/observe` + the
dashboard's session list, so wiring `view` into them became *possible* — but it
buys `view` nothing. `view` is a foreground, in-process, **agentless** session:
with no agent driving it there is nothing to *observe* (the observe UI exists to
watch/steer an agent's `execute_code` history), and being a foreground process the
user Ctrl-C's directly there is no lifecycle to control remotely. So `view` stays
fully standalone — it prints its direct `/mcp` URL and works whether or not the
control is running; registration/observe/lifecycle-control remain **shim-only**
(the agent path). This keeps `view` a self-contained convenience with no dependency
on the control.

---

## 7. Layer 4 — Algorithm plane under control (config extraction) *(later)*

The largest and most separable piece — sequence it last so it cannot stall the
core.

- Move algorithm-server configuration **out of MCP** into the control's config.
- The control supervises algorithm-server lifecycles (subprocess / container) like
  the data plane, and sessions discover them through the control rather than
  carrying their own config.
- Ops (`ProcessImage` / `ObjectDetection`) remain wired into the kernel namespace
  as today; only *which servers exist and how they start* moves to the control.

### 7.1 Config: federate, don't merge *(decided)*

The config surfaces (data-plane `~/.config/biopb/biopb.json`, MCP
`~/.config/biopb-mcp/config.json`, and a future control config) stay **separate
files, each owned by one process** — they do **not** collapse into one
control-owned file. The governing rule:

> **One writer *domain* per file, where the domain is the process that validates
> that file's schema.** Whoever validates a file owns writing it.

Why merging is wrong:

- **The data-plane config is written *and validated* by the data plane itself.**
  `PUT /api/admin/config` (the dashboard's `/data_plane/viewer/admin`, blind-proxied
  through the control) runs the tensor server's full load-time validation
  (`build_config_schema` + `validate_config_dict`) **in the tensor-server process**
  before `save_config`, so "the form accepted it" == "the server will load it". The
  control **cannot** run that validation without importing `biopb_tensor_server` —
  the I2 line — so it cannot own writing `biopb.json`. Merging would either move
  tensor-schema validation into the control (I2 break) or put two writers on one
  file with split validation and write races.
- **Docker already runs the boundary.** The container runs the *control* as its
  single public origin (8813) with the tensor sidecar private-loopback behind it,
  and the control **blind-proxies** the admin config page to the sidecar without
  ever parsing the config. Control and the data-plane config surface coexist as
  separate ownership domains in the smallest real composition — the proof that
  "control present, data-plane config still data-plane-owned" is the steady state,
  not a compromise.
- **A control config must be optional.** In the container the control is configured
  entirely from **env/argv** (entrypoint derives ports; token from
  `BIOPB_TENSOR_TOKEN`) and writes no file. So any `control.json` is
  **env/argv-first, file-second** — an additive desktop surface, never a boot
  dependency.

Writer model:

| File | Owning domain (writes + validates) |
|---|---|
| `~/.config/biopb/biopb.json` | **data plane** — installer seeds; `biopb server migrate-config`; the **admin API** at runtime; the container entrypoint (env→file). The control never writes or parses it (blind proxy). |
| `control.json` *(new, optional)* | **control** — its own API is the sole writer; env/argv overrides and suffices. Holds the algorithm-server registry + session policy; empty/absent in the container. |
| `~/.config/biopb-mcp/config.json` | **MCP** — per-session kernel/dask/viewer/timeout knobs. Absent in the container. |

What actually moves in Layer 4 is **only the algorithm-server registry**, MCP →
control. It is the one piece with a real reason to move: today it is
installer-seeded in the MCP file with no runtime writer; under the control it gains
one — a native control-owned admin surface (add/remove a server from the dashboard)
that writes `control.json` and validates it *in the control*, exactly parallel to
the tensor admin page writing `biopb.json` in the tensor process. Two admin
surfaces, two files, two validators, one origin fronting both. Everything the
tensor server validates stays in `biopb.json`; everything per-session stays in the
MCP file.

Sequencing note: a **read-only** algorithm-plane inspector already shipped (the
control dashboard's Algorithm plane card + `biopb image servers`, both reading the
current MCP location through the shared `biopb._algorithms.servers_from_config`
seam). Because the control and the MCP kernel now source the list through that one
seam, the Layer-4 flip is localized: repoint the seam at `control.json` (with a
`_migrate_legacy_keys`-style read-old / warn / prefer-new shim), then add the
control's write path + supervision.

Open sub-decision: whether `biopb control start` reads `control.json` itself or
keeps resolving via the CLI and passing argv (as it does for `biopb.json`'s
host/port/token today). The env/argv-first rule allows either; letting the control
read the file avoids serializing a rich registry through flags.

---

## 8. Phasing

Each phase is independently shippable; the old shim keeps working until Layer 1
flips it. Suggested order (value-first):

1. **Control skeleton + data-plane supervision (Layer 2 core).** ✅ *Shipped.* Lean
   `biopb-control` package supervises the tensor server (spawn/adopt/restart);
   `_connection` gutted to a pure client that asks the control to ensure the plane.
   Breaks the bootstrap loop. (Session self-registration + the control-fronted
   observe route are deferred to Layer 3.)
2. **De-daemonize sessions + registry (Layer 1).** ✅ *Shipped* (before the Layer-2
   core, reordering the value-first list to fix #98/#403 first). Serve-core
   refactor, shim ownership, dynamic-port handoff. Closes #98 (env inheritance)
   and lands #403/#402 in the shim-owned model. Session self-registration with the
   control is the remaining registry piece, folded into Layer 3.
3. **Single-origin front (Layer 3).** Control serves the whole browser UI as one
   base-`/` SPA at its root (dashboard `/`, dataviewer `/viewer`, observe
   `/session/<id>/observe`), namespaces each plane under its prefix
   (`/data_plane/*` API-only, `/session/<id>/api/*`), routes per-session observe,
   and terminates auth/TLS. Downgrade `biopb mcp` to the standalone wrapper.
   *(Built: the ASGI origin, the control-served SPA bundle, the namespaced
   API-only `/data_plane/*` proxy, the session registry, per-session observe, and
   the control API — §6.1. Auth/TLS termination and the `biopb mcp` downgrade
   loose ends remain.)*
4. **Algorithm plane under control (Layer 4).** Config extraction + supervision.

---

## 9. Risks / open questions

- **Control as SPOF (supervision/routing only).** Keep it lean and crash-only;
  already-spawned planes must survive a control blip and re-register.
- **Resource multiplication.** N concurrent sessions = N dask `LocalCluster`s +
  N kernels + N viewers (was one shared). Consider a soft cap or a shared-cluster
  opt-in if it bites.
- **"Human starts viewer, agent joins it."** Default is spawn-your-own, so an
  agent does not attach to a `biopb mcp start` session. If wanted, add an opt-in
  (`BIOPB_MCP_CONNECT_URL`) that makes the shim bridge to an existing URL instead
  of spawning. Out of scope; door left open.
- **Startup latency moves to per-session** (no daemon to amortize the http-stack
  import). The heavy kernel/dask/napari start stays lazy (first `start_kernel`),
  so the shim-spawn cost is the import only. Acceptable for session-scoped clients.
- **Control's packaging home.** Subprocess-based (I2) keeps it decoupled, so it can
  live wherever the `biopb` launcher lives (core `biopb`) without depending on the
  unpublished tensor server or on `biopb-mcp`. Decide: extend the existing
  `biopb` CLI process, or a dedicated `biopb control` entrypoint.
- **What keeps the control alive** (systemd / launchd / Windows service vs. a
  detached process the installer manages) — the one place a persistent daemon is
  still wanted, now isolated to the lean control plane.
- **Config unification.** *Resolved (see §7.1): federate, don't merge.* Each
  plane's config stays a separate file owned by the process that validates it
  (data-plane `biopb.json` written by the tensor server / admin API; a new
  *optional* control-owned `control.json`; the MCP file for per-session knobs).
  Only the algorithm-server registry moves MCP → control. Merging is blocked by
  I2 — the control can't validate the tensor schema, and whoever validates a file
  owns writing it.
- **Transition / back-compat.** `biopb server` / `biopb mcp` semantics during the
  migration. *Resolved (Layer 2 core): the control does NOT adopt an already-running
  tensor server.* Supervising both spawned and adopted servers made the state
  fragile (a crashed child's handle could no longer be trusted to tell "ours" from
  "someone else's") and left teardown incomplete (`control stop` couldn't stop an
  adopted plane). The control is now the **sole owner**: it always spawns its own
  plane and treats a port already in use as a conflict to refuse. During migration
  the old standalone `biopb server start` daemon still exists, but you run *either*
  it *or* the control, not both against one port — the control will not attach to a
  server it did not start.

---

## 10. Relationship to existing docs

- `daemon-migration.md` — kept as the record of *why the shim/heavy split exists*
  (fd-1 safety). That split is retained; this doc undoes only the detachment and
  the daemon's role as lifecycle root.
- The earlier single-origin reverse-proxy sketch and the standalone
  de-daemonization note are **absorbed** here as Layers 3 and 1.
- `CLAUDE.md` §2.6 / §3 (kernel isolation) should be updated once Layer 1 lands to
  describe the durable-planes + ephemeral-shim-owned-sessions model.
