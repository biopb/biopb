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
  workspace package (own `control-v*` tag), managed by `biopb control
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
- **The tensor server's HTTP sidecar is now fronted by the control**, no longer
  meant as its own public origin: the Layer-3 front reverse-proxies the
  dataviewer + `/api/*` + `/ws/render` through 8813, forwarding the Bearer token
  *(done — Layer-3 origin step)*. Trimming the sidecar further to a minimal,
  control-only internal HTTP (dropping its public web role entirely) is a later
  refinement; the proxying that makes the data plane a private upstream is in
  place.

Result: the cycle in §1.1 becomes a tree rooted at the control.

---

## 6. Layer 3 — Single-origin web front (a facet of the control)

The control *is* the durable web origin, so single origin falls out — it is no
longer a reverse-proxy bolted onto the ephemeral sidecar.

> **Status: origin + data-plane proxy landed.** The control API is now a
> Starlette/uvicorn app on `127.0.0.1:8813` (replacing the stdlib
> `ThreadingHTTPServer`) that answers its own `/health` + `/data_plane/ensure`
> and reverse-proxies everything else to the supervised tensor server's HTTP
> sidecar — dataviewer static app, `/api/*`, health probes, and the `/ws/render`
> WebSocket — with the Bearer token forwarded through (`biopb_control/_control.py`).
> One origin now fronts the dataviewer. **Remaining:** session `/observe` routing
> — the filesystem session registry + per-session reverse proxy (needs the
> observe frontend's base-path fix, §3-gotcha-1).

**Routing:**

| Path | Target | Hop |
|---|---|---|
| `/`, control UI, dataviewer | control's own static app | in-process |
| `/api/slice`, data API | tensor server's internal HTTP | loopback proxy |
| `/observe`, observe API | the registered session's dynamic port | loopback proxy |
| `/session/<id>/*` | that session's port (multi-session) | loopback proxy |
| `/mcp` (agent JSON-RPC) | **not routed here** — shim → child, direct/private | — |

Keep the common case trivial: one active session → `/observe` → "the current
session"; `/session/<id>/` is the multi-session refinement.

**Session discovery:** a **filesystem registry** (recommended for localhost) —
each session writes `~/.local/share/biopb/sessions/<id>.json` (port + pid) on
startup, removes it on reap; the control reads/watches that dir. Matches the
existing sentinel/PID-file idioms; no new endpoint. (HTTP self-registration is
the alternative if the front and children may later cross a machine boundary.)

**Why it is worth more than one bookmark:**
- **Security win.** The observe UI currently fronts an `execute_code` RCE behind
  loopback + Host/Origin only. Behind the control origin, the tensor server's Bearer
  token gates it *uniformly*, and the loopback hop to the child is trusted. This
  is the surface you put TLS + authn in front of for the §1 "trusted intranet"
  hardening — **one origin to front, not N dynamic ports.**
- **No CORS** — same origin for dataviewer + observe → shared token/cookie, no
  cross-origin config, no per-session-port CORS churn.

**Gotchas (same class as the merge audit):**
1. **Streaming.** Observe uses SSE; the proxy must stream (no buffering) and be
   **exempt from any SPA 404→`index.html` fallback and the StaticFiles catch-all**
   — register proxied routes first; use a streaming ASGI passthrough, not a
   buffering fetch.
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

The eventual control-registration + control-fronted observe link remain future work
(Layers 2–3); `biopb mcp view` prints the direct `/mcp` URL in the meantime.

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
3. **Single-origin front (Layer 3).** Control routes `/observe`, absorbs the
   dataviewer + control UI, terminates auth/TLS. Downgrade `biopb mcp` to the
   standalone wrapper. *(Partial: the ASGI origin + the tensor-sidecar reverse
   proxy — dataviewer, `/api/*`, `/ws/render` — shipped; `/observe` + the session
   registry are the remaining sub-step.)*
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
- **Config unification.** Data-plane, algorithm-plane, and MCP config could
  collapse into one control-owned surface, or stay per-plane with the control
  referencing them. Decide during Layer 4.
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
