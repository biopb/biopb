# biopb-mcp

## Overview

`biopb-mcp` connects [napari](https://napari.org) and AI agents to
[biopb](https://github.com/biopb/biopb) servers. It has two faces:

1. **napari plugin** — a `Tensor Browser` widget that browses/loads images from a
   biopb tensor server (Arrow Flight), plus two **experimental demo widgets**
   (`Object Detection`, `Image Processing`) that call the `biopb.image`
   ProcessImage gRPC protocol. The demo widgets exist to test algorithm servers;
   they are **not** the primary interface.
2. **MCP server** — a process that exposes a live napari viewer to an AI agent
   over MCP. Thesis: *"agent first; provide tools only if they help."* The agent
   drives napari through a real Python kernel; image results go to the viewer,
   other results to the agent's chat.

Read [docs/napari.md](docs/napari.md) for napari's plugin architecture.

### Runtime shape

A biopb deployment is a **tree rooted at a durable control plane** — no cycles:

```
   control plane   (durable ROOT — lean: supervise + route + serve the web UI)
        ├── supervises ─► data plane      (tensor Flight server + HTTP sidecar)
        ├── supervises ─► algorithm plane (algorithm servers)          [pending]
        └── observes   ◄─ MCP sessions    (ephemeral, SHIM-owned; self-register)
                            env inherited from the shim  (the #98 fix)
                            USE the planes; never START them
```

| Component | Lifetime | Owned by | Role |
|---|---|---|---|
| Control plane | durable (root) | OS service / `biopb` launcher | supervise, route, single-origin web front, session registry, auth |
| Data plane | durable | control (subprocess) | pixels, cache, remote-data proxy |
| Algorithm plane | durable | control (subprocess) | compute ops *(pending)* |
| MCP session | ephemeral | the **shim** | kernel + dask + viewer; env-inherited; registers with control |
| Shim | per client connection | the MCP client | stdio↔http bridge; spawns & reaps its session child |

This shape resolves three problems the older shared-daemon model had:

1. **fd-1 corruption → the shim/heavy split.** Under stdio MCP, **fd 1 *is* the
   JSON-RPC channel**; any stray stdout from a heavy process (uvicorn/Qt/dask/
   kernel) corrupts it. So the client spawns a **featherweight shim** that owns
   fd 1 and imports only the mcp SDK, and all heavy work runs in a separate child
   it bridges to over http — making fd-1 corruption structurally impossible.
2. **A bootstrap cycle.** The session depends on the data plane, yet its data
   layer used to *start* it — a cycle with no clean owner. The control owns the
   plane; `_connection` becomes a pure client.
3. **N ephemeral web surfaces.** Each session's `observe` UI lives on a dynamic
   port; a single web origin needs an owner that discovers sessions dynamically.

Plus the daemon model's env/orphan bugs: **#98** (a login-time daemon freezes
`DISPLAY`, so the viewer lands on the wrong display) and **#403** (Windows kernel
orphans).

---

## Lifecycle

Two invariants keep the tree correct:

- **I1 — the control *observes* sessions, never *spawns* them.** A control-spawned
  session would inherit the control's frozen env, re-breaking #98. Sessions stay
  shim-owned and env-inherited; they only **register** so the control can route to
  and list them.
- **I2 — the control stays lean and subprocess-based.** It supervises components
  as subprocesses (`python -m biopb_tensor_server …`, `python -m biopb_mcp.mcp …`),
  never by importing them — no Qt/napari/dask/kernel enters it. Shared facts (the
  control endpoint, the session-file contract, auth predicates, the process/
  lifecycle helpers) live in **stdlib-only core-SDK modules** (`biopb._config_control`,
  `biopb._config_sessions`, `biopb._web_auth`, `biopb._proc`, `biopb._lifecycle`)
  that neither side imports from the other.

### Control plane (durable root)

A small always-on Starlette/uvicorn app (`biopb-control` package; `biopb control
start/stop/status/run`) on `127.0.0.1:8813` that **supervises the data plane**,
**is the single web origin**, and **holds the session registry**.

`DataPlaneSupervisor` spawns the tensor server, polls liveness (a stdlib TCP
connect — no pyarrow/grpc imported, I2), and restarts it on crash with capped
backoff. It is the **sole owner**: it always spawns its own child and **refuses a
port already held by a foreign process** (a *conflict*, surfaced not adopted) — so
`self._proc` is the whole state and `control stop` is a complete teardown, with no
"adopted, left running" case. `biopb control start` brings the plane up by default
(`--no-data-plane` runs the control alone).

**The plane is bound to the control's lifetime (Pattern O).** An orphaned plane
keeps holding the gRPC port, which the next control start reads as a conflict it
refuses — so a crashed/killed/logged-out control would wedge every restart (and
the installer's stop→start) behind a plane nobody owns. The supervisor closes that
by tying the plane's life to its own, using the shared `biopb._lifecycle`
primitives:

- **POSIX** — the child inherits a **parent-death pipe** (`deathwatch`) and runs
  in its own session; the tensor server's `launch`/`serve` call
  `deathwatch.install()`, which self-terminates the plane (a contained group-kill)
  when the control dies **uncatchably** (SIGKILL/OOM/crash/logout).
- **Windows** — the child is assigned to a **kill-on-close Job Object** (`winjob`)
  the control holds; the OS reaps the plane (and its descendants) when the
  control's last handle closes.

This is **orthogonal to the graceful stop** path — SIGTERM (POSIX) / a sentinel
file (Windows) still run the plane's orderly shutdown (releasing the file-cache
process lock) whenever the control is alive to ask; the bind is only the backstop
for when it is not. The cost is a brief data-serving gap across a control restart
(the plane comes back with it); keeping the control lean and crash-only-restartable
bounds that gap.

> The Windows **sentinel file** is the tensor server's alone. A process needs it
> only when all three hold: its stop is a raw signal (uncatchable on Windows), it
> has no in-band shutdown channel, and it holds costly durable state (the cache
> lock/WAL). The kernel has an in-band channel (jupyter ZMQ), the session child is
> ephemeral, and the dask workers self-terminate on scheduler loss — so none of
> them use it. See `biopb-tensor-server/ARCHITECTURE.md`.

### Shim-owned MCP sessions

The MCP server process **is http-only** (loopback streamable-http on
`transport.port`). `--transport stdio` (still the default, so installer-seeded
client configs keep working) no longer serves MCP from the launcher: it runs the
**shim** (`mcp/_shim.py`), which

1. **spawns its own ephemeral session child** — FastMCP/uvicorn + the kernel host
   — on a **dynamic OS-assigned port** (`--port 0`), reported back over an
   inherited pipe fd (`BIOPB_PORT_REPORT_FD`);
2. **bridges** stdio JSON-RPC ↔ that child's `/mcp` until the client closes stdin,
   replaying the child's initialize result **verbatim** (including `instructions`,
   the field the generic `mcp-proxy` drops — which is why the bridge is vendored);
   any bridge failure exits the shim so the client sees EOF, never a hung server;
3. **reaps** the child (and its kernel grandchild) on the way out.

There is **no probe-and-reuse, no shared daemon, no fixed port**: each stdio
client spawns and owns its own session, so N clients get N independent sessions (N
viewers), by design. The child:

- **inherits the shim's live environment** (`DISPLAY`/`XAUTHORITY`/`WAYLAND_DISPLAY`
  are the user's current session — the **#98 fix**), so the agent's viewer lands
  on the human's real display;
- **registers with the control** on startup and deregisters on reap;
- is **reaped as a tree** — POSIX via the shared process group + a parent-death
  pipe; Windows via a **Job Object** the shim creates and assigns (**#403**), so a
  force-killed shim takes the whole subtree down. A client-death watchdog covers a
  multi-process client that keeps the shim's stdin open past its own exit.

Native http (`claude mcp add --transport http biopb http://127.0.0.1:8765/mcp`)
skips the shim entirely and is preferred where the client supports it. The child's
output goes to a **per-session** log (`transport.kernel_log` empty by default →
`~/.local/state/biopb/mcp/sessions/<id>.log`; set it to force one shared file).

### On-demand kernel

The process owns a **single child Jupyter kernel** (real IPython kernel via
`jupyter_client`) that hosts the napari viewer, dask, and the tensor client. Agent
code runs *in that kernel*, not on the MCP thread or napari's Qt loop — so a
runaway execution can be interrupted (`SIGINT`) or hard-restarted (process-group
`SIGKILL` + respawn) without killing the MCP server. A single `RLock` serializes
access to the kernel.

The kernel is **launched lazily, not at boot**, so a long-running server binds
cheaply and never pops a viewer with nobody connected. The `start_kernel` tool
drives `KernelHost.ensure_started()` — **synchronous** (blocks until ready or
failure; FastMCP runs sync tool handlers on the event loop) and idempotent, and is
also the recovery path after a dead kernel. Until then kernel-dependent tools
return a structured not-ready status: `not_started` / `error` / `dead` (→ call
`start_kernel`) or `starting` (a watchdog respawn in flight).

**Closing the napari window tears the kernel back down to idle.** POSIX: the
kernel holds the write end of a `BIOPB_WINDOW_CLOSE_FD` pipe fired from the
window's `destroyed` signal, and a server reader thread calls `shutdown()` on the
byte. Windows (no `pass_fds`): the launcher **polls** the in-kernel
`_viewer_window_alive()` probe (default 2 s) and shuts down on a confirmed close —
acting only on a clean reading from a ready, idle kernel, so it never aborts a
running job (a close during a long job is caught on the next idle tick). Either
way a user-attributed `_teardown_reason` reaches the agent via `execute()` /
`server_status`. Rebuild with `start_kernel`.

### dask cluster

The **session child** (not the kernel) owns the dask `LocalCluster`
(`DaskClusterHost`), so it survives kernel restart/respawn/window-close — the
kernel attaches via an injected scheduler address, with no cold worker re-spawn
per restart (the dominant restart cost on Windows). Worker/memory changes need a
*session* restart, not just `restart_kernel`. On an uncatchable session-child
death the workers **self-terminate on scheduler loss** (which is why the `mcp`
extra floors `distributed>=2023.9`); on Windows they fall under the shim's Job
Object by nesting. Any distributed mode lets `cancel_job` stop an in-flight
`.compute()`.

### The standalone `biopb mcp view`

The no-agent path: a **foreground, blocking, Ctrl-C** viewer that opens napari
immediately, binds a dynamic port, prints its `/mcp` URL for optional agent
attach, and writes no PID file. It stays fully standalone — it does *not* register
with the control and works whether or not the control is running.

---

## Security model

**The kernel is a real IPython kernel with imports allowed — `execute_code` is
arbitrary code execution by design.** Do not describe it as sandboxed. The whole
system assumes a **localhost / trusted-intranet** deployment; untrusted-network
exposure is expected to be fronted by a separately-documented reverse proxy.

- **Loopback bind + DNS-rebinding allowlist.** Every server binds loopback only
  and enforces an `Origin`/`Host` allowlist (`build_transport_security()` in
  `_server.py`), so a malicious page in the user's browser is rejected (`403`/
  `421`) before it reaches the kernel. There is **no loopback token** — the
  `Origin` check already blocks the browser-attacker threat. Extend the allowlist
  via `config['transport']['allowed_origins']`/`allowed_hosts` for a reverse-proxy
  front (http only).
- **Web-origin auth** (`biopb._web_auth`, shared by control + sidecar + observe):
  when a data-plane token is configured it is required (`Bearer` / `X-Biopb-Token`);
  in local mode (all-loopback) a **loopback-Host** check is the DNS-rebinding
  backstop; and every state-changing verb refuses a forgeable cross-site request
  (CSRF). Because the `/session/<id>/*` proxy hop strips the child's own
  Host/Origin guard, this control-side check is the child's **only** auth.
- **The `/session/<id>` proxy is an allowlist, not a denylist** (its `/mcp` is an
  RCE sharing the child port). httpx normalizes dot-segments, so a denylist would
  let `api/../mcp` collapse onto `/mcp`; only a first path segment of `observe` or
  `api` is proxied, parent-traversal rejected.
- **Supervised restart is control-routed, not blind-proxied** (**#418**). The
  sidecar's `/api/admin/restart` self-restarts by spawning a detached process —
  correct standalone, but under supervision it would SIGTERM the control's tracked
  child and race the supervisor for the port. So the control marks its child
  (`BIOPB_DATA_PLANE_SUPERVISED`), the sidecar surfaces `supervised` and **refuses**
  self-restart (409), and the admin UI routes restart to `/api/data_plane/restart`.
  Config *edits* stay a blind proxy (the tensor process is the sole validator of
  `biopb.json`); only *restart* — an ownership action — is control-routed.

---

## Components

### Data connection (`_connection.py`)

`TensorConnection` is a **GUI-independent** data-access service (imports neither Qt
nor napari), so the `TensorBrowserWidget` and the headless MCP kernel share one
implementation. It owns the `biopb.tensor.TensorFlightClient`, the source catalog,
and URL/token resolution + persistence.

- **Connect policy** (`auto_connect`, shared by both faces) is **control-first**:
  it asks the **control** to ensure the data plane (one `ensure_data_plane()` POST
  — the single source of truth since #413 — which brings the plane up if down and
  returns the *authoritative* gRPC endpoint), and **only when no control answers**
  falls back to a direct connect on a locally-resolved `(url, token)`, waiting the
  server through a `STARTING` scan. `_connection` is a **pure client**: it never
  starts a server itself — that is the control's job. It is best-effort (failures
  recorded in `last_status`/`last_message`, never raised) and must be driven **off
  the caller's main thread** because `connect()` blocks on I/O (the kernel runs it
  on a daemon thread; the widget on a connect worker that signals the tree render
  back to the Qt main thread).
- **Fallback resolution** (`resolve_from_config`, used only on that no-control
  path — the control's endpoint wins whenever it answers): env
  (`BIOPB_TENSOR_URL`/`BIOPB_TENSOR_TOKEN`) → config (`tensor_browser.server_url`)
  → default `grpc://localhost:8815`. Only the URL is persisted; the token is read
  from the environment.
- **Self-healing catalog** (`start_source_watch`, **#44**): a catalog cached at
  connect can be *partial* (the server reports `SERVING` before it finishes
  enumerating scenes). A daemon thread `health_check()`s and re-lists on any
  `source_count` change, backing off exponentially between
  `tensor.health_poll_min/max_interval` while stable. It only ever *rebinds*
  `self.sources` to a fresh dict, so agent and widget see no torn reads (no lock).
  Both faces start it; idempotent. Known limit: a source gaining scenes (1→18
  tensors) doesn't bump `source_count`, so that specific partial isn't caught.

### MCP server module (`mcp/`)

Launch with the console script `biopb-mcp` or `python -m biopb_mcp.mcp` (`pip
install biopb-mcp[mcp]`). The launcher comes up **idle** — the heavy kernel/viewer
starts on demand (above).

- `__main__.py` / `__init__.py` — launcher; builds the `KernelHost` **idle**,
  injects the bootstrap via `IPKernelApp.exec_lines`, then runs FastMCP.
- `_server.py` — FastMCP server (`mcp = FastMCP("biopb-mcp")`); defines the
  tools/resources and dispatches them to the `KernelHost`. `run()` serves
  streamable-http on `127.0.0.1:<port>/mcp` — the only serving path here.
- `_shim.py` — the stdio↔http bridge: `spawn_session` (dynamic-port child, env
  inheritance, port pipe, tree reap) + the vendored bridge (see Lifecycle).
- `_kernel.py` — `KernelHost`: the one child kernel. The `RLock` is held only for
  *quick* snippets (submit/poll/screenshot/status), never during long compute (it
  runs in a detached kernel thread). `execute()` has an `execute_timeout` (bounds
  quick snippets, SIGINT on timeout) and a `busy_lock_timeout` (`"busy"` if another
  quick call holds the lock). `ensure_started()` is the synchronous idempotent
  on-demand start; `restart()` releases dask then group-kills and respawns. Wires
  the window-close teardown (POSIX pipe reader / Windows probe poll).
- `_bootstrap.py` — runs **inside the kernel** via `exec_lines`. Enables `%gui qt`,
  configures dask, builds the `TensorConnection`, opens the viewer + Tensor
  Browser, builds `ops`, installs the job runner and the viewer proxy, wires the
  window-close hook, starts the source watcher (#44, also for headless), populates
  the namespace, then loads **user plugins** into it (below). On failure prints a
  `BOOTSTRAP_ERROR` sentinel; the host detects success via `'viewer' in dir()`.
- `_jobs.py` — runs **inside the kernel**. The async job runner: `execute_code`
  runs agent code in a **background daemon thread**, so the kernel main thread (and
  its `%gui qt` loop) stays free to service `take_screenshot`/`server_status`/
  `poll_job` mid-job. It waits up to `promote_after` seconds, then returns a job
  handle (`job-N`) and keeps running. One job at a time. The stop ladder:
  `cancel`/`cancel_current` is cooperative (a flag polled via `cancelled()`) and
  cancels in-flight distributed futures; `interrupt_current` additionally raises
  `KeyboardInterrupt` into the worker thread via `PyThreadState_SetAsyncExc` (a
  `SIGINT` reaches only the main thread). A `reason=…` threads a human-readable
  `cancel_reason` into the record so a user-triggered stop surfaces to the agent.
- `_viewer_proxy.py` — runs **inside the kernel**. The agent-facing `viewer` is a
  transparent **main-thread marshaling proxy** over the real `napari.Viewer`:
  because job code runs off-main but napari/Qt are main-thread-only, any off-main
  mutation can segfault the kernel (**#100**). It marshals mutations/calls to the
  Qt main thread, **re-wraps** every napari handle it returns (so
  `viewer.layers[0]`, `viewer.dims`, … never leak unwrapped), passes inert values
  through, and fail-loud-guards raw-Qt objects. Completeness is enforced by a
  graph-walk test; see `docs/viewer-thread-safety.md`.
- `_observe.py` — runs **in the MCP server process** (not the kernel). The web
  "observe" UI (`observe.enabled`, on by default): job history + global
  cancel/interrupt/restart. **http transport only** — the routes mount on the
  FastMCP app via `mcp.custom_route` (same loop/port as `/mcp`); under stdio a
  second uvicorn server would risk the fd-1 channel and race the kernel, so it is
  skipped. Custom routes carry their own Host/Origin guard.
- `_process_ops.py` — `build_ops()`: a thin `Run()` callable per configured
  ProcessImage server (discovered via `GetOpNames`), exposed as `ops`.
- `_resources.py` — resource content strings. `_helpers.py` — `add_tensor()`
  viewer patch + `viewer_window_alive()` closed-window probe.

**Tools (10):** `find_skills`, `start_kernel`, `take_screenshot`, `execute_code`,
`poll_job`, `cancel_job`, `inspect_object`, `interrupt_kernel`, `restart_kernel`,
`server_status`.

**Resources (6):** `guide://kernel`, `guide://viewer`, `guide://tensor`,
`guide://annotations`, `guide://ops`, and the `skill://{skill_id}` template.

**`execute_code` namespace** (populated by `_bootstrap.py`): `viewer` (live
`napari.Viewer`), `np`/`da` (numpy / dask.array), `client` (`TensorFlightClient`
or `None`, refreshed per-call), `ops` (`dict[str, callable]` of ProcessImage ops,
possibly empty).

**User plugins — the low-friction "bring your own tool" path (#92).** Beyond the
built-in handles, `_bootstrap._load_namespace_plugins` (gated by
`services.namespace_enabled`, on by default) loads two extra sources into the same
namespace, so a lab adds capability by *putting objects in scope*, not by extending
a protocol:

- **`~/.config/biopb/kernel/*.py`** — `exec`'d directly in the namespace (IPython
  `startup/` semantics): a file's top-level defs land beside `viewer`/`client`/
  `ops`, and its functions resolve those live handles as globals **at call time**,
  so a `client` refreshed per-job is seen — exactly like agent code. The lowest-
  friction path: drop a file, no packaging. **biopb-mcp ships its built-in example
  here**: the source lives in the wheel (`biopb_mcp/plugins/`, importable +
  unit-tested), and the installer *seeds a copy* into this dir via the
  `biopb-mcp-seed-plugins` console script (`plugins/_seed.py`) — idempotent,
  never clobbering a user edit. Delivering it as a *file the user can see and
  edit* (not a buried module) is the point of goal (1) "as an example," and the
  startup-file load is robust to the kernel interpreter's metadata view (the
  `python3` kernelspec need not be the tool env, so an entry point could be
  invisible). The example, `rolling_ball.py`, exports `subtract_background` /
  `rolling_ball_background` — a fast (shrink-then-roll, ~75× over
  `skimage.restoration.rolling_ball` at radius 50) port of ImageJ's rolling-ball
  background subtraction — filling a real algorithm gap (a segmentation-grade
  background estimator the agent can't reproduce accurately in ad-hoc numpy). Its
  imports/helpers are privately aliased so the startup exec contributes only the
  two public callables; the seeded `__init__.py` (skipped by the loader — leading
  `_`) documents the dir.
- **`biopb_mcp.namespace` entry points** — the distribution path (a lab publishes a
  plugin package). An entry point resolves to a `register(namespace)` hook (reads
  the live handles, adds names) or a module/mapping whose public names (honoring
  `__all__`, dropping imported modules) are merged. biopb-mcp registers none of its
  own (its built-in ships via the kernel-dir seed above); this path is for
  third-party packages.

Both are **fail-open per unit** (one bad plugin logs and is skipped, never aborts
the bootstrap — the `build_ops`/skills precedent) and pass a **reserved-name
guard**: a plugin overwriting a load-bearing handle (`viewer`/`client`/`np`/`da`/
`ops`/…) is restored-or-skipped with a warning (the two attach-thread-owned names,
`_dask_client`/`_dask_attach_done`, are left untouched to avoid racing that
thread). There is **no generated enumeration** — the `guide://kernel` resource
tells the agent plugins may exist and where they came from, and the agent
discovers them with `dir()` / `inspect_object` (the docstring *is* the doc, so code
and doc can't drift). The control dashboard's algorithm-plane panel shows a
**static** listing of the files + installed packages (read never executed,
invariant I2; via `biopb._kernel_plugins`), not the live namespace.

### Configuration (`_config.py`)

The config lives at `~/.config/biopb/mcp-config.json` — co-located with the tensor
server's `biopb.json` (via `biopb._config_location.mcp_config_path`). It is
**deep-merged** with `DEFAULT_CONFIG` on load, so a partial nested section
overrides only its own leaves. Read values with `get_setting(config,
"dotted.path")`, which falls back to `DEFAULT_CONFIG` so call sites never restate a
default. Sections are **flat / top-level** (each maps 1:1 to a section dataclass).

- **Demo-widget / shared knobs:** `widget` (`server_url`, `is_3d`), `detection`,
  `grid`; `tensor_browser.server_url` (data-plane URL, top-level so the headless
  `TensorConnection` reads it too), `pyramid`, `timeout`, `grpc.max_message_size_mb`,
  `memory`.
- **`transport`** — `kind` (`stdio`/`http`, default `stdio`), `port` (8765),
  `display_mode`, `kernel_log` (stdio-only, empty by default → per-session log),
  `session_log_keep`, `allowed_origins`/`allowed_hosts` (http only),
  `server_start_timeout`.
- **`kernel`** — `name`, `startup_timeout`, `execute_timeout` (120s; quick
  snippets only), `busy_lock_timeout` (5s), `promote_after` (10s),
  `parent_death_pipe`, `watchdog_*` (orphan-hardening, #13).
- **`dask`** — `scheduler` (`distributed` default → auto-spun multi-process
  `LocalCluster`; `threads`/`synchronous` for in-process, no mid-compute cancel),
  `address` (attach to an external scheduler), `num_workers`/`threads_per_worker`/
  `memory_limit`/`dashboard_address` (dashboard loopback only), `cache_budget`
  (cluster-wide chunk cache, split across workers). The session child owns the
  cluster (see Lifecycle).
- **`tensor`** — `health_poll_min/max_interval` (the #44 source watcher's backoff;
  min ≤ 0 disables). (The localhost client-cache decision lives in the tensor
  client — `_resolve_cache_bytes`, off by default, `BIOPB_CACHE_LOCAL=1` to opt in.)
- **`viewer`** — `compute_scheduler` (default `threads`; pins the viewer's *serial*
  slice reads to a single-process scheduler so they share one main-process chunk
  cache instead of scattering across per-worker caches — **#8**; `_viewer_compute.
  wrap_levels`). `async_slicing` (default `True`; fetches slices off the Qt main
  thread via `NAPARI_ASYNC`, so a zoom into a cold pyramid level keeps the current
  texture instead of freezing; `take_screenshot` force-syncs first).
- **`services.process_image_servers`** — the ProcessImage URLs exposed as `ops`;
  `services.skills_*` (the skills catalog); `services.namespace_enabled` (load user
  kernel plugins, #92 — see the `execute_code` namespace above).
- **`observe`** — `enabled` (on by default, http only), `max_output_chars`,
  `poll_interval_ms`.

### Plugin entry points

Defined in `napari.yaml` (registered via the `napari.manifest` entry point):
- `Tensor Browser` → `biopb_mcp.tensor_browser:TensorBrowserWidget`
- `Object Detection` → `biopb_mcp.image_processing:ObjectDetectionWidget`
- `Image Processing` → `biopb_mcp.image_processing:ImageProcessingWidget`

The console script `biopb-mcp` (`[project.scripts]`) launches the MCP server. The
`image_processing` demo widgets use napari's `@thread_worker` (yield progress/
results; connect `.yielded`/`.finished`/`.errored`; cancel via `.quit()`).

---

## Misc

### Single-origin web front

The control serves **one base-`/` SPA** and gives every plane its own path prefix,
so the three `/api/*` namespaces that used to collide at the root never meet:

| Path | Target | Hop |
|---|---|---|
| `/`, `/viewer`, `/admin`, `/assets/*` | control-served `web/` SPA | in-process |
| `/api/*` | control's own API (`status`, `sessions`, `data_plane/{ensure,stop,restart}`) | in-process |
| `/health` | bare liveness (installer / `_control_client`) | in-process |
| `/data_plane/api/*`, `/data_plane/ws/render` | tensor sidecar (API-only) | loopback proxy |
| `/session/<id>/observe` | control-served SPA observe shell | in-process |
| `/session/<id>/api/*` | that session's observe API | loopback proxy |
| `/mcp` | agent JSON-RPC — **not routed here**; shim → child, direct | — |

The SPA is built with **base `/`** so `/assets/*` resolve from the root under any
shell prefix. `/data_plane/*` is a pure prefix-stripping proxy into the (API-only)
sidecar; control *verbs about* the plane live under `/api/data_plane/*`, so proxy
and verbs never mix. Observe uses **SSE**, so the proxy is a streaming ASGI
passthrough; explicit prefix `Mount`s (no root catch-all) keep the static
`/`-fallback from swallowing `/session/<id>/*` or `/data_plane/*`.

**Session registry.** Each session writes `~/.local/state/biopb/sessions/<id>.json`
(host + port + pid + `/mcp` url) once reachable and removes it on reap; the control
reads that dir. The contract is stdlib-only `biopb._config_sessions` (shim writes,
control reads). `resolve()`/`list_sessions()` **self-heal** by pruning records
whose owning pid is dead — or alive on a recycled pid (create-time token mismatch,
the #138 PID-reuse guard, via `biopb._proc`) — so a dead session expires to a clean
"session ended" rather than a hang.

### Error-propagation model

Letting an exception propagate *upward* — including out of a Qt callback — is
**safe in both hosts and will not qFatal-abort the process**, so prefer surfacing
real failures over swallowing them. PyQt6 only calls `qFatal()`/`abort()` on an
unhandled slot exception when `sys.excepthook` is the *default* one, and a
non-default hook is always installed: napari's `notification_manager` in the plugin
(via `napari.run()`), ipykernel's `ZMQInteractiveShell.excepthook` in the MCP
kernel (where `napari.run()` is never called — the `%gui qt` inputhook drives the
loop). The only difference is *where* the error surfaces — a GUI notification vs. a
printed traceback in the per-session kernel log. Caveat: this is crash-safety only;
a propagated error in the kernel produces a log traceback, not an agent-facing
message, so still catch where you want a clean, reported failure.

### macOS CI testing

macOS headless CI cannot initialize OpenGL, so napari-viewer creation segfaults.
Skip such tests on macOS CI (apply at class level for viewer-using classes):

```python
@pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("CI") == "true",
    reason="OpenGL context unavailable on macOS CI headless environment",
)
```

See `test_widget.py` and `test_progress_timer.py`. Headless testing elsewhere uses
`QT_QPA_PLATFORM=offscreen`; the QT API is pinned to pyqt6 in the pytest config.

### Project structure

- `src/biopb_mcp/`
  - `_connection.py` — GUI-independent `TensorConnection` (incl. `start_source_watch`, #44)
  - `_config.py` — configuration; `_tensor_utils.py` — pyramid/dimension utils
  - `_viewer_compute.py` — `wrap_levels`/`_ViewerArray` (single-process viewer reads, #8)
  - `_typing.py`, `_utils.py`, `_version.py`, `napari.yaml`
  - `tensor_browser/` — `_widget.py` (`TensorBrowserWidget`)
  - `image_processing/` — demo widgets + `biopb.image` gRPC (`_grpc.py`, `_chunking.py`, `_render.py`, …)
  - `plugins/` — built-in example kernel plugin (`rolling_ball.py`) + `_seed.py`
    (installer seeds it into `~/.config/biopb/kernel/`) + `__init__.py` namespace doc (#92)
  - `mcp/` — MCP server module (optional `[mcp]` extra); see the file list above
  - `_tests/`

### Dependencies

Main: numpy, pandas, magicgui, qtpy, scikit-image, scipy (rolling-ball plugin),
`biopb[tensor] >= 0.5.4` (Arrow Flight client), opencv-python-headless,
grpcio-tools, grpcio-health-checking, vedo (3D).
MCP extra (`[mcp]`): `mcp >= 1.20`, `uvicorn >= 0.29`, `jupyter_client`,
`ipykernel`, `psutil`, `distributed >= 2023.9`, `napari[all]`, `pyqt6`.
Testing: pytest(-cov/-qt/-env), napari, pyqt6, jupyter_client, ipykernel,
napari-skimage-regionprops. Versioning via setuptools_scm.

### Code anchors

- `mcp/_shim.py` — session spawn/bridge/reap. `mcp/{_server,_kernel}.py` — FastMCP
  app + `KernelHost`. `mcp/_observe.py` — per-session observe UI.
- `_connection.py` — the pure client (asks the control to ensure the plane).
- `biopb-control/src/biopb_control/` — `_control.py` (ASGI origin: SPA + plane
  proxies), `_supervisor.py` (`DataPlaneSupervisor`).
- `biopb/{_config_control,_config_sessions,_web_auth,_proc,_lifecycle}` — the
  stdlib-only cross-process seams.
