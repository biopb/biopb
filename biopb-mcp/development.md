# biopb-mcp

## Python Environment

This project uses a **`uv`-managed virtualenv at `.venv/`**, pinned by `uv.lock`.
Always run tooling through that interpreter — e.g. `.venv/bin/python -m pytest …`,
`.venv/bin/python -m black …` — **not** the bare `python`/`pytest` on `PATH`. A
system or user-site Python can shadow the project env with a *different* (often
older) `biopb`, which silently breaks tests that depend on newer client APIs
(e.g. `make_cache_plugin`, added in `biopb >= 0.5.8`).

biopb-mcp now lives in the **biopb monorepo** (`biopb/biopb-mcp/`) as a uv
workspace member; see `docs/monorepo-migration.md`. `biopb` and
`biopb-tensor-server` resolve to the sibling workspace members, so run uv from
the **repo root**, selecting this package:

```bash
uv sync --package biopb-mcp --extra mcp --group testing
```

After pulling changes, re-run that to bring `.venv` back in line with the root
`uv.lock`. If imports or versions look wrong, first check the interpreter
(`which python` / `.venv/bin/python -c "import biopb; print(biopb.__version__)"`)
before debugging the code; a stale `.venv` (missing `uv sync`) is the usual cause.

Test/dev deps live in **PEP 735 dependency *groups*, not extras**, so they stay
out of biopb-mcp's published metadata. Two groups:

- **`integration`** — the runnable full stack with *no* test tooling: the MCP
  server deps plus the in-tree `biopb-tensor-server[web]` and `biopb[tensor]`
  workspace members. Sync this for a checkout that needs a real local tensor
  server (`biopb server start` / autostart) without pytest:
  `uv sync --package biopb-mcp --extra mcp --group integration`.
- **`testing`** — `integration` **plus** pytest. The normal dev/CI profile.

Because these are groups, select them with `--group <name>`, *not*
`--extra <name>`. Version-pairing across the three packages is automatic from
the checkout (the workspace resolves them from the tree) — there are no more
release-asset URLs to bump.

**Important — groups don't reach PyPI users.** A dependency group is not part of
biopb-mcp's published wheel/sdist metadata, and the workspace resolution applies
only to a source checkout. So `pip install biopb-mcp[...]` can never pull the
tensor server (it is not on PyPI); the groups + workspace only help a checkout.

## Build and Test Commands

```bash
# Install the package in development mode (uv — run from the repo root)
uv sync --package biopb-mcp --extra mcp --group testing

# Run all tests with coverage
uv run pytest -v --cov=biopb_mcp --cov-report=xml biopb-mcp/src/biopb_mcp/_tests

# Run a single test file
uv run pytest biopb-mcp/src/biopb_mcp/_tests/test_mcp_kernel.py -v

# Run a single test function
uv run pytest biopb-mcp/src/biopb_mcp/_tests/test_grpc.py::test_encode_image -v

# Format code with black (line-length 79)
black biopb-mcp/src/

# Run pre-commit hooks on all files
pre-commit run --all-files
```

## Architecture Overview

`biopb-mcp` connects [napari](https://napari.org) and AI agents to
[biopb](https://github.com/biopb/biopb) servers. It has two faces:

1. **napari plugin** — a `Tensor Browser` widget that browses/loads images from
   a biopb tensor server (Arrow Flight), plus two **experimental demo widgets**,
   `Object Detection` and `Image Processing`, that call the `biopb.image`
   ProcessImage gRPC protocol. The demo widgets exist to test algorithm servers;
   they are not the primary interface.
2. **MCP server** — a standalone process that exposes a live napari viewer to an
   AI agent over MCP. The project thesis is *"agent first;
   provide tools only if they help."* The agent drives napari through a real
   Python kernel; image results go to the viewer, other results to the agent's
   chat.

Read [napari.md](docs/napari.md) for a summary of napari's architecture and plugin system.

### Data connection (`_connection.py`)

`TensorConnection` is a **GUI-independent** data-access service (imports neither
Qt nor napari). It owns the `biopb.tensor.TensorFlightClient`, the source
catalog, and URL/token resolution + persistence. Both the `TensorBrowserWidget`
and the MCP kernel *consume* this service rather than owning a client.

URL/token resolution fallback chain (`resolve_from_config`):

1. **Environment variables**: `BIOPB_TENSOR_URL`, `BIOPB_TENSOR_TOKEN`
2. **Config file**: `tensor_browser.server_url` in the config
3. **Default**: `grpc://localhost:8815`

**Connect policy (`auto_connect`).** `auto_connect()` is the single connect
policy shared by **both** faces: try the resolved `(url, token)`, wait the
server through a `STARTING` data-folder scan, and — with no prompt — run
`start_local_server()` (`biopb server start`) as a last resort when the URL is
local *and* the `biopb` CLI is on PATH. It is best-effort (failures are recorded
in `last_status`/`last_message`, never raised) and **must be driven off the
caller's main thread** because `connect()` blocks on network I/O: the MCP kernel
runs it on a daemon thread (`_bootstrap`'s headless branch), the
`TensorBrowserWidget` on a connect worker that signals the tree render back to
the Qt main thread (`_start_connect` → `_connect_done`). This replaced the
widget's old QTimer poll chain **and** its modal autostart dialog — a modal on
the kernel's Qt loop could wedge the synchronous `start_kernel` (which blocks on
a kernel-main-thread readiness probe) until the user clicked. Only the URL is
persisted; the token is read from the environment, not saved.

**Self-healing catalog (`start_source_watch`, issue #44).** The catalog cached
at connect time can be *partial* — the server reports `SERVING` (port bound)
before it finishes enumerating scenes, so a mid-index `list_sources()` looks
complete but isn't. `start_source_watch()` spawns a **daemon thread** (not a
`QTimer` — it must run headless with no Qt loop) that periodically
`health_check()`s and re-lists (`refresh()`) whenever the server's
`source_count` changes, reconciling against the count cached at connect on its
first poll. The poll interval backs off exponentially from
`mcp.tensor.health_poll_min_interval` to `health_poll_max_interval` while stable
and snaps back to the min on a change. The watcher only ever *rebinds*
`self.sources` to a fresh dict, so the agent (which reads `_conn.sources` live)
and the widget (which wires `on_sources_changed` to a queued Qt signal to
rebuild its tree) see no torn reads — no lock. Both the kernel bootstrap and the
widget start it; the call is idempotent. Known limit: `source_count` doesn't
grow when an existing source gains scenes (1→18 tensors), so that specific
partial isn't caught by count alone — the common "sources still being
discovered" case is.

### MCP Server Module (`mcp/`)

The MCP server **is its own process** — it does *not* auto-start on plugin
import. Launch it with the console script `biopb-mcp` or `python -m
biopb_mcp.mcp`. Install the optional deps with `pip install biopb-mcp[mcp]`.
The launcher comes up **idle**: the heavy kernel (and, with a display, the
napari viewer window) is **not** spawned at boot — it starts on demand when an
agent calls the `start_kernel` tool (see the kernel-lifecycle note below).

**Transport.** The MCP server itself is **http-only** (loopback
streamable-http on `transport.port`; daemon migration Direction 1,
docs/daemon-migration.md). `--transport` / `config['mcp']['transport']['kind']`
still accepts `stdio` (deprecated; still the default so installer-seeded
client configs keep working unchanged), but stdio no longer serves MCP from
the launcher process: it runs the **shim** (`mcp/_shim.py`) — a featherweight
bridge that probes `127.0.0.1:<port>`, spawns the http daemon detached if
nothing is listening (daemon + kernel output goes to `transport.kernel_log`,
default `~/.local/share/biopb-mcp/log/mcp-server.log`), and pumps stdio JSON-RPC
to `/mcp` until the client closes stdin. Concurrent shims race benignly (the
port bind picks one daemon; losers exit on EADDRINUSE and every shim converges
on the winner). The shim process owns fd 1 as a protocol channel but imports
only the mcp SDK — no Qt/dask/uvicorn — so the old fd-1 corruption class is
structurally impossible; it replays the daemon's initialize result verbatim
(including `instructions` — the field the generic `mcp-proxy` bridge drops;
see docs/mcp-proxy-vet.md for why the bridge is vendored), and any bridge
failure exits the shim so the client sees EOF, never a hung server. Notable
behavior shifts vs. the old direct-stdio serving: the daemon (and kernel)
**outlives the client** and is **shared across concurrent clients** — one
kernel, one viewer, the project thesis made literal — and the Host/Origin
allowlist plus the observe UI now apply to stdio-bridged sessions too, since
everything is the one http server. Native http
(`claude mcp add --transport http biopb http://127.0.0.1:8765/mcp`) skips the
shim entirely and is preferred where the client supports it.

The process owns a **single child Jupyter kernel** (real IPython kernel via
`jupyter_client`) that hosts the napari viewer, dask, and the tensor client.
Agent code runs *in that kernel*, not on the MCP server's thread or napari's Qt
loop — so a runaway execution can be interrupted (`SIGINT`) or hard-restarted
(process-group `SIGKILL` + respawn) without killing the MCP server.

**On-demand kernel lifecycle.** The kernel is launched lazily, not at boot, so a
long-running server binds cheaply and never pops a viewer (or Qt-aborts on a
display-less host) with nobody connected. The `start_kernel` tool drives
`KernelHost.ensure_started()` — **synchronous** (it blocks until the kernel is
ready or the bring-up fails, like `restart_kernel`/`restart()`; FastMCP runs sync
tool handlers on the event loop, so both lifecycle tools block) and idempotent;
it is also the recovery path after a failed/dead kernel. Until then the host is
idle and the kernel-dependent tools funnel through `KernelHost.execute()`, which
returns a structured not-ready status — `not_started` (idle → call
`start_kernel`), `error`/`dead` (call `start_kernel` to retry), or `starting` (a
watchdog respawn in flight, derived from `is_alive() && !ready`). Conversely,
**closing the napari window tears the kernel back down to idle**: on POSIX a
reverse of the parent-death pipe (the kernel holds the write end of a
`BIOPB_WINDOW_CLOSE_FD` pipe and the bootstrap fires it from the window's
`destroyed` signal; a server reader thread calls `shutdown()` on the byte). On
**Windows** the inherited fd isn't available (no `subprocess` `pass_fds`), so the
launcher instead **polls** the in-kernel `_viewer_window_alive()` probe on a
thread (`_poll_window_close`, default every 2 s) and calls the same `shutdown()`
on a confirmed close. The poll only acts on a clean "window gone" reading from a
ready, idle kernel — a busy kernel (a job holds the lock), a not-ready kernel, or
an in-flight stop are skipped — so it never aborts a running job; the trade-off
vs. the POSIX byte is that a close during a long job is detected on the next idle
tick rather than immediately. Either way a user-attributed `_teardown_reason` is
surfaced via `execute()` / `server_status` so the agent learns *why* a running
job vanished. Rebuild with `start_kernel`.

**Async job model (`_jobs.py`):** `execute_code` runs agent code in a
**background daemon thread inside the kernel**, so the kernel main thread (and
its `%gui qt` Qt loop) stays free to service `take_screenshot` / `server_status`
/ `poll_job` mid-job — the agent is not blind during long work. `execute_code`
waits up to `promote_after` seconds; if the code finishes it returns the result
inline, otherwise it returns a job handle (`job-N`) and keeps running. One job
at a time. Because the viewer has Qt main-thread affinity, GUI mutations from
the worker thread are marshaled to the main thread: `_bootstrap` wraps
`add_tensor` + the `add_*` family, and `run_on_main(fn)` is exposed for the
rest; `cancelled()` supports cooperative `cancel_job`. Rich IPython `display()`
output is not captured (code is exec'd, not run via `run_cell`).

**Files:**
- `__main__.py` / `__init__.py` — launcher; builds the `KernelHost` **idle**
  (does not eager-start it), injects the bootstrap via `IPKernelApp.exec_lines`,
  then runs the FastMCP server. The kernel is brought up later by the
  `start_kernel` tool.
- `_server.py` — FastMCP server (`mcp = FastMCP("biopb-mcp")`); defines the
  tools/resources and dispatches them to the `KernelHost`. `run()` serves
  streamable-http on `127.0.0.1:<port>/mcp` — the only serving path in this
  process.
- `_shim.py` — the stdio bridge (`--transport stdio`): ensures the http
  daemon is listening (spawning it detached if not) and pumps stdio JSON-RPC
  to `/mcp`, replaying the daemon's initialize result verbatim.
- `_kernel.py` — `KernelHost`: manages the one child kernel. A single
  `threading.RLock` serializes access to the kernel shell channel; with the job
  model it is held only for the *quick* snippets (submit/poll/screenshot/status)
  — never during the long compute, which runs in a detached kernel thread.
  `execute()` has an `execute_timeout` (now bounds only those quick snippets,
  SIGINT on timeout) and a `busy_lock_timeout` (returns `"busy"` if another
  quick call holds the lock). `ensure_started()` is the synchronous, idempotent
  on-demand start (no eager boot); `restart()` releases dask then group-kills and
  respawns (re-running bootstrap, which clears job state). `_watch_window_close`
  (POSIX: a reader thread on the inherited window-close pipe) / `_poll_window_close`
  (Windows: a thread polling the `_viewer_window_alive()` probe) tears the kernel
  down to idle when the user closes the viewer, recording a `_teardown_reason`.
- `_bootstrap.py` — runs *inside* the kernel via `exec_lines`. Enables `%gui qt`,
  configures dask, constructs the `TensorConnection`, opens the napari viewer +
  Tensor Browser, builds `ops`, installs the job runner (`_jobs.install`) and
  the agent-facing viewer proxy (`_viewer_proxy.make_viewer_proxy`, see below),
  wires the window-close hook (the viewer's
  `destroyed` signal → a byte on `BIOPB_WINDOW_CLOSE_FD`), starts the background
  source watcher (`conn.start_source_watch`, issue #44 — covers headless, where
  there is no widget to start it), and populates the `execute_code` namespace.
  On failure it prints a `BOOTSTRAP_ERROR` sentinel; the host's health probe
  detects success via `'viewer' in dir()`.
- `_jobs.py` — runs *inside* the kernel. The async job runner: a `_jobs`
  registry, `submit`/`poll`/`cancel`/`cancel_current`/`interrupt_current`/
  `jobs_summary`, a thread-aware stdout dispatcher, `run_on_main` (Qt main-thread
  marshaling with proper exception propagation), and `cancelled()`. The stop
  ladder: `cancel`/`cancel_current` is
  cooperative (sets a flag the code polls via `cancelled()`) and also cancels
  in-flight distributed-dask futures; `interrupt_current` does that *plus* raises
  a `KeyboardInterrupt` directly into the job's worker thread via
  `PyThreadState_SetAsyncExc` (a `SIGINT` only reaches the kernel main thread, not
  the worker, so it can't stop a background job), forcing an uncooperative
  pure-Python loop to stop short of `restart_kernel`. A `reason=...` on either
  threads a human-readable `cancel_reason` into the job record (prefixed onto the
  finalized `error_text`), so a stop triggered by a *user* in the observe UI
  surfaces to the agent via `poll_job` instead of an unexplained cancellation.
- `_viewer_proxy.py` — runs *inside* the kernel. The agent-facing `viewer` in the
  `execute_code` namespace is a **transparent main-thread marshaling proxy** over
  the real `napari.Viewer`. Because job code runs on a background thread but
  napari/Qt are main-thread-only, any off-main viewer mutation can segfault the
  whole kernel (biopb/biopb#100). The proxy marshals mutations (`__setattr__`)
  and method calls to the Qt main thread via `_jobs.run_on_main`, **re-wraps**
  every napari handle it returns (so `viewer.layers`, `viewer.dims`,
  `viewer.layers[0]`, … never leak unwrapped — the gap the old `add_*`-only wrap
  left), passes inert values (arrays/scalars) through, and **fail-loud**-guards
  raw-Qt objects (`viewer.window`) so off-main access raises `ViewerThreadError`
  instead of crashing. Handle-set completeness is enforced by a graph-walk test
  (`_tests/test_viewer_proxy.py`); see `docs/viewer-thread-safety.md`.
- `_observe.py` — runs *in the MCP server process* (not the kernel). The web
  "observe" UI (`mcp.observe.enabled`, **on by default / opt-out**): job history
  (the submitted code + truncated output) + global cancel/interrupt/restart
  knobs. **http transport only** — the routes mount on the existing FastMCP app
  via `mcp.custom_route`
  (same loop and port as `/mcp`). It is deliberately *not* supported under stdio:
  a second uvicorn server in the stdio launcher process would risk the fd-1
  JSON-RPC channel (uvicorn attaches a stdout log handler) and a second
  event-loop thread driving the one `KernelHost` races the MCP thread and can
  wedge the kernel — so under stdio the launcher logs a hint and skips it. Custom
  routes are *not* covered by FastMCP's transport-security, so each route carries
  its **own** Host/Origin guard reusing the SDK's `TransportSecurityMiddleware`
  validators with the same loopback allowlist. Reads reuse `_server._run_job_call`;
  controls are direct `KernelHost` calls — no new IPC, no dask-dashboard
  dependence. Wired by `__main__._setup_observe` (fully guarded; a failure never
  blocks the server).
- `_process_ops.py` — `build_ops()`: a thin `Run()` callable per configured
  `ProcessImage` servicer URL (discovered via `GetOpNames`), exposed as `ops`.
- `_resources.py` — string constants served as MCP resources.
- `_helpers.py` — monkey-patches `viewer.add_tensor()` for agent use; also
  `viewer_window_alive()`, the closed-window liveness probe (bound into the
  kernel namespace as `_viewer_window_alive`) that lets `server_status` /
  `take_screenshot` / `execute_code` detect a user-closed window instead of
  silently mutating a destroyed viewer.

**Tools (9):** `start_kernel`, `take_screenshot`, `execute_code`, `poll_job`,
`cancel_job`, `inspect_object`, `interrupt_kernel`, `restart_kernel`,
`server_status`.

**Resources (5):** `guide://kernel`, `guide://viewer`, `guide://tensor`,
`guide://annotations`, `guide://ops`.

**`execute_code` namespace** (populated by `_bootstrap.py`):
- `viewer` — the live `napari.Viewer`
- `np` — numpy, `da` — dask.array
- `client` — `TensorFlightClient` (or `None` if not connected); refreshed
  per-call from the connection service
- `ops` — `dict[str, callable]` of `biopb.image` ProcessImage ops from the
  configured servers (may be empty)

**Security model (important):** the kernel is a real IPython kernel with imports
allowed — `execute_code` is arbitrary code execution by design. The server binds
**loopback only** and assumes a localhost / trusted-intranet deployment;
untrusted-network exposure is expected to be fronted by a separately-documented
reverse proxy. The server **enforces a DNS-rebinding / `Origin` / `Host`
allowlist** (loopback only, set explicitly in `_server.py` via
`build_transport_security()`), so a malicious page in the user's browser is
rejected (`403`/`421`) before it can reach the kernel. The allowlist is
extensible via `config['mcp']['transport']['allowed_origins']` /
`config['mcp']['transport']['allowed_hosts']` for a reverse-proxy front; there
is no loopback token (the `Origin` check already
blocks the browser-attacker threat). Do not describe this as sandboxed.

**Error-propagation model (general — applies beyond startup).** Letting an
exception propagate *upward* — including out of a Qt callback (timer/signal slot,
e.g. the connect/autostart ticks in `tensor_browser/_widget.py`) — is **generally
safe in both hosts and will not qFatal-abort the process**, so prefer surfacing
real failures over silently swallowing them. The reason: PyQt6 only calls
`qFatal()`/`abort()` on an unhandled slot exception when `sys.excepthook` is the
*default* `sys.__excepthook__`, and a **non-default hook is always installed** in
both contexts — napari's `notification_manager` in the standalone plugin (set by
`napari.run()` via `with notification_manager:`), and ipykernel's
`ZMQInteractiveShell.excepthook` in the **MCP kernel** (where `napari.run()` is
never called — the kernel's `%gui qt` inputhook drives the loop and `run()`
early-returns under an IPython loop — so napari's hooks are absent but the kernel's
own hook covers it). The only difference is *where the error surfaces*: a GUI
notification in the plugin vs. a printed traceback in the kernel output (mcp-server.log
under stdio) for MCP. Verified empirically: a slot exception aborts with SIGABRT
only under the default hook and exits cleanly under any override, and the ipykernel
hook stays non-default through `enable_gui("qt")` and `napari.Viewer()`. Caveat:
this only covers crash-safety — a propagated error in the MCP kernel produces a log
traceback, not an agent- or user-facing message, so still catch where you want a
clean, reported failure.

### Configuration (`_config.py`)

`DEFAULT_CONFIG` is stored under the home-relative config dir
(`~/.config/biopb-mcp`, matching the biopb server and installer on all
platforms) and **deep-merged** with defaults on load — a partial nested user
section overrides only its own leaves and leaves sibling defaults intact. Read
values with `get_setting(config, "dotted.path")`, which falls back to
`DEFAULT_CONFIG` so call sites never restate a default literal.

Top-level sections: `widget` (the experimental demo widgets' settings —
`server_url`, `is_3d`, `detection`, and `grid` tiling — the only consumer is
`image_processing/`), `tensor_browser.server_url` (data-plane URL, deliberately
top-level because the GUI-independent `TensorConnection` reads it from the
headless kernel too), `timeout`, `grpc.max_message_size_mb`, `memory`, and `mcp`.
The `mcp` section is grouped by concern:

- **`mcp.transport`** — `kind` (`stdio`/`http`, default `stdio`; chosen at
  startup, also via `--transport`), `port` (8765), `display_mode`, `kernel_log`
  (stdio-only; where the kernel's native stdout/stderr is redirected so it can't
  corrupt fd 1, default `~/.local/share/biopb-mcp/log/mcp-server.log`), and
  `allowed_origins`/`allowed_hosts` (extra Host/Origin values appended to the
  loopback DNS-rebinding allowlist for a reverse-proxy front; http only).
- **`mcp.kernel`** — `name`, `startup_timeout`, `execute_timeout` (120s; now
  bounds only the quick in-band snippets), `busy_lock_timeout` (5s),
  `promote_after` (10s; how long `execute_code` waits inline before returning a
  job handle), `parent_death_pipe`, and the `watchdog_*` orphan-hardening knobs
  (issue #13).
- **`mcp.dask`** — `scheduler` defaults to `distributed`, which with an empty
  `address` auto-spins a multi-process `LocalCluster`; any distributed mode lets
  `cancel_job` stop an in-flight `.compute()`. Set a non-empty `address` to
  attach to an external scheduler, or `threads`/`synchronous` for a low-overhead
  in-process scheduler with no mid-compute cancel. `owner` (default `daemon`)
  decides who owns the auto-spun cluster: `daemon` — the MCP daemon owns it and
  it survives kernel restart/respawn/window-close (the kernel attaches via an
  injected scheduler address; no cold worker re-spawn per restart, which is the
  dominant restart cost on Windows), so worker/memory changes need a *daemon*
  restart, not just `restart_kernel`; `kernel` — the legacy per-kernel cluster.
  `num_workers`/`threads_per_worker`/`memory_limit`/`dashboard_address` size the
  auto-spun cluster (dashboard binds loopback only); `cache_budget` bounds the
  cluster-wide chunk cache (split across workers).
- **`mcp.tensor`** — `health_poll_min_interval`/`health_poll_max_interval` (the
  background source watcher's backoff bounds; the kernel re-lists when
  `source_count` changes so a catalog cached mid-index self-heals — issue #44;
  min `<= 0` disables it). (The localhost client-cache decision lives in the
  tensor client itself — `_resolve_cache_bytes`, off by default with
  `BIOPB_CACHE_LOCAL=1` as the opt-out — the MCP kernel no longer overrides it.)
- **`mcp.viewer`** — `compute_scheduler` (default `threads`; pins the napari
  viewer's *serial* slice reads to a single-process scheduler via
  `_viewer_compute.wrap_levels` so they share the one main-process `conn.client`
  chunk cache instead of scattering across the distributed cluster's per-worker
  caches — issue #8. The viewer being serial makes this free; the agent's
  explicit `da` computes keep the distributed default. `synchronous` for fully
  serial reads, `""` to disable wrapping). `async_slicing` (default `True`)
  fetches viewer slices *off* the Qt main thread (napari experimental async
  slicing, enabled via the `NAPARI_ASYNC` env var set in `_bootstrap.py` before
  `import napari` — the `_LayerSlicer` captures the flag once at construction),
  so a zoom into a not-yet-cached pyramid level keeps the current coarse texture
  on screen instead of freezing for the cold read; the read still uses the
  wrapped serial scheduler, just off-thread. `take_screenshot` force-syncs the
  current view first (`_helpers.resync_view_for_capture`) so the agent's capture
  reflects the state it set, not a pre-load frame. Set `False` to restore fully
  synchronous slicing.
- **`mcp.services.process_image_servers`** — the `grpc://`/`grpcs://`
  ProcessImage URLs exposed as `ops`.
- **`mcp.observe`** — the web observe UI (`_observe.py`): `enabled` (**on by
  default / opt-out**; **http transport only** — it mounts on the MCP app so it
  adds no surface beyond `/mcp`; under stdio it is silently skipped, since a
  second HTTP server in the stdio process risks the fd-1 JSON-RPC channel and
  races the kernel), `max_output_chars` (detail-view stdout cap, tail kept),
  `poll_interval_ms` (page poll cadence).
- **`mcp.server_start_timeout`** — the autostart boot-wait budget (issue #12).

For back-compat, `load_config` relocates the one legacy flat key the biopb
installer still seeds (`mcp.process_image_servers` →
`mcp.services.process_image_servers`); the next `save_config` persists the nested
form, so it self-heals.

### Plugin Entry Points

Defined in `napari.yaml` (manifest registered via the
`napari.manifest` entry point in `pyproject.toml`):
- `Tensor Browser` → `biopb_mcp.tensor_browser:TensorBrowserWidget`
- `Object Detection` → `biopb_mcp.image_processing:ObjectDetectionWidget`
- `Image Processing` → `biopb_mcp.image_processing:ImageProcessingWidget`

The console script `biopb-mcp` (`pyproject.toml [project.scripts]`) launches the
MCP server.

### napari Thread Worker Pattern

The `image_processing` demo widgets use the `@thread_worker` decorator from
`napari.qt.threading`. Workers:
- Yield progress updates (`yield None`) or results (`yield value`)
- Connect `.yielded`, `.finished`, `.errored` signals to callbacks
- Support cancellation via `.quit()` and `.await_workers()`

## macOS CI Testing

macOS headless CI environments cannot initialize OpenGL contexts, causing
segfaults when napari creates a viewer. Tests that use `make_napari_viewer` must
be skipped on macOS CI:

```python
@pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("CI") == "true",
    reason="OpenGL context unavailable on macOS CI headless environment",
)
```

Apply at the class level for test classes that require napari viewers. See
`test_widget.py` and `test_progress_timer.py` for examples.

## Project Structure

- `src/biopb_mcp/` — main source
  - `_connection.py` — GUI-independent `TensorConnection` data-access service
    (incl. `start_source_watch`, the background self-healing-catalog watcher —
    issue #44)
  - `_config.py` — configuration management
  - `_tensor_utils.py` — pyramid-level building and dimension utilities
  - `_viewer_compute.py` — `wrap_levels` / `_ViewerArray`: pin viewer slice
    reads to a single-process scheduler (issue #8)
  - `_typing.py`, `_utils.py`, `_version.py`
  - `napari.yaml` — plugin manifest
  - `tensor_browser/` — `_widget.py` (`TensorBrowserWidget`)
  - `image_processing/` — experimental demo widgets and `biopb.image` gRPC:
    - `_object_det_widget.py`, `_image_processing_widget.py`, `_widget_base.py`
    - `_grpc.py`, `_chunking.py`, `_render.py`
  - `mcp/` — MCP server module (optional; `pip install biopb-mcp[mcp]`)
    - `__main__.py` / `__init__.py` — launcher + public API
    - `_server.py` — FastMCP server, tools, resources
    - `_kernel.py` — `KernelHost` (child Jupyter kernel lifecycle)
    - `_jobs.py` — in-kernel async job runner (submit/poll/cancel, main-thread
      marshaling, thread-aware stdout, user-action attribution)
    - `_observe.py` — loopback web "observe" UI (job history + global
      cancel/interrupt/restart), in the MCP server process
    - `_bootstrap.py` — in-kernel bootstrap (namespace, viewer, dask, ops, jobs)
    - `_process_ops.py` — `build_ops()` ProcessImage callables
    - `_resources.py` — resource content strings
    - `_helpers.py` — `add_tensor()` viewer patch + `viewer_window_alive()`
      closed-window probe
  - `_tests/` — test files

## Dependencies

Main (from `pyproject.toml`):
- numpy, pandas, magicgui, qtpy, scikit-image
- `biopb[tensor] >= 0.5.4` (Arrow Flight tensor client)
- opencv-python-headless
- grpcio-tools, grpcio-health-checking
- vedo (3D visualization)

Optional MCP extra (`pip install biopb-mcp[mcp]`):
- `mcp >= 1.20`, `uvicorn >= 0.29`, `jupyter_client`, `ipykernel`, `psutil`

Testing group includes: pytest, pytest-cov, pytest-qt, pytest-env, napari,
pyqt6, jupyter_client, ipykernel, napari-skimage-regionprops.

## Development Notes

- setuptools_scm for versioning
- magicgui for automatic GUI generation in the demo widgets
- QT API set to pyqt6 in pytest configuration
- Headless testing uses `QT_QPA_PLATFORM=offscreen`
