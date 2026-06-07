# biopb-mcp

## Python Environment

This project uses a **`uv`-managed virtualenv at `.venv/`**, pinned by `uv.lock`.
Always run tooling through that interpreter — e.g. `.venv/bin/python -m pytest …`,
`.venv/bin/python -m black …` — **not** the bare `python`/`pytest` on `PATH`. A
system or user-site Python can shadow the project env with a *different* (often
older) `biopb`, which silently breaks tests that depend on newer client APIs
(e.g. `make_cache_plugin`, added in `biopb >= 0.5.8`).

After pulling changes — especially a bump to the `biopb[tensor]` pin in
`pyproject.toml` — run `uv sync --extra mcp --group testing` to bring `.venv`
back in line with `uv.lock`. If imports or versions look wrong, first check the
interpreter (`which python` / `.venv/bin/python -c "import biopb; print(biopb.__version__)"`)
before debugging the code; a stale `.venv` (missing `uv sync`) is the usual cause.

Test/dev deps live in **PEP 735 dependency *groups*, not extras** (groups can
pin direct-URL wheels that PyPI would reject in published metadata — see the
comment in `pyproject.toml`). Two groups:

- **`integration`** — the runnable full stack with *no* test tooling: the MCP
  server deps plus the `biopb-tensor-server` (Arrow Flight) wheel and the
  **version-paired `biopb` dev wheel** from the latest biopb-tensor-server
  GitHub release. Sync this for a checkout that needs a real local tensor server
  (`biopb server start` / autostart) without pytest/tox:
  `uv sync --extra mcp --group integration`.
- **`testing`** — `integration` **plus** pytest/tox. This is the normal dev/CI
  profile and what tox installs (`dependency_groups = testing`).

Because these are groups, select them with `--group <name>`, *not*
`--extra <name>`. The paired wheels pin `biopb` to the release's dev build
rather than the latest PyPI release — that pairing is exactly how the server
ships and how end users get it. To move to a newer server release, bump both
release-asset URLs in `pyproject.toml`'s `[dependency-groups]` and re-run
`uv lock`.

**Important — groups don't reach PyPI users.** A dependency group is not part of
biopb-mcp's published wheel/sdist metadata, and `--group` resolves against this
`pyproject.toml`, not a PyPI-installed package. So `pip install biopb-mcp[...]`
can never pull the tensor server; the groups only help a source checkout (incl.
the `release_bundle.yml` build, which could `pip install . --group integration`
to bake a local server into the bundled app). Delivering the server to
`pip install biopb-mcp` users would require publishing biopb-tensor-server to
PyPI and exposing it as a real extra.

## Build and Test Commands

```bash
# Install the package in development mode (uv — preferred)
uv sync --extra mcp --group testing
# …or with pip >= 25.1 (dependency groups):
pip install -e ".[mcp]" --group testing

# Run all tests with coverage
pytest -v --cov=biopb_mcp --cov-report=xml

# Run a single test file
pytest src/biopb_mcp/_tests/test_mcp_kernel.py -v

# Run a single test function
pytest src/biopb_mcp/_tests/test_grpc.py::test_encode_image -v

# Run tests via tox (tests across Python 3.10-3.12)
tox

# Format code with black (line-length 79)
black src/

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
   AI agent over MCP (streamable-http). The project thesis is *"agent first;
   provide tools only if they help."* The agent drives napari through a real
   Python kernel; image results go to the viewer, other results to the agent's
   chat.

Read `napari.md` for a summary of napari's architecture and plugin system.

### Data connection (`_connection.py`)

`TensorConnection` is a **GUI-independent** data-access service (imports neither
Qt nor napari). It owns the `biopb.tensor.TensorFlightClient`, the source
catalog, and URL/token resolution + persistence. Both the `TensorBrowserWidget`
and the MCP kernel *consume* this service rather than owning a client.

URL/token resolution fallback chain (`resolve_from_config`):

1. **Environment variables**: `BIOPB_TENSOR_URL`, `BIOPB_TENSOR_TOKEN`
2. **Config file**: `tensor_browser.server_url` in the config
3. **Default**: `grpc://localhost:8815`

If the initial connect fails and the URL is local *and* the `biopb` CLI is on
PATH, `start_local_server()` can run `biopb server start` as a last-resort
fallback (the widget drives this). Only the URL is persisted; the token is read
from the environment, not saved.

### MCP Server Module (`mcp/`)

The MCP server **is its own process** — it does *not* auto-start on plugin
import. Launch it with the console script `biopb-mcp` or `python -m
biopb_mcp.mcp` (requires `$DISPLAY`; a viewer window appears). Install the
optional deps with `pip install biopb-mcp[mcp]`.

**Transport** is chosen once at startup (not co-served) via `--transport` /
`config['mcp']['transport']['kind']`: `stdio` (default — JSON-RPC on
stdin/stdout, for a client that spawns `biopb-mcp --transport stdio` as a
subprocess) or `http` (loopback streamable-http on `transport.port`). The
kernel/viewer/dask stack is identical either way. In stdio mode fd 1 *is* the
protocol channel, so the launcher logs to stderr and redirects the kernel's
*native* stdout/stderr to a log file
(`config['mcp']['transport']['kernel_log']`, default
`~/.config/biopb-mcp/kernel.log`) — otherwise Qt/GL/dask/gRPC C-level writes
would corrupt the stream. stdio has no network surface, so the Host/Origin
allowlist (`transport.allowed_origins`/`transport.allowed_hosts`) is http-only
and not applied. NB:
the orphaned-kernel risks (issue #13) are amplified under stdio (one kernel tree
per client launch); the `PR_SET_PDEATHSIG`/watchdog hardening there is a
prerequisite before recommending stdio for production.

The process owns a **single child Jupyter kernel** (real IPython kernel via
`jupyter_client`) that hosts the napari viewer, dask, and the tensor client.
Agent code runs *in that kernel*, not on the MCP server's thread or napari's Qt
loop — so a runaway execution can be interrupted (`SIGINT`) or hard-restarted
(process-group `SIGKILL` + respawn) without killing the MCP server.

**Async job model (`_jobs.py`):** `execute_code` runs agent code in a
**background daemon thread inside the kernel**, so the kernel main thread (and
its `%gui qt` Qt loop) stays free to service `take_screenshot` / `server_status`
/ `poll_job` mid-job — the agent is not blind during long work. `execute_code`
waits up to `promote_after` seconds; if the code finishes it returns the result
inline, otherwise it returns a job handle (`job-N`) and keeps running. One job
at a time. Because the viewer has Qt main-thread affinity, GUI mutations from
the worker thread are marshaled to the main thread: `_bootstrap` wraps
`load_tensor` + the `add_*` family, and `run_on_main(fn)` is exposed for the
rest; `cancelled()` supports cooperative `cancel_job`. Rich IPython `display()`
output is not captured (code is exec'd, not run via `run_cell`).

**Files:**
- `__main__.py` / `__init__.py` — launcher; builds the `KernelHost`, injects the
  bootstrap via `IPKernelApp.exec_lines`, then runs the FastMCP server.
- `_server.py` — FastMCP server (`mcp = FastMCP("biopb-mcp")`); defines the
  tools/resources and dispatches them to the `KernelHost`. `run()` serves
  streamable-http on `127.0.0.1:<port>/mcp`; `run_stdio()` serves the same
  tools over stdio (no port, no Host/Origin allowlist).
- `_kernel.py` — `KernelHost`: manages the one child kernel. A single
  `threading.RLock` serializes access to the kernel shell channel; with the job
  model it is held only for the *quick* snippets (submit/poll/screenshot/status)
  — never during the long compute, which runs in a detached kernel thread.
  `execute()` has an `execute_timeout` (now bounds only those quick snippets,
  SIGINT on timeout) and a `busy_lock_timeout` (returns `"busy"` if another
  quick call holds the lock). `restart()` releases dask then group-kills and
  respawns (re-running bootstrap, which clears job state).
- `_bootstrap.py` — runs *inside* the kernel via `exec_lines`. Enables `%gui qt`,
  configures dask, constructs the `TensorConnection`, opens the napari viewer +
  Tensor Browser, builds `ops`, installs the job runner (`_jobs.install` +
  `wrap_viewer_for_threads`), and populates the `execute_code` namespace. On
  failure it prints a `BOOTSTRAP_ERROR` sentinel; the host's health probe
  detects success via `'viewer' in dir()`.
- `_jobs.py` — runs *inside* the kernel. The async job runner: a `_jobs`
  registry, `submit`/`poll`/`cancel`/`jobs_summary`, a thread-aware stdout
  dispatcher, `run_on_main` (Qt main-thread marshaling with proper exception
  propagation), `cancelled()`, and `wrap_viewer_for_threads`. `cancel` is
  cooperative and also cancels in-flight distributed-dask futures.
- `_process_ops.py` — `build_ops()`: a thin `Run()` callable per configured
  `ProcessImage` servicer URL (discovered via `GetOpNames`), exposed as `ops`.
- `_resources.py` — string constants served as MCP resources.
- `_helpers.py` — monkey-patches `viewer.load_tensor()` for agent use.

**Tools (8):** `take_screenshot`, `execute_code`, `poll_job`, `cancel_job`,
`inspect_object`, `interrupt_kernel`, `restart_kernel`, `server_status`.

**Resources (5):** `guide://main`, `guide://viewer`, `guide://tensor`,
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
notification in the plugin vs. a printed traceback in the kernel output (kernel.log
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
  corrupt fd 1, default `~/.config/biopb-mcp/kernel.log`), and
  `allowed_origins`/`allowed_hosts` (extra Host/Origin values appended to the
  loopback DNS-rebinding allowlist for a reverse-proxy front; http only).
- **`mcp.kernel`** — `name`, `startup_timeout`, `execute_timeout` (120s; now
  bounds only the quick in-band snippets), `busy_lock_timeout` (5s),
  `promote_after` (10s; how long `execute_code` waits inline before returning a
  job handle), `parent_death_pipe`, and the `watchdog_*` orphan-hardening knobs
  (issue #13).
- **`mcp.dask`** — `scheduler` defaults to `distributed`, which with an empty
  `address` auto-spins a kernel-local multi-process `LocalCluster` (the only mode
  where `cancel_job` can stop an in-flight `.compute()`); set a non-empty
  `address` to attach to an external scheduler, or `threads`/`synchronous` for a
  low-overhead in-process scheduler with no mid-compute cancel.
  `num_workers`/`threads_per_worker`/`memory_limit`/`dashboard_address` size the
  auto-spun cluster (dashboard binds loopback only); `cache_budget` bounds the
  cluster-wide chunk cache (split across workers).
- **`mcp.tensor`** — `cache_local` (let the data-plane client cache chunks even
  for a localhost server; translated to `BIOPB_CACHE_LOCAL` in the kernel env).
- **`mcp.viewer`** — `compute_scheduler` (default `threads`; pins the napari
  viewer's *serial* slice reads to a single-process scheduler via
  `_viewer_compute.wrap_levels` so they share the one main-process `conn.client`
  chunk cache instead of scattering across the distributed cluster's per-worker
  caches — issue #8. The viewer being serial makes this free; the agent's
  explicit `da` computes keep the distributed default. `synchronous` for fully
  serial reads, `""` to disable wrapping).
- **`mcp.services.process_image_servers`** — the `grpc://`/`grpcs://`
  ProcessImage URLs exposed as `ops`.
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
      marshaling, thread-aware stdout)
    - `_bootstrap.py` — in-kernel bootstrap (namespace, viewer, dask, ops, jobs)
    - `_process_ops.py` — `build_ops()` ProcessImage callables
    - `_resources.py` — resource content strings
    - `_helpers.py` — `load_tensor()` viewer patch
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

Testing extra includes: tox, pytest, pytest-cov, pytest-qt, pytest-env, napari,
pyqt6, jupyter_client, ipykernel, napari-skimage-regionprops.

## Development Notes

- setuptools_scm for versioning
- Black formatter, line-length 79; Ruff for linting
- magicgui for automatic GUI generation in the demo widgets
- QT API set to pyqt6 in pytest configuration
- Headless testing uses `QT_QPA_PLATFORM=offscreen`
