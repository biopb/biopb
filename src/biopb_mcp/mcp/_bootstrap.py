"""Bootstrap executed *inside* the MCP child kernel.

Injected via IPython ``exec_lines`` so it runs before the kernel services any
tool calls.  It enables the Qt event loop, configures dask in the process
where compute actually happens, opens a visible napari viewer with the Tensor
Browser widget, and populates the ``execute_code`` namespace.

A failure here does not abort the kernel (exec_lines errors are swallowed by
IPython), so ``bootstrap`` prints a ``BOOTSTRAP_ERROR`` sentinel that the
host's health probe detects via the absence of ``viewer`` in the namespace.
"""

import logging
import os
import traceback

logger = logging.getLogger(__name__)


class _HeadlessViewer:
    """Stand-in bound to ``viewer`` when the kernel runs without a display.

    Any attribute access raises a clear, quotable error so agent code that
    reaches for the viewer (``viewer.add_image(...)``, ``viewer.layers`` …)
    surfaces a self-describing message — relayed to the user by the model —
    instead of a cryptic ``AttributeError`` on ``None``.  Falsy so kernel
    snippets can guard with ``if viewer:``.
    """

    _MSG = (
        "napari viewer unavailable: this biopb-mcp kernel started headless "
        "(no display). Data access (client), compute (ops), and execute_code "
        "still work — there is just no viewer window or screenshot."
    )

    def __getattr__(self, name):
        raise RuntimeError(self._MSG)

    def __repr__(self):
        return "<headless: no napari viewer (no display)>"

    def __bool__(self):
        return False


def _configure_dask(config: dict):
    """Set up dask in the kernel process.

    Returns ``(client, cluster)``:

    * ``"distributed"`` + an external ``mcp.dask.address`` -> a
      ``Client`` attached to that scheduler; ``cluster`` is ``None``.
    * ``"distributed"`` + no address -> a kernel-local multi-process
      ``LocalCluster`` and a ``Client`` bound to it. This is the default and
      the only mode where ``cancel_job`` can stop an in-flight ``compute()``.
    * ``"threads"`` / ``"synchronous"`` -> in-process scheduler; both ``None``.

    A failure spinning the local cluster degrades gracefully to the in-process
    ``threads`` scheduler rather than aborting the bootstrap.
    """
    import dask

    from .._config import get_setting

    scheduler = get_setting(config, "mcp.dask.scheduler")
    num_workers = get_setting(config, "mcp.dask.num_workers") or None
    address = get_setting(config, "mcp.dask.address")

    if scheduler == "distributed":
        try:
            from dask.distributed import Client

            if address:
                client = Client(address)
                logger.info("Dask using distributed scheduler at %s", address)
                return client, None

            from dask.distributed import LocalCluster

            # Put worker spill dirs under a launcher-owned temp dir (when set)
            # so the launcher can rmtree them on shutdown — a group-SIGKILL of
            # the kernel leaves workers no chance to clean up after themselves
            # (issue #13, secondary disk-leak note).
            local_directory = os.environ.get("BIOPB_DASK_LOCAL_DIR") or None

            cluster = LocalCluster(
                n_workers=num_workers,
                processes=True,
                threads_per_worker=get_setting(
                    config, "mcp.dask.threads_per_worker"
                ),
                memory_limit=get_setting(config, "mcp.dask.memory_limit"),
                dashboard_address=get_setting(
                    config, "mcp.dask.dashboard_address"
                ),
                local_directory=local_directory,
            )
            client = Client(cluster)
            logger.info(
                "Dask using local cluster: %d worker(s) at %s",
                len(cluster.workers),
                cluster.scheduler_address,
            )
            return client, cluster
        except Exception:
            # Covers a missing `distributed` install, an unreachable external
            # address, or a LocalCluster spawn failure -- degrade to the
            # in-process scheduler so the bootstrap (and the viewer) survives.
            logger.exception(
                "Distributed dask unavailable; "
                "falling back to in-process threads scheduler"
            )
            scheduler = "threads"

    dask.config.set(scheduler=scheduler, num_workers=num_workers)
    logger.info("Dask scheduler: %s, num_workers: %s", scheduler, num_workers)
    return None, None


def _register_cache_plugin(
    dask_client, url, token, config: dict, planned_workers=None
):
    """Pin the data-plane chunk-cache budget across dask workers.

    For a **remote** server, splits ``mcp.dask.cache_budget`` evenly across the
    workers and installs a worker-init plugin so each worker (current and future)
    caps its per-process cache at ``budget // n_workers``.

    For a **localhost** server, pins every worker to **no cache** (0). A dask
    worker consumes a whole chunk as its task unit, so the slice-chunk-mismatch
    amortization that justifies a client-side cache -- the viewer slicing a small
    region out of a large chunk -- never applies to a worker. On localhost the
    server's mmap/page-cache fast path already shares decoded chunks across all
    workers for free, so a per-worker cache is pure replicated memory (measured:
    ~1.7x a plane's size, with no speedup). Pinning workers off here restores
    biopb's own localhost default for them while leaving the main-process viewer
    cache (the ``BIOPB_CACHE_LOCAL`` opt-in) intact -- that env var, inherited by
    every worker, would otherwise leak a real cache into each one.

    No-op without a distributed client. Best-effort: a failure here must not
    break the connect flow that invokes it. Called from
    ``TensorConnection.on_connect`` with the final ``(url, token)`` (the token is
    only known after connect).
    """
    if dask_client is None:
        return
    try:
        from biopb.tensor.client import make_cache_plugin
        from dask.utils import parse_bytes

        from .._config import get_setting

        try:
            from biopb.tensor.client import _is_localhost_location
        except Exception:  # pragma: no cover - older biopb without the helper

            def _is_localhost_location(_url):
                return False

        n_workers = max(
            1,
            planned_workers
            or len(dask_client.scheduler_info().get("workers", {})),
        )

        if _is_localhost_location(url):
            # Workers consume whole chunks and share the server's mmap/page-cache
            # path, so a per-worker cache only replicates memory (see docstring).
            per_worker = 0
        else:
            budget_cfg = get_setting(config, "mcp.dask.cache_budget")
            budget = (
                int(budget_cfg)
                if isinstance(budget_cfg, int | float)
                else parse_bytes(budget_cfg)
            )
            per_worker = max(0, budget // n_workers)

        plugin = make_cache_plugin(url, token, per_worker)
        if plugin is None:
            return
        dask_client.register_plugin(plugin)
        logger.info(
            "Chunk-cache plugin: %d B/worker x %d workers (%s)",
            per_worker,
            n_workers,
            url,
        )
    except Exception:
        logger.exception("Failed to register chunk-cache budget plugin")


def _install_window_close_hook(viewer):
    """Signal the launcher when the user closes the napari window.

    The launcher inherits the *write* end of a pipe via ``BIOPB_WINDOW_CLOSE_FD``
    (set by ``KernelHost._launch``, name = ``_kernel.ENV_WINDOW_CLOSE_FD``); a
    reader thread there reaps this kernel back to idle on the byte we write. We
    connect to the Qt main window's ``destroyed`` signal — the same
    ``viewer.window._qt_window`` the closed-window probe (``viewer_window_alive``)
    keys off — which fires once the C++ window is deleted (a user X-close deletes
    it). Idempotent and fully best-effort: a missing fd, an absent window, or any
    wiring/IO failure must never break the bootstrap.
    """
    fd_str = os.environ.get("BIOPB_WINDOW_CLOSE_FD")
    if not fd_str:
        return
    try:
        fd = int(fd_str)
    except ValueError:
        return

    fired = {"done": False}

    def _notify(*_args):
        if fired["done"]:
            return
        fired["done"] = True
        try:
            os.write(fd, b"x")
        except OSError:
            pass

    try:
        viewer.window._qt_window.destroyed.connect(_notify)
    except Exception:
        logger.exception("Failed to install napari window-close hook")


def bootstrap():
    """Entry point called from the kernel's exec_lines."""
    try:
        _bootstrap_impl()
    except Exception:
        tb = traceback.format_exc()
        # Stash the traceback in the kernel namespace so the host's health
        # probe can fetch and surface it.  exec_lines output is otherwise
        # swallowed by IPython, leaving the probe with only "viewer absent".
        try:
            from IPython import get_ipython

            get_ipython().user_ns["_BOOTSTRAP_ERROR"] = tb
        except Exception:
            pass
        print("BOOTSTRAP_ERROR: " + tb)


def _bootstrap_impl():
    import dask.array as da
    import numpy as np
    from IPython import get_ipython

    from .._config import get_setting, load_config
    from .._connection import TensorConnection
    from . import _jobs
    from ._process_ops import build_ops

    ip = get_ipython()
    config = load_config()

    # Headless (compute-only) mode: the launcher sets BIOPB_HEADLESS when no
    # display is available (or display_mode forces it), so we skip Qt/napari
    # entirely rather than crash on a missing display.  client/ops/execute_code
    # still work; `viewer` is a self-describing sentinel.
    headless = bool(os.environ.get("BIOPB_HEADLESS"))

    # 1. Qt integration must be enabled before the viewer is created so napari
    #    shares the kernel's integrated Qt event loop (programmatic %gui qt).
    if not headless:
        ip.enable_gui("qt")

    # 2. Configure dask in the compute process.
    dask_client, dask_cluster = _configure_dask(config)

    # 3. Data-access service, shared by the widget and the agent namespace.
    conn = TensorConnection(config)

    # 3b. Bound the data-plane chunk cache across the worker processes, which
    #     would otherwise each replicate it. For a localhost server the plugin
    #     pins workers to no cache (they consume whole chunks and share the
    #     server's mmap/page-cache path; the main-process viewer keeps its cache
    #     via BIOPB_CACHE_LOCAL for its sub-chunk slice reads); for a remote
    #     server it splits mcp.dask.cache_budget across workers. Registered via
    #     the connect hook because the token is only final after connect, and
    #     re-runs on reconnect (the plugin is named, so re-registration replaces).
    #     Divide by the cluster's *planned* worker count (worker_spec), not the
    #     live scheduler count, which lags while workers are still registering.
    planned_workers = (
        len(dask_cluster.worker_spec)
        if dask_cluster is not None and hasattr(dask_cluster, "worker_spec")
        else None
    )
    conn.on_connect = lambda url, token: _register_cache_plugin(
        dask_client, url, token, config, planned_workers
    )

    # 4. Visible napari viewer + Tensor Browser (auto-connects on its own tick).
    #    compute_scheduler pins the viewer's serial slice reads to a
    #    single-process scheduler so they share the main-process chunk cache
    #    instead of scattering across the distributed cluster (issue #8).
    #    Headless: no viewer — `viewer` is a self-describing sentinel instead.
    compute_scheduler = get_setting(config, "mcp.viewer.compute_scheduler")
    if headless:
        viewer = _HeadlessViewer()
        logger.info("Headless mode: no napari viewer (no display).")
        # No widget exists to drive the initial connect (the GUI branch's
        # TensorBrowserWidget runs the same conn.auto_connect policy off a
        # worker thread), so drive it here. On a daemon thread: connect() blocks
        # on network I/O and we must not stall kernel bring-up (this runs in
        # exec_lines, ahead of start_kernel returning). execute_code refreshes
        # `client` from `_conn.client` per job, so a connect that lands after the
        # kernel is ready is still seen.
        import threading

        threading.Thread(
            target=conn.auto_connect,
            name="biopb-headless-connect",
            daemon=True,
        ).start()
    else:
        # Enable napari async slicing via its NAPARI_ASYNC env override, set
        # BEFORE importing napari. The settings singleton reads the env at load,
        # and the viewer's _LayerSlicer captures the flag once at construction
        # (_layer_slicer.py: ``self._force_sync = not ...async_``) -- so the env
        # var is the only reliable hook; assigning the settings object after
        # import is too late (the settings load resets it). Async slicing
        # fetches slices off the Qt main thread so a zoom into a not-yet-cached
        # level doesn't freeze the viewer (vispy keeps the current coarse
        # texture until the finer slice resolves); take_screenshot force-syncs a
        # slice before capturing so the agent still sees the requested frame
        # (resync_view_for_capture).
        os.environ["NAPARI_ASYNC"] = (
            "1" if get_setting(config, "mcp.viewer.async_slicing") else "0"
        )

        import napari

        from ..tensor_browser import TensorBrowserWidget

        viewer = napari.Viewer()
        tbw = TensorBrowserWidget(
            viewer, connection=conn, compute_scheduler=compute_scheduler
        )
        viewer.window.add_dock_widget(tbw, name="Tensor Browser")
        # Tear the kernel down to idle when the user closes the window: signal
        # the launcher's reader thread over the inherited window-close pipe.
        _install_window_close_hook(viewer)

    # 5. ProcessImage ops: thin Run() callables for each configured servicer.
    #    client_getter reads conn.client lazily so the async-connecting tensor
    #    client is picked up at call time.
    max_msg_bytes = (
        get_setting(config, "grpc.max_message_size_mb") * 1024 * 1024
    )
    channel_options = [
        ("grpc.max_receive_message_length", max_msg_bytes),
        ("grpc.max_send_message_length", max_msg_bytes),
    ]
    try:
        ops = build_ops(
            client_getter=lambda: conn.client,
            server_urls=get_setting(
                config, "mcp.services.process_image_servers"
            ),
            op_names_timeout=get_setting(config, "timeout.get_op_names"),
            run_timeout=get_setting(config, "timeout.process_image"),
            channel_options=channel_options,
        )
    except Exception:
        logger.exception("Failed to build ProcessImage ops")
        ops = {}

    # 6. Async job runner: execute_code runs in a background kernel thread so
    #    the main thread / Qt loop stays free for screenshot/status mid-job.
    #    install() stores the shell, installs the thread-aware stdout streams,
    #    and clears any prior job state; wrap_viewer_for_threads marshals the
    #    common viewer-mutating methods to the Qt main thread.
    _jobs.install(ip)
    if not headless:
        from ._helpers import patch_viewer_add_tensor

        patch_viewer_add_tensor(
            viewer, conn, compute_scheduler=compute_scheduler
        )
        _jobs.wrap_viewer_for_threads(viewer)

    # 7. Namespace for execute_code.  client is refreshed per-job by the job
    #    runner (the connection service connects asynchronously).
    #    _viewer_window_alive lets the tools detect a user-closed window (the
    #    Python `viewer` survives a window close, so mutations silently no-op).
    from ._helpers import resync_view_for_capture, viewer_window_alive

    ip.user_ns.update(
        {
            "viewer": viewer,
            "np": np,
            "da": da,
            "client": None,
            "ops": ops,
            "_conn": conn,
            "_dask_client": dask_client,
            "_dask_cluster": dask_cluster,
            "_jobs": _jobs,
            "run_on_main": _jobs.run_on_main,
            "cancelled": _jobs.cancelled,
            "_viewer_window_alive": lambda: viewer_window_alive(viewer),
            "_resync_view": lambda: resync_view_for_capture(viewer),
        }
    )

    # 8. Background source-catalog watcher (issue #44): a daemon thread that
    #    health-checks the server and re-lists sources when its source_count
    #    changes, so a catalog cached while the server was still indexing
    #    self-heals — for the agent (reads `_conn.sources` live) and, in a GUI
    #    session, the widget (which wires its own tree rebuild and also starts
    #    the watch; the call is idempotent). Thread-based, not a QTimer, so it
    #    runs even headless where there is no Qt loop.
    try:
        conn.start_source_watch(
            min_interval=get_setting(
                config, "mcp.tensor.health_poll_min_interval"
            ),
            max_interval=get_setting(
                config, "mcp.tensor.health_poll_max_interval"
            ),
        )
    except Exception:
        logger.exception("Failed to start source watcher")
