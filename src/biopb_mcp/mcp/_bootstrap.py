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


def _configure_dask(mcp_config: dict):
    """Set up dask in the kernel process.

    Returns ``(client, cluster)``:

    * ``"distributed"`` + an external ``dask_distributed_address`` -> a
      ``Client`` attached to that scheduler; ``cluster`` is ``None``.
    * ``"distributed"`` + no address -> a kernel-local multi-process
      ``LocalCluster`` and a ``Client`` bound to it. This is the default and
      the only mode where ``cancel_job`` can stop an in-flight ``compute()``.
    * ``"threads"`` / ``"synchronous"`` -> in-process scheduler; both ``None``.

    A failure spinning the local cluster degrades gracefully to the in-process
    ``threads`` scheduler rather than aborting the bootstrap.
    """
    import dask

    scheduler = mcp_config.get("dask_scheduler", "distributed")
    num_workers = mcp_config.get("dask_num_workers", 0) or None
    address = mcp_config.get("dask_distributed_address", "")

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
                threads_per_worker=mcp_config.get(
                    "dask_threads_per_worker", 1
                ),
                memory_limit=mcp_config.get("dask_memory_limit", "auto"),
                dashboard_address=mcp_config.get(
                    "dask_dashboard_address", "127.0.0.1:0"
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
    dask_client, url, token, mcp_config: dict, planned_workers=None
):
    """Pin a cluster-wide data-plane chunk-cache budget across dask workers.

    Splits ``mcp.dask_cache_budget`` evenly across the live workers and installs
    a worker-init plugin so each worker (current and future) caps its per-process
    cache at ``budget // n_workers``. No-op without a distributed client; the
    plugin itself resolves a localhost server to no cache. Best-effort: a failure
    here must not break the connect flow that invokes it.

    Called from ``TensorConnection.on_connect`` with the final ``(url, token)``
    (the token is only known after connect).
    """
    if dask_client is None:
        return
    try:
        from dask.utils import parse_bytes

        from biopb.tensor.client import make_cache_plugin

        budget_cfg = mcp_config.get("dask_cache_budget", "1G")
        budget = (
            int(budget_cfg)
            if isinstance(budget_cfg, (int, float))
            else parse_bytes(budget_cfg)
        )
        n_workers = planned_workers or len(
            dask_client.scheduler_info().get("workers", {})
        )
        n_workers = max(1, n_workers)
        per_worker = max(0, budget // n_workers)

        plugin = make_cache_plugin(url, token, per_worker)
        if plugin is None:
            return
        dask_client.register_plugin(plugin)
        logger.info(
            "Chunk-cache budget %s -> %d B/worker across %d workers (%s)",
            budget_cfg,
            per_worker,
            n_workers,
            url,
        )
    except Exception:
        logger.exception("Failed to register chunk-cache budget plugin")


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

    from .._config import load_config
    from .._connection import TensorConnection
    from . import _jobs
    from ._process_ops import build_ops

    ip = get_ipython()
    config = load_config()
    mcp_config = config.get("mcp", {})

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
    dask_client, dask_cluster = _configure_dask(mcp_config)

    # 3. Data-access service, shared by the widget and the agent namespace.
    conn = TensorConnection(config)

    # 3b. Pin a bounded, cluster-wide chunk-cache budget across the worker
    #     processes. The data-plane client's per-process cache is otherwise
    #     replicated in every worker (budget x n_workers); splitting one budget
    #     across workers bounds the aggregate. A localhost server resolves to no
    #     cache regardless, so this only bites for remote servers. Registered via
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
        dask_client, url, token, mcp_config, planned_workers
    )

    # 4. Visible napari viewer + Tensor Browser (auto-connects on its own tick).
    #    compute_scheduler pins the viewer's serial slice reads to a
    #    single-process scheduler so they share the main-process chunk cache
    #    instead of scattering across the distributed cluster (issue #8).
    #    Headless: no viewer — `viewer` is a self-describing sentinel instead.
    compute_scheduler = mcp_config.get("viewer_compute_scheduler", "threads")
    if headless:
        viewer = _HeadlessViewer()
        logger.info("Headless mode: no napari viewer (no display).")
    else:
        import napari

        from ..tensor_browser import TensorBrowserWidget

        viewer = napari.Viewer()
        tbw = TensorBrowserWidget(
            viewer, connection=conn, compute_scheduler=compute_scheduler
        )
        viewer.window.add_dock_widget(tbw, name="Tensor Browser")

    # 5. ProcessImage ops: thin Run() callables for each configured servicer.
    #    client_getter reads conn.client lazily so the async-connecting tensor
    #    client is picked up at call time.
    timeout_config = config.get("timeout", {})
    grpc_config = config.get("grpc", {})
    max_msg_bytes = grpc_config.get("max_message_size_mb", 512) * 1024 * 1024
    channel_options = [
        ("grpc.max_receive_message_length", max_msg_bytes),
        ("grpc.max_send_message_length", max_msg_bytes),
    ]
    try:
        ops = build_ops(
            client_getter=lambda: conn.client,
            server_urls=mcp_config.get("process_image_servers", []),
            op_names_timeout=timeout_config.get("get_op_names", 10.0),
            run_timeout=timeout_config.get("process_image", 300.0),
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
        from ._helpers import patch_viewer_load_tensor

        patch_viewer_load_tensor(
            viewer, conn, compute_scheduler=compute_scheduler
        )
        _jobs.wrap_viewer_for_threads(viewer)

    # 7. Namespace for execute_code.  client is refreshed per-job by the job
    #    runner (the connection service connects asynchronously).
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
        }
    )
