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
import traceback

logger = logging.getLogger(__name__)


def _configure_dask(mcp_config: dict):
    """Set up dask in the kernel process.

    Returns a distributed ``Client`` when connecting to an external cluster,
    otherwise ``None`` (in-process ``threads``/``synchronous`` scheduler).
    """
    import dask

    scheduler = mcp_config.get("dask_scheduler", "threads")
    num_workers = mcp_config.get("dask_num_workers", 0) or None
    address = mcp_config.get("dask_distributed_address", "")

    if scheduler == "distributed" and address:
        from dask.distributed import Client

        client = Client(address)
        logger.info("Dask using distributed scheduler at %s", address)
        return client

    dask.config.set(scheduler=scheduler, num_workers=num_workers)
    logger.info("Dask scheduler: %s, num_workers: %s", scheduler, num_workers)
    return None


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
    import napari
    import numpy as np
    from IPython import get_ipython

    from .._config import load_config
    from .._connection import TensorConnection
    from ..tensor_browser import TensorBrowserWidget
    from ._helpers import patch_viewer_load_tensor
    from ._process_ops import build_ops

    # 1. Qt integration must be enabled before the viewer is created so napari
    #    shares the kernel's integrated Qt event loop (programmatic %gui qt).
    ip = get_ipython()
    ip.enable_gui("qt")

    config = load_config()
    mcp_config = config.get("mcp", {})

    # 2. Configure dask in the compute process.
    dask_client = _configure_dask(mcp_config)

    # 3. Data-access service, shared by the widget and the agent namespace.
    conn = TensorConnection(config)

    # 4. Visible napari viewer + Tensor Browser (auto-connects on its own tick).
    viewer = napari.Viewer()
    tbw = TensorBrowserWidget(viewer, connection=conn)
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

    # 6. Namespace for execute_code.  client is refreshed per-call by the
    #    server (the connection service connects asynchronously).
    patch_viewer_load_tensor(viewer, conn)
    ip.user_ns.update(
        {
            "viewer": viewer,
            "np": np,
            "da": da,
            "client": None,
            "ops": ops,
            "_conn": conn,
            "_dask_client": dask_client,
        }
    )
