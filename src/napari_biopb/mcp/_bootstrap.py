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
        print("BOOTSTRAP_ERROR: " + traceback.format_exc())


def _bootstrap_impl():
    import dask.array as da
    import napari
    import numpy as np
    from IPython import get_ipython

    from .._config import load_config
    from ..tensor_browser import TensorBrowserWidget
    from ._helpers import patch_viewer_load_tensor

    # 1. Qt integration must be enabled before the viewer is created so napari
    #    shares the kernel's integrated Qt event loop (programmatic %gui qt).
    ip = get_ipython()
    ip.enable_gui("qt")

    config = load_config()
    mcp_config = config.get("mcp", {})

    # 2. Configure dask in the compute process.
    dask_client = _configure_dask(mcp_config)

    # 3. Visible napari viewer + Tensor Browser (auto-connects on its own tick).
    viewer = napari.Viewer()
    tbw = TensorBrowserWidget(viewer)
    viewer.window.add_dock_widget(tbw, name="Tensor Browser")

    # 4. Namespace for execute_code.  client is refreshed per-call by the
    #    server (the widget connects asynchronously).
    patch_viewer_load_tensor(viewer, tbw)
    ip.user_ns.update(
        {
            "viewer": viewer,
            "np": np,
            "da": da,
            "client": None,
            "_tbw": tbw,
            "_dask_client": dask_client,
        }
    )
