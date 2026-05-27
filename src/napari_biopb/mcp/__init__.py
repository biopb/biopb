"""MCP server for napari-biopb.

Exposes the napari viewer to AI agents over streamable-http MCP transport.
Install optional dependencies with ``pip install napari-biopb[mcp]``.

Public API::

    from napari_biopb.mcp import start_mcp_server, stop_mcp_server

    start_mcp_server(viewer)   # called automatically on plugin load
    stop_mcp_server()          # optional cleanup
"""

import logging
import threading

logger = logging.getLogger(__name__)

_started = False
_lock = threading.Lock()


def start_mcp_server(viewer, port=None):
    """Start the MCP server for *viewer*.

    Args:
        viewer: A ``napari.Viewer`` instance.
        port: HTTP port (default: from config, fallback 8765).
    """
    global _started
    with _lock:
        if _started:
            logger.debug("MCP server already running")
            return

        from .._config import load_config
        from ._bridge import ThreadBridge
        from ._server import launch_server

        config = load_config()
        mcp_config = config.get("mcp", {})
        if port is None:
            port = mcp_config.get("port", 8765)

        bridge = ThreadBridge(viewer)
        launch_server(bridge, port=port, mcp_config=mcp_config)
        _started = True


def stop_mcp_server():
    """Stop the MCP bridge timer."""
    global _started
    with _lock:
        from ._server import shutdown_server

        shutdown_server()
        _started = False


# ---------------------------------------------------------------------------
# Auto-start: when this module is imported and a viewer exists, start serving.
# Uses QTimer.singleShot so the actual work runs on the Qt event loop.
# ---------------------------------------------------------------------------
_auto_start_attempts = 0


def _try_auto_start():
    global _auto_start_attempts
    _auto_start_attempts += 1

    import napari

    viewer = napari.current_viewer()
    if viewer is None:
        if _auto_start_attempts < 10:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(500, _try_auto_start)
        return

    try:
        start_mcp_server(viewer)
    except Exception:
        logger.warning("MCP auto-start failed", exc_info=True)


try:
    import mcp as _mcp_pkg  # noqa: F401 — skip if [mcp] extras not installed
    from qtpy.QtCore import QTimer

    QTimer.singleShot(0, _try_auto_start)
except ImportError:
    pass
