"""Launcher for the biopb-mcp MCP server.

This process *is* the MCP server: it owns a child Jupyter kernel that hosts a
visible napari viewer (requires ``$DISPLAY``).  Run it with::

    biopb-mcp        # console script
    python -m biopb_mcp.mcp

Install the optional dependencies first: ``pip install biopb-mcp[mcp]``.
"""

import atexit
import logging
import signal
import sys

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    from .._config import load_config
    from . import _server
    from ._kernel import KernelHost

    config = load_config()
    mcp_config = config.get("mcp", {})
    port = mcp_config.get("port", 8765)

    bootstrap_line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
    extra_arguments = [f"--IPKernelApp.exec_lines={bootstrap_line}"]

    host = KernelHost(
        extra_arguments=extra_arguments,
        kernel_name=mcp_config.get("kernel_name", "python3"),
        startup_timeout=mcp_config.get("kernel_startup_timeout", 60.0),
        execute_timeout=mcp_config.get("execute_timeout", 120.0),
        busy_lock_timeout=mcp_config.get("busy_lock_timeout", 5.0),
    )
    _server.set_kernel_host(host)

    logger.info("Starting napari kernel (a viewer window will appear)...")
    host.start()
    logger.info("Kernel ready.")

    atexit.register(host.shutdown)

    def _handle_signal(signum, frame):
        logger.info("Received signal %s, shutting down.", signum)
        host.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _server.run(port)


if __name__ == "__main__":
    main()
