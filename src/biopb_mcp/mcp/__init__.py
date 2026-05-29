"""MCP server for biopb-mcp.

The server runs as its own process that owns a child Jupyter kernel hosting a
visible napari viewer.  Start it with the console script or module entry point::

    biopb-mcp        # console script
    python -m biopb_mcp.mcp

Install the optional dependencies with ``pip install biopb-mcp[mcp]``.
"""

from ._kernel import KernelHost

__all__ = ["KernelHost", "main"]


def main():
    """Console-script entry point (see ``biopb_mcp.mcp.__main__``)."""
    from .__main__ import main as _main

    return _main()
