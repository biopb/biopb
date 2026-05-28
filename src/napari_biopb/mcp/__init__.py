"""MCP server for napari-biopb.

The server runs as its own process that owns a child Jupyter kernel hosting a
visible napari viewer.  Start it with the console script or module entry point::

    napari-biopb-mcp        # console script
    python -m napari_biopb.mcp

Install the optional dependencies with ``pip install napari-biopb[mcp]``.
"""

from ._kernel import KernelHost

__all__ = ["KernelHost", "main"]


def main():
    """Console-script entry point (see ``napari_biopb.mcp.__main__``)."""
    from .__main__ import main as _main

    return _main()
