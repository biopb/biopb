"""biopb-mcp: an MCP server that drives a napari viewer for AI agents, plus a
Tensor Browser napari plugin widget for human data browsing.

The package root is intentionally minimal — it does not import the GUI widgets —
so that data-layer and MCP modules (e.g. ``biopb_mcp._connection``) can be
imported without pulling in Qt/napari. The plugin widgets live in their
subpackages (``biopb_mcp.tensor_browser`` / ``biopb_mcp.image_processing``) and
are referenced directly by the napari manifest (``napari.yaml``).
"""

try:
    from ._version import version as __version__
except ImportError:
    import importlib.metadata

    __version__ = importlib.metadata.version("biopb-mcp")
except Exception:
    __version__ = "unknown"
