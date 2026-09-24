"""biopb-mcp: an MCP server that drives a napari viewer for AI agents.

The napari widgets it docks come from ``biopb-napari-widget``. The package root
imports nothing heavy, so the MCP modules (e.g. ``biopb_mcp.workflow_env``)
import without Qt/napari.
"""

try:
    from ._version import version as __version__
except ImportError:
    import importlib.metadata

    __version__ = importlib.metadata.version("biopb-mcp")
except Exception:
    __version__ = "unknown"
