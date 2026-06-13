"""Backward-compatibility shim for the renamed ``biopb-mcp`` package.

``napari-biopb`` was renamed to ``biopb-mcp``. This distribution exists only
so that ``pip install napari-biopb`` keeps working: it depends on
``biopb-mcp`` and re-exports the plugin widgets so that existing
``from napari_biopb import TensorBrowserWidget`` code continues to run.

Submodules (``napari_biopb.mcp``, ``napari_biopb._config``, ...) are *not*
re-exported — import them from ``biopb_mcp`` instead. This shim will not be
updated; migrate to ``biopb-mcp``.
"""

import warnings

warnings.warn(
    "'napari-biopb' has been renamed to 'biopb-mcp'. Update your dependency "
    "to 'biopb-mcp' and import from 'biopb_mcp'. This shim re-exports the "
    "plugin widgets only and will not be maintained.",
    DeprecationWarning,
    stacklevel=2,
)

# biopb-mcp's package root is intentionally bare; the widgets now live in
# subpackages. Re-export them here so legacy top-level imports keep working.
from biopb_mcp import __version__  # noqa: F401,E402
from biopb_mcp.image_processing import (  # noqa: F401,E402
    ImageProcessingWidget,
    ObjectDetectionWidget,
)
from biopb_mcp.tensor_browser import TensorBrowserWidget  # noqa: F401,E402

__all__ = [
    "ImageProcessingWidget",
    "ObjectDetectionWidget",
    "TensorBrowserWidget",
    "__version__",
]
