"""Backward-compatibility shim for the renamed ``biopb-mcp`` package.

``napari-biopb`` was renamed to ``biopb-mcp``. This distribution exists only
so that ``pip install napari-biopb`` keeps working: it depends on
``biopb-mcp`` and re-exports its top-level names so that existing
``import napari_biopb`` code continues to run.

Submodules (``napari_biopb.mcp``, ``napari_biopb._config``, ...) are *not*
re-exported — import them from ``biopb_mcp`` instead. This shim will not be
updated; migrate to ``biopb-mcp``.
"""

import warnings

warnings.warn(
    "'napari-biopb' has been renamed to 'biopb-mcp'. Update your dependency "
    "to 'biopb-mcp' and import from 'biopb_mcp'. This shim re-exports the "
    "top-level API only and will not be maintained.",
    DeprecationWarning,
    stacklevel=2,
)

from biopb_mcp import *  # noqa: F401,F403,E402
from biopb_mcp import __all__, __version__  # noqa: F401,E402
