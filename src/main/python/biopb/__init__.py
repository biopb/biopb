"""Top-level BioPB Python package metadata."""

from __future__ import annotations

try:
    import importlib.metadata as _importlib_metadata
except ImportError:
    import importlib_metadata as _importlib_metadata

try:
    __version__ = _importlib_metadata.version("biopb")
except Exception:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0"

__all__ = ["__version__"]
