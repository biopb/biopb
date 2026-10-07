"""BioPB Python package: the version, and the public subpackages.

``biopb.tensor`` reads tensors from a data plane, ``biopb.image`` calls
algorithm servers. Everything else lives in private modules. Stdlib only, so a
bare ``import biopb`` stays cheap.
"""

from __future__ import annotations

# Prefer the build-time generated file: dist-info METADATA is stamped at install
# time and goes stale in an editable checkout.
try:
    from ._version import version as __version__
except ImportError:
    try:
        import importlib.metadata as _importlib_metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as _importlib_metadata

    try:
        __version__ = _importlib_metadata.version("biopb")
    except Exception:
        __version__ = "0.0.0"

__all__ = ["__version__"]
