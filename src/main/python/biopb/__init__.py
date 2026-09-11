"""Top-level BioPB Python package metadata."""

from __future__ import annotations

# The generated file first, dist-info METADATA only as the fallback -- the order
# the other packages in this repo already use (biopb-control adopted it in
# biopb/biopb#910). METADATA is stamped at install time and the generated file at
# build time, so preferring METADATA reports whenever the SDK was last installed
# rather than what is being imported: in an editable checkout that drifts behind
# its own source. The `v*` tag line this package ships on is separate from the
# product's `release-v*` line, but the drift is the same.
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
