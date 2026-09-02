"""biopb-control: the biopb control plane (supervision + single-origin front).

Lean by construction — supervises the durable planes as subprocesses and never
imports them (invariant I2, biopb-mcp/ARCHITECTURE.md). Public surface:
:func:`run_control` (the blocking entry) and :class:`DataPlaneSupervisor`.
"""

from __future__ import annotations

# The generated file first, dist-info METADATA only as the fallback -- the same
# order as biopb-mcp and biopb-tensor-server, which share this package's
# `release-v*` tag line. METADATA is stamped at install time and the generated
# file at build time, so preferring METADATA reported the last `uv sync` rather
# than the last build: in an editable checkout that drifts releases behind.
try:
    from ._version import version as __version__
except ImportError:
    try:
        import importlib.metadata as _importlib_metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as _importlib_metadata

    try:
        __version__ = _importlib_metadata.version("biopb-control")
    except Exception:
        __version__ = "0.0.0"

from ._run import run_control
from ._supervisor import DataPlaneSupervisor

__all__ = ["__version__", "run_control", "DataPlaneSupervisor"]
