"""biopb-admin: the biopb control plane (supervision + single-origin front).

Lean by construction — supervises the durable planes as subprocesses and never
imports them (invariant I2, mcp-dedaemonization-migration.md). Public surface:
:func:`run_admin` (the blocking entry) and :class:`DataPlaneSupervisor`.
"""

from __future__ import annotations

try:
    import importlib.metadata as _importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as _importlib_metadata

try:
    __version__ = _importlib_metadata.version("biopb-admin")
except Exception:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0"

from ._admin import run_admin
from ._supervisor import DataPlaneSupervisor

__all__ = ["__version__", "run_admin", "DataPlaneSupervisor"]
