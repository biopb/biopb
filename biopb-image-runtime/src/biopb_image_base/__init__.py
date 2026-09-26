"""Serve functions over the biopb.image ``Ops`` protocol.

This package provides:
- ``op`` / ``serve``: a server of the ``Ops`` protocol from decorated functions
- The gRPC health service and token authentication its servers use
- Logging configuration matching tensor-server pattern
- The flow-stitching helpers (``stitch``, ``dynamics_local``)

The core needs only ``biopb`` and gRPC. Lazy input and a plane sink need the
``[lazy]`` extra (dask, pyarrow), and the stitching helpers the ``[stitch]``
extra (scipy), so the names that need them load on first use.
"""

import importlib

from biopb_image_base.health import HealthServicer, add_health_servicer
from biopb_image_base.logging_config import get_log_level_from_env, setup_logging
from biopb_image_base.ops import Tensor, op, serve

_LAZY_MODULES = ("stitch", "dynamics_local")
_LAZY_NAMES = {
    "stitch_lazy_segmentation": "stitch",
    "uniform_core": "stitch",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name in _LAZY_NAMES:
        module = importlib.import_module(f"{__name__}.{_LAZY_NAMES[name]}")
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "Tensor",
    "op",
    "serve",
    "setup_logging",
    "get_log_level_from_env",
    "HealthServicer",
    "add_health_servicer",
    "stitch",
    "dynamics_local",
    "stitch_lazy_segmentation",
    "uniform_core",
]
