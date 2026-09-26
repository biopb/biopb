"""Base utilities for biopb.image gRPC services.

This package provides:
- ``op`` / ``serve``: a server of the ``Ops`` protocol from decorated functions
- Server creation helpers with health checks and authentication
- Image encoding/decoding for both eager and lazy data
- Logging configuration matching tensor-server pattern
- Debug utilities for stats tracking and system info
- Base servicer class with error handling

The core needs only ``biopb`` and gRPC. Lazy input and a plane sink need the
``[lazy]`` extra (dask, pyarrow), and the stitching helpers the ``[stitch]``
extra (scipy), so the names that need them load on first use.
"""

import importlib

from biopb_image_base.common import (
    BiopbServicerBase,
    decode_image_data,
    encode_image,
    ensure_eager,
    parse_kwargs,
    return_lazy_or_eager,
    validate_kwargs,
)
from biopb_image_base.health import HealthServicer, add_health_servicer
from biopb_image_base.logging_config import get_log_level_from_env, setup_logging
from biopb_image_base.ops import Tensor, op, serve

_LAZY_MODULES = ("stitch", "dynamics_local")
_LAZY_NAMES = {
    "create_server": "server",
    "run_server": "server",
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
    "create_server",
    "run_server",
    "BiopbServicerBase",
    "decode_image_data",
    "encode_image",
    "return_lazy_or_eager",
    "parse_kwargs",
    "validate_kwargs",
    "ensure_eager",
    "HealthServicer",
    "add_health_servicer",
    "stitch",
    "dynamics_local",
    "stitch_lazy_segmentation",
    "uniform_core",
]
