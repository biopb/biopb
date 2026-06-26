"""Base utilities for biopb.image gRPC services.

This package provides:
- Server creation helpers with health checks and authentication
- Image encoding/decoding for both eager and lazy data
- Logging configuration matching tensor-server pattern
- Debug utilities for stats tracking and system info
- Base servicer class with error handling
"""

from biopb_image_base import dynamics_local, stitch
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
from biopb_image_base.server import create_server, run_server
from biopb_image_base.stitch import stitch_lazy_segmentation, uniform_core

__all__ = [
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
