"""Logging configuration for biopb-image-runtime.

Provides centralized logging setup with configurable level via:
- CLI option --log-level
- Environment variable BIOPB_LOG_LEVEL
- Config file [server.log_level]

Default level is INFO, which shows operational messages without debug noise.

By default (scope_to_biopb=True), logging level changes only affect the
biopb_image_base package hierarchy, not external packages like grpc, numpy, etc.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_log_level_from_env() -> LogLevel | None:
    """Get log level from environment variable BIOPB_LOG_LEVEL."""
    env_level = os.environ.get("BIOPB_LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return env_level
    return None


def setup_logging(
    level: LogLevel | str = "INFO",
    log_format: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    scope_to_biopb: bool = True,
) -> None:
    """Configure logging for biopb-image-runtime.

    Sets up a StreamHandler with standard format including timestamp,
    level, module name, and message.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format string
        datefmt: Datetime format string for timestamps
        scope_to_biopb: If True, only configure biopb_image_base logger hierarchy.
                        If False, configure the root logger (affects all packages).

    Example:
        setup_logging("DEBUG")  # Show all messages (scoped to biopb_image_base)
        setup_logging("DEBUG", scope_to_biopb=False)  # Affects all packages
        setup_logging("WARNING")  # Only warnings and errors
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Normalize level string
    level_str = str(level).upper()
    numeric_level = level_map.get(level_str, logging.INFO)

    if scope_to_biopb:
        # Configure only the biopb_image_base logger hierarchy
        logger = logging.getLogger("biopb_image_base")
        logger.setLevel(numeric_level)

        # Add a handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(log_format, datefmt))
            logger.addHandler(handler)

        # Log the configuration
        logger.debug(f"Logging configured: level={level_str}, scope=biopb_image_base")
    else:
        # Configure root logger (affects all packages)
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            datefmt=datefmt,
            handlers=[logging.StreamHandler()],
        )

        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.debug(f"Logging configured: level={level_str}, scope=root")
