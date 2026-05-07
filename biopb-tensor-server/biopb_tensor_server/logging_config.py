"""Logging configuration for biopb-tensor-server.

Provides centralized logging setup with configurable level via:
- CLI option --log-level
- Environment variable BIOPB_LOG_LEVEL
- Config file [server.log_level]

Default level is INFO, which shows operational messages without debug noise.
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
    format: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Configure logging for biopb-tensor-server.

    Sets up a StreamHandler with standard format including timestamp,
    level, module name, and message.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        datefmt: Datetime format string for timestamps

    Example:
        setup_logging("DEBUG")  # Show all messages
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

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format,
        datefmt=datefmt,
        handlers=[logging.StreamHandler()],
    )

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={level_str}")