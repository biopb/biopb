"""Logging configuration for biopb-tensor-server.

Provides centralized logging setup with configurable level via:
- CLI option --log-level
- Environment variable BIOPB_LOG_LEVEL
- Config file [server.log_level]

Default level is INFO, which shows operational messages without debug noise.

By default (scope_to_biopb=True), logging level changes only affect the
biopb_tensor_server package hierarchy, not external packages like grpc, numpy, etc.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Literal, Optional

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
    log_file: Optional[str] = None,
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_count: int = 5,
) -> None:
    """Configure logging for biopb-tensor-server.

    Sets up a StreamHandler with standard format including timestamp,
    level, module name, and message. Optionally adds a rotating file handler.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format string
        datefmt: Datetime format string for timestamps
        scope_to_biopb: If True, only configure biopb_tensor_server logger hierarchy.
                        If False, configure the root logger (affects all packages).
        log_file: Path to log file. When set, adds a rotating file handler alongside
                  the console handler.
        log_max_bytes: Maximum size of each log file before rotation (default 10MB).
        log_backup_count: Number of rotated log files to keep (default 5).

    Example:
        setup_logging("DEBUG")  # Show all messages (scoped to biopb_tensor_server)
        setup_logging("DEBUG", scope_to_biopb=False)  # Affects all packages
        setup_logging("WARNING")  # Only warnings and errors
        setup_logging("INFO", log_file="/var/log/biopb.log")  # Also write to rotating file
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
    formatter = logging.Formatter(log_format, datefmt)

    if scope_to_biopb:
        # Configure only the biopb_tensor_server logger hierarchy
        logger = logging.getLogger("biopb_tensor_server")
        logger.setLevel(numeric_level)

        # Add a stream handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Add rotating file handler if log_file specified and not already present
        if log_file and not any(
            isinstance(h, RotatingFileHandler) for h in logger.handlers
        ):
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.debug(
            f"Logging configured: level={level_str}, scope=biopb_tensor_server"
        )
    else:
        # Configure root logger (affects all packages)
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_file:
            handlers.append(
                RotatingFileHandler(
                    log_file,
                    maxBytes=log_max_bytes,
                    backupCount=log_backup_count,
                )
            )
        for h in handlers:
            h.setFormatter(formatter)
        logging.basicConfig(
            level=numeric_level,
            format=format,
            datefmt=datefmt,
            handlers=handlers,
        )

        logger = logging.getLogger(__name__)
        logger.debug(f"Logging configured: level={level_str}, scope=root")
