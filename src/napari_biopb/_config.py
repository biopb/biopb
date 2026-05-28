"""Configuration management for napari-biopb plugin.

Provides persistent storage of user settings and configurable parameters.
Uses platformdirs for cross-platform config directory location.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "server": {
        "url": "lacss.biopb.org",
    },
    "tensor_browser": {
        "server_url": "grpc://localhost:8815",
    },
    "detection": {
        "min_score": 0.4,
        "size_hint": 32.0,
        "nms": "Off",
        "z_aspect_ratio": 1.0,
    },
    "grid": {
        "2d_size": [4096, 4096],
        "2d_stride": [4000, 4000],
        "3d_size": [64, 512, 512],
        "3d_stride": [48, 480, 480],
    },
    "timeout": {
        "health_check": 5.0,
        "get_op_names": 10.0,
        "detection_2d": 15,
        "detection_3d": 300,
        "process_image": 300,
    },
    "grpc": {
        "max_message_size_mb": 512,
        "max_concurrent_calls": 4,
    },
    "memory": {
        "warn_threshold_mb": 500,  # Log warning if chunk > 500MB
        "error_threshold_mb": 2000,  # Raise MemoryError if chunk > 2GB
    },
    "mcp": {
        "port": 8765,
        "dask_scheduler": "threads",
        "dask_num_workers": 0,
        "dask_distributed_address": "",
        "kernel_name": "python3",
        "kernel_startup_timeout": 60.0,
        "execute_timeout": 120.0,
        "busy_lock_timeout": 5.0,
        # biopb.image ProcessImage servicer URLs (grpc:// or grpcs://).
        # Each is queried via GetOpNames and exposed as callables in the
        # agent kernel's `ops` dict.
        "process_image_servers": [],
    },
}


def get_default_config() -> dict:
    """Return a deep copy of the default configuration.

    Returns:
        Fresh copy of DEFAULT_CONFIG to prevent mutation.
    """
    return copy.deepcopy(DEFAULT_CONFIG)


def get_config_dir() -> Path:
    """Get the platform-appropriate config directory.

    Returns:
        Path to the config directory for napari-biopb.
    """
    from platformdirs import user_config_dir

    config_dir = Path(user_config_dir("napari-biopb"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the path to the config file.

    Returns:
        Path to config.json file.
    """
    return get_config_dir() / "config.json"


def load_config() -> dict:
    """Load configuration from file.

    If the config file doesn't exist, returns default config.
    If the file exists but is malformed, returns default config and logs error.

    Returns:
        Configuration dict with all expected keys.
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.debug("Config file not found, using defaults")
        return get_default_config()

    try:
        with config_path.open("r") as f:
            config = json.load(f)

        # Merge with defaults to ensure all keys exist
        merged = get_default_config()
        for key in merged:
            if key in config:
                if isinstance(merged[key], dict):
                    merged[key].update(config[key])
                else:
                    merged[key] = config[key]

        logger.debug("Loaded config from %s", config_path)
        return merged

    except json.JSONDecodeError as e:
        logger.warning("Config file malformed, using defaults: %s", e)
        return get_default_config()
    except Exception as e:
        logger.warning("Failed to load config, using defaults: %s", e)
        return get_default_config()


def save_config(config: dict) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dict to save.
    """
    config_path = get_config_path()

    try:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        logger.debug("Saved config to %s", config_path)
    except Exception as e:
        logger.warning("Failed to save config: %s", e)


def get_grid_params(
    is_3d: bool, config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Get grid size and stride from config.

    Args:
        is_3d: Whether processing 3D data.
        config: Configuration dict.

    Returns:
        Tuple of (grid_size, stride) as numpy arrays.
    """
    grid_config = config.get("grid", DEFAULT_CONFIG["grid"])

    if is_3d:
        grid_size = np.array(
            grid_config.get("3d_size", DEFAULT_CONFIG["grid"]["3d_size"]),
            dtype=int,
        )
        stride = np.array(
            grid_config.get("3d_stride", DEFAULT_CONFIG["grid"]["3d_stride"]),
            dtype=int,
        )
    else:
        grid_size = np.array(
            grid_config.get("2d_size", DEFAULT_CONFIG["grid"]["2d_size"]),
            dtype=int,
        )
        stride = np.array(
            grid_config.get("2d_stride", DEFAULT_CONFIG["grid"]["2d_stride"]),
            dtype=int,
        )

    return grid_size, stride
