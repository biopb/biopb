"""Configuration management for biopb-mcp plugin.

Provides persistent storage of user settings and configurable parameters.
Config lives under the home-relative XDG path ~/.config/biopb-mcp, matching the
biopb server (~/.config/biopb) and the installer across all platforms.
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
        # Dask defaults to a kernel-local multi-process distributed cluster
        # (LocalCluster). This is the only configuration where cancel_job can
        # stop an in-flight .compute() mid-execution, and it gives real
        # (non-GIL) CPU parallelism. Set dask_scheduler to "threads" /
        # "synchronous" for a low-overhead in-process scheduler (no
        # mid-compute cancel), or set dask_distributed_address to attach to an
        # external scheduler instead of spinning a local one.
        "dask_scheduler": "distributed",
        # n_workers for the auto-spun LocalCluster (0 -> let dask pick, ~n_cores).
        # When connecting to an external scheduler this is ignored.
        "dask_num_workers": 0,
        # Non-empty -> connect to this external scheduler address; empty -> spin
        # a kernel-local LocalCluster.
        "dask_distributed_address": "",
        # LocalCluster sizing (used only when spinning a local cluster).
        "dask_threads_per_worker": 1,
        "dask_memory_limit": "auto",
        # Cluster-wide chunk-cache budget for the data-plane client, split
        # evenly across dask workers (each worker caches budget // n_workers).
        # Bounds aggregate cache regardless of worker count -- the per-process
        # client cache is otherwise replicated in every worker. Accepts a
        # human-readable size ("1G", "512M", "2GiB") or an int (bytes), parsed
        # with dask.utils.parse_bytes. Localhost servers cache nothing
        # regardless (the tensor server already caches); this applies to remote.
        "dask_cache_budget": "1G",
        # Bokeh dashboard bind address; loopback-only to match the server's
        # loopback-only security model. ":0" picks a free port.
        "dask_dashboard_address": "127.0.0.1:0",
        "kernel_name": "python3",
        "kernel_startup_timeout": 60.0,
        # Give-up budget (seconds) for start_local_server's just-launched boot
        # wait, where connection-refused is tolerated while the server binds and
        # scans its data folder. The normal auto-connect path has no timeout: it
        # fails fast on a down server and polls a STARTING (scanning) server
        # indefinitely with progress feedback (issue #12).
        "server_start_timeout": 60.0,
        # execute_timeout now bounds only the *quick* in-band kernel snippets
        # (screenshot / status / inspect / job submit+poll), not long jobs:
        # execute_code runs agent code in a background thread that may run
        # indefinitely (stop it with cancel_job / restart_kernel).
        "execute_timeout": 120.0,
        "busy_lock_timeout": 5.0,
        # Seconds execute_code waits for a job to finish before "promoting" it:
        # if it completes within this window the result is returned inline,
        # otherwise a job handle (job_id) is returned and it keeps running.
        "promote_after": 10.0,
        # Extra Host/Origin header values appended to the loopback allowlist
        # that guards the server against DNS-rebinding / cross-origin browser
        # requests. Set these only when fronting the server with a reverse
        # proxy that needs its own Host/Origin permitted.
        "allowed_origins": [],
        "allowed_hosts": [],
        # biopb.image ProcessImage servicer URLs (grpc:// or grpcs://).
        # Each is queried via GetOpNames and exposed as callables in the
        # agent kernel's `ops` dict.
        "process_image_servers": [],
        # Prefer the gRPC socket over the tensor server's /dev/shm fast-path.
        # Translated into BIOPB_SHM_TRANSFER_DISABLED in the kernel env by
        # __main__.py. The shm path creates+writes+unlinks a fresh POSIX
        # segment per chunk, which measured ~2.4-3x slower than the socket for
        # large localhost chunks; this is a stopgap until the shm path is fixed
        # (biopb/biopb#9). No-op on Windows (no POSIX shm). Off by
        # default to preserve current behavior; the installer seeds it on.
        "tensor_disable_shm": False,
        # Let the data-plane client cache chunks even for a *localhost* tensor
        # server. Translated into BIOPB_CACHE_LOCAL in the kernel/worker env by
        # __main__.py. By default the client skips a localhost cache (the
        # server already caches), but that means an interactive viewer scrubbing
        # planes re-fetches the whole enclosing chunk per plane; caching makes
        # repeated/overlapping reads instant. Memory is bounded by the existing
        # dask_cache_budget plugin (each worker caps at budget // n_workers), so
        # this does not reintroduce the per-worker replication that the unbounded
        # cache caused. A structural fix that removes the underlying read
        # amplification (client-selectable read granularity) is tracked in
        # biopb/biopb#8.
        "tensor_cache_local": True,
        # Scheduler for the napari viewer's slice reads. The viewer scrubs
        # planes one at a time (serial np.asarray(data[slices])), but the
        # bootstrap makes a distributed LocalCluster the *default* scheduler, so
        # each single-chunk slice fetch scatters across a rotating worker and the
        # opaque per-worker chunk cache misses + replicates (issue #8). Pinning
        # the viewer's arrays to a single-process scheduler runs every slice read
        # in the kernel main process against the one shared conn.client cache:
        # ~100% hit on revisit, 1x memory, no scatter -- and costs no
        # parallelism because the viewer is serial. The agent's explicit `da`
        # computes still use the distributed default. "threads" keeps parallel
        # multi-chunk reads within a slice; "synchronous" is fully serial;
        # "" disables wrapping (viewer slices compute on the global default).
        "viewer_compute_scheduler": "threads",
    },
}


def get_default_config() -> dict:
    """Return a deep copy of the default configuration.

    Returns:
        Fresh copy of DEFAULT_CONFIG to prevent mutation.
    """
    return copy.deepcopy(DEFAULT_CONFIG)


def get_config_dir() -> Path:
    """Get the config directory (~/.config/biopb-mcp on all platforms).

    Returns:
        Path to the config directory for biopb-mcp.
    """
    config_dir = Path.home() / ".config" / "biopb-mcp"
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
