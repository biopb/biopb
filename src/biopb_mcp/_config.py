"""Configuration management for biopb-mcp plugin.

Provides persistent storage of user settings and configurable parameters.
Config lives under the home-relative XDG path ~/.config/biopb-mcp, matching the
biopb server (~/.config/biopb) and the installer across all platforms.

The schema is grouped by concern:

- ``widget``         -- settings for the experimental napari demo widgets
                        (``image_processing/``): the ProcessImage server URL, the
                        2D/3D mode toggle, and detection/grid parameters.
- ``tensor_browser`` -- data-plane URL, read by the GUI-independent
                        ``TensorConnection`` (so the headless MCP kernel uses it
                        too -- this is deliberately *not* under ``widget``).
- ``timeout``/``grpc``/``memory`` -- compute-plane knobs shared by the demo
                        widgets' gRPC path and the MCP kernel's ``ops``.
- ``mcp``            -- the MCP server, split into ``transport`` / ``kernel`` /
                        ``dask`` / ``tensor`` / ``viewer`` / ``services``.

Read settings with :func:`get_setting`, which falls back to ``DEFAULT_CONFIG``
so call sites never duplicate a default literal.
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
    # Experimental napari demo widgets (image_processing/). Only those widgets
    # read this section; the MCP server does not.
    "widget": {
        # ProcessImage server the demo widgets target.
        "server_url": "localhost:50051",
        # Whether the widgets operate in 3D mode (persisted toggle).
        "is_3d": False,
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
    },
    "tensor_browser": {
        "server_url": "grpc://localhost:8815",
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
        "transport": {
            # Front-end transport: "stdio" (JSON-RPC on stdin/stdout, for a
            # client that spawns biopb-mcp as a subprocess, the default) or
            # "http" (loopback streamable-http on `port`). The kernel/viewer/dask
            # stack is identical either way. Overridable per-launch with
            # `--transport`.
            "kind": "stdio",
            "port": 8765,
            # Whether the kernel opens a visible napari viewer:
            #   "auto"    -> visible if a display is present
            #                ($DISPLAY/$WAYLAND_DISPLAY on Linux; always present
            #                on macOS/Windows), else headless.
            #   "visible" -> require a display; fail fast at startup if none
            #                (preserves the shared-viewer contract -- see
            #                docs/biopb-architecture.md).
            #   "headless"-> never open a viewer (compute-only: client/ops/
            #                execute_code work, take_screenshot and viewer.* do
            #                not).
            # Headless avoids a hard Qt abort when launched from a CLI with no
            # display (e.g. an MCP client over stdio on a remote box). The agent
            # is told via the initialize `instructions` field, and
            # viewer-dependent tools return a clear message; the `viewer`
            # namespace object self-describes on access.
            "display_mode": "auto",
            # Where the child kernel's *native* (C-level) stdout/stderr is
            # written in stdio mode, so it never corrupts the JSON-RPC stream on
            # fd 1. Empty -> ~/.config/biopb-mcp/kernel.log. Ignored in http mode
            # (the kernel inherits the launcher's fds, which are not a protocol
            # channel).
            "kernel_log": "",
            # Extra Host/Origin header values appended to the loopback allowlist
            # that guards the server against DNS-rebinding / cross-origin browser
            # requests. Set these only when fronting the server with a reverse
            # proxy that needs its own Host/Origin permitted. http transport
            # only; ignored in stdio mode (no network surface).
            "allowed_origins": [],
            "allowed_hosts": [],
        },
        "kernel": {
            "name": "python3",
            "startup_timeout": 60.0,
            # execute_timeout now bounds only the *quick* in-band kernel snippets
            # (screenshot / status / inspect / job submit+poll), not long jobs:
            # execute_code runs agent code in a background thread that may run
            # indefinitely (stop it with cancel_job / restart_kernel).
            "execute_timeout": 120.0,
            "busy_lock_timeout": 5.0,
            # Seconds execute_code waits for a job to finish before "promoting"
            # it: if it completes within this window the result is returned
            # inline, otherwise a job handle (job_id) is returned and it keeps
            # running.
            "promote_after": 10.0,
            # Orphan hardening (issue #13). The kernel runs in its own session,
            # so an abnormal launcher/kernel death would otherwise orphan the
            # kernel + its dask LocalCluster.
            #   parent_death_pipe: kernel inherits a pipe read-end and group-kills
            #     itself when the launcher *process* dies (POSIX only; mode 1).
            #   watchdog_interval: seconds between liveness polls; on an
            #     unexpected kernel death the host reaps the orphaned dask group
            #     and respawns (0 disables the watchdog; mode 2).
            #   watchdog_max_respawns / watchdog_respawn_window: bound respawns to
            #     avoid a crash-respawn thrash loop; once exceeded the host is
            #     marked dead until restart_kernel.
            "parent_death_pipe": True,
            "watchdog_interval": 5.0,
            "watchdog_max_respawns": 3,
            "watchdog_respawn_window": 60.0,
        },
        "dask": {
            # Dask defaults to a kernel-local multi-process distributed cluster
            # (LocalCluster). This is the only configuration where cancel_job can
            # stop an in-flight .compute() mid-execution, and it gives real
            # (non-GIL) CPU parallelism. Set scheduler to "threads" /
            # "synchronous" for a low-overhead in-process scheduler (no
            # mid-compute cancel), or set address to attach to an external
            # scheduler instead of spinning a local one.
            "scheduler": "distributed",
            # n_workers for the auto-spun LocalCluster (0 -> let dask pick,
            # ~n_cores). When connecting to an external scheduler this is ignored.
            "num_workers": 0,
            # Non-empty -> connect to this external scheduler address; empty ->
            # spin a kernel-local LocalCluster.
            "address": "",
            # LocalCluster sizing (used only when spinning a local cluster).
            "threads_per_worker": 1,
            "memory_limit": "auto",
            # Bokeh dashboard bind address; loopback-only to match the server's
            # loopback-only security model. ":0" picks a free port.
            "dashboard_address": "127.0.0.1:0",
            # Cluster-wide chunk-cache budget for the data-plane client, split
            # evenly across dask workers (each worker caches budget //
            # n_workers). Bounds aggregate cache regardless of worker count --
            # the per-process client cache is otherwise replicated in every
            # worker. Accepts a human-readable size ("1G", "512M", "2GiB") or an
            # int (bytes), parsed with dask.utils.parse_bytes. Localhost servers
            # cache nothing regardless (the tensor server already caches); this
            # applies to remote.
            "cache_budget": "1G",
        },
        "tensor": {
            # Let the data-plane client cache chunks even for a *localhost*
            # tensor server. Translated into BIOPB_CACHE_LOCAL in the
            # kernel/worker env by __main__.py. By default the client skips a
            # localhost cache (the server already caches), but that means an
            # interactive viewer scrubbing planes re-fetches the whole enclosing
            # chunk per plane; caching makes repeated/overlapping reads instant.
            # Memory is bounded by the existing dask.cache_budget plugin (each
            # worker caps at budget // n_workers), so this does not reintroduce
            # the per-worker replication that the unbounded cache caused. A
            # structural fix that removes the underlying read amplification
            # (client-selectable read granularity) is tracked in biopb/biopb#8.
            "cache_local": True,
        },
        "viewer": {
            # Scheduler for the napari viewer's slice reads. The viewer scrubs
            # planes one at a time (serial np.asarray(data[slices])), but the
            # bootstrap makes a distributed LocalCluster the *default* scheduler,
            # so each single-chunk slice fetch scatters across a rotating worker
            # and the opaque per-worker chunk cache misses + replicates (issue
            # #8). Pinning the viewer's arrays to a single-process scheduler runs
            # every slice read in the kernel main process against the one shared
            # conn.client cache: ~100% hit on revisit, 1x memory, no scatter --
            # and costs no parallelism because the viewer is serial. The agent's
            # explicit `da` computes still use the distributed default. "threads"
            # keeps parallel multi-chunk reads within a slice; "synchronous" is
            # fully serial; "" disables wrapping (viewer slices compute on the
            # global default).
            "compute_scheduler": "threads",
        },
        "services": {
            # biopb.image ProcessImage servicer URLs (grpc:// or grpcs://).
            # Each is queried via GetOpNames and exposed as callables in the
            # agent kernel's `ops` dict.
            "process_image_servers": [],
        },
        # Give-up budget (seconds) for start_local_server's just-launched boot
        # wait, where connection-refused is tolerated while the server binds and
        # scans its data folder. The normal auto-connect path has no timeout: it
        # fails fast on a down server and polls a STARTING (scanning) server
        # indefinitely with progress feedback (issue #12).
        "server_start_timeout": 60.0,
    },
}


def get_default_config() -> dict:
    """Return a deep copy of the default configuration.

    Returns:
        Fresh copy of DEFAULT_CONFIG to prevent mutation.
    """
    return copy.deepcopy(DEFAULT_CONFIG)


_MISSING = object()


def get_setting(config: dict, path: str, default=_MISSING):
    """Read an absolute dotted *path* from *config*, else ``DEFAULT_CONFIG``.

    ``get_setting(config, "mcp.dask.scheduler")`` walks ``config`` by the dotted
    path; on a miss at any level it falls back to *default* if one was supplied,
    otherwise to the value at the same path in ``DEFAULT_CONFIG``. Mutable
    defaults are deep-copied so callers cannot alias (and mutate) the shared
    ``DEFAULT_CONFIG``. Centralizing the fallback here keeps each default
    declared exactly once -- call sites never restate a literal.

    Raises:
        KeyError: the path is absent from both *config* and ``DEFAULT_CONFIG``
            and no *default* was given (a programmer error, not a config issue).
    """
    node = config
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            break
    else:
        return node

    if default is not _MISSING:
        return default

    node = DEFAULT_CONFIG
    for key in path.split("."):
        node = node[key]
    return copy.deepcopy(node)


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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* in place, returning *base*.

    Nested dicts are merged key-by-key so a partial user section (e.g. only
    ``{"mcp": {"dask": {"num_workers": 4}}}``) overrides just that leaf and
    leaves its sibling defaults intact. Non-dict values (and dict-vs-non-dict
    mismatches) replace wholesale.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _migrate_legacy_keys(config: dict) -> dict:
    """Relocate the one legacy flat key the biopb installer still writes.

    The installer seeds ``mcp.process_image_servers`` at the old flat location;
    the schema now nests it under ``mcp.services``. Move it in place so an
    installer-written (or hand-edited pre-nesting) config keeps populating
    ``ops``. A new-style nested value, if also present, wins. Mutates and
    returns *config*; the next ``save_config`` then persists the nested form,
    so this self-heals over time.
    """
    mcp = config.get("mcp")
    if isinstance(mcp, dict) and "process_image_servers" in mcp:
        servers = mcp.pop("process_image_servers")
        services = mcp.setdefault("services", {})
        services.setdefault("process_image_servers", servers)
    return config


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

        # Back-compat for the installer-seeded flat key, then deep-merge with
        # defaults so partial nested user sections override only their own
        # leaves and every expected key still resolves.
        _migrate_legacy_keys(config)
        merged = _deep_merge(get_default_config(), config)

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
    prefix = "3d" if is_3d else "2d"
    grid_size = np.array(
        get_setting(config, f"widget.grid.{prefix}_size"), dtype=int
    )
    stride = np.array(
        get_setting(config, f"widget.grid.{prefix}_stride"), dtype=int
    )
    return grid_size, stride
