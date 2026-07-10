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
- ``pyramid``        -- multiscale pyramid construction knobs (``threshold`` /
                        ``downscale_factor``), shared by the browser widget and
                        MCP ``add_tensor``; GUI-independent like
                        ``tensor_browser``.
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
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Shared with the tensor server so the pyramid knobs (`pyramid.*`) are validated
# against the exact same rules in both packages -- they are the same bug, not
# just analogous (biopb/biopb#182, #34). `Range`/`Enum` are the format-agnostic
# constraint primitives; PYRAMID_CONSTRAINTS carries the three shared rows.
from biopb._config_constraints import PYRAMID_CONSTRAINTS, Enum, Range

logger = logging.getLogger(__name__)

# Platform-dependent defaults for the two kernel-bring-up knobs that Windows
# pays a structural penalty on. Windows has no fork(): dask's multi-process
# LocalCluster must *spawn* every worker -- a fresh interpreter that re-imports
# the whole numpy/dask/distributed stack cold -- where Linux/macOS fork near-
# free via copy-on-write. With num_workers=0 (dask picks ~n_cores) that
# spawn+reimport storm both lengthens kernel bring-up (risking startup_timeout)
# and multiplies memory. So on Windows we cap the worker count and widen the
# startup budget; POSIX keeps the lean defaults. A user config.json still
# overrides either leaf on any platform (deep-merge), so this only sets the
# floor for an unconfigured install.
_IS_WINDOWS = os.name == "nt"
_DEFAULT_STARTUP_TIMEOUT = 120.0 if _IS_WINDOWS else 60.0
_DEFAULT_DASK_NUM_WORKERS = 4 if _IS_WINDOWS else 0

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
    # Multiscale pyramid construction for large tensors, shared by the Tensor
    # Browser widget and the MCP `add_tensor` (both call
    # `_tensor_utils.build_pyramid_levels`). Top-level because it is a
    # GUI-independent data-plane concern, like `tensor_browser`.
    "pyramid": {
        # Don't build a pyramid unless an x/y dimension exceeds this size; also
        # the stop criterion -- levels are emitted until the coarsest one fits
        # within `threshold` in both x and y. So the coarsest level always lands
        # in (threshold // downscale_factor, threshold].
        "threshold": 4096,
        # Linear downscale between successive levels. A 4x step (vs the
        # conventional 2x) roughly halves the level count for a large image,
        # halving the per-level `get_tensor` round trips paid on every
        # `add_tensor` (the dominant cost of pyramid construction). Coarser than
        # 2x, so napari may read up to 4x more pixels per slice when zoom falls
        # between levels; napari infers level ratios from array shapes, so
        # non-2x steps are fine.
        "downscale_factor": 4,
        # Per-axis edge length of the coarsest level's 3D whole-volume read
        # (issue #29). napari's 3D mode reads the entire coarsest level (its last
        # three axes -- z, y, x) in one shot, so an XY-only pyramid OOMs on a
        # deep stack. The voxel budget is this value CUBED (Lx*Ly*Lz); we store
        # the cube root, not the product, so the floor `min(root, threshold)` and
        # the budget `root**3` are exact integers instead of a fragile
        # `round(product ** (1/3))`. Levels are emitted until the coarsest fits
        # BOTH the budget and `threshold` in x/y; x/y/z are downsampled
        # individually, each stopping at this floor so small axes (channels,
        # time, thin z) are never over-shrunk. Default 512 -> ~134M voxels.
        # Voxel count, not bytes -- multiply by the dtype itemsize for the
        # actual read size.
        "pixel_budget_cubic_root": 512,
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
            # Front-end transport: "http" (loopback streamable-http on `port`;
            # the real server) or "stdio" (for a client that spawns biopb-mcp as
            # a subprocess). stdio no longer serves MCP from this process: it
            # spawns and OWNS a private http session child on a *dynamic* port and
            # bridges stdin/stdout to it (de-daemonization,
            # docs/mcp-dedaemonization-migration.md). The default stays "stdio" so
            # installer-seeded client configs keep working unchanged.
            # Overridable per-launch with `--transport`.
            "kind": "stdio",
            # Fixed loopback port for the http server. NOTE: since
            # de-daemonization the stdio shim no longer uses it (its owned child
            # binds a dynamic port), and `biopb mcp view` is dynamic too — so this
            # now applies only to a directly-launched `biopb-mcp --transport http`
            # and the *deprecated* `biopb mcp start` daemon. The observe UI shares
            # this port when one is fixed.
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
            # Force the stdio bridge's session child to log to ONE fixed file
            # instead of the default per-session file. Empty (default) -> each
            # shim session gets its own ~/.local/share/biopb-mcp/log/sessions/
            # <id>.log (pruned to session_log_keep), so concurrent sessions don't
            # interleave. Set a path here to send every session to that single
            # file instead (the pre-de-daemonization behavior). Unused when
            # launched directly with --transport http (output stays on the
            # launching terminal/service manager) or by the deprecated
            # `biopb mcp start` (the CLI redirects to mcp-server.log itself).
            "kernel_log": "",
            # How many per-session shim logs to keep under log/sessions/ (newest
            # by mtime); older ones are pruned on each new session. Ignored when
            # kernel_log forces a single shared file.
            "session_log_keep": 5,
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
            # 60.0 on POSIX; 120.0 on Windows, where the cold spawn+reimport of
            # dask workers makes bring-up legitimately slower (see
            # _DEFAULT_STARTUP_TIMEOUT).
            "startup_timeout": _DEFAULT_STARTUP_TIMEOUT,
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
            # kernel (and, under mcp.dask.owner="kernel", the cluster it spawned;
            # the default daemon-owned cluster is reaped separately by the daemon).
            #   parent_death_pipe: kernel inherits a pipe read-end and group-kills
            #     itself when the launcher *process* dies (POSIX only; mode 1).
            #   watchdog_interval: seconds between liveness polls; on an
            #     unexpected kernel death the host reaps the orphaned process
            #     group and respawns (0 disables the watchdog; mode 2).
            #   watchdog_max_respawns / watchdog_respawn_window: bound respawns to
            #     avoid a crash-respawn thrash loop; once exceeded the host is
            #     marked dead until restart_kernel.
            "parent_death_pipe": True,
            "watchdog_interval": 5.0,
            "watchdog_max_respawns": 3,
            "watchdog_respawn_window": 60.0,
        },
        "dask": {
            # Dask defaults to a multi-process distributed cluster (LocalCluster)
            # owned by the MCP *daemon* (owner="daemon"); the kernel attaches to
            # it via an injected scheduler address, so the cluster survives kernel
            # restarts (no cold worker re-spawn -- the dominant restart cost on
            # Windows). Any distributed mode lets cancel_job stop an in-flight
            # .compute() and gives real (non-GIL) CPU parallelism. Set scheduler
            # to "threads"/"synchronous" for a low-overhead in-process scheduler
            # (no mid-compute cancel), or set address to attach to an external
            # scheduler instead of spinning a local one.
            "scheduler": "distributed",
            # Who owns the auto-spun LocalCluster. "daemon" (default): the MCP
            # daemon owns it, so it outlives kernel restart/respawn/window-close.
            # "kernel": the legacy behavior where each kernel spins (and tears
            # down) its own cluster. Ignored when scheduler is not "distributed"
            # or an external address is set (the kernel attaches to that).
            "owner": "daemon",
            # n_workers for the auto-spun LocalCluster (0 -> let dask pick,
            # ~n_cores). When connecting to an external scheduler this is ignored.
            # 0 on POSIX (fork makes ~n_cores workers cheap); capped at 4 on
            # Windows, where each worker is a cold spawn (see
            # _DEFAULT_DASK_NUM_WORKERS).
            "num_workers": _DEFAULT_DASK_NUM_WORKERS,
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
            # NOTE: the localhost client-cache decision now lives entirely in the
            # tensor client (biopb.tensor.client._resolve_cache_bytes): localhost
            # gets no client cache by default (the server caches and the
            # cache-file mmap fast path makes a re-fetch cheap), with
            # BIOPB_CACHE_LOCAL=1 as biopb's own opt-out for e.g. a slow loopback
            # proxy. The MCP kernel no longer overrides that default.
            # Background source-catalog watcher (issue #44). A daemon thread in
            # the kernel periodically health-checks the server and re-lists
            # sources when its source_count changes, so a catalog cached while
            # the server was still indexing (a partial scene list) self-heals
            # without a manual refresh. The poll interval backs off
            # exponentially from health_poll_min_interval to
            # health_poll_max_interval while the count is stable, and snaps back
            # to the min on a change so active indexing is tracked promptly. 0
            # for health_poll_min_interval disables the watcher.
            "health_poll_min_interval": 2.0,
            "health_poll_max_interval": 60.0,
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
            # Fetch napari slices off the Qt main thread (napari experimental
            # async slicing). A zoom into a not-yet-cached pyramid level can
            # take seconds (a cold full-res tile read); synchronous slicing
            # freezes the whole viewer for that long, while async slicing keeps
            # the current (coarse, upscaled) texture on screen until the finer
            # slice resolves. `take_screenshot` force-syncs a slice before
            # capturing (see mcp/_helpers.resync_view_for_capture) so the agent
            # still sees exactly the frame it requested. Set False to keep the
            # old fully-synchronous slicing.
            "async_slicing": True,
        },
        "services": {
            # biopb.image ProcessImage servicer URLs (grpc:// or grpcs://).
            # Each is queried via GetOpNames and exposed as callables in the
            # agent kernel's `ops` dict.
            "process_image_servers": [],
            # Curated skills catalog (docs/skill-interface.md). The find_skills
            # tool + skill://<id> resource fetch this catalog and lazily fetch
            # skill bodies; both fail open to an on-disk cache, then a bundled
            # snapshot, so they never block or crash when offline.
            "skills": {
                # Master switch for discovery/retrieval. When False, find_skills
                # returns nothing and no catalog fetch is attempted.
                "enabled": True,
                # Published metadata catalog. Point at a self-hosted catalog to
                # serve a lab's own curated set (an ordered multi-source model is
                # a later addition, see skill-interface.md §7.5).
                "catalog_url": "https://biopb.org/skills/catalog.json",
                # Seconds a fetched catalog is reused before re-fetching. A stale
                # on-disk cache is still used past this if the network is down.
                "cache_ttl": 3600,
            },
        },
        "observe": {
            # Minimal loopback web UI for watching execute_code job history and
            # cancelling/interrupting/restarting the current execution. **http
            # transport only**: it mounts on the existing MCP app (sharing
            # `mcp.transport.port` and the MCP event loop), so it adds no network
            # surface beyond `/mcp` — same loopback bind, same Host/Origin
            # allowlist (the kernel is already RCE over that port). It is
            # therefore **on by default (opt-out)** in http; set False to
            # disable. It is never available under stdio — a second HTTP server
            # in the stdio launcher process would risk the fd-1 JSON-RPC channel
            # and race the kernel — so under stdio it is silently skipped.
            "enabled": True,
            # Detail-view stdout cap (chars). The tail is kept (most recent
            # output matters for a running job) with a truncation marker; the
            # full length is reported alongside.
            "max_output_chars": 20000,
            # How often (ms) the page polls the job list / status. Deliberately
            # slow: each poll is a kernel round-trip competing with agent calls.
            "poll_interval_ms": 3000,
        },
        "update": {
            # Kernel-start auto-updater (issue #87). When a napari window exists,
            # the kernel checks once — in the background, fail-open — whether a
            # newer biopb release-v* *deployment* is available than the one the
            # installer recorded (~/.config/biopb/release.version), and offers to
            # re-run the installer. The check never blocks or crashes the viewer.
            "enabled": True,
            # owner/name of the deployment repo whose release-v* line is the
            # product. Overridable for forks / testing; the release line prefix
            # ("release-v") is fixed (see docs/release-model.md).
            "repo": "biopb/biopb",
            # "stable"     -> newest clean release-vX.Y.Z (prereleases skipped,
            #                 matching the installer's default).
            # "prerelease" -> also consider release candidates (…a/b/rc).
            "channel": "stable",
            # A version the user chose to skip ("Skip vX.Y.Z"): suppresses the
            # prompt for exactly that version. Empty -> nothing skipped.
            "skipped_version": "",
            # Per-request network timeout (seconds) for the GitHub API fetch.
            # Short on purpose — the check must never delay viewer start.
            "timeout": 5.0,
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


def get_log_dir() -> Path:
    """Get the log directory (~/.local/share/biopb-mcp/log on all platforms).

    Logs are persistent runtime state, not user-editable config, so they live
    under the XDG *data* tree rather than ~/.config — matching the biopb tensor
    server, which writes to ~/.local/share/biopb/log.

    Returns:
        Path to the log directory for biopb-mcp.
    """
    log_dir = Path.home() / ".local" / "share" / "biopb-mcp" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_session_log_dir() -> Path:
    """Directory for per-session stdio-shim logs (<log dir>/sessions).

    Each shim-owned session (de-daemonization Layer 1) writes its own logfile
    here rather than the shared ``mcp-server.log``, so concurrent sessions never
    interleave; retention (``mcp.transport.session_log_keep``) prunes it to the
    newest N. The standalone ``biopb mcp start`` daemon keeps using
    ``mcp-server.log`` — a separate domain.
    """
    d = get_log_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_daemon_log_file(config: Optional[dict] = None) -> Path:
    """Path of the http daemon's combined stdout/stderr log.

    The daemon logs to stderr (``logging.basicConfig`` in ``biopb_mcp.mcp``);
    whoever launches it redirects that stream into this file. A *single*
    canonical location so every launcher (the stdio shim's ``spawn_session``,
    the ``biopb mcp start`` CLI, a manual ``python -m biopb_mcp.mcp``) and every
    reader (``biopb mcp logs`` / ``status``) agree on one file regardless of who
    started the daemon -- otherwise the launchers disagree on the filename (a
    shim-spawned daemon and a ``mcp start`` CLI writing different files) and one
    reader wrongly reports the server "never started". Honors
    ``mcp.transport.kernel_log`` if set, else ``<log dir>/mcp-server.log``.
    ``config`` is loaded (cached singleton) when not supplied; the shim passes
    the dict it already holds.
    """
    if config is None:
        config = load_config()
    override = get_setting(config, "mcp.transport.kernel_log")
    if override:
        return Path(override)
    return get_log_dir() / "mcp-server.log"


def get_pid_file() -> Path:
    """Path to the http daemon's PID file (~/.local/share/biopb-mcp/mcp-server.pid).

    The running http daemon writes its own PID here once it is up, regardless of
    who launched it (the `biopb mcp` CLI, the stdio shim, or a manual
    `python -m biopb_mcp.mcp`), so `biopb mcp status` can detect it uniformly.
    The biopb CLI hardcodes the same path (``biopb.cli.MCP_PID_FILE``); keep the
    two in sync.
    """
    return Path.home() / ".local" / "share" / "biopb-mcp" / "mcp-server.pid"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* in place, returning *base*.

    Nested dicts are merged key-by-key so a partial user section (e.g. only
    ``{"mcp": {"dask": {"num_workers": 4}}}``) overrides just that leaf and
    leaves its sibling defaults intact. Non-dict values (and dict-vs-non-dict
    mismatches) replace wholesale.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
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


# --- Declarative config validation (biopb/biopb#182) --------------------------
#
# The merged dict is otherwise trusted: a valid-JSON but out-of-range leaf passes
# straight through and only blows up on a hot path -- downscale_factor=1 -> a
# single full-res level (no pyramid); pixel_budget_cubic_root=0 -> an infinite
# get_tensor loop in build_pyramid_levels; a JSON string "4" -> a TypeError deep
# in `sx * downscale_factor`. Validate here, keyed by dotted path, reusing the
# tensor server's constraint primitives so the shared `pyramid.*` rows are judged
# by the exact same rule in both packages and cannot drift (#182, #34).
#
# Severity per #182: the deep-merge already degrades gracefully, so a bad leaf is
# a logged WARNING, not a startup failure. The offending leaf is reset to its
# known-good DEFAULT_CONFIG value so the config stays usable -- which also
# neutralizes the no-coercion wrinkle (a string where a number is expected fails
# its Range check and is replaced by the numeric default).
_CONSTRAINTS = {
    # Pyramid knobs -- the identical rows the tensor server enforces.
    **{f"pyramid.{field}": rule for field, rule in PYRAMID_CONSTRAINTS.items()},
    # MCP server transport / kernel / dask / tensor knobs.
    "mcp.transport.kind": Enum({"http", "stdio"}),
    "mcp.transport.port": Range(min=1, max=65535),
    "mcp.transport.display_mode": Enum({"auto", "visible", "headless"}),
    "mcp.transport.session_log_keep": Range(min=1),  # keep at least the current
    "mcp.kernel.startup_timeout": Range(exclusive_min=0),
    "mcp.kernel.execute_timeout": Range(exclusive_min=0),
    "mcp.kernel.busy_lock_timeout": Range(exclusive_min=0),
    "mcp.kernel.promote_after": Range(exclusive_min=0),
    "mcp.kernel.watchdog_interval": Range(min=0),  # 0 disables the watchdog
    "mcp.kernel.watchdog_max_respawns": Range(min=0),
    "mcp.kernel.watchdog_respawn_window": Range(min=0),
    "mcp.dask.scheduler": Enum({"distributed", "threads", "synchronous"}),
    "mcp.dask.owner": Enum({"daemon", "kernel"}),
    "mcp.dask.num_workers": Range(min=0),  # 0 -> dask picks ~n_cores
    "mcp.tensor.health_poll_min_interval": Range(min=0),  # 0 disables the watcher
    "mcp.tensor.health_poll_max_interval": Range(min=0),
    "mcp.services.skills.cache_ttl": Range(min=0),
    # Compute-plane timeouts / limits shared by the demo widgets and `ops`.
    "timeout.health_check": Range(exclusive_min=0),
    "timeout.get_op_names": Range(exclusive_min=0),
    "timeout.detection_2d": Range(exclusive_min=0),
    "timeout.detection_3d": Range(exclusive_min=0),
    "timeout.process_image": Range(exclusive_min=0),
    "grpc.max_message_size_mb": Range(exclusive_min=0),
    "grpc.max_concurrent_calls": Range(min=1),
    "memory.warn_threshold_mb": Range(exclusive_min=0),
    "memory.error_threshold_mb": Range(exclusive_min=0),
}


def _walk_path(node: dict, keys):
    """Follow dotted-path *keys* into *node*; return ``_MISSING`` on any miss."""
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _validate_and_clamp(config: dict) -> dict:
    """Warn on each out-of-range leaf and reset it to its default, in place.

    A leaf absent from the merged dict is skipped (nothing to check); a
    present-but-invalid leaf is logged and replaced with its ``DEFAULT_CONFIG``
    value so the config stays usable (biopb/biopb#182). Returns *config*.
    """
    for path, constraint in _CONSTRAINTS.items():
        keys = path.split(".")
        value = _walk_path(config, keys)
        if value is _MISSING or constraint.ok(value):
            continue
        default = _walk_path(DEFAULT_CONFIG, keys)
        logger.warning(
            "Invalid config value %s=%r (expected %s); using default %r. "
            "See biopb/biopb#182.",
            path,
            value,
            constraint.describe(),
            default,
        )
        if default is _MISSING:
            continue
        # value was found, so every intermediate on the path is a present dict.
        parent = config
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = copy.deepcopy(default)

    # Cross-field: the health-poll interval must not invert (min > max), which
    # would break the exponential backoff. Reset both to defaults if it does.
    lo_path = ["mcp", "tensor", "health_poll_min_interval"]
    hi_path = ["mcp", "tensor", "health_poll_max_interval"]
    lo, hi = _walk_path(config, lo_path), _walk_path(config, hi_path)
    if lo is not _MISSING and hi is not _MISSING and lo > hi:
        logger.warning(
            "Invalid config: mcp.tensor.health_poll_min_interval=%r > "
            "health_poll_max_interval=%r; using defaults. See biopb/biopb#182.",
            lo,
            hi,
        )
        tensor = config["mcp"]["tensor"]
        tensor["health_poll_min_interval"] = copy.deepcopy(
            _walk_path(DEFAULT_CONFIG, lo_path)
        )
        tensor["health_poll_max_interval"] = copy.deepcopy(
            _walk_path(DEFAULT_CONFIG, hi_path)
        )
    return config


def _read_and_merge_from_disk() -> dict:
    """Read the config file, migrate legacy keys, and merge onto defaults.

    Returns a fully-merged config dict with every expected key present. A
    missing, malformed, or unreadable file falls back to ``get_default_config()``
    (logged). This is the single code path from disk into memory; the ``CONFIG``
    singleton and :func:`load_config` both route through it.
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
        # Reject out-of-range / bad-enum leaves (warn + reset to default) before
        # any hot path reads them (biopb/biopb#182).
        _validate_and_clamp(merged)

        logger.debug("Loaded config from %s", config_path)
        return merged

    except json.JSONDecodeError as e:
        logger.warning("Config file malformed, using defaults: %s", e)
        return get_default_config()
    except Exception as e:
        logger.warning("Failed to load config, using defaults: %s", e)
        return get_default_config()


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp file + ``os.replace``).

    Writing to a sibling temp file and renaming over the target means a reader
    never sees a half-written config: the rename is atomic on POSIX and Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process *and* thread so two concurrent writers never collide on
    # the temp file (CONFIG._save serializes them under the lock, but this keeps
    # the helper safe if called directly, e.g. via save_config).
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        logger.debug("Saved config to %s", path)
    except Exception as e:
        logger.warning("Failed to save config: %s", e)
        try:
            tmp.unlink()
        except OSError:
            pass


class _Config:
    """Process-wide config singleton: lazy cached read, write-through to disk.

    One instance (:data:`CONFIG`) is the single runtime source of truth. The
    first access reads disk once (via :func:`_read_and_merge_from_disk`) and
    caches the merged dict for the process lifetime; :meth:`reload` invalidates
    it. Every read routes through :meth:`get` and every write through
    :meth:`set`, so nothing else touches the file or re-merges defaults.

    Thread-safe: the MCP kernel runs agent code in background threads that may
    read config, so lazy init / set / reload take an ``RLock``.
    """

    def __init__(self) -> None:
        self._data: dict | None = None
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._data is None:
                self._data = _read_and_merge_from_disk()

    def get(self, path: str, default=_MISSING):
        """Read a dotted *path*, falling back to ``DEFAULT_CONFIG``.

        Same contract as :func:`get_setting` (the ambient form of it): on a miss
        at any level, returns *default* if given, else the ``DEFAULT_CONFIG``
        value, else raises ``KeyError``.

        Holds the lock across the dotted-path walk so a read is never serialized
        against a concurrent :meth:`set` / :meth:`reload`: without it a reader
        could observe a half-applied multi-key write (a created-but-empty
        intermediate section) or a momentarily-``None`` cache mid-reload, and
        silently fall back to the default. ``RLock`` makes the nested
        :meth:`_ensure_loaded` acquisition re-entrant.
        """
        with self._lock:
            self._ensure_loaded()
            return get_setting(self._data, path, default)

    def set(self, path: str, value, *, persist: bool = True) -> None:
        """Set a dotted *path* in the cache; write through to disk by default.

        Walks/creates intermediate sections, sets the leaf, and (when *persist*)
        atomically rewrites the file so cache and disk stay in lockstep. Pass
        ``persist=False`` to batch several sets, then call :meth:`save` once.
        """
        self._ensure_loaded()
        with self._lock:
            node = self._data
            keys = path.split(".")
            for key in keys[:-1]:
                child = node.get(key)
                if not isinstance(child, dict):
                    child = {}
                    node[key] = child
                node = child
            node[keys[-1]] = value
            if persist:
                self._save()

    def save(self) -> None:
        """Persist the current cached config to disk (single atomic write)."""
        self._ensure_loaded()
        with self._lock:
            self._save()

    def reload(self) -> None:
        """Invalidate the cache; the next read re-reads disk.

        For tests, external edits, or a future SIGHUP-style reload.
        """
        with self._lock:
            self._data = None

    def as_dict(self) -> dict:
        """Return the *live* cached merged dict (not a copy).

        Escape hatch for code that threads the raw dict (the MCP bootstrap, the
        explicit-dict ``get_setting`` form). Callers must treat it as read-only;
        all mutation goes through :meth:`set`.

        Unlike :meth:`get`, this returns a reference the caller dereferences
        *outside* the lock, so it is **not** safe to hold across a concurrent
        :meth:`set` -- use it for startup/single-threaded threading (the
        bootstrap) and prefer :meth:`get` for thread-safe ambient reads.
        """
        self._ensure_loaded()
        return self._data

    def _save(self) -> None:
        _atomic_write_json(get_config_path(), self._data)


# The one config instance in the process. All reads/writes route through it.
CONFIG = _Config()


def load_config() -> dict:
    """Return the process config dict (cached singleton).

    Back-compat shim over :data:`CONFIG`: returns the live merged dict so callers
    that thread a dict (e.g. the MCP bootstrap) keep working. The result is the
    shared cached instance -- treat it read-only and write via ``CONFIG.set``.
    """
    return CONFIG.as_dict()


def save_config(config: dict) -> None:
    """Persist *config* to disk and refresh the singleton cache.

    Back-compat shim: atomically writes the given dict, then invalidates the
    cache so the next read re-merges it from disk (avoids aliasing an external
    dict into the cache). Prefer ``CONFIG.set(path, value)`` for targeted writes.

    Args:
        config: Configuration dict to save.
    """
    _atomic_write_json(get_config_path(), config)
    CONFIG.reload()


def get_grid_params(is_3d: bool, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Get grid size and stride from config.

    Args:
        is_3d: Whether processing 3D data.
        config: Configuration dict.

    Returns:
        Tuple of (grid_size, stride) as numpy arrays.
    """
    prefix = "3d" if is_3d else "2d"
    grid_size = np.array(get_setting(config, f"widget.grid.{prefix}_size"), dtype=int)
    stride = np.array(get_setting(config, f"widget.grid.{prefix}_stride"), dtype=int)
    return grid_size, stride
