"""Configuration management for biopb-mcp.

The config is defined as **flat dataclasses** (one per section) with each field's
help in ``field(metadata={"help": ...})`` and its validation rule in the
class-keyed :data:`_CONSTRAINTS` table -- the same machinery the tensor server
uses, so both project a JSON Schema from one source of truth (biopb/biopb#34) and
a schema-driven admin editor can render either. ``DEFAULT_CONFIG`` is just
``asdict(McpConfig())``, so the dataclass defaults are the only place a default
literal lives.

Config lives at ``~/.config/biopb/mcp-config.json`` -- co-located with the
tensor server's ``biopb.json`` and the installer's client-definition ``mcp.json``
(a *distinct* file: that one registers biopb-mcp with MCP clients; this one is
biopb-mcp's own runtime settings). The XDG *data* tree
(``~/.local/share/biopb-mcp``) still holds logs/pid -- runtime state, not config.

Sections are flat (no ``mcp.``/``widget.`` wrapper): ``transport`` / ``kernel`` /
``dask`` / ``tensor`` / ``viewer`` / ``services`` / ``observe`` / ``update`` are
the MCP-server knobs; ``widget`` / ``detection`` / ``grid`` are the demo napari
widgets (``image_processing/``); ``tensor_browser`` / ``pyramid`` are
GUI-independent data-plane knobs read by the headless kernel too; ``timeout`` /
``grpc`` / ``memory`` are compute-plane knobs shared by the widgets and ``ops``.

Read settings with :func:`get_setting`, which falls back to ``DEFAULT_CONFIG`` so
call sites never duplicate a default literal.
"""

import copy
import dataclasses
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Shared with the tensor server: the constraint primitives (so the pyramid knobs
# are validated by the exact same rules in both packages -- the same bug, not an
# analogous one; biopb/biopb#182, #34) and the per-field schema projection.
from biopb._config_constraints import PYRAMID_CONSTRAINTS, Enum, Range

logger = logging.getLogger(__name__)

# Platform-dependent defaults for the two kernel-bring-up knobs Windows pays a
# structural penalty on. Windows has no fork(): dask's multi-process LocalCluster
# must *spawn* every worker -- a fresh interpreter re-importing the whole
# numpy/dask/distributed stack cold -- where POSIX forks near-free via
# copy-on-write. With num_workers=0 (dask picks ~n_cores) that spawn+reimport
# storm both lengthens bring-up (risking startup_timeout) and multiplies memory.
# So Windows caps the worker count and widens the startup budget; POSIX keeps the
# lean defaults. A user config still overrides either leaf on any platform.
_IS_WINDOWS = os.name == "nt"
_DEFAULT_STARTUP_TIMEOUT = 120.0 if _IS_WINDOWS else 60.0
_DEFAULT_DASK_NUM_WORKERS = 4 if _IS_WINDOWS else 0


def _h(default, help_text, **kw):
    """A dataclass field carrying user-facing help (-> schema description)."""
    return field(default=default, metadata={"help": help_text}, **kw)


def _hlist(default_list, help_text):
    """A list-valued field with help (defaults are copied per instance)."""
    return field(
        default_factory=lambda: list(default_list), metadata={"help": help_text}
    )


# --- Section dataclasses ------------------------------------------------------
# Each maps 1:1 to a top-level section in mcp-config.json; every field's `help`
# becomes its schema `description` (the single source of truth). List fields are
# added to the schema by the composer in _config_schema.py.


@dataclass
class WidgetConfig:
    """Settings for the experimental napari demo widgets (image_processing/).

    Only those widgets read this section; the MCP server does not.
    """

    server_url: str = _h(
        "localhost:50051", "ProcessImage server the demo widgets target."
    )
    is_3d: bool = _h(
        False, "Whether the demo widgets operate in 3D mode (persisted toggle)."
    )


@dataclass
class DetectionConfig:
    """Object-detection parameters for the demo widgets."""

    min_score: float = _h(0.4, "Minimum detection score to keep a predicted object.")
    size_hint: float = _h(
        32.0, "Approximate object size (px) hint passed to the detector."
    )
    nms: str = _h("Off", "Non-maximum-suppression mode for overlapping detections.")
    z_aspect_ratio: float = _h(1.0, "Z-vs-XY voxel aspect ratio for 3D detection.")


@dataclass
class GridConfig:
    """Tiling grid for large-image detection (size/stride per 2D and 3D mode).

    Each is a per-axis pixel vector; stride < size gives overlap between tiles.
    """

    size_2d: List[int] = _hlist([4096, 4096], "2D tile size [y, x] (px).")
    stride_2d: List[int] = _hlist(
        [4000, 4000], "2D tile stride [y, x] (px); < size overlaps tiles."
    )
    size_3d: List[int] = _hlist([64, 512, 512], "3D tile size [z, y, x] (px).")
    stride_3d: List[int] = _hlist(
        [48, 480, 480], "3D tile stride [z, y, x] (px); < size overlaps tiles."
    )


@dataclass
class TensorBrowserConfig:
    """Data-plane connection, read by the GUI-independent TensorConnection.

    Not under ``widget`` on purpose: the headless MCP kernel uses it too.
    """

    server_url: str = _h(
        "grpc://localhost:8815", "Arrow Flight tensor-server URL for the data plane."
    )


@dataclass
class PyramidConfig:
    """Multiscale pyramid construction for large tensors.

    Shared by the Tensor Browser widget and MCP ``add_tensor`` (both call
    ``_tensor_utils.build_pyramid_levels``). GUI-independent, like
    ``tensor_browser``. The numeric bounds are the identical rows the tensor
    server enforces (PYRAMID_CONSTRAINTS), so the two cannot drift.
    """

    threshold: int = _h(
        4096,
        "Build a pyramid only if an x/y dimension exceeds this size; also the stop "
        "criterion (coarsest level fits within threshold in x and y).",
    )
    downscale_factor: int = _h(
        4,
        "Linear downscale between successive levels. 4x (vs 2x) roughly halves the "
        "level count and the per-level get_tensor round trips; napari infers level "
        "ratios from shapes, so non-2x is fine.",
    )
    pixel_budget_cubic_root: int = _h(
        512,
        "Per-axis edge length of the coarsest level's 3D whole-volume read (#29); "
        "the voxel budget is this value cubed. Default 512 -> ~134M voxels.",
    )


@dataclass
class TimeoutConfig:
    """Per-call gRPC timeouts (seconds) for the compute plane."""

    health_check: float = _h(5.0, "Timeout for a server health check.")
    get_op_names: float = _h(10.0, "Timeout for listing a server's op names.")
    detection_2d: int = _h(15, "Timeout for a 2D detection call.")
    detection_3d: int = _h(300, "Timeout for a 3D detection call.")
    process_image: int = _h(300, "Timeout for a ProcessImage call.")


@dataclass
class GrpcConfig:
    """gRPC channel limits for the compute plane."""

    max_message_size_mb: int = _h(512, "Max gRPC message size (MB) for send/receive.")
    max_concurrent_calls: int = _h(4, "Max concurrent gRPC calls to a server.")


@dataclass
class MemoryConfig:
    """Chunk-size guardrails for eager transfers."""

    warn_threshold_mb: int = _h(
        500, "Log a warning when a single chunk exceeds this size (MB)."
    )
    error_threshold_mb: int = _h(
        2000, "Raise MemoryError when a single chunk exceeds this size (MB)."
    )


@dataclass
class TransportConfig:
    """The MCP server's front-end transport and its network guards."""

    kind: str = _h(
        "stdio",
        'Front-end transport: "http" (loopback streamable-http on `port`) or '
        '"stdio" (the client spawns biopb-mcp; the shim owns a private http '
        "session child on a dynamic port and bridges stdin/stdout to it).",
    )
    port: int = _h(
        8765,
        "Fixed loopback port for the http server. Applies only to a directly-"
        "launched `--transport http` and the deprecated `biopb mcp start` daemon "
        "(the stdio shim and `biopb mcp view` use dynamic ports).",
    )
    display_mode: str = _h(
        "auto",
        'Whether the kernel opens a visible napari viewer: "auto" (visible if a '
        'display is present, else headless), "visible" (require a display; fail '
        'fast if none), "headless" (never open a viewer -- compute-only).',
    )
    kernel_log: str = _h(
        "",
        "Force the stdio bridge's session child to log to ONE fixed file instead "
        "of the default per-session file. Empty -> each session gets its own log.",
    )
    session_log_keep: int = _h(
        5,
        "How many per-session shim logs to keep (newest by mtime); older ones are "
        "pruned on each new session. Ignored when kernel_log forces a shared file.",
    )
    allowed_origins: List[str] = _hlist(
        [],
        "Extra Origin header values appended to the loopback allowlist guarding "
        "against DNS-rebinding / cross-origin browser requests. http transport only.",
    )
    allowed_hosts: List[str] = _hlist(
        [],
        "Extra Host header values appended to the loopback allowlist. Set only when "
        "fronting the server with a reverse proxy. http transport only.",
    )
    server_start_timeout: float = _h(
        60.0,
        "Give-up budget (seconds) applied twice by auto_connect's down-plane "
        "fallback (control-ensure, then boot wait), so the worst-case wall wait is "
        "~2x this. The normal path (plane already up) has no timeout.",
    )


@dataclass
class KernelConfig:
    """The child Jupyter kernel that runs agent code (separate process)."""

    name: str = _h("python3", "Jupyter kernel name to launch.")
    startup_timeout: float = _h(
        _DEFAULT_STARTUP_TIMEOUT,
        "Seconds to wait for kernel bring-up. 60 on POSIX; 120 on Windows, where "
        "the cold spawn+reimport of dask workers makes bring-up legitimately slower.",
    )
    execute_timeout: float = _h(
        120.0,
        "Bounds only the quick in-band kernel snippets (screenshot / status / "
        "inspect / job submit+poll), not long jobs -- execute_code runs agent code "
        "in a background thread that may run indefinitely.",
    )
    busy_lock_timeout: float = _h(
        5.0, "Seconds to wait for the kernel RLock before reporting busy."
    )
    promote_after: float = _h(
        10.0,
        "Seconds execute_code waits before promoting a job: completes within this "
        "window -> inline result; otherwise a job handle is returned and it runs on.",
    )
    parent_death_pipe: bool = _h(
        True,
        "Kernel inherits a pipe read-end and group-kills itself when the launcher "
        "process dies (POSIX only; orphan hardening, #13).",
    )
    watchdog_interval: float = _h(
        5.0,
        "Seconds between kernel liveness polls; on an unexpected death the host "
        "reaps the orphaned process group and respawns. 0 disables the watchdog.",
    )
    watchdog_max_respawns: int = _h(
        3, "Max respawns within the window before the host is marked dead."
    )
    watchdog_respawn_window: float = _h(
        60.0, "Sliding window (seconds) over which respawns are counted."
    )


@dataclass
class DaskConfig:
    """Dask scheduler / cluster for the kernel's compute."""

    scheduler: str = _h(
        "distributed",
        'Dask scheduler: "distributed" (a LocalCluster; enables mid-compute '
        'cancel and real CPU parallelism), "threads"/"synchronous" (low-overhead '
        "in-process, no mid-compute cancel).",
    )
    owner: str = _h(
        "daemon",
        'Who owns the auto-spun LocalCluster. "daemon": the MCP daemon owns it, so '
        'it outlives kernel restart. "kernel": each kernel spins/tears down its own. '
        "Ignored unless scheduler is distributed with no external address.",
    )
    num_workers: int = _h(
        _DEFAULT_DASK_NUM_WORKERS,
        "n_workers for the auto-spun LocalCluster (0 -> dask picks ~n_cores). 0 on "
        "POSIX (fork is cheap); capped at 4 on Windows (each worker is a cold spawn).",
    )
    address: str = _h(
        "",
        "Non-empty -> connect to this external scheduler address; empty -> spin a "
        "kernel-local LocalCluster.",
    )
    threads_per_worker: int = _h(
        1, "LocalCluster threads per worker (local cluster only)."
    )
    memory_limit: str = _h(
        "auto", "LocalCluster per-worker memory limit (local cluster only)."
    )
    dashboard_address: str = _h(
        "127.0.0.1:0",
        "Bokeh dashboard bind address; loopback-only to match the server's security "
        'model. ":0" picks a free port.',
    )
    cache_budget: str = _h(
        "1G",
        "Cluster-wide chunk-cache budget for the data-plane client, split evenly "
        "across workers. Human size (1G/512M/2GiB) or int bytes. Localhost servers "
        "cache nothing regardless; this applies to remote.",
    )


@dataclass
class TensorRuntimeConfig:
    """Background source-catalog watcher in the kernel (#44)."""

    health_poll_min_interval: float = _h(
        2.0,
        "Min poll interval (s) for the source-catalog watcher; it re-lists sources "
        "when the server's source_count changes so a partial catalog self-heals. "
        "The interval backs off to the max while stable, snaps back on a change. "
        "0 disables the watcher.",
    )
    health_poll_max_interval: float = _h(
        60.0,
        "Max poll interval (s) the watcher backs off to while the catalog is stable.",
    )


@dataclass
class ViewerConfig:
    """napari viewer slice-read behavior in the kernel."""

    compute_scheduler: str = _h(
        "threads",
        "Scheduler for the viewer's serial plane reads. Pinning to a single-process "
        'scheduler ("threads"/"synchronous") uses the one shared client cache (~100% '
        'hit on revisit, no worker scatter; #8); "" computes on the global default.',
    )
    async_slicing: bool = _h(
        True,
        "Fetch napari slices off the Qt main thread (async slicing), so a cold "
        "tile read doesn't freeze the viewer. take_screenshot force-syncs before "
        "capturing. False keeps fully-synchronous slicing.",
    )


@dataclass
class ServicesConfig:
    """Compute-plane servers and the skills catalog wired into the kernel."""

    process_image_servers: List[str] = _hlist(
        [],
        "biopb.image ProcessImage servicer URLs (grpc:// or grpcs://). Each is "
        "queried via GetOpNames and exposed as callables in the kernel's `ops` dict.",
    )
    skills_enabled: bool = _h(
        True,
        "Master switch for skills discovery/retrieval. When off, find_skills returns "
        "nothing and no catalog fetch is attempted.",
    )
    skills_catalog_url: str = _h(
        "https://biopb.org/skills/catalog.json",
        "Published skills metadata catalog. Point at a self-hosted catalog to serve "
        "a lab's own curated set.",
    )
    skills_cache_ttl: int = _h(
        3600,
        "Seconds a fetched skills catalog is reused before re-fetching. A stale "
        "on-disk cache is still used past this if the network is down.",
    )


@dataclass
class ObserveConfig:
    """Minimal loopback web UI for watching execute_code job history (http only)."""

    enabled: bool = _h(
        True,
        "Enable the observe UI. http transport only (it shares the MCP port and "
        "event loop, adding no network surface); silently skipped under stdio.",
    )
    max_output_chars: int = _h(
        20000,
        "Detail-view stdout cap (chars); the tail is kept with a truncation marker "
        "and the full length is reported alongside.",
    )
    poll_interval_ms: int = _h(
        3000,
        "How often (ms) the observe page polls the job list/status. Deliberately "
        "slow: each poll is a kernel round-trip competing with agent calls.",
    )


@dataclass
class UpdateConfig:
    """Kernel-start auto-updater (#87): offers to re-run the installer on a newer release."""

    enabled: bool = _h(
        True, "Enable the background, fail-open update check at kernel start."
    )
    repo: str = _h(
        "biopb/biopb",
        "owner/name of the deployment repo whose release-v* line is the product "
        "(overridable for forks/testing).",
    )
    channel: str = _h(
        "stable",
        '"stable" -> newest clean release-vX.Y.Z (prereleases skipped); '
        '"prerelease" -> also consider release candidates.',
    )
    skipped_version: str = _h(
        "",
        'A version the user chose to skip ("Skip vX.Y.Z"); suppresses the prompt for exactly that version.',
    )
    timeout: float = _h(
        5.0,
        "Per-request network timeout (s) for the GitHub API fetch; short so the check never delays viewer start.",
    )


@dataclass
class McpConfig:
    """The whole biopb-mcp config: one field per top-level section."""

    widget: WidgetConfig = field(default_factory=WidgetConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    tensor_browser: TensorBrowserConfig = field(default_factory=TensorBrowserConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    grpc: GrpcConfig = field(default_factory=GrpcConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    dask: DaskConfig = field(default_factory=DaskConfig)
    tensor: TensorRuntimeConfig = field(default_factory=TensorRuntimeConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    observe: ObserveConfig = field(default_factory=ObserveConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)


# Section name -> its dataclass, derived from McpConfig so the two never drift.
# Used by validation and by the schema composer (_config_schema.py).
_SECTION_CLASSES = {
    f.name: type(getattr(McpConfig(), f.name)) for f in dataclasses.fields(McpConfig)
}


# Per-class validation rules (biopb/biopb#182). Keyed by class name like the
# tensor server's table; the shared Range/Enum primitives judge the same knobs
# (notably the pyramid rows) identically in both packages.
_CONSTRAINTS = {
    "PyramidConfig": {**PYRAMID_CONSTRAINTS},
    "TimeoutConfig": {
        "health_check": Range(exclusive_min=0),
        "get_op_names": Range(exclusive_min=0),
        "detection_2d": Range(exclusive_min=0),
        "detection_3d": Range(exclusive_min=0),
        "process_image": Range(exclusive_min=0),
    },
    "GrpcConfig": {
        "max_message_size_mb": Range(exclusive_min=0),
        "max_concurrent_calls": Range(min=1),
    },
    "MemoryConfig": {
        "warn_threshold_mb": Range(exclusive_min=0),
        "error_threshold_mb": Range(exclusive_min=0),
    },
    "TransportConfig": {
        "kind": Enum({"http", "stdio"}),
        "port": Range(min=1, max=65535),
        "display_mode": Enum({"auto", "visible", "headless"}),
        "session_log_keep": Range(min=1),  # keep at least the current
        "server_start_timeout": Range(exclusive_min=0),
    },
    "KernelConfig": {
        "startup_timeout": Range(exclusive_min=0),
        "execute_timeout": Range(exclusive_min=0),
        "busy_lock_timeout": Range(exclusive_min=0),
        "promote_after": Range(exclusive_min=0),
        "watchdog_interval": Range(min=0),  # 0 disables the watchdog
        "watchdog_max_respawns": Range(min=0),
        "watchdog_respawn_window": Range(min=0),
    },
    "DaskConfig": {
        "scheduler": Enum({"distributed", "threads", "synchronous"}),
        "owner": Enum({"daemon", "kernel"}),
        "num_workers": Range(min=0),  # 0 -> dask picks ~n_cores
    },
    "TensorRuntimeConfig": {
        "health_poll_min_interval": Range(min=0),  # 0 disables the watcher
        "health_poll_max_interval": Range(min=0),
    },
    "ServicesConfig": {
        "skills_cache_ttl": Range(min=0),
    },
}


# The default config as a plain nested dict (the runtime form). asdict() is the
# single place defaults come from -- there is no separate default literal.
DEFAULT_CONFIG = dataclasses.asdict(McpConfig())


def get_default_config() -> dict:
    """Return a deep copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


_MISSING = object()


def get_setting(config: dict, path: str, default=_MISSING):
    """Read an absolute dotted *path* from *config*, else ``DEFAULT_CONFIG``.

    ``get_setting(config, "dask.scheduler")`` walks *config* by the dotted path;
    on a miss at any level it falls back to *default* if given, else to the value
    at the same path in ``DEFAULT_CONFIG``. Mutable defaults are deep-copied so
    callers cannot alias the shared ``DEFAULT_CONFIG``. Centralizing the fallback
    keeps each default declared exactly once -- call sites never restate a literal.

    Raises:
        KeyError: the path is absent from both *config* and ``DEFAULT_CONFIG`` and
            no *default* was given (a programmer error, not a config issue).
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
    """Config directory (``~/.config/biopb`` on all platforms).

    Co-located with the tensor server's ``biopb.json`` and the installer's
    client-definition ``mcp.json`` (see :mod:`biopb._config_location`); computed
    from ``Path.home()`` at call time so tests can redirect it.
    """
    config_dir = Path.home() / ".config" / "biopb"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Path to the config file (``~/.config/biopb/mcp-config.json``)."""
    return get_config_dir() / "mcp-config.json"


def get_log_dir() -> Path:
    """Log directory (``~/.local/share/biopb-mcp/log`` on all platforms).

    Logs are persistent runtime state, not user-editable config, so they live
    under the XDG *data* tree rather than ~/.config -- matching the biopb tensor
    server, which writes to ~/.local/share/biopb/log.
    """
    log_dir = Path.home() / ".local" / "share" / "biopb-mcp" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_session_log_dir() -> Path:
    """Directory for per-session stdio-shim logs (``<log dir>/sessions``).

    Each shim-owned session writes its own logfile here rather than the shared
    ``mcp-server.log``, so concurrent sessions never interleave; retention
    (``transport.session_log_keep``) prunes it to the newest N.
    """
    d = get_log_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_daemon_log_file(config: Optional[dict] = None) -> Path:
    """Path of the http daemon's combined stdout/stderr log.

    A *single* canonical location so every launcher and every reader agree on one
    file regardless of who started the daemon. Honors ``transport.kernel_log`` if
    set, else ``<log dir>/mcp-server.log``. ``config`` is loaded (cached
    singleton) when not supplied; the shim passes the dict it already holds.
    """
    if config is None:
        config = load_config()
    override = get_setting(config, "transport.kernel_log")
    if override:
        return Path(override)
    return get_log_dir() / "mcp-server.log"


def get_pid_file() -> Path:
    """Path to the http daemon's PID file (``~/.local/share/biopb-mcp/mcp-server.pid``).

    The running http daemon writes its own PID here once it is up, regardless of
    who launched it, so ``biopb mcp status`` can detect it uniformly. The biopb
    CLI hardcodes the same path (``biopb.cli.MCP_PID_FILE``); keep the two in sync.
    """
    return Path.home() / ".local" / "share" / "biopb-mcp" / "mcp-server.pid"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* in place, returning *base*.

    Nested dicts are merged key-by-key so a partial user section (e.g. only
    ``{"dask": {"num_workers": 4}}``) overrides just that leaf and leaves its
    sibling defaults intact. Non-dict values (and dict-vs-non-dict mismatches)
    replace wholesale.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# --- Declarative config validation (biopb/biopb#182) --------------------------
#
# The merged dict is otherwise trusted: a valid-JSON but out-of-range leaf passes
# straight through and only blows up on a hot path. Validate here, driven by the
# dataclass _CONSTRAINTS + section->class map, so a value is judged by the same
# rule the schema advertises. Severity per #182: the deep-merge already degrades
# gracefully, so a bad leaf is a logged WARNING and is reset to its known-good
# DEFAULT_CONFIG value (which also neutralizes the no-coercion wrinkle -- a string
# where a number is expected fails its Range check and is replaced by the default).


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
    for section, cls in _SECTION_CLASSES.items():
        for field_name, constraint in _CONSTRAINTS.get(cls.__name__, {}).items():
            keys = (section, field_name)
            value = _walk_path(config, keys)
            if value is _MISSING or constraint.ok(value):
                continue
            default = _walk_path(DEFAULT_CONFIG, keys)
            logger.warning(
                "Invalid config value %s=%r (expected %s); using default %r. "
                "See biopb/biopb#182.",
                ".".join(keys),
                value,
                constraint.describe(),
                default,
            )
            if default is _MISSING:
                continue
            config[section][field_name] = copy.deepcopy(default)

    # Cross-field: the health-poll interval must not invert (min > max), which
    # would break the exponential backoff. Reset both to defaults if it does.
    lo_path = ("tensor", "health_poll_min_interval")
    hi_path = ("tensor", "health_poll_max_interval")
    lo, hi = _walk_path(config, lo_path), _walk_path(config, hi_path)
    if lo is not _MISSING and hi is not _MISSING and lo > hi:
        logger.warning(
            "Invalid config: tensor.health_poll_min_interval=%r > "
            "health_poll_max_interval=%r; using defaults. See biopb/biopb#182.",
            lo,
            hi,
        )
        tensor = config["tensor"]
        tensor["health_poll_min_interval"] = copy.deepcopy(
            _walk_path(DEFAULT_CONFIG, lo_path)
        )
        tensor["health_poll_max_interval"] = copy.deepcopy(
            _walk_path(DEFAULT_CONFIG, hi_path)
        )
    return config


def _read_and_merge_from_disk() -> dict:
    """Read the config file and merge onto defaults.

    Returns a fully-merged config dict with every expected key present. A missing,
    malformed, or unreadable file falls back to ``get_default_config()`` (logged).
    This is the single code path from disk into memory; the ``CONFIG`` singleton
    and :func:`load_config` both route through it.
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.debug("Config file not found, using defaults")
        return get_default_config()

    try:
        with config_path.open("r") as f:
            config = json.load(f)

        # Deep-merge with defaults so partial user sections override only their own
        # leaves and every expected key still resolves.
        merged = _deep_merge(get_default_config(), config)
        # Reject out-of-range / bad-enum leaves (warn + reset) before any hot path
        # reads them (biopb/biopb#182).
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
    # the temp file.
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
    it. Every read routes through :meth:`get` and every write through :meth:`set`.

    Thread-safe: the MCP kernel runs agent code in background threads that may
    read config, so lazy init / set / reload take an ``RLock``.
    """

    def __init__(self) -> None:
        self._data: Optional[dict] = None
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._data is None:
                self._data = _read_and_merge_from_disk()

    def get(self, path: str, default=_MISSING):
        """Read a dotted *path*, falling back to ``DEFAULT_CONFIG``.

        Same contract as :func:`get_setting` (the ambient form of it). Holds the
        lock across the dotted-path walk so a read never observes a half-applied
        multi-key write or a momentarily-``None`` cache mid-reload.
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
        """Invalidate the cache; the next read re-reads disk."""
        with self._lock:
            self._data = None

    def as_dict(self) -> dict:
        """Return the *live* cached merged dict (not a copy).

        Escape hatch for code that threads the raw dict (the MCP bootstrap).
        Callers must treat it as read-only; all mutation goes through :meth:`set`.
        Not safe to hold across a concurrent :meth:`set` -- use it for
        startup/single-threaded threading and prefer :meth:`get` otherwise.
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

    Back-compat shim: atomically writes the given dict, then invalidates the cache
    so the next read re-merges it from disk. Prefer ``CONFIG.set(path, value)``
    for targeted writes.
    """
    _atomic_write_json(get_config_path(), config)
    CONFIG.reload()


def get_grid_params(is_3d: bool, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Get grid size and stride from config as (grid_size, stride) int arrays."""
    suffix = "3d" if is_3d else "2d"
    grid_size = np.array(get_setting(config, f"grid.size_{suffix}"), dtype=int)
    stride = np.array(get_setting(config, f"grid.stride_{suffix}"), dtype=int)
    return grid_size, stride
