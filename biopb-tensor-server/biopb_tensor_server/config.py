"""Configuration management for TensorFlight server.

Supports TOML config files with:
- Server settings (host, port)
- Data source definitions (explicit files or directory auto-discovery)
- Credential profiles for remote storage (S3, GCS, etc.)

Example config (explicit):
```toml
[server]
host = "0.0.0.0"
port = 8815

[[sources]]
type = "zarr"
url = "/data/images.zarr"
source_id = "my-image"
dim_labels = ["z", "y", "x"]

[[sources]]
type = "hdf5"
url = "/data/sample.h5"
dataset = "/images/channel0"
```

Example config (relaxed auto-discovery):
```toml
[server]
host = "0.0.0.0"
port = 8815

[[sources]]
url = "/data/"  # No type, no source_id - recursive auto-discovery

# HDF5 requires explicit 'type' and 'dataset' - not auto-detected
[[sources]]
type = "hdf5"
url = "/data/sample.h5"
dataset = "/images"
```

Example config (remote storage):
```toml
[server]
host = "0.0.0.0"
port = 8815

[credentials]
default_profile = "aws-prod"

[[credentials.profiles]]
name = "aws-prod"
storage_type = "s3"
key = "AKIAIOSFODNN7EXAMPLE"
secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
region = "us-east-1"

[[sources]]
type = "ome-zarr"
url = "s3://bucket/experiment.ome.zarr"
credentials_profile = "aws-prod"
```
"""

from __future__ import annotations

import copy
import getpass
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Config file location & format preference live in the core `biopb` package so
# the umbrella CLI and biopb-mcp share one definition (all three depend on
# `biopb`). Re-exported for back-compat (`biopb_tensor_server.config.find_config`
# and the name constants). See biopb._config_location for the JSON-canonical
# rationale (biopb/biopb#34).
from biopb._config_location import (
    CANONICAL_CONFIG_NAME as CANONICAL_CONFIG_NAME,
    DEFAULT_CONFIG_DIR as DEFAULT_CONFIG_DIR,
    LEGACY_CONFIG_NAME as LEGACY_CONFIG_NAME,
    find_config as find_config,
)

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    discover_sources as claim_based_discover,
    generate_source_id,
    get_file_identity,
    is_remote_url,
)
from biopb_tensor_server.remote import (
    CredentialProfile,
    CredentialsConfig,
)

# Alias for backward compatibility with internal usage
_is_remote_url = is_remote_url

logger = logging.getLogger(__name__)

# Python 3.11+ has tomllib in stdlib
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# Default on-disk cache location for the file backend. Uses the system temp dir
# (node-local fast scratch on HPC, honoring $TMPDIR/$TEMP/$TMP) rather than home,
# which is often slow/quota-bound NFS. The directory is scoped per user so that
# multiple users sharing one node (HPC/Singularity) don't collide on a single
# /tmp/biopb-cache and its process lock. Resolves correctly on every platform
# (on Windows this lands under %TEMP%\biopb-cache-<username>).
def _default_file_cache_dir() -> Path:
    # Prefer the POSIX uid (always present, no env dependency); fall back to the
    # username on platforms without getuid (Windows), then to a fixed label.
    try:
        ident = str(os.getuid())
    except AttributeError:
        try:
            ident = getpass.getuser()
        except Exception:
            ident = "default"
    return Path(tempfile.gettempdir()) / f"biopb-cache-{ident}"


DEFAULT_FILE_CACHE_DIR = _default_file_cache_dir()


def _default_cache_backend() -> str:
    """Default cache backend when a config doesn't specify one.

    Always "file". The file backend caches *decoded* chunks as Arrow IPC
    segments, so repeat reads skip re-decoding the raw format -- and decoding
    TIFF/CZI/etc. is typically slower than an Arrow IPC read-back. It also
    persists across restarts and powers the localhost cache-file fast path
    (issue #9). Windows is safe now that the backend copies batches off the
    segment mmap so eviction can unlink (copy-on-read, biopb/biopb#5). The
    localhost cache-file *handoff* stays POSIX-only (a client's cross-process
    mmap still blocks unlink on Windows), but that is a client-side gate,
    independent of the server's choice of backend.
    """
    return "file"


# --- Declarative config validation (biopb/biopb#34) ---------------------------
#
# Out-of-range / bad-enum values used to be accepted silently and blow up later:
# downscale_factor=0 -> ZeroDivisionError in GetFlightInfo; pixel_budget_cubic_root
# <= 0 -> infinite loop in the precache worker; reduction_method="bogus" -> a
# read-time ValueError; downscale_factor=1 -> a silently single-level pyramid.
# Validate at construction instead, in each dataclass's __post_init__, so EVERY
# path that builds a config -- both file formats AND direct dataclass
# construction (resolve_all_sources, the future generator) -- is covered by one
# declarative source of truth.
#
# Severity is WARN during the TOML deprecation window: a config that loaded
# before must not become a hard startup failure on upgrade. Flip
# _STRICT_VALIDATION -> True (warn -> raise) when the legacy read path is removed
# (the "fail fast at startup" end state). The same table feeds the planned JSON
# Schema emitter, so the constraints are declared once.

_STRICT_VALIDATION = False

# Normalized methods + aliases accepted by downsample.normalize_reduction_method
# (matched case-insensitively, mirroring that function).
_REDUCTION_METHODS = {
    "area",
    "linear",
    "nearest",
    "precompute",
    "stride",
    "decimate",
    "mean",
    "precomputed",
}


class _Range:
    """A numeric bound; ``min``/``max`` are inclusive, either may be omitted."""

    def __init__(self, *, min=None, max=None):
        self.min = min
        self.max = max

    def ok(self, value) -> bool:
        # bool is an int subclass; a bool where a number is expected is wrong.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        below = self.min is not None and value < self.min
        above = self.max is not None and value > self.max
        return not (below or above)

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            return f"a number in [{self.min}, {self.max}]"
        if self.min is not None:
            return f"a number >= {self.min}"
        return f"a number <= {self.max}"

    def to_json_schema(self) -> dict:
        """Inclusive bounds as JSON Schema keywords (for the schema emitter)."""
        out: dict = {}
        if self.min is not None:
            out["minimum"] = self.min
        if self.max is not None:
            out["maximum"] = self.max
        return out


class _Enum:
    """Membership in a fixed set, optionally case-insensitive for strings."""

    def __init__(self, allowed, *, case_insensitive=False):
        self.case_insensitive = case_insensitive
        self._display = set(allowed)
        # When case-insensitive, fold the allowed set too so the comparison is
        # symmetric (else a lowercased value never matches an upper-case member).
        self.allowed = {
            a.lower() if (case_insensitive and isinstance(a, str)) else a
            for a in allowed
        }

    def ok(self, value) -> bool:
        if self.case_insensitive and isinstance(value, str):
            value = value.strip().lower()
        return value in self.allowed

    def describe(self) -> str:
        return "one of: " + ", ".join(sorted(map(str, self._display)))

    def to_json_schema(self) -> dict:
        """The allowed set as a JSON Schema ``enum`` -- but only for
        case-sensitive enums. A case-insensitive set (``log_level``,
        ``reduction_method``) accepts any casing the server folds, so a hard
        ``enum`` of the canonical members would reject values the server
        actually honors; there we stay lenient and surface the set via
        :meth:`describe` in the property description instead."""
        if self.case_insensitive:
            return {}
        return {"enum": sorted(self._display, key=str)}


# Per-dataclass field constraints, keyed by class name (so this table can sit
# above the class definitions). full_rescan_interval is intentionally absent:
# a value <= 0 *disables* the periodic full-scan backstop (documented sentinel).
_CONSTRAINTS = {
    "CacheConfig": {
        "backend": _Enum({"memory", "file"}),
        "memory_max_entries": _Range(min=1),
        "memory_max_bytes": _Range(min=1),
        "file_max_segment_bytes": _Range(min=1),
        "file_max_total_bytes": _Range(min=1),
    },
    "PyramidConfig": {
        "reduction_method": _Enum(_REDUCTION_METHODS, case_insensitive=True),
        "threshold": _Range(min=1),
        "downscale_factor": _Range(min=2),
        "pixel_budget_cubic_root": _Range(min=1),
    },
    "PrecacheConfig": {
        "idle_debounce_seconds": _Range(min=0),
        "backlog_high_water": _Range(min=0.0, max=1.0),
        "backlog_idle_recheck_seconds": _Range(min=0),
    },
    "MetadataDbConfig": {
        "max_query_results": _Range(min=1),
        "max_list_flights_results": _Range(min=1),
        "query_timeout_ms": _Range(min=1),
    },
    "ServerConfig": {
        "port": _Range(min=1, max=65535),
        "log_level": _Enum(
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, case_insensitive=True
        ),
        # compute_backend is matched case-sensitively, as configure_compute_backend
        # does (it raises on anything outside this set).
        "compute_backend": _Enum({"auto", "cpu", "gpu"}),
        "gpu_min_input_mb": _Range(min=0),
        "gpu_min_linear_input_mb": _Range(min=0),
        "gpu_memory_safety_factor": _Range(min=1),
        "gpu_min_merged_chunks": _Range(min=1),
        "rescan_interval": _Range(min=0),
        "stability_window": _Range(min=0),
        "stable_rescans_required": _Range(min=0),
    },
}

# Class name -> the config section it maps to, for friendlier messages.
_SECTION_FOR = {
    "CacheConfig": "cache",
    "PyramidConfig": "pyramid",
    "PrecacheConfig": "precache",
    "MetadataDbConfig": "metadata_db",
    "ServerConfig": "server",
}


def _validate_config(instance) -> None:
    """Check *instance*'s fields against :data:`_CONSTRAINTS` (warn, or raise
    when ``_STRICT_VALIDATION``). Called from each config dataclass's
    ``__post_init__`` so every construction path is covered."""
    name = type(instance).__name__
    constraints = _CONSTRAINTS.get(name)
    if not constraints:
        return
    problems = [
        f"{key}={getattr(instance, key)!r} (expected {c.describe()})"
        for key, c in constraints.items()
        if not c.ok(getattr(instance, key))
    ]
    if not problems:
        return
    msg = f"Invalid config value(s) in [{_SECTION_FOR.get(name, name)}]: " + "; ".join(
        problems
    )
    if _STRICT_VALIDATION:
        raise ValueError(msg)
    logger.warning(
        "%s. Used as-is for now; this will become an error in a future release. "
        "See biopb/biopb#34.",
        msg,
    )


@dataclass
class SourceConfig:
    """Configuration for a single data source.

    A source may contain multiple tensors (multifield support) - the adapter
    handles tensor enumeration via list_tensor_descriptors() at runtime.

    Attributes:
        url: URL or path to the data source (supports local paths and remote URLs)
        type: Storage type - "zarr", "hdf5", "ome-tiff", "ome-tiff-multifile", "ome-zarr", "ome-zarr-hcs",
              "aics", or "tensor-server" (an upstream biopb tensor server, fronted as a caching proxy).
              Optional - auto-detected if None (local files, or a grpc:// endpoint -> "tensor-server").
        source_id: Unique identifier for the data source (auto-generated from URL if None)
        dim_labels: Dimension labels (optional, applies to all tensors in source)
        dataset: HDF5 dataset path (required for HDF5 type)
        monitor: Enable live filesystem monitoring for this source (local directories only)
                 When True, the server will watch for file add/delete events and update
                 the catalog automatically.
        cloud: Treat this root as cloud/synced-folder storage (OneDrive, Dropbox, ...).
               When True, dehydrated (offline-placeholder) entries under this root are
               admitted as *unresolved* sources instead of being skipped, and their
               shape/dtype is resolved lazily on first access (cloud-storage phase 2).
               Opt-in only -- it does not weaken the global placeholder guard elsewhere.
        credentials_profile: Name of credential profile for remote URLs (overrides global default)
        alias: Namespace prefix for an upstream "tensor-server" source. The proxy
               mirrors the upstream's sources under "<alias>__<upstream_source_id>"
               so multiple upstreams (and local sources) coexist in one flat,
               source_id-keyed catalog without id collisions. Must be slash-free
               (it becomes part of a slash-free source_id). Ignored for non-proxy
               source types.
        is_remote: Flag indicating if this is a remote source (set during discovery)
    """

    url: str
    type: Optional[
        Literal[
            "zarr",
            "hdf5",
            "ome-tiff",
            "ome-tiff-multifile",
            "ome-zarr",
            "ome-zarr-hcs",
            "aics",
            "tensor-server",
        ]
    ] = None
    source_id: Optional[str] = None
    dim_labels: Optional[List[str]] = None
    dataset: Optional[str] = None  # For HDF5
    monitor: bool = False  # Enable live filesystem monitoring
    cloud: bool = False  # Treat as cloud/synced-folder root (admit unresolved sources)
    credentials_profile: Optional[str] = None  # Override global credential profile
    alias: Optional[str] = None  # Namespace prefix for a "tensor-server" upstream
    _is_remote: Optional[bool] = field(
        default=None, init=False
    )  # Internal field, computed from URL

    @property
    def is_remote(self) -> bool:
        """Check if this source is a remote URL."""
        if self._is_remote is None:
            # Compute lazily
            object.__setattr__(self, "_is_remote", _is_remote_url(self.url))
        return self._is_remote

    def __post_init__(self):
        if self.url is None or self.url == "":
            raise ValueError("SourceConfig requires a valid 'url'")

        # An alias becomes part of a slash-free source_id (<alias>__<upstream_id>),
        # so it must not contain '/' (the array_id source boundary).
        if self.alias is not None and "/" in self.alias:
            raise ValueError(
                f"SourceConfig 'alias' must be slash-free, got: {self.alias!r}"
            )

        # Compute is_remote from URL
        object.__setattr__(self, "_is_remote", _is_remote_url(self.url))

        # Generate source_id from URL hash if not provided
        if self.source_id is None:
            detected_type = self.type or detect_source_type(self.url) or "data"
            object.__setattr__(
                self, "source_id", generate_source_id(self.url, detected_type)
            )

    @property
    def local_path(self) -> Optional[Path]:
        """Return Path if url is a local file path, else None.

        For remote URLs (s3://, http://, etc.), returns None.
        For local paths, returns the resolved absolute Path.
        """
        if _is_remote_url(self.url):
            return None
        return Path(self.url).resolve()


@dataclass
class CacheConfig:
    """Cache configuration for computed virtual chunks.

    Attributes:
        backend: Cache backend type - "memory" or "file"
        memory_max_entries: Maximum number of cached entries (memory backend)
        memory_max_bytes: Maximum total bytes to cache (memory backend, default 512 MB)
        file_cache_dir: Directory for cache files (file backend, default: system temp dir/biopb-cache-<uid>, scoped per user)
        file_max_segment_bytes: Maximum bytes per segment file (file backend, default 64 MB)
        file_max_total_bytes: Maximum total bytes across all segments (file backend, default 4 GB)
    """

    backend: str = field(default_factory=_default_cache_backend)
    memory_max_entries: int = 1024
    memory_max_bytes: int = 512 * 1024 * 1024  # 512 MB
    file_cache_dir: Path = DEFAULT_FILE_CACHE_DIR
    file_max_segment_bytes: int = 64 * 1024 * 1024  # 64 MB per segment
    file_max_total_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB total

    def __post_init__(self):
        if isinstance(self.file_cache_dir, str):
            self.file_cache_dir = Path(self.file_cache_dir)
        _validate_config(self)


@dataclass
class PyramidConfig:
    """Resolution-pyramid level definition -- the server's single source of truth.

    These knobs decide the levels the server *advertises* on a tensor descriptor
    (``TensorDescriptor.pyramid``, filled by GetFlightInfo) and that the precache
    worker warms. Owning them server-side is the point: the client reads the
    advertised levels and the precache worker warms those same scales, so the two
    can no longer drift (previously these lived in biopb-mcp's [pyramid] config
    and were mirrored here by hand). They still default to biopb-mcp's historical
    values so an un-upgraded client computing the pyramid itself stays aligned.

    Attributes:
        reduction_method: On-the-fly downsampling reduction for computed levels
            ("area" = proper averaging). Native (on-disk) levels are served via
            "precompute" regardless of this.
        threshold: Max X/Y extent of the coarsest level.
        downscale_factor: Per-level linear step (per spatial axis).
        pixel_budget_cubic_root: Cube root of the coarsest level's voxel budget
            (Lx*Ly*Lz <= this**3); bounds napari's 3-D whole-volume read.
    """

    reduction_method: str = "area"
    threshold: int = 4096
    downscale_factor: int = 4
    pixel_budget_cubic_root: int = 512

    def __post_init__(self):
        _validate_config(self)


@dataclass
class PrecacheConfig:
    """Background pre-cache worker configuration.

    The worker warms the file cache for newly-added sources at the *coarsest*
    pyramid level a client requests on open, so the first view is already warm.
    It is inert unless the file cache backend is in use, and stays off the wire
    while live reads are in flight. The level *definition* (which scale, which
    reduction) lives in :class:`PyramidConfig`; this holds only the worker's
    operational knobs.

    Attributes:
        enabled: Run the background precache worker (no-op on a memory backend).
        idle_debounce_seconds: Quiet period required before the worker resumes
            after live Flight traffic.
        backlog_enabled: Also warm sources already present at startup (the
            "backlog"), as a secondary pass behind live additions.
        backlog_high_water: Stop backlog warming once the file cache fills past
            this fraction of its max_bytes, so precache never evicts live data.
        backlog_idle_recheck_seconds: When the cache is over the high-water mark,
            how long the backlog naps before re-checking for freed room.
    """

    enabled: bool = True
    idle_debounce_seconds: float = 2.0
    # Startup-backlog (existing sources) knobs.
    backlog_enabled: bool = True
    backlog_high_water: float = 0.8
    backlog_idle_recheck_seconds: float = 5.0

    def __post_init__(self):
        _validate_config(self)


@dataclass
class MetadataDbConfig:
    """Configuration for DuckDB metadata database and source catalog safety limits.

    Enables efficient SQL filtering for large source catalogs (>100k sources).
    Replaces O(n) in-memory scans with indexed DuckDB queries.

    Attributes:
        enabled: DEPRECATED (biopb/biopb#225) -- the metadata database is
            becoming mandatory and this flag will be removed. Setting it (and
            especially ``enabled = false``) emits a deprecation warning at config
            load; disabling breaks the server-side ``query_sources`` SQL catalog
            the biopb-mcp guide relies on. Still honored during the migration
            window. Enable metadata database for source filtering queries.
        max_query_results: Safety cap on SQL query returned rows (truncation signaled via schema metadata)
        max_list_flights_results: Safety cap on list_flights() returned sources (truncation signaled via schema metadata)
        query_timeout_ms: Query execution timeout in milliseconds
    """

    enabled: bool = True
    max_query_results: int = 100000
    max_list_flights_results: int = 100000
    query_timeout_ms: int = 30000

    def __post_init__(self):
        _validate_config(self)


@dataclass
class ServerConfig:
    """Server configuration.

    Attributes:
        host: Server host
        port: Server port
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_scope_to_biopb: If True, logging level changes only affect biopb_tensor_server
                            packages, not external packages like grpc, numpy, etc.
        monitor_mode: Directory monitoring mode.
            "periodic": Run periodic rescans for monitored local directories.
            "off": Disable background rescans after initial discovery.
        rescan_interval: Seconds between background rescan attempts.
        full_rescan_interval: Seconds between forced full rescans that bypass
            subtree pruning. Values <= 0 disable the periodic full-scan backstop.
        stability_window: Minimum quiet period before a path/subtree is eligible
            for discovery or removal checks.
        stable_rescans_required: Additional unchanged rescans required before a
            path is considered stable. The default 0 preserves current behavior.
        probe_open_files: When True, perform a best-effort append-open probe
            before considering candidate files stable. This is advisory only.
        aggressive_dir_pruning: When True, allow unchanged monitored roots to be
            pruned in addition to descendant subdirectories. This reduces scan
            cost further but may defer root-level file updates until a later scan.
        claim_generic_images: When True, recursive directory discovery also
            claims generic raster/video files (.png/.jpg/.jpeg/.gif/.bmp and
            .avi/.mov/.mp4/.mpeg/.mpg) as aics sources. Off by default: these
            are almost never microscopy tensors and flood the catalog with
            screenshots/icons/thumbnails (biopb/biopb#40). Explicitly configured
            sources (type set, or a single-file url) are unaffected.
        writable: Enable write mode for source creation and data upload
        write_dir: Directory for zarr-backed uploaded sources (None = no zarr uploads)
        cache: Cache configuration
        credentials: Credentials configuration for remote storage
        metadata_db: DuckDB metadata database configuration for source filtering
        sources: List of data sources (each may contain multiple tensors)
    """

    host: str = "0.0.0.0"
    port: int = 8815
    log_level: str = "INFO"
    log_scope_to_biopb: bool = True
    compute_backend: str = "auto"
    gpu_min_input_mb: float = 4.0
    gpu_min_linear_input_mb: float = 2.0
    gpu_memory_safety_factor: int = 4
    gpu_min_merged_chunks: int = 4
    monitor_mode: str = "periodic"
    rescan_interval: float = 30.0
    full_rescan_interval: float = 3600.0
    stability_window: float = 30.0
    stable_rescans_required: int = 0
    probe_open_files: bool = True
    aggressive_dir_pruning: bool = False
    claim_generic_images: bool = False  # Claim generic raster/video in discovery (#40)
    writable: bool = False  # Enable write mode
    write_dir: Optional[Path] = None  # Directory for zarr-backed sources
    cache: CacheConfig = field(default_factory=CacheConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    precache: PrecacheConfig = field(default_factory=PrecacheConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    metadata_db: MetadataDbConfig = field(default_factory=MetadataDbConfig)
    sources: List[SourceConfig] = field(default_factory=list)

    def __post_init__(self):
        _validate_config(self)


def load_config(path: Path) -> ServerConfig:
    """Load configuration from a JSON or TOML file.

    JSON is the canonical format; TOML is still read for back-compat during the
    migration window (biopb/biopb#34) and emits a deprecation warning. Format is
    chosen by file extension, with a content sniff for unknown / extension-less
    paths. Either way the file is parsed to a plain dict and handed to the
    format-agnostic :func:`parse_config`.

    Args:
        path: Path to JSON (``.json``) or TOML (``.toml``) config file

    Returns:
        ServerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If the file cannot be parsed as its declared/sniffed format
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = _read_config_file(path)
    return parse_config(data)


# Name of the sibling JSON Schema file save_config drops next to the config so
# editors validate it offline (relative $schema), independent of any hosted URL.
SCHEMA_SIDECAR_NAME = "biopb.schema.json"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write *data* as pretty JSON to *path* atomically (temp file + os.replace).

    Writing to a sibling temp file and renaming over the target means a reader
    never sees a half-written file, and a failed write leaves the original
    untouched. Unlike biopb-mcp's same-named helper this *raises* on failure --
    the admin endpoint surfaces a write/permission error to the user rather than
    silently swallowing it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def save_config(data: Dict[str, Any], path: Path) -> Path:
    """Write *data* to disk as canonical JSON, atomically, and return the path.

    The admin endpoint's config writer (biopb/biopb#237). The inverse of
    :func:`load_config`, but it round-trips on the **raw dict** the caller
    supplies -- not a dataclass -- so advanced or future keys the form never
    surfaced survive the write. Routing through ``parse_config`` -> dataclass ->
    ``asdict`` would clobber them (and there is no dataclass->dict projection).

    Behavior:
    - JSON is canonical. If *path* points at a legacy ``biopb.toml`` the write
      targets the sibling ``biopb.json`` and the old TOML is renamed to
      ``biopb.toml.bak``, so :func:`find_config`'s both-files shadow warning
      never fires (biopb/biopb#34).
    - A sibling ``biopb.schema.json`` (the output of ``build_config_schema``) is
      written next to the config and a *relative* ``"$schema":
      "./biopb.schema.json"`` pointer is embedded, so editors validate the
      config offline with no hosted schema URL.
    - The write is atomic; on failure the file on disk is untouched and the error
      propagates so the caller can surface it.
    """
    if isinstance(path, str):
        path = Path(path)

    # JSON is canonical: redirect a .toml target to its sibling biopb.json and
    # back the legacy file up so find_config's both-files warning never fires.
    legacy_toml: Optional[Path] = None
    if path.suffix.lower() == ".toml":
        legacy_toml = path
        path = path.with_name(CANONICAL_CONFIG_NAME)

    schema_path = path.with_name(SCHEMA_SIDECAR_NAME)

    # build_config_schema lives in config_schema, which imports the dataclasses
    # here -- import lazily to avoid the cycle (see _known_config_keys).
    from biopb_tensor_server.config_schema import build_config_schema

    # Embed a relative $schema pointer (offline editor validation, no hosted URL).
    payload = dict(data)
    payload["$schema"] = f"./{SCHEMA_SIDECAR_NAME}"

    _atomic_write_json(schema_path, build_config_schema())
    _atomic_write_json(path, payload)

    if legacy_toml is not None and legacy_toml.exists():
        backup = legacy_toml.with_name(legacy_toml.name + ".bak")
        legacy_toml.replace(backup)
        logger.info(
            "Migrated legacy %s to %s; backed up the old file to %s (biopb/biopb#34).",
            legacy_toml.name,
            path.name,
            backup.name,
        )

    return path


# Secret-bearing keys on a [[credentials.profiles]] entry. These are at-rest
# secrets (S3/GCS/Azure credentials, per remote.CredentialProfile); the admin
# endpoint redacts them out of GET /api/config so they never reach the browser,
# and restores them from disk on PUT so saving the redacted form does not
# clobber them (biopb/biopb#237).
REDACTED_SENTINEL = "***REDACTED***"
_SECRET_PROFILE_KEYS = ("key", "secret", "token")


def _iter_profile_dicts(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The `[[credentials.profiles]]` dicts in a raw config, or [] if absent."""
    if not isinstance(config, dict):
        return []
    creds = config.get("credentials")
    profiles = creds.get("profiles") if isinstance(creds, dict) else None
    if not isinstance(profiles, list):
        return []
    return [p for p in profiles if isinstance(p, dict)]


def redact_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *config* with credential-profile secrets masked.

    Each present, non-empty ``key``/``secret``/``token`` on every credential
    profile is replaced with :data:`REDACTED_SENTINEL` so it is never sent to
    the browser by ``GET /api/config``. The on-disk file is untouched;
    :func:`restore_redacted_secrets` puts the real values back on a later PUT.
    """
    redacted = copy.deepcopy(config)
    for profile in _iter_profile_dicts(redacted):
        for key in _SECRET_PROFILE_KEYS:
            if profile.get(key):
                profile[key] = REDACTED_SENTINEL
    return redacted


def restore_redacted_secrets(
    incoming: Dict[str, Any], existing: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a copy of *incoming* with redaction sentinels resolved from disk.

    For every credential-profile secret whose incoming value is the redaction
    sentinel (the UI round-tripped the masked GET response unchanged), substitute
    the real value from the matching profile in *existing* (matched by ``name``).
    If no prior value exists the sentinel key is dropped rather than persisted, so
    the literal sentinel is never written to disk. A profile that supplies a real
    new value overwrites as usual.
    """
    merged = copy.deepcopy(incoming)
    existing_by_name = {
        p.get("name"): p for p in _iter_profile_dicts(existing) if p.get("name")
    }
    for profile in _iter_profile_dicts(merged):
        prior = existing_by_name.get(profile.get("name"), {})
        for key in _SECRET_PROFILE_KEYS:
            if profile.get(key) == REDACTED_SENTINEL:
                if prior.get(key):
                    profile[key] = prior[key]
                else:
                    profile.pop(key, None)
    return merged


def _read_config_file(path: Path) -> Dict[str, Any]:
    """Read a config file into a plain dict, dispatching on format.

    Dispatch is by extension (``.json`` / ``.toml``); an unknown or
    extension-less path is sniffed (JSON first, since it is canonical, then
    TOML). Reading TOML logs a one-line deprecation warning naming the canonical
    alternative.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _load_json(path)
    if suffix == ".toml":
        _warn_toml_deprecated(path)
        return _load_toml(path)

    # Unknown / extension-less: sniff. Prefer JSON (canonical); on a JSON parse
    # failure fall back to TOML rather than guessing from bytes.
    raw = path.read_bytes()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    _warn_toml_deprecated(path)
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"Config file {path} is neither valid JSON nor valid TOML: {e}"
        ) from e


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except ValueError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}") from e


def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in config file {path}: {e}") from e


def _warn_toml_deprecated(path: Path) -> None:
    logger.warning(
        "Config file %s uses the legacy TOML format, which is deprecated; "
        "JSON (%s) is now canonical. TOML will keep working during the "
        "migration window. See biopb/biopb#34.",
        path,
        CANONICAL_CONFIG_NAME,
    )


# The known-key set for the unknown-key warning is the config JSON Schema's
# property lists, derived once and cached. The schema is generated from these
# same dataclasses + _CONSTRAINTS (config_schema.build_config_schema), so the
# warning, the published schema, and the value validation can no longer drift
# from one hand-maintained key table -- they share one source (biopb/biopb#34,
# superseding #234's hardcoded _KNOWN_* sets). Imported lazily to avoid a config
# <-> config_schema import cycle (config_schema imports the dataclasses here).
_KNOWN_KEYS_CACHE: Optional[tuple] = None


def _known_config_keys() -> tuple:
    global _KNOWN_KEYS_CACHE
    if _KNOWN_KEYS_CACHE is None:
        from biopb_tensor_server.config_schema import known_config_keys

        _KNOWN_KEYS_CACHE = known_config_keys()
    return _KNOWN_KEYS_CACHE


def _warn_extra_keys(table: Dict[str, Any], known: set, label: str) -> None:
    """Warn for each key in *table* not in *known* (warn-only; never raises)."""
    for key in sorted(k for k in table if k not in known):
        logger.warning(
            "Unknown config key `%s` in [%s]; it is ignored and the default is "
            "used. Check for a typo or a renamed option. Known keys: %s.",
            key,
            label,
            ", ".join(sorted(known)),
        )


def _warn_unknown_config_keys(data: Dict[str, Any]) -> None:
    """Warn for unrecognized config sections / keys before they silently drop.

    An unknown key is otherwise dropped and the default used with no signal --
    the classic trap is ``[cache] memory_max_entries`` (the dataclass field
    name) where the parser reads ``max_entries``, so a 1-entry cache silently
    stays at the 1024/512MB default. Value validation (range/enum) lives in the
    dataclasses; this only flags *keys the parser never reads*. The known-key
    set is the config JSON Schema's property lists (see :func:`_known_config_keys`).
    Warn-only.
    """
    if not isinstance(data, dict):
        return
    sections, section_keys, source_keys, profile_keys = _known_config_keys()
    for section in sorted(data):
        if section.startswith("$"):
            # `$schema` / `$id` are tolerated meta keys (save_config embeds a
            # relative `$schema`); they are neither config sections nor typos.
            continue
        if section not in sections:
            logger.warning(
                "Unknown config section [%s]; it is ignored. Known sections: %s.",
                section,
                ", ".join(sorted(sections)),
            )
            continue
        value = data[section]
        if section == "sources":
            # [[sources]] is a list of tables; validate each item's keys.
            for src in value if isinstance(value, list) else []:
                if isinstance(src, dict):
                    _warn_extra_keys(src, source_keys, "sources")
            continue
        if not isinstance(value, dict):
            continue
        _warn_extra_keys(value, section_keys.get(section, set()), section)
        if section == "credentials":
            for prof in value.get("profiles", []) or []:
                if isinstance(prof, dict):
                    _warn_extra_keys(prof, profile_keys, "credentials.profiles")


def parse_config(data: Dict[str, Any]) -> ServerConfig:
    """Parse configuration from a dictionary.

    Format-agnostic: ``data`` is a plain dict already read from JSON or TOML by
    :func:`load_config`.

    Args:
        data: Config dictionary (from JSON or TOML)

    Returns:
        ServerConfig object
    """
    _warn_unknown_config_keys(data)

    # Parse server settings
    server_data = data.get("server", {})
    host = server_data.get("host", "0.0.0.0")
    port = server_data.get("port", 8815)
    log_level = server_data.get("log_level", "INFO")
    log_scope_to_biopb = server_data.get("log_scope_to_biopb", True)
    monitor_mode = server_data.get("monitor_mode")
    if monitor_mode is None:
        legacy_watcher_type = server_data.get("watcher_type", "auto")
        monitor_mode = "off" if legacy_watcher_type == "off" else "periodic"

    rescan_interval = server_data.get("rescan_interval")
    if rescan_interval is None:
        rescan_interval = server_data.get("poll_interval", 30.0)

    full_rescan_interval = server_data.get("full_rescan_interval", 3600.0)
    stability_window = server_data.get("stability_window", 30.0)
    stable_rescans_required = server_data.get("stable_rescans_required", 0)
    probe_open_files = server_data.get("probe_open_files", True)
    aggressive_dir_pruning = server_data.get("aggressive_dir_pruning", False)
    claim_generic_images = server_data.get("claim_generic_images", False)
    writable = server_data.get("writable", False)
    write_dir_str = server_data.get("write_dir", None)
    write_dir = Path(write_dir_str) if write_dir_str else None

    # Parse compute settings
    compute_data = data.get("compute", {})
    compute_backend = compute_data.get("backend", "auto")
    gpu_min_input_mb = compute_data.get("gpu_min_input_mb", 4.0)
    gpu_min_linear_input_mb = compute_data.get("gpu_min_linear_input_mb", 2.0)
    gpu_memory_safety_factor = compute_data.get("gpu_memory_safety_factor", 4)
    gpu_min_merged_chunks = compute_data.get("gpu_min_merged_chunks", 4)

    # Parse cache settings
    cache_data = data.get("cache", {})
    cache_backend = cache_data.get("backend", _default_cache_backend())

    # Parse memory backend settings
    memory_max_entries = cache_data.get("max_entries", 1024)
    memory_max_bytes = cache_data.get("max_bytes", 512 * 1024 * 1024)

    # Parse file backend settings (convert MB/GB to bytes if specified)
    file_cache_dir_raw = cache_data.get("file_cache_dir")
    file_cache_dir = (
        Path(file_cache_dir_raw) if file_cache_dir_raw else DEFAULT_FILE_CACHE_DIR
    )
    file_max_segment_mb = cache_data.get("file_max_segment_mb", 64)
    file_max_segment_bytes = int(file_max_segment_mb) * 1024 * 1024
    file_max_total_gb = cache_data.get("file_max_total_gb", 4)
    file_max_total_bytes = int(file_max_total_gb) * 1024 * 1024 * 1024

    cache_config = CacheConfig(
        backend=cache_backend,
        memory_max_entries=memory_max_entries,
        memory_max_bytes=memory_max_bytes,
        file_cache_dir=file_cache_dir,
        file_max_segment_bytes=file_max_segment_bytes,
        file_max_total_bytes=file_max_total_bytes,
    )

    # Parse pyramid settings -- the authority for level definition, shared by the
    # advertised TensorDescriptor.pyramid and the precache worker.
    pyramid_data = data.get("pyramid", {})
    precache_data = data.get("precache", {})

    # Back-compat: these knobs used to live under [precache]; honor an old
    # [precache] value when [pyramid] omits the key, so existing configs don't
    # silently revert to defaults.
    def _pyramid_knob(key, default, cast):
        if key in pyramid_data:
            return cast(pyramid_data[key])
        if key in precache_data:
            return cast(precache_data[key])
        return default

    pyramid_config = PyramidConfig(
        reduction_method=_pyramid_knob("reduction_method", "area", str),
        threshold=_pyramid_knob("threshold", 4096, int),
        downscale_factor=_pyramid_knob("downscale_factor", 4, int),
        pixel_budget_cubic_root=_pyramid_knob("pixel_budget_cubic_root", 512, int),
    )

    # Parse precache settings (operational knobs only).
    precache_config = PrecacheConfig(
        enabled=bool(precache_data.get("enabled", True)),
        idle_debounce_seconds=float(precache_data.get("idle_debounce_seconds", 2.0)),
        backlog_enabled=bool(precache_data.get("backlog_enabled", True)),
        backlog_high_water=float(precache_data.get("backlog_high_water", 0.8)),
        backlog_idle_recheck_seconds=float(
            precache_data.get("backlog_idle_recheck_seconds", 5.0)
        ),
    )

    # Parse credentials settings (NEW)
    credentials_data = data.get("credentials", {})
    credentials_default_profile = credentials_data.get("default_profile", None)
    credentials_profiles_data = credentials_data.get("profiles", [])

    credentials_profiles = []
    for profile_data in credentials_profiles_data:
        profile = CredentialProfile(
            name=profile_data.get("name", ""),
            storage_type=profile_data.get("storage_type", "s3"),
            key=profile_data.get("key", None),
            secret=profile_data.get("secret", None),
            region=profile_data.get("region", None),
            token=profile_data.get("token", None),
            endpoint_url=profile_data.get("endpoint_url", None),
        )
        if profile.name:
            credentials_profiles.append(profile)

    credentials_config = CredentialsConfig(
        default_profile=credentials_default_profile,
        profiles=credentials_profiles,
    )

    # Parse metadata_db settings
    metadata_db_data = data.get("metadata_db", {})
    # `metadata_db.enabled` is deprecated: the metadata DB (server-side DuckDB
    # catalog) is becoming mandatory because it is the canonical source-browsing
    # surface the rest of the system assumes -- the biopb-mcp guide steers agents
    # to `client.query_sources(sql, ...)` (complete, server-side) over the capped
    # `list_sources()`, and that SQL path only exists when the DB is enabled.
    # Warn during the migration window; removal (always-on) is biopb/biopb#225.
    if "enabled" in metadata_db_data:
        if metadata_db_data.get("enabled"):
            logger.warning(
                "Config option `metadata_db.enabled` is deprecated and will be "
                "removed; the metadata database is becoming mandatory (always on). "
                "Drop the flag from your config. See biopb/biopb#225."
            )
        else:
            logger.warning(
                "Config option `metadata_db.enabled = false` is deprecated and "
                "will stop being honored; the metadata database is becoming "
                "mandatory. Disabling it breaks the server-side SQL catalog "
                "(`client.query_sources(...)`) the biopb-mcp guide relies on, "
                "forcing the capped `list_sources()` fallback. See biopb/biopb#225."
            )
    metadata_db_enabled = metadata_db_data.get("enabled", True)
    metadata_db_max_query_results = metadata_db_data.get("max_query_results", 100000)
    metadata_db_max_list_flights_results = metadata_db_data.get(
        "max_list_flights_results", 100000
    )
    metadata_db_query_timeout_ms = metadata_db_data.get("query_timeout_ms", 30000)

    metadata_db_config = MetadataDbConfig(
        enabled=metadata_db_enabled,
        max_query_results=metadata_db_max_query_results,
        max_list_flights_results=metadata_db_max_list_flights_results,
        query_timeout_ms=metadata_db_query_timeout_ms,
    )

    # Parse sources
    sources_data = data.get("sources", [])
    sources: List[SourceConfig] = []

    for src_data in sources_data:
        # Support both 'url' (new) and 'path' (legacy) for backward compatibility
        url = src_data.get("url") or src_data.get("path")
        if url is None:
            raise ValueError("Source config requires 'url' field")

        source = SourceConfig(
            type=src_data.get("type"),  # Optional - auto-detected if None
            url=url,
            source_id=src_data.get("source_id"),  # Optional - auto-generated if None
            dim_labels=src_data.get("dim_labels"),
            dataset=src_data.get("dataset"),
            monitor=src_data.get("monitor", False),  # Optional - default False
            cloud=src_data.get("cloud", False),  # Optional - default False
            credentials_profile=src_data.get("credentials_profile", None),  # NEW
            alias=src_data.get(
                "alias", None
            ),  # Optional - tensor-server proxy namespace
        )
        sources.append(source)

    return ServerConfig(
        host=host,
        port=port,
        log_level=log_level,
        log_scope_to_biopb=log_scope_to_biopb,
        monitor_mode=monitor_mode,
        rescan_interval=float(rescan_interval),
        full_rescan_interval=float(full_rescan_interval),
        stability_window=float(stability_window),
        stable_rescans_required=max(0, int(stable_rescans_required)),
        probe_open_files=bool(probe_open_files),
        aggressive_dir_pruning=bool(aggressive_dir_pruning),
        claim_generic_images=bool(claim_generic_images),
        compute_backend=compute_backend,
        gpu_min_input_mb=float(gpu_min_input_mb),
        gpu_min_linear_input_mb=float(gpu_min_linear_input_mb),
        gpu_memory_safety_factor=int(gpu_memory_safety_factor),
        gpu_min_merged_chunks=int(gpu_min_merged_chunks),
        writable=writable,
        write_dir=write_dir,
        cache=cache_config,
        pyramid=pyramid_config,
        precache=precache_config,
        credentials=credentials_config,
        metadata_db=metadata_db_config,
        sources=sources,
    )


def detect_source_type(url: str) -> Optional[str]:
    """Detect source type from URL characteristics.

    Returns one of: "zarr", "ome-zarr", "ome-zarr-hcs", "ome-tiff", "ome-tiff-multifile", "aics",
    "tensor-server", or None if type cannot be determined.

    Note: HDF5 is NOT auto-detected because it requires explicit dataset path.
    Note: Remote URLs cannot be auto-detected (require explicit 'type' in config),
          except a grpc(+tls) endpoint, which is unambiguously an upstream biopb
          tensor server (the "tensor-server" caching-proxy source type).
    """
    import json

    # A grpc(+tls) endpoint is always an upstream biopb tensor server.
    if url.lower().startswith(("grpc://", "grpc+tls://", "grpcs://")):
        return "tensor-server"

    # Remote URLs cannot be auto-detected
    if _is_remote_url(url):
        return None

    path = Path(url).resolve()

    if path.is_file():
        name = path.name.lower()

        # aicsimageio-supported vendor formats
        aics_extensions = [
            ".czi",
            ".lif",
            ".nd2",
            ".dv",
            ".lsm",
            ".oif",
            ".oib",
            ".xml",
        ]
        for ext in aics_extensions:
            if name.endswith(ext):
                return "aics"

        # OME-TIFF files
        if name.endswith(".ome.tiff") or name.endswith(".ome.tif"):
            return "ome-tiff"

        # Plain TIFF files - skip for now (could check for OME-XML in header)
        if name.endswith(".tiff") or name.endswith(".tif"):
            # Could implement OME-XML header detection here
            # For now, treat as ome-tiff (tifffile can handle plain TIFF)
            return "ome-tiff"

        # HDF5 files - return None (requires explicit config with dataset path)
        if name.endswith(".h5") or name.endswith(".hdf5"):
            return None

        return None

    elif path.is_dir():
        name = path.name.lower()

        # Zarr directories (must have .zarray or .zattrs)
        if name.endswith(".zarr"):
            zattrs_path = path / ".zattrs"
            zarray_path = path / ".zarray"

            # Check for OME-Zarr first (more specific)
            if zattrs_path.exists():
                try:
                    with open(zattrs_path) as f:
                        zattrs = json.load(f)

                    # Check for HCS plate metadata first (highest priority)
                    if "plate" in zattrs:
                        return "ome-zarr-hcs"

                    if "multiscales" in zattrs:
                        return "ome-zarr"
                except (OSError, json.JSONDecodeError, KeyError):
                    pass

            # Plain Zarr (has .zarray or .zattrs without multiscales)
            if zarray_path.exists() or zattrs_path.exists():
                return "zarr"

            return None

        # Check for multi-file OME-TIFF dataset
        metadata_file = path / "_metadata.txt"
        img_files = list(path.glob("img_*.ome.tiff")) + list(path.glob("img_*.ome.tif"))

        if metadata_file.exists() or len(img_files) > 1:
            return "ome-tiff-multifile"

        return None

    return None


def scan_directory_for_sources(
    directory: Path,
    dim_labels: Optional[List[str]] = None,
    recursive: bool = True,
    _skipped_extensions: Optional[set] = None,
) -> List[SourceConfig]:
    """Recursively scan directory for data sources.

    Returns one SourceConfig per data source (file or directory).
    Multi-tensor sources (like aics multi-scene files) are NOT expanded -
    the adapter handles tensor enumeration via list_tensor_descriptors() at runtime.

    Args:
        directory: Root directory to scan (local filesystem only)
        dim_labels: Optional dimension labels to apply to all sources
        recursive: If True, scan subdirectories recursively

    Returns:
        List of discovered SourceConfig objects with auto-detected types and source_ids
    """
    discovered = []
    skipped_hdf5 = []
    skipped_unknown = []

    for item in directory.iterdir():
        # Skip hidden files/directories
        if item.name.startswith("."):
            continue

        if item.is_file():
            detected_type = detect_source_type(str(item))
            if detected_type:
                # source_id auto-generated from url hash in __post_init__
                discovered.append(
                    SourceConfig(
                        type=detected_type,
                        url=str(item),
                        dim_labels=dim_labels,
                    )
                )
            elif item.name.lower().endswith(".h5") or item.name.lower().endswith(
                ".hdf5"
            ):
                skipped_hdf5.append(item)
            else:
                skipped_unknown.append(item)

        elif item.is_dir():
            # Check if this directory itself is a data source
            detected_type = detect_source_type(str(item))
            if detected_type:
                # source_id auto-generated from url hash in __post_init__
                discovered.append(
                    SourceConfig(
                        type=detected_type,
                        url=str(item),
                        dim_labels=dim_labels,
                    )
                )
            elif recursive:
                # Not a data source itself, but might contain them - recurse
                discovered.extend(
                    scan_directory_for_sources(item, dim_labels, recursive)
                )

    # Print warnings for skipped files (only at top level to avoid spam)
    if skipped_hdf5 and directory == directory:
        print(
            "Warning: Skipped HDF5 files (require explicit 'type' and 'dataset' in config):"
        )
        for h5_path in skipped_hdf5[:5]:  # Limit to 5 to avoid spam
            print(f"  - {h5_path}")
        if len(skipped_hdf5) > 5:
            print(f"  ... and {len(skipped_hdf5) - 5} more")

    return discovered


def _discover_by_type(
    path: Path,
    source_type: str,
    dim_labels: Optional[List[str]] = None,
    dataset: Optional[str] = None,
) -> List[SourceConfig]:
    """Discover sources by type in a directory (backward compatibility helper).

    Args:
        path: Directory to scan (local filesystem only)
        source_type: Type of sources to look for
        dim_labels: Optional dimension labels
        dataset: HDF5 dataset path (for HDF5 type)

    Returns:
        List of discovered SourceConfig objects
    """
    import json

    discovered = []

    if source_type == "zarr":
        for zarr_path in sorted(path.glob("*.zarr")):
            if zarr_path.is_dir():
                discovered.append(
                    SourceConfig(
                        type="zarr",
                        url=str(zarr_path),
                        dim_labels=dim_labels,
                    )
                )

    elif source_type == "ome-zarr":
        for zarr_path in sorted(path.glob("*.zarr")):
            if zarr_path.is_dir():
                zattrs_path = zarr_path / ".zattrs"
                if zattrs_path.exists():
                    try:
                        with open(zattrs_path) as f:
                            zattrs = json.load(f)
                        if "multiscales" in zattrs:
                            discovered.append(
                                SourceConfig(
                                    type="ome-zarr",
                                    url=str(zarr_path),
                                    dim_labels=dim_labels,
                                )
                            )
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass

    elif source_type == "hdf5":
        for pattern in ["*.h5", "*.hdf5"]:
            for h5_path in sorted(path.glob(pattern)):
                if h5_path.is_file():
                    discovered.append(
                        SourceConfig(
                            type="hdf5",
                            url=str(h5_path),
                            dim_labels=dim_labels,
                            dataset=dataset,
                        )
                    )

    elif source_type == "ome-tiff":
        metadata_file = path / "_metadata.txt"
        img_files = list(path.glob("img_*.ome.tiff")) + list(path.glob("img_*.ome.tif"))

        if metadata_file.exists() or len(img_files) > 1:
            discovered.append(
                SourceConfig(
                    type="ome-tiff-multifile",
                    url=str(path),
                    dim_labels=dim_labels,
                )
            )
        else:
            for pattern in ["*.ome.tiff", "*.ome.tif", "*.tif", "*.tiff"]:
                for tiff_path in sorted(path.glob(pattern)):
                    if tiff_path.is_file():
                        discovered.append(
                            SourceConfig(
                                type="ome-tiff",
                                url=str(tiff_path),
                                dim_labels=dim_labels,
                            )
                        )

    elif source_type == "aics":
        # Discover aicsimageio-supported files (one source per file)
        aics_extensions = ["*.czi", "*.lif", "*.nd2", "*.dv", "*.lsm"]
        for pattern in aics_extensions:
            for file_path in sorted(path.glob(pattern)):
                if file_path.is_file():
                    discovered.append(
                        SourceConfig(
                            type="aics",
                            url=str(file_path),
                            dim_labels=dim_labels,
                        )
                    )

    return discovered


def _namespaced_source_id(alias: Optional[str], upstream_source_id: str) -> str:
    """Local source_id for a mirrored upstream source.

    The proxy serves many upstreams + local sources from one flat, source_id-keyed
    catalog, so an upstream's ids are namespaced by the configured ``alias``:
    ``<alias>__<upstream_source_id>`` (slash-free -- ``__`` is a cosmetic
    separator, and the upstream id is slash-free by the array_id spec). A lone
    upstream with no alias keeps the verbatim id.
    """
    return f"{alias}__{upstream_source_id}" if alias else upstream_source_id


def _discover_tensor_server(
    source: SourceConfig, credentials_config: Optional[Any]
) -> List[SourceConfig]:
    """Expand a ``tensor-server`` source into one concrete source per upstream tensor.

    ``grpc://host:port/<id>`` mirrors a single upstream source; a bare
    ``grpc://host:port`` connects to the upstream and mirrors *every* source it
    lists (the network analogue of directory discovery). Each concrete source
    carries the single-source url form and an alias-namespaced ``source_id`` so
    the adapter (``RemoteTensorAdapter.create_from_config``) and the rest of the
    server machinery treat it like any other source.
    """
    from biopb_tensor_server.adapters.remote_tensor import (
        _resolve_upstream_token,
        _split_grpc_url,
        list_upstream_source_ids,
    )

    endpoint, upstream_source_id = _split_grpc_url(source.url)

    if upstream_source_id is not None:
        # Single-source form: register under the alias-namespaced local id.
        local_id = _namespaced_source_id(source.alias, upstream_source_id)
        return [replace(source, source_id=local_id)]

    # Bare-host form: mirror every source on the upstream. Enumerate via the
    # complete server-side catalog (not the capped list_sources -- see
    # list_upstream_source_ids); an incomplete fallback list still mirrors what
    # it can.
    from biopb.tensor import TensorFlightClient

    token = _resolve_upstream_token(source, credentials_config)
    client = TensorFlightClient(endpoint, cache_bytes=0, token=token)
    try:
        ids, _complete = list_upstream_source_ids(client)
        upstream_ids = sorted(ids)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()

    expanded = []
    for upstream_id in upstream_ids:
        local_id = _namespaced_source_id(source.alias, upstream_id)
        expanded.append(
            replace(
                source,
                url=f"{endpoint}/{upstream_id}",
                source_id=local_id,
                type="tensor-server",
            )
        )
    return expanded


def _check_tensor_server_id_collisions(sources: List[SourceConfig]) -> None:
    """Raise if a tensor-server source_id collides with any other source.

    With distinct aliases the namespaces are disjoint by construction, so this
    only fires on a genuine misconfiguration. The message names the offending id
    and the fix.
    """
    by_id: Dict[str, SourceConfig] = {}
    for src in sources:
        prior = by_id.get(src.source_id)
        if prior is not None and "tensor-server" in (src.type, prior.type):
            raise ValueError(
                f"Duplicate source_id {src.source_id!r} from {prior.url} and "
                f"{src.url}. Set a distinct 'alias' on the conflicting "
                f"tensor-server entry to namespace its mirrored sources."
            )
        by_id[src.source_id] = src


def discover_sources(
    source: SourceConfig,
    registry: Optional[AdapterRegistry] = None,
    credentials_config: Optional[Any] = None,
) -> List[SourceConfig]:
    """Expand a source config to actual data sources.

    Uses claim-based discovery for directory scanning, while maintaining
    backward compatibility for explicit configs.

    Supports multiple modes:
    - Explicit source: type + source_id both set -> single source
    - Remote URL: requires explicit 'type' in config (cannot auto-discover)
    - tensor-server (grpc://) URL: a caching proxy in front of an upstream biopb
      tensor server -- a bare ``grpc://host:port`` mirrors *every* upstream source
      (one concrete source each, alias-namespaced), and ``grpc://host:port/<id>``
      mirrors a single upstream source.
    - Local file with no type: try to detect type from file characteristics
    - Local directory with type but no source_id: discover by type (backward compatible)
    - Local directory with no type and no source_id: claim-based discovery

    Args:
        source: Source configuration
        registry: Optional adapter registry (uses default if None)
        credentials_config: Optional CredentialsConfig, used to authenticate the
            upstream ``list_sources`` call when expanding a tensor-server source.

    Returns:
        List of concrete SourceConfig objects (one per data source, NOT expanded to tensors)

    Raises:
        ValueError: If remote URL lacks explicit 'type'
    """
    if registry is None:
        registry = get_default_registry()

    # Case 0: Remote URLs require an explicit (or auto-detectable) type.
    # A grpc(+tls) endpoint auto-detects to "tensor-server"; other remote
    # schemes (s3://, http://, ...) still require an explicit 'type'.
    if source.is_remote:
        resolved_type = source.type or detect_source_type(source.url)
        if resolved_type is None:
            raise ValueError(
                f"Remote URL requires explicit 'type' in config: {source.url}"
            )
        if source.type is None:
            source = replace(source, type=resolved_type)
        if resolved_type == "tensor-server":
            return _discover_tensor_server(source, credentials_config)
        # For other remote URLs, return the source as-is (no directory discovery)
        return [source]

    # Local filesystem handling
    local_path = source.local_path
    if local_path is None:
        raise ValueError(f"Could not resolve local path from: {source.url}")

    if not local_path.exists():
        raise ValueError(f"Path does not exist: {local_path}")

    # Case 1: Explicit source (type + source_id both set)
    if source.type and source.source_id:
        return [source]

    # Case 2: File with no type - try claim-based detection
    if local_path.is_file():
        # Try claim-based detection first. cloud_root carries the multi-file ban
        # (OME-TIFF/DICOM-series -> single file) onto a directly-configured cloud
        # file, matching the monitored path.
        ctx = ClaimContext(local_path, cloud_root=source.cloud)
        state = DiscoveryState()
        try:
            identity = get_file_identity(local_path)
            state.visited_identities.add(identity)
        except OSError:
            pass

        claims = registry.get_claims_for_path(ctx, state)
        if claims:
            claim = claims[0]
            return [_claim_to_source_config(claim, source)]

        # Fallback to legacy detection
        detected_type = detect_source_type(source.url)
        if detected_type:
            return [
                SourceConfig(
                    type=detected_type,
                    url=source.url,
                    source_id=source.source_id,
                    dim_labels=source.dim_labels,
                    dataset=source.dataset,
                )
            ]
        raise ValueError(
            f"Could not detect type for file: {local_path}. "
            f"Please specify 'type' explicitly in config."
        )

    # Case 3: Directory with type but no source_id - discover by type (backward compatible)
    if source.type and not source.source_id:
        return _discover_by_type(
            local_path, source.type, source.dim_labels, source.dataset
        )

    # Case 4: Directory with no type and no source_id - use claim-based discovery
    # First check if the directory itself is a data source
    ctx = ClaimContext(local_path, cloud_root=source.cloud)
    state = DiscoveryState()
    try:
        identity = get_file_identity(local_path)
        state.visited_identities.add(identity)
    except OSError:
        pass

    claims = registry.get_claims_for_path(ctx, state)
    if claims:
        claim = claims[0]
        return [_claim_to_source_config(claim, source)]

    # Directory is not itself a data source - do recursive claim-based scan. Under
    # a cloud root, admit dehydrated placeholders so the one-shot startup scan of a
    # monitor=false cloud directory still catalogues offline data as unresolved
    # sources, and set cloud_root so the multi-file OME-TIFF / DICOM-series ban
    # applies -- the same gating the monitored rescan uses (cloud-storage phase 2).
    state = claim_based_discover(
        local_path,
        registry,
        dim_labels=source.dim_labels,
        admit_nonresident=source.cloud,
        cloud_root=source.cloud,
    )
    return [_claim_to_source_config(claim, source) for claim in state.get_all_claims()]


def _claim_to_source_config(
    claim: SourceClaim, original_source: SourceConfig
) -> SourceConfig:
    """Convert a SourceClaim to SourceConfig.

    Args:
        claim: SourceClaim from discovery
        original_source: Original SourceConfig for dim_labels and credentials_profile inheritance

    Returns:
        SourceConfig with claim information
    """
    source_id = claim.source_id or generate_source_id(
        str(claim.primary_path), claim.source_type
    )

    # Handle HDF5 special case - needs dataset path
    dataset = None
    if claim.source_type == "hdf5":
        # HDF5 claims have needs_dataset flag in extra_config
        if claim.extra_config.get("needs_dataset"):
            # This will fail at adapter creation unless dataset is provided
            # For backward compatibility, we pass through any original dataset
            dataset = original_source.dataset

    return SourceConfig(
        type=claim.source_type,
        url=str(claim.primary_path),
        source_id=source_id,
        dim_labels=claim.dim_labels or original_source.dim_labels,
        dataset=dataset,
        credentials_profile=original_source.credentials_profile,  # preserve credentials_profile
        cloud=original_source.cloud,  # propagate cloud gating to expanded sources
    )


def resolve_all_sources(
    config: ServerConfig,
    registry: Optional[AdapterRegistry] = None,
    *,
    sources: Optional[List[SourceConfig]] = None,
    tolerant: bool = False,
) -> List[SourceConfig]:
    """Resolve all sources in config, expanding directories.

    Uses claim-based discovery for automatic source detection.

    Args:
        config: Server configuration
        registry: Optional adapter registry (uses default if None)
        sources: Optional explicit list of source entries to expand. When None,
            ``config.sources`` is used. The serve path passes a filtered subset
            (local ``monitor=true`` directories are discovered by the rescan
            instead, not expanded here) to avoid an extra pre-bind walk that also
            crashes on a not-yet-mounted directory (biopb/biopb#54).
        tolerant: When True, a source that fails to resolve (e.g. a missing
            static path) is logged and skipped instead of aborting the whole
            expansion. Used by the serve path so one bad entry cannot take down
            the server; ``validate``/``list_tensors`` keep the default (False) so
            a broken source is surfaced as a hard error.

    Returns:
        List of all concrete SourceConfig objects (one per data source)
    """
    if registry is None:
        registry = get_default_registry()

    source_list = sources if sources is not None else config.sources
    credentials_config = getattr(config, "credentials", None)

    all_sources = []
    hdf5_warnings = []

    for source in source_list:
        try:
            discovered = discover_sources(source, registry, credentials_config)
        except Exception as e:
            if not tolerant:
                raise
            logger.warning(
                "Skipping source that could not be resolved: %s (%s)",
                source.url,
                e,
            )
            continue
        for src in discovered:
            # Track HDF5 sources that need dataset config
            if src.type == "hdf5" and src.dataset is None:
                hdf5_warnings.append(src.url)
            all_sources.append(src)

    # source_id collisions involving a tensor-server proxy are a misconfiguration
    # (two upstreams sharing an alias, or a local source named like a proxied id):
    # the catalog is one flat source_id space, so a clash would silently shadow a
    # source. Fail fast with the fix (set/disambiguate the alias). Non-proxy
    # collisions keep the historical last-wins behavior.
    _check_tensor_server_id_collisions(all_sources)

    # Print warnings for HDF5 files that need explicit dataset
    if hdf5_warnings:
        print("Warning: HDF5 files require explicit 'dataset' path in config:")
        for h5_url in hdf5_warnings[:5]:
            print(f"  - {h5_url}")
        if len(hdf5_warnings) > 5:
            print(f"  ... and {len(hdf5_warnings) - 5} more")

    return all_sources
