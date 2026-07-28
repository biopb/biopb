"""Configuration management for TensorFlight server.

Reads JSON config files (``biopb.json``) carrying:
- Server settings (host, port)
- Data source definitions (explicit files or directory auto-discovery)
- Credential profiles for remote storage (S3, GCS, etc.)

JSON is the only supported format; a pre-#34 ``biopb.toml`` is converted with
``biopb server migrate-config`` (see biopb/biopb#34).

Example config (explicit):
```json
{
  "server": { "host": "0.0.0.0", "port": 8815 },
  "sources": [
    {
      "type": "zarr",
      "url": "/data/images.zarr",
      "alias": "my-image",
      "dim_labels": ["z", "y", "x"]
    },
    { "type": "hdf5", "url": "/data/sample.h5", "dataset": "/images/channel0" }
  ]
}
```
(``alias`` is a friendly display name; ``source_id`` is derived from the URL.)

Example config (relaxed auto-discovery):
```json
{
  "server": { "host": "0.0.0.0", "port": 8815 },
  "sources": [
    { "url": "/data/" },
    { "type": "hdf5", "url": "/data/sample.h5", "dataset": "/images" }
  ]
}
```
A bare ``url`` with no ``type`` triggers recursive auto-discovery; HDF5 always
needs an explicit ``type`` + ``dataset`` (it is not auto-detected).

Example config (remote storage):
```json
{
  "server": { "host": "0.0.0.0", "port": 8815 },
  "credentials": {
    "default_profile": "aws-prod",
    "profiles": [
      {
        "name": "aws-prod",
        "storage_type": "s3",
        "key": "AKIAIOSFODNN7EXAMPLE",
        "secret": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "region": "us-east-1"
      }
    ]
  },
  "sources": [
    {
      "type": "ome-zarr",
      "url": "s3://bucket/experiment.ome.zarr",
      "credentials_profile": "aws-prod"
    }
  ]
}
```
"""

from __future__ import annotations

import copy
import getpass
import json
import logging
import os
import tempfile
from dataclasses import MISSING as _DC_MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# The constraint primitives and the shared pyramid-knob bounds live in the core
# biopb package so biopb-mcp (which cannot depend on this server package -- not
# on PyPI) validates the same pyramid rows against the same rules, with no drift
# (biopb/biopb#34, #182). `_Range`/`_Enum` stay as local aliases so the
# `_CONSTRAINTS` table and config_schema keep their existing spelling.
from biopb._config_constraints import (
    PYRAMID_CONSTRAINTS,
    Enum as _Enum,
    Range as _Range,
)

# The one validation scheme, shared with biopb-mcp and the control's admin
# endpoints: check at the read step, warn and fall back to the default, stay
# strict only where a human submitted the value (biopb/biopb#34).
from biopb._config_io import atomic_write_json
from biopb._config_validate import (
    MISSING,
    Problem,
    check_sections,
    warn_and_clamp,
)

# Config file location & format preference live in the core `biopb` package so
# the umbrella CLI shares one definition (both depend on `biopb`). Re-exported
# for back-compat (`biopb_tensor_server.core.config.find_config` and the name
# constants). See biopb._locations for the JSON-canonical rationale
# (biopb/biopb#34).
from biopb._locations import (
    CANONICAL_CONFIG_NAME as CANONICAL_CONFIG_NAME,
    DEFAULT_CONFIG_DIR as DEFAULT_CONFIG_DIR,
    LEGACY_CONFIG_NAME as LEGACY_CONFIG_NAME,
    find_config as find_config,
)

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    discover_sources as claim_based_discover,
    generate_source_id,
    get_file_identity,
)
from biopb_tensor_server.core.remote import (
    CredentialProfile,
    CredentialsConfig,
    is_remote_url,
)

# Alias for backward compatibility with internal usage
_is_remote_url = is_remote_url

logger = logging.getLogger(__name__)


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
# The declarative fix is this table, checked at the read step (parse_config) by
# the shared biopb._config_validate walker -- the same walker and the same policy
# biopb-mcp and the control's admin endpoints use, so a knob is judged identically
# wherever it is met. The same table also feeds the JSON Schema emitter
# (config_schema.py), so the constraints are declared exactly once.
#
# Policy: warn and use the default (never raise). See _config_validate's module
# docstring for why -- in short, this server is a control-plane child that is
# restarted on crash with capped backoff, so refusing to load would turn one bad
# number into a permanent restart loop whose real cause is buried in a log. The
# bad value still never reaches the request path, which was the actual ask.
# `validate` and the admin PUT stay strict: a human is there to act on it.

# Methods PyramidConfig.reduction_method accepts (matched case-insensitively):
# the *computable* subset of the protocol vocabulary, plus its aliases.
# Intentionally narrower than downsample.normalize_reduction_method, which also
# accepts "precompute"/"precomputed" -- that value is a protocol concern (a
# client requesting a native on-disk level), not a way the server can compute a
# pyramid level, so it is invalid here. "linear" stays as a tolerated deprecated
# alias: old configs keep validating, and normalize_reduction_method folds it
# to "area" with a warning at read time.
_REDUCTION_METHODS = {
    "area",
    "linear",
    "nearest",
    "stride",
    "decimate",
    "mean",
}


# Per-dataclass field constraints, keyed by class name (so this table can sit
# above the class definitions). full_rescan_interval is intentionally absent:
# a value <= 0 *disables* the periodic full-scan backstop (documented sentinel).
# `_Range`/`_Enum` and the pyramid rows (PYRAMID_CONSTRAINTS) come from
# biopb._config_constraints so biopb-mcp validates the same knobs identically.
_CONSTRAINTS = {
    "CacheConfig": {
        "backend": _Enum({"memory", "file"}),
        "memory_max_entries": _Range(min=1),
        "memory_max_bytes": _Range(min=1),
        "file_max_segment_bytes": _Range(min=1),
        "file_max_total_bytes": _Range(min=1),
    },
    "PyramidConfig": {
        # reduction_method is server-local (on-the-fly reduction is a compute
        # concern, not a biopb-mcp knob); the numeric rows are shared.
        "reduction_method": _Enum(_REDUCTION_METHODS, case_insensitive=True),
        **PYRAMID_CONSTRAINTS,
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
        # 0 is not a typo-shaped value here: it is the "let the OS assign an
        # ephemeral port" sentinel the flight server binds on (used by the test
        # harness and by anyone running two planes on one box), so the floor is
        # 0, not 1. Above 65535 there is no such reading -- that is a typo.
        "port": _Range(min=0, max=65535),
        "log_level": _Enum(
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, case_insensitive=True
        ),
        "rescan_interval": _Range(min=0),
        "stability_window": _Range(min=0),
        "stable_rescans_required": _Range(min=0),
        "handle_reaper_ttl": _Range(min=0),
    },
}

# Class name -> the config section it maps to. Read by the schema emitter
# (config_schema.ondisk_location) and by the messages here.
_SECTION_FOR = {
    "CacheConfig": "cache",
    "PyramidConfig": "pyramid",
    "PrecacheConfig": "precache",
    "MetadataDbConfig": "metadata_db",
    "ServerConfig": "server",
}

# The nested sections the checker walks. ServerConfig itself is the "server"
# section (its scalars are top-level fields, not a nested dataclass), so it is
# passed separately in _sections_of.
_NESTED_SECTIONS = ("cache", "pyramid", "precache", "metadata_db")


def _sections_of(config: ServerConfig) -> List[Tuple[str, Any]]:
    """``(section_name, section_object)`` pairs for :func:`check_sections`.

    The dataclass instances are passed as-is -- the shared walker reads a
    dataclass or a dict alike, so the tensor server needs no dict projection of
    its config just to validate it.
    """
    return [("server", config)] + [
        (name, getattr(config, name)) for name in _NESTED_SECTIONS
    ]


def _config_problems(config: ServerConfig) -> List[Problem]:
    """Every :data:`_CONSTRAINTS` violation in *config* (empty when valid)."""
    return check_sections(_sections_of(config), _CONSTRAINTS)


def _dataclass_default(cls, key: str) -> Any:
    """The declared default of field *key* on dataclass *cls*.

    Read off the field rather than off a constructed instance so resolving one
    bad leaf does not build a whole config (and its nested sections) again.
    """
    for f in fields(cls):
        if f.name != key:
            continue
        if f.default is not _DC_MISSING:
            return f.default
        if f.default_factory is not _DC_MISSING:  # type: ignore[misc]
            return f.default_factory()  # type: ignore[misc]
    return MISSING


def _clamp_invalid(config: ServerConfig) -> None:
    """Warn about each violation and reset that field to its dataclass default.

    The load-path policy (see :mod:`biopb._config_validate`): a bad knob must not
    reach the request path, but must also not stop the server from coming up --
    it is supervised, and refusing would just be restarted into the same failure.
    Falling back to the *dataclass* default means "the default" is exactly what
    an omitted key would have produced.
    """

    def _target(section: str):
        return config if section == "server" else getattr(config, section)

    def default_for(path: Tuple[str, ...]) -> Any:
        section, key = path
        return _dataclass_default(type(_target(section)), key)

    def apply(path: Tuple[str, ...], value: Any) -> None:
        section, key = path
        setattr(_target(section), key, value)

    warn_and_clamp(_config_problems(config), default_for, apply, logger)


@dataclass
class SourceConfig:
    """Configuration for a single data source.

    A source may contain multiple tensors (multifield support) - the adapter
    handles tensor enumeration via list_tensor_descriptors() at runtime.

    Remote/cloud source features are EXPERIMENTAL: remote URLs, ``cloud = true``
    synced-folder roots, the ``tensor-server`` proxy type, its ``alias``, and
    ``credentials_profile``. Local-file sources are stable.

    Per-field help lives in each field's ``metadata["help"]`` -- the single
    source the config JSON Schema reads (see ``config_schema.py``); a few fields
    carry extra maintainer rationale in inline comments below.

    The ``alias`` help is deliberately terse; the full behavior: for a
    tensor-server proxy it is the namespace prefix mirroring the upstream's
    sources under ``<alias>__<upstream_source_id>`` (so multiple upstreams share
    one flat source_id-keyed catalog); for a local source it is the catalog tree
    root the source is re-rooted under (display-only, honored on the static /
    one-shot-expand path -- a ``monitor = true`` directory re-merges into the
    shared tree on rescan and the alias is ignored with a warning).
    """

    url: str = field(
        metadata={
            "help": "URL or path to the data source. Local paths are stable; "
            "remote URLs (s3://, http(s)://, grpc://) are experimental."
        }
    )
    type: Optional[
        Literal[
            "zarr",
            "hdf5",
            "ome-tiff",
            "ome-tiff-multifile",
            "ome-zarr",
            "ome-zarr-hcs",
            "aics",
            "qptiff",
            "mrc",
            "emd",
            "tensor-server",
        ]
    ] = field(
        default=None,
        metadata={
            "help": "Storage type; auto-detected for local files when omitted. "
            "('tensor-server' is experimental.)"
        },
    )
    source_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Deprecated and ignored: a source's id is derived from its "
            "resolved URL (biopb/biopb#308). Use `alias` for a display name."
        },
    )
    dim_labels: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Dimension labels applied to all tensors in the source."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "HDF5 dataset path (required for HDF5 sources)."},
    )
    monitor: bool = field(
        default=False,
        metadata={
            "help": "Watch this local directory for add/delete events and update "
            "the catalog automatically."
        },
    )
    cloud: bool = field(
        default=False,
        metadata={
            "help": "(experimental) Treat as a cloud/synced-folder root: admit "
            "offline placeholders, resolved lazily on first access."
        },
    )
    credentials_profile: Optional[str] = field(
        default=None,
        metadata={
            "help": "(experimental) Credential profile for a remote-URL source "
            "(overrides the default profile)."
        },
    )
    alias: Optional[str] = field(
        default=None,
        metadata={
            "help": "(experimental) Display name this source appears under: a "
            "namespace prefix for a tensor-server upstream, or the catalog tree "
            "root for a local source. Must be slash-free."
        },
    )
    # Internal/derived (leading underscore): not user-facing config keys, so they
    # are excluded from the config-schema drift guard.
    # _catalog_url: display-only tree-root override, derived from `alias` for a
    # local source during expansion; threaded to the descriptor's source_url,
    # never affects source_id.
    _catalog_url: Optional[str] = None
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

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema). Note the on-disk key names differ for the size fields
    (``memory_max_bytes`` -> ``max_bytes``, ``file_max_segment_bytes`` ->
    ``file_max_segment_mb``, ``file_max_total_bytes`` -> ``file_max_total_gb``);
    the help is phrased for the on-disk form the editor shows.
    """

    backend: str = field(
        default_factory=_default_cache_backend,
        metadata={
            "help": "Chunk cache backend: 'memory' (in-process only) or 'file' "
            "(adds an on-disk cache)."
        },
    )
    memory_max_entries: int = field(
        default=1024,
        metadata={"help": "Maximum number of decoded chunks kept in memory."},
    )
    memory_max_bytes: int = field(
        default=512 * 1024 * 1024,  # 512 MB
        metadata={"help": "Maximum total bytes of decoded chunks kept in memory."},
    )
    file_cache_dir: Path = field(
        default=DEFAULT_FILE_CACHE_DIR,
        metadata={"help": "Directory for the on-disk chunk cache (file backend)."},
    )
    file_max_segment_bytes: int = field(
        default=64 * 1024 * 1024,  # 64 MB per segment
        metadata={"help": "Maximum size of one on-disk cache segment file (MB)."},
    )
    file_max_total_bytes: int = field(
        default=4 * 1024 * 1024 * 1024,  # 4 GB total
        metadata={"help": "Maximum total size of the on-disk chunk cache (GB)."},
    )

    def __post_init__(self):
        if isinstance(self.file_cache_dir, str):
            self.file_cache_dir = Path(self.file_cache_dir)


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

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema).
    """

    reduction_method: str = field(
        default="area",
        metadata={
            "help": "Downsampling method for computed levels ('area' = averaging). "
            "Native on-disk levels are served precomputed regardless."
        },
    )
    threshold: int = field(
        default=4096,
        metadata={"help": "Maximum X/Y extent (pixels) of the coarsest level."},
    )
    downscale_factor: int = field(
        default=4,
        metadata={"help": "Per-level linear downsampling step, per spatial axis."},
    )
    pixel_budget_cubic_root: int = field(
        default=512,
        metadata={
            "help": "Cube root of the coarsest level's voxel budget "
            "(Lx*Ly*Lz <= this**3); bounds a whole-volume 3-D read."
        },
    )


@dataclass
class PrecacheConfig:
    """Background pre-cache worker configuration.

    The worker warms the file cache for newly-added sources at the *coarsest*
    pyramid level a client requests on open, so the first view is already warm.
    It is inert unless the file cache backend is in use, and stays off the wire
    while live reads are in flight. The level *definition* (which scale, which
    reduction) lives in :class:`PyramidConfig`; this holds only the worker's
    operational knobs.

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema).
    """

    enabled: bool = field(
        default=True,
        metadata={
            "help": "Run the background pre-cache worker to warm new sources "
            "(no-op on the memory backend)."
        },
    )
    idle_debounce_seconds: float = field(
        default=2.0,
        metadata={
            "help": "Quiet period after live traffic before the worker resumes "
            "(seconds)."
        },
    )
    # Startup-backlog (existing sources) knobs.
    backlog_enabled: bool = field(
        default=True,
        metadata={
            "help": "Also warm sources already present at startup, behind live "
            "additions."
        },
    )
    backlog_high_water: float = field(
        default=0.8,
        metadata={
            "help": "Stop backlog warming once the file cache fills past this "
            "fraction of its budget (0-1), so precache never evicts live data."
        },
    )
    backlog_idle_recheck_seconds: float = field(
        default=5.0,
        metadata={
            "help": "Over the high-water mark, seconds the backlog naps before "
            "re-checking for freed room."
        },
    )


@dataclass
class MetadataDbConfig:
    """Configuration for DuckDB metadata database and source catalog safety limits.

    Enables efficient SQL filtering for large source catalogs (>100k sources).
    Replaces O(n) in-memory scans with indexed DuckDB queries.

    The metadata database is **mandatory** (biopb/biopb#225): it is the canonical
    source-browsing surface (``client.query_sources``), so there is no ``enabled``
    flag -- the DB is always constructed. A lingering ``metadata_db.enabled`` key
    in an old config is ignored with a warning (see ``parse_config``).

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema).
    """

    max_query_results: int = field(
        default=100000,
        metadata={"help": "Safety cap on rows returned by a catalog SQL query."},
    )
    max_list_flights_results: int = field(
        default=100000,
        metadata={"help": "Safety cap on sources returned by ListFlights."},
    )
    query_timeout_ms: int = field(
        default=30000,
        metadata={"help": "Catalog SQL query timeout (milliseconds)."},
    )


@dataclass
class ServerConfig:
    """Server configuration.

    Per-field help lives in each field's ``metadata["help"]`` -- the single
    source the config JSON Schema reads (see ``config_schema.py``). The nested
    section objects (``cache``/``pyramid``/``precache``/``credentials``/
    ``metadata_db``) and ``sources`` document themselves.
    """

    host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "Address the Flight gRPC server binds. Loopback (127.0.0.1) "
            "is local mode; a public address requires a token (remote mode)."
        },
    )
    port: int = field(
        default=8815,
        metadata={"help": "TCP port for the Flight gRPC data-plane server."},
    )
    log_level: str = field(
        default="INFO",
        metadata={"help": "Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)."},
    )
    log_scope_to_biopb: bool = field(
        default=True,
        metadata={
            "help": "Apply the log level only to biopb loggers, leaving "
            "third-party libraries (grpc, numpy, ...) at their defaults."
        },
    )
    monitor_mode: str = field(
        default="periodic",
        metadata={
            "help": "How monitored folders are watched: 'periodic' rescans, or "
            "'off' to stop background rescans after initial discovery."
        },
    )
    rescan_interval: float = field(
        default=30.0,
        metadata={"help": "Seconds between background rescans of monitored folders."},
    )
    full_rescan_interval: float = field(
        default=3600.0,
        metadata={
            "help": "Seconds between forced full rescans that bypass subtree "
            "pruning (<= 0 disables this backstop)."
        },
    )
    handle_reaper_ttl: float = field(
        default=150.0,
        metadata={
            "help": "Seconds an idle persistent file handle (OME-TIFF store, "
            "NDTiff acquisition) is kept warm before it is closed; the next read "
            "reopens it (0 disables reaping). Adapters that reopen per read "
            "(hdf5, mrc, ...) are unaffected."
        },
    )
    stability_window: float = field(
        default=30.0,
        metadata={
            "help": "Minimum quiet period before a path is eligible for discovery "
            "or removal (seconds)."
        },
    )
    stable_rescans_required: int = field(
        default=0,
        metadata={
            "help": "Extra unchanged rescans required before a path is considered "
            "stable (0 relies on the stability window alone)."
        },
    )
    probe_open_files: bool = field(
        default=True,
        metadata={
            "help": "Best-effort append-open probe to skip files still being "
            "written (advisory)."
        },
    )
    aggressive_dir_pruning: bool = field(
        default=False,
        metadata={
            "help": "Also prune unchanged monitored roots (faster scans; may "
            "defer root-level file updates to a later scan)."
        },
    )
    claim_generic_images: bool = field(
        default=False,
        metadata={
            "help": "Also catalog generic raster/video files "
            "(.png/.jpg/.gif/.bmp/.mp4/...) during discovery. Off by default -- "
            "they are rarely microscopy tensors (biopb/biopb#40)."
        },
    )
    writable: bool = field(
        default=False,
        metadata={"help": "Enable write mode: allow source creation and data upload."},
    )
    write_dir: Optional[Path] = field(
        default=None,
        metadata={
            "help": "Directory for zarr-backed uploaded sources (unset = no zarr "
            "uploads)."
        },
    )
    cache: CacheConfig = field(default_factory=CacheConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    precache: PrecacheConfig = field(default_factory=PrecacheConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    metadata_db: MetadataDbConfig = field(default_factory=MetadataDbConfig)
    sources: List[SourceConfig] = field(default_factory=list)


def load_config(path: Path) -> ServerConfig:
    """Load configuration from a JSON file.

    JSON is the only format read (biopb/biopb#34); a legacy ``biopb.toml``
    raises with the migration command. The file is parsed to a plain dict and
    handed to the format-agnostic :func:`parse_config`.

    Args:
        path: Path to a JSON config file (``biopb.json``)

    Returns:
        ServerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If the file is not valid JSON (including a legacy TOML), or
            if it carries an out-of-range / bad-enum value
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
    from biopb_tensor_server.core.config_schema import build_config_schema

    # Embed a relative $schema pointer (offline editor validation, no hosted URL).
    payload = dict(data)
    payload["$schema"] = f"./{SCHEMA_SIDECAR_NAME}"

    atomic_write_json(schema_path, build_config_schema(), raise_on_error=True)
    atomic_write_json(path, payload, raise_on_error=True)

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


# Appended to every read failure: a legacy TOML is the one shape of "not JSON"
# with a one-command fix, and a user meeting a parse error has no other way to
# learn the format changed (biopb/biopb#34).
_MIGRATE_HINT = (
    "JSON is the only supported config format; convert a legacy "
    f"{LEGACY_CONFIG_NAME} with `biopb server migrate-config`. See biopb/biopb#34."
)


def _read_config_file(path: Path) -> Dict[str, Any]:
    """Read a config file into a plain dict.

    JSON only. A ``.toml`` path is rejected without parsing (the read path was
    dropped once the deprecation window closed); every other extension --
    including none -- is read as JSON, so an unconventionally-named config still
    loads. Both failures name ``biopb server migrate-config``.
    """
    if path.suffix.lower() == ".toml":
        raise ValueError(
            f"Config file {path} is in the legacy TOML format. " + _MIGRATE_HINT
        )
    return _load_json(path)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except ValueError as e:
        raise ValueError(
            f"Invalid JSON in config file {path}: {e}. " + _MIGRATE_HINT
        ) from e


def read_legacy_toml(path: Path) -> Dict[str, Any]:
    """Read a pre-#34 ``biopb.toml`` into a plain dict.

    The **only** remaining TOML reader, and deliberately not reachable from
    :func:`load_config`: it exists for `biopb server migrate-config`, which
    converts the old file to canonical JSON. ``tomllib`` is imported lazily so
    the server's own read path carries no TOML dependency.
    """
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in config file {path}: {e}") from e


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
        from biopb_tensor_server.core.config_schema import known_config_keys

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


# Sections that used to exist and are now silently dropped by the parser; they
# get a tailored deprecation warning instead of the generic "unknown section".
_DEPRECATED_SECTIONS = {"compute"}


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
        if section in _DEPRECATED_SECTIONS:
            logger.warning(
                "Config section [%s] is deprecated and ignored; the GPU compute "
                "backend was removed.",
                section,
            )
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


def _carry(
    dst: Dict[str, Any],
    field: str,
    src: Dict[str, Any],
    key: Optional[str] = None,
    cast=None,
) -> None:
    """Copy ``src[key]`` into ``dst[field]`` iff the key is present and non-null.

    The heart of the "defaults live in one place" contract (biopb/biopb#277 A):
    an absent (or ``null``) config key is simply *not* forwarded, so the
    dataclass constructor supplies its own default -- no default literal is
    written a second time here. ``key`` defaults to ``field`` (they differ only
    for the on-disk aliases catalogued in ``config_schema._ONDISK_OVERRIDES``);
    ``cast`` applies the field's wire->value coercion (unit scaling, float/int).
    """
    on_disk = key or field
    value = src.get(on_disk)
    if value is not None:
        dst[field] = cast(value) if cast is not None else value


def parse_config(data: Dict[str, Any]) -> ServerConfig:
    """Build a :class:`ServerConfig` from a raw config dict, checked and clamped.

    The read step, and the single place a config from disk (or from the admin
    endpoint) is validated: :func:`_build_config` does the wire->dataclass
    mapping, then :func:`_clamp_invalid` warns about every out-of-range /
    bad-enum knob and substitutes its default, so nothing invalid reaches the
    request path and the server still starts (biopb/biopb#34).
    """
    config = _build_config(data)
    _clamp_invalid(config)
    return config


def _build_config(data: Dict[str, Any]) -> ServerConfig:
    """Construct the config dataclasses from a raw dict, without validating.

    Split from :func:`parse_config` so the value check has something to run
    *against* -- :func:`validate_config_dict` needs the unclamped objects to
    report what was wrong, which it cannot see once the defaults are in place.

    Format-agnostic: ``data`` is a plain dict already read from JSON by
    :func:`load_config`.

    Field **defaults** are owned solely by the config dataclasses: this parser
    forwards only the keys actually present in ``data`` (via :func:`_carry`) and
    lets each dataclass fill the rest, so a default is never declared twice
    (biopb/biopb#277 item A). What stays here is the wire<->dataclass mapping the
    dataclasses cannot express: on-disk key aliases (``cache.max_entries`` ->
    ``memory_max_entries``), unit scaling (``*_mb``/``*_gb`` -> ``*_bytes``),
    legacy back-compat keys (``watcher_type``, ``poll_interval``, the ``[precache]``
    pyramid knobs, source ``path``), and per-field coercions.

    Args:
        data: Config dictionary (from JSON)

    Returns:
        ServerConfig object, values as given
    """
    _warn_unknown_config_keys(data)

    # Parse server settings. Only present keys are carried; ServerConfig supplies
    # every default.
    server_data = data.get("server", {})
    server_kwargs: Dict[str, Any] = {}
    _carry(server_kwargs, "host", server_data)
    _carry(server_kwargs, "port", server_data)
    _carry(server_kwargs, "log_level", server_data)
    _carry(server_kwargs, "log_scope_to_biopb", server_data)

    # monitor_mode: honor the value directly, else derive it from the legacy
    # `watcher_type` alias; if neither is set, ServerConfig's default applies.
    monitor_mode = server_data.get("monitor_mode")
    if monitor_mode is None and "watcher_type" in server_data:
        monitor_mode = "off" if server_data.get("watcher_type") == "off" else "periodic"
    if monitor_mode is not None:
        server_kwargs["monitor_mode"] = monitor_mode

    # rescan_interval: `poll_interval` is the legacy alias.
    _carry(server_kwargs, "rescan_interval", server_data, cast=float)
    if "rescan_interval" not in server_kwargs:
        _carry(
            server_kwargs, "rescan_interval", server_data, "poll_interval", cast=float
        )

    _carry(server_kwargs, "full_rescan_interval", server_data, cast=float)
    _carry(server_kwargs, "handle_reaper_ttl", server_data, cast=float)
    _carry(server_kwargs, "stability_window", server_data, cast=float)
    _carry(
        server_kwargs,
        "stable_rescans_required",
        server_data,
        cast=lambda v: max(0, int(v)),
    )
    _carry(server_kwargs, "probe_open_files", server_data, cast=bool)
    _carry(server_kwargs, "aggressive_dir_pruning", server_data, cast=bool)
    _carry(server_kwargs, "claim_generic_images", server_data, cast=bool)
    _carry(server_kwargs, "writable", server_data)
    write_dir_str = server_data.get("write_dir")
    if write_dir_str:
        server_kwargs["write_dir"] = Path(write_dir_str)

    # Parse cache settings. The wire form of four fields diverges from the
    # dataclass (aliases + MB/GB->bytes scaling); everything else is a direct
    # carry. Mapping mirrors config_schema._ONDISK_OVERRIDES.
    cache_data = data.get("cache", {})
    cache_kwargs: Dict[str, Any] = {}
    _carry(cache_kwargs, "backend", cache_data)
    _carry(cache_kwargs, "memory_max_entries", cache_data, "max_entries")
    _carry(cache_kwargs, "memory_max_bytes", cache_data, "max_bytes")
    _carry(
        cache_kwargs,
        "file_max_segment_bytes",
        cache_data,
        "file_max_segment_mb",
        cast=lambda mb: int(mb) * 1024 * 1024,
    )
    _carry(
        cache_kwargs,
        "file_max_total_bytes",
        cache_data,
        "file_max_total_gb",
        cast=lambda gb: int(gb) * 1024 * 1024 * 1024,
    )
    # CacheConfig.__post_init__ coerces a str file_cache_dir to Path; a falsy
    # value means "unset" -> the dataclass default.
    if cache_data.get("file_cache_dir"):
        cache_kwargs["file_cache_dir"] = Path(cache_data["file_cache_dir"])
    cache_config = CacheConfig(**cache_kwargs)

    # Parse pyramid settings -- the authority for level definition, shared by the
    # advertised TensorDescriptor.pyramid and the precache worker.
    pyramid_data = data.get("pyramid", {})
    precache_data = data.get("precache", {})

    # Back-compat: these knobs used to live under [precache]; honor an old
    # [precache] value when [pyramid] omits the key. An absent knob falls through
    # to PyramidConfig's default.
    pyramid_kwargs: Dict[str, Any] = {}
    for _knob, _cast in (
        ("reduction_method", str),
        ("threshold", int),
        ("downscale_factor", int),
        ("pixel_budget_cubic_root", int),
    ):
        if _knob in pyramid_data:
            pyramid_kwargs[_knob] = _cast(pyramid_data[_knob])
        elif _knob in precache_data:
            pyramid_kwargs[_knob] = _cast(precache_data[_knob])
    pyramid_config = PyramidConfig(**pyramid_kwargs)

    # Parse precache settings (operational knobs only).
    precache_kwargs: Dict[str, Any] = {}
    _carry(precache_kwargs, "enabled", precache_data, cast=bool)
    _carry(precache_kwargs, "idle_debounce_seconds", precache_data, cast=float)
    _carry(precache_kwargs, "backlog_enabled", precache_data, cast=bool)
    _carry(precache_kwargs, "backlog_high_water", precache_data, cast=float)
    _carry(precache_kwargs, "backlog_idle_recheck_seconds", precache_data, cast=float)
    precache_config = PrecacheConfig(**precache_kwargs)

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
    # `metadata_db.enabled` was removed (biopb/biopb#225): the metadata DB is now
    # mandatory (always on) because it is the canonical source-browsing surface --
    # the biopb-mcp guide steers agents to `client.query_sources(sql, ...)`
    # (complete, server-side) over the capped `list_sources()`, and that SQL path
    # only exists when the DB is present. A lingering flag in an old config is
    # ignored (not honored) with a warning; `enabled = false` gets the stronger
    # message because the DB comes up ON regardless -- the opposite of what that
    # config asked for.
    if "enabled" in metadata_db_data:
        if metadata_db_data.get("enabled"):
            logger.warning(
                "Config option `metadata_db.enabled` was removed and is ignored; "
                "the metadata database is now always on. Drop the flag from your "
                "config. See biopb/biopb#225."
            )
        else:
            logger.warning(
                "Config option `metadata_db.enabled = false` is no longer honored: "
                "the metadata database is now mandatory (always on), so the server "
                "starts WITH the SQL catalog (`client.query_sources(...)`) despite "
                "this setting. Drop the flag from your config. See biopb/biopb#225."
            )
    metadata_db_kwargs: Dict[str, Any] = {}
    _carry(metadata_db_kwargs, "max_query_results", metadata_db_data)
    _carry(metadata_db_kwargs, "max_list_flights_results", metadata_db_data)
    _carry(metadata_db_kwargs, "query_timeout_ms", metadata_db_data)
    metadata_db_config = MetadataDbConfig(**metadata_db_kwargs)

    # Parse sources. `url` accepts the legacy `path` alias; every other field is
    # carried only when present so SourceConfig owns the defaults.
    sources_data = data.get("sources", [])
    sources: List[SourceConfig] = []

    for src_data in sources_data:
        # Support both 'url' (new) and 'path' (legacy) for backward compatibility
        url = src_data.get("url") or src_data.get("path")
        if url is None:
            raise ValueError("Source config requires 'url' field")

        src_kwargs: Dict[str, Any] = {"url": url}
        _carry(src_kwargs, "type", src_data)  # auto-detected when omitted
        # `source_id` is derived from the resolved URL (a stable content
        # identity), never user-assigned. Honoring an explicit id let two configs
        # aim the same bytes at two ids -> duplicate catalog rows for one source
        # (biopb/biopb#308). It is ignored with a warning; `alias` is the way to
        # give a source a friendly display name (source_url is the user-facing
        # text).
        if "source_id" in src_data:
            logger.warning(
                "Config key `sources.source_id` is deprecated and ignored: a "
                "source's id is derived from its resolved URL so the same data "
                "always maps to one catalog entry (dropping explicit ids closes "
                "biopb/biopb#308). Use `alias` to give the source a display name."
            )
        _carry(src_kwargs, "dim_labels", src_data)
        _carry(src_kwargs, "dataset", src_data)
        _carry(src_kwargs, "monitor", src_data)
        _carry(src_kwargs, "cloud", src_data)
        _carry(src_kwargs, "credentials_profile", src_data)
        _carry(src_kwargs, "alias", src_data)  # tensor-server proxy namespace
        sources.append(SourceConfig(**src_kwargs))

    return ServerConfig(
        cache=cache_config,
        pyramid=pyramid_config,
        precache=precache_config,
        credentials=credentials_config,
        metadata_db=metadata_db_config,
        sources=sources,
        **server_kwargs,
    )


def validate_config_dict(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a raw config dict with the SAME rules the server enforces at
    load, reported rather than raised.

    Returns ``[{"path": [section, key], "message": str}, ...]`` (empty = valid),
    using the on-disk ``(section, key)`` paths
    :func:`config_schema.ondisk_location` assigns, so a caller can merge/dedupe
    these against JSON-Schema errors by path.

    This is the strict end of the one validation scheme: the load path warns and
    clamps, but a human submitting a value gets it rejected instead, with every
    problem at once so the form can highlight each field. It catches everything
    :data:`_CONSTRAINTS` covers -- notably the case-insensitive enums
    (``log_level``, ``reduction_method``) the published JSON Schema deliberately
    cannot express (``_Enum.to_json_schema`` emits no hard ``enum`` for them, so
    a schema-only check waves a bad value through). The admin config-save
    endpoint runs this alongside JSON-Schema validation so a config the form
    accepts is one the server will actually load. See biopb/biopb#34.
    """
    try:
        # _build_config, not parse_config: the clamp would have already replaced
        # every bad value with its default, leaving nothing to report.
        cfg = _build_config(data)
    except Exception as exc:  # noqa: BLE001 - untrusted input must never crash the gate
        # Structural failure: a missing url or un-coercible number (ValueError /
        # TypeError), or a wrong-typed section that makes the parser walk a
        # non-dict (AttributeError, e.g. {"server": "x"}). Report as a single
        # root-level problem rather than letting it crash the caller (the admin
        # endpoint would otherwise 500 instead of returning a clean 422).
        return [{"path": [], "message": str(exc)}]

    # Lazy import: config_schema imports this module, so importing it at module
    # scope is a cycle (mirrors save_config's build_config_schema import).
    from biopb_tensor_server.core.config_schema import ondisk_location

    # The checker reports dataclass-field paths; the endpoint needs the on-disk
    # ones (CacheConfig.memory_max_entries lives at [cache] max_entries), so the
    # section/key is remapped before it leaves -- in the message too, whose
    # `field=...` lead-in would otherwise name the internal field the on-disk path
    # doesn't (e.g. path [cache] max_entries with "memory_max_entries=...").
    problems: List[Dict[str, Any]] = []
    for problem in _config_problems(cfg):
        section, key = problem.path
        class_name = type(
            cfg if section == "server" else getattr(cfg, section)
        ).__name__
        on_section, on_key = ondisk_location(class_name, key)
        message = problem.message
        if on_key != key and message.startswith(f"{key}="):
            message = f"{on_key}=" + message[len(key) + 1 :]
        problems.append({"path": [on_section, on_key], "message": message})
    return problems


def detect_source_type(url: str) -> Optional[str]:
    """Auto-detect the source type of a *remote* URL from its scheme.

    Filesystem format detection is **not** done here -- that is the adapters'
    sole responsibility, via the ``claim()`` protocol (one source of truth;
    biopb/biopb#277 item B). The only type unambiguous from a URL alone is a
    grpc(+tls) endpoint, which is always an upstream biopb tensor server (the
    "tensor-server" caching-proxy source type). Every other remote scheme
    (``s3://``, ``http://``, ...) and every local path returns ``None``: a
    remote source needs an explicit ``type`` in config, and a local path is
    typed by whichever adapter claims it.
    """
    # A grpc(+tls) endpoint is always an upstream biopb tensor server.
    if url.lower().startswith(("grpc://", "grpc+tls://", "grpcs://")):
        return "tensor-server"

    return None


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
    # EXPERIMENTAL: the tensor-server remote-source proxy is not yet stable. Its
    # config surface (url forms, `alias`, monitor re-list) and the on-disk
    # segment-cache keys for proxied sources may change without notice in a future
    # release (biopb/biopb#178). Warned once per configured upstream at expansion.
    logger.warning(
        "Source %r uses the EXPERIMENTAL tensor-server remote proxy: its config "
        "surface (url forms, 'alias', monitor re-list) and the on-disk cache keys "
        "for proxied sources may change without notice in a future release.",
        source.url,
    )
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
        ids, _complete = list_upstream_source_ids(client, endpoint)
        upstream_ids = sorted(ids)
    finally:
        # Never let a failing close() replace the upstream error propagating out
        # of the try: body (biopb/biopb#529).
        try:
            client.close()
        except Exception:
            logger.debug("error closing upstream client", exc_info=True)

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


def _resolve_tensor_server_id_collisions(
    sources: List[SourceConfig],
) -> List[SourceConfig]:
    """Drop -- don't abort on -- source_id collisions involving a tensor-server proxy.

    With distinct aliases the namespaces are disjoint by construction, so a clash
    is a misconfiguration (two upstreams sharing an alias, or a local source named
    like a proxied id). A single bad entry must not take down the whole catalog,
    so keep the first source for each id and skip later colliders, logging the fix
    (set a distinct alias). Non-proxy collisions are left untouched (the historical
    last-wins at registration).
    """
    seen: Dict[str, SourceConfig] = {}
    result: List[SourceConfig] = []
    for src in sources:
        prior = seen.get(src.source_id)
        if prior is not None and "tensor-server" in (src.type, prior.type):
            logger.warning(
                "Skipping source_id %r from %s (%s): it collides with %s (%s) "
                "already in the catalog. Set a distinct 'alias' on the conflicting "
                "tensor-server entry to namespace its mirrored sources.",
                src.source_id,
                src.url,
                src.type,
                prior.url,
                prior.type,
            )
            continue
        seen.setdefault(src.source_id, src)
        result.append(src)
    return result


def _reroot_catalog_url(label: str, root_path: str, primary_path: str) -> str:
    """Re-root ``primary_path`` under ``label``, preserving its position beneath
    ``root_path``. Shared core of the two re-rooting entry points -- drag-drop
    (``SourceManager._drop_catalog_url``, ``label`` = the dropped item's basename)
    and a configured ``alias`` (``_alias_catalog_url``, ``label`` = the alias).

    The tensor-browser (and web viewer) build their tree by splitting each
    source's ``source_url`` on ``/``, so ``label`` becomes the top-level root and
    the sub-structure beneath ``root_path`` is preserved under it:

        root /data/exp, primary /data/exp            -> "<label>"
        root /data/exp, primary /data/exp/sub/b.tif  -> "<label>/sub/b.tif"

    Display-only: it feeds the descriptor's ``source_url`` and never the
    ``source_id`` (which hashes the raw path), so a bare virtual path with no
    scheme is fine.
    """
    try:
        rel = os.path.relpath(str(primary_path), str(root_path)).replace("\\", "/")
    except ValueError:  # different drive on Windows, etc. -- can't relativize
        rel = "."
    if rel in (".", "") or rel.startswith("../"):
        # primary IS the root (single file / dataset dir), or (defensively) not
        # under it -- keep the whole thing as one root, never emit a "../" url.
        return label
    return f"{label}/{rel}"


def _alias_catalog_url(alias: str, root_path: str, primary_path: str) -> str:
    """Catalog ``source_url`` that re-roots a configured local source under ``alias``.

    The config-line analogue of ``SourceManager._drop_catalog_url``: the root
    label is the configured ``alias`` (rather than a dropped item's basename), and
    the sub-structure of a configured folder is preserved relative to it:

        alias "exp", configure /data/exp/            (root_path == primary_path)
            -> "exp"
        alias "exp", configure folder /data/exp/ with
            .../exp/a.tif, .../exp/sub/b.tif -> "exp/a.tif", "exp/sub/b.tif"

    Display-only (never touches ``source_id``). Applied on the static / one-shot
    expand path; a monitored directory's alias is dropped upstream (its rescan
    re-discovers native paths), so this is never fed a live-monitored source.
    """
    return _reroot_catalog_url(alias, root_path, primary_path)


def discover_sources(
    source: SourceConfig,
    registry: Optional[AdapterRegistry] = None,
    credentials_config: Optional[Any] = None,
) -> List[SourceConfig]:
    """Expand a source config to actual data sources.

    Directory scanning and file typing are done by the adapters' claim()
    protocol -- the single source of truth for format detection
    (biopb/biopb#277 item B). The only URL-derived typing left here is remote
    scheme routing (grpc -> tensor-server).

    Supports multiple modes:
    - Explicit source: type set -> returned as-is (single source). source_id is
      always present (__post_init__ auto-fills it), so type is the discriminator.
    - Remote URL: requires explicit 'type' in config (cannot auto-discover)
    - tensor-server (grpc://) URL: a caching proxy in front of an upstream biopb
      tensor server -- a bare ``grpc://host:port`` mirrors *every* upstream source
      (one concrete source each, alias-namespaced), and ``grpc://host:port/<id>``
      mirrors a single upstream source.
    - Local file with no type: claim-based detection (error if unclaimed)
    - Local directory with no type: claim-based discovery

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

    # Case 1: explicit type -> return as-is. source_id is always set by
    # __post_init__, so it never discriminates here; type is the real gate.
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

        # No adapter recognized the file. There is no legacy fallback: format
        # detection lives only in the adapters (biopb/biopb#277 item B), so an
        # unclaimed file is a hard error rather than a guessed (often wrong) type.
        raise ValueError(
            f"Could not detect type for file: {local_path}. "
            f"Please specify 'type' explicitly in config."
        )

    # Case 3: Directory with no type - use claim-based discovery.
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
    if claim.source_type == "hdf5" and claim.extra_config.get("needs_dataset"):
        # HDF5 claims have needs_dataset flag in extra_config. This will fail at
        # adapter creation unless dataset is provided; for backward compatibility,
        # pass through any original dataset.
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
        # A local source's `alias` re-roots it (and everything discovered under a
        # configured folder) into its own catalog tree root -- the config-line
        # analogue of a drag-dropped folder becoming its own root. Compute the
        # display source_url now, while both the configured root and each concrete
        # child are in hand. Skipped for remote entries: a tensor-server upstream's
        # alias means the source_id namespace (handled by the proxy adapter's own
        # display authority), not a tree root. A monitored local *directory* never
        # reaches here (it is discovered by the rescan, not expanded), so its alias
        # is correctly never applied -- see _resolve_serve_sources's warning.
        reroot = bool(source.alias) and not source.is_remote
        root_path = source.local_path if reroot else None
        for src in discovered:
            # Track HDF5 sources that need dataset config
            if src.type == "hdf5" and src.dataset is None:
                hdf5_warnings.append(src.url)
            if reroot and root_path is not None:
                src = replace(
                    src,
                    _catalog_url=_alias_catalog_url(
                        source.alias, str(root_path), src.url
                    ),
                )
            all_sources.append(src)

    # source_id collisions involving a tensor-server proxy are a misconfiguration
    # (two upstreams sharing an alias, or a local source named like a proxied id):
    # the catalog is one flat source_id space, so a clash would silently shadow a
    # source. Drop the colliding entry (keeping the first) and warn with the fix --
    # a single bad source must not abort the whole catalog. Non-proxy collisions
    # keep the historical last-wins behavior.
    all_sources = _resolve_tensor_server_id_collisions(all_sources)

    # Print warnings for HDF5 files that need explicit dataset
    if hdf5_warnings:
        print("Warning: HDF5 files require explicit 'dataset' path in config:")
        for h5_url in hdf5_warnings[:5]:
            print(f"  - {h5_url}")
        if len(hdf5_warnings) > 5:
            print(f"  ... and {len(hdf5_warnings) - 5} more")

    return all_sources
