"""Configuration schema and file I/O for the TensorFlight server.

This module describes and parses config; it never acts on it. Executing a source
entry -- walking a directory through the adapters, expanding a ``grpc://``
upstream -- is :mod:`biopb_tensor_server.sources.resolve`, which sits above the
adapters where the registry is reachable.

Reads JSON config files (``biopb.json``) carrying:
- Server settings (host, port)
- Data source definitions (explicit files or directory auto-discovery)
- Credential profiles for remote storage (S3, GCS, etc.)

JSON is the only supported config format (see biopb/biopb#34).

Example config (explicit):
```json
{
  "sources": [
    {
      "type": "zarr",
      "url": "/data/images.zarr",
      "alias": "my-image",
    },
    { "type": "hdf5", "url": "/data/sample.h5", "dataset": "/images/channel0" }
  ]
}
```
(``alias`` is a friendly display name; ``source_id`` is derived from the URL.)

Example config (relaxed auto-discovery):
```json
{
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
from dataclasses import MISSING as _DC_MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    find_config as find_config,
)

# The constraint primitives and the shared pyramid-knob bounds live in the core
# biopb package so biopb-mcp (which cannot depend on this server package -- not
# on PyPI) validates the same pyramid rows against the same rules, with no drift
# (biopb/biopb#34, #182). `_Range`/`_Enum` stay as local aliases so the
# `_CONSTRAINTS` table and config_schema keep their existing spelling.
from biopb_tensor_server.core.discovery import (
    generate_source_id,
    local_path_is_rooted,
    resolve_local_path,
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
        "file_max_segment_bytes": _Range(min=1),
        "file_max_total_bytes": _Range(min=1),
        # 0 is the off switch (measure, classify nothing); negative would be a
        # threshold every measured array clears, i.e. the off switch's opposite
        # spelled like it.
        "cheap_decode_mbps": _Range(min=0),
    },
    "PyramidConfig": {
        # reduction_method and plane_max_pixels are server-local: on-the-fly
        # reduction is a compute concern, and biopb-mcp has no 2-D plane cap
        # (its fallback plan scales X/Y/Z against one voxel budget). The rows
        # both packages do share come from PYRAMID_CONSTRAINTS.
        "reduction_method": _Enum(_REDUCTION_METHODS, case_insensitive=True),
        # <= 0 never satisfies the Phase 1 stop condition, so the ladder runs to
        # the per-axis floor and the knob silently stops meaning what it says.
        "plane_max_pixels": _Range(min=1),
        **PYRAMID_CONSTRAINTS,
    },
    "PrecacheConfig": {
        "idle_debounce_seconds": _Range(min=0),
        # <= 0 would narrow every selection axis to a single index regardless of
        # size, which is a different policy wearing a budget's name.
        "warm_budget_bytes": _Range(min=1),
        "backlog_high_water": _Range(min=0.0, max=1.0),
        "backlog_idle_recheck_seconds": _Range(min=0),
    },
    "AnnotationsConfig": {
        # 0 or negative fails every write with "Annotation limit reached" -- a
        # cap of nothing is a disabled store wearing a limit's name; `enabled`
        # is the switch for that.
        "max_rois_per_tensor": _Range(min=1),
        # Negative is meaningless and SourceManager clamps it to 0 anyway, so
        # without this the config accepts a value it silently ignores.
        "prune_unseen_days": _Range(min=0),
    },
    "MetadataDbConfig": {
        "max_query_results": _Range(min=1),
        "query_timeout_ms": _Range(min=1),
    },
    "ServerConfig": {
        "log_level": _Enum(
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, case_insensitive=True
        ),
        "rescan_interval": _Range(min=0),
        "stability_window": _Range(min=0),
        "handle_reaper_ttl": _Range(min=0),
        "upload_ttl": _Range(min=0),
        "scratch_ttl": _Range(min=0),
    },
}

# Class name -> the config section it maps to. Read by the schema emitter
# (config_schema.ondisk_location) and by the messages here.
_SECTION_FOR = {
    "CacheConfig": "cache",
    "PyramidConfig": "pyramid",
    "PrecacheConfig": "precache",
    "MetadataDbConfig": "metadata_db",
    "AnnotationsConfig": "annotations",
    "CatalogConfig": "catalog",
    "ServerConfig": "server",
}

# The nested sections the checker walks. ServerConfig itself is the "server"
# section (its scalars are top-level fields, not a nested dataclass), so it is
# passed separately in _sections_of.
_NESTED_SECTIONS = (
    "cache",
    "pyramid",
    "precache",
    "metadata_db",
    "annotations",
    "catalog",
)


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
            "help": "URL or path to the data source. A local path must be "
            "absolute. Local paths are stable; remote URLs (s3://, http(s)://, "
            "grpc://) are experimental."
        }
    )
    type: Optional[
        Literal[
            "zarr",
            "hdf5",
            "ome-tiff",
            "ome-tiff-multifile",
            "tiff",
            "lsm",
            "czi",
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
            "help": "(experimental) Name this source appears under: the catalog "
            "tree root for a local source (display-only), or the id namespace for "
            "a tensor-server upstream -- there it is part of source_id, so "
            "renaming it re-keys the cache and detaches ROI annotations. Must be "
            "slash-free."
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

        # A local url must name a path from a filesystem root. `Path.resolve()`
        # -- which `local_path` and the `source_id` hash both run -- completes a
        # rootless path from the *process* cwd, and a server is started by the
        # control plane, by systemd or by a container entrypoint, each leaving a
        # different one: the same config would name different directories, under
        # different source_ids, per launch (biopb/biopb#947). Config parsing
        # drops such an entry before it reaches here, so this is the backstop for
        # a programmatic construction.
        if not _is_remote_url(self.url) and not local_path_is_rooted(self.url):
            raise ValueError(
                f"Source 'url' must be a rooted path or a remote URL, got: "
                f"{self.url!r}. A path with no root is completed from whatever "
                "directory the server happened to be started in."
            )

        # Compute is_remote from URL
        object.__setattr__(self, "_is_remote", _is_remote_url(self.url))

        # Mint the id from the URL unless the caller supplied one. This is durable
        # identity -- the metadata DB's ROI rows and the segment-cache keys hang off
        # it -- so whatever enters the hash becomes something a user cannot change
        # without detaching their data. Url-derivation's known cost is that `mv`
        # re-keys a local source. Supplying an id explicitly is how the
        # tensor-server proxy opts out: `sources.resolve._namespaced_source_id`
        # builds one from (alias, upstream_source_id) with no endpoint in it, so a
        # moved upstream keeps its cache and its annotations. Config never reaches
        # this branch -- `sources.source_id` is ignored with a warning
        # (biopb/biopb#308) -- so an explicit id is always internal.
        if self.source_id is None:
            detected_type = self.type or detect_source_type(self.url) or "data"
            object.__setattr__(
                self, "source_id", generate_source_id(self.url, detected_type)
            )

    @property
    def local_path(self) -> Optional[Path]:
        """Return Path if url is a local file path, else None.

        For remote URLs (s3://, http://, etc.), returns None. For a local url --
        a plain path or a ``file://`` one -- the canonical Path, through the same
        :func:`resolve_local_path` the ``source_id`` hash uses, so a source's
        identity and the location it reads can never disagree. ``__post_init__``
        refuses a rootless url, so the resolution here only folds ``file://``,
        symlinks and ``..``, never the cwd.
        """
        if _is_remote_url(self.url):
            return None
        return Path(resolve_local_path(self.url))


@dataclass
class CacheConfig:
    """Cache configuration for computed virtual chunks.

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema). Note the on-disk key names differ for the size fields
    (``file_max_segment_bytes`` -> ``file_max_segment_mb``, ``file_max_total_bytes``
    -> ``file_max_total_gb``); the help is phrased for the on-disk form the
    editor shows.
    """

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
    source_scaled_reads: bool = field(
        default=True,
        metadata={
            "help": "Serve a scaled (downsampled) chunk from the "
            "full-resolution chunks already in the cache instead of reading the "
            "source again, where all of them are present. Faster, and it leaves "
            "those chunks' pages resident for the full-resolution read a coarse "
            "one usually precedes. Set false to always read the source."
        },
    )
    cheap_decode_mbps: float = field(
        default=0.0,
        metadata={
            "help": "Evict a full-resolution chunk early once its tensor is "
            "measured to decode at least this fast (MB/s) -- rebuilding it is "
            "cheaper than the cache space it holds. 0 (the default) measures "
            "but classifies nothing. Read the measurements with "
            "`biopb tensor decode-rates` and pick a threshold from them: what "
            "counts as fast enough depends on the machine's disk and the "
            "formats on it, so there is no portable default. The measurements "
            "live in the catalog database, so clearing the cache does not "
            "reset them -- and they are session-only when catalog.persist is "
            "off."
        },
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
        default="nearest",
        metadata={
            "help": "Downsampling method for computed levels ('nearest' = strided "
            "pick, 'area' = averaging). Native on-disk levels are served "
            "precomputed regardless."
        },
    )
    threshold: int = field(
        default=4096,
        metadata={"help": "Maximum X/Y extent (pixels) of the coarsest level."},
    )
    downscale_factor: int = field(
        default=2,
        metadata={"help": "Per-level linear downsampling step, per spatial axis."},
    )
    pixel_budget_cubic_root: int = field(
        default=448,
        metadata={
            "help": "Cube root of the coarsest level's voxel budget "
            "(Lx*Ly*Lz <= this**3); bounds a whole-volume 3-D read."
        },
    )
    plane_max_pixels: int = field(
        default=4_000_000,
        metadata={
            "help": "Max X*Y pixels of the coarsest 2-D level. Also the gate on "
            "whether a tensor gets 2-D levels at all: below it the plane is "
            "already cheap to read whole. Values below "
            "min(pixel_budget_cubic_root, threshold)**2 are unreachable -- the "
            "rungs stop at that per-axis floor first."
        },
    )

    def level_kwargs(self) -> Dict[str, int]:
        """The knobs that shape the ladder, as ``chunk.py`` takes them.

        Every caller of ``_pyramid_levels`` and its wrappers needs exactly these
        four, and the point of this class is that they cannot drift between the
        levels the server advertises and the ones the precache warms -- so they
        are unpacked once here rather than at each call.
        """
        return {
            "threshold": self.threshold,
            "downscale_factor": self.downscale_factor,
            "pixel_budget_cubic_root": self.pixel_budget_cubic_root,
            "plane_max_pixels": self.plane_max_pixels,
        }


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

    # On by default again (biopb/biopb#826 turned it off). What made warming
    # unaffordable was an unbounded warm set: every level of the ladder, over the
    # whole T/C/Z cross-product, so nothing it warmed survived eviction on the
    # shipped 4 GiB cache. The plan is now at most two levels
    # (compute_warm_targets) and each is capped at warm_budget_bytes over a
    # window from index 0 (compute_warm_selection), which is what makes residency
    # reachable rather than aspirational.
    enabled: bool = field(
        default=True,
        metadata={
            "help": "Run the background pre-cache worker to warm new sources "
            "(no-op on the memory backend). Warms at most two levels per tensor, "
            "each bounded by warm_budget_bytes."
        },
    )
    idle_debounce_seconds: float = field(
        default=2.0,
        metadata={
            "help": "Quiet period after live traffic before the worker resumes "
            "(seconds)."
        },
    )
    # 256 MiB, per warm level. Not sized to make a catalog fit -- at a few hundred
    # tensors nothing in the hundreds of MiB does, and the backlog high-water gate
    # is what stops a full cache. What it bounds is one tensor eating the cache,
    # and above the first plane it buys *scrub headroom*: at this value a
    # 200-plane confocal keeps 128 of its Z planes rather than 32, so paging
    # through the stack stays warm instead of only the opening slab.
    warm_budget_bytes: int = field(
        default=256 * 1024 * 1024,
        metadata={
            "help": "Per warm level, the cap on the selection cross-product "
            "(T/Z/C). Over it, those axes are narrowed to a window from index 0, "
            "where a viewer opens. A budget on what is REQUESTED, not on disk: "
            "the cache stores whole chunks, so a window that ends mid-chunk "
            "still writes that chunk entire and the footprint rounds up. Does "
            "not bound the plane or the 3-D volume themselves -- "
            "pyramid.plane_max_pixels and pyramid.pixel_budget_cubic_root do "
            "that."
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
    query_timeout_ms: int = field(
        default=30000,
        metadata={"help": "Catalog SQL query timeout (milliseconds)."},
    )


@dataclass
class CatalogConfig:
    """The DuckDB catalog file: whether there is one, and where.

    Three tables share it and only one is annotations, which is why these keys
    are not under ``annotations`` any more (biopb/biopb#1002):

    - ``sources`` -- scan output, dropped and recreated on every open.
    - ``rois`` -- drawn annotations. Nothing can
      reproduce these, which is what makes the file worth having.
    - ``decode_rates`` -- the cache's measured per-tensor decode throughput.
      Re-measurable by reading, but a run's worth of it at a time.

    State tree, not cache: ``cache_dir()`` is documented as safe for a janitor
    to empty and this file is not, which is the whole reason the measurements
    live here rather than beside the segments they describe.

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema).
    """

    persist: bool = field(
        default=True,
        metadata={
            "help": "Back the catalog with a file so it outlives the server. "
            "Off keeps the whole catalog in memory: drawn ROIs are lost when "
            "the server stops, and so are the cache's decode measurements."
        },
    )
    store_path: str = field(
        default="",
        metadata={
            "help": "Where the on-disk catalog lives. Empty derives it from the "
            "config file's path, which is what keeps two servers on two configs "
            "off each other's file."
        },
    )


@dataclass
class AnnotationsConfig:
    """User-drawn ROI annotations.

    Annotations live in the DuckDB catalog next to ``sources`` and are served
    on the ``roi`` flight (DoGet / DoPut, authorized per source). They are NOT
    tied to ``writable``: an annotation writes no pixels, so the token is its
    boundary and ``enabled`` is the switch for a deployment that wants a strictly
    read-only catalog.

    Whether that catalog reaches a file, and which one, is :class:`CatalogConfig`
    -- three tables share it and only this one is annotations
    (biopb/biopb#1002).

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema).
    """

    enabled: bool = field(
        default=True,
        metadata={
            "help": "Serve the roi flight (annotation reads and writes). Off "
            "makes the catalog strictly read-only -- the "
            "token says who may read, this says whether anyone may write. It "
            "does not stop the catalog being persisted: `persist` decides that."
        },
    )
    max_rois_per_tensor: int = field(
        default=5000,
        metadata={
            "help": "Cap on stored annotations per tensor. Deliberately "
            "human-scale: this is an annotation store, not an object store -- a "
            "segmentation belongs in a label tensor."
        },
    )
    prune_unseen_days: int = field(
        default=0,
        metadata={
            "help": "Delete annotations whose source has not been seen in this "
            "many days. 0 (the default) never deletes: these are hand-drawn, and "
            "a source can be absent because a drive is unmounted or a proxy "
            "upstream is down rather than because the image is gone. Even when "
            "set, deleting only arms once the server has been up longer than "
            "this -- before that it has not watched long enough to conclude "
            "anything. `unseen_rois` reports orphans either way."
        },
    )


@dataclass
class ServerConfig:
    """Server configuration.

    Per-field help lives in each field's ``metadata["help"]`` -- the single
    source the config JSON Schema reads (see ``config_schema.py``). The nested
    section objects (``cache``/``pyramid``/``precache``/``credentials``/
    ``metadata_db``) and ``sources`` document themselves.

    **The network bind is deliberately not here** (biopb/biopb#604). ``host``,
    ``port``, ``tls``, ``tls_cert`` and ``tls_key`` used to live in this section
    and now come only from the CLI (``serve``/``launch`` flags, which the control
    passes explicitly when it spawns the plane). This file answers *what to
    serve* -- sources, cache, pyramid, credentials -- while *where and how to
    expose it* is a deployment decision owned by whoever starts the process. The
    sidecar's own bind (``--web-host``/``--web-port``) was always CLI-only; this
    just makes the flight plane consistent with it.

    Two things fall out. The control cannot hold a stale view of a bind it
    dictated, so a port edited underneath it can no longer wedge its liveness
    probe. And "public + tokenless" stops being a config state that has to be
    *validated* against and becomes unrepresentable: the flag that binds publicly
    (``--remote``) is the same one that requires a token.
    """

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
            "help": "Ceiling, in seconds, on how long an idle persistent file "
            "handle (OME-TIFF store, NDTiff acquisition, CZI or ND2 reader) is "
            "kept warm before it is closed; the next read reopens it (0 disables "
            "reaping). A ceiling, not an assignment: each format keeps its own "
            "shorter value where reopening it is cheap, so raising this never "
            "lengthens a pin. Adapters that reopen per read (hdf5, mrc, ...) are "
            "unaffected."
        },
    )
    upload_ttl: float = field(
        default=3600.0,
        metadata={
            "help": "Seconds an upload may sit without a write before it is "
            "discarded as abandoned, and a discarded upload stays registered "
            "(so a straggler still learns why its writes fail) before its name "
            "is freed. Finished uploads are never reclaimed. 0 disables the sweep."
        },
    )
    scratch_ttl: float = field(
        default=86400.0,
        metadata={
            "help": "Ceiling, in seconds, on how long a tensor uploaded to "
            "the scratch source is kept, after which it is discarded. An "
            "upload asking for less gets what it asked for; one asking for "
            "more, or for nothing, gets this. 0 keeps them until someone "
            "discards them."
        },
    )
    stability_window: float = field(
        default=30.0,
        metadata={
            "help": "Minimum quiet period before a path is eligible for discovery "
            "or removal (seconds). Raise it above the interval at which a slow "
            "acquisition touches its files, or a dataset can be claimed between "
            "writes; it only ever delays, never drops."
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
            "uploads). Keep it outside every source directory: an uploaded "
            "store is registered by the upload path, and discovery walking it "
            "too would catalog it a second time."
        },
    )
    cache: CacheConfig = field(default_factory=CacheConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    precache: PrecacheConfig = field(default_factory=PrecacheConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    metadata_db: MetadataDbConfig = field(default_factory=MetadataDbConfig)
    annotations: AnnotationsConfig = field(default_factory=AnnotationsConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    sources: List[SourceConfig] = field(default_factory=list)


def load_config(path: Path) -> ServerConfig:
    """Load configuration from a JSON file.

    JSON is the only format read (biopb/biopb#34). The file is parsed to a
    plain dict and handed to the format-agnostic :func:`parse_config`.

    Args:
        path: Path to a JSON config file (``biopb.json``)

    Returns:
        ServerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If the file is not valid JSON, or if it carries an
            out-of-range / bad-enum value
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
    - A sibling ``biopb.schema.json`` (the output of ``build_config_schema``) is
      written next to the config and a *relative* ``"$schema":
      "./biopb.schema.json"`` pointer is embedded, so editors validate the
      config offline with no hosted schema URL.
    - The write is atomic; on failure the file on disk is untouched and the error
      propagates so the caller can surface it.
    """
    if isinstance(path, str):
        path = Path(path)

    schema_path = path.with_name(SCHEMA_SIDECAR_NAME)

    # build_config_schema lives in config_schema, which imports the dataclasses
    # here -- import lazily to avoid the cycle (see _known_config_keys).
    from biopb_tensor_server.core.config_schema import build_config_schema

    # Embed a relative $schema pointer (offline editor validation, no hosted URL).
    payload = dict(data)
    payload["$schema"] = f"./{SCHEMA_SIDECAR_NAME}"

    atomic_write_json(schema_path, build_config_schema(), raise_on_error=True)
    atomic_write_json(path, payload, raise_on_error=True)

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
    """Read a config file into a plain dict.

    Extension-blind: the file is read as JSON whatever it is called, so an
    unconventionally-named config still loads.
    """
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except ValueError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}") from e


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
    the classic trap is ``[cache] file_max_segment_bytes`` (the dataclass field
    name) where the parser reads ``file_max_segment_mb``, so an oversized-segment
    tweak silently stays at the 64MB default. Value validation (range/enum)
    lives in the dataclasses; this only flags *keys the parser never reads*. The
    known-key set is the config JSON Schema's property lists (see
    :func:`_known_config_keys`). Warn-only.
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
        known = section_keys.get(section, set())
        if section == "server":
            # The retired bind keys get their own, far more actionable message
            # from _warn_retired_bind_keys; suppress the generic "unknown key"
            # line so it does not bury that one.
            known = known | set(_RETIRED_BIND_KEYS)
        _warn_extra_keys(value, known, section)
        if section == "credentials":
            for prof in value.get("profiles", []) or []:
                if isinstance(prof, dict):
                    _warn_extra_keys(prof, profile_keys, "credentials.profiles")


# Bind/TLS keys that used to live in [server] and are now CLI-only
# (biopb/biopb#604). Mapped to the flag that replaced each one.
_RETIRED_BIND_KEYS = {
    "host": "--host",
    "port": "--port",
    "tls": "--tls",
    "tls_cert": "--tls-cert",
    "tls_key": "--tls-key",
}


def _warn_retired_bind_keys(server_data: Dict[str, Any]) -> None:
    """Warn — loudly — for a ``[server]`` bind key that is no longer read.

    Ignoring these silently would be the worst possible break: a config saying
    ``"host": "0.0.0.0"`` was someone's *remote* deployment, and quietly moving it
    to the loopback default would take their server off the network with no
    signal. So the message names the exact flag that replaces the key, and it is
    a warning rather than a hard error because a config is often shared across
    hosts that have already migrated their launch command.
    """
    if not isinstance(server_data, dict):
        return
    for key in _RETIRED_BIND_KEYS:
        if server_data.get(key) is None:
            continue
        logger.warning(
            "Config key `server.%s` is no longer read and is IGNORED: the "
            "network bind moved to the command line (biopb/biopb#604). Pass "
            "`%s` to `biopb-tensor-server serve/launch`, or -- under the control "
            "plane -- use `biopb control start [--remote] [--tls]`, which passes "
            "it down. Remove the key to silence this.",
            key,
            _RETIRED_BIND_KEYS[key],
        )


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


def _normalized_source_url(url: str) -> Optional[str]:
    """The url to record for a source, or None if it names no fixed location.

    A remote url passes through untouched (``Path`` would mangle the scheme's
    ``//`` and prepend the cwd) and ``~`` expands -- it used to become
    ``$PWD/~/...``. Anything else is returned exactly as written: pushing a
    rooted path through ``Path`` would flip its separators on Windows and take
    the ``source_id`` with them.

    A url with no root names nothing the server can honor -- there is no anchor
    it could use that the person writing the config would recognize
    (biopb/biopb#947) -- so it comes back None for the caller to report.
    """
    if _is_remote_url(url):
        return url
    if url.startswith("~"):
        try:
            expanded = str(Path(url).expanduser())
        except RuntimeError:
            # No home directory to expand against: POSIX with no HOME and no
            # passwd entry, or Windows with neither USERPROFILE nor HOMEPATH (a
            # service account). `~` then names nothing, which is this function's
            # None -- the entry is dropped like any other unusable url rather
            # than taking the whole config load down with a RuntimeError.
            return None
    else:
        expanded = url
    return expanded if local_path_is_rooted(expanded) else None


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
    dataclasses cannot express: on-disk key aliases (``cache.file_max_segment_mb``
    -> ``file_max_segment_bytes``), unit scaling (``*_mb``/``*_gb`` -> ``*_bytes``),
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
    _warn_retired_bind_keys(server_data)
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
    _carry(server_kwargs, "upload_ttl", server_data, cast=float)
    _carry(server_kwargs, "scratch_ttl", server_data, cast=float)
    _carry(server_kwargs, "stability_window", server_data, cast=float)
    _carry(server_kwargs, "aggressive_dir_pruning", server_data, cast=bool)
    _carry(server_kwargs, "claim_generic_images", server_data, cast=bool)
    _carry(server_kwargs, "writable", server_data)
    write_dir_str = server_data.get("write_dir")
    if write_dir_str:
        server_kwargs["write_dir"] = Path(write_dir_str)

    # Parse cache settings. The wire form of two fields diverges from the
    # dataclass (MB/GB->bytes scaling); everything else is a direct carry.
    # Mapping mirrors config_schema._ONDISK_OVERRIDES.
    cache_data = data.get("cache", {})
    cache_kwargs: Dict[str, Any] = {}
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
        ("plane_max_pixels", int),
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
    _carry(precache_kwargs, "warm_budget_bytes", precache_data, cast=int)
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
            # Per-upstream TLS trust (biopb/biopb#604 item 4). Both were added to
            # CredentialProfile and read by resolve_upstream_credentials, but
            # never parsed here -- so a configured anchor was dropped on load and
            # every `grpcs://` upstream silently fell back to TOFU pinning, which
            # is precisely the silent degradation that design set out to prevent.
            # They are *known* keys (config_schema derives them from the
            # dataclass), so nothing warned either.
            tls_ca_file=profile_data.get("tls_ca_file", None),
            tls_fingerprint=profile_data.get("tls_fingerprint", None),
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
    _carry(metadata_db_kwargs, "query_timeout_ms", metadata_db_data)
    metadata_db_config = MetadataDbConfig(**metadata_db_kwargs)

    # Parse annotations settings
    annotations_data = data.get("annotations", {})
    annotations_kwargs: Dict[str, Any] = {}
    _carry(annotations_kwargs, "enabled", annotations_data)
    _carry(annotations_kwargs, "max_rois_per_tensor", annotations_data)
    _carry(annotations_kwargs, "prune_unseen_days", annotations_data, cast=int)
    annotations_config = AnnotationsConfig(**annotations_kwargs)

    # Parse catalog settings. These lived under `annotations` for the five days
    # between biopb/biopb#946 and biopb/biopb#1002 -- no alias, because nothing
    # was deployed on them and a permanent second spelling costs more than the
    # window it covers. An old config's keys land on the unknown-key warning,
    # which names them.
    catalog_data = data.get("catalog", {})
    catalog_kwargs: Dict[str, Any] = {}
    _carry(catalog_kwargs, "persist", catalog_data)
    _carry(catalog_kwargs, "store_path", catalog_data)
    catalog_config = CatalogConfig(**catalog_kwargs)

    # Parse sources. `url` accepts the legacy `path` alias; every other field is
    # carried only when present so SourceConfig owns the defaults.
    sources_data = data.get("sources", [])
    sources: List[SourceConfig] = []

    for src_data in sources_data:
        # Support both 'url' (new) and 'path' (legacy) for backward compatibility
        url = src_data.get("url") or src_data.get("path")
        if url is None:
            raise ValueError("Source config requires 'url' field")

        normalized_url = _normalized_source_url(url)
        if normalized_url is None:
            # A relative url has no anchor the server can trust: `Path.resolve`
            # would take the process cwd, which the control plane, systemd and a
            # container entrypoint each leave set differently, so the entry names
            # different data per launch (biopb/biopb#947). Drop it rather than
            # serve a guess -- at the volume biopb/biopb#608 set for a config
            # that will never come right on its own, since one bad line must not
            # keep the other sources off the air. `validate` is the surface that
            # fails hard on it.
            logger.error(
                "NOT SERVING %s: a source url must be a rooted path (or a remote "
                "URL). A path with no root is completed from whatever directory "
                "the server was started in, so it names different data per "
                "launch. This will not resolve until the config is corrected.",
                url,
            )
            continue

        src_kwargs: Dict[str, Any] = {"url": normalized_url}
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
        annotations=annotations_config,
        catalog=catalog_config,
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
    # ones (CacheConfig.file_max_segment_bytes lives at [cache]
    # file_max_segment_mb), so the section/key is remapped before it leaves --
    # in the message too, whose `field=...` lead-in would otherwise name the
    # internal field the on-disk path doesn't (e.g. path [cache]
    # file_max_segment_mb with "file_max_segment_bytes=...").
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

    # `_build_config` drops a source whose url is not absolute and keeps serving
    # the rest -- right for a supervised start, wrong here, where a human is
    # asking whether the file is good. Report the skip, per field, so `validate`
    # fails and the admin form can mark it (biopb/biopb#947).
    for src_data in data.get("sources") or []:
        if not isinstance(src_data, dict):
            continue
        url = src_data.get("url") or src_data.get("path")
        if isinstance(url, str) and url and _normalized_source_url(url) is None:
            problems.append(
                {
                    "path": ["sources", "url"],
                    "message": (
                        f"url={url!r}: a source url must be a rooted path or a "
                        "remote URL. A path with no root is completed from "
                        "whatever directory the server was started in, so it "
                        "names different data per launch."
                    ),
                }
            )
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
