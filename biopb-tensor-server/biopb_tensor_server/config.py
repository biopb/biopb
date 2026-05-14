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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
    is_remote_url as is_remote_url_v2,
)

# Alias for backward compatibility with internal usage
_is_remote_url = is_remote_url

# Python 3.11+ has tomllib in stdlib
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class SourceConfig:
    """Configuration for a single data source.

    A source may contain multiple tensors (multifield support) - the adapter
    handles tensor enumeration via list_tensor_descriptors() at runtime.

    Attributes:
        url: URL or path to the data source (supports local paths and remote URLs)
        type: Storage type - "zarr", "hdf5", "ome-tiff", "ome-tiff-multifile", "ome-zarr", "ome-zarr-hcs", or "aics".
              Optional - auto-detected if None (only for local files).
        source_id: Unique identifier for the data source (auto-generated from URL if None)
        dim_labels: Dimension labels (optional, applies to all tensors in source)
        dataset: HDF5 dataset path (required for HDF5 type)
        monitor: Enable live filesystem monitoring for this source (local directories only)
                 When True, the server will watch for file add/delete events and update
                 the catalog automatically.
        credentials_profile: Name of credential profile for remote URLs (overrides global default)
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
        ]
    ] = None
    source_id: Optional[str] = None
    dim_labels: Optional[List[str]] = None
    dataset: Optional[str] = None  # For HDF5
    monitor: bool = False  # Enable live filesystem monitoring
    credentials_profile: Optional[str] = None  # Override global credential profile
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

    @property
    def is_remote(self) -> bool:
        """Check if this source is a remote URL."""
        return _is_remote_url(self.url)


@dataclass
class CacheConfig:
    """Cache configuration for computed virtual chunks.

    Attributes:
        backend: Cache backend type - "memory" or "file"
        memory_max_entries: Maximum number of cached entries (memory backend)
        memory_max_bytes: Maximum total bytes to cache (memory backend, default 512 MB)
        file_cache_dir: Directory for cache files (file backend, default /tmp/biopb-cache)
        file_max_segment_bytes: Maximum bytes per segment file (file backend, default 64 MB)
        file_max_total_bytes: Maximum total bytes across all segments (file backend, default 4 GB)
    """

    backend: str = "memory"
    memory_max_entries: int = 1024
    memory_max_bytes: int = 512 * 1024 * 1024  # 512 MB
    file_cache_dir: Path = Path("/tmp/biopb-cache")
    file_max_segment_bytes: int = 64 * 1024 * 1024  # 64 MB per segment
    file_max_total_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB total

    def __post_init__(self):
        if isinstance(self.file_cache_dir, str):
            self.file_cache_dir = Path(self.file_cache_dir)


@dataclass
class MetadataDbConfig:
    """Configuration for DuckDB metadata database and source catalog safety limits.

    Enables efficient SQL filtering for large source catalogs (>100k sources).
    Replaces O(n) in-memory scans with indexed DuckDB queries.

    Attributes:
        enabled: Enable metadata database for source filtering queries
        max_query_results: Safety cap on SQL query returned rows (truncation signaled via schema metadata)
        max_list_flights_results: Safety cap on list_flights() returned sources (truncation signaled via schema metadata)
        query_timeout_ms: Query execution timeout in milliseconds
    """

    enabled: bool = True
    max_query_results: int = 100000
    max_list_flights_results: int = 100000
    query_timeout_ms: int = 30000


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
        stability_window: Minimum quiet period before a path/subtree is eligible
            for discovery or removal checks.
        probe_open_files: When True, verify candidate files can be opened for append
            before considering them stable.
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
    stability_window: float = 30.0
    probe_open_files: bool = True
    writable: bool = False  # Enable write mode
    write_dir: Optional[Path] = None  # Directory for zarr-backed sources
    cache: CacheConfig = field(default_factory=CacheConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    metadata_db: MetadataDbConfig = field(default_factory=MetadataDbConfig)
    sources: List[SourceConfig] = field(default_factory=list)


def load_config(path: Path) -> ServerConfig:
    """Load configuration from a TOML file.

    Args:
        path: Path to TOML config file

    Returns:
        ServerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return parse_config(data)


def parse_config(data: Dict[str, Any]) -> ServerConfig:
    """Parse configuration from a dictionary.

    Args:
        data: Config dictionary (from TOML)

    Returns:
        ServerConfig object
    """
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

    stability_window = server_data.get("stability_window", 30.0)
    probe_open_files = server_data.get("probe_open_files", True)
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
    cache_backend = cache_data.get("backend", "memory")

    # Parse memory backend settings
    memory_max_entries = cache_data.get("max_entries", 1024)
    memory_max_bytes = cache_data.get("max_bytes", 512 * 1024 * 1024)

    # Parse file backend settings (convert MB/GB to bytes if specified)
    file_cache_dir = Path(cache_data.get("file_cache_dir", "/tmp/biopb-cache"))
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
            credentials_profile=src_data.get("credentials_profile", None),  # NEW
        )
        sources.append(source)

    return ServerConfig(
        host=host,
        port=port,
        log_level=log_level,
        log_scope_to_biopb=log_scope_to_biopb,
        monitor_mode=monitor_mode,
        rescan_interval=float(rescan_interval),
        stability_window=float(stability_window),
        probe_open_files=bool(probe_open_files),
        compute_backend=compute_backend,
        gpu_min_input_mb=float(gpu_min_input_mb),
        gpu_min_linear_input_mb=float(gpu_min_linear_input_mb),
        gpu_memory_safety_factor=int(gpu_memory_safety_factor),
        gpu_min_merged_chunks=int(gpu_min_merged_chunks),
        writable=writable,
        write_dir=write_dir,
        cache=cache_config,
        credentials=credentials_config,
        metadata_db=metadata_db_config,
        sources=sources,
    )


def detect_source_type(url: str) -> Optional[str]:
    """Detect source type from URL characteristics.

    Returns one of: "zarr", "ome-zarr", "ome-zarr-hcs", "ome-tiff", "ome-tiff-multifile", "aics"
    or None if type cannot be determined.

    Note: HDF5 is NOT auto-detected because it requires explicit dataset path.
    Note: Remote URLs cannot be auto-detected - requires explicit 'type' in config.
    """
    import json

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
                except (json.JSONDecodeError, KeyError, IOError):
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
                    except (json.JSONDecodeError, KeyError, IOError):
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


def discover_sources(
    source: SourceConfig, registry: Optional[AdapterRegistry] = None
) -> List[SourceConfig]:
    """Expand a source config to actual data sources.

    Uses claim-based discovery for directory scanning, while maintaining
    backward compatibility for explicit configs.

    Supports multiple modes:
    - Explicit source: type + source_id both set -> single source
    - Remote URL: requires explicit 'type' in config (cannot auto-discover)
    - Local file with no type: try to detect type from file characteristics
    - Local directory with type but no source_id: discover by type (backward compatible)
    - Local directory with no type and no source_id: claim-based discovery

    Args:
        source: Source configuration
        registry: Optional adapter registry (uses default if None)

    Returns:
        List of concrete SourceConfig objects (one per data source, NOT expanded to tensors)

    Raises:
        ValueError: If remote URL lacks explicit 'type'
    """
    if registry is None:
        registry = get_default_registry()

    # Case 0: Remote URLs require explicit type
    if source.is_remote:
        if source.type is None:
            raise ValueError(
                f"Remote URL requires explicit 'type' in config: {source.url}"
            )
        # For remote URLs, return the source as-is (no directory discovery)
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
        # Try claim-based detection first
        ctx = ClaimContext(local_path)
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
    ctx = ClaimContext(local_path)
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

    # Directory is not itself a data source - do recursive claim-based scan
    state = claim_based_discover(local_path, registry, dim_labels=source.dim_labels)
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
    )


def resolve_all_sources(
    config: ServerConfig, registry: Optional[AdapterRegistry] = None
) -> List[SourceConfig]:
    """Resolve all sources in config, expanding directories.

    Uses claim-based discovery for automatic source detection.

    Args:
        config: Server configuration
        registry: Optional adapter registry (uses default if None)

    Returns:
        List of all concrete SourceConfig objects (one per data source)
    """
    if registry is None:
        registry = get_default_registry()

    all_sources = []
    hdf5_warnings = []

    for source in config.sources:
        discovered = discover_sources(source, registry)
        for src in discovered:
            # Track HDF5 sources that need dataset config
            if src.type == "hdf5" and src.dataset is None:
                hdf5_warnings.append(src.url)
            all_sources.append(src)

    # Print warnings for HDF5 files that need explicit dataset
    if hdf5_warnings:
        print("Warning: HDF5 files require explicit 'dataset' path in config:")
        for h5_url in hdf5_warnings[:5]:
            print(f"  - {h5_url}")
        if len(hdf5_warnings) > 5:
            print(f"  ... and {len(hdf5_warnings) - 5} more")

    return all_sources
