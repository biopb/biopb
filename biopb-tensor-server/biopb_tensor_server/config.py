"""Configuration management for TensorFlight server.

Supports TOML config files with:
- Server settings (host, port)
- Data source definitions (explicit files or directory auto-discovery)

Example config (explicit):
```toml
[server]
host = "0.0.0.0"
port = 8815

[[sources]]
type = "zarr"
path = "/data/images.zarr"
array_id = "my-image"
dim_labels = ["z", "y", "x"]

[[sources]]
type = "hdf5"
path = "/data/sample.h5"
dataset = "/images/channel0"
```

Example config (relaxed auto-discovery):
```toml
[server]
host = "0.0.0.0"
port = 8815

[[sources]]
path = "/data/"  # No type, no array_id - recursive auto-discovery

# HDF5 requires explicit 'type' and 'dataset' - not auto-detected
[[sources]]
type = "hdf5"
path = "/data/sample.h5"
dataset = "/images"
```
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

# Python 3.11+ has tomllib in stdlib
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class TensorSource:
    """Configuration for a single tensor data source.

    Attributes:
        path: Path to file or directory
        type: Storage type - "zarr", "hdf5", or "ome-tiff". Optional - auto-detected if None.
        array_id: Unique identifier (required for files, auto-generated for directories)
        dim_labels: Dimension labels (optional)
        dataset: HDF5 dataset path (required for HDF5 type)
    """
    path: Path
    type: Optional[Literal["zarr", "hdf5", "ome-tiff", "ome-tiff-multifile", "ome-zarr"]] = None
    array_id: Optional[str] = None
    dim_labels: Optional[List[str]] = None
    dataset: Optional[str] = None  # For HDF5

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)


@dataclass
class ServerConfig:
    """Server configuration.

    Attributes:
        host: Server host
        port: Server port
        sources: List of tensor sources
    """
    host: str = "0.0.0.0"
    port: int = 8815
    compute_backend: str = "auto"
    gpu_min_input_mb: float = 4.0
    gpu_min_linear_input_mb: float = 2.0
    gpu_memory_safety_factor: int = 4
    gpu_min_merged_chunks: int = 4
    sources: List[TensorSource] = field(default_factory=list)


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

    # Parse compute settings
    compute_data = data.get("compute", {})
    compute_backend = compute_data.get("backend", "auto")
    gpu_min_input_mb = compute_data.get("gpu_min_input_mb", 4.0)
    gpu_min_linear_input_mb = compute_data.get("gpu_min_linear_input_mb", 2.0)
    gpu_memory_safety_factor = compute_data.get("gpu_memory_safety_factor", 4)
    gpu_min_merged_chunks = compute_data.get("gpu_min_merged_chunks", 4)

    # Parse sources
    sources_data = data.get("sources", [])
    sources: List[TensorSource] = []

    for src_data in sources_data:
        source = TensorSource(
            type=src_data.get("type"),  # Optional - auto-detected if None
            path=Path(src_data["path"]),
            array_id=src_data.get("array_id"),
            dim_labels=src_data.get("dim_labels"),
            dataset=src_data.get("dataset"),
        )
        sources.append(source)

    return ServerConfig(
        host=host,
        port=port,
        compute_backend=compute_backend,
        gpu_min_input_mb=float(gpu_min_input_mb),
        gpu_min_linear_input_mb=float(gpu_min_linear_input_mb),
        gpu_memory_safety_factor=int(gpu_memory_safety_factor),
        gpu_min_merged_chunks=int(gpu_min_merged_chunks),
        sources=sources,
    )


def detect_source_type(path: Path) -> Optional[str]:
    """Detect source type from path characteristics.

    Returns one of: "zarr", "ome-zarr", "ome-tiff", "ome-tiff-multifile"
    or None if type cannot be determined.

    Note: HDF5 is NOT auto-detected because it requires explicit dataset path.
    """
    import json

    if path.is_file():
        name = path.name.lower()

        # OME-TIFF files
        if name.endswith('.ome.tiff') or name.endswith('.ome.tif'):
            return 'ome-tiff'

        # Plain TIFF files - skip for now (could check for OME-XML in header)
        if name.endswith('.tiff') or name.endswith('.tif'):
            # Could implement OME-XML header detection here
            # For now, treat as ome-tiff (tifffile can handle plain TIFF)
            return 'ome-tiff'

        # HDF5 files - return None (requires explicit config with dataset path)
        if name.endswith('.h5') or name.endswith('.hdf5'):
            return None

        return None

    elif path.is_dir():
        name = path.name.lower()

        # Zarr directories (must have .zarray or .zattrs)
        if name.endswith('.zarr'):
            zattrs_path = path / '.zattrs'
            zarray_path = path / '.zarray'

            # Check for OME-Zarr first (more specific)
            if zattrs_path.exists():
                try:
                    with open(zattrs_path) as f:
                        zattrs = json.load(f)
                    if 'multiscales' in zattrs:
                        return 'ome-zarr'
                except (json.JSONDecodeError, KeyError, IOError):
                    pass

            # Plain Zarr (has .zarray or .zattrs without multiscales)
            if zarray_path.exists() or zattrs_path.exists():
                return 'zarr'

            return None

        # Check for multi-file OME-TIFF dataset
        metadata_file = path / '_metadata.txt'
        img_files = list(path.glob('img_*.ome.tiff')) + list(path.glob('img_*.ome.tif'))

        if metadata_file.exists() or len(img_files) > 1:
            return 'ome-tiff-multifile'

        return None

    return None


def scan_directory_for_sources(
    directory: Path,
    dim_labels: Optional[List[str]] = None,
    recursive: bool = True,
    _skipped_extensions: Optional[set] = None,
) -> List[TensorSource]:
    """Recursively scan directory for tensor sources.

    Args:
        directory: Root directory to scan
        dim_labels: Optional dimension labels to apply to all sources
        recursive: If True, scan subdirectories recursively

    Returns:
        List of discovered TensorSource objects with auto-detected types and array_ids
    """
    discovered = []
    skipped_hdf5 = []
    skipped_unknown = []

    for item in directory.iterdir():
        # Skip hidden files/directories
        if item.name.startswith('.'):
            continue

        if item.is_file():
            detected_type = detect_source_type(item)
            if detected_type:
                array_id = item.stem
                discovered.append(TensorSource(
                    type=detected_type,
                    path=item,
                    array_id=array_id,
                    dim_labels=dim_labels,
                ))
            elif item.name.lower().endswith('.h5') or item.name.lower().endswith('.hdf5'):
                skipped_hdf5.append(item)
            else:
                skipped_unknown.append(item)

        elif item.is_dir():
            # Check if this directory itself is a tensor source
            detected_type = detect_source_type(item)
            if detected_type:
                # Use directory name without extension as array_id
                # For .zarr directories, strip the .zarr suffix
                name = item.name
                if name.lower().endswith('.zarr'):
                    array_id = name[:-5] if name.endswith('.zarr') else name[:-4]
                else:
                    array_id = name
                discovered.append(TensorSource(
                    type=detected_type,
                    path=item,
                    array_id=array_id,
                    dim_labels=dim_labels,
                ))
            elif recursive:
                # Not a tensor source itself, but might contain them - recurse
                discovered.extend(scan_directory_for_sources(item, dim_labels, recursive))

    # Print warnings for skipped files (only at top level to avoid spam)
    if skipped_hdf5 and directory == directory:
        print(f"Warning: Skipped HDF5 files (require explicit 'type' and 'dataset' in config):")
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
) -> List[TensorSource]:
    """Discover sources by type in a directory (backward compatibility helper).

    Args:
        path: Directory to scan
        source_type: Type of sources to look for
        dim_labels: Optional dimension labels
        dataset: HDF5 dataset path (for HDF5 type)

    Returns:
        List of discovered TensorSource objects
    """
    import json

    discovered = []

    if source_type == "zarr":
        for zarr_path in sorted(path.glob("*.zarr")):
            if zarr_path.is_dir():
                discovered.append(TensorSource(
                    type="zarr",
                    path=zarr_path,
                    array_id=zarr_path.stem,
                    dim_labels=dim_labels,
                ))

    elif source_type == "ome-zarr":
        for zarr_path in sorted(path.glob("*.zarr")):
            if zarr_path.is_dir():
                zattrs_path = zarr_path / ".zattrs"
                if zattrs_path.exists():
                    try:
                        with open(zattrs_path) as f:
                            zattrs = json.load(f)
                        if 'multiscales' in zattrs:
                            discovered.append(TensorSource(
                                type="ome-zarr",
                                path=zarr_path,
                                array_id=zarr_path.stem,
                                dim_labels=dim_labels,
                            ))
                    except (json.JSONDecodeError, KeyError, IOError):
                        pass

    elif source_type == "hdf5":
        for pattern in ["*.h5", "*.hdf5"]:
            for h5_path in sorted(path.glob(pattern)):
                if h5_path.is_file():
                    discovered.append(TensorSource(
                        type="hdf5",
                        path=h5_path,
                        array_id=h5_path.stem,
                        dim_labels=dim_labels,
                        dataset=dataset,
                    ))

    elif source_type == "ome-tiff":
        metadata_file = path / "_metadata.txt"
        img_files = list(path.glob("img_*.ome.tiff")) + list(path.glob("img_*.ome.tif"))

        if metadata_file.exists() or len(img_files) > 1:
            discovered.append(TensorSource(
                type="ome-tiff-multifile",
                path=path,
                array_id=path.name,
                dim_labels=dim_labels,
            ))
        else:
            for pattern in ["*.ome.tiff", "*.ome.tif", "*.tif", "*.tiff"]:
                for tiff_path in sorted(path.glob(pattern)):
                    if tiff_path.is_file():
                        name = tiff_path.name
                        if name.endswith(".ome.tiff"):
                            stem = name[:-9]
                        elif name.endswith(".ome.tif"):
                            stem = name[:-8]
                        else:
                            stem = tiff_path.stem
                        discovered.append(TensorSource(
                            type="ome-tiff",
                            path=tiff_path,
                            array_id=stem,
                            dim_labels=dim_labels,
                        ))

    return discovered


def discover_sources(source: TensorSource) -> List[TensorSource]:
    """Expand a source config to actual tensor sources.

    Supports multiple modes:
    - Explicit source: type + array_id both set -> single source
    - File with no type: try to detect type from file characteristics
    - Directory with type but no array_id: discover by type (backward compatible)
    - Directory with no type and no array_id: recursive auto-discovery

    Args:
        source: Source configuration

    Returns:
        List of concrete TensorSource objects
    """
    path = source.path

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    # Case 1: Explicit source (type + array_id both set)
    if source.type and source.array_id:
        return [source]

    # Case 2: File with no type - try to detect
    if path.is_file():
        detected_type = source.type or detect_source_type(path)
        if detected_type:
            array_id = source.array_id or path.stem
            return [TensorSource(
                type=detected_type,
                path=path,
                array_id=array_id,
                dim_labels=source.dim_labels,
                dataset=source.dataset,
            )]
        # Could not detect type for file
        raise ValueError(
            f"Could not detect type for file: {path}. "
            f"Please specify 'type' explicitly in config."
        )

    # Case 3: Directory with type but no array_id - discover by type (backward compatible)
    if source.type and not source.array_id:
        return _discover_by_type(path, source.type, source.dim_labels, source.dataset)

    # Case 4: Directory with no type and no array_id
    # First check if the directory itself is a tensor source (e.g., .zarr directory)
    detected_type = detect_source_type(path)
    if detected_type:
        # Use directory name without extension as array_id
        name = path.name
        if name.lower().endswith('.zarr'):
            array_id = name[:-5] if name.endswith('.zarr') else name[:-4]
        else:
            array_id = name
        return [TensorSource(
            type=detected_type,
            path=path,
            array_id=source.array_id or array_id,
            dim_labels=source.dim_labels,
            dataset=source.dataset,
        )]

    # Directory is not itself a tensor source - do recursive scan
    return scan_directory_for_sources(path, source.dim_labels, recursive=True)


def resolve_all_sources(config: ServerConfig) -> List[TensorSource]:
    """Resolve all sources in config, expanding directories.

    Args:
        config: Server configuration

    Returns:
        List of all concrete TensorSource objects
    """
    all_sources = []
    for source in config.sources:
        discovered = discover_sources(source)
        all_sources.extend(discovered)
    return all_sources