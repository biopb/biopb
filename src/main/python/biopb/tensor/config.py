"""Configuration management for TensorFlight server.

Supports TOML config files with:
- Server settings (host, port)
- Data source definitions (explicit files or directory auto-discovery)

Example config:
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
type = "zarr"
path = "/data/datasets/"  # Directory - auto-discover all .zarr
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
        type: Storage type - "zarr", "hdf5", or "ome-tiff"
        path: Path to file or directory
        array_id: Unique identifier (required for files, auto-generated for directories)
        dim_labels: Dimension labels (optional)
        dataset: HDF5 dataset path (required for HDF5 type)
    """
    type: Literal["zarr", "hdf5", "ome-tiff", "ome-tiff-multifile", "ome-zarr"]
    path: Path
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

    # Parse sources
    sources_data = data.get("sources", [])
    sources: List[TensorSource] = []

    for src_data in sources_data:
        source = TensorSource(
            type=src_data["type"],
            path=Path(src_data["path"]),
            array_id=src_data.get("array_id"),
            dim_labels=src_data.get("dim_labels"),
            dataset=src_data.get("dataset"),
        )
        sources.append(source)

    return ServerConfig(host=host, port=port, sources=sources)


def discover_sources(source: TensorSource) -> List[TensorSource]:
    """Expand a source config to actual tensor sources.

    If source.path is a file, returns a single source.
    If source.path is a directory without array_id, discovers all matching files.
    If source.path is a directory with array_id, treats it as a single source.

    Args:
        source: Source configuration

    Returns:
        List of concrete TensorSource objects
    """
    path = source.path

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    # If array_id is provided, treat as explicit source (even if directory like .zarr)
    if source.array_id:
        return [source]

    # Directory auto-discovery (no array_id)
    if not path.is_dir():
        raise ValueError(f"array_id required for file source: {path}")

    discovered = []

    if source.type == "zarr":
        # Find .zarr directories
        for zarr_path in sorted(path.glob("*.zarr")):
            if zarr_path.is_dir():
                array_id = source.array_id or zarr_path.stem
                discovered.append(TensorSource(
                    type="zarr",
                    path=zarr_path,
                    array_id=array_id,
                    dim_labels=source.dim_labels,
                ))

    elif source.type == "ome-zarr":
        # Find .zarr directories with OME metadata
        for zarr_path in sorted(path.glob("*.zarr")):
            if zarr_path.is_dir():
                # Check for OME-Zarr metadata (.zattrs with multiscales)
                zattrs_path = zarr_path / ".zattrs"
                is_ome_zarr = False
                if zattrs_path.exists():
                    try:
                        import json
                        with open(zattrs_path) as f:
                            zattrs = json.load(f)
                        if 'multiscales' in zattrs:
                            is_ome_zarr = True
                    except (json.JSONDecodeError, KeyError):
                        pass

                if is_ome_zarr:
                    array_id = source.array_id or zarr_path.stem
                    discovered.append(TensorSource(
                        type="ome-zarr",
                        path=zarr_path,
                        array_id=array_id,
                        dim_labels=source.dim_labels,
                    ))

    elif source.type == "hdf5":
        # Find .h5 and .hdf5 files
        for pattern in ["*.h5", "*.hdf5"]:
            for h5_path in sorted(path.glob(pattern)):
                if h5_path.is_file():
                    array_id = source.array_id or h5_path.stem
                    discovered.append(TensorSource(
                        type="hdf5",
                        path=h5_path,
                        array_id=array_id,
                        dim_labels=source.dim_labels,
                        dataset=source.dataset,
                    ))

    elif source.type == "ome-tiff":
        # Check for multi-file OME-TIFF dataset (e.g., Micro-Manager format)
        # Look for _metadata.txt or multiple img_*.ome.tiff files
        metadata_file = path / "_metadata.txt"
        img_files = list(path.glob("img_*.ome.tiff")) + list(path.glob("img_*.ome.tif"))

        if metadata_file.exists() or len(img_files) > 1:
            # Multi-file dataset - treat directory as single source
            array_id = source.array_id or path.name
            discovered.append(TensorSource(
                type="ome-tiff-multifile",
                path=path,
                array_id=array_id,
                dim_labels=source.dim_labels,
            ))
        else:
            # Single file OME-TIFF
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

                        array_id = source.array_id or stem
                        discovered.append(TensorSource(
                            type="ome-tiff",
                            path=tiff_path,
                            array_id=array_id,
                            dim_labels=source.dim_labels,
                        ))

    return discovered


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