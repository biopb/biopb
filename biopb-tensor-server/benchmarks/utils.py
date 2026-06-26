"""Shared utilities for benchmark suite.

Provides synthetic dataset generation, cache stats helpers,
and test data configuration from environment variables.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# Test data configuration (environment variables)
# =============================================================================

S3_TEST_DATA_URL = os.environ.get(
    "S3_TEST_DATA_URL",
    "s3://idr-public/ngff/6001240.zarr"  # IDR public OME-Zarr sample
)

NFS_TEST_DATA_DIR = os.environ.get("NFS_TEST_DATA_DIR", "/data/microscopy")


# =============================================================================
# Cache utilities
# =============================================================================


def clear_os_page_cache() -> None:
    """Clear OS page cache (requires root in container).

    This simulates cold cache conditions by dropping all cached pages.
    Requires privileged container or root access.
    """
    os.system("sync")
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except PermissionError:
        # Not running as root - skip cache clear
        pass


def get_cache_stats() -> Dict:
    """Get cache statistics from CacheManager singleton.

    Returns:
        Dictionary with cache hit/miss/eviction statistics
    """
    from biopb_tensor_server.cache import CacheManager

    manager = CacheManager.get_instance()
    if manager is None:
        return {}

    stats = manager.stats()
    return {
        "entries": stats.total_entries,
        "bytes": stats.total_bytes,
        "hits": stats.hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
        "hit_rate": stats.hits / (stats.hits + stats.misses) if (stats.hits + stats.misses) > 0 else 0.0,
    }


def reset_cache() -> None:
    """Reset the CacheManager singleton."""
    from biopb_tensor_server.cache import CacheManager

    CacheManager.reset()


def clear_cache_entries() -> None:
    """Clear cache entries but keep CacheManager initialized.

    For warm cache tests that need a fresh cache state without
    destroying the singleton.
    """
    from biopb_tensor_server.cache import CacheManager

    manager = CacheManager.get_instance()
    if manager is not None:
        manager.clear()


# =============================================================================
# Synthetic dataset generation
# =============================================================================


def generate_synthetic_hcs_plate(
    path: str,
    wells: int = 96,
    fields: int = 4,
    shape: Tuple[int, int] = (512, 512),
    chunks: Tuple[int, int] = (256, 256),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, List[str], Dict]:
    """Generate synthetic HCS plate-like zarr for cache testing.

    Creates an OME-Zarr structure simulating a high-content screening plate
    with multiple wells and fields. Used for testing cache hit rates under
    sequential scan + revisit patterns.

    Args:
        path: Directory to create the zarr
        wells: Number of wells (e.g., 96 for 96-well plate)
        fields: Number of fields per well
        shape: Image shape per field (height, width)
        chunks: Chunk size for zarr array
        dtype: Data type

    Returns:
        Tuple of (zarr_path, well_names, metadata)
    """
    import zarr

    zarr_path = Path(path) / "plate.ome.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zarr_path), mode="w")
    well_names = []

    # Create plate metadata
    plate_meta = {
        "plate": {
            "name": "synthetic-plate",
            "version": "0.4",
            "wells": [],
        }
    }

    for well_idx in range(wells):
        # Generate well name (A01, A02, B01, etc.)
        row = well_idx // 12  # 12 columns per row
        col = well_idx % 12
        well_name = f"{chr(65 + row)}{col + 1:02d}"
        well_names.append(well_name)

        well_grp = root.create_group(well_name)

        # Create well metadata
        plate_meta["plate"]["wells"].append({"path": well_name})

        # Create field images
        well_meta = {"well": {"images": [], "version": "0.4"}}

        for field_idx in range(fields):
            field_name = f"Field{field_idx}"
            arr = well_grp.zeros(
                f"{field_name}/0",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
            )
            # Fill with unique data per well/field for testing
            # Use modulo to ensure value fits in dtype range
            fill_value = ((well_idx + 1) * 100 + field_idx) % 65536
            arr[:] = fill_value

            well_meta["well"]["images"].append({"path": f"{field_name}/0"})

        # Write well metadata using Path
        well_zattrs_path = zarr_path / well_name / ".zattrs"
        well_zattrs_path.parent.mkdir(parents=True, exist_ok=True)
        well_zattrs_path.write_text(json.dumps(well_meta))

    # Write plate metadata
    zattrs_path = zarr_path / ".zattrs"
    zattrs_path.write_text(json.dumps(plate_meta))

    return str(zarr_path), well_names, plate_meta


def generate_synthetic_zarr(
    path: str,
    shape: Tuple[int, ...] = (512, 512),
    chunks: Tuple[int, ...] = (256, 256),
    dtype: str = "uint16",
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...]]:
    """Generate simple synthetic zarr array for baseline tests.

    Args:
        path: Directory to create the zarr
        shape: Array shape
        chunks: Chunk size
        dtype: Data type string

    Returns:
        Tuple of (zarr_path, shape, chunks)
    """
    import zarr

    zarr_path = Path(path) / "test.zarr"
    arr = zarr.open_array(
        str(zarr_path),
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
    )

    # Populate with random data
    np.random.seed(42)  # Reproducible benchmarks
    arr[:] = np.random.randint(0, 1000, size=shape, dtype=np.dtype(dtype))

    return str(zarr_path), shape, chunks


def generate_multiresolution_zarr(
    path: str,
    base_shape: Tuple[int, int] = (1024, 1024),
    chunks: Tuple[int, int] = (256, 256),
    n_levels: int = 4,
    dtype: str = "uint16",
) -> Tuple[str, List[Tuple[int, int]], Dict]:
    """Generate multi-resolution zarr for scaled read benchmarks.

    Creates an OME-Zarr pyramid structure for testing virtual scaling
    and downsampling performance.

    Args:
        path: Directory to create the zarr
        base_shape: Shape of full-resolution level
        chunks: Chunk size
        n_levels: Number of pyramid levels (2x downsampling each)
        dtype: Data type string

    Returns:
        Tuple of (zarr_path, level_shapes, ome_metadata)
    """
    import zarr

    zarr_path = Path(path) / "pyramid.ome.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zarr_path), mode="w")
    level_shapes = []
    datasets = []

    for level in range(n_levels):
        scale_factor = 2 ** level
        level_shape = (
            max(1, base_shape[0] // scale_factor),
            max(1, base_shape[1] // scale_factor),
        )
        level_shapes.append(level_shape)

        arr = root.zeros(
            str(level),
            shape=level_shape,
            chunks=chunks,
            dtype=dtype,
        )
        # Fill with distinguishable data per level
        arr[:] = level * 100

        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale_factor, scale_factor]}
            ]
        })

    # Write OME-Zarr metadata
    ome_meta = {
        "multiscales": [{
            "name": "synthetic-pyramid",
            "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": datasets,
            "version": "0.4",
        }]
    }

    zattrs_path = zarr_path / ".zattrs"
    zattrs_path.write_text(json.dumps(ome_meta))

    return str(zarr_path), level_shapes, ome_meta


def generate_synthetic_tiff(
    path: str,
    shape: Tuple[int, int] = (256, 256),
    tile: Tuple[int, int] = (128, 128),
    dtype: str = "uint16",
) -> str:
    """Generate synthetic tiled TIFF for adapter tests.

    Args:
        path: Directory to create the TIFF
        shape: Image shape
        tile: Tile size for TIFF
        dtype: Data type string

    Returns:
        TIFF file path
    """
    import tifffile

    tiff_path = Path(path) / "synthetic.tif"
    np.random.seed(42)
    data = np.random.randint(0, 1000, size=shape, dtype=np.dtype(dtype))
    tifffile.imwrite(str(tiff_path), data, photometric="minisblack", tile=tile)

    return str(tiff_path)


def generate_synthetic_hdf5(
    path: str,
    shape: Tuple[int, int] = (256, 256),
    chunks: Tuple[int, int] = (128, 128),
    dtype: str = "uint16",
) -> str:
    """Generate synthetic HDF5 dataset for adapter tests.

    Args:
        path: Directory to create the HDF5
        shape: Dataset shape
        chunks: Chunk size
        dtype: Data type string

    Returns:
        HDF5 file path
    """
    import h5py

    h5_path = Path(path) / "synthetic.h5"
    np.random.seed(42)
    data = np.random.randint(0, 1000, size=shape, dtype=np.dtype(dtype))

    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("data", data=data, chunks=chunks)

    return str(h5_path)


# =============================================================================
# Measurement utilities
# =============================================================================


def measure_read_time(arr, slice_tuple: Tuple[slice, ...]) -> float:
    """Measure time to read a slice from a dask or zarr/numpy array.

    Args:
        arr: Dask array or zarr/numpy array
        slice_tuple: Slice specification

    Returns:
        Time in seconds
    """
    start = time.perf_counter()
    sliced = arr[slice_tuple]
    # Handle both dask arrays (need .compute()) and numpy/zarr (direct)
    if hasattr(sliced, 'compute'):
        sliced.compute()
    elapsed = time.perf_counter() - start
    return elapsed


def measure_throughput(
    arr,
    total_bytes: int,
    n_reads: int = 10,
) -> float:
    """Measure throughput in MB/s for multiple reads.

    Args:
        arr: Dask array
        total_bytes: Total bytes per read
        n_reads: Number of reads to average

    Returns:
        Throughput in MB/s
    """
    start = time.perf_counter()
    for _ in range(n_reads):
        arr.compute()
    elapsed = time.perf_counter() - start

    return (total_bytes * n_reads) / elapsed / 1e6  # MB/s
