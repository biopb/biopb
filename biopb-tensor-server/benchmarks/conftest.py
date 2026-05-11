"""pytest fixtures for benchmark suite.

Provides server/client setup, BaselineClient for direct library comparison,
and data source configuration loading for parametrization.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import pytest
import numpy as np

from biopb_tensor_server.config import CacheConfig
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.server import TensorFlightServer

from benchmarks.utils import (
    S3_TEST_DATA_URL,
    NFS_TEST_DATA_DIR,
    generate_synthetic_hcs_plate,
    generate_synthetic_zarr,
    generate_multiresolution_zarr,
    generate_synthetic_tiff,
    generate_synthetic_hdf5,
    reset_cache,
)


# =============================================================================
# Data source configuration loading
# =============================================================================


def load_all_source_configs() -> List[Dict]:
    """Load all data source configs from data_sources/*.json."""
    config_dir = Path(__file__).parent / "data_sources"
    configs = []
    for config_file in config_dir.glob("*.json"):
        config = json.loads(config_file.read_text())
        configs.extend(config.get("sources", []))
    return configs


def get_all_source_ids() -> List[str]:
    """Get IDs of all configured sources (synthetic, S3, NFS)."""
    return [c["id"] for c in load_all_source_configs()]


def is_synthetic_source(source_id: str) -> bool:
    """Check if source is synthetic (no network required)."""
    configs = load_all_source_configs()
    spec = next((c for c in configs if c["id"] == source_id), None)
    return spec and "generator" in spec


def is_s3_source(source_id: str) -> bool:
    """Check if source is S3 remote."""
    configs = load_all_source_configs()
    spec = next((c for c in configs if c["id"] == source_id), None)
    return spec and spec.get("url", "").startswith("s3://")


def is_nfs_source(source_id: str) -> bool:
    """Check if source requires NFS mount."""
    configs = load_all_source_configs()
    spec = next((c for c in configs if c["id"] == source_id), None)
    return spec and spec.get("path_env")


# =============================================================================
# BaselineClient - direct library access matching TensorFlightClient API
# =============================================================================


class BaselineClient:
    """Direct library access matching TensorFlightClient API.

    Wraps low-level libraries directly (zarr/tifffile/h5py),
    returning dask arrays for lazy loading. No Flight protocol overhead,
    no application-level cache (only OS page cache).
    """

    def __init__(self, source_configs: List[Dict], cache_dir: Optional[str] = None):
        self._source_configs = source_configs
        self._cache_dir = cache_dir
        self._sources: Dict[str, Dict] = {}
        self._readers: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._paths: Dict[str, str] = {}  # Generated paths
        self._init_sources()

    def _init_sources(self):
        """Initialize sources from configs."""
        for spec in self._source_configs:
            source_id = spec["id"]

            # Skip NFS sources if env var not set
            if spec.get("path_env"):
                path = os.environ.get(spec["path_env"])
                if not path or not Path(path).exists():
                    continue
                spec = dict(spec)
                spec["path"] = path

            self._sources[source_id] = spec

    def generate_data(self, spec: Dict) -> str:
        """Generate synthetic data based on config."""
        generator = spec.get("generator")
        params = spec.get("params", {})
        source_id = spec["id"]
        cache_dir = self._cache_dir or tempfile.mkdtemp()

        if generator == "generate_synthetic_zarr":
            shape = tuple(params.get("shape", [512, 512]))
            chunks = tuple(params.get("chunks", [256, 256]))
            dtype = params.get("dtype", "uint16")
            zarr_path, _, _ = generate_synthetic_zarr(cache_dir, shape, chunks, dtype)
            self._paths[source_id] = zarr_path
            return zarr_path

        elif generator == "generate_synthetic_hcs_plate":
            wells = params.get("wells", 96)
            fields = params.get("fields", 4)
            shape = tuple(params.get("shape", [256, 256]))
            chunks = tuple(params.get("chunks", [128, 128]))
            dtype = np.dtype(params.get("dtype", "uint16"))
            zarr_path, _, plate_meta = generate_synthetic_hcs_plate(
                cache_dir, wells, fields, shape, chunks, dtype
            )
            self._metadata[source_id] = plate_meta
            self._paths[source_id] = zarr_path
            return zarr_path

        elif generator == "generate_multiresolution_zarr":
            base_shape = tuple(params.get("base_shape", [1024, 1024]))
            chunks = tuple(params.get("chunks", [256, 256]))
            n_levels = params.get("n_levels", 4)
            dtype = params.get("dtype", "uint16")
            zarr_path, _, ome_meta = generate_multiresolution_zarr(
                cache_dir, base_shape, chunks, n_levels, dtype
            )
            self._metadata[source_id] = ome_meta
            self._paths[source_id] = zarr_path
            return zarr_path

        elif generator == "generate_synthetic_tiff":
            shape = tuple(params.get("shape", [256, 256]))
            tile = tuple(params.get("tile", [128, 128]))
            dtype = params.get("dtype", "uint16")
            tiff_path = generate_synthetic_tiff(cache_dir, shape, tile, dtype)
            self._paths[source_id] = tiff_path
            return tiff_path

        elif generator == "generate_synthetic_hdf5":
            shape = tuple(params.get("shape", [256, 256]))
            chunks = tuple(params.get("chunks", [128, 128]))
            dtype = params.get("dtype", "uint16")
            h5_path = generate_synthetic_hdf5(cache_dir, shape, chunks, dtype)
            self._paths[source_id] = h5_path
            return h5_path

        raise ValueError(f"Unknown generator: {generator}")

    def _open_reader(self, source_id: str) -> Any:
        """Open direct reader based on source config."""
        import zarr

        spec = self._sources[source_id]
        source_type = spec.get("type")

        # Use existing path if already generated (from data_source fixture)
        path = spec.get("path")
        if path and source_id not in self._paths:
            self._paths[source_id] = path

        # Generate if needed
        if source_id not in self._paths and "generator" in spec:
            self.generate_data(spec)

        path = self._paths.get(source_id)

        if source_type == "zarr":
            return zarr.open_array(path, mode="r")

        elif source_type in ("ome_zarr", "ome_zarr_hcs"):
            if spec.get("url"):  # S3
                import s3fs
                fs = s3fs.S3FileSystem(anon=True)
                return zarr.open_group(s3fs.S3Map(fs, spec["url"]), mode="r")
            return zarr.open_group(path, mode="r")

        elif source_type in ("ome_tiff", "tiff"):
            import tifffile
            return tifffile.TiffFile(path)

        elif source_type == "hdf5":
            import h5py
            return h5py.File(path, "r")

        raise ValueError(f"Unknown source type: {source_type}")

    def list_sources(self) -> Dict[str, Dict]:
        """List available data sources."""
        result = {}
        for source_id, spec in self._sources.items():
            tensors = spec.get("expected_tensors", [source_id])
            result[source_id] = {
                "source_id": source_id,
                "source_url": spec.get("url") or self._paths.get(source_id),
                "source_type": spec.get("type"),
                "tensors": [{"array_id": t} for t in tensors],
            }
        return result

    def get_tensor(
        self,
        source_id: str,
        tensor_id: str,
        slice_hint: Optional[Tuple[slice, ...]] = None,
        scale_hint: Optional[Sequence[int]] = None,
    ) -> Any:
        """Get lazy dask array for tensor."""
        import dask.array as da
        import zarr

        spec = self._sources.get(source_id)
        if not spec:
            raise ValueError(f"Unknown source: {source_id}")

        if source_id not in self._readers:
            self._readers[source_id] = self._open_reader(source_id)

        reader = self._readers[source_id]
        source_type = spec.get("type")

        if source_type == "zarr":
            arr = da.from_zarr(reader)

        elif source_type in ("ome_zarr", "ome_zarr_hcs"):
            if tensor_id.isdigit():
                arr = da.from_zarr(reader[str(int(tensor_id))])
            else:
                parts = tensor_id.split("/")
                zarr_obj = reader
                for part in parts:
                    zarr_obj = zarr_obj[part]
                arr = da.from_zarr(zarr_obj)

        elif source_type in ("ome_tiff", "tiff"):
            page_idx = int(tensor_id) if tensor_id.isdigit() else 0
            numpy_arr = reader.pages[page_idx].asarray()
            chunks = spec.get("params", {}).get("tile", (256, 256))
            arr = da.from_array(numpy_arr, chunks=chunks)

        elif source_type == "hdf5":
            dataset_name = tensor_id if tensor_id != source_id else "data"
            h5_ds = reader[dataset_name]
            arr = da.from_array(h5_ds, chunks=h5_ds.chunks)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        if slice_hint is not None:
            arr = arr[slice_hint]

        return arr

    def get_source_metadata(self, source_id: str) -> dict:
        """Get source-level metadata."""
        return self._metadata.get(source_id, {})

    def cache_info(self) -> Dict:
        """Return empty stats - baseline uses OS page cache only."""
        return {"size_bytes": 0, "max_bytes": 0, "item_count": 0}

    def close(self):
        """Close any open handles."""
        for reader in self._readers.values():
            if hasattr(reader, "close"):
                reader.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Server fixtures
# =============================================================================


@pytest.fixture(scope="session")
def temp_cache_dir() -> Generator[str, None, None]:
    """Clean scratch directory for synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def bench_server(
    temp_cache_dir: str,
    request: pytest.FixtureRequest,
) -> Generator[TensorFlightServer, None, None]:
    """Start TensorFlightServer with configurable cache backend."""
    backend = getattr(request, "param", "file")

    if backend == "memory":
        config = CacheConfig(
            backend="memory",
            memory_max_entries=1024,
            memory_max_bytes=512 * 1024 * 1024,
        )
    elif backend == "file":
        config = CacheConfig(
            backend="file",
            file_cache_dir=Path(temp_cache_dir),
            file_max_segment_bytes=256 * 1024 * 1024,
            file_max_total_bytes=64 * 1024 * 1024 * 1024,
        )
    else:
        raise ValueError(f"Unknown cache backend: {backend}")

    CacheManager.initialize(config)

    import random
    port = random.randint(8900, 8999)
    server = TensorFlightServer(f"grpc://localhost:{port}")
    server._bench_port = port
    server._bench_backend = backend
    server._bench_cache_dir = temp_cache_dir

    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    yield server

    server.shutdown()
    reset_cache()


# =============================================================================
# Data source fixture (parametrized from config)
# =============================================================================


def _generate_and_get_path(spec: Dict, cache_dir: str) -> str:
    """Generate synthetic data and return path."""
    generator = spec.get("generator")
    params = spec.get("params", {})
    source_id = spec["id"]

    if generator == "generate_synthetic_zarr":
        shape = tuple(params.get("shape", [512, 512]))
        chunks = tuple(params.get("chunks", [256, 256]))
        dtype = params.get("dtype", "uint16")
        zarr_path, _, _ = generate_synthetic_zarr(cache_dir, shape, chunks, dtype)
        return zarr_path

    elif generator == "generate_synthetic_hcs_plate":
        wells = params.get("wells", 96)
        fields = params.get("fields", 4)
        shape = tuple(params.get("shape", [256, 256]))
        chunks = tuple(params.get("chunks", [128, 128]))
        dtype = np.dtype(params.get("dtype", "uint16"))
        zarr_path, _, _ = generate_synthetic_hcs_plate(cache_dir, wells, fields, shape, chunks, dtype)
        return zarr_path

    elif generator == "generate_multiresolution_zarr":
        base_shape = tuple(params.get("base_shape", [1024, 1024]))
        chunks = tuple(params.get("chunks", [256, 256]))
        n_levels = params.get("n_levels", 4)
        dtype = params.get("dtype", "uint16")
        zarr_path, _, _ = generate_multiresolution_zarr(cache_dir, base_shape, chunks, n_levels, dtype)
        return zarr_path

    elif generator == "generate_synthetic_tiff":
        shape = tuple(params.get("shape", [256, 256]))
        tile = tuple(params.get("tile", [128, 128]))
        dtype = params.get("dtype", "uint16")
        return generate_synthetic_tiff(cache_dir, shape, tile, dtype)

    elif generator == "generate_synthetic_hdf5":
        shape = tuple(params.get("shape", [256, 256]))
        chunks = tuple(params.get("chunks", [128, 128]))
        dtype = params.get("dtype", "uint16")
        return generate_synthetic_hdf5(cache_dir, shape, chunks, dtype)

    raise ValueError(f"Unknown generator: {generator}")


def _register_source_with_server(spec: Dict, path: str, server: TensorFlightServer) -> str:
    """Register source with Flight server using registry mechanism.

    Uses the server's adapter registry and SourceConfig pattern,
    avoiding hardcoded type mappings.

    Returns source_id.
    """
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.config import SourceConfig

    source_id = spec["id"]
    source_type = spec.get("type")

    # Map benchmark config types to registry types
    # Registry uses: "zarr", "ome-zarr", "ome-tiff", "hdf5", "aics", etc.
    registry_type_map = {
        "zarr": "zarr",
        "ome_zarr": "ome-zarr",
        "ome_zarr_hcs": "ome-zarr",
        "ome_tiff": "ome-tiff",
        "tiff": "ome-tiff",
        "hdf5": "hdf5",
        # AicsImageIO formats (covers many vendor formats)
        "czi": "aics",
        "lif": "aics",
        "nd2": "aics",
        "dv": "aics",
        "lsm": "aics",
        "oif": "aics",
        "oib": "aics",
        # Medical imaging
        "dicom": "dicom",
        "nifti": "nifti",
    }

    registry_type = registry_type_map.get(source_type, source_type)

    # Create SourceConfig
    source_config = SourceConfig(
        url=path,
        type=registry_type,
        source_id=source_id,
        dim_labels=spec.get("dim_labels"),
        dataset="data" if source_type == "hdf5" else None,
    )

    # Get adapter class from registry and create instance
    registry = get_default_registry()
    adapter_cls = registry.get_adapter_for_type(registry_type)

    if adapter_cls is None:
        raise ValueError(f"No adapter registered for type: {registry_type}")

    adapter = adapter_cls.create_from_config(source_config)
    server.register_source(source_id, adapter)

    return source_id


@pytest.fixture
def data_source(
    request: pytest.FixtureRequest,
    temp_cache_dir: str,
    bench_server: TensorFlightServer,
) -> Generator[Dict, None, None]:
    """Parametrized fixture over ALL data source configs.

    Generates synthetic data and registers with server.
    Skips S3/NFS sources if not available.

    Returns the source spec dict with keys:
    - id: source_id
    - type: source type (zarr, ome_zarr, etc.)
    - expected_tensors: list of tensor_ids to test
    - path: generated data path (for synthetic)
    """
    source_id = request.param

    # Find the config
    configs = load_all_source_configs()
    spec = next(c for c in configs if c["id"] == source_id)

    # Handle different source types
    if is_s3_source(source_id):
        pytest.skip(f"S3 source {source_id} requires network (run with -m s3)")

    elif is_nfs_source(source_id):
        path_env = spec.get("path_env")
        nfs_path = os.environ.get(path_env)
        if not nfs_path or not Path(nfs_path).exists():
            pytest.skip(f"NFS source requires {path_env} env var")
        spec["path"] = nfs_path
        # NFS discovery would go here when implemented
        pytest.skip(f"NFS discovery not yet implemented for {source_id}")

    else:
        # Synthetic sources - generate and register
        path = _generate_and_get_path(spec, temp_cache_dir)
        spec["path"] = path

        _register_source_with_server(spec, path, bench_server)

        yield spec

        bench_server.unregister_source(source_id)


# =============================================================================
# Client fixtures (parametrized over baseline vs flight)
# =============================================================================


@pytest.fixture
def bench_client_baseline(temp_cache_dir: str) -> BaselineClient:
    """Baseline client for direct library access."""
    configs = load_all_source_configs()
    synthetic_configs = [c for c in configs if "generator" in c]
    return BaselineClient(synthetic_configs, cache_dir=temp_cache_dir)


@pytest.fixture
def bench_client_flight(bench_server: TensorFlightServer):
    """Flight client connected to benchmark server."""
    from biopb.tensor import TensorFlightClient

    port = getattr(bench_server, "_bench_port", 8815)
    client = TensorFlightClient(f"grpc://localhost:{port}")

    yield client

    client.close()


@pytest.fixture(params=["baseline", "flight"])
def bench_client(
    request: pytest.FixtureRequest,
    bench_client_baseline: BaselineClient,
    bench_client_flight,
) -> Generator[Any, None, None]:
    """Parametrized client: baseline (direct) or flight.

    Tests using this fixture run against BOTH clients.
    """
    client_type = request.param

    if client_type == "baseline":
        yield bench_client_baseline
    else:
        yield bench_client_flight


# =============================================================================
# Cleanup fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_cache_after_test():
    """Automatically reset cache after each test."""
    yield
    reset_cache()