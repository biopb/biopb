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

import numpy as np
import pytest
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.config import CacheConfig
from biopb_tensor_server.server import TensorFlightServer

from benchmarks.utils import (
    generate_multiresolution_zarr,
    generate_synthetic_hcs_plate,
    generate_synthetic_hdf5,
    generate_synthetic_tiff,
    generate_synthetic_zarr,
    reset_cache,
)

# =============================================================================
# pytest marker registration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "s3: tests requiring S3 network access")
    config.addinivalue_line(
        "markers", "nfs: tests requiring NFS/local filesystem access"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take longer to run (e.g., large file downloads)"
    )
    config.option.benchmark_sort = "name"
    config.option.benchmark_time_unit = "ms"
    if not getattr(config.option, "benchmark_json", None):
        benchmark_output = (
            Path(__file__).resolve().parent.parent / ".benchmarks" / "output.json"
        )
        benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        config.option.benchmark_json = benchmark_output.open("wb")


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


def get_all_source_ids() -> List:
    """Get pytest.param objects for all sources with appropriate marks.

    Use this for @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)

    Synthetic sources: no mark (always run)
    S3 sources: mark.s3 (run with -m s3 or when explicitly selected with -k)
    NFS sources: mark.nfs (run with -m nfs or when explicitly selected)
    """
    params = []
    for spec in load_all_source_configs():
        source_id = spec["id"]
        if "generator" in spec:
            # Synthetic - always run
            params.append(source_id)
        elif spec.get("url", "").startswith("s3://"):
            # S3 source - requires network
            params.append(pytest.param(source_id, marks=pytest.mark.s3))
        elif spec.get("path_env"):
            # NFS source - requires local filesystem
            params.append(pytest.param(source_id, marks=pytest.mark.nfs))
        else:
            params.append(source_id)
    return params


def get_synthetic_source_ids() -> List[str]:
    """Get IDs of synthetic sources only (no network required)."""
    return [c["id"] for c in load_all_source_configs() if "generator" in c]


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
        """Open direct reader based on source config.

        Uses specialized libraries for known formats, bioio as catch-all.
        """
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

        # Get path or URL - S3 sources have url, local sources have path
        path = self._paths.get(source_id)
        url = spec.get("url")

        # Specialized handlers for known formats
        if source_type == "zarr":
            return zarr.open_array(path, mode="r")

        elif source_type in ("ome_zarr", "ome_zarr_hcs"):
            if url:  # S3
                import s3fs

                fs = s3fs.S3FileSystem(anon=True)
                return zarr.open_group(s3fs.S3Map(fs, url), mode="r")
            return zarr.open_group(path, mode="r")

        elif source_type == "hdf5":
            import h5py

            return h5py.File(path, "r")

        # Catch-all: use bioio for any unhandled format
        # Supports: OME-TIFF, TIFF, CZI, ND2, LIF, DV, and many more
        # bioio can read from local path or S3 URL directly
        import bioio

        target = url or path
        fs_kwargs = {"anon": True} if target.startswith("s3://") else {}
        return bioio.BioImage(target, fs_kwargs=fs_kwargs)

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
            # For single-tensor sources, tensor_id == source_id means use first level
            if tensor_id == source_id or tensor_id.isdigit():
                level_key = str(int(tensor_id) if tensor_id.isdigit() else 0)
                arr = da.from_zarr(reader[level_key])
            else:
                # Navigate to specific tensor path (e.g., well/field for HCS)
                parts = tensor_id.split("/")
                zarr_obj = reader
                for part in parts:
                    zarr_obj = zarr_obj[part]
                arr = da.from_zarr(zarr_obj)

        elif source_type == "hdf5":
            dataset_name = tensor_id if tensor_id != source_id else "data"
            h5_ds = reader[dataset_name]
            arr = da.from_array(h5_ds, chunks=h5_ds.chunks)

        else:
            # Catch-all via bioio: use xarray_dask_data for lazy loading
            # BioImage handles OME-TIFF, TIFF, CZI, ND2, LIF, DV, etc.
            # xarray_dask_data returns dask-backed DataArray with proper dims
            import bioio

            if isinstance(reader, bioio.BioImage):
                # Handle scene selection - tensor_id may be scene name like "Image:0"
                if tensor_id in reader.scenes:
                    reader.set_scene(tensor_id)
                elif tensor_id.isdigit() and int(tensor_id) < len(reader.scenes):
                    reader.set_scene(int(tensor_id))

                # Get dask array from xarray DataArray
                xarr = reader.xarray_dask_data
                arr = xarr.data  # Extract dask array from xarray
            else:
                raise ValueError(f"Unexpected reader type: {type(reader)}")

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


@pytest.fixture
def temp_cache_dir() -> Generator[str, None, None]:
    """Clean scratch directory for synthetic data.

    Function-scoped to ensure each test gets its own directory,
    avoiding data conflicts between tests that overwrite files.

    Location controlled by BIOPB_SYNTHETIC_DATA_DIR env var,
    defaults to system temp directory if not set.
    """
    synthetic_dir = os.environ.get("BIOPB_SYNTHETIC_DATA_DIR")
    if synthetic_dir:
        # Create unique subdirectory in specified location
        subdir = Path(synthetic_dir) / f"bench_{os.getpid()}_{int(time.time() * 1000)}"
        subdir.mkdir(parents=True, exist_ok=True)
        yield str(subdir)
        # Cleanup
        import shutil

        shutil.rmtree(subdir, ignore_errors=True)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir


@pytest.fixture
def bench_server(
    temp_cache_dir: str,
    request: pytest.FixtureRequest,
) -> Generator[TensorFlightServer, None, None]:
    """Start TensorFlightServer with configurable cache backend.

    When BIOPB_BENCH_SERVER_URL is set, connects to existing production server
    instead of creating a new one (for container-based benchmarking).

    Cache backend controlled by:
    - pytest parametrize (request.param) - for explicit comparison tests
    - BIOPB_CACHE_BACKEND env var - default backend override
    - defaults to "file" if neither specified
    """
    # Check for existing production server
    existing_url = os.environ.get("BIOPB_BENCH_SERVER_URL")
    if existing_url:
        # Connect to existing server - no setup/teardown needed
        # Return a mock object with the URL for client fixtures
        server = TensorFlightServer(existing_url)
        server._bench_port = int(existing_url.split(":")[-1])
        server._bench_backend = "production"
        server._bench_cache_dir = temp_cache_dir
        server._production_mode = True  # Flag to skip shutdown
        yield server
        return

    # Create ephemeral test server (default behavior)
    # Priority: pytest param > env var > default "file"
    backend = getattr(request, "param", None)
    if backend is None:
        backend = os.environ.get("BIOPB_CACHE_BACKEND", "file")

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
    server._production_mode = False

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
        zarr_path, _, _ = generate_synthetic_hcs_plate(
            cache_dir, wells, fields, shape, chunks, dtype
        )
        return zarr_path

    elif generator == "generate_multiresolution_zarr":
        base_shape = tuple(params.get("base_shape", [1024, 1024]))
        chunks = tuple(params.get("chunks", [256, 256]))
        n_levels = params.get("n_levels", 4)
        dtype = params.get("dtype", "uint16")
        zarr_path, _, _ = generate_multiresolution_zarr(
            cache_dir, base_shape, chunks, n_levels, dtype
        )
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


def _register_source_with_server(
    spec: Dict, path: str, server: TensorFlightServer
) -> str:
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

    In development mode: generates synthetic data and registers with ephemeral server.
    In production mode (BIOPB_BENCH_SERVER_URL set): assumes sources exist via file watcher.

    Skips S3/NFS sources if not available.

    Returns the source spec dict with keys:
    - id: source_id
    - type: source type (zarr, ome_zarr, etc.)
    - expected_tensors: list of tensor_ids to test
    - path: generated data path (for synthetic, development mode)
    """
    source_id = request.param

    # Find the config
    configs = load_all_source_configs()
    spec = next(c for c in configs if c["id"] == source_id)

    # Production mode: skip synthetic sources, assume real data exists
    production_mode = getattr(bench_server, "_production_mode", False)
    if production_mode:
        if "generator" in spec:
            pytest.skip(f"Synthetic source {source_id} skipped in production mode")
        # For production mode, just return spec without registering
        yield spec
        return

    # Development mode: handle different source types
    # Check if test was selected via marker (s3/nfs) - don't skip if so
    has_s3_marker = request.node.get_closest_marker("s3") is not None
    has_nfs_marker = request.node.get_closest_marker("nfs") is not None

    if is_s3_source(source_id):
        if not has_s3_marker:
            pytest.skip(f"S3 source {source_id} - run with -m s3 or select explicitly")

        # S3 source - register URL-based adapter with server
        url = spec.get("url")
        source_type = spec.get("type")

        # For public S3 buckets, use anonymous access
        from biopb_tensor_server.remote import CredentialProfile, CredentialsConfig

        # Create anon credentials profile (empty key/secret triggers anon=True)
        anon_profile = CredentialProfile(
            name="anon",
            storage_type="s3",
            key=None,
            secret=None,
        )
        anon_credentials = CredentialsConfig(
            default_profile="anon",
            profiles=[anon_profile],
        )

        # Map benchmark config type to adapter class
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        adapter_cls_map = {
            "ome_tiff": OmeTiffAdapter,
            "tiff": OmeTiffAdapter,
            "ome_zarr": OmeZarrAdapter,
        }
        adapter_cls = adapter_cls_map.get(source_type)
        if adapter_cls is None:
            pytest.skip(f"No adapter for source type: {source_type}")

        # Create SourceConfig for S3 with anon credentials profile
        from biopb_tensor_server.config import SourceConfig

        source_config = SourceConfig(
            url=url,
            type=adapter_cls.SOURCE_TYPE,
            source_id=source_id,
            credentials_profile="anon",
        )

        # Create adapter from config with anon credentials
        adapter = adapter_cls.create_from_config(source_config, anon_credentials)
        bench_server.register_source(source_id, adapter)

        yield spec

        bench_server.unregister_source(source_id)

    elif is_nfs_source(source_id):
        if not has_nfs_marker:
            pytest.skip(
                f"NFS source {source_id} - run with -m nfs or select explicitly"
            )

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
    """Baseline client for direct library access.

    Includes all configured sources (synthetic, S3, NFS).
    S3 sources are handled via bioio catch-all.
    """
    configs = load_all_source_configs()
    return BaselineClient(configs, cache_dir=temp_cache_dir)


@pytest.fixture
def bench_client_flight(bench_server: TensorFlightServer):
    """Flight client connected to benchmark server."""
    from biopb.tensor import TensorFlightClient

    port = getattr(bench_server, "_bench_port", 8815)
    client = TensorFlightClient(f"grpc://localhost:{port}")

    yield client

    client.close()


@pytest.fixture
def bench_client_flight_no_cache(bench_server: TensorFlightServer):
    """Flight client with cache disabled for server cache benchmarks."""
    from biopb.tensor import TensorFlightClient

    port = getattr(bench_server, "_bench_port", 8815)
    client = TensorFlightClient(f"grpc://localhost:{port}", cache_bytes=0)

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
    """Automatically reset cache after each test.

    In production mode, skips cache reset to preserve server state.
    """
    yield

    # Skip reset in production mode (server managed externally)
    # Check via env var to avoid fixture teardown ordering issues
    if os.environ.get("BIOPB_BENCH_SERVER_URL"):
        return

    reset_cache()
