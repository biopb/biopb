"""BioPB Tensor Arrow Flight server for multi-dimensional array storage.

This package provides server-side components for exposing chunked
multi-dimensional arrays through Apache Arrow Flight protocol.

Key components:
- TensorFlightServer: Flight server implementation
- BackendAdapter: Abstract interface for storage backends
- Adapters: ZarrAdapter, Hdf5Adapter, OmeTiffAdapter, OmeZarrAdapter
- Cache: CacheManager, CacheConfig for caching computed virtual chunks

Note: This package is not distributed via PyPI. Install locally with:
    pip install -e biopb-tensor-server/
"""

from biopb_tensor_server.server import (
    TensorFlightServer,
    serve,
)

from biopb_tensor_server.base import (
    BackendAdapter,
    ChunkEndpoint,
    TensorReadPlan,
    ComputeBackendOptions,
    build_arrow_schema,
    configure_compute_backend,
    get_compute_backend_options,
    plan_tensor_read,
    resolve_chunk_data,
    _chunks_intersect,
    _encode_chunk_id,
    _decode_chunk_id,
)

from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
from biopb_tensor_server.adapters.tiff import OmeTiffAdapter, MultiFileOmeTiffAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

from biopb_tensor_server.cache import (
    CacheManager,
    CacheBackend,
    CacheEntry,
    CacheKey,
    CacheStats,
    MemoryCacheBackend,
    EntryState,
)

from biopb_tensor_server.config import (
    CacheConfig,
)

try:
    import importlib.metadata as _importlib_metadata
except ImportError:
    import importlib_metadata as _importlib_metadata

try:
    __version__ = _importlib_metadata.version("biopb-tensor-server")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    'TensorFlightServer',
    'serve',
    'BackendAdapter',
    'ChunkEndpoint',
    'TensorReadPlan',
    'ComputeBackendOptions',
    'build_arrow_schema',
    'configure_compute_backend',
    'get_compute_backend_options',
    'plan_tensor_read',
    'resolve_chunk_data',
    '_chunks_intersect',
    '_encode_chunk_id',
    '_decode_chunk_id',
    'ZarrAdapter',
    'Hdf5Adapter',
    'OmeTiffAdapter',
    'MultiFileOmeTiffAdapter',
    'OmeZarrAdapter',
    'CacheManager',
    'CacheBackend',
    'CacheEntry',
    'CacheKey',
    'CacheStats',
    'MemoryCacheBackend',
    'EntryState',
    'CacheConfig',
]