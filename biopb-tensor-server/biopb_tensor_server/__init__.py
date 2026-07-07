"""BioPB Tensor Arrow Flight server for multi-dimensional array storage.

This package provides server-side components for exposing chunked
multi-dimensional arrays through Apache Arrow Flight protocol.

Key components:
- TensorFlightServer: Flight server implementation
- BackendAdapter: Abstract interface for storage backends
- Adapters: ZarrAdapter, Hdf5Adapter, OmeZarrAdapter, OmeTiffAdapter, ZeissAdapter, etc.
- Cache: CacheManager, CacheConfig for caching computed virtual chunks

Note: This package is not distributed via PyPI. Install locally with:
    pip install -e biopb-tensor-server/
"""

from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter

# OME-TIFF is pure-tifffile (always available); it lives in its own module.
from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.tiff import (
    MicroManagerLegacyAdapter,
    TiffSequenceAdapter,
)
from biopb_tensor_server.adapters.zarr import ZarrAdapter

# Optional bioio vendor-format adapters (installed via the [aics] extra)
try:
    from biopb_tensor_server.adapters.bioio import (
        AicsImageIoAdapter,
        DvAdapter,
        LeicaAdapter,
        NikonAdapter,
        OlympusAdapter,
        ZeissAdapter,
    )
except ImportError:
    ZeissAdapter = None  # type: ignore
    LeicaAdapter = None  # type: ignore
    NikonAdapter = None  # type: ignore
    DvAdapter = None  # type: ignore
    OlympusAdapter = None  # type: ignore
    AicsImageIoAdapter = None  # type: ignore

from biopb_tensor_server.base import (
    BackendAdapter,
    SourceAdapter,
    TensorAdapter,
    TensorReadPlan,
)
from biopb_tensor_server.cache import (
    CacheBackend,
    CacheEntry,
    CacheManager,
    CacheStats,
    EntryState,
    MemoryCacheBackend,
)
from biopb_tensor_server.chunk import ChunkEndpoint
from biopb_tensor_server.config import (
    CacheConfig,
)
from biopb_tensor_server.server import (
    TensorFlightServer,
    serve,
)

try:
    from biopb_tensor_server._version import __version__
except ImportError:
    try:
        import importlib.metadata as _importlib_metadata
    except ImportError:
        import importlib_metadata as _importlib_metadata

    try:
        __version__ = _importlib_metadata.version("biopb-tensor-server")
    except Exception:
        __version__ = "0.1.0"

__all__ = [
    "TensorFlightServer",
    "serve",
    "SourceAdapter",
    "TensorAdapter",
    "BackendAdapter",
    "ChunkEndpoint",
    "TensorReadPlan",
    "ZarrAdapter",
    "Hdf5Adapter",
    "OmeTiffAdapter",
    "ZeissAdapter",
    "LeicaAdapter",
    "NikonAdapter",
    "DvAdapter",
    "OlympusAdapter",
    "AicsImageIoAdapter",
    "TiffSequenceAdapter",
    "MicroManagerLegacyAdapter",
    "OmeZarrAdapter",
    "CacheManager",
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "MemoryCacheBackend",
    "EntryState",
    "CacheConfig",
]
