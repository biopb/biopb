"""Cache-backed source adapter for ephemeral uploaded data.

Design: One adapter instance per cache-backed source, following the existing
BackendAdapter pattern used by OmeZarrAdapter, etc.

- CachedSourceAdapter instances registered in server._sources dict (same as other adapters)
- Metadata (shape, dtype, chunk_shape) stored in adapter instance
- Chunk data stored in CacheManager keyed by chunk_id
- When cache evicts chunks, adapter returns Flight error on read (source "gone")
- "Dead" adapters accumulate until server restart (acceptable for ephemeral sources)

Registration flow (bypasses discovery/registry):
DoPut → direct instantiation → server._sources

Chunk ID format: array_id + "/" + chunk_key
Chunk data: stored in CacheManager keyed by full chunk_id
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.chunk import ChunkEndpoint, encode_chunk_id
from biopb_tensor_server.cache import CacheManager

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import ClaimContext, DiscoveryState, SourceClaim

logger = logging.getLogger(__name__)


class CachedSourceAdapter(SourceAdapter, TensorAdapter):
    """Adapter for cache-backed uploaded sources.

    One instance per source, registered in server._sources dict.

    Chunk ID format: array_id + "/" + chunk_key (bounds start coords as "0/1/2")
    Chunk data stored in CacheManager keyed by full chunk_id.

    Cache-backed sources allow arbitrary chunk bounds (no uniformity enforcement).
    """

    _single_tensor_source = True

    @classmethod
    def create_from_config(cls, source: 'SourceConfig', credentials_config: Optional[Any] = None) -> 'CachedSourceAdapter':
        """Cache-backed sources are not created from config.

        Raises NotImplementedError - use direct instantiation via DoPut.
        """
        raise NotImplementedError("CachedSourceAdapter is created via DoPut, not config")

    def __init__(
        self,
        source_id: str,
        shape: List[int],
        dtype: str,
        chunk_shape: List[int],
        dim_labels: Optional[List[str]] = None,
        ome_metadata: Optional[dict] = None,
    ):
        """Initialize cache-backed source adapter.

        Args:
            source_id: Unique source identifier (e.g., "cache_abc123")
            shape: Array shape
            dtype: Data type string (numpy format)
            chunk_shape: Nominal chunk size per dimension
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict
        """
        self.source_id = source_id
        self._shape = tuple(shape)
        self._dtype = dtype
        self._chunk_shape = tuple(chunk_shape)
        self._dim_labels = dim_labels or [f"dim{i}" for i in range(len(shape))]
        self._ome_metadata = ome_metadata or {}

        # Track actually-written chunks: start coords -> full bounds
        self._written_chunks: Dict[bytes, ChunkBounds] = {}

        # Required fields for BackendAdapter
        self._source_url = f"cache://{source_id}"
        self._source_type = "cache"

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return TensorDescriptor for this cache source."""
        return TensorDescriptor(
            array_id=self.source_id,
            dim_labels=self._dim_labels,
            shape=list(self._shape),
            chunk_shape=list(self._chunk_shape),
            dtype=self._dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """Cache sources are single-tensor."""
        return [self.get_tensor_descriptor()]

    def get_metadata(self) -> dict:
        """Return OME metadata."""
        return self._ome_metadata

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds.

        Cache-backed sources have no backend data source - data is only
        available in cache and must be accessed via chunk_id.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Raises:
            FlightServerError: Cannot read data from cache-backed source
        """
        # Validate bounds first (base class validation)
        desc = self.get_tensor_descriptor()
        shape = tuple(desc.shape)
        self._validate_bounds(bounds, shape)

        raise flight.FlightServerError(
            f"Cannot read data from cache-backed source {self.source_id}. "
            f"Cache-backed sources have no backend data - use chunk_id access."
        )

    def write_chunk(self, bounds: ChunkBounds, data: np.ndarray) -> None:
        """Write chunk data to cache.

        For cache-backed sources, arbitrary bounds allowed.

        Args:
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data (any shape matching bounds)
        """
        flat_data = pa.array([data.ravel()])
        self.write_chunk_arrow(bounds, flat_data, data.shape, data.dtype)

    def write_chunk_arrow(
        self,
        bounds: ChunkBounds,
        data: pa.Array | pa.ChunkedArray,
        logical_shape: Tuple[int, ...] | List[int],
        dtype: np.dtype | str,
    ) -> None:
        """Write chunk data to cache without converting Arrow payloads through NumPy.

        Args:
            bounds: Chunk start/stop coordinates
            data: Flattened Arrow values or list array payload for the chunk
            logical_shape: Logical chunk shape matching bounds
            dtype: NumPy dtype or dtype string for the chunk data
        """
        cache_manager = CacheManager.get_instance()
        if cache_manager is None:
            raise RuntimeError("Cache not initialized")

        chunk_id = encode_chunk_id(self.source_id, bounds)

        if isinstance(data, pa.ChunkedArray):
            data = data.combine_chunks()

        if pa.types.is_list(data.type):
            list_arr = data
            values = data.values
        else:
            offsets = pa.array([0, len(data)], type=pa.int32())
            list_arr = pa.ListArray.from_arrays(offsets, data)
            values = data

        logical_shape = list(logical_shape)
        dtype_str = np.dtype(dtype).str
        size_bytes = values.nbytes

        batch = pa.RecordBatch.from_arrays(
            [
                list_arr,
                pa.array([logical_shape]),
                pa.array([dtype_str]),
            ],
            ["data", "shape", "dtype"]
        )

        entry, is_owner = cache_manager.start_compute(
            chunk_id,
            metadata={"bounds": bounds.SerializeToString()},
        )
        if is_owner:
            cache_manager.complete_entry(chunk_id, batch, size_bytes)

        self._written_chunks[chunk_id] = bounds

        logger.debug(
            f"write_chunk_arrow: stored {size_bytes} bytes at bounds "
            f"{list(bounds.start)} to {list(bounds.stop)}"
        )