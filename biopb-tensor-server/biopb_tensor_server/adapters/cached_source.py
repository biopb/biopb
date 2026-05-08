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
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import BackendAdapter
from biopb_tensor_server.chunk import (
    ChunkEndpoint,
    encode_chunk_id,
)
from biopb_tensor_server.cache import CacheManager

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import SourceClaim

logger = logging.getLogger(__name__)


class CachedSourceAdapter(BackendAdapter):
    """Adapter for cache-backed uploaded sources.

    One instance per source, registered in server._sources dict.

    Chunk ID format: array_id + "/" + chunk_key (bounds start coords as "0/1/2")
    Chunk data stored in CacheManager keyed by full chunk_id.

    Cache-backed sources allow arbitrary chunk bounds (no uniformity enforcement).
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Cache-backed sources are not discovered from filesystem.

        Returns None - these sources are created via DoPut, not discovery.
        """
        return None

    @classmethod
    def create_from_config(cls, source: 'SourceConfig') -> 'CachedSourceAdapter':
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

    def get_raw_chunk_endpoints(self) -> Iterator[ChunkEndpoint]:
        """Enumerate actually-written chunk positions.

        For cache-backed sources, only chunks that have been uploaded
        are enumerated, with their exact bounds. This differs from
        uniform-chunk sources which enumerate all theoretical grid positions.
        """
        for chunk_id, bounds in self._written_chunks.items():
            yield ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=bounds,
            )

    def get_chunk_array(self, chunk_id: bytes) -> np.ndarray:
        """Read chunk data from cache.

        For cache-backed sources, this is called by resolve_chunk_data's compute_fn
        only when the chunk is NOT in cache (missing or evicted). In that case,
        we raise an error since cache-backed sources have no fallback backend.

        Args:
            chunk_id: Encoded chunk identifier

        Raises:
            FlightServerError: Chunk not found or evicted
        """
        raise flight.FlightServerError(
            f"Chunk not found or evicted for source {self.source_id}. "
            f"Source no longer available."
        )

    def write_chunk(self, bounds: ChunkBounds, data: np.ndarray) -> None:
        """Write chunk data to cache.

        For cache-backed sources, arbitrary bounds allowed.

        Args:
            bounds: Chunk start/stop coordinates
            data: Numpy array with chunk data (any shape matching bounds)
        """
        cache_manager = CacheManager.get_instance()
        if cache_manager is None:
            raise RuntimeError("Cache not initialized")

        # Encode chunk_id from bounds start coords
        chunk_key = "/".join(str(s) for s in bounds.start)
        chunk_id = encode_chunk_id(self.source_id, chunk_key.encode('utf-8'))

        # Flatten data and create Arrow RecordBatch
        flat_data = data.ravel()
        batch = pa.RecordBatch.from_arrays([pa.array(flat_data)], ["data"])
        size_bytes = data.nbytes

        # Use start_compute + complete_entry pattern to directly write
        entry, is_owner = cache_manager.start_compute(chunk_id, metadata={
            'bounds': bounds.SerializeToString()
        })

        if is_owner:
            # We own the compute - complete it with our data
            cache_manager.complete_entry(chunk_id, batch, size_bytes)

        # Track the written chunk with its full bounds
        self._written_chunks[chunk_id] = bounds

        logger.debug(f"write_chunk: stored {size_bytes} bytes at {chunk_key}")