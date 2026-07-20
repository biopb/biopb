"""Cache-backed source adapter for ephemeral uploaded data.

Design: One adapter instance per cache-backed source, following the existing
fused source+tensor adapter pattern used by OmeZarrAdapter, etc.

- CachedSourceAdapter instances registered in server.sources registry (same as other adapters)
- Metadata (shape, dtype, chunk_shape) stored in adapter instance
- Chunk data stored in CacheManager keyed by chunk_id
- When cache evicts chunks, adapter returns Flight error on read (source "gone")
- "Dead" adapters accumulate until server restart (acceptable for ephemeral sources)

Registration flow (bypasses discovery/registry):
DoPut → direct instantiation → server.sources registry

Chunk ID format: array_id + "/" + chunk_key
Chunk data: stored in CacheManager keyed by full chunk_id
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.base import (
    CHUNK_WIRE_SCHEMA,
    TensorAdapter,
)
from biopb_tensor_server.core.chunk import encode_chunk_id

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig

logger = logging.getLogger(__name__)


class CachedSourceAdapter(TensorAdapter):
    """Adapter for cache-backed uploaded sources.

    One instance per source, registered in server.sources registry.

    Chunk ID format: array_id + "/" + chunk_key (bounds start coords as "0/1/2")
    Chunk data stored in CacheManager keyed by full chunk_id.

    Cache-backed sources allow arbitrary chunk bounds (no uniformity enforcement).
    """

    _single_tensor_source = True

    @classmethod
    def create_from_config(
        cls, source: SourceConfig, credentials_config: Optional[Any] = None
    ) -> CachedSourceAdapter:
        """Cache-backed sources are not created from config.

        Raises NotImplementedError - use direct instantiation via DoPut.
        """
        raise NotImplementedError(
            "CachedSourceAdapter is created via DoPut, not config"
        )

    def __init__(
        self,
        source_id: str,
        shape: List[int],
        dtype: str,
        chunk_shape: List[int],
        dim_labels: Optional[List[str]] = None,
        ome_metadata: Optional[dict] = None,
        physical_scale: Optional[List[float]] = None,
        physical_unit: Optional[List[str]] = None,
    ):
        """Initialize cache-backed source adapter.

        Args:
            source_id: Unique source identifier (e.g., "cache_abc123")
            shape: Array shape
            dtype: Data type string (numpy format)
            chunk_shape: Nominal chunk size per dimension
            dim_labels: Optional dimension labels
            ome_metadata: Optional OME metadata dict
            physical_scale: Optional per-dimension physical pixel size, aligned
                1:1 with ``dim_labels`` (as the uploader sent it).
            physical_unit: Optional per-dimension unit string for
                ``physical_scale``, aligned 1:1 with ``dim_labels``.
        """
        self.source_id = source_id
        # Optional per-source capability token. When set, the Flight server
        # requires callers to present a matching Bearer token to read this
        # source (see TensorFlightServer._authorize_source). None = no per-source
        # gate (falls back to the server-wide token, if any).
        self._capability_token: Optional[str] = None
        self._shape = tuple(shape)
        self._dtype = dtype
        self._chunk_shape = tuple(chunk_shape)
        self._dim_labels = dim_labels or [f"dim{i}" for i in range(len(shape))]
        self._ome_metadata = ome_metadata or {}
        # Client-provided physical calibration, echoed verbatim on the wire. The
        # uploader already aligned these to dim_labels, so unlike the file
        # adapters there is nothing to parse or canonicalise -- storing and
        # surfacing them is exact (issue #272).
        self._physical_scale_vec = list(physical_scale) if physical_scale else []
        self._physical_unit_vec = list(physical_unit) if physical_unit else []

        # Track actually-written chunks: start coords -> full bounds
        self._written_chunks: Dict[bytes, ChunkBounds] = {}

        # Required fields for the adapter interface
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

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Echo the uploader's physical calibration onto the wire descriptor.

        Cache-backed sources carry whatever ``physical_scale`` / ``physical_unit``
        the DoPut request supplied, already aligned 1:1 with ``dim_labels`` (the
        client sent it that way), so there is nothing to parse or map -- return
        the stored vectors verbatim. Returns ``None`` when the upload carried no
        calibration, so the base clears the fields rather than advertising empty
        vectors. See ``TensorAdapter._physical_scale``.
        """
        if not self._physical_scale_vec or not self._physical_unit_vec:
            return None
        return list(self._physical_scale_vec), list(self._physical_unit_vec)

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds.

        Cache-backed sources have no backend data source - data is only
        available in cache and must be accessed via chunk_id.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Raises:
            FlightServerError: Cannot read data from cache-backed source
        """
        # Validate bounds via the base TensorAdapter.get_data contract.
        super().get_data(bounds)

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
        # Pass the flat element values (a primitive Arrow array), NOT a list<T>
        # wrapper -- write_chunk_arrow stores the raw value buffer directly.
        self.write_chunk_arrow(bounds, pa.array(data.ravel()), data.shape, data.dtype)

    def put_chunk(self, bounds, data, expected_shape, dtype) -> None:
        """Arbitrary-bounds write: cache-backed sources accept any bounds.

        Delegates straight to ``write_chunk_arrow`` (no NumPy round-trip -- the
        Arrow value buffer is stored as the unified binary chunk schema).
        """
        self.write_chunk_arrow(bounds, data, expected_shape, dtype)

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
            data: The chunk's flattened element *values* as a primitive Arrow
                array (e.g. ``int16``) -- NOT a ``list<T>`` wrapper. Its raw value
                buffer is stored directly as the unified binary chunk schema
                (biopb/biopb#293), the same schema ``resolve_chunk_data`` serves,
                so uploaded chunks read back byte-identically.
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
            # A list<T> wrapper's buffer[1] is offsets, not the value bytes --
            # storing it would silently corrupt the chunk. The binary schema takes
            # the flat values directly (biopb/biopb#293), so reject the wrapper.
            raise TypeError(
                "write_chunk_arrow expects the flat element values (a primitive "
                "array), not a list<T> wrapper"
            )

        logical_shape = list(logical_shape)
        dtype_str = np.dtype(dtype).str
        values_buf = data.buffers()[1]  # primitive array buffers: [validity, data]
        size_bytes = values_buf.size
        offsets = pa.py_buffer(np.array([0, size_bytes], dtype=np.int32))
        data_col = pa.Array.from_buffers(pa.binary(), 1, [None, offsets, values_buf])

        batch = pa.RecordBatch.from_arrays(
            [data_col, pa.array([logical_shape]), pa.array([dtype_str])],
            schema=CHUNK_WIRE_SCHEMA,
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

    def resolve_chunk_data(
        self,
        chunk_id: bytes,
        cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Resolve chunk data from cache manager.

        Cache-backed sources have no backend data - all data is stored in
        the cache manager via write_chunk/write_chunk_arrow.

        Args:
            chunk_id: Chunk identifier bytes
            cache_manager: CacheManager instance (required for cache-backed sources)

        Returns:
            RecordBatch with chunk data

        Raises:
            FlightServerError: If cache_manager is None or chunk not in cache
        """
        if cache_manager is None:
            raise flight.FlightServerError(
                f"CacheManager required for cache-backed source {self.source_id}"
            )

        # Check if chunk was written
        if chunk_id not in self._written_chunks:
            raise flight.FlightServerError(
                f"Chunk not found in cache-backed source {self.source_id}"
            )

        # Retrieve from cache manager directly
        entry = cache_manager.get_or_acquire(
            chunk_id,
            lambda: (_raise_no_backend(self.source_id), 0),  # Never actually called
            metadata={"array_id": self.source_id},
        )
        data = entry.data
        cache_manager.release(chunk_id)

        return data


def _raise_no_backend(source_id: str):
    """Helper function that raises when called."""
    raise flight.FlightServerError(
        f"Cannot compute data for cache-backed source {source_id}. "
        f"Data must be written via write_chunk first."
    )
