"""Zarr adapter for tensor storage."""

from typing import List, Optional
from functools import lru_cache

import numpy as np
import pyarrow as pa

from biopb_tensor_server.base import (
    BackendAdapter,
    ChunkEndpoint,
    _decode_chunk_id,
    _encode_chunk_id,
)
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint


class ZarrAdapter(BackendAdapter):
    """Adapter for Zarr/N5 chunked arrays.

    Chunk ID format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - M bytes: chunk key (UTF-8, e.g., "0/1/2")

    Uses LRU caching for decoded chunks.
    """

    def __init__(
        self,
        zarr_array,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256
    ):
        """Initialize Zarr adapter.

        Args:
            zarr_array: Zarr array object
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
            cache_size: Number of chunks to cache (default 256)
        """
        self.zarr_array = zarr_array
        self.array_id = array_id
        self.dim_labels = dim_labels or [f"dim{i}" for i in range(zarr_array.ndim)]
        self.cache_size = cache_size

        # Initialize LRU cache for decoded chunks
        self._get_chunk_data_cached = lru_cache(maxsize=cache_size)(self._get_chunk_data_uncached)

    def _get_chunk_data_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Read a chunk from zarr (uncached)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        chunk_key = backend_data.decode('utf-8')
        chunk_idx = tuple(int(i) for i in chunk_key.split('/'))
        chunks = self.zarr_array.chunks

        # Compute slice for this chunk
        slices = tuple(
            slice(idx * chunks[d], (idx + 1) * chunks[d])
            for d, idx in enumerate(chunk_idx)
        )

        return self.zarr_array[slices]

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        data = self._get_chunk_data_cached(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.zarr_array.shape),
            chunk_shape=list(self.zarr_array.chunks),
            dtype=self.zarr_array.dtype.str,
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        shape = self.zarr_array.shape
        chunks = self.zarr_array.chunks
        ndim = len(shape)

        endpoints = []

        def iter_chunk_indices(dim: int = 0, prefix: tuple = ()):
            if dim == ndim:
                yield prefix
            else:
                n_chunks = (shape[dim] + chunks[dim] - 1) // chunks[dim]
                for i in range(n_chunks):
                    yield from iter_chunk_indices(dim + 1, prefix + (i,))

        for chunk_idx in iter_chunk_indices():
            chunk_start = [idx * chunks[d] for d, idx in enumerate(chunk_idx)]
            chunk_stop = [
                min((idx + 1) * chunks[d], shape[d])
                for d, idx in enumerate(chunk_idx)
            ]

            # Filter by slice_hint
            if slice_hint is not None:
                # Check intersection
                intersects = True
                for d in range(ndim):
                    if chunk_stop[d] <= slice_hint.start[d] or chunk_start[d] >= slice_hint.stop[d]:
                        intersects = False
                        break
                if not intersects:
                    continue

            chunk_key = "/".join(str(i) for i in chunk_idx)
            chunk_id = _encode_chunk_id(self.array_id, chunk_key.encode('utf-8'))

            endpoints.append(ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            ))

        return endpoints