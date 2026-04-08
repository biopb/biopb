"""Backend adapters for tensor storage formats.

This module provides a consistent interface for reading chunked multi-dimensional
arrays from various storage backends (Zarr, HDF5, OME-TIFF, TileDB).

Each adapter maps storage-specific chunk layouts to Arrow Flight endpoints:
- chunk_id: Opaque bytes identifying a chunk in the backend
- ChunkBounds: Array coordinates (start, stop) for the chunk

The adapters integrate with Arrow Flight's GetFlightInfo/DoGet flow:
1. GetFlightInfo returns FlightEndpoints with chunk_id tickets
2. DoGet uses the chunk_id to fetch the actual data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from math import lcm
from typing import List, Optional, Tuple, Iterator
import struct
import os

import numpy as np
import pyarrow as pa

from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint, TensorReadOptions


@dataclass
class ChunkEndpoint:
    """A chunk with its metadata for Flight endpoint creation.

    Attributes:
        chunk_id: Backend-specific chunk identifier (bytes)
        bounds: Array coordinates (start, stop) for this chunk
    """
    chunk_id: bytes
    bounds: ChunkBounds


@dataclass
class TensorReadPlan:
    """Logical tensor read plan returned by the server planning layer."""

    descriptor: TensorDescriptor
    chunk_endpoints: List[ChunkEndpoint]


class BackendAdapter(ABC):
    """Abstract base class for tensor storage backend adapters.

    Each adapter handles a specific storage format (Zarr, HDF5, OME-TIFF, etc.)
    and provides methods to discover chunks and read chunk data.
    """

    @abstractmethod
    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the TensorDescriptor for this tensor.

        Returns:
            TensorDescriptor with array_id, shape, chunk_shape, dtype, dim_labels
        """
        pass

    @abstractmethod
    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        """Get chunk endpoints covering the tensor (or a slice of it).

        Args:
            slice_hint: Optional slice range. If provided, return only chunks
                       that intersect this range. If None, return all chunks.

        Returns:
            List of ChunkEndpoint objects with chunk_id and bounds
        """
        pass

    @abstractmethod
    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Read a chunk's data as an Arrow RecordBatch.

        Args:
            chunk_id: Backend-specific chunk identifier

        Returns:
            Arrow RecordBatch containing the chunk's data
        """
        pass

    def get_arrow_schema(self, desc: Optional[TensorDescriptor] = None) -> pa.Schema:
        """Get the Arrow schema for this tensor.

        Returns:
            Arrow Schema with tensor extension type
        """
        return build_arrow_schema(desc or self.get_tensor_descriptor())


def build_arrow_schema(desc: TensorDescriptor) -> pa.Schema:
    """Build an Arrow schema from a tensor descriptor."""
    dtype = np.dtype(desc.dtype)
    field = pa.field(
        "data",
        pa.from_numpy_dtype(dtype),
    )

    metadata = {
        "tensor_shape": ",".join(str(s) for s in desc.shape),
        "chunk_shape": ",".join(str(s) for s in desc.chunk_shape),
        "dim_labels": ",".join(desc.dim_labels) if desc.dim_labels else "",
    }

    return pa.schema([field], metadata=metadata)


def _chunks_intersect(
    chunk_start: List[int],
    chunk_stop: List[int],
    slice_start: List[int],
    slice_stop: List[int]
) -> bool:
    """Check if a chunk intersects with a slice range.

    Args:
        chunk_start: Chunk start coordinates
        chunk_stop: Chunk stop coordinates (exclusive)
        slice_start: Slice start coordinates
        slice_stop: Slice stop coordinates (exclusive)

    Returns:
        True if the chunk intersects the slice
    """
    for cs, ce, ss, se in zip(chunk_start, chunk_stop, slice_start, slice_stop):
        if ce <= ss or cs >= se:
            return False
    return True


def _encode_chunk_id(array_id: str, backend_data: bytes) -> bytes:
    """Encode array_id and backend-specific data into chunk_id.

    Format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - M bytes: backend_data

    Args:
        array_id: Tensor identifier
        backend_data: Backend-specific chunk data

    Returns:
        Encoded chunk_id bytes
    """
    array_id_bytes = array_id.encode('utf-8')
    return struct.pack('>I', len(array_id_bytes)) + array_id_bytes + backend_data


def _decode_chunk_id(chunk_id: bytes) -> Tuple[str, bytes]:
    """Decode array_id and backend data from chunk_id.

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        Tuple of (array_id, backend_data)
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    array_id = chunk_id[4:4+array_id_len].decode('utf-8')
    backend_data = chunk_id[4+array_id_len:]
    return array_id, backend_data


_VIRTUAL_CHUNK_MAGIC = b'virt1'
_DEFAULT_REDUCTION_METHOD = 'stride'
_SUPPORTED_REDUCTION_METHODS = {'stride', 'mean'}
_STRIDE_ALIASES = {'nearest', 'decimate'}


def _encode_int64_sequence(values: Tuple[int, ...]) -> bytes:
    return b''.join(struct.pack('>q', value) for value in values)


def _decode_int64_sequence(data: bytes, offset: int, length: int) -> Tuple[Tuple[int, ...], int]:
    values = []
    for _ in range(length):
        values.append(struct.unpack('>q', data[offset:offset + 8])[0])
        offset += 8
    return tuple(values), offset


def _normalize_reduction_method(method: str) -> str:
    normalized = (method or _DEFAULT_REDUCTION_METHOD).strip().lower()
    if normalized in _STRIDE_ALIASES:
        normalized = 'stride'
    if normalized not in _SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            f"Unsupported reduction method: {method}. "
            f"Supported methods: {sorted(_SUPPORTED_REDUCTION_METHODS)}"
        )
    return normalized


def _normalized_slice_bounds(
    shape: Tuple[int, ...],
    slice_hint: Optional[SliceHint],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if slice_hint is None:
        return tuple(0 for _ in shape), tuple(int(dim) for dim in shape)

    start = tuple(int(value) for value in slice_hint.start)
    stop = tuple(int(value) for value in slice_hint.stop)

    if len(start) != len(shape) or len(stop) != len(shape):
        raise ValueError(
            f"Slice hint dimensionality mismatch: expected {len(shape)}, "
            f"got start={len(start)} stop={len(stop)}"
        )

    for axis, (axis_start, axis_stop, axis_shape) in enumerate(zip(start, stop, shape)):
        if axis_start < 0 or axis_stop < 0:
            raise ValueError(f"Slice bounds must be non-negative on axis {axis}")
        if axis_start > axis_stop:
            raise ValueError(f"Slice start must be <= stop on axis {axis}")
        if axis_stop > axis_shape:
            raise ValueError(f"Slice stop exceeds tensor shape on axis {axis}")

    return start, stop


def _normalized_scale_hint(
    shape: Tuple[int, ...],
    read_options: Optional[TensorReadOptions],
) -> Optional[Tuple[int, ...]]:
    if read_options is None or len(read_options.scale_hint) == 0:
        return None

    scale_hint = tuple(int(value) for value in read_options.scale_hint)
    if len(scale_hint) != len(shape):
        raise ValueError(
            f"Scale hint dimensionality mismatch: expected {len(shape)}, got {len(scale_hint)}"
        )

    for axis, scale in enumerate(scale_hint):
        if scale <= 0:
            raise ValueError(f"Scale hint must be positive on axis {axis}")

    if all(scale == 1 for scale in scale_hint):
        return None

    return scale_hint


def _logical_chunk_shape(
    chunk_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    logical_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    virtual_chunk = []
    for axis, (chunk, scale, axis_shape) in enumerate(zip(chunk_shape, scale_hint, logical_shape)):
        if chunk % scale != 0 and scale % chunk != 0:
            raise NotImplementedError(
                "Scaling currently requires each chunk dimension and requested scale "
                f"to divide one another exactly; axis {axis} has chunk={chunk}, scale={scale}"
            )
        virtual_axis = lcm(chunk, scale) // scale
        virtual_chunk.append(min(max(virtual_axis, 1), axis_shape))
    return tuple(virtual_chunk)


def _output_dtype(base_dtype: str, reduction_method: str) -> str:
    if reduction_method == 'mean':
        return np.dtype(np.float64).str
    return np.dtype(base_dtype).str


def _encode_virtual_chunk_payload(
    source_start: Tuple[int, ...],
    source_stop: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> bytes:
    method_bytes = reduction_method.encode('utf-8')
    ndim = len(source_start)
    return b''.join([
        _VIRTUAL_CHUNK_MAGIC,
        struct.pack('>H', ndim),
        _encode_int64_sequence(source_start),
        _encode_int64_sequence(source_stop),
        _encode_int64_sequence(scale_hint),
        struct.pack('>H', len(method_bytes)),
        method_bytes,
    ])


def _decode_virtual_chunk_payload(data: bytes) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str]:
    if not data.startswith(_VIRTUAL_CHUNK_MAGIC):
        raise ValueError('Chunk payload is not a virtual chunk')

    offset = len(_VIRTUAL_CHUNK_MAGIC)
    ndim = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    source_start, offset = _decode_int64_sequence(data, offset, ndim)
    source_stop, offset = _decode_int64_sequence(data, offset, ndim)
    scale_hint, offset = _decode_int64_sequence(data, offset, ndim)
    method_len = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    reduction_method = data[offset:offset + method_len].decode('utf-8')
    return source_start, source_stop, scale_hint, reduction_method


def _record_batch_to_numpy(
    record_batch: pa.RecordBatch,
    bounds: ChunkBounds,
    dtype: str,
) -> np.ndarray:
    array = record_batch.column(0).to_numpy()
    chunk_shape = tuple(int(stop - start) for start, stop in zip(bounds.start, bounds.stop))
    return np.asarray(array, dtype=np.dtype(dtype)).reshape(chunk_shape)


def _downsample_block(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    reduction_method = _normalize_reduction_method(reduction_method)
    if reduction_method == 'stride':
        return data[tuple(slice(0, None, scale) for scale in scale_hint)]

    reduced = np.asarray(data, dtype=np.float64)
    for axis in reversed(range(reduced.ndim)):
        scale = scale_hint[axis]
        axis_size = reduced.shape[axis]
        if axis_size % scale != 0:
            raise ValueError(
                f"Mean downsampling requires divisibility on axis {axis}: "
                f"size={axis_size}, scale={scale}"
            )
        new_shape = (
            reduced.shape[:axis]
            + (axis_size // scale, scale)
            + reduced.shape[axis + 1:]
        )
        reduced = reduced.reshape(new_shape).mean(axis=axis + 1)
    return reduced


def _shift_bounds(
    bounds: ChunkBounds,
    logical_origin: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> ChunkBounds:
    start = [int((bounds.start[axis] - logical_origin[axis]) // scale_hint[axis]) for axis in range(len(bounds.start))]
    stop = [int((bounds.stop[axis] - logical_origin[axis]) // scale_hint[axis]) for axis in range(len(bounds.stop))]
    return ChunkBounds(start=start, stop=stop)


def plan_tensor_read(
    adapter: BackendAdapter,
    request_desc: TensorDescriptor,
) -> TensorReadPlan:
    """Plan a logical tensor read, including virtual scaled chunking."""
    base_desc = adapter.get_tensor_descriptor()
    base_shape = tuple(int(dim) for dim in base_desc.shape)
    base_chunk_shape = tuple(int(dim) for dim in base_desc.chunk_shape)
    slice_hint = request_desc.slice_hint if request_desc.HasField('slice_hint') else None
    read_options = request_desc.read_options if request_desc.HasField('read_options') else None
    source_start, source_stop = _normalized_slice_bounds(base_shape, slice_hint)
    source_shape = tuple(stop - start for start, stop in zip(source_start, source_stop))
    scale_hint = _normalized_scale_hint(base_shape, read_options)

    if scale_hint is None:
        chunk_endpoints = adapter.get_chunk_endpoints(
            SliceHint(start=list(source_start), stop=list(source_stop))
            if source_start != tuple(0 for _ in base_shape) or source_stop != base_shape
            else None
        )
        logical_endpoints = [
            ChunkEndpoint(
                chunk_id=endpoint.chunk_id,
                bounds=ChunkBounds(
                    start=[int(endpoint.bounds.start[axis] - source_start[axis]) for axis in range(len(base_shape))],
                    stop=[int(endpoint.bounds.stop[axis] - source_start[axis]) for axis in range(len(base_shape))],
                ),
            )
            for endpoint in chunk_endpoints
        ]
        logical_desc = TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=list(source_shape),
            chunk_shape=[min(int(chunk), int(size)) for chunk, size in zip(base_chunk_shape, source_shape)],
            dtype=base_desc.dtype,
        )
        if slice_hint is not None:
            logical_desc.slice_hint.CopyFrom(slice_hint)
        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)
        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)

    for axis, (extent, scale) in enumerate(zip(source_shape, scale_hint)):
        if extent % scale != 0:
            raise NotImplementedError(
                "Scaled reads currently require each requested extent to be divisible by the scale; "
                f"axis {axis} has extent={extent}, scale={scale}"
            )

    reduction_method = _normalize_reduction_method(read_options.reduction_method)
    logical_shape = tuple(extent // scale for extent, scale in zip(source_shape, scale_hint))
    logical_chunk_shape = _logical_chunk_shape(base_chunk_shape, scale_hint, logical_shape)

    endpoints: List[ChunkEndpoint] = []

    def iter_virtual_chunks(dim: int = 0, logical_offset: Tuple[int, ...] = ()):
        if dim == len(logical_shape):
            yield logical_offset
            return
        axis_chunk = logical_chunk_shape[dim]
        axis_extent = logical_shape[dim]
        for axis_start in range(0, axis_extent, axis_chunk):
            yield from iter_virtual_chunks(dim + 1, logical_offset + (axis_start,))

    for logical_start in iter_virtual_chunks():
        logical_stop = tuple(
            min(logical_start[axis] + logical_chunk_shape[axis], logical_shape[axis])
            for axis in range(len(logical_shape))
        )
        source_chunk_start = tuple(
            source_start[axis] + logical_start[axis] * scale_hint[axis]
            for axis in range(len(logical_shape))
        )
        source_chunk_stop = tuple(
            source_start[axis] + logical_stop[axis] * scale_hint[axis]
            for axis in range(len(logical_shape))
        )
        payload = _encode_virtual_chunk_payload(
            source_start=source_chunk_start,
            source_stop=source_chunk_stop,
            scale_hint=scale_hint,
            reduction_method=reduction_method,
        )
        endpoints.append(ChunkEndpoint(
            chunk_id=_encode_chunk_id(base_desc.array_id, payload),
            bounds=ChunkBounds(start=list(logical_start), stop=list(logical_stop)),
        ))

    logical_desc = TensorDescriptor(
        array_id=base_desc.array_id,
        dim_labels=base_desc.dim_labels,
        shape=list(logical_shape),
        chunk_shape=list(logical_chunk_shape),
        dtype=_output_dtype(base_desc.dtype, reduction_method),
    )
    if slice_hint is not None:
        logical_desc.slice_hint.CopyFrom(slice_hint)
    logical_desc.read_options.CopyFrom(read_options)
    return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=endpoints)


def resolve_chunk_data(adapter: BackendAdapter, chunk_id: bytes) -> pa.RecordBatch:
    """Resolve either a real backend chunk or a virtual scaled chunk."""
    array_id, backend_data = _decode_chunk_id(chunk_id)
    if not backend_data.startswith(_VIRTUAL_CHUNK_MAGIC):
        return adapter.get_chunk_data(chunk_id)

    desc = adapter.get_tensor_descriptor()
    dtype = np.dtype(desc.dtype)
    source_start, source_stop, scale_hint, reduction_method = _decode_virtual_chunk_payload(backend_data)
    source_slice = SliceHint(start=list(source_start), stop=list(source_stop))
    endpoints = adapter.get_chunk_endpoints(source_slice)
    source_shape = tuple(int(stop - start) for start, stop in zip(source_start, source_stop))
    source_block = np.zeros(source_shape, dtype=dtype)

    for endpoint in endpoints:
        chunk_data = _record_batch_to_numpy(adapter.get_chunk_data(endpoint.chunk_id), endpoint.bounds, desc.dtype)
        overlap_start = [max(int(endpoint.bounds.start[axis]), source_start[axis]) for axis in range(len(source_shape))]
        overlap_stop = [min(int(endpoint.bounds.stop[axis]), source_stop[axis]) for axis in range(len(source_shape))]
        source_slices = tuple(
            slice(overlap_start[axis] - int(endpoint.bounds.start[axis]), overlap_stop[axis] - int(endpoint.bounds.start[axis]))
            for axis in range(len(source_shape))
        )
        target_slices = tuple(
            slice(overlap_start[axis] - source_start[axis], overlap_stop[axis] - source_start[axis])
            for axis in range(len(source_shape))
        )
        source_block[target_slices] = chunk_data[source_slices]

    reduced = _downsample_block(source_block, scale_hint, reduction_method)
    reduced = np.asarray(reduced, dtype=np.dtype(_output_dtype(desc.dtype, reduction_method)))
    array = pa.array(reduced.ravel())
    return pa.RecordBatch.from_arrays([array], ["data"])


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

        # Iterate over all chunk indices
        def iter_chunk_indices(dim: int = 0, prefix: Tuple[int, ...] = ()):
            if dim == ndim:
                yield prefix
            else:
                n_chunks = (shape[dim] + chunks[dim] - 1) // chunks[dim]
                for i in range(n_chunks):
                    yield from iter_chunk_indices(dim + 1, prefix + (i,))

        for chunk_idx in iter_chunk_indices():
            # Compute chunk bounds
            chunk_start = [idx * chunks[d] for d, idx in enumerate(chunk_idx)]
            chunk_stop = [
                min((idx + 1) * chunks[d], shape[d])
                for d, idx in enumerate(chunk_idx)
            ]

            # Check intersection with slice hint
            if slice_hint is not None:
                if not _chunks_intersect(
                    chunk_start, chunk_stop,
                    list(slice_hint.start), list(slice_hint.stop)
                ):
                    continue

            # Generate chunk key
            chunk_key = "/".join(str(i) for i in chunk_idx)
            chunk_id = _encode_chunk_id(self.array_id, chunk_key.encode('utf-8'))

            endpoints.append(ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            ))

        return endpoints


def _encode_backend_coords(chunk_idx: Tuple[int, ...]) -> bytes:
    """Encode chunk coordinates to bytes (for HDF5)."""
    parts = [struct.pack('>H', len(chunk_idx))]
    for idx in chunk_idx:
        parts.append(struct.pack('>q', idx))  # int64
    return b''.join(parts)


def _decode_backend_coords(data: bytes) -> Tuple[int, ...]:
    """Decode chunk coordinates from bytes (for HDF5)."""
    ndim = struct.unpack('>H', data[:2])[0]
    indices = []
    offset = 2
    for _ in range(ndim):
        idx = struct.unpack('>q', data[offset:offset+8])[0]
        indices.append(idx)
        offset += 8
    return tuple(indices)


class Hdf5Adapter(BackendAdapter):
    """Adapter for HDF5 chunked datasets.

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - uint16 ndim
    - int64[ndim] chunk indices

    Uses LRU caching for decoded chunks.
    """

    def __init__(
        self,
        h5_dataset,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256
    ):
        """Initialize HDF5 adapter.

        Args:
            h5_dataset: h5py Dataset object
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
            cache_size: Number of chunks to cache (default 256)
        """
        self.h5_dataset = h5_dataset
        self.array_id = array_id
        self.dim_labels = dim_labels or [f"dim{i}" for i in range(len(h5_dataset.shape))]
        self.cache_size = cache_size

        # Initialize LRU cache for decoded chunks
        self._get_chunk_data_cached = lru_cache(maxsize=cache_size)(self._get_chunk_data_uncached)

    def _get_chunk_data_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Read a chunk from HDF5 (uncached)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        chunk_idx = _decode_backend_coords(backend_data)
        chunks = self.h5_dataset.chunks

        slices = tuple(
            slice(idx * chunks[d], min((idx + 1) * chunks[d], self.h5_dataset.shape[d]))
            for d, idx in enumerate(chunk_idx)
        )

        return self.h5_dataset[slices]

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        data = self._get_chunk_data_cached(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.h5_dataset.shape),
            chunk_shape=list(self.h5_dataset.chunks),
            dtype=self.h5_dataset.dtype.str,
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        shape = self.h5_dataset.shape
        chunks = self.h5_dataset.chunks
        ndim = len(shape)

        endpoints = []

        # Iterate over all chunk indices
        def iter_chunk_indices(dim: int = 0, prefix: Tuple[int, ...] = ()):
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

            if slice_hint is not None:
                if not _chunks_intersect(
                    chunk_start, chunk_stop,
                    list(slice_hint.start), list(slice_hint.stop)
                ):
                    continue

            chunk_id = _encode_chunk_id(self.array_id, _encode_backend_coords(chunk_idx))

            endpoints.append(ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            ))

        return endpoints


def _encode_ome_tile(ifd_index: int, tile_indices: Tuple[int, ...]) -> bytes:
    """Encode IFD index and tile indices to bytes (for OME-TIFF)."""
    parts = [
        struct.pack('>H', ifd_index),
        struct.pack('>H', len(tile_indices))
    ]
    for idx in tile_indices:
        parts.append(struct.pack('>q', idx))
    return b''.join(parts)


def _decode_ome_tile(data: bytes) -> Tuple[int, Tuple[int, ...]]:
    """Decode IFD index and tile indices from bytes (for OME-TIFF)."""
    ifd_index = struct.unpack('>H', data[:2])[0]
    ndim = struct.unpack('>H', data[2:4])[0]
    indices = []
    offset = 4
    for _ in range(ndim):
        idx = struct.unpack('>q', data[offset:offset+8])[0]
        indices.append(idx)
        offset += 8
    return ifd_index, tuple(indices)


def _encode_ome_multifile_tile(file_index: int, ifd_index: int, tile_indices: Tuple[int, ...]) -> bytes:
    """Encode file index, IFD index and tile indices for multi-file OME-TIFF."""
    parts = [
        struct.pack('>H', file_index),
        struct.pack('>H', ifd_index),
        struct.pack('>H', len(tile_indices))
    ]
    for idx in tile_indices:
        parts.append(struct.pack('>q', idx))
    return b''.join(parts)


def _decode_ome_multifile_tile(data: bytes) -> Tuple[int, int, Tuple[int, ...]]:
    """Decode file index, IFD index and tile indices for multi-file OME-TIFF."""
    file_index = struct.unpack('>H', data[:2])[0]
    ifd_index = struct.unpack('>H', data[2:4])[0]
    ndim = struct.unpack('>H', data[4:6])[0]
    indices = []
    offset = 6
    for _ in range(ndim):
        idx = struct.unpack('>q', data[offset:offset+8])[0]
        indices.append(idx)
        offset += 8
    return file_index, ifd_index, tuple(indices)


class OmeTiffAdapter(BackendAdapter):
    """Adapter for OME-TIFF files using tifffile.

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - uint16 ifd_index (page index)
    - uint16 ndim
    - int64[ndim] tile indices

    Uses LRU caching for decoded tiles to avoid repeated decompression.
    """

    def __init__(
        self,
        tiff_file,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256
    ):
        """Initialize OME-TIFF adapter.

        Args:
            tiff_file: tifffile.TiffFile object
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
            cache_size: Number of tiles to cache (default 256)
        """
        self.tiff_file = tiff_file
        self.array_id = array_id
        self.cache_size = cache_size

        # Get series info
        self.series = tiff_file.series[0]
        self.series_shape = self.series.shape
        self.series_dims = self.series.dims

        # Get tile info from first page
        first_page = tiff_file.pages[0]
        if not first_page.is_tiled:
            raise ValueError("OME-TIFF must be tiled")

        self.tile_width = first_page.tilewidth
        self.tile_length = first_page.tilelength
        self.tiles_per_row = (first_page.shape[1] + self.tile_width - 1) // self.tile_width
        self.tiles_per_col = (first_page.shape[0] + self.tile_length - 1) // self.tile_length

        # Derive chunk shape (tile_y, tile_x) for each plane
        self.chunk_shape = [self.tile_length, self.tile_width]

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            # Infer from series dims
            self.dim_labels = [d if isinstance(d, str) else str(d) for d in self.series_dims]

        # Full shape includes all planes
        # series_shape is (n_planes, height, width) or similar
        self.full_shape = list(self.series_shape)

        # Adjust chunk_shape to match full_shape dimensions
        # Non-spatial dimensions (like channel/time) have chunk size = 1
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self.chunk_shape

        # Initialize LRU cache for decoded tiles
        # Cache key is the chunk_id (bytes, which is hashable)
        self._get_decoded_tile = lru_cache(maxsize=cache_size)(self._get_decoded_tile_uncached)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        # Get dtype from first page
        first_page = self.tiff_file.pages[0]
        dtype = first_page.dtype

        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=str(dtype),
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        endpoints = []

        # Iterate over all IFDs (pages/planes)
        for ifd_index in range(len(self.tiff_file.pages)):
            page = self.tiff_file.pages[ifd_index]

            if not page.is_tiled:
                continue

            # Compute plane offset in full array
            # For XYZCT layout, each IFD corresponds to one (C, T, Z) combo
            # This depends on the specific OME-TIFF structure
            plane_offset = ifd_index

            # Iterate over tiles in this page
            for tile_row in range(self.tiles_per_col):
                for tile_col in range(self.tiles_per_row):
                    # Compute chunk bounds
                    y_start = tile_row * self.tile_length
                    y_stop = min((tile_row + 1) * self.tile_length, page.shape[0])
                    x_start = tile_col * self.tile_width
                    x_stop = min((tile_col + 1) * self.tile_width, page.shape[1])

                    # Full bounds including plane dimension
                    chunk_start = [plane_offset, y_start, x_start]
                    chunk_stop = [plane_offset + 1, y_stop, x_stop]

                    if slice_hint is not None:
                        if not _chunks_intersect(
                            chunk_start, chunk_stop,
                            list(slice_hint.start), list(slice_hint.stop)
                        ):
                            continue

                    chunk_id = _encode_chunk_id(self.array_id, _encode_ome_tile(ifd_index, (tile_row, tile_col)))

                    endpoints.append(ChunkEndpoint(
                        chunk_id=chunk_id,
                        bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
                    ))

        return endpoints

    def _get_decoded_tile_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Decode a tile from the TIFF file (uncached)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        ifd_index, tile_indices = _decode_ome_tile(backend_data)
        tile_row, tile_col = tile_indices

        page = self.tiff_file.pages[ifd_index]

        # Compute tile index
        tile_idx = tile_row * self.tiles_per_row + tile_col

        # Read tile using tifffile's low-level API
        offset = page.dataoffsets[tile_idx]
        bytecount = page.databytecounts[tile_idx]

        fh = self.tiff_file.filehandle
        fh.seek(offset)
        raw_data = fh.read(bytecount)

        # Decode the tile
        decoded = page.decode(raw_data, tile_idx)
        data = decoded[0].squeeze()  # Remove singleton dimensions

        # Ensure 2D output
        if data.ndim == 3:
            data = data[0]

        return data

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        # Use cached tile decoding
        data = self._get_decoded_tile(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])


class MultiFileOmeTiffAdapter(BackendAdapter):
    """Adapter for multi-file OME-TIFF datasets (e.g., Micro-Manager format).

    Handles datasets where multiple TIFF files form a single logical image:
    - sample/img_0.ome.tiff (channel 0)
    - sample/img_1.ome.tiff (channel 1)
    - sample/_metadata.txt (OME-XML metadata)

    Chunk ID format:
    - array_id prefix (via _encode_chunk_id)
    - uint16 file_index (which file in the series)
    - uint16 ifd_index (page index within file)
    - uint16 ndim
    - int64[ndim] tile indices

    Uses LRU caching for decoded tiles.
    """

    def __init__(
        self,
        directory: str,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256
    ):
        """Initialize multi-file OME-TIFF adapter.

        Args:
            directory: Path to directory containing multi-file dataset
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels
            cache_size: Number of tiles to cache (default 256)
        """
        import tifffile
        from pathlib import Path

        self.directory = Path(directory)
        self.array_id = array_id
        self.cache_size = cache_size

        # Find all OME-TIFF files in directory
        patterns = ["img_*.ome.tiff", "img_*.ome.tif", "*.ome.tiff", "*.ome.tif"]
        tiff_files = []
        for pattern in patterns:
            tiff_files.extend(sorted(self.directory.glob(pattern)))

        if not tiff_files:
            raise ValueError(f"No OME-TIFF files found in {directory}")

        # Open first file - tifffile auto-discovers related files via OME-XML
        self.tiff_file = tifffile.TiffFile(str(tiff_files[0]))

        # Get unified series info (spans all files)
        if len(self.tiff_file.series) == 0:
            raise ValueError("No series found in OME-TIFF dataset")

        self.series = self.tiff_file.series[0]
        self.series_shape = self.series.shape
        self.series_dims = self.series.dims

        # Get tile info from first page of first file
        first_page = self.tiff_file.pages[0]

        # Check if tiled - if not, we'll handle as single "tile" per plane
        if first_page.is_tiled:
            self.is_tiled = True
            self.tile_width = first_page.tilewidth
            self.tile_length = first_page.tilelength
            self.tiles_per_row = (first_page.shape[1] + self.tile_width - 1) // self.tile_width
            self.tiles_per_col = (first_page.shape[0] + self.tile_length - 1) // self.tile_length
            self.chunk_shape = [self.tile_length, self.tile_width]
        else:
            # Non-tiled: each IFD is a single chunk (the whole plane)
            self.is_tiled = False
            self.tile_width = first_page.shape[1]
            self.tile_length = first_page.shape[0]
            self.tiles_per_row = 1
            self.tiles_per_col = 1
            self.chunk_shape = [first_page.shape[0], first_page.shape[1]]

        # Dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        else:
            self.dim_labels = [d if isinstance(d, str) else str(d) for d in self.series_dims]

        # Full shape
        self.full_shape = list(self.series_shape)
        self.chunk_shape = [1] * (len(self.full_shape) - 2) + self.chunk_shape

        # Build file index for IFD access
        # Each file contains multiple IFDs (planes)
        # We need to map global IFD index to (file_index, local_ifd_index)
        self._file_ifd_map = []  # List of (file_path, local_ifd_count)
        self._ifd_to_file = []   # Maps global IFD index to (file_index, local_ifd_index)

        # For simple case: assume each file has same number of IFDs
        # More complex: parse OME-XML to understand C,Z,T distribution
        global_ifd_index = 0
        for file_path in tiff_files:
            with tifffile.TiffFile(str(file_path)) as tf:
                n_pages = len(tf.pages)
                self._file_ifd_map.append((file_path, n_pages))
                for local_idx in range(n_pages):
                    self._ifd_to_file.append((len(self._file_ifd_map) - 1, local_idx))
                global_ifd_index += n_pages

        self._total_ifds = global_ifd_index
        self._tiff_files = tiff_files

        # Initialize LRU cache
        self._get_decoded_tile = lru_cache(maxsize=cache_size)(self._get_decoded_tile_uncached)

    def get_tensor_descriptor(self) -> TensorDescriptor:
        # Get dtype from first page
        first_page = self.tiff_file.pages[0]
        dtype = first_page.dtype

        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=self.full_shape,
            chunk_shape=self.chunk_shape,
            dtype=str(dtype),
        )

    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        import tifffile

        endpoints = []

        # Iterate over all IFDs across all files
        for global_ifd_index in range(self._total_ifds):
            file_index, local_ifd_index = self._ifd_to_file[global_ifd_index]
            file_path = self._tiff_files[file_index]

            # Open file to get page info
            with tifffile.TiffFile(str(file_path)) as tf:
                page = tf.pages[local_ifd_index]

                # Use pre-computed tile counts (handles both tiled and non-tiled)
                tiles_per_row = self.tiles_per_row
                tiles_per_col = self.tiles_per_col

                for tile_row in range(tiles_per_col):
                    for tile_col in range(tiles_per_row):
                        y_start = tile_row * self.tile_length
                        y_stop = min((tile_row + 1) * self.tile_length, page.shape[0])
                        x_start = tile_col * self.tile_width
                        x_stop = min((tile_col + 1) * self.tile_width, page.shape[1])

                        chunk_start = [global_ifd_index, y_start, x_start]
                        chunk_stop = [global_ifd_index + 1, y_stop, x_stop]

                        if slice_hint is not None:
                            if not _chunks_intersect(
                                chunk_start, chunk_stop,
                                list(slice_hint.start), list(slice_hint.stop)
                            ):
                                continue

                        chunk_id = _encode_chunk_id(
                            self.array_id,
                            _encode_ome_multifile_tile(file_index, local_ifd_index, (tile_row, tile_col))
                        )

                        endpoints.append(ChunkEndpoint(
                            chunk_id=chunk_id,
                            bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
                        ))

        return endpoints

    def _get_decoded_tile_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Decode a tile from the multi-file dataset (uncached)."""
        import tifffile

        _, backend_data = _decode_chunk_id(chunk_id)
        file_index, local_ifd_index, tile_indices = _decode_ome_multifile_tile(backend_data)
        tile_row, tile_col = tile_indices

        file_path = self._tiff_files[file_index]

        with tifffile.TiffFile(str(file_path)) as tf:
            page = tf.pages[local_ifd_index]

            if self.is_tiled:
                # Tiled: read specific tile
                tiles_per_row = (page.shape[1] + self.tile_width - 1) // self.tile_width
                tile_idx = tile_row * tiles_per_row + tile_col

                offset = page.dataoffsets[tile_idx]
                bytecount = page.databytecounts[tile_idx]

                fh = tf.filehandle
                fh.seek(offset)
                raw_data = fh.read(bytecount)

                decoded = page.decode(raw_data, tile_idx)
                data = decoded[0].squeeze()
            else:
                # Non-tiled: read entire page as single "tile"
                data = page.asarray()

            if data.ndim == 3:
                data = data[0]

            return data

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        data = self._get_decoded_tile(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])


class OmeZarrAdapter(BackendAdapter):
    """Adapter for OME-Zarr (OME-NGFF) datasets.

    Extends ZarrAdapter with OME metadata support:
    - multiscales: Multiple resolution levels
    - axes: Dimension labels with types (channel, space, time)
    - coordinate_transformations: Physical scales
    - omero: Channel colors, names

    Chunk ID format: Same as ZarrAdapter
    - array_id prefix
    - chunk key (UTF-8, e.g., "0/1/2")
    """

    def __init__(
        self,
        zarr_array,
        array_id: str,
        dim_labels: Optional[List[str]] = None,
        cache_size: int = 256,
        resolution_level: int = 0
    ):
        """Initialize OME-Zarr adapter.

        Args:
            zarr_array: Zarr array object (from specific resolution level)
            array_id: Unique identifier for this tensor
            dim_labels: Optional dimension labels (overrides OME metadata)
            cache_size: Number of chunks to cache (default 256)
            resolution_level: Which resolution level to use (default 0)
        """
        import json
        from urllib.parse import urlparse

        self.zarr_array = zarr_array
        self.array_id = array_id
        self.cache_size = cache_size
        self.resolution_level = resolution_level

        # Try to read OME metadata from .zattrs
        self.ome_metadata = {}
        self.axes = []
        self.channel_names = []

        # Read .zattrs from the zarr group root
        try:
            store = zarr_array.store
            # Get the filesystem path from store
            store_str = str(store)
            if store_str.startswith('file://'):
                store_path = str(urlparse(store_str).path)
            elif hasattr(store, 'root'):
                store_path = str(store.root)
            else:
                store_path = store_str

            # .zattrs is at the group root level
            zattrs_path = os.path.join(store_path, '.zattrs')

            if os.path.exists(zattrs_path):
                with open(zattrs_path) as f:
                    zattrs = json.load(f)
                    self.ome_metadata = zattrs
                    if 'multiscales' in zattrs:
                        self.axes = zattrs['multiscales'][0].get('axes', [])
                        if 'omero' in zattrs:
                            channels = zattrs['omero'].get('channels', [])
                            self.channel_names = [ch.get('label', f'ch{i}') for i, ch in enumerate(channels)]
        except (json.JSONDecodeError, KeyError, FileNotFoundError, AttributeError):
            pass

        # Set dimension labels
        if dim_labels:
            self.dim_labels = dim_labels
        elif self.axes:
            self.dim_labels = [
                ax.get('name', f'dim{i}') if isinstance(ax, dict) else str(ax)
                for i, ax in enumerate(self.axes)
            ]
        else:
            self.dim_labels = [f"dim{i}" for i in range(zarr_array.ndim)]

        # Initialize LRU cache
        self._get_chunk_data_cached = lru_cache(maxsize=cache_size)(self._get_chunk_data_uncached)

    def _get_chunk_data_uncached(self, chunk_id: bytes) -> np.ndarray:
        """Read a chunk from zarr (uncached)."""
        _, backend_data = _decode_chunk_id(chunk_id)
        chunk_key = backend_data.decode('utf-8')
        chunk_idx = tuple(int(i) for i in chunk_key.split('/'))
        chunks = self.zarr_array.chunks

        slices = tuple(
            slice(idx * chunks[d], (idx + 1) * chunks[d])
            for d, idx in enumerate(chunk_idx)
        )

        return self.zarr_array[slices]

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

        def iter_chunk_indices(dim: int = 0, prefix: Tuple[int, ...] = ()):
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

            if slice_hint is not None:
                if not _chunks_intersect(
                    chunk_start, chunk_stop,
                    list(slice_hint.start), list(slice_hint.stop)
                ):
                    continue

            chunk_key = "/".join(str(i) for i in chunk_idx)
            chunk_id = _encode_chunk_id(self.array_id, chunk_key.encode('utf-8'))

            endpoints.append(ChunkEndpoint(
                chunk_id=chunk_id,
                bounds=ChunkBounds(start=chunk_start, stop=chunk_stop),
            ))

        return endpoints

    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        data = self._get_chunk_data_cached(chunk_id)
        arr = pa.array(data.ravel())
        return pa.RecordBatch.from_arrays([arr], ["data"])

    def get_ome_metadata(self) -> dict:
        """Return OME-Zarr metadata."""
        return self.ome_metadata

    def get_channel_info(self) -> List[dict]:
        """Return channel information from OME metadata."""
        if not self.channel_names:
            return [{'label': f'ch{i}'} for i in range(self.zarr_array.shape[0])]

        omero = self.ome_metadata.get('omero', {})
        channels = omero.get('channels', [])

        result = []
        for i, name in enumerate(self.channel_names):
            ch_info = {'label': name}
            if i < len(channels):
                ch_info.update(channels[i])
            result.append(ch_info)
        return result