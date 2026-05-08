"""Utilities for encoding and decoding chunk identifiers (chunk_id) used in Flight endpoints.

This module contains:
- ChunkEndpoint dataclass for chunk metadata
- Chunk ID encoding/decoding functions
- Virtual chunk encoding/decoding
- Chunk operations (splitting, slicing, intersection)
- Read plan helper functions
"""

import logging
import struct
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import SliceHint, TensorDescriptor, TensorReadOptions
from biopb.tensor.ticket_pb2 import ChunkBounds

logger = logging.getLogger(__name__)

# Constants
MAX_ARROW_BATCH_BYTES = 2 * 1024 * 1024 * 1024 - 1  # ~2GB
VIRTUAL_CHUNK_MAGIC = b'virt1'  # Magic prefix to identify virtual chunk payloads

# Keep legacy name for backward compatibility
_VIRTUAL_CHUNK_MAGIC = VIRTUAL_CHUNK_MAGIC

if TYPE_CHECKING:
    from biopb_tensor_server.base import BackendAdapter


@dataclass
class ChunkEndpoint:
    """A chunk with its metadata for Flight endpoint creation.

    Attributes:
        chunk_id: Backend-specific chunk identifier (bytes)
        bounds: Array coordinates (start, stop) for this chunk
    """
    chunk_id: bytes
    bounds: ChunkBounds


def encode_chunk_id(
    array_id: str, 
    backend_data: bytes,     
    split_index: int = 0,
    split_max: int = 1,
) -> bytes:
    """Encode array_id and backend-specific data into chunk_id.

    Format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - M bytes: backend_data
    - 2 bytes: split_index (uint16, big-endian)
    - 2 bytes: split_max (uint16, big-endian)

    Args:
        array_id: Tensor identifier
        backend_data: Backend-specific chunk data
        split_index: Index of this split (0-based)
        split_max: Total number of splits for this chunk

    Returns:
        Encoded chunk_id bytes
    """
    array_id_bytes = array_id.encode('utf-8')
    backend_data = backend_data + struct.pack('>H', split_index) + struct.pack('>H', split_max)
    return struct.pack('>I', len(array_id_bytes)) + array_id_bytes + backend_data


def decode_chunk_id(chunk_id: bytes) -> Tuple[str, bytes, int, int]:
    """Decode array_id and backend data from chunk_id.

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        Tuple of (array_id, backend_data, split_index, split_max)
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    array_id = chunk_id[4:4+array_id_len].decode('utf-8')
    
    data = chunk_id[4+array_id_len:]

    split_max = struct.unpack('>H', data[-2:])[0]
    split_index = struct.unpack('>H', data[-4:-2])[0]
    
    backend_data = data[:-4]

    return array_id, backend_data, split_index, split_max


def get_backend_data(chunk_id: bytes) -> bytes:
    """Extract backend-specific data from chunk_id."""
    _, backend_data, _, _ = decode_chunk_id(chunk_id)
    return backend_data


# =============================================================================
# Chunk Intersection and Bounds Helpers
# =============================================================================


def chunks_intersect(
    chunk_start: List[int],
    chunk_stop: List[int],
    slice_start: List[int],
    slice_stop: List[int],
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


def normalized_slice_bounds(
    shape: Tuple[int, ...],
    slice_hint: Optional[SliceHint],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Normalize slice bounds from slice_hint.

    Args:
        shape: Tensor shape
        slice_hint: Optional slice hint from request

    Returns:
        Tuple of (start, stop) coordinates

    Raises:
        ValueError: If slice hint dimensionality mismatch or invalid bounds
    """
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


def normalized_scale_hint(
    shape: Tuple[int, ...],
    read_options: Optional[TensorReadOptions],
) -> Optional[Tuple[int, ...]]:
    """Normalize scale hint from read_options.

    Args:
        shape: Tensor shape
        read_options: Optional read options from request

    Returns:
        Scale hint tuple if valid and non-trivial, None otherwise

    Raises:
        ValueError: If scale hint dimensionality mismatch or invalid values
    """
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


def logical_chunk_shape(
    chunk_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    logical_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Compute virtual chunk shape for scaled read.

    Args:
        chunk_shape: Base chunk shape
        scale_hint: Scale factors per axis
        logical_shape: Output shape at target scale

    Returns:
        Virtual chunk shape at target scale
    """
    virtual_chunk = []
    for chunk, scale, axis_shape in zip(chunk_shape, scale_hint, logical_shape):
        virtual_axis = lcm(chunk, scale) // scale
        virtual_chunk.append(min(max(virtual_axis, 1), axis_shape))
    return tuple(virtual_chunk)


# =============================================================================
# Chunk Splitting Helpers
# =============================================================================


def get_chunk_bounds_from_backend_key(
    adapter: "BackendAdapter",
    backend_key: bytes,
) -> ChunkBounds:
    """Get chunk bounds from backend key by finding matching endpoint.

    Args:
        adapter: Backend adapter for the tensor
        backend_key: Backend-specific chunk identifier

    Returns:
        ChunkBounds for the matching chunk

    Raises:
        ValueError: If no matching chunk found
    """
    for endpoint in adapter.get_raw_chunk_endpoints():
        endpoint_backend_data = get_backend_data(endpoint.chunk_id)
        if endpoint_backend_data == backend_key:
            return endpoint.bounds

    raise ValueError(f"Could not find chunk bounds for backend_key: {backend_key}")


def split_endpoint(
    array_id: str,
    ep: ChunkEndpoint,
    dtype: str,
) -> List[ChunkEndpoint]:
    """Split an oversized endpoint into sub-endpoints within size limit.

    Strategy: split along the largest axis for even division.
    Each sub-endpoint has:
    - chunk_id: parent payload with split_index/split_max encoded
    - bounds: sub-bounds within parent (logical coordinates)

    Args:
        array_id: Tensor identifier
        ep: ChunkEndpoint to split
        dtype: Data type string for size calculation

    Returns:
        List of sub-endpoints that each fit within the Arrow batch limit
    """
    parent_shape = tuple(stop - start for start, stop in zip(ep.bounds.start, ep.bounds.stop))
    item_size = np.dtype(dtype).itemsize

    parent_bytes = int(np.prod(parent_shape)) * item_size
    n_splits = int(np.ceil(parent_bytes / MAX_ARROW_BATCH_BYTES))

    split_axis = max(range(len(parent_shape)), key=lambda ax: parent_shape[ax])

    axis_size = parent_shape[split_axis]
    sub_axis_size = axis_size // n_splits

    parent_backend_data = get_backend_data(ep.chunk_id)

    sub_endpoints = []
    for i in range(n_splits):
        axis_start = ep.bounds.start[split_axis] + i * sub_axis_size
        axis_stop = min(ep.bounds.start[split_axis] + (i + 1) * sub_axis_size, ep.bounds.stop[split_axis])

        sub_start = list(ep.bounds.start)
        sub_stop = list(ep.bounds.stop)
        sub_start[split_axis] = axis_start
        sub_stop[split_axis] = axis_stop

        sub_bounds = ChunkBounds(start=sub_start, stop=sub_stop)
        sub_chunk_id = encode_chunk_id(array_id, parent_backend_data, i, n_splits)

        sub_endpoints.append(ChunkEndpoint(chunk_id=sub_chunk_id, bounds=sub_bounds))

    return sub_endpoints


def slice_array(
    array: np.ndarray,
    parent_bounds: ChunkBounds,
    split_index: int,
    split_max: int,
) -> np.ndarray:
    """Slice a numpy array along the largest axis based on split info.

    Args:
        array: Full chunk data as numpy array
        parent_bounds: Original bounds of the full chunk
        split_index: Which sub-chunk (0 to split_max-1)
        split_max: Total number of splits

    Returns:
        Smaller numpy array containing the sub-chunk data
    """
    parent_shape = tuple(stop - start for start, stop in zip(parent_bounds.start, parent_bounds.stop))

    split_axis = max(range(len(parent_shape)), key=lambda ax: parent_shape[ax])

    axis_size = parent_shape[split_axis]
    sub_axis_size = axis_size // split_max

    axis_start = split_index * sub_axis_size
    axis_stop = min((split_index + 1) * sub_axis_size, axis_size)

    slices = tuple(
        slice(0, parent_shape[ax]) if ax != split_axis
        else slice(axis_start, axis_stop)
        for ax in range(len(parent_shape))
    )

    return array[slices]


def slice_result(
    record_batch: pa.RecordBatch,
    parent_bounds: ChunkBounds,
    split_index: int,
    split_max: int,
    dtype: str,
) -> pa.RecordBatch:
    """Slice a RecordBatch along the largest axis based on split info.

    Args:
        record_batch: Full chunk data as RecordBatch
        parent_bounds: Original bounds of the full chunk
        split_index: Which sub-chunk (0 to split_max-1)
        split_max: Total number of splits
        dtype: Data type string for size calculation

    Returns:
        Smaller RecordBatch containing the sub-chunk data
    """
    parent_shape = tuple(stop - start for start, stop in zip(parent_bounds.start, parent_bounds.stop))
    parent_arr = record_batch_to_numpy(record_batch, parent_bounds, dtype)

    split_axis = max(range(len(parent_shape)), key=lambda ax: parent_shape[ax])

    axis_size = parent_shape[split_axis]
    sub_axis_size = axis_size // split_max

    axis_start = split_index * sub_axis_size
    axis_stop = min((split_index + 1) * sub_axis_size, axis_size)

    slices = tuple(
        slice(0, parent_shape[ax]) if ax != split_axis
        else slice(axis_start, axis_stop)
        for ax in range(len(parent_shape))
    )

    sub_arr = parent_arr[slices]
    array = pa.array(sub_arr.ravel())
    return pa.RecordBatch.from_arrays([array], ["data"])


def record_batch_to_numpy(
    record_batch: pa.RecordBatch,
    bounds: ChunkBounds,
    dtype: str,
) -> np.ndarray:
    """Convert RecordBatch to numpy array with proper shape.

    Args:
        record_batch: Arrow RecordBatch with single column
        bounds: Chunk bounds for shape calculation
        dtype: Data type string

    Returns:
        Numpy array with chunk shape
    """
    array = record_batch.column(0).to_numpy()
    chunk_shape = tuple(int(stop - start) for start, stop in zip(bounds.start, bounds.stop))
    return np.asarray(array, dtype=np.dtype(dtype)).reshape(chunk_shape)


# =============================================================================
# Virtual Chunk Helpers
# =============================================================================


def compute_virtual_chunk(adapter: "BackendAdapter", backend_data: bytes) -> np.ndarray:
    """Compute a virtual scaled chunk from source data.

    This is the internal computation logic for virtual chunk resolution.
    Returns the full virtual chunk before any splitting is applied.

    Args:
        adapter: Backend adapter for the tensor
        backend_data: Decoded virtual chunk payload

    Returns:
        Numpy array containing the computed chunk data
    """
    from biopb_tensor_server.downsample import (
        _cast_reduced_array,
        _downsample_block,
        _output_dtype,
        _pad_array_edge,
    )

    desc = adapter.get_tensor_descriptor()
    dtype = np.dtype(desc.dtype)

    source_start, source_stop, valid_stop, scale_hint, reduction_method = decode_virtual_chunk_payload(backend_data)

    logger.debug(
        f"compute_virtual_chunk: source_start={source_start}, source_stop={source_stop}, "
        f"scale={scale_hint}, method={reduction_method}"
    )

    source_slice = SliceHint(start=list(source_start), stop=list(valid_stop))
    endpoints = adapter.get_chunk_endpoints(source_slice)
    logger.debug(f"compute_virtual_chunk: found {len(endpoints)} source chunks")

    source_shape = tuple(int(stop - start) for start, stop in zip(source_start, source_stop))
    valid_shape = tuple(int(stop - start) for start, stop in zip(source_start, valid_stop))
    source_block = np.zeros(valid_shape, dtype=dtype)

    for endpoint in endpoints:
        # Use get_chunk_array directly - returns numpy array with proper shape
        chunk_data = adapter.get_chunk_array(endpoint.chunk_id)

        # Defensive reshape: backends may squeeze singleton dimensions (e.g., TIFF tiles)
        # but chunk bounds expect full-dimensional arrays. Restore expected shape.
        expected_chunk_shape = tuple(
            int(stop - start) for start, stop in zip(endpoint.bounds.start, endpoint.bounds.stop)
        )
        if chunk_data.shape != expected_chunk_shape:
            # Only reshape if sizes match (singleton dims were squeezed)
            if chunk_data.size == int(np.prod(expected_chunk_shape)):
                chunk_data = chunk_data.reshape(expected_chunk_shape)
            else:
                raise ValueError(
                    f"Chunk data size mismatch: got {chunk_data.size} elements "
                    f"but expected {int(np.prod(expected_chunk_shape))} for shape {expected_chunk_shape}"
                )

        overlap_start = [max(int(endpoint.bounds.start[axis]), source_start[axis]) for axis in range(len(valid_shape))]
        overlap_stop = [min(int(endpoint.bounds.stop[axis]), valid_stop[axis]) for axis in range(len(valid_shape))]
        source_slices = tuple(
            slice(overlap_start[axis] - int(endpoint.bounds.start[axis]), overlap_stop[axis] - int(endpoint.bounds.start[axis]))
            for axis in range(len(valid_shape))
        )
        target_slices = tuple(
            slice(overlap_start[axis] - source_start[axis], overlap_stop[axis] - source_start[axis])
            for axis in range(len(valid_shape))
        )
        source_block[target_slices] = chunk_data[source_slices]

    source_block = _pad_array_edge(source_block, source_shape)

    reduced = _downsample_block(
        source_block,
        scale_hint,
        reduction_method,
        merged_chunk_count=len(endpoints),
    )
    target_dtype = np.dtype(_output_dtype(desc.dtype, reduction_method))
    reduced = _cast_reduced_array(reduced, target_dtype)
    logger.debug(f"compute_virtual_chunk: output shape={reduced.shape}, dtype={target_dtype}")

    return reduced


def encode_virtual_chunk_payload(
    source_start: Tuple[int, ...],
    source_stop: Tuple[int, ...],
    valid_stop: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> bytes:
    """Encode virtual chunk payload with optional split info.

    Format:
    - 5 bytes: magic (_VIRTUAL_CHUNK_MAGIC)
    - 2 bytes: ndim (uint16, big-endian)
    - 8*ndim bytes: source_start (int64, big-endian)
    - 8*ndim bytes: source_stop (int64, big-endian)
    - 8*ndim bytes: valid_stop (int64, big-endian)
    - 8*ndim bytes: scale_hint (int64, big-endian)
    - 2 bytes: method length (uint16, big-endian)
    - N bytes: reduction_method (UTF-8)

    Unsplit chunks have split_index=0, split_max=1 (encoded at end).
    """
    def encode_int64_sequence(values: Tuple[int, ...]) -> bytes:
        return b''.join(struct.pack('>q', value) for value in values)

    method_bytes = reduction_method.encode('utf-8')
    ndim = len(source_start)
    return b''.join([
        _VIRTUAL_CHUNK_MAGIC,
        struct.pack('>H', ndim),
        encode_int64_sequence(source_start),
        encode_int64_sequence(source_stop),
        encode_int64_sequence(valid_stop),
        encode_int64_sequence(scale_hint),
        struct.pack('>H', len(method_bytes)),
        method_bytes,
    ])


def decode_virtual_chunk_payload(
    data: bytes,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str]:
    """Decode virtual chunk payload including split info.

    Args:
        data: Encoded payload bytes

    Returns:
        Tuple of (source_start, source_stop, valid_stop, scale_hint, reduction_method)

    Raises:
        ValueError: If data is not a virtual chunk payload
    """
    if not data.startswith(_VIRTUAL_CHUNK_MAGIC):
        raise ValueError('Chunk payload is not a virtual chunk')

    def decode_int64_sequence(data: bytes, offset: int, length: int) -> Tuple[Tuple[int, ...], int]:
        values = []
        for _ in range(length):
            values.append(struct.unpack('>q', data[offset:offset + 8])[0])
            offset += 8
        return tuple(values), offset

    offset = len(_VIRTUAL_CHUNK_MAGIC)
    ndim = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    source_start, offset = decode_int64_sequence(data, offset, ndim)
    source_stop, offset = decode_int64_sequence(data, offset, ndim)
    valid_stop, offset = decode_int64_sequence(data, offset, ndim)
    scale_hint, offset = decode_int64_sequence(data, offset, ndim)
    method_len = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    reduction_method = data[offset:offset + method_len].decode('utf-8')

    return source_start, source_stop, valid_stop, scale_hint, reduction_method


# =============================================================================
# Size Estimation Helpers
# =============================================================================


def estimate_chunk_bytes(shape: Tuple[int, ...], dtype: str) -> int:
    """Estimate chunk size in bytes from shape and dtype.

    Args:
        shape: Chunk shape
        dtype: Data type string

    Returns:
        Estimated size in bytes
    """
    num_elements = int(np.prod(shape, dtype=np.int64))
    return num_elements * np.dtype(dtype).itemsize


def needs_splitting(chunk_shape: Tuple[int, ...], dtype: str) -> bool:
    """Check if chunk exceeds Arrow batch limit.

    Args:
        chunk_shape: Chunk shape
        dtype: Data type string

    Returns:
        True if chunk needs splitting
    """
    return estimate_chunk_bytes(chunk_shape, dtype) > MAX_ARROW_BATCH_BYTES
