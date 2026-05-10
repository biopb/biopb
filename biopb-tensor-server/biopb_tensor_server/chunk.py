"""Utilities for encoding and decoding chunk identifiers (chunk_id) used in Flight endpoints.

This module contains:
- ChunkEndpoint dataclass for chunk metadata
- Chunk ID encoding/decoding functions
- Chunk operations (intersection)
- Read plan helper functions
"""

import logging
import struct
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import SliceHint, TensorReadOptions
from biopb.tensor.ticket_pb2 import ChunkBounds

logger = logging.getLogger(__name__)

# Constants
MAX_ARROW_BATCH_BYTES = 2 * 1024 * 1024 * 1024 - 1  # ~2GB

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
    bounds: "ChunkBounds",
) -> bytes:
    """Encode array_id and bounds into chunk_id.

    Format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - 2 bytes: ndim (uint16, big-endian)
    - 8*ndim bytes: bounds.start (int64, big-endian)
    - 8*ndim bytes: bounds.stop (int64, big-endian)

    Args:
        array_id: Tensor identifier
        bounds: Chunk bounds (start, stop coordinates)

    Returns:
        Encoded chunk_id bytes
    """
    array_id_bytes = array_id.encode('utf-8')
    ndim = len(bounds.start)

    parts = [
        struct.pack('>I', len(array_id_bytes)),
        array_id_bytes,
        struct.pack('>H', ndim),
    ]

    for val in bounds.start:
        parts.append(struct.pack('>q', int(val)))
    for val in bounds.stop:
        parts.append(struct.pack('>q', int(val)))

    return b''.join(parts)


def decode_chunk_id(chunk_id: bytes) -> Tuple[str, "ChunkBounds"]:
    """Decode array_id and bounds from chunk_id. Works for both regular 
    and virtual chunk_ids (ignores virtual payload).

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        Tuple of (array_id, bounds)
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    array_id = chunk_id[4:4+array_id_len].decode('utf-8')

    offset = 4 + array_id_len
    ndim = struct.unpack('>H', chunk_id[offset:offset+2])[0]
    offset += 2

    start = []
    for _ in range(ndim):
        start.append(struct.unpack('>q', chunk_id[offset:offset+8])[0])
        offset += 8

    stop = []
    for _ in range(ndim):
        stop.append(struct.unpack('>q', chunk_id[offset:offset+8])[0])
        offset += 8

    from biopb.tensor.ticket_pb2 import ChunkBounds
    bounds = ChunkBounds(start=start, stop=stop)

    return array_id, bounds


def get_bounds_from_chunk_id(chunk_id: bytes) -> "ChunkBounds":
    """Extract bounds from chunk_id."""
    _, bounds = decode_chunk_id(chunk_id)
    return bounds


# =============================================================================
# Slice and Scale Normalization
# =============================================================================


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


def _choose_split_axis(
    shape: Tuple[int, ...],
    dim_labels: Optional[List[str]],
    n_splits: int,
) -> int:
    """Choose axis for splitting with semantic priority.

    Priority order (preserves Y-X spatial plane for common visualization patterns):
    1. Any axis NOT in {y, x, z, c} — handles 't', 'v', 'frame', unlabeled, etc.
       If multiple candidates, pick largest.
    2. 'c' (channel)
    3. 'z' (depth)
    4. Larger of 'y' or 'x' (spatial plane)
    5. Fallback: largest axis (current behavior)

    Args:
        shape: Chunk shape tuple
        dim_labels: Optional dimension labels (may be None or partial)
        n_splits: Number of splits needed (axis must have size >= n_splits)

    Returns:
        Axis index to split along
    """
    SPATIAL_LABELS = {'y', 'x', 'z', 'c'}

    # Build label -> axis mapping (case-insensitive)
    label_to_axis: Dict[str, int] = {}
    if dim_labels:
        for ax, label in enumerate(dim_labels):
            label_to_axis[label.lower()] = ax

    # Priority 1: Non-spatial axes (t, v, frame, unlabeled, etc.)
    non_spatial_candidates: List[int] = []
    if dim_labels:
        for ax, label in enumerate(dim_labels):
            if label.lower() not in SPATIAL_LABELS and shape[ax] >= n_splits:
                non_spatial_candidates.append(ax)
    else:
        # No labels: treat all axes as non-spatial candidates, pick largest
        non_spatial_candidates = [ax for ax in range(len(shape)) if shape[ax] >= n_splits]

    if non_spatial_candidates:
        return max(non_spatial_candidates, key=lambda ax: shape[ax])

    # Priority 2: 'c' (channel)
    if 'c' in label_to_axis and shape[label_to_axis['c']] >= n_splits:
        return label_to_axis['c']

    # Priority 3: 'z' (depth)
    if 'z' in label_to_axis and shape[label_to_axis['z']] >= n_splits:
        return label_to_axis['z']

    # Priority 4: Larger of 'y' or 'x' (preserve spatial plane integrity)
    y_axis = label_to_axis.get('y')
    x_axis = label_to_axis.get('x')
    if y_axis is not None and x_axis is not None:
        if shape[y_axis] >= n_splits and shape[x_axis] >= n_splits:
            return y_axis if shape[y_axis] >= shape[x_axis] else x_axis
        elif shape[y_axis] >= n_splits:
            return y_axis
        elif shape[x_axis] >= n_splits:
            return x_axis

    # Fallback: largest axis (current behavior)
    return max(range(len(shape)), key=lambda ax: shape[ax])


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


# =============================================================================
# Scaled Chunk Encoding Helpers
# =============================================================================


def encode_chunk_id_with_scale(
    array_id: str,
    bounds: ChunkBounds,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> bytes:
    """Encode chunk_id with bounds and scale info appended.

    Format:
    - Standard bounds encoding (array_id + ndim + start + stop)
    - 8*ndim bytes: scale_hint (int64, big-endian)
    - 2 bytes: method length (uint16)
    - N bytes: method string

    Detection: if len(chunk_id) > bounds_end, it's a scaled chunk.

    Args:
        array_id: Tensor identifier
        bounds: Chunk bounds (start, stop coordinates)
        scale_hint: Scale factors per axis
        reduction_method: Reduction method string

    Returns:
        Encoded chunk_id bytes with scale info appended
    """
    base = encode_chunk_id(array_id, bounds)

    method_bytes = reduction_method.encode('utf-8')
    ndim = len(scale_hint)

    scale_payload = b''.join([
        b''.join(struct.pack('>q', s) for s in scale_hint),
        struct.pack('>H', len(method_bytes)),
        method_bytes,
    ])

    return base + scale_payload


def is_scaled_chunk(chunk_id: bytes) -> bool:
    """Check if chunk_id has scale info appended after bounds.

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        True if chunk_id contains scale info
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    offset = 4 + array_id_len
    ndim = struct.unpack('>H', chunk_id[offset:offset + 2])[0]
    bounds_end = offset + 2 + ndim * 8 + ndim * 8
    return len(chunk_id) > bounds_end


def decode_scale_info(chunk_id: bytes) -> Tuple[Tuple[int, ...], str]:
    """Decode scale_hint and reduction_method from scaled chunk_id.

    Args:
        chunk_id: Encoded chunk identifier with scale info

    Returns:
        Tuple of (scale_hint, reduction_method)
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    offset = 4 + array_id_len
    ndim = struct.unpack('>H', chunk_id[offset:offset + 2])[0]
    bounds_end = offset + 2 + ndim * 8 + ndim * 8

    # Decode scale_hint
    scale_hint = []
    for ax in range(ndim):
        scale_hint.append(struct.unpack('>q', chunk_id[bounds_end + ax*8:bounds_end + ax*8 + 8])[0])

    # Decode method
    method_offset = bounds_end + ndim * 8
    method_len = struct.unpack('>H', chunk_id[method_offset:method_offset + 2])[0]
    method = chunk_id[method_offset + 2:method_offset + 2 + method_len].decode('utf-8')

    return tuple(scale_hint), method
