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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import SliceHint
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.downsample import ceil_div

logger = logging.getLogger(__name__)

# Constants
# 64MB threshold for chunk splitting - enables parallel Flight transfers
MAX_ARROW_BATCH_BYTES = 64 * 1024 * 1024

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
    scale_hint: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Normalize scale hint from request.

    Args:
        shape: Tensor shape
        scale_hint: Optional scale hint from request (repeated int64 field)

    Returns:
        Scale hint tuple if valid and non-trivial, None otherwise

    Raises:
        ValueError: If scale hint dimensionality mismatch or invalid values
    """
    if scale_hint is None or len(scale_hint) == 0:
        return None

    scale_hint_tuple = tuple(int(value) for value in scale_hint)
    if len(scale_hint_tuple) != len(shape):
        raise ValueError(
            f"Scale hint dimensionality mismatch: expected {len(shape)}, got {len(scale_hint_tuple)}"
        )

    for axis, scale in enumerate(scale_hint_tuple):
        if scale <= 0:
            raise ValueError(f"Scale hint must be positive on axis {axis}")

    if all(scale == 1 for scale in scale_hint_tuple):
        return None

    return scale_hint_tuple


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


# Defaults mirroring biopb-mcp's [pyramid] config (build_pyramid_levels). These
# decide the coarsest pyramid level the client requests on open; the precache
# worker must warm exactly that scale or its chunk_ids won't match. Keep in sync
# with biopb-mcp/src/biopb_mcp/_config.py if that is retuned.
PRECACHE_THRESHOLD = 4096
PRECACHE_DOWNSCALE_FACTOR = 4
PRECACHE_PIXEL_BUDGET_CUBIC_ROOT = 512


def _precache_xy_indices(shape: Sequence[int], dim_labels) -> Tuple[int, int]:
    """(y_idx, x_idx), matching biopb-mcp's get_xy_dim_indices.

    Prefers 'y'/'x' dim_labels; falls back to the ``[..., Y, X]`` convention
    (X last, Y second-to-last).
    """
    ndim = len(shape)
    if dim_labels:
        labels_lower = [str(label).lower() for label in dim_labels]
        try:
            return labels_lower.index("y"), labels_lower.index("x")
        except ValueError:
            pass
    if ndim < 2:
        raise ValueError(f"Cannot identify x/y dimensions: tensor is {ndim}-D")
    return ndim - 2, ndim - 1


def _precache_z_index(shape: Sequence[int], dim_labels) -> Optional[int]:
    """Index of the z axis or None, matching biopb-mcp's get_z_dim_index.

    Prefers a 'z' dim_label (absent label => no depth axis); else the positional
    ``[..., Z, Y, X]`` convention (third-from-last) for 3-D+ tensors.
    """
    ndim = len(shape)
    if dim_labels:
        labels_lower = [str(label).lower() for label in dim_labels]
        return labels_lower.index("z") if "z" in labels_lower else None
    return ndim - 3 if ndim >= 3 else None


def compute_precache_scale_hint(
    shape: Sequence[int],
    dim_labels=None,
    threshold: int = PRECACHE_THRESHOLD,
    downscale_factor: int = PRECACHE_DOWNSCALE_FACTOR,
    pixel_budget_cubic_root: int = PRECACHE_PIXEL_BUDGET_CUBIC_ROOT,
) -> List[int]:
    """Per-axis scale_hint for the *coarsest* pyramid level a client requests.

    This is a faithful port of biopb-mcp's ``build_pyramid_levels`` loop: X, Y
    and Z are downsampled individually (all other axes stay at 1), each stopping
    at ``axis_floor = min(pixel_budget_cubic_root, threshold)``, until the level
    satisfies ``Lx*Ly*Lz <= pixel_budget_cubic_root**3`` and ``Lx, Ly <=
    threshold``. ``ceil_div(L, s)`` is the server's own ``logical_shape``
    (base.py), so the resulting scale matches the client's terminal level and the
    warmed chunk_ids line up exactly.

    A tensor with no z axis is treated as ``Lz = 1`` and never gets a z factor.
    """
    ndim = len(shape)
    budget = pixel_budget_cubic_root ** 3
    floor = min(pixel_budget_cubic_root, threshold)

    y_idx, x_idx = _precache_xy_indices(shape, dim_labels)
    z_idx = _precache_z_index(shape, dim_labels)
    # A degenerate label set could map z onto an x/y axis; drop it if so.
    if z_idx is not None and z_idx in (x_idx, y_idx):
        z_idx = None

    sx = sy = sz = 1
    while True:
        lx = ceil_div(shape[x_idx], sx)
        ly = ceil_div(shape[y_idx], sy)
        lz = ceil_div(shape[z_idx], sz) if z_idx is not None else 1
        if lx * ly * lz <= budget and lx <= threshold and ly <= threshold:
            break
        nsx = sx * downscale_factor if lx > floor else sx
        nsy = sy * downscale_factor if ly > floor else sy
        nsz = sz * downscale_factor if (z_idx is not None and lz > floor) else sz
        if (nsx, nsy, nsz) == (sx, sy, sz):
            break  # nothing left to shrink; avoid an infinite loop
        sx, sy, sz = nsx, nsy, nsz

    scale = [1] * ndim
    scale[x_idx] = sx
    scale[y_idx] = sy
    if z_idx is not None:
        scale[z_idx] = sz
    return scale


def compute_safe_chunk_size(
    chunk_size: Tuple[int, ...],
    dtype: str,
    dim_labels: Optional[List[str]],
) -> Tuple[int, ...]:
    """Compute a chunk size that fits within Arrow batch limit.

    Uses hierarchical splitting: split along highest priority axis first,
    then next priority axis if still too large, etc.

    Args:
        chunk_size: Original chunk size tuple
        dtype: Data type string
        dim_labels: Optional dimension labels for semantic axis mapping

    Returns:
        Chunk size tuple guaranteed to fit within MAX_ARROW_BATCH_BYTES
    """
    item_size = np.dtype(dtype).itemsize
    chunk_bytes = int(np.prod(chunk_size)) * item_size

    if chunk_bytes <= MAX_ARROW_BATCH_BYTES:
        return chunk_size

    # Hierarchical splitting: iteratively reduce axes by priority
    safe_size = list(chunk_size)
    axes_already_split = set()  # Track axes we've already reduced

    while chunk_bytes > MAX_ARROW_BATCH_BYTES:
        # Calculate how many more splits we need
        n_splits_needed = int(np.ceil(chunk_bytes / MAX_ARROW_BATCH_BYTES))

        # Choose next axis to split (excluding already-split axes)
        split_axis = _choose_split_axis_excluding(
            tuple(safe_size), dim_labels, n_splits_needed, axes_already_split
        )

        if split_axis is None:
            # No more axes can be split - shouldn't happen if MAX_ARROW_BATCH_BYTES > 0
            logger.warning(
                f"Cannot split chunk further: size={safe_size}, "
                f"bytes={chunk_bytes}, target={MAX_ARROW_BATCH_BYTES}"
            )
            break

        # Calculate splits for this axis
        axis_size = safe_size[split_axis]
        # Number of splits on this axis (at least 2, at most axis_size)
        n_axis_splits = min(axis_size, max(2, n_splits_needed))

        # Reduce axis size
        safe_size[split_axis] = axis_size // n_axis_splits
        axes_already_split.add(split_axis)

        # Recalculate bytes
        chunk_bytes = int(np.prod(safe_size)) * item_size

    return tuple(safe_size)


def _choose_split_axis_excluding(
    shape: Tuple[int, ...],
    dim_labels: Optional[List[str]],
    n_splits: int,
    exclude_axes: Set[int],
) -> Optional[int]:
    """Choose axis for splitting, excluding already-split axes.

    Uses same priority as _choose_split_axis but skips excluded axes.

    Returns None if no eligible axis can accommodate n_splits.
    """
    SPATIAL_LABELS = {'y', 'x', 'z', 'c'}

    # Build label -> axis mapping
    label_to_axis: Dict[str, int] = {}
    if dim_labels:
        for ax, label in enumerate(dim_labels):
            label_to_axis[label.lower()] = ax

    # Eligible axes: not excluded and large enough for splits
    eligible = [ax for ax in range(len(shape))
                if ax not in exclude_axes and shape[ax] >= 2]

    if not eligible:
        return None

    # Priority 1: Non-spatial axes (t, v, frame, etc.)
    non_spatial = []
    if dim_labels:
        for ax in eligible:
            label = dim_labels[ax].lower()
            if label not in SPATIAL_LABELS:
                non_spatial.append(ax)
    else:
        non_spatial = eligible

    if non_spatial:
        return max(non_spatial, key=lambda ax: shape[ax])

    # Priority 2: 'c' (channel)
    if 'c' in label_to_axis:
        c_ax = label_to_axis['c']
        if c_ax in eligible:
            return c_ax

    # Priority 3: 'z' (depth)
    if 'z' in label_to_axis:
        z_ax = label_to_axis['z']
        if z_ax in eligible:
            return z_ax

    # Priority 4: Larger of 'y' or 'x'
    y_ax = label_to_axis.get('y')
    x_ax = label_to_axis.get('x')
    y_eligible = y_ax in eligible if y_ax else False
    x_eligible = x_ax in eligible if x_ax else False

    if y_eligible and x_eligible:
        return y_ax if shape[y_ax] >= shape[x_ax] else x_ax
    elif y_eligible:
        return y_ax
    elif x_eligible:
        return x_ax

    # Fallback: largest eligible axis
    return max(eligible, key=lambda ax: shape[ax])
