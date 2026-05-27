"""Shared tensor utilities for napari-biopb.

Functions for building pyramid levels and determining dimension indices,
used by both the tensor browser widget and the MCP server.
"""

import logging
from typing import List, Tuple

from biopb.tensor import TensorFlightClient

logger = logging.getLogger(__name__)

PYRAMID_THRESHOLD = 4096


def get_xy_dim_indices(tensor_desc) -> Tuple[int, int]:
    """Get indices of x and y dimensions from tensor descriptor.

    Uses dim_labels as primary source (looks for 'x', 'y').
    Falls back to last two dimensions if dim_labels not available.

    Returns:
        Tuple of (y_index, x_index) - y first for row/col convention
    """
    ndim = len(tensor_desc.shape)

    if tensor_desc.dim_labels:
        labels_lower = [l.lower() for l in tensor_desc.dim_labels]
        try:
            x_idx = labels_lower.index("x")
            y_idx = labels_lower.index("y")
            return (y_idx, x_idx)
        except ValueError:
            pass

    if ndim >= 2:
        return (ndim - 1, ndim - 2)

    return (0, 1) if ndim == 2 else (0, 0)


def build_pyramid_levels(
    client: TensorFlightClient,
    source_id: str,
    tensor_id: str,
    tensor_desc,
) -> List:
    """Build pyramid levels for large x-y datasets.

    Returns:
        List of dask arrays at different resolution levels (pyramid)
    """
    shape = tensor_desc.shape
    ndim = len(shape)

    y_idx, x_idx = get_xy_dim_indices(tensor_desc)

    x_size = shape[x_idx]
    y_size = shape[y_idx]

    if x_size <= PYRAMID_THRESHOLD and y_size <= PYRAMID_THRESHOLD:
        return [client.get_tensor(source_id, tensor_id)]

    levels = []
    scale = 1
    min_size = 256

    while True:
        scale_hint = [1] * ndim
        scale_hint[y_idx] = scale
        scale_hint[x_idx] = scale

        arr = client.get_tensor(source_id, tensor_id, scale_hint=scale_hint)
        levels.append(arr)

        scaled_x = x_size // scale
        scaled_y = y_size // scale
        if scaled_x < min_size or scaled_y < min_size:
            break

        scale *= 2

    return levels
