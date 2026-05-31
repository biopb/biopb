"""Shared tensor utilities for biopb-mcp.

Functions for building pyramid levels and determining dimension indices,
used by both the tensor browser widget and the MCP server.
"""

import logging
from typing import List, Optional, Tuple

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


def build_layer_scale(
    client: TensorFlightClient,
    source_id: str,
    tensor_desc,
    source_desc=None,
) -> Tuple[Optional[List[float]], Optional[dict]]:
    """Build a napari ``scale`` vector from a source's OME pixel sizes.

    Reads ``client.get_source_metadata`` (an ``ome_types`` OME object) and maps
    ``physical_size_x/y/z`` onto the tensor's dimension axes, so areas/volumes
    the agent computes come out in physical units (e.g. µm²) instead of pixels.

    Axis order comes from ``tensor_desc.dim_labels``, falling back to the source
    descriptor's ``dim_labels`` (``source_desc``) when the per-tensor labels are
    empty, then to positional x/y.

    Returns:
        ``(scale, info)`` where *scale* is a per-axis list aligned to
        ``tensor_desc`` dims (``None`` if no physical sizes are available) and
        *info* is a small dict of the physical sizes + units for surfacing to
        the agent (``None`` if unavailable).
    """

    def _positive_float(value):
        """Coerce to a positive float, or None for missing/garbage values."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    try:
        metadata = client.get_source_metadata(source_id)

        images = getattr(metadata, "images", None)
        if not images:
            return None, None
        pixels = getattr(images[0], "pixels", None)
        if pixels is None:
            return None, None

        psize = {
            "x": _positive_float(getattr(pixels, "physical_size_x", None)),
            "y": _positive_float(getattr(pixels, "physical_size_y", None)),
            "z": _positive_float(getattr(pixels, "physical_size_z", None)),
        }
        if not any(psize.values()):
            return None, None

        ndim = len(tensor_desc.shape)
        dim_labels = tensor_desc.dim_labels or getattr(
            source_desc, "dim_labels", None
        )
        labels = [str(label).lower() for label in (dim_labels or [])]

        scale = [1.0] * ndim
        for axis, value in psize.items():
            if value and axis in labels:
                scale[labels.index(axis)] = value

        # Fall back to the conventional trailing (..., y, x) axes when the
        # descriptor carries no usable labels.
        if "x" not in labels and "y" not in labels and ndim >= 2:
            if psize["x"]:
                scale[ndim - 1] = psize["x"]
            if psize["y"]:
                scale[ndim - 2] = psize["y"]

        info = {
            "physical_size_x": psize["x"],
            "physical_size_y": psize["y"],
            "physical_size_z": psize["z"],
            "physical_size_x_unit": getattr(
                pixels, "physical_size_x_unit", None
            ),
            "physical_size_y_unit": getattr(
                pixels, "physical_size_y_unit", None
            ),
            "physical_size_z_unit": getattr(
                pixels, "physical_size_z_unit", None
            ),
        }
        return scale, info
    except Exception as exc:
        logger.warning("build_layer_scale failed for %s: %s", source_id, exc)
        return None, None
