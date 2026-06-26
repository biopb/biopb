"""Chunking utilities for dimensional iteration based on InputShapeHint."""

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

if TYPE_CHECKING:
    import biopb.image as proto
    import napari

logger = logging.getLogger(__name__)

FULL_ORDER = "TZCYX"


class IterationSpec(NamedTuple):
    iter_dims: set[
        str
    ]  # Axis names to iterate (e.g., {"T", "Z"}, empty set if no iteration)
    axis_order: str  # Full axis order for serialization (e.g., "TZYXC")


def _get_axis_mapping(image_layer: "napari.layers.Image", is_3d: bool) -> str:
    """Return axis order string respecting napari's interpretation.

    Uses layer.rgb, shape, and dtype to determine axis semantics.
    Handles all valid napari shapes (2-5 dims after multiscale handling).

    Args:
        image_layer: napari Image layer (uses .rgb, .multiscale, .data properties)
        is_3d: user's 3D checkbox setting (determines Z interpretation)

    Returns:
        Axis order string (e.g., "YX", "YXC", "ZYX", "TZYXC")

    Raises:
        ValueError: If ndim not in [2, 3, 4, 5] after multiscale handling
    """
    # Handle multiscale: use highest resolution level
    data = image_layer.data
    if image_layer.multiscale:
        data = data[0]
        logger.debug(
            f"Image is multiscale, using highest resolution level for axis mapping. Shape: {data.shape}"
        )

    ndim = data.ndim
    if ndim not in (2, 3, 4, 5):
        raise ValueError(f"Data must have 2-5 dimensions, got {ndim}")

    is_rgb = image_layer.rgb
    if ndim == 2:
        assert not is_rgb, "RGB/RGBA cannot be 2D arrays"
        channels = "YX"
    elif ndim == 3:
        channels = "YXC" if is_rgb else "ZYX" if is_3d else "TYX"
    elif ndim == 4:
        if is_rgb:
            channels = "ZYXC" if is_3d else "TYXC"
        else:
            channels = "TZYX" if is_3d else "TCYX"
    else:
        channels = "TZYXC" if is_rgb else "TZCYX"

    logger.debug(
        f"Determined axis order '{channels}' for data shape {data.shape} with rgb={is_rgb} and is_3d={is_3d}"
    )

    return channels


def _validate_data_shape(
    data: np.ndarray,
    axis_order: str,
    hint: Optional["proto.InputShapeHint"],
) -> None:
    """Validate data shape against hint, return warnings.

    Args:
        data: Image data array
        axis_order: Axis order string (e.g., "YXC")
        hint: InputShapeHint from op schema (may be None)
    """
    if hint is None:
        return

    for axis_name in hint.required_multivalue:
        axis_name = str(axis_name).upper()  # Ensure uppercase for comparison
        if axis_name not in axis_order:
            logger.warning(
                f"Service requires multivalue axis '{axis_name}' but it's not present in data"
            )
        elif data.shape[axis_order.index(axis_name)] == 1:
            logger.warning(
                f"Dimension '{axis_name}' has size 1 but service requires >1"
            )


def _get_iter_spec(
    axis_order: str,
    hint: Optional["proto.InputShapeHint"],
) -> IterationSpec:
    """Compute iteration dims and submitting order.

    Args:
        axis_order: Axis order string (e.g., "YXC")
        hint: InputShapeHint from op schema (may be None)

    Returns:
        IterationSpec with iter_dims and axis_order
    """
    submission_axes = set("ZYXC")

    # Step 1: Remove axes from expected_singletons (but never remove X or Y)
    if hint is not None and hint.expected_singletons:
        submission_axes -= set(hint.expected_singletons) - {"X", "Y"}

    # Step 2: Add required_multivalue axes to submission
    if hint is not None and hint.required_multivalue:
        submission_axes |= set(hint.required_multivalue)

    # Step 3: Remove any axes not present in axis_order
    submission_axes &= set(axis_order)

    logger.debug(f"_get_iter_spec: submission_axes={submission_axes}")

    # Step 4: Remaining axes are iter_dims
    iter_dims = set(axis_order) - submission_axes
    logger.debug(f"_get_iter_spec: iter_dims={iter_dims}")

    return IterationSpec(
        iter_dims=iter_dims,
        axis_order=axis_order,
    )


def _data_iterator(data: np.ndarray, iter_dims: set[str], axis_order: str):
    """Iterate over data slices for given axis names.

    Args:
        data: Image data array
        iter_dims: Set of axis names to iterate (e.g., {"T", "Z"}, empty set if none)
        axis_order: Full axis order string (e.g., "TZYXC")

    Yields:
        (position_dict, sliced_data) where position_dict maps numeric dim indices to values
    """
    if not iter_dims:
        yield {}, data
        return

    # Convert axis names to numeric indices
    iter_indices = [axis_order.index(ax) for ax in iter_dims]

    # Build shape for iteration dims only
    iter_shape = tuple(data.shape[idx] for idx in iter_indices)

    for idx_tuple in np.ndindex(iter_shape):
        position = dict(zip(iter_indices, idx_tuple, strict=False))

        # Build slices: iterate dims get single index, others get full slice
        slices = [slice(None)] * data.ndim
        for dim_idx, idx in position.items():
            slices[dim_idx] = slice(idx, idx + 1)

        yield position, data[tuple(slices)]


class ResultBuilder:
    """Lazy result buffer construction for 5D TZCYX output."""

    def __init__(self, iter_spec: IterationSpec, original_shape: tuple):
        self.iter_spec = iter_spec  # Store full iter_spec
        self.original_shape = original_shape  # Input shape (may be 2D-5D)
        self.buffer = None
        self.dtype = None

    def add_result(self, position: dict, chunk: np.ndarray):
        """Add result chunk to buffer.

        Args:
            position: Dict mapping numeric dim indices to index values
            chunk: Result chunk array (5D TZCYX from server)
        """
        logger.debug(f"Adding chunk with shape {chunk.shape} at position {position}")

        if self.buffer is None:
            output_shape = self._infer_output_shape(chunk)
            self.dtype = chunk.dtype
            self.buffer = np.zeros(output_shape, dtype=self.dtype)
            logger.debug(f"Created buffer with shape {output_shape}")

        placement_slices = self._build_placement_slices(position)
        self.buffer[placement_slices] = chunk

    def _infer_output_shape(self, chunk: np.ndarray) -> tuple:
        """Infer output buffer shape from first chunk (5D TZCYX).

        Three cases for each axis in TZCYX:
        1. Axis in iter_dims: use original input shape (number of tiles)
        2. Axis present in chunk: use chunk size (server output)
        3. Else: set to 1
        """
        output_shape = []
        for full_idx, axis_name in enumerate(FULL_ORDER):
            if axis_name in self.iter_spec.iter_dims:
                # Iterated axis - use original input size
                input_idx = self.iter_spec.axis_order.index(axis_name)
                output_shape.append(self.original_shape[input_idx])
            else:
                output_shape.append(chunk.shape[full_idx])

        return tuple(output_shape)

    def _build_placement_slices(self, position: dict) -> tuple:
        """Build slice tuple for placing chunk in 5D buffer.

        position maps numeric dim indices (in axis_order) to index values.
        """
        slices = [slice(None)] * 5  # 5D buffer
        for input_dim_idx, idx in position.items():
            axis_name = self.iter_spec.axis_order[input_dim_idx]
            full_idx = FULL_ORDER.index(axis_name)
            slices[full_idx] = slice(idx, idx + 1)
        return tuple(slices)

    def get_result(self) -> np.ndarray:
        """Return final assembled result."""
        return self.buffer
