"""Downsampling for virtual chunk computation.

This module is self-contained (no protobuf / Arrow dependencies) and handles:
- Array padding for non-multiple-of-scale inputs
- Downsampling by nearest (strided) and area (mean-pool)
- Reduction method normalisation and dtype casting

The "precompute" method is normalised here but never computed: it signals that
a native on-disk pyramid level should be served (see adapters/ome_zarr.py).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default matches PyramidConfig.reduction_method so an unspecified-method
# request agrees with the advertised pyramid levels and what precache warms.
_DEFAULT_REDUCTION_METHOD = "area"
_SUPPORTED_REDUCTION_METHODS = {"nearest", "area", "precompute"}
_METHOD_ALIASES = {
    "stride": "nearest",
    "decimate": "nearest",
    "mean": "area",
    "precomputed": "precompute",
    # Deprecated: the linear interpolation method was removed; area is the
    # closest remaining averaging reduction.
    "linear": "area",
}


def normalize_reduction_method(method: str) -> str:
    normalized = (method or _DEFAULT_REDUCTION_METHOD).strip().lower()
    if normalized == "linear":
        logger.warning(
            "reduction_method 'linear' is deprecated and no longer supported; "
            "using 'area' instead"
        )
    normalized = _METHOD_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            f"Unsupported reduction method: {method}. "
            f"Supported methods: {sorted(_SUPPORTED_REDUCTION_METHODS)}"
        )
    return normalized


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _pad_shape_to_scale_multiple(
    shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(
        ceil_div(extent, scale) * scale
        for extent, scale in zip(shape, scale_hint, strict=True)
    )


def _pad_array_edge(
    data: np.ndarray,
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    if tuple(int(dim) for dim in data.shape) == tuple(int(dim) for dim in target_shape):
        return data

    if any(
        target < current
        for target, current in zip(target_shape, data.shape, strict=True)
    ):
        raise ValueError(
            f"Target shape {target_shape} must be >= data shape {data.shape}"
        )

    pad_width = [
        (0, int(target) - int(current))
        for current, target in zip(data.shape, target_shape, strict=True)
    ]
    if data.size == 0 or any(dim == 0 for dim in data.shape):
        return np.pad(data, pad_width, mode="constant")

    return np.pad(data, pad_width, mode="edge")


def _area_reduce(arr: np.ndarray, scale_hint: Tuple[int, ...]) -> np.ndarray:
    """Mean-pool arr by scale_hint along each axis in turn."""
    reduced = arr
    for axis in reversed(range(reduced.ndim)):
        scale = scale_hint[axis]
        axis_size = reduced.shape[axis]
        new_shape = (
            reduced.shape[:axis]
            + (axis_size // scale, scale)
            + reduced.shape[axis + 1 :]
        )
        reduced = reduced.reshape(new_shape).mean(axis=axis + 1)
    return reduced


def get_output_dtype(base_dtype: str, reduction_method: str) -> str:
    return np.dtype(base_dtype).str


def downsample_block(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    reduction_method = normalize_reduction_method(reduction_method)

    logger.debug(
        f"downsample_block: shape={data.shape}, scale={scale_hint}, method={reduction_method}"
    )

    original_dtype = data.dtype

    if reduction_method == "nearest":
        return data[tuple(slice(0, None, scale) for scale in scale_hint)]

    padded_shape = _pad_shape_to_scale_multiple(
        tuple(int(dim) for dim in data.shape), scale_hint
    )
    padded = _pad_array_edge(data, padded_shape)

    result = _area_reduce(np.asarray(padded, dtype=np.float64), scale_hint)

    # Cast back to original dtype with safe rounding for integers
    if original_dtype != result.dtype:
        if np.issubdtype(original_dtype, np.integer):
            # Round and clip to valid range before casting
            info = np.iinfo(original_dtype)
            result = np.clip(np.round(result), info.min, info.max)
        result = result.astype(original_dtype)

    return result
