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
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default matches PyramidConfig.reduction_method so an unspecified-method
# request agrees with the advertised pyramid levels and what precache warms.
_DEFAULT_REDUCTION_METHOD = "area"
# Public alias: since reduction_method left the chunk_id (biopb/biopb#178) it is
# advisory (the cache key never distinguished it, #76), so a cold compute that has
# no request in scope -- resolve_chunk_data on the do_get path -- downsamples with
# this default.
DEFAULT_REDUCTION_METHOD = _DEFAULT_REDUCTION_METHOD
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


def _blocked_shape(shape: Tuple[int, ...], axis: int, scale: int) -> Tuple[int, ...]:
    """shape with `axis` split into (axis_size // scale, scale)."""
    return shape[:axis] + (shape[axis] // scale, scale) + shape[axis + 1 :]


def _area_reduce(arr: np.ndarray, scale_hint: Tuple[int, ...]) -> np.ndarray:
    """Mean-pool arr by scale_hint along each axis in turn.

    Axes at scale 1 are skipped: a mean over blocks of one cannot change a
    value, but it still copies the array as it stands at that stage -- which is
    full resolution for any unit axis sitting after a reduced one (samples-last
    RGB, say).
    """
    reduced = arr
    for axis in reversed(range(reduced.ndim)):
        scale = scale_hint[axis]
        if scale == 1:
            continue
        new_shape = _blocked_shape(reduced.shape, axis, scale)
        reduced = reduced.reshape(new_shape).mean(axis=axis + 1)
    return reduced


def _area_reduce_integer(
    arr: np.ndarray,
    scale_hint: Tuple[int, ...],
    accumulator: np.dtype,
) -> np.ndarray:
    """Block-*sum* arr into `accumulator`, leaving the divide to the caller.

    Summing into a widened integer keeps the full-resolution pass off float64,
    so the array is already small by the time any float arithmetic happens.
    Only valid where :func:`_integer_accumulator` returns an accumulator.
    """
    reduced = arr
    for axis in reversed(range(reduced.ndim)):
        scale = scale_hint[axis]
        if scale == 1:
            continue
        new_shape = _blocked_shape(reduced.shape, axis, scale)
        reduced = reduced.reshape(new_shape).sum(axis=axis + 1, dtype=accumulator)
    return reduced


# Widest integer a float64 holds without loss. The block sum has to stay under
# it so the single closing divide is exact -- that exactness is what makes
# sum-then-divide agree bit for bit with the staged float64 means.
_FLOAT64_EXACT_INT_MAX = 2**53
_UNSIGNED_ACCUMULATORS = (np.uint32, np.uint64)
_SIGNED_ACCUMULATORS = (np.int32, np.int64)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _float_accumulator(dtype: np.dtype) -> np.dtype:
    """Width to mean-pool at on the float path: the input's own, but >= float32.

    Widening every input to float64 made float32 the *slowest* dtype here --
    slower than float64 at twice the bytes -- because the promotion doubles the
    full-resolution working set to produce the same picture.

    float16 is the deliberate exception to that rule, and widening it costs
    nothing: x86 has no native float16 arithmetic without AVX512-FP16, so numpy
    emulates it per element and loses vectorization. Reducing a 5032x5032 plane
    by 4x4 measured 159.3 ms at float16 against 97.6 ms at float32, at a max
    error against a float64 reference of 5.6e-2 against 5.4e-6 -- faster *and*
    ~1e4 more accurate, so there is nothing to trade off. numpy does not rescue
    a native reduction either: mean() on a float16 array returns float16, so it
    rounds to a 10-bit mantissa at every stage.

    Non-float inputs that reach this path (bool, and the integers
    :func:`_plan_integer_area` refused) keep float64, so their fallback is
    unchanged.

    Note this is *not* bit-identical to the old always-float64 behavior for
    float32 input -- the staged means now round at float32. That is the
    intended trade; integer inputs, which the equivalence suite pins bit for
    bit, do not take this path at all.
    """
    if np.issubdtype(dtype, np.floating):
        return np.result_type(dtype, np.float32)
    return np.dtype(np.float64)


def _integer_accumulator(dtype: np.dtype, block_size: int) -> Optional[np.dtype]:
    """Smallest integer accumulator that holds a whole block sum exactly.

    None means no candidate fits -- 64-bit inputs, or an implausibly large
    scale -- and the caller must stay on the float64 path. Sizing this from
    ``block_size * dtype_max`` rather than hardcoding uint32 is what keeps the
    sum from wrapping on wide inputs or deep reductions.
    """
    info = np.iinfo(dtype)
    bound = max(abs(int(info.min)), int(info.max)) * block_size
    if bound > _FLOAT64_EXACT_INT_MAX:
        return None
    candidates = _SIGNED_ACCUMULATORS if info.min < 0 else _UNSIGNED_ACCUMULATORS
    for candidate in candidates:
        if bound <= int(np.iinfo(candidate).max):
            return np.dtype(candidate)
    return None


def _plan_integer_area(
    dtype: np.dtype, scale_hint: Tuple[int, ...]
) -> Tuple[Optional[np.dtype], int]:
    """(accumulator, block_size) for the integer area path, or (None, 0).

    Two conditions, both needed for bit-identical output:

    - integer input, so summing is exact (bool and float fall through);
    - every scale a power of two, so that dividing once by the product equals
      the per-axis float64 means it replaces. A non-dyadic divisor (scale 3,
      say) rounds at each stage, and those roundings do not compose into the
      single divide.
    """
    if not np.issubdtype(dtype, np.integer):
        return None, 0
    if not all(_is_power_of_two(int(scale)) for scale in scale_hint):
        return None, 0
    block_size = 1
    for scale in scale_hint:
        block_size *= int(scale)
    return _integer_accumulator(dtype, block_size), block_size


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

    accumulator, block_size = _plan_integer_area(original_dtype, scale_hint)
    if accumulator is not None:
        # Sum at input-ish width, then divide/round/clip on the *reduced* array
        # -- kilobytes instead of the hundreds of megabytes a full-resolution
        # float64 promotion would touch.
        reduced = _area_reduce_integer(padded, scale_hint, accumulator)
        info = np.iinfo(original_dtype)
        result = np.clip(np.round(reduced / block_size), info.min, info.max)
        return result.astype(original_dtype)

    result = _area_reduce(
        np.asarray(padded, dtype=_float_accumulator(original_dtype)), scale_hint
    )

    # Cast back to original dtype with safe rounding for integers
    if original_dtype != result.dtype:
        if np.issubdtype(original_dtype, np.integer):
            # Round and clip to valid range before casting
            info = np.iinfo(original_dtype)
            result = np.clip(np.round(result), info.min, info.max)
        result = result.astype(original_dtype)

    return result
