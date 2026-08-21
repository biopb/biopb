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
from itertools import product
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# What an unspecified reduction_method resolves to. "nearest" is a strided pick:
# it never touches the bytes it discards, so it is the cheapest reduction at
# every scale and the only one that gets cheaper as the level gets coarser (at
# scale 32 it copies 1/1024 of the extent). It aliases where "area" averages,
# which is the trade being made -- a client that wants the averaged pixels asks
# for "area" explicitly.
#
_DEFAULT_REDUCTION_METHOD = "nearest"

# What an ABSENT method byte in a scaled chunk_id means (core/chunk.py). Not a
# policy default and not tied to the line above: it is a fact about bytes already
# written, and it is frozen.
#
# The encoder now carries a code byte for every computed method, so nothing this
# server mints is byte-free. A byte-free scaled chunk_id can therefore only come
# from before that change -- an old cache entry, an id a client still holds, or
# one a remote proxy forwarded from an older upstream -- and everything from
# before that change was area. Repointing this at the current request default
# would re-read all of them as the wrong method: same cache key, different
# pixels, no error.
CHUNK_ID_IMPLICIT_REDUCTION_METHOD = "area"
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


# Block size at or below which the strided-add kernel is chosen. The two
# kernels scale in opposite directions: reshape-sum gets cheaper as the block
# grows (longer contiguous inner sums), while the strided one costs a pass over
# the source per offset regardless. Measured on contiguous uint16 planes from
# 0.24 MiB to 256 MiB, strided wins 1.4-7.6x for blocks up to 256 and loses ~2x
# at 1024 -- which is the block the tensor browser's first, blank-screen read
# uses on a large scene, so the bound is not a formality. Block 512 is
# untested; 256 stays on the measured side of it.
_STRIDED_ADD_MAX_BLOCK = 256


def _area_reduce_strided(
    padded: np.ndarray,
    scale_hint: Tuple[int, ...],
    accumulator: np.dtype,
) -> np.ndarray:
    """Block-sum `padded` by strided adds. Same contract as _area_reduce_integer.

    `accumulator` MUST be the one :func:`_plan_integer_area` sized, never a
    hardcoded uint32: it is chosen from block_size * dtype_max, so a uint16
    input reduces into uint32 but a uint32 input needs uint64. Hardcoding the
    width wraps silently -- the sum of block_size elements is exact in the
    sized accumulator and garbage in a narrow one, with no error either way.

    `padded` must already be a multiple of the scale on every axis
    (:func:`_pad_array_edge` ran first), which is what makes the strided slices
    tile it exactly. Do not substitute a trim: the pad is edge-replicated and
    divided by the FULL block size, so trimming changes the values at a tensor
    boundary.
    """
    acc = None
    for offsets in product(*(range(scale) for scale in scale_hint)):
        piece = padded[
            tuple(
                slice(offset, None, scale)
                for offset, scale in zip(offsets, scale_hint, strict=True)
            )
        ]
        if acc is None:
            acc = piece.astype(accumulator)  # widens once, on the first term
        else:
            np.add(acc, piece, out=acc)  # every later term adds in place
    return acc


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


def streaming_area_plan(
    dtype: np.dtype, scale_hint: Tuple[int, ...]
) -> Tuple[Optional[np.dtype], int]:
    """The area plan :func:`downsample_block` would use, for a streamed reduce.

    Public because composing a scaled chunk out of full-resolution chunks
    (``core/compose.py``) has to reproduce this function's output bit for bit,
    and can only do so on the integer path: block sums are exact and order
    independent, so they may be accumulated chunk by chunk in any order. The
    staged float means are neither, so a composer that gets ``None`` here has to
    read the whole extent and call :func:`downsample_block` instead.

    Exported rather than reimplemented so the two paths cannot drift on which
    inputs qualify.
    """
    return _plan_integer_area(dtype, scale_hint)


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
        # float64 promotion would touch. Both kernels compute the same exact
        # integer sum; only the traversal order differs (see
        # _STRIDED_ADD_MAX_BLOCK).
        if block_size <= _STRIDED_ADD_MAX_BLOCK:
            reduced = _area_reduce_strided(padded, scale_hint, accumulator)
        else:
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
