"""Downsampling and compute backend selection for virtual chunk computation.

This module is self-contained (no protobuf / Arrow dependencies) and handles:
- Compute backend selection (CPU vs GPU via CuPy)
- Array padding for non-multiple-of-scale inputs
- Downsampling by nearest, area (mean-pool), and linear interpolation
- Reduction method normalisation and dtype casting
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    cp = importlib.import_module("cupy")
    cupy_ndimage = importlib.import_module("cupyx.scipy.ndimage")
    _HAS_CUPY = True
    logger.debug("CuPy available: GPU backend enabled")
except ImportError:
    cp = None
    cupy_ndimage = None
    _HAS_CUPY = False


@dataclass
class ComputeBackendOptions:
    """Server-side compute backend selection options."""

    force_backend: str = "auto"
    gpu_min_input_bytes: int = 4 * 1024 * 1024
    gpu_min_linear_input_bytes: int = 2 * 1024 * 1024
    gpu_memory_safety_factor: int = 4
    gpu_min_merged_chunks: int = 4


_DEFAULT_REDUCTION_METHOD = "nearest"
_SUPPORTED_REDUCTION_METHODS = {"nearest", "area", "linear", "precompute"}
_METHOD_ALIASES = {
    "stride": "nearest",
    "decimate": "nearest",
    "mean": "area",
    "precomputed": "precompute",
}
_GPU_PREFERRED_METHODS = {"area", "linear"}
_COMPUTE_BACKEND_OPTIONS = ComputeBackendOptions()


def configure_compute_backend(**kwargs) -> ComputeBackendOptions:
    """Update server-side compute backend selection options."""
    global _COMPUTE_BACKEND_OPTIONS

    options = ComputeBackendOptions(
        **{
            **_COMPUTE_BACKEND_OPTIONS.__dict__,
            **kwargs,
        }
    )
    if options.force_backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("force_backend must be one of: auto, cpu, gpu")
    if options.gpu_min_input_bytes < 0:
        raise ValueError("gpu_min_input_bytes must be non-negative")
    if options.gpu_min_linear_input_bytes < 0:
        raise ValueError("gpu_min_linear_input_bytes must be non-negative")
    if options.gpu_memory_safety_factor < 1:
        raise ValueError("gpu_memory_safety_factor must be >= 1")
    if options.gpu_min_merged_chunks < 1:
        raise ValueError("gpu_min_merged_chunks must be >= 1")

    _COMPUTE_BACKEND_OPTIONS = options
    logger.debug(
        f"Compute backend configured: force_backend={options.force_backend}, "
        f"gpu_min_input_bytes={options.gpu_min_input_bytes}, "
        f"gpu_min_linear_input_bytes={options.gpu_min_linear_input_bytes}, "
        f"gpu_memory_safety_factor={options.gpu_memory_safety_factor}, "
        f"gpu_min_merged_chunks={options.gpu_min_merged_chunks}"
    )
    return _COMPUTE_BACKEND_OPTIONS


def get_compute_backend_options() -> ComputeBackendOptions:
    """Return current server-side compute backend selection options."""
    return ComputeBackendOptions(**_COMPUTE_BACKEND_OPTIONS.__dict__)


def normalize_reduction_method(method: str) -> str:
    normalized = (method or _DEFAULT_REDUCTION_METHOD).strip().lower()
    normalized = _METHOD_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            f"Unsupported reduction method: {method}. "
            f"Supported methods: {sorted(_SUPPORTED_REDUCTION_METHODS)}"
        )
    return normalized


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _logical_shape_for_scale(
    source_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(
        ceil_div(extent, scale) for extent, scale in zip(source_shape, scale_hint)
    )


def _pad_shape_to_scale_multiple(
    shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(
        ceil_div(extent, scale) * scale for extent, scale in zip(shape, scale_hint)
    )


def _pad_array_edge(
    data: np.ndarray,
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    if tuple(int(dim) for dim in data.shape) == tuple(int(dim) for dim in target_shape):
        return data

    if any(target < current for target, current in zip(target_shape, data.shape)):
        raise ValueError(
            f"Target shape {target_shape} must be >= data shape {data.shape}"
        )

    pad_width = [
        (0, int(target) - int(current))
        for current, target in zip(data.shape, target_shape)
    ]
    if data.size == 0 or any(dim == 0 for dim in data.shape):
        return np.pad(data, pad_width, mode="constant")

    return np.pad(data, pad_width, mode="edge")


def _pad_array_edge_gpu(
    data,
    target_shape: Tuple[int, ...],
):
    if tuple(int(dim) for dim in data.shape) == tuple(int(dim) for dim in target_shape):
        return data

    if any(target < current for target, current in zip(target_shape, data.shape)):
        raise ValueError(
            f"Target shape {target_shape} must be >= data shape {data.shape}"
        )

    pad_width = [
        (0, int(target) - int(current))
        for current, target in zip(data.shape, target_shape)
    ]
    mode = (
        "constant" if data.size == 0 or any(dim == 0 for dim in data.shape) else "edge"
    )
    return cp.pad(data, pad_width, mode=mode)


def _area_reduce(arr, scale_hint: Tuple[int, ...]):
    """Mean-pool arr by scale_hint along each axis in turn.

    Works with both numpy and cupy arrays since the operations are identical.
    """
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


def _get_backend_override() -> str:
    override = (
        os.environ.get(
            "BIOPB_TENSOR_FORCE_BACKEND", _COMPUTE_BACKEND_OPTIONS.force_backend
        )
        .strip()
        .lower()
    )
    if override in {"cpu", "gpu", "auto"}:
        return override
    return "auto"


def _get_gpu_free_bytes() -> int:
    if not _HAS_CUPY:
        return 0
    try:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        return int(free_bytes)
    except Exception:
        return 0


def _estimate_array_bytes(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def _select_compute_backend(
    source_shape: Tuple[int, ...],
    dtype: np.dtype,
    reduction_method: str,
    scale_hint: Tuple[int, ...],
    merged_chunk_count: int,
) -> str:
    reduction_method = normalize_reduction_method(reduction_method)
    override = _get_backend_override()

    if override == "cpu":
        logger.debug("Backend forced to CPU via override")
        return "cpu"

    if override == "gpu":
        if _HAS_CUPY and (reduction_method != "linear" or cupy_ndimage is not None):
            logger.debug("Backend forced to GPU via override")
            return "gpu"
        logger.debug("Backend forced to GPU but CuPy unavailable, falling back to CPU")
        return "cpu"

    if not _HAS_CUPY:
        logger.debug("Backend auto: CuPy unavailable, using CPU")
        return "cpu"

    if reduction_method not in _GPU_PREFERRED_METHODS:
        logger.debug(
            f"Backend auto: method '{reduction_method}' not GPU-preferred, using CPU"
        )
        return "cpu"

    if reduction_method == "linear" and cupy_ndimage is None:
        logger.debug(
            "Backend auto: linear method requires cupyx.scipy.ndimage, using CPU"
        )
        return "cpu"

    input_bytes = _estimate_array_bytes(source_shape, dtype)
    output_shape = _logical_shape_for_scale(source_shape, scale_hint)
    output_bytes = _estimate_array_bytes(output_shape, dtype)
    total_bytes = input_bytes + output_bytes
    min_input_bytes = (
        _COMPUTE_BACKEND_OPTIONS.gpu_min_linear_input_bytes
        if reduction_method == "linear"
        else _COMPUTE_BACKEND_OPTIONS.gpu_min_input_bytes
    )

    if input_bytes < min_input_bytes:
        logger.debug(
            f"Backend auto: input {input_bytes}B < threshold {min_input_bytes}B, using CPU"
        )
        return "cpu"

    if (
        merged_chunk_count < _COMPUTE_BACKEND_OPTIONS.gpu_min_merged_chunks
        and input_bytes < (min_input_bytes * 2)
    ):
        logger.debug(
            f"Backend auto: merged_chunks {merged_chunk_count} < threshold, input small, using CPU"
        )
        return "cpu"

    free_bytes = _get_gpu_free_bytes()
    if (
        free_bytes > 0
        and free_bytes < total_bytes * _COMPUTE_BACKEND_OPTIONS.gpu_memory_safety_factor
    ):
        logger.debug(
            f"Backend auto: GPU free {free_bytes}B < required {total_bytes * _COMPUTE_BACKEND_OPTIONS.gpu_memory_safety_factor}B, using CPU"
        )
        return "cpu"

    logger.debug(
        f"Backend auto: selected GPU for {reduction_method}, input {input_bytes}B, shape {source_shape}"
    )
    return "gpu"


def downsample_block(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
    backend: Optional[str] = None,
    merged_chunk_count: int = 1,
) -> np.ndarray:
    reduction_method = normalize_reduction_method(reduction_method)
    selected_backend = backend or _select_compute_backend(
        source_shape=tuple(int(dim) for dim in data.shape),
        dtype=np.dtype(data.dtype),
        reduction_method=reduction_method,
        scale_hint=scale_hint,
        merged_chunk_count=merged_chunk_count,
    )

    logger.debug(
        f"downsample_block: shape={data.shape}, scale={scale_hint}, method={reduction_method}, backend={selected_backend}"
    )

    if selected_backend == "gpu":
        try:
            result = _downsample_block_gpu(data, scale_hint, reduction_method)
            logger.debug(
                f"downsample_block: GPU downsampling succeeded, output shape={result.shape}"
            )
            return result
        except Exception as e:
            logger.warning(f"downsample_block: GPU failed ({e}), falling back to CPU")
            pass

    return _downsample_block_cpu(data, scale_hint, reduction_method)


def _downsample_block_cpu(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    original_dtype = data.dtype

    if reduction_method == "nearest":
        return data[tuple(slice(0, None, scale) for scale in scale_hint)]

    padded_shape = _pad_shape_to_scale_multiple(
        tuple(int(dim) for dim in data.shape), scale_hint
    )
    padded = _pad_array_edge(data, padded_shape)
    target_shape = tuple(
        padded.shape[axis] // scale_hint[axis] for axis in range(padded.ndim)
    )

    if reduction_method == "linear":
        result = _resample_linear(padded, target_shape)
    else:
        result = _area_reduce(np.asarray(padded, dtype=np.float64), scale_hint)

    # Cast back to original dtype with safe rounding for integers
    if original_dtype != result.dtype:
        if np.issubdtype(original_dtype, np.integer):
            # Round and clip to valid range before casting
            info = np.iinfo(original_dtype)
            result = np.clip(np.round(result), info.min, info.max)
        result = result.astype(original_dtype)

    return result


def _downsample_block_gpu(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    if not _HAS_CUPY:
        raise RuntimeError("CuPy is not available")

    original_dtype = data.dtype
    gpu_data = cp.asarray(data)

    if reduction_method == "nearest":
        result = gpu_data[tuple(slice(0, None, scale) for scale in scale_hint)]
        return cp.asnumpy(result)

    padded_shape = _pad_shape_to_scale_multiple(
        tuple(int(dim) for dim in data.shape), scale_hint
    )
    gpu_data = _pad_array_edge_gpu(gpu_data, padded_shape)

    if reduction_method == "area":
        result = _area_reduce(gpu_data.astype(cp.float64), scale_hint)
    elif cupy_ndimage is None:
        raise RuntimeError("cupyx.scipy.ndimage is not available")
    else:
        target_shape = tuple(
            gpu_data.shape[axis] // scale_hint[axis] for axis in range(gpu_data.ndim)
        )
        zoom = tuple(
            target_shape[axis] / gpu_data.shape[axis] for axis in range(gpu_data.ndim)
        )
        result = cupy_ndimage.zoom(
            gpu_data.astype(cp.float32), zoom=zoom, order=1, mode="nearest"
        )

    # Cast back to original dtype with safe rounding for integers
    result = cp.asnumpy(result)
    if original_dtype != result.dtype:
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            result = np.clip(np.round(result), info.min, info.max)
        result = result.astype(original_dtype)

    return result


def _resample_linear(data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Downsample using separable linear interpolation at output pixel centers."""
    resampled = np.asarray(data, dtype=np.float64)

    for axis, dst_len in enumerate(target_shape):
        src_len = resampled.shape[axis]
        if dst_len == src_len:
            continue
        if dst_len <= 0:
            raise ValueError(f"Target shape must be positive on axis {axis}")

        src_coords = np.arange(src_len, dtype=np.float64)
        dst_coords = (
            (np.arange(dst_len, dtype=np.float64) + 0.5) * (src_len / dst_len)
        ) - 0.5
        dst_coords = np.clip(dst_coords, 0.0, float(max(src_len - 1, 0)))

        resampled = np.apply_along_axis(
            lambda values: np.interp(dst_coords, src_coords, values),
            axis,
            resampled,
        )

    return resampled
