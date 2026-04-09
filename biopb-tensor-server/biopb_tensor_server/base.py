"""Backend adapters for tensor storage formats.

This module provides a consistent interface for reading chunked multi-dimensional
arrays from various storage backends (Zarr, HDF5, OME-TIFF, TileDB).

Each adapter maps storage-specific chunk layouts to Arrow Flight endpoints:
- chunk_id: Opaque bytes identifying a chunk in the backend
- ChunkBounds: Array coordinates (start, stop) for the chunk

The adapters integrate with Arrow Flight's GetFlightInfo/DoGet flow:
1. GetFlightInfo returns FlightEndpoints with chunk_id tickets
2. DoGet uses the chunk_id to fetch the actual data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from math import lcm
from typing import List, Optional, Tuple, Iterator
import importlib
import struct
import os

import numpy as np
import pyarrow as pa

try:
    cp = importlib.import_module('cupy')
    cupy_ndimage = importlib.import_module('cupyx.scipy.ndimage')
    _HAS_CUPY = True
except ImportError:
    cp = None
    cupy_ndimage = None
    _HAS_CUPY = False

from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint, TensorReadOptions


@dataclass
class ChunkEndpoint:
    """A chunk with its metadata for Flight endpoint creation.

    Attributes:
        chunk_id: Backend-specific chunk identifier (bytes)
        bounds: Array coordinates (start, stop) for this chunk
    """
    chunk_id: bytes
    bounds: ChunkBounds


@dataclass
class TensorReadPlan:
    """Logical tensor read plan returned by the server planning layer."""

    descriptor: TensorDescriptor
    chunk_endpoints: List[ChunkEndpoint]


@dataclass
class ComputeBackendOptions:
    """Server-side compute backend selection options."""

    force_backend: str = 'auto'
    gpu_min_input_bytes: int = 4 * 1024 * 1024
    gpu_min_linear_input_bytes: int = 2 * 1024 * 1024
    gpu_memory_safety_factor: int = 4
    gpu_min_merged_chunks: int = 4


class BackendAdapter(ABC):
    """Abstract base class for tensor storage backend adapters.

    Each adapter handles a specific storage format (Zarr, HDF5, OME-TIFF, etc.)
    and provides methods to discover chunks and read chunk data.
    """

    @abstractmethod
    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the TensorDescriptor for this tensor.

        Returns:
            TensorDescriptor with array_id, shape, chunk_shape, dtype, dim_labels
        """
        pass

    @abstractmethod
    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        """Get chunk endpoints covering the tensor (or a slice of it).

        Args:
            slice_hint: Optional slice range. If provided, return only chunks
                       that intersect this range. If None, return all chunks.

        Returns:
            List of ChunkEndpoint objects with chunk_id and bounds
        """
        pass

    @abstractmethod
    def get_chunk_data(self, chunk_id: bytes) -> pa.RecordBatch:
        """Read a chunk's data as an Arrow RecordBatch.

        Args:
            chunk_id: Backend-specific chunk identifier

        Returns:
            Arrow RecordBatch containing the chunk's data
        """
        pass

    def get_arrow_schema(self, desc: Optional[TensorDescriptor] = None) -> pa.Schema:
        """Get the Arrow schema for this tensor.

        Returns:
            Arrow Schema with tensor extension type
        """
        return build_arrow_schema(desc or self.get_tensor_descriptor())

    def get_metadata(self) -> dict:
        """Return OME-compatible metadata as dict.

        For OME-Zarr: returns parsed .zattrs (multiscales, axes, omero, etc.)
        For OME-TIFF: returns extracted OME-XML as JSON-compatible dict
        For plain Zarr/HDF5: returns empty dict

        Will be serialized to metadata_json in TensorDescriptor.
        Override in subclasses to provide format-specific metadata.
        """
        return {}

    def get_scaled_read_plan(
        self,
        scale_hint: Tuple[int, ...],
        slice_hint: Optional[SliceHint],
        read_options: Optional[TensorReadOptions],
    ) -> TensorReadPlan:
        """Return read plan for requested scale.

        Default implementation: virtual scaling (compute virtual chunks from base).
        Override in OmeZarrAdapter to support precomputed levels via method="precompute".

        Args:
            scale_hint: Per-dimension scale factors
            slice_hint: Optional slice in base coordinates
            read_options: Optional read options including reduction_method

        Returns:
            TensorReadPlan with descriptor and chunk endpoints
        """
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(dim) for dim in base_desc.shape)
        base_chunk_shape = tuple(int(dim) for dim in base_desc.chunk_shape)
        source_start, source_stop = _normalized_slice_bounds(base_shape, slice_hint)
        source_shape = tuple(stop - start for start, stop in zip(source_start, source_stop))

        reduction_method = _normalize_reduction_method(
            read_options.reduction_method if read_options else None
        )

        logical_shape = _logical_shape_for_scale(source_shape, scale_hint)
        logical_chunk_shape = _logical_chunk_shape(base_chunk_shape, scale_hint, logical_shape)

        endpoints: List[ChunkEndpoint] = []

        def iter_virtual_chunks(dim: int = 0, logical_offset: Tuple[int, ...] = ()):
            if dim == len(logical_shape):
                yield logical_offset
                return
            axis_chunk = logical_chunk_shape[dim]
            axis_extent = logical_shape[dim]
            for axis_start in range(0, axis_extent, axis_chunk):
                yield from iter_virtual_chunks(dim + 1, logical_offset + (axis_start,))

        for logical_start in iter_virtual_chunks():
            logical_stop = tuple(
                min(logical_start[axis] + logical_chunk_shape[axis], logical_shape[axis])
                for axis in range(len(logical_shape))
            )
            source_chunk_start = tuple(
                source_start[axis] + logical_start[axis] * scale_hint[axis]
                for axis in range(len(logical_shape))
            )
            source_chunk_stop = tuple(
                source_start[axis] + logical_stop[axis] * scale_hint[axis]
                for axis in range(len(logical_shape))
            )
            valid_chunk_stop = tuple(
                min(source_chunk_stop[axis], source_stop[axis])
                for axis in range(len(logical_shape))
            )
            payload = _encode_virtual_chunk_payload(
                source_start=source_chunk_start,
                source_stop=source_chunk_stop,
                valid_stop=valid_chunk_stop,
                scale_hint=scale_hint,
                reduction_method=reduction_method,
            )
            endpoints.append(ChunkEndpoint(
                chunk_id=_encode_chunk_id(base_desc.array_id, payload),
                bounds=ChunkBounds(start=list(logical_start), stop=list(logical_stop)),
            ))

        logical_desc = TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=list(logical_shape),
            chunk_shape=list(logical_chunk_shape),
            dtype=_output_dtype(base_desc.dtype, reduction_method),
        )
        if slice_hint is not None:
            logical_desc.slice_hint.CopyFrom(slice_hint)
        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)

        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=endpoints)


def build_arrow_schema(desc: TensorDescriptor) -> pa.Schema:
    """Build an Arrow schema from a tensor descriptor."""
    dtype = np.dtype(desc.dtype)
    field = pa.field(
        "data",
        pa.from_numpy_dtype(dtype),
    )

    metadata = {
        "tensor_shape": ",".join(str(s) for s in desc.shape),
        "chunk_shape": ",".join(str(s) for s in desc.chunk_shape),
        "dim_labels": ",".join(desc.dim_labels) if desc.dim_labels else "",
    }

    return pa.schema([field], metadata=metadata)


def _chunks_intersect(
    chunk_start: List[int],
    chunk_stop: List[int],
    slice_start: List[int],
    slice_stop: List[int]
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


def _encode_chunk_id(array_id: str, backend_data: bytes) -> bytes:
    """Encode array_id and backend-specific data into chunk_id.

    Format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - M bytes: backend_data

    Args:
        array_id: Tensor identifier
        backend_data: Backend-specific chunk data

    Returns:
        Encoded chunk_id bytes
    """
    array_id_bytes = array_id.encode('utf-8')
    return struct.pack('>I', len(array_id_bytes)) + array_id_bytes + backend_data


def _decode_chunk_id(chunk_id: bytes) -> Tuple[str, bytes]:
    """Decode array_id and backend data from chunk_id.

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        Tuple of (array_id, backend_data)
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    array_id = chunk_id[4:4+array_id_len].decode('utf-8')
    backend_data = chunk_id[4+array_id_len:]
    return array_id, backend_data


_VIRTUAL_CHUNK_MAGIC = b'virt1'
_DEFAULT_REDUCTION_METHOD = 'nearest'
_SUPPORTED_REDUCTION_METHODS = {'nearest', 'area', 'linear', 'precompute'}
_METHOD_ALIASES = {
    'stride': 'nearest',
    'decimate': 'nearest',
    'mean': 'area',
    'precomputed': 'precompute',
}
_GPU_PREFERRED_METHODS = {'area', 'linear'}
_COMPUTE_BACKEND_OPTIONS = ComputeBackendOptions()


def configure_compute_backend(**kwargs) -> ComputeBackendOptions:
    """Update server-side compute backend selection options."""
    global _COMPUTE_BACKEND_OPTIONS

    options = ComputeBackendOptions(**{
        **_COMPUTE_BACKEND_OPTIONS.__dict__,
        **kwargs,
    })
    if options.force_backend not in {'auto', 'cpu', 'gpu'}:
        raise ValueError("force_backend must be one of: auto, cpu, gpu")
    if options.gpu_min_input_bytes < 0:
        raise ValueError('gpu_min_input_bytes must be non-negative')
    if options.gpu_min_linear_input_bytes < 0:
        raise ValueError('gpu_min_linear_input_bytes must be non-negative')
    if options.gpu_memory_safety_factor < 1:
        raise ValueError('gpu_memory_safety_factor must be >= 1')
    if options.gpu_min_merged_chunks < 1:
        raise ValueError('gpu_min_merged_chunks must be >= 1')

    _COMPUTE_BACKEND_OPTIONS = options
    return _COMPUTE_BACKEND_OPTIONS


def get_compute_backend_options() -> ComputeBackendOptions:
    """Return current server-side compute backend selection options."""
    return ComputeBackendOptions(**_COMPUTE_BACKEND_OPTIONS.__dict__)


def _encode_int64_sequence(values: Tuple[int, ...]) -> bytes:
    return b''.join(struct.pack('>q', value) for value in values)


def _decode_int64_sequence(data: bytes, offset: int, length: int) -> Tuple[Tuple[int, ...], int]:
    values = []
    for _ in range(length):
        values.append(struct.unpack('>q', data[offset:offset + 8])[0])
        offset += 8
    return tuple(values), offset


def _normalize_reduction_method(method: str) -> str:
    normalized = (method or _DEFAULT_REDUCTION_METHOD).strip().lower()
    normalized = _METHOD_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            f"Unsupported reduction method: {method}. "
            f"Supported methods: {sorted(_SUPPORTED_REDUCTION_METHODS)}"
        )
    return normalized


def _normalized_slice_bounds(
    shape: Tuple[int, ...],
    slice_hint: Optional[SliceHint],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
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


def _normalized_scale_hint(
    shape: Tuple[int, ...],
    read_options: Optional[TensorReadOptions],
) -> Optional[Tuple[int, ...]]:
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


def _logical_chunk_shape(
    chunk_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    logical_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    virtual_chunk = []
    for chunk, scale, axis_shape in zip(chunk_shape, scale_hint, logical_shape):
        virtual_axis = lcm(chunk, scale) // scale
        virtual_chunk.append(min(max(virtual_axis, 1), axis_shape))
    return tuple(virtual_chunk)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _logical_shape_for_scale(
    source_shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(_ceil_div(extent, scale) for extent, scale in zip(source_shape, scale_hint))


def _pad_shape_to_scale_multiple(
    shape: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> Tuple[int, ...]:
    return tuple(_ceil_div(extent, scale) * scale for extent, scale in zip(shape, scale_hint))


def _pad_array_edge(
    data: np.ndarray,
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    if tuple(int(dim) for dim in data.shape) == tuple(int(dim) for dim in target_shape):
        return data

    if any(target < current for target, current in zip(target_shape, data.shape)):
        raise ValueError(f"Target shape {target_shape} must be >= data shape {data.shape}")

    pad_width = [
        (0, int(target) - int(current))
        for current, target in zip(data.shape, target_shape)
    ]
    if all(pad_after == 0 for _, pad_after in pad_width):
        return data

    if data.size == 0 or any(dim == 0 for dim in data.shape):
        return np.pad(data, pad_width, mode='constant')

    return np.pad(data, pad_width, mode='edge')


def _pad_array_edge_gpu(
    data,
    target_shape: Tuple[int, ...],
):
    if tuple(int(dim) for dim in data.shape) == tuple(int(dim) for dim in target_shape):
        return data

    if any(target < current for target, current in zip(target_shape, data.shape)):
        raise ValueError(f"Target shape {target_shape} must be >= data shape {data.shape}")

    pad_width = [
        (0, int(target) - int(current))
        for current, target in zip(data.shape, target_shape)
    ]
    if all(pad_after == 0 for _, pad_after in pad_width):
        return data

    mode = 'constant' if data.size == 0 or any(dim == 0 for dim in data.shape) else 'edge'
    return cp.pad(data, pad_width, mode=mode)


def _output_dtype(base_dtype: str, reduction_method: str) -> str:
    return np.dtype(base_dtype).str


def _cast_reduced_array(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        data = np.rint(data)
        data = np.clip(data, info.min, info.max)
        return data.astype(target_dtype)

    if np.issubdtype(target_dtype, np.floating):
        return data.astype(target_dtype)

    return data.astype(target_dtype)


def _get_backend_override() -> str:
    override = os.environ.get('BIOPB_TENSOR_FORCE_BACKEND', _COMPUTE_BACKEND_OPTIONS.force_backend).strip().lower()
    if override in {'cpu', 'gpu', 'auto'}:
        return override
    return 'auto'


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
    reduction_method = _normalize_reduction_method(reduction_method)
    override = _get_backend_override()

    if override == 'cpu':
        return 'cpu'

    if override == 'gpu':
        if _HAS_CUPY and (reduction_method != 'linear' or cupy_ndimage is not None):
            return 'gpu'
        return 'cpu'

    if not _HAS_CUPY:
        return 'cpu'

    if reduction_method not in _GPU_PREFERRED_METHODS:
        return 'cpu'

    if reduction_method == 'linear' and cupy_ndimage is None:
        return 'cpu'

    input_bytes = _estimate_array_bytes(source_shape, dtype)
    output_shape = _logical_shape_for_scale(source_shape, scale_hint)
    output_bytes = _estimate_array_bytes(output_shape, dtype)
    total_bytes = input_bytes + output_bytes
    min_input_bytes = (
        _COMPUTE_BACKEND_OPTIONS.gpu_min_linear_input_bytes
        if reduction_method == 'linear'
        else _COMPUTE_BACKEND_OPTIONS.gpu_min_input_bytes
    )

    if input_bytes < min_input_bytes:
        return 'cpu'

    if (
        merged_chunk_count < _COMPUTE_BACKEND_OPTIONS.gpu_min_merged_chunks
        and input_bytes < (min_input_bytes * 2)
    ):
        return 'cpu'

    free_bytes = _get_gpu_free_bytes()
    if free_bytes > 0 and free_bytes < total_bytes * _COMPUTE_BACKEND_OPTIONS.gpu_memory_safety_factor:
        return 'cpu'

    return 'gpu'


def _encode_virtual_chunk_payload(
    source_start: Tuple[int, ...],
    source_stop: Tuple[int, ...],
    valid_stop: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> bytes:
    method_bytes = reduction_method.encode('utf-8')
    ndim = len(source_start)
    return b''.join([
        _VIRTUAL_CHUNK_MAGIC,
        struct.pack('>H', ndim),
        _encode_int64_sequence(source_start),
        _encode_int64_sequence(source_stop),
        _encode_int64_sequence(valid_stop),
        _encode_int64_sequence(scale_hint),
        struct.pack('>H', len(method_bytes)),
        method_bytes,
    ])


def _decode_virtual_chunk_payload(
    data: bytes,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str]:
    if not data.startswith(_VIRTUAL_CHUNK_MAGIC):
        raise ValueError('Chunk payload is not a virtual chunk')

    offset = len(_VIRTUAL_CHUNK_MAGIC)
    ndim = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    source_start, offset = _decode_int64_sequence(data, offset, ndim)
    source_stop, offset = _decode_int64_sequence(data, offset, ndim)
    valid_stop, offset = _decode_int64_sequence(data, offset, ndim)
    scale_hint, offset = _decode_int64_sequence(data, offset, ndim)
    method_len = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    reduction_method = data[offset:offset + method_len].decode('utf-8')
    return source_start, source_stop, valid_stop, scale_hint, reduction_method


def _record_batch_to_numpy(
    record_batch: pa.RecordBatch,
    bounds: ChunkBounds,
    dtype: str,
) -> np.ndarray:
    array = record_batch.column(0).to_numpy()
    chunk_shape = tuple(int(stop - start) for start, stop in zip(bounds.start, bounds.stop))
    return np.asarray(array, dtype=np.dtype(dtype)).reshape(chunk_shape)


def _downsample_block(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
    backend: Optional[str] = None,
    merged_chunk_count: int = 1,
) -> np.ndarray:
    reduction_method = _normalize_reduction_method(reduction_method)
    selected_backend = backend or _select_compute_backend(
        source_shape=tuple(int(dim) for dim in data.shape),
        dtype=np.dtype(data.dtype),
        reduction_method=reduction_method,
        scale_hint=scale_hint,
        merged_chunk_count=merged_chunk_count,
    )

    if selected_backend == 'gpu':
        try:
            return _downsample_block_gpu(data, scale_hint, reduction_method)
        except Exception:
            pass

    return _downsample_block_cpu(data, scale_hint, reduction_method)


def _downsample_block_cpu(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    reduction_method = _normalize_reduction_method(reduction_method)
    if reduction_method == 'nearest':
        return data[tuple(slice(0, None, scale) for scale in scale_hint)]

    padded_shape = _pad_shape_to_scale_multiple(tuple(int(dim) for dim in data.shape), scale_hint)
    padded = _pad_array_edge(data, padded_shape)
    target_shape = tuple(padded.shape[axis] // scale_hint[axis] for axis in range(padded.ndim))

    if reduction_method == 'linear':
        return _resample_linear(padded, target_shape)

    reduced = np.asarray(padded, dtype=np.float64)
    for axis in reversed(range(reduced.ndim)):
        scale = scale_hint[axis]
        axis_size = reduced.shape[axis]
        new_shape = (
            reduced.shape[:axis]
            + (axis_size // scale, scale)
            + reduced.shape[axis + 1:]
        )
        reduced = reduced.reshape(new_shape).mean(axis=axis + 1)
    return reduced


def _downsample_block_gpu(
    data: np.ndarray,
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> np.ndarray:
    if not _HAS_CUPY:
        raise RuntimeError('CuPy is not available')

    reduction_method = _normalize_reduction_method(reduction_method)
    gpu_data = cp.asarray(data)

    if reduction_method == 'nearest':
        result = gpu_data[tuple(slice(0, None, scale) for scale in scale_hint)]
        return cp.asnumpy(result)

    padded_shape = _pad_shape_to_scale_multiple(tuple(int(dim) for dim in data.shape), scale_hint)
    gpu_data = _pad_array_edge_gpu(gpu_data, padded_shape)

    if reduction_method == 'area':
        reduced = gpu_data.astype(cp.float64)
        for axis in reversed(range(reduced.ndim)):
            scale = scale_hint[axis]
            axis_size = reduced.shape[axis]
            new_shape = (
                reduced.shape[:axis]
                + (axis_size // scale, scale)
                + reduced.shape[axis + 1:]
            )
            reduced = reduced.reshape(new_shape).mean(axis=axis + 1)
        return cp.asnumpy(reduced)

    if cupy_ndimage is None:
        raise RuntimeError('cupyx.scipy.ndimage is not available')

    target_shape = tuple(gpu_data.shape[axis] // scale_hint[axis] for axis in range(gpu_data.ndim))
    zoom = tuple(target_shape[axis] / gpu_data.shape[axis] for axis in range(gpu_data.ndim))
    result = cupy_ndimage.zoom(gpu_data.astype(cp.float32), zoom=zoom, order=1, mode='nearest')
    return cp.asnumpy(result)


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
        dst_coords = ((np.arange(dst_len, dtype=np.float64) + 0.5) * (src_len / dst_len)) - 0.5
        dst_coords = np.clip(dst_coords, 0.0, float(max(src_len - 1, 0)))

        resampled = np.apply_along_axis(
            lambda values: np.interp(dst_coords, src_coords, values),
            axis,
            resampled,
        )

    return resampled


def _shift_bounds(
    bounds: ChunkBounds,
    logical_origin: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
) -> ChunkBounds:
    start = [int((bounds.start[axis] - logical_origin[axis]) // scale_hint[axis]) for axis in range(len(bounds.start))]
    stop = [int((bounds.stop[axis] - logical_origin[axis]) // scale_hint[axis]) for axis in range(len(bounds.stop))]
    return ChunkBounds(start=start, stop=stop)


def plan_tensor_read(
    adapter: BackendAdapter,
    request_desc: TensorDescriptor,
) -> TensorReadPlan:
    """Plan a logical tensor read by delegating to adapter.

    For unscaled reads: direct chunk mapping from base.
    For scaled reads: delegates to adapter's get_scaled_read_plan().
    """
    base_desc = adapter.get_tensor_descriptor()
    base_shape = tuple(int(dim) for dim in base_desc.shape)
    base_chunk_shape = tuple(int(dim) for dim in base_desc.chunk_shape)
    slice_hint = request_desc.slice_hint if request_desc.HasField('slice_hint') else None
    read_options = request_desc.read_options if request_desc.HasField('read_options') else None
    source_start, source_stop = _normalized_slice_bounds(base_shape, slice_hint)
    source_shape = tuple(stop - start for start, stop in zip(source_start, source_stop))
    scale_hint = _normalized_scale_hint(base_shape, read_options)

    if scale_hint is None:
        # No scaling - direct read from base
        chunk_endpoints = adapter.get_chunk_endpoints(
            SliceHint(start=list(source_start), stop=list(source_stop))
            if source_start != tuple(0 for _ in base_shape) or source_stop != base_shape
            else None
        )
        logical_endpoints = [
            ChunkEndpoint(
                chunk_id=endpoint.chunk_id,
                bounds=ChunkBounds(
                    start=[int(endpoint.bounds.start[axis] - source_start[axis]) for axis in range(len(base_shape))],
                    stop=[int(endpoint.bounds.stop[axis] - source_start[axis]) for axis in range(len(base_shape))],
                ),
            )
            for endpoint in chunk_endpoints
        ]
        logical_desc = TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=list(source_shape),
            chunk_shape=[min(int(chunk), int(size)) for chunk, size in zip(base_chunk_shape, source_shape)],
            dtype=base_desc.dtype,
        )
        if slice_hint is not None:
            logical_desc.slice_hint.CopyFrom(slice_hint)
        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)
        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)

    # Scaled read - delegate to adapter
    return adapter.get_scaled_read_plan(scale_hint, slice_hint, read_options)


def resolve_chunk_data(adapter: BackendAdapter, chunk_id: bytes) -> pa.RecordBatch:
    """Resolve either a real backend chunk or a virtual scaled chunk."""
    array_id, backend_data = _decode_chunk_id(chunk_id)
    if not backend_data.startswith(_VIRTUAL_CHUNK_MAGIC):
        return adapter.get_chunk_data(chunk_id)

    desc = adapter.get_tensor_descriptor()
    dtype = np.dtype(desc.dtype)
    source_start, source_stop, valid_stop, scale_hint, reduction_method = _decode_virtual_chunk_payload(backend_data)
    source_slice = SliceHint(start=list(source_start), stop=list(valid_stop))
    endpoints = adapter.get_chunk_endpoints(source_slice)
    source_shape = tuple(int(stop - start) for start, stop in zip(source_start, source_stop))
    valid_shape = tuple(int(stop - start) for start, stop in zip(source_start, valid_stop))
    source_block = np.zeros(valid_shape, dtype=dtype)

    for endpoint in endpoints:
        chunk_data = _record_batch_to_numpy(adapter.get_chunk_data(endpoint.chunk_id), endpoint.bounds, desc.dtype)
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
    array = pa.array(reduced.ravel())
    return pa.RecordBatch.from_arrays([array], ["data"])