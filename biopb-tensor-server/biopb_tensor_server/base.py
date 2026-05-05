"""Backend adapters for tensor storage formats.

This module provides a consistent interface for reading chunked multi-dimensional
arrays from various storage backends (Zarr, HDF5, OME-TIFF, TileDB).

Each adapter maps storage-specific chunk layouts to Arrow Flight endpoints:
- chunk_id: Opaque bytes identifying a chunk in the backend
- ChunkBounds: Array coordinates (start, stop) for the chunk

The adapters integrate with Arrow Flight's GetFlightInfo/DoGet flow:
1. GetFlightInfo returns FlightEndpoints with chunk_id tickets
2. DoGet uses the chunk_id to fetch the actual data

Raw data caching relies on OS page cache. Virtual chunk (scaled read) caching
uses the CacheManager with future/promise pattern for thread safety.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import lcm
from pathlib import Path
from typing import List, Optional, Tuple, Set, TYPE_CHECKING
import struct

import numpy as np
import pyarrow as pa

from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb.tensor.descriptor_pb2 import TensorDescriptor, SliceHint, TensorReadOptions, DataSourceDescriptor
from biopb_tensor_server.downsample import (
    ComputeBackendOptions,
    configure_compute_backend,
    get_compute_backend_options,
    _normalize_reduction_method,
    _ceil_div,
    _logical_shape_for_scale,
    _pad_array_edge,
    _output_dtype,
    _cast_reduced_array,
    _downsample_block,
)

if TYPE_CHECKING:
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.discovery import SourceClaim


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


class BackendAdapter(ABC):
    """Abstract base class for tensor storage backend adapters.

    Each adapter handles a specific storage format (Zarr, HDF5, OME-TIFF, etc.)
    and provides methods to discover chunks and read chunk data.
    """

    # === Claim Protocol ===

    @classmethod
    @abstractmethod
    def claim(cls, path: Path, visited_identities: Set[str]) -> Optional[SourceClaim]:
        """Claim a filesystem path as a data source.

        This method is called during discovery to detect if this adapter
        handles a given path. Adapters should check for format-specific
        characteristics (file extensions, metadata files, etc.).

        Args:
            path: Path to check (file or directory)
            visited_identities: Set of already-visited file identities
                               (for symlink/hardlink detection)

        Returns:
            SourceClaim if this adapter handles this path, None otherwise

        Example implementations:

            ZarrAdapter.claim():
                - Check if path is a .zarr directory
                - Verify .zarray or .zattrs exists

            OmeTiffAdapter.claim():
                - Check if path is a .ome.tiff file

            MultiFileOmeTiffAdapter.claim():
                - Check if path is a directory with multi-file structure
                - Return multi-node claim (directory + all TIFF files)
        """
        pass

    # === Instance methods ===

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

    # === Source-level methods (multifield support) ===

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors available in this source.

        Single tensor adapters return [self.get_tensor_descriptor()].
        Multi tensor adapters return list of descriptors for all contained tensors.

        The metadata_json field is NOT populated in these descriptors.
        Metadata is returned via GetFlightInfo in the response TensorDescriptor.

        Returns:
            List of TensorDescriptor for all tensors in this source
        """
        return [self.get_tensor_descriptor()]

    def get_tensor_adapter(self, tensor_id: str) -> 'BackendAdapter':
        """Get BackendAdapter for a specific tensor within this source.

        Usually this is just self. But multi-tensor adapters may return 
        a new BackendAdapter for that tensor.

        Args:
            tensor_id: Identifier for the specific tensor within this source

        Returns:
            BackendAdapter for the specified tensor
        """
        return self

    def get_source_descriptor(self) -> DataSourceDescriptor:
        """Build DataSourceDescriptor from this adapter.

        Returns:
            DataSourceDescriptor with source_id, source_type, tensor list.
            Some field are delayed fil, e.g., metadata_json.
            response's TensorDescriptor instead.
        """
        descriptor = self.get_tensor_descriptor()

        return DataSourceDescriptor(
            source_id=descriptor.array_id,
            source_url=self._source_url if hasattr(self, '_source_url') else "",
            source_type=self._source_type if hasattr(self, '_source_type') else "",
            tensors=self.list_tensor_descriptors(),
            metadata_json="",  # Not populated; returned via GetFlightInfo instead
        )

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
        ndim = len(base_shape)

        reduction_method = _normalize_reduction_method(
            read_options.reduction_method if read_options else None
        )

        # Step 1: find intersecting real chunks (slice_hint is in source coordinates)
        source_start, source_stop = _normalized_slice_bounds(base_shape, slice_hint)
        real_endpoints = self.get_chunk_endpoints(
            SliceHint(start=list(source_start), stop=list(source_stop))
            if slice_hint is not None else None
        )

        # Step 2: snap the real-chunk bounding box to lcm-group boundaries so that
        # each virtual chunk covers a whole number of real chunks and a whole number
        # of output pixels (handles cases like z-chunk=1, scale=2).
        lcm_per_axis = tuple(lcm(base_chunk_shape[ax], scale_hint[ax]) for ax in range(ndim))
        if real_endpoints:
            bbox_start = tuple(
                min(int(ep.bounds.start[ax]) for ep in real_endpoints) for ax in range(ndim)
            )
            bbox_stop = tuple(
                max(int(ep.bounds.stop[ax]) for ep in real_endpoints) for ax in range(ndim)
            )
        else:
            bbox_start = source_start
            bbox_stop = source_start

        snapped_start = tuple(
            (bbox_start[ax] // lcm_per_axis[ax]) * lcm_per_axis[ax]
            for ax in range(ndim)
        )
        snapped_stop = tuple(
            min(_ceil_div(bbox_stop[ax], lcm_per_axis[ax]) * lcm_per_axis[ax], base_shape[ax])
            for ax in range(ndim)
        )
        snapped_shape = tuple(snapped_stop[ax] - snapped_start[ax] for ax in range(ndim))

        logical_shape = _logical_shape_for_scale(snapped_shape, scale_hint)
        logical_chunk_shape = _logical_chunk_shape(base_chunk_shape, scale_hint, logical_shape)

        endpoints: List[ChunkEndpoint] = []

        def iter_virtual_chunks(dim: int = 0, logical_offset: Tuple[int, ...] = ()):
            if dim == ndim:
                yield logical_offset
                return
            axis_chunk = logical_chunk_shape[dim]
            axis_extent = logical_shape[dim]
            for axis_start in range(0, axis_extent, axis_chunk):
                yield from iter_virtual_chunks(dim + 1, logical_offset + (axis_start,))

        for logical_start in iter_virtual_chunks():
            logical_stop = tuple(
                min(logical_start[ax] + logical_chunk_shape[ax], logical_shape[ax])
                for ax in range(ndim)
            )
            source_chunk_start = tuple(
                snapped_start[ax] + logical_start[ax] * scale_hint[ax]
                for ax in range(ndim)
            )
            source_chunk_stop = tuple(
                snapped_start[ax] + logical_stop[ax] * scale_hint[ax]
                for ax in range(ndim)
            )
            # Clip to array boundary; edge virtual chunks may extend past valid data
            valid_chunk_stop = tuple(
                min(source_chunk_stop[ax], base_shape[ax])
                for ax in range(ndim)
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
        # Report realized slice in source coordinates (snapped to chunk/lcm boundaries)
        if snapped_start != tuple(0 for _ in range(ndim)) or snapped_stop != base_shape:
            logical_desc.slice_hint.start[:] = list(snapped_start)
            logical_desc.slice_hint.stop[:] = list(snapped_stop)
        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)

        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=endpoints)


def build_arrow_schema(desc: TensorDescriptor) -> pa.Schema:
    """Build an Arrow schema from a tensor descriptor."""
    import importlib.metadata

    dtype = np.dtype(desc.dtype)
    field = pa.field("data", pa.from_numpy_dtype(dtype))

    # Schema metadata: biopb version for compatibility tracking
    # TensorDescriptor schema version matches biopb protobuf package version
    metadata = {
        "tensor_schema_version": importlib.metadata.version("biopb"),
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


def _encode_int64_sequence(values: Tuple[int, ...]) -> bytes:
    return b''.join(struct.pack('>q', value) for value in values)


def _decode_int64_sequence(data: bytes, offset: int, length: int) -> Tuple[Tuple[int, ...], int]:
    values = []
    for _ in range(length):
        values.append(struct.unpack('>q', data[offset:offset + 8])[0])
        offset += 8
    return tuple(values), offset


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

    ndim = len(base_shape)
    if scale_hint is None:
        # No scaling - direct read from base
        chunk_endpoints = adapter.get_chunk_endpoints(
            SliceHint(start=list(source_start), stop=list(source_stop))
            if slice_hint is not None else None
        )
        # Realized bounds = bounding box of intersecting chunks (always non-negative)
        if chunk_endpoints:
            realized_start = tuple(
                min(int(ep.bounds.start[ax]) for ep in chunk_endpoints) for ax in range(ndim)
            )
            realized_stop = tuple(
                max(int(ep.bounds.stop[ax]) for ep in chunk_endpoints) for ax in range(ndim)
            )
        else:
            realized_start = source_start
            realized_stop = source_start
        realized_shape = tuple(realized_stop[ax] - realized_start[ax] for ax in range(ndim))
        logical_endpoints = [
            ChunkEndpoint(
                chunk_id=endpoint.chunk_id,
                bounds=ChunkBounds(
                    start=[int(endpoint.bounds.start[ax] - realized_start[ax]) for ax in range(ndim)],
                    stop=[int(endpoint.bounds.stop[ax] - realized_start[ax]) for ax in range(ndim)],
                ),
            )
            for endpoint in chunk_endpoints
        ]
        logical_desc = TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=list(realized_shape),
            chunk_shape=[min(int(chunk), int(size)) for chunk, size in zip(base_chunk_shape, realized_shape)],
            dtype=base_desc.dtype,
        )
        if realized_start != tuple(0 for _ in range(ndim)) or realized_stop != base_shape:
            logical_desc.slice_hint.start[:] = list(realized_start)
            logical_desc.slice_hint.stop[:] = list(realized_stop)
        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)
        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)

    # Scaled read - delegate to adapter
    return adapter.get_scaled_read_plan(scale_hint, slice_hint, read_options)


def resolve_chunk_data(
    adapter: BackendAdapter,
    chunk_id: bytes,
    cache_manager: Optional[CacheManager] = None,
) -> pa.RecordBatch:
    """Resolve either a real backend chunk or a virtual scaled chunk.

    For virtual chunks (scaled reads), uses cache if cache_manager is provided.
    The cache uses future/promise pattern: only one thread computes while
    others wait on the pending entry.

    Args:
        adapter: Backend adapter for the tensor
        chunk_id: Encoded chunk identifier
        cache_manager: Optional cache manager for caching virtual chunks

    Returns:
        Arrow RecordBatch containing the chunk data
    """
    from biopb_tensor_server.cache import CacheKey
    from biopb_tensor_server.cache.base import EntryState

    array_id, backend_data = _decode_chunk_id(chunk_id)

    # Not virtual chunk - delegate to adapter (no caching needed)
    if not backend_data.startswith(_VIRTUAL_CHUNK_MAGIC):
        return adapter.get_chunk_data(chunk_id)

    # Virtual chunk - build cache key
    source_start, source_stop, valid_stop, scale_hint, reduction_method = _decode_virtual_chunk_payload(backend_data)
    cache_key = CacheKey(
        array_id=array_id,
        scale_hint=scale_hint,
        source_start=source_start,
        source_stop=source_stop,
        valid_stop=valid_stop,
        reduction_method=reduction_method,
    )
    key_bytes = cache_key.to_bytes()

    if cache_manager is None:
        # No cache - just compute
        return _compute_virtual_chunk(adapter, backend_data)

    # With cache - use future/promise pattern
    entry, is_owner = cache_manager.start_compute(key_bytes, metadata={'array_id': array_id})

    if is_owner:
        # We own the computation - compute and complete
        try:
            result = _compute_virtual_chunk(adapter, backend_data)
            size_bytes = sum(col.nbytes for col in result.columns)
            cache_manager.complete_entry(key_bytes, result, size_bytes)
        except Exception as e:
            cache_manager.fail_entry(key_bytes, e)
            raise
    else:
        # Another thread is computing - wait for it
        # entry is acquired by start_compute, state may be PENDING
        if entry.state == EntryState.PENDING:
            # Wait outside lock (event is set by computing thread)
            entry.wait_ready()  # Raises if computation failed

    # Entry is now READY, acquired (ref_count >= 1)
    # Release after getting data - Arrow buffers have their own ref counting
    data = entry.data
    cache_manager.release(key_bytes)
    return data


def _compute_virtual_chunk(adapter: BackendAdapter, backend_data: bytes) -> pa.RecordBatch:
    """Compute a virtual scaled chunk from source data.

    This is the internal computation logic extracted from resolve_chunk_data
    for separation of concerns.

    Args:
        adapter: Backend adapter for the tensor
        backend_data: Decoded virtual chunk payload

    Returns:
        Arrow RecordBatch containing the computed chunk data
    """
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