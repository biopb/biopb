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

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import lcm
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    SliceHint,
    TensorDescriptor,
    TensorReadOptions,
)
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.chunk import (
    ChunkEndpoint,
    decode_chunk_id,
    encode_chunk_id,
    get_backend_data,
)
from biopb_tensor_server.downsample import (
    _cast_reduced_array,
    _ceil_div,
    _downsample_block,
    _logical_shape_for_scale,
    _normalize_reduction_method,
    _output_dtype,
    _pad_array_edge,
)

MAX_ARROW_BATCH_BYTES = 2 * 1024 * 1024 * 1024 - 1  # ~2GB
_VIRTUAL_CHUNK_MAGIC = b'virt1'  # Magic prefix to identify virtual chunk payloads

if TYPE_CHECKING:
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.discovery import SourceClaim


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

    # === flight methods ===

    @abstractmethod
    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors available in this source.

        Returns:
            List of TensorDescriptor for all tensors in this source
        """
        pass


    def get_tensor_adapter(self, tensor_id: str) -> 'BackendAdapter':
        """ Factory method to return adapter with specific tensor context.

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
        return DataSourceDescriptor(
            source_id=self.array_id,
            source_url=self._source_url if hasattr(self, '_source_url') else "",
            source_type=self._source_type if hasattr(self, '_source_type') else "",
            tensors=self.list_tensor_descriptors(),
            metadata_json="",  # filled by GetFlightInfo()
        )


    # === tensor methods ===

    @abstractmethod
    def get_raw_chunk_endpoints(self) -> Iterator[ChunkEndpoint]:
        """Get raw chunk endpoints from the backend (before filtering/splitting).

        Subclasses implement this to return their native chunk layout.
        The base class get_chunk_endpoints() handles filtering and splitting.

        Returns:
            Iterator of ChunkEndpoint objects with chunk_id and bounds.
            Adapters enumerate ALL chunks; filtering is done by base class.
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


    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the TensorDescriptor for this adapter.

        Returns:
            TensorDescriptor with array_id, shape, chunk_shape, dtype, dim_labels
        """
        return self.list_tensor_descriptors()[0]  # Default: single tensor


    def get_chunk_endpoints(
        self,
        slice_hint: Optional[SliceHint] = None
    ) -> List[ChunkEndpoint]:
        """Get unscaled chunk endpoints, filtering by slice_hint.

        This method wraps the adapter-specific chunk discovery logic with
        filtering by slice_hint (only chunks intersecting the slice).

        Note: Chunk splitting is handled in get_read_plan() for oversized chunks.

        Args:
            slice_hint: Optional slice range. If provided, return only chunks
                       that intersect this range. If None, return all chunks.

        Returns:
            List of ChunkEndpoint objects with chunk_id and bounds.
        """
        raw_endpoints = self.get_raw_chunk_endpoints()

        # Filter by slice_hint if provided
        if slice_hint is not None:
            slice_start = list(slice_hint.start)
            slice_stop = list(slice_hint.stop)
            filtered_endpoints = [
                ep for ep in raw_endpoints
                if _chunks_intersect(
                    list(ep.bounds.start), list(ep.bounds.stop),
                    slice_start, slice_stop
                )
            ]
            return filtered_endpoints

        return list(raw_endpoints)


    def get_arrow_schema(self, desc: Optional[TensorDescriptor] = None) -> pa.Schema:
        """Get the Arrow schema for this tensor.

        Returns:
            Arrow Schema with tensor extension type
        """
        import importlib.metadata

        desc = desc or self.get_tensor_descriptor()

        dtype = np.dtype(desc.dtype)
        field = pa.field("data", pa.from_numpy_dtype(dtype))

        # Schema metadata: biopb version for compatibility tracking
        metadata = {
            "tensor_schema_version": importlib.metadata.version("biopb"),
        }

        return pa.schema([field], metadata=metadata)


    def get_metadata(self) -> dict:
        """Return metadata as dict. In most cases this is OME metadata.

        For OME-Zarr: returns parsed .zattrs (multiscales, axes, omero, etc.)
        For OME-TIFF: returns extracted OME-XML as JSON-compatible dict
        For plain Zarr/HDF5: returns empty dict

        Will be serialized to metadata_json in TensorDescriptor.
        Override in subclasses to provide format-specific metadata.
        """
        return {}


    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Plan a logical tensor read by delegating to adapter.
        """
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(dim) for dim in base_desc.shape)
        chunk_shape = tuple(int(dim) for dim in base_desc.chunk_shape)
        slice_hint = request_desc.slice_hint if request_desc.HasField('slice_hint') else None
        read_options = request_desc.read_options if request_desc.HasField('read_options') else None
        source_start, source_stop = _normalized_slice_bounds(base_shape, slice_hint)
        scale_hint = _normalized_scale_hint(base_shape, read_options)
        reduction_method = _normalize_reduction_method(
            read_options.reduction_method if read_options else None
        )
        ndim = len(base_shape)

        # find intersecting real chunks (slice_hint is in source coordinates)
        real_endpoints = self.get_chunk_endpoints(
            SliceHint(start=list(source_start), stop=list(source_stop))
            if slice_hint is not None else None
        )
        realized_start = tuple(
            min(int(ep.bounds.start[ax]) for ep in real_endpoints) for ax in range(ndim)
        )
        realized_stop = tuple(
            max(int(ep.bounds.stop[ax]) for ep in real_endpoints) for ax in range(ndim)
        )
        realized_shape = tuple(realized_stop[ax] - realized_start[ax] for ax in range(ndim))

        if scale_hint is None:
            # Real branch: check for oversized chunks and split if needed
            logical_endpoints: List[ChunkEndpoint] = []
            for endpoint in real_endpoints:
                chunk_shape = tuple(int(stop - start) for start, stop in zip(endpoint.bounds.start, endpoint.bounds.stop))
                if _needs_splitting(chunk_shape, base_desc.dtype):
                    shifted_bounds = ChunkBounds(
                        start=[int(endpoint.bounds.start[ax] - realized_start[ax]) for ax in range(ndim)],
                        stop=[int(endpoint.bounds.stop[ax] - realized_start[ax]) for ax in range(ndim)],
                    )
                    shifted_endpoint = ChunkEndpoint(chunk_id=endpoint.chunk_id, bounds=shifted_bounds)
                    sub_endpoints = _split_endpoint(base_desc.array_id, shifted_endpoint, base_desc.dtype, is_virtual=False)
                    logical_endpoints.extend(sub_endpoints)
                else:
                    logical_endpoints.append(ChunkEndpoint(
                        chunk_id=endpoint.chunk_id,
                        bounds=ChunkBounds(
                            start=[int(endpoint.bounds.start[ax] - realized_start[ax]) for ax in range(ndim)],
                            stop=[int(endpoint.bounds.stop[ax] - realized_start[ax]) for ax in range(ndim)],
                        ),
                    ))
            logical_desc = TensorDescriptor(
                array_id=base_desc.array_id,
                dim_labels=base_desc.dim_labels,
                shape=list(realized_shape),
                chunk_shape=list(chunk_shape),
                dtype=base_desc.dtype,
            )

            if realized_start != tuple(0 for _ in range(ndim)) or realized_stop != base_shape:
                logical_desc.slice_hint.start[:] = list(realized_start)
                logical_desc.slice_hint.stop[:] = list(realized_stop)

        else:
            lcm_per_axis = tuple(lcm(chunk_shape[ax], scale_hint[ax]) for ax in range(ndim))

            snapped_start = tuple(
                (realized_start[ax] // lcm_per_axis[ax]) * lcm_per_axis[ax]
                for ax in range(ndim)
            )
            snapped_stop = tuple(
                min(_ceil_div(realized_stop[ax], lcm_per_axis[ax]) * lcm_per_axis[ax], base_shape[ax])
                for ax in range(ndim)
            )
            snapped_shape = tuple(snapped_stop[ax] - snapped_start[ax] for ax in range(ndim))

            logical_shape = _logical_shape_for_scale(snapped_shape, scale_hint)
            logical_chunk_shape = _logical_chunk_shape(chunk_shape, scale_hint, logical_shape)
            output_dtype = _output_dtype(base_desc.dtype, reduction_method)

            logical_endpoints: List[ChunkEndpoint] = []

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

                # Check if virtual chunk needs splitting
                virtual_chunk_shape = tuple(logical_stop[ax] - logical_start[ax] for ax in range(ndim))
                if _needs_splitting(virtual_chunk_shape, output_dtype):
                    # Create virtual chunk payload (without split info first)
                    base_payload = _encode_virtual_chunk_payload(
                        source_start=source_chunk_start,
                        source_stop=source_chunk_stop,
                        valid_stop=valid_chunk_stop,
                        scale_hint=scale_hint,
                        reduction_method=reduction_method,
                    )
                    # Create endpoint with bounds, then split
                    virtual_ep = ChunkEndpoint(
                        chunk_id=encode_chunk_id(base_desc.array_id, base_payload),
                        bounds=ChunkBounds(start=list(logical_start), stop=list(logical_stop)),
                    )
                    sub_endpoints = _split_endpoint(base_desc.array_id, virtual_ep, output_dtype, is_virtual=True)
                    logical_endpoints.extend(sub_endpoints)
                else:
                    payload = _encode_virtual_chunk_payload(
                        source_start=source_chunk_start,
                        source_stop=source_chunk_stop,
                        valid_stop=valid_chunk_stop,
                        scale_hint=scale_hint,
                        reduction_method=reduction_method,
                    )
                    logical_endpoints.append(ChunkEndpoint(
                        chunk_id=encode_chunk_id(base_desc.array_id, payload),
                        bounds=ChunkBounds(start=list(logical_start), stop=list(logical_stop)),
                    ))

            logical_desc = TensorDescriptor(
                array_id=base_desc.array_id,
                dim_labels=base_desc.dim_labels,
                shape=list(logical_shape),
                chunk_shape=list(logical_chunk_shape),
                dtype=_output_dtype(base_desc.dtype, reduction_method),
            )

            if snapped_start != tuple(0 for _ in range(ndim)) or snapped_stop != base_shape:
                logical_desc.slice_hint.start[:] = list(snapped_start)
                logical_desc.slice_hint.stop[:] = list(snapped_stop)

        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)

        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)    


    def resolve_chunk_data(
        self, chunk_id: bytes,  cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Resolve either a real backend chunk, a split chunk, or a virtual scaled chunk.

        Args:
            adapter: Backend adapter for the tensor
            chunk_id: Encoded chunk identifier
            cache_manager: Optional cache manager for caching chunks

        Returns:
            Arrow RecordBatch containing the chunk data
        """
        from biopb_tensor_server.cache import ArrowFileBackend

        desc = self.get_tensor_descriptor()
        array_id, backend_data, split_index, split_max = decode_chunk_id(chunk_id)

        is_virtual = backend_data.startswith(_VIRTUAL_CHUNK_MAGIC)
        is_file_backend = cache_manager is not None and isinstance(cache_manager.backend, ArrowFileBackend)

        should_cache = cache_manager is not None and (is_virtual or is_file_backend)

        # Use chunk_id directly as stable cache key (for both virtual and regular chunks)
        key_bytes = chunk_id

        def compute_fn():
            if is_virtual:
                source_start, _, valid_stop, _, _ =  _decode_virtual_chunk_payload(backend_data)
                parent_bounds = ChunkBounds(
                    start=list(source_start),
                    stop=list(valid_stop),
                )

                result = _compute_virtual_chunk(self, backend_data)

                if split_max > 1:
                    # Slice the virtual chunk result
                    result = _slice_result(result, parent_bounds, split_index, split_max, desc.dtype)
            else:
                # For real chunks, get the bounds from the backend
                result = self.get_chunk_data(chunk_id)

                if split_max > 1:
                    parent_bounds = _get_chunk_bounds_from_backend_key(self, backend_data)
                    result = _slice_result(result, parent_bounds, split_index, split_max, desc.dtype)

            return result, sum(col.nbytes for col in result.columns)

        if should_cache:
            entry = cache_manager.get_or_acquire(key_bytes, compute_fn, metadata={'array_id': array_id})
            data = entry.data
            cache_manager.release(key_bytes)

        else:
            data, _ = compute_fn()

        return data


# === helper functions ===


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


def _get_chunk_bounds_from_backend_key(
    adapter: BackendAdapter,
    backend_key: bytes,
) -> ChunkBounds:
    """Get chunk bounds from backend key by finding matching endpoint.

    Args:
        adapter: Backend adapter for the tensor
        backend_key: Backend-specific chunk identifier

    Returns:
        ChunkBounds for the matching chunk
    """
    # Iterate through all endpoints to find matching one
    for endpoint in adapter.get_raw_chunk_endpoints():
        # Decode the endpoint's chunk_id to get its backend_data
        endpoint_backend_data = get_backend_data(endpoint.chunk_id)
        # For unsplit real chunks, backend_data == backend_key
        if endpoint_backend_data == backend_key:
            return endpoint.bounds

    raise ValueError(f"Could not find chunk bounds for backend_key: {backend_key}")


def _split_endpoint(
    array_id: str,
    ep: ChunkEndpoint,
    dtype: str,
    is_virtual: bool,
) -> List[ChunkEndpoint]:
    """Split an oversized endpoint into sub-endpoints within size limit.

    Strategy: split along the largest axis for even division.
    Each sub-endpoint has:
    - chunk_id: parent payload with split_index/split_max encoded
    - bounds: sub-bounds within parent (logical coordinates)

    Args:
        array_id: Tensor identifier
        ep: ChunkEndpoint to split
        dtype: Data type string for size calculation
        is_virtual: True if this is a virtual chunk

    Returns:
        List of sub-endpoints that each fit within the Arrow batch limit
    """
    parent_shape = tuple(stop - start for start, stop in zip(ep.bounds.start, ep.bounds.stop))
    item_size = np.dtype(dtype).itemsize

    # Calculate how many splits needed
    parent_bytes = int(np.prod(parent_shape)) * item_size
    n_splits = int(np.ceil(parent_bytes / MAX_ARROW_BATCH_BYTES))

    # Choose axis with largest dimension
    split_axis = max(range(len(parent_shape)), key=lambda ax: parent_shape[ax])

    axis_size = parent_shape[split_axis]
    sub_axis_size = axis_size // n_splits

    # Get parent backend_data from chunk_id
    parent_backend_data = get_backend_data(ep.chunk_id)

    sub_endpoints = []
    for i in range(n_splits):
        # Calculate sub-chunk bounds on split axis (in logical coordinates)
        axis_start = ep.bounds.start[split_axis] + i * sub_axis_size
        axis_stop = min(ep.bounds.start[split_axis] + (i + 1) * sub_axis_size, ep.bounds.stop[split_axis])

        sub_start = list(ep.bounds.start)
        sub_stop = list(ep.bounds.stop)
        sub_start[split_axis] = axis_start
        sub_stop[split_axis] = axis_stop

        sub_bounds = ChunkBounds(start=sub_start, stop=sub_stop)

        sub_chunk_id = encode_chunk_id(array_id, parent_backend_data, i, n_splits)

        sub_endpoints.append(ChunkEndpoint(chunk_id=sub_chunk_id, bounds=sub_bounds))

    return sub_endpoints


def _slice_result(
    record_batch: pa.RecordBatch,
    parent_bounds: ChunkBounds,
    split_index: int,
    split_max: int,
    dtype: str,
) -> pa.RecordBatch:
    """Slice a RecordBatch along the largest axis based on split info.

    Args:
        record_batch: Full chunk data as RecordBatch
        parent_bounds: Original bounds of the full chunk
        split_index: Which sub-chunk (0 to split_max-1)
        split_max: Total number of splits
        dtype: Data type string for size calculation

    Returns:
        Smaller RecordBatch containing the sub-chunk data
    """
    parent_shape = tuple(stop - start for start, stop in zip(parent_bounds.start, parent_bounds.stop))
    parent_arr = _record_batch_to_numpy(record_batch, parent_bounds, dtype)

    # Find axis with largest dimension
    split_axis = max(range(len(parent_shape)), key=lambda ax: parent_shape[ax])

    # Compute sub-chunk bounds on split axis
    axis_size = parent_shape[split_axis]
    sub_axis_size = axis_size // split_max

    axis_start = split_index * sub_axis_size
    axis_stop = min((split_index + 1) * sub_axis_size, axis_size)

    # Build slice tuple
    slices = tuple(
        slice(0, parent_shape[ax]) if ax != split_axis
        else slice(axis_start, axis_stop)
        for ax in range(len(parent_shape))
    )

    sub_arr = parent_arr[slices]
    array = pa.array(sub_arr.ravel())
    return pa.RecordBatch.from_arrays([array], ["data"])


def _record_batch_to_numpy(
    record_batch: pa.RecordBatch,
    bounds: ChunkBounds,
    dtype: str,
) -> np.ndarray:
    array = record_batch.column(0).to_numpy()
    chunk_shape = tuple(int(stop - start) for start, stop in zip(bounds.start, bounds.stop))
    return np.asarray(array, dtype=np.dtype(dtype)).reshape(chunk_shape)


def _compute_virtual_chunk(adapter: BackendAdapter, backend_data: bytes) -> pa.RecordBatch:
    """Compute a virtual scaled chunk from source data.

    This is the internal computation logic extracted from resolve_chunk_data
    for separation of concerns. Returns the full virtual chunk before any
    splitting is applied.

    Args:
        adapter: Backend adapter for the tensor
        backend_data: Decoded virtual chunk payload

    Returns:
        Arrow RecordBatch containing the computed chunk data
    """
    desc = adapter.get_tensor_descriptor()
    dtype = np.dtype(desc.dtype)
    # Decode payload - split_index/split_max are ignored here (slicing done in resolve_chunk_data)
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

def _encode_virtual_chunk_payload(
    source_start: Tuple[int, ...],
    source_stop: Tuple[int, ...],
    valid_stop: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    reduction_method: str,
) -> bytes:
    """Encode virtual chunk payload with optional split info.

    Format:
    - 5 bytes: magic (_VIRTUAL_CHUNK_MAGIC)
    - 2 bytes: ndim (uint16, big-endian)
    - 8*ndim bytes: source_start (int64, big-endian)
    - 8*ndim bytes: source_stop (int64, big-endian)
    - 8*ndim bytes: valid_stop (int64, big-endian)
    - 8*ndim bytes: scale_hint (int64, big-endian)
    - 2 bytes: method length (uint16, big-endian)
    - N bytes: reduction_method (UTF-8)

    Unsplit chunks have split_index=0, split_max=1 (encoded at end).
    """
    def encode_int64_sequence(values: Tuple[int, ...]) -> bytes:
        return b''.join(struct.pack('>q', value) for value in values)

    method_bytes = reduction_method.encode('utf-8')
    ndim = len(source_start)
    return b''.join([
        _VIRTUAL_CHUNK_MAGIC,
        struct.pack('>H', ndim),
        encode_int64_sequence(source_start),
        encode_int64_sequence(source_stop),
        encode_int64_sequence(valid_stop),
        encode_int64_sequence(scale_hint),
        struct.pack('>H', len(method_bytes)),
        method_bytes,
    ])


def _decode_virtual_chunk_payload(
    data: bytes,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str]:
    """Decode virtual chunk payload including split info.

    Returns:
        Tuple of (source_start, source_stop, valid_stop, scale_hint, reduction_method, split_index, split_max)
    """
    if not data.startswith(_VIRTUAL_CHUNK_MAGIC):
        raise ValueError('Chunk payload is not a virtual chunk')

    def decode_int64_sequence(data: bytes, offset: int, length: int) -> Tuple[Tuple[int, ...], int]:
        values = []
        for _ in range(length):
            values.append(struct.unpack('>q', data[offset:offset + 8])[0])
            offset += 8
        return tuple(values), offset

    offset = len(_VIRTUAL_CHUNK_MAGIC)
    ndim = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    source_start, offset = decode_int64_sequence(data, offset, ndim)
    source_stop, offset = decode_int64_sequence(data, offset, ndim)
    valid_stop, offset = decode_int64_sequence(data, offset, ndim)
    scale_hint, offset = decode_int64_sequence(data, offset, ndim)
    method_len = struct.unpack('>H', data[offset:offset + 2])[0]
    offset += 2
    reduction_method = data[offset:offset + method_len].decode('utf-8')
    offset += method_len

    return source_start, source_stop, valid_stop, scale_hint, reduction_method

def _estimate_chunk_bytes(shape: Tuple[int, ...], dtype: str) -> int:
    """Estimate chunk size in bytes from shape and dtype."""
    num_elements = int(np.prod(shape, dtype=np.int64))
    return num_elements * np.dtype(dtype).itemsize


def _needs_splitting(chunk_shape: Tuple[int, ...], dtype: str) -> bool:
    """Check if chunk exceeds Arrow batch limit."""
    return _estimate_chunk_bytes(chunk_shape, dtype) > MAX_ARROW_BATCH_BYTES