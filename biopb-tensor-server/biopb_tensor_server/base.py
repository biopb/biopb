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

import logging
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import lcm
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    SliceHint,
    TensorDescriptor,
)
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.chunk import (
    ChunkEndpoint,
    _VIRTUAL_CHUNK_MAGIC,
    chunks_intersect,
    compute_virtual_chunk,
    decode_chunk_id,
    decode_virtual_chunk_payload,
    encode_chunk_id,
    encode_virtual_chunk_payload,
    get_backend_data,
    get_chunk_bounds_from_backend_key,
    logical_chunk_shape,
    needs_splitting,
    normalized_scale_hint,
    normalized_slice_bounds,
    slice_array,
    split_endpoint,
)
from biopb_tensor_server.downsample import (
    _ceil_div,
    _logical_shape_for_scale,
    _normalize_reduction_method,
    _output_dtype,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.discovery import SourceClaim


class MethodContext(Enum):
    """Context requirement for BackendAdapter methods.

    SOURCE: Must be called before get_tensor_adapter() (source-level methods)
    TENSOR: Must be called after get_tensor_adapter() (tensor-level methods)
    ANY: Can be called in either context
    """
    SOURCE = "source"
    TENSOR = "tensor"
    ANY = "any"


class BackendAdapterMeta(ABCMeta):
    """Metaclass that wraps methods with context checks.

    This enforces that methods are called in the correct context:
    - Source methods must be called before get_tensor_adapter()
    - Tensor methods must be called after get_tensor_adapter()
    """

    CONTEXT_MAP = {
        'list_tensor_descriptors': MethodContext.ANY,
        'get_metadata': MethodContext.ANY,
        # Source methods (called before tensor selection)
        'get_source_descriptor': MethodContext.SOURCE,
        'get_tensor_adapter': MethodContext.SOURCE,
        # Tensor methods (called after get_tensor_adapter())
        'get_tensor_descriptor': MethodContext.TENSOR,
        'get_chunk_endpoints': MethodContext.TENSOR,
        'get_raw_chunk_endpoints': MethodContext.TENSOR,
        'get_chunk_array': MethodContext.TENSOR,
        'get_read_plan': MethodContext.TENSOR,
        'resolve_chunk_data': MethodContext.TENSOR,
        'get_arrow_schema': MethodContext.TENSOR,
    }

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Wrap methods defined in this class
        for method_name, context in mcs.CONTEXT_MAP.items():
            if method_name in namespace:
                original = namespace[method_name]
                if callable(original) and not method_name.startswith('_'):
                    wrapped = mcs._wrap_with_context(original, context, name)
                    setattr(cls, method_name, wrapped)

        return cls

    @staticmethod
    def _wrap_with_context(method, context: MethodContext, class_name: str):
        """Wrap a method with context checking."""
        def wrapper(self, *args, **kwargs):
            # Skip context enforcement for single-tensor sources
            # They can call tensor methods without get_tensor_adapter()
            if getattr(self, '_single_tensor_source', True):
                return method(self, *args, **kwargs)

            if context == MethodContext.SOURCE:
                if self._tensor_context:
                    raise RuntimeError(
                        f"{method.__name__}() is a source method but adapter is in tensor context "
                        f"(called after get_tensor_adapter()). Class: {class_name}"
                    )
            elif context == MethodContext.TENSOR:
                if not self._tensor_context:
                    raise RuntimeError(
                        f"{method.__name__}() is a tensor method but adapter is in source context "
                        f"(call get_tensor_adapter() first). Class: {class_name}"
                    )
            return method(self, *args, **kwargs)

        wrapper._context_wrapped = True
        wrapper.__name__ = method.__name__
        wrapper.__doc__ = method.__doc__
        return wrapper


@dataclass
class TensorReadPlan:
    """Logical tensor read plan returned by the server planning layer."""

    descriptor: TensorDescriptor
    chunk_endpoints: List[ChunkEndpoint]


class BackendAdapter(ABC, metaclass=BackendAdapterMeta):
    """Abstract base class for tensor storage backend adapters.

    Each adapter handles a specific storage format (Zarr, HDF5, OME-TIFF, etc.)
    and provides methods to discover chunks and read chunk data.

    Context Management:
    - Source context: Adapter at source level, can call source methods
    - Tensor context: Adapter at tensor level, can call tensor methods
    - get_tensor_adapter() transitions to tensor context
    - Single-tensor adapters (where get_tensor_adapter returns self) can call
      tensor methods in source context (no need for explicit transition)

    Required fields (subclasses must set these):
    - source_id: Data source identifier (stable across all adapters for this source)
    - _source_url: URL/path to the data source
    - _source_type: Source type identifier
    - _tensor_name: Optional tensor name (set by get_tensor_adapter for multi-tensor)

    The array_id property computes the tensor identifier used in chunk encoding:
    - Single-tensor: array_id == source_id
    - Multi-tensor: array_id == source_id/tensor_name
    """

    # Context tracking
    _tensor_context: bool = False

    # Single-tensor flag: True if get_tensor_adapter returns self
    # Set False for multi-tensor adapters like AicsImageIoAdapter
    _single_tensor_source: bool = True

    # Required fields
    source_id: str  # Data source identifier
    _source_url: str  # URL/path to the data source
    _source_type: str  # Source type identifier
    _tensor_name: Optional[str] = None  # Tensor name (for multi-tensor)

    @property
    def array_id(self) -> str:
        """Tensor identifier used in chunk encoding.

        For single-tensor adapters: returns source_id
        For multi-tensor adapters: returns source_id/tensor_name

        This is used in chunk_id encoding to identify which tensor the chunk belongs to.
        """
        if self._tensor_name is None:
            return self.source_id
        return f"{self.source_id}/{self._tensor_name}"

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

    # === Utility methods ===

    def get_metadata(self) -> dict:
        """Return metadata as dict. In most cases this is OME metadata.

        For OME-Zarr: returns parsed .zattrs (multiscales, axes, omero, etc.)
        For OME-TIFF: returns extracted OME-XML as JSON-compatible dict
        For plain Zarr/HDF5: returns empty dict

        Will be serialized to metadata_json in TensorDescriptor.
        Override in subclasses to provide format-specific metadata.
        """
        return {}


    # === flight methods ===

    @abstractmethod
    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors available in this source.

        [ANY method] Can be called in source or tensor context.

        This method is primarily for source listing/discovery. It returns
        lightweight descriptors without requiring expensive operations like
        scene switching or chunk layout computation.

        Returns:
            List of TensorDescriptor for all tensors in this source.

            Required fields:
            - array_id: Unique tensor identifier (for single-tensor: source_id;
              for multi-tensor: source_id/tensor_name)
            - shape: Tensor shape as list of ints

            Optional fields (populated by get_tensor_descriptor() for actual reads):
            - chunk_shape: Chunk shape. Can be empty [] if not readily available
              without expensive computation (e.g., multi-scene files).
              Clients should call get_tensor_adapter() + get_tensor_descriptor()
              for accurate chunk info before reading.
            - dtype: Data type string. Can be omitted if expensive to compute.
              Must be populated by get_tensor_descriptor() for actual reads.

            Recommended optional fields:
            - dim_labels: Dimension labels (cheap to include)
        """
        pass


    def get_source_descriptor(self) -> DataSourceDescriptor:
        """Build DataSourceDescriptor from this adapter.

        [SOURCE method] Must be called before get_tensor_adapter().

        Returns:
            DataSourceDescriptor.
        """
        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=self._source_url,
            source_type=self._source_type,
            tensors=self.list_tensor_descriptors(),
            metadata_json="",  # filled by GetFlightInfo()
        )


    def get_tensor_adapter(self, tensor_id: str|None) -> 'BackendAdapter':
        """Factory method to return adapter with specific tensor context.

        Transitions the adapter from source context to tensor context.
        Single-tensor adapters return self with tensor context set.
        Multi-tensor adapters override this to return a new adapter for the tensor.

        Args:
            tensor_id: Identifier for the specific tensor within this source

        Returns:
            BackendAdapter for the specified tensor, with tensor context set
        """
        # For single-tensor sources, just set context and return self
        self._tensor_context = True
        # Only set _tensor_name if it's different from source_id
        # (for multi-tensor sources or when tensor is a sub-component)
        if tensor_id and tensor_id != self.source_id:
            self._tensor_name = tensor_id
        return self


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
    def get_chunk_array(self, chunk_id: bytes) -> np.ndarray:
        """Read raw chunk data from backend as numpy array.

        [TENSOR method] Must be called after get_tensor_adapter().

        Args:
            chunk_id: Backend-specific chunk identifier (decoded from Flight ticket)

        Returns:
            Numpy array with the chunk's data in its native shape.
            The chunk shape is provided via ChunkBounds in app_metadata.

        Note: This method reads raw backend chunks, not split or virtual chunks.
        Split/virtual chunk resolution is handled by resolve_chunk_data().
        """
        pass


    @abstractmethod
    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the TensorDescriptor for this specific tensor adapter.

        [TENSOR method] Must be called after get_tensor_adapter().

        Field must be populated:
            - array_id: Unique tensor identifier (for single-tensor: source_id;
              for multi-tensor: source_id/tensor_name)
            - shape: Tensor shape as list of ints
            - chunk_shape: Chunk shape as list of ints
            - dtype: Data type string (numpy dtype.str format)
        Recommended fields to populate:
            - dim_labels: Dimension labels

        Returns:
            TensorDescriptor with required fields populated (see list_tensor_descriptors
            for field requirements).
        """
        pass


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
                if chunks_intersect(
                    list(ep.bounds.start), list(ep.bounds.stop),
                    slice_start, slice_stop
                )
            ]
            return filtered_endpoints

        return list(raw_endpoints)


    def get_arrow_schema(self, desc: Optional[TensorDescriptor] = None) -> pa.Schema:
        """Get the Arrow schema for this tensor.

        Schema format:
        - data: list<dtype> - flattened tensor elements per chunk
        - shape: list<int64> - shape tuple per chunk
        - dtype: string - numpy dtype string per chunk

        Each RecordBatch has 1 row per chunk, making data self-describing.

        Returns:
            Arrow Schema with data, shape, and dtype fields
        """
        import importlib.metadata

        desc = desc or self.get_tensor_descriptor()

        dtype = np.dtype(desc.dtype)
        data_field = pa.field("data", pa.list_(pa.from_numpy_dtype(dtype)))
        shape_field = pa.field("shape", pa.list_(pa.int64()))
        dtype_field = pa.field("dtype", pa.string())

        # Schema metadata: biopb version for compatibility tracking
        metadata = {
            "tensor_schema_version": importlib.metadata.version("biopb"),
        }

        return pa.schema([data_field, shape_field, dtype_field], metadata=metadata)


    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Plan a logical tensor read by delegating to adapter.
        """
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(dim) for dim in base_desc.shape)
        chunk_shape = tuple(int(dim) for dim in base_desc.chunk_shape)
        slice_hint = request_desc.slice_hint if request_desc.HasField('slice_hint') else None
        read_options = request_desc.read_options if request_desc.HasField('read_options') else None
        source_start, source_stop = normalized_slice_bounds(base_shape, slice_hint)
        scale_hint = normalized_scale_hint(base_shape, read_options)
        reduction_method = _normalize_reduction_method(
            read_options.reduction_method if read_options else None
        )
        ndim = len(base_shape)

        # Validate chunk_shape dimensionality
        if len(chunk_shape) != ndim:
            logger.error(
                f"chunk_shape dimensionality mismatch: tensor {base_desc.array_id} has "
                f"shape={base_shape} ({ndim}D) but chunk_shape={chunk_shape} ({len(chunk_shape)}D)"
            )
            raise ValueError(
                f"chunk_shape dimensionality mismatch: expected {ndim}, got {len(chunk_shape)}"
            )

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
                if needs_splitting(chunk_shape, base_desc.dtype):
                    shifted_bounds = ChunkBounds(
                        start=[int(endpoint.bounds.start[ax] - realized_start[ax]) for ax in range(ndim)],
                        stop=[int(endpoint.bounds.stop[ax] - realized_start[ax]) for ax in range(ndim)],
                    )
                    shifted_endpoint = ChunkEndpoint(chunk_id=endpoint.chunk_id, bounds=shifted_bounds)
                    sub_endpoints = split_endpoint(base_desc.array_id, shifted_endpoint, base_desc.dtype)
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

            # Always set slice_hint when user requested a slice, so client can crop correctly
            # This is needed even when realized bounds equal full bounds (single-chunk arrays)
            if slice_hint is not None:
                logical_desc.slice_hint.start[:] = list(realized_start)
                logical_desc.slice_hint.stop[:] = list(realized_stop)
            elif realized_start != tuple(0 for _ in range(ndim)) or realized_stop != base_shape:
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
            logical_chunk_shape_val = logical_chunk_shape(chunk_shape, scale_hint, logical_shape)
            output_dtype = _output_dtype(base_desc.dtype, reduction_method)

            logical_endpoints: List[ChunkEndpoint] = []

            def iter_virtual_chunks(dim: int = 0, logical_offset: Tuple[int, ...] = ()):
                if dim == ndim:
                    yield logical_offset
                    return
                axis_chunk = logical_chunk_shape_val[dim]
                axis_extent = logical_shape[dim]
                for axis_start in range(0, axis_extent, axis_chunk):
                    yield from iter_virtual_chunks(dim + 1, logical_offset + (axis_start,))

            for logical_start in iter_virtual_chunks():
                logical_stop = tuple(
                    min(logical_start[ax] + logical_chunk_shape_val[ax], logical_shape[ax])
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
                if needs_splitting(virtual_chunk_shape, output_dtype):
                    # Create virtual chunk payload (without split info first)
                    base_payload = encode_virtual_chunk_payload(
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
                    sub_endpoints = split_endpoint(base_desc.array_id, virtual_ep, output_dtype)
                    logical_endpoints.extend(sub_endpoints)
                else:
                    payload = encode_virtual_chunk_payload(
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
                chunk_shape=list(logical_chunk_shape_val),
                dtype=_output_dtype(base_desc.dtype, reduction_method),
            )

            # Always set slice_hint when user requested a slice, so client can crop correctly
            # This is needed even when snapped bounds equal full bounds (single-chunk arrays)
            if slice_hint is not None:
                logical_desc.slice_hint.start[:] = list(snapped_start)
                logical_desc.slice_hint.stop[:] = list(snapped_stop)
            elif snapped_start != tuple(0 for _ in range(ndim)) or snapped_stop != base_shape:
                logical_desc.slice_hint.start[:] = list(snapped_start)
                logical_desc.slice_hint.stop[:] = list(snapped_stop)

        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)

        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)    


    def resolve_chunk_data(
        self, chunk_id: bytes,  cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Resolve chunk data, handling split and virtual chunks.

        [TENSOR method] Must be called after get_tensor_adapter().

        This is the main entry point for DoGet. It handles:
        1. Raw backend chunks: delegates to get_chunk_array()
        2. Split chunks: reads parent chunk and slices
        3. Virtual chunks: computes downscaled data

        Args:
            chunk_id: Encoded chunk identifier (may contain split info or virtual payload)
            cache_manager: Optional cache for virtual/split chunks

        Returns:
            Arrow RecordBatch with chunk data
        """
        from biopb_tensor_server.cache import ArrowFileBackend

        array_id, backend_data, split_index, split_max = decode_chunk_id(chunk_id)

        is_virtual = backend_data.startswith(_VIRTUAL_CHUNK_MAGIC)
        is_file_backend = cache_manager is not None and isinstance(cache_manager.backend, ArrowFileBackend)

        should_cache = cache_manager is not None and (is_virtual or is_file_backend)

        logger.debug(
            f"resolve_chunk_data: array_id={array_id}, virtual={is_virtual}, "
            f"split={split_index}/{split_max}, cache={should_cache}"
        )

        # Use chunk_id directly as stable cache key (for both virtual and regular chunks)
        key_bytes = chunk_id

        def compute_fn():
            if is_virtual:
                source_start, _, valid_stop, _, _ = decode_virtual_chunk_payload(backend_data)
                parent_bounds = ChunkBounds(
                    start=list(source_start),
                    stop=list(valid_stop),
                )

                # compute_virtual_chunk now returns numpy array directly
                # The result is already downscaled, so result_arr.shape is the logical shape
                result_arr = compute_virtual_chunk(self, backend_data)

                if split_max > 1:
                    # Slice the numpy array directly
                    result_arr = slice_array(result_arr, parent_bounds, split_index, split_max)

                # For virtual chunks, use result_arr.shape (already downscaled)
                logical_shape = list(result_arr.shape)
            else:
                # For real chunks, get the numpy array directly
                result_arr = self.get_chunk_array(chunk_id)

                # Defensive reshape: backends may squeeze singleton dimensions
                # Get expected bounds from backend_data
                parent_bounds = get_chunk_bounds_from_backend_key(self, backend_data)
                expected_shape = tuple(
                    int(stop - start) for start, stop in zip(parent_bounds.start, parent_bounds.stop)
                )
                if result_arr.shape != expected_shape:
                    if result_arr.size == int(np.prod(expected_shape)):
                        result_arr = result_arr.reshape(expected_shape)
                    else:
                        raise ValueError(
                            f"Chunk data size mismatch: got {result_arr.size} elements "
                            f"but expected {int(np.prod(expected_shape))} for shape {expected_shape}"
                        )

                if split_max > 1:
                    result_arr = slice_array(result_arr, parent_bounds, split_index, split_max)

                # For real chunks, use logical shape from bounds (may differ from raw array shape)
                logical_shape = [int(stop - start) for start, stop in zip(parent_bounds.start, parent_bounds.stop)]

            # Convert to RecordBatch with 1 row per chunk
            # data: list<dtype> - flattened tensor elements
            # shape: list<int64> - LOGICAL chunk shape (from bounds for real chunks, from result for virtual)
            # dtype: string - numpy dtype string
            dtype_str = str(result_arr.dtype)
            result = pa.RecordBatch.from_arrays(
                [
                    pa.array([result_arr.ravel()]),  # list of flattened elements
                    pa.array([logical_shape]),       # logical shape
                    pa.array([dtype_str]),           # numpy dtype string
                ],
                ["data", "shape", "dtype"]
            )
            size_bytes = result_arr.nbytes
            logger.debug(f"resolve_chunk_data: computed {size_bytes} bytes")
            return result, size_bytes

        if should_cache:
            entry = cache_manager.get_or_acquire(key_bytes, compute_fn, metadata={'array_id': array_id})
            data = entry.data
            cache_manager.release(key_bytes)

        else:
            data, _ = compute_fn()

        return data