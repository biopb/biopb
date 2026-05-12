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
    decode_chunk_id,
    encode_chunk_id,
    encode_chunk_id_with_scale,
    is_scaled_chunk,
    decode_scale_info,
    needs_splitting,
    normalized_scale_hint,
    normalized_slice_bounds,
)
from biopb_tensor_server.downsample import (
    _ceil_div,
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
        'get_read_plan': MethodContext.TENSOR,
        'resolve_chunk_data': MethodContext.TENSOR,
        'get_arrow_schema': MethodContext.TENSOR,
        'get_chunk_size': MethodContext.TENSOR,
        'get_data': MethodContext.TENSOR,
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
        return None  # Default implementation claims nothing, override in subclasses

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

    def _validate_bounds(self, bounds: ChunkBounds, shape: Tuple[int, ...]) -> None:
        """Validate that bounds are within array shape.

        Args:
            bounds: Chunk bounds (start, stop coordinates)
            shape: Array shape

        Raises:
            ValueError: If bounds are out-of-bounds or invalid
        """
        ndim = len(shape)
        if len(bounds.start) != ndim or len(bounds.stop) != ndim:
            raise ValueError(
                f"Bounds dimensionality mismatch: expected {ndim}, "
                f"got start={len(bounds.start)}, stop={len(bounds.stop)}"
            )
        for ax, (s, e, dim) in enumerate(zip(bounds.start, bounds.stop, shape)):
            if s < 0:
                raise ValueError(f"Bounds start[{ax}]={s} is negative")
            if e > dim:
                raise ValueError(f"Bounds stop[{ax}]={e} exceeds shape[{ax}]={dim}")
            if s >= e:
                raise ValueError(f"Bounds start[{ax}]={s} >= stop[{ax}]={e}")


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


    def get_chunk_size(self) -> Tuple[int, ...]:
        """Return the chunk size for this tensor adapter.

        [TENSOR method] Must be called after get_tensor_adapter().

        Returns:
            Tuple of chunk dimensions (e.g., (64, 64, 64) for 3D chunks)
        """
        desc = self.get_tensor_descriptor()
        return tuple(int(dim) for dim in desc.chunk_shape)


    @abstractmethod
    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from the backend.

        [TENSOR method] Must be called after get_tensor_adapter().

        Subclasses should call super().get_data(bounds) to validate bounds,
        then read data from their backend.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape
        """
        desc = self.get_tensor_descriptor()
        shape = tuple(int(dim) for dim in desc.shape)
        self._validate_bounds(bounds, shape)


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


    def _compute_safe_chunk_size(
        self,
        chunk_size: Tuple[int, ...],
        dtype: str,
        dim_labels: Optional[List[str]],
    ) -> Tuple[int, ...]:
        """Compute a chunk size that fits within Arrow batch limit.

        Splits along semantic axis (same logic as old split_endpoint).
        Returns chunk size that is guaranteed to not need splitting.
        """
        from biopb_tensor_server.chunk import MAX_ARROW_BATCH_BYTES, _choose_split_axis

        item_size = np.dtype(dtype).itemsize
        chunk_bytes = int(np.prod(chunk_size)) * item_size

        if chunk_bytes <= MAX_ARROW_BATCH_BYTES:
            return chunk_size

        n_splits = int(np.ceil(chunk_bytes / MAX_ARROW_BATCH_BYTES))

        # Choose split axis using semantic priority
        split_axis = _choose_split_axis(chunk_size, dim_labels, n_splits)

        # Compute safe size on split axis
        safe_axis_size = chunk_size[split_axis] // n_splits

        safe_size = list(chunk_size)
        safe_size[split_axis] = safe_axis_size

        return tuple(safe_size)

    def _crop_and_downsample(
        self,
        arr: np.ndarray,
        scale_hint: Tuple[int, ...],
        reduction_method: str,
    ) -> np.ndarray:
        """Downsample array by scale_hint.

        Note: No padding needed because logical bounds are computed via floor_div //,
        meaning input is always properly aligned to scale_hint multiples.
        """
        from biopb_tensor_server.downsample import _downsample_block

        # Input shape is guaranteed to be multiple of scale_hint due to floor_div
        # Just downsample directly
        return _downsample_block(arr, scale_hint, reduction_method)

    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Plan a logical tensor read using uniform chunk grid.

        Key ordering: Check splitting FIRST, then compute virtual_chunk_size
        from the split-safe chunk_size. This ensures scaled chunks never need
        re-checking for splitting.
        """
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(dim) for dim in base_desc.shape)
        chunk_size = self.get_chunk_size()
        slice_hint = request_desc.slice_hint if request_desc.HasField('slice_hint') else None
        read_options = request_desc.read_options if request_desc.HasField('read_options') else None

        # Normalize inputs
        source_start, source_stop = normalized_slice_bounds(base_shape, slice_hint)
        scale_hint = normalized_scale_hint(base_shape, read_options)
        reduction_method = _normalize_reduction_method(
            read_options.reduction_method if read_options else None
        )
        ndim = len(base_shape)

        # STEP 1: Check if base chunk_size needs splitting FIRST
        if needs_splitting(chunk_size, base_desc.dtype):
            # Split chunk_size to safe_sub_chunk_size
            safe_chunk_size = self._compute_safe_chunk_size(
                chunk_size, base_desc.dtype, base_desc.dim_labels
            )
        else:
            safe_chunk_size = chunk_size

        # STEP 2: Now compute virtual_chunk_size from safe_chunk_size
        # (guaranteed to not need splitting since base is already safe)
        if scale_hint is None:
            virtual_chunk_size = safe_chunk_size
            logical_chunk_size = safe_chunk_size
            output_dtype = base_desc.dtype
        else:
            virtual_chunk_size = tuple(lcm(safe_chunk_size[ax], scale_hint[ax]) for ax in range(ndim))
            logical_chunk_size = tuple(virtual_chunk_size[ax] // scale_hint[ax] for ax in range(ndim))
            output_dtype = _output_dtype(base_desc.dtype, reduction_method)

        # Snap bounds to virtual_chunk_size grid
        realized_start = tuple(
            (source_start[ax] // virtual_chunk_size[ax]) * virtual_chunk_size[ax]
            for ax in range(ndim)
        )
        realized_stop = tuple(
            min(_ceil_div(source_stop[ax], virtual_chunk_size[ax]) * virtual_chunk_size[ax], base_shape[ax])
            for ax in range(ndim)
        )
        realized_shape = tuple(realized_stop[ax] - realized_start[ax] for ax in range(ndim))

        # Compute logical shape (for scale_hint case)
        if scale_hint is not None:
            logical_shape = tuple(_ceil_div(realized_shape[ax], scale_hint[ax]) for ax in range(ndim))
        else:
            logical_shape = realized_shape

        # Generate chunk endpoints by iterating grid using np.ndindex
        # NO NEED to check splitting here - safe_chunk_size ensures it's always safe
        logical_endpoints: List[ChunkEndpoint] = []

        # Compute number of chunks along each axis
        n_chunks_per_axis = tuple(
            _ceil_div(realized_stop[ax] - realized_start[ax], virtual_chunk_size[ax])
            for ax in range(ndim)
        )

        # Iterate over chunk grid
        for chunk_idx in np.ndindex(*n_chunks_per_axis):
            virtual_start = tuple(
                realized_start[ax] + chunk_idx[ax] * virtual_chunk_size[ax]
                for ax in range(ndim)
            )
            virtual_stop = tuple(
                min(virtual_start[ax] + virtual_chunk_size[ax], base_shape[ax])
                for ax in range(ndim)
            )

            # Compute logical bounds for this chunk
            if scale_hint is not None:
                logical_start = tuple(
                    (virtual_start[ax] - realized_start[ax]) // scale_hint[ax]
                    for ax in range(ndim)
                )
                logical_stop = tuple(
                    _ceil_div(virtual_stop[ax] - realized_start[ax], scale_hint[ax])
                    for ax in range(ndim)
                )
            else:
                logical_start = tuple(virtual_start[ax] - realized_start[ax] for ax in range(ndim))
                logical_stop = tuple(virtual_stop[ax] - realized_start[ax] for ax in range(ndim))

            virtual_bounds = ChunkBounds(start=list(virtual_start), stop=list(virtual_stop))
            logical_bounds = ChunkBounds(start=list(logical_start), stop=list(logical_stop))

            # NO splitting check needed - safe_chunk_size guarantees it fits

            # Encode: array_id + virtual_bounds + optional scale_hint
            if scale_hint is not None:
                chunk_id = encode_chunk_id_with_scale(
                    base_desc.array_id, virtual_bounds, scale_hint, reduction_method
                )
            else:
                chunk_id = encode_chunk_id(base_desc.array_id, virtual_bounds)

            logical_endpoints.append(ChunkEndpoint(chunk_id=chunk_id, bounds=logical_bounds))

        # Build descriptor
        logical_desc = TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=list(logical_shape),
            chunk_shape=list(logical_chunk_size),
            dtype=output_dtype,
        )

        # Set slice_hint for client cropping
        if slice_hint is not None or realized_start != tuple(0 for _ in range(ndim)):
            logical_desc.slice_hint.start[:] = list(realized_start)
            logical_desc.slice_hint.stop[:] = list(realized_stop)

        if read_options is not None:
            logical_desc.read_options.CopyFrom(read_options)

        return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)    


    def resolve_chunk_data(
        self, chunk_id: bytes,
        cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Resolve chunk data, handling scaled chunks.

        Uses get_data() directly for all reads instead of complex chunk_array logic.
        """
        from biopb_tensor_server.cache import ArrowFileBackend

        array_id, bounds = decode_chunk_id(chunk_id)

        # Check if scaled chunk (has extra bytes after bounds encoding)
        is_scaled_chunk_flag = is_scaled_chunk(chunk_id)

        should_cache = cache_manager is not None and (
            is_scaled_chunk_flag or isinstance(cache_manager.backend, ArrowFileBackend)
        )

        logger.debug(
            f"resolve_chunk_data: array_id={array_id}, scaled={is_scaled_chunk_flag}, "
            f"bounds_start={list(bounds.start)}, bounds_stop={list(bounds.stop)}, cache={should_cache}"
        )

        def compute_fn():
            # Read data directly using get_data()
            result_arr = self.get_data(bounds)

            if is_scaled_chunk_flag:
                scale_hint, reduction_method = decode_scale_info(chunk_id)
                # Crop and downsample (no padding needed - bounds aligned via floor_div)
                result_arr = self._crop_and_downsample(
                    result_arr, scale_hint, reduction_method
                )

            logical_shape = list(result_arr.shape)
            dtype_str = str(result_arr.dtype)

            result = pa.RecordBatch.from_arrays(
                [
                    pa.array([result_arr.ravel()]),
                    pa.array([logical_shape]),
                    pa.array([dtype_str]),
                ],
                ["data", "shape", "dtype"]
            )
            return result, result_arr.nbytes

        if should_cache:
            entry = cache_manager.get_or_acquire(chunk_id, compute_fn, metadata={'array_id': array_id})
            data = entry.data
            cache_manager.release(chunk_id)
        else:
            data, _ = compute_fn()

        return data