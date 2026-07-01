"""Backend adapters for tensor storage formats.

This module provides a consistent interface for reading chunked multi-dimensional
arrays from various storage backends (Zarr, HDF5, OME-TIFF, TileDB).

Each adapter maps storage-specific chunk layouts to Arrow Flight endpoints:
- chunk_id: Opaque bytes identifying a chunk in the backend
- ChunkBounds: Array coordinates (start, stop) for the chunk

The adapters integrate with Arrow Flight's GetFlightInfo/DoGet flow:
1. GetFlightInfo returns FlightEndpoints with chunk_id tickets
2. DoGet uses the chunk_id to fetch the actual data

Caching behavior depends on the configured CacheManager backend:
- memory cache stores computed virtual chunks (for example scaled reads)
- file cache stores both virtual chunks and raw chunks as mmap-backed Arrow batches
    keyed by chunk_id
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    TensorDescriptor,
)
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.chunk import (
    ChunkEndpoint,
    compute_safe_chunk_size,
    decode_chunk_id,
    decode_scale_info,
    encode_chunk_id,
    encode_chunk_id_with_scale,
    is_scaled_chunk,
    needs_splitting,
    normalized_scale_hint,
    normalized_slice_bounds,
)
from biopb_tensor_server.downsample import (
    ceil_div,
    downsample_block,
    get_output_dtype,
    normalize_reduction_method,
)
from biopb_tensor_server.errors import SourceUnresolvedError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb.tensor.descriptor_pb2 import PyramidLevel

    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import ClaimContext, DiscoveryState, SourceClaim


# A real URL scheme is 2+ chars followed by "://" (so a bare Windows drive
# "C:\..." — one char then ":" then "\" — never matches).
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+://")
# A forward-slashed Windows absolute path: "C:/Users/...".
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:/")


def to_catalog_url(raw: str) -> str:
    """Normalize a source's URL for the catalog (the descriptor's ``source_url``).

    Local filesystem paths are rewritten to a forward-slash ``file://`` form so a
    Windows-indexed catalog is byte-for-byte consistent with a POSIX one and every
    consumer can split on ``/`` alone (biopb/biopb#131)::

        C:\\Users\\me\\Screenshots 1\\img.png  ->  file:///C:/Users/me/Screenshots 1/img.png
        /data/cells/img.tif                    ->  file:///data/cells/img.tif

    Separators only: spaces / unicode are left literal for readability, not
    percent-encoded. Already-schemed URLs are returned unchanged — remote stores
    (s3://, http://, …), an existing ``file://``, and virtual schemes (cache://).

    This is display/catalog only and must not be fed back to the filesystem
    (callers keep the raw path for that). ``source_id`` is unaffected: it hashes
    the resolved path, not this string, so the normalization needs no re-index.
    """
    if not raw:
        return raw
    if _URL_SCHEME_RE.match(raw):
        return raw  # already a URL (remote / file:// / cache:// / …)
    fwd = raw.replace("\\", "/")
    if _WIN_DRIVE_RE.match(fwd):
        return "file:///" + fwd  # Windows absolute -> file:///C:/...
    if fwd.startswith("/"):
        return "file://" + fwd  # POSIX absolute -> file:///data/... (// + /data)
    return "file:///" + fwd.lstrip("/")  # relative / other: best effort


@dataclass
class TensorReadPlan:
    """Logical tensor read plan returned by the server planning layer."""

    descriptor: TensorDescriptor
    chunk_endpoints: List[ChunkEndpoint]


def require_resolved(desc: TensorDescriptor) -> None:
    """Guard the read-planning boundary against an unresolved descriptor.

    A descriptor is "resolved" iff it carries a concrete shape and dtype. An
    unresolved one (e.g. a not-yet-hydrated cloud source) would otherwise crash
    deep in the planner -- ``np.dtype("")`` in ``get_arrow_schema`` or the
    pyramid/ndim logic in ``chunk`` -- with raw, illegible errors. Fail here
    with a clean ``SourceUnresolvedError`` instead.
    """
    if not desc.shape or not desc.dtype:
        raise SourceUnresolvedError(
            f"tensor {desc.array_id!r} is unresolved (shape/dtype unknown) -- "
            f"open the source to resolve it"
        )


class SourceAdapter(ABC):
    """Abstract base class for source-level adapters.

    Each adapter handles a specific storage format (Zarr, HDF5, OME-TIFF, etc.)
    and provides methods to discover tensors and read metadata.

    Adapters must implement the claim() method to detect if they handle a given path.
    """

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

    @classmethod
    def claim(cls, ctx: ClaimContext, state: DiscoveryState) -> Optional[SourceClaim]:
        """Claim a filesystem path as a data source.

        This method is called during discovery to detect if this adapter
        handles a given path. Adapters should check for format-specific
        characteristics (file extensions, metadata files, etc.).

        Multi-file sources should use state.try_claim_path() for each
        path they want to claim.

        Args:
            ctx: ClaimContext for unified filesystem access (local or remote)
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim if this adapter handles this path, None otherwise
        """
        return None  # Default implementation claims nothing, override in subclasses

    @classmethod
    @abstractmethod
    def create_from_config(
        cls, source: SourceConfig, credentials_config: Optional[Any] = None
    ) -> SourceAdapter:
        """Create adapter instance from SourceConfig.

        This is used by the server to instantiate adapters based on discovery claims.

        Args:
            source: SourceConfig with url, source_id, dim_labels, and format-specific options
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            An instance of a SourceAdapter subclass initialized with the provided config
        """

    @abstractmethod
    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List all tensors available in this source.

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

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return metadata as dict. In most cases this is OME metadata.

        Used by the metadata engine to create database.
        Will be serialized to metadata_json in TensorDescriptor.
        """

    def metadata_covers_all_tensors(self) -> bool:
        """Whether ``get_metadata()`` applies to *every* tensor of this source.

        When True (the default), the catalog's source-level ``metadata_json`` is a
        valid serve answer for any tensor, so ``GetFlightInfo(with_metadata)`` may
        read it from the catalog instead of recomputing on the adapter
        (biopb/biopb#253). Override to False when metadata is genuinely per-tensor
        (OME-Zarr HCS plates: the source row holds the *plate* ``.zattrs``, which
        is not any individual field's OME metadata) so the serve path falls back
        to the per-tensor ``get_metadata()`` for correctness.
        """
        return True

    def get_source_descriptor(self) -> DataSourceDescriptor:
        """Build DataSourceDescriptor from this adapter.

        Returns:
            DataSourceDescriptor.
        """
        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=to_catalog_url(self._source_url),
            source_type=self._source_type,
            tensors=self.list_tensor_descriptors(),
            metadata_json="",  # filled by GetFlightInfo()
            data_resident=self.is_resident(),
        )

    def resolve(self) -> DataSourceDescriptor:
        """Hydrate this source if needed, then return its full descriptor.

        This is the ONE consented entry point that may perform an extended,
        blocking recall (e.g. downloading a whole cloud / synced-folder file).
        It is the sole resolution trigger: the serve paths (get_tensor_adapter ->
        GetFlightInfo / DoGet) never resolve on their own -- they raise
        SourceUnresolvedError on an unresolved source so the only thing that
        downloads is an explicit ``resolve``.

        For an already-resident source this is a cheap no-op that just returns the
        current descriptor (idempotent), so the server's ``resolve`` action works
        uniformly across all source kinds. ``UnresolvedSourceAdapter`` overrides
        it to actually hydrate.
        """
        return self.get_source_descriptor()

    def is_resident(self) -> bool:
        """Best-effort, recall-free: is this source's content local and cheap to
        read right now?

        Remote (fsspec) sources are never resident until their pixels are
        materialized into a local copy (a later phase); a local source is
        resident unless it is an offline cloud placeholder. This is the
        authoritative, point-in-time residency gate -- VOLATILE, so evaluate it
        at the moment of use and never cache the result. ``data_resident`` on the
        descriptor is only an advisory snapshot of this.
        """
        # Lazy import: base <-> discovery only cross-import under TYPE_CHECKING,
        # so importing these at module scope would be circular.
        from pathlib import Path

        from biopb_tensor_server.discovery import (
            _is_offline_placeholder,
            is_remote_url,
        )

        if is_remote_url(self._source_url):
            return False
        path = Path(self._source_url)
        # The offline-placeholder signal (st_blocks == 0) is a per-*file* concept
        # -- discovery only consults it for files (see should_skip_walk_entry,
        # which gates it on `not is_dir`). A directory-based source (zarr,
        # ome-zarr store) legitimately reports st_blocks == 0 on some filesystems
        # (e.g. macOS APFS), so applying the file check to it would wrongly flag
        # an entirely local store as non-resident. Treat a directory as resident.
        if path.is_dir():
            return True
        return not _is_offline_placeholder(path)

    def get_tensor_adapter(self, tensor_id: str | None) -> TensorAdapter:
        """Factory method to return adapter with specific tensor context.

        Transitions the adapter from source context to tensor context.
        Single-tensor adapters return self with tensor context set.
        Multi-tensor adapters override this to return a new adapter for the tensor.

        Args:
            tensor_id: Identifier for the specific tensor within this source
        Returns:
            TensorAdapter for the specified tensor, with tensor context set
        """
        return self

    def _within_source_field(self, tensor_id: Optional[str]) -> Optional[str]:
        """Reduce a source-qualified array_id to its within-source field.

        Per the tensor identity policy, a tensor's array_id is ``source_id`` or
        ``source_id/field``. Multi-tensor ``get_tensor_adapter`` overrides key on
        the ``field`` part, but a caller may legitimately hand them the full
        array_id ("a tensor is identifiable by array_id alone"). Strip the
        ``source_id/`` prefix when present (split on the first '/'); a bare field
        is returned unchanged. Idempotent with the server's own reduction.
        """
        if tensor_id and self.source_id and tensor_id.startswith(f"{self.source_id}/"):
            return tensor_id[len(self.source_id) + 1 :]
        return tensor_id

    def has_native_pyramid(self) -> bool:
        """Whether this source ships a well-formed multi-resolution pyramid.

        Default False. Formats that natively store precomputed downsampled
        levels (e.g. OME-Zarr multiscales) override this to report True, which
        lets the precache worker skip them -- they already serve overviews
        cheaply from their own coarse levels.
        """
        return False

    def get_native_pyramid_levels(
        self, tensor_id: Optional[str] = None
    ) -> Optional[List[PyramidLevel]]:
        """Native (precomputed on-disk) pyramid levels for *tensor_id*, or None.

        Returns ``None`` for formats without a real on-disk pyramid (the default),
        in which case the server advertises a *computed* pyramid via
        ``chunk.build_pyramid_plan``. Formats that store downsampled levels
        natively (e.g. OME-Zarr multiscales) override this to return one
        ``PyramidLevel`` per native dataset, each with ``native=True`` and
        ``reduction_method="precompute"`` so the client requests the on-disk level
        directly. Each level's ``scale_hint`` MUST be the value the adapter's own
        ``get_read_plan`` "precompute" routing matches on, so an advertised level
        round-trips to its dataset.
        """
        return None

    def get_physical_scale(
        self, tensor_id: Optional[str] = None
    ) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dimension physical pixel size + unit, source axis order.

        Returns ``(scale, unit)``: two equal-length lists aligned 1:1 with the
        ``dim_labels`` this source's ``get_tensor_descriptor()`` emits (so the
        server can copy them straight onto ``TensorDescriptor.physical_scale`` /
        ``physical_unit`` without remapping). Element ``i`` is the physical
        extent of one sample along dimension ``i``; ``0.0`` / ``""`` mark a
        dimension with no known physical size (e.g. T/C axes).

        Returns ``None`` when no physical sizes are known. This is the compact
        ~200-byte summary the tensor-load hot path needs (issue #31), so it must
        be **cheap** -- read it straight off the resident metadata model, never
        a full ``get_metadata()`` dump. Default ``None``; format adapters that
        carry physical voxel sizes override it.
        """
        return None


class TensorAdapter(ABC):
    """Abstract base class for tensor-level adapters.

    This interface provides methods to read specific tensors, get chunk layouts,
    and read chunk data. It is returned by get_tensor_adapter() on the source adapter.

    Tensor-level adapters are created for specific tensors within a source, allowing
    them to maintain tensor-specific state (e.g., current scene in multi-scene files).
    """

    @abstractmethod
    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the TensorDescriptor for this specific tensor adapter.

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

    def get_chunk_size(self) -> Tuple[int, ...]:
        """Return the chunk size for this tensor adapter.

        Returns:
            Tuple of chunk dimensions (e.g., (64, 64, 64) for 3D chunks)
        """
        desc = self.get_tensor_descriptor()
        return tuple(int(dim) for dim in desc.chunk_shape)

    @abstractmethod
    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from the backend.
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
        require_resolved(desc)

        dtype = np.dtype(desc.dtype)
        data_field = pa.field("data", pa.list_(pa.from_numpy_dtype(dtype)))
        shape_field = pa.field("shape", pa.list_(pa.int64()))
        dtype_field = pa.field("dtype", pa.string())

        # Schema metadata: server version for compatibility tracking and feature detection
        metadata = {
            "tensor_schema_version": importlib.metadata.version("biopb-tensor-server"),
        }

        return pa.schema([data_field, shape_field, dtype_field], metadata=metadata)

    def resolve_chunk_data(
        self,
        chunk_id: bytes,
        cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Resolve chunk data, handling scaled chunks and backend caching.

        The default implementation reads raw chunk data with ``self.get_data()``.
        Scaled chunks are always cacheable when a CacheManager is available.
        With the file-backed Arrow cache, raw chunks are also cached by chunk_id.
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
            result_arr = self.get_data(bounds)

            if is_scaled_chunk_flag:
                scale_hint, reduction_method = decode_scale_info(chunk_id)
                # Crop and downsample (no padding needed - bounds aligned via floor_div)
                result_arr = downsample_block(result_arr, scale_hint, reduction_method)

            logical_shape = list(result_arr.shape)
            dtype_str = str(result_arr.dtype)

            result = pa.RecordBatch.from_arrays(
                [
                    pa.array([result_arr.ravel()]),
                    pa.array([logical_shape]),
                    pa.array([dtype_str]),
                ],
                ["data", "shape", "dtype"],
            )
            return result, result_arr.nbytes

        if should_cache:
            entry = cache_manager.get_or_acquire(
                chunk_id, compute_fn, metadata={"array_id": array_id}
            )
            data = entry.data
            cache_manager.release(chunk_id)
        else:
            data, _ = compute_fn()

        return data

    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Generate a read plan for the requested tensor descriptor.

        Default implementation uses uniform chunk grid planning.

        Args:
            request_desc: TensorDescriptor from the client's read request, which may
                          include slice_hint and scale_hint/reduction_method directly.
        Returns:
            TensorReadPlan with the logical descriptor and list of chunk endpoints to read.
        """
        base_desc = self.get_tensor_descriptor()
        chunk_size = self.get_chunk_size()
        return _get_read_plan(base_desc, request_desc, chunk_size)


class BackendAdapter(SourceAdapter, TensorAdapter):
    pass


def _get_read_plan(
    base_desc: TensorDescriptor,
    request_desc: TensorDescriptor,
    chunk_size: Tuple[int, ...],
) -> TensorReadPlan:
    """Plan a logical tensor read using uniform chunk grid.

    Plan try to maintain a uniform chunk grid aligned with the base chunk_size, but may adjust chunk size if raw chunks are too
    large to read in one go (e.g., due to Arrow IPC limits).
    """
    require_resolved(base_desc)
    base_shape = tuple(int(dim) for dim in base_desc.shape)
    slice_hint = (
        request_desc.slice_hint if request_desc.HasField("slice_hint") else None
    )

    # Normalize inputs - use scale_hint/reduction_method directly from TensorDescriptor
    source_start, source_stop = normalized_slice_bounds(base_shape, slice_hint)
    scale_hint = normalized_scale_hint(base_shape, request_desc.scale_hint)
    reduction_method = normalize_reduction_method(request_desc.reduction_method)
    ndim = len(base_shape)

    # STEP 1: Check if base chunk_size needs splitting FIRST
    if needs_splitting(chunk_size, base_desc.dtype):
        # Split chunk_size to safe_sub_chunk_size
        safe_chunk_size = compute_safe_chunk_size(
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
        virtual_chunk_size = tuple(
            lcm(safe_chunk_size[ax], scale_hint[ax]) for ax in range(ndim)
        )
        logical_chunk_size = tuple(
            virtual_chunk_size[ax] // scale_hint[ax] for ax in range(ndim)
        )
        output_dtype = get_output_dtype(base_desc.dtype, reduction_method)

    # Snap bounds to virtual_chunk_size grid
    realized_start = tuple(
        (source_start[ax] // virtual_chunk_size[ax]) * virtual_chunk_size[ax]
        for ax in range(ndim)
    )
    realized_stop = tuple(
        min(
            ceil_div(source_stop[ax], virtual_chunk_size[ax]) * virtual_chunk_size[ax],
            base_shape[ax],
        )
        for ax in range(ndim)
    )
    realized_shape = tuple(realized_stop[ax] - realized_start[ax] for ax in range(ndim))

    # Compute logical shape (for scale_hint case)
    if scale_hint is not None:
        logical_shape = tuple(
            ceil_div(realized_shape[ax], scale_hint[ax]) for ax in range(ndim)
        )
    else:
        logical_shape = realized_shape

    # Generate chunk endpoints by iterating grid using np.ndindex
    # NO NEED to check splitting here - safe_chunk_size ensures it's always safe
    logical_endpoints: List[ChunkEndpoint] = []

    # Compute number of chunks along each axis
    n_chunks_per_axis = tuple(
        ceil_div(realized_stop[ax] - realized_start[ax], virtual_chunk_size[ax])
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
                ceil_div(virtual_stop[ax] - realized_start[ax], scale_hint[ax])
                for ax in range(ndim)
            )
        else:
            logical_start = tuple(
                virtual_start[ax] - realized_start[ax] for ax in range(ndim)
            )
            logical_stop = tuple(
                virtual_stop[ax] - realized_start[ax] for ax in range(ndim)
            )

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

        logical_endpoints.append(
            ChunkEndpoint(chunk_id=chunk_id, bounds=logical_bounds)
        )

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

    # Copy scale_hint and reduction_method to logical descriptor
    if scale_hint is not None:
        logical_desc.scale_hint[:] = list(scale_hint)
    if reduction_method:
        logical_desc.reduction_method = reduction_method

    return TensorReadPlan(descriptor=logical_desc, chunk_endpoints=logical_endpoints)
