"""Backend adapters for tensor storage formats.

This module provides a consistent interface for reading chunked multi-dimensional
arrays from various storage backends (Zarr, OME-TIFF, TileDB).

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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pyarrow as pa
from biopb.tensor.descriptor_pb2 import (
    SliceHint,
    TensorDescriptor,
)
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.cache_source import cache_sourced_units
from biopb_tensor_server.core.chunk import (
    ChunkEndpoint,
    array_id_from_chunk_id,
    build_pyramid_plan,
    cache_key_for_chunk_id,
    compute_safe_chunk_size,
    current_epoch,
    decode_chunk_id,
    decode_reduction_method,
    decode_scale_info,
    is_scaled_chunk,
    mint_chunk_id,
    normalized_scale_hint,
    normalized_slice_bounds,
    scaled_virtual_chunk_size,
    split_chunk_version as _split_chunk_version,
)
from biopb_tensor_server.core.chunk_batch import (
    CHUNK_WIRE_SCHEMA,
    pack_chunk_batch,
)
from biopb_tensor_server.core.downsample import (
    ceil_div,
    downsample_block,
    get_output_dtype,
    normalize_reduction_method,
)
from biopb_tensor_server.core.errors import (
    SourceUnresolvedError,
    StaleChunkError,
    TensorNotFound,
    WriteNotSupportedError,
)
from biopb_tensor_server.core.normalize import (
    permutation_of,
    permute_descriptor,
    to_canonical,
    to_native,
)
from biopb_tensor_server.core.read_mask import ENDPOINTS, PYRAMID, read_mask
from biopb_tensor_server.core.retention import (
    computed_ladder,
    record_decode,
    retention_for_array,
    retention_for_scale,
)
from biopb_tensor_server.core.stream_reduce import (
    stream_reduce,
    streaming_unit,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorReadOption

    from biopb_tensor_server.cache import CacheManager, ChunkLocation, RetentionClass
    from biopb_tensor_server.core.config import PyramidConfig, SourceConfig
    from biopb_tensor_server.core.discovery import (
        ClaimContext,
        DiscoveryState,
        SourceClaim,
    )


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


def strip_source_prefix(source_id: str, array_id: Optional[str]) -> Optional[str]:
    """Reduce an ``array_id`` to its within-source field by removing the
    ``source_id/`` prefix.

    Pure and total: ``None``/``""`` pass through, and an id with no prefix -- a
    bare field, or the source_id itself for a single-tensor source -- is returned
    unchanged. It carries **no** "default tensor" policy; that decision belongs to
    the caller (only the server's ``_field_within_source`` maps the no-field cases
    to ``None`` = default tensor).

    Identity policy (proto/biopb/tensor/descriptor.proto): array_id is
    ``source_id`` or ``source_id/field``; source_id is slash-free, so the first
    '/' is the source boundary and the field may itself contain '/'.
    """
    if array_id and source_id and array_id.startswith(f"{source_id}/"):
        return array_id[len(source_id) + 1 :]
    return array_id


def bounds_to_slices(bounds: ChunkBounds) -> Tuple[slice, ...]:
    """Per-axis ``slice`` tuple for indexing a backend array with ``bounds``."""
    return tuple(
        slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop, strict=True)
    )


def bounds_to_strided_slices(
    bounds: ChunkBounds, step: Tuple[int, ...]
) -> Tuple[slice, ...]:
    """:func:`bounds_to_slices` with a per-axis stride, for a decimated read.

    Kept next to its unstrided sibling so the two index a store identically
    apart from the step -- which is the whole of what makes a fused
    ``nearest`` bit-identical to reading the extent and slicing it.
    """
    return tuple(
        slice(int(s), int(e), max(1, int(size)))
        for s, e, size in zip(bounds.start, bounds.stop, step, strict=True)
    )


def validate_bounds(bounds: ChunkBounds, shape: Tuple[int, ...]) -> None:
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
    for ax, (s, e, dim) in enumerate(
        zip(bounds.start, bounds.stop, shape, strict=True)
    ):
        if s < 0:
            raise ValueError(f"Bounds start[{ax}]={s} is negative")
        if e > dim:
            raise ValueError(f"Bounds stop[{ax}]={e} exceeds shape[{ax}]={dim}")
        if s >= e:
            raise ValueError(f"Bounds start[{ax}]={s} >= stop[{ax}]={e}")


@dataclass(frozen=True)
class TensorEntry:
    """What a *source* lists about one of its tensors: the **structural** facts.

    ``array_id`` / ``dim_labels`` / ``shape`` / ``dtype`` -- stable per tensor,
    derivable from the container's own index, and what the DuckDB
    ``sources.tensors`` rows carry.
    """

    array_id: str
    dim_labels: Tuple[str, ...] = ()
    shape: Tuple[int, ...] = ()
    dtype: str = ""


def catalog_entry(desc: Any) -> TensorEntry:
    """Project a descriptor (or anything shaped like one) onto a :class:`TensorEntry`."""
    return TensorEntry(
        array_id=desc.array_id,
        dim_labels=tuple(str(label) for label in desc.dim_labels),
        shape=tuple(int(dim) for dim in desc.shape),
        dtype=desc.dtype,
    )


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

    Each adapter handles a specific storage format (Zarr, OME-TIFF, etc.)
    and provides methods to discover tensors and read metadata.

    Adapters that participate in filesystem auto-discovery override the claim()
    classmethod to detect whether they handle a given path; it is not abstract
    (the default claims nothing) so a config-only format like a remote
    tensor-server can opt out.
    """

    # Required fields
    source_id: str  # Data source identifier
    _source_url: Optional[str] = None  # URL/path to the data source
    _source_type: Optional[str] = None  # Source type identifier

    # Optional content-version token (biopb/biopb#178), folded into every
    # chunk_id this adapter mints and hence into the cache key, so a
    # re-registered source with new bytes gets a fresh cache namespace instead
    # of serving stale chunks. None means unversioned. Opaque -- the codec
    # namespaces by it, never reads it.
    # Declared on the source because the value is per TENSOR. A tensor whose bytes
    # live elsewhere carries its own (e.g., uploaded label set).
    _content_version: Optional[bytes] = None

    # Display-only override for the catalog ``source_url``. Optional, because the
    # adapter could use the path string, ``_source_url``.
    # The ``register_local_path`` path sets it to a re-rooted url so each drop
    # renders as its own top-level root instead of nesting deep under the shared
    # absolute-path tree (see SourceManager._drop_catalog_url).
    _catalog_url: Optional[str] = None

    @property
    def source_url(self) -> Optional[str]:
        """The source's real, addressable URL/path: a filesystem path this
        adapter reads bytes from, or the dial address of an upstream it
        proxies.
        """
        return self._source_url

    @property
    def source_type(self) -> Optional[str]:
        """Format/source-type identifier (e.g. ``"ome_zarr"``)."""
        return self._source_type

    @property
    def catalog_url(self) -> str:
        """The display URL the catalog row carries -- what clients group the tree by."""
        return self._catalog_url or to_catalog_url(self._source_url)

    def check_readable(self) -> None:  # noqa: B027 - concrete no-op default
        """Raise if this source cannot answer a pixel read right now.

        Usually a noop. An unfinished upload is a notable exception.
        """

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
            source: SourceConfig with url, source_id, and format-specific options
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            An instance of a SourceAdapter subclass initialized with the provided config
        """

    @abstractmethod
    def list_tensors(self) -> List[TensorEntry]:
        """List this source's tensors as **structural catalog entries**.

        The source-listing/discovery surface: what the DuckDB catalog stores in
        ``sources.tensors``. It returns lightweight entries without expensive
        operations like scene switching or chunk-layout computation.

        Returns:
            List of :class:`TensorEntry`, each a :func:`catalog_entry` projection:

        A single-tensor source returns ``[catalog_entry(self._native_descriptor())]``.
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return the source-level metadata as a dict. Usually OME metadata.

        Called once per registered adapter, by
        :meth:`MetadataDatabase.sync_source_added`, to populate
        ``sources.metadata_json``; a changed file or a source that resolves
        registers a new adapter. (A remote mirror's re-seed and a rolled-back
        replace sync the same adapter again.) The serve path reads that row back,
        never this method (biopb/biopb#253), and nothing else calls it: an adapter
        that needs a value from its metadata keeps a private copy of that value.
        The catalog is the cache, so this need not memoize. Genuinely per-tensor
        metadata that the source row cannot represent is exposed on the tensor
        adapter via :meth:`TensorAdapter.get_tensor_metadata` instead.
        """

    def get_embedded_rois(
        self,
        metadata: Mapping[str, Any],
        tensors: Sequence[Tuple[str, Sequence[str]]],
        *,
        max_per_tensor: Optional[int] = None,
    ) -> Tuple[Dict[str, List[Any]], Any]:
        """ROIs this source's own file carries, keyed by ``array_id``.

        Some formats store annotations beside their pixels -- OME-XML ``<ROI>``
        elements, ImageJ overlays, a GeoJSON sidecar. Those land in the reserved
        ``@ome``-style set the catalog keeps read-only (biopb/biopb#951).

        Returns:
            ``(rois_by_array_id, report)``. The report is opaque to the caller
            beyond having a ``summary()`` for the log, and may be ``None``.

        Raising is not fatal but IS a bug: the caller runs this inside source
        registration and swallows failures, because a source is its pixels first
        and an imported set is rebuilt on the next registration anyway.
        """
        return {}, None

    def is_resolved(self) -> bool:
        """Deterministic: is there a hydrated adapter backing this source?

        True by default; only a remote proxy mirroring an unresolved upstream
        source overrides it. A source of this server that is not resolved has no
        adapter at all, only a catalog row (``sources.unresolved_reason`` says why). Unlike
        residency (:func:`~biopb_tensor_server.core.discovery.source_is_resident`),
        this never flips back to False once True.
        """
        return True

    def get_tensor_adapter(self, tensor_id: str | None) -> TensorAdapter:
        """Factory method to return adapter with specific tensor context.

        Transitions the adapter from source context to tensor context.
        Single-tensor adapters return self with tensor context set -- sound
        because they are ``TensorAdapter`` subclasses, i.e. sources that also
        fill the tensor role (see :class:`TensorAdapter`). A source that does
        not must override this; the default below would otherwise hand back a
        self that cannot serve pixels.
        Multi-tensor adapters override this to return a new adapter for the tensor.

        Args:
            tensor_id: Identifier for the specific tensor within this source
        Returns:
            TensorAdapter for the specified tensor, with tensor context set
        Raises:
            TensorNotFound: ``tensor_id`` names a field this source does not have.
        """
        field = strip_source_prefix(self.source_id, tensor_id)
        if field and field != self.source_id:
            raise TensorNotFound(
                f"tensor {tensor_id!r} not found in source {self.source_id!r} "
                f"(single-tensor source has no field {field!r})",
                reason="unknown_field",
            )
        return self

    def close(self) -> None:  # noqa: B027 - concrete no-op default, not abstract
        """Release any long-lived OS handles this source holds.

        **An in-flight read is this method's problem, not its caller's.** An override
        holding a handle that a read is decoding through must deal with it itself.
        """

    @classmethod
    def create_from_payload(
        cls,
        source: SourceConfig,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        credentials_config: Optional[Any] = None,
    ) -> Optional[SourceAdapter]:
        """Rebuild the adapter a restart finds a row for, without parsing its file.

        The inverse of :meth:`catalog_payload`: *payload* is what that returned,
        *metadata* the row's ``metadata_json``. The caller has checked that the
        file is as it was when the row was written. ``None`` (the default) means
        this adapter has no such path, and the source is built from its claim.
        """
        return None

    def catalog_payload(self) -> Optional[Dict[str, Any]]:
        """What a restart needs to rebuild this adapter without parsing its file.

        The payload carries the serve path's derived state, such as descriptors
        with their transfer grid. Adding a key needs no version bump; changing
        the meaning of one bumps ``SOURCE_CATALOG_FORMAT``. Optional.
        """
        return None

    def release_registration_cache(  # noqa: B027 - concrete no-op default
        self,
    ) -> None:
        """Drop whatever was held only to answer registration, keeping derived state.

        Called by :meth:`MetadataDatabase.sync_source_added` once the catalog row
        is committed.
        """


class TensorAdapter(SourceAdapter):
    """Abstract base class for tensor-level adapters.

    This interface provides methods to read specific tensors, get chunk layouts,
    and read chunk data. It is returned by get_tensor_adapter() on the source adapter.

    **A tensor adapter is a source adapter that can also serve pixels.** The two
    roles nest rather than sit side by side, because every tensor adapter in this
    codebase is in fact a full source object. The role *scopes* stay disjoint at the
    point of declaration -- see the role-scope guard below -- so a tensor-scoped method
    still can never be declared on ``SourceAdapter``.
    """

    # The grant this tensor carries of its own. When set, reading it takes
    # either this or the server-wide token
    # (``TensorFlightServer._authorize_read``); None = no gate here, and the
    # server-wide rule alone.
    _capability_token: Optional[str] = None
    # Set on a tensor that is one of several in its source; None for the sole
    # tensor. ``array_id`` is minted from it.
    _tensor_name: Optional[str] = None

    @property
    def capability_token(self) -> Optional[str]:
        """The grant this tensor carries, or None for the server-wide rule."""
        return self._capability_token

    @capability_token.setter
    def capability_token(self, value: Optional[str]) -> None:
        self._capability_token = value

    #: Whether the values are ids, not measurements -- a label set. Averaging
    #: ids produces ids that exist nowhere, so a categorical tensor's computed
    #: pyramid is ``nearest`` whatever the server is configured to.
    categorical: bool = False

    @property
    def array_id(self) -> str:
        """This tensor's identifier, the one chunk ids are minted from:
        ``source_id``, or ``source_id/<tensor name>`` for a multi-tensor source.
        """
        if self._tensor_name is None:
            return self.source_id
        return f"{self.source_id}/{self._tensor_name}"

    @property
    def content_version(self) -> Optional[bytes]:
        """This tensor's content-version token, folded into the chunk_ids it
        mints, or None when its content is unversioned (see ``_content_version``).
        """
        return self._content_version

    def check_chunk_version(self, chunk_id: bytes) -> None:
        """Raise :class:`StaleChunkError` if ``chunk_id`` predates a re-registration.

        Pure in-memory comparison and a cheap guard ahead of an actual read.
        A unversioned chunk_id (``held_version`` None) always passes.

        :class:`RemoteTensorAdapter` overrides this to compare the proxy
        envelope's own version instead -- it never mints a plain (non-envelope)
        chunk_id, so this base implementation would misparse one of its chunk_ids.
        """
        held_epoch, held_version, _inner = _split_chunk_version(chunk_id)
        stale_content = (
            held_version is not None and held_version != self.content_version
        )
        if stale_content or held_epoch != current_epoch():
            raise StaleChunkError(
                f"chunk_id for {self.array_id!r} was minted against a "
                "version this source no longer serves; re-request the "
                "read plan (GetFlightInfo) rather than retrying this chunk_id.",
                reason="stale_content_version",
            )

    def put_chunk(
        self,
        bounds: ChunkBounds,
        data: pa.Array | pa.ChunkedArray,
        expected_shape: Tuple[int, ...],
        dtype: Any,
    ) -> None:
        """Write one uploaded chunk into this source's backing store.

        The DoPut path calls this after reading a chunk's Arrow payload, instead
        of sniffing the adapter's attributes. The default rejects the write --
        most source formats are read-only. Writable formats override with their
        own contract: ``ZarrAdapter`` enforces chunk-grid alignment, while
        ``CachedSourceAdapter`` accepts arbitrary bounds.

        Args:
            bounds: Chunk start/stop coordinates for the write.
            data: The chunk's flattened element values as a primitive Arrow array.
            expected_shape: Logical chunk shape implied by ``bounds``.
            dtype: NumPy dtype (or string) of the chunk elements.
        """
        raise WriteNotSupportedError(
            f"source {self.source_id!r} ({self._source_type}) does not support writes"
        )

    # Whether timing ``get_data`` measures what re-producing the chunk would
    # cost -- the premise the measured retention rule rests on, since "cheap"
    # means "cheap to rebuild" (see ``core.retention``). True wherever the
    # adapter decodes its own bytes from a backend it can read again.
    #
    # False for the two kinds that would be measured wrong rather than not at
    # all, in opposite directions. An upload and a passthrough proxy.
    _decode_time_is_rebuild_cost: bool = True

    # --- canonical axis order (biopb/biopb#596) ---------------------------------
    # A leaf adapter is written in whatever order its reader emits and carries
    # ``@canonical_axes`` (``core.normalize``): the decorator wraps the leaf's
    # ``get_data``, ``get_decimated_data``, ``read_block_shape``,
    # ``get_native_pyramid_levels`` and ``list_tensors`` so the planner, the scaled
    # and streamed reads and the pyramid -- all built on the public surface --
    # never see a native axis. The leaf reads its own geometry through
    # ``_native_descriptor()``, never the public descriptor.

    def _axis_perm(self) -> Optional[Tuple[int, ...]]:
        """This tensor's native -> canonical permutation, or None for identity.

        Derived from the native descriptor on every call, deliberately **not**
        memoized: an adapter's advertised labels are not immutable -- a source can
        be undescribable now and describable later, or differently later -- and a
        cached permutation cannot notice. The permutation itself is O(ndim); the
        cost is the ``_native_descriptor()`` call, which builds a proto for most
        adapters -- small next to the read it precedes.
        """
        try:
            return permutation_of(self, self._native_descriptor())
        except Exception:
            return None

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """The full **serving** descriptor for this bound tensor -- canonical
        axis order for a ``@canonical_axes`` class. Subclasses implement
        :meth:`_native_descriptor`."""
        desc = self._native_descriptor()
        perm = permutation_of(self, desc)
        return desc if perm is None else permute_descriptor(desc, perm)

    @abstractmethod
    def _native_descriptor(self) -> TensorDescriptor:
        """Return the full **serving** descriptor for this bound tensor, in the
        order the reader emits.

        Field must be populated:
            - array_id: Unique tensor identifier (for single-tensor: source_id;
              for multi-tensor: source_id/tensor_name)
            - shape: Tensor shape as list of ints
            - chunk_shape: The transfer grid, as list of ints, for THIS tensor --
              sized against the bound tensor's own dtype, labels, and backend
              block, never a sibling's. The adapter owns the choice; the server
              only clamps it to the Arrow ceiling (biopb/biopb#809). An adapter
              with no layout knowledge to apply calls
              ``default_transfer_chunk_shape``.
            - dtype: Data type string (numpy dtype.str format)
        Recommended fields to populate:
            - dim_labels: Dimension labels
        Returns:
            TensorDescriptor with required fields populated.

        A source-level adapter that also fills the tensor role (see the class
        docstring) answers here for its default tensor, and must do so by binding
        it -- not by reading its own catalog listing back, which carries no grid.
        """

    @abstractmethod
    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from the backend. Subclasses call
        super().get_data(bounds) to validate bounds, then read from their
        backend. A ``@canonical_axes`` class implements this in its reader's own
        order: ``bounds`` arrive native and the array is transposed to canonical
        without a copy.

        The returned array's memory MUST NOT have its lifetime tied to a
        closable handle. A transpose or slice view over an array this adapter
        owns is fine; a view onto a reader-owned mmap that ``_handle_reaper``
        can close is not -- the caller may hold it well past the adapter's lock,
        and :func:`~.normalize.canonical_axes` transposes it without copying. An adapter
        reading through a mapping copies before returning.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)
        Returns:
            Numpy array with data within the requested bounds
        Raises:
            ValueError: If bounds exceed array shape
        """
        desc = self._native_descriptor()
        shape = tuple(int(dim) for dim in desc.shape)
        validate_bounds(bounds, shape)

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """What this backend's reads are quantized to, or ``None`` for none.

        **This is the ``native=`` seed the adapter passes to**
        :func:`~.chunk.default_transfer_chunk_shape`.

        The streamed scaled read floors its tile here
        (:func:`~.stream_reduce.streaming_unit`), because the transfer grid is
        derived from this same granularity by *dividing* it whenever it exceeds
        the transfer target.

        ``None`` claims something stronger than "unknown": that no part of a read
        is wasted, which is true of an mmap and of a backend that forwards
        arbitrary bounds.
        """
        return None

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Every ``step``-th element of ``bounds``, or ``None`` to decline.

        ``None`` is the default and means "read the extent and stride it".
        An adapter implements this if strided read costs in proportion to what
        it *returns* rather than to the extent it spans. The candidates are the
        backends that already report no :attr:`read_block_shape`.

        An implementation must hold to all of:

        1. **Shape** -- ``len(range(start, stop, step))`` per axis, which is what
           ``downsample_block``'s ``nearest`` returns for the same extent.
        2. **Values** -- bit-identical to ``get_data(bounds)[::step]``. A pick is
           exact by construction for every dtype and every scale, so unlike
           ``area`` there is no accuracy trade hiding here.
        3. **Ownership** -- an owned array, exactly as :meth:`get_data` must
           return: no view onto a reader-owned mapping may escape.

        Args:
            bounds: Chunk bounds, in this adapter's own axis order.
            step: Per-axis stride, same order and length as ``bounds``.
        Returns:
            The picked array, or ``None`` to leave the caller on its own path.
        """
        return None

    def get_scaled_data(
        self,
        bounds: ChunkBounds,
        scale_hint: Tuple[int, ...],
        reduction_method: str,
        cache_manager: Optional[CacheManager] = None,
    ) -> np.ndarray:
        """Read ``bounds`` and reduce it by ``scale_hint`` in one step.

        The default streams the extent in tiles of the transfer grid, floored at
        :attr:`read_block_shape`, reducing each tile as it arrives, so peak
        residency is one tile.

        ``cache_manager`` lets the default source its units from the
        full-resolution chunks the cache already holds rather than decode them
        from the source again -- see :func:`~.cache_source.cache_sourced_units`. ``None``
        streams from the source throughout.

        ``nearest`` takes a shorter route where the backend offers one: it is a
        pick, so :meth:`get_decimated_data` expresses it whole, and an adapter
        implements that one method rather than this one. Streaming does not
        apply there -- a decimated read already materialises exactly the output.

        An adapter whose reader can deliver the extent in pieces more cheaply
        than ``get_data`` can overrides this.

        An override must hold to all of:

        1. **Shape** -- ``ceil((stop - start) / scale)`` per axis, identical to
           :func:`downsample_block`'s output, edge padding included.
        2. **Dtype** -- ``get_output_dtype(base_dtype, method)``, i.e. the
           input's own.
        3. **Values** -- bit-identical to
           ``downsample_block(self.get_data(bounds), scale_hint, method)``.
           Anything that cannot be is not an override: it is a different
           reduction, and must not be reached for a method it does not compute
           (CZI's ``read(zoom=)`` matches ``nearest`` and differs from ``area``
           in 100% of pixels).
        4. **Ownership** -- an owned array. In particular a fused ``nearest``
           must materialise: only the default's single-unit path may return the
           strided view ``TestZeroCopyContract`` pins, because only there is the
           base array already owned and already off the reader -- and where that
           base is instead borrowed from the cache, that path materialises too. The streamed
           path materialises by construction, writing picks into its own output.
        5. **Fallback** -- anything the fused path cannot express bit-identically
           calls ``super().get_scaled_data(...)`` rather than approximating, and
           forwards ``cache_manager`` when it does, or the fallback silently
           loses the cache-sourced path.

        Args:
            bounds: Chunk bounds, in canonical axis order.
            scale_hint: Per-axis reduction factor, same order as ``bounds``.
            reduction_method: Normalized method, decoded from the chunk_id.
            cache_manager: The chunk cache, when the caller has one.
        Returns:
            The reduced array.
        """
        step = tuple(max(1, int(scale)) for scale in scale_hint)
        if reduction_method == "nearest" and any(size > 1 for size in step):
            # A pick needs none of what it skips, so a backend that can stride
            # its own read never touches those bytes. Tried before the tile is
            # sized because it makes tiling moot: the result IS the output, so
            # residency is bounded at the chunk without streaming anything.
            picked = self.get_decimated_data(bounds, step)
            if picked is not None:
                return picked

        descriptor = self.get_tensor_descriptor()
        tensor_shape = tuple(int(dim) for dim in descriptor.shape)
        start = tuple(int(value) for value in bounds.start)
        stop = tuple(int(value) for value in bounds.stop)
        extent = tuple(hi - lo for lo, hi in zip(start, stop, strict=True))

        def read(unit_start, unit_stop):
            return self.get_data(
                ChunkBounds(start=list(unit_start), stop=list(unit_stop))
            )

        transfer = tuple(max(1, int(size)) for size in transfer_chunk_size(descriptor))
        unit = streaming_unit(extent, transfer, self.read_block_shape, scale_hint)
        unit, fetch, borrowed = cache_sourced_units(
            cache_manager,
            descriptor,
            self.content_version,
            start,
            stop,
            unit,
            scale_hint,
            reduction_method,
            read,
            transfer,
        )

        try:
            if all(
                hi - lo <= size for lo, hi, size in zip(start, stop, unit, strict=True)
            ):
                reduced = downsample_block(
                    fetch(start, stop), scale_hint, reduction_method
                )
                # Only if *this* unit was borrowed: it is a view onto the segment
                # mapping and ``nearest`` picks a view of that, which cannot
                # outlive the entry it is returned past. Materialising the
                # reduced array costs prod(scale) less than the unit copy it
                # replaces; a buffered unit owns its bytes already.
                if borrowed.key is not None:
                    return np.ascontiguousarray(reduced)
                return reduced

            return stream_reduce(
                fetch,
                start,
                stop,
                tensor_shape,
                unit,
                scale_hint,
                reduction_method,
                descriptor.dtype,
            )
        finally:
            borrowed.release()

    def get_arrow_schema(self, desc: Optional[TensorDescriptor] = None) -> pa.Schema:
        """Get the Arrow schema for this tensor.

        Schema format (the unified binary wire schema, biopb/biopb#293):
        - data: binary - the chunk's raw C-contiguous bytes per row
        - shape: list<int64> - shape tuple per chunk
        - dtype: string - numpy dtype string (carries endianness) per chunk

        Each RecordBatch has 1 row per chunk, making data self-describing. The
        client reconstructs the array with ``np.frombuffer(data, dtype)``; see
        ``CHUNK_WIRE_SCHEMA``.

        Returns:
            Arrow Schema with data, shape, and dtype fields
        """
        from biopb.tensor._wire_version import (
            TENSOR_WIRE_PROTOCOL_VERSION,
            WIRE_PROTOCOL_METADATA_KEY,
        )

        desc = desc or self.get_tensor_descriptor()
        require_resolved(desc)

        metadata = {
            WIRE_PROTOCOL_METADATA_KEY: str(TENSOR_WIRE_PROTOCOL_VERSION),
        }

        return CHUNK_WIRE_SCHEMA.with_metadata(metadata)

    def _retention_for_chunk(
        self, chunk_id: bytes, *, array_id: Optional[str] = None
    ) -> RetentionClass:
        """What a miss for this chunk would cost (see ``cache.RetentionClass``).

        Never a property of the caller that stored it: the class is written into
        a cache segment, so a first writer would otherwise fix it for good. A
        scaled chunk answers from the chunk_id alone; an unscaled one from what
        its array has been measured to decode at, which is shared process state
        for the same reason. ``core.retention`` owns both decisions.

        ``array_id``, if the caller already decoded it, saves re-parsing the
        chunk_id's prefix on the unscaled arm -- ``resolve_chunk_data`` always
        has, since it needs it for ``compute_fn`` regardless.

        Unscaled answers from the measured table and returns before the ladder
        is built, so a source nobody reads a scaled chunk from never pays for
        one. The ladder is memoized because this runs per chunk and a miss
        builds a descriptor; unkeyed, because a tensor's shape cannot change
        under one adapter and the config is installed once at server start.
        """
        if not is_scaled_chunk(chunk_id):
            # Full resolution, or a native level's own store -- either way the
            # ladder has nothing to say about it, and only what it costs to
            # decode can separate one worth keeping from one worth dropping.
            if not self._decode_time_is_rebuild_cost:
                return "normal"
            if array_id is None:
                array_id = array_id_from_chunk_id(chunk_id)
            return retention_for_array(array_id)
        ladder = getattr(self, "_ladder_cache", None)
        if ladder is None:
            desc = self.get_tensor_descriptor()
            ladder = self._ladder_cache = computed_ladder(desc.shape, desc.dim_labels)
        return retention_for_scale(decode_scale_info(chunk_id), ladder)

    def locate_chunk(self, chunk_id: bytes) -> Optional[ChunkLocation]:
        """Where this chunk's bytes already sit, for the localhost handoff.

        None by default, which sends ``server._handle_chunk_locate`` down its
        usual route: resolve the chunk into the chunk cache, then answer the
        byte range there. An adapter whose store *is* the served batch answers
        its own range instead and skips that copy
        (``adapters.cache_member.CacheMember``). Never a reason to fail a read
        -- a None here only means the client takes do_get.
        """
        return None

    def resolve_chunk_data(
        self,
        chunk_id: bytes,
        cache_manager: Optional[CacheManager] = None,
    ) -> pa.RecordBatch:
        """Resolve chunk data, handling scaled chunks and backend caching.

        The default implementation reads raw chunk data with ``self.get_data()``.
        Every chunk -- scaled or raw -- is cached by chunk_id whenever a
        CacheManager is available.

        What a stored chunk costs to produce again (:meth:`_retention_for_chunk`)
        is declared from the chunk when it is scaled, and measured when it is
        not. Only the unscaled arm below is timed: a scaled build is a
        full-resolution read plus a reduction, over an extent the cache may or
        may not already hold, so timing it would measure the cache's own state
        and then decide what the cache keeps by it. The unscaled arm cannot be
        served from the cache -- reaching it means the entry was missing -- so
        it is the one honest sample of what this array costs to decode.

        Raises:
            StaleChunkError: chunk_id carries a content_version that no longer
                matches this source's current one -- it was minted against an
                earlier registration (biopb/biopb#178). See
                :meth:`check_chunk_version`, called first, before any bytes
                are read.
        """
        self.check_chunk_version(chunk_id)
        self.check_readable()
        array_id, bounds = decode_chunk_id(chunk_id)

        # Check if scaled chunk (has extra bytes after bounds encoding)
        is_scaled_chunk_flag = is_scaled_chunk(chunk_id)

        should_cache = cache_manager is not None

        logger.debug(
            f"resolve_chunk_data: array_id={array_id}, scaled={is_scaled_chunk_flag}, "
            f"bounds_start={list(bounds.start)}, bounds_stop={list(bounds.stop)}, cache={should_cache}"
        )

        def compute_fn():
            if is_scaled_chunk_flag:
                result_arr = self.get_scaled_data(
                    bounds,
                    decode_scale_info(chunk_id),
                    decode_reduction_method(chunk_id),
                    cache_manager,
                )
            else:
                started = time.perf_counter()
                result_arr = self.get_data(bounds)
                if self._decode_time_is_rebuild_cost:
                    record_decode(
                        array_id, result_arr.nbytes, time.perf_counter() - started
                    )

            # Serialize into the unified binary wire schema: raw bytes + dtype
            # string, wrapped zero-copy.
            result = pack_chunk_batch(result_arr)
            return result, result_arr.nbytes

        if should_cache:
            cache_key = cache_key_for_chunk_id(chunk_id)
            entry = cache_manager.get_or_acquire(
                cache_key,
                compute_fn,
                self._retention_for_chunk(chunk_id, array_id=array_id),
            )
            data = entry.data
            cache_manager.release(cache_key)
        else:
            data, _ = compute_fn()

        return data

    def get_read_plan(self, request_desc: TensorDescriptor) -> TensorReadPlan:
        """Generate a read plan for the requested tensor descriptor.

        A native-pyramid adapter routes a ``precompute`` + ``scale_hint`` read to
        its matching on-disk level (see :meth:`_plan_precomputed_read`); every
        other read plans on the default uniform chunk grid, downsampling on the fly
        for a computed scale.

        Args:
            request_desc: TensorDescriptor from the client's read request, which may
                          include slice_hint and scale_hint/reduction_method directly.
        Returns:
            TensorReadPlan with the logical descriptor and list of chunk endpoints to read.
        """
        base_desc = self.get_tensor_descriptor()
        base_shape = tuple(int(dim) for dim in base_desc.shape)
        scale_hint = normalized_scale_hint(base_shape, request_desc.scale_hint)
        reduction_method = normalize_reduction_method(request_desc.reduction_method)
        if reduction_method == "precompute" and scale_hint is not None:
            return self._plan_precomputed_read(request_desc, scale_hint)

        chunk_size = transfer_chunk_size(base_desc)
        return _get_read_plan(
            base_desc,
            request_desc,
            chunk_size,
            content_version=self.content_version,
        )

    # ---- native-pyramid precompute routing ---------------------------------
    # Turning a ``precompute`` read into a read against one on-disk level's store.

    def _plan_precomputed_read(
        self, request_desc: TensorDescriptor, scale_hint: Tuple[int, ...]
    ) -> TensorReadPlan:
        """Plan a ``precompute`` read against the level matching ``scale_hint``.

        Find the native level whose downsample factors equal ``scale_hint``,
        translate the request's slice into that level's coordinates, and plan the
        read against the level's own store.
        """
        # The hint arrives canonical; the level lookup matches native factors.
        perm = self._axis_perm()
        native_hint = tuple(to_native(scale_hint, perm))
        level = self._find_level_for_scale(native_hint)
        if level is None:
            raise ValueError(
                f"No precomputed level matching scale_hint {tuple(scale_hint)}."
            )
        slice_hint = (
            request_desc.slice_hint if request_desc.HasField("slice_hint") else None
        )
        factors = to_canonical(self._level_downsample_factors(level), perm)
        level_slice = _convert_slice_to_level(slice_hint, factors)
        return self._plan_from_precomputed(level, level_slice)

    def _find_level_for_scale(self, scale_hint: Tuple[int, ...]) -> Optional[Any]:
        """Native level key whose downsample factors equal ``scale_hint``, else None.

        Overridden by native-pyramid adapters. The key is opaque to the shared
        routing -- an OME-Zarr dataset path, a QPTIFF integer index -- and only
        round-trips through :meth:`_level_downsample_factors` and
        the pyramid adapter's level lookup. The default advertises no native levels.
        """
        return None

    def _level_downsample_factors(self, level: Any) -> List[int]:
        """Per-axis integer downsample factors of ``level`` vs level 0.

        The counterpart to :meth:`_find_level_for_scale`: for the level key it
        returned, the factors that translate a base-coordinate slice into that
        level's grid. Overridden by native-pyramid adapters.
        """
        raise NotImplementedError

    def _plan_from_precomputed(
        self, level: Any, level_slice: Optional[SliceHint]
    ) -> TensorReadPlan:
        """Build a read plan whose chunks target one native level's store.

        The level adapter's descriptor carries ``array_id = source_id/{level}``, so
        the base planner encodes that into every chunk_id and ``DoGet`` dispatches
        back through :meth:`get_tensor_adapter`. The returned descriptor's
        ``array_id`` is reset to this tensor's, so the client still sees one tensor.
        """
        level_adapter = self.get_tensor_adapter(f"{self.array_id}/{level}")
        level_desc = level_adapter.get_tensor_descriptor()
        request = TensorDescriptor(
            array_id=level_desc.array_id,
            dim_labels=level_desc.dim_labels,
            shape=list(level_desc.shape),
            chunk_shape=list(level_desc.chunk_shape),
            dtype=level_desc.dtype,
        )
        if level_slice is not None:
            request.slice_hint.start[:] = level_slice.start
            request.slice_hint.stop[:] = level_slice.stop
        read_plan = level_adapter.get_read_plan(request)
        read_plan.descriptor.array_id = self.array_id
        return read_plan

    def plan_flight_info(
        self, read_opt: TensorReadOption, pyramid_config: PyramidConfig
    ) -> TensorReadPlan:
        """Build the open-time read plan for this tensor -- descriptor-authoritative.

        The single seam the server's ``get_flight_info`` calls per tensor. It
        returns the ``TensorReadPlan`` whose descriptor carries the native chunk
        grid, the server-advertised resolution pyramid, and the compact physical
        scale -- everything a client needs to open the tensor, filled here at open
        time (never in ``list_flights``, like ``metadata_json``, so discovery
        stays lean). ``metadata_json`` itself is still filled by the server from
        the catalog, not here.

        ``read_opt.fields`` selects the optional parts (biopb/biopb#563, a
        a ``FieldMask``):

        - ``endpoints`` gates the per-request chunk plan. Without it this is a
          **describe-only** call: the request's slice/scale/reduction hints do
          not apply (describe is the stable per-tensor fact, not a per-request
          read), the descriptor is this tensor's base descriptor, and
          ``get_read_plan`` -- the O(chunks) enumeration -- is skipped entirely.
        - ``pyramid`` gates the advertisement, whose native-level sizing is the
          expensive part; ``physical_scale`` is the cheap describe fact and is
          filled unconditionally.

        The default plans locally. A remote-proxy adapter overrides this to
        forward the upstream's authoritative plan instead (biopb/biopb#295).
        """
        base_desc = self.get_tensor_descriptor()
        desc = TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=base_desc.shape,
            chunk_shape=base_desc.chunk_shape,
            dtype=base_desc.dtype,
        )

        # Opt-in: an empty mask is a describe, and the O(chunks) plan is the
        # most expensive thing here, so it is never what saying nothing buys.
        mask = read_mask(read_opt)
        if ENDPOINTS in mask:
            if read_opt.HasField("slice_hint"):
                desc.slice_hint.CopyFrom(read_opt.slice_hint)
            if read_opt.scale_hint:
                desc.scale_hint[:] = list(read_opt.scale_hint)
            if read_opt.reduction_method:
                desc.reduction_method = read_opt.reduction_method
            read_plan = self.get_read_plan(desc)
        else:
            desc.chunk_shape[:] = list(transfer_chunk_size(base_desc))
            read_plan = TensorReadPlan(
                descriptor=desc,
                chunk_endpoints=[],
            )

        read_plan.descriptor.ClearField("pyramid")
        if PYRAMID in mask:
            read_plan.descriptor.pyramid.extend(
                self._advertised_pyramid(base_desc, pyramid_config)
            )
        self._fill_physical_scale(read_plan.descriptor)
        return read_plan

    def get_native_pyramid_levels(self) -> Optional[List[PyramidLevel]]:
        """Native (precomputed on-disk) pyramid levels for this tensor, or None.

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

    def has_native_pyramid(self) -> bool:
        """Whether this tensor ships a well-formed multi-resolution pyramid.

        Derived by default from :meth:`get_native_pyramid_levels` -- a tensor
        has a native pyramid iff it advertises native levels -- so a new format
        need only override the levels method. The precache worker skips a tensor
        that reports True: it already serves overviews cheaply from its own coarse
        levels. Adapters may override this with a cheaper check that avoids
        enumerating level shapes (e.g. OME-Zarr reads its root ``.zattrs``).
        """
        return self.get_native_pyramid_levels() is not None

    def _advertised_pyramid(
        self, base_desc: TensorDescriptor, pyramid_config: PyramidConfig
    ) -> List[PyramidLevel]:
        """The resolution-pyramid levels to advertise for this tensor.

        Native (precomputed on-disk) levels when the tensor ships them, else a
        computed pyramid from the server's ``[pyramid]`` knobs (``pyramid_config``,
        threaded in because adapters are constructed without the server's config).
        Cheap (arithmetic + already-memoized level adapters), so it is recomputed
        per open rather than cached. Consumed by ``plan_flight_info``.
        """
        levels = None
        try:
            levels = self.get_native_pyramid_levels()
        except Exception:
            logger.exception(
                "pyramid: native enumeration failed for %s", base_desc.array_id
            )
            levels = None
        if levels is None:
            cfg = pyramid_config
            if self.categorical:
                cfg = replace(cfg, reduction_method="nearest")
            levels = build_pyramid_plan(
                list(base_desc.shape),
                list(base_desc.dim_labels),
                reduction_method=cfg.reduction_method,
                **cfg.level_kwargs(),
            )
        return levels

    def get_tensor_metadata(self) -> Optional[dict]:
        """Per-tensor metadata fields the source-level catalog row does not carry.

        The serve path (``GetFlightInfo(with_metadata)``) reads a source's
        metadata from the catalog row that :meth:`SourceAdapter.get_metadata`
        produced once at registration -- the cache -- and **merges** this method's
        return over it (``row.update(get_tensor_metadata())``). So a tensor
        adapter returns only the *delta*: the cheap, per-tensor fields the
        source-level row cannot represent -- an OME-Zarr HCS field's own OME
        metadata over the plate ``.zattrs`` row, or an EMD signal's
        ``original_metadata`` over the source's ``{"format": "emd"}`` row.

        Keeping the shared bulk in the catalog and overlaying only the per-tensor
        delta here is the point (biopb/biopb#253): per-tensor metadata never needs
        a catalog row of its own. ``None`` (the default) means no delta -- the
        source-level row fully describes this tensor.
        """
        return None

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dimension physical pixel size + unit for this tensor, axis order.

        Returns ``(scale, unit)``: two equal-length lists aligned 1:1 with the
        ``dim_labels`` this tensor's ``get_tensor_descriptor()`` emits. Element
        ``i`` is the physical extent of one sample along dimension ``i``; ``0.0``
        / ``""`` mark a dimension with no known physical size (e.g. T/C axes).

        Returns ``None`` when no physical sizes are known. This is the compact
        ~200-byte summary the tensor-load hot path needs (issue #31), so it must
        be **cheap** -- read it straight off the resident metadata model, never a
        full ``get_metadata()`` dump. Default ``None``; format adapters that carry
        physical voxel sizes override it. There is no standalone public accessor:
        physical scale reaches clients only via the descriptor's
        ``physical_scale`` / ``physical_unit`` fields, filled by
        ``_fill_physical_scale`` inside ``plan_flight_info``.
        """
        return None

    def _fill_physical_scale(self, descriptor: TensorDescriptor) -> None:
        """Copy this tensor's compact physical scale onto ``descriptor``.

        Filled at open time (always, like the pyramid -- NOT gated on
        ``with_metadata``), never in ``list_flights``, so the common tensor-load
        path gets physical sizes without fetching the full OME tree (issue #31).
        Full-res values: physical scale is level-0 and is not rescaled by
        ``scale_hint``. Clears the fields when ``_physical_scale`` is unknown or
        its lengths do not match the descriptor's ``dim_labels``.
        """
        descriptor.ClearField("physical_scale")
        descriptor.ClearField("physical_unit")

        if hasattr(self, "_physical_scale_cache"):
            phys = self._physical_scale_cache
        else:
            try:
                phys = self._physical_scale()
            except Exception:
                phys = None
            self._physical_scale_cache = phys
        if phys is not None:
            scale_vec, unit_vec = phys
            perm = self._axis_perm()
            scale_vec, unit_vec = (
                to_canonical(scale_vec, perm),
                to_canonical(unit_vec, perm),
            )
            ndim = len(descriptor.dim_labels)
            if ndim and len(scale_vec) == ndim and len(unit_vec) == ndim:
                descriptor.physical_scale[:] = scale_vec
                descriptor.physical_unit[:] = unit_vec


# --- role-scope enforcement -------------------------------------------------
_SOURCE_SCOPED_API = frozenset(
    {
        "source_url",
        "source_type",
        "check_readable",
        "claim",
        "create_from_config",
        "create_from_payload",
        "list_tensors",
        "get_metadata",
        "get_embedded_rois",
        "catalog_url",
        "is_resolved",
        "get_tensor_adapter",
        "close",
        "release_registration_cache",
        "catalog_payload",
    }
)
_TENSOR_SCOPED_API = frozenset(
    {
        "capability_token",
        "array_id",
        "categorical",
        "content_version",
        "check_chunk_version",
        "put_chunk",
        "get_tensor_descriptor",
        "read_block_shape",
        "get_data",
        "get_decimated_data",
        "get_scaled_data",
        "get_arrow_schema",
        "resolve_chunk_data",
        "locate_chunk",
        "get_read_plan",
        "get_native_pyramid_levels",
        "has_native_pyramid",
        "get_tensor_metadata",
        "plan_flight_info",
    }
)


def transfer_chunk_size(desc: TensorDescriptor) -> Tuple[int, ...]:
    """The transfer grid a tensor is read on, from its *desc*riptor,, clamped to the Arrow ceiling.

    ``chunk_shape`` *is* the transfer grid and the adapter chose it
    (biopb/biopb#809); the server sizes nothing on its behalf. The one thing
    left is the wire bound: ``MAX_ARROW_BATCH_BYTES`` is a property of Arrow
    IPC, so a declared grid above it is re-split rather than failing mid-transfer.

    An empty or short ``chunk_shape`` (a bulk-seeded remote proxy whose upstream
    is unreachable, a :func:`catalog_entry` that was never bound) falls back to
    the whole tensor under the ceiling: safe, never good (biopb/biopb#292). An
    unresolved descriptor raises ``SourceUnresolvedError`` rather than a raw
    ``TypeError`` out of ``np.dtype("")``.
    """
    require_resolved(desc)
    shape = tuple(int(dim) for dim in desc.shape)
    chunk_shape = tuple(int(dim) for dim in desc.chunk_shape)
    if len(chunk_shape) != len(shape):
        chunk_shape = shape
    return compute_safe_chunk_size(
        tuple(
            min(max(1, chunk), dim)
            for chunk, dim in zip(chunk_shape, shape, strict=True)
        ),
        desc.dtype,
        list(desc.dim_labels),
    )


def _public_api(cls: type) -> frozenset:
    """Public (non-underscore) attribute names declared directly on ``cls``."""
    return frozenset(name for name in vars(cls) if not name.startswith("_"))


assert _public_api(SourceAdapter) == _SOURCE_SCOPED_API, (
    "SourceAdapter public API drifted from its declared source-level scope: "
    f"{sorted(_public_api(SourceAdapter) ^ _SOURCE_SCOPED_API)} "
    "(classify new methods in base._SOURCE_SCOPED_API / _TENSOR_SCOPED_API)"
)
assert _public_api(TensorAdapter) == _TENSOR_SCOPED_API, (
    "TensorAdapter public API drifted from its declared tensor-level scope: "
    f"{sorted(_public_api(TensorAdapter) ^ _TENSOR_SCOPED_API)} "
    "(classify new methods in base._SOURCE_SCOPED_API / _TENSOR_SCOPED_API)"
)
assert _SOURCE_SCOPED_API.isdisjoint(_TENSOR_SCOPED_API), (
    "source/tensor adapter scopes overlap: "
    f"{sorted(_SOURCE_SCOPED_API & _TENSOR_SCOPED_API)}"
)


def _convert_slice_to_level(
    slice_hint: Optional[SliceHint], level_scale: List[int]
) -> Optional[SliceHint]:
    """Translate a base-coordinate slice into a level's downsampled grid.

    A pure transform in the precompute read path (like :func:`_get_read_plan`),
    so it is a module function, not a method: it reads no adapter state, and
    ``TensorAdapter._plan_precomputed_read`` supplies the level's downsample
    factors from the per-format hook.
    """
    if slice_hint is None:
        return None
    level_start = [s // sc for s, sc in zip(slice_hint.start, level_scale, strict=True)]
    level_stop = [
        ceil_div(s, sc) for s, sc in zip(slice_hint.stop, level_scale, strict=True)
    ]
    return SliceHint(start=level_start, stop=level_stop)


def _get_read_plan(
    base_desc: TensorDescriptor,
    request_desc: TensorDescriptor,
    chunk_size: Tuple[int, ...],
    content_version: Optional[bytes] = None,
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

    # STEP 1 was performed by transfer_chunk_size(). This
    # helper receives the public transfer grid and leaves it unchanged.
    transfer_chunk_size = chunk_size

    # STEP 2: Compute the scaled grid from the bounded transfer grid. The
    # logical scaled shape is no larger per axis than transfer_chunk_size.
    if scale_hint is None:
        virtual_chunk_size = transfer_chunk_size
        logical_chunk_size = transfer_chunk_size
        output_dtype = base_desc.dtype
    else:
        # Scale the read extent up with the scale factor so the *delivered*
        # chunk lands on the transfer target, rather than at 1/scale of it per
        # axis for identical read work (biopb/biopb#805).
        virtual_chunk_size = scaled_virtual_chunk_size(
            transfer_chunk_size,
            base_shape,
            scale_hint,
            base_desc.dtype,
            list(base_desc.dim_labels),
            get_output_dtype(base_desc.dtype, reduction_method),
        )
        # ceil_div, not //: clamping the extent to the tensor leaves a partial
        # scale block on axes whose length is not a whole multiple of the scale.
        logical_chunk_size = tuple(
            ceil_div(virtual_chunk_size[ax], scale_hint[ax]) for ax in range(ndim)
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
    # No split check is needed here: the unscaled grid is the bounded transfer
    # grid, and a scaled grid's read block may only grow to where its reduction
    # still clears the Arrow ceiling (scaled_virtual_chunk_size).
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

        # Encode: array_id + virtual_bounds + optional scale_hint + the requested
        # reduction_method, then apply the content-version namespace.
        chunk_id = mint_chunk_id(
            base_desc.array_id,
            virtual_bounds,
            scale_hint,
            reduction_method,
            content_version,
        )

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

    return TensorReadPlan(
        descriptor=logical_desc,
        chunk_endpoints=logical_endpoints,
    )
