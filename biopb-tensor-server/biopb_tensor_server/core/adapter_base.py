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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

from biopb_tensor_server.core.attached import is_published, split_attached_field
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
from biopb_tensor_server.core.labels import (
    extent_mismatch,
    join_fields,
    label_image_axes,
    split_label_field,
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


def catalog_entry(desc: TensorDescriptor) -> TensorDescriptor:
    """Project a descriptor onto the **structural catalog entry** a source lists.

    The catalog surface and the serving surface are different facts about a
    tensor, and only one of them a *source* can answer (biopb/biopb#812):

    * Structural -- ``array_id`` / ``dim_labels`` / ``shape`` / ``dtype``. Stable
      per tensor, derivable from the container's own index, and what
      ``ListFlights`` and the DuckDB ``sources.tensors`` rows carry.
    * Serving -- above all the transfer ``chunk_shape``, plus the pyramid and the
      physical scale. These belong to the *tensor-bound* adapter that will
      actually serve the read: the grid can depend on the bound scene's Dask
      chunks, its own backend block, its native pyramid level, and the request's
      scale. A source-level adapter that answers for them is guessing on behalf
      of a scene it has not selected, and the guess is published as fact.

    So a source lists this projection, and ``GetFlightInfo`` -- which resolves
    the tensor adapter first -- is the one place a grid is published. Keeping
    ``TensorDescriptor`` as the wire type for both (rather than splitting the
    proto message) makes the invariant "``chunk_shape`` is empty on every
    catalog entry", enforced here and re-applied by :func:`catalog_tensors`.
    """
    return TensorDescriptor(
        array_id=desc.array_id,
        dim_labels=desc.dim_labels,
        shape=desc.shape,
        dtype=desc.dtype,
    )


def catalog_tensors(adapter: Any) -> List[TensorDescriptor]:
    """A source's tensors as the catalog stores them.

    The catalog invariant's enforcement point: the one path into the DuckDB
    ``sources.tensors`` column goes through here, so a listing that still
    carries a serving field cannot reach a client (biopb/biopb#812). Re-applies
    :func:`catalog_entry` even though implementations are asked to, because a
    source that forgets must not be able to publish a grid it guessed.

    Duck-typed on ``list_tensor_descriptors`` alone, like the rest of the
    registration surface ``sync_source_added`` reads -- plus what the upload
    path attached, listed **after** the format's own tensors: a source's first
    tensor is the one every listing reads as its picture -- the browser groups
    on it, and SQL reaches for ``tensors[1]`` -- and neither a label set
    (biopb/biopb#1059) nor an uploaded field may ever be that.

    A registered source has no tensors of its own, so its whole listing comes
    from ``attached_fields`` -- which is the same path a discovered source's
    uploaded fields take, and why nothing here needs to know which kind it has.
    """
    tensors = [catalog_entry(t) for t in adapter.list_tensor_descriptors()]
    attached = getattr(adapter, "attached_fields", None) or {}
    for field in attached.values():
        tensors.append(catalog_entry(field.get_tensor_descriptor()))
    sets = getattr(adapter, "label_sets", None) or {}
    for label_set in sets.values():
        tensors.append(catalog_entry(label_set.get_tensor_descriptor()))
    return tensors


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

    Adapters that participate in filesystem auto-discovery override the claim()
    classmethod to detect whether they handle a given path; it is not abstract
    (the default claims nothing) so a config-only format like HDF5 can opt out.
    """

    # Required fields
    source_id: str  # Data source identifier
    _source_url: Optional[str] = None  # URL/path to the data source
    _source_type: Optional[str] = None  # Source type identifier
    _tensor_name: Optional[str] = None  # Tensor name (for multi-tensor)

    # Optional capability token. When set, reading this adapter takes either it
    # or the server-wide token (``TensorFlightServer._authorize_read``). None =
    # no gate here (falls back to the server-wide rule). Set on a *source* it
    # covers every tensor in it (the result-cache adapter, whose source is one
    # result); set on an attached *tensor* it covers that tensor alone
    # (:meth:`tensor_capability_token`). The base default keeps the
    # ``capability_token`` property total for every other adapter.
    _capability_token: Optional[str] = None

    # Optional content-version token (biopb/biopb#178), folded into every
    # chunk_id this adapter mints and hence into the cache key, so a
    # re-registered source with new bytes gets a fresh cache namespace instead
    # of serving stale chunks. None means unversioned: no header, and an adapter
    # opts in only when it has a cheap, reliable change signal (a local file's
    # stat signature). Opaque -- the codec namespaces by it, never reads it.
    #
    # Declared on the source because a source is the usual owner of a content
    # lifetime, but the value is per TENSOR. A tensor whose bytes live elsewhere
    # carries its own (an uploaded label set, ``adapters/labels.py``); one
    # reading out of the source file keeps the source's (a discovered NGFF set).
    # Read it off the adapter that serves the bytes, not off the source the
    # array_id happens to name.
    _content_version: Optional[bytes] = None

    # Display-only override for the catalog ``source_url`` (the descriptor field
    # the tensor-browser / web viewer group the tree by). Normally None, so the
    # descriptor derives ``source_url`` from the raw path via ``to_catalog_url``.
    # The drag-drop runtime-add path sets it to a re-rooted url so each drop
    # renders as its own top-level root instead of nesting deep under the shared
    # absolute-path tree (see SourceManager._drop_catalog_url). It never touches
    # ``_source_url`` (filesystem ops) or ``source_id`` (path hash), so it is
    # purely cosmetic and needs no re-index.
    _catalog_url: Optional[str] = None

    # Whether the server may bring this source into canonical axis order by
    # *permuting* it -- wrapping the adapter in a ``NormalizingAdapter``
    # (biopb/biopb#596). True for every adapter that reads its own bytes: the
    # server owns the whole read path for those, so it can transpose them.
    #
    # False when the axis order is owned by ANOTHER party who has aligned the
    # rest of their state to it -- today only the remote proxy, whose upstream
    # mints the chunk_ids, plans the reads (biopb/biopb#295) and sizes the grid.
    # Permuting behind such an owner is the same desynchronization the write path
    # already refuses at ``add_tensor``, so those sources are validated and
    # refused at their read boundary instead. See ``core.normalize`` and
    # ``core.axes.noncanonical_order``.
    _normalizable_axes: bool = True

    @property
    def source_url(self) -> Optional[str]:
        """The source's URL/path, used for filesystem ops (warm/recall).

        Wraps the backing ``_source_url``; None when the adapter never set one.
        """
        return self._source_url

    @property
    def source_type(self) -> Optional[str]:
        """Format/source-type identifier (e.g. ``"ome_zarr"``).

        Wraps the backing ``_source_type``; None when the adapter never set one.
        """
        return self._source_type

    @property
    def catalog_url(self) -> str:
        """The URL the catalog row carries -- what clients group the tree by.

        The display form of :attr:`source_url`: a ``_catalog_url`` override when
        one was set (drag-drop re-rooting), else the raw path normalized by
        :func:`to_catalog_url`. Never used for filesystem ops.
        """
        return self._catalog_url or to_catalog_url(self._source_url)

    @property
    def capability_token(self) -> Optional[str]:
        """Per-source capability token, or None for the server-wide auth fallback.

        A *narrow grant*, never a replacement: it opens this source's pixels and
        annotations to a holder with no server-wide token, and the server-wide
        token still opens them too (``TensorFlightServer._authorize_read``). It
        grants reads only -- writes, ``resolve`` and ``warm`` take full access,
        because their cost is not scoped to this source. The catalog row stays
        public either way. The result-cache adapter sets it -- either from
        inside the adapter or, for an externally-granted capability (the embedded
        tensor cache mints a per-result token), through the setter below. Assign
        via this property, never the backing ``_capability_token`` field: the
        typed seam is the whole point (#278E).
        """
        return self._capability_token

    @capability_token.setter
    def capability_token(self, value: Optional[str]) -> None:
        self._capability_token = value

    def tensor_capability_token(self, array_id: Optional[str]) -> Optional[str]:
        """The grant the tensor *array_id* carries of its own, or None.

        Only an **attached** tensor can carry one: a format's own tensors are
        the source's, while what the upload path put here was produced by one
        caller and may be readable by that caller alone.

        Read off the routable index (:meth:`_attached_for`) rather than through
        :meth:`resolve_tensor`, so the auth path asks no format to resolve
        anything and a tensor still uploading is gated exactly as a published
        one is.

        Checked on every read through :meth:`TensorFlightServer._grants`, so a
        source with nothing attached (the common case) skips the field parse
        below rather than paying it on every chunk.
        """
        if not self._attached_tensors:
            return None
        attached = self._attached_for(self._within_source_field(array_id))
        return attached.capability_token if attached is not None else None

    @property
    def content_version(self) -> Optional[bytes]:
        """Opaque content-version token folded into this adapter's chunk_ids, or
        None when its content is unversioned (see ``_content_version``).

        The version of the bytes THIS adapter serves, which for a multi-tensor
        source is not always the source's own -- see ``_content_version``.

        The content signal alone. A chunk_id also carries the server's
        serving-semantics epoch, framed separately (``core.chunk``); a cache
        misses on either, while a consumer asking "did the data change?" -- an
        ROI's ``drawn_against_version``, the descriptor field -- wants this one.
        """
        return self._content_version

    def check_readable(self) -> None:  # noqa: B027 - concrete no-op default
        """Raise if this source cannot answer a pixel read right now.

        A no-op for everything that reads a file: a store on disk is readable
        whenever it is registered. An upload is not -- it is published by its
        producer, and refuses until then (``WritableSource.check_readable``).

        Asked by every read path that can serve bytes: ``resolve_chunk_data``
        and, because it answers a warm chunk without calling it, the localhost
        locate path (``server._handle_chunk_locate``). Pure in-memory, like
        :meth:`check_chunk_version` beside it, so both are cheap enough to run
        on every read.
        """

    def check_chunk_version(self, chunk_id: bytes) -> None:
        """Raise :class:`StaleChunkError` if ``chunk_id`` predates a re-registration.

        Pure in-memory comparison of the chunk_id's framed versions against
        this source's and this server's -- no adapter I/O -- so a caller can run it as a
        cheap guard ahead of a cache lookup (``server._handle_chunk_locate``) as
        well as ahead of an actual read (:meth:`TensorAdapter.resolve_chunk_data`),
        without paying for a second adapter lookup or (for a native-pyramid
        adapter) forcing a lazy level open just to validate. A legacy
        unversioned chunk_id (``held_version`` None) always passes, matching the
        byte-identical-format backward-compat promise in ``chunk.py``.

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
            source: SourceConfig with url, source_id, and format-specific options
            credentials_config: Optional CredentialsConfig for remote authentication

        Returns:
            An instance of a SourceAdapter subclass initialized with the provided config
        """

    @abstractmethod
    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """List this source's tensors as **structural catalog entries**.

        The source-listing/discovery surface: what the DuckDB catalog stores and
        what ``ListFlights`` publishes. It returns lightweight entries without
        expensive operations like scene switching or chunk-layout computation.

        Returns:
            List of TensorDescriptor, each a :func:`catalog_entry` projection:

            Required fields:
            - array_id: Unique tensor identifier (for single-tensor: source_id;
              for multi-tensor: source_id/tensor_name)
            - shape: Tensor shape as list of ints

            Optional fields:
            - dtype: Data type string. Can be omitted if expensive to compute.
              Must be populated by get_tensor_descriptor() for actual reads.

            Recommended optional fields:
            - dim_labels: Dimension labels (cheap to include)

            Required to be EMPTY:
            - chunk_shape: the transfer grid is a *serving* fact owned by the
              tensor-bound adapter (:meth:`TensorAdapter.get_tensor_descriptor`),
              not a catalog one. A source lists every tensor without binding any
              of them, so any grid it names here is a guess about a scene it has
              not selected -- published as fact to every client
              (biopb/biopb#812). ``GetFlightInfo`` resolves the tensor adapter
              first and is the one place a grid is answered.
            - pyramid / physical_scale / metadata_json: likewise open-time only.

        Implementations return :func:`catalog_entry` of whatever they have (the
        single-tensor idiom is ``[catalog_entry(self.get_tensor_descriptor())]``);
        :func:`catalog_tensors` re-applies it so the invariant holds for the
        catalog even if an implementation forgets.
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return the source-level metadata as a dict. Usually OME metadata.

        Called **once at registration** to populate the catalog's
        ``sources.metadata_json`` row (:meth:`MetadataDatabase.sync_source_added`);
        the serve path reads it back from the catalog, never by recomputing here
        (biopb/biopb#253). It must therefore be a pure producer -- do not memoize
        the result across calls (the catalog is the cache). Genuinely per-tensor
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
        ``@ome``-style set the catalog keeps read-only (biopb/biopb#951), and
        this is where a format says how to read its own.

        Default ``({}, None)``: no source carries annotations unless it says so.
        That is the safe default rather than a conservative one -- the server
        does not police what :meth:`get_metadata` returns, so a ``rois`` key in
        an EMD's ``original_metadata`` or an OME-Zarr's ``.zattrs`` means
        whatever that format meant by it, and reading it as OME-XML would invent
        annotations. It is deliberately not keyed on ``source_type`` either:
        that is a name, and it lies in both directions -- ``ome-zarr`` carries
        NGFF, while ``zeiss`` / ``leica`` / ``nikon`` and the rest are ome-types
        dumps through bioio.

        ``metadata`` is what this adapter just returned from
        :meth:`get_metadata` and ``tensors`` is ``(array_id, dim_labels)`` per
        tensor -- both passed in rather than recomputed, since the caller holds
        them and ``get_metadata`` is a pure producer that would re-parse.

        Returns:
            ``(rois_by_array_id, report)``. The report is opaque to the caller
            beyond having a ``summary()`` for the log, and may be ``None``.

        Raising is not fatal but IS a bug: the caller runs this inside source
        registration and swallows failures, because a source is its pixels first
        and an imported set is rebuilt on the next registration anyway.
        """
        return {}, None

    # -- attached tensors (biopb/biopb#1059) -----------------------------------
    # A tensor of this source the format did not produce: a label set the
    # server minted for an upload, or a field uploaded onto the source. Not
    # chained through __init__ (adapters set their own attributes), so the
    # slots are class-level None, materialized on use.
    #
    # One index whatever the kind, keyed by within-source field, holding every
    # such tensor from ``add_tensor`` until the reclaim sweep detaches it.
    # Whether one may be *listed* is its own upload record's answer
    # (:func:`~biopb_tensor_server.core.attached.is_published`), so an upload in
    # flight and the tombstone of one that was discarded stay routable here: a
    # status poll and a straggler's write both have to find their adapter.
    _attached_tensors: Optional[Dict[str, TensorAdapter]] = None
    # The sets the source's own file carries, kept apart because only these are
    # the file's: a native set cannot be detached, an attached one can.
    _embedded_label_sets: Optional[Dict[str, TensorAdapter]] = None
    # The validated, normalized merge of both label origins; None means rebuild.
    _label_sets_view: Optional[Dict[str, TensorAdapter]] = None

    def get_embedded_labels(self) -> Dict[str, TensorAdapter]:
        """Label sets this source's own file carries, keyed by within-source field.

        The pixel counterpart of :meth:`get_embedded_rois`: a format that stores
        labels beside its image -- an OME-Zarr's NGFF ``labels/`` group -- says
        here how to read them. Called once and memoized by :attr:`label_sets`.
        Default: none. Keys are ``[<image field>/]labels/<name>``
        (:func:`~biopb_tensor_server.core.labels.label_field`); values are
        tensor adapters bound to the set, in the format's own axis order --
        the base normalizes them.
        """
        return {}

    @property
    def label_sets(self) -> Dict[str, TensorAdapter]:
        """Every label set of this source, keyed by within-source field.

        The file's own (:meth:`get_embedded_labels`, read once) and the
        published label sets of :attr:`attached_tensors`, each normalized like
        any registered tensor and each checked -- whatever its origin -- against
        the image it binds to: that image must be a tensor of this source, and
        the set must span it (:func:`~biopb_tensor_server.core.labels.extent_mismatch`).
        A set that fails is dropped with a warning rather than served
        misaligned. Empty until the source is resolved, since its tensors are
        unknown before that; rebuilt after every attach or detach.

        This is the *published* view -- what the catalog lists and what a read
        resolves first. A set still being uploaded is in :attr:`label_uploads`
        instead, and joins this one when its upload reaches READY.
        """
        view = self._label_sets_view
        if view is not None:
            return view
        if not self.is_resolved():
            return {}
        from biopb_tensor_server.core.normalize import normalize_adapter

        if self._embedded_label_sets is None:
            self._embedded_label_sets = dict(self.get_embedded_labels())
        images = self._normalized_tensors()
        candidates = {
            **self._embedded_label_sets,
            **self._attached_label_sets(published=True),
        }
        view = {}
        for field, adapter in candidates.items():
            normalized = normalize_adapter(adapter)
            why = self.label_binding_error(
                field, normalized.get_tensor_descriptor(), images=images
            )
            if why is not None:
                logger.warning(f"labels: {self.source_id}/{field} dropped: {why}")
                continue
            view[field] = normalized
        self._label_sets_view = view
        return view

    def _normalized_tensors(self) -> Dict[str, TensorDescriptor]:
        """This source's tensors by ``array_id``, in canonical axis order.

        What a label set is checked against, and read once per check rather
        than per set -- ``list_tensor_descriptors`` re-derives on an HCS plate.
        The uploaded fields are in it because a set may bind to one: a field is
        a tensor of this source like any other, and only its bytes live
        elsewhere.
        """
        from biopb_tensor_server.core.normalize import _normalize_descriptor

        descs = list(self.list_tensor_descriptors())
        descs += [a.get_tensor_descriptor() for a in self.attached_fields.values()]
        return {d.array_id: _normalize_descriptor(d) for d in descs}

    def label_binding_error(
        self,
        field: str,
        desc: TensorDescriptor,
        images: Optional[Dict[str, TensorDescriptor]] = None,
    ) -> Optional[str]:
        """Why a set of *desc* cannot be served at label *field*, or None.

        One rule, checked at both ends: the upload kind calls it before it
        mints a sidecar, and :attr:`label_sets` calls it again for every
        origin when the sets are listed -- a native NGFF group and a sidecar
        from an earlier server life never passed through the upload. *desc* is
        in canonical order (both callers normalize first), and *images* is
        :meth:`_normalized_tensors` when the caller already holds it.
        """
        if split_label_field(field) is None:
            return f"{field!r} does not name a label set"
        image = self.label_image_descriptor(field, images=images)
        if image is None:
            return "binds to no tensor of the source"
        why = extent_mismatch(
            desc.dim_labels, desc.shape, image.dim_labels, image.shape
        )
        return f"does not span its image: {why}" if why is not None else None

    def label_image_axes(
        self,
        field: str,
        desc: TensorDescriptor,
        images: Optional[Dict[str, TensorDescriptor]] = None,
    ) -> Optional[List[int]]:
        """Which of the image's axes each axis of the set at *field* indexes.

        The mapping the extent rule already implies, stated so a client reads
        it instead of re-deriving it. Re-deriving is not hypothetical: the two
        tensors do not number their axes alike, so a client matching them by
        name gets `t`/`z` right and an unnamed axis wrong -- which is frame 0
        of a timelapse where frame 40 was asked for, a picture rather than an
        error.

        Answered here rather than by the set's adapter because the mapping is
        a fact about the *pair*, and only the source holds both -- in canonical
        order, which is what makes the answer meaningful when a native set had
        to be permuted to get there. ``None`` when the set does not span its
        image, which is the same set :attr:`label_sets` drops.
        """
        image = self.label_image_descriptor(field, images=images)
        if image is None:
            return None
        return label_image_axes(desc.dim_labels, image.dim_labels)

    def label_image_descriptor(
        self,
        field: str,
        images: Optional[Dict[str, TensorDescriptor]] = None,
    ) -> Optional[TensorDescriptor]:
        """The image a label *field* binds to, normalized, or None if it has none.

        What the extent is measured against, and what the upload kind reads to
        fill in the axes of a request that named none.
        """
        parsed = split_label_field(field)
        if parsed is None or parsed.level is not None:
            return None
        if images is None:
            images = self._normalized_tensors()
        return images.get(join_fields(self.source_id, parsed.image_field))

    @property
    def attached_tensors(self) -> Dict[str, TensorAdapter]:
        """Every tensor the upload path put on this source, keyed by field.

        The **routable** set -- published, still filling, or a tombstone --
        handed out as attached rather than normalized, because an upload refuses
        a non-canonical order at create and the boundary needs the writable
        adapter itself (``put_chunk``, ``set_status``).

        What may be *read* is the checked views over this: :attr:`label_sets`
        and :attr:`attached_fields`.
        """
        return dict(self._attached_tensors or {})

    def attached_tensor(self, field: str) -> Optional[TensorAdapter]:
        """The tensor attached at *field*, whatever its state, or None."""
        return (self._attached_tensors or {}).get(field)

    def attach_tensor(self, field: str, adapter: TensorAdapter) -> None:
        """Make *adapter* answer for *field* on this source.

        Every kind arrives here: ``add_tensor`` attaches an upload the moment
        it mints one -- that is what routes the tensor's own writes -- and the
        registration hooks attach what an earlier life left on disk.
        Handed over in its own axis order and checked when the source's tensors
        are next listed, not here: an unresolved source has no tensors to check
        a set against yet, and the upload kinds validate at create anyway.

        Attaching is not listing (:meth:`attachment_changed`).
        """
        self._attach("_attached_tensors", field, adapter)

    def detach_tensor(self, field: str) -> Optional[TensorAdapter]:
        """Stop answering for *field*; returns what was attached, or None.

        Only what was attached: a set the file carries is the file's.
        """
        return self._detach("_attached_tensors", field)

    def attachment_changed(self) -> None:
        """Rebuild the checked views: an attached tensor's state moved.

        Attaching and detaching say so themselves; this is for the transition
        that changes what may be listed without touching the index -- an upload
        reaching READY, or discarded into a tombstone that stays routable.
        """
        self._label_sets_view = None

    def _attach(self, attr: str, field: str, adapter: TensorAdapter) -> None:
        """Set *field* -> *adapter* in the dict named *attr*, lazily created."""
        d = getattr(self, attr)
        if d is None:
            d = {}
            setattr(self, attr, d)
        d[field] = adapter
        self.attachment_changed()

    def _detach(self, attr: str, field: str) -> Optional[TensorAdapter]:
        """Pop *field* from the dict named *attr*; returns it, or None."""
        d = getattr(self, attr)
        if not d:
            return None
        removed = d.pop(field, None)
        if removed is not None:
            self.attachment_changed()
        return removed

    def _attached_label_sets(self, *, published: bool) -> Dict[str, TensorAdapter]:
        """The attached label sets on one side of the published gate."""
        return {
            field: adapter
            for field, adapter in (self._attached_tensors or {}).items()
            if split_label_field(field) is not None
            and is_published(adapter) is published
        }

    @property
    def label_uploads(self) -> Dict[str, TensorAdapter]:
        """Label sets of this source the upload path is still filling, by field.

        Plus the tombstones of ones it gave up on: routable, so a status poll
        and a straggler's write both find their adapter, but never listed --
        nobody may read them yet, or the bytes are gone. A set reaches
        :attr:`label_sets` by becoming readable, not by being moved.
        """
        return self._attached_label_sets(published=False)

    @property
    def attached_fields(self) -> Dict[str, TensorAdapter]:
        """The published fields uploaded onto this source, keyed by field.

        A tensor of this source whose bytes the upload path owns, under the
        marked segment that keeps its id off a native one
        (:mod:`~biopb_tensor_server.core.attached`). Listed after the format's
        own (:func:`catalog_tensors`) and unchecked, unlike a label set: a field
        binds to nothing, so there is nothing for it to fail to span.
        """
        return {
            field: adapter
            for field, adapter in (self._attached_tensors or {}).items()
            if split_attached_field(field) is not None and is_published(adapter)
        }

    def resolve_tensor(self, tensor_id: Optional[str]) -> TensorAdapter:
        """The adapter bound to *tensor_id*: an attached tensor of this source,
        else whatever :meth:`get_tensor_adapter` answers.

        The one lookup the serve path uses (``get_flight_info``, the precache),
        so an attached tensor is reachable by its ``array_id`` like any other
        while the format's own routing is untouched. An id under a marked
        segment that names nothing attached here is handed to the format anyway
        rather than refused: a proxy's upstream may serve it, and a format that
        cannot raises its own ``TensorNotFound``.
        """
        attached = self._attached_for(self._within_source_field(tensor_id))
        if attached is not None:
            return attached
        return self.get_tensor_adapter(tensor_id)

    def _attached_for(self, field: Optional[str]) -> Optional[TensorAdapter]:
        """The attached tensor answering for within-source *field*, or None.

        Only a **marked** field reaches one: a label set through its
        right-to-left parse, an uploaded field through the whole of its
        ``@fields/<name>``. Everything else is the format's own routing, which
        is the whole of the rule -- the upload path mints no bare field.
        """
        parsed = split_label_field(field)
        if parsed is not None:
            if parsed.level is not None:
                return None
            return self._label_set_for(parsed.set_field)
        if field is None or split_attached_field(field) is None:
            return None
        return self.attached_tensor(field)

    def _label_set_for(self, set_field: str) -> Optional[TensorAdapter]:
        """The adapter answering for label field *set_field*, listed or in flight.

        A set being uploaded is addressable from ``add_tensor`` onwards --
        that is how its producer polls it to READY (biopb/biopb#1048) -- so
        both views are consulted, the published one first.
        """
        label_set = self.label_sets.get(set_field)
        if label_set is not None:
            return label_set
        return self.label_uploads.get(set_field)

    def resolve_chunk_adapter(self, field: Optional[str]) -> TensorAdapter:
        """The adapter that serves a chunk whose route carries *field*.

        A within-source suffix on a chunk names either a native pyramid level
        (OME-Zarr / QPTIFF precompute) or a tensor field, and for a label set
        either of those *under* the set: ``labels/nuclei/1`` is level ``1`` of
        set ``nuclei``. A native-pyramid adapter answers the level's backend
        from :meth:`get_level_adapter`; every other adapter (and a bare
        suffix) answers None and the read routes to the tensor.
        """
        parsed = split_label_field(field)
        label_set = (
            self._label_set_for(parsed.set_field) if parsed is not None else None
        )
        if label_set is not None:
            level = label_set.get_level_adapter(parsed.level) if parsed.level else None
            return level or label_set
        attached = self._attached_for(field)
        if attached is not None:
            return attached
        level = self.get_level_adapter(field) if field is not None else None
        return level or self.get_tensor_adapter(field)

    def get_level_adapter(self, path: str) -> Optional[TensorAdapter]:
        """Backend adapter for native pyramid level ``path``, or ``None``.

        Declared here -- rather than sniffed with ``hasattr`` in the chunk
        dispatch -- for the same reason :meth:`close` and :meth:`put_chunk`
        are: an optional capability the dispatch drives on every registered
        source belongs in the interface, where a delegating wrapper's author
        can see it (biopb/biopb#557). On the source role because the chunk
        route is source-scoped: ``source_id/<field>`` is split before the
        lookup, and every registered source answers it (``UnresolvedSourceAdapter``
        forwards it). The default ``None`` means "no native levels," so
        :meth:`resolve_chunk_adapter` falls back to :meth:`get_tensor_adapter`.
        A native-pyramid adapter overrides this to return the level's own
        backend adapter, whose ``array_id`` is ``source_id/{level}`` -- the
        value a precompute chunk_id carries, so ``DoGet`` routes the level's
        chunks straight back here.
        """
        return None

    def is_resolved(self) -> bool:
        """Deterministic: is there a hydrated adapter backing this source?

        True by default; only ``UnresolvedSourceAdapter`` overrides it. Unlike
        ``is_resident()``, this never flips back to False once True in a
        process (a source isn't un-resolved by re-dehydrating) -- the signal
        for "should a client offer to resolve this".
        """
        return True

    def resolve(self) -> None:
        """Hydrate this source if needed.

        This is the ONE consented entry point that may perform an extended,
        blocking recall (e.g. downloading a whole cloud / synced-folder file).
        It is the sole resolution trigger: the serve paths (get_tensor_adapter ->
        GetFlightInfo / DoGet) never resolve on their own -- they raise
        SourceUnresolvedError on an unresolved source so the only thing that
        downloads is an explicit ``resolve``.

        For an already-resident source this is a cheap no-op (idempotent), so
        the server's ``resolve`` action works uniformly across all source kinds.
        ``UnresolvedSourceAdapter`` overrides it to actually hydrate.

        Returns nothing: what the caller wants afterwards is the source's now-
        concrete catalog row, which resolution writes (``on_resolved`` ->
        ``sync_source_added``) and the server reads back.
        """
        return None  # a resident source is already resolved

    def is_resident(self) -> bool:
        """Best-effort, recall-free: is this source's content local and cheap to
        read right now?

        Remote (fsspec) sources are never resident until their pixels are
        materialized into a local copy (a later phase); a local source is
        resident unless it is an offline cloud placeholder. This is the
        authoritative, point-in-time residency gate -- VOLATILE, so evaluate it
        at the moment of use and never cache the result. Nothing stores the
        answer: the catalog has no residency column and the ``is_resident``
        action re-asks this on every call (biopb/biopb#1035).
        """
        # Lazy import: base <-> discovery only cross-import under TYPE_CHECKING,
        # so importing these at module scope would be circular.
        from pathlib import Path

        from biopb_tensor_server.core.discovery import (
            _is_offline_placeholder,
            directory_is_resident,
        )
        from biopb_tensor_server.core.remote import is_remote_url

        if is_remote_url(self._source_url):
            return False
        path = Path(self._source_url)
        # The offline-placeholder signal (st_blocks == 0) is a per-*file* concept
        # -- discovery only consults it for files (see should_skip_walk_entry,
        # which gates it on `not is_dir`). A directory-based source (zarr,
        # ome-zarr store) legitimately reports st_blocks == 0 on some filesystems
        # (e.g. macOS APFS), so applying the file check to the directory path
        # itself would wrongly flag an entirely local store as non-resident.
        # `directory_is_resident` instead samples files *inside* the directory.
        if path.is_dir():
            return directory_is_resident(path)
        return not _is_offline_placeholder(path)

    def get_tensor_adapter(self, tensor_id: str | None) -> TensorAdapter:
        """Factory method to return adapter with specific tensor context.

        Transitions the adapter from source context to tensor context.
        Single-tensor adapters return self with tensor context set -- sound
        because they are ``TensorAdapter`` subclasses, i.e. sources that also
        fill the tensor role (see :class:`TensorAdapter`). A source that does
        *not* (``UnresolvedSourceAdapter``) must override this; the default
        below would otherwise hand back a self that cannot serve pixels.
        Multi-tensor adapters override this to return a new adapter for the tensor.

        Total by contract: a single-tensor source has exactly one tensor, so any
        *unknown nonempty* field is rejected with a typed ``TensorNotFound`` (gRPC
        NOT_FOUND) -- the miss is representable rather than silently returning the
        base tensor under the wrong ``array_id``. Three inputs still resolve to
        that sole tensor (reduced to a within-source field by
        ``_within_source_field``): an empty/``None`` id, the source's own id (a
        bare ``source_id`` or the full ``source_id`` array_id -> falsy or
        ``== source_id``), and -- for a source whose one tensor carries a name --
        that ``_tensor_name`` (a single-scene aicsimageio file names its lone
        tensor, e.g. ``"Image:0"``, so ``source_id/Image:0`` is a valid read).

        Args:
            tensor_id: Identifier for the specific tensor within this source
        Returns:
            TensorAdapter for the specified tensor, with tensor context set
        Raises:
            TensorNotFound: ``tensor_id`` names a field this source does not have.
        """
        field = self._within_source_field(tensor_id)
        if field and field != self.source_id and field != self._tensor_name:
            raise TensorNotFound(
                f"tensor {tensor_id!r} not found in source {self.source_id!r} "
                f"(single-tensor source has no field {field!r})",
                reason="unknown_field",
            )
        return self

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

    def close(self) -> None:  # noqa: B027 - concrete no-op default, not abstract
        """Release any long-lived OS handles this source holds.

        Declared here, rather than sniffed with ``getattr(adapter, "close",
        None)``, for the same reason :meth:`put_chunk` is: an optional capability
        the registry drives on every adapter belongs in the interface, where a
        delegating wrapper's author can see it. ``UnresolvedSourceAdapter``
        forwarding everything *except* ``close`` is precisely what a duck-typed
        hook could not catch (biopb/biopb#71).

        Most adapters hold nothing between reads -- see the file-handle policy in
        ARCHITECTURE.md -- so the default is a no-op and only the persistent-handle
        adapters override it. An override must be safe to call twice.

        **An in-flight read is this method's problem, not its caller's.** Nothing
        drains before calling: ``SourceRegistry.unregister`` and ``close_all``
        close on the spot, and a replace closes the displaced adapter as soon as
        the swap has committed (``SourceRegistry.swap``). So an override holding
        a handle that a read is decoding through must deal with it itself --
        drain on ``_active_reads`` under a deadline (``OmeTiffAdapter``), decline
        and leave the release to the idle reaper (mrc / dv / qptiff), or release
        under ``_io_lock`` (czi / nd2 / ndtiff / bioio).
        """

    def release_registration_cache(  # noqa: B027 - concrete no-op default
        self,
    ) -> None:
        """Drop whatever was held only to answer registration, keeping derived state.

        Called by :meth:`MetadataDatabase.sync_source_added` once the catalog row
        is committed -- the moment the catalog, not the adapter, owns this
        source's metadata (biopb/biopb#253). An adapter that parked a bulky
        intermediate on itself to build that row may release it here; anything
        the serve path still needs must survive.

        Declared on the interface rather than sniffed with ``getattr`` for the
        same reason :meth:`close` is: a delegating wrapper's author has to see it
        (biopb/biopb#71). Default no-op -- only adapters with something big to
        drop override it. Like ``close``, an override must be safe to call twice,
        and must not make the released state unrecoverable: ``sync_source_added``
        runs again when an unresolved source resolves.

        A server with no catalog (the embedded image-base cache builds its
        ``TensorFlightServer`` with ``metadata_db=None``) never calls this, so
        nothing is released out from under a source whose metadata has nowhere
        else to live.
        """

    def _within_source_field(self, tensor_id: Optional[str]) -> Optional[str]:
        """Reduce a source-qualified array_id to its within-source field, for the
        multi-tensor ``get_tensor_adapter`` overrides.

        A caller may legitimately hand them the full array_id ("a tensor is
        identifiable by array_id alone"); this strips the ``source_id/`` prefix
        (pure reduction -- see :func:`strip_source_prefix`), leaving a bare field
        unchanged. This layer never invents ``None``: "default tensor" is decided
        upstream at the server chokepoint, not here.
        """
        return strip_source_prefix(self.source_id, tensor_id)


class TensorAdapter(SourceAdapter):
    """Abstract base class for tensor-level adapters.

    This interface provides methods to read specific tensors, get chunk layouts,
    and read chunk data. It is returned by get_tensor_adapter() on the source adapter.

    Tensor-level adapters are created for specific tensors within a source, allowing
    them to maintain tensor-specific state (e.g., current scene in multi-scene files).

    **A tensor adapter is a source adapter that can also serve pixels.** The two
    roles nest rather than sit side by side, because every tensor adapter in this
    codebase is in fact a full source object: single-tensor formats return ``self``
    from ``get_tensor_adapter``, the multi-tensor ones (bioio / OME-TIFF / EMD)
    return a clone of their own class with tensor context set, and the OME-Zarr /
    QPTIFF level and HCS-field adapters are plain ``ZarrAdapter`` instances. The
    serve path relies on it -- an HCS field's per-field metadata comes from the
    tensor adapter's :meth:`get_tensor_metadata`, which the plate's source-level
    catalog row cannot represent (biopb/biopb#253). Nesting types that reality
    instead of contradicting it (biopb/biopb#380).

    The converse does not hold: ``UnresolvedSourceAdapter`` is a source that has no
    tensors until it resolves, and stays a plain ``SourceAdapter``. So "source" is
    the general role and "tensor" the specialization, which is the direction this
    inheritance encodes.

    The role *scopes* stay disjoint at the point of declaration -- see the
    role-scope guard below -- so a tensor-scoped method still can never be declared
    on ``SourceAdapter``.
    """

    # Whether timing ``get_data`` measures what re-producing the chunk would
    # cost -- the premise the measured retention rule rests on, since "cheap"
    # means "cheap to rebuild" (see ``core.retention``). True wherever the
    # adapter decodes its own bytes from a backend it can read again.
    #
    # False for the two kinds that would be measured wrong rather than not at
    # all, in opposite directions. An upload has no backend: its cache entry is
    # the only copy, so a local read times a memcpy out of the very cache being
    # classified and a "cheap" stamp is data loss. A passthrough proxy has one,
    # but a miss costs an upstream round trip that the local hand-off does not
    # contain, so timing it clocks a LAN upstream as fast and evicts it first.
    # Both are declined here rather than measured and ignored: a rate nobody may
    # act on is a diagnostic that reads as fact.
    #
    # Both also override ``resolve_chunk_data`` today and so never reach the
    # sample site, which is what makes this a guard rather than a live switch.
    # It is declared because that structural exemption is invisible from either
    # adapter -- neither file mentions retention -- so narrowing an override
    # (biopb/biopb#265, to serve a cache source's scaled reads from the chunks
    # it already holds) would otherwise enrol it silently.
    _decode_time_is_rebuild_cost: bool = True

    @abstractmethod
    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return the full **serving** descriptor for this bound tensor.

        The counterpart to :meth:`SourceAdapter.list_tensor_descriptors`, which
        answers the structural half for every tensor without binding any of them.
        This is called on an adapter that ``get_tensor_adapter(array_id)`` has
        already bound to one tensor -- a bioio/CZI scene, an OME-Zarr HCS field,
        a QPTIFF level -- so it can answer for facts that only exist once that
        selection is made (biopb/biopb#812).

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

    def get_transfer_chunk_size(self) -> Tuple[int, ...]:
        """Return this tensor's transfer grid, clamped to the Arrow ceiling.

        ``chunk_shape`` *is* the transfer grid and the adapter chose it
        (biopb/biopb#809). The server sizes nothing on the adapter's behalf here
        -- an adapter that knows its physical layout would only have its answer
        undone, which is what made biopb/biopb#806 unfixable while the planner
        ran on this seam -- and re-planning would also break the cache-backed
        sources, which serve *only* the chunk_ids that were written, so a grid
        that is not theirs asks for bounds that do not exist.

        The one thing left is the wire bound: ``MAX_ARROW_BATCH_BYTES`` is a
        property of Arrow IPC, not of any format, so a declared grid above it is
        re-split here rather than failing mid-transfer.

        A descriptor may still reach here with an empty ``chunk_shape``: a
        bulk-seeded remote proxy whose upstream is unreachable, or a
        :func:`catalog_entry` handed in by a caller that should have bound the
        tensor first. Handing the read planner a too-short tuple indexes out of
        range against the full-rank shape (biopb/biopb#292), so fall back to the
        whole tensor split under the ceiling -- a safe answer, never a good one,
        which is why the catalog grid is not a fallback anyone may plan on.

        An *unresolved* descriptor (empty shape/dtype -- a not-yet-hydrated
        cloud/remote source) is rejected up front: the fallback would otherwise
        reach ``np.dtype("")`` inside ``compute_safe_chunk_size`` and raise a raw,
        illegible ``TypeError``. ``require_resolved`` converts it to a clean
        ``SourceUnresolvedError`` at this read-planning boundary, exactly as
        ``get_arrow_schema`` and ``_get_read_plan`` already do.
        """
        desc = self.get_tensor_descriptor()
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

    @abstractmethod
    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from the backend.
        Subclasses should call super().get_data(bounds) to validate bounds,
        then read data from their backend.

        The returned array's memory MUST NOT have its lifetime tied to a
        closable handle. A transpose or slice view over an array this adapter
        owns is fine; a view onto a reader-owned mmap that ``_handle_reaper``
        can close is not -- the caller may hold it well past the adapter's lock,
        and ``core/normalize.py`` transposes it without copying. An adapter
        reading through a mapping copies before returning.

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

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """What this backend's reads are quantized to, or ``None`` for none.

        A zarr chunk, an HDF5 chunk, a TIFF page: reading any part of one costs
        the whole one. The streamed scaled read floors its tile here
        (:func:`~.stream_reduce.streaming_unit`), because the transfer grid is
        derived from this same granularity by *dividing* it whenever it exceeds
        the transfer target -- and a tile inside a block re-reads that block once
        per tile. Unfloored that is 8-11x on a tiled 8192^2 OME-TIFF page and ~3x
        on an OME-Zarr chunked at 4096^2.

        **This is the ``native=`` seed the adapter already passes to**
        :func:`~.chunk.default_transfer_chunk_shape`, not a second fact -- state
        them from one expression so they cannot drift.

        ``None`` claims something stronger than "unknown": that no part of a read
        is wasted, which is true of an mmap and of a backend that forwards
        arbitrary bounds. Declaring it wrongly is silent -- every value stays
        bit-identical, the read just costs more -- so ``adapter_read_block_test``
        requires every adapter class to appear in one list or the other rather
        than letting a new one default in.

        Note the seed is an upper bound on granularity and a reader may beat it:
        ``NikonAdapter`` seeds its grid with a whole C/Y/X ND2 frame (1.1 GiB on
        a 14234^2 scene) that ``read_frame`` hands back as an mmap view, then
        crops -- so it declares ``None`` and is right to.

        A property rather than a class attribute because the answer is per
        *instance* -- a tiled and a striped TIFF are the same adapter with
        different answers -- and derived live rather than captured in
        ``__init__`` because an adapter may not hold its store yet.
        """
        return None

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Every ``step``-th element of ``bounds``, or ``None`` to decline.

        ``None`` is the default and means "read the extent and stride it", which
        is what the caller does anyway. An adapter implements this only where a
        strided read costs in proportion to what it *returns* rather than to the
        extent it spans -- and that is exactly what a ``nearest`` reduction is:
        ``data[::step]``, element 0 of every block, needing none of the elements
        it skips.

        The candidates are the backends that already report no
        :attr:`read_block_shape`, and the two answers correlate for one reason:
        a quantized backend has to decode a whole block to hand back any of it,
        so skipping elements inside it saves no reading -- only a memcpy, which
        the streamed path already bounds. Where nothing is quantized, the skipped
        elements are never touched at all. That is the one reduction where fusing
        removes *reads* and not just heap: ``area`` has to visit every source
        element whatever it does with them.

        Declaring this where the backend cannot honour it cheaply is silent in
        the same way :attr:`read_block_shape` is -- every value stays
        bit-identical, the read merely costs more -- so ``decimated_read_test``
        requires every adapter class to appear in one list or the other.

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
        residency is one tile rather than the extent (see
        :mod:`~.stream_reduce`). An extent that is already one tile is read and
        reduced whole, which is what every unscaled read and most small scaled
        ones do.

        ``cache_manager`` lets the default source its units from the
        full-resolution chunks the cache already holds rather than decode them
        from the source again -- see :func:`~.cache_source.cache_sourced_units`. ``None``
        streams from the source throughout.

        ``nearest`` takes a shorter route where the backend offers one: it is a
        pick, so :meth:`get_decimated_data` expresses it whole, and an adapter
        implements that one method rather than this one. Streaming does not
        apply there -- a decimated read already materialises exactly the output.

        An adapter whose reader can deliver the extent in pieces more cheaply
        than ``get_data`` can (a CZI ``read(zoom=)``, a native pyramid level)
        overrides this instead, so no view onto a reader-owned mapping ever
        leaves the adapter's lock. Overriding to bound memory is no longer a
        reason: the default already does.

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
            bounds: Chunk bounds, in this adapter's own axis order.
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

        transfer = tuple(max(1, int(size)) for size in self.get_transfer_chunk_size())
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

    @staticmethod
    def _bounds_to_slices(bounds: ChunkBounds) -> Tuple[slice, ...]:
        """Per-axis ``slice`` tuple for indexing a backend array with ``bounds``.

        The bounds->slices idiom every ``get_data`` needs to turn chunk bounds
        into a numpy/zarr/h5py index; shared here so each adapter slices its
        store the same way.
        """
        return tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )

    @staticmethod
    def _bounds_to_strided_slices(
        bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Tuple[slice, ...]:
        """:meth:`_bounds_to_slices` with a per-axis stride, for a decimated read.

        Kept next to its unstrided sibling so the two index a store identically
        apart from the step -- which is the whole of what makes a fused
        ``nearest`` bit-identical to reading the extent and slicing it.
        """
        return tuple(
            slice(int(s), int(e), max(1, int(size)))
            for s, e, size in zip(bounds.start, bounds.stop, step, strict=True)
        )

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
        for ax, (s, e, dim) in enumerate(
            zip(bounds.start, bounds.stop, shape, strict=True)
        ):
            if s < 0:
                raise ValueError(f"Bounds start[{ax}]={s} is negative")
            if e > dim:
                raise ValueError(f"Bounds stop[{ax}]={e} exceeds shape[{ax}]={dim}")
            if s >= e:
                raise ValueError(f"Bounds start[{ax}]={s} >= stop[{ax}]={e}")

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

        # One key, one contract: the wire-protocol version the client enforces
        # (biopb/biopb#293). A `tensor_schema_version` release tag sat here too
        # until biopb/biopb#1070 -- it stopped meaning anything when its one
        # consumer (an shm feature probe) was replaced, and a release tag beside
        # a byte-encoding gate reads like a second gate. It was: the Java client
        # implemented it.
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
        # The two read gates, together and ahead of any I/O. Here rather than
        # in a writable mixin's override because that made the gate depend on
        # an adapter's base order, and on every override of this method
        # remembering to call up; both defaults are no-ops, so every adapter
        # is covered and a new one cannot forget.
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
                # The requested reduction_method rides in the chunk_id (#578), so a
                # do_get honors it; a byte-free (pre-#578) scaled chunk_id decodes
                # to area. Read and reduce through one call so an adapter that can
                # do both at once never materializes the full-resolution extent.
                result_arr = self.get_scaled_data(
                    bounds,
                    decode_scale_info(chunk_id),
                    decode_reduction_method(chunk_id),
                    cache_manager,
                )
            else:
                # The one read that is always a real decode: a scaled read may
                # be sourced from the cache (#965) and would time the cache's
                # own state instead. Measured here rather than inside
                # ``get_data`` so no adapter has to cooperate, and so a
                # delegating wrapper's transpose -- a view, materialized after
                # this returns -- stays out of the number.
                started = time.perf_counter()
                result_arr = self.get_data(bounds)
                if self._decode_time_is_rebuild_cost:
                    record_decode(
                        array_id, result_arr.nbytes, time.perf_counter() - started
                    )

            # Serialize into the unified binary wire schema: raw bytes + dtype
            # string, wrapped zero-copy. This preserves the exact dtype including
            # endianness (big-endian FITS '>i2' round-trips without conversion)
            # and avoids the per-element typed-list encoding (biopb/biopb#293).
            result = pack_chunk_batch(result_arr)
            return result, result_arr.nbytes

        if should_cache:
            # The method is part of the key, not advisory: since #578 the
            # chunk_id carries a method byte and cache_key_for_chunk_id keeps
            # it, so a nearest read cannot be served an area chunk (this
            # reverses biopb/biopb#76).
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

        chunk_size = self.get_transfer_chunk_size()
        # content_version is a SourceAdapter property; every TensorAdapter is a
        # SourceAdapter, so it is always present -- an unversioned source
        # returns None. The epoch is the codec's to add.
        return _get_read_plan(
            base_desc,
            request_desc,
            chunk_size,
            content_version=self.content_version,
        )

    # ---- native-pyramid precompute routing ---------------------------------
    # Turning a ``precompute`` read into a read against one on-disk level's store
    # is shared here, so the native-pyramid adapters (OME-Zarr multiscales,
    # QPTIFF) stop duplicating it near-verbatim (biopb/biopb#557). A leaf adapter
    # supplies only the per-format level lookup + scale extraction:
    # :meth:`_find_level_for_scale`, :meth:`_level_downsample_factors`, and
    # :meth:`get_level_adapter`.

    def _plan_precomputed_read(
        self, request_desc: TensorDescriptor, scale_hint: Tuple[int, ...]
    ) -> TensorReadPlan:
        """Plan a ``precompute`` read against the level matching ``scale_hint``.

        Find the native level whose downsample factors equal ``scale_hint``,
        translate the request's slice into that level's coordinates, and plan the
        read against the level's own store.
        """
        level = self._find_level_for_scale(scale_hint)
        if level is None:
            raise ValueError(
                f"No precomputed level matching scale_hint {tuple(scale_hint)}."
            )
        slice_hint = (
            request_desc.slice_hint if request_desc.HasField("slice_hint") else None
        )
        level_slice = _convert_slice_to_level(
            slice_hint, self._level_downsample_factors(level)
        )
        return self._plan_from_precomputed(level, level_slice)

    def _find_level_for_scale(self, scale_hint: Tuple[int, ...]) -> Optional[Any]:
        """Native level key whose downsample factors equal ``scale_hint``, else None.

        Overridden by native-pyramid adapters. The key is opaque to the shared
        routing -- an OME-Zarr dataset path, a QPTIFF integer index -- and only
        round-trips through :meth:`_level_downsample_factors` and
        :meth:`get_level_adapter`. The default advertises no native levels.
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
        back through :meth:`get_level_adapter`. The returned descriptor's
        ``array_id`` is reset to this tensor's, so the client still sees one tensor.
        """
        level_adapter = self.get_level_adapter(str(level))
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

    @staticmethod
    def _base_structural_descriptor(base_desc: TensorDescriptor) -> TensorDescriptor:
        """The stable per-tensor facts alone: shape/dtype/dim_labels/chunk_shape.

        Copies only the structural fields off ``base_desc``, deliberately dropping
        any pyramid / physical_scale / metadata_json the adapter's own
        ``get_tensor_descriptor`` may already carry -- ``plan_flight_info`` re-fills
        those under the response field masks (biopb/biopb#563), so a straight
        ``CopyFrom`` would leak an unmasked pyramid or scale into the response.
        """
        return TensorDescriptor(
            array_id=base_desc.array_id,
            dim_labels=base_desc.dim_labels,
            shape=base_desc.shape,
            chunk_shape=base_desc.chunk_shape,
            dtype=base_desc.dtype,
        )

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

        # Opt-in: an empty mask is a describe, and the O(chunks) plan is the
        # most expensive thing here, so it is never what saying nothing buys.
        mask = read_mask(read_opt)

        if ENDPOINTS in mask:
            request_desc = self._base_structural_descriptor(base_desc)
            if read_opt.HasField("slice_hint"):
                request_desc.slice_hint.CopyFrom(read_opt.slice_hint)
            # scale_hint / reduction_method route the read to a downsampled level.
            if read_opt.scale_hint:
                request_desc.scale_hint[:] = list(read_opt.scale_hint)
            if read_opt.reduction_method:
                request_desc.reduction_method = read_opt.reduction_method
            read_plan = self.get_read_plan(
                request_desc,
            )
        else:
            # Describe-only still exposes the server's transfer grid, never the
            # adapter's private file/dask read geometry (#684).
            desc = self._base_structural_descriptor(base_desc)
            desc.chunk_shape[:] = list(self.get_transfer_chunk_size())
            read_plan = TensorReadPlan(
                descriptor=desc,
                chunk_endpoints=[],
            )

        # Advertise the server-decided resolution pyramid (opt-in -- native-level
        # sizing is the costly part), then the compact physical scale (cheap,
        # always filled) -- both open-time only (never in list_flights).
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
        # Physical scale is constant for the lifetime of an adapter. Some format
        # adapters derive it from expensive resident metadata, so cache both a
        # value and a computed ``None`` result (there is no base __init__ shared
        # by all adapters).
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
            ndim = len(descriptor.dim_labels)
            if ndim and len(scale_vec) == ndim and len(unit_vec) == ndim:
                descriptor.physical_scale[:] = scale_vec
                descriptor.physical_unit[:] = unit_vec


# --- role-scope enforcement -------------------------------------------------
# The two role interfaces must stay disjoint *as declared* and match their
# declared scope, so a tensor-scoped method can never silently land on
# SourceAdapter again (the past scramble that this split fixes). TensorAdapter
# inherits SourceAdapter's methods, but must not re-declare or override any of
# them -- _public_api reads `vars(cls)`, so the checks below are about where a
# method is written, not what an instance can answer. Adding a public method to
# either ABC without classifying it here fails the equality check; any overlap
# fails the disjointness check. Underscore-private helpers are intentionally
# excluded.
_SOURCE_SCOPED_API = frozenset(
    {
        "array_id",
        "source_url",
        "source_type",
        "capability_token",
        "tensor_capability_token",
        "content_version",
        "check_chunk_version",
        "check_readable",
        "claim",
        "create_from_config",
        "list_tensor_descriptors",
        "get_metadata",
        "get_embedded_rois",
        "catalog_url",
        "resolve",
        "is_resident",
        "is_resolved",
        "get_tensor_adapter",
        "put_chunk",
        "close",
        "release_registration_cache",
        # attached tensors (biopb/biopb#1059)
        "get_embedded_labels",
        "label_sets",
        "label_uploads",
        "attached_fields",
        "attached_tensors",
        "attached_tensor",
        "attach_tensor",
        "detach_tensor",
        "attachment_changed",
        "label_binding_error",
        "label_image_axes",
        "label_image_descriptor",
        "resolve_tensor",
        "resolve_chunk_adapter",
        # the level lookup of the chunk route, which is source-scoped
        "get_level_adapter",
    }
)
_TENSOR_SCOPED_API = frozenset(
    {
        "get_tensor_descriptor",
        "get_transfer_chunk_size",
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

    Start floors and stop **ceils**, so the half-open range covers every level
    pixel the base range touches. Flooring both -- which this did until
    biopb/biopb#889 -- drops the partial pixel at a ragged end, and that is not
    a rounding taste: it disagrees with the computed path, which decimates with
    ``data[::s]`` and therefore returns ``ceil(extent / s)``. The two must
    agree, because the same region at the same scale is served either way
    depending only on whether the tensor happens to ship a pyramid. Worked
    through, a level read of ``[a, b)`` at factor ``f`` then reduced by ``r``
    yields ``ceil(ceil((b-a)/f)/r) == ceil((b-a)/(f*r))`` -- the computed count
    exactly. With a floored stop the identity breaks, and where the last tile is
    one pixel wide the result is empty rather than short.
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

    ``content_version`` (biopb/biopb#178), when set, is folded into every minted
    chunk_id so the cache namespaces by it -- alongside the serving-semantics
    epoch, which ``mint_chunk_id`` adds (biopb/biopb#1076).
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

    # STEP 1 was performed by TensorAdapter.get_transfer_chunk_size(). This
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
