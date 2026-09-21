"""Arrow Flight server for tensor storage.

This module implements a Flight server that exposes chunked multi-dimensional
arrays through the source/tensor adapter interface.

Three flights, and the dispatch is always a proto oneof (never a sentinel
or a byte-prefix sniff):

- ``catalog`` -- the public DuckDB catalog. ListFlights advertises one flight
  per table with its schema; GetFlightInfo / DoGet take a ``CatalogQuery``.
  Gated by the server-wide token.
- ``data`` -- pixels. GetFlightInfo takes a ``TensorReadOption`` and plans
  chunk endpoints; DoGet serves one ``chunk_id``; DoPut takes a
  ``ChunkUpload`` (writable servers). Private: gated per source.
- ``roi`` -- annotations. DoGet serves one tensor's set as ROI rows; DoPut
  takes a ``RoiPut`` / ``RoiDelete``. Private: gated per source.

Custom actions (``list_actions``) cover health, uploads, cache locate, cloud
resolve / warm, runtime source add / remove, and annotation pruning.
"""

import hmac
import json
import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.flight as flight
from biopb.image._roi_rows import (
    rois_to_table,
    table_to_roi_ids,
    table_to_rois,
)
from biopb.image.annotation_pb2 import (
    RoiDeleteResult,
    RoiPruneRequest,
    RoiPruneResult,
    RoiPutResult,
    RoiUnseen,
)
from biopb.tensor._session import split_array_id
from biopb.tensor._wire_version import FLIGHT_PROTOCOL_VERSION
from biopb.tensor.descriptor_pb2 import (
    AddSourceProgress,
    AddSourceRequest,
    AddSourceResult,
    AddSourceStreamMessage,
    CatalogQuery,
    FlightRequest,
    RemoveSourceRequest,
    RemoveSourceResult,
    ResolveProgress,
    ResolveStreamMessage,
    TensorDescriptor,
    UploadStatus as UploadStatusPb,
    WarmProgress,
    WarmStreamMessage,
)
from biopb.tensor.ticket_pb2 import (
    ChunkBounds,
    PutCommand,
    SetUploadStatus,
    TensorTicket,
)
from google.protobuf.message import DecodeError, Message

from biopb_tensor_server.adapters._writable import (
    SETTABLE_STATES,
    UploadProgress,
    UploadStatus,
    upload_of,
)
from biopb_tensor_server.adapters.labels import labels_root, sidecar_attacher
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import (
    SourceAdapter,
    TensorAdapter,
    strip_source_prefix,
)
from biopb_tensor_server.core.chunk import cache_key_for_chunk_id, routing_array_id
from biopb_tensor_server.core.config import PyramidConfig
from biopb_tensor_server.core.errors import (
    SourceResolveRetriableError,
    SourceUnresolvedError,
    TensorNotFound,
    TensorResolutionError,
    UnknownResolutionError,
)
from biopb_tensor_server.core.labels import split_label_field
from biopb_tensor_server.core.read_mask import (
    IS_RESIDENT,
    METADATA_JSON,
    UPLOAD_STATUS,
    read_mask,
)
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.core.retention import set_active_pyramid_config
from biopb_tensor_server.core.source_registry import SourceRegistry
from biopb_tensor_server.serving.activity import ActivityTracker
from biopb_tensor_server.serving.metadata_db import (
    MetadataDatabase,
    NumpyEncoder,
    is_reserved_set,
)
from biopb_tensor_server.serving.upload_manager import (
    DEFAULT_UPLOAD_TTL,
    UploadManager,
)

logger = logging.getLogger(__name__)

#: The reads a narrow grant can cover. Two, not one: pixels and annotations are
#: separate reads of the same source, and a bare capability token grants both
#: today -- naming them is what lets that stop being true without touching a
#: call site (biopb/biopb#1048).
READ_PIXELS = "read:pixels"
READ_ANNOTATIONS = "read:annotations"
_CAPABILITY_ACTIONS = frozenset({READ_PIXELS, READ_ANNOTATIONS})


def _ensure_tls_scheme(location: str) -> str:
    """Rewrite a location to Arrow Flight's TLS scheme (``grpc+tls://``).

    A TLS server binds a ``grpc+tls://`` location; callers commonly pass the
    plaintext ``grpc://`` (default) or the client-facing ``grpcs://`` shorthand,
    so accept either and normalize. An already-``grpc+tls://`` location is left
    untouched.
    """
    for prefix in ("grpc://", "grpcs://"):
        if location.startswith(prefix):
            return "grpc+tls://" + location[len(prefix) :]
    return location


def to_flight_error(exc: Exception) -> flight.FlightError:
    """Map a tensor-server domain error to a typed Flight error at the boundary.

    The single field/tensor-resolution boundary handler shared by the read verbs
    (``get_flight_info`` / ``do_get`` / the cache-file locate path, via
    ``_adapter_lookup_error``). pyarrow's Flight-in-Python exposes only a *subset*
    of gRPC's canonical status codes as typed exceptions -- there is no
    ``FlightNotFoundError``, ``FlightInvalidArgumentError``, or
    ``FlightFailedPreconditionError`` -- so this maps the domain taxonomy two ways
    at once, and keeps the two in step:

    - to the best-available typed class for **coarse retryability**: a retriable
      ``FlightUnavailableError`` for an unresolved source (open to resolve), or a
      terminal ``FlightServerError`` for a client error (NOT_FOUND /
      INVALID_ARGUMENT), never the "server bug, don't retry"
      ``FlightInternalError``, and
    - to the **precise** canonical code + machine reason in ``extra_info``
      (e.g. ``{"code": "NOT_FOUND", "reason": "unknown_field"}``), which a client
      can switch on despite the missing typed classes.

    Invariant: the ``extra_info`` code's retryability always matches the class the
    boundary picks -- an unresolved source is ``UNAVAILABLE`` under the retriable
    ``FlightUnavailableError`` (there is no representable ``FAILED_PRECONDITION``,
    and a blind retry is harmless since GetFlightInfo never resolves on serve),
    and every terminal code rides ``FlightServerError`` -- so a client switching on
    the class and one switching on the code never disagree.
    """
    code = getattr(exc, "grpc_code", "INTERNAL")
    reason = getattr(exc, "reason", None)
    payload = {"code": code, "reason": reason} if reason else {"code": code}
    extra_info = json.dumps(payload).encode()

    if isinstance(exc, SourceUnresolvedError):
        # Unresolved source -> retriable "open to resolve". The message MUST carry
        # "unresolved": the Python client's resolve-steering (biopb/tensor/
        # _session.py) catches FlightUnavailableError and substring-matches it.
        # grpc_code is UNAVAILABLE (errors.py), so the code matches the class.
        return flight.FlightUnavailableError(
            f"Source unresolved (open to resolve): {exc}", extra_info
        )
    # TensorNotFound / InvalidTensorId (and any other terminal domain error):
    # the caller's mistake, terminal -- do not retry -- but NOT an INTERNAL
    # "server bug". FlightServerError is the coarsest terminal class Flight offers.
    return flight.FlightServerError(str(exc), extra_info)


def _with_label_axes(metadata: dict, source_adapter: Any, desc: Any) -> dict:
    """*metadata* plus ``biopb.labels.image_axes`` when *desc* is a label set.

    The mapping the extent rule already implies -- which of the image's axes
    each axis of the set indexes -- said out loud, so a client reads it instead
    of re-deriving it from two descriptors (design, "Extent"). Stamped at the
    one place a tensor's served metadata is assembled, and measured against
    the descriptor being served, so it lines up with the axes the client is
    about to see.

    NGFF's own ``image-label`` block is left alone: that is a spec block and
    this is not in the spec. It rides in the ``biopb`` namespace beside it,
    merged into whatever the source already has there -- an ``ome_zarr:``
    upload's own ``.zattrs`` carry an upload marker -- rather than replacing it.

    ``source_id`` is the slash-free prefix by the identity policy, so the
    within-source field is everything after the first "/".
    """
    parsed = split_label_field(desc.array_id.partition("/")[2])
    if parsed is None or source_adapter is None:
        return metadata
    axes = source_adapter.label_image_axes(parsed.set_field, desc)
    if axes is None:
        return metadata
    biopb = {**(metadata.get("biopb") or {})}
    biopb["labels"] = {**(biopb.get("labels") or {}), "image_axes": list(axes)}
    return {**metadata, "biopb": biopb}


def _adapter_lookup_error(exc: Exception, miss_context: str) -> flight.FlightError:
    """Map an exception raised while resolving the adapter for a read request to a
    Flight error, identically for every read verb (biopb/biopb#378).

    ``get_flight_info`` / ``do_get`` / the cache-file locate path all resolve an
    adapter (``_get_adapter_for_tensor`` / ``_get_adapter_for_chunk``) before doing
    any work, and a miss there must map the same way no matter which verb hit it:

    - ``SourceUnresolvedError`` -> retriable "open to resolve"
      (``FlightUnavailableError``), via :func:`to_flight_error`.
    - ``TensorResolutionError`` -> the typed taxonomy (terminal NOT_FOUND /
      INVALID_ARGUMENT with the precise code in ``extra_info``).
    - any other bare exception (``ValueError`` / ``KeyError`` / ``AttributeError``
      / ``TypeError``) from an adapter that predates the typed taxonomy -> coerced
      to a terminal, honestly *unclassified* ``UNKNOWN`` -- never leaked as a
      "server bug" ``FlightInternalError``, but not a fabricated ``NOT_FOUND``
      either (once every adapter raises the taxonomy, a real field miss no longer
      reaches this fallback, so what lands here could as easily be a server bug as
      a client one -- ``UNKNOWN`` says exactly that).
    """
    if isinstance(exc, (SourceUnresolvedError, TensorResolutionError)):
        return to_flight_error(exc)
    return to_flight_error(
        UnknownResolutionError(f"{miss_context}: {exc}", reason="unclassified")
    )


# How often the ``resolve`` action emits an (empty-body) heartbeat Result while a
# resolution is in flight. Kept well under common proxy idle read timeouts
# (nginx ``grpc_read_timeout`` defaults to 60s) so a minutes-long recall doesn't
# get its stream reset.
_RESOLVE_HEARTBEAT_SECONDS = 15.0

# Block size for the ``warm`` action's recall reads, and the minimum interval
# between its progress messages. The min interval throttles the stream to a
# smooth UI cadence (rather than one message per file) while staying well under
# the proxy idle timeout, so it doubles as the heartbeat during a single large
# file's read. Enumeration (a long recall-free stat walk) falls back to the
# resolve heartbeat cadence.
_WARM_READ_BLOCK_BYTES = 8 * 1024 * 1024
_WARM_PROGRESS_MIN_INTERVAL = 0.5
_WARM_MAX_WORKERS = 4
_WARM_POLL_SECONDS = 0.1


class _AuthMiddleware(flight.ServerMiddleware):
    """Per-call middleware that carries the caller's presented Bearer token.

    Handlers retrieve it via ``context.get_middleware("auth")`` and decide what
    it opens (``TensorFlightServer._authorize`` / ``_authorize_read``).
    """

    def __init__(self, token: Optional[str]) -> None:
        self.token = token

    def sending_headers(self) -> dict:
        return {}

    def call_completed(self, exception: Optional[Exception]) -> None:
        pass


class BearerAuthMiddlewareFactory(flight.ServerMiddlewareFactory):
    """Capture the caller's presented Bearer token for the handlers.

    Deliberately decides nothing: which token a call needs depends on what it
    is about -- the server-wide token opens everything, while a source's own
    capability token opens that source's reads and nothing else -- and only the
    handler knows which it is asking about. A factory that rejected every
    non-server bearer up front would lock a capability holder out of the one
    source it may read (biopb/biopb#1010).

    Header value must be exactly ``Bearer <token>`` (case-sensitive).
    """

    def start_call(
        self,
        info: flight.CallInfo,
        headers: dict,
    ) -> Optional[flight.ServerMiddleware]:
        # Header values are lists; gRPC lowercases header names.
        values: List[str] = headers.get("authorization", [])
        bearer = values[0] if values else ""
        provided = bearer[len("Bearer ") :] if bearer.startswith("Bearer ") else None
        return _AuthMiddleware(provided)


def _fill_upload_status(
    desc: TensorDescriptor, upload: UploadProgress, source_id: str
) -> None:
    """Copy an upload's live progress onto the descriptor GetFlightInfo returns.

    Read from the record at response time and stored nowhere, which is what
    biopb/biopb#1035 requires of any live per-source fact. Shipping it on the
    descriptor is not a second home for it -- nothing keeps the copy, and both
    client caches strip it.

    An unrecognized state is left unset rather than guessed at, so a client
    reads "I do not know" instead of a wrong PENDING.
    """
    _copy_upload_status(desc.upload_status, upload.as_status_dict(source_id))


def _copy_upload_status(pb: UploadStatusPb, status: Dict[str, Any]) -> None:
    """The manager's status dict onto the wire message.

    Shared by the descriptor field and the ``set_upload_status`` reply so the
    two cannot drift into describing the same upload differently.
    """
    state = _UPLOAD_STATES.get(status["state"])
    if state is None:
        return
    pb.state = state
    pb.expected_chunks = int(status["expected_chunks"])
    pb.uploaded_chunks = int(status["uploaded_chunks"])
    pb.reason = status.get("reason") or ""


#: ``UploadStatus.state`` strings -> the wire enum. UNKNOWN is deliberately
#: absent: it is what the manager says about a source tracking no upload, and
#: such a source leaves the whole field unset instead.
_UPLOAD_STATES = {
    "PENDING": UploadStatusPb.PENDING,
    "READY": UploadStatusPb.READY,
    "FINISHED": UploadStatusPb.FINISHED,
    "DISCARDED": UploadStatusPb.DISCARDED,
}


#: The wire enum -> what ``set_upload_status`` may ask for. The inverse of
#: ``_UPLOAD_STATES`` restricted to the settable states, so a request naming
#: PENDING or an unrecognized value is refused at the boundary rather than
#: reaching an adapter that would refuse it anyway with a worse message.
_UPLOAD_TARGETS = {
    UploadStatusPb.READY: UploadStatus.READY,
    UploadStatusPb.FINISHED: UploadStatus.FINISHED,
    UploadStatusPb.DISCARDED: UploadStatus.DISCARDED,
}


def _roi_source_id(array_id: str) -> str:
    """The source an annotation's tensor belongs to, for authorization.

    Same split-on-the-first-'/' rule the ticket path uses: array_id is
    authoritative, source_id is the prefix before the first '/'.
    """
    if not array_id:
        raise ValueError("array_id is required")
    source_id, _ = split_array_id(array_id)
    return source_id


class TensorFlightServer(flight.FlightServerBase):
    """Arrow Flight server for tensor storage.

    This server exposes multi-dimensional arrays through the Flight protocol,
    with each chunk represented as a separate FlightEndpoint.

    Supports multifield acquisitions where tensors within a data source
    have different shapes (e.g., MicroManager multi-position datasets).

    State is delegated to three collaborators (biopb/biopb#278 item A), so this
    class stays a thin Flight protocol handler:

    - ``self.sources`` (:class:`SourceRegistry`) -- the ``source_id`` -> adapter
      map, registration chokepoint, and adapter-lifecycle cleanup.
    - ``self.activity`` (:class:`ActivityTracker`) -- in-flight-read counters
      (the precache idle signal) and the warm-in-progress guard.
    - ``self.uploads`` (:class:`UploadManager`) -- the writable-server DoPut path
      (source creation, chunk writing, upload-progress state).

    The normal entry point is the ``biopb-tensor-server`` CLI
    (``launch`` / ``serve``), which scans the data folder and calls
    ``mark_ready()`` for you once registration completes. The low-level usage
    below drives the server directly and must therefore mark itself ready after
    registering its sources, or the ``health`` action reports ``STARTING``
    forever.

    Usage:
        # Create an adapter for your data
        import zarr
        arr = zarr.open_array('data.zarr', mode='r')
        adapter = ZarrAdapter(arr, 'my-tensor')

        # Start the server. Pass a catalog if the sources must be browsable:
        # registration is the registry, cataloguing is a separate second step
        # and the registering caller's job (see ``metadata_db``).
        db = MetadataDatabase()
        server = TensorFlightServer('grpc://0.0.0.0:8815', metadata_db=db)
        db.sync_source_added('my-tensor', server.register_source('my-tensor', adapter))
        server.mark_ready()  # registration done -> health reports SERVING
        server.serve()
    """

    def __init__(
        self,
        location: str = "grpc://0.0.0.0:8815",
        token: Optional[str] = None,
        writable: bool = False,
        write_dir: Optional[Path] = None,
        metadata_db: Optional[MetadataDatabase] = None,
        annotations_enabled: bool = True,
        grpc_max_message_size: Optional[int] = None,
        pyramid_config: Optional[PyramidConfig] = None,
        tls_cert_chain: Optional[bytes] = None,
        tls_private_key: Optional[bytes] = None,
        upload_ttl: float = DEFAULT_UPLOAD_TTL,
        **kwargs,
    ):
        """Initialize the Flight server.

        Args:
            location: Server location (e.g., 'grpc://0.0.0.0:8815')
            token: The server-wide Bearer token (the catalog tier, and the
                fallback for every source without a capability token of its
                own). ``None`` disables it.
            writable: Enable write mode for source creation and data upload
            write_dir: Directory for zarr-backed uploaded sources (required if writable)
            upload_ttl: Seconds before a PENDING upload with no writes is
                discarded and a discarded one is unregistered
                (``UploadManager.reap``); 0 disables the sweep.
            metadata_db: The catalog -- the browse surface behind the
                ``catalog`` and ``roi`` flights. ``None`` builds a catalog-less
                server: its sources are addressed by ``source_id`` and served,
                but nothing can be listed, queried or annotated, and every
                catalog surface refuses with ``FlightUnavailableError``. That
                is the embedded in-process cache's shape (biopb-image-runtime),
                where a result's id goes back to its one caller directly.
            annotations_enabled: Serve the ``roi`` flight. Not tied to
                ``writable``: an annotation writes no pixels, so the token is its
                boundary. This is the switch for a deployment that wants a
                strictly read-only catalog.
            grpc_max_message_size: gRPC max message size in bytes (default: 16MB)
            tls_cert_chain: PEM-encoded server certificate chain. When supplied
                together with ``tls_private_key`` the server serves TLS: the
                location scheme is forced to ``grpc+tls://`` and clients must
                connect with ``grpcs://`` (see ``TensorFlightClient``). Pass
                neither for a plaintext ``grpc://`` server; passing exactly one
                is an error.
            tls_private_key: PEM-encoded private key matching ``tls_cert_chain``.
            **kwargs: Additional arguments passed to FlightServerBase
        """
        # TLS is all-or-nothing: a cert without its key (or vice versa) can't
        # serve, so fail loudly rather than silently drop to plaintext.
        if (tls_cert_chain is None) != (tls_private_key is None):
            raise ValueError(
                "tls_cert_chain and tls_private_key must be provided together"
            )
        if tls_cert_chain is not None:
            # FlightServerBase serves TLS only when the location scheme says so;
            # accept the plaintext/grpcs shorthands and rewrite to Arrow's form.
            location = _ensure_tls_scheme(location)
            kwargs["tls_certificates"] = [(tls_cert_chain, tls_private_key)]

        # Apply gRPC max message size via URL query parameter
        if grpc_max_message_size:
            separator = "&" if "?" in location else "?"
            location = f"{location}{separator}grpc.max_send_message_size={grpc_max_message_size}&grpc.max_receive_message_size={grpc_max_message_size}"

        middleware = kwargs.pop("middleware", {})
        middleware.setdefault("auth", BearerAuthMiddlewareFactory())
        super().__init__(location, middleware=middleware, **kwargs)
        # Uploaded label sets live under write_dir/labels/<source_id>/ and are
        # attached to their source at registration (biopb/biopb#1059).
        self.sources = SourceRegistry(
            on_register=sidecar_attacher(labels_root(Path(write_dir)))
            if write_dir is not None
            else None
        )
        self._writable = writable
        # The catalog, or None for a catalog-less server. The server never
        # writes it: registering a source and cataloguing it are two steps, and
        # the caller that registers owns the second one, because only it knows
        # whether a failed catalog write should roll the registration back (the
        # reconciler) or be swallowed (an upload, whose id already reached its
        # one client).
        self._metadata_db: Optional[MetadataDatabase] = metadata_db
        self._annotations_enabled = annotations_enabled
        # The server-wide token: what the public catalog tier requires, and
        # what a private source without a capability token of its own falls
        # back to. None disables it (local mode).
        self._server_token: Optional[str] = token or None
        # Authoritative resolution-pyramid knobs. Used to advertise
        # TensorDescriptor.pyramid in get_flight_info (computed levels) and shared
        # with the precache worker so the warmed scales can't drift from the
        # advertised ones.
        self._pyramid_config = pyramid_config or PyramidConfig()
        # The read path classifies each chunk's retention against this ladder.
        # Installed once here rather than threaded through resolve_chunk_data
        # and every override of it (see core.retention).
        set_active_pyramid_config(self._pyramid_config)
        self._start_time: float = time.time()
        # DoPut upload path: source creation, chunk writes, and per-source upload
        # progress. Registers created sources through the shared registry.
        self.uploads = UploadManager(
            self.sources, write_dir, self._metadata_db, ttl=upload_ttl
        )
        # What a crashed server left half-written goes before anything can
        # register it: the caller's discovery scan runs after this returns.
        self.uploads.discard_unfinished_stores()
        # Reclaims dead uploads and aged tombstones (``UploadManager.reap``);
        # stopped in ``shutdown``.
        self.uploads.start_sweep()
        # Readiness gate: the Flight port binds (and gRPC starts serving) in the
        # base __init__ above, *before* the caller scans/registers the data
        # folder -- a scan that can be slow for large catalogs. Until the caller
        # finishes that scan and calls ``mark_ready()``, the ``health`` action
        # reports ``STARTING`` so a connecting client can tell "booting" apart
        # from "down" and wait instead of timing out. Set on the main thread,
        # read from gRPC handler threads, hence an Event.
        self._ready = threading.Event()

        # Flight activity + warm-guard tracking for the background precache
        # worker: counts in-flight heavy reads (do_get/warm), stamps the last one
        # to finish (so the worker parks while real traffic flows), and holds the
        # set of sources with a warm in flight (so a concurrent warm of the same
        # source is rejected). Cheap -- one uncontended lock.
        self.activity = ActivityTracker()

        # Catalog-freshness signals for the ``health`` action (progressive
        # discovery, biopb/biopb#212). ``SERVING`` only means "up and serving the
        # possibly-still-populating catalog"; these two fields carry *how fresh*
        # the catalog is. Written by the SourceManager's single event-loop thread
        # via the setters below, read from gRPC handler threads -- guarded by a
        # dedicated lock so a health read never contends with catalog/activity
        # locks. ``None`` until the first full scan succeeds.
        self._scan_status_lock = threading.Lock()
        self._full_scan_in_progress = False
        self._last_full_scan_at: Optional[float] = None

        # Runtime source registration (the "add_source" Flight action / tensor-
        # browser drag-drop). The SourceManager injects its ``add_local_source``
        # generator via ``set_add_source_handler`` at launch (the server holds no
        # SourceManager reference otherwise). ``None`` means the feature is
        # unavailable (e.g. a server with no source manager); the action then
        # reports a clear error. Distinct from ``_writable`` (upload mode): a
        # normal read-only server still registers dropped local files, so this
        # gates on its own flag defaulting on -- a hardened deployment can set it
        # off to refuse runtime path registration.
        self._add_source_handler: Optional[Callable[..., Any]] = None
        self._allow_runtime_source_add = True

        # Runtime removal of a drag-dropped source branch (the "remove_source"
        # action / tensor-browser [x] button). Injected via
        # ``set_remove_source_handler`` alongside the add handler. Gated on the
        # SAME ``_allow_runtime_source_add`` flag: a server that cannot add has no
        # dnd:// sources to remove, so removal is a no-op there anyway.
        self._remove_source_handler: Optional[Callable[..., Any]] = None

    def flight_idle_for(self, seconds: float) -> bool:
        """True if no heavy read is in flight and none finished within *seconds*.

        Used by the precache worker to debounce against live traffic
        (delegates to ``activity``).
        """
        return self.activity.idle_for(seconds)

    def mark_ready(self) -> None:
        """Signal that initial source registration is complete.

        Flips the ``health`` action's status from ``STARTING`` to ``SERVING``.
        Called once by the startup path after the data folder has been scanned
        and all sources registered.
        """
        self._ready.set()

    @property
    def is_ready(self) -> bool:
        """Whether initial source registration has completed."""
        return self._ready.is_set()

    def set_full_scan_in_progress(self, in_progress: bool) -> None:
        """Record whether a full catalog rescan is running right now.

        Called by the SourceManager around a force-full rescan; surfaced on the
        ``health`` action so a client can tell "a scan is running" from "idle".
        """
        with self._scan_status_lock:
            self._full_scan_in_progress = bool(in_progress)

    def set_last_full_scan(self, timestamp: float) -> None:
        """Record the epoch-seconds time a full catalog rescan last succeeded.

        Surfaced on ``health`` as ``last_full_scan_finished_at`` -- the catalog
        freshness signal that unifies boot with steady-state periodic rescans.
        """
        with self._scan_status_lock:
            self._last_full_scan_at = float(timestamp)

    def set_add_source_handler(self, handler: Optional[Callable[..., Any]]) -> None:
        """Wire the SourceManager's ``add_local_source`` for the add_source action.

        The server holds no SourceManager reference; the launcher injects the
        handler here so the ``add_source`` Flight action can route a dropped path
        into the claim -> adapter -> catalog pipeline. ``None`` leaves the action
        reporting "not enabled".
        """
        self._add_source_handler = handler

    def set_remove_source_handler(self, handler: Optional[Callable[..., Any]]) -> None:
        """Wire the SourceManager's ``remove_dropped_root`` for the remove_source action.

        Injected by the launcher alongside ``set_add_source_handler`` (the server
        holds no SourceManager reference). ``None`` leaves the action reporting
        "not enabled".
        """
        self._remove_source_handler = handler

    @property
    def metadata_db(self) -> Optional[MetadataDatabase]:
        """The catalog behind the ``catalog`` and ``roi`` flights, or ``None``.

        Public because registering a source does not catalogue it: a caller that
        wants its source browsable calls ``metadata_db.sync_source_added``
        itself, under whatever failure policy that call site owes its own caller
        (the reconciler rolls the registration back, an upload swallows it).
        ``None`` on a catalog-less server -- see ``_require_catalog``.
        """
        return self._metadata_db

    def _require_catalog(self) -> MetadataDatabase:
        """The catalog, or the refusal a catalog-less server owes its client.

        ``metadata_db=None`` is a deployment shape, not a degenerate one: the
        embedded in-process cache hands each result's ``source_id`` straight
        back to the caller that asked for it, so there is nothing to browse and
        no store to annotate into. Unavailable rather than ServerError, the same
        distinction ``_require_annotations`` draws: "this server does not offer
        the feature", not "your request was bad".
        """
        if self._metadata_db is None:
            raise flight.FlightUnavailableError(
                "This server has no catalog: its sources are addressed by "
                "source_id, not listed"
            )
        return self._metadata_db

    def register_source(self, source_id: str, adapter: SourceAdapter) -> SourceAdapter:
        """Register a data source with the server (delegates to ``sources``).

        The registry only -- cataloguing is the caller's second step, see
        ``metadata_db``.

        Returns the adapter as registered: the registry normalizes a
        non-canonical axis order on the way in (biopb/biopb#596), so a caller
        that keeps using the adapter afterwards must use the returned one.
        """
        return self.sources.register(source_id, adapter)

    def swap_source(
        self, source_id: str, adapter: SourceAdapter
    ) -> Tuple[SourceAdapter, Optional[SourceAdapter]]:
        """Replace a registered source's adapter in place (delegates to ``sources``).

        Returns ``(registered, displaced)``. Upload state is deliberately NOT
        forgotten: the source is not going away, only its adapter is being
        rebuilt against the current bytes. The displaced adapter is left open
        for the caller to close once in-flight reads have drained.
        """
        return self.sources.swap(source_id, adapter)

    def unregister_source(self, source_id: str) -> None:
        """Unregister a data source (its upload state, if any, goes with it).

        The registry only, mirroring ``register_source``.
        """
        self.sources.unregister(source_id)

    def shutdown(self) -> None:
        """Release source-adapter resources, then shut down the Flight server.

        Some adapters hold long-lived OS handles (e.g. the OME-TIFF adapter's
        persistent aszarr store). Closing them on shutdown releases those
        handles -- required on Windows, where an open file cannot be deleted
        (otherwise a test's TemporaryDirectory cleanup raises WinError 32).
        """
        self.uploads.stop_sweep()
        self.sources.close_all()
        super().shutdown()

    def _presented_token(self, context: flight.ServerCallContext) -> Optional[str]:
        # In-process callers (tests, the embedded runtime) pass no context.
        mw = context.get_middleware("auth") if context is not None else None
        return getattr(mw, "token", None) if mw is not None else None

    def _has_full_access(self, provided: Optional[str]) -> bool:
        """The server-wide token, which grants everything.

        False in local mode (no token configured): there is no *credential*
        granting it. The distinction matters one caller up, where a
        capability-gated source must stay gated in local mode.
        """
        if self._server_token is None:
            return False
        return provided is not None and hmac.compare_digest(
            provided, self._server_token
        )

    def _grants(
        self, provided: Optional[str], action: str, source_id: str
    ) -> Optional[bool]:
        """Does *provided* carry a narrow grant covering (*action*, *source_id*)?

        ``None`` means the source carries no grant at all, which is not a
        refusal -- it is "this object has opted into nothing, so the ordinary
        rule applies". ``False`` is a real refusal.

        Today a grant is a token on the adapter covering both reads of its own
        source, so the body is an equality test. A grant table or a signed
        token (biopb/biopb#1048) replaces this body and nothing else: call
        sites ask here rather than comparing tokens themselves.
        """
        adapter = self.sources.get(source_id)
        expected = adapter.capability_token if adapter is not None else None
        if not expected:
            return None
        if provided is None or not hmac.compare_digest(provided, expected):
            return False
        return action in _CAPABILITY_ACTIONS

    def _authorize(self, context: flight.ServerCallContext) -> None:
        """Full access: everything that is not a narrow read.

        The catalog flights and SQL, health, cache stats, every ``do_action``,
        and every write. Requires the server-wide token when one is configured;
        open otherwise (local mode -- the machine is the boundary).

        **A capability never reaches this.** Actions are the control surface,
        and a grant meaning "read this one tensor" must not authorize, say,
        ``warm`` -- whose cost is not scoped to that tensor at all, since it
        walks the page-cache LRU and evicts the segments serving every other
        source (biopb/biopb#1043).
        """
        if self._server_token is None:
            return
        if not self._has_full_access(self._presented_token(context)):
            raise flight.FlightUnauthenticatedError("Invalid or missing Bearer token")

    def _authorize_read(
        self, context: flight.ServerCallContext, source_id: str, action: str
    ) -> None:
        """Full access, or a narrow grant covering this read of this source.

        The server-wide token is checked first and grants everything, so a
        capability *adds* access rather than replacing it -- do not reorder
        these (biopb/biopb#1048).

        A source carrying no grant is as open as the catalog is, so it falls
        through to :meth:`_authorize`. A source carrying one stays gated even in
        local mode: that is why the embedded result cache can mint them on a
        server with no server-wide token at all.

        Knowing a source_id is not what this gates -- a private source may still
        be catalogued. Reading it is.
        """
        provided = self._presented_token(context)
        if self._has_full_access(provided):
            return
        granted = self._grants(provided, action, source_id)
        if granted:
            return
        if granted is None:
            self._authorize(context)
            return
        raise flight.FlightUnauthenticatedError("Invalid or missing source token")

    @staticmethod
    def _parse(msg: Message, data: bytes, what: str) -> Message:
        """Decode a wire message, or refuse the call with the reason.

        The oneof arm is the dispatch, so a payload that decodes but sets no
        arm is refused here too -- that is what a protocol-1 client's request
        or bare ticket looks like, and the message says so.
        """
        try:
            msg.ParseFromString(data)
        except DecodeError as exc:
            raise flight.FlightServerError(
                f"{what} is not a {type(msg).__name__}: {exc}"
            )
        if not msg.ListFields():
            raise flight.FlightServerError(
                f"{what} names nothing: expected a {type(msg).__name__} with one arm "
                f"set (this server speaks Flight protocol v{FLIGHT_PROTOCOL_VERSION})"
            )
        return msg

    def _parse_ticket(self, ticket: flight.Ticket) -> TensorTicket:
        """Parse a TensorTicket from a Flight Ticket."""
        return self._parse(TensorTicket(), ticket.ticket, "ticket")

    def _encode_metadata(self, bounds: ChunkBounds) -> bytes:
        """Encode ChunkBounds to bytes for app_metadata.

        Args:
            bounds: Chunk bounds to encode

        Returns:
            Serialized bytes
        """
        return bounds.SerializeToString()

    @staticmethod
    def _field_within_source(source_id: str, tensor_id: str) -> Optional[str]:
        """Reduce a request tensor_id to the within-source field, or ``None``
        meaning *select the source's default (first) tensor*.

        ``None`` has exactly one meaning here, which ``get_tensor_adapter``
        honors. Both inputs that name no within-source field map to it: a bare
        ``source_id`` (single-tensor source addressed by its own id) and an
        unset/empty id (a degenerate request whose default-substitution upstream
        could not resolve). Any real field is returned verbatim -- the
        ``== source_id`` test runs *before* the strip, so a genuine field that
        happens to equal the source_id (array_id ``src/src`` -> field ``src``) is
        preserved rather than collapsing to ``None``. The prefix strip itself is
        the shared :func:`strip_source_prefix` (identity policy: array_id is
        ``source_id`` or ``source_id/field``, source_id slash-free).
        """
        if not tensor_id or tensor_id == source_id:
            return None
        return strip_source_prefix(source_id, tensor_id)

    @staticmethod
    def _catalog_endpoint(sql: str) -> flight.FlightEndpoint:
        """The one endpoint every catalog ``FlightInfo`` carries: a ticket with
        the SQL itself, so nothing is parked server-side between this call and
        the DoGet.
        """
        ticket = TensorTicket(catalog_query=CatalogQuery(sql=sql))
        return flight.FlightEndpoint(
            ticket=flight.Ticket(ticket.SerializeToString()), locations=[]
        )

    def _catalog_flight_info(self, table: str) -> flight.FlightInfo:
        """One catalog table as a flight: its real schema and a ticket that
        reads it whole. What ListFlights advertises and what GetFlightInfo on
        the table's path answers."""
        try:
            schema = self._require_catalog().table_schema(table)
        except ValueError as e:
            raise flight.FlightServerError(str(e)) from e
        return flight.FlightInfo(
            schema=schema,
            descriptor=flight.FlightDescriptor.for_path(table),
            endpoints=[self._catalog_endpoint(f"SELECT * FROM {table}")],
            total_records=-1,
            total_bytes=-1,
        )

    def _get_adapter_for_tensor(
        self, source_id: str, tensor_id: str
    ) -> Optional[TensorAdapter]:
        """Get adapter for a specific tensor within a source.

        Args:
            source_id: The data source identifier
            tensor_id: The within-source field name (already reduced from any
                source-qualified array_id by ``_field_within_source``), or None
                for the source's sole/default tensor.

        Returns:
            TensorAdapter for the specified tensor, or None if not found
        """
        source_adapter = self.sources.get(source_id)
        if source_adapter is None:
            return None

        return source_adapter.resolve_tensor(tensor_id)

    def _get_adapter_for_chunk(self, chunk_id: bytes) -> TensorAdapter:
        """Get the adapter responsible for a chunk, by its chunk_id.

        Raises rather than returning None, and maps a lookup failure itself, so
        every verb that resolves a chunk -- ``do_get`` and the cache-file locate
        path -- reports a miss identically without restating the mapping
        (biopb/biopb#378):

        - a lookup failure (unresolved source, a ticket naming a field the source
          no longer has, a legacy adapter raising) goes through
          :func:`_adapter_lookup_error`: retriable "open to resolve", or terminal
          NOT_FOUND / INVALID_ARGUMENT *with a code* -- never a bare exception,
          which Flight would surface as INTERNAL.
        - a stale ticket whose source is no longer registered -> terminal
          NOT_FOUND with ``reason="unknown_source"``, matching
          ``get_flight_info``'s tensor_adapter-is-None sibling.

        Args:
            chunk_id: The chunk identifier bytes

        Returns:
            The TensorAdapter responsible for the chunk

        Raises:
            flight.FlightError: mapped per the taxonomy above
        """
        try:
            # routing_array_id handles both a plain/versioned chunk_id and a proxy
            # envelope (whose route token IS the local array_id) without decoding an
            # opaque envelope inner (biopb/biopb#178 W1).
            array_id = routing_array_id(chunk_id)
            source_id, *rest = array_id.split("/")
            rest = "/".join(rest) if rest else None

            adapter = None
            source_adapter = self.sources.get(source_id)
            if source_adapter is not None:
                # A within-source suffix names a native pyramid level, a tensor
                # field, or a label set (and a level under it); the source
                # decides which (``SourceAdapter.resolve_chunk_adapter``).
                adapter = source_adapter.resolve_chunk_adapter(rest)
        except (
            SourceUnresolvedError,
            TensorResolutionError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
        ) as e:
            raise _adapter_lookup_error(
                e, f"Tensor not found for chunk {chunk_id[:16]!r}"
            ) from e

        if adapter is None:
            raise to_flight_error(
                TensorNotFound(
                    f"Adapter not found for chunk_id: {chunk_id[:16]!r}",
                    reason="unknown_source",
                )
            )
        return adapter

    def _require_annotations(self) -> MetadataDatabase:
        """The store behind the ``roi`` flight, or the refusal.

        Unavailable vs. ServerError is load-bearing for the HTTP sidecar: it
        maps the former to 501 (this server does not offer the feature) and the
        latter to 422 (your request was rejected).
        """
        if not self._annotations_enabled:
            raise flight.FlightUnavailableError(
                "ROI annotations are disabled on this server"
            )
        # Annotations live in the catalog's store, so a catalog-less server has
        # nowhere to put them either.
        return self._require_catalog()

    def _handle_roi_prune(self, req: RoiPruneRequest) -> bytes:
        """Report, and with ``apply`` delete, annotations whose source is gone.

        Orphans have no live source to authorize against, which is why this is
        a catalog-tier action rather than a ``roi`` flight verb. The report
        and the delete share the store's one predicate, so what was shown is
        what goes.
        """
        db = self._require_annotations()
        if req.unseen_days <= 0:
            raise flight.FlightServerError("roi_prune: unseen_days must be positive")
        before = datetime.now(timezone.utc) - timedelta(days=req.unseen_days)
        unseen = [
            RoiUnseen(
                array_id=u.array_id,
                source_url=u.source_url or "",
                count=u.count,
                last_seen_at_unix_ms=(
                    int(u.last_seen_at.timestamp() * 1000) if u.last_seen_at else 0
                ),
            )
            for u in db.unseen_rois(before)
        ]
        deleted = db.prune_unseen(before) if req.apply and unseen else 0
        return RoiPruneResult(unseen=unseen, deleted=deleted).SerializeToString()

    def list_actions(
        self,
        context: flight.ServerCallContext,
    ) -> List[flight.ActionType]:
        """List available actions on this server."""
        self._authorize(context)
        return [
            flight.ActionType("health", "Health check - returns server status JSON"),
            flight.ActionType(
                "create_tensor",
                "Create a writable single-tensor source from a TensorDescriptor",
            ),
            flight.ActionType(
                "set_upload_status",
                "Move an upload: READY (publish), FINISHED (seal), DISCARDED (give up)",
            ),
            flight.ActionType(
                "chunk_locate", "Locate a cached chunk on disk for localhost mmap reads"
            ),
            flight.ActionType(
                "cache_stats", "Cache statistics - returns backend CacheStats JSON"
            ),
            flight.ActionType(
                "warm",
                "Hydrate-ahead: recall a resolved cloud source's member files server-side",
            ),
            flight.ActionType(
                "add_source",
                "Register a local path/dir as a served source at runtime (streams progress)",
            ),
            flight.ActionType(
                "remove_source",
                "Deregister a drag-dropped (dnd://) source branch at runtime",
            ),
            flight.ActionType(
                "roi_prune",
                "Report (or with apply, delete) annotations whose source is gone",
            ),
        ]

    def do_action(
        self,
        context: flight.ServerCallContext,
        action: flight.Action,
    ) -> Iterator[bytes]:
        """Execute a custom action.

        Every arm takes full access (:meth:`_authorize`). Actions are the
        control surface: a capability means "read this one tensor", and none of
        what is reachable here is scoped to one tensor -- ``warm`` walks the
        page-cache LRU and evicts the segments serving every other source
        (biopb/biopb#1043), and a mutation on a source is never covered by a
        read grant on it. ``chunk_locate`` is no exception, though it looks
        like one: it is the localhost handoff for a read, and the read itself
        (``do_get``) is where a capability is honoured.

        Args:
            context: Server call context
            action: Action to execute

        Yields:
            Result bytes (JSON-encoded for health action)
        """
        if action.type == "health":
            self._authorize(context)
            uptime_seconds = int(time.time() - self._start_time)
            with self._scan_status_lock:
                full_scan_in_progress = self._full_scan_in_progress
                last_full_scan_at = self._last_full_scan_at
            db = self._metadata_db
            health_status = {
                "status": "SERVING" if self._ready.is_set() else "STARTING",
                # The Flight protocol shape this server speaks; the SDK checks
                # it before its first call. A server without the key is v1.
                "protocol": FLIGHT_PROTOCOL_VERSION,
                "source_count": len(self.sources),
                # Whether this server offers a catalog at all. A constant True
                # since #225 ("every server has a catalog now"), which is the
                # assumption retiring _owns_catalog removes -- so it is a real
                # signal again, and the one a client can read *before* calling a
                # catalog surface rather than eating its refusal.
                "metadata_db_enabled": db is not None,
                "writable": self._writable,
                "uptime_seconds": uptime_seconds,
                # Catalog-freshness signals (progressive discovery). ``SERVING``
                # no longer implies a complete catalog; these say whether a full
                # scan is running and when one last finished (epoch seconds, or
                # null until the first full scan succeeds). See biopb/biopb#212.
                "full_scan_in_progress": full_scan_in_progress,
                "last_full_scan_finished_at": last_full_scan_at,
                # Whether drawn ROIs survive a restart. A store that was asked
                # for and could not be opened is fatal at startup, so this is
                # False for a deliberately session-only server, or one with no
                # catalog at all (which cannot take annotations either) -- both
                # of which a client may want to say out loud before someone
                # spends a morning tracing.
                "annotations_persisted": db is not None and db.annotations_persisted,
                # The same question one level down, and not the same answer: the
                # catalog also holds `decode_rates`, and a server with the
                # annotation actions off keeps a file for those alone. A sibling
                # key rather than a redefinition -- `annotations_persisted` is
                # already on the wire and means what it says.
                "catalog_persisted": db is not None and db.store_path is not None,
            }
            yield json.dumps(health_status).encode("utf-8")
        elif action.type == "create_tensor":
            self._authorize(context)
            if not self._writable:
                raise flight.FlightUnauthenticatedError("Server not in write mode")

            req_desc = TensorDescriptor.FromString(action.body.to_pybytes())
            yield self.uploads.create_tensor(req_desc).SerializeToString()
        elif action.type == "set_upload_status":
            self._authorize(context)
            if not self._writable:
                raise flight.FlightUnauthenticatedError("Server not in write mode")

            req = self._parse(
                SetUploadStatus(), action.body.to_pybytes(), "set_upload_status request"
            )
            state = _UPLOAD_TARGETS.get(req.state)
            if state is None:
                raise flight.FlightServerError(
                    f"set_upload_status: {UploadStatusPb.State.Name(req.state)} is "
                    f"not a state an upload can be moved to; use one of "
                    f"{', '.join(s.value for s in SETTABLE_STATES)}."
                )
            status = self.uploads.set_status(req.array_id, state, req.reason)
            reply = UploadStatusPb()
            _copy_upload_status(reply, status)
            yield reply.SerializeToString()
        elif action.type == "chunk_locate":
            self._authorize(context)
            ticket = self._parse_ticket(flight.Ticket(action.body.to_pybytes()))
            if ticket.WhichOneof("payload") != "chunk_id":
                raise flight.FlightServerError("chunk_locate takes a chunk ticket")
            yield self._handle_chunk_locate(ticket.chunk_id).encode("utf-8")
        elif action.type == "cache_stats":
            self._authorize(context)
            from dataclasses import asdict

            manager = CacheManager.get_instance()
            if manager is None:
                raise flight.FlightServerError("Cache not initialized")
            # asdict recurses into the per-pool PoolStats dataclasses under pool_stats.
            yield json.dumps(asdict(manager.stats())).encode("utf-8")
        elif action.type == "resolve":
            self._authorize(context)
            source_id = action.body.to_pybytes().decode("utf-8")
            yield from self._handle_resolve(source_id)
        elif action.type == "warm":
            self._authorize(context)
            source_id = action.body.to_pybytes().decode("utf-8")
            yield from self._handle_warm(source_id, context)
        elif action.type == "add_source":
            self._authorize(context)
            req = AddSourceRequest.FromString(action.body.to_pybytes())
            yield from self._handle_add_source(req, context)
        elif action.type == "remove_source":
            self._authorize(context)
            req = RemoveSourceRequest.FromString(action.body.to_pybytes())
            yield self._handle_remove_source(req)
        elif action.type == "roi_prune":
            self._authorize(context)
            req = RoiPruneRequest.FromString(action.body.to_pybytes())
            yield self._handle_roi_prune(req)
        else:
            self._authorize(context)
            raise flight.FlightServerError(f"Unknown action: {action.type}")

    def _handle_resolve(self, source_id: str) -> Iterator[bytes]:
        """Stream the result of resolving a source.

        Resolution is the ONE consented, possibly minutes-long recall (it may
        download a whole cloud / synced-folder file). It runs on a daemon thread
        so this handler can emit ``ResolveStreamMessage`` progress heartbeats
        while it blocks -- a silent multi-minute response would otherwise trip
        proxy idle read timeouts (e.g. nginx ``grpc_read_timeout``, default 60s)
        and reset the stream, and the elapsed/size fields let a client show
        progress and decide whether to cancel. The single terminal message
        carries the source's now-concrete catalog row.

        The row, not a descriptor rebuilt from the adapter: resolution has
        already written it, and returning a second encoding of the same
        projection let the two disagree (the adapter answers ``is_resident()``
        live, the row is a snapshot).

        Resolving an already-resident source is a cheap no-op. If the client
        disconnects mid-resolve the daemon thread runs to completion and caches
        the result on the adapter, so a retry coalesces onto the finished work
        rather than downloading again.
        """
        adapter = self.sources.get(source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {source_id}")
        # The terminal message IS the catalog row, so refuse before the recall
        # rather than after minutes of download with nothing to hand back.
        catalog = self._require_catalog()

        # Name/size of what is being recalled, computed once (stat is recall-free).
        # Best-effort: an unresolved adapter exposes its URL; a directory or a
        # remote URL has no single file size, so target_bytes stays 0 (unknown).
        source_url = adapter.source_url or source_id
        target_name = os.path.basename(str(source_url).rstrip("/")) or str(source_url)
        target_bytes = 0
        try:
            if os.path.isfile(source_url):
                target_bytes = os.path.getsize(source_url)
        except OSError:
            pass

        started = time.monotonic()

        def _progress() -> bytes:
            return ResolveStreamMessage(
                progress=ResolveProgress(
                    elapsed_seconds=time.monotonic() - started,
                    target_name=target_name,
                    target_bytes=target_bytes,
                )
            ).SerializeToString()

        result: dict = {}

        def _run() -> None:
            try:
                adapter.resolve()
            except BaseException as exc:  # surfaced on the stream below
                result["err"] = exc

        worker = threading.Thread(target=_run, name=f"resolve-{source_id}", daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(timeout=_RESOLVE_HEARTBEAT_SECONDS)
            if worker.is_alive():
                yield _progress()  # heartbeat: warm + progress, carries no pixels

        if "err" in result:
            exc = result["err"]
            # Retriable subclass first (it IS a SourceUnresolvedError): a transient
            # recall/IO failure -> UNAVAILABLE so the client may retry the resolve.
            if isinstance(exc, SourceResolveRetriableError):
                raise flight.FlightUnavailableError(
                    f"Source resolve failed transiently (retry): {exc}"
                ) from exc
            # A bare SourceUnresolvedError here is a permanent resolution failure
            # (unsupported type / parse error) -> INTERNAL so the client does not
            # retry forever. (Contrast: an *unresolved-but-resolvable* source is
            # caught in get_flight_info and mapped to UNAVAILABLE "open to resolve".)
            if isinstance(exc, SourceUnresolvedError):
                raise flight.FlightInternalError(
                    f"Source could not be resolved: {exc}"
                ) from exc
            raise flight.FlightServerError(
                f"resolve failed for {source_id!r}: {exc}"
            ) from exc

        # The row read back is the one the adapter's ``on_resolved`` callback
        # just backfilled -- resolution fires it, and that is the only thing
        # that overwrites the NULL-shape placeholder registration wrote. An
        # unresolved source built without the callback has no way to correct its
        # row, which is why this reads the catalog rather than re-syncing here.
        row = catalog.source_row_ipc(source_id)
        if row is None:
            raise flight.FlightServerError(
                f"resolve succeeded for {source_id!r} but the catalog has no row for it"
            )
        yield ResolveStreamMessage(source_row=row).SerializeToString()

    def _handle_warm(
        self, source_id: str, context: flight.ServerCallContext
    ) -> Iterator[bytes]:
        """Stream the progress of *warming* (hydrate-ahead) a resolved source.

        After ``resolve`` populates a multi-file cloud source's metadata, its
        member data files are still dehydrated and recall one-at-a-time onto the
        lazy ``do_get`` read path (the canonical case is zarr/ome-zarr: resolve
        reads only ``.zattrs``/``.zarray``, so every chunk file recalls the first
        time the viewer scrubs to it). ``warm`` opts into pulling them all
        resident up front: it walks the source directory and reads every file to
        force the sync engine's recall -- entirely server-side, so no pixels
        cross the wire, only the ``WarmStreamMessage`` progress.

        Unlike ``resolve`` (one opaque blocking call wrapped on a daemon thread),
        warming is our own loop, so it runs inline in this generator: a bounded
        worker pool recalls files concurrently while the generator polls for
        cancellation and emits progress (throttled to
        ``_WARM_PROGRESS_MIN_INTERVAL``). Warming is a pure side-effect
        (residency), so a cancel genuinely stops -- there is no result to
        preserve.

        Properties:
        - **No-op for single-file sources** -- their one file was already
          recalled by resolve, so this emits one terminal ``done`` with
          ``files_total == 0`` and returns. A *remote* source raises instead;
          nothing here can be made resident.
        - **Read every file unconditionally** -- residency is volatile (eviction /
          re-dehydration can flip it underneath us), so a "skip already-resident"
          check would be a TOCTOU trap; an unconditional read is idempotent
          (already-warm files are cheap local reads, cold files recall).
        - **Counts as Flight activity** (wrapped in ``activity.serving_request``) so the
          background precache worker yields to it for the duration.
        - Does **not** hold the adapter's per-source IO lock -- these are plain
          filesystem reads, concurrency-safe with real reads, so warming never
          blocks a live viewer read.
        """
        adapter = self.sources.get(source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {source_id}")

        root = adapter.source_url
        # A remote source has no local tree to walk, so refuse rather than fall
        # into the no-op below: `files_total == 0` is how a client learns a
        # source is single-file, and must not also mean "not applicable"
        # (biopb/biopb#1035). Scheme only, so a mirror's aliased `source_url`
        # (display authority, never the dial address) is still sound to ask.
        if root and is_remote_url(root):
            scheme = root.split("://", 1)[0]
            raise flight.FlightServerError(
                f"Cannot warm {source_id!r}: it is a remote ({scheme}) source, "
                "and warm recalls member files onto the serving machine's own "
                "filesystem. Nothing here can be made resident. Warm it on the "
                "server that holds the data."
            )
        # Single-file / non-directory source: nothing to warm beyond what
        # resolve already recalled. One terminal `done`, files_total == 0.
        if not root or not os.path.isdir(root):
            yield WarmStreamMessage(done=WarmProgress()).SerializeToString()
            return

        # Reject a second concurrent warm of the same source (avoid doubling the
        # disk/recall pressure); the browser also disables re-trigger while running.
        if not self.activity.begin_warm(source_id):
            raise flight.FlightServerError(
                f"warm already in progress for {source_id!r}"
            )

        started = time.monotonic()
        last_yield = 0.0
        progress_lock = threading.Lock()
        cancel_event = threading.Event()
        progress_state = {
            "files_done": 0,
            "bytes_done": 0,
            "current_name": "",
        }
        worker_local = threading.local()

        def _progress(
            files_total: int,
            files_done: int,
            bytes_total: int,
            bytes_done: int,
            current_name: str,
        ) -> bytes:
            return WarmStreamMessage(
                progress=WarmProgress(
                    files_total=files_total,
                    files_done=files_done,
                    bytes_total=bytes_total,
                    bytes_done=bytes_done,
                    current_name=current_name,
                    elapsed_seconds=time.monotonic() - started,
                )
            ).SerializeToString()

        def _snapshot() -> Tuple[int, int, str]:
            with progress_lock:
                return (
                    progress_state["files_done"],
                    progress_state["bytes_done"],
                    progress_state["current_name"],
                )

        def _read_file(fpath: str) -> None:
            """Read one file, updating the shared warm progress snapshot."""
            if cancel_event.is_set():
                return

            name = os.path.basename(fpath)
            with progress_lock:
                # A cancellation can race with a worker being scheduled. Do not
                # open a new file once the main loop has observed cancellation.
                if cancel_event.is_set():
                    return
                progress_state["current_name"] = name

            try:
                # Each executor worker owns and reuses its buffer: sharing the
                # old single buffer would race readinto() calls and corrupt the
                # byte tally.
                buf = getattr(worker_local, "buf", None)
                if buf is None:
                    buf = bytearray(_WARM_READ_BLOCK_BYTES)
                    worker_local.buf = buf
                with open(fpath, "rb", buffering=0) as fh:
                    while not cancel_event.is_set():
                        n = fh.readinto(buf)
                        if not n:
                            break
                        # current_name was already set above; re-stamping it on
                        # every block would retake the lock for no new value and
                        # serialize the fast path -- already-resident files that
                        # read fast enough for 4 workers to contend on this lock.
                        with progress_lock:
                            progress_state["bytes_done"] += n
            except OSError as exc:
                logger.warning("warm: skipping %s: %s", fpath, exc)
            finally:
                with progress_lock:
                    progress_state["files_done"] += 1

        try:
            # Warming registers as in-flight activity so the precache worker parks.
            with self.activity.serving_request():
                # 1. Enumerate + stat (recursive, recall-free). os.walk separates
                #    directories from `names`, so `names` are the data files we
                #    want; stat does not recall a placeholder.
                entries: List[Tuple[int, str]] = []
                bytes_total = 0
                for dirpath, _dirs, names in os.walk(root):
                    if context.is_cancelled():
                        yield WarmStreamMessage(
                            done=WarmProgress(
                                elapsed_seconds=time.monotonic() - started
                            )
                        ).SerializeToString()
                        return
                    for name in names:
                        fpath = os.path.join(dirpath, name)
                        try:
                            size = os.stat(fpath).st_size
                        except OSError:
                            continue  # vanished/unreadable between walk and stat
                        entries.append((size, fpath))
                        bytes_total += size
                    now = time.monotonic()
                    if now - last_yield >= _RESOLVE_HEARTBEAT_SECONDS:
                        last_yield = now
                        yield _progress(0, 0, 0, 0, "")  # still enumerating

                # 2. Ascending size: for pyramidal data this approximates
                #    coarsest-level-first (coarse levels are the small files) so the
                #    viewer becomes responsive earliest; otherwise a harmless tie.
                entries.sort(key=lambda e: e[0])
                files_total = len(entries)
                files_done = 0
                bytes_done = 0

                # Immediately surface the total before the first (possibly long) read.
                last_yield = time.monotonic()
                yield _progress(files_total, files_done, bytes_total, bytes_done, "")

                # 3. Recall loop: read every file to completion (forces residency).
                # Keep only one batch of work per worker in flight. Besides
                # bounding threads, fds, and per-worker buffers, this preserves
                # the existing smallest-files-first launch order without queuing
                # tens of thousands of futures in the executor.
                pending = set()
                next_entry = 0

                def _check_cancel() -> None:
                    if context.is_cancelled():
                        cancel_event.set()

                with ThreadPoolExecutor(max_workers=_WARM_MAX_WORKERS) as pool:

                    def _refill() -> None:
                        nonlocal next_entry
                        while (
                            not cancel_event.is_set()
                            and len(pending) < _WARM_MAX_WORKERS
                            and next_entry < files_total
                        ):
                            fpath = entries[next_entry][1]
                            next_entry += 1
                            pending.add(pool.submit(_read_file, fpath))

                    try:
                        _check_cancel()
                        _refill()

                        while pending:
                            _check_cancel()

                            completed, pending = wait(
                                pending,
                                timeout=_WARM_POLL_SECONDS,
                                return_when=FIRST_COMPLETED,
                            )
                            for future in completed:
                                future.result()

                            _check_cancel()
                            _refill()

                            now = time.monotonic()
                            if now - last_yield >= _WARM_PROGRESS_MIN_INTERVAL:
                                last_yield = now
                                done_files, done_bytes, current_name = _snapshot()
                                yield _progress(
                                    files_total,
                                    done_files,
                                    bytes_total,
                                    done_bytes,
                                    current_name,
                                )

                        # The executor has no queued work here. Shutdown still
                        # joins any read that was in progress before cancellation.
                        done_files, done_bytes, current_name = _snapshot()
                        files_done = done_files
                        bytes_done = done_bytes
                    finally:
                        # A client may close the generator at a progress yield.
                        # Set the flag before the `with` block joins the
                        # executor's in-flight workers so they stop before
                        # opening another file.
                        cancel_event.set()

                # 4. Terminal done (partial counts if cancelled mid-loop).
                yield WarmStreamMessage(
                    done=WarmProgress(
                        files_total=files_total,
                        files_done=files_done,
                        bytes_total=bytes_total,
                        bytes_done=bytes_done,
                        elapsed_seconds=time.monotonic() - started,
                    )
                ).SerializeToString()
        finally:
            self.activity.end_warm(source_id)

    def _handle_add_source(
        self, req: AddSourceRequest, context: flight.ServerCallContext
    ) -> Iterator[bytes]:
        """Stream registration of a runtime-added local path (drag-drop).

        Wraps the SourceManager's ``add_local_source`` generator: each event
        tuple it yields is mapped onto an ``AddSourceStreamMessage`` (zero or
        more ``progress`` heartbeats, then one terminal ``result``). A dropped
        directory can register several sources, so this streams -- a client shows
        rows appearing and can cancel (the stream closing sets is_cancelled(),
        which stops discovery while keeping what is already registered).

        Whole-request failures (path not found / unreadable, or a remote URL)
        raise from the generator on first iteration and are mapped to a
        FlightServerError so the client surfaces a clean message.
        """
        if not self._allow_runtime_source_add or self._add_source_handler is None:
            raise flight.FlightServerError(
                "Runtime source registration is not enabled on this server."
            )

        def _should_cancel() -> bool:
            return context.is_cancelled()

        try:
            events = self._add_source_handler(
                req.url,
                source_type=req.source_type,
                should_cancel=_should_cancel,
            )
            for event in events:
                kind = event[0]
                if kind == "progress":
                    _, added_count, current_path = event
                    progress = AddSourceProgress(
                        added_count=added_count,
                        current_path=current_path or "",
                    )
                    yield AddSourceStreamMessage(progress=progress).SerializeToString()
                else:  # "result"
                    _, tally = event
                    result = AddSourceResult(
                        added=tally.added,
                        already_present=tally.already_present,
                        refreshed=tally.refreshed,
                        removed=tally.removed,
                    )
                    for path, reason in tally.failed:
                        result.failed.add(path=path, reason=reason)
                    yield AddSourceStreamMessage(result=result).SerializeToString()
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            raise flight.FlightServerError(str(exc)) from exc

    def _handle_remove_source(self, req: RemoveSourceRequest) -> bytes:
        """Deregister a drag-dropped source branch (single, non-streamed result).

        Delegates to the SourceManager's ``remove_dropped_root``, which removes
        only sources whose catalog ``source_url`` carries the ``dnd://`` origin
        scheme -- so a request for anything else is rejected (``ValueError`` ->
        ``FlightServerError``). Removal is quick (unregister N adapters), so
        unlike add it does not stream: one ``RemoveSourceResult`` is returned.
        """
        if not self._allow_runtime_source_add or self._remove_source_handler is None:
            raise flight.FlightServerError(
                "Runtime source removal is not enabled on this server."
            )
        try:
            removed, failed = self._remove_source_handler(req.root_url)
        except ValueError as exc:
            raise flight.FlightServerError(str(exc)) from exc
        result = RemoveSourceResult(removed=removed)
        for source_id, reason in failed:
            result.failed.add(path=source_id, reason=reason)
        return result.SerializeToString()

    def list_flights(
        self, context: flight.ServerCallContext, criteria: bytes
    ) -> Iterator[flight.FlightInfo]:
        """Advertise the public catalog: one flight per queryable table.

        Each carries the table's real Arrow schema and one endpoint whose
        ticket is ``SELECT * FROM <table>``, so a stock Flight client can list,
        see the columns, and DoGet the whole catalog without a biopb proto.
        Browsing *sources* is a catalog query too (the SDK's ``list_sources``);
        the pixels and annotations of one source are not listable, they are
        addressed.
        """
        self._authorize(context)
        for table in sorted(self._require_catalog().allowed_tables):
            yield self._catalog_flight_info(table)

    def get_flight_info(
        self, context: flight.ServerCallContext, descriptor: flight.FlightDescriptor
    ) -> flight.FlightInfo:
        """Plan a read.

        A path descriptor names a catalog table (public tier): its schema and
        a ticket that reads it. A command is a ``FlightRequest`` whose
        ``tensor_read`` binds one tensor and plans its chunk endpoints (private,
        authorized as a pixel read of the tensor's source). An arbitrary catalog query
        needs no GetFlightInfo: the SQL rides the DoGet ticket.
        """
        import json

        if descriptor.descriptor_type == flight.DescriptorType.PATH:
            self._authorize(context)
            table = "/".join(
                part.decode() if isinstance(part, bytes) else part
                for part in descriptor.path
            )
            return self._catalog_flight_info(table)

        req = self._parse(FlightRequest(), descriptor.command, "GetFlightInfo command")
        read_opt = req.tensor_read
        source_id, tensor_id = split_array_id(read_opt.array_id)
        if not source_id:
            raise flight.FlightServerError("tensor_read: array_id is required")

        self._authorize_read(context, source_id, READ_PIXELS)
        mask = read_mask(read_opt)

        # Reduce the request array_id to the within-source field -- or None =
        # "the source's default (first) tensor" (identity policy: array_id is
        # source_id or source_id/field). The wire descriptor still reports the
        # full array_id, carried by get_tensor_descriptor().
        field = self._field_within_source(source_id, tensor_id)

        # Substitute the source's default (first) tensor for every no-field
        # request (#44). get_flight_info / get_source / get_physical_scale are
        # documented to accept a bare source_id; forwarding None to a
        # multi-tensor adapter's get_tensor_adapter would otherwise select a
        # bogus field (a bioio scene lookup on None, OME-Zarr HCS field parsing
        # crashing on None.split), so honor the documented default in this one
        # chokepoint rather than at every adapter call site.
        if field is None:
            default_adapter = self.sources.get(source_id)
            if default_adapter is not None:
                descriptors = default_adapter.list_tensor_descriptors()
                if descriptors:
                    field = self._field_within_source(
                        source_id, descriptors[0].array_id
                    )

        logger.debug(
            f"get_flight_info: source_id={source_id}, tensor_id={tensor_id}, field={field}"
        )

        # Get tensor adapter for the specified source and tensor. Field resolution
        # is total: an unknown field raises a typed TensorResolutionError, which
        # the boundary handler maps to a terminal Flight error carrying the
        # canonical code in extra_info (NOT a "server bug" FlightInternalError).
        # A bare ValueError/KeyError from an adapter that predates the taxonomy is
        # still a field-resolution miss, not a server bug, so it is coerced the
        # same way rather than leaking as INTERNAL (issue #378).
        try:
            tensor_adapter = self._get_adapter_for_tensor(source_id, field)
        except (
            SourceUnresolvedError,
            TensorResolutionError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
        ) as e:
            # One handler for every resolution outcome (see _adapter_lookup_error):
            # an unresolved source -> retriable "open to resolve"; a typed field
            # miss -> terminal NOT_FOUND / INVALID_ARGUMENT with the canonical code
            # in extra_info; any other bare exception from a legacy adapter ->
            # terminal UNKNOWN, never leaked as INTERNAL (issue #378).
            if not isinstance(e, SourceUnresolvedError):
                logger.warning(f"Tensor not found: {source_id}/{field} ({e})")
            raise _adapter_lookup_error(
                e, f"Tensor not found: {source_id}/{field}"
            ) from e
        if tensor_adapter is None:
            logger.warning(f"Source not found: {source_id}")
            raise to_flight_error(
                TensorNotFound(
                    f"Source not found: {source_id}",
                    reason="unknown_source",
                )
            )

        # Build the read plan and advertise the pyramid + physical scale. Each
        # adapter owns this seam (plan_flight_info): a remote proxy forwards the
        # upstream's authoritative GetFlightInfo (native grid + server-advertised
        # pyramid + physical scale, localized), while every other adapter plans
        # locally against its native grid and the server's pyramid config.
        # metadata_json is still filled below from the local mirror catalog (the
        # #253 no-extra-RPC path), not by the adapter.
        try:
            read_plan = tensor_adapter.plan_flight_info(read_opt, self._pyramid_config)

            schema = tensor_adapter.get_arrow_schema(read_plan.descriptor)

            source_adapter = self.sources.get(source_id)

            # A serving field, filled from the bound adapter (biopb/biopb#780).
            # Deliberately here rather than on the catalog listing: consumers use
            # it to decide whether a cache entry is still valid, and this call is
            # fetch-per-call by contract while a listing is a natural thing to
            # cache. None stays unset -- absent is "no claim", not "unchanged".
            #
            # A claim about the data, so the raw content_version and never the
            # serving epoch: a consumer builds an identity from this (the HTTP
            # sidecar's versioned array_id) and stamps its own records with it
            # (an ROI's ``drawn_against_version``), neither of which may move on
            # a server upgrade that changed no data.
            #
            # From the TENSOR adapter: an uploaded label set's bytes are its own
            # (``adapters/labels.py``), not the source's.
            if tensor_adapter.content_version:
                read_plan.descriptor.content_version = tensor_adapter.content_version

            # Populate metadata_json in response descriptor if requested
            if METADATA_JSON in mask:
                # One scheme (biopb/biopb#253): the source-level metadata is
                # computed once at registration and read back from the catalog --
                # the cache -- never recomputed on the adapter. A DB read error
                # propagates (no fallback); a NULL row is a legitimate "no
                # metadata" (empty base). A catalog-less server has no cache to
                # read, and nothing released the adapter's registration copy
                # either (``sync_source_added`` is what does that), so there it
                # is the adapter that answers.
                raw_metadata = (
                    self._metadata_db.get_metadata_json(source_id)
                    if self._metadata_db is not None
                    else source_adapter.get_metadata()
                ) or {}
                # Overlay the tensor adapter's cheap per-tensor delta -- fields the
                # source-level row cannot carry (an OME-Zarr HCS field's own OME
                # metadata; an EMD signal's original_metadata). Merged over the
                # cached row so per-tensor metadata needs no catalog row of its own.
                tensor_extra = tensor_adapter.get_tensor_metadata()
                if tensor_extra:
                    raw_metadata = {**raw_metadata, **tensor_extra}
                raw_metadata = _with_label_axes(
                    raw_metadata, source_adapter, read_plan.descriptor
                )
                if raw_metadata and source_adapter is not None:
                    wrapped_metadata = {
                        "type": source_adapter.source_type,
                        "dim_label": list(read_plan.descriptor.dim_labels),
                        "metadata": raw_metadata,
                    }
                    read_plan.descriptor.metadata_json = json.dumps(
                        wrapped_metadata, cls=NumpyEncoder
                    )
        except (SourceUnresolvedError, TensorResolutionError) as e:
            # The typed taxonomy, mapped precisely: a not-yet-hydrated source to
            # the retriable "open to resolve", and a caller's malformed slice or
            # scale hint to a terminal INVALID_ARGUMENT. Both would otherwise be
            # buried in "Metadata error" as a bare ValueError -> INTERNAL, which
            # blames the server and tells a client nothing it can act on. Must
            # precede the ValueError clause (both subclass ValueError).
            raise to_flight_error(e) from e
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise flight.FlightInternalError(
                f"Metadata error for {source_id}: {e}"
            ) from e

        # Upload progress, for a source that is one. Here rather than in an
        # action because `do_action` takes full access, and the caller waiting
        # on a fast-return result holds a per-source read capability and
        # nothing else (biopb/biopb#1048). An empty mask makes the poll cheap:
        # no endpoint enumeration, so this is the only work it does.
        #
        # A discarded source answers too. The adapter stays registered as a
        # tombstone and describe is not a chunk read, so it never reaches
        # `_refuse_if_discarded` -- a poller learns the reason instead of
        # meeting a dead call.
        if UPLOAD_STATUS in mask:
            # The tensor first: a label set is a tensor of a source that is not
            # itself an upload, and it is the set's own progress its producer
            # polls (biopb/biopb#1059). The two source kinds answer from the
            # source, where the array_id and the source_id are the same string.
            upload = upload_of(tensor_adapter) or upload_of(source_adapter)
            if upload is not None:
                _fill_upload_status(
                    read_plan.descriptor, upload, read_plan.descriptor.array_id
                )

        # Residency, only when asked. It is a bounded stat walk of the source,
        # so it is never free -- which is why it is per-source and opt-in rather
        # than the catalog-wide action it was: that made a live filesystem walk
        # the cost of listing, for every source, on every browse
        # (biopb/biopb#1035, biopb/biopb#1048).
        #
        # A source that cannot answer leaves the field unset. Unset reads as
        # "unknown", never as False -- which would send a client to hydrate what
        # is already on disk.
        if IS_RESIDENT in mask:
            try:
                read_plan.descriptor.is_resident = bool(source_adapter.is_resident())
            except Exception:  # noqa: BLE001 -- a balky adapter is not the request
                logger.debug("is_resident failed for %s", source_id, exc_info=True)

        # Convert to FlightEndpoints. Each endpoint carries the server-minted
        # chunk_id as an opaque ticket and the chunk's bounds as app_metadata;
        # the client echoes the ticket back to do_get and never decodes the
        # chunk_id byte format (a strictly server-side concern).
        endpoints = []
        for ce in read_plan.chunk_endpoints:
            ticket = TensorTicket(chunk_id=ce.chunk_id)
            endpoint = flight.FlightEndpoint(
                ticket=flight.Ticket(ticket.SerializeToString()),
                locations=[],
                app_metadata=self._encode_metadata(ce.bounds),
            )
            endpoints.append(endpoint)

        logger.debug(f"get_flight_info: returning {len(endpoints)} chunk endpoints")
        # The requested slice, verbatim, so the plan says what it was asked for
        # as well as what it realized: the descriptor's slice_hint is snapped
        # outward to chunk-aligned bounds, and a consumer -- this connection or
        # one handed the FlightInfo as a SerializedTensor -- crops back to the
        # request from here rather than remembering it separately.
        return flight.FlightInfo(
            schema=schema,
            descriptor=flight.FlightDescriptor.for_command(
                read_plan.descriptor.SerializeToString()
            ),
            endpoints=endpoints,
            total_records=-1,
            total_bytes=-1,
            app_metadata=(
                read_opt.slice_hint.SerializeToString()
                if read_opt.HasField("slice_hint")
                else b""
            ),
        )

    def do_get(
        self, context: flight.ServerCallContext, ticket: flight.Ticket
    ) -> flight.FlightDataStream:
        """Serve what the ticket's oneof arm names: a catalog query result,
        one tensor's annotation set, or one pixel chunk."""
        tensor_ticket = self._parse_ticket(ticket)
        arm = tensor_ticket.WhichOneof("payload")

        if arm == "catalog_query":
            self._authorize(context)
            try:
                table = self._require_catalog().query(tensor_ticket.catalog_query.sql)
            except ValueError as e:
                raise flight.FlightServerError(f"Catalog query failed: {e}") from e
            return flight.RecordBatchStream(table)

        if arm == "roi_read":
            return self._roi_read_stream(context, tensor_ticket.roi_read)

        # Heavy chunk-read path: track it as in-flight so the background
        # precache worker stays idle while real reads are happening.
        with self.activity.serving_request():
            logger.debug(f"do_get: chunk_id={tensor_ticket.chunk_id[:16]}...")

            source_id = routing_array_id(tensor_ticket.chunk_id).split("/")[0]
            self._authorize_read(context, source_id, READ_PIXELS)

            adapter = self._get_adapter_for_chunk(tensor_ticket.chunk_id)

            # Get cache manager singleton (if initialized)
            cache_manager = CacheManager.get_instance()

            # Read the chunk, using the configured cache backend when applicable.
            try:
                record_batch = adapter.resolve_chunk_data(
                    tensor_ticket.chunk_id, cache_manager
                )
            except TensorResolutionError as e:
                # A stale chunk_id (biopb/biopb#178) is the client's held ticket
                # outliving a re-registration, not a server bug -- surface it as
                # the typed terminal taxonomy (to_flight_error) rather than
                # burying it in "I/O error" as a bare ValueError -> INTERNAL.
                # Must precede the ValueError clause (it subclasses ValueError).
                raise to_flight_error(e) from e
            except (OSError, ValueError) as e:
                # ValueError can be raised by bounds validation or parsing failures
                raise flight.FlightInternalError(
                    f"I/O error reading chunk data: {e}"
                ) from e

            batch_size = sum(col.nbytes for col in record_batch.columns)
            logger.debug(f"do_get: returning {batch_size} bytes")

            # zero-copy wrapper - do _not_ convert to pa.Table!
            reader = pa.RecordBatchReader.from_batches(
                record_batch.schema, [record_batch]
            )
            return flight.RecordBatchStream(reader)

    def _roi_read_stream(
        self, context: flight.ServerCallContext, req
    ) -> flight.FlightDataStream:
        """One tensor's annotations as ROI rows; ``truncated`` and the tensor's
        ``sets`` (JSON) ride the stream's schema metadata."""
        db = self._require_annotations()
        try:
            self._authorize_read(
                context, _roi_source_id(req.array_id), READ_ANNOTATIONS
            )
            rois, truncated = db.list_rois(req.array_id, req.set_name)
            sets = [
                {"set_name": name, "count": count, "reserved": is_reserved_set(name)}
                for name, count in db.list_roi_sets(req.array_id)
            ]
        except ValueError as e:
            raise flight.FlightServerError(str(e))
        table = rois_to_table(
            rois,
            {
                b"truncated": str(truncated).encode(),
                b"sets": json.dumps(sets).encode(),
            },
        )
        return flight.RecordBatchStream(table)

    def _handle_chunk_locate(self, chunk_id: bytes) -> str:
        """Locate a cached chunk on disk for the localhost cache-file handoff.

        Locates the chunk's Arrow IPC message in the file cache and returns its
        on-disk byte range as JSON. If the chunk isn't cached yet, materializes
        it first (same path as do_get) and retries the locate -- so a warm chunk
        is answered without re-reading or re-decoding it. Returns
        ``{"available": false}`` when the chunk can't be located (memory backend,
        oversized/uncached chunk, or any resolve/locate failure) so the client
        falls back to do_get. (issue #9)

        **Counts as Flight activity** (``activity.serving_request``) so the
        background precache worker parks while a localhost client is reading.
        This path *replaces* ``do_get`` rather than accompanying it, so without
        the wrapper the server observes nothing for the whole of such a read
        (biopb/biopb#548). It is also where a cold miss decodes the chunk. A warm
        locate is cheap and gets counted anyway -- over-reporting a read is the
        safe direction for a debounce, and the client's mmap read that follows is
        real I/O the server cannot see at all.
        """
        cache_manager = CacheManager.get_instance()
        if cache_manager is None:
            return json.dumps({"available": False})

        with self.activity.serving_request():
            adapter = self._get_adapter_for_chunk(chunk_id)

            # Entries are stored under the method-stripped canonical key
            # (biopb/biopb#76); locate with the same key or a warm chunk cached
            # under a different reduction_method is never found.
            cache_key = cache_key_for_chunk_id(chunk_id)
            try:
                # Reject a stale chunk_id before consulting the cache: a cache
                # HIT below returns its mmap location directly and never calls
                # resolve_chunk_data, so without this a chunk_id from before a
                # re-registration would silently return whatever old-version
                # bytes are still resident instead of the StaleChunkError a
                # do_get on the same id would raise (biopb/biopb#178). Pure
                # in-memory comparison -- no adapter I/O -- so it costs nothing
                # to run on every locate, hit or miss.
                adapter.check_chunk_version(chunk_id)

                # And the same for the read gate, for the same reason: a warm
                # chunk of an upload nobody has published yet -- or of one that
                # has been discarded -- is still sitting in the cache, and this
                # path would hand out its byte range without the adapter ever
                # being asked (biopb/biopb#1048).
                adapter.check_readable()

                # If the chunk is already cached, just locate it. Resolving first
                # would, on a chunk whose in-RAM entry has been trimmed, re-read the
                # whole chunk from its segment server-side for nothing. Only
                # materialize (same path as do_get) on a genuine cold miss.
                location = cache_manager.locate_entry(cache_key)
                if location is None:
                    # Resolving caches the chunk synchronously, so by the time
                    # this returns the bytes are on disk and the second locate
                    # can answer with their range.
                    adapter.resolve_chunk_data(chunk_id, cache_manager)
                    location = cache_manager.locate_entry(cache_key)
            except TensorResolutionError as e:
                # Same stale-chunk_id mapping as do_get (biopb/biopb#178); must
                # precede the ValueError clause (it subclasses ValueError).
                raise to_flight_error(e) from e
            except (OSError, ValueError) as e:
                raise flight.FlightInternalError(
                    f"I/O error locating chunk data: {e}"
                ) from e

            if location is None:
                return json.dumps({"available": False})

            return json.dumps(
                {
                    "available": True,
                    "segment_path": location.segment_path,
                    "byte_offset": location.byte_offset,
                    "byte_length": location.byte_length,
                    "generation_id": location.generation_id,
                }
            )

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        """Take what the command's oneof arm names: a pixel chunk, or an
        annotation put / delete. Each arm gates itself: pixels need a writable
        server, annotations need the store; both authorize on the source."""
        cmd = self._parse(PutCommand(), descriptor.command, "DoPut command")
        arm = cmd.WhichOneof("command")

        if arm == "chunk":
            if not self._writable:
                raise flight.FlightUnauthenticatedError("Server not in write mode")
            self._authorize(context)
            self.uploads.write_chunk(cmd.chunk, reader)
            return

        db = self._require_annotations()
        try:
            if arm == "roi_put":
                self._authorize(context)
                rois = table_to_rois(reader.read_all())
                stored, conflicts = db.put_rois(
                    cmd.roi_put.array_id, rois, check_rev=cmd.roi_put.check_rev
                )
                reply = RoiPutResult(stored=stored, conflicts=conflicts)
            else:
                self._authorize(context)
                roi_ids = table_to_roi_ids(reader.read_all())
                deleted = db.delete_rois(
                    cmd.roi_delete.array_id, roi_ids, cmd.roi_delete.set_name
                )
                reply = RoiDeleteResult(deleted=deleted)
        except ValueError as e:
            # Rejected geometry, a mismatched array_id, a breached cap, a stream
            # not in the row schema: the caller's problem, so say which.
            raise flight.FlightServerError(str(e))
        writer.write(reply.SerializeToString())


def serve(
    adapters: Dict[str, SourceAdapter], location: str = "grpc://0.0.0.0:8815", **kwargs
) -> None:
    """Start a Flight server with the given adapters.

    Args:
        adapters: Dictionary mapping source_id to SourceAdapter
        location: Server location
        **kwargs: Additional arguments passed to FlightServerBase
    """
    server = TensorFlightServer(location, **kwargs)
    for source_id, adapter in adapters.items():
        registered = server.register_source(source_id, adapter)
        # Registration is the registry; the catalog row is what makes a source
        # browsable, and only a caller that passed a ``metadata_db`` has one.
        if server.metadata_db is not None:
            server.metadata_db.sync_source_added(source_id, registered)
    # All sources registered up front -> ready immediately (health: SERVING).
    server.mark_ready()

    print(f"Starting Flight server at {location}")
    server.serve()
