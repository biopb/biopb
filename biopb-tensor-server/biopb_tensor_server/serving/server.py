"""Arrow Flight server for tensor storage.

This module implements a Flight server that exposes chunked multi-dimensional
arrays through the BackendAdapter interface.

The server supports:
- ListFlights: Browse available tensors
- GetFlightInfo: Get tensor metadata and chunk endpoints
- DoGet: Fetch individual chunk data
- DoPut: Upload data (when writable mode enabled)
- Metadata queries: SQL queries against source catalog (via DuckDB)
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import (
    AddSourceProgress,
    AddSourceRequest,
    AddSourceResult,
    AddSourceStreamMessage,
    FlightCmd,
    RemoveSourceRequest,
    RemoveSourceResult,
    ResolveProgress,
    ResolveStreamMessage,
    TensorDescriptor,
    WarmProgress,
    WarmStreamMessage,
)
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload, TensorTicket

from biopb_tensor_server.cache import CACHE_FILE_FORMAT_VERSION, CacheManager
from biopb_tensor_server.core.activity import ActivityTracker
from biopb_tensor_server.core.base import (
    SourceAdapter,
    TensorAdapter,
    decode_chunk_id,
    strip_source_prefix,
)
from biopb_tensor_server.core.chunk import cache_key_for_chunk_id
from biopb_tensor_server.core.config import PyramidConfig
from biopb_tensor_server.core.errors import (
    SourceResolveRetriableError,
    SourceUnresolvedError,
    TensorNotFound,
    TensorResolutionError,
    UnknownResolutionError,
)
from biopb_tensor_server.core.metadata_db import MetadataDatabase, NumpyEncoder
from biopb_tensor_server.core.source_registry import SourceRegistry
from biopb_tensor_server.serving.upload_manager import UploadManager

logger = logging.getLogger(__name__)


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


class _AuthMiddleware(flight.ServerMiddleware):
    """Per-call middleware that carries the caller's presented Bearer token.

    Handlers retrieve it via ``context.get_middleware("auth")`` to enforce
    per-source capability tokens (see ``_authorize_source``).
    """

    def __init__(self, token: Optional[str]) -> None:
        self.token = token

    def sending_headers(self) -> dict:
        return {}

    def call_completed(self, exception: Optional[Exception]) -> None:
        pass


class BearerAuthMiddlewareFactory(flight.ServerMiddlewareFactory):
    """Validate the server-wide Bearer token and expose the presented token.

    Header value must be exactly ``Bearer <token>`` (case-sensitive).
    When *token* is ``None`` or empty the server-wide check is disabled (the
    factory becomes a no-op for it), but the presented token is still captured
    so per-source capability tokens can be enforced in the handlers.
    """

    def __init__(self, token: Optional[str]) -> None:
        self._expected = f"Bearer {token}" if token else None

    def start_call(
        self,
        info: flight.CallInfo,
        headers: dict,
    ) -> Optional[flight.ServerMiddleware]:
        # Header values are lists; gRPC lowercases header names.
        values: List[str] = headers.get("authorization", [])
        bearer = values[0] if values else ""
        if self._expected is not None and bearer != self._expected:
            raise flight.FlightUnauthenticatedError("Invalid or missing Bearer token")
        provided = bearer[len("Bearer ") :] if bearer.startswith("Bearer ") else None
        return _AuthMiddleware(provided)


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

        # Start the server
        server = TensorFlightServer('grpc://0.0.0.0:8815')
        server.register_source('my-tensor', adapter)
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
        max_list_flights_results: int = 100000,
        grpc_max_message_size: Optional[int] = None,
        pyramid_config: Optional[PyramidConfig] = None,
        **kwargs,
    ):
        """Initialize the Flight server.

        Args:
            location: Server location (e.g., 'grpc://0.0.0.0:8815')
            token: Bearer token required on every call.  ``None`` disables auth.
            writable: Enable write mode for source creation and data upload
            write_dir: Directory for zarr-backed uploaded sources (required if writable)
            metadata_db: MetadataDatabase instance for source filtering queries (optional)
            max_list_flights_results: Safety cap on list_flights() returned sources
            grpc_max_message_size: gRPC max message size in bytes (default: 16MB)
            **kwargs: Additional arguments passed to FlightServerBase
        """
        # Apply gRPC max message size via URL query parameter
        if grpc_max_message_size:
            separator = "&" if "?" in location else "?"
            location = f"{location}{separator}grpc.max_send_message_size={grpc_max_message_size}&grpc.max_receive_message_size={grpc_max_message_size}"

        middleware = kwargs.pop("middleware", {})
        middleware.setdefault("auth", BearerAuthMiddlewareFactory(token))
        super().__init__(location, middleware=middleware, **kwargs)
        self.sources = SourceRegistry()
        self._writable = writable
        self._metadata_db: Optional[MetadataDatabase] = metadata_db
        self._max_list_flights_results = max_list_flights_results
        # Authoritative resolution-pyramid knobs. Used to advertise
        # TensorDescriptor.pyramid in get_flight_info (computed levels) and shared
        # with the precache worker so the warmed scales can't drift from the
        # advertised ones.
        self._pyramid_config = pyramid_config or PyramidConfig()
        self._start_time: float = time.time()
        # DoPut upload path: source creation, chunk writes, and per-source upload
        # progress. Registers created sources through the shared registry.
        self.uploads = UploadManager(self.sources, write_dir, metadata_db)
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

    def register_source(self, source_id: str, adapter: SourceAdapter) -> None:
        """Register a data source with the server (delegates to ``sources``)."""
        self.sources.register(source_id, adapter)

    def unregister_source(self, source_id: str) -> None:
        """Unregister a data source and drop any in-flight upload state."""
        self.sources.unregister(source_id)
        self.uploads.forget(source_id)

    def shutdown(self) -> None:
        """Release source-adapter resources, then shut down the Flight server.

        Some adapters hold long-lived OS handles (e.g. the OME-TIFF adapter's
        persistent aszarr store). Closing them on shutdown releases those
        handles -- required on Windows, where an open file cannot be deleted
        (otherwise a test's TemporaryDirectory cleanup raises WinError 32).
        """
        self.sources.close_all()
        super().shutdown()

    def _authorize_source(
        self, context: flight.ServerCallContext, source_id: str
    ) -> None:
        """Enforce a per-source capability token when the source carries one.

        Sources created with a ``token`` attribute (e.g. embedded result caches)
        are readable only by callers presenting the matching Bearer token. This
        binds the result to the requester that received its ``SerializedTensor``,
        without relying on the server-wide token.

        Sources without a token fall back to the server-wide auth performed by
        ``BearerAuthMiddlewareFactory``, so this is backward compatible with the
        standalone tensor server.
        """
        adapter = self.sources.get(source_id)
        expected = adapter.capability_token if adapter is not None else None
        if not expected:
            return
        mw = context.get_middleware("auth")
        provided = getattr(mw, "token", None) if mw is not None else None
        if provided != expected:
            raise flight.FlightUnauthenticatedError("Invalid or missing source token")

    def _parse_ticket(self, ticket: flight.Ticket) -> TensorTicket:
        """Parse a TensorTicket from a Flight Ticket.

        Args:
            ticket: Flight ticket with ticket bytes

        Returns:
            Parsed TensorTicket
        """
        return TensorTicket.FromString(ticket.ticket)

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

        return source_adapter.get_tensor_adapter(tensor_id)

    def _get_adapter_for_chunk(self, chunk_id: bytes) -> Optional[TensorAdapter]:
        """Get adapter for a specific chunk based on its chunk_id.

        Args:
            chunk_id: The chunk identifier bytes

        Returns:
            TensorAdapter responsible for the chunk, or None if not found
        """
        array_id, *_ = decode_chunk_id(chunk_id)
        source_id, *rest = array_id.split("/")
        rest = "/".join(rest) if rest else None

        source_adapter = self.sources.get(source_id)
        if source_adapter is None:
            return None

        # Check for level adapter (OME-Zarr) only when there's an explicit level path
        if rest is not None and hasattr(source_adapter, "get_level_adapter"):
            return source_adapter.get_level_adapter(rest)

        # Otherwise get tensor adapter (for virtual scaling or single-tensor sources)
        return source_adapter.get_tensor_adapter(rest)

    def list_actions(
        self,
        context: flight.ServerCallContext,
    ) -> List[flight.ActionType]:
        """List available actions on this server.

        Returns:
            List of ActionType objects describing available actions.
        """
        return [
            flight.ActionType("health", "Health check - returns server status JSON"),
            flight.ActionType(
                "create_source",
                "Create a writable source from a TensorDescriptor request",
            ),
            flight.ActionType("upload_status", "Upload status for a writable source"),
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
        ]

    def do_action(
        self,
        context: flight.ServerCallContext,
        action: flight.Action,
    ) -> Iterator[bytes]:
        """Execute a custom action.

        Args:
            context: Server call context
            action: Action to execute

        Yields:
            Result bytes (JSON-encoded for health action)
        """
        if action.type == "health":
            uptime_seconds = int(time.time() - self._start_time)
            with self._scan_status_lock:
                full_scan_in_progress = self._full_scan_in_progress
                last_full_scan_at = self._last_full_scan_at
            health_status = {
                "status": "SERVING" if self._ready.is_set() else "STARTING",
                "source_count": len(self.sources),
                "metadata_db_enabled": self._metadata_db is not None,
                "writable": self._writable,
                "uptime_seconds": uptime_seconds,
                # Catalog-freshness signals (progressive discovery). ``SERVING``
                # no longer implies a complete catalog; these say whether a full
                # scan is running and when one last finished (epoch seconds, or
                # null until the first full scan succeeds). See biopb/biopb#212.
                "full_scan_in_progress": full_scan_in_progress,
                "last_full_scan_finished_at": last_full_scan_at,
            }
            yield json.dumps(health_status).encode("utf-8")
        elif action.type == "create_source":
            if not self._writable:
                raise flight.FlightUnauthenticatedError("Server not in write mode")

            req_desc = TensorDescriptor.FromString(action.body.to_pybytes())
            response_desc = self.uploads.create_source(req_desc)
            yield response_desc.SerializeToString()
        elif action.type == "upload_status":
            source_id = action.body.to_pybytes().decode("utf-8")
            yield json.dumps(self.uploads.status(source_id)).encode("utf-8")
        elif action.type == "chunk_locate":
            ticket_bytes = action.body.to_pybytes()
            ticket = self._parse_ticket(flight.Ticket(ticket_bytes))
            source_id = decode_chunk_id(ticket.chunk_id)[0].split("/")[0]
            self._authorize_source(context, source_id)
            yield self._handle_chunk_locate(ticket.chunk_id).encode("utf-8")
        elif action.type == "cache_stats":
            from dataclasses import asdict

            manager = CacheManager.get_instance()
            if manager is None:
                raise flight.FlightServerError("Cache not initialized")
            # asdict recurses into the per-pool PoolStats dataclasses under pool_stats.
            yield json.dumps(asdict(manager.stats())).encode("utf-8")
        elif action.type == "resolve":
            source_id = action.body.to_pybytes().decode("utf-8")
            self._authorize_source(context, source_id)
            yield from self._handle_resolve(source_id)
        elif action.type == "warm":
            source_id = action.body.to_pybytes().decode("utf-8")
            self._authorize_source(context, source_id)
            yield from self._handle_warm(source_id, context)
        elif action.type == "add_source":
            req = AddSourceRequest.FromString(action.body.to_pybytes())
            yield from self._handle_add_source(req, context)
        elif action.type == "remove_source":
            req = RemoveSourceRequest.FromString(action.body.to_pybytes())
            yield self._handle_remove_source(req)
        else:
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
        carries the now-resolved ``DataSourceDescriptor`` in its ``result`` arm.

        Resolving an already-resident source is a cheap no-op (returns its
        descriptor). If the client disconnects mid-resolve the daemon thread runs
        to completion and caches the result on the adapter, so a retry coalesces
        onto the finished work rather than downloading again.
        """
        adapter = self.sources.get(source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {source_id}")

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
                result["desc"] = adapter.resolve()
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
        yield ResolveStreamMessage(result=result["desc"]).SerializeToString()

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
        warming is our own loop, so it runs inline in this generator: progress is
        yielded between files (throttled to ``_WARM_PROGRESS_MIN_INTERVAL``) and
        ``context.is_cancelled()`` is polled between files and read blocks, so a
        client closing the stream halts the recall promptly. Warming is a pure
        side-effect (residency), so a cancel genuinely stops -- there is no result
        to preserve.

        Properties:
        - **No-op for non-directory sources** -- a single-file source's one file
          was already recalled by resolve, so this emits one terminal ``done``
          with ``files_total == 0`` and returns.
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
        # Single-file / remote / non-directory source: nothing to warm beyond what
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
        buf = bytearray(_WARM_READ_BLOCK_BYTES)

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
                for _size, fpath in entries:
                    if context.is_cancelled():
                        break
                    name = os.path.basename(fpath)
                    try:
                        with open(fpath, "rb", buffering=0) as fh:
                            while True:
                                if context.is_cancelled():
                                    break
                                n = fh.readinto(buf)
                                if not n:
                                    break
                                bytes_done += n
                                now = time.monotonic()
                                if now - last_yield >= _WARM_PROGRESS_MIN_INTERVAL:
                                    last_yield = now
                                    yield _progress(
                                        files_total,
                                        files_done,
                                        bytes_total,
                                        bytes_done,
                                        name,
                                    )
                    except OSError as exc:
                        logger.warning("warm: skipping %s: %s", fpath, exc)
                    files_done += 1
                    now = time.monotonic()
                    if now - last_yield >= _WARM_PROGRESS_MIN_INTERVAL:
                        last_yield = now
                        yield _progress(
                            files_total,
                            files_done,
                            bytes_total,
                            bytes_done,
                            name,
                        )

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
                dim_labels=list(req.dim_labels),
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
                    _, added, already_present, failed = event
                    result = AddSourceResult(
                        already_present=already_present,
                    )
                    result.added.extend(d for d in added if d is not None)
                    for path, reason in failed:
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
        """List all available data sources.

        Each flight represents a data source (which may contain multiple tensors).

        Results are capped at `max_list_flights_results` for safety. Truncation
        is signaled via schema metadata on all returned FlightInfos.

        Served from the DuckDB catalog when a metadata DB is present, so the
        browse surface cannot drift from ``query_sources`` (biopb/biopb#265); an
        embedded/test server built without a DB (``metadata_db=None``) falls back
        to iterating adapters.

        Args:
            context: Server call context
            criteria: Unused criteria bytes

        Yields:
            FlightInfo for each registered data source (up to max_list_flights_results)
        """
        if self._metadata_db is not None:
            yield from self._list_flights_from_catalog()
        else:
            yield from self._list_flights_from_adapters()

    def _list_flights_from_catalog(self) -> Iterator[flight.FlightInfo]:
        """Build ListFlights results from the DuckDB catalog (the default path).

        One SQL read replaces the per-adapter ``get_source_descriptor()`` calls.
        Token-protected sources never appear here: the only ones that exist live
        in the embedded image-base server, which runs with ``metadata_db=None``
        and therefore takes the adapter fallback instead -- so the DuckDB path
        has no tokened source to leak (biopb/biopb#265).
        """
        max_sources = self._max_list_flights_results
        descriptors, total_sources = self._metadata_db.list_source_descriptors(
            limit=max_sources
        )
        returned_count = len(descriptors)
        truncated = total_sources > returned_count

        if truncated:
            logger.warning(
                f"list_flights truncated: returning {returned_count} of {total_sources} sources"
            )

        base_metadata = {
            b"total_sources": str(total_sources).encode(),
            b"max_sources": str(max_sources).encode(),
            b"returned_sources": str(returned_count).encode(),
            b"truncated": str(truncated).encode(),
        }

        for source_desc in descriptors:
            schema = pa.schema([], metadata=base_metadata)
            flight_descriptor = flight.FlightDescriptor.for_command(
                source_desc.SerializeToString()
            )
            endpoint = flight.FlightEndpoint(
                ticket=flight.Ticket(b""),  # Empty ticket for listing
                locations=[],
            )
            yield flight.FlightInfo(
                schema=schema,
                descriptor=flight_descriptor,
                endpoints=[endpoint],
                total_records=-1,
                total_bytes=-1,
            )

    def _list_flights_from_adapters(self) -> Iterator[flight.FlightInfo]:
        """List sources by iterating adapters (fallback when no metadata DB).

        Used by embedded/test servers built with ``metadata_db=None`` (e.g. the
        image-base result-cache server). Honors per-source capability tokens by
        skipping token-protected sources from enumeration.
        """
        source_items = self.sources.snapshot()
        total_sources = len(source_items)
        max_sources = self._max_list_flights_results
        returned_count = min(total_sources, max_sources)
        truncated = total_sources > max_sources

        if truncated:
            logger.warning(
                f"list_flights truncated: returning {max_sources} of {total_sources} sources"
            )

        # Build base schema metadata for truncation signaling
        base_metadata = {
            b"total_sources": str(total_sources).encode(),
            b"max_sources": str(max_sources).encode(),
            b"returned_sources": str(returned_count).encode(),
            b"truncated": str(truncated).encode(),
        }

        count = 0
        skipped = 0
        for source_id, adapter in source_items:
            if count >= max_sources:
                break

            # Token-protected sources (per-source capabilities) are not
            # enumerable: knowing the source_id must not be enough to list them.
            if adapter.capability_token:
                continue

            # Building a source's descriptor can fail (e.g. an aicsimageio
            # source whose scene-switching fallback raises). A single bad source
            # must not abort the whole listing, so skip it and continue — the
            # FlightInfo is built fully inside the try and only yielded on
            # success, so a partial flight is never emitted.
            try:
                source_desc = adapter.get_source_descriptor()

                # Build schema with truncation metadata
                schema = pa.schema([], metadata=base_metadata)

                # Create a FlightDescriptor for this source
                flight_descriptor = flight.FlightDescriptor.for_command(
                    source_desc.SerializeToString()
                )

                # Create a single endpoint for listing (no specific tensor selected)
                endpoint = flight.FlightEndpoint(
                    ticket=flight.Ticket(b""),  # Empty ticket for listing
                    locations=[],
                )

                info = flight.FlightInfo(
                    schema=schema,
                    descriptor=flight_descriptor,
                    endpoints=[endpoint],
                    total_records=-1,
                    total_bytes=-1,
                )
            except Exception as e:
                logger.error(
                    f"list_flights: skipping source {source_id} due to "
                    f"descriptor build failure: {e}",
                    exc_info=True,
                )
                skipped += 1
                continue

            yield info
            count += 1

        if skipped:
            logger.warning(
                f"list_flights: skipped {skipped} source(s) that failed to build descriptors"
            )

    def get_flight_info(
        self, context: flight.ServerCallContext, descriptor: flight.FlightDescriptor
    ) -> flight.FlightInfo:
        """Get metadata and chunk endpoints for a tensor.

        Args:
            context: Server call context
            descriptor: Flight descriptor with FlightCmd

        Returns:
            FlightInfo with schema and chunk endpoints
        """
        import json

        cmd = FlightCmd.FromString(descriptor.command)

        # Dispatch based on source_id
        if cmd.source_id == "__metadata_query__":
            # Metadata SQL query branch
            if cmd.HasField("metadata_query"):
                sql = cmd.metadata_query.sql
                logger.debug(f"get_flight_info: metadata_query sql={sql[:100]}...")
                if self._metadata_db is None:
                    # The CLI always attaches a metadata DB (mandatory,
                    # biopb/biopb#225); reaching here means this server was
                    # constructed without one (an embedded/test instance).
                    raise flight.FlightServerError(
                        "This server has no metadata database attached, so SQL "
                        "queries are unavailable."
                    )
                try:
                    return self._metadata_db.handle_query(sql)
                except ValueError as e:
                    # Query validation or execution failure
                    raise flight.FlightInternalError(
                        f"Metadata query failed: {e}"
                    ) from e
            else:
                raise flight.FlightServerError(
                    "Metadata query source_id but no MetadataQueryOption"
                )

        # Tensor read branch
        if not cmd.HasField("tensor_read"):
            raise flight.FlightServerError(
                f"Tensor read source_id '{cmd.source_id}' but no TensorReadOption"
            )

        read_opt = cmd.tensor_read
        source_id = cmd.source_id
        tensor_id = read_opt.tensor_id

        self._authorize_source(context, source_id)

        # Reduce the request tensor_id to the within-source field -- or None =
        # "the source's default (first) tensor" (identity policy: array_id is
        # source_id or source_id/field). Both a falsy tensor_id (proto3 default
        # "") and a bare source_id reduce to None. The wire descriptor still
        # reports the full array_id, carried by get_tensor_descriptor().
        field = self._field_within_source(source_id, tensor_id)

        # Substitute the source's default (first) tensor for every no-field
        # request -- empty tensor_id *and* a bare source_id, both -> field None
        # (#44). get_flight_info / get_source / get_physical_scale are documented
        # to accept no tensor_id; forwarding None to a multi-tensor adapter's
        # get_tensor_adapter would otherwise select a bogus field (a bioio scene
        # lookup on None, OME-Zarr HCS field parsing crashing on None.split), so
        # honor the documented default in this one chokepoint rather than at every
        # adapter call site. The first descriptor's array_id is the same default
        # the client's own get_tensor path resolves to; re-reducing it yields the
        # sole tensor's field (None for a single-tensor source, whose array_id ==
        # source_id, so the base still returns self).
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

            # Populate metadata_json in response descriptor if requested
            if read_opt.with_metadata:
                # Prefer the catalog's stored metadata_json (biopb/biopb#253):
                # computed once at registration, read back with a cheap local
                # SELECT -- no adapter recompute, and for a remote proxy no
                # upstream RPC (read the local mirror row directly, never
                # adapter.get_metadata()). Fall back to the adapter only when
                # there is no DB, or get_metadata_json returns None -- an absent/
                # NULL row (empty metadata, or an unresolved source whose real row
                # isn't written yet), unparseable JSON, or a catalog read error
                # (it parses and degrades internally, never raising).
                #
                # Escape hatch: the catalog row is source-level, so read it only
                # when the source's metadata covers every tensor. HCS plates hold
                # per-field metadata (the row is the plate .zattrs, not a field's
                # OME metadata), so they fall through to the per-tensor adapter --
                # preserving the field-level answer (biopb/biopb#253).
                raw_metadata = None
                if (
                    self._metadata_db is not None
                    and source_adapter is not None
                    and source_adapter.metadata_covers_all_tensors()
                ):
                    raw_metadata = self._metadata_db.get_metadata_json(source_id)
                if raw_metadata is None:
                    raw_metadata = tensor_adapter.get_metadata()
                if raw_metadata and source_adapter is not None:
                    wrapped_metadata = {
                        "type": source_adapter.source_type,
                        "dim_label": list(read_plan.descriptor.dim_labels),
                        "metadata": raw_metadata,
                    }
                    read_plan.descriptor.metadata_json = json.dumps(
                        wrapped_metadata, cls=NumpyEncoder
                    )
        except SourceUnresolvedError as e:
            # Expected for a not-yet-hydrated source: surface the same retriable
            # "open to resolve" mapping as the adapter-lookup path (to_flight_error),
            # rather than burying it in "Metadata error" as a bare ValueError ->
            # INTERNAL. Must precede the ValueError clause (it subclasses ValueError).
            raise to_flight_error(e) from e
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise flight.FlightInternalError(
                f"Metadata error for {source_id}: {e}"
            ) from e

        # Compact-grid response (biopb/biopb#346): when the client opted in and
        # the plan is a regular tiling, omit the whole endpoint list and let the
        # client regenerate every chunk_id + bounds from the descriptor -- turning
        # O(n_chunks) serialize/transfer/parse into O(1). The descriptor already
        # carries shape/chunk_shape/scale_hint/reduction_method; we add the two
        # things the client cannot otherwise recover:
        #   * chunk_array_id -- the array_id the chunk_ids are encoded with, read
        #     off an actual endpoint. It differs from descriptor.array_id on a
        #     precompute plan (chunk_ids carry source_id/{level}); reading it from
        #     the endpoint is correct for every planner without special-casing.
        #   * the realized virtual-coordinate bounds [start, stop) as slice_hint,
        #     taken from the first/last chunk (np.ndindex emits the min/max corner
        #     first/last), so the client's virtual-grid reconstruction is exact
        #     even for a full read (where slice_hint was otherwise unset and the
        #     base extent is unrecoverable under a scale_hint's ceil-division).
        if (
            read_opt.compact_grid_ok
            and read_plan.regular_grid
            and read_plan.chunk_endpoints
        ):
            desc = read_plan.descriptor
            chunk_array_id, first_bounds = decode_chunk_id(
                read_plan.chunk_endpoints[0].chunk_id
            )
            _, last_bounds = decode_chunk_id(read_plan.chunk_endpoints[-1].chunk_id)
            desc.chunk_array_id = chunk_array_id
            desc.ClearField("slice_hint")
            desc.slice_hint.start[:] = list(first_bounds.start)
            desc.slice_hint.stop[:] = list(last_bounds.stop)
            logger.debug(
                "get_flight_info: compact grid, omitting %d endpoints",
                len(read_plan.chunk_endpoints),
            )
            return flight.FlightInfo(
                schema=schema,
                descriptor=flight.FlightDescriptor.for_command(
                    desc.SerializeToString()
                ),
                endpoints=[],
                total_records=-1,
                total_bytes=-1,
            )

        # Convert to FlightEndpoints
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
        return flight.FlightInfo(
            schema=schema,
            descriptor=flight.FlightDescriptor.for_command(
                read_plan.descriptor.SerializeToString()
            ),
            endpoints=endpoints,
            total_records=-1,
            total_bytes=-1,
        )

    def do_get(
        self, context: flight.ServerCallContext, ticket: flight.Ticket
    ) -> flight.FlightDataStream:
        """Fetch a chunk's data or metadata query result.

        Args:
            context: Server call context
            ticket: Flight ticket with TensorTicket or metadata query ID

        Returns:
            FlightDataStream with the chunk data or query result
        """
        # Check for metadata query result by checking the ticket prefix
        ticket_bytes = ticket.ticket
        metadata_prefix = b"metadata-query-"
        if ticket_bytes.startswith(metadata_prefix):
            ticket_id = ticket_bytes.decode()
            logger.debug(f"do_get: metadata query result ticket={ticket_id}")
            if self._metadata_db is None:
                raise flight.FlightInternalError("Metadata database not enabled")
            result = self._metadata_db.get_pending_result(ticket_id)
            if result is None:
                # Internal error: pending result should exist if ticket was valid
                raise flight.FlightInternalError(
                    f"Metadata query result not found: {ticket_id}"
                )
            return flight.RecordBatchStream(result)

        # Heavy chunk-read path: track it as in-flight so the background
        # precache worker stays idle while real reads are happening.
        with self.activity.serving_request():
            tensor_ticket = self._parse_ticket(ticket)
            logger.debug(f"do_get: chunk_id={tensor_ticket.chunk_id[:16]}...")

            source_id = decode_chunk_id(tensor_ticket.chunk_id)[0].split("/")[0]
            self._authorize_source(context, source_id)

            try:
                adapter = self._get_adapter_for_chunk(tensor_ticket.chunk_id)
            except (
                SourceUnresolvedError,
                TensorResolutionError,
                ValueError,
                KeyError,
                AttributeError,
                TypeError,
            ) as e:
                # Same resolution mapping as get_flight_info: a ticket naming a
                # field the source no longer has -> terminal NOT_FOUND with a code
                # (never a bare exception -> INTERNAL), an unresolved source ->
                # retriable "open to resolve" (issue #378).
                raise _adapter_lookup_error(
                    e, f"Tensor not found for chunk {tensor_ticket.chunk_id[:16]!r}"
                ) from e
            if adapter is None:
                # Same taxonomy as get_flight_info's tensor_adapter-is-None sibling:
                # a stale ticket whose source is no longer registered -> terminal
                # NOT_FOUND *with a code*, not a bare FlightServerError (issue #378).
                raise to_flight_error(
                    TensorNotFound(
                        f"Adapter not found for chunk_id: "
                        f"{tensor_ticket.chunk_id[:16]!r}",
                        reason="unknown_source",
                    )
                )

            # Get cache manager singleton (if initialized)
            cache_manager = CacheManager.get_instance()

            # Read the chunk, using the configured cache backend when applicable.
            try:
                record_batch = adapter.resolve_chunk_data(
                    tensor_ticket.chunk_id, cache_manager
                )
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

    def _handle_chunk_locate(self, chunk_id: bytes) -> str:
        """Locate a cached chunk on disk for the localhost cache-file handoff.

        Locates the chunk's Arrow IPC message in the file cache and returns its
        on-disk byte range as JSON. If the chunk isn't cached yet, materializes
        it first (same path as do_get) and retries the locate -- so a warm chunk
        is answered without re-reading or re-decoding it. Returns
        ``{"available": false}`` when the chunk can't be located (memory backend,
        oversized/uncached chunk, or any resolve/locate failure) so the client
        falls back to do_get. (issue #9)
        """
        cache_manager = CacheManager.get_instance()
        if cache_manager is None:
            return json.dumps({"available": False})

        try:
            adapter = self._get_adapter_for_chunk(chunk_id)
        except (
            SourceUnresolvedError,
            TensorResolutionError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
        ) as e:
            # Same resolution mapping as get_flight_info / do_get (issue #378).
            raise _adapter_lookup_error(
                e, f"Tensor not found for chunk {chunk_id[:16]!r}"
            ) from e
        if adapter is None:
            # Same taxonomy as get_flight_info / do_get: an unregistered source ->
            # terminal NOT_FOUND with a code, not a bare FlightServerError (#378).
            raise to_flight_error(
                TensorNotFound(
                    f"Adapter not found for chunk_id: {chunk_id[:16]!r}",
                    reason="unknown_source",
                )
            )

        # Entries are stored under the method-stripped canonical key
        # (biopb/biopb#76); locate with the same key or a warm chunk cached
        # under a different reduction_method is never found.
        cache_key = cache_key_for_chunk_id(chunk_id)
        try:
            # If the chunk is already cached, just locate it. Resolving first
            # would, on a chunk whose in-RAM entry has been trimmed, re-read the
            # whole chunk from its segment server-side for nothing. Only
            # materialize (same path as do_get) on a genuine cold miss.
            location = cache_manager.locate_entry(cache_key)
            if location is None:
                adapter.resolve_chunk_data(chunk_id, cache_manager)
                location = cache_manager.locate_entry(cache_key)
        except (OSError, ValueError) as e:
            raise flight.FlightInternalError(
                f"I/O error locating chunk data: {e}"
            ) from e

        if location is None:
            return json.dumps({"available": False})

        return json.dumps(
            {
                "available": True,
                "format_version": CACHE_FILE_FORMAT_VERSION,
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
        """Handle source creation and chunk upload.

        Args:
            context: Server call context
            descriptor: Flight descriptor with command bytes
            reader: Flight data stream reader
            writer: Flight metadata writer for responses

        Raises:
            FlightUnauthenticatedError: If server not in write mode
            FlightServerError: If source/chunk creation fails
        """
        if not self._writable:
            raise flight.FlightUnauthenticatedError("Server not in write mode")

        command = descriptor.command

        # Try TensorDescriptor (source creation)
        try:
            req_desc = TensorDescriptor.FromString(command)
            if req_desc.shape and req_desc.dtype:
                writer.write(self.uploads.create_source(req_desc).SerializeToString())
                return
        except Exception:
            pass

        # Chunk upload - use ChunkUpload wrapper
        try:
            upload = ChunkUpload.FromString(command)
            self.uploads.write_chunk(upload, reader)
            return
        except Exception as e:
            raise flight.FlightServerError(f"Invalid upload command: {e}")


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
        server.register_source(source_id, adapter)
    # All sources registered up front -> ready immediately (health: SERVING).
    server.mark_ready()

    print(f"Starting Flight server at {location}")
    server.serve()
