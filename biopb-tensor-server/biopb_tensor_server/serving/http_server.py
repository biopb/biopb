"""FastAPI HTTP sidecar for TensorFlight server.

Exposes the TensorFlightClient over a browser-friendly HTTP/JSON + binary API.

Endpoints:
  GET  /livez                        — liveness probe (no auth)
  GET  /readyz                       — readiness probe (no auth)
  GET  /healthz                      — alias for /readyz (no auth)
  GET  /api/diagnostics              — runtime diagnostics (token required)
  GET  /api/sources                  — list DataSourceDescriptors (token required)
  POST /api/sources/query            — SQL query against source metadata (token required)
  GET  /api/sources/{source_id}      — single DataSourceDescriptor (token required)
  GET  /api/sources/{source_id}/metadata — parsed metadata_json (token required)
  POST /api/slice                    — fetch array slice as binary (token required)

Authentication:
  Pass the website token in the ``Authorization: Bearer <token>`` header or
  ``X-Biopb-Token`` header on every protected request.  /livez, /readyz and
  /healthz are always unauthenticated so proxies can probe them.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.flight as flight
from biopb import _web_auth
from biopb.tensor.client import TensorFlightClient, _request_crop_slices
from biopb.tensor.ticket_pb2 import TensorTicket
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version / constants
# ---------------------------------------------------------------------------

try:
    import importlib.metadata as _importlib_metadata
except ImportError:
    import importlib_metadata as _importlib_metadata

try:
    _VERSION = _importlib_metadata.version("biopb-tensor-server")
except Exception:
    _VERSION = "0.1.0"
_SERVICE = "biopb-tensor-http"

# Number of completed requests to track for latency percentiles
_LATENCY_WINDOW = 200
# Minimum samples before we report percentiles as stable
_METRICS_READY_MIN = 20


# ---------------------------------------------------------------------------
# Diagnostics ring buffer
# ---------------------------------------------------------------------------


class _LatencyTracker:
    """Rolling window of request latency samples (thread-safe)."""

    def __init__(self, window: int = _LATENCY_WINDOW) -> None:
        self._lock = threading.Lock()
        self._samples: Deque[float] = collections.deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._samples.append(latency_ms)

    def percentile(self, p: float) -> Optional[float]:
        with self._lock:
            if not self._samples:
                return None
            sorted_samples = sorted(self._samples)
            idx = (len(sorted_samples) - 1) * p / 100.0
            lo = int(idx)
            hi = lo + 1
            if hi >= len(sorted_samples):
                return round(sorted_samples[lo], 2)
            frac = idx - lo
            return round(sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac, 2)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def metrics_ready(self) -> bool:
        return self.count >= _METRICS_READY_MIN


class _ErrorEvent(BaseModel):
    timestamp: str
    code: Optional[str]
    message: Optional[str]


class _DiagnosticsState:
    """Shared mutable state for the diagnostics endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.connection_state: str = "disconnected"
        self.degraded_mode: bool = False
        self.pixel_budget: Optional[int] = None  # last used, client-supplied
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.latency = _LatencyTracker()
        self._errors: Deque[_ErrorEvent] = collections.deque(maxlen=20)
        # per-session rate limiting: session_id → (count, window_start)
        self._rate_limit: Dict[str, Tuple[int, float]] = {}

    # --- Connection state helpers ---

    def mark_connected(self) -> None:
        with self._lock:
            self.connection_state = "connected"
            self.degraded_mode = False

    def mark_error(self, code: Optional[str], message: Optional[str]) -> None:
        with self._lock:
            self.connection_state = "error"
            self._errors.append(
                _ErrorEvent(
                    timestamp=_now_rfc3339(),
                    code=code,
                    message=_redact(message),
                )
            )

    def mark_degraded(self) -> None:
        with self._lock:
            self.degraded_mode = True

    # --- Cache helpers ---

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    # --- Rate-limit helper (1 req/s per session) ---

    def check_rate_limit(self, session_id: str) -> bool:
        """Return True if the request is allowed."""
        now = time.monotonic()
        with self._lock:
            count, window_start = self._rate_limit.get(session_id, (0, now))
            if now - window_start >= 1.0:
                # new window
                self._rate_limit[session_id] = (1, now)
                return True
            if count < 1:
                self._rate_limit[session_id] = (count + 1, window_start)
                return True
            return False

    # --- Snapshot ---

    def snapshot(self, dev_mode: bool) -> Dict[str, Any]:
        with self._lock:
            hits = self.cache_hits
            misses = self.cache_misses
            total = hits + misses
            cache_hit_rate = round(hits / total, 4) if total > 0 else None
            last_error = self._errors[-1] if self._errors else None

        return {
            "status": "ok",
            "timestamp": _now_rfc3339(),
            "dev_mode": dev_mode,
            "connection_state": self.connection_state,
            "degraded_mode": self.degraded_mode,
            "pixel_budget": self.pixel_budget,
            "cache_hit_rate": cache_hit_rate,
            "latency_p50_ms": self.latency.percentile(50),
            "latency_p95_ms": self.latency.percentile(95),
            "last_error_code": last_error.code if last_error else None,
            "last_error_message": last_error.message if last_error else None,
            "metrics_ready": self.latency.metrics_ready,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_PATH_LIKE = re.compile(r"(/[^\s]{3,}|[A-Za-z]:\\[^\s]{3,})")
_TOKEN_LIKE = re.compile(r"[A-Za-z0-9_\-]{16,}")


def _redact(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = _PATH_LIKE.sub("[REDACTED]", text)
    text = _TOKEN_LIKE.sub("[REDACTED]", text)
    return text


def _tensor_matches(td_array_id: str, req_tensor_id: str, source_id: str) -> bool:
    """Whether *req_tensor_id* refers to the descriptor whose array_id is
    *td_array_id*, tolerant of identity-policy forms.

    A catalog descriptor carries the globally-unique array_id (``source_id`` or
    ``source_id/field``), but a browser/TS caller may address a tensor by the
    bare within-source ``field``. Compare after reducing both sides to the field
    (strip a leading ``source_id/``) so the lookup matches either form. Used only
    for the best-effort dim-label attachment, never for the read itself.
    """
    if td_array_id == req_tensor_id:
        return True
    prefix = f"{source_id}/"

    def field(value: str) -> str:
        return value[len(prefix) :] if value.startswith(prefix) else value

    return field(td_array_id) == field(req_tensor_id)


def _request_array_id(source_id: str, tensor_id: Optional[str]) -> str:
    """Build the globally-unique array_id (identity policy) from a request's
    separate ``(source_id, tensor_id)`` fields.

    A tensor is addressed by its array_id ALONE -- ``source_id`` for a
    single-tensor source or ``source_id/field`` for a multi-tensor one (see the
    policy at the top of ``proto/biopb/tensor/descriptor.proto``). The TS client
    sends the array_id verbatim in ``tensor_id``; a browser/HTTP caller may
    tolerantly send a bare within-source ``field`` (or nothing). Normalize all
    three to the qualified array_id so the read goes through the array_id-first
    SDK path without the deprecated ``(source_id, tensor_id)`` addressing.
    """
    if not tensor_id or tensor_id == source_id:
        return source_id
    if tensor_id.startswith(f"{source_id}/"):
        return tensor_id
    return f"{source_id}/{tensor_id}"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SliceRequest(BaseModel):
    source_id: str
    tensor_id: str
    slice_start: Optional[List[int]] = None
    slice_stop: Optional[List[int]] = None
    scale_hint: Optional[List[int]] = None
    reduction_method: Optional[str] = None
    pixel_budget: Optional[int] = None  # informational, stored in diagnostics


class QuerySourcesRequest(BaseModel):
    sql: str


class RenderRequest(BaseModel):
    """Request for backend-rendered image output.

    Returns PNG/JPEG image instead of raw numpy bytes.
    Uses VTK or PIL for rendering on the server side.
    """

    source_id: str
    tensor_id: str
    slice_start: Optional[List[int]] = None
    slice_stop: Optional[List[int]] = None
    scale_hint: Optional[List[int]] = None
    reduction_method: Optional[str] = None
    percentile_lo: float = 1.0
    percentile_hi: float = 99.0
    color: str = "auto"  # preset name or hex (#rrggbb)
    channel_name: Optional[str] = None  # for auto color resolution
    use_min_max: bool = False  # use full min-max range instead of percentiles
    output_format: str = "png"  # "png" or "jpeg"
    pixel_budget: Optional[int] = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-app context (was: closures captured inside create_app)
# ---------------------------------------------------------------------------


class _SidecarContext:
    """Per-app mutable state + helpers, stored on ``app.state.sidecar``.

    Holds exactly what the route handlers used to close over when they were
    nested inside ``create_app``: the lazily-connected Flight client, the
    diagnostics aggregator, and the auth config. Handlers reach it via
    ``request.app.state.sidecar`` (or ``websocket.app.state.sidecar``), so each
    handler is now a module-level, individually-testable function rather than a
    closure — which is what keeps ``create_app`` itself trivially simple.
    """

    def __init__(
        self,
        flight_location: str,
        token: Optional[str],
        cache_bytes: int,
        config_path: Optional[str] = None,
        web_host: Optional[str] = None,
        web_port: Optional[int] = None,
        supervised: bool = False,
    ) -> None:
        self.flight_location = flight_location
        self.token = token
        self.cache_bytes = cache_bytes
        # Admin-route config: the config file this daemon was launched with and
        # the bind args echoed into a self-restart so it comes back identically
        # (biopb/biopb#237).
        self.config_path = config_path
        self.web_host = web_host
        self.web_port = web_port
        # True when the biopb control spawned + supervises this data plane (it
        # sets BIOPB_DATA_PLANE_SUPERVISED in our env). The admin self-restart is
        # then forbidden: the control is the sole owner, so a self-spawned
        # `biopb server restart` would race the supervisor and silently hand the
        # plane to a daemon the control did not start (biopb/biopb#418).
        self.supervised = supervised
        self.diag = _DiagnosticsState()
        # Lazy-init Flight client (first request will connect)
        self._client_lock = threading.Lock()
        self._client_holder: Dict[str, Optional[TensorFlightClient]] = {"client": None}
        # Latches once POST /api/admin/restart spawns the detached restart child,
        # so a rapid second request can't spawn a competing `biopb server restart`
        # that would race on the PID file / port (biopb/biopb#237).
        self._restart_state: Dict[str, bool] = {"in_progress": False}

    def get_client(self) -> TensorFlightClient:
        """Return the Flight client, connecting on first use."""
        with self._client_lock:
            if self._client_holder["client"] is None:
                try:
                    logger.debug(
                        f"Connecting to Flight server at {self.flight_location}"
                    )
                    self._client_holder["client"] = TensorFlightClient(
                        location=self.flight_location,
                        cache_bytes=self.cache_bytes,
                        token=self.token,
                    )
                    self.diag.mark_connected()
                    logger.info(f"Connected to Flight server at {self.flight_location}")
                except Exception as exc:
                    self.diag.mark_error("CONNECTION_FAILED", str(exc))
                    logger.error(f"Failed to connect to Flight server: {exc}")
                    raise
            return self._client_holder["client"]

    def peek_client(self) -> Optional[TensorFlightClient]:
        """Return the client only if already connected (never forces a connect)."""
        with self._client_lock:
            return self._client_holder["client"]

    def check_token(self, request: Request) -> None:
        """Raise 401 if the request does not carry a valid token.

        Delegates the token decision to the shared ``biopb._web_auth`` policy
        (the single source the control uses too). A ``None`` token — local mode,
        where every listener is loopback-bound — is the "no token enforced" case,
        expressed as a falsy ``expected``.
        """
        expected = self.token
        if not _web_auth.token_valid(request.headers.get, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing token")


def _sidecar(request: Request) -> _SidecarContext:
    """Fetch the per-app context off ``app.state`` (handler dependency)."""
    return request.app.state.sidecar


def _require_same_origin(request: Request) -> None:
    """Refuse drive-by cross-origin state changes on the mutating routes.

    The admin routes are the sidecar's first *mutating* surface. A page the
    user merely visits can fire a cross-origin ``POST``/``PUT`` at the
    loopback sidecar; it cannot read the response (CORS) but a state change
    does not need to. The CSRF decision lives in the shared
    ``biopb._web_auth.is_forgeable_cross_site`` policy: a request carrying a
    token header is not forgeable, and a browser that stamped
    ``Sec-Fetch-Site`` cross-site is the vector; a non-browser client (curl)
    sends none and is allowed -- a token-gated server still enforces
    ``check_token`` independently.
    """
    if _web_auth.is_forgeable_cross_site(request.headers.get):
        raise HTTPException(status_code=403, detail="Cross-origin request refused")


# ---------------------------------------------------------------------------
# Shared request helpers (deduplicated out of the slice/render/ws handlers)
# ---------------------------------------------------------------------------


def _build_slice_hint(
    slice_start: Optional[List[int]],
    slice_stop: Optional[List[int]],
) -> Optional[Tuple[slice, ...]]:
    """Build a slice-hint tuple (world coords) from start/stop lists.

    Returns ``None`` when either bound is absent. Raises ``HTTPException(422)``
    on a length mismatch; the slice/render handlers catch ``HTTPException`` and
    re-raise it unchanged, and the websocket handler turns it into an error
    message with the same text.
    """
    if slice_start is None or slice_stop is None:
        return None
    if len(slice_start) != len(slice_stop):
        raise HTTPException(
            status_code=422,
            detail="slice_start and slice_stop must have the same length",
        )
    # slice_hint is applied BEFORE scaling, so coordinates are in original units
    return tuple(slice(s, e) for s, e in zip(slice_start, slice_stop))


def _dim_labels_for(
    client: TensorFlightClient,
    source_id: str,
    tensor_id: Optional[str],
) -> List[str]:
    """Look up a tensor's dim labels from the client's cached descriptors.

    Returns ``[]`` when not found (callers apply their own fallback). Mirrors
    the inline lookup the slice/render handlers used against ``client._sources``.
    """
    try:
        sources = client._sources  # type: ignore[attr-defined]
        if source_id in sources:
            for td in sources[source_id].tensors:
                if _tensor_matches(td.array_id, tensor_id, source_id):
                    return list(td.dim_labels)
    except Exception:
        pass
    return []


def _image_media_type(output_format: str) -> str:
    """Map a render output format to its HTTP media type."""
    fmt = output_format.lower()
    if fmt == "raw":
        return "application/octet-stream"  # Raw RGBA bytes
    if fmt == "png":
        return "image/png"
    return "image/jpeg"


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Coerce to native byte order + C-contiguous for predictable wire bytes."""
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Routes
#
# All handlers are module-level functions registered on this one router (the
# registration order below is load-bearing: the /metadata and /ticket routes
# must precede the greedy {source_id:path} catch-all). create_app() simply
# include_router()s it, so per-handler complexity is measured per-handler.
# ---------------------------------------------------------------------------

_router = APIRouter()

# Cap on entries returned from one /api/admin/browse listing, so a directory with
# tens of thousands of files can't produce a giant payload; the chooser shows a
# "truncated" note and the user navigates in rather than paginating.
_BROWSE_MAX_ENTRIES = 2000


# -- Health endpoints (unauthenticated) -------------------------------------


@_router.get("/livez")
async def livez() -> JSONResponse:
    return JSONResponse({"status": "ok", "timestamp": _now_rfc3339()})


@_router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    backend_health = None
    client = ctx.peek_client()
    if client is not None:
        try:
            backend_health = client.health_check()
        except Exception as e:
            logger.warning(f"Backend health check failed: {e}")

    ready = (
        backend_health and backend_health.get("status") == "SERVING"
    ) or ctx.diag.connection_state == "connected"

    return JSONResponse(
        {
            "status": "ok" if ready else "degraded",
            "timestamp": _now_rfc3339(),
            "ready": ready,
            "dev_mode": ctx.token is None,
            "service": _SERVICE,
            "version": _VERSION,
            "backend_health": backend_health,
            "source_count": backend_health.get("source_count", 0)
            if backend_health
            else 0,
        }
    )


@_router.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    return await readyz(request)


# -- Diagnostics (token required) -------------------------------------------


@_router.get("/api/diagnostics")
async def diagnostics(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    # Soft rate limit per session (identify by token itself as session key)
    session_id = request.headers.get("X-Biopb-Token", "") or request.headers.get(
        "Authorization", ""
    )
    if not ctx.diag.check_rate_limit(session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (1 req/s)")
    # Sync cache stats from Flight client if available
    client = ctx.peek_client()
    if client is not None:
        info = client.cache_info()
        ctx.diag.cache_hits = info.get("hits", ctx.diag.cache_hits)
        ctx.diag.cache_misses = info.get("misses", ctx.diag.cache_misses)
    return JSONResponse(ctx.diag.snapshot(dev_mode=ctx.token is None))


# -- Sources ----------------------------------------------------------------


@_router.get("/api/sources")
async def list_sources(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()
    try:
        client = ctx.get_client()
        sources = client.list_sources()
        result = [_source_desc_to_dict(desc) for desc in sources.values()]
        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.debug(f"list_sources: returned {len(result)} sources in {elapsed:.1f}ms")
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("LIST_SOURCES_FAILED", str(exc))
        logger.error(f"list_sources failed: {exc}")
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )


@_router.post("/api/sources/query")
async def query_sources(req: QuerySourcesRequest, request: Request) -> Response:
    """Execute SQL query against source metadata database.

    Request body: {"sql": "SELECT source_id, source_type FROM sources WHERE ..."}
    Response headers:
      X-Total-Sources    — total matching (before truncation)
      X-Returned-Sources — actual rows returned
      X-Truncated        — "true" if truncated
    Response body: JSON array of query results
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()

    try:
        client = ctx.get_client()
        arrow_table = client.query_sources(req.sql)

        # Convert Arrow Table to JSON
        result = arrow_table.to_pylist()

        # Truncation info from schema metadata
        total = int(arrow_table.schema.metadata.get(b"total_sources", len(result)))
        returned = int(
            arrow_table.schema.metadata.get(b"returned_sources", len(result))
        )
        truncated = total > returned

        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.debug(
            f"query_sources: returned {returned}/{total} rows in {elapsed:.1f}ms"
        )

        headers = {
            "X-Total-Sources": str(total),
            "X-Returned-Sources": str(returned),
            "X-Truncated": str(truncated).lower(),
        }

        return JSONResponse(result, headers=headers)

    except ValueError as exc:
        # SQL validation error (forbidden keyword, disallowed table)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        ctx.diag.mark_error("QUERY_SOURCES_FAILED", str(exc))
        logger.error(f"query_sources failed: {exc}")
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )


# NOTE: the /metadata and /ticket routes must be registered before the greedy
# {source_id:path} route, otherwise Starlette's first-match routing would
# shadow them.
@_router.get("/api/sources/{source_id:path}/metadata")
async def get_source_metadata(source_id: str, request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()
    try:
        client = ctx.get_client()
        metadata = client.get_source_metadata(source_id)
        ctx.diag.latency.record((time.monotonic() - t0) * 1000)
        return JSONResponse(metadata)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        ctx.diag.mark_error("GET_METADATA_FAILED", str(exc))
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )


# -- Chunk (binary response via ticket) -------------------------------------


@_router.get("/api/sources/{source_id:path}/ticket/{ticket_hex}")
async def get_chunk(source_id: str, ticket_hex: str, request: Request) -> Response:
    """Fetch a chunk's raw binary data by hex-encoded ticket.

    Path params:
      - source_id: Data source identifier
      - ticket_hex: TensorTicket.SerializeToString() encoded as hex string

    Response headers:
      X-Shape        — comma-separated dimensions of the returned chunk
      X-Dtype        — numpy dtype string (e.g. "uint16", "float32")
      X-Chunk-Start  — comma-separated start coordinates of the chunk
      X-Chunk-Stop   — comma-separated stop coordinates of the chunk (exclusive)

    Response body:
      C-contiguous raw bytes of the numpy array (no framing).
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()

    logger.debug(f"get_chunk: source={source_id}, ticket_hex={ticket_hex[:16]}...")

    try:
        # Decode hex string to bytes
        ticket_bytes = bytes.fromhex(ticket_hex)

        # Parse TensorTicket to validate (raises on malformed ticket)
        TensorTicket.FromString(ticket_bytes)

        # Get Flight client
        client = ctx.get_client()

        # Fetch chunk data via do_get
        reader = client._client.do_get(
            flight.Ticket(ticket_bytes),
            options=client._call_options,
        )

        # Read all data from the stream. do_get returns the unified binary chunk
        # schema (biopb/biopb#293); decode it, then ensure native byte order +
        # C-contiguous layout for the browser.
        from biopb_tensor_server.core.base import unpack_chunk_array

        table = reader.read_all()
        arr = _normalize_array(unpack_chunk_array(table.to_batches()[0]))

        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.debug(
            f"get_chunk: returned shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes}B in {elapsed:.1f}ms"
        )

        # Build response headers
        headers = {
            "X-Shape": ",".join(str(d) for d in arr.shape),
            "X-Dtype": str(arr.dtype),
            "X-Chunk-Start": "",  # Not available from do_get alone
            "X-Chunk-Stop": "",  # Not available from do_get alone
        }

        return Response(
            content=arr.tobytes(),
            media_type="application/octet-stream",
            headers=headers,
        )

    except ValueError as exc:
        # Invalid hex string or protobuf parse error
        logger.warning(f"get_chunk: invalid ticket: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid ticket: {exc}")
    except flight.FlightError as exc:
        ctx.diag.mark_error("CHUNK_FETCH_FAILED", str(exc))
        logger.error(f"get_chunk: Flight error: {exc}")
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )
    except Exception as exc:
        ctx.diag.mark_error("CHUNK_FAILED", str(exc))
        logger.error(f"get_chunk: unexpected error: {exc}")
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )


@_router.get("/api/sources/{source_id:path}")
async def get_source(source_id: str, request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()
    try:
        client = ctx.get_client()
        sources = client.list_sources()
        if source_id not in sources:
            raise HTTPException(
                status_code=404, detail=f"Source not found: {source_id}"
            )
        ctx.diag.latency.record((time.monotonic() - t0) * 1000)
        return JSONResponse(_source_desc_to_dict(sources[source_id]))
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("GET_SOURCE_FAILED", str(exc))
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )


# -- Slice (binary response) ------------------------------------------------


@_router.post("/api/slice")
async def slice_tensor(req: SliceRequest, request: Request) -> Response:
    """Fetch a slice of a tensor and return raw bytes.

    Response headers:
      X-Shape     — comma-separated dimensions of the returned array
      X-Dtype     — numpy dtype string (e.g. "uint16", "float32")
      X-Dim-Labels — comma-separated semantic axis labels

    Response body:
      C-contiguous raw bytes of the numpy array (no framing).
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()

    logger.debug(
        f"slice: source={req.source_id}, tensor={req.tensor_id}, "
        f"slice={req.slice_start}-{req.slice_stop}, scale={req.scale_hint}, method={req.reduction_method}"
    )

    if req.pixel_budget is not None:
        ctx.diag.pixel_budget = req.pixel_budget

    try:
        client = ctx.get_client()
        slice_hint = _build_slice_hint(req.slice_start, req.slice_stop)

        # Pass slice_hint to gRPC for optimized slicing (world coordinates)
        arr_lazy = client.get_tensor(
            _request_array_id(req.source_id, req.tensor_id),
            slice_hint=slice_hint,
            scale_hint=req.scale_hint or None,
            reduction_method=req.reduction_method or None,
        )

        # Compute (blocking)
        arr = _normalize_array(arr_lazy.compute())

        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.debug(
            f"slice: computed shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes}B in {elapsed:.1f}ms"
        )

        # Attach dim labels from the cached descriptor (empty string if unknown)
        headers = {
            "X-Shape": ",".join(str(d) for d in arr.shape),
            "X-Dtype": str(arr.dtype),
            "X-Dim-Labels": ",".join(
                _dim_labels_for(client, req.source_id, req.tensor_id)
            ),
        }

        return Response(
            content=arr.tobytes(),
            media_type="application/octet-stream",
            headers=headers,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        ctx.diag.mark_error("SLICE_FAILED", str(exc))
        logger.error(f"slice failed: {exc}")
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )


# -- Render (image output) --------------------------------------------------


@_router.post("/api/render")
async def render_tensor(req: RenderRequest, request: Request) -> Response:
    """Render a tensor slice and return PNG/JPEG image.

    Backend rendering using VTK or PIL. Returns compressed image
    instead of raw bytes, potentially more efficient for large datasets.

    Response headers:
      X-Image-Width        — width of rendered image
      X-Image-Height       — height of rendered image
      X-Percentile-Lo-Value — actual computed lo percentile value
      X-Percentile-Hi-Value — actual computed hi percentile value

    Response body:
      PNG or JPEG image bytes.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()

    logger.debug(
        f"render: source={req.source_id}, tensor={req.tensor_id}, "
        f"slice={req.slice_start}-{req.slice_stop}, scale={req.scale_hint}, "
        f"percentiles={req.percentile_lo}-{req.percentile_hi}, "
        f"color={req.color}, format={req.output_format}"
    )

    if req.pixel_budget is not None:
        ctx.diag.pixel_budget = req.pixel_budget

    try:
        client = ctx.get_client()
        slice_hint = _build_slice_hint(req.slice_start, req.slice_stop)

        arr_lazy = client.get_tensor(
            _request_array_id(req.source_id, req.tensor_id),
            slice_hint=slice_hint,
            scale_hint=req.scale_hint or None,
            reduction_method=req.reduction_method or None,
        )

        # Compute (blocking)
        t0_compute = time.monotonic()
        arr: np.ndarray = arr_lazy.compute()
        compute_ms = (time.monotonic() - t0_compute) * 1000

        # Dim labels from descriptor, with a shape-based fallback
        dim_labels = _dim_labels_for(client, req.source_id, req.tensor_id)
        if not dim_labels:
            dim_labels = [f"d{i}" for i in range(arr.ndim)]

        logger.debug(
            f"render: computed shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes}B, "
            f"dim_labels={dim_labels}, compute_time={compute_ms:.1f}ms"
        )

        # Import renderer
        from .renderer import render_array_to_image_bytes

        t0_render = time.monotonic()
        image_bytes, width, height, lo_val, hi_val = render_array_to_image_bytes(
            arr=arr,
            dim_labels=dim_labels,
            percentile_lo=req.percentile_lo if not req.use_min_max else 0.0,
            percentile_hi=req.percentile_hi if not req.use_min_max else 100.0,
            color=req.color,
            channel_name=req.channel_name,
            output_format=req.output_format,
        )
        render_ms = (time.monotonic() - t0_render) * 1000

        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.debug(
            f"render: image size={width}x{height}, "
            f"bytes={len(image_bytes)}, total={elapsed:.1f}ms, "
            f"compute={compute_ms:.1f}ms, render={render_ms:.1f}ms"
        )

        headers = {
            "X-Image-Width": str(width),
            "X-Image-Height": str(height),
            "X-Percentile-Lo-Value": str(lo_val),
            "X-Percentile-Hi-Value": str(hi_val),
            "X-Image-Format": req.output_format.lower(),  # Tell client format used
        }

        return Response(
            content=image_bytes,
            media_type=_image_media_type(req.output_format),
            headers=headers,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Rendering not available: {exc}")
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        ctx.diag.mark_error("RENDER_FAILED", str(exc))
        logger.error(f"render failed: {exc}\n{tb}")
        raise HTTPException(
            status_code=502, detail=f"Render error: {type(exc).__name__}: {exc}"
        )


# -- WebSocket render endpoint ----------------------------------------------


def _ws_authorized(websocket: WebSocket, ctx: _SidecarContext) -> bool:
    """Validate the websocket token from headers or the ``token`` query param.

    Browsers can't set custom headers on a WebSocket handshake, so the shared
    ``biopb._web_auth`` policy accepts the ``?token=`` fallback here; a ``None``
    token (local mode) is the falsy-``expected`` bypass.
    """
    expected = ctx.token
    return _web_auth.token_valid_with_query(
        websocket.headers.get, websocket.query_params.get, expected
    )


def _ws_crop_to_request(dask_arr: Any, ctx_: Any, y_idx: int, x_idx: int) -> Any:
    """Crop the uncropped dask array back to the originally-requested bounds on
    every axis except Y/X, accounting for the realized slice start and scale."""
    if ctx_.original_slice_hint is None or not ctx_.descriptor.HasField("slice_hint"):
        return dask_arr
    scale = list(ctx_.read_opt.scale_hint) if ctx_.read_opt.scale_hint else None
    return dask_arr[
        _request_crop_slices(
            len(ctx_.descriptor.shape),
            ctx_.original_slice_hint,
            ctx_.descriptor.slice_hint,
            scale,
            keep_axes=(y_idx, x_idx),
        )
    ]


def _ws_loaded_region(
    ctx_: Any, dim_labels: List[str], y_idx: int, x_idx: int
) -> Optional[Dict[str, Any]]:
    """Loaded-region metadata from the realized (not requested) slice bounds."""
    if not ctx_.descriptor.HasField("slice_hint"):
        return None
    realized = ctx_.descriptor.slice_hint
    return {
        "x": int(realized.start[x_idx]),
        "y": int(realized.start[y_idx]),
        "width": int(realized.stop[x_idx] - realized.start[x_idx]),
        "height": int(realized.stop[y_idx] - realized.start[y_idx]),
        "scale_factors": list(ctx_.descriptor.scale_hint)
        if ctx_.descriptor.scale_hint
        else [1] * len(dim_labels),
    }


async def _ws_render_one(
    websocket: WebSocket, ctx: _SidecarContext, params: RenderRequest
) -> None:
    """Render a single websocket request and stream metadata + image bytes."""
    t0 = time.monotonic()
    logger.info(
        f"ws/render: source={params.source_id}, tensor={params.tensor_id}, "
        f"slice={params.slice_start}-{params.slice_stop}, scale={params.scale_hint}"
    )

    if params.pixel_budget is not None:
        ctx.diag.pixel_budget = params.pixel_budget

    try:
        client = ctx.get_client()
        slice_hint = _build_slice_hint(params.slice_start, params.slice_stop)

        # Get tensor context (includes realized slice bounds), build the
        # uncropped dask array from its endpoints.
        cctx = client._get_tensor_context(
            source_id=params.source_id,
            tensor_id=params.tensor_id,
            slice_hint=slice_hint,
            scale_hint=params.scale_hint or None,
            reduction_method=params.reduction_method or None,
        )
        dask_arr = client._build_dask_array(
            desc=cctx.descriptor,
            chunks=[ep[0] for ep in cctx.endpoints],
            chunk_bounds=[ep[1] for ep in cctx.endpoints],
        )

        dim_labels: List[str] = list(cctx.descriptor.dim_labels)
        if not dim_labels:
            dim_labels = [f"d{i}" for i in range(dask_arr.ndim)]

        # Case-insensitive Y/X lookup (descriptor labels are uppercase "TCZYXS").
        # A raw dim_labels.index("y") misses "Y" and, for a 6-D RGB TCZYXS
        # layout, its positional fallback would pick X/S as Y/X.
        from .renderer import build_axis_map

        _axis_map = build_axis_map(dim_labels)
        y_idx = _axis_map["y"] if _axis_map["y"] is not None else len(dim_labels) - 2
        x_idx = _axis_map["x"] if _axis_map["x"] is not None else len(dim_labels) - 1

        # Slice to the originally requested bounds (except y/x) before computing.
        dask_arr = _ws_crop_to_request(dask_arr, cctx, y_idx, x_idx)

        t0_compute = time.monotonic()
        arr: np.ndarray = await asyncio.get_event_loop().run_in_executor(
            None, dask_arr.compute
        )
        compute_ms = (time.monotonic() - t0_compute) * 1000

        loaded_region = _ws_loaded_region(cctx, dim_labels, y_idx, x_idx)

        # Import renderer
        from .renderer import render_array_to_image_bytes

        t0_render = time.monotonic()
        image_bytes, width, height, lo_val, hi_val = render_array_to_image_bytes(
            arr=arr,
            dim_labels=dim_labels,
            percentile_lo=params.percentile_lo if not params.use_min_max else 0.0,
            percentile_hi=params.percentile_hi if not params.use_min_max else 100.0,
            color=params.color,
            channel_name=params.channel_name,
            output_format=params.output_format,
        )
        render_ms = (time.monotonic() - t0_render) * 1000

        format_lower = params.output_format.lower()
        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.info(
            f"ws/render: done {width}x{height} {format_lower} "
            f"total={elapsed:.0f}ms compute={compute_ms:.0f}ms render={render_ms:.0f}ms"
        )

        # Send metadata JSON first, then the binary image data
        render_start_msg = {
            "action": "render_start",
            "width": width,
            "height": height,
            "format": format_lower,
            "percentile_lo_value": lo_val,
            "percentile_hi_value": hi_val,
        }
        if loaded_region is not None:
            render_start_msg["loaded_region"] = loaded_region
        await websocket.send_json(render_start_msg)
        await websocket.send_bytes(image_bytes)

    except HTTPException as exc:
        # slice_start/slice_stop length mismatch (preserves the original text)
        await websocket.send_json({"action": "error", "message": exc.detail})
    except ValueError as exc:
        await websocket.send_json({"action": "error", "message": str(exc)})
    except ImportError as exc:
        await websocket.send_json(
            {"action": "error", "message": f"Rendering not available: {exc}"}
        )
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        ctx.diag.mark_error("WS_RENDER_FAILED", str(exc))
        logger.error(f"ws/render failed: {exc}\n{tb}")
        await websocket.send_json(
            {"action": "error", "message": f"Render error: {type(exc).__name__}"}
        )


async def _ws_dispatch(
    websocket: WebSocket, ctx: _SidecarContext, data: Dict[str, Any]
) -> None:
    """Validate one received message and route a render request."""
    action = data.get("action")
    if action != "render":
        await websocket.send_json(
            {"action": "error", "message": f"Unknown action: {action}"}
        )
        return
    try:
        params = RenderRequest(**data.get("params", {}))
    except Exception as e:
        await websocket.send_json(
            {"action": "error", "message": f"Invalid params: {e}"}
        )
        return
    await _ws_render_one(websocket, ctx, params)


@_router.websocket("/ws/render")
async def websocket_render(websocket: WebSocket) -> None:
    """WebSocket endpoint for rendering tensor slices.

    Protocol:
      1. Client connects, sends nothing
      2. Server validates token from headers or query params
      3. Client sends JSON: { action: "render", params: RenderRequest }
      4. Server sends JSON metadata: { action: "render_start", width, height, format }
      5. Server sends binary: JPEG/PNG image bytes
      6. Repeat steps 3-5 for subsequent requests

    No session state — WebSocket is purely request/response. Token is accepted
    from the Authorization header, X-Biopb-Token header, or a "token" query
    parameter (for browsers that can't send custom headers).
    """
    ctx = websocket.app.state.sidecar
    if not _ws_authorized(websocket, ctx):
        await websocket.close(code=4001, reason="Invalid or missing token")
        return

    await websocket.accept()
    logger.info("ws/render: client connected")

    try:
        while True:
            data = await websocket.receive_json()
            await _ws_dispatch(websocket, ctx, data)
    except WebSocketDisconnect:
        logger.info("ws/render: client disconnected")
    except Exception as exc:
        logger.error(f"ws/render: unexpected error: {exc}")
        await websocket.close(code=1011, reason="Internal error")


# -- Admin: config read/write, status, restart (biopb/biopb#237) ------------


@_router.get("/api/config")
async def get_config(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    if not ctx.config_path:
        raise HTTPException(status_code=404, detail="This server has no config path")
    from pathlib import Path

    from biopb_tensor_server.core.config import _read_config_file, redact_config_secrets
    from biopb_tensor_server.core.config_schema import build_config_schema

    p = Path(ctx.config_path)
    raw: Dict[str, Any] = {}
    if p.exists():
        try:
            raw = _read_config_file(p)
        except ValueError as e:
            raise HTTPException(
                status_code=500, detail=f"Config on disk is unreadable: {e}"
            )
    # Mask credential secrets so they never reach the browser; the PUT route
    # restores them from disk (biopb/biopb#237).
    return JSONResponse(
        {
            "path": str(p),
            "config": redact_config_secrets(raw),
            "schema": build_config_schema(),
        }
    )


@_router.put("/api/config")
async def put_config(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    _require_same_origin(request)
    if not ctx.config_path:
        raise HTTPException(status_code=404, detail="This server has no config path")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body is not valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Config body must be a JSON object")

    from pathlib import Path

    from jsonschema import Draft202012Validator

    from biopb_tensor_server.core.config import (
        _read_config_file,
        restore_redacted_secrets,
        save_config,
        validate_config_dict,
    )
    from biopb_tensor_server.core.config_schema import build_config_schema

    # The form round-trips redacted secrets back as a sentinel; resolve those
    # from the on-disk config so a save never clobbers a real credential with
    # the mask (biopb/biopb#237).
    existing: Dict[str, Any] = {}
    cfg_file = Path(ctx.config_path)
    if cfg_file.exists():
        try:
            existing = _read_config_file(cfg_file)
        except ValueError:
            existing = {}
    body = restore_redacted_secrets(body, existing)

    validator = Draft202012Validator(build_config_schema())
    errors = [
        {"path": [str(x) for x in e.absolute_path], "message": e.message}
        for e in validator.iter_errors(body)
    ]
    # The published JSON Schema deliberately can't express the case-insensitive
    # enums (log_level / reduction_method), so also run the server's real
    # load-time validation and add any problem the schema did not already flag
    # (deduped by path). This keeps "the form accepted it" == "the server will
    # load it" -- one rule set gates both surfaces. See biopb/biopb#34.
    schema_paths = {tuple(e["path"]) for e in errors}
    for problem in validate_config_dict(body):
        path = [str(x) for x in problem["path"]]
        # A root-level ([]) problem is validate_config_dict's structural-failure
        # fallback (parse_config could not build the config). The JSON Schema is
        # the structural layer, so when it already reported errors its precise
        # per-field paths supersede this catch-all -- skip it to avoid a
        # redundant root error. Keep it only when the schema found nothing (the
        # rare schema-valid-but-unparseable body).
        if not path and errors:
            continue
        if tuple(path) not in schema_paths:
            errors.append({"path": path, "message": problem["message"]})
    if errors:
        errors.sort(key=lambda d: d["path"])
        return JSONResponse(
            status_code=422,
            content={"detail": "Config failed validation", "errors": errors},
        )
    try:
        written = save_config(body, Path(ctx.config_path))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not write config: {e}")
    return JSONResponse({"saved": True, "restart_required": True, "path": str(written)})


@_router.get("/api/admin/status")
async def admin_status(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    health: Optional[Dict[str, Any]] = None
    try:
        health = ctx.get_client().health_check()
    except Exception as e:
        logger.warning("admin status: backend health check failed: %s", e)
    running = bool(health and health.get("status") == "SERVING")

    def _h(key: str) -> Any:
        return health.get(key) if health else None

    return JSONResponse(
        {
            "running": running,
            "pid": os.getpid(),
            "version": _VERSION,
            # Control-owned: the admin UI must route a restart through the
            # control, not the sidecar self-restart (biopb/biopb#418).
            "supervised": ctx.supervised,
            # Local mode ⇔ no token enforced (every listener loopback-bound, one
            # machine). It is the single two-mode signal (biopb/biopb#447): the
            # admin UI keys the local-only server-side file chooser (#244) off it,
            # since in local mode the server's filesystem *is* the user's own box.
            "local": ctx.token is None,
            "config_path": str(ctx.config_path) if ctx.config_path else None,
            "health": _h("status"),
            "source_count": _h("source_count"),
            "writable": _h("writable"),
            "uptime_seconds": _h("uptime_seconds"),
            "full_scan_in_progress": _h("full_scan_in_progress"),
            "last_full_scan_finished_at": _h("last_full_scan_finished_at"),
        }
    )


@_router.get("/api/admin/browse")
async def admin_browse(request: Request) -> JSONResponse:
    """List a directory on the server's filesystem for the Sources file chooser.

    Local-mode only (biopb/biopb#244): a browsable FS listing is an
    info-disclosure surface, so it is served **only** when no token is enforced
    — the two-mode signal (biopb/biopb#447) for a loopback-bound, single-machine
    deployment where the server's filesystem *is* the user's own box. In remote
    mode it 404s (feature absent), matching how the admin UI hides the "Browse…"
    button unless ``/api/admin/status`` reports ``local``.

    Returns ``{path, parent, entries: [{name, is_dir}], truncated}``. No path (or
    a blank one) starts at the server user's home directory; a path that is a file
    resolves to its containing directory so the chooser can navigate from it. One
    unreadable entry never fails the whole listing.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)  # no-op in local mode; guards a misconfigured caller
    if ctx.token is not None:
        # Remote mode: never expose the server's filesystem to a remote browser.
        raise HTTPException(
            status_code=404, detail="File browsing is available only in local mode"
        )

    from pathlib import Path

    raw = request.query_params.get("path") or ""
    try:
        base = (Path(raw).expanduser() if raw else Path.home()).resolve()
    except (OSError, RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Bad path: {e}")

    # A file selection resolves to its parent so the chooser keeps navigating.
    try:
        directory = base if base.is_dir() else base.parent
        if not directory.is_dir():
            raise HTTPException(status_code=404, detail="Not a directory")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot access path: {e}")

    entries: List[Dict[str, Any]] = []
    truncated = False
    try:
        with os.scandir(directory) as it:
            for de in it:
                try:
                    is_dir = de.is_dir(follow_symlinks=True)
                except OSError:
                    is_dir = False  # broken symlink / race: list it as a file
                entries.append({"name": de.name, "is_dir": is_dir})
                if len(entries) >= _BROWSE_MAX_ENTRIES:
                    truncated = True
                    break
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot list directory: {e}")

    # Directories first, then files; each ordered case-insensitively by name.
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

    parent = str(directory.parent) if directory.parent != directory else None
    return JSONResponse(
        {
            "path": str(directory),
            "parent": parent,
            "entries": entries,
            "truncated": truncated,
        }
    )


@_router.post("/api/admin/restart")
async def admin_restart(request: Request) -> JSONResponse:
    ctx = _sidecar(request)
    ctx.check_token(request)
    _require_same_origin(request)
    # Control-owned plane: the sidecar must NOT self-restart. The control is the
    # sole owner (it spawned + supervises this process); a self-spawned `biopb
    # server restart` would SIGTERM the control's tracked child, the supervisor
    # would respawn its own replacement, and both would race for the gRPC port —
    # if the standalone daemon won, the control would see a port it didn't spawn,
    # mark it a conflict, and stop managing the plane (biopb/biopb#418). Restart
    # is instead a request to the control (POST /api/data_plane/restart), which
    # the admin UI routes to when it sees supervised=True in /api/admin/status.
    if ctx.supervised:
        raise HTTPException(
            status_code=409,
            detail="This data plane is supervised by the biopb control; restart "
            "it via the control (POST /api/data_plane/restart), not the sidecar "
            "self-restart, which would conflict with supervision.",
        )
    # Refuse a second restart while one is already underway (e.g. a
    # double-click): two `biopb server restart` children would race on the
    # PID file and the freed port. This async handler has no await before the
    # latch, so the check-and-set is atomic on the event loop.
    if ctx._restart_state["in_progress"]:
        raise HTTPException(status_code=409, detail="A restart is already in progress")
    ctx._restart_state["in_progress"] = True
    import subprocess

    from biopb.cli import _detach_kwargs

    # Echo this daemon's own launch args so the restart comes back identically
    # (same config / port / host); a bare restart would fall back to defaults and
    # return mismatched (biopb/biopb#237).
    cmd = [sys.executable, "-m", "biopb.cli", "server", "restart"]
    if ctx.config_path:
        cmd += ["--config", str(ctx.config_path)]
    if ctx.web_port is not None:
        cmd += ["--web-port", str(ctx.web_port)]
    if ctx.web_host:
        cmd += ["--web-host", str(ctx.web_host)]

    # The token rides the child's environment (never the visible command
    # line), matching how `biopb server start` hands it to the daemon.
    env = dict(os.environ)
    if ctx.token:
        env["BIOPB_TENSOR_TOKEN"] = ctx.token

    # Detach so the child outlives this dying parent: `restart` SIGTERMs us,
    # waits for the port to free, then relaunches a fresh daemon.
    try:
        subprocess.Popen(cmd, env=env, **_detach_kwargs())
    except Exception as e:
        # Spawn failed: nothing is bouncing, so clear the latch to allow retry.
        ctx._restart_state["in_progress"] = False
        raise HTTPException(status_code=500, detail=f"Could not spawn restart: {e}")
    return JSONResponse(status_code=202, content={"restarting": True})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    flight_location: str = "grpc://localhost:8815",
    token: Optional[str] = None,
    cache_bytes: int = 512 * 1024 * 1024,  # 512MB default (fits ~8 chunks of 64MB)
    cors_origins: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    web_host: Optional[str] = None,
    web_port: Optional[int] = None,
    supervised: Optional[bool] = None,
) -> FastAPI:
    """Create and return the FastAPI application.

    This only *wires* the app: it builds the per-app ``_SidecarContext`` (lazy
    Flight client + diagnostics + auth config), adds CORS, and includes the
    module-level route ``_router``. All request logic lives in the module-level
    handlers, which read their context off ``app.state.sidecar``. The sidecar is
    API-only — the web UI is served by the control front, the single web origin,
    which proxies here for the data API and /ws/render.

    Args:
        flight_location: Arrow Flight server to connect to.
        token: Shared secret token. ``None`` disables auth (local mode, where
            every listener is loopback-bound).
        cache_bytes: Bytes for the in-process chunk cache.
        cors_origins: Allowed CORS origins. Defaults to localhost variants.
        config_path: Path to the config file this daemon was launched with. The
            admin routes read/write it and echo it into the restart command so a
            self-restart comes back identically (biopb/biopb#237).
        web_host: Host this HTTP sidecar was bound to (echoed into restart).
        web_port: Port this HTTP sidecar was bound to (echoed into restart).
        supervised: Whether the biopb control owns/supervises this data plane;
            when it does, the admin self-restart is refused (biopb/biopb#418).
            Defaults to reading ``BIOPB_DATA_PLANE_SUPERVISED`` from the env the
            control set, so a directly-launched ``biopb server start`` is not
            supervised and keeps its self-restart.

    Returns:
        Configured FastAPI application.
    """
    if supervised is None:
        supervised = os.environ.get("BIOPB_DATA_PLANE_SUPERVISED") == "1"
    if cors_origins is None:
        cors_origins = [
            "http://localhost:8814",
            "http://127.0.0.1:8814",
            "http://[::1]:8814",
        ]

    app = FastAPI(title=_SERVICE, version=_VERSION, docs_url=None, redoc_url=None)
    app.state.sidecar = _SidecarContext(
        flight_location=flight_location,
        token=token,
        cache_bytes=cache_bytes,
        config_path=config_path,
        web_host=web_host,
        web_port=web_port,
        supervised=supervised,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Authorization", "X-Biopb-Token", "Content-Type"],
        expose_headers=[
            "X-Shape",
            "X-Dtype",
            "X-Dim-Labels",
            "X-Image-Width",
            "X-Image-Height",
            "X-Percentile-Lo-Value",
            "X-Percentile-Hi-Value",
        ],
    )
    # Note: WebSocket CORS is handled by the browser during the handshake.

    app.include_router(_router)

    return app


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _source_desc_to_dict(desc: Any) -> Dict[str, Any]:
    """Convert a DataSourceDescriptor proto to a JSON-serialisable dict."""
    return {
        "source_id": desc.source_id,
        "source_url": desc.source_url,
        "source_type": desc.source_type,
        "metadata_json": desc.metadata_json or None,
        "tensors": [_tensor_desc_to_dict(t) for t in desc.tensors],
    }


def _tensor_desc_to_dict(td: Any) -> Dict[str, Any]:
    return {
        "array_id": td.array_id,
        "dim_labels": list(td.dim_labels),
        "shape": [int(x) for x in td.shape],
        "chunk_shape": [int(x) for x in td.chunk_shape],
        "dtype": td.dtype,
    }


# ---------------------------------------------------------------------------
# Entrypoint for direct uvicorn launch
# ---------------------------------------------------------------------------


def shutdown_sentinel_path() -> os.PathLike:
    """Path of the shutdown sentinel file `biopb server stop` writes.

    Must stay in sync with the path biopb.cli computes (duplicated there to avoid
    importing heavy server deps just to stop). A single fixed name in the user's
    biopb data dir - NOT keyed by PID: on Windows the process `start` launches
    can differ from the one running launch()/uvicorn (Store-Python/uv shims), so
    the daemon's os.getpid() and the PID file may disagree. There is only ever
    one daemon (the PID file is singular too), so a fixed name is unambiguous.
    """
    from pathlib import Path

    return Path.home() / ".local" / "share" / "biopb" / "tensor-server.stop"


def _install_windows_shutdown_listener(server) -> None:
    """Windows-only: let `biopb server stop` shut the daemon down gracefully.

    The daemon is a windowless background process (CREATE_NO_WINDOW) in its own
    process group, so it has no console to receive a CTRL_BREAK and Win32 named
    objects are awkward across sessions/elevation. So `stop` instead drops a
    small sentinel *file* that this watcher thread polls for; when it appears we
    ask uvicorn to exit (should_exit + force_exit, so an open browser connection
    can't stall shutdown). uvicorn then returns from run(), so launch()'s
    ``finally -> _graceful_shutdown`` runs and the file-cache lock is released.

    A leftover sentinel from a previous run is cleared once, up front, so the
    watch loop can treat any existing sentinel as a live stop request with no
    clock comparison. (The former mtime guard compared the filesystem's mtime
    against a process-clock ``time.time()``; on a filesystem whose mtime
    granularity is coarser than ``time.time()`` a freshly written sentinel could
    round to just below install time and be misread as stale, dropping a real
    stop -- biopb/biopb#345.) No-op off Windows (POSIX uses SIGTERM).
    Best-effort: on any error `stop` force-kills after its timeout.
    """
    if sys.platform != "win32":
        return

    sentinel = shutdown_sentinel_path()
    # Clear a stale leftover exactly once at install, so "fresh vs. leftover"
    # needs no mtime/clock comparison: after this, any sentinel that appears was
    # written by a `stop` racing or following this watcher.
    try:
        os.remove(sentinel)
    except OSError:
        pass

    def _watch() -> None:
        while True:
            try:
                if os.path.exists(sentinel):
                    logger.info("Shutdown sentinel found; requesting graceful exit.")
                    server.should_exit = True
                    server.force_exit = True
                    try:
                        os.remove(sentinel)
                    except OSError:
                        pass
                    return
            except OSError:
                pass
            time.sleep(0.2)

    threading.Thread(target=_watch, name="win-shutdown-listener", daemon=True).start()
    logger.info("Windows shutdown listener installed (sentinel: %s).", sentinel)


def run(
    flight_location: str = "grpc://localhost:8815",
    token: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8816,
    cache_bytes: int = 512 * 1024 * 1024,  # 512MB default (fits ~8 chunks of 64MB)
    cors_origins: Optional[List[str]] = None,
    config_path: Optional[str] = None,
) -> None:
    """Start the HTTP sidecar with uvicorn (blocking)."""
    import uvicorn

    app = create_app(
        flight_location=flight_location,
        token=token,
        cache_bytes=cache_bytes,
        cors_origins=cors_origins,
        config_path=config_path,
        web_host=host,
        web_port=port,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    # Windows: enable graceful `biopb server stop` via a sentinel-file watcher
    # that flips server.should_exit (no-op on other platforms, which use SIGTERM).
    _install_windows_shutdown_listener(server)
    server.run()
