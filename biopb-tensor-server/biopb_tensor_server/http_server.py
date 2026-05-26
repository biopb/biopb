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

import collections
import logging
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.flight as flight
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.ticket_pb2 import TensorTicket
from fastapi import (
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
from starlette.staticfiles import StaticFiles

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


def _dtype_to_numpy(dtype_str: str) -> np.dtype:
    """Convert numpy-style dtype string to np.dtype (handles common aliases)."""
    return np.dtype(dtype_str)


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


def create_app(
    flight_location: str = "grpc://localhost:8815",
    token: Optional[str] = None,
    dev_mode: bool = False,
    cache_bytes: int = 512 * 1024 * 1024,  # 512MB default (fits ~8 chunks of 64MB)
    cors_origins: Optional[List[str]] = None,
    static_dir: Optional[str] = None,  # Directory for static webapp files (None = API only)
) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        flight_location: Arrow Flight server to connect to.
        token: Shared secret token. ``None`` disables auth (only in dev_mode).
        dev_mode: Whether the website token bypass is active.
        cache_bytes: Bytes for the in-process chunk cache (default 0, disabled
            since sidecar runs on same machine as gRPC server which already caches).
        cors_origins: Allowed CORS origins. Defaults to localhost variants.

    Returns:
        Configured FastAPI application.
    """
    if cors_origins is None:
        cors_origins = [
            "http://localhost:8814",
            "http://127.0.0.1:8814",
            "http://[::1]:8814",
        ]

    # Lazy-init Flight client (first request will connect)
    _client_lock = threading.Lock()
    _client_holder: Dict[str, Optional[TensorFlightClient]] = {"client": None}

    diag = _DiagnosticsState()

    def _get_client() -> TensorFlightClient:
        with _client_lock:
            if _client_holder["client"] is None:
                try:
                    logger.debug(f"Connecting to Flight server at {flight_location}")
                    _client_holder["client"] = TensorFlightClient(
                        location=flight_location,
                        cache_bytes=cache_bytes,
                        token=token,
                    )
                    diag.mark_connected()
                    logger.info(f"Connected to Flight server at {flight_location}")
                except Exception as exc:
                    diag.mark_error("CONNECTION_FAILED", str(exc))
                    logger.error(f"Failed to connect to Flight server: {exc}")
                    raise
            return _client_holder["client"]

    # -----------------------------------------------------------------------
    # App
    # -----------------------------------------------------------------------

    app = FastAPI(title=_SERVICE, version=_VERSION, docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-Biopb-Token", "Content-Type"],
        expose_headers=[
            "X-Shape", "X-Dtype", "X-Dim-Labels",
            "X-Image-Width", "X-Image-Height",
            "X-Percentile-Lo-Value", "X-Percentile-Hi-Value",
        ],
    )

    # Note: WebSocket CORS is handled by the browser during the handshake.
    # The headers above are for HTTP requests only. WebSocket connections
    # use the same origin validation via the Sec-WebSocket-Origin header.

    # -----------------------------------------------------------------------
    # Auth helper
    # -----------------------------------------------------------------------

    def _check_token(request: Request) -> None:
        """Raise 401 if the request does not carry a valid token."""
        if dev_mode or token is None:
            return  # bypass in dev mode
        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            provided = bearer[len("Bearer ") :]
        else:
            provided = request.headers.get("X-Biopb-Token", "")
        if not secrets.compare_digest(provided.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing token")

    # -----------------------------------------------------------------------
    # Health endpoints (unauthenticated)
    # -----------------------------------------------------------------------

    @app.get("/livez")
    async def livez() -> JSONResponse:
        return JSONResponse({"status": "ok", "timestamp": _now_rfc3339()})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        backend_health = None
        with _client_lock:
            client = _client_holder["client"]

        if client is not None:
            try:
                backend_health = client.health_check()
            except Exception as e:
                logger.warning(f"Backend health check failed: {e}")

        ready = (
            backend_health and backend_health.get("status") == "SERVING"
        ) or diag.connection_state == "connected"

        return JSONResponse(
            {
                "status": "ok" if ready else "degraded",
                "timestamp": _now_rfc3339(),
                "ready": ready,
                "dev_mode": dev_mode,
                "service": _SERVICE,
                "version": _VERSION,
                "backend_health": backend_health,
                "source_count": backend_health.get("source_count", 0)
                if backend_health
                else 0,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return await readyz()

    # -----------------------------------------------------------------------
    # Diagnostics (token required)
    # -----------------------------------------------------------------------

    @app.get("/api/diagnostics")
    async def diagnostics(request: Request) -> JSONResponse:
        _check_token(request)
        # Soft rate limit per session (identify by token itself as session key)
        session_id = request.headers.get("X-Biopb-Token", "") or request.headers.get(
            "Authorization", ""
        )
        if not diag.check_rate_limit(session_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded (1 req/s)")
        # Sync cache stats from Flight client if available
        with _client_lock:
            client = _client_holder["client"]
        if client is not None:
            info = client.cache_info()
            diag.cache_hits = info.get("hits", diag.cache_hits)
            diag.cache_misses = info.get("misses", diag.cache_misses)
        return JSONResponse(diag.snapshot(dev_mode=dev_mode))

    # -----------------------------------------------------------------------
    # Sources
    # -----------------------------------------------------------------------

    @app.get("/api/sources")
    async def list_sources(request: Request) -> JSONResponse:
        _check_token(request)
        t0 = time.monotonic()
        try:
            client = _get_client()
            sources = client.list_sources()
            result = []
            for source_id, desc in sources.items():
                result.append(_source_desc_to_dict(desc))
            elapsed = (time.monotonic() - t0) * 1000
            diag.latency.record(elapsed)
            logger.debug(
                f"list_sources: returned {len(result)} sources in {elapsed:.1f}ms"
            )
            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as exc:
            diag.mark_error("LIST_SOURCES_FAILED", str(exc))
            logger.error(f"list_sources failed: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )

    @app.post("/api/sources/query")
    async def query_sources(req: QuerySourcesRequest, request: Request) -> Response:
        """Execute SQL query against source metadata database.

        Request body: {"sql": "SELECT source_id, source_type FROM sources WHERE ..."}
        Response headers:
          X-Total-Sources    — total matching (before truncation)
          X-Returned-Sources — actual rows returned
          X-Truncated        — "true" if truncated
        Response body: JSON array of query results
        """
        _check_token(request)
        t0 = time.monotonic()

        try:
            client = _get_client()
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
            diag.latency.record(elapsed)
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
            diag.mark_error("QUERY_SOURCES_FAILED", str(exc))
            logger.error(f"query_sources failed: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )

    # NOTE: the /metadata and /ticket routes must be registered before the greedy
    # {source_id:path} route, otherwise Starlette's first-match routing
    # would shadow them.
    @app.get("/api/sources/{source_id:path}/metadata")
    async def get_source_metadata(source_id: str, request: Request) -> JSONResponse:
        _check_token(request)
        t0 = time.monotonic()
        try:
            client = _get_client()
            metadata = client.get_source_metadata(source_id)
            diag.latency.record((time.monotonic() - t0) * 1000)
            return JSONResponse(metadata)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            diag.mark_error("GET_METADATA_FAILED", str(exc))
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )

    # -----------------------------------------------------------------------
    # Chunk (binary response via ticket)
    # -----------------------------------------------------------------------

    @app.get("/api/sources/{source_id:path}/ticket/{ticket_hex}")
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
        _check_token(request)
        t0 = time.monotonic()

        logger.debug(f"get_chunk: source={source_id}, ticket_hex={ticket_hex[:16]}...")

        try:
            # Decode hex string to bytes
            ticket_bytes = bytes.fromhex(ticket_hex)

            # Parse TensorTicket to validate
            tensor_ticket = TensorTicket.FromString(ticket_bytes)

            # Get Flight client
            client = _get_client()

            # Fetch chunk data via do_get
            reader = client._client.do_get(
                flight.Ticket(ticket_bytes),
                options=client._call_options,
            )

            # Read all data from the stream
            table = reader.read_all()

            # Convert to numpy
            arr = table.column(0).to_numpy()

            # Ensure C-contiguous layout
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
            arr = np.ascontiguousarray(arr)

            elapsed = (time.monotonic() - t0) * 1000
            diag.latency.record(elapsed)
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
            diag.mark_error("CHUNK_FETCH_FAILED", str(exc))
            logger.error(f"get_chunk: Flight error: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )
        except Exception as exc:
            diag.mark_error("CHUNK_FAILED", str(exc))
            logger.error(f"get_chunk: unexpected error: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )

    @app.get("/api/sources/{source_id:path}")
    async def get_source(source_id: str, request: Request) -> JSONResponse:
        _check_token(request)
        t0 = time.monotonic()
        try:
            client = _get_client()
            sources = client.list_sources()
            if source_id not in sources:
                raise HTTPException(
                    status_code=404, detail=f"Source not found: {source_id}"
                )
            diag.latency.record((time.monotonic() - t0) * 1000)
            return JSONResponse(_source_desc_to_dict(sources[source_id]))
        except HTTPException:
            raise
        except Exception as exc:
            diag.mark_error("GET_SOURCE_FAILED", str(exc))
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )

    # -----------------------------------------------------------------------
    # Slice (binary response)
    # -----------------------------------------------------------------------

    def _get_slice(
        client: TensorFlightClient,
        req: SliceRequest,
    ) -> np.ndarray:
        """Helper to fetch a slice of a tensor as a numpy array."""
        slice_hint: Optional[Tuple[slice, ...]] = None
        if req.slice_start is not None and req.slice_stop is not None:
            if len(req.slice_start) != len(req.slice_stop):
                raise HTTPException(
                    status_code=422,
                    detail="slice_start and slice_stop must have the same length",
                )
            slice_hint = tuple(
                slice(s, e) for s, e in zip(req.slice_start, req.slice_stop)
            )

        # Build read_options
        scale_hint = req.scale_hint or None
        reduction_method = req.reduction_method or None

        # Pass slice_hint to gRPC for optimized slicing (in world coordinates)
        # slice_hint is applied BEFORE scaling, so coordinates are in original tensor units
        arr_lazy = client.get_tensor(
            source_id=req.source_id,
            tensor_id=req.tensor_id,
            slice_hint=slice_hint,
            scale_hint=scale_hint,
            reduction_method=reduction_method,
        )

        arr: np.ndarray = arr_lazy.compute()

        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

        # Ensure C-contiguous layout for predictable byte order on the client
        arr = np.ascontiguousarray(arr)

        return arr


    @app.post("/api/slice")
    async def slice_tensor(req: SliceRequest, request: Request) -> Response:
        """Fetch a slice of a tensor and return raw bytes.

        Response headers:
          X-Shape     — comma-separated dimensions of the returned array
          X-Dtype     — numpy dtype string (e.g. "uint16", "float32")
          X-Dim-Labels — comma-separated semantic axis labels

        Response body:
          C-contiguous raw bytes of the numpy array (no framing).
        """
        _check_token(request)
        t0 = time.monotonic()

        logger.debug(
            f"slice: source={req.source_id}, tensor={req.tensor_id}, "
            f"slice={req.slice_start}-{req.slice_stop}, scale={req.scale_hint}, method={req.reduction_method}"
        )

        if req.pixel_budget is not None:
            diag.pixel_budget = req.pixel_budget

        try:
            client = _get_client()

            # Build slice_hint
            slice_hint: Optional[Tuple[slice, ...]] = None
            if req.slice_start is not None and req.slice_stop is not None:
                if len(req.slice_start) != len(req.slice_stop):
                    raise HTTPException(
                        status_code=422,
                        detail="slice_start and slice_stop must have the same length",
                    )
                slice_hint = tuple(
                    slice(s, e) for s, e in zip(req.slice_start, req.slice_stop)
                )

            # Build read_options
            scale_hint = req.scale_hint or None
            reduction_method = req.reduction_method or None

            # Pass slice_hint to gRPC for optimized slicing (in world coordinates)
            # slice_hint is applied BEFORE scaling, so coordinates are in original tensor units
            arr_lazy = client.get_tensor(
                source_id=req.source_id,
                tensor_id=req.tensor_id,
                slice_hint=slice_hint,
                scale_hint=scale_hint,
                reduction_method=reduction_method,
            )

            # Compute (blocking)
            arr: np.ndarray = arr_lazy.compute()

            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

            # Ensure C-contiguous layout for predictable byte order on the client
            arr = np.ascontiguousarray(arr)

            elapsed = (time.monotonic() - t0) * 1000
            diag.latency.record(elapsed)
            logger.debug(
                f"slice: computed shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes}B in {elapsed:.1f}ms"
            )

            # Build response headers
            headers = {
                "X-Shape": ",".join(str(d) for d in arr.shape),
                "X-Dtype": str(arr.dtype),
                "X-Dim-Labels": "",  # filled below if we have a descriptor
            }

            # Try to attach dim labels from the cached descriptor
            try:
                sources = client._sources  # type: ignore[attr-defined]
                if req.source_id in sources:
                    for td in sources[req.source_id].tensors:
                        if td.array_id == req.tensor_id:
                            headers["X-Dim-Labels"] = ",".join(td.dim_labels)
                            break
            except Exception:
                pass

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
            diag.mark_error("SLICE_FAILED", str(exc))
            logger.error(f"slice failed: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Flight error: {type(exc).__name__}"
            )

    # -----------------------------------------------------------------------
    # Render (image output)
    # -----------------------------------------------------------------------

    @app.post("/api/render")
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
        _check_token(request)
        t0 = time.monotonic()

        logger.debug(
            f"render: source={req.source_id}, tensor={req.tensor_id}, "
            f"slice={req.slice_start}-{req.slice_stop}, scale={req.scale_hint}, "
            f"percentiles={req.percentile_lo}-{req.percentile_hi}, "
            f"color={req.color}, format={req.output_format}"
        )

        if req.pixel_budget is not None:
            diag.pixel_budget = req.pixel_budget

        try:
            client = _get_client()

            # Build slice_hint (same logic as slice endpoint)
            slice_hint: Optional[Tuple[slice, ...]] = None
            if req.slice_start is not None and req.slice_stop is not None:
                if len(req.slice_start) != len(req.slice_stop):
                    raise HTTPException(
                        status_code=422,
                        detail="slice_start and slice_stop must have the same length",
                    )
                slice_hint = tuple(
                    slice(s, e) for s, e in zip(req.slice_start, req.slice_stop)
                )

            # Get tensor
            scale_hint = req.scale_hint or None
            reduction_method = req.reduction_method or None

            arr_lazy = client.get_tensor(
                source_id=req.source_id,
                tensor_id=req.tensor_id,
                slice_hint=slice_hint,
                scale_hint=scale_hint,
                reduction_method=reduction_method,
            )

            # Compute (blocking)
            t0_compute = time.monotonic()
            arr: np.ndarray = arr_lazy.compute()
            compute_ms = (time.monotonic() - t0_compute) * 1000

            # Get dim_labels from descriptor
            dim_labels: List[str] = []
            try:
                sources = client._sources  # type: ignore[attr-defined]
                if req.source_id in sources:
                    for td in sources[req.source_id].tensors:
                        if td.array_id == req.tensor_id:
                            dim_labels = list(td.dim_labels)
                            break
            except Exception:
                pass

            # Use shape-based fallback if dim_labels not found
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
            diag.latency.record(elapsed)
            logger.debug(
                f"render: image size={width}x{height}, "
                f"bytes={len(image_bytes)}, total={elapsed:.1f}ms, "
                f"compute={compute_ms:.1f}ms, render={render_ms:.1f}ms"
            )

            # Build response
            format_lower = req.output_format.lower()
            if format_lower == "raw":
                media_type = "application/octet-stream"  # Raw RGBA bytes
            elif format_lower == "png":
                media_type = "image/png"
            else:
                media_type = "image/jpeg"
            headers = {
                "X-Image-Width": str(width),
                "X-Image-Height": str(height),
                "X-Percentile-Lo-Value": str(lo_val),
                "X-Percentile-Hi-Value": str(hi_val),
                "X-Image-Format": format_lower,  # Tell client what format was used
            }

            return Response(
                content=image_bytes,
                media_type=media_type,
                headers=headers,
            )

        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Rendering not available: {exc}"
            )
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            diag.mark_error("RENDER_FAILED", str(exc))
            logger.error(f"render failed: {exc}\n{tb}")
            raise HTTPException(
                status_code=502, detail=f"Render error: {type(exc).__name__}: {exc}"
            )

    # -----------------------------------------------------------------------
    # WebSocket render endpoint
    # -----------------------------------------------------------------------

    @app.websocket("/ws/render")
    async def websocket_render(websocket: WebSocket) -> None:
        """WebSocket endpoint for rendering tensor slices.

        Protocol:
          1. Client connects, sends nothing
          2. Server validates token from headers or query params
          3. Client sends JSON: { action: "render", params: RenderRequest }
          4. Server sends JSON metadata: { action: "render_start", width, height, format }
          5. Server sends binary: JPEG/PNG image bytes
          6. Repeat steps 3-5 for subsequent requests

        No session state - WebSocket is purely request/response.

        Token validation: Accept token from Authorization header, X-Biopb-Token header,
        or query parameter "token" (for browsers that can't send custom headers).
        """
        # Validate token from headers or query params
        if not dev_mode and token is not None:
            # Check headers first (Authorization or X-Biopb-Token)
            bearer = websocket.headers.get("Authorization", "")
            if bearer.startswith("Bearer "):
                provided = bearer[len("Bearer ") :]
            else:
                provided = websocket.headers.get("X-Biopb-Token", "")

            # If no header token, check query parameter (for browser WebSocket)
            if not provided:
                provided = websocket.query_params.get("token", "")

            if not secrets.compare_digest(provided.encode(), token.encode()):
                await websocket.close(code=4001, reason="Invalid or missing token")
                return

        await websocket.accept()

        try:
            while True:
                # Receive JSON request
                data = await websocket.receive_json()

                # Validate action
                action = data.get("action")
                if action != "render":
                    await websocket.send_json({
                        "action": "error",
                        "message": f"Unknown action: {action}",
                    })
                    continue

                # Parse render params
                try:
                    params = RenderRequest(**data.get("params", {}))
                except Exception as e:
                    await websocket.send_json({
                        "action": "error",
                        "message": f"Invalid params: {e}",
                    })
                    continue

                t0 = time.monotonic()

                logger.debug(
                    f"ws/render: source={params.source_id}, tensor={params.tensor_id}, "
                    f"slice={params.slice_start}-{params.slice_stop}, scale={params.scale_hint}"
                )

                if params.pixel_budget is not None:
                    diag.pixel_budget = params.pixel_budget

                try:
                    client = _get_client()

                    # Build slice_hint (same logic as HTTP render endpoint)
                    slice_hint: Optional[Tuple[slice, ...]] = None
                    if params.slice_start is not None and params.slice_stop is not None:
                        if len(params.slice_start) != len(params.slice_stop):
                            await websocket.send_json({
                                "action": "error",
                                "message": "slice_start and slice_stop must have the same length",
                            })
                            continue
                        slice_hint = tuple(
                            slice(s, e) for s, e in zip(params.slice_start, params.slice_stop)
                        )

                    scale_hint = params.scale_hint or None
                    reduction_method = params.reduction_method or None

                    # Get tensor context (includes realized slice bounds)
                    ctx = client._get_tensor_context(
                        source_id=params.source_id,
                        tensor_id=params.tensor_id,
                        slice_hint=slice_hint,
                        scale_hint=scale_hint,
                        reduction_method=reduction_method,
                    )

                    # Build dask array from context (uncropped)
                    chunks = [ep[0] for ep in ctx.endpoints]
                    chunk_bounds_list = [ep[1] for ep in ctx.endpoints]
                    dask_arr = client._build_dask_array(
                        desc=ctx.descriptor,
                        chunks=chunks,
                        chunk_bounds=chunk_bounds_list,
                    )

                    # Get dim_labels from descriptor
                    dim_labels: List[str] = list(ctx.descriptor.dim_labels)
                    if not dim_labels:
                        dim_labels = [f"d{i}" for i in range(arr.ndim)]

                    # Build axis map to find Y and X indices
                    y_idx = dim_labels.index("y") if "y" in dim_labels else len(dim_labels) - 2
                    x_idx = dim_labels.index("x") if "x" in dim_labels else len(dim_labels) - 1

                    # Slice dask array to the originally requested bounds (except y/x) before computing.
                    if ctx.original_slice_hint is not None and ctx.descriptor.HasField("slice_hint"):
                        realized = ctx.descriptor.slice_hint
                        ndim = len(ctx.descriptor.shape)
                        scale = list(ctx.read_opt.scale_hint) if ctx.read_opt.scale_hint else None
                        crop = []
                        for ax in range(ndim):
                            if ax in (y_idx, x_idx):
                                crop.append(slice(None))  # keep full range for y/x
                            else:
                                req_start = int(ctx.original_slice_hint.start[ax])
                                req_stop = int(ctx.original_slice_hint.stop[ax])
                                ret_start = int(realized.start[ax])
                                s = int(scale[ax]) if scale and ax < len(scale) else 1
                                logical_start = (req_start - ret_start) // s
                                logical_stop = (req_stop - ret_start + s - 1) // s
                                crop.append(slice(logical_start, logical_stop))
                        dask_arr = dask_arr[tuple(crop)]

                    t0_compute = time.monotonic()
                    arr: np.ndarray = dask_arr.compute()
                    compute_ms = (time.monotonic() - t0_compute) * 1000

                    # Compute loaded region from realized slice bounds (not requested)
                    loaded_region = None
                    if ctx.descriptor.HasField("slice_hint"):
                        realized = ctx.descriptor.slice_hint
                        loaded_region = {
                            "x": int(realized.start[x_idx]),
                            "y": int(realized.start[y_idx]),
                            "width": int(realized.stop[x_idx] - realized.start[x_idx]),
                            "height": int(realized.stop[y_idx] - realized.start[y_idx]),
                            "scale_factors": list(ctx.descriptor.scale_hint) if ctx.descriptor.scale_hint else [1] * len(dim_labels),
                        }

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

                    elapsed = (time.monotonic() - t0) * 1000
                    diag.latency.record(elapsed)
                    logger.debug(
                        f"ws/render: image size={width}x{height}, "
                        f"bytes={len(image_bytes)}, total={elapsed:.1f}ms, "
                        f"compute={compute_ms:.1f}ms, render={render_ms:.1f}ms"
                    )

                    # Send metadata JSON first
                    format_lower = params.output_format.lower()
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

                    # Send binary image data
                    await websocket.send_bytes(image_bytes)

                except ValueError as exc:
                    await websocket.send_json({
                        "action": "error",
                        "message": str(exc),
                    })
                except ImportError as exc:
                    await websocket.send_json({
                        "action": "error",
                        "message": f"Rendering not available: {exc}",
                    })
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    diag.mark_error("WS_RENDER_FAILED", str(exc))
                    logger.error(f"ws/render failed: {exc}\n{tb}")
                    await websocket.send_json({
                        "action": "error",
                        "message": f"Render error: {type(exc).__name__}",
                    })

        except WebSocketDisconnect:
            logger.debug("ws/render: client disconnected")
        except Exception as exc:
            logger.error(f"ws/render: unexpected error: {exc}")
            await websocket.close(code=1011, reason="Internal error")

    # -----------------------------------------------------------------------
    # Static files (optional)
    # -----------------------------------------------------------------------

    if static_dir:
        from pathlib import Path as _Path
        static_path = _Path(static_dir)
        if static_path.is_dir():
            # SPA fallback middleware - serve index.html for non-API routes
            @app.middleware("http")
            async def spa_fallback(request: Request, call_next):
                response = await call_next(request)
                # Only intercept 404s for non-API, non-health routes
                if response.status_code == 404:
                    path = request.url.path
                    if not path.startswith("/api") and not path.startswith("/live") and not path.startswith("/ready") and not path.startswith("/health"):
                        index_file = static_path / "index.html"
                        if index_file.exists():
                            return Response(
                                content=index_file.read_bytes(),
                                media_type="text/html",
                            )
                return response

            # Mount static files at root (must be after all API routes)
            app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

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


def run(
    flight_location: str = "grpc://localhost:8815",
    token: Optional[str] = None,
    dev_mode: bool = False,
    host: str = "127.0.0.1",
    port: int = 8816,
    cache_bytes: int = 512 * 1024 * 1024,  # 512MB default (fits ~8 chunks of 64MB)
    cors_origins: Optional[List[str]] = None,
    static_dir: Optional[str] = None,  # Directory for static webapp files
) -> None:
    """Start the HTTP sidecar with uvicorn (blocking)."""
    import uvicorn

    app = create_app(
        flight_location=flight_location,
        token=token,
        dev_mode=dev_mode,
        cache_bytes=cache_bytes,
        cors_origins=cors_origins,
        static_dir=static_dir,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
