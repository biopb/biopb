"""FastAPI HTTP sidecar for TensorFlight server.

Exposes the TensorFlightClient over a browser-friendly HTTP/JSON + binary API.

Endpoints:
  GET  /livez                        — liveness probe (no auth)
  GET  /readyz                       — readiness probe (no auth)
  GET  /healthz                      — alias for /readyz (no auth)
  GET  /api/diagnostics              — runtime diagnostics (token required)
  GET  /api/sources                  — list DataSourceDescriptors (token required)
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
import json
import math
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from biopb.tensor.descriptor_pb2 import SliceHint, TensorReadOptions
from biopb.tensor.ticket_pb2 import TensorTicket, ChunkBounds
from biopb.tensor.client import TensorFlightClient


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
            self._errors.append(_ErrorEvent(
                timestamp=_now_rfc3339(),
                code=code,
                message=_redact(message),
            ))

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(
    flight_location: str = "grpc://localhost:8815",
    token: Optional[str] = None,
    dev_mode: bool = False,
    cache_bytes: int = 1_000_000_000,
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        flight_location: Arrow Flight server to connect to.
        token: Shared secret token. ``None`` disables auth (only in dev_mode).
        dev_mode: Whether the website token bypass is active.
        cache_bytes: Bytes for the in-process chunk cache on the Python side.
        cors_origins: Allowed CORS origins. Defaults to localhost variants.

    Returns:
        Configured FastAPI application.
    """
    if cors_origins is None:
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://[::1]:5173",
        ]

    # Lazy-init Flight client (first request will connect)
    _client_lock = threading.Lock()
    _client_holder: Dict[str, Optional[TensorFlightClient]] = {"client": None}

    diag = _DiagnosticsState()

    def _get_client() -> TensorFlightClient:
        with _client_lock:
            if _client_holder["client"] is None:
                try:
                    _client_holder["client"] = TensorFlightClient(
                        location=flight_location,
                        cache_bytes=cache_bytes,
                        token=token,
                    )
                    diag.mark_connected()
                except Exception as exc:
                    diag.mark_error("CONNECTION_FAILED", str(exc))
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
        expose_headers=["X-Shape", "X-Dtype", "X-Dim-Labels"],
    )

    # -----------------------------------------------------------------------
    # Auth helper
    # -----------------------------------------------------------------------

    def _check_token(request: Request) -> None:
        """Raise 401 if the request does not carry a valid token."""
        if dev_mode or token is None:
            return  # bypass in dev mode
        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            provided = bearer[len("Bearer "):]
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
        ready = diag.connection_state in ("connected", "disconnected")  # not "error"
        return JSONResponse({
            "status": "ok" if ready else "degraded",
            "timestamp": _now_rfc3339(),
            "ready": ready,
            "dev_mode": dev_mode,
            "service": _SERVICE,
            "version": _VERSION,
        })

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
        session_id = request.headers.get("X-Biopb-Token", "") or \
            request.headers.get("Authorization", "")
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
            diag.latency.record((time.monotonic() - t0) * 1000)
            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as exc:
            diag.mark_error("LIST_SOURCES_FAILED", str(exc))
            raise HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")

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
            raise HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")

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

            # Build response headers
            headers = {
                "X-Shape": ",".join(str(d) for d in arr.shape),
                "X-Dtype": str(arr.dtype),
                "X-Chunk-Start": "",  # Not available from do_get alone
                "X-Chunk-Stop": "",   # Not available from do_get alone
            }

            return Response(
                content=arr.tobytes(),
                media_type="application/octet-stream",
                headers=headers,
            )

        except ValueError as exc:
            # Invalid hex string or protobuf parse error
            raise HTTPException(status_code=400, detail=f"Invalid ticket: {exc}")
        except flight.FlightError as exc:
            diag.mark_error("CHUNK_FETCH_FAILED", str(exc))
            raise HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")
        except Exception as exc:
            diag.mark_error("CHUNK_FAILED", str(exc))
            raise HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")

    @app.get("/api/sources/{source_id:path}")
    async def get_source(source_id: str, request: Request) -> JSONResponse:
        _check_token(request)
        t0 = time.monotonic()
        try:
            client = _get_client()
            sources = client.list_sources()
            if source_id not in sources:
                raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
            diag.latency.record((time.monotonic() - t0) * 1000)
            return JSONResponse(_source_desc_to_dict(sources[source_id]))
        except HTTPException:
            raise
        except Exception as exc:
            diag.mark_error("GET_SOURCE_FAILED", str(exc))
            raise HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")

    # -----------------------------------------------------------------------
    # Slice (binary response)
    # -----------------------------------------------------------------------

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

            # Get lazy dask array WITHOUT slice_hint (client-side cropping for correct size at scale)
            arr_lazy = client.get_tensor(
                source_id=req.source_id,
                tensor_id=req.tensor_id,
                slice_hint=None,
                scale_hint=scale_hint,
                reduction_method=reduction_method,
            )

            # Apply slice on dask array BEFORE compute (lazy slicing)
            # This ensures proper cropping when scale_hint is used
            if slice_hint is not None:
                arr_lazy = arr_lazy[slice_hint]

            # Compute (blocking)
            arr: np.ndarray = arr_lazy.compute()

            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

            # Ensure C-contiguous layout for predictable byte order on the client
            arr = np.ascontiguousarray(arr)

            elapsed = (time.monotonic() - t0) * 1000
            diag.latency.record(elapsed)

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
            raise HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")

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
    cache_bytes: int = 1_000_000_000,
    cors_origins: Optional[List[str]] = None,
) -> None:
    """Start the HTTP sidecar with uvicorn (blocking)."""
    import uvicorn

    app = create_app(
        flight_location=flight_location,
        token=token,
        dev_mode=dev_mode,
        cache_bytes=cache_bytes,
        cors_origins=cors_origins,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
