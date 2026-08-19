"""FastAPI HTTP sidecar for TensorFlight server.

Exposes the TensorFlightClient over a browser-friendly HTTP/JSON + binary API.

Endpoints (unauthenticated — probes):
  GET  /livez                        — liveness probe (never touches the backend)
  GET  /readyz                       — readiness probe; asks Flight, connecting if
                                       needed. 200 when SERVING, 503 otherwise
  GET  /healthz                      — alias for /readyz

Endpoints (token required):
  GET  /api/diagnostics              — runtime diagnostics
  GET  /api/sources                  — list DataSourceDescriptors
  POST /api/sources/query            — SQL query against source metadata
  GET  /api/sources/{source_id}/metadata          — parsed metadata_json
  GET  /api/sources/{source_id}/ticket/{ticket_hex} — resolve a Flight ticket to bytes
  GET  /api/sources/{source_id}      — single DataSourceDescriptor
  GET  /api/tile_info/{source_id}    — tile grid + pyramid levels for a tensor
  GET  /api/tile/{source_id}         — one tile, cacheable (raw | png | jpeg)
  POST /api/slice                    — fetch array slice as binary
  POST /api/render                   — server-rendered RGB image of a slice
  WS   /ws/render                    — streaming render channel
  GET  /api/config                   — current config (secrets redacted)
  PUT  /api/config                   — update config (same-origin guarded)
  GET  /api/admin/status             — server/catalog status for the admin page
  GET  /api/admin/browse             — server-side filesystem browse (data-folder picker)

  The specific /api/sources/{id}/… routes are registered before the greedy
  /{id:path} catch-all so Starlette does not shadow them (see route defs).

Authentication:
  Pass the website token in the ``Authorization: Bearer <token>`` header or
  ``X-Biopb-Token`` header on every protected request.  /livez, /readyz and
  /healthz are always unauthenticated so proxies can probe them.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.flight as flight
from biopb import _web_auth
from biopb.tensor.client import TensorFlightClient, _request_crop_slices
from biopb.tensor.ticket_pb2 import TensorTicket
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
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
        self.cancelled: int = 0
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

    # --- Cancellation helper ---

    def record_cancelled(self) -> None:
        """Count one read abandoned because the client hung up before it ran."""
        with self._lock:
            self.cancelled += 1

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
            cancelled = self.cancelled
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
            "cancelled_reads": cancelled,
            "latency_p50_ms": self.latency.percentile(50),
            "latency_p95_ms": self.latency.percentile(95),
            "last_error_code": last_error.code if last_error else None,
            "last_error_message": last_error.message if last_error else None,
            "metrics_ready": self.latency.metrics_ready,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PATH_LIKE = re.compile(r"(/[^\s]{3,}|[A-Za-z]:\\[^\s]{3,})")
_TOKEN_LIKE = re.compile(r"[A-Za-z0-9_\-]{16,}")


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    gamma: float = 1.0  # exponent on the normalized intensity; 1.0 is linear
    output_format: str = "png"  # "png" or "jpeg"
    pixel_budget: Optional[int] = None


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
        supervised: bool = False,
        tls_ca_pem: Optional[bytes] = None,
    ) -> None:
        self.flight_location = flight_location
        self.token = token
        self.cache_bytes = cache_bytes
        # PEM the flight plane serves, when it serves TLS. We are co-located with
        # that plane and read this off local disk, so it is an explicit trust
        # anchor -- not a trust-on-first-use pin. None for a plaintext plane.
        self.tls_ca_pem = tls_ca_pem
        # The config file this daemon was launched with (read/written by the
        # /api/config endpoints).
        self.config_path = config_path
        # True when the biopb control spawned + supervises this data plane (it
        # sets BIOPB_DATA_PLANE_SUPERVISED in our env). Reported on
        # /api/admin/status so the admin UI routes a restart to the control (which
        # owns the process); a self-managed plane can't be restarted from the
        # browser (biopb/biopb#418).
        self.supervised = supervised
        self.diag = _DiagnosticsState()
        # Lazy-init Flight client (first request will connect)
        self._client_lock = threading.Lock()
        self._client_holder: Dict[str, Optional[TensorFlightClient]] = {"client": None}

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
                        tls_ca_pem=self.tls_ca_pem,
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

    def backend_snapshot(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Flight's health, connecting first if nothing has yet. ``(health, error)``.

        Readiness is a question about the backend, so it has to be answered by
        asking the backend -- which means being willing to open the connection.
        Peeking instead made readiness a function of *traffic*: nothing but the
        token-protected data routes ever called ``get_client()``, so a probe
        reported "not ready" against a perfectly healthy Flight server until an
        unrelated request happened to connect (biopb/biopb#755).

        Exactly one of the two returns is set, so a caller can tell "never
        reached the backend" from "backend answered, but not SERVING" -- a null
        health with no error used to also mean "nobody has asked yet."

        BLOCKING: connect and the health action are both synchronous gRPC. Call
        it off the event loop (the route uses ``run_in_threadpool``), or one
        unreachable backend stalls every other request the sidecar is serving.
        """
        try:
            client = self.get_client()
        except Exception as exc:  # get_client already logged and marked the error
            return None, f"connect failed: {exc}"
        try:
            return client.health_check(), None
        except Exception as exc:
            logger.warning(f"Backend health check failed: {exc}")
            return None, f"health check failed: {exc}"

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
    return tuple(slice(s, e) for s, e in zip(slice_start, slice_stop, strict=True))


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
# Cancellation
# ---------------------------------------------------------------------------


class _ClientGone(Exception):
    """The caller hung up before the read this handler was about to do."""


async def _abort_if_client_gone(request: Request, ctx: _SidecarContext) -> None:
    """Drop the request if the client has already disconnected.

    The read handlers compute on the event loop, so a burst of requests
    serializes: by the time a queued one runs, the browser that issued it may
    have panned away and aborted its ``fetch`` long ago. Checking the ASGI
    receive channel here is what turns a client-side ``AbortController`` into
    actually-skipped backend work -- without it, cancellation only stops the
    *client* from looking at bytes the server already paid to produce.

    The check is a poll, not a guarantee: a client that disconnects *during*
    the compute is not noticed, because neither the Flight read nor the dask
    graph is interruptible. So this reclaims queued work, not in-flight work,
    which is precisely the tile-burst case (biopb/biopb#762 covers making the
    queue shorter in the first place).
    """
    if await request.is_disconnected():
        ctx.diag.record_cancelled()
        raise _ClientGone()


# ---------------------------------------------------------------------------
# Tile addressing
# ---------------------------------------------------------------------------

# Preferred tile edge in pixels. 512x512 uint16 is 512 KB on the wire, which is
# where the per-request cost of the control's /data_plane proxy stops dominating
# (biopb/biopb#762): below ~256 KB of payload the deployment spends its capacity
# on proxy overhead rather than pixels. See docs/remote-viewer-tiles.md.
_TILE_TARGET_EDGE = 512

# Tiles are cached by URL, and the URL carries no token -- auth rides in the
# Authorization header so a token rotation does not invalidate the whole cache.
# That makes `private` load-bearing rather than conservative: RFC 9111 section
# 3.5 lets a *shared* cache reuse a response to an authenticated request for
# some other request when the response says `public` (or `s-maxage`, or
# `must-revalidate`), and with no token in the cache key that other request can
# be an unauthenticated stranger's. An nginx proxy_cache, CDN, or corporate
# proxy in front of a `--remote` deployment would then serve tiles with the
# token checked exactly once, for someone else. `private` keeps the win we
# actually wanted -- a per-user browser cache across pan/zoom and reload -- and
# withholds the one that cannot be made safe under bearer auth.
#
# `Vary: Authorization` is belt-and-braces for a cache that stores it anyway:
# entries then key on the credential instead of colliding across users.
#
# An hour rather than `immutable` because tile content is only stable while the
# array_id is: re-indexing a source today reuses the id, so a year-long cache
# would pin stale pixels in every browser that saw them. Lengthening this needs
# the version in the array_id namespace (the policy compact-grid settled on);
# `public` needs a different auth model altogether, e.g. signed URLs that put
# the grant in the cache key.
_TILE_CACHE_CONTROL_TEMPLATE = "private, max-age={max_age}"
_TILE_MAX_AGE = 3600


def _tensor_desc_by_array_id(client: TensorFlightClient, array_id: str) -> Any:
    """The catalog TensorDescriptor named by *array_id*, or ``None``.

    Addressed by array_id ALONE, per the identity policy at the top of
    ``proto/biopb/tensor/descriptor.proto``: array_id is globally unique and
    authoritative, ``source_id`` is only the slash-free routing prefix. The
    older routes here take a ``(source_id, tensor_id)`` pair and rejoin it with
    :func:`_request_array_id` before reading -- a split made only to be undone,
    and one that let geometry and the read resolve differently (a bare
    multi-tensor id gave tensor[0]'s shape while the read went to the source's
    own default).

    A bare source_id stays valid for a single-tensor source, which is what the
    policy says its array_id *is*. For a multi-tensor source it is refused
    rather than guessed (biopb/biopb#75); the caller turns ``None`` into a 404.
    """
    sources = client.list_sources()
    desc = sources.get(array_id.split("/", 1)[0])
    if desc is None:
        return None
    for td in desc.tensors:
        if td.array_id == array_id:
            return td
    if array_id == desc.source_id and len(desc.tensors) == 1:
        return desc.tensors[0]
    return None


def _tensor_candidates(client: TensorFlightClient, array_id: str) -> List[str]:
    """array_ids of the source *array_id* points at, for a 404 that helps."""
    desc = client.list_sources().get(array_id.split("/", 1)[0])
    return [td.array_id for td in desc.tensors] if desc else []


def _no_such_tensor(array_id: str, candidates: List[str]) -> str:
    """404 text. Naming the alternatives is most of the value when the caller
    addressed a multi-tensor source by its bare source_id."""
    if candidates:
        return f"No tensor {array_id!r}; this source has: {', '.join(candidates)}"
    return f"No tensor {array_id!r}"


def _tile_edge(
    shape: Sequence[int], chunk_shape: Sequence[int], y_idx: int, x_idx: int
) -> int:
    """Square tile edge for a tensor, in pixels at full resolution.

    Chosen so a tile *nests* inside a stored chunk -- ``chunk / 2**k`` -- rather
    than equalling it. Straddling a chunk boundary is what costs: locally it is
    a few extra page touches, but against a proxied upstream it turns one cold
    chunk pull into two (docs/remote-viewer-tiles.md). Nesting keeps the segment
    cache hitting while letting the transport unit be sized for latency instead
    of for mmap locality, which is the whole point of separating the two.
    """
    height, width = int(shape[y_idx]), int(shape[x_idx])
    plane_max = max(height, width, 1)
    cy, cx = int(chunk_shape[y_idx] or 0), int(chunk_shape[x_idx] or 0)

    # One chunk already covers the whole plane (or the adapter advertised no
    # chunking at all): there are no interior boundaries left to straddle, so
    # nesting constrains nothing and the transport target wins outright. The
    # backing read still pulls that chunk whole the first time; every other tile
    # in it is then a strided copy out of the mmap-backed segment cache. Without
    # this branch a single-chunk plane with an odd extent (1411) yields a single
    # 1411px tile -- tiling switched off for exactly the images that need it.
    if cy <= 0 or cx <= 0 or (cy >= height and cx >= width):
        return max(1, min(_TILE_TARGET_EDGE, plane_max))

    edge = min(cy, cx)
    while edge > _TILE_TARGET_EDGE and edge % 2 == 0:
        edge //= 2
    # An odd chunk edge has no power-of-two divisor to land on. Keep the chunk
    # whole rather than straddling: a tile spanning two chunks doubles the
    # cold-fetch cost against a proxied upstream, which is worse than one tile
    # over target.
    return max(1, min(edge, plane_max))


def _tile_levels(
    shape: Sequence[int], y_idx: int, x_idx: int, edge: int
) -> List[Dict[str, int]]:
    """Pyramid descriptor for the tile grid: index 0 is FULL resolution.

    Matches Viv/deck.gl's ``PixelSource[]`` convention (array index 0 = highest
    resolution), not the map-tile one where z grows with detail. Levels stop
    once the whole plane fits in a single tile.
    """
    height, width = int(shape[y_idx]), int(shape[x_idx])
    levels: List[Dict[str, int]] = []
    level = 0
    while True:
        scale = 1 << level
        lh = -(-height // scale)
        lw = -(-width // scale)
        levels.append(
            {
                "level": level,
                "scale": scale,
                "height": lh,
                "width": lw,
                "cols": max(1, -(-lw // edge)),
                "rows": max(1, -(-lh // edge)),
            }
        )
        if (lh <= edge and lw <= edge) or level > 24:
            return levels
        level += 1


def _resolve_tile_level(
    levels: List[Dict[str, int]], level: int, col: int, row: int
) -> Dict[str, int]:
    """The advertised grid entry a request addresses, or 404.

    The single gate on ``(level, col, row)``, checked against exactly the list
    ``/api/tile_info`` publishes so the two cannot disagree about what exists.

    ``level`` needs this most and is the least obvious: an unadvertised level is
    not a harmless over-zoom. ``scale_hint`` is honoured all the way down into
    ``downsample_block``, which pads its input up to a multiple of the scale
    factor before reducing -- so level 17 on a 512x512 plane asks the *data
    plane* to allocate and edge-pad a 65536x65536 array. numpy refuses the
    absurd sizes (a 502), but the band that merely exhausts memory succeeds and
    writes it, in a process shared by every other caller. One query parameter
    must not size an allocation in the backend.
    """
    if level >= len(levels):
        raise HTTPException(
            status_code=404,
            detail=(
                f"level {level} does not exist "
                f"(this tensor has {len(levels)}: 0..{len(levels) - 1})"
            ),
        )
    entry = levels[level]
    if col >= entry["cols"] or row >= entry["rows"]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"tile ({col},{row}) is outside level {level} "
                f"({entry['cols']}x{entry['rows']} tiles)"
            ),
        )
    return entry


def _plane_axes_set(y_idx: int, x_idx: int, s_idx: Optional[int]) -> set:
    """The axes the tile *is*, as opposed to the ones that select which tile."""
    return {y_idx, x_idx} | ({s_idx} if s_idx is not None else set())


def _unaddressable_axes(
    dim_labels: List[str], shape: List[int], plane: set
) -> List[Dict[str, Any]]:
    """Non-plane axes with extent > 1 that ``t``/``z``/``c`` cannot reach.

    Such an axis is served at index 0 and the rest of it is unreachable through
    this route -- true of an unlabelled axis (``POS``), and of the second of two
    axes sharing a label, since ``labeled_axis_index`` takes the first. That is
    a real limit on what the tile API can address, so publish it instead of
    leaving a client to infer it by diffing ``dim_labels`` against
    ``selectable``.
    """
    from biopb_tensor_server.core.axes import labeled_axis_index

    named = {labeled_axis_index(dim_labels, a) for a in ("t", "z", "c")}
    return [
        {
            "axis": idx,
            "label": dim_labels[idx] if idx < len(dim_labels) else "?",
            "extent": int(shape[idx]),
        }
        for idx in range(len(shape))
        if idx not in plane and idx not in named and int(shape[idx]) > 1
    ]


def _resolve_tile_selection(
    dim_labels: List[str], shape: List[int], plane: set, selection: Dict[str, int]
) -> Dict[int, int]:
    """``{axis index: chosen index}`` for every non-plane axis, or 422.

    Validation has to iterate the *parameters* as well as the axes. Checking
    only axes -- which is what the loop building the slices did -- silently
    drops a selection naming an axis the tensor does not have: the loop never
    visits it, so ``t=7`` on a plain 2-D tensor returned index 0's pixels with a
    200 and no hint, under an ETag that varied with the ignored number. One
    tile, unbounded distinct cache entries, and a client told it got a plane it
    did not get.

    Index 0 is exempt because it is the default every client sends; only a
    non-zero request for an axis that does not exist is a mistake worth
    refusing. That is the same rule the Viv adapter applies client-side, so the
    two agree on what is addressable.

    Extents are the full-resolution ones, which is correct at every level:
    ``scale_hint`` is 1 on non-plane axes, so pyramid depth never changes them.
    """
    from biopb_tensor_server.core.axes import labeled_axis_index

    named = {axis: labeled_axis_index(dim_labels, axis) for axis in ("t", "z", "c")}

    for axis, want in selection.items():
        axis_idx = named.get(axis)
        if axis_idx is None and want:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"tensor has no '{axis}' axis to select "
                    f"(dim_labels {dim_labels}); only index 0 is meaningful"
                ),
            )

    resolved: Dict[int, int] = {}
    for idx in range(len(shape)):
        if idx in plane:
            continue
        want = 0
        for axis, axis_idx in named.items():
            if axis_idx == idx:
                want = int(selection.get(axis, 0))
                break
        if not 0 <= want < int(shape[idx]):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"index {want} out of range for axis {idx} "
                    f"('{dim_labels[idx] if idx < len(dim_labels) else '?'}', "
                    f"extent {shape[idx]})"
                ),
            )
        resolved[idx] = want
    return resolved


def _tile_slices(
    td: Any,
    y_idx: int,
    x_idx: int,
    s_idx: Optional[int],
    edge: int,
    level: int,
    col: int,
    row: int,
    resolved: Dict[int, int],
) -> Tuple[List[int], List[int], List[int]]:
    """``(slice_start, slice_stop, scale_hint)`` for one tile.

    Bounds are full-resolution world coordinates (the units ``slice_hint`` is
    applied in, before scaling); ``scale_hint`` then downsamples Y/X by
    ``2**level`` so the returned plane is at most ``edge x edge``.

    Assumes ``(level, col, row)`` already passed :func:`_resolve_tile_level` and
    the selection :func:`_resolve_tile_selection`; this derives geometry and does
    not re-check either.
    """
    shape = [int(d) for d in td.shape]
    scale = 1 << level
    step = edge * scale

    y0, x0 = row * step, col * step

    start = [0] * len(shape)
    stop = list(shape)
    scale_hint = [1] * len(shape)

    start[y_idx], stop[y_idx] = y0, min(y0 + step, shape[y_idx])
    start[x_idx], stop[x_idx] = x0, min(x0 + step, shape[x_idx])
    scale_hint[y_idx] = scale_hint[x_idx] = scale

    # Every leading axis collapses to one index so the response is a single
    # plane; an axis left whole would silently multiply the payload.
    for idx, want in resolved.items():
        start[idx], stop[idx] = want, want + 1

    return start, stop, scale_hint


def _tile_etag(array_id: str, params: Sequence[Tuple[str, Any]]) -> str:
    """Strong ETag over the tile's identity tuple.

    Content is a pure function of (array_id, level, col, row, selection, format,
    render settings), so hashing those is exact -- *given* that an array_id
    denotes fixed bytes. That is the same assumption `_TILE_MAX_AGE` hedges
    against; both tighten together when array_id carries a version.
    """
    identity = "|".join([array_id] + [f"{k}={v}" for k, v in params])
    return '"' + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32] + '"'


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
    """Readiness: is the Flight backend serving *right now*?

    Answers only from what Flight just said. The old expression also accepted
    ``diag.connection_state == "connected"``, which is a record of a past
    successful connect and is never revised when the backend goes away -- so a
    sidecar whose backend had died still reported ready, which is precisely the
    window (a data-plane restart) the admin page polls this endpoint through.

    503 when not ready, so probes that can only see status work: a Kubernetes
    ``readinessProbe``, a ``curl -f`` wait loop, and the web bootstrap (which
    already backs off and retries on 503) were all being told to proceed by the
    unconditional 200 that accompanied ``"ready": false`` (biopb/biopb#755).
    """
    ctx = _sidecar(request)
    # Off the event loop: connect + health are blocking gRPC (see backend_snapshot).
    backend_health, backend_error = await run_in_threadpool(ctx.backend_snapshot)
    ready = bool(backend_health) and backend_health.get("status") == "SERVING"

    return JSONResponse(
        {
            "status": "ok" if ready else "degraded",
            "timestamp": _now_rfc3339(),
            "ready": ready,
            "dev_mode": ctx.token is None,
            "service": _SERVICE,
            "version": _VERSION,
            "backend_health": backend_health,
            "backend_error": backend_error,
            "source_count": backend_health.get("source_count", 0)
            if backend_health
            else 0,
        },
        status_code=200 if ready else 503,
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

        # Truncation info from schema metadata. An untagged Arrow table has
        # `schema.metadata is None`, not an empty dict, so go through a fallback:
        # a result carrying no truncation keys must degrade to "returned ==
        # total" rather than raise an AttributeError that the handler below would
        # then report as a 502 Flight error.
        table_metadata = arrow_table.schema.metadata or {}
        total = int(table_metadata.get(b"total_sources", len(result)))
        returned = int(table_metadata.get(b"returned_sources", len(result)))
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
        from biopb_tensor_server.core.adapter_base import unpack_chunk_array

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


# -- Tiles (cacheable GET reads) --------------------------------------------


@_router.get("/api/tile_info/{array_id:path}")
async def tile_info(array_id: str, request: Request) -> JSONResponse:
    """Everything a tiled client needs to address this tensor.

    The browser must not derive the tile grid itself: the edge follows the
    stored ``chunk_shape`` so tiles nest (see :func:`_tile_edge`), and that is a
    server-side fact. Shaped to drop straight into a Viv ``PixelSource[]`` --
    one entry per level, index 0 full resolution.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    try:
        client = ctx.get_client()
        td = _tensor_desc_by_array_id(client, array_id)
        candidates = [] if td is not None else _tensor_candidates(client, array_id)
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("TILE_INFO_FAILED", str(exc))
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )
    if td is None:
        raise HTTPException(
            status_code=404, detail=_no_such_tensor(array_id, candidates)
        )

    from biopb_tensor_server.core.axes import labeled_axis_index, plane_axes

    shape = [int(d) for d in td.shape]
    dim_labels = list(td.dim_labels)
    if len(shape) < 2:
        raise HTTPException(
            status_code=422, detail=f"Tensor is not tileable (shape {shape})"
        )
    y_idx, x_idx, s_idx = plane_axes(dim_labels, shape)
    edge = _tile_edge(shape, [int(d) for d in td.chunk_shape], y_idx, x_idx)

    return JSONResponse(
        {
            "array_id": td.array_id,
            "dim_labels": dim_labels,
            "shape": shape,
            "chunk_shape": [int(d) for d in td.chunk_shape],
            "dtype": td.dtype,
            "tile_size": edge,
            "plane": {"y": y_idx, "x": x_idx, "s": s_idx},
            "selectable": {
                axis: labeled_axis_index(dim_labels, axis) for axis in ("t", "z", "c")
            },
            "pinned": _unaddressable_axes(
                dim_labels, shape, _plane_axes_set(y_idx, x_idx, s_idx)
            ),
            "levels": _tile_levels(shape, y_idx, x_idx, edge),
        }
    )


@_router.get("/api/tile/{array_id:path}")
async def get_tile(
    array_id: str,
    request: Request,
    level: int = Query(0, ge=0, le=24),
    col: int = Query(0, ge=0),
    row: int = Query(0, ge=0),
    t: int = Query(0, ge=0),
    z: int = Query(0, ge=0),
    c: int = Query(0, ge=0),
    fmt: str = Query("raw", pattern="^(raw|png|jpeg)$"),
    lo: float = Query(1.0, ge=0.0, le=100.0),
    hi: float = Query(99.0, ge=0.0, le=100.0),
    color: str = Query("auto"),
    use_min_max: bool = Query(False),
    reduction_method: Optional[str] = Query(None),
) -> Response:
    """One tile of a tensor, addressed by pyramid level and grid position.

    A **GET** with the whole request in the URL, because ``POST /api/slice``
    cannot be cached by any browser under any header -- which left the viewer
    re-fetching pixels it had already seen on every pan and every reload.
    Everything that decides the bytes is in the URL, so the response carries an
    ETag and revalidates cheaply.

    ``fmt`` selects the transport, not a different viewer: ``raw`` ships the
    tile's own dtype for client-side (WebGL) contrast and blending, while
    ``png``/``jpeg`` bake appearance server-side for slow links and high channel
    counts. Both are valid backing stores for the same tiled client, which is
    why it is one route and not two (docs/remote-viewer-tiles.md).

    Response headers mirror /api/slice (``X-Shape``/``X-Dtype``/``X-Dim-Labels``)
    plus ``X-Tile-Size``/``X-Tile-Level``/``X-Tile-Col``/``X-Tile-Row`` so a
    client can verify the grid it assumed against the one it got.

    Cached ``private`` only, never ``public`` -- see ``_TILE_MAX_AGE``: the URL
    holds no token, so `public` would authorise a shared cache to hand an
    authenticated tile to whoever asks next.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    t0 = time.monotonic()

    # Cheapest possible bail-out, before any backend call: under a tile burst
    # this handler may have sat in the queue long enough for the browser to pan
    # away and abort. Repeated below, once the geometry is known and the
    # expensive read is the next thing that would happen.
    await _abort_if_client_gone(request, ctx)

    try:
        client = await run_in_threadpool(ctx.get_client)
        td = await run_in_threadpool(_tensor_desc_by_array_id, client, array_id)
        candidates = (
            []
            if td is not None
            else await run_in_threadpool(_tensor_candidates, client, array_id)
        )
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("TILE_FAILED", str(exc))
        raise HTTPException(
            status_code=502, detail=f"Flight error: {type(exc).__name__}"
        )
    if td is None:
        raise HTTPException(
            status_code=404, detail=_no_such_tensor(array_id, candidates)
        )

    from biopb_tensor_server.core.axes import plane_axes

    shape = [int(d) for d in td.shape]
    dim_labels = list(td.dim_labels)
    if len(shape) < 2:
        raise HTTPException(
            status_code=422, detail=f"Tensor is not tileable (shape {shape})"
        )
    y_idx, x_idx, s_idx = plane_axes(dim_labels, shape)
    edge = _tile_edge(shape, [int(d) for d in td.chunk_shape], y_idx, x_idx)

    # Before the ETag, not after: a revalidation must not be able to answer 304
    # for a tile that does not exist, and an out-of-grid request should cost the
    # same 404 whether or not the caller happens to hold a matching ETag.
    _resolve_tile_level(_tile_levels(shape, y_idx, x_idx, edge), level, col, row)
    resolved = _resolve_tile_selection(
        dim_labels,
        shape,
        _plane_axes_set(y_idx, x_idx, s_idx),
        {"t": t, "z": z, "c": c},
    )

    render_identity = (
        [("lo", lo), ("hi", hi), ("color", color), ("mm", use_min_max)]
        if fmt != "raw"
        else []
    )
    etag = _tile_etag(
        td.array_id,
        [
            ("level", level),
            ("col", col),
            ("row", row),
            ("t", t),
            ("z", z),
            ("c", c),
            ("fmt", fmt),
            ("red", reduction_method or ""),
            ("edge", edge),
        ]
        + render_identity,
    )
    cache_headers = {
        "ETag": etag,
        "Cache-Control": _TILE_CACHE_CONTROL_TEMPLATE.format(max_age=_TILE_MAX_AGE),
        "Vary": "Authorization",
        "X-Tile-Size": str(edge),
        "X-Tile-Level": str(level),
        "X-Tile-Col": str(col),
        "X-Tile-Row": str(row),
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_headers)

    start, stop, scale_hint = _tile_slices(
        td, y_idx, x_idx, s_idx, edge, level, col, row, resolved
    )

    # Last chance to skip the read: on a saturated loop this request may have sat
    # in the queue long enough for the browser to pan away and abort it.
    await _abort_if_client_gone(request, ctx)

    def _read() -> np.ndarray:
        arr_lazy = client.get_tensor(
            # The array_id the geometry above was read from, not a rebuilt one:
            # the two used to be derived separately and could disagree.
            td.array_id,
            slice_hint=_build_slice_hint(start, stop),
            scale_hint=scale_hint,
            reduction_method=reduction_method or None,
        )
        return _normalize_array(arr_lazy.compute())

    try:
        # Off the event loop, and not only to keep this request responsive: a
        # blocking compute here starves the loop of the turn it needs to notice
        # that *other* queued callers have hung up, which silently defeats
        # _abort_if_client_gone for the whole burst behind it. Concurrency
        # against the Flight client is not new -- dask's threaded scheduler
        # already drives it from several threads inside one compute().
        arr = await run_in_threadpool(_read)

        ctx.diag.latency.record((time.monotonic() - t0) * 1000)

        if fmt == "raw":
            return Response(
                content=arr.tobytes(),
                media_type="application/octet-stream",
                headers={
                    **cache_headers,
                    "X-Shape": ",".join(str(d) for d in arr.shape),
                    "X-Dtype": str(arr.dtype),
                    "X-Dim-Labels": ",".join(dim_labels),
                },
            )

        from .renderer import render_array_to_image_bytes

        image_bytes, width, height, lo_val, hi_val = render_array_to_image_bytes(
            arr,
            dim_labels,
            percentile_lo=lo if not use_min_max else 0.0,
            percentile_hi=hi if not use_min_max else 100.0,
            color=color,
            output_format=fmt,
        )
        return Response(
            content=image_bytes,
            media_type=_image_media_type(fmt),
            headers={
                **cache_headers,
                "X-Image-Width": str(width),
                "X-Image-Height": str(height),
                "X-Percentile-Lo-Value": str(lo_val),
                "X-Percentile-Hi-Value": str(hi_val),
            },
        )

    except _ClientGone:
        raise
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        ctx.diag.mark_error("TILE_FAILED", str(exc))
        logger.error(f"tile failed: {exc}")
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

    await _abort_if_client_gone(request, ctx)

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

    await _abort_if_client_gone(request, ctx)

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
            gamma=req.gamma,
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
            _request_array_id(params.source_id, params.tensor_id),
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

        # The same plane the renderer will reduce to, resolved the same way --
        # this crop and that reduction must agree on which axis is Y, or the
        # request and the picture disagree silently. Sharing plane_axes is what
        # makes that structural; it also covers the samples axis, which the
        # hand-rolled fallback here did not (a 6-D RGB TCZYXS with unrecognized
        # labels picked X/S as Y/X).
        from biopb_tensor_server.core.axes import plane_axes

        y_idx, x_idx, _ = plane_axes(dim_labels, dask_arr.shape)

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
            gamma=params.gamma,
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
            # Control-owned: the admin UI routes a restart through the control
            # (which owns the process). A self-managed plane can't be restarted
            # from the browser (biopb/biopb#418).
            "supervised": ctx.supervised,
            # No token enforced. The admin UI keys the local-only server-side file
            # chooser (#244) off this, since a tokenless plane is loopback-bound
            # and its filesystem *is* the user's own box.
            #
            # KNOWN LIMITATION: this is a proxy for topology, not topology itself.
            # It was exact while the two-mode model tied the token to the bind
            # (biopb/biopb#447), but local mode now accepts an optional token, so a
            # loopback-bound plane behind one reports false here and loses the file
            # chooser even though its filesystem really is the user's own box. It
            # fails closed (a feature hides), so it is accepted for now; the fix is
            # to derive this from the actual bind. See biopb/biopb#470.
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

    Tokenless deployments only (biopb/biopb#244): a browsable FS listing is an
    info-disclosure surface, so it is served **only** when no token is enforced —
    a loopback-bound, single-machine deployment where the server's filesystem
    *is* the user's own box. Otherwise it 404s (feature absent), matching how the
    admin UI hides the "Browse…" button unless ``/api/admin/status`` reports
    ``local``.

    Same known limitation as the ``local`` flag this mirrors (see
    ``admin_status``): a *local* plane behind an optional token 404s here even
    though its filesystem is the user's own box. Fails closed; see
    biopb/biopb#470.

    Returns ``{path, parent, entries: [{name, is_dir}], truncated}``. No path (or
    a blank one) starts at the server user's home directory; a path that is a file
    resolves to its containing directory so the chooser can navigate from it. One
    unreadable entry never fails the whole listing.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)  # no-op in local mode; guards a misconfigured caller
    if ctx.token is not None:
        # A token is enforced: never expose the server's filesystem to what may be
        # a remote browser. (Conservative — a local plane behind an optional token
        # is caught here too; see the limitation noted above.)
        raise HTTPException(
            status_code=404,
            detail="File browsing is available only on a tokenless local server",
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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    flight_location: str = "grpc://localhost:8815",
    token: Optional[str] = None,
    cache_bytes: int = 512 * 1024 * 1024,  # 512MB default (fits ~8 chunks of 64MB)
    cors_origins: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    supervised: Optional[bool] = None,
    tls_ca_pem: Optional[bytes] = None,
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
        config_path: Path to the config file this daemon was launched with; the
            /api/config routes read and write it.
        supervised: Whether the biopb control owns/supervises this data plane.
            Reported on /api/admin/status so the admin UI routes a restart to the
            control (which owns the process); a self-managed plane can't be
            restarted from the browser (biopb/biopb#418). Defaults to reading
            ``BIOPB_DATA_PLANE_SUPERVISED`` from the env the control set, so a
            directly-launched ``biopb-tensor-server launch`` is not supervised.
        tls_ca_pem: PEM certificate the flight plane serves, when TLS is on. The
            sidecar is co-located with that plane and reads this off local disk,
            so it trusts it explicitly instead of pinning it on first use.
            ``flight_location`` must then be a ``grpcs://`` URL.

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
        supervised=supervised,
        tls_ca_pem=tls_ca_pem,
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
            "X-Tile-Size",
            "X-Tile-Level",
            "X-Tile-Col",
            "X-Tile-Row",
        ],
    )
    # Note: WebSocket CORS is handled by the browser during the handshake.

    @app.exception_handler(_ClientGone)
    async def _client_gone_handler(request: Request, exc: _ClientGone) -> Response:
        """499, the status nginx uses for "client closed request".

        Nobody reads it -- the socket is already gone -- but it keeps an
        abandoned read out of the 5xx error budget, where it would otherwise
        look like the server failing under exactly the load it is shedding.
        """
        return Response(status_code=499)

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
    """Path of the shutdown sentinel file the control supervisor writes (Windows).

    The one definition ``DataPlaneSupervisor._win_stop_sentinel`` also binds to
    (both call ``biopb._locations.tensor_stop_sentinel``), so the writer and
    this watcher cannot drift. A single fixed name in the user's biopb state dir -
    NOT keyed by PID: on Windows the process the supervisor records can differ from
    the one running launch()/uvicorn (Store-Python/uv shims), so a PID in the name
    would make writer and watcher disagree. The control is the sole owner of the
    plane, so a fixed name is unambiguous.
    """
    from biopb import _locations

    return _locations.tensor_stop_sentinel()


def _install_windows_shutdown_listener(server) -> None:
    """Windows-only: let the control supervisor shut the daemon down gracefully.

    The daemon is a windowless background process (CREATE_NO_WINDOW) in its own
    process group, so it has no console to receive a CTRL_BREAK and Win32 named
    objects are awkward across sessions/elevation. So the supervisor instead drops a
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
    tls_ca_pem: Optional[bytes] = None,
) -> None:
    """Start the HTTP sidecar with uvicorn (blocking)."""
    import uvicorn

    app = create_app(
        flight_location=flight_location,
        token=token,
        cache_bytes=cache_bytes,
        cors_origins=cors_origins,
        config_path=config_path,
        tls_ca_pem=tls_ca_pem,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    # Windows: enable graceful `biopb server stop` via a sentinel-file watcher
    # that flips server.should_exit (no-op on other platforms, which use SIGTERM).
    _install_windows_shutdown_listener(server)
    server.run()
