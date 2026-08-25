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
  GET  /api/tile_info/{array_id}     — tile grid + pyramid levels for a tensor
  GET  /api/tile/{array_id}          — one tile, cacheable (raw | png | jpeg)
  POST /api/slice                    — fetch array slice as binary (body: array_id)
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
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.ticket_pb2 import TensorTicket
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
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


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SliceRequest(BaseModel):
    # The whole address (identity policy, descriptor.proto): ``source_id`` for a
    # single-tensor source, ``source_id/field`` otherwise. Not a
    # ``(source_id, tensor_id)`` pair -- that split had to be rejoined before
    # every read, and let geometry and the read resolve to different tensors.
    array_id: str
    slice_start: Optional[List[int]] = None
    slice_stop: Optional[List[int]] = None
    scale_hint: Optional[List[int]] = None
    reduction_method: Optional[str] = None
    pixel_budget: Optional[int] = None  # informational, stored in diagnostics


class QuerySourcesRequest(BaseModel):
    sql: str


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
    """The authoritative TensorDescriptor named by *array_id*, or ``None``.

    Addressed by array_id ALONE, per the identity policy at the top of
    ``proto/biopb/tensor/descriptor.proto``: array_id is globally unique and
    authoritative, ``source_id`` is only the slash-free routing prefix. Every
    route resolves here, which is the point: the routes that once took a
    ``(source_id, tensor_id)`` pair rejoined it before each read, and two
    derivations of one identity could disagree -- a bare multi-tensor id gave
    tensor[0]'s shape while the read went to the source's own default.

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
            return client.get_descriptor(array_id, with_pyramid=False)
    if array_id == desc.source_id and len(desc.tensors) == 1:
        return client.get_descriptor(array_id, with_pyramid=False)
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


def _unnamed_axes(
    dim_labels: List[str], shape: List[int], plane: set
) -> List[Dict[str, Any]]:
    """Non-plane axes with extent > 1 that ``t``/``z``/``c`` cannot *name*.

    An unlabelled axis (``POS``), a domain-specific one (a TIFF sequence's
    ``i``), the second of two axes sharing a label -- ``labeled_axis_index``
    takes the first, so the second has no name to be reached by.

    Naming is not addressing: these are selectable through the ``sel``
    parameter, which addresses an axis by its wire index and needs no name at
    all. What this list says is that a client wanting one of them must use
    ``sel``, and that it has no semantic title to put on the slider -- so it
    should show the label the source gave (``i``), not invent ``Z``. Publishing
    it beats leaving a client to infer it by diffing ``dim_labels`` against
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


# A module-level singleton rather than a call in the signature's default: the
# annotation is a container type, which is the shape B008 is there to catch.
# Declared (rather than read off `request.query_params`) so the parameter still
# appears in the OpenAPI schema.
_SEL_QUERY = Query(
    None,
    description="Select an axis by wire index: repeatable, '<axis>:<index>'.",
)

# One ``sel`` entry: an axis's wire index and the index chosen on it.
_SEL_RE = re.compile(r"^(\d{1,3}):(\d{1,12})$")

# More entries than any tensor has axes is a malformed request, not a big one.
_SEL_MAX_ENTRIES = 32


def _parse_sel(sel: Sequence[str]) -> Dict[int, int]:
    """``sel=<axis>:<index>`` repetitions -> ``{axis index: chosen index}``.

    Positional because there is nothing else to address these axes *by*: they
    are exactly the axes with no name (:func:`_unnamed_axes`), and inventing one
    server-side would be the guess ``core.axes`` declines to make. The wire index
    is unambiguous, stable, and already what the client reads out of
    ``tile_info``.

    Syntax only here -- what the axes mean is :func:`_resolve_tile_selection`'s
    job, which is where the plane and the named axes are known. A repeat of the
    same axis is refused rather than last-wins: two different answers to one
    question is a client bug, and silently picking one hides it behind an ETag
    that varies with the discarded number.
    """
    if len(sel) > _SEL_MAX_ENTRIES:
        raise HTTPException(
            status_code=422,
            detail=f"too many 'sel' parameters ({len(sel)}; max {_SEL_MAX_ENTRIES})",
        )
    out: Dict[int, int] = {}
    for item in sel:
        match = _SEL_RE.match(item.strip())
        if match is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"malformed 'sel' parameter {item!r}; "
                    f"expected '<axis>:<index>', e.g. 'sel=0:37'"
                ),
            )
        axis, want = int(match.group(1)), int(match.group(2))
        if axis in out:
            raise HTTPException(
                status_code=422,
                detail=f"axis {axis} selected twice ('sel={axis}:{out[axis]}' and {item!r})",
            )
        out[axis] = want
    return out


def _resolve_tile_selection(
    dim_labels: List[str],
    shape: List[int],
    plane: set,
    selection: Dict[str, int],
    positional: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """``{axis index: chosen index}`` for every non-plane axis, or 422.

    Two ways in, resolving to one answer. ``selection`` names an axis
    semantically (``t``/``z``/``c``) and is the readable form for the TCZYX
    tensors that are most of the catalog; ``positional`` (from ``sel``) names it
    by wire index, which is the only handle an axis with no semantic name has.

    Validation has to iterate the *parameters* as well as the axes. Checking
    only axes -- which is what the loop building the slices did -- silently
    drops a selection naming an axis the tensor does not have: the loop never
    visits it, so ``t=7`` on a plain 2-D tensor returned index 0's pixels with a
    200 and no hint, under an ETag that varied with the ignored number. One
    tile, unbounded distinct cache entries, and a client told it got a plane it
    did not get.

    Index 0 is exempt because it is the default every client sends; only a
    non-zero request for an axis that does not exist is a mistake worth
    refusing. ``sel`` gets no such exemption -- it is never a default, so
    ``sel=9:0`` on a 3-D tensor is a client addressing an axis it believes
    exists, and saying so is more useful than serving the plane it happened to
    ask around.

    An axis reachable both ways is refused rather than merged, even when the two
    agree: one axis with two spellings in one URL is two cache keys for one
    tile, and letting them disagree would make which one wins a silent policy.

    Extents are the full-resolution ones, which is correct at every level:
    ``scale_hint`` is 1 on non-plane axes, so pyramid depth never changes them.
    """
    from biopb_tensor_server.core.axes import labeled_axis_index

    named = {axis: labeled_axis_index(dim_labels, axis) for axis in ("t", "z", "c")}
    positional = positional or {}

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

    by_index = {idx: axis for axis, idx in named.items() if idx is not None}
    for idx in positional:
        if not 0 <= idx < len(shape):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'sel' names axis {idx}, which this tensor does not have "
                    f"(it has {len(shape)}: 0..{len(shape) - 1})"
                ),
            )
        if idx in plane:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"axis {idx} ('{dim_labels[idx] if idx < len(dim_labels) else '?'}') "
                    f"is part of the tile plane and cannot be selected; the tile "
                    f"*is* that axis"
                ),
            )
        if idx in by_index:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"axis {idx} is named '{by_index[idx]}' by this tensor; select it "
                    f"as '{by_index[idx]}={positional[idx]}', not 'sel={idx}:{positional[idx]}'"
                ),
            )

    resolved: Dict[int, int] = {}
    for idx in range(len(shape)):
        if idx in plane:
            continue
        want = positional.get(idx, 0)
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
    GetFlightInfo transfer ``chunk_shape`` so tiles nest (see :func:`_tile_edge`),
    and that is a server-side fact. Shaped to drop straight into a Viv
    ``PixelSource[]`` --
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
            "sel_axes": _unnamed_axes(
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
    sel: Optional[List[str]] = _SEL_QUERY,
    fmt: str = Query("raw", pattern="^(raw|png|jpeg)$"),
    reduction_method: Optional[str] = Query(None),
) -> Response:
    """One tile of a tensor, addressed by pyramid level and grid position.

    A **GET** with the whole request in the URL, because ``POST /api/slice``
    cannot be cached by any browser under any header -- which left the viewer
    re-fetching pixels it had already seen on every pan and every reload.
    Everything that decides the bytes is in the URL, so the response carries an
    ETag and revalidates cheaply.

    The plane is chosen by ``t``/``z``/``c`` where the tensor's labels name
    those axes, and by ``sel=<axis>:<index>`` -- repeatable, addressing an axis
    by its wire index -- where they do not. The second exists because the first
    cannot express a TIFF sequence's ``i`` or a plate's ``POS``: those axes have
    no name to be selected by, and the server will not invent one (``core.axes``
    on why a positional guess must not become a wire assertion). Before ``sel``
    they were served at index 0 with the rest of the axis unreachable, which
    made a 155-file sequence a one-frame tensor to every tiled client.
    ``/api/tile_info`` lists them under ``sel_axes``.

    Tiles ship as ``raw`` -- the tile's own dtype, for client-side (WebGL)
    contrast and blending. ``fmt`` survives only to *refuse* the server-composited
    ``png``/``jpeg`` forms this route used to serve: they have no caller since the
    server-rendered viewer was retired, and answering raw bytes to a request that
    asked for an image would be the silent-wrong-content failure the ``sel`` work
    was written to avoid (docs/remote-viewer-tiles.md).

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

    if fmt != "raw":
        # 410, not 400: the form was valid and is now withdrawn, which is what a
        # caller pinned to an older server needs to be told apart from a typo.
        raise HTTPException(
            status_code=410,
            detail=(
                f"fmt={fmt} (server-composited tiles) was removed with the "
                "server-rendered viewer; request fmt=raw and apply appearance "
                "client-side"
            ),
        )

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
        _parse_sel(sel or []),
    )

    etag = _tile_etag(
        td.array_id,
        [
            ("level", level),
            ("col", col),
            ("row", row),
            # The *resolved* selection, not the raw parameters: two URLs that
            # address the same plane by different spellings (`z=3` where the
            # labels name axis 0 'z', `sel=0:3` where they do not) must not mint
            # two cache entries for one tile, and a parameter the resolution
            # ignored must not vary the key at all.
            ("sel", ",".join(f"{i}:{v}" for i, v in sorted(resolved.items()))),
            ("red", reduction_method or ""),
            ("edge", edge),
        ],
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
        f"slice: array={req.array_id}, "
        f"slice={req.slice_start}-{req.slice_stop}, scale={req.scale_hint}, method={req.reduction_method}"
    )

    if req.pixel_budget is not None:
        ctx.diag.pixel_budget = req.pixel_budget

    await _abort_if_client_gone(request, ctx)

    try:
        client = ctx.get_client()

        # The same resolution the tile routes use, so one id cannot mean two
        # tensors depending on which route asked. It also refuses a bare
        # source_id on a multi-tensor source rather than guessing (#75).
        td = _tensor_desc_by_array_id(client, req.array_id)
        if td is None:
            raise HTTPException(
                status_code=404,
                detail=_no_such_tensor(
                    req.array_id, _tensor_candidates(client, req.array_id)
                ),
            )

        slice_hint = _build_slice_hint(req.slice_start, req.slice_stop)

        # Pass slice_hint to gRPC for optimized slicing (world coordinates)
        arr_lazy = client.get_tensor(
            # The array_id the descriptor above was read from, not a rebuilt one.
            td.array_id,
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

        headers = {
            "X-Shape": ",".join(str(d) for d in arr.shape),
            "X-Dtype": str(arr.dtype),
            "X-Dim-Labels": ",".join(td.dim_labels),
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
    which proxies here for the data API.

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
            "X-Tile-Size",
            "X-Tile-Level",
            "X-Tile-Col",
            "X-Tile-Row",
        ],
    )

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
    """JSON form of one tensor entry inside a source listing.

    ``chunk_shape`` is carried for shape-compatibility with the TS
    ``TensorDescriptor`` and is always ``[]`` here: a source listing is
    structural, and the transfer grid is answered per resolved tensor by
    ``/api/tile_info`` (which describes the tensor) -- biopb/biopb#812.
    """
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
