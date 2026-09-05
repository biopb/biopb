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
  GET  /api/tile_info/{array_id}     — tile grid, pyramid levels + volume plan
  GET  /api/tile/{array_id}          — one tile, cacheable (raw | png | jpeg)
  POST /api/slice                    — fetch array slice as binary (body: array_id);
                                       `scale_policy` delegates the scale to the
                                       server (3-D volumes)
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
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pyarrow.flight as flight
from biopb import _web_auth
from biopb.image.annotation_pb2 import RoiPutRequest
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
from google.protobuf import json_format
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
    # Let the SERVER pick the scale instead of naming one. Only "volume" today:
    # the single scale a whole 3-D volume is kept warm at (see _volume_plan).
    # Mutually exclusive with scale_hint -- two answers to one question.
    scale_policy: Optional[str] = None
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
        tls_fingerprint: Optional[str] = None,
    ) -> None:
        self.flight_location = flight_location
        self.token = token
        self.cache_bytes = cache_bytes
        # SHA-256 of the leaf the flight plane serves, when it serves TLS. We are
        # co-located with that plane and take this from the material it was
        # handed, so the anchor is verified rather than trusted-on-first-use --
        # and pinning the exact certificate is stronger than trusting the CA
        # bundle it came from, which would accept any sibling that CA issued.
        #
        # A fingerprint rather than the PEM (biopb/biopb#916): passing the PEM
        # resolves entirely offline, which also skips the hostname-override probe
        # -- and this dial is loopback, so a certificate minted for the host's
        # public name alone (the ordinary shape of an operator's own cert) fails
        # verification with "Peer name 127.0.0.1 is not in peer certificate". The
        # fingerprint path reaches the wire, ends up with the presented leaf as
        # its anchor, and so earns the override that makes the loopback dial
        # verify. None for a plaintext plane.
        self.tls_fingerprint = tls_fingerprint
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
                        tls_fingerprint=self.tls_fingerprint,
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
# Two policies, chosen by whether the request's array_id carried a content
# version (biopb/biopb#780).
#
# Versioned: the URL names the content, so its bytes cannot change under it --
# a re-index mints a different array_id and the old URL 404s instead of
# answering stale pixels. `immutable` is then honest, and it is worth more than
# the saved round trip: every revalidation skipped also skips a whole-catalog
# listing (biopb/biopb#834).
#
# Unversioned: a source whose URL cannot be stat'd has no version to publish, so
# nothing distinguishes its content across a re-index and an hour is the old
# hedge, kept. Absence is "no claim", never "unchanged".
#
# `public` needs a different auth model altogether (signed URLs that put the
# grant in the cache key), regardless of versioning.
_TILE_CACHE_CONTROL_TEMPLATE = "private, max-age={max_age}"
_TILE_MAX_AGE = 3600
_TILE_IMMUTABLE_MAX_AGE = 31_536_000  # a year; `immutable` is the real signal
_TILE_IMMUTABLE_CACHE_CONTROL = f"private, max-age={_TILE_IMMUTABLE_MAX_AGE}, immutable"

# Separates the source_id from its content version in the HTTP-only versioned
# array_id form, `source_id "@" token [ "/" field ]` (descriptor.proto). A
# source_id is `<type>_<hex>` and can never contain this; a *field* may, which
# is why only the half before the first "/" is ever parsed.
_VERSION_SEP = "@"


def _version_token(content_version: bytes) -> str:
    """A short, URL-safe token standing for *content_version*.

    Hashed rather than passed through: the raw token is a stat signature
    (`mtime_ns:size`) whose length varies and whose contents leak an mtime into
    every tile URL and access log. Eight hex characters is 32 bits -- collisions
    matter only between two versions OF ONE SOURCE, where the alternative to a
    collision is today's behaviour (no versioning at all), so the trade is
    strictly favourable.
    """
    return hashlib.sha256(content_version).hexdigest()[:8]


def _descriptor_version_token(td: Any) -> Optional[str]:
    """The version token carried by a bound TensorDescriptor, or None.

    Taken from the descriptor rather than the source listing on purpose: this is
    the freshest thing in the request (``get_descriptor`` is fetch-per-call by
    contract), while the listing was the expensive part and the obvious thing to
    memoize. It is off the resolution path entirely now (biopb/biopb#834), which
    is exactly why the guarantee could not have hung on it.

    Read by truthiness rather than ``HasField``: an unset proto3 field, a
    zero-length token and a server too old to carry the field at all are the
    same thing here -- no claim about content.
    """
    version = getattr(td, "content_version", None)
    return _version_token(version) if version else None


def _split_array_version(array_id: str) -> Tuple[str, Optional[str]]:
    """``(array_id without its version, token | None)``.

    Only the source half is parsed: `plate_a1b2@9f1c4e2b/A01/0` yields
    `("plate_a1b2/A01/0", "9f1c4e2b")`, while a field containing "@" is left
    alone.
    """
    head, slash, field = array_id.partition("/")
    source, sep, token = head.partition(_VERSION_SEP)
    if not sep:
        return array_id, None
    return source + slash + field, token


def _versioned_array_id(array_id: str, token: Optional[str]) -> str:
    """*array_id* with *token* spliced into its source half, or unchanged."""
    if not token:
        return array_id
    source, slash, field = array_id.partition("/")
    return f"{source}{_VERSION_SEP}{token}{slash}{field}"


def _tensor_desc_by_array_id(
    client: TensorFlightClient, array_id: str
) -> Tuple[Any, Optional[str]]:
    """``(TensorDescriptor, current version token)`` for *array_id*.

    The descriptor is ``None`` when nothing answers to the id. The token is the
    tensor's *current* one, read off the descriptor this already fetched, so no
    caller has to ask for it separately.

    Addressed by array_id ALONE, per the identity policy at the top of
    ``proto/biopb/tensor/descriptor.proto``: array_id is globally unique and
    authoritative, ``source_id`` is only the slash-free routing prefix. Every
    route resolves here, which is the point: the routes that once took a
    ``(source_id, tensor_id)`` pair rejoined it before each read, and two
    derivations of one identity could disagree -- a bare multi-tensor id gave
    tensor[0]'s shape while the read went to the source's own default.

    A bare source_id resolves to whatever the Flight server binds for it -- its
    default tensor. The sidecar does not second-guess that: array_id policy is
    the server's, and the answer comes back carrying the array_id it resolved
    to, which is the one this hands onward. The geometry and the read therefore
    come from one derivation, which is what biopb/biopb#75 was really about.

    A **content-versioned** array_id (`source@token[/field]`, biopb/biopb#780)
    resolves only while its token is the current one. A superseded token names
    content this server no longer has, so it resolves to nothing -- a 404 like
    any other id that names no tensor, not a distinct status. The caller does
    not have to distinguish them: a stale bookmark and a typo both want "ask
    again", and the 404 lists the ids that do exist.

    The token compared against comes from the *descriptor*, not the listing, so
    the listing being cheap, cached or absent cannot weaken this.
    """
    array_id, asked_version = _split_array_version(array_id)
    try:
        bound = client.get_descriptor(array_id, with_pyramid=False)
    except (flight.FlightServerError, ValueError):
        # The two terminal answers: a Flight-side addressing refusal (NOT_FOUND
        # / INVALID_ARGUMENT ride FlightServerError -- pyarrow exposes no typed
        # class for either), and the client's own directive for an unresolved
        # cloud source. Both are 404s, as they were when the listing answered
        # them. A dead backend raises neither, and still reaches the 502.
        return None, None

    current = _descriptor_version_token(bound)
    # Checked after the fetch rather than before, because the version lives on
    # the bound descriptor. That costs one descriptor read on the reject path,
    # which is cold -- a superseded id is a stale bookmark, not a hot loop.
    if asked_version is not None and asked_version != current:
        return None, current
    return bound, current


def _tensor_candidates(client: TensorFlightClient, array_id: str) -> List[str]:
    """array_ids of the source *array_id* points at, for a 404 that helps.

    Unversioned ids: they are what the catalog holds, and an unversioned request
    resolves fine (it just gets the hour-long cache policy rather than
    ``immutable``). The canonical versioned form comes from ``tile_info``.
    """
    desc = client.list_sources().get(_split_array_version(array_id)[0].split("/", 1)[0])
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


# -- The advertised pyramid -------------------------------------------------
#
# The server decides the resolution ladder and publishes it on the descriptor,
# one `PyramidLevel` per rung with the exact `scale_hint` its read path matches
# on. Two kinds ride the same field:
#
# - **native** levels (OME-Zarr multiscales, QPTIFF): a real on-disk pyramid,
#   `reduction_method="precompute"`. Reading one reads that level's own store.
#   Reading the same scale any other way decimates full resolution on the fly,
#   every time, which is what a pyramidal source used to get here
#   (biopb/biopb#889).
# - **computed** levels: what `build_pyramid_plan` emits and precache warms.
#
# Taking both from the descriptor is what lets this route stop *deriving* the
# ladder. It used to recompute the warm rung from PRECACHE_PLANE_MAX_PIXELS and
# admit in the docstring that a plane whose config this sidecar does not own
# would cost a cold read. There is nothing left to be wrong about: the ladder
# comes from the server that owns it.
#
# The tile ladder itself is untouched. Its rungs stay powers of two, which Viv's
# PixelSource[] convention requires; what the descriptor decides is where a rung
# is READ from.

_ADVERTISED_LOCK = threading.Lock()
# array_id -> (version token, levels coarsest-first). Bounded by a flush rather
# than an LRU: the entries are a handful of small tuples each, so the cap exists
# only to keep an unbounded catalog from growing this forever.
_ADVERTISED: Dict[str, Tuple[str, Tuple[_Level, ...]]] = {}
_ADVERTISED_MAX = 4096


class _Level(NamedTuple):
    """One rung of the server-advertised pyramid."""

    scale: Tuple[int, ...]
    shape: Tuple[int, ...]
    method: str
    native: bool


def _scale_magnitude(scale: Sequence[int]) -> int:
    """Product of a scale vector -- how much coarser than full resolution it is."""
    total = 1
    for value in scale:
        total *= int(value)
    return total


def _advertised_levels(
    client: TensorFlightClient, td: Any, version: Optional[str]
) -> Tuple[_Level, ...]:
    """The server's pyramid for *td*, coarsest first; ``()`` when it advertises none.

    Fetched with a second ``get_descriptor`` rather than by flipping the
    ``with_pyramid=False`` in :func:`_tensor_desc_by_array_id`: that one runs on
    every request by contract ("fetch-per-call"), and per-level sizing is the
    cost its mask exists to skip. Memoized on the tensor's content version
    instead -- the same token the ETag already trusts, and the right one, since
    the ladder is a property of the stored content and cannot change while that
    does not.

    **A tensor with no version is not memoized at all.** There is then nothing
    that changes when the file does, so a stored ladder would outlive the content
    it describes for the life of the process -- and a stale ladder is not a slow
    tile, it is a read addressed to a level that may no longer be there. Such a
    source pays one descriptor call per tile, which is the safe direction to err
    and is what this route cost before any of this.

    Level 0 is dropped: an identity scale names full resolution, which is what a
    caller gets without asking for a level at all. A level whose arity does not
    match the shape is dropped rather than raised on -- a ladder this cannot read
    is one this route does not use, and the tile still has to be served.
    """
    key = td.array_id
    if version is not None:
        with _ADVERTISED_LOCK:
            hit = _ADVERTISED.get(key)
        if hit is not None and hit[0] == version:
            return hit[1]

    ndim = len(td.shape)
    try:
        pyramid = client.get_descriptor(key, with_pyramid=True).pyramid
    except Exception:
        # A ladder that could not be fetched is not a failed tile: fall back to
        # reading the rung itself. Not cached, so a transient failure does not
        # pin the tensor to the slow path for the life of the process.
        logger.debug("tile: pyramid fetch failed for %s", key, exc_info=True)
        return ()

    levels = [
        _Level(
            scale=tuple(int(s) for s in level.scale_hint),
            shape=tuple(int(d) for d in level.shape),
            method=str(level.reduction_method or ""),
            native=bool(level.native),
        )
        for level in pyramid
    ]
    levels = [
        level
        for level in levels
        if len(level.scale) == ndim
        and all(s >= 1 for s in level.scale)
        and any(s > 1 for s in level.scale)
    ]
    levels.sort(key=lambda level: _scale_magnitude(level.scale), reverse=True)
    result = tuple(levels)

    if version is not None:
        with _ADVERTISED_LOCK:
            if len(_ADVERTISED) >= _ADVERTISED_MAX:
                _ADVERTISED.clear()
            _ADVERTISED[key] = (version, result)
    return result


def _pick_level(
    levels: Sequence[_Level], target: Sequence[int]
) -> Optional[Tuple[_Level, List[int]]]:
    """``(level, residual)`` for the coarsest advertised level dividing *target*.

    Coarsest by scale magnitude rather than by position, so the answer does not
    depend on the order the levels arrive in -- picking a finer qualifying level
    reads whole multiples more bytes for the same tile.

    A level qualifies only if its factors divide the wanted scale on **every**
    axis, which is what makes the residual whole. That also, for free, rules out
    a level that downsamples an axis the caller wants kept: a 2-D tile request
    carries 1 on z, and 1 % 2 != 0, so the 3-D target is never picked for a tile
    while remaining the obvious pick for a volume.

    The level's ``scale`` goes back to the caller verbatim because that is what
    the read path matches on -- a native level's ``_find_level_for_scale``
    compares for equality, so a recomputed-but-equivalent vector is a miss, not a
    near miss; and a computed level's chunk_ids are the ones precache warmed.
    """
    best: Optional[_Level] = None
    for level in levels:
        if len(level.scale) != len(target):
            continue
        if not all(
            int(t) % int(s) == 0 for s, t in zip(level.scale, target, strict=True)
        ):
            continue
        if best is None or _scale_magnitude(level.scale) > _scale_magnitude(best.scale):
            best = level
    if best is None:
        return None
    return best, [int(t) // int(s) for s, t in zip(best.scale, target, strict=True)]


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


class _TileRead(NamedTuple):
    """Where one rung is read from, and what is reduced afterwards.

    ``scale_hint`` is an advertised level's vector, or ``None`` to read the rung
    itself at ``read_level``. ``method`` is the ``reduction_method`` to send --
    the chosen level's own -- or ``None`` to leave it to the server's default.
    ``residual`` is the local decimation applied to the result, or ``None``.
    """

    scale_hint: Optional[List[int]]
    read_level: Optional[int]
    method: Optional[str]
    residual: Optional[List[int]]


def _level_matches_grid(level: _Level, shape: Sequence[int]) -> bool:
    """Whether *level*'s own extent is the one the tile grid derives.

    :func:`_tile_levels` sizes every rung ``ceil(base / scale)``, and so does the
    read path -- decimation returns ``ceil(extent / scale)``. A native level
    whose writer **floored** its shape holds genuinely fewer pixels than that, so
    routing a tile to it would promise a column the store does not have: a short
    last tile at a ragged edge, and an *empty* one where the last tile is a
    single pixel wide. Rather than reshape the published grid per rung, such a
    level is left to the Flight clients that address it by name, and this route
    reads the rung itself.

    An unstated shape is not evidence of disagreement, so it passes: a computed
    level is ceil by construction, and the check exists for on-disk levels.
    """
    if not level.shape:
        return True
    return len(level.shape) == len(shape) and all(
        int(extent) == -(-int(base) // int(scale))
        for base, scale, extent in zip(shape, level.scale, level.shape, strict=True)
    )


def _tile_read(
    shape: Sequence[int],
    y_idx: int,
    x_idx: int,
    level: int,
    levels: Sequence[_Level],
) -> _TileRead:
    """Serve one rung from the coarsest advertised level that divides it.

    The remainder is decimated in-process rather than asked of the data plane as
    a separate scaled read, so one advertised level serves the whole tail of the
    ladder above it (docs/precache-policy.md 4.2) and mints no second cache
    entry. A rung *finer* than every advertised level reads full resolution,
    which is the planner's own position: it omits the intermediate rungs because
    they cost a client a level-0 read anyway and save it nothing.

    What the level is decides what the read costs. A **computed** level is the
    one precache warmed, so this is a warm read plus a decimation. A **native**
    level is a read of that level's own store, which is the whole of
    biopb/biopb#889 -- the same scale asked for any other way decimates full
    resolution on the fly, every time.

    A rung finer than every advertised level reads its own scale, with one
    exception: a tensor whose ladder is *only* full resolution has no coarser
    level to be finer than, so full resolution is itself the anchor and the tail
    above it is still reduced from there. That is the shape the old
    ``warm_level == 0`` case had, kept because it is the right one -- reducing
    in-process reuses the level-0 chunks every other rung reads, where asking
    the data plane for the scale mints a second entry that nothing warmed.

    A level whose stored extent disagrees with the grid's arithmetic is skipped
    (:func:`_level_matches_grid`), so a rung never promises pixels the store does
    not hold.

    Always a decimation, because a tile is the display path: it is read from
    whichever level of the server's pyramid is cheapest, and a stored level is
    the writer's downsampling whatever kernel the caller might have named. A
    caller that wants a *specific* kernel wants ``POST /api/slice``, where
    ``reduction_method`` is forwarded verbatim and is part of ``chunk_id``.

    Exactness differs between the two kinds of level, and only the computed case
    claims it: the
    chunk grid is absolute, so a tile origin is a multiple of ``edge * 2**level``
    and therefore on the sample grid at both scales -- ``data[::32]`` and
    ``data[::8][::4]`` pick the same elements, and
    ``ceil(ceil(n/a)/b) == ceil(n/(a*b))`` gives the same count. Reducing from a
    **native** level does not reproduce a level-0 decimation, because the stored
    level is the writer's downsampling and not ours. It reproduces what the
    pyramid says the image looks like at that scale -- which is what every Flight
    client following the advertised pyramid already gets, and the reason the
    pyramid is worth reading at all.
    """
    target = [1] * len(shape)
    target[y_idx] = target[x_idx] = 1 << level
    picked = _pick_level(
        [entry for entry in levels if _level_matches_grid(entry, shape)], target
    )
    if picked is not None:
        picked_level, residual = picked
        return _TileRead(
            scale_hint=list(picked_level.scale),
            read_level=None,
            method=picked_level.method or None,
            residual=residual if any(f > 1 for f in residual) else None,
        )
    if not levels and level > 0:
        residual = [1] * len(shape)
        residual[y_idx] = residual[x_idx] = 1 << level
        return _TileRead([1] * len(shape), None, None, residual)
    return _TileRead(None, level, None, None)


# -- Volume (3-D) -----------------------------------------------------------
#
# A volume is not a rung of either ladder: `XR3DLayer` and napari's 3-D mode
# both upload one whole 3-D texture, so there is nothing to tile and nothing to
# zoom between. What they need is the single scale the precache worker keeps a
# whole volume warm at -- the Flight ladder's coarsest level (N1,
# docs/precache-policy.md 3.2, 5). The server decides it; a client that guessed
# would miss the warm chunks by a factor of two and pay a cold decode.


def _volume_plan(
    shape: Sequence[int],
    dim_labels: Sequence[str],
    dtype: str,
    levels: Sequence[_Level],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(plan, None)`` for a tensor that can be rendered as one volume, else
    ``(None, reason)``.

    The plan's ``scale_hint`` is the coarsest rung of ``levels`` -- the
    server-advertised ladder, which is what ``precache`` warms whole (5) for a
    computed pyramid and the coarsest stored level for a native one. It is what
    ``scale_policy="volume"`` resolves to on ``POST /api/slice``. Everything
    else in the plan is derived from that level so a client can size the read,
    and the VRAM it will cost, before issuing it.

    Taken from the descriptor rather than recomputed from the module defaults,
    which is what this used to do: the plane's pyramid config belongs to the
    Flight server, and this sidecar can address one whose config file it does
    not own. A tensor that advertises no ladder at all gets full resolution --
    the same answer, since there was no coarser level to find.

    The one thing not taken on trust is the **size**. A server advertises a
    native pyramid instead of its computed plan, so a ladder that downsamples
    only Y/X leaves the 3-D voxel budget unapplied (biopb/biopb#891); this falls
    back to the computed plan when that happens, and refuses when even that
    cannot fit. The budget is the module default, which is the weaker half of
    the same caveat -- but a ceiling from the wrong constant still bounds the
    read, where no ceiling at all does not.

    The refusals are facts about the tensor, and each is a thing a 3-D renderer
    cannot do rather than a thing this route declines to do:

    - **no z axis, or z extent 1.** There is no depth to render. An unlabeled
      leading axis is *not* promoted to z (``precache_z_index`` -- it may be T
      or C, and guessing would render a timelapse as a solid block).
    - **interleaved samples.** ``XR3DLayer`` takes one scalar volume per
      channel; an interleaved RGB(A) axis is a per-voxel tuple, which would have
      to be de-interleaved into three volumes. Refused rather than served as a
      volume three times too wide.
    """
    from biopb_tensor_server.core.axes import plane_axes
    from biopb_tensor_server.core.chunk import (
        PRECACHE_PIXEL_BUDGET_CUBIC_ROOT,
        compute_pyramid_scale_hints,
        estimate_chunk_bytes,
        precache_z_index,
    )
    from biopb_tensor_server.core.downsample import ceil_div

    shape = [int(d) for d in shape]
    dim_labels = list(dim_labels)
    if len(shape) < 3:
        return None, f"tensor has {len(shape)} axes; a volume needs at least 3"

    y_idx, x_idx, s_idx = plane_axes(dim_labels, shape)
    if s_idx is not None:
        return None, (
            f"axis {s_idx} is an interleaved samples axis; 3-D rendering takes "
            "one scalar volume per channel"
        )

    z_idx = precache_z_index(shape, dim_labels)
    if z_idx is None or z_idx in (y_idx, x_idx):
        return None, (
            f"no z axis to give the volume depth (dim_labels {dim_labels}); "
            "an unlabeled axis is not assumed to be z"
        )
    if shape[z_idx] <= 1:
        return None, f"z axis (axis {z_idx}) has extent {shape[z_idx]}, not a volume"

    # The coarsest advertised rung, or full resolution when nothing coarser is
    # advertised. `level.shape` is the server's own extent for that level, which
    # for a native level is the stored shape -- not necessarily `ceil_div(base,
    # scale)`, since writers differ on how they round. Trust it where it is
    # well-formed and fall back to the arithmetic where it is not.
    coarsest = (
        max(levels, key=lambda level: _scale_magnitude(level.scale)) if levels else None
    )
    scale = list(coarsest.scale) if coarsest is not None else [1] * len(shape)
    method = (coarsest.method or None) if coarsest is not None else None
    if coarsest is not None and len(coarsest.shape) == len(shape):
        level_shape = list(coarsest.shape)
    else:
        level_shape = [ceil_div(d, s) for d, s in zip(shape, scale, strict=True)]
    # Then bound it. A native pyramid is advertised *instead of* the computed
    # plan (adapter_base._advertised_pyramid), and NGFF multiscales commonly
    # downsample Y/X only -- so the coarsest level of a 3-D stack can be a
    # full-depth volume with nothing having applied the 3-D budget at all
    # (biopb/biopb#891). Unbounded here is not a slow render: an 11x-budget
    # volume is gigabytes on the wire and gigabytes of VRAM after Viv's Float32
    # upload, which kills the tab.
    #
    # The fallback is the server's own computed plan rather than a second
    # planner living here -- this is a ceiling, not a choice of ladder. It costs
    # a computed read (decimating full resolution, uncached) where the advertised
    # level would have been cheap, which is the right trade against not
    # rendering. Refuse only where Phase 2 itself cannot meet the budget, i.e.
    # every axis already at the floor.
    budget = PRECACHE_PIXEL_BUDGET_CUBIC_ROOT**3
    if level_shape[z_idx] * level_shape[y_idx] * level_shape[x_idx] > budget:
        scale = list(compute_pyramid_scale_hints(shape, dim_labels)[-1])
        method = None
        level_shape = [ceil_div(d, s) for d, s in zip(shape, scale, strict=True)]
        voxels = level_shape[z_idx] * level_shape[y_idx] * level_shape[x_idx]
        if voxels > budget:
            return None, (
                f"the coarsest level this server can offer is {voxels:,} voxels, "
                f"over the {budget:,} a 3-D texture holds"
            )

    depth, height, width = (
        level_shape[z_idx],
        level_shape[y_idx],
        level_shape[x_idx],
    )
    return {
        "axes": {"z": z_idx, "y": y_idx, "x": x_idx},
        "scale_hint": scale,
        # The level's own reduction_method ("precompute" for a native level),
        # sent verbatim: a native level is routed by exact scale match, so the
        # method is half of the address, not a preference.
        "reduction_method": method,
        "level_shape": level_shape,
        "depth": depth,
        "height": height,
        "width": width,
        # What the read returns, not what the GPU holds: Viv casts every volume
        # to Float32 on upload regardless of source dtype (3.1), so VRAM is
        # 4 bytes per voxel and the client sizes that from the extents.
        "bytes": estimate_chunk_bytes((depth, height, width), dtype),
    }, None


def _volume_block(td: Any, levels: Sequence[_Level]) -> Dict[str, Any]:
    """The ``volume`` field of ``/api/tile_info``: what a 3-D read of *td* gets.

    Always present, and always answers the availability question first --
    ``available: false`` with a ``reason`` is the useful answer for a plain 2-D
    tensor, and lets a viewer say why the 3-D toggle is off instead of offering
    a button that 422s.

    ``spacing`` is the physical extent of one voxel **of the returned volume**,
    i.e. the descriptor's full-resolution ``physical_scale`` already multiplied
    by this plan's per-axis scale. Publishing the product rather than the two
    factors is deliberate: a renderer needs the anisotropy ratio, and deriving
    it from the wrong one of the two (which is easy to do -- one is per-axis in
    wire order, the other in z/y/x order) silently stretches the volume.

    The three are reduced to **one unit** here, because the ratio compares them
    against each other and ``physical_unit`` is per-axis. Adapters do not all
    normalise: NIfTI reports whatever ``xyzt_units`` says (``m``/``mm``/``µm``,
    and ``mm`` is the medical convention), and the EM readers pass rsciio's
    units through untouched (``nm``, ``Å``). Both are volumetric formats, so
    this is the path where it would bite. A z in ``nm`` beside an x/y in ``µm``
    is a stack rendered a thousand times too deep, with nothing on screen to say
    so.

    ``null`` -- render isotropic -- when any of the three lacks a positive size,
    or when they carry units that differ and cannot all be placed on a common
    scale. Units that differ but *convert* are converted; units that are equal
    are kept as they are, including when they are equally unknown, since a ratio
    of like-for-like is valid whether or not we can name the unit.
    """
    plan, reason = _volume_plan(td.shape, td.dim_labels, td.dtype, levels)
    if plan is None:
        return {"available": False, "reason": reason}
    axes = plan["axes"]
    scale = plan["scale_hint"]
    spacing, unit = _volume_spacing(td, axes, scale)
    return {
        "available": True,
        "reason": None,
        "axes": axes,
        "scale_hint": scale,
        "depth": plan["depth"],
        "height": plan["height"],
        "width": plan["width"],
        "bytes": plan["bytes"],
        "spacing": spacing,
        "unit": unit,
    }


def _volume_spacing(
    td: Any, axes: Dict[str, int], scale: Sequence[int]
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """``(spacing, unit)`` for :func:`_volume_block`, or ``(None, None)``.

    Split out because the unit reconciliation is the whole of it; see that
    function's docstring for why the three axes cannot be assumed to share one.
    """
    from biopb_tensor_server.adapters._scale import MICRON, unit_to_um

    physical = [float(v) for v in td.physical_scale]
    units = [str(u) for u in td.physical_unit]
    order = ("z", "y", "x")
    if len(physical) != len(scale) or len(units) != len(scale):
        return None, None
    if not all(physical[axes[a]] > 0 for a in order):
        return None, None

    raw = {a: physical[axes[a]] * scale[axes[a]] for a in order}
    factors = {a: unit_to_um(units[axes[a]]) for a in order}
    if all(factors[a] for a in order):
        return {a: raw[a] * factors[a] for a in order}, MICRON
    # Not all placeable on a common scale. Equal units still give a valid ratio
    # -- including equally unknown ones, which is how a NIfTI with an unset
    # `xyzt_units` keeps its anisotropy.
    spellings = {units[axes[a]] for a in order}
    if len(spellings) == 1:
        return raw, spellings.pop() or None
    return None, None


# The scale decisions a caller may delegate to the server. One today; named
# rather than boolean because the warm set has two targets (2-D and 3-D,
# docs/precache-policy.md 5) and "the warm scale" would not say which.
_SCALE_POLICIES = ("volume",)


def _resolve_scale(
    req: SliceRequest, td: Any, levels: Sequence[_Level]
) -> Tuple[Optional[List[int]], Optional[str]]:
    """``(scale, reduction_method)`` a ``/api/slice`` read is issued at.

    The caller's ``scale_hint`` unless it delegated the choice; see
    :func:`slice_tensor` for why delegating is the only way to hit the warm
    chunks for a volume.

    The method rides along because under a policy the two are one answer: the
    resolved level may be a **native** one, which is addressed by an exact
    ``(scale_hint, "precompute")`` pair and would be missed by either half
    alone. ``None`` means the caller's own method stands.
    """
    policy = (req.scale_policy or "").strip().lower()
    if not policy:
        return req.scale_hint or None, None
    if req.scale_hint:
        raise HTTPException(
            status_code=422,
            detail=(
                f"scale_policy={policy!r} and scale_hint={req.scale_hint} both "
                "set; a read has one scale -- send one or the other"
            ),
        )
    if policy not in _SCALE_POLICIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown scale_policy {policy!r} (known: {', '.join(_SCALE_POLICIES)})"
            ),
        )
    plan, reason = _volume_plan(td.shape, td.dim_labels, td.dtype, levels)
    if plan is None:
        # The same sentence /api/tile_info's `volume.reason` carries, so a
        # client that skipped the info call is told the same thing.
        raise HTTPException(
            status_code=422,
            detail=f"scale_policy='volume' does not apply to {td.array_id!r}: {reason}",
        )
    return list(plan["scale_hint"]), plan["reduction_method"]


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
    read_level: Optional[int] = None,
    read_scale_hint: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """``(slice_start, slice_stop, scale_hint)`` for one tile.

    Bounds are full-resolution world coordinates (the units ``slice_hint`` is
    applied in, before scaling); ``scale_hint`` then downsamples Y/X by
    ``2**level`` so the returned plane is at most ``edge x edge``.

    ``read_level`` (default ``level``) splits the two uses of the scale apart:
    the *bounds* always come from the level being addressed, while the
    ``scale_hint`` comes from the level actually read. Passing a finer
    ``read_level`` asks for the same world region at a finer sampling, which the
    caller then reduces the rest of the way -- see :func:`_tile_read`.

    ``read_scale_hint`` overrides that derivation with a whole per-axis vector,
    for the one source of a rung that is not a power of two on Y/X: a native
    pyramid level, whose scale must go to the adapter exactly as advertised
    (:func:`_pick_level`). It supersedes ``read_level``; the bounds are
    unaffected either way.

    Assumes ``(level, col, row)`` already passed :func:`_resolve_tile_level` and
    the selection :func:`_resolve_tile_selection`; this derives geometry and does
    not re-check either.
    """
    shape = [int(d) for d in td.shape]
    scale = 1 << level
    read_scale = 1 << (level if read_level is None else read_level)
    step = edge * scale

    y0, x0 = row * step, col * step

    start = [0] * len(shape)
    stop = list(shape)
    scale_hint = [1] * len(shape)

    start[y_idx], stop[y_idx] = y0, min(y0 + step, shape[y_idx])
    start[x_idx], stop[x_idx] = x0, min(x0 + step, shape[x_idx])
    if read_scale_hint is not None:
        scale_hint = [int(s) for s in read_scale_hint]
    else:
        scale_hint[y_idx] = scale_hint[x_idx] = read_scale

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


# ---------------------------------------------------------------------------
# ROI annotations (biopb-tensor-server/docs/roi-annotations.md)
#
# Its own /api/rois/* namespace, so nothing here is shadowed by the greedy
# /api/sources/{source_id:path} catch-all. Bodies are canonical proto3 JSON in
# both directions -- json_format here, protobuf-es in the SPA -- so one schema
# serves both ends and neither hand-writes a DTO.
# ---------------------------------------------------------------------------


def _roi_bare_id(array_id: str) -> Tuple[str, Optional[str]]:
    """Strip the HTTP version token before the store ever sees the array_id.

    Annotations anchor on the UNVERSIONED array_id so they outlive an in-place
    edit of the image; the token is spliced back onto the way out so the SPA
    keeps addressing tensors in the form it already uses.
    """
    return _split_array_version(array_id)


def _roi_flight_error(exc: Exception) -> HTTPException:
    """Map a Flight failure onto the status the caller can act on."""
    if isinstance(exc, flight.FlightUnavailableError):
        # The server does not offer annotations (disabled, or no metadata DB).
        return HTTPException(status_code=501, detail=str(exc))
    if isinstance(exc, flight.FlightServerError):
        # Rejected geometry, mismatched array_id, cap breached: caller's problem.
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=502, detail=f"Flight error: {type(exc).__name__}")


@_router.get("/api/rois/{array_id:path}")
async def list_rois(array_id: str, request: Request) -> JSONResponse:
    """A tensor's whole annotation set, optionally one layer (``?set=``).

    No plane or bbox filter by design: the client needs every ROI resident to
    hit-test, drag a vertex and re-render, and a viewport-filtered fetch would
    make the ROI being edited vanish on a pan.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    bare_id, token = _roi_bare_id(array_id)
    set_name = request.query_params.get("set", "")
    try:
        result = ctx.get_client().list_rois(bare_id, set_name)
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("ROI_LIST_FAILED", str(exc))
        raise _roi_flight_error(exc)
    return JSONResponse(_roi_result_to_dict(result, token))


@_router.post("/api/rois/{array_id:path}")
async def put_rois(array_id: str, request: Request) -> JSONResponse:
    """Create or update annotations: ``{"rois": [...], "check_rev": bool}``.

    ``drawn_against_version`` is the caller's to set -- the SPA already holds the
    tensor's descriptor, and filling it here would cost a describe round trip on
    every save.
    """
    ctx = _sidecar(request)
    ctx.check_token(request)
    _require_same_origin(request)
    bare_id, token = _roi_bare_id(array_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body is not valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body must be a JSON object")

    req = RoiPutRequest()
    try:
        json_format.ParseDict(
            {
                "rois": body.get("rois", []),
                "checkRev": bool(body.get("check_rev", body.get("checkRev", False))),
            },
            req,
        )
    except json_format.ParseError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid annotation: {exc}")

    # Strip the version from the BODY too, not just the path. Responses carry
    # versioned array_ids (so the SPA keeps addressing tensors the way it
    # already does), which means the natural read-edit-write round trip hands
    # them straight back -- and the store, which only ever sees bare ids, would
    # reject them as a mismatched tensor. The sidecar owns this translation at
    # every boundary it has: path in, body in, body out.
    for roi in req.rois:
        if roi.array_id:
            roi.array_id = _split_array_version(roi.array_id)[0]

    try:
        result = ctx.get_client().put_rois(
            bare_id, list(req.rois), check_rev=req.check_rev
        )
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("ROI_PUT_FAILED", str(exc))
        raise _roi_flight_error(exc)

    payload = json_format.MessageToDict(result)
    for stored in payload.get("stored", []):
        stored["arrayId"] = _versioned_array_id(stored.get("arrayId", ""), token)
    return JSONResponse(payload)


@_router.delete("/api/rois/{array_id:path}")
async def delete_rois(array_id: str, request: Request) -> JSONResponse:
    """Delete annotations: ``?ids=a,b`` for specific ones, else the whole
    tensor's set, narrowed by ``?set=`` when given."""
    ctx = _sidecar(request)
    ctx.check_token(request)
    _require_same_origin(request)
    bare_id, _token = _roi_bare_id(array_id)
    raw_ids = request.query_params.get("ids", "")
    roi_ids = [part for part in raw_ids.split(",") if part]
    set_name = request.query_params.get("set", "")
    try:
        result = ctx.get_client().delete_rois(bare_id, roi_ids, set_name)
    except HTTPException:
        raise
    except Exception as exc:
        ctx.diag.mark_error("ROI_DELETE_FAILED", str(exc))
        raise _roi_flight_error(exc)
    return JSONResponse(json_format.MessageToDict(result))


def _roi_result_to_dict(result, token: Optional[str]) -> Dict[str, Any]:
    """Proto3 JSON for a RoiListResult, with array_ids re-versioned."""
    payload = json_format.MessageToDict(result)
    for roi in payload.get("rois", []):
        roi["arrayId"] = _versioned_array_id(roi.get("arrayId", ""), token)
    return payload


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
        # Published here and nowhere else: the viewer threads this array_id
        # through every subsequent tile URL, so the versioned form IS the
        # delivery mechanism -- no new field, no client change (biopb/biopb#780).
        td, version = _tensor_desc_by_array_id(client, array_id)
        candidates = [] if td is not None else _tensor_candidates(client, array_id)
        levels = () if td is None else _advertised_levels(client, td, version)
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
            "array_id": _versioned_array_id(td.array_id, version),
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
            # Advisory: the ladder the SERVER advertises, which is what the rungs
            # above are actually read from -- a native on-disk level where the
            # source ships one, else the computed level precache warms. Published
            # for diagnosis; the rungs are addressed identically either way.
            "pyramid": [
                {
                    "scale_hint": list(level.scale),
                    "shape": list(level.shape),
                    "reduction_method": level.method,
                    "native": level.native,
                }
                for level in levels
            ],
            # Not a rung of this ladder -- a 3-D renderer takes one whole
            # volume, not tiles -- but published here because this is the one
            # call a viewer already makes before it can address the tensor at
            # all. What it describes is the read `scale_policy="volume"` issues.
            "volume": _volume_block(td, levels),
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

    # Parsed once and used twice: it decides the ETag and the cache policy, and
    # the two must not be able to disagree about whether this URL is versioned.
    _, asked_version = _split_array_version(array_id)

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

    if reduction_method:
        from biopb_tensor_server.core.downsample import normalize_reduction_method

        if normalize_reduction_method(reduction_method) != "nearest":
            # 410 for the same reason `fmt` is: the form was valid and is now
            # withdrawn, which a caller pinned to an older server needs told
            # apart from a typo. A tile is the display path -- it is served from
            # whichever level of the server's pyramid is cheapest, so what it
            # once selected was a *store*, not a kernel. Aliases resolve first,
            # so `decimate` is still `nearest` and still fine.
            raise HTTPException(
                status_code=410,
                detail=(
                    f"reduction_method={reduction_method!r} on a tile was "
                    "withdrawn: tiles are read from the server's pyramid, whose "
                    "levels carry their own reduction. Use POST /api/slice to "
                    "choose a kernel"
                ),
            )

    # Cheapest possible bail-out, before any backend call: under a tile burst
    # this handler may have sat in the queue long enough for the browser to pan
    # away and abort. Repeated below, once the tile is known to exist and the
    # expensive read is the next thing that would happen.
    await _abort_if_client_gone(request, ctx)

    try:
        client = await run_in_threadpool(ctx.get_client)
        td, current_version = await run_in_threadpool(
            _tensor_desc_by_array_id, client, array_id
        )
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
    from biopb_tensor_server.core.downsample import downsample_block

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
            ("edge", edge),
            # The source's CURRENT version, not the one the caller asked for.
            # The versioned URL already changes on a re-index; this is what
            # stops the *unversioned* URL -- stable across re-index -- from
            # answering 304 for bytes that changed. Empty when the source
            # publishes no version, which keeps exactly today's semantics.
            ("cv", current_version or ""),
        ],
    )
    # The resolution above already refused a superseded token, so a token that
    # is still here is current -- and the URL therefore names its own content.
    cache_headers = {
        "ETag": etag,
        "Cache-Control": (
            _TILE_IMMUTABLE_CACHE_CONTROL
            if asked_version is not None
            else _TILE_CACHE_CONTROL_TEMPLATE.format(max_age=_TILE_MAX_AGE)
        ),
        "Vary": "Authorization",
        "X-Tile-Size": str(edge),
        "X-Tile-Level": str(level),
        "X-Tile-Col": str(col),
        "X-Tile-Row": str(row),
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_headers)

    # Last chance to skip the read: on a saturated loop this request may have sat
    # in the queue long enough for the browser to pan away and abort it. The tile
    # is already known to exist -- the resolution above refused an out-of-grid
    # one -- so what is skipped here is only the expensive half.
    await _abort_if_client_gone(request, ctx)

    def _read() -> np.ndarray:
        # Resolved here, not above: reading the ladder costs a Flight call on a
        # cache miss (_advertised_levels) and this is the event loop. Nothing
        # between the handler's entry and the read depends on the answer.
        plan = _tile_read(
            shape,
            y_idx,
            x_idx,
            level,
            _advertised_levels(client, td, current_version),
        )
        start, stop, scale_hint = _tile_slices(
            td,
            y_idx,
            x_idx,
            s_idx,
            edge,
            level,
            col,
            row,
            resolved,
            read_level=plan.read_level,
            read_scale_hint=plan.scale_hint,
        )
        arr_lazy = client.get_tensor(
            # The array_id the geometry above was read from, not a rebuilt one:
            # the two used to be derived separately and could disagree.
            td.array_id,
            slice_hint=_build_slice_hint(start, stop),
            scale_hint=scale_hint,
            reduction_method=plan.method,
        )
        arr = _normalize_array(arr_lazy.compute())
        if plan.residual is None:
            return arr
        return _normalize_array(downsample_block(arr, tuple(plan.residual), "nearest"))

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

    The scale is normally the caller's (``scale_hint``). ``scale_policy`` hands
    that decision back to the server: ``"volume"`` reads at the one scale a
    whole 3-D volume is kept warm at, which is the level napari 3-D and
    ``XR3DLayer`` upload as a single texture (docs/precache-policy.md 5). A
    client cannot compute that itself without reimplementing the pyramid
    planner, and a guess that lands one rung away misses every warmed chunk and
    pays a cold decode of the source instead. The two are mutually exclusive:
    one read has one scale, and letting them disagree would make which one wins
    a silent policy.

    Response headers:
      X-Shape     — comma-separated dimensions of the returned array
      X-Dtype     — numpy dtype string (e.g. "uint16", "float32")
      X-Dim-Labels — comma-separated semantic axis labels
      X-Scale-Hint — comma-separated per-axis scale actually read at

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
        td, version = _tensor_desc_by_array_id(client, req.array_id)
        if td is None:
            raise HTTPException(
                status_code=404,
                detail=_no_such_tensor(
                    req.array_id, _tensor_candidates(client, req.array_id)
                ),
            )

        slice_hint = _build_slice_hint(req.slice_start, req.slice_stop)
        # Only a delegating read needs the ladder, and fetching it costs a
        # descriptor call on a cold cache -- so a caller that named its own
        # scale does not pay for one.
        levels = _advertised_levels(client, td, version) if req.scale_policy else ()
        scale_hint, scale_method = _resolve_scale(req, td, levels)

        # Last chance to skip the read, and the one that matters most on this
        # route: a scale_policy read is a whole volume, so the queue behind it
        # can be seconds long.
        await _abort_if_client_gone(request, ctx)

        def _read() -> np.ndarray:
            # Pass slice_hint to gRPC for optimized slicing (world coordinates)
            arr_lazy = client.get_tensor(
                # The array_id the descriptor above was read from, not a rebuilt one.
                td.array_id,
                slice_hint=slice_hint,
                scale_hint=scale_hint,
                reduction_method=scale_method or req.reduction_method or None,
            )
            return _normalize_array(arr_lazy.compute())

        # Off the event loop for the same reason the tile route is: a blocking
        # compute here starves the loop of the turn it needs to notice that
        # *other* queued callers have hung up. A volume read makes that starve
        # long enough to matter.
        arr = await run_in_threadpool(_read)

        elapsed = (time.monotonic() - t0) * 1000
        ctx.diag.latency.record(elapsed)
        logger.debug(
            f"slice: computed shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes}B in {elapsed:.1f}ms"
        )

        headers = {
            "X-Shape": ",".join(str(d) for d in arr.shape),
            "X-Dtype": str(arr.dtype),
            "X-Dim-Labels": ",".join(td.dim_labels),
            # Echoed always, not only under scale_policy: the point of the
            # policy is that the caller did not choose, so the answer has to
            # say what it got -- and a caller that did choose can assert it.
            "X-Scale-Hint": ",".join(
                str(v) for v in (scale_hint or [1] * len(td.shape))
            ),
        }

        return Response(
            content=arr.tobytes(),
            media_type="application/octet-stream",
            headers=headers,
        )

    except _ClientGone:
        raise
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
    tls_fingerprint: Optional[str] = None,
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
        tls_fingerprint: SHA-256 of the leaf the flight plane serves, when TLS is
            on. The sidecar is co-located with that plane and takes this from the
            material it was handed, so it verifies the certificate on every
            connect instead of pinning whatever answers first.
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
        tls_fingerprint=tls_fingerprint,
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
    tls_fingerprint: Optional[str] = None,
) -> None:
    """Start the HTTP sidecar with uvicorn (blocking)."""
    import uvicorn

    app = create_app(
        flight_location=flight_location,
        token=token,
        cache_bytes=cache_bytes,
        cors_origins=cors_origins,
        config_path=config_path,
        tls_fingerprint=tls_fingerprint,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    # Windows: enable graceful `biopb server stop` via a sentinel-file watcher
    # that flips server.should_exit (no-op on other platforms, which use SIGTERM).
    _install_windows_shutdown_listener(server)
    server.run()
