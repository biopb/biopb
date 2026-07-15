"""The control plane's single web origin — a Starlette/uvicorn ASGI app on 8813.

This is the Layer-3 front of the de-daemonization migration
(``biopb-mcp/docs/mcp-dedaemonization-migration.md``, §6.1). It replaces the
earlier stdlib ``ThreadingHTTPServer`` control API with a real ASGI app **on the
same port**, and routes by namespace so no two upstreams share a path prefix:

- ``GET  /health``                -> ``{"control": "ok", "data_plane": {...}}`` —
                                     the control's own liveness (what
                                     ``_control_client`` and the installer poll).
                                     Bare, kept byte-for-byte.
- ``POST /api/data_plane/{ensure,stop,restart}`` -> supervisor verbs: ensure the
                                     plane is up (bounded wait), stop it, or bounce
                                     it, each returning the snapshot. ``biopb-mcp``
                                     calls ``ensure`` in place of shelling out
                                     ``biopb server start``; the dashboard drives all
                                     three. Under ``/api/`` — control *verbs about*
                                     the plane live there.
- ``GET  /api/status``            -> the control's own liveness + the data-plane
                                     snapshot + a live-session count (what the
                                     dashboard polls).
- ``GET  /api/sessions``          -> the live MCP sessions from the registry, each
                                     with its ``/session/<id>/observe`` link.
- ``GET  /`` (and every other non-API, non-proxy GET) -> the built ``web/``
                                     SPA bundle (``static_dir``). The control is
                                     the **single web origin**: it serves the
                                     bundle's static assets and falls back to
                                     ``index.html`` for deep links, so the
                                     dashboard (``/``), the dataviewer
                                     (``/viewer``), and each session's observe
                                     shell (``/session/<id>/observe``) are all
                                     React routes of that one SPA — no build-time
                                     namespacing, base ``/``. (No bundle ->
                                     API-only.)
- ``/data_plane/{api,ws,livez,...}`` is reverse-proxied to the supervised tensor
  server's HTTP sidecar — a ``Mount`` that strips its prefix, so the sidecar
  (which serves ``/api/*`` + ``/ws/render`` at its own root) needs no knowledge of
  the ``/data_plane`` namespace. The sidecar no longer serves static assets (the
  control owns the whole UI), so there is no ``/data_plane/viewer`` mount. Auth
  headers pass straight through; the sidecar re-validates.

The three ``/api/*`` namespaces therefore never collide: the control's own API is
``/api/*``, the data plane's is ``/data_plane/api/*``, and (later) each session's
is ``/session/<id>/api/*``.

Keeping the control lean (invariant I2) still holds: the ASGI stack
(starlette/uvicorn/httpx/websockets) is light and pulls in no napari/dask/Qt/
pyarrow, and the tensor server is still a *supervised subprocess* the control
never imports — the proxy reaches it over loopback like any other client.

- ``/session/<id>/observe`` serves the control's own SPA shell (the React
  ObservePage), while ``/session/<id>/api/*`` is reverse-proxied to the shim-owned
  MCP session child on its dynamic loopback port, resolved per-request from the
  filesystem registry (``biopb._config_sessions``); an unknown or dead session
  yields a clean 404 (and the dead record is pruned). Unlike the data-plane proxy,
  the ``/api/*`` hop drops both ``Host`` and ``Origin``: httpx then sets ``Host``
  to the loopback target (satisfying the child's own loopback Host guard) and the
  absent ``Origin`` passes its Origin guard — so the trusted control→child hop is
  accepted regardless of which external hostname the browser used to reach the
  control. (Rebinding/token protection for the origin as a whole is a §6.1
  follow-up, same as the data-plane proxy's.)

This module lands the namespaced origin, the data-plane API proxy, per-session
observe routing, and the control-served SPA bundle — the full Layer-3
single-origin front.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import sys
import threading
from pathlib import Path

import httpx
import uvicorn
from biopb import _agents, _algorithms, _config_sessions, _web_auth
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect as ws_connect

from ._supervisor import DataPlaneSupervisor

logger = logging.getLogger(__name__)

# Headroom (seconds) between the server's ensure wait and the client's HTTP
# timeout: the server must send its verdict BEFORE the client's urlopen times
# out, else the client treats a working-but-slow control plane as unreachable.
_RESPONSE_MARGIN = 5.0
_MIN_ENSURE_WAIT = 1.0

# Response headers we must not copy verbatim from the upstream tensor server:
# hop-by-hop headers and framing that StreamingResponse re-derives itself.
_HOP_BY_HOP = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"}
)

# The only session-child surface the control will proxy: its data API (matched by
# first path segment). The observe *page* is now the SPA shell the control serves
# itself (/session/<id>/observe -> index.html); only its /api/* data calls reach
# the child. This is an ALLOWLIST on purpose — the child also serves /mcp (the
# agent RCE transport) on the same port, and this hop strips its only auth
# (Host/Origin), so anything not explicitly allowed must be refused. A denylist
# would be unsafe: httpx normalizes dot-segments, so a traversal like
# `api/../mcp` (or its %2e%2e form, already decoded by the ASGI server) collapses
# to /mcp past a naive "startswith('mcp')" check.
_SESSION_ALLOWED_ROOTS = frozenset({"api"})

# HTTP methods that change state (so they carry a CSRF risk); safe verbs
# (GET/HEAD/OPTIONS) don't.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# The one /api/ route left unauthenticated: biopb-mcp's _control_client POSTs it
# to bring the plane up, and it is idempotent (spawns the plane the control
# already owns), so it stays open rather than forcing the mcp client to carry the
# token (the accepted #417 posture). The dangerous verbs (stop/restart) and the
# enumerating reads (status/sessions) are gated.
#
# Residual (biopb/biopb#424 item 2): this is an unauthenticated state-change on
# any token-gated deployment. It is idempotent and non-destructive, so it is
# accepted rather than gated. The exemption is *load-bearing*, not merely
# tolerated: _control_client has no way to obtain the token (the control hands
# back the plane's endpoint but never a credential), so gating this route would
# lock biopb-mcp out of a gated deployment entirely.
#
# NOTE: the old rationale for deferring #424's fix — "a local control is
# tokenless, so gating it would buy nothing in the supported modes" — no longer
# holds now that local mode accepts an optional token. On a shared host (the
# scenario that motivates a local token) loopback is reachable by every uid, so
# an untrusted local user can drive this route on a deployment the owner asked
# to gate. Dropping the exemption is blocked on giving local clients a
# credential path -- see biopb/biopb#470.
_AUTH_EXEMPT_API_PATHS = frozenset({"/api/data_plane/ensure"})

# Data-plane log tail (the dashboard /logs page polls it). Bound BOTH the returned
# line count and the bytes read off the end of the file, so tailing a multi-GB log
# never loads it whole: we seek to the final _LOG_TAIL_MAX_BYTES and keep the last
# N lines of that window.
_LOG_TAIL_DEFAULT_LINES = 200
_LOG_TAIL_MAX_LINES = 2000
_LOG_TAIL_MAX_BYTES = 512 * 1024


def _tail_file(path: Path, max_lines: int, max_bytes: int) -> tuple[list[str], bool]:
    """Return ``(lines, truncated)`` for the tail of *path*.

    Reads at most the final *max_bytes* and returns at most *max_lines* lines from
    the end. ``truncated`` is True when older content exists that was not returned
    (the byte window didn't reach the file start, or the line cap trimmed more).

    The child (tensor server) and its native libraries emit arbitrary bytes, so
    decode UTF-8 with ``errors="replace"`` rather than risk a decode error. When
    the byte window starts mid-file its first line is almost certainly a fragment,
    so drop it.
    """
    size = path.stat().st_size
    read_bytes = min(size, max_bytes)
    with path.open("rb") as f:
        if read_bytes < size:
            f.seek(size - read_bytes)
        data = f.read(read_bytes)
    partial = read_bytes < size
    lines = data.decode("utf-8", "replace").splitlines()
    if partial and lines:
        lines = lines[1:]  # drop the leading fragment
    truncated = partial
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated = True
    return lines, truncated


def _is_session_api_path(path: str) -> bool:
    """True for ``/session/<id>/<root>/...`` where ``<root>`` is a proxied session
    surface — i.e. exactly what ``session_proxy`` forwards to the child.

    Derived from the *same* ``_SESSION_ALLOWED_ROOTS`` the proxy's own gate uses,
    so the guard and the thing it guards cannot drift: any path the proxy would
    forward (including a bare ``/session/<id>/api`` with no further segment) is
    gated, and any future root added to the allowlist is covered automatically.
    Not ``/session/<id>/observe`` (the SPA shell), and not a bare
    ``/session/<id>``."""
    if not path.startswith("/session/"):
        return False
    rest = path[len("/session/") :]  # "<id>/<sub_path...>" (session ids are slash-free)
    slash = rest.find("/")
    if slash == -1:
        return False  # bare /session/<id>
    return rest[slash + 1 :].split("/")[0] in _SESSION_ALLOWED_ROOTS


class _ControlAuthMiddleware:
    """Gate the control's web API at the single origin (§6.1) — both the
    control's **own** ``/api/*`` and each session's proxied ``/session/<id>/api/*``.

    A pure-ASGI middleware (not ``BaseHTTPMiddleware``) so it touches only the
    guarded API paths and leaves the streaming ``/data_plane`` proxy, the observe
    SPA shell, and the static bundle to pass straight through untouched — wrapping
    those in ``BaseHTTPMiddleware`` would interfere with the proxies'
    ``StreamingResponse`` + background-close.

    Policy, mirroring the tensor sidecar so the two agree:

    - **Token configured** → require a valid ``Authorization: Bearer`` /
      ``X-Biopb-Token`` (401 otherwise). This is the whole point of the single
      origin: the token that already gates the data plane now also gates the
      control's stop/restart verbs and the session enumeration.
    - **No token** (local mode, all listeners loopback-bound) → require a
      **loopback Host** (421 otherwise), so a DNS-rebinding page can't drive the
      token-less origin.
    - **Unsafe method** (POST/…) → additionally refuse a forgeable cross-site
      request (403) — a token header or a same-origin ``Sec-Fetch-Site`` passes,
      a browser's cross-site POST does not (CSRF).

    ``/session/<id>/api/*`` gets the *same* policy (biopb/biopb#424): the observe
    API drives mutating kernel verbs (interrupt/restart, job cancel), the proxy
    hop deliberately strips Host/Origin toward the child (so the child cannot
    judge the browser origin itself), and session ids are guessable
    (``<timestamp>-<pid>``) — so a guessed id must not be drivable cross-site or
    via DNS-rebinding. The ``/observe`` shell (a plain SPA GET serving only the
    app bundle) stays open. ``/data_plane/*`` keeps its own gate (the sidecar
    re-validates the forwarded token), so it is not touched here.
    """

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._guarded(scope["path"]):
            get = Headers(scope=scope).get
            denial = self._deny(scope["method"], get)
            if denial is not None:
                await denial(scope, receive, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _guarded(path: str) -> bool:
        if path in _AUTH_EXEMPT_API_PATHS:
            return False
        if path.startswith("/api/"):
            return True
        return _is_session_api_path(path)

    def _deny(self, method: str, get: _web_auth.HeaderGetter) -> Response | None:
        """The response to send if the request is refused, else ``None``."""
        if self._token:
            if not _web_auth.token_valid(get, self._token):
                return JSONResponse(
                    {"error": "invalid or missing token"}, status_code=401
                )
        elif not _web_auth.host_is_loopback(get("host")):
            return JSONResponse({"error": "invalid Host header"}, status_code=421)
        if method in _UNSAFE_METHODS and _web_auth.is_forgeable_cross_site(get):
            return JSONResponse(
                {"error": "cross-site request refused"}, status_code=403
            )
        return None


def _bounded_ensure_wait(ensure_timeout: float, client_timeout: float) -> float:
    """How long ``/data_plane/ensure`` should wait for the plane to come up.

    Bounded strictly below the client's HTTP timeout (by ``_RESPONSE_MARGIN``) so
    the server always answers first; also capped by the server's own configured
    ``ensure_timeout``, and floored at ``_MIN_ENSURE_WAIT``. A missing/invalid
    client hint (``<= 0``) falls back to the configured ``ensure_timeout`` (the
    client then relies on its own urlopen timeout being generous).
    """
    if client_timeout <= 0:
        return ensure_timeout
    return max(_MIN_ENSURE_WAIT, min(ensure_timeout, client_timeout - _RESPONSE_MARGIN))


def _loopback_url(host: str, port: int, scheme: str = "http") -> str:
    """A loopback-reachable base URL for a server that may bind a wildcard.

    A tensor server bound to ``0.0.0.0`` / ``::`` is reached over its loopback
    address; anything else (an explicit host) is used as given. An IPv6 literal
    is bracketed so the ``:port`` suffix stays unambiguous. Mirrors the
    supervisor's liveness-probe convention.
    """
    reachable = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(host, host)
    if ":" in reachable:  # IPv6 literal must be bracketed in a URL (e.g. [::1])
        reachable = f"[{reachable}]"
    return f"{scheme}://{reachable}:{port}"


# How long each per-session kernel probe may take. The dashboard polls the
# session list every few seconds and the probes run concurrently, so this is
# kept short: a slow or wedged child yields "unknown" rather than stalling the
# whole list.
_KERNEL_PROBE_TIMEOUT = 0.6

# Timeouts for the reverse proxies into the sidecar / session children. Not
# ``None`` (biopb#420): a wedged upstream that accepts the connection but never
# answers must fail eventually, not hang the request forever. The ``read`` bound
# is per read-event, not total, and is set generously — every upstream buffers
# its whole response before sending (no long-poll / SSE / chunked-with-gaps path,
# so a large slice/render streams without inter-chunk stalls), so 300s only trips
# on a genuinely stuck upstream, never on legitimately large or slow-computed
# transfers. ``connect``/``write``/``pool`` are short since every hop is loopback.
_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)


def _kernel_state(health: dict) -> str:
    """Map a session child's ``/api/status`` health dict to a dashboard kernel
    state.

    The kernel is the heavy component and starts on demand, so the useful bit is
    attached-or-not. A *live* kernel always reports its live state (a stale
    ``start_error`` from an earlier, since-recovered attempt never masks it):
    ``ready`` (booted past its bootstrap probe), ``busy`` (executing), or
    ``starting`` (process up, not yet ready). A kernel that is not alive is
    ``error`` if it failed / died, else ``none`` (never started this session).
    """
    if health.get("alive"):
        if not health.get("ready"):
            return "starting"
        return "busy" if health.get("busy") else "ready"
    if health.get("start_error") or health.get("dead"):
        return "error"
    return "none"


async def _probe_kernel(client: httpx.AsyncClient, rec: dict) -> str:
    """Best-effort "is a kernel attached?" for one session.

    A single cheap loopback GET to the child's ``/api/status`` — which returns
    ``KernelHost.health()`` with no kernel round-trip and whose ``api`` observe
    root the control already proxies. Never raises: a missing port, an
    unreachable/slow child, a non-200, or unparseable JSON all degrade to
    ``"unknown"`` so the session list is never blocked or truncated by a probe.
    httpx sets ``Host`` from the target (satisfying the child's loopback guard)
    and sends no ``Origin`` (passing its Origin guard), like the session proxy.
    """
    port = rec.get("port")
    if not port:
        return "unknown"
    url = _loopback_url(rec.get("host", "127.0.0.1"), port) + "/api/status"
    try:
        resp = await client.get(url, timeout=_KERNEL_PROBE_TIMEOUT)
        if resp.status_code != 200:
            return "unknown"
        return _kernel_state(resp.json())
    except Exception:  # noqa: BLE001 - a probe is decorative; never fail the list
        return "unknown"


def build_app(
    supervisor: DataPlaneSupervisor,
    ensure_timeout: float,
    data_web_url: str,
    token: str | None = None,
    static_dir: str | Path | None = None,
) -> Starlette:
    """Build the control-plane ASGI app.

    ``data_web_url`` is the loopback base URL of the supervised tensor server's
    HTTP sidecar; the ``/data_plane`` namespace reverse-proxies there. ``token``
    is the data-plane access token (``None`` in local mode, where every listener
    is loopback-bound); the ``/api/*`` gate enforces it when set, else falls back
    to a loopback Host check. ``static_dir`` is the built ``web/`` bundle (``web/packages/app/
    dist``); when present the control serves it at its root as the single web
    origin — the dashboard (``/``), the dataviewer (``/viewer``), and each
    session's observe shell (``/session/<id>/observe``) are all React routes of
    that one SPA. Split out from :func:`serve_control_api` so it is unit-testable
    against a fake upstream without binding uvicorn.
    """
    ws_base = data_web_url.replace("http://", "ws://", 1).replace(
        "https://", "wss://", 1
    )

    # The built SPA bundle the control serves at its root (None / missing ->
    # API-only: the control still answers /health + /api/* + the proxies, but
    # serves no web UI). Resolved once; index.html is the SPA shell every
    # non-API, non-proxy GET falls back to.
    web_root = Path(static_dir) if static_dir else None
    if web_root is not None and not (web_root / "index.html").is_file():
        logger.warning("web bundle not found at %s; serving API only", web_root)
        web_root = None

    # One pooled client to the sidecar for the process lifetime. Held in a
    # closure (not ``app.state``) because the proxy runs inside a *mounted*
    # sub-app whose ``request.app`` is the sub-app, not this one -- ``app.state``
    # would read the wrong app's state. Closed by the app lifespan below. The
    # generous ``_PROXY_TIMEOUT`` (not ``None``) keeps large slice responses
    # flowing while ensuring a wedged sidecar fails cleanly (biopb#420).
    proxy_client = httpx.AsyncClient(base_url=data_web_url, timeout=_PROXY_TIMEOUT)

    # A second pooled client for the session proxy. No base_url: each session's
    # target is a *different* dynamic loopback port resolved per-request from the
    # registry, so the proxy builds absolute URLs (httpx pools connections per
    # host:port automatically). Also closed by the lifespan below.
    session_client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)

    # --- control-owned endpoints (sync: they take the supervisor lock and do a
    # blocking TCP liveness probe, so Starlette runs them in its threadpool) --- #

    def health(_request: Request) -> JSONResponse:
        # `auth_required` is the SPA's public probe: the browser bundle + this
        # endpoint stay unauthenticated always, and the app reads this to decide
        # whether to gate itself behind the unlock page. It tracks the *token*,
        # not the network mode: always true in remote (which requires one), and
        # true in local mode too when an optional token was supplied.
        return JSONResponse(
            {
                "control": "ok",
                "auth_required": token is not None,
                "data_plane": supervisor.snapshot(),
            }
        )

    def data_plane_ensure(request: Request) -> JSONResponse:
        # The client passes ?client_timeout=<its HTTP timeout>; cap our wait
        # below it so we return a verdict before the client gives up (and wrongly
        # treats a slow-but-working control plane as unreachable).
        try:
            client_timeout = float(request.query_params.get("client_timeout", "0"))
        except ValueError:
            client_timeout = 0.0
        wait = _bounded_ensure_wait(ensure_timeout, client_timeout)
        # ensure()/_spawn_locked count a spawn failure toward the backoff and do
        # not raise, but wrap defensively so any unexpected error still returns a
        # clean JSON verdict (with the snapshot reflecting the counted failure)
        # rather than an unhandled 500.
        try:
            supervisor.ensure()
            supervisor.wait_until_up(wait)
            return JSONResponse({"data_plane": supervisor.snapshot()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("data_plane/ensure failed")
            return JSONResponse(
                {"error": str(exc), "data_plane": supervisor.snapshot()},
                status_code=500,
            )

    def data_plane_stop(_request: Request) -> JSONResponse:
        # Full teardown of the data plane (want=False): the control stays up, but
        # its supervised child is stopped and won't be respawned until an ensure.
        try:
            supervisor.stop()
            return JSONResponse({"data_plane": supervisor.snapshot()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("data_plane/stop failed")
            return JSONResponse(
                {"error": str(exc), "data_plane": supervisor.snapshot()},
                status_code=500,
            )

    def data_plane_restart(request: Request) -> JSONResponse:
        # Bounce the plane: stop() (want=False, so a racing supervision tick backs
        # off instead of seeing the down port as a conflict) then ensure() flips
        # want back on and spawns a fresh child, bounded like /ensure so we answer
        # before the client's HTTP timeout.
        try:
            client_timeout = float(request.query_params.get("client_timeout", "0"))
        except ValueError:
            client_timeout = 0.0
        wait = _bounded_ensure_wait(ensure_timeout, client_timeout)
        try:
            supervisor.stop()
            supervisor.ensure()
            supervisor.wait_until_up(wait)
            return JSONResponse({"data_plane": supervisor.snapshot()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("data_plane/restart failed")
            return JSONResponse(
                {"error": str(exc), "data_plane": supervisor.snapshot()},
                status_code=500,
            )

    def api_data_plane_logs(request: Request) -> JSONResponse:
        # The dashboard /logs page polls this: the tail of the data-plane
        # subprocess's stdout/stderr log (the file the supervisor writes the tensor
        # server to). Read is bounded in both lines and bytes (see _tail_file), so
        # tailing a huge log stays cheap; no-store so each poll sees fresh output.
        # Never raises -- a bad read degrades to an error field, not a 500 trace.
        try:
            n = int(request.query_params.get("lines", _LOG_TAIL_DEFAULT_LINES))
        except (TypeError, ValueError):
            n = _LOG_TAIL_DEFAULT_LINES
        n = max(1, min(n, _LOG_TAIL_MAX_LINES))
        headers = {"Cache-Control": "no-store"}
        path = supervisor.log_path
        if path is None:
            return JSONResponse(
                {
                    "path": None,
                    "exists": False,
                    "lines": [],
                    "truncated": False,
                    "note": "data plane logs to the control's stderr "
                    "(no log file configured)",
                },
                headers=headers,
            )
        try:
            if not path.exists():
                return JSONResponse(
                    {
                        "path": str(path),
                        "exists": False,
                        "lines": [],
                        "truncated": False,
                    },
                    headers=headers,
                )
            lines, truncated = _tail_file(path, n, _LOG_TAIL_MAX_BYTES)
            return JSONResponse(
                {
                    "path": str(path),
                    "exists": True,
                    "size": path.stat().st_size,
                    "lines": lines,
                    "truncated": truncated,
                },
                headers=headers,
            )
        except OSError as exc:
            logger.info("data plane log read failed: %s", exc)
            return JSONResponse(
                {
                    "path": str(path),
                    "exists": False,
                    "lines": [],
                    "error": f"could not read log: {exc}",
                },
                status_code=500,
                headers=headers,
            )

    def api_status(_request: Request) -> JSONResponse:
        # What the dashboard polls: the control is up (it answered), the data
        # plane's supervisor snapshot, and how many sessions are live. Sync (the
        # snapshot probes the port and list_sessions() touches the filesystem), so
        # Starlette runs it in the threadpool.
        return JSONResponse(
            {
                "control": "ok",
                "data_plane": supervisor.snapshot(),
                "sessions": len(_config_sessions.list_sessions()),
            }
        )

    async def api_sessions(_request: Request) -> JSONResponse:
        # The live MCP sessions, newest first, projected to what the dashboard
        # needs — the id, when it started, its loopback port, the control-relative
        # observe link, and a best-effort "kernel" state (the heavy on-demand
        # component, probed concurrently over one cheap GET each; see
        # _probe_kernel). list_sessions() self-heals (prunes dead/reused records)
        # on read, so a stale session never lingers on the page. Async so the
        # per-session probes fan out concurrently rather than serializing.
        records = [
            rec for rec in _config_sessions.list_sessions() if rec.get("session_id")
        ]
        kernels = await asyncio.gather(
            *(_probe_kernel(session_client, rec) for rec in records)
        )
        sessions = [
            {
                "session_id": rec["session_id"],
                "started_at": rec.get("started_at"),
                "port": rec.get("port"),
                "observe_url": f"/session/{rec['session_id']}/observe",
                "kernel": kernel,
            }
            for rec, kernel in zip(records, kernels, strict=True)
        ]
        return JSONResponse({"sessions": sessions})

    def api_agents(_request: Request) -> JSONResponse:
        # The supported MCP clients and whether biopb is registered with each.
        # Reads are subprocess-free (biopb._agents), so the dashboard can poll
        # this without spawning anything; still sync (filesystem), so Starlette
        # runs it in the threadpool.
        try:
            return JSONResponse({"agents": _agents.statuses()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("api/agents failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    def _agent_action(request: Request, action) -> JSONResponse:
        # Register/unregister biopb with one client, returning its fresh status.
        # A bad request (unknown client, unparseable client config) is the
        # caller's fault -> 400; anything else -> 500. Both write user config
        # files (Claude Code via its CLI), so these are token-gated /api/* verbs.
        agent_id = request.path_params["agent_id"]
        try:
            return JSONResponse({"agent": action(agent_id)})
        except _agents.AgentError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("agent %s failed", getattr(action, "__name__", "action"))
            return JSONResponse({"error": str(exc)}, status_code=500)

    def agent_register(request: Request) -> JSONResponse:
        return _agent_action(request, _agents.register)

    def agent_unregister(request: Request) -> JSONResponse:
        return _agent_action(request, _agents.unregister)

    def api_algorithms(_request: Request) -> JSONResponse:
        # The configured algorithm-plane servers (biopb.image ProcessImage
        # servicers listed in the biopb-mcp config) with a live health + ops
        # probe. Read-only inspection — no lifecycle control (that is Layer 4).
        # Sync: statuses() reads a config file and makes blocking gRPC calls (run
        # concurrently, bounded by one probe timeout), so Starlette runs it in the
        # threadpool. Polled on demand (a dashboard button), not on the interval,
        # because it dials external servers.
        try:
            return JSONResponse({"servers": _algorithms.statuses()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("api/algorithms failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    def api_mcp_config(_request: Request) -> JSONResponse:
        # The biopb-mcp settings editor's backing read: the raw on-disk config +
        # its path + the JSON Schema (labels/help/bounds), mirroring the tensor
        # sidecar's GET /api/config so the same schema-driven admin UI renders it.
        # The control OWNS this because the config is global (~/.config/biopb/
        # mcp-config.json) while mcp sessions are ephemeral/dynamic-port -- none of
        # them owns the file. biopb_mcp is soft-imported (only for the schema): the
        # lean control does not hard-depend on it (invariant I2), but a real biopb
        # deployment always co-installs it. mcp_config_path lives in core biopb.
        from biopb._config_location import mcp_config_path

        try:
            from biopb_mcp._config_schema import build_mcp_config_schema
        except Exception as exc:  # noqa: BLE001 - biopb-mcp not installed here
            return JSONResponse(
                {"error": f"biopb-mcp is not installed: {exc}"}, status_code=501
            )
        p = mcp_config_path()
        raw: dict = {}
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            except (OSError, ValueError) as exc:
                return JSONResponse(
                    {"error": f"config on disk is unreadable: {exc}"}, status_code=500
                )
        # no-store: a config editor must always see the live file, never a cached
        # GET (a stale empty {} cached before the file was populated would render
        # the wrong config and clobber it on save).
        return JSONResponse(
            {"path": str(p), "config": raw, "schema": build_mcp_config_schema()},
            headers={"Cache-Control": "no-store"},
        )

    async def api_mcp_config_save(request: Request) -> JSONResponse:
        # Validate + write the biopb-mcp config. Validation reuses biopb-mcp's own
        # _CONSTRAINTS table (the exact rules it clamps to at load time), so "the
        # form accepted it" == "biopb-mcp will accept it" with no jsonschema
        # dependency in the lean control. Changes apply to the NEXT session (each
        # session reads config fresh at bootstrap), so there is no server to
        # restart -- unlike the data plane.
        try:
            from biopb_mcp._config import _CONSTRAINTS, _SECTION_CLASSES, save_config
        except Exception as exc:  # noqa: BLE001 - biopb-mcp not installed here
            return JSONResponse(
                {"error": f"biopb-mcp is not installed: {exc}"}, status_code=501
            )
        from biopb._config_location import mcp_config_path

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {"detail": "Request body is not valid JSON"}, status_code=422
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {"detail": "Config body must be a JSON object"}, status_code=422
            )

        errors: list[dict] = []
        for section, cls in _SECTION_CLASSES.items():
            sec = body.get(section)
            if not isinstance(sec, dict):
                continue
            for field, constraint in _CONSTRAINTS.get(cls.__name__, {}).items():
                if field in sec and not constraint.ok(sec[field]):
                    errors.append(
                        {
                            "path": [section, field],
                            "message": f"expected {constraint.describe()}",
                        }
                    )
        # Cross-field: the health-poll backoff must not invert (min > max).
        tensor = body.get("tensor")
        if isinstance(tensor, dict):
            lo, hi = (
                tensor.get("health_poll_min_interval"),
                tensor.get("health_poll_max_interval"),
            )
            if (
                isinstance(lo, (int, float))
                and isinstance(hi, (int, float))
                and lo > hi
            ):
                errors.append(
                    {
                        "path": ["tensor", "health_poll_min_interval"],
                        "message": "must be <= health_poll_max_interval",
                    }
                )
        if errors:
            errors.sort(key=lambda d: d["path"])
            return JSONResponse(
                {"detail": "Config failed validation", "errors": errors},
                status_code=422,
            )
        try:
            save_config(body)
        except OSError as exc:
            return JSONResponse(
                {"error": f"could not write config: {exc}"}, status_code=500
            )
        return JSONResponse({"saved": True, "path": str(mcp_config_path())})

    def _serve_shell() -> Response:
        # The SPA shell (index.html) every non-API GET falls back to; the React
        # router then renders the right surface for the URL. web_root is checked
        # by the caller, so index.html exists here.
        assert web_root is not None
        return FileResponse(web_root / "index.html")

    async def spa(request: Request) -> Response:
        # Catch-all for the single web origin: serve a real static file from the
        # bundle when the path names one (/assets/<hash>, /favicon.ico, …), else
        # the SPA shell so a deep link like /viewer or /admin boots the router.
        # Registered LAST, after every API route and proxy mount, so it never
        # shadows them.
        if web_root is None:
            return JSONResponse({"error": "web bundle not installed"}, status_code=404)
        rel = request.path_params["path"].lstrip("/")
        if rel:
            candidate = (web_root / rel).resolve()
            # Contain traversal: only serve files that resolve inside web_root.
            if web_root.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
        return _serve_shell()

    # --- reverse proxy into the tensor server's HTTP sidecar ---------------- #
    # Handlers forward the *mount-relative* path (``Mount`` has already stripped
    # the ``/data_plane[/viewer]`` prefix into ``path_params``), so the sidecar
    # always sees a root-relative path regardless of which mount matched.

    async def proxy(request: Request) -> Response:
        target = "/" + request.path_params["path"]
        # Append the query only when present -- an empty one would render a bare
        # trailing "?" that changes the path the sidecar sees.
        if request.url.query:
            target = f"{target}?{request.url.query}"
        # Drop Host so httpx sets it from base_url; forward everything else
        # (Authorization / X-Biopb-Token pass through, the sidecar re-validates).
        headers = [(k, v) for k, v in request.headers.raw if k.lower() != b"host"]
        # Request bodies here are small JSON (e.g. POST /api/slice params); read
        # fully so GETs carry no chunked body. Responses (images) are streamed.
        body = await request.body()
        upstream = proxy_client.build_request(
            request.method, target, headers=headers, content=body
        )
        try:
            resp = await proxy_client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            # Any upstream/transport failure -- refused connect, an upstream that
            # accepts then dies mid-response (RemoteProtocolError/ReadError), or a
            # read/connect timeout on a wedged sidecar -- is a gateway error, not a
            # control-plane bug: surface a clean 502, never a 500 traceback
            # (biopb#420). A failure *after* the headers stream (in aiter_raw) can't
            # be turned into a 502 anymore, but the timeout still bounds the hang.
            logger.info("data plane proxy to %s failed: %s", target, exc)
            return JSONResponse({"error": "data plane not reachable"}, status_code=502)
        # HTTP headers are latin-1 on the wire (RFC 9110 / ASGI). A header value
        # may carry a legitimate high byte (e.g. a non-ASCII Content-Disposition
        # filename); decoding it as UTF-8 would raise and 500 the proxy. latin-1
        # is total and round-trips -- Starlette re-encodes response headers as
        # latin-1 too.
        resp_headers = [
            (k, v)
            for k, v in resp.headers.raw
            if k.decode("latin-1").lower() not in _HOP_BY_HOP
        ]
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={k.decode("latin-1"): v.decode("latin-1") for k, v in resp_headers},
            background=BackgroundTask(resp.aclose),
        )

    async def ws_proxy(client_ws: WebSocket) -> None:
        # The dataviewer's render channel; the sidecar serves it at /ws/render.
        # Token travels as a ?token= query param (browsers can't set WS headers),
        # so forwarding the query authenticates.
        await client_ws.accept()
        target = ws_base + "/ws/render"
        if client_ws.url.query:
            target += "?" + client_ws.url.query
        try:
            async with ws_connect(target, max_size=None) as upstream:
                await _pump_websocket(client_ws, upstream)
        except Exception as exc:  # noqa: BLE001 - upstream down / handshake failed
            logger.info("ws proxy to %s failed: %s", target, exc)
        finally:
            with contextlib.suppress(Exception):
                await client_ws.close()

    async def session_proxy(request: Request) -> Response:
        # The outer Mount captured {session_id}; the inner catch-all captured the
        # rest into {path} (both survive in path_params). Resolve the session to a
        # live loopback target via the registry — an unknown/dead one is a clean
        # 404 (and the dead record is pruned by resolve()).
        session_id = request.path_params["session_id"]
        sub_path = request.path_params["path"]
        # Allowlist the session data API only (§6.1) — the observe page itself is
        # the control-served SPA shell (session_observe below), so only /api/*
        # proxies here. The child's /mcp agent transport is deliberately off this
        # origin — agents reach it directly on the child's own loopback port
        # (stdio shim bridge / `biopb mcp view`), never via the control — and this
        # hop strips /mcp's entire auth (Host/Origin), so exposing it would be an
        # RCE hole on the public origin. Require an allowed first segment AND
        # reject any parent-traversal, so no path (raw, encoded, or dot-collapsed
        # by httpx) can escape /api/* into /mcp.
        segments = sub_path.split("/")
        if segments[0] not in _SESSION_ALLOWED_ROOTS or ".." in segments:
            return JSONResponse({"error": "not found"}, status_code=404)
        rec = _config_sessions.resolve(session_id)
        if rec is None:
            return JSONResponse(
                {"error": f"session {session_id!r} not found or ended"},
                status_code=404,
            )
        base = _loopback_url(rec.get("host", "127.0.0.1"), rec["port"])
        target = base + "/" + sub_path
        if request.url.query:
            target = f"{target}?{request.url.query}"
        # Drop Host AND Origin: httpx sets Host from the target (127.0.0.1:<port>,
        # matching the child's loopback Host allowlist) and an absent Origin
        # passes the child's Origin guard, so the trusted control->child hop is
        # accepted whatever external host the browser used. Everything else
        # forwards verbatim.
        headers = [
            (k, v)
            for k, v in request.headers.raw
            if k.lower() not in (b"host", b"origin")
        ]
        body = await request.body()
        upstream = session_client.build_request(
            request.method, target, headers=headers, content=body
        )
        try:
            resp = await session_client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            # Same as the data-plane proxy: any upstream/transport failure or
            # timeout on a wedged session child is a clean 502, not a 500
            # traceback (biopb#420).
            logger.info("session proxy to %s failed: %s", target, exc)
            return JSONResponse({"error": "session not reachable"}, status_code=502)
        resp_headers = [
            (k, v)
            for k, v in resp.headers.raw
            if k.decode("latin-1").lower() not in _HOP_BY_HOP
        ]
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={k.decode("latin-1"): v.decode("latin-1") for k, v in resp_headers},
            background=BackgroundTask(resp.aclose),
        )

    async def session_observe(request: Request) -> Response:
        # The observe *page* is the control-served SPA shell (React ObservePage);
        # only its /api/* data calls proxy to the child (session_proxy above).
        # Resolve the session first so an unknown/dead id is a clean 404 rather
        # than a shell wired to a session that no longer exists.
        if web_root is None:
            return JSONResponse({"error": "web bundle not installed"}, status_code=404)
        session_id = request.path_params["session_id"]
        if _config_sessions.resolve(session_id) is None:
            return JSONResponse(
                {"error": f"session {session_id!r} not found or ended"},
                status_code=404,
            )
        return _serve_shell()

    # One sub-app proxying to the sidecar root, mounted at /data_plane (the data
    # plane's /api, /ws/render, health). It strips its prefix, so the sidecar
    # needs no knowledge of the namespace. (The dataviewer's static assets are no
    # longer proxied out of the sidecar — the control serves the whole SPA itself,
    # so there is no /data_plane/viewer mount.) /ws/render must precede the
    # catch-all.
    sidecar = Starlette(
        routes=[
            WebSocketRoute("/ws/render", ws_proxy),
            Route(
                "/{path:path}",
                proxy,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
        ]
    )

    # A session sub-app mounted under /session/{session_id}: /observe serves the
    # control's SPA shell, everything else (/api/*) proxies to the child.
    session_app = Starlette(
        routes=[
            Route("/observe", session_observe, methods=["GET"]),
            Route(
                "/{path:path}",
                session_proxy,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
        ]
    )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/sessions", api_sessions, methods=["GET"]),
        Route("/api/data_plane/ensure", data_plane_ensure, methods=["POST"]),
        Route("/api/data_plane/stop", data_plane_stop, methods=["POST"]),
        Route("/api/data_plane/restart", data_plane_restart, methods=["POST"]),
        Route("/api/data_plane/logs", api_data_plane_logs, methods=["GET"]),
        Route("/api/agents", api_agents, methods=["GET"]),
        Route("/api/agents/{agent_id}/register", agent_register, methods=["POST"]),
        Route("/api/agents/{agent_id}/unregister", agent_unregister, methods=["POST"]),
        Route("/api/algorithms", api_algorithms, methods=["GET"]),
        Route("/api/mcp_config", api_mcp_config, methods=["GET"]),
        Route("/api/mcp_config", api_mcp_config_save, methods=["PUT"]),
        Mount("/data_plane", sidecar),
        # Per-session observe: /session/<id>/observe (SPA shell) + /session/<id>/
        # api/* (proxied). The {session_id} convertor is slash-free (session ids
        # are), so it stops at the first slash and the remainder falls to the
        # session sub-app.
        Mount("/session/{session_id}", session_app),
        # The single web origin's catch-all: static asset or SPA shell. LAST so
        # every API route and proxy mount above wins first.
        Route(
            "/{path:path}",
            spa,
            methods=["GET", "HEAD"],
        ),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        try:
            yield
        finally:
            await proxy_client.aclose()
            await session_client.aclose()

    # The /api/* auth gate wraps the whole app but acts only on /api/* (pure ASGI,
    # so the streaming proxies pass through untouched).
    middleware = [Middleware(_ControlAuthMiddleware, token=token)]
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


async def _pump_websocket(client_ws: WebSocket, upstream) -> None:
    """Bidirectionally shuttle frames between the browser and the tensor server.

    Runs both directions concurrently and tears both down as soon as either
    side closes, so neither a client disconnect nor an upstream close leaks a
    half-open pump task.
    """

    async def client_to_upstream() -> None:
        while True:
            msg = await client_ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("text") is not None:
                await upstream.send(msg["text"])
            elif msg.get("bytes") is not None:
                await upstream.send(msg["bytes"])

    async def upstream_to_client() -> None:
        async for message in upstream:
            if isinstance(message, str):
                await client_ws.send_text(message)
            else:
                await client_ws.send_bytes(message)

    tasks = [
        asyncio.ensure_future(client_to_upstream()),
        asyncio.ensure_future(upstream_to_client()),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _ControlServer:
    """Shutdown handle the caller holds for teardown.

    Wraps the :class:`uvicorn.Server` (run in a background thread) so ``_run``'s
    teardown keeps its existing ``server.shutdown()`` call. Signalling
    ``should_exit`` unwinds uvicorn's serve loop from another thread.
    """

    def __init__(self, server: uvicorn.Server) -> None:
        self._server = server

    def shutdown(self) -> None:
        self._server.should_exit = True


def serve_control_api(
    host: str,
    port: int,
    supervisor: DataPlaneSupervisor,
    ensure_timeout: float,
    data_web_url: str | None = None,
) -> tuple[_ControlServer, threading.Thread]:
    """Start the control-plane web origin on ``host:port`` in a background thread.

    Binds the listening socket **eagerly, in the caller's thread**, so a port
    clash surfaces here (a control plane already running) instead of in a
    detached uvicorn thread — preserving the old stdlib server's fail-fast
    contract. ``data_web_url`` defaults to the supervised tensor server's sidecar
    (loopback of its configured web host/port); pass it explicitly in tests.

    Returns ``(server, thread)``; the caller stops it with ``server.shutdown()``.
    """
    spec = supervisor._spec
    if data_web_url is None:
        data_web_url = _loopback_url(spec.web_host, spec.web_port)

    # The data-plane token gates the control's own /api/* too (single origin,
    # §6.1). None in local mode -> the gate falls back to a loopback Host check
    # instead.
    app = build_app(
        supervisor,
        ensure_timeout,
        data_web_url,
        token=spec.token,
        static_dir=spec.static_dir,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform == "win32":
        # On Windows SO_REUSEADDR lets a *second* bind to the same port SUCCEED and
        # then delivers incoming connections to one of the sockets nondeterministically
        # -- so it would defeat the single-owner guarantee a concurrent `control start`
        # relies on (you could end up with two live controls on 8813). SO_EXCLUSIVEADDRUSE
        # instead makes the second bind fail with EADDRINUSE, which is the behavior POSIX
        # SO_REUSEADDR already gives here (it only reuses a TIME_WAIT port, never a live
        # bind). So the eager bind stays the true arbiter of "one control per port" on
        # both platforms.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))  # raises OSError on clash -> caller reports it
    sock.listen(128)

    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        # The modern (non-legacy) websockets server impl, so the /ws/render proxy
        # server side doesn't pull in websockets.legacy (deprecated, dropped in a
        # future websockets release). Our upstream WS client already uses the
        # asyncio API (ws_connect above).
        ws="websockets-sansio",
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        # uvicorn skips signal-handler installation off the main thread, so this
        # is safe to run in a daemon thread alongside the supervision loop.
        server.run(sockets=[sock])

    thread = threading.Thread(target=_run, name="control-api", daemon=True)
    thread.start()
    logger.info("Control plane origin listening on http://%s:%d", host, port)
    return _ControlServer(server), thread
