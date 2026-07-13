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
- ``GET  /``                      -> the control's own **buildless dashboard** — a
                                     single embedded HTML+vanilla-JS page (the
                                     ``_observe.py`` pattern, no Vite/npm build) that
                                     polls the two APIs above, exposes the data-plane
                                     controls, and links to the dataviewer + each
                                     session's observe page.
- ``/data_plane/viewer[/*]`` and ``/data_plane/{api,ws,livez,...}`` are
  reverse-proxied to the supervised tensor server's HTTP sidecar. Each is a
  ``Mount`` that strips its prefix, so the sidecar (which serves ``/`` +
  ``/api/*`` + ``/ws/render`` at its own root) needs no knowledge of the
  ``/data_plane`` namespace. Auth headers pass straight through; the sidecar
  re-validates.

The three ``/api/*`` namespaces therefore never collide: the control's own API is
``/api/*``, the data plane's is ``/data_plane/api/*``, and (later) each session's
is ``/session/<id>/api/*``.

Keeping the control lean (invariant I2) still holds: the ASGI stack
(starlette/uvicorn/httpx/websockets) is light and pulls in no napari/dask/Qt/
pyarrow, and the tensor server is still a *supervised subprocess* the control
never imports — the proxy reaches it over loopback like any other client.

- ``/session/<id>/*`` is reverse-proxied to the shim-owned MCP session child on
  its dynamic loopback port, resolved per-request from the filesystem registry
  (``biopb._config_sessions``). The observe UI (``/session/<id>/observe`` +
  ``/session/<id>/api/*``) is the consumer; an unknown or dead session yields a
  clean 404 (and the dead record is pruned). Unlike the data-plane proxy, this
  hop drops both ``Host`` and ``Origin``: httpx then sets ``Host`` to the
  loopback target (satisfying the child's own loopback Host guard) and the absent
  ``Origin`` passes its Origin guard — so the trusted control→child hop is
  accepted regardless of which external hostname the browser used to reach the
  control. (Rebinding/token protection for the origin as a whole is a §6.1
  follow-up, same as the data-plane proxy's.)

This module lands the namespaced origin, the data-plane proxy, per-session observe
routing, and the control-owned dashboard — the full Layer-3 single-origin front.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
import threading

import httpx
import uvicorn
from biopb import _agents, _algorithms, _config_sessions, _web_auth
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
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

# The only session-child surfaces the control will proxy: the observe page and
# its data API (matched by first path segment). This is an ALLOWLIST on purpose —
# the child also serves /mcp (the agent RCE transport) on the same port, and this
# hop strips its only auth (Host/Origin), so anything not explicitly allowed must
# be refused. A denylist would be unsafe: httpx normalizes dot-segments, so a
# traversal like `api/../mcp` (or its %2e%2e form, already decoded by the ASGI
# server) collapses to /mcp past a naive "startswith('mcp')" check.
_SESSION_ALLOWED_ROOTS = frozenset({"observe", "api"})

# HTTP methods that change state (so they carry a CSRF risk); safe verbs
# (GET/HEAD/OPTIONS) don't.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# The one /api/ route left unauthenticated: biopb-mcp's _control_client POSTs it
# to bring the plane up, and it is idempotent (spawns the plane the control
# already owns), so it stays open rather than forcing the mcp client to carry the
# token (the accepted #417 posture). The dangerous verbs (stop/restart) and the
# enumerating reads (status/sessions) are gated.
_AUTH_EXEMPT_API_PATHS = frozenset({"/api/data_plane/ensure"})


class _ControlAuthMiddleware:
    """Gate the control's **own** API (``/api/*``) at the single origin (§6.1).

    A pure-ASGI middleware (not ``BaseHTTPMiddleware``) so it touches *only*
    ``/api/*`` requests and leaves the streaming ``/data_plane`` and ``/session``
    proxies to pass straight through untouched — wrapping those in
    ``BaseHTTPMiddleware`` would interfere with their ``StreamingResponse`` +
    background-close.

    Policy, mirroring the tensor sidecar so the two agree:

    - **Token configured** → require a valid ``Authorization: Bearer`` /
      ``X-Biopb-Token`` (401 otherwise). This is the whole point of the single
      origin: the token that already gates the data plane now also gates the
      control's stop/restart verbs and the session enumeration.
    - **No token** (the all-localhost dev-bypass) → require a **loopback Host**
      (421 otherwise), so a DNS-rebinding page can't drive the token-less origin.
    - **Unsafe method** (POST/…) → additionally refuse a forgeable cross-site
      request (403) — a token header or a same-origin ``Sec-Fetch-Site`` passes,
      a browser's cross-site POST does not (CSRF).

    ``/data_plane/*`` keeps its own gate (the sidecar re-validates the forwarded
    token) and ``/session/*`` is deferred to step 3 (observe must learn to send
    the token first); neither is touched here.
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
        return path.startswith("/api/") and path not in _AUTH_EXEMPT_API_PATHS

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
) -> Starlette:
    """Build the control-plane ASGI app.

    ``data_web_url`` is the loopback base URL of the supervised tensor server's
    HTTP sidecar; the ``/data_plane`` namespace reverse-proxies there. ``token``
    is the data-plane access token (``None`` in the all-localhost dev-bypass
    case); the ``/api/*`` gate enforces it when set, else falls back to a loopback
    Host check. Split out from :func:`serve_control_api` so it is unit-testable
    against a fake upstream without binding uvicorn.
    """
    ws_base = data_web_url.replace("http://", "ws://", 1).replace(
        "https://", "wss://", 1
    )

    # One pooled client to the sidecar for the process lifetime; no timeout so
    # large slice responses and long-poll probes are never cut off. Held in a
    # closure (not ``app.state``) because the proxy runs inside a *mounted*
    # sub-app whose ``request.app`` is the sub-app, not this one -- ``app.state``
    # would read the wrong app's state. Closed by the app lifespan below.
    proxy_client = httpx.AsyncClient(base_url=data_web_url, timeout=None)

    # A second pooled client for the session proxy. No base_url: each session's
    # target is a *different* dynamic loopback port resolved per-request from the
    # registry, so the proxy builds absolute URLs (httpx pools connections per
    # host:port automatically). Also closed by the lifespan below.
    session_client = httpx.AsyncClient(timeout=None)

    # --- control-owned endpoints (sync: they take the supervisor lock and do a
    # blocking TCP liveness probe, so Starlette runs them in its threadpool) --- #

    def health(_request: Request) -> JSONResponse:
        return JSONResponse({"control": "ok", "data_plane": supervisor.snapshot()})

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
            for rec, kernel in zip(records, kernels)
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

    async def root(_request: Request) -> Response:
        # The control's own buildless dashboard (single embedded page, no build
        # step). It polls /api/status + /api/sessions and drives the data-plane
        # verbs above.
        return HTMLResponse(_DASHBOARD_HTML)

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
        except httpx.ConnectError:
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
        # Allowlist the browser observe surface only (§6.1). The child's /mcp
        # agent transport is deliberately off this origin — agents reach it
        # directly on the child's own loopback port (stdio shim bridge /
        # `biopb mcp view`), never via the control — and this hop strips /mcp's
        # entire auth (Host/Origin), so exposing it would be an RCE hole on the
        # public origin. Require an allowed first segment AND reject any
        # parent-traversal, so no path (raw, encoded, or dot-collapsed by httpx)
        # can escape /observe + /api/* into /mcp.
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
        except httpx.ConnectError:
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

    # One sub-app proxying to the sidecar root, mounted at BOTH /data_plane/viewer
    # (the dataviewer's static base) and /data_plane (its /api, /ws, health). Each
    # Mount strips its own prefix, so the same handlers serve both without the
    # sidecar knowing the namespace. Reusing one instance is safe -- it is
    # stateless (routes only). /ws/render must precede the catch-all.
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

    # A session sub-app mounted under /session/{session_id}; its catch-all strips
    # nothing further (session_proxy reads {session_id} + {path} from path_params).
    session_app = Starlette(
        routes=[
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
        Route("/api/agents", api_agents, methods=["GET"]),
        Route("/api/agents/{agent_id}/register", agent_register, methods=["POST"]),
        Route("/api/agents/{agent_id}/unregister", agent_unregister, methods=["POST"]),
        Route("/api/algorithms", api_algorithms, methods=["GET"]),
        Route("/", root, methods=["GET"]),
        # /data_plane/viewer FIRST so it wins over the broader /data_plane mount.
        Mount("/data_plane/viewer", sidecar),
        Mount("/data_plane", sidecar),
        # Per-session observe: /session/<id>/observe + /session/<id>/api/*. The
        # {session_id} convertor is slash-free (session ids are), so it stops at
        # the first slash and the remainder falls to the catch-all.
        Mount("/session/{session_id}", session_app),
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
    # §6.1). None in the all-localhost dev-bypass case -> the gate falls back to a
    # loopback Host check instead.
    app = build_app(supervisor, ensure_timeout, data_web_url, token=spec.token)

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


# ---------------------------------------------------------------------------
# The control's own root dashboard (single static page; vanilla JS, no build
# step -- the same buildless pattern as biopb_mcp's _observe.py, so the lean
# control needs no Vite/npm toolchain). It polls /api/status + /api/sessions and
# POSTs the data-plane verbs. Served at "/", which is this origin's root, so its
# API calls are plain root-absolute "/api/*".
# ---------------------------------------------------------------------------

# The biopb mark (the tensor server's favicon-32.png) embedded as a data URI so
# the tab icon is self-contained -- the dashboard renders it whether or not the
# data plane is up, and the control needs no static-asset route of its own. Kept
# on one line so no accidental whitespace enters the base64 payload.
_FAVICON = "data:image/png;base64,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"  # noqa: E501

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>biopb · control</title>
<link rel="icon" type="image/png" href="__FAVICON__">
<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #111; color: #ddd; }
  header { padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; position: sticky; top: 0; }
  h1 { font-size: 15px; margin: 0; font-weight: 600; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .5px; color: #6a8;
       margin: 0 0 10px; }
  #conn { font-size: 12px; color: #9aa; margin-left: auto; }
  main { padding: 16px; max-width: 760px; }
  .card { border: 1px solid #333; border-radius: 6px; padding: 14px 16px; margin-bottom: 16px;
          background: #161616; }
  .badge { font-size: 11px; padding: 1px 8px; border-radius: 10px; text-transform: uppercase;
           vertical-align: middle; }
  .serving { background: #243; color: #7e7; }
  .starting { background: #234; color: #8bf; }
  .down, .conflict { background: #422; color: #f99; }
  .stopped, .unknown { background: #333; color: #aaa; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; margin: 12px 0 0; }
  dt { color: #888; }
  dd { margin: 0; font-family: ui-monospace, Menlo, monospace; font-size: 12px; word-break: break-all; }
  .err { color: #f99; }
  .controls { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button { font: inherit; padding: 4px 12px; border: 1px solid #444; border-radius: 4px;
           background: #222; color: #ddd; cursor: pointer; }
  button:hover:not(:disabled) { background: #2c2c2c; }
  button:disabled { opacity: .45; cursor: default; }
  button.danger { border-color: #844; }
  a.link { color: #8bf; text-decoration: none; margin-left: auto; }
  /* Only the first link in a run is right-pushed; siblings sit next to it. */
  a.link + a.link { margin-left: 0; }
  a.link:hover { text-decoration: underline; }
  a.link.off { color: #667; pointer-events: none; }
  ul { list-style: none; margin: 0; padding: 0; }
  li { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-top: 1px solid #262626; }
  li:first-child { border-top: 0; }
  .sid { font-family: ui-monospace, Menlo, monospace; font-weight: 600; }
  .when { color: #888; font-size: 12px; }
  .empty { color: #777; padding: 6px 0; }
  a.obs { color: #7e7; text-decoration: none; }
  a.obs:hover { text-decoration: underline; }
  /* Kernel (the heavy, on-demand component) state per session. Pushed right so
     it groups with the observe link. */
  .kbadge { margin-left: auto; font-size: 11px; padding: 1px 8px; border-radius: 10px;
            white-space: nowrap; }
  .k-ready { background: #243; color: #7e7; }
  .k-busy { background: #143; color: #9d7; }
  .k-starting { background: #234; color: #8bf; }
  .k-none { background: #2a2a2a; color: #999; }
  .k-error { background: #422; color: #f99; }
  .k-unknown { background: #2a2a2a; color: #777; }
  .mini { padding: 0 8px; font-size: 12px; margin-left: 8px; vertical-align: middle; }
  .state { color: #888; font-size: 12px; }
  .note { color: #667; font-size: 12px; margin: 12px 0 0; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #555;
         display: inline-block; flex: none; }
  .dot.registered { background: #7e7; }
  .dot.installed { background: #8bf; }
  .dot.drift { background: #fb4; }
  .agent-btns { margin-left: auto; display: flex; gap: 6px; }
  button.warn { border-color: #a83; color: #fc9; }
  .dot.serving { background: #7e7; }
  .dot.down { background: #f77; }
  .tls { font-size: 10px; color: #8bf; border: 1px solid #345; border-radius: 3px;
         padding: 0 4px; margin-left: 6px; vertical-align: middle; }
  .ops { margin-left: auto; color: #789; font-size: 12px; max-width: 55%;
         overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
         font-family: ui-monospace, Menlo, monospace; }
  .ops.err { color: #d88; }
</style>
</head>
<body>
<header>
  <h1>biopb · control</h1>
  <span id="conn">…</span>
</header>
<main>
  <div class="card">
    <h2>Data plane</h2>
    <div><span id="dp-badge" class="badge unknown">unknown</span></div>
    <dl id="dp-fields"></dl>
    <div class="controls">
      <button id="ensure">Ensure up</button>
      <button id="restart">Restart</button>
      <button id="stop" class="danger">Stop</button>
      <a id="viewer" class="link off" href="/data_plane/viewer/"
         target="_blank" rel="noopener">View Data →</a>
      <a id="config" class="link off" href="/data_plane/viewer/admin"
         target="_blank" rel="noopener">Config →</a>
    </div>
  </div>
  <div class="card">
    <h2>Algorithm plane<button id="algos-refresh" class="mini">↻</button></h2>
    <ul id="algos"><li class="empty">loading…</li></ul>
    <p class="note">Read-only view of the biopb.image ProcessImage servers
       configured for agent kernels, with a live health + ops probe. Lifecycle
       control is not offered here.</p>
  </div>
  <div class="card">
    <h2>Agent clients<button id="agents-refresh" class="mini">↻</button></h2>
    <ul id="agents"><li class="empty">loading…</li></ul>
    <p class="note">Registers biopb-mcp with the client. Restart the client after
       (un)registering for the change to take effect.</p>
  </div>
  <div class="card">
    <h2>Agent sessions</h2>
    <ul id="sessions"><li class="empty">loading…</li></ul>
  </div>
</main>
<script>
const POLL = 3000;

function el(id) { return document.getElementById(id); }
// Escape for BOTH text and attribute (href="...") contexts: the textContent
// trick only covers &<>, not quotes, so an interpolated value in an attribute
// could break out. Server-built values are trusted today (session ids are
// <ts>-<pid>), but escape totally so a future registry field can't inject.
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// The control's own /api/* is token-gated at this single origin. Reuse the
// dataviewer's sessionStorage token (same origin now, key 'biopb_token') as an
// Authorization: Bearer header; on a 401, prompt once and retry. In the common
// no-token localhost deployment there is no token and the header is just absent.
async function fetchAuth(url, opts) {
  opts = opts || {};
  const t = sessionStorage.getItem('biopb_token');
  opts.headers = Object.assign({}, opts.headers,
                               t ? {Authorization: 'Bearer ' + t} : {});
  const r = await fetch(url, opts);
  if (r.status === 401 && !opts._retried) {
    const nt = prompt('Access token required:');
    if (nt) {
      sessionStorage.setItem('biopb_token', nt);
      opts._retried = true;
      return fetchAuth(url, opts);
    }
  }
  return r;
}

async function jpost(url) {
  try {
    const r = await fetchAuth(url, {method: 'POST'});
    return await r.json().catch(() => ({}));
  } catch (e) { return {error: String(e)}; }
}

function renderDataPlane(dp) {
  const state = (dp && dp.state) || 'unknown';
  const badge = el('dp-badge');
  badge.textContent = state;
  badge.className = 'badge ' + state;
  const rows = [
    ['gRPC', dp.grpc_url],
    ['Web', dp.web_url],
    ['PID', dp.pid == null ? '—' : dp.pid],
    ['Restarts', dp.restarts == null ? 0 : dp.restarts],
  ];
  let html = rows.map(([k, v]) => '<dt>' + k + '</dt><dd>' + esc(v) + '</dd>').join('');
  if (dp.last_error) html += '<dt>Error</dt><dd class="err">' + esc(dp.last_error) + '</dd>';
  el('dp-fields').innerHTML = html;
  // The dataviewer and its admin/config page are only reachable once the sidecar
  // is up behind a serving plane.
  const off = state !== 'serving';
  el('viewer').classList.toggle('off', off);
  el('config').classList.toggle('off', off);
}

async function pollStatus() {
  let s;
  try { s = await (await fetchAuth('/api/status')).json(); }
  catch (e) { el('conn').textContent = 'control unreachable'; return; }
  el('conn').textContent = 'control: ok · ' + (s.sessions || 0) + ' session(s)';
  renderDataPlane(s.data_plane || {});
}

// A per-session kernel badge: the kernel is the heavy component and starts on
// demand, so surfacing whether one is attached (and its state) is the useful
// signal. 'none' reads as "no kernel"; every other state is shown verbatim.
function kernelBadge(k) {
  k = k || 'unknown';
  const label = k === 'none' ? 'no kernel' : 'kernel: ' + k;
  return '<span class="kbadge k-' + esc(k) + '">' + esc(label) + '</span>';
}

async function pollSessions() {
  let data;
  try { data = await (await fetchAuth('/api/sessions')).json(); }
  catch (e) { return; }
  const list = (data && data.sessions) || [];
  const box = el('sessions');
  if (!list.length) { box.innerHTML = '<li class="empty">no agent sessions</li>'; return; }
  box.innerHTML = list.map(s => {
    const when = s.started_at ? new Date(s.started_at * 1000).toLocaleTimeString() : '';
    return '<li><span class="sid">' + esc(s.session_id) + '</span>'
      + '<span class="when">:' + esc(s.port) + (when ? ' · ' + esc(when) : '') + '</span>'
      + kernelBadge(s.kernel)
      + '<a class="obs" href="' + esc(s.observe_url) + '" target="_blank" rel="noopener">observe →</a></li>';
  }).join('');
}

// The agent-client buttons for one row. not_installed rows get none; a
// registered row offers Re-register (amber when the stored command has drifted
// from the current biopb-mcp location) + Unregister; installed offers Register.
function agentButtons(a) {
  if (a.state === 'not_installed') return '';
  const id = esc(a.id);
  if (a.state === 'registered') {
    const cls = a.drifted ? ' class="warn"' : '';
    const label = a.drifted ? 'Re-register (update)' : 'Re-register';
    return '<button data-act="register" data-id="' + id + '"' + cls + '>' + label + '</button>'
      + '<button data-act="unregister" data-id="' + id + '" class="danger">Unregister</button>';
  }
  return '<button data-act="register" data-id="' + id + '">Register</button>';
}

// Fetched on load / after an action / via the ↻ button -- deliberately NOT on
// the status+sessions poll interval: reading agent status touches third-party
// config files, and nothing here changes on its own between user actions.
async function pollAgents() {
  let data;
  try { data = await (await fetchAuth('/api/agents')).json(); }
  catch (e) { return; }
  const list = (data && data.agents) || [];
  const box = el('agents');
  if (!list.length) { box.innerHTML = '<li class="empty">none</li>'; return; }
  box.innerHTML = list.map(a => {
    const label = a.state === 'registered'
      ? (a.drifted ? 'registered · update available' : 'registered')
      : a.state.replace('_', ' ');
    const dot = 'dot ' + esc(a.state) + (a.drifted ? ' drift' : '');
    return '<li><span class="' + dot + '"></span>'
      + '<span class="sid">' + esc(a.name) + '</span>'
      + '<span class="state">' + esc(label) + '</span>'
      + '<span class="agent-btns">' + agentButtons(a) + '</span></li>';
  }).join('');
}

// One delegated handler for every register/unregister button (rows are rebuilt
// on each refresh, so per-button onclicks would go stale). Disable the whole
// group while the write is in flight, then re-read status.
el('agents').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-act]');
  if (!btn) return;
  const id = btn.getAttribute('data-id');
  const act = btn.getAttribute('data-act');
  if (act === 'unregister' && !confirm('Unregister biopb from ' + id + '?')) return;
  el('agents').querySelectorAll('button').forEach(b => b.disabled = true);
  const res = await jpost('/api/agents/' + encodeURIComponent(id) + '/' + act);
  if (res && res.error) alert('Failed: ' + res.error);
  await pollAgents();
});
el('agents-refresh').onclick = pollAgents;

// One algorithm-plane row: a status dot, the host:port (a TLS tag for grpcs),
// the state + op count, and an ops preview (full list in the hover title). A
// non-serving server shows its error message in the preview slot instead.
function algoRow(s) {
  const serving = s.state === 'serving';
  const dot = 'dot ' + (serving ? 'serving' : (s.state === 'unknown' ? '' : 'down'));
  let stateLabel = s.state;
  if (serving) stateLabel = s.single_op ? 'serving · single-op'
    : 'serving · ' + (s.op_count || 0) + ' op' + (s.op_count === 1 ? '' : 's');
  else if (s.state === 'invalid') stateLabel = 'invalid URL';
  else if (s.state === 'unknown') stateLabel = 'gRPC unavailable';
  const tls = s.scheme === 'grpcs' ? '<span class="tls">TLS</span>' : '';
  let tail = '';
  if (serving && s.ops && s.ops.length) {
    const joined = s.ops.join(', ');
    tail = '<span class="ops" title="' + esc(joined) + '">' + esc(joined) + '</span>';
  } else if (!serving && s.error) {
    tail = '<span class="ops err" title="' + esc(s.error) + '">' + esc(s.error) + '</span>';
  }
  return '<li><span class="' + dot + '"></span>'
    + '<span class="sid">' + esc(s.target) + '</span>' + tls
    + '<span class="state">' + esc(stateLabel) + '</span>' + tail + '</li>';
}

// Fetched on load / via the ↻ button -- like the agent block, deliberately NOT on
// the status+sessions interval: each refresh dials every configured gRPC server.
async function pollAlgos() {
  let data;
  try { data = await (await fetchAuth('/api/algorithms')).json(); }
  catch (e) { return; }
  const list = (data && data.servers) || [];
  const box = el('algos');
  if (!list.length) {
    box.innerHTML = '<li class="empty">no algorithm servers configured</li>';
    return;
  }
  box.innerHTML = list.map(algoRow).join('');
}
el('algos-refresh').onclick = pollAlgos;

function refresh() { pollStatus(); pollSessions(); }

// A data-plane verb button: disable the trio while the request is in flight (a
// restart blocks server-side until the plane is back up), then refresh.
async function verb(url, confirmMsg) {
  if (confirmMsg && !confirm(confirmMsg)) return;
  const btns = [el('ensure'), el('restart'), el('stop')];
  btns.forEach(b => b.disabled = true);
  const res = await jpost(url);
  btns.forEach(b => b.disabled = false);
  if (res && res.error) alert('Failed: ' + res.error);
  if (res && res.data_plane) renderDataPlane(res.data_plane);
  refresh();
}

el('ensure').onclick = () => verb('/api/data_plane/ensure');
el('restart').onclick = () => verb('/api/data_plane/restart', 'Restart the data plane?');
el('stop').onclick = () => verb('/api/data_plane/stop', 'Stop the data plane? Clients lose it until an Ensure.');

refresh();
pollAgents();
pollAlgos();
setInterval(refresh, POLL);
</script>
</body>
</html>
"""

# Bake the favicon into the served page once at import (rather than per request).
_DASHBOARD_HTML = _DASHBOARD_HTML.replace("__FAVICON__", _FAVICON)
