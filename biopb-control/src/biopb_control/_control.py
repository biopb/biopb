"""The control plane's single web origin — a Starlette/uvicorn ASGI app on 8813.

This is the Layer-3 front of the de-daemonization migration
(``biopb-mcp/docs/mcp-dedaemonization-migration.md``, §6.1). It replaces the
earlier stdlib ``ThreadingHTTPServer`` control API with a real ASGI app **on the
same port**, and routes by namespace so no two upstreams share a path prefix:

- ``GET  /health``                -> ``{"control": "ok", "data_plane": {...}}`` —
                                     the control's own liveness (what
                                     ``_control_client`` and the installer poll).
                                     Bare, kept byte-for-byte.
- ``POST /api/data_plane/ensure`` -> ensure the plane is up (bounded wait), then
                                     the snapshot; ``biopb-mcp`` calls this in
                                     place of shelling out ``biopb server start``.
                                     Moved under ``/api/`` — control *verbs about*
                                     the plane live there.
- ``GET  /``                      -> redirect to the dataviewer for now; the
                                     control's own buildless dashboard lands here
                                     in a later Layer-3 step.
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

The buildless dashboard at ``/`` is the remaining Layer-3 step; this module lands
the namespaced origin, the data-plane proxy, and per-session observe routing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading

import httpx
import uvicorn
from biopb import _config_sessions
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route, WebSocketRoute
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


def build_app(
    supervisor: DataPlaneSupervisor,
    ensure_timeout: float,
    data_web_url: str,
) -> Starlette:
    """Build the control-plane ASGI app.

    ``data_web_url`` is the loopback base URL of the supervised tensor server's
    HTTP sidecar; the ``/data_plane`` namespace reverse-proxies there. Split out
    from :func:`serve_control_api` so it is unit-testable against a fake upstream
    without binding uvicorn.
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

    async def root(_request: Request) -> Response:
        # The control's own buildless dashboard lands here in a later Layer-3
        # step; until then send the origin root to the dataviewer so `/` is not a
        # dead end.
        return RedirectResponse(url="/data_plane/viewer/")

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
        rec = _config_sessions.resolve(session_id)
        if rec is None:
            return JSONResponse(
                {"error": f"session {session_id!r} not found or ended"},
                status_code=404,
            )
        base = _loopback_url(rec.get("host", "127.0.0.1"), rec["port"])
        target = base + "/" + request.path_params["path"]
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
        Route("/api/data_plane/ensure", data_plane_ensure, methods=["POST"]),
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

    return Starlette(routes=routes, lifespan=lifespan)


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
    if data_web_url is None:
        spec = supervisor._spec
        data_web_url = _loopback_url(spec.web_host, spec.web_port)

    app = build_app(supervisor, ensure_timeout, data_web_url)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
