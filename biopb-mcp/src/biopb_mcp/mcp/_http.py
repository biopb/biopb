"""The session child's HTTP surface: the guard every custom route wears.

Runs **in the MCP server process** (not the kernel). Two surfaces mount routes
on the FastMCP app -- the observe UI (:mod:`_observe`) and the chat pane
(:mod:`_chat_api`) -- and everything they share about *being* an HTTP route
lives here rather than in whichever of them was written first.

That sharing is not a convenience. The Host/Origin check below is this child's
**only** authentication: the control's ``/session/<id>/*`` proxy hop strips the
browser's Host and Origin, so a guard that a new surface forgot to apply is not
a weaker route, it is an unauthenticated one. Keeping it here means a route
gets it by being registered through :func:`route` / :func:`json_route`, not by
its author remembering.
"""

import functools
import logging

from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.responses import JSONResponse, PlainTextResponse

from . import _server

logger = logging.getLogger(__name__)

# Extra Host/Origin allowlist entries (a reverse-proxy front), mirroring the
# `transport` config section. Set through configure() before routes are served.
_extra_origins = ()
_extra_hosts = ()

# Lazily-built Host/Origin validator.
_mw = None


def configure(allowed_origins=(), allowed_hosts=()):
    """Extend the loopback Host/Origin allowlist. Rebuilds the validator."""
    global _extra_origins, _extra_hosts, _mw
    _extra_origins = tuple(allowed_origins)
    _extra_hosts = tuple(allowed_hosts)
    _mw = None  # rebuilt with the new extras on next request


def _get_mw():
    global _mw
    if _mw is None:
        _mw = TransportSecurityMiddleware(
            _server.build_transport_security(_extra_origins, _extra_hosts)
        )
    return _mw


def check_origin(request):
    """Return an error Response if Host/Origin is disallowed, else None.

    Reuses the SDK validators (same loopback allowlist as ``/mcp``) but skips
    its content-type rule — our control POSTs carry no JSON body.
    """
    mw = _get_mw()
    if not mw._validate_host(request.headers.get("host")):
        return PlainTextResponse("Invalid Host header", status_code=421)
    if not mw._validate_origin(request.headers.get("origin")):
        return PlainTextResponse("Invalid Origin header", status_code=403)
    return None


def route(fn):
    """Wrap a handler with the Host/Origin guard + a catch-all 500.

    Applied to every route so a new one can't forget the guard, and a wedged
    kernel surfaces a clean JSON 500 instead of leaking a traceback.
    """

    @functools.wraps(fn)
    async def wrapper(request):
        denied = check_origin(request)
        if denied is not None:
            return denied
        try:
            return await fn(request)
        except Exception as exc:  # noqa: BLE001 - report, never crash
            logger.exception("http handler error")
            return JSONResponse(
                {"error": "internal error", "detail": str(exc)},
                status_code=500,
            )

    return wrapper


def json_route(fn):
    """:func:`route` plus the SDK's ``Content-Type: application/json`` rule.

    :func:`check_origin` deliberately skips that rule because the other control
    POSTs carry no body — but a JSON content-type is one a cross-site form POST
    **cannot** set (it is not a CORS-simple value, so it preflights), which makes
    it a real CSRF defense on the one route that submits code. Restored here
    rather than added to :func:`route` so the exemption above stays true of the
    routes it describes, and so a body-carrying route cannot inherit the
    body-less guard by accident.
    """
    guarded = route(fn)

    @functools.wraps(fn)
    async def wrapper(request):
        if not _get_mw()._validate_content_type(request.headers.get("content-type")):
            return PlainTextResponse("Invalid Content-Type header", status_code=400)
        return await guarded(request)

    return wrapper


def require_host():
    """Return ``(host, None)`` or ``(None, 503 response)`` if no kernel host."""
    host = _server._kernel_host
    if host is None:
        return None, JSONResponse(
            {"error": "kernel host not initialized"}, status_code=503
        )
    return host, None


async def json_body(request):
    """``(payload, None)`` for a JSON object body, or ``(None, 400 response)``.

    Both surfaces' body-carrying routes start here, so "what does a malformed
    body get told?" has one answer.
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is the client's error
        return None, JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(payload, dict):
        return None, JSONResponse({"error": "invalid JSON body"}, status_code=400)
    return payload, None


def kernel_error(res):
    """Map a non-ok job round-trip to a response.

    A ``busy`` kernel is transient (another quick call holds the lock) -> 200
    with a ``busy`` marker the UI retries on; anything else -> 502.
    """
    status = res.get("status")
    if status == "busy":
        return JSONResponse({"busy": True, "jobs": []})
    return JSONResponse(
        {
            "error": status or "kernel error",
            "detail": _server._format_execute_result(res),
        },
        status_code=502,
    )
