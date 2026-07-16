"""Loopback data API backing the observe web UI.

The ``/api/*`` calls behind the observe page: ``execute_code`` job history with
truncated output plus global control knobs — interrupt the current job (force a
KeyboardInterrupt into its thread), hard-restart the kernel, and save the session
as a notebook. On by default (opt-out via ``observe.enabled``).

The observe **page** itself is served by the control front — it is the React
``ObservePage`` in the ``web/`` SPA, served at ``/session/<id>/observe`` — and it
calls back into this API at ``/session/<id>/api/*``, which the control proxies to
this child. So this module owns only the API; the presentation moved to the
single web origin (see ``biopb-control`` / ``web/``).

The API is hosted in the *MCP server process* (the one that owns the
:class:`~biopb_mcp.mcp._kernel.KernelHost`), so controls are direct method calls
and reads reuse the same in-kernel job round-trip the tools use — no new IPC, and
no dependence on the dask scheduler/dashboard.

**Mounted on the http server.** :func:`register_http_routes` mounts the routes
on the *existing* FastMCP Starlette app via ``mcp.custom_route``, so they share
the MCP loop and port (``transport.port``) with ``/mcp``. The server is
http-only (ARCHITECTURE.md, Lifecycle), so the API is always
available — stdio clients reach it too: they connect through the launcher's
stdio→http bridge (``mcp/_shim.py``) and so hit ``/api/*`` on the shared daemon
like any other http client. (It was once skipped under a stdio-*serving*
launcher, where standing a second uvicorn up inside the protocol process risked
the fd-1 JSON-RPC channel and raced the one ``KernelHost`` — that launcher no
longer exists.)

Security: the kernel is RCE by design, so every route carries its **own**
Host/Origin guard (:func:`_check_origin`) — FastMCP's transport-security only
wraps the ``/mcp`` mount, not sibling custom routes. The guard reuses the SDK's
:class:`TransportSecurityMiddleware` host/origin validators with the same
loopback allowlist as the MCP port. There is no token: loopback bind + Host/Origin
is the whole boundary (same trust model as the MCP server). When the control
front proxies these ``/api/*`` calls (``/session/<id>/api/*`` -> this child), that
trusted loopback hop presents a loopback Host and no Origin, so the guard still
passes; the SPA derives its API base from ``window.location`` (the
``/session/<id>`` prefix), so this process needs no knowledge of its prefix.
"""

import functools
import json
import logging

from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.applications import Starlette
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
)
from starlette.routing import Route

from . import _notebook, _server

logger = logging.getLogger(__name__)

# Reason string threaded into the job record (via _jobs.interrupt_current) so the
# agent sees, through its normal poll_job / execute_code result, that a *user* —
# not it — stopped the work.
_USER_INTERRUPT_MSG = "Interrupted by user via the observe web UI."

# Launcher-tunable settings (defaults mirror _config DEFAULT_CONFIG). Set by
# configure() before the routes are registered/served.
_max_output_chars = 20000
_poll_interval_ms = 3000
_extra_origins = ()
_extra_hosts = ()

# Lazily-built Host/Origin validator.
_mw = None

# Whether the routes were mounted on the MCP app (for server_status). Stays
# False when observe is disabled, in stdio mode, or if registration failed.
_mounted_http = False


def configure(
    *,
    max_output_chars=None,
    poll_interval_ms=None,
    allowed_origins=(),
    allowed_hosts=(),
):
    """Apply config before registering/serving (idempotent).

    ``allowed_origins`` / ``allowed_hosts`` extend the loopback Host/Origin
    allowlist (e.g. a reverse-proxy front), mirroring the ``transport`` section.
    """
    global _max_output_chars, _poll_interval_ms, _extra_origins, _extra_hosts
    global _mw
    if max_output_chars is not None:
        _max_output_chars = int(max_output_chars)
    if poll_interval_ms is not None:
        _poll_interval_ms = int(poll_interval_ms)
    _extra_origins = tuple(allowed_origins)
    _extra_hosts = tuple(allowed_hosts)
    _mw = None  # rebuilt with the new extras on next request


# ---------------------------------------------------------------------------
# Host/Origin guard (own copy — custom routes are NOT covered by FastMCP's)
# ---------------------------------------------------------------------------


def _get_mw():
    global _mw
    if _mw is None:
        _mw = TransportSecurityMiddleware(
            _server.build_transport_security(_extra_origins, _extra_hosts)
        )
    return _mw


def _check_origin(request):
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


def _route(fn):
    """Wrap a handler with the Host/Origin guard + a catch-all 500.

    Applied to every route so a new one can't forget the guard, and a wedged
    kernel surfaces a clean JSON 500 instead of leaking a traceback.
    """

    @functools.wraps(fn)
    async def wrapper(request):
        denied = _check_origin(request)
        if denied is not None:
            return denied
        try:
            return await fn(request)
        except Exception as exc:  # noqa: BLE001 - report, never crash
            logger.exception("observe handler error")
            return JSONResponse(
                {"error": "internal error", "detail": str(exc)},
                status_code=500,
            )

    return wrapper


def _require_host():
    """Return ``(host, None)`` or ``(None, 503 response)`` if no kernel host."""
    host = _server._kernel_host
    if host is None:
        return None, JSONResponse(
            {"error": "kernel host not initialized"}, status_code=503
        )
    return host, None


def _kernel_error(res):
    """Map a non-ok job round-trip to a response.

    A ``busy`` kernel is transient (another quick call holds the lock) -> 200
    with a ``busy`` marker the UI retries on; anything else -> 502.
    """
    status = res.get("status")
    if status == "busy":
        return JSONResponse({"busy": True, "jobs": [], "headless": _server._headless})
    return JSONResponse(
        {
            "error": status or "kernel error",
            "detail": _server._format_execute_result(res),
        },
        status_code=502,
    )


def _truncate_tail(text):
    """Keep the trailing ``_max_output_chars`` of *text*.

    Returns ``(shown, truncated, full_len)``. The tail is kept because for a
    running job the most recent output is what matters.
    """
    full_len = len(text)
    if full_len <= _max_output_chars:
        return text, False, full_len
    return "…(truncated)…\n" + text[-_max_output_chars:], True, full_len


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def _api_jobs(request):
    host, err = _require_host()
    if err is not None:
        return err
    result, res, _w = _server._run_job_call(host, "jobs_summary()")
    if result is None:
        return _kernel_error(res)
    return JSONResponse({"jobs": result, "headless": _server._headless})


async def _api_job_detail(request):
    host, err = _require_host()
    if err is not None:
        return err
    job_id = request.path_params["job_id"]
    snap, res, win = _server._run_job_call(host, "poll(" + repr(job_id) + ")")
    if snap is None:
        return _kernel_error(res)
    if snap.get("status") == "unknown":
        return JSONResponse({"error": "no such job", "job_id": job_id}, 404)
    shown, truncated, full_len = _truncate_tail(snap.get("stdout", ""))
    snap["stdout"] = shown
    snap["truncated"] = truncated
    snap["stdout_len"] = full_len
    snap["window_alive"] = win
    return JSONResponse(snap)


async def _api_notebook(request):
    host, err = _require_host()
    if err is not None:
        return err
    # Read the full job history on the kernel main thread (a plain read like
    # jobs_summary(), no background job thread), then serialize to a notebook in
    # this process.
    jobs, res, _w = _server._run_job_call(host, "export()")
    if jobs is None:
        return _kernel_error(res)
    nb = _notebook.build_notebook(jobs, headless=_server._headless)
    filename = _notebook.suggested_filename()
    return Response(
        json.dumps(nb, indent=1),
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # Exposed for the fetch+blob download path (Content-Disposition is
            # not readable from a same-origin fetch in all browsers).
            "X-Filename": filename,
        },
    )


async def _api_interrupt(request):
    host, err = _require_host()
    if err is not None:
        return err
    # Force a KeyboardInterrupt into the running job's worker thread (SIGINT only
    # reaches the kernel main thread, not the job), attributed to the user.
    data, res, _w = _server._run_job_call(
        host, "interrupt_current(" + repr(_USER_INTERRUPT_MSG) + ")"
    )
    if data is None:
        return _kernel_error(res)
    return JSONResponse(data)


async def _api_restart(request):
    host, err = _require_host()
    if err is not None:
        return err
    try:
        host.restart()
    except Exception as exc:  # noqa: BLE001 - report restart failure
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True})


async def _api_status(request):
    host, err = _require_host()
    if err is not None:
        return err
    # poll_interval_ms rides the status payload so the observe SPA (served by the
    # control front, not this child) can adopt the launcher-tuned cadence instead
    # of hardcoding it — the page is now static and can't be server-templated.
    return JSONResponse(
        {
            **host.health(),
            "headless": _server._headless,
            "poll_interval_ms": _poll_interval_ms,
        }
    )


# (path, methods, handler) — shared by the http custom routes and the standalone
# stdio app so both surfaces are identical. The observe *page* is served by the
# control front (the React ObservePage in web/); this child serves only the
# /api/* data + control calls that page makes.
_ROUTES = [
    ("/api/jobs", ["GET"], _route(_api_jobs)),
    ("/api/jobs/{job_id}", ["GET"], _route(_api_job_detail)),
    ("/api/notebook", ["GET"], _route(_api_notebook)),
    ("/api/kernel/interrupt", ["POST"], _route(_api_interrupt)),
    ("/api/kernel/restart", ["POST"], _route(_api_restart)),
    ("/api/status", ["GET"], _route(_api_status)),
]


# ---------------------------------------------------------------------------
# Wiring: http (mount on the MCP app) / stdio (standalone server)
# ---------------------------------------------------------------------------


def register_http_routes():
    """Mount the observe routes on the existing FastMCP app (http transport).

    Must run before ``_server.run()`` — custom routes are read when the
    streamable-http app is built. The routes become siblings of ``/mcp`` on the
    same loopback port and share the MCP event loop (no new thread, no new
    stdout handler).
    """
    global _mounted_http
    for path, methods, handler in _ROUTES:
        _server.mcp.custom_route(path, methods=methods)(handler)
    _mounted_http = True
    logger.info("observe API mounted on the MCP app at /api/*")


def _build_standalone_app():
    """Build a Starlette app wrapping the observe routes.

    Used only by tests to exercise the handlers through Starlette's TestClient
    (no server is ever run from it). Production always goes through
    :func:`register_http_routes` on the MCP app.
    """
    return Starlette(routes=[Route(p, h, methods=m) for p, m, h in _ROUTES])


def describe(mcp_port=None):
    """Whether the observe data API is mounted, for ``server_status``.

    Returns ``{"running": bool, "url": str | None, "mode": str | None}``. Runs in
    the MCP server process, so it needs no kernel round-trip. ``mcp_port`` is the
    MCP app's port (the API shares it). The observe *page* is served by the
    control front at ``/session/<id>/observe`` (the React SPA in ``web/``); this
    child hosts only the ``/api/*`` calls it makes, so ``url`` points at the API
    root rather than a page.
    """
    if _mounted_http:
        host = f"127.0.0.1:{mcp_port}" if mcp_port else "127.0.0.1"
        return {
            "running": True,
            "url": f"http://{host}/api",
            "mode": "observe API on the MCP app (http); page served by the control",
        }
    return {"running": False, "url": None, "mode": None}
