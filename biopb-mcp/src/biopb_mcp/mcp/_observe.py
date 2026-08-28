"""Loopback data API backing the observe web UI.

The ``/api/*`` calls behind the observe page: ``execute_code`` job history with
truncated output plus global control knobs — interrupt the current job (force a
KeyboardInterrupt into its thread), hard-restart the kernel, save the session as
a notebook, and — where this session owns its own reap — end it. On by default
(opt-out via ``observe.enabled``).

**Stopping the session** (``/api/shutdown``) exists only for an agentless
``biopb mcp view`` viewer, and runs the launcher's own ``_shutdown``: the same
single path Ctrl-C and SIGTERM take, injected at wiring time
(:func:`set_session_owns_its_reap`) rather than reimplemented. That is what
keeps the control out of the ownership question — it proxies a session ending
*itself*, so a viewer someone started in a terminal and one the dashboard
launched behave identically and neither is anybody's to kill. A shim-owned child
gets no such route: its shim owns its reap, and ending it here would leave that
shim bridging to a dead process.

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

**The user console** (``/console/execute``, ``observe.console_enabled``) is the
one route here that submits code: it is how a human runs a cell in this kernel,
alongside the agent and through the same one-at-a-time job runner
(``docs/user-console.md``). It lives under its own path root, not under
``/api/``, because the *control* proxies the two differently — ``api`` always,
``console`` only when the control is loopback-bound. That decision cannot be made
here: the proxy hop strips Host and Origin, so this process cannot tell a browser
from the trusted hop, and its own guard passes either way. So this module's job
is only to be honest about which routes exist; the reachability question belongs
to ``biopb-control``.
"""

import asyncio
import functools
import json
import logging

from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.applications import Starlette
from starlette.background import BackgroundTask
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

# Cap on the console's optional "why" note. A label, not the payload -- the job
# row shows one line of it and the export renders it as a sentence -- so it is
# truncated rather than refused: rejecting a cell because its note ran long
# would cost the user the run, which is the part that matters.
_MAX_INTENT_CHARS = 500

# Launcher-tunable settings (defaults mirror _config DEFAULT_CONFIG). Set by
# configure() before the routes are registered/served.
_max_output_chars = 20000
_poll_interval_ms = 3000
_console_enabled = True
# Whether the built-in chat client is actually mounted on this session. Not a
# config mirror: chat is served only on an agentless `biopb mcp view` session
# and only when enabled, so `_setup_chat`'s verdict is the one truth. Set by
# set_chat_enabled() rather than configure(), which resets its extras on every
# call and so cannot be called twice.
_chat_enabled = False
# Whether this session owns its own reap -- an agentless `biopb mcp view`
# viewer, as opposed to a child a stdio shim spawned and will reap. The stop
# route exists only for the former: ending a shim's child would leave the shim
# bridging to a dead process and its MCP client reading errors instead of a
# clean close. Deliberately NOT keyed off _chat_enabled, which is a config
# switch that is off by default -- a viewer with chat disabled still owns its
# reap and still needs a way out.
_agentless = False
# The session's own teardown (the launcher's `_shutdown`), or None where there
# is nothing this session may end. Injected rather than reimplemented: it is the
# same single path Ctrl-C and SIGTERM take, so a stop from the web de-registers,
# reaps the kernel and closes the cluster in exactly the same order.
_shutdown_hook = None
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
    console_enabled=None,
    allowed_origins=(),
    allowed_hosts=(),
):
    """Apply config before registering/serving (idempotent).

    ``allowed_origins`` / ``allowed_hosts`` extend the loopback Host/Origin
    allowlist (e.g. a reverse-proxy front), mirroring the ``transport`` section.

    ``console_enabled`` off drops the console route entirely rather than serving
    a refusing one — the same shape as the control's gate, so "is there a way to
    submit code here?" has one answer, not a status code to interpret. It can
    only narrow: with the control's gate closed the route is unreachable however
    this is set.
    """
    global _max_output_chars, _poll_interval_ms, _console_enabled
    global _extra_origins, _extra_hosts
    global _mw
    if max_output_chars is not None:
        _max_output_chars = int(max_output_chars)
    if poll_interval_ms is not None:
        _poll_interval_ms = int(poll_interval_ms)
    if console_enabled is not None:
        _console_enabled = bool(console_enabled)
    _extra_origins = tuple(allowed_origins)
    _extra_hosts = tuple(allowed_hosts)
    _mw = None  # rebuilt with the new extras on next request


def set_session_owns_its_reap(agentless, on_shutdown=None):
    """Record that this session may be stopped from the web, and how.

    Must run before :func:`register_http_routes`, which reads it to decide
    whether the stop route exists at all -- an absent route rather than a
    refusing one, the same shape the console gate uses, so "can this session be
    ended from here?" is one answer and not a status code to interpret.
    """
    global _agentless, _shutdown_hook
    _agentless = bool(agentless)
    _shutdown_hook = on_shutdown if _agentless else None


def set_chat_enabled(enabled):
    """Record whether this session mounted the chat routes, for ``/api/status``.

    Separate from :func:`configure` because the launcher only knows this *after*
    it has tried to mount chat, and configure() is not safely re-callable (it
    resets ``allowed_origins``/``allowed_hosts`` whether or not they were
    passed).
    """
    global _chat_enabled
    _chat_enabled = bool(enabled)


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


def _json_route(fn):
    """:func:`_route` plus the SDK's ``Content-Type: application/json`` rule.

    :func:`_check_origin` deliberately skips that rule because the other control
    POSTs carry no body — but a JSON content-type is one a cross-site form POST
    **cannot** set (it is not a CORS-simple value, so it preflights), which makes
    it a real CSRF defense on the one route that submits code. Restored here
    rather than added to ``_route`` so the exemption above stays true of the
    routes it describes, and so a body-carrying route cannot inherit the
    body-less guard by accident.
    """
    guarded = _route(fn)

    @functools.wraps(fn)
    async def wrapper(request):
        if not _get_mw()._validate_content_type(request.headers.get("content-type")):
            return PlainTextResponse("Invalid Content-Type header", status_code=400)
        return await guarded(request)

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
        return JSONResponse({"busy": True, "jobs": []})
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
    return JSONResponse({"jobs": result})


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
    nb = _notebook.build_notebook(jobs)
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
    """Restart the kernel on the user's behalf.

    Never gated on the kernel's one-agent claim, which makes this the recovery
    path for a session held by a client that is gone: an agent cannot take a
    kernel from another agent, but the person at the machine can always replace
    it. Clearing the mirrored claim keeps that honest — the next agent to run
    code here is the new holder, and must not be measured against the old one.
    """
    host, err = _require_host()
    if err is not None:
        return err
    try:
        host.restart()
    except Exception as exc:  # noqa: BLE001 - report restart failure
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    _server.clear_claim()
    return JSONResponse({"ok": True})


async def _console_execute(request):
    """Run a cell the *user* typed, in the same kernel the agent uses.

    Submitted through the one job runner with ``origin='user'``, so the two
    writers are serialized by the rule that already exists: one job at a time.
    A collision is therefore an ordinary, expected outcome — reported as ``409``
    with *whose* job is running so the page can render it as state ("kernel busy
    · job-7 (agent)") rather than as a failed action. There is no preemption and
    no queue: a person who wants the kernel now uses Interrupt, which is theirs
    to use and attributes the stop to them.

    The optional ``intent`` is the same field the agent fills through
    ``execute_code``: why this cell is being run. It is what lets the notebook
    export say why a *human's* cell ran rather than only showing its code --
    without it the audit records a reason for every writer except the person.
    """
    host, err = _require_host()
    if err is not None:
        return err
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is the client's error
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    code = payload.get("code") if isinstance(payload, dict) else None
    if not isinstance(code, str) or not code.strip():
        return JSONResponse({"error": "missing 'code'"}, status_code=400)
    # Optional, and unlike the agent's it is nobody's obligation: someone running
    # one line to look at a variable owes no reason for it. So absent, blank and
    # not-a-string all mean "no intent" rather than a bad request -- the job
    # simply carries none, exactly as it does today.
    raw = payload.get("intent") if isinstance(payload, dict) else None
    intent = raw.strip()[:_MAX_INTENT_CHARS] if isinstance(raw, str) else ""

    submitted, res, _w = _server._run_job_call(
        host,
        "submit(" + repr(code) + ", origin='user', intent=" + repr(intent) + ")",
    )
    if submitted is None:
        # Distinct from the job-busy case below: this is the kernel *lock*, held
        # by another quick snippet for a moment. Transient, so retryable.
        if res.get("status") == "busy":
            return JSONResponse(
                {"error": "kernel busy", "retry": True}, status_code=503
            )
        return JSONResponse(
            {
                "error": res.get("status") or "kernel error",
                "detail": _server._format_execute_result(res),
            },
            status_code=502,
        )
    if submitted.get("error") == "busy":
        return JSONResponse(
            {
                "error": "busy",
                "running_job_id": submitted.get("running_job_id"),
                "running_job_origin": submitted.get("running_job_origin"),
            },
            status_code=409,
        )
    return JSONResponse(submitted)


async def _api_status(request):
    host, err = _require_host()
    if err is not None:
        return err
    # poll_interval_ms rides the status payload so the observe SPA (served by the
    # control front, not this child) can adopt the launcher-tuned cadence instead
    # of hardcoding it — the page is now static and can't be server-templated.
    # console_enabled rides here so the page knows whether to offer an editor at
    # all. It is only *this* half of the answer -- the control's gate is the
    # other -- so the SPA needs both before it renders one (see ObservePage).
    # chat_enabled rides here for a different reader: the control's dashboard,
    # which probes this endpoint per session anyway and needs it to label the
    # session's link -- a `biopb mcp view` session leads with chat, an MCP
    # client's child with the job list. Reporting it beside console_enabled
    # keeps that one probe the whole answer.
    return JSONResponse(
        {
            **host.health(),
            "poll_interval_ms": _poll_interval_ms,
            "console_enabled": _console_enabled,
            "chat_enabled": _chat_enabled,
            # Two different questions, both read by the control's dashboard off
            # this one probe: chat_enabled says what the page leads with,
            # agentless says who owns the reap -- and so whether to offer a stop.
            "agentless": _agentless,
        }
    )


# How long the stop route waits before tearing the process down. The teardown
# ends in os._exit, so it must not run until the response has left: a background
# task already runs after the body is sent, and this covers the flush behind it.
_SHUTDOWN_DELAY = 0.25


async def _api_shutdown(_request):
    """End this session -- the same teardown Ctrl-C runs.

    The control proxies this rather than signalling a pid, which is what keeps
    the ownership question from arising: a session started from a terminal and
    one started from the dashboard are the same process ending itself the same
    way, and the control needs no record of which it launched.

    Answers *before* it exits. The teardown's own log line carries the reason,
    so a viewer started in a terminal says why it is returning rather than
    dying silently under the person watching it.
    """
    if _shutdown_hook is None:
        return JSONResponse(
            {"error": "this session does not own its own shutdown"},
            status_code=404,
        )

    async def _teardown():
        await asyncio.sleep(_SHUTDOWN_DELAY)
        _shutdown_hook()

    logger.info("Stop requested from the web; shutting down.")
    return JSONResponse({"stopping": True}, background=BackgroundTask(_teardown))


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

# Served only where this session owns its own reap. Under ``api`` rather than a
# root of its own: the control already proxies that root everywhere, and it
# already carries a comparably destructive verb in /api/kernel/restart. This is
# not an execute surface, so it needs none of the console's local-only gating.
_SHUTDOWN_ROUTES = [
    ("/api/shutdown", ["POST"], _route(_api_shutdown)),
]

# Under its own root, so the control can proxy it on a different rule than
# /api/* (biopb-control `_session_proxy_roots`).
_CONSOLE_ROUTES = [
    ("/console/execute", ["POST"], _json_route(_console_execute)),
]


def _routes():
    """The routes to serve: the data API, plus the console and the stop route
    where each is enabled."""
    routes = list(_ROUTES)
    if _console_enabled:
        routes += _CONSOLE_ROUTES
    if _agentless:
        routes += _SHUTDOWN_ROUTES
    return routes


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
    for path, methods, handler in _routes():
        _server.mcp.custom_route(path, methods=methods)(handler)
    _mounted_http = True
    logger.info(
        "observe API mounted on the MCP app at /api/* (console: %s, stop: %s)",
        "on" if _console_enabled else "off",
        "on" if _agentless else "off",
    )


def _build_standalone_app():
    """Build a Starlette app wrapping the observe routes.

    Used only by tests to exercise the handlers through Starlette's TestClient
    (no server is ever run from it). Production always goes through
    :func:`register_http_routes` on the MCP app.
    """
    return Starlette(routes=[Route(p, h, methods=m) for p, m, h in _routes()])


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
