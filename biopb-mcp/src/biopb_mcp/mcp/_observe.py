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
"""

import asyncio
import functools
import json
import logging

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import _app, _http, _notebook, _scratch, _writers

logger = logging.getLogger(__name__)

# Reason string threaded into the job record (via _jobs.interrupt) so the
# agent sees, through its normal poll_job / execute_code result, that a *user* —
# not it — stopped the work.
_USER_INTERRUPT_MSG = "Interrupted by user via the observe web UI."

# Launcher-tunable settings (defaults mirror _config DEFAULT_CONFIG). Set by
# configure() before the routes are registered/served.
_max_output_chars = 20000
_poll_interval_ms = 3000
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
    global _max_output_chars, _poll_interval_ms
    if max_output_chars is not None:
        _max_output_chars = int(max_output_chars)
    if poll_interval_ms is not None:
        _poll_interval_ms = int(poll_interval_ms)
    _http.configure(allowed_origins, allowed_hosts)


def set_session_owns_its_reap(agentless, on_shutdown=None):
    """Record that this session may be stopped from the web, and how.

    Must run before :func:`register_http_routes`, which reads it to decide
    whether the stop route exists at all -- an absent route rather than a
    refusing one, the same shape the chat gate uses, so "can this session be
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


# The shared HTTP-surface layer lives in `_http`: the Host/Origin guard, the
# catch-all 500, the JSON-body parse and the kernel-host/error mapping are the
# same for the chat routes, so they are not this page's to own. Aliased rather
# than re-spelled at each use so the route tables below stay readable.
_route = _http.route
_check_origin = _http.check_origin
_require_host = _http.require_host


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
    # The host's records: this poll runs about once a second for the life of
    # the session and never enters the kernel.
    result = {"jobs": host.jobs.summary()}
    # The scratch kernel's runs and its verified workflow are the session
    # child's to report: both live here, and the session kernel cannot see
    # either. Its runs are a *separate* list rather than rows merged into
    # `jobs`, because they ran in a different kernel -- the page shows one
    # kernel at a time and says which. Merged into the same reply because the
    # page redraws from all of it and this poll runs about once a second for
    # the life of the session.
    result["workflow"] = _scratch.verified_summary()
    result["verify_jobs"] = _scratch.runs_view()
    return JSONResponse(result)


async def _api_job_detail(request):
    host, err = _require_host()
    if err is not None:
        return err
    job_id = request.path_params["job_id"]
    if _scratch.owns(job_id):
        # A verification runs in a kernel this one has never heard of, so asking
        # it would 404 the very row _api_jobs put in the list.
        snap = _scratch.detail(job_id)
        if snap is None:
            return JSONResponse({"error": "no such job", "job_id": job_id}, 404)
        shown, truncated, full_len = _truncate_tail(snap.get("stdout", ""))
        snap["stdout"] = shown
        snap["truncated"] = truncated
        snap["stdout_len"] = full_len
        return JSONResponse(snap)
    snap = host.jobs.poll(job_id)
    if snap.get("status") == "unknown":
        return JSONResponse({"error": "no such job", "job_id": job_id}, 404)
    shown, truncated, full_len = _truncate_tail(snap.get("stdout", ""))
    snap["stdout"] = shown
    # Two truncations can apply: this view's tail cap, and the job record's own
    # output cap upstream of it. `stdout_len` is what the cell actually printed,
    # so it comes from the record's total rather than from what survived here --
    # otherwise a capped job reports its kept tail as its full size.
    total = snap.get("stdout_total", full_len)
    snap["truncated"] = truncated or total > full_len
    snap["stdout_len"] = total
    return JSONResponse(snap)


async def _api_notebook(request):
    """The session as a notebook: the audit export, or ``?workflow=1`` for the
    verified one.

    Two documents rather than one with a flag inside it, because they answer
    different questions and a reader wants to have chosen. The default is
    unchanged, so an older page (and a bookmarked URL) still gets the audit.
    """
    host, err = _require_host()
    if err is not None:
        return err
    if request.query_params.get("workflow"):
        # From this process, not the kernel: the run happened in a scratch
        # kernel that no longer exists.
        record = _scratch.verified()
        if record is None:
            return JSONResponse({"error": "no verified workflow in this session"}, 404)
        nb = _notebook.build_workflow_notebook(record)
        filename = _notebook.suggested_workflow_filename(record.get("title", ""))
        return _notebook_response(nb, filename)

    nb = _notebook.build_notebook(host.jobs.export())
    filename = _notebook.suggested_filename()
    return _notebook_response(nb, filename)


def _notebook_response(nb, filename):
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
    # Whichever kernel holds the running job. The slot spans both -- a session
    # cell is refused while a verification runs and vice versa -- so "the
    # running job" is unambiguous, and the button exists only on the row that
    # is it.
    if _scratch.running() is not None:
        data = await asyncio.to_thread(_scratch.interrupt, _USER_INTERRUPT_MSG)
        return JSONResponse(data or {"interrupted": False})
    # The row's job, which the kernel stops only if it is still the one
    # running; without one, the job the records say is running.
    job_id = request.query_params.get("job_id")
    if job_id is None:
        running = host.jobs.running()
        if running is None:
            return JSONResponse({"interrupted": False, "status": "idle"})
        job_id = running["job_id"]
    try:
        data = await asyncio.to_thread(
            host.control, "interrupt", job_id=job_id, reason=_USER_INTERRUPT_MSG
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return JSONResponse({"error": "kernel error", "detail": str(exc)}, 502)
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
        # Discard, restart and clear as one critical section
        # (_writers.restart_for_user): an agent's submit can land between them
        # and lose the claim it just made on the new kernel, and an in-flight
        # verification holds the session's one job slot in *this* process, so
        # leaving it would hand the user a kernel that can accept nothing. Off
        # the loop because a restart is seconds of teardown and bring-up, which
        # would otherwise take this page with it.
        await asyncio.to_thread(
            _writers.restart_for_user,
            host,
            functools.partial(_scratch.discard, "the user restarted the kernel"),
        )
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
    # chat_enabled rides here for the control's dashboard, which probes this
    # endpoint per session anyway and needs it to label the session's link -- a
    # `biopb mcp view` session leads with chat, an MCP client's child with the
    # job list -- so that one probe is the whole answer.
    return JSONResponse(
        {
            **host.health(),
            "poll_interval_ms": _poll_interval_ms,
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
# not an execute surface, so it needs none of the chat root's local-only gating.
_SHUTDOWN_ROUTES = [
    ("/api/shutdown", ["POST"], _route(_api_shutdown)),
]


def _routes():
    """The routes to serve: the data API, plus the stop route where this
    session owns its reap."""
    routes = list(_ROUTES)
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
        _app.mcp.custom_route(path, methods=methods)(handler)
    _mounted_http = True
    logger.info(
        "observe API mounted on the MCP app at /api/* (stop: %s)",
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
