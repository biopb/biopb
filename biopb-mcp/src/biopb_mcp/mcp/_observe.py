"""Minimal loopback web UI for observing and controlling the kernel.

A small web interface (``/observe`` + ``/api/*``) that shows ``execute_code``
job history with truncated output and exposes global control knobs — interrupt
the current job (force a KeyboardInterrupt into its thread), hard-restart the
kernel, and save the session as a notebook. On by default (opt-out via
``mcp.observe.enabled``).

It is hosted in the *MCP server process* (the one that owns the
:class:`~biopb_mcp.mcp._kernel.KernelHost`), so controls are direct method calls
and reads reuse the same in-kernel job round-trip the tools use — no new IPC, and
no dependence on the dask scheduler/dashboard.

**Mounted on the http server.** :func:`register_http_routes` mounts the routes
on the *existing* FastMCP Starlette app via ``mcp.custom_route``, so they share
the MCP loop and port (``mcp.transport.port``) with ``/mcp``. The server is
http-only (daemon migration, docs/daemon-migration.md), so the UI is always
available — stdio clients reach it too: they connect through the launcher's
stdio→http bridge (``mcp/_shim.py``) and so hit ``/observe`` on the shared
daemon like any other http client. (It was once skipped under a stdio-*serving*
launcher, where standing a second uvicorn up inside the protocol process risked
the fd-1 JSON-RPC channel and raced the one ``KernelHost`` — that launcher no
longer exists.)

Security: the kernel is RCE by design, so every route carries its **own**
Host/Origin guard (:func:`_check_origin`) — FastMCP's transport-security only
wraps the ``/mcp`` mount, not sibling custom routes. The guard reuses the SDK's
:class:`TransportSecurityMiddleware` host/origin validators with the same
loopback allowlist as the MCP port. There is no token: loopback bind + Host/Origin
is the whole boundary (same trust model as the MCP server). When the control
front reverse-proxies this UI at ``/session/<id>/observe`` (Layer 3), that trusted
loopback hop presents a loopback Host and no Origin, so the guard still passes;
the frontend derives its API base from ``window.location`` so the same page works
served directly or behind the prefix, without this process knowing its prefix.
"""

import functools
import json
import logging

from mcp.server.transport_security import TransportSecurityMiddleware
from starlette.applications import Starlette
from starlette.responses import (
    HTMLResponse,
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
    allowlist (e.g. a reverse-proxy front), mirroring ``mcp.transport``.
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


async def _observe_page(request):
    return HTMLResponse(_OBSERVE_HTML.replace("__POLL_MS__", str(_poll_interval_ms)))


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
    return JSONResponse({**host.health(), "headless": _server._headless})


# (path, methods, handler) — shared by the http custom routes and the
# standalone stdio app so both surfaces are identical.
_ROUTES = [
    ("/observe", ["GET"], _route(_observe_page)),
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
    logger.info("observe UI mounted on the MCP app at /observe")


def _build_standalone_app():
    """Build a Starlette app wrapping the observe routes.

    Used only by tests to exercise the handlers through Starlette's TestClient
    (no server is ever run from it). Production always goes through
    :func:`register_http_routes` on the MCP app.
    """
    return Starlette(routes=[Route(p, h, methods=m) for p, m, h in _ROUTES])


def describe(mcp_port=None):
    """Where the observe UI is served, for ``server_status``.

    Returns ``{"running": bool, "url": str | None, "mode": str | None}``. Runs in
    the MCP server process, so it needs no kernel round-trip. ``mcp_port`` is the
    MCP app's port (the UI shares it).
    """
    if _mounted_http:
        host = f"127.0.0.1:{mcp_port}" if mcp_port else "127.0.0.1"
        return {
            "running": True,
            "url": f"http://{host}/observe",
            "mode": "mounted on the MCP app (http)",
        }
    return {"running": False, "url": None, "mode": None}


# ---------------------------------------------------------------------------
# Frontend (single static page; vanilla JS polling, no build step)
# ---------------------------------------------------------------------------

_OBSERVE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>biopb-mcp · observe</title>
<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #111; color: #ddd; }
  header { padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; position: sticky; top: 0; }
  h1 { font-size: 15px; margin: 0; font-weight: 600; }
  #status { font-size: 12px; color: #9aa; margin-right: auto; }
  button { font: inherit; padding: 4px 10px; border: 1px solid #444; border-radius: 4px;
           background: #222; color: #ddd; cursor: pointer; }
  button:hover { background: #2c2c2c; }
  button.danger { border-color: #844; }
  button.primary { background: #1d6b3f; border-color: #2a5; color: #eafff0;
                   font-weight: 600; margin-right: 6px; }
  button.primary:hover { background: #25804b; }
  main { padding: 12px 16px; }
  .job { border: 1px solid #333; border-radius: 5px; margin-bottom: 8px; overflow: hidden; }
  .row { display: flex; gap: 10px; align-items: center; padding: 8px 12px; cursor: pointer; }
  .row:hover { background: #1a1a1a; }
  .jid { font-weight: 600; }
  .badge { font-size: 11px; padding: 1px 7px; border-radius: 10px; text-transform: uppercase; }
  .running { background: #243; color: #7e7; }
  .ok { background: #234; color: #8bf; }
  .error { background: #422; color: #f99; }
  .cancelled { background: #432; color: #fc9; }
  .interrupted { background: #324; color: #c9f; }
  .preview { color: #8a8; font-family: ui-monospace, Menlo, monospace; font-size: 12px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
  .elapsed { color: #888; font-size: 12px; margin-left: auto; }
  .detail { border-top: 1px solid #333; padding: 10px 12px; display: none; }
  .job.open .detail { display: block; }
  .label { color: #6a8; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; margin: 8px 0 2px; }
  .label:first-child { margin-top: 0; }
  pre { white-space: pre-wrap; word-break: break-word; margin: 0;
        background: #0c0c0c; padding: 8px; border-radius: 4px; max-height: 50vh; overflow: auto;
        font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  pre.code { background: #0a0d0a; border-left: 2px solid #2a5; max-height: 30vh; }
  .meta { color: #888; font-size: 12px; margin-bottom: 4px; }
  .empty { color: #777; padding: 20px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>biopb-mcp · observe</h1>
  <span id="status">…</span>
  <button id="save" class="primary">⤓ Save notebook</button>
  <button id="interrupt">Interrupt</button>
  <button id="restart" class="danger">Restart kernel</button>
</header>
<main><div id="jobs"><div class="empty">loading…</div></div></main>
<script>
const POLL = __POLL_MS__;
// The API base, derived from where this page was actually loaded, so the same
// page works served directly ("/observe" -> BASE "") and behind the control
// front ("/session/<id>/observe" -> BASE "/session/<id>"). The child needs no
// knowledge of its external prefix, and the control never rewrites this HTML.
const BASE = window.location.pathname.replace(/\\/observe\\/?$/, '');
const expanded = new Set();
const rows = new Map();          // job_id -> row record (DOM kept across polls)
let lastNewest = null;

function setStatus(t) { document.getElementById('status').textContent = t; }

async function jpost(url) {
  try {
    const r = await fetch(url, {method: 'POST'});
    return await r.json().catch(() => ({}));
  } catch (e) { return {error: String(e)}; }
}

// Build a row's stable DOM once; later polls patch text nodes in place (no
// teardown), which is what kills the per-poll flicker.
function makeRow(id) {
  const wrap = document.createElement('div');
  wrap.className = 'job';
  wrap.innerHTML =
      '<div class="row">'
    +   '<span class="jid">' + id + '</span>'
    +   '<span class="badge"></span>'
    +   '<span class="preview"></span>'
    +   '<span class="elapsed"></span>'
    + '</div><div class="detail"></div>';
  const rec = {
    wrap,
    badge: wrap.querySelector('.badge'),
    preview: wrap.querySelector('.preview'),
    elapsed: wrap.querySelector('.elapsed'),
    detail: wrap.querySelector('.detail'),
    status: null, renderedFor: null, built: false,
  };
  wrap.querySelector('.row').onclick = () => {
    if (expanded.has(id)) { expanded.delete(id); wrap.classList.remove('open'); }
    else { expanded.add(id); wrap.classList.add('open');
           rec.renderedFor = null; renderDetail(rec, id); }
  };
  rows.set(id, rec);
  return rec;
}

async function renderDetail(rec, id) {
  let d;
  try {
    const r = await fetch(BASE + '/api/jobs/' + encodeURIComponent(id));
    if (!r.ok) return;
    d = await r.json();
  } catch (e) { return; }

  if (!rec.built) {              // build skeleton once; updates patch text only
    rec.detail.innerHTML =
        (d.code ? '<div class="label">code</div><pre class="code"></pre>' : '')
      + '<div class="label">output</div><div class="meta"></div><pre class="out"></pre>';
    rec.codePre = rec.detail.querySelector('pre.code');
    rec.meta = rec.detail.querySelector('.meta');
    rec.out = rec.detail.querySelector('pre.out');
    if (rec.codePre) rec.codePre.textContent = d.code || '';
    rec.built = true;
  }

  const note = d.truncated ? ('stdout truncated to last of ' + d.stdout_len + ' chars · ') : '';
  rec.meta.textContent = note + d.elapsed + 's'
    + (d.window_alive === false ? ' · viewer window closed' : '');

  const text = ((d.stdout || '')
    + (d.result_text ? '\\n' + d.result_text : '')
    + (d.error_text ? '\\n' + d.error_text : '')) || '(no output)';
  if (rec.out.textContent !== text) {
    // For a live job, keep the tail visible — but only if the user is already
    // at the bottom, so scrolling up to read isn't yanked back.
    const stick = rec.status === 'running'
      && (rec.out.scrollHeight - rec.out.scrollTop - rec.out.clientHeight < 4);
    rec.out.textContent = text;
    if (stick) rec.out.scrollTop = rec.out.scrollHeight;
  }
}

async function poll() {
  let data;
  try { data = await (await fetch(BASE + '/api/jobs')).json(); }
  catch (e) { setStatus('unreachable'); return; }
  if (data.busy) return;                 // transient; keep current render
  const jobs = data.jobs || [];
  const box = document.getElementById('jobs');

  if (!jobs.length) {
    box.innerHTML = '<div class="empty">no jobs yet</div>';
    rows.clear(); expanded.clear(); lastNewest = null; return;
  }
  const empty = box.querySelector('.empty');
  if (empty) empty.remove();

  const newest = jobs[jobs.length - 1].job_id;
  if (newest !== lastNewest) {            // autocollapse all but the newest
    expanded.clear(); expanded.add(newest); lastNewest = newest;
  }

  const seen = new Set();
  let prevEl = null;
  for (let k = jobs.length - 1; k >= 0; k--) {   // render newest-first
    const j = jobs[k];
    seen.add(j.job_id);
    const rec = rows.get(j.job_id) || makeRow(j.job_id);
    rec.status = j.status;

    if (rec.badge.textContent !== j.status) {
      rec.badge.textContent = j.status;
      rec.badge.className = 'badge ' + j.status;
    }
    const el = j.elapsed + 's';
    if (rec.elapsed.textContent !== el) rec.elapsed.textContent = el;
    const pv = j.code_preview || '';
    if (rec.preview.textContent !== pv) rec.preview.textContent = pv;

    const open = expanded.has(j.job_id);
    rec.wrap.classList.toggle('open', open);
    // Terminal jobs never change -> render their detail once. Only the running
    // job is refreshed each poll.
    if (open && (j.status === 'running' || rec.renderedFor !== j.status)) {
      renderDetail(rec, j.job_id);
      rec.renderedFor = j.status;
    }

    const want = prevEl ? prevEl.nextSibling : box.firstChild;
    if (rec.wrap !== want) box.insertBefore(rec.wrap, want);
    prevEl = rec.wrap;
  }

  for (const [id, rec] of rows) {          // drop rows for evicted jobs
    if (!seen.has(id)) { rec.wrap.remove(); rows.delete(id); expanded.delete(id); }
  }
}

async function pollStatus() {
  try {
    const s = await (await fetch(BASE + '/api/status')).json();
    const bits = [s.alive ? 'alive' : 'dead'];
    if (s.headless) bits.push('headless');
    if (s.busy) bits.push('busy');
    if (!s.ready) bits.push('starting');
    setStatus('kernel: ' + bits.join(' · '));
  } catch (e) { setStatus('unreachable'); }
}

document.getElementById('save').onclick = async () => {
  let r;
  try { r = await fetch(BASE + '/api/notebook'); }
  catch (e) { alert('Save failed: ' + e); return; }
  if (!r.ok) { alert('Save failed (' + r.status + ')'); return; }
  const blob = await r.blob();
  const name = r.headers.get('X-Filename') || 'biopb-mcp-session.ipynb';

  // Chromium (secure context, and 127.0.0.1 counts): native Save-As dialog so
  // the user picks the folder + filename. Firefox/Safari lack it -> prompt for a
  // name (preset as default) and save to the default download location.
  if (window.showSaveFilePicker) {
    let handle;
    try {
      handle = await window.showSaveFilePicker({
        suggestedName: name,
        types: [{description: 'Jupyter notebook',
                 accept: {'application/x-ipynb+json': ['.ipynb']}}],
      });
    } catch (e) { if (e.name === 'AbortError') return; }   // user cancelled
    if (handle) {
      const w = await handle.createWritable();
      await w.write(blob); await w.close();
      return;
    }
  }
  const chosen = prompt('Save notebook as:', name);
  if (chosen === null) return;                              // user cancelled
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = chosen || name;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
};
document.getElementById('interrupt').onclick = async () => {
  const d = await jpost(BASE + '/api/kernel/interrupt');
  if (d && d.interrupted === false && d.status === 'idle') alert('No running job.');
  poll();
};
document.getElementById('restart').onclick = async () => {
  if (!confirm('Hard-restart the kernel? All variables and layers are lost.')) return;
  await jpost(BASE + '/api/kernel/restart');
  document.getElementById('jobs').innerHTML = '';
  rows.clear(); expanded.clear(); lastNewest = null; poll();
};

poll(); pollStatus();
setInterval(poll, POLL);
setInterval(pollStatus, POLL);
</script>
</body>
</html>
"""
