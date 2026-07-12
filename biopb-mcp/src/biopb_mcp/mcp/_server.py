"""FastMCP server exposing the napari viewer through a child Jupyter kernel.

The server runs in the foreground (uvicorn, streamable-http on
127.0.0.1:<port>/mcp) and owns a :class:`~biopb_mcp.mcp._kernel.KernelHost`.
Every tool call is a round-trip into that kernel, where the napari viewer,
dask, and the TensorFlightClient live.  The kernel can be interrupted or
hard-restarted independently of this process.
"""

import json
import logging
import os
import time

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent, TextContent

from . import _resources, _skills
from ._kernel import KernelHost

logger = logging.getLogger(__name__)

_kernel_host: KernelHost | None = None

# Seconds execute_code waits for a job to finish before returning a job handle
# instead of an inline result (set from config by the launcher).
_promote_after: float = 10.0

# Compute-only mode: no display, so the kernel has no napari viewer. Set by the
# launcher (set_headless) before serving. Viewer-dependent tools return a clear
# message and the agent is told via the initialize `instructions` field.
_headless: bool = False

# This process's logfile path (set by the launcher), surfaced by server_status so
# an agent can find its own log. None when output goes to a terminal (foreground
# `--transport http` / `biopb mcp view`) rather than a file.
_session_log_path: str | None = None

# Handed to the client in the initialize handshake (the only handshake-time
# carrier MCP defines). Clients that honor it inject it into the model's
# context from the first turn (compliance is up to the client/agent), so this
# field carries the guidance that must hold on *every* turn — the operation
# guardrails.
_BASE_INSTRUCTIONS = (
    "This biopb-mcp session drives a live napari viewer through a child IPython "
    "kernel; `execute_code` runs arbitrary Python in that kernel. Read these resources "
    "for detail before non-trivial work: guide://kernel (namespace, examples, "
    "long-running jobs & cancellation), guide://tensor (data access/upload), "
    "guide://viewer (layers/camera/dims), guide://annotations "
    "(labels/points/shapes), guide://ops (server-side image-processing ops).\n"
    "\n"
    "The napari kernel does NOT auto-start. Call `start_kernel` once at the "
    "start of the session (and again to recover after a failure or after the "
    "user closes the viewer window); it blocks until the kernel is ready.\n"
    "\n"
    "At the start of a task, call `find_skills` to check for a curated workflow "
    "before improvising; read the matching `skill://<id>` resource for the "
    "steps.\n"
    "\n"
    "Operation guardrails (apply on every turn):\n"
    "- Use data from `client` or `viewer`; avoid the filesystem unless the user "
    "explicitly asks.\n"
    '- Browse the catalog with `client.query_sources(sql, format="pandas")` '
    "(server-side DuckDB, complete), not `client.list_sources()` "
    "(server-capped for large catalogs); the `sources` columns are source_id, "
    "source_url, source_type, dtype, indexed_at, metadata_json, "
    "shape_summary, data_resident, and `tensors` (a LIST of "
    "STRUCT(array_id, dim_labels, shape, chunk_shape, dtype), one per tensor -- "
    "query per-tensor with UNNEST(tensors) or list_filter; the scalar "
    "dtype/shape_summary only describe tensors[0]). Unresolved (cloud / "
    "synced-folder) sources "
    "have NULL dtype/shape_summary, so a predicate like `WHERE dtype='uint8'` "
    "silently drops them; filter on `data_resident` to opt them in/out on "
    "purpose (`WHERE NOT data_resident` finds what hasn't been resolved yet).\n"
    "- Prefer lazy dask operations; only `.compute()` the final result.\n"
    "- Put intermediate results back on `viewer` for the user to validate at "
    "each step.\n"
    "- Do not assume — ask the user to clarify uncertainties; they know the "
    "data better than you do.\n"
    "- After accomplishing a task, ask the user whether a new skill should be generated and added "
    "to the agent's toolbox for future use."
)

# Appended to _BASE_INSTRUCTIONS when the session is headless. Phrased to fire
# only when the user actually reaches for the viewer (compliance is up to the
# client/agent).
_HEADLESS_INSTRUCTIONS = (
    "This biopb-mcp session is running HEADLESS: it was started without a "
    "display, so there is NO napari viewer window and screenshots are not "
    "available. If the user asks to access, open, view, or look at the biopb "
    "viewer/napari (or asks for a screenshot), alert them plainly that no "
    "viewer is available in this session because it started without a display. "
    "You can still load data, run image-processing ops, and compute results "
    "via execute_code (using client and ops); offer results as values/arrays. "
    "Do not call take_screenshot or use viewer.* methods."
)

# DNS-rebinding / cross-origin protection (review finding A2).  execute_code is
# a full kernel (RCE by design), so the only thing standing between a malicious
# page in the user's own browser and the loopback port is Host/Origin
# validation.  The MCP SDK enforces these lists; we set them explicitly rather
# than relying on its implicit loopback auto-enable so the control can't
# silently regress.  Wildcard ports mean the configured port never matters.
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]


def build_transport_security(
    extra_origins=(), extra_hosts=()
) -> TransportSecuritySettings:
    """Build DNS-rebinding protection settings for the loopback server.

    The loopback allowlists are always enforced; ``extra_origins`` /
    ``extra_hosts`` (from ``config['mcp']['transport']``) are appended so an
    admin fronting
    the server with a reverse proxy can permit the proxy's Host/Origin.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_LOOPBACK_HOSTS + list(extra_hosts),
        allowed_origins=_LOOPBACK_ORIGINS + list(extra_origins),
    )


mcp = FastMCP("biopb-mcp", transport_security=build_transport_security())

# FastMCP built the low-level server with instructions=None at import; seed the
# always-on base guidance now so it is present even if set_headless is never
# called (e.g. tests, or a standalone import). set_headless recomposes from this
# base.
mcp._mcp_server.instructions = _BASE_INSTRUCTIONS

_PNG_DELIM = "<<PNG_B64>>"

# Delimiter for the single-line JSON payload the in-kernel job runner prints in
# reply to a submit/poll/cancel/list snippet (mirrors the _PNG_DELIM pattern).
_JOB_DELIM = "<<JOB_JSON>>"

# Sentinel printed by the screenshot snippet when the napari window has been
# closed (the viewer survives in the namespace, but its canvas is destroyed).
_WINDOW_CLOSED_DELIM = "<<WINDOW_CLOSED>>"

# Appended to a result when the agent's code ran but the viewer window is closed,
# so the silent no-op of viewer mutations is surfaced rather than read as success.
_WINDOW_CLOSED_NOTE = (
    "\n\n⚠ The napari viewer window is closed — viewer layers won't be "
    "displayed (data/compute results are still valid). Call restart_kernel to "
    "restore the viewer."
)


def _job_snippet(call: str) -> str:
    """Build a snippet that prints ``_jobs.<call>``'s result as delimited JSON.

    ``call`` is a fully-formed call expression with arguments already embedded
    via ``repr`` by the caller (agent code is RCE by design, but embedding via
    ``repr`` keeps the payload a valid literal regardless of its contents).

    The payload is ``{"r": <call result>, "w": <viewer window alive?>}`` so the
    same round-trip also reports whether the viewer window is still open (a
    user-closed window turns viewer mutations into silent no-ops). The liveness
    probe is auxiliary, so a kernel that never bound ``_viewer_window_alive``
    (e.g. a partial/test bootstrap) reports ``w: null`` rather than breaking the
    job round-trip.
    """
    return (
        "import json as _json\n"
        "print('" + _JOB_DELIM + "' + _json.dumps("
        "{'r': _jobs." + call + ", "
        "'w': globals().get('_viewer_window_alive', lambda: None)()}))\n"
    )


_SCREENSHOT_SNIPPET = (
    "import base64 as _b64, cv2 as _cv2\n"
    "if not _viewer_window_alive():\n"
    "    print('" + _WINDOW_CLOSED_DELIM + "')\n"
    "else:\n"
    # Under async slicing, force-sync the current view so the capture reflects
    # the state the agent just set, not a pre-load frame. No-op when async is
    # off or the bootstrap predates the helper (defensive globals().get).
    "    globals().get('_resync_view', lambda: None)()\n"
    "    _arr = viewer.screenshot(canvas_only={canvas_only})\n"
    "    _bgra = _cv2.cvtColor(_arr, _cv2.COLOR_RGBA2BGRA)\n"
    "    _ok, _buf = _cv2.imencode('.png', _bgra)\n"
    "    print('" + _PNG_DELIM + "' + _b64.b64encode(_buf.tobytes()).decode())\n"
)

# Self-contained inspection snippet.  Built by string concatenation (no
# f-strings/format) so the object path is the only injected value.
_INSPECT_TEMPLATE = """
import inspect as _inspect
__path = __PATH__
try:
    __obj = eval(__path)
except Exception as __exc:
    print("Error resolving " + repr(__path) + ": " + str(__exc))
else:
    __lines = [
        "Type: " + type(__obj).__name__,
        "Docstring: " + (_inspect.getdoc(__obj) or "No documentation."),
        "",
        "Attributes:",
    ]
    for __name in sorted(dir(__obj)):
        if __name.startswith("_"):
            continue
        try:
            __attr = getattr(__obj, __name)
        except Exception:
            continue
        if _inspect.ismethod(__attr) or _inspect.isfunction(__attr):
            try:
                __sig = str(_inspect.signature(__attr))
                __short = (_inspect.getdoc(__attr) or "").split(chr(10))[0]
                __lines.append("  ." + __name + __sig + "  -- " + __short)
            except (ValueError, TypeError):
                __lines.append("  ." + __name + "(...)")
        else:
            __lines.append("  ." + __name + " [" + type(__attr).__name__ + "]")
    print(chr(10).join(__lines))
"""

_STATUS_SNIPPET = """
print("## Dask")
try:
    import dask as _dask
    print("  scheduler: " + str(_dask.config.get("scheduler", default="unknown")))
except Exception as _e:
    print("  error: " + str(_e))
try:
    if _dask_client is not None:
        _info = _dask_client.scheduler_info()
        print("  distributed_workers: " + str(len(_info.get("workers", {}))))
        print("  dashboard: " + str(_dask_client.dashboard_link))
    elif not globals().get("_dask_attach_done", True):
        print("  distributed: starting (attaching to cluster)")
    else:
        print("  distributed: not active")
except Exception:
    print("  distributed: not active")

print("")
print("## Tensor Server")
_tc = _conn.client
if _tc is not None:
    try:
        print("  connected: true")
        print("  health: " + str(_tc.health_check()))
        print("  sources_cached: " + str(len(_conn.sources or {})))
    except Exception as _e:
        print("  connected: true")
        print("  health_error: " + str(_e))
elif getattr(_conn, "last_status", "") == "starting":
    print("  connected: false")
    print("  state: starting — " + str(getattr(_conn, "last_message", "")))
else:
    print("  connected: false")
    _lm = str(getattr(_conn, "last_message", ""))
    if _lm:
        # issue #86: surface the reason (auth required / unreachable) instead of
        # a bare "connected: false" the agent can't act on.
        print("  error: " + _lm)

print("")
print("## Viewer")
if not viewer:
    print("  headless: no viewer (no display)")
elif not _viewer_window_alive():
    print("  window: CLOSED — the napari window was closed; layer mutations")
    print("    won't display. Data/compute still work; restart_kernel to restore.")
    print("  layers: " + str(len(viewer.layers)) + " (model only, not shown)")
else:
    print("  window: open")
    print("  layers: " + str(len(viewer.layers)))
    for _layer in list(viewer.layers)[:10]:
        _shape = getattr(_layer.data, "shape", "?")
        print("    - " + str(_layer.name) + " (" + str(_shape) + ")")

print("")
print("## Jobs")
try:
    _js = _jobs.jobs_summary()
    if _js:
        for _j in _js:
            print(
                "  - " + _j["job_id"] + ": " + _j["status"]
                + " (" + str(_j["elapsed"]) + "s, stdout "
                + str(_j["stdout_len"]) + "b)"
            )
    else:
        print("  (none)")
except Exception as _e:
    print("  error: " + str(_e))
"""


def set_kernel_host(host: KernelHost):
    """Register the kernel host the tools dispatch to."""
    global _kernel_host
    _kernel_host = host


def set_promote_after(seconds: float):
    """Set how long execute_code waits inline before returning a job handle."""
    global _promote_after
    _promote_after = float(seconds)


def set_session_log_path(path: str | None):
    """Record this process's logfile path for server_status to report."""
    global _session_log_path
    _session_log_path = path


def set_headless(headless: bool):
    """Mark the session compute-only (no viewer) and advertise it to the agent
    via the initialize ``instructions`` field.

    Recomposes that field from ``_BASE_INSTRUCTIONS`` in both directions
    (idempotent): append the headless directive when headless, drop it
    otherwise, so a flip back to visible can't leave a stale HEADLESS directive
    in the handshake while preserving the always-on base guidance. The
    low-level Server holds the `instructions` returned in the handshake.
    """
    global _headless
    _headless = bool(headless)
    mcp._mcp_server.instructions = (
        _BASE_INSTRUCTIONS + "\n\n" + _HEADLESS_INSTRUCTIONS
        if _headless
        else _BASE_INSTRUCTIONS
    )


def _format_execute_result(res: dict) -> str:
    status = res.get("status")
    stdout = res.get("stdout", "")
    result_text = res.get("result_text", "")
    error_text = res.get("error_text", "")

    if status == "ok":
        out = stdout
        if result_text:
            out += result_text
        return out or "(no output)"

    parts = []
    if stdout:
        parts.append(stdout)
    if error_text:
        parts.append(error_text)
    return "\n".join(parts) if parts else f"(status: {status})"


def _extract_delimited(text: str, delimiter: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(delimiter):
            return line[len(delimiter) :]
    return None


def _extract_json(text: str):
    """Parse the single-line ``<<JOB_JSON>>`` payload from a job snippet."""
    payload = _extract_delimited(text, _JOB_DELIM)
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


def _run_job_call(host, call: str):
    """Run a ``_jobs.<call>`` snippet.

    Returns ``(result, raw_result, window_alive)`` where ``result`` is the
    parsed ``_jobs.<call>`` value (None if the snippet failed) and
    ``window_alive`` is the viewer-window liveness flag carried in the same
    payload (None when unknown, e.g. the snippet did not run cleanly).
    """
    res = host.execute(_job_snippet(call))
    if res.get("status") != "ok":
        return None, res, None
    payload = _extract_json(res.get("stdout", ""))
    if payload is None:
        return None, res, None
    return payload.get("r"), res, payload.get("w")


def _window_note(window_alive) -> str:
    """Closed-window warning to append when a result returns with no viewer.

    Gated on non-headless (headless has no window and carries its own
    messaging). ``window_alive`` is None when liveness is unknown -> no note.
    """
    if not _headless and window_alive is False:
        return _WINDOW_CLOSED_NOTE
    return ""


def _format_job_status(snap: dict) -> str:
    """Render a job snapshot (poll_job output)."""
    job_id = snap.get("job_id", "?")
    status = snap.get("status")
    header = f"{job_id}: {status} ({snap.get('elapsed', '?')}s)"
    body = _format_execute_result(snap)
    if status == "running":
        return header + "\nPartial output:\n" + (body or "(none yet)")
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("guide://kernel")
def get_kernel_guide() -> str:
    """Overview: available namespaces, helper functions, resource URIs."""
    return _resources.GUIDE


@mcp.resource("guide://viewer")
def get_viewer_guide() -> str:
    """Viewer operations: layers, camera, dims, display."""
    return _resources.VIEWER


@mcp.resource("guide://tensor")
def get_tensor_guide() -> str:
    """Tensor data: listing sources, loading, uploading."""
    return _resources.TENSOR


@mcp.resource("guide://annotations")
def get_annotations_guide() -> str:
    """Annotation: points, shapes, labels creation/editing."""
    return _resources.ANNOTATIONS


@mcp.resource("guide://ops")
def get_ops_guide() -> str:
    """Image processing operations: segmentation, feature extraction, super-resolution."""
    return _resources.OPS


@mcp.resource("skill://{skill_id}")
def get_skill(skill_id: str) -> str:
    """Full workflow body for a curated skill; discover ids with `find_skills`.

    The catalog (metadata) is served separately via the `find_skills` tool; this
    resource lazily fetches one skill's markdown body, verifies it against the
    catalog checksum, and caches it. Fail-open: returns a short explanatory
    string rather than erroring when a skill is unknown or unreachable.
    """
    return _skills.get_skill_body(skill_id)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def find_skills(query: str = "") -> list:
    """Discover curated biopb workflows ("skills"). Call at the start of a task.

    Skills are vetted, reusable recipes (e.g. "segment nuclei", "measure
    labels"). `query` filters by title/description/tags; empty returns all.
    Each result includes a `uri` (`skill://<id>`) — read that resource for the
    full step-by-step workflow. Prefer an existing skill over improvising.

    Fail-open: returns an empty list (never errors) when the catalog is
    unreachable and nothing is cached or bundled.
    """
    return _skills.find_skills(query)


@mcp.tool()
def take_screenshot(canvas_only: bool = True) -> list:
    """Capture the napari viewer as a PNG image.

    Args:
        canvas_only: If True, capture only the canvas area. If False,
            capture the entire viewer window.

    Returns a PNG screenshot as an image content block.
    """
    if _headless:
        return [
            TextContent(
                type="text",
                text=(
                    "No screenshot available: this session is running headless "
                    "(started without a display), so there is no viewer window "
                    "to capture. Data loading and compute via execute_code/ops "
                    "still work — tell the user the viewer is unavailable here."
                ),
            )
        ]

    host = _kernel_host
    if host is None:
        return [TextContent(type="text", text="Kernel host not initialized")]

    snippet = _SCREENSHOT_SNIPPET.format(canvas_only=bool(canvas_only))
    res = host.execute(snippet)
    if _extract_delimited(res.get("stdout", ""), _WINDOW_CLOSED_DELIM) is not None:
        return [
            TextContent(
                type="text",
                text=(
                    "No screenshot: the napari viewer window was closed. Data "
                    "access and compute via execute_code still work; call "
                    "restart_kernel to restore a viewer window."
                ),
            )
        ]
    data = _extract_delimited(res.get("stdout", ""), _PNG_DELIM)
    if data is None:
        detail = res.get("error_text") or res.get("stdout") or res.get("status")
        return [TextContent(type="text", text=f"Screenshot failed: {detail}")]
    return [ImageContent(type="image", mimeType="image/png", data=data)]


@mcp.tool()
def execute_code(python_code: str) -> str:
    """Execute Python code in the napari kernel.

    The kernel is a full Jupyter/IPython kernel (imports allowed) with the
    namespace: viewer (with an add_tensor method), client(image data access), and ops (a
    dict of image processing operations). np and da are also imported. Variables persist
    across calls until the kernel is restarted.

    Code runs in a background thread so it does not block the main thread.
    If it finishes quickly the result is returned inline; otherwise this returns
    a job handle (job-N) and the code keeps running. Poll it with poll_job,
    watch it with take_screenshot / server_status, and stop it with cancel_job
    (cooperative) or restart_kernel (guaranteed). Only one job runs at a time.

    Results include print() output and the last expression's repr. Rich IPython
    display() output is not captured; use print().

    * viewer mutations (see guide://viewer for more details):
    The viewer is thread-safe: mutations are auto-marshaled to the Qt main
    thread, so mutate it directly from job code. run_on_main(fn) is optional --
    use it to batch many mutations into one main-thread hop, or to touch raw Qt
    (viewer.window), which still requires the main thread.

    * data access (see guide://tensor for more details):
    - client.query_sources(sql, format="pandas") runs server-side DuckDB and
      returns a DataFrame. The `sources` table columns are: source_id,
      source_url, source_type, dtype, indexed_at, metadata_json, shape_summary,
      data_resident (note source_url, not "url"). Prefer this over
      client.list_sources() (server-capped for large catalogs). Unresolved
      (cloud) sources have NULL dtype/shape_summary, so a `WHERE dtype=...`
      predicate hides them; use `data_resident` to filter on residency on
      purpose (e.g. `WHERE NOT data_resident` to list what isn't resolved yet).
    - viewer.add_tensor(source_id, tensor_id=None) loads a source as a layer
      (auto-handles the multiscale pyramid). client.get_tensor(source_id,
      tensor_id=None) returns a lazy dask array without adding a layer.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    submitted, res, window_alive = _run_job_call(
        host, "submit(" + repr(python_code) + ")"
    )
    if submitted is None:
        return _format_execute_result(res)
    if submitted.get("error") == "busy":
        running = submitted.get("running_job_id")
        return (
            f"A job ({running}) is already running. Poll it with "
            f"poll_job('{running}'), or stop it with cancel_job('{running}') / "
            "restart_kernel before starting another."
        )

    job_id = submitted["job_id"]
    deadline = time.monotonic() + _promote_after
    snap = submitted
    while time.monotonic() < deadline:
        time.sleep(0.4)
        snap, res, window_alive = _run_job_call(host, "poll(" + repr(job_id) + ")")
        if snap is None:
            return _format_execute_result(res)
        if snap.get("status") != "running":
            # terminal: inline result
            return _format_execute_result(snap) + _window_note(window_alive)

    # Still running after promote_after: hand back a job handle.
    partial = snap.get("stdout", "") if snap else ""
    return (
        f"Job {job_id} is still running after {_promote_after:.0f}s. "
        f"Poll it with poll_job('{job_id}'); watch with take_screenshot / "
        f"server_status; stop with cancel_job('{job_id}') or restart_kernel.\n"
        "Partial output:\n" + (partial or "(none yet)") + _window_note(window_alive)
    )


@mcp.tool()
def poll_job(job_id: str) -> str:
    """Get the status and output of a job started by execute_code.

    Returns the job's status (running/ok/error/cancelled), elapsed time, and
    output so far (full output once terminal). Job records persist until the
    kernel is restarted (older terminal jobs are eventually evicted).
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    snap, res, window_alive = _run_job_call(host, "poll(" + repr(job_id) + ")")
    if snap is None:
        return _format_execute_result(res)
    if snap.get("status") == "unknown":
        return f"No such job '{job_id}'."
    note = _window_note(window_alive) if snap.get("status") != "running" else ""
    return _format_job_status(snap) + note


@mcp.tool()
def cancel_job(job_id: str) -> str:
    """Request cancellation of a running job (cooperative; best-effort).

    Sets a flag the job's code can poll via cancelled(), and—if a distributed
    dask cluster is in use—cancels its in-flight futures. Pure-Python loops that
    don't check cancelled(), in-process dask, and native gRPC calls won't stop
    this way; use restart_kernel for a guaranteed stop.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    data, res, _window_alive = _run_job_call(host, "cancel(" + repr(job_id) + ")")
    if data is None:
        return _format_execute_result(res)
    status = data.get("status")
    if data.get("cancelled"):
        return (
            f"Cancellation requested for {job_id} (cooperative). If it does not "
            "stop, use restart_kernel for a guaranteed stop."
        )
    if status == "unknown":
        return f"No such job '{job_id}'."
    return f"Job {job_id}: {status} (nothing to cancel)."


@mcp.tool()
def inspect_object(object_path: str) -> str:
    """Inspect a live object in the napari kernel namespace.

    Returns the type, docstring, and public methods/attributes.
    Example: inspect_object("viewer.layers") or inspect_object("viewer.camera")
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    snippet = _INSPECT_TEMPLATE.replace("__PATH__", repr(object_path))
    res = host.execute(snippet)
    if res.get("status") == "ok":
        return res.get("stdout", "").rstrip() or "(no output)"
    return res.get("error_text") or f"(status: {res.get('status')})"


@mcp.tool()
def interrupt_kernel() -> str:
    """Force-stop the current job by raising KeyboardInterrupt in its thread.

    Stronger than cancel_job: it does not require the code to check cancelled().
    The job runs in a background worker thread, so a SIGINT (which Python delivers
    only to the kernel main thread) can't reach it — this raises the exception
    directly into the worker. Best-effort: it lands at the next bytecode, so a
    blocking C-level call (gRPC tensor fetch, native dask compute) stops only when
    it returns to Python; if the kernel stays stuck, use restart_kernel — the
    guaranteed stop.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    data, res, _w = _run_job_call(host, "interrupt_current()")
    if data is None:
        return _format_execute_result(res)
    if data.get("interrupted"):
        return (
            f"Interrupted job {data.get('job_id')} (KeyboardInterrupt raised in "
            "its thread). If it does not stop, use restart_kernel."
        )
    return "No running job to interrupt."


@mcp.tool()
def start_kernel() -> str:
    """Start the napari kernel on demand (it does not auto-start).

    The MCP server stays cheap and idle until you call this; it then brings up
    the child IPython kernel, dask, the tensor client, and -- unless the session
    is headless -- the napari viewer window. This BLOCKS until the kernel is
    ready (or the bring-up fails), so on return you can use execute_code /
    inspect_object directly, plus take_screenshot when a viewer is present (no
    polling needed). A ready kernel is a no-op. The return message reports
    whether the session is headless.

    Call this once at the start of a session. It is also the recovery path:
    after a failed start, a dead kernel, or the user closing the viewer window
    (which tears the kernel down to idle), call start_kernel again to rebuild.
    (restart_kernel is for hard-restarting an already-running kernel.)
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    result = host.ensure_started()
    if result.get("state") == "ready":
        if _headless:
            # No napari window in a headless session, so take_screenshot is
            # unavailable -- say so rather than claiming a viewer that isn't there.
            return (
                "Kernel ready (headless -- no napari viewer; screenshots "
                "unavailable). dask and the tensor client are up; use "
                "execute_code / inspect_object now."
            )
        return (
            "Kernel ready. The napari viewer, dask, and tensor client are up; "
            "use execute_code / take_screenshot now."
        )
    return (
        "Kernel failed to start: "
        + str(result.get("error", "unknown error"))
        + " Check server_status; call start_kernel to retry."
    )


@mcp.tool()
def restart_kernel() -> str:
    """Hard-restart the kernel: the guaranteed stop for runaway execution.

    Kills the kernel process group (reaping any dask child processes) and
    respawns a fresh kernel, rebuilding the tensor client and -- unless the
    session is headless -- the napari viewer. All variables defined in previous
    execute_code calls are lost; when a viewer is present, a new desktop window
    replaces the old one.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    try:
        host.restart()
    except Exception as exc:
        return f"Kernel restart failed: {exc}"
    if _headless:
        # No napari window in a headless session -- don't claim a rebuilt viewer.
        return (
            "Kernel restarted (headless -- no napari viewer). dask and the "
            "tensor client are up; previous variables are gone."
        )
    return "Kernel restarted. Viewer rebuilt; previous variables are gone."


@mcp.tool()
def server_status() -> str:
    """Report server health, system load, and resource usage.

    Returns CPU/memory usage (this MCP process / host), kernel liveness, and —
    queried from the kernel — dask scheduler info, tensor server
    connectivity, and viewer layer count. Use before heavy computation.
    """
    import psutil

    host = _kernel_host

    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    proc_mem = process.memory_info()

    lines = [
        "## System",
        f"  cpu_usage: {cpu_percent}%",
        f"  cpu_count: {os.cpu_count()}",
        f"  memory_total: {mem.total / (1024**3):.1f} GB",
        f"  memory_available: {mem.available / (1024**3):.1f} GB",
        f"  memory_used_percent: {mem.percent}%",
        f"  process_rss: {proc_mem.rss / (1024**2):.0f} MB",
        f"  log_file: {_session_log_path or 'stdout (not file-logged)'}",
        "",
    ]

    # Observe web UI: server-process state, independent of the kernel, so report
    # it before (and regardless of) kernel health. No kernel round-trip.
    from . import _observe

    obs = _observe.describe(getattr(mcp.settings, "port", None))
    lines.append("## Observe")
    if obs["running"]:
        lines.append(f"  url: {obs['url']}")
        lines.append(f"  mode: {obs['mode']}")
    else:
        lines.append(
            "  status: not running (mcp.observe.enabled off or failed to start)"
        )
    lines.append("")
    lines.append("## Kernel")

    if host is None:
        lines.append("  state: not initialized")
        return "\n".join(lines)

    lines.append(f"  display: {'headless (no viewer)' if _headless else 'visible'}")
    health = host.health()
    lines.append(f"  alive: {health['alive']}")
    lines.append(f"  ready: {health['ready']}")
    lines.append(f"  busy: {health['busy']}")
    lines.append(f"  watchdog_running: {health['watchdog_running']}")
    if health["recent_respawns"]:
        lines.append(f"  recent_respawns: {health['recent_respawns']}")

    # Kernel-state summary: dead / failed / starting / not-started are mutually
    # exclusive (each implies ready is false), so report exactly one and return —
    # don't fall through and print a second, contradictory state. Each also skips
    # the kernel query below, which would block on readiness for the whole
    # startup budget. A user-attributed teardown reason (window close) is shown.
    teardown = health.get("teardown_reason")
    if health["dead"]:
        lines.append("  state: DEAD — respawn budget exhausted; call start_kernel")
        if health.get("start_error"):
            lines.append(f"    last error: {health['start_error']}")
        return "\n".join(lines)
    if not health["ready"]:
        # A recorded start_error means the bring-up failed terminally (vs. still
        # in progress); report it as failed, not "starting", so a broken
        # bootstrap is distinguishable from a slow boot.
        if health.get("start_error"):
            lines.append("  state: failed — kernel startup error:")
            lines.append(f"    {health['start_error']}")
            lines.append("  (call start_kernel to retry)")
        elif health.get("alive"):
            # A kernel process exists but isn't ready yet (e.g. a watchdog
            # respawn in flight). start_kernel itself blocks, so its caller won't
            # see this — but a concurrent observer / respawn can.
            lines.append(
                "  state: starting — kernel/viewer still booting; retry shortly"
            )
        else:
            line = "  state: not started — call start_kernel to launch the kernel"
            if teardown:
                line += f" (torn down: {teardown})"
            lines.append(line)
        return "\n".join(lines)

    res = host.execute(_STATUS_SNIPPET, timeout=15.0)
    if res.get("status") == "ok":
        lines.append("")
        lines.append(res.get("stdout", "").rstrip())
    elif res.get("status") == "busy":
        lines.append("  (kernel busy — dask/tensor/viewer status unavailable)")
    else:
        lines.append("")
        lines.append(
            "  kernel query error: " + (res.get("error_text") or str(res.get("status")))
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def run(port: int = 8765, allowed_origins=(), allowed_hosts=(), *, sock=None):
    """Run the MCP server in the foreground (streamable-http).

    ``allowed_origins`` / ``allowed_hosts`` extend the loopback Host/Origin
    allowlist (see :func:`build_transport_security`).  They are applied before
    serving, when the streamable-http app reads ``transport_security``.

    ``sock`` is an already-bound listening socket. When given we serve over it
    with an explicit ``uvicorn.Server`` instead of letting FastMCP bind ``port``
    itself: the de-daemonized shim-owned child (docs/mcp-dedaemonization-migration.md,
    Layer 1) binds port 0 up front so it can report the OS-assigned port back to
    its shim *before* serving, then hands the socket here. The Starlette app
    FastMCP builds carries the ``session_manager.run()`` lifespan on its own
    (``streamable_http_app``), so a plain uvicorn run drives it — identical to
    the ``mcp.run`` path, only with the socket pre-bound.
    """
    mcp.settings.transport_security = build_transport_security(
        allowed_origins, allowed_hosts
    )
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = port
    logger.info("MCP server listening on http://127.0.0.1:%d/mcp", port)
    if sock is None:
        mcp.run(transport="streamable-http")
        return

    import asyncio

    import uvicorn

    config = uvicorn.Config(
        mcp.streamable_http_app(),
        host="127.0.0.1",
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


# run_stdio() is gone: this process serves http only (daemon migration,
# Direction 1). stdio clients are served by the launcher's bridge mode
# instead — see `_shim`, which fronts this server's /mcp endpoint.
