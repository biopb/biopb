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

from . import _resources
from ._kernel import KernelHost

logger = logging.getLogger(__name__)

_kernel_host: KernelHost | None = None

# Seconds execute_code waits for a job to finish before returning a job handle
# instead of an inline result (set from config by the launcher).
_promote_after: float = 10.0

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
    ``extra_hosts`` (from ``config['mcp']``) are appended so an admin fronting
    the server with a reverse proxy can permit the proxy's Host/Origin.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_LOOPBACK_HOSTS + list(extra_hosts),
        allowed_origins=_LOOPBACK_ORIGINS + list(extra_origins),
    )


mcp = FastMCP("biopb-mcp", transport_security=build_transport_security())

_PNG_DELIM = "<<PNG_B64>>"

# Delimiter for the single-line JSON payload the in-kernel job runner prints in
# reply to a submit/poll/cancel/list snippet (mirrors the _PNG_DELIM pattern).
_JOB_DELIM = "<<JOB_JSON>>"


def _job_snippet(call: str) -> str:
    """Build a snippet that prints ``_jobs.<call>``'s result as delimited JSON.

    ``call`` is a fully-formed call expression with arguments already embedded
    via ``repr`` by the caller (agent code is RCE by design, but embedding via
    ``repr`` keeps the payload a valid literal regardless of its contents).
    """
    return (
        "import json as _json\n"
        "print('" + _JOB_DELIM + "' + _json.dumps(_jobs." + call + "))\n"
    )


_SCREENSHOT_SNIPPET = (
    "import base64 as _b64, cv2 as _cv2\n"
    "_arr = viewer.screenshot(canvas_only={canvas_only})\n"
    "_bgra = _cv2.cvtColor(_arr, _cv2.COLOR_RGBA2BGRA)\n"
    "_ok, _buf = _cv2.imencode('.png', _bgra)\n"
    "print('" + _PNG_DELIM + "' + _b64.b64encode(_buf.tobytes()).decode())\n"
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

print("")
print("## Viewer")
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
    """Run a ``_jobs.<call>`` snippet; return (parsed_json, raw_result)."""
    res = host.execute(_job_snippet(call))
    if res.get("status") != "ok":
        return None, res
    return _extract_json(res.get("stdout", "")), res


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


@mcp.resource("guide://main")
def get_guide() -> str:
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def take_screenshot(canvas_only: bool = True) -> list:
    """Capture the napari viewer as a PNG image.

    Args:
        canvas_only: If True, capture only the canvas area. If False,
            capture the entire viewer window.

    Returns a PNG screenshot as an image content block.
    """
    host = _kernel_host
    if host is None:
        return [TextContent(type="text", text="Kernel host not initialized")]

    snippet = _SCREENSHOT_SNIPPET.format(canvas_only=bool(canvas_only))
    res = host.execute(snippet)
    data = _extract_delimited(res.get("stdout", ""), _PNG_DELIM)
    if data is None:
        detail = (
            res.get("error_text") or res.get("stdout") or res.get("status")
        )
        return [TextContent(type="text", text=f"Screenshot failed: {detail}")]
    return [ImageContent(type="image", mimeType="image/png", data=data)]


@mcp.tool()
def execute_code(python_code: str) -> str:
    """Execute Python code in the napari kernel.

    **READ THE `guide://main` RESOURCE FOR EXECUTION GUARDRAILS BEFORE USE.**

    The kernel is a full Jupyter/IPython kernel (imports allowed) with the
    namespace: viewer (with a load_tensor method), np, da, client, and ops (a
    dict of biopb.image ProcessImage operations). Use print() to produce
    output; the last expression's repr is also returned. Variables persist
    across calls until the kernel is restarted.

    Code runs in a background thread, so a long job does NOT block the viewer:
    if it finishes quickly the result is returned inline; otherwise this returns
    a job handle (job-N) and the code keeps running. Poll it with poll_job,
    watch it with take_screenshot / server_status, and stop it with cancel_job
    (cooperative) or restart_kernel (guaranteed). Only one job runs at a time.

    Inside a long job, mutate the viewer via the auto-wrapped load_tensor /
    add_* methods or run_on_main(fn) (GUI calls must reach the main thread);
    note rich IPython display() output is not captured (use print). Read
    guide://main for namespace details and guide://viewer / guide://tensor /
    guide://annotations for domain-specific patterns.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    submitted, res = _run_job_call(host, "submit(" + repr(python_code) + ")")
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
        snap, res = _run_job_call(host, "poll(" + repr(job_id) + ")")
        if snap is None:
            return _format_execute_result(res)
        if snap.get("status") != "running":
            return _format_execute_result(snap)  # terminal: inline result

    # Still running after promote_after: hand back a job handle.
    partial = snap.get("stdout", "") if snap else ""
    return (
        f"Job {job_id} is still running after {_promote_after:.0f}s. "
        f"Poll it with poll_job('{job_id}'); watch with take_screenshot / "
        f"server_status; stop with cancel_job('{job_id}') or restart_kernel.\n"
        "Partial output:\n" + (partial or "(none yet)")
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

    snap, res = _run_job_call(host, "poll(" + repr(job_id) + ")")
    if snap is None:
        return _format_execute_result(res)
    if snap.get("status") == "unknown":
        return f"No such job '{job_id}'."
    return _format_job_status(snap)


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

    data, res = _run_job_call(host, "cancel(" + repr(job_id) + ")")
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
    """Interrupt the current kernel execution by sending SIGINT.

    Best-effort: pure-Python loops stop with KeyboardInterrupt, but blocking
    C-level calls (gRPC tensor fetches, native dask compute) ignore SIGINT.
    If the kernel stays unresponsive, use restart_kernel — the guaranteed stop.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    host.interrupt()
    return "Interrupt signal (SIGINT) sent to kernel."


@mcp.tool()
def restart_kernel() -> str:
    """Hard-restart the kernel: the guaranteed stop for runaway execution.

    Kills the kernel process group (reaping any dask child processes) and
    respawns a fresh kernel, rebuilding the napari viewer and tensor client.
    All variables defined in previous execute_code calls are lost, and a new
    desktop viewer window replaces the old one.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"
    try:
        host.restart()
    except Exception as exc:
        return f"Kernel restart failed: {exc}"
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
        f"  memory_total: {mem.total / (1024 ** 3):.1f} GB",
        f"  memory_available: {mem.available / (1024 ** 3):.1f} GB",
        f"  memory_used_percent: {mem.percent}%",
        f"  process_rss: {proc_mem.rss / (1024 ** 2):.0f} MB",
        "",
        "## Kernel",
    ]

    if host is None:
        lines.append("  state: not initialized")
        return "\n".join(lines)

    lines.append(f"  alive: {host.is_alive()}")
    lines.append(f"  busy: {host.is_busy()}")

    res = host.execute(_STATUS_SNIPPET, timeout=15.0)
    if res.get("status") == "ok":
        lines.append("")
        lines.append(res.get("stdout", "").rstrip())
    elif res.get("status") == "busy":
        lines.append("  (kernel busy — dask/tensor/viewer status unavailable)")
    else:
        lines.append("")
        lines.append(
            "  kernel query error: "
            + (res.get("error_text") or str(res.get("status")))
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def run(port: int = 8765, allowed_origins=(), allowed_hosts=()):
    """Run the MCP server in the foreground (streamable-http).

    ``allowed_origins`` / ``allowed_hosts`` extend the loopback Host/Origin
    allowlist (see :func:`build_transport_security`).  They are applied before
    serving, when the streamable-http app reads ``transport_security``.
    """
    mcp.settings.transport_security = build_transport_security(
        allowed_origins, allowed_hosts
    )
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = port
    logger.info("MCP server listening on http://127.0.0.1:%d/mcp", port)
    mcp.run(transport="streamable-http")
