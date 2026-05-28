"""FastMCP server exposing the napari viewer through a child Jupyter kernel.

The server runs in the foreground (uvicorn, streamable-http on
127.0.0.1:<port>/mcp) and owns a :class:`~napari_biopb.mcp._kernel.KernelHost`.
Every tool call is a round-trip into that kernel, where the napari viewer,
dask, and the TensorFlightClient live.  The kernel can be interrupted or
hard-restarted independently of this process.
"""

import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from . import _resources
from ._kernel import KernelHost

logger = logging.getLogger(__name__)

_kernel_host: Optional[KernelHost] = None

mcp = FastMCP("napari-biopb")

# Prepended to every execute_code call so client tracks the
# asynchronously-connecting Tensor Browser widget.
_REFRESH_PREFIX = "client = _tbw._client\n"

_PNG_DELIM = "<<PNG_B64>>"

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
_tc = _tbw._client
if _tc is not None:
    try:
        print("  connected: true")
        print("  health: " + str(_tc.health_check()))
        print("  sources_cached: " + str(len(_tbw._sources or {})))
    except Exception as _e:
        print("  connected: true")
        print("  health_error: " + str(_e))
else:
    print("  connected: false")

print("")
print("## Viewer")
print("  layers: " + str(len(viewer.layers)))
for _layer in list(viewer.layers)[:10]:
    _shape = getattr(_layer.data, "shape", "?")
    print("    - " + str(_layer.name) + " (" + str(_shape) + ")")
"""


def set_kernel_host(host: KernelHost):
    """Register the kernel host the tools dispatch to."""
    global _kernel_host
    _kernel_host = host


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


def _extract_delimited(text: str, delimiter: str) -> Optional[str]:
    for line in text.splitlines():
        if line.startswith(delimiter):
            return line[len(delimiter) :]
    return None


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("napari://guide")
def get_guide() -> str:
    """Overview: available namespaces, helper functions, resource URIs."""
    return _resources.GUIDE


@mcp.resource("napari://viewer")
def get_viewer_guide() -> str:
    """Viewer operations: layers, camera, dims, display."""
    return _resources.VIEWER


@mcp.resource("napari://tensor")
def get_tensor_guide() -> str:
    """Tensor data: listing sources, loading, uploading."""
    return _resources.TENSOR


@mcp.resource("napari://annotations")
def get_annotations_guide() -> str:
    """Annotation: points, shapes, labels creation/editing."""
    return _resources.ANNOTATIONS


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

    The kernel is a full Jupyter/IPython kernel (imports allowed) with the
    namespace: viewer (with a load_tensor method), np, da, client, sources.
    Use print() to produce output; the last expression's repr is also
    returned. Variables persist across calls until the kernel is restarted.

    Long or runaway executions can be stopped with interrupt_kernel (SIGINT,
    best-effort) or restart_kernel (guaranteed). Read napari://guide for the
    namespace details, and napari://viewer / napari://tensor /
    napari://annotations for domain-specific patterns.
    """
    host = _kernel_host
    if host is None:
        return "Error: kernel host not initialized"

    res = host.execute(_REFRESH_PREFIX + python_code)
    return _format_execute_result(res)


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


def run(port: int = 8765):
    """Run the MCP server in the foreground (streamable-http)."""
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = port
    logger.info("MCP server listening on http://127.0.0.1:%d/mcp", port)
    mcp.run(transport="streamable-http")
