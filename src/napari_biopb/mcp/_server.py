"""FastMCP server with 3 tools and 4 resources for napari-biopb.

The server runs on a daemon thread via uvicorn, serving streamable-http
on 127.0.0.1:<port>/mcp. All viewer mutations are dispatched through the
ThreadBridge so they execute on the Qt main thread.
"""

import base64
import inspect
import io
import logging
import sys
import threading
from typing import Optional

import dask
import dask.array as da
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from . import _resources
from ._bridge import ThreadBridge
from ._helpers import patch_viewer_load_tensor

logger = logging.getLogger(__name__)

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "zip": zip,
    "enumerate": enumerate,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
    "type": type,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "slice": slice,
    "getattr": getattr,
    "setattr": setattr,
    "hasattr": hasattr,
    "dir": dir,
    "repr": repr,
    "True": True,
    "False": False,
    "None": None,
}

_bridge: Optional[ThreadBridge] = None
_server_thread: Optional[threading.Thread] = None

mcp = FastMCP("napari-biopb")


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


def _build_namespace(bridge: ThreadBridge) -> dict:
    return {
        "viewer": bridge.viewer,
        "np": np,
        "da": da,
        "client": bridge.tensor_client,
        "sources": bridge.tensor_sources,
        "__builtins__": _SAFE_BUILTINS,
    }


@mcp.tool()
def take_screenshot(canvas_only: bool = True) -> list:
    """Capture the napari viewer as a PNG image.

    Args:
        canvas_only: If True, capture only the canvas area. If False,
            capture the entire viewer window.

    Returns a PNG screenshot as an image content block.
    """
    if _bridge is None:
        return [TextContent(type="text", text="MCP bridge not initialized")]

    try:
        from PIL import Image
    except ImportError:
        return [
            TextContent(
                type="text",
                text="Pillow is not installed. "
                "Install with: pip install 'napari-biopb[mcp]'",
            )
        ]

    def _capture(canvas_only=canvas_only):
        return _bridge.viewer.screenshot(canvas_only=canvas_only)

    arr = _bridge.run_on_gui_thread(_capture)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()

    return [ImageContent(type="image", mimeType="image/png", data=data)]


@mcp.tool()
def execute_code(python_code: str) -> str:
    """Execute Python code in the napari namespace.

    The namespace includes: viewer (with load_tensor method), np, da,
    client, sources. Use print() to produce output. The last expression's
    repr is appended if it is not None.

    Read the napari://guide resource for details on available objects and
    helpers.  Read napari://viewer, napari://tensor, or napari://annotations
    for domain-specific patterns.
    """
    if _bridge is None:
        return "Error: MCP bridge not initialized"

    def _exec(code=python_code):
        namespace = _build_namespace(_bridge)
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            # Try as expression first for implicit return
            try:
                result = eval(code, namespace)
                output = captured.getvalue()
                if result is not None:
                    output += repr(result)
                return output or "(no output)"
            except SyntaxError:
                pass

            exec(code, namespace)
            output = captured.getvalue()
            return output or "(no output)"
        except Exception as exc:
            output = captured.getvalue()
            return f"{output}Error: {exc}"
        finally:
            sys.stdout = old_stdout

    return _bridge.run_on_gui_thread(_exec)


@mcp.tool()
def inspect_object(object_path: str) -> str:
    """Inspect a live object in the napari namespace.

    Returns the type, docstring, and public methods/attributes.
    Example: inspect_object("viewer.layers") or inspect_object("viewer.camera")
    """
    if _bridge is None:
        return "Error: MCP bridge not initialized"

    def _inspect(path=object_path):
        namespace = _build_namespace(_bridge)
        safe_globals = {"__builtins__": {}}
        try:
            obj = eval(path, safe_globals, namespace)
        except Exception as exc:
            return f"Error resolving '{path}': {exc}"

        obj_type = type(obj).__name__
        obj_doc = inspect.getdoc(obj) or "No documentation."

        lines = [
            f"Type: {obj_type}",
            f"Docstring: {obj_doc}",
            "",
            "Attributes:",
        ]

        for name in sorted(dir(obj)):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(obj, name)
            except Exception:
                continue
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                try:
                    sig = inspect.signature(attr)
                    short_doc = (inspect.getdoc(attr) or "").split("\n")[0]
                    lines.append(f"  .{name}{sig}  -- {short_doc}")
                except (ValueError, TypeError):
                    lines.append(f"  .{name}(...)")
            else:
                lines.append(f"  .{name} [{type(attr).__name__}]")

        return "\n".join(lines)

    return _bridge.run_on_gui_thread(_inspect)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _configure_dask(mcp_config: dict):
    """Set up the dask environment from MCP config."""
    scheduler = mcp_config.get("dask_scheduler", "threads")
    num_workers = mcp_config.get("dask_num_workers", 0) or None
    address = mcp_config.get("dask_distributed_address", "")

    if scheduler == "distributed" and address:
        from distributed import Client

        client = Client(address)
        logger.info("Dask using distributed scheduler at %s", address)
        return client

    dask.config.set(scheduler=scheduler, num_workers=num_workers)
    logger.info("Dask scheduler: %s, num_workers: %s", scheduler, num_workers)
    return None


def launch_server(
    bridge: ThreadBridge, port: int = 8765, mcp_config: dict = None
):
    """Start the MCP server on a daemon thread.

    Must be called from the Qt main thread (so the bridge timer starts
    on the right thread).
    """
    global _bridge, _server_thread
    import uvicorn

    if mcp_config is None:
        mcp_config = {}
    _configure_dask(mcp_config)

    _bridge = bridge
    patch_viewer_load_tensor(bridge)
    bridge.start_timer()

    app = mcp.streamable_http_app()

    _server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "log_level": "warning",
        },
        daemon=True,
    )
    _server_thread.start()
    logger.info("MCP server listening on http://127.0.0.1:%d/mcp", port)


def shutdown_server():
    """Stop the bridge timer (the daemon thread dies with the process)."""
    global _bridge
    if _bridge is not None:
        _bridge.stop_timer()
        _bridge = None
        logger.info("MCP bridge stopped")
