"""Launcher for the biopb-mcp MCP server.

This process *is* the MCP server: it owns a child Jupyter kernel that hosts a
visible napari viewer when a display is available, or a compute-only headless
kernel when none is (see ``mcp.transport.display_mode``).  Run it with::

    biopb-mcp        # console script
    python -m biopb_mcp.mcp

Install the optional dependencies first: ``pip install biopb-mcp[mcp]``.
"""

import argparse
import atexit
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading

logger = logging.getLogger(__name__)


def _parse_args(argv, default_transport, default_port):
    """Parse launcher CLI args (separated out so it is unit-testable)."""
    parser = argparse.ArgumentParser(
        prog="biopb-mcp",
        description="MCP server exposing a napari viewer to an AI agent.",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default=default_transport,
        help="Front-end transport (default from config; falls back to stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port for the http transport (ignored for stdio).",
    )
    return parser.parse_args(argv)


def _has_display():
    """Whether a GUI display is available for a visible napari viewer.

    macOS / Windows always have a window server; on Linux it gates on an X11
    ($DISPLAY) or Wayland ($WAYLAND_DISPLAY) session being set.
    """
    if sys.platform == "darwin" or os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _resolve_headless(display_mode, has_display):
    """Map ``mcp.transport.display_mode`` + display availability to headless.

    ``"headless"`` -> always; ``"visible"`` -> never (the caller fails fast if
    no display); ``"auto"`` / anything else -> headless only when no display.
    """
    if display_mode == "headless":
        return True
    if display_mode == "visible":
        return False
    return not has_display


def _config_defaults(config):
    """Validate/coerce the config-derived launcher defaults.

    argparse only type-checks and constrains *CLI-provided* values, not
    ``default=`` values — so a malformed config (a bad ``transport.kind``
    string, a stringified ``transport.port``) would otherwise flow straight
    through. Return a clean ``(transport, port)`` falling back to the documented
    defaults.
    """
    from .._config import get_setting

    transport = get_setting(config, "mcp.transport.kind")
    if transport not in ("http", "stdio"):
        logger.warning("Unknown mcp.transport.kind %r; using stdio", transport)
        transport = "stdio"
    try:
        port = int(get_setting(config, "mcp.transport.port"))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid mcp.transport.port %r; using 8765",
            get_setting(config, "mcp.transport.port"),
        )
        port = 8765
    return transport, port


def _open_kernel_log(config):
    """Open the file the kernel's native stdout/stderr is redirected to in
    stdio mode (keeps Qt/GL/dask/gRPC output off the JSON-RPC channel).

    Opened binary, append, unbuffered: the handle's fd is the subprocess'
    native stdout/stderr, which emits arbitrary bytes (Qt/GL/dask/gRPC), so a
    byte sink avoids any text-layer translation and matches how KernelHost is
    exercised. On failure, falls back to the launcher's stderr buffer so the
    kernel still starts.
    """
    from .._config import get_config_dir, get_setting

    path = get_setting(config, "mcp.transport.kernel_log") or str(
        get_config_dir() / "kernel.log"
    )
    try:
        return open(path, "ab", buffering=0)
    except OSError:
        logger.warning(
            "Could not open kernel log %s; routing kernel output to stderr",
            path,
        )
        # Native fd redirection needs a byte sink; sys.stderr is a text wrapper,
        # so hand back its binary buffer when present (absent under some test
        # capture shims, where the text stream itself is the safe fallback).
        return getattr(sys.stderr, "buffer", sys.stderr)


def main(argv=None):
    # Log to stderr always: in stdio mode fd 1 is the JSON-RPC channel and any
    # stray byte on it corrupts the stream; stderr is harmless in both modes.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from .._config import get_setting, load_config
    from . import _server
    from ._kernel import KernelHost

    config = load_config()
    default_transport, default_port = _config_defaults(config)
    opts = _parse_args(
        argv,
        default_transport=default_transport,
        default_port=default_port,
    )
    transport = opts.transport
    port = opts.port

    # Decide whether the kernel opens a visible viewer. With no display, a Qt
    # viewer hard-aborts the kernel (SIGABRT, not a catchable error), so unless
    # the user demands "visible" we degrade to a compute-only headless kernel.
    display_mode = get_setting(config, "mcp.transport.display_mode")
    has_display = _has_display()
    if display_mode == "visible" and not has_display:
        kernel_log_path = (
            get_setting(config, "mcp.transport.kernel_log") or "the kernel log"
        )
        logger.error(
            "display_mode='visible' but no display detected "
            "($DISPLAY/$WAYLAND_DISPLAY are unset). Start an X/Wayland session, "
            "or set mcp.transport.display_mode to 'auto' or 'headless'. "
            "(Kernel output: %s)",
            kernel_log_path,
        )
        return 2
    headless = _resolve_headless(display_mode, has_display)

    bootstrap_line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
    extra_arguments = [f"--IPKernelApp.exec_lines={bootstrap_line}"]

    # Pin BLAS/OpenMP to one thread in the kernel.  numpy's OpenBLAS parallel
    # LU path (dgetrf_parallel, reached via np.linalg.inv) allocates a large
    # working buffer on the *caller's* stack; napari's StatusChecker QThread —
    # which inverts the layer affine on every cursor move — has only a ~512 KB
    # stack, so that buffer overruns the guard page and segfaults the viewer
    # (observed on Intel macOS).  These matrices are tiny, so single-threaded
    # BLAS costs nothing here.  Must be set before numpy is imported in the
    # child; setdefault leaves any explicit user override intact.
    kernel_env = os.environ.copy()
    kernel_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    kernel_env.setdefault("OMP_NUM_THREADS", "1")

    # Tell the kernel bootstrap to skip Qt/napari and bind `viewer` to a
    # headless sentinel (compute-only). Resolved here so the launcher owns the
    # display decision (and can fail fast for display_mode='visible').
    if headless:
        kernel_env["BIOPB_HEADLESS"] = "1"

    # Let the data-plane client cache chunks for a localhost tensor server
    # (otherwise skipped as redundant with the server's own cache). Bridges
    # interactive viewer responsiveness: repeated/overlapping plane reads
    # within a chunk hit the cache instead of re-fetching the whole chunk.
    # Inherited by the LocalCluster workers, where the bootstrap's cache plugin
    # bounds each worker at mcp.dask.cache_budget // n_workers.
    if get_setting(config, "mcp.tensor.cache_local"):
        kernel_env.setdefault("BIOPB_CACHE_LOCAL", "1")

    # In stdio mode the kernel must not inherit the launcher's fd 1 (the
    # JSON-RPC channel): redirect its native stdout/stderr to a log file. In
    # http mode it inherits the launcher's fds as before (None).
    kernel_stdout = kernel_stderr = None
    if transport == "stdio":
        kernel_stdout = kernel_stderr = _open_kernel_log(config)

    # Launcher-owned scratch dir for the dask LocalCluster's worker spill files.
    # The launcher rmtree's it on shutdown so a group-SIGKILL of the kernel
    # (which leaves workers no chance to clean up) doesn't leak spill dirs
    # (issue #13, secondary disk-leak note).
    dask_local_dir = tempfile.mkdtemp(prefix="biopb-mcp-dask-")
    kernel_env["BIOPB_DASK_LOCAL_DIR"] = dask_local_dir

    def _cleanup_dask_dir():
        shutil.rmtree(dask_local_dir, ignore_errors=True)

    # Register now (before host.start()) so the scratch dir is still removed on
    # interpreter exit if start() raises. rmtree(ignore_errors) makes this and
    # the explicit calls on the os._exit paths harmless if they both run.
    atexit.register(_cleanup_dask_dir)

    host = KernelHost(
        extra_arguments=extra_arguments,
        kernel_name=get_setting(config, "mcp.kernel.name"),
        startup_timeout=get_setting(config, "mcp.kernel.startup_timeout"),
        execute_timeout=get_setting(config, "mcp.kernel.execute_timeout"),
        busy_lock_timeout=get_setting(config, "mcp.kernel.busy_lock_timeout"),
        env=kernel_env,
        kernel_stdout=kernel_stdout,
        kernel_stderr=kernel_stderr,
        watchdog_interval=get_setting(config, "mcp.kernel.watchdog_interval"),
        watchdog_max_respawns=get_setting(
            config, "mcp.kernel.watchdog_max_respawns"
        ),
        watchdog_respawn_window=get_setting(
            config, "mcp.kernel.watchdog_respawn_window"
        ),
        parent_death_pipe=get_setting(config, "mcp.kernel.parent_death_pipe"),
    )
    _server.set_kernel_host(host)
    _server.set_promote_after(get_setting(config, "mcp.kernel.promote_after"))
    # Surfaces headless state to the agent (initialize `instructions`) and the
    # viewer-dependent tools (take_screenshot / server_status).
    _server.set_headless(headless)

    if headless:
        logger.info(
            "Starting kernel in headless mode (no viewer; no display)."
        )
    else:
        logger.info("Starting napari kernel (a viewer window will appear)...")

    # Bring the kernel up OFF the main thread so the MCP handshake is served
    # immediately. The kernel + napari/Qt viewer + dask bring-up is slow and, on
    # WSL, routinely exceeds an MCP client's startup timeout (e.g. 30s) — which
    # made `biopb-mcp --transport stdio` fail to register there whenever the
    # bring-up was slow. Tools dispatch through KernelHost.execute(), which waits
    # on the host's readiness signal, so a call that lands before the kernel is
    # ready reports "starting" instead of racing a half-built kernel.
    def _start_kernel():
        try:
            host.start()
            logger.info("Kernel ready.")
        except Exception:
            logger.exception(
                "Kernel startup failed; tools will report a terminal startup "
                "error (call restart_kernel to retry). See the kernel log for "
                "the bootstrap traceback."
            )

    threading.Thread(
        target=_start_kernel, name="kernel-start", daemon=True
    ).start()

    # Reap the kernel on exit even if it is still mid-bringup when we stop.
    atexit.register(host.shutdown)

    def _handle_signal(signum, frame):
        logger.info("Received signal %s, shutting down.", signum)
        host.shutdown()
        _cleanup_dask_dir()
        # Skip Python finalization: this process still has a live asyncio/epoll
        # event-loop thread and the numpy OpenBLAS worker pool running, and
        # tearing down the interpreter on top of them segfaults inside
        # Py_FinalizeEx (refcount write into a read-only static-type page).
        # The launcher's only remaining job is to exit, so exit immediately.
        os._exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if transport == "stdio":
        # Client closing stdin (EOF) returns from run_stdio(); the post-run
        # shutdown below then reaps the kernel. No port / Origin allowlist.
        _server.run_stdio()
    else:
        _server.run(
            port,
            allowed_origins=get_setting(
                config, "mcp.transport.allowed_origins"
            ),
            allowed_hosts=get_setting(config, "mcp.transport.allowed_hosts"),
        )

    # If the server loop returns on its own, exit the same way: shut down the
    # kernel and bypass Py_FinalizeEx (atexit handlers do not run after
    # os._exit, so shut down explicitly here).
    host.shutdown()
    _cleanup_dask_dir()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
