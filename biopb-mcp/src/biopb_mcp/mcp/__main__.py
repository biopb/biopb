"""Launcher for the biopb-mcp MCP server.

Under the http transport this process *is* the MCP server: it owns a child
Jupyter kernel that hosts a visible napari viewer when a display is available,
or a compute-only headless kernel when none is (see
``mcp.transport.display_mode``).  Under the (deprecated) stdio transport it is
instead a thin bridge: it ensures the http daemon is running on the configured
loopback port — spawning it detached if needed — and pumps stdio JSON-RPC to
it (see ``_shim``).  Run it with::

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
import socket
import sys
import tempfile

logger = logging.getLogger(__name__)


def _port_listening(port, timeout=0.5):
    """Whether something already accepts TCP connections on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _self_create_time():
    """This process's creation-time token, or None if it can't be determined.

    Delegates to biopb._proc.process_create_time (the single source of truth
    shared with biopb.cli, so the token this daemon writes is computed exactly
    the way the CLI reads it) for os.getpid(): a per-process identity that lets
    `biopb mcp stop`/`status` tell our daemon apart from an unrelated process
    that later inherits a reused PID (Windows never cleans the PID file at logout
    and recycles PIDs aggressively). None -> the CLI falls back to a liveness-only
    check, the pre-fix behavior. Best-effort: any failure degrades to None.
    """
    try:
        from biopb._proc import process_create_time

        return process_create_time(os.getpid())
    except Exception:
        return None


def _pidfile_contents():
    """The text to write into the PID file: `pid` plus, when known, a `pid\\ntoken`
    create-time identity. Read back by biopb.cli._read_pid_record (two whitespace-
    separated ints, tolerant of the legacy bare-pid form)."""
    pid = os.getpid()
    token = _self_create_time()
    return f"{pid}\n{token}" if token is not None else str(pid)


def _write_pidfile(port):
    """Best-effort: record this daemon's PID so `biopb mcp status` finds it.

    The PID file is the one signal `status` trusts, and the stdio shim spawns
    the daemon detached without writing it — so the daemon registers itself
    here, covering every launch path uniformly. Best-effort: a write failure
    only costs `status` visibility, never the server.

    Records a create-time identity token alongside the PID (see
    :func:`_self_create_time`) so the CLI can distinguish this daemon from an
    unrelated process that later inherits a reused PID.

    Concurrent first-run shims can each spawn a daemon; only the one that binds
    the port survives (the rest die on EADDRINUSE). Re-checking the port
    immediately before writing keeps a soon-to-die loser from clobbering the
    winner's PID; pid-safe removal (see :func:`_remove_pidfile`) keeps a loser's
    exit from deleting the winner's file.
    """
    from .._config import get_pid_file

    pidfile = get_pid_file()
    try:
        if _port_listening(port):
            # Someone already owns the port; we are about to lose the bind.
            return None
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(_pidfile_contents())
        return pidfile
    except OSError:
        logger.warning("Could not write PID file %s", pidfile, exc_info=True)
        return None


def _remove_pidfile(pidfile):
    """Remove our PID file, but only if it still names this process.

    Pid-safe so a losing daemon's exit never deletes the winner's file. Compares
    only the first whitespace-separated field (the PID), ignoring any trailing
    create-time token, so the match holds regardless of token presence.
    """
    if pidfile is None:
        return
    try:
        if pidfile.read_text().split()[0] == str(os.getpid()):
            pidfile.unlink()
    except (OSError, ValueError, IndexError):
        pass


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
        help="Front-end transport (default from config; falls back to stdio). "
        "stdio is deprecated: it is now served by bridging to the local http "
        "daemon (spawned on demand); prefer connecting over http directly.",
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


def _setup_observe(config):
    """Wire up the web observe UI.

    On by default (``mcp.observe.enabled``, opt-out); it mounts on the existing
    MCP app and shares its loop/port. Fully guarded — an observe failure logs
    and is swallowed so it can never block the MCP server. Returns True if
    mounted.
    """
    from .._config import get_setting

    if not get_setting(config, "mcp.observe.enabled"):
        return False
    try:
        from . import _observe

        _observe.configure(
            max_output_chars=get_setting(
                config, "mcp.observe.max_output_chars"
            ),
            poll_interval_ms=get_setting(
                config, "mcp.observe.poll_interval_ms"
            ),
            allowed_origins=get_setting(
                config, "mcp.transport.allowed_origins"
            ),
            allowed_hosts=get_setting(config, "mcp.transport.allowed_hosts"),
        )
        _observe.register_http_routes()
        return True
    except Exception:
        logger.exception("observe UI failed to start; continuing without it")
        return False


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


def main(argv=None):
    # Log to stderr always: in stdio (bridge) mode fd 1 is the JSON-RPC
    # channel and any stray byte on it corrupts the stream; stderr is harmless
    # in both modes.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from .._config import load_config

    config = load_config()
    default_transport, default_port = _config_defaults(config)
    opts = _parse_args(
        argv,
        default_transport=default_transport,
        default_port=default_port,
    )

    if opts.transport == "stdio":
        # Bridge mode: keep this process featherweight — the heavy stack
        # (FastMCP/uvicorn/kernel plumbing) is only imported by the daemon it
        # spawns. Any bridge failure exits nonzero so the client sees EOF
        # rather than a hung server entry.
        from . import _shim

        try:
            _shim.serve(config, opts.port)
        except Exception:
            logger.exception("stdio bridge failed")
            return 1
        return 0

    return _serve_http(config, opts.port)


def _serve_http(config, port):
    """Run the real MCP server (streamable-http) in the foreground."""
    from .._config import get_setting
    from . import _server
    from ._kernel import KernelHost

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

    # The kernel inherits this process' fds. fd 1 is not a protocol channel
    # under http, so native Qt/GL/dask/gRPC output is harmless: it lands on
    # the launcher's stdout/stderr — which, when the daemon was spawned by the
    # stdio shim, is the daemon log file (see _shim._open_daemon_log).

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
        watchdog_interval=get_setting(config, "mcp.kernel.watchdog_interval"),
        watchdog_max_respawns=get_setting(
            config, "mcp.kernel.watchdog_max_respawns"
        ),
        watchdog_respawn_window=get_setting(
            config, "mcp.kernel.watchdog_respawn_window"
        ),
        parent_death_pipe=get_setting(config, "mcp.kernel.parent_death_pipe"),
        # The window-close pipe only matters with a viewer; a headless kernel
        # has no window to close, so don't wire it up.
        window_close_pipe=not headless,
    )
    _server.set_kernel_host(host)
    _server.set_promote_after(get_setting(config, "mcp.kernel.promote_after"))
    # Surfaces headless state to the agent (initialize `instructions`) and the
    # viewer-dependent tools (take_screenshot / server_status).
    _server.set_headless(headless)

    # On-demand start: the kernel is NOT launched here. The server stays cheap
    # and idle (no viewer window pops, no Qt abort on a display-less daemon)
    # until an agent calls the `start_kernel` tool, which drives
    # host.ensure_started() — a synchronous bring-up that blocks that one tool
    # call until the kernel is ready. Other tool calls landing before then get a
    # structured "not started" status (see KernelHost.execute).
    if headless:
        logger.info(
            "Headless mode (no viewer; no display). Kernel starts on the first "
            "start_kernel call."
        )
    else:
        logger.info(
            "Ready. The napari kernel (and viewer window) starts on the first "
            "start_kernel call."
        )

    # Reap the kernel on exit even if it is still mid-bringup when we stop
    # (a no-op safe on an idle, never-started host).
    atexit.register(host.shutdown)

    # Register the daemon's PID so `biopb mcp status` can detect it no matter
    # how it was launched (CLI, stdio shim, or manual). Written just before the
    # bind below; removed pid-safely on every exit path.
    pidfile = _write_pidfile(port)
    atexit.register(_remove_pidfile, pidfile)

    def _handle_signal(signum, frame):
        logger.info("Received signal %s, shutting down.", signum)
        host.shutdown()
        _remove_pidfile(pidfile)
        _cleanup_dask_dir()
        # Skip Python finalization: this process still has a live asyncio/epoll
        # event-loop thread and the numpy OpenBLAS worker pool running, and
        # tearing down the interpreter on top of them segfaults inside
        # Py_FinalizeEx (refcount write into a read-only static-type page).
        # The launcher's only remaining job is to exit, so exit immediately.
        os._exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Opt-in web "observe" UI. Set up before the (blocking) transport run:
    # custom routes are read when the streamable-http app is built.
    _setup_observe(config)

    _server.run(
        port,
        allowed_origins=get_setting(config, "mcp.transport.allowed_origins"),
        allowed_hosts=get_setting(config, "mcp.transport.allowed_hosts"),
    )

    # If the server loop returns on its own, exit the same way: shut down the
    # kernel and bypass Py_FinalizeEx (atexit handlers do not run after
    # os._exit, so shut down explicitly here).
    host.shutdown()
    _remove_pidfile(pidfile)
    _cleanup_dask_dir()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
