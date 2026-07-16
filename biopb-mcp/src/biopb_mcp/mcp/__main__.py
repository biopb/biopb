"""Launcher for the biopb-mcp MCP server.

Under the http transport this process *is* the MCP server: it owns a child
Jupyter kernel that hosts a visible napari viewer when a display is available,
or a compute-only headless kernel when none is (see
``transport.display_mode``).  Under the (deprecated) stdio transport it is
instead a thin bridge: it spawns its own http session child on a dynamic port
and pumps stdio JSON-RPC to it, reaping it on disconnect (see ``_shim``).  Run
it with::

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


# Env var carrying the path of the file a shim-owned child publishes its
# OS-assigned port to (de-daemonization Layer 1). Presence of this var is also
# how _serve_http tells a shim-owned child (dynamic port, reported back) from a
# direct `--transport http` launch (fixed port). Kept in sync with _shim.
ENV_PORT_REPORT_FILE = "BIOPB_PORT_REPORT_FILE"

# Env var the stdio shim sets to the child's own session logfile path, so the
# child can report it (server_status) and the agent's execute_code can read it
# from os.environ. Kept in sync with _shim.ENV_SESSION_LOG.
ENV_SESSION_LOG = "BIOPB_MCP_SESSION_LOG"


def _report_port(path, port):
    """Publish the OS-assigned ``port`` to the shim's report file.

    The stdio shim (docs/mcp-dedaemonization-migration.md, Layer 1) spawns this
    child with ``--port 0`` and a unique ``BIOPB_PORT_REPORT_FILE``, then polls
    that file for the real port to build its bridge URL. Written atomically
    (temp + ``os.replace``) so the shim never reads a half-written value.

    A cross-platform file rather than the inherited-pipe handshake used for the
    kernel token (``BIOPB_TOKEN_REPORT_FD``): that pipe pattern is POSIX-only
    here (``_kernel`` gates it on ``os.name == 'posix'`` — fd inheritance across
    a Windows spawn is fragile), whereas a file is uniform. Best-effort: a write
    failure only costs the shim its port (it times out; the client sees EOF),
    never the server.
    """
    try:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(str(port))
        os.replace(tmp, path)
    except OSError:
        logger.warning("Could not write port report file %s", path, exc_info=True)


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
        "stdio is deprecated: it is now served by bridging to a private http "
        "session child the shim spawns on demand; prefer connecting over http "
        "directly.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port for the http transport (ignored for stdio).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Agentless viewer: open the napari viewer directly in the "
        "foreground and block until Ctrl-C. Forces a visible display and an "
        "eager kernel start (the window opens now, not on a start_kernel call); "
        "still serves /mcp on a dynamic port for optional agent attach. A "
        "user-owned foreground session. Fronted by `biopb mcp view`.",
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
    """Map ``transport.display_mode`` + display availability to headless.

    ``"headless"`` -> always; ``"visible"`` -> never (the caller fails fast if
    no display); ``"auto"`` / anything else -> headless only when no display.
    """
    if display_mode == "headless":
        return True
    if display_mode == "visible":
        return False
    return not has_display


def _resolve_headless_logged(display_mode, has_display):
    """:func:`_resolve_headless`, but warn on a silent ``auto`` -> headless.

    When ``display_mode='auto'`` degrades to headless purely because no display
    was found, the viewer and screenshots silently go away -- the exact silent
    path #98 named. Surface it once so the operator knows why. An explicit
    ``'headless'`` choice is intentional and needs no warning; the ``'visible'``
    + no-display case is handled (fatally) by the caller.
    """
    headless = _resolve_headless(display_mode, has_display)
    if headless and display_mode != "headless" and not has_display:
        logger.warning(
            "display_mode='auto' resolved to headless: no display detected "
            "($DISPLAY/$WAYLAND_DISPLAY are unset), so the kernel runs "
            "compute-only (no napari viewer; screenshots unavailable). Set "
            "transport.display_mode to 'visible' once a display is available."
        )
    return headless


def _setup_observe(config):
    """Wire up the web observe UI.

    On by default (``observe.enabled``, opt-out); it mounts on the existing
    MCP app and shares its loop/port. Fully guarded — an observe failure logs
    and is swallowed so it can never block the MCP server. Returns True if
    mounted.
    """
    from .._config import get_setting

    if not get_setting(config, "observe.enabled"):
        return False
    try:
        from . import _observe

        _observe.configure(
            max_output_chars=get_setting(config, "observe.max_output_chars"),
            poll_interval_ms=get_setting(config, "observe.poll_interval_ms"),
            allowed_origins=get_setting(config, "transport.allowed_origins"),
            allowed_hosts=get_setting(config, "transport.allowed_hosts"),
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

    transport = get_setting(config, "transport.kind")
    if transport not in ("http", "stdio"):
        logger.warning("Unknown transport.kind %r; using stdio", transport)
        transport = "stdio"
    try:
        port = int(get_setting(config, "transport.port"))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid transport.port %r; using 8765",
            get_setting(config, "transport.port"),
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

    if opts.view:
        # Agentless foreground viewer (fronted by `biopb mcp view`): serve http
        # with a visible, eagerly-started viewer, regardless of the configured
        # transport. Blocks until Ctrl-C.
        return _serve_http(config, opts.port, view=True)

    if opts.transport == "stdio":
        # Bridge mode: keep this process featherweight — the heavy stack
        # (FastMCP/uvicorn/kernel plumbing) is only imported by the owned session
        # child it spawns. Any bridge failure exits nonzero so the client sees EOF
        # rather than a hung server entry.
        from . import _shim

        try:
            _shim.serve(config, opts.port)
        except Exception:
            logger.exception("stdio bridge failed")
            return 1
        return 0

    return _serve_http(config, opts.port)


def _serve_http(config, port, view=False):
    """Run the real MCP server (streamable-http) in the foreground.

    ``view`` selects the agentless-viewer mode (`biopb mcp view`): force a
    visible display, bind a dynamic port and print its URL, and start the
    kernel/viewer eagerly so the window opens immediately instead of on the
    first ``start_kernel`` tool call.
    """
    from .._config import get_setting
    from . import _server
    from ._cluster import DaskClusterHost
    from ._kernel import KernelHost

    # Windows: serve on the Selector event loop, not the default Proactor one
    # (biopb/biopb#383). The Proactor accept loop treats *any* OSError from
    # AcceptEx as fatal -- it closes the listening socket and never re-arms
    # accept -- leaving the server "alive but not serving."
    # The Selector loop's also silences zmq's "Proactor does not implement
    # add_reader" warning, since jupyter_client's kernel channels want exactly
    # this loop. Safe to set because both child kernel and Dask's `LocalCluster`
    # uses synchronous `subprocess.Popen`, so the Selector loop's lack of
    # asyncio-subprocess support is fine. Caveat: the Windows Selector loop is
    # select()-based (FD_SETSIZE 512); this single-agent localhost transport
    # handles only the listener plus a handful of /mcp + observe connections,
    # far under that ceiling.
    if os.name == "nt":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Decide whether the kernel opens a visible viewer. With no display, a Qt
    # viewer hard-aborts the kernel (SIGABRT, not a catchable error), so unless
    # the user demands "visible" we degrade to a compute-only headless kernel.
    # `--view` demands visible by definition (the human wants the window).
    display_mode = "visible" if view else get_setting(config, "transport.display_mode")
    has_display = _has_display()
    if display_mode == "visible" and not has_display:
        kernel_log_path = (
            get_setting(config, "transport.kernel_log") or "the kernel log"
        )
        logger.error(
            "display_mode='visible' but no display detected "
            "($DISPLAY/$WAYLAND_DISPLAY are unset). Start an X/Wayland session, "
            "or set transport.display_mode to 'auto' or 'headless'. "
            "(Kernel output: %s)",
            kernel_log_path,
        )
        return 2
    headless = _resolve_headless_logged(display_mode, has_display)

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

    # The kernel inherits this process' fds. fd 1 is not a protocol channel
    # under http, so native Qt/GL/dask/gRPC output is harmless: it lands on
    # the launcher's stdout/stderr — which, for a shim-spawned session child, is
    # that session's log file (biopb._lifecycle.owned_child.open_child_log).

    # Launcher-owned scratch dir for the dask LocalCluster's worker spill files.
    # The launcher rmtree's it on shutdown so a group-SIGKILL of the kernel
    # (which leaves workers no chance to clean up) doesn't leak spill dirs
    # (issue #13, secondary disk-leak note). Consumed by the session-child-owned
    # cluster (below) via DaskClusterHost.local_dir.
    dask_local_dir = tempfile.mkdtemp(prefix="biopb-mcp-dask-")

    def _cleanup_dask_dir():
        shutil.rmtree(dask_local_dir, ignore_errors=True)

    # Register now (before host.start()) so the scratch dir is still removed on
    # interpreter exit if start() raises. rmtree(ignore_errors) makes this and
    # the explicit calls on the os._exit paths harmless if they both run.
    atexit.register(_cleanup_dask_dir)

    # Session-child-owned dask cluster: spun lazily on the first kernel launch
    # (from KernelHost._launch, which injects its address), kept warm across
    # kernel restarts, and closed only on real process exit (the _shutdown
    # chokepoint + atexit backstop). Detaching the cluster from the kernel avoids
    # re-spinning N cold workers on every restart_kernel — the dominant restart
    # cost on Windows (no fork). Construction is cheap (no dask import until
    # ensure()); atexit is a backstop for exits that skip _shutdown.
    cluster_host = DaskClusterHost(config, local_dir=dask_local_dir)
    atexit.register(cluster_host.close)

    host = KernelHost(
        extra_arguments=extra_arguments,
        kernel_name=get_setting(config, "kernel.name"),
        startup_timeout=get_setting(config, "kernel.startup_timeout"),
        execute_timeout=get_setting(config, "kernel.execute_timeout"),
        busy_lock_timeout=get_setting(config, "kernel.busy_lock_timeout"),
        env=kernel_env,
        watchdog_interval=get_setting(config, "kernel.watchdog_interval"),
        watchdog_max_respawns=get_setting(config, "kernel.watchdog_max_respawns"),
        watchdog_respawn_window=get_setting(config, "kernel.watchdog_respawn_window"),
        parent_death_pipe=get_setting(config, "kernel.parent_death_pipe"),
        # The window-close pipe only matters with a viewer; a headless kernel
        # has no window to close, so don't wire it up.
        window_close_pipe=not headless,
        # Session-child-owned dask cluster; _launch calls ensure() and injects
        # its scheduler address so the kernel attaches instead of spinning its own.
        cluster_host=cluster_host,
    )
    _server.set_kernel_host(host)
    _server.set_promote_after(get_setting(config, "kernel.promote_after"))
    # Surfaces headless state to the agent (initialize `instructions`) and the
    # viewer-dependent tools (take_screenshot / server_status).
    _server.set_headless(headless)

    # Tell server_status where this process's log lives, so an agent can find it.
    #   * shim session -> the per-session file (BIOPB_MCP_SESSION_LOG, set by the
    #     shim); also visible to execute_code via os.environ.
    #   * a non-tty direct `--transport http` launch (output redirected to a
    #     file) -> the canonical mcp-server.log.
    #   * a terminal (foreground `--transport http` / `biopb mcp view`) -> None,
    #     reported as stdout.
    from .._config import get_daemon_log_file

    if os.environ.get(ENV_SESSION_LOG):
        session_log = os.environ[ENV_SESSION_LOG]
    elif not sys.stdout.isatty():
        session_log = str(get_daemon_log_file(config))
    else:
        session_log = None
    _server.set_session_log_path(session_log)

    # On-demand start: the kernel is NOT launched here. The server stays cheap
    # and idle (no viewer window pops, no Qt abort on a display-less server)
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

    # Two foreground modes bind a *dynamic* port and report it back rather than
    # binding the configured fixed port:
    #   * the de-daemonized shim-owned child (Layer 1) — the shim set
    #     BIOPB_PORT_REPORT_FILE and passed --port 0; it reaps us directly (own
    #     process group / Job Object) and we report the OS-assigned port back;
    #   * the agentless `biopb mcp view` viewer — a user-owned Ctrl-C session; it
    #     prints its URL instead.
    # A direct `--transport http` binds the configured port. The POSIX signal
    # handlers below reap our kernel gracefully in every mode.
    report_file = os.environ.get(ENV_PORT_REPORT_FILE)
    shim_owned = bool(report_file)
    dynamic_port = shim_owned or view
    listen_sock = None
    if dynamic_port:
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind(("127.0.0.1", port))  # port 0 -> OS assigns one
        port = listen_sock.getsockname()[1]
        if shim_owned:
            _report_port(report_file, port)
        else:  # view
            print(
                f"biopb-mcp viewer serving on http://127.0.0.1:{port}/mcp "
                "(Ctrl-C to stop; an agent may attach at this URL).",
                flush=True,
            )

    def _shutdown(reason):
        """One teardown for every deliberate-exit path — POSIX signals, the
        server loop returning: reap the kernel, close the session-child-owned
        dask cluster, remove our scratch, exit.

        Skips Python finalization: this process still has a live asyncio/epoll
        event-loop thread and the numpy OpenBLAS worker pool running, and
        tearing down the interpreter on top of them segfaults inside
        Py_FinalizeEx (refcount write into a read-only static-type page).
        The launcher's only remaining job is to exit, so exit immediately.
        """
        logger.info("Shutting down (%s).", reason)
        host.shutdown()
        # After the kernel is reaped (no clients left attached): stop the
        # session-child-owned cluster, then rmtree its now-idle spill dir. This
        # is the only path that closes the cluster — kernel restart/reap leaves
        # it warm.
        cluster_host.close()
        _cleanup_dask_dir()
        os._exit(0)

    def _handle_signal(signum, frame):
        _shutdown(f"signal {signum}")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Opt-in web "observe" UI. Set up before the (blocking) transport run:
    # custom routes are read when the streamable-http app is built.
    _setup_observe(config)

    if view:
        # Agentless viewer: bring the window up now (the human wants it
        # immediately) rather than waiting for a start_kernel tool call. Same
        # synchronous bring-up the start_kernel tool drives.
        logger.info("Opening the napari viewer (Ctrl-C to stop)...")
        try:
            host.ensure_started()
        except Exception:
            logger.exception("Failed to open the viewer; exiting")
            return 1  # atexit reaps the kernel/cluster and cleans the spill dir

    _server.run(
        port,
        allowed_origins=get_setting(config, "transport.allowed_origins"),
        allowed_hosts=get_setting(config, "transport.allowed_hosts"),
        sock=listen_sock,
    )

    # If the server loop returns on its own, exit the same way (atexit
    # handlers do not run after os._exit, so tear down explicitly here).
    _shutdown("server loop exited")


if __name__ == "__main__":
    sys.exit(main())
