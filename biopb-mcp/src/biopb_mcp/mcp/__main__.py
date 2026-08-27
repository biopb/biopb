"""Launcher for the biopb-mcp MCP server.

Under the http transport this process *is* the MCP server: it owns a child
Jupyter kernel that hosts the napari viewer — on the user's display when one
is present, else on a launcher-owned Xvfb virtual display (see ``_xvfb``; the
viewer and screenshots always exist).  Under the (deprecated) stdio transport it is
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

from biopb._locations import MCP_SESSION_LOG_ENV

logger = logging.getLogger(__name__)


# Env var carrying the path of the file a shim-owned child publishes its
# OS-assigned port to (the shim-owned session model). Presence of this var is also
# how _serve_http tells a shim-owned child (dynamic port, reported back) from a
# direct `--transport http` launch (fixed port). Kept in sync with _shim.
ENV_PORT_REPORT_FILE = "BIOPB_PORT_REPORT_FILE"

# Env var naming this process's own logfile, so it can report it (server_status)
# and the agent's execute_code can read it from os.environ. Set by whoever
# redirected our output: the stdio shim for the child it spawns, the control for
# a viewer it launches. Bound from the core SDK rather than repeated, since it is
# now three processes across two packages that must agree on one string.
ENV_SESSION_LOG = MCP_SESSION_LOG_ENV


def _report_port(path, port):
    """Publish the OS-assigned ``port`` to the shim's report file.

    The stdio shim (ARCHITECTURE.md, Lifecycle) spawns this
    child with ``--port 0`` and a unique ``BIOPB_PORT_REPORT_FILE``, then polls
    that file for the real port to build its bridge URL. Written atomically
    (temp + ``os.replace``) so the shim never reads a half-written value.

    A cross-platform file rather than the inherited-pipe handshake ``_kernel``
    uses for its death/window signals: that pipe pattern is POSIX-only there (fd
    inheritance across a Windows spawn is fragile), whereas a file is uniform.
    Best-effort: a write
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


def _register_view_session(port):
    """Publish this agentless viewer in the session registry; return its id.

    The control lists live sessions and proxies ``/session/<id>/*`` from this
    registry, so a session nobody publishes is a session with no observe page
    and no dashboard entry. The stdio shim publishes the child it owns
    (``_shim.spawn_session``); a `biopb mcp view` session has no shim, so it
    publishes itself.

    Best-effort, and broadly caught for the same reason the shim's publish is
    (biopb/biopb#422): a registry write failure — a serialization error, an
    unwritable state dir — must cost the viewer its discoverability and nothing
    else. Returns ``None`` in that case, leaving the caller nothing to
    de-register.
    """
    from biopb import _sessions

    try:
        session_id = _sessions.new_session_id()
        _sessions.register(
            session_id,
            port=port,
            pid=os.getpid(),
            mcp_url=f"http://127.0.0.1:{port}/mcp",
        )
    except Exception:
        logger.warning("Could not register this session", exc_info=True)
        return None
    logger.info("Registered session %s for the control plane.", session_id)
    return session_id


def _unregister_session(session_id):
    """Drop ``session_id``'s routing record. No-op when it was never registered.

    Best-effort like the publish: teardown must not fail because a record was
    already gone. The registry's own pid-liveness prune
    (:func:`biopb._sessions.list_sessions`) is the backstop for a kill abrupt
    enough that this never runs.
    """
    if session_id is None:
        return
    from biopb import _sessions

    try:
        _sessions.unregister(session_id)
    except Exception:
        logger.warning("Could not unregister session %s", session_id, exc_info=True)


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


def _setup_observe(config, agentless=False, on_shutdown=None):
    """Wire up the web observe UI.

    On by default (``observe.enabled``, opt-out); it mounts on the existing
    MCP app and shares its loop/port. Fully guarded — an observe failure logs
    and is swallowed so it can never block the MCP server. Returns True if
    mounted.

    *agentless* / *on_shutdown* decide whether this session serves the stop
    route, and what it runs. Passed in rather than read here because the
    launcher's ``_shutdown`` is the thing being handed over, and it must be
    registered before the routes are, which is inside this call.
    """
    from .._config import get_setting

    if not get_setting(config, "observe.enabled"):
        return False
    try:
        from . import _observe

        _observe.set_session_owns_its_reap(agentless, on_shutdown=on_shutdown)
        _observe.configure(
            max_output_chars=get_setting(config, "observe.max_output_chars"),
            poll_interval_ms=get_setting(config, "observe.poll_interval_ms"),
            console_enabled=get_setting(config, "observe.console_enabled"),
            allowed_origins=get_setting(config, "transport.allowed_origins"),
            allowed_hosts=get_setting(config, "transport.allowed_hosts"),
        )
        _observe.register_http_routes()
        return True
    except Exception:
        logger.exception("observe UI failed to start; continuing without it")
        return False


def _is_agentless_viewer(view, shim_owned):
    """Whether this session is a viewer a human opened, not a harness's child.

    Two things follow from it and must not drift apart: such a session publishes
    *itself* to the registry (nothing else owns its reap), and it is the only
    kind that gets the built-in chat loop. A shim-owned child is serving an MCP
    client; a direct ``--transport http`` launch is neither, and publishes no
    session at all, so it has no observe page for a pane to live on.
    """
    return bool(view and not shim_owned)


def _setup_chat(config, agentless):
    """Wire up the built-in chat client.

    Off by default (``observe.chat_enabled``, beside the console's switch): it
    spends the user's own provider credits, so an install must not turn it on
    for them. Guarded like observe — a chat failure logs and is swallowed rather
    than blocking the MCP server, which is the surface an already-working
    harness depends on. Returns True if mounted.

    *agentless* says whether this session is a `biopb mcp view` viewer rather
    than a child some MCP client is driving; chat is served only on the former.

    The verdict is also published on ``/api/status``
    (:func:`_observe.set_chat_enabled`), so the control's dashboard can label
    this session by what it actually serves — a viewer leads with chat, an MCP
    client's child with the job list — instead of guessing from launch flags it
    never sees. Set on every path, so a failed mount reads as off rather than
    stale.
    """
    from . import _observe

    mounted = False
    try:
        from . import _chat_api

        if _chat_api.configure(config, agentless=agentless):
            _chat_api.register_http_routes()
            mounted = True
    except Exception:
        logger.exception("chat API failed to start; continuing without it")
    _observe.set_chat_enabled(mounted)
    return mounted


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
    from . import _server, _xvfb
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

    # Decide where the kernel's viewer renders. With no display a Qt viewer
    # hard-aborts the kernel (SIGABRT, not a catchable error): a display-less
    # Linux host gets a launcher-owned Xvfb virtual display — a real viewer,
    # working screenshots, no human-visible window (#90) — or, when the binary
    # is missing, a fail-fast with the install hint. There is no compute-only
    # fallback. `--view` opens a window *for a human*, so a virtual display is
    # useless there and it fails fast instead.
    xvfb_proc = None
    virtual_display = None
    if not _has_display():
        if view:
            logger.error(
                "No display detected ($DISPLAY/$WAYLAND_DISPLAY are unset); "
                "`biopb mcp view` opens a viewer window for a human, so it "
                "needs a real X/Wayland session."
            )
            return 2
        try:
            xvfb_proc, virtual_display = _xvfb.start()
        except RuntimeError as exc:
            logger.error("Cannot start the napari viewer: %s", exc)
            return 2
        # Backstop for exits that skip _shutdown; _xvfb.stop is idempotent.
        atexit.register(_xvfb.stop, xvfb_proc)
        logger.info(
            "No display detected; the napari viewer will render on virtual "
            "display %s (screenshots work; no visible window).",
            virtual_display,
        )

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

    # Point the kernel's Qt at the launcher-owned Xvfb. BIOPB_VIRTUAL_DISPLAY
    # lets the in-kernel server_status report that the window, while real, is
    # not visible to the user.
    if virtual_display:
        kernel_env["DISPLAY"] = virtual_display
        kernel_env["BIOPB_VIRTUAL_DISPLAY"] = "1"

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
        # Session-child-owned dask cluster; _launch calls ensure() and injects
        # its scheduler address so the kernel attaches instead of spinning its own.
        cluster_host=cluster_host,
    )
    # Now that the kernel host exists, let the cluster's idle reaper ask it
    # whether a kernel is attached — the one thing that makes a teardown safe.
    cluster_host.set_kernel_alive(host.is_alive)
    cluster_host.start_reaper()
    _server.set_kernel_host(host)
    _server.set_promote_after(get_setting(config, "kernel.promote_after"))
    # Advertise the curated-skills catalog only when it is enabled (off by
    # default) — mirrors what find_skills / the skill:// resource actually serve.
    _server.set_skills_enabled(get_setting(config, "services.skills_enabled"))

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
    logger.info(
        "Ready. The napari kernel (and viewer window) starts on the first "
        "start_kernel call."
    )

    # Reap the kernel on exit even if it is still mid-bringup when we stop
    # (a no-op safe on an idle, never-started host).
    atexit.register(host.shutdown)

    # Two foreground modes bind a *dynamic* port and report it back rather than
    # binding the configured fixed port:
    #   * the de-daemonized shim-owned child — the shim set
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

    # Set by the registration below; read by _shutdown. Declared here because the
    # signal handlers are installed before that point, so this name has to exist
    # even on a Ctrl-C that arrives during the viewer's bring-up.
    session_id = None

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
        # Drop the routing record before anything else, so a control stops
        # routing here while this process can still refuse a connection cleanly
        # rather than after it has stopped answering. Teardown and
        # de-registration are one path, as they are in the shim's _reap_session;
        # the registry's own pid-liveness prune is the backstop for a kill this
        # never runs for.
        _unregister_session(session_id)
        host.shutdown()
        # After the kernel is reaped (no clients left attached): stop the
        # session-child-owned cluster, then rmtree its now-idle spill dir. This
        # is the only path that closes the cluster — kernel restart/reap leaves
        # it warm. The Xvfb display outlives kernel restarts the same way, so
        # it too goes down only here (its X clients died with the kernel).
        cluster_host.close()
        _cleanup_dask_dir()
        _xvfb.stop(xvfb_proc)
        os._exit(0)

    def _handle_signal(signum, frame):
        _shutdown(f"signal {signum}")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Whether this session owns its own reap. Two things hang off it, and the
    # single expression keeps them from drifting: the built-in chat loop, and
    # the stop route (a shim-owned child is reaped by its shim, so ending it
    # here would leave that shim bridging to a dead process).
    agentless = _is_agentless_viewer(view, shim_owned)

    # Opt-in web "observe" UI. Set up before the (blocking) transport run:
    # custom routes are read when the streamable-http app is built. `_shutdown`
    # goes with it, so a stop from the web takes the same path Ctrl-C does.
    _setup_observe(
        config,
        agentless=agentless,
        on_shutdown=lambda: _shutdown("stopped from the web"),
    )
    # A shim-owned child is serving an MCP client, which is the one situation the
    # built-in loop is not for.
    _setup_chat(config, agentless=agentless)

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

    # Only the agentless viewer registers *itself*: a shim-owned child is
    # published by the shim that owns its reap (and so its de-registration), and
    # a direct `--transport http` launch binds the configured fixed port its
    # operator already knows. Done last, with the kernel up and the serve loop
    # the next statement, so a record implies a session that is all but
    # answering.
    if _is_agentless_viewer(view, shim_owned):
        session_id = _register_view_session(port)

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
