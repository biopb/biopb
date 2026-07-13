"""stdio bridge ("shim") to a private, shim-owned biopb-mcp http session.

``biopb-mcp --transport stdio`` no longer serves MCP over fd 0/1 from the
heavy launcher process. Instead the launcher runs this module, which

1. spawns its **own** http session child — FastMCP/uvicorn + the kernel host —
   on an OS-assigned port, inheriting this shim's live environment
   (``spawn_session``), then
2. bridges stdio JSON-RPC <-> that child's streamable-http endpoint
   (``run_bridge``) until the client closes stdin, then
3. reaps the child (and its kernel grandchild) on the way out (``_reap_session``).

This is de-daemonization Layer 1 (docs/mcp-dedaemonization-migration.md). It
retains the shim/heavy *split* of docs/daemon-migration.md — the process that
owns fd 1 as a protocol channel imports nothing that could write to stdout (no
Qt, dask, uvicorn, or kernel — only the mcp SDK), so the fd-1 corruption class
is structurally impossible here — but undoes the daemon's *detachment* and its
role as a shared, client-outliving lifecycle root:

* **Ephemeral & owned.** Every shim spawns its own child and tears it down when
  its client disconnects. No probe-and-reuse, no shared daemon, no fixed port —
  two clients get two independent sessions (two viewers), by design.
* **Env-inherited (the #98 fix).** The child inherits *this* shim's environment,
  so ``DISPLAY`` / ``XAUTHORITY`` / ``WAYLAND_DISPLAY`` are always the user's
  current session — never a value frozen into a long-lived daemon by whoever
  happened to spawn it first.
* **Reaped, cross-platform (the #403 fix, generalized).** POSIX: the child stays
  in this shim's process group (no ``start_new_session``), so the MCP client's
  process-group teardown takes it — and its kernel, via the kernel's own
  parent-death pipe — down with the shim; the bridge-close path also reaps it
  explicitly. Windows: this shim holds a kill-on-close Job Object the child is
  assigned to, so a force-killed shim reaps the whole tree (see ``_winjob``);
  and a client-death watchdog (``_install_client_death_watchdog``) reaps the
  shim itself when its stdio *client* exits without the bridge seeing stdin EOF
  -- e.g. when a multi-process client (Claude Code: a daemon + pty host) keeps a
  duplicate of the shim's stdin write handle open in a surviving helper after
  the launching process is gone, so the session would otherwise outlive every
  real client (biopb#403, the client side).

The bridge itself is vendored rather than delegated to ``mcp-proxy`` per the
vetting report (docs/mcp-proxy-vet.md): mcp-proxy drops the initialize
``instructions`` field that carries biopb-mcp's operation guardrails, has no
lifetime guard when the server dies, and floats its dependencies. Here the
remote's initialize result — capabilities, serverInfo, and ``instructions`` —
is replayed to the stdio client verbatim, and any bridge failure exits the
shim so the client sees EOF instead of a hung proxy.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import anyio
from biopb import _config_sessions
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel.server import Server, request_ctx
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .. import _control_client
from . import _winjob

logger = logging.getLogger(__name__)

# How long spawn_session waits for a spawned child to report its port and start
# listening. Dominated by the http stack's import time (FastMCP + uvicorn + the
# tensor client), not the kernel — the kernel starts later, on the first
# `start_kernel` tool call.
SESSION_START_TIMEOUT = 60.0
_PROBE_INTERVAL = 0.25
# How long a reap waits for the child to exit at each escalation step.
REAP_TIMEOUT = 10.0

# Env var carrying the path of the file the child publishes its OS-assigned port
# to; the child also keys "am I shim-owned?" off its presence. Kept in sync with
# __main__.ENV_PORT_REPORT_FILE (a literal here, like the sentinel paths, to keep
# this featherweight module from importing the heavy launcher).
ENV_PORT_REPORT_FILE = "BIOPB_PORT_REPORT_FILE"

# Env var telling the child the path of its own session logfile, so it can report
# it (server_status) and the agent's execute_code can read it from os.environ.
# The child inherits it, and so does the kernel it spawns. Kept in sync with
# __main__.ENV_SESSION_LOG.
ENV_SESSION_LOG = "BIOPB_MCP_SESSION_LOG"


def _port_listening(port, timeout=0.5):
    """Whether something accepts TCP connections on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _session_command():
    """The argv that launches the http session child this shim owns.

    Binds a dynamic port (``--port 0``); the child reports the OS-assigned port
    back through the file named in ``BIOPB_PORT_REPORT_FILE``. A frozen
    (PyInstaller) build has no importable module tree behind ``sys.executable``,
    but its entry binary *is* the launcher, so plain args suffice; a normal
    install re-enters via ``-m``.
    """
    if getattr(sys, "frozen", False):
        cmd = [sys.executable]
    else:
        cmd = [sys.executable, "-m", "biopb_mcp.mcp"]
    return [*cmd, "--transport", "http", "--port", "0"]


def _new_session_id():
    """A sortable, unique id for this shim session: ``<timestamp>-<shim-pid>``."""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def _session_log_path(config, session_id):
    """Where this session's child logs. Per-session by default; a single shared
    file when ``mcp.transport.kernel_log`` is set (opt back into the old behavior).
    """
    from .._config import get_session_log_dir, get_setting

    override = get_setting(config, "mcp.transport.kernel_log")
    if override:
        return str(override)
    return str(get_session_log_dir() / f"{session_id}.log")


def _prune_session_logs(keep):
    """Keep only the newest ``keep`` per-session logs; best-effort.

    Run after the current session's log is created (it is newest, so it always
    survives). A prune failure never affects the session.
    """
    from .._config import get_session_log_dir

    try:
        logs = sorted(
            get_session_log_dir().glob("*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old in logs[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def _open_session_log(path):
    """Open ``path`` for the session child's stdout/stderr (binary, append,
    unbuffered — native Qt/GL/dask/gRPC writers emit arbitrary bytes, and the
    child's fds are inherited by its kernel). On failure, falls back to the shim's
    stderr buffer so the child still starts (its output then interleaves with the
    shim's logging — harmless, stderr is not a protocol channel).
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return open(path, "ab", buffering=0)
    except OSError:
        logger.warning(
            "Could not open session log %s; routing child output to stderr", path
        )
        return getattr(sys.stderr, "buffer", sys.stderr)


def _read_port_file(path):
    """The port the child published, or None if not yet written / unparseable.

    The child writes atomically (temp + ``os.replace``), so a read never sees a
    partial value; an empty/missing file just means "not reported yet".
    """
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        port = int(text)
    except ValueError:
        return None
    return port if port > 0 else None


def _await_port(proc, port_file, timeout):
    """Block until the child publishes its port, returning it.

    Raises RuntimeError if the child exits first (its log has the trace) or
    TimeoutError if nothing is reported within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        port = _read_port_file(port_file)
        if port is not None:
            return port
        if proc.poll() is not None:
            raise RuntimeError(
                f"biopb-mcp session child exited (status {proc.returncode}) "
                "before reporting its port; check the daemon log"
            )
        time.sleep(_PROBE_INTERVAL)
    raise TimeoutError(
        f"biopb-mcp session child did not report its port within {timeout:.0f}s; "
        "check the daemon log"
    )


def _await_listening(proc, port, timeout):
    """Block until the child accepts connections on ``port``.

    The child publishes its port right after binding but before it listens, so
    the bridge must wait for the listener to come up. Raises like ``_await_port``
    on child death / timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_listening(port):
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"biopb-mcp session child exited (status {proc.returncode}) "
                f"before listening on 127.0.0.1:{port}; check the daemon log"
            )
        time.sleep(_PROBE_INTERVAL)
    raise TimeoutError(
        f"biopb-mcp session child did not start listening on 127.0.0.1:{port} "
        f"within {timeout:.0f}s; check the daemon log"
    )


def spawn_session(config, timeout=SESSION_START_TIMEOUT):
    """Spawn a private http child this shim owns; return (proc, url, job, session_id).

    See the module docstring for the ownership model. Inherits this shim's live
    environment (the #98 fix), binds a dynamic port the child reports back, and
    ties the child's lifetime to this shim (POSIX process group; Windows Job
    Object). On any startup failure the child is reaped before the error
    propagates, so a failed bring-up never leaks a process.

    Raises TimeoutError / RuntimeError if the child never becomes reachable.
    """
    from .._config import get_setting

    # Per-session logfile (not the shared mcp-server.log): concurrent sessions no
    # longer interleave. Prune older ones to the configured cap after opening the
    # new one (it is newest, so it survives).
    session_id = _new_session_id()
    log_path = _session_log_path(config, session_id)
    log = _open_session_log(log_path)
    logged_to_file = log is not getattr(sys.stderr, "buffer", sys.stderr)
    if logged_to_file and not get_setting(config, "mcp.transport.kernel_log"):
        _prune_session_logs(get_setting(config, "mcp.transport.session_log_keep", 5))

    cmd = _session_command()
    fd, port_file = tempfile.mkstemp(prefix="biopb-mcp-port-", suffix=".txt")
    os.close(fd)  # the child writes it by path, not fd

    # Inherit THIS shim's live environment (the #98 fix). Explicit copy so the
    # intent is legible; we add the port-report channel and — so the child can
    # report its own logfile (server_status) and the agent's execute_code can
    # read it from os.environ — the session log path.
    env = os.environ.copy()
    env[ENV_PORT_REPORT_FILE] = port_file
    if logged_to_file:
        env[ENV_SESSION_LOG] = log_path

    popen_kwargs = {}
    if os.name == "nt":
        # CREATE_NO_WINDOW, NOT DETACHED_PROCESS / CREATE_NEW_PROCESS_GROUP:
        # keep this shim from pinning a console while giving the child a hidden
        # console the kernel inherits silently (DETACHED_PROCESS would force a
        # fresh *visible* console when the child later spawns the console-
        # subsystem Jupyter kernel — an empty window popping on the desktop). The
        # child is reaped via the Job Object below, not group signals, so it
        # needs no new process group.
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    # POSIX: intentionally NO start_new_session — the child shares this shim's
    # process group, so the MCP client's process-group teardown takes it (and
    # its kernel, via the kernel's parent-death pipe) down with the shim.

    logger.info("Spawning owned biopb-mcp session: %s", cmd)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            close_fds=True,
            env=env,
            **popen_kwargs,
        )
    finally:
        if log is not getattr(sys.stderr, "buffer", sys.stderr):
            log.close()  # the child holds its own duplicate of the fd

    # Windows: hold a kill-on-close Job Object containing the child, so a
    # force-killed shim reaps the child (and, via the child's own nested kernel
    # job, the kernel) — the #403 fix, now shim-owned. POSIX ties via the
    # process group above.
    job = None
    if os.name == "nt":
        job = _winjob.create_kill_on_close_job()
        _winjob.assign_process(job, proc.pid)

    try:
        port = _await_port(proc, port_file, timeout)
        _await_listening(proc, port, timeout)
    except BaseException:
        _reap_session(proc, job)
        raise
    finally:
        try:
            os.unlink(port_file)
        except OSError:
            pass

    url = f"http://127.0.0.1:{port}/mcp"
    # Publish the now-reachable session so the control can list it and proxy
    # /session/<id>/* to it. Registered only after the child is fully up (a
    # failed bring-up above reaps and never reaches here, so it leaves no
    # record); the reap path unregisters. Best-effort — a registry write failure
    # must not fail an otherwise-working session, only cost its discoverability.
    # Catch broadly (not just OSError): a serialization TypeError/ValueError, or
    # register()'s own unsafe-id ValueError, must not escape and break the session
    # either (biopb/biopb#422).
    try:
        _config_sessions.register(session_id, port=port, pid=proc.pid, mcp_url=url)
    except Exception as e:
        logger.warning("Could not register session %s: %s", session_id, e)

    return proc, url, job, session_id


def _reap_session(proc, job, session_id=None):
    """Tear down the owned child (and its kernel grandchild).

    The bridge-close / signal counterpart to the OS-level ties set at spawn.
    POSIX: SIGTERM the child so its own handler reaps the kernel *gracefully*
    (dask/spill cleanup), then escalate to SIGKILL; the kernel is in its own
    session watching the child's parent-death pipe, so even an abrupt child
    death still reaps it. Windows: TerminateJobObject force-reaps the whole tree
    (no catchable signal there), then the handle is released. Idempotent and
    best-effort — safe to call more than once and on an already-dead child.

    ``session_id`` (when this reaps a registered session) is dropped from the
    filesystem registry here so teardown and de-registration are one path — a
    force-killed child leaves no routing ghost; the registry's own pid-liveness
    prune (:func:`biopb._config_sessions.list_sessions`) is the backstop for the
    case where even this reap is skipped.
    """
    if session_id is not None:
        _config_sessions.unregister(session_id)

    if proc.poll() is not None:
        if job is not None:
            _winjob.close_job(job)  # release the handle for an already-gone child
        return

    if os.name == "nt":
        # terminate_job force-reaps the whole tree when the child was assigned;
        # proc.kill() (TerminateProcess) is the backstop for when the Job Object
        # is unavailable (winjob is best-effort — a ctypes/OS hiccup returns
        # None) or the assign failed, so the child is still reaped directly, and
        # its own kill-on-close kernel job then reaps the kernel.
        _winjob.terminate_job(job)
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
        _winjob.close_job(job)
        return

    try:
        proc.terminate()  # SIGTERM -> the child's handler reaps its kernel
    except OSError:
        pass
    try:
        proc.wait(timeout=REAP_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()  # SIGKILL -> kernel self-reaps off its parent-death pipe
    except OSError:
        pass
    try:
        proc.wait(timeout=REAP_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass


def _install_shim_reaper(proc, job, session_id=None):
    """POSIX: reap the child if this shim is signalled to exit.

    A SIGTERM/SIGHUP delivered to the shim *alone* (not its whole group) would
    otherwise orphan the child, since Python's default handler exits without
    running ``serve``'s ``finally``. SIGINT is left to its default: it raises
    KeyboardInterrupt out of ``run_bridge`` and the ``finally`` reaps. On Windows
    there are no such signals — the Job Object reaps on any shim *death*, and
    ``_install_client_death_watchdog`` covers the shim being *orphaned* by its
    client — so this is a no-op there.

    ``session_id`` is passed through so the signal path de-registers too (it
    ``os._exit``s past ``serve``'s ``finally``, so it must clean the registry
    itself).
    """
    if os.name == "nt":
        return

    def _on_signal(signum, frame):
        _reap_session(proc, job, session_id)
        os._exit(0)

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


# Substrings that mark a process in *our own* stdio launcher chain (the shim and
# any interpreter-launcher stubs above it), as opposed to the MCP client that
# spawned it. The whole chain shares the argv the client invoked us with
# (``... biopb[-_]mcp ... --transport stdio``), and re-exec launchers (e.g. a
# venv built on Microsoft Store Python inserts one) preserve it. The client's own
# cmdline (claude.exe, an editor, a shell) matches neither.
_OURS_MARKERS = ("biopb", "stdio")


def _find_client_process():
    """Walk up past our own launcher chain to the MCP client that owns us.

    ``os.getppid()`` is NOT the client when an interpreter-launcher stub sits
    between us and it — the case that made a naive parent-watch useless: on a
    venv built on Store Python the stub *outlives* the client (it only waits on
    us), so watching it never fires. We instead climb ancestors while their
    cmdline looks like our chain (:data:`_OURS_MARKERS`) and return the first
    foreign one — the real client. ``None`` if it can't be determined (psutil
    missing, an unreadable/inaccessible ancestor, or the chain reaches the top),
    in which case the watchdog simply does not arm.
    """
    try:
        import psutil
    except Exception:
        return None
    try:
        node = psutil.Process().parent()
        while node is not None:
            cl = " ".join(node.cmdline()).lower()
            if not all(m in cl for m in _OURS_MARKERS):
                return node  # first ancestor not in our chain == the client
            node = node.parent()
    except Exception:
        # AccessDenied / NoSuchProcess / a gone ancestor mid-walk: give up rather
        # than risk watching the wrong process. stdin EOF stays the backstop.
        return None
    return None


def _is_windows() -> bool:
    """Platform check, isolated as a seam the watchdog tests patch.

    Tests must exercise the Windows-only branch below *without* forcing the global
    ``os.name = "nt"`` — on a POSIX runner that makes ``pathlib.Path`` build a
    ``WindowsPath``, which raises on Python < 3.12 and crashes pytest itself (its
    coverage / cache / location machinery calls ``Path``). See the same note in
    ``_tests/test_update.py``.
    """
    return os.name == "nt"


def _install_client_death_watchdog(proc, job, session_id=None):
    """Windows: reap the owned child if the stdio *client* dies unseen by stdin.

    Normal teardown runs when ``run_bridge`` returns on stdin EOF (client hung
    up) or, on POSIX, when the client's process-group teardown / a SIGTERM fires
    ``_install_shim_reaper``. Windows has neither group teardown nor those
    signals, and a multi-process client can keep a duplicate of the shim's stdin
    write handle open in a surviving helper after the launching process exits, so
    EOF never arrives — the shim blocks forever in the bridge and the child +
    kernel leak (they outlive every real client). This watchdog closes that gap:
    it blocks on the client's process handle and, when the client exits for any
    reason, reaps the owned tree (de-registering ``session_id``) and exits.

    The client is found by :func:`_find_client_process` (walking past our own
    launcher stubs — see its docstring for why ``os.getppid()`` is not enough).
    We hold a handle to that exact process object, so the wait is immune to pid
    reuse. If the client cannot be found or opened at arm time, we do not arm: a
    live session is never reaped off an uncertain baseline; stdin EOF and the Job
    Object remain the backstops. A no-op off Windows (POSIX is covered by the
    process group + signal reaper).

    Returns the watchdog thread (daemon), or ``None`` if not armed.
    """
    if not _is_windows():
        return None
    client = _find_client_process()
    if client is None:
        logger.debug("client-death watchdog not armed (client not identifiable)")
        return None
    handle = _winjob.open_for_wait(client.pid)
    if not handle:
        logger.debug(
            "client-death watchdog not armed (client pid %s un-openable)", client.pid
        )
        return None
    thread = threading.Thread(
        target=_client_deathwatch,
        args=(handle, client.pid, proc, job, session_id),
        name="biopb-client-deathwatch",
        daemon=True,
    )
    thread.start()
    logger.info("client-death watchdog armed on client pid %s", client.pid)
    return thread


def _client_deathwatch(handle, client_pid, proc, job, session_id):
    """Block on the client's ``handle``; on its exit, reap the tree and exit.

    A wait error is treated as *undecided* — we do not reap, since a spurious
    reap would tear down a live session. On a real exit this ``os._exit``s past
    ``serve``'s ``finally`` (having already reaped), matching ``_on_signal``.
    """
    if not _winjob.wait_for_process(handle):
        return  # undecided (wait errored) — leave teardown to the other paths
    logger.info(
        "stdio client (pid %s) exited; reaping owned session %s", client_pid, session_id
    )
    _reap_session(proc, job, session_id)
    os._exit(0)


def build_proxy(remote):
    """Build the stdio-facing MCP server that forwards every request to
    ``remote`` (a ClientSession connected to the session child).

    Handlers are registered unconditionally and the *remote's* capabilities
    are advertised verbatim (see ``run_bridge``), so the bridge never narrows
    what the child offers; a client simply won't call what the capabilities
    don't advertise. Server->client traffic other than tool-call progress
    (sampling, elicitation, list_changed) is not forwarded — biopb-mcp emits
    none of it (the on-demand kernel design avoids dynamic tool lists on
    purpose), and a future feature that needs it must extend this bridge.
    """
    app = Server(name="biopb-mcp")

    async def _list_tools(_):
        return types.ServerResult(await remote.list_tools())

    app.request_handlers[types.ListToolsRequest] = _list_tools

    async def _call_tool(req):
        meta = dict(req.params.meta) if req.params.meta else None
        progress_token = meta.get("progressToken") if meta else None
        progress_callback = None
        if progress_token is not None:
            ctx = request_ctx.get()

            async def _forward_progress(progress, total, message):
                await ctx.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=progress,
                    total=total,
                    message=message,
                    related_request_id=str(ctx.request_id),
                )

            progress_callback = _forward_progress
        try:
            result = await remote.call_tool(
                req.params.name,
                req.params.arguments or {},
                progress_callback=progress_callback,
                meta=meta,
            )
            return types.ServerResult(result)
        except Exception as e:  # surface as a tool error, not a dead bridge
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(e))],
                    isError=True,
                )
            )

    app.request_handlers[types.CallToolRequest] = _call_tool

    async def _list_resources(_):
        return types.ServerResult(await remote.list_resources())

    app.request_handlers[types.ListResourcesRequest] = _list_resources

    async def _list_resource_templates(_):
        return types.ServerResult(await remote.list_resource_templates())

    app.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates

    async def _read_resource(req):
        return types.ServerResult(await remote.read_resource(req.params.uri))

    app.request_handlers[types.ReadResourceRequest] = _read_resource

    async def _list_prompts(_):
        return types.ServerResult(await remote.list_prompts())

    app.request_handlers[types.ListPromptsRequest] = _list_prompts

    async def _get_prompt(req):
        return types.ServerResult(
            await remote.get_prompt(req.params.name, req.params.arguments)
        )

    app.request_handlers[types.GetPromptRequest] = _get_prompt

    async def _complete(req):
        return types.ServerResult(
            await remote.complete(req.params.ref, req.params.argument.model_dump())
        )

    app.request_handlers[types.CompleteRequest] = _complete

    async def _set_logging_level(req):
        await remote.set_logging_level(req.params.level)
        return types.ServerResult(types.EmptyResult())

    app.request_handlers[types.SetLevelRequest] = _set_logging_level

    async def _forward_progress_notification(req):
        await remote.send_progress_notification(
            req.params.progressToken, req.params.progress, req.params.total
        )

    app.notification_handlers[types.ProgressNotification] = (
        _forward_progress_notification
    )

    return app


def replay_init_options(init):
    """Map the child's InitializeResult onto the options the bridge serves.

    Verbatim replay — most importantly ``instructions``, the handshake-time
    carrier for the operation guardrails and the headless notice (the field
    mcp-proxy drops, per docs/mcp-proxy-vet.md), and the remote's capability
    set rather than one recomputed from the bridge's own handlers.
    """
    return InitializationOptions(
        server_name=init.serverInfo.name,
        server_version=init.serverInfo.version,
        capabilities=init.capabilities,
        instructions=init.instructions,
        website_url=init.serverInfo.websiteUrl,
        icons=init.serverInfo.icons,
    )


async def _bridge(url):
    async with (
        streamablehttp_client(url=url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        app = build_proxy(session)
        options = replay_init_options(init)
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, options)


def run_bridge(url):
    """Bridge stdio <-> the session child at ``url`` until the client closes
    stdin.

    Any failure (child death mid-session included) propagates out, so the shim
    process exits and the client sees EOF — never a hung bridge.
    """
    anyio.run(_bridge, url)


def serve(config, port=None):
    """Launcher entry point for ``--transport stdio``: spawn, bridge, reap.

    ``port`` (the configured ``mcp.transport.port``) is vestigial here — the
    owned child binds a dynamic port — and is accepted only for call-site
    compatibility with the launcher's dispatch.
    """
    logger.warning(
        "stdio is served by bridging to a private biopb-mcp http session this "
        "shim spawns and owns (torn down when this client disconnects). Native "
        "http is recommended where the client supports it: "
        "`claude mcp add --transport http biopb http://127.0.0.1:<port>/mcp` "
        "against a `biopb mcp start` session."
    )
    # Best-effort, non-blocking: get the durable control plane (which owns the data
    # plane the kernel will talk to) coming up -- WITHOUT waiting for or verifying
    # it, since the bridge below must be ready within the MCP client's handshake
    # timeout. It boots in parallel with spawn_session's import-dominated child
    # startup; if it is still not up when the child first needs the data plane, the
    # child surfaces the error (see _control_client.start_control_detached). Fully
    # guarded -- a control-start hiccup must never abort the shim's bridge.
    try:
        _control_client.start_control_detached()
    except Exception:  # noqa: BLE001 - best-effort; the child surfaces real errors
        logger.info("control auto-start attempt failed", exc_info=True)
    proc, url, job, session_id = spawn_session(config)
    _install_shim_reaper(proc, job, session_id)
    _install_client_death_watchdog(proc, job, session_id)
    logger.info("Bridging stdio to %s (owned session pid %s)", url, proc.pid)
    try:
        run_bridge(url)
    finally:
        _reap_session(proc, job, session_id)
