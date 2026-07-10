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
  assigned to, so a force-killed shim reaps the whole tree (see ``_winjob``).

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
import time

import anyio
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel.server import Server, request_ctx
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

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


def _open_daemon_log(config):
    """Open the file the session child's stdout/stderr is sent to.

    The canonical daemon log (``mcp.transport.kernel_log``, empty ->
    <log dir>/mcp-server.log) shared with the ``biopb mcp`` CLI so `mcp logs` /
    `status` read whatever this writes: the child's fds are inherited by its
    kernel, so this file carries the same native Qt/GL/dask/gRPC output the key
    always named — plus the child's own logs. Binary, append, unbuffered, for
    the same reason the kernel redirection always was: native writers emit
    arbitrary bytes. On failure, falls back to the shim's stderr buffer so the
    child still starts (its output then interleaves with the shim's logging —
    harmless, stderr is not a protocol channel).
    """
    from .._config import get_daemon_log_file

    path = str(get_daemon_log_file(config))
    try:
        return open(path, "ab", buffering=0)
    except OSError:
        logger.warning(
            "Could not open daemon log %s; routing child output to stderr",
            path,
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
    """Spawn a private http session child this shim owns; return (proc, url, job).

    See the module docstring for the ownership model. Inherits this shim's live
    environment (the #98 fix), binds a dynamic port the child reports back, and
    ties the child's lifetime to this shim (POSIX process group; Windows Job
    Object). On any startup failure the child is reaped before the error
    propagates, so a failed bring-up never leaks a process.

    Raises TimeoutError / RuntimeError if the child never becomes reachable.
    """
    log = _open_daemon_log(config)
    cmd = _session_command()
    fd, port_file = tempfile.mkstemp(prefix="biopb-mcp-port-", suffix=".txt")
    os.close(fd)  # the child writes it by path, not fd

    # Inherit THIS shim's live environment (the #98 fix). Explicit copy so the
    # intent is legible; we only add the port-report channel.
    env = os.environ.copy()
    env[ENV_PORT_REPORT_FILE] = port_file

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

    return proc, f"http://127.0.0.1:{port}/mcp", job


def _reap_session(proc, job):
    """Tear down the owned child (and its kernel grandchild).

    The bridge-close / signal counterpart to the OS-level ties set at spawn.
    POSIX: SIGTERM the child so its own handler reaps the kernel *gracefully*
    (dask/spill cleanup), then escalate to SIGKILL; the kernel is in its own
    session watching the child's parent-death pipe, so even an abrupt child
    death still reaps it. Windows: TerminateJobObject force-reaps the whole tree
    (no catchable signal there), then the handle is released. Idempotent and
    best-effort — safe to call more than once and on an already-dead child.
    """
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


def _install_shim_reaper(proc, job):
    """POSIX: reap the child if this shim is signalled to exit.

    A SIGTERM/SIGHUP delivered to the shim *alone* (not its whole group) would
    otherwise orphan the child, since Python's default handler exits without
    running ``serve``'s ``finally``. SIGINT is left to its default: it raises
    KeyboardInterrupt out of ``run_bridge`` and the ``finally`` reaps. On Windows
    there are no such signals — the Job Object reaps on any shim death — so this
    is a no-op there.
    """
    if os.name == "nt":
        return

    def _on_signal(signum, frame):
        _reap_session(proc, job)
        os._exit(0)

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


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
    proc, url, job = spawn_session(config)
    _install_shim_reaper(proc, job)
    logger.info("Bridging stdio to %s (owned session pid %s)", url, proc.pid)
    try:
        run_bridge(url)
    finally:
        _reap_session(proc, job)
