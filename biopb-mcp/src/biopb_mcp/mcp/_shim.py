"""stdio bridge ("shim") to a private, shim-owned biopb-mcp http session.

``biopb-mcp --transport stdio`` no longer serves MCP over fd 0/1 from the
heavy launcher process. Instead the launcher runs this module, which

1. answers ``initialize`` and the list requests itself, from the shared
   ``_instructions`` and the shipped ``_surface`` snapshot, so a client that
   never calls a tool costs no child;
2. on the first request that needs one, spawns its **own** http session child —
   FastMCP/uvicorn + the kernel host — on an OS-assigned port, inheriting this
   shim's live environment (``spawn_session``, driven by ``_LazySession``), and
   bridges requests to that child's streamable-http endpoint until the client
   closes stdin; then
3. reaps the child (and its kernel grandchild) on the way out (``_reap_session``).

This is the de-daemonized, shim-owned session model (ARCHITECTURE.md, Lifecycle). It
retains the shim/heavy *split* described there — the process that
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

The bridge itself is vendored rather than delegated to ``mcp-proxy``: mcp-proxy
drops the initialize ``instructions`` field that carries biopb-mcp's operation
guardrails, has no lifetime guard when the server dies, and floats its
dependencies. Here the handshake carries the child's own capabilities and
``instructions``, a child that fails to start is a tool error the agent can read,
and a child that dies after starting exits the shim so the client sees EOF
instead of a hung proxy.
"""

import logging
import os
import signal
import socket
import sys
import tempfile
import threading
import time

import anyio
from biopb import _locations, _sessions
from biopb._lifecycle import winjob as _winjob
from biopb._lifecycle.owned_child import OwnedChild, open_child_log
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel.server import Server, request_ctx
from mcp.server.stdio import stdio_server

from .. import _control_client

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

# Env var carrying the session id this shim mints for its child, which registers
# itself under it. Kept in sync with __main__.ENV_SESSION_ID.
ENV_SESSION_ID = "BIOPB_MCP_SESSION_ID"

# Env var telling the child the path of its own session logfile, so it can report
# it (server_status) and the agent's execute_code can read it from os.environ.
# The child inherits it, and so does the kernel it spawns. Bound from the core
# SDK -- unlike the sentinel paths above, this one is not private to biopb-mcp:
# the control sets it too, for the viewers it launches. `biopb._locations` is
# stdlib-only, so this stays as featherweight as a literal.
ENV_SESSION_LOG = _locations.MCP_SESSION_LOG_ENV


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
    back through the file named in ``BIOPB_PORT_REPORT_FILE``.
    """
    return [sys.executable, "-m", "biopb_mcp.mcp", "--transport", "http", "--port", "0"]


def _session_log_path(config, session_id):
    """Where this session's child logs. Per-session by default; a single shared
    file when ``transport.kernel_log`` is set (opt back into the old behavior).
    """
    from .._config import get_session_log_dir, get_setting

    override = get_setting(config, "transport.kernel_log")
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


def spawn_session(config, timeout=SESSION_START_TIMEOUT, on_spawned=None):
    """Spawn a private http child this shim owns; return (child, url, session_id).

    ``child`` is an :class:`OwnedChild`. See the module docstring for the
    ownership model. Inherits this shim's live environment (the #98 fix), binds a
    dynamic port the child reports back, and ties the child's lifetime to this
    shim (POSIX process group; Windows Job Object). On any startup failure the
    child is reaped before the error propagates, so a failed bring-up never leaks
    a process. *on_spawned* gets the child as soon as it exists, so a reap that
    fires during the bring-up can find it.

    The child registers itself with the control under ``session_id``, minted
    here because it also names the child's logfile.

    Raises TimeoutError / RuntimeError if the child never becomes reachable.
    """
    from .._config import get_setting

    # Per-session logfile (not the shared mcp-server.log): concurrent sessions no
    # longer interleave. Prune older ones to the configured cap after opening the
    # new one (it is newest, so it survives).
    session_id = _sessions.new_session_id()
    log_path = _session_log_path(config, session_id)
    log, logged_to_file = open_child_log(log_path)
    if logged_to_file and not get_setting(config, "transport.kernel_log"):
        _prune_session_logs(get_setting(config, "transport.session_log_keep", 5))

    cmd = _session_command()
    fd, port_file = tempfile.mkstemp(prefix="biopb-mcp-port-", suffix=".txt")
    os.close(fd)  # the child writes it by path, not fd

    # Inherit THIS shim's live environment (the #98 fix). Explicit copy so the
    # intent is legible; we add the port-report channel and — so the child can
    # report its own logfile (server_status) and the agent's execute_code can
    # read it from os.environ — the session log path.
    env = os.environ.copy()
    env[ENV_PORT_REPORT_FILE] = port_file
    env[ENV_SESSION_ID] = session_id
    if logged_to_file:
        env[ENV_SESSION_LOG] = log_path

    # OwnedChild applies the owned-child spawn conventions and the Windows Job
    # Object bind (CREATE_NO_WINDOW, no new process group; POSIX shares this
    # shim's group so the client's teardown reaps it — and its kernel, via the
    # kernel's parent-death pipe). See biopb._lifecycle.owned_child.
    logger.info("Spawning owned biopb-mcp session: %s", cmd)
    child = OwnedChild(cmd, log=log, env=env)
    try:
        child.spawn()
    finally:
        if logged_to_file:
            log.close()  # the child holds its own duplicate of the fd
    if on_spawned is not None:
        on_spawned(child)

    try:
        port = _await_port(child.proc, port_file, timeout)
        _await_listening(child.proc, port, timeout)
    except BaseException:
        child.stop()
        raise
    finally:
        try:
            os.unlink(port_file)
        except OSError:
            pass

    return child, f"http://127.0.0.1:{port}/mcp", session_id


def _reap_session(child):
    """Tear down the owned child (and its kernel grandchild).

    The bridge-close / signal counterpart to the OS-level ties set at spawn.
    Delegates the platform reap escalation to :meth:`OwnedChild.stop` (POSIX:
    SIGTERM so the child's handler reaps the kernel gracefully, then SIGKILL;
    Windows: TerminateJobObject force-reaps the whole tree, then releases the
    handle). Idempotent and best-effort — safe to call more than once and on an
    already-dead child.

    The child drops its own registry record on the SIGTERM path; a force kill
    leaves it to the registry's pid-liveness prune.
    """
    child.stop(timeout=REAP_TIMEOUT)


def _install_shim_reaper(reap):
    """POSIX: run *reap* (tear down whatever child exists) if this shim is
    signalled to exit.

    A SIGTERM/SIGHUP delivered to the shim *alone* (not its whole group) would
    otherwise orphan the child, since Python's default handler exits without
    running ``serve``'s ``finally``. SIGINT is left to its default: it raises
    KeyboardInterrupt out of ``anyio.run`` and ``serve``'s ``finally`` reaps. On Windows
    there are no such signals — the Job Object reaps on any shim *death*, and
    ``_install_client_death_watchdog`` covers the shim being *orphaned* by its
    client — so this is a no-op there.
    """
    if os.name == "nt":
        return

    def _on_signal(signum, frame):
        reap()
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


def _install_client_death_watchdog(reap):
    """Windows: run *reap* if the stdio *client* dies unseen by stdin.

    Normal teardown runs when the bridge returns on stdin EOF (client hung
    up) or, on POSIX, when the client's process-group teardown / a SIGTERM fires
    ``_install_shim_reaper``. Windows has neither group teardown nor those
    signals, and a multi-process client can keep a duplicate of the shim's stdin
    write handle open in a surviving helper after the launching process exits, so
    EOF never arrives — the shim blocks forever in the bridge and the child +
    kernel leak (they outlive every real client). This watchdog closes that gap:
    it blocks on the client's process handle and, when the client exits for any
    reason, reaps the owned tree and exits.

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
        args=(handle, client.pid, reap),
        name="biopb-client-deathwatch",
        daemon=True,
    )
    thread.start()
    logger.info("client-death watchdog armed on client pid %s", client.pid)
    return thread


def _client_deathwatch(handle, client_pid, reap):
    """Block on the client's ``handle``; on its exit, reap the tree and exit.

    A wait error is treated as *undecided* — we do not reap, since a spurious
    reap would tear down a live session. On a real exit this ``os._exit``s past
    ``serve``'s ``finally`` (having already reaped), matching ``_on_signal``.
    """
    if not _winjob.wait_for_process(handle):
        return  # undecided (wait errored) — leave teardown to the other paths
    logger.info("stdio client (pid %s) exited; reaping the owned session", client_pid)
    reap()
    os._exit(0)


class _LazySession:
    """The session child and the client connection to it, started on first need.

    One task (:meth:`run`) owns the connection for the shim's lifetime, because
    the http client's context must be entered and left in the same task; request
    handlers only ask it to start (:meth:`get`). A failed start is reported to
    the requests waiting on it and retried by the next one. The child dying
    after a successful start ends the task with an error, which ends the shim,
    so the client sees EOF rather than a bridge to nothing.
    """

    def __init__(self, config):
        self._config = config
        self.child = None
        self._session = None
        self._error = None
        self._want = anyio.Event()
        self._done = anyio.Event()

    def reap(self):
        """Tear down the child, if one was spawned. Safe from any thread."""
        if self.child is not None:
            _reap_session(self.child)

    def _own(self, child):
        self.child = child

    async def get(self):
        """The connected ClientSession, starting the child if need be."""
        if self._session is None:
            done = self._done
            self._want.set()
            await done.wait()
            if self._session is None:
                raise RuntimeError(
                    f"the biopb-mcp session did not start: {self._error}"
                )
        return self._session

    async def run(self):
        while True:
            await self._want.wait()
            try:
                await self._start_and_serve()
            except Exception as e:  # noqa: BLE001 - reported to the waiting requests
                if self._session is not None:
                    raise  # it started, then died: end the shim
                logger.exception("biopb-mcp session failed to start")
                self.reap()
                self.child = None
                self._error = e
                done = self._done
                self._want, self._done = anyio.Event(), anyio.Event()
                done.set()

    async def _start_and_serve(self):
        # Best-effort: get the durable control plane (which owns the data plane
        # the kernel will talk to) coming up in parallel with the child's
        # import-dominated startup, without waiting for it. If it is still not up
        # when the child first needs the data plane, the child surfaces the error.
        try:
            _control_client.start_control_detached()
        except Exception:  # noqa: BLE001 - the child surfaces real errors
            logger.info("control auto-start attempt failed", exc_info=True)
        _child, url, _sid = await anyio.to_thread.run_sync(
            lambda: spawn_session(self._config, on_spawned=self._own)
        )
        logger.info("Bridging stdio to %s (owned session pid %s)", url, self.child.pid)
        async with (
            streamablehttp_client(url=url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            self._session = session
            self._done.set()
            await anyio.sleep_forever()


def build_proxy(connect, surface):
    """Build the stdio-facing MCP server.

    The list requests are answered from *surface*, the child's shipped snapshot
    (``_surface``), so listing costs no child. Every other request awaits
    ``connect()`` -- the ClientSession to the session child, started on first
    use -- and is forwarded to it. A child that cannot start fails the request,
    and a tool call as a tool error, so the agent reads why.

    Server->client traffic other than tool-call progress (sampling, elicitation,
    list_changed) is not forwarded -- biopb-mcp emits none of it, and a future
    feature that needs it must extend this bridge.
    """
    app = Server(name="biopb-mcp")

    async def _list_tools(_):
        return types.ServerResult(types.ListToolsResult(tools=surface.tools))

    app.request_handlers[types.ListToolsRequest] = _list_tools

    async def _list_resources(_):
        return types.ServerResult(
            types.ListResourcesResult(resources=surface.resources)
        )

    app.request_handlers[types.ListResourcesRequest] = _list_resources

    async def _list_resource_templates(_):
        return types.ServerResult(
            types.ListResourceTemplatesResult(
                resourceTemplates=surface.resource_templates
            )
        )

    app.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates

    async def _list_prompts(_):
        return types.ServerResult(types.ListPromptsResult(prompts=surface.prompts))

    app.request_handlers[types.ListPromptsRequest] = _list_prompts

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
            remote = await connect()
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

    async def _read_resource(req):
        remote = await connect()
        return types.ServerResult(await remote.read_resource(req.params.uri))

    app.request_handlers[types.ReadResourceRequest] = _read_resource

    async def _get_prompt(req):
        remote = await connect()
        return types.ServerResult(
            await remote.get_prompt(req.params.name, req.params.arguments)
        )

    app.request_handlers[types.GetPromptRequest] = _get_prompt

    async def _complete(req):
        remote = await connect()
        return types.ServerResult(
            await remote.complete(req.params.ref, req.params.argument.model_dump())
        )

    app.request_handlers[types.CompleteRequest] = _complete

    async def _set_logging_level(req):
        remote = await connect()
        await remote.set_logging_level(req.params.level)
        return types.ServerResult(types.EmptyResult())

    app.request_handlers[types.SetLevelRequest] = _set_logging_level

    return app


def init_options(app, surface):
    """What the shim answers ``initialize`` with: the child's capabilities, and
    the ``instructions`` the child would compose (the operation guardrails)."""
    from ._instructions import compose

    return app.create_initialization_options().model_copy(
        update={"capabilities": surface.capabilities, "instructions": compose()}
    )


async def _serve_stdio(lazy):
    from . import _surface

    surface = _surface.load()
    app = build_proxy(lazy.get, surface)
    options = init_options(app, surface)
    async with anyio.create_task_group() as tg:
        tg.start_soon(lazy.run)
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, options)
        tg.cancel_scope.cancel()


def serve(config, port=None):
    """Launcher entry point for ``--transport stdio``: bridge, spawn on first
    use, reap.

    ``port`` (the configured ``transport.port``) is vestigial here — the
    owned child binds a dynamic port — and is accepted only for call-site
    compatibility with the launcher's dispatch.
    """
    logger.warning(
        "stdio is served by bridging to a private biopb-mcp http session this "
        "shim spawns on the first request that needs it and owns (torn down when "
        "this client disconnects). Native http is recommended where the client "
        "supports it: run a persistent `biopb-mcp --transport http` server and "
        "attach with `claude mcp add --transport http biopb "
        "http://127.0.0.1:<port>/mcp`."
    )
    lazy = _LazySession(config)
    _install_shim_reaper(lazy.reap)
    _install_client_death_watchdog(lazy.reap)
    try:
        anyio.run(_serve_stdio, lazy)
    finally:
        lazy.reap()
