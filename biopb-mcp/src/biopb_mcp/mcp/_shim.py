"""stdio bridge ("shim") to a biopb-mcp http session.

``biopb-mcp --transport stdio`` does not serve MCP over fd 0/1 from the heavy
launcher process. Instead the launcher runs this module, which

1. starts **unbound**, answering ``initialize`` and the list requests itself,
   from the FastMCP server a session runs (imported, never served) plus a local
   ``attach`` tool, so a client that never attaches costs no session;
2. on ``attach``, takes the lease of a live session (``/api/lease``) and bridges
   requests to its streamable-http endpoint, or -- for ``new``, or ``--session
   new`` -- spawns a session of its own on an OS-assigned port, inheriting this
   shim's live environment (``spawn_session``, driven by ``_Binding``);
3. releases the lease, or reaps the session it spawned, on the way out
   (``_Binding.reap``).

The process that owns fd 1 as a protocol channel imports nothing that could write
to stdout (no Qt, dask, uvicorn, or kernel -- only the mcp SDK and the tool
definitions), so the fd-1 corruption class is structurally impossible here.

Ownership is per session, and only a session this shim spawned is its to end:

* **Attached.** A session another process started (the dashboard's, a
  ``biopb mcp view``) keeps running when this shim goes. The shim renews its
  lease every ``RENEW_INTERVAL`` and releases it on exit; a shim killed too hard
  to release leaves a lease that lapses by itself (``_lease.TTL``). One holder at
  a time: an attach to a held session is refused unless forced.
* **Spawned (``new``).** Ephemeral and owned: the child inherits *this* shim's
  environment, so ``DISPLAY`` / ``XAUTHORITY`` / ``WAYLAND_DISPLAY`` are the
  user's current session -- never a value frozen into a long-lived daemon (the
  #98 fix). POSIX: the child stays in this shim's process group (no
  ``start_new_session``), so the MCP client's process-group teardown takes it --
  and its kernel, via the kernel's own parent-death pipe -- down with the shim.
  Windows: this shim holds a kill-on-close Job Object the child is assigned to
  (see ``_winjob``), and a client-death watchdog
  (``_install_client_death_watchdog``) reaps the shim itself when its stdio
  *client* exits without the bridge seeing stdin EOF -- e.g. a multi-process
  client (Claude Code: a daemon + pty host) keeping a duplicate of the shim's
  stdin write handle open in a surviving helper (biopb#403, the client side).

The bridge itself is vendored rather than delegated to ``mcp-proxy``: mcp-proxy
drops the initialize ``instructions`` field that carries biopb-mcp's operation
guardrails, has no lifetime guard when the server dies, and floats its
dependencies. Here an attach carries the session's own ``instructions`` back to
the agent, a session that fails to start is a tool error the agent can read, and
one that stops answering unbinds the shim rather than leaving a hung proxy.
"""

import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import anyio
from biopb import _locations, _sessions
from biopb.lifecycle import winjob as _winjob
from biopb.lifecycle.owned_child import OwnedChild, open_child_log
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


def spawn_session(config, timeout=SESSION_START_TIMEOUT, on_spawned=None):
    """Spawn a private http child this shim owns; return (child, url, session_id).

    ``child`` is an :class:`OwnedChild`. See the module docstring for the
    ownership model. Inherits this shim's live environment (the #98 fix), binds a
    dynamic port the child reports back, and ties the child's lifetime to this
    shim (POSIX process group; Windows Job Object). On any startup failure the
    child is reaped before the error propagates, so a failed bring-up never leaks
    a process. *on_spawned* gets ``(child, session_id)`` as soon as the child
    exists, so a reap that
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
    # kernel's parent-death pipe). See biopb.lifecycle.owned_child.
    logger.info("Spawning owned biopb-mcp session: %s", cmd)
    child = OwnedChild(cmd, log=log, env=env)
    try:
        child.spawn()
    finally:
        if logged_to_file:
            log.close()  # the child holds its own duplicate of the fd
    if on_spawned is not None:
        on_spawned(child, session_id)

    try:
        # The child listens before it reports, so a reported port is ready.
        port = _await_port(child.proc, port_file, timeout)
    except BaseException:
        child.stop()
        raise
    finally:
        try:
            os.unlink(port_file)
        except OSError:
            pass

    return child, f"http://127.0.0.1:{port}/mcp", session_id


def _reap_session(child, session_id=None):
    """Tear down the owned child and its kernel grandchild (idempotent).

    Then drop *session_id*'s registry record: the child drops its own on
    SIGTERM, but Windows kills it outright, and a dead child's record is anyone's
    to prune.
    """
    child.stop(timeout=REAP_TIMEOUT)
    if session_id is not None:
        _sessions.unregister(session_id)


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


# Lease cadence: a session lapses an agent's lease after 30 s without a renewal
# (``_lease.TTL``), so the beat is a third of that and three misses in a row
# count as the session being gone.
RENEW_INTERVAL = 10.0
RENEW_FAILURES = 3
_CALL_TIMEOUT = 2.0

# No proxy for the loopback calls: an environment ``http_proxy`` would send a
# session's own port to somebody else.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class NotAttached(RuntimeError):
    """A request that needs a session arrived before one was attached."""


class AttachError(RuntimeError):
    """An attach that could not be done, worded for the agent to read."""


def _session_base(rec):
    return f"http://{rec.get('host') or '127.0.0.1'}:{rec['port']}"


def _call_session(base, method, path, body=None, timeout=_CALL_TIMEOUT):
    """``(status, json)`` from a session's loopback API; OSError if unreachable."""
    data = headers = None
    if body is not None:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        base + path, data=data, headers=headers or {}, method=method
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    return status, json.loads(raw or b"{}")


def _probe(rec):
    """A session's ``/api/status``, or None if it does not answer."""
    try:
        status, data = _call_session(_session_base(rec), "GET", "/api/status")
    except (OSError, ValueError):
        return None
    return data if status == 200 else None


def session_listing():
    """The live sessions and whether each can be attached to, as text.

    Read off the registry, each session asked for its own lease, so a record
    whose process is up but not answering reads ``unreachable`` rather than
    ``free``.
    """
    recs = [
        r for r in _sessions.list_sessions() if r.get("session_id") and r.get("port")
    ]
    if not recs:
        return "No live sessions. `attach(session='new')` starts one."
    with ThreadPoolExecutor(max_workers=min(8, len(recs))) as pool:
        statuses = list(pool.map(_probe, recs))
    lines = []
    for rec, status in zip(recs, statuses, strict=True):
        if status is None:
            state, window = "unreachable", ""
        else:
            holder = (status.get("lease") or {}).get("holder")
            state = f"held by {holder}" if holder else "free"
            window = ", viewer" if status.get("viewer") else ", no viewer"
        lines.append(f"- {rec['session_id']}: {state}{window}")
    return (
        "Live sessions (`attach(session='<id>')` takes a free one; "
        "`attach(session='new')` starts your own):\n" + "\n".join(lines)
    )


class _Binding:
    """The session this shim is attached to, and the connection to it.

    Starts unbound: the shim owns nothing until the agent attaches. Attaching to
    an existing session takes its lease and leaves its lifetime alone -- the shim
    only ever releases it. Attaching to ``new`` spawns a session this shim does
    own (:func:`spawn_session`) and reaps with it, as the shim always did.

    :meth:`attach` starts the connection in ``task_group`` (set by
    ``_serve_stdio``), where its context lives until the shim exits or the
    attachment ends. An attachment ends when the session stops answering or
    another holder takes the lease; the shim then goes back to unbound, and the
    next request is told why.

    *preselect* attaches on the first request that needs a session, for a
    launcher that already knows which one (``--session``).
    """

    def __init__(self, config, preselect=None):
        self._config = config
        self.preselect = preselect
        self.child = None  # set only for a session this shim spawned
        self.session_id = None
        self.task_group = None
        self.session = None
        self.token = uuid.uuid4().hex
        self.lost = None  # why the last attachment ended
        self.instructions = ""
        self._base = None
        self._scope = None
        self._lock = anyio.Lock()
        self._control_started = False

    def reap(self):
        """Tear down what this shim owns and release what it only holds. Safe
        from any thread."""
        child, session_id, base = self.child, self.session_id, self._base
        if child is not None:
            _reap_session(child, session_id)
        elif base is not None:
            try:
                _call_session(base, "POST", "/api/lease/release", {"token": self.token})
            except (OSError, ValueError):
                pass  # a session that is gone has no lease to release

    def _forget(self):
        self.child = self.session_id = self._base = self.session = None

    def _own(self, child, session_id):
        self.child, self.session_id = child, session_id

    async def connect(self):
        """The connected ClientSession, attaching to the preselected session
        if there is one."""
        if self.session is not None:
            return self.session
        if self.preselect is not None:
            await self.attach(self.preselect)
            return self.session
        why = f"{self.lost}. " if self.lost else ""
        raise NotAttached(
            f"{why}No session is attached. Call `attach` first.\n"
            + await anyio.to_thread.run_sync(session_listing)
        )

    async def listing(self):
        return await anyio.to_thread.run_sync(session_listing)

    async def attach(self, selector, force=False):
        """Attach to *selector* (a session id, or ``new``); the text to hand
        the agent. Raises :class:`AttachError`."""
        async with self._lock:
            if self.session is not None:
                raise AttachError(f"already attached to session {self.session_id}")
            self.lost = None
            try:
                self.session = await self.task_group.start(self._serve, selector, force)
            except AttachError:
                await anyio.to_thread.run_sync(self.reap)
                self._forget()
                raise
            except Exception as e:
                logger.exception("biopb-mcp session failed to attach")
                await anyio.to_thread.run_sync(self.reap)
                self._forget()
                raise AttachError(f"could not attach to {selector!r}: {e}") from e
        return (
            f"Attached to session {self.session_id}. The tool list has changed to "
            "this session's.\n\n" + self.instructions
        ).strip()

    def _spawn(self):
        # The durable control (which owns the data plane the kernel will talk
        # to) boots in parallel with the child's import-dominated startup, not
        # waited for: a child that needs it before it is up surfaces the error.
        if not self._control_started:
            self._control_started = True
            try:
                _control_client.start_control_detached()
            except Exception:  # noqa: BLE001 - the child surfaces real errors
                logger.info("control auto-start attempt failed", exc_info=True)
        return spawn_session(self._config, on_spawned=self._own)

    def _acquire(self, selector, force):
        """Take the lease on *selector*; return its ``/mcp`` url."""
        if selector == "new":
            _, url, _ = self._spawn()
            self._base = url.rsplit("/mcp", 1)[0]
            status, data = _call_session(
                self._base, "POST", "/api/lease/acquire", {"token": self.token}
            )
            if status != 200:
                raise AttachError(f"the new session refused its lease: {data}")
            return url
        rec = _sessions.resolve(selector)
        if rec is None or not rec.get("port"):
            raise AttachError(f"no live session {selector!r}.\n{session_listing()}")
        base = _session_base(rec)
        try:
            status, data = _call_session(
                base,
                "POST",
                "/api/lease/acquire",
                {"token": self.token, "force": force},
            )
        except (OSError, ValueError) as e:
            raise AttachError(f"session {selector} is not reachable: {e}") from e
        if status == 409:
            raise AttachError(
                f"session {selector} is held by its {data.get('holder')} "
                f"(for {data.get('age', '?')}s). `attach(session='{selector}', "
                "force=true)` takes it from them."
            )
        if status != 200:
            raise AttachError(f"session {selector} refused the attach: {data}")
        self._base, self.session_id = base, selector
        return rec.get("mcp_url") or f"{base}/mcp"

    async def _serve(self, selector, force, *, task_status):
        url = await anyio.to_thread.run_sync(self._acquire, selector, force)
        logger.info("Bridging stdio to %s (session %s)", url, self.session_id)
        started = False
        try:
            with anyio.CancelScope() as self._scope:
                async with (
                    streamablehttp_client(url=url) as (read, write, _),
                    ClientSession(read, write) as session,
                ):
                    init = await session.initialize()
                    self.instructions = init.instructions or ""
                    task_status.started(session)
                    started = True
                    async with anyio.create_task_group() as beat:
                        beat.start_soon(self._heartbeat)
                        await anyio.sleep_forever()
        except Exception as e:
            if not started:
                raise
            logger.warning("the attached session's connection failed", exc_info=True)
            self.lost = self.lost or f"The session's connection failed ({e})"
        if self.lost:
            # Not a shim exit (that cancels us past this point): the attachment
            # ended, so give back what it held and go back to unbound.
            await anyio.to_thread.run_sync(self.reap)
            self._forget()

    async def _heartbeat(self):
        base, token = self._base, self.token
        failures = 0
        while True:
            await anyio.sleep(RENEW_INTERVAL)
            try:
                status, _ = await anyio.to_thread.run_sync(
                    lambda: _call_session(
                        base, "POST", "/api/lease/renew", {"token": token}
                    )
                )
            except (OSError, ValueError):
                failures += 1
                if failures >= RENEW_FAILURES:
                    self._lose("The session stopped answering")
                    return
                continue
            failures = 0
            if status == 409:
                self._lose("Another holder took the session, or its lease lapsed")
                return

    def _lose(self, reason):
        logger.warning("attachment ended: %s", reason)
        self.lost = reason
        self._scope.cancel()


_ATTACH_TOOL = types.Tool(
    name="attach",
    description=(
        "Attach to a biopb session before using any other tool. With no "
        "arguments, lists the live sessions and whether each is free. "
        "`session` is an id from that list, or 'new' for a session of your "
        "own. A session you attach to keeps running when you disconnect; a "
        "'new' one ends with you. `force` takes a session held by something "
        "else."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "session": {"type": "string"},
            "force": {"type": "boolean"},
        },
    },
)

_ATTACH_PREFACE = (
    "Before `start_kernel` or any other tool: call `attach`. With no arguments "
    "it lists the live biopb sessions and whether each is free; "
    "`attach(session='<id>')` takes one, `attach(session='new')` starts a "
    "session of your own. Nothing else works until you have attached. The "
    "guidance below is for the session you attach to, and attaching returns "
    "its own version of it.\n\n"
)


def build_proxy(binding, server):
    """Build the stdio-facing MCP server.

    Until a session is attached, the list requests are answered by *server* --
    the FastMCP low-level server the child runs, imported here but never served
    -- with the local ``attach`` tool added, so listing costs no session. Once
    attached, the tool list is the session's own: it composes its tools and
    instructions, and it may not be this shim's version.

    Every other request awaits ``binding.connect()`` and is forwarded; with no
    session attached that is an error the agent reads, and a tool call as a tool
    error.

    Server->client traffic other than tool-call progress and the tool-list
    change on attach (sampling, elicitation, other list_changed) is not
    forwarded -- biopb-mcp emits none of it, and a feature that needs it must
    extend this bridge.
    """
    app = Server(name="biopb-mcp")
    for req in (
        types.ListResourcesRequest,
        types.ListResourceTemplatesRequest,
        types.ListPromptsRequest,
    ):
        app.request_handlers[req] = server.request_handlers[req]

    local_tools = server.request_handlers[types.ListToolsRequest]

    async def _list_tools(req):
        if binding.session is not None:
            try:
                return types.ServerResult(await binding.session.list_tools())
            except Exception:  # noqa: BLE001 - the local list is the fallback
                logger.info("session tool list failed; using the local one")
        result = await local_tools(req)
        tools = [*result.root.tools, _ATTACH_TOOL]
        return types.ServerResult(types.ListToolsResult(tools=tools))

    app.request_handlers[types.ListToolsRequest] = _list_tools

    def _text(text, error=False):
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], isError=error
            )
        )

    async def _attach(arguments):
        selector = str(arguments.get("session") or "").strip()
        try:
            if not selector:
                return _text(await binding.listing())
            text = await binding.attach(selector, bool(arguments.get("force")))
        except AttachError as e:
            return _text(str(e), error=True)
        try:
            await request_ctx.get().session.send_tool_list_changed()
        except Exception:  # noqa: BLE001 - a client that ignores it still works
            logger.debug("could not send tool list_changed", exc_info=True)
        return _text(text)

    async def _call_tool(req):
        if req.params.name == "attach":
            return await _attach(req.params.arguments or {})
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
            remote = await binding.connect()
            result = await remote.call_tool(
                req.params.name,
                req.params.arguments or {},
                progress_callback=progress_callback,
                meta=meta,
            )
            return types.ServerResult(result)
        except Exception as e:  # surface as a tool error, not a dead bridge
            return _text(str(e), error=True)

    app.request_handlers[types.CallToolRequest] = _call_tool

    def _forward(req_type, call):
        async def handler(req):
            result = await call(await binding.connect(), req.params)
            return types.ServerResult(result or types.EmptyResult())

        app.request_handlers[req_type] = handler

    _forward(types.ReadResourceRequest, lambda r, p: r.read_resource(p.uri))
    _forward(types.GetPromptRequest, lambda r, p: r.get_prompt(p.name, p.arguments))
    _forward(
        types.CompleteRequest,
        lambda r, p: r.complete(p.ref, p.argument.model_dump()),
    )
    _forward(types.SetLevelRequest, lambda r, p: r.set_logging_level(p.level))
    return app


async def _serve_stdio(binding):
    # The tool surface, and the instructions composed per session, are the
    # child's own because they are its code: importing it costs ~15 ms here and
    # starts nothing.
    from . import _server  # noqa: F401 - registers the tools on _app.mcp
    from ._app import mcp

    server = mcp._mcp_server
    app = build_proxy(binding, server)
    options = server.create_initialization_options()
    if binding.preselect is None:
        options = options.model_copy(
            update={"instructions": _ATTACH_PREFACE + (options.instructions or "")}
        )
    async with anyio.create_task_group() as tg:
        binding.task_group = tg
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, options)
        tg.cancel_scope.cancel()


def serve(config, port=None, session=None):
    """Launcher entry point for ``--transport stdio``: bridge, attach on request,
    release.

    The shim starts unbound; the agent's ``attach`` tool picks a session, or
    *session* (``--session`` / ``$BIOPB_SESSION``) picks one for it on the first
    request that needs one -- ``new`` being a session of this shim's own, torn
    down when the client disconnects.

    ``port`` (the configured ``transport.port``) is vestigial here -- a session
    this shim spawns binds a dynamic port -- and is accepted only for call-site
    compatibility with the launcher's dispatch.
    """
    logger.warning(
        "stdio is served by bridging to a biopb-mcp session: one the agent "
        "attaches to, or a private one this shim spawns for `attach new` and "
        "owns (torn down when this client disconnects). Native http is "
        "recommended where the client supports it: run a persistent "
        "`biopb-mcp --transport http` server and attach with `claude mcp add "
        "--transport http biopb http://127.0.0.1:<port>/mcp`."
    )
    binding = _Binding(config, preselect=session or None)
    _install_shim_reaper(binding.reap)
    _install_client_death_watchdog(binding.reap)
    try:
        anyio.run(_serve_stdio, binding)
    finally:
        binding.reap()
