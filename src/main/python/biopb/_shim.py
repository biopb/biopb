"""stdio bridge ("shim") to a biopb-mcp http session.

``biopb-shim`` (``biopb-mcp --transport stdio`` runs the same code) does not serve
MCP over fd 0/1 from a session. It is the SDK's own light process, needing only
the ``mcp`` package (``biopb[shim]``), and it

1. starts **unbound**, answering ``initialize`` with a paragraph that says to call
   the local ``attach`` tool, which is the only tool it lists, so a client that
   never attaches costs no session. It holds no copy of any session's tools,
   resources, prompts or instructions;
2. on ``attach``, takes the lease of a live session (``/api/lease``) and bridges
   requests to its streamable-http endpoint, and tells the client its lists
   changed (``list_changed``). For ``new`` it first asks the control to launch a
   session, carrying this client's display environment -- an error if no control
   answers, since a session without one has no data plane. Attaching returns the
   session's own operating rules, and attaching again to the same session
   returns them again;
3. releases the lease on the way out.

A client that cannot follow ``list_changed`` (Codex does not refresh its tool
list within a turn) is served by ``--session`` instead: the shim binds an id,
``new``, or ``auto`` (the newest free session, else a new one) *before* it
answers ``initialize``, so the answer already carries the session's own
instructions and tools, and the client never needs a refresh. The client gives up
choosing a session in exchange.

The process that owns fd 1 as a protocol channel imports nothing that could write
to stdout (no Qt, dask, uvicorn, kernel, or session code -- only the mcp SDK), so
the fd-1 corruption class is structurally impossible here. The session reaches the
shim only over HTTP: ``/api/lease``, ``/api/status``, ``/api/sessions`` and the
session registry are the contract (docs/session-attach.md).

With ``--remote`` the sessions are another machine's, reached through its control
under the token (``_Remote``): the same lease, status and ``/mcp``, at
``<control>/session/<id>/...``. Everything below holds there too.

The shim owns no session. Every session is a detached process the control
launched, or a person did (``biopb mcp view``, the dashboard), and it runs until
a person stops it from the dashboard; the shim only holds its lease while the
agent is attached, and the next agent can attach to the same session. A session
that is stopped, or that another holder takes by force, unbinds the shim, and its
next request says why. So there is nothing here to reap: no child, no process
group, no Job Object.

What the shim does own is itself. A shim that outlived its client would go on
renewing its lease and keep the session locked, so it releases and exits when the
client goes: on stdin EOF, on SIGTERM/SIGHUP (``_install_release_on_signal``),
and -- where a multi-process client can keep a duplicate of the stdin write
handle open after it is gone, as Claude Code does on Windows, biopb#403 --
through a watchdog on the client's process (``_install_client_death_watchdog``).
A shim killed too hard to release leaves a lease that lapses by itself
(``_lease.TTL``).

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
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlencode

import anyio
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel.server import NotificationOptions, Server, request_ctx
from mcp.server.stdio import stdio_server

from . import _control, _control_launch, _locations, _sessions
from .lifecycle import winjob as _winjob

logger = logging.getLogger(__name__)


def _install_release_on_signal(release):
    """POSIX: run *release* (give the session's lease back) if this shim is
    signalled to exit.

    A SIGTERM/SIGHUP would otherwise end the shim through Python's default
    handler without running ``serve``'s ``finally``, leaving the lease to lapse
    on its own 30 s later and the session locked until then. SIGINT is left to
    its default: it raises KeyboardInterrupt out of ``anyio.run`` and ``serve``'s
    ``finally`` releases. On Windows there are no such signals;
    ``_install_client_death_watchdog`` covers the shim being *orphaned* by its
    client, so this is a no-op there.
    """
    if os.name == "nt":
        return

    def _on_signal(signum, frame):
        release()
        os._exit(0)

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass


# A process in *our own* launcher chain (the shim and any interpreter-launcher
# stubs above it), as opposed to the MCP client that spawned it, shares the argv
# the client invoked us with (``biopb-shim ...``, or the older ``biopb-mcp
# --transport stdio``), and re-exec launchers (e.g. a venv built on Microsoft
# Store Python inserts one) preserve it. The client's own cmdline (claude.exe, an
# editor, a shell) matches neither.
def _is_ours(cmdline):
    return "biopb" in cmdline and ("shim" in cmdline or "stdio" in cmdline)


def _find_client_process():
    """Walk up past our own launcher chain to the MCP client that owns us.

    ``os.getppid()`` is NOT the client when an interpreter-launcher stub sits
    between us and it — the case that made a naive parent-watch useless: on a
    venv built on Store Python the stub *outlives* the client (it only waits on
    us), so watching it never fires. We instead climb ancestors while their
    cmdline looks like our chain (:func:`_is_ours`) and return the first
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
            if not _is_ours(cl):
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


def _install_client_death_watchdog(release):
    """Windows: run *release* and exit if the stdio *client* dies unseen by stdin.

    Normal teardown runs when the bridge returns on stdin EOF (client hung up)
    or, on POSIX, when the client's process-group teardown / a SIGTERM fires
    ``_install_release_on_signal``. Windows has neither group teardown nor those
    signals, and a multi-process client can keep a duplicate of the shim's stdin
    write handle open in a surviving helper after the launching process exits, so
    EOF never arrives -- the shim blocks forever in the bridge, renewing its
    lease for a client that is gone, and the session stays locked to it. This
    watchdog closes that gap: it blocks on the client's process handle and, when
    the client exits for any reason, releases the lease and exits.

    The client is found by :func:`_find_client_process` (walking past our own
    launcher stubs -- see its docstring for why ``os.getppid()`` is not enough).
    We hold a handle to that exact process object, so the wait is immune to pid
    reuse. If the client cannot be found or opened at arm time, we do not arm:
    the shim is never ended off an uncertain baseline; stdin EOF remains the
    backstop. A no-op off Windows.

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
        args=(handle, client.pid, release),
        name="biopb-client-deathwatch",
        daemon=True,
    )
    thread.start()
    logger.info("client-death watchdog armed on client pid %s", client.pid)
    return thread


def _client_deathwatch(handle, client_pid, release):
    """Block on the client's ``handle``; on its exit, release the lease and exit.

    A wait error is treated as *undecided* -- we do not end the shim, since a
    spurious exit would drop a live attachment. On a real exit this ``os._exit``s
    past ``serve``'s ``finally`` (having already released), matching the signal
    handler.
    """
    if not _winjob.wait_for_process(handle):
        return  # undecided (wait errored) -- leave teardown to the other paths
    logger.info("stdio client (pid %s) exited; releasing the session", client_pid)
    release()
    os._exit(0)


# Lease cadence: a session lapses an agent's lease after 30 s without a renewal
# (``_lease.TTL``), so the beat is a third of that and three misses in a row
# count as the session being gone.
RENEW_INTERVAL = 10.0
RENEW_FAILURES = 3
_CALL_TIMEOUT = 2.0

# How long `attach new` waits for a control to answer, and for it to launch a
# session.
CONTROL_WAIT = 20.0
CONTROL_LAUNCH_TIMEOUT = 70.0

# Where a viewer window can appear: what the control is told to use for a
# session it launches for this client, in place of its own frozen environment.
# Mirrors the control's allowlist (``_DISPLAY_ENV``); it drops anything else.
_DISPLAY_ENV = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_RUNTIME_DIR",
    "XDG_SESSION_TYPE",
)

# No proxy for the loopback calls: an environment ``http_proxy`` would send a
# session's own port to somebody else.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class NotAttached(RuntimeError):
    """A request that needs a session arrived before one was attached."""


class AttachError(RuntimeError):
    """An attach that could not be done, worded for the agent to read."""


def _session_base(rec):
    """Where a session's API is: its own loopback port, or (a record from a
    remote control) the control's proxy of it."""
    return rec.get("base") or f"http://{rec.get('host') or '127.0.0.1'}:{rec['port']}"


def _call_session(
    base, method, path, body=None, timeout=_CALL_TIMEOUT, headers=None, opener=_OPENER
):
    """``(status, json)`` from a session's API; OSError if unreachable."""
    data = None
    headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with opener.open(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    return status, json.loads(raw or b"{}")


def _probe(rec, headers=None, opener=_OPENER):
    """A session's ``/api/status``, or None if it does not answer."""
    try:
        status, data = _call_session(
            _session_base(rec), "GET", "/api/status", headers=headers, opener=opener
        )
    except (OSError, ValueError):
        return None
    return data if status == 200 else None


def _session_states():
    """``[(record, /api/status or None)]`` for the live sessions, newest first,
    each asked for its own lease and window."""
    recs = [
        r for r in _sessions.list_sessions() if r.get("session_id") and r.get("port")
    ]
    if not recs:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(recs))) as pool:
        return list(zip(recs, pool.map(_probe, recs), strict=True))


def _holder(status):
    return (status.get("lease") or {}).get("holder")


def session_listing():
    """The live sessions and whether each can be attached to, as text.

    A record whose process is up but not answering reads ``unreachable`` rather
    than ``free``.
    """
    return _format_listing(_session_states())


def _format_listing(states):
    if not states:
        return "No live sessions. `attach(session='new')` starts one."
    lines = []
    for rec, status in states:
        if status is None:
            state, window = "unreachable", ""
        else:
            holder = _holder(status)
            state = f"held by {holder}" if holder else "free"
            window = ", viewer" if status.get("viewer") else ", no viewer"
        lines.append(f"- {rec['session_id']}: {state}{window}")
    return (
        "Live sessions (`attach(session='<id>')` takes a free one; "
        "`attach(session='new')` starts your own):\n" + "\n".join(lines)
    )


def _newest_free_session():
    """The newest session nothing holds, or None.

    What ``--session auto`` takes for a client that cannot choose: it decides
    before it has said a word to the agent, so it cannot be asked.
    """
    for rec, status in _session_states():
        if status is not None and not _holder(status):
            return rec
    return None


class _Local:
    """Sessions on this machine: the registry, each one's own loopback port, and
    a control on this machine to launch more."""

    headers = {}
    remote = False

    def call(self, base, method, path, body=None):
        return _call_session(base, method, path, body)

    def states(self):
        return _session_states()

    def listing(self):
        return session_listing()

    def newest_free(self):
        return _newest_free_session()

    def find(self, selector):
        """``(base, mcp_url)`` of the live session *selector*, or None."""
        rec = _sessions.resolve(selector)
        if rec is None or not rec.get("port"):
            return None
        base = _session_base(rec)
        return base, rec.get("mcp_url") or f"{base}/mcp"

    def launch(self):
        """``(session_id, base, mcp_url)`` of a session the control launched for
        this client. ``AttachError`` when it cannot be reached or will not launch
        one.

        There is deliberately no session of the shim's own to fall back to: a
        session reaches the data and algorithm planes through the control, so one
        started without it would attach "successfully" and fail on first use,
        hiding the cause.
        """
        log = getattr(_locations, "control_log", None)
        where = f" (its log: {log()})" if log is not None else ""
        try:
            if not _control_launch.ensure_control(CONTROL_WAIT):
                raise AttachError(
                    f"no control answered within {CONTROL_WAIT:.0f}s{where}. "
                    "Retry, or start it with `biopb control start`."
                )
            answer = _control_launch.launch_session(
                display={k: os.environ[k] for k in _DISPLAY_ENV if k in os.environ},
                timeout=CONTROL_LAUNCH_TIMEOUT,
            )
        except (OSError, ValueError) as e:
            raise AttachError(
                f"the control would not launch a session: {e}{where}"
            ) from e
        state = answer.get("state")
        if state == "failed":
            raise AttachError(
                f"the session did not start: {answer.get('error')}\n"
                f"{answer.get('log') or ''}".strip()
            )
        rec = _sessions.resolve(answer.get("session_id") or "")
        if state != "started" or rec is None or not rec.get("port"):
            raise AttachError(
                "the session is still starting; `attach` with no arguments "
                "lists it once it is up"
            )
        base = _session_base(rec)
        return rec["session_id"], base, rec.get("mcp_url") or f"{base}/mcp"


class _Remote:
    """Sessions on another machine, through its control: the one public
    listener, under the token. Lease, status and ``/mcp`` of session ``<id>`` are
    ``<control>/session/<id>/...``; the sessions themselves stay on the host's
    loopback."""

    remote = True

    def __init__(self, url, token, headers=None):
        self.url = url.rstrip("/")
        # *headers* are for whatever stands in front of the control (a portal's
        # session cookie); the token travels in its own header.
        self.headers = {**(headers or {}), "X-Biopb-Token": token}
        # Unlike a session's loopback port, this address is reached the way any
        # other is: through the environment's proxy settings.
        self._opener = urllib.request.build_opener()

    def call(self, base, method, path, body=None, timeout=_CALL_TIMEOUT):
        return _call_session(
            base,
            method,
            path,
            body,
            timeout,
            headers=self.headers,
            opener=self._opener,
        )

    def _base(self, session_id):
        return f"{self.url}/session/{quote(session_id, safe='')}"

    def _ask(self, method, path, timeout=10.0):
        """The control's JSON answer; ``AttachError`` if it cannot be had."""
        try:
            status, data = self.call(self.url, method, path, timeout=timeout)
        except (OSError, ValueError) as e:
            raise AttachError(f"the control at {self.url} is not reachable: {e}") from e
        if status in (401, 403):
            raise AttachError(
                f"the control at {self.url} refused the token "
                "(--token or $BIOPB_TENSOR_TOKEN)"
            )
        if status != 200:
            raise AttachError(f"the control at {self.url} answered {status}: {data}")
        return data

    def check(self):
        """Refuse early, with the reason, when this control serves no ``/mcp``."""
        if self._ask("GET", "/health").get("mcp_proxied") is not True:
            raise AttachError(
                f"the control at {self.url} does not serve /mcp for remote "
                "agents: it must run on a public bind (`biopb control start "
                "--remote`)"
            )

    def states(self):
        recs = [
            {"session_id": row["session_id"], "base": self._base(row["session_id"])}
            for row in self._ask("GET", "/api/sessions").get("sessions", [])
            if row.get("session_id")
        ]
        if not recs:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(recs))) as pool:
            statuses = pool.map(
                lambda rec: _probe(rec, self.headers, self._opener), recs
            )
            return list(zip(recs, statuses, strict=True))

    def listing(self):
        try:
            return _format_listing(self.states())
        except AttachError as e:
            return f"Could not list the sessions: {e}"

    def newest_free(self):
        for rec, status in self.states():
            if status is not None and not _holder(status):
                return rec
        return None

    def find(self, selector):
        base = self._base(selector)
        return base, f"{base}/mcp"

    def launch(self):
        """A session on the host, started without a display: the viewer, and the
        window it would open, belong to the host and not to this client."""
        query = urlencode(
            {
                "start_kernel": 0,
                "display": "{}",
                "client_timeout": CONTROL_LAUNCH_TIMEOUT,
            }
        )
        try:
            status, answer = self.call(
                self.url,
                "POST",
                f"/api/sessions/new?{query}",
                timeout=CONTROL_LAUNCH_TIMEOUT,
            )
        except (OSError, ValueError) as e:
            raise AttachError(f"the control would not launch a session: {e}") from e
        if status in (401, 403):
            raise AttachError(f"the control at {self.url} refused the token")
        state = answer.get("state")
        if status != 200 or state == "failed":
            raise AttachError(
                f"the session did not start: {answer.get('error') or answer}"
            )
        if state != "started" or not answer.get("session_id"):
            raise AttachError(
                "the session is still starting; `attach` with no arguments "
                "lists it once it is up"
            )
        base = self._base(answer["session_id"])
        return answer["session_id"], base, f"{base}/mcp"


class _Binding:
    """The session this shim is attached to, and the connection to it.

    Starts unbound, and never owns a session: attaching takes its lease and
    leaves its lifetime alone, and the shim only ever releases it. ``new`` has
    the control launch one first.

    :meth:`attach` starts the connection in ``task_group`` (set by
    ``_serve_stdio``), where its context lives until the shim exits or the
    attachment ends. An attachment ends when the session stops answering or
    another holder takes the lease; the shim then goes back to unbound, and the
    next request is told why.

    *preselect* attaches on the first request that needs a session, for a
    launcher that already knows which one (``--session``).
    """

    def __init__(self, preselect=None, plane=None):
        self.plane = plane or _Local()
        self.preselect = preselect
        self.session_id = None
        self.task_group = None
        self.session = None
        self.token = uuid.uuid4().hex
        self.lost = None  # why the last attachment ended
        self.managed = False  # a session the control launched for this client
        self.instructions = ""
        self.server_session = None  # the client-facing session, to notify through
        self._base = None
        self._scope = None
        self._lock = anyio.Lock()

    def release(self):
        """Give back the lease, if one is held. Safe from any thread."""
        base = self._base
        if base is None:
            return
        try:
            self.plane.call(base, "POST", "/api/lease/release", {"token": self.token})
        except (OSError, ValueError):
            pass  # a session that is gone has no lease to release

    def _forget(self):
        self.session_id = self._base = self.session = None
        self.managed = False

    async def connect(self):
        """The connected ClientSession, attaching to the preselected session
        if there is one and it has not been lost."""
        if self.session is not None:
            return self.session
        if self.preselect is not None and not self.lost:
            # A bind that failed at start is retried; a session that was lost
            # (stopped, or taken by force) is reported, not silently replaced.
            await self.attach(self.preselect)
            return self.session
        why = f"{self.lost}. " if self.lost else ""
        raise NotAttached(
            f"{why}No session is attached. Call `attach` first.\n"
            + await anyio.to_thread.run_sync(self.plane.listing)
        )

    async def listing(self):
        return await anyio.to_thread.run_sync(self.plane.listing)

    async def attach(self, selector, force=False):
        """Attach to *selector* (a session id, or ``new``); the text to hand
        the agent. Raises :class:`AttachError`.

        Attaching to the session already attached is not an error: it hands back
        the same text, so an agent whose context no longer holds the rules can
        ask for them again.
        """
        async with self._lock:
            if self.session is not None:
                if selector == self.session_id:
                    return self._attached_text()
                raise AttachError(f"already attached to session {self.session_id}")
            self.lost = None
            try:
                self.session = await self.task_group.start(self._serve, selector, force)
            except AttachError:
                await anyio.to_thread.run_sync(self.release)
                self._forget()
                raise
            except Exception as e:
                logger.exception("biopb-mcp session failed to attach")
                await anyio.to_thread.run_sync(self.release)
                self._forget()
                raise AttachError(f"could not attach to {selector!r}: {e}") from e
        return self._attached_text()

    def _attached_text(self):
        note = (
            " It keeps running after you disconnect, so it can be attached to "
            "again; the user stops it from the dashboard."
            if self.managed
            else ""
        )
        return (
            f"Attached to session {self.session_id}.{note} Its tools, resources "
            "and prompts have replaced the ones listed before. The rules below "
            "apply to every later turn; call `attach` again with this session id "
            "to read them again.\n\n" + self.instructions
        ).strip()

    async def announce_change(self):
        """Tell the client its tool, resource and prompt lists changed.

        Best-effort: a client that ignores these is the one ``--session`` is for.
        """
        notifier = self.server_session
        if notifier is None:
            return
        for send in (
            notifier.send_tool_list_changed,
            notifier.send_resource_list_changed,
            notifier.send_prompt_list_changed,
        ):
            try:
                await send()
            except Exception:  # noqa: BLE001 - the client may be gone
                logger.debug("could not send a list_changed", exc_info=True)

    def _take_lease(self, base, label, force=False):
        """Take the lease on the session at *base*, named *label*."""
        try:
            status, data = self.plane.call(
                base,
                "POST",
                "/api/lease/acquire",
                {"token": self.token, "force": force},
            )
        except (OSError, ValueError) as e:
            raise AttachError(f"session {label} is not reachable: {e}") from e
        if status == 404:
            raise AttachError(f"no live session {label!r}.\n{self.plane.listing()}")
        if status == 409:
            raise AttachError(
                f"session {label} is held by its {data.get('holder')} "
                f"(for {data.get('age', '?')}s). `attach(session='{label}', "
                "force=true)` takes it from them."
            )
        if status != 200:
            raise AttachError(f"session {label} refused the attach: {data}")
        self._base, self.session_id = base, label

    def _launch_via_control(self):
        """A session the control launched for this client, leased; its ``/mcp``
        url. The session is the user's from here: it is detached and runs until
        they stop it, so this shim only ever releases it."""
        session_id, base, url = self.plane.launch()
        self._take_lease(base, session_id)
        self.managed = True
        return url

    def _acquire(self, selector, force):
        """Take the lease on *selector*; return its ``/mcp`` url.

        *selector* is a session id, ``new``, or ``auto``: the newest free
        session, else a new one.
        """
        if self.plane.remote:
            self.plane.check()
        if selector == "auto":
            free = self.plane.newest_free()
            selector = free["session_id"] if free else "new"
        if selector == "new":
            return self._launch_via_control()
        found = self.plane.find(selector)
        if found is None:
            raise AttachError(f"no live session {selector!r}.\n{self.plane.listing()}")
        base, url = found
        self._take_lease(base, selector, force)
        return url

    async def _serve(self, selector, force, *, task_status):
        url = await anyio.to_thread.run_sync(self._acquire, selector, force)
        logger.info("Bridging stdio to %s (session %s)", url, self.session_id)
        started = False
        try:
            with anyio.CancelScope() as self._scope:
                async with (
                    streamablehttp_client(
                        url=url, headers=self.plane.headers or None
                    ) as (
                        read,
                        write,
                        _,
                    ),
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
            await anyio.to_thread.run_sync(self.release)
            self._forget()
            await self.announce_change()

    async def _heartbeat(self):
        base, token, call = self._base, self.token, self.plane.call
        failures = 0
        while True:
            await anyio.sleep(RENEW_INTERVAL)
            try:
                status, _ = await anyio.to_thread.run_sync(
                    lambda: call(base, "POST", "/api/lease/renew", {"token": token})
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
        "own. A session you attach to keeps running when you disconnect. "
        "`force` takes a session held by something else. The result carries "
        "the session's operating rules, which apply to every later turn; "
        "attach to the same session again to read them again."
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
    "session of your own. Nothing else works until you have attached. What "
    "attaching returns are that session's own operating rules and its tools: "
    "follow the rules for every later turn, and call `attach` again with the "
    "same session id if they are no longer in your context.\n\n"
)


def build_proxy(binding):
    """Build the stdio-facing MCP server.

    Until a session is attached, ``tools/list`` lists the local ``attach`` tool
    alone, and resources and prompts are empty: the shim holds no copy of any
    session's surface. Once attached, every list is the session's own, and the
    client is told so by ``list_changed``. A client that cannot follow that is
    served by binding first (``--session``; see :func:`_serve_stdio`), so its
    first list is already the session's.

    Every other request awaits ``binding.connect()`` and is forwarded; with no
    session attached that is an error the agent reads, and a tool call as a tool
    error.

    Server->client traffic other than tool-call progress and the list changes on
    attach and detach (sampling, elicitation) is not forwarded -- biopb-mcp emits
    none of it, and a feature that needs it must extend this bridge.
    """
    app = Server(name="biopb-mcp")

    def _list(req_type, remote_name, empty, extra=()):
        async def handler(req):
            if binding.session is not None:
                try:
                    return types.ServerResult(
                        await getattr(binding.session, remote_name)()
                    )
                except Exception:  # noqa: BLE001 - unbound is the fallback
                    logger.info("session %s failed; listing none", remote_name)
            return types.ServerResult(empty(list(extra)))

        app.request_handlers[req_type] = handler

    _list(
        types.ListToolsRequest,
        "list_tools",
        lambda items: types.ListToolsResult(tools=items),
        extra=(_ATTACH_TOOL,),
    )
    _list(
        types.ListResourcesRequest,
        "list_resources",
        lambda items: types.ListResourcesResult(resources=items),
    )
    _list(
        types.ListResourceTemplatesRequest,
        "list_resource_templates",
        lambda items: types.ListResourceTemplatesResult(resourceTemplates=items),
    )
    _list(
        types.ListPromptsRequest,
        "list_prompts",
        lambda items: types.ListPromptsResult(prompts=items),
    )

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
            binding.server_session = request_ctx.get().session
        except LookupError:  # called outside a request, as the unit tests do
            pass
        await binding.announce_change()
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


def _handshake(app, binding, note=""):
    """The initialize answer.

    Bound before it (``--session``), it is the session's own: its instructions
    in the privileged slot a client puts in front of the model from the first
    turn. Unbound, it is the attach paragraph, plus *note* if a requested
    session could not be attached.

    It declares ``listChanged`` either way: the lists change on attach and on
    detach, and a client only follows those notifications from a server that
    says it sends them.
    """
    changes = NotificationOptions(
        tools_changed=True, resources_changed=True, prompts_changed=True
    )
    options = app.create_initialization_options(changes)
    if binding.session is not None:
        instructions = binding.instructions
    else:
        instructions = _ATTACH_PREFACE + note
    return options.model_copy(update={"instructions": instructions})


async def _serve_stdio(binding):
    app = build_proxy(binding)
    async with anyio.create_task_group() as tg:
        binding.task_group = tg
        note = ""
        if binding.preselect is not None:
            # Before the first byte is read: the client's `initialize` waits in
            # the pipe, and what it gets back is the session's own tools and
            # rules, so a client that never refreshes its tool list still has
            # the right one.
            try:
                await binding.attach(binding.preselect)
            except AttachError as e:
                note = (
                    f"Attaching `{binding.preselect}` at start failed: {e}\n\n"
                    "It is retried on the first request that needs a session.\n\n"
                )
        options = _handshake(app, binding, note)
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, options)
        tg.cancel_scope.cancel()


def serve(session=None, remote=None, token=None, headers=None):
    """Bridge stdio to a session: attach on request, release on the way out.

    The shim starts unbound; the agent's ``attach`` tool picks a session. Or
    *session* (``--session`` / ``$BIOPB_SESSION``) binds one before the
    handshake: an id, ``new`` (the control launches one), or ``auto`` (the newest
    free session, else a new one). That is for a client that cannot follow
    ``list_changed``.

    *remote* is the URL of another machine's control (``--remote``): sessions are
    then that machine's, reached through it under *token* (and any extra
    *headers*), and "new" starts one there.
    """
    plane = _Remote(remote, token, headers) if remote else None
    binding = _Binding(preselect=session or None, plane=plane)
    _install_release_on_signal(binding.release)
    _install_client_death_watchdog(binding.release)
    try:
        anyio.run(_serve_stdio, binding)
    finally:
        binding.release()


def main(argv=None):
    """``biopb-shim [--session <id|new|auto>]``: the entry a client spawns."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="biopb-shim",
        description="stdio bridge to a biopb-mcp session an agent attaches to.",
    )
    parser.add_argument(
        "--session",
        default=os.environ.get("BIOPB_SESSION") or None,
        help="Bind a session before the handshake: an id, 'new' (the control "
        "launches one) or 'auto' (the newest free session, else a new one). For a "
        "client that cannot follow tools/list_changed; default: attach on request.",
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get("BIOPB_REMOTE") or None,
        metavar="URL",
        help="Attach to the sessions of another machine's control (a `biopb "
        "control start --remote`), e.g. https://host:8813. Also $BIOPB_REMOTE. "
        "Needs the control's token: --token or $BIOPB_TENSOR_TOKEN.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="The remote control's token (default: $BIOPB_TENSOR_TOKEN). This "
        "machine's own credential file is never sent to a remote.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="'Name: value'",
        help="An extra header for every request to --remote, e.g. the session "
        "cookie of a portal in front of the control. Repeatable. Also "
        "$BIOPB_REMOTE_HEADERS, one per line.",
    )
    opts = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    token = None
    headers = {}
    if opts.remote:
        lines = [*os.environ.get("BIOPB_REMOTE_HEADERS", "").splitlines(), *opts.header]
        for line in lines:
            name, sep, value = line.partition(":")
            if not line.strip():
                continue
            if not sep or not name.strip() or any(c in name for c in " \t"):
                parser.error(f"--header wants 'Name: value', not {line!r}")
            headers[name.strip()] = value.strip()
        token = _control.resolve_data_plane_token(
            opts.token, allow_credential_file=False
        )
        if not token:
            parser.error(
                "--remote needs the control's token (--token or $BIOPB_TENSOR_TOKEN)"
            )
    try:
        serve(session=opts.session, remote=opts.remote, token=token, headers=headers)
    except Exception:
        logger.exception("stdio bridge failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
