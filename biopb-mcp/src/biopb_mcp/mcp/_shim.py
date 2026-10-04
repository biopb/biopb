"""stdio bridge ("shim") to a biopb-mcp http session.

``biopb-mcp --transport stdio`` does not serve MCP over fd 0/1 from the heavy
launcher process. Instead the launcher runs this module, which

1. starts **unbound**, answering ``initialize`` and the list requests itself,
   from the FastMCP server a session runs (imported, never served, and only
   where the import works; otherwise ``attach`` alone is listed and the rest
   follows ``list_changed``) plus a local ``attach`` tool, so a client that never
   attaches costs no session;
2. on ``attach``, takes the lease of a live session (``/api/lease``) and bridges
   requests to its streamable-http endpoint. For ``new`` (or ``--session new``)
   it first asks the control to launch a session, carrying this client's display
   environment -- an error if no control answers, since a session without one
   has no data plane. Attaching returns the session's own operating rules, and
   attaching again to the same session returns them again;
3. releases the lease on the way out.

The process that owns fd 1 as a protocol channel imports nothing that could write
to stdout (no Qt, dask, uvicorn, or kernel -- only the mcp SDK and the tool
definitions), so the fd-1 corruption class is structurally impossible here.

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
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import anyio
from biopb import _locations, _sessions
from biopb.lifecycle import winjob as _winjob
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel.server import NotificationOptions, Server, request_ctx
from mcp.server.stdio import stdio_server

from .. import _control_client

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

    def __init__(self, preselect=None):
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
            _call_session(base, "POST", "/api/lease/release", {"token": self.token})
        except (OSError, ValueError):
            pass  # a session that is gone has no lease to release

    def _forget(self):
        self.session_id = self._base = self.session = None
        self.managed = False

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

        Best-effort: a client that ignores these still works against the full
        list the shim advertised unbound, where it has one.
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
            status, data = _call_session(
                base,
                "POST",
                "/api/lease/acquire",
                {"token": self.token, "force": force},
            )
        except (OSError, ValueError) as e:
            raise AttachError(f"session {label} is not reachable: {e}") from e
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
        url. ``AttachError`` when the control cannot be reached or will not
        launch one.

        There is deliberately no session of the shim's own to fall back to: a
        session reaches the data and algorithm planes through the control, so one
        started without it would attach "successfully" and fail on first use,
        hiding the cause. The session is the user's from here: it is detached
        and runs until they stop it, so this shim only ever releases it.
        """
        log = getattr(_locations, "control_log", None)
        where = f" (its log: {log()})" if log is not None else ""
        try:
            if not _control_client.ensure_control(CONTROL_WAIT):
                raise AttachError(
                    f"no control answered within {CONTROL_WAIT:.0f}s{where}. "
                    "Retry, or start it with `biopb control start`."
                )
            answer = _control_client.launch_session(
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
        self._take_lease(base, rec["session_id"])
        self.managed = True
        return rec.get("mcp_url") or f"{base}/mcp"

    def _acquire(self, selector, force):
        """Take the lease on *selector*; return its ``/mcp`` url."""
        if selector == "new":
            return self._launch_via_control()
        rec = _sessions.resolve(selector)
        if rec is None or not rec.get("port"):
            raise AttachError(f"no live session {selector!r}.\n{session_listing()}")
        base = _session_base(rec)
        self._take_lease(base, selector, force)
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
            await anyio.to_thread.run_sync(self.release)
            self._forget()
            await self.announce_change()

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


def _local_server():
    """The FastMCP low-level server of a session, imported but never served; or
    None where it cannot be imported.

    It is the shim's copy of the tool, resource and prompt lists and of the
    handshake instructions, so a client that cannot follow ``list_changed`` still
    sees every tool. Without it the shim advertises ``attach`` alone and relies on
    ``list_changed`` for the rest. Importing it costs ~15 ms and starts nothing.
    """
    try:
        from . import _server  # noqa: F401 - registers the tools on _app.mcp
        from ._app import mcp

        return mcp._mcp_server
    except Exception:  # noqa: BLE001 - a lighter shim is the fallback
        logger.info(
            "the session surface cannot be imported; advertising `attach` only",
            exc_info=True,
        )
        return None


def build_proxy(binding, server=None):
    """Build the stdio-facing MCP server.

    Until a session is attached, the list requests are answered by *server* --
    the FastMCP low-level server a session runs, imported here but never served
    (:func:`_local_server`) -- with the local ``attach`` tool added, so listing
    costs no session. With no *server* they list ``attach`` alone, and empty
    resources and prompts. Once attached, every list is the session's own: it
    composes its tools and instructions, and it may not be this shim's version.

    Every other request awaits ``binding.connect()`` and is forwarded; with no
    session attached that is an error the agent reads, and a tool call as a tool
    error.

    Server->client traffic other than tool-call progress and the list changes on
    attach and detach (sampling, elicitation) is not forwarded -- biopb-mcp emits
    none of it, and a feature that needs it must extend this bridge.
    """
    app = Server(name="biopb-mcp")

    def _list(req_type, remote_name, empty, extra=()):
        local = server.request_handlers[req_type] if server is not None else None

        async def handler(req):
            if binding.session is not None:
                try:
                    return types.ServerResult(
                        await getattr(binding.session, remote_name)()
                    )
                except Exception:  # noqa: BLE001 - the local list is the fallback
                    logger.info("session %s failed; using the local one", remote_name)
            result = empty([]) if local is None else (await local(req)).root
            return types.ServerResult(empty([*_items(result), *extra]))

        app.request_handlers[req_type] = handler

    def _items(result):
        for name in ("tools", "resources", "resourceTemplates", "prompts"):
            if hasattr(result, name):
                return getattr(result, name)
        return []

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


def _handshake(app, server, preselect):
    """The initialize answer: the session surface's own where the shim has it,
    plus the attach paragraph unless a session is preselected.

    The lists change on attach and detach, so it declares ``listChanged``: a
    client only follows those notifications from a server that says it sends
    them.
    """
    changes = NotificationOptions(
        tools_changed=True, resources_changed=True, prompts_changed=True
    )
    options = (server or app).create_initialization_options(changes)
    if preselect is None:
        options = options.model_copy(
            update={"instructions": _ATTACH_PREFACE + (options.instructions or "")}
        )
    return options


async def _serve_stdio(binding):
    server = _local_server()
    app = build_proxy(binding, server)
    options = _handshake(app, server, binding.preselect)
    async with anyio.create_task_group() as tg:
        binding.task_group = tg
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, options)
        tg.cancel_scope.cancel()


def serve(session=None):
    """Launcher entry point for ``--transport stdio``: bridge, attach on request,
    release.

    The shim starts unbound; the agent's ``attach`` tool picks a session, or
    *session* (``--session`` / ``$BIOPB_SESSION``) picks one for it on the first
    request that needs one -- ``new`` having the control launch it.
    """
    logger.warning(
        "stdio is served by bridging to a biopb-mcp session the agent attaches "
        "to. Native http is recommended where the client supports it: run a "
        "persistent `biopb-mcp --transport http` server and attach with `claude "
        "mcp add --transport http biopb http://127.0.0.1:<port>/mcp`."
    )
    binding = _Binding(preselect=session or None)
    _install_release_on_signal(binding.release)
    _install_client_death_watchdog(binding.release)
    try:
        anyio.run(_serve_stdio, binding)
    finally:
        binding.release()
