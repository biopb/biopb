"""Tests for the stdio bridge ("shim"): the attach tool, the lease it holds, and
the request forwarding of the vendored proxy.

Unit tests cover the proxy (what an unattached shim lists and refuses, what an
attached one forwards), the session listing, the binding's attach / release /
lost-lease paths, and the guards that end a shim whose client is gone. Two
end-to-end tests run the real thing: ``biopb-mcp --transport stdio`` as a
subprocess, against a real control that launches a real session -- which must
outlive the agent, take the next one, and stop when the user stops it.
"""

import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import anyio
import pytest
from mcp import types

from biopb_mcp.mcp import (
    _server,  # noqa: F401 - registers the tools
    _shim,
)
from biopb_mcp.mcp._app import mcp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeProc:
    """psutil.Process stand-in for the ancestor-walk: a cmdline + a parent link."""

    def __init__(self, cmdline, parent=None):
        self._cmdline = cmdline
        self._parent = parent

    def cmdline(self):
        return self._cmdline

    def parent(self):
        return self._parent

    @property
    def pid(self):
        return 4242


def _fake_psutil(me):
    import types as _types

    mod = _types.SimpleNamespace(Process=lambda *a, **k: me)
    return mod


class TestFindClientProcess:
    """The ancestor walk that skips our own launcher stubs to reach the real
    client -- the fix's crux (``getppid()`` is the stub, not the client, on a
    venv built atop Store Python, where the stub outlives the client)."""

    def test_walks_past_our_launcher_chain_to_client(self, monkeypatch):
        # serve (us) -> venv launcher stub -> the client (claude). Both of our
        # processes carry the shim argv (biopb + stdio); the client carries none.
        client = _FakeProc([r"C:\claude.exe", "mcp"])
        stub = _FakeProc(
            [r"C:\.venv\python.exe", "-m", "biopb_mcp.mcp", "--transport", "stdio"],
            parent=client,
        )
        me = _FakeProc(
            [r"python3.10.exe", "-m", "biopb_mcp.mcp", "--transport", "stdio"],
            parent=stub,
        )
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(me))
        assert _shim._find_client_process() is client

    def test_none_when_chain_reaches_top(self, monkeypatch):
        # If every ancestor still looks like ours (no foreign client found), do
        # not guess -- return None so the watchdog does not arm.
        stub = _FakeProc(["python", "-m", "biopb_mcp.mcp", "--transport", "stdio"])
        me = _FakeProc(
            ["python", "-m", "biopb_mcp.mcp", "--transport", "stdio"], parent=stub
        )
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(me))
        assert _shim._find_client_process() is None

    def test_none_on_unreadable_ancestor(self, monkeypatch):
        class _Boom:
            def cmdline(self):
                raise RuntimeError("access denied")

            def parent(self):
                return None

        me = _FakeProc(["python", "-m", "biopb_mcp.mcp", "stdio"], parent=_Boom())
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(me))
        assert _shim._find_client_process() is None


class TestClientDeathWatchdog:
    """The Windows gap-filler: reap the owned session when the stdio client dies
    without the bridge seeing stdin EOF (a surviving client helper holding the
    stdin write handle). The reap/exit decision is tested OS-independently by
    driving ``_client_deathwatch`` directly; arming is Windows-only."""

    def test_installer_is_noop_off_windows(self):
        if os.name == "nt":
            pytest.skip("Windows arms a real watchdog thread")
        assert _shim._install_client_death_watchdog(lambda: None) is None

    def test_not_armed_when_client_unidentifiable(self, monkeypatch):
        monkeypatch.setattr(_shim, "_is_windows", lambda: True)
        monkeypatch.setattr(_shim, "_find_client_process", lambda: None)
        assert _shim._install_client_death_watchdog(lambda: None) is None

    def test_not_armed_when_client_handle_unopenable(self, monkeypatch):
        # A client we found but cannot open (returns None) must not arm --
        # never reap a live session off a baseline we could not establish.
        monkeypatch.setattr(_shim, "_is_windows", lambda: True)
        monkeypatch.setattr(_shim, "_find_client_process", lambda: _FakeProc([]))
        monkeypatch.setattr(_shim._winjob, "open_for_wait", lambda pid: None)
        assert _shim._install_client_death_watchdog(lambda: None) is None

    def test_arms_thread_when_client_found_and_openable(self, monkeypatch):
        import threading

        monkeypatch.setattr(_shim, "_is_windows", lambda: True)
        monkeypatch.setattr(_shim, "_find_client_process", lambda: _FakeProc([]))
        monkeypatch.setattr(_shim._winjob, "open_for_wait", lambda pid: "HANDLE")
        monkeypatch.setattr(_shim, "_client_deathwatch", lambda *a, **k: None)
        t = _shim._install_client_death_watchdog(lambda: None)
        assert isinstance(t, threading.Thread)
        t.join(timeout=5)

    def test_releases_and_exits_when_client_exits(self, monkeypatch):
        events = []
        monkeypatch.setattr(_shim._winjob, "wait_for_process", lambda h: True)
        monkeypatch.setattr(
            _shim.os, "_exit", lambda code: events.append(("exit", code))
        )
        _shim._client_deathwatch("HANDLE", 4242, lambda: events.append("reap"))
        # The client's exit must both release and exit.
        assert events == ["reap", ("exit", 0)]

    def test_does_not_release_on_wait_error(self, monkeypatch):
        # An undecided wait (error / not-signalled) must NOT end a shim whose
        # client may still be live -- stdin EOF remains the backstop.
        events = []
        monkeypatch.setattr(_shim._winjob, "wait_for_process", lambda h: False)
        monkeypatch.setattr(_shim.os, "_exit", lambda code: events.append("exit"))
        _shim._client_deathwatch("HANDLE", 4242, lambda: events.append("reap"))
        assert events == []


class _FakeRemote:
    """ClientSession stand-in recording forwarded calls."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments, progress_callback=None, *, meta=None):
        self.calls.append(("call_tool", name, arguments, meta))
        if name == "explodes":
            raise RuntimeError("kaboom")
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    async def read_resource(self, uri):
        self.calls.append(("read_resource", str(uri)))
        return types.ReadResourceResult(contents=[])

    async def list_tools(self):
        return types.ListToolsResult(
            tools=[types.Tool(name="from_session", inputSchema={"type": "object"})]
        )


class _FakeBinding:
    """The slice of ``_Binding`` that ``build_proxy`` talks to."""

    def __init__(self, remote=None, attach_text="attached", attach_error=None):
        self.session = remote
        self.attached = []
        self._text, self._error = attach_text, attach_error

    async def connect(self):
        if self.session is None:
            raise _shim.NotAttached("No session is attached. Call `attach` first.")
        return self.session

    async def listing(self):
        return "Live sessions:\n- s1: free, viewer"

    async def attach(self, selector, force=False):
        self.attached.append((selector, force))
        if self._error:
            raise _shim.AttachError(self._error)
        return self._text


def _call(name, arguments=None):
    return types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments or {}),
    )


class TestBuildProxy:
    def _call(self, handler, req):
        return anyio.run(lambda: handler(req))

    def _list(self, binding):
        app = _shim.build_proxy(binding, mcp._mcp_server)
        return self._call(
            app.request_handlers[types.ListToolsRequest],
            types.ListToolsRequest(method="tools/list"),
        )

    def test_unattached_lists_come_from_the_fastmcp_server_plus_attach(self):
        tools = self._list(_FakeBinding())
        names = {t.name for t in tools.root.tools}
        assert {"start_kernel", "execute_code", "attach"} <= names
        app = _shim.build_proxy(_FakeBinding(), mcp._mcp_server)
        resources = self._call(
            app.request_handlers[types.ListResourcesRequest],
            types.ListResourcesRequest(method="resources/list"),
        )
        assert [str(r.uri) for r in resources.root.resources] == ["docs://index"]

    def test_an_attached_tool_list_is_the_sessions_own(self):
        tools = self._list(_FakeBinding(_FakeRemote()))
        assert [t.name for t in tools.root.tools] == ["from_session"]

    def test_call_tool_forwards_name_and_args(self):
        remote = _FakeRemote()
        app = _shim.build_proxy(_FakeBinding(remote), mcp._mcp_server)
        result = self._call(
            app.request_handlers[types.CallToolRequest],
            _call("server_status", {"a": 1}),
        )
        assert remote.calls == [("call_tool", "server_status", {"a": 1}, None)]
        assert result.root.isError is False

    def test_call_tool_failure_becomes_tool_error_not_bridge_death(self):
        app = _shim.build_proxy(_FakeBinding(_FakeRemote()), mcp._mcp_server)
        result = self._call(
            app.request_handlers[types.CallToolRequest], _call("explodes")
        )
        assert result.root.isError is True
        assert "kaboom" in result.root.content[0].text

    def test_a_tool_before_attach_is_a_tool_error_that_says_to_attach(self):
        app = _shim.build_proxy(_FakeBinding(), mcp._mcp_server)
        result = self._call(
            app.request_handlers[types.CallToolRequest], _call("start_kernel")
        )
        assert result.root.isError is True
        assert "`attach`" in result.root.content[0].text

    def test_attach_without_a_session_lists_the_sessions(self):
        binding = _FakeBinding()
        app = _shim.build_proxy(binding, mcp._mcp_server)
        result = self._call(
            app.request_handlers[types.CallToolRequest], _call("attach")
        )
        assert result.root.isError is False
        assert "s1: free" in result.root.content[0].text
        assert binding.attached == []

    def test_attach_takes_the_named_session(self):
        binding = _FakeBinding(attach_text="Attached to session s1.")
        app = _shim.build_proxy(binding, mcp._mcp_server)
        result = self._call(
            app.request_handlers[types.CallToolRequest],
            _call("attach", {"session": " s1 ", "force": True}),
        )
        assert binding.attached == [("s1", True)]
        assert result.root.content[0].text == "Attached to session s1."

    def test_a_refused_attach_is_a_tool_error_with_the_reason(self):
        binding = _FakeBinding(attach_error="session s1 is held by its chat")
        app = _shim.build_proxy(binding, mcp._mcp_server)
        result = self._call(
            app.request_handlers[types.CallToolRequest],
            _call("attach", {"session": "s1"}),
        )
        assert result.root.isError is True
        assert "held by its chat" in result.root.content[0].text


class TestSessionListing:
    def _patch(self, monkeypatch, recs, statuses):
        monkeypatch.setattr(_shim._sessions, "list_sessions", lambda: recs)
        monkeypatch.setattr(_shim, "_probe", lambda rec: statuses[rec["session_id"]])

    def test_no_sessions_points_at_new(self, monkeypatch):
        self._patch(monkeypatch, [], {})
        assert "attach(session='new')" in _shim.session_listing()

    def test_each_session_says_whether_it_is_free_and_whether_it_has_a_window(
        self, monkeypatch
    ):
        recs = [
            {"session_id": "a", "port": 1},
            {"session_id": "b", "port": 2},
            {"session_id": "c", "port": 3},
        ]
        self._patch(
            monkeypatch,
            recs,
            {
                "a": {"lease": {"holder": None}, "viewer": True},
                "b": {"lease": {"holder": "chat"}, "viewer": False},
                "c": None,
            },
        )
        text = _shim.session_listing()
        assert "- a: free, viewer" in text
        assert "- b: held by chat, no viewer" in text
        assert "- c: unreachable" in text


class TestBinding:
    """Drives the real ``attach``/``connect``: the sessions and the connection
    are the fakes."""

    def _binding(self, monkeypatch, preselect=None, lease_status=200):
        env = SimpleNamespace(calls=[], lease_status=lease_status)

        def _call_session(base, method, path, body=None, timeout=None):
            env.calls.append((base, method, path, body))
            if path.endswith("/acquire"):
                if env.lease_status == 409:
                    return 409, {"holder": "chat", "age": 12.0}
                return env.lease_status, {"ok": True}
            if path.endswith("/renew"):
                return env.renew_status(), {}
            return 200, {}

        env.renew_status = lambda: 200

        @contextlib.asynccontextmanager
        async def _client(url):
            env.url = url
            yield "READ", "WRITE", None

        class _Session:
            def __init__(self, read, write):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                return SimpleNamespace(instructions="THE GUIDANCE")

        monkeypatch.setattr(_shim, "_call_session", _call_session)
        monkeypatch.setattr(_shim, "streamablehttp_client", _client)
        monkeypatch.setattr(_shim, "ClientSession", _Session)
        monkeypatch.setattr(_shim, "session_listing", lambda: "LISTING")
        monkeypatch.setattr(
            _shim._sessions,
            "resolve",
            lambda sid: (
                {"session_id": sid, "port": 7, "mcp_url": "http://127.0.0.1:7/mcp"}
                if sid in ("live", "managed")
                else None
            ),
        )
        # No control unless a test brings one: the real one would be whatever
        # answers on this machine's control port.
        env.control_up = False
        env.launch_answer = None
        env.launches = []
        monkeypatch.setattr(
            _shim._control_client, "ensure_control", lambda wait: env.control_up
        )

        def _launch(**kw):
            env.launches.append(kw)
            if isinstance(env.launch_answer, Exception):
                raise env.launch_answer
            return env.launch_answer

        monkeypatch.setattr(_shim._control_client, "launch_session", _launch)

        def use_control(up=True):
            env.control_up = up

        env.use_control = use_control
        return _shim._Binding(preselect=preselect), env

    def _drive(self, binding, *steps):
        results = []

        async def main():
            async with anyio.create_task_group() as tg:
                binding.task_group = tg
                for step in steps:
                    try:
                        results.append(await step())
                    except (_shim.AttachError, _shim.NotAttached) as e:
                        results.append(str(e))
                tg.cancel_scope.cancel()

        anyio.run(main)
        return results

    def test_nothing_happens_until_asked(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        self._drive(binding)
        binding.release()
        assert env.calls == []

    def test_a_request_before_attach_says_to_attach_and_lists_sessions(
        self, monkeypatch
    ):
        binding, _ = self._binding(monkeypatch)
        [text] = self._drive(binding, binding.connect)
        assert "`attach`" in text and "LISTING" in text

    def test_attaching_takes_the_lease(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        [text] = self._drive(binding, lambda: binding.attach("live"))
        assert "Attached to session live" in text and "THE GUIDANCE" in text
        base, method, path, body = env.calls[0]
        assert (base, path) == ("http://127.0.0.1:7", "/api/lease/acquire")
        assert body == {"token": binding.token, "force": False}
        assert env.url == "http://127.0.0.1:7/mcp"

    def test_the_shim_only_releases_a_session_it_attached_to(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        self._drive(binding, lambda: binding.attach("live"))
        binding.release()
        assert env.calls[-1][2] == "/api/lease/release"
        assert env.calls[-1][3] == {"token": binding.token}

    def test_a_held_session_is_refused_and_force_is_offered(self, monkeypatch):
        binding, env = self._binding(monkeypatch, lease_status=409)
        [text] = self._drive(binding, lambda: binding.attach("live"))
        assert "held by its chat" in text and "force=true" in text
        assert binding.session is None
        # Nothing was acquired, so nothing is released.
        assert not [c for c in env.calls if c[2].endswith("/release")]

    def test_force_is_passed_through(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        self._drive(binding, lambda: binding.attach("live", force=True))
        assert env.calls[0][3]["force"] is True

    def test_an_unknown_session_is_an_error_with_the_listing(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        [text] = self._drive(binding, lambda: binding.attach("gone"))
        assert "no live session 'gone'" in text and "LISTING" in text
        assert env.calls == []

    def test_new_asks_the_control_for_a_session_when_one_answers(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        env.use_control()
        env.launch_answer = {"state": "started", "session_id": "managed"}
        monkeypatch.setenv("DISPLAY", ":3")
        monkeypatch.setenv("LD_PRELOAD", "/tmp/not-sent.so")
        [text] = self._drive(binding, lambda: binding.attach("new"))
        assert "Attached to session managed" in text
        assert "keeps running after you disconnect" in text
        # The client's own display goes along, and nothing else of its
        # environment.
        assert env.launches[0]["display"].get("DISPLAY") == ":3"
        assert "LD_PRELOAD" not in env.launches[0]["display"]
        assert env.calls[0][:3] == ("http://127.0.0.1:7", "POST", "/api/lease/acquire")

    def test_a_control_launched_session_is_released_not_stopped(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        env.use_control()
        env.launch_answer = {"state": "started", "session_id": "managed"}
        self._drive(binding, lambda: binding.attach("new"))
        binding.release()
        assert env.calls[-1][2] == "/api/lease/release"

    def test_a_launch_that_failed_says_why_and_does_not_fall_back(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        env.use_control()
        env.launch_answer = {"state": "failed", "error": "exited 2", "log": "no napari"}
        [text] = self._drive(binding, lambda: binding.attach("new"))
        assert "exited 2" in text and "no napari" in text

    def test_a_slow_launch_points_at_the_listing(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        env.use_control()
        env.launch_answer = {"state": "starting"}
        [text] = self._drive(binding, lambda: binding.attach("new"))
        assert "still starting" in text

    def test_a_control_that_will_not_launch_is_an_error_not_a_fallback(
        self, monkeypatch
    ):
        # A session without the control has no data plane: attaching to one would
        # succeed and then fail on first use, hiding the cause.
        binding, env = self._binding(monkeypatch)
        env.use_control()
        env.launch_answer = OSError("refused")
        [text] = self._drive(binding, lambda: binding.attach("new"))
        assert "would not launch a session" in text and "refused" in text

    def test_no_control_is_an_error_that_says_how_to_start_one(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        env.use_control(up=False)
        [text] = self._drive(binding, lambda: binding.attach("new"))
        assert "no control answered" in text and "biopb control start" in text
        assert env.launches == []

    def test_a_preselected_session_attaches_on_first_use(self, monkeypatch):
        binding, env = self._binding(monkeypatch, preselect="new")
        env.use_control()
        env.launch_answer = {"state": "started", "session_id": "managed"}
        self._drive(binding, binding.connect)
        assert len(env.launches) == 1 and binding.session_id == "managed"

    def test_a_lost_lease_unbinds_and_the_next_request_says_why(self, monkeypatch):
        monkeypatch.setattr(_shim, "RENEW_INTERVAL", 0.01)
        binding, env = self._binding(monkeypatch)
        env.renew_status = lambda: 409

        async def lose():
            await binding.attach("live")
            for _ in range(200):
                if binding.session is None:
                    break
                await anyio.sleep(0.01)
            return binding.session

        async def later():
            return await binding.connect()

        gone, msg = self._drive(binding, lose, later)
        assert gone is None
        assert "Another holder took the session" in msg and "`attach`" in msg
        # It handed back what it held on the way out.
        assert env.calls[-1][2] == "/api/lease/release"

    def test_a_session_that_stops_answering_unbinds_after_three_misses(
        self, monkeypatch
    ):
        monkeypatch.setattr(_shim, "RENEW_INTERVAL", 0.01)
        binding, env = self._binding(monkeypatch)
        real = _shim._call_session

        def dead(base, method, path, body=None, timeout=None):
            if path.endswith("/renew"):
                raise OSError("connection refused")
            return real(base, method, path, body, timeout)

        async def lose():
            await binding.attach("live")
            monkeypatch.setattr(_shim, "_call_session", dead)
            for _ in range(200):
                if binding.session is None:
                    break
                await anyio.sleep(0.01)
            return binding.lost

        [lost] = self._drive(binding, lose)
        assert "stopped answering" in lost


# --------------------------------------------------------------------------- #
# End-to-end: platform-aware helpers
#
# The shim and its owned child use real OS process management, which differs by
# platform. biopb_mcp resolves every dir under ``Path.home()`` (``os.path.
# expanduser('~')``), which reads ``HOME`` on POSIX and ``USERPROFILE`` (then
# ``HOMEDRIVE``+``HOMEPATH``) on Windows — so isolation sets the right var. And
# liveness must not perturb the child: ``os.kill(pid, 0)`` is a probe on POSIX
# but on Windows *any* signal other than CTRL_* is an unconditional
# TerminateProcess, so Windows checks liveness with ``psutil`` instead. These
# helpers let one e2e cover all three OSes rather than skipping Windows (where
# the reap is the very thing #403 was about).
# --------------------------------------------------------------------------- #
def _home_env(tmp_path):
    """Env that redirects ``Path.home()`` to ``tmp_path`` (isolates config/log/pid)."""
    env = os.environ.copy()
    home = str(tmp_path)
    if os.name == "nt":
        env["USERPROFILE"] = home
        drive, tail = os.path.splitdrive(home)
        env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
    else:
        env["HOME"] = home
    # The XDG base dirs override HOME in _locations; drop any inherited
    # values so the HOME-based defaults (state/config/data under tmp_path) apply.
    for var in ("BIOPB_STATE_HOME", "BIOPB_CONFIG_HOME", "BIOPB_DATA_HOME"):
        env.pop(var, None)
    return env


def _pid_alive(pid):
    """Whether ``pid`` names a live process — WITHOUT killing or perturbing it."""
    if os.name == "nt":
        import psutil  # a biopb-mcp dep; only needed on the Windows leg

        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours


def _await_dead(pid, timeout):
    """Poll until ``pid`` is gone (reaping is asynchronous to the shim's exit)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"owned child {pid} still alive after {timeout:.0f}s")


def _force_kill(pid):
    """Best-effort teardown of a still-running child (SIGKILL / TerminateProcess)."""
    try:
        if os.name == "nt":
            import psutil

            psutil.Process(pid).kill()
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _extract(pattern, text, what):
    m = re.search(pattern, text)
    assert m, f"could not find {what} in daemon log:\n{text[-2000:]}"
    return m.group(1)


def _stop_control(env):
    """Stop the control a test's shim started, on the test's own port."""
    biopb = os.path.join(
        os.path.dirname(sys.executable), "biopb.exe" if os.name == "nt" else "biopb"
    )
    subprocess.run(
        [biopb, "control", "stop"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )


def _rpc_init(shim):
    """Handshake a stdio shim subprocess."""
    for msg in (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        },
        None,
    ):
        shim.stdin.write(
            (
                json.dumps(
                    msg or {"jsonrpc": "2.0", "method": "notifications/initialized"}
                )
                + "\n"
            ).encode()
        )
        shim.stdin.flush()
        if msg is not None:
            json.loads(shim.stdout.readline())


def _rpc_call(shim, id_, name, arguments=None):
    """A tool call over a shim's stdio: ``(result, text)``, notifications skipped."""
    shim.stdin.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": id_,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}},
                }
            )
            + "\n"
        ).encode()
    )
    shim.stdin.flush()
    while True:
        msg = json.loads(shim.stdout.readline())
        if msg.get("id") == id_:
            break
    result = msg["result"]
    return result, result["content"][0]["text"]


class TestControlLaunchedSession:
    """The real control launches the session an agent attaches to. It outlives
    the agent, the next agent can attach to it, and the user stops it from the
    dashboard's verb."""

    def test_a_session_outlives_its_agent_until_the_user_stops_it(self, tmp_path):
        # biopb-control is not a dependency of biopb-mcp; where it is not
        # installed there is no control for the shim to start.
        pytest.importorskip("biopb_control")
        env = _home_env(tmp_path)
        env.pop("BIOPB_TENSOR_URL", None)
        # A control of this test's own, on a port that is not the user's.
        control_port = _free_port()
        env["BIOPB_CONTROL_PORT"] = str(control_port)
        reg_dir = tmp_path / ".local/state/biopb/sessions"

        def start_shim():
            return subprocess.Popen(
                [sys.executable, "-m", "biopb_mcp.mcp", "--transport", "stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )

        first = start_shim()
        second = None
        pid = None
        try:
            _rpc_init(first)
            result, text = _rpc_call(first, 2, "attach", {"session": "new"})
            assert result.get("isError") is not True, text
            assert "keeps running after you disconnect" in text, text

            records = list(reg_dir.glob("*.json"))
            assert len(records) == 1, records
            rec = json.loads(records[0].read_text())
            pid, session_id = rec["pid"], rec["session_id"]
            assert rec["mode"] == "durable"
            status, _ = _rpc_call(first, 3, "server_status")
            assert status.get("isError") is not True, status

            # The agent goes. The session is no child of its shim, so it stays.
            first.stdin.close()
            assert first.wait(timeout=40) == 0
            time.sleep(1.0)
            assert _pid_alive(pid)

            # The next agent attaches to the same session: its lease was
            # released, and its kernel is still there.
            second = start_shim()
            _rpc_init(second)
            _, listing = _rpc_call(second, 2, "attach")
            assert f"{session_id}: free" in listing, listing
            result, text = _rpc_call(second, 3, "attach", {"session": session_id})
            assert result.get("isError") is not True, text
            status, _ = _rpc_call(second, 4, "server_status")
            assert status.get("isError") is not True, status

            # The user stops it from the dashboard's verb, through the control.
            import urllib.request

            req = urllib.request.Request(
                f"http://127.0.0.1:{control_port}/session/{session_id}/api/shutdown",
                data=b"",
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                assert json.loads(resp.read())["stopping"] is True
            _await_dead(pid, timeout=30)
            deadline = time.monotonic() + 10
            while list(reg_dir.glob("*.json")) and time.monotonic() < deadline:
                time.sleep(0.2)
            assert list(reg_dir.glob("*.json")) == []
        finally:
            for shim in (first, second):
                if shim is not None and shim.poll() is None:
                    shim.kill()
            if pid is not None:
                _force_kill(pid)
            _stop_control(env)


class TestServe:
    def test_releases_the_lease_on_the_way_out(self, monkeypatch):
        """The signal handler and the watchdog are armed before anything is
        attached, over the binding, and serve releases whatever it holds when the
        bridge ends."""
        armed, released = [], []

        async def _fake_serve(binding):
            binding._base = "http://127.0.0.1:9"

        monkeypatch.setattr(_shim, "_install_release_on_signal", armed.append)
        monkeypatch.setattr(_shim, "_install_client_death_watchdog", armed.append)
        monkeypatch.setattr(_shim, "_serve_stdio", _fake_serve)
        monkeypatch.setattr(
            _shim,
            "_call_session",
            lambda base, method, path, body=None, timeout=None: released.append(
                (base, path)
            ),
        )

        _shim.serve()
        assert len(armed) == 2
        assert released == [("http://127.0.0.1:9", "/api/lease/release")]

    def test_nothing_attached_releases_nothing(self, monkeypatch):
        calls = []
        monkeypatch.setattr(_shim, "_install_release_on_signal", lambda r: None)
        monkeypatch.setattr(_shim, "_install_client_death_watchdog", lambda r: None)

        async def _noop(binding):
            pass

        monkeypatch.setattr(_shim, "_serve_stdio", _noop)
        monkeypatch.setattr(_shim, "_call_session", lambda *a, **k: calls.append(a))
        _shim.serve()
        assert calls == []
