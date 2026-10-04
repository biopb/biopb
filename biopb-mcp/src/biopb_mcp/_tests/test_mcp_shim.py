"""Tests for the stdio bridge ("shim") to its owned http session child.

Unit tests cover the session-spawn logic (dynamic-port handoff, env
inheritance, startup/timeout), the port-report file parsing, the reap, the
lazy start, the shipped surface snapshot, and the request forwarding of the
vendored proxy. One end-to-end test runs the real thing: ``biopb-mcp
--transport stdio`` as a subprocess, which must answer the handshake and the
listings with no child, spawn its own http session child on the first tool
call, and **reap** that child when the client hangs up.
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
from biopb.lifecycle.owned_child import OwnedChild
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


def _port_listening(port, timeout=0.5):
    """Whether something accepts TCP connections on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _cfg(**transport):
    return {"transport": transport}


# A stand-in session child: reads its port-report file from the env, binds and
# listens on a dynamic port, publishes it atomically (temp + os.replace), as the
# real child does, then accepts connections so a test's own probe succeeds.
_FAKE_CHILD = (
    "import os, socket, sys, threading, time\n"
    "pf = os.environ['BIOPB_PORT_REPORT_FILE']\n"
    "s = socket.socket()\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', 0))\n"
    "s.listen(16)\n"
    "port = s.getsockname()[1]\n"
    "tmp = pf + '.tmp'\n"
    "open(tmp, 'w').write(str(port))\n"
    "os.replace(tmp, pf)\n"
    "def _serve():\n"
    "    while True:\n"
    "        try:\n"
    "            conn, _ = s.accept(); conn.close()\n"
    "        except OSError:\n"
    "            break\n"
    "threading.Thread(target=_serve, daemon=True).start()\n"
    "time.sleep(30)\n"
)


class TestSessionCommand:
    def test_module_reentry_binds_dynamic_port(self):
        cmd = _shim._session_command()
        assert cmd[:3] == [sys.executable, "-m", "biopb_mcp.mcp"]
        # Dynamic port: the child reports the OS-assigned one back via a file.
        assert cmd[3:] == ["--transport", "http", "--port", "0"]


class TestReadPortFile:
    def test_valid_port(self, tmp_path):
        p = tmp_path / "port.txt"
        p.write_text("8899")
        assert _shim._read_port_file(str(p)) == 8899

    def test_missing_file(self, tmp_path):
        assert _shim._read_port_file(str(tmp_path / "nope.txt")) is None

    def test_empty_file_not_yet_reported(self, tmp_path):
        p = tmp_path / "port.txt"
        p.write_text("")
        assert _shim._read_port_file(str(p)) is None

    def test_garbage_is_none(self, tmp_path):
        p = tmp_path / "port.txt"
        p.write_text("not-a-port")
        assert _shim._read_port_file(str(p)) is None

    def test_nonpositive_is_none(self, tmp_path):
        p = tmp_path / "port.txt"
        p.write_text("0")
        assert _shim._read_port_file(str(p)) is None


class TestSessionLogPath:
    def test_default_is_per_session_under_sessions_dir(self, tmp_path, monkeypatch):
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "get_log_dir", lambda: tmp_path)
        p = _shim._session_log_path(_cfg(), "20260101-000000-42")
        assert p == str(tmp_path / "sessions" / "20260101-000000-42.log")

    def test_kernel_log_override_forces_single_file(self, tmp_path):
        override = tmp_path / "one.log"
        p = _shim._session_log_path(_cfg(kernel_log=str(override)), "sid")
        assert p == str(override)


class TestPruneSessionLogs:
    def test_keeps_newest_n_by_mtime(self, tmp_path, monkeypatch):
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "get_log_dir", lambda: tmp_path)
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        for i in range(7):
            p = sessions / f"s{i}.log"
            p.write_text("x")
            os.utime(p, (1000 + i, 1000 + i))  # ascending mtime: s6 newest
        _shim._prune_session_logs(3)
        assert sorted(q.name for q in sessions.glob("*.log")) == [
            "s4.log",
            "s5.log",
            "s6.log",
        ]

    def test_missing_dir_is_noop(self, tmp_path, monkeypatch):
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "get_log_dir", lambda: tmp_path / "nope")
        _shim._prune_session_logs(5)  # must not raise


class TestSpawnSession:
    def test_reports_port_and_waits_until_listening(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _shim, "_session_command", lambda: [sys.executable, "-c", _FAKE_CHILD]
        )
        cfg = _cfg(kernel_log=str(tmp_path / "d.log"))
        child, url, session_id = _shim.spawn_session(cfg, timeout=15)
        try:
            m = re.match(r"http://127\.0\.0\.1:(\d+)/mcp$", url)
            assert m, url
            port = int(m.group(1))
            assert _port_listening(port) is True
            assert child.poll() is None  # still running while we bridge
            if os.name == "nt":
                assert child.job is not None  # Windows: a kill-on-close Job Object
            else:
                assert child.job is None  # POSIX: reaped via the group, not a job
        finally:
            _shim._reap_session(child)
        assert child.poll() is not None  # reaped on the way out

    def test_inherits_live_env_and_wires_dynamic_port(self, tmp_path, monkeypatch):
        # The #98 fix: the child inherits THIS shim's current environment, so a
        # live DISPLAY flows through instead of a value frozen into a daemon.
        monkeypatch.setenv("DISPLAY", ":test-99")
        captured = {}

        class _FakeProc:
            pid = 4242
            returncode = None

            def poll(self):
                return None

        def _fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs["env"]
            # Stand in for the child publishing its port.
            with open(kwargs["env"]["BIOPB_PORT_REPORT_FILE"], "w") as f:
                f.write("54321")
            return _FakeProc()

        from biopb.lifecycle import owned_child

        monkeypatch.setattr(owned_child.subprocess, "Popen", _fake_popen)

        spawned = []
        child, url, session_id = _shim.spawn_session(
            _cfg(kernel_log=str(tmp_path / "d.log")),
            timeout=5,
            on_spawned=lambda *a: spawned.append(a),
        )
        assert url == "http://127.0.0.1:54321/mcp"
        assert spawned == [(child, session_id)]
        # The child registers itself, under the id minted here.
        assert captured["env"]["BIOPB_MCP_SESSION_ID"] == session_id
        assert captured["env"]["DISPLAY"] == ":test-99"
        assert captured["env"]["BIOPB_PORT_REPORT_FILE"]  # port channel wired
        # session log path handed to the child (here the kernel_log override).
        assert captured["env"]["BIOPB_MCP_SESSION_LOG"] == str(tmp_path / "d.log")
        assert captured["cmd"][-2:] == ["--port", "0"]  # dynamic port

    def test_raises_when_child_dies_before_reporting(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _shim,
            "_session_command",
            lambda: [sys.executable, "-c", "raise SystemExit(3)"],
        )
        cfg = _cfg(kernel_log=str(tmp_path / "d.log"))
        with pytest.raises(RuntimeError, match="before reporting its port"):
            _shim.spawn_session(cfg, timeout=5)


class TestReapSession:
    def test_reaps_running_child(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        assert proc.poll() is None
        _shim._reap_session(OwnedChild.adopt(proc))
        assert proc.poll() is not None

    def test_drops_the_record_a_killed_child_could_not(self, tmp_path, monkeypatch):
        # Windows kills the child outright, so its _shutdown never runs.
        from biopb import _sessions

        monkeypatch.setenv("BIOPB_SESSIONS_DIR", str(tmp_path / "sessions"))
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        _sessions.register("20260101-000000-1", port=1, pid=proc.pid)
        _shim._reap_session(OwnedChild.adopt(proc), "20260101-000000-1")
        assert _sessions.read_session("20260101-000000-1") is None

    def test_idempotent_on_dead_child(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        # Both calls must be no-op-safe on an already-dead child.
        child = OwnedChild.adopt(proc)
        _shim._reap_session(child)
        _shim._reap_session(child)
        assert proc.poll() is not None


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

    def test_reaps_and_exits_when_client_exits(self, monkeypatch):
        events = []
        monkeypatch.setattr(_shim._winjob, "wait_for_process", lambda h: True)
        monkeypatch.setattr(
            _shim.os, "_exit", lambda code: events.append(("exit", code))
        )
        _shim._client_deathwatch("HANDLE", 4242, lambda: events.append("reap"))
        # The client's exit must both reap and exit.
        assert events == ["reap", ("exit", 0)]

    def test_does_not_reap_on_wait_error(self, monkeypatch):
        # An undecided wait (error / not-signalled) must NOT tear down a session
        # that may still be live -- the other teardown paths remain the backstop.
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

    def _binding(self, monkeypatch, preselect=None, fail=0, lease_status=200):
        env = SimpleNamespace(spawns=[], reaped=[], calls=[], lease_status=lease_status)

        class _Child:
            pid = 4242

        def _spawn(config, on_spawned=None):
            child = _Child()
            env.spawns.append(child)
            on_spawned(child, "newsid")
            if len(env.spawns) <= fail:
                raise RuntimeError(f"spawn {len(env.spawns)} failed")
            return child, "http://127.0.0.1:1/mcp", "newsid"

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

        monkeypatch.setattr(_shim, "spawn_session", _spawn)
        monkeypatch.setattr(_shim, "_call_session", _call_session)
        monkeypatch.setattr(_shim, "streamablehttp_client", _client)
        monkeypatch.setattr(_shim, "ClientSession", _Session)
        monkeypatch.setattr(
            _shim, "_reap_session", lambda child, sid: env.reaped.append(child)
        )
        monkeypatch.setattr(_shim, "session_listing", lambda: "LISTING")
        monkeypatch.setattr(
            _shim._sessions,
            "resolve",
            lambda sid: (
                {"session_id": sid, "port": 7, "mcp_url": "http://127.0.0.1:7/mcp"}
                if sid == "live"
                else None
            ),
        )
        monkeypatch.setattr(
            _shim._control_client, "start_control_detached", lambda: True
        )
        return _shim._Binding(config=object(), preselect=preselect), env

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
        binding.reap()
        assert env.spawns == [] and env.reaped == [] and env.calls == []

    def test_a_request_before_attach_says_to_attach_and_lists_sessions(
        self, monkeypatch
    ):
        binding, _ = self._binding(monkeypatch)
        [text] = self._drive(binding, binding.connect)
        assert "`attach`" in text and "LISTING" in text

    def test_attaching_takes_the_lease_and_spawns_nothing(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        [text] = self._drive(binding, lambda: binding.attach("live"))
        assert "Attached to session live" in text and "THE GUIDANCE" in text
        assert env.spawns == []
        base, method, path, body = env.calls[0]
        assert (base, path) == ("http://127.0.0.1:7", "/api/lease/acquire")
        assert body == {"token": binding.token, "force": False}
        assert env.url == "http://127.0.0.1:7/mcp"

    def test_the_shim_only_releases_a_session_it_attached_to(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        self._drive(binding, lambda: binding.attach("live"))
        binding.reap()
        assert env.reaped == []  # not ours to stop
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

    def test_new_spawns_a_session_the_shim_owns_and_reaps_it(self, monkeypatch):
        binding, env = self._binding(monkeypatch)
        [text] = self._drive(binding, lambda: binding.attach("new"))
        assert "Attached to session newsid" in text
        assert len(env.spawns) == 1
        assert env.calls[0][:3] == ("http://127.0.0.1:1", "POST", "/api/lease/acquire")
        binding.reap()
        assert env.reaped == env.spawns

    def test_a_failed_new_is_reaped_and_can_be_retried(self, monkeypatch):
        binding, env = self._binding(monkeypatch, fail=1)
        first, second = self._drive(
            binding, lambda: binding.attach("new"), lambda: binding.attach("new")
        )
        assert "could not attach" in first and "spawn 1 failed" in first
        assert "Attached to session" in second
        assert len(env.spawns) == 2
        assert env.reaped == env.spawns[:1]

    def test_a_preselected_session_attaches_on_first_use(self, monkeypatch):
        binding, env = self._binding(monkeypatch, preselect="new")
        self._drive(binding, binding.connect)
        assert len(env.spawns) == 1

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


class TestEndToEnd:
    """The real thing (all OSes): the shim answers the handshake alone, spawns its
    OWN session child on the first tool call, bridges it, and reaps that child
    when the client disconnects."""

    def test_session_is_private_and_reaped_on_disconnect(self, tmp_path):
        env = _home_env(tmp_path)  # isolate config + log dirs, per platform

        shim = subprocess.Popen(
            [sys.executable, "-m", "biopb_mcp.mcp", "--transport", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        child_pid = None
        try:

            def send(obj):
                shim.stdin.write((json.dumps(obj) + "\n").encode())
                shim.stdin.flush()

            send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "t", "version": "0"},
                    },
                }
            )
            init = json.loads(shim.stdout.readline())["result"]
            # Identity and instructions come from the shim itself.
            assert init["serverInfo"]["name"] == "biopb-mcp"
            assert "execute_code" in (init.get("instructions") or "")

            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            tools = json.loads(shim.stdout.readline())["result"]["tools"]
            assert {"start_kernel", "execute_code", "attach"} <= {
                t["name"] for t in tools
            }
            assert "attach" in init["instructions"].split("\n")[0]

            # Nothing has needed a session yet, so there is none.
            sessions_dir = tmp_path / ".local/state/biopb/mcp/sessions"
            reg_dir = tmp_path / ".local/state/biopb/sessions"
            assert list(sessions_dir.glob("*.log")) == []
            assert list(reg_dir.glob("*.json")) == []

            def call(shim, id_, name, arguments=None):
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
                while True:  # skip notifications (tools/list_changed)
                    msg = json.loads(shim.stdout.readline())
                    if msg.get("id") == id_:
                        break
                result = msg["result"]
                return result, result["content"][0]["text"]

            # Unattached, a tool is an error that says to attach -- and starts
            # nothing.
            result, text = call(shim, 3, "server_status")
            assert result["isError"] is True and "`attach`" in text
            assert list(sessions_dir.glob("*.log")) == []

            # `attach new` starts a session this shim owns.
            result, text = call(shim, 4, "attach", {"session": "new"})
            assert result.get("isError") is not True, text
            assert "Attached to session" in text
            status, _ = call(shim, 5, "server_status")
            assert status.get("isError") is not True, status

            # A second client sees that session as held, and cannot take it.
            other = subprocess.Popen(
                [sys.executable, "-m", "biopb_mcp.mcp", "--transport", "stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                other.stdin.write(
                    (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "initialize",
                                "params": {
                                    "protocolVersion": "2025-03-26",
                                    "capabilities": {},
                                    "clientInfo": {"name": "t2", "version": "0"},
                                },
                            }
                        )
                        + "\n"
                    ).encode()
                )
                other.stdin.flush()
                json.loads(other.stdout.readline())
                other.stdin.write(
                    b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n'
                )
                _, listing = call(other, 2, "attach")
                session_id = next(reg_dir.glob("*.json")).stem
                assert f"{session_id}: held by agent" in listing, listing
                result, text = call(other, 3, "attach", {"session": session_id})
                assert result["isError"] is True and "held by its agent" in text
            finally:
                other.stdin.close()
                other.wait(timeout=30)

            # The owned child logs its PID (uvicorn's "Started server process
            # [pid]") and its dynamic listen URL (_server.run) to its own
            # per-session logfile under log/sessions/ (NOT the shared
            # mcp-server.log — that separation is the session-log feature).
            session_logs = list(sessions_dir.glob("*.log"))
            assert len(session_logs) == 1, session_logs
            log = session_logs[0].read_bytes().decode(errors="replace")
            child_pid = int(
                _extract(r"Started server process \[(\d+)\]", log, "child pid")
            )
            port = int(_extract(r"http://127\.0\.0\.1:(\d+)/mcp", log, "listen port"))
            assert _pid_alive(child_pid)  # up now
            assert _port_listening(port) is True

            # The child registered itself for control discovery (the shared
            # biopb state tree, isolated here via HOME), under its own pid.
            records = list(reg_dir.glob("*.json"))
            assert len(records) == 1, records
            rec = json.loads(records[0].read_text())
            assert rec["port"] == port
            assert rec["pid"] == child_pid

            # Client hangs up: the shim must exit AND reap its private child
            # (the shared daemon used to survive — that is exactly what changed).
            shim.stdin.close()
            assert shim.wait(timeout=40) == 0
            _await_dead(child_pid, timeout=20)
            assert _port_listening(port) is False  # server truly gone
            # No routing ghost, on every OS: the reap drops the record even
            # where it kills the child outright (Windows).
            assert list(reg_dir.glob("*.json")) == []
        finally:
            if shim.poll() is None:
                shim.kill()
            if child_pid is not None:
                _force_kill(child_pid)


class TestServe:
    def test_reaps_the_child_on_the_way_out(self, monkeypatch):
        """The reaper and watchdog are armed before any child exists, over the
        binding, and serve reaps whatever it holds when the bridge ends."""
        armed, reaped = [], []

        async def _fake_serve(binding):
            binding._own("CHILD", "sid")

        monkeypatch.setattr(_shim, "_install_shim_reaper", armed.append)
        monkeypatch.setattr(_shim, "_install_client_death_watchdog", armed.append)
        monkeypatch.setattr(_shim, "_serve_stdio", _fake_serve)
        monkeypatch.setattr(
            _shim, "_reap_session", lambda child, sid: reaped.append((child, sid))
        )

        _shim.serve(object())
        assert len(armed) == 2
        assert reaped == [("CHILD", "sid")]
