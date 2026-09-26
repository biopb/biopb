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

import anyio
import pytest
from biopb._lifecycle.owned_child import OwnedChild
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

        from biopb._lifecycle import owned_child

        monkeypatch.setattr(owned_child.subprocess, "Popen", _fake_popen)

        spawned = []
        child, url, session_id = _shim.spawn_session(
            _cfg(kernel_log=str(tmp_path / "d.log")),
            timeout=5,
            on_spawned=spawned.append,
        )
        assert url == "http://127.0.0.1:54321/mcp"
        assert spawned == [child]
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


def _connect_to(remote):
    async def connect():
        return remote

    return connect


async def _no_child():
    raise AssertionError("a listing must not start the session child")


class TestBuildProxy:
    def _call(self, handler, req):
        return anyio.run(lambda: handler(req))

    def test_lists_come_from_the_fastmcp_server_not_the_child(self):
        app = _shim.build_proxy(_no_child, mcp._mcp_server)
        tools = self._call(
            app.request_handlers[types.ListToolsRequest],
            types.ListToolsRequest(method="tools/list"),
        )
        assert {"start_kernel", "execute_code"} <= {t.name for t in tools.root.tools}
        resources = self._call(
            app.request_handlers[types.ListResourcesRequest],
            types.ListResourcesRequest(method="resources/list"),
        )
        assert [str(r.uri) for r in resources.root.resources] == ["docs://index"]

    def test_call_tool_forwards_name_and_args(self):
        remote = _FakeRemote()
        app = _shim.build_proxy(_connect_to(remote), mcp._mcp_server)
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="server_status", arguments={"a": 1}
            ),
        )
        result = self._call(app.request_handlers[types.CallToolRequest], req)
        assert remote.calls == [("call_tool", "server_status", {"a": 1}, None)]
        assert result.root.isError is False

    def test_call_tool_failure_becomes_tool_error_not_bridge_death(self):
        remote = _FakeRemote()
        app = _shim.build_proxy(_connect_to(remote), mcp._mcp_server)
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="explodes", arguments={}),
        )
        result = self._call(app.request_handlers[types.CallToolRequest], req)
        assert result.root.isError is True
        assert "kaboom" in result.root.content[0].text

    def test_a_child_that_cannot_start_is_a_tool_error(self):
        async def connect():
            raise RuntimeError("the biopb-mcp session did not start: no port")

        app = _shim.build_proxy(connect, mcp._mcp_server)
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="start_kernel", arguments={}),
        )
        result = self._call(app.request_handlers[types.CallToolRequest], req)
        assert result.root.isError is True
        assert "did not start: no port" in result.root.content[0].text


class TestLazySession:
    """Drives the real ``get()``: the spawn and the connection are the fakes."""

    def _lazy(self, monkeypatch, fail=0):
        spawns, reaped = [], []

        class _Child:
            pid = 4242

        def _spawn(config, on_spawned=None):
            child = _Child()
            spawns.append(child)
            on_spawned(child)
            if len(spawns) <= fail:
                raise RuntimeError(f"spawn {len(spawns)} failed")
            return child, "http://127.0.0.1:1/mcp", "sid"

        @contextlib.asynccontextmanager
        async def _client(url):
            yield "READ", "WRITE", None

        class _Session:
            def __init__(self, read, write):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                pass

        monkeypatch.setattr(_shim, "spawn_session", _spawn)
        monkeypatch.setattr(_shim, "streamablehttp_client", _client)
        monkeypatch.setattr(_shim, "ClientSession", _Session)
        monkeypatch.setattr(_shim, "_reap_session", reaped.append)
        monkeypatch.setattr(
            _shim._control_client, "start_control_detached", lambda: True
        )
        return _shim._LazySession(config=object()), spawns, reaped

    def _drive(self, lazy, *steps):
        results = []

        async def main():
            async with anyio.create_task_group() as tg:
                lazy.task_group = tg
                for step in steps:
                    try:
                        results.append(await step())
                    except RuntimeError as e:
                        results.append(str(e))
                tg.cancel_scope.cancel()

        anyio.run(main)
        return results

    def test_nothing_spawns_until_asked(self, monkeypatch):
        lazy, spawns, reaped = self._lazy(monkeypatch)
        self._drive(lazy)
        lazy.reap()  # no child: a no-op
        assert spawns == [] and reaped == []

    def test_concurrent_requests_share_one_child(self, monkeypatch):
        lazy, spawns, _ = self._lazy(monkeypatch)

        async def both():
            out = []

            async def one():
                out.append(await lazy.get())

            async with anyio.create_task_group() as tg:
                tg.start_soon(one)
                tg.start_soon(one)
            return out

        [(first, second)] = self._drive(lazy, both)
        assert first is second
        assert len(spawns) == 1

    def test_a_failed_start_is_reported_then_retried(self, monkeypatch):
        lazy, spawns, reaped = self._lazy(monkeypatch, fail=1)
        first, second = self._drive(lazy, lazy.get, lazy.get)
        assert "did not start: spawn 1 failed" in first
        assert not isinstance(second, str)
        assert len(spawns) == 2
        assert reaped == spawns[:1]  # the failed attempt's child


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
            assert {"start_kernel", "execute_code"} <= {t["name"] for t in tools}

            # Nothing has needed the child yet, so there is none.
            sessions_dir = tmp_path / ".local/state/biopb/mcp/sessions"
            reg_dir = tmp_path / ".local/state/biopb/sessions"
            assert list(sessions_dir.glob("*.log")) == []
            assert list(reg_dir.glob("*.json")) == []

            # The first tool call starts it.
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "server_status", "arguments": {}},
                }
            )
            status = json.loads(shim.stdout.readline())["result"]
            assert status.get("isError") is not True, status

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
            # No routing ghost: POSIX reaps with SIGTERM, and the child drops
            # its own record; Windows force-kills the tree, leaving a record
            # whose dead pid the registry prunes on the next read.
            if os.name == "nt":
                for leftover in reg_dir.glob("*.json"):
                    assert not _pid_alive(json.loads(leftover.read_text())["pid"])
            else:
                assert list(reg_dir.glob("*.json")) == []
        finally:
            if shim.poll() is None:
                shim.kill()
            if child_pid is not None:
                _force_kill(child_pid)


class TestServe:
    def test_reaps_the_child_on_the_way_out(self, monkeypatch):
        """The reaper and watchdog are armed before any child exists, over the
        lazy session, and serve reaps whatever it holds when the bridge ends."""
        armed, reaped = [], []

        async def _fake_serve(lazy):
            lazy._own("CHILD")

        monkeypatch.setattr(_shim, "_install_shim_reaper", armed.append)
        monkeypatch.setattr(_shim, "_install_client_death_watchdog", armed.append)
        monkeypatch.setattr(_shim, "_serve_stdio", _fake_serve)
        monkeypatch.setattr(_shim, "_reap_session", reaped.append)

        _shim.serve(object())
        assert len(armed) == 2
        assert reaped == ["CHILD"]
