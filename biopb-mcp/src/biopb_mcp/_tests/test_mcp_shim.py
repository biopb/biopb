"""Tests for the stdio bridge ("shim") to its owned http session child.

Unit tests cover the session-spawn logic (dynamic-port handoff, env
inheritance, startup/timeout), the port-report file parsing, the reap, the
faithful initialize replay (the `instructions` field above all — losing it is
the defect that disqualified delegating to mcp-proxy), and the request
forwarding of the vendored proxy. One end-to-end test runs the real thing:
``biopb-mcp --transport stdio`` as a subprocess, which must spawn its own http
session child, bridge a full JSON-RPC session, and — de-daemonization Layer 1 —
**reap** that child when the client hangs up (the shared daemon used to survive).
"""

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
from biopb import _config_sessions
from mcp import types

from biopb_mcp.mcp import _shim


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(**transport):
    return {"transport": transport}


# A stand-in session child: reads its port-report file from the env, binds a
# dynamic port, publishes it atomically (temp + os.replace, as the real child
# does), then accepts connections so the readiness probe succeeds. It must
# accept, not merely listen(): the probe opens a fresh TCP connection on every
# poll, and an un-drained backlog makes a later probe fail on macOS's stricter
# socket stack (the real uvicorn child accepts, so this is a fixture artifact).
_FAKE_CHILD = (
    "import os, socket, sys, threading, time\n"
    "pf = os.environ['BIOPB_PORT_REPORT_FILE']\n"
    "s = socket.socket()\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', 0))\n"
    "port = s.getsockname()[1]\n"
    "tmp = pf + '.tmp'\n"
    "open(tmp, 'w').write(str(port))\n"
    "os.replace(tmp, pf)\n"
    "s.listen(16)\n"
    "def _serve():\n"
    "    while True:\n"
    "        try:\n"
    "            conn, _ = s.accept(); conn.close()\n"
    "        except OSError:\n"
    "            break\n"
    "threading.Thread(target=_serve, daemon=True).start()\n"
    "time.sleep(30)\n"
)


class TestPortListening:
    def test_true_for_listening_socket(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            assert _shim._port_listening(s.getsockname()[1]) is True

    def test_false_for_closed_port(self):
        assert _shim._port_listening(_free_port()) is False


class TestSessionCommand:
    def test_module_reentry_binds_dynamic_port(self):
        cmd = _shim._session_command()
        assert cmd[:3] == [sys.executable, "-m", "biopb_mcp.mcp"]
        # Dynamic port: the child reports the OS-assigned one back via a file.
        assert cmd[3:] == ["--transport", "http", "--port", "0"]

    def test_frozen_build_calls_its_own_binary(self, monkeypatch):
        # PyInstaller: sys.executable IS the launcher; no module tree to -m.
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        cmd = _shim._session_command()
        assert cmd == [sys.executable, "--transport", "http", "--port", "0"]


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


class TestOpenSessionLog:
    def test_writes_to_path_creating_parent(self, tmp_path):
        path = tmp_path / "sub" / "sess.log"  # parent does not exist yet
        f = _shim._open_session_log(str(path))
        try:
            f.write(b"hello\n")
        finally:
            f.close()
        assert path.read_bytes() == b"hello\n"

    def test_falls_back_to_stderr_when_unopenable(self, tmp_path):
        # Parent path is a FILE, so mkdir(parents=True) fails regardless of
        # privilege (works even when tests run as root).
        blocker = tmp_path / "afile"
        blocker.write_text("x")
        f = _shim._open_session_log(str(blocker / "sub" / "x.log"))
        assert f is getattr(sys.stderr, "buffer", sys.stderr)


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
        proc, url, job, session_id = _shim.spawn_session(cfg, timeout=15)
        try:
            m = re.match(r"http://127\.0\.0\.1:(\d+)/mcp$", url)
            assert m, url
            port = int(m.group(1))
            assert _shim._port_listening(port) is True
            assert proc.poll() is None  # still running while we bridge
            if os.name == "nt":
                assert job is not None  # Windows: a kill-on-close Job Object
            else:
                assert job is None  # POSIX: reaped via the process group, not a job
            # Self-registered so the control can discover + proxy to it.
            rec = _config_sessions.read_session(session_id)
            assert rec is not None
            assert rec["port"] == port and rec["pid"] == proc.pid
            assert rec["mcp_url"] == url
        finally:
            _shim._reap_session(proc, job, session_id)
        assert proc.poll() is not None  # reaped on the way out
        # Reap de-registers, so the record is gone.
        assert _config_sessions.read_session(session_id) is None

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

        monkeypatch.setattr(_shim.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(_shim, "_await_listening", lambda *a, **k: None)

        proc, url, job, session_id = _shim.spawn_session(
            _cfg(kernel_log=str(tmp_path / "d.log")), timeout=5
        )
        assert url == "http://127.0.0.1:54321/mcp"
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
        _shim._reap_session(proc, None)
        assert proc.poll() is not None

    def test_idempotent_on_dead_child(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        # Both calls must be no-op-safe on an already-dead child.
        _shim._reap_session(proc, None)
        _shim._reap_session(proc, None)
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
        assert _shim._install_client_death_watchdog(object(), None, "sid") is None

    def test_not_armed_when_client_unidentifiable(self, monkeypatch):
        monkeypatch.setattr(_shim, "_is_windows", lambda: True)
        monkeypatch.setattr(_shim, "_find_client_process", lambda: None)
        assert _shim._install_client_death_watchdog(object(), None, "sid") is None

    def test_not_armed_when_client_handle_unopenable(self, monkeypatch):
        # A client we found but cannot open (returns None) must not arm --
        # never reap a live session off a baseline we could not establish.
        monkeypatch.setattr(_shim, "_is_windows", lambda: True)
        monkeypatch.setattr(_shim, "_find_client_process", lambda: _FakeProc([]))
        monkeypatch.setattr(_shim._winjob, "open_for_wait", lambda pid: None)
        assert _shim._install_client_death_watchdog(object(), None, "sid") is None

    def test_arms_thread_when_client_found_and_openable(self, monkeypatch):
        import threading

        monkeypatch.setattr(_shim, "_is_windows", lambda: True)
        monkeypatch.setattr(_shim, "_find_client_process", lambda: _FakeProc([]))
        monkeypatch.setattr(_shim._winjob, "open_for_wait", lambda pid: "HANDLE")
        monkeypatch.setattr(_shim, "_client_deathwatch", lambda *a, **k: None)
        t = _shim._install_client_death_watchdog(object(), None, "sid")
        assert isinstance(t, threading.Thread)
        t.join(timeout=5)

    def test_reaps_and_exits_when_client_exits(self, monkeypatch):
        events = []
        monkeypatch.setattr(_shim._winjob, "wait_for_process", lambda h: True)
        monkeypatch.setattr(
            _shim, "_reap_session", lambda p, j, s=None: events.append(("reap", s))
        )
        monkeypatch.setattr(
            _shim.os, "_exit", lambda code: events.append(("exit", code))
        )
        _shim._client_deathwatch("HANDLE", 4242, object(), None, "sid")
        # The client's exit must both reap (de-registering the session) and exit.
        assert events == [("reap", "sid"), ("exit", 0)]

    def test_does_not_reap_on_wait_error(self, monkeypatch):
        # An undecided wait (error / not-signalled) must NOT tear down a session
        # that may still be live -- the other teardown paths remain the backstop.
        events = []
        monkeypatch.setattr(_shim._winjob, "wait_for_process", lambda h: False)
        monkeypatch.setattr(
            _shim, "_reap_session", lambda *a, **k: events.append("reap")
        )
        monkeypatch.setattr(_shim.os, "_exit", lambda code: events.append("exit"))
        _shim._client_deathwatch("HANDLE", 4242, object(), None, "sid")
        assert events == []


class _FakeRemote:
    """ClientSession stand-in recording forwarded calls."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        self.calls.append("list_tools")
        return types.ListToolsResult(tools=[])

    async def call_tool(self, name, arguments, progress_callback=None, *, meta=None):
        self.calls.append(("call_tool", name, arguments, meta))
        if name == "explodes":
            raise RuntimeError("kaboom")
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    async def read_resource(self, uri):
        self.calls.append(("read_resource", str(uri)))
        return types.ReadResourceResult(contents=[])


class TestBuildProxy:
    def _call(self, handler, req):
        return anyio.run(lambda: handler(req))

    def test_list_tools_forwards(self):
        remote = _FakeRemote()
        app = _shim.build_proxy(remote)
        result = self._call(
            app.request_handlers[types.ListToolsRequest],
            types.ListToolsRequest(method="tools/list"),
        )
        assert remote.calls == ["list_tools"]
        assert isinstance(result.root, types.ListToolsResult)

    def test_call_tool_forwards_name_and_args(self):
        remote = _FakeRemote()
        app = _shim.build_proxy(remote)
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
        app = _shim.build_proxy(remote)
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="explodes", arguments={}),
        )
        result = self._call(app.request_handlers[types.CallToolRequest], req)
        assert result.root.isError is True
        assert "kaboom" in result.root.content[0].text


class TestReplayInitOptions:
    def test_instructions_and_identity_survive(self):
        # The whole point of vendoring the bridge: nothing from the child's
        # initialize result may be dropped on the floor (mcp-proxy loses
        # `instructions`; docs/mcp-proxy-vet.md finding 2).
        init = types.InitializeResult(
            protocolVersion="2025-03-26",
            capabilities=types.ServerCapabilities(
                tools=types.ToolsCapability(listChanged=False)
            ),
            serverInfo=types.Implementation(name="biopb-mcp", version="9.9"),
            instructions="guardrails live here",
        )
        opts = _shim.replay_init_options(init)
        assert opts.instructions == "guardrails live here"
        assert opts.server_name == "biopb-mcp"
        assert opts.server_version == "9.9"
        assert opts.capabilities is init.capabilities


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
    return env


def _pid_alive(pid):
    """Whether ``pid`` names a live process — WITHOUT killing or perturbing it."""
    if os.name == "nt":
        import psutil  # a biopb-mcp[mcp] dep; only needed on the Windows leg

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
    """The real thing (all OSes): the shim spawns its OWN session child, bridges
    a full stdio session, and reaps that child when the client disconnects."""

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
            # Identity and instructions must round-trip through the bridge.
            assert init["serverInfo"]["name"] == "biopb-mcp"
            assert "execute_code" in (init.get("instructions") or "")

            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            tools = json.loads(shim.stdout.readline())["result"]["tools"]
            assert {"start_kernel", "execute_code"} <= {t["name"] for t in tools}

            # The owned child logs its PID (uvicorn's "Started server process
            # [pid]") and its dynamic listen URL (_server.run) to its own
            # per-session logfile under log/sessions/ (NOT the shared
            # mcp-server.log — that separation is the session-log feature).
            sessions_dir = tmp_path / ".local/share/biopb-mcp/log/sessions"
            session_logs = list(sessions_dir.glob("*.log"))
            assert len(session_logs) == 1, session_logs
            log = session_logs[0].read_bytes().decode(errors="replace")
            child_pid = int(
                _extract(r"Started server process \[(\d+)\]", log, "child pid")
            )
            port = int(_extract(r"http://127\.0\.0\.1:(\d+)/mcp", log, "listen port"))
            assert _pid_alive(child_pid)  # up now
            assert _shim._port_listening(port) is True

            # Self-registered for control discovery while up (the shared biopb
            # data tree, isolated here via HOME).
            reg_dir = tmp_path / ".local/share/biopb/sessions"
            records = list(reg_dir.glob("*.json"))
            assert len(records) == 1, records
            rec = json.loads(records[0].read_text())
            # The record carries the session's port and the pid the shim owns and
            # reaps (spawn_session registers proc.pid). On POSIX that equals the
            # uvicorn pid the log reports (child_pid); on Windows a Store-Python/uv
            # launcher shim sits between them, so proc.pid is the launcher, not the
            # inner child_pid. Assert the record is live and routable rather than
            # pid-equal to the inner process.
            assert rec["port"] == port
            assert isinstance(rec["pid"], int) and _pid_alive(rec["pid"])

            # Client hangs up: the shim must exit AND reap its private child
            # (the shared daemon used to survive — that is exactly what changed).
            shim.stdin.close()
            assert shim.wait(timeout=40) == 0
            _await_dead(child_pid, timeout=20)
            assert _shim._port_listening(port) is False  # server truly gone
            # Reap de-registered it — no routing ghost left behind.
            assert list(reg_dir.glob("*.json")) == []
        finally:
            if shim.poll() is None:
                shim.kill()
            if child_pid is not None:
                _force_kill(child_pid)


class TestServe:
    def test_fires_control_start_before_spawning_child(self, monkeypatch):
        """serve() kicks off the control (fire-and-forget) BEFORE spawning the
        child, so the lean control boots in parallel with the child's import-
        dominated startup and is up by the time the child needs the data plane."""
        calls = []

        class _FakeProc:
            pid = 4242

        monkeypatch.setattr(
            _shim._control_client,
            "start_control_detached",
            lambda: calls.append("control"),
        )

        def _fake_spawn(config, *a, **k):
            calls.append("spawn")
            return _FakeProc(), "http://127.0.0.1:1/mcp", None, "sid"

        monkeypatch.setattr(_shim, "spawn_session", _fake_spawn)
        monkeypatch.setattr(_shim, "_install_shim_reaper", lambda *a, **k: None)
        monkeypatch.setattr(
            _shim, "_install_client_death_watchdog", lambda *a, **k: None
        )
        monkeypatch.setattr(_shim, "run_bridge", lambda url: calls.append("bridge"))
        monkeypatch.setattr(_shim, "_reap_session", lambda *a, **k: None)

        _shim.serve(object())
        assert calls == ["control", "spawn", "bridge"]

    def test_control_start_failure_does_not_block_serve(self, monkeypatch):
        """A control-start that raises must never abort the shim -- the bridge is
        the priority; control errors surface later, from the session child."""

        class _FakeProc:
            pid = 4242

        def _boom():
            raise RuntimeError("control start blew up")

        monkeypatch.setattr(_shim._control_client, "start_control_detached", _boom)
        monkeypatch.setattr(
            _shim,
            "spawn_session",
            lambda *a, **k: (_FakeProc(), "http://127.0.0.1:1/mcp", None, "sid"),
        )
        monkeypatch.setattr(_shim, "_install_shim_reaper", lambda *a, **k: None)
        monkeypatch.setattr(
            _shim, "_install_client_death_watchdog", lambda *a, **k: None
        )
        bridged = []
        monkeypatch.setattr(_shim, "run_bridge", lambda url: bridged.append(url))
        monkeypatch.setattr(_shim, "_reap_session", lambda *a, **k: None)

        _shim.serve(object())
        assert bridged == ["http://127.0.0.1:1/mcp"]
