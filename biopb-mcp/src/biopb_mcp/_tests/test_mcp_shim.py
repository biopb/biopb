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
from mcp import types

from biopb_mcp.mcp import _shim


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(**transport):
    return {"mcp": {"transport": transport}}


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


class TestOpenDaemonLog:
    def test_uses_configured_path(self, tmp_path):
        path = tmp_path / "d.log"
        f = _shim._open_daemon_log(_cfg(kernel_log=str(path)))
        try:
            f.write(b"hello\n")
        finally:
            f.close()
        assert path.read_bytes() == b"hello\n"

    def test_empty_path_defaults_under_log_dir(self, tmp_path, monkeypatch):
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "get_log_dir", lambda: tmp_path)
        f = _shim._open_daemon_log(_cfg(kernel_log=""))
        try:
            assert (tmp_path / "mcp-server.log").exists()
        finally:
            f.close()

    def test_falls_back_to_stderr_on_open_error(self):
        f = _shim._open_daemon_log(
            _cfg(kernel_log="/nonexistent_dir/deep/path/kernel.log")
        )
        assert f is getattr(sys.stderr, "buffer", sys.stderr)


class TestSpawnSession:
    def test_reports_port_and_waits_until_listening(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _shim, "_session_command", lambda: [sys.executable, "-c", _FAKE_CHILD]
        )
        cfg = _cfg(kernel_log=str(tmp_path / "d.log"))
        proc, url, job = _shim.spawn_session(cfg, timeout=15)
        try:
            m = re.match(r"http://127\.0\.0\.1:(\d+)/mcp$", url)
            assert m, url
            assert _shim._port_listening(int(m.group(1))) is True
            assert proc.poll() is None  # still running while we bridge
            if os.name == "nt":
                assert job is not None  # Windows: a kill-on-close Job Object
            else:
                assert job is None  # POSIX: reaped via the process group, not a job
        finally:
            _shim._reap_session(proc, job)
        assert proc.poll() is not None  # reaped on the way out

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

        proc, url, job = _shim.spawn_session(
            _cfg(kernel_log=str(tmp_path / "d.log")), timeout=5
        )
        assert url == "http://127.0.0.1:54321/mcp"
        assert captured["env"]["DISPLAY"] == ":test-99"
        assert captured["env"]["BIOPB_PORT_REPORT_FILE"]  # port channel wired
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


def _await_dead(pid, timeout):
    """Poll until ``pid`` is gone (reaping is asynchronous to the shim's exit)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return  # exists but not ours: treat as gone/unknown, not a failure
        time.sleep(0.1)
    raise AssertionError(f"owned child {pid} still alive after {timeout:.0f}s")


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process management in the test"
)
class TestEndToEnd:
    """The real thing: the shim spawns its OWN session child, bridges a full
    stdio session, and reaps that child when the client disconnects."""

    def test_session_is_private_and_reaped_on_disconnect(self, tmp_path):
        env = os.environ.copy()
        env["HOME"] = str(tmp_path)  # isolate config + log dirs

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

            # The owned child logs its PID to the daemon log under the isolated
            # HOME (uvicorn's "Started server process [pid]").
            log = (
                (tmp_path / ".local/share/biopb-mcp/log/mcp-server.log")
                .read_bytes()
                .decode(errors="replace")
            )
            child_pid = int(
                re.search(r"Started server process \[(\d+)\]", log).group(1)
            )
            os.kill(child_pid, 0)  # alive now

            # Client hangs up: the shim must exit AND reap its private child
            # (the shared daemon used to survive — that is exactly what changed).
            shim.stdin.close()
            assert shim.wait(timeout=20) == 0
            _await_dead(child_pid, timeout=15)
        finally:
            if shim.poll() is None:
                shim.kill()
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except OSError:
                    pass
