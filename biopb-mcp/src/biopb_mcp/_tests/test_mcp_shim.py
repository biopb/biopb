"""Tests for the stdio bridge ("shim") to the http daemon.

Unit tests cover the daemon-activation logic (probe / spawn / timeout), the
faithful initialize replay (the `instructions` field above all — losing it is
the defect that disqualified delegating to mcp-proxy), and the request
forwarding of the vendored proxy. One end-to-end test runs the real thing:
``biopb-mcp --transport stdio`` as a subprocess, which must spawn the http
daemon, bridge a full JSON-RPC session, and leave the daemon running after
the client hangs up.
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys

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


class TestPortListening:
    def test_true_for_listening_socket(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            assert _shim._port_listening(s.getsockname()[1]) is True

    def test_false_for_closed_port(self):
        assert _shim._port_listening(_free_port()) is False


class TestDaemonCommand:
    def test_module_reentry_when_not_frozen(self):
        cmd = _shim._daemon_command(8765)
        assert cmd[:3] == [sys.executable, "-m", "biopb_mcp.mcp"]
        assert cmd[3:] == ["--transport", "http", "--port", "8765"]

    def test_frozen_build_calls_its_own_binary(self, monkeypatch):
        # PyInstaller: sys.executable IS the launcher; no module tree to -m.
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        cmd = _shim._daemon_command(8765)
        assert cmd == [sys.executable, "--transport", "http", "--port", "8765"]


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
            assert (tmp_path / "kernel.log").exists()
        finally:
            f.close()

    def test_falls_back_to_stderr_on_open_error(self):
        f = _shim._open_daemon_log(
            _cfg(kernel_log="/nonexistent_dir/deep/path/kernel.log")
        )
        assert f is getattr(sys.stderr, "buffer", sys.stderr)


class TestEnsureDaemon:
    def test_no_spawn_when_already_listening(self, monkeypatch):
        def _no_spawn(*a, **k):
            raise AssertionError("must not spawn when the port is served")

        monkeypatch.setattr(_shim.subprocess, "Popen", _no_spawn)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            assert _shim.ensure_daemon({}, s.getsockname()[1]) is False

    def test_spawns_and_waits_until_listening(self, tmp_path, monkeypatch):
        port = _free_port()
        # Stand-in daemon: sleeps briefly (simulating import time), then binds
        # and *accepts* connections. It must accept, not merely listen(): the
        # readiness probe opens a fresh TCP connection on every poll, and an
        # un-drained accept backlog makes a later probe fail on macOS's stricter
        # socket stack (the real uvicorn daemon accepts, so this is a fixture
        # artifact, not a production bug).
        script = (
            "import socket, sys, threading, time\n"
            "time.sleep(1)\n"
            "s = socket.socket()\n"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "s.bind(('127.0.0.1', int(sys.argv[1])))\n"
            "s.listen(16)\n"
            "def _serve():\n"
            "    while True:\n"
            "        try:\n"
            "            conn, _ = s.accept()\n"
            "            conn.close()\n"
            "        except OSError:\n"
            "            break\n"
            "threading.Thread(target=_serve, daemon=True).start()\n"
            "time.sleep(30)\n"
        )
        monkeypatch.setattr(
            _shim,
            "_daemon_command",
            lambda p: [sys.executable, "-c", script, str(p)],
        )
        # Track the detached child by handle so we reap it deterministically
        # (no pattern-matching / pkill).
        spawned = []
        real_popen = _shim.subprocess.Popen

        def _tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(_shim.subprocess, "Popen", _tracking_popen)
        cfg = _cfg(kernel_log=str(tmp_path / "d.log"))
        try:
            assert _shim.ensure_daemon(cfg, port, timeout=15) is True
            assert _shim._port_listening(port) is True
        finally:
            for proc in spawned:
                proc.kill()

    def test_timeout_when_daemon_dies_on_boot(self, tmp_path, monkeypatch):
        port = _free_port()
        monkeypatch.setattr(
            _shim,
            "_daemon_command",
            lambda p: [sys.executable, "-c", "raise SystemExit(1)"],
        )
        cfg = _cfg(kernel_log=str(tmp_path / "d.log"))
        with pytest.raises(TimeoutError, match="daemon log"):
            _shim.ensure_daemon(cfg, port, timeout=1)


class _FakeRemote:
    """ClientSession stand-in recording forwarded calls."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        self.calls.append("list_tools")
        return types.ListToolsResult(tools=[])

    async def call_tool(
        self, name, arguments, progress_callback=None, *, meta=None
    ):
        self.calls.append(("call_tool", name, arguments, meta))
        if name == "explodes":
            raise RuntimeError("kaboom")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="ok")]
        )

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
        # The whole point of vendoring the bridge: nothing from the daemon's
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


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process management in the test"
)
class TestEndToEnd:
    """The real thing: shim subprocess spawns the daemon and bridges stdio."""

    def test_full_session_and_daemon_survival(self, tmp_path):
        port = _free_port()
        env = os.environ.copy()
        env["HOME"] = str(tmp_path)  # isolate config + log dirs

        shim = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "biopb_mcp.mcp",
                "--transport",
                "stdio",
                "--port",
                str(port),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        daemon_pid = None
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
            assert {"start_kernel", "execute_code"} <= {
                t["name"] for t in tools
            }

            # Client hangs up; the shim must exit promptly and cleanly...
            shim.stdin.close()
            assert shim.wait(timeout=15) == 0

            # ...while the daemon survives (uvicorn logs its pid to the
            # daemon log, which the spawned-detached daemon writes under the
            # isolated HOME).
            log = (
                (tmp_path / ".local/share/biopb-mcp/log/kernel.log")
                .read_bytes()
                .decode(errors="replace")
            )
            daemon_pid = int(
                re.search(r"Started server process \[(\d+)\]", log).group(1)
            )
            assert _shim._port_listening(port) is True
            os.kill(daemon_pid, 0)  # alive
        finally:
            if shim.poll() is None:
                shim.kill()
            if daemon_pid is None:
                # Best effort: find the daemon via its commandline.
                subprocess.run(
                    ["pkill", "-f", f"--transport http --port {port}"],
                    check=False,
                )
            else:
                os.kill(daemon_pid, signal.SIGTERM)
