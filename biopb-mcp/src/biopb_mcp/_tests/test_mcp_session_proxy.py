"""The session proxy: a kernel Lab can start that relays to the session kernel."""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from jupyter_client import BlockingKernelClient  # noqa: E402
from jupyter_client.connect import write_connection_file  # noqa: E402

from biopb_mcp.mcp import _session_proxy  # noqa: E402
from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


def _info_file(root, name, port, mtime=None):
    path = root / name
    path.write_text(
        json.dumps(
            {"ip": "127.0.0.1", "shell_port": port, "key": "k", "transport": "tcp"}
        )
    )
    if mtime:
        os.utime(path, (mtime, mtime))
    return str(path)


@pytest.fixture
def listener():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    yield s
    s.close()


def _dead_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestFindSession:
    def test_takes_the_newest_file_that_answers(self, tmp_path, listener):
        now = time.time()
        _info_file(tmp_path, "kernel-biopb-stale.json", _dead_port(), now)
        live = _info_file(
            tmp_path, "kernel-biopb-live.json", listener.getsockname()[1], now - 10
        )
        assert _session_proxy.find_session(str(tmp_path))[0] == live

    def test_none_when_nothing_answers(self, tmp_path):
        _info_file(tmp_path, "kernel-biopb-stale.json", _dead_port())
        assert _session_proxy.find_session(str(tmp_path)) is None

    def test_env_pins_a_file(self, tmp_path, listener, monkeypatch):
        now = time.time()
        _info_file(tmp_path, "kernel-biopb-newer.json", listener.getsockname()[1], now)
        pinned = _info_file(tmp_path, "other.json", listener.getsockname()[1], now - 10)
        monkeypatch.setenv(_session_proxy.ENV_CONNECTION_FILE, pinned)
        assert _session_proxy.find_session(str(tmp_path))[0] == pinned


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def proxy(runtime):
    """A proxy process and a client on the connection file Lab would give it."""
    cf, _ = write_connection_file(str(runtime / "proxy.json"))
    p = subprocess.Popen(
        [sys.executable, "-m", "biopb_mcp.mcp._session_proxy", "-f", cf]
    )
    kc = BlockingKernelClient(connection_file=cf)
    kc.load_connection_file()
    kc.start_channels()
    yield p, kc
    kc.stop_channels()
    p.terminate()
    p.wait(timeout=10)


def _run(kc, code, **kw):
    msgs = []
    reply = kc.execute_interactive(code, timeout=30, output_hook=msgs.append, **kw)
    return reply["content"], msgs


def test_without_a_session_a_request_says_so(proxy):
    _, kc = proxy
    kc.wait_for_ready(timeout=30)
    content, _ = _run(kc, "1")
    assert content["status"] == "error"
    assert "no running biopb session" in content["evalue"]


def test_relays_to_the_session_and_shutdown_only_detaches(proxy):
    p, kc = proxy
    host = KernelHost(health_probe_code=None, startup_timeout=60.0)
    host.start()
    try:
        kc.wait_for_ready(timeout=30)
        content, msgs = _run(kc, "x = 6 * 7\nprint('x is', x)")
        assert content["status"] == "ok"
        assert any("x is 42" in m["content"].get("text", "") for m in msgs)
        # An input() round trip goes through the stdin relay.
        content, msgs = _run(
            kc,
            "print('got', input('? '))",
            allow_stdin=True,
            stdin_hook=lambda m: kc.input("y"),
        )
        assert any("got y" in m["content"].get("text", "") for m in msgs)
        # The host sees it as a foreign cell, and the session outlives the proxy.
        kc.shutdown()
        p.wait(timeout=10)
        assert host.is_alive()
        assert host.execute("print(x)")["stdout"].strip() == "42"
    finally:
        host.shutdown()


def test_a_session_restart_does_not_strand_the_proxy(proxy):
    """The restarted kernel keeps its connection file, key and ports, so the
    proxy's sockets reconnect and the notebook goes on without being restarted."""
    _, kc = proxy
    host = KernelHost(health_probe_code=None, startup_timeout=60.0)
    host.start()
    try:
        path = host.connection_file
        kc.wait_for_ready(timeout=30)
        content, _ = _run(kc, "x = 1")
        assert content["status"] == "ok"
        host.restart()
        assert host.connection_file == path
        content, msgs = _run(kc, "print('after', 6 * 7)")
        assert content["status"] == "ok"
        assert any("after 42" in m["content"].get("text", "") for m in msgs)
        # The namespace is the new kernel's: the restart cleared it.
        content, _ = _run(kc, "x")
        assert content["status"] == "error"
    finally:
        host.shutdown()
