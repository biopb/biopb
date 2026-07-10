"""Tests for the data-plane supervisor + control API.

Deterministic and hermetic: no real tensor server. The spawn/adopt paths use a
trivial child that just binds the gRPC port (liveness is a TCP connect), and the
restart/backoff logic is exercised against a fake process so timing is exact.
"""

import socket
import sys
import time
from unittest.mock import MagicMock

import pytest

from biopb_admin._control import serve_control_api
from biopb_admin._supervisor import DataPlaneSpec, DataPlaneSupervisor


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _binder_argv(port: int) -> list[str]:
    """A child that binds ``port`` and idles -- a stand-in tensor server for the
    supervisor's TCP liveness probe."""
    code = (
        "import socket,time;"
        "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
        f"s.bind(('127.0.0.1',{port}));s.listen(5);time.sleep(60)"
    )
    return [sys.executable, "-c", code]


def _listener(port: int) -> socket.socket:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(5)
    return s


@pytest.fixture
def spec(tmp_path):
    return DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
    )


# --------------------------------------------------------------------------- #
# spawn / adopt
# --------------------------------------------------------------------------- #
def test_ensure_spawns_and_reports_serving(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    monkeypatch.setattr(sup, "_build_argv", lambda: _binder_argv(spec.grpc_port))
    try:
        sup.ensure()
        assert sup.wait_until_up(10.0) is True
        snap = sup.snapshot()
        assert snap["state"] == "serving"
        assert snap["owned"] is True
        assert snap["pid"] is not None
    finally:
        sup.stop()
    # stop() reaps the owned child, so the port is free again.
    assert sup._port_up() is False


def test_ensure_adopts_already_running_server(spec, monkeypatch):
    listener = _listener(spec.grpc_port)
    try:
        sup = DataPlaneSupervisor(spec)
        spawn = MagicMock()
        monkeypatch.setattr(sup, "_spawn_locked", spawn)
        sup.ensure()
        spawn.assert_not_called()  # adopted, not double-bound
        snap = sup.snapshot()
        assert snap["state"] == "serving"
        assert snap["owned"] is False
        assert snap["pid"] is None
    finally:
        listener.close()


def test_stop_leaves_adopted_plane_running(spec, monkeypatch):
    listener = _listener(spec.grpc_port)
    try:
        sup = DataPlaneSupervisor(spec)
        monkeypatch.setattr(sup, "_spawn_locked", MagicMock())
        sup.ensure()
        terminate = MagicMock()
        monkeypatch.setattr(sup, "_terminate", terminate)
        sup.stop()
        terminate.assert_not_called()  # we did not spawn it; leave it be
        assert sup._port_up() is True
    finally:
        listener.close()


# --------------------------------------------------------------------------- #
# restart / backoff (fake process, exact timing)
# --------------------------------------------------------------------------- #
class _DeadProc:
    returncode = 1

    def poll(self):
        return 1  # exited


def _downed(sup, monkeypatch):
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: False)
    spawn = MagicMock()
    monkeypatch.setattr(sup, "_spawn_locked", spawn)
    sup._proc = _DeadProc()
    sup._state.want = True
    sup._state.owned = True
    return spawn


def test_tick_waits_out_backoff_before_restart(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    spawn = _downed(sup, monkeypatch)
    sup._state.next_attempt_at = time.monotonic() + 100  # not due yet
    sup.tick()
    spawn.assert_not_called()


def test_tick_restarts_crashed_owned_child(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    spawn = _downed(sup, monkeypatch)
    sup._state.next_attempt_at = 0.0  # due
    sup.tick()
    spawn.assert_called_once()
    assert sup._state.restarts == 1
    assert sup._state.failures == 1
    assert "restarting" in (sup._state.last_error or "")


def test_tick_noop_when_not_wanted(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    spawn = _downed(sup, monkeypatch)
    sup._state.want = False  # stopped / --no-data-plane
    sup._state.next_attempt_at = 0.0
    sup.tick()
    spawn.assert_not_called()


def test_backoff_grows_then_resets_on_recovery(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    sup._state.failures = 3
    assert sup._backoff() == 4.0  # _BACKOFF_SCHEDULE[3]
    # A run of healthy ticks past the reset window clears the failure count.
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: True)
    sup._state.want = True
    sup._proc = None
    sup._state.up_since = time.monotonic() - 120  # long healthy
    sup.tick()
    assert sup._state.failures == 0


# --------------------------------------------------------------------------- #
# control API
# --------------------------------------------------------------------------- #
def test_control_api_health_and_ensure(spec, monkeypatch):
    import json
    import urllib.request

    listener = _listener(spec.grpc_port)  # so ensure adopts, no real spawn
    sup = DataPlaneSupervisor(spec)
    api_port = _free_port()
    server, _thread = serve_control_api("127.0.0.1", api_port, sup, ensure_timeout=5.0)
    base = f"http://127.0.0.1:{api_port}"
    try:
        health = json.loads(urllib.request.urlopen(f"{base}/health", timeout=3).read())
        assert health["admin"] == "ok"
        assert health["data_plane"]["state"] == "serving"

        req = urllib.request.Request(
            f"{base}/data_plane/ensure", data=b"", method="POST"
        )
        ensured = json.loads(urllib.request.urlopen(req, timeout=6).read())
        assert ensured["data_plane"]["state"] == "serving"
        assert ensured["data_plane"]["owned"] is False

        # Unknown path -> 404 JSON, not a hang.
        try:
            urllib.request.urlopen(f"{base}/nope", timeout=3)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        listener.close()
