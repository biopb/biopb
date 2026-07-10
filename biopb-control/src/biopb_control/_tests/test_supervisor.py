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

from biopb_control._control import serve_control_api
from biopb_control._supervisor import DataPlaneSpec, DataPlaneSupervisor


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
# spawn (sole ownership) / conflict
# --------------------------------------------------------------------------- #
def test_ensure_spawns_and_reports_serving(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    monkeypatch.setattr(sup, "_build_argv", lambda: _binder_argv(spec.grpc_port))
    try:
        sup.ensure()
        assert sup.wait_until_up(10.0) is True
        snap = sup.snapshot()
        assert snap["state"] == "serving"
        assert snap["pid"] is not None
    finally:
        sup.stop()
    # stop() reaps the child (the control owns it exclusively), so the port frees.
    assert sup._port_up() is False


def test_ensure_refuses_when_port_held_by_another(spec, monkeypatch):
    # The control is the sole owner: a port already held by a process it did not
    # start is a conflict it refuses, not a server to adopt.
    listener = _listener(spec.grpc_port)
    try:
        sup = DataPlaneSupervisor(spec)
        spawn = MagicMock()
        monkeypatch.setattr(sup, "_spawn_locked", spawn)
        sup.ensure()
        spawn.assert_not_called()  # refused, not adopted, not double-bound
        snap = sup.snapshot()
        assert snap["state"] == "conflict"
        assert snap["pid"] is None
        assert "did not start" in (snap["last_error"] or "")
    finally:
        listener.close()


def test_stop_terminates_the_owned_plane(spec, monkeypatch):
    # There is no 'adopted, leave it running' case: stop always tears the plane
    # down. Here _spawn_locked is faked to install a live child; stop must reap it.
    sup = DataPlaneSupervisor(spec)
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: False)

    class _LiveProc:
        pid = 4321

        def poll(self):
            return None  # alive

    monkeypatch.setattr(
        sup, "_spawn_locked", lambda: setattr(sup, "_proc", _LiveProc())
    )
    sup.ensure()
    assert sup._proc is not None
    terminate = MagicMock()
    monkeypatch.setattr(sup, "_terminate", terminate)
    sup.stop()
    terminate.assert_called_once()  # the plane we own is always stopped


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


def test_tick_reaps_dead_handle_even_while_backing_off(spec, monkeypatch):
    # The moment a crash is observed, the dead Popen handle must be cleared to
    # None -- even if the backoff window forbids respawning this tick -- so
    # `self._proc is None` stays an honest "no live child of ours" signal.
    sup = DataPlaneSupervisor(spec)
    _downed(sup, monkeypatch)
    sup._state.next_attempt_at = time.monotonic() + 100  # can't respawn yet
    sup.tick()
    assert sup._proc is None  # dead handle reaped, not left dangling
    assert sup._state.restarts == 1  # crash still counted exactly once


def test_crashed_child_with_port_still_held_reports_conflict(spec, monkeypatch):
    # An owned child dies AND something else now holds the port. The supervisor
    # reaps the stale dead handle and, as the sole owner, does NOT restart or
    # claim the unowned server -- it reports a conflict.
    sup = DataPlaneSupervisor(spec)
    spawn = _downed(sup, monkeypatch)
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: True)  # something up
    sup.tick()
    spawn.assert_not_called()
    assert sup._proc is None
    snap = sup.snapshot()
    assert snap["state"] == "conflict"
    assert "did not start" in (snap["last_error"] or "")


def test_ensure_after_crash_is_not_fooled_by_dead_handle(spec, monkeypatch):
    # ensure() decides spawn-vs-refuse from `self._proc`; a crashed child left in
    # it must not be mistaken for a live one. With the port down it spawns fresh.
    sup = DataPlaneSupervisor(spec)
    spawn = _downed(sup, monkeypatch)  # dead handle, port down, spawn mocked
    sup.ensure()
    spawn.assert_called_once()  # reaped the dead handle, then spawned
    assert sup._proc is None  # (spawn is mocked, so no new handle set)


def test_backoff_grows_then_resets_on_recovery(spec, monkeypatch):
    sup = DataPlaneSupervisor(spec)
    sup._state.failures = 3
    assert sup._backoff() == 4.0  # _BACKOFF_SCHEDULE[3]
    # A live child up past the reset window clears the failure count.
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: True)
    sup._state.want = True

    class _LiveProc:
        pid = 4321

        def poll(self):
            return None  # alive

    sup._proc = _LiveProc()
    sup._state.up_since = time.monotonic() - 120  # long healthy
    sup.tick()
    assert sup._state.failures == 0


# --------------------------------------------------------------------------- #
# control API
# --------------------------------------------------------------------------- #
def test_control_api_health_and_ensure(spec, monkeypatch):
    import json
    import urllib.request

    sup = DataPlaneSupervisor(spec)
    # POST /data_plane/ensure spawns a real (binder) child the control owns.
    monkeypatch.setattr(sup, "_build_argv", lambda: _binder_argv(spec.grpc_port))
    api_port = _free_port()
    server, _thread = serve_control_api("127.0.0.1", api_port, sup, ensure_timeout=8.0)
    base = f"http://127.0.0.1:{api_port}"
    try:
        # Before ensure: nothing running.
        health = json.loads(urllib.request.urlopen(f"{base}/health", timeout=3).read())
        assert health["control"] == "ok"
        assert health["data_plane"]["state"] == "stopped"

        # Ensure spawns and waits until the port is up.
        req = urllib.request.Request(
            f"{base}/data_plane/ensure", data=b"", method="POST"
        )
        ensured = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert ensured["data_plane"]["state"] == "serving"
        assert ensured["data_plane"]["pid"] is not None

        # Unknown path -> 404 JSON, not a hang.
        try:
            urllib.request.urlopen(f"{base}/nope", timeout=3)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        sup.stop()  # reap the binder child
