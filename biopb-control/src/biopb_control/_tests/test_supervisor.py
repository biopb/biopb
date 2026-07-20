"""Tests for the data-plane supervisor + control API.

Deterministic and hermetic: no real tensor server. The spawn/adopt paths use a
trivial child that just binds the gRPC port (liveness is a TCP connect), and the
restart/backoff logic is exercised against a fake process so timing is exact.
"""

import os
import socket
import sys
import time
from unittest.mock import MagicMock

import pytest
from biopb._lifecycle import deathwatch as _deathwatch

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
    # web_port points at a free (closed) port so the control's reverse proxy to
    # the tensor web sidecar fails deterministically (connection refused -> 502)
    # in tests that exercise a path the control does not own itself.
    return DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        web_host="127.0.0.1",
        web_port=_free_port(),
        server_log=tmp_path / "server.log",
    )


def test_child_env_marks_the_plane_supervised(spec):
    # The child (tensor server) must learn it is control-owned so its HTTP
    # sidecar refuses the admin self-restart that would race us (biopb/biopb#418).
    env = DataPlaneSupervisor(spec)._child_env()
    assert env["BIOPB_DATA_PLANE_SUPERVISED"] == "1"


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


def test_spawn_failure_counts_toward_backoff(spec, monkeypatch):
    # A Popen that raises (bad executable, ENOMEM, EAGAIN) must NOT escape
    # ensure()/the /data_plane/ensure handler: it is counted toward the backoff
    # (failures bumped, retry window armed, last_error recorded) so repeated
    # ensures don't hammer with no backoff.
    sup = DataPlaneSupervisor(spec)
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: False)  # port down

    def boom(*_a, **_k):
        raise OSError("cannot spawn")

    monkeypatch.setattr("biopb_control._supervisor.subprocess.Popen", boom)

    sup.ensure()  # must not raise
    assert sup._proc is None
    assert sup._state.failures == 1
    assert sup._state.next_attempt_at > 0  # backoff armed
    assert "failed to spawn" in (sup._state.last_error or "")
    # snapshot stays clean/serviceable (the endpoint returns it as its verdict).
    assert sup.snapshot()["state"] == "down"


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
# death-binding: the plane dies with the control (Pattern O)
# --------------------------------------------------------------------------- #
class _LiveFake:
    pid = 12345

    def poll(self):
        return None  # alive


@pytest.mark.skipif(os.name == "nt", reason="POSIX parent-death pipe wiring")
def test_pipe_failure_counts_toward_backoff(spec, monkeypatch):
    # os.pipe() failing (fd exhaustion) must be counted like a Popen failure, not
    # propagate out of ensure/tick uncounted -- it is armed inside the same try.
    sup = DataPlaneSupervisor(spec)
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: False)

    def boom():
        raise OSError("EMFILE: too many open files")

    monkeypatch.setattr("biopb_control._supervisor.os.pipe", boom)
    sup.ensure()  # must not raise
    assert sup._proc is None
    assert sup._death_w is None  # nothing half-armed left behind
    assert sup._state.failures == 1
    assert sup._state.next_attempt_at > 0  # backoff armed
    assert "failed to spawn" in (sup._state.last_error or "")
    assert sup.snapshot()["state"] == "down"


@pytest.mark.skipif(os.name == "nt", reason="POSIX parent-death pipe wiring")
def test_spawn_arms_parent_death_pipe_posix(spec, monkeypatch):
    # On POSIX the child is tied to the control via a parent-death pipe: it
    # inherits the read end (fd in BIOPB_PARENT_DEATH_FD, passed via pass_fds) and
    # runs in its own session so its self-kill reaps only the plane subtree.
    sup = DataPlaneSupervisor(spec)
    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _LiveFake()

    monkeypatch.setattr("biopb_control._supervisor.subprocess.Popen", _fake_popen)
    assert sup._spawn_locked() is True
    kw = captured["kwargs"]
    fd_str = kw["env"][_deathwatch.ENV_FD]
    assert kw["pass_fds"] == (int(fd_str),)  # the child inherits exactly that fd
    assert kw["start_new_session"] is True  # own session -> contained killpg
    assert sup._death_w is not None  # control keeps the write end live
    # The parent's copy of the read end is closed right after spawn (finally).
    with pytest.raises(OSError):
        os.fstat(int(fd_str))
    # stop() releases the write end.
    monkeypatch.setattr(sup, "_terminate", lambda proc, timeout=10.0: None)
    sup.stop()
    assert sup._death_w is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX parent-death pipe wiring")
def test_reap_dead_child_releases_death_pipe(spec):
    # A crashed child held the read end; reaping it must release the control's
    # write end so the next spawn re-arms a fresh pipe rather than leaking the fd.
    sup = DataPlaneSupervisor(spec)
    r, sup._death_w = os.pipe()
    os.close(r)  # emulate the (dead) child that owned the read end
    sup._proc = _DeadProc()
    assert sup._reap_dead_child() == 1
    assert sup._proc is None
    assert sup._death_w is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX has no Job Object")
def test_no_job_object_on_posix(spec, monkeypatch):
    # POSIX binds via the pipe, never a Job Object; _assign_to_job is a no-op and
    # the stop-time bindings teardown must tolerate no job.
    sup = DataPlaneSupervisor(spec)
    monkeypatch.setattr(sup, "_port_up", lambda timeout=0.5: False)
    monkeypatch.setattr(
        sup, "_spawn_locked", lambda: setattr(sup, "_proc", _LiveFake())
    )
    sup.ensure()
    assert sup._winjob is None
    monkeypatch.setattr(sup, "_terminate", lambda proc, timeout=10.0: None)
    sup.stop()  # must not raise with no job to close
    assert sup._winjob is None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object binding")
def test_spawn_binds_job_on_windows(spec, monkeypatch):
    # On Windows the child is assigned to a kill-on-close Job Object the control
    # holds, so an uncatchable control death reaps the plane. No POSIX pipe.
    sup = DataPlaneSupervisor(spec)
    assigned = {}
    monkeypatch.setattr(
        "biopb_control._supervisor._winjob.create_kill_on_close_job", lambda: "JOB"
    )
    monkeypatch.setattr(
        "biopb_control._supervisor._winjob.assign_process",
        lambda job, pid: assigned.setdefault("call", (job, pid)),
    )
    monkeypatch.setattr(
        "biopb_control._supervisor.subprocess.Popen", lambda argv, **k: _LiveFake()
    )
    assert sup._spawn_locked() is True
    assert sup._winjob == "JOB"
    assert assigned["call"] == ("JOB", _LiveFake.pid)
    assert sup._death_w is None  # no parent-death pipe on Windows


# --------------------------------------------------------------------------- #
# control API
# --------------------------------------------------------------------------- #
def test_bounded_ensure_wait():
    # The server's ensure wait must stay strictly below the client's HTTP timeout
    # (by the margin), be capped by the server's own configured ensure_timeout,
    # floored at the minimum, and fall back to the configured value with no hint.
    from biopb_control._control import _RESPONSE_MARGIN, _bounded_ensure_wait

    assert _bounded_ensure_wait(60, 60) == 60 - _RESPONSE_MARGIN  # below client's
    assert _bounded_ensure_wait(60, 60) < 60
    assert _bounded_ensure_wait(10, 60) == 10  # server's own cap wins
    w = _bounded_ensure_wait(60, 3)  # tiny client timeout -> floor, still < client
    assert w == 1.0 and w < 3
    assert _bounded_ensure_wait(30, 0) == 30  # no hint -> configured value


def test_control_api_health_and_ensure(spec, monkeypatch):
    import json
    import urllib.request

    sup = DataPlaneSupervisor(spec)
    # POST /api/data_plane/ensure spawns a real (binder) child the control owns.
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
            f"{base}/api/data_plane/ensure", data=b"", method="POST"
        )
        ensured = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert ensured["data_plane"]["state"] == "serving"
        assert ensured["data_plane"]["pid"] is not None

        # A path under the /data_plane namespace is reverse-proxied to the tensor
        # web sidecar. None is running (the fixture's web_port is closed), so the
        # proxy returns a clean 502 -- not a control 404, not a hang.
        try:
            urllib.request.urlopen(f"{base}/data_plane/nope", timeout=3)
            raise AssertionError("expected an HTTP error from the proxy")
        except urllib.error.HTTPError as exc:
            assert exc.code == 502
    finally:
        server.shutdown()
        sup.stop()  # reap the binder child
