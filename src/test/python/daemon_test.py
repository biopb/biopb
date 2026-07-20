"""Unit tests for the detached-daemon lifecycle helpers in
:mod:`biopb._lifecycle.daemon` (moved out of ``biopb.cli``).

Focuses on the cross-platform graceful-stop path (POSIX SIGTERM vs the Windows
stop-sentinel file) and the force-kill fallback, plus pidfile identity across a
reused PID. OS calls are mocked so the tests are deterministic and fast on any
platform; time.sleep is neutralized.
"""

from unittest.mock import MagicMock, patch

import pytest
from biopb._lifecycle import daemon


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Keep the wait/force-kill loops instant."""
    monkeypatch.setattr(daemon.time, "sleep", lambda *_a, **_k: None)


class TestRequestGracefulStop:
    def test_posix_sends_sigterm(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.platform", "linux")
        with patch.object(daemon.os, "kill") as kill:
            assert daemon.request_graceful_stop(4321, tmp_path / "d.stop") is True
            kill.assert_called_once_with(4321, daemon.signal.SIGTERM)

    def test_posix_returns_false_when_kill_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.platform", "linux")
        with patch.object(daemon.os, "kill", side_effect=OSError):
            assert daemon.request_graceful_stop(4321, tmp_path / "d.stop") is False

    def test_windows_routes_to_sentinel(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.platform", "win32")
        sentinel = tmp_path / "d.stop"
        with patch.object(daemon, "win_request_shutdown", return_value=True) as req:
            assert daemon.request_graceful_stop(4321, sentinel) is True
            req.assert_called_once_with(sentinel)  # the pid plays no part


class TestStopDaemon:
    """The stop path the control daemon drives (`stop_daemon`): wait-then-
    force-kill, gated on identity, driven by a `sentinel` path and `remove_pid`
    callback. Delivery is mocked here; the real SIGTERM/sentinel mechanism is
    exercised in TestStopDaemonDelivery."""

    def test_returns_true_when_process_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.platform", "linux")
        # Ours for the delivery check, gone on the first wait poll.
        ours = iter([True, False])
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: next(ours))
        monkeypatch.setattr(daemon, "request_graceful_stop", lambda *_a: True)
        remove = MagicMock()
        with patch.object(daemon.os, "kill") as kill:
            assert (
                daemon.stop_daemon(
                    1234, timeout=5, sentinel=tmp_path / "d.stop", remove_pid=remove
                )
                is True
            )
            kill.assert_not_called()  # never escalated to force-kill
        remove.assert_called_once()

    def test_force_kills_when_unresponsive(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: True)
        monkeypatch.setattr(daemon, "request_graceful_stop", lambda *_a: True)
        remove = MagicMock()
        with patch.object(daemon.os, "kill") as kill:
            assert (
                daemon.stop_daemon(
                    1234, timeout=2, sentinel=tmp_path / "d.stop", remove_pid=remove
                )
                is False
            )
            # Last call is the force-kill: SIGKILL on POSIX, SIGTERM on Windows
            # (signal.SIGKILL doesn't exist there) - matches the code's fallback.
            expected_sig = getattr(daemon.signal, "SIGKILL", daemon.signal.SIGTERM)
            kill.assert_called_with(1234, expected_sig)
        remove.assert_called_once()

    def test_force_kill_notifies_when_delivery_failed(self, monkeypatch, tmp_path):
        # Windows graceful stop could not be delivered -> the diagnostic is handed
        # to the caller's `notify` callback (the module itself stays console-free).
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: True)
        monkeypatch.setattr(daemon, "request_graceful_stop", lambda *_a: False)
        monkeypatch.setattr(
            daemon, "_LAST_WIN_SHUTDOWN_DIAG", "could not write shutdown sentinel: boom"
        )
        monkeypatch.setattr(daemon, "win_remove_sentinel", MagicMock())
        notify = MagicMock()
        with patch.object(daemon.os, "kill"):
            assert (
                daemon.stop_daemon(
                    1234,
                    timeout=1,
                    sentinel=tmp_path / "d.stop",
                    remove_pid=MagicMock(),
                    notify=notify,
                )
                is False
            )
        notify.assert_called_once_with("could not write shutdown sentinel: boom")

    def test_reused_pid_is_never_signaled(self, monkeypatch, tmp_path):
        # A stale PID now owned by an unrelated process: identity fails, so
        # delivery and force-kill are both skipped, but the PID file is cleaned.
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: False)
        remove = MagicMock()
        with patch.object(daemon.os, "kill") as kill:
            assert (
                daemon.stop_daemon(
                    1234,
                    timeout=3,
                    token=42,
                    sentinel=tmp_path / "d.stop",
                    remove_pid=remove,
                )
                is True
            )
            kill.assert_not_called()  # the innocent PID owner is untouched
        remove.assert_called_once()  # but the stale file is still cleaned up


class TestPidIdentity:
    """The PID file carries a create-time identity token so a stale + reused PID
    (rife on Windows) is not mistaken for our daemon (issue #138, item 1)."""

    def test_record_roundtrip_with_token(self, tmp_path):
        f = tmp_path / "x.pid"
        daemon.write_pid_file(f, 1234, 9988776655)
        assert f.read_text() == "1234\n9988776655"
        assert daemon.read_pid_record(f) == (1234, 9988776655)

    def test_legacy_bare_pid_reads_with_no_token(self, tmp_path):
        # A pre-upgrade file (bare PID) must still read; token None -> liveness only.
        f = tmp_path / "x.pid"
        f.write_text("4321")
        assert daemon.read_pid_record(f) == (4321, None)

    def test_missing_or_garbage_is_none(self, tmp_path):
        assert daemon.read_pid_record(tmp_path / "absent.pid") == (None, None)
        f = tmp_path / "junk.pid"
        f.write_text("not-a-pid")
        assert daemon.read_pid_record(f) == (None, None)

    def test_matching_token_is_our_daemon(self, monkeypatch):
        monkeypatch.setattr(daemon, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(daemon, "_process_create_time", lambda _p: 555)
        assert daemon.is_our_daemon(100, 555) is True

    def test_mismatched_token_is_not_ours(self, monkeypatch):
        # Alive PID, but a DIFFERENT creation time -> a reused PID, not our daemon.
        monkeypatch.setattr(daemon, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(daemon, "_process_create_time", lambda _p: 999)
        assert daemon.is_our_daemon(100, 555) is False

    def test_absent_token_falls_back_to_liveness(self, monkeypatch):
        # Legacy file / unsupported platform -> trust liveness (pre-fix behavior).
        monkeypatch.setattr(daemon, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(daemon, "_process_create_time", lambda _p: 777)
        assert daemon.is_our_daemon(100, None) is True

    def test_dead_pid_is_not_ours(self, monkeypatch):
        monkeypatch.setattr(daemon, "_is_process_running", lambda _p: False)
        assert daemon.is_our_daemon(100, 555) is False


class TestWinRequestShutdown:
    def test_writes_and_removes_sentinel(self, tmp_path):
        # The Windows graceful-stop primitives: drop a "stop" file, tidy it.
        sentinel = tmp_path / "some-daemon.stop"
        assert daemon.win_request_shutdown(sentinel) is True
        assert sentinel.exists()
        assert sentinel.read_text() == "stop"
        daemon.win_remove_sentinel(sentinel)
        assert not sentinel.exists()


class TestStopDaemonDelivery:
    """The real graceful-stop delivery `stop_daemon` drives: SIGTERM on POSIX,
    a stop-sentinel file on Windows (issue #323 - os.kill there is
    TerminateProcess, so the daemon's handlers never run and the napari kernel is
    not reaped gracefully), then wait / force-kill / tidy the sentinel."""

    def test_posix_sends_sigterm_and_waits(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.platform", "linux")
        # Ours before the stop request, gone on the first wait poll.
        ours = iter([True, False])
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: next(ours))
        remove = MagicMock()
        with patch.object(daemon.os, "kill") as kill:
            assert (
                daemon.stop_daemon(
                    1234,
                    timeout=5,
                    token=42,
                    sentinel=tmp_path / "control.stop",
                    remove_pid=remove,
                )
                is True
            )
            kill.assert_called_once_with(1234, daemon.signal.SIGTERM)
        remove.assert_called_once()

    def test_windows_writes_sentinel_never_signals(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        # Keep the final tidy-up from consuming the sentinel we want to inspect.
        monkeypatch.setattr(daemon, "win_remove_sentinel", MagicMock())
        ours = iter([True, False])
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: next(ours))
        sentinel = tmp_path / "control.stop"
        with patch.object(daemon.os, "kill") as kill:
            assert (
                daemon.stop_daemon(
                    1234, timeout=5, token=42, sentinel=sentinel, remove_pid=MagicMock()
                )
                is True
            )
            kill.assert_not_called()  # graceful stop must not TerminateProcess
        # What the daemon's watcher polls for: the sentinel path.
        assert sentinel.read_text() == "stop"

    def test_windows_force_kill_tidies_unconsumed_sentinel(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(daemon, "is_our_daemon", lambda *_a: True)  # wedged
        sentinel = tmp_path / "control.stop"
        with patch.object(daemon.os, "kill") as kill:
            assert (
                daemon.stop_daemon(
                    1234, timeout=1, token=42, sentinel=sentinel, remove_pid=MagicMock()
                )
                is False
            )
            expected_sig = getattr(daemon.signal, "SIGKILL", daemon.signal.SIGTERM)
            kill.assert_called_with(1234, expected_sig)
        # The daemon never consumed it, so stop removed it (a lingering fresh
        # sentinel would stop the next daemon the moment it starts).
        assert not sentinel.exists()
