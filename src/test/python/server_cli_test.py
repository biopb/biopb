"""Unit tests for biopb.cli daemon-management stop logic.

Focuses on the cross-platform graceful-stop path (POSIX SIGTERM vs the Windows
named-event request) and the force-kill fallback. OS calls are mocked so the
tests are deterministic and fast on any platform; time.sleep is neutralized.
"""

from unittest.mock import MagicMock, patch

import pytest

import biopb.cli as cli


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Keep the wait/force-kill loops instant."""
    monkeypatch.setattr(cli.time, "sleep", lambda *_a, **_k: None)


class TestRequestGracefulStop:
    def test_posix_sends_sigterm(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        with patch.object(cli.os, "kill") as kill:
            assert cli._request_graceful_stop(4321) is True
            kill.assert_called_once_with(4321, cli.signal.SIGTERM)

    def test_posix_returns_false_when_kill_raises(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        with patch.object(cli.os, "kill", side_effect=OSError):
            assert cli._request_graceful_stop(4321) is False

    def test_windows_routes_to_named_event(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        with patch.object(cli, "_win_set_shutdown_event", return_value=True) as ev:
            assert cli._request_graceful_stop(4321) is True
            ev.assert_called_once_with(4321)


class TestGracefulStop:
    def test_returns_true_when_process_exits(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        # Alive on the first poll, gone on the second.
        alive = iter([True, False])
        monkeypatch.setattr(cli, "_is_process_running", lambda _pid: next(alive))
        monkeypatch.setattr(cli, "_request_graceful_stop", lambda _pid: True)
        remove = MagicMock()
        monkeypatch.setattr(cli, "_remove_pid", remove)
        with patch.object(cli.os, "kill") as kill:
            assert cli._graceful_stop(1234, timeout=5) is True
            kill.assert_not_called()  # never escalated to force-kill
        remove.assert_called_once()

    def test_force_kills_when_unresponsive(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(cli, "_is_process_running", lambda _pid: True)
        monkeypatch.setattr(cli, "_request_graceful_stop", lambda _pid: True)
        remove = MagicMock()
        monkeypatch.setattr(cli, "_remove_pid", remove)
        with patch.object(cli.os, "kill") as kill:
            assert cli._graceful_stop(1234, timeout=2) is False
            # Last call is the force-kill (SIGKILL on POSIX).
            kill.assert_called_with(1234, cli.signal.SIGKILL)
        remove.assert_called_once()

    def test_windows_retries_event_when_daemon_still_starting(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        # Initial request fails (event not created yet), then succeeds on retry.
        ev = MagicMock(side_effect=[False, False, True])
        monkeypatch.setattr(cli, "_win_set_shutdown_event", ev)
        monkeypatch.setattr(cli, "_request_graceful_stop", lambda _pid: False)
        # Process exits right after the event is finally set.
        alive = iter([False])
        monkeypatch.setattr(cli, "_is_process_running", lambda _pid: next(alive))
        monkeypatch.setattr(cli, "_remove_pid", MagicMock())
        assert cli._graceful_stop(1234, timeout=5) is True
        assert ev.call_count == 3  # retried until it landed


class TestWinSetShutdownEventOffWindows:
    def test_returns_false_safely_off_windows(self):
        # ctypes.windll doesn't exist off Windows; must be caught, not raised.
        assert cli._win_set_shutdown_event(1234) is False
