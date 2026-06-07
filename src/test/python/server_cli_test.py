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

    def test_windows_routes_to_sentinel(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        with patch.object(cli, "_win_request_shutdown", return_value=True) as req:
            assert cli._request_graceful_stop(4321) is True
            req.assert_called_once_with(4321)


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
            # Last call is the force-kill: SIGKILL on POSIX, SIGTERM on Windows
            # (signal.SIGKILL doesn't exist there) - matches the code's fallback.
            expected_sig = getattr(cli.signal, "SIGKILL", cli.signal.SIGTERM)
            kill.assert_called_with(1234, expected_sig)
        remove.assert_called_once()

    def test_force_kill_surfaces_diag_when_delivery_failed(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(cli, "_request_graceful_stop", lambda _pid: False)
        monkeypatch.setattr(cli, "_LAST_WIN_SHUTDOWN_DIAG", "could not write shutdown sentinel: boom")
        monkeypatch.setattr(cli, "_is_process_running", lambda _pid: True)
        monkeypatch.setattr(cli, "_remove_pid", MagicMock())
        with patch.object(cli.os, "kill"):
            assert cli._graceful_stop(1234, timeout=1) is False
        assert "Graceful stop unavailable" in capsys.readouterr().out


class TestWinRequestShutdown:
    def test_writes_sentinel_file(self, tmp_path, monkeypatch):
        # Redirect the biopb data dir to a temp location.
        monkeypatch.setattr(cli, "PID_FILE", tmp_path / "tensor-server.pid")
        assert cli._win_request_shutdown(4321) is True
        sentinel = cli._win_shutdown_sentinel(4321)
        assert sentinel.exists()
        assert sentinel.read_text() == "stop"
        assert sentinel == tmp_path / "tensor-server-4321.stop"
