"""Unit tests for biopb.cli daemon-management stop logic.

Focuses on the cross-platform graceful-stop path (POSIX SIGTERM vs the Windows
named-event request) and the force-kill fallback. OS calls are mocked so the
tests are deterministic and fast on any platform; time.sleep is neutralized.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

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
            req.assert_called_once_with()  # sentinel name is not pid-keyed


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
        monkeypatch.setattr(cli, "_win_remove_sentinel", MagicMock())
        with patch.object(cli.os, "kill"):
            assert cli._graceful_stop(1234, timeout=1) is False
        assert "Graceful stop unavailable" in capsys.readouterr().out


class TestStatusHealth:
    """`server status` now queries the live Flight health (status + source_count)
    and can emit JSON for scripting (used by the installer)."""

    def _invoke(self, monkeypatch, *, pid, running, health, extra_args=()):
        monkeypatch.setattr(cli, "_read_pid", lambda: pid)
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: running)
        if isinstance(health, list):
            seq = iter(health)
            monkeypatch.setattr(cli, "_query_health", lambda *_a, **_k: next(seq))
        else:
            monkeypatch.setattr(cli, "_query_health", lambda *_a, **_k: health)
        monkeypatch.setattr(cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None))
        res = CliRunner().invoke(cli.app, ["server", "status", "--json", *extra_args])
        assert res.exit_code == 0, res.output
        return json.loads(res.stdout.strip().splitlines()[-1])

    def test_running_serving_reports_sources(self, monkeypatch):
        d = self._invoke(
            monkeypatch,
            pid=123,
            running=True,
            health={"status": "SERVING", "source_count": 7, "writable": False,
                    "uptime_seconds": 42},
        )
        assert d["running"] is True and d["pid"] == 123
        assert d["status"] == "running"
        assert d["health"] == "SERVING" and d["source_count"] == 7

    def test_not_running(self, monkeypatch):
        d = self._invoke(monkeypatch, pid=None, running=False, health=None)
        assert d["running"] is False and d["pid"] is None
        assert d["status"] == "stopped"
        assert d["health"] is None and d["source_count"] is None

    def test_stale_pid(self, monkeypatch):
        d = self._invoke(monkeypatch, pid=999, running=False, health=None)
        assert d["running"] is False and d["status"] == "stale"

    def test_running_but_no_sources(self, monkeypatch):
        d = self._invoke(
            monkeypatch,
            pid=5,
            running=True,
            health={"status": "SERVING", "source_count": 0},
        )
        assert d["health"] == "SERVING" and d["source_count"] == 0

    def test_unreachable_health(self, monkeypatch):
        # Process up but Flight unreachable -> health None, still running.
        d = self._invoke(monkeypatch, pid=5, running=True, health=None)
        assert d["running"] is True and d["health"] is None

    def test_wait_polls_until_serving(self, monkeypatch):
        # First probe STARTING, second SERVING; --wait drives the poll loop
        # (time.sleep is neutralized by the autouse fixture).
        d = self._invoke(
            monkeypatch,
            pid=5,
            running=True,
            health=[
                {"status": "STARTING", "source_count": 0},
                {"status": "SERVING", "source_count": 3},
            ],
            extra_args=["--wait", "5"],
        )
        assert d["health"] == "SERVING" and d["source_count"] == 3

    def test_wait_logs_progress(self, monkeypatch):
        # --wait should log a human-facing progress report while STARTING; JSON
        # stays the last line (parsed by _invoke).
        monkeypatch.setattr(cli, "_read_pid", lambda: 5)
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        seq = iter(
            [
                {"status": "STARTING", "source_count": 2},
                {"status": "SERVING", "source_count": 4},
            ]
        )
        monkeypatch.setattr(cli, "_query_health", lambda *_a, **_k: next(seq))
        monkeypatch.setattr(cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None))
        res = CliRunner().invoke(
            cli.app, ["server", "status", "--json", "--wait", "5"]
        )
        assert res.exit_code == 0, res.output
        assert "starting" in res.output and "2 source" in res.output
        # JSON (final verdict) is still the last line.
        last = json.loads(res.stdout.strip().splitlines()[-1])
        assert last["health"] == "SERVING" and last["source_count"] == 4

    def test_no_wait_is_silent(self, monkeypatch):
        # Without --wait, a single probe and no progress chatter.
        monkeypatch.setattr(cli, "_read_pid", lambda: 5)
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(
            cli, "_query_health", lambda *_a, **_k: {"status": "SERVING", "source_count": 1}
        )
        monkeypatch.setattr(cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None))
        res = CliRunner().invoke(cli.app, ["server", "status", "--json"])
        assert res.exit_code == 0
        assert "starting" not in res.output and "waiting" not in res.output


class TestWinRequestShutdown:
    def test_writes_fixed_sentinel_file(self, tmp_path, monkeypatch):
        # Redirect the biopb data dir to a temp location.
        monkeypatch.setattr(cli, "PID_FILE", tmp_path / "tensor-server.pid")
        assert cli._win_request_shutdown() is True
        sentinel = cli._win_shutdown_sentinel()
        assert sentinel.exists()
        assert sentinel.read_text() == "stop"
        assert sentinel == tmp_path / "tensor-server.stop"  # not pid-keyed
        cli._win_remove_sentinel()
        assert not sentinel.exists()
