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
        # token None -> _is_our_daemon falls back to the (mocked) liveness check.
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (pid, None))
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
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (5, None))
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
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (5, None))
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(
            cli, "_query_health", lambda *_a, **_k: {"status": "SERVING", "source_count": 1}
        )
        monkeypatch.setattr(cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None))
        res = CliRunner().invoke(cli.app, ["server", "status", "--json"])
        assert res.exit_code == 0
        assert "starting" not in res.output and "waiting" not in res.output


class TestLogLineHelpers:
    """Pure helpers behind `server logs --level`: parsing a line's level and
    filtering with carry-forward for off-format continuation lines."""

    def test_line_level_parses_format(self):
        line = "[2026-06-12 10:00:00] WARNING biopb_tensor_server.x: msg"
        assert cli._line_level(line) == "WARNING"

    def test_line_level_none_for_off_format(self):
        assert cli._line_level("--- Started at 2026-06-12 10:00:00 ---") is None
        assert cli._line_level("") is None
        assert cli._line_level("    File \"foo.py\", line 3, in bar") is None
        # A bracketed-but-unknown token is not a level.
        assert cli._line_level("[2026-06-12 10:00:00] NOTALEVEL x: y") is None

    def test_filter_none_keeps_all(self):
        lines = ["[t] INFO a: x", "plain", "[t] ERROR a: y"]
        assert cli._filter_lines(lines, None) == lines

    def test_filter_drops_below_threshold(self):
        lines = [
            "[t] DEBUG a: d",
            "[t] INFO a: i",
            "[t] WARNING a: w",
            "[t] ERROR a: e",
        ]
        assert cli._filter_lines(lines, "WARNING") == [
            "[t] WARNING a: w",
            "[t] ERROR a: e",
        ]

    def test_filter_carries_continuation_with_kept_record(self):
        # A kept WARNING carries its off-format traceback continuation lines.
        lines = [
            "[t] INFO a: dropped",
            "[t] WARNING a: boom",
            "Traceback (most recent call last):",
            "    raise ValueError",
            "[t] DEBUG a: dropped-again",
        ]
        assert cli._filter_lines(lines, "WARNING") == [
            "[t] WARNING a: boom",
            "Traceback (most recent call last):",
            "    raise ValueError",
        ]

    def test_filter_initial_decision_is_keep(self):
        # Leading off-format banner before any leveled line is kept.
        lines = ["--- Started ---", "[t] ERROR a: e"]
        assert cli._filter_lines(lines, "ERROR") == lines


class TestLogs:
    """`server logs` reads ~/.local/share/biopb/logs/tensor-server.log."""

    @pytest.fixture
    def log_file(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr(cli, "LOG_DIR", log_dir)
        return log_dir / "tensor-server.log"

    def _run(self, *args):
        res = CliRunner().invoke(cli.app, ["server", "logs", *args])
        return res

    def test_path_prints_and_exits(self, log_file):
        res = self._run("--path")
        assert res.exit_code == 0
        assert str(log_file) in res.output

    def test_missing_file_is_not_an_error(self, log_file):
        res = self._run()
        assert res.exit_code == 0
        assert "No log file" in res.output

    def test_tail_last_n_lines(self, log_file):
        log_file.write_text("\n".join(f"line {i}" for i in range(20)) + "\n")
        res = self._run("-n", "5")
        assert res.exit_code == 0
        out = res.output.strip().splitlines()
        assert out == ["line 15", "line 16", "line 17", "line 18", "line 19"]

    def test_level_filter_drops_below_threshold(self, log_file):
        log_file.write_text(
            "[t] INFO a: i\n[t] DEBUG a: d\n[t] WARNING a: w\n[t] ERROR a: e\n"
        )
        res = self._run("--level", "WARNING")
        assert res.exit_code == 0
        assert "WARNING a: w" in res.output and "ERROR a: e" in res.output
        assert "INFO a: i" not in res.output and "DEBUG a: d" not in res.output

    def test_level_filter_keeps_continuation(self, log_file):
        log_file.write_text(
            "[t] INFO a: i\n[t] WARNING a: boom\ncontinuation trace\n"
        )
        res = self._run("--level", "warning")  # case-insensitive
        assert res.exit_code == 0
        assert "WARNING a: boom" in res.output
        assert "continuation trace" in res.output
        assert "INFO a: i" not in res.output

    def test_invalid_level_exits_1(self, log_file):
        log_file.write_text("[t] INFO a: i\n")
        res = self._run("--level", "FOO")
        assert res.exit_code == 1
        assert "Invalid --level" in res.output


class TestCacheStats:
    """`server cache-stats` queries the running daemon's cache hit/miss stats
    and can emit JSON for scripting."""

    _STATS = {
        "hits": 80,
        "misses": 20,
        "evictions": 3,
        "pending_waits": 0,
        "oversized_skips": 0,
        "ref_held_evictions_skipped": 0,
        "total_entries": 12,
        "total_bytes": 5 * 1024 * 1024,
        "max_bytes": 512 * 1024 * 1024,
        "pool_stats": {
            "unified-tiny": {"hits": 50, "misses": 10, "segments": 2, "bytes": 1048576},
        },
    }

    def _run(self, monkeypatch, *, running, stats, args=()):
        monkeypatch.setattr(
            cli, "_read_pid_record", lambda *_a: (123 if running else None, None)
        )
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: running)
        monkeypatch.setattr(cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None))
        monkeypatch.setattr(cli, "_query_cache_stats", lambda *_a, **_k: stats)
        return CliRunner().invoke(cli.app, ["server", "cache-stats", *args])

    def test_not_running_exits_1(self, monkeypatch):
        res = self._run(monkeypatch, running=False, stats=None)
        assert res.exit_code == 1
        assert "not running" in res.output

    def test_unreachable_exits_1(self, monkeypatch):
        # Process up but the cache_stats action failed -> _query_cache_stats None.
        res = self._run(monkeypatch, running=True, stats=None)
        assert res.exit_code == 1
        assert "Could not retrieve cache stats" in res.output

    def test_json_emits_raw_dict(self, monkeypatch):
        res = self._run(monkeypatch, running=True, stats=self._STATS, args=["--json"])
        assert res.exit_code == 0, res.output
        payload = json.loads(res.stdout.strip().splitlines()[-1])
        assert payload["hits"] == 80 and payload["misses"] == 20
        assert payload["pool_stats"]["unified-tiny"]["segments"] == 2

    def test_table_renders_hit_rate_and_pools(self, monkeypatch):
        res = self._run(monkeypatch, running=True, stats=self._STATS)
        assert res.exit_code == 0, res.output
        out = res.output
        assert "Cache Statistics" in out and "80.0%" in out  # 80/(80+20)
        assert "Per-pool Statistics" in out and "unified-tiny" in out

    def test_hit_rate_guards_empty_cache(self):
        assert cli._hit_rate(0, 0) == "n/a"
        assert cli._hit_rate(3, 1) == "75.0%"


class TestMcpGate:
    """`mcp` subcommands are gated on the optional biopb-mcp package via
    _require_biopb_mcp (checks the import spec, no heavy import)."""

    def test_require_raises_when_absent(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        with pytest.raises(cli.typer.Exit) as ei:
            cli._require_biopb_mcp()
        assert ei.value.exit_code == 1

    def test_require_passes_when_present(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
        cli._require_biopb_mcp()  # no raise

    def test_gate_blocks_command_when_absent(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        res = CliRunner().invoke(cli.app, ["mcp", "status"])
        assert res.exit_code == 1
        assert "biopb-mcp" in res.output and "not installed" in res.output

    def test_gate_passes_command_when_present(self, monkeypatch):
        # Spec present -> the gate is a no-op and the command proceeds (here a
        # stopped daemon, so status exits 0 with "Not running").
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (None, None))
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: False)
        res = CliRunner().invoke(cli.app, ["mcp", "status"])
        assert res.exit_code == 0
        assert "Not running" in res.output


class TestMcpLineLevel:
    """`mcp logs --level` parses the biopb-mcp log format (basicConfig/uvicorn)."""

    def test_basicconfig_and_uvicorn_formats(self):
        assert cli._mcp_line_level("INFO:biopb_mcp.mcp.__main__:Ready") == "INFO"
        assert cli._mcp_line_level("WARNING:biopb_mcp:slow") == "WARNING"
        # uvicorn pads after the level: "LEVEL:    message"
        assert cli._mcp_line_level("INFO:     127.0.0.1:8765 - GET /mcp") == "INFO"

    def test_off_format_is_unclassified(self):
        assert cli._mcp_line_level("Traceback (most recent call last):") is None
        assert cli._mcp_line_level("    File \"x.py\", line 3, in f") is None
        assert cli._mcp_line_level("") is None
        # dask's " - LEVEL - " shape is not classified (best-effort).
        assert cli._mcp_line_level("distributed.worker - INFO - busy") is None

    def test_filter_with_mcp_level_carries_continuation(self):
        lines = ["INFO:n:i", "WARNING:n:w", "  stack frame", "DEBUG:n:d"]
        assert cli._filter_lines(lines, "WARNING", cli._mcp_line_level) == [
            "WARNING:n:w",
            "  stack frame",
        ]


class TestMcpStart:
    """`mcp start` launches `python -m biopb_mcp.mcp --transport http`."""

    def _setup(self, monkeypatch, tmp_path, *, existing, existing_alive, child_alive):
        monkeypatch.setattr(cli, "_require_biopb_mcp", lambda: None)
        monkeypatch.setattr(cli, "MCP_LOG_DIR", tmp_path)
        monkeypatch.setattr(cli, "MCP_PID_FILE", tmp_path / "mcp-server.pid")
        monkeypatch.setattr(cli, "_mcp_default_port", lambda: 8765)
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (existing, None))
        # Deterministic PID-file write: no token, so the file is the bare PID.
        monkeypatch.setattr(cli, "_process_create_time", lambda _p: None)

        def running(pid):
            return existing_alive if pid == existing else child_alive

        monkeypatch.setattr(cli, "_is_process_running", running)
        proc = MagicMock()
        proc.pid = 4242
        popen = MagicMock(return_value=proc)
        monkeypatch.setattr(cli.subprocess, "Popen", popen)
        return popen

    def test_starts_and_records_pid(self, monkeypatch, tmp_path):
        popen = self._setup(
            monkeypatch, tmp_path, existing=None, existing_alive=False, child_alive=True
        )
        res = CliRunner().invoke(cli.app, ["mcp", "start"])
        assert res.exit_code == 0, res.output
        assert "started (PID 4242)" in res.output
        assert "http://127.0.0.1:8765/mcp" in res.output
        # Launches the http transport in a child process.
        cmd = popen.call_args.args[0]
        assert cmd[1:] == ["-m", "biopb_mcp.mcp", "--transport", "http", "--port", "8765"]
        assert (tmp_path / "mcp-server.pid").read_text() == "4242"

    def test_custom_port(self, monkeypatch, tmp_path):
        popen = self._setup(
            monkeypatch, tmp_path, existing=None, existing_alive=False, child_alive=True
        )
        res = CliRunner().invoke(cli.app, ["mcp", "start", "--port", "9000"])
        assert res.exit_code == 0, res.output
        assert "http://127.0.0.1:9000/mcp" in res.output
        assert popen.call_args.args[0][-1] == "9000"

    def test_already_running_is_noop(self, monkeypatch, tmp_path):
        popen = self._setup(
            monkeypatch, tmp_path, existing=999, existing_alive=True, child_alive=True
        )
        res = CliRunner().invoke(cli.app, ["mcp", "start"])
        assert res.exit_code == 0
        assert "already running" in res.output
        popen.assert_not_called()

    def test_failed_start_exits_1(self, monkeypatch, tmp_path):
        self._setup(
            monkeypatch, tmp_path, existing=None, existing_alive=False, child_alive=False
        )
        res = CliRunner().invoke(cli.app, ["mcp", "start"])
        assert res.exit_code == 1
        assert "Failed to start" in res.output


class TestMcpStop:
    def _run(self, monkeypatch, *, pid, running, graceful=True):
        monkeypatch.setattr(cli, "_require_biopb_mcp", lambda: None)
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (pid, None))
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: running)
        monkeypatch.setattr(cli, "_remove_mcp_pid", MagicMock())
        stop = MagicMock(return_value=graceful)
        monkeypatch.setattr(cli, "_stop_mcp", stop)
        res = CliRunner().invoke(cli.app, ["mcp", "stop"])
        return res, stop

    def test_not_running(self, monkeypatch):
        res, stop = self._run(monkeypatch, pid=None, running=False)
        assert res.exit_code == 0
        assert "No biopb-mcp server running" in res.output
        stop.assert_not_called()

    def test_running_stops_gracefully(self, monkeypatch):
        res, stop = self._run(monkeypatch, pid=123, running=True)
        assert res.exit_code == 0
        assert "stopped" in res.output
        stop.assert_called_once()

    def test_force_kill_reported(self, monkeypatch):
        res, stop = self._run(monkeypatch, pid=123, running=True, graceful=False)
        assert res.exit_code == 0
        assert "force killed" in res.output


class TestMcpStatus:
    def _json(self, monkeypatch, *, pid, running, listening):
        monkeypatch.setattr(cli, "_require_biopb_mcp", lambda: None)
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (pid, None))
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: running)
        monkeypatch.setattr(cli, "_mcp_default_port", lambda: 8765)
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: listening)
        res = CliRunner().invoke(cli.app, ["mcp", "status", "--json"])
        assert res.exit_code == 0, res.output
        return json.loads(res.stdout.strip().splitlines()[-1])

    def test_running_and_listening(self, monkeypatch):
        d = self._json(monkeypatch, pid=42, running=True, listening=True)
        assert d["running"] is True and d["pid"] == 42
        assert d["status"] == "running" and d["listening"] is True
        assert d["port"] == 8765 and d["url"] == "http://127.0.0.1:8765/mcp"

    def test_running_not_listening(self, monkeypatch):
        d = self._json(monkeypatch, pid=42, running=True, listening=False)
        assert d["running"] is True and d["listening"] is False

    def test_stopped(self, monkeypatch):
        d = self._json(monkeypatch, pid=None, running=False, listening=False)
        assert d["running"] is False and d["status"] == "stopped"
        assert d["pid"] is None and d["url"] is None

    def test_stale(self, monkeypatch):
        d = self._json(monkeypatch, pid=999, running=False, listening=False)
        assert d["status"] == "stale" and d["running"] is False


class TestMcpLogs:
    """`mcp logs` reads ~/.local/share/biopb-mcp/log/mcp-server.log."""

    @pytest.fixture
    def log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_require_biopb_mcp", lambda: None)
        monkeypatch.setattr(cli, "MCP_LOG_DIR", tmp_path)
        # `mcp logs` resolves its path via _get_mcp_log_file(), which prefers
        # biopb_mcp._config.get_daemon_log_file() whenever biopb-mcp is importable
        # -- bypassing MCP_LOG_DIR and reading the user's REAL daemon log. Patch the
        # actual seam so the test reads the temp file regardless of environment.
        log = tmp_path / "mcp-server.log"
        monkeypatch.setattr(cli, "_get_mcp_log_file", lambda: log)
        return log

    def _run(self, *args):
        return CliRunner().invoke(cli.app, ["mcp", "logs", *args])

    def test_path_prints_and_exits(self, log_file):
        res = self._run("--path")
        assert res.exit_code == 0
        assert str(log_file) in res.output

    def test_missing_file_is_not_an_error(self, log_file):
        res = self._run()
        assert res.exit_code == 0
        assert "No log file" in res.output

    def test_tail_last_n_lines(self, log_file):
        log_file.write_text("\n".join(f"line {i}" for i in range(20)) + "\n")
        res = self._run("-n", "5")
        out = res.output.strip().splitlines()
        assert out == ["line 15", "line 16", "line 17", "line 18", "line 19"]

    def test_level_filter_mcp_format(self, log_file):
        log_file.write_text("INFO:n:i\nDEBUG:n:d\nWARNING:n:w\nERROR:n:e\n")
        res = self._run("--level", "WARNING")
        assert res.exit_code == 0
        assert "WARNING:n:w" in res.output and "ERROR:n:e" in res.output
        assert "INFO:n:i" not in res.output and "DEBUG:n:d" not in res.output

    def test_invalid_level_exits_1(self, log_file):
        log_file.write_text("INFO:n:i\n")
        res = self._run("--level", "FOO")
        assert res.exit_code == 1
        assert "Invalid --level" in res.output


class TestPidIdentity:
    """The PID file carries a create-time identity token so a stale + reused PID
    (rife on Windows) is not mistaken for our daemon (issue #138, item 1)."""

    def test_record_roundtrip_with_token(self, tmp_path):
        f = tmp_path / "x.pid"
        cli._write_pid_file(f, 1234, 9988776655)
        assert f.read_text() == "1234\n9988776655"
        assert cli._read_pid_record(f) == (1234, 9988776655)

    def test_legacy_bare_pid_reads_with_no_token(self, tmp_path):
        # A pre-upgrade file (bare PID) must still read; token None -> liveness only.
        f = tmp_path / "x.pid"
        f.write_text("4321")
        assert cli._read_pid_record(f) == (4321, None)

    def test_missing_or_garbage_is_none(self, tmp_path):
        assert cli._read_pid_record(tmp_path / "absent.pid") == (None, None)
        f = tmp_path / "junk.pid"
        f.write_text("not-a-pid")
        assert cli._read_pid_record(f) == (None, None)

    def test_matching_token_is_our_daemon(self, monkeypatch):
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(cli, "_process_create_time", lambda _p: 555)
        assert cli._is_our_daemon(100, 555) is True

    def test_mismatched_token_is_not_ours(self, monkeypatch):
        # Alive PID, but a DIFFERENT creation time -> a reused PID, not our daemon.
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(cli, "_process_create_time", lambda _p: 999)
        assert cli._is_our_daemon(100, 555) is False

    def test_absent_token_falls_back_to_liveness(self, monkeypatch):
        # Legacy file / unsupported platform -> trust liveness (pre-fix behavior).
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(cli, "_process_create_time", lambda _p: 777)
        assert cli._is_our_daemon(100, None) is True

    def test_dead_pid_is_not_ours(self, monkeypatch):
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: False)
        assert cli._is_our_daemon(100, 555) is False

    def test_stop_skips_force_kill_on_reused_pid(self, monkeypatch):
        """`stop` on a stale PID now owned by an unrelated process must clean the
        PID file WITHOUT TerminateProcess-ing that innocent process."""
        monkeypatch.setattr("sys.platform", "linux")
        # The PID is alive, but its identity does not match the recorded token.
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (1234, 555))
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(cli, "_process_create_time", lambda _p: 999)  # reused
        remove = MagicMock()
        monkeypatch.setattr(cli, "_remove_pid", remove)
        with patch.object(cli.os, "kill") as kill:
            res = CliRunner().invoke(cli.app, ["server", "stop"])
        assert res.exit_code == 0
        kill.assert_not_called()  # the innocent reused PID is never signaled
        remove.assert_called_once()  # but the stale file is cleaned up


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
