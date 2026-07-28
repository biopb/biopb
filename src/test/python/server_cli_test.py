"""Unit tests for the biopb CLI's server / control commands and their probes.

Covers the `server cache-stats` query path, the daemon liveness/health probe,
the `control status` / `control run` argv wiring, mode resolution, and
migrate-config. The lower-level detached-daemon lifecycle helpers those commands
call now live in :mod:`biopb._lifecycle.daemon` and are tested in
``daemon_test.py``. OS calls are mocked so the tests are deterministic and fast
on any platform; time.sleep is neutralized.
"""

import json
from unittest.mock import MagicMock, patch

import biopb.cli as cli
import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Keep the wait/force-kill loops instant."""
    monkeypatch.setattr(cli.time, "sleep", lambda *_a, **_k: None)


class TestProbeDaemon:
    """`_probe_daemon` is the one liveness/health snapshot both `status` commands
    and the readiness loop share. It must never raise: a failed health RPC comes
    back health=None, a closed port listening=False."""

    def test_health_answer_defines_liveness(self, monkeypatch):
        # A daemon that answers its health RPC is, by that fact, listening.
        health = {"status": "SERVING", "source_count": 3}
        probe = cli._probe_daemon("h", 1, health_fn=lambda: health)
        assert probe.listening is True and probe.health is health

    def test_unreachable_health_is_not_listening(self, monkeypatch):
        # health_fn already swallows errors and returns None; the probe treats
        # that as down without a TCP fallback (the RPC *is* the liveness signal).
        probe = cli._probe_daemon("h", 1, health_fn=lambda: None)
        assert probe.listening is False and probe.health is None

    def test_tcp_only_when_no_health_fn(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: True)
        probe = cli._probe_daemon("127.0.0.1", 8765)
        assert probe.listening is True and probe.health is None

    def test_tcp_closed_port(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: False)
        assert cli._probe_daemon("127.0.0.1", 8765).listening is False


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

    def _run(self, monkeypatch, *, stats, args=()):
        # Liveness is the Flight query itself (no PID-file gate): an unreachable
        # server just yields no stats.
        monkeypatch.setattr(
            cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None)
        )
        monkeypatch.setattr(cli, "_query_cache_stats", lambda *_a, **_k: stats)
        return CliRunner().invoke(cli.app, ["server", "cache-stats", *args])

    def test_unreachable_exits_1(self, monkeypatch):
        # Server unreachable / cache action failed -> _query_cache_stats None.
        res = self._run(monkeypatch, stats=None)
        assert res.exit_code == 1
        assert "Could not retrieve cache stats" in res.output

    def test_json_emits_raw_dict(self, monkeypatch):
        res = self._run(monkeypatch, stats=self._STATS, args=["--json"])
        assert res.exit_code == 0, res.output
        payload = json.loads(res.stdout.strip().splitlines()[-1])
        assert payload["hits"] == 80 and payload["misses"] == 20
        assert payload["pool_stats"]["unified-tiny"]["segments"] == 2

    def test_table_renders_hit_rate_and_pools(self, monkeypatch):
        res = self._run(monkeypatch, stats=self._STATS)
        assert res.exit_code == 0, res.output
        out = res.output
        assert "Cache Statistics" in out and "80.0%" in out  # 80/(80+20)
        assert "Per-pool Statistics" in out and "unified-tiny" in out

    def test_hit_rate_guards_empty_cache(self):
        assert cli._hit_rate(0, 0) == "n/a"
        assert cli._hit_rate(3, 1) == "75.0%"

    def test_explicit_token_is_passed_through(self, monkeypatch):
        # Regression: --token must reach _query_cache_stats verbatim.
        captured = {}
        monkeypatch.setattr(
            cli, "_resolve_grpc_endpoint", lambda _c: ("grpc://x", None)
        )

        def fake_query(location, token):
            captured["token"] = token
            return self._STATS

        monkeypatch.setattr(cli, "_query_cache_stats", fake_query)
        res = CliRunner().invoke(
            cli.app, ["server", "cache-stats", "--token", "secret"]
        )
        assert res.exit_code == 0, res.output
        assert captured["token"] == "secret"


class TestQueryServerHelper:
    """_query_server is the shared body behind the status / cache-stats probes
    (biopb/biopb#277 item F): open a short-lived client, run a call, always close."""

    def _inject_client(self, monkeypatch, fake_cls):
        import sys
        import types

        mod = types.ModuleType("biopb.tensor.client")
        mod.TensorFlightClient = fake_cls
        monkeypatch.setitem(sys.modules, "biopb.tensor.client", mod)

    def test_runs_call_and_closes_client(self, monkeypatch):
        closed = []

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            def health_check(self):
                return {"ok": True}

            def close(self):
                closed.append(True)

        self._inject_client(monkeypatch, FakeClient)
        result = cli._query_server("grpc://x", "tok", lambda c: c.health_check())
        assert result == {"ok": True}
        assert closed == [True]  # closed even on the success path

    def test_call_failure_returns_none_but_still_closes(self, monkeypatch):
        closed = []

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            def close(self):
                closed.append(True)

        self._inject_client(monkeypatch, FakeClient)

        def boom(_c):
            raise RuntimeError("action failed")

        assert cli._query_server("grpc://x", None, boom) is None
        assert closed == [True]

    def test_missing_client_package_returns_none(self, monkeypatch):
        import sys

        # Simulate the tensor client being unavailable -> the inner import raises.
        monkeypatch.setitem(sys.modules, "biopb.tensor.client", None)
        assert cli._query_server("grpc://x", None, lambda c: c.health_check()) is None

    def test_wrapper_routes_to_the_right_client_method(self, monkeypatch):
        class FakeClient:
            def cache_stats(self):
                return {"m": "cache"}

        captured = {}

        def fake_query_server(location, token, call):
            captured["args"] = (location, token)
            return call(FakeClient())

        monkeypatch.setattr(cli, "_query_server", fake_query_server)
        assert cli._query_cache_stats("grpc://x", "t") == {"m": "cache"}
        assert captured["args"] == ("grpc://x", "t")


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
        # `mcp view` gates on the package (via _require_biopb_mcp) before it
        # spawns anything, so an absent package exits 1 with the install hint.
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        res = CliRunner().invoke(cli.app, ["mcp", "view"])
        assert res.exit_code == 1
        assert "biopb-mcp" in res.output and "not installed" in res.output


class TestAwaitListening:
    """Readiness probe: did the daemon actually bind, not just stay alive."""

    def test_true_when_port_comes_up(self, monkeypatch):
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: True)
        assert cli._await_listening(123, "127.0.0.1", 8815, 5.0) is True

    def test_false_when_process_dies(self, monkeypatch):
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: False)
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: True)
        assert cli._await_listening(123, "127.0.0.1", 8815, 5.0) is False

    def test_false_on_timeout(self, monkeypatch):
        monkeypatch.setattr(cli, "_is_process_running", lambda _p: True)
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: False)
        monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
        clock = iter([0.0, 0.1, 99.0])
        monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
        assert cli._await_listening(123, "127.0.0.1", 8815, 5.0) is False


class TestControlStatus:
    """`biopb control status` — pidfile + control-API /health, no biopb_control needed."""

    def _json(self, monkeypatch, *, pid, running, health):
        monkeypatch.setattr(cli, "_require_biopb_control", lambda: None)
        monkeypatch.setattr(cli, "_read_pid_record", lambda *_a: (pid, None))
        # `control status` decides liveness via _is_our_daemon (now backed by
        # _lifecycle.daemon); stub the verdict directly.
        monkeypatch.setattr(cli, "_is_our_daemon", lambda *_a: running)
        monkeypatch.setattr(cli, "_control_endpoint", lambda: ("127.0.0.1", 8813))
        monkeypatch.setattr(cli, "_query_control_health", lambda *_a, **_k: health)
        res = CliRunner().invoke(cli.app, ["control", "status", "--json"])
        assert res.exit_code == 0, res.output
        return json.loads(res.stdout.strip().splitlines()[-1])

    def test_running_with_data_plane(self, monkeypatch):
        health = {
            "control": "ok",
            "data_plane": {
                "state": "serving",
                "grpc_url": "grpc://127.0.0.1:8815",
                "restarts": 2,
            },
        }
        d = self._json(monkeypatch, pid=7, running=True, health=health)
        assert d["running"] is True and d["pid"] == 7
        assert d["control_api"] is True
        assert d["data_plane"]["state"] == "serving"
        assert d["control_url"] == "http://127.0.0.1:8813"

    def test_running_but_control_api_silent(self, monkeypatch):
        # Process alive, but /health does not answer (still booting, or wedged).
        d = self._json(monkeypatch, pid=7, running=True, health=None)
        assert d["running"] is True and d["control_api"] is False
        assert d["data_plane"] is None

    def test_stopped(self, monkeypatch):
        d = self._json(monkeypatch, pid=None, running=False, health=None)
        assert d["running"] is False and d["status"] == "stopped"

    def test_stale(self, monkeypatch):
        d = self._json(monkeypatch, pid=999, running=False, health=None)
        assert d["status"] == "stale" and d["running"] is False

    def test_help_lists_lifecycle_commands(self):
        res = CliRunner().invoke(cli.app, ["control", "--help"])
        assert res.exit_code == 0, res.output
        for cmd in ("start", "stop", "status", "run"):
            assert cmd in res.output


class TestRejectLegacyToml:
    """`control start` / `run` refuse a pre-#34 `biopb.toml` up front.

    Every config probe further in is best-effort (`_read_flight_host` fails
    *closed* to a public bind on an unreadable config), so without this gate a
    legacy config surfaces as an unrelated "public bind needs a token" refusal
    -- or as a plane serving defaults instead of the user's data.
    """

    def test_legacy_toml_exits_with_the_migration_command(self, tmp_path, capsys):
        legacy = tmp_path / "biopb.toml"
        legacy.write_text("[server]\nport = 8815\n")
        with pytest.raises(typer.Exit) as exc:
            cli._reject_legacy_toml(legacy)
        assert exc.value.exit_code == 1
        assert "migrate-config" in capsys.readouterr().out

    def test_json_config_passes(self, tmp_path):
        config = tmp_path / "biopb.json"
        config.write_text('{"server": {"port": 8815}}')
        cli._reject_legacy_toml(config)  # no raise

    def test_absent_toml_passes(self, tmp_path):
        # find_config hands back the canonical name when nothing exists; a
        # never-created .toml path must not be mistaken for a legacy install.
        cli._reject_legacy_toml(tmp_path / "biopb.toml")


class TestControlRunArgv:
    """`_control_run_argv` must never put the access token on the child command
    line -- a process command line is world-readable via `ps` / Task Manager,
    which leaks the secret on exactly the multi-user hosts it protects
    (biopb/biopb#414). The token travels via BIOPB_TENSOR_TOKEN in the env."""

    @pytest.fixture(autouse=True)
    def _stub_helpers(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            cli, "_resolve_grpc_hostport", lambda _c: ("127.0.0.1", 8815)
        )
        monkeypatch.setattr(cli, "_control_endpoint", lambda: ("127.0.0.1", 8813))
        monkeypatch.setattr(
            cli, "_get_log_file", lambda: tmp_path / "tensor-server.log"
        )
        monkeypatch.setattr(
            cli, "_control_shutdown_sentinel", lambda: tmp_path / "control.stop"
        )

    def _argv(self, tmp_path, *, remote):
        return cli._control_run_argv(
            config=tmp_path / "biopb.json",
            static_dir=None,
            web_host="127.0.0.1",
            web_port=8814,
            log_level="INFO",
            data_plane=True,
            remote=remote,
        )

    def test_token_never_on_argv(self, tmp_path):
        argv = self._argv(tmp_path, remote=True)
        assert "--token" not in argv
        # And no generated-looking secret slipped in as a bare positional.
        assert not any("BIOPB_TENSOR_TOKEN" in a for a in argv)

    def test_remote_signalled_on_argv_and_binds_control_public(self, tmp_path):
        # --remote is not a secret, so it stays explicit on the argv; it also
        # flips the control's own listener to a public bind.
        argv = self._argv(tmp_path, remote=True)
        assert "--remote" in argv
        assert argv[argv.index("--control-host") + 1] == "0.0.0.0"

    def test_local_mode_has_no_remote_flag_and_loopback_control(self, tmp_path):
        argv = self._argv(tmp_path, remote=False)
        assert "--remote" not in argv
        assert argv[argv.index("--control-host") + 1] == "127.0.0.1"


class TestResolveMode:
    """`_resolve_mode` decides the enforced token. Token enforcement is
    independent of the network mode (a token is allowed in *either*); the single
    fail-closed rule is that a public listener is never left unauthenticated:
    remote always carries a token, and a local mode with a public flight bind is
    refused unless a token is supplied."""

    @pytest.fixture(autouse=True)
    def _loopback_flight(self, monkeypatch):
        # Default the flight bind to loopback; individual tests override. Clear any
        # ambient BIOPB_TENSOR_TOKEN so "tokenless" cases resolve deterministically
        # (the resolver now reads the env token in either mode).
        monkeypatch.setattr(cli, "_read_flight_host", lambda _c: "127.0.0.1")
        monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)

    def test_local_loopback_is_tokenless_by_default(self, tmp_path):
        assert cli._resolve_mode(tmp_path / "c.json", remote=False, token=None) is None

    def test_local_accepts_explicit_token(self, tmp_path):
        # A token is now allowed in local mode (defense-in-depth on a shared
        # machine); it is enforced across the loopback-bound listeners.
        assert (
            cli._resolve_mode(
                tmp_path / "c.json", remote=False, token="local-token-0123456"
            )
            == "local-token-0123456"
        )

    def test_local_reads_env_token(self, tmp_path, monkeypatch):
        # The token travels via BIOPB_TENSOR_TOKEN in either mode; local mode now
        # honors it too (matching what the supervised child already enforces).
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token-0123456789")
        assert (
            cli._resolve_mode(tmp_path / "c.json", remote=False, token=None)
            == "env-token-0123456789"
        )

    def test_local_rejects_malformed_token(self, tmp_path):
        # A supplied-but-malformed token is refused in local mode too, so it is
        # never silently ignored downstream (which would leave the listeners open).
        with pytest.raises(typer.Exit):
            cli._resolve_mode(tmp_path / "c.json", remote=False, token="too-short")

    def test_public_flight_refused_when_tokenless(self, tmp_path, monkeypatch):
        # Fail-closed: a config that binds the flight server publicly must not run
        # tokenless. This is the guard that makes "public + open" unrepresentable.
        monkeypatch.setattr(cli, "_read_flight_host", lambda _c: "0.0.0.0")
        with pytest.raises(typer.Exit):
            cli._resolve_mode(tmp_path / "c.json", remote=False, token=None)

    def test_local_public_flight_bind_allowed_with_token(self, tmp_path, monkeypatch):
        # A token satisfies the fail-closed guard: a public flight bind behind a
        # token is authenticated, so it is representable without --remote.
        monkeypatch.setattr(cli, "_read_flight_host", lambda _c: "0.0.0.0")
        assert (
            cli._resolve_mode(
                tmp_path / "c.json", remote=False, token="local-token-0123456"
            )
            == "local-token-0123456"
        )

    def test_remote_uses_supplied_token(self, tmp_path):
        assert (
            cli._resolve_mode(
                tmp_path / "c.json", remote=True, token="supplied-token-0123"
            )
            == "supplied-token-0123"
        )

    def test_remote_reads_env_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token-0123456789")
        assert (
            cli._resolve_mode(tmp_path / "c.json", remote=True, token=None)
            == "env-token-0123456789"
        )

    def test_remote_rejects_malformed_token(self, tmp_path):
        # Validated with the shared `_web_auth.valid_token` rule the tensor
        # `launch` also applies, so the layers can't drift: a too-short (or
        # non-URL-safe) token is refused here rather than silently regenerated
        # downstream, which would leave the browser holding a rejected token.
        with pytest.raises(typer.Exit):
            cli._resolve_mode(tmp_path / "c.json", remote=True, token="too-short")

    def test_remote_generates_token_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)
        tok = cli._resolve_mode(tmp_path / "c.json", remote=True, token=None)
        assert tok and len(tok) >= 16


class TestMigrateConfig:
    """`biopb server migrate-config`: legacy biopb.toml -> canonical biopb.json."""

    _TOML = (
        "[server]\n"
        'host = "127.0.0.1"\n'
        "port = 8815\n\n"
        "[cache]\n"
        "max_bytes = 3000000000\n\n"
        "[[sources]]\n"
        'url = "/data/microscopy"\n'
        "monitor = true\n\n"
        "# advanced/unknown key that must survive the migration\n"
        "[experimental]\n"
        'foo = "bar"\n'
    )

    def _run(self, config_dir, *extra):
        return CliRunner().invoke(
            cli.app, ["server", "migrate-config", "--config", str(config_dir), *extra]
        )

    def test_migrates_toml_and_preserves_unknown_keys(self, tmp_path):
        # The actual migration reuses biopb_tensor_server.core.config (save_config /
        # read_legacy_toml). That package ships only with the full installer,
        # not on PyPI, so it is absent from the lightweight `biopb[test,tensor]`
        # CI env -- skip there; the command's own "unavailable" fallback is what
        # runs in that case.
        pytest.importorskip("biopb_tensor_server")
        (tmp_path / "biopb.toml").write_text(self._TOML)
        res = self._run(tmp_path)
        assert res.exit_code == 0, res.output

        json_path = tmp_path / "biopb.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["server"]["port"] == 8815
        assert data["cache"]["max_bytes"] == 3000000000
        assert data["sources"][0]["url"] == "/data/microscopy"
        # The unknown table survives (raw-dict round-trip, not dataclass).
        assert data["experimental"] == {"foo": "bar"}
        # Legacy file retired to .bak; schema sidecar written.
        assert (tmp_path / "biopb.toml.bak").exists()
        assert not (tmp_path / "biopb.toml").exists()
        assert (tmp_path / "biopb.schema.json").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        pytest.importorskip("biopb_tensor_server")  # see note above
        (tmp_path / "biopb.toml").write_text(self._TOML)
        res = self._run(tmp_path, "--dry-run")
        assert res.exit_code == 0, res.output
        assert (tmp_path / "biopb.toml").exists()  # untouched
        assert not (tmp_path / "biopb.json").exists()
        assert not (tmp_path / "biopb.toml.bak").exists()

    def test_already_json_is_noop(self, tmp_path):
        (tmp_path / "biopb.json").write_text('{"server": {"port": 8815}}')
        res = self._run(tmp_path)
        assert res.exit_code == 0
        assert "Already canonical" in res.output
        assert not (tmp_path / "biopb.toml.bak").exists()

    def test_both_present_retires_toml_without_touching_json(self, tmp_path):
        (tmp_path / "biopb.toml").write_text("[server]\nport = 8815\n")
        # A JSON that must be left byte-for-byte untouched (it already wins).
        original = '{"server": {"port": 9999}}'
        (tmp_path / "biopb.json").write_text(original)
        res = self._run(tmp_path)
        assert res.exit_code == 0, res.output
        assert (tmp_path / "biopb.json").read_text() == original  # untouched
        assert (tmp_path / "biopb.toml.bak").exists()
        assert not (tmp_path / "biopb.toml").exists()

    def test_no_config_present(self, tmp_path):
        res = self._run(tmp_path)
        assert res.exit_code == 0
        assert "No legacy config found" in res.output

    def test_config_pointing_at_file_uses_its_dir(self, tmp_path):
        pytest.importorskip("biopb_tensor_server")  # see note above
        # --config may name the file itself, not just the directory.
        toml = tmp_path / "biopb.toml"
        toml.write_text(self._TOML)
        res = self._run(toml)
        assert res.exit_code == 0, res.output
        assert (tmp_path / "biopb.json").exists()


class TestDashboardCommand:
    """`biopb dashboard`: ensure the control plane is up, then open the browser.

    The command is the desktop shortcut's target and the one-liner install
    summary points users at. It reuses `control_start` for the actual startup,
    so these tests mock both the port probe and the browser to stay hermetic.
    """

    def test_already_running_opens_browser_without_starting(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: True)
        start = MagicMock()
        monkeypatch.setattr(cli, "control_start", start)
        opened = []
        with patch("webbrowser.open", lambda url: opened.append(url) or True):
            res = CliRunner().invoke(cli.app, ["dashboard"])
        assert res.exit_code == 0, res.output
        start.assert_not_called()  # already up -> no start attempt
        assert opened == ["http://127.0.0.1:8813"]

    def test_starts_control_plane_when_not_listening(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: False)
        start = MagicMock(side_effect=typer.Exit(0))
        monkeypatch.setattr(cli, "control_start", start)
        opened = []
        with patch("webbrowser.open", lambda url: opened.append(url) or True):
            res = CliRunner().invoke(cli.app, ["dashboard"])
        assert res.exit_code == 0, res.output
        start.assert_called_once()
        # remote defaults off; local mode carries no token.
        assert start.call_args.kwargs["remote"] is False
        assert opened == ["http://127.0.0.1:8813"]

    def test_start_failure_aborts_without_opening_browser(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: False)
        monkeypatch.setattr(cli, "control_start", MagicMock(side_effect=typer.Exit(1)))
        opened = []
        with patch("webbrowser.open", lambda url: opened.append(url) or True):
            res = CliRunner().invoke(cli.app, ["dashboard"])
        assert res.exit_code == 1
        assert opened == []  # never point a browser at a dead URL

    def test_no_browser_prints_url_only(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: True)
        monkeypatch.setattr(cli, "control_start", MagicMock())
        opened = []
        with patch("webbrowser.open", lambda url: opened.append(url) or True):
            res = CliRunner().invoke(cli.app, ["dashboard", "--no-browser"])
        assert res.exit_code == 0, res.output
        assert opened == []
        assert "http://127.0.0.1:8813" in res.output

    def test_remote_flag_forwarded_to_control_start(self, monkeypatch):
        monkeypatch.setattr(cli, "_port_listening", lambda *_a, **_k: False)
        start = MagicMock(side_effect=typer.Exit(0))
        monkeypatch.setattr(cli, "control_start", start)
        with patch("webbrowser.open", lambda url: True):
            res = CliRunner().invoke(cli.app, ["dashboard", "--remote"])
        assert res.exit_code == 0, res.output
        assert start.call_args.kwargs["remote"] is True


class TestVersionCommand:
    """`biopb version` reports the two version lines: the product deployment
    (`release`, from the installer's marker file — the shared release-v* version
    of tensor-server / mcp / control / web) and the `biopb` SDK (its own v* line).
    The release line is deliberately NOT any single wheel's version."""

    @staticmethod
    def _labels(output: str) -> dict:
        """Map each `label: value` line to its value, collapsing the alignment
        padding the command inserts between the label and the version."""
        out = {}
        for line in output.splitlines():
            if ":" in line:
                label, _, value = line.partition(":")
                out[label.strip()] = value.strip()
        return out

    def test_reports_release_and_sdk(self, monkeypatch, tmp_path):
        # Two lines only: the product deployment (marker) and the biopb SDK.
        marker = tmp_path / "release.version"
        marker.write_text("1.2.3\n")
        monkeypatch.setattr(cli, "_RELEASE_VERSION_FILE", marker)
        monkeypatch.setattr(
            cli,
            "_package_version",
            lambda name: {"biopb": "0.9.3"}.get(name, "not installed"),
        )

        res = CliRunner().invoke(cli.app, ["version"])

        assert res.exit_code == 0, res.output
        labels = self._labels(res.output)
        # Deployment version is the marker's contents (the release-v* product
        # line), distinct from the biopb SDK's own v* version.
        assert labels["release"] == "1.2.3"
        assert labels["biopb"] == "0.9.3"
        # The product wheels are no longer listed individually — they all share
        # the release version, so the marker stands in for the set.
        assert set(labels) == {"release", "biopb"}
        assert "biopb-tensor-server" not in labels
        assert "biopb-mcp" not in labels

    def test_release_version_unknown_when_marker_absent(self, monkeypatch, tmp_path):
        # A dev checkout / non-installer setup has no marker: report 'unknown',
        # never crash.
        monkeypatch.setattr(cli, "_RELEASE_VERSION_FILE", tmp_path / "missing.version")

        res = CliRunner().invoke(cli.app, ["version"])

        assert res.exit_code == 0, res.output
        assert self._labels(res.output)["release"] == "unknown"

    def test_read_release_version_strips_marker_contents(self, monkeypatch, tmp_path):
        marker = tmp_path / "release.version"
        # The installer writes no trailing newline; be tolerant of both.
        marker.write_text("  9.9.9\n")
        monkeypatch.setattr(cli, "_RELEASE_VERSION_FILE", marker)
        assert cli._read_release_version() == "9.9.9"

    def test_read_release_version_corrupt_marker_is_unknown(
        self, monkeypatch, tmp_path
    ):
        # A corrupt (non-UTF-8) marker must degrade to 'unknown', not raise a
        # UnicodeDecodeError out of the command -- reading a version is
        # best-effort, like _package_version.
        marker = tmp_path / "release.version"
        marker.write_bytes(b"\xff\xfe\x00bad")
        monkeypatch.setattr(cli, "_RELEASE_VERSION_FILE", marker)
        assert cli._read_release_version() == "unknown"

    def test_package_version_missing_is_not_installed(self):
        # A distribution name that is guaranteed absent maps to 'not installed'.
        assert (
            cli._package_version("biopb-definitely-not-a-real-package")
            == "not installed"
        )

    def test_package_version_present_returns_metadata_version(self):
        # biopb itself is always installed in the test env; its reported version
        # matches importlib.metadata, confirming we read metadata not an import.
        from importlib.metadata import version as _v

        assert cli._package_version("biopb") == _v("biopb")


class TestControlLogs:
    """`biopb control logs`: which file it reads, and the --level filter."""

    def _write(self, tmp_path, monkeypatch, *, control="", data_plane=""):
        c, d = tmp_path / "control.log", tmp_path / "tensor-server.log"
        c.write_text(control)
        d.write_text(data_plane)
        monkeypatch.setattr(cli, "_control_log_file", lambda: c)
        monkeypatch.setattr(cli, "_get_log_file", lambda: d)
        return c, d

    def test_defaults_to_the_control_log(self, tmp_path, monkeypatch):
        self._write(
            tmp_path, monkeypatch, control="control line", data_plane="plane line"
        )
        res = CliRunner().invoke(cli.app, ["control", "logs"])
        assert res.exit_code == 0, res.output
        assert "control line" in res.output
        assert "plane line" not in res.output

    def test_data_plane_selects_the_tensor_server_log(self, tmp_path, monkeypatch):
        self._write(
            tmp_path, monkeypatch, control="control line", data_plane="plane line"
        )
        res = CliRunner().invoke(cli.app, ["control", "logs", "--data-plane"])
        assert res.exit_code == 0, res.output
        assert "plane line" in res.output
        assert "control line" not in res.output

    def test_path_prints_the_file_and_exits(self, tmp_path, monkeypatch):
        c, d = self._write(tmp_path, monkeypatch)
        res = CliRunner().invoke(cli.app, ["control", "logs", "--path"])
        assert res.exit_code == 0 and str(c) in res.output
        res = CliRunner().invoke(cli.app, ["control", "logs", "--path", "--data-plane"])
        assert res.exit_code == 0 and str(d) in res.output

    def test_missing_log_is_reported_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_control_log_file", lambda: tmp_path / "absent.log")
        res = CliRunner().invoke(cli.app, ["control", "logs"])
        # Never started is a normal state, not a failure.
        assert res.exit_code == 0
        assert "No log file" in res.output

    def test_lines_takes_the_tail(self, tmp_path, monkeypatch):
        self._write(
            tmp_path, monkeypatch, control="\n".join(f"L{i}" for i in range(10))
        )
        res = CliRunner().invoke(cli.app, ["control", "logs", "-n", "3"])
        assert res.exit_code == 0, res.output
        assert "L9" in res.output and "L6" not in res.output

    def test_zero_lines_shows_all(self, tmp_path, monkeypatch):
        self._write(
            tmp_path, monkeypatch, control="\n".join(f"L{i}" for i in range(10))
        )
        res = CliRunner().invoke(cli.app, ["control", "logs", "-n", "0"])
        assert res.exit_code == 0, res.output
        assert "L0" in res.output and "L9" in res.output

    def test_bad_level_exits_1(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, control="anything")
        res = CliRunner().invoke(cli.app, ["control", "logs", "--level", "LOUD"])
        assert res.exit_code == 1
        assert "Invalid --level" in res.output

    def test_level_filters_control_format_and_carries_continuations(
        self, tmp_path, monkeypatch
    ):
        # Both shapes control.log actually carries: the control's basicConfig
        # (level in the 3rd token) and uvicorn's (`LEVEL:` first). The unleveled
        # traceback line must ride along with the ERROR record it belongs to.
        log = "\n".join(
            [
                "2026-06-12 10:00:00,123 INFO biopb_control._run: booting",
                "INFO:     Uvicorn running on http://127.0.0.1:8813",
                "2026-06-12 10:00:01,000 ERROR biopb_control._run: tick failed",
                "  File 'x.py', line 1, in tick",
                "2026-06-12 10:00:02,000 INFO biopb_control._run: recovered",
            ]
        )
        self._write(tmp_path, monkeypatch, control=log)
        res = CliRunner().invoke(cli.app, ["control", "logs", "--level", "warning"])
        assert res.exit_code == 0, res.output
        assert "tick failed" in res.output
        assert "line 1, in tick" in res.output  # continuation carried
        assert "booting" not in res.output
        assert "Uvicorn running" not in res.output
        assert "recovered" not in res.output

    def test_level_filters_the_data_plane_format(self, tmp_path, monkeypatch):
        # tensor-server.log's own format is bracketed-timestamp; the supervisor's
        # banner has no level and (leading the file) is kept by carry-forward.
        log = "\n".join(
            [
                "--- control: starting data plane at 2026-06-12 10:00:00 ---",
                "[2026-06-12 10:00:00] INFO biopb_tensor_server.server: serving",
                "[2026-06-12 10:00:01] ERROR biopb_tensor_server.server: boom",
            ]
        )
        self._write(tmp_path, monkeypatch, data_plane=log)
        res = CliRunner().invoke(
            cli.app, ["control", "logs", "--data-plane", "--level", "ERROR"]
        )
        assert res.exit_code == 0, res.output
        assert "boom" in res.output
        assert "serving" not in res.output
