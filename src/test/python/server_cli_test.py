"""Unit tests for the biopb CLI's server / control commands and their probes.

Covers the `server cache-stats` query path, the daemon liveness/health probe,
the `control status` / `control run` argv wiring, mode resolution, and
migrate-config. The lower-level detached-daemon lifecycle helpers those commands
call now live in :mod:`biopb._lifecycle.daemon` and are tested in
``daemon_test.py``. OS calls are mocked so the tests are deterministic and fast
on any platform; time.sleep is neutralized.
"""

import json
import os
from unittest.mock import MagicMock, patch

import biopb.cli as cli
import pytest
import typer
from biopb import _locations
from biopb._lifecycle import daemon as _daemon
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

    Every config probe further in is best-effort, so without this gate a legacy
    config surfaces as a plane serving defaults instead of the user's data.
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
        monkeypatch.delenv("BIOPB_CONTROL_PORT", raising=False)
        monkeypatch.delenv("BIOPB_CONTROL_HOST", raising=False)
        monkeypatch.setattr(
            cli, "_get_log_file", lambda: tmp_path / "tensor-server.log"
        )
        monkeypatch.setattr(
            cli, "_control_shutdown_sentinel", lambda: tmp_path / "control.stop"
        )

    def _argv(self, tmp_path, *, grpc_bind):
        return cli._control_run_argv(
            config=tmp_path / "biopb.json",
            static_dir=None,
            web_host="127.0.0.1",
            base_port=8810,
            log_level="INFO",
            data_plane=True,
            grpc_bind=grpc_bind,
        )

    def test_token_never_on_argv(self, tmp_path):
        argv = self._argv(tmp_path, grpc_bind="0.0.0.0")
        assert "--token" not in argv
        # And no generated-looking secret slipped in as a bare positional.
        assert not any("BIOPB_TENSOR_TOKEN" in a for a in argv)

    def test_public_plane_leaves_the_control_on_loopback(self, tmp_path):
        # --grpc-bind publishes the *flight* plane only: the control's own
        # listener stays on loopback (biopb/biopb#614) because it is plaintext
        # HTTP with no TLS support, so a public bind would carry the data/admin
        # token in the clear.
        argv = self._argv(tmp_path, grpc_bind="0.0.0.0")
        assert argv[argv.index("--grpc-host") + 1] == "0.0.0.0"
        assert argv[argv.index("--control-host") + 1] == "127.0.0.1"

    def test_no_remote_flag_is_forwarded(self, tmp_path):
        """--grpc-host carries the fact --remote used to signal.

        The child re-derives "is this deployment public?" from the address with
        the same shared predicate, so the two layers cannot disagree about it.
        """
        assert "--remote" not in self._argv(tmp_path, grpc_bind="0.0.0.0")
        assert "--remote" not in self._argv(tmp_path, grpc_bind="127.0.0.1")

    def test_loopback_binds_everything_locally(self, tmp_path):
        argv = self._argv(tmp_path, grpc_bind="127.0.0.1")
        assert argv[argv.index("--grpc-host") + 1] == "127.0.0.1"
        assert argv[argv.index("--control-host") + 1] == "127.0.0.1"
        assert argv[argv.index("--web-host") + 1] == "127.0.0.1"


class TestUiTunnelHint:
    """With the UI off the network, the SSH tunnel is the supported way to reach
    it off-box -- so `--remote` prints the exact command rather than leaving it as
    folklore (biopb/biopb#614)."""

    def test_prints_a_copyable_forward_for_the_control_port(self, capsys):
        cli._print_ui_tunnel_hint(8813)
        out = capsys.readouterr().out
        assert "ssh -L 8813:localhost:8813 " in out
        assert "http://localhost:8813" in out

    def test_honors_a_non_default_control_port(self, capsys):
        cli._print_ui_tunnel_hint(19999)
        out = capsys.readouterr().out
        assert "ssh -L 19999:localhost:19999 " in out


class TestResolveMode:
    """`_resolve_mode` decides the enforced token from the flight bind alone.

    Token enforcement is independent of the bind (a token is allowed with
    *either*); the single fail-closed rule is that a public listener is never
    left unauthenticated. There is no "config binds publicly but no token" case
    to refuse (biopb/biopb#604): the address is the CLI's, read once through the
    same `_web_auth.host_is_public_bind` the tensor `launch` and the control's
    own guard use, so the three cannot drift."""

    @pytest.fixture(autouse=True)
    def _no_ambient_token(self, monkeypatch):
        # Clear any ambient BIOPB_TENSOR_TOKEN so "tokenless" cases resolve
        # deterministically (the resolver reads the env token with either bind).
        monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)

    def test_loopback_is_tokenless_by_default(self):
        assert cli._resolve_mode("127.0.0.1", token=None) is None

    def test_loopback_accepts_explicit_token(self):
        # A token is allowed on a loopback bind (defense-in-depth on a shared
        # machine); it is enforced across the loopback-bound listeners.
        assert (
            cli._resolve_mode("127.0.0.1", token="local-token-0123456")
            == "local-token-0123456"
        )

    def test_loopback_reads_env_token(self, monkeypatch):
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token-0123456789")
        assert cli._resolve_mode("127.0.0.1", token=None) == "env-token-0123456789"

    def test_loopback_rejects_malformed_token(self):
        # A supplied-but-malformed token is refused on a loopback bind too, so it
        # is never silently ignored downstream (which would leave listeners open).
        with pytest.raises(typer.Exit):
            cli._resolve_mode("127.0.0.1", token="too-short")

    def test_public_uses_supplied_token(self):
        assert (
            cli._resolve_mode("0.0.0.0", token="supplied-token-0123")
            == "supplied-token-0123"
        )

    def test_public_reads_env_token(self, monkeypatch):
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token-0123456789")
        assert cli._resolve_mode("0.0.0.0", token=None) == "env-token-0123456789"

    def test_public_rejects_malformed_token(self):
        # Validated with the shared `_web_auth.valid_token` rule the tensor
        # `launch` also applies, so the layers can't drift: a too-short (or
        # non-URL-safe) token is refused here rather than silently regenerated
        # downstream, which would leave the browser holding a rejected token.
        with pytest.raises(typer.Exit):
            cli._resolve_mode("0.0.0.0", token="too-short")

    def test_public_generates_token_when_absent(self, monkeypatch):
        monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)
        tok = cli._resolve_mode("0.0.0.0", token=None)
        assert tok and len(tok) >= 16

    def test_a_specific_public_ip_also_requires_a_token(self, monkeypatch):
        """Not just the wildcard: any non-loopback address is public.

        `--grpc-bind 10.0.0.5` (one interface, e.g. a VPN) is exactly as
        reachable as 0.0.0.0, and `host_is_public_bind` is fail-closed on
        anything it does not recognize as loopback.
        """
        monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)
        assert cli._resolve_mode("10.0.0.5", token=None)


class TestBindDrivesTls:
    """The bind decides the TLS default, not the other way round.

    Coupling it this way keeps each flag's name matching its own effect, and
    makes the dangerous combination the one you have to ask for by name: a public
    bind with `--no-tls` puts the access token on the wire in cleartext on every
    gRPC call -- biopb/biopb#614's objection to the control, transplanted onto the
    data plane. It stays possible (a trusted intranet is real) but is spelled out.
    """

    def test_public_bind_defaults_tls_on(self):
        assert cli._resolve_tls(None, "0.0.0.0") is True
        assert cli._resolve_tls(None, "10.0.0.5") is True

    def test_loopback_defaults_tls_off(self):
        assert cli._resolve_tls(None, "127.0.0.1") is False

    def test_explicit_flags_win_both_ways(self):
        # --tls alone still means "encrypted, loopback only", which is what
        # exercising the TOFU pinning / SAN-verification paths needs.
        assert cli._resolve_tls(True, "127.0.0.1") is True
        assert cli._resolve_tls(False, "0.0.0.0") is False

    def test_public_plaintext_warns(self, capsys):
        cli._warn_public_plaintext("0.0.0.0", tls=False)
        assert "cleartext" in capsys.readouterr().out

    def test_no_warning_when_encrypted_or_loopback(self, capsys):
        cli._warn_public_plaintext("0.0.0.0", tls=True)
        cli._warn_public_plaintext("127.0.0.1", tls=False)
        assert capsys.readouterr().out == ""


class TestRemoteAlias:
    """`--remote` survives as a deprecated alias for `--grpc-bind 0.0.0.0`.

    It named a *mode* when it also published the browser UI; since
    biopb/biopb#614 it sets one address, so the flag is named for that now. The
    alias stays because `--remote` is in install scripts, service units, and
    every doc.
    """

    def test_remote_maps_to_the_public_wildcard(self, capsys):
        assert cli._resolve_grpc_bind(None, remote=True) == "0.0.0.0"
        assert "deprecated" in capsys.readouterr().out

    def test_default_is_loopback(self):
        assert cli._resolve_grpc_bind(None, remote=False) == "127.0.0.1"

    def test_explicit_bind_wins_over_the_alias(self, capsys):
        # Naming an address is more specific than asking for "public".
        assert cli._resolve_grpc_bind("127.0.0.1", remote=True) == "127.0.0.1"
        assert "ignored" in capsys.readouterr().out


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


class TestPlaneBind:
    """The flight plane's bind is the CLI's, from `--grpc-bind` (biopb/biopb#604).

    It used to be read from `biopb.json`'s `server.host`, which made the
    deployment's exposure a property of a file the control snapshotted at
    startup: a config edit could disagree with the running plane, and "local
    mode" was public whenever the config said so. It is now one address the
    caller types, and every downstream decision -- token required, TLS on by
    default -- derives from that one address.
    """

    def test_default_binds_loopback(self):
        assert cli._plane_bind("127.0.0.1", 8810)[0] == "127.0.0.1"

    def test_public_bind_is_carried_through(self):
        assert cli._plane_bind("0.0.0.0", 8810)[0] == "0.0.0.0"
        # A specific interface is expressible too, not just the wildcard.
        assert cli._plane_bind("10.0.0.5", 8810)[0] == "10.0.0.5"

    def test_probes_always_target_loopback(self):
        # A wildcard is a bind target, not a connect target.
        assert cli._probe_hostport("0.0.0.0", 8810)[0] == "127.0.0.1"
        assert cli._probe_hostport("127.0.0.1", 8810)[0] == "127.0.0.1"

    def test_a_config_bind_is_not_consulted(self, tmp_path, monkeypatch):
        """The decisive property: nothing on this path reads the config."""
        config = tmp_path / "biopb.json"
        config.write_text('{"server": {"host": "0.0.0.0", "port": 9999}}')
        monkeypatch.setattr(cli, "_get_log_file", lambda: tmp_path / "s.log")
        monkeypatch.setattr(
            cli, "_control_shutdown_sentinel", lambda: tmp_path / "c.stop"
        )
        argv = cli._control_run_argv(
            config=config,
            static_dir=None,
            web_host="127.0.0.1",
            base_port=8810,
            log_level="INFO",
            data_plane=True,
            grpc_bind="127.0.0.1",
        )
        assert argv[argv.index("--grpc-host") + 1] == "127.0.0.1"
        assert argv[argv.index("--grpc-port") + 1] == "8815"

    def test_tls_is_signalled_on_the_child_argv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_get_log_file", lambda: tmp_path / "s.log")
        monkeypatch.setattr(
            cli, "_control_shutdown_sentinel", lambda: tmp_path / "c.stop"
        )
        kwargs = {
            "config": tmp_path / "biopb.json",
            "static_dir": None,
            "web_host": "127.0.0.1",
            "base_port": 8810,
            "log_level": "INFO",
            "data_plane": True,
            "grpc_bind": "127.0.0.1",
        }
        assert "--tls" not in cli._control_run_argv(**kwargs)
        assert "--tls" in cli._control_run_argv(**kwargs, tls=True)


class TestBasePort:
    """One number places all three listeners (base+3 / +4 / +5).

    The offsets are the container's (`entrypoint.sh`: BIOPB_BASE_PORT, sidecar
    +4, gRPC +5) extended with the control at +3 -- deliberately not a second
    scheme that would agree at the defaults and diverge the moment either base
    moved.
    """

    def test_default_base_reproduces_the_historical_ports(self):
        base = cli._endpoints.BASE_DEFAULT_PORT
        assert base == 8810
        assert cli._endpoints.control_port_for(base) == 8813
        assert cli._sidecar_port(base) == 8814
        assert cli._flight_port(base) == 8815

    def test_a_moved_base_moves_all_three_together(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BIOPB_CONTROL_PORT", raising=False)
        monkeypatch.setattr(cli, "_get_log_file", lambda: tmp_path / "s.log")
        monkeypatch.setattr(
            cli, "_control_shutdown_sentinel", lambda: tmp_path / "c.stop"
        )
        argv = cli._control_run_argv(
            config=tmp_path / "biopb.json",
            static_dir=None,
            web_host="127.0.0.1",
            base_port=9000,
            log_level="INFO",
            data_plane=True,
            grpc_bind="127.0.0.1",
        )
        assert argv[argv.index("--control-port") + 1] == "9003"
        assert argv[argv.index("--web-port") + 1] == "9004"
        assert argv[argv.index("--grpc-port") + 1] == "9005"

    def test_bind_never_follows_a_published_record(self, tmp_path, monkeypatch):
        """A crashed control's stale record must not dictate the next bind.

        `_control_endpoint` (discovery) reads the record; `_control_bind_endpoint`
        must not, or a control that died on 9003 would drag every later start off
        the port its own --base-port names.
        """
        monkeypatch.delenv("BIOPB_CONTROL_PORT", raising=False)
        # XDG_STATE_HOME is what relocates the state tree; there is no
        # BIOPB_STATE_HOME. Setting a name biopb does not read leaves the tree
        # where it was, so the record lands in the developer's real state dir
        # and every client then discovers a dead port -- silently, since the
        # test still passes. See endpoints_test's fixture.
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        cli._endpoints.write_runtime_record("127.0.0.1", 9999, 4242)
        assert cli._control_endpoint()[1] == 9999  # discovery follows it
        assert cli._control_bind_endpoint(8810)[1] == 8813  # the bind does not

    def test_env_still_overrides_the_derived_control_port(self, monkeypatch):
        monkeypatch.setenv("BIOPB_CONTROL_PORT", "7777")
        assert cli._control_bind_endpoint(9000)[1] == 7777


class TestLiveForegroundControl:
    """`control status` / `control stop` recognizing a foreground `control run`.

    Such a control writes no pid file, so its endpoint record is the only trace
    of it. A clean stop retracts the record, which leaves a *crash* as the way to
    strand one -- so the pid it carries has to be checked for identity, not just
    liveness, or a recycled pid makes `status` report a dead control as Running
    and `stop` refuse to touch the daemon it was asked about.
    """

    @pytest.fixture(autouse=True)
    def _isolated_state(self, tmp_path, monkeypatch):
        # XDG_STATE_HOME, the real one -- see endpoints_test's fixture for what
        # a wrong name silently does to the developer's own state dir.
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    def test_no_record_is_no_foreground_control(self):
        assert cli._live_foreground_control() is None

    def test_live_pid_with_matching_token_is_recognized(self):
        cli._endpoints.write_runtime_record("127.0.0.1", 9003, os.getpid())
        live = cli._live_foreground_control()
        assert live is not None
        record, pid = live
        assert pid == os.getpid()
        assert record["port"] == 9003

    def _repoint_token(self, value):
        """Rewrite the record's create-time token, leaving the live pid alone."""
        path = _locations.control_runtime_file()
        record = json.loads(path.read_text())
        record["create_time"] = value
        path.write_text(json.dumps(record))

    def test_recycled_pid_is_not_a_running_control(self, monkeypatch):
        """The case the pid alone cannot see: alive, but a different process.

        The *live* token is stubbed rather than read, because whether one exists
        is platform-dependent -- macOS has none, and identity there legitimately
        degrades to liveness (see the next test). Stubbing keeps the wiring that
        matters, recorded token vs live token, covered on every platform.
        """
        monkeypatch.setattr(_daemon, "_process_create_time", lambda _p: 111)
        cli._endpoints.write_runtime_record("127.0.0.1", 9003, os.getpid())
        self._repoint_token(222)  # some other process holds this pid now
        assert cli._live_foreground_control() is None

    def test_platform_without_create_time_degrades_to_liveness(self, monkeypatch):
        """No identity source (macOS) must not read as "stopped".

        Stranding a live control is the worse error of the two: the fallback
        risks believing a recycled pid, refusing to believe a running control
        would send the user chasing a process that is serving fine.
        """
        monkeypatch.setattr(_daemon, "_process_create_time", lambda _p: None)
        cli._endpoints.write_runtime_record("127.0.0.1", 9003, os.getpid())
        self._repoint_token(222)  # unverifiable, so not disqualifying
        live = cli._live_foreground_control()
        assert live is not None and live[1] == os.getpid()

    def test_dead_pid_is_not_a_running_control(self, monkeypatch):
        cli._endpoints.write_runtime_record("127.0.0.1", 9003, os.getpid())
        monkeypatch.setattr(_daemon, "_is_process_running", lambda _p: False)
        assert cli._live_foreground_control() is None

    def test_tokenless_record_degrades_to_liveness(self):
        """A record from before the token existed must not read as "stopped"."""
        cli._endpoints.write_runtime_record("127.0.0.1", 9003, os.getpid())
        path = _locations.control_runtime_file()
        record = json.loads(path.read_text())
        del record["create_time"]  # the shape written before this field existed
        path.write_text(json.dumps(record))
        live = cli._live_foreground_control()
        assert live is not None and live[1] == os.getpid()


class TestTlsExtraPreflight:
    """`--tls` is checked before anything is spawned (biopb/biopb#604).

    `cryptography` is an opt-in extra, so a default install cannot mint the
    self-signed cert `--tls` needs. Without this check the control starts and
    reports success, then its supervised plane exits 2 on every spawn and
    crash-loops on backoff with the one useful sentence buried in
    tensor-server.log -- a control that started and a plane that never serves.
    """

    def _cryptography(self, monkeypatch, *, installed: bool):
        """Force the answer for `cryptography` only, leaving other lookups real.

        Both directions are stubbed rather than read off the ambient
        environment: whether the [tls] extra is present is exactly what varies
        between a dev venv (synced --all-extras) and a default install (CI), so
        an unstubbed test asserts the environment, not the code.
        """
        import importlib.util

        real = importlib.util.find_spec
        spec = real("importlib.util") if installed else None
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: (
                spec if name == "cryptography" else real(name, *a, **k)
            ),
        )

    def _without_cryptography(self, monkeypatch):
        self._cryptography(monkeypatch, installed=False)

    def test_passes_when_the_extra_is_installed(self, monkeypatch):
        self._cryptography(monkeypatch, installed=True)
        cli._require_tls_extra()  # no exception

    def test_exits_2_with_an_install_hint(self, monkeypatch, capsys):
        self._without_cryptography(monkeypatch)
        with pytest.raises(typer.Exit) as exc:
            cli._require_tls_extra()
        assert exc.value.exit_code == 2
        out = capsys.readouterr().out
        assert "cryptography" in out
        assert "pip install" in out

    def test_the_install_command_survives_a_narrow_terminal(self, monkeypatch):
        """The one line the reader copies verbatim must not be wrapped or eaten.

        Rich reads a bare `[tls]` as a style tag (so the extra silently vanishes)
        and hard-wraps at the terminal width (so the command splits mid-path).
        """
        import io

        from rich.console import Console

        self._without_cryptography(monkeypatch)
        buf = io.StringIO()
        monkeypatch.setattr(cli, "console", Console(file=buf, width=40))
        with pytest.raises(typer.Exit):
            cli._require_tls_extra()
        out = buf.getvalue()
        assert "biopb-tensor-server[tls]" in out  # markup did not eat the extra
        assert any(
            "pip install 'biopb-tensor-server[tls]'" in line
            for line in out.splitlines()
        )  # ...and it is on one line, at 40 columns
