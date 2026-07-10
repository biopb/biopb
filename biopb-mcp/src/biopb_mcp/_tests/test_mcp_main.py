"""Tests for the launcher's transport selection and dispatch.

These exercise the pure plumbing in ``biopb_mcp.mcp.__main__`` (arg parsing
and the stdio-vs-http dispatch) without starting a real kernel, viewer, or
daemon.
"""

import os
import sys
import threading

import pytest

from biopb_mcp.mcp import __main__ as launcher
from biopb_mcp.mcp.__main__ import (
    _config_defaults,
    _has_display,
    _install_shutdown_sentinel_watcher,
    _parse_args,
    _remove_pidfile,
    _resolve_headless,
    _setup_observe,
    _shutdown_sentinel_path,
    _write_pidfile,
    main,
)


class TestParseArgs:
    def test_defaults_come_from_config(self):
        opts = _parse_args([], default_transport="http", default_port=8765)
        assert opts.transport == "http"
        assert opts.port == 8765

    def test_config_default_can_be_stdio(self):
        opts = _parse_args([], default_transport="stdio", default_port=8765)
        assert opts.transport == "stdio"

    def test_transport_flag_overrides(self):
        opts = _parse_args(
            ["--transport", "stdio"], default_transport="http", default_port=1
        )
        assert opts.transport == "stdio"

    def test_port_flag_overrides(self):
        opts = _parse_args(
            ["--port", "9000"], default_transport="http", default_port=8765
        )
        assert opts.port == 9000

    def test_unknown_transport_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(
                ["--transport", "ftp"],
                default_transport="http",
                default_port=8765,
            )

    def test_view_defaults_false(self):
        opts = _parse_args([], default_transport="http", default_port=8765)
        assert opts.view is False

    def test_view_flag_sets_true(self):
        opts = _parse_args(["--view"], default_transport="http", default_port=8765)
        assert opts.view is True


def _cfg(**transport):
    """Build a full config carrying only the given mcp.transport overrides."""
    return {"mcp": {"transport": transport}}


class TestConfigDefaults:
    def test_clean_config_passes_through(self):
        assert _config_defaults(_cfg(kind="http", port=9000)) == (
            "http",
            9000,
        )

    def test_unknown_transport_falls_back_to_stdio(self):
        transport, _ = _config_defaults(_cfg(kind="ftp"))
        assert transport == "stdio"

    def test_stringified_port_is_coerced_to_int(self):
        _, port = _config_defaults(_cfg(port="8765"))
        assert port == 8765

    def test_garbage_port_falls_back(self):
        _, port = _config_defaults(_cfg(port="not-a-number"))
        assert port == 8765

    def test_empty_config_uses_documented_defaults(self):
        assert _config_defaults({}) == ("stdio", 8765)


class TestMainDispatch:
    """main() routes stdio to the shim without touching the heavy stack."""

    @pytest.fixture(autouse=True)
    def empty_config(self, monkeypatch):
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "load_config", dict)

    def test_stdio_runs_the_shim(self, monkeypatch):
        from biopb_mcp.mcp import _shim

        calls = []
        monkeypatch.setattr(
            _shim, "serve", lambda config, port: calls.append((config, port))
        )
        assert main(["--transport", "stdio", "--port", "9123"]) == 0
        assert calls == [({}, 9123)]

    def test_stdio_bridge_failure_exits_nonzero(self, monkeypatch):
        from biopb_mcp.mcp import _shim

        def _boom(config, port):
            raise TimeoutError("daemon never came up")

        monkeypatch.setattr(_shim, "serve", _boom)
        # The shim failing must surface as a nonzero exit (client sees EOF),
        # never a traceback-crash or a hung launcher.
        assert main(["--transport", "stdio"]) == 1

    def test_view_routes_to_serve_http_view_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            launcher,
            "_serve_http",
            lambda config, port, view=False: calls.append((port, view)) or 0,
        )
        assert main(["--view", "--port", "0"]) == 0
        assert calls == [(0, True)]

    def test_view_takes_precedence_over_stdio_default(self, monkeypatch):
        # empty config -> default transport stdio, but --view wins (viewer path).
        calls = []
        monkeypatch.setattr(
            launcher,
            "_serve_http",
            lambda config, port, view=False: calls.append(view) or 0,
        )
        assert main(["--view"]) == 0
        assert calls == [True]


class TestHasDisplay:
    def test_linux_gates_on_display_env(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _has_display() is False
        monkeypatch.setenv("DISPLAY", ":0")
        assert _has_display() is True

    def test_linux_wayland_counts(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert _has_display() is True

    def test_macos_always_has_display(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _has_display() is True


class TestResolveHeadless:
    def test_explicit_headless_always_true(self):
        assert _resolve_headless("headless", True) is True
        assert _resolve_headless("headless", False) is True

    def test_explicit_visible_always_false(self):
        # The launcher fails fast separately when visible + no display.
        assert _resolve_headless("visible", False) is False
        assert _resolve_headless("visible", True) is False

    def test_auto_follows_display(self):
        assert _resolve_headless("auto", True) is False
        assert _resolve_headless("auto", False) is True


class TestSetupObserve:
    @pytest.fixture
    def fake_observe(self, monkeypatch):
        from biopb_mcp.mcp import _observe

        calls = {"configure": 0, "http": 0}
        monkeypatch.setattr(
            _observe,
            "configure",
            lambda **k: calls.__setitem__("configure", calls["configure"] + 1),
        )
        monkeypatch.setattr(
            _observe,
            "register_http_routes",
            lambda: calls.__setitem__("http", calls["http"] + 1),
        )
        return calls

    def test_enabled_by_default(self, fake_observe):
        # Opt-out: empty config -> on.
        assert _setup_observe({}) is True
        assert fake_observe["http"] == 1

    def test_explicitly_disabled(self, fake_observe):
        cfg = {"mcp": {"observe": {"enabled": False}}}
        assert _setup_observe(cfg) is False
        assert fake_observe == {"configure": 0, "http": 0}

    def test_failure_is_swallowed(self, monkeypatch):
        from biopb_mcp.mcp import _observe

        def _boom():
            raise RuntimeError("nope")

        monkeypatch.setattr(_observe, "configure", lambda **k: None)
        monkeypatch.setattr(_observe, "register_http_routes", _boom)
        cfg = {"mcp": {"observe": {"enabled": True}}}
        # An observe failure must never propagate out of the launcher.
        assert _setup_observe(cfg) is False


class TestPidfile:
    """The daemon registers its own PID so `biopb mcp status` finds it
    regardless of launch path (CLI, stdio shim, or manual)."""

    @pytest.fixture
    def pidfile(self, tmp_path, monkeypatch):
        path = tmp_path / "mcp-server.pid"
        # _write_pidfile resolves the path lazily via _config.get_pid_file.
        monkeypatch.setattr("biopb_mcp._config.get_pid_file", lambda: path)
        return path

    def test_writes_pid_and_create_time_token(self, pidfile, monkeypatch):
        monkeypatch.setattr(launcher, "_port_listening", lambda *_a, **_k: False)
        monkeypatch.setattr(launcher, "_self_create_time", lambda: 4242)
        returned = _write_pidfile(8765)
        assert returned == pidfile
        # pid + create-time token, whitespace-separated (read back by the CLI's
        # _read_pid_record); the token lets stop/status reject a reused PID.
        assert pidfile.read_text() == f"{os.getpid()}\n4242"

    def test_writes_bare_pid_when_create_time_unknown(self, pidfile, monkeypatch):
        # No create-time available (e.g. macOS) -> legacy bare-PID form, which
        # the CLI still reads (token None -> liveness-only check).
        monkeypatch.setattr(launcher, "_port_listening", lambda *_a, **_k: False)
        monkeypatch.setattr(launcher, "_self_create_time", lambda: None)
        _write_pidfile(8765)
        assert pidfile.read_text() == str(os.getpid())

    def test_skips_write_when_port_taken(self, pidfile, monkeypatch):
        # A racing loser sees the winner already on the port: don't clobber it.
        monkeypatch.setattr(launcher, "_port_listening", lambda *_a, **_k: True)
        assert _write_pidfile(8765) is None
        assert not pidfile.exists()

    def test_write_failure_is_swallowed(self, pidfile, monkeypatch):
        monkeypatch.setattr(launcher, "_port_listening", lambda *_a, **_k: False)

        def _boom(*_a, **_k):
            raise OSError("read-only fs")

        monkeypatch.setattr(type(pidfile), "write_text", _boom)
        # A write failure costs `status` visibility, never the server.
        assert _write_pidfile(8765) is None

    def test_remove_is_pid_safe(self, pidfile):
        pidfile.write_text(str(os.getpid()))
        _remove_pidfile(pidfile)
        assert not pidfile.exists()

    def test_remove_matches_pid_with_token_present(self, pidfile):
        # The token-bearing form still self-deletes: match is on the PID field.
        pidfile.write_text(f"{os.getpid()}\n4242")
        _remove_pidfile(pidfile)
        assert not pidfile.exists()

    def test_remove_leaves_other_pids(self, pidfile):
        # A losing daemon's exit must not delete the winner's PID file.
        pidfile.write_text("999999999")
        _remove_pidfile(pidfile)
        assert pidfile.exists()

    def test_remove_none_is_noop(self):
        _remove_pidfile(None)  # write was skipped/failed; nothing to undo


class TestShutdownSentinelWatcher:
    """The Windows stop path (issue #323): `biopb mcp stop` cannot deliver a
    catchable signal there (os.kill is TerminateProcess), so it drops a
    sentinel file and this daemon-side watcher runs the shared shutdown. The
    watcher itself is platform-agnostic — only its installation in _serve_http
    is Windows-gated — so these tests run on every OS."""

    def test_sentinel_triggers_shutdown_and_is_consumed(self, tmp_path):
        sentinel = tmp_path / "mcp-server.stop"
        fired = threading.Event()
        reasons = []

        def _shutdown(reason):
            reasons.append(reason)
            fired.set()

        _install_shutdown_sentinel_watcher(sentinel, _shutdown, poll=0.01)
        sentinel.write_text("stop")
        assert fired.wait(5), "watcher never fired on a fresh sentinel"
        assert reasons == ["stop sentinel"]
        # Consumed before shutdown, so a daemon started later can't trip on it
        # (belt and braces on top of the delete-at-install clear).
        assert not sentinel.exists()

    def test_stale_sentinel_is_ignored(self, tmp_path):
        # A leftover from a previous run must not stop a freshly started daemon.
        # The watcher clears any pre-existing sentinel once at install (#345), so
        # it is neither acted on nor left behind to trip a later check -- no mtime
        # comparison, regardless of the leftover's timestamp.
        sentinel = tmp_path / "mcp-server.stop"
        sentinel.write_text("stop")
        fired = threading.Event()
        _install_shutdown_sentinel_watcher(sentinel, lambda _r: fired.set(), poll=0.01)
        assert not fired.wait(0.3)
        assert not sentinel.exists()  # cleared at install, not acted on

    def test_missing_sentinel_never_fires(self, tmp_path):
        fired = threading.Event()
        _install_shutdown_sentinel_watcher(
            tmp_path / "mcp-server.stop", lambda _r: fired.set(), poll=0.01
        )
        assert not fired.wait(0.2)

    def test_sentinel_path_is_fixed_pidfile_sibling(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "biopb_mcp._config.get_pid_file", lambda: tmp_path / "mcp-server.pid"
        )
        # Fixed name (not pid-keyed: uv/Store-Python shims make PIDs ambiguous
        # on Windows) in the PID file's dir; biopb.cli hardcodes the same.
        assert _shutdown_sentinel_path() == tmp_path / "mcp-server.stop"
