"""Tests for the launcher's transport selection and dispatch.

These exercise the pure plumbing in ``biopb_mcp.mcp.__main__`` (arg parsing
and the stdio-vs-http dispatch) without starting a real kernel or viewer.
"""

import dataclasses
import os
import sys

import pytest

from biopb_mcp._config import McpConfig
from biopb_mcp.mcp import __main__ as launcher
from biopb_mcp.mcp.__main__ import (
    _config_defaults,
    _has_display,
    _is_agentless_viewer,
    _parse_args,
    _register_view_session,
    _setup_chat,
    _setup_observe,
    _unregister_session,
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
    return {"transport": transport}


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
        cfg = {"observe": {"enabled": False}}
        assert _setup_observe(cfg) is False
        assert fake_observe == {"configure": 0, "http": 0}

    def test_failure_is_swallowed(self, monkeypatch):
        from biopb_mcp.mcp import _observe

        def _boom():
            raise RuntimeError("nope")

        monkeypatch.setattr(_observe, "configure", lambda **k: None)
        monkeypatch.setattr(_observe, "register_http_routes", _boom)
        cfg = {"observe": {"enabled": True}}
        # An observe failure must never propagate out of the launcher.
        assert _setup_observe(cfg) is False


class TestSetupChat:
    """Chat mounts on the config the launcher actually threads.

    This exists because it did not. `load_config()` returns a **dict**, and
    `_chat_api.configure` read it with attribute access -- correct for the
    `McpConfig` dataclass every test built, and an `AttributeError` on the real
    thing. `_setup_chat` catches broadly so a chat failure can never take down
    the MCP server, so the raise became one swallowed log line and chat simply
    never appeared. Nothing was red.

    Two lessons, both encoded here: exercise the launcher's own function rather
    than the thing it calls, and feed it the launcher's own shape.
    """

    @pytest.fixture
    def mounted(self, monkeypatch):
        from biopb_mcp.mcp import _chat_api

        calls = []
        monkeypatch.setattr(_chat_api, "register_http_routes", lambda: calls.append(1))
        return calls

    @pytest.mark.parametrize(
        "cfg",
        [
            # A file with only the switch in it -- get_setting fills the rest
            # from DEFAULT_CONFIG, which is what a real mcp-config.json does.
            {"observe": {"enabled": True, "chat_enabled": True}},
            # And the fully-populated shape, as the admin page writes it.
            {
                **dataclasses.asdict(McpConfig()),
                "observe": {
                    **dataclasses.asdict(McpConfig().observe),
                    "chat_enabled": True,
                },
            },
        ],
        ids=["partial", "full"],
    )
    def test_it_mounts_for_a_viewer(self, cfg, mounted):
        assert _setup_chat(cfg, agentless=True) is True
        assert mounted == [1]

    def test_it_mounts_nothing_for_a_harness_session(self, mounted):
        cfg = {"observe": {"enabled": True, "chat_enabled": True}}
        assert _setup_chat(cfg, agentless=False) is False
        assert mounted == []

    def test_it_mounts_nothing_when_the_switch_is_off(self, mounted):
        cfg = {"observe": {"enabled": True, "chat_enabled": False}}
        assert _setup_chat(cfg, agentless=True) is False
        assert mounted == []

    @pytest.mark.parametrize(
        "cfg, agentless, expected",
        [
            ({"observe": {"enabled": True, "chat_enabled": True}}, True, True),
            ({"observe": {"enabled": True, "chat_enabled": True}}, False, False),
            ({"observe": {"enabled": True, "chat_enabled": False}}, True, False),
        ],
    )
    def test_it_publishes_the_verdict_on_api_status(
        self, mounted, cfg, agentless, expected
    ):
        # The control's dashboard labels a session's link by what it serves, and
        # reads that off /api/status. Set on every path, so a session that never
        # mounted chat -- or failed to -- reads as off rather than stale.
        from biopb_mcp.mcp import _observe

        _observe.set_chat_enabled(not expected)  # a value that must be overwritten
        assert _setup_chat(cfg, agentless=agentless) is expected
        assert _observe._chat_enabled is expected

    def test_setup_observe_hands_over_the_session_teardown(self, monkeypatch):
        # The stop route runs the launcher's own `_shutdown`, so it has to be
        # handed over at wiring time -- before the routes are registered, which
        # happens inside this call.
        from biopb_mcp.mcp import _observe

        calls = []
        _setup_observe(
            {"observe": {"enabled": True}},
            agentless=True,
            on_shutdown=lambda: calls.append(1),
        )
        assert _observe._agentless is True
        _observe._shutdown_hook()
        assert calls == [1]
        _observe.set_session_owns_its_reap(False)

    def test_setup_observe_gives_a_shim_child_no_teardown(self, monkeypatch):
        # A shim-owned child is its shim's to reap; handing it a teardown here
        # would let the web end a session an MCP client is still bridging to.
        from biopb_mcp.mcp import _observe

        _setup_observe(
            {"observe": {"enabled": True}},
            agentless=False,
            on_shutdown=lambda: None,
        )
        assert _observe._agentless is False
        assert _observe._shutdown_hook is None

    def test_a_failed_mount_reads_as_off(self, monkeypatch):
        from biopb_mcp.mcp import _chat_api, _observe

        monkeypatch.setattr(
            _chat_api, "configure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError)
        )
        _observe.set_chat_enabled(True)
        cfg = {"observe": {"enabled": True, "chat_enabled": True}}
        assert _setup_chat(cfg, agentless=True) is False
        assert _observe._chat_enabled is False


class TestAgentlessViewer:
    """Which sessions count as a viewer a human opened.

    Two things hang off this and must not drift apart: such a session publishes
    itself to the registry, and it is the only kind served the built-in chat
    loop. Pinned as a truth table rather than trusted to two inline expressions,
    which is what they were.
    """

    @pytest.mark.parametrize(
        "view,shim_owned,expected",
        [
            # `biopb mcp view`: a human opened a window; no agent is attached.
            (True, False, True),
            # A shim-owned child is serving an MCP client. It cannot reach here
            # with view=True today, and must answer False if it ever does.
            (True, True, False),
            # The stdio shim's ordinary child.
            (False, True, False),
            # A direct `--transport http` launch: wired to something by its
            # operator, and publishes no session, so it has no observe page.
            (False, False, False),
        ],
    )
    def test_only_a_shimless_viewer_counts(self, view, shim_owned, expected):
        assert _is_agentless_viewer(view, shim_owned) is expected


class TestViewSessionRegistration:
    """`biopb mcp view` has no shim, so it publishes itself into the shared
    registry the control reads (`biopb._sessions`). Without this an agentless
    viewer is invisible: no dashboard entry, no observe page, no
    `/session/<id>/*` proxying."""

    @pytest.fixture(autouse=True)
    def _isolated_registry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_SESSIONS_DIR", str(tmp_path / "sessions"))

    def test_registers_a_routable_record(self):
        from biopb import _sessions

        session_id = _register_view_session(45678)
        assert session_id is not None
        rec = _sessions.read_session(session_id)
        # Everything the control needs to route /session/<id>/* here.
        assert rec["port"] == 45678
        assert rec["host"] == "127.0.0.1"
        assert rec["pid"] == os.getpid()
        assert rec["mcp_url"] == "http://127.0.0.1:45678/mcp"

    def test_registered_session_is_listed_as_live(self):
        from biopb import _sessions

        session_id = _register_view_session(45678)
        # Our own pid owns the record, so the liveness prune must keep it --
        # this is what makes the session show up on the dashboard at all.
        assert session_id in [r["session_id"] for r in _sessions.list_sessions()]

    def test_unregister_removes_the_record(self):
        from biopb import _sessions

        session_id = _register_view_session(45678)
        _unregister_session(session_id)
        assert _sessions.read_session(session_id) is None

    def test_unregister_none_is_a_noop(self):
        # The teardown path runs on every exit, including one where the publish
        # failed or never ran (Ctrl-C during the viewer's bring-up).
        _unregister_session(None)

    def test_publish_failure_costs_only_discoverability(self, monkeypatch):
        from biopb import _sessions

        def _boom(*a, **k):
            raise OSError("read-only state dir")

        monkeypatch.setattr(_sessions, "register", _boom)
        # No exception out of the launcher, and nothing to de-register.
        assert _register_view_session(45678) is None

    def test_unregister_failure_does_not_break_teardown(self, monkeypatch):
        from biopb import _sessions

        def _boom(*a, **k):
            raise OSError("gone")

        monkeypatch.setattr(_sessions, "unregister", _boom)
        _unregister_session("20260101-000000-1")
