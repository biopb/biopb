"""Tests for the TensorConnection data-access service.

The service is exercised with a mocked ``TensorFlightClient`` — no real server,
no Qt widget, no napari viewer needed. (Note: ``import biopb_mcp`` still pulls in
the GUI stack via the package ``__init__``, which eagerly imports the widgets;
the service module itself imports no Qt/napari.)
"""

from unittest.mock import MagicMock

import pytest

from biopb_mcp import _connection
from biopb_mcp._config import DEFAULT_CONFIG
from biopb_mcp._connection import (
    SERVER_QUERY_THRESHOLD,
    TensorConnection,
    connect_error_message,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("BIOPB_TENSOR_URL", raising=False)
    monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)


def _fake_client(sources):
    client = MagicMock()
    client.list_sources.return_value = sources
    client.health_check.return_value = {"status": "SERVING"}
    return client


# ---------------------------------------------------------------------------
# resolve_from_config
# ---------------------------------------------------------------------------


class TestResolveFromConfig:
    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://env:1")
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "tok")
        cfg = {"tensor_browser": {"server_url": "grpc://cfg:2"}}
        url, token = TensorConnection.resolve_from_config(cfg)
        assert url == "grpc://env:1"
        assert token == "tok"

    def test_config_when_no_env(self):
        cfg = {"tensor_browser": {"server_url": "grpc://cfg:2"}}
        url, token = TensorConnection.resolve_from_config(cfg)
        assert url == "grpc://cfg:2"
        assert token is None

    def test_default_when_nothing(self):
        url, token = TensorConnection.resolve_from_config({})
        assert url == DEFAULT_CONFIG["tensor_browser"]["server_url"]
        assert token is None


# ---------------------------------------------------------------------------
# connect / refresh
# ---------------------------------------------------------------------------


class TestConnect:
    def test_sets_state_and_persists(self, monkeypatch):
        sources = {"a": MagicMock(), "b": MagicMock()}
        client = _fake_client(sources)
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        persisted = {}
        monkeypatch.setattr(
            TensorConnection,
            "persist_url",
            lambda self: persisted.update(url=self.url),
        )

        conn = TensorConnection(config={})
        result = conn.connect("grpc://host:9", token="t")

        assert result is sources
        assert conn.client is client
        assert conn.sources == sources
        assert conn.url == "grpc://host:9"
        assert conn.token == "t"
        assert conn.use_server_query is False
        assert conn.is_connected is True
        assert persisted == {"url": "grpc://host:9"}

    def test_on_connect_hook_fires_with_final_url_token(self, monkeypatch):
        client = _fake_client({"a": MagicMock()})
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        seen = []
        conn = TensorConnection(config={})
        conn.on_connect = lambda url, token: seen.append((url, token))
        conn.connect("grpc://host:9", token="t")

        # the hook must receive the *final* (url, token) settled by connect()
        assert seen == [("grpc://host:9", "t")]

    def test_on_connect_hook_failure_does_not_break_connect(self, monkeypatch):
        sources = {"a": MagicMock()}
        client = _fake_client(sources)
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        def boom(url, token):
            raise RuntimeError("hook boom")

        conn = TensorConnection(config={})
        conn.on_connect = boom
        # connect must still succeed despite a failing hook
        result = conn.connect("grpc://host:9", token="t")
        assert conn.is_connected is True
        assert result is sources

    def test_use_server_query_above_threshold(self, monkeypatch):
        big = {str(i): MagicMock() for i in range(SERVER_QUERY_THRESHOLD + 1)}
        client = _fake_client(big)
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        conn = TensorConnection(config={})
        conn.connect("grpc://host:9")
        assert conn.use_server_query is True

    def test_failure_resets_and_raises(self, monkeypatch):
        def boom(url, token=None):
            raise RuntimeError("nope")

        monkeypatch.setattr(_connection, "TensorFlightClient", boom)

        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="nope"):
            conn.connect("grpc://host:9")
        assert conn.client is None
        assert conn.sources == {}
        assert conn.use_server_query is False
        assert conn.is_connected is False
        # issue #86 secondary: the failure now records a non-empty reason (no
        # longer a blank error that server_status/the widget can't explain).
        assert conn.last_status == "error"
        assert conn.last_message
        assert "grpc://host:9" in conn.last_message

    def test_refresh_requires_connection(self):
        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.refresh()

    def test_refresh_relists(self, monkeypatch):
        client = _fake_client({"a": MagicMock()})
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        conn = TensorConnection(config={})
        conn.connect("grpc://host:9")
        client.list_sources.return_value = {"a": MagicMock(), "b": MagicMock()}
        result = conn.refresh()
        assert len(result) == 2
        assert conn.sources == result

    def test_resolve_source_requires_connection(self):
        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.resolve_source("cloud_x")

    def test_resolve_source_delegates_and_refreshes(self, monkeypatch):
        # resolve() returns the resolved descriptor; the connection then re-lists
        # so its snapshot carries the now-populated field set.
        resolved = MagicMock(name="resolved-descriptor")
        client = _fake_client({"cloud_x": MagicMock()})
        client.resolve.return_value = resolved
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        conn = TensorConnection(config={})
        conn.connect("grpc://host:9")
        full = {"cloud_x": MagicMock(), "other": MagicMock()}
        client.list_sources.return_value = full

        out = conn.resolve_source("cloud_x")

        # progress/cancel hooks are forwarded verbatim (None when the caller,
        # e.g. the headless agent, supplies neither).
        client.resolve.assert_called_once_with(
            "cloud_x", on_progress=None, should_cancel=None
        )
        assert out is resolved  # the resolved descriptor is returned verbatim
        assert conn.sources == full  # snapshot refreshed via list_sources()

    def test_warm_source_requires_connection(self):
        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.warm_source("cloud_x")

    def test_warm_source_delegates(self, monkeypatch):
        # warm() returns the terminal WarmProgress; the connection forwards the
        # progress/cancel hooks verbatim and does NOT refresh (warm changes
        # residency, not the descriptor).
        terminal = MagicMock(name="warm-done")
        client = _fake_client({"cloud_x": MagicMock()})
        client.warm.return_value = terminal
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        conn = TensorConnection(config={})
        conn.connect("grpc://host:9")

        out = conn.warm_source("cloud_x")

        client.warm.assert_called_once_with(
            "cloud_x", on_progress=None, should_cancel=None
        )
        assert out is terminal


# ---------------------------------------------------------------------------
# readiness gating (issue #12): STARTING vs SERVING vs down
# ---------------------------------------------------------------------------


class TestConnectReadiness:
    def test_starting_raises_server_starting(self, monkeypatch):
        client = _fake_client({"a": MagicMock()})
        client.health_check.return_value = {
            "status": "STARTING",
            "source_count": 3,
        }
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        conn = TensorConnection(config={})
        with pytest.raises(_connection.ServerStarting):
            conn.connect("grpc://host:9")

        # The half-built catalog is never trusted while starting.
        assert conn.is_connected is False
        assert conn.last_status == "starting"
        assert "scanning" in conn.last_message
        client.list_sources.assert_not_called()

    def test_health_probe_error_falls_through(self, monkeypatch):
        # Older server with no health action: the advisory probe raises, but
        # list_sources() still works -> connect succeeds (backward compatible).
        sources = {"a": MagicMock()}
        client = _fake_client(sources)
        client.health_check.side_effect = RuntimeError("no health action")
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

        conn = TensorConnection(config={})
        result = conn.connect("grpc://host:9")

        assert result is sources
        assert conn.is_connected is True
        assert conn.last_status == "connected"

    def test_down_fails_fast(self, monkeypatch):
        client = _fake_client({})
        client.list_sources.side_effect = RuntimeError("unavailable")
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )

        conn = TensorConnection(config={})
        # A stale STARTING message from a prior attempt must not linger.
        conn.last_status = "starting"
        conn.last_message = "scanning…"
        with pytest.raises(RuntimeError, match="unavailable"):
            conn.connect("grpc://host:9")
        assert conn.is_connected is False
        assert conn.last_status == "error"
        # The stale "starting" message is replaced by the error reason (#86):
        # an unreachable server gets a friendly, actionable hint.
        assert "scanning" not in conn.last_message
        assert "Cannot reach" in conn.last_message
        assert "grpc://host:9" in conn.last_message

    def test_starting_message_includes_zero_counts(self):
        # source_count=0 is meaningful (just started) and must not be dropped.
        msg = _connection._starting_message(
            {"status": "STARTING", "source_count": 0, "uptime_seconds": 0}
        )
        assert "0 sources registered so far" in msg
        assert "up 0s" in msg


class TestConnectWhenBooted:
    def test_waits_through_refused_and_starting(self, monkeypatch):
        conn = TensorConnection(config={})
        sources = {"a": MagicMock()}
        outcomes = [
            RuntimeError("connection refused"),  # pre-bind window
            _connection.ServerStarting("STARTING"),  # scanning
            sources,  # ready
        ]

        def fake_connect(url, token=None):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(conn, "connect", fake_connect)
        monkeypatch.setattr(_connection.time, "sleep", lambda s: None)

        result = conn.connect_when_booted("grpc://host:9", timeout=30.0)
        assert result is sources
        assert outcomes == []  # all three attempts consumed

    def test_timeout_raises(self, monkeypatch):
        conn = TensorConnection(config={})

        def always_starting(url, token=None):
            raise _connection.ServerStarting("STARTING")

        monkeypatch.setattr(conn, "connect", always_starting)
        monkeypatch.setattr(_connection.time, "sleep", lambda s: None)
        # Fake clock that jumps past the deadline on the first check.
        clock = {"t": 0.0}

        def fake_monotonic():
            clock["t"] += 10.0
            return clock["t"]

        monkeypatch.setattr(_connection.time, "monotonic", fake_monotonic)

        with pytest.raises(RuntimeError, match="did not become ready"):
            conn.connect_when_booted("grpc://host:9", timeout=5.0)


# ---------------------------------------------------------------------------
# background source watcher (issue #44)
# ---------------------------------------------------------------------------


class _FakeStop:
    """Deterministic stand-in for the watcher's stop Event.

    ``wait`` returns ``False`` (not stopped) for the first *allow* calls so the
    loop body runs exactly that many times, then the next ``is_set`` returns
    ``True`` and the loop exits — no real sleeping, no thread, no timing races.
    """

    def __init__(self, allow):
        self.allow = allow
        self.waits = 0
        self._set = False

    def is_set(self):
        return self._set or self.waits >= self.allow

    def wait(self, _timeout):
        if self.is_set():
            return True
        self.waits += 1
        return False

    def set(self):
        self._set = True

    def clear(self):
        self._set = False


def _connected_conn(monkeypatch, sources, health_results):
    """A connected TensorConnection whose client serves *health_results*.

    ``connect()`` consumes one ``health_check`` (its readiness probe) and one
    ``list_sources``; we arm the per-poll health sequence and clear the call
    history *after* connecting, so the watch-loop assertions see only loop
    activity.
    """
    client = _fake_client(sources)
    monkeypatch.setattr(
        _connection, "TensorFlightClient", lambda url, token=None: client
    )
    monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)
    conn = TensorConnection(config={})
    conn.connect("grpc://host:9")
    client.health_check.reset_mock()
    client.list_sources.reset_mock()
    client.health_check.side_effect = list(health_results)
    return conn, client


class TestSourceWatch:
    def test_relists_when_count_changes(self, monkeypatch):
        # Cached 2 sources; server grows to 3 -> the watcher re-lists once.
        conn, client = _connected_conn(
            monkeypatch,
            {"a": MagicMock(), "b": MagicMock()},
            [{"source_count": 2}, {"source_count": 3}],
        )
        grown = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}
        client.list_sources.return_value = grown
        changed = []
        conn.on_sources_changed = changed.append

        conn._watch_stop = _FakeStop(allow=2)
        conn._source_watch_loop(0.0, 0.0)

        assert conn.sources is grown
        assert changed == [grown]

    def test_relists_on_connect_mid_index_partial(self, monkeypatch):
        # Connected mid-index: cached 1 source, server already reports 18 ->
        # re-list on the very first poll (issue #44 reconciliation).
        conn, client = _connected_conn(
            monkeypatch, {"a": MagicMock()}, [{"source_count": 18}]
        )
        full = {str(i): MagicMock() for i in range(18)}
        client.list_sources.return_value = full

        conn._watch_stop = _FakeStop(allow=1)
        conn._source_watch_loop(0.0, 0.0)

        assert len(conn.sources) == 18

    def test_stable_count_does_not_relist(self, monkeypatch):
        conn, client = _connected_conn(
            monkeypatch,
            {"a": MagicMock(), "b": MagicMock()},
            [{"source_count": 2}, {"source_count": 2}],
        )
        conn._watch_stop = _FakeStop(allow=2)
        conn._source_watch_loop(0.0, 0.0)

        client.list_sources.assert_not_called()

    def test_health_error_is_tolerated(self, monkeypatch):
        conn, client = _connected_conn(
            monkeypatch, {"a": MagicMock()}, [RuntimeError("blip")]
        )
        conn._watch_stop = _FakeStop(allow=1)
        conn._source_watch_loop(0.0, 0.0)  # must not raise

        client.list_sources.assert_not_called()

    def test_disconnected_does_not_relist(self, monkeypatch):
        conn, client = _connected_conn(monkeypatch, {"a": MagicMock()}, [])
        conn.client = None
        conn._watch_stop = _FakeStop(allow=1)
        conn._source_watch_loop(0.0, 0.0)

        client.list_sources.assert_not_called()
        client.health_check.assert_not_called()

    def test_missing_source_count_does_not_relist(self, monkeypatch):
        # Older server without source_count in health: nothing to watch.
        conn, client = _connected_conn(
            monkeypatch, {"a": MagicMock()}, [{"status": "SERVING"}]
        )
        conn._watch_stop = _FakeStop(allow=1)
        conn._source_watch_loop(0.0, 0.0)

        client.list_sources.assert_not_called()

    def test_relist_failure_keeps_watcher_alive(self, monkeypatch):
        conn, client = _connected_conn(
            monkeypatch,
            {"a": MagicMock()},
            [{"source_count": 2}, {"source_count": 3}],
        )
        client.list_sources.side_effect = RuntimeError("list boom")

        conn._watch_stop = _FakeStop(allow=2)
        conn._source_watch_loop(0.0, 0.0)  # must not raise

        # Both polls ran despite the re-list error on the second.
        assert client.health_check.call_count == 2

    def test_start_is_idempotent(self, monkeypatch):
        conn = TensorConnection(config={})
        started = []
        monkeypatch.setattr(
            _connection.threading,
            "Thread",
            lambda *a, **k: started.append(k.get("name")) or MagicMock(),
        )
        conn.start_source_watch(min_interval=1.0, max_interval=2.0)
        # The MagicMock thread reports alive -> a second start is a no-op.
        conn.start_source_watch(min_interval=1.0, max_interval=2.0)
        assert started == ["biopb-source-watch"]

    def test_disabled_when_min_interval_zero(self, monkeypatch):
        conn = TensorConnection(config={})
        monkeypatch.setattr(
            _connection.threading,
            "Thread",
            lambda *a, **k: pytest.fail("watcher must not start"),
        )
        conn.start_source_watch(min_interval=0, max_interval=10.0)
        assert conn._watch_thread is None


# ---------------------------------------------------------------------------
# persist_url
# ---------------------------------------------------------------------------


class TestPersistUrl:
    def test_preserves_unowned_keys(self):
        # persist_url() writes via CONFIG.set, which preserves keys the service
        # does not own (here mcp.services.process_image_servers) -- both in the
        # cache and on disk.
        import json

        from biopb_mcp._config import CONFIG, get_config_path

        CONFIG.set(
            "mcp.services.process_image_servers",
            ["grpc://ops:5"],
            persist=False,
        )
        CONFIG.set("tensor_browser.server_url", "grpc://old:1", persist=False)

        conn = TensorConnection(config={})
        conn.url = "grpc://new:2"
        conn.persist_url()

        assert CONFIG.get("tensor_browser.server_url") == "grpc://new:2"
        assert CONFIG.get("mcp.services.process_image_servers") == ["grpc://ops:5"]
        with get_config_path().open() as f:
            saved = json.load(f)
        assert saved["tensor_browser"]["server_url"] == "grpc://new:2"
        assert saved["mcp"]["services"]["process_image_servers"] == ["grpc://ops:5"]


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_delegates(monkeypatch):
    client = _fake_client({})
    client.health_check.return_value = "SERVING"
    monkeypatch.setattr(
        _connection, "TensorFlightClient", lambda url, token=None: client
    )
    monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)

    conn = TensorConnection(config={})
    assert conn.health() is None  # not connected yet
    conn.connect("grpc://host:9")
    assert conn.health() == "SERVING"


# ---------------------------------------------------------------------------
# local-server autostart fallback
# ---------------------------------------------------------------------------


class TestIsLocalUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "grpc://localhost:8815",
            "grpc://127.0.0.1:8815",
            "grpc://[::1]:8815",
            "grpc://:8815",
        ],
    )
    def test_local(self, url):
        assert _connection.is_local_url(url) is True

    @pytest.mark.parametrize(
        "url",
        ["grpc://example.com:8815", "grpc://10.0.0.5:8815"],
    )
    def test_remote(self, url):
        assert _connection.is_local_url(url) is False


class TestCanAutostart:
    def test_true_when_local_and_cli_present(self, monkeypatch):
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: True)
        conn = TensorConnection(config={})  # default URL is localhost
        assert conn.can_autostart_server() is True

    def test_false_without_cli(self, monkeypatch):
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: False)
        conn = TensorConnection(config={})
        assert conn.can_autostart_server() is False

    def test_false_for_remote(self, monkeypatch):
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: True)
        cfg = {"tensor_browser": {"server_url": "grpc://example.com:8815"}}
        conn = TensorConnection(config=cfg)
        assert conn.can_autostart_server() is False


class TestLaunchLocalServer:
    def test_launches_without_connecting(self, monkeypatch, tmp_path):
        # launch_local_server only spawns the daemon; it must not connect.
        client = _fake_client({"a": MagicMock()})
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: True)
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(
            _connection, "DEFAULT_SERVER_CONFIG", tmp_path / "missing.toml"
        )
        calls = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            return MagicMock(returncode=0)

        monkeypatch.setattr(_connection.subprocess, "run", fake_run)

        conn = TensorConnection(config={})
        assert conn.launch_local_server() is None
        assert calls["cmd"][:3] == ["biopb", "server", "start"]
        # No connection attempted by the launch step.
        client.list_sources.assert_not_called()
        assert conn.is_connected is False

    def test_raises_without_cli(self, monkeypatch):
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: False)
        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="biopb CLI not found"):
            conn.launch_local_server()

    def test_server_start_timeout_from_config(self):
        from biopb_mcp._config import CONFIG

        CONFIG.set("mcp.server_start_timeout", 12.5, persist=False)
        conn = TensorConnection(config={})
        assert conn.server_start_timeout() == 12.5


class TestStartLocalServer:
    def test_starts_and_connects_without_token(self, monkeypatch, tmp_path):
        sources = {"a": MagicMock()}
        client = _fake_client(sources)
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: True)
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)
        # No real config file -> --config omitted.
        monkeypatch.setattr(
            _connection, "DEFAULT_SERVER_CONFIG", tmp_path / "missing.toml"
        )

        calls = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            return MagicMock(returncode=0)

        monkeypatch.setattr(_connection.subprocess, "run", fake_run)

        conn = TensorConnection(config={})
        assert conn.token is None
        result = conn.start_local_server()

        assert result is sources
        assert conn.is_connected is True
        # Token logic is left to the CLI; we never fabricate one.
        assert conn.token is None
        assert calls["cmd"][:3] == ["biopb", "server", "start"]
        assert "--token" not in calls["cmd"]
        assert "--config" not in calls["cmd"]  # missing file

    def test_passes_existing_config(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / "biopb.toml"
        cfg_file.write_text("")
        client = _fake_client({"a": MagicMock()})
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: True)
        monkeypatch.setattr(
            _connection, "TensorFlightClient", lambda url, token=None: client
        )
        monkeypatch.setattr(TensorConnection, "persist_url", lambda self: None)
        monkeypatch.setattr(_connection, "DEFAULT_SERVER_CONFIG", cfg_file)

        calls = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(_connection.subprocess, "run", fake_run)

        conn = TensorConnection(config={})
        conn.start_local_server()
        assert "--config" in calls["cmd"]
        assert str(cfg_file) in calls["cmd"]

    def test_raises_without_cli(self, monkeypatch):
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: False)
        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="biopb CLI not found"):
            conn.start_local_server()

    def test_raises_when_cli_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_connection, "biopb_cli_available", lambda: True)
        monkeypatch.setattr(
            _connection, "DEFAULT_SERVER_CONFIG", tmp_path / "missing.toml"
        )
        monkeypatch.setattr(
            _connection.subprocess,
            "run",
            lambda cmd, **kw: MagicMock(returncode=1),
        )
        conn = TensorConnection(config={})
        with pytest.raises(RuntimeError, match="failed"):
            conn.start_local_server()


class TestAutoConnect:
    """The shared connect policy used by both the kernel and the widget.

    Both callers drive this off their own worker thread (the MCP bootstrap on a
    daemon thread, the widget on a connect worker); here we call it directly with
    ``connect`` and friends mocked.
    """

    def test_connects_on_first_try(self, monkeypatch):
        conn = TensorConnection(config={})
        conn.url, conn.token = "grpc://host:9", "tok"
        connect = MagicMock(return_value={"a": MagicMock()})
        monkeypatch.setattr(conn, "connect", connect)
        booted = MagicMock()
        monkeypatch.setattr(conn, "connect_when_booted", booted)
        started = MagicMock()
        monkeypatch.setattr(conn, "start_local_server", started)

        conn.auto_connect()

        connect.assert_called_once_with("grpc://host:9", "tok")
        # A clean connect short-circuits — no boot wait, no autostart.
        booted.assert_not_called()
        started.assert_not_called()

    def test_starting_waits_through_boot(self, monkeypatch):
        conn = TensorConnection(config={})
        conn.url, conn.token = "grpc://host:9", None
        monkeypatch.setattr(
            conn,
            "connect",
            MagicMock(side_effect=_connection.ServerStarting("STARTING")),
        )
        booted = MagicMock(return_value={"a": MagicMock()})
        monkeypatch.setattr(conn, "connect_when_booted", booted)
        monkeypatch.setattr(conn, "server_start_timeout", lambda: 42.0)
        started = MagicMock()
        monkeypatch.setattr(conn, "start_local_server", started)

        conn.auto_connect()

        # A STARTING server is waited through, not autostarted.
        booted.assert_called_once_with("grpc://host:9", None, timeout=42.0)
        started.assert_not_called()

    def test_starting_then_boot_timeout_is_swallowed(self, monkeypatch):
        conn = TensorConnection(config={})
        monkeypatch.setattr(
            conn,
            "connect",
            MagicMock(side_effect=_connection.ServerStarting("STARTING")),
        )
        monkeypatch.setattr(
            conn,
            "connect_when_booted",
            MagicMock(side_effect=RuntimeError("did not become ready")),
        )
        monkeypatch.setattr(conn, "server_start_timeout", lambda: 1.0)
        started = MagicMock()
        monkeypatch.setattr(conn, "start_local_server", started)

        # Best-effort: a boot timeout must not raise out of auto_connect.
        conn.auto_connect()
        # Already up (just slow), so we do NOT fall through to autostart.
        started.assert_not_called()

    def test_unreachable_autostarts_local(self, monkeypatch):
        conn = TensorConnection(config={})
        conn.url = "grpc://localhost:8815"
        monkeypatch.setattr(
            conn,
            "connect",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        monkeypatch.setattr(conn, "can_autostart_server", lambda: True)
        started = MagicMock()
        monkeypatch.setattr(conn, "start_local_server", started)

        conn.auto_connect()
        started.assert_called_once_with()

    def test_unreachable_no_autostart_is_swallowed(self, monkeypatch):
        conn = TensorConnection(config={})
        conn.url = "grpc://remote:9"
        monkeypatch.setattr(
            conn,
            "connect",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        # Remote URL / no CLI -> cannot autostart; auto_connect just gives up.
        monkeypatch.setattr(conn, "can_autostart_server", lambda: False)
        started = MagicMock()
        monkeypatch.setattr(conn, "start_local_server", started)

        conn.auto_connect()  # must not raise
        started.assert_not_called()

    def test_autostart_failure_is_swallowed(self, monkeypatch):
        conn = TensorConnection(config={})
        conn.url = "grpc://localhost:8815"
        monkeypatch.setattr(
            conn,
            "connect",
            MagicMock(side_effect=RuntimeError("connection refused")),
        )
        monkeypatch.setattr(conn, "can_autostart_server", lambda: True)
        monkeypatch.setattr(
            conn,
            "start_local_server",
            MagicMock(side_effect=RuntimeError("boot failed")),
        )
        # A failed autostart must not propagate out of the best-effort policy.
        conn.auto_connect()


# ---------------------------------------------------------------------------
# connect_error_message (issue #86 secondary)
# ---------------------------------------------------------------------------


class TestConnectErrorMessage:
    URL = "grpc://host:9"

    def test_auth_required_when_no_token(self):
        exc = RuntimeError("FlightUnauthenticatedError: token required")
        msg = connect_error_message(exc, self.URL, token=None)
        assert "Authentication required" in msg
        assert self.URL in msg
        # Names the fix so the user knows what to do.
        assert "token" in msg.lower()

    def test_auth_failed_when_token_present(self):
        exc = RuntimeError("PermissionDenied: invalid token")
        msg = connect_error_message(exc, self.URL, token="bad")
        assert "Authentication failed" in msg
        assert "rejected" in msg
        assert self.URL in msg

    def test_unreachable_gets_friendly_hint(self):
        exc = RuntimeError("FlightUnavailableError: failed to connect to all addresses")
        msg = connect_error_message(exc, self.URL, token=None)
        assert "Cannot reach" in msg
        assert self.URL in msg

    def test_other_error_echoes_underlying(self):
        exc = ValueError("something odd")
        msg = connect_error_message(exc, self.URL, token=None)
        assert "something odd" in msg

    def test_never_blank(self):
        # Even an exception with an empty str() yields an actionable message.
        msg = connect_error_message(RuntimeError(), self.URL, token=None)
        assert msg.strip()
