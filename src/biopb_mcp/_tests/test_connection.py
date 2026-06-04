"""Tests for the TensorConnection data-access service.

The service is exercised with a mocked ``TensorFlightClient`` — no real server,
no Qt widget, no napari viewer needed. (Note: ``import biopb_mcp`` still pulls in
the GUI stack via the package ``__init__``, which eagerly imports the widgets;
the service module itself imports no Qt/napari.)
"""

from unittest.mock import MagicMock

import pytest

from biopb_mcp import _connection
from biopb_mcp._connection import (
    _DEFAULT_URL,
    SERVER_QUERY_THRESHOLD,
    TensorConnection,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("BIOPB_TENSOR_URL", raising=False)
    monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)


def _fake_client(sources):
    client = MagicMock()
    client.list_sources.return_value = sources
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
        assert url == _DEFAULT_URL
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


# ---------------------------------------------------------------------------
# persist_url
# ---------------------------------------------------------------------------


class TestPersistUrl:
    def test_preserves_unowned_keys(self, monkeypatch):
        # A fresh config with a key the service does not own.
        existing = {
            "tensor_browser": {"server_url": "grpc://old:1"},
            "mcp": {"process_image_servers": ["grpc://ops:5"]},
        }
        saved = {}
        monkeypatch.setattr(_connection, "load_config", lambda: dict(existing))
        monkeypatch.setattr(
            _connection, "save_config", lambda cfg: saved.update(cfg)
        )

        conn = TensorConnection(config={})
        conn.url = "grpc://new:2"
        conn.persist_url()

        assert saved["tensor_browser"]["server_url"] == "grpc://new:2"
        assert saved["mcp"]["process_image_servers"] == ["grpc://ops:5"]


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
