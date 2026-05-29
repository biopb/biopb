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
