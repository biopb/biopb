"""Tests for TensorFlightServer custom Flight do_action handlers."""

import json

import pyarrow.flight as flight
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.config import CacheConfig
from biopb_tensor_server.server import TensorFlightServer


def test_cache_stats_action_returns_stats():
    """do_action('cache_stats') returns the backend's CacheStats as JSON."""
    CacheManager.initialize(CacheConfig(backend="memory"))
    try:
        server = TensorFlightServer("grpc://localhost:0")
        (raw,) = list(server.do_action(None, flight.Action("cache_stats", b"")))
        stats = json.loads(bytes(raw))
        for field in (
            "hits",
            "misses",
            "evictions",
            "pending_waits",
            "total_entries",
            "total_bytes",
            "pool_stats",
        ):
            assert field in stats
    finally:
        CacheManager.reset()


def test_cache_stats_action_errors_without_cache():
    """Without an initialized cache the action raises rather than crashing."""
    CacheManager.reset()
    server = TensorFlightServer("grpc://localhost:0")
    try:
        list(server.do_action(None, flight.Action("cache_stats", b"")))
        assert False, "expected FlightServerError"
    except flight.FlightServerError as exc:
        assert "Cache not initialized" in str(exc)


def test_cache_stats_listed_in_actions():
    """cache_stats is advertised by list_actions."""
    server = TensorFlightServer("grpc://localhost:0")
    action_types = {a.type for a in server.list_actions(None)}
    assert "cache_stats" in action_types
