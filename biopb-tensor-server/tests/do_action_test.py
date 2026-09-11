"""Tests for TensorFlightServer custom Flight do_action handlers."""

import json

import pyarrow.flight as flight
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.serving.server import TensorFlightServer


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
        raise AssertionError("expected FlightServerError")
    except flight.FlightServerError as exc:
        assert "Cache not initialized" in str(exc)


def test_cache_stats_listed_in_actions():
    """cache_stats is advertised by list_actions."""
    server = TensorFlightServer("grpc://localhost:0")
    action_types = {a.type for a in server.list_actions(None)}
    assert "cache_stats" in action_types


def test_decode_rates_action_returns_the_table():
    """do_action('decode_rates') returns the measured per-array_id throughput."""
    from biopb_tensor_server.core.retention import (
        DecodeRates,
        active_decode_rates,
        set_active_decode_rates,
    )

    previous = active_decode_rates()
    rates = DecodeRates()
    for _ in range(8):
        rates.record("src/0", 8 << 20, 0.01)
    set_active_decode_rates(rates)
    try:
        server = TensorFlightServer("grpc://localhost:0")
        (raw,) = list(server.do_action(None, flight.Action("decode_rates", b"")))
        table = json.loads(bytes(raw))
        assert table["src/0"]["samples"] == 8
        assert table["src/0"]["mbps"] > 0
    finally:
        set_active_decode_rates(previous)


def test_decode_rates_action_answers_before_anything_is_measured():
    """An empty table is the honest answer for a server that has only served
    downsampled reads -- not an error, unlike an uninitialized cache."""
    from biopb_tensor_server.core.retention import (
        DecodeRates,
        active_decode_rates,
        set_active_decode_rates,
    )

    previous = active_decode_rates()
    set_active_decode_rates(DecodeRates())
    try:
        server = TensorFlightServer("grpc://localhost:0")
        (raw,) = list(server.do_action(None, flight.Action("decode_rates", b"")))
        assert json.loads(bytes(raw)) == {}
    finally:
        set_active_decode_rates(previous)


def test_decode_rates_listed_in_actions():
    server = TensorFlightServer("grpc://localhost:0")
    assert "decode_rates" in {a.type for a in server.list_actions(None)}
