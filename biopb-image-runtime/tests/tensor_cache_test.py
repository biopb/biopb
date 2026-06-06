"""Focused tests for image-server tensor cache helpers."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_image_base.server import EmbeddedTensorCache


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


@pytest.fixture
def embedded_cache(tmp_path: Path) -> EmbeddedTensorCache:
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig
    from biopb_tensor_server.server import TensorFlightServer

    CacheManager.reset()
    CacheManager.initialize(
        CacheConfig(
            backend="file",
            file_cache_dir=tmp_path,
            file_max_total_bytes=128 * 1024 * 1024,
        )
    )
    tensor_server = TensorFlightServer(location="grpc://0.0.0.0:0", writable=True)

    try:
        yield EmbeddedTensorCache(
            tensor_server=tensor_server,
            external_location="grpc://127.0.0.1:9999",
        )
    finally:
        tensor_server.shutdown()
        CacheManager.reset()


@pytest.fixture
def served_embedded_cache(tmp_path: Path):
    """An embedded cache backed by a real served (read-only) Flight server."""
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig
    from biopb_tensor_server.server import TensorFlightServer

    CacheManager.reset()
    CacheManager.initialize(
        CacheConfig(
            backend="file",
            file_cache_dir=tmp_path,
            file_max_total_bytes=128 * 1024 * 1024,
        )
    )

    port = _free_tcp_port()
    location = f"grpc://127.0.0.1:{port}"
    # Mirror the production embedded server: read-only over Flight.
    tensor_server = TensorFlightServer(location=location, writable=False)
    thread = threading.Thread(target=tensor_server.serve, daemon=True)
    thread.start()

    client = TensorFlightClient(location)
    for _ in range(20):
        try:
            client.health_check()
            break
        except Exception:
            time.sleep(0.1)
    else:
        tensor_server.shutdown()
        CacheManager.reset()
        raise RuntimeError("Timed out waiting for tensor server to start")

    try:
        yield EmbeddedTensorCache(
            tensor_server=tensor_server,
            external_location=location,
        )
    finally:
        tensor_server.shutdown()
        CacheManager.reset()


def test_embedded_cache_reports_serving(tmp_path: Path):
    """_start_embedded_tensor_cache marks itself ready (no scan stage).

    The embedded cache bypasses the CLI's scan/mark_ready lifecycle, so it must
    self-mark-ready immediately -- otherwise health would report STARTING
    forever and readiness-gating clients would wait indefinitely.
    """
    from biopb_tensor_server.cache import CacheManager
    from biopb_image_base.server import _start_embedded_tensor_cache

    CacheManager.reset()
    port = _free_tcp_port()
    tensor_server, location = _start_embedded_tensor_cache(
        cache_dir=tmp_path,
        cache_size=128 * 1024 * 1024,
        tensor_port=port,
        tensor_host="127.0.0.1",
    )
    try:
        assert tensor_server.is_ready is True
        client = TensorFlightClient(location)
        for _ in range(20):
            try:
                health = client.health_check()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for embedded cache to start")
        assert health["status"] == "SERVING"
    finally:
        tensor_server.shutdown()
        CacheManager.reset()


def _uniform_template() -> da.Array:
    return da.zeros((4, 4), chunks=(2, 2), dtype=np.float32)


def _non_uniform_template() -> da.Array:
    return da.zeros((4, 4), chunks=((2, 2), (1, 3)), dtype=np.float32)


def test_embedded_create_array_tracks_upload_status(embedded_cache: EmbeddedTensorCache):
    serialized = embedded_cache.create_array("cache:", ["Y", "X"], _uniform_template())

    source_id = serialized.tensor_descriptor.array_id
    assert serialized.location == "grpc://127.0.0.1:9999"
    assert len(serialized.endpoints) == 0

    status = embedded_cache.get_upload_status(source_id)
    assert status == {
        "source_id": source_id,
        "state": "PENDING",
        "expected_chunks": 4,
        "uploaded_chunks": 0,
    }

    embedded_cache.upload_array_chunks(
        source_id,
        ChunkBounds(start=[0, 0], stop=[2, 2]),
        np.ones((2, 2), dtype=np.float32),
    )
    status = embedded_cache.get_upload_status(source_id)
    assert status["state"] == "PENDING"
    assert status["uploaded_chunks"] == 1


@pytest.mark.parametrize("method_name", ["create_array", "create_source"])
def test_embedded_cache_rejects_non_uniform_chunks(
    embedded_cache: EmbeddedTensorCache,
    method_name: str,
):
    with pytest.raises(ValueError, match="Non-uniform dask chunks"):
        if method_name == "create_array":
            embedded_cache.create_array("cache:", ["Y", "X"], _non_uniform_template())
        else:
            embedded_cache.create_source(_non_uniform_template(), "cache:", ["Y", "X"])


def test_per_source_token_gates_readback(served_embedded_cache: EmbeddedTensorCache):
    """A result carries a per-source token; reads need it and it isn't enumerable."""
    import pyarrow.flight as flight

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    source_id = served_embedded_cache.create_source(data, "cache:", ["Y", "X"])
    serialized = served_embedded_cache.to_serialized_tensor(source_id)

    # The result advertises a non-empty per-source capability token.
    assert serialized.auth_token

    # With the token (carried in the SerializedTensor) the read-back succeeds.
    result = TensorFlightClient.tensor_from_pb(serialized).compute()
    np.testing.assert_array_equal(result, data)

    # Stripping the token must make the identical read fail closed.
    no_token = type(serialized)()
    no_token.CopyFrom(serialized)
    no_token.auth_token = ""
    with pytest.raises(flight.FlightError):
        TensorFlightClient.tensor_from_pb(no_token).compute()

    # The token-protected source is not enumerable via list_flights.
    location = served_embedded_cache._external_location
    listed = TensorFlightClient(location).list_sources()
    assert source_id not in listed