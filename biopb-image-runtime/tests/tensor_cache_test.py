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

from biopb_image_base.server import EmbeddedTensorCache, ExternalTensorCache


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
def external_cache(tmp_path: Path) -> ExternalTensorCache:
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
    tensor_server = TensorFlightServer(location=location, writable=True)
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
        yield ExternalTensorCache(location)
    finally:
        tensor_server.shutdown()
        CacheManager.reset()


def _uniform_template() -> da.Array:
    return da.zeros((4, 4), chunks=(2, 2), dtype=np.float32)


def _non_uniform_template() -> da.Array:
    return da.zeros((4, 4), chunks=((2, 2), (1, 3)), dtype=np.float32)


def _upload_all_chunks(cache, source_id: str) -> None:
    chunks = [
        (ChunkBounds(start=[0, 0], stop=[2, 2]), np.ones((2, 2), dtype=np.float32)),
        (ChunkBounds(start=[0, 2], stop=[2, 4]), np.full((2, 2), 2.0, dtype=np.float32)),
        (ChunkBounds(start=[2, 0], stop=[4, 2]), np.full((2, 2), 3.0, dtype=np.float32)),
        (ChunkBounds(start=[2, 2], stop=[4, 4]), np.full((2, 2), 4.0, dtype=np.float32)),
    ]
    for bounds, chunk in chunks:
        cache.upload_array_chunks(source_id, bounds, chunk)


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


def test_external_create_array_and_chunk_uploads_are_fetchable(
    external_cache: ExternalTensorCache,
):
    serialized = external_cache.create_array("cache:", ["Y", "X"], _uniform_template())

    source_id = serialized.tensor_descriptor.array_id
    assert serialized.location == external_cache._location
    assert len(serialized.endpoints) == 0

    status = external_cache.get_upload_status(source_id)
    assert status["state"] == "PENDING"
    assert status["expected_chunks"] == 4
    assert status["uploaded_chunks"] == 0

    _upload_all_chunks(external_cache, source_id)

    status = external_cache.get_upload_status(source_id)
    assert status == {
        "source_id": source_id,
        "state": "READY",
        "expected_chunks": 4,
        "uploaded_chunks": 4,
    }

    result = TensorFlightClient.tensor_from_pb(serialized).compute()
    expected = np.block(
        [
            [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 2.0, dtype=np.float32)],
            [np.full((2, 2), 3.0, dtype=np.float32), np.full((2, 2), 4.0, dtype=np.float32)],
        ]
    )
    np.testing.assert_array_equal(result, expected)


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


@pytest.mark.parametrize("method_name", ["create_array", "create_source"])
def test_external_cache_rejects_non_uniform_chunks(
    external_cache: ExternalTensorCache,
    method_name: str,
):
    with pytest.raises(ValueError, match="Non-uniform dask chunks"):
        if method_name == "create_array":
            external_cache.create_array("cache:", ["Y", "X"], _non_uniform_template())
        else:
            external_cache.create_source(_non_uniform_template(), "cache:", ["Y", "X"])