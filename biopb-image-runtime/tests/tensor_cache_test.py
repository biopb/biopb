"""Focused tests for image-server tensor cache helpers."""

from __future__ import annotations

import secrets
import socket
import threading
import time
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_image_base.server import RESULT_TTL_S, EmbeddedTensorCache


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


@pytest.fixture
def embedded_cache(tmp_path: Path) -> EmbeddedTensorCache:
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.core.config import CacheConfig
    from biopb_tensor_server.serving.server import TensorFlightServer

    CacheManager.reset()
    CacheManager.initialize(
        CacheConfig(
            file_cache_dir=tmp_path,
            file_max_total_bytes=128 * 1024 * 1024,
        )
    )
    # Production's shape: the Flight write path refused, a write_dir anyway --
    # that is what gives the server the scratch source results are added to.
    tensor_server = TensorFlightServer(
        location="grpc://0.0.0.0:0",
        writable=False,
        write_dir=tmp_path / "uploads",
        scratch_ttl=RESULT_TTL_S,
    )

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
    from biopb_tensor_server.core.config import CacheConfig
    from biopb_tensor_server.serving.server import TensorFlightServer

    CacheManager.reset()
    CacheManager.initialize(
        CacheConfig(
            file_cache_dir=tmp_path,
            file_max_total_bytes=128 * 1024 * 1024,
        )
    )

    port = _free_tcp_port()
    location = f"grpc://127.0.0.1:{port}"
    # Mirror the production embedded server: read-only over Flight, a
    # write_dir for the scratch source its results go on, and a server-wide
    # token nobody is given -- there so `_authorize` fails closed.
    server_token = secrets.token_urlsafe(16)
    tensor_server = TensorFlightServer(
        location=location,
        token=server_token,
        writable=False,
        write_dir=tmp_path / "uploads",
        scratch_ttl=RESULT_TTL_S,
    )
    thread = threading.Thread(target=tensor_server.serve, daemon=True)
    thread.start()

    # Ungated, which is why a readiness probe needs no credential at all.
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
        cache = EmbeddedTensorCache(
            tensor_server=tensor_server,
            external_location=location,
        )
        cache.server_token = server_token
        yield cache
    finally:
        tensor_server.shutdown()
        CacheManager.reset()


def test_embedded_cache_reports_serving(tmp_path: Path):
    """_start_embedded_tensor_cache marks itself ready (no scan stage).

    The embedded cache bypasses the CLI's scan/mark_ready lifecycle, so it must
    self-mark-ready immediately -- otherwise health would report STARTING
    forever and readiness-gating clients would wait indefinitely.
    """
    from biopb_image_base.server import _start_embedded_tensor_cache
    from biopb_tensor_server.cache import CacheManager

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


def test_a_result_is_a_tensor_of_the_scratch_source(embedded_cache):
    """Added to a source that already exists, not registered as one of its own.

    That is what gives it a deadline and a grant of its own; the marked
    ``@fields`` segment is what keeps its id off a native one.
    """
    array_id = embedded_cache.create_source(
        np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
    )

    source_id, _, field = array_id.partition("/")
    assert source_id == "scratch"
    assert field.startswith("@fields/")


def test_a_result_carries_a_deadline(embedded_cache):
    """The scratch source caps every upload on it, so nothing the consumer
    never collected is held for the life of the process."""
    array_id = embedded_cache.create_source(
        np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
    )
    descriptor = TensorFlightClient.descriptor_from_pb(
        embedded_cache.to_serialized_tensor(array_id)
    )

    assert 0 < descriptor.ttl_seconds <= RESULT_TTL_S


def test_two_results_under_one_name_do_not_collide(embedded_cache):
    """A field is taken while its tensor is served, so the name is salted --
    a servicer naming every result 'cache:' would otherwise be refused the
    second one."""
    first = embedded_cache.create_source(
        np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
    )
    second = embedded_cache.create_source(
        np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
    )

    assert first != second


def test_each_result_carries_its_own_grant(embedded_cache):
    """The reason the grant is not on the source: every result shares the
    scratch source, so one producer's token must not open another's."""
    mine = embedded_cache.create_source(
        np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
    )
    theirs = embedded_cache.create_source(
        np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
    )

    my_token = embedded_cache.to_serialized_tensor(mine).auth_token
    their_token = embedded_cache.to_serialized_tensor(theirs).auth_token

    assert my_token and their_token and my_token != their_token


def test_a_restart_does_not_adopt_the_last_run_s_results(tmp_path: Path):
    """They are cleared, not re-served. A result's capability is minted in
    memory, so one adopted from an earlier life would come back readable by
    anyone who reaches the port.
    """
    from biopb_image_base.server import _start_embedded_tensor_cache
    from biopb_tensor_server.cache import CacheManager

    def _run(port):
        CacheManager.reset()
        server, _ = _start_embedded_tensor_cache(
            cache_dir=tmp_path,
            cache_size=128 * 1024 * 1024,
            tensor_port=port,
            tensor_host="127.0.0.1",
        )
        return server

    first = _run(_free_tcp_port())
    try:
        cache = EmbeddedTensorCache(first, "grpc://127.0.0.1:9999")
        array_id = cache.create_source(
            np.zeros((4, 4), dtype=np.float32), "cache:", ["Y", "X"]
        )
        field = array_id.partition("/")[2]
        assert first.sources.get("scratch").attached_tensor(field) is not None
    finally:
        first.shutdown()

    second = _run(_free_tcp_port())
    try:
        assert second.sources.get("scratch").attached_tensors == {}
    finally:
        second.shutdown()
        CacheManager.reset()


def _uniform_template() -> da.Array:
    return da.zeros((4, 4), chunks=(2, 2), dtype=np.float32)


def _non_uniform_template() -> da.Array:
    return da.zeros((4, 4), chunks=((2, 2), (1, 3)), dtype=np.float32)


def _flight_info(serialized):
    import pyarrow.flight as flight

    return flight.FlightInfo.deserialize(serialized.flight_info)


def _array_id(serialized) -> str:
    return TensorFlightClient.descriptor_from_pb(serialized).array_id


def test_embedded_create_array_tracks_upload_status(
    embedded_cache: EmbeddedTensorCache,
):
    serialized = embedded_cache.create_array("cache:", ["Y", "X"], _uniform_template())

    source_id = _array_id(serialized)
    assert serialized.location == "grpc://127.0.0.1:9999"
    # Describe-only: the consumer plans its own read once the result is READY.
    assert len(_flight_info(serialized).endpoints) == 0

    status = embedded_cache.get_upload_status(source_id)
    assert status == {
        "source_id": source_id,
        "state": "PENDING",
        "expected_chunks": 4,
        "uploaded_chunks": 0,
        "reason": "",
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
    """A result carries a per-source token, and reads need it."""
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

    # Two gates, and the outer one is the server-wide token nobody holds: a
    # caller that merely reaches the port cannot enumerate anything.
    location = served_embedded_cache._external_location
    with pytest.raises(flight.FlightUnauthenticatedError):
        TensorFlightClient(location).list_sources()

    # And behind it, this server is catalog-less (metadata_db=None): a result
    # is reachable only through the array_id its SerializedTensor carries. The
    # catalog surface says so rather than answering "empty".
    with pytest.raises(flight.FlightError, match="no catalog"):
        TensorFlightClient(
            location, token=served_embedded_cache.server_token
        ).list_sources()


def test_finish_seals_the_result_and_refuses_later_writes(
    embedded_cache: EmbeddedTensorCache,
):
    """The success counterpart of discard, and the only route to READY.

    READY is what a consumer polling for the result waits on, and it seals the
    upload in the same move -- a late chunk has no way to invalidate a read
    already served under its ``chunk_id``.
    """
    import pyarrow.flight as flight

    serialized = embedded_cache.create_array("cache:", ["Y", "X"], _uniform_template())
    source_id = _array_id(serialized)

    assert embedded_cache.finish(source_id)["state"] == "READY"

    with pytest.raises(flight.FlightCancelledError):
        embedded_cache.upload_array_chunks(
            source_id,
            ChunkBounds(start=[0, 0], stop=[2, 2]),
            np.ones((2, 2), dtype=np.float32),
        )


def test_discard_refuses_later_writes_with_the_reason(
    embedded_cache: EmbeddedTensorCache,
):
    """What a servicer does when its background job dies or is told to stop.

    A write still in flight then fails with the reason rather than with a
    missing source, which is what tells the worker whether it was given up on
    or hit a genuine fault (biopb/biopb#1).
    """
    import pyarrow.flight as flight

    serialized = embedded_cache.create_array("cache:", ["Y", "X"], _uniform_template())
    source_id = _array_id(serialized)

    status = embedded_cache.discard(source_id, "client disconnected")

    assert status["state"] == "DISCARDED"
    assert (
        embedded_cache.get_upload_status(source_id)["reason"] == "client disconnected"
    )

    with pytest.raises(flight.FlightCancelledError, match="client disconnected"):
        embedded_cache.upload_array_chunks(
            source_id,
            ChunkBounds(start=[0, 0], stop=[2, 2]),
            np.ones((2, 2), dtype=np.float32),
        )
