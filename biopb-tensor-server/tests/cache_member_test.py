"""A ``cache://`` member keeps its own bytes (step 7).

The format exists to skip the decode step -- what was
uploaded as a batch is served as that batch -- and, since this step, to do it
from a store of its own under the member rather than from a chunk-cache entry
that an eviction could take away.

What that buys, and what is tested here: the chunks outlive the process, they
are not on the cache's budget, a localhost reader is handed a byte range in the
member's own sealed segment with nothing resolved first, and the grid a read is
planned on is the grid the chunks were uploaded on.
"""

import json
import threading

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor import TensorFlightClient
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.cache_member import bounds_key
from biopb_tensor_server.adapters.members import MEMBER_DESCRIPTOR
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.chunk import mint_chunk_id
from biopb_tensor_server.core.config import CacheConfig

from tests import catalog_server

SHAPE = (4, 6)
CHUNK = (2, 3)


def _arr(fill=1, shape=SHAPE):
    return np.full(shape, fill, dtype=np.uint16)


def _add(client, source, field, arr=None, chunk_shape=CHUNK):
    arr = _arr() if arr is None else arr
    return client.add_tensor(f"cache://{source}/{field}", arr, chunk_shape=chunk_shape)


def _member(server, source, field):
    return server.sources.get(source).members[field]


def _chunk_id(member, start, stop):
    return mint_chunk_id(
        member.array_id,
        ChunkBounds(start=list(start), stop=list(stop)),
        content_version=member.content_version,
    )


class TestTheBytesAreItsOwn:
    def test_the_store_is_under_the_member(self, writable_server, client, source):
        """Segments beside a descriptor, in the member's directory: the layout
        the adoption pass reads back and the discard removes whole."""
        desc = _add(client, source, "img")
        client.upload_array(desc, _arr())

        store = writable_server.sources.get(source).member_store("img")
        names = sorted(p.name for p in store.iterdir())
        assert MEMBER_DESCRIPTOR in names
        assert any(n.endswith(".arrow") for n in names)

    def test_the_chunks_are_not_on_the_caches_budget(
        self, writable_server, client, source
    ):
        """The segments are the tensor, not a cache of it, so they are outside
        ``max_total_bytes`` and outside the eviction sweep."""
        before = CacheManager.get_instance().stats().total_bytes
        desc = _add(client, source, "big", arr=_arr(1, (64, 64)), chunk_shape=(32, 32))
        client.upload_array(desc, _arr(1, (64, 64)))

        assert CacheManager.get_instance().stats().total_bytes == before

    def test_a_chunk_reads_back_as_the_batch_it_was_sent(self, client, source):
        desc = _add(client, source, "img")
        client.upload_array(desc, _arr(7))

        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), _arr(7)
        )

    def test_a_chunk_never_uploaded_reads_as_zeros(self, client, source):
        """Sparse is cheap: one uploaded chunk, the rest background. A gap is a
        bounds the index does not hold, and it is answered rather than raised."""
        desc = _add(client, source, "sparse")
        client.upload_chunk(
            desc, ChunkBounds(start=[0, 0], stop=[2, 3]), _arr(5, CHUNK)
        )
        client.set_upload_status(desc, "READY")

        got = client.get_tensor(desc.array_id).compute()
        assert got[:2, :3].max() == 5
        assert got[2:, 3:].max() == 0


class TestItSurvivesARestart:
    """What step 7 is for: before it, a ``cache://`` member's chunks were cache
    entries, so the tensor died with the process that uploaded it."""

    @staticmethod
    def _server(tmp_path):
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        server = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=tmp_path / "w"
        )
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        return server

    def test_it_reads_back_in_the_next_life(self, tmp_path):
        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            source = client.register_source("keepme")
            desc = _add(client, source, "img")
            client.upload_array(desc, _arr(9))
            client.close()
        finally:
            first.shutdown()

        second = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{second.port}")
            np.testing.assert_array_equal(
                client.get_tensor(desc.array_id).compute(), _arr(9)
            )
            client.close()
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_the_chunk_ids_are_this_builds_and_the_bytes_are_not(self, tmp_path):
        """The index keys by bounds; the ids are minted over it at registration.

        So an id a client cached still resolves after a restart -- the token is
        persisted -- and an epoch bump would rebuild the map without touching a
        byte on disk.
        """
        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            source = client.register_source("ids")
            desc = _add(client, source, "img")
            client.upload_array(desc, _arr())
            before = _chunk_id(_member(first, source, "img"), [0, 0], [2, 3])
            client.close()
        finally:
            first.shutdown()

        second = self._server(tmp_path)
        try:
            member = _member(second, source, "img")
            assert _chunk_id(member, [0, 0], [2, 3]) == before
            assert before in member._written_chunks
        finally:
            second.shutdown()
            CacheManager.reset()

    def test_a_member_left_pending_is_swept_rather_than_adopted(self, tmp_path):
        """A crash before READY leaves a store nothing can publish now."""
        first = self._server(tmp_path)
        try:
            client = TensorFlightClient(f"grpc://localhost:{first.port}")
            source = client.register_source("crashed")
            desc = _add(client, source, "half")
            client.upload_chunk(
                desc, ChunkBounds(start=[0, 0], stop=[2, 3]), _arr(5, CHUNK)
            )
            store = first.sources.get(source).member_store("half")
            client.close()
        finally:
            first.shutdown()  # never published

        assert store.exists()
        second = self._server(tmp_path)
        try:
            assert not store.exists()
            assert second.sources.get(source).list_tensor_descriptors() == []
        finally:
            second.shutdown()
            CacheManager.reset()


class TestReadySealsTheSegments:
    def test_the_sidecar_is_written_at_ready(self, writable_server, client, source):
        """The index the next boot reads instead of faulting the bodies, and
        what makes every chunk servable by byte range."""
        desc = _add(client, source, "img")
        client.upload_chunk(
            desc, ChunkBounds(start=[0, 0], stop=[2, 3]), _arr(5, CHUNK)
        )
        store = writable_server.sources.get(source).member_store("img")

        assert not list(store.glob("*.idx"))
        client.set_upload_status(desc, "READY")
        assert list(store.glob("*.idx"))

    def test_the_marker_flips_to_ready(self, writable_server, client, source):
        desc = _add(client, source, "img")
        store = writable_server.sources.get(source).member_store("img")
        state = lambda: json.loads((store / MEMBER_DESCRIPTOR).read_text())["biopb"][  # noqa: E731
            "upload"
        ]["state"]

        assert state() == "pending"
        client.upload_array(desc, _arr())
        assert state() == "ready"


class TestTheLocalhostFastPath:
    def test_it_answers_the_members_own_segment(self, writable_server, client, source):
        """Cold, and with no copy into the chunk cache: the stored batch is
        already the one the client wants, so there is nothing to resolve."""
        desc = _add(client, source, "img")
        client.upload_array(desc, _arr(3))
        member = _member(writable_server, source, "img")

        located = json.loads(
            writable_server._handle_chunk_locate(_chunk_id(member, [0, 0], [2, 3]))
        )

        assert located["available"] is True
        assert located["segment_path"].startswith(str(member.store))
        assert located["byte_length"] > 0
        assert CacheManager.get_instance().stats().total_bytes == 0

    def test_a_scaled_ticket_falls_through_to_the_cache(
        self, writable_server, client, source
    ):
        """A scaled read has no stored batch: it is assembled from the stored
        chunks, then reduced and cached like any other."""
        desc = _add(client, source, "img")
        client.upload_array(desc, _arr(3))
        member = _member(writable_server, source, "img")
        scaled = mint_chunk_id(
            member.array_id,
            ChunkBounds(start=[0, 0], stop=[4, 6]),
            scale_hint=(2, 2),
            content_version=member.content_version,
        )

        assert member.locate_chunk(scaled) is None


class TestWhatItRefuses:
    def test_a_chunk_written_twice(self, writable_server, client, source):
        """Put-once: a second copy would strand the first as bytes no index
        names, and a reader may already hold a range into it."""
        desc = _add(client, source, "img")
        bounds = ChunkBounds(start=[0, 0], stop=[2, 3])
        client.upload_chunk(desc, bounds, _arr(5, CHUNK))

        with pytest.raises(flight.FlightError, match="written once"):
            client.upload_chunk(desc, bounds, _arr(6, CHUNK))

    def test_a_discard_takes_the_store_with_it(self, writable_server, client, source):
        desc = _add(client, source, "img")
        client.upload_array(desc, _arr())
        store = writable_server.sources.get(source).member_store("img")

        assert store.exists()
        client.set_upload_status(desc, "DISCARDED", "replaced")
        assert not store.exists()


class TestTheStoreFollowsThePlan:
    def test_the_grid_is_the_uploaded_one_exactly(
        self, writable_server, client, source
    ):
        """Unlike a zarr member, which coalesces: this format stores the chunks
        as they arrive, and the index knows those bounds and no others."""
        desc = _add(client, source, "img")

        assert tuple(desc.chunk_shape) == CHUNK
        assert (
            tuple(_member(writable_server, source, "img").get_transfer_chunk_size())
            == CHUNK
        )

    def test_the_index_is_keyed_by_bounds(self, writable_server, client, source):
        """Not by chunk id: an id carries the serving-semantics epoch, which
        moves on an upgrade that changes what the bytes mean. These do not."""
        desc = _add(client, source, "img")
        client.upload_array(desc, _arr())

        stored = set(_member(writable_server, source, "img")._chunks.stored_keys())
        assert bounds_key(ChunkBounds(start=[0, 0], stop=[2, 3])) in stored
        assert all(b"/" not in key for key in stored)
