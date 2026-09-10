"""Retention classes: what a cache miss costs, decided from the chunk.

Eviction on the file backend is whole-segment, so the class cannot be consulted
when a victim is chosen -- it has to pick the segment a chunk is written into.
These tests pin that down at both ends: the pool a chunk lands in, and the order
the pools are reclaimed in.
"""

import numpy as np
import pytest
import zarr
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.cache import (
    ArrowFileBackend,
    ArrowFileConfig,
    CacheManager,
    MemoryCacheBackend,
    MemoryCacheConfig,
)
from biopb_tensor_server.core.chunk import encode_chunk_id, encode_chunk_id_with_scale
from biopb_tensor_server.core.chunk_batch import pack_chunk_batch
from biopb_tensor_server.core.config import CacheConfig, PyramidConfig
from biopb_tensor_server.core.retention import set_active_pyramid_config


def _store(backend, key, values, retention="normal", size_bytes=None):
    """Store one chunk in the same wire schema the server writes."""
    batch = pack_chunk_batch(np.asarray(values, dtype=np.uint8))
    backend.start_compute(key, retention)
    backend.complete_entry(key, batch, size_bytes or len(values))
    backend.release(key)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


class TestPoolAssignment:
    def test_the_class_keys_the_pool(self, cache_dir):
        """A cheap chunk and a normal one never share a segment."""
        backend = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        _store(backend, b"c", [1] * 32, retention="cheap")
        _store(backend, b"n", [2] * 32, retention="normal")

        pools = set(backend._pool_queues)
        assert ("cheap", "tiny") in pools
        assert ("normal", "tiny") in pools
        assert backend._metadata[b"c"].segment_id != backend._metadata[b"n"].segment_id
        backend.close()

    def test_default_is_normal(self, cache_dir):
        backend = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        _store(backend, b"k", [1] * 32)

        assert set(backend._pool_queues) == {("normal", "tiny")}
        backend.close()

    def test_stats_name_both_halves_of_the_key(self, cache_dir):
        backend = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        _store(backend, b"c", [1] * 32, retention="cheap")

        assert "cheap-tiny" in backend.stats().pool_stats
        backend.close()


class TestReclamationOrder:
    def test_cheap_goes_first_however_well_it_is_hit(self, cache_dir):
        """The class outranks the hit rate: that is the point of the split.

        The cheap pool here is the *better* pool by hit rate, which is exactly
        the case the old lowest-hit-rate-wins selection got backwards once a
        miss stopped costing the same everywhere.
        """
        backend = ArrowFileBackend(
            ArrowFileConfig(
                cache_dir=cache_dir, max_segment_bytes=400, max_total_bytes=1600
            )
        )
        for i in range(2):
            _store(backend, f"cheap{i}".encode(), [i] * 300, "cheap", 300)
        for i in range(2):
            _store(backend, f"norm{i}".encode(), [i] * 300, "normal", 300)
        # Hit the cheap pool, and only the cheap pool.
        for _ in range(3):
            backend.start_compute(b"cheap0")
            backend.release(b"cheap0")

        assert backend._select_pool_for_eviction()[0] == "cheap"
        backend.close()

    def test_pinned_is_never_a_victim(self, cache_dir):
        backend = ArrowFileBackend(
            ArrowFileConfig(
                cache_dir=cache_dir, max_segment_bytes=400, max_total_bytes=1600
            )
        )
        _store(backend, b"p", [1] * 300, "pinned", 300)

        assert backend._select_pool_for_eviction() is None
        assert backend._evict_segment_sieve_k() is False

        _store(backend, b"n", [2] * 300, "normal", 300)
        assert backend._select_pool_for_eviction()[0] == "normal"
        backend.close()

    def test_a_full_cache_reclaims_the_cheap_segments(self, cache_dir):
        """End to end: writing past the budget unlinks cheap segments, not normal
        ones, while every pool still has cold segments to give."""
        backend = ArrowFileBackend(
            ArrowFileConfig(
                cache_dir=cache_dir, max_segment_bytes=400, max_total_bytes=2000
            )
        )
        for i in range(3):
            _store(backend, f"norm{i}".encode(), [i] * 300, "normal", 300)
        for i in range(6):
            _store(backend, f"cheap{i}".encode(), [i] * 300, "cheap", 300)

        assert backend.stats().evictions > 0
        assert all(f"norm{i}".encode() in backend._metadata for i in range(3))
        backend.close()


class TestClassSurvivesRestart:
    def test_the_sidecar_carries_it(self, cache_dir):
        backend = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        _store(backend, b"c", [1] * 32, retention="cheap")
        backend.close()  # seals the segment and writes its .idx

        reopened = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        assert set(reopened._pool_queues) == {("cheap", "tiny")}
        reopened.close()

    def test_a_body_walk_defaults_to_normal(self, cache_dir):
        """No sidecar, no record of the class -- the segment rejoins as normal.

        It costs that segment its place in the reclamation order until it turns
        over, which is the documented price of not putting the class in the
        segment body (where it would be a wire-format change).
        """
        backend = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        _store(backend, b"c", [1] * 32, retention="cheap")
        backend.close()
        for idx in (cache_dir / "segments").glob("*.idx"):
            idx.unlink()

        reopened = ArrowFileBackend(ArrowFileConfig(cache_dir=cache_dir))
        assert set(reopened._pool_queues) == {("normal", "tiny")}
        reopened.close()


class TestMemoryBackend:
    """No pooling there -- eviction is per-entry -- so only "pinned" bites."""

    def test_pinned_entries_are_not_evicted(self):
        backend = MemoryCacheBackend(MemoryCacheConfig(max_entries=4, max_bytes=4 * 32))
        _store(backend, b"pin", [0] * 32, "pinned", 32)
        for i in range(8):
            _store(backend, f"k{i}".encode(), [i] * 32, "normal", 32)

        assert b"pin" in backend._entries
        assert backend.stats().evictions > 0


class TestDeclarationAtTheReadSeam:
    """Who decides the class, at the one call every cached chunk goes through.

    The chunk decides, not the caller: the precache and a client can ask for the
    same coarse chunk_id, and whichever gets there first must not fix the class
    for good.
    """

    # Knobs that give a 64x64 tensor a one-rung computed ladder at scale (4, 4).
    LADDER_CFG = PyramidConfig(threshold=16, plane_max_pixels=1024)

    @pytest.fixture
    def ladder(self):
        """Install the server's ladder the way TensorFlightServer.__init__ does."""
        set_active_pyramid_config(self.LADDER_CFG)
        yield
        set_active_pyramid_config(None)

    @pytest.fixture
    def adapter(self, tmp_path):
        path = tmp_path / "a.zarr"
        arr = zarr.open_array(
            str(path), mode="w", shape=(64, 64), chunks=(32, 32), dtype="uint16"
        )
        arr[:] = (np.arange(64 * 64, dtype=np.uint16) % 4093).reshape(64, 64)
        return ZarrAdapter(zarr.open_array(str(path), mode="r"), "src", ["y", "x"])

    @pytest.fixture
    def manager(self, cache_dir):
        mgr = CacheManager(CacheConfig(backend="file", file_cache_dir=cache_dir))
        yield mgr
        mgr.close()

    def _chunk_id(self, kind, scale=None):
        bounds = ChunkBounds(start=[0, 0], stop=[64, 64])
        if kind == "scaled":
            return encode_chunk_id_with_scale("src", bounds, scale, "area")
        array_id = "src/1" if kind == "native" else "src"
        return encode_chunk_id(array_id, ChunkBounds(start=[0, 0], stop=[32, 32]))

    def _classes(self, manager):
        return {key[0] for key in manager.backend._pool_queues}

    @pytest.mark.parametrize(
        "chunk,expected",
        [
            # A rung of the ladder: what a client returns to on every open, and
            # what the precache warms.
            (("scaled", (4, 4)), "normal"),
            # A one-off scale, regenerable from the chunks under it.
            (("scaled", (2, 2)), "cheap"),
            (("full", None), "normal"),
            # A native level: minted against the level's own store, so it
            # arrives unscaled and is a level-0 read here.
            (("native", None), "normal"),
        ],
    )
    def test_the_chunk_decides_its_class(
        self, adapter, manager, ladder, chunk, expected
    ):
        adapter.resolve_chunk_data(self._chunk_id(*chunk), manager)

        assert self._classes(manager) == {expected}

    def test_the_ladder_is_the_installed_config_not_a_default(self, adapter):
        """Under the shipped knobs a 64x64 plane has no computed levels at all,
        so the chunk_id that is a rung under the fixture's config is a one-off.

        This is also why the class cannot come from the caller: the same
        chunk_id has to classify the same whether the precache or a client got
        there first, and only the chunk and the installed ladder decide it.
        """
        assert adapter._retention_for_chunk(self._chunk_id("scaled", (4, 4))) == "cheap"

    def test_an_unscaled_chunk_never_builds_a_ladder(self, adapter, ladder):
        """Full resolution is normal by inspection, so a source nobody reads a
        scaled chunk from pays nothing for the classification."""

        def explode():
            raise AssertionError("ladder built for an unscaled chunk")

        adapter.get_tensor_descriptor = explode
        assert adapter._retention_for_chunk(self._chunk_id("full")) == "normal"

    def test_the_ladder_is_memoized(self, adapter, ladder, monkeypatch):
        """The hot path must not build a descriptor per chunk."""
        adapter._retention_for_chunk(self._chunk_id("scaled", (4, 4)))

        def explode():
            raise AssertionError("descriptor rebuilt on a memoized lookup")

        monkeypatch.setattr(adapter, "get_tensor_descriptor", explode)
        assert adapter._retention_for_chunk(self._chunk_id("scaled", (2, 2))) == "cheap"
