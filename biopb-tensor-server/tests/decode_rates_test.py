"""Measured retention: what a full-resolution chunk actually cost to decode.

The declared rule (``cache_retention_test.py``) can only ever call a
full-resolution chunk "normal" -- it has no ladder to judge one against. These
tests cover the half that separates them by measurement: where the sample is
taken, which reads are refused as samples, and the two source kinds that must
never be classified this way at all.
"""

import shutil

import numpy as np
import pytest
import zarr
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.chunk import encode_chunk_id, encode_chunk_id_with_scale
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from biopb_tensor_server.core.retention import (
    DecodeRates,
    active_decode_rates,
    set_active_decode_rates,
)

# An ordinary full-resolution chunk. Nothing turns on the size -- every read is
# a sample -- so this is just a realistic number to divide by.
CHUNK = 4 << 20


@pytest.fixture(autouse=True)
def restore_active_rates():
    """The table is process state; a test that installs one must not leak it."""
    previous = active_decode_rates()
    yield
    set_active_decode_rates(previous)


def _converge(rates, array_id, mbps, samples=1):
    """Feed ``samples`` identical reads, so the EMA sits exactly on ``mbps``.

    One is enough: the first sample sets the EMA directly (there is no prior
    value to blend against).
    """
    for _ in range(samples):
        rates.record(array_id, CHUNK, CHUNK / 1e6 / mbps)


class TestTheStatistic:
    def test_one_rate_per_array(self):
        rates = DecodeRates()
        _converge(rates, "src/0", 800.0)
        _converge(rates, "src/1", 100.0)

        assert rates.rate("src/0") == pytest.approx(800.0)
        assert rates.rate("src/1") == pytest.approx(100.0)

    def test_it_follows_the_recent_reads(self):
        """An EMA, not a mean: a source that gets slower is eventually described
        by what it costs now, which is the whole reason the class can move."""
        rates = DecodeRates()
        _converge(rates, "src", 1000.0, samples=20)
        _converge(rates, "src", 100.0, samples=20)

        assert rates.rate("src") == pytest.approx(100.0, rel=0.05)

    def test_an_unread_array_has_no_rate(self):
        rates = DecodeRates()

        assert rates.rate("never-read") is None

    def test_a_small_read_is_a_sample_like_any_other(self):
        """A small chunk's MB/s is mostly per-call cost -- the re-mmap in
        biopb/biopb#816, a seek, opening a handle -- but that cost is part of
        what rebuilding it would take, which is the question. Dropping these
        left an array whose chunks are all small permanently unmeasured, and so
        permanently "normal", when some of them are the cheapest things in the
        cache."""
        tiny = 64 << 10
        rates = DecodeRates(cheap_mbps=100.0)
        rates.record("tiny", tiny, tiny / 1e6 / 1024)  # 64 KiB at 1024 MB/s

        assert rates.rate("tiny") == pytest.approx(1024.0)
        assert rates.retention_for_array("tiny") == "cheap"

    def test_a_chunk_size_shows_up_in_the_rate(self):
        """The number is cost per byte as the read path issues it, not what the
        format can sustain: the same per-call overhead over fewer bytes reads
        slower, which is why a threshold has to come off a real table."""
        rates = DecodeRates()
        overhead = 200e-6  # a fixed per-read cost, independent of size
        rates.record("small", 64 << 10, overhead)
        rates.record("large", 8 << 20, overhead)

        assert rates.snapshot()["small"][0] < rates.snapshot()["large"][0]

    def test_a_zero_duration_read_is_not_a_sample(self):
        """The only read dropped: a duration the clock could not resolve cannot
        be divided by."""
        rates = DecodeRates()
        rates.record("src", CHUNK, 0.0)

        assert rates.rate("src") is None


class TestTheThreshold:
    def test_unset_measures_but_classifies_nothing(self):
        """The shipped default. There is no portable "fast enough", so until an
        operator has read the table the measurement is a diagnostic only."""
        rates = DecodeRates()
        _converge(rates, "src", 10_000.0)

        assert rates.rate("src") == pytest.approx(10_000.0)
        assert rates.retention_for_array("src") == "normal"

    def test_over_the_threshold_is_cheap(self):
        rates = DecodeRates(cheap_mbps=500.0)
        _converge(rates, "src", 800.0)

        assert rates.retention_for_array("src") == "cheap"

    def test_under_the_threshold_keeps_the_declared_answer(self):
        rates = DecodeRates(cheap_mbps=500.0)
        _converge(rates, "src", 100.0)

        assert rates.retention_for_array("src") == "normal"

    def test_an_unmeasured_array_keeps_the_declared_answer(self):
        """Why an EMA that starts unset needs no special case: an array nobody
        has read at full resolution has no full-resolution chunks to classify,
        and if one arrives it is normal -- the answer it had before this
        existed."""
        rates = DecodeRates(cheap_mbps=1.0)

        assert rates.retention_for_array("never-read") == "normal"


class TestPersistence:
    """The table lives in the catalog database, beside `rois` -- not beside the
    cache segments it describes. The cache directory is the operator's to
    delete; clearing it should cost the bytes, not a run of measurement."""

    @staticmethod
    def _catalog(tmp_path):
        db = MetadataDatabase(store_path=tmp_path / "catalog.duckdb")
        db.open()
        return db

    def test_a_run_does_not_reserve_its_warmup(self, tmp_path):
        """Without this every restart re-measures, and every restart stamps a
        stratum of "normal" segments before it converges."""
        first = self._catalog(tmp_path)
        written = DecodeRates()
        written.attach(first)
        _converge(written, "src", 800.0)
        written.flush()
        first.close()

        second = self._catalog(tmp_path)
        try:
            restored = DecodeRates(cheap_mbps=500.0)
            restored.attach(second)

            assert restored.rate("src") == pytest.approx(800.0)
            assert restored.retention_for_array("src") == "cheap"
        finally:
            second.close()

    def test_clearing_the_cache_does_not_clear_the_measurements(self, tmp_path):
        """The reason this is not a file under file_cache_dir. An operator
        reclaiming disk is not asking to re-measure every source."""
        cache_dir = tmp_path / "cache"
        db = self._catalog(tmp_path)
        try:
            manager = CacheManager(
                CacheConfig(backend="file", file_cache_dir=cache_dir)
            )
            active_decode_rates().attach(db)
            _converge(active_decode_rates(), "src", 800.0)
            manager.close()

            shutil.rmtree(cache_dir)

            after = DecodeRates()
            after.attach(db)
            assert after.rate("src") == pytest.approx(800.0)
        finally:
            db.close()

    def test_a_live_measurement_beats_a_stored_one(self, tmp_path):
        """Attaching cannot undo what this run already measured: the store is
        the older observation by construction, whatever its timestamp says."""
        db = self._catalog(tmp_path)
        try:
            db.save_decode_rates({"src": (800.0, 40)})
            rates = DecodeRates()
            _converge(rates, "src", 100.0)
            rates.attach(db)

            assert rates.rate("src") == pytest.approx(100.0)
        finally:
            db.close()

    def test_the_table_is_read_as_sql_rather_than_through_an_action(self, tmp_path):
        """The whole client surface. A table an operator wants to sort, filter
        and join against `sources` is better asked in SQL than in an RPC that
        can only ever hand back all of it."""
        db = self._catalog(tmp_path)
        try:
            db.save_decode_rates({"fast": (800.0, 12), "slow": (90.0, 3)})

            info = db.handle_query(
                "SELECT array_id, samples FROM decode_rates "
                "WHERE mbps > 500 ORDER BY array_id"
            )
            table = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())

            assert table.to_pydict() == {"array_id": ["fast"], "samples": [12]}
        finally:
            db.close()

    def test_the_first_write_may_be_what_opens_the_database(self, tmp_path):
        """A write that arrives before anything has opened the connection has
        to build it -- and _get_connection takes the same non-reentrant write
        lock, so taking the lock first deadlocked the read path here."""
        db = MetadataDatabase(store_path=tmp_path / "catalog.duckdb")
        try:
            db.save_decode_rates({"src": (800.0, 4)})

            assert db.load_decode_rates() == {"src": (800.0, 4)}
        finally:
            db.close()

    def test_a_schema_change_drops_the_rows_rather_than_refusing(self, tmp_path):
        """The difference from `rois`, which refuses a catalog it cannot
        migrate: every row here is re-measurable by reading, so a version bump
        costs one warmup and needs no migration ladder."""
        store = tmp_path / "catalog.duckdb"
        first = MetadataDatabase(store_path=store)
        first.open()
        first.save_decode_rates({"src": (800.0, 12)})
        first._get_connection().execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES "
            "('decode_rates_schema_version', '0')"
        )
        first.close()

        second = MetadataDatabase(store_path=store)
        second.open()
        try:
            assert second.load_decode_rates() == {}
            second.save_decode_rates({"src": (100.0, 1)})
            assert second.load_decode_rates() == {"src": (100.0, 1)}
        finally:
            second.close()

    def test_a_store_that_will_not_load_leaves_this_run_measuring(self):
        """One warmup, not a refusal to serve: these are diagnostics."""

        class Broken:
            def load_decode_rates(self):
                raise RuntimeError("no")

            def save_decode_rates(self, rows):
                raise RuntimeError("no")

        rates = DecodeRates()
        rates.attach(Broken())
        _converge(rates, "src", 800.0)
        rates.flush()

        assert rates.rate("src") == pytest.approx(800.0)

    def test_an_unattached_table_persists_nowhere(self):
        """An embedded server built without a catalog measures for its own
        lifetime, and nothing on the read path has to know that."""
        rates = DecodeRates()
        _converge(rates, "src", 800.0)
        rates.flush()

        assert rates.rate("src") == pytest.approx(800.0)


class TestTheFlushCadence:
    """Write-through is debounced: a read is milliseconds and rewriting the
    table is comparable, so persisting per sample would cost more than the
    thing being measured."""

    class Recorder:
        def __init__(self):
            self.writes = []

        def load_decode_rates(self):
            return {}

        def save_decode_rates(self, rows):
            self.writes.append(dict(rows))

    def test_samples_do_not_each_reach_the_store(self):
        store = self.Recorder()
        rates = DecodeRates()
        rates.attach(store)
        _converge(rates, "src", 800.0, samples=50)

        assert store.writes == []

    def test_a_due_sample_writes_the_table_through(self):
        """So a live query sees a bounded-staleness table without waiting for a
        shutdown that may never be clean."""
        store = self.Recorder()
        rates = DecodeRates()
        rates.attach(store)
        rates._next_flush = 0.0  # as if _FLUSH_SECONDS had elapsed
        _converge(rates, "src", 800.0)

        assert store.writes == [{"src": (pytest.approx(800.0), 1)}]

    def test_the_write_rearms_the_debounce(self):
        store = self.Recorder()
        rates = DecodeRates()
        rates.attach(store)
        rates._next_flush = 0.0
        _converge(rates, "src", 800.0, samples=10)

        assert len(store.writes) == 1


class TestTheSampleSite:
    """Where the measurement is taken, through a real adapter and cache."""

    # Deliberately small: a chunk this size is mostly per-call cost, and has to
    # be measured anyway.
    SHAPE = (256, 256)
    CHUNKS = (128, 128)

    @pytest.fixture
    def adapter(self, tmp_path):
        path = tmp_path / "a.zarr"
        arr = zarr.open_array(
            str(path),
            mode="w",
            shape=self.SHAPE,
            chunks=self.CHUNKS,
            dtype="uint16",
        )
        arr[:] = 7
        return ZarrAdapter(zarr.open_array(str(path), mode="r"), "src", ["y", "x"])

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = CacheManager(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        yield mgr
        mgr.close()

    def _full_chunk_id(self):
        return encode_chunk_id("src", ChunkBounds(start=[0, 0], stop=list(self.CHUNKS)))

    def test_a_full_resolution_read_is_measured(self, adapter, manager):
        adapter.resolve_chunk_data(self._full_chunk_id(), manager)

        assert active_decode_rates().snapshot()["src"][1] == 1

    def test_a_cache_hit_is_not_measured_again(self, adapter, manager):
        """Only a miss reaches the decode. Counting a hit would fold the cache's
        own speed into the number that decides what the cache keeps."""
        chunk_id = self._full_chunk_id()
        adapter.resolve_chunk_data(chunk_id, manager)
        adapter.resolve_chunk_data(chunk_id, manager)

        assert active_decode_rates().snapshot()["src"][1] == 1

    def test_a_scaled_read_is_not_measured(self, adapter, manager):
        """A scaled build may be sourced from cached full-resolution chunks
        (biopb/biopb#965), so timing one measures the cache's state and then
        decides what the cache keeps by it."""
        scaled = encode_chunk_id_with_scale(
            "src", ChunkBounds(start=[0, 0], stop=list(self.CHUNKS)), (2, 2), "area"
        )
        adapter.resolve_chunk_data(scaled, manager)

        assert "src" not in active_decode_rates().snapshot()

    def test_a_measured_fast_array_lands_in_the_cheap_pool(self, adapter, manager):
        """End to end: the measurement has to reach the pool the chunk is
        written into, because eviction is whole-segment and cannot consult a
        class after the fact."""
        rates = DecodeRates(cheap_mbps=1e-6)  # anything measurable clears it
        _converge(rates, "src", 1.0)
        set_active_decode_rates(rates)

        adapter.resolve_chunk_data(self._full_chunk_id(), manager)

        assert {key[0] for key in manager.backend._pool_queues} == {"cheap"}

    def test_a_slow_array_still_lands_in_the_normal_pool(self, adapter, manager):
        rates = DecodeRates(cheap_mbps=1e12)  # nothing real clears it
        _converge(rates, "src", 1.0)
        set_active_decode_rates(rates)

        adapter.resolve_chunk_data(self._full_chunk_id(), manager)

        assert {key[0] for key in manager.backend._pool_queues} == {"normal"}


class TestTheManagerWiring:
    def test_the_manager_installs_the_configured_threshold(self, tmp_path):
        """The threshold is cache policy, so it comes from CacheConfig; the
        store is the catalog's, attached separately at startup."""
        mgr = CacheManager(
            CacheConfig(
                backend="file",
                file_cache_dir=tmp_path / "cache",
                cheap_decode_mbps=500.0,
            )
        )
        try:
            _converge(active_decode_rates(), "src", 800.0)

            assert active_decode_rates().retention_for_array("src") == "cheap"
        finally:
            mgr.close()

    def test_close_flushes_what_the_debounce_has_not(self, tmp_path):
        """A clean shutdown keeps the last minute of measurement."""
        db = MetadataDatabase(store_path=tmp_path / "catalog.duckdb")
        db.open()
        try:
            mgr = CacheManager(CacheConfig(backend="memory"))
            active_decode_rates().attach(db)
            _converge(active_decode_rates(), "src", 800.0)
            mgr.close()

            assert db.load_decode_rates()["src"][0] == pytest.approx(800.0)
        finally:
            db.close()

    def test_the_cache_directory_holds_no_measurements(self, tmp_path):
        cache_dir = tmp_path / "cache"
        mgr = CacheManager(CacheConfig(backend="file", file_cache_dir=cache_dir))
        _converge(active_decode_rates(), "src", 800.0)
        mgr.close()

        assert not list(cache_dir.rglob("*decode*"))


class TestTheExemptSourceKinds:
    """Two source kinds must never be classified by measured decode cost, and
    both are exempt by construction: they override ``resolve_chunk_data`` and so
    never reach the sample site or ``_retention_for_chunk``. Pinned here because
    the exemption is structural -- nothing in either file mentions retention, so
    deleting an override would silently enrol it."""

    def test_the_proxy_overrides_the_seam_that_would_classify_it(self):
        """A miss on a passthrough proxy is an upstream round trip plus load on
        someone else's server. Timing the local hand-off would clock a LAN
        upstream as fast and evict it first, which is backwards."""
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
        from biopb_tensor_server.core.adapter_base import TensorAdapter

        assert (
            RemoteTensorAdapter.resolve_chunk_data
            is not TensorAdapter.resolve_chunk_data
        )

    def test_an_upload_is_neither_measured_nor_reclassified(self, tmp_path):
        """The cache entry is an upload's only copy -- ``get_data`` raises, so
        there is nothing to measure and a "cheap" stamp would be data loss."""
        from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter

        set_active_decode_rates(DecodeRates(cheap_mbps=1e-6))  # everything clears it
        manager = CacheManager(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        try:
            CacheManager._instance = manager
            adapter = CachedSourceAdapter(
                source_id="up", shape=[64, 64], dtype="uint8", chunk_shape=[64, 64]
            )
            bounds = ChunkBounds(start=[0, 0], stop=[64, 64])
            adapter.write_chunk(bounds, np.ones((64, 64), dtype=np.uint8))
            adapter.resolve_chunk_data(encode_chunk_id("up", bounds), manager)

            assert active_decode_rates().snapshot() == {}
            assert {key[0] for key in manager.backend._pool_queues} == {"normal"}
        finally:
            CacheManager._instance = None
            manager.close()
