"""Measured retention: what a full-resolution chunk actually cost to decode.

The declared rule (``cache_retention_test.py``) can only ever call a
full-resolution chunk "normal" -- it has no ladder to judge one against. These
tests cover the half that separates them by measurement: where the sample is
taken, which reads are refused as samples, and the two source kinds that must
never be classified this way at all.
"""

import json

import numpy as np
import pytest
import zarr
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.manager import DECODE_RATES_FILE
from biopb_tensor_server.core.chunk import encode_chunk_id, encode_chunk_id_with_scale
from biopb_tensor_server.core.config import CacheConfig
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

    def test_one_sample_is_already_a_rate(self):
        """No warmup gate: the first read is a real, if noisy, measurement."""
        rates = DecodeRates()
        rates.record("src", CHUNK, CHUNK / 1e6 / 800.0)

        assert rates.rate("src") == pytest.approx(800.0)

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

        assert rates.snapshot()["small"]["mbps"] < rates.snapshot()["large"]["mbps"]

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
    def test_a_run_does_not_reserve_its_warmup(self, tmp_path):
        """Without this every restart re-measures, and every restart stamps a
        stratum of "normal" segments before it converges."""
        path = tmp_path / DECODE_RATES_FILE
        written = DecodeRates()
        _converge(written, "src", 800.0)
        written.save(path)

        restored = DecodeRates(cheap_mbps=500.0)
        restored.load(path)

        assert restored.rate("src") == pytest.approx(800.0)
        assert restored.retention_for_array("src") == "cheap"

    def test_an_unreadable_table_is_a_first_run(self, tmp_path):
        path = tmp_path / DECODE_RATES_FILE
        path.write_text("{ truncated")
        rates = DecodeRates()
        rates.load(path)

        assert rates.snapshot() == {}

    def test_a_hand_edited_row_is_dropped_not_believed(self, tmp_path):
        path = tmp_path / DECODE_RATES_FILE
        path.write_text(json.dumps({"ok": {"mbps": 800.0, "samples": 8}, "bad": {}}))
        rates = DecodeRates()
        rates.load(path)

        assert set(rates.snapshot()) == {"ok"}

    def test_an_empty_table_writes_no_file(self, tmp_path):
        path = tmp_path / DECODE_RATES_FILE
        DecodeRates().save(path)

        assert not path.exists()


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

        assert active_decode_rates().snapshot()["src"]["samples"] == 1

    def test_a_cache_hit_is_not_measured_again(self, adapter, manager):
        """Only a miss reaches the decode. Counting a hit would fold the cache's
        own speed into the number that decides what the cache keeps."""
        chunk_id = self._full_chunk_id()
        adapter.resolve_chunk_data(chunk_id, manager)
        adapter.resolve_chunk_data(chunk_id, manager)

        assert active_decode_rates().snapshot()["src"]["samples"] == 1

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
    def test_the_table_is_loaded_and_saved_beside_the_segments(self, tmp_path):
        """Not in the metadata DB: decode throughput is a read-path fact about
        an adapter, not a catalog fact, and clearing the cache clears it."""
        cache_dir = tmp_path / "cache"
        config = CacheConfig(
            backend="file", file_cache_dir=cache_dir, cheap_decode_mbps=500.0
        )

        first = CacheManager(config)
        _converge(active_decode_rates(), "src", 800.0)
        first.close()

        assert (cache_dir / DECODE_RATES_FILE).exists()

        second = CacheManager(config)
        try:
            assert active_decode_rates().rate("src") == pytest.approx(800.0)
            assert active_decode_rates().retention_for_array("src") == "cheap"
        finally:
            second.close()

    def test_the_memory_backend_keeps_its_table_in_process(self, tmp_path):
        """Nothing to persist beside: there are no segments to outlive."""
        mgr = CacheManager(CacheConfig(backend="memory"))
        _converge(active_decode_rates(), "src", 800.0)
        mgr.close()

        assert not list(tmp_path.iterdir())


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
