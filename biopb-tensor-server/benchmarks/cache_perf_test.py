"""Cache performance benchmarks.

Measures server cache behavior under different read patterns.
Backend controlled via BIOPB_CACHE_BACKEND env var.

Note: Memory backend only caches scaled reads (computed chunks).
      File backend caches both scaled and unscaled reads.
"""

import os

import pytest

from benchmarks.conftest import get_all_source_ids


def get_cache_backend() -> str:
    """Get current cache backend from environment."""
    return os.environ.get("BIOPB_CACHE_BACKEND", "file")


def get_cache_stats() -> dict:
    """Get server cache statistics."""
    from biopb_tensor_server.cache import CacheManager

    manager = CacheManager.get_instance()
    if manager is None:
        return {"hits": 0, "misses": 0, "entries": 0, "hit_rate": 0.0}

    stats = manager.stats()
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "entries": stats.total_entries,
        "hit_rate": stats.hits / (stats.hits + stats.misses) if (stats.hits + stats.misses) > 0 else 0.0,
    }


class TestScaledColdRead:
    """Cold cache scaled reads - each iteration starts fresh."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_scaled_single_chunk_cold(self, benchmark, data_source, bench_client_flight_no_cache, bench_server):
        """Read one scaled chunk with fresh cache each iteration."""
        source_id = data_source["id"]
        tensor_id = data_source.get("expected_tensors", [source_id])[0]
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        # Get tensor to check dimensionality
        arr_base = bench_client_flight_no_cache.get_tensor(source_id, tensor_id)
        ndim = len(arr_base.shape)
        scale_hint = tuple([1] * (ndim - 2) + [2, 2])

        arr = bench_client_flight_no_cache.get_tensor(source_id, tensor_id, scale_hint=scale_hint)

        from benchmarks.utils import clear_cache_entries

        def cold_read():
            # Clear entries for true cold read (keeps CacheManager initialized)
            clear_cache_entries()
            return arr[..., 0:chunks[0]//2, 0:chunks[1]//2].compute()

        result = benchmark(cold_read)
        assert result.shape[-2:] == (chunks[0] // 2, chunks[1] // 2)

        stats = get_cache_stats()
        backend = getattr(bench_server, "_bench_backend", "file")
        print(f"\nBackend: {backend}")
        print(f"Cold read: hits={stats['hits']}, misses={stats['misses']}")


class TestScaledWarmRead:
    """Warm cache scaled reads - cache persists between iterations."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_scaled_single_chunk_warm(self, benchmark, data_source, bench_client_flight_no_cache):
        """Read same scaled chunk repeatedly - should hit server cache."""
        source_id = data_source["id"]
        tensor_id = data_source.get("expected_tensors", [source_id])[0]
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        # Clear entries but keep CacheManager initialized
        from benchmarks.utils import clear_cache_entries
        clear_cache_entries()

        # Get tensor to check dimensionality
        arr_base = bench_client_flight_no_cache.get_tensor(source_id, tensor_id)
        ndim = len(arr_base.shape)
        scale_hint = tuple([1] * (ndim - 2) + [2, 2])

        arr = bench_client_flight_no_cache.get_tensor(
            source_id, tensor_id, scale_hint=scale_hint
        )

        # Prime cache with first read (not benchmarked)
        arr[..., 0:chunks[0]//2, 0:chunks[1]//2].compute()
        stats_before = get_cache_stats()

        def warm_read():
            return arr[..., 0:chunks[0]//2, 0:chunks[1]//2].compute()

        result = benchmark(warm_read)
        assert result.shape[-2:] == (chunks[0] // 2, chunks[1] // 2)

        stats_after = get_cache_stats()
        new_hits = stats_after["hits"] - stats_before["hits"]
        print(f"\nBackend: {get_cache_backend()}")
        print(f"New hits during benchmark: {new_hits}")
        print(f"Total: hits={stats_after['hits']}, misses={stats_after['misses']}, hit_rate={stats_after['hit_rate']:.2%}")


class TestScaledSequentialScan:
    """Sequential scan of scaled chunks."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_scaled_scan(self, benchmark, data_source, bench_client_flight_no_cache):
        """Scan multiple scaled chunks, report cache stats."""
        source_id = data_source["id"]
        tensor_id = data_source.get("expected_tensors", [source_id])[0]
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        from benchmarks.utils import clear_cache_entries
        clear_cache_entries()

        # Get tensor to check dimensionality
        arr_base = bench_client_flight_no_cache.get_tensor(source_id, tensor_id)
        ndim = len(arr_base.shape)
        scale_hint = tuple([1] * (ndim - 2) + [2, 2])

        arr = bench_client_flight_no_cache.get_tensor(
            source_id, tensor_id, scale_hint=scale_hint
        )

        scaled_chunks = (chunks[0] // 2, chunks[1] // 2)

        def scan():
            results = []
            for i in range(4):
                y0, y1 = i * scaled_chunks[0], (i + 1) * scaled_chunks[0]
                results.append(arr[..., y0:y1, 0:scaled_chunks[1]].compute())
            return results

        results = benchmark(scan)
        stats = get_cache_stats()
        print(f"\nBackend: {get_cache_backend()}")
        print(f"Scan: {len(results)} chunks, hits={stats['hits']}, misses={stats['misses']}")


class TestUnscaledRead:
    """Unscaled reads - only meaningful for file backend."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_unscaled_warm_read(self, benchmark, data_source, bench_client_flight_no_cache):
        """Unscaled warm read - file backend only."""
        if get_cache_backend() == "memory":
            pytest.skip("Memory backend doesn't cache unscaled reads")

        source_id = data_source["id"]
        tensor_id = data_source.get("expected_tensors", [source_id])[0]
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        from benchmarks.utils import clear_cache_entries
        clear_cache_entries()

        arr = bench_client_flight_no_cache.get_tensor(source_id, tensor_id)

        # Prime cache
        arr[..., 0:chunks[0], 0:chunks[1]].compute()
        stats_before = get_cache_stats()

        def warm_read():
            return arr[..., 0:chunks[0], 0:chunks[1]].compute()

        result = benchmark(warm_read)
        stats_after = get_cache_stats()
        print(f"\nBackend: {get_cache_backend()} (file caches unscaled)")
        print(f"New hits: {stats_after['hits'] - stats_before['hits']}")