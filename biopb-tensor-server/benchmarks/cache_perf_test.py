"""Cache performance benchmarks.

Validates LRU cache performance for read patterns.
"""

import pytest

from benchmarks.utils import get_cache_stats
from benchmarks.conftest import get_all_source_ids, BaselineClient


class TestLRUCachePerformance:
    """Benchmarks for client-side cache hit rate."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_bench_cache_hit_rate(self, benchmark, data_source, bench_client):
        """Read same region twice - warm should be faster."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        # First read (cold)
        arr[0:chunks[0], 0:chunks[1]].compute()

        def warm_read():
            return arr[0:chunks[0], 0:chunks[1]].compute()

        result = benchmark(warm_read)
        assert result.shape == (chunks[0], chunks[1])

        cache_info = bench_client.cache_info()
        print(f"\nClient cache: {cache_info['size_bytes'] / 1e3:.1f}KB")


@pytest.mark.skip(reason="Sieve-K algorithm not yet implemented")
class TestSieveKCachePerformance:
    """Placeholders for Sieve-K cache benchmarks."""


class TestCacheBackendComparison:
    """Compare different cache backends."""

    @pytest.mark.parametrize("bench_server", ["memory", "file"], indirect=True)
    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_memory_vs_file_backend(self, benchmark, bench_server, data_source, bench_client_flight):
        """Compare memory vs file cache backend performance."""
        backend = getattr(bench_server, "_bench_backend", "memory")
        source_id = data_source["id"]
        source_type = data_source.get("type")

        arr = bench_client_flight.get_tensor(source_id, source_id)
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        # Prime cache
        arr[0:chunks[0], 0:chunks[1]].compute()

        def cached_read():
            return arr[0:chunks[0], 0:chunks[1]].compute()

        result = benchmark(cached_read)
        assert result.shape == (chunks[0], chunks[1])

        stats = get_cache_stats(bench_server)
        print(f"\nBackend: {backend}")
        print(f"Cache hit rate: {stats['hit_rate']:.2%}")