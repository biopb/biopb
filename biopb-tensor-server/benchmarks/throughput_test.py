"""Throughput benchmarks.

Measures concurrent read performance and chunk coalescing efficiency.
Uses config-based parametrization over all synthetic sources.
"""

import concurrent.futures
import time

import pytest
import numpy as np

from benchmarks.conftest import get_all_source_ids, BaselineClient


class TestSingleClientThroughput:
    """Single client read throughput benchmarks."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_bench_single_client_sequential(self, benchmark, data_source, bench_client):
        """Sequential read throughput (single client)."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        def sequential_read():
            return arr.compute()

        benchmark(sequential_read)

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_bench_single_client_chunked(self, benchmark, data_source, bench_client):
        """Chunk-by-chunk read throughput."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        n_chunks_y = arr.shape[0] // chunks[0]
        n_chunks_x = arr.shape[1] // chunks[1]
        chunk_coords = [(i * chunks[0], j * chunks[1]) for i in range(n_chunks_y) for j in range(n_chunks_x)]

        def chunked_read():
            results = []
            for i, j in chunk_coords:
                results.append(arr[i:i+chunks[0], j:j+chunks[1]].compute())
            return results

        results = benchmark(chunked_read)

        print(f"\nChunks read: {len(results)}")
        assert len(results) == len(chunk_coords)


class TestMultiClientThroughput:
    """Concurrent client throughput benchmarks (Flight only - baseline has no server)."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_two_concurrent_clients(self, benchmark, data_source, bench_server):
        """Throughput with 2 concurrent clients."""
        source_id = data_source["id"]
        port = getattr(bench_server, "_bench_port", 8815)
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        from biopb.tensor import TensorFlightClient

        def concurrent_read():
            def read_chunk(chunk_idx):
                client = TensorFlightClient(f"grpc://localhost:{port}")
                arr = client.get_tensor(source_id, source_id)
                i, j = chunk_idx
                result = arr[i:i+chunks[0], j:j+chunks[1]].compute()
                client.close()
                return result

            chunks_to_read = [(0, 0), (chunks[0], chunks[1])]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(read_chunk, c) for c in chunks_to_read]
                results = [f.result() for f in futures]

            return results

        results = benchmark(concurrent_read)
        assert len(results) == 2

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_four_concurrent_clients(self, benchmark, data_source, bench_server):
        """Throughput with 4 concurrent clients."""
        source_id = data_source["id"]
        port = getattr(bench_server, "_bench_port", 8815)
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        from biopb.tensor import TensorFlightClient

        def concurrent_read():
            def read_chunk(chunk_idx):
                client = TensorFlightClient(f"grpc://localhost:{port}")
                arr = client.get_tensor(source_id, source_id)
                i, j = chunk_idx
                result = arr[i:i+chunks[0], j:j+chunks[1]].compute()
                client.close()
                return result

            chunks_to_read = [(0, 0), (0, chunks[1]), (chunks[0], 0), (chunks[0], chunks[1])]

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(read_chunk, c) for c in chunks_to_read]
                results = [f.result() for f in futures]

            return results

        results = benchmark(concurrent_read)
        assert len(results) == 4

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_eight_concurrent_clients(self, benchmark, data_source, bench_server):
        """Throughput with 8 concurrent clients."""
        source_id = data_source["id"]
        port = getattr(bench_server, "_bench_port", 8815)
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [128, 128]))

        from biopb.tensor import TensorFlightClient

        def concurrent_read():
            def read_chunk(chunk_idx):
                client = TensorFlightClient(f"grpc://localhost:{port}")
                arr = client.get_tensor(source_id, source_id)
                i, j = chunk_idx
                result = arr[i:i+chunks[0], j:j+chunks[1]].compute()
                client.close()
                return result

            chunks_to_read = [(i * chunks[0], j * chunks[1]) for i in range(4) for j in range(2)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(read_chunk, c) for c in chunks_to_read]
                results = [f.result() for f in futures]

            return results

        results = benchmark(concurrent_read)
        assert len(results) == 8


class TestChunkCoalescing:
    """Measure efficiency of merged chunk reads (Flight only)."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_adjacent_chunks_vs_separate(self, data_source, bench_client_flight):
        """Compare reading adjacent chunks together vs separately."""
        source_id = data_source["id"]
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        arr = bench_client_flight.get_tensor(source_id, source_id)

        def merged_read():
            return arr[0:chunks[0]*2, 0:chunks[1]].compute()

        merged_time = time.perf_counter()
        merged_read()
        merged_time = time.perf_counter() - merged_time

        def separate_read():
            results = []
            for i in range(2):
                results.append(arr[i*chunks[0]:(i+1)*chunks[0], 0:chunks[1]].compute())
            return np.concatenate(results, axis=0)

        separate_time = time.perf_counter()
        separate_read()
        separate_time = time.perf_counter() - separate_time

        print(f"\nMerged: {merged_time*1000:.2f}ms")
        print(f"Separate: {separate_time*1000:.2f}ms")
        print(f"Ratio: {separate_time / merged_time:.2f}x")


class TestThroughputMeasurement:
    """Detailed throughput measurement benchmarks."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_bench_read_bandwidth(self, data_source, bench_client):
        """Measure raw read bandwidth in MB/s."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        total_bytes = arr.shape[0] * arr.shape[1] * arr.dtype.itemsize

        times = []
        for _ in range(5):
            start = time.perf_counter()
            arr.compute()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = total_bytes / avg_time / 1e6

        print(f"\nAverage throughput: {throughput:.1f} MB/s")
        print(f"Array size: {total_bytes / 1e6:.1f} MB")
        print(f"Average time: {avg_time*1000:.2f}ms")

        assert throughput > 0

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_bench_client_cache_efficiency(self, data_source, bench_client):
        """Measure client-side cache hit rate."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        start = time.perf_counter()
        arr.compute()
        cold_time = time.perf_counter() - start

        start = time.perf_counter()
        arr.compute()
        warm_time = time.perf_counter() - start

        cache_info = bench_client.cache_info()

        print(f"\nCold read: {cold_time*1000:.2f}ms")
        print(f"Warm read: {warm_time*1000:.2f}ms")
        print(f"Cache speedup: {cold_time / warm_time:.2f}x")
        print(f"Client cache: {cache_info['size_bytes'] / 1e6:.1f}MB / {cache_info['max_bytes'] / 1e6:.1f}MB")