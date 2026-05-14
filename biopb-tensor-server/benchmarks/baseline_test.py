"""Baseline latency benchmarks.
Tests run across all configured synthetic sources.
"""

import pytest

from benchmarks.conftest import get_all_source_ids, BaselineClient


class TestReadLatency:
    """Read latency.
    """

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_first_read_latency(self, benchmark, data_source, bench_client):
        """Cold read: baseline should show lower latency."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        # Use ellipsis to select last two spatial dims (works for 2D and 5D)
        def read():
            return arr[..., 0:256, 0:256].compute()

        result = benchmark(read)
        assert result.shape[-2:] == (256, 256)

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_warm_read_latency(self, benchmark, data_source, bench_client):
        """Warm read after priming caches."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        # Prime cache and read - use ellipsis for last two spatial dims
        arr[..., 0:256, 0:256].compute()
        def read():
            return arr[..., 0:256, 0:256].compute()

        result = benchmark(read)
        assert result.shape[-2:] == (256, 256)

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_sequential_scan(self, benchmark, data_source, bench_client):
        """Read all chunks in sequence."""
        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        # Get chunk size from config or default
        params = data_source.get("params", {})
        chunks = params.get("chunks", params.get("tile", [256, 256]))

        # Use ellipsis to read last two spatial dims
        def scan():
            results = []
            results.append(arr[..., 0:chunks[0], 0:chunks[1]].compute())
            results.append(arr[..., 0:chunks[0], chunks[1]:chunks[1]*2].compute())
            results.append(arr[..., chunks[0]:chunks[0]*2, 0:chunks[1]].compute())
            results.append(arr[..., chunks[0]:chunks[0]*2, chunks[1]:chunks[1]*2].compute())
            return results

        results = benchmark(scan)
        assert len(results) == 4

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_random_access(self, benchmark, data_source, bench_client):
        """Read random chunks to measure cache behavior."""
        import random

        source_id = data_source["id"]
        source_type = data_source.get("type")
        tensor_id = data_source.get("expected_tensors", [source_id])[0]

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        arr = bench_client.get_tensor(source_id, tensor_id)

        params = data_source.get("params", {})
        chunks = params.get("chunks", [512, 512])

        random.seed(42)
        # Use last two dimensions for spatial chunking
        n_chunks_x = arr.shape[-1] // chunks[1]
        n_chunks_y = arr.shape[-2] // chunks[0]

        chunk_indices = [
            (i * chunks[0], j * chunks[1])
            for i, j in random.sample(
                [(x, y) for x in range(n_chunks_y) for y in range(n_chunks_x)],
                min(8, n_chunks_y * n_chunks_x)
            )
        ]

        # Use ellipsis to read last two spatial dims
        def read_random():
            return [arr[..., i:i+chunks[0], j:j+chunks[1]].compute() for i, j in chunk_indices]

        results = benchmark(read_random)
        assert len(results) == len(chunk_indices)


class TestMetadataLatency:
    """Compare metadata discovery latency: baseline vs flight."""

#     @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
#     def test_list_sources_latency(self, benchmark, data_source, bench_client)_flight:
#         """Measure list_sources() latency."""
#         def list_sources():
#             return bench_client_flight.list_sources()
# 
#         result = benchmark(list_sources)
#         assert isinstance(result, dict)

#     @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
#     def test_metadata_discovery_latency(self, benchmark, data_source, bench_client_flight):
#         """Measure metadata discovery through Flight stack."""
#         source_id = data_source["id"]
# 
#         def discover_metadata():
#             bench_client_flight._sources.clear()
#             sources = bench_client_flight.list_sources()
#             source_desc = sources.get(source_id)
#             if source_desc and source_desc.tensors:
#                 tensor_desc = source_desc.tensors[0]
#                 return {
#                     "source_id": source_id,
#                     "shape": tensor_desc.shape,
#                     "dtype": tensor_desc.dtype,
#                 }
#             return None
# 
#         result = benchmark(discover_metadata)
#         assert result is not None
#         assert result["source_id"] == source_id

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    @pytest.mark.parametrize("bench_client", ["baseline", "flight"], indirect=True)
    def test_get_source_metadata_latency(self, benchmark, data_source, bench_client):
        """Measure get_source_metadata() latency."""
        source_id = data_source["id"]
        source_type = data_source.get("type")

        # HDF5 can't have multiple file handles - skip baseline for HDF5
        if source_type == "hdf5" and isinstance(bench_client, BaselineClient):
            pytest.skip("HDF5 baseline tests skipped - file already open by server adapter")

        def fetch_metadata():
            return bench_client.get_source_metadata(source_id)

        result = benchmark(fetch_metadata)
        assert isinstance(result, dict)
