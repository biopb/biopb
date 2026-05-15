"""Flight throughput benchmark: theoretical ceiling vs real server."""

import tempfile
import time
import threading
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
from pathlib import Path
import json

import pytest


# Schema matching TensorFlightServer chunk format
CHUNK_SCHEMA = pa.schema([
    ('data', pa.list_(pa.uint16())),
    ('shape', pa.list_(pa.int64())),
    ('dtype', pa.string()),
])


def make_chunk_batch(edge: int) -> pa.RecordBatch:
    """Create a chunk batch matching TensorFlightServer format."""
    data = np.zeros((edge, edge), dtype=np.uint16)
    return pa.RecordBatch.from_arrays(
        [pa.array([data.ravel()]), pa.array([[edge, edge]]), pa.array(["uint16"])],
        ["data", "shape", "dtype"]
    )


class MockChunkServer(flight.FlightServerBase):
    """Mock server returning pre-built chunks - tests Flight protocol ceiling."""

    def __init__(self, location, batch):
        super().__init__(location)
        self._batch = batch

    def do_get(self, context, ticket):
        return flight.RecordBatchStream(pa.RecordBatchReader.from_batches(
            self._batch.schema, [self._batch]
        ))


def bench_flight_ceiling(port: int, batch: pa.RecordBatch, concurrency: int, iterations: int):
    """Benchmark Flight ceiling with mock server."""
    location = f"grpc://127.0.0.1:{port}"

    server = MockChunkServer(location, batch)
    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    def worker():
        client = flight.FlightClient(location)
        opts = flight.FlightCallOptions()
        ticket = flight.Ticket(b"x")

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            reader = client.do_get(ticket, options=opts)
            reader.read_all()
            times.append(time.perf_counter() - start)
        client.close()
        return times

    results = []
    threads = []
    start_wall = time.perf_counter()
    for _ in range(concurrency):
        t = threading.Thread(target=lambda: results.append(worker()))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    wall_time = time.perf_counter() - start_wall
    all_times = [t for r in results for t in r]
    total_bytes = concurrency * iterations * batch.nbytes

    server.shutdown()

    return {
        "throughput_mb_s": total_bytes / wall_time / 1024 / 1024,
        "avg_latency_ms": np.mean(all_times) * 1000,
        "total_time_s": wall_time,
    }


def bench_real_server(port: int, batch: pa.RecordBatch, concurrency: int, iterations: int, use_file_cache: bool = True):
    """Benchmark real TensorFlightServer."""
    from biopb_tensor_server.server import TensorFlightServer
    from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig

    location = f"grpc://127.0.0.1:{port}"
    edge = int(np.sqrt(batch.nbytes / 2))  # uint16 = 2 bytes
    shape = (edge * 2, edge * 2)
    chunks = (edge, edge)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate zarr
        import zarr
        zarr_path = Path(tmpdir) / "test.ome.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        arr = root.zeros("0", shape=shape, chunks=chunks, dtype="uint16")
        arr[:] = np.arange(shape[0]*shape[1], dtype=np.uint16).reshape(shape)
        ome = {"multiscales": [{"version": "0.4", "axes": [{"name":"y","type":"space"},{"name":"x","type":"space"}], "datasets": [{"path": "0"}]}]}
        (zarr_path / ".zattrs").write_text(json.dumps(ome))

        # Setup cache
        if use_file_cache:
            CacheManager.initialize(CacheConfig(
                backend="file",
                file_cache_dir=Path(tmpdir) / "fcache",
                file_max_segment_bytes=256*1024*1024,
                file_max_total_bytes=512*1024*1024,
            ))
        else:
            CacheManager.initialize(CacheConfig(
                backend="memory",
                memory_max_entries=0,  # No caching for raw reads
                memory_max_bytes=0,
            ))

        grp = zarr.open_group(zarr_path, mode="r")
        level_arr = grp["0"]
        adapter = OmeZarrAdapter(level_arr, "test")

        server = TensorFlightServer(location)
        server.register_source("test", adapter)
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(0.3)

        # Get ticket via GetFlightInfo
        from biopb.tensor.descriptor_pb2 import TensorSelection
        client_meta = flight.FlightClient(location)
        opts = flight.FlightCallOptions()
        sel = TensorSelection(source_id="test", tensor_id="test")
        desc = flight.FlightDescriptor.for_command(sel.SerializeToString())
        info = client_meta.get_flight_info(desc, options=opts)
        ticket_bytes = info.endpoints[0].ticket.ticket
        client_meta.close()

        # Warm up (populate cache if using file cache)
        warmup_client = flight.FlightClient(location)
        warmup_client.do_get(flight.Ticket(ticket_bytes), options=opts).read_all()
        warmup_client.close()

        def worker():
            client = flight.FlightClient(location)
            opts = flight.FlightCallOptions()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                reader = client.do_get(flight.Ticket(ticket_bytes), options=opts)
                reader.read_all()
                times.append(time.perf_counter() - start)
            client.close()
            return times

        results = []
        threads = []
        start_wall = time.perf_counter()
        for _ in range(concurrency):
            t = threading.Thread(target=lambda: results.append(worker()))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        wall_time = time.perf_counter() - start_wall
        all_times = [t for r in results for t in r]
        total_bytes = concurrency * iterations * batch.nbytes

        server.shutdown()
        CacheManager.reset()

        return {
            "throughput_mb_s": total_bytes / wall_time / 1024 / 1024,
            "avg_latency_ms": np.mean(all_times) * 1000,
            "total_time_s": wall_time,
        }


@pytest.fixture(scope="module")
def batch_4mb():
    """4MB chunk batch."""
    edge = int(np.sqrt(4 * 1024 * 1024 / 2))
    return make_chunk_batch(edge)


@pytest.fixture(scope="module")
def batch_8mb():
    """8MB chunk batch."""
    edge = int(np.sqrt(8 * 1024 * 1024 / 2))
    return make_chunk_batch(edge)


class TestFlightCeiling:
    """Test theoretical Flight throughput with mock server."""

    @pytest.mark.parametrize("concurrency", [1, 4, 8])
    def test_ceiling_4mb(self, batch_4mb, concurrency):
        """Flight ceiling with 4MB chunks."""
        result = bench_flight_ceiling(8915, batch_4mb, concurrency, iterations=20)
        print(f"\n  Mock server (4MB): concurrency={concurrency}, throughput={result['throughput_mb_s']:.0f} MB/s, latency={result['avg_latency_ms']:.2f}ms")

        # Ceiling should be > 3000 MB/s at 4+ threads
        if concurrency >= 4:
            assert result['throughput_mb_s'] > 3000, f"Expected >3000 MB/s ceiling, got {result['throughput_mb_s']:.0f}"

    @pytest.mark.parametrize("concurrency", [1, 4, 8])
    def test_ceiling_8mb(self, batch_8mb, concurrency):
        """Flight ceiling with 8MB chunks."""
        result = bench_flight_ceiling(8916, batch_8mb, concurrency, iterations=20)
        print(f"\n  Mock server (8MB): concurrency={concurrency}, throughput={result['throughput_mb_s']:.0f} MB/s, latency={result['avg_latency_ms']:.2f}ms")

        if concurrency >= 4:
            assert result['throughput_mb_s'] > 3000, f"Expected >3000 MB/s ceiling, got {result['throughput_mb_s']:.0f}"


class TestRealServerThroughput:
    """Test real TensorFlightServer throughput."""

    @pytest.mark.parametrize("concurrency", [1, 4, 8])
    def test_cached_reads_4mb(self, batch_4mb, concurrency):
        """Real server with file cache (cached reads)."""
        result = bench_real_server(8917, batch_4mb, concurrency, iterations=20, use_file_cache=True)
        print(f"\n  Real server cached (4MB): concurrency={concurrency}, throughput={result['throughput_mb_s']:.0f} MB/s, latency={result['avg_latency_ms']:.2f}ms")

        # Cached reads should approach ceiling (>2500 MB/s at 4+ threads)
        if concurrency >= 4:
            assert result['throughput_mb_s'] > 2500, f"Expected >2500 MB/s for cached reads, got {result['throughput_mb_s']:.0f}"

    @pytest.mark.parametrize("concurrency", [1, 4, 8])
    def test_raw_reads_4mb(self, batch_4mb, concurrency):
        """Real server without cache (raw zarr reads)."""
        result = bench_real_server(8918, batch_4mb, concurrency, iterations=20, use_file_cache=False)
        print(f"\n  Real server raw (4MB): concurrency={concurrency}, throughput={result['throughput_mb_s']:.0f} MB/s, latency={result['avg_latency_ms']:.2f}ms")

        # Raw reads are slower due to zarr + Arrow construction
        # Should still scale with concurrency
        assert result['throughput_mb_s'] > 500, f"Expected >500 MB/s for raw reads, got {result['throughput_mb_s']:.0f}"