"""Metadata database benchmarks.

Measures performance for:
- Concurrent query access (multiple threads)
- Large-scale source catalogs (10k, 50k, 100k sources)
- Query latency with various filter complexity
"""

import concurrent.futures
import time
import threading

import pytest
import pyarrow as pa

from biopb_tensor_server.metadata_db import MetadataDatabase


class MockAdapter:
    """Mock adapter for benchmark testing."""

    def __init__(self, source_id, source_url, source_type, shape, dtype):
        self.source_id = source_id
        self._source_url = source_url
        self._source_type = source_type
        self._shape = shape
        self._dtype = dtype

    def get_source_descriptor(self):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=self._source_url,
            source_type=self._source_type,
            tensors=[
                TensorDescriptor(
                    array_id=self.source_id,
                    shape=self._shape,
                    dtype=self._dtype,
                )
            ],
        )


    def get_metadata(self):
        return {"plate_id": self.source_id.split("-")[0], "acquisition_date": "2024-01-01"}


def populate_database(db: MetadataDatabase, n_sources: int) -> None:
    """Populate database with N synthetic sources."""
    for i in range(n_sources):
        plate_num = i // 10
        source_id = f"plate-{plate_num:04d}-well-{i % 10:02d}"
        source_url = f"/data/experiment-{plate_num:04d}/{source_id}.zarr"
        adapter = MockAdapter(source_id, source_url, "ome-zarr", [512, 512, 64], "uint16")
        db.sync_source_added(source_id, adapter)


class TestConcurrentAccess:
    """Concurrent read/write access benchmarks."""

    def test_bench_concurrent_queries_same_db(self, benchmark, tmp_path):
        """Multiple threads querying the same database simultaneously."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 1000)

        def concurrent_queries():
            def run_query(thread_id):
                # Different queries per thread to avoid cache hits
                if thread_id % 3 == 0:
                    sql = "SELECT source_id FROM sources WHERE source_type='ome-zarr' LIMIT 10"
                elif thread_id % 3 == 1:
                    sql = "SELECT source_id, dtype FROM sources WHERE dtype='uint16' LIMIT 10"
                else:
                    sql = "SELECT COUNT(*) FROM sources"

                info = db.handle_query(sql)
                result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
                return result.num_rows

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(run_query, i) for i in range(32)]
                results = [f.result() for f in futures]

            return results

        results = benchmark(concurrent_queries)
        assert len(results) == 32
        db.close()

    def test_bench_concurrent_sync_and_query(self, benchmark, tmp_path):
        """Concurrent source additions while queries are running."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 500)  # Initial population

        def concurrent_mixed():
            errors = []

            def sync_worker(thread_id):
                try:
                    source_id = f"dynamic-{thread_id}"
                    adapter = MockAdapter(source_id, f"/data/{source_id}.zarr", "ome-zarr", [256, 256], "float32")
                    db.sync_source_added(source_id, adapter)
                    db.sync_source_removed(source_id)
                except Exception as e:
                    errors.append(str(e))

            def query_worker(thread_id):
                try:
                    sql = "SELECT COUNT(*) FROM sources"
                    info = db.handle_query(sql)
                    db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
                except Exception as e:
                    errors.append(str(e))

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = []
                # 8 sync threads, 8 query threads
                for i in range(8):
                    futures.append(executor.submit(sync_worker, i))
                for i in range(8):
                    futures.append(executor.submit(query_worker, i))

                concurrent.futures.wait(futures)

            return errors

        errors = benchmark(concurrent_mixed)
        assert len(errors) == 0
        db.close()


class TestLargeScale:
    """Performance with large source catalogs."""

    def test_bench_populate_10k_sources(self, benchmark, tmp_path):
        """Time to populate database with 10,000 sources."""
        db = MetadataDatabase(enabled=True)

        def populate_10k():
            populate_database(db, 10000)

        benchmark(populate_10k)

        # Verify count
        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 10000
        db.close()

    def test_bench_populate_50k_sources(self, benchmark, tmp_path):
        """Time to populate database with 50,000 sources."""
        db = MetadataDatabase(enabled=True)

        def populate_50k():
            populate_database(db, 50000)

        benchmark(populate_50k)

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 50000
        db.close()

    @pytest.mark.slow
    def test_bench_populate_100k_sources(self, benchmark, tmp_path):
        """Time to populate database with 100,000 sources."""
        db = MetadataDatabase(enabled=True)

        def populate_100k():
            populate_database(db, 100000)

        benchmark(populate_100k)

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 100000
        db.close()

    def test_bench_query_latency_10k(self, benchmark, tmp_path):
        """Query latency with 10k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 10000)

        def query_all():
            sql = "SELECT source_id FROM sources"
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == 10000
        db.close()

    def test_bench_query_latency_50k(self, benchmark, tmp_path):
        """Query latency with 50k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 50000)

        def query_all():
            sql = "SELECT source_id FROM sources"
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == 50000
        db.close()

    @pytest.mark.slow
    def test_bench_query_latency_100k(self, benchmark, tmp_path):
        """Query latency with 100k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 100000)

        def query_all():
            sql = "SELECT source_id FROM sources"
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == 100000
        db.close()


class TestQueryComplexity:
    """Performance with different query complexity levels."""

    def test_bench_simple_count_50k(self, benchmark, tmp_path):
        """Simple COUNT query with 50k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 50000)

        def count_query():
            sql = "SELECT COUNT(*) FROM sources"
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.column(0).to_pylist()[0]

        count = benchmark(count_query)
        assert count == 50000
        db.close()

    def test_bench_filtered_query_50k(self, benchmark, tmp_path):
        """Filtered query (WHERE clause) with 50k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 50000)

        def filtered_query():
            # This filter matches ~1000 sources (plate-0000 and plate-0001)
            sql = "SELECT source_id FROM sources WHERE source_url LIKE '%experiment-0000%' OR source_url LIKE '%experiment-0001%'"
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.num_rows

        n_rows = benchmark(filtered_query)
        assert n_rows == 20  # 2 plates * 10 wells each
        db.close()

    def test_bench_json_field_query_50k(self, benchmark, tmp_path):
        """Query using JSON field extraction with 50k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 50000)

        def json_query():
            sql = "SELECT source_id, metadata_json->>'plate_id' as plate FROM sources LIMIT 1000"
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.num_rows

        n_rows = benchmark(json_query)
        assert n_rows == 1000
        db.close()

    def test_bench_complex_filter_50k(self, benchmark, tmp_path):
        """Complex multi-condition filter with 50k sources."""
        db = MetadataDatabase(enabled=True)
        populate_database(db, 50000)

        def complex_query():
            sql = """
                SELECT source_id, source_url
                FROM sources
                WHERE source_type='ome-zarr'
                  AND dtype='uint16'
                  AND shape_summary LIKE '%512%'
                LIMIT 500
            """
            info = db.handle_query(sql)
            result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
            return result.num_rows

        n_rows = benchmark(complex_query)
        assert n_rows == 500
        db.close()


class TestInitialSync:
    """Benchmark initial batch sync operations."""

    def test_bench_initial_sync_10k(self, benchmark, tmp_path):
        """Batch insert via initial_sync with 10k sources."""
        db = MetadataDatabase(enabled=True)

        sources = {}
        for i in range(10000):
            source_id = f"batch-{i}"
            sources[source_id] = MockAdapter(
                source_id, f"/data/{source_id}.zarr", "ome-zarr", [256, 256], "uint8"
            )

        def batch_sync():
            db.initial_sync(sources)

        benchmark(batch_sync)

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 10000
        db.close()

    def test_bench_initial_sync_50k(self, benchmark, tmp_path):
        """Batch insert via initial_sync with 50k sources."""
        db = MetadataDatabase(enabled=True)

        sources = {}
        for i in range(50000):
            source_id = f"batch-{i}"
            sources[source_id] = MockAdapter(
                source_id, f"/data/{source_id}.zarr", "ome-zarr", [256, 256], "uint8"
            )

        def batch_sync():
            db.initial_sync(sources)

        benchmark(batch_sync)

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 50000
        db.close()