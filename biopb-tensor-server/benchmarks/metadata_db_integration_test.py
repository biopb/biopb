"""Metadata database integration benchmarks.

Measures performance for end-to-end query path through TensorFlightClient:
- Concurrent query access (multiple threads via Flight client)
- Large-scale source catalogs (10k, 50k, 100k sources)
- Query latency with various filter complexity

Unlike metadata_db_test.py which tests MetadataDatabase directly,
these tests measure full request path:
  TensorFlightClient.query_sources(sql)
    -> FlightClient.get_flight_info(TensorSelection with metadata_query)
      -> TensorFlightServer.get_flight_info()
        -> MetadataDatabase.handle_query(sql)
    -> FlightClient.do_get(ticket)
      -> Returns Arrow Table with query results
"""

import concurrent.futures
import random
import threading
import time

import pytest
from biopb.tensor import TensorFlightClient
from biopb_tensor_server.metadata_db import MetadataDatabase
from biopb_tensor_server.server import TensorFlightServer


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
        return {
            "plate_id": self.source_id.split("-")[0],
            "acquisition_date": "2024-01-01",
        }


def populate_server_sources(
    server: TensorFlightServer, metadata_db: MetadataDatabase, n_sources: int
) -> None:
    """Populate server and metadata_db with N synthetic sources.

    Sources are registered via server.register_source() and synced to
    metadata_db via metadata_db.sync_source_added().
    """
    for i in range(n_sources):
        plate_num = i // 10
        source_id = f"plate-{plate_num:04d}-well-{i % 10:02d}"
        source_url = f"/data/experiment-{plate_num:04d}/{source_id}.zarr"
        adapter = MockAdapter(
            source_id, source_url, "ome-zarr", [512, 512, 64], "uint16"
        )
        server.register_source(source_id, adapter)
        metadata_db.sync_source_added(source_id, adapter)


def _create_server_with_sources(n_sources: int):
    """Create a fresh server with n_sources pre-populated.

    Used by session-scoped fixtures to create independent servers
    for different source counts.
    """
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig

    CacheManager.initialize(CacheConfig(backend="memory"))

    port = random.randint(8900, 8999)
    location = f"grpc://127.0.0.1:{port}"

    db = MetadataDatabase(enabled=True)
    server = TensorFlightServer(location, metadata_db=db)

    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    populate_server_sources(server, db, n_sources)

    return {"server": server, "location": location, "db": db, "n_sources": n_sources}


@pytest.fixture(scope="session")
def populated_server_1k():
    """Server pre-populated with 1k sources (setup, not benchmarked)."""
    env = _create_server_with_sources(1000)
    yield {"location": env["location"], "n_sources": 1000}
    env["server"].shutdown()
    env["db"].close()


@pytest.fixture(scope="session")
def populated_server_10k():
    """Server pre-populated with 10k sources (setup, not benchmarked)."""
    env = _create_server_with_sources(10000)
    yield {"location": env["location"], "n_sources": 10000}
    env["server"].shutdown()
    env["db"].close()


@pytest.fixture(scope="session")
def populated_server_50k():
    """Server pre-populated with 50k sources (setup, not benchmarked)."""
    env = _create_server_with_sources(50000)
    yield {"location": env["location"], "n_sources": 50000}
    env["server"].shutdown()
    env["db"].close()


@pytest.fixture
def server_with_metadata_db():
    """Server with MetadataDatabase enabled for populate benchmark tests.

    This is function-scoped because populate tests benchmark the populate
    operation itself (need fresh state each time).
    """
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig

    CacheManager.initialize(CacheConfig(backend="memory"))

    port = random.randint(8900, 8999)
    location = f"grpc://127.0.0.1:{port}"

    db = MetadataDatabase(enabled=True)
    server = TensorFlightServer(location, metadata_db=db)

    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    yield {"server": server, "location": location, "db": db}

    server.shutdown()
    db.close()


class TestConcurrentAccess:
    """Concurrent read/write access benchmarks via Flight client."""

    def test_bench_concurrent_queries_same_db(self, benchmark, server_with_metadata_db):
        """Multiple threads querying the same database via TensorFlightClient."""
        server = server_with_metadata_db["server"]
        location = server_with_metadata_db["location"]
        db = server_with_metadata_db["db"]

        populate_server_sources(server, db, 1000)

        # Pre-create client pool (8 clients for 8 workers)
        clients = [TensorFlightClient(location, cache_bytes=0) for _ in range(8)]

        def concurrent_queries():
            def run_query(thread_id):
                client = clients[thread_id % 8]  # reuse pooled client
                # Different queries per thread to avoid cache hits
                if thread_id % 3 == 0:
                    sql = "SELECT source_id FROM sources WHERE source_type='ome-zarr' LIMIT 10"
                elif thread_id % 3 == 1:
                    sql = "SELECT source_id, dtype FROM sources WHERE dtype='uint16' LIMIT 10"
                else:
                    sql = "SELECT COUNT(*) FROM sources"

                result = client.query_sources(sql)
                return result.num_rows

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(run_query, i) for i in range(32)]
                results = [f.result() for f in futures]

            return results

        results = benchmark(concurrent_queries)
        assert len(results) == 32

        # Clean up client pool
        for c in clients:
            c.close()

    def test_bench_concurrent_sync_and_query(self, benchmark, server_with_metadata_db):
        """Concurrent source additions while queries are running via Flight client."""
        server = server_with_metadata_db["server"]
        location = server_with_metadata_db["location"]
        db = server_with_metadata_db["db"]

        populate_server_sources(server, db, 1000)  # Initial population

        # Pre-create client pool for query workers
        query_clients = [TensorFlightClient(location, cache_bytes=0) for _ in range(8)]

        def concurrent_mixed():
            errors = []

            def sync_worker(thread_id):
                try:
                    source_id = f"dynamic-{thread_id}"
                    adapter = MockAdapter(
                        source_id,
                        f"/data/{source_id}.zarr",
                        "ome-zarr",
                        [256, 256],
                        "float32",
                    )
                    server.register_source(source_id, adapter)
                    db.sync_source_added(source_id, adapter)
                    db.sync_source_removed(source_id)
                    server.unregister_source(source_id)
                except Exception as e:
                    errors.append(str(e))

            def query_worker(thread_id):
                try:
                    client = query_clients[thread_id % 8]  # reuse pooled client
                    sql = "SELECT COUNT(*) FROM sources"
                    client.query_sources(sql)
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

        # Clean up client pool
        for c in query_clients:
            c.close()


class TestLargeScale:
    """Performance with large source catalogs via Flight client."""

    def test_bench_populate_10k_sources(self, benchmark, server_with_metadata_db):
        """Time to populate server and metadata_db with 10,000 sources."""
        server = server_with_metadata_db["server"]
        db = server_with_metadata_db["db"]

        benchmark.pedantic(lambda: populate_server_sources(server, db, 10000), rounds=1)

        # Verify count via direct db query
        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 10000

    @pytest.mark.slow
    def test_bench_populate_50k_sources(self, benchmark, server_with_metadata_db):
        """Time to populate server and metadata_db with 50,000 sources."""
        server = server_with_metadata_db["server"]
        db = server_with_metadata_db["db"]

        benchmark.pedantic(lambda: populate_server_sources(server, db, 50000), rounds=1)

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 50000

    @pytest.mark.slow
    def test_bench_populate_100k_sources(self, benchmark, server_with_metadata_db):
        """Time to populate server and metadata_db with 100,000 sources."""
        server = server_with_metadata_db["server"]
        db = server_with_metadata_db["db"]

        benchmark.pedantic(
            lambda: populate_server_sources(server, db, 100000), rounds=1
        )

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 100000

    def test_bench_query_latency_1k(self, benchmark, populated_server_1k):
        """Query latency with 1k sources via TensorFlightClient."""
        location = populated_server_1k["location"]
        n_sources = populated_server_1k["n_sources"]

        client = TensorFlightClient(location, cache_bytes=0)

        def query_all():
            sql = "SELECT source_id FROM sources"
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == n_sources

        client.close()

    def test_bench_query_latency_10k(self, benchmark, populated_server_10k):
        """Query latency with 10k sources via TensorFlightClient."""
        location = populated_server_10k["location"]
        n_sources = populated_server_10k["n_sources"]

        client = TensorFlightClient(location, cache_bytes=0)

        def query_all():
            sql = "SELECT source_id FROM sources"
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == n_sources

        client.close()

    def test_bench_query_latency_50k(self, benchmark, populated_server_50k):
        """Query latency with 50k sources via TensorFlightClient."""
        location = populated_server_50k["location"]
        n_sources = populated_server_50k["n_sources"]

        client = TensorFlightClient(location, cache_bytes=0)

        def query_all():
            sql = "SELECT source_id FROM sources"
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == n_sources

        client.close()

    @pytest.mark.slow
    def test_bench_query_latency_100k(self, benchmark, server_with_metadata_db):
        """Query latency with 100k sources via TensorFlightClient."""
        server = server_with_metadata_db["server"]
        location = server_with_metadata_db["location"]
        db = server_with_metadata_db["db"]

        populate_server_sources(server, db, 100000)

        client = TensorFlightClient(location, cache_bytes=0)

        def query_all():
            sql = "SELECT source_id FROM sources"
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(query_all)
        assert n_rows == 100000

        client.close()


class TestQueryComplexity:
    """Performance with different query complexity levels via Flight client."""

    def test_bench_simple_count_50k(self, benchmark, populated_server_50k):
        """Simple COUNT query with 50k sources via TensorFlightClient."""
        location = populated_server_50k["location"]
        n_sources = populated_server_50k["n_sources"]

        client = TensorFlightClient(location, cache_bytes=0)

        def count_query():
            sql = "SELECT COUNT(*) FROM sources"
            result = client.query_sources(sql)
            return result.column(0).to_pylist()[0]

        count = benchmark(count_query)
        assert count == n_sources

        client.close()

    def test_bench_filtered_query_50k(self, benchmark, populated_server_50k):
        """Filtered query (WHERE clause) with 50k sources via TensorFlightClient."""
        location = populated_server_50k["location"]

        client = TensorFlightClient(location, cache_bytes=0)

        def filtered_query():
            # This filter matches ~20 sources (plate-0000 and plate-0001)
            sql = (
                "SELECT source_id FROM sources "
                "WHERE source_url LIKE '%experiment-0000%' "
                "OR source_url LIKE '%experiment-0001%'"
            )
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(filtered_query)
        assert n_rows == 20  # 2 plates * 10 wells each

        client.close()

    def test_bench_json_field_query_50k(self, benchmark, populated_server_50k):
        """Query using JSON field extraction with 50k sources via TensorFlightClient."""
        location = populated_server_50k["location"]

        client = TensorFlightClient(location, cache_bytes=0)

        def json_query():
            sql = "SELECT source_id, metadata_json->>'plate_id' as plate FROM sources LIMIT 1000"
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(json_query)
        assert n_rows == 1000

        client.close()

    def test_bench_complex_filter_50k(self, benchmark, populated_server_50k):
        """Complex multi-condition filter with 50k sources via TensorFlightClient."""
        location = populated_server_50k["location"]

        client = TensorFlightClient(location, cache_bytes=0)

        def complex_query():
            sql = """
                SELECT source_id, source_url
                FROM sources
                WHERE source_type='ome-zarr'
                  AND dtype='uint16'
                  AND shape_summary LIKE '%512%'
                LIMIT 500
            """
            result = client.query_sources(sql)
            return result.num_rows

        n_rows = benchmark(complex_query)
        assert n_rows == 500

        client.close()
