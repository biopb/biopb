"""Unit tests for MetadataDatabase.

Tests cover:
- Database initialization and schema creation
- Source insert/delete/query operations
- SQL validation (blocked keywords, blocked table references)
- Security boundary tests
- Schema metadata for truncation signaling
"""

import json

import pytest
from biopb_tensor_server.metadata_db import MetadataDatabase


class MockAdapter:
    """Mock adapter for testing metadata sync."""

    def __init__(
        self, source_id, source_url, source_type, shape, dtype, data_resident=True
    ):
        self.source_id = source_id
        self._source_url = source_url
        self._source_type = source_type
        self._shape = shape
        self._dtype = dtype
        self._data_resident = data_resident

    def get_source_descriptor(self):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=self._source_url,
            source_type=self._source_type,
            data_resident=self._data_resident,
            tensors=[
                TensorDescriptor(
                    array_id=self.source_id,
                    shape=self._shape,
                    dtype=self._dtype,
                )
            ],
        )

    def get_metadata(self):
        return {"test_key": "test_value", "nested": {"a": 1, "b": 2}}


class TestMetadataDatabaseInit:
    """Test database initialization."""

    def test_init_lazy(self):
        """The DB is mandatory (no enabled flag); the connection is still lazy."""
        db = MetadataDatabase()
        assert db._conn is None  # Lazy initialization

    def test_lazy_initialization(self):
        """Test that database is created on first access."""
        db = MetadataDatabase()
        assert db._conn is None

        # Trigger initialization via sync
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        assert db._conn is not None

    def test_schema_created(self):
        """Test that sources table is created with correct schema."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        conn = db._get_connection()

        # Check table exists
        result = conn.execute(
            "SELECT * FROM sources WHERE source_id='test-1'"
        ).fetchone()
        assert result is not None
        assert result[0] == "test-1"


class TestSourceSync:
    """Test source sync operations."""

    def test_sync_source_added(self):
        """Test adding a source to the database."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "plate-001", "/data/plate.ome.zarr", "ome-zarr", [512, 512, 64], "uint16"
        )

        db.sync_source_added("plate-001", adapter)

        conn = db._get_connection()
        result = conn.execute(
            "SELECT * FROM sources WHERE source_id='plate-001'"
        ).fetchone()

        assert result is not None
        assert result[0] == "plate-001"
        assert result[1] == "/data/plate.ome.zarr"
        assert result[2] == "ome-zarr"
        assert result[3] == "uint16"
        assert result[5] is not None  # metadata_json
        assert result[6] == "[512, 512, 64]"  # shape_summary

    def test_sync_numpy_types(self):
        """Test that numpy scalar types are serialized correctly."""
        import numpy as np

        class NumpyMockAdapter(MockAdapter):
            def get_metadata(self):
                return {
                    "int16": np.int16(42),
                    "int32": np.int32(100),
                    "float32": np.float32(3.14),
                    "float64": np.float64(2.71),
                    "array": np.array([1, 2, 3]),
                    "bytes_utf8": b"hello",
                    "bytes_binary": b"\xff\xfe",
                }

        db = MetadataDatabase()
        adapter = NumpyMockAdapter(
            "numpy-test", "/data/test.nii", "nifti", [64, 64, 64], "int16"
        )

        db.sync_source_added("numpy-test", adapter)

        conn = db._get_connection()
        result = conn.execute(
            "SELECT metadata_json FROM sources WHERE source_id='numpy-test'"
        ).fetchone()

        assert result is not None

        metadata = json.loads(result[0])
        assert metadata["int16"] == 42
        assert metadata["int32"] == 100
        assert abs(metadata["float32"] - 3.14) < 0.01
        assert abs(metadata["float64"] - 2.71) < 0.01
        assert metadata["array"] == [1, 2, 3]

    def test_sync_source_removed(self):
        """Test removing a source from the database."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )

        db.sync_source_added("test-1", adapter)

        # Verify source exists
        conn = db._get_connection()
        result = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE source_id='test-1'"
        ).fetchone()
        assert result[0] == 1

        # Remove source
        db.sync_source_removed("test-1")

        # Verify source removed
        result = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE source_id='test-1'"
        ).fetchone()
        assert result[0] == 0

    def test_sync_source_added_propagates_failure(self):
        """A descriptor-read error surfaces to the caller instead of being
        swallowed, so the registration path can roll back (issue #223)."""

        class FailingAdapter(MockAdapter):
            def get_source_descriptor(self):
                raise RuntimeError("Simulated failure")

        db = MetadataDatabase()

        with pytest.raises(RuntimeError, match="Simulated failure"):
            db.sync_source_added(
                "failing",
                FailingAdapter(
                    "failing", "/data/fail.zarr", "ome-zarr", [100, 100], "uint8"
                ),
            )

        # Nothing partially written for the failed source.
        conn = db._get_connection()
        row = conn.execute(
            "SELECT source_id FROM sources WHERE source_id='failing'"
        ).fetchone()
        assert row is None


class MultiTensorAdapter:
    """Mock adapter exposing several tensors (multi-field / HCS source)."""

    def __init__(self, source_id, source_url, source_type, tensors, data_resident=True):
        self.source_id = source_id
        self._source_url = source_url
        self._source_type = source_type
        self._tensors = (
            tensors  # list of dicts: array_id, dim_labels, shape, chunk_shape, dtype
        )
        self._data_resident = data_resident

    def get_source_descriptor(self):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=self._source_url,
            source_type=self._source_type,
            data_resident=self._data_resident,
            tensors=[TensorDescriptor(**t) for t in self._tensors],
        )

    def get_metadata(self):
        return {}


class TestPerTensorCatalog:
    """Full per-tensor catalog column (biopb/biopb#224)."""

    def _fields(self):
        return [
            {
                "array_id": "hcs/A1/0",
                "dim_labels": ["y", "x"],
                "shape": [512, 512],
                "chunk_shape": [512, 512],
                "dtype": "uint16",
            },
            {
                "array_id": "hcs/A2/0",
                "dim_labels": ["z", "y", "x"],
                "shape": [8, 256, 256],
                "chunk_shape": [1, 256, 256],
                "dtype": "uint8",
            },
        ]

    def test_all_tensors_stored_not_just_first(self):
        """Every tensor is persisted, with its full structural fields -- not the
        old first-tensor projection only."""
        db = MetadataDatabase()
        db.sync_source_added(
            "hcs",
            MultiTensorAdapter("hcs", "/data/hcs.zarr", "ome-zarr", self._fields()),
        )

        conn = db._get_connection()
        rows = conn.execute(
            "SELECT u.t.array_id, u.t.dim_labels, u.t.shape, u.t.chunk_shape, u.t.dtype "
            "FROM sources, UNNEST(tensors) AS u(t) ORDER BY u.t.array_id"
        ).fetchall()

        assert rows == [
            ("hcs/A1/0", ["y", "x"], [512, 512], [512, 512], "uint16"),
            ("hcs/A2/0", ["z", "y", "x"], [8, 256, 256], [1, 256, 256], "uint8"),
        ]

    def test_per_tensor_dtype_filter(self):
        """A dtype predicate over the nested list finds a source by ANY of its
        tensors -- the multi-field case the first-tensor projection missed."""
        db = MetadataDatabase()
        # first tensor is uint16; the uint8 tensor is only reachable per-tensor
        db.sync_source_added(
            "hcs",
            MultiTensorAdapter("hcs", "/data/hcs.zarr", "ome-zarr", self._fields()),
        )
        db.sync_source_added(
            "plain",
            MockAdapter("plain", "/data/p.zarr", "zarr", [10, 10], "float32"),
        )

        conn = db._get_connection()
        rows = conn.execute(
            "SELECT source_id FROM sources "
            "WHERE len(list_filter(tensors, t -> t.dtype = 'uint8')) > 0"
        ).fetchall()
        assert rows == [("hcs",)]

    def test_scalar_projection_still_first_tensor(self):
        """The back-compat scalar dtype/shape_summary stay the first-tensor
        projection, written in the same upsert so they can't desync."""
        db = MetadataDatabase()
        db.sync_source_added(
            "hcs",
            MultiTensorAdapter("hcs", "/data/hcs.zarr", "ome-zarr", self._fields()),
        )
        conn = db._get_connection()
        dtype, shape_summary = conn.execute(
            "SELECT dtype, shape_summary FROM sources WHERE source_id='hcs'"
        ).fetchone()
        assert dtype == "uint16"  # tensors[0]
        assert shape_summary == "[512, 512]"

    def test_no_tensors_is_empty_list(self):
        """An unresolved (no-tensor) source stores an empty list, so per-tensor
        predicates exclude it while `WHERE NOT data_resident` still finds it."""
        db = MetadataDatabase()
        db.sync_source_added(
            "unresolved",
            MultiTensorAdapter(
                "unresolved", "s3://b/x.zarr", "zarr", [], data_resident=False
            ),
        )
        conn = db._get_connection()
        tensors, resident = conn.execute(
            "SELECT tensors, data_resident FROM sources WHERE source_id='unresolved'"
        ).fetchone()
        assert tensors == []
        assert resident is False

    def test_per_tensor_query_roundtrips_through_handle_query(self):
        """The documented per-tensor idiom works through the real query path
        (handle_query: SQL validator -> Arrow -> Flight ticket), not just the raw
        connection the other tests use. UNNEST(tensors) must pass _validate_query
        (it references the column, not a table) and the nested LIST(STRUCT) column
        must round-trip whole through the Arrow serialization behind DoGet."""
        db = MetadataDatabase(enabled=True)
        db.sync_source_added(
            "hcs",
            MultiTensorAdapter("hcs", "/data/hcs.zarr", "ome-zarr", self._fields()),
        )
        db.sync_source_added(
            "unresolved",
            MultiTensorAdapter(
                "unresolved", "s3://b/x.zarr", "zarr", [], data_resident=False
            ),
        )

        # UNNEST -> one row per tensor, through the validator + Arrow path.
        info = db.handle_query(
            "SELECT source_id, t.array_id, t.dtype "
            "FROM sources, UNNEST(tensors) AS u(t) ORDER BY t.array_id"
        )
        rows = db.get_pending_result(
            info.endpoints[0].ticket.ticket.decode()
        ).to_pylist()
        assert rows == [
            {"source_id": "hcs", "array_id": "hcs/A1/0", "dtype": "uint16"},
            {"source_id": "hcs", "array_id": "hcs/A2/0", "dtype": "uint8"},
        ]

        # The nested column itself round-trips whole -- including the empty list
        # for the unresolved source (Arrow/Flight handles the LIST(STRUCT) type).
        info = db.handle_query(
            "SELECT source_id, tensors FROM sources ORDER BY source_id"
        )
        by_id = {
            r["source_id"]: r["tensors"]
            for r in db.get_pending_result(
                info.endpoints[0].ticket.ticket.decode()
            ).to_pylist()
        }
        assert by_id["unresolved"] == []
        assert [t["dtype"] for t in by_id["hcs"]] == ["uint16", "uint8"]
        assert by_id["hcs"][1]["shape"] == [8, 256, 256]  # full struct, not projection


class TestQueryHandling:
    """Test SQL query handling."""

    def test_handle_query_simple(self):
        """Test simple SELECT query."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        info = db.handle_query("SELECT source_id, source_type FROM sources")
        assert info is not None
        assert len(info.endpoints) == 1

        # Check schema metadata
        assert info.schema.metadata is not None
        assert b"total_sources" in info.schema.metadata
        assert int(info.schema.metadata[b"total_sources"]) == 1

    def test_handle_query_with_filter(self):
        """Test SELECT query with WHERE clause."""
        db = MetadataDatabase()

        # Add multiple sources
        db.sync_source_added(
            "zarr-1",
            MockAdapter("zarr-1", "/data/z1.zarr", "ome-zarr", [100, 100], "uint16"),
        )
        db.sync_source_added(
            "zarr-2",
            MockAdapter("zarr-2", "/data/z2.zarr", "ome-zarr", [200, 200], "uint16"),
        )
        db.sync_source_added(
            "tiff-1",
            MockAdapter("tiff-1", "/data/t1.tiff", "ome-tiff", [300, 300], "uint8"),
        )

        info = db.handle_query(
            "SELECT source_id FROM sources WHERE source_type='ome-zarr'"
        )
        assert info is not None

        # Retrieve result
        result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
        assert result is not None
        assert result.num_rows == 2

    def test_handle_query_json_field(self):
        """Test query using DuckDB JSON operators."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        # Query JSON field
        info = db.handle_query(
            "SELECT source_id, metadata_json->>'test_key' as test_key FROM sources"
        )
        assert info is not None

        result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
        assert result is not None
        assert result.num_rows == 1

    def test_handle_query_truncation(self):
        """Test truncation signaling when results exceed max."""
        db = MetadataDatabase(max_query_results=2)

        # Add 5 sources
        for i in range(5):
            db.sync_source_added(
                f"test-{i}",
                MockAdapter(
                    f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"
                ),
            )

        info = db.handle_query("SELECT source_id FROM sources")
        assert info is not None

        # Check truncation metadata
        assert int(info.schema.metadata[b"total_sources"]) == 5
        assert int(info.schema.metadata[b"returned_sources"]) == 2

        # Result should be truncated
        result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
        assert result.num_rows == 2


class TestSQLValidation:
    """Test SQL query validation."""

    def test_validate_simple_select(self):
        """Test that simple SELECT passes validation."""
        db = MetadataDatabase()

        db._validate_query("SELECT * FROM sources")
        db._validate_query(
            "SELECT source_id, source_type FROM sources WHERE dtype='uint16'"
        )

    def test_validate_forbidden_insert(self):
        """Test that INSERT is blocked."""
        db = MetadataDatabase()

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("INSERT INTO sources VALUES ('test', 'test')")

    def test_validate_forbidden_update(self):
        """Test that UPDATE is blocked."""
        db = MetadataDatabase()

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query(
                "UPDATE sources SET source_id='new' WHERE source_id='old'"
            )

    def test_validate_forbidden_delete(self):
        """Test that DELETE is blocked."""
        db = MetadataDatabase()

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("DELETE FROM sources WHERE source_id='test'")

    def test_validate_forbidden_drop(self):
        """Test that DROP is blocked."""
        db = MetadataDatabase()

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("DROP TABLE sources")

    def test_validate_forbidden_create(self):
        """Test that CREATE is blocked."""
        db = MetadataDatabase()

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("CREATE TABLE evil AS SELECT * FROM sources")

    def test_validate_disallowed_table(self):
        """Test that references to non-sources tables are blocked."""
        db = MetadataDatabase()

        # Should pass - sources table
        db._validate_query("SELECT * FROM sources")

        # Should fail - imaginary table
        with pytest.raises(ValueError, match="disallowed table"):
            db._validate_query("SELECT * FROM other_table")

    def test_external_file_access_blocked(self, tmp_path):
        """File-reading SQL is blocked at the engine (A3).

        A comma-join slips a file-reading table function past the FROM-only
        keyword/table denylist, so the real defense is the connection's
        enable_external_access=False: the query must fail rather than read the
        file.
        """
        db = MetadataDatabase()
        db.sync_source_added(
            "z1",
            MockAdapter("z1", "/data/z1.zarr", "ome-zarr", [10, 10], "uint16"),
        )

        secret = tmp_path / "secret.txt"
        secret.write_text("TOP-SECRET")

        # Bypasses the denylist validator (FROM only sees `sources`)...
        db._validate_query(f"SELECT * FROM sources, read_text('{secret}')")
        # ...but execution is blocked by enable_external_access=False.
        with pytest.raises(ValueError, match="file system operations are disabled"):
            db.handle_query(f"SELECT content FROM sources, read_text('{secret}')")

    def test_set_external_access_cannot_be_reenabled(self):
        """An attacker can't turn external access back on mid-query."""
        db = MetadataDatabase()
        db.sync_source_added(
            "z1",
            MockAdapter("z1", "/data/z1.zarr", "ome-zarr", [10, 10], "uint16"),
        )
        with pytest.raises(ValueError):
            db.handle_query("SET enable_external_access=true; SELECT * FROM sources")


class TestFlightInfo:
    """Test FlightInfo generation."""

    def test_flight_info_schema(self):
        """Test that FlightInfo has correct schema."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        info = db.handle_query("SELECT source_id, source_type FROM sources")

        # Schema should have the queried columns
        assert info.schema.names == ["source_id", "source_type"]

    def test_flight_info_metadata(self):
        """Test that FlightInfo schema metadata contains counts."""
        db = MetadataDatabase()

        for i in range(3):
            db.sync_source_added(
                f"test-{i}",
                MockAdapter(
                    f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"
                ),
            )

        info = db.handle_query("SELECT source_id FROM sources")

        assert info.schema.metadata is not None
        assert b"total_sources" in info.schema.metadata
        assert b"returned_sources" in info.schema.metadata
        assert b"query_elapsed_ms" in info.schema.metadata

    def test_get_pending_result(self):
        """Test retrieval of pending results."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        info = db.handle_query("SELECT source_id FROM sources")
        ticket_id = info.endpoints[0].ticket.ticket.decode()

        # Retrieve result
        result = db.get_pending_result(ticket_id)
        assert result is not None
        assert result.num_rows == 1

        # Second retrieval should return None (result was consumed)
        result2 = db.get_pending_result(ticket_id)
        assert result2 is None

    def test_get_pending_result_not_found(self):
        """Test retrieval of non-existent ticket."""
        db = MetadataDatabase()

        result = db.get_pending_result("nonexistent-ticket")
        assert result is None


class TestClose:
    """Test database close."""

    def test_close(self):
        """Test that close() cleans up connection."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        assert db._conn is not None

        db.close()

        assert db._conn is None

    def test_close_without_init(self):
        """Test that close() works even if database wasn't initialized."""
        db = MetadataDatabase()

        # Should not raise
        db.close()


class TestListFlightsTruncation:
    """Test list_flights truncation signaling."""

    def test_list_flights_schema_metadata(self):
        """Test that list_flights includes truncation metadata in schema."""
        from biopb_tensor_server.server import TensorFlightServer

        # Bind to port 0: the OS assigns a free port and this test never connects
        # a client (it calls list_flights in-process), so the value is irrelevant.
        # Fixed-range random ports caused flaky "Address already in use" failures
        # under the full suite.
        server = TensorFlightServer(
            "grpc://localhost:0",
            max_list_flights_results=5,
        )

        # Register 3 sources (under limit)
        for i in range(3):
            adapter = MockAdapter(
                f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"
            )
            server.register_source(f"test-{i}", adapter)

        # Call list_flights
        results = list(server.list_flights(None, b""))

        assert len(results) == 3

        # Check schema metadata
        info = results[0]
        assert info.schema.metadata is not None
        assert int(info.schema.metadata[b"total_sources"]) == 3
        assert int(info.schema.metadata[b"max_sources"]) == 5
        assert info.schema.metadata[b"truncated"].decode() == "False"

    def test_list_flights_truncation(self):
        """Test that list_flights truncates and signals via metadata."""
        from biopb_tensor_server.server import TensorFlightServer

        # Bind to port 0: the OS assigns a free port and this test never connects
        # a client (it calls list_flights in-process), so the value is irrelevant.
        # Fixed-range random ports caused flaky "Address already in use" failures
        # under the full suite.
        server = TensorFlightServer(
            "grpc://localhost:0",
            max_list_flights_results=3,
        )

        # Register 10 sources (over limit)
        for i in range(10):
            adapter = MockAdapter(
                f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"
            )
            server.register_source(f"test-{i}", adapter)

        # Call list_flights
        results = list(server.list_flights(None, b""))

        # Should only get 3 results (truncated)
        assert len(results) == 3

        # Check schema metadata signals truncation
        info = results[0]
        assert info.schema.metadata is not None
        assert int(info.schema.metadata[b"total_sources"]) == 10
        assert int(info.schema.metadata[b"max_sources"]) == 3
        assert info.schema.metadata[b"truncated"].decode() == "True"

    def test_list_flights_uses_stable_snapshot_during_mutation(self):
        """list_flights should not fail if sources mutate mid-iteration."""
        from biopb_tensor_server.server import TensorFlightServer

        # Bind to port 0: the OS assigns a free port and this test never connects
        # a client (it calls list_flights in-process), so the value is irrelevant.
        # Fixed-range random ports caused flaky "Address already in use" failures
        # under the full suite.
        server = TensorFlightServer(
            "grpc://localhost:0",
            max_list_flights_results=10,
        )

        class MutatingAdapter(MockAdapter):
            def __init__(self, *args, on_descriptor=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._on_descriptor = on_descriptor

            def get_source_descriptor(self):
                if self._on_descriptor is not None:
                    self._on_descriptor()
                    self._on_descriptor = None
                return super().get_source_descriptor()

        def mutate_sources():
            server.unregister_source("test-2")

        server.register_source(
            "test-1",
            MutatingAdapter(
                "test-1",
                "/data/test1.zarr",
                "ome-zarr",
                [100, 100],
                "uint16",
                on_descriptor=mutate_sources,
            ),
        )
        server.register_source(
            "test-2",
            MockAdapter("test-2", "/data/test2.zarr", "ome-zarr", [100, 100], "uint16"),
        )

        results = list(server.list_flights(None, b""))

        assert len(results) == 2


class TestDataResidentColumn:
    """The `data_resident` column (#110): a queryable residency signal so
    unresolved (cloud) sources can be filtered on purpose, not hidden by NULLs."""

    class _UnresolvedAdapter:
        """A cloud / synced-folder source catalogued by URL only: no tensors
        (so NULL dtype/shape_summary) and not resident until resolved."""

        def __init__(self, source_id, source_url):
            self.source_id = source_id
            self._source_url = source_url

        def get_source_descriptor(self):
            from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

            return DataSourceDescriptor(
                source_id=self.source_id,
                source_url=self._source_url,
                source_type="unresolved",
                data_resident=False,  # not local yet
                # no tensors -> NULL dtype / shape_summary
            )

        def get_metadata(self):
            return {}

    def test_resident_local_source_is_true(self):
        db = MetadataDatabase()
        db.sync_source_added(
            "local-1",
            MockAdapter(
                "local-1",
                "/data/x.zarr",
                "ome-zarr",
                [10, 10],
                "uint8",
                data_resident=True,
            ),
        )
        row = (
            db._get_connection()
            .execute("SELECT data_resident FROM sources WHERE source_id='local-1'")
            .fetchone()
        )
        assert row[0] is True

    def test_unresolved_source_is_false_and_filterable(self):
        # An unresolved source has NULL dtype, so a dtype predicate hides it;
        # data_resident makes it filterable on purpose instead.
        db = MetadataDatabase()
        db.sync_source_added(
            "local-1",
            MockAdapter("local-1", "/x.zarr", "ome-zarr", [10, 10], "uint8"),
        )
        db.sync_source_added(
            "cloud-1", self._UnresolvedAdapter("cloud-1", "https://x/y.zarr")
        )
        conn = db._get_connection()

        resident = conn.execute(
            "SELECT data_resident FROM sources WHERE source_id='cloud-1'"
        ).fetchone()
        assert resident[0] is False

        # The footgun: a dtype filter silently drops the unresolved source...
        by_dtype = conn.execute(
            "SELECT source_id FROM sources WHERE dtype='uint8'"
        ).fetchall()
        assert [r[0] for r in by_dtype] == ["local-1"]

        # ...but residency makes the unresolved one discoverable on purpose.
        unresolved = conn.execute(
            "SELECT source_id FROM sources WHERE NOT data_resident"
        ).fetchall()
        assert [r[0] for r in unresolved] == ["cloud-1"]

    def test_column_is_not_null_with_false_default(self):
        # The NOT NULL DEFAULT FALSE constraint: an insert omitting data_resident
        # gets FALSE (future insert paths can't accidentally leave it NULL), and
        # an explicit NULL is rejected -- so the column always partitions cleanly.
        import duckdb

        db = MetadataDatabase()
        db.sync_source_added(
            "seed", MockAdapter("seed", "/s.zarr", "ome-zarr", [4, 4], "uint8")
        )
        conn = db._get_connection()

        conn.execute(
            "INSERT INTO sources (source_id, source_url) VALUES ('partial', '/p')"
        )
        row = conn.execute(
            "SELECT data_resident FROM sources WHERE source_id='partial'"
        ).fetchone()
        assert row[0] is False  # DEFAULT filled it, not NULL

        with pytest.raises(duckdb.ConstraintException):
            conn.execute(
                "INSERT INTO sources (source_id, data_resident) VALUES ('bad', NULL)"
            )
