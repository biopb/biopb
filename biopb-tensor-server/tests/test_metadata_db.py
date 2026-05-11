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
import pyarrow as pa
import pyarrow.flight as flight

from biopb_tensor_server.metadata_db import MetadataDatabase


class MockAdapter:
    """Mock adapter for testing metadata sync."""

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
        return {"test_key": "test_value", "nested": {"a": 1, "b": 2}}


class TestMetadataDatabaseInit:
    """Test database initialization."""

    def test_init_enabled(self):
        """Test initialization with enabled=True."""
        db = MetadataDatabase(enabled=True)
        assert db._enabled is True
        assert db._conn is None  # Lazy initialization

    def test_init_disabled(self):
        """Test initialization with enabled=False."""
        db = MetadataDatabase(enabled=False)
        assert db._enabled is False

    def test_lazy_initialization(self):
        """Test that database is created on first access."""
        db = MetadataDatabase(enabled=True)
        assert db._conn is None

        # Trigger initialization via sync
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
        db.sync_source_added("test-1", adapter)

        assert db._conn is not None

    def test_schema_created(self):
        """Test that sources table is created with correct schema."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
        db.sync_source_added("test-1", adapter)

        conn = db._get_connection()

        # Check table exists
        result = conn.execute("SELECT * FROM sources WHERE source_id='test-1'").fetchone()
        assert result is not None
        assert result[0] == "test-1"


class TestSourceSync:
    """Test source sync operations."""

    def test_sync_source_added(self):
        """Test adding a source to the database."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("plate-001", "/data/plate.ome.zarr", "ome-zarr", [512, 512, 64], "uint16")

        db.sync_source_added("plate-001", adapter)

        conn = db._get_connection()
        result = conn.execute("SELECT * FROM sources WHERE source_id='plate-001'").fetchone()

        assert result is not None
        assert result[0] == "plate-001"
        assert result[1] == "/data/plate.ome.zarr"
        assert result[2] == "ome-zarr"
        assert result[3] == "uint16"
        assert result[5] is not None  # metadata_json
        assert result[6] == "[512, 512, 64]"  # shape_summary

    def test_sync_source_removed(self):
        """Test removing a source from the database."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")

        db.sync_source_added("test-1", adapter)

        # Verify source exists
        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources WHERE source_id='test-1'").fetchone()
        assert result[0] == 1

        # Remove source
        db.sync_source_removed("test-1")

        # Verify source removed
        result = conn.execute("SELECT COUNT(*) FROM sources WHERE source_id='test-1'").fetchone()
        assert result[0] == 0

    def test_sync_disabled_noop(self):
        """Test that sync operations are no-ops when disabled."""
        db = MetadataDatabase(enabled=False)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")

        # These should be no-ops
        db.sync_source_added("test-1", adapter)
        db.sync_source_removed("test-1")

        # Database should not be initialized
        assert db._conn is None

    def test_initial_sync_batch_insert(self):
        """Test batch insert of existing sources."""
        db = MetadataDatabase(enabled=True)

        sources = {
            "plate-001": MockAdapter("plate-001", "/data/plate1.zarr", "ome-zarr", [256, 256], "uint8"),
            "plate-002": MockAdapter("plate-002", "/data/plate2.zarr", "ome-zarr", [512, 512], "uint16"),
            "plate-003": MockAdapter("plate-003", "/data/plate3.zarr", "ome-zarr", [1024, 1024], "float32"),
        }

        db.initial_sync(sources)

        conn = db._get_connection()
        result = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        assert result[0] == 3

        # Check each source
        for source_id in sources:
            row = conn.execute("SELECT source_id FROM sources WHERE source_id=?", [source_id]).fetchone()
            assert row is not None


class TestQueryHandling:
    """Test SQL query handling."""

    def test_handle_query_simple(self):
        """Test simple SELECT query."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
        db.sync_source_added("test-1", adapter)

        info = db.handle_query("SELECT source_id, source_type FROM sources")
        assert info is not None
        assert len(info.endpoints) == 1

        # Check schema metadata
        assert info.schema.metadata is not None
        assert b'total_sources' in info.schema.metadata
        assert int(info.schema.metadata[b'total_sources']) == 1

    def test_handle_query_with_filter(self):
        """Test SELECT query with WHERE clause."""
        db = MetadataDatabase(enabled=True)

        # Add multiple sources
        db.sync_source_added("zarr-1", MockAdapter("zarr-1", "/data/z1.zarr", "ome-zarr", [100, 100], "uint16"))
        db.sync_source_added("zarr-2", MockAdapter("zarr-2", "/data/z2.zarr", "ome-zarr", [200, 200], "uint16"))
        db.sync_source_added("tiff-1", MockAdapter("tiff-1", "/data/t1.tiff", "ome-tiff", [300, 300], "uint8"))

        info = db.handle_query("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
        assert info is not None

        # Retrieve result
        result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
        assert result is not None
        assert result.num_rows == 2

    def test_handle_query_json_field(self):
        """Test query using DuckDB JSON operators."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
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
        db = MetadataDatabase(enabled=True, max_query_results=2)

        # Add 5 sources
        for i in range(5):
            db.sync_source_added(f"test-{i}", MockAdapter(f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"))

        info = db.handle_query("SELECT source_id FROM sources")
        assert info is not None

        # Check truncation metadata
        assert int(info.schema.metadata[b'total_sources']) == 5
        assert int(info.schema.metadata[b'returned_sources']) == 2

        # Result should be truncated
        result = db.get_pending_result(info.endpoints[0].ticket.ticket.decode())
        assert result.num_rows == 2

    def test_handle_query_disabled_raises(self):
        """Test that query raises when disabled."""
        db = MetadataDatabase(enabled=False)

        with pytest.raises(ValueError, match="disabled"):
            db.handle_query("SELECT * FROM sources")


class TestSQLValidation:
    """Test SQL query validation."""

    def test_validate_simple_select(self):
        """Test that simple SELECT passes validation."""
        db = MetadataDatabase(enabled=True)

        db._validate_query("SELECT * FROM sources")
        db._validate_query("SELECT source_id, source_type FROM sources WHERE dtype='uint16'")

    def test_validate_forbidden_insert(self):
        """Test that INSERT is blocked."""
        db = MetadataDatabase(enabled=True)

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("INSERT INTO sources VALUES ('test', 'test')")

    def test_validate_forbidden_update(self):
        """Test that UPDATE is blocked."""
        db = MetadataDatabase(enabled=True)

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("UPDATE sources SET source_id='new' WHERE source_id='old'")

    def test_validate_forbidden_delete(self):
        """Test that DELETE is blocked."""
        db = MetadataDatabase(enabled=True)

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("DELETE FROM sources WHERE source_id='test'")

    def test_validate_forbidden_drop(self):
        """Test that DROP is blocked."""
        db = MetadataDatabase(enabled=True)

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("DROP TABLE sources")

    def test_validate_forbidden_create(self):
        """Test that CREATE is blocked."""
        db = MetadataDatabase(enabled=True)

        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query("CREATE TABLE evil AS SELECT * FROM sources")

    def test_validate_disallowed_table(self):
        """Test that references to non-sources tables are blocked."""
        db = MetadataDatabase(enabled=True)

        # Should pass - sources table
        db._validate_query("SELECT * FROM sources")

        # Should fail - imaginary table
        with pytest.raises(ValueError, match="disallowed table"):
            db._validate_query("SELECT * FROM other_table")


class TestFlightInfo:
    """Test FlightInfo generation."""

    def test_flight_info_schema(self):
        """Test that FlightInfo has correct schema."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
        db.sync_source_added("test-1", adapter)

        info = db.handle_query("SELECT source_id, source_type FROM sources")

        # Schema should have the queried columns
        assert info.schema.names == ["source_id", "source_type"]

    def test_flight_info_metadata(self):
        """Test that FlightInfo schema metadata contains counts."""
        db = MetadataDatabase(enabled=True)

        for i in range(3):
            db.sync_source_added(f"test-{i}", MockAdapter(f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"))

        info = db.handle_query("SELECT source_id FROM sources")

        assert info.schema.metadata is not None
        assert b'total_sources' in info.schema.metadata
        assert b'returned_sources' in info.schema.metadata
        assert b'query_elapsed_ms' in info.schema.metadata

    def test_get_pending_result(self):
        """Test retrieval of pending results."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
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
        db = MetadataDatabase(enabled=True)

        result = db.get_pending_result("nonexistent-ticket")
        assert result is None


class TestClose:
    """Test database close."""

    def test_close(self):
        """Test that close() cleans up connection."""
        db = MetadataDatabase(enabled=True)
        adapter = MockAdapter("test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16")
        db.sync_source_added("test-1", adapter)

        assert db._conn is not None

        db.close()

        assert db._conn is None

    def test_close_without_init(self):
        """Test that close() works even if database wasn't initialized."""
        db = MetadataDatabase(enabled=True)

        # Should not raise
        db.close()


class TestListFlightsTruncation:
    """Test list_flights truncation signaling."""

    def test_list_flights_schema_metadata(self):
        """Test that list_flights includes truncation metadata in schema."""
        from biopb_tensor_server.server import TensorFlightServer
        import pyarrow.flight as flight
        import random

        port = random.randint(8900, 8999)
        server = TensorFlightServer(
            f"grpc://localhost:{port}",
            max_list_flights_results=5,
        )

        # Register 3 sources (under limit)
        for i in range(3):
            adapter = MockAdapter(f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16")
            server.register_source(f"test-{i}", adapter)

        # Call list_flights
        results = list(server.list_flights(None, b""))

        assert len(results) == 3

        # Check schema metadata
        info = results[0]
        assert info.schema.metadata is not None
        assert int(info.schema.metadata[b'total_sources']) == 3
        assert int(info.schema.metadata[b'max_sources']) == 5
        assert info.schema.metadata[b'truncated'].decode() == 'False'

    def test_list_flights_truncation(self):
        """Test that list_flights truncates and signals via metadata."""
        from biopb_tensor_server.server import TensorFlightServer
        import pyarrow.flight as flight
        import random

        port = random.randint(8900, 8999)
        server = TensorFlightServer(
            f"grpc://localhost:{port}",
            max_list_flights_results=3,
        )

        # Register 10 sources (over limit)
        for i in range(10):
            adapter = MockAdapter(f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16")
            server.register_source(f"test-{i}", adapter)

        # Call list_flights
        results = list(server.list_flights(None, b""))

        # Should only get 3 results (truncated)
        assert len(results) == 3

        # Check schema metadata signals truncation
        info = results[0]
        assert info.schema.metadata is not None
        assert int(info.schema.metadata[b'total_sources']) == 10
        assert int(info.schema.metadata[b'max_sources']) == 3
        assert info.schema.metadata[b'truncated'].decode() == 'True'