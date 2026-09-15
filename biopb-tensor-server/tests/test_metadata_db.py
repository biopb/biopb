"""Unit tests for MetadataDatabase.

Tests cover:
- Database initialization and schema creation
- Source insert/delete/query operations
- SQL validation (blocked keywords, blocked table references)
- Security boundary tests
- Schema metadata for truncation signaling
"""

import json

import duckdb
import pytest
from biopb_tensor_server.serving.metadata_db import MetadataDatabase


class MockAdapter:
    """Mock adapter for testing metadata sync."""

    capability_token = None

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
            "SELECT u.t.array_id, u.t.dim_labels, u.t.shape, u.t.dtype "
            "FROM sources, UNNEST(tensors) AS u(t) ORDER BY u.t.array_id"
        ).fetchall()

        assert rows == [
            ("hcs/A1/0", ["y", "x"], [512, 512], "uint16"),
            ("hcs/A2/0", ["z", "y", "x"], [8, 256, 256], "uint8"),
        ]

    def test_catalog_is_structural_and_stores_no_transfer_grid(self):
        """The per-tensor STRUCT carries structure only -- no read plan.

        ``chunk_shape`` is the transfer grid of the adapter bound to a *specific*
        tensor, and it can depend on facts that exist only after that binding (a
        scene's own Dask chunks, its labels, its native pyramid level, the
        request's scale). A source-level listing that names one is guessing for a
        scene it never selected, so the catalog stores none and GetFlightInfo
        answers it per resolved tensor (biopb/biopb#812). The adapter double here
        hands one in anyway; the column it would go in does not exist.
        """
        db = MetadataDatabase()
        db.sync_source_added(
            "hcs",
            MultiTensorAdapter("hcs", "/data/hcs.zarr", "ome-zarr", self._fields()),
        )

        conn = db._get_connection()
        (struct,) = conn.execute(
            "SELECT u.t FROM sources, UNNEST(tensors) AS u(t) LIMIT 1"
        ).fetchone()
        assert set(struct) == {
            "array_id",
            "dim_labels",
            "shape",
            "dtype",
        }

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

    def test_per_tensor_query_roundtrips_through_query(self):
        """The documented per-tensor idiom works through the real query path
        (query: SQL validator -> Arrow table), not just the raw
        connection the other tests use. UNNEST(tensors) must pass _validate_query
        (it references the column, not a table) and the nested LIST(STRUCT) column
        must round-trip whole through the Arrow serialization behind DoGet."""
        db = MetadataDatabase()
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
        rows = db.query(
            "SELECT source_id, t.array_id, t.dtype "
            "FROM sources, UNNEST(tensors) AS u(t) ORDER BY t.array_id"
        ).to_pylist()
        assert rows == [
            {"source_id": "hcs", "array_id": "hcs/A1/0", "dtype": "uint16"},
            {"source_id": "hcs", "array_id": "hcs/A2/0", "dtype": "uint8"},
        ]

        # The nested column itself round-trips whole -- including the empty list
        # for the unresolved source (Arrow/Flight handles the LIST(STRUCT) type).
        by_id = {
            r["source_id"]: r["tensors"]
            for r in db.query(
                "SELECT source_id, tensors FROM sources ORDER BY source_id"
            ).to_pylist()
        }
        assert by_id["unresolved"] == []
        assert [t["dtype"] for t in by_id["hcs"]] == ["uint16", "uint8"]
        assert by_id["hcs"][1]["shape"] == [8, 256, 256]  # full struct, not projection


def _descriptors(db):
    """The SDK's projection of the catalog rows (what list_sources returns)."""
    from biopb.tensor._catalog_rows import SOURCE_ROW_COLUMNS, descriptors_from_rows

    return descriptors_from_rows(
        db.query(
            f"SELECT {SOURCE_ROW_COLUMNS} FROM sources ORDER BY source_id"
        ).to_pylist()
    )


class TestSourceRowProjection:
    """A catalog row rebuilds the lean DataSourceDescriptor (biopb/biopb#265)."""

    def test_empty_catalog_returns_no_rows(self):
        assert _descriptors(MetadataDatabase()) == []

    def test_reconstructs_lean_descriptor(self):
        db = MetadataDatabase()
        db.sync_source_added(
            "s1", MockAdapter("s1", "/data/s1.zarr", "zarr", [8, 512, 512], "uint16")
        )

        descriptors = _descriptors(db)
        assert len(descriptors) == 1
        d = descriptors[0]
        assert d.source_id == "s1"
        assert d.source_url == "/data/s1.zarr"
        assert d.source_type == "zarr"
        assert d.data_resident is True
        assert d.metadata_json == ""  # lean: filled only by GetFlightInfo
        assert len(d.tensors) == 1
        assert d.tensors[0].array_id == "s1"
        assert list(d.tensors[0].shape) == [8, 512, 512]
        assert d.tensors[0].dtype == "uint16"

    def test_multi_tensor_all_reconstructed(self):
        db = MetadataDatabase()
        fields = [
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
        db.sync_source_added(
            "hcs", MultiTensorAdapter("hcs", "/data/hcs.zarr", "ome-zarr", fields)
        )

        tensors = _descriptors(db)[0].tensors
        assert [t.array_id for t in tensors] == ["hcs/A1/0", "hcs/A2/0"]
        assert list(tensors[1].dim_labels) == ["z", "y", "x"]
        assert list(tensors[1].shape) == [8, 256, 256]
        # Structural only: no grid is stored, so none is reconstructed
        # (biopb/biopb#812).
        assert list(tensors[1].chunk_shape) == []
        assert tensors[1].dtype == "uint8"

    def test_ordered_by_source_id(self):
        db = MetadataDatabase()
        for sid in ("c", "a", "b"):
            db.sync_source_added(
                sid, MockAdapter(sid, f"/d/{sid}", "zarr", [4, 4], "uint8")
            )
        assert [d.source_id for d in _descriptors(db)] == ["a", "b", "c"]

    def test_unresolved_source_has_no_tensors(self):
        db = MetadataDatabase()
        db.sync_source_added(
            "u",
            MultiTensorAdapter("u", "s3://b/x.zarr", "zarr", [], data_resident=False),
        )
        descriptors = _descriptors(db)
        assert len(descriptors[0].tensors) == 0
        assert descriptors[0].data_resident is False


class TestGetMetadataJson:
    """Local metadata read-back for the serve path (biopb/biopb#253)."""

    def test_returns_parsed_dict(self):
        db = MetadataDatabase()
        db.sync_source_added(
            "s1", MockAdapter("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8")
        )
        assert db.get_metadata_json("s1") == {
            "test_key": "test_value",
            "nested": {"a": 1, "b": 2},
        }

    def test_raises_on_read_error(self):
        """A DuckDB read failure propagates: the catalog is the mandatory,
        authoritative metadata source (no adapter fallback), so a read error must
        surface as a failed request rather than be masked as 'no metadata'."""
        import pytest

        db = MetadataDatabase()
        db.sync_source_added(
            "s1", MockAdapter("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8")
        )

        class _BoomCursor:
            def execute(self, *a, **k):
                raise RuntimeError("duckdb read boom")

        # Force the read to raise; the method must propagate, not swallow.
        db._get_cursor = lambda: _BoomCursor()
        with pytest.raises(RuntimeError, match="duckdb read boom"):
            db.get_metadata_json("s1")

    def test_none_on_corrupt_json(self):
        """A non-JSON stored value degrades to None rather than propagating."""
        db = MetadataDatabase()
        db.sync_source_added(
            "s1", MockAdapter("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8")
        )
        with db._write_lock:
            db._get_connection().execute(
                "UPDATE sources SET metadata_json = ? WHERE source_id = ?",
                ["{not valid json", "s1"],
            )
        assert db.get_metadata_json("s1") is None

    def test_none_for_empty_metadata(self):
        # MultiTensorAdapter.get_metadata() -> {} -> stored as SQL NULL.
        db = MetadataDatabase()
        db.sync_source_added(
            "s2",
            MultiTensorAdapter(
                "s2",
                "/d/s2.zarr",
                "zarr",
                [
                    {
                        "array_id": "s2",
                        "dim_labels": ["y", "x"],
                        "shape": [4, 4],
                        "chunk_shape": [4, 4],
                        "dtype": "uint8",
                    }
                ],
            ),
        )
        assert db.get_metadata_json("s2") is None

    def test_none_for_absent_source(self):
        db = MetadataDatabase()
        assert db.get_metadata_json("nope") is None


class TestQueryHandling:
    """Test SQL query handling."""

    def test_query_simple(self):
        """Test simple SELECT query."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        table = db.query("SELECT source_id, source_type FROM sources")
        assert table.num_rows == 1
        assert table.schema.metadata is not None
        assert int(table.schema.metadata[b"total_sources"]) == 1

    def test_query_with_filter(self):
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

        result = db.query("SELECT source_id FROM sources WHERE source_type='ome-zarr'")
        assert result.num_rows == 2

    def test_query_json_field(self):
        """Test query using DuckDB JSON operators."""
        db = MetadataDatabase()
        adapter = MockAdapter(
            "test-1", "/data/test.zarr", "ome-zarr", [100, 100], "uint16"
        )
        db.sync_source_added("test-1", adapter)

        result = db.query(
            "SELECT source_id, metadata_json->>'test_key' as test_key FROM sources"
        )
        assert result.num_rows == 1

    def test_query_truncation(self):
        """Truncation is signaled on the table's own schema metadata -- the
        table DoGet streams, so a client reading the stream sees it."""
        db = MetadataDatabase(max_query_results=2)

        # Add 5 sources
        for i in range(5):
            db.sync_source_added(
                f"test-{i}",
                MockAdapter(
                    f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"
                ),
            )

        result = db.query("SELECT source_id FROM sources")
        assert result.num_rows == 2
        md = result.schema.metadata
        assert md[b"truncated"] == b"True"
        assert (md[b"total_rows"], md[b"returned_rows"]) == (b"5", b"2")
        assert int(md[b"total_sources"]) == 5

    def test_table_schema_names_a_public_table_only(self):
        db = MetadataDatabase()
        assert db.table_schema("sources").names[:2] == ["source_id", "source_url"]
        with pytest.raises(ValueError, match="unknown catalog table"):
            db.table_schema("rois")


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

    def test_validate_keyword_inside_literal_or_identifier_allowed(self):
        """Forbidden keywords appearing inside string literals or as substrings
        of identifiers must not trip the denylist (regression: substring match
        rejected legitimate queries).
        """
        db = MetadataDatabase()

        # Keyword as a substring of a string literal
        db._validate_query(
            "SELECT source_id FROM sources WHERE source_url LIKE '%/uploads/%'"
        )
        db._validate_query(
            "SELECT source_id FROM sources WHERE metadata_json LIKE '%update%'"
        )
        db._validate_query(
            "SELECT source_id FROM sources WHERE metadata_json LIKE '%dropdown%'"
        )
        # Escaped single quote inside the literal
        db._validate_query(
            "SELECT source_id FROM sources WHERE source_url LIKE '%don''t delete%'"
        )
        # A real trailing statement is still blocked even after a benign literal
        with pytest.raises(ValueError, match="forbidden keyword"):
            db._validate_query(
                "SELECT source_id FROM sources WHERE source_url LIKE '%ok%'; "
                "DROP TABLE sources"
            )

    @pytest.mark.parametrize(
        "sql",
        [
            'SELECT * FROM "rois"',
            "SELECT * FROM 'rois'",
            "SELECT * FROM main.rois",
            "SELECT r.* FROM sources s, rois r",
            "SELECT * FROM sources s JOIN rois r ON true",
            "WITH x AS (SELECT * FROM rois) SELECT * FROM x",
            "SELECT * FROM sources WHERE source_id IN (SELECT roi_id FROM rois)",
            "SELECT (SELECT count(*) FROM rois)",
            "FROM rois",
        ],
    )
    def test_every_way_of_naming_a_private_table_is_refused(self, sql):
        """The guard walks DuckDB's parse, not the text: quoting, schema
        qualification, comma joins, CTEs and subqueries all resolve to the
        table they read (biopb/biopb#1010)."""
        db = MetadataDatabase()
        with pytest.raises(ValueError, match="disallowed table: rois"):
            db._validate_query(sql)

    def test_describe_and_show_are_refused(self):
        db = MetadataDatabase()
        for sql in ("DESCRIBE rois", "SUMMARIZE sources", "SHOW TABLES"):
            with pytest.raises(ValueError, match="not available here"):
                db._validate_query(sql)

    def test_a_cte_may_shadow_nothing_but_its_own_name(self):
        db = MetadataDatabase()
        db._validate_query("WITH x AS (SELECT * FROM sources) SELECT * FROM x")
        db._validate_query(
            "SELECT source_id, t.array_id FROM sources, UNNEST(tensors) AS u(t)"
        )
        db._validate_query("SELECT * FROM sources LIMIT 1 ;  ")
        with pytest.raises(
            ValueError, match="disallowed table function: duckdb_tables"
        ):
            db._validate_query("SELECT * FROM duckdb_tables()")

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

        # Refused by the validator as a table function outside the allowlist...
        with pytest.raises(ValueError, match="disallowed table function: read_text"):
            db._validate_query(f"SELECT * FROM sources, read_text('{secret}')")
        # ...and, behind that, execution is blocked by enable_external_access=False.
        with pytest.raises(duckdb.Error, match="file system operations are disabled"):
            db._get_cursor().execute(f"SELECT content FROM read_text('{secret}')")

    def test_set_external_access_cannot_be_reenabled(self):
        """An attacker can't turn external access back on mid-query."""
        db = MetadataDatabase()
        db.sync_source_added(
            "z1",
            MockAdapter("z1", "/data/z1.zarr", "ome-zarr", [10, 10], "uint16"),
        )
        with pytest.raises(ValueError):
            db.query("SET enable_external_access=true; SELECT * FROM sources")


class TestQueryMetadata:
    """The stream carries the counts a client sizes the browse surface by."""

    def test_query_metadata(self):
        db = MetadataDatabase()

        for i in range(3):
            db.sync_source_added(
                f"test-{i}",
                MockAdapter(
                    f"test-{i}", f"/data/test{i}.zarr", "ome-zarr", [100, 100], "uint16"
                ),
            )

        md = db.query("SELECT source_id FROM sources").schema.metadata
        assert md is not None
        assert b"total_sources" in md
        assert b"returned_rows" in md
        assert b"query_elapsed_ms" in md


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
