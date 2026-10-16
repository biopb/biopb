"""Where each source sits in the catalog tree (``source_tree``), kept beside the
row, and the generation that names the catalog's content."""

import inspect
import re

import duckdb
import pytest
from biopb_tensor_server.core.discovery import SourceClaim
from biopb_tensor_server.serving.metadata_db import (
    CatalogRecord,
    MetadataDatabase,
    MirroredRow,
)
from biopb_tensor_server.serving.tree_paths import (
    SEP,
    catalog_url,
    dec,
    enc,
    tree_columns,
    tree_parts,
)

from tests.test_metadata_db import MockAdapter


def _add(db, source_id, url, record=None):
    db.sync_source_added(
        source_id, MockAdapter(source_id, url, "zarr", [4, 4], "uint8"), record
    )


def _place(db, source_id):
    (row,) = db.query(
        f"SELECT dir, name FROM source_tree WHERE source_id = '{source_id}'"
    ).to_pylist()
    return row["dir"], row["name"]


class TestTreeColumns:
    def test_a_listing_takes_its_place_from_its_url(self):
        db = MetadataDatabase()
        _add(db, "a", "file:///data/exp/a.zarr")
        assert _place(db, "a") == (f"data{SEP}exp", "a.zarr")

    def test_a_row_under_a_root_takes_the_roots_url_and_its_own_path(self):
        db = MetadataDatabase()
        db.sync_roots([("r", "file:///lab")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "plate/a.zarr"))
        assert _place(db, "a") == (f"lab{SEP}plate", "a.zarr")

    def test_an_alias_root_is_its_own_prefix(self):
        db = MetadataDatabase()
        db.sync_roots([("r", "scope/room1")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "a.zarr"))
        assert _place(db, "a") == (f"scope{SEP}room1", "a.zarr")

    def test_a_mirrored_row_roots_at_its_endpoint(self):
        db = MetadataDatabase()
        db.ensure_root("m", "grpc://lab")
        db.sync_mirrored_rows(
            "m", "tensor-server", [MirroredRow("lab__a", "data/a.tif", None, True, [])]
        )
        assert _place(db, "lab__a") == (f"grpc\x02lab{SEP}data", "a.tif")

    def test_a_source_with_no_path_has_neither_folder_nor_name(self):
        db = MetadataDatabase()
        _add(db, "a", "")
        assert _place(db, "a") == ("", "")

    def test_an_alias_edited_moves_every_row_under_the_root(self):
        db = MetadataDatabase()
        db.sync_roots([("r", "old/place")])
        for i in range(3):
            _add(db, f"s{i}", "x", CatalogRecord(None, {}, "r", f"d/s{i}.zarr"))

        db.sync_roots([("r", "new")])

        assert {_place(db, f"s{i}") for i in range(3)} == {
            (f"new{SEP}d", f"s{i}.zarr") for i in range(3)
        }

    def test_a_root_that_arrives_after_its_rows_places_them(self):
        db = MetadataDatabase()
        _add(db, "a", "x", CatalogRecord(None, {}, "late", "d/a.zarr"))
        db.ensure_root("late", "dnd://lab")
        assert _place(db, "a") == (f"lab{SEP}d", "a.zarr")

    def test_a_drag_dropped_root_is_marked_at_the_top(self):
        db = MetadataDatabase()
        db.sync_roots([("r", "file:///lab")])
        db.ensure_root("d", "dnd://dropped")
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "a.zarr"))
        _add(db, "b", "x", CatalogRecord(None, {}, "d", "b.zarr"))
        _add(db, "c", "grpc://host:1/up/c.zarr")

        folders = {f["name"]: f for f in db.catalog_children("")["folders"]}

        assert folders["dropped"].get("dropped") is True
        # Every other root, an upstream's endpoint included, is left unmarked: the
        # field is there only when true, so those levels are what they were.
        assert "dropped" not in folders["lab"]
        assert "dropped" not in folders["grpc://host:1/up"]

    def test_only_the_top_is_marked(self):
        db = MetadataDatabase()
        db.ensure_root("d", "dnd://lab")
        _add(db, "a", "x", CatalogRecord(None, {}, "d", "one/a.zarr"))
        _add(db, "b", "x", CatalogRecord(None, {}, "d", "two/b.zarr"))

        assert db.catalog_children("")["folders"] == [
            {"name": "lab", "key": "lab", "dropped": True}
        ]
        assert db.catalog_children("lab")["folders"] == [
            {"name": "one", "key": "lab/one"},
            {"name": "two", "key": "lab/two"},
        ]

    def test_a_relisted_internal_row_follows_its_new_url(self):
        db = MetadataDatabase()
        _add(db, "a", "file:///one/a.zarr")
        _add(db, "a", "file:///two/a.zarr")
        assert _place(db, "a") == ("two", "a.zarr")

    def test_a_relisted_row_under_a_root_keeps_its_place(self):
        db = MetadataDatabase()
        db.sync_roots([("r", "file:///lab")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "d/a.zarr"))
        _add(db, "a", "file:///elsewhere/a.zarr")
        assert _place(db, "a") == (f"lab{SEP}d", "a.zarr")

    def test_rows_reattributed_to_another_root_are_replaced(self):
        db = MetadataDatabase()
        db.sync_roots([("r1", "file:///one"), ("r2", "file:///two")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r1", "a.zarr"))
        db.rewrite_row_roots([("a", "r2", "sub/a.zarr")])
        assert _place(db, "a") == (f"two{SEP}sub", "a.zarr")

    def test_it_shows_the_rows_sources_shows(self):
        db = MetadataDatabase()
        db.sync_roots([("r", "file:///lab")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "a.zarr"))
        assert db.query("SELECT source_id FROM source_tree").num_rows == 1
        db.sync_roots([])
        assert db.query("SELECT source_id FROM sources").num_rows == 0
        assert db.query("SELECT source_id FROM source_tree").num_rows == 0

    def test_a_resolved_source_with_no_tensor_is_not_visible_and_a_pending_one_is(self):
        db = MetadataDatabase()
        _add(db, "a", "file:///d/a.zarr")
        db._get_connection().execute(
            "UPDATE source_catalog SET tensors = [] WHERE source_id = 'a'"
        )
        url = "file:///d/p.zarr"
        db.sync_pending_source(SourceClaim("zarr", url, "p"), url)
        visible = {
            r["source_id"]: r["visible"]
            for r in db.query("SELECT source_id, visible FROM source_tree").to_pylist()
        }
        assert visible == {"a": False, "p": True}

    def test_a_catalog_from_before_the_columns_keeps_its_rows(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.open()
        db.sync_roots([("r", "file:///lab")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "d/a.zarr"))
        db.close()

        raw = duckdb.connect(str(store))
        for view in ("source_tree", "sources", "source_confirmation"):
            raw.execute(f"DROP VIEW {view}")
        raw.execute("ALTER TABLE source_catalog DROP COLUMN tree_dir")
        raw.execute("ALTER TABLE source_catalog DROP COLUMN tree_name")
        raw.close()

        db = MetadataDatabase(store_path=store)
        db.open()
        try:
            assert db.query("SELECT source_id FROM sources").num_rows == 1
            assert _place(db, "a") == (f"lab{SEP}d", "a.zarr")
        finally:
            db.close()

    def test_an_unfinished_fill_is_finished_at_the_next_open(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.open()
        db.sync_roots([("r", "file:///lab")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "d/a.zarr"))
        db.close()

        # The columns exist but a crash left them unfilled.
        raw = duckdb.connect(str(store))
        raw.execute("UPDATE source_catalog SET tree_dir = '', tree_name = ''")
        raw.close()

        db = MetadataDatabase(store_path=store)
        db.open()
        try:
            assert _place(db, "a") == (f"lab{SEP}d", "a.zarr")
        finally:
            db.close()


class TestTreePaths:
    def test_only_a_leading_endpoint_is_one(self):
        assert enc("grpc://h:1/a/b") == "grpc\x02h:1/a/b"
        assert enc("data/http://x/img") == "data/http://x/img"
        # A later "://" collapses like any "//": it never reaches a segment.
        assert tree_parts("dnd://data/http://x/img.tif") == [
            "data",
            "http:",
            "x",
            "img.tif",
        ]

    def test_catalog_url_is_the_views_source_url(self):
        assert catalog_url("", "a/b") == "a/b"
        assert catalog_url("file:///r", ".") == "file:///r"
        assert catalog_url("file:///r", "a/b") == "file:///r/a/b"

    def test_columns_split_the_last_segment_off(self):
        assert tree_columns("", "file:///a/b/c.zarr") == (f"a{SEP}b", "c.zarr")
        assert tree_columns("", "file:///c.zarr") == ("", "c.zarr")
        assert tree_columns("", "") == ("", "")

    def test_an_endpoint_is_one_segment(self):
        assert tree_parts("grpc://h:1/a") == ["grpc\x02h:1", "a"]
        assert dec(tree_parts("grpc://h:1/a")[0]) == "grpc://h:1"


def _writers():
    """Each way a test can change what the catalog shows, as ``(name, fn(db))``."""

    def seeded():
        db = MetadataDatabase()
        db.sync_roots([("r", "file:///lab")])
        _add(db, "a", "x", CatalogRecord(None, {}, "r", "a.zarr"))
        return db

    def pending(db):
        url = "file:///d/p.zarr"
        db.sync_pending_source(SourceClaim("zarr", url, "p"), url)

    return seeded, [
        ("added", lambda db: _add(db, "b", "file:///d/b.zarr")),
        ("pending one", pending),
        ("removed", lambda db: db.sync_source_removed("a")),
        (
            "mirrored rows",
            lambda db: (
                db.ensure_root("m", "grpc://lab"),
                db.sync_mirrored_rows(
                    "m", "tensor-server", [MirroredRow("m1", "a.tif", None, True, [])]
                ),
            ),
        ),
        ("mirrored removed", lambda db: db.sync_mirrored_removed(["a"])),
        ("dropped rows", lambda db: db.drop_catalog_rows(["a"])),
        ("rewritten roots", lambda db: db.rewrite_row_roots([("a", "r", "b.zarr")])),
        ("synced roots", lambda db: db.sync_roots([("r", "file:///other")])),
        ("ensured root", lambda db: db.ensure_root("e", "dnd://x")),
    ]


class TestGeneration:
    @pytest.mark.parametrize("name", [n for n, _ in _writers()[1]])
    def test_a_write_advances_it(self, name):
        seeded, writers = _writers()
        db = seeded()
        before = db.catalog_generation
        dict(writers)[name](db)
        assert db.catalog_generation != before

    def test_a_read_and_a_walk_confirmation_do_not(self):
        seeded, _ = _writers()
        db = seeded()
        before = db.catalog_generation
        db.query("SELECT source_id FROM sources")
        db.confirm_root("r")
        assert db.catalog_generation == before

    def test_every_writer_of_the_catalog_says_so(self):
        """A writer that forgot would leave a 304 answering for stale content, so
        the ones in this class are found by what they run."""
        runs = re.compile(
            r"(INSERT (OR REPLACE )?INTO|UPDATE|DELETE FROM)\s+"
            r"(source_catalog|catalog_roots)\b"
        )
        # Open-time work, a walk's bookkeeping column, and helpers whose callers
        # advance it.
        exempt = {
            "_create_source_catalog",
            "confirm_root",
            "_retree_root",
            "_upsert_root",
            "_pending_inserts",
        }
        forgot = [
            name
            for name, fn in inspect.getmembers(MetadataDatabase, inspect.isfunction)
            if name not in exempt
            and runs.search(inspect.getsource(fn))
            and "_catalog_written(" not in inspect.getsource(fn)
            and "_catalog_txn(" not in inspect.getsource(fn)
        ]
        assert forgot == []

    def test_it_never_repeats_across_a_restart(self, tmp_path):
        store = tmp_path / "catalog.duckdb"
        seen = []
        for _ in range(2):
            db = MetadataDatabase(store_path=store)
            db.open()
            seen.append(db.catalog_generation)
            db.close()
        assert seen[0] != seen[1]

    def test_a_query_carries_it(self):
        db = MetadataDatabase()
        table = db.query("SELECT 1 AS one")
        assert table.schema.metadata[b"catalog_generation"] == (
            db.catalog_generation.encode()
        )

    def test_the_source_count_is_taken_once_per_generation(self):
        db = MetadataDatabase()
        _add(db, "a", "file:///d/a.zarr")
        counted = db.query("SELECT 1").schema.metadata[b"total_sources"]
        # A row written around the writers: the count stands until a write says
        # the catalog changed.
        db._get_connection().execute("DELETE FROM source_catalog")
        assert db.query("SELECT 1").schema.metadata[b"total_sources"] == counted
        db._catalog_written()
        assert db.query("SELECT 1").schema.metadata[b"total_sources"] == b"0"
