"""The persisted catalog: ``source_catalog``, the volatile table and the view.

Only written so far; nothing reads the persisted rows back yet. These pin what is
written (routing by whether the source has a claim, the private columns, the
claim-time signature, the payload) and what must keep working (queries through the
view, an older build opening the file).
"""

import json
import os

import duckdb
import numpy as np
import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import DiscoveryState, SourceClaim
from biopb_tensor_server.serving.metadata_db import (
    SOURCE_CATALOG_FORMAT,
    CatalogRecord,
    MetadataDatabase,
)
from biopb_tensor_server.sources.roots import path_under_root

from tests import catalog_server, make_manager
from tests.test_metadata_db import MockAdapter

tifffile = pytest.importorskip("tifffile")


class _Restorable(MockAdapter):
    def catalog_payload(self):
        return {"k": 1}


def _db():
    """A catalog with the one root, ``/d``, that ``_record`` sits under."""
    db = MetadataDatabase()
    db.sync_roots([("r1", "file:///d")])
    return db


def _record(path="/d/s1.zarr", signature=None):
    claim = SourceClaim(
        "zarr", path, "s1", extra_config={"alias": "lab"}, member_paths=[path + "/m"]
    )
    return CatalogRecord(
        claim, signature or {path: (1, 2, 3, 4)}, "r1", path_under_root("/d", path)
    )


def _row(db, source_id="s1"):
    cols = (
        "is_resolved, unresolved_reason, unresolved_error, payload, signature, "
        "primary_path, len(tensors)"
    )
    return (
        db._get_connection()
        .execute(f"SELECT {cols} FROM source_catalog WHERE source_id = ?", [source_id])
        .fetchone()
    )


def _count(db, table, source_id="s1"):
    return (
        db._get_connection()
        .execute(f"SELECT count(*) FROM {table} WHERE source_id = ?", [source_id])
        .fetchone()[0]
    )


class TestRouting:
    def test_a_restorable_source_lands_in_the_persistent_table_only(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        assert (_count(db, "source_catalog"), _count(db, "sources_volatile")) == (1, 0)
        assert db.query("SELECT source_id FROM sources").num_rows == 1

    def test_a_claim_without_a_payload_is_still_persisted(self):
        db = _db()
        db.sync_source_added(
            "s1", MockAdapter("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        assert (_count(db, "source_catalog"), _count(db, "sources_volatile")) == (1, 0)
        (payload,) = (
            db._get_connection()
            .execute("SELECT payload FROM source_catalog")
            .fetchone()
        )
        assert payload is None

    def test_without_a_record_it_stays_volatile(self):
        db = _db()
        db.sync_source_added(
            "s2", _Restorable("s2", "/d/s2.zarr", "zarr", [4, 4], "uint8")
        )
        assert (
            _count(db, "source_catalog", "s2"),
            _count(db, "sources_volatile", "s2"),
        ) == (0, 1)
        assert db.query("SELECT source_id FROM sources").num_rows == 1

    def test_an_unresolved_row_with_a_claim_is_persisted_without_a_payload(self):
        db = _db()
        adapter = _Restorable(
            "s1", "/d/s1.zarr", "zarr", [4, 4], "uint8", is_resolved=False
        )
        db.sync_source_added("s1", adapter, _record())
        assert (_count(db, "source_catalog"), _count(db, "sources_volatile")) == (1, 0)
        assert _row(db)[3] is None

    def test_a_cloud_claim_is_persisted_without_a_payload(self):
        db = _db()
        record = CatalogRecord(
            _record().claim, {"/d/s1.zarr": (1,)}, "r1", "s1.zarr", cloud=True
        )
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), record
        )
        assert _count(db, "source_catalog") == 1
        assert _row(db)[3] is None

    def test_a_pending_row_with_a_claim_is_persisted_and_one_without_is_not(self):
        db = _db()
        db.sync_pending_source(_record().claim, record=_record())
        other = SourceClaim("zarr", "/drop/s2.zarr", "s2")
        db.sync_pending_source(other)
        assert _count(db, "source_catalog") == 1
        assert _count(db, "sources_volatile", "s2") == 1
        row = _row(db)
        assert row[:3] == (False, "pending", None)
        assert json.loads(row[4]) == {"/d/s1.zarr": [1, 2, 3, 4]}
        assert row[5] == "/d/s1.zarr"

    def test_registration_fills_in_the_pending_row_it_finds(self):
        db = _db()
        db.sync_pending_source(_record().claim, record=_record())
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        assert (_count(db, "source_catalog"), _count(db, "sources_volatile")) == (1, 0)
        row = _row(db)
        assert row[:3] == (True, None, None)
        assert json.loads(row[3]) == {"k": 1}
        assert row[6] == 1  # its tensors

    def test_a_pending_batch_splits_by_whether_the_row_has_a_record(self):
        from biopb_tensor_server.sources.pending_rows import PendingRow

        db = _db()
        kept = _record()
        db.sync_pending_sources(
            [
                PendingRow(kept.claim, record=kept),
                PendingRow(SourceClaim("zarr", "/drop/s2.zarr", "s2")),
                PendingRow(SourceClaim("zarr", "/cloud/s3.zarr", "s3"), recall=True),
            ]
        )
        assert _count(db, "source_catalog") == 1
        assert _count(db, "sources_volatile", "s2") == 1
        assert db.query("SELECT source_id FROM sources").num_rows == 3

    def test_a_batched_row_does_not_replace_a_registered_one(self):
        from biopb_tensor_server.sources.pending_rows import PendingRow

        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.sync_pending_sources([PendingRow(_record().claim, record=_record())])
        assert _row(db)[:2] == (True, None)

    def test_every_source_shows_once_across_both_tables(self):
        from biopb_tensor_server.sources.pending_rows import PendingRow

        db = _db()
        restorable = _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8")
        plain = MockAdapter("s2", "/d/s2.zarr", "zarr", [4, 4], "uint8")
        db.sync_source_added("s1", restorable, _record())
        db.sync_source_added("s2", plain)
        # Registered again and pended again: each stays in its own table.
        db.sync_source_added("s1", restorable, _record())
        db.sync_source_added("s2", plain)
        db.sync_pending_sources([PendingRow(_record().claim, record=_record())])
        assert (
            _count(db, "source_catalog", "s1"),
            _count(db, "sources_volatile", "s1"),
            _count(db, "source_catalog", "s2"),
            _count(db, "sources_volatile", "s2"),
        ) == (1, 0, 0, 1)
        rows = db.query("SELECT source_id FROM sources").to_pylist()
        assert sorted(r["source_id"] for r in rows) == ["s1", "s2"]

    def test_a_failed_registration_row_replaces_a_persisted_one(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.sync_pending_source(_record().claim, error="boom", record=_record())
        assert (_count(db, "source_catalog"), _count(db, "sources_volatile")) == (1, 0)
        row = _row(db)
        assert row[:4] == (False, "failed", "boom", None)
        assert row[6] == 0

    def test_removal_deletes_from_both(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.sync_source_added(
            "s2", MockAdapter("s2", "/d/s2.zarr", "zarr", [4, 4], "uint8")
        )
        db.sync_source_removed("s1")
        db.sync_source_removed("s2")
        assert db.query("SELECT source_id FROM sources").num_rows == 0
        assert _count(db, "source_catalog", "s1") == 0


class TestPrivacy:
    def test_the_claim_columns_are_not_in_the_view(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        cols = db.query("SELECT * FROM sources").column_names
        assert not {"extra_config", "member_paths", "payload", "signature"} & set(cols)

    @pytest.mark.parametrize(
        "table", ["source_catalog", "sources_volatile", "catalog_roots"]
    )
    def test_the_physical_tables_are_not_queryable(self, table):
        db = _db()
        with pytest.raises(ValueError):
            db.query(f"SELECT * FROM {table}")


class TestRow:
    def test_the_persisted_row_holds_the_claim_and_the_payload(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        row = (
            db._get_connection()
            .execute(
                "SELECT primary_path, member_paths, extra_config, signature, payload "
                "FROM source_catalog"
            )
            .fetchone()
        )
        assert row[0] == "/d/s1.zarr"
        assert row[1] == ["/d/s1.zarr", "/d/s1.zarr/m"]
        assert json.loads(row[2]) == {"alias": "lab"}
        assert json.loads(row[3]) == {"/d/s1.zarr": [1, 2, 3, 4]}
        assert json.loads(row[4]) == {"k": 1}


class TestRootsInTheTable:
    """``source_url`` is the view's: the root's url, then the row's path beneath it."""

    def test_an_alias_edit_shows_with_no_source_row_written(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        url = "SELECT source_url FROM sources WHERE source_id = 's1'"
        assert db.query(url).to_pylist() == [{"source_url": "file:///d/s1.zarr"}]

        db.sync_roots([("r1", "lab")])

        assert db.query(url).to_pylist() == [{"source_url": "lab/s1.zarr"}]

    def test_a_sweep_drops_only_the_rows_nothing_claims(self):
        db = _db()
        for source_id in ("s1", "s2", "s3"):
            path = f"/d/{source_id}.zarr"
            claim = SourceClaim("zarr", path, source_id, member_paths=[path])
            db.sync_pending_source(
                claim,
                record=CatalogRecord(claim, {path: (1,)}, "r1", f"{source_id}.zarr"),
            )

        swept = db.sweep_root("r1", lambda source_id: source_id != "s2")

        assert swept == 1
        left = db.query("SELECT source_id FROM sources ORDER BY source_id")
        assert [r["source_id"] for r in left.to_pylist()] == ["s1", "s3"]

    def test_a_row_whose_root_is_gone_is_not_shown(self):
        db = _db()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.sync_roots([])
        assert db.query("SELECT source_id FROM sources").num_rows == 0

    def test_the_view_agrees_with_the_roots_for_every_kind_of_root(self, tmp_path):
        import biopb_tensor_server.adapters  # noqa: F401
        from biopb_tensor_server.core.adapter_base import to_catalog_url
        from biopb_tensor_server.core.config import SourceConfig

        from tests import deferred_registration_test as drt

        plain, aliased, once = (tmp_path / n for n in ("plain", "aliased", "once"))
        for d in (plain, aliased, once):
            d.mkdir()
        drt._make_zarr(plain, "a.zarr")
        drt._make_zarr(aliased, "b.zarr")
        sub = aliased / "sub"
        sub.mkdir()
        drt._make_zarr(sub, "c.zarr")
        drt._make_zarr(once, "d.zarr")
        solo = drt._make_zarr(tmp_path, "solo.zarr")
        server = catalog_server("localhost:0")
        manager = make_manager(
            server=server,
            registry=get_default_registry(),
            discovery_state=DiscoveryState(),
            metadata_db=server.metadata_db,
            monitored_dirs={plain, aliased},
            monitored_aliases={aliased: "lab"},
            scan_once_sources=[
                SourceConfig(url=str(once), alias="once-lab"),
                SourceConfig(url=str(solo), alias="solo-lab"),
            ],
            stability_window=0,
        )
        manager._handle_rescan()

        rows = [
            {"source_url": url, "primary_path": path}
            for url, path in server.metadata_db._get_connection()
            .execute(
                "SELECT s.source_url, c.primary_path "
                "FROM sources s JOIN source_catalog c USING (source_id)"
            )
            .fetchall()
        ]
        assert len(rows) == 5
        for row in rows:
            expected = manager._roots.display_url(
                row["primary_path"]
            ) or to_catalog_url(row["primary_path"])
            assert row["source_url"] == expected, row
        assert {r["source_url"] for r in rows} >= {
            "lab/b.zarr",
            "lab/sub/c.zarr",
            "once-lab/d.zarr",
            "solo-lab",
        }


class TestStore:
    def _db(self, path):
        return MetadataDatabase(store_path=path)

    def test_rows_do_not_outlive_the_run_until_something_reads_them(self, tmp_path):
        path = tmp_path / "c.duckdb"
        db = self._db(path)
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.close()
        db = self._db(path)
        assert db.query("SELECT source_id FROM sources").num_rows == 0

    def test_a_format_mismatch_drops_the_table(self, tmp_path):
        path = tmp_path / "c.duckdb"
        db = self._db(path)
        db.open()
        conn = db._get_connection()
        conn.execute("ALTER TABLE source_catalog ADD COLUMN stale_marker INT")
        conn.execute(
            "UPDATE catalog_meta SET value = ? WHERE key = 'source_catalog_format'",
            [str(SOURCE_CATALOG_FORMAT + 1)],
        )
        db.close()
        db = self._db(path)
        db.open()
        cols = [
            r[0]
            for r in db._get_connection()
            .execute(
                "SELECT column_name FROM duckdb_columns() "
                "WHERE table_name = 'source_catalog'"
            )
            .fetchall()
        ]
        assert "stale_marker" not in cols

    def test_a_file_with_an_older_physical_sources_table_opens(self, tmp_path):
        path = tmp_path / "c.duckdb"
        conn = duckdb.connect(str(path))
        conn.execute("CREATE TABLE sources (source_id TEXT PRIMARY KEY)")
        conn.close()
        db = self._db(path)
        db.sync_source_added(
            "s1", MockAdapter("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8")
        )
        assert db.query("SELECT source_id FROM sources").num_rows == 1

    def test_an_older_build_cannot_open_the_file_until_the_view_is_dropped(
        self, tmp_path
    ):
        path = tmp_path / "c.duckdb"
        db = self._db(path)
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.close()
        # What the previous build runs at open: it fails on the view, and
        # dropping the view is the whole downgrade step (the file also holds
        # annotations, so deleting it is not an option).
        conn = duckdb.connect(str(path))
        with pytest.raises(duckdb.CatalogException):
            conn.execute("DROP TABLE IF EXISTS sources")
        conn.execute("DROP VIEW sources")
        conn.execute("DROP TABLE IF EXISTS sources")
        conn.close()

    def test_a_cursor_sees_the_view(self):
        db = _db()
        db.sync_source_added(
            "s1", MockAdapter("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8")
        )
        assert (
            db._get_cursor().execute("SELECT count(*) FROM sources").fetchone()[0] == 1
        )


def _ome_tiff(parent, name="a.ome.tiff"):
    path = os.path.join(parent, name)
    tifffile.imwrite(
        path, np.zeros((2, 16, 16), dtype="uint16"), ome=True, metadata={"axes": "ZYX"}
    )
    return path


def _manager(root):
    server = catalog_server("localhost:0")
    manager = make_manager(
        server=server,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        metadata_db=server.metadata_db,
        monitored_dirs={root},
        stability_window=0,
        registration_workers=0,
    )
    return manager, server


def _persisted(server):
    return (
        server.metadata_db._get_connection()
        .execute(
            "SELECT source_id, primary_path, signature, payload FROM source_catalog"
        )
        .fetchall()
    )


class TestRegistration:
    def test_an_ome_tiff_is_persisted_with_its_signature_and_payload(self, tmp_path):
        path = _ome_tiff(tmp_path)
        manager, server = _manager(tmp_path)
        manager._handle_rescan()

        ((source_id, primary, signature, payload),) = _persisted(server)
        st = os.stat(path)
        assert primary == path
        # No st_dev: it renumbers across boots.
        assert json.loads(signature) == {
            path: [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        }
        (scene,) = json.loads(payload)["scenes"]
        # Golden: a change to this shape or its meaning needs a
        # SOURCE_CATALOG_FORMAT bump.
        assert set(scene) == {"array_id", "dim_labels", "shape", "chunk_shape", "dtype"}
        assert scene["array_id"].startswith(source_id + "/")
        assert scene["dim_labels"] == ["T", "C", "Z", "Y", "X"]
        assert scene["shape"] == [1, 1, 2, 16, 16]
        # A file this small is one transfer chunk.
        assert scene["chunk_shape"] == [1, 1, 2, 16, 16]
        assert scene["dtype"] == "<u2"

    def test_the_payload_matches_the_descriptors_the_row_lists(self, tmp_path):
        _ome_tiff(tmp_path)
        manager, server = _manager(tmp_path)
        manager._handle_rescan()
        ((_, _, _, payload),) = _persisted(server)
        (scene,) = json.loads(payload)["scenes"]
        (row,) = server.metadata_db.query("SELECT tensors FROM sources").to_pylist()
        (tensor,) = row["tensors"]
        assert tensor["array_id"] == scene["array_id"]
        assert tensor["shape"] == scene["shape"]
        assert tensor["dim_labels"] == scene["dim_labels"]
        assert tensor["dtype"] == scene["dtype"]

    def test_a_file_changed_during_the_parse_keeps_its_claim_time_signature(
        self, tmp_path, monkeypatch
    ):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        path = _ome_tiff(tmp_path)
        before = os.stat(path)
        real = OmeTiffAdapter._scene_descriptors

        def touching(self):
            result = real(self)
            with open(path, "ab") as f:  # the file moves under the parse
                f.write(b"\0")
            return result

        monkeypatch.setattr(OmeTiffAdapter, "_scene_descriptors", touching)
        manager, server = _manager(tmp_path)
        manager._handle_rescan()

        ((_, _, signature, _),) = _persisted(server)
        stored = json.loads(signature)[path]
        assert stored[1] == before.st_size
        assert stored != [
            os.stat(path).st_ino,
            os.stat(path).st_size,
            os.stat(path).st_mtime_ns,
            os.stat(path).st_ctime_ns,
        ]

    def test_a_masked_file_is_persisted_with_its_label_tensor(self, tmp_path):
        from tests import ome_mask_test

        raw = np.packbits(np.eye(4, dtype=np.uint8).flatten()).tobytes()
        ome_mask_test.TestFastMetadataRealBitmap()._write(tmp_path, raw)
        manager, server = _manager(tmp_path)
        manager._handle_rescan()

        ((source_id, _, _, payload),) = _persisted(server)
        payload = json.loads(payload)
        assert payload["has_rois"] is True
        (mask,) = payload["masks"]
        assert mask["field"] == "Image:0/@labels/@ome"
        assert mask["parent_array_id"] == f"{source_id}/Image:0"
        # The catalog lists the label tensor the payload describes.
        (row,) = server.metadata_db.query("SELECT tensors FROM sources").to_pylist()
        listed = {t["array_id"]: t for t in row["tensors"]}
        label = listed[f"{source_id}/{mask['field']}"]
        assert label["shape"] == mask["shape"]
        assert label["dim_labels"] == mask["dim_labels"]

    def test_a_source_without_a_payload_is_persisted_from_its_claim(
        self, tmp_path, monkeypatch
    ):
        import zarr
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        monkeypatch.setattr(ZarrAdapter, "catalog_payload", lambda self: None)

        z = zarr.open_array(
            os.path.join(tmp_path, "a.zarr"),
            mode="w",
            shape=(2, 8, 8),
            chunks=(1, 8, 8),
            dtype="uint16",
        )
        z[:] = 1
        manager, server = _manager(tmp_path)
        manager._handle_rescan()
        ((_, primary, signature, payload),) = _persisted(server)
        assert primary.endswith("a.zarr")
        assert json.loads(signature)  # a directory is stamped too
        assert payload is None
        assert server.metadata_db.query("SELECT source_id FROM sources").num_rows == 1


class TestListFlights:
    def test_the_sources_flight_has_the_published_schema_and_ticket(self):
        server = catalog_server("localhost:0")
        infos = {
            info.descriptor.path[0].decode(): info
            for info in server.list_flights(None, b"")
        }
        sources = infos["sources"]
        assert sources.schema.names == [
            "source_id",
            "source_url",
            "source_type",
            "indexed_at",
            "metadata_json",
            "is_resolved",
            "unresolved_reason",
            "unresolved_error",
            "tensors",
        ]
        assert sources.endpoints[0].ticket.ticket
        assert "source_catalog" not in infos
        assert "sources_volatile" not in infos
