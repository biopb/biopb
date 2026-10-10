"""Every source sits under a root: a configured one, a drop, an upstream, or the
catalog's built-in one. The roots that are not persisted go at the next open with
the rows under them."""

from pathlib import Path

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import DiscoveryState, SourceClaim
from biopb_tensor_server.serving.metadata_db import (
    INTERNAL_ROOT_ID,
    CatalogRecord,
    MetadataDatabase,
)
from biopb_tensor_server.sources.roots import Root, RootKind

from tests import catalog_server, make_manager
from tests.test_metadata_db import MockAdapter


def _manager(root):
    server = catalog_server("localhost:0")
    return make_manager(
        server=server,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        metadata_db=server.metadata_db,
        monitored_dirs={root},
        stability_window=0,
    )


def _roots(db):
    return {
        r[0]: (r[1], r[2])
        for r in db._get_connection()
        .execute("SELECT root_id, root_url, persisted FROM catalog_roots")
        .fetchall()
    }


class TestRecordsOfSourcesThatAreNotWalked:
    def test_a_drop_has_its_root_in_the_catalog(self, tmp_path):
        manager = _manager(tmp_path / "root")
        elsewhere = Path(tmp_path) / "dropped"
        root = Root(RootKind.DROPPED, str(elsewhere), label="x")
        manager._roots.add(root)

        manager._reconciler._catalog_record(
            SourceClaim("zarr", str(elsewhere / "a.zarr"), "x")
        )

        db = manager._reconciler._metadata_db
        assert _roots(db)[root.root_id] == ("dnd://x", False)


class TestTheViewAndTheOpen:
    def _drop_row(self, db):
        root = Root(RootKind.DROPPED, "/dropped/exp", label="exp")
        db.ensure_root(root.root_id, root.root_url)
        db.sync_source_added(
            "d1",
            MockAdapter("d1", "/dropped/exp/sub/a.zarr", "zarr", [4, 4], "uint8"),
            CatalogRecord(None, {}, root.root_id, "sub/a.zarr"),
        )
        return root

    def test_a_row_under_a_drop_shows_its_roots_url(self):
        db = MetadataDatabase()
        self._drop_row(db)

        assert db.query("SELECT source_url FROM sources").to_pylist() == [
            {"source_url": "dnd://exp/sub/a.zarr"}
        ]

    def test_a_row_under_a_root_that_is_not_persisted_is_confirmed(self):
        db = MetadataDatabase()
        self._drop_row(db)

        confirmed = (
            db._get_connection()
            .execute("SELECT confirmed FROM source_confirmation")
            .fetchall()
        )
        assert confirmed == [(True,)]

    def test_an_open_drops_the_roots_that_are_not_persisted_and_their_rows(
        self, tmp_path
    ):
        store = tmp_path / "catalog.duckdb"
        db = MetadataDatabase(store_path=store)
        db.open()
        db.sync_roots([("r1", "file:///d")])
        drop = self._drop_row(db)
        db.sync_source_added(
            "api", MockAdapter("api", "/x/api.zarr", "zarr", [4, 4], "uint8")
        )
        assert drop.root_id in _roots(db)
        db.close()

        db = MetadataDatabase(store_path=store)
        db.open()
        try:
            assert set(_roots(db)) == {"r1", INTERNAL_ROOT_ID}
            assert db.query("SELECT source_id FROM sources").num_rows == 0
        finally:
            db.close()

    def test_syncing_the_configured_roots_leaves_the_others(self):
        db = MetadataDatabase()
        drop = self._drop_row(db)

        db.sync_roots([("r1", "file:///d")])

        assert set(_roots(db)) == {"r1", drop.root_id, INTERNAL_ROOT_ID}
        db.sync_roots([])
        assert set(_roots(db)) == {drop.root_id, INTERNAL_ROOT_ID}
