"""Restoring the last run's sources (``catalog.restore``).

Two runs share one catalog file. The second restores what the first persisted, and
its first walk is a rescan against those claims.
"""

import os
import shutil

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import DiscoveryState
from biopb_tensor_server.core.errors import SourceRegistrationError
from biopb_tensor_server.serving.metadata_db import MetadataDatabase

from tests import catalog_server, deferred_registration_test as drt, make_manager


class _Run:
    """One server run over a catalog file, driven without the loop thread."""

    def __init__(self, tmp_path, *, restore=True, aliases=None, once=()):
        self.db = MetadataDatabase(
            store_path=tmp_path / "catalog.duckdb", restore_sources=restore
        )
        self.db.open()
        self.server = catalog_server("localhost:0", metadata_db=self.db)
        self.monitored = tmp_path / "monitored"
        self.monitored.mkdir(exist_ok=True)
        self.manager = make_manager(
            server=self.server,
            registry=get_default_registry(),
            discovery_state=DiscoveryState(),
            metadata_db=self.db,
            monitored_dirs={self.monitored},
            monitored_aliases={self.monitored: aliases} if aliases else None,
            scan_once_sources=[SourceConfig(url=str(p)) for p in once],
            stability_window=0,
            registration_workers=0,
        )
        self.reconciler = self.manager._reconciler

    def restore(self):
        self.manager._restore_catalog()

    def first_scan(self):
        self.reconciler.set_defer_registration(True)
        self.manager._handle_rescan()

    def rows(self):
        return drt._rows(self.server)

    def urls(self):
        table = self.db.query("SELECT source_id, source_url FROM sources")
        return {r["source_id"]: r["source_url"] for r in table.to_pylist()}

    def stop(self):
        self.db.close()


def _first_run(tmp_path, names=("a.zarr", "b.zarr"), **kwargs):
    run = _Run(tmp_path, **kwargs)
    for name in names:
        if not os.path.exists(run.monitored / name):
            drt._make_zarr(run.monitored, name)
    run.manager._handle_rescan()
    ids = sorted(run.rows())
    run.stop()
    return ids


class TestRestore:
    def test_the_sources_list_before_any_scan_and_register_on_a_read(self, tmp_path):
        ids = _first_run(tmp_path)
        assert len(ids) == 2

        run = _Run(tmp_path)
        run.restore()

        assert sorted(run.rows()) == ids
        assert all(run.rows()[i]["is_resolved"] for i in ids)
        assert all(run.reconciler.is_pending(i) for i in ids)
        assert run.server.sources.get(ids[0]) is None

        adapter = run.server.sources.get_registered(ids[0])  # hydrates on a read

        assert adapter is not None
        assert not run.reconciler.is_pending(ids[0])
        assert run.reconciler.is_pending(ids[1])
        run.stop()

    def test_the_first_walk_leaves_what_it_finds_unchanged(self, tmp_path, monkeypatch):
        ids = _first_run(tmp_path)
        run = _Run(tmp_path)
        run.restore()
        refreshed = []
        monkeypatch.setattr(
            run.reconciler, "_refresh_claim", lambda c: refreshed.append(c) or True
        )

        run.first_scan()

        assert refreshed == []
        assert sorted(run.rows()) == ids
        # Still the rows the last run wrote: resolved, with their tensors.
        assert all(r["is_resolved"] and r["tensors"] for r in run.rows().values())
        run.stop()

    def test_a_file_changed_while_the_server_was_down_is_refreshed(self, tmp_path):
        ids = _first_run(tmp_path)
        drt._make_zarr(tmp_path / "monitored", "a.zarr", shape=(3, 8, 8))
        run = _Run(tmp_path)
        run.restore()

        run.first_scan()

        (a,) = [
            i for i in ids if run.reconciler.claim_primary_path(i).endswith("a.zarr")
        ]
        assert run.server.sources.get(a) is not None  # rebuilt against the new bytes
        assert run.rows()[a]["tensors"] >= 1
        run.stop()

    def test_a_file_removed_while_the_server_was_down_goes_after_two_walks(
        self, tmp_path
    ):
        ids = _first_run(tmp_path)
        shutil.rmtree(tmp_path / "monitored" / "b.zarr")
        run = _Run(tmp_path)
        run.restore()

        run.first_scan()
        assert len(run.rows()) == 2  # one miss is not enough
        run.manager._handle_rescan()

        assert list(run.rows()) == [i for i in ids if i in run.rows()]
        assert len(run.rows()) == 1
        run.stop()

    def test_a_removed_root_takes_its_sources_with_it(self, tmp_path):
        once = tmp_path / "once"
        once.mkdir()
        drt._make_zarr(once, "o.zarr")
        first = _Run(tmp_path, once=[once])
        drt._make_zarr(first.monitored, "m.zarr")
        first.manager._handle_rescan()
        assert len(first.rows()) == 2
        first.stop()

        run = _Run(tmp_path)  # the scan-once root is no longer configured
        run.restore()

        assert len(run.rows()) == 1
        (only,) = run.rows()
        assert run.reconciler.claim_primary_path(only).endswith("m.zarr")
        run.stop()

    def test_an_alias_edited_between_runs_shows_in_the_url(self, tmp_path):
        ids = _first_run(tmp_path, aliases="old")
        run = _Run(tmp_path, aliases="new")
        run.restore()

        assert {u.split("/")[0] for u in run.urls().values()} == {"new"}
        assert len(run.urls()) == len(ids)
        run.stop()

    def test_a_failed_row_comes_back_failed_and_is_not_retried(self, tmp_path):
        # A failed row, as a registration that raised leaves it.
        first = _Run(tmp_path, restore=False)
        drt._make_zarr(first.monitored, "a.zarr")
        first.reconciler.set_defer_registration(True)
        first.manager._handle_rescan()
        (sid,) = first.rows()
        claim = first.reconciler._state.claims[sid]
        first.db.sync_pending_source(
            claim,
            error="boom",
            record=first.reconciler._state_record(claim),
        )
        first.stop()

        run = _Run(tmp_path)
        run.restore()

        assert run.rows()[sid]["unresolved_reason"] == "failed"
        with pytest.raises(SourceRegistrationError, match="boom"):
            run.server.sources.get_registered(sid)
        run.stop()

    def test_a_read_of_a_source_that_vanished_fails_it_instead_of_listing_it(
        self, tmp_path
    ):
        ids = _first_run(tmp_path)
        shutil.rmtree(tmp_path / "monitored" / "a.zarr")
        run = _Run(tmp_path)
        run.restore()
        (gone,) = [
            i for i in ids if run.reconciler.claim_primary_path(i).endswith("a.zarr")
        ]

        with pytest.raises(SourceRegistrationError):
            run.server.sources.get_registered(gone)

        assert run.rows()[gone]["unresolved_reason"] == "failed"
        assert not run.rows()[gone]["is_resolved"]
        run.stop()

    def test_nothing_is_restored_when_the_setting_is_off(self, tmp_path):
        _first_run(tmp_path)
        run = _Run(tmp_path, restore=False)
        run.restore()
        assert run.rows() == {}
        assert run.reconciler.claim_ids() == []
        run.stop()

    def test_an_unreadable_row_is_dropped_and_found_again(self, tmp_path):
        ids = _first_run(tmp_path)
        db = MetadataDatabase(
            store_path=tmp_path / "catalog.duckdb", restore_sources=True
        )
        db.open()
        db._get_connection().execute(
            "UPDATE source_catalog SET signature = 'not json' WHERE source_id = ?",
            [ids[0]],
        )
        db.close()

        run = _Run(tmp_path)
        run.restore()
        assert list(run.rows()) == [ids[1]]

        run.first_scan()
        assert sorted(run.rows()) == ids
        run.stop()


class TestHydrateFromPayload:
    @pytest.fixture
    def probes(self, monkeypatch):
        """An nd2 file that cannot be told from a real one, and a count of how many
        times it was probed."""
        pytest.importorskip("nd2")
        from biopb_tensor_server.adapters import nd2 as nd2_module

        from tests.nd2_adapter_test import _install_fake

        _install_fake(monkeypatch)
        calls = []
        real = nd2_module.read_layout

        def counting(path):
            calls.append(path)
            return real(path)

        monkeypatch.setattr(nd2_module, "read_layout", counting)
        return calls

    def _first_run(self, tmp_path):
        run = _Run(tmp_path)
        (run.monitored / "img.nd2").write_bytes(b"\x00")
        run.manager._handle_rescan()
        (sid,) = run.rows()
        tensors = run.server.sources.get(sid).list_tensor_descriptors()
        run.stop()
        return sid, [t.array_id for t in tensors]

    def test_a_restored_nd2_is_rebuilt_without_probing_the_file(self, tmp_path, probes):
        sid, array_ids = self._first_run(tmp_path)
        assert len(probes) == 1

        run = _Run(tmp_path)
        run.restore()
        adapter = run.server.sources.get_registered(sid)

        assert len(probes) == 1  # the layout came from the row
        assert [t.array_id for t in adapter.list_tensor_descriptors()] == array_ids
        run.stop()

    def test_a_file_changed_while_down_is_probed_not_rebuilt(self, tmp_path, probes):
        sid, array_ids = self._first_run(tmp_path)
        (tmp_path / "monitored" / "img.nd2").write_bytes(b"\x00\x00\x00")

        run = _Run(tmp_path)
        run.restore()
        run.server.sources.get_registered(sid)

        assert len(probes) == 2
        run.stop()


def _confirmed(run):
    rows = (
        run.db._get_connection()
        .execute("SELECT source_id, confirmed FROM source_confirmation")
        .fetchall()
    )
    return dict(rows)


class TestConfirmation:
    def test_a_restored_row_is_confirmed_when_its_root_has_been_walked(self, tmp_path):
        ids = _first_run(tmp_path)
        run = _Run(tmp_path)
        run.restore()
        assert _confirmed(run) == dict.fromkeys(ids, False)

        run.first_scan()

        assert _confirmed(run) == dict.fromkeys(ids, True)
        run.stop()

    def test_the_published_columns_do_not_carry_it(self, tmp_path):
        _first_run(tmp_path)
        run = _Run(tmp_path)
        run.restore()
        cols = run.db.query("SELECT * FROM sources LIMIT 1").schema.names
        assert "confirmed" not in cols
        run.stop()

    def test_an_unconfirmed_source_is_not_a_sighting_of_its_annotations(self, tmp_path):
        from tests.roi_annotations_test import _annotation

        ids = _first_run(tmp_path)
        run = _Run(tmp_path)
        run.restore()
        run.db.put_rois(f"{ids[0]}/Image:0", [_annotation()])

        assert run.db.mark_sources_seen() == 0  # listed, not yet verified

        run.first_scan()
        assert run.db.mark_sources_seen() == 1
        run.stop()

    def test_a_row_no_claim_holds_is_swept_when_its_root_is_walked(self, tmp_path):
        from biopb_tensor_server.core.discovery import SourceClaim

        _first_run(tmp_path)
        run = _Run(tmp_path)
        run.restore()
        path = str(run.monitored / "stray.zarr")
        stray = SourceClaim("zarr", path, "zarr_stray", member_paths=[path])
        run.db.sync_pending_source(stray, record=run.reconciler._catalog_record(stray))
        assert "zarr_stray" in run.rows()

        run.first_scan()

        assert "zarr_stray" not in run.rows()
        assert len(run.rows()) == 2
        run.stop()

    def test_a_source_nobody_has_seen_for_a_month_is_not_restored(self, tmp_path):
        ids = _first_run(tmp_path)
        db = MetadataDatabase(
            store_path=tmp_path / "catalog.duckdb", restore_sources=True
        )
        db.open()
        conn = db._get_connection()
        conn.execute("UPDATE catalog_roots SET last_scanned = now() - INTERVAL 60 DAY")
        conn.execute(
            "UPDATE source_catalog SET last_seen = now() - INTERVAL 60 DAY "
            "WHERE source_id = ?",
            [ids[0]],
        )
        conn.execute(
            "UPDATE source_catalog SET last_seen = now() WHERE source_id = ?", [ids[1]]
        )
        db.close()

        run = _Run(tmp_path)
        run.restore()

        assert list(run.rows()) == [ids[1]]
        run.stop()
