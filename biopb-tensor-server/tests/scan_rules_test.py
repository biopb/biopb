"""What the walk writes for each claim it finds, and what leaves a row alone.

Real zarr sources against a real catalog, one tick at a time. A claim with a
catalog record has its row in ``source_catalog`` from the moment the walk finds it;
registration fills the row in; a failed one stays failed until a new signature, a
drop or a resolve.
"""

import json
import os
from pathlib import Path

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import DiscoveryState, SourceClaim
from biopb_tensor_server.sources.roots import Root, RootKind

from tests import catalog_server, deferred_registration_test as drt, make_manager


def _catalog_row(server, source_id):
    return (
        server.metadata_db._get_connection()
        .execute(
            "SELECT is_resolved, unresolved_reason, unresolved_error, primary_path, "
            "signature FROM source_catalog WHERE source_id = ?",
            [source_id],
        )
        .fetchone()
    )


def _volatile_count(server):
    return (
        server.metadata_db._get_connection()
        .execute("SELECT count(*) FROM source_catalog WHERE root_id IS NULL")
        .fetchone()[0]
    )


def _touch(zarr_path):
    with open(os.path.join(zarr_path, "marker"), "w") as f:
        f.write("x")


class TestNewClaim:
    def test_the_walk_writes_its_row_to_the_persistent_table(self, tmp_path):
        path = drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)

        resolved, reason, error, primary, signature = _catalog_row(server, sid)
        assert (resolved, reason, error) == (False, "pending", None)
        assert primary == path
        assert path in json.loads(signature)
        assert _volatile_count(server) == 0

    def test_registration_fills_that_row_in(self, tmp_path):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)
        before = _catalog_row(server, sid)

        drt._resolve(manager, server, sid)

        resolved, reason, _, primary, signature = _catalog_row(server, sid)
        assert (resolved, reason) == (True, None)
        assert (primary, signature) == before[3:]
        assert _volatile_count(server) == 0

    def test_the_catalog_is_not_reread_so_a_rescan_leaves_resolved_rows_alone(
        self, tmp_path, monkeypatch
    ):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)
        drt._resolve(manager, server, sid)

        writes = []
        db = server.metadata_db
        for name in (
            "sync_pending_source",
            "sync_pending_sources",
            "sync_source_added",
        ):
            real = getattr(db, name)
            monkeypatch.setattr(
                db, name, lambda *a, _real=real, **k: (writes.append(1), _real(*a, **k))
            )
        manager._handle_rescan()
        assert writes == []


class TestUnresolvedRows:
    def _flip(self, manager, residents):
        """Make the reconciler see the claim as unresolved (or not)."""
        manager._reconciler._claim_is_unresolved = lambda claim: not residents

    def test_a_pending_claim_that_stops_being_resident_waits_for_a_client(
        self, tmp_path
    ):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)

        self._flip(manager, residents=False)
        manager._handle_rescan()

        assert _catalog_row(server, sid)[:2] == (False, "needs_recall")
        assert sid in manager._reconciler._recall

    def test_a_cloud_claim_that_became_resident_is_queued_again(self, tmp_path):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)
        self._flip(manager, residents=False)
        manager._handle_rescan()

        queued = []
        manager._reconciler.set_pending_hook(queued.append)
        self._flip(manager, residents=True)
        manager._handle_rescan()

        assert _catalog_row(server, sid)[:2] == (False, "pending")
        assert sid not in manager._reconciler._recall
        assert queued == [sid]

    def test_a_failed_row_is_left_alone(self, tmp_path):
        path = drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)
        drt.TestFailure()._break(path)
        assert not manager._reconciler.ensure_registered(sid)

        self._flip(manager, residents=False)
        manager._handle_rescan()

        assert _catalog_row(server, sid)[:2] == (False, "failed")
        assert sid not in manager._reconciler._recall


class TestFailedClaim:
    def _second_zarr(self, tmp_path, manager, server):
        """A zarr that appears after the first scan, with metadata that will not parse."""
        path = drt._make_zarr(tmp_path, "b.zarr")
        drt.TestFailure()._break(path)
        return path

    def test_a_drop_that_fails_leaves_nothing_behind(self, tmp_path):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        known = set(drt._only_ids(server))
        path = self._second_zarr(tmp_path, manager, server)

        events = list(manager.add_local_source(path))

        (result,) = [e[1] for e in events if e[0] == "result"]
        assert result.failed
        assert set(drt._only_ids(server)) == known

    def test_the_walk_keeps_a_failed_claim_and_does_not_retry_it(self, tmp_path):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        known = set(drt._only_ids(server))
        self._second_zarr(tmp_path, manager, server)

        manager._handle_rescan()
        (sid,) = set(drt._only_ids(server)) - known
        assert _catalog_row(server, sid)[:2] == (False, "failed")
        assert _catalog_row(server, sid)[2]  # why

        attempts = []
        reconciler = manager._reconciler
        real = reconciler._register_source_claim
        reconciler._register_source_claim = lambda *a, **k: (
            attempts.append(1),
            real(*a, **k),
        )[1]
        manager._handle_rescan()
        manager._handle_rescan()
        assert attempts == []


class TestFailedRefresh:
    def test_a_refresh_that_fails_to_write_its_row_keeps_the_old_one(
        self, tmp_path, monkeypatch
    ):
        path = drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (sid,) = drt._only_ids(server)
        drt._resolve(manager, server, sid)
        before = _catalog_row(server, sid)

        db = server.metadata_db
        real = db.sync_source_added
        calls = []

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("catalog write failed")
            return real(*a, **k)

        monkeypatch.setattr(db, "sync_source_added", flaky)
        _touch(path)
        manager._handle_rescan()

        assert len(calls) == 2  # the refresh, then the restore of what served
        after = _catalog_row(server, sid)
        assert after[:2] == (True, None)
        assert after[3] == before[3]
        assert _volatile_count(server) == 0
        assert server.sources.get(sid) is not None


class TestRecord:
    def _manager(self, root, **kw):
        server = catalog_server("localhost:0")
        manager = make_manager(
            server=server,
            registry=get_default_registry(),
            discovery_state=DiscoveryState(),
            metadata_db=server.metadata_db,
            monitored_dirs={root},
            stability_window=0,
            **kw,
        )
        return manager

    def test_a_cloud_root_has_a_record_that_says_so(self, tmp_path):
        manager = self._manager(tmp_path, cloud_roots={tmp_path})
        claim = SourceClaim("zarr", str(tmp_path / "a.zarr"), "x")
        record = manager._reconciler._catalog_record(claim)
        assert record is not None and record.cloud

    def test_a_local_root_has_a_record_that_does_not(self, tmp_path):
        manager = self._manager(tmp_path)
        claim = SourceClaim("zarr", str(tmp_path / "a.zarr"), "x")
        record = manager._reconciler._catalog_record(claim)
        assert record is not None and not record.cloud

    def test_a_drop_has_none(self, tmp_path):
        manager = self._manager(tmp_path / "root")
        elsewhere = Path(tmp_path) / "dropped"
        manager._roots.add(Root(RootKind.DROPPED, str(elsewhere), label="dnd://x"))
        claim = SourceClaim("zarr", str(elsewhere / "a.zarr"), "x")
        assert manager._reconciler._catalog_record(claim) is None

    def test_a_mirror_has_none(self, tmp_path):
        manager = self._manager(tmp_path)
        claim = SourceClaim("tensor-server", str(tmp_path / "a.zarr"), "x")
        assert manager._reconciler._catalog_record(claim) is None
