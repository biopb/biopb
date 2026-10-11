"""Claim signatures: when they are taken, and which one is persisted."""

import json
import os

from biopb_tensor_server.sources.reconciler import Reconciler

from tests import deferred_registration_test as drt


def _touch(zarr_path):
    """Change a zarr directory's own stat, which is what its signature reads."""
    with open(os.path.join(zarr_path, "marker"), "w") as f:
        f.write("x")


def _count_signatures(monkeypatch):
    calls = []
    real = Reconciler._build_claim_signatures

    def counting(self, claim):
        calls.append(claim.source_id)
        return real(self, claim)

    monkeypatch.setattr(Reconciler, "_build_claim_signatures", counting)
    return calls


def _stored(server, source_id):
    row = (
        server.metadata_db._get_connection()
        .execute(
            "SELECT signature FROM source_catalog WHERE source_id = ?", [source_id]
        )
        .fetchone()
    )
    return json.loads(row[0]) if row else None


class TestFirstScan:
    def test_each_claim_is_stat_ed_once(self, tmp_path, monkeypatch):
        for name in ("a.zarr", "b.zarr", "c.zarr"):
            drt._make_zarr(tmp_path, name)
        manager, _ = drt._manager(tmp_path)
        calls = _count_signatures(monkeypatch)
        drt._first_scan(manager)
        # The walk takes each one; the end-of-walk reconcile does not repeat it.
        assert sorted(calls) == sorted(set(calls)) and len(calls) == 3

    def test_a_change_after_the_claim_is_found_by_the_next_rescan(
        self, tmp_path, monkeypatch
    ):
        paths = [drt._make_zarr(tmp_path, f"{n}.zarr") for n in ("a", "b", "c")]
        manager, _ = drt._manager(tmp_path)
        drt._first_scan(manager)

        refreshed = []
        real = Reconciler._refresh_claim
        monkeypatch.setattr(
            Reconciler,
            "_refresh_claim",
            lambda self, claim: (
                refreshed.append(claim.primary_path),
                real(self, claim),
            )[1],
        )
        _touch(paths[1])
        manager._handle_rescan()
        assert refreshed == [paths[1]]


class TestRegistration:
    def test_a_pending_claim_registers_with_its_claim_time_signature(self, tmp_path):
        path = drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (source_id,) = manager._reconciler._pending
        claim_time = manager._reconciler._source_signatures[source_id][path]

        # The file moves between the claim and its registration.
        _touch(path)
        drt._resolve(manager, server, source_id)

        stored = _stored(server, source_id)[path]
        assert stored == list(claim_time[1:])  # no st_dev
        st = os.stat(path)
        assert stored != [st.st_ino, st.st_mtime_ns, st.st_ctime_ns]

    def test_a_refresh_persists_the_new_signature_not_the_old_one(self, tmp_path):
        path = drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        (source_id,) = manager._reconciler._pending
        drt._resolve(manager, server, source_id)
        before = _stored(server, source_id)[path]

        _touch(path)
        manager._handle_rescan()

        after = _stored(server, source_id)[path]
        st = os.stat(path)
        assert after != before
        assert after == [st.st_ino, st.st_mtime_ns, st.st_ctime_ns]


def _count_by_id(monkeypatch):
    calls = []
    real = Reconciler._build_claim_signatures

    def counting(self, claim):
        calls.append(claim.source_id)
        return real(self, claim)

    monkeypatch.setattr(Reconciler, "_build_claim_signatures", counting)
    return calls


class TestRegisteredWhereFound:
    """A claim that registers as it is found is stat'ed once, before the parse."""

    def test_an_add_stats_each_claim_once(self, tmp_path, monkeypatch):
        for name in ("a.zarr", "b.zarr", "c.zarr"):
            drt._make_zarr(tmp_path, name)
        manager, _ = drt._manager(tmp_path)
        calls = _count_by_id(monkeypatch)

        manager._handle_rescan()  # no deferral: each registers as the walk finds it

        assert sorted(calls) == sorted(set(calls)) and len(calls) == 3

    def test_a_refresh_stats_the_claim_once_to_compare_and_once_to_rebuild(
        self, tmp_path, monkeypatch
    ):
        paths = [drt._make_zarr(tmp_path, f"{n}.zarr") for n in ("a", "b", "c")]
        manager, server = drt._manager(tmp_path)
        manager._handle_rescan()
        by_path = {
            manager._reconciler.claim_primary_path(sid): sid
            for sid in manager._reconciler.claim_ids()
        }
        calls = _count_by_id(monkeypatch)

        _touch(paths[1])
        manager._handle_rescan()

        assert calls.count(by_path[paths[1]]) == 2
        assert calls.count(by_path[paths[0]]) == calls.count(by_path[paths[2]]) == 1

    def test_a_file_that_changes_during_the_parse_is_refreshed_next_scan(
        self, tmp_path, monkeypatch
    ):
        path = drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        real = Reconciler._register_source_claim
        touched = []

        def touching(self, claim, *a, **k):
            if not touched:
                touched.append(1)
                _touch(path)  # after the signature was taken, before the parse ends
            return real(self, claim, *a, **k)

        monkeypatch.setattr(Reconciler, "_register_source_claim", touching)
        manager._handle_rescan()
        monkeypatch.setattr(Reconciler, "_register_source_claim", real)

        refreshed = []
        real_refresh = Reconciler._refresh_claim
        monkeypatch.setattr(
            Reconciler,
            "_refresh_claim",
            lambda self, claim: (
                refreshed.append(claim.primary_path),
                real_refresh(self, claim),
            )[1],
        )
        manager._handle_rescan()
        assert refreshed == [path]
