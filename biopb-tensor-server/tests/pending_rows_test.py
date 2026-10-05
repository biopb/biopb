"""Batched pending rows: the writer, the bulk insert, and the first scan's use of them."""

import threading
import time

from biopb_tensor_server.core.discovery import SourceClaim
from biopb_tensor_server.serving.metadata_db import MetadataDatabase
from biopb_tensor_server.sources.pending_rows import PendingRow, PendingRowWriter

from tests import deferred_registration_test as drt
from tests.test_metadata_db import MockAdapter


def _row(i, recall=False):
    return PendingRow(
        SourceClaim("zarr", f"/d/s{i}.zarr", f"s{i}"), None, recall=recall
    )


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestWriter:
    def test_a_full_batch_is_written_without_waiting_for_the_interval(self):
        batches = []
        writer = PendingRowWriter(batches.append, batch_size=5, interval=3600)
        try:
            for i in range(5):
                writer.add(_row(i))
            assert _wait(lambda: len(batches) == 1)
            assert sorted(r.claim.source_id for r in batches[0]) == [
                f"s{i}" for i in range(5)
            ]
        finally:
            writer.close()

    def test_a_partial_batch_is_written_when_the_interval_passes(self):
        batches = []
        writer = PendingRowWriter(batches.append, batch_size=100, interval=0.05)
        try:
            writer.add(_row(0))
            assert _wait(lambda: len(batches) == 1)
        finally:
            writer.close()

    def test_close_writes_what_is_left(self):
        batches = []
        writer = PendingRowWriter(batches.append, batch_size=100, interval=3600)
        writer.add(_row(0))
        writer.add(_row(1))
        writer.close()
        assert [len(b) for b in batches] == [2]

    def test_a_discarded_row_is_not_written(self):
        batches = []
        writer = PendingRowWriter(batches.append, batch_size=100, interval=3600)
        writer.add(_row(0))
        writer.add(_row(1))
        writer.discard("s0")
        writer.close()
        assert [r.claim.source_id for b in batches for r in b] == ["s1"]

    def test_discard_waits_out_a_write_in_progress(self):
        started, release = threading.Event(), threading.Event()
        written = []

        def slow_write(batch):
            started.set()
            release.wait(5)
            written.extend(r.claim.source_id for r in batch)

        writer = PendingRowWriter(slow_write, batch_size=1, interval=3600)
        writer.add(_row(0))
        assert started.wait(5)

        done = threading.Event()
        threading.Thread(
            target=lambda: (writer.discard("s0"), done.set()), daemon=True
        ).start()
        assert not done.wait(0.2)  # held by the write
        release.set()
        assert done.wait(5)
        assert written == ["s0"]  # it was in the batch: the caller's delete follows
        writer.close()

    def test_a_failed_batch_is_logged_and_the_writer_carries_on(self, caplog):
        calls = []

        def write(batch):
            calls.append(len(batch))
            if len(calls) == 1:
                raise RuntimeError("disk full")

        writer = PendingRowWriter(write, batch_size=1, interval=3600)
        writer.add(_row(0))
        assert _wait(lambda: len(calls) == 1)
        writer.add(_row(1))
        assert _wait(lambda: len(calls) == 2)
        writer.close()
        assert "Could not write 1 pending catalog rows" in caplog.text


class TestBulkInsert:
    def _count(self, db, table):
        return (
            db._get_connection().execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        )

    def test_it_writes_pending_and_recall_rows(self):
        db = MetadataDatabase()
        db.sync_pending_sources([_row(0), _row(1, recall=True)])
        rows = {
            r["source_id"]: r
            for r in db.query(
                "SELECT source_id, is_resolved, unresolved_reason, "
                "len(tensors) AS n FROM sources"
            ).to_pylist()
        }
        assert rows["s0"]["unresolved_reason"] == "pending"
        assert rows["s1"]["unresolved_reason"] == "needs_recall"
        assert all(r["is_resolved"] is False and r["n"] == 0 for r in rows.values())

    def test_it_matches_the_one_row_writer(self):
        one, many = MetadataDatabase(), MetadataDatabase()
        claim = SourceClaim("zarr", "/d/a.zarr", "a")
        one.sync_pending_source(claim, "file:///alias/a.zarr")
        many.sync_pending_sources([PendingRow(claim, "file:///alias/a.zarr")])
        sql = (
            "SELECT source_id, source_url, source_type, metadata_json, is_resolved, "
            "unresolved_reason, unresolved_error, len(tensors) AS n FROM sources"
        )
        assert one.query(sql).to_pylist() == many.query(sql).to_pylist()

    def test_more_rows_than_one_statement_takes(self):
        db = MetadataDatabase()
        n = MetadataDatabase._PENDING_CHUNK * 2 + 37
        db.sync_pending_sources([_row(i) for i in range(n)])
        assert self._count(db, "sources_volatile") == n

    def test_a_registered_row_is_not_replaced(self):
        db = MetadataDatabase()
        db.sync_source_added(
            "s0", MockAdapter("s0", "/d/s0.zarr", "zarr", [4, 4], "uint8")
        )
        db.sync_pending_sources([_row(0), _row(1)])
        rows = {
            r["source_id"]: r["is_resolved"]
            for r in db.query("SELECT source_id, is_resolved FROM sources").to_pylist()
        }
        assert rows == {"s0": True, "s1": False}

    def test_a_persisted_row_is_not_shadowed(self):
        from tests.catalog_persistence_test import _record, _Restorable

        db = MetadataDatabase()
        db.sync_source_added(
            "s1", _Restorable("s1", "/d/s1.zarr", "zarr", [4, 4], "uint8"), _record()
        )
        db.sync_pending_sources([_row(1)])
        assert self._count(db, "source_catalog") == 1
        assert self._count(db, "sources_volatile") == 0
        assert db.query("SELECT source_id FROM sources").num_rows == 1

    def test_nothing_to_write(self):
        MetadataDatabase().sync_pending_sources([])


class TestFirstScan:
    def test_the_first_scan_writes_in_batches_and_leaves_every_row_behind(
        self, tmp_path, monkeypatch
    ):
        for name in ("a.zarr", "b.zarr", "c.zarr"):
            drt._make_zarr(tmp_path, name)
        manager, server = drt._manager(tmp_path)
        bulk, single = [], []
        db = server.metadata_db
        real_bulk, real_single = db.sync_pending_sources, db.sync_pending_source
        monkeypatch.setattr(
            db,
            "sync_pending_sources",
            lambda rows: (bulk.append(len(rows)), real_bulk(rows)),
        )
        monkeypatch.setattr(
            db,
            "sync_pending_source",
            lambda *a, **k: (single.append(1), real_single(*a, **k)),
        )
        drt._first_scan(manager)

        assert sum(bulk) == 3 and not single
        rows = drt._rows(server)
        assert len(rows) == 3
        assert {r["unresolved_reason"] for r in rows.values()} == {"pending"}

    def test_a_later_rescan_writes_one_row_per_claim(self, tmp_path, monkeypatch):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        drt._first_scan(manager)
        manager._reconciler.set_defer_registration(True)
        db = server.metadata_db
        bulk = []
        monkeypatch.setattr(
            db, "sync_pending_sources", lambda rows: bulk.append(len(rows))
        )
        drt._make_zarr(tmp_path, "b.zarr")
        manager._handle_rescan()
        assert bulk == []
        assert len(drt._rows(server)) == 2

    def test_a_source_removed_while_buffered_leaves_no_row(self, tmp_path):
        drt._make_zarr(tmp_path, "a.zarr")
        manager, server = drt._manager(tmp_path)
        reconciler = manager._reconciler
        reconciler.set_defer_registration(True)
        # Nothing flushes until the batch ends.
        reconciler._pending_writer = PendingRowWriter(
            server.metadata_db.sync_pending_sources, batch_size=10**6, interval=3600
        )
        manager._handle_rescan()
        (source_id,) = reconciler._pending
        reconciler._teardown_source_bookkeeping(source_id)
        reconciler.end_pending_batch()

        assert source_id not in drt._rows(server)
