"""Deferred registration: the first scan claims, registration runs afterwards.

The catalog has a row for every source the first scan finds long before any file
is opened (``is_resolved`` false, ``unresolved_reason`` ``pending``); a worker
registers them, newest first, and a read that needs one registers it at once.
Real zarr sources against a real catalog, driven one tick at a time.
"""

import json
import os
import threading
import time

import numpy as np
import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.adapters.unresolved import PendingSourceAdapter
from biopb_tensor_server.core.discovery import DiscoveryState
from biopb_tensor_server.core.errors import SourceRegistrationError
from biopb_tensor_server.core.source_registry import SourceRegistry
from biopb_tensor_server.sources.registration_worker import RegistrationWorker

from tests import catalog_server, make_manager

zarr = pytest.importorskip("zarr")


def _make_zarr(parent, name, shape=(2, 8, 8)):
    path = os.path.join(parent, name)
    z = zarr.open_array(
        path, mode="w", shape=shape, chunks=(1,) + shape[1:], dtype="uint16"
    )
    z[:] = np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape)
    return path


def _manager(root, workers=0):
    server = catalog_server("localhost:0")
    manager = make_manager(
        server=server,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        metadata_db=server.metadata_db,
        monitored_dirs={root},
        stability_window=0,
        registration_workers=workers,
    )
    return manager, server


def _first_scan(manager):
    """The first tick, deferred, without the loop thread or the worker."""
    manager._reconciler.set_defer_registration(True)
    manager._handle_rescan()


def _rows(server):
    table = server.metadata_db.query(
        "SELECT source_id, is_resolved, unresolved_reason, metadata_json, "
        "len(tensors) AS tensors FROM sources ORDER BY source_id"
    )
    return {row["source_id"]: row for row in table.to_pylist()}


def _only_ids(server):
    return sorted(_rows(server))


class TestFirstScan:
    def test_it_catalogues_every_source_without_registering_any(self, tmp_path):
        for name in ("a.zarr", "b.zarr", "c.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)

        rows = _rows(server)
        assert len(rows) == 3
        for row in rows.values():
            assert row["is_resolved"] is False
            assert row["unresolved_reason"] == "pending"
            assert row["tensors"] == 0
        assert manager.pending_registrations() == 3
        assert all(
            isinstance(server.sources.get(sid), PendingSourceAdapter) for sid in rows
        )

    def test_the_scan_is_over_while_registration_is_not(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, _ = _manager(tmp_path)
        _first_scan(manager)
        assert manager._initial_scan_done
        assert manager.pending_registrations() == 1
        assert not manager.registration_idle()

    def test_without_a_worker_pool_nothing_is_deferred(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        manager._handle_rescan()  # defer never switched on, as with 0 workers
        (row,) = _rows(server).values()
        assert row["is_resolved"] is True
        assert row["unresolved_reason"] is None
        assert manager.pending_registrations() == 0

    def test_a_source_found_after_the_first_scan_registers_as_it_is_claimed(
        self, tmp_path
    ):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        _make_zarr(tmp_path, "b.zarr")
        manager._handle_rescan()

        rows = _rows(server)
        assert len(rows) == 2
        resolved = [r for r in rows.values() if r["is_resolved"]]
        assert len(resolved) == 1 and resolved[0]["tensors"] == 1
        assert manager.pending_registrations() == 1


class TestReadRegistersTheSource:
    def test_get_registered_swaps_in_the_real_adapter_and_fills_the_row(self, tmp_path):
        for name in ("a.zarr", "b.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        first, second = _only_ids(server)

        adapter = server.sources.get_registered(first)

        assert not isinstance(adapter, PendingSourceAdapter)
        assert server.sources.get(first) is adapter
        row = _rows(server)[first]
        assert row["is_resolved"] is True
        assert row["unresolved_reason"] is None
        assert row["tensors"] == 1
        # The other source is untouched.
        assert isinstance(server.sources.get(second), PendingSourceAdapter)
        assert manager.pending_registrations() == 1

    def test_plain_get_never_registers(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        assert isinstance(server.sources.get(sid), PendingSourceAdapter)
        assert manager.pending_registrations() == 1

    def test_concurrent_reads_share_one_registration(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)

        reconciler = manager._reconciler
        built = []
        real = reconciler._register_source_claim

        def counting(*args, **kwargs):
            built.append(1)
            time.sleep(0.2)
            return real(*args, **kwargs)

        reconciler._register_source_claim = counting
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(server.sources.get_registered(sid))
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1
        assert len({id(a) for a in results}) == 1
        assert not isinstance(results[0], PendingSourceAdapter)

    def test_the_placeholder_does_not_get_uploaded_tensors_attached(self, tmp_path):
        attached = []
        registry = SourceRegistry(on_register=lambda sid, a: attached.append(a))
        stub = PendingSourceAdapter(_config("x", str(tmp_path / "x.zarr")))
        registry.register("x", stub)
        assert attached == []


class TestTheWorker:
    def test_it_registers_everything(self, tmp_path):
        for name in ("a.zarr", "b.zarr", "c.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path, workers=2)
        manager._registration_worker.start()
        try:
            _first_scan(manager)
            _wait_until(lambda: manager.pending_registrations() == 0)
        finally:
            manager._registration_worker.stop()

        for row in _rows(server).values():
            assert row["is_resolved"] is True
            assert row["tensors"] == 1
        assert manager.registration_idle()

    def test_newest_file_goes_first(self, tmp_path):
        paths = {
            name: _make_zarr(tmp_path, name) for name in ("a.zarr", "b.zarr", "c.zarr")
        }
        now = time.time()
        for age, name in enumerate(("b.zarr", "c.zarr", "a.zarr")):
            os.utime(paths[name], (now - 100 * (age + 1),) * 2)
        manager, server = _manager(tmp_path, workers=1)
        order = []
        worker = manager._registration_worker
        inner = worker._register
        worker._register = lambda sid: (order.append(sid), inner(sid))[1]

        _first_scan(manager)  # queued, worker not running yet
        by_path = {
            os.path.basename(server.sources.get(sid).source_url): sid
            for sid in _only_ids(server)
        }
        worker.start()
        try:
            _wait_until(lambda: manager.pending_registrations() == 0)
        finally:
            worker.stop()

        assert order == [by_path["b.zarr"], by_path["c.zarr"], by_path["a.zarr"]]

    def test_readers_racing_the_worker_register_each_source_once(self, tmp_path):
        import collections
        import random

        count = 12
        for i in range(count):
            _make_zarr(tmp_path, f"s{i}.zarr")
        manager, server = _manager(tmp_path, workers=3)
        reconciler = manager._reconciler
        built = collections.Counter()
        real = reconciler._register_source_claim

        def counting(claim, *args, **kwargs):
            built[claim.source_id] += 1
            return real(claim, *args, **kwargs)

        reconciler._register_source_claim = counting
        errors = []

        def reader(ids):
            try:
                for sid in ids:
                    adapter = server.sources.get_registered(sid)
                    assert not isinstance(adapter, PendingSourceAdapter)
            except Exception as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        worker = manager._registration_worker
        worker.start()
        try:
            # The walk claims while the pool is already registering.
            _first_scan(manager)
            ids = _only_ids(server)
            threads = [
                threading.Thread(target=reader, args=(random.sample(ids, len(ids)),))
                for _ in range(6)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            _wait_until(lambda: manager.pending_registrations() == 0)
        finally:
            worker.stop()

        assert errors == []
        assert len(built) == count
        assert set(built.values()) == {1}
        for row in _rows(server).values():
            assert row["is_resolved"] is True and row["tensors"] == 1

    def test_stop_waits_one_deadline_for_the_whole_pool(self):
        release = threading.Event()
        worker = RegistrationWorker(lambda sid: release.wait(30) or True, workers=3)
        for sid in "abc":
            worker.enqueue(sid)
        worker.start()
        _wait_until(lambda: worker.queued() == 0)
        started = time.monotonic()
        try:
            worker.stop(join_timeout=0.5)
            # Three threads stuck in a file open: one deadline, not three.
            assert time.monotonic() - started < 1.2
        finally:
            release.set()

    def test_a_queued_source_is_not_queued_twice(self):
        seen = []
        worker = RegistrationWorker(lambda sid: seen.append(sid) or True, workers=1)
        worker.enqueue("a", 1.0)
        worker.enqueue("a", 2.0)
        worker.enqueue("b", 3.0)
        assert worker.queued() == 2
        worker.start()
        try:
            _wait_until(lambda: len(seen) == 2)
        finally:
            worker.stop()
        assert seen == ["b", "a"]

    def test_a_failing_registration_does_not_stop_the_pool(self):
        seen = []

        def register(sid):
            seen.append(sid)
            if sid == "bad":
                raise RuntimeError("boom")
            return True

        worker = RegistrationWorker(register, workers=1)
        worker.enqueue("bad", 2.0)
        worker.enqueue("good", 1.0)
        worker.start()
        try:
            _wait_until(lambda: seen == ["bad", "good"])
        finally:
            worker.stop()


class TestFailure:
    def _break(self, path):
        for name in (".zarray", "zarr.json"):
            meta = os.path.join(path, name)
            if os.path.exists(meta):
                with open(meta, "w") as f:
                    f.write("{ not json")
                return meta
        raise AssertionError("no zarr metadata to break")

    def test_a_failed_registration_leaves_a_row_that_says_why(self, tmp_path):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        self._break(path)

        adapter = server.sources.get_registered(sid)

        assert isinstance(adapter, PendingSourceAdapter)
        row = _rows(server)[sid]
        assert row["is_resolved"] is False
        assert row["unresolved_reason"] == "failed"
        assert json.loads(row["metadata_json"])["registration_error"]
        # Nothing is waiting on it.
        assert manager.pending_registrations() == 0
        with pytest.raises(SourceRegistrationError, match="could not be registered"):
            adapter.get_tensor_adapter(None)
        with pytest.raises(SourceRegistrationError):
            adapter.resolve()

    def test_a_failed_source_is_not_retried_by_every_read(self, tmp_path):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        self._break(path)
        server.sources.get_registered(sid)

        reconciler = manager._reconciler
        attempts = []
        real = reconciler._register_source_claim
        reconciler._register_source_claim = lambda *a, **k: (
            attempts.append(1),
            real(*a, **k),
        )[1]
        server.sources.get_registered(sid)
        server.sources.get_registered(sid)
        assert attempts == []  # inside its backoff window

    def test_it_is_retried_once_the_backoff_has_passed_and_the_file_is_fixed(
        self, tmp_path
    ):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        meta = self._break(path)
        good = os.path.join(str(tmp_path), "good.zarr")
        _make_zarr(tmp_path, "good.zarr")
        server.sources.get_registered(sid)

        reconciler = manager._reconciler
        assert reconciler.failed_pending_due() == []
        reconciler._failed_sources[sid].next_retry_at = 0.0
        assert reconciler.failed_pending_due() == [sid]

        with open(os.path.join(good, os.path.basename(meta))) as src:
            fixed = src.read()
        with open(meta, "w") as dst:
            dst.write(fixed)
        assert reconciler.ensure_registered(sid)

        row = _rows(server)[sid]
        assert row["is_resolved"] is True
        assert row["unresolved_reason"] is None
        assert manager.pending_registrations() == 0

    def test_the_tick_requeues_what_failed(self, tmp_path):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path, workers=1)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        self._break(path)
        server.sources.get_registered(sid)
        manager._reconciler._failed_sources[sid].next_retry_at = 0.0

        manager._requeue_failed_registrations()
        assert manager._registration_worker.queued() == 1


class TestWhilePending:
    def test_a_removed_source_is_not_registered_back(self, tmp_path):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)

        import shutil

        shutil.rmtree(path)
        manager._handle_rescan()
        manager._handle_rescan()  # removal needs two consecutive misses

        assert server.sources.get(sid) is None
        assert _rows(server) == {}
        assert manager.pending_registrations() == 0
        assert manager._reconciler.ensure_registered(sid)
        assert server.sources.get(sid) is None

    def test_a_changed_source_is_registered_by_its_refresh(self, tmp_path):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)

        # The claim's member is the store directory, so that is what changes.
        later = time.time() + 5
        os.utime(path, (later, later))
        manager._handle_rescan()

        row = _rows(server)[sid]
        assert row["is_resolved"] is True and row["tensors"] == 1
        assert manager.pending_registrations() == 0
        assert not isinstance(server.sources.get(sid), PendingSourceAdapter)


class TestPrecacheRouting:
    def test_the_backlog_seed_skips_what_is_pending(self, tmp_path):
        for name in ("a.zarr", "b.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        first, second = _only_ids(server)
        assert manager.iter_local_source_mtimes() == []

        server.sources.get_registered(first)
        assert [sid for sid, _ in manager.iter_local_source_mtimes()] == [first]

    def test_a_registered_source_reaches_the_backlog_hook(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        routed = []
        manager.set_source_registered_hook(
            lambda sid, mtime: routed.append((sid, mtime))
        )
        live = []
        manager.set_source_committed_hook(live.append)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        assert routed == []

        server.sources.get_registered(sid)

        assert [s for s, _ in routed] == [sid]
        assert routed[0][1] > 0
        assert live == []  # startup set: the backlog, not the live tier

    def test_the_gate_opens_when_the_scan_is_over_and_nothing_is_pending(
        self, tmp_path
    ):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        assert not manager.registration_idle()  # first scan not done
        _first_scan(manager)
        assert not manager.registration_idle()
        server.sources.get_registered(_only_ids(server)[0])
        assert manager.registration_idle()

    def test_enqueue_backlog_orders_newest_first_and_skips_what_is_queued(self):
        from biopb_tensor_server.core.config import PrecacheConfig
        from biopb_tensor_server.serving.precache import PrecacheWorker

        worker = PrecacheWorker(object(), PrecacheConfig())
        worker.enqueue_backlog("a", 5.0)
        worker.enqueue_backlog("b", 9.0)
        worker.enqueue_backlog("a", 7.0)  # already queued
        assert worker._pop_backlog() == (-9.0, "b")
        assert worker._pop_backlog() == (-5.0, "a")
        assert worker._pop_backlog() is None

    def test_the_backlog_is_held_while_its_gate_is_closed(self):
        from biopb_tensor_server.core.config import PrecacheConfig
        from biopb_tensor_server.serving.precache import PrecacheWorker

        worker = PrecacheWorker(object(), PrecacheConfig())
        warmed = []
        worker._cache_active = lambda: True
        worker._has_headroom = lambda: True
        worker._process_source = lambda sid, backlog=False: warmed.append(sid) or False
        gate = [False]
        worker.backlog_gate = lambda: gate[0]
        worker.enqueue_backlog("a", 1.0)
        worker.start()
        try:
            time.sleep(0.3)
            assert warmed == []
            gate[0] = True
            _wait_until(lambda: warmed == ["a"])
        finally:
            worker.stop()

    def test_a_live_addition_is_not_held_by_the_gate(self):
        from biopb_tensor_server.core.config import PrecacheConfig
        from biopb_tensor_server.serving.precache import PrecacheWorker

        worker = PrecacheWorker(object(), PrecacheConfig())
        warmed = []
        worker._cache_active = lambda: True
        worker._has_headroom = lambda: True
        worker._process_source = lambda sid, backlog=False: warmed.append(sid) or False
        worker.backlog_gate = lambda: False
        worker.enqueue_backlog("startup", 1.0)
        worker.start()
        try:
            worker.enqueue("live")
            _wait_until(lambda: warmed == ["live"])
            assert worker._backlog_has_items()
        finally:
            worker.stop()


class TestOverFlight:
    """A client never has to know a source was still pending."""

    def _serve(self, tmp_path, count=2):
        from biopb.tensor import TensorFlightClient

        for i in range(count):
            _make_zarr(tmp_path, f"s{i}.zarr")
        manager, server = _manager(tmp_path)
        server.set_registration_pending_provider(manager.pending_registrations)
        _first_scan(manager)
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(1)
        return manager, server, TensorFlightClient(f"grpc://localhost:{server.port}")

    def test_the_catalog_shows_pending_rows_and_health_counts_them(self, tmp_path):
        manager, server, client = self._serve(tmp_path)
        try:
            rows = client.query(
                "SELECT source_id, is_resolved, unresolved_reason FROM sources",
                format="records",
            )
            assert [r["unresolved_reason"] for r in rows] == ["pending", "pending"]
            assert not any(r["is_resolved"] for r in rows)
            assert client.health_check()["registration_pending"] == 2
        finally:
            client.close()
            server.shutdown()

    def test_reading_a_pending_source_registers_it(self, tmp_path):
        manager, server, client = self._serve(tmp_path)
        try:
            first, second = _only_ids(server)

            data = client.get_tensor(first).compute()

            assert data.shape == (2, 8, 8)
            assert int(data[1, 0, 0]) == 64
            rows = {r["source_id"]: r for r in _rows(server).values()}
            assert rows[first]["is_resolved"] is True
            assert rows[second]["unresolved_reason"] == "pending"
            assert client.health_check()["registration_pending"] == 1
        finally:
            client.close()
            server.shutdown()

    def test_resolve_on_a_pending_source_returns_its_filled_row(self, tmp_path):
        manager, server, client = self._serve(tmp_path)
        try:
            first, _ = _only_ids(server)

            row = client.resolve_source(first)

            assert row["source_id"] == first
            assert row["is_resolved"] is True
            assert len(row["tensors"]) == 1
        finally:
            client.close()
            server.shutdown()

    def test_the_sdk_registers_a_pending_source_before_it_would_refuse(self, tmp_path):
        """``get_source_metadata`` and ``get_descriptor`` check the catalog row
        first, and refuse an unresolved one without asking the server."""
        manager, server, client = self._serve(tmp_path, count=3)
        try:
            first, second, third = _only_ids(server)

            assert client.get_source_metadata(first) is not None
            assert manager._reconciler.is_pending(first) is False

            desc = client.get_descriptor(second)
            assert list(desc.shape) == [2, 8, 8]
            assert manager._reconciler.is_pending(second) is False
            assert manager._reconciler.is_pending(third) is True
        finally:
            client.close()
            server.shutdown()

    def test_a_failed_registration_reaches_the_sdk_with_its_reason(self, tmp_path):
        manager, server, client = self._serve(tmp_path, count=1)
        try:
            (sid,) = _only_ids(server)
            for name in (".zarray", "zarr.json"):
                meta = os.path.join(tmp_path, "s0.zarr", name)
                if os.path.exists(meta):
                    with open(meta, "w") as f:
                        f.write("{ not json")
            with pytest.raises(Exception, match="could not be registered"):
                client.get_descriptor(sid)
        finally:
            client.close()
            server.shutdown()


class TestStats:
    def test_a_registration_is_recorded_by_type_with_its_sizes(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        server.sources.get_registered(_only_ids(server)[0])

        lines = manager._reconciler.stats.summary_lines()
        zarr_line = next(line for line in lines if line.strip().startswith("zarr:"))
        assert "n=1" in zarr_line
        for field in ("create", "normalize", "metadata", "upsert"):
            assert f"{field} p50=" in zarr_line
        for field in ("metadata_bytes", "tensors_bytes", "descriptor_bytes", "members"):
            assert f"{field} mean=" in zarr_line

    def test_the_walk_times_each_adapters_claims(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, _ = _manager(tmp_path)
        _first_scan(manager)
        claim_lines = [
            line
            for line in manager._reconciler.stats.summary_lines()
            if line.strip().startswith("claim ")
        ]
        assert any("claimed=1" in line for line in claim_lines)

    def test_sync_source_added_reports_what_the_row_cost(self, tmp_path):
        from biopb_tensor_server.core.registration_stats import SyncCost

        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        adapter = server.sources.get_registered(sid)

        cost = server.metadata_db.sync_source_added(sid, adapter)

        assert isinstance(cost, SyncCost)
        assert cost.tensors_bytes > 0 and cost.descriptor_bytes > 0
        assert cost.metadata_s >= 0 and cost.upsert_s >= 0

    def test_the_summary_is_logged_once_when_registration_drains(
        self, tmp_path, caplog
    ):
        import logging

        for name in ("a.zarr", "b.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        with caplog.at_level(logging.INFO):
            for sid in _only_ids(server):
                server.sources.get_registered(sid)
        logged = [r for r in caplog.records if "Registration cost" in r.getMessage()]
        assert len(logged) == 1
        assert "(2 sources)" in logged[0].getMessage()


def _config(source_id, url):
    from biopb_tensor_server.core.config import SourceConfig

    return SourceConfig(type="zarr", url=url, source_id=source_id)


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")
