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
from biopb_tensor_server.core.discovery import DiscoveryState
from biopb_tensor_server.core.errors import (
    SourceRegistrationError,
    SourceUnresolvedError,
)
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
        # Not registered: the catalog has the row, the registry has no adapter.
        assert all(server.sources.get(sid) is None for sid in rows)
        assert all(manager._reconciler.is_pending(sid) for sid in rows)

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


class TestResolveRegistersTheSource:
    def test_materialize_swaps_in_the_real_adapter_and_fills_the_row(self, tmp_path):
        for name in ("a.zarr", "b.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        first, second = _only_ids(server)

        adapter = server.sources.materialize(first)

        assert adapter is not None
        assert server.sources.get(first) is adapter
        row = _rows(server)[first]
        assert row["is_resolved"] is True
        assert row["unresolved_reason"] is None
        assert row["tensors"] == 1
        # The other source is untouched.
        assert server.sources.get(second) is None
        assert manager.pending_registrations() == 1

    def test_a_read_of_a_pending_source_asks_for_a_resolve_and_registers_nothing(
        self, tmp_path
    ):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)

        with pytest.raises(SourceUnresolvedError, match="resolve"):
            server.sources.get_registered(sid)

        assert server.sources.get(sid) is None
        assert manager.pending_registrations() == 1

    def test_a_read_of_a_registered_source_is_served(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        adapter = server.sources.materialize(sid)
        assert server.sources.get_registered(sid) is adapter

    def test_an_unknown_source_is_none_not_an_error(self, tmp_path):
        manager, server = _manager(tmp_path)
        assert server.sources.get_registered("nope") is None

    def test_the_catalog_url_covers_pending_and_registered_sources(self, tmp_path):
        for name in ("a.zarr", "b.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        first, second = _only_ids(server)
        reconciler = manager._reconciler

        pending = reconciler.catalog_url_of(second)
        assert pending.endswith(os.path.basename(reconciler.claim_primary_path(second)))

        server.sources.materialize(first)
        assert reconciler.catalog_url_of(first) == server.sources.get(first).catalog_url
        assert reconciler.catalog_url_of("nope") is None

    def test_unregistered_sources_counts_pending_and_failed(self, tmp_path):
        for name in ("a.zarr", "b.zarr"):
            _make_zarr(tmp_path, name)
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        first, second = _only_ids(server)
        assert manager.unregistered_sources() == 2

        server.sources.materialize(first)
        assert manager.unregistered_sources() == 1
        assert len(server.sources) == 1

    def test_plain_get_never_registers(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        assert server.sources.get(sid) is None
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
                target=lambda: results.append(server.sources.materialize(sid))
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1
        assert len({id(a) for a in results}) == 1
        assert results[0] is not None


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
            os.path.basename(manager._reconciler.claim_primary_path(sid)): sid
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
                    adapter = server.sources.materialize(sid)
                    assert adapter is not None
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

    def test_a_paused_pool_registers_nothing_until_resumed(self):
        seen = []
        worker = RegistrationWorker(lambda sid: seen.append(sid) or True, workers=2)
        worker.pause()
        worker.start()
        try:
            worker.enqueue("a", 1.0)
            worker.enqueue("b", 2.0)
            time.sleep(0.2)
            assert seen == []
            assert worker.queued() == 2
            worker.resume()
            _wait_until(lambda: len(seen) == 2)
        finally:
            worker.stop()

    def test_a_paused_pool_still_stops(self):
        worker = RegistrationWorker(lambda sid: True, workers=2)
        worker.pause()
        worker.start()
        worker.enqueue("a")
        started = time.monotonic()
        worker.stop(join_timeout=2.0)
        assert time.monotonic() - started < 1.0
        assert all(not t.is_alive() for t in worker._threads)

    def test_the_first_scan_walks_before_any_source_registers(self, tmp_path):
        for i in range(6):
            _make_zarr(tmp_path, f"s{i}.zarr")
        manager, server = _manager(tmp_path, workers=2)
        worker = manager._registration_worker
        inner = worker._register
        done_when_registered = []
        worker._register = lambda sid: (
            done_when_registered.append(manager._initial_scan_done),
            inner(sid),
        )[1]

        manager.start()
        try:
            _wait_until(lambda: manager.pending_registrations() == 0)
            _wait_until(lambda: manager.registration_idle())
            assert len(done_when_registered) == 6
            # No registration began before the first scan was over.
            assert all(done_when_registered)
            # A later scan does not hold the pool again.
            manager._handle_rescan()
            assert worker._paused is False
        finally:
            manager.stop()

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

        with pytest.raises(SourceRegistrationError, match="could not be registered"):
            server.sources.materialize(sid)

        row = _rows(server)[sid]
        assert row["is_resolved"] is False
        assert row["unresolved_reason"] == "failed"
        assert json.loads(row["metadata_json"])["registration_error"]
        # Nothing is waiting on it, and nothing serves it.
        assert manager.pending_registrations() == 0
        assert server.sources.get(sid) is None
        assert manager._reconciler.is_pending(sid)

    def test_a_failed_source_is_not_retried_by_every_read(self, tmp_path):
        path = _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        self._break(path)
        assert not manager._reconciler.ensure_registered(sid)

        reconciler = manager._reconciler
        attempts = []
        real = reconciler._register_source_claim
        reconciler._register_source_claim = lambda *a, **k: (
            attempts.append(1),
            real(*a, **k),
        )[1]
        for _ in range(2):
            with pytest.raises(SourceRegistrationError):
                server.sources.materialize(sid)
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
        assert not manager._reconciler.ensure_registered(sid)

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
        assert not manager._reconciler.ensure_registered(sid)
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

    def test_a_source_removed_while_queued_does_not_stay_in_deferred(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        assert sid in manager._deferred

        assert manager._reconciler._commit_remove_source(sid)
        # What the registration worker runs for a queued source.
        assert manager._register_pending(sid)

        assert sid not in manager._deferred

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
        assert server.sources.get(sid) is not None


class TestPrecacheRouting:
    def test_a_registered_source_reaches_the_backlog_hook(self, tmp_path):
        _make_zarr(tmp_path, "a.zarr")
        manager, server = _manager(tmp_path)
        routed = []
        manager.set_startup_source_hook(lambda sid, mtime: routed.append((sid, mtime)))
        live = []
        manager.set_source_committed_hook(live.append)
        _first_scan(manager)
        (sid,) = _only_ids(server)
        assert routed == []

        server.sources.materialize(sid)

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
        server.sources.materialize(_only_ids(server)[0])
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
        server.set_unregistered_provider(manager.unregistered_sources)
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
            health = client.health_check()
            assert health["registration_pending"] == 2
            # No adapter is registered yet; the catalog still has both sources.
            assert health["source_count"] == 2
        finally:
            client.close()
            server.shutdown()

    def test_reading_a_pending_source_asks_for_a_resolve(self, tmp_path):
        manager, server, client = self._serve(tmp_path)
        try:
            first, second = _only_ids(server)

            with pytest.raises(ValueError, match="resolve_source"):
                client.get_tensor(first)
            assert manager._reconciler.is_pending(first)

            client.resolve_source(first)
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

    @staticmethod
    def _wrap_raw(client, **overrides):
        """Replace the connection with one that answers ``overrides`` itself and
        forwards everything else; returns the real connection."""
        state = client._catalog._state
        raw = state.raw_client

        class Wrapped:
            def __getattr__(self, name):
                return overrides.get(name) or getattr(raw, name)

        state.raw_client = Wrapped()
        return raw

    def test_a_listing_projection_carries_the_reason(self, tmp_path):
        manager, server, client = self._serve(tmp_path)
        try:
            columns = client.source_row_columns()
            assert columns.endswith("unresolved_reason")
            rows = client.query(f"SELECT {columns} FROM sources", format="records")
            assert [r["unresolved_reason"] for r in rows] == ["pending", "pending"]
        finally:
            client.close()
            server.shutdown()

    @pytest.mark.parametrize(
        "error, remembered",
        [("FlightUnauthorizedError", True), ("FlightUnavailableError", False)],
    )
    def test_a_failed_schema_read_projects_the_base_columns(
        self, tmp_path, error, remembered
    ):
        """A refusal is remembered; a dropped call is asked again."""
        from biopb.tensor._catalog_rows import SOURCE_ROW_COLUMNS
        from pyarrow import flight

        manager, server, client = self._serve(tmp_path)
        try:

            def get_flight_info(*a, **k):
                raise getattr(flight, error)("no")

            self._wrap_raw(client, get_flight_info=get_flight_info)

            assert client.source_row_columns() == SOURCE_ROW_COLUMNS
            state = client._catalog._state
            assert (state.catalog_columns is not None) is remembered
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

    def test_the_sdk_refuses_a_pending_source_until_it_is_resolved(self, tmp_path):
        """``get_source_metadata`` and ``get_descriptor`` refuse an unresolved row
        without registering anything: only ``resolve_source`` does."""
        manager, server, client = self._serve(tmp_path, count=3)
        try:
            first, second, third = _only_ids(server)

            for call in (client.get_source_metadata, client.get_descriptor):
                with pytest.raises(ValueError, match="resolve_source"):
                    call(first)
            assert manager._reconciler.is_pending(first)

            client.resolve_source(first)
            assert client.get_source_metadata(first) is not None
            desc = client.get_descriptor(first)
            assert list(desc.shape) == [2, 8, 8]
            assert manager._reconciler.is_pending(second) is True
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
            with pytest.raises(ValueError, match="resolve_source"):
                client.get_descriptor(sid)
            with pytest.raises(Exception, match="could not be registered"):
                client.resolve_source(sid)
        finally:
            client.close()
            server.shutdown()


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


class TestReasonsFromRows:
    def test_with_reason_appends_only_when_the_schema_has_it(self):
        from biopb.tensor._catalog_rows import with_reason

        assert (
            with_reason("a, b", {"a", "unresolved_reason"}) == "a, b, unresolved_reason"
        )
        assert with_reason("a, b", {"a"}) == "a, b"

    def test_a_projected_reason_needs_no_query(self):
        from biopb.tensor._catalog_rows import reasons_for

        rows = [
            {"source_id": "x", "is_resolved": False, "unresolved_reason": "pending"},
            {"source_id": "y", "is_resolved": True, "unresolved_reason": None},
        ]

        def never(sql):
            raise AssertionError(sql)

        assert reasons_for(rows, never) == {"x": "pending", "y": None}

    def test_without_the_column_it_asks_once_and_only_for_unresolved_rows(self):
        from biopb.tensor._catalog_rows import reasons_for

        asked = []

        def ask(sql):
            asked.append(sql)
            return [{"source_id": "x", "unresolved_reason": "failed"}]

        resolved = [{"source_id": "y", "is_resolved": True}]
        pending = [{"source_id": "x", "is_resolved": False}]
        assert reasons_for(resolved, ask) == {} and not asked
        assert reasons_for(pending, ask) == {"x": "failed"} and len(asked) == 1
