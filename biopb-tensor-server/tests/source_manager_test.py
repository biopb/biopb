"""Regression tests for SourceManager event handling."""

import threading
from pathlib import Path

from biopb_tensor_server.discovery import DiscoveryState, SourceClaim
from biopb_tensor_server.source_manager import SourceManager


class _FakeServer:
    def __init__(self):
        self.unregistered = []

    def register_source(self, source_id, adapter):
        pass

    def unregister_source(self, source_id):
        self.unregistered.append(source_id)


class _FakeRegistry:
    pass


class TestSourceManagerRegressions:
    def test_directory_delete_does_not_deadlock_while_event_loop_lock_held(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        deleted_dir = monitored_dir / "imagesTs"
        deleted_dir.mkdir(parents=True)

        claimed_path = deleted_dir / "sample.ome.zarr"
        claim = SourceClaim(
            source_type="ome-zarr",
            primary_path=str(claimed_path),
            source_id="ome_zarr_deadlock_case",
        )

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
        )

        state.claims[claim.source_id] = claim
        state.path_to_source[claim.primary_path] = claim.source_id
        state.consumed_paths.add(claim.primary_path)

        completed = threading.Event()

        def delete_from_event_loop_context():
            with manager._lock:
                manager._handle_deleted(deleted_dir, is_directory=True)
            completed.set()

        worker = threading.Thread(target=delete_from_event_loop_context, daemon=True)
        worker.start()
        worker.join(timeout=1)

        assert completed.is_set(), "directory delete handling deadlocked"
        assert claim.source_id not in state.claims
        assert state.get_source_for_path(claim.primary_path) is None
        assert server.unregistered == [claim.source_id]

    def test_remove_callback_runs_outside_manager_lock(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        claimed_path = monitored_dir / "sample.ome.zarr"

        claim = SourceClaim(
            source_type="ome-zarr",
            primary_path=str(claimed_path),
            source_id="ome_zarr_callback_case",
        )

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
        )

        state.claims[claim.source_id] = claim
        state.path_to_source[claim.primary_path] = claim.source_id
        state.consumed_paths.add(claim.primary_path)

        callback_observed_unlocked = []

        def on_source_removed(source_id):
            acquired = manager._lock.acquire(blocking=False)
            callback_observed_unlocked.append(acquired)
            if acquired:
                manager._lock.release()

        state.on_source_removed = on_source_removed

        manager._handle_deleted(claimed_path, is_directory=False)

        assert callback_observed_unlocked == [True]
        assert claim.source_id not in state.claims