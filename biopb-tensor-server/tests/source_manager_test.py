"""Regression tests for periodic SourceManager reconciliation."""

import os
import time
from pathlib import Path

import pytest
from biopb_tensor_server.discovery import (
    DiscoveryState,
    SourceClaim,
    generate_source_id,
)
from biopb_tensor_server.source_manager import SourceManager
from biopb_tensor_server.watcher import WatcherEvent, WatcherEventType


class _FakeAdapter:
    @classmethod
    def claim(cls, ctx, state):
        if not ctx.is_file() or not ctx.path_str.endswith(".dat"):
            return None
        if not state.try_claim_path(ctx.path_str):
            return None
        return SourceClaim(source_type="fake", primary_path=ctx.path_str)

    @classmethod
    def create_from_config(cls, source_config, credentials_config=None):
        return {"url": source_config.url, "type": source_config.type}


class _FakeMetadataDb:
    def __init__(self):
        self.added = []
        self.removed = []

    def sync_source_added(self, source_id, adapter):
        self.added.append(source_id)

    def sync_source_removed(self, source_id):
        self.removed.append(source_id)


class _FailingMetadataDb(_FakeMetadataDb):
    def __init__(self, fail_add=False, fail_remove=False):
        super().__init__()
        self._fail_add = fail_add
        self._fail_remove = fail_remove

    def sync_source_added(self, source_id, adapter):
        if self._fail_add:
            raise RuntimeError("metadata add failed")
        super().sync_source_added(source_id, adapter)

    def sync_source_removed(self, source_id):
        if self._fail_remove:
            raise RuntimeError("metadata remove failed")
        super().sync_source_removed(source_id)


class _FakeServer:
    def __init__(self):
        self.registered = []
        self.unregistered = []
        self._metadata_db = _FakeMetadataDb()
        # Progressive-discovery freshness signals, recorded for assertions.
        self.full_scan_in_progress = False
        self.scan_in_progress_history = []
        self.last_full_scan_at = None

    def register_source(self, source_id, adapter):
        self.registered.append(source_id)

    def unregister_source(self, source_id):
        self.unregistered.append(source_id)

    def set_full_scan_in_progress(self, in_progress):
        self.full_scan_in_progress = bool(in_progress)
        self.scan_in_progress_history.append(bool(in_progress))

    def set_last_full_scan(self, timestamp):
        self.last_full_scan_at = float(timestamp)


class _FailingRegisterServer(_FakeServer):
    def register_source(self, source_id, adapter):
        super().register_source(source_id, adapter)
        raise RuntimeError("register failed")


class _FailingUnregisterServer(_FakeServer):
    def unregister_source(self, source_id):
        raise RuntimeError("unregister failed")


class _FakeRegistry:
    def get_claims_for_path(self, ctx, state):
        claim = _FakeAdapter.claim(ctx, state)
        return [claim] if claim is not None else []

    def get_adapter_for_type(self, source_type):
        if source_type == "fake":
            return _FakeAdapter
        return None


class _FailingAdapter:
    @classmethod
    def create_from_config(cls, source_config, credentials_config=None):
        raise RuntimeError("adapter create failed")


class _RegistryWithFailingAdapter(_FakeRegistry):
    def get_adapter_for_type(self, source_type):
        if source_type == "fake":
            return _FailingAdapter
        return None


class _FlakyAdapter:
    calls = 0

    @classmethod
    def claim(cls, ctx, state):
        return _FakeAdapter.claim(ctx, state)

    @classmethod
    def create_from_config(cls, source_config, credentials_config=None):
        cls.calls += 1
        raise RuntimeError("transient adapter failure")


class _RegistryWithFlakyAdapter(_FakeRegistry):
    def get_adapter_for_type(self, source_type):
        if source_type == "fake":
            return _FlakyAdapter
        return None


def _make_manager(server, **kwargs):
    """Construct a SourceManager wired to the fake server's metadata DB.

    Centralizes the ``metadata_db=server._metadata_db`` injection so individual
    tests don't repeat it; all other SourceManager kwargs pass straight through.
    """
    return SourceManager(server=server, metadata_db=server._metadata_db, **kwargs)


class TestSourceManagerRegressions:
    def setup_method(self):
        _FlakyAdapter.calls = 0

    def test_process_event_ignores_legacy_event_types(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        ignored_event = WatcherEvent(
            event_type=WatcherEventType.CREATED,
            path=monitored_dir / "ignored.dat",
            is_directory=False,
        )

        manager._process_event(ignored_event)

        assert state.claims == {}
        assert server.registered == []

    def test_periodic_rescan_adds_and_removes_sources(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        manager._handle_rescan()

        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]
        assert server._metadata_db.added == [source_id]

        data_path.unlink()
        manager._handle_rescan()

        assert state.claims == {}
        assert server.unregistered == [source_id]
        assert server._metadata_db.removed == [source_id]

    def test_first_rescan_does_not_cycle_unchanged_sources(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        assert manager._commit_add_claim(claim) is True

        server.registered.clear()
        server.unregistered.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        manager._handle_rescan()

        assert server.unregistered == []
        assert server.registered == []
        assert server._metadata_db.added == []
        assert server._metadata_db.removed == []

    def test_skipped_stable_subtree_preserves_existing_claims(
        self, tmp_path, monkeypatch
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        stable_dir = monitored_dir / "stable"
        stable_dir.mkdir()
        data_path = stable_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))
        server.registered.clear()
        server.unregistered.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        clock["now"] = 101.0
        manager._handle_rescan()

        assert str(stable_dir.resolve()) in manager._skipped_stable_dirs
        assert source_id in state.claims
        assert server.registered == []
        assert server.unregistered == []
        assert server._metadata_db.added == []
        assert server._metadata_db.removed == []

    def test_deleted_monitored_root_removes_claims_and_stops_monitoring(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))

        data_path.unlink()
        monitored_dir.rmdir()

        manager._handle_rescan()

        assert source_id not in state.claims
        assert monitored_dir not in manager._monitored_dirs
        assert server.unregistered == [source_id]
        assert server._metadata_db.removed == [source_id]

    def test_full_rescan_backstop_recovers_stale_skipped_subtree(
        self, tmp_path, monkeypatch
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
            full_rescan_interval=10.0,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))
        data_path.unlink()

        original_refresh = manager._refresh_entry_state
        refresh_calls = {"count": 0}

        def fake_refresh(force_full=False, publish=True):
            refresh_calls["count"] += 1
            if refresh_calls["count"] == 1:
                manager._skipped_stable_dirs = {str(monitored_dir.resolve())}
                return (
                    manager._entry_state,
                    manager._entry_stable_observations,
                    manager._entry_pending_scan,
                    manager._skipped_stable_dirs,
                    {},  # next_cloud (empty: no cloud roots in this test)
                )
            return original_refresh(force_full=force_full, publish=publish)

        monkeypatch.setattr(manager, "_refresh_entry_state", fake_refresh)

        clock["now"] = 101.0
        manager._handle_rescan()
        assert source_id in state.claims

        clock["now"] = 111.0
        manager._handle_rescan()

        assert source_id not in state.claims
        assert server.unregistered == [source_id]

    def test_failed_rescan_preserves_previous_entry_cache(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        manager._handle_rescan()
        previous_entry_state = dict(manager._entry_state)
        previous_stable_observations = dict(manager._entry_stable_observations)
        previous_pending_scan = dict(manager._entry_pending_scan)
        previous_skipped_dirs = set(manager._skipped_stable_dirs)

        data_path.write_text("changed")
        monkeypatch.setattr(
            manager,
            "_reconcile_discovered_state",
            lambda discovered_state, unstable_paths: (_ for _ in ()).throw(
                RuntimeError("reconcile failed")
            ),
        )

        with pytest.raises(RuntimeError, match="reconcile failed"):
            manager._handle_rescan()

        assert manager._entry_state == previous_entry_state
        assert manager._entry_stable_observations == previous_stable_observations
        assert manager._entry_pending_scan == previous_pending_scan
        assert manager._skipped_stable_dirs == previous_skipped_dirs

    def test_process_event_dispatches_rescan(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        calls = []
        monkeypatch.setattr(manager, "_handle_rescan", lambda: calls.append("rescan"))

        manager._process_event(WatcherEvent(WatcherEventType.RESCAN, monitored_dir))

        assert calls == ["rescan"]

    def test_should_scan_path_respects_stability_window(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=30.0,
            probe_open_files=False,
        )

        manager._entry_state[str(data_path.resolve())] = (
            False,
            (1, 2, 3, 4, 5),
            100.0,
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("biopb_tensor_server.source_manager.time.time", lambda: 110.0)
            assert manager._should_scan_path(data_path) is False
            mp.setattr("biopb_tensor_server.source_manager.time.time", lambda: 131.0)
            assert manager._should_scan_path(data_path) is True

    def test_should_scan_path_uses_open_probe_for_files(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=True,
        )

        manager._entry_state[str(data_path.resolve())] = (
            False,
            (1, 2, 3, 4, 5),
            100.0,
        )
        monkeypatch.setattr(manager, "_can_open_for_append", lambda path: False)

        assert manager._should_scan_path(data_path) is False

    def test_should_scan_path_can_require_multiple_stable_rescans(
        self, tmp_path, monkeypatch
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=30.0,
            stable_rescans_required=2,
            probe_open_files=False,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._refresh_entry_state()
        assert manager._should_scan_path(data_path) is False

        clock["now"] = 131.0
        manager._refresh_entry_state()
        assert manager._should_scan_path(data_path) is False

        clock["now"] = 162.0
        manager._refresh_entry_state()
        assert manager._should_scan_path(data_path) is True

    def test_aggressive_dir_pruning_can_skip_monitored_root(
        self, tmp_path, monkeypatch
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
            aggressive_dir_pruning=True,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))
        server.registered.clear()
        server.unregistered.clear()

        clock["now"] = 101.0
        manager._handle_rescan()

        assert str(monitored_dir.resolve()) in manager._skipped_stable_dirs
        assert source_id in state.claims
        assert server.registered == []
        assert server.unregistered == []

    def test_non_aggressive_root_rescan_after_stability_delay(
        self, tmp_path, monkeypatch
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=30.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
            aggressive_dir_pruning=False,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()

        clock["now"] = 131.0
        manager._handle_rescan()
        assert str(monitored_dir.resolve()) not in manager._skipped_stable_dirs

        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        clock["now"] = 132.0
        manager._handle_rescan()
        assert state.claims == {}
        assert str(monitored_dir.resolve()) not in manager._skipped_stable_dirs

        clock["now"] = 163.0
        manager._handle_rescan()

        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]
        assert str(monitored_dir.resolve()) not in manager._skipped_stable_dirs

    def test_aggressive_root_rescan_uses_child_fs_age_after_skipped_root_change(
        self, tmp_path, monkeypatch
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=30.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
            aggressive_dir_pruning=True,
        )

        base_time = time.time()
        clock = {"now": base_time}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()

        clock["now"] = base_time + 31.0
        manager._handle_rescan()

        clock["now"] = base_time + 62.0
        manager._handle_rescan()
        assert str(monitored_dir.resolve()) in manager._skipped_stable_dirs

        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")
        changed_at = base_time + 63.0
        changed_ns = int(changed_at * 1_000_000_000)
        os.utime(data_path, ns=(changed_ns, changed_ns))
        os.utime(monitored_dir, ns=(changed_ns, changed_ns))

        clock["now"] = changed_at
        manager._handle_rescan()
        assert state.claims == {}

        clock["now"] = base_time + 94.0
        manager._handle_rescan()

        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]

    def test_pruned_root_keeps_walking_while_copied_file_settles(
        self, tmp_path, monkeypatch
    ):
        """biopb/biopb#53: a copy into an already-pruned aggressive root must
        not be frozen out while its files are still settling.

        Once the root is pruned its own mtime stops changing (writes deep in the
        subtree don't bump it), so before the fix the root was re-pruned every
        cycle and the still-young copied file never got discovered. The prune
        gate must refuse while a descendant is pending, then resume once it
        settles.
        """
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=30.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
            aggressive_dir_pruning=True,
        )

        base_time = time.time()
        clock = {"now": base_time}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        # Settle and prune the (empty) root.
        manager._handle_rescan()
        clock["now"] = base_time + 31.0
        manager._handle_rescan()
        clock["now"] = base_time + 62.0
        manager._handle_rescan()
        assert str(monitored_dir.resolve()) in manager._skipped_stable_dirs

        # Simulate `cp -r newdir/` landing under the pruned root: creating the
        # new child bumps the root's mtime, creating the file bumps newdir's.
        newdir = monitored_dir / "newdir"
        newdir.mkdir()
        data_path = newdir / "a.dat"
        data_path.write_text("partial")
        appeared_at = base_time + 63.0
        appeared_ns = int(appeared_at * 1_000_000_000)
        os.utime(data_path, ns=(appeared_ns, appeared_ns))
        os.utime(newdir, ns=(appeared_ns, appeared_ns))
        os.utime(monitored_dir, ns=(appeared_ns, appeared_ns))

        # The bumped root is walked once, but the file is too young to claim.
        clock["now"] = appeared_at
        manager._handle_rescan()
        assert state.claims == {}

        # The file keeps being written (its mtime advances) but neither newdir
        # nor the root mtime changes. Before the fix the root's signature now
        # looks stable and old enough, so it would be re-pruned and the pending
        # file frozen; the fix refuses the prune while the file is pending.
        writing_at = base_time + 70.0
        writing_ns = int(writing_at * 1_000_000_000)
        data_path.write_text("more data written")
        os.utime(data_path, ns=(writing_ns, writing_ns))

        clock["now"] = base_time + 93.0
        manager._handle_rescan()
        assert str(monitored_dir.resolve()) not in manager._skipped_stable_dirs
        assert state.claims == {}

        # Once the file is quiet >= stability_window it is discovered.
        clock["now"] = base_time + 101.0
        manager._handle_rescan()
        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]

        # With nothing left pending, pruning resumes — the guard is not a
        # permanent disable.
        clock["now"] = base_time + 102.0
        manager._handle_rescan()
        assert str(monitored_dir.resolve()) in manager._skipped_stable_dirs
        assert source_id in state.claims

    def test_pruned_nested_dir_keeps_walking_while_copied_file_settles(
        self, tmp_path, monkeypatch
    ):
        """biopb/biopb#53, default config: child directories are always pruned
        (allow_prune hard-coded True for children), so the same freeze occurs one
        level down even with aggressive_dir_pruning disabled.
        """
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        sub = monitored_dir / "sub"
        sub.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=30.0,
            probe_open_files=False,
            full_rescan_interval=0.0,
            aggressive_dir_pruning=False,
        )

        base_time = time.time()
        clock = {"now": base_time}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        # Settle and prune the (empty) nested subdirectory.
        manager._handle_rescan()
        clock["now"] = base_time + 31.0
        manager._handle_rescan()
        clock["now"] = base_time + 62.0
        manager._handle_rescan()
        assert str(sub.resolve()) in manager._skipped_stable_dirs

        # Copy a file into the settled subdir: its own mtime bumps once, then
        # settles while the file is still being written.
        data_path = sub / "a.dat"
        data_path.write_text("partial")
        appeared_at = base_time + 63.0
        appeared_ns = int(appeared_at * 1_000_000_000)
        os.utime(data_path, ns=(appeared_ns, appeared_ns))
        os.utime(sub, ns=(appeared_ns, appeared_ns))

        clock["now"] = appeared_at
        manager._handle_rescan()
        assert state.claims == {}

        writing_at = base_time + 70.0
        writing_ns = int(writing_at * 1_000_000_000)
        data_path.write_text("more data written")
        os.utime(data_path, ns=(writing_ns, writing_ns))

        clock["now"] = base_time + 93.0
        manager._handle_rescan()
        assert str(sub.resolve()) not in manager._skipped_stable_dirs
        assert state.claims == {}

        clock["now"] = base_time + 101.0
        manager._handle_rescan()
        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]

        clock["now"] = base_time + 102.0
        manager._handle_rescan()
        assert str(sub.resolve()) in manager._skipped_stable_dirs
        assert source_id in state.claims

    def test_reconcile_keeps_unstable_missing_source(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        assert manager._commit_add_claim(claim) is True
        server.unregistered.clear()
        server._metadata_db.removed.clear()

        manager._reconcile_discovered_state(
            DiscoveryState(),
            unstable_paths=[data_path.resolve()],
        )

        assert claim.source_id in state.claims
        assert server.unregistered == []
        assert server._metadata_db.removed == []

    def test_commit_add_claim_fails_when_adapter_creation_raises(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_RegistryWithFailingAdapter(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        claim = SourceClaim(source_type="fake", primary_path=str(data_path.resolve()))

        assert manager._commit_add_claim(claim) is False
        assert state.claims == {}
        assert server.registered == []
        assert server.unregistered == []

    def test_commit_add_claim_rolls_back_when_state_add_fails(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        first_path = monitored_dir / "first.dat"
        second_path = monitored_dir / "second.dat"
        first_path.write_text("hello")
        second_path.write_text("world")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        first_claim = SourceClaim(
            source_type="fake",
            primary_path=str(first_path.resolve()),
            source_id=generate_source_id(str(first_path.resolve()), "fake"),
            member_paths={str(first_path.resolve())},
        )
        assert manager._commit_add_claim(first_claim) is True

        conflicting_claim = SourceClaim(
            source_type="fake",
            primary_path=str(second_path.resolve()),
            source_id="different_source_id",
            member_paths={str(first_path.resolve()), str(second_path.resolve())},
        )

        server.registered.clear()
        server.unregistered.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        assert manager._commit_add_claim(conflicting_claim) is False
        assert state.claims == {first_claim.source_id: first_claim}
        assert server.registered == ["different_source_id"]
        assert server.unregistered == ["different_source_id"]
        assert server._metadata_db.removed == ["different_source_id"]

    def test_commit_add_claim_rolls_back_when_metadata_sync_fails(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        server._metadata_db = _FailingMetadataDb(fail_add=True)
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        claim = SourceClaim(source_type="fake", primary_path=str(data_path.resolve()))

        assert manager._commit_add_claim(claim) is False
        assert state.claims == {}
        assert len(server.registered) == 1
        assert len(server.unregistered) == 1

    def test_commit_remove_source_returns_false_when_unregister_fails(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FailingUnregisterServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        state.add_claim(claim, notify=False)

        assert manager._commit_remove_source(claim.source_id) is False
        assert claim.source_id in state.claims

    def test_unregister_source_claim_completes_when_metadata_remove_fails(
        self, tmp_path
    ):
        """A catalog-delete failure must not skip the server-unregister or the
        _path_to_source_id cleanup (issue #223 follow-up). The removal still
        completes; only the leaked catalog row is logged.
        """
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        server._metadata_db = _FailingMetadataDb(fail_remove=True)
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        assert manager._commit_add_claim(claim) is True
        assert claim.primary_path in manager._path_to_source_id

        # The catalog DELETE raises, but the removal still completes.
        assert manager._commit_remove_source(claim.source_id) is True
        assert server.unregistered == [claim.source_id]
        # Path map cleaned -- no stale entry to mislead a later re-add/reconcile.
        assert claim.primary_path not in manager._path_to_source_id
        assert claim.source_id not in state.claims

    def test_reconcile_changed_source_removes_old_source_when_readd_fails(
        self, tmp_path
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))

        server.registered.clear()
        server.unregistered.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        manager._registry = _RegistryWithFailingAdapter()
        data_path.write_text("hello world")

        manager._handle_rescan()

        assert state.claims == {}
        assert server.registered == []
        assert server.unregistered == [source_id]
        assert server._metadata_db.added == []
        assert server._metadata_db.removed == [source_id]
        assert source_id not in manager._source_signatures

    def test_rollback_source_registration_survives_rollback_errors(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FailingUnregisterServer()
        server._metadata_db = _FailingMetadataDb(fail_remove=True)
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )
        manager._path_to_source_id[str(monitored_dir)] = "source-1"

        manager._rollback_source_registration("source-1")

        assert manager._path_to_source_id == {}

    def test_failed_dataset_retries_with_backoff(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_RegistryWithFlakyAdapter(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()
        source_id = generate_source_id(str(data_path.resolve()), "fake")

        assert _FlakyAdapter.calls == 1
        assert manager._failed_sources[source_id].attempts == 1

        clock["now"] = 100.5
        manager._handle_rescan()

        assert _FlakyAdapter.calls == 1
        assert manager._failed_sources[source_id].attempts == 1

        clock["now"] = 101.1
        manager._handle_rescan()

        assert _FlakyAdapter.calls == 2
        assert manager._failed_sources[source_id].attempts == 2
        assert manager._failed_sources[source_id].next_retry_at == pytest.approx(103.1)

    def test_failed_dataset_logs_are_rate_limited(self, tmp_path, monkeypatch, caplog):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_RegistryWithFlakyAdapter(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.source_manager.time.time", lambda: clock["now"]
        )

        caplog.set_level("ERROR")

        manager._handle_rescan()
        clock["now"] = 101.1
        manager._handle_rescan()
        clock["now"] = 104.0
        manager._handle_rescan()

        error_records = [
            record
            for record in caplog.records
            if "Failed to create adapter for source" in record.message
        ]
        assert len(error_records) == 1

        clock["now"] = 132.0
        manager._handle_rescan()

        error_records = [
            record
            for record in caplog.records
            if "Failed to create adapter for source" in record.message
        ]
        assert len(error_records) == 2


def _make_signature_manager(monitored_dirs):
    """A SourceManager wired only enough to exercise the signature scan."""
    return _make_manager(
        _FakeServer(),
        registry=_FakeRegistry(),
        discovery_state=DiscoveryState(),
        watcher=None,
        monitored_dirs=set(monitored_dirs),
        stability_window=0.0,
        probe_open_files=False,
    )


def _scan(manager):
    """Run one signature refresh and return its {resolved_path_str: ...} map."""
    next_state, _obs, _pending, _skipped, _cloud = manager._refresh_entry_state(
        force_full=True, publish=False
    )
    return next_state


class TestSignatureScanLoopAndSkip:
    """The signature/stability scan (_scan_tree_state) must share the claim
    walk's skip policy and never wedge on a directory loop. Symlink-correctness
    alone is not enough: junctions / hardlinks / bind mounts don't present as
    symlinks, so the real guard is filesystem-identity dedup.
    """

    def _symlink_or_skip(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform/filesystem")

    def test_terminates_on_symlink_cycle(self, tmp_path):
        # root/loop -> root is a self-cycle. Before the fix, _scan_tree_state
        # checked is_symlink() on the *resolved* path (always False), so the
        # symlink guard never fired and the cycle recursed to RecursionError.
        root = tmp_path / "monitored"
        (root / "real").mkdir(parents=True)
        (root / "real" / "sample.dat").write_text("hi")
        self._symlink_or_skip(root / "loop", root)

        manager = _make_signature_manager({root})
        next_state = _scan(manager)  # must not raise RecursionError

        assert str((root / "real" / "sample.dat").resolve()) in next_state

    def test_does_not_follow_symlink_out_of_tree(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.dat").write_text("not ours")

        root = tmp_path / "monitored"
        root.mkdir()
        (root / "sample.dat").write_text("ours")
        self._symlink_or_skip(root / "escape", outside)

        next_state = _scan(_make_signature_manager({root}))

        assert str((root / "sample.dat").resolve()) in next_state
        # The symlinked dir is recorded but never descended into.
        assert str((outside / "secret.dat").resolve()) not in next_state

    def test_prunes_system_and_offline_entries(self, tmp_path):
        # Parity with walk_with_identity_tracking: the signature scan must also
        # skip system/cloud dirs and offline placeholders, by the same policy.
        root = tmp_path / "monitored"
        (root / "Microscopy").mkdir(parents=True)
        (root / "Microscopy" / "good.dat").write_text("data")
        (root / "AppData" / "Local").mkdir(parents=True)
        (root / "AppData" / "Local" / "junk.dat").write_text("junk")
        (root / "OneDrive - Lab").mkdir(parents=True)
        (root / "OneDrive - Lab" / "cloud.dat").write_text("cloud")

        stub = root / "placeholder.dat"
        with open(stub, "wb") as fh:
            fh.truncate(4 * 1024 * 1024)
        # getattr: native Windows os.stat has no st_blocks, so the offline
        # sub-check below is skipped there while the system-dir pruning still runs.
        offline_supported = getattr(os.stat(stub), "st_blocks", None) == 0

        next_state = _scan(_make_signature_manager({root}))

        assert str((root / "Microscopy" / "good.dat").resolve()) in next_state
        assert str((root / "AppData").resolve()) not in next_state
        assert (
            str((root / "AppData" / "Local" / "junk.dat").resolve()) not in next_state
        )
        assert str((root / "OneDrive - Lab").resolve()) not in next_state
        if offline_supported:
            assert str(stub.resolve()) not in next_state

    def test_overlapping_roots_dedup_by_identity(self, tmp_path):
        # Two roots where one nests inside the other: identity dedup (no symlink
        # involved) must stop the shared subtree being walked twice, and the
        # scan must still capture the data file.
        root = tmp_path / "monitored"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / "sample.dat").write_text("hi")

        next_state = _scan(_make_signature_manager({root, sub}))

        assert str((sub / "sample.dat").resolve()) in next_state

    def _defeat_identity_dedup(self, monkeypatch):
        # Make get_file_identity return a fresh value on every call, so the
        # visited_identities loop guard NEVER fires -- exactly the cloud/zeroed-inode
        # junction failure mode (its path-hash identity grows with the loop, so no two
        # descents share an identity). Without the depth cap this recurses forever.
        import itertools

        from biopb_tensor_server import source_manager as sm_module

        counter = itertools.count()
        monkeypatch.setattr(
            sm_module, "get_file_identity", lambda *a, **k: f"id-{next(counter)}"
        )
        return sm_module

    def test_depth_cap_breaks_loop_when_identity_dedup_fails(
        self, tmp_path, monkeypatch
    ):
        # With identity dedup defeated, a tree deeper than the cap must still
        # terminate (no RecursionError that the os.scandir except OSError can't catch
        # and that would kill the refresh thread) and must stop descending at the cap.
        sm_module = self._defeat_identity_dedup(monkeypatch)
        monkeypatch.setattr(sm_module, "_MAX_WALK_DEPTH", 4)

        root = tmp_path / "monitored"
        deep = root
        for i in range(8):  # deeper than the cap of 4
            deep = deep / f"d{i}"
        deep.mkdir(parents=True)
        (deep / "sample.dat").write_text("hi")

        next_state = _scan(_make_signature_manager({root}))  # must not raise

        # Descent stopped at the cap: the below-cap data file is never reached.
        assert str((deep / "sample.dat").resolve()) not in next_state
        # ...but entries within the cap were still recorded.
        assert str((root / "d0").resolve()) in next_state

    def test_depth_cap_logs_warning(self, tmp_path, monkeypatch, caplog):
        sm_module = self._defeat_identity_dedup(monkeypatch)
        monkeypatch.setattr(sm_module, "_MAX_WALK_DEPTH", 3)

        root = tmp_path / "monitored"
        deep = root
        for i in range(6):
            deep = deep / f"d{i}"
        deep.mkdir(parents=True)
        (deep / "sample.dat").write_text("hi")

        with caplog.at_level("WARNING"):
            _scan(_make_signature_manager({root}))

        assert any("max depth" in rec.message for rec in caplog.records)


class _FakeStat:
    """Minimal stand-in for ``os.stat_result`` with a controllable ``st_ino``.

    Lets us simulate the Windows ``DirEntry.stat()`` behaviour (st_ino/st_dev
    reported as zero) on any platform, so the cloud inode-backfill skip is
    testable on the POSIX CI box where a real stat would never return inode 0.
    """

    def __init__(
        self,
        *,
        st_ino,
        st_dev=0,
        mode=0o100644,
        size=10,
        mtime_ns=111,
        ctime_ns=222,
        st_blocks=8,
    ):
        self.st_ino = st_ino
        self.st_dev = st_dev
        self.st_mode = mode
        self.st_size = size
        self.st_mtime_ns = mtime_ns
        self.st_ctime_ns = ctime_ns
        self.st_mtime = mtime_ns / 1_000_000_000
        self.st_ctime = ctime_ns / 1_000_000_000
        self.st_blocks = st_blocks


class _FakeDirEntry:
    """``os.DirEntry`` stand-in exposing only what ``_scan_tree_state`` reads."""

    def __init__(self, path, stat_result):
        self.path = str(path)
        self._stat = stat_result

    def is_symlink(self):
        return False

    def stat(self):
        return self._stat


class TestCloudInodeBackfillSkip:
    """Windows ``DirEntry.stat()`` zeroes ``st_ino``, so ``_scan_tree_state``
    backfills it with a real ``os.stat``. Under a cloud root that backfill is an
    extra whole network round-trip per entry, so it is skipped (biopb/biopb#190,
    Finding 1). This is correct ONLY because the cloud signature is identity-only
    and degrades safely to ``(0, 0)``; these tests pin that contract.
    """

    def _scan_one(self, manager, tmp_path, *, cloud, monkeypatch):
        f = tmp_path / "sample.dat"
        f.write_text("x")
        entry = _FakeDirEntry(f, _FakeStat(st_ino=0))

        backfill_calls = []

        def _counting_stat(p, *a, **k):
            backfill_calls.append(str(p))
            return _FakeStat(st_ino=4242, st_dev=7)

        monkeypatch.setattr(os, "stat", _counting_stat)

        next_state, next_cloud = {}, {}
        manager._scan_tree_state(
            path=f,
            now=time.time(),
            next_state=next_state,
            next_stable_observations={},
            next_pending_scan={},
            next_cloud=next_cloud,
            skipped_dirs=set(),
            force_full=True,
            allow_prune=True,
            visited_identities=set(),
            is_root=False,
            dir_entry=entry,
            cloud=cloud,
        )
        return str(f), next_state, next_cloud, backfill_calls

    def test_cloud_skips_backfill_and_degrades_signature(self, tmp_path, monkeypatch):
        manager = _make_signature_manager({tmp_path})
        path_str, next_state, next_cloud, backfill_calls = self._scan_one(
            manager, tmp_path, cloud=True, monkeypatch=monkeypatch
        )

        # The whole point: no second os.stat -> no extra cloud round-trip.
        assert backfill_calls == []
        # The entry is still recorded, with the signature degraded to the
        # identity-only (0, 0) (residency-invariant, never flaps on hydrate).
        assert path_str in next_state
        _is_dir, signature, _changed = next_state[path_str]
        assert signature == (0, 0)
        assert next_cloud[path_str] is True

    def test_non_cloud_still_backfills_inode(self, tmp_path, monkeypatch):
        manager = _make_signature_manager({tmp_path})
        path_str, next_state, next_cloud, backfill_calls = self._scan_one(
            manager, tmp_path, cloud=False, monkeypatch=monkeypatch
        )

        # Non-cloud Windows must still pay the backfill: the real inode is
        # load-bearing for the full mtime/size signature and NTFS identity (#56).
        assert backfill_calls == [path_str]
        assert path_str in next_state
        _is_dir, signature, _changed = next_state[path_str]
        assert signature[:2] == (7, 4242)  # backfilled (st_dev, st_ino)
        assert next_cloud[path_str] is False

    def test_get_file_identity_path_hash_fallback_distinguishes_zero_inode(
        self, tmp_path
    ):
        # Condition 3 of the skip: with a zeroed inode, get_file_identity falls
        # back to hashing the resolved path, so distinct cloud entries still get
        # distinct identities and visited_identities dedup does not collapse the
        # walk into a single (0, 0) bucket.
        from biopb_tensor_server.discovery import get_file_identity

        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        id_a = get_file_identity(a, _FakeStat(st_ino=0, mode=0o040755))
        id_b = get_file_identity(b, _FakeStat(st_ino=0, mode=0o040755))

        assert id_a != id_b


class TestProgressiveDiscoveryFreshness:
    """Health freshness signals pushed by _handle_rescan (progressive #212)."""

    def _manager_with_source(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        (monitored_dir / "sample.dat").write_text("hello")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
        )
        return server, manager

    def test_first_full_rescan_pushes_freshness_and_fires_callback(self, tmp_path):
        server, manager = self._manager_with_source(tmp_path)
        fired = []
        manager._on_initial_scan_complete = lambda: fired.append(True)

        manager._handle_rescan()

        # in_progress was raised for the scan and lowered when it finished.
        assert server.scan_in_progress_history == [True, False]
        assert server.full_scan_in_progress is False
        assert server.last_full_scan_at is not None
        assert manager._initial_scan_done is True
        assert fired == [True]

    def test_incremental_rescan_leaves_freshness_untouched(self, tmp_path):
        server, manager = self._manager_with_source(tmp_path)
        fired = []
        manager._on_initial_scan_complete = lambda: fired.append(True)

        manager._handle_rescan()  # first: force-full
        first_ts = server.last_full_scan_at

        # Immediately after, the default 3600s interval makes the next rescan
        # incremental -> it must not toggle in_progress, advance the timestamp,
        # or re-fire the one-shot completion callback.
        manager._handle_rescan()

        assert server.scan_in_progress_history == [True, False]
        assert server.last_full_scan_at == first_ts
        assert fired == [True]

    def test_failed_first_scan_resets_progress_and_retries(self, tmp_path, monkeypatch):
        server, manager = self._manager_with_source(tmp_path)
        fired = []
        manager._on_initial_scan_complete = lambda: fired.append(True)

        # Fail the reconcile of the first (force-full) scan.
        def boom(*a, **k):
            raise RuntimeError("reconcile failed")

        monkeypatch.setattr(manager, "_reconcile_discovered_state", boom)
        with pytest.raises(RuntimeError, match="reconcile failed"):
            manager._handle_rescan()

        # in_progress reset, no timestamp, gate still closed, callback not fired.
        assert server.scan_in_progress_history == [True, False]
        assert server.full_scan_in_progress is False
        assert server.last_full_scan_at is None
        assert manager._initial_scan_done is False
        assert fired == []

        # Next tick retries (still force-full since the timestamp never advanced)
        # and succeeds.
        monkeypatch.undo()
        manager._handle_rescan()
        assert manager._initial_scan_done is True
        assert server.last_full_scan_at is not None
        assert fired == [True]


class TestProgressiveStreaming:
    """Option B: the first scan streams adds within the walk (progressive #212)."""

    def _manager(self, tmp_path, n_sources, **kw):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        for i in range(n_sources):
            (monitored_dir / f"s{i}.dat").write_text(f"data{i}")

        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
            probe_open_files=False,
            **kw,
        )
        return server, manager

    def test_first_scan_streams_adds_before_reconcile(self, tmp_path, monkeypatch):
        server, manager = self._manager(tmp_path, n_sources=3)

        # Capture how many sources were already registered when reconcile starts.
        # Under Option B every first-scan add is streamed *during* the walk, so
        # the catalog is already full before the end-of-walk reconcile runs.
        seen_at_reconcile = []
        orig_reconcile = manager._reconcile_discovered_state

        def spy(discovered_state, unstable_paths):
            seen_at_reconcile.append(len(server.registered))
            return orig_reconcile(discovered_state, unstable_paths)

        monkeypatch.setattr(manager, "_reconcile_discovered_state", spy)

        manager._handle_rescan()

        assert seen_at_reconcile == [3]  # all 3 registered before reconcile
        assert len(server.registered) == 3
        # Reconcile did not re-add them (idempotent): exactly one register each.
        assert len(server.registered) == len(set(server.registered))

    def test_steady_state_rescan_does_not_stream(self, tmp_path, monkeypatch):
        # After the first scan, a force-full steady-state rescan must NOT stream
        # (its on_source_added stays unset) -- adds go through batch reconcile.
        server, manager = self._manager(tmp_path, n_sources=2)
        manager._handle_rescan()  # first scan: _initial_scan_done -> True

        # Force the next rescan to be a full one and add a new source.
        manager._last_full_rescan_at = float("-inf")
        (tmp_path / "monitored" / "s2.dat").write_text("data2")

        seen_at_reconcile = []
        orig_reconcile = manager._reconcile_discovered_state

        def spy(discovered_state, unstable_paths):
            # The new source is NOT yet registered when reconcile starts: it is
            # added by reconcile (batch), not streamed during the walk.
            seen_at_reconcile.append(len(server.registered))
            return orig_reconcile(discovered_state, unstable_paths)

        monkeypatch.setattr(manager, "_reconcile_discovered_state", spy)
        manager._handle_rescan()

        assert seen_at_reconcile == [2]  # only the original two before reconcile
        assert len(server.registered) == 3

    def test_retried_first_scan_does_not_unregister_streamed_sources(
        self, tmp_path, monkeypatch
    ):
        # A first scan that streams its sources then fails (before flipping
        # _initial_scan_done) must, on retry, NOT re-commit -> never hit the
        # duplicate-rollback that would unregister them.
        server, manager = self._manager(tmp_path, n_sources=2)

        calls = {"n": 0}
        orig_reconcile = manager._reconcile_discovered_state

        def flaky(discovered_state, unstable_paths):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("reconcile failed")
            return orig_reconcile(discovered_state, unstable_paths)

        monkeypatch.setattr(manager, "_reconcile_discovered_state", flaky)

        with pytest.raises(RuntimeError, match="reconcile failed"):
            manager._handle_rescan()

        # Both sources were streamed before the failure and remain registered.
        assert len(server.registered) == 2
        assert manager._initial_scan_done is False

        # Retry (still force-full: timestamp never advanced) completes cleanly
        # without re-registering or unregistering anything.
        manager._handle_rescan()
        assert manager._initial_scan_done is True
        assert len(server.registered) == 2  # not 4
        assert server.unregistered == []

    def test_unstable_first_scan_claim_is_not_streamed(self, tmp_path):
        # stable_rescans_required=2: a fresh entry needs two unchanged
        # observations before it is eligible, so the first scan defers it and
        # streams nothing -- the same stability gate the batch path uses.
        server, manager = self._manager(
            tmp_path, n_sources=1, stable_rescans_required=2
        )

        manager._handle_rescan()  # first (force-full) scan

        assert server.registered == []  # deferred, not streamed
        assert manager._initial_scan_done is True  # scan still completed
        assert server.last_full_scan_at is not None

        # Not lost: once stable, a later rescan registers it (batch path).
        for _ in range(5):
            if server.registered:
                break
            manager._handle_rescan()
        assert len(server.registered) == 1


class _CatalogStubAdapter:
    """Adapter whose descriptor/metadata the real MetadataDatabase can index.

    Unlike _FakeAdapter (returns a bare dict), this implements the
    get_source_descriptor()/get_metadata() surface that
    MetadataDatabase.sync_source_added reads, so a static source can flow
    through the real registration + catalog-sync path.
    """

    def __init__(self, source_id, source_url):
        self._source_id = source_id
        self._source_url = source_url

    @classmethod
    def create_from_config(cls, source_config, credentials_config=None):
        return cls(source_config.source_id, source_config.url)

    def get_source_descriptor(self):
        from biopb.tensor.descriptor_pb2 import (
            DataSourceDescriptor,
            TensorDescriptor,
        )

        return DataSourceDescriptor(
            source_id=self._source_id,
            source_url=self._source_url,
            source_type="zarr",
            data_resident=True,
            tensors=[
                TensorDescriptor(array_id=self._source_id, shape=[8, 8], dtype="uint8")
            ],
        )

    def get_metadata(self):
        return {}


class _CatalogStubRegistry(_FakeRegistry):
    def get_adapter_for_type(self, source_type):
        return _CatalogStubAdapter


class TestStaticCatalogSeeding:
    """Guard for issue #223 item 1: static sources populate the metadata DB via
    the normal registration path, so the vestigial initial_sync is unnecessary.
    """

    def test_static_sources_populate_catalog_without_initial_sync(self, tmp_path):
        from biopb_tensor_server.config import SourceConfig
        from biopb_tensor_server.metadata_db import MetadataDatabase
        from biopb_tensor_server.source_manager import create_source_manager

        db = MetadataDatabase(enabled=True)
        server = _FakeServer()
        static = SourceConfig(url=str(tmp_path / "plate.zarr"), type="zarr")

        manager = create_source_manager(
            server=server,
            registry=_CatalogStubRegistry(),
            watcher=None,
            static_sources=[static],
            metadata_db=db,
        )

        assert manager is not None
        # No initial_sync was called -- the catalog is already populated.
        rows = db._get_connection().execute("SELECT source_id FROM sources").fetchall()
        assert [r[0] for r in rows] == [static.source_id]
        assert server.registered == [static.source_id]


class TestRescanCarryForwardPrefix:
    """The per-rescan carry-forward and pending-scan scans match descendants by an
    exact path-prefix (``root + os.sep``) over the cached entry set -- not a
    pathlib ``Path(cached_path).is_relative_to(root)`` per entry. The pathlib
    variant parsed a ``Path`` for every one of (potentially tens of thousands of)
    carried-forward cloud entries on every 30s incremental rescan, holding the GIL
    and stalling reads (biopb/biopb). These pin the cheap-string behavior and the
    exact-prefix boundary a bare ``startswith(root)`` would get wrong.
    """

    def _manager(self, tmp_path):
        return SourceManager(
            server=_FakeServer(),
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs={tmp_path / "data"},
            stability_window=0.0,
            probe_open_files=False,
        )

    def test_copy_cached_subtree_entries_matches_exact_prefix(self, tmp_path):
        mgr = self._manager(tmp_path)
        root = str(tmp_path / "data")
        child = str(tmp_path / "data" / "sub" / "f.tif")
        deep = str(tmp_path / "data" / "a.tif")
        # Sibling sharing the *string* prefix "data" but NOT under "data/" -- the
        # case a bare startswith(root) (no separator) would wrongly include.
        decoy = str(tmp_path / "data_backup" / "x.tif")
        unrelated = str(tmp_path / "other" / "y.tif")
        sig = (False, (0, 0), 0.0)
        mgr._entry_state = {
            root: (True, (0, 0), 0.0),
            child: sig,
            deep: sig,
            decoy: sig,
            unrelated: sig,
        }
        mgr._entry_stable_observations = {child: 3, deep: 2}
        mgr._entry_pending_scan = {child: True}

        next_state = {root: (True, (0, 0), 0.0)}  # root already recorded by the walk
        nso: dict = {}
        nps: dict = {}
        mgr._copy_cached_subtree_entries(root, next_state, nso, nps)

        assert set(next_state) - {root} == {child, deep}
        assert decoy not in next_state  # exact-prefix guard (root + os.sep)
        assert unrelated not in next_state
        assert nso == {child: 3, deep: 2}
        assert nps[child] is True
        assert nps[deep] is False  # default when no pending record exists

    def test_subtree_has_pending_scan_matches_exact_prefix(self, tmp_path):
        mgr = self._manager(tmp_path)
        root = str(tmp_path / "data")

        # bare-prefix sibling must not count
        mgr._entry_pending_scan = {str(tmp_path / "data_backup" / "x.tif"): True}
        assert mgr._subtree_has_pending_scan(root) is False
        # a real pending descendant counts
        mgr._entry_pending_scan = {str(tmp_path / "data" / "sub" / "f.tif"): True}
        assert mgr._subtree_has_pending_scan(root) is True
        # the root's own pending flag does not count (the prune gate handles it)
        mgr._entry_pending_scan = {root: True}
        assert mgr._subtree_has_pending_scan(root) is False
        # a non-pending descendant does not count
        mgr._entry_pending_scan = {str(tmp_path / "data" / "sub" / "f.tif"): False}
        assert mgr._subtree_has_pending_scan(root) is False
