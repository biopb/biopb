"""Regression tests for periodic SourceManager reconciliation."""

from pathlib import Path

import pytest

from biopb_tensor_server.discovery import DiscoveryState, SourceClaim, generate_source_id
from biopb_tensor_server.watcher import WatcherEvent, WatcherEventType
from biopb_tensor_server.source_manager import SourceManager


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

    def register_source(self, source_id, adapter):
        self.registered.append(source_id)

    def unregister_source(self, source_id):
        self.unregistered.append(source_id)


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


class TestSourceManagerRegressions:
    def setup_method(self):
        _FlakyAdapter.calls = 0

    def test_process_event_ignores_legacy_event_types(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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

    def test_skipped_stable_subtree_preserves_existing_claims(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        stable_dir = monitored_dir / "stable"
        stable_dir.mkdir()
        data_path = stable_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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

    def test_full_rescan_backstop_recovers_stale_skipped_subtree(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
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

        def fake_refresh(force_full=False):
            refresh_calls["count"] += 1
            if refresh_calls["count"] == 1:
                manager._skipped_stable_dirs = {str(monitored_dir.resolve())}
                return manager._skipped_stable_dirs
            return original_refresh(force_full=force_full)

        monkeypatch.setattr(manager, "_refresh_entry_state", fake_refresh)

        clock["now"] = 101.0
        manager._handle_rescan()
        assert source_id in state.claims

        clock["now"] = 111.0
        manager._handle_rescan()

        assert source_id not in state.claims
        assert server.unregistered == [source_id]

    def test_process_event_dispatches_rescan(self, tmp_path, monkeypatch):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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

    def test_reconcile_keeps_unstable_missing_source(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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

    def test_reconcile_changed_source_removes_old_source_when_readd_fails(
        self, tmp_path
    ):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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
        manager = SourceManager(
            server=server,
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