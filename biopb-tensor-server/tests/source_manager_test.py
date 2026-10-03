"""Regression tests for periodic SourceManager reconciliation."""

import os
import threading
import time
from datetime import datetime, timedelta

import pytest
from biopb_tensor_server.core.discovery import (
    DiscoveryState,
    SourceClaim,
    generate_source_id,
)
from biopb_tensor_server.sources.roots import RootKind

from tests import make_manager


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


class _ClosingAdapter:
    """An adapter that records ``close()``, so a leaked one is visible."""

    built = []

    def __init__(self, url):
        self.url = url
        self.closed = 0
        _ClosingAdapter.built.append(self)

    def close(self):
        self.closed += 1


class _ClosingAdapterFactory:
    @classmethod
    def claim(cls, ctx, state):
        return _FakeAdapter.claim(ctx, state)

    @classmethod
    def create_from_config(cls, source_config, credentials_config=None):
        return _ClosingAdapter(source_config.url)


class _FakeMetadataDb:
    def __init__(self):
        self.added = []
        self.removed = []
        # The orphan clock, driven from _mark_catalog_complete.
        self.seen_calls = 0
        self.pruned = []

    def mark_sources_seen(self):
        self.seen_calls += 1
        return 0

    def prune_unseen(self, before):
        self.pruned.append(before)
        return 0

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
        self.swapped = []
        self.sources = {}
        self._metadata_db = _FakeMetadataDb()
        # Progressive-discovery freshness signals, recorded for assertions.
        self.full_scan_in_progress = False
        self.scan_in_progress_history = []
        self.last_full_scan_at = None

    def register_source(self, source_id, adapter):
        self.registered.append(source_id)
        self.sources[source_id] = adapter

    def swap_source(self, source_id, adapter):
        self.swapped.append(source_id)
        displaced = self.sources.get(source_id)
        self.sources[source_id] = adapter
        return adapter, displaced

    def unregister_source(self, source_id):
        self.unregistered.append(source_id)
        self.sources.pop(source_id, None)

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


class _ClosingRegistry(_FakeRegistry):
    def get_adapter_for_type(self, source_type):
        if source_type == "fake":
            return _ClosingAdapterFactory
        return None


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
    return make_manager(server=server, metadata_db=server._metadata_db, **kwargs)


class TestOrphanClockSeam:
    """The sweep hangs off scan completion, which is where completeness is held."""

    @staticmethod
    def _manager(server, tmp_path, **kwargs):
        return _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs={tmp_path / "data"},
            stability_window=0.0,
            **kwargs,
        )

    def test_a_completed_scan_records_presence(self, tmp_path):
        server = _FakeServer()
        manager = self._manager(server, tmp_path)
        manager.complete_initial_scan()
        assert server._metadata_db.seen_calls == 1
        # Freshness is still published; the clock rides along, it does not replace it.
        assert server.last_full_scan_at is not None

    def test_a_young_server_never_prunes_however_many_scans_it_finishes(self, tmp_path):
        # The scan count is not the question. An upstream that is down at boot
        # is usually still down an hour later, so gating on the first scan
        # alone would let the second one delete its annotations.
        server = _FakeServer()
        manager = self._manager(server, tmp_path, prune_unseen_days=7)
        for _ in range(5):
            manager._mark_catalog_complete()
        assert server._metadata_db.seen_calls == 5
        assert server._metadata_db.pruned == []

    def test_pruning_arms_once_the_server_has_outlived_the_threshold(self, tmp_path):
        server = _FakeServer()
        manager = self._manager(server, tmp_path, prune_unseen_days=7)
        manager._mark_catalog_complete()
        assert server._metadata_db.pruned == []

        # Seven days of uptime, less a second.
        manager._started_at -= 7 * 86400 - 1
        manager._mark_catalog_complete()
        assert server._metadata_db.pruned == []

        manager._started_at -= 2
        manager._mark_catalog_complete()
        assert len(server._metadata_db.pruned) == 1
        age = datetime.now() - server._metadata_db.pruned[0]
        assert timedelta(days=7) <= age < timedelta(days=7, seconds=30)

    def test_prune_is_off_by_default(self, tmp_path):
        server = _FakeServer()
        manager = self._manager(server, tmp_path)
        manager._started_at -= 10 * 365 * 86400  # a decade of uptime
        manager._mark_catalog_complete()
        assert server._metadata_db.pruned == []
        # ... and the sweep still runs; it is the deleting that is opt-in.
        assert server._metadata_db.seen_calls == 1

    def test_a_failing_clock_does_not_fail_the_scan(self, tmp_path):
        server = _FakeServer()

        def _boom():
            raise RuntimeError("catalog is wedged")

        server._metadata_db.mark_sources_seen = _boom
        manager = self._manager(server, tmp_path)
        manager.complete_initial_scan()
        # Freshness published and the startup gate flipped regardless.
        assert server.last_full_scan_at is not None
        assert manager._initial_scan_done


class TestRescanLoop:
    """The rescan timer the manager runs itself (no watcher object)."""

    @staticmethod
    def _manager(server, monitored_dirs, **kwargs):
        return _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs=monitored_dirs,
            stability_window=0.0,
            **kwargs,
        )

    def test_the_monitored_walk_tells_claims_it_is_monitored(
        self, tmp_path, monkeypatch
    ):
        """A monitored root is walked again every tick, so its claims may keep
        their content probes."""
        import biopb_tensor_server.sources.source_manager as sm

        seen = []
        real = sm.discover_sources

        def spy(*args, **kwargs):
            seen.append(kwargs.get("monitored"))
            return real(*args, **kwargs)

        monkeypatch.setattr(sm, "discover_sources", spy)
        root = tmp_path / "data"
        root.mkdir()
        (root / "a.dat").write_text("a")
        manager = self._manager(_FakeServer(), {root})

        manager._handle_rescan()

        assert seen == [True]

    def test_a_config_with_nothing_to_scan_still_completes_on_the_first_tick(
        self, tmp_path
    ):
        """A static-only config has no tree to walk, but the loop still starts and
        its first tick completes the startup protocol (freshness, precache gate,
        completion hook) -- there is no separate launcher path for it."""
        server = _FakeServer()
        manager = self._manager(server, set())
        fired = threading.Event()
        manager.set_initial_scan_complete_hook(fired.set)
        manager._rescan_interval = 3600.0
        try:
            manager.start()
            assert manager.is_running() is True
            assert fired.wait(5)
        finally:
            manager.stop()
        assert manager._initial_scan_done is True
        assert server.last_full_scan_at is not None

    def test_a_non_positive_interval_is_floored_not_a_hot_loop_or_off(self, tmp_path):
        """The loop always runs; an interval at or below zero is the same mistake
        as a tiny positive one, and is floored the same way."""
        manager = self._manager(_FakeServer(), set(), rescan_interval=0)
        assert manager._rescan_interval == 0.1

    def test_the_loop_rescans_on_the_interval(self, tmp_path, monkeypatch):
        """First tick immediately, then one per interval."""
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        manager = self._manager(_FakeServer(), {monitored_dir}, rescan_interval=0.01)

        rescans = threading.Semaphore(0)
        monkeypatch.setattr(manager, "_handle_rescan", rescans.release)
        try:
            manager.start()
            assert manager.is_running() is True
            for _ in range(3):
                assert rescans.acquire(timeout=5)
        finally:
            manager.stop(join_timeout=5)
        assert manager.is_running() is False

    def test_a_failing_rescan_does_not_kill_the_loop(self, tmp_path, monkeypatch):
        """The next pass sees the same tree, so a failure is logged and retried."""
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        manager = self._manager(_FakeServer(), {monitored_dir}, rescan_interval=0.01)

        attempts = threading.Semaphore(0)

        def boom():
            attempts.release()
            raise RuntimeError("scan blew up")

        monkeypatch.setattr(manager, "_handle_rescan", boom)
        try:
            manager.start()
            for _ in range(2):
                assert attempts.acquire(timeout=5)
        finally:
            manager.stop(join_timeout=5)

    def test_stop_does_not_wait_out_the_interval(self, tmp_path, monkeypatch):
        """The loop waits on an Event, so stop() returns rather than sleeping off
        a 30s interval -- which is what the shutdown budget depends on."""
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        manager = self._manager(_FakeServer(), {monitored_dir}, rescan_interval=300.0)

        first = threading.Event()
        monkeypatch.setattr(manager, "_handle_rescan", first.set)
        manager.start()
        assert first.wait(timeout=5)  # the immediate first tick ran

        started = time.monotonic()
        manager.stop(join_timeout=5)
        assert time.monotonic() - started < 2.0
        assert manager.is_running() is False


class TestScanOnceRoots:
    """A ``monitor = false`` directory is discovered by the first tick, once."""

    @staticmethod
    def _manager(server, root, state, **kwargs):
        from biopb_tensor_server.core.config import SourceConfig

        sources = [SourceConfig(url=str(root), monitor=False, **kwargs)]
        return _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs=set(),
            scan_once_sources=sources,
            stability_window=0.0,
        )

    def test_first_tick_registers_and_completes_the_startup_protocol(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "a.dat").write_text("a")
        server = _FakeServer()
        state = DiscoveryState()
        manager = self._manager(server, root, state)

        manager._handle_rescan()

        assert len(state.claims) == 1
        assert len(server.registered) == 1
        # Nothing else in this config completes a scan, so this tick must.
        assert manager._initial_scan_done is True
        assert server.last_full_scan_at is not None
        assert server.full_scan_in_progress is False

    def test_it_is_not_told_it_is_monitored(self, tmp_path, monkeypatch):
        """It is walked once, so its claims keep no content probes."""
        import biopb_tensor_server.sources.source_manager as sm

        seen = []
        real = sm.discover_sources

        def spy(*args, **kwargs):
            seen.append(kwargs.get("monitored", False))
            return real(*args, **kwargs)

        monkeypatch.setattr(sm, "discover_sources", spy)
        root = tmp_path / "data"
        root.mkdir()
        (root / "a.dat").write_text("a")
        manager = self._manager(_FakeServer(), root, DiscoveryState())

        manager._handle_rescan()

        assert seen == [False]

    def test_it_is_never_scanned_again(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "a.dat").write_text("a")
        server = _FakeServer()
        state = DiscoveryState()
        manager = self._manager(server, root, state)
        manager._handle_rescan()

        (root / "b.dat").write_text("b")
        manager._handle_rescan()

        assert len(state.claims) == 1

    def test_what_it_registered_survives_a_rescan_that_did_not_see_it(self, tmp_path):
        """Its claims sit outside every monitored root, so a monitored walk's
        removal diff never reaches them."""
        root = tmp_path / "data"
        other = tmp_path / "watched"
        root.mkdir()
        other.mkdir()
        (root / "a.dat").write_text("a")
        (other / "w.dat").write_text("w")
        server = _FakeServer()
        state = DiscoveryState()
        from biopb_tensor_server.core.config import SourceConfig

        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs={other},
            scan_once_sources=[SourceConfig(url=str(root), monitor=False)],
            stability_window=0.0,
        )

        manager._handle_rescan()
        manager._handle_rescan()

        assert len(state.claims) == 2
        assert server.unregistered == []

    def test_a_vanished_root_is_skipped_and_the_rest_still_scan(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (good / "a.dat").write_text("a")
        from biopb_tensor_server.core.config import SourceConfig

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs=set(),
            scan_once_sources=[
                SourceConfig(url=str(tmp_path / "gone"), monitor=False),
                SourceConfig(url=str(good), monitor=False),
            ],
            stability_window=0.0,
        )

        manager._handle_rescan()

        assert len(state.claims) == 1

    def test_alias_re_roots_what_is_found_under_it(self, tmp_path):
        from types import SimpleNamespace

        class _Registry(_FakeRegistry):
            def get_adapter_for_type(self, source_type):
                return SimpleNamespace(
                    create_from_config=lambda config, creds=None: SimpleNamespace()
                )

        from biopb_tensor_server.core.config import SourceConfig
        from biopb_tensor_server.sources.resolve import _alias_catalog_url

        root = tmp_path / "data"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "a.dat").write_text("a")
        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_Registry(),
            discovery_state=state,
            monitored_dirs=set(),
            scan_once_sources=[SourceConfig(url=str(root), monitor=False, alias="lab")],
            stability_window=0.0,
        )

        manager._handle_rescan()

        claim = next(iter(state.claims.values()))
        adapter = server.sources[claim.source_id]
        assert adapter._catalog_url == _alias_catalog_url(
            "lab", str(root.resolve()), claim.primary_path
        )
        assert adapter._catalog_url.startswith("lab")

    def test_the_loop_starts_for_a_config_that_has_only_one_shot_roots(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        manager = self._manager(_FakeServer(), root, DiscoveryState())
        manager._rescan_interval = 3600.0
        try:
            manager.start()
            assert manager.is_running()
        finally:
            manager.stop()


class _ScriptedRegistry(_FakeRegistry):
    """A fake registry whose ``claim`` can be told to decline, per call."""

    def __init__(self, script=None):
        self.decline = False
        # Per-call overrides, consumed in order (True = decline), before ``decline``.
        self.script = list(script or [])

    def get_claims_for_path(self, ctx, state):
        declined = self.script.pop(0) if self.script else self.decline
        if declined and ctx.path_str.endswith(".dat"):
            return []
        return super().get_claims_for_path(ctx, state)

    def get_adapter_for_type(self, source_type):
        # A drop stamps a display url on the adapter, which a dict cannot take.
        from types import SimpleNamespace

        return SimpleNamespace(
            create_from_config=lambda config, creds=None: SimpleNamespace()
        )


def _drain_drop(manager, path):
    result = None
    for event in manager.add_local_source(str(path)):
        if event[0] == "result":
            result = event[1]
    return result


class TestDropRemoval:
    """A re-drop removes what its walk no longer finds, with a shield."""

    @staticmethod
    def _manager(registry, **kwargs):
        kwargs.setdefault("stability_window", 0.0)
        kwargs.setdefault("monitored_dirs", set())
        server = _FakeServer()
        manager = _make_manager(
            server, registry=registry, discovery_state=DiscoveryState(), **kwargs
        )
        manager.complete_initial_scan()  # drops wait for the first scan
        return server, manager

    def test_a_claim_an_adapter_now_declines_is_removed_though_its_path_exists(
        self, tmp_path
    ):
        # The gap the old existence-only rule left: a sequence worn down to one
        # file stops being claimed, but its directory is still there.
        (tmp_path / "a.dat").write_text("a")
        registry = _ScriptedRegistry()
        server, manager = self._manager(registry)
        assert _drain_drop(manager, tmp_path).added

        registry.decline = True
        result = _drain_drop(manager, tmp_path)

        assert len(result.removed) == 1
        assert server.unregistered == result.removed

    def test_a_claim_that_comes_back_on_a_second_look_is_kept(self, tmp_path):
        # A transient decline: the walk misses it, the re-probe finds it. With no
        # later pass to correct a wrong removal, the probe is the second look.
        (tmp_path / "a.dat").write_text("a")
        registry = _ScriptedRegistry()
        server, manager = self._manager(registry)
        _drain_drop(manager, tmp_path)

        registry.script = [True, False]  # the walk declines, the probe claims
        result = _drain_drop(manager, tmp_path)

        assert result.removed == []
        assert server.unregistered == []

    def test_a_deleted_file_is_removed_without_a_second_look(self, tmp_path):
        data = tmp_path / "a.dat"
        data.write_text("a")
        registry = _ScriptedRegistry()
        server, manager = self._manager(registry)
        _drain_drop(manager, tmp_path)

        data.unlink()
        (tmp_path / "b.dat").write_text("b")  # something else, so the drop proceeds
        result = _drain_drop(manager, tmp_path)

        assert len(result.removed) == 1

    def test_a_source_that_is_still_churning_is_kept(self, tmp_path):
        (tmp_path / "a.dat").write_text("a")
        registry = _ScriptedRegistry()
        server, manager = self._manager(registry, stability_window=10**9)
        _drain_drop(manager, tmp_path)

        registry.decline = True
        result = _drain_drop(manager, tmp_path)

        assert result.removed == []

    def test_a_source_under_a_monitored_root_is_left_to_the_rescan(self, tmp_path):
        (tmp_path / "a.dat").write_text("a")
        registry = _ScriptedRegistry()
        server, manager = self._manager(registry, monitored_dirs={tmp_path})
        _drain_drop(manager, tmp_path)

        registry.decline = True
        result = _drain_drop(manager, tmp_path)

        assert result.removed == []
        assert server.unregistered == []

    def test_a_source_in_a_directory_the_walk_declined_is_kept(self, tmp_path):
        # The skip policy never enters a system/cloud directory, so a source
        # registered there (by dropping it directly) is not "gone" when a later
        # drop of its parent does not find it. Both drops are inside a configured
        # root, which is what lets the parent be dropped over its own child.
        from biopb_tensor_server.core.config import SourceConfig

        hidden = tmp_path / "OneDrive - Lab"
        hidden.mkdir()
        (hidden / "x.dat").write_text("x")
        registry = _ScriptedRegistry()
        server, manager = self._manager(
            registry,
            scan_once_sources=[SourceConfig(url=str(tmp_path), monitor=False)],
        )
        assert _drain_drop(manager, hidden).added

        result = _drain_drop(manager, tmp_path)

        assert result.removed == []
        assert server.unregistered == []


class TestMonitoredRescanSeesDepth:
    """No snapshot to prune against, so a change at any depth is seen next tick."""

    def test_a_dataset_added_three_levels_down_is_found_by_an_incremental_rescan(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "data"
        deep = root / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "old.dat").write_text("old")
        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs={root},
            stability_window=0.0,
        )
        manager._handle_rescan()  # the first, full, pass
        assert len(state.claims) == 1

        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: False)
        (deep / "new.dat").write_text("new")
        manager._handle_rescan()

        assert len(state.claims) == 2

    def test_a_directory_the_gate_defers_does_not_cost_its_quiet_sources(
        self, tmp_path, monkeypatch
    ):
        """A busy directory is not entered this pass; what is registered in it is
        absent from the walk but not gone."""
        root = tmp_path / "data"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / "a.dat").write_text("a")
        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs={root},
            stability_window=0.0,
        )
        manager._handle_rescan()
        assert len(state.claims) == 1

        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: False)
        manager._stability_window = 10**9  # nothing is quiet any more
        for _ in range(3):
            manager._handle_rescan()

        assert len(state.claims) == 1
        assert server.unregistered == []


class _NamespaceRegistry(_FakeRegistry):
    """Builds adapters that can carry a ``_catalog_url``."""

    def get_adapter_for_type(self, source_type):
        from types import SimpleNamespace

        if source_type == "fake":
            return SimpleNamespace(
                create_from_config=lambda config, creds=None: SimpleNamespace()
            )
        return None


class TestMonitoredAlias:
    """A monitored directory's ``alias`` is the display root of what its walk finds."""

    def _manager(self, tmp_path, aliases):
        root = tmp_path / "data"
        (root / "sub").mkdir(parents=True)
        server = _FakeServer()
        adapters = {}
        register = server.register_source
        server.register_source = lambda sid, adapter: (
            adapters.__setitem__(sid, adapter),
            register(sid, adapter),
        )[1]
        manager = _make_manager(
            server,
            registry=_NamespaceRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs={root},
            stability_window=0.0,
            monitored_aliases={root.resolve(): aliases} if aliases else None,
        )
        return root, manager, adapters

    def test_walk_registers_sources_under_the_alias(self, tmp_path):
        root, manager, adapters = self._manager(tmp_path, "lab")
        (root / "a.dat").write_text("a")
        (root / "sub" / "b.dat").write_text("b")

        manager._handle_rescan()

        urls = sorted(a._catalog_url for a in adapters.values())
        assert urls == ["lab/a.dat", "lab/sub/b.dat"]

    def test_a_later_addition_gets_the_alias_too(self, tmp_path, monkeypatch):
        root, manager, adapters = self._manager(tmp_path, "lab")
        (root / "a.dat").write_text("a")
        manager._handle_rescan()
        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: False)

        (root / "sub" / "b.dat").write_text("b")
        manager._handle_rescan()

        assert sorted(a._catalog_url for a in adapters.values()) == [
            "lab/a.dat",
            "lab/sub/b.dat",
        ]

    def test_without_an_alias_the_url_is_native(self, tmp_path):
        root, manager, adapters = self._manager(tmp_path, None)
        (root / "a.dat").write_text("a")

        manager._handle_rescan()

        assert all(getattr(a, "_catalog_url", None) is None for a in adapters.values())

    def test_the_innermost_aliased_root_wins(self, tmp_path):
        from types import SimpleNamespace

        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        manager = _make_manager(
            _FakeServer(),
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs={outer, inner},
            monitored_aliases={outer.resolve(): "o", inner.resolve(): "i"},
        )
        claim = SimpleNamespace(primary_path=str(inner / "x.dat"))
        assert manager._roots.display_url(claim.primary_path) == "i/x.dat"
        claim = SimpleNamespace(primary_path=str(outer / "y.dat"))
        assert manager._roots.display_url(claim.primary_path) == "o/y.dat"
        claim = SimpleNamespace(primary_path=str(tmp_path / "z.dat"))
        assert manager._roots.display_url(claim.primary_path) is None


class TestUnavailableMonitoredRoot:
    """A root that cannot be listed keeps its sources and is walked again later."""

    def _setup(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        (monitored_dir / "sample.dat").write_text("hello")
        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )
        return monitored_dir, server, state, manager

    def test_an_unmounted_root_keeps_its_sources_and_stays_monitored(
        self, tmp_path, caplog
    ):
        monitored_dir, server, state, manager = self._setup(tmp_path)
        manager._handle_rescan()
        source_id = next(iter(state.claims))

        away = tmp_path / "away"
        monitored_dir.rename(away)
        with caplog.at_level("WARNING"):
            for _ in range(4):  # well past the two-miss rule
                manager._handle_rescan()

        assert list(state.claims) == [source_id]
        assert server.unregistered == []
        assert monitored_dir in [
            r.path for r in manager._roots.of_kind(RootKind.MONITORED)
        ]
        assert caplog.text.count("is not available") == 1  # once, not per tick

        away.rename(monitored_dir)
        manager._handle_rescan()

        assert list(state.claims) == [source_id]
        assert server.unregistered == []
        assert server.registered == [source_id]  # nothing was re-registered

    def test_a_root_missing_at_start_is_picked_up_when_it_appears(self, tmp_path):
        monitored_dir, server, state, manager = self._setup(tmp_path)
        away = tmp_path / "away"
        monitored_dir.rename(away)

        manager._handle_rescan()
        assert state.claims == {}

        away.rename(monitored_dir)
        manager._handle_rescan()

        assert len(state.claims) == 1

    def test_a_source_deleted_inside_an_available_root_is_still_removed(self, tmp_path):
        monitored_dir, server, state, manager = self._setup(tmp_path)
        manager._handle_rescan()
        (monitored_dir / "sample.dat").unlink()

        manager._handle_rescan()
        manager._handle_rescan()

        assert state.claims == {}


class TestClaimSpelling:
    """A claim stays under the root its walk found it in, wherever a link points.

    Containment is lexical on the walk's own spelling (a claim is never resolved to
    ask it), and the scan walks a stored root as given, so neither a symlinked file
    nor a root that later becomes a link takes a claim out from under its root.
    """

    @staticmethod
    def _manager(root, **kwargs):
        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs={root},
            stability_window=0.0,
            **kwargs,
        )
        return server, state, manager

    @staticmethod
    def _link_into(root, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "x.dat").write_text("x")
        os.symlink(outside / "x.dat", root / "link.dat")

    def test_a_symlinked_file_is_reconciled_like_any_other(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        root.mkdir()
        (root / "real.dat").write_text("r")
        self._link_into(root, tmp_path)
        server, state, manager = self._manager(root)

        manager._handle_rescan()
        link_claim = next(c for c in state.claims.values() if "link" in c.primary_path)
        assert manager._reconciler._is_monitored_claim(link_claim)
        registered = list(server.registered)
        assert len(registered) == 2

        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: False)
        manager._handle_rescan()
        manager._handle_rescan()

        assert server.registered == registered  # not re-added every pass
        assert server.unregistered == []

    def test_a_symlinked_file_that_goes_away_is_removed(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        root.mkdir()
        self._link_into(root, tmp_path)
        server, state, manager = self._manager(root)
        manager._handle_rescan()
        assert len(state.claims) == 1

        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: False)
        (root / "link.dat").unlink()
        manager._handle_rescan()
        manager._handle_rescan()

        assert state.claims == {}

    def test_a_root_that_becomes_a_link_stays_monitored(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        root.mkdir()
        (root / "a.dat").write_text("a")
        server, state, manager = self._manager(root)
        manager._handle_rescan()
        old_id = generate_source_id(str(root / "a.dat"), "fake")
        assert list(state.claims) == [old_id]

        # A migration moves the directory and leaves a link where it was. The id
        # hashes the resolved location, so the moved file is a new source: the old
        # one is dropped by the two-miss rule and the new one added behind it. A
        # claim the reconcile no longer saw as monitored would never converge.
        moved = tmp_path / "moved"
        root.rename(moved)
        os.symlink(moved, root)
        new_id = generate_source_id(str(root / "a.dat"), "fake")
        assert new_id != old_id
        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: False)
        for _ in range(4):
            manager._reconciler._failed_sources.clear()  # not the retry gate's test
            manager._handle_rescan()

        assert list(state.claims) == [new_id]
        assert set(server.sources) == {new_id}

        (root / "a.dat").unlink()  # and it is still monitored
        manager._handle_rescan()
        manager._handle_rescan()
        assert state.claims == {}

    def test_a_symlinked_file_gets_its_roots_alias(self, tmp_path):
        from types import SimpleNamespace

        root = tmp_path / "data"
        root.mkdir()
        self._link_into(root, tmp_path)
        _, _, manager = self._manager(root, monitored_aliases={root: "lab"})

        claim = SimpleNamespace(primary_path=str(root / "link.dat"))

        assert manager._roots.display_url(claim.primary_path) == "lab/link.dat"

    def test_a_symlinked_file_under_a_cloud_root_is_cloud(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        self._link_into(root, tmp_path)
        _, _, manager = self._manager(root, cloud_roots={root})

        assert manager._roots.is_cloud(str(root / "link.dat"))
        assert manager._reconciler._roots.is_cloud(str(root / "link.dat"))

    def test_a_dropped_link_whose_target_is_gone_is_removed(self, tmp_path):
        drop = tmp_path / "drop"
        drop.mkdir()
        (drop / "keep.dat").write_text("k")
        self._link_into(drop, tmp_path)
        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_ScriptedRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs=set(),
            stability_window=0.0,
        )
        manager.complete_initial_scan()
        assert len(_drain_drop(manager, drop).added) == 2

        (tmp_path / "outside" / "x.dat").unlink()  # the link now dangles
        result = _drain_drop(manager, drop)

        assert len(result.removed) == 1
        assert len(server.sources) == 1


class TestSourceManagerRegressions:
    def setup_method(self):
        _FlakyAdapter.calls = 0

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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        manager._handle_rescan()

        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]
        assert server._metadata_db.added == [source_id]

        data_path.unlink()
        manager._handle_rescan()

        # One miss is not enough: a source can briefly stop being claimable with
        # its files still in place, and removing it now would unregister a working
        # source only to rebuild it a tick later.
        assert list(state.claims) == [source_id]
        assert server.unregistered == []

        manager._handle_rescan()

        assert state.claims == {}
        assert server.unregistered == [source_id]
        assert server._metadata_db.removed == [source_id]

    def test_a_source_found_again_forfeits_its_missed_scans(self, tmp_path):
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )
        manager._handle_rescan()
        source_id = next(iter(state.claims))

        moved = tmp_path / "away.dat"
        data_path.rename(moved)
        manager._handle_rescan()  # one miss
        moved.rename(data_path)
        manager._handle_rescan()  # found again: the count resets
        moved_again = tmp_path / "away2.dat"
        data_path.rename(moved_again)
        manager._handle_rescan()  # one miss again, not two

        assert list(state.claims) == [source_id]
        assert server.unregistered == []

    def test_cloud_misses_count_on_full_passes_only(self, tmp_path, monkeypatch):
        """An incremental tick does not walk a cloud root, so it neither counts a
        miss for its sources nor wipes the count a full pass left."""
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
            monitored_dirs={monitored_dir},
            cloud_roots={monitored_dir.resolve()},
            stability_window=0.0,
        )
        manager._handle_rescan()
        source_id = next(iter(state.claims))
        data_path.unlink()

        full = iter([True, False, False, True])
        monkeypatch.setattr(manager, "_should_force_full_rescan", lambda: next(full))
        manager._handle_rescan()  # full: first miss
        manager._handle_rescan()  # incremental: says nothing
        manager._handle_rescan()  # incremental: says nothing
        assert list(state.claims) == [source_id]

        manager._handle_rescan()  # full: second miss
        assert state.claims == {}
        assert server.unregistered == [source_id]

    def test_a_read_only_file_is_not_refused(self, tmp_path):
        """biopb/biopb#1042: the append probe could not tell "not allowed to
        write" from "still being written", so a file the server can read but not
        write was never discovered. The gate is now signature age alone."""
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")
        os.chmod(data_path, 0o444)

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=state,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        manager._handle_rescan()

        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]

    def test_a_newly_read_only_file_is_not_removed(self, tmp_path, monkeypatch):
        """biopb/biopb#1042: the append probe was a gate condition the removal
        shield did not know about, so a file that later lost write permission was
        deregistered. Gate and shield now share one predicate."""
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
            monitored_dirs={monitored_dir},
            stability_window=30.0,
        )

        base_time = time.time()
        clock = {"now": base_time}
        monkeypatch.setattr(
            "biopb_tensor_server.sources.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()
        clock["now"] = base_time + 31.0
        manager._handle_rescan()

        assert len(state.claims) == 1
        source_id = next(iter(state.claims))
        assert server.registered == [source_id]

        os.chmod(data_path, 0o444)
        # Outlive the chmod's ctime bump so the signature settles again while
        # the file stays permanently unwritable.
        clock["now"] = base_time + 62.0
        manager._handle_rescan()
        clock["now"] = base_time + 93.0
        manager._handle_rescan()

        assert source_id in state.claims
        assert server.unregistered == []

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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        assert manager._reconciler._commit_add_claim(claim) is True

        server.registered.clear()
        server.unregistered.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        manager._handle_rescan()

        assert server.unregistered == []
        assert server.registered == []
        assert server._metadata_db.added == []
        assert server._metadata_db.removed == []

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
            monitored_dirs={monitored_dir},
            # Wide enough that the just-written file is still churning, so the
            # shield must keep it even though the walk found nothing.
            stability_window=3600.0,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        assert manager._reconciler._commit_add_claim(claim) is True
        server.unregistered.clear()
        server._metadata_db.removed.clear()

        manager._reconciler._reconcile_discovered_state(DiscoveryState())

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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        claim = SourceClaim(source_type="fake", primary_path=str(data_path.resolve()))

        assert manager._reconciler._commit_add_claim(claim) is False
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        first_claim = SourceClaim(
            source_type="fake",
            primary_path=str(first_path.resolve()),
            source_id=generate_source_id(str(first_path.resolve()), "fake"),
            member_paths={str(first_path.resolve())},
        )
        assert manager._reconciler._commit_add_claim(first_claim) is True

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

        assert manager._reconciler._commit_add_claim(conflicting_claim) is False
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        claim = SourceClaim(source_type="fake", primary_path=str(data_path.resolve()))

        assert manager._reconciler._commit_add_claim(claim) is False
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        state.add_claim(claim, notify=False)

        assert manager._reconciler._commit_remove_source(claim.source_id) is False
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(data_path.resolve()),
            source_id=generate_source_id(str(data_path.resolve()), "fake"),
            member_paths={str(data_path.resolve())},
        )
        assert manager._reconciler._commit_add_claim(claim) is True
        assert claim.primary_path in manager._reconciler._path_to_source_id

        # The catalog DELETE raises, but the removal still completes.
        assert manager._reconciler._commit_remove_source(claim.source_id) is True
        assert server.unregistered == [claim.source_id]
        # Path map cleaned -- no stale entry to mislead a later re-add/reconcile.
        assert claim.primary_path not in manager._reconciler._path_to_source_id
        assert claim.source_id not in state.claims

    def test_reconcile_changed_source_rebuilds_it_in_place(self, tmp_path):
        """A rewritten file is rebuilt on top of the live adapter, never removed
        and re-added: the source stays in ListFlights and in the catalog for the
        whole rebuild (biopb/biopb#944)."""
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))

        server.registered.clear()
        server.unregistered.clear()
        server.swapped.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        data_path.write_text("hello world")
        manager._handle_rescan()

        assert server.swapped == [source_id]
        assert server.unregistered == []
        assert server._metadata_db.removed == []
        # sync_source_added is an upsert, so the row is replaced in place.
        assert server._metadata_db.added == [source_id]
        assert source_id in state.claims

    def test_a_rebuild_that_does_not_take_closes_the_adapter_it_built(self, tmp_path):
        """The replacement holds the handles it opened. Restoring the previous
        adapter leaves it referenced by nothing, so this call is the only one
        that can release it -- the mirror of the leak a bare ``register`` over a
        live id causes."""
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        data_path = monitored_dir / "sample.dat"
        data_path.write_text("hello")

        server = _FakeServer()
        state = DiscoveryState()
        manager = _make_manager(
            server,
            registry=_ClosingRegistry(),
            discovery_state=state,
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        _ClosingAdapter.built.clear()
        manager._handle_rescan()
        source_id = next(iter(state.claims))
        (serving,) = _ClosingAdapter.built

        # The swap lands, then the catalog write fails.
        manager._reconciler._metadata_db = _FailingMetadataDb(fail_add=True)
        data_path.write_text("hello world")
        manager._handle_rescan()

        built = [a for a in _ClosingAdapter.built if a is not serving]
        assert len(built) == 1, "expected exactly one replacement to be built"
        assert built[0].closed == 1, "the replacement was dropped still open"
        # And the one that was serving is still serving, still open.
        assert server.sources[source_id] is serving
        assert serving.closed == 0

    def test_reconcile_changed_source_keeps_serving_when_rebuild_fails(self, tmp_path):
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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        manager._handle_rescan()
        source_id = next(iter(state.claims))

        server.registered.clear()
        server.unregistered.clear()
        server._metadata_db.added.clear()
        server._metadata_db.removed.clear()

        manager._reconciler._registry = _RegistryWithFailingAdapter()
        data_path.write_text("hello world")

        manager._handle_rescan()

        # The rebuild failed, and that must cost nothing: the previously
        # registered adapter is still the one serving. Remove-then-add used to
        # deregister the source here -- a *rescan request* losing a working
        # source (biopb/biopb#944).
        assert source_id in state.claims
        assert server.registered == []
        assert server.unregistered == []
        assert server._metadata_db.removed == []
        assert source_id in manager._reconciler._source_signatures

    def test_rollback_source_registration_survives_rollback_errors(self, tmp_path):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()

        server = _FailingUnregisterServer()
        server._metadata_db = _FailingMetadataDb(fail_remove=True)
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )
        manager._reconciler._path_to_source_id[str(monitored_dir)] = "source-1"

        manager._reconciler._rollback_source_registration("source-1")

        assert manager._reconciler._path_to_source_id == {}

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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.sources.source_manager.time.time", lambda: clock["now"]
        )

        manager._handle_rescan()
        source_id = generate_source_id(str(data_path.resolve()), "fake")

        assert _FlakyAdapter.calls == 1
        assert manager._reconciler._failed_sources[source_id].attempts == 1

        clock["now"] = 100.5
        manager._handle_rescan()

        assert _FlakyAdapter.calls == 1
        assert manager._reconciler._failed_sources[source_id].attempts == 1

        clock["now"] = 101.1
        manager._handle_rescan()

        assert _FlakyAdapter.calls == 2
        assert manager._reconciler._failed_sources[source_id].attempts == 2
        assert manager._reconciler._failed_sources[
            source_id
        ].next_retry_at == pytest.approx(103.1)

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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )

        clock = {"now": 100.0}
        monkeypatch.setattr(
            "biopb_tensor_server.sources.source_manager.time.time", lambda: clock["now"]
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


def test_get_file_identity_path_hash_fallback_distinguishes_zero_inode(tmp_path):
    """A zeroed inode (Windows ``DirEntry.stat()``, cloud placeholders) falls back to
    hashing the resolved path, so distinct entries still get distinct identities
    and ``visited_identities`` dedup does not collapse a walk into one bucket."""
    from types import SimpleNamespace

    from biopb_tensor_server.core.discovery import get_file_identity

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    zeroed = SimpleNamespace(st_ino=0, st_dev=0, st_mode=0o040755, st_nlink=1)

    assert get_file_identity(a, zeroed) != get_file_identity(b, zeroed)


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
            monitored_dirs={monitored_dir},
            stability_window=0.0,
        )
        return server, manager

    def test_first_full_rescan_pushes_freshness_and_fires_callback(self, tmp_path):
        server, manager = self._manager_with_source(tmp_path)
        fired = []
        manager.set_initial_scan_complete_hook(lambda: fired.append(True))

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
        manager.set_initial_scan_complete_hook(lambda: fired.append(True))

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
        manager.set_initial_scan_complete_hook(lambda: fired.append(True))

        # Fail the reconcile of the first (force-full) scan.
        def boom(*a, **k):
            raise RuntimeError("reconcile failed")

        monkeypatch.setattr(manager._reconciler, "_reconcile_discovered_state", boom)
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

    def test_the_first_tick_drives_the_bootstrap_scan(self, tmp_path):
        server, manager = self._manager_with_source(tmp_path)
        fired = []
        manager.set_initial_scan_complete_hook(lambda: fired.append(True))

        manager._handle_rescan()

        assert server.scan_in_progress_history == [True, False]
        assert server.last_full_scan_at is not None
        assert manager._initial_scan_done is True
        assert fired == [True]

    def test_complete_initial_scan_advances_protocol_without_walking(self, tmp_path):
        # A config with no scan of its own (static sources only): the startup
        # protocol still stamps freshness, flips the gate, and fires the one-shot
        # completion hook.
        server, manager = self._manager_with_source(tmp_path)
        fired = []
        manager.set_initial_scan_complete_hook(lambda: fired.append(True))

        manager.complete_initial_scan()

        # No walk ran; completing the protocol only clears the flag.
        assert server.scan_in_progress_history == [False]
        assert server.last_full_scan_at is not None
        assert manager._initial_scan_done is True
        assert fired == [True]

        # Idempotent: a second call re-stamps freshness but never re-fires.
        manager.complete_initial_scan()
        assert fired == [True]

    def test_initial_scan_completion_is_logged_once(self, tmp_path, caplog):
        # A log-only reader has no other way to tell the first scan is done.
        server, manager = self._manager_with_source(tmp_path)
        with caplog.at_level("INFO"):
            manager.complete_initial_scan()
            manager.complete_initial_scan()
        lines = [
            r.message for r in caplog.records if "Initial scan complete" in r.message
        ]
        assert len(lines) == 1
        assert f"{len(server.sources)} sources" in lines[0]


class TestProgressiveStreaming:
    """Option B: the first scan streams adds within the walk (progressive #212)."""

    def _manager(self, tmp_path, n_sources, **kw):
        monitored_dir = tmp_path / "monitored"
        monitored_dir.mkdir()
        for i in range(n_sources):
            (monitored_dir / f"s{i}.dat").write_text(f"data{i}")

        server = _FakeServer()
        kw.setdefault("stability_window", 0.0)
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs={monitored_dir},
            **kw,
        )
        return server, manager

    def test_first_scan_streams_adds_before_reconcile(self, tmp_path, monkeypatch):
        server, manager = self._manager(tmp_path, n_sources=3)

        # Capture how many sources were already registered when reconcile starts.
        # Under Option B every first-scan add is streamed *during* the walk, so
        # the catalog is already full before the end-of-walk reconcile runs.
        seen_at_reconcile = []
        orig_reconcile = manager._reconciler._reconcile_discovered_state

        def spy(discovered_state, force_full=False):
            seen_at_reconcile.append(len(server.registered))
            return orig_reconcile(discovered_state, force_full=force_full)

        monkeypatch.setattr(manager._reconciler, "_reconcile_discovered_state", spy)

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
        orig_reconcile = manager._reconciler._reconcile_discovered_state

        def spy(discovered_state, force_full=False):
            # The new source is NOT yet registered when reconcile starts: it is
            # added by reconcile (batch), not streamed during the walk.
            seen_at_reconcile.append(len(server.registered))
            return orig_reconcile(discovered_state, force_full=force_full)

        monkeypatch.setattr(manager._reconciler, "_reconcile_discovered_state", spy)
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
        orig_reconcile = manager._reconciler._reconcile_discovered_state

        def flaky(discovered_state, force_full=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("reconcile failed")
            return orig_reconcile(discovered_state, force_full=force_full)

        monkeypatch.setattr(manager._reconciler, "_reconcile_discovered_state", flaky)

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
        # A just-written entry is inside the stability window, so the first scan
        # defers it and streams nothing -- claiming it would register whatever
        # half of the file is on disk.
        server, manager = self._manager(tmp_path, n_sources=1, stability_window=3600.0)

        manager._handle_rescan()  # first (force-full) scan

        assert server.registered == []  # deferred, not streamed
        assert manager._initial_scan_done is True  # scan still completed
        assert server.last_full_scan_at is not None

        # Not lost: once quiet, a later rescan registers it (batch path).
        manager._stability_window = 0.0
        manager._stability_window = 0.0
        for _ in range(5):
            if server.registered:
                break
            manager._handle_rescan()
        assert len(server.registered) == 1


class _CatalogStubAdapter:
    """Adapter whose descriptor/metadata the real MetadataDatabase can index.

    Unlike _FakeAdapter (returns a bare dict), this implements the
    catalog_url / source_type / is_resident / list_tensor_descriptors /
    get_metadata surface that MetadataDatabase.sync_source_added reads, so a
    static source can flow through the real registration + catalog-sync path.
    """

    def __init__(self, source_id, source_url):
        self._source_id = source_id
        self._source_url = source_url

    @classmethod
    def create_from_config(cls, source_config, credentials_config=None):
        return cls(source_config.source_id, source_config.url)

    source_type = "zarr"

    @property
    def catalog_url(self):
        return self._source_url

    def is_resident(self):
        return True

    def is_resolved(self):
        return True

    def list_tensor_descriptors(self):
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        return [TensorDescriptor(array_id=self._source_id, shape=[8, 8], dtype="uint8")]

    def get_metadata(self):
        return {}


class _CatalogStubRegistry(_FakeRegistry):
    def get_adapter_for_type(self, source_type):
        return _CatalogStubAdapter


class TestStaticCatalogSeeding:
    """Guard for issue #223 item 1: static sources populate the metadata DB via
    the normal registration path, so the vestigial initial_sync is unnecessary.
    """

    def test_static_sources_populate_catalog_without_initial_sync(self):
        from biopb_tensor_server.core.config import SourceConfig
        from biopb_tensor_server.serving.metadata_db import MetadataDatabase
        from biopb_tensor_server.sources.source_manager import create_source_manager

        db = MetadataDatabase()
        server = _FakeServer()
        static = SourceConfig(url="s3://bucket/plate.zarr", type="zarr")

        manager = create_source_manager(
            server=server,
            registry=_CatalogStubRegistry(),
            sources=[static],
            metadata_db=db,
        )

        assert manager is not None
        # No initial_sync was called -- the catalog is already populated.
        rows = db._get_connection().execute("SELECT source_id FROM sources").fetchall()
        assert [r[0] for r in rows] == [static.source_id]
        assert server.registered == [static.source_id]


class TestOneStabilityPredicate:
    """The claim gate and the removal shield answer one question, one way.

    "Quiet enough to claim" and "quiet enough to have stopped existing" used to
    be two implementations of the same test, and they drifted: the gate grew an
    append-open probe the shield knew nothing about, so a file the gate refused
    was a file the shield let the diff delete (biopb/biopb#1042).
    """

    @staticmethod
    def _manager(tmp_path, **kwargs):
        monitored = tmp_path / "data"
        monitored.mkdir(exist_ok=True)
        server = _FakeServer()
        manager = _make_manager(
            server,
            registry=_FakeRegistry(),
            discovery_state=DiscoveryState(),
            monitored_dirs={monitored},
            **kwargs,
        )
        return server, manager, monitored

    def test_a_churning_claim_is_shielded_from_removal(self, tmp_path):
        # The shield's job: a claim missing from a walk that is still being
        # written is churning, not gone. Asked of the claim's own paths, so it
        # needs no walk snapshot -- which is what lets a statically registered
        # or drag-dropped claim get a real answer too.
        server, manager, monitored = self._manager(tmp_path, stability_window=0.0)
        data = monitored / "sample.dat"
        data.write_text("hello")
        manager._handle_rescan()
        (claim,) = manager._reconciler._state.claims.values()

        manager._stability_window = 3600.0
        manager._reconciler._stability_window = 3600.0
        data.write_text("still writing")
        assert manager._reconciler._claim_is_quiet(claim) is False

        manager._reconciler._stability_window = 0.0
        assert manager._reconciler._claim_is_quiet(claim) is True

    def test_a_vanished_member_reads_as_gone_not_churning(self, tmp_path):
        # Absence is the case removal exists to act on; it must not be mistaken
        # for "cannot stat, therefore might be changing".
        _, manager, monitored = self._manager(tmp_path, stability_window=3600.0)
        data = monitored / "sample.dat"
        data.write_text("hello")
        manager._handle_rescan()

        claim = SourceClaim(
            source_type="fake",
            primary_path=str(monitored / "never-existed.dat"),
            source_id="gone",
        )
        assert manager._reconciler._claim_is_quiet(claim) is True
