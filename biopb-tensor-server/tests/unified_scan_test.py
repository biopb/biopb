"""One scan for a monitored root and a scan-once root.

Both walk a root, commit what is new as it is found, then compare the walk with
that root's own claims. They differ in two things: a monitored root is walked again,
so it gates on the stability window and removes a claim only after two misses; a
scan-once root is walked once, so it does neither.
"""

import shutil

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import DiscoveryState, discover_sources
from biopb_tensor_server.sources.roots import RootKind

from tests import catalog_server, deferred_registration_test as drt, make_manager


def _manager(tmp_path, stability_window=0):
    monitored = tmp_path / "monitored"
    once = tmp_path / "once"
    monitored.mkdir()
    once.mkdir()
    server = catalog_server("localhost:0")
    manager = make_manager(
        server=server,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        metadata_db=server.metadata_db,
        monitored_dirs={monitored},
        scan_once_sources=[SourceConfig(url=str(once))],
        stability_window=stability_window,
    )
    return manager, server, monitored, once


def _root(manager, kind):
    (root,) = manager._roots.of_kind(kind)
    return root


def _ids(server):
    return set(drt._only_ids(server))


class TestScanOnce:
    def test_it_ignores_the_stability_window_a_monitored_root_keeps(self, tmp_path):
        manager, server, monitored, once = _manager(tmp_path, stability_window=3600)
        drt._make_zarr(monitored, "fresh.zarr")
        drt._make_zarr(once, "fresh.zarr")

        manager._handle_rescan()

        (sid,) = _ids(server)  # only the scan-once root's: nothing walks it again
        assert manager._reconciler.claim_primary_path(sid).startswith(str(once))

    def test_it_removes_what_is_gone_at_once_and_a_monitored_root_after_two(
        self, tmp_path
    ):
        manager, server, monitored, once = _manager(tmp_path)
        drt._make_zarr(monitored, "m.zarr")
        drt._make_zarr(once, "o.zarr")
        manager._handle_rescan()
        assert len(_ids(server)) == 2
        shutil.rmtree(monitored / "m.zarr")
        shutil.rmtree(once / "o.zarr")

        manager._scan_root(_root(manager, RootKind.SCAN_ONCE), recurring=False)
        assert len(_ids(server)) == 1  # the scan-once source, at once

        monitored_root = _root(manager, RootKind.MONITORED)
        manager._scan_root(monitored_root, recurring=True)
        assert len(_ids(server)) == 1  # one miss
        manager._scan_root(monitored_root, recurring=True)
        assert _ids(server) == set()

    def test_a_claim_that_is_still_there_and_still_claimed_is_not_removed(
        self, tmp_path, monkeypatch
    ):
        manager, server, monitored, once = _manager(tmp_path)
        drt._make_zarr(once, "o.zarr")
        manager._handle_rescan()
        reconciler = manager._reconciler
        root = _root(manager, RootKind.SCAN_ONCE)
        snapshot = reconciler.claims_under(root.path)

        # The walk missed it, but an adapter claims it on a second look.
        monkeypatch.setattr(reconciler, "_claimed_again", lambda claim: True)
        reconciler._reconcile_root(snapshot, DiscoveryState(), recurring=False)
        assert len(_ids(server)) == 1

        monkeypatch.setattr(reconciler, "_claimed_again", lambda claim: False)
        reconciler._reconcile_root(snapshot, DiscoveryState(), recurring=False)
        assert _ids(server) == set()


class TestOneRootAtATime:
    def test_a_scan_leaves_the_other_roots_claims_alone(self, tmp_path):
        manager, server, monitored, once = _manager(tmp_path)
        drt._make_zarr(monitored, "m.zarr")
        drt._make_zarr(once, "o.zarr")
        manager._handle_rescan()
        assert len(_ids(server)) == 2

        shutil.rmtree(monitored / "m.zarr")
        monitored_root = _root(manager, RootKind.MONITORED)
        for _ in range(3):
            manager._scan_root(monitored_root, recurring=True)

        # The monitored claim is gone; the scan-once one was never in its snapshot.
        (sid,) = _ids(server)
        assert manager._reconciler.claim_primary_path(sid).startswith(str(once))

    def test_claims_under_is_a_prefix_filter_on_the_spelled_path(self, tmp_path):
        manager, server, monitored, once = _manager(tmp_path)
        drt._make_zarr(monitored, "m.zarr")
        drt._make_zarr(once, "o.zarr")
        manager._handle_rescan()
        reconciler = manager._reconciler

        under_monitored = reconciler.claims_under(monitored)
        under_once = reconciler.claims_under(once)

        assert len(under_monitored) == len(under_once) == 1
        assert not set(under_monitored) & set(under_once)
        assert reconciler.claims_under(tmp_path / "elsewhere") == {}


class TestTheComparisonAddsNothing:
    def test_a_claim_the_walk_did_not_stream_is_not_added_by_the_comparison(
        self, tmp_path
    ):
        manager, server, monitored, once = _manager(tmp_path)
        drt._make_zarr(monitored, "m.zarr")
        # A walk with no streaming hook: what it found is only in its own state.
        found = discover_sources(
            monitored, get_default_registry(), DiscoveryState(), monitored=True
        )
        assert len(found.claims) == 1

        manager._reconciler._reconcile_root({}, found, recurring=True)

        assert _ids(server) == set()
        assert not manager._reconciler.claim_ids()

    def test_a_claim_that_did_not_commit_when_streamed_is_found_again_next_tick(
        self, tmp_path, monkeypatch
    ):
        manager, server, monitored, once = _manager(tmp_path)
        drt._make_zarr(monitored, "m.zarr")
        reconciler = manager._reconciler
        real = reconciler._commit_add_claim
        monkeypatch.setattr(reconciler, "_commit_add_claim", lambda *a, **k: False)
        manager._handle_rescan()
        assert not reconciler.claim_ids()

        monkeypatch.setattr(reconciler, "_commit_add_claim", real)
        manager._handle_rescan()
        assert len(reconciler.claim_ids()) == 1
