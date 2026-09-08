"""Runtime source registration: the ``add_source`` Flight action / drag-drop.

Covers the SourceManager entrypoint ``add_local_source`` (in-process, fast) for
every drop outcome, and one full server->client gRPC round-trip through the
``add_source`` action so the streaming envelope mapping is exercised end to end.

Cases mirror the plan's per-case table: single file/dir add, exact-path dedup
(already_present), subdir-of-existing rejection (containment, case 4), parent-dir
re-discovery (case 5), empty-dir (6), unrecognized-file (7), and a bogus path.
"""

import os
import threading
import time

import numpy as np
import pytest
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import DiscoveryState
from biopb_tensor_server.sources.source_manager import (
    DND_URL_PREFIX,
    SourceManager,
    _drop_catalog_url,
)


def _zarr_available() -> bool:
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _zarr_available(), reason="zarr not available")


def _make_zarr(parent, name, shape=(4, 8, 8)):
    import zarr

    path = os.path.join(parent, name)
    z = zarr.open_array(
        path, mode="w", shape=shape, chunks=(1,) + shape[1:], dtype="uint16"
    )
    z[:] = np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape)
    return path


def _make_manager():
    server = TensorFlightServer("grpc://localhost:0")
    manager = SourceManager(
        server=server,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        watcher=None,
        monitored_dirs=set(),
        metadata_db=None,
    )
    return manager, server


def _drain_all(gen):
    """Run an add_local_source generator to its terminal ``(added, already,
    refreshed, removed, failed)`` result tuple."""
    result = ([], [], [], [], [])
    for event in gen:
        if event[0] == "result":
            result = event[1:]
    return result


def _drain(gen):
    """The three original tallies; ``_drain_all`` for the refresh/removal ones."""
    added, already, _refreshed, _removed, failed = _drain_all(gen)
    return added, already, failed


class TestDropCatalogUrl:
    """Unit coverage for the ``dnd://`` origin scheme on a drop's catalog url."""

    def test_single_file_drop_marked_dnd(self):
        url = _drop_catalog_url("/data/exp.zarr", "/data/exp.zarr")
        assert url == "dnd://exp.zarr"
        assert url.startswith(DND_URL_PREFIX)

    def test_folder_drop_children_marked_dnd_under_one_root(self):
        assert (
            _drop_catalog_url("/data/exp", "/data/exp/sub/b.tif")
            == "dnd://exp/sub/b.tif"
        )

    def test_mark_dnd_false_reroots_without_scheme(self):
        # A drop under a monitored root: tidy display root, but no removable
        # marker (the periodic rescan governs it).
        url = _drop_catalog_url("/data/exp.zarr", "/data/exp.zarr", mark_dnd=False)
        assert url == "exp.zarr"
        assert not url.startswith(DND_URL_PREFIX)


class TestAddLocalSource:
    """In-process coverage of SourceManager.add_local_source."""

    def test_add_single_zarr_dir(self, tmp_path):
        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")

        added, already, failed = _drain(manager.add_local_source(zpath))

        assert len(added) == 1 and not already and not failed
        sid = added[0].source_id
        assert sid in server.sources

    def test_dropped_single_source_reroots_to_basename(self, tmp_path):
        """A dropped dataset's catalog source_url is re-rooted at its own
        basename under the ``dnd://`` origin scheme, so the browser renders it as
        its own top-level root instead of nesting it under the shared
        absolute-path tree, and can tell it is a removable drop."""
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")

        added, *_ = _drain(manager.add_local_source(zpath))

        assert added[0].source_url == "dnd://exp.zarr"

    def test_dropped_folder_reroots_children_under_folder(self, tmp_path):
        """Dropping a plain folder re-roots every discovered source under the
        folder's basename (one drop == one root, sources as children), all under
        the ``dnd://`` origin scheme."""
        manager, _ = _make_manager()
        root = tmp_path / "my_experiment"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        _make_zarr(str(root), "b.zarr")

        added, _, failed = _drain(manager.add_local_source(str(root)))

        assert not failed
        urls = sorted(d.source_url for d in added)
        assert urls == ["dnd://my_experiment/a.zarr", "dnd://my_experiment/b.zarr"]

    def test_overlapping_drop_keeps_native_url_no_reroot(self, tmp_path):
        """A drop that overlaps an already-registered source (e.g. re-dropping a
        monitor=false config dir to pick up new files) is a rescan of a known
        location, so new siblings keep their native source_url -- they must NOT
        re-root into a separate tree root and split the dir across two places."""
        manager, _ = _make_manager()
        root = tmp_path / "proj"
        root.mkdir()
        a = _make_zarr(str(root), "a.zarr")

        # a.zarr registered first (stands in for the startup config scan).
        _drain(manager.add_local_source(a))
        # A new dataset lands, then the dir is dropped to force a rescan.
        _make_zarr(str(root), "b.zarr")
        added, already, failed = _drain(manager.add_local_source(str(root)))

        assert not failed and len(added) == 1  # only b.zarr is new
        assert len(already) == 1  # a.zarr already present
        # The overlap suppressed re-rooting: b.zarr keeps a native file:// url,
        # coherent with a.zarr, instead of a bare "proj/b.zarr" own-root url.
        assert added[0].source_url.startswith("file://")
        assert added[0].source_url.endswith("/proj/b.zarr")

    def test_static_config_via_symlink_containment(self, tmp_path):
        """A static-config source configured through a symlinked path still
        catches a drop that lands inside it. The containment guard keys on
        os.path.realpath, so the seeded claim must store the resolved path (case
        4 over a symlinked config path)."""
        from biopb_tensor_server.core.config import SourceConfig
        from biopb_tensor_server.sources.source_manager import create_source_manager

        real = tmp_path / "real"
        real.mkdir()
        zpath = _make_zarr(str(real), "exp.zarr")
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("cannot create symlinks (e.g. Windows without privilege)")

        server = TensorFlightServer("grpc://localhost:0")
        manager = create_source_manager(
            server=server,
            registry=get_default_registry(),
            watcher=None,
            static_sources=[SourceConfig(url=str(link / "exp.zarr"), type="zarr")],
        )
        assert manager is not None

        # Drop a subdir inside the source, reached via the real (resolved) path.
        sub = os.path.join(zpath, "sub")
        os.makedirs(sub, exist_ok=True)
        added, already, failed = _drain(manager.add_local_source(sub))

        assert added == [] and len(failed) == 1
        assert "already part of" in failed[0][1]

    def test_static_config_alias_reroots_descriptor_source_url(self, tmp_path):
        """A configured local source with an ``alias`` surfaces that alias as its
        catalog tree root -- the config-line analogue of a drag-dropped folder
        getting its own root. End-to-end: resolve_all_sources computes the
        catalog_url from the alias, create_source_manager threads it as the
        descriptor's display source_url override (never the source_id)."""
        from biopb_tensor_server.core.config import (
            ServerConfig,
            SourceConfig,
            resolve_all_sources,
        )
        from biopb_tensor_server.sources.source_manager import create_source_manager

        root = tmp_path / "acquisition"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        _make_zarr(str(root), "b.zarr")

        cfg = ServerConfig(sources=[SourceConfig(url=str(root), alias="exp")])
        static_sources = resolve_all_sources(cfg)

        server = TensorFlightServer("grpc://localhost:0")
        manager = create_source_manager(
            server=server,
            registry=get_default_registry(),
            watcher=None,
            static_sources=static_sources,
        )
        assert manager is not None

        urls = sorted(
            adapter.get_source_descriptor().source_url
            for adapter in server.sources.values()
        )
        assert urls == ["exp/a.zarr", "exp/b.zarr"]
        # Display-only: source_id still hashes the raw path, not the alias.
        assert all("exp" not in sid for sid in server.sources)

    def test_readd_same_path_is_already_present(self, tmp_path):
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id

        added2, already2, failed2 = _drain(manager.add_local_source(zpath))

        assert added2 == [] and already2 == [sid] and failed2 == []

    def test_subdir_of_existing_source_rejected(self, tmp_path):
        """Case 4: a drop inside an existing source is rejected (containment)."""
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        _drain(manager.add_local_source(zpath))
        sub = os.path.join(zpath, "sub")
        os.makedirs(sub, exist_ok=True)

        added, already, failed = _drain(manager.add_local_source(sub))

        assert added == [] and len(failed) == 1
        assert "already part of" in failed[0][1]

    def test_parent_dir_adds_new_keeps_existing(self, tmp_path):
        """Case 5: dropping a parent adds new siblings, existing -> already_present."""
        manager, _ = _make_manager()
        z1 = _make_zarr(str(tmp_path), "a.zarr")
        _make_zarr(str(tmp_path), "b.zarr")
        added1, *_ = _drain(manager.add_local_source(z1))
        sid1 = added1[0].source_id

        added, already, failed = _drain(manager.add_local_source(str(tmp_path)))

        assert len(added) == 1  # only b.zarr is new
        assert sid1 in already
        assert not failed

    def test_empty_dir_reports_no_datasets(self, tmp_path):
        """Case 6: an empty folder gets a distinct 'no datasets' reason."""
        manager, _ = _make_manager()
        empty = tmp_path / "empty"
        empty.mkdir()

        added, _, failed = _drain(manager.add_local_source(str(empty)))

        assert added == [] and len(failed) == 1
        assert "no supported datasets" in failed[0][1]

    def test_junk_file_reports_unrecognized(self, tmp_path):
        """Case 7: a non-image file gets a distinct 'not recognized' reason."""
        manager, _ = _make_manager()
        junk = tmp_path / "notes.txt"
        junk.write_text("hello")

        added, _, failed = _drain(manager.add_local_source(str(junk)))

        assert added == [] and len(failed) == 1
        assert "not a recognized" in failed[0][1]

    def test_bogus_path_raises(self, tmp_path):
        manager, _ = _make_manager()
        with pytest.raises(FileNotFoundError):
            _drain(manager.add_local_source(str(tmp_path / "nope")))

    def test_remote_url_rejected(self):
        manager, _ = _make_manager()
        with pytest.raises(ValueError, match="local filesystem paths only"):
            _drain(manager.add_local_source("grpc://host:8815/x"))

    @pytest.mark.parametrize("spelling", ["s.zarr", "./s.zarr", "../tmp/s.zarr"])
    def test_rootless_path_rejected(self, tmp_path, monkeypatch, spelling):
        """A rootless path over the wire would silently mean the server's cwd.

        The caller cannot see that directory and did not choose it, so the same
        request means different data depending on how the server was launched
        (biopb/biopb#947). The path here EXISTS relative to the cwd, so nothing
        but the explicit check stops it.

        The predicate is `discovery.local_path_is_rooted`, shared with the config
        loader so the two surfaces cannot drift apart.
        """
        _make_zarr(str(tmp_path), "s.zarr")
        monkeypatch.chdir(tmp_path)
        manager, _ = _make_manager()
        with pytest.raises(ValueError, match="rooted path"):
            _drain(manager.add_local_source(spelling))

    def test_file_url_registers_under_the_path_it_names(self, tmp_path):
        """`file://` and the plain path are one source, not two.

        `resolve_local_path` strips the scheme, so the url form reaches the same
        identity -- an id equal to the plain path's, and a second add that
        deduplicates instead of registering a twin.
        """
        zpath = _make_zarr(str(tmp_path), "s.zarr")
        manager, _ = _make_manager()

        added, *_ = _drain(manager.add_local_source(f"file://{zpath}"))
        assert len(added) == 1

        again_added, already, _ = _drain(manager.add_local_source(zpath))
        assert again_added == [] and already == [added[0].source_id]

    def test_cancel_keeps_already_committed(self, tmp_path):
        """A cancel between sources stops discovery but keeps what registered."""
        manager, server = _make_manager()
        for i in range(3):
            _make_zarr(str(tmp_path), f"s{i}.zarr")

        # Cancel after the first source is committed.
        state = {"n": 0}

        def should_cancel():
            return state["n"] >= 1

        gen = manager.add_local_source(str(tmp_path), should_cancel=should_cancel)
        added, already, failed = [], [], []
        for event in gen:
            if event[0] == "progress":
                state["n"] += 1
            else:
                _, added, already, _refreshed, _removed, failed = event

        assert 1 <= len(added) < 3  # stopped early, kept what was registered
        for desc in added:
            assert desc.source_id in server.sources


class TestAddSourceRoundtrip:
    """Full server -> client gRPC round-trip through the add_source action."""

    def test_client_add_source_over_grpc(self, tmp_path):
        from biopb.tensor import TensorFlightClient

        manager, server = _make_manager()
        server.set_add_source_handler(manager.add_local_source)
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(1)

        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            result = client.add_source(zpath)
            assert len(result.added) == 1
            sid = result.added[0].source_id
            assert not result.failed

            # The new source is now listable and readable.
            assert sid in client.list_sources()
            darr = client.get_tensor(sid)
            assert darr.compute().shape == (4, 8, 8)

            # Re-add over the wire -> already_present, no duplicate.
            again = client.add_source(zpath)
            assert again.added == [] and list(again.already_present) == [sid]

            # A bogus path is a whole-request failure surfaced as a server error.
            import pyarrow.flight as flight

            with pytest.raises(flight.FlightServerError):
                client.add_source(str(tmp_path / "nope"))

            client.close()
        finally:
            server.shutdown()

    def test_plain_tiff_folder_drop_carries_no_dim_labels(self, tmp_path):
        """A drop that names no labels must not register as labelled.

        ``AddSourceRequest.dim_labels`` is unset here, and an unset repeated
        field is ``[]``. Passing that on stamps the empty list onto the claim,
        and an adapter that reads a claim's labels as an override then rejects
        every series -- so this drop failed with "cannot read TIFF source" for
        every plain TIFF, drag-drop included.

        A *folder*, because that is the branch that carried it: the walk hands
        its labels to ``discover_sources``, while the per-claim assignment
        beside it already tests them for truth and so never saw the bug.
        """
        import tifffile
        from biopb.tensor import TensorFlightClient

        manager, server = _make_manager()
        server.set_add_source_handler(manager.add_local_source)
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(1)

        folder = os.path.join(str(tmp_path), "drop")
        os.makedirs(folder)
        tifffile.imwrite(
            os.path.join(folder, "plain.tif"),
            np.arange(5 * 8 * 9, dtype="uint16").reshape(5, 8, 9),
        )
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            result = client.add_source(folder)

            assert not result.failed
            assert len(result.added) == 1
            source_id = result.added[0].source_id

            # Unlabelled, so the adapter fills the canonical slots itself.
            descriptor = client.get_descriptor(f"{source_id}/Image:0")
            assert list(descriptor.dim_labels) == ["T", "C", "Z", "Y", "X"]
            assert list(descriptor.shape) == [1, 1, 5, 8, 9]
            assert client.get_tensor(f"{source_id}/Image:0").compute().shape == (
                1,
                1,
                5,
                8,
                9,
            )
            client.close()
        finally:
            server.shutdown()


class TestRemoveDroppedRoot:
    """SourceManager.remove_dropped_root: deregister a dnd:// drop branch."""

    def test_remove_single_dropped_source(self, tmp_path):
        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id
        assert added[0].source_url == "dnd://exp.zarr"
        assert sid in server.sources

        removed, failed = manager.remove_dropped_root("dnd://exp.zarr")
        assert removed == [sid] and not failed
        assert sid not in server.sources

    def test_remove_dropped_folder_removes_all_siblings_as_a_unit(self, tmp_path):
        manager, server = _make_manager()
        root = tmp_path / "my_experiment"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        _make_zarr(str(root), "b.zarr")
        added, _, failed = _drain(manager.add_local_source(str(root)))
        assert not failed and len(added) == 2
        sids = {d.source_id for d in added}

        removed, failed = manager.remove_dropped_root("dnd://my_experiment")
        assert set(removed) == sids and not failed
        for sid in sids:
            assert sid not in server.sources

    def test_remove_root_matches_only_its_own_branch(self, tmp_path):
        # A sibling drop sharing a name prefix must NOT be swept up.
        manager, server = _make_manager()
        added_a, *_ = _drain(
            manager.add_local_source(_make_zarr(str(tmp_path), "exp.zarr"))
        )
        added_b, *_ = _drain(
            manager.add_local_source(_make_zarr(str(tmp_path), "exp2.zarr"))
        )
        sid_a, sid_b = added_a[0].source_id, added_b[0].source_id

        removed, failed = manager.remove_dropped_root("dnd://exp.zarr")
        assert removed == [sid_a] and not failed
        assert sid_b in server.sources  # exp2.zarr untouched

    def test_remove_non_dnd_root_rejected(self):
        # Authorization boundary: only drag-dropped (dnd://) sources are removable.
        manager, _ = _make_manager()
        with pytest.raises(ValueError, match="dnd://"):
            manager.remove_dropped_root("file:///data/x.zarr")

    def test_remove_bare_scheme_rejected(self):
        # The bare scheme would rstrip("/") to a "dnd:/" prefix that matches every
        # drop -- it must be refused, not resolve to a wildcard "remove all drops".
        # (Rejected up front, before any catalog access, like the non-dnd case.)
        manager, _ = _make_manager()
        for bare in ("dnd://", "dnd:///", "dnd://\\"):
            with pytest.raises(ValueError, match="bare scheme"):
                manager.remove_dropped_root(bare)

    def test_remove_unknown_root_removes_nothing(self):
        manager, _ = _make_manager()
        removed, failed = manager.remove_dropped_root("dnd://never_dropped")
        assert removed == [] and failed == []


class TestRemoveSourceRoundtrip:
    """Full server -> client gRPC round-trip through the remove_source action."""

    def test_client_remove_source_over_grpc(self, tmp_path):
        from biopb.tensor import TensorFlightClient

        manager, server = _make_manager()
        server.set_add_source_handler(manager.add_local_source)
        server.set_remove_source_handler(manager.remove_dropped_root)
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(1)

        root = tmp_path / "exp"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        _make_zarr(str(root), "b.zarr")
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            added = client.add_source(str(root))
            assert len(added.added) == 2
            sids = {d.source_id for d in added.added}
            assert sids <= set(client.list_sources())

            # Remove the whole dropped branch by its dnd:// root.
            result = client.remove_source("dnd://exp")
            assert set(result.removed) == sids and not result.failed
            assert not (sids & set(client.list_sources()))

            # A non-dnd root is refused server-side.
            import pyarrow.flight as flight

            with pytest.raises(flight.FlightServerError):
                client.remove_source("file:///data/x.zarr")

            client.close()
        finally:
            server.shutdown()


class TestAddedSourceSurvivesRescanUnderSkippedDir:
    """A source added under a name-skipped subtree (e.g. OneDrive) must survive
    the periodic monitored-tree rescan.

    ``add_source`` treats the drop as a root, which is exempt from the discovery
    skip policy, so it happily indexes content the monitored-tree walk will not
    descend into (a server pointed at the Windows home dir, data on the user's
    OneDrive-backed Desktop). The reconcile must preserve those claims instead of
    reaping them as "disappeared" (biopb/biopb#309 drag-drop follow-up).
    """

    def _manager(self, monitored_dirs):
        server = TensorFlightServer("grpc://localhost:0")
        manager = SourceManager(
            server=server,
            registry=get_default_registry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs=set(monitored_dirs),
            metadata_db=None,
            stability_window=0.0,
        )
        return manager, server

    def test_monitor_false_drop_under_onedrive_survives_rescan(self, tmp_path):
        # Monitored root is an ANCESTOR of a OneDrive dir; the added folder is not
        # itself monitored (monitor=false, the MCP client's default).
        home = tmp_path / "home"
        drop = home / "OneDrive" / "Desktop" / "samples"
        drop.mkdir(parents=True)
        _make_zarr(str(drop), "exp.zarr")

        manager, server = self._manager({home})

        added, _already, failed = _drain(manager.add_local_source(str(drop)))
        assert len(added) == 1 and not failed
        sid = added[0].source_id
        assert sid in server.sources
        # Re-rooted for a tidy display root, but NOT stamped ``dnd://``: it sits
        # under a monitored root, so the rescan re-discovers it -- it is not
        # safely removable, so it must not carry the removable marker.
        assert added[0].source_url == "samples/exp.zarr"

        # First rescan is force_full; the second is the steady-state incremental
        # that actually reaped the source before the fix (~20 s after the drop).
        manager._rescan_monitored_dirs()
        assert sid in server.sources, "reaped by the initial force_full rescan"
        manager._rescan_monitored_dirs()
        assert sid in server.sources, "reaped by the steady-state incremental rescan"


class TestReDropRebuilds:
    """A re-drop of an already-registered path rebuilds it (biopb/biopb#944).

    ``add_source`` used to report a known ``source_id`` as ``already_present``
    and stop there, so a ``monitor: false`` root had a way to pick up new files
    and none at all to notice a changed or removed one. The descriptor AND the
    ``content_version`` that namespaces the chunk cache are both sampled when
    the adapter is built, so noticing means building a new adapter.
    """

    def test_redrop_reports_refreshed_and_already_present(self, tmp_path):
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id

        _, already, refreshed, removed, failed = _drain_all(
            manager.add_local_source(zpath)
        )

        assert refreshed == [sid]
        # Still already_present: an older client reads a re-drop as "already
        # present" rather than "nothing happened".
        assert already == [sid]
        assert not removed and not failed

    def test_redrop_picks_up_an_in_place_rewrite(self, tmp_path):
        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr", shape=(4, 8, 8))
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id
        before = server.sources.get(sid)

        _make_zarr(str(tmp_path), "exp.zarr", shape=(2, 5, 5))
        _drain(manager.add_local_source(zpath))

        adapter = server.sources.get(sid)
        assert adapter is not before, "the adapter was not rebuilt"
        shape = tuple(adapter.get_source_descriptor().tensors[0].shape)
        assert shape[-2:] == (5, 5)

    def test_rebuild_moves_the_content_version(self, tmp_path):
        """The cache namespaces on this token, so a rebuild that left it alone
        would keep serving the pre-edit chunks."""
        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr", shape=(4, 8, 8))
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id
        before = server.sources.get(sid).content_version

        _make_zarr(str(tmp_path), "exp.zarr", shape=(2, 5, 5))
        _drain(manager.add_local_source(zpath))

        assert server.sources.get(sid).content_version != before

    def test_rebuild_precaches_only_when_the_version_moved(self, tmp_path):
        """A warm re-reads the whole source. An unmoved content_version means
        the cache namespace did not move either, so what is warm is still what
        a read would look up."""
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr", shape=(4, 8, 8))
        _drain(manager.add_local_source(zpath))

        committed = []
        manager._reconciler._notify_source_committed = committed.append

        _drain(manager.add_local_source(zpath))
        assert committed == [], "an unchanged re-drop re-warmed the source"

        _make_zarr(str(tmp_path), "exp.zarr", shape=(2, 5, 5))
        _, _, refreshed, _, _ = _drain_all(manager.add_local_source(zpath))
        assert committed == refreshed

    def test_rebuild_keeps_the_dnd_display_root(self, tmp_path):
        """Re-deriving the url would hand the source its native ``file://`` one
        back, and ``remove_source`` authorizes on the ``dnd://`` scheme."""
        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        _drain(manager.add_local_source(zpath))

        _, _, refreshed, _, _ = _drain_all(manager.add_local_source(zpath))

        assert server.sources.get(refreshed[0]).get_source_descriptor().source_url == (
            "dnd://exp.zarr"
        )

    def test_a_failed_rebuild_keeps_the_source_served(self, tmp_path):
        """Register-before-unregister: the replacement goes in on top of the
        live adapter, so a rebuild that fails costs nothing."""

        class _FailingDb:
            def sync_source_added(self, *a, **kw):
                raise RuntimeError("catalog write failed")

            def sync_source_removed(self, *a, **kw):
                pass

        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id
        original = server.sources.get(sid)

        manager._reconciler._metadata_db = _FailingDb()
        _, already, refreshed, _, failed = _drain_all(manager.add_local_source(zpath))

        assert not refreshed
        assert already == [sid]
        assert len(failed) == 1
        assert sid in server.sources
        assert server.sources.get(sid) is original


class TestReDropRemovesVanished:
    """The removal half: a registered source whose files are gone."""

    def test_redrop_deregisters_a_deleted_source(self, tmp_path):
        import shutil

        manager, server = _make_manager()
        root = tmp_path / "exp"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        _make_zarr(str(root), "b.zarr")
        added, *_ = _drain(manager.add_local_source(str(root)))
        by_url = {d.source_url: d.source_id for d in added}

        shutil.rmtree(str(root / "a.zarr"))
        _, _, _, removed, failed = _drain_all(manager.add_local_source(str(root)))

        assert removed == [by_url["dnd://exp/a.zarr"]]
        assert by_url["dnd://exp/a.zarr"] not in server.sources
        assert by_url["dnd://exp/b.zarr"] in server.sources
        assert not failed

    def test_emptied_folder_removes_instead_of_failing(self, tmp_path):
        """A folder whose only dataset was deleted is how a stale entry gets
        noticed, so the drop reports the removal rather than "nothing here"."""
        import shutil

        manager, server = _make_manager()
        root = tmp_path / "exp"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        added, *_ = _drain(manager.add_local_source(str(root)))
        sid = added[0].source_id

        shutil.rmtree(str(root / "a.zarr"))
        _, _, _, removed, failed = _drain_all(manager.add_local_source(str(root)))

        assert removed == [sid]
        assert not failed
        assert sid not in server.sources

    def test_removal_is_scoped_to_the_drop(self, tmp_path):
        """The periodic reconcile's diff is whole-catalog; run against a subtree
        walk it would deregister everything outside the drop."""
        import shutil

        manager, server = _make_manager()
        outside = _make_zarr(str(tmp_path), "outside.zarr")
        added_outside, *_ = _drain(manager.add_local_source(outside))

        root = tmp_path / "exp"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        added, *_ = _drain(manager.add_local_source(str(root)))

        shutil.rmtree(str(root / "a.zarr"))
        _, _, _, removed, _ = _drain_all(manager.add_local_source(str(root)))

        assert removed == [added[0].source_id]
        assert added_outside[0].source_id in server.sources

    def test_a_still_present_but_unclaimed_path_is_kept(self, tmp_path):
        """Absence from the walk is not evidence of deletion -- a drop has none
        of the stability gating the periodic path removes under -- so only a
        vanished path deregisters."""
        manager, server = _make_manager()
        root = tmp_path / "exp"
        root.mkdir()
        zpath = _make_zarr(str(root), "a.zarr")
        added, *_ = _drain(manager.add_local_source(str(root)))
        sid = added[0].source_id

        # Break the store so nothing claims it, but leave the path in place.
        for meta in ("zarr.json", ".zarray"):
            if os.path.exists(os.path.join(zpath, meta)):
                os.remove(os.path.join(zpath, meta))
        _, _, _, removed, _ = _drain_all(manager.add_local_source(str(root)))

        assert not removed
        assert sid in server.sources
