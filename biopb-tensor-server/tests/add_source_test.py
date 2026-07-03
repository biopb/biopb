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
from biopb_tensor_server.discovery import DiscoveryState
from biopb_tensor_server.source_manager import SourceManager


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


def _drain(gen):
    """Run an add_local_source generator to its terminal result tuple."""
    added, already, failed, needs_confirm = [], [], [], False
    for event in gen:
        if event[0] == "result":
            _, added, already, failed, needs_confirm = event
    return added, already, failed, needs_confirm


class TestAddLocalSource:
    """In-process coverage of SourceManager.add_local_source."""

    def test_add_single_zarr_dir(self, tmp_path):
        manager, server = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")

        added, already, failed, _ = _drain(manager.add_local_source(zpath))

        assert len(added) == 1 and not already and not failed
        sid = added[0].source_id
        assert sid in server._sources

    def test_dropped_single_source_reroots_to_basename(self, tmp_path):
        """A dropped dataset's catalog source_url is re-rooted at its own
        basename, so the browser renders it as its own top-level root instead of
        nesting it under the shared absolute-path tree."""
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")

        added, *_ = _drain(manager.add_local_source(zpath))

        assert added[0].source_url == "exp.zarr"

    def test_dropped_folder_reroots_children_under_folder(self, tmp_path):
        """Dropping a plain folder re-roots every discovered source under the
        folder's basename (one drop == one root, sources as children)."""
        manager, _ = _make_manager()
        root = tmp_path / "my_experiment"
        root.mkdir()
        _make_zarr(str(root), "a.zarr")
        _make_zarr(str(root), "b.zarr")

        added, _, failed, _ = _drain(manager.add_local_source(str(root)))

        assert not failed
        urls = sorted(d.source_url for d in added)
        assert urls == ["my_experiment/a.zarr", "my_experiment/b.zarr"]

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
        added, already, failed, _ = _drain(manager.add_local_source(str(root)))

        assert not failed and len(added) == 1  # only b.zarr is new
        assert len(already) == 1  # a.zarr already present
        # The overlap suppressed re-rooting: b.zarr keeps a native file:// url,
        # coherent with a.zarr, instead of a bare "proj/b.zarr" own-root url.
        assert added[0].source_url.startswith("file://")
        assert added[0].source_url.endswith("/proj/b.zarr")

    def test_readd_same_path_is_already_present(self, tmp_path):
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        added, *_ = _drain(manager.add_local_source(zpath))
        sid = added[0].source_id

        added2, already2, failed2, _ = _drain(manager.add_local_source(zpath))

        assert added2 == [] and already2 == [sid] and failed2 == []

    def test_subdir_of_existing_source_rejected(self, tmp_path):
        """Case 4: a drop inside an existing source is rejected (containment)."""
        manager, _ = _make_manager()
        zpath = _make_zarr(str(tmp_path), "exp.zarr")
        _drain(manager.add_local_source(zpath))
        sub = os.path.join(zpath, "sub")
        os.makedirs(sub, exist_ok=True)

        added, already, failed, _ = _drain(manager.add_local_source(sub))

        assert added == [] and len(failed) == 1
        assert "already part of" in failed[0][1]

    def test_parent_dir_adds_new_keeps_existing(self, tmp_path):
        """Case 5: dropping a parent adds new siblings, existing -> already_present."""
        manager, _ = _make_manager()
        z1 = _make_zarr(str(tmp_path), "a.zarr")
        _make_zarr(str(tmp_path), "b.zarr")
        added1, *_ = _drain(manager.add_local_source(z1))
        sid1 = added1[0].source_id

        added, already, failed, _ = _drain(manager.add_local_source(str(tmp_path)))

        assert len(added) == 1  # only b.zarr is new
        assert sid1 in already
        assert not failed

    def test_empty_dir_reports_no_datasets(self, tmp_path):
        """Case 6: an empty folder gets a distinct 'no datasets' reason."""
        manager, _ = _make_manager()
        empty = tmp_path / "empty"
        empty.mkdir()

        added, _, failed, _ = _drain(manager.add_local_source(str(empty)))

        assert added == [] and len(failed) == 1
        assert "no supported datasets" in failed[0][1]

    def test_junk_file_reports_unrecognized(self, tmp_path):
        """Case 7: a non-image file gets a distinct 'not recognized' reason."""
        manager, _ = _make_manager()
        junk = tmp_path / "notes.txt"
        junk.write_text("hello")

        added, _, failed, _ = _drain(manager.add_local_source(str(junk)))

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
        added, already, failed, _ = [], [], [], False
        for event in gen:
            if event[0] == "progress":
                state["n"] += 1
            else:
                _, added, already, failed, _ = event

        assert 1 <= len(added) < 3  # stopped early, kept what was registered
        for desc in added:
            assert desc.source_id in server._sources


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
