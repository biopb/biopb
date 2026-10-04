"""The parallel discovery walk finds what the serial walk finds."""

import os
import threading

import numpy as np
import pytest
import tifffile
import zarr
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import (
    DiscoveryState,
    WalkReport,
    discover_sources,
)


def _zarr(parent, name):
    path = os.path.join(parent, name)
    z = zarr.open_array(path, mode="w", shape=(2, 8, 8), chunks=(1, 8, 8), dtype="u2")
    z[:] = np.arange(128, dtype="u2").reshape(2, 8, 8)
    return path


def _tiff(parent, name):
    path = os.path.join(parent, name)
    tifffile.imwrite(path, np.arange(64, dtype="u2").reshape(8, 8))
    return path


def _tree(root, dirs=12, per_dir=5):
    """Directories nested two deep, each with TIFFs and one zarr store."""
    for i in range(dirs):
        d = os.path.join(root, f"g{i % 4}", f"d{i}")
        os.makedirs(d)
        for j in range(per_dir):
            _tiff(d, f"img_{j}.tif")
        _zarr(d, "store.zarr")


def _ids(state):
    return {c.source_id: (c.source_type, c.primary_path) for c in state.claims.values()}


class TestSameAsSerial:
    def test_the_same_sources(self, tmp_path):
        _tree(str(tmp_path))
        serial = discover_sources(tmp_path, get_default_registry())
        parallel = discover_sources(tmp_path, get_default_registry(), walk_threads=4)
        assert len(serial.claims) >= 12
        assert _ids(parallel) == _ids(serial)

    def test_the_same_members_of_each_source(self, tmp_path):
        _tree(str(tmp_path), dirs=6)
        serial = discover_sources(tmp_path, get_default_registry())
        parallel = discover_sources(tmp_path, get_default_registry(), walk_threads=4)
        assert {i: c.member_paths for i, c in parallel.claims.items()} == {
            i: c.member_paths for i, c in serial.claims.items()
        }

    def test_a_claimed_directory_is_not_entered(self, tmp_path):
        _zarr(str(tmp_path), "a.zarr")
        probed = []
        registry = get_default_registry()
        real = registry.get_claims_for_path
        mu = threading.Lock()

        def spy(ctx, state):
            with mu:
                probed.append(ctx.path_str)
            return real(ctx, state)

        registry.get_claims_for_path = spy
        discover_sources(tmp_path, registry, walk_threads=4)
        inside = os.path.join(str(tmp_path), "a.zarr") + os.sep
        assert not [p for p in probed if p.startswith(inside)]

    def test_the_skip_policy_and_the_filter_apply(self, tmp_path):
        os.makedirs(tmp_path / ".hidden")
        _zarr(str(tmp_path / ".hidden"), "h.zarr")
        os.makedirs(tmp_path / "old")
        _zarr(str(tmp_path / "old"), "o.zarr")
        _zarr(str(tmp_path), "keep.zarr")
        reports = {}
        for threads in (1, 3):
            reports[threads] = WalkReport()
            state = discover_sources(
                tmp_path,
                get_default_registry(),
                path_filter=lambda p: p.name != "old",
                report=reports[threads],
                walk_threads=threads,
            )
            assert {
                os.path.basename(c.primary_path) for c in state.claims.values()
            } == {"keep.zarr"}
        assert reports[3].declined_dirs == reports[1].declined_dirs
        assert str(tmp_path / "old") in reports[3].declined_dirs
        assert str(tmp_path / ".hidden") in reports[3].declined_dirs

    def test_a_directory_symlink_is_not_entered(self, tmp_path):
        real = tmp_path / "real"
        os.makedirs(real)
        _zarr(str(real), "a.zarr")
        os.symlink(real, tmp_path / "alias", target_is_directory=True)
        parallel = discover_sources(tmp_path, get_default_registry(), walk_threads=4)
        paths = {c.primary_path for c in parallel.claims.values()}
        assert paths == {str(real / "a.zarr")}

    def test_a_loop_stops_at_the_depth_limit(self, tmp_path):
        # An ordinary directory that is not a symlink cannot loop; a deep tree
        # stops at max_depth with a declined directory instead of running away.
        d = tmp_path
        for i in range(5):
            d = d / f"l{i}"
            os.makedirs(d)
        _zarr(str(d), "deep.zarr")
        from biopb_tensor_server.core import discovery

        state = DiscoveryState()
        report = WalkReport()
        discovery._discover_parallel(
            tmp_path,
            get_default_registry(),
            state,
            None,
            False,
            False,
            report,
            False,
            threads=2,
            max_depth=2,
        )
        assert not state.claims
        assert report.declined_dirs


class TestWithoutTheIdentitySet:
    def test_a_hardlinked_file_is_a_source_at_each_path(self, tmp_path):
        a = _tiff(str(tmp_path), "a.tif")
        os.link(a, tmp_path / "b.tif")
        state = discover_sources(tmp_path, get_default_registry(), walk_threads=2)
        assert {os.path.basename(c.primary_path) for c in state.claims.values()} == {
            "a.tif",
            "b.tif",
        }


class TestTheScheduler:
    def test_claims_are_applied_on_the_calling_thread(self, tmp_path):
        _tree(str(tmp_path), dirs=8)
        seen = set()
        caller = threading.get_ident()

        def on_added(claim):
            seen.add(threading.get_ident())

        state = DiscoveryState(on_source_added=on_added)
        discover_sources(tmp_path, get_default_registry(), state, walk_threads=4)
        assert state.claims
        assert seen == {caller}

    def test_concurrent_probes_each_record_their_own_members(self, tmp_path):
        # The recorder was one slot on the shared state: concurrent probes
        # overwrote each other and a claim took another's members.
        _tree(str(tmp_path), dirs=16, per_dir=3)
        serial = discover_sources(tmp_path, get_default_registry())
        for _ in range(3):
            parallel = discover_sources(
                tmp_path, get_default_registry(), walk_threads=8
            )
            assert {i: c.member_paths for i, c in parallel.claims.items()} == {
                i: c.member_paths for i, c in serial.claims.items()
            }

    def test_a_failing_walk_raises_and_does_not_hang(self, tmp_path):
        _tree(str(tmp_path), dirs=4)
        registry = get_default_registry()

        def boom(ctx, state):
            raise RuntimeError("probe blew up")

        registry.get_claims_for_path = boom
        with pytest.raises(RuntimeError, match="probe blew up"):
            discover_sources(tmp_path, registry, walk_threads=3)
