"""Discovery must not probe the interior of a directory it claimed (biopb/biopb#55).

When a container store (e.g. a `.zarr`) is claimed, everything under it belongs to
it. Probing each interior chunk file for a claim that can never fire would make a
scan's cost proportional to the number of chunk *files* rather than logical
sources, and for a root that is itself a store, even stat-ing them is a visible
hang on a drop.

These tests pin that for ``discover_under``, the one-shot walk behind a drop and a
``monitor = false`` directory.
"""

from pathlib import Path

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr
from biopb_tensor_server.sources import scan_root
from biopb_tensor_server.sources.scan_root import discover_under


def _spy_registry(seen_paths):
    """Wrap the default registry so every probed path is recorded."""
    registry = get_default_registry()
    real = registry.get_claims_for_path

    def spy(ctx, state):
        seen_paths.append(str(ctx.path_str))
        return real(ctx, state)

    registry.get_claims_for_path = spy
    return registry


class TestClaimedDirsAreNotProbed:
    def test_chunk_files_under_claimed_store_are_not_probed(self, tmp_path):
        """A claimed OME-Zarr store's interior chunk files are never probed."""
        pytest.importorskip("zarr")
        # Small chunks => hundreds of interior chunk files, so an accidental
        # descent would be unmistakable.
        store, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path), base_shape=(512, 512), chunk_size=(32, 32)
        )
        store = Path(store)

        seen = []
        state = discover_under(tmp_path, _spy_registry(seen))

        assert {c.primary_path for c in state.claims.values()} == {str(store)}
        inside = [p for p in seen if store in Path(p).parents]
        assert inside == [], f"probed interior store paths: {inside[:5]} ..."

    def test_unclaimed_dirs_are_still_descended(self, tmp_path):
        """Plain subdirectories (no claim) keep being walked into."""
        pytest.importorskip("zarr")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        store, _, _ = create_multiresolution_ome_zarr(str(nested))

        state = discover_under(tmp_path, get_default_registry())

        assert {c.primary_path for c in state.claims.values()} == {str(Path(store))}


class TestRootThatIsADataset:
    def test_a_claimed_root_is_not_walked(self, tmp_path, monkeypatch):
        """Dropping a store itself claims it without stat-ing its interior.

        The snapshot walk stats every entry; for a 20k-chunk store that is the
        difference between ~0.2 ms and ~200 ms, on the most common drop there is.
        """
        pytest.importorskip("zarr")
        store, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path / "plate"), base_shape=(256, 256), chunk_size=(32, 32)
        )

        def _no_walk(*args, **kwargs):
            raise AssertionError("a claimed root must not be walked")

        monkeypatch.setattr(scan_root._ONE_SHOT_SCANNER, "scan_once", _no_walk)

        state = discover_under(Path(store), get_default_registry())

        assert {c.primary_path for c in state.claims.values()} == {str(Path(store))}

    def test_a_single_file_is_claimed_in_place(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        import numpy as np

        path = tmp_path / "a.tif"
        tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint16))

        state = discover_under(path, get_default_registry())

        assert {c.primary_path for c in state.claims.values()} == {str(path)}

    def test_an_unrecognized_file_claims_nothing(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello")

        assert discover_under(path, get_default_registry()).claims == {}


class TestLoopsAreBounded:
    def test_a_symlink_cycle_does_not_recurse(self, tmp_path):
        """A directory symlinked back to its parent is not followed."""
        pytest.importorskip("zarr")
        sub = tmp_path / "sub"
        sub.mkdir()
        store, _, _ = create_multiresolution_ome_zarr(str(sub / "plate"))
        try:
            (sub / "loop").symlink_to(tmp_path, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks unavailable")

        state = discover_under(tmp_path, get_default_registry())

        assert {c.primary_path for c in state.claims.values()} == {str(Path(store))}
