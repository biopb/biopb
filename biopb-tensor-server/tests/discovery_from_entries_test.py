"""Snapshot-driven claim discovery (biopb/biopb#56 item 4).

`discover_sources_from_entries` reproduces `discover_sources`'s claim protocol from
the (path, is_dir) snapshot the state walk already built, instead of re-walking the
filesystem. These tests pin that it (a) yields the same claims as a real walk, (b)
prunes a claimed directory's subtree (#55) and a skipped/filtered subtree, and (c)
answers is_dir/is_file from the cached flag without stat'ing the entry (the item-3
mechanism that stops every adapter re-stat'ing).
"""

import os
from pathlib import Path

import pytest

from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.discovery import (
    ClaimContext,
    discover_sources,
    discover_sources_from_entries,
)
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr


def _snapshot(root):
    """A (resolved_path_str, is_dir) DFS parent-first snapshot of ``root``.

    Mirrors what ``SourceManager._scan_tree_state`` records into ``next_state``:
    resolved path strings, parent inserted before its children.
    """
    root = Path(root).resolve()
    out = [(str(root), root.is_dir())]

    def rec(d):
        for entry in sorted(os.scandir(d), key=lambda e: e.path):
            out.append((str(Path(entry.path)), entry.is_dir()))
            if entry.is_dir() and not entry.is_symlink():
                rec(entry.path)

    if root.is_dir():
        rec(root)
    return out


def _spy_registry(seen_paths):
    """Wrap the default registry so every probed path is recorded."""
    registry = get_default_registry()
    real = registry.get_claims_for_path

    def spy(ctx, state):
        seen_paths.append(str(ctx.path_str))
        return real(ctx, state)

    registry.get_claims_for_path = spy
    return registry


class TestEquivalenceWithWalk:
    def test_same_claims_as_discover_sources(self, tmp_path):
        """The snapshot path discovers exactly the sources a real walk does."""
        pytest.importorskip("zarr")
        store, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path / "plate"), base_shape=(256, 256), chunk_size=(64, 64)
        )
        nested, _, _ = create_multiresolution_ome_zarr(str(tmp_path / "a" / "b"))

        walked = discover_sources(tmp_path, get_default_registry())
        snap = discover_sources_from_entries(_snapshot(tmp_path), get_default_registry())

        assert {c.primary_path for c in snap.claims.values()} == {
            c.primary_path for c in walked.claims.values()
        }
        assert {str(Path(store)), str(Path(nested))} <= {
            c.primary_path for c in snap.claims.values()
        }


class TestPruning:
    def test_claimed_store_interior_not_probed(self, tmp_path):
        """A claimed OME-Zarr store's interior chunk files are never probed (#55)."""
        pytest.importorskip("zarr")
        store, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path), base_shape=(512, 512), chunk_size=(32, 32)
        )
        store = Path(store)

        seen = []
        state = discover_sources_from_entries(_snapshot(tmp_path), _spy_registry(seen))

        assert {c.primary_path for c in state.claims.values()} == {str(store)}
        inside = [p for p in seen if store in Path(p).parents]
        assert inside == [], f"probed interior store paths: {inside[:5]} ..."

    def test_skipped_dir_subtree_not_probed(self, tmp_path):
        """Entries under a skipped_dirs root are carried in the snapshot but pruned.

        Reproduces the rescan's stable-subtree skip: the descendants are present in
        next_state (carried forward) yet must not be re-probed; their claims are
        preserved elsewhere.
        """
        pytest.importorskip("zarr")
        skip = tmp_path / "stable"
        skip.mkdir()
        store, _, _ = create_multiresolution_ome_zarr(str(skip / "plate"))

        seen = []
        state = discover_sources_from_entries(
            _snapshot(tmp_path),
            _spy_registry(seen),
            skipped_dirs={str(skip.resolve())},
        )

        assert state.claims == {}
        under_skip = [p for p in seen if skip.resolve() in Path(p).parents or Path(p) == skip.resolve()]
        assert under_skip == [], f"probed under skipped dir: {under_skip[:5]} ..."

    def test_path_filter_false_dir_prunes_subtree(self, tmp_path):
        """A directory that fails path_filter is not descended (matches the walk)."""
        pytest.importorskip("zarr")
        blocked = tmp_path / "unstable"
        blocked.mkdir()
        create_multiresolution_ome_zarr(str(blocked / "plate"))

        blocked_str = str(blocked.resolve())
        state = discover_sources_from_entries(
            _snapshot(tmp_path),
            get_default_registry(),
            path_filter=lambda p: p != blocked_str,
        )

        # The store lives under the filtered dir, so nothing is discovered.
        assert state.claims == {}


class TestCachedIsDirNoStat:
    def test_is_dir_is_file_exists_answered_without_filesystem(self):
        """A cached is_dir answers is_dir/is_file/exists with no stat.

        Using a path that does not exist proves the cache short-circuits the
        filesystem: a live ``exists()`` would return False.
        """
        missing = Path("/definitely/not/here/foo.tif")

        file_ctx = ClaimContext(missing, is_dir=False)
        assert file_ctx.is_dir() is False
        assert file_ctx.is_file() is True
        assert file_ctx.exists() is True  # cached: did not stat the missing path

        dir_ctx = ClaimContext(missing, is_dir=True)
        assert dir_ctx.is_dir() is True
        assert dir_ctx.is_file() is False
        assert dir_ctx.exists() is True

        # join() sub-contexts carry no cache, so they stat live (here: missing).
        assert dir_ctx.join(".zattrs").exists() is False

    def test_no_cache_falls_back_to_stat(self, tmp_path):
        """Without a cached flag, ClaimContext stats as before."""
        f = tmp_path / "a.txt"
        f.write_text("x")
        ctx = ClaimContext(f)
        assert ctx.is_file() is True
        assert ctx.is_dir() is False
        assert ClaimContext(tmp_path).is_dir() is True
