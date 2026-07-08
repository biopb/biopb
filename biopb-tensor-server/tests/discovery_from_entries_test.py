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
from biopb_tensor_server.core.discovery import (
    ClaimContext,
    discover_sources,
    discover_sources_from_entries,
)
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr


def _sig(stat_result, is_dir):
    """The content-identity signature shape ``build_entry_signature`` emits."""
    if is_dir:
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _snapshot(root):
    """A (resolved_path_str, is_dir, signature) DFS parent-first snapshot of ``root``.

    Mirrors what ``TreeScanner._scan_tree_state`` records into ``next_state``:
    resolved path strings with their stat signature, parent before its children.
    """
    root = Path(root).resolve()
    out = [(str(root), root.is_dir(), _sig(root.stat(), root.is_dir()))]

    def rec(d):
        for entry in sorted(os.scandir(d), key=lambda e: e.path):
            is_dir = entry.is_dir()
            out.append((str(Path(entry.path)), is_dir, _sig(entry.stat(), is_dir)))
            if is_dir and not entry.is_symlink():
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
        snap = discover_sources_from_entries(
            _snapshot(tmp_path), get_default_registry()
        )

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
        under_skip = [
            p
            for p in seen
            if skip.resolve() in Path(p).parents or Path(p) == skip.resolve()
        ]
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


class TestSignaturePlumbing:
    """The state walk's signature rides the snapshot onto the ClaimContext (#56 item 6)."""

    def test_snapshot_signature_reaches_adapter(self, tmp_path):
        f = tmp_path / "img.tif"
        f.write_text("x")

        sigs = {}

        def spy(ctx, state):
            sigs[str(ctx.path_str)] = ctx.signature
            return []

        registry = get_default_registry()
        registry.get_claims_for_path = spy
        discover_sources_from_entries(_snapshot(tmp_path), registry)

        # Every probed entry carried a signature tuple shaped like the state walk's.
        assert sigs[str(f)] is not None
        assert isinstance(sigs[str(f)], tuple) and len(sigs[str(f)]) == 5

    def test_live_walk_context_has_no_signature(self):
        """Live-walk / join contexts carry no signature, so probes run uncached."""
        assert ClaimContext("/some/file.tif").signature is None
        assert ClaimContext("/d", is_dir=True).join("img.tif").signature is None


class TestCachedGlobNoReaddir:
    """Snapshot-driven claim globs are served from the cached child listing
    instead of re-reading the directory (biopb/biopb#65).

    A directory-claiming adapter globs its candidate directory up to 6× per
    rescan cycle; on cloud storage each glob is a directory-enumeration
    round-trip. The state walk already enumerated every directory's children, so
    the claim phase reuses that listing.
    """

    def _explode_on_readdir(self, monkeypatch):
        """Make any real directory read (Path.glob) raise, so a test fails loudly
        if a glob falls through to the filesystem instead of the cached listing."""

        def boom(self, pattern):
            raise AssertionError(f"Path.glob({pattern!r}) hit the filesystem")

        monkeypatch.setattr(Path, "glob", boom)

    def test_glob_matches_real_glob(self, tmp_path):
        """Cached glob is byte-identical to a real glob for the adapters' patterns."""
        for name in ["a.tif", "b.tif", "c.tiff", "C.TIF", "metadata.txt", "x.dcm"]:
            (tmp_path / name).write_text("x")
        listing = [str(p) for p in tmp_path.iterdir()]
        ctx = ClaimContext(tmp_path, is_dir=True, child_listing=listing)

        for pattern in ["*.tif", "*.tiff", "*.companion.ome", "metadata.txt", "*.dcm"]:
            cached = sorted(c.name for c in ctx.glob(pattern))
            real = sorted(p.name for p in tmp_path.glob(pattern))
            assert cached == real, pattern

    def test_glob_served_without_reading_directory(self, tmp_path, monkeypatch):
        """With a cached listing, glob never touches the filesystem."""
        listing = [
            str(tmp_path / "a.tif"),
            str(tmp_path / "b.tif"),
            str(tmp_path / "notes.txt"),
        ]
        ctx = ClaimContext(tmp_path, is_dir=True, child_listing=listing)
        self._explode_on_readdir(monkeypatch)

        tifs = sorted(c.name for c in ctx.glob("*.tif"))
        assert tifs == ["a.tif", "b.tif"]
        assert [c.name for c in ctx.glob("notes.txt")] == ["notes.txt"]
        assert ctx.glob("*.nope") == []

    def test_no_listing_falls_back_to_real_glob(self, tmp_path):
        """Without a cached listing, glob reads the directory as before."""
        (tmp_path / "a.tif").write_text("x")
        ctx = ClaimContext(tmp_path, is_dir=True)  # no child_listing
        assert [c.name for c in ctx.glob("*.tif")] == ["a.tif"]

    def test_multilevel_pattern_falls_back(self, tmp_path, monkeypatch):
        """A pattern spanning directory levels can't be served from a flat
        single-level listing, so it falls back to a real glob even when cached."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.tif").write_text("x")
        ctx = ClaimContext(tmp_path, is_dir=True, child_listing=[str(tmp_path / "sub")])
        # The cached path would never match a nested pattern; prove it falls
        # through by checking it finds the nested file a real glob would.
        assert [c.name for c in ctx.glob("sub/*.tif")] == ["a.tif"]

        # ...and that single-level patterns on the same ctx do NOT read disk.
        self._explode_on_readdir(monkeypatch)
        assert ctx.glob("*.tif") == []

    def test_snapshot_dir_ctx_carries_child_listing(self, tmp_path):
        """discover_sources_from_entries hands each directory its recorded
        children, and files carry no listing."""
        (tmp_path / "a.tif").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.tif").write_text("x")

        listings = {}

        def spy(ctx, state):
            listings[str(ctx.path_str)] = ctx._child_listing
            return []

        registry = get_default_registry()
        registry.get_claims_for_path = spy
        discover_sources_from_entries(_snapshot(tmp_path), registry)

        root = str(tmp_path.resolve())
        assert set(listings[root]) == {
            str(tmp_path.resolve() / "a.tif"),
            str(tmp_path.resolve() / "sub"),
        }
        # The directory's listing holds its own children.
        assert listings[str(tmp_path.resolve() / "sub")] == [
            str(tmp_path.resolve() / "sub" / "b.tif")
        ]
        # A file entry carries no listing (files never glob).
        assert listings[str(tmp_path.resolve() / "a.tif")] is None

    def test_tiff_sequence_claimed_from_snapshot_without_readdir(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: a TIFF sequence is claimed off the snapshot with the
        whole claim phase's globs served from memory (no directory reads)."""
        tifffile = pytest.importorskip("tifffile")
        import numpy as np

        seq = tmp_path / "seq"
        seq.mkdir()
        # A coherent numbered sequence; 30 frames clears the claim floor
        # (_MIN_TIFF_FILES). See tiff_sequence_test.py.
        for i in range(1, 31):
            tifffile.imwrite(
                str(seq / f"s1-{i:04d}_bf.tif"), np.zeros((4, 4), dtype="uint8")
            )

        snapshot = _snapshot(tmp_path)
        self._explode_on_readdir(monkeypatch)
        state = discover_sources_from_entries(snapshot, get_default_registry())

        claims = {c.primary_path: c.source_type for c in state.claims.values()}
        assert claims == {str(seq.resolve()): "tiff-sequence"}
