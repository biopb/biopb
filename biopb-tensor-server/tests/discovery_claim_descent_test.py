"""Discovery must not descend into a directory it just claimed (biopb/biopb#55).

`walk_with_identity_tracking` used to recurse into every directory unconditionally,
so when `discover_sources` claimed a container store mid-walk (e.g. a `.zarr`),
the walk still yielded every interior chunk file and the registry probed each one
for a claim that can never fire. For a chunked store that makes the walk's cost
proportional to the number of chunk *files* rather than logical sources.

These tests pin the new claim-feedback behavior: once a directory is claimed,
its subtree is never walked or probed.
"""

from pathlib import Path

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.discovery import (
    discover_sources,
    walk_with_identity_tracking,
)
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr


def _spy_registry(seen_paths):
    """Wrap the default registry so every probed path is recorded."""
    registry = get_default_registry()
    real = registry.get_claims_for_path

    def spy(ctx, state):
        seen_paths.append(str(ctx.path_str))
        return real(ctx, state)

    registry.get_claims_for_path = spy
    return registry


class TestClaimedDirsAreNotWalked:
    def test_chunk_files_under_claimed_store_are_not_probed(self, tmp_path):
        """A claimed OME-Zarr store's interior chunk files are never probed.

        Before the fix, every chunk file under the store was yielded by the walk
        and passed to get_claims_for_path; after it, only the store directory is.
        """
        pytest.importorskip("zarr")
        # Small chunks => hundreds of interior chunk files, so an accidental
        # descent would be unmistakable.
        store, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path), base_shape=(512, 512), chunk_size=(32, 32)
        )
        store = Path(store)

        seen = []
        registry = _spy_registry(seen)
        state = discover_sources(tmp_path, registry)

        # Exactly the store was claimed.
        claim_paths = {c.primary_path for c in state.claims.values()}
        assert claim_paths == {str(store)}

        # No path *inside* the store was ever probed for a claim.
        inside = [p for p in seen if store in Path(p).parents]
        assert inside == [], f"probed interior store paths: {inside[:5]} ..."

    def test_unclaimed_dirs_are_still_descended(self, tmp_path):
        """Plain subdirectories (no claim) keep being walked into.

        Guards against the callback over-pruning: a store nested under ordinary
        directories must still be discovered.
        """
        pytest.importorskip("zarr")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        store, _, _ = create_multiresolution_ome_zarr(str(nested))

        state = discover_sources(tmp_path, get_default_registry())

        claim_paths = {c.primary_path for c in state.claims.values()}
        assert claim_paths == {str(Path(store))}


class TestShouldDescendCallback:
    def test_should_descend_false_skips_subtree(self, tmp_path):
        """should_descend=False for a dir prunes its entire subtree from the walk."""
        keep = tmp_path / "keep"
        skip = tmp_path / "skip"
        keep.mkdir()
        skip.mkdir()
        (keep / "a.txt").write_text("a")
        (skip / "deep").mkdir()
        (skip / "deep" / "b.txt").write_text("b")

        visited = set()
        walked = {
            str(p)
            for p in walk_with_identity_tracking(
                tmp_path, visited, should_descend=lambda p: p.name != "skip"
            )
        }

        # The skip dir is yielded (so the consumer can claim it) but nothing
        # under it is.
        assert str(skip) in walked
        assert str(skip / "deep") not in walked
        assert str(skip / "deep" / "b.txt") not in walked
        # The sibling subtree is untouched by the prune.
        assert str(keep / "a.txt") in walked

    def test_no_callback_descends_everywhere(self, tmp_path):
        """Omitting should_descend preserves the original full-walk behavior."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "x.txt").write_text("x")

        visited = set()
        walked = {str(p) for p in walk_with_identity_tracking(tmp_path, visited)}

        assert str(sub) in walked
        assert str(sub / "x.txt") in walked
