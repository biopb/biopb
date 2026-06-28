"""Tests for TiffSequenceAdapter file-grouping / claiming logic.

Covers the single-varying-numeric-field grouping introduced to replace the old
"last number + equal file size" heuristic:
- numeric index anywhere in the name, with constant numeric tokens ignored
- compressed sequences with differing file sizes are still claimed
- dimension consistency is verified lazily in __init__, not in claim()
- multi-varying-field directories are rejected
- off-pattern sibling files are ignored, not fatal
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.tiff import (
    TiffSequenceAdapter,
    _group_tiff_sequence,
)
from biopb_tensor_server.discovery import ClaimContext, DiscoveryState


def _write_tiff(path: Path, shape=(8, 8), *, seed: int = 0, compression=None):
    """Write a small uint16 TIFF.

    The image is mostly zeros with a seed-proportional band of random pixels, so
    that compressed files have *differing* sizes (random data alone is
    incompressible and would yield identical sizes).
    """
    rng = np.random.default_rng(seed)
    data = np.zeros(shape, dtype=np.uint16)
    flat = data.reshape(-1)
    n_random = min(flat.size, seed * (flat.size // 8 + 1))
    if n_random:
        flat[:n_random] = rng.integers(0, 65535, size=n_random, dtype=np.uint16)
    tifffile.imwrite(str(path), data, compression=compression)


def _claim(tmpdir):
    ctx = ClaimContext(Path(tmpdir))
    state = DiscoveryState()
    claim = TiffSequenceAdapter.claim(ctx, state)
    return claim, state


class TestTiffSequenceClaim:
    def test_claim_middle_varying_field_with_constant_token(self):
        """`s1-NNNN_bf.tif`: constant `s1` ignored, NNNN is the index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"s1-{i:04d}_bf.tif" for i in range(1, 6)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)

            claim, state = _claim(tmpdir)

            assert claim is not None
            assert claim.source_type == "tiff-sequence"
            assert claim.primary_path == str(tmpdir)
            # Dir-claiming policy: the directory is the dataset boundary and the
            # only recorded member; the interior TIFFs are covered by the dir's
            # subtree prune, not enumerated individually.
            assert str(tmpdir) in state.consumed_paths
            assert claim.member_paths == {str(tmpdir)}

    def test_claim_with_differing_file_sizes(self):
        """Compressed sequence with all-distinct file sizes is still claimed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"s1-{i:04d}_bf.tif" for i in range(1, 6)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j, compression="zlib")

            sizes = {f.stat().st_size for f in files}
            assert len(sizes) > 1, "fixture should produce differing sizes"

            claim, _ = _claim(tmpdir)
            assert claim is not None

            # Same-shaped pixels -> instantiation succeeds.
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert len(adapter._tiff_files) == 5

    def test_inconsistent_dimensions_padded_not_rejected(self):
        """Mismatched Y/X no longer rejects the source: the tensor is sized to
        the per-axis maximum and smaller frames are zero-padded (biopb/biopb#198).

        Rejecting is fatal at cloud resolve (no fallback), so a few-pixel size
        drift between frames must not make the whole source unloadable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", shape=(8, 8), seed=i)
            # One larger frame -> the tensor grows to the max (16, 16).
            _write_tiff(Path(tmpdir) / "s1-0005_bf.tif", shape=(16, 16), seed=99)

            claim, _ = _claim(tmpdir)
            assert claim is not None  # claim() is metadata-free

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")  # must NOT raise
            assert adapter.full_shape == [5, 16, 16]

            # A small (8, 8) frame reads back padded to the (16, 16) chunk extent,
            # zeros outside its real data.
            chunk = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[1, 16, 16]))
            assert chunk.shape == (1, 16, 16)
            assert int(chunk[0, 8:, :].max()) == 0 and int(chunk[0, :, 8:].max()) == 0

            # The large frame reads back in full.
            big = adapter.get_data(ChunkBounds(start=[4, 0, 0], stop=[5, 16, 16]))
            assert big.shape == (1, 16, 16)

    def test_trailing_constant_number_orders_by_middle_field(self):
        """`s1-NNNN_bf2.tif`: trailing constant `2` ignored; order by NNNN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"s1-{i:04d}_bf2.tif" for i in range(1, 5)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)

            claim, _ = _claim(tmpdir)
            assert claim is not None

            ordered = _group_tiff_sequence(list(Path(tmpdir).glob("*.tif")))
            assert [f.name for f in ordered] == [
                "s1-0001_bf2.tif",
                "s1-0002_bf2.tif",
                "s1-0003_bf2.tif",
                "s1-0004_bf2.tif",
            ]

    def test_no_claim_multiple_varying_fields(self):
        """Two numeric fields varying together -> not a 1-D sequence -> reject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names = ["r1-c1_x.tif", "r2-c2_x.tif", "r3-c3_x.tif"]
            for j, n in enumerate(names):
                _write_tiff(Path(tmpdir) / n, seed=j)

            claim, _ = _claim(tmpdir)
            assert claim is None
            assert _group_tiff_sequence(list(Path(tmpdir).glob("*.tif"))) is None

    def test_sibling_non_numbered_ignored(self):
        """A digit-less sibling does not reject the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seq = [Path(tmpdir) / f"ND{i:03d}_aligned.tiff" for i in range(4)]
            for j, f in enumerate(seq):
                _write_tiff(f, seed=j)
            readme = Path(tmpdir) / "readme.tif"
            _write_tiff(readme, seed=42)

            claim, state = _claim(tmpdir)
            assert claim is not None
            # Dir-claiming: the directory is the recorded member; the digit-less
            # readme.tif neither rejects the claim nor is itself catalogued (the
            # dir-claim prunes the whole subtree).
            assert claim.member_paths == {str(tmpdir)}
            assert str(readme) not in state.consumed_paths

    def test_regression_end_number_pattern(self):
        """Original supported pattern still works and orders numerically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in (2, 0, 1):  # write out of order
                _write_tiff(Path(tmpdir) / f"ND{i:03d}_aligned.tiff", seed=i)

            claim, _ = _claim(tmpdir)
            assert claim is not None

            ordered = _group_tiff_sequence(list(Path(tmpdir).glob("*.tiff")))
            assert [f.name for f in ordered] == [
                "ND000_aligned.tiff",
                "ND001_aligned.tiff",
                "ND002_aligned.tiff",
            ]

    def test_no_claim_fewer_than_three(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 3):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_claim_when_micromanager_metadata_present(self):
        """A metadata.txt no longer blocks the claim.

        MicroManagerLegacyAdapter has higher priority and prunes any valid MM
        dataset before this adapter runs, so a metadata.txt that reaches here is
        one MM could not parse (e.g. truncated from an aborted acquisition). The
        sequence claim is the wanted fallback instead of N per-frame sources.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)
            (Path(tmpdir) / "metadata.txt").write_text("{ truncated")

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_claim_micromanager_img_frames_without_metadata(self):
        """img_* single-frame sequences (no metadata) are claimed as one source.

        Regression: these previously fell through every directory adapter and the
        walk registered each frame as its own per-file aics source.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):
                _write_tiff(Path(tmpdir) / f"img_{i:09d}__000.tif", seed=i)

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_no_claim_when_ome_companion_present(self):
        """*.companion.ome still defers to the file-level OmeTiffAdapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)
            (Path(tmpdir) / "set.companion.ome").write_text("<OME/>")

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_no_claim_for_ome_tiff_directory(self):
        """A folder of .ome.tif files is left to the file-level OmeTiffAdapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"img_{i:03d}.ome.tif", seed=i)

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_claim_ome_tiff_directory_under_cloud(self):
        """Under a cloud root OmeTiffAdapter is disabled, so the OME guards lift
        and a folder of .ome.tif is grouped into one sequence (claimed unresolved)
        instead of degrading to one per-file source per frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"img_{i:03d}.ome.tif", seed=i)

            ctx = ClaimContext(Path(tmpdir), cloud_root=True)
            claim = TiffSequenceAdapter.claim(ctx, DiscoveryState())
            assert claim is not None
            assert claim.source_type == "tiff-sequence"
            assert claim.unresolved is True

    def test_claim_ome_companion_under_cloud(self):
        """The *.companion.ome guard also lifts under a cloud root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"img_{i:03d}.ome.tif", seed=i)
            (Path(tmpdir) / "set.companion.ome").write_text("<OME/>")

            ctx = ClaimContext(Path(tmpdir), cloud_root=True)
            claim = TiffSequenceAdapter.claim(ctx, DiscoveryState())
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_init_matches_claim_order(self):
        """__init__ file order equals the claimed member order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"s1-{i:04d}_bf.tif" for i in range(1, 6)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)

            ordered = _group_tiff_sequence(list(Path(tmpdir).glob("*.tif")))
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter._tiff_files == ordered


class TestTiffSequenceInit:
    def test_empty_directory_raises(self):
        """A directory with no valid sequence -> __init__ raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No TIFF sequence found"):
                TiffSequenceAdapter(str(tmpdir), "sid")

    def test_heterogeneous_dtype_promotes_to_widest(self):
        """Mixed per-file dtypes no longer raise: the descriptor advertises the
        lossless promotion across the sequence and reads are cast to it
        (biopb/biopb#197/#198)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First three files uint16; the last one uint8 (with a non-zero
            # value, so the upcast is observable and lossless).
            for i in range(1, 4):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)
            tifffile.imwrite(
                str(Path(tmpdir) / "s1-0004_bf.tif"),
                np.full((8, 8), 7, dtype=np.uint8),
            )

            # Construction succeeds (no raise) and the descriptor advertises the
            # promoted dtype (uint16, wide enough for every frame).
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter._dtype == "uint16"
            assert adapter.get_tensor_descriptor().dtype == "uint16"

            # The uint8 file (index 3) reads back upcast to the descriptor dtype,
            # values preserved.
            chunk = adapter.get_data(ChunkBounds(start=[3, 0, 0], stop=[4, 8, 8]))
            assert str(chunk.dtype) == "uint16"
            assert int(chunk.max()) == 7

    def test_dtype_promotes_up_never_down(self):
        """The descriptor is the *widest* dtype, not the first file's: a uint8
        first frame among uint16 frames promotes to uint16 (so the uint16 frames
        are read losslessly), never the reverse (which would clip them)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First file uint8 (e.g. a preview frame), the rest uint16 with values
            # that exceed the uint8 range -- a down-cast would lose them.
            tifffile.imwrite(
                str(Path(tmpdir) / "s1-0001_bf.tif"),
                np.zeros((8, 8), dtype=np.uint8),
            )
            for i in range(2, 5):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"s1-{i:04d}_bf.tif"),
                    np.full((8, 8), 4000, dtype=np.uint16),
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter._dtype == "uint16"
            assert adapter.get_tensor_descriptor().dtype == "uint16"

            # The uint8 first frame upcasts cleanly...
            first = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[1, 8, 8]))
            assert str(first.dtype) == "uint16"
            # ...and the wide uint16 frames keep their full value (no clipping).
            wide = adapter.get_data(ChunkBounds(start=[1, 0, 0], stop=[2, 8, 8]))
            assert str(wide.dtype) == "uint16"
            assert int(wide.max()) == 4000

    def test_inconsistent_page_count_raises(self):
        """A file with a different page count -> __init__ raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"s1-{i:04d}_bf.tif"),
                    np.zeros((4, 8, 8), dtype=np.uint16),
                    photometric="minisblack",
                )
            # single-page file among multi-page ones
            tifffile.imwrite(
                str(Path(tmpdir) / "s1-0004_bf.tif"),
                np.zeros((8, 8), dtype=np.uint16),
            )
            with pytest.raises(ValueError, match="Inconsistent TIFF page count"):
                TiffSequenceAdapter(str(tmpdir), "sid")

    def test_explicit_dim_labels_single_page(self):
        """Caller-supplied dim_labels are used for single-page files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)

            adapter = TiffSequenceAdapter(
                str(tmpdir), "sid", dim_labels=["z", "y", "x"]
            )
            assert adapter.dim_labels == ["z", "y", "x"]
            assert adapter.full_shape == [3, 8, 8]

    def test_multi_page_files(self):
        """Multi-page files -> (num_files, pages, Y, X) with t,z,y,x labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                data = np.zeros((4, 8, 8), dtype=np.uint16)  # 4 pages
                tifffile.imwrite(
                    str(Path(tmpdir) / f"s1-{i:04d}_bf.tif"),
                    data,
                    photometric="minisblack",
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [3, 4, 8, 8]
            assert adapter.dim_labels == ["t", "z", "y", "x"]

    def test_tiled_tiff(self):
        """Tiled TIFFs drive the tile-info branch and chunk shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                data = np.zeros((32, 32), dtype=np.uint16)
                tifffile.imwrite(
                    str(Path(tmpdir) / f"s1-{i:04d}_bf.tif"), data, tile=(16, 16)
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.is_tiled is True
            assert adapter._spatial_chunk == [16, 16]
            assert adapter.chunk_shape == [1, 16, 16]


class TestGroupHelper:
    def test_dominant_mask_wins(self):
        """Mixed mask families: only the dominant group is returned, sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            big = [Path(tmpdir) / f"s1-{i:04d}_bf.tif" for i in range(1, 6)]
            small = [Path(tmpdir) / f"other_{i}.tif" for i in range(3)]
            for j, f in enumerate(big + small):
                _write_tiff(f, seed=j)

            ordered = _group_tiff_sequence(big + small)
            assert ordered == big

    def test_groups_micromanager_img_frames(self):
        """img_* frames are grouped (MM exclusion removed -- they are the fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names = ["img_000.tif", "img_001.tif", "img_002.tif"]
            files = [Path(tmpdir) / n for n in names]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files) == files

    def test_excludes_ome_off_cloud(self):
        """.ome.tif names are excluded off cloud (owned by OmeTiffAdapter)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"img_{i:03d}.ome.tif" for i in range(3)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files) is None

    def test_groups_ome_on_cloud(self):
        """Under cloud, .ome.tif are grouped (OmeTiffAdapter is disabled there)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"img_{i:03d}.ome.tif" for i in range(3)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files, exclude_ome=False) == files
