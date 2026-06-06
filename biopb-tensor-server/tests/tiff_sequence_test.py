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
            for f in files:
                assert str(f) in state.consumed_paths

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

    def test_inconsistent_dimensions_claimed_but_init_raises(self):
        """Template matches -> claim() ok; mismatched Y/X -> __init__ raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", shape=(8, 8), seed=i)
            # One file with a different spatial shape.
            _write_tiff(Path(tmpdir) / "s1-0005_bf.tif", shape=(16, 16), seed=99)

            claim, _ = _claim(tmpdir)
            assert claim is not None  # claim() is metadata-free

            with pytest.raises(ValueError, match="Inconsistent TIFF dimensions"):
                TiffSequenceAdapter(str(tmpdir), "sid")

    def test_trailing_constant_number_orders_by_middle_field(self):
        """`s1-NNNN_bf2.tif`: trailing constant `2` ignored; order by NNNN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"s1-{i:04d}_bf2.tif" for i in range(1, 5)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)

            claim, _ = _claim(tmpdir)
            assert claim is not None

            ordered = _group_tiff_sequence(
                list(Path(tmpdir).glob("*.tif"))
            )
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
            for f in seq:
                assert str(f) in state.consumed_paths
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

    def test_no_claim_when_metadata_present(self):
        """metadata.txt guard is unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)
            (Path(tmpdir) / "metadata.txt").write_text("{}")

            claim, _ = _claim(tmpdir)
            assert claim is None

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

    def test_inconsistent_dtype_raises(self):
        """A file with a different dtype -> __init__ raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)
            # uint8 instead of the uint16 the others use
            tifffile.imwrite(
                str(Path(tmpdir) / "s1-0004_bf.tif"),
                np.zeros((8, 8), dtype=np.uint8),
            )
            with pytest.raises(ValueError, match="Inconsistent TIFF dtype"):
                TiffSequenceAdapter(str(tmpdir), "sid")

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

            adapter = TiffSequenceAdapter(str(tmpdir), "sid", dim_labels=["z", "y", "x"])
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

    def test_excludes_ome_and_micromanager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            names = ["img_000.tif", "img_001.tif", "img_002.tif"]
            files = [Path(tmpdir) / n for n in names]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files) is None
