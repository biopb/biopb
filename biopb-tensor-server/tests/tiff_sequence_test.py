"""Tests for TiffSequenceAdapter claiming + stack-all behavior (#215).

Stack-all policy: every uniformly-shaped TIFF in a directory is stacked along an
opaque file axis (label ``i``); the axis's semantic structure (channel / time /
site / z) is NOT inferred. Per-file names are exposed via get_metadata() for a
downstream agent to interpret. Covered here:
- claim is metadata-free and no longer needs a single varying numeric field;
  multi-field names (index x channel, MetaMorph ``_w/_s/_t``) are claimed too
- the dominant *shape* bucket is stacked; odd-shaped/dtype/page siblings are not
  stacked but ARE surfaced in metadata (nothing silently dropped)
- uppercase ``.TIF`` extensions are matched (MetaMorph and friends)
- the retained _group_tiff_sequence helper (single-field ordering reference)
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
    def test_claim_single_field_sequence(self):
        """`s1-NNNN_bf.tif`: a plain sequence is claimed (dir is the boundary)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"s1-{i:04d}_bf.tif" for i in range(1, 6)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)

            claim, state = _claim(tmpdir)

            assert claim is not None
            assert claim.source_type == "tiff-sequence"
            assert claim.primary_path == str(tmpdir)
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

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert len(adapter._tiff_files) == 5

    def test_claim_multi_field_filenames(self):
        """Stack-all (#215): two numeric fields varying together no longer block
        the claim -- the directory is claimed and the files stacked, with the
        agent left to interpret the axis from the names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names = ["r1-c1_x.tif", "r2-c2_x.tif", "r3-c3_x.tif"]
            for j, n in enumerate(names):
                _write_tiff(Path(tmpdir) / n, seed=j)

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_claim_metamorph_channel_site_time(self):
        """The driving #215 example: `..._w<c>NAME_s<s>_t<t>.TIF` (MetaMorph).

        Multi-axis names + uppercase extension are claimed and stacked; the
        per-file names reach the agent via get_metadata for reshape/relabel.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for ch in ("w1DIC", "w2GFP"):
                for s in (1, 2):
                    for t in (1, 2, 10):
                        _write_tiff(
                            Path(tmpdir) / f"07122017_Sample2_{ch}_s{s}_t{t}.TIF",
                            seed=t,
                        )

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [12, 8, 8]
            assert adapter.dim_labels == ["i", "y", "x"]
            files = adapter.get_metadata()["files"]
            assert len(files) == 12
            # Natural order: t1, t2, t10 within a (channel, site) group.
            assert files[:3] == [
                "07122017_Sample2_w1DIC_s1_t1.TIF",
                "07122017_Sample2_w1DIC_s1_t2.TIF",
                "07122017_Sample2_w1DIC_s1_t10.TIF",
            ]

    def test_uppercase_tif_extension_claimed(self):
        """Case-insensitive extension match: a folder of `.TIF` is claimed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                _write_tiff(Path(tmpdir) / f"frame_{i}.TIF", seed=i)

            claim, _ = _claim(tmpdir)
            assert claim is not None
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape[0] == 3

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
        one MM could not parse (e.g. truncated from an aborted acquisition).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 5):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)
            (Path(tmpdir) / "metadata.txt").write_text("{ truncated")

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_claim_micromanager_img_frames_without_metadata(self):
        """img_* single-frame sequences (no metadata) are claimed as one source."""
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
        and a folder of .ome.tif is claimed (unresolved) instead of degrading to
        one per-file source per frame."""
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


class TestTiffSequenceStackAll:
    def test_stacks_all_uniform_files_with_provenance(self):
        """All uniform files stack; metadata.files is index-aligned to axis 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names = [f"sp29_{i:04d}_{c}.tif" for i in range(1, 4) for c in "rgb"]
            for j, n in enumerate(names):
                _write_tiff(Path(tmpdir) / n, seed=j)

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [9, 8, 8]
            assert adapter.dim_labels == ["i", "y", "x"]
            md = adapter.get_metadata()
            assert len(md["files"]) == 9
            assert "unstacked_files" not in md
            # index-aligned: metadata order == stacked file order
            assert md["files"] == [p.name for p in adapter._tiff_files]

    def test_odd_shaped_dtype_page_siblings_unstacked_not_dropped(self):
        """The dominant shape bucket stacks; files differing in shape, dtype, or
        page-count are NOT stacked but ARE surfaced -- no raise, nothing lost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):  # dominant: (8,8) uint16 single-page
                _write_tiff(Path(tmpdir) / f"frame_{i}.tif", seed=i)
            _write_tiff(Path(tmpdir) / "overview.tif", shape=(16, 16), seed=9)
            tifffile.imwrite(  # different dtype
                str(Path(tmpdir) / "mask.tif"), np.zeros((8, 8), np.uint8)
            )
            tifffile.imwrite(  # different page count
                str(Path(tmpdir) / "zstack.tif"),
                np.zeros((4, 8, 8), np.uint16),
                photometric="minisblack",
            )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [4, 8, 8]
            unstacked = adapter.get_metadata()["unstacked_files"]
            assert set(unstacked) == {"overview.tif", "mask.tif", "zstack.tif"}

    def test_same_shape_sibling_is_stacked_and_listed(self):
        """A same-shape digit-less sibling (e.g. readme.tif) is physically
        stackable, so it IS stacked -- and visible in metadata for the agent to
        disregard. (Contrast the old behavior, which silently ignored it.)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):
                _write_tiff(Path(tmpdir) / f"ND{i:03d}_aligned.tiff", seed=i)
            _write_tiff(Path(tmpdir) / "readme.tif", seed=42)

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape[0] == 5
            assert "readme.tif" in adapter.get_metadata()["files"]

    def test_unreadable_tiff_listed_as_unstacked(self):
        """A corrupt/unreadable TIFF is not stacked but is surfaced, not fatal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                _write_tiff(Path(tmpdir) / f"frame_{i}.tif", seed=i)
            (Path(tmpdir) / "truncated.tif").write_bytes(b"II*\x00garbage")

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape[0] == 3
            assert "truncated.tif" in adapter.get_metadata()["unstacked_files"]

    def test_natural_sort_order(self):
        """The file axis defaults to numeric-aware order: img_2 before img_10."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in (1, 2, 10):
                _write_tiff(Path(tmpdir) / f"img_{i}.tif", seed=i)

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.get_metadata()["files"] == [
                "img_1.tif",
                "img_2.tif",
                "img_10.tif",
            ]

    def test_get_data_returns_ordered_stack(self):
        """get_data reads the stacked axis in file order (real, ordered data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"img_{i}.tif"),
                    np.full((8, 8), i, dtype=np.uint16),
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            out = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 8, 8]))
            assert out.shape == (3, 8, 8)
            assert [int(out[k, 0, 0]) for k in range(3)] == [0, 1, 2]

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No TIFF files found"):
                TiffSequenceAdapter(str(tmpdir), "sid")

    def test_explicit_dim_labels_single_page(self):
        """Caller-supplied dim_labels override the default 'i' file axis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                _write_tiff(Path(tmpdir) / f"s1-{i:04d}_bf.tif", seed=i)

            adapter = TiffSequenceAdapter(
                str(tmpdir), "sid", dim_labels=["z", "y", "x"]
            )
            assert adapter.dim_labels == ["z", "y", "x"]
            assert adapter.full_shape == [3, 8, 8]

    def test_multi_page_files(self):
        """Multi-page files -> (num_files, pages, Y, X) with i,z,y,x labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"s1-{i:04d}_bf.tif"),
                    np.zeros((4, 8, 8), dtype=np.uint16),
                    photometric="minisblack",
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [3, 4, 8, 8]
            assert adapter.dim_labels == ["i", "z", "y", "x"]

    def test_tiled_tiff(self):
        """Tiled TIFFs drive the tile-info branch and chunk shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"s1-{i:04d}_bf.tif"),
                    np.zeros((32, 32), dtype=np.uint16),
                    tile=(16, 16),
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.is_tiled is True
            assert adapter._spatial_chunk == [16, 16]
            assert adapter.chunk_shape == [1, 16, 16]


class TestGroupHelper:
    """_group_tiff_sequence is retained (single-field ordering reference); it is
    no longer the claim gate under stack-all (#215)."""

    def test_dominant_mask_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            big = [Path(tmpdir) / f"s1-{i:04d}_bf.tif" for i in range(1, 6)]
            small = [Path(tmpdir) / f"other_{i}.tif" for i in range(3)]
            for j, f in enumerate(big + small):
                _write_tiff(f, seed=j)

            ordered = _group_tiff_sequence(big + small)
            assert ordered == big

    def test_orders_by_single_varying_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in (2, 0, 1):  # write out of order
                _write_tiff(Path(tmpdir) / f"ND{i:03d}_aligned.tiff", seed=i)

            ordered = _group_tiff_sequence(list(Path(tmpdir).glob("*.tiff")))
            assert [f.name for f in ordered] == [
                "ND000_aligned.tiff",
                "ND001_aligned.tiff",
                "ND002_aligned.tiff",
            ]

    def test_multi_varying_field_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            names = ["r1-c1_x.tif", "r2-c2_x.tif", "r3-c3_x.tif"]
            for j, n in enumerate(names):
                _write_tiff(Path(tmpdir) / n, seed=j)
            assert _group_tiff_sequence(list(Path(tmpdir).glob("*.tif"))) is None

    def test_groups_micromanager_img_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            names = ["img_000.tif", "img_001.tif", "img_002.tif"]
            files = [Path(tmpdir) / n for n in names]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files) == files

    def test_excludes_ome_off_cloud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"img_{i:03d}.ome.tif" for i in range(3)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files) is None

    def test_groups_ome_on_cloud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"img_{i:03d}.ome.tif" for i in range(3)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j)
            assert _group_tiff_sequence(files, exclude_ome=False) == files
