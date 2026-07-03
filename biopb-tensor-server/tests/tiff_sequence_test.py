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

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.tiff import (
    _COHERENT_FRACTION,
    _MIN_TIFF_FILES,
    TiffSequenceAdapter,
    _group_tiff_sequence,
    _looks_like_tiff_sequence,
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


# Enough coherent TIFFs to clear the claim floor (_MIN_TIFF_FILES). Small dirs
# now fall back to per-file sources, so claim tests write a full-size sequence.
_N = _MIN_TIFF_FILES


def _write_seq(dirpath, n=_N, template="s1-{i:04d}_bf.tif", *, start=1, **kw):
    """Write ``n`` coherent single-page TIFFs named by ``template``; return paths."""
    paths = []
    for i in range(start, start + n):
        p = Path(dirpath) / template.format(i=i)
        _write_tiff(p, seed=i, **kw)
        paths.append(p)
    return paths


def _claim(tmpdir):
    ctx = ClaimContext(Path(tmpdir))
    state = DiscoveryState()
    claim = TiffSequenceAdapter.claim(ctx, state)
    return claim, state


class TestTiffSequenceClaim:
    def test_claim_single_field_sequence(self):
        """`s1-NNNN_bf.tif`: a plain sequence is claimed (dir is the boundary)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir)

            claim, state = _claim(tmpdir)

            assert claim is not None
            assert claim.source_type == "tiff-sequence"
            assert claim.primary_path == str(tmpdir)
            assert str(tmpdir) in state.consumed_paths
            assert claim.member_paths == {str(tmpdir)}

    def test_claim_with_differing_file_sizes(self):
        """Compressed sequence with all-distinct file sizes is still claimed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _write_seq(tmpdir, compression="zlib")

            sizes = {f.stat().st_size for f in files}
            assert len(sizes) > 1, "fixture should produce differing sizes"

            claim, _ = _claim(tmpdir)
            assert claim is not None

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert len(adapter._tiff_files) == _N

    def test_claim_multi_field_filenames(self):
        """Stack-all (#215): two numeric fields varying together no longer block
        the claim -- the directory is claimed and the files stacked, with the
        agent left to interpret the axis from the names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="r{i}-c{i}_x.tif")

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_claim_metamorph_channel_site_time(self):
        """The driving #215 example: `..._w<c>NAME_s<s>_t<t>.TIF` (MetaMorph).

        Multi-axis names + uppercase extension are claimed and stacked; the
        per-file names reach the agent via get_metadata for reshape/relabel.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 0
            for ch in ("w1DIC", "w2GFP"):
                for s in (1, 2):
                    for t in range(1, 9):  # 2*2*8 = 32 frames, clears the floor
                        _write_tiff(
                            Path(tmpdir) / f"07122017_Sample2_{ch}_s{s}_t{t}.TIF",
                            seed=t,
                        )
                        n += 1

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [n, 8, 8]
            assert adapter.dim_labels == ["i", "y", "x"]
            assert len(adapter.get_metadata()["files"]) == n

    def test_uppercase_tif_extension_claimed(self):
        """Case-insensitive extension match: a folder of `.TIF` is claimed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="frame_{i}.TIF", start=0)

            claim, _ = _claim(tmpdir)
            assert claim is not None
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape[0] == _N

    def test_no_claim_below_min_files(self):
        """The claim floor: a handful of TIFFs (below _MIN_TIFF_FILES) is left to
        per-file fallback rather than welded into a sequence -- even when the
        names cohere perfectly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, n=_MIN_TIFF_FILES - 1)  # one short of the floor

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_no_claim_pattern_below_threshold(self):
        """Coherence gate: enough files to clear the floor, but fewer than the
        required fraction share one pattern -- a real sequence with too many
        unrelated strays reads as a grab-bag and is left to per-file fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 26 coherent frames + 4 unrelated strays = 30 files; 26/30 = 87% < 90%
            _write_seq(tmpdir, n=26, template="frame_{i:04d}.tif")
            for stray in ("alpha.tif", "beta.tif", "gamma.tif", "delta.tif"):
                _write_tiff(Path(tmpdir) / stray, seed=1)

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_claim_short_stem_numbered_sequence(self):
        """A numbered sequence with a tiny stem (a1/a2/a3...) coheres via its
        shared digit-template even though the common prefix is only one char."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="a{i}.tif")

            claim, _ = _claim(tmpdir)
            assert claim is not None

    def test_claim_indexed_channel_tokens(self):
        """An indexed channel set (sp_NNNN_{red,green,blue}) coheres via its
        shared stem even though no single digit-template reaches the threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 11):  # 10 indices x 3 channels = 30 files
                for c in ("red", "green", "blue"):
                    _write_tiff(Path(tmpdir) / f"sp_{i:04d}_{c}.tif", seed=i)

            claim, _ = _claim(tmpdir)
            assert claim is not None

    def test_claim_when_micromanager_metadata_present(self):
        """A metadata.txt no longer blocks the claim.

        MicroManagerLegacyAdapter has higher priority and prunes any valid MM
        dataset before this adapter runs, so a metadata.txt that reaches here is
        one MM could not parse (e.g. truncated from an aborted acquisition).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir)
            (Path(tmpdir) / "metadata.txt").write_text("{ truncated")

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_claim_micromanager_img_frames_without_metadata(self):
        """img_* single-frame sequences (no metadata) are claimed as one source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="img_{i:09d}__000.tif", start=0)

            claim, _ = _claim(tmpdir)
            assert claim is not None
            assert claim.source_type == "tiff-sequence"

    def test_no_claim_when_ome_companion_present(self):
        """*.companion.ome still defers to the file-level OmeTiffAdapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir)
            (Path(tmpdir) / "set.companion.ome").write_text("<OME/>")

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_no_claim_for_ome_tiff_directory(self):
        """A folder of .ome.tif files is left to the file-level OmeTiffAdapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="img_{i:03d}.ome.tif")

            claim, _ = _claim(tmpdir)
            assert claim is None

    def test_claim_ome_tiff_directory_under_cloud(self):
        """Under a cloud root OmeTiffAdapter is disabled, so the OME guards lift
        and a folder of .ome.tif is claimed (unresolved) instead of degrading to
        one per-file source per frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="img_{i:03d}.ome.tif")

            ctx = ClaimContext(Path(tmpdir), cloud_root=True)
            claim = TiffSequenceAdapter.claim(ctx, DiscoveryState())
            assert claim is not None
            assert claim.source_type == "tiff-sequence"
            assert claim.unresolved is True

    def test_claim_ome_companion_under_cloud(self):
        """The *.companion.ome guard also lifts under a cloud root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_seq(tmpdir, template="img_{i:03d}.ome.tif")
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

    def test_only_page_count_splits_the_stack(self):
        """Page count is the one un-normalizable mismatch -> a different-page file
        is a sibling. Differing shape and dtype are normalized into the stack
        (#198), not demoted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(4):  # (8,8) uint16 single-page
                _write_tiff(Path(tmpdir) / f"frame_{i}.tif", seed=i)
            _write_tiff(Path(tmpdir) / "frame_big.tif", shape=(16, 16), seed=9)
            tifffile.imwrite(  # different dtype -> promoted, not demoted
                str(Path(tmpdir) / "frame_u8.tif"), np.zeros((8, 8), np.uint8)
            )
            tifffile.imwrite(  # different page count -> the only sibling
                str(Path(tmpdir) / "frame_zstack.tif"),
                np.zeros((4, 8, 8), np.uint16),
                photometric="minisblack",
            )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            # 6 single-page files stacked, padded to the max plane, promoted dtype
            assert adapter.full_shape == [6, 16, 16]
            assert adapter._dtype == "uint16"
            assert adapter.get_metadata()["unstacked_files"] == ["frame_zstack.tif"]

    def test_dtype_promotes_up_never_down(self):
        """The descriptor takes the widest dtype; a uint16 value survives a
        uint8 sibling instead of clipping (#198)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tifffile.imwrite(
                str(Path(tmpdir) / "img_0.tif"), np.full((8, 8), 5, np.uint8)
            )
            tifffile.imwrite(
                str(Path(tmpdir) / "img_1.tif"), np.full((8, 8), 300, np.uint16)
            )
            tifffile.imwrite(
                str(Path(tmpdir) / "img_2.tif"), np.full((8, 8), 7, np.uint8)
            )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter._dtype == "uint16"
            out = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 8, 8]))
            assert out.dtype == np.uint16
            assert int(out[0, 0, 0]) == 5  # uint8 frame upcast
            assert int(out[1, 0, 0]) == 300  # uint16 value not clipped to 255

    def test_smaller_frame_zero_padded(self):
        """A frame smaller than the max plane reads back zero-padded (#198)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tifffile.imwrite(
                str(Path(tmpdir) / "img_0.tif"), np.full((4, 4), 7, np.uint16)
            )
            for i in (1, 2):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"img_{i}.tif"), np.full((8, 8), 3, np.uint16)
                )

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape == [3, 8, 8]
            out = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 8, 8]))
            assert int(out[0, 0, 0]) == 7 and int(out[0, 3, 3]) == 7  # data
            assert int(out[0, 4, 4]) == 0 and int(out[0, 7, 7]) == 0  # padding
            assert (out[1] == 3).all()  # full-size frame intact

    def test_init_rejects_incoherent_names(self):
        """The coherence gate runs at resolve too: an explicit source pointed at
        a grab-bag raises rather than welding it into a tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in ("logo.tif", "figure3.tif", "scalebar.tif"):
                _write_tiff(Path(tmpdir) / n, seed=1)

            with pytest.raises(ValueError, match="do not look like one sequence"):
                TiffSequenceAdapter(str(tmpdir), "sid")

    def test_init_page_count_split_reports_pages_not_names(self):
        """Regression (biopb/biopb): when coherent filenames fragment by page
        count, the resolve error must name the page-count split -- not blame the
        filenames. The VivaView case: sp11_07032020_DIC.tif etc. cohere as a set
        yet each is a different-length stack, so the dominant page-count bucket
        collapses to one file and cannot stack. The old message said 'do not look
        like one sequence', hiding the actual cause. (Resolve imposes no claim
        floor, so an explicitly-pointed small set still reaches this path.)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Names share the 'sp11_070' stem -> coherent; only the per-page-count
            # subsets are too small to stack.
            for date, pages in (("07032020", 1), ("07052020", 2), ("07092020", 3)):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"sp11_{date}_DIC.tif"),
                    np.zeros((pages, 8, 8), np.uint16),
                    photometric="minisblack",
                )

            with pytest.raises(ValueError) as excinfo:
                TiffSequenceAdapter(str(tmpdir), "sid")
            msg = str(excinfo.value)
            assert "page-count group" in msg  # names the real cause
            assert "do not look like one sequence" not in msg  # not a name complaint

    def test_init_grab_bag_with_page_split_blames_names_not_pages(self):
        """The page-count message must not be given to a genuine grab-bag that
        merely happens to have mixed page counts: the '...filenames do cohere...'
        claim would be false. An explicit-config source skips the claim-time
        coherence check, so unrelated names can reach resolve; the branch is gated
        on the coherence of ALL files, so this gets the filename complaint, not the
        page-count one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Unrelated names (no shared mask or stem), and mixed page counts so
            # the directory also fragments into >1 page-count bucket.
            for name, pages in (
                ("logo.tif", 1),
                ("photo.tif", 1),
                ("banner.tif", 1),
                ("chart.tif", 2),
                ("diagram.tif", 3),
            ):
                tifffile.imwrite(
                    str(Path(tmpdir) / name),
                    np.zeros((pages, 8, 8), np.uint16),
                    photometric="minisblack",
                )

            with pytest.raises(ValueError) as excinfo:
                TiffSequenceAdapter(str(tmpdir), "sid")
            msg = str(excinfo.value)
            assert "do not look like one sequence" in msg  # honest name complaint
            assert "page-count group" not in msg  # no false coherence claim

    def test_same_shape_sibling_is_stacked_and_listed(self):
        """A same-shape digit-less sibling (e.g. readme.tif) is physically
        stackable, so it IS stacked -- and visible in metadata for the agent to
        disregard. (Contrast the old behavior, which silently ignored it.) The
        sequence stays a large super-majority so the lone sibling doesn't trip the
        coherence gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 30
            for i in range(n):
                _write_tiff(Path(tmpdir) / f"ND{i:03d}_aligned.tiff", seed=i)
            _write_tiff(Path(tmpdir) / "readme.tif", seed=42)

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            assert adapter.full_shape[0] == n + 1
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

    def test_transport_error_propagates_not_demoted(self, monkeypatch):
        """An OSError while reading a member (e.g. a failed cloud recall) is
        re-raised, not swallowed into a silently-undersized stack. Contrast a
        corrupt file, which demotes (see test_unreadable_tiff_listed_as_unstacked).
        """
        import tifffile as tff

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                _write_tiff(Path(tmpdir) / f"img_{i}.tif", seed=i)

            real_open = tff.TiffFile
            failing = str(Path(tmpdir) / "img_1.tif")

            def fake_open(path, *a, **k):
                if str(path) == failing:
                    raise OSError("simulated recall failure")
                return real_open(path, *a, **k)

            monkeypatch.setattr(tff, "TiffFile", fake_open)
            with pytest.raises(OSError, match="simulated recall failure"):
                TiffSequenceAdapter(str(tmpdir), "sid")

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

    def test_trailing_singleton_samples_axis(self):
        """`(Y, X, 1)` TIFFs (series.axes 'YXQ') read back intact (#220).

        Files written from a `(Y, X, 1)` array carry a trailing singleton
        samples axis, so `series.aszarr()` yields a 3-D `(Y, X, 1)` store. The
        reader must take Y/X from the axes string -- not `shape[-2:]` (which is
        `(X, samples)`) -- and must not treat the 3-D shape as `(page, Y, X)`,
        which collapsed every plane to its first column.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            planes = [
                np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) + i * 100
                for i in range(3)
            ]
            for i, plane in enumerate(planes):
                tifffile.imwrite(
                    str(Path(tmpdir) / f"img_{i}.tif"), plane.reshape(8, 8, 1)
                )
                # sanity: the fixture really has the trailing samples axis
                with tifffile.TiffFile(str(Path(tmpdir) / f"img_{i}.tif")) as tf:
                    assert tf.series[0].axes == "YXQ"

            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            # descriptor stays 2-D spatial: the samples axis is not modeled
            assert adapter.full_shape == [3, 8, 8]
            assert adapter.dim_labels == ["i", "y", "x"]

            out = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 8, 8]))
            assert out.shape == (3, 8, 8)
            for i, plane in enumerate(planes):
                np.testing.assert_array_equal(out[i], plane)

            # a sub-region read must also map Y/X correctly (not just full plane)
            sub = adapter.get_data(ChunkBounds(start=[1, 2, 3], stop=[2, 6, 7]))
            assert sub.shape == (1, 4, 4)
            np.testing.assert_array_equal(sub[0], planes[1][2:6, 3:7])

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


class TestCoherenceGate:
    """_looks_like_tiff_sequence: the filename-only coherence check. Pure, no I/O,
    so the >=90% fraction and the small-sample floor are tested directly on name
    lists rather than by fabricating dozens of files per case."""

    def test_mask_shared_by_exactly_the_threshold_coheres(self):
        # 27 of 30 share the mask -> exactly 90% -> coheres.
        names = [f"frame_{i:03d}.tif" for i in range(27)]
        names += ["aaa.tif", "bbb.tif", "ccc.tif"]
        assert len(names) == 30
        assert _looks_like_tiff_sequence(names) is True

    def test_mask_below_threshold_does_not_cohere(self):
        # 26 of 30 share the mask; the 4 strays share no stem -> < 90% -> grab-bag.
        names = [f"frame_{i:03d}.tif" for i in range(26)]
        names += ["aardvark.tif", "bison.tif", "cheetah.tif", "dingo.tif"]
        assert len(names) == 30
        assert _looks_like_tiff_sequence(names) is False

    def test_stem_shared_by_threshold_coheres_without_mask_majority(self):
        # 27 files share the stem 'exp_000' across three rotating channel masks
        # (each mask only 9 of 30, so it is the stem -- not a mask -- that carries
        # coherence); 3 strays make up the balance.
        names = [
            f"exp_{i:04d}_{tok}.tif"
            for i in range(9)
            for tok in ("red", "green", "blue")
        ]
        names += ["aaa.tif", "bbb.tif", "ccc.tif"]
        assert len(names) == 30
        assert _looks_like_tiff_sequence(names) is True

    def test_below_pattern_floor_never_coheres(self):
        # Under _MIN_PATTERN_FILES a shared mask/stem is trivially met and means
        # nothing, so it is rejected regardless.
        assert _looks_like_tiff_sequence(["a.tif", "b.tif"]) is False

    def test_small_but_coherent_set_passes_the_pattern_check(self):
        # The pattern check itself has only the small-sample floor (the 30-file
        # bar is the separate claim floor), so a tiny coherent set still reads as
        # a sequence -- this is the resolve path for an explicit small source.
        assert _looks_like_tiff_sequence(["s1.tif", "s2.tif", "s3.tif"]) is True

    def test_threshold_matches_the_configured_fraction(self):
        # Guard the boundary against drift if _COHERENT_FRACTION changes.
        n = 30
        need = math.ceil(_COHERENT_FRACTION * n)
        at = [f"f_{i}.tif" for i in range(need)] + [
            f"x{j}_.tif" for j in range(n - need)
        ]
        below = [f"f_{i}.tif" for i in range(need - 1)] + [
            f"x{j}_.tif" for j in range(n - need + 1)
        ]
        assert _looks_like_tiff_sequence(at) is True
        assert _looks_like_tiff_sequence(below) is False


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


class TestTiffSequencePerFileLock:
    """get_data locks per file, not adapter-wide: a stalled read of one file must
    not block reads of *other* files (the interactive-scrub freeze). The former
    single ``_io_lock`` serialized every get_data, so one slow read (a cloud/VM
    I/O stall) froze every other frame the async viewer / precache asked for."""

    def test_distinct_files_use_distinct_locks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"img_{i:03d}.tif" for i in range(3)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j + 1)
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")
            a, b = adapter._tiff_files[0], adapter._tiff_files[1]
            assert adapter._lock_for(a) is adapter._lock_for(a)  # stable per file
            assert adapter._lock_for(a) is not adapter._lock_for(b)  # not shared

    def test_slow_read_of_one_file_does_not_block_another(self, monkeypatch):
        import threading

        import tifffile as tifffile_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            files = [Path(tmpdir) / f"img_{i:03d}.tif" for i in range(3)]
            for j, f in enumerate(files):
                _write_tiff(f, seed=j + 1)
            adapter = TiffSequenceAdapter(str(tmpdir), "sid")

            # Stall the read of the first stacked member's file (held *inside* its
            # per-file lock, since get_data acquires the lock then opens the file).
            blocked_path = str(adapter._tiff_files[0])
            entered = threading.Event()  # frame-0 read is now holding file-0's lock
            gate = threading.Event()  # release frame-0
            real_TiffFile = tifffile_mod.TiffFile

            def gated_TiffFile(path, *a, **k):
                if str(path) == blocked_path:
                    entered.set()
                    assert gate.wait(timeout=5), "gate never released"
                return real_TiffFile(path, *a, **k)

            monkeypatch.setattr(tifffile_mod, "TiffFile", gated_TiffFile)

            errors: dict = {}

            def read_frame0():
                try:
                    adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[1, 8, 8]))
                except Exception as e:  # pragma: no cover - safety net
                    errors["f0"] = e

            out: dict = {}
            done = threading.Event()

            def read_frame1():
                out["v"] = adapter.get_data(
                    ChunkBounds(start=[1, 0, 0], stop=[2, 8, 8])
                )
                done.set()

            t0 = threading.Thread(target=read_frame0)
            t0.start()
            assert entered.wait(timeout=5), "frame-0 read never started"
            try:
                # Frame 1 is a DIFFERENT file -> different lock -> must complete
                # while frame 0 is stalled. With the old adapter-wide lock this
                # would hang until the gate releases (asserts on timeout).
                t1 = threading.Thread(target=read_frame1)
                t1.start()
                assert done.wait(timeout=3), (
                    "frame-1 read blocked behind stalled frame-0"
                )
                t1.join()
                assert out["v"].shape == (1, 8, 8)
            finally:
                gate.set()
                t0.join(timeout=5)
            assert not errors
