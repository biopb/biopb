"""Unit tests for the chunked connected-components plugin (biopb_mcp.plugins.chunked_label).

The plugin exists to prevent one silent wrong answer — objects straddling a chunk
boundary counted once per block — so the first thing pinned here is that failure
itself, alongside the correct result, on the fixture from issue #671.

Everything else is agreement with ``scipy.ndimage.label`` run on the whole array,
which is the definition this is reproducing: same object count, same partition,
same numbering-from-1, across chunk layouts, connectivities and dimensionality.
The corner case is called out separately because it is the one that survived a
first implementation: under diagonal connectivity two objects can meet where
2**ndim blocks touch, and an axis-aligned face never sees both pixels.

Also checks the documented limits (an instance segmentation merges; background
stays 0), laziness, and the delivery path. No kernel/display needed.
"""

import numpy as np
import pytest

da = pytest.importorskip("dask.array")
import scipy.ndimage as ndi  # noqa: E402

from biopb_mcp.plugins import chunked_label as cl  # noqa: E402

FULL = "full"
FACE = "face"


def _structure(kind, ndim):
    if kind == FULL:
        return np.ones((3,) * ndim, dtype=bool)
    return ndi.generate_binary_structure(ndim, 1)


def _same_partition(got, want):
    """Do two label images induce the same partition, ignoring label values?

    The two agree iff the observed ``(got, want)`` value pairs are a bijection.
    Counting pairs against *both* sides is what makes that a real check: a merge
    leaves more pairs than ``got`` has values, a split more than ``want`` has, and
    testing only one side misses one of the two.
    """
    if not np.array_equal(got > 0, want > 0):
        return False
    pairs = set(zip(got.ravel().tolist(), want.ravel().tolist(), strict=True))
    return len(pairs) == len(np.unique(got)) == len(np.unique(want))


def _agrees_with_scipy(mask, chunks, structure):
    labels, n = cl.label(da.from_array(mask, chunks=chunks), structure=structure)
    got, count = labels.compute(), int(n.compute())
    want, want_n = ndi.label(mask, structure)
    assert count == want_n
    assert _same_partition(got, want)
    return got, count


def _discs_on_a_seam(height=128, width=512, count=8, radius=20):
    """The #671 fixture: discs centred on the row a chunk boundary will fall on."""
    mask = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    for k in range(count):
        cy, cx = height // 2, 32 + k * (width - 64) // max(count - 1, 1)
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = 1
    return mask


def _naive_count(arr):
    """What ``map_blocks(scipy.ndimage.label)`` really reports: per-block counts.

    Not ``len(unique(...))`` — block-local labels collide across blocks, so that
    would undercount and hide the very bug this compares against.
    """
    per_block = arr.map_blocks(
        lambda b: np.full((1,) * b.ndim, ndi.label(b)[0].max()),
        dtype=np.int64,
        chunks=(1,) * arr.ndim,
    )
    return int(per_block.sum().compute())


class TestTheSeam:
    """The failure the plugin exists to prevent, and the correct answer beside it."""

    def test_objects_spanning_a_chunk_boundary_are_counted_once(self):
        mask = _discs_on_a_seam()
        arr = da.from_array(mask, chunks=(mask.shape[0] // 2, mask.shape[1]))
        _, n = cl.label(arr)
        assert int(n.compute()) == 8
        assert ndi.label(mask)[0].max() == 8  # ... which is the whole-array answer

    def test_the_naive_approach_double_counts_them(self):
        # Pinned so the comparison in the module docstring stays true: this is the
        # 8 -> 16 that makes the plugin worth having, and it is silent.
        mask = _discs_on_a_seam()
        arr = da.from_array(mask, chunks=(mask.shape[0] // 2, mask.shape[1]))
        assert _naive_count(arr) == 16

    def test_every_disc_really_does_span_both_chunks(self):
        # Guards the fixture, not the code: if the discs drifted off the boundary
        # the test above would pass without exercising anything.
        mask = _discs_on_a_seam()
        seam = mask.shape[0] // 2
        labels, _ = cl.label(da.from_array(mask, chunks=(seam, mask.shape[1])))
        got = labels.compute()
        for value in range(1, 9):
            rows = np.flatnonzero((got == value).any(axis=1))
            assert rows.min() < seam <= rows.max()

    def test_sizes_survive_the_reconciliation(self):
        mask = _discs_on_a_seam()
        seam = mask.shape[0] // 2
        labels, n = cl.label(da.from_array(mask, chunks=(seam, mask.shape[1])))
        sizes = cl.object_sizes(labels, int(n.compute())).compute()
        assert len(sizes) == 8
        assert sizes.sum() == int((mask > 0).sum())
        assert len(set(sizes.tolist())) == 1  # identical discs, identical areas


class TestAgreementWithScipy:
    @pytest.mark.parametrize("chunks", [(4, 4), (5, 7), (16, 3), (32, 32), (1, 32)])
    @pytest.mark.parametrize("kind", [FACE, FULL])
    def test_a_blobby_2d_field_matches_at_any_chunking(self, chunks, kind):
        rng = np.random.default_rng(11)
        mask = (ndi.uniform_filter(rng.random((32, 32)), 3) > 0.52).astype(np.uint8)
        assert mask.any() and not mask.all()
        _agrees_with_scipy(mask, chunks, _structure(kind, 2))

    @pytest.mark.parametrize("chunks", [(3, 3, 3), (8, 4, 5), (16, 16, 16)])
    @pytest.mark.parametrize("kind", [FACE, FULL])
    def test_a_blobby_3d_volume_matches_at_any_chunking(self, chunks, kind):
        rng = np.random.default_rng(7)
        mask = (ndi.uniform_filter(rng.random((16, 16, 16)), 2) > 0.5).astype(np.uint8)
        assert mask.any() and not mask.all()
        _agrees_with_scipy(mask, chunks, _structure(kind, 3))

    def test_a_1d_signal_matches(self):
        mask = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1], dtype=np.uint8)
        _agrees_with_scipy(mask, (3,), _structure(FACE, 1))

    def test_labels_are_numbered_contiguously_from_one(self):
        rng = np.random.default_rng(3)
        mask = (ndi.uniform_filter(rng.random((24, 24)), 3) > 0.52).astype(np.uint8)
        got, count = _agrees_with_scipy(mask, (5, 5), _structure(FACE, 2))
        assert sorted(np.unique(got).tolist()) == list(range(count + 1))


class TestTheCorner:
    """Diagonal connectivity where 2**ndim blocks meet — the axis-aligned blind spot."""

    def test_a_diagonal_pair_across_a_block_corner_is_one_object(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = mask[2, 2] = 1  # blocks (0,0) and (1,1) under chunks (2, 2)
        labels, n = cl.label(
            da.from_array(mask, chunks=(2, 2)), structure=np.ones((3, 3), bool)
        )
        assert int(n.compute()) == 1
        assert ndi.label(mask, np.ones((3, 3), bool))[0].max() == 1

    def test_the_backward_diagonal_across_the_same_corner_is_one_object(self):
        # Blocks (0,1) and (1,0) — never a "forward" neighbour of each other, so
        # this only works because the corner slab spans all four blocks.
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 2] = mask[2, 1] = 1
        labels, n = cl.label(
            da.from_array(mask, chunks=(2, 2)), structure=np.ones((3, 3), bool)
        )
        assert int(n.compute()) == 1

    def test_face_connectivity_keeps_that_pair_apart(self):
        # The same fixture under the default structure: two objects, and the
        # plugin must not invent a connection the structure does not allow.
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = mask[2, 2] = 1
        _, n = cl.label(da.from_array(mask, chunks=(2, 2)))
        assert int(n.compute()) == 2

    def test_a_3d_corner_where_eight_blocks_meet(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1, 1, 1] = mask[2, 2, 2] = 1
        structure = np.ones((3, 3, 3), bool)
        _, n = cl.label(da.from_array(mask, chunks=(2, 2, 2)), structure=structure)
        assert int(n.compute()) == 1


class TestDocumentedLimits:
    def test_an_instance_segmentation_merges_touching_objects(self):
        # Stated in the module docstring, and the reason this is not a tiled
        # instance-segmentation tool. Two touching cells are one component.
        instances = np.zeros((8, 8), dtype=np.int32)
        instances[2:6, 1:4] = 1
        instances[2:6, 4:7] = 2
        _, n = cl.label(da.from_array(instances, chunks=(4, 4)))
        assert int(n.compute()) == 1
        assert ndi.label(instances)[0].max() == 1  # scipy on the whole array agrees

    def test_background_stays_zero(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        labels, _ = cl.label(da.from_array(mask, chunks=(4, 4)))
        got = labels.compute()
        assert np.array_equal(got == 0, mask == 0)

    @pytest.mark.parametrize(
        "name,mask,chunks",
        [
            ("all background", np.zeros((8, 8), np.uint8), (3, 3)),
            ("all foreground", np.ones((8, 8), np.uint8), (3, 3)),
            ("one block", (np.mgrid[:8, :8][0] % 3 == 0).astype(np.uint8), (8, 8)),
            (
                "single-pixel chunks",
                (np.mgrid[:6, :6][0] % 2 == 0).astype(np.uint8),
                (1, 1),
            ),
        ],
    )
    def test_degenerate_inputs_match_scipy(self, name, mask, chunks):
        _agrees_with_scipy(mask, chunks, _structure(FACE, 2))

    def test_a_numpy_array_is_accepted(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        _, n = cl.label(mask)
        assert int(n.compute()) == 1


class TestLaziness:
    def test_nothing_is_computed_and_the_chunking_is_preserved(self):
        arr = da.from_array(np.ones((16, 16), np.uint8), chunks=(4, 8))
        labels, n = cl.label(arr)
        assert isinstance(labels, da.Array)
        assert isinstance(n, da.Array)
        assert labels.shape == arr.shape
        assert labels.chunks == arr.chunks
        assert labels.dtype == cl.LABEL_DTYPE

    def test_object_sizes_stays_lazy_and_drops_the_background(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 px
        mask[5:8, 5:8] = 1  # 9 px
        labels, n = cl.label(da.from_array(mask, chunks=(4, 4)))
        sizes = cl.object_sizes(labels, int(n.compute()))
        assert isinstance(sizes, da.Array)
        assert sorted(sizes.compute().tolist()) == [4, 9]  # not 51 background px


class TestSeeding:
    """The delivery path: the installer seeds the plugin into the kernel dir."""

    def test_seed_includes_the_label_plugin(self, tmp_path):
        from biopb_mcp.plugins._seed import SEED_FILES, seed_kernel_plugins

        assert "chunked_label.py" in SEED_FILES
        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)
        assert (dest / "chunked_label.py").exists()

    def test_seeded_file_loads_with_a_clean_namespace_surface(self, tmp_path):
        from biopb_mcp.mcp import _bootstrap
        from biopb_mcp.plugins._seed import seed_kernel_plugins

        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)
        # Other seeded plugins have their own surface tests; drop them so this
        # assertion stays an exact set for *this* file rather than a superset check.
        for other in dest.glob("*.py"):
            if other.name not in ("__init__.py", "chunked_label.py"):
                other.unlink()

        class IP:
            def __init__(self):
                self.user_ns = {"viewer": 1, "client": 1, "np": np, "da": 1, "ops": {}}

        ip = IP()
        _bootstrap._load_plugin_files(ip, dest)
        builtins_ = {"viewer", "client", "np", "da", "ops"}
        contributed = {
            n for n in ip.user_ns if not n.startswith("_") and n not in builtins_
        }
        assert contributed == {"chunked_label"}
        plug = ip.user_ns["chunked_label"]
        assert {"label", "object_sizes"} <= set(dir(plug))
        assert ip.user_ns["np"] is np  # reserved handle untouched

    def test_seeded_plugin_is_callable_from_the_namespace(self, tmp_path):
        from biopb_mcp.mcp import _bootstrap
        from biopb_mcp.plugins._seed import seed_kernel_plugins

        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)

        class IP:
            def __init__(self):
                self.user_ns = {"viewer": 1, "client": 1, "np": np, "da": 1, "ops": {}}

        ip = IP()
        _bootstrap._load_plugin_files(ip, dest)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        plug = ip.user_ns["chunked_label"]
        _, n = plug.label(da.from_array(mask, chunks=(4, 4)))
        assert int(n.compute()) == 1
