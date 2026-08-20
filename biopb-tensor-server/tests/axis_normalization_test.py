"""Canonical axis-order normalization at the TensorAdapter seam (biopb/biopb#596).

Covers the rule (``core.axes.canonical_permutation``), the seam that applies it
(``core.normalize``), and the three scope decisions taken on the issue:

1. unlabeled stores (plain zarr / HDF5) are out of scope -- normalization must be
   provably the identity for them, and must not invent semantic labels;
2. the guarantee is unconditional on the read path, including for the geometry a
   read plan hands a client and for what lands in the chunk cache;
3. an axis order this server does not own is refused rather than permuted -- an
   uploader's declared order at ``create_source``, and a remote upstream's
   advertised order at the proxy's read boundary.
"""

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import TensorDescriptor, TensorReadOption
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.axes import canonical_axis, canonical_permutation
from biopb_tensor_server.core.config import CacheConfig, PyramidConfig
from biopb_tensor_server.core.normalize import (
    NormalizingAdapter,
    normalize_adapter,
)
from biopb_tensor_server.core.source_registry import SourceRegistry
from biopb_tensor_server.serving.server import TensorFlightServer


def _zarr_available() -> bool:
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


requires_zarr = pytest.mark.skipif(not _zarr_available(), reason="zarr not installed")


def _zarr_adapter(tmp, arr, dim_labels, name="src"):
    """A ZarrAdapter over ``arr``, advertising ``dim_labels`` as its axis order."""
    import zarr
    from biopb_tensor_server import ZarrAdapter

    zpath = str(Path(tmp) / f"{name}.zarr")
    chunks = tuple(max(1, d // 2) for d in arr.shape)
    z = zarr.open_array(
        zpath, mode="w", shape=arr.shape, chunks=chunks, dtype=arr.dtype
    )
    z[:] = arr
    labels = list(dim_labels) if dim_labels else None
    return ZarrAdapter(zarr.open_array(zpath, mode="r"), name, labels)


# ==============================================================================
# The rule
# ==============================================================================


class TestCanonicalPermutation:
    """``canonical_permutation`` is the single definition of canonical order."""

    @pytest.mark.parametrize(
        "labels,shape",
        [
            # The adapter survey's "compliant by construction" bucket: bioio's
            # fixed TCZYXS, OME-TIFF, qptiff, tiff-sequence, ndtiff, DICOM.
            (["t", "c", "z", "y", "x", "s"], [1, 1, 1, 512, 512, 3]),
            (["t", "c", "z", "y", "x"], [2, 3, 4, 5, 6]),
            (["y", "x"], [5, 6]),
            (["i", "z", "y", "x"], [3, 4, 5, 6]),
            (["frame", "y", "x"], [3, 4, 5]),
            (["z", "y", "x"], [4, 5, 6]),
            (["dim0", "dim1", "y", "x"], [2, 3, 4, 5]),
        ],
    )
    def test_compliant_orders_are_identity(self, labels, shape):
        assert canonical_permutation(labels, shape) is None

    @pytest.mark.parametrize(
        "labels,shape,expected",
        [
            # nifti: the one adapter family whose behavior actually changes.
            (["x", "y", "z"], [10, 20, 30], ["z", "y", "x"]),
            (["t", "x", "y", "z"], [4, 10, 20, 30], ["t", "z", "y", "x"]),
            (["v", "t", "x", "y", "z"], [2, 4, 10, 20, 30], ["v", "t", "z", "y", "x"]),
            (["x", "y"], [10, 20], ["y", "x"]),
            # A Z that sorts ahead of an unrecognized leading axis.
            (["z", "t", "y", "x"], [3, 4, 5, 6], ["t", "z", "y", "x"]),
            # Synonyms classify through the same vocabulary as everything else.
            (["width", "height", "depth"], [10, 20, 30], ["depth", "height", "width"]),
        ],
    )
    def test_divergent_orders_are_reordered(self, labels, shape, expected):
        perm = canonical_permutation(labels, shape)
        assert perm is not None
        assert [labels[p] for p in perm] == expected

    def test_leading_axes_keep_their_relative_order(self):
        """T/C/unrecognized form one group and are never reshuffled among
        themselves -- only moved ahead of the Z/Y/X/S trailing group."""
        labels = ["c", "t", "dimq", "x", "y"]
        perm = canonical_permutation(labels, [2, 3, 4, 5, 6])
        assert [labels[p] for p in perm] == ["c", "t", "dimq", "y", "x"]

    def test_an_unrecognized_label_keeps_its_order_not_its_index(self):
        """The precise claim, which is easy to overstate: ``dimq`` moved relative
        to nothing, but a trailing axis moved out from in front of it, so its
        index changed. The guarantee is about relative order, not position."""
        labels = ["z", "dimq", "y", "x"]
        perm = canonical_permutation(labels, [2, 3, 4, 5])
        assert [labels[p] for p in perm] == ["dimq", "z", "y", "x"]
        assert perm.index(1) == 0  # dimq: was axis 1, now leads

    def test_t_and_c_are_recognized_but_have_no_canonical_place(self):
        """The other half of the wording: ``canonical_axis`` classifies T and C,
        yet neither is part of ``[..., Z, Y, X, S]`` -- they ride in the leading
        group with the unlabeled rather than sorting into the tail."""
        assert canonical_axis("channel") == "c" and canonical_axis("frame") == "t"
        labels = ["z", "t", "c", "y", "x"]
        perm = canonical_permutation(labels, [2, 3, 4, 5, 6])
        assert [labels[p] for p in perm] == ["t", "c", "z", "y", "x"]

    # -- Decision 1: unlabeled stores are out of scope --------------------------

    @pytest.mark.parametrize(
        "labels", [["dim0", "dim1", "dim2"], ["a", "b"], [], ["dim0"]]
    )
    def test_unlabeled_axes_are_identity(self, labels):
        """Plain zarr / HDF5 emit ``dimN``. There is nothing to reorder, and
        relabeling them z/y/x would turn the consumers' positional *guess*
        into a wire *assertion* -- wrong for e.g. an unlabeled [y, x, c]."""
        assert canonical_permutation(labels, [2] * len(labels)) is None

    def test_unlabeled_axes_are_not_given_semantic_labels(self):
        """The permutation is all this rule produces; it never renames an axis."""
        labels = ["dim0", "dim1", "dim2"]
        assert canonical_permutation(labels, [4, 5, 6]) is None
        assert labels == ["dim0", "dim1", "dim2"]

    # -- fail-safe posture ------------------------------------------------------

    def test_rank_mismatch_is_identity(self):
        assert canonical_permutation(["y", "x"], [4, 5, 6]) is None

    def test_duplicate_canonical_axis_is_identity(self):
        """Two axes claiming the same role have no one right answer."""
        assert canonical_permutation(["y", "y", "x"], [4, 5, 6]) is None
        assert canonical_permutation(["x", "y", "x"], [4, 5, 6]) is None

    def test_samples_label_failing_the_size_gate_is_identity(self):
        """``samples_axis`` refuses an S axis that is not 3 or 4 deep; labels
        that lie about themselves are not a basis for moving pixels."""
        assert canonical_permutation(["x", "y", "s"], [10, 20, 2]) is None
        # ... and the same labels with a believable S do get normalized.
        assert canonical_permutation(["x", "y", "s"], [10, 20, 3]) == (1, 0, 2)


# ==============================================================================
# The seam
# ==============================================================================


@requires_zarr
class TestNormalizeAdapter:
    def test_compliant_adapter_is_returned_unchanged(self):
        """The common case keeps its object identity: no wrapper, no delegation
        on the hot path, no behavior change at all."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), ["z", "y", "x"])
            assert normalize_adapter(adapter) is adapter

    def test_unlabeled_adapter_is_returned_unchanged(self):
        """Decision 1, asserted at the seam and not just in the rule."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), None)
            assert list(adapter.get_tensor_descriptor().dim_labels) == [
                "dim0",
                "dim1",
                "dim2",
            ]
            assert normalize_adapter(adapter) is adapter

    def test_divergent_adapter_is_wrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), ["x", "y", "z"])
            assert isinstance(normalize_adapter(adapter), NormalizingAdapter)

    def test_wrapping_is_idempotent(self):
        """What makes normalization composable: applying it twice is applying it
        once, so a caller may normalize defensively without stacking wrappers."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), ["x", "y", "z"])
            once = normalize_adapter(adapter)
            assert normalize_adapter(once) is once

    def test_a_duck_typed_double_is_left_alone(self):
        class NotAnAdapter:
            source_id = "x"

        double = NotAnAdapter()
        assert normalize_adapter(double) is double

    def test_registry_applies_the_guarantee(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = SourceRegistry()
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), ["x", "y", "z"])
            returned = registry.register("src", adapter)
            assert isinstance(returned, NormalizingAdapter)
            assert registry.get("src") is returned

    def test_wrapping_is_announced_once(self, caplog):
        """Reordering a source is a visible behavior change; without a log line
        the only evidence of it is the transposed data. Once, not once per chunk
        read -- normalize_adapter is re-entered per get_tensor_adapter."""
        import logging as _logging

        from biopb_tensor_server.core import normalize as _normalize

        with tempfile.TemporaryDirectory() as tmp:
            _normalize._reported.clear()
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), ["x", "y", "z"])
            with caplog.at_level(_logging.INFO, logger=_normalize.__name__):
                for _ in range(3):
                    normalize_adapter(adapter)
            lines = [r for r in caplog.records if "axis normalization" in r.message]
            assert len(lines) == 1
            said = lines[0].getMessage()
            assert "['z', 'y', 'x']" in said and "['x', 'y', 'z']" in said

    def test_the_chunk_lookup_path_does_not_reclassify(self, monkeypatch):
        """``get_tensor_adapter`` / ``get_level_adapter`` sit on the do_get path
        -- the server resolves a chunk's adapter through them on *every* read --
        so running the source-level classifier there cost a full
        ``list_tensor_descriptors`` per chunk. A view needs no decision: whether
        it permutes is ``perm``'s business, made per access.
        """
        from biopb_tensor_server import ZarrAdapter

        calls = {"n": 0}
        real = ZarrAdapter.list_tensor_descriptors
        monkeypatch.setattr(
            ZarrAdapter,
            "list_tensor_descriptors",
            lambda self: (calls.__setitem__("n", calls["n"] + 1), real(self))[1],
        )

        with tempfile.TemporaryDirectory() as tmp:
            src = np.arange(2 * 3 * 8, dtype=np.uint16).reshape(2, 3, 8)
            registry = SourceRegistry()
            adapter = registry.register("src", _zarr_adapter(tmp, src, ["x", "y", "z"]))
            assert isinstance(adapter, NormalizingAdapter)
            plan = adapter.get_tensor_adapter(None).plan_flight_info(
                TensorReadOption(tensor_id="src"), PyramidConfig()
            )
            assert len(plan.chunk_endpoints) > 1

            calls["n"] = 0
            for ce in plan.chunk_endpoints:  # the do_get inner loop
                adapter.get_tensor_adapter(None).resolve_chunk_data(ce.chunk_id)
            assert calls["n"] == 0

    def test_registry_leaves_a_compliant_source_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = SourceRegistry()
            adapter = _zarr_adapter(tmp, np.zeros((4, 5, 6), np.uint8), ["z", "y", "x"])
            assert registry.register("src", adapter) is adapter
            assert registry.get("src") is adapter


@requires_zarr
class TestNormalizedDescriptorAndData:
    """Every per-axis surface moves together, or the view is incoherent."""

    def _wrapped(self, tmp):
        src = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)  # x, y, z
        adapter = normalize_adapter(_zarr_adapter(tmp, src, ["x", "y", "z"]))
        return adapter, src

    def test_descriptor_is_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, src = self._wrapped(tmp)
            desc = adapter.get_tensor_descriptor()
            assert list(desc.dim_labels) == ["z", "y", "x"]
            assert list(desc.shape) == [4, 3, 2]
            assert list(desc.chunk_shape) == [2, 1, 1]

    def test_source_descriptor_and_catalog_row_are_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, _ = self._wrapped(tmp)
            tensors = adapter.get_source_descriptor().tensors
            assert [list(t.dim_labels) for t in tensors] == [["z", "y", "x"]]
            assert [list(t.shape) for t in tensors] == [[4, 3, 2]]
            assert [list(t.dim_labels) for t in adapter.list_tensor_descriptors()] == [
                ["z", "y", "x"]
            ]

    def test_chunk_size_is_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, _ = self._wrapped(tmp)
            assert adapter.get_chunk_size() == (2, 1, 1)

    def test_get_data_takes_and_returns_canonical_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, src = self._wrapped(tmp)
            got = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[4, 3, 2]))
            assert got.shape == (4, 3, 2)
            np.testing.assert_array_equal(got, src.transpose(2, 1, 0))

    def test_get_data_subregion_is_the_canonical_subregion(self):
        """The bounds are read in canonical order, so a sub-box must land where
        the caller meant it to -- an inverse-permutation bug shows up here."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter, src = self._wrapped(tmp)
            got = adapter.get_data(ChunkBounds(start=[1, 0, 0], stop=[3, 2, 1]))
            np.testing.assert_array_equal(got, src.transpose(2, 1, 0)[1:3, 0:2, 0:1])

    def test_read_plan_geometry_matches_its_chunks(self):
        """The strongest local invariant: for every endpoint the plan advertises,
        the bytes DoGet returns for its chunk_id have exactly the shape the
        endpoint's bounds claim."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter, _ = self._wrapped(tmp)
            plan = adapter.plan_flight_info(
                TensorReadOption(tensor_id="src"), PyramidConfig()
            )
            assert list(plan.descriptor.dim_labels) == ["z", "y", "x"]
            assert list(plan.descriptor.shape) == [4, 3, 2]
            assert plan.chunk_endpoints
            for ce in plan.chunk_endpoints:
                expected = tuple(
                    stop - start
                    for start, stop in zip(ce.bounds.start, ce.bounds.stop, strict=True)
                )
                batch = adapter.resolve_chunk_data(ce.chunk_id)
                assert tuple(batch.column("shape").to_pylist()[0]) == expected

    def test_read_plan_reassembles_into_the_canonical_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, src = self._wrapped(tmp)
            plan = adapter.plan_flight_info(
                TensorReadOption(tensor_id="src"), PyramidConfig()
            )
            out = np.zeros(tuple(plan.descriptor.shape), dtype=np.uint16)
            for ce in plan.chunk_endpoints:
                from biopb_tensor_server.core.adapter_base import unpack_chunk_array

                arr = unpack_chunk_array(adapter.resolve_chunk_data(ce.chunk_id))
                sl = tuple(
                    slice(int(s), int(e))
                    for s, e in zip(ce.bounds.start, ce.bounds.stop, strict=True)
                )
                out[sl] = arr
            np.testing.assert_array_equal(out, src.transpose(2, 1, 0))

    def test_slice_hint_is_interpreted_in_canonical_order(self):
        """A client's hints arrive canonical and must be inverse-permuted before
        the delegate plans against them."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter, src = self._wrapped(tmp)
            read_opt = TensorReadOption(tensor_id="src")
            read_opt.slice_hint.start[:] = [0, 0, 0]
            read_opt.slice_hint.stop[:] = [2, 3, 2]
            plan = adapter.plan_flight_info(read_opt, PyramidConfig())
            # The canonical slice is interpreted correctly. The small source
            # retains its native grid to preserve endpoint parallelism.
            assert list(plan.descriptor.shape) == [2, 3, 2]
            assert list(plan.descriptor.slice_hint.start) == [0, 0, 0]
            assert list(plan.descriptor.slice_hint.stop) == [2, 3, 2]

    def test_scaled_read_is_coherent_in_canonical_order(self):
        """A downsampled read is the subtlest path: the client's ``scale_hint``
        is canonical, the delegate downsamples in native order inside the
        chunk_id, and the result comes back transposed. All three have to agree."""
        from biopb_tensor_server.core.adapter_base import unpack_chunk_array

        with tempfile.TemporaryDirectory() as tmp:
            src = (np.arange(4 * 32 * 64, dtype=np.uint16) % 251).reshape(4, 32, 64)
            adapter = normalize_adapter(_zarr_adapter(tmp, src, ["x", "y", "z"]))
            canonical = src.transpose(2, 1, 0)

            read_opt = TensorReadOption(tensor_id="src")
            read_opt.scale_hint[:] = [2, 2, 1]  # canonical: z/2, y/2, x untouched
            plan = adapter.plan_flight_info(read_opt, PyramidConfig())
            assert list(plan.descriptor.shape) == [32, 16, 4]
            assert list(plan.descriptor.scale_hint) == [2, 2, 1]

            out = np.zeros(tuple(plan.descriptor.shape), dtype=np.uint16)
            for ce in plan.chunk_endpoints:
                arr = unpack_chunk_array(adapter.resolve_chunk_data(ce.chunk_id))
                sl = tuple(
                    slice(int(s), int(e))
                    for s, e in zip(ce.bounds.start, ce.bounds.stop, strict=True)
                )
                assert arr.shape == tuple(s.stop - s.start for s in sl)
                out[sl] = arr
            expected = canonical.reshape(32, 2, 16, 2, 4).mean(axis=(1, 3))
            np.testing.assert_allclose(out, expected, atol=1)

    def test_advertised_pyramid_levels_are_canonical(self):
        """Each level carries its own per-axis ``shape`` and ``scale_hint``, so
        the permutation has to reach inside them too."""
        with tempfile.TemporaryDirectory() as tmp:
            src = np.zeros((128, 64, 4), np.uint16)  # x, y, z
            adapter = normalize_adapter(_zarr_adapter(tmp, src, ["x", "y", "z"]))
            plan = adapter.plan_flight_info(
                TensorReadOption(tensor_id="src", with_pyramid=True),
                PyramidConfig(threshold=32),
            )
            assert list(plan.descriptor.shape) == [4, 64, 128]
            assert len(plan.descriptor.pyramid) > 1
            for level in plan.descriptor.pyramid:
                expected = [
                    -(-full // scale)
                    for full, scale in zip(
                        plan.descriptor.shape, level.scale_hint, strict=True
                    )
                ]
                assert list(level.shape) == expected
            # A pyramid reduces X and Y. Those are the *canonical* trailing axes
            # here, not the ones the store labels x/y -- so a coarse level having
            # reduced index 2 is what proves the plan was built against the
            # normalized labels.
            coarsest = plan.descriptor.pyramid[-1]
            assert coarsest.scale_hint[2] > 1
            assert coarsest.shape[2] <= 32

    def test_writes_are_refused_rather_than_permuted(self):
        from biopb_tensor_server.core.errors import WriteNotSupportedError

        with tempfile.TemporaryDirectory() as tmp:
            adapter, _ = self._wrapped(tmp)
            with pytest.raises(WriteNotSupportedError, match="canonical"):
                adapter.put_chunk(
                    ChunkBounds(start=[0, 0, 0], stop=[1, 1, 1]), None, (1, 1, 1), "u2"
                )

    def test_the_permutation_is_re_derived_not_frozen(self):
        """``perm`` reads the inner adapter's labels every time.

        A memoized permutation cannot notice that its source's labels changed,
        and would keep transposing against a store that no longer needs it. The
        adapter family that actually mutates its descriptors in place -- the
        remote proxy -- is refused rather than wrapped, so nothing today can hit
        this; re-deriving keeps it true for whatever is wrapped next.
        """
        with tempfile.TemporaryDirectory() as tmp:
            inner = _zarr_adapter(tmp, np.zeros((2, 3, 4), np.uint8), ["x", "y", "z"])
            wrapper = normalize_adapter(inner)
            assert wrapper.perm == (2, 1, 0)

            inner.dim_labels = ["z", "y", "x"]
            assert wrapper.perm is None
            desc = wrapper.get_tensor_descriptor()
            assert list(desc.dim_labels) == ["z", "y", "x"]
            assert list(desc.shape) == [2, 3, 4]

    def test_an_undescribable_source_reports_no_permutation(self):
        """A descriptor that cannot be fetched degrades to identity -- and, since
        nothing is cached, the source normalizes normally once it can be."""
        with tempfile.TemporaryDirectory() as tmp:
            inner = _zarr_adapter(tmp, np.zeros((2, 3, 4), np.uint8), ["x", "y", "z"])
            wrapper = normalize_adapter(inner)
            broken = {"raise": True}
            real = inner.get_tensor_descriptor

            def flaky():
                if broken["raise"]:
                    raise RuntimeError("upstream down")
                return real()

            inner.get_tensor_descriptor = flaky
            assert wrapper.perm is None  # outage -> identity, not a crash
            broken["raise"] = False
            assert wrapper.perm == (2, 1, 0)  # recovered, not stranded

    def test_delegated_identity_fields_are_not_shadowed(self):
        """The wrapper inherits SourceAdapter's per-source class attributes, so
        each has to be re-declared as a delegating property or it reads None."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter, _ = self._wrapped(tmp)
            assert adapter.source_id == "src"
            assert adapter.array_id == "src"
            assert adapter.source_type == "zarr"
            assert adapter.source_url is not None
            assert adapter.content_version is not None
            assert adapter._source_url == adapter.source_url


@requires_zarr
class TestNormalizedCaching:
    """The cache must hold what the client is served, not what the store holds:
    on the localhost fast path the client reads the segment directly, with the
    server no longer in the loop to transpose it (hence the format bump)."""

    def test_cached_chunk_is_stored_in_canonical_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            CacheManager.reset()
            CacheManager.initialize(
                CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache"))
            )
            try:
                cache = CacheManager.get_instance()
                src = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
                adapter = normalize_adapter(_zarr_adapter(tmp, src, ["x", "y", "z"]))
                plan = adapter.plan_flight_info(
                    TensorReadOption(tensor_id="src"), PyramidConfig()
                )
                ce = plan.chunk_endpoints[0]
                first = adapter.resolve_chunk_data(ce.chunk_id, cache)
                # Second read is served out of the cache; it must agree with the
                # first, i.e. the transpose happened before the store, not after.
                second = adapter.resolve_chunk_data(ce.chunk_id, cache)
                assert first.equals(second)
                expected = tuple(
                    stop - start
                    for start, stop in zip(ce.bounds.start, ce.bounds.stop, strict=True)
                )
                assert tuple(second.column("shape").to_pylist()[0]) == expected
            finally:
                CacheManager.reset()

    def test_format_version_was_bumped_for_the_transpose(self):
        from biopb.tensor._pool import _CACHEFILE_SUPPORTED_FORMAT
        from biopb_tensor_server.cache.file_backend import CACHE_FILE_FORMAT_VERSION

        assert CACHE_FILE_FORMAT_VERSION >= 2
        # The layout did not change, so this client parses the new version; an
        # older one declines the fast path and falls back to do_get.
        assert _CACHEFILE_SUPPORTED_FORMAT >= CACHE_FILE_FORMAT_VERSION


# ==============================================================================
# Decision 3: writable sources are validated, not permuted
# ==============================================================================


class TestCreateSourceValidation:
    def _manager(self):
        from biopb_tensor_server.serving.upload_manager import UploadManager

        return UploadManager(SourceRegistry(), None, None)

    def test_non_canonical_upload_is_rejected(self):
        with pytest.raises(flight.FlightServerError, match="canonical"):
            self._manager().create_source(
                TensorDescriptor(
                    array_id="cache:bad",
                    dim_labels=["x", "y", "z"],
                    shape=[4, 5, 6],
                    chunk_shape=[4, 5, 6],
                    dtype="<u2",
                )
            )

    def test_the_error_names_the_order_to_use(self):
        with pytest.raises(flight.FlightServerError) as exc:
            self._manager().create_source(
                TensorDescriptor(
                    array_id="cache:bad",
                    dim_labels=["x", "y"],
                    shape=[4, 5],
                    chunk_shape=[4, 5],
                    dtype="<u2",
                )
            )
        assert "['y', 'x']" in str(exc.value)

    def test_canonical_upload_is_accepted(self):
        desc = self._manager().create_source(
            TensorDescriptor(
                array_id="cache:good",
                dim_labels=["z", "y", "x"],
                shape=[4, 5, 6],
                chunk_shape=[4, 5, 6],
                dtype="<u2",
            )
        )
        assert list(desc.dim_labels) == ["z", "y", "x"]

    def test_unlabeled_upload_is_accepted(self):
        """An uploader that declares no semantics is not forced to invent any."""
        desc = self._manager().create_source(
            TensorDescriptor(
                array_id="cache:plain",
                shape=[4, 5, 6],
                chunk_shape=[4, 5, 6],
                dtype="<u2",
            )
        )
        assert list(desc.shape) == [4, 5, 6]


# ==============================================================================
# End to end, over the wire
# ==============================================================================


@requires_zarr
class TestServedOverFlight:
    def test_client_sees_canonical_axes_and_matching_pixels(self):
        """What a real client gets: canonical dim_labels, a canonical shape, and
        pixels that agree with both."""
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        src = (np.arange(8 * 12 * 3, dtype=np.uint16) % 251).reshape(8, 12, 3)
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("nii", _zarr_adapter(tmp, src, ["x", "y", "z"], "nii"))
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(0.8)
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            desc = client.get_descriptor("nii")
            assert list(desc.dim_labels) == ["z", "y", "x"]
            assert list(desc.shape) == [3, 12, 8]
            got = client.get_tensor("nii").compute(scheduler="threads")
            np.testing.assert_array_equal(got, src.transpose(2, 1, 0))
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()

    def test_a_compliant_source_is_served_exactly_as_before(self):
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        src = (np.arange(8 * 12, dtype=np.uint16) % 251).reshape(8, 12)
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("img", _zarr_adapter(tmp, src, ["y", "x"], "img"))
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(0.8)
        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")
            desc = client.get_descriptor("img")
            assert list(desc.dim_labels) == ["y", "x"]
            np.testing.assert_array_equal(
                client.get_tensor("img").compute(scheduler="threads"), src
            )
            client.close()
        finally:
            server.shutdown()
            CacheManager.reset()


# ==============================================================================
# An order this server does not own: the remote proxy refuses, never permutes
# ==============================================================================


def _proxy_adapter(upstream_port, source_id="m", upstream_source_id="u"):
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    return RemoteTensorAdapter(
        source_id=source_id,
        upstream_location=f"grpc://localhost:{upstream_port}",
        upstream_source_id=upstream_source_id,
    )


def _legacy_upstream(tmp, arr, labels, name="u"):
    """A server advertising ``labels`` verbatim -- i.e. a pre-#596 upstream.

    Registered straight into the registry dict rather than through
    ``register_source``, which would normalize it and defeat the point: what is
    under test is a *downstream* facing a server that never learned the
    guarantee.
    """
    server = TensorFlightServer("grpc://localhost:0")
    server.sources._sources[name] = _zarr_adapter(tmp, arr, labels, name)
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(0.8)
    return server


@requires_zarr
class TestRemoteProxyRefusesRatherThanPermutes:
    """The upstream owns a mirrored source's axis order -- it mints the chunk_ids,
    plans the reads (#295) and sizes the grid -- so the server validates that
    order instead of permuting behind it, exactly as ``create_source`` does for an
    uploader's declared order."""

    def test_a_proxy_is_never_wrapped(self):
        """Not because it is compliant -- it is asserted non-canonical here -- but
        because it enforces the contract itself."""
        proxy = _proxy_adapter(1)
        proxy.seed_catalog(
            [{"array_id": "u", "dim_labels": ["x", "y", "z"], "shape": [2, 3, 4]}],
            {},
            True,
            None,
        )
        assert canonical_permutation(["x", "y", "z"], [2, 3, 4]) is not None
        assert normalize_adapter(proxy) is proxy
        assert SourceRegistry().register("m", proxy) is proxy

    def test_a_legacy_upstream_is_refused_at_open(self):
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        src = np.arange(2 * 3 * 8, dtype=np.uint16).reshape(2, 3, 8)
        up = _legacy_upstream(tmp, src, ["x", "y", "z"])
        down = TensorFlightServer("grpc://localhost:0")
        down.register_source("m", _proxy_adapter(up.port))
        down.mark_ready()
        threading.Thread(target=down.serve, daemon=True).start()
        time.sleep(0.8)
        try:
            client = TensorFlightClient(f"grpc://localhost:{down.port}")
            with pytest.raises(Exception) as exc:
                client.get_tensor("m")
            msg = str(exc.value)
            # Names the offending order, the order to use, and where to fix it.
            assert "['x', 'y', 'z']" in msg
            assert "['z', 'y', 'x']" in msg
            assert "upstream" in msg
            client.close()
        finally:
            down.shutdown()
            up.shutdown()
            CacheManager.reset()

    def test_the_catalog_keeps_a_refused_source_but_describe_refuses(self):
        """Where the refusal is drawn.

        ``list_sources`` still enumerates it -- refusing the read must not hide
        the broken thing from an operator. But a *describe* is refused along with
        a read: the descriptor it would hand back is itself the violation, and a
        consumer trusting it mis-maps its axes just as surely as one that read
        the pixels.
        """
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        src = np.arange(2 * 3 * 8, dtype=np.uint16).reshape(2, 3, 8)
        up = _legacy_upstream(tmp, src, ["x", "y", "z"])
        down = TensorFlightServer("grpc://localhost:0")
        down.register_source("m", _proxy_adapter(up.port))
        down.mark_ready()
        threading.Thread(target=down.serve, daemon=True).start()
        time.sleep(0.8)
        try:
            client = TensorFlightClient(f"grpc://localhost:{down.port}")
            assert "m" in client.list_sources()
            with pytest.raises(Exception, match="canonical"):
                client.get_descriptor("m")
            client.close()
        finally:
            down.shutdown()
            up.shutdown()
            CacheManager.reset()

    def test_a_canonical_upstream_is_mirrored_unchanged(self):
        from biopb.tensor.client import TensorFlightClient

        tmp = tempfile.mkdtemp()
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory"))
        src = (np.arange(4 * 3 * 8, dtype=np.uint16) % 251).reshape(4, 3, 8)
        up = _legacy_upstream(tmp, src, ["z", "y", "x"])
        down = TensorFlightServer("grpc://localhost:0")
        down.register_source("m", _proxy_adapter(up.port))
        down.mark_ready()
        threading.Thread(target=down.serve, daemon=True).start()
        time.sleep(0.8)
        try:
            client = TensorFlightClient(f"grpc://localhost:{down.port}")
            desc = client.get_descriptor("m")
            assert list(desc.dim_labels) == ["z", "y", "x"]
            np.testing.assert_array_equal(
                client.get_tensor("m").compute(scheduler="threads"), src
            )
            client.close()
        finally:
            down.shutdown()
            up.shutdown()
            CacheManager.reset()

    def test_an_upstream_upgrade_is_picked_up_without_re_registration(self):
        """The regression this design exists to prevent.

        ``seed_catalog`` replaces a mirror's descriptors in place on every
        reconcile pass, so any decision cached at registration goes stale. A
        wrapper would have frozen the permutation derived from the legacy labels
        and kept applying it to the now-canonical upstream -- serving reversed
        axes under a reversed descriptor. Because the check holds nothing, the
        same adapter object flips from refusing to serving.
        """

        def row(labels, shape):
            return [
                {
                    "array_id": "u",
                    "dim_labels": labels,
                    "shape": shape,
                    "chunk_shape": shape,
                    "dtype": "<u2",
                }
            ]

        proxy = _proxy_adapter(1)

        proxy.seed_catalog(row(["x", "y", "z"], [2, 3, 4]), {}, True, None)
        with pytest.raises(flight.FlightServerError, match="canonical"):
            proxy.get_read_plan(proxy.get_tensor_descriptor())

        # Same adapter object, no re-registration: the upstream upgraded and the
        # mirror follows on the very next read.
        proxy.seed_catalog(row(["z", "y", "x"], [4, 3, 2]), {}, True, None)
        plan = proxy.get_read_plan(proxy.get_tensor_descriptor())
        assert list(plan.descriptor.dim_labels) == ["z", "y", "x"]
        assert list(plan.descriptor.shape) == [4, 3, 2]
        # No permutation was ever applied, so there is none to un-apply: the
        # mirror is exactly what the upstream now advertises.
        assert (
            canonical_permutation(plan.descriptor.dim_labels, plan.descriptor.shape)
            is None
        )

    def test_the_refusal_wording_is_shared_with_create_source(self):
        """One rule stated once: both seams that validate an order they do not own
        report it through ``noncanonical_order``."""
        from biopb_tensor_server.core.axes import noncanonical_order
        from biopb_tensor_server.serving.upload_manager import UploadManager

        why = noncanonical_order(["x", "y"], [4, 5])
        assert why is not None and "['y', 'x']" in why

        with pytest.raises(flight.FlightServerError) as upload_exc:
            UploadManager(SourceRegistry(), None, None).create_source(
                TensorDescriptor(
                    array_id="cache:bad",
                    dim_labels=["x", "y"],
                    shape=[4, 5],
                    chunk_shape=[4, 5],
                    dtype="<u2",
                )
            )
        proxy = _proxy_adapter(1)
        proxy.seed_catalog(
            [{"array_id": "u", "dim_labels": ["x", "y"], "shape": [4, 5]}],
            {},
            True,
            None,
        )
        with pytest.raises(flight.FlightServerError) as proxy_exc:
            proxy.get_read_plan(proxy.get_tensor_descriptor())

        assert why in str(upload_exc.value)
        assert why in str(proxy_exc.value)
