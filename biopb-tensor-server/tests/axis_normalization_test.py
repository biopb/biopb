"""Canonical axis-order normalization at the TensorAdapter seam (biopb/biopb#596).

Covers the rule (``core.axes.canonical_permutation``), the seam that applies it
(``core.normalize``), and the three scope decisions taken on the issue:

1. unlabeled stores (plain zarr / HDF5) are out of scope -- normalization must be
   provably the identity for them, and must not invent semantic labels;
2. the guarantee is unconditional on the read path, including for the geometry a
   read plan hands a client and for what lands in the chunk cache;
3. writable sources are validated at ``create_source`` rather than permuted.
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
from biopb_tensor_server.core.axes import canonical_permutation
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

    # -- Decision 1: unlabeled stores are out of scope --------------------------

    @pytest.mark.parametrize(
        "labels", [["dim0", "dim1", "dim2"], ["a", "b"], [], ["dim0"]]
    )
    def test_unlabeled_axes_are_identity(self, labels):
        """Plain zarr / HDF5 emit ``dimN``. There is nothing to reorder, and
        relabeling them z/y/x would turn build_axis_map's positional *guess*
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
            assert list(plan.descriptor.shape) == [2, 3, 2]

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
