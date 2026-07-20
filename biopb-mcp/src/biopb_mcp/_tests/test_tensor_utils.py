"""Tests for _tensor_utils shared utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import dask.array as da
import pytest

from biopb_mcp._config import get_default_config, get_setting
from biopb_mcp._tensor_utils import (
    _advertised_pyramid_levels,
    _origin_initial_view,
    add_tensor_layer,
    build_layer_scale,
    build_pyramid_levels,
    get_xy_dim_indices,
    get_z_dim_index,
)


def _adv_level(scale_hint, reduction_method):
    return SimpleNamespace(
        scale_hint=list(scale_hint), reduction_method=reduction_method
    )


def _recording_get_tensor(full_shape):
    """get_tensor stub that records (scale_hint, reduction_method) and returns a
    real dask array shaped by the scale, so canonicalization runs for real."""
    calls = []

    def _gt(array_id, scale_hint=None, reduction_method=None):
        hint = scale_hint or [1] * len(full_shape)
        calls.append((tuple(hint), reduction_method))
        new = [max(1, s // h) for s, h in zip(full_shape, hint, strict=False)]
        return da.zeros(new, chunks=new)

    return _gt, calls


# Pyramid params now live in the ``pyramid`` config section. Resolve the
# defaults once and pass the config explicitly to build_pyramid_levels so the
# tests don't depend on any on-disk config override.
_CFG = get_default_config()
THRESHOLD = get_setting(_CFG, "pyramid.threshold")
FACTOR = get_setting(_CFG, "pyramid.downscale_factor")


def _make_tensor_desc(shape, dim_labels=None):
    desc = MagicMock()
    desc.shape = shape
    desc.dim_labels = dim_labels or []
    return desc


def _arr_with_shape(shape):
    arr = MagicMock()
    arr.shape = list(shape)
    return arr


def _scaling_side_effect(shape):
    """get_tensor side effect returning a mock whose ``.shape`` is *shape*
    downsampled per ``scale_hint`` (floor division).

    build_pyramid_levels reads the *returned* array's real extents (the server's
    downsample rounding isn't part of the API), so multi-level pyramid tests
    must hand back arrays whose shape actually shrinks with the scale hint. Use
    this for tests that only inspect the ``scale_hint`` call args -- the
    canonicalizing transpose/expand at the end is a harmless no-op on mocks."""

    def _get_tensor(array_id, scale_hint=None):
        hint = scale_hint or [1] * len(shape)
        return _arr_with_shape(
            [max(1, s // h) for s, h in zip(shape, hint, strict=False)]
        )

    return _get_tensor


def _dask_scaling_side_effect(shape):
    """Like ``_scaling_side_effect`` but returns real dask arrays, so the
    canonicalizing transpose/singleton-insert produce real output shapes and a
    real ``.ndim`` (needed wherever the test inspects the returned levels)."""

    def _get_tensor(array_id, scale_hint=None):
        hint = scale_hint or [1] * len(shape)
        new = [max(1, s // h) for s, h in zip(shape, hint, strict=False)]
        return da.zeros(new, chunks=new)

    return _get_tensor


class TestGetXyDimIndices:
    def test_uses_dim_labels(self):
        assert get_xy_dim_indices([10, 512, 512], ["z", "y", "x"]) == (1, 2)

    def test_case_insensitive_labels(self):
        assert get_xy_dim_indices([512, 512], ["Y", "X"]) == (0, 1)

    def test_fallback_last_two_dims(self):
        # Standard [..., Y, X]: Y is second-to-last, X is last.
        assert get_xy_dim_indices([3, 100, 200]) == (1, 2)

    def test_2d_no_labels(self):
        # [Y, X]: Y first, X last.
        assert get_xy_dim_indices([100, 200]) == (0, 1)

    def test_labels_without_xy_falls_back(self):
        # Labels present but no x/y -> positional [..., Y, X] fallback.
        assert get_xy_dim_indices([10, 512, 512], ["z", "r", "c"]) == (1, 2)

    def test_under_2d_raises(self):
        # A 1-D tensor is not a displayable image; fail loud rather than
        # return a bogus (0, 0).
        with pytest.raises(ValueError):
            get_xy_dim_indices([100])


class TestGetZDimIndex:
    def test_label_z_wins_over_position(self):
        assert get_z_dim_index([10, 5, 512, 512], ["z", "c", "y", "x"]) == 0

    def test_labels_without_z_is_none(self):
        # Labels present but no z -> not volumetric (e.g. [C, Y, X]).
        assert get_z_dim_index([3, 512, 512], ["c", "y", "x"]) is None

    def test_no_labels_3d_positional(self):
        # No labels -> assume [..., Z, Y, X]; z is third-from-last.
        assert get_z_dim_index([20, 512, 512]) == 0

    def test_no_labels_2d_is_none(self):
        assert get_z_dim_index([512, 512]) is None


class TestBuildPyramidLevels:
    def test_small_image_returns_single_level(self):
        desc = _make_tensor_desc([256, 256])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((256, 256))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        assert len(levels) == 1
        # Canonical output always carries an explicit (here singleton) Z.
        assert levels[0].shape == (1, 256, 256)
        # Unified loop always passes a scale_hint, even for a single level.
        client.get_tensor.assert_called_once_with("t1", scale_hint=[1, 1])

    def test_threshold_boundary_no_pyramid(self):
        desc = _make_tensor_desc([THRESHOLD, THRESHOLD])
        client = MagicMock()
        client.get_tensor.return_value = _arr_with_shape([THRESHOLD, THRESHOLD])

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert len(levels) == 1

    def test_large_image_builds_pyramid(self):
        desc = _make_tensor_desc([8192, 8192])
        client = MagicMock()
        client.get_tensor.side_effect = _scaling_side_effect([8192, 8192])

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        assert len(levels) > 1
        # First call should be scale=1 (no scale_hint with all 1s)
        first_call = client.get_tensor.call_args_list[0]
        assert first_call == call("t1", scale_hint=[1, 1])

    def test_small_z_is_not_downsampled(self):
        # A thin z (10 < floor) stays full-res while x/y shrink.
        desc = _make_tensor_desc([10, 8192, 8192], dim_labels=["z", "y", "x"])
        client = MagicMock()
        client.get_tensor.side_effect = _scaling_side_effect([10, 8192, 8192])

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        assert len(levels) > 1
        # First level: scale_hint = [1, 1, 1]
        first_hint = client.get_tensor.call_args_list[0][1]["scale_hint"]
        assert first_hint == [1, 1, 1]
        # Second level: z stays 1 (too small to scale), y and x scale.
        second_hint = client.get_tensor.call_args_list[1][1]["scale_hint"]
        assert second_hint[0] == 1  # z
        assert second_hint[1] == FACTOR  # y
        assert second_hint[2] == FACTOR  # x
        # z stays full-res at every level.
        assert all(c[1]["scale_hint"][0] == 1 for c in client.get_tensor.call_args_list)

    def test_pyramid_coarsest_level_fits_within_threshold(self):
        # Levels are emitted until the coarsest fits within `threshold`.
        # Because the previous level still exceeded it, the coarsest always
        # lands in (threshold // factor, threshold].
        size = 100000
        desc = _make_tensor_desc([size, size])
        client = MagicMock()
        client.get_tensor.side_effect = _scaling_side_effect([size, size])

        build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        # x is the last dim for a 2D source; scale is symmetric in x/y.
        scales = [c[1]["scale_hint"][1] for c in client.get_tensor.call_args_list]
        coarsest = size // scales[-1]
        assert coarsest <= THRESHOLD
        assert coarsest > THRESHOLD // FACTOR
        # Every level before the coarsest still exceeded the threshold --
        # that's why another level was emitted.
        for s in scales[:-1]:
            assert size // s > THRESHOLD

    def test_deep_stack_bounds_coarsest_volume_including_z(self):
        # A deep stack must downsample z too, so the coarsest level's whole
        # volume (Lz*Ly*Lx) fits the voxel budget (issue #29).
        desc = _make_tensor_desc([3000, 8192, 8192], dim_labels=["z", "y", "x"])
        client = MagicMock()
        client.get_tensor.side_effect = _scaling_side_effect([3000, 8192, 8192])

        build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        budget = get_setting(_CFG, "pyramid.pixel_budget_cubic_root") ** 3
        last = client.get_tensor.call_args_list[-1][1]["scale_hint"]
        sz, sy, sx = last[0], last[1], last[2]
        lz, ly, lx = 3000 // sz, 8192 // sy, 8192 // sx
        assert lx * ly * lz <= budget
        assert sz > 1  # z was genuinely downsampled, not left at full res

    def test_no_z_axis_emits_2d_scale_hints(self):
        # No z label -> Lz treated as 1; the pyramid never adds a z factor.
        desc = _make_tensor_desc([8192, 8192], dim_labels=["y", "x"])
        client = MagicMock()
        client.get_tensor.side_effect = _scaling_side_effect([8192, 8192])

        build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        for c in client.get_tensor.call_args_list:
            assert len(c[1]["scale_hint"]) == 2


class TestAdvertisedPyramid:
    """When the server advertises a pyramid, build_pyramid_levels requests each
    level by the advertised scale_hint AND reduction_method (so the client's
    chunk_ids match what the server serves and pre-warms), and skips the
    client-side scale loop entirely."""

    _FULL = [1, 4, 1, 800, 800]
    _LABELS = ["t", "c", "z", "y", "x"]

    def test_uses_descriptor_pyramid_with_reduction(self):
        gt, calls = _recording_get_tensor(self._FULL)
        client = MagicMock()
        client.get_tensor.side_effect = gt
        desc = SimpleNamespace(
            shape=self._FULL,
            dim_labels=self._LABELS,
            pyramid=[
                _adv_level([1, 1, 1, 1, 1], "area"),
                _adv_level([1, 1, 1, 4, 4], "area"),
            ],
        )

        levels = build_pyramid_levels(client, "src", "src/A2", desc, config=_CFG)

        assert len(levels) == 2
        # Both requests carry the advertised scale_hint AND reduction_method.
        assert calls == [
            ((1, 1, 1, 1, 1), "area"),
            ((1, 1, 1, 4, 4), "area"),
        ]
        # Descriptor already had a pyramid -> no extra open-time fetch.
        client.get_descriptor.assert_not_called()

    def test_fetches_open_time_descriptor_when_catalog_is_lean(self):
        gt, calls = _recording_get_tensor(self._FULL)
        client = MagicMock()
        client.get_tensor.side_effect = gt
        # The lean list_sources descriptor carries no pyramid; the open-time
        # descriptor (get_descriptor) does.
        client.get_descriptor.return_value = SimpleNamespace(
            array_id="src/A2",
            pyramid=[
                _adv_level([1, 1, 1, 1, 1], "area"),
                _adv_level([1, 1, 1, 4, 4], "area"),
            ],
        )
        lean = SimpleNamespace(shape=self._FULL, dim_labels=self._LABELS)

        levels = build_pyramid_levels(client, "src", "src/A2", lean, config=_CFG)

        client.get_descriptor.assert_called_once_with("src/A2")
        assert len(levels) == 2
        assert [c[1] for c in calls] == ["area", "area"]

    def test_empty_reduction_passes_none(self):
        # An advertised level with reduction_method "" must forward None (let
        # the server pick), not the empty string.
        gt, calls = _recording_get_tensor(self._FULL)
        client = MagicMock()
        client.get_tensor.side_effect = gt
        desc = SimpleNamespace(
            shape=self._FULL,
            dim_labels=self._LABELS,
            pyramid=[_adv_level([1, 1, 1, 1, 1], "")],
        )

        build_pyramid_levels(client, "src", "src/A2", desc, config=_CFG)

        assert calls[0][1] is None


class TestAdvertisedPyramidLevelsHelper:
    def test_prefers_descriptor_pyramid(self):
        desc = SimpleNamespace(pyramid=[_adv_level([1, 1], "area")])
        client = MagicMock()
        out = _advertised_pyramid_levels(client, "src", "src/A2", desc)
        assert len(out) == 1
        client.get_descriptor.assert_not_called()

    def test_returns_empty_on_lookup_failure(self):
        client = MagicMock()
        client.get_descriptor.side_effect = RuntimeError("boom")
        desc = SimpleNamespace()  # no pyramid attr
        assert _advertised_pyramid_levels(client, "src", "src/A2", desc) == []

    def test_uses_open_time_descriptor_pyramid(self):
        # The passed descriptor carries no pyramid -> fetch the open-time
        # descriptor by array_id and read its pyramid.
        client = MagicMock()
        client.get_descriptor.return_value = SimpleNamespace(
            array_id="src/A2", pyramid=[_adv_level([1], "x")]
        )
        out = _advertised_pyramid_levels(client, "src", "src/A2", SimpleNamespace())
        assert len(out) == 1
        client.get_descriptor.assert_called_once_with("src/A2")


class TestBuildPyramidCanonicalOrder:
    """build_pyramid_levels emits levels in napari display order
    [..., Z, Y, X], transposing mis-ordered sources and inserting a singleton
    Z when the tensor has none."""

    def test_inserts_singleton_z_for_2d(self):
        desc = _make_tensor_desc([64, 64], ["y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((64, 64))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (1, 64, 64)

    def test_reorders_trailing_channel(self):
        # [Y, X, C] -> [C, Z(=1), Y, X].
        desc = _make_tensor_desc([64, 32, 3], ["y", "x", "c"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((64, 32, 3))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (3, 1, 64, 32)

    def test_reorders_buried_z(self):
        # [Z, C, Y, X] -> [C, Z, Y, X]; real z, so no singleton inserted.
        desc = _make_tensor_desc([10, 3, 64, 64], ["z", "c", "y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((10, 3, 64, 64))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (3, 10, 64, 64)

    def test_uses_source_desc_labels_when_tensor_unlabeled(self):
        # Per-tensor labels missing -> fall back to the source descriptor's
        # labels for both the reorder and the (implicit) scale alignment.
        desc = _make_tensor_desc([64, 32, 3], None)
        source_desc = MagicMock(dim_labels=["y", "x", "c"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((64, 32, 3))

        levels = build_pyramid_levels(
            client, "src", "t1", desc, source_desc=source_desc, config=_CFG
        )
        assert levels[0].shape == (3, 1, 64, 32)


def _make_physical_client(scale_vec=None, unit_vec=None, raises=False):
    """Mock TensorFlightClient whose ``get_physical_scale`` returns the compact
    per-dimension ``(scale, unit)`` summary in *source* axis order (the
    descriptor field the server folds on, biopb issue #31), or ``None`` when no
    physical scale is advertised (old server / format without physical sizes).
    """
    client = MagicMock()
    if raises:
        client.get_physical_scale.side_effect = RuntimeError("boom")
        return client
    if scale_vec is None:
        client.get_physical_scale.return_value = None
    else:
        client.get_physical_scale.return_value = (
            scale_vec,
            unit_vec if unit_vec is not None else ["" for _ in scale_vec],
        )
    return client


def test_build_layer_scale_maps_canonical_trailing_axes():
    # Source order [t, c, z, y, x] -> physical sizes map to x/y/z by label and
    # land on the canonical [..., Z, Y, X] trailing axes; leading axes get 1.0.
    client = _make_physical_client(
        [0.0, 0.0, 2.0, 0.325, 0.325],
        ["", "", "µm", "µm", "µm"],
    )
    desc = _make_tensor_desc([1, 3, 10, 64, 64], ["t", "c", "z", "y", "x"])
    scale, info = build_layer_scale(
        client, "src", ndim=4, tensor_id="t1", tensor_desc=desc
    )
    assert scale == [1.0, 2.0, 0.325, 0.325]
    assert info["physical_size_x"] == 0.325
    assert info["physical_size_x_unit"] == "µm"


def test_build_layer_scale_2d_canonical_has_singleton_z():
    # A 2D source is canonicalized to [Z(=1), Y, X], so ndim is 3 and the
    # singleton z gets 1.0 (no physical_size_z).
    client = _make_physical_client([0.25, 0.5], ["µm", "µm"])
    desc = _make_tensor_desc([512, 512], ["y", "x"])
    scale, _ = build_layer_scale(
        client, "src", ndim=3, tensor_id="t1", tensor_desc=desc
    )
    assert scale == [1.0, 0.25, 0.5]


def test_build_layer_scale_maps_misordered_source_axes_by_label():
    # Source order [y, x, c]: x/y must be resolved by label, not position.
    client = _make_physical_client([0.25, 0.5, 0.0], ["µm", "µm", ""])
    desc = _make_tensor_desc([64, 32, 3], ["y", "x", "c"])
    scale, info = build_layer_scale(
        client, "src", ndim=4, tensor_id="t1", tensor_desc=desc
    )
    # Canonical [C, Z(=1), Y, X].
    assert scale == [1.0, 1.0, 0.25, 0.5]
    assert info["physical_size_y"] == 0.25
    assert info["physical_size_x"] == 0.5


def test_build_layer_scale_none_when_no_physical_sizes():
    # All-zero source scale -> nothing to apply.
    client = _make_physical_client([0.0, 0.0, 0.0], ["", "", ""])
    desc = _make_tensor_desc([10, 64, 64], ["z", "y", "x"])
    assert build_layer_scale(
        client, "src", ndim=3, tensor_id="t1", tensor_desc=desc
    ) == (None, None)


def test_build_layer_scale_none_on_old_server():
    # Old server / no summary advertised -> get_physical_scale returns None and
    # we do NOT fall back to the full-OME get_source_metadata fetch (issue #31).
    client = _make_physical_client(None)
    desc = _make_tensor_desc([10, 64, 64], ["z", "y", "x"])
    assert build_layer_scale(
        client, "src", ndim=3, tensor_id="t1", tensor_desc=desc
    ) == (None, None)
    client.get_source_metadata.assert_not_called()


def test_build_layer_scale_none_on_error():
    client = _make_physical_client(raises=True)
    desc = _make_tensor_desc([10, 64, 64], ["z", "y", "x"])
    assert build_layer_scale(
        client, "src", ndim=3, tensor_id="t1", tensor_desc=desc
    ) == (None, None)


class TestAddTensorLayer:
    """The shared build-pyramid -> wrap -> physical scale -> add_image pipeline
    used by both the Tensor Browser widget and the MCP add_tensor."""

    def test_multiscale_with_physical_scale_and_metadata(self):
        viewer = MagicMock()
        client = _make_physical_client([0.25, 0.5], ["µm", "µm"])
        client.get_tensor.side_effect = _dask_scaling_side_effect([8192, 8192])
        desc = _make_tensor_desc([8192, 8192], ["y", "x"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        levels_arg = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        assert isinstance(levels_arg, list) and len(levels_arg) > 1
        assert kwargs["name"] == "lyr"
        assert kwargs["multiscale"] is True
        # Canonical [Z(=1), Y, X]: scale is [z, y, x] = [1.0, 0.25, 0.5].
        assert kwargs["scale"] == [1.0, 0.25, 0.5]
        phys = kwargs["metadata"]["ome_physical_size"]
        assert phys["physical_size_x"] == 0.5
        # The whole point of #31: no full-OME fetch on the hot path.
        client.get_source_metadata.assert_not_called()

    def test_single_level_omits_multiscale_and_scale(self):
        viewer = MagicMock()
        # No physical sizes -> no scale/metadata kwargs.
        client = _make_physical_client(None)
        client.get_tensor.return_value = da.zeros((256, 256))
        desc = _make_tensor_desc([256, 256], ["y", "x"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        _, kwargs = viewer.add_image.call_args
        assert kwargs == {"name": "lyr"}

    def test_misordered_axes_canonicalized_with_aligned_scale(self):
        viewer = MagicMock()
        client = _make_physical_client([0.25, 0.5, 0.0], ["µm", "µm", ""])
        # [Y, X, C] layout: without canonicalization napari would display the
        # wrong plane. A real dask array so the transpose actually reorders.
        client.get_tensor.return_value = da.zeros((64, 32, 3))
        desc = _make_tensor_desc([64, 32, 3], ["y", "x", "c"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        arr = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        # Reordered to [C, Z(=1), Y, X].
        assert arr.shape == (3, 1, 64, 32)
        # Scale aligns to the canonical output: [c, z, y, x].
        assert kwargs["scale"] == [1.0, 1.0, 0.25, 0.5]


class TestOriginInitialView:
    """The context manager that pins the first layer's view to the origin so a
    multi-channel tensor decodes one coarse plane at load, not two (thumbnail
    @origin + display @center)."""

    def test_suppresses_and_restores_center_step(self):
        class FakeDims:
            def _go_to_center_step(self):
                pass

        orig = FakeDims._go_to_center_step
        viewer = MagicMock()
        viewer.dims = FakeDims()

        with _origin_initial_view(viewer):
            # While active, centering is neutralized to a no-op...
            assert FakeDims._go_to_center_step is not orig
            assert FakeDims._go_to_center_step(viewer.dims) is None
        # ...and restored to the real method afterwards.
        assert FakeDims._go_to_center_step is orig

    def test_restores_on_exception(self):
        class FakeDims:
            def _go_to_center_step(self):
                pass

        orig = FakeDims._go_to_center_step
        viewer = MagicMock()
        viewer.dims = FakeDims()

        with pytest.raises(ValueError):
            with _origin_initial_view(viewer):
                raise ValueError("boom")
        assert FakeDims._go_to_center_step is orig

    def test_noop_for_mock_viewer(self):
        # A MagicMock viewer has no real Dims class method -> the manager must
        # not raise or mutate the global MagicMock class.
        with _origin_initial_view(MagicMock()):
            pass
        assert not hasattr(MagicMock, "_go_to_center_step")


class TestToNativeByteorder:
    """#296: napari's thumbnail convert_to_uint8 rejects a non-native-endian
    array, so add_tensor_layer normalizes levels to native byte order first."""

    def test_swaps_big_endian_and_preserves_values(self):
        import numpy as np

        from biopb_mcp._tensor_utils import _to_native_byteorder

        be = (np.arange(6, dtype=">i2").reshape(2, 3) - 1).astype(">i2")
        lv_be = da.from_array(be, chunks=(2, 3))
        lv_native = da.from_array(np.arange(6, dtype="<i2").reshape(2, 3))

        out = _to_native_byteorder([lv_be, lv_native])

        # Big-endian level -> native order, values identical (lazy swap).
        assert out[0].dtype.isnative
        np.testing.assert_array_equal(out[0].compute(), be)
        # A native level passes through untouched (same object).
        assert out[1] is lv_native

    def test_unblocks_napari_convert_to_uint8(self):
        import numpy as np

        pytest.importorskip("napari")
        from napari.layers.utils.layer_utils import convert_to_uint8

        from biopb_mcp._tensor_utils import _to_native_byteorder

        be = (np.arange(6, dtype=">i2").reshape(2, 3) * 5000).astype(">i2")
        # The underlying napari bug (#296): the raw big-endian array trips the
        # ufunc byte-order TypeError -- the reason this workaround exists.
        with pytest.raises(TypeError):
            convert_to_uint8(be.copy())
        # After normalization the (native) array converts fine.
        (native_lv,) = _to_native_byteorder([da.from_array(be, chunks=be.shape)])
        out = convert_to_uint8(native_lv.compute())
        assert out.dtype == np.uint8
