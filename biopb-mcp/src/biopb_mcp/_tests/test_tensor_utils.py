"""Tests for _tensor_utils shared utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import dask.array as da
import pytest

from biopb_mcp._config import get_default_config, get_setting
from biopb_mcp._tensor_utils import (
    _advertised_pyramid_levels,
    _origin_initial_view,
    _resolve_axes,
    add_tensor_layer,
    build_layer_scale,
    build_pyramid_levels,
    canonical_dim_labels,
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
    singleton-Z expand at the end is a harmless no-op on mocks."""

    def _get_tensor(array_id, scale_hint=None):
        hint = scale_hint or [1] * len(shape)
        return _arr_with_shape(
            [max(1, s // h) for s, h in zip(shape, hint, strict=False)]
        )

    return _get_tensor


def _dask_scaling_side_effect(shape):
    """Like ``_scaling_side_effect`` but returns real dask arrays, so the
    singleton-Z insert produces real output shapes and a real ``.ndim`` (needed
    wherever the test inspects the returned levels)."""

    def _get_tensor(array_id, scale_hint=None):
        hint = scale_hint or [1] * len(shape)
        new = [max(1, s // h) for s, h in zip(shape, hint, strict=False)]
        return da.zeros(new, chunks=new)

    return _get_tensor


class TestResolveAxes:
    """``(y, x, z, s)`` read off the canonical [..., Z, Y, X, S] wire order the
    data plane guarantees (biopb/biopb#596). X and Y are positions; the labels
    answer only the two presence questions position cannot."""

    def test_zyx(self):
        assert _resolve_axes([10, 512, 512], ["z", "y", "x"]) == (1, 2, 0, None)

    def test_case_insensitive_labels(self):
        assert _resolve_axes([512, 512], ["Y", "X"]) == (0, 1, None, None)

    def test_2d_has_no_z(self):
        assert _resolve_axes([100, 200], ["y", "x"]) == (0, 1, None, None)

    def test_a_leading_axis_that_is_not_z(self):
        # [C, Y, X] is 3-D but not volumetric -- the depth slot is empty.
        assert _resolve_axes([3, 512, 512], ["c", "y", "x"]) == (1, 2, None, None)

    def test_z_synonym(self):
        assert _resolve_axes([10, 512, 512], ["plane", "y", "x"]) == (1, 2, 0, None)

    def test_only_the_slot_ahead_of_y_can_hold_z(self):
        # [T, Z, Y, X]: canonical, so z is exactly where the order says.
        assert _resolve_axes([5, 10, 64, 64], ["t", "z", "y", "x"]) == (2, 3, 1, None)
        # [Z, C, Y, X] is an order the server does not serve -- it normalizes to
        # [C, Z, Y, X]. No search is made for the buried z.
        assert _resolve_axes([10, 3, 64, 64], ["z", "c", "y", "x"]) == (
            2,
            3,
            None,
            None,
        )

    def test_no_labels_is_positional(self):
        # Nothing to read: fall back to the positional [..., Z, Y, X] reading
        # the server's own plane_axes uses.
        assert _resolve_axes([20, 512, 512], None) == (1, 2, 0, None)
        assert _resolve_axes([512, 512], None) == (0, 1, None, None)

    def test_dimn_labels_leave_the_depth_slot_empty(self):
        # A plain zarr's dimN labels place nothing, but they are labels, and
        # none of them says depth -- so no axis is claimed as z.
        assert _resolve_axes([20, 512, 512], ["dim0", "dim1", "dim2"]) == (
            1,
            2,
            None,
            None,
        )

    def test_label_count_mismatch_falls_back_to_positional(self):
        assert _resolve_axes([3, 64, 32], ["y", "x"]) == (1, 2, 0, None)

    def test_under_2d_raises(self):
        # A 1-D tensor is not a displayable image; fail loud rather than
        # return a bogus (0, 0).
        with pytest.raises(ValueError):
            _resolve_axes([100], ["x"])

    def test_detects_interleaved_rgb(self):
        assert _resolve_axes([1, 1, 1, 8, 8, 3], ["T", "C", "Z", "Y", "X", "S"]) == (
            3,
            4,
            2,
            5,
        )

    def test_detects_rgba(self):
        assert _resolve_axes([1, 1, 1, 8, 8, 4], ["T", "C", "Z", "Y", "X", "S"]) == (
            3,
            4,
            2,
            5,
        )

    def test_samples_synonym(self):
        assert _resolve_axes([2, 8, 8, 3], ["z", "y", "x", "samples"]) == (1, 2, 0, 3)

    def test_ignores_a_samples_axis_of_the_wrong_size(self):
        # An "S" of 5 is not colour, so it stays an ordinary trailing axis --
        # the same labels the server refuses to reorder for the same reason.
        assert _resolve_axes([1, 1, 1, 8, 8, 5], ["T", "C", "Z", "Y", "X", "S"]) == (
            4,
            5,
            None,
            None,
        )

    def test_samples_needs_room_for_y_and_x(self):
        # Rank 2 leaves nothing behind a samples axis, so it is not one.
        assert _resolve_axes([8, 3], ["y", "s"]) == (0, 1, None, None)

    def test_trailing_size_three_is_not_colour_without_the_label(self):
        # The whole point of label-gating: a 3-channel stack must not be
        # rendered as false colour.
        assert _resolve_axes([3, 8, 8], ["c", "y", "x"])[3] is None
        assert _resolve_axes([8, 8, 3], None)[3] is None


class TestBuildPyramidLevels:
    def test_small_image_returns_single_level(self):
        desc = _make_tensor_desc([256, 256])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((256, 256))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)

        assert len(levels) == 1
        # The layer is the source array -- no rank-evening.
        assert levels[0].shape == (256, 256)
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

        client.get_descriptor.assert_called_once_with("src/A2", with_pyramid=True)
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
        client.get_descriptor.assert_called_once_with("src/A2", with_pyramid=True)


class TestBuildPyramidCanonicalOrder:
    """The levels arrive in napari display order already -- the data plane's
    guarantee -- so no axis work happens here at all: what the server serves is
    what the layer gets, at the source's own rank."""

    def test_2d_stays_2d(self):
        # No singleton Z is inserted: the layer is the source array.
        desc = _make_tensor_desc([64, 64], ["y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((64, 64))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (64, 64)

    def test_a_leading_channel_is_untouched(self):
        # [C, Y, X] stays [C, Y, X] -- no Z conjured between C and Y.
        desc = _make_tensor_desc([3, 64, 32], ["c", "y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((3, 64, 32))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (3, 64, 32)

    def test_a_real_z_is_left_alone(self):
        # [C, Z, Y, X] passes through untouched -- no singleton, no transpose.
        desc = _make_tensor_desc([3, 10, 64, 64], ["c", "z", "y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((3, 10, 64, 64))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (3, 10, 64, 64)

    def test_keeps_samples_axis_trailing(self):
        # The biopb/biopb#596 case: [T, C, Z, Y, X, S] renders as colour, not as
        # a 3-plane stack behind a slider.
        desc = _make_tensor_desc([1, 1, 1, 512, 512, 3], ["T", "C", "Z", "Y", "X", "S"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((1, 1, 1, 512, 512, 3))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (1, 1, 1, 512, 512, 3)

    def test_rgb_without_a_z_keeps_the_source_rank(self):
        # [Y, X, S] stays rank 3 -- previously padded to [Z(=1), Y, X, S].
        desc = _make_tensor_desc([64, 32, 3], ["y", "x", "s"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((64, 32, 3))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (64, 32, 3)

    def test_size_three_channel_axis_is_not_treated_as_colour(self):
        # [C, Y, X] with C=3 is a channel stack, not RGB. Shape alone cannot
        # show that now, so assert the samples axis was not detected.
        desc = _make_tensor_desc([3, 64, 32], ["c", "y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((3, 64, 32))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (3, 64, 32)
        assert _resolve_axes([3, 64, 32], ["c", "y", "x"])[3] is None

    def test_a_non_canonical_source_is_served_as_ordered(self):
        # [Y, X, C] is an order the data plane no longer serves -- it normalizes
        # it to [C, Y, X]. Should one arrive anyway (a server predating the
        # guarantee), the client trusts the wire rather than re-deriving an
        # order behind it: C reads as X, and nothing is moved to "fix" it.
        # Deliberate -- the fix belongs upstream.
        desc = _make_tensor_desc([64, 32, 3], ["y", "x", "c"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((64, 32, 3))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        assert levels[0].shape == (64, 32, 3)
        assert _resolve_axes([64, 32, 3], ["y", "x", "c"])[:2] == (1, 2)

    def test_uses_source_desc_labels_when_tensor_unlabeled(self):
        # Per-tensor labels missing -> fall back to the source descriptor's.
        # The rank no longer varies, so the observable difference is whether the
        # trailing 3 is recognized as colour: unlabeled it is just an axis.
        viewer = MagicMock()
        source_desc = MagicMock(dim_labels=["y", "x", "s"])
        client = _make_physical_client(None)
        client.get_tensor.return_value = da.zeros((64, 32, 3))

        add_tensor_layer(
            viewer,
            client,
            "src",
            "t1",
            _make_tensor_desc([64, 32, 3], None),
            name="lyr",
            source_desc=source_desc,
            config=_CFG,
        )
        assert viewer.add_image.call_args[1]["rgb"] is True


class TestCanonicalDimLabels:
    """The source's labels, lowercased -- they name the layer's axes one for one
    now that the layer array is the source array (biopb/biopb#651)."""

    def test_tczyx_passes_through_unswapped(self):
        # The bug case: the leading pair is (T, C), not the (C, T) the writer's
        # positional fallback guesses.
        desc = _make_tensor_desc([2, 3, 4, 64, 64], ["T", "C", "Z", "Y", "X"])
        assert canonical_dim_labels(desc) == ["t", "c", "z", "y", "x"]

    def test_no_z_is_invented_for_a_source_without_one(self):
        # Previously ["c", "z", "y", "x"] -- naming an axis the array gained.
        desc = _make_tensor_desc([3, 64, 32], ["c", "y", "x"])
        assert canonical_dim_labels(desc) == ["c", "y", "x"]

    def test_2d_keeps_two_labels(self):
        desc = _make_tensor_desc([64, 64], ["y", "x"])
        assert canonical_dim_labels(desc) == ["y", "x"]

    def test_samples_axis_is_named_and_stays_last(self):
        desc = _make_tensor_desc([64, 32, 3], ["y", "x", "s"])
        assert canonical_dim_labels(desc) == ["y", "x", "s"]

    def test_matches_the_level_rank_and_order(self):
        # Lockstep with the arrays: same length, same axis in every slot.
        desc = _make_tensor_desc([2, 64, 32, 3], ["t", "y", "x", "s"])
        client = MagicMock()
        client.get_tensor.return_value = da.zeros((2, 64, 32, 3))

        levels = build_pyramid_levels(client, "src", "t1", desc, config=_CFG)
        labels = canonical_dim_labels(desc)
        assert labels == ["t", "y", "x", "s"]
        assert len(labels) == levels[0].ndim

    def test_none_without_labels(self):
        # Nothing to name the leading axes with -- the caller keeps its fallback.
        assert canonical_dim_labels(_make_tensor_desc([3, 64, 32])) is None

    def test_none_on_a_length_mismatch(self):
        assert canonical_dim_labels(_make_tensor_desc([3, 64, 32], ["y", "x"])) is None

    def test_falls_back_to_the_source_descriptor(self):
        desc = _make_tensor_desc([3, 64, 32], None)
        source_desc = MagicMock(dim_labels=["c", "y", "x"])
        assert canonical_dim_labels(desc, source_desc=source_desc) == ["c", "y", "x"]


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
    # Source order [t, c, z, y, x] -> each physical size lands on the axis it
    # describes; leading axes (t, c) get 1.0. ndim is the source rank.
    client = _make_physical_client(
        [0.0, 0.0, 2.0, 0.325, 0.325],
        ["", "", "µm", "µm", "µm"],
    )
    desc = _make_tensor_desc([1, 3, 10, 64, 64], ["t", "c", "z", "y", "x"])
    scale, info = build_layer_scale(
        client, "src", ndim=5, tensor_id="t1", tensor_desc=desc
    )
    assert scale == [1.0, 1.0, 2.0, 0.325, 0.325]
    assert info["physical_size_x"] == 0.325
    assert info["physical_size_x_unit"] == "µm"


def test_build_layer_scale_2d_stays_2d():
    # A 2-D source stays 2-D, so the scale is a 2-vector. Under the old
    # rank-evening this was [1.0, 0.25, 0.5]; writing scale[-3] on a 2-element
    # list would now raise, which is why placement is by axis index.
    client = _make_physical_client([0.25, 0.5], ["µm", "µm"])
    desc = _make_tensor_desc([512, 512], ["y", "x"])
    scale, _ = build_layer_scale(
        client, "src", ndim=2, tensor_id="t1", tensor_desc=desc
    )
    assert scale == [0.25, 0.5]


def test_build_layer_scale_skips_the_leading_axes():
    # Source order [c, y, x]: the summary is in the same canonical order, so
    # x/y come off the tail and the leading channel axis gets 1.0.
    client = _make_physical_client([0.0, 0.25, 0.5], ["", "µm", "µm"])
    desc = _make_tensor_desc([3, 64, 32], ["c", "y", "x"])
    scale, info = build_layer_scale(
        client, "src", ndim=3, tensor_id="t1", tensor_desc=desc
    )
    # Layer is [C, Y, X] -- no z slot, and C is not mistaken for one.
    assert scale == [1.0, 0.25, 0.5]
    assert info["physical_size_y"] == 0.25
    assert info["physical_size_x"] == 0.5


def test_build_layer_scale_rgb_drops_the_samples_axis():
    # Canonical rgb output is [T, C, Z, Y, X, S] (ndim 6), but napari does not
    # count S as a layer dimension -- layer.ndim is 5 and len(scale) must match.
    client = _make_physical_client(
        [0.0, 0.0, 2.0, 0.325, 0.325, 0.0],
        ["", "", "µm", "µm", "µm", ""],
    )
    desc = _make_tensor_desc([1, 1, 10, 64, 64, 3], ["T", "C", "Z", "Y", "X", "S"])
    scale, info = build_layer_scale(
        client, "src", ndim=6, tensor_id="t1", tensor_desc=desc, rgb=True
    )
    assert scale == [1.0, 1.0, 2.0, 0.325, 0.325]
    assert len(scale) == 5  # napari does not count S
    assert info["physical_size_x"] == 0.325


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
        # A 2-D source stays 2-D: scale is [y, x].
        assert kwargs["scale"] == [0.25, 0.5]
        phys = kwargs["metadata"]["ome_physical_size"]
        assert phys["physical_size_x"] == 0.5
        # The whole point of #31: no full-OME fetch on the hot path.
        client.get_source_metadata.assert_not_called()

    def test_single_level_omits_multiscale_and_scale(self):
        viewer = MagicMock()
        # No physical sizes -> no scale kwarg, and metadata carries the axis
        # names alone.
        client = _make_physical_client(None)
        client.get_tensor.return_value = da.zeros((256, 256))
        desc = _make_tensor_desc([256, 256], ["y", "x"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        _, kwargs = viewer.add_image.call_args
        assert kwargs == {
            "name": "lyr",
            "metadata": {"array_id": "t1", "dim_labels": ["y", "x"]},
        }

    def test_attaches_canonical_dim_labels(self):
        # The layer is the only place the OME-Zarr writer can learn its axis
        # names from (biopb/biopb#651). The layer is the source array now, so
        # they are the source's labels -- no invented axis to name.
        viewer = MagicMock()
        client = _make_physical_client([0.0, 0.25, 0.5], ["", "µm", "µm"])
        client.get_tensor.return_value = da.zeros((3, 64, 32))
        desc = _make_tensor_desc([3, 64, 32], ["C", "Y", "X"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        arr = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        assert kwargs["metadata"]["dim_labels"] == ["c", "y", "x"]
        assert len(kwargs["metadata"]["dim_labels"]) == arr.ndim
        # The physical size stays alongside, not replaced by the labels.
        assert "ome_physical_size" in kwargs["metadata"]

    def test_no_dim_labels_when_the_source_declares_none(self):
        # Unlabeled source -> nothing to name the leading axes with; the writer
        # keeps its positional fallback rather than being handed a guess.
        viewer = MagicMock()
        client = _make_physical_client(None)
        client.get_tensor.return_value = da.zeros((256, 256))
        desc = _make_tensor_desc([256, 256])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        _, kwargs = viewer.add_image.call_args
        assert kwargs == {"name": "lyr", "metadata": {"array_id": "t1"}}

    def test_records_the_originating_array_id(self):
        # The layer's only record of where it came from: the name is a display
        # stem the user can rename, so provenance has to live in metadata.
        viewer = MagicMock()
        client = _make_physical_client(None)
        client.get_tensor.return_value = da.zeros((256, 256))
        desc = _make_tensor_desc([256, 256], ["y", "x"])

        add_tensor_layer(
            viewer, client, "multi", "multi/Image:0", desc, name="lyr", config=_CFG
        )

        _, kwargs = viewer.add_image.call_args
        # The qualified array_id client.get_tensor takes, not the routing prefix.
        assert kwargs["metadata"]["array_id"] == "multi/Image:0"

    def test_scale_lands_on_the_axis_it_describes(self):
        viewer = MagicMock()
        client = _make_physical_client([0.0, 0.25, 0.5], ["", "µm", "µm"])
        # [C, Y, X] with no z: the channel axis must get 1.0, not z's spacing.
        client.get_tensor.return_value = da.zeros((3, 64, 32))
        desc = _make_tensor_desc([3, 64, 32], ["c", "y", "x"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        arr = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        assert arr.shape == (3, 64, 32)
        assert kwargs["scale"] == [1.0, 0.25, 0.5]

    def test_interleaved_rgb_gets_rgb_kwarg_and_shorter_scale(self):
        viewer = MagicMock()
        client = _make_physical_client(
            [0.0, 0.0, 0.0, 0.25, 0.5, 0.0], ["", "", "", "µm", "µm", ""]
        )
        client.get_tensor.return_value = da.zeros((1, 1, 1, 64, 32, 3))
        desc = _make_tensor_desc([1, 1, 1, 64, 32, 3], ["T", "C", "Z", "Y", "X", "S"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        arr = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        assert arr.shape == (1, 1, 1, 64, 32, 3)
        assert kwargs["rgb"] is True
        # napari's layer.ndim is data.ndim - 1 for rgb, and scale must match it.
        assert len(kwargs["scale"]) == arr.ndim - 1
        assert kwargs["scale"] == [1.0, 1.0, 1.0, 0.25, 0.5]

    def test_non_rgb_leaves_rgb_unset(self):
        # napari's own auto-detection must keep applying to unlabelled data.
        viewer = MagicMock()
        client = _make_physical_client(None)
        client.get_tensor.return_value = da.zeros((256, 256))
        desc = _make_tensor_desc([256, 256], ["y", "x"])

        add_tensor_layer(viewer, client, "src", "t1", desc, name="lyr", config=_CFG)

        _, kwargs = viewer.add_image.call_args
        assert "rgb" not in kwargs


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
