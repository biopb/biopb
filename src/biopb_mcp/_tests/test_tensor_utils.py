"""Tests for _tensor_utils shared utilities."""

from unittest.mock import MagicMock, call

import dask.array as da
import pytest

from biopb_mcp._config import get_default_config, get_setting
from biopb_mcp._tensor_utils import (
    add_tensor_layer,
    build_layer_scale,
    build_pyramid_levels,
    get_xy_dim_indices,
    get_z_dim_index,
)

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

    def _get_tensor(source_id, tensor_id, scale_hint=None):
        hint = scale_hint or [1] * len(shape)
        return _arr_with_shape([max(1, s // h) for s, h in zip(shape, hint)])

    return _get_tensor


def _dask_scaling_side_effect(shape):
    """Like ``_scaling_side_effect`` but returns real dask arrays, so the
    canonicalizing transpose/singleton-insert produce real output shapes and a
    real ``.ndim`` (needed wherever the test inspects the returned levels)."""

    def _get_tensor(source_id, tensor_id, scale_hint=None):
        hint = scale_hint or [1] * len(shape)
        new = [max(1, s // h) for s, h in zip(shape, hint)]
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
        client.get_tensor.assert_called_once_with(
            "src", "t1", scale_hint=[1, 1]
        )

    def test_threshold_boundary_no_pyramid(self):
        desc = _make_tensor_desc([THRESHOLD, THRESHOLD])
        client = MagicMock()
        client.get_tensor.return_value = _arr_with_shape(
            [THRESHOLD, THRESHOLD]
        )

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
        assert first_call == call("src", "t1", scale_hint=[1, 1])

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
        assert all(
            c[1]["scale_hint"][0] == 1
            for c in client.get_tensor.call_args_list
        )

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
        scales = [
            c[1]["scale_hint"][1] for c in client.get_tensor.call_args_list
        ]
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
        desc = _make_tensor_desc(
            [3000, 8192, 8192], dim_labels=["z", "y", "x"]
        )
        client = MagicMock()
        client.get_tensor.side_effect = _scaling_side_effect(
            [3000, 8192, 8192]
        )

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


def _make_metadata_client(
    psx=None, psy=None, psz=None, unit="µm", raises=False
):
    """Mock TensorFlightClient whose get_source_metadata returns the OME
    metadata as a dict (the server's OME model dumped to JSON), matching the
    real client contract."""
    client = MagicMock()
    if raises:
        client.get_source_metadata.side_effect = RuntimeError("boom")
        return client
    pixels = {
        "physical_size_x": psx,
        "physical_size_y": psy,
        "physical_size_z": psz,
        "physical_size_x_unit": unit,
        "physical_size_y_unit": unit,
        "physical_size_z_unit": unit,
    }
    metadata = {"images": [{"pixels": pixels}]}
    client.get_source_metadata.return_value = metadata
    return client


def test_build_layer_scale_maps_canonical_trailing_axes():
    # Canonical [..., Z, Y, X]: psz/psy/psx land on the last three axes,
    # leading axes (channel) get 1.0.
    client = _make_metadata_client(psx=0.325, psy=0.325, psz=2.0)
    scale, info = build_layer_scale(client, "src", ndim=4)
    assert scale == [1.0, 2.0, 0.325, 0.325]
    assert info["physical_size_x"] == 0.325
    assert info["physical_size_x_unit"] == "µm"


def test_build_layer_scale_2d_canonical_has_singleton_z():
    # A 2D source is canonicalized to [Z(=1), Y, X], so ndim is 3 and the
    # singleton z gets 1.0 (no physical_size_z).
    client = _make_metadata_client(psx=0.5, psy=0.25)
    scale, _ = build_layer_scale(client, "src", ndim=3)
    assert scale == [1.0, 0.25, 0.5]


def test_build_layer_scale_none_when_no_physical_sizes():
    client = _make_metadata_client()
    assert build_layer_scale(client, "src", ndim=3) == (None, None)


def test_build_layer_scale_none_on_metadata_error():
    client = _make_metadata_client(raises=True)
    assert build_layer_scale(client, "src", ndim=3) == (None, None)


class TestAddTensorLayer:
    """The shared build-pyramid -> wrap -> OME scale -> add_image pipeline
    used by both the Tensor Browser widget and the MCP add_tensor."""

    def test_multiscale_with_ome_scale_and_metadata(self):
        viewer = MagicMock()
        client = _make_metadata_client(psx=0.5, psy=0.25)
        client.get_tensor.side_effect = _dask_scaling_side_effect([8192, 8192])
        desc = _make_tensor_desc([8192, 8192], ["y", "x"])

        add_tensor_layer(
            viewer, client, "src", "t1", desc, name="lyr", config=_CFG
        )

        levels_arg = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        assert isinstance(levels_arg, list) and len(levels_arg) > 1
        assert kwargs["name"] == "lyr"
        assert kwargs["multiscale"] is True
        # Canonical [Z(=1), Y, X]: scale is [z, y, x] = [1.0, 0.25, 0.5].
        assert kwargs["scale"] == [1.0, 0.25, 0.5]
        phys = kwargs["metadata"]["ome_physical_size"]
        assert phys["physical_size_x"] == 0.5

    def test_single_level_omits_multiscale_and_scale(self):
        viewer = MagicMock()
        # No physical sizes -> no scale/metadata kwargs.
        client = _make_metadata_client()
        client.get_tensor.return_value = da.zeros((256, 256))
        desc = _make_tensor_desc([256, 256], ["y", "x"])

        add_tensor_layer(
            viewer, client, "src", "t1", desc, name="lyr", config=_CFG
        )

        _, kwargs = viewer.add_image.call_args
        assert kwargs == {"name": "lyr"}

    def test_misordered_axes_canonicalized_with_aligned_scale(self):
        viewer = MagicMock()
        client = _make_metadata_client(psx=0.5, psy=0.25)
        # [Y, X, C] layout: without canonicalization napari would display the
        # wrong plane. A real dask array so the transpose actually reorders.
        client.get_tensor.return_value = da.zeros((64, 32, 3))
        desc = _make_tensor_desc([64, 32, 3], ["y", "x", "c"])

        add_tensor_layer(
            viewer, client, "src", "t1", desc, name="lyr", config=_CFG
        )

        arr = viewer.add_image.call_args[0][0]
        _, kwargs = viewer.add_image.call_args
        # Reordered to [C, Z(=1), Y, X].
        assert arr.shape == (3, 1, 64, 32)
        # Scale aligns to the canonical output: [c, z, y, x].
        assert kwargs["scale"] == [1.0, 1.0, 0.25, 0.5]
