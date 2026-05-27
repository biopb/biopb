"""Tests for _tensor_utils shared utilities."""

from unittest.mock import MagicMock, call

import pytest

from napari_biopb._tensor_utils import (
    PYRAMID_THRESHOLD,
    build_pyramid_levels,
    get_xy_dim_indices,
)


def _make_tensor_desc(shape, dim_labels=None):
    desc = MagicMock()
    desc.shape = shape
    desc.dim_labels = dim_labels or []
    return desc


class TestGetXyDimIndices:
    def test_uses_dim_labels(self):
        desc = _make_tensor_desc([10, 512, 512], dim_labels=["z", "y", "x"])
        y_idx, x_idx = get_xy_dim_indices(desc)
        assert y_idx == 1
        assert x_idx == 2

    def test_case_insensitive_labels(self):
        desc = _make_tensor_desc([512, 512], dim_labels=["Y", "X"])
        y_idx, x_idx = get_xy_dim_indices(desc)
        assert y_idx == 0
        assert x_idx == 1

    def test_fallback_last_two_dims(self):
        desc = _make_tensor_desc([3, 100, 200])
        y_idx, x_idx = get_xy_dim_indices(desc)
        assert y_idx == 2
        assert x_idx == 1

    def test_2d_no_labels(self):
        desc = _make_tensor_desc([100, 200])
        y_idx, x_idx = get_xy_dim_indices(desc)
        assert y_idx == 1
        assert x_idx == 0

    def test_labels_without_xy_falls_back(self):
        desc = _make_tensor_desc([10, 512, 512], dim_labels=["z", "r", "c"])
        y_idx, x_idx = get_xy_dim_indices(desc)
        assert y_idx == 2
        assert x_idx == 1


class TestBuildPyramidLevels:
    def test_small_image_returns_single_level(self):
        desc = _make_tensor_desc([256, 256])
        client = MagicMock()
        mock_arr = MagicMock()
        client.get_tensor.return_value = mock_arr

        levels = build_pyramid_levels(client, "src", "t1", desc)

        assert len(levels) == 1
        assert levels[0] is mock_arr
        client.get_tensor.assert_called_once_with("src", "t1")

    def test_threshold_boundary_no_pyramid(self):
        desc = _make_tensor_desc([PYRAMID_THRESHOLD, PYRAMID_THRESHOLD])
        client = MagicMock()
        client.get_tensor.return_value = MagicMock()

        levels = build_pyramid_levels(client, "src", "t1", desc)
        assert len(levels) == 1

    def test_large_image_builds_pyramid(self):
        desc = _make_tensor_desc([8192, 8192])
        client = MagicMock()
        client.get_tensor.return_value = MagicMock()

        levels = build_pyramid_levels(client, "src", "t1", desc)

        assert len(levels) > 1
        # First call should be scale=1 (no scale_hint with all 1s)
        first_call = client.get_tensor.call_args_list[0]
        assert first_call == call("src", "t1", scale_hint=[1, 1])

    def test_pyramid_only_scales_xy(self):
        desc = _make_tensor_desc([10, 8192, 8192], dim_labels=["z", "y", "x"])
        client = MagicMock()
        client.get_tensor.return_value = MagicMock()

        levels = build_pyramid_levels(client, "src", "t1", desc)

        assert len(levels) > 1
        # First level: scale_hint = [1, 1, 1]
        first_hint = client.get_tensor.call_args_list[0][1]["scale_hint"]
        assert first_hint == [1, 1, 1]
        # Second level: z stays 1, y and x scale to 2
        second_hint = client.get_tensor.call_args_list[1][1]["scale_hint"]
        assert second_hint[0] == 1  # z
        assert second_hint[1] == 2  # y
        assert second_hint[2] == 2  # x
