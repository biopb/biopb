"""Tests for _chunking.py chunking utilities."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from napari_biopb.image_processing._chunking import (
    IterationSpec,
    ResultBuilder,
    _data_iterator,
    _get_axis_mapping,
    _get_iter_spec,
    _validate_data_shape,
)


class MockImageLayer:
    """Mock napari Image layer for testing."""

    def __init__(self, data, rgb=False, multiscale=False, name="test"):
        self.data = data
        self.rgb = rgb
        self.multiscale = multiscale
        self.name = name


class TestGetAxisMapping:
    """Tests for _get_axis_mapping function."""

    def test_2d_grayscale(self):
        """2D grayscale (Y, X) maps correctly."""
        data = np.random.rand(100, 100)
        layer = MockImageLayer(data, rgb=False)
        axis_order = _get_axis_mapping(layer, is_3d=False)
        assert axis_order == "YX"

    def test_2d_rgb(self):
        """2D RGB (Y, X, C) maps correctly."""
        data = np.random.rand(100, 100, 3)
        layer = MockImageLayer(data, rgb=True)
        axis_order = _get_axis_mapping(layer, is_3d=False)
        assert axis_order == "YXC"

    def test_3d_grayscale(self):
        """3D grayscale (Z, Y, X) maps correctly when is_3d=True."""
        data = np.random.rand(10, 100, 100)
        layer = MockImageLayer(data, rgb=False)
        axis_order = _get_axis_mapping(layer, is_3d=True)
        assert axis_order == "ZYX"

    def test_3d_non_rgb_2d_mode(self):
        """3D non-RGB in 2D mode maps as (T, Y, X)."""
        data = np.random.rand(10, 100, 100)
        layer = MockImageLayer(data, rgb=False)
        axis_order = _get_axis_mapping(layer, is_3d=False)
        assert axis_order == "TYX"

    def test_3d_rgb(self):
        """3D RGB (Z, Y, X, C) maps correctly."""
        data = np.random.rand(10, 100, 100, 3)
        layer = MockImageLayer(data, rgb=True)
        axis_order = _get_axis_mapping(layer, is_3d=True)
        assert axis_order == "ZYXC"

    def test_4d_rgb_time_series(self):
        """4D RGB time series (T, Y, X, C) maps correctly."""
        data = np.random.rand(5, 100, 100, 3)
        layer = MockImageLayer(data, rgb=True)
        axis_order = _get_axis_mapping(layer, is_3d=False)
        assert axis_order == "TYXC"

    def test_4d_grayscale_time_series_3d(self):
        """4D grayscale 3D time series (T, Z, Y, X) maps correctly."""
        data = np.random.rand(5, 10, 100, 100)
        layer = MockImageLayer(data, rgb=False)
        axis_order = _get_axis_mapping(layer, is_3d=True)
        assert axis_order == "TZYX"

    def test_5d_rgb(self):
        """5D RGB (T, Z, Y, X, C) maps correctly."""
        data = np.random.rand(2, 10, 100, 100, 3)
        layer = MockImageLayer(data, rgb=True)
        axis_order = _get_axis_mapping(layer, is_3d=True)
        assert axis_order == "TZYXC"

    def test_multiscale(self):
        """Multiscale data uses first level."""
        data = [np.random.rand(100, 100), np.random.rand(50, 50)]
        layer = MockImageLayer(data, rgb=False, multiscale=True)
        axis_order = _get_axis_mapping(layer, is_3d=False)
        assert axis_order == "YX"

    def test_invalid_ndim_raises(self):
        """Invalid number of dimensions raises ValueError."""
        data = np.random.rand(100)  # 1D
        layer = MockImageLayer(data, rgb=False)
        with pytest.raises(ValueError, match="must have 2-5 dimensions"):
            _get_axis_mapping(layer, is_3d=False)


class TestValidateDataShape:
    """Tests for _validate_data_shape function."""

    def test_no_hint_no_error(self):
        """No hint returns without error."""
        data = np.random.rand(100, 100, 3)
        _validate_data_shape(data, "YXC", None)  # Should not raise

    def test_required_multivalue_warning(self):
        """required_multivalue with size=1 logs warning."""
        data = np.random.rand(100, 100, 1)  # C=1

        # Mock hint
        hint = MagicMock()
        hint.required_multivalue = ["C"]

        _validate_data_shape(data, "YXC", hint)  # Should not raise

    def test_required_multivalue_passes(self):
        """required_multivalue with size>1 passes."""
        data = np.random.rand(100, 100, 3)  # C=3

        hint = MagicMock()
        hint.required_multivalue = ["C"]

        _validate_data_shape(data, "YXC", hint)  # Should not raise


class TestGetIterSpec:
    """Tests for _get_iter_spec function."""

    def test_no_hint_returns_empty_iter_dims(self):
        """No hint returns empty iter_dims set."""
        spec = _get_iter_spec("YXC", None)
        assert spec.iter_dims == set()
        assert spec.axis_order == "YXC"

    def test_expected_singletons(self):
        """expected_singletons removes axis from submission, iter_dims tracks it."""
        hint = MagicMock()
        hint.expected_singletons = ["C"]
        hint.required_multivalue = []

        spec = _get_iter_spec("YXC", hint)
        assert spec.iter_dims == {
            "C"
        }  # C removed from submission, becomes iter_dim
        assert spec.axis_order == "YXC"

    def test_required_multivalue(self):
        """required_multivalue ensures axis stays in submission."""
        hint = MagicMock()
        hint.expected_singletons = []
        hint.required_multivalue = ["T"]  # T must be in submission

        spec = _get_iter_spec("TYXC", hint)
        assert spec.iter_dims == set()  # T in submission, no iteration

    def test_axis_order_preserved(self):
        """axis_order preserves full axis order, T becomes iter_dim if not in default submission."""
        spec = _get_iter_spec("TZYXC", None)
        assert spec.axis_order == "TZYXC"
        assert spec.iter_dims == {"T"}  # T not in default ZYXC


class TestDataIterator:
    """Tests for _data_iterator function."""

    def test_no_iter_dims_yields_once(self):
        """No iter_dims yields full data once."""
        data = np.random.rand(100, 100, 3)
        results = list(_data_iterator(data, set(), "YXC"))
        assert len(results) == 1
        position, chunk = results[0]
        assert position == {}
        assert chunk.shape == data.shape

    def test_single_iter_dim(self):
        """Single iter_dim iterates over that dimension."""
        data = np.random.rand(100, 100, 3)
        results = list(_data_iterator(data, {"C"}, "YXC"))  # Iterate over C
        assert len(results) == 3  # 3 channels
        for i, (position, chunk) in enumerate(results):
            assert position == {2: i}  # C is at index 2 in "YXC"
            assert chunk.shape == (100, 100, 1)

    def test_multi_iter_dims(self):
        """Multiple iter_dims iterates over all combinations."""
        data = np.random.rand(2, 3, 100, 100)  # T, Z, Y, X
        results = list(
            _data_iterator(data, {"T", "Z"}, "TZYX")
        )  # Iterate over T and Z
        assert len(results) == 6  # 2 * 3

        # Each result should have singleton T and Z
        for position, chunk in results:
            assert chunk.shape == (1, 1, 100, 100)


class TestResultBuilder:
    """Tests for ResultBuilder class."""

    def test_single_result(self):
        """Single result builds buffer correctly."""
        iter_spec = IterationSpec(iter_dims=set(), axis_order="YXC")
        builder = ResultBuilder(iter_spec, (1, 1, 1, 100, 100))
        # Chunk is 5D TZCYX
        chunk = np.random.rand(1, 1, 1, 100, 100)
        builder.add_result({}, chunk)
        result = builder.get_result()
        assert result.shape == (1, 1, 1, 100, 100)
        np.testing.assert_array_equal(result, chunk)

    def test_multiple_results(self):
        """Multiple results assemble correctly."""
        # Iterate over T: original has T=3, Y=100, X=100, C=1
        iter_spec = IterationSpec(iter_dims={"T"}, axis_order="TYXC")
        builder = ResultBuilder(iter_spec, (3, 100, 100))
        for i in range(3):
            # Chunk is 5D TZCYX with T=1 (iterated)
            chunk = np.ones((1, 1, 1, 100, 100)) * i
            builder.add_result({0: i}, chunk)  # T is at index 0 in "TYXC"
        result = builder.get_result()
        # Output shape: T=3 (original), Z=1, C=1, Y=100, X=100 from chunk
        assert result.shape == (3, 1, 1, 100, 100)
        for i in range(3):
            np.testing.assert_array_equal(result[i, :, :, :, :], i)

    def test_infer_output_shape(self):
        """Output shape is inferred from first chunk."""
        # Iterate over T and Z: original has T=2, Z=10, Y=100, X=100
        iter_spec = IterationSpec(iter_dims={"T", "Z"}, axis_order="TZYX")
        builder = ResultBuilder(iter_spec, (2, 10, 100, 100))
        # Chunk is 5D TZCYX with T=1, Z=1 (iterated)
        chunk = np.random.rand(1, 1, 1, 100, 100)
        builder.add_result({0: 0, 1: 0}, chunk)
        assert builder.buffer is not None
        # Output: T=2 (orig), Z=10 (orig), C=1 (chunk), Y=100 (chunk), X=100 (chunk)
        assert builder.buffer.shape == (2, 10, 1, 100, 100)

    def test_infer_output_shape_non_iter_dim_size_one(self):
        """Non-iter dimension with size 1 uses chunk size, not original."""
        # Edge case: iterate over T, server returns different Y/X and C=1
        iter_spec = IterationSpec(iter_dims={"T"}, axis_order="TYXC")
        builder = ResultBuilder(iter_spec, (10, 100, 100, 3))
        # Chunk is 5D TZCYX: T=1 (iter), Z=1, C=1, Y=50, X=50 (server downsamples)
        chunk = np.random.rand(1, 1, 1, 50, 50)
        builder.add_result({0: 0}, chunk)
        assert builder.buffer is not None
        # T uses original (10), Z/C/Y/X use chunk size (1, 1, 50, 50)
        assert builder.buffer.shape == (10, 1, 1, 50, 50)
