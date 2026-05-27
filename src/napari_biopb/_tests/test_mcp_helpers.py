"""Tests for MCP helper functions (viewer.load_tensor)."""

from unittest.mock import MagicMock, patch

import pytest

from napari_biopb.mcp._helpers import patch_viewer_load_tensor


@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    bridge.tensor_client = None
    bridge.tensor_sources = {}
    bridge.viewer = MagicMock()
    return bridge


def _make_source(source_url, tensors):
    """Create a mock DataSourceDescriptor."""
    src = MagicMock()
    src.source_url = source_url
    src.tensors = tensors
    return src


def _make_tensor(array_id, shape, dtype="float32"):
    t = MagicMock()
    t.array_id = array_id
    t.shape = shape
    t.dtype = dtype
    t.dim_labels = []
    return t


class TestPatchViewerLoadTensor:
    """Tests for the monkey-patched viewer.load_tensor."""

    def test_patches_method_on_viewer(self, mock_bridge):
        patch_viewer_load_tensor(mock_bridge)
        assert hasattr(mock_bridge.viewer, "load_tensor")
        assert callable(mock_bridge.viewer.load_tensor)

    def test_raises_when_no_client(self, mock_bridge):
        patch_viewer_load_tensor(mock_bridge)
        with pytest.raises(RuntimeError, match="No tensor server connected"):
            mock_bridge.viewer.load_tensor("some_source")

    def test_raises_when_source_not_found(self, mock_bridge):
        mock_bridge.tensor_client = MagicMock()
        mock_bridge.tensor_sources = {"a": MagicMock()}
        patch_viewer_load_tensor(mock_bridge)

        with pytest.raises(ValueError, match="not found"):
            mock_bridge.viewer.load_tensor("nonexistent")

    def test_auto_selects_single_tensor(self, mock_bridge):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/my_image", [tensor])
        mock_bridge.tensor_client = MagicMock()
        mock_bridge.tensor_sources = {"src1": src}

        mock_arr = MagicMock()
        with patch(
            "napari_biopb._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_load_tensor(mock_bridge)
            name = mock_bridge.viewer.load_tensor("src1")

        assert name == "my_image"
        mock_bridge.viewer.add_image.assert_called_once_with(
            mock_arr, name="my_image"
        )

    def test_requires_tensor_id_for_multi_tensor(self, mock_bridge):
        t1 = _make_tensor("t1", [256, 256])
        t2 = _make_tensor("t2", [128, 128])
        src = _make_source("http://server/data/multi", [t1, t2])
        mock_bridge.tensor_client = MagicMock()
        mock_bridge.tensor_sources = {"src1": src}
        patch_viewer_load_tensor(mock_bridge)

        with pytest.raises(ValueError, match="specify tensor_id"):
            mock_bridge.viewer.load_tensor("src1")

    def test_explicit_tensor_id_and_name(self, mock_bridge):
        t1 = _make_tensor("t1", [256, 256])
        t2 = _make_tensor("t2", [128, 128])
        src = _make_source("http://server/data/multi", [t1, t2])
        mock_bridge.tensor_client = MagicMock()
        mock_bridge.tensor_sources = {"src1": src}

        mock_arr = MagicMock()
        with patch(
            "napari_biopb._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_load_tensor(mock_bridge)
            name = mock_bridge.viewer.load_tensor(
                "src1", tensor_id="t2", name="custom"
            )

        assert name == "custom"
        mock_bridge.viewer.add_image.assert_called_once_with(
            mock_arr, name="custom"
        )

    def test_multiscale_pyramid(self, mock_bridge):
        tensor = _make_tensor("t1", [8192, 8192])
        src = _make_source("http://server/big", [tensor])
        mock_bridge.tensor_client = MagicMock()
        mock_bridge.tensor_sources = {"src1": src}

        levels = [MagicMock(), MagicMock()]
        with patch(
            "napari_biopb._tensor_utils.build_pyramid_levels",
            return_value=levels,
        ):
            patch_viewer_load_tensor(mock_bridge)
            name = mock_bridge.viewer.load_tensor("src1")

        mock_bridge.viewer.add_image.assert_called_once_with(
            levels, name="big", multiscale=True
        )

    def test_raises_for_invalid_tensor_id(self, mock_bridge):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/img", [tensor])
        mock_bridge.tensor_client = MagicMock()
        mock_bridge.tensor_sources = {"src1": src}
        patch_viewer_load_tensor(mock_bridge)

        with pytest.raises(ValueError, match="Tensor 'wrong' not found"):
            mock_bridge.viewer.load_tensor("src1", tensor_id="wrong")
