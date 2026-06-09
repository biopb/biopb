"""Tests for MCP helper functions (viewer.add_tensor)."""

from unittest.mock import MagicMock, patch

import pytest

from biopb_mcp.mcp._helpers import (
    patch_viewer_add_tensor,
    viewer_window_alive,
)


@pytest.fixture
def viewer():
    return MagicMock()


@pytest.fixture
def connection():
    w = MagicMock()
    w.client = None
    w.sources = {}
    return w


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


class TestPatchViewerAddTensor:
    """Tests for the monkey-patched viewer.add_tensor."""

    def test_patches_method_on_viewer(self, viewer, connection):
        patch_viewer_add_tensor(viewer, connection)
        assert hasattr(viewer, "add_tensor")
        assert callable(viewer.add_tensor)

    def test_raises_when_no_client(self, viewer, connection):
        patch_viewer_add_tensor(viewer, connection)
        with pytest.raises(RuntimeError, match="No tensor server connected"):
            viewer.add_tensor("some_source")

    def test_raises_when_source_not_found_without_get_source(
        self, viewer, connection
    ):
        client = MagicMock()
        del client.get_source  # simulate biopb without the direct-fetch method
        connection.client = client
        connection.sources = {"a": MagicMock()}
        patch_viewer_add_tensor(viewer, connection)

        with pytest.raises(ValueError, match="not found"):
            viewer.add_tensor("nonexistent")

    def test_fallback_to_get_source_when_uncached(self, viewer, connection):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/remote_img", [tensor])
        client = MagicMock()
        client.get_source.return_value = src
        client.get_physical_scale.return_value = None
        connection.client = client
        connection.sources = {}  # source absent from the cached catalog

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_add_tensor(viewer, connection)
            name = viewer.add_tensor("remote_src")

        client.get_source.assert_called_once_with("remote_src", None)
        assert name == "remote_img"
        viewer.add_image.assert_called_once_with(mock_arr, name="remote_img")

    def test_fallback_forwards_tensor_id(self, viewer, connection):
        t2 = _make_tensor("t2", [128, 128])
        src = _make_source("", [t2])  # synthesized descriptor: no source_url
        client = MagicMock()
        client.get_source.return_value = src
        connection.client = client
        connection.sources = {}

        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[MagicMock()],
        ):
            patch_viewer_add_tensor(viewer, connection)
            viewer.add_tensor("remote_src", tensor_id="t2", name="x")

        client.get_source.assert_called_once_with("remote_src", "t2")

    def test_auto_selects_single_tensor(self, viewer, connection):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/my_image", [tensor])
        connection.client = MagicMock()
        connection.client.get_physical_scale.return_value = None
        connection.sources = {"src1": src}

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_add_tensor(viewer, connection)
            name = viewer.add_tensor("src1")

        assert name == "my_image"
        viewer.add_image.assert_called_once_with(mock_arr, name="my_image")

    def test_compute_scheduler_wraps_layer_array(self, viewer, connection):
        """With a scheduler set, the array passed to add_image is pinned to a
        single-process scheduler (issue #8)."""
        from biopb_mcp._viewer_compute import _ViewerArray

        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/my_image", [tensor])
        connection.client = MagicMock()
        connection.client.get_physical_scale.return_value = None
        connection.sources = {"src1": src}

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_add_tensor(
                viewer, connection, compute_scheduler="threads"
            )
            viewer.add_tensor("src1")

        (passed,), kwargs = viewer.add_image.call_args
        assert isinstance(passed, _ViewerArray)
        assert passed._arr is mock_arr
        assert passed._scheduler == "threads"

    def test_requires_tensor_id_for_multi_tensor(self, viewer, connection):
        t1 = _make_tensor("t1", [256, 256])
        t2 = _make_tensor("t2", [128, 128])
        src = _make_source("http://server/data/multi", [t1, t2])
        connection.client = MagicMock()
        connection.sources = {"src1": src}
        patch_viewer_add_tensor(viewer, connection)

        with pytest.raises(ValueError, match="specify tensor_id"):
            viewer.add_tensor("src1")

    def test_explicit_tensor_id_and_name(self, viewer, connection):
        t1 = _make_tensor("t1", [256, 256])
        t2 = _make_tensor("t2", [128, 128])
        src = _make_source("http://server/data/multi", [t1, t2])
        connection.client = MagicMock()
        connection.client.get_physical_scale.return_value = None
        connection.sources = {"src1": src}

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_add_tensor(viewer, connection)
            name = viewer.add_tensor("src1", tensor_id="t2", name="custom")

        assert name == "custom"
        viewer.add_image.assert_called_once_with(mock_arr, name="custom")

    def test_multiscale_pyramid(self, viewer, connection):
        tensor = _make_tensor("t1", [8192, 8192])
        src = _make_source("http://server/big", [tensor])
        connection.client = MagicMock()
        connection.client.get_physical_scale.return_value = None
        connection.sources = {"src1": src}

        levels = [MagicMock(), MagicMock()]
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=levels,
        ):
            patch_viewer_add_tensor(viewer, connection)
            viewer.add_tensor("src1")

        viewer.add_image.assert_called_once_with(
            levels, name="big", multiscale=True
        )

    def test_raises_for_invalid_tensor_id(self, viewer, connection):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/img", [tensor])
        connection.client = MagicMock()
        connection.sources = {"src1": src}
        patch_viewer_add_tensor(viewer, connection)

        with pytest.raises(ValueError, match="Tensor 'wrong' not found"):
            viewer.add_tensor("src1", tensor_id="wrong")

    def test_applies_ome_scale_and_metadata(self, viewer, connection):
        tensor = _make_tensor("t1", [256, 256])
        tensor.dim_labels = ["y", "x"]
        src = _make_source("http://server/data/cal", [tensor])
        client = MagicMock()
        # get_physical_scale returns the compact per-dim (scale, unit) summary
        # in source axis order [y, x] (the descriptor field, issue #31).
        client.get_physical_scale.return_value = ([0.25, 0.5], ["µm", "µm"])
        connection.client = client
        connection.sources = {"src1": src}

        # build_pyramid_levels emits canonical [..., Z, Y, X] levels; a 2D
        # source becomes [Z(=1), Y, X], so the level reports ndim 3 and
        # build_layer_scale maps psz/psy/psx onto the trailing axes.
        mock_arr = MagicMock()
        mock_arr.ndim = 3
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_add_tensor(viewer, connection)
            viewer.add_tensor("src1")

        _, kwargs = viewer.add_image.call_args
        assert kwargs["scale"] == [1.0, 0.25, 0.5]
        phys = kwargs["metadata"]["ome_physical_size"]
        assert phys["physical_size_x"] == 0.5
        assert phys["physical_size_y"] == 0.25


class TestViewerWindowAlive:
    """Tests for the closed-window liveness probe."""

    def _viewer_with_window(self, is_visible):
        viewer = MagicMock()
        viewer.window._qt_window.isVisible.return_value = is_visible
        return viewer

    def test_alive_when_visible(self):
        assert viewer_window_alive(self._viewer_with_window(True)) is True

    def test_alive_when_minimized_or_hidden(self):
        # isVisible() returning False just means hidden/minimized, not destroyed.
        assert viewer_window_alive(self._viewer_with_window(False)) is True

    def test_dead_when_qt_object_deleted(self):
        # PyQt raises this RuntimeError on access to a destroyed C++ object.
        viewer = MagicMock()
        viewer.window._qt_window.isVisible.side_effect = RuntimeError(
            "wrapped C/C++ object of type CanvasBackendDesktop has been deleted"
        )
        assert viewer_window_alive(viewer) is False

    def test_dead_when_qt_window_missing(self):
        # Programmatic Window.close() does `del self._qt_window`.
        viewer = MagicMock()
        viewer.window = MagicMock(spec=[])  # no _qt_window attribute
        assert viewer_window_alive(viewer) is False

    def test_dead_when_window_is_none(self):
        viewer = MagicMock()
        viewer.window = None
        assert viewer_window_alive(viewer) is False

    def test_dead_for_headless_sentinel(self):
        class _Sentinel:
            def __getattr__(self, name):
                raise RuntimeError("napari viewer unavailable: headless")

        assert viewer_window_alive(_Sentinel()) is False
