"""Tests for MCP helper functions (viewer.add_tensor)."""

from unittest.mock import MagicMock, patch

import pytest
from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_mcp.mcp._helpers import (
    _get_url_stem,
    patch_viewer_tensor_methods,
    resync_view_for_capture,
    viewer_window_alive,
)


@pytest.fixture
def viewer():
    return MagicMock()


@pytest.fixture
def connection():
    w = MagicMock()
    w.client = None
    return w


def _row(source_url, array_ids):
    """A `sources` row as ``query_sources(format="records")`` returns it."""
    return {"source_url": source_url, "tensors": [{"array_id": a} for a in array_ids]}


def _desc(array_id, shape, dim_labels=()):
    return TensorDescriptor(
        array_id=array_id, shape=shape, dtype="float32", dim_labels=dim_labels
    )


def _serve(connection, rows, descriptors=()):
    """Connect *connection* to a fake server holding *rows* and *descriptors*."""
    by_id = {d.array_id: d for d in descriptors}

    def query_sources(sql, format=None):  # noqa: A002 - mirrors the client
        return [row for sid, row in rows.items() if f"'{sid}'" in sql]

    def get_descriptor(array_id):
        if array_id not in by_id:
            raise RuntimeError(f"no such tensor: {array_id}")
        return by_id[array_id]

    client = MagicMock()
    client.query_sources.side_effect = query_sources
    client.get_descriptor.side_effect = get_descriptor
    client.get_physical_scale.return_value = None
    connection.client = client
    return client


class TestGetUrlStem:
    """source_url -> last path component (used to name the added viewer layer)."""

    def test_file_url(self):
        assert _get_url_stem("file:///home/me/data/img.tif") == "img.tif"

    def test_bare_path(self):
        assert _get_url_stem("/data/cells/exp.zarr") == "exp.zarr"

    def test_dnd_single_source_strips_scheme(self):
        # dnd:// puts the basename in the netloc, so a naive urlparse().path
        # yields "" and falls back to the raw url. Must return the basename.
        assert _get_url_stem("dnd://exp.zarr") == "exp.zarr"

    def test_dnd_folder_child_returns_leaf(self):
        assert _get_url_stem("dnd://my_experiment/sub/b.tif") == "b.tif"

    def test_empty_url(self):
        assert _get_url_stem("") == ""


class TestPatchViewerAddTensor:
    """Tests for the monkey-patched viewer.add_tensor."""

    def test_patches_method_on_viewer(self, viewer, connection):
        patch_viewer_tensor_methods(viewer, connection)
        assert hasattr(viewer, "add_tensor")
        assert callable(viewer.add_tensor)

    def test_raises_when_no_client(self, viewer, connection):
        patch_viewer_tensor_methods(viewer, connection)
        with pytest.raises(RuntimeError, match="No tensor server connected"):
            viewer.add_tensor("some_source")

    def test_raises_when_source_not_found(self, viewer, connection):
        _serve(connection, {})
        patch_viewer_tensor_methods(viewer, connection)

        with pytest.raises(ValueError, match="not found"):
            viewer.add_tensor("nonexistent")

    def test_a_source_the_catalog_does_not_list_is_described_directly(
        self, viewer, connection
    ):
        # No row (a tensor attached to a source rather than listed in it): the
        # descriptor is asked for by the id itself, and the layer is named by it.
        client = _serve(connection, {}, [_desc("remote_src", [256, 256])])

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("remote_src")

        client.get_descriptor.assert_called_once_with("remote_src")
        assert name == "remote_src"
        viewer.add_image.assert_called_once_with(
            mock_arr, name="remote_src", metadata={"array_id": "remote_src"}
        )

    def test_the_source_is_looked_up_by_a_quoted_literal(self, viewer, connection):
        client = _serve(connection, {})
        patch_viewer_tensor_methods(viewer, connection)
        with pytest.raises(ValueError):
            viewer.add_tensor("it's")
        (sql,), _ = client.query_sources.call_args
        assert "source_id = 'it''s'" in sql

    def test_auto_selects_single_tensor(self, viewer, connection):
        _serve(
            connection,
            {"src1": _row("http://server/data/my_image", ["t1"])},
            [_desc("t1", [256, 256])],
        )

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("src1")

        assert name == "my_image"
        viewer.add_image.assert_called_once_with(
            mock_arr, name="my_image", metadata={"array_id": "t1"}
        )

    def test_compute_scheduler_wraps_layer_array(self, viewer, connection):
        """With a scheduler set, the array passed to add_image is pinned to a
        single-process scheduler (issue #8)."""
        from biopb_mcp._viewer_compute import _ViewerArray

        _serve(
            connection,
            {"src1": _row("http://server/data/my_image", ["t1"])},
            [_desc("t1", [256, 256])],
        )

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_tensor_methods(viewer, connection, compute_scheduler="threads")
            viewer.add_tensor("src1")

        (passed,), kwargs = viewer.add_image.call_args
        assert isinstance(passed, _ViewerArray)
        assert passed._arr is mock_arr
        assert passed._scheduler == "threads"

    def test_requires_tensor_id_for_multi_tensor(self, viewer, connection):
        _serve(connection, {"src1": _row("http://server/data/multi", ["t1", "t2"])})
        patch_viewer_tensor_methods(viewer, connection)

        with pytest.raises(ValueError, match="specify tensor_id"):
            viewer.add_tensor("src1")

    def test_explicit_tensor_id_and_name(self, viewer, connection):
        _serve(
            connection,
            {"src1": _row("http://server/data/multi", ["t1", "t2"])},
            [_desc("t1", [256, 256]), _desc("t2", [128, 128])],
        )

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("src1", tensor_id="t2", name="custom")

        assert name == "custom"
        viewer.add_image.assert_called_once_with(
            mock_arr, name="custom", metadata={"array_id": "t2"}
        )

    def test_qualified_array_id_selects_the_tensor(self, viewer, connection):
        # One id, addressed exactly as client.get_tensor addresses it (#650):
        # the slash-free prefix routes, the full id picks the tensor.
        t2 = _desc("src1/t2", [128, 128])
        _serve(
            connection,
            {"src1": _row("http://server/data/multi", ["src1/t1", "src1/t2"])},
            [_desc("src1/t1", [256, 256]), t2],
        )

        with patch("biopb_mcp._tensor_utils.add_tensor_layer") as add_layer:
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("src1/t2")

        _, _, source_id, tensor_id, tensor_desc = add_layer.call_args[0]
        assert (source_id, tensor_id) == ("src1", "src1/t2")
        assert tensor_desc is t2
        assert name == "multi/t2"

    def test_legacy_source_id_keyword_still_accepted(self, viewer, connection):
        _serve(
            connection,
            {"src1": _row("http://server/data/my_image", ["t1"])},
            [_desc("t1", [256, 256])],
        )

        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[MagicMock()],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor(source_id="src1")

        assert name == "my_image"

    def test_requires_an_id(self, viewer, connection):
        patch_viewer_tensor_methods(viewer, connection)
        with pytest.raises(TypeError, match="requires an array_id"):
            viewer.add_tensor()

    def test_multiscale_pyramid(self, viewer, connection):
        _serve(
            connection,
            {"src1": _row("http://server/big", ["t1"])},
            [_desc("t1", [8192, 8192])],
        )

        levels = [MagicMock(), MagicMock()]
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=levels,
        ):
            patch_viewer_tensor_methods(viewer, connection)
            viewer.add_tensor("src1")

        viewer.add_image.assert_called_once_with(
            levels, name="big", multiscale=True, metadata={"array_id": "t1"}
        )

    def test_raises_for_invalid_tensor_id(self, viewer, connection):
        _serve(
            connection,
            {"src1": _row("http://server/data/img", ["t1"])},
            [_desc("t1", [256, 256])],
        )
        patch_viewer_tensor_methods(viewer, connection)

        with pytest.raises(ValueError, match="Tensor 'wrong' not found"):
            viewer.add_tensor("src1", tensor_id="wrong")

    def test_applies_ome_scale_and_metadata(self, viewer, connection):
        client = _serve(
            connection,
            {"src1": _row("http://server/data/cal", ["t1"])},
            [_desc("t1", [256, 256], dim_labels=["y", "x"])],
        )
        # get_physical_scale returns the compact per-dim (scale, unit) summary
        # in source axis order [y, x] (the descriptor field, issue #31).
        client.get_physical_scale.return_value = ([0.25, 0.5], ["µm", "µm"])

        # build_pyramid_levels returns the source array as served: a 2-D source
        # stays 2-D, so the level reports ndim 2 and build_layer_scale places
        # psy/psx on the axes they describe.
        mock_arr = MagicMock()
        mock_arr.ndim = 2
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            viewer.add_tensor("src1")

        _, kwargs = viewer.add_image.call_args
        assert kwargs["scale"] == [0.25, 0.5]
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

    def test_dead_when_every_attribute_raises(self):
        # Defensive: a viewer stand-in whose attribute access raises must read
        # as "no window", never propagate.
        class _Raising:
            def __getattr__(self, name):
                raise RuntimeError("napari viewer unavailable")

        assert viewer_window_alive(_Raising()) is False


class _FakeLayer:
    """A layer whose ``loaded`` walks a sequence (sticking on the last value),
    so the resync pump loop can be driven deterministically."""

    def __init__(self, loaded_values):
        self._vals = list(loaded_values)

    @property
    def loaded(self):
        v = self._vals[0]
        if len(self._vals) > 1:
            self._vals.pop(0)
        return v


@patch("qtpy.QtWidgets.QApplication.processEvents")
class TestResyncViewForCapture:
    """Before a screenshot, wait for the current view's async slice to load so
    the capture reflects the requested state (not a pre-load frame)."""

    def _viewer(self, layers, slicer=None):
        v = MagicMock()
        v.layers = layers
        v._layer_slicer = MagicMock() if slicer is None else slicer
        return v

    @patch("napari.settings.get_settings")
    def test_noop_when_async_off(self, get_settings, proc):
        get_settings.return_value.experimental.async_ = False
        v = self._viewer([_FakeLayer([False])])
        resync_view_for_capture(v)
        v._layer_slicer.submit.assert_not_called()
        proc.assert_not_called()

    @patch("napari.settings.get_settings")
    def test_noop_when_no_layers(self, get_settings, proc):
        get_settings.return_value.experimental.async_ = True
        v = self._viewer([])
        resync_view_for_capture(v)
        v._layer_slicer.submit.assert_not_called()
        proc.assert_not_called()

    @patch("napari.settings.get_settings")
    def test_submits_and_skips_pump_when_loaded(self, get_settings, proc):
        get_settings.return_value.experimental.async_ = True
        v = self._viewer([_FakeLayer([True])])
        resync_view_for_capture(v)
        v._layer_slicer.submit.assert_called_once_with(
            layers=v.layers, dims=v.dims, force=True
        )
        proc.assert_not_called()  # already loaded -> loop body never runs

    @patch("time.sleep")
    @patch("napari.settings.get_settings")
    def test_pumps_until_loaded(self, get_settings, sleep, proc):
        get_settings.return_value.experimental.async_ = True
        v = self._viewer([_FakeLayer([False, False, True])])
        resync_view_for_capture(v, timeout=5)
        assert proc.call_count >= 2
        v._layer_slicer.submit.assert_called_once()

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("napari.settings.get_settings")
    def test_breaks_at_timeout(self, get_settings, monotonic, sleep, proc):
        get_settings.return_value.experimental.async_ = True
        # Each call jumps 10s, so the deadline is exceeded almost immediately:
        # the loop must break rather than hang on a never-loading layer.
        ticks = [0.0]

        def _mono():
            ticks[0] += 10.0
            return ticks[0]

        monotonic.side_effect = _mono
        v = self._viewer([_FakeLayer([False])])
        resync_view_for_capture(v, timeout=5)  # returns, does not hang
        assert proc.called

    @patch("napari.settings.get_settings")
    def test_no_slicer_skips_submit_but_pumps(self, get_settings, proc):
        get_settings.return_value.experimental.async_ = True
        v = self._viewer([_FakeLayer([True])], slicer=None)
        v._layer_slicer = None
        resync_view_for_capture(v)  # must not raise

    @patch("napari.settings.get_settings")
    def test_never_raises_on_submit_error(self, get_settings, proc):
        get_settings.return_value.experimental.async_ = True
        v = self._viewer([_FakeLayer([True])])
        v._layer_slicer.submit.side_effect = RuntimeError("boom")
        resync_view_for_capture(v)  # swallowed, no raise


class TestViewerTensor:
    """``viewer.tensor(layer)`` -- reading a layer back as a plain array.

    Retires ``layer.data[0] if layer.multiscale else layer.data``, which meant
    level 0 on a multiscale layer and plane 0 on a single-scale one, and
    yielded a ``_ViewerArray`` proxy either way (biopb/biopb#973, #974).
    """

    @staticmethod
    def _loaded_layer(levels, array_id="t1"):
        """A layer as ``add_tensor_layer`` builds one: wrapped, with an origin."""
        import napari

        from biopb_mcp._viewer_compute import wrap_levels

        wrapped = wrap_levels(levels, "synchronous")
        metadata = {"array_id": array_id} if array_id else {}
        if len(levels) > 1:
            return napari.layers.Image(wrapped, multiscale=True, metadata=metadata)
        return napari.layers.Image(wrapped[0], metadata=metadata)

    @staticmethod
    def _pyramid():
        import dask.array as da

        return [
            da.zeros((4, 512, 512), dtype="uint8"),
            da.zeros((4, 256, 256), dtype="uint8"),
            da.zeros((4, 128, 128), dtype="uint8"),
        ]

    def test_a_loaded_layer_unwraps_to_its_own_level_0(self, viewer, connection):
        import dask.array as da

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        levels = self._pyramid()

        arr = viewer.tensor(self._loaded_layer(levels, array_id="src/t1"))

        assert arr is levels[0]
        assert isinstance(arr, da.Array), "the proxy was handed back unwrapped"

    def test_it_never_goes_back_to_the_server(self, viewer, connection):
        client = MagicMock()
        connection.client = client
        patch_viewer_tensor_methods(viewer, connection)

        viewer.tensor(self._loaded_layer(self._pyramid(), array_id="src/t1"))

        client.get_tensor.assert_not_called()
        client.get_descriptor.assert_not_called()

    def test_a_name_resolves_against_the_viewer(self, viewer, connection):
        import dask.array as da

        connection.client = MagicMock()
        levels = self._pyramid()
        viewer.layers = {"big": self._loaded_layer(levels)}
        patch_viewer_tensor_methods(viewer, connection)

        arr = viewer.tensor("big")

        assert arr is levels[0]
        assert isinstance(arr, da.Array)

    def test_a_layer_the_agent_built_unwraps_the_same_way(self, viewer, connection):
        import dask.array as da

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)

        own = da.zeros((4, 64, 64), dtype="uint8")

        arr = viewer.tensor(self._loaded_layer([own], array_id=None))

        assert arr is own
        assert isinstance(arr, da.Array)

    def test_an_unattributed_pyramid_unwraps_level_0(self, viewer, connection):
        import dask.array as da

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        levels = self._pyramid()

        arr = viewer.tensor(self._loaded_layer(levels, array_id=None))

        assert arr is levels[0]
        assert isinstance(arr, da.Array)

    def test_level_0_is_the_full_resolution_shape_reports(self, viewer, connection):
        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        layer = self._loaded_layer(self._pyramid())

        assert viewer.tensor(layer).shape == layer.data.shape

    def test_a_numpy_layer_comes_back_as_it_went_in(self, viewer, connection):
        import napari
        import numpy as np

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        own = np.zeros((8, 8), dtype=np.int32)

        arr = viewer.tensor(napari.layers.Labels(own))

        assert arr is own

    def test_a_disconnected_server_changes_nothing(self, viewer, connection):
        import dask.array as da

        connection.client = None
        patch_viewer_tensor_methods(viewer, connection)
        levels = self._pyramid()

        assert viewer.tensor(self._loaded_layer(levels)) is levels[0]
        assert isinstance(levels[0], da.Array)

    def test_something_that_is_not_a_layer_is_rejected(self, viewer, connection):
        patch_viewer_tensor_methods(viewer, connection)

        with pytest.raises(TypeError, match="takes a layer or a layer name"):
            viewer.tensor(object())

    def test_it_beats_np_asarray_on_the_layer(self, viewer, connection):
        """``MultiScaleData.__array__`` returns the *lowest* level, so numpy
        or scikit-image on ``layer.data`` silently computes at the bottom of
        the pyramid (biopb/biopb#973)."""
        import numpy as np

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        layer = self._loaded_layer(self._pyramid())

        assert np.asarray(layer.data).shape == (4, 128, 128)  # the trap
        assert viewer.tensor(layer).shape == (4, 512, 512)  # the way out

    def test_it_survives_the_agent_facing_viewer_proxy(self, connection):
        """The agent holds the main-thread marshaling proxy, not the real
        viewer, so the layer handle and the returned array have to survive it.

        Headless ``ViewerModel`` for the reason ``test_viewer_proxy`` gives:
        the real viewer's GL canvas segfaults on offscreen runners.
        """
        import dask.array as da
        from napari.components import ViewerModel

        from biopb_mcp._viewer_compute import wrap_levels
        from biopb_mcp.mcp._viewer_proxy import make_viewer_proxy

        real = ViewerModel()
        levels = self._pyramid()
        real.add_image(
            wrap_levels(levels, "synchronous"),
            multiscale=True,
            name="big",
            metadata={"array_id": "src/t1"},
        )
        connection.client = MagicMock()
        patch_viewer_tensor_methods(real, connection)

        proxy = make_viewer_proxy(real)

        assert proxy.tensor(proxy.layers["big"]) is levels[0]
        assert isinstance(levels[0], da.Array)
        assert proxy.tensor("big") is levels[0]
