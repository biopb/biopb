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
        client = MagicMock()
        # The uncached-source fetch fails -> add_tensor surfaces "not found".
        client.get_descriptor.side_effect = RuntimeError("no such source")
        connection.client = client
        connection.sources = {"a": MagicMock()}
        patch_viewer_tensor_methods(viewer, connection)

        with pytest.raises(ValueError, match="not found"):
            viewer.add_tensor("nonexistent")

    def test_fallback_to_get_descriptor_when_uncached(self, viewer, connection):
        # Source absent from the (possibly truncated) cached catalog -> fetch the
        # tensor descriptor directly by a bare source_id (resolves the default
        # tensor) and wrap it as a single-tensor source.
        client = MagicMock()
        client.get_descriptor.return_value = TensorDescriptor(
            array_id="remote_src", shape=[256, 256], dtype="float32"
        )
        client.get_physical_scale.return_value = None
        connection.client = client
        connection.sources = {}

        mock_arr = MagicMock()
        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[mock_arr],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("remote_src")

        client.get_descriptor.assert_called_once_with("remote_src")
        # No source_url on a descriptor-only fetch, so the layer name is the id.
        assert name == "remote_src"
        viewer.add_image.assert_called_once_with(
            mock_arr, name="remote_src", metadata={"array_id": "remote_src"}
        )

    def test_fallback_forwards_tensor_id(self, viewer, connection):
        # A within-source field is fetched by its qualified array_id.
        client = MagicMock()
        client.get_descriptor.return_value = TensorDescriptor(
            array_id="remote_src/t2", shape=[128, 128], dtype="float32"
        )
        client.get_physical_scale.return_value = None
        connection.client = client
        connection.sources = {}

        with patch(
            "biopb_mcp._tensor_utils.build_pyramid_levels",
            return_value=[MagicMock()],
        ):
            patch_viewer_tensor_methods(viewer, connection)
            viewer.add_tensor("remote_src", tensor_id="remote_src/t2", name="x")

        client.get_descriptor.assert_called_once_with("remote_src/t2")

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
            patch_viewer_tensor_methods(viewer, connection, compute_scheduler="threads")
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
        patch_viewer_tensor_methods(viewer, connection)

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
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("src1", tensor_id="t2", name="custom")

        assert name == "custom"
        viewer.add_image.assert_called_once_with(
            mock_arr, name="custom", metadata={"array_id": "t2"}
        )

    def test_qualified_array_id_selects_the_tensor(self, viewer, connection):
        # One id, addressed exactly as client.get_tensor addresses it (#650):
        # the slash-free prefix routes, the full id picks the tensor.
        t1 = _make_tensor("src1/t1", [256, 256])
        t2 = _make_tensor("src1/t2", [128, 128])
        src = _make_source("http://server/data/multi", [t1, t2])
        connection.client = MagicMock()
        connection.client.get_physical_scale.return_value = None
        connection.sources = {"src1": src}

        with patch("biopb_mcp._tensor_utils.add_tensor_layer") as add_layer:
            patch_viewer_tensor_methods(viewer, connection)
            name = viewer.add_tensor("src1/t2")

        _, _, source_id, tensor_id, tensor_desc = add_layer.call_args[0]
        assert (source_id, tensor_id) == ("src1", "src1/t2")
        assert tensor_desc is t2
        assert name == "multi/t2"

    def test_qualified_array_id_when_source_uncached(self, viewer, connection):
        # The descriptor fetch takes the full array_id, and the wrapped
        # single-tensor source is keyed by the routing prefix.
        client = MagicMock()
        client.get_descriptor.return_value = TensorDescriptor(
            array_id="remote_src/t2", shape=[128, 128], dtype="float32"
        )
        client.get_physical_scale.return_value = None
        connection.client = client
        connection.sources = {}

        with patch("biopb_mcp._tensor_utils.add_tensor_layer") as add_layer:
            patch_viewer_tensor_methods(viewer, connection)
            viewer.add_tensor("remote_src/t2")

        client.get_descriptor.assert_called_once_with("remote_src/t2")
        _, _, source_id, tensor_id, _ = add_layer.call_args[0]
        assert (source_id, tensor_id) == ("remote_src", "remote_src/t2")

    def test_legacy_source_id_keyword_still_accepted(self, viewer, connection):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/my_image", [tensor])
        connection.client = MagicMock()
        connection.client.get_physical_scale.return_value = None
        connection.sources = {"src1": src}

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
            patch_viewer_tensor_methods(viewer, connection)
            viewer.add_tensor("src1")

        viewer.add_image.assert_called_once_with(
            levels, name="big", multiscale=True, metadata={"array_id": "t1"}
        )

    def test_raises_for_invalid_tensor_id(self, viewer, connection):
        tensor = _make_tensor("t1", [256, 256])
        src = _make_source("http://server/data/img", [tensor])
        connection.client = MagicMock()
        connection.sources = {"src1": src}
        patch_viewer_tensor_methods(viewer, connection)

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

    The accessor that retires ``layer.data[0] if layer.multiscale else
    layer.data`` (biopb/biopb#974). That idiom is not one expression: on a
    multiscale layer ``data[0]`` is *level 0*, on a single-scale one it is
    *plane 0*, and what it yields either way is a ``_ViewerArray`` proxy that
    fails ``isinstance(..., da.Array)`` (biopb/biopb#973).
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

    def test_a_loaded_layer_is_read_from_the_server(self, viewer, connection):
        import dask.array as da

        full = da.zeros((4, 512, 512), dtype="uint8")
        client = MagicMock()
        client.get_tensor.return_value = full
        connection.client = client
        patch_viewer_tensor_methods(viewer, connection)

        layer = self._loaded_layer(self._pyramid(), array_id="src/t1")

        assert viewer.tensor(layer) is full
        client.get_tensor.assert_called_once_with("src/t1")

    def test_a_name_resolves_against_the_viewer(self, viewer, connection):
        import dask.array as da

        client = MagicMock()
        client.get_tensor.return_value = da.zeros((2, 2), dtype="uint8")
        connection.client = client
        layer = self._loaded_layer(self._pyramid())
        viewer.layers = {"big": layer}
        patch_viewer_tensor_methods(viewer, connection)

        viewer.tensor("big")

        client.get_tensor.assert_called_once_with("t1")

    def test_it_is_not_level_0_of_the_advertised_pyramid(self, viewer, connection):
        """The reason this goes to the server rather than unwrapping: level 0 is
        whatever the server *advertised*, which is full resolution by
        convention and not by contract (a native OME-Zarr pyramid whose first
        dataset carries no integer scale is skipped when the levels are built).
        ``get_tensor`` is full resolution by definition."""
        import dask.array as da

        levels = self._pyramid()
        full = da.zeros((4, 1024, 1024), dtype="uint8")  # what the tensor really is
        client = MagicMock()
        client.get_tensor.return_value = full
        connection.client = client
        patch_viewer_tensor_methods(viewer, connection)

        arr = viewer.tensor(self._loaded_layer(levels))

        assert arr.shape == (4, 1024, 1024)
        assert arr is not levels[0]

    def test_a_layer_the_agent_built_is_unwrapped_not_fetched(self, viewer, connection):
        import dask.array as da

        client = MagicMock()
        connection.client = client
        patch_viewer_tensor_methods(viewer, connection)

        own = da.zeros((4, 64, 64), dtype="uint8")
        layer = self._loaded_layer([own], array_id=None)

        arr = viewer.tensor(layer)

        assert arr is own
        assert isinstance(arr, da.Array), "the proxy was handed back unwrapped"
        client.get_tensor.assert_not_called()

    def test_an_unattributed_pyramid_unwraps_level_0(self, viewer, connection):
        """No ``array_id`` and multiscale: ``data[0]`` is the level, and the
        proxy around it still has to come off."""
        import dask.array as da

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        levels = self._pyramid()

        arr = viewer.tensor(self._loaded_layer(levels, array_id=None))

        assert arr is levels[0]
        assert isinstance(arr, da.Array)

    def test_a_numpy_layer_comes_back_as_it_went_in(self, viewer, connection):
        import napari
        import numpy as np

        connection.client = MagicMock()
        patch_viewer_tensor_methods(viewer, connection)
        own = np.zeros((8, 8), dtype=np.int32)

        arr = viewer.tensor(napari.layers.Labels(own))

        assert arr is own

    def test_a_failed_server_read_falls_back_to_the_layer(self, viewer, connection):
        """The pixels on screen are still readable; raising here would fail a
        read the layer can satisfy on its own."""
        import dask.array as da

        client = MagicMock()
        client.get_tensor.side_effect = RuntimeError("source is gone")
        connection.client = client
        patch_viewer_tensor_methods(viewer, connection)
        levels = self._pyramid()

        arr = viewer.tensor(self._loaded_layer(levels))

        assert arr is levels[0]
        assert isinstance(arr, da.Array)

    def test_no_server_still_reads_the_layer(self, viewer, connection):
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
        """The failure this accessor exists to make unreachable: napari's
        ``MultiScaleData.__array__`` returns the **lowest** level, so handing
        ``layer.data`` to numpy or scikit-image silently computes on the bottom
        of the pyramid -- no error, no signal, wrong resolution
        (biopb/biopb#973)."""
        import dask.array as da
        import numpy as np

        levels = self._pyramid()
        full = da.zeros((4, 512, 512), dtype="uint8")
        client = MagicMock()
        client.get_tensor.return_value = full
        connection.client = client
        patch_viewer_tensor_methods(viewer, connection)
        layer = self._loaded_layer(levels)

        assert np.asarray(layer.data).shape == (4, 128, 128)  # the trap
        assert viewer.tensor(layer).shape == (4, 512, 512)  # the way out

    def test_it_survives_the_agent_facing_viewer_proxy(self, connection):
        """The agent never holds the real viewer -- it holds the main-thread
        marshaling proxy (``_viewer_proxy``), which forwards method calls and
        unwraps proxied arguments. A layer handle taken off that proxy has to
        arrive here as a napari layer, and the dask array has to come back out
        un-proxied (it is inert, so nothing should wrap it).

        Headless ``ViewerModel`` rather than ``napari.Viewer`` for the reason
        ``test_viewer_proxy`` gives: the real viewer's GL canvas segfaults on
        offscreen runners, and nothing here needs a canvas.
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
        full = da.zeros((4, 512, 512), dtype="uint8")
        connection.client = MagicMock()
        connection.client.get_tensor.return_value = full
        patch_viewer_tensor_methods(real, connection)

        proxy = make_viewer_proxy(real)

        assert proxy.tensor(proxy.layers["big"]) is full
        assert proxy.tensor("big") is full
