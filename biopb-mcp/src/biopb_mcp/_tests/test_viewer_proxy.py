"""Tests for the main-thread marshaling viewer proxy (_viewer_proxy).

Covers the proxy's transparency/identity, the mutations-only read policy, that
mutations from a *background thread* are marshaled to the Qt main thread (the
biopb/biopb#100 crash class), the fail-loud guard on raw-Qt objects, and a
graph-walk tripwire that no napari handle leaks through the proxy unwrapped.

A real napari.Viewer is used (offscreen Qt). To exercise marshaling we run a
mutation on a worker thread while the (main) test thread pumps the Qt event loop
-- exactly the kernel's job-thread vs. main-thread arrangement.
"""

import threading
import time

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from biopb_mcp.mcp._viewer_proxy import (
    ViewerThreadError,
    _is_proxy,
    _unwrap,
    make_viewer_proxy,
)


@pytest.fixture
def viewer(qtbot):
    import napari

    v = napari.Viewer(show=False)
    yield v
    v.close()


@pytest.fixture
def proxy(viewer):
    return make_viewer_proxy(viewer)


def _run_in_worker(fn, timeout=10.0):
    """Run ``fn`` on a daemon thread while pumping the Qt loop here (main).

    Returns fn's result, or re-raises its exception. Raises TimeoutError if the
    worker never finishes -- which is what a marshaling deadlock would look like.
    """
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - relay to the test
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while t.is_alive():
        QApplication.processEvents()
        time.sleep(0.005)
        if time.monotonic() > deadline:
            raise TimeoutError("worker did not finish (marshaling deadlock?)")
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# -- transparency / identity ------------------------------------------------


def test_isinstance_transparency(proxy, viewer):
    import napari
    from napari.components.layerlist import LayerList

    assert isinstance(proxy, napari.components.ViewerModel)
    assert isinstance(proxy.layers, LayerList)
    assert isinstance(proxy.dims, type(viewer.dims))
    # type() still reveals the proxy, so internal detection stays reliable.
    assert _is_proxy(proxy)
    assert _is_proxy(proxy.layers)
    assert _is_proxy(proxy.dims)


def test_layer_isinstance(proxy):
    proxy.add_image(np.zeros((8, 8), dtype=np.uint8))
    import napari.layers

    layer = proxy.layers[0]
    assert _is_proxy(layer)
    assert isinstance(layer, napari.layers.Image)


def test_identity_is_stable(proxy):
    proxy.add_image(np.zeros((4, 4)))
    assert proxy.layers is proxy.layers
    assert proxy.dims is proxy.dims
    assert proxy.layers[0] is proxy.layers[0]


def test_reads_pass_through(proxy):
    proxy.add_image(np.zeros((4, 5), dtype=np.float32))
    # plain field reads are not wrapped...
    assert isinstance(proxy.dims.ndim, int)
    assert not _is_proxy(proxy.dims.ndim)
    # ...and array data comes back as a real array, not a proxy.
    data = proxy.layers[0].data
    assert isinstance(data, np.ndarray)
    assert not _is_proxy(data)


# -- marshaling from a worker thread (the #100 crash class) ------------------


def test_layers_clear_from_worker(proxy, viewer):
    """The exact biopb/biopb#100 repro: layers.clear() off the main thread.

    Previously this ran the layer-removal cascade -> QtDims widget mutation on
    the worker thread and segfaulted the process. Now it must be marshaled.
    """
    proxy.add_image(np.zeros((8, 8)))
    proxy.add_image(np.zeros((8, 8)))
    assert len(viewer.layers) == 2

    _run_in_worker(proxy.layers.clear)
    assert len(viewer.layers) == 0


def test_assorted_mutations_from_worker(proxy, viewer):
    # add_image from a worker returns a wrapped layer
    layer = _run_in_worker(lambda: proxy.add_image(np.zeros((6, 6), np.uint8)))
    assert _is_proxy(layer)
    assert len(viewer.layers) == 1

    # attribute mutation (the kind the add_*-only wrap missed)
    _run_in_worker(lambda: setattr(layer, "opacity", 0.25))
    assert viewer.layers[0].opacity == pytest.approx(0.25)

    # data replacement
    new = np.ones((10, 10), np.uint8)
    _run_in_worker(lambda: setattr(layer, "data", new))
    assert viewer.layers[0].data.shape == (10, 10)

    # dims mutation (scrubbing)
    proxy.add_image(np.zeros((3, 6, 6)))
    _run_in_worker(lambda: proxy.dims.set_current_step(0, 2))
    assert viewer.dims.current_step[0] == 2

    # a layer-type-specific mutating method, reached via __getattr__
    pts = _run_in_worker(lambda: proxy.add_points(np.array([[1.0, 1.0]])))
    _run_in_worker(lambda: pts.add(np.array([[2.0, 2.0]])))
    assert len(viewer.layers[-1].data) == 2

    # del viewer.layers[i]
    _run_in_worker(lambda: proxy.layers.__delitem__(0))
    assert len(viewer.layers) == 2


def test_container_iteration_wraps(proxy):
    proxy.add_image(np.zeros((4, 4)))
    proxy.add_image(np.zeros((4, 4)))
    assert all(_is_proxy(layer) for layer in proxy.layers)
    assert all(_is_proxy(layer) for layer in proxy.layers[:])


# -- fail-loud guard on raw Qt ----------------------------------------------


def test_window_guard_off_thread_raises(proxy):
    # Reaching into the raw Qt widget tree from a worker must raise, not segfault.
    with pytest.raises(ViewerThreadError):
        _run_in_worker(lambda: proxy.window.qt_viewer)


def test_window_accessible_on_main(proxy):
    # On the main thread the guard passes through (e.g. inside run_on_main).
    assert proxy.window is not None
    _ = proxy.window.qt_viewer  # no raise on the main thread


# -- tripwire: no napari handle leaks through the proxy ----------------------


def test_no_handle_leaks_through_proxy(proxy):
    """Walk the documented graph *through the proxy* and assert every reachable
    napari handle (EventedModel / Layer / evented container) is wrapped.

    Guards against a re-wrap gap (e.g. a forgotten container dunder) and against
    a future napari sub-object the proxy fails to wrap. Add a representative of
    each Layer subclass so the layer hierarchy is exercised.
    """
    from napari.layers import Layer
    from napari.utils.events import EventedModel
    from napari.utils.events.containers import EventedList, Selection

    proxy.add_image(np.zeros((4, 4), np.uint8))
    proxy.add_labels(np.zeros((4, 4), np.int32))
    proxy.add_points(np.array([[1.0, 1.0]]))
    proxy.add_shapes(
        [np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])]
    )
    proxy.add_vectors(np.zeros((2, 2, 2)))

    HANDLE = (EventedModel, Layer, EventedList, Selection)
    seen: set[int] = set()
    leaks: list[str] = []

    def visit(value, path):
        real = _unwrap(value)
        if isinstance(real, HANDLE) and not _is_proxy(value):
            leaks.append(f"{path}: unwrapped {type(real).__qualname__}")
            return
        if id(real) in seen:
            return
        seen.add(id(real))
        if not _is_proxy(value):
            return  # inert leaf -- nothing to recurse into
        for name in getattr(type(real), "model_fields", {}):
            try:
                child = getattr(value, name)
            except Exception:  # noqa: BLE001 - some fields are not always live
                continue
            visit(child, f"{path}.{name}")
        if isinstance(real, (EventedList, Selection)):
            for i in range(len(real)):
                visit(value[i], f"{path}[{i}]")

    visit(proxy, "viewer")
    assert not leaks, "napari handles leaked unwrapped:\n  " + "\n  ".join(
        leaks
    )
