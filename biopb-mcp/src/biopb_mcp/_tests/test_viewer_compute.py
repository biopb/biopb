"""Tests for the viewer slice-read scheduler pinning (issue #8)."""

from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import pytest

from biopb_mcp._viewer_compute import _ViewerArray, wrap_levels


@pytest.fixture
def base():
    """A small lazy dask array with a couple of chunks."""
    return da.from_array(np.arange(24).reshape(4, 6), chunks=(2, 3))


def test_array_protocol_attributes_delegate(base):
    w = _ViewerArray(base, "threads")
    assert w.shape == base.shape
    assert w.dtype == base.dtype
    assert w.ndim == base.ndim
    assert w.size == base.size
    assert len(w) == len(base)


def test_satisfies_napari_layer_data_protocol(base):
    """The proxy must pass napari's runtime_checkable LayerDataProtocol.

    Regression: napari's isinstance check resolves the protocol's required
    members (shape/dtype/ndim/size/__getitem__) with inspect.getattr_static,
    which bypasses __getattr__. A member delegated only via __getattr__ (as
    ``size`` once was) is invisible to that check, so add_image() rejects the
    proxy ("does not implement 'LayerDataProtocol'") and tensor loads fail.
    """
    import inspect

    protocols = pytest.importorskip("napari.layers._data_protocols")
    proto = protocols.LayerDataProtocol

    w = _ViewerArray(base, "threads")
    # The members napari's LayerDataProtocol requires. Each must resolve
    # *statically* (no __getattr__) -- that is how the protocol's isinstance
    # check (inspect.getattr_static, Py 3.12) sees them.
    _MISSING = object()
    for attr in ("shape", "dtype", "ndim", "size", "__getitem__"):
        assert inspect.getattr_static(w, attr, _MISSING) is not _MISSING, attr
    assert isinstance(w, proto)


def test_asarray_materializes_correctly(base):
    w = _ViewerArray(base, "synchronous")
    np.testing.assert_array_equal(np.asarray(w), np.asarray(base))


def test_array_pins_scheduler():
    """__array__ must compute with the configured scheduler, not the default."""
    stub = MagicMock()
    stub.compute.return_value = np.zeros((2, 2), dtype="uint8")
    w = _ViewerArray(stub, "threads")

    np.asarray(w)

    stub.compute.assert_called_once_with(scheduler="threads")


def test_getitem_rewraps_and_pins(base):
    """A sliced viewer array stays pinned (napari computes np.asarray(data[s]))."""
    w = _ViewerArray(base, "synchronous")
    sub = w[1:3, 0:2]
    assert isinstance(sub, _ViewerArray)
    assert sub._scheduler == "synchronous"
    np.testing.assert_array_equal(np.asarray(sub), np.asarray(base[1:3, 0:2]))


def test_getattr_delegates_to_dask(base):
    """Explicit dask ops delegate to the underlying array (cluster default)."""
    w = _ViewerArray(base, "threads")
    # .mean() returns a real dask array operating on the underlying array, so an
    # agent's explicit compute uses the global (distributed) default scheduler.
    reduced = w.mean()
    assert isinstance(reduced, da.Array)
    assert float(reduced.compute()) == float(base.mean().compute())


@pytest.mark.parametrize(
    "op, expected",
    [
        (lambda w: w + 1, lambda b: b + 1),
        (lambda w: 1 + w, lambda b: 1 + b),
        (lambda w: w > 5, lambda b: b > 5),
        (lambda w: -w, lambda b: -b),
        (lambda w: np.add(w, 1), lambda b: np.add(b, 1)),
        (lambda w: np.greater(w, 5), lambda b: np.greater(b, 5)),
    ],
)
def test_operators_forward_lazily_to_dask(base, op, expected):
    """Operators/comparisons/ufuncs delegate to the dask array and stay lazy.

    __getattr__ cannot delegate operator dunders (Python looks them up on the
    type), so this exercises NDArrayOperatorsMixin + __array_ufunc__.
    """
    w = _ViewerArray(base, "threads")
    result = op(w)
    # Plain lazy dask array (not a re-wrapped proxy, not eagerly computed).
    assert isinstance(result, da.Array)
    assert not isinstance(result, _ViewerArray)
    np.testing.assert_array_equal(result.compute(), expected(base).compute())


def test_operators_never_trigger_compute(base):
    """An operator must not invoke __array__ (no in-process materialization)."""

    class _NoCompute(_ViewerArray):
        def __array__(self, dtype=None, copy=None):
            raise AssertionError("operator triggered __array__/compute")

    w = _NoCompute(base, "threads")
    # Would raise if the operator path went through __array__ instead of dask.
    assert isinstance(w + 1, da.Array)
    assert isinstance(w > 5, da.Array)


def test_unhashable_like_dask(base):
    """Mirrors dask arrays (elementwise __eq__ -> unhashable); napari never
    hashes layer .data."""
    w = _ViewerArray(base, "threads")
    with pytest.raises(TypeError):
        hash(w)


def test_wrap_levels_single_array(base):
    wrapped = wrap_levels(base, "threads")
    assert isinstance(wrapped, _ViewerArray)


def test_wrap_levels_pyramid_list(base):
    levels = [base, base[::2, ::2]]
    wrapped = wrap_levels(levels, "threads")
    assert isinstance(wrapped, list)
    assert all(isinstance(a, _ViewerArray) for a in wrapped)
    assert len(wrapped) == 2


def test_wrap_levels_none_passthrough(base):
    """Standalone plugin (scheduler None) leaves arrays untouched."""
    levels = [base]
    assert wrap_levels(levels, None) is levels
    assert wrap_levels(base, "") is base
