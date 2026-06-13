"""Pin the napari viewer's lazy layer arrays to a single-process scheduler.

The MCP bootstrap registers a distributed ``LocalCluster`` as dask's *default*
scheduler so the agent's heavy ``da`` computes run in parallel. The viewer,
however, scrubs planes **one at a time** (serial ``np.asarray(data[slices])``),
so computing those slices on the cluster buys zero parallelism while scattering
each single-chunk fetch across a rotating worker — the per-worker chunk cache is
an opaque side-effect dask's locality scheduler can't see, so same-chunk reads
miss and replicate (issue #8).

``wrap_levels`` wraps each layer array in a :class:`_ViewerArray` proxy that
forces the *implicit* materialization napari performs (``np.asarray`` ->
``__array__``) onto a single-process scheduler (``threads`` by default), so all
slice reads run in the kernel main process against the one shared
``conn.client`` chunk cache: ~100% hit on revisit, 1x memory, no scatter.

Only ``__array__`` (implicit ``np.asarray`` coercion) and ``__getitem__`` (the
slice it coerces) are pinned to the single-process scheduler. Everything else
delegates to the underlying dask array and stays lazy on the default
(distributed) scheduler: attribute/method access (``.compute()``, ``.mean()``,
``.rechunk()``) via ``__getattr__``, and operators / comparisons / NumPy ufuncs
(``data + 1``, ``data > 0``, ``np.add(data, 1)``) via ``__array_ufunc__``, which
returns a plain dask array. So the agent's explicit computes on a layer's
``.data`` still use the cluster. The viewer being serial is exactly why pinning
its reads is free.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class _ViewerArray(NDArrayOperatorsMixin):
    """Array-like proxy that computes implicit materializations in-process.

    Wraps a lazy dask array. ``__array__`` (what ``np.asarray`` calls, the path
    napari uses to realize a slice) computes with the configured single-process
    *scheduler*; ``__getitem__`` re-wraps so the sliced array napari materializes
    stays pinned.

    Everything else behaves like the underlying dask array: ``__getattr__``
    delegates attribute/method access, and ``NDArrayOperatorsMixin`` +
    ``__array_ufunc__`` forward operators / comparisons / ufuncs to it lazily
    (returning a plain dask array, *not* a re-wrapped proxy) — Python looks up
    operator dunders on the type, so ``__getattr__`` alone cannot delegate them.
    Like a dask array, this proxy is unhashable (it has an elementwise
    ``__eq__``); napari never hashes layer ``.data``.
    """

    __slots__ = ("_arr", "_scheduler")

    def __init__(self, arr, scheduler: str = "threads"):
        self._arr = arr
        self._scheduler = scheduler

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Forward operators/comparisons/ufuncs (data + 1, data > 0, np.add(...))
        # to the underlying dask array: unwrap any _ViewerArray operand, then let
        # NumPy dispatch to dask's own __array_ufunc__. The result is a plain
        # lazy dask array on the default (distributed) scheduler -- not re-wrapped
        # -- so derived agent expressions are not pinned to the viewer scheduler.
        inputs = tuple(
            x._arr if isinstance(x, _ViewerArray) else x for x in inputs
        )
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(
                x._arr if isinstance(x, _ViewerArray) else x for x in out
            )
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __array__(self, dtype=None, copy=None):
        # napari realizes a slice via ``np.asarray(data[slices])``; force that
        # compute onto the single-process scheduler so it hits the main-process
        # chunk cache instead of scattering across cluster workers. ``copy`` is
        # accepted for the NumPy 2.0 protocol; ``compute`` already returns a
        # fresh array, so it is a no-op here.
        result = self._arr.compute(scheduler=self._scheduler)
        return np.asarray(result, dtype=dtype)

    def __getitem__(self, idx):
        return _ViewerArray(self._arr[idx], self._scheduler)

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def size(self):
        # napari's LayerDataProtocol (a runtime_checkable Protocol) lists
        # ``size`` as a required non-callable member, and its isinstance check
        # resolves members with ``inspect.getattr_static`` -- which bypasses
        # ``__getattr__``. So ``size`` must exist as a real class attribute,
        # not be delegated below, or napari rejects the proxy with
        # "does not implement 'LayerDataProtocol'" and add_image() fails.
        return self._arr.size

    def __len__(self):
        return len(self._arr)

    def __getattr__(self, name):
        # Delegate everything else (.compute, .mean, .rechunk, ...) to the
        # underlying dask array, so explicit agent computes use the global
        # (distributed) default scheduler. __getattr__ only fires for names not
        # found normally, so the pinned __array__/__getitem__ above always win.
        return getattr(self._arr, name)

    def __repr__(self):
        return f"_ViewerArray({self._arr!r}, scheduler={self._scheduler!r})"


def wrap_levels(levels, scheduler: str | None):
    """Wrap viewer layer array(s) so their slice reads compute in-process.

    *levels* is what the load paths pass to ``viewer.add_image``: a single array
    or a pyramid list of arrays. Returns the same shape with each array wrapped
    in a :class:`_ViewerArray`. A falsy *scheduler* (e.g. ``None`` in the
    standalone napari plugin, where there is no distributed default) returns
    *levels* unchanged.
    """
    if not scheduler:
        return levels
    if isinstance(levels, list | tuple):
        return type(levels)(_ViewerArray(a, scheduler) for a in levels)
    return _ViewerArray(levels, scheduler)
