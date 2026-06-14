"""Main-thread marshaling proxy for the agent-facing ``viewer``.

``execute_code`` runs agent code on a background daemon thread (see ``_jobs``),
but napari/Qt objects are **main-thread-only**: any viewer mutation from the
worker that synchronously emits a napari event into a Qt slot touches a Qt widget
off-thread and *segfaults the whole kernel* (e.g. ``viewer.layers.clear()`` ->
``QtDims._resize_slice_labels``; biopb/biopb#100).

The old mitigation wrapped only the ``add_*`` methods, which leaks: any call that
*returns* a live napari sub-object (``viewer.layers``, ``viewer.dims``,
``viewer.layers[0]``) hands the worker an unguarded handle. This module closes
that by putting a **transparent proxy** in the namespace as ``viewer``: the real
``napari.Viewer`` is untouched (napari/Qt keep their direct references); the agent
only ever holds a proxy that

  * marshals mutations (``__setattr__``) and method calls to the Qt main thread
    via :func:`_jobs.run_on_main` (a no-op when already on the main thread);
  * **re-wraps** any napari handle it returns, so handles never leak unwrapped;
  * passes inert values (arrays, scalars, ...) straight through (field *reads*
    are not marshaled -- the mutations-only policy);
  * **fail-loud**: returns a guard for raw-Qt objects (``viewer.window``) that
    raises :class:`ViewerThreadError` on any off-main access rather than handing
    back a handle that will segfault.

Completeness of the handle set is enforced by a graph-walk test (see
``_tests/test_viewer_proxy.py``), not assumed: a future napari that returns an
unregistered handle type fails CI instead of segfaulting at runtime. See
``docs/viewer-thread-safety.md``.
"""

import functools
import threading
import weakref

from napari.layers import Layer
from napari.utils.events import EventedModel
from napari.utils.events.containers import EventedList, Selection

from ._jobs import run_on_main

# Top-level package names whose objects are Qt-affine and must never be handed to
# a worker thread unguarded. ``napari._qt`` (e.g. the QtViewer behind
# ``viewer.window``) is matched separately by prefix.
_QT_TOPLEVEL = frozenset(
    {"PyQt6", "PyQt5", "PySide6", "PySide2", "qtpy", "vispy"}
)


class ViewerThreadError(RuntimeError):
    """Raised when a raw-Qt object is accessed off the Qt main thread.

    The fail-loud alternative to a segfault: wrap the access in
    ``run_on_main(...)`` (also injected into the kernel namespace).
    """


def _on_main() -> bool:
    return threading.current_thread() is threading.main_thread()


def _is_proxy(value) -> bool:
    # ``type()`` reads the real class (ignoring the ``__class__`` spoof below), so
    # this reliably detects our proxies even though ``isinstance`` would not.
    return issubclass(type(value), _ProxyBase)


def _unwrap(value):
    return (
        object.__getattribute__(value, "_real") if _is_proxy(value) else value
    )


def _marshal_call(bound):
    """Return a callable that runs ``bound`` on the main thread and wraps the
    result. Proxy arguments are unwrapped so napari sees its own objects."""

    def caller(*args, **kwargs):
        args = tuple(_unwrap(a) for a in args)
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return wrap(run_on_main(bound, *args, **kwargs))

    try:
        return functools.wraps(bound)(caller)
    except (AttributeError, TypeError):
        return caller


class _ProxyBase:
    """Common transparent-proxy machinery. Holds the real object in a slot and
    forwards identity/repr so the proxy is hard to tell from the real thing."""

    __slots__ = ("_real", "__weakref__")

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    # isinstance(proxy, napari.layers.Image) etc. still work; ``type(proxy)``
    # still returns the proxy class (so _is_proxy stays reliable).
    @property
    def __class__(self):  # noqa: A003
        return type(object.__getattribute__(self, "_real"))

    def __repr__(self):
        return repr(object.__getattribute__(self, "_real"))

    def __dir__(self):
        return dir(object.__getattribute__(self, "_real"))

    def __eq__(self, other):
        return object.__getattribute__(self, "_real") == _unwrap(other)

    def __ne__(self, other):
        return object.__getattribute__(self, "_real") != _unwrap(other)

    def __hash__(self):
        return hash(object.__getattribute__(self, "_real"))


class _HandleProxy(_ProxyBase):
    """Proxy for evented models (Viewer/Dims/Camera/Cursor/GridCanvas/Tooltip)
    and Layers: marshal mutations + method calls, re-wrap returned handles."""

    __slots__ = ()

    def __getattr__(self, name):
        # Only reached for names not resolved on the proxy class itself.
        real = object.__getattribute__(self, "_real")
        attr = getattr(real, name)
        if callable(attr) and not isinstance(attr, type):
            return _marshal_call(attr)
        return wrap(attr)

    def __setattr__(self, name, value):
        real = object.__getattribute__(self, "_real")
        run_on_main(setattr, real, name, _unwrap(value))

    def __delattr__(self, name):
        real = object.__getattribute__(self, "_real")
        run_on_main(delattr, real, name)


class _ContainerProxy(_HandleProxy):
    """Proxy for napari evented containers (LayerList, the layers Selection).

    List/set mutating *methods* (append/remove/clear/...) flow through
    ``_HandleProxy.__getattr__`` and are marshaled; the dunders below must be
    defined explicitly because Python resolves them on the type, not via
    ``__getattr__``."""

    __slots__ = ()

    def __len__(self):
        return len(object.__getattribute__(self, "_real"))

    def __iter__(self):
        return (wrap(x) for x in object.__getattribute__(self, "_real"))

    def __contains__(self, value):
        return _unwrap(value) in object.__getattribute__(self, "_real")

    def __getitem__(self, key):
        real = object.__getattribute__(self, "_real")
        result = real[key]
        if isinstance(key, slice):
            return [wrap(x) for x in result]
        return wrap(result)

    def __setitem__(self, key, value):
        real = object.__getattribute__(self, "_real")
        if isinstance(value, (list, tuple)):
            value = type(value)(_unwrap(v) for v in value)
        else:
            value = _unwrap(value)
        run_on_main(real.__setitem__, key, value)

    def __delitem__(self, key):
        real = object.__getattribute__(self, "_real")
        run_on_main(real.__delitem__, key)


class _GuardProxy(_ProxyBase):
    """Fail-loud guard for raw-Qt objects (``viewer.window`` and below).

    Usable on the main thread (e.g. inside ``run_on_main``); any off-main access
    raises instead of returning a handle that would segfault."""

    __slots__ = ()

    def _guard(self):
        if not _on_main():
            real = object.__getattribute__(self, "_real")
            raise ViewerThreadError(
                f"{type(real).__module__}.{type(real).__qualname__} is a Qt "
                "object and can only be touched on the Qt main thread. Wrap the "
                "access in run_on_main(lambda: ...)."
            )

    def __getattr__(self, name):
        self._guard()
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        self._guard()
        setattr(object.__getattribute__(self, "_real"), name, value)

    def __call__(self, *args, **kwargs):
        self._guard()
        return object.__getattribute__(self, "_real")(*args, **kwargs)


def _proxy_cls(obj):
    """The proxy class for *obj*, or ``None`` if it is safe to pass through
    unwrapped (inert Python/numpy data, not Qt-affine)."""
    if isinstance(obj, (EventedList, Selection)):
        return _ContainerProxy
    if isinstance(obj, (EventedModel, Layer)):
        return _HandleProxy
    module = type(obj).__module__ or ""
    if module.split(".", 1)[0] in _QT_TOPLEVEL or module.startswith(
        "napari._qt"
    ):
        return _GuardProxy
    return None


# id(real) -> proxy, so repeated access returns the same proxy
# (``viewer.layers[0] is viewer.layers[0]``). The proxy strongly refs its real
# object, so while the proxy is alive id(real) stays valid; the entry is weak so
# it drops when the proxy is collected.
_CACHE: "weakref.WeakValueDictionary[int, _ProxyBase]" = (
    weakref.WeakValueDictionary()
)


def wrap(obj):
    """Wrap *obj* in the appropriate marshaling proxy, or return it unchanged if
    it is inert (non-Qt) data. Idempotent and identity-stable."""
    if _is_proxy(obj):
        return obj
    cls = _proxy_cls(obj)
    if cls is None:
        return obj
    cached = _CACHE.get(id(obj))
    if cached is not None and object.__getattribute__(cached, "_real") is obj:
        return cached
    proxy = cls(obj)
    try:
        _CACHE[id(obj)] = proxy
    except TypeError:  # unhashable id key can't happen, but be safe
        pass
    return proxy


def make_viewer_proxy(viewer):
    """Public entry: the agent-facing, thread-safe handle for *viewer*."""
    return wrap(viewer)
