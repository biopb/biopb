"""Helper functions injected into the execute_code namespace.

``add_tensor`` is monkey-patched onto the viewer instance so the agent
calls ``viewer.add_tensor("source_id")``.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Origin scheme the tensor server stamps on a drag-dropped source's source_url
# (server-side DND_URL_PREFIX). Display-only; stripped here so a derived layer
# name matches what the tensor browser shows. Keep in sync with the server
# constant and the tree builders' strip (_widget.py, web SourceTree.tsx).
_DND_URL_PREFIX = "dnd://"


def _get_url_stem(url: str) -> str:
    """Extract last path component from a URL.

    A drag-dropped source's ``dnd://`` origin scheme is stripped first -- it puts
    the re-rooted basename in the netloc rather than the path, so ``urlparse``
    would otherwise yield an empty path and fall back to the raw ``dnd://...``
    string as the "stem."
    """
    try:
        if url.startswith(_DND_URL_PREFIX):
            path = url[len(_DND_URL_PREFIX) :].replace("\\", "/")
        else:
            path = urlparse(url).path
        parts = [p for p in path.split("/") if p]
        return parts[-1] if parts else url
    except Exception:
        return url


def _tensor_short_name(array_id: str) -> str:
    parts = [p for p in array_id.split("/") if p]
    return parts[-1] if parts else array_id


def viewer_window_alive(viewer) -> bool:
    """True only if the napari Qt window's C++ object is still alive.

    Detects a user-closed / destroyed window so the tools stop reporting a
    half-dead viewer as healthy: after the user closes the window the Python
    ``viewer`` survives (the kernel namespace holds it), but its Qt window and
    vispy canvas are destroyed, so mutations silently no-op and ``screenshot``
    raises.  Covers all three teardown shapes: a user X-close (``_qt_window``
    wraps a deleted C++ object, so ``isVisible()`` raises ``RuntimeError``), a
    programmatic ``Window.close()`` (``del self._qt_window`` -> attribute gone),
    and the headless sentinel (``__getattr__`` raises).  An alive-but-minimized
    or hidden window is still "alive" (``isVisible()`` returns a bool rather than
    raising)."""
    try:
        window = getattr(viewer, "window", None)
        if window is None:
            return False
        qt_window = getattr(window, "_qt_window", None)
        if qt_window is None:  # programmatic close did `del self._qt_window`
            return False
        qt_window.isVisible()  # raises RuntimeError if the C++ object is deleted
        return True
    except Exception:
        return False


def resync_view_for_capture(viewer, timeout: float = 30.0) -> None:
    """Wait for the current view's slice to load before a screenshot.

    With async slicing on (``mcp.viewer.async_slicing``) a dims/zoom change
    fetches the new slice *off* the Qt main thread, so a screenshot taken right
    after could capture the previous (pre-load) frame -- fine for an interactive
    human, wrong for the agent, which expects ``take_screenshot`` to reflect the
    state it just set. This (1) submits a fresh slice of the current view so a
    slice is guaranteed in flight even if the triggering event hasn't dispatched
    yet, then (2) pumps the Qt event loop until every layer reports ``loaded``
    -- so the slice completes and is applied -- before the caller captures.

    The re-submit reads the current view's chunk a second time, but that read
    is a cache hit (the in-flight slice already warmed the server/client chunk
    cache), so it is cheap; the guarantee it buys -- no stale capture when the
    view changed but the slice wasn't requested yet -- is worth it. The pump,
    not napari's bare ``submit``, is what actually *awaits* completion:
    ``submit`` returns the async future without blocking, so ``layer.loaded`` +
    ``processEvents`` is the await. Runs on the kernel main thread (where the
    screenshot snippet already runs) so pumping lets the slice's main-thread
    apply callback fire.

    No-op when async slicing is off (synchronous slicing already left the view
    loaded) or when there are no layers. Bounded by *timeout* so a genuinely
    slow/stuck read degrades to a best-effort capture rather than hanging.
    Best-effort: never raises."""
    try:
        import napari

        if not napari.settings.get_settings().experimental.async_:
            return
    except Exception:
        return
    try:
        import time

        from qtpy.QtWidgets import QApplication

        layers = list(getattr(viewer, "layers", None) or [])
        if not layers:
            return
        # Guarantee a slice of the *current* view is in flight (closes the
        # change->dispatch race); the extra read is a cache hit.
        slicer = getattr(viewer, "_layer_slicer", None)
        if slicer is not None:
            try:
                slicer.submit(layers=layers, dims=viewer.dims, force=True)
            except Exception:
                logger.debug("resync submit failed", exc_info=True)
        # Await: pump the loop until the slice completes and is applied.
        deadline = time.monotonic() + timeout
        while not all(getattr(layer, "loaded", True) for layer in layers):
            QApplication.processEvents()
            if time.monotonic() > deadline:
                break
            time.sleep(0.005)
    except Exception:
        logger.debug("resync_view_for_capture failed", exc_info=True)


def patch_viewer_add_tensor(viewer, connection, compute_scheduler=None):
    """Monkey-patch ``add_tensor`` onto *viewer*, reading client/sources from
    the live ``TensorConnection`` *connection*.

    *compute_scheduler*, when set, pins the loaded layer's slice reads to a
    single-process dask scheduler (see ``_viewer_compute.wrap_levels``) so the
    serial viewer hits the shared main-process chunk cache instead of scattering
    across the distributed cluster (issue #8)."""

    def add_tensor(
        source_id: str,
        tensor_id: str | None = None,
        name: str | None = None,
    ) -> str:
        """Add a tensor dataset to the viewer as an image layer.

        Args:
            source_id: Source identifier on the tensor server.
            tensor_id: Tensor array_id within the source.  If *None* and the
                source has exactly one tensor, it is selected automatically.
            name: Layer name.  Auto-generated from source URL if *None*.

        Returns:
            The name of the created viewer layer.
        """
        from .._tensor_utils import add_tensor_layer

        client = connection.client
        if client is None:
            raise RuntimeError(
                "No tensor server connected. "
                "Open the Tensor Browser widget and connect first."
            )

        sources = connection.sources or {}
        src = sources.get(source_id)
        if src is None:
            # Not in the (possibly truncated) cached catalog — fetch the
            # descriptor directly from the server.  Requires
            # TensorFlightClient.get_source (added to biopb separately); until
            # that ships this stays a no-op and we raise as before.
            get_source = getattr(client, "get_source", None)
            if get_source is not None:
                src = get_source(source_id, tensor_id)
            else:
                raise ValueError(
                    f"Source '{source_id}' not found. "
                    f"Available: {list(sources.keys())[:20]}"
                )

        if tensor_id is None:
            if len(src.tensors) == 1 and src.tensors[0]:
                tensor_id = src.tensors[0].array_id
            else:
                ids = [t.array_id for t in src.tensors]
                raise ValueError(
                    f"Source has {len(src.tensors)} tensors — "
                    f"specify tensor_id. Available: {ids}"
                )

        tensor_desc = next((t for t in src.tensors if t.array_id == tensor_id), None)
        if tensor_desc is None:
            raise ValueError(f"Tensor '{tensor_id}' not found in source '{source_id}'")

        if name is None:
            stem = _get_url_stem(src.source_url) or source_id
            if len(src.tensors) > 1:
                name = f"{stem}/{_tensor_short_name(tensor_id)}"
            else:
                name = stem

        # Shared build-pyramid -> wrap -> OME scale -> add_image pipeline
        # (also used by the Tensor Browser widget).
        add_tensor_layer(
            viewer,
            client,
            source_id,
            tensor_id,
            tensor_desc,
            name=name,
            source_desc=src,
            compute_scheduler=compute_scheduler,
        )

        return name

    # napari.Viewer is a pydantic evented model with validate_assignment, so a
    # plain ``viewer.add_tensor = ...`` is rejected.  Write through to the
    # instance dict to bypass field validation.
    object.__setattr__(viewer, "add_tensor", add_tensor)
