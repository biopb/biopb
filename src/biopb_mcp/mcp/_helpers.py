"""Helper functions injected into the execute_code namespace.

``load_tensor`` is monkey-patched onto the viewer instance so the agent
calls ``viewer.load_tensor("source_id")``.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _get_url_stem(url: str) -> str:
    """Extract last path component from a URL."""
    try:
        parts = [p for p in urlparse(url).path.split("/") if p]
        return parts[-1] if parts else url
    except Exception:
        return url


def _tensor_short_name(array_id: str) -> str:
    parts = [p for p in array_id.split("/") if p]
    return parts[-1] if parts else array_id


def patch_viewer_load_tensor(viewer, connection, compute_scheduler=None):
    """Monkey-patch ``load_tensor`` onto *viewer*, reading client/sources from
    the live ``TensorConnection`` *connection*.

    *compute_scheduler*, when set, pins the loaded layer's slice reads to a
    single-process dask scheduler (see ``_viewer_compute.wrap_levels``) so the
    serial viewer hits the shared main-process chunk cache instead of scattering
    across the distributed cluster (issue #8)."""

    def load_tensor(
        source_id: str,
        tensor_id: str | None = None,
        name: str | None = None,
    ) -> str:
        """Load a tensor dataset into the viewer.

        Args:
            source_id: Source identifier on the tensor server.
            tensor_id: Tensor array_id within the source.  If *None* and the
                source has exactly one tensor, it is selected automatically.
            name: Layer name.  Auto-generated from source URL if *None*.

        Returns:
            The name of the created viewer layer.
        """
        from .._tensor_utils import build_layer_scale, build_pyramid_levels
        from .._viewer_compute import wrap_levels

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

        tensor_desc = next(
            (t for t in src.tensors if t.array_id == tensor_id), None
        )
        if tensor_desc is None:
            raise ValueError(
                f"Tensor '{tensor_id}' not found in source '{source_id}'"
            )

        levels = build_pyramid_levels(
            client, source_id, tensor_id, tensor_desc
        )

        if name is None:
            stem = _get_url_stem(src.source_url) or source_id
            if len(src.tensors) > 1:
                name = f"{stem}/{_tensor_short_name(tensor_id)}"
            else:
                name = stem

        # Pull OME physical pixel size so the agent's areas/volumes come out
        # in physical units (e.g. µm²) rather than pixels (review finding B3).
        scale, phys = build_layer_scale(
            client, source_id, tensor_desc, source_desc=src
        )
        add_kwargs = {"name": name}
        if scale is not None:
            add_kwargs["scale"] = scale
        if phys is not None:
            add_kwargs["metadata"] = {"ome_physical_size": phys}

        # Pin the viewer's slice reads to a single-process scheduler so the
        # serial viewer shares the main-process chunk cache (issue #8).
        levels = wrap_levels(levels, compute_scheduler)

        if len(levels) > 1:
            viewer.add_image(levels, multiscale=True, **add_kwargs)
        else:
            viewer.add_image(levels[0], **add_kwargs)

        return name

    # napari.Viewer is a pydantic evented model with validate_assignment, so a
    # plain ``viewer.load_tensor = ...`` is rejected.  Write through to the
    # instance dict to bypass field validation.
    object.__setattr__(viewer, "load_tensor", load_tensor)
