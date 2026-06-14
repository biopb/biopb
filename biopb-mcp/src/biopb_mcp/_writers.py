"""GUI-free OME-Zarr writers for the biopb-mcp napari plugin.

Registered as ``contributions.writers`` in ``napari.yaml`` so that
File -> Save Selected Layer(s) can persist a napari image/labels layer to a
local, standalone OME-Zarr directory with **no extra install**: ``ome-zarr``
(ome-zarr-py) and ``zarr`` already ship transitively via ``biopb[tensor]`` (and
``ome-zarr`` is a direct dependency of this package).

This module imports neither Qt nor napari, consistent with the package's
GUI-free rule (see ``biopb_mcp/__init__.py``): napari calls these functions with
plain ``(path, data, meta)`` values, so only numpy/zarr/ome_zarr are needed. The
heavy imports are function-local so importing this module stays cheap.

The OME metadata (multiscales axis typing) mirrors the biopb tensor server's
``_build_minimal_ome_metadata`` (``biopb_tensor_server.server``) so the output
round-trips through the server's ``OmeZarrAdapter``, which derives ``dim_labels``
from ``multiscales[0].axes[*].name``.

Both image and labels layers are written as a **top-level** OME-Zarr image with
the layer's dtype preserved (integer for masks). This is simpler than the OME
spec's nested ``labels/<name>`` group and is exactly what loads back as a
first-class browsable source through ``OmeZarrAdapter``.

FUTURE EXTENSION POINT: the preferred next format is NDTiff (not OME-TIFF). To
add it, register a new command + writer entry in ``napari.yaml`` pointing at a
``write_*_ndtiff`` function here, reusing the ``_derive_axes`` helper below.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Axis-typing convention -- mirrors biopb_tensor_server.server
# ._build_minimal_ome_metadata so written axes round-trip through OmeZarrAdapter.
_SPACE = ("x", "y", "z")
_CHANNEL = ("c", "channel")
_TIME = ("t", "time")


def _axis_dict(name: str) -> dict[str, str]:
    """One OME multiscales axis dict, typed by name (server convention)."""
    low = name.lower()
    if low in _SPACE:
        return {"name": name, "type": "space"}
    if low in _CHANNEL:
        return {"name": name, "type": "channel"}
    if low in _TIME:
        return {"name": name, "type": "time"}
    return {"name": name}


def _default_dim_labels(ndim: int) -> list[str]:
    """Positional fallback when no explicit labels are available.

    biopb-mcp layers are canonical ``[..., Z, Y, X]`` (see ``_tensor_utils``), so
    the trailing dims are X, Y, Z; remaining leading dims get c, then t, then
    generic ``dim{i}`` names.
    """
    trailing = ["z", "y", "x"]
    n_trailing = min(ndim, len(trailing))
    leading = ndim - n_trailing
    leading_names = ["c", "t"]
    labels = [
        leading_names[i] if i < len(leading_names) else f"dim{i}"
        for i in range(leading)
    ]
    labels.extend(trailing[len(trailing) - n_trailing :])
    return labels


def _derive_dim_labels(ndim: int, meta: dict) -> list[str]:
    """Resolve per-axis labels for an OME multiscales ``axes`` block.

    Prefer explicit labels stashed on the layer metadata (forward-compatible if
    the tensor browser ever attaches them); otherwise fall back to the canonical
    ``[..., Z, Y, X]`` positional mapping.
    """
    inner = meta.get("metadata") or {}
    for key in ("dim_labels", "axis_labels"):
        labels = inner.get(key)
        if labels and len(labels) == ndim:
            return [str(x) for x in labels]
    return _default_dim_labels(ndim)


def _derive_axes(ndim: int, meta: dict) -> list[dict[str, str]]:
    return [_axis_dict(name) for name in _derive_dim_labels(ndim, meta)]


def _scale_transform(ndim: int, meta: dict) -> list[list[dict[str, Any]]]:
    """coordinateTransformations from the layer's napari ``scale`` (physical
    pixel size), defaulting to 1.0 per axis (matches the server convention)."""
    scale = meta.get("scale")
    if scale is None or len(scale) != ndim:
        scale = [1.0] * ndim
    return [[{"type": "scale", "scale": [float(s) for s in scale]}]]


def _open_group(path: str):
    """Open a writable zarr group at *path* (the local directory the user chose)."""
    import zarr
    from ome_zarr.io import parse_url

    loc = parse_url(path, mode="w")
    if loc is None:
        raise ValueError(f"Cannot open OME-Zarr store at {path!r}")
    return zarr.group(store=loc.store, overwrite=True)


def _write(path: str, data, meta: dict) -> list[str]:
    """Write a single napari layer to a standalone OME-Zarr directory at *path*.

    ``data`` is the layer array (or a list of arrays for multiscale layers -- we
    use the full-resolution level and let nothing else through, writing a single
    resolution so the output matches the server's single-dataset convention and
    keeps integer label arrays exact).
    """
    from ome_zarr.scale import Scaler
    from ome_zarr.writer import write_image

    level0 = np.asarray(data[0] if isinstance(data, (list, tuple)) else data)
    ndim = level0.ndim

    group = _open_group(path)
    # max_layer=0 -> a single resolution level "0" (no downsampled pyramid),
    # keeping axes/coordinateTransformations in lockstep and labels integer-exact.
    write_image(
        level0,
        group,
        scaler=Scaler(max_layer=0),
        axes=_derive_axes(ndim, meta),
        coordinate_transformations=_scale_transform(ndim, meta),
    )
    return [path]


# --- napari single-layer writer entry points (func(path, data, meta) -> paths) ---
def write_image_ome_zarr(path: str, data, meta: dict) -> list[str]:
    """napari writer for ``image`` layers -> standalone OME-Zarr directory."""
    return _write(path, data, meta)


def write_labels_ome_zarr(path: str, data, meta: dict) -> list[str]:
    """napari writer for ``labels`` layers -> standalone OME-Zarr directory.

    Written as a top-level integer OME-Zarr image (dtype preserved) rather than a
    nested ``labels/<name>`` group, so the mask loads back as a browsable source.
    """
    return _write(path, data, meta)
