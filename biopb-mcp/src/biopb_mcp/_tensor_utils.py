"""Shared tensor utilities for biopb-mcp.

Functions for building pyramid levels and determining dimension indices,
used by both the tensor browser widget and the MCP server.
"""

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from typing import List, Tuple

from biopb.tensor import TensorFlightClient

from ._config import CONFIG, get_setting

logger = logging.getLogger(__name__)


@contextmanager
def _origin_initial_view(viewer):
    """Render the first-added layer at the dataset *origin*, not its center.

    napari builds a layer's thumbnail from the origin slice but renders the
    view at the *center* of every axis: adding the first layer runs
    ``ViewerModel._add_layer_from_data`` -> ``dims._go_to_center_step()``. For a
    tensor with non-spatial axes (channel/time) those are two *different*
    slices, so loading one layer materializes two coarse planes. With a source
    that has no native pyramid (e.g. nd2), each such plane is a full-resolution
    server-side decode, so the redundant center slice roughly doubles cold load
    time. Pinning the initial view to the origin makes the displayed slice
    coincide with the thumbnail slice -> one decode instead of two.

    Trade-off: the default view sits at index 0 on the sliced axes (e.g. the
    first channel / first Z) rather than the middle. ``_go_to_center_step`` only
    runs for the first layer, so this is a no-op for subsequent adds.
    """
    dims_cls = type(getattr(viewer, "dims", None))
    orig = getattr(dims_cls, "_go_to_center_step", None)
    if not callable(orig):
        # Not a real napari viewer (e.g. a test mock) -- nothing to suppress.
        yield
        return
    dims_cls._go_to_center_step = lambda self: None
    try:
        yield
    finally:
        dims_cls._go_to_center_step = orig


def get_xy_dim_indices(
    shape: Sequence[int], dim_labels: Sequence[str] | None = None
) -> Tuple[int, int]:
    """Indices of the y and x dimensions for a tensor of *shape*.

    Uses *dim_labels* as the primary source (looks for 'x', 'y'), falling back
    to the last two dimensions under the standard ``[..., Y, X]`` convention
    (X last, Y second-to-last) when labels are unavailable.

    Returns:
        Tuple of (y_index, x_index) -- y first, matching the row/col convention.

    Raises:
        ValueError: the tensor has fewer than 2 dimensions (not a displayable
            image).
    """
    ndim = len(shape)

    if dim_labels:
        labels_lower = [str(label).lower() for label in dim_labels]
        try:
            return (labels_lower.index("y"), labels_lower.index("x"))
        except ValueError:
            pass

    if ndim < 2:
        raise ValueError(
            f"Cannot identify x/y dimensions: tensor is {ndim}-D; napari needs "
            "at least 2 dimensions to display an image."
        )
    # Standard [..., Y, X]: X is the last axis, Y the second-to-last.
    return (ndim - 2, ndim - 1)


def get_z_dim_index(
    shape: Sequence[int], dim_labels: Sequence[str] | None = None
) -> int | None:
    """Index of the z (depth) axis, or ``None`` when the tensor has none.

    Respects *dim_labels* ('z') first: when labels are present but carry no
    'z', the tensor is taken to have no depth axis (``None``) -- not every 3-D+
    tensor is volumetric (``[T, Y, X]``, ``[C, Y, X]`` have no z). With no
    labels, assume the positional ``[..., Z, Y, X]`` convention -- the
    third-from-last axis -- for 3-D+ tensors, and ``None`` for <3-D.

    Mis-identifying a small channel/time axis as z is low-risk: such axes stay
    below the pyramid floor and are never downsampled.
    """
    ndim = len(shape)
    if dim_labels:
        labels_lower = [str(label).lower() for label in dim_labels]
        return labels_lower.index("z") if "z" in labels_lower else None
    return ndim - 3 if ndim >= 3 else None


# Interleaved RGB(A) samples axis. Mirrors ``AXIS_S_LABELS`` / ``samples_axis``
# in biopb-tensor-server ``core.axes``: biopb-mcp does not depend on the tensor
# server at runtime, so the rule is duplicated rather than imported -- the same
# reason ``core.axes`` itself mirrors the frontend ``buildAxisMap``. Keep the
# two in sync; biopb/biopb#596 tracks collapsing them onto a server-guaranteed
# axis order.
AXIS_S_LABELS = frozenset({"s", "samples"})


def get_samples_dim_index(
    shape: Sequence[int], dim_labels: Sequence[str] | None = None
) -> int | None:
    """Index of an interleaved RGB(A) samples axis, or ``None``.

    Detected by *label* (``S`` / ``samples``) gated on a size of 3 or 4. This
    axis holds the colour components of one pixel, so napari must composite it
    (``rgb=True``) rather than slide over it one plane at a time like T/Z/C.

    Label-only by design -- no positional fallback. Guessing "a trailing size-3
    axis means colour" would render a 3-channel ``[C, Y, X]`` stack as a false
    RGB image, which is worse than the slider it would replace.
    """
    if not dim_labels:
        return None
    for i, label in enumerate(dim_labels):
        if (
            str(label).lower() in AXIS_S_LABELS
            and i < len(shape)
            and shape[i] in (3, 4)
        ):
            return i
    return None


def _resolve_axes(
    shape: Sequence[int], dim_labels: Sequence[str] | None
) -> Tuple[int, int, int | None, int | None]:
    """``(y_idx, x_idx, z_idx, s_idx)`` for a source tensor, degenerate axes dropped.

    The one place the canonicalization consumers (:func:`build_pyramid_levels`,
    :func:`build_layer_scale`, :func:`add_tensor_layer`) resolve axes, so their
    notion of "which axis is what" cannot drift apart.

    A z or samples axis that collides with an already-resolved axis is dropped
    rather than trusted: a degenerate label set (``"CYS"``, ``"YS"``, a source
    labelling two axes ``y``) then renders as a plain plane instead of raising
    or transposing one axis into two slots.
    """
    y_idx, x_idx = get_xy_dim_indices(shape, dim_labels)
    z_idx = get_z_dim_index(shape, dim_labels)
    if z_idx is not None and z_idx in (x_idx, y_idx):
        z_idx = None
    s_idx = get_samples_dim_index(shape, dim_labels)
    if s_idx is not None and s_idx in (x_idx, y_idx, z_idx):
        s_idx = None
    return y_idx, x_idx, z_idx, s_idx


def _canonical_axis_plan(
    ndim: int, y_idx: int, x_idx: int, z_idx: int | None, s_idx: int | None
) -> Tuple[Tuple[int, ...], int | None]:
    """How to canonicalize a source's axes to ``[..., Z, Y, X]`` / ``[..., Z, Y, X, S]``.

    Returns ``(perm, z_insert_tail)``: the transpose permutation, and -- when the
    tensor carries no z axis -- how many axes the inserted singleton Z sits ahead
    of (Y, X, and a samples axis when there is one), or ``None`` when z is real.
    Z goes ahead of Y/X so it stays third-from-last among the spatial axes, and
    ahead of S so S stays strictly last.

    :func:`build_pyramid_levels` applies the plan to the level arrays and
    :func:`canonical_dim_labels` applies it to the labels, so the arrays and the
    names describing them cannot drift apart.
    """
    trailing = ([z_idx] if z_idx is not None else []) + [y_idx, x_idx]
    if s_idx is not None:
        trailing.append(s_idx)
    perm = tuple([i for i in range(ndim) if i not in trailing] + trailing)
    z_insert_tail = None if z_idx is not None else (3 if s_idx is not None else 2)
    return perm, z_insert_tail


def canonical_dim_labels(tensor_desc, source_desc=None) -> List[str] | None:
    """Per-axis labels for the array :func:`build_pyramid_levels` returns.

    The source's own labels put through the same canonicalization as the pixels
    -- reordered by the transpose, with a ``"z"`` spliced in where the singleton
    Z is inserted -- so the result names the *layer's* axes rather than the
    source's. Lowercased, both because the spliced ``"z"`` would otherwise clash
    with an upper-case source convention and because that is the NGFF axis-name
    convention this feeds (``_writers._axis_dict``).

    The length matches the array's rank, not napari's ``layer.ndim``: an
    interleaved samples axis is a real array axis (napari just doesn't count it),
    and the OME-Zarr writer sees the raw array.

    Returns ``None`` when the source declares no usable labels -- there is then
    nothing to name the leading axes with, and the caller keeps its own fallback.
    """
    dim_labels = tensor_desc.dim_labels or getattr(source_desc, "dim_labels", None)
    shape = list(tensor_desc.shape)
    if not dim_labels or len(dim_labels) != len(shape):
        return None
    y_idx, x_idx, z_idx, s_idx = _resolve_axes(shape, dim_labels)
    perm, z_insert_tail = _canonical_axis_plan(len(shape), y_idx, x_idx, z_idx, s_idx)
    labels = [str(dim_labels[i]).lower() for i in perm]
    if z_insert_tail is not None:
        labels.insert(len(labels) - z_insert_tail, "z")
    return labels


def _advertised_pyramid_levels(client, source_id, tensor_id, tensor_desc):
    """The server-advertised pyramid: per-level ``scale_hint`` + ``reduction_method``.

    Returns the advertised level descriptors, or ``[]`` when the server
    advertises none (older servers) or the lookup fails.

    Why this matters: the server folds its downsample plan onto the descriptor
    (``scale_hint`` *and* ``reduction_method`` per level) and pre-warms exactly
    those chunk_ids. If the client builds its own ``scale_hint`` and omits the
    reduction, the server falls back to a *different* default (e.g. ``nearest``
    vs the advertised ``area``), so the client's chunk_ids never match the
    pre-warmed ones and every first load pays a full cold read. Honoring the
    advertised levels keeps the client's requests byte-identical to what the
    server serves and precaches.

    The lean catalog descriptor from ``list_sources`` carries no pyramid -- it
    is filled only at open time (``get_flight_info``) -- so when the passed
    *tensor_desc* lacks one, fetch the open-time descriptor once via
    ``get_descriptor``. That fetch is a **describe** (biopb/biopb#563): it asks
    for the pyramid (``with_pyramid=True``) but not the O(chunks) read plan (the
    default ``with_read_plan=False``) nor the heavy OME tree (``with_metadata``
    defaults False) -- so learning the levels no longer builds and discards a
    level-0 plan, and this probe is cheap enough to run per open.
    """
    levels = list(getattr(tensor_desc, "pyramid", None) or [])
    if levels:
        return levels
    try:
        full = client.get_descriptor(tensor_id, with_pyramid=True)
        return list(getattr(full, "pyramid", None) or [])
    except Exception:  # noqa: BLE001 - advisory; fall back to a client plan
        logger.debug(
            "advertised-pyramid lookup failed for %s/%s",
            source_id,
            tensor_id,
            exc_info=True,
        )
    return []


def build_pyramid_levels(
    client: TensorFlightClient,
    source_id: str,
    tensor_id: str,
    tensor_desc,
    source_desc=None,
    config: dict | None = None,
) -> List:
    """Build resolution-pyramid levels for a tensor in napari display order.

    When the server advertises a pyramid (the open-time descriptor carries one;
    see :func:`_advertised_pyramid_levels`), each level is requested by its
    advertised ``scale_hint`` *and* ``reduction_method`` so the client's
    chunk_ids match what the server serves and pre-warms. The config-driven plan
    below is the fallback for servers that advertise none.

    That fallback uses one unified rule for 2-D and 3-D data and bounds napari's
    3-D whole-volume read (issue #29). All knobs come from the ``pyramid`` config
    section (``config`` defaults to the on-disk config):

    - ``threshold`` -- max x/y extent of the coarsest level (caps 2-D reads),
    - ``downscale_factor`` -- linear step between levels,
    - ``pixel_budget_cubic_root`` -- per-axis floor; its cube is the max voxels
      (``Lx*Ly*Lz``) allowed in the coarsest level, bounding the whole-volume
      read napari issues in 3-D. Stored as the cube root (not the product) so
      the floor and the budget are exact integers, free of cube-root rounding.

    Each level is requested at the current per-axis scale, then x, y and z are
    downsampled *individually* -- skipping any axis that has reached the floor
    (``pixel_budget_cubic_root``, capped at ``threshold``) -- until the coarsest
    level fits both the voxel budget and ``threshold`` in x/y. The floor keeps
    small axes (channels, time, thin z) from being over-shrunk and guarantees
    termination: once every axis is at or below it, ``Lx*Ly*Lz <= floor**3 <=
    budget`` and ``Lx, Ly <= threshold``. A tensor without a z axis is treated
    as ``Lz = 1`` and never gets a z scale factor.

    The per-level extents are read from the *returned* array's shape, not
    computed as ``L // scale`` -- the server's downsample rounding (floor vs
    ceil) is not part of the API contract, so trusting the real shape keeps the
    budget check correct either way.

    **Output axis order.** napari displays the *last* ndisplay axes by position
    and ignores ``dim_labels`` for layout, so a source advertising an
    out-of-order layout (``[Y, X, C]``, a buried Z, swapped X/Y) would render
    the wrong plane silently. Using the labels (per-tensor, falling back to
    *source_desc*), each level is transposed so X is last, Y second-to-last, and
    Z third-to-last -- with a singleton Z *inserted* when the tensor has none.

    An interleaved RGB(A) **samples** axis (see :func:`get_samples_dim_index`)
    is kept *after* X instead, giving ``[..., Z, Y, X, S]``. Without this the
    samples axis is unrecognized, falls into the leading "everything else"
    bucket, and a ``[T, C, Z, Y, X, S]`` RGB source loads as a 3-plane stack
    behind a slider rather than a colour image (biopb/biopb#596).

    The result is therefore **always** canonical ``[..., Z, Y, X]`` (rank >= 3),
    or ``[..., Z, Y, X, S]`` (rank >= 4) for interleaved colour -- which lets
    ``build_layer_scale`` map physical sizes onto fixed trailing positions
    without re-deriving the labels. The transpose is a lazy dask graph relabel;
    the real server emits ordered axes, so it is normally a no-op guard.

    Returns:
        List of dask arrays at canonical ``[..., Z, Y, X]`` resolution levels,
        or ``[..., Z, Y, X, S]`` when the tensor carries interleaved samples.
    """
    if config is None:
        config = CONFIG.as_dict()
    threshold = get_setting(config, "pyramid.threshold")
    downscale_factor = get_setting(config, "pyramid.downscale_factor")
    budget_root = get_setting(config, "pyramid.pixel_budget_cubic_root")
    pixel_budget = budget_root**3

    shape = tensor_desc.shape
    ndim = len(shape)

    # Per-tensor labels win; fall back to the source descriptor's labels.
    dim_labels = tensor_desc.dim_labels or getattr(source_desc, "dim_labels", None)
    y_idx, x_idx, z_idx, s_idx = _resolve_axes(shape, dim_labels)

    # Stop shrinking an axis once it reaches this floor; see the docstring for
    # why the cube-root-capped-at-threshold value guarantees termination.
    axis_floor = min(budget_root, threshold)

    # Prefer the server-advertised pyramid: request each level by the
    # advertised scale_hint *and* reduction_method so the client's chunk_ids
    # match what the server serves and pre-warms. The config-driven loop below
    # is the fallback for servers that advertise no pyramid -- it recomputes the
    # scale plan and (deliberately) omits reduction_method, which is fine only
    # when there is nothing pre-warmed to miss.
    advertised = _advertised_pyramid_levels(client, source_id, tensor_id, tensor_desc)
    if advertised:
        levels = [
            client.get_tensor(
                tensor_id,
                scale_hint=list(lv.scale_hint),
                reduction_method=lv.reduction_method or None,
            )
            for lv in advertised
        ]
    else:
        levels = []
        sx = sy = sz = 1

        while True:
            # scale_hint is in the *source* axis order the server expects.
            scale_hint = [1] * ndim
            scale_hint[x_idx] = sx
            scale_hint[y_idx] = sy
            if z_idx is not None:
                scale_hint[z_idx] = sz

            arr = client.get_tensor(tensor_id, scale_hint=scale_hint)
            levels.append(arr)

            # Real downsampled extents from the returned array, not
            # floor(L/scale).
            lx = arr.shape[x_idx]
            ly = arr.shape[y_idx]
            lz = arr.shape[z_idx] if z_idx is not None else 1
            if lx * ly * lz <= pixel_budget and lx <= threshold and ly <= threshold:
                break

            # Downsample each axis individually, leaving any at the floor.
            nsx = sx * downscale_factor if lx > axis_floor else sx
            nsy = sy * downscale_factor if ly > axis_floor else sy
            nsz = (
                sz * downscale_factor if (z_idx is not None and lz > axis_floor) else sz
            )
            if (nsx, nsy, nsz) == (sx, sy, sz):
                break  # nothing left to shrink; avoid an infinite loop
            sx, sy, sz = nsx, nsy, nsz

    # Canonicalize to [..., Z, Y, X], reusing the indices computed above (no
    # second pass over the labels). Transpose moves X/Y/Z into place; a missing
    # Z is inserted as a singleton so the output rank and trailing axes are
    # uniform for every tensor. Both ops are lazy on dask arrays.
    perm, z_insert_tail = _canonical_axis_plan(ndim, y_idx, x_idx, z_idx, s_idx)
    if perm != tuple(range(ndim)):
        levels = [level.transpose(perm) for level in levels]
    if z_insert_tail is not None:
        expand = (Ellipsis, None) + (slice(None),) * z_insert_tail
        levels = [level[expand] for level in levels]

    return levels


def build_layer_scale(
    client: TensorFlightClient,
    source_id: str,
    ndim: int,
    *,
    tensor_id: str | None = None,
    tensor_desc=None,
    source_desc=None,
    rgb: bool = False,
) -> Tuple[List[float] | None, dict | None]:
    """Build a napari ``scale`` vector from a source's physical pixel sizes.

    Reads ``client.get_physical_scale`` -- the compact per-dimension summary the
    server folds onto the descriptor ``get_tensor`` already fetches (biopb issue
    #31) -- so areas/volumes the agent computes come out in physical units (e.g.
    µm²) instead of pixels, without the heavy ``get_source_metadata`` (full OME)
    round trip. The summary is in *source* axis order; the source's
    ``dim_labels`` (per-tensor, falling back to *source_desc*) map each physical
    size onto x/y/z via :func:`get_xy_dim_indices` / :func:`get_z_dim_index`.

    *ndim* is the rank of the layer **array**, which ``build_pyramid_levels``
    guarantees is in canonical ``[..., Z, Y, X]`` order (rank >= 3, with an
    explicit -- possibly singleton -- Z). So the resolved x/y/z sizes land on
    fixed trailing positions: X last, Y second-to-last, Z third-to-last; leading
    axes (channel, time) get 1.0.

    Pass *rgb* when the array is ``[..., Z, Y, X, S]`` interleaved colour. napari
    does not count the samples axis as a layer dimension (``layer.ndim ==
    data.ndim - 1``) and requires ``len(scale) == layer.ndim``, so the returned
    vector is one shorter -- X is last in it either way, and returning a full
    *ndim*-length scale for an rgb layer would raise.

    When the server advertises no physical scale (an older server, or a format
    that carries none), returns ``(None, None)`` -- the layer simply gets no
    physical scale. There is no full-OME fallback.

    Returns:
        ``(scale, info)`` where *scale* is a per-axis list of length *ndim*
        (``None`` if no physical sizes are available) and *info* is a small dict
        of the physical sizes + units for surfacing to the agent (``None`` if
        unavailable).
    """

    def _positive_float(value):
        """Coerce to a positive float, or None for missing/garbage values."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    try:
        phys = client.get_physical_scale(tensor_id)
        if phys is None:
            return None, None
        scale_vec, unit_vec = phys

        # Map source-order physical sizes onto x/y/z by dim label.
        dim_labels = None
        if tensor_desc is not None:
            dim_labels = tensor_desc.dim_labels
        if not dim_labels:
            dim_labels = getattr(source_desc, "dim_labels", None)
        src_shape = list(tensor_desc.shape) if tensor_desc is not None else scale_vec
        y_idx, x_idx, z_idx, _ = _resolve_axes(src_shape, dim_labels)

        def _at(idx):
            return (
                scale_vec[idx] if (idx is not None and idx < len(scale_vec)) else None
            )

        def _unit_at(idx):
            return unit_vec[idx] if (idx is not None and idx < len(unit_vec)) else None

        psx = _positive_float(_at(x_idx))
        psy = _positive_float(_at(y_idx))
        psz = _positive_float(_at(z_idx))
        if not any((psx, psy, psz)):
            return None, None

        # Canonical [..., Z, Y, X(, S)]: physical sizes land on the trailing
        # spatial axes. For rgb the samples axis is not a napari layer dimension,
        # so it is excluded here and X is last in the scale vector either way.
        scale = [1.0] * (ndim - 1 if rgb else ndim)
        scale[-1] = psx or 1.0
        scale[-2] = psy or 1.0
        scale[-3] = psz or 1.0

        info = {
            "physical_size_x": psx,
            "physical_size_y": psy,
            "physical_size_z": psz,
            "physical_size_x_unit": _unit_at(x_idx) or None,
            "physical_size_y_unit": _unit_at(y_idx) or None,
            "physical_size_z_unit": _unit_at(z_idx) or None,
        }
        return scale, info
    except Exception as exc:
        logger.warning("build_layer_scale failed for %s: %s", source_id, exc)
        return None, None


def _to_native_byteorder(levels):
    """Return *levels* with any non-native-endian array swapped to native order.

    Workaround for a napari thumbnail bug (biopb/biopb#296): a big-endian array
    (e.g. a FITS ``>i2`` source, preserved end-to-end by the #293 binary wire
    schema) trips ``np.maximum(data, 0, out=data, dtype=data.dtype)`` in napari's
    ``convert_to_uint8`` -- numpy rejects a ufunc ``dtype=`` that carries byte
    order. The ``astype`` is lazy on a dask array, so the source bytes are never
    materialized here and the values are unchanged (only the in-memory byte order
    napari sees). Native levels pass through untouched. Remove when napari handles
    non-native byte order upstream.
    """
    return [
        lv.astype(lv.dtype.newbyteorder("=")) if not lv.dtype.isnative else lv
        for lv in levels
    ]


def add_tensor_layer(
    viewer,
    client: TensorFlightClient,
    source_id: str,
    tensor_id: str,
    tensor_desc,
    *,
    name: str,
    source_desc=None,
    compute_scheduler: str | None = None,
    config: dict | None = None,
):
    """Build a tensor's pyramid and add it to *viewer* as an image layer.

    The shared "load a tensor into the viewer" pipeline used by both the Tensor
    Browser widget and the MCP ``add_tensor``: build pyramid levels (already
    canonicalized to napari's ``[..., Z, Y, X]`` display order, or
    ``[..., Z, Y, X, S]`` plus ``rgb=True`` for interleaved colour), pin their slice
    reads to a single-process scheduler so the serial viewer shares the
    main-process chunk cache (issue #8; no-op standalone), attach the source's
    OME physical pixel size as ``scale`` + ``metadata['ome_physical_size']`` so
    the agent's areas/volumes come out in physical units, attach the
    canonicalized axis names as ``metadata['dim_labels']`` (the only way a
    writer, which sees just ``(path, data, meta)``, can name the axes), then
    ``add_image`` (``multiscale=True`` when there is more than one level).

    Source resolution, layer *name*, and any cursor/logging/error handling stay
    with the caller; everything from building levels through ``add_image`` is
    uniform here so the three call sites can't drift.

    Returns the created napari layer.
    """
    from ._viewer_compute import wrap_levels

    levels = build_pyramid_levels(
        client,
        source_id,
        tensor_id,
        tensor_desc,
        source_desc=source_desc,
        config=config,
    )
    # Present napari native-byte-order levels (biopb/biopb#296). napari's
    # thumbnail path (convert_to_uint8) does np.maximum(data, 0, out=data,
    # dtype=data.dtype), and numpy rejects a ufunc dtype= carrying byte order ->
    # TypeError on a big-endian array (e.g. a FITS '>i2' source, now preserved
    # end-to-end by the #293 binary wire schema). The swap is lazy and only
    # affects what napari sees; the wire/source bytes stay faithful. Remove once
    # napari handles non-native byte order (tracked upstream from #296).
    levels = _to_native_byteorder(levels)
    # An interleaved samples axis is now trailing (build_pyramid_levels put it
    # there). napari composites a trailing size-3/4 axis into colour only when
    # told to: rgb is left unset otherwise so napari's own auto-detection still
    # applies to unlabelled data exactly as before.
    dim_labels = tensor_desc.dim_labels or getattr(source_desc, "dim_labels", None)
    _, _, _, s_idx = _resolve_axes(tensor_desc.shape, dim_labels)
    rgb = s_idx is not None

    # Levels are in canonical [..., Z, Y, X(, S)] order, so the scale maps onto
    # the output rank directly -- no reordering to keep in sync.
    out_ndim = levels[0].ndim
    levels = wrap_levels(levels, compute_scheduler)

    add_kwargs = {"name": name}
    if rgb:
        add_kwargs["rgb"] = True
    scale, phys = build_layer_scale(
        client,
        source_id,
        out_ndim,
        tensor_id=tensor_id,
        tensor_desc=tensor_desc,
        source_desc=source_desc,
        rgb=rgb,
    )
    if scale is not None:
        add_kwargs["scale"] = scale
    metadata = {}
    if phys is not None:
        metadata["ome_physical_size"] = phys
    # Name the layer's axes for the OME-Zarr writer (biopb/biopb#651): it is
    # handed only napari's (path, data, meta), so with no labels there it falls
    # back to a positional guess that mislabels every leading pair that isn't
    # (C, T) -- writing a TCZYX source with T and C swapped.
    labels = canonical_dim_labels(tensor_desc, source_desc=source_desc)
    if labels:
        metadata["dim_labels"] = labels
    if metadata:
        add_kwargs["metadata"] = metadata

    with _origin_initial_view(viewer):
        if len(levels) > 1:
            return viewer.add_image(levels, multiscale=True, **add_kwargs)
        return viewer.add_image(levels[0], **add_kwargs)
