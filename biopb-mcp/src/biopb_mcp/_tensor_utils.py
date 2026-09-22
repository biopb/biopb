"""Shared tensor utilities for biopb-mcp.

Functions for building pyramid levels and determining dimension indices,
used by both the tensor browser widget and the MCP server.
"""

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
from biopb.tensor import TensorFlightClient
from biopb.tensor._labels import label_image_axes, split_label_array_id

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


# The two axis questions position cannot answer. The data plane guarantees the
# wire order (biopb/biopb#596): Z, Y, X and S appear last, in that relative
# order, so "which axis is X" is a position rather than a lookup. What position
# cannot say is whether those trailing slots are *occupied* -- so labels are
# still read for two presence checks, and for nothing else. Mirrors
# ``AXIS_Z_LABELS`` / ``AXIS_S_LABELS`` in biopb-tensor-server ``core.axes``;
# biopb-mcp does not depend on the tensor server at runtime, so the vocabulary
# is duplicated rather than imported.
AXIS_Z_LABELS = frozenset({"z", "depth", "plane", "planes", "slice"})
AXIS_S_LABELS = frozenset({"s", "samples"})


def _resolve_axes(
    shape: Sequence[int], dim_labels: Sequence[str] | None
) -> Tuple[int, int, int | None, int | None]:
    """``(y_idx, x_idx, z_idx, s_idx)`` for a tensor in canonical wire order.

    The one place the consumers below (:func:`build_pyramid_levels`,
    :func:`build_layer_scale`, :func:`canonical_dim_labels`,
    :func:`add_tensor_layer`) decide which axis is what.

    The server advertises ``[..., Z, Y, X, S]``, so X and Y are read off the
    tail and the labels answer only the two presence questions: is the last axis
    interleaved colour (label ``S``/``samples`` gated on a size of 3 or 4, so a
    3-channel ``[C, Y, X]`` stack is never rendered as false colour), and is the
    axis ahead of Y a depth axis rather than a channel or time one (``[C, Y, X]``
    and ``[T, Y, X]`` are 3-D but not volumetric).

    A descriptor whose labels contradict the guarantee is served as ordered, not
    second-guessed -- re-deriving an order here is exactly what the wire contract
    exists to delete, and a server that predates it is one to upgrade.

    Unlabeled ``dimN`` stores (plain zarr, HDF5) are the case the contract
    deliberately leaves alone: nothing is placed, so a tensor with no labels at
    all falls back to the positional ``[..., Z, Y, X]`` reading the server's own
    ``plane_axes`` uses.

    Raises:
        ValueError: fewer than two non-samples axes -- not a displayable image.
    """
    ndim = len(shape)
    labels = (
        [str(label).lower() for label in dim_labels]
        if dim_labels and len(dim_labels) == ndim
        else None
    )

    s_idx = (
        ndim - 1
        if labels and ndim >= 3 and labels[-1] in AXIS_S_LABELS and shape[-1] in (3, 4)
        else None
    )
    x_idx = ndim - 1 if s_idx is None else ndim - 2
    y_idx = x_idx - 1
    if y_idx < 0:
        raise ValueError(
            f"Cannot identify x/y dimensions: tensor is {ndim}-D; napari needs "
            "at least 2 dimensions to display an image."
        )
    if y_idx == 0:
        z_idx = None
    elif labels is None:
        z_idx = y_idx - 1
    else:
        z_idx = y_idx - 1 if labels[y_idx - 1] in AXIS_Z_LABELS else None
    return y_idx, x_idx, z_idx, s_idx


def canonical_dim_labels(tensor_desc) -> List[str] | None:
    """Per-axis labels for the array :func:`build_pyramid_levels` returns.

    The source's own labels, lowercased -- which is the whole job now that the
    layer array *is* the source array: the server guarantees the order and the
    client no longer changes the rank, so the source's names describe the
    layer's axes one for one. Lowercased because that is the NGFF axis-name
    convention this feeds (``_writers._axis_dict``).

    The length matches the array's rank, not napari's ``layer.ndim``: an
    interleaved samples axis is a real array axis (napari just doesn't count it),
    and the OME-Zarr writer sees the raw array.

    Returns ``None`` when the source declares no usable labels -- there is then
    nothing to name the axes with, and the caller keeps its own fallback.
    """
    dim_labels = tensor_desc.dim_labels
    shape = list(tensor_desc.shape)
    if not dim_labels or len(dim_labels) != len(shape):
        return None
    return [str(label).lower() for label in dim_labels]


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

    The lean catalog row carries no pyramid -- it
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
) -> List:
    """Build resolution-pyramid levels for a tensor in napari display order.

    The levels are the server's, always. Each is requested by its advertised
    ``scale_hint`` *and* ``reduction_method`` (see
    :func:`_advertised_pyramid_levels`) so the client's chunk_ids match what the
    server serves and pre-warms.

    **When the server advertises none, this loads level 0 and stops.** There used
    to be a config-driven plan here that recomputed the ladder client-side. It
    drifted: the server moved to XY-only rungs plus one 3-D rung against a
    separate plane cap, while this still scaled X, Y and Z together against a
    single voxel budget, and it omitted ``reduction_method`` so its chunk_ids
    could not match a pre-warmed level anyway. A second, contradictory statement
    of a policy that is the server's to make is worse than no statement: the
    server has advertised a pyramid for every tensor since biopb/biopb#826, so
    the fallback was unreachable in practice and wrong wherever it did fire.

    Full resolution is the honest fallback. An old server that advertises nothing
    is also one that pre-warms nothing, so a client-computed coarse level would
    cost the same read as level 0 and deliver a blurrier picture -- measured, see
    ``chunk.compute_pyramid_scale_hints``.

    **Output axis order.** napari displays the *last* ndisplay axes by position
    and ignores ``dim_labels`` for layout, so a mis-ordered source (``[Y, X, C]``,
    a buried Z, swapped X/Y) would render the wrong plane silently. The order is
    the data plane's guarantee, not this function's job: it advertises
    ``[..., Z, Y, X, S]`` (biopb/biopb#596), so the levels arrive in display
    order and **no axis work happens here at all** -- nothing is transposed, and
    the rank is the source's.

    In particular a tensor with no Z does *not* get a singleton one inserted.
    That used to happen so ``build_layer_scale`` could write physical sizes to
    fixed trailing slots, but it made every layer disagree in rank with its
    source -- so ``layer.ndim`` had to be reasoned about separately from the
    descriptor, a 2-D image round-tripped through the OME-Zarr writer gained a
    phantom Z, and the agent guide needed a trap for the offset. The scale is
    placed by axis index instead, and the layer is now exactly the source array.

    Returns:
        List of dask arrays at canonical ``[..., Z, Y, X]`` resolution levels,
        or ``[..., Z, Y, X, S]`` when the tensor carries interleaved samples.
    """
    advertised = _advertised_pyramid_levels(client, source_id, tensor_id, tensor_desc)
    if not advertised:
        # No pyramid advertised: full resolution, and no client-side guess.
        return [client.get_tensor(tensor_id)]
    return [
        client.get_tensor(
            tensor_id,
            scale_hint=list(lv.scale_hint),
            reduction_method=lv.reduction_method or None,
        )
        for lv in advertised
    ]


def build_layer_scale(
    client: TensorFlightClient,
    source_id: str,
    ndim: int,
    *,
    tensor_id: str | None = None,
    tensor_desc=None,
    rgb: bool = False,
) -> Tuple[List[float] | None, dict | None]:
    """Build a napari ``scale`` vector from a source's physical pixel sizes.

    Reads ``client.get_physical_scale`` -- the compact per-dimension summary the
    server folds onto the descriptor ``get_tensor`` already fetches (biopb issue
    #31) -- so areas/volumes the agent computes come out in physical units (e.g.
    µm²) instead of pixels, without the heavy ``get_source_metadata`` (full OME)
    round trip. The summary is in *source* axis order -- which the server
    guarantees is canonical -- so :func:`_resolve_axes` reads x/y/z off it
    positionally, using the tensor's ``dim_labels`` only to tell whether a z
    axis is there at all.

    *ndim* is the rank of the layer **array**, which is the source's own rank:
    ``build_pyramid_levels`` neither transposes nor pads. So each resolved size
    is written to the axis it describes, and an axis the source does not have
    (no Z) is simply not written; every other axis (channel, time) gets 1.0.

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
        dim_labels = tensor_desc.dim_labels if tensor_desc is not None else None
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

        # The layer array is the source array, so a size goes to the axis it
        # describes -- no fixed trailing slot, and nothing written for an axis
        # the source does not have. (A blind ``scale[-3] = psz`` was safe only
        # while every layer was rank-evened with a singleton Z; it would now put
        # depth on the channel axis of a [C, Y, X], and IndexError on a 2-D one.)
        # For rgb the samples axis is not a napari layer dimension, so the vector
        # is one shorter -- S is last, so the x/y/z indices stay in range.
        scale = [1.0] * (ndim - 1 if rgb else ndim)
        scale[x_idx] = psx or 1.0
        scale[y_idx] = psy or 1.0
        if z_idx is not None:
            scale[z_idx] = psz or 1.0

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


def _label_binding(client, tensor_id: str, tensor_desc):
    """What a label set needs to line up with its image, or ``None``.

    ``None`` when *tensor_id* names no set -- the path is the only thing that
    marks one (``biopb.tensor._labels``). Otherwise
    ``(label_desc, image_desc, image_axes)``: the set's own descriptor, its
    image's, and for each axis of the set the index of the image axis it
    indexes. Either of the last two is ``None`` when the image cannot be
    described or the set does not span it, which leaves the set a Labels layer
    at its own rank rather than a failure.

    The set's descriptor is re-fetched because the lean catalog row carries
    neither the pyramid ``build_pyramid_levels`` needs nor the metadata holding
    the server's stated axis mapping -- one call answers both, where letting
    each ask separately would open the tensor twice.
    """
    address = split_label_array_id(tensor_id)
    if address is None:
        return None
    try:
        label_desc = client.get_descriptor(
            tensor_id, with_pyramid=True, with_metadata=True
        )
    except Exception:  # noqa: BLE001 - advisory; the catalog row still draws
        logger.warning("Label descriptor lookup failed for %s", tensor_id)
        return tensor_desc, None, None
    try:
        image_desc = client.get_descriptor(address.image_array_id)
    except Exception:  # noqa: BLE001 - advisory; see the docstring
        logger.warning(
            "Label set %s: image %s could not be described",
            tensor_id,
            address.image_array_id,
        )
        return label_desc, None, None
    axes = label_image_axes(label_desc, image_desc)
    if axes is not None and list(axes) != sorted(axes):
        # Alignment inserts the missing axes and never permutes, which the
        # extent rule makes sufficient: a set's axes are the image's in the
        # image's own order. A mapping that says otherwise is a server the rule
        # no longer describes, and inserting into it would mislay every axis
        # silently -- so nothing is aligned and the warning says why.
        logger.warning(
            "Label set %s states a reordering axis mapping %s; not aligning",
            tensor_id,
            list(axes),
        )
        axes = None
    return label_desc, image_desc, axes


def align_label_levels(levels, image_shape, image_axes, *, drop_samples=False):
    """Each level of a label set, reshaped onto its image layer's axes.

    napari aligns layers of differing rank from the **right**, so a ``T Z Y X``
    set added beside a ``T C Z Y X`` image would slide its T onto the image's
    channel axis -- frame 0 shown where frame 40 was asked for, a picture rather
    than an error. Giving the layer the image's own rank makes napari's
    positional alignment the correct one, and *image_axes* says where the set's
    axes land. It must be **ascending** -- this inserts, it does not permute,
    which the extent rule makes sufficient; :func:`_label_binding` rejects a
    mapping that would need a transpose.

    The image axes a set does not have are its channel axes, and they are
    **broadcast, not left singleton**: a singleton puts the layer outside its
    own extent at every channel but the first, where napari draws nothing rather
    than clamping -- the mask blanks as you flip channels. A broadcast axis is a
    view, mapping the whole span onto the one underlying chunk, so the mask
    shows on every channel for a single read.

    *drop_samples* for an image whose layer is ``rgb``: napari does not count
    the interleaved samples axis as a layer dimension, so the set's copy of it
    (the extent rule keeps S at full length) goes too.
    """
    missing = [i for i in range(len(image_shape)) if i not in set(image_axes)]
    aligned = []
    for level in levels:
        out = level
        # Ascending, so each insert lands at its own index in image space:
        # earlier inserts have already shifted what follows.
        for axis in missing:
            out = np.expand_dims(out, axis)
        target = list(out.shape)
        for axis in missing:
            target[axis] = int(image_shape[axis])
        out = np.broadcast_to(out, tuple(target))
        if drop_samples:
            out = out[..., 0]
        aligned.append(out)
    return aligned


def _add_label_layer(
    viewer,
    client,
    source_id: str,
    levels,
    label_desc,
    image_desc,
    image_axes,
    *,
    name: str,
    compute_scheduler: str | None,
):
    """Add a label set's levels to *viewer* as a ``Labels`` layer.

    The pyramid is the server's, exactly as for an image. That is safe for ids
    because a set's computed levels are advertised ``nearest`` -- the server
    refuses to average label values (``adapters/labels.py``) -- and napari never
    downsamples the data itself, it only picks a level. napari also sets
    ``editable = not multiscale``, so a multiscale set is already read-only.
    """
    from ._viewer_compute import wrap_levels

    if image_axes is not None and image_desc is not None:
        _, _, _, s_idx = _resolve_axes(image_desc.shape, image_desc.dim_labels)
        levels = align_label_levels(
            levels,
            image_desc.shape,
            image_axes,
            drop_samples=s_idx is not None,
        )
        # An aligned set has the image's axes, and by the extent rule the image's
        # grid, so the image's physical sizes are the layer's scale vector and
        # its names are the layer's axis names.
        scale_desc = image_desc
    else:
        logger.warning(
            "Label set %s is not bound to its image's axes; adding it at its own rank",
            getattr(label_desc, "array_id", "?"),
        )
        scale_desc = label_desc

    out_ndim = levels[0].ndim
    levels = wrap_levels(levels, compute_scheduler)

    add_kwargs = {"name": name}
    scale, phys = build_layer_scale(
        client,
        source_id,
        out_ndim,
        tensor_id=scale_desc.array_id,
        tensor_desc=scale_desc,
    )
    if scale is not None:
        add_kwargs["scale"] = scale
    metadata = {"array_id": label_desc.array_id}
    if image_desc is not None:
        # Which image this set annotates. The layer name is a display stem the
        # user can rename, and an aligned set's axes are the image's, so without
        # this the pair cannot be recovered from the layer.
        metadata["image_array_id"] = image_desc.array_id
    if phys is not None:
        metadata["ome_physical_size"] = phys
    dim_labels = canonical_dim_labels(scale_desc)
    if dim_labels:
        metadata["dim_labels"] = dim_labels
    add_kwargs["metadata"] = metadata

    with _origin_initial_view(viewer):
        if len(levels) > 1:
            layer = viewer.add_labels(levels, multiscale=True, **add_kwargs)
        else:
            layer = viewer.add_labels(levels[0], **add_kwargs)
    # Said out loud for the single-level case, where napari would leave it True:
    # a set is a write-once server tensor, and the brush on a dask-backed layer
    # raises out of napari rather than refusing, so an editable one offers an
    # edit that cannot land. (Multiscale is already read-only.)
    layer.editable = False
    return layer


def add_tensor_layer(
    viewer,
    client: TensorFlightClient,
    source_id: str,
    tensor_id: str,
    tensor_desc,
    *,
    name: str,
    compute_scheduler: str | None = None,
):
    """Build a tensor's pyramid and add it to *viewer* as an image layer.

    The shared "load a tensor into the viewer" pipeline used by both the Tensor
    Browser widget and the MCP ``add_tensor``: build pyramid levels (in napari's
    ``[..., Z, Y, X]`` display order, or ``[..., Z, Y, X, S]`` plus ``rgb=True``
    for interleaved colour), pin their slice
    reads to a single-process scheduler so the serial viewer shares the
    main-process chunk cache (issue #8; no-op standalone), attach the source's
    OME physical pixel size as ``scale`` + ``metadata['ome_physical_size']`` so
    the agent's areas/volumes come out in physical units, attach the
    canonicalized axis names as ``metadata['dim_labels']`` (the only way a
    writer, which sees just ``(path, data, meta)``, can name the axes) and the
    originating ``metadata['array_id']``, then ``add_image``
    (``multiscale=True`` when there is more than one level).

    **A label set becomes a ``Labels`` layer**, not an image one. The set is an
    ordinary tensor and its ``array_id`` is the only thing that says so
    (``<image array_id>/@labels/<name>``), so the decision is made here rather
    than at each call site -- the Tensor Browser and the MCP ``add_tensor``
    would otherwise have to agree about it separately. Everything up to the add
    is the same pipeline; what differs is the colour model and that the levels
    are first aligned onto the image's axes (:func:`align_label_levels`).
    Nothing is added implicitly: a set is a layer only when it is asked for.

    Source resolution, layer *name*, and any cursor/logging/error handling stay
    with the caller; everything from building levels through the add is uniform
    here so the three call sites can't drift.

    Returns the created napari layer.
    """
    from ._viewer_compute import wrap_levels

    # First, because a set's descriptor comes back with the pyramid on it and
    # so build_pyramid_levels needs no second open.
    binding = _label_binding(client, tensor_id, tensor_desc)
    if binding is not None:
        tensor_desc = binding[0]

    levels = build_pyramid_levels(
        client,
        source_id,
        tensor_id,
        tensor_desc,
    )
    # Present napari native-byte-order levels (biopb/biopb#296). napari's
    # thumbnail path (convert_to_uint8) does np.maximum(data, 0, out=data,
    # dtype=data.dtype), and numpy rejects a ufunc dtype= carrying byte order ->
    # TypeError on a big-endian array (e.g. a FITS '>i2' source, now preserved
    # end-to-end by the #293 binary wire schema). The swap is lazy and only
    # affects what napari sees; the wire/source bytes stay faithful. Remove once
    # napari handles non-native byte order (tracked upstream from #296).
    levels = _to_native_byteorder(levels)

    if binding is not None:
        return _add_label_layer(
            viewer,
            client,
            source_id,
            levels,
            *binding,
            name=name,
            compute_scheduler=compute_scheduler,
        )

    # An interleaved samples axis is trailing (the server's guarantee). napari
    # composites a trailing size-3/4 axis into colour only when
    # told to: rgb is left unset otherwise so napari's own auto-detection still
    # applies to unlabelled data exactly as before.
    _, _, _, s_idx = _resolve_axes(tensor_desc.shape, tensor_desc.dim_labels)
    rgb = s_idx is not None

    # Levels are the source arrays, in canonical order, at the source's rank,
    # so the scale maps onto them by axis index -- nothing to keep in sync.
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
        rgb=rgb,
    )
    if scale is not None:
        add_kwargs["scale"] = scale
    # Where this layer came from. Nothing else on the layer records it -- the
    # name is a display stem that the user can rename -- so without this a layer
    # cannot be traced back to its tensor, and re-reading the full-resolution
    # source-order array means guessing at the catalog.
    metadata = {"array_id": tensor_id}
    if phys is not None:
        metadata["ome_physical_size"] = phys
    # Name the layer's axes for the OME-Zarr writer (biopb/biopb#651): it is
    # handed only napari's (path, data, meta), so with no labels there it falls
    # back to a positional guess that mislabels every leading pair that isn't
    # (C, T) -- writing a TCZYX source with T and C swapped.
    labels = canonical_dim_labels(tensor_desc)
    if labels:
        metadata["dim_labels"] = labels
    add_kwargs["metadata"] = metadata

    with _origin_initial_view(viewer):
        if len(levels) > 1:
            return viewer.add_image(levels, multiscale=True, **add_kwargs)
        return viewer.add_image(levels[0], **add_kwargs)
