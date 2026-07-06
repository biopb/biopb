"""GPU-free fork of the pieces of ``ucell.dynamics`` needed for chunk-wise
(lazy) segmentation, shared across flow-based servicers via ``biopb_image_base``.

Why this exists
---------------
``ucell.dynamics`` lives only in the ucell ML image, imports ``torch`` /
``fastremap`` (absent from the dev venv), and its public ``compute_masks``
discards the per-pixel *destination* map that cross-chunk stitching depends on.
This module re-implements just two steps in pure numpy/scipy so the logic is
importable and unit-testable without Docker or a GPU:

1. ``follow_flows`` -- Euler integration of the flow field, returning the final
   location ("destination", ~cell center) of every foreground pixel. The torch
   version normalises coordinates to [-1, 1] and grid-samples; algebra shows one
   step is simply ``p += dP(p)`` in pixel units with edge-clamping, which is what
   we do here via ``scipy.ndimage.map_coordinates`` (bilinear).
2. ``cluster_destinations`` -- the "light clustering" that turns the (noisy,
   spread-over-a-few-pixels) destinations into one label per cell: histogram the
   destinations, pick peaks as seeds, grow each seed, then label every pixel by
   the seed its destination fell into. This mirrors ``get_masks_torch``.

``cluster_destinations`` additionally accepts **injected seeds** carrying IDs
from previously processed chunks. A cluster that lands on an injected seed
inherits that ID -- this is the mechanism that keeps instance IDs consistent
across chunk borders without re-reading saved labels.

Constants (rpad=20, kernel 5 for peak detection, h>10, 11^ndim seed window,
``h>2`` growth, 5 growth iters, ``max_size_fraction``) match the upstream
``get_masks_torch`` so per-chunk behaviour stays close to the original. The QC
step (``remove_bad_flow_masks``) is intentionally omitted: the server calls
``compute_masks`` with ``flow_threshold=0``, so it is a no-op.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter


def follow_flows(dP: np.ndarray, inds: tuple, niter: int = 200) -> np.ndarray:
    """Euler-integrate the flow field to find each foreground pixel's destination.

    Args:
        dP: flow field, shape ``(ndim, *spatial)`` ordered to match ``inds``
            (e.g. 2D is ``(dy, dx)``). Already scaled by the caller.
        inds: tuple of index arrays (the output of ``np.nonzero(foreground)``),
            one array per spatial axis.
        niter: number of integration steps.

    Returns:
        Float array ``(ndim, n_points)`` of final pixel locations.
    """
    shape = dP.shape[1:]
    ndim = len(shape)
    p = np.array(inds, dtype=np.float32)  # (ndim, n_points)

    for _ in range(niter):
        # Sample each flow component at the current sub-pixel locations.
        step = np.stack(
            [map_coordinates(dP[k], p, order=1, mode="nearest") for k in range(ndim)]
        )
        p = p + step
        for k in range(ndim):
            np.clip(p[k], 0, shape[k] - 1, out=p[k])

    return p


def compute_destinations(
    dP: np.ndarray,
    cellprob: np.ndarray,
    cellprob_threshold: float = 0.0,
    niter: int = 200,
) -> tuple[tuple, Optional[np.ndarray]]:
    """Foreground indices and their integrated destinations.

    Mirrors the front half of ``ucell.dynamics.compute_masks``: threshold the
    cell probability, then follow the (masked, /5-scaled) flows.

    Args:
        dP: flow field ``(ndim, *spatial)`` as passed to ``compute_masks``
            (i.e. already multiplied by the model's flow scale, e.g. ``*4.0``).
        cellprob: cell-probability logits, shape ``spatial``.
        cellprob_threshold: foreground threshold on ``cellprob``.
        niter: integration steps.

    Returns:
        ``(inds, p)`` where ``inds`` is the foreground index tuple and ``p`` is
        the ``(ndim, n_points)`` destination array (``None`` if no foreground).
    """
    mask = cellprob > cellprob_threshold
    inds = np.nonzero(mask)
    if len(inds[0]) == 0:
        return inds, None

    p = follow_flows(dP * mask / 5.0, inds, niter=niter)
    return inds, p


def _detect_seeds(h: np.ndarray, min_count: int) -> np.ndarray:
    """Peak coordinates of the destination histogram, sorted by ascending count.

    A voxel is a seed if it is a local maximum within a 5-window and collects
    more than ``min_count`` destinations. Ascending sort means higher-count seeds
    are written last during labelling and therefore win on overlap (matching the
    upstream ordering).
    """
    hmax = maximum_filter(h, size=5)
    seeds = np.array(np.nonzero((h >= hmax) & (h > min_count)))  # (ndim, n_seeds)
    if seeds.shape[1] == 0:
        return seeds
    counts = h[tuple(seeds)]
    order = np.argsort(counts)
    return seeds[:, order]


def cluster_destinations(
    p: Optional[np.ndarray],
    inds: tuple,
    shape0: tuple,
    injected_seeds: Optional[list] = None,
    rpad: int = 20,
    max_size_fraction: float = 0.4,
    min_count: int = 10,
    match_tol: float = 5.0,
):
    """Group destinations into per-cell labels, inheriting IDs from injected seeds.

    Args:
        p: ``(ndim, n_points)`` destinations from :func:`follow_flows`, in the
            ``shape0`` coordinate frame. ``None`` / empty => empty result.
        inds: foreground index tuple aligned with ``p`` (same frame).
        shape0: spatial shape of the (tile-local) label image to produce.
        injected_seeds: optional list of ``(coord, id)`` where ``coord`` is an
            ndim sequence in the ``shape0`` frame and ``id`` is the global
            instance id to inherit. A cluster whose seed lands within
            ``match_tol`` of an injected coord adopts that id.
        rpad / max_size_fraction / min_count: see module docstring (upstream
            ``get_masks_torch`` parameters).
        match_tol: max distance (pixels) for an own-peak to be considered the
            same cell as an injected seed.

    Returns:
        ``(labels, info)``:
          * ``labels``: ``shape0`` int32 image with *temporary* labels 1..K
            (0 = background). Temporary because new cells don't yet have a
            global id -- the caller assigns those.
          * ``info``: dict ``temp_label -> {"inherited": id_or_None,
            "seed": seed_coord (ndim,) in the shape0 frame}``.
    """
    ndim = len(shape0)
    labels = np.zeros(shape0, dtype=np.int32)
    info: dict = {}

    if p is None or p.shape[1] == 0:
        return labels, info

    # Histogram destinations in a padded frame so seed windows never run off edge.
    pt = np.round(p).astype(np.int64) + rpad
    for i in range(ndim):
        np.clip(pt[i], 0, shape0[i] + rpad - 1, out=pt[i])
    padded_shape = tuple(int(s + 2 * rpad) for s in shape0)
    h = np.zeros(padded_shape, dtype=np.int32)
    np.add.at(h, tuple(pt), 1)

    own_seeds = _detect_seeds(h, min_count)  # (ndim, n_own), padded coords

    # Build the combined, deduped seed list. Injected seeds are forced in and
    # carry an inherited id; own peaks near an injected seed are dropped so the
    # cell keeps the inherited id rather than a fresh one.
    seed_coords: list = []  # each ndim tuple in padded coords
    seed_inherit: list = []  # inherited id or None
    injected_seeds = injected_seeds or []
    for coord, gid in injected_seeds:
        pc = tuple(int(round(coord[i])) + rpad for i in range(ndim))
        pc = tuple(min(max(pc[i], 0), padded_shape[i] - 1) for i in range(ndim))
        seed_coords.append(pc)
        seed_inherit.append(gid)

    inj_arr = (
        np.array([c for c in seed_coords], dtype=np.float64) if seed_coords else None
    )
    for k in range(own_seeds.shape[1]):
        c = own_seeds[:, k]
        if inj_arr is not None and len(inj_arr):
            d = np.sqrt(((inj_arr - c[None, :]) ** 2).sum(axis=1)).min()
            if d <= match_tol:
                continue  # same cell as an injected seed -> inherit, skip own peak
        seed_coords.append(tuple(int(v) for v in c))
        seed_inherit.append(None)

    if not seed_coords:
        return labels, info

    # Grow each seed inside an 11^ndim window over the destination histogram,
    # then label every destination pixel by the seed region it falls into.
    n_seeds = len(seed_coords)
    win = 11
    half = win // 2
    M1 = np.zeros(padded_shape, dtype=np.int64)
    grow_struct_size = 3
    for k in range(n_seeds):
        center = seed_coords[k]
        # Clip window to padded bounds and track the center offset within it.
        win_slices = []
        center_in_win = []
        for i in range(ndim):
            lo = max(center[i] - half, 0)
            hi = min(center[i] + half + 1, padded_shape[i])
            win_slices.append(slice(lo, hi))
            center_in_win.append(center[i] - lo)
        win_slices = tuple(win_slices)
        h_slc = h[win_slices]
        sm = np.zeros(h_slc.shape, dtype=np.float32)
        sm[tuple(center_in_win)] = 1
        for _ in range(5):
            sm = maximum_filter(sm, size=grow_struct_size)
            sm = sm * (h_slc > 2)
        region = np.nonzero(sm > 0)
        global_region = tuple(region[i] + win_slices[i].start for i in range(ndim))
        M1[global_region] = k + 1  # temp label (seed index + 1)

    # Each foreground pixel takes the label of its destination location.
    temp_per_pixel = M1[tuple(pt)]
    labels[inds] = temp_per_pixel

    # Remove implausibly large masks (matches upstream max_size_fraction guard).
    uniq, counts = np.unique(labels, return_counts=True)
    big = np.prod(shape0) * max_size_fraction
    for lab, cnt in zip(uniq, counts):
        if lab != 0 and cnt > big:
            labels[labels == lab] = 0

    # Record surviving labels and their seed (in the shape0 frame).
    present = set(int(v) for v in np.unique(labels) if v != 0)
    for k in range(n_seeds):
        temp_label = k + 1
        if temp_label not in present:
            continue
        seed_unpadded = tuple(int(seed_coords[k][i] - rpad) for i in range(ndim))
        info[temp_label] = {"inherited": seed_inherit[k], "seed": seed_unpadded}

    return labels, info


def remove_small_instances(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Zero out labels with fewer than ``min_area`` pixels (numpy port)."""
    if min_area <= 0:
        return labels
    uniq, counts = np.unique(labels, return_counts=True)
    small = uniq[(counts < min_area) & (uniq != 0)]
    if len(small):
        labels = labels.copy()
        labels[np.isin(labels, small)] = 0
    return labels
