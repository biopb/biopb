"""Raster-order, chunk-wise instance segmentation that fits in bounded memory.

This is the pure orchestration layer for processing a (potentially
larger-than-memory) image in overlapping tiles and emitting a single, globally
consistent label mask -- without ever holding the whole image or the whole
output. It has no torch / tensor-server / GPU dependencies: model inference and
output upload are supplied as callbacks, so the linking logic can be
unit-tested on synthetic destination fields.

Cross-chunk ID consistency (the hard part)
------------------------------------------
Every foreground pixel's flow integrates to a *destination* ~ its cell center
(see :mod:`biopb_image_base.dynamics_local`). Two pixels are the same cell iff they share a
destination. We keep a small, pruned set of already-discovered
``(global_destination, id)`` pairs. When a tile is clustered, the in-bounds
known destinations are *injected* as seeds carrying their IDs; any cluster that
lands on an injected seed inherits that ID. So a cell straddling a border is
discovered once, and every later tile that sees it reuses the same ID -- no
re-reading of saved labels, no union-find, no rewrite pass.

This works regardless of which tile's core contains the cell center, provided
``margin >= max cell diameter`` so that every tile whose core touches a cell
contains the *whole* cell and therefore computes the same destination.

Memory: the known-destination set is pruned to a band near the processing
frontier (one entry per boundary cell in that band, never per cell in the
image), so it stays bounded for arbitrarily large images.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Optional

import numpy as np

from biopb_image_base import dynamics_local


def uniform_core(dim: int, target: int) -> int:
    """Pick a core size that **evenly divides** ``dim`` and is closest to ``target``.

    The output tensor must be registered with a uniform chunk shape (the tensor
    server rejects ragged template chunks), and it is cleanest if processing
    cores line up with those chunks. Returns ``dim`` itself when there is no
    interior divisor (e.g. a prime dimension) -- the caller then processes it as
    a single tile on that axis.
    """
    if dim <= target:
        return dim
    best = dim
    best_gap = abs(dim - target)
    for c in range(1, int(dim**0.5) + 1):
        if dim % c == 0:
            for cand in (c, dim // c):
                gap = abs(cand - target)
                if gap < best_gap or (gap == best_gap and cand > best):
                    best, best_gap = cand, gap
    return best


def _iter_core_origins(full_shape: tuple, core_shape: tuple):
    """Yield core start coordinates in raster order (outer axis varies slowest)."""
    axis_starts = [
        range(0, full_shape[i], core_shape[i]) for i in range(len(full_shape))
    ]
    yield from product(*axis_starts)


def _tile_bounds(core_start, core_shape, full_shape, margin):
    """Core bounds and the padded tile bounds (clipped to the image) for a core."""
    ndim = len(full_shape)
    core_stop = tuple(
        min(core_start[i] + core_shape[i], full_shape[i]) for i in range(ndim)
    )
    tile_start = tuple(max(core_start[i] - margin, 0) for i in range(ndim))
    tile_stop = tuple(min(core_stop[i] + margin, full_shape[i]) for i in range(ndim))
    return core_stop, tile_start, tile_stop


def stitch_lazy_segmentation(
    full_shape: tuple,
    core_shape: tuple,
    margin: int,
    compute_chunk: Callable[[tuple, tuple], tuple],
    write_core: Callable[[tuple, tuple, np.ndarray], None],
    *,
    min_area: int = 0,
    edge_tol: int = 2,
    cluster_kwargs: Optional[dict] = None,
    observer: Optional[Callable[[dict], None]] = None,
) -> int:
    """Segment an image tile-by-tile with globally consistent instance IDs.

    Args:
        full_shape: spatial shape of the whole image, e.g. ``(Y, X)``.
        core_shape: non-overlapping tile (core) size per axis.
        margin: overlap added on every side of each core. **Must be >= the
            maximum expected cell diameter** for border IDs to be consistent.
        compute_chunk: ``(tile_start, tile_stop) -> (inds, p)``. Runs the model
            (or, in tests, returns synthetic data) on the tile and returns the
            foreground index tuple ``inds`` and destination array ``p``
            (``(ndim, n_points)``), both in **tile-local** coordinates. May
            return ``(empty_inds, None)`` for an empty tile.
        write_core: ``(core_start, core_stop, labels) -> None``. Uploads the
            cropped core label array for the given global bounds.
        min_area: drop instances smaller than this many pixels (per tile).
        edge_tol: pixels whose destination lands within this distance of an
            *internal* (clamped, non-image) tile edge are dropped before
            clustering. They are slivers of a cell centered outside this tile;
            keeping them would create a spurious seed clamped to the boundary.
            Such slivers are always in the margin (never the core), so dropping
            them is safe.
        cluster_kwargs: extra kwargs forwarded to
            :func:`biopb_image_base.dynamics_local.cluster_destinations`
            (e.g. ``rpad``, ``min_count``, ``match_tol``).
        observer: optional debug hook called after each core with
            ``{"core_start", "known_size", "next_id"}`` (used by tests to assert
            the known-destination set stays bounded).

    Returns:
        Total number of instance IDs assigned.
    """
    ndim = len(full_shape)
    cluster_kwargs = dict(cluster_kwargs or {})

    next_id = 1
    # Known cell destinations near the frontier: list of (global_coord tuple, id).
    known: list = []

    for core_start in _iter_core_origins(full_shape, core_shape):
        core_stop, tile_start, tile_stop = _tile_bounds(
            core_start, core_shape, full_shape, margin
        )
        tile_shape = tuple(tile_stop[i] - tile_start[i] for i in range(ndim))

        inds, p = compute_chunk(tile_start, tile_stop)

        # Drop slivers of cells centered outside this tile: their flow clamps to
        # an internal tile edge and would seed a spurious instance.
        if p is not None and p.shape[1] > 0 and edge_tol > 0:
            keep = np.ones(p.shape[1], dtype=bool)
            for i in range(ndim):
                if tile_start[i] > 0:  # internal (top/left) edge
                    keep &= p[i] > edge_tol
                if tile_stop[i] < full_shape[i]:  # internal (bottom/right) edge
                    keep &= p[i] < tile_shape[i] - 1 - edge_tol
            if not keep.all():
                inds = tuple(a[keep] for a in inds)
                p = p[:, keep]

        # Inject known destinations that fall inside this tile (tile-local coords).
        injected = []
        for gcoord, gid in known:
            if all(tile_start[i] <= gcoord[i] < tile_stop[i] for i in range(ndim)):
                injected.append(
                    (tuple(gcoord[i] - tile_start[i] for i in range(ndim)), gid)
                )

        labels, info = dynamics_local.cluster_destinations(
            p, inds, tile_shape, injected_seeds=injected, **cluster_kwargs
        )

        if min_area > 0:
            labels = dynamics_local.remove_small_instances(labels, min_area)
            present = set(int(v) for v in np.unique(labels) if v != 0)
            info = {t: m for t, m in info.items() if t in present}

        # Resolve temp labels -> global IDs (inherit or assign fresh + register).
        if info:
            max_temp = max(info)
            remap = np.zeros(max_temp + 1, dtype=np.int32)
            for temp_label, meta in info.items():
                if meta["inherited"] is not None:
                    gid = int(meta["inherited"])
                else:
                    gid = next_id
                    next_id += 1
                    seed_global = tuple(
                        int(meta["seed"][i] + tile_start[i]) for i in range(ndim)
                    )
                    known.append((seed_global, gid))
                remap[temp_label] = gid
            labels = remap[labels]
        # else: labels is all zeros; nothing to remap.

        # Crop the tile down to its core and upload.
        crop = tuple(
            slice(core_start[i] - tile_start[i], core_stop[i] - tile_start[i])
            for i in range(ndim)
        )
        core_labels = labels[crop]
        write_core(core_start, core_stop, core_labels)

        # Prune destinations no future (raster-order) tile can reach. Future
        # tiles start at >= core_start[0] - margin on the outer axis, so anything
        # above that line is unreachable from here on.
        keep_threshold = core_start[0] - margin
        if known:
            known = _prune_known(known, keep_threshold)

        if observer is not None:
            observer(
                {"core_start": core_start, "known_size": len(known), "next_id": next_id}
            )

    return next_id - 1


def _prune_known(known: list, keep_threshold: int) -> list:
    """Drop known destinations above ``keep_threshold`` on the outer axis.

    Pure helper so the pruning rule can be unit-tested in isolation. A
    destination is kept iff a future raster-order tile could still contain it,
    i.e. its outer-axis coordinate is at or below the frontier minus the margin.
    """
    return [kv for kv in known if kv[0][0] >= keep_threshold]
