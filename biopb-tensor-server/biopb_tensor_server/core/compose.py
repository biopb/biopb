"""Serve a scaled chunk by reducing the full-resolution chunks beneath it.

A scaled chunk reads ``virtual`` source elements and delivers ``virtual //
scale`` of them. Today it reads that extent in one ``get_data`` call and throws
the source pixels away: a cold 1/4 overview of a 1.62 GB scene pulls the whole
scene off disk and caches only its 101 MB of output, so the next read at any
scale -- including full resolution -- pays for the same bytes again.

Composing instead means fetching the full-resolution *transfer chunks* covering
that extent, through the same cache the read path uses, so they land under the
chunk_ids a full-resolution read would ask for. The reduction is what makes it
affordable: a chunk is folded into the output the moment it arrives and then
dropped, so nothing ever holds the extent. Peak memory is one chunk plus the
output, not the 512 MiB ``max_read_block_mb`` allows the extent to reach.

Streaming is only sound where the reduction is exact and order independent, and
this module refuses the rest (:func:`can_compose`) rather than quietly changing
results:

- ``nearest`` is a strided pick, so each chunk contributes the elements whose
  *global* index is a multiple of the scale, wherever its boundaries fall.
- ``area`` on the integer path is a block sum with a single closing divide, and
  integer sums compose in any order. On the float path the means are staged and
  rounded per axis, which does not reassociate; ``streaming_area_plan`` is the
  one place that decides which is which.

Two boundary details decide whether the result is bit-identical rather than
merely close, and both are handled here:

- A chunk's bounds need not be a multiple of the scale. The extent is (it grows
  in whole ``lcm(transfer, scale)`` units), but the chunks tiling it are
  multiples of the transfer extent only -- a 1182-wide grid at scale 4 puts a
  block boundary inside a chunk. Blocks are therefore accumulated by their
  global position, so a block straddling two chunks simply receives from both.
- ``downsample_block`` pads the extent up to a multiple of the scale by edge
  replication and divides by the *full* block size, so a partial block at the
  tensor edge averages the edge element repeated, not just the elements present.
  The chunk that ends at the extent's end carries that same pad, replicating the
  same element for the same reason.

This module is self-contained in the style of ``downsample.py``: no protobuf, no
Arrow, no cache. The caller injects ``fetch``, which is where chunk_ids, the
segment cache and its reference counting live.
"""

from __future__ import annotations

import itertools
import logging
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from biopb_tensor_server.core.downsample import (
    ceil_div,
    normalize_reduction_method,
    streaming_area_plan,
)

logger = logging.getLogger(__name__)

Bounds = Tuple[Tuple[int, ...], Tuple[int, ...]]

# Per-thread composition state, deliberately NOT parameters on
# ``resolve_chunk_data``. That method is the adapter contract -- quoted in
# docs/remote-tensor-cache.md and docs/volume-rendering.md, listed in
# ``_TENSOR_SCOPED_API`` -- and an out-of-tree adapter overriding it with the
# signature those documents show would raise TypeError the first time the server
# passed a new keyword. Both of these are dynamically scoped anyway ("for this
# call and everything under it"), which is what a thread-local is for.
_state = threading.local()


def _depth() -> int:
    return getattr(_state, "depth", 0)


def suppressed() -> bool:
    """Whether this thread has opted out of composing."""
    return getattr(_state, "suppressed", False)


@contextmanager
def without_composition() -> Iterator[None]:
    """Opt this thread out. For a caller that must not populate at scale.

    Precache is the one that matters. It warms scaled chunks only, on purpose:
    a full-resolution warm charges a cache write to a workflow with no re-read
    to pay it back. Composing writes those full-resolution chunks anyway, as a
    side effect of every scaled warm, so it would multiply what one warmed
    chunk costs the cache by the scale product -- past a high-water check that
    runs between chunks and cannot see it coming.
    """
    previous = suppressed()
    _state.suppressed = True
    try:
        yield
    finally:
        _state.suppressed = previous


@contextmanager
def descending() -> Iterator[None]:
    """Mark the inner fetches of a composition, so they cannot compose again.

    The acyclicity guard. Composing adds scaled -> raw edges to the cache's
    promise graph and raw keys wait on nothing, so it stays bipartite -- but
    only while nothing composes from a composed chunk. Correctness does not
    *depend* on this (an inner fetch carries a raw chunk_id, which has nothing to
    compose), which is why per-thread scope is enough: it turns the invariant
    into something a future change trips over rather than something a reader has
    to re-derive.
    """
    _state.depth = _depth() + 1
    try:
        yield
    finally:
        _state.depth -= 1


def enabled_for(cache_manager: object) -> bool:
    """Whether this call should compose, given the cache it would populate.

    The policy rides on the cache manager because that is what composing acts
    on, and because it is already an argument of every ``resolve_chunk_data``
    signature in existence.
    """
    return (
        getattr(cache_manager, "compose_scaled_reads", False)
        and not suppressed()
        and _depth() == 0
    )


def can_compose(
    dtype: str,
    scale_hint: Sequence[int],
    reduction_method: str,
) -> bool:
    """True when a streamed reduction reproduces ``downsample_block`` exactly.

    Not a performance judgement -- a correctness one. The caller must fall back
    to reading the whole extent when this is False.
    """
    method = normalize_reduction_method(reduction_method)
    if method == "nearest":
        return True
    if method != "area":
        # "precompute" names a native on-disk level; there is nothing to reduce.
        return False
    accumulator, _ = streaming_area_plan(
        np.dtype(dtype), tuple(int(scale) for scale in scale_hint)
    )
    return accumulator is not None


def is_grid_aligned(
    start: Sequence[int],
    stop: Sequence[int],
    transfer_chunk: Sequence[int],
    tensor_shape: Sequence[int],
) -> bool:
    """True when the extent is a whole number of transfer chunks.

    ``scaled_virtual_chunk_size`` grows the extent in units of ``lcm(transfer,
    scale)``, so this holds on the ordinary path. It does not hold where that
    unit was clamped to the tensor, nor on the fallback an oversized ``lcm``
    takes (a scale coprime with the transfer extent), and those must read
    directly: a misaligned extent would need chunks reaching outside it, whose
    contributions would have to be clipped -- correct, but no longer the same
    chunk_ids a full-resolution read asks for, which is the entire point.

    A stop landing on the tensor's own end is aligned by definition: that is the
    grid's last chunk, short only because the tensor is.
    """
    for axis, (lo, hi) in enumerate(zip(start, stop, strict=True)):
        grid = max(1, int(transfer_chunk[axis]))
        if int(lo) % grid:
            return False
        if int(hi) % grid and int(hi) != int(tensor_shape[axis]):
            return False
    return True


def covering_chunks(
    start: Sequence[int],
    stop: Sequence[int],
    transfer_chunk: Sequence[int],
    tensor_shape: Sequence[int],
) -> Iterator[Bounds]:
    """The transfer-grid chunks tiling an aligned extent, in row-major order.

    Row-major is deliberate: consecutive chunks share a row band, and on a
    format whose rows are contiguous the second and later chunks of a band come
    off the page cache the first one populated.
    """
    axes = []
    for axis, (lo, hi) in enumerate(zip(start, stop, strict=True)):
        grid = max(1, int(transfer_chunk[axis]))
        axes.append((range(int(lo), int(hi), grid), grid, int(tensor_shape[axis])))
    for origin in itertools.product(*(steps for steps, _, _ in axes)):
        chunk_start = tuple(int(value) for value in origin)
        chunk_stop = tuple(
            min(int(value) + grid, shape)
            for value, (_, grid, shape) in zip(origin, axes, strict=True)
        )
        yield chunk_start, chunk_stop


def _edge_padded(
    chunk: np.ndarray,
    local_start: Sequence[int],
    extent_shape: Sequence[int],
    scale_hint: Sequence[int],
) -> np.ndarray:
    """Carry ``downsample_block``'s edge pad on the chunk that ends the extent.

    Padding the trailing chunk replicates the extent's own last element along
    that axis -- it is the same element -- so the padded block sums match those
    taken over a padded extent, corners included.
    """
    pad_width: List[Tuple[int, int]] = []
    padded = False
    for axis, size in enumerate(chunk.shape):
        scale = max(1, int(scale_hint[axis]))
        extent = int(extent_shape[axis])
        short = ceil_div(extent, scale) * scale - extent
        at_end = int(local_start[axis]) + int(size) == extent
        if short and at_end:
            pad_width.append((0, short))
            padded = True
        else:
            pad_width.append((0, 0))
    if not padded:
        return chunk
    return np.pad(chunk, pad_width, mode="edge")


def _block_breaks(local_lo: int, size: int, scale: int) -> List[int]:
    """Segment starts for ``np.add.reduceat`` along one axis.

    A chunk need not begin on a block boundary, so the first segment can be a
    block's tail. Every segment after it is a whole block, except a trailing
    partial one the next chunk finishes.
    """
    first = (-local_lo) % scale
    breaks = [] if first == 0 else [0]
    breaks.extend(range(first, size, scale))
    return breaks


def _accumulate_area(
    accumulator: np.ndarray,
    chunk: np.ndarray,
    local_start: Sequence[int],
    scale_hint: Sequence[int],
) -> None:
    """Add one chunk's block sums into the output accumulator, in place.

    ``reduceat`` accumulates at ``dtype`` off the narrow input, so the widening
    never materializes: casting the chunk up front instead would allocate and
    touch a second, wider copy of every chunk -- 2.4 GB across a 1.2 GB scene,
    to produce the same sums. The first reduced axis does the widening; later
    axes are already at accumulator width.
    """
    work = chunk
    reduced = False
    for axis in range(work.ndim):
        scale = max(1, int(scale_hint[axis]))
        if scale == 1:
            continue
        breaks = _block_breaks(int(local_start[axis]), work.shape[axis], scale)
        work = np.add.reduceat(work, breaks, axis=axis, dtype=accumulator.dtype)
        reduced = True
    if not reduced:
        # scale 1 on every axis: no block to sum, but the accumulator still has
        # to receive this chunk at its own width.
        work = work.astype(accumulator.dtype, copy=False)
    offsets = [
        int(local_start[axis]) // max(1, int(scale_hint[axis]))
        for axis in range(work.ndim)
    ]
    target = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, work.shape, strict=True)
    )
    accumulator[target] += work


def _assign_nearest(
    output: np.ndarray,
    chunk: np.ndarray,
    local_start: Sequence[int],
    scale_hint: Sequence[int],
) -> None:
    """Write the elements this chunk contributes to a strided pick."""
    picks, offsets = [], []
    for axis in range(chunk.ndim):
        scale = max(1, int(scale_hint[axis]))
        lo = int(local_start[axis])
        picks.append(slice((-lo) % scale, None, scale))
        offsets.append(ceil_div(lo, scale))
    picked = chunk[tuple(picks)]
    target = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, picked.shape, strict=True)
    )
    output[target] = picked


def compose_scaled_chunk(
    fetch: Callable[[Tuple[int, ...], Tuple[int, ...]], np.ndarray],
    start: Sequence[int],
    stop: Sequence[int],
    tensor_shape: Sequence[int],
    transfer_chunk: Sequence[int],
    scale_hint: Sequence[int],
    reduction_method: str,
    dtype: str,
) -> Optional[np.ndarray]:
    """Reduce the extent by fetching and folding one chunk at a time.

    ``fetch`` takes chunk bounds and returns that chunk's array; caching, and
    the reference the cache hands out for it, are its business. Each result is
    folded in and dropped before the next is asked for, so the extent is never
    resident.

    Returns None when the request cannot be composed exactly -- an unsupported
    reduction, or an extent off the transfer grid -- leaving the caller to read
    the extent directly. Never returns an approximation.
    """
    method = normalize_reduction_method(reduction_method)
    if not can_compose(dtype, scale_hint, method):
        return None
    if not is_grid_aligned(start, stop, transfer_chunk, tensor_shape):
        return None

    extent_shape = tuple(int(hi) - int(lo) for lo, hi in zip(start, stop, strict=True))
    output_shape = tuple(
        ceil_div(size, max(1, int(scale)))
        for size, scale in zip(extent_shape, scale_hint, strict=True)
    )
    source_dtype = np.dtype(dtype)

    block_size = 0
    if method == "nearest":
        output = np.empty(output_shape, dtype=source_dtype)
    else:
        accumulator_dtype, block_size = streaming_area_plan(
            source_dtype, tuple(int(scale) for scale in scale_hint)
        )
        output = np.zeros(output_shape, dtype=accumulator_dtype)

    fetched = 0
    for chunk_start, chunk_stop in covering_chunks(
        start, stop, transfer_chunk, tensor_shape
    ):
        chunk = fetch(chunk_start, chunk_stop)
        local_start = tuple(
            int(cs) - int(lo) for cs, lo in zip(chunk_start, start, strict=True)
        )
        if method == "nearest":
            _assign_nearest(output, chunk, local_start, scale_hint)
        else:
            _accumulate_area(
                output,
                _edge_padded(chunk, local_start, extent_shape, scale_hint),
                local_start,
                scale_hint,
            )
        fetched += 1
        del chunk

    logger.debug("compose_scaled_chunk: %d chunks -> %s", fetched, output_shape)

    if method == "nearest":
        return output
    # The closing arithmetic downsample_block applies to its own block sums.
    info = np.iinfo(source_dtype)
    return np.clip(np.round(output / block_size), info.min, info.max).astype(
        source_dtype
    )
