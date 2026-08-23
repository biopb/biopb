"""Reduce an extent by streaming it in tiles, never holding it whole.

A scaled read materialises ``extent / scale`` bytes but must *touch* the whole
extent to get there. Reading it in one call makes peak residency the extent --
1159 MiB for one ND2 plane -- which is what used to bound how large a scaled
chunk could be planned. Streaming decouples the two: the working set is one
tile, and the extent may be as large as the tensor.

**Tiles land on block boundaries**, which is what makes this a five-line fold
rather than a streaming reduction. Every output element then depends only on
elements inside one tile, so reducing each tile with
:func:`~.downsample.downsample_block` and writing the result into place is
*identical* to reducing the whole extent -- for every method and every dtype,
including the float ``area`` whose staged means neither commute nor associate,
and the non-dyadic scales that take the same float path. There is no accumulator
to carry across tiles, and no reduction that has to decline and read whole.

The one thing it costs is that :func:`streaming_unit` may not choose freely: a
tile is rounded up to a whole number of blocks. For the dyadic scales and
power-of-two grids that reach this in practice, that rounding is a no-op.

**The tile is the transfer grid**, one rule for every backend. That is the
simple thing rather than the fast thing, deliberately: what a backend would
rather have differs by layout, is worth multiples, and belongs in a phase-2
per-adapter override rather than in a layout branch here. Both known directions
are measured and waiting:

- A **contiguous** reader -- an ND2 frame, an MRC mmap -- wants a full-width row
  band, because a tile out of a wide frame reads strided and a band is one
  sequential run. Worth 1852 -> 592 ms warm on an ND2 at area/8.
- A **chunked** store whose blocks exceed the transfer grid wants its tile
  floored at that block, because otherwise every block is fetched and decoded
  once per tile overlapping it. Measured at 260 ms against 125 ms read whole on
  a 4096-chunk zarr, and -- the sharp one -- 1726 ms against 165 ms on a tiled
  8192^2 OME-TIFF page, whose ``aszarr(chunkmode="page")`` block is the *entire
  page*: 16 tiles, 16 whole-page decodes.

Neither is expressible without the adapter declaring what its reads are granular
in, which is what phase 2 adds. Until then a page-mode reader pays that floor's
absence, knowingly: the cost is bounded, proportional to plane size, and visible
in the benchmark rather than silent.

Self-contained in the style of ``downsample.py``: no protobuf, no Arrow, no
cache. The caller injects ``fetch``.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterator, Sequence, Tuple

import numpy as np

from .downsample import ceil_div, downsample_block

Bounds = Tuple[Tuple[int, ...], Tuple[int, ...]]


def _whole_blocks(size: int, scale: int) -> int:
    """``size`` rounded up to a whole number of reduction blocks, never below one."""
    scale = max(1, int(scale))
    return max(1, ceil_div(max(1, int(size)), scale)) * scale


def streaming_unit(
    extent: Sequence[int],
    transfer_chunk: Sequence[int],
    scale_hint: Sequence[int],
) -> Tuple[int, ...]:
    """The region one ``fetch`` covers: the transfer grid, on block boundaries.

    Rounded up to a whole number of reduction blocks, which is what
    :func:`stream_reduce` requires, and clamped to the extent -- an axis the tile
    already spans is covered by a single tile, so there is no boundary on it to
    place and rounding would only shave a sliver off the end to read separately.
    """
    return tuple(
        min(_whole_blocks(size, scale), int(dim))
        for size, scale, dim in zip(transfer_chunk, scale_hint, extent, strict=True)
    )


def covering_units(
    start: Sequence[int],
    stop: Sequence[int],
    unit: Sequence[int],
    tensor_shape: Sequence[int],
) -> Iterator[Bounds]:
    """The units tiling ``[start, stop)``, in row-major order.

    Row-major is deliberate: consecutive units share a row band, so on a format
    whose rows are contiguous the second and later units of a band come off the
    page cache the first one populated.

    Units are laid from ``start``, not from the tensor origin, so an extent that
    does not begin on the grid still tiles exactly, and every unit's offset
    within the extent stays a multiple of the unit -- which is what makes it a
    multiple of the scale too.
    """
    axes = []
    for axis, (lo, hi) in enumerate(zip(start, stop, strict=True)):
        step = max(1, int(unit[axis]))
        axes.append((range(int(lo), int(hi), step), step, int(hi)))
    for origin in itertools.product(*(steps for steps, _, _ in axes)):
        unit_start = tuple(int(value) for value in origin)
        unit_stop = tuple(
            min(int(value) + step, end)
            for value, (_, step, end) in zip(origin, axes, strict=True)
        )
        yield unit_start, unit_stop


def stream_reduce(
    fetch: Callable[[Tuple[int, ...], Tuple[int, ...]], np.ndarray],
    start: Sequence[int],
    stop: Sequence[int],
    tensor_shape: Sequence[int],
    unit: Sequence[int],
    scale_hint: Sequence[int],
    reduction_method: str,
    dtype: str,
) -> np.ndarray:
    """Reduce ``[start, stop)`` by streaming it through ``fetch``, unit by unit.

    Byte-identical to ``downsample_block`` over the whole extent. ``unit`` must
    be a whole number of reduction blocks on every axis -- use
    :func:`streaming_unit` -- because a unit that split a block would reduce its
    halves independently and produce a different pixel.
    """
    extent_shape = tuple(int(hi) - int(lo) for lo, hi in zip(start, stop, strict=True))
    for axis, (size, scale) in enumerate(zip(unit, scale_hint, strict=True)):
        if int(size) % max(1, int(scale)) and int(size) != extent_shape[axis]:
            raise ValueError(
                f"streaming unit {tuple(unit)} splits a reduction block on axis "
                f"{axis} at scale {tuple(scale_hint)}"
            )

    output = np.empty(
        tuple(
            ceil_div(size, max(1, int(scale)))
            for size, scale in zip(extent_shape, scale_hint, strict=True)
        ),
        dtype=np.dtype(dtype),
    )

    for unit_start, unit_stop in covering_units(start, stop, unit, tensor_shape):
        reduced = downsample_block(
            fetch(unit_start, unit_stop), scale_hint, reduction_method
        )
        # The unit's offset within the extent is a multiple of the unit, and the
        # unit is a multiple of the scale, so this divides exactly.
        offsets = [
            (int(us) - int(lo)) // max(1, int(scale))
            for us, lo, scale in zip(unit_start, start, scale_hint, strict=True)
        ]
        output[
            tuple(
                slice(offset, offset + size)
                for offset, size in zip(offsets, reduced.shape, strict=True)
            )
        ] = reduced

    return output
