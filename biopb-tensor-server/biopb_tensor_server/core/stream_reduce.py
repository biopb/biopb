"""Reduce an extent by streaming it in units, never holding it whole.

A scaled read materialises ``extent / scale`` bytes but must *touch* the whole
extent to get there. Reading it in one call makes peak residency the extent --
1159 MiB for one ND2 plane -- which is what used to bound how large a scaled
chunk could be planned. Streaming decouples the two: the working set is one unit,
and the extent may be as large as the tensor.

**Units land on block boundaries**, which is what makes this a five-line fold
rather than a streaming reduction. Every output element then depends only on
elements inside one unit, so reducing each unit with
:func:`~.downsample.downsample_block` and writing the result into place is
*identical* to reducing the whole extent -- for every method and every dtype,
including the float ``area`` whose staged means neither commute nor associate,
and the non-dyadic scales that take the same float path. There is no accumulator
to carry across units, and no reduction that has to decline and read whole.

The one thing it costs is that ``streaming_unit`` may not choose freely: a unit
is rounded up to a whole number of blocks. For the dyadic scales and
power-of-two grids that reach this in practice, that rounding is a no-op.

**The unit's shape is the backend's**, and the two layouts want opposite shapes:

- A **chunked** store -- zarr, HDF5 -- reads a whole stored chunk whatever part
  of it you ask for, so the unit is the transfer grid floored at that chunk.
  Warm on an 8192^2 OME-Zarr, per-unit against one wide read: 66-72 ms vs
  94-122 ms at 512^2 chunks, 63-67 ms vs 117-119 ms at 2048^2. Without the floor
  the case that breaks is a store whose chunks exceed the transfer target, where
  the grid is *finer* than the stored block and each block is fetched and
  decompressed once per unit overlapping it: 260 ms against 125 ms wide, which
  the floor restores to 81 ms. Row bands are what must not be used here -- a band
  shorter than a stored chunk re-reads it once per band, measured at 7.1x.
- A **contiguous** store -- an ND2 frame, a TIFF page, an mmap -- addresses any
  sub-region directly, so all that matters is that a unit be one sequential run.
  That makes it a full-width row band. Square tiles out of a 14234-wide frame
  read strided and cost 30% on a cold plane (3160 ms against 2432 ms) for a warm
  gain that banding delivers anyway.

So the rule is one idea -- *a unit is the largest sequential run the backend can
deliver* -- and the layout decides what that is. Getting it backwards is what
both of the measured disasters above have in common.

Self-contained in the style of ``downsample.py``: no protobuf, no Arrow, no
cache. The caller injects ``fetch``.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterator, Optional, Sequence, Tuple

import numpy as np

from .downsample import ceil_div, downsample_block

Bounds = Tuple[Tuple[int, ...], Tuple[int, ...]]

# Band budget for a contiguous backend, whose unit is not set by any stored
# block. A cache size, deliberately not a config knob: banding is loop tiling
# against L3 as much as a residency measure, worth ~2x on data already in RAM
# with byte-identical arithmetic, because reducing a whole extent makes two
# passes over it while a band is still in cache when the reduction touches it.
# Swept on contiguous uint16 at scale 4/8, the ratio against the unbanded
# reduction peaks at 8 MiB and falls off both sides: under ~2 MiB per-band
# overhead bites, over ~16 MiB the band stops fitting in cache. The right value
# is a property of the cache hierarchy, not an operator preference.
_CONTIGUOUS_BAND_BYTES = 8 * 1024 * 1024


def _whole_blocks(size: int, scale: int, *, up: bool) -> int:
    """``size`` rounded to a whole number of reduction blocks, never below one.

    The direction differs by branch and it matters both ways: a chunked unit
    rounds *up* so it cannot fall back below the stored block the floor just
    raised it to, while a band rounds *down* so it stays inside its cache budget.
    """
    scale = max(1, int(scale))
    size = max(1, int(size))
    blocks = ceil_div(size, scale) if up else size // scale
    return max(1, blocks) * scale


def streaming_unit(
    extent: Sequence[int],
    transfer_chunk: Sequence[int],
    read_block: Optional[Sequence[int]],
    scale_hint: Sequence[int],
    itemsize: int,
) -> Tuple[int, ...]:
    """The region one ``fetch`` covers: the largest sequential run of this backend.

    ``read_block`` is the smallest region the backend reads without
    amplification -- a zarr or HDF5 chunk -- or ``None`` where any sub-region is
    directly addressable, which is what selects the shape:

    - **Chunked**: the transfer grid, floored at the stored block so a unit
      boundary never lands inside one. The floor binds only where a stored block
      exceeds the transfer target; everywhere else the grid is already a whole
      multiple of it and this is the identity.
    - **Contiguous**: a full-width row band deep enough to fill
      ``_CONTIGUOUS_BAND_BYTES``. Full width because a narrower unit reads
      strided; ``ndim - 2`` is the row axis because every layout here is
      row-major in its trailing pair.

    Either way the result is a whole number of reduction blocks, which is what
    :func:`stream_reduce` requires -- rounded up for a chunked unit so it cannot
    drop back below the stored block, down for a band so it stays inside its
    cache budget.
    """
    extent = [int(dim) for dim in extent]
    round_up = read_block is not None

    if read_block is not None:
        unit = [max(1, int(size)) for size in transfer_chunk]
        for axis, block in enumerate(read_block):
            block = max(1, int(block))
            if block > unit[axis]:
                unit[axis] = max(ceil_div(block, unit[axis]) * unit[axis], block)
    elif len(extent) < 2:
        unit = list(extent)
    else:
        row_axis = len(extent) - 2
        row_bytes = itemsize
        for axis, size in enumerate(extent):
            if axis != row_axis:
                row_bytes *= size
        unit = list(extent)
        if row_bytes > 0:
            unit[row_axis] = max(1, _CONTIGUOUS_BAND_BYTES // row_bytes)

    # An axis the unit already spans is left alone: it is covered by a single
    # unit, so there is no boundary on it to land on a block. Rounding it would
    # only shave a sliver off the end and read that separately.
    return tuple(
        dim if size >= dim else min(_whole_blocks(size, scale, up=round_up), dim)
        for size, scale, dim in zip(unit, scale_hint, extent, strict=True)
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
