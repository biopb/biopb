"""Serving a scaled read out of the full-resolution chunks the cache holds.

The chunks under a scaled read are ordinary cache entries, so where they are all
warm the reduction can be sourced from them rather than decoded from the store a
second time -- see :func:`cache_sourced_units` for what that buys and when it
declines.

Module functions rather than adapter methods: the only adapter state they need
is ``content_version`` (one argument), and as methods every subclass and every
delegating wrapper inherited a seam it never meant to offer.
"""

from __future__ import annotations

from itertools import islice
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.chunk import cache_key_for_chunk_id, mint_chunk_id
from biopb_tensor_server.core.chunk_batch import unpack_chunk_view
from biopb_tensor_server.core.stream_reduce import covering_units, streaming_unit

if TYPE_CHECKING:
    from biopb_tensor_server.cache import CacheManager


class BorrowedUnit:
    """The cache entry the unit last fetched is a view of, if it is one.

    A cache-sourced ``fetch`` hands back either a view onto a segment mapping or
    a copy in its own buffer (:func:`cache_sourced_units`), and only the view
    constrains what the caller may do with the reduction of it. This is what says
    which, and what frees the entry once the unit has been reduced.
    """

    __slots__ = ("_cache", "key")

    def __init__(self, cache_manager: Optional[CacheManager]) -> None:
        self._cache = cache_manager
        self.key: Optional[bytes] = None

    def hold(self, key: bytes) -> None:
        self.key = key

    def release(self) -> None:
        if self.key is not None:
            self._cache.release(self.key)
            self.key = None


def _acquired_chunk_view(
    cache_manager: CacheManager,
    key: bytes,
    expected_shape: Tuple[int, ...],
    dtype: np.dtype,
) -> Optional[np.ndarray]:
    """A view of the cached chunk at *key*, still acquired, or None.

    ``touch=False``: these reads stand in for one the adapter could do itself,
    and one coarse extent covers every chunk of the source -- crediting them all
    would let a zoomed-out read decide what stays cached
    (:meth:`CacheBackend.try_acquire`).

    None where the entry went between the probe and here, or where its stored
    shape or dtype is not the one this extent expects -- a stale key collision,
    not something to read blind. The entry is released before returning None, so
    only a view hands back a held reference.
    """
    entry = cache_manager.try_acquire(key, touch=False)
    if entry is None:
        return None
    block = unpack_chunk_view(entry.data) if entry.data is not None else None
    if block is None or block.shape != expected_shape or block.dtype != dtype:
        cache_manager.release(key)
        return None
    return block


def cache_sourced_units(
    cache_manager: Optional[CacheManager],
    descriptor: TensorDescriptor,
    content_version: Optional[bytes],
    start: Tuple[int, ...],
    stop: Tuple[int, ...],
    unit: Tuple[int, ...],
    scale_hint: Tuple[int, ...],
    reduction_method: str,
    read: Callable[..., np.ndarray],
    transfer: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Callable[..., np.ndarray], BorrowedUnit]:
    """``(unit, fetch, borrowed)`` for serving this extent out of the cache.

    ``fetch`` may hand back a view onto a cache segment rather than a copy,
    and ``borrowed`` is both how the caller tells (``.key``, per unit) and
    how it frees the last one (``.release()``, once the reduction is done).
    Declining hands back the caller's own ``unit`` and ``read`` with nothing
    held, so the caller has one shape to drive either way.

    The full-resolution chunks under a scaled read are ordinary cache
    entries: a scaled chunk's bounds snap to ``transfer x scale``
    (:func:`~.chunk.scaled_virtual_chunk_size`), so its extent tiles exactly
    into the absolute transfer grid the unscaled read plan mints chunk_ids
    on, and a unit is a whole number of those chunks. Where every one of them
    is already there -- warmed by full-resolution reads, or by an earlier
    pass -- units are assembled from them instead of decoded from the source
    a second time.

    What it trades is two read paths, not a decode: a segment is mmap'd, so
    a cached chunk is a view and the only cost is one memcpy, where a store
    hands its bytes back through a file read that is then materialised. On
    an 8192^2 uint16 zarr chunked at 4096, one scale-16 virtual chunk, page
    cache warm: 99 -> 9 ms for ``nearest``, 141 -> 56 ms for ``area``, and
    the codec barely moves either (docs/fused-scaling-path.md, sec 9). An
    adapter whose ``get_data`` is *already* an mmap crop has nothing to gain,
    and those are the ones that report no :attr:`~.adapter_base.TensorAdapter.read_block_shape` and
    implement :meth:`~.adapter_base.TensorAdapter.get_decimated_data`, so they never arrive here.

    **The unit follows the reduction's cost model.** A pick needs none of
    what it skips, so the smallest unit that still tiles into whole chunks
    wins -- which is also what lets it be *borrowed* rather than copied
    (:func:`borrow_cached_unit`): a unit that is one chunk is one mapping
    the pick reads straight out of. An averaging reduction runs
    ``prod(scale)`` strided adds per unit whatever its size, so more units is
    the same bytes over more, smaller numpy calls (195 ms against 80); it
    keeps the caller's unit, which a source read would have held anyway, so
    residency cannot regress.

    **On by default** (``cache.source_scaled_reads``), and the timings above
    are the smaller half of why. A scaled read is usually a pretext for work
    on real pixels -- a viewer that just fit a plane to its window is about
    to zoom into it, an analysis that sized a volume is about to read it --
    so sourcing the coarse read from the full-resolution chunks leaves those
    chunks' segment pages resident, and the full-res read that follows is a
    warm mmap read instead of a cold source decode. That is worth paying for
    even where the coarse read itself gains little, which is why this does
    not wait for a per-store measurement to be enabled.

    Declines, leaving the streamed source read exactly as it was, when there
    is no cache, when the extent or the unit does not sit on the chunk grid
    (a non-dyadic ``scale_hint`` rounds a 512 unit up to 513), or when any
    chunk is missing -- short-circuiting on the first, so a cold extent pays
    one index lookup rather than one per chunk.

    A probe is a decision, never a promise: an entry can be evicted between
    it and the read, so the fetch falls back to the source per unit.
    """
    declined = (unit, read, BorrowedUnit(None))
    if cache_manager is None or not cache_manager.source_scaled_reads:
        return declined

    shape = tuple(int(dim) for dim in descriptor.shape)
    extent = tuple(hi - lo for lo, hi in zip(start, stop, strict=True))
    # ``nearest`` sizes its unit at one chunk, so the copy that unit would
    # take moves prod(scale) times the bytes the pick then reads -- borrow
    # the entry's own mapping instead (:func:`borrow_cached_unit`). ``area``
    # reads every byte anyway and its edge units go through the padding and
    # strided-add kernels, so it keeps the buffer.
    borrows = reduction_method == "nearest"
    if borrows:
        # No read block: that floor speaks for a source read this path does
        # not make. The rounding to whole reduction blocks stays, so a chunk
        # below one block still yields a unit that does not split one.
        unit = streaming_unit(extent, transfer, None, scale_hint)
    # The extent has to sit on the chunk grid at both ends, or the bounds
    # derived here name a chunk the plan never minted -- a partial chunk
    # ending mid-grid keys differently from the whole one that is cached, so
    # the probe would miss on every entry rather than crop. A plan-minted
    # scaled chunk always qualifies (it snaps to transfer x scale, and its
    # last chunk on an axis ends at the tensor), so this only declines an
    # extent no read plan asks for.
    if any(lo % size for lo, size in zip(start, transfer, strict=True)):
        return declined
    if any(
        hi % size and hi != dim
        for hi, size, dim in zip(stop, transfer, shape, strict=True)
    ):
        return declined
    if any(
        span % size and span != whole
        for span, size, whole in zip(unit, transfer, extent, strict=True)
    ):
        return declined
    # Keyed by chunk origin and kept: the fetch below needs the same keys per
    # unit, and minting one is ~4.6 us against the ~45 us the chunk's memcpy
    # costs -- at prod(scale) chunks per scaled read, re-minting them was a
    # tenth of this path.
    keys: Dict[Tuple[int, ...], bytes] = {}
    for chunk_start, key in chunk_cache_keys(
        descriptor, content_version, start, stop, transfer
    ):
        if not cache_manager.contains(key):
            return declined
        keys[chunk_start] = key

    # One buffer for every unit, not one per unit: a fresh 32 MiB
    # destination measured 31.7 ms against 8.3 ms reused, because untouched
    # pages fault interleaved with the copy out of the mapping. The first
    # unit is the largest (``covering_units`` clamps only the last on an
    # axis), so it sizes the buffer and the rest slice it. Safe because
    # ``stream_reduce`` writes each unit out before asking for the next.
    buffer: Optional[np.ndarray] = None
    dtype = np.dtype(descriptor.dtype)
    borrowed = BorrowedUnit(cache_manager)

    def fetch(unit_start, unit_stop):
        nonlocal buffer
        # The previous unit has been reduced and written out by now -- the
        # same fact that makes the buffer reusable makes its entry free.
        borrowed.release()
        span = tuple(
            int(hi) - int(lo) for lo, hi in zip(unit_start, unit_stop, strict=True)
        )
        if borrows:
            lent = borrow_cached_unit(
                cache_manager, keys, unit_start, unit_stop, transfer, shape, dtype
            )
            if lent is not None:
                key, block = lent
                borrowed.hold(key)
                return block
        if buffer is None or any(
            size > held for size, held in zip(span, buffer.shape, strict=True)
        ):
            buffer = np.empty(span, dtype=dtype)
        out = buffer[tuple(slice(0, size) for size in span)]
        if assemble_from_cache(
            cache_manager, keys, unit_start, unit_stop, transfer, shape, out
        ):
            return out
        # Evicted since the probe. One source read of the unit, which is what
        # this extent would have cost all along.
        return read(unit_start, unit_stop)

    return unit, fetch, borrowed


def chunk_cache_keys(
    descriptor: TensorDescriptor,
    content_version: Optional[bytes],
    start: Sequence[int],
    stop: Sequence[int],
    transfer: Sequence[int],
) -> Iterator[Tuple[Tuple[int, ...], bytes]]:
    """``(chunk_start, cache_key)`` for the full-resolution chunks tiling
    ``[start, stop)``.

    Minted exactly as ``_get_read_plan`` mints an unscaled endpoint -- same
    absolute grid, same ``array_id``, same ``content_version`` header --
    because a key that differs by one byte is a probe that never hits.
    """
    shape = tuple(int(dim) for dim in descriptor.shape)
    for chunk_start, chunk_stop in covering_units(start, stop, transfer, shape):
        chunk_id = mint_chunk_id(
            descriptor.array_id,
            ChunkBounds(start=list(chunk_start), stop=list(chunk_stop)),
            content_version=content_version,
        )
        yield chunk_start, cache_key_for_chunk_id(chunk_id)


def borrow_cached_unit(
    cache_manager: CacheManager,
    keys: Dict[Tuple[int, ...], bytes],
    start: Sequence[int],
    stop: Sequence[int],
    transfer: Sequence[int],
    shape: Sequence[int],
    dtype: np.dtype,
) -> Optional[Tuple[bytes, np.ndarray]]:
    """``(key, view)`` where one cached chunk *is* this unit, else None.

    The unit is then not copied at all: the reduction reads the segment's
    mapping directly, which for ``nearest`` is the whole point -- the pick
    touches one byte in ``prod(scale)`` of what the copy would have moved.

    The entry is returned still acquired, because the view is only valid
    while it is: the caller releases it once the reduction has consumed the
    unit, and never lets a view of it escape (see
    :meth:`~.adapter_base.TensorAdapter.get_scaled_data`, contract 4).

    None where the unit is more than one chunk, or where
    :func:`_acquired_chunk_view` declines the entry. The caller falls back to
    :func:`assemble_from_cache`.
    """
    chunks = list(islice(covering_units(start, stop, transfer, shape), 2))
    if len(chunks) != 1:
        return None
    key = keys.get(chunks[0][0])
    if key is None:
        return None
    span = tuple(int(hi) - int(lo) for lo, hi in zip(start, stop, strict=True))
    block = _acquired_chunk_view(cache_manager, key, span, dtype)
    if block is None:
        return None
    return key, block


def assemble_from_cache(
    cache_manager: CacheManager,
    keys: Dict[Tuple[int, ...], bytes],
    start: Sequence[int],
    stop: Sequence[int],
    transfer: Sequence[int],
    shape: Sequence[int],
    out: np.ndarray,
) -> bool:
    """Fill *out* with ``[start, stop)`` from the cache, chunk by chunk.

    False means "not all of it is here": an eviction since the probe, or an
    entry :func:`_acquired_chunk_view` declines. The caller reads the unit
    from the source instead -- *out* may have been partly written by then,
    and is overwritten wholesale by that read.

    One copy per chunk, straight into *out*, and the entry is held for
    exactly that copy: the view is onto the segment's mapping, so the copy
    reads file-backed pages and must not outlive its reference.
    """
    for chunk_start, chunk_stop in covering_units(start, stop, transfer, shape):
        key = keys.get(chunk_start)
        if key is None:
            return False
        where = tuple(
            slice(int(lo) - int(origin), int(hi) - int(origin))
            for lo, hi, origin in zip(chunk_start, chunk_stop, start, strict=True)
        )
        target = out[where]
        block = _acquired_chunk_view(cache_manager, key, target.shape, out.dtype)
        if block is None:
            return False
        try:
            target[...] = block
        finally:
            cache_manager.release(key)
    return True
