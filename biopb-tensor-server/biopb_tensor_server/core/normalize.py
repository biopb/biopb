"""Canonical axis order: the permutation helpers the adapter base applies (#596).

Adapters advertise ``dim_labels`` in whatever order their upstream reader emits.
The adapter base turns that into a **wire guarantee** so no consumer has to
re-derive "which axis is Y/X/Z/S" for itself:

    Z, Y, X and S appear last, in that relative order; every other axis -- T, C,
    and any unrecognized label -- keeps its relative order ahead of them.

The rule lives in :func:`biopb_tensor_server.core.axes.canonical_permutation`.
:class:`~biopb_tensor_server.core.adapter_base.TensorAdapter` applies it: a leaf
adapter implements the ``_native_*`` hooks in its reader's own order, and the
public ``get_tensor_descriptor`` / ``get_data`` / ``read_block_shape`` /
``get_decimated_data`` / ``get_native_pyramid_levels`` present them canonical.
Everything built on those -- the read planner, scaled and streamed reads, the
pyramid advertisement -- therefore works in canonical order with no translation
of its own, and a cached segment holds exactly what a client is served.

``chunk_id`` is minted from the canonical geometry, so a cached chunk written
before an adapter was normalized would be served in the wrong order; a change of
this shape bumps ``CHUNK_SEMANTICS_EPOCH`` (``core.chunk``).

An order this server does not own is refused, not permuted
-----------------------------------------------------------

Permuting reads works only where the server owns the whole read path. Two seams
do not, and both **validate** instead, reporting through
:func:`biopb_tensor_server.core.axes.noncanonical_order`:

**Writes** (#596 Decision 3). A writable source carries the uploader's own
declared order, with ``physical_scale`` and ``chunk_shape`` aligned to it;
silently permuting reads would desynchronize them from what ``put_chunk`` wrote.
``serving.upload_manager`` refuses the order at ``add_tensor``.

**The remote proxy.** Its upstream owns the order in exactly the same sense: that
server mints the chunk_ids, plans the reads (biopb/biopb#295) and sizes the grid.
So ``adapters.remote_tensor`` opts out (``_normalizable_axes = False``) and
refuses a non-canonical upstream at its read boundary instead.

Refusing rather than repairing keeps the guarantee *unconditional* while leaving
nothing stateful to go stale: the test is re-run against the descriptor in hand
on every open, so an upstream that upgrades is picked up immediately and one that
has not cannot be silently mis-served.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, List, Optional, Sequence, Tuple

from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.axes import canonical_permutation

logger = logging.getLogger(__name__)


def permute(values: Sequence[Any], perm: Tuple[int, ...]) -> List[Any]:
    """Reorder ``values`` (native order) into canonical order."""
    return [values[p] for p in perm]


def invert(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """The inverse permutation: applying it undoes ``perm``.

    ``permute(canonical_vector, invert(perm))`` is the native-order vector, so
    the same helper serves both directions.
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return tuple(inverse)


def to_canonical(values: Sequence[Any], perm: Optional[Tuple[int, ...]]) -> List[Any]:
    """:func:`permute`, but identity for ``perm=None`` and for ``values`` of some
    other rank."""
    if perm is None or len(values) != len(perm):
        return list(values)
    return permute(values, perm)


def to_native(values: Sequence[Any], perm: Optional[Tuple[int, ...]]) -> List[Any]:
    """The inverse of :func:`to_canonical`: canonical-order ``values`` to native."""
    return to_canonical(values, None if perm is None else invert(perm))


def permute_repeated(field, perm: Tuple[int, ...]) -> None:
    """Permute a repeated proto field in place, iff its length matches the rank.

    A per-axis field that is absent (``chunk_shape`` is documented as optionally
    empty) or of some other rank is left alone rather than guessed at.
    """
    if len(field) == len(perm):
        field[:] = permute(list(field), perm)


def permute_level(level: Any, perm: Tuple[int, ...]) -> None:
    """Permute a ``PyramidLevel``'s per-axis fields in place."""
    permute_repeated(level.shape, perm)
    permute_repeated(level.scale_hint, perm)


def permute_bounds(bounds: ChunkBounds, perm: Tuple[int, ...]) -> ChunkBounds:
    """A copy of ``bounds`` with its axes reordered by ``perm``."""
    if len(bounds.start) != len(perm) or len(bounds.stop) != len(perm):
        return bounds
    return ChunkBounds(
        start=permute(list(bounds.start), perm),
        stop=permute(list(bounds.stop), perm),
    )


def permute_descriptor(
    desc: TensorDescriptor, perm: Tuple[int, ...]
) -> TensorDescriptor:
    """A copy of ``desc`` with every per-axis field reordered by ``perm``.

    Covers the whole client-visible geometry, including the fields a read plan
    fills in (``slice_hint`` / ``scale_hint``) and the pyramid levels, each of
    which carries its own per-axis ``shape`` and ``scale_hint``.
    """
    out = TensorDescriptor()
    out.CopyFrom(desc)
    permute_repeated(out.shape, perm)
    permute_repeated(out.chunk_shape, perm)
    permute_repeated(out.dim_labels, perm)
    permute_repeated(out.scale_hint, perm)
    permute_repeated(out.physical_scale, perm)
    permute_repeated(out.physical_unit, perm)
    if out.HasField("slice_hint"):
        permute_repeated(out.slice_hint.start, perm)
        permute_repeated(out.slice_hint.stop, perm)
    for level in out.pyramid:
        permute_level(level, perm)
    return out


def permute_entry(entry: Any, perm: Tuple[int, ...]) -> Any:
    """A copy of a ``TensorEntry`` with its axes reordered by ``perm``."""
    return replace(
        entry,
        dim_labels=tuple(to_canonical(entry.dim_labels, perm)),
        shape=tuple(to_canonical(entry.shape, perm)),
    )


def descriptor_permutation(desc: Any) -> Optional[Tuple[int, ...]]:
    """The permutation the labels of ``desc`` (a descriptor or a ``TensorEntry``)
    imply, or None for identity."""
    return canonical_permutation(desc.dim_labels, desc.shape)


def log_reordering(source_id: str, entries: Sequence[Any]) -> None:
    """Say, at INFO, which of a source's tensors are served reordered.

    Reordering is a visible behavior change -- it is what a client sees, and it
    costs the zero-copy read for that tensor -- so the only evidence of it should
    not be the transposed data itself. Takes the **native** entries.
    """
    for d in entries:
        perm = descriptor_permutation(d)
        if perm is not None:
            native = list(d.dim_labels)
            logger.info(
                "axis normalization: serving %s as %s (stored as %s)",
                d.array_id or source_id,
                permute(native, perm),
                native,
            )
