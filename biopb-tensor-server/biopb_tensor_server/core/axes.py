"""Semantic axis-label vocabulary and resolution for the data plane.

One source of truth for "which dimension is T/Z/C/Y/X". Three resolvers classify
labels through :func:`canonical_axis`, so the synonym vocabulary can never drift
between them:

- :func:`canonical_permutation` -- the wire-contract resolver (biopb/biopb#596):
  the permutation that reorders an adapter's native axes into the canonical
  trailing order the server advertises. See ``core.normalize``.
- :func:`labeled_axis_index` -- label-only (no positional fallback), used by the
  pyramid helpers in ``core.chunk`` where an unlabeled leading axis (possibly T/C)
  must never be downsampled as if it were depth.
- :func:`plane_axes` -- the render-facing resolver: which axes the HTTP sidecar
  displays. A *consumer* of the wire contract rather than a party to it, so it
  reads the plane off the canonical order positionally.

:func:`noncanonical_order` states the same rule as a refusal rather than a
transform, for the seams that validate an order they do not own (an upload's
declared order, a remote upstream's advertised one).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Recognized axis labels (matched case-insensitively).
AXIS_T_LABELS = {"t", "time", "frame", "frames"}
AXIS_Z_LABELS = {"z", "depth", "plane", "planes", "slice"}
AXIS_C_LABELS = {"c", "channel", "channels", "band", "bands"}
AXIS_Y_LABELS = {"y", "height", "row", "rows"}
AXIS_X_LABELS = {"x", "width", "col", "cols", "column", "columns"}
# Interleaved RGB(A) samples axis. aicsimageio labels the samples axis of a
# photometric-RGB image "S" (dims "TCZYXS"); its size is 3 (RGB) or 4 (RGBA).
AXIS_S_LABELS = {"s", "samples"}

# Canonical axis name -> its synonym set, in classification order.
_AXIS_LABEL_SETS = (
    ("t", AXIS_T_LABELS),
    ("z", AXIS_Z_LABELS),
    ("c", AXIS_C_LABELS),
    ("y", AXIS_Y_LABELS),
    ("x", AXIS_X_LABELS),
)


def canonical_axis(label: str) -> Optional[str]:
    """Canonical axis name (``t``/``z``/``c``/``y``/``x``) for a dim label, or None."""
    low = str(label).lower()
    for name, labels in _AXIS_LABEL_SETS:
        if low in labels:
            return name
    return None


def labeled_axis_index(dim_labels: Sequence[str], axis: str) -> Optional[int]:
    """Index of the first axis whose label maps to ``axis``, or None.

    Label-only: there is no positional fallback, so a hit means the axis is
    actually named (possibly by synonym). Duplicate matches resolve to the first
    occurrence.
    """
    for i, label in enumerate(dim_labels):
        if canonical_axis(label) == axis:
            return i
    return None


def samples_axis(dim_labels: list[str], shape: tuple[int, ...]) -> Optional[int]:
    """Index of an interleaved RGB(A) samples axis, or ``None``.

    Detected by *label* (``S`` / ``samples``) gated on a size of 3 or 4, so a
    size-3 channel or Z axis is never mistaken for color. This axis holds the
    color components of one pixel and must be composited into RGB, not selected
    one-plane-at-a-time like T/Z/C.
    """
    for i, label in enumerate(dim_labels):
        if label.lower() in AXIS_S_LABELS and i < len(shape) and shape[i] in (3, 4):
            return i
    return None


# --- canonical wire order (biopb/biopb#596) ---------------------------------
# The trailing axis order the data plane advertises: [..., Z, Y, X, S]. Axes that
# are not one of these four -- T, C, and every unrecognized label -- form the
# leading group and keep their relative order. The rank is the sort key; the
# leading group shares rank 0 and a *stable* sort therefore preserves it.
_CANONICAL_RANK = {"z": 1, "y": 2, "x": 3}
_SAMPLES_RANK = 4

# What the guarantee actually says, and what it deliberately does not:
#
#   Z, Y, X and S appear last, in that relative order; every other axis -- T, C,
#   and any unrecognized label -- keeps its relative order ahead of them.
#
# Note what the second clause does NOT say: an unrecognized label keeps its
# relative order, not its *index*. [z, dimq, y, x] normalizes to
# [dimq, z, y, x] -- dimq did not move relative to anything, but a trailing axis
# moved out from in front of it. Only "recognized" in the [..., Z, Y, X, S]
# sense counts as trailing: T and C classify through the same vocabulary but have
# no canonical place, so they ride in the leading group with the unlabeled.
#
# It is NOT "every axis is labeled". An all-``dimN`` tensor (plain zarr / HDF5)
# has no axis with a canonical place, so its permutation is the identity and the
# consumers' positional reading (:func:`plane_axes` here, ``_resolve_axes`` in
# biopb-mcp) keeps doing the work -- relabeling ``dimN`` to z/y/x would promote a
# documented *guess* into a wire *assertion*, and that assertion is wrong for
# e.g. a [y, x, c] array stored unlabeled.
CANONICAL_TRAILING_ORDER = ("z", "y", "x", "s")


def canonical_permutation(
    dim_labels: Sequence[str], shape: Sequence[int]
) -> Optional[Tuple[int, ...]]:
    """Permutation reordering native axes into canonical ``[..., Z, Y, X, S]``.

    Returns ``perm`` such that ``normalized[i] = native[perm[i]]`` -- i.e. the
    argument to ``np.transpose`` that normalizes an array, and the index vector
    that permutes a per-axis descriptor field. Returns ``None`` for "identity",
    which is both the already-canonical case and every case the labels are not
    trustworthy enough to act on. ``None`` rather than ``tuple(range(ndim))`` so
    callers can skip the whole normalization path (and its array copy) on the
    overwhelmingly common no-op.

    Fail-safe by construction, matching the posture ``serving.renderer`` already
    takes toward adapter-supplied labels: anything ambiguous degrades to identity
    rather than reordering pixels on a guess. Specifically ``None`` is returned
    when

    - the labels are absent or their count does not match ``shape``;
    - no axis has a canonical place (the unlabeled ``dimN`` case -- Decision 1 of
      biopb/biopb#596: out of scope, and provably a no-op here);
    - two axes claim the same canonical role (duplicate ``y``, say), so there is
      no one right answer;
    - an axis is *labeled* ``S``/``samples`` but fails the size-3/4 gate
      :func:`samples_axis` applies, which means the label cannot be believed.
    """
    if not dim_labels or len(dim_labels) != len(shape):
        return None

    labels = list(dim_labels)
    s_idx = samples_axis(labels, tuple(shape))

    ranks: list[int] = []
    for i, label in enumerate(labels):
        if i == s_idx:
            ranks.append(_SAMPLES_RANK)
            continue
        if str(label).lower() in AXIS_S_LABELS:
            # Labeled samples but not 3/4 deep: samples_axis refused it, so the
            # labels are not describing what they claim. Do not reorder.
            logger.debug(
                "axes: %r labels a samples axis that fails the size gate; "
                "leaving axis order untouched",
                labels,
            )
            return None
        ranks.append(_CANONICAL_RANK.get(canonical_axis(label) or "", 0))

    trailing = [r for r in ranks if r]
    if len(set(trailing)) != len(trailing):
        logger.debug(
            "axes: %r claims a canonical axis twice; leaving axis order untouched",
            labels,
        )
        return None
    if not trailing:
        return None  # nothing placed -- unlabeled store, identity by design

    # Stable sort: the rank-0 leading group (T, C, dimN, ...) keeps its relative
    # order, and the placed axes land in Z, Y, X, S order behind it.
    perm = tuple(sorted(range(len(labels)), key=lambda i: ranks[i]))
    return None if perm == tuple(range(len(labels))) else perm


def noncanonical_order(
    dim_labels: Sequence[str], shape: Sequence[int]
) -> Optional[str]:
    """One sentence naming a non-canonical order and the order to use, else None.

    The same rule as :func:`canonical_permutation`, phrased as a refusal instead
    of a transform. Two seams validate an axis order rather than permuting it,
    because in both the order is *owned by another party* who has aligned the
    rest of their state to it (biopb/biopb#596):

    - an upload's declared order (``serving.upload_manager.create_source``), whose
      ``physical_scale`` / ``chunk_shape`` arrive aligned to it and whose
      ``put_chunk`` writes in it;
    - a remote upstream's advertised order (``adapters.remote_tensor``), whose
      server mints the chunk_ids, plans the reads (biopb/biopb#295) and sizes the
      grid.

    Sharing the wording keeps the two refusals reporting the same fact the same
    way; each caller raises its own error type around it.
    """
    perm = canonical_permutation(dim_labels, shape)
    if perm is None:
        return None
    labels = list(dim_labels)
    return (
        f"dim_labels {labels} are not in canonical [..., Z, Y, X, S] order "
        f"(expected {[labels[p] for p in perm]})"
    )


def plane_axes(
    dim_labels: Sequence[str], shape: Sequence[int]
) -> Tuple[int, int, Optional[int]]:
    """``(y_idx, x_idx, s_idx)`` for the 2-D plane the render path displays.

    The consumer-side resolver for the order :func:`canonical_permutation`
    guarantees: Y and X are the last two axes, sitting behind an interleaved
    RGB(A) samples axis when there is one. Labels are read for exactly one
    question -- is the *trailing* axis samples -- because that is the only thing
    position cannot answer. :func:`samples_axis` supplies the size-3/4 gate, so a
    3-channel ``[C, Y, X]`` stack is never composited as false color; an ``S``
    found anywhere but last is ignored, since that is not an order this server
    serves.

    Requires ``len(shape) >= 2`` (callers guard, or accept the negative index a
    sub-2-D tensor yields, exactly as the positional fallback here always did).

    The three indices are distinct and in range **by construction**, which is
    what lets this stay short where the label-matching resolver it replaced
    needed explicit collision handling: nothing is derived from where a label
    *sits*, so a duplicate ``y``, an ``S`` shadowing ``X``, or an all-unknown
    label set cannot produce a repeated transpose axis. They simply leave the
    plane where the wire order says it is.

    Mirrors ``buildAxisMap`` in the frontend ``@biopb/tensor-flight-client``,
    which resolves the same plane the same way (it keeps a label lookup for the
    T/C slider axes, which the canonical order does not position).
    """
    ndim = len(shape)
    trailing_samples = samples_axis(list(dim_labels), tuple(shape)) == ndim - 1
    s_idx = ndim - 1 if (ndim >= 3 and trailing_samples) else None
    x_idx = ndim - 1 if s_idx is None else ndim - 2
    return x_idx - 1, x_idx, s_idx
