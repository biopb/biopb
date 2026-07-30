"""Semantic axis-label vocabulary and resolution for the data plane.

One source of truth for "which dimension is T/Z/C/Y/X". Three resolvers classify
labels through :func:`canonical_axis`, so the synonym vocabulary can never drift
between them:

- :func:`build_axis_map` -- the render/client-facing resolver, with a positional
  fallback that assigns *every* unmapped x/y/z from the trailing axes.
- :func:`labeled_axis_index` -- label-only (no positional fallback), used by the
  pyramid helpers in ``core.chunk`` where an unlabeled leading axis (possibly T/C)
  must never be downsampled as if it were depth.
- :func:`canonical_permutation` -- the wire-contract resolver (biopb/biopb#596):
  the permutation that reorders an adapter's native axes into the canonical
  trailing order the server advertises. See ``core.normalize``.

Mirrors the frontend ``buildAxisMap`` in ``@biopb/tensor-flight-client``.
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

    Label-only: unlike :func:`build_axis_map` there is no positional fallback, so a
    hit means the axis is actually named (possibly by synonym). Duplicate matches
    resolve to the first occurrence.
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
#   RECOGNIZED axes appear in canonical relative order [..., Z, Y, X, S];
#   unrecognized labels hold their positions.
#
# It is NOT "every axis is labeled". An all-``dimN`` tensor (plain zarr / HDF5)
# has no recognized axis, so its permutation is the identity and the positional
# fallback in :func:`build_axis_map` keeps doing the work -- relabeling ``dimN``
# to z/y/x would promote a documented *guess* into a wire *assertion*, and that
# assertion is wrong for e.g. a [y, x, c] array stored unlabeled.
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
    - no axis is recognized at all (the unlabeled ``dimN`` case -- Decision 1 of
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
        return None  # nothing recognized -- unlabeled store, identity by design

    # Stable sort: the rank-0 leading group (T, C, dimN, ...) keeps its relative
    # order, and the recognized axes land in Z, Y, X, S order behind it.
    perm = tuple(sorted(range(len(labels)), key=lambda i: ranks[i]))
    return None if perm == tuple(range(len(labels))) else perm


def build_axis_map(dim_labels: list[str]) -> dict[str, Optional[int]]:
    """Map semantic axis names to dimension indices.

    Mirrors frontend buildAxisMap() in tensor-flight-client.
    """
    result: dict[str, Optional[int]] = {
        "t": None,
        "z": None,
        "c": None,
        "y": None,
        "x": None,
    }

    unassigned = []
    for i, label in enumerate(dim_labels):
        canonical = canonical_axis(label)
        if canonical is None:
            unassigned.append(i)
        else:
            result[canonical] = i

    # Positional fallback for unmapped axes: last → X, second-last → Y,
    # third-last → Z.
    if result["x"] is None and unassigned:
        result["x"] = unassigned.pop()  # last
    if result["y"] is None and unassigned:
        result["y"] = unassigned.pop()  # second-last
    if result["z"] is None and unassigned:
        result["z"] = unassigned.pop()  # third-last

    return result
