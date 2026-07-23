"""Semantic axis-label vocabulary and resolution for the data plane.

One source of truth for "which dimension is T/Z/C/Y/X". Two resolvers classify
labels through :func:`canonical_axis`, so the synonym vocabulary can never drift
between them:

- :func:`build_axis_map` -- the render/client-facing resolver, with a positional
  fallback that assigns *every* unmapped x/y/z from the trailing axes.
- :func:`labeled_axis_index` -- label-only (no positional fallback), used by the
  pyramid helpers in ``core.chunk`` where an unlabeled leading axis (possibly T/C)
  must never be downsampled as if it were depth.

Mirrors the frontend ``buildAxisMap`` in ``@biopb/tensor-flight-client``.
"""

from __future__ import annotations

from typing import Optional, Sequence

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
