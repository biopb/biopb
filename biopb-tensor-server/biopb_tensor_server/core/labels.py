"""The shape of a label tensor's name, and what it must span (biopb/biopb#1059).

A label set is a tensor of its image, addressed by the NGFF layout::

    <image array_id>/labels/<name>[/<level>]

so its within-source field is ``[<image field>/]labels/<name>``. ``name`` is
slash-free, which is what lets the field be split unambiguously: the **last**
``labels`` segment that has a name after it is the one, whatever the image
field before it contains, and anything after the name is a native pyramid
level. This module is pure: strings and shapes. What a set *is* lives in
:mod:`biopb_tensor_server.adapters.labels`, and how a source answers for one in
:class:`~biopb_tensor_server.core.adapter_base.SourceAdapter`.

Design: ``biopb-tensor-server/docs/label-tensors.md``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from biopb_tensor_server.core.axes import canonical_axis

__all__ = [
    "LABELS_SEGMENT",
    "LabelField",
    "extent_mismatch",
    "join_fields",
    "label_field",
    "split_label_field",
]

#: The path segment that marks a label set under its image.
LABELS_SEGMENT = "labels"


class LabelField(NamedTuple):
    """A label field taken apart: which image, which set, which level."""

    image_field: str  #: the image's within-source field; "" for a sole tensor
    name: str  #: the set's name, slash-free
    level: Optional[str]  #: a native pyramid level path under the set, if any

    @property
    def set_field(self) -> str:
        """The set's own within-source field, level stripped."""
        return label_field(self.image_field, self.name)


def join_fields(*segments: Optional[str]) -> str:
    """Path segments joined with ``/``, empty ones dropped -- the one way an
    ``array_id`` or a within-source field is composed here."""
    return "/".join(s for s in segments if s)


def label_field(image_field: Optional[str], name: str) -> str:
    """The within-source field of set *name* on the image at *image_field*."""
    return join_fields(image_field, LABELS_SEGMENT, name)


def split_label_field(field: Optional[str]) -> Optional[LabelField]:
    """Take a within-source field apart, or ``None`` if it names no label set.

    ``"labels/nuclei"`` -> ``("", "nuclei", None)``;
    ``"A/1/labels/nuclei/2"`` -> ``("A/1", "nuclei", "2")``. A trailing
    ``labels`` with nothing after it is not a set.
    """
    if not field:
        return None
    parts = field.split("/")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == LABELS_SEGMENT and parts[i + 1]:
            level = "/".join(parts[i + 2 :]) or None
            return LabelField("/".join(parts[:i]), parts[i + 1], level)
    return None


def _axis_key(label: str) -> str:
    """What two axes must agree on: the canonical name where the label has
    one, else the label itself (an unrecognized axis matches only itself)."""
    return canonical_axis(label) or str(label).lower()


def extent_mismatch(
    label_labels: Sequence[str],
    label_shape: Sequence[int],
    image_labels: Sequence[str],
    image_shape: Sequence[int],
) -> Optional[str]:
    """Why a set of *label_shape* over *label_labels* does not span the image,
    or None when it does.

    The extent rule (design, "Extent"): a set's axes are the image's canonical
    axes with the channel axis dropped, each at the image's full length, so a
    label pixel and its image pixel share an index. Both sides are compared
    in canonical order -- the caller hands in normalized descriptors -- by
    canonical axis name, so ``"Z"`` and ``"depth"`` agree.
    """
    expected_axes, expected_shape = [], []
    for label, size in zip(image_labels, image_shape, strict=True):
        if canonical_axis(label) != "c":
            expected_axes.append(_axis_key(label))
            expected_shape.append(int(size))
    if [_axis_key(label) for label in label_labels] != expected_axes:
        return (
            f"axes {list(label_labels)} do not match the image's non-channel "
            f"axes {expected_axes}"
        )
    if [int(s) for s in label_shape] != expected_shape:
        return f"shape {list(label_shape)} does not match the image's {expected_shape}"
    return None
