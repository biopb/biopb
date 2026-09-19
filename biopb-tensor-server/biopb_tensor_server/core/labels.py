"""The shape of a label tensor's name (biopb/biopb#1059).

A label set is a tensor of its image, addressed by the NGFF layout::

    <image array_id>/labels/<name>[/<level>]

so its within-source field is ``[<image field>/]labels/<name>``. ``name`` is
slash-free, which is what lets the field be split unambiguously: the **last**
``labels`` segment that has a name after it is the one, whatever the image
field before it contains, and anything after the name is a native pyramid
level. This module is pure string handling; what a set *is* lives in
:mod:`biopb_tensor_server.adapters.labels`, and how a source answers for one in
:class:`~biopb_tensor_server.core.adapter_base.SourceAdapter`.

Design: ``biopb-tensor-server/docs/label-tensors.md``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

__all__ = [
    "LABELS_SEGMENT",
    "RESERVED_LABEL_PREFIX",
    "LabelField",
    "is_reserved_label_name",
    "label_field",
    "split_label_field",
]

#: The path segment that marks a label set under its image.
LABELS_SEGMENT = "labels"

#: A set whose name starts with this is server-owned (``@ome`` for the set
#: rasterized from an OME-TIFF's masks): readable, never created or deleted
#: by a client. The same convention as the reserved ROI set names.
RESERVED_LABEL_PREFIX = "@"


class LabelField(NamedTuple):
    """A label field taken apart: which image, which set, which level."""

    image_field: str  #: the image's within-source field; "" for a sole tensor
    name: str  #: the set's name, slash-free
    level: Optional[str]  #: a native pyramid level path under the set, if any

    @property
    def set_field(self) -> str:
        """The set's own within-source field, level stripped."""
        return label_field(self.image_field, self.name)


def label_field(image_field: Optional[str], name: str) -> str:
    """The within-source field of set *name* on the image at *image_field*."""
    if image_field:
        return f"{image_field}/{LABELS_SEGMENT}/{name}"
    return f"{LABELS_SEGMENT}/{name}"


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


def is_reserved_label_name(name: str) -> bool:
    return name.startswith(RESERVED_LABEL_PREFIX)
