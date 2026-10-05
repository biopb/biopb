"""The shape of a label tensor's name, and what it must span (biopb/biopb#1059).

A label set is a tensor of its image, addressed under a **marked** segment::

    <image array_id>/@labels/<name>[/<level>]

so its within-source field is ``[<image field>/]@labels/<name>``. ``name`` is
slash-free, which is what lets the field be split unambiguously: the **last**
``@labels`` segment that has a name after it is the one, whatever the image
field before it contains, and anything after the name is a native pyramid
level.

**The marker is on the wire id only; the NGFF group on disk stays ``labels/``**
(``adapters.labels.native_label_sets`` opens it by that name, and never through
this constant). Marking the segment is what keeps an attached tensor's id off a
native one -- a scene of the user's own file can plausibly be called ``labels``
and cannot plausibly be called ``@labels`` -- the same argument ``@ome`` is
chosen by (``RESERVED_PREFIX``).

This module is pure: strings and shapes. What a set *is* lives in
:mod:`biopb_tensor_server.adapters.labels`, and how a source answers for one in
:class:`~biopb_tensor_server.core.adapter_base.SourceAdapter`.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

from biopb_tensor_server.core.attached import MARKER as RESERVED_PREFIX
from biopb_tensor_server.core.axes import canonical_axis, samples_axis

__all__ = [
    "LABELS_SEGMENT",
    "RESERVED_PREFIX",
    "LabelField",
    "extent_mismatch",
    "join_fields",
    "label_extent",
    "label_field",
    "last_named_segment",
    "split_label_field",
]

#: The segment that marks a label set under its image, in a **wire id**. The
#: NGFF group on disk is ``labels/`` and is opened by that literal name: the two
#: were one string until the marker, and nothing derives a path from this.
LABELS_SEGMENT = f"{RESERVED_PREFIX}labels"

#: A name under this prefix is server-owned, in the spirit of the ``@ome`` ROI
#: set: ``@labels/@ome`` is the set rasterized from an OME-TIFF's masks, and a
#: native NGFF set keeps whatever name the file gave it. Clients read a
#: reserved set; they never upload or delete one.
#:
#: The same prefix marks a *segment* (:data:`LABELS_SEGMENT`), which is why an
#: uploaded field may not open with it (``_writable.unsafe_field_name``): one
#: prefix, one meaning -- this name is the server's, not yours. Re-exported
#: from :mod:`~biopb_tensor_server.core.attached` (its ``MARKER``) rather than
#: redeclared, so the two modules cannot disagree on what the prefix is.


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


def last_named_segment(parts: Sequence[str], segment: str) -> Optional[int]:
    """Index of the last *segment* in *parts* that has something after it.

    The one right-to-left scan both a live split and the v1->v2 marker
    migration (``serving.metadata_db._mark_label_segment``, which runs it
    against the pre-marker ``"labels"`` literal) need: whatever the field
    before it contains, the last occurrence with a name after it is the one
    that names a set, never an earlier segment that merely shares the word.
    """
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == segment and parts[i + 1]:
            return i
    return None


def split_label_field(field: Optional[str]) -> Optional[LabelField]:
    """Take a within-source field apart, or ``None`` if it names no label set.

    ``"@labels/nuclei"`` -> ``("", "nuclei", None)``;
    ``"A/1/@labels/nuclei/2"`` -> ``("A/1", "nuclei", "2")``. A trailing
    ``@labels`` with nothing after it is not a set.
    """
    if not field:
        return None
    parts = field.split("/")
    i = last_named_segment(parts, LABELS_SEGMENT)
    if i is None:
        return None
    level = "/".join(parts[i + 2 :]) or None
    return LabelField("/".join(parts[:i]), parts[i + 1], level)


def _axis_key(label: str) -> str:
    """What two axes must agree on: the canonical name where the label has
    one, else the label itself (an unrecognized axis matches only itself)."""
    return canonical_axis(label) or str(label).lower()


def label_extent(
    image_labels: Sequence[str], image_shape: Sequence[int]
) -> Tuple[List[str], List[int]]:
    """The axes and lengths a set of this image has.

    **The extent rule**: the image's own axes at the image's lengths, except that
    a channel axis is a singleton and an interleaved RGB(A) samples axis is
    dropped. So a set's axis *j* is the image's axis *j* for every axis a client
    treats as one, and a label pixel and its image pixel share an index. The
    channel axis is a singleton (not dropped) so the set has the image's rank and
    lines up with it by position; the samples axis is dropped (not a singleton)
    because it is a pixel's colour components, which a mask has none of, and a
    trailing singleton would shift the spatial axes of a right-aligned viewer.

    The rule has one form. A set without a channel axis (a native NGFF group,
    which the spec lets omit ``c``, or a sidecar an older server wrote) does not
    span its image and is not listed. Axes come back as the image spells them,
    which for the normalized descriptors the callers hand in is canonical.
    """
    if len(image_labels) != len(image_shape):
        raise ValueError(
            f"image_labels {list(image_labels)} and image_shape "
            f"{list(image_shape)} have different lengths"
        )
    samples = samples_axis([str(x) for x in image_labels], tuple(image_shape))
    kept = [i for i in range(len(image_labels)) if i != samples]
    return (
        [str(image_labels[i]) for i in kept],
        [
            1 if canonical_axis(image_labels[i]) == "c" else int(image_shape[i])
            for i in kept
        ],
    )


def extent_mismatch(
    label_labels: Sequence[str],
    label_shape: Sequence[int],
    image_labels: Sequence[str],
    image_shape: Sequence[int],
) -> Optional[str]:
    """Why a set of *label_shape* over *label_labels* does not span the image,
    or None when it does.

    The rule is :func:`label_extent`. Axes are matched by canonical name, so
    ``"Z"`` and ``"depth"`` agree.
    """
    want_labels, want_shape = label_extent(image_labels, image_shape)
    axes = [_axis_key(x) for x in want_labels]
    if [_axis_key(x) for x in label_labels] != axes:
        return (
            f"axes {list(label_labels)} do not match the image's {axes} (the "
            "channel axis a singleton, an RGB samples axis left out)"
        )
    shape = [int(s) for s in label_shape]
    if shape != want_shape:
        return f"shape {shape} does not match the image's {want_shape}"
    return None
