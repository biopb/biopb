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
    "LabelForm",
    "extent_mismatch",
    "join_fields",
    "label_extent",
    "label_forms",
    "label_image_axes",
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


class LabelForm(NamedTuple):
    """One legal extent for a set of an image: its axes, lengths and the image
    axis each one indexes."""

    labels: List[str]
    shape: List[int]
    image_axes: List[int]


def label_forms(
    image_labels: Sequence[str], image_shape: Sequence[int]
) -> List[LabelForm]:
    """The extents a set of this image may have, preferred first.

    **The extent rule**: the image's own axes at the image's lengths, except that
    a channel axis is a singleton and an interleaved RGB(A) samples axis is
    dropped. So a set's axis *j* is the image's axis *j* for every axis a client
    treats as one, and a label pixel and its image pixel share an index. The
    channel axis is a singleton (not dropped) so the set has the image's rank and
    lines up with it by position; the samples axis is dropped (not a singleton)
    because it is a pixel's colour components, which a mask has none of, and a
    trailing singleton would shift the spatial axes of a right-aligned viewer.

    **The earlier rule** -- the channel axis dropped altogether -- is listed
    second, and only a read accepts it: a native NGFF group (a file we cannot
    rewrite, and one the spec allows to omit ``c``) and a sidecar an earlier server
    wrote keep being served. A new set must have the preferred form
    (``extent_mismatch(..., allow_earlier=False)``). It is the same list when the
    image has no channel axis. Axes come back as the image spells them,
    which for the normalized descriptors the callers hand in is canonical.
    """
    if len(image_labels) != len(image_shape):
        raise ValueError(
            f"image_labels {list(image_labels)} and image_shape "
            f"{list(image_shape)} have different lengths"
        )
    samples = samples_axis([str(x) for x in image_labels], tuple(image_shape))
    kept = [i for i in range(len(image_labels)) if i != samples]

    is_channel = [canonical_axis(label) == "c" for label in image_labels]
    preferred = LabelForm(
        [str(image_labels[i]) for i in kept],
        [1 if is_channel[i] else int(image_shape[i]) for i in kept],
        kept,
    )
    without_channel = [i for i in kept if not is_channel[i]]
    earlier = LabelForm(
        [str(image_labels[i]) for i in without_channel],
        [int(image_shape[i]) for i in without_channel],
        without_channel,
    )
    # The forms differ in rank exactly when the image has a channel axis, which is
    # what lets `label_image_axes` tell them apart by rank.
    return [preferred] if earlier == preferred else [preferred, earlier]


def label_extent(
    image_labels: Sequence[str], image_shape: Sequence[int]
) -> Tuple[List[str], List[int]]:
    """The axes and lengths a new set of this image has (design, "Extent").

    The preferred :func:`label_forms` entry: what :func:`extent_mismatch`
    compares against first and what the upload fills in for a request that
    named no axes.
    """
    preferred = label_forms(image_labels, image_shape)[0]
    return preferred.labels, preferred.shape


def label_image_axes(
    label_labels: Sequence[str],
    image_labels: Sequence[str],
    image_shape: Sequence[int],
) -> Optional[List[int]]:
    """For each axis of a set, the wire index of the image axis it indexes.

    ``[0, 1, 2, 3, 4]`` for a ``T C Z Y X`` set of a ``T C Z Y X`` image, and
    ``[0, 2, 3, 4]`` for a set written under the earlier rule (no channel axis).
    The form is told apart by rank, which is why the server states the mapping
    rather than leave each client to guess it: reading the wrong one is frame 0 of
    a timelapse where frame 40 was asked for, and that is a picture rather than
    an error (biopb/biopb#1059).

    ``None`` when *label_labels* matches no form of *image_labels*, so there is
    no mapping to state. Callers that have already run :func:`extent_mismatch`
    never see it.
    """
    for form in label_forms(image_labels, image_shape):
        if len(form.labels) == len(label_labels):
            return list(form.image_axes)
    return None


def extent_mismatch(
    label_labels: Sequence[str],
    label_shape: Sequence[int],
    image_labels: Sequence[str],
    image_shape: Sequence[int],
    *,
    allow_earlier: bool = True,
) -> Optional[str]:
    """Why a set of *label_shape* over *label_labels* does not span the image,
    or None when it does.

    The rule is :func:`label_forms`; this is the comparison, against each form
    in turn, or against the preferred one alone when *allow_earlier* is false (a
    set being created). Axes are matched by canonical name, so ``"Z"`` and
    ``"depth"`` agree.
    """
    forms = label_forms(image_labels, image_shape)
    if not allow_earlier:
        forms = forms[:1]
    axes = [_axis_key(label) for label in label_labels]
    shape = [int(s) for s in label_shape]
    same_axes = [f for f in forms if axes == [_axis_key(label) for label in f.labels]]
    if not same_axes:
        return (
            f"axes {list(label_labels)} do not match the image's "
            f"{[_axis_key(label) for label in forms[0].labels]} (the channel "
            "axis a singleton, an RGB samples axis left out)"
        )
    if any(shape == f.shape for f in same_axes):
        return None
    return f"shape {shape} does not match the image's {same_axes[0].shape}"
