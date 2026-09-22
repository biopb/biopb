"""Where a label set's ``array_id`` says it belongs (biopb/biopb#1059).

A label set is an ordinary tensor of its image, addressed under a marked
segment: ``<image array_id>/@labels/<name>[/<level>]``. Nothing else on the wire
marks one: the catalog has no ``role`` column, and a listing's per-tensor entry
carries no ``metadata_json`` to read the ``image-label`` block out of. The path
*is* the statement, so this module is the one place in the SDK that reads it.

Mirror of the server's ``core/labels.py`` and of the web SDK's ``label-id.ts``,
and deliberately the same rule rather than a looser one: the **last**
``@labels`` segment with a name after it is the one, whatever the image's own
field contains, and a name is slash-free, so anything past it is a native
pyramid level.

The marker is on the id, not on disk: an OME-Zarr store's own group stays
``labels/``. It is what stops an uploaded tensor's id colliding with a native
one, since a scene may plausibly be called ``labels`` and not ``@labels``.

Design: ``biopb-tensor-server/docs/label-tensors.md``.
"""

from __future__ import annotations

import json
from typing import Any, List, NamedTuple, Optional, Sequence

__all__ = [
    "LABELS_SEGMENT",
    "RESERVED_LABEL_PREFIX",
    "LabelAddress",
    "is_reserved_label_name",
    "label_image_axes",
    "split_label_array_id",
]

#: The segment that marks a label set under its image, in a wire id.
LABELS_SEGMENT = "@labels"

#: A name under this prefix is the server's own -- the set rasterized from a
#: file's masks, or a native NGFF set the source came with. Clients read one;
#: they never upload or delete one.
RESERVED_LABEL_PREFIX = "@"

# The channel-axis vocabulary, mirroring the server's ``core.axes.AXIS_C_LABELS``.
# Duplicated rather than imported: biopb-tensor-server is not a dependency of
# the SDK (nor installable from PyPI), and this is the one axis question the
# extent rule asks.
_CHANNEL_LABELS = frozenset({"c", "channel", "channels", "band", "bands"})


class LabelAddress(NamedTuple):
    """A label set's ``array_id``, taken apart."""

    image_array_id: str  #: the image this set annotates, as an ``array_id``
    name: str  #: the set's name, slash-free
    level: Optional[str]  #: a native pyramid level under the set, if named


def split_label_array_id(array_id: str) -> Optional[LabelAddress]:
    """Take a label set's ``array_id`` apart, or ``None`` if it names no set.

    ``"src0/@labels/nuclei"`` -> image ``"src0"``, name ``"nuclei"``. Pass a
    *stable* id: a content-pinned ``id@token`` is not one.
    """
    parts = array_id.split("/")
    # From 1: ``parts[0]`` is the source_id, which is slash-free by the identity
    # policy and so cannot be the ``labels`` segment of a field. A bare source
    # called "@labels" is a source.
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] != LABELS_SEGMENT or not parts[i + 1]:
            continue
        return LabelAddress(
            image_array_id="/".join(parts[:i]),
            name=parts[i + 1],
            level="/".join(parts[i + 2 :]) or None,
        )
    return None


def is_reserved_label_name(name: str) -> bool:
    """Whether *name* is a set the server owns -- read-only to every client."""
    return name.startswith(RESERVED_LABEL_PREFIX)


def label_image_axes(label_desc: Any, image_desc: Any) -> Optional[List[int]]:
    """For each axis of a set, the index of the image axis it indexes.

    ``[0, 2, 3, 4]`` for a ``T Z Y X`` set of a ``T C Z Y X`` image: a set spans
    the image's **non-channel** extent, so every axis after the image's ``c``
    sits one place to the left in the set. A client that instead matched axes by
    position reads frame 0 of a timelapse where frame 40 was asked for, which is
    a picture rather than an error.

    **Read, not derived, when the server says.** ``biopb.labels.image_axes`` in
    the set's ``metadata_json`` is the server's own statement of the mapping;
    a descriptor fetched without metadata, or from a server that predates the
    field, has none, and the extent rule is re-derived here instead -- the same
    answer, from the one place that still has to know the rule.

    ``None`` when the set does not span the image at all, which leaves the
    caller nothing to align and is the server's own answer in that case too.
    """
    stated = _stated_image_axes(label_desc)
    if stated is not None and len(stated) == len(label_desc.shape):
        return stated
    non_channel = [
        i
        for i, label in enumerate(image_desc.dim_labels)
        if str(label).lower() not in _CHANNEL_LABELS
    ]
    # Ranks are the check. An image whose labels went missing derives an empty
    # or over-long list, which cannot match a real set and so answers None --
    # the same "nothing to align" the server gives.
    return non_channel if len(non_channel) == len(label_desc.shape) else None


def _stated_image_axes(label_desc: Any) -> Optional[List[int]]:
    """``biopb.labels.image_axes`` off a descriptor's metadata, or ``None``.

    The server wraps per-tensor metadata as ``{"type", "dim_label", "metadata"}``
    (``serving/server.py``), so the block sits one level in. Any shape but a
    list of ints is treated as absent rather than raised on: this is an
    advisory field, and a caller that cannot read it still has the rule.
    """
    raw = getattr(label_desc, "metadata_json", "") or ""
    if not raw:
        return None
    try:
        wrapped = json.loads(raw)
        block = wrapped.get("metadata", wrapped)
        axes = block["biopb"]["labels"]["image_axes"]
    except (ValueError, AttributeError, KeyError, TypeError):
        return None
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        return None
    try:
        return [int(a) for a in axes]
    except (TypeError, ValueError):
        return None
