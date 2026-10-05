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
"""

from __future__ import annotations

from typing import NamedTuple, Optional

__all__ = [
    "LABELS_SEGMENT",
    "RESERVED_LABEL_PREFIX",
    "LabelAddress",
    "is_reserved_label_name",
    "split_label_array_id",
]

#: The segment that marks a label set under its image, in a wire id.
LABELS_SEGMENT = "@labels"

#: A name under this prefix is the server's own -- the set rasterized from a
#: file's masks, or a native NGFF set the source came with. Clients read one;
#: they never upload or delete one.
RESERVED_LABEL_PREFIX = "@"


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
