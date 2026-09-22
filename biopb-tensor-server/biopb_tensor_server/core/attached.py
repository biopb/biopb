"""The marked segment an attached tensor's id carries.

A tensor the *upload path* put on a source -- rather than one the source's own
format produced -- is addressed under a segment marked with :data:`MARKER`::

    <source_id>/@fields/<name>       a field uploaded onto a source
    <image array_id>/@labels/<name>  a label set of one of its tensors

**Every** uploaded tensor, whatever kind of source it was added to. The mark is
what keeps such an id off a native one: ``<source_id>/<field>`` is exactly the
shape of a native tensor id, so an attached field named ``0`` or ``scene1``
would shadow a scene of the user's own file -- silently, since the attached
tensors are listed after the format's own. ``<source_id>/@fields/0`` cannot, so
neither segment needs a namespace rule of its own, and a bare field always
means a tensor some format produced.

What a field *is* lives in :mod:`biopb_tensor_server.adapters.fields`; the label
half of the grammar is :mod:`biopb_tensor_server.core.labels`.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "FIELDS_SEGMENT",
    "MARKER",
    "attached_field",
    "is_published",
    "split_attached_field",
]

#: What marks a segment -- or a name -- as the server's rather than the user's.
#: The prefix ``@ome`` is chosen by, for the reason ``metadata_db`` gives it: a
#: user set called "ome" is plausible where "@ome" is not. An uploaded field may
#: not open with it (``adapters._writable.unsafe_field_name``): one prefix, one
#: meaning.
MARKER = "@"

#: The segment that marks a field uploaded onto a source the server discovered.
#: Its store is a member directory under ``<write_dir>/fields/<source_id>/``,
#: never inside the user's own file -- those bytes are theirs.
FIELDS_SEGMENT = f"{MARKER}fields"


def attached_field(name: str) -> str:
    """The within-source field of an uploaded field called *name*."""
    return f"{FIELDS_SEGMENT}/{name}"


def split_attached_field(field: Optional[str]) -> Optional[str]:
    """The name in *field*, or None if it does not name an uploaded field.

    ``"@fields/raw"`` -> ``"raw"``. The name is slash-free, so a deeper id --
    ``"@fields/raw/@labels/nuclei"``, a set of an uploaded field -- is *not* one
    of these: it names the set, and the label parse (right-to-left) is what
    takes it apart.
    """
    if not field:
        return None
    head, slash, name = field.partition("/")
    if head != FIELDS_SEGMENT or not slash or not name or "/" in name:
        return None
    return name


def is_published(adapter: object) -> bool:
    """Whether *adapter* may be listed: it is READY, or it tracks no upload.

    An attached tensor is routable from the moment ``add_tensor`` mints it --
    that is what carries its own writes and its status polls -- and listed only
    once it is readable.

    "No record" reads as published: the only tensor without one was adopted at
    boot, and only a published store is adopted.

    The upload is read by attribute, like ``_writable.upload_of`` and for the
    same reason; spelled out again because ``core`` does not import
    ``adapters``.
    """
    progress = getattr(adapter, "upload", None)
    return progress is None or progress.is_readable
