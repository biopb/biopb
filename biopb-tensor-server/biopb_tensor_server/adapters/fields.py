"""A tensor uploaded onto a source the server discovered (docs/upload-model.md).

A registered source is a container the upload path minted and may fill; a
*discovered* source is a file of the user's, and the upload path may not write
into it. So a field added to one keeps its bytes in a member directory of its
own, under ``<write_dir>/fields/<source_id>/<name>/``, and is bound as a tensor
of the source it was added to -- id ``<source_id>/@fields/<name>``.

**This is the label sidecar generalized, and the generalization is a
subtraction.** A label set carries three things a plain field does not: a
binding rule (it must name a tensor of the source and span it), a format's own
embedded sets, and the axis mapping the server publishes for it. A field binds
to nothing, is read from nothing, and maps to nothing -- so what is left is the
shared half, which was already generic and only *named* for labels:
``SourceAdapter.attach_tensor`` / ``detach_tensor``, the attached-tensor index,
``resolve_tensor``'s fall-through, ``catalog_tensors``' append-after-natives,
and attach-on-READY. This module adds the layout and the second
``on_register`` attacher, and nothing else.

**The marked segment is why no namespace rule is needed.**
``<source_id>/<field>`` is exactly the shape of a native tensor id, so a field
named ``0`` or ``scene1`` would shadow a scene of the user's own file --
silently, since the attached tensors are listed after the format's own.
``<source_id>/@fields/0`` cannot (``core.attached``).

**An orphaned field outlives its source.** A discovery root that goes away
leaves ``<write_dir>/fields/<source_id>/`` behind with nothing to attach it to.
A label sidecar has the same shape of problem and answers it by simply not
attaching; a field is different in kind -- it is the only copy of data a user
uploaded, where a set is usually derived -- so it is kept rather than swept, and
a source that returns re-attaches it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._writable import folded_match, unsafe_field_name
from biopb_tensor_server.adapters.members import member_marker
from biopb_tensor_server.adapters.registered import create_member_at, open_any_member
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, upload_state
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.attached import attached_field, split_attached_field

__all__ = [
    "create_field_upload",
    "fields_attacher",
    "fields_root",
    "scan_source_fields",
    "source_fields_dir",
    "upload_attacher",
]

logger = logging.getLogger(__name__)


def fields_root(write_dir: Path) -> Path:
    """Where every source's uploaded fields live: ``<write_dir>/fields/``.

    The one definition of the layout, as ``registered.sources_root`` and
    ``labels.labels_root`` are for the other two: the attacher reads it, the
    upload kind writes under it, and the boot sweep globs it.
    """
    return Path(write_dir) / "fields"


def source_fields_dir(fields_dir: Path, source_id: str) -> Path:
    """Where one source's uploaded fields live: ``<fields>/<source_id>/``."""
    return Path(fields_dir) / source_id


def create_field_upload(
    parent: Any,
    field: str,
    scheme: str,
    desc: TensorDescriptor,
    *,
    fields_dir: Path,
) -> TensorAdapter:
    """Mint the store for a new uploaded field on *parent* and track its upload.

    *field* is the id's within-source half, ``@fields/<name>``, already split by
    the boundary. The store is a member directory in either format
    (``registered.create_member_at``) under :func:`source_fields_dir`, rather
    than inside the parent's own store: the parent's bytes are the user's.

    Raises ``ValueError`` for a request the kind cannot serve -- a field that is
    not one of these, an unusable or marked name, a name already taken on this
    parent -- and nothing touches disk until every one of them has passed, so a
    refused request leaves no store behind.
    """
    array_id = f"{parent.source_id}/{field}"
    name = split_attached_field(field)
    if name is None:
        raise ValueError(
            f"{array_id!r} does not name an uploaded field: one is "
            f"'<source_id>/@fields/<name>' with a slash-free name."
        )
    # The store rules plus the marker: the name becomes a directory the server
    # creates and, on discard, removes whole, and a name under the marker would
    # claim a segment the server owns.
    why = unsafe_field_name(name)
    if why is not None:
        raise ValueError(f"{array_id!r}: the field's name {why}.")
    # Folded, because NTFS, APFS and HFS+ are case-insensitive and HFS+ stores
    # NFD. Only against what is attached here: the marked segment is what keeps
    # the name off the parent's own tensors, so those cannot collide.
    taken = folded_match(field, parent.attached_tensors)
    if taken is not None:
        raise ValueError(
            f"{array_id!r} already exists as {taken!r}. A field is taken for as "
            f"long as it is served, and two names differing only by case or "
            f"accent form are one directory on Windows and macOS; discard that "
            f"tensor, or add this one under another name."
        )
    store = source_fields_dir(fields_dir, parent.source_id) / name
    return create_member_at(store, parent.source_id, field, scheme, desc)


def scan_source_fields(source_id: str, fields_dir: Path) -> Dict[str, TensorAdapter]:
    """The finished uploaded fields of *source_id*, keyed by within-source field.

    Derived from the directory rather than from a dict something has to remember
    to fill, the shape ``registered.scan_members`` and ``labels.sidecar_label_sets``
    both use. A field still PENDING is one a crash left behind: the boot sweep
    removes it, and this pass skips whatever the sweep has not reached rather
    than adopting a half-written tensor. The format is read off the directory,
    which is what lets the scheme stay out of the stored ``array_id``.
    """
    root = source_fields_dir(fields_dir, source_id)
    if not root.is_dir():
        return {}
    fields: Dict[str, TensorAdapter] = {}
    for store in sorted(root.iterdir()):
        if not store.is_dir() or store.name.startswith("."):
            continue
        marker = member_marker(store)
        if upload_state(marker) == UPLOAD_PENDING:
            logger.info(f"field: {store} was left pending; not adopted")
            continue
        field = attached_field(store.name)
        adapter = open_any_member(
            store, source_id=source_id, field=field, marker=marker
        )
        if adapter is not None:
            fields[field] = adapter
    return fields


def fields_attacher(fields_dir: Path) -> Callable[[str, Any], None]:
    """The registry's ``on_register`` hook: attach a source's uploaded fields.

    Runs at the one registration chokepoint, because a field is keyed by
    ``source_id`` and no format knows about it -- the shape
    ``labels.sidecar_attacher`` already has. A field that will not open costs
    the field, never the source.
    """

    def attach(source_id: str, adapter: Any) -> None:
        for field, tensor in scan_source_fields(source_id, fields_dir).items():
            adapter.attach_tensor(field, tensor)

    return attach


def upload_attacher(
    *attachers: Callable[[str, Any], None],
) -> Callable[[str, Any], None]:
    """Several attachers as the one hook the registry takes.

    What the upload path leaves beside a source is more than one layout, and a
    failure in either is the registry's to log rather than the registration's to
    fail (``SourceRegistry.register``) -- but one that escaped here would cost
    the *other* attacher's tensors too, so each is caught on its own.
    """

    def attach(source_id: str, adapter: Any) -> None:
        for attacher in attachers:
            try:
                attacher(source_id, adapter)
            except Exception:
                logger.warning(
                    f"attaching uploaded tensors of {source_id} failed", exc_info=True
                )

    return attach
