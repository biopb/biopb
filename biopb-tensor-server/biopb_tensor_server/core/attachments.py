"""The tensors the upload path attached to one source.

A tensor the source's own format did not produce -- an uploaded field, a label
set the server minted -- is held here, by the registry, per source id, rather
than on the adapter: an adapter is rebuilt on a refresh and may be evicted,
while an upload in flight must keep routing and a finished one must stay listed.

One index whatever the kind, keyed by within-source field, holding every such
tensor from ``add_tensor`` until the reclaim sweep detaches it. Whether one may
be *listed* is its own upload record's answer (:func:`is_published`), so an
upload in flight and the tombstone of one that was discarded stay routable: a
status poll and a straggler's write both have to find their adapter.

**A label set is checked where things change, not where it is read.** It must
bind to an image of the source and span it
(:func:`~biopb_tensor_server.core.labels.extent_mismatch`). The images are the
parent's own, snapshotted when the registry registers an adapter
(:meth:`rebind` -- the single chokepoint, so a refresh and a rebuild after
eviction pass through it), and the source's published uploaded fields. The
verdict is recomputed whenever either moves (:meth:`rebind`, :meth:`attach`,
:meth:`detach`, :meth:`revalidate`) and kept, so a read is a lookup. A set that
fails stays listed, is logged, and raises :class:`AttachedTensorMismatch` when
read, rather than being served misaligned or vanishing. The sets a source's own
file carries are its own tensors and are not checked here: the file is the
user's.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

from biopb_tensor_server.core.adapter_base import (
    SourceAdapter,
    TensorAdapter,
    TensorEntry,
    catalog_entry,
    strip_source_prefix,
)
from biopb_tensor_server.core.attached import (
    MARKER,
    is_published,
    owning_field,
    split_attached_field,
)
from biopb_tensor_server.core.errors import AttachedTensorMismatch
from biopb_tensor_server.core.labels import (
    extent_mismatch,
    join_fields,
    label_extent,
    split_label_field,
)

__all__ = ["Attachments"]

logger = logging.getLogger(__name__)


class Attachments:
    """One source's attached tensors, and the verdict on each label set."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._lock = threading.RLock()
        # The routable index: published, still filling, or a tombstone.
        self._tensors: Dict[str, TensorAdapter] = {}
        # The parent's image tensors by array_id, as of its last registration;
        # None before the first, when there is nothing to judge a set against.
        self._native: Optional[Dict[str, TensorEntry]] = None
        # Why a label set cannot be read, by field.
        self._invalid: Dict[str, str] = {}

    # -- state ------------------------------------------------------------------

    def get(self, field: str) -> Optional[TensorAdapter]:
        """The tensor attached at exactly *field*, whatever its state."""
        return self._tensors.get(field)

    def items(self) -> Dict[str, TensorAdapter]:
        """Every attached tensor by field, whatever its state: a copy."""
        with self._lock:
            return dict(self._tensors)

    def error(self, field: str) -> Optional[str]:
        """Why the label set at *field* cannot be read, or None."""
        return self._invalid.get(field)

    # -- changes ----------------------------------------------------------------

    def attach(self, field: str, tensor: TensorAdapter) -> None:
        """Make *tensor* answer for *field*."""
        with self._lock:
            self._tensors[field] = tensor
            self._revalidate()

    def detach(self, field: str) -> Optional[TensorAdapter]:
        """Stop answering for *field*; returns what was attached, or None.

        A set bound to a detached uploaded field no longer has its image and is
        marked so; it is not deleted.
        """
        with self._lock:
            removed = self._tensors.pop(field, None)
            if removed is not None:
                self._revalidate()
            return removed

    def rebind(self, parent: SourceAdapter) -> None:
        """*parent* is the source's adapter now: judge every label set against it.

        Called by the registry on every registration of the source. Reads the
        parent's tensors once, here, and never again.
        """
        entries = [
            e
            for e in parent.list_tensors()
            if split_label_field(strip_source_prefix(self.source_id, e.array_id))
            is None
        ]
        with self._lock:
            self._native = {e.array_id: e for e in entries}
            self._revalidate()

    def revalidate(self) -> None:
        """An attached tensor's state moved -- an upload reached READY, or was
        discarded -- which changes which uploaded fields a set may bind to."""
        with self._lock:
            self._revalidate()

    # -- reads ------------------------------------------------------------------

    def listed(self) -> List[Tuple[str, TensorAdapter]]:
        """The published tensors, uploaded fields first and then label sets.

        A set that failed its check is in it: it is listed, and refuses to be
        read. One still uploading, or the tombstone of one that was discarded,
        is routable through :meth:`route` and joins this list when its upload
        reaches READY.
        """
        published = [(f, t) for f, t in self.items().items() if is_published(t)]
        return [(f, t) for f, t in published if split_attached_field(f) is not None] + [
            (f, t) for f, t in published if split_label_field(f) is not None
        ]

    def route(self, field: Optional[str]) -> Optional[TensorAdapter]:
        """The attached tensor answering for within-source *field*, or None.

        Only a **marked** field reaches one (:mod:`~biopb_tensor_server.core.attached`);
        everything else is the format's own routing, which is the whole of the
        rule -- the upload path mints no bare field. The tensor is the one whose
        field is the longest prefix of *field*, and it resolves what remains
        itself, so a native level of a label set is routed like any other id.

        Any state is routable: a set being uploaded is addressable from
        ``add_tensor`` onwards, which is how its producer polls it to READY
        (biopb/biopb#1048). Raises :class:`AttachedTensorMismatch` for a label
        set that failed its check.
        """
        if not field or MARKER not in field:
            return None
        with self._lock:
            key = owning_field(field, self._tensors)
            if key is None:
                return None
            why = self._invalid.get(key)
            tensor = self._tensors[key]
        if why is not None:
            raise AttachedTensorMismatch(f"{self.source_id}/{key} {why}")
        if field == key:
            return tensor
        return tensor.get_tensor_adapter(join_fields(self.source_id, field))

    def capability_token(self, array_id: Optional[str]) -> Optional[str]:
        """The grant the tensor *array_id* carries, or None. The only reader of
        :attr:`TensorAdapter.capability_token`.

        Only an **attached** tensor can carry one: a format's own tensors are
        the source's, while what the upload path put here was produced by one
        caller and may be readable by that caller alone. Nothing consults a
        *source's* own token, so a source cannot gate what is attached to it --
        the tensors on one scratch source have different producers.

        Read off the index rather than through :meth:`route`, so the auth path
        asks no format to resolve anything and a tensor still uploading is gated
        exactly as a published one is. Checked on every read, so a source with
        nothing attached (the common case) returns before the field parse.
        """
        if not self._tensors:
            return None
        field = strip_source_prefix(self.source_id, array_id)
        if not field or MARKER not in field:
            return None
        with self._lock:
            key = owning_field(field, self._tensors)
            return self._tensors[key].capability_token if key is not None else None

    # -- the upload kind's check ---------------------------------------------------

    def conform_label(self, field: str, desc: TensorEntry) -> Optional[str]:
        """Before a set is minted: why a set of *desc* cannot be served at
        *field*, or None.

        The extent rule leaves exactly one legal set of axes for the image, so
        a request that named none is filled in rather than refused -- in place,
        because *desc* is also what the client is answered with and what the
        sidecar's NGFF and chunk grid are built from. *desc* is in canonical
        order (the boundary refuses any other).
        """
        with self._lock:
            images = self._images()
        if not desc.dim_labels:
            image = self._image_of(field, images)
            if image is not None:
                desc.dim_labels.extend(label_extent(image.dim_labels, image.shape)[0])
        return self._binding_error(field, desc, images)

    # -- the check ---------------------------------------------------------------

    def _images(self) -> Dict[str, TensorEntry]:
        """What a set may bind to: the parent's images and the published
        uploaded fields, by array_id."""
        images = dict(self._native or {})
        for field, tensor in self._tensors.items():
            if split_attached_field(field) is not None and is_published(tensor):
                entry = catalog_entry(tensor.get_tensor_descriptor())
                images[entry.array_id] = entry
        return images

    def _image_of(
        self, field: str, images: Dict[str, TensorEntry]
    ) -> Optional[TensorEntry]:
        parsed = split_label_field(field)
        if parsed is None or parsed.level is not None:
            return None
        return images.get(join_fields(self.source_id, parsed.image_field))

    def _binding_error(
        self, field: str, desc: TensorEntry, images: Dict[str, TensorEntry]
    ) -> Optional[str]:
        if split_label_field(field) is None:
            return f"{field!r} does not name a label set"
        image = self._image_of(field, images)
        if image is None:
            return "binds to no tensor of the source"
        why = extent_mismatch(
            desc.dim_labels, desc.shape, image.dim_labels, image.shape
        )
        return f"does not span its image: {why}" if why is not None else None

    def _revalidate(self) -> None:
        """Recompute the verdict on every label set. Caller holds the lock."""
        sets = {f: t for f, t in self._tensors.items() if split_label_field(f)}
        if self._native is None or not sets:  # no parent yet, or nothing to judge
            self._invalid = {}
            return
        images = self._images()
        invalid: Dict[str, str] = {}
        for field, tensor in sets.items():
            why = self._binding_error(field, tensor.get_tensor_descriptor(), images)
            if why is None:
                continue
            invalid[field] = why
            if self._invalid.get(field) != why:
                logger.error(f"labels: {self.source_id}/{field} cannot be read: {why}")
        self._invalid = invalid
