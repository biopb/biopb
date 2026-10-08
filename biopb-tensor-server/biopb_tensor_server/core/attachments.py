"""The tensors the upload path attached to one source.

A tensor the source's own format did not produce -- an uploaded field, a label
set the server minted -- is held here, by the registry, per source id, rather
than on the adapter: an adapter is rebuilt on a refresh and may be evicted,
while an upload in flight must keep routing and a finished one must stay listed.
Everything that reads them takes the source's adapter as an argument, because
what a label set may bind to is the adapter's tensors.

One index whatever the kind, keyed by within-source field, holding every such
tensor from ``add_tensor`` until the reclaim sweep detaches it. Whether one may
be *listed* is its own upload record's answer (:func:`is_published`), so an
upload in flight and the tombstone of one that was discarded stay routable: a
status poll and a straggler's write both have to find their adapter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

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
    split_label_field,
)

__all__ = ["Attachments"]

logger = logging.getLogger(__name__)


class Attachments:
    """One source's attached tensors, and the checked views over them."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        #: The routable index: published, still filling, or a tombstone.
        self.tensors: Dict[str, TensorAdapter] = {}
        # The published sets, for the parent they were built against; a rebuilt
        # parent is a different object and misses.
        self._view: Optional[Tuple[Any, Dict[str, TensorAdapter]]] = None
        # Why a listed set cannot be read, by field; rebuilt with the view.
        self._mismatch: Dict[str, str] = {}

    def attach(self, field: str, tensor: TensorAdapter) -> None:
        """Make *tensor* answer for *field*."""
        self.tensors[field] = tensor
        self.changed()

    def detach(self, field: str) -> Optional[TensorAdapter]:
        """Stop answering for *field*; returns what was attached, or None."""
        removed = self.tensors.pop(field, None)
        if removed is not None:
            self.changed()
        return removed

    def changed(self) -> None:
        """An attached tensor's state moved: rebuild the checked views.

        For the transition that changes what may be listed without touching the
        index too -- an upload reaching READY, or discarded into a tombstone.
        """
        self._view = None

    # -- the views ---------------------------------------------------------------

    def _labels(self, *, published: bool) -> Dict[str, TensorAdapter]:
        return {
            field: tensor
            for field, tensor in self.tensors.items()
            if split_label_field(field) is not None
            and is_published(tensor) is published
        }

    def label_uploads(self) -> Dict[str, TensorAdapter]:
        """Label sets the upload path is still filling, by field.

        Plus the tombstones of ones it gave up on: routable, so a status poll
        and a straggler's write both find their adapter, but never listed. A set
        reaches :meth:`label_sets` by becoming readable, not by being moved.
        """
        return self._labels(published=False)

    def attached_fields(self) -> Dict[str, TensorAdapter]:
        """The published fields uploaded onto this source, keyed by field.

        A tensor of this source whose bytes the upload path owns, under the
        marked segment that keeps its id off a native one
        (:mod:`~biopb_tensor_server.core.attached`). Listed after the format's
        own and unchecked, unlike a label set: a field binds to nothing, so
        there is nothing for it to fail to span.
        """
        return {
            field: tensor
            for field, tensor in self.tensors.items()
            if split_attached_field(field) is not None and is_published(tensor)
        }

    def label_sets(self, parent: SourceAdapter) -> Dict[str, TensorAdapter]:
        """Every label set of *parent*, keyed by within-source field.

        The published label sets attached here, each checked against the image
        it binds to: that image must be a tensor of the source, and the set must
        span it (:func:`~biopb_tensor_server.core.labels.extent_mismatch`). The
        sets the source's own file carries are its own tensors (``list_tensors``)
        and are not checked: the file is the user's. A set that
        fails stays listed, is logged as an error, and raises
        :class:`AttachedTensorMismatch` when read, rather than being served
        misaligned or vanishing. Empty until the source is resolved, since its
        tensors are unknown before that.

        This is the *published* view -- what the catalog lists and what a read
        resolves first. A set still being uploaded is in :meth:`label_uploads`
        and joins this one when its upload reaches READY.
        """
        if self._view is not None and self._view[0] is parent:
            return self._view[1]
        if not parent.is_resolved():
            return {}
        candidates = self._labels(published=True)
        images = self.normalized_tensors(parent) if candidates else {}
        view: Dict[str, TensorAdapter] = {}
        self._mismatch = {}
        for field, tensor in candidates.items():
            why = self.label_binding_error(
                parent, field, tensor.get_tensor_descriptor(), images=images
            )
            if why is not None:
                logger.error(f"labels: {self.source_id}/{field} cannot be read: {why}")
                self._mismatch[field] = why
            view[field] = tensor
        self._view = (parent, view)
        return view

    def normalized_tensors(self, parent: SourceAdapter) -> Dict[str, TensorEntry]:
        """*parent*'s tensors by ``array_id``, in canonical axis order.

        The images, not the sets the file carries. What a label set is checked
        against, and read once per check rather
        than per set -- ``list_tensors`` re-derives on an HCS plate.
        The uploaded fields are in it because a set may bind to one: a field is
        a tensor of this source like any other, and only its bytes live
        elsewhere.
        """
        entries = [
            e
            for e in parent.list_tensors()
            if split_label_field(strip_source_prefix(self.source_id, e.array_id))
            is None
        ]
        entries += [
            catalog_entry(t.get_tensor_descriptor())
            for t in self.attached_fields().values()
        ]
        return {e.array_id: e for e in entries}

    def label_binding_error(
        self,
        parent: SourceAdapter,
        field: str,
        desc: TensorEntry,
        images: Optional[Dict[str, TensorEntry]] = None,
    ) -> Optional[str]:
        """Why a set of *desc* cannot be served at label *field*, or None.

        One rule, checked at both ends: the upload kind calls it before it
        mints a sidecar, and :meth:`label_sets` calls it again for every origin
        when the sets are listed -- a native NGFF group and a sidecar from an
        earlier server life never passed through the upload. *desc* is in
        canonical order (both callers normalize first), and *images* is
        :meth:`normalized_tensors` when the caller already holds it.
        """
        if split_label_field(field) is None:
            return f"{field!r} does not name a label set"
        image = self.label_image_descriptor(parent, field, images=images)
        if image is None:
            return "binds to no tensor of the source"
        why = extent_mismatch(
            desc.dim_labels, desc.shape, image.dim_labels, image.shape
        )
        return f"does not span its image: {why}" if why is not None else None

    def label_image_descriptor(
        self,
        parent: SourceAdapter,
        field: str,
        images: Optional[Dict[str, TensorEntry]] = None,
    ) -> Optional[TensorEntry]:
        """The image a label *field* binds to, normalized, or None if it has none.

        What the extent is measured against, and what the upload kind reads to
        fill in the axes of a request that named none.
        """
        parsed = split_label_field(field)
        if parsed is None or parsed.level is not None:
            return None
        if images is None:
            images = self.normalized_tensors(parent)
        return images.get(join_fields(self.source_id, parsed.image_field))

    # -- the listing -------------------------------------------------------------

    def catalog_tensors(self, parent: SourceAdapter) -> List[TensorEntry]:
        """*parent*'s tensors as the catalog stores them.

        The one path into the DuckDB ``sources.tensors`` column. Entries are
        :class:`TensorEntry` records, which have no field for a serving fact
        (biopb/biopb#812); a bound tensor's descriptor is projected onto one.

        The attached tensors are listed **after** the format's own: a source's
        first tensor is the one every listing reads as its picture -- the
        browser groups on it, and SQL reaches for ``tensors[1]`` -- and neither
        a label set (biopb/biopb#1059) nor an uploaded field may ever be that.
        A scratch source has no tensors of its own, so its whole listing is
        fields, the same path a discovered source's uploaded fields take.
        """
        tensors = list(parent.list_tensors())
        for tensor in self.attached_fields().values():
            tensors.append(catalog_entry(tensor.get_tensor_descriptor()))
        for tensor in self.label_sets(parent).values():
            tensors.append(catalog_entry(tensor.get_tensor_descriptor()))
        return tensors

    # -- routing -----------------------------------------------------------------

    def for_field(
        self, parent: SourceAdapter, field: Optional[str]
    ) -> Optional[TensorAdapter]:
        """The attached tensor answering for within-source *field*, or None.

        Only a **marked** field reaches one (:mod:`~biopb_tensor_server.core.attached`);
        everything else is the format's own routing, which is the whole of the
        rule -- the upload path mints no bare field. The tensor is the one whose
        field is the longest prefix of *field*, and it resolves what remains
        itself, so a native level of a label set is routed like any other id.

        A set being uploaded is addressable from ``add_tensor`` onwards -- that
        is how its producer polls it to READY (biopb/biopb#1048) -- so the
        published sets are consulted first, then the whole index.
        """
        if not field or MARKER not in field:
            return None
        sets = self.label_sets(parent)
        key = owning_field(field, sets)
        if key is not None:
            why = self._mismatch.get(key)
            if why is not None:
                raise AttachedTensorMismatch(f"{self.source_id}/{key} {why}")
            tensor = sets[key]
        else:
            key = owning_field(field, self.tensors)
            if key is None:
                return None
            tensor = self.tensors[key]
        if field == key:
            return tensor
        return tensor.get_tensor_adapter(join_fields(self.source_id, field))

    def resolve_tensor(
        self, parent: SourceAdapter, tensor_id: Optional[str]
    ) -> TensorAdapter:
        """The adapter bound to *tensor_id*: an attached tensor of the source,
        else whatever ``parent.get_tensor_adapter`` answers.

        An id under a marked segment that names nothing attached here is handed
        to the format anyway rather than refused: a proxy's upstream may serve
        it, and a format that cannot raises its own ``TensorNotFound``.
        """
        field = strip_source_prefix(self.source_id, tensor_id)
        attached = self.for_field(parent, field)
        if attached is not None:
            return attached
        return parent.get_tensor_adapter(tensor_id)

    def capability_token(self, array_id: Optional[str]) -> Optional[str]:
        """The grant the tensor *array_id* carries, or None. The only reader of
        :attr:`TensorAdapter.capability_token`.

        Only an **attached** tensor can carry one: a format's own tensors are
        the source's, while what the upload path put here was produced by one
        caller and may be readable by that caller alone. Nothing consults a
        *source's* own token, so a source cannot gate what is attached to it --
        the tensors on one scratch source have different producers.

        Read off the routable index rather than through :meth:`resolve_tensor`,
        so the auth path asks no format to resolve anything and a tensor still
        uploading is gated exactly as a published one is. Checked on every read,
        so a source with nothing attached (the common case) returns before the
        field parse.
        """
        if not self.tensors:
            return None
        field = strip_source_prefix(self.source_id, array_id)
        key = owning_field(field, self.tensors) if field else None
        return self.tensors[key].capability_token if key is not None else None
