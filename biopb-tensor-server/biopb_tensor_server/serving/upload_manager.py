"""Writable-server upload path: the DoPut boundary.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). What is left
here is only what a boundary does:

- **Addressing** -- ``add_tensor`` takes ``<scheme>://<source_id>/<field>``
  for a member of a registered source, ``<scheme>://<source_id>/@fields/<name>``
  for a field on a discovered one, and ``zarr://<array_id>/@labels/<name>`` for
  a label set (biopb/biopb#1059). Nothing here creates a source: the parent must
  already be registered, and the new tensor is attached to it. The scheme names
  the store format and nothing else (``adapters.registered.STORE_FORMATS``), so
  the catalog row kept in step is always the parent's.
- **Error translation** -- adapters stay transport-agnostic and raise typed
  errors; this is where they become Flight errors.
- **Lookup** -- ``status`` / ``set_status`` find the adapter and hand over
  (``_locate``: the registry, then the parent's one attachment index).
  ``write_chunk`` is handed one: a DoPut ticket routes the way a DoGet ticket
  does.
- **Reclamation** -- ``reap`` sweeps every upload by its ``updated_at``, in
  the registry and in each source's attached tensors: one quiet past ``ttl``
  is discarded (a job that died) and a tombstone older than ``ttl`` is
  dropped, freeing the name. The sweep runs on a daemon thread started and
  stopped by the server that owns the manager, and
  ``discard_unfinished_stores`` removes at startup what a crashed server left
  behind. The catalog row of a ``durable`` kind is dropped here when its
  upload is discarded, by either route; the store itself is the adapter's own
  to release.

Progress, completion and disposal are the adapter's own
(:class:`~biopb_tensor_server.adapters._writable.WritableSource`), so there is no
second registry to keep in step with ``SourceRegistry``: an upload's state is
created with its adapter, lives as long as that adapter is reachable -- from
the registry, or from its parent source for a label set -- and a discarded one
stays reachable as a tombstone until reclaimed.

The manager registers created sources through the shared ``SourceRegistry`` and
never holds a back-reference to the server, so the collaborators stay acyclic.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._writable import (
    UploadStatus,
    unknown_upload_status,
    upload_of,
)
from biopb_tensor_server.adapters.fields import create_field_upload, fields_root
from biopb_tensor_server.adapters.labels import create_label_upload, labels_root
from biopb_tensor_server.adapters.members import member_marker
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.registered import (
    RegisterAdapter,
    create_member,
    create_registered_source,
    scan_registered_sources,
    sources_root,
)
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, upload_state
from biopb_tensor_server.core.attached import FIELDS_SEGMENT
from biopb_tensor_server.core.axes import noncanonical_order
from biopb_tensor_server.core.chunk import get_bounds_from_chunk_id
from biopb_tensor_server.core.errors import (
    UploadClosedError,
    UploadDiscardedError,
    UploadTransitionError,
    WriteNotSupportedError,
)
from biopb_tensor_server.core.labels import split_label_field
from biopb_tensor_server.core.source_registry import SourceRegistry, close_adapter
from biopb_tensor_server.serving.metadata_db import MetadataDatabase

__all__ = [
    "DEFAULT_UPLOAD_TTL",
    "UploadManager",
    "UploadStatus",
    "write_dir_under_root",
]

logger = logging.getLogger(__name__)

#: Seconds a PENDING upload may go without a write, and a tombstone may stand,
#: before ``reap`` acts. Generous: what it bounds is only how long a dead job's
#: name stays taken and how long a straggler can still learn why its writes
#: fail. A live upload touches on every chunk, so the gap it must exceed is one
#: chunk's compute, not the whole job's. Overridden by ``ServerConfig.upload_ttl``.
DEFAULT_UPLOAD_TTL = 3600.0

#: How ``add_tensor`` spells a tensor it is asked to create. The scheme names
#: the store format and nothing else; everything after it is the tensor's
#: ``array_id``, which is also the id it keeps and is answered with.
_ID_GRAMMAR = (
    "'<scheme>://<source_id>/<field>' to add a tensor to a registered source, "
    "'<scheme>://<source_id>/@fields/<name>' to add one to a source the server "
    "discovered, or 'zarr://<array_id>/@labels/<name>' to add a label set to "
    "one of its tensors"
)


def write_dir_under_root(
    write_dir: Optional[Path], roots: Iterable[Path]
) -> Optional[Path]:
    """The discovery root that contains *write_dir*, if any.

    A store minted under ``write_dir`` is registered by the upload path under
    its own id. If discovery also walks that directory, a finished store is
    claimed a second time under discovery's id -- listed twice, served twice --
    and an unfinished one is only kept out by the claims checking the upload
    marker. So ``write_dir`` belongs outside every discovered directory; the
    launcher warns when it is not (biopb/biopb#1059). Compared resolved, so a
    symlinked root still matches.
    """
    if write_dir is None:
        return None
    target = write_dir.resolve()
    for root in roots:
        resolved = root.resolve()
        if target == resolved or resolved in target.parents:
            return root
    return None


def _refused(exc: UploadClosedError) -> flight.FlightCancelledError:
    """An upload-closed error as the wire sees it.

    ``FlightCancelledError`` for both kinds -- the one thing a writer must act
    on is *this upload is over* -- with the terminal state in ``extra_info``,
    so the client switches on a field, not a substring. No ``code``: both kinds
    map to the same exception class, so the gRPC code is implied by the class,
    not data. ``detail`` is the discard's reason.
    """
    payload = {
        "reason": exc.wire_reason,
        "state": exc.state,
        "source_id": exc.source_id,
        "detail": getattr(exc, "reason", ""),
    }
    return flight.FlightCancelledError(str(exc), json.dumps(payload).encode())


def _attached(adapter: Any) -> Dict[str, Any]:
    """The tensors the upload path put on *adapter*, or none.

    By attribute, like :func:`upload_of`: the registry also holds adapters from
    outside this package, and one that knows nothing about the upload path has
    none of them.

    One index for every kind -- a member, a label set, a field on a discovered
    source (``SourceAdapter.attached_tensors``) -- which is what lets this
    boundary locate, publish, unlist and reap them with no per-kind branch.
    """
    return getattr(adapter, "attached_tensors", None) or {}


def _reap_step(adapter: Any, now: float, ttl: float) -> Tuple[bool, bool]:
    """One upload's turn in the sweep: ``(just expired, tombstone is stale)``.

    Both halves come from ``WritableSource.reap_step``; this only turns the
    tombstone's age into the sweep's yes/no, so the registered kinds and the
    label sets attached to a source take the identical step.
    """
    expired, age = adapter.reap_step(now, ttl)
    return expired, age is not None and age > ttl


def _reap_outcome(expired_now: bool, stale: bool) -> Optional[str]:
    """``"expired"``, ``"reclaimed"``, or None, for one entity's sweep step.

    The registered kinds and a source's label uploads take the identical
    decision in :meth:`reap`; only what each does about it -- a different
    namespace, registry vs. a source's own dict -- differs per loop.
    """
    if expired_now:
        return "expired"
    if stale:
        return "reclaimed"
    return None


class UploadManager:
    """The DoPut boundary: picks the kind, registers, translates errors."""

    def __init__(
        self,
        registry: SourceRegistry,
        write_dir: Optional[Path],
        metadata_db: Optional[MetadataDatabase],
        ttl: float = DEFAULT_UPLOAD_TTL,
    ) -> None:
        self._registry = registry
        self._write_dir = write_dir
        self._metadata_db = metadata_db
        #: Read live by every sweep, so a retune takes effect without a restart.
        #: 0 disables reclamation.
        self.ttl = float(ttl)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lookup ----------------------------------------------------------------

    def _locate(self, upload_id: str) -> Tuple[Any, Any, Optional[str]]:
        """The adapter tracking upload *upload_id*, its source, and its field.

        Every upload is a tensor of a source that already exists, so this is
        one lookup in two halves: the source from the registry, then the field
        from that source's attachment index -- whatever kind of tensor it is.
        Returns ``(adapter, parent, field)``, with *adapter* None when nothing
        holds the id, which every caller already has to handle.

        A bare ``source_id`` answers ``(the source, None, None)``: it is a real
        adapter and tracks no upload, which is what makes ``status`` report
        UNKNOWN on a registered source rather than raising.
        """
        source_id, _, field = upload_id.partition("/")
        if not field:
            return self._registry.get(upload_id), None, None
        parent = self._registry.get(source_id)
        if parent is None:
            return None, None, None
        return _attached(parent).get(field), parent, field

    def status(self, source_id: str) -> Dict[str, Any]:
        """The ``upload_status`` answer: UNKNOWN for anything not tracking an upload."""
        adapter, _, _ = self._locate(source_id)
        upload = upload_of(adapter)
        if upload is None:
            return unknown_upload_status(source_id)
        return upload.as_status_dict(source_id)

    def discard(self, array_id: str, reason: str = "") -> Dict[str, Any]:
        """Give up on an upload; see ``WritableSource.discard``.

        Reachable from every state, READY included -- which is what makes it
        the one way a published upload is ever removed, and why there is no
        second verb for deleting one. A durable kind's store goes with the
        discard (the adapter's own), and its listing here: the catalog row for
        the registered kinds, the parent's attachment for a label set.

        Total, and a statement about the end state rather than a receipt: an
        id that is not tracking an upload -- never was, has since been
        reclaimed, or names a set an earlier life of this server uploaded --
        reads UNKNOWN rather than raising, so a retry after the tombstone is
        gone is not an error.
        """
        adapter, parent, field = self._locate(array_id)
        if upload_of(adapter) is None:
            return self._delete_adopted_tensor(array_id)
        status = adapter.discard(reason)
        if parent is None:
            self._drop_catalog_row(adapter, array_id)
        else:
            self._unlist(parent, field)
        return status

    def set_status(
        self, array_id: str, state: UploadStatus, reason: str = ""
    ) -> Dict[str, Any]:
        """The ``set_upload_status`` action: move one upload along its ladder.

        The adapter owns the transition (``WritableSource.set_status``); what
        is here is the half only the boundary can do -- turning the adapter's
        typed errors into Flight ones, and publishing a label set into its
        parent's listing the first time it becomes readable.

        The listing follows READY because READY is what makes the set readable
        at all, and a set nobody can reach is not one to
        advertise. It happens after the store is published on disk, never
        before: the catalog must not name a set a restart would sweep away.

        ``DISCARDED`` routes to :meth:`discard`, which is total; every other
        target requires an upload, because a caller moving one believes it has
        been writing somewhere it may not have been.
        """
        if state is UploadStatus.DISCARDED:
            return self.discard(array_id, reason)
        adapter, parent, field = self._locate(array_id)
        progress = upload_of(adapter)
        if progress is None:
            raise flight.FlightServerError(
                f"set_upload_status: {array_id} is not an upload in progress"
            )
        was_readable = progress.is_readable
        try:
            status = adapter.set_status(state, reason)
        except UploadDiscardedError as e:
            raise _refused(e) from e
        except UploadTransitionError as e:
            raise flight.FlightServerError(str(e)) from e
        except OSError as e:
            # The store could not be published; the upload stays PENDING and
            # the caller retries, so this is a server error, not a refusal.
            raise flight.FlightServerError(
                f"set_upload_status: could not publish {array_id} on disk: {e}"
            ) from e
        if parent is not None and not was_readable and progress.is_readable:
            self._publish(parent, field, adapter)
        return status

    def _delete_adopted_tensor(self, array_id: str) -> Dict[str, Any]:
        """Remove a listed tensor that this life of the server did not upload.

        A member or sidecar re-attached at registration carries no upload
        record, so :meth:`_locate` finds nothing to discard -- only a store,
        still the server's own (it was minted under ``write_dir``). Unlisted
        first, so no read can be routed to a store that is about to go.

        Reaches exactly what the detach verbs reach, which is what the upload
        path published. A set the *format* carries (an NGFF ``labels/`` group;
        a rasterized ``@ome``) is embedded, not attached, so it stays; anything
        else is a no-op, and answers UNKNOWN like the rest of :meth:`discard`.
        """
        source_id, _, field = array_id.partition("/")
        parent = self._registry.get(source_id) if field else None
        if parent is None:
            return unknown_upload_status(array_id)
        adapter = parent.detach_tensor(field)
        if adapter is None:
            return unknown_upload_status(array_id)
        adapter.delete_store()
        self._sync_parent_row(parent)
        logger.info(f"Deleted tensor {array_id}")
        return unknown_upload_status(array_id)

    # -- listing ---------------------------------------------------------------

    def _sync_parent_row(self, parent: Any) -> None:
        """Re-publish a parent's catalog row after its sets changed.

        ``sync_source_added`` is an upsert and ``catalog_tensors`` reads the
        sets off the adapter, so re-registering the parent is the whole of it
        (the ROI re-import it triggers is idempotent).
        """
        self._sync_row(parent.source_id, parent)

    def _publish(self, parent: Any, field: Optional[str], adapter: Any) -> None:
        """List a tensor that has just become readable, under its source.

        Every kind was attached at ``add_tensor`` -- that is what routes its own
        writes -- and becomes *listed* by becoming readable, which its own upload
        record answers. So all that is owed here is telling the source its views
        are stale, and the row.
        """
        parent.attachment_changed()
        self._sync_parent_row(parent)

    def _unlist(self, parent: Any, field: Optional[str]) -> None:
        """Take a tensor out of its source's listing, if it was in it.

        Nothing is detached: a tensor leaves the listing by ceasing to be
        readable, and stays reachable as the tombstone a straggler polls until
        the reclaim sweep drops it. A tensor that was never published was never
        listed, so this owes a catalog write only where one is owed -- the
        ordinary discard of an upload in flight is a no-op.
        """
        if parent is None or field is None:
            return
        adapter = _attached(parent).get(field)
        if adapter is None:
            return
        parent.attachment_changed()
        self._sync_parent_row(parent)

    def _add_label_set(
        self, parent: Any, field: str, req_desc: TensorDescriptor
    ) -> TensorDescriptor:
        """A new label set on a tensor that already exists (biopb/biopb#1059).

        The set is attached to its source, which is what routes its own writes
        and its status polls; it is not *listed* until it reaches READY
        (``SourceAdapter.label_sets``). No catalog write happens at create: the
        row the source already has still describes what a reader may see.
        """
        metadata = (
            self._parse_metadata_json(req_desc.metadata_json)
            if req_desc.metadata_json
            else None
        )
        try:
            adapter = create_label_upload(
                parent,
                field,
                req_desc,
                labels_dir=labels_root(self._write_dir),
                metadata=metadata,
            )
        except ValueError as e:
            raise flight.FlightServerError(f"add_tensor: {e}") from e
        parent.attach_tensor(field, adapter)
        logger.info(f"Added label set {adapter.array_id}")
        return adapter.upload_response(req_desc)

    def _drop_catalog_row(self, adapter: Any, source_id: str) -> None:
        """Take a discarded durable upload out of the catalog.

        The row was written at create (a durable upload is listed while it
        fills), so a tombstone would otherwise stay browsable with nothing
        behind it. Idempotent, like the discard it follows.
        """
        if getattr(adapter, "durable", False):
            self._drop_row(source_id)

    def _drop_row(self, source_id: str) -> None:
        """Take *source_id* out of the catalog; best-effort.

        Best-effort for the same reason the write was: the catalog must not
        fail the upload, and a leaked row is the worst case -- the same policy
        the reconciler's own teardown keeps.
        """
        if self._metadata_db is None:
            return
        try:
            self._metadata_db.sync_source_removed(source_id)
        except Exception as e:
            logger.warning(f"Failed to drop upload {source_id} from the catalog: {e}")

    def discard_unfinished_stores(self) -> int:
        """Delete the stores of uploads a previous server never finished.

        A durable store is born ``pending`` and marked ``ready`` when its
        upload reaches READY (``adapters.zarr`` upload marker). One still
        pending when this server starts belonged to an upload whose progress
        died with the process before anyone published it; nothing can publish
        it now, so it is removed rather than left for discovery to serve as a
        partial source. Runs from the server's
        constructor, before any source is registered. Returns the count.

        The layouts the server mints: the members of a registered source
        (``sources/<name>.zarr/<field>``, in either store format), the fields
        uploaded onto a discovered one (``fields/<source_id>/<name>``, likewise),
        the label sidecars
        (``labels/<source_id>/*.zarr``), and -- for one more release -- the
        single-array stores the removed ``ome_zarr:`` kind left directly under
        ``write_dir``. They differ in what the catalog owes them, which is why
        they are swept separately.
        """
        write_dir = self._write_dir
        if write_dir is None or not write_dir.is_dir():
            return 0
        removed = 0
        for store in sorted(sources_root(write_dir).glob("*/*")):
            # A member has no row of its own: it is a tensor of its source,
            # whose row is rebuilt when that source is adopted -- which is
            # after this sweep. The collection itself is never pending, so a
            # ``.zattrs`` or ``.zgroup`` file here is simply not a directory.
            if store.is_dir():
                removed += self._remove_unfinished(store)
        for store in sorted(fields_root(write_dir).glob("*/*")):
            # A field has no row of its own either: its source is the user's
            # own file, and the reconciler owns that row. Only the *pending*
            # ones go -- a published field is the only copy of what someone
            # uploaded, and is kept even when its source has gone away.
            if store.is_dir():
                removed += self._remove_unfinished(store)
        for store in sorted(labels_root(write_dir).glob("*/*.zarr")):
            # A sidecar has no row of its own either, for the same reason.
            removed += self._remove_unfinished(store)
        removed += self._drop_legacy_ome_zarr_stores(write_dir)
        return removed

    def _drop_legacy_ome_zarr_stores(self, write_dir: Path) -> int:
        """Retire what the removed ``ome_zarr:`` kind left under ``write_dir``.

        Such a store is a single array, not a collection, so there is nothing
        here that could adopt it and its catalog row has had no adapter behind
        it since the kind went. The row goes either way; the **bytes** go only
        if the upload never finished, because a finished store is the user's
        data and this server no longer has a claim to it -- it is left where it
        is, for them to point a discovery root at (``docs/upload-model.md``,
        Migration). Returns how many stores were removed.
        """
        removed = 0
        for store in sorted(write_dir.glob("*.zarr")):
            self._drop_row(OmeZarrAdapter.upload_source_id(store))
            removed += self._remove_unfinished(store)
        return removed

    @staticmethod
    def _remove_unfinished(store: Path) -> bool:
        """Delete *store* if it is still pending; whether it was.

        Through ``member_marker``, so the marker is read off whichever file
        this store's format keeps it in -- the sweep decides before any adapter
        is asked to open the directory, and neither format is privileged.
        """
        if upload_state(member_marker(store)) != UPLOAD_PENDING:
            return False
        shutil.rmtree(store, ignore_errors=True)
        logger.info(f"Removed unfinished upload store {store}")
        return True

    # -- write path ------------------------------------------------------------

    @staticmethod
    def _require_canonical_axes(req_desc: TensorDescriptor) -> None:
        """Reject an upload whose declared axis order is not canonical (#596).

        A writable source is the one place the axis order is *declared* rather
        than read out of a file, and both ends of it belong to the same client:
        ``physical_scale`` and ``chunk_shape`` come in aligned to these labels,
        and ``put_chunk`` writes in this order. So the server refuses the order
        up front instead of permuting reads behind the uploader's back, which
        would desynchronize what it reads back from what it wrote. Transposing
        before upload is the client-side fix, and it is a cheap one.

        This keeps the canonical-order guarantee unconditional -- it holds for
        uploaded sources too -- at zero cost on the write data path. The remote
        proxy refuses for the same reason, in the same words
        (``core.axes.noncanonical_order``): there too the order belongs to
        someone who has aligned the rest of their state to it.
        """
        why = noncanonical_order(req_desc.dim_labels, req_desc.shape)
        if why is None:
            return
        raise flight.FlightServerError(
            f"add_tensor: {why}. The data plane advertises canonical order on "
            f"every source (biopb/biopb#596); transpose the array before "
            f"uploading."
        )

    def add_tensor(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """Add a tensor to a source that already exists; answer its descriptor.

        **Nothing here creates a source.** ``register_source`` does that, and
        an upload names what it is adding to: ``<scheme>://<source_id>/<field>``
        for a tensor of a registered source,
        ``<scheme>://<source_id>/@fields/<name>`` for one on a source the server
        discovered, and ``zarr://<array_id>/@labels/<name>`` for a label set of
        any tensor of either. The scheme names the store
        format (``registered.STORE_FORMATS``) and nothing else -- the answered
        ``array_id`` carries none, because the format is a property of the
        stored tensor, read off its directory at the next registration.

        Which of the three it is comes off the **field**, not the parent's type:
        a marked segment says the upload path owns the tensor's bytes and where
        they go, and a bare field is the parent's own member.

        A field is taken for as long as its tensor is served. Replacing one is
        a discard first, which is what makes an ``array_id`` name a single
        attempt: a second add under a live name would otherwise land on the
        first's adapter, leaving the first writer's chunks and transitions in a
        tensor that is no longer its own with nothing to tell it so.
        """
        self._require_canonical_axes(req_desc)
        if self._write_dir is None:
            raise flight.FlightServerError(
                "add_tensor: write_dir is not configured, so there is nowhere "
                "to put a tensor"
            )

        scheme, sep, rest = req_desc.array_id.partition("://")
        source_id, _, field = rest.partition("/")
        if not sep or not source_id or not field:
            raise flight.FlightServerError(
                f"add_tensor: {req_desc.array_id!r} names no tensor. Use {_ID_GRAMMAR}."
            )
        parent = self._registry.get(source_id)
        if parent is None:
            raise flight.FlightServerError(
                f"add_tensor: {req_desc.array_id!r} names no registered source "
                f"{source_id!r}. A tensor is added to a source the server "
                f"already serves; it does not create one."
            )

        if split_label_field(field) is not None:
            if scheme != "zarr":
                raise flight.FlightServerError(
                    f"add_tensor: a label set is NGFF, so it takes 'zarr://', "
                    f"not {scheme + '://'!r}."
                )
            return self._add_label_set(parent, field, req_desc)

        if req_desc.metadata_json:
            raise flight.FlightServerError(
                "add_tensor: metadata is the source's, not a tensor's -- pass it "
                "to register_source. A tensor declares shape, dtype, chunk grid "
                "and axes; the one exception is a label set's 'image-label' "
                "block, which rides on its own add_tensor."
            )
        if field == FIELDS_SEGMENT or field.startswith(f"{FIELDS_SEGMENT}/"):
            # The *segment* dispatches, not the whole parse: a malformed name
            # under it is the field kind's to refuse, and must not fall through
            # to the member path and be answered about members.
            adapter = self._create_field(parent, field, scheme, req_desc)
        else:
            adapter = self._create_member(parent, source_id, field, scheme, req_desc)
        # Attached, not listed: this is what routes the tensor's own writes, and
        # the published gate keeps it out of the source's tensors until it
        # reaches READY. So no catalog write is owed here -- the row the source
        # already has still describes what a reader may see.
        parent.attach_tensor(field, adapter)
        logger.info(f"Added {scheme} tensor: {adapter.array_id}")
        return adapter.upload_response(req_desc)

    def _create_member(
        self,
        parent: Any,
        source_id: str,
        field: str,
        scheme: str,
        desc: TensorDescriptor,
    ) -> Any:
        """Mint a tensor of a source the server registered, inside its store."""
        if not isinstance(parent, RegisterAdapter):
            raise flight.FlightServerError(
                f"add_tensor: {source_id!r} is not a registered source, so it "
                f"has no store to put a tensor of its own in. Add it beside the "
                f"source instead: "
                f"'{scheme}://{source_id}/{FIELDS_SEGMENT}/<name>'."
            )
        try:
            return create_member(parent, field, scheme, desc)
        except ValueError as e:
            raise flight.FlightServerError(f"add_tensor: {e}") from e

    def _create_field(
        self, parent: Any, field: str, scheme: str, desc: TensorDescriptor
    ) -> Any:
        """Mint a tensor beside a source the server discovered.

        Refused on a registered source: that one has a store of its own, and a
        second place for its tensors to live would be two layouts to adopt, to
        sweep and to reclaim for one kind of source.
        """
        if isinstance(parent, RegisterAdapter):
            raise flight.FlightServerError(
                f"add_tensor: {parent.source_id!r} is a registered source, so "
                f"its tensors are members of its own store: drop the "
                f"{FIELDS_SEGMENT!r} segment."
            )
        try:
            return create_field_upload(
                parent, field, scheme, desc, fields_dir=fields_root(self._write_dir)
            )
        except ValueError as e:
            raise flight.FlightServerError(f"add_tensor: {e}") from e

    def register_source(self, name: str = "", metadata_json: str = "") -> str:
        """Mint a source a client can add tensors to; answers its ``source_id``.

        Registration, not upload: the collection holds no bytes and has no
        upload state, so it is readable (and empty) the moment this returns.
        What makes it outlive the process is that :meth:`adopt_registered_sources`
        finds it again, not that anything discovers it -- ``write_dir`` stays
        outside every discovery root.

        *metadata_json* is the source's OME block as the request carried it,
        parsed here rather than by the caller -- the same boundary, and the
        same refusal, ``add_tensor`` gives a malformed one.
        """
        if self._write_dir is None:
            raise flight.FlightServerError(
                "register_source: write_dir is not configured, so there is "
                "nowhere to put a source"
            )
        metadata = self._parse_metadata_json(metadata_json) if metadata_json else None
        try:
            adapter = create_registered_source(
                name, metadata, sources_root(self._write_dir)
            )
        except (ValueError, OSError) as e:
            raise flight.FlightServerError(str(e)) from e

        registered = self._registry.register_new(adapter.source_id, adapter)
        if registered is None:
            # A minted id that is already registered means the mint collided,
            # which 48 random bits make a non-event -- but the store is already
            # on disk, so release it rather than leave the server's own bytes
            # for the next boot to adopt under an id it cannot serve.
            adapter.dispose_store()
            close_adapter(adapter)
            raise flight.FlightServerError(
                f"register_source: {adapter.source_id} is already registered; retry."
            )
        self._sync_row(adapter.source_id, registered)
        return adapter.source_id

    def adopt_registered_sources(self) -> int:
        """Re-register what the last life left under ``<write_dir>/sources``.

        The half of registration that was missing (biopb/biopb#1048): the
        upload path registered a minted store in the life that created it, and
        nothing brought it back, so its catalog row outlived the adapter behind
        it. Runs at startup after the pending sweep and before the caller's
        discovery scan, so an adopted id is already registered by the time
        anything can ask for it. Returns the count.
        """
        if self._write_dir is None:
            return 0
        count = 0
        for source_id, adapter in scan_registered_sources(
            sources_root(self._write_dir)
        ).items():
            if self._registry.register_new(source_id, adapter) is None:
                # Something else already holds the id. Leave its store alone --
                # this pass adopts, it does not arbitrate.
                logger.warning(f"Registered source {source_id} is already served")
                close_adapter(adapter)
                continue
            self._sync_row(source_id, adapter)
            count += 1
        if count:
            logger.info(f"Adopted {count} registered source(s)")
        return count

    def _sync_row(self, source_id: str, adapter: Any) -> None:
        """Put *source_id* in the catalog; the one best-effort catalog write.

        Best-effort for the reason every catalog write on this path is: the
        row is how a source is *browsable*, and it must not be able to fail the
        registration that made it *readable*. A leaked row is the worst case,
        and the boot sweep is what collects those.
        """
        if self._metadata_db is None:
            return
        try:
            self._metadata_db.sync_source_added(source_id, adapter)
        except Exception as e:
            logger.warning(
                f"Failed to sync {source_id} to the catalog "
                f"(readable by id, not listed): {e}"
            )

    def write_chunk(
        self,
        adapter: Any,
        chunk_id: bytes,
        reader: flight.MetadataRecordBatchReader,
    ) -> None:
        """Hand one uploaded chunk to its tensor's ``put_chunk``.

        *adapter* and *chunk_id* come from the DoPut ticket, routed and
        version-checked at the boundary exactly as a DoGet ticket is
        (``server.do_put``): one planner minted it, so the bounds are the
        plan's own and no alignment grammar is left for the write path to
        enforce.

        Each source format still owns what it does with the chunk, and an
        upload that is discarded or sealed refuses. Adapters stay
        transport-agnostic, so their errors become Flight errors here.

        Both refusals map to ``FlightCancelledError`` (:func:`_refused`): they
        share the one thing a writer must act on -- this upload is over, stop
        sending -- and a client discriminates on the type rather than the
        message (biopb/biopb#1). Which of the two it was rides in
        ``extra_info``.
        """
        table = reader.read_all()
        data_column = table.column(0)

        bounds = get_bounds_from_chunk_id(chunk_id)
        expected_shape = tuple(
            stop - start for start, stop in zip(bounds.start, bounds.stop, strict=True)
        )
        dtype = table.schema.field(0).type.to_pandas_dtype()
        try:
            adapter.put_chunk(bounds, data_column, expected_shape, dtype)
        except UploadClosedError as e:
            raise _refused(e) from e
        except (ValueError, WriteNotSupportedError) as e:
            raise flight.FlightServerError(str(e)) from e

        logger.debug(
            f"Uploaded chunk to {adapter.array_id}: "
            f"bounds={list(bounds.start)}-{list(bounds.stop)}"
        )

    # -- reclamation -----------------------------------------------------------

    def reap(self, now: Optional[float] = None) -> Tuple[int, int]:
        """One sweep over the registry by ``updated_at``; the unit the thread runs.

        Two halves, one clock. A PENDING upload with no write for ``ttl``
        seconds is discarded with a reason -- the job that owned it died, and
        a discard is the one terminal transition there is, so a straggler
        that writes later is refused the same way it would be after an
        explicit discard. A tombstone that has stood for ``ttl`` seconds is
        unregistered: its name is free again and its status reads UNKNOWN,
        which a poller already treats as fail-fast (biopb/biopb#109). An
        expired upload therefore takes two sweeps to vanish, and is a
        tombstone in between.

        Only PENDING is swept. Past it the upload has been published, and a
        published result's lifetime is its reader's, not this sweep's -- so an
        upload left READY without ever being finished keeps its name and stays
        writable, which is the cost of letting a producer publish early. A durable
        kind is swept like any other: its store goes with the discard and its
        catalog row is dropped here, since both were the server's own
        (biopb/biopb#1059).

        Every source's attached tensors take the same step, because a tensor is
        attached to its source rather than registered and would otherwise have
        no sweep at all: a quiet pending member, field or label set is discarded
        (its store with it) and unlisted, and its tombstone is later detached,
        which is what frees the field.

        A registered source left with no live tensor goes with them, once it
        too has been quiet for ``ttl``. That covers both ends of the same
        mistake: a create nobody ever added to, and a source whose last tensor
        was discarded. Pending counts as live, so an upload in flight never
        has the source pulled out from under it.

        The chunks a reclaimed tombstone wrote stay in the cache until the
        LRU evicts them. They are unreachable: a re-created name gets a fresh
        ``content_version`` namespace, so its chunk ids never collide with
        the old ones (``CachedSourceAdapter.next_content_version``).

        Returns ``(expired, reclaimed)`` counts. *now* is injectable for tests;
        it is on the monotonic clock, like ``updated_at``.
        """
        ttl = self.ttl
        if ttl <= 0:
            return 0, 0
        if now is None:
            now = time.monotonic()
        expired = reclaimed = 0
        for source_id, adapter in self._registry.snapshot():
            if upload_of(adapter) is not None:
                expired_now, stale = _reap_step(adapter, now, ttl)
                outcome = _reap_outcome(expired_now, stale)
                if outcome == "expired":
                    self._drop_catalog_row(adapter, source_id)
                    expired += 1
                elif outcome == "reclaimed":
                    # Safe without a compare-and-remove: a tombstone is
                    # terminal and its id cannot be re-registered while it
                    # stands, so this is still the adapter the snapshot saw.
                    self._registry.unregister(source_id)
                    reclaimed += 1
                    logger.info(f"Reclaimed discarded upload {source_id}")
            # ...and every tensor attached to it -- member, uploaded field or
            # label set -- which are tracked on the source rather than in the
            # registry, and take the identical step.
            tensor_expired, tensor_reclaimed, emptied_now = self._reap_tensors(
                adapter,
                source_id,
                _attached(adapter),
                # Lazy: an adapter with none to reap may not define this at all
                # ("outside this package", per _attached). Bound as a default so
                # the callable doesn't chase the loop's own name.
                lambda field, adapter=adapter: adapter.detach_tensor(field),
                now,
                ttl,
            )
            expired += tensor_expired
            reclaimed += tensor_reclaimed
            if (
                not emptied_now
                and isinstance(adapter, RegisterAdapter)
                and self._reclaim_empty_source(source_id, adapter, now, ttl)
            ):
                reclaimed += 1
        return expired, reclaimed

    def _reap_tensors(
        self,
        adapter: Any,
        source_id: str,
        tensors: Dict[str, Any],
        detach: Any,
        now: float,
        ttl: float,
    ) -> Tuple[int, int, bool]:
        """One reap pass over *adapter*'s attachment index.

        Every kind takes the identical expire-then-reclaim step, which is what
        one index buys. Returns ``(expired, reclaimed, reclaimed_any)`` -- the
        last only meaningful on a registered source, which uses it to decide
        whether the collection itself just went empty.
        """
        expired = reclaimed = 0
        reclaimed_any = False
        for field, tensor in list(tensors.items()):
            expired_now, stale = _reap_step(tensor, now, ttl)
            outcome = _reap_outcome(expired_now, stale)
            if outcome == "expired":
                self._unlist(adapter, field)
                expired += 1
            elif outcome == "reclaimed":
                detach(field)
                reclaimed += 1
                reclaimed_any = True
                logger.info(f"Reclaimed discarded tensor {source_id}/{field}")
        return expired, reclaimed, reclaimed_any

    def _reclaim_empty_source(
        self, source_id: str, adapter: RegisterAdapter, now: float, ttl: float
    ) -> bool:
        """Drop a registered source with no live tensor, quiet past *ttl*.

        Empty means every tensor gone, tombstones included: a discarded one is
        still reachable to whoever is polling it, so the source it belongs to
        has to outlive it. :meth:`reap` skips this for a source it emptied in
        the same pass, so a caller replacing a discarded tensor gets a full TTL
        to do it in, and the clock is touched on every add, so a source being
        filled is never a candidate however long the filling takes.

        What this closes is an abandoned ``register_source`` leaving a
        directory and a catalog row nothing will ever reach.
        """
        if adapter.attached_tensors:
            return False
        if now - adapter.touched_at < ttl:
            return False
        self._registry.unregister(source_id)
        self._drop_row(source_id)
        close_adapter(adapter)
        adapter.dispose_store()
        logger.info(f"Reclaimed empty registered source {source_id}")
        return True

    def start_sweep(self) -> None:
        """Run :meth:`reap` on a daemon thread until :meth:`stop_sweep`.

        A no-op when ``ttl`` is 0. The interval is a quarter of the TTL,
        clamped to [1 s, 60 s] and recomputed each pass, so a retune takes
        effect without a restart -- the same shape as the handle reaper.
        """
        if self.ttl <= 0 or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="upload-reaper", daemon=True
        )
        self._thread.start()

    def stop_sweep(self) -> None:
        """Stop the sweep thread and wait for it; idempotent."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(max(1.0, min(self.ttl / 4.0, 60.0))):
            try:
                self.reap()
            except Exception:  # pragma: no cover - the sweep must never die
                logger.debug("upload reap failed", exc_info=True)

    @staticmethod
    def _parse_metadata_json(metadata_json: str) -> dict:
        """Parse the request's ``metadata_json`` into an OME-metadata dict,
        translating a malformed payload into a legible Flight error at the create
        boundary.

        A bare ``json.loads`` would raise ``JSONDecodeError``: on the DoPut path
        it is swallowed by the command-discrimination try (mis-surfaced as
        "Invalid upload command"), and on the ``add_tensor`` Flight action it
        escapes as a generic internal error. Either way the client gets no
        actionable signal, so map it to ``FlightServerError`` here (biopb/biopb#354).

        Well-formed JSON that isn't an object (e.g. ``"123"`` or ``"[...]"``) is
        rejected too: callers spread the result into adapter metadata / a
        ``.zattrs``, both of which require a mapping, so a non-dict must fail here
        rather than surface as a confusing error downstream.
        """
        try:
            parsed = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            raise flight.FlightServerError(f"invalid metadata_json: {e}") from e
        if not isinstance(parsed, dict):
            raise flight.FlightServerError(
                f"invalid metadata_json: expected a JSON object, got {type(parsed).__name__}"
            )
        return parsed
