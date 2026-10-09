"""Writable-server upload path: the DoPut boundary.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). What is left
here is only what a boundary does:

- **Addressing** -- ``add_tensor`` takes
  ``<scheme>://<source_id>/@fields/<name>`` for a tensor uploaded onto any
  source, and ``zarr://<array_id>/@labels/<name>`` for a label set
  (biopb/biopb#1059). Both carry a marked segment, because a bare field is a
  native tensor id and the upload path mints none. **Nothing here creates a
  source**: the parent must already be registered, and an intermediate result
  goes on the scratch source (``adapters.scratch``). The scheme names the store
  format and nothing else (``adapters.member_formats.STORE_FORMATS``), so the
  catalog row kept in step is always the parent's.
- **Error translation** -- adapters stay transport-agnostic and raise typed
  errors; this is where they become Flight errors.
- **Lookup** -- ``status`` / ``set_status`` find the adapter and hand over
  (``_locate``: the registry, then the parent's attachment index).
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

The manager registers the scratch source through the shared ``SourceRegistry``
and never holds a back-reference to the server, so the collaborators stay
acyclic.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._writable import (
    UploadStatus,
    unknown_upload_status,
    upload_of,
)
from biopb_tensor_server.adapters.fields import (
    create_field_upload,
    fields_root,
    source_fields_dir,
)
from biopb_tensor_server.adapters.labels import create_label_upload, labels_root
from biopb_tensor_server.adapters.members import member_marker
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.scratch import (
    SCRATCH_SOURCE_ID,
    ScratchSource,
)
from biopb_tensor_server.adapters.zarr import (
    UPLOAD_PENDING,
    upload_expires_at,
    upload_state,
)
from biopb_tensor_server.core.attached import FIELDS_SEGMENT
from biopb_tensor_server.core.axes import noncanonical_order
from biopb_tensor_server.core.chunk import (
    MAX_ARROW_BATCH_BYTES,
    estimate_chunk_bytes,
    get_bounds_from_chunk_id,
)
from biopb_tensor_server.core.errors import (
    SourceUnresolvedError,
    TensorResolutionError,
    UploadClosedError,
    UploadDiscardedError,
    UploadTransitionError,
    WriteNotSupportedError,
)
from biopb_tensor_server.core.labels import LABELS_SEGMENT, split_label_field
from biopb_tensor_server.serving.metadata_db import MetadataDatabase
from biopb_tensor_server.sources.source_registry import SourceRegistry

__all__ = [
    "DEFAULT_UPLOAD_TTL",
    "UploadManager",
    "UploadStatus",
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
    f"'<scheme>://<source_id>/{FIELDS_SEGMENT}/<name>' to add a tensor to a "
    f"source, or 'zarr://<array_id>/{LABELS_SEGMENT}/<name>' to add a label "
    "set to one of its tensors"
)


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


def _reap_step(
    adapter: Any, now: float, ttl: float, wall_now: float
) -> Tuple[bool, bool]:
    """One upload's turn in the sweep: ``(just expired, tombstone is stale)``.

    Both halves come from ``WritableSource.reap_step``; this only turns the
    tombstone's age into the sweep's yes/no, so the registered kinds and the
    label sets attached to a source take the identical step.

    Two clocks, because the sweep asks two kinds of question: *now* is
    monotonic and measures how long something has been idle, *wall_now* is the
    wall clock a recorded deadline is written against.
    """
    expired, age = adapter.reap_step(now, ttl, wall_now)
    return expired, age is not None and age > ttl


def _past_deadline(expires_at: Optional[float]) -> bool:
    """Whether a recorded deadline has already gone by, on the wall clock.

    Separate from ``WritableSource.expired`` because the boot sweep asks about
    a *directory*, before anything has opened it as an adapter.
    """
    return expires_at is not None and time.time() >= expires_at


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

    def _deadline_for(self, parent: Any, desc: TensorDescriptor) -> Optional[float]:
        """When a tensor added to *parent* stops being served, or None.

        The request's ``ttl_seconds`` is what the producer asked for, unset
        meaning no deadline; the parent's ``max_upload_ttl`` is what the source
        allows, and applies to an unset request too. The cap wins, so a request
        can only shorten a lifetime -- which makes it a policy, not a default.
        """
        requested = desc.ttl_seconds if desc.HasField("ttl_seconds") else None
        if requested is not None and requested <= 0:
            raise flight.FlightServerError(
                "add_tensor: ttl_seconds is a lifetime in seconds and must be "
                "positive; leave it unset to ask for no deadline."
            )
        cap = getattr(parent, "max_upload_ttl", None)
        ttl = float(requested) if requested is not None else None
        if cap is not None:
            ttl = float(cap) if ttl is None else min(ttl, float(cap))
        return None if ttl is None else time.time() + ttl

    @property
    def write_dir(self) -> Optional[Path]:
        """The subtree this server's uploads live under, or None if read-only.

        Public because every uploaded tensor has the one layout under it
        (``fields/<source_id>/<name>``), so naming a store takes the root
        rather than the source that holds it.
        """
        return self._write_dir

    # -- lookup ----------------------------------------------------------------

    def _locate(self, upload_id: str) -> Tuple[Any, Any, Optional[str]]:
        """The adapter tracking upload *upload_id*, its source, and its field.

        Every upload is a tensor of a source that already exists, so this is
        one lookup in two halves: the source from the registry, then the field
        from that source's attachment index, whatever kind of tensor it is.
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
        return self._registry.attached(source_id, field), parent, field

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
        the registered kinds, the parent's attachment for a label set, and any
        annotations drawn against the array_id (biopb/biopb#1155) -- the name
        is free after this and must not hand a later tensor another one's
        rows.

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
        self._forget_rois(array_id)
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
            self._registry.attachment_changed(parent.source_id)
        return status

    def _delete_adopted_tensor(self, array_id: str) -> Dict[str, Any]:
        """Remove a listed tensor that this life of the server did not upload.

        A field or sidecar re-attached at registration carries no upload
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
        adapter = self._registry.detach(source_id, field)
        if adapter is None:
            return unknown_upload_status(array_id)
        adapter.delete_store()
        self._forget_rois(array_id)
        logger.info(f"Deleted tensor {array_id}")
        return unknown_upload_status(array_id)

    # -- listing ---------------------------------------------------------------

    def _unlist(self, parent: Any, field: Optional[str]) -> None:
        """Take a tensor out of its source's listing, if it was in it.

        Nothing is detached: a tensor leaves the listing by ceasing to be
        readable, and stays reachable as the tombstone a straggler polls until
        the reclaim sweep drops it.
        """
        if parent is None or field is None:
            return
        if self._registry.attached(parent.source_id, field) is None:
            return
        self._registry.attachment_changed(parent.source_id)

    def _add_label_set(
        self, parent: Any, field: str, req_desc: TensorDescriptor
    ) -> TensorDescriptor:
        """A new label set on a tensor that already exists (biopb/biopb#1059).

        The set is attached to its source, which routes its own writes and its
        status polls; it is not *listed* until it reaches READY
        (``Attachments.label_sets``). No catalog write at create: the row the
        source already has still describes what a reader may see.

        It takes a deadline the same way a field does: a set is an uploaded
        tensor, so a source that caps lifetimes caps this one too.
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
                attached=self._registry.attached_to(parent.source_id),
                metadata=metadata,
                expires_at=self._deadline_for(parent, req_desc),
            )
        except ValueError as e:
            raise flight.FlightServerError(f"add_tensor: {e}") from e
        self._registry.attach(parent.source_id, field, adapter)
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

    def _forget_rois(self, array_id: str) -> None:
        """Drop annotations on a discarded array_id; best-effort, like the row.

        Called wherever an upload's data is actually disposed of -- explicit
        discard, either half of the reap sweep, or deleting an adopted
        tensor -- so a name the reaper frees is never handed to a later,
        unrelated tensor with another one's annotations still attached
        (biopb/biopb#1155).
        """
        if self._metadata_db is None:
            return
        try:
            self._metadata_db.discard_array_rois(array_id)
        except Exception as e:
            logger.warning(f"Failed to drop annotations for {array_id}: {e}")

    def discard_unfinished_stores(self) -> int:
        """Delete the stores of uploads a previous server never finished.

        A durable store is born ``pending`` and marked ``ready`` when its
        upload reaches READY (``adapters.zarr`` upload marker). One still
        pending when this server starts belonged to an upload whose progress
        died with the process before anyone published it; nothing can publish
        it now, so it is removed rather than left for discovery to serve as a
        partial source. Runs from the server's
        constructor, before any source is registered. Returns the count.

        The layouts the server mints: the uploaded fields
        (``fields/<source_id>/<name>``, in either store format), the label
        sidecars (``labels/<source_id>/*.zarr``), and -- for one more release --
        the single-array stores the removed ``ome_zarr:`` kind left directly
        under ``write_dir``. They differ in what the catalog owes them, which is
        why they are swept separately.

        The per-source directories themselves are collected last
        (:meth:`_prune_empty_source_dirs`), once every store that is going has
        gone. They are not stores and are not counted.
        """
        write_dir = self._write_dir
        if write_dir is None or not write_dir.is_dir():
            return 0
        removed = 0
        for store in sorted(fields_root(write_dir).glob("*/*")):
            # A field has no row of its own: it is a tensor of its source, whose
            # row is the reconciler's or is rebuilt when a registered source is
            # adopted -- either way after this sweep. Only the *pending* ones go
            # -- a published field is the only copy of what someone uploaded,
            # and is kept even once its source has gone away.
            if store.is_dir():
                removed += self._remove_unfinished(store)
        for store in sorted(labels_root(write_dir).glob("*/*.zarr")):
            # A sidecar has no row of its own either, for the same reason.
            removed += self._remove_unfinished(store)
        removed += self._drop_legacy_ome_zarr_stores(write_dir)
        self._prune_empty_source_dirs(fields_root(write_dir))
        self._prune_empty_source_dirs(labels_root(write_dir))
        return removed

    def _drop_legacy_ome_zarr_stores(self, write_dir: Path) -> int:
        """Retire what the removed ``ome_zarr:`` kind left under ``write_dir``.

        Such a store is a single array, not a collection, so there is nothing
        here that could adopt it and its catalog row has had no adapter behind
        it since the kind went. The row goes either way; the **bytes** go only
        if the upload never finished, because a finished store is the user's
        data and this server no longer has a claim to it -- it is left where it
        is, for them to point a discovery root at. Returns how many stores were
        removed.
        """
        removed = 0
        for store in sorted(write_dir.glob("*.zarr")):
            self._drop_row(OmeZarrAdapter.upload_source_id(store))
            removed += self._remove_unfinished(store)
        return removed

    @staticmethod
    def _remove_unfinished(store: Path) -> bool:
        """Delete *store* if this life should not serve it; whether it was.

        Two reasons, both read off the marker before any adapter is asked to
        open the directory -- through ``member_marker``, so it is found in
        whichever file this store's format keeps it in and neither format is
        privileged.

        **Still pending**: a crash left it half written, and nothing here can
        finish it.

        **Past its deadline**: its lifetime ran out while the server was down.
        Swept rather than adopted and left to the first sweep, which would
        serve expired bytes until it came round.
        """
        marker = member_marker(store)
        if upload_state(marker) == UPLOAD_PENDING:
            why = "unfinished"
        elif _past_deadline(upload_expires_at(marker)):
            why = "expired"
        else:
            return False
        shutil.rmtree(store, ignore_errors=True)
        logger.info(f"Removed {why} upload store {store}")
        return True

    @staticmethod
    def _prune_empty_source_dirs(root: Path) -> None:
        """Drop the per-source directories under *root* that hold nothing.

        ``fields/<source_id>/`` and ``labels/<source_id>/`` are minted
        implicitly by ``store.mkdir(parents=True)`` and nothing removes them:
        disposing an upload takes its store, and the last one out leaves the
        directory behind. One empty directory per source that has ever held an
        upload, never collected.

        Here rather than at the disposal sites because this runs from the
        constructor, before any source is registered, so it cannot race the
        mkdir it undoes -- ``mkdir(parents=True)`` creates the parent and the
        store separately, and a sweep landing between the two would leave the
        upload creating them with nowhere to write. ``rmdir`` is the guard
        itself: it refuses a directory that still holds a store, which is
        exactly the directory to keep.
        """
        if not root.is_dir():
            return
        for source_dir in sorted(root.glob("*")):
            if not source_dir.is_dir():
                continue
            try:
                source_dir.rmdir()
            except OSError:
                # Still holds a store, or the OS would not part with it. Both
                # are reasons to leave it alone; neither is worth a log line on
                # every boot.
                continue
            logger.info(f"Removed the empty upload directory {source_dir}")

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

    @staticmethod
    def _require_bounded_grid(req_desc: TensorDescriptor) -> None:
        """Reject an upload whose write grid exceeds the Arrow batch ceiling.

        The write grid is the transfer grid: a read plans on the chunks that
        were stored, and a chunk above ``MAX_ARROW_BATCH_BYTES`` cannot be
        streamed whole nor re-split into chunks no upload stored. Splitting it
        before upload is the client-side fix.
        """
        shape = [int(dim) for dim in req_desc.shape]
        if not shape or len(req_desc.chunk_shape) != len(shape):
            return
        grid = tuple(
            min(max(1, int(chunk)), dim)
            for chunk, dim in zip(req_desc.chunk_shape, shape, strict=True)
        )
        nbytes = estimate_chunk_bytes(grid, req_desc.dtype)
        if nbytes > MAX_ARROW_BATCH_BYTES:
            raise flight.FlightServerError(
                f"add_tensor: chunk_shape {list(grid)} is {nbytes} bytes, above "
                f"the {MAX_ARROW_BATCH_BYTES}-byte limit of one chunk; "
                f"upload in smaller chunks."
            )

    def add_tensor(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """Add a tensor to a source that already exists; answer its descriptor.

        **Nothing here creates a source.** An upload names what it is adding
        to: ``<scheme>://<source_id>/@fields/<name>`` for a tensor of a source,
        ``zarr://<array_id>/@labels/<name>`` for a label set of one of its
        tensors. A result that belongs to no source of the user's goes on the
        scratch source, at the fixed id a writable server always serves
        (``adapters.scratch.SCRATCH_SOURCE_ID``). The scheme names the store
        format (``member_formats.STORE_FORMATS``) and nothing else -- the answered
        ``array_id`` carries none, because the format is a property of the
        stored tensor, read off its directory at the next registration.

        **Every uploaded tensor is addressed under a marked segment**
        (:mod:`~biopb_tensor_server.core.attached`). A bare field is a native
        tensor id, which is the format's to mint, so the upload path never
        answers one -- and the kind is read off the field, not the parent.

        A field is taken for as long as its tensor is served. Replacing one is
        a discard first, which is what makes an ``array_id`` name a single
        attempt: a second add under a live name would otherwise land on the
        first's adapter, leaving the first writer's chunks and transitions in a
        tensor that is no longer its own with nothing to tell it so.
        """
        self._require_canonical_axes(req_desc)
        self._require_bounded_grid(req_desc)
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
        try:
            parent = self._registry.get_registered(source_id)
        except (SourceUnresolvedError, TensorResolutionError) as exc:
            raise flight.FlightServerError(f"add_tensor: {exc}") from exc
        if parent is None:
            raise flight.FlightServerError(
                f"add_tensor: {req_desc.array_id!r} names no source "
                f"{source_id!r}. A tensor is added to a source the server "
                f"already serves; it does not create one. Use "
                f"{SCRATCH_SOURCE_ID!r} for a result that belongs to no source "
                f"of yours."
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
                "add_tensor: metadata is the source's, not a tensor's, and an "
                "uploaded tensor inherits its source's. A tensor declares "
                "shape, dtype, chunk grid and axes; the one exception is a "
                "label set's 'image-label' block, which rides on its own "
                "add_tensor."
            )
        adapter = self._create_field(
            parent, field, scheme, req_desc, self._deadline_for(parent, req_desc)
        )
        # Attached, not listed: this routes the tensor's own writes, and the
        # published gate keeps it out of the source's tensors until READY. So no
        # catalog write is owed -- the row the source has still describes what a
        # reader may see.
        self._registry.attach(parent.source_id, field, adapter)
        logger.info(f"Added {scheme} tensor: {adapter.array_id}")
        return adapter.upload_response(req_desc)

    def _create_field(
        self,
        parent: Any,
        field: str,
        scheme: str,
        desc: TensorDescriptor,
        expires_at: Optional[float] = None,
    ) -> Any:
        """Mint an uploaded tensor beside its source, whatever kind it is.

        One layout for every source: a discovered one's bytes are the user's
        and the scratch source holds none, so neither has anywhere of its own
        to put a tensor. ``create_field_upload`` refuses a field that is not
        ``@fields/<name>``.
        """
        try:
            return create_field_upload(
                parent,
                field,
                scheme,
                desc,
                fields_dir=fields_root(self._write_dir),
                attached=self._registry.attached_to(parent.source_id),
                expires_at=expires_at,
            )
        except ValueError as e:
            raise flight.FlightServerError(f"add_tensor: {e}") from e

    def install_scratch(self, max_ttl: Optional[float]) -> Optional[str]:
        """Put the scratch source on the registry and in the catalog.

        Neither half is a walk: the id is fixed, and the source keeps no
        directory of its own -- its tensors come back through the
        registry's attachments, adopted from ``fields`` at boot.

        *max_ttl* caps every upload added here, an unset one included; None
        leaves them undated. Returns the id, or None on a server with no
        ``write_dir``, where nothing can be added anyway.
        """
        if self._write_dir is None:
            return None
        fields_dir = source_fields_dir(fields_root(self._write_dir), SCRATCH_SOURCE_ID)
        adapter = ScratchSource(max_ttl, fields_dir=fields_dir)
        registered = self._registry.register(SCRATCH_SOURCE_ID, adapter)
        self._sync_row(SCRATCH_SOURCE_ID, registered)
        logger.info(f"Serving the scratch source as {SCRATCH_SOURCE_ID}")
        return SCRATCH_SOURCE_ID

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
        # Not DataType.to_pandas_dtype(): it imports pandas, which the server
        # does not depend on.
        dtype = pa.array([], type=data_column.type).to_numpy(zero_copy_only=False).dtype
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

    def reap(
        self, now: Optional[float] = None, wall_now: Optional[float] = None
    ) -> Tuple[int, int]:
        """One sweep over the registry; the unit the reclaim thread runs.

        Two halves, two clocks -- see *now* and *wall_now* below. A PENDING
        upload with no write for ``ttl`` seconds is discarded with a reason -- the job that owned it died, and
        a discard is the one terminal transition there is, so a straggler
        that writes later is refused the same way it would be after an
        explicit discard. A tombstone that has stood for ``ttl`` seconds is
        unregistered: its name is free again and its status reads UNKNOWN,
        which a poller already treats as fail-fast (biopb/biopb#109). An
        expired upload therefore takes two sweeps to vanish, and is a
        tombstone in between.

        **A deadline is swept at any state, idleness only at PENDING.** A
        tensor whose lifetime has run out is discarded wherever it is on the
        ladder -- READY is where a finished result spends its life, so a
        deadline that stopped applying there would be none. Idleness is
        PENDING's alone: past it nothing is expected to make progress, so an
        upload left READY without ever being finished keeps its name and stays
        writable, which is the cost of letting a producer publish early. A
        durable kind is swept like any other: its store goes with the discard
        and its catalog row is dropped here, since both were the server's own
        (biopb/biopb#1059).

        Every source's attached tensors take the same step, because a tensor is
        attached to its source rather than registered and would otherwise have
        no sweep at all: a quiet pending field or label set is discarded (its
        store with it) and unlisted, and its tombstone is later detached, which
        frees the name.

        The chunks a reclaimed tombstone wrote stay in the cache until the
        LRU evicts them. They are unreachable: a re-created name gets a fresh
        ``content_version`` namespace, so its chunk ids never collide with
        the old ones (``CachedSourceAdapter.next_content_version``).

        Returns ``(expired, reclaimed)`` counts. Both clocks are injectable
        for tests: *now* is monotonic, like ``updated_at``, and measures how
        long something has been idle; *wall_now* is unix seconds and is what a
        recorded deadline was written against (``WritableSource.expires_at``).
        A recorded deadline has to survive a restart, which a monotonic
        reading does not.

        ``ttl <= 0`` disables the sweep whole -- deadlines included: it is the
        knob for "do not reclaim on this server", and a deadline is
        reclamation.
        """
        ttl = self.ttl
        if ttl <= 0:
            return 0, 0
        if now is None:
            now = time.monotonic()
        if wall_now is None:
            wall_now = time.time()
        expired = reclaimed = 0
        for source_id, adapter in self._registry.snapshot():
            if upload_of(adapter) is not None:
                expired_now, stale = _reap_step(adapter, now, ttl, wall_now)
                if expired_now:
                    self._drop_catalog_row(adapter, source_id)
                    self._forget_rois(source_id)
                    expired += 1
                if stale:
                    # Safe without a compare-and-remove: a tombstone is
                    # terminal and its id cannot be re-registered while it
                    # stands, so this is still the adapter the snapshot saw.
                    self._registry.unregister(source_id)
                    reclaimed += 1
                    logger.info(f"Reclaimed discarded upload {source_id}")
        # ...and every tensor attached to a source -- uploaded field or label
        # set -- held by the registry and taking the identical step.
        for source_id, tensors in self._registry.attachment_snapshot():
            tensor_expired, tensor_reclaimed = self._reap_tensors(
                source_id, tensors, now, ttl, wall_now
            )
            expired += tensor_expired
            reclaimed += tensor_reclaimed
        return expired, reclaimed

    def _reap_tensors(
        self,
        source_id: str,
        tensors: Dict[str, Any],
        now: float,
        ttl: float,
        wall_now: float,
    ) -> Tuple[int, int]:
        """One reap pass over *source_id*'s attachments."""
        expired = reclaimed = 0
        for field, tensor in tensors.items():
            expired_now, stale = _reap_step(tensor, now, ttl, wall_now)
            if expired_now:
                self._unlist(self._registry.get(source_id), field)
                self._forget_rois(tensor.array_id)
                expired += 1
            if stale:
                self._registry.detach(source_id, field)
                reclaimed += 1
                logger.info(f"Reclaimed discarded tensor {source_id}/{field}")
        return expired, reclaimed

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
