"""Writable-server upload path: the DoPut boundary.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). What is left
here is only what a boundary does:

- **Kind selection** -- the ``cache:`` / ``ome_zarr:`` ``array_id`` prefix names
  the adapter class (``UPLOAD_KINDS``); the class builds its own upload
  (``create_upload``) and the manager registers it, syncing the catalog when
  the kind is durable. An ``array_id`` with no prefix and a ``/labels/``
  segment is the third kind (biopb/biopb#1059): it creates no source, but a
  tensor of one that already exists, so it is attached to its parent rather
  than registered and the catalog row it keeps in step is the parent's.
- **Error translation** -- adapters stay transport-agnostic and raise typed
  errors; this is where they become Flight errors.
- **Lookup** -- ``status`` / ``finish`` / ``discard`` / ``write_chunk`` find the
  adapter and hand over (``_locate``: the registry, or a parent's
  ``label_uploads``).
- **Reclamation** -- ``reap`` sweeps every upload by its ``updated_at``, in
  the registry and in each source's ``label_uploads``: one quiet past ``ttl``
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
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkUpload

from biopb_tensor_server.adapters._writable import (
    UploadStatus,
    WritableSource,
    unknown_upload_status,
    upload_of,
)
from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.adapters.labels import create_label_upload, labels_root
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, read_zattrs, upload_state
from biopb_tensor_server.core.axes import noncanonical_order
from biopb_tensor_server.core.errors import (
    UploadClosedError,
    UploadDiscardedError,
    WriteNotSupportedError,
)
from biopb_tensor_server.core.labels import split_label_field
from biopb_tensor_server.core.source_registry import SourceRegistry, close_adapter
from biopb_tensor_server.serving.metadata_db import MetadataDatabase

__all__ = [
    "DEFAULT_UPLOAD_TTL",
    "UPLOAD_KINDS",
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

#: ``array_id`` prefix -> the adapter class that builds an upload of that kind.
#: The prefixes are a wire contract, so the table is closed; what each kind
#: does with a request is the class's own (``create_upload``).
UPLOAD_KINDS: Dict[str, Type[WritableSource]] = {
    "cache": CachedSourceAdapter,
    "ome_zarr": OmeZarrAdapter,
}


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


def _label_uploads(adapter: Any) -> Dict[str, Any]:
    """The label sets an adapter is still filling, or none.

    By attribute, like :func:`upload_of`: the registry also holds adapters
    from outside this package, and one that knows nothing about labels has
    none of them.
    """
    return getattr(adapter, "label_uploads", None) or {}


def _reap_step(adapter: Any, now: float, ttl: float) -> Tuple[bool, bool]:
    """One upload's turn in the sweep: ``(just expired, tombstone is stale)``.

    Both halves come from ``WritableSource.reap_step``; this only turns the
    tombstone's age into the sweep's yes/no, so the registered kinds and the
    label sets attached to a source take the identical step.
    """
    expired, age = adapter.reap_step(now, ttl)
    return expired, age is not None and age > ttl


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
        """The adapter tracking upload *upload_id*, and its parent if it is a set.

        Two namespaces, one id: a ``cache:`` / ``ome_zarr:`` upload *is* a
        source and answers from the registry, while a label set is a tensor of
        a registered source and answers from that source's ``label_uploads``.
        Returns ``(adapter, parent, field)`` -- ``parent`` and ``field`` are
        None for the registered kinds -- with ``adapter`` None when nothing
        holds the id, which every caller already has to handle.
        """
        adapter = self._registry.get(upload_id)
        if adapter is not None:
            return adapter, None, None
        source_id, _, field = upload_id.partition("/")
        if not field:
            return None, None, None
        parent = self._registry.get(source_id)
        if parent is None:
            return None, None, None
        return _label_uploads(parent).get(field), parent, field

    def status(self, source_id: str) -> Dict[str, Any]:
        """The ``upload_status`` answer: UNKNOWN for anything not tracking an upload."""
        adapter, _, _ = self._locate(source_id)
        upload = upload_of(adapter)
        if upload is None:
            return unknown_upload_status(source_id)
        return upload.as_status_dict(source_id)

    def discard(self, source_id: str, reason: str = "") -> Dict[str, Any]:
        """Give up on an upload; see ``WritableSource.discard``.

        Total: a source that is not tracking an upload -- never was, or has
        since been reclaimed -- reads UNKNOWN rather than raising, so a retry
        after the tombstone is gone is not an error. A durable kind's store
        goes with the discard (the adapter's own), and its listing here: the
        catalog row for the registered kinds, the parent's attachment for a
        label set.
        """
        adapter, parent, field = self._locate(source_id)
        if upload_of(adapter) is None:
            return unknown_upload_status(source_id)
        status = adapter.discard(reason)
        if parent is None:
            self._drop_catalog_row(adapter, source_id)
        else:
            self._unlist_label_set(parent, field)
        return status

    # -- label sets ------------------------------------------------------------

    def _sync_parent_row(self, parent: Any) -> None:
        """Re-publish a parent's catalog row after its sets changed.

        ``sync_source_added`` is an upsert and ``catalog_tensors`` reads the
        sets off the adapter, so re-registering the parent is the whole of it
        (the ROI re-import it triggers is idempotent). Best-effort for the
        same reason every other catalog write on this path is: the catalog
        must not fail the upload.
        """
        if self._metadata_db is None:
            return
        try:
            self._metadata_db.sync_source_added(parent.source_id, parent)
        except Exception as e:
            logger.warning(
                f"Failed to re-sync {parent.source_id} to the catalog after a "
                f"label set changed (served, listing stale until reindex): {e}"
            )

    def _unlist_label_set(self, parent: Any, field: Optional[str]) -> None:
        """Take a set out of its parent's listing, if it was in it.

        A set that never finished was never listed, so this is a no-op for
        the ordinary discard of an upload in flight and a catalog write only
        where one is owed.
        """
        if field is None:
            return
        if parent.detach_label_set(field) is not None:
            self._sync_parent_row(parent)

    def _create_label_set(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """The third upload kind: a new set on a source that already exists.

        Reached when the request ``array_id`` carries no ``kind:`` prefix, so
        this is also where an unrecognized id is refused. The set is attached
        to its parent as an upload -- routable from here on, so its producer
        can poll it to READY, but not listed until ``finish``
        (``SourceAdapter.label_uploads``). No catalog write happens at create,
        which is the one place this kind differs from ``ome_zarr:``: there is
        no row of its own to write, and the parent's must not advertise a set
        whose bytes have not arrived.
        """
        source_id, _, field = req_desc.array_id.partition("/")
        if split_label_field(field) is None:
            raise flight.FlightServerError(
                f"Invalid array_id format: {req_desc.array_id}. Use a 'cache:' or "
                f"'ome_zarr:' prefix to create a source, or "
                f"'<image array_id>/labels/<name>' to upload a label set."
            )
        if self._write_dir is None:
            raise flight.FlightServerError(
                "create_tensor: write_dir is not configured, so there is nowhere "
                "to put a label set's sidecar"
            )
        parent = self._registry.get(source_id)
        if parent is None:
            raise flight.FlightServerError(
                f"create_tensor: {req_desc.array_id!r} names no registered source "
                f"{source_id!r}. A label set is a tensor of an image the server "
                f"already serves; it does not create one."
            )
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
            raise flight.FlightServerError(f"create_tensor: {e}") from e
        parent.attach_label_upload(field, adapter)
        logger.info(f"Created label upload: {adapter.array_id}")
        return adapter.upload_response(req_desc)

    def delete_labels(self, array_id: str) -> Dict[str, Any]:
        """Remove a finished uploaded set: unlist it, then delete its sidecar.

        The one new action of biopb/biopb#1059, and the only way a set is ever
        removed short of its parent going. Unlisted first, so no read can be
        routed to a store that is about to go; the store then goes through the
        adapter, which owns it (``LabelSetAdapter.delete_store``). The name is
        free again at once and safely, because the next set under it mints its
        own ``content_version`` and so cannot hit the cache entries this one
        leaves behind.

        Refused for anything ``detach_label_set`` does not reach, which is
        exactly the sets the upload path published -- this server's, or an
        earlier life's re-attached at registration. A set the format carries
        (an NGFF ``labels/`` group; a rasterized ``@ome``) is embedded, not
        attached, and an upload still in flight is not listed at all, so
        neither can be deleted here.
        """
        source_id, _, field = array_id.partition("/")
        parent = self._registry.get(source_id) if field else None
        adapter = parent.detach_label_set(field) if parent is not None else None
        if adapter is None:
            raise flight.FlightServerError(
                f"delete_labels: {array_id!r} is not a deletable label set. Only "
                f"a finished uploaded set can be deleted -- a set the file "
                f"carries is the file's, and one still uploading is not finished."
            )
        parent.detach_label_upload(field)
        adapter.delete_store()
        self._sync_parent_row(parent)
        logger.info(f"Deleted label set {array_id}")
        return {"array_id": array_id, "deleted": True}

    def _drop_catalog_row(self, adapter: WritableSource, source_id: str) -> None:
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

        A durable store is born ``pending`` and marked ``ready`` by ``finish``
        (``adapters.zarr`` upload marker). One still pending when this server
        starts belonged to an upload whose progress died with the process;
        nothing can finish it, so it is removed rather than left for
        discovery to serve as a partial source. Runs from the server's
        constructor, before any source is registered. Returns the count.

        The two layouts the server mints: ``write_dir/*.zarr``
        (``OmeZarrAdapter.create_upload``) and the label sidecars
        ``write_dir/labels/<source_id>/*.zarr``. They differ in what the
        catalog owes them, which is why they are swept separately.
        """
        write_dir = self._write_dir
        if write_dir is None or not write_dir.is_dir():
            return 0
        removed = 0
        for store in sorted(write_dir.glob("*.zarr")):
            if not self._remove_unfinished(store):
                continue
            # An ``ome_zarr:`` upload is a source of its own, and a persisted
            # catalog still carries the row written at its create in the life
            # that died. Nothing else will drop it -- write_dir is outside
            # every discovery root, so the reconciler never sees this id -- and
            # the row would otherwise stay browsable with no store behind it.
            self._drop_row(OmeZarrAdapter.upload_source_id(store))
            removed += 1
        for store in sorted(labels_root(write_dir).glob("*/*.zarr")):
            # A sidecar has no row of its own: it is a tensor of its parent,
            # whose row is rebuilt when that parent registers -- which is
            # after this sweep, since it runs from the constructor.
            removed += self._remove_unfinished(store)
        return removed

    @staticmethod
    def _remove_unfinished(store: Path) -> bool:
        """Delete *store* if it is still pending; whether it was."""
        if upload_state(read_zattrs(store)) != UPLOAD_PENDING:
            return False
        shutil.rmtree(store, ignore_errors=True)
        logger.info(f"Removed unfinished upload store {store}")
        return True

    def finish(self, source_id: str) -> Dict[str, Any]:
        """Seal an upload; see ``WritableSource.finish``.

        Unlike ``discard``, a source that is tracking no upload is an error
        rather than UNKNOWN: ``discard`` is total because a retry after the
        tombstone is reclaimed must not fail, whereas a ``finish`` naming
        nothing means the caller believes it has been writing somewhere it has
        not.
        """
        adapter, parent, field = self._locate(source_id)
        if upload_of(adapter) is None:
            raise flight.FlightServerError(
                f"finish: {source_id} is not an upload in progress"
            )
        try:
            status = adapter.finish()
        except UploadDiscardedError as e:
            raise _refused(e) from e
        except OSError as e:
            # The store could not be sealed; the upload stays PENDING and the
            # caller retries finish, so this is a server error, not a refusal.
            raise flight.FlightServerError(
                f"finish: could not seal {source_id} on disk: {e}"
            ) from e
        if parent is not None:
            # Sealed, so it is a tensor now: list it under its image and
            # re-publish the parent's row. After the store, never before --
            # the catalog must not name a set a restart would sweep away.
            parent.attach_label_set(field, adapter)
            self._sync_parent_row(parent)
        return status

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
            f"create_tensor: {why}. The data plane advertises canonical order on "
            f"every source (biopb/biopb#596); transpose the array before "
            f"uploading."
        )

    def create_tensor(self, req_desc: TensorDescriptor) -> TensorDescriptor:
        """Create a single-tensor source from a TensorDescriptor, return its
        resolved descriptor.

        Named for what it declares -- one tensor -- so ``create_tensor`` stays
        free for a multi-tensor source API.

        array_id format in request:
        - "cache:name" → cache-backed with given name
        - "cache:" → cache-backed with server-generated name
        - "ome_zarr:name" → zarr-backed with given name
        - "ome_zarr:" → zarr-backed with server-generated name

        A name is taken for as long as its source is registered. A named
        ``cache:`` upload has a deterministic id, so a second create under the
        same name would land on the first's adapter -- and replacing it would
        leave the first writer's chunks, and its ``finish``, landing in a
        source that is no longer its own, with nothing to tell it so (status
        is keyed by the id it still holds). Refusing the collision is what
        makes ``source_id`` alone name an attempt. Sealed sources are not an
        exception: a name that could be reclaimed by finishing is a name a
        straggler can be raced for. What does free a name is :meth:`reap`
        unregistering its tombstone, ``ttl`` seconds after the discard.
        """
        self._require_canonical_axes(req_desc)

        prefix, sep, name = req_desc.array_id.partition(":")
        kind = UPLOAD_KINDS.get(prefix) if sep else None
        if kind is None:
            # No kind prefix: the label kind, which creates a tensor of an
            # existing source rather than a source, and refuses anything that
            # is not shaped like one (biopb/biopb#1059).
            return self._create_label_set(req_desc)

        # Parsed at the boundary: a malformed payload is the request's fault and
        # must fail before the kind touches anything (biopb/biopb#354).
        metadata = (
            self._parse_metadata_json(req_desc.metadata_json)
            if req_desc.metadata_json
            else None
        )
        try:
            adapter = kind.create_upload(
                name, req_desc, metadata=metadata, write_dir=self._write_dir
            )
        except ValueError as e:
            raise flight.FlightServerError(str(e)) from e

        source_id = adapter.source_id
        registered = self._registry.register_new(source_id, adapter)
        if registered is None:
            # The refused create has already minted its store (``ome_zarr:``
            # creates one before it has an id to register), so release it here
            # rather than leave the server's own bytes on disk for the next
            # boot sweep to find. No catalog row was written yet.
            adapter.discard("create refused: the name is already taken")
            close_adapter(adapter)
            raise flight.FlightServerError(
                f"create_tensor: {req_desc.array_id!r} already exists as "
                f"{source_id}. A name stays taken while its source is "
                "registered -- for the life of the server once finished, and "
                "until the reclaim sweep passes once discarded; upload under a "
                f"new name, or '{prefix}:' for a server-minted one."
            )

        # Only a durable upload belongs in the catalog; a volatile one is
        # readable by its returned id but not enumerable (biopb/biopb#265).
        # Best-effort: a catalog write must not fail the upload.
        if kind.durable and self._metadata_db is not None:
            try:
                self._metadata_db.sync_source_added(source_id, registered)
            except Exception as e:
                logger.warning(
                    f"Failed to sync uploaded source {source_id} to catalog "
                    f"(readable by id, not listed): {e}"
                )

        logger.info(f"Created {prefix} upload: {source_id}")
        return adapter.upload_response(req_desc)

    def write_chunk(
        self, upload: ChunkUpload, reader: flight.MetadataRecordBatchReader
    ) -> None:
        """Hand one uploaded chunk to its source's ``put_chunk``.

        Each source format owns its write contract: OmeZarr/Zarr enforce
        chunk-grid alignment; cache-backed sources accept arbitrary bounds;
        read-only formats reject the write; an upload that is discarded or
        sealed refuses. Adapters stay transport-agnostic, so their errors
        become Flight errors here.

        Both refusals map to ``FlightCancelledError`` (:func:`_refused`): they
        share the one thing a writer must act on -- this upload is over, stop
        sending -- and a client discriminates on the type rather than the
        message (biopb/biopb#1). Which of the two it was rides in
        ``extra_info``.
        """
        table = reader.read_all()
        data_column = table.column(0)

        adapter, _, _ = self._locate(upload.source_id)
        if adapter is None:
            raise flight.FlightServerError(f"Source not found: {upload.source_id}")

        bounds = upload.bounds
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
            f"Uploaded chunk to {upload.source_id}: bounds={list(bounds.start)}-{list(bounds.stop)}"
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

        READY sources are not touched: a finished upload is a published
        result, and its lifetime is its reader's, not this sweep's. A durable
        kind is swept like any other: its store goes with the discard and its
        catalog row is dropped here, since both were the server's own
        (biopb/biopb#1059).

        Every registered source's label uploads take the same step, because a
        set is attached to its parent rather than registered and would
        otherwise have no sweep at all: a quiet pending set is discarded (its
        sidecar with it) and unlisted, and its tombstone is later detached,
        which is what frees the name.

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
                if expired_now:
                    self._drop_catalog_row(adapter, source_id)
                    expired += 1
                elif stale:
                    # Safe without a compare-and-remove: a tombstone is
                    # terminal and its id cannot be re-registered while it
                    # stands, so this is still the adapter the snapshot saw.
                    self._registry.unregister(source_id)
                    reclaimed += 1
                    logger.info(f"Reclaimed discarded upload {source_id}")
            # ...and the label sets being uploaded onto it, which are tracked
            # on the source rather than in the registry.
            for field, label_set in list(_label_uploads(adapter).items()):
                expired_now, stale = _reap_step(label_set, now, ttl)
                if expired_now:
                    self._unlist_label_set(adapter, field)
                    expired += 1
                elif stale:
                    adapter.detach_label_upload(field)
                    reclaimed += 1
                    logger.info(f"Reclaimed discarded label upload {source_id}/{field}")
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
        "Invalid upload command"), and on the ``create_tensor`` Flight action it
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
