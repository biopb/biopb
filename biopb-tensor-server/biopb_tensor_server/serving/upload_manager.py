"""Writable-server upload path: the DoPut boundary.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). What is left
here is only what a boundary does:

- **Kind selection** -- the ``cache:`` / ``ome_zarr:`` ``array_id`` prefix names
  the adapter class (``UPLOAD_KINDS``); the class builds its own upload
  (``create_upload``) and the manager registers it, syncing the catalog when
  the kind is durable.
- **Error translation** -- adapters stay transport-agnostic and raise typed
  errors; this is where they become Flight errors.
- **Lookup** -- ``status`` / ``finish`` / ``discard`` / ``write_chunk`` find the
  adapter and hand over.
- **Reclamation** -- ``reap`` sweeps the registry by each upload's
  ``updated_at``: a PENDING upload quiet past ``ttl`` is discarded (a job that
  died), and a tombstone older than ``ttl`` is unregistered. The sweep thread
  is started and stopped by the server that owns the manager.

Progress, completion and disposal are the adapter's own
(:class:`~biopb_tensor_server.adapters._writable.WritableSource`), so there is no
second registry to keep in step with ``SourceRegistry``: an upload's state is
created with its adapter, lives as long as it is registered, and a discarded
one stays registered as a tombstone until reclaimed.

The manager registers created sources through the shared ``SourceRegistry`` and
never holds a back-reference to the server, so the collaborators stay acyclic.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

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
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.core.axes import noncanonical_order
from biopb_tensor_server.core.errors import (
    UploadClosedError,
    UploadDiscardedError,
    WriteNotSupportedError,
)
from biopb_tensor_server.core.source_registry import SourceRegistry, close_adapter
from biopb_tensor_server.serving.metadata_db import MetadataDatabase

__all__ = ["DEFAULT_UPLOAD_TTL", "UPLOAD_KINDS", "UploadManager", "UploadStatus"]

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

    def status(self, source_id: str) -> Dict[str, Any]:
        """The ``upload_status`` answer: UNKNOWN for anything not tracking an upload."""
        upload = upload_of(self._registry.get(source_id))
        if upload is None:
            return unknown_upload_status(source_id)
        return upload.as_status_dict(source_id)

    def discard(self, source_id: str, reason: str = "") -> Dict[str, Any]:
        """Give up on an upload; see ``WritableSource.discard``.

        Total: a source that is not tracking an upload -- never was, or has
        since been reclaimed -- reads UNKNOWN rather than raising, so a retry
        after the tombstone is gone is not an error. A kind that refuses raises
        ``ValueError``; callers today are in-process, so it is not translated.
        """
        adapter = self._registry.get(source_id)
        if upload_of(adapter) is None:
            return unknown_upload_status(source_id)
        return adapter.discard(reason)

    def finish(self, source_id: str) -> Dict[str, Any]:
        """Seal an upload; see ``WritableSource.finish``.

        Unlike ``discard``, a source that is tracking no upload is an error
        rather than UNKNOWN: ``discard`` is total because a retry after the
        tombstone is reclaimed must not fail, whereas a ``finish`` naming
        nothing means the caller believes it has been writing somewhere it has
        not.
        """
        adapter = self._registry.get(source_id)
        if upload_of(adapter) is None:
            raise flight.FlightServerError(
                f"finish: {source_id} is not an upload in progress"
            )
        try:
            return adapter.finish()
        except UploadDiscardedError as e:
            raise _refused(e) from e

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
            raise flight.FlightServerError(
                f"Invalid array_id format: {req_desc.array_id}. Use 'cache:' or "
                f"'ome_zarr:' prefix"
            )

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

        adapter = self._registry.get(upload.source_id)
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
        result, and its lifetime is its reader's, not this sweep's. Durable
        kinds are not touched either -- their bytes are on disk and in the
        catalog, which discard refuses to disown -- so a quiet ``ome_zarr``
        upload stays PENDING and writable.

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
            if upload_of(adapter) is None:
                continue
            expired_now, age = adapter.reap_step(now, ttl)
            if expired_now:
                expired += 1
                continue
            if age is not None and age > ttl:
                # Safe without a compare-and-remove: a tombstone is terminal
                # and its id cannot be re-registered while it stands, so this
                # is still the adapter the snapshot saw.
                self._registry.unregister(source_id)
                reclaimed += 1
                logger.info(f"Reclaimed discarded upload {source_id} after {age:.0f} s")
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
