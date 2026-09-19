"""The upload half of an adapter that receives its bytes over DoPut.

A source that is written rather than read has state a file-backed source does
not: which chunks have arrived, whether the upload is complete, and whether its
owner has since given up on it. That state lives **on the adapter**, because
every part of it is either a field the adapter already holds (the expected
chunk count is ``shape / chunk_shape``) or a consequence of a call the adapter
already receives (``put_chunk``). A record kept beside the registry had to be
created after every registration, dropped after every unregistration, and
consulted before every write, by hand, in three places.

:class:`WritableSource` is a mixin for the two writable formats
(``CachedSourceAdapter``, ``ZarrAdapter``). It owns the shared half --
progress, status, disposal -- and leaves the format its own half: how a chunk
is stored (``_store_chunk``), how an upload of this kind is created
(``create_upload``), and what its store needs told when the upload ends
(``_dispose_store`` on discard, ``_mark_store_finished`` on finish). The DoPut
boundary (``serving.upload_manager``) picks the kind, registers the result,
keeps the catalog row of a ``durable`` kind in step, and translates
exceptions; nothing else about an upload lives there.

Wrappers: ``SourceRegistry.register`` may wrap an adapter (``normalize_adapter``),
and a wrapper forwards attributes rather than inheriting this class. Uploads are
refused unless canonical, so today none is wrapped; a caller that needs the
upload half should still reach it by attribute (``adapter.upload``,
``adapter.put_chunk``) and not by ``isinstance``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Set, Tuple

import pyarrow as pa
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.chunk import encode_chunk_id
from biopb_tensor_server.core.errors import UploadDiscardedError, UploadSealedError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Wire-facing upload state (the string values are the ``upload_status`` API
    contract, parsed by the client SDK).

    Two terminal states, not three: a job that dies has nothing to say that
    ``DISCARDED`` with a reason does not already say (biopb/biopb#1). A separate
    FAILED would never refuse a write -- a crashed job is not writing -- and
    would never be the state its own uploader learns, since that uploader set
    it. Its only reader is a remote poller, and the reason string is what that
    poller actually wants.
    """

    PENDING = "PENDING"
    READY = "READY"
    DISCARDED = "DISCARDED"
    UNKNOWN = "UNKNOWN"


def unknown_upload_status(source_id: str) -> Dict[str, Any]:
    """The status of a source that is not an upload, or is no longer registered."""
    return {
        "source_id": source_id,
        "state": UploadStatus.UNKNOWN.value,
        "expected_chunks": 0,
        "uploaded_chunks": 0,
        "reason": "",
    }


def upload_of(adapter: object) -> Optional[UploadProgress]:
    """The upload progress an adapter is tracking, or None.

    By attribute, not ``isinstance`` (the mixin's own contract -- see the
    module docstring): a caller outside this module that needs to know
    whether an adapter is an upload, and whether it's discarded, should go
    through this rather than repeating ``getattr(adapter, "upload", None)``.
    """
    return getattr(adapter, "upload", None)


def _expected_chunk_count(shape: Sequence[int], chunk_shape: Sequence[int]) -> int:
    count = 1
    for dim, chunk in zip(shape, chunk_shape, strict=True):
        count *= ceil(dim / chunk)
    return count


@dataclass
class UploadProgress:
    """How far an upload has got, and whether anyone still wants it.

    ``READY`` once the producer calls ``finish``, and only then: the chunk
    count is a coverage *proxy*, and a cache-backed source accepts arbitrary
    bounds, so off-grid writes can reach the expected count without tiling the
    array or never reach it at all. Both terminal states seal the source
    against further writes. ``DISCARDED`` keeps the adapter registered as a
    tombstone so a writer still unwinding learns *why* its write failed rather
    than that its source never existed.

    ``updated_at`` bounds that afterlife (``UploadManager.reap``). One field
    serves both halves of reclamation, because for a discarded upload the last
    touch *is* the discard: a tombstone's age is how long since it was written,
    and a PENDING upload's is how long since it last made progress -- the
    signal for a job that died without discarding. ``time.monotonic``, matching
    :mod:`~biopb_tensor_server.adapters._handle_reaper`: these are intervals, so
    a wall clock that steps would corrupt them, and nothing outside this process
    reads it.
    """

    expected_chunks: int
    status: UploadStatus = UploadStatus.PENDING
    uploaded_chunk_ids: Set[bytes] = field(default_factory=set)
    reason: str = ""
    updated_at: float = field(default_factory=time.monotonic)
    # Set at discard, where the ids are dropped but how far the upload got is
    # still worth reporting -- it is how much work a poller learns was lost.
    final_chunk_count: Optional[int] = None
    # Guards this record. Lives here rather than beside the optional `_upload`
    # on WritableSource, so the two are never out of step with each other.
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def uploaded_chunks(self) -> int:
        if self.final_chunk_count is not None:
            return self.final_chunk_count
        return len(self.uploaded_chunk_ids)

    @property
    def is_discarded(self) -> bool:
        return self.status is UploadStatus.DISCARDED

    @property
    def is_sealed(self) -> bool:
        """No further chunk will be accepted -- finished, or given up on."""
        return self.status in (UploadStatus.READY, UploadStatus.DISCARDED)

    def touch(self, now: Optional[float] = None) -> None:
        self.updated_at = time.monotonic() if now is None else now

    def idle_for(self, now: float) -> float:
        """Seconds since the last touch, on the monotonic clock."""
        return now - self.updated_at

    def as_status_dict(self, source_id: str) -> Dict[str, Any]:
        return {
            "source_id": source_id,
            "state": self.status.value,
            "expected_chunks": self.expected_chunks,
            "uploaded_chunks": self.uploaded_chunks,
            "reason": self.reason,
        }


class WritableSource:
    """Mixin: an adapter whose bytes arrive through :meth:`put_chunk`.

    Must precede ``TensorAdapter`` in the bases so this ``put_chunk`` shadows
    the read-only default. Adapters set their own attributes rather than
    chaining ``__init__``, so the upload half is started explicitly with
    :meth:`begin_upload`; an adapter that never begins one (a catalogued zarr
    store, say) still accepts writes but tracks nothing, and reports UNKNOWN.
    """

    #: A durable upload owns bytes that outlive the process (a store on disk)
    #: and belongs in the catalog. A volatile one is readable by its returned id
    #: only: it has no removal hook, so a catalog row would dangle after
    #: eviction (biopb/biopb#265). The flag is the boundary's cue to keep the
    #: catalog row in step -- written at create, dropped at discard -- and the
    #: store itself is the kind's own to release (``_dispose_store``): it was
    #: minted under the server's ``write_dir``, so it is the server's to throw
    #: away (biopb/biopb#1059).
    durable: bool = False

    #: Every host class sets this before ``begin_upload`` is reachable; declared
    #: here so the mixin's own methods don't each need to silence the type
    #: checker for an attribute they assume rather than define.
    source_id: str

    _upload: Optional[UploadProgress] = None

    # -- the format's half -----------------------------------------------------

    @classmethod
    def create_upload(
        cls,
        name: str,
        desc: TensorDescriptor,
        *,
        metadata: Optional[dict],
        write_dir: Optional[Path],
    ) -> WritableSource:
        """Build an adapter for a new upload of this kind, with its upload begun.

        *name* is what followed the ``kind:`` prefix in the request (may be
        empty: the kind then mints one). *metadata* is the request's parsed
        ``metadata_json``, or None when it carried none. Raises ``ValueError``
        for a request this kind cannot serve; the boundary translates it.
        """
        raise NotImplementedError

    def _store_chunk(
        self,
        bounds: ChunkBounds,
        data: pa.Array | pa.ChunkedArray,
        expected_shape: Tuple[int, ...],
        dtype: Any,
    ) -> None:
        """Write one chunk into the backing store; the format's write contract.

        ``ZarrAdapter`` enforces chunk-grid alignment, ``CachedSourceAdapter``
        accepts arbitrary bounds. Raises ``ValueError`` for a chunk the contract
        refuses.
        """
        raise NotImplementedError

    def _dispose_store(self) -> None:  # noqa: B027 - concrete no-op default
        """Release the backing store once the upload is discarded.

        Called after the DISCARDED transition, **outside** ``progress.lock``,
        exactly once per upload. A kind whose bytes live only in the chunk
        cache has nothing to do (the LRU reclaims them); a kind with a store on
        disk removes it here. Must be safe against a write that passed
        :meth:`_refuse_write` a moment earlier -- the kind serializes the two.
        """

    def _mark_store_finished(self) -> None:  # noqa: B027 - concrete no-op default
        """Seal the store on disk; READY is announced only once this returns.

        Called by :meth:`finish` **before** the READY transition, outside
        ``progress.lock``. A durable store carries a pending marker from
        creation so a crash leaves something a restart can recognize and
        remove; this is where the marker is flipped, and it is the last write
        the store's metadata ever sees. Raises ``OSError`` if the seal cannot
        be written, in which case the upload stays PENDING and ``finish`` is
        retried.
        """

    def upload_response(self, desc: TensorDescriptor) -> TensorDescriptor:
        """The descriptor ``create_tensor`` answers with.

        Echoes the request under the minted ``source_id``. Physical calibration
        is deliberately not echoed here: a kind adds it only if a later read
        will reproduce it verbatim (issue #272).
        """
        return TensorDescriptor(
            array_id=self.source_id,
            dim_labels=desc.dim_labels,
            shape=desc.shape,
            chunk_shape=desc.chunk_shape,
            dtype=desc.dtype,
        )

    def discard(self, reason: str = "") -> Dict[str, Any]:
        """Give up on this upload: refuse further writes, keep the record.

        Disposal, not cancellation (biopb/biopb#1). Stopping the compute belongs
        to whoever runs it; this is only the fate of the output. The adapter
        stays registered as a tombstone, so a job still unwinding reaches it
        and learns the reason, rather than reaching nothing and concluding the
        source never existed. Reclaiming the tombstone is the sweep's job
        (``UploadManager.reap``, by ``updated_at``), not this method's.

        Total and idempotent: returns the resulting status either way. The
        first reason wins, and with it the first timestamp -- a second discard
        must not extend the tombstone's life.

        A durable kind releases its store on the first discard
        (:meth:`_dispose_store`); its catalog row is the boundary's to drop.
        """
        source_id = self.source_id
        progress = self._upload
        if progress is None:
            return unknown_upload_status(source_id)
        with progress.lock:
            transitioned = self._discard_locked(progress, reason)
            status = progress.as_status_dict(source_id)
        if transitioned:
            self._dispose_store()
        return status

    def _discard_locked(
        self, progress: UploadProgress, reason: str, now: Optional[float] = None
    ) -> bool:
        """The DISCARDED transition; caller holds ``progress.lock``.

        Returns whether this call made the transition, so the caller can run
        :meth:`_dispose_store` once, after releasing the lock. *now* stamps the
        tombstone on the sweep's clock when expiry is the cause, so its age is
        measured from the same instant the sweep judged the upload dead."""
        if progress.is_discarded:
            return False
        progress.status = UploadStatus.DISCARDED
        progress.reason = reason
        # No more chunks will arrive, so the ids are dead weight for as long
        # as the tombstone lives. The count outlives them.
        progress.final_chunk_count = progress.uploaded_chunks
        progress.uploaded_chunk_ids = set()
        progress.touch(now)
        logger.info(f"Discarded upload {self.source_id}: {reason or 'no reason given'}")
        return True

    def reap_step(self, now: float, ttl: float) -> Tuple[bool, Optional[float]]:
        """The reclaim sweep's two questions for this upload, one lock hold.

        Returns ``(expired, tombstone_age)``:

        - *expired*: a PENDING upload with no progress for *ttl* seconds was
          just discarded here, with a reason -- one terminal transition, not
          a second path into oblivion. The check and the transition share the
          lock hold so a ``finish`` racing the sweep either lands first (and
          the upload stays READY) or is refused as discarded; it can never be
          undone. A durable kind's store goes with it (:meth:`_dispose_store`).
        - *tombstone_age*: seconds since this upload was discarded, or
          ``None`` if it was not -- including right after this call just
          discarded it, since a fresh tombstone's age is not yet the sweep's
          concern.
        """
        progress = self._upload
        if progress is None:
            return False, None
        with progress.lock:
            if progress.status is UploadStatus.PENDING and progress.idle_for(now) > ttl:
                self._discard_locked(progress, f"expired: no write for {ttl:g} s", now)
                expired = True
            else:
                expired = False
            age = progress.idle_for(now) if progress.is_discarded else None
        if expired:
            self._dispose_store()
            return True, None
        return False, age

    def finish(self) -> Dict[str, Any]:
        """Seal this upload: PENDING -> READY, no further chunks.

        The only route to READY, and it replaced a count-derived one that could
        not coexist with it: a source that counted its way to READY stayed
        writable, where a finished one is sealed -- two security states under
        one state name. The count was never a completeness check anyway, since
        ``CachedSourceAdapter`` takes arbitrary bounds; it now only reports
        progress.

        Idempotent, so a retried ``finish`` is not an error -- and, matching
        :meth:`discard`, a no-op past the first call: the first seal's touch
        is the one a reclaim sweep should see, not a retry's.

        The store is sealed on disk (:meth:`_mark_store_finished`) **before**
        READY is announced, never after: a poller that has seen READY must
        find the store finished after a crash, not pending and swept away at
        the next boot. The seal runs outside ``progress.lock`` (it takes the
        kind's write lock, which orders ahead of this one), so the transition
        re-checks for a discard that landed in between -- the seal on a store
        that discard has already removed is what that path reports.
        """
        progress = self._upload
        if progress is None:
            return unknown_upload_status(self.source_id)
        with progress.lock:
            if progress.is_discarded:
                raise UploadDiscardedError(self.source_id, progress.reason)
            if progress.status is UploadStatus.READY:
                return progress.as_status_dict(self.source_id)
        try:
            self._mark_store_finished()
        except OSError:
            with progress.lock:
                if progress.is_discarded:
                    raise UploadDiscardedError(self.source_id, progress.reason)
            raise
        with progress.lock:
            if progress.is_discarded:
                raise UploadDiscardedError(self.source_id, progress.reason)
            if progress.status is not UploadStatus.READY:
                progress.status = UploadStatus.READY
                progress.touch()
                logger.info(
                    f"Finished upload {self.source_id}: "
                    f"{progress.uploaded_chunks}/{progress.expected_chunks} chunks"
                )
            return progress.as_status_dict(self.source_id)

    # -- the shared half -------------------------------------------------------

    def begin_upload(self, shape: Sequence[int], chunk_shape: Sequence[int]) -> None:
        """Start tracking an upload of ``shape`` written in ``chunk_shape`` units."""
        self._upload = UploadProgress(
            expected_chunks=_expected_chunk_count(list(shape), list(chunk_shape))
        )

    @property
    def upload(self) -> Optional[UploadProgress]:
        """The upload's progress, or None if this adapter is not tracking one."""
        return self._upload

    def upload_status(self) -> Dict[str, Any]:
        progress = self._upload
        if progress is None:
            return unknown_upload_status(self.source_id)
        with progress.lock:
            return progress.as_status_dict(self.source_id)

    def put_chunk(
        self,
        bounds: ChunkBounds,
        data: pa.Array | pa.ChunkedArray,
        expected_shape: Tuple[int, ...],
        dtype: Any,
    ) -> None:
        """Store one chunk and count it, if this upload still accepts writes."""
        self._refuse_write()
        self._store_chunk(bounds, data, expected_shape, dtype)
        self._mark_chunk(bounds)

    def _raise_if_discarded_locked(self, progress: UploadProgress) -> None:
        """Raise :class:`UploadDiscardedError` if *progress* is discarded.

        Caller holds ``progress.lock``. Shared by :meth:`_refuse_write` and
        :meth:`_refuse_if_discarded`, discarded checked first in the former: it
        carries a reason, which "already finished" does not.
        """
        if progress.is_discarded:
            raise UploadDiscardedError(self.source_id, progress.reason)

    def _refuse_write(self) -> None:
        """The two ways a write can arrive too late.

        There is no third: a source's id has one adapter for as long as it is
        registered (``UploadManager.create_tensor`` refuses a name collision),
        so a write cannot land in a stranger's source by naming its own. The
        one exception is by design: a discarded name is reclaimed after
        ``upload_ttl`` seconds, and a straggler quiet for that long writes
        into whoever took the name next.
        """
        progress = self._upload
        if progress is None:
            return
        with progress.lock:
            self._raise_if_discarded_locked(progress)
            if progress.is_sealed:
                raise UploadSealedError(self.source_id)

    def _refuse_if_discarded(self) -> None:
        """Raise :class:`UploadDiscardedError` if the upload has been given up on.

        The read path's own check (``CachedSourceAdapter.resolve_chunk_data``):
        a tombstone answers a still-unwinding reader the same reason it gives a
        writer rather than "no chunk here". The write path's equivalent is
        :meth:`_refuse_write`, which also refuses a sealed-but-not-discarded
        upload. Each boundary maps the exception to its own wire error
        (write -> ``FlightCancelledError``, read -> ``FlightServerError``).
        """
        progress = self._upload
        if progress is None:
            return
        with progress.lock:
            self._raise_if_discarded_locked(progress)

    def _mark_chunk(self, bounds: ChunkBounds) -> None:
        """Record that the chunk at *bounds* landed.

        Counting only: reaching ``expected_chunks`` no longer flips anything,
        because :meth:`finish` is the only route to READY. What the count is
        still for is the progress a poller watches.

        A seal between :meth:`_refuse_write` and here leaves the chunk stored
        but uncounted: a sealed upload must not be walked back into PENDING by
        a write nobody is waiting for, nor have its clock touched.
        """
        progress = self._upload
        if progress is None:
            return
        chunk_id = encode_chunk_id(self.source_id, bounds)
        with progress.lock:
            if progress.is_sealed:
                return
            progress.uploaded_chunk_ids.add(chunk_id)
            progress.touch()
