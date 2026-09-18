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
is stored (``_store_chunk``) and how an upload of this kind is created
(``create_upload``). Disposal (``discard``) is shared too, gated by the
``durable`` flag: a durable kind (a real store on disk, catalogued) refuses it
outright rather than overriding the method. The DoPut boundary
(``serving.upload_manager``) picks the kind, registers the result and
translates exceptions; nothing else about an upload lives there.

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

    ``updated_at`` is what will bound that afterlife. One field serves both
    halves of reclamation, because for a discarded upload the last touch *is*
    the discard: a tombstone's age is how long since it was written, and a
    PENDING upload's is how long since it last made progress -- the signal for
    a job that died without discarding. ``time.monotonic``, matching
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

    def touch(self) -> None:
        self.updated_at = time.monotonic()

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
    #: eviction (biopb/biopb#265). Durable uploads also own something disposal
    #: can't release (a store on disk, a catalog row), so the base ``discard``
    #: below refuses for them.
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
        source never existed. Reclaiming the tombstone is a sweep over the
        registry by ``updated_at``, not this method's concern.

        Total and idempotent: returns the resulting status either way. The
        first reason wins, and with it the first timestamp -- a second discard
        must not extend the tombstone's life.

        Refuses for a durable kind (``ZarrAdapter``): a real ``.zarr`` on disk
        and a catalog row are not this call's to release.
        """
        source_id = self.source_id
        if self.durable:
            raise ValueError(
                f"discard: {source_id} is not a cache-backed upload. A "
                "zarr-backed source owns a .zarr directory and a catalog row, "
                "which this does not remove."
            )
        progress = self._upload
        if progress is None:
            return unknown_upload_status(source_id)
        with progress.lock:
            if not progress.is_discarded:
                progress.status = UploadStatus.DISCARDED
                progress.reason = reason
                # No more chunks will arrive, so the ids are dead weight for as
                # long as the tombstone lives. The count outlives them.
                progress.final_chunk_count = progress.uploaded_chunks
                progress.uploaded_chunk_ids = set()
                progress.touch()
                logger.info(
                    f"Discarded upload {source_id}: {reason or 'no reason given'}"
                )
            return progress.as_status_dict(source_id)

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
        """
        progress = self._upload
        if progress is None:
            return unknown_upload_status(self.source_id)
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

        There is no third: a source's id has one adapter for the life of the
        server (``UploadManager.create_tensor`` refuses a name collision), so
        a write cannot land in a stranger's source by naming its own.
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
