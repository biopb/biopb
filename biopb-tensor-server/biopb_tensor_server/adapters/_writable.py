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
is stored (``_store_chunk``) and what its store needs told when the upload
ends (``_dispose_store`` on discard, ``_publish_store`` on publish). Minting
one belongs to whoever owns the layout (``adapters.registered.create_member``,
``adapters.labels.create_label_upload``); the DoPut boundary
(``serving.upload_manager``) attaches the result to its source, keeps the
catalog row in step, and translates exceptions. Nothing else about an upload
lives there.

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
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import pyarrow as pa
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.chunk import (
    default_transfer_chunk_shape,
    encode_chunk_id,
)
from biopb_tensor_server.core.errors import (
    UploadDiscardedError,
    UploadDiscardedReadError,
    UploadNotPublishedError,
    UploadSealedError,
    UploadTransitionError,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Wire-facing upload state (the string values are the ``upload_status`` API
    contract, parsed by the client SDK).

    One gate, not two. ``READY`` seals the upload against writes and opens it
    to reads in the same move, so there is no state in which a read and a write
    can both land. A read is cacheable and a write cannot undo one: a chunk
    that never arrived reads as zeros (``CachedSourceAdapter.get_data``), and a
    later write filling that hole has no way to invalidate an answer already
    stored under its ``chunk_id`` -- least of all in the client's own pool,
    which the server cannot reach. ``PENDING`` therefore refuses reads outright
    rather than zero-filling: before someone says the holes are meant to be
    holes, zeros and not-yet-arrived are the same picture.

    Reopening a READY upload is not expressible today, and nothing here
    forecloses it: invalidation would be a fresh ``content_version`` token in
    the store, which every cache on both sides of the wire keys by. See
    ``docs/upload-model.md``.

    One terminal state, not two: a job that dies has nothing to say that
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


#: The ladder a live upload climbs, lowest first. A transition moves up it or
#: stands still; ``DISCARDED`` is off the ladder and reachable from every rung.
#: ``UNKNOWN`` is not a state an upload is ever *in* -- it is what a source
#: tracking no upload answers -- so it is absent here too.
_STATE_RANK: Dict[UploadStatus, int] = {
    UploadStatus.PENDING: 0,
    UploadStatus.READY: 1,
}

#: What ``set_upload_status`` may ask for. ``PENDING`` is where an upload
#: starts and nothing returns it there, so it is not settable either.
SETTABLE_STATES = (UploadStatus.READY, UploadStatus.DISCARDED)


def upload_grid(desc: TensorDescriptor) -> List[int]:
    """The grid an uploaded zarr store is chunked on, planned on and read on.

    A zarr adapter advertises ``default_transfer_chunk_shape`` as its transfer
    grid -- one store block per endpoint was measured as too many endpoints
    (biopb/biopb#684) -- and a write now lands on the grid the planner mints
    (``docs/upload-model.md`` step 4). Minting the store on that same grid is
    what keeps them one thing: on disk, on the wire, for reads and for writes,
    and across a restart, where nothing remembers what the client asked for.
    The request's ``chunk_shape`` is the seed it is grown from, not the layout.

    The alternative -- store the request's grid and coalesce only on the wire --
    puts a planned write across several store blocks, splits ``expected_chunks``
    from what the plan actually sends, and coarsens a sparse set's skip anyway,
    since the skip is per planned chunk.

    Idempotent: the coalescer is a fixed point on its own output, so the
    adapter re-deriving the grid from the store's chunks answers this again.
    """
    return default_transfer_chunk_shape(
        list(desc.shape),
        desc.dtype,
        list(desc.dim_labels) or None,
        native=list(desc.chunk_shape),
    )


def unsettable_state_message(target: Any) -> str:
    """The refusal for a state nothing can be moved to.

    Built here because both the wire boundary (which refuses before reaching an
    adapter) and :meth:`WritableSource.set_status` (which refuses an in-process
    caller) answer the same question, and a caller should not be able to tell
    which one turned it down.
    """
    name = getattr(target, "value", target)
    return (
        f"set_upload_status: {name} is not a state an upload can be moved to; "
        f"use one of {', '.join(state.value for state in SETTABLE_STATES)}."
    )


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


#: What a store's directory name may be at most, extension included. 255
#: **bytes** is the single-component limit ext4, APFS and NTFS share, so a
#: non-ASCII name is measured encoded -- it passes a character count and still
#: overflows the directory entry.
_MAX_STORE_NAME = 255

#: Names MS-DOS gave to devices, which Windows still refuses as files -- with
#: any extension, so ``CON`` mints a ``CON.zarr`` that cannot be opened there.
#: Matched on the stem before the first dot, case-insensitively, as Windows
#: does.
_WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{d}" for d in range(1, 10)]
    + [f"LPT{d}" for d in range(1, 10)]
)

#: Refused on NTFS. Separators, ``:`` and NUL are refused separately, with
#: their own messages.
_WINDOWS_FORBIDDEN = '<>"|?*'

#: Field names an upload may not take. ``labels`` is both the NGFF group name
#: and the wire segment that addresses a set, so a field called ``labels``
#: makes ``<array_id>/labels/<name>`` ambiguous by construction (parsed by
#: ``core.labels.split_label_field``). Reserved rather than marked with an
#: ``@``: marking would move every stored ``array_id``, ``rois.array_id``
#: included, and that is user data. See ``docs/upload-model.md``.
#:
#: A set *named* ``labels`` stays legal -- the parse is right-to-left, so
#: ``<field>/labels/labels`` is unambiguous.
RESERVED_FIELD_NAMES = frozenset({"labels"})


def fold_name(name: str) -> str:
    """The form two names are compared in: NFC, case-folded.

    Names that differ only by case or by Unicode normalization are **one**
    directory on NTFS, APFS and HFS+ and two on ext4 -- and HFS+ stores NFD, so
    a name sent composed comes back decomposed. Comparing folded is what keeps
    one name one identity wherever the same ``write_dir`` is served from.

    Only the comparison folds; the name is stored and displayed as the caller
    wrote it, so ``Nuclei`` stays ``Nuclei``.
    """
    return unicodedata.normalize("NFC", name).casefold()


def folded_match(name: str, candidates: Iterable[str]) -> Optional[str]:
    """The candidate *name* collides with under :func:`fold_name`, or None.

    The one application of the folding rule, so a fifth store kind cannot copy
    whichever of four hand-rolled scans it happened to find. Returns the
    colliding spelling rather than a bool: the refusal is far more useful when
    it can say what the name is already taken *as*.
    """
    folded = fold_name(name)
    return next((c for c in candidates if fold_name(c) == folded), None)


def taken_store_name(
    directory: Path, name: str, suffix: str = ".zarr"
) -> Optional[Path]:
    """The store in *directory* that *name* would collide with, or None.

    The on-disk half of :func:`folded_match`. An exclusive ``mkdir`` catches an
    exact repeat by itself, and on a case-insensitive filesystem it catches the
    folded ones too -- but not on ext4, where the store would then be one
    directory of two on the next host to serve this ``write_dir``. So the fold
    is checked here rather than left to the filesystem.
    """
    if not directory.is_dir():
        return None
    stems = {p.name[: -len(suffix)]: p for p in directory.glob(f"*{suffix}")}
    hit = folded_match(name, stems)
    return stems[hit] if hit is not None else None


def unsafe_store_name(name: str, suffix: str = ".zarr") -> Optional[str]:
    """Why *name* cannot be a store's directory name, or None if it can.

    A format that puts its bytes on disk names the directory after what the
    client asked for -- a registered source becomes
    ``<write_dir>/sources/<name>.zarr``, its member becomes ``<that>/<field>``,
    and a label set becomes ``<write_dir>/labels/<source_id>/<name>.zarr`` --
    so the name is untrusted input that turns into a path component, of a
    directory the server later creates and, on discard, deletes whole. It must
    therefore name one component *inside* the directory the server chose:
    ``../../x`` otherwise mints and later removes a store two levels above
    ``write_dir``.

    Refused rather than sanitized, because the name is an identity as well as
    a path: it is what ``add_tensor`` refuses a collision on, so folding two
    different requests onto one store would trade a traversal for a mix-up.

    **Every rule here is checked on every platform, not the running one.** A
    store minted on Linux has to open, and keep its identity, when the same
    ``write_dir`` is later served from Windows or macOS -- so this refuses what
    any of the three would refuse, and :func:`fold_name` handles the two rules
    that are about collision rather than legality.

    Every field reaches the filesystem, whatever format it is stored in
    (:func:`unsafe_field_name`): a ``cache://`` member is a directory of
    segments and a ``zarr://`` one an image group, so a name legal only off
    disk would be a tensor no restart could reopen.
    """
    if not name:
        return "is empty"
    # Bytes, not characters: _MAX_STORE_NAME is a directory-entry limit.
    if len(name.encode("utf-8")) + len(suffix) > _MAX_STORE_NAME:
        return f"is longer than {_MAX_STORE_NAME - len(suffix)} bytes"
    if "/" in name or "\\" in name:
        return "contains a path separator"
    if ":" in name:
        return "contains ':', which names a drive or a stream on Windows"
    if "\x00" in name:
        return "contains a NUL"
    if name in (".", ".."):
        return "is a relative path component"
    if name.startswith("."):
        return "starts with '.', which hides it and collides with zarr's own metadata"
    # Windows strips a trailing space or dot silently, so "x." and "x" would be
    # one directory there and two identities here -- the mix-up this refuses
    # rather than sanitizes.
    if name[-1] in " .":
        return "ends with a space or '.', which Windows strips"
    if any(ch in _WINDOWS_FORBIDDEN for ch in name):
        return f"contains one of {_WINDOWS_FORBIDDEN}, refused on NTFS"
    if any(ord(ch) < 32 for ch in name):
        return "contains a control character"
    if name.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES:
        return (
            f"is {name.split('.')[0].upper()}, a Windows device name, which is "
            f"reserved there with any extension"
        )
    return None


def unsafe_field_name(name: str) -> Optional[str]:
    """Why *name* cannot be an uploaded tensor's field, or None if it can.

    A field is a path component of its source's group, so it takes the store
    rules with no extension of its own, plus the reserved words. Applied by
    ``adapters.registered.create_member`` to every field, whatever format it
    asked for.
    """
    if fold_name(name) in RESERVED_FIELD_NAMES:
        return (
            "is reserved: a label set is addressed as "
            "'<array_id>/labels/<name>', so a field of that name would be "
            "unaddressable"
        )
    return unsafe_store_name(name, suffix="")


def _expected_chunk_count(shape: Sequence[int], chunk_shape: Sequence[int]) -> int:
    count = 1
    for dim, chunk in zip(shape, chunk_shape, strict=True):
        count *= ceil(dim / chunk)
    return count


@dataclass
class UploadProgress:
    """How far an upload has got, and whether anyone still wants it.

    The state moves only when the producer says so (``set_status``), never on
    the chunk count: the count is a coverage *proxy*, and a cache-backed source
    accepts arbitrary bounds, so off-grid writes can reach the expected count
    without tiling the array or never reach it at all. ``READY`` and
    ``DISCARDED`` seal the source against further writes; ``DISCARDED`` keeps
    the adapter registered as a tombstone so a writer still unwinding learns
    *why* its write failed rather than that its source never existed.

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
        """No further chunk will be accepted -- published, or given up on."""
        return self.status in (UploadStatus.READY, UploadStatus.DISCARDED)

    @property
    def is_readable(self) -> bool:
        """Published: DoGet is served, and an unwritten chunk reads as zeros.

        False for ``PENDING`` -- where a hole is indistinguishable from a chunk
        still in flight -- and for ``DISCARDED``, which answers a reader the
        same reason it answers a writer. The complement of :attr:`is_sealed`
        only on a live upload: both are true at READY, which is the point.
        """
        return self.status is UploadStatus.READY

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

    def _publish_store(self) -> None:  # noqa: B027 - concrete no-op default
        """Mark the store on disk as one a restart should keep.

        Called on the **READY** transition, before it is announced and outside
        ``progress.lock``. A durable store carries a pending marker from
        creation so a crash leaves something a restart can recognize and
        remove; this is where the marker is flipped. READY is what publishes
        the store, and a set listed under its image must not be one the next
        boot sweeps away, so the marker flips here and not at create.
        Raises ``OSError`` if the marker cannot be written, in which case the
        upload stays PENDING and the transition is retried.
        """

    def upload_response(self, desc: TensorDescriptor) -> TensorDescriptor:
        """The descriptor ``add_tensor`` answers with.

        Echoes the request under the tensor's own ``array_id`` -- which every
        later write, poll and transition names.

        The **grid is the store's, not the request's**. A request's
        ``chunk_shape`` is the seed the layout is grown from and a zarr store
        coalesces it (``upload_grid``), so echoing what was asked for would
        hand back a grid no plan mints and every ``upload_chunk`` on it would
        be refused. This is the same number ``get_flight_info`` advertises.

        Physical calibration is deliberately not echoed here: a format adds it
        only if a later read will reproduce it verbatim (issue #272).
        """
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=desc.dim_labels,
            shape=desc.shape,
            chunk_shape=list(self.get_transfer_chunk_size()),
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
        source_id = self.array_id
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
        logger.info(f"Discarded upload {self.array_id}: {reason or 'no reason given'}")
        return True

    def reap_step(self, now: float, ttl: float) -> Tuple[bool, Optional[float]]:
        """The reclaim sweep's two questions for this upload, one lock hold.

        Returns ``(expired, tombstone_age)``:

        - *expired*: a PENDING upload with no progress for *ttl* seconds was
          just discarded here, with a reason -- one terminal transition, not
          a second path into oblivion. The check and the transition share the
          lock hold so a ``set_status`` racing the sweep either lands first
          (and the upload keeps the state it reached) or is refused as
          discarded; it can never be undone. A durable kind's store goes with it (:meth:`_dispose_store`).
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

    def set_status(self, target: UploadStatus, reason: str = "") -> Dict[str, Any]:
        """Move this upload to *target*, or stand still; the only mover there is.

        The ladder is PENDING -> READY, and a call may only climb it: setting
        the state the upload is already in is a no-op (so a retried transition
        is not an error), and setting one below it raises
        ``UploadTransitionError`` -- a published result cannot be unpublished,
        nor reopened for writes a reader has already been told are over.

        ``DISCARDED`` is not on the ladder and is reachable from every rung; it
        routes to :meth:`discard`, which is total and idempotent.

        The store is published on disk (:meth:`_publish_store`) **before**
        READY is announced, never after: a reader that has seen READY must find
        the store kept after a crash, not pending and swept away at the next
        boot. That runs outside ``progress.lock`` (it takes the kind's write
        lock, which orders ahead of this one), so the transition re-checks for
        a discard that landed in between -- publishing a store that discard has
        already removed is what that path reports.
        """
        if target is UploadStatus.DISCARDED:
            return self.discard(reason)
        if target not in SETTABLE_STATES:
            raise UploadTransitionError(unsettable_state_message(target))
        progress = self._upload
        if progress is None:
            return unknown_upload_status(self.array_id)

        with progress.lock:
            self._raise_if_discarded_locked(progress)
            current = progress.status
            if current is target:
                return progress.as_status_dict(self.array_id)
            # Unreachable while the ladder has one climbable rung: the only
            # settable non-DISCARDED target is READY, and a source already
            # there returned above. Kept as the ladder's own guard -- it is
            # what a second rung (a reopen, ``docs/upload-model.md``) would
            # need, and it is cheaper to leave than to rediscover.
            if _STATE_RANK[current] > _STATE_RANK[target]:
                raise UploadTransitionError(
                    f"set_upload_status: {self.array_id} is {current.value} and "
                    f"cannot go back to {target.value}."
                )

        # Unconditional: the only settable non-DISCARDED target is READY, and
        # an upload already there returned above, so *current* is PENDING here.
        try:
            self._publish_store()
        except OSError:
            with progress.lock:
                self._raise_if_discarded_locked(progress)
            raise

        with progress.lock:
            self._raise_if_discarded_locked(progress)
            # Re-checked after the unlocked publish above, not a ladder test:
            # what this catches is a status that moved while the lock was down.
            if progress.status is not target:
                progress.status = target
                progress.touch()
                logger.info(
                    f"Upload {self.array_id} -> {target.value}: "
                    f"{progress.uploaded_chunks}/{progress.expected_chunks} chunks"
                )
            return progress.as_status_dict(self.array_id)

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
            return unknown_upload_status(self.array_id)
        with progress.lock:
            return progress.as_status_dict(self.array_id)

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

        Caller holds ``progress.lock``. Discarded is checked before sealed in
        :meth:`_refuse_write`: it carries a reason, which "already finished"
        does not.
        """
        if progress.is_discarded:
            raise UploadDiscardedError(self.array_id, progress.reason)

    def _refuse_write(self) -> None:
        """The two ways a write can arrive too late.

        There is no third: an ``array_id`` has one adapter for as long as it is
        served (``UploadManager.add_tensor`` refuses a field collision), so a
        write cannot land in a stranger's tensor by naming its own. The one
        exception is by design: a discarded field is reclaimed after
        ``upload_ttl`` seconds, and a straggler quiet for that long writes into
        whoever took the field next.
        """
        progress = self._upload
        if progress is None:
            return
        with progress.lock:
            self._raise_if_discarded_locked(progress)
            if progress.is_sealed:
                raise UploadSealedError(self.array_id)

    def check_readable(self) -> None:
        """The two ways a read can arrive at an upload that cannot answer it.

        Overrides the read base's no-op. Here rather than at the Flight
        boundary because it is the adapter that knows whether it is an upload
        at all: the same classes serve a discovered zarr store, where
        ``_upload`` is None and this costs one attribute read.

        ``DISCARDED`` -- a tombstone answers a still-unwinding reader the same
        reason it gives a writer, rather than "no chunk here"; the store is
        gone, and zarr would otherwise answer fill values for it while the
        cache answered the bytes it still holds. The exception is raised for a
        *writer* (the DoPut boundary turns it into a cancellation, meaning stop
        sending), so a reader hears it as the read path's terminal error
        instead. ``PENDING`` -- nobody has said the holes are meant to be holes
        yet, so a chunk that has not arrived cannot honestly be served as zeros
        and must not be served as an error the caller reads as data loss
        either.

        Public, and a sibling of ``check_chunk_version`` rather than a private
        step of :meth:`resolve_chunk_data`, because the two reads that skip
        that method have to ask as well: the localhost locate path answers a
        warm chunk straight out of the cache (``server._handle_chunk_locate``),
        and ``CachedSourceAdapter`` serves an uploaded chunk from the entry
        that *is* the chunk. The write path's equivalent is
        :meth:`_refuse_write`.
        """
        progress = self._upload
        if progress is None:
            return
        with progress.lock:
            if progress.is_discarded:
                raise UploadDiscardedReadError(self.array_id, progress.reason)
            if not progress.is_readable:
                raise UploadNotPublishedError(self.array_id)

    def _mark_chunk(self, bounds: ChunkBounds) -> None:
        """Record that the chunk at *bounds* landed.

        Counting only: reaching ``expected_chunks`` no longer flips anything,
        because :meth:`set_status` is the only route off PENDING. What the
        count is still for is the progress a poller watches.

        A seal between :meth:`_refuse_write` and here leaves the chunk stored
        but uncounted: a sealed upload must not be walked back into PENDING by
        a write nobody is waiting for, nor have its clock touched.
        """
        progress = self._upload
        if progress is None:
            return
        chunk_id = encode_chunk_id(self.array_id, bounds)
        with progress.lock:
            if progress.is_sealed:
                return
            progress.uploaded_chunk_ids.add(chunk_id)
            progress.touch()
