"""Crash recovery utilities for file-based cache.

Provides:
- WriteAheadLog: Detect incomplete writes after crash
- ProcessLock: Single-owner lock on the cache dir + unclean-exit detection
- SegmentEntryInfo: Metadata for entries stored in segments
- SegmentInfo: Metadata for segment files
- RecoveryStatus: Result of crash recovery
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from biopb._lifecycle.file_lock import ExclusiveFileLock


@dataclass
class SegmentEntryInfo:
    """Metadata for an entry stored in a segment."""

    segment_id: int
    offset: int  # Entry index within segment (used by the sequential reader)
    size_bytes: int
    created_at: float = 0.0
    last_access_time: float = 0.0  # Updated on each read
    # Byte location of the entry's encapsulated Arrow IPC message within the
    # segment file. Lets a localhost client mmap the segment and read just this
    # message (issue #9). Defaults of 0 mean "unknown" (e.g. legacy index);
    # locate_entry() returns unavailable in that case so the client uses do_get.
    byte_offset: int = 0
    byte_length: int = 0


@dataclass
class SegmentInfo:
    """Metadata for a segment file."""

    size_bytes: int
    created_at: float
    last_access_time: float  # Updated on each read from segment
    entry_count: int


@dataclass
class SieveKSegmentInfo:
    """Sieve-K metadata for a segment.

    Attributes:
        segment_id: Unique segment identifier
        size_bytes: Total size of segment file
        created_at: Creation timestamp
        last_access_time: Last access timestamp
        entry_count: Number of entries in this segment
        frequency: Saturating counter (0 to K=2) for Sieve-K algorithm
        mmap_released: True if mmap handle was released for cold segment
    """

    segment_id: int
    size_bytes: int
    created_at: float
    last_access_time: float
    entry_count: int
    frequency: int = 0  # Saturating counter (0 to K=2)
    mmap_released: bool = False  # True if mmap handle was released for cold segment


@dataclass
class PoolQueueInfo:
    """Per-pool Sieve-K queue state.

    Uses same pattern as reference implementation:
    - queue: ordered segment_ids (newest at left/head, oldest at right/tail)
    - segments: dict for metadata lookup (segment_id -> SieveKSegmentInfo)

    Attributes:
        pool_key: Tuple of (schema_key, size_class)
        hand: Current hand offset from tail (0 = tail position)
        queue: deque of segment_ids ordered (newest at left)
        segments: dict mapping segment_id to SieveKSegmentInfo
        hits: Number of cache hits in this pool
        misses: Number of cache misses in this pool
    """

    pool_key: Tuple[str, str]  # (schema_key, size_class)
    hand: int = 0  # Current hand offset from tail (0 = tail position)
    queue: deque = field(default_factory=deque)  # segment_ids ordered (newest at left)
    segments: Dict[int, SieveKSegmentInfo] = field(
        default_factory=dict
    )  # segment_id -> info
    hits: int = 0
    misses: int = 0


@dataclass
class RecoveryStatus:
    """Result of crash recovery."""

    recovered_entries: int
    lost_entries: int
    recovered_bytes: int
    lost_bytes: int
    errors: List[str] = field(default_factory=list)


class WriteAheadLog:
    """Simple WAL for detecting incomplete writes.

    Tracks pending writes before they are committed to segment files.
    On recovery, entries in WAL but missing from segments are considered lost.
    """

    def __init__(self, path: Path):
        self._path = path
        self._pending: Dict[str, float] = {}  # key_hex -> timestamp
        self._load()

    def _load(self) -> None:
        """Load existing WAL state from disk."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._pending = data.get("pending", {})
            except (OSError, json.JSONDecodeError):
                # Corrupted WAL - start fresh
                self._pending = {}

    def _save(self) -> None:
        """Save WAL state to disk."""
        data = {"pending": self._pending}
        with open(self._path, "w") as f:
            json.dump(data, f)

    def log_pending(self, key: bytes) -> None:
        """Log key as pending write."""
        key_hex = key.hex()
        self._pending[key_hex] = time.time()
        self._save()

    def log_committed(self, key: bytes) -> None:
        """Mark write as committed."""
        key_hex = key.hex()
        self._pending.pop(key_hex, None)
        self._save()

    def get_pending_keys(self) -> List[bytes]:
        """Get all keys with pending writes."""
        return [bytes.fromhex(k) for k in self._pending]

    def clear(self) -> None:
        """Clear WAL after clean shutdown."""
        self._pending.clear()
        if self._path.exists():
            self._path.unlink()

    def has_pending(self) -> bool:
        """Check if there are pending writes."""
        return len(self._pending) > 0


class ProcessLock:
    """Single-owner lock on a cache directory, plus an unclean-exit signal.

    Two jobs, and they are now carried by two different files:

    * **Exclusion** is an OS advisory lock on ``<path>`` held for the life of the
      process (``biopb._lifecycle.file_lock.ExclusiveFileLock``). It lives on an
      open descriptor, so the OS releases it when the owner exits *however* it
      exits.
    * **Crash detection** is a sibling ``<path>.owner`` record, written after the
      lock is taken and removed on a clean release. Finding one while acquiring
      means the previous owner never released -- it crashed -- which is what
      drives WAL recovery.

    Splitting them is what removes the pid bookkeeping this class used to carry
    (biopb/biopb#544). Exclusion by a *record* cannot be atomic: reading a pid
    file, judging its owner dead, and writing your own is a check-then-act, so
    several starters racing on one cache dir could all conclude they owned it --
    and judging "dead" needed liveness plus a create-time token to survive pid
    reuse. A descriptor-held lock has no window to race and no owner to
    identify: acquire either wins or loses. The ``.owner`` record is left purely
    informational -- who held it, since when -- and is never consulted to decide
    ownership, so a torn or stale one can mislead nobody.

    Not thread-safe; one instance per cache directory.
    """

    def __init__(self, path: Path):
        self._path = path
        self._owner_path = path.with_name(path.name + ".owner")
        self._lock = ExclusiveFileLock(path)
        self._prior_owner: Optional[dict] = None

    def acquire(self) -> bool:
        """Take exclusive ownership of the cache directory.

        Returns True on success (or if this instance already holds it), False if
        another live process owns it. Records the previous owner's leftover
        marker, if any, for :meth:`is_stale` -- so unlike the old lock-file
        scheme there is nothing to capture *before* acquiring.
        """
        if self._lock.is_held():
            return True
        if not self._lock.acquire():
            return False

        # We are now the sole owner, so the previous owner's record (if it got
        # one) can only be leftovers: a clean release removes it.
        self._prior_owner = self._read_owner_record()
        self._write_owner_record()
        return True

    def release(self) -> None:
        """Release ownership. Idempotent.

        Removes the owner record *before* dropping the lock, so the next owner
        can never see a released lock alongside a record that outlived it and
        mistake a clean shutdown for a crash.
        """
        if not self._lock.is_held():
            return
        try:
            self._owner_path.unlink(missing_ok=True)
        except OSError:
            pass  # A record we can't remove costs a spurious recovery, no more.
        self._lock.release()

    def is_stale(self) -> bool:
        """Whether the previous owner exited without releasing (i.e. crashed).

        Meaningful once :meth:`acquire` has succeeded; False before that. A
        corrupt/unparseable record still counts as a crash -- it exists, which
        is the whole signal, and its contents are only ever diagnostic.
        """
        return self._prior_owner is not None

    def prior_owner(self) -> Optional[dict]:
        """The crashed owner's record (``pid`` / ``acquired_at``), for logging."""
        return self._prior_owner

    def is_acquired(self) -> bool:
        """Check if this instance holds the lock."""
        return self._lock.is_held()

    def _read_owner_record(self) -> Optional[dict]:
        """The leftover owner record, or None if there is none.

        Returns a dict for *any* file that is present -- an empty one when it
        can't be parsed -- because presence, not content, is the signal.
        """
        if not self._owner_path.exists():
            return None
        try:
            with open(self._owner_path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_owner_record(self) -> None:
        """Stamp our ownership. Best effort: the lock is what confers ownership,
        so failing to write the record must not fail the acquire -- it only
        costs the next owner its crash signal."""
        try:
            with open(self._owner_path, "w") as f:
                json.dump({"pid": os.getpid(), "acquired_at": time.time()}, f)
        except OSError:
            pass
