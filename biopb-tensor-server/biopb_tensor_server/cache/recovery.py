"""Crash recovery utilities for file-based cache.

Provides:
- WriteAheadLog: Detect incomplete writes after crash
- ProcessLock: File-based lock for crash detection
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


@dataclass
class SegmentEntryInfo:
    """Metadata for an entry stored in a segment."""
    segment_id: int
    offset: int  # Entry index within segment (used by the sequential reader)
    size_bytes: int
    metadata: dict = field(default_factory=dict)
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
    segments: Dict[int, SieveKSegmentInfo] = field(default_factory=dict)  # segment_id -> info
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
                with open(self._path, 'r') as f:
                    data = json.load(f)
                self._pending = data.get('pending', {})
            except (json.JSONDecodeError, IOError):
                # Corrupted WAL - start fresh
                self._pending = {}

    def _save(self) -> None:
        """Save WAL state to disk."""
        data = {'pending': self._pending}
        with open(self._path, 'w') as f:
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
        return [bytes.fromhex(k) for k in self._pending.keys()]

    def clear(self) -> None:
        """Clear WAL after clean shutdown."""
        self._pending.clear()
        if self._path.exists():
            self._path.unlink()

    def has_pending(self) -> bool:
        """Check if there are pending writes."""
        return len(self._pending) > 0


class ProcessLock:
    """File-based lock for crash detection.

    Creates a lock file containing the process ID. On startup, if a lock
    file exists with a PID that is no longer running, it indicates a crash.
    """

    def __init__(self, path: Path):
        self._path = path
        self._pid: Optional[int] = None
        self._acquired = False

    def acquire(self) -> bool:
        """Acquire lock, detect stale lock from crashed process.

        Returns True if lock acquired (new or stale lock removed).
        Returns False if lock held by another running process.
        """
        if self._acquired:
            return True

        # Check for existing lock
        if self._path.exists():
            try:
                with open(self._path, 'r') as f:
                    data = json.load(f)
                existing_pid = data.get('pid')

                if existing_pid and self._is_process_running(existing_pid):
                    # Lock held by another running process
                    return False

                # Stale lock - process is dead, remove it
                self._path.unlink()
            except (json.JSONDecodeError, IOError):
                # Corrupted lock file - remove it
                self._path.unlink()

        # Create new lock
        self._pid = os.getpid()
        self._acquired = True
        data = {
            'pid': self._pid,
            'acquired_at': time.time(),
        }
        with open(self._path, 'w') as f:
            json.dump(data, f)

        return True

    def release(self) -> None:
        """Release lock."""
        if self._acquired and self._path.exists():
            self._path.unlink()
        self._acquired = False
        self._pid = None

    def is_stale(self) -> bool:
        """Check if existing lock is from dead process.

        Returns True if lock file exists but process is dead.
        Returns False if no lock file or process is running.
        """
        if not self._path.exists():
            return False

        try:
            with open(self._path, 'r') as f:
                data = json.load(f)
            existing_pid = data.get('pid')
            if existing_pid and not self._is_process_running(existing_pid):
                return True
        except (json.JSONDecodeError, IOError):
            return True  # Corrupted lock is stale

        return False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            return True
        except OSError:
            return False

    def is_acquired(self) -> bool:
        """Check if this instance holds the lock."""
        return self._acquired