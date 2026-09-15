"""Bringing a cache directory up: its layout, ownership, format version, recovery.

Everything here runs once, from ``ArrowFileBackend.__init__``, before the
backend is reachable by any other thread. So none of it takes the backend's
locks or touches its index, and it can be read without the lock discipline that
governs the rest of ``file_backend.py``. That is the seam: the counterpart
shutdown path cannot move out here, because it tears down the very index and
writer thread the serving code is built around.

What does NOT live here is the index rebuild. It reads the same directory, but
it *constructs* the backend's index, so it stays with the code that owns that
index (``ArrowFileBackend._rebuild_index_from_segments``); the file-format half
it calls into is ``segment_index.py``.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from biopb_tensor_server.cache.recovery import ProcessLock, RecoveryStatus

__all__ = [
    "CACHE_FILE_FORMAT_VERSION",
    "FORMAT_VERSION_MARKER",
    "CacheLayout",
    "acquire_cache_dir",
    "enforce_format_version",
    "recovery_status",
]

logger = logging.getLogger(__name__)


# On-disk segment format version for the localhost cache-file handoff (issue
# #9). The server reports it in chunk_locate and a client declines the fast
# path (falls back to do_get) for any version it doesn't understand.
#
# v2: biopb/biopb#596 implemented axis order normalization. A v1 segment holds the
# pre-transpose bytes, so reusing it would serve axes in the wrong order.
CACHE_FILE_FORMAT_VERSION = 2

# Name of the on-disk marker file (in the cache root, beside ``lock`` and
# ``lock``) recording the CACHE_FILE_FORMAT_VERSION the segments were written
# under. ``enforce_format_version`` reads it at init and wipes the cache when it
# is missing or mismatched, so a segment-layout / key-composition change can't
# silently reuse incompatible on-disk segments.
FORMAT_VERSION_MARKER = "format_version"


@dataclass(frozen=True)
class CacheLayout:
    """Where everything lives under a cache directory.

    One definition of the on-disk layout, shared by the boot path, the serving
    backend and the eviction sweep -- so "which file is segment 7?" cannot be
    answered two ways.
    """

    cache_dir: Path

    @property
    def segments_dir(self) -> Path:
        """Directory holding the segment bodies and their sidecar indexes."""
        return self.cache_dir / "segments"

    @property
    def legacy_wal_path(self) -> Path:
        """A write-ahead log left by a build before it was removed.

        Nothing reads it. Boot unlinks it so an upgraded cache dir does not keep
        a file that no longer means anything.
        """
        return self.cache_dir / "wal.json"

    @property
    def lock_path(self) -> Path:
        return self.cache_dir / "lock"

    @property
    def marker_path(self) -> Path:
        return self.cache_dir / FORMAT_VERSION_MARKER

    def segment_path(self, segment_id: int) -> Path:
        """Path of a segment's body file (``seg_NNNN.arrow``)."""
        return self.segments_dir / f"seg_{segment_id:04d}.arrow"

    def sidecar_path(self, segment_id: int) -> Path:
        """Path of a segment's sidecar index file (``seg_NNNN.idx``)."""
        return self.segments_dir / f"seg_{segment_id:04d}.idx"


def acquire_cache_dir(layout: CacheLayout) -> Tuple[ProcessLock, bool]:
    """Create the cache dir and take exclusive ownership of it.

    Returns the held lock and whether the previous owner left it uncleanly (the
    caller's cue to run recovery). Raises if another live process holds it.

    The cache root is restricted to the owner (0o700): segment files hold
    decoded chunk payloads and their paths are handed to localhost clients
    (issue #9), so other users on a shared host must not be able to read them.
    (mode is masked by umask; re-assert with chmod. No-op hardening on Windows,
    which ignores POSIX modes.)

    The lock is held on an open descriptor for the life of this process, so it
    is the acquire itself that answers "is someone else using this?" --
    staleness is read after, from the leftover owner record (biopb/biopb#544).
    """
    layout.segments_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(layout.cache_dir, 0o700)
    except OSError:
        pass

    # Drop a write-ahead log left by an older build; nothing reads it now.
    try:
        layout.legacy_wal_path.unlink(missing_ok=True)
    except OSError:
        pass

    process_lock = ProcessLock(layout.lock_path)
    if not process_lock.acquire():
        raise RuntimeError(
            f"Cannot acquire cache lock at {layout.lock_path}. "
            "Another process is using the cache."
        )

    was_stale = process_lock.is_stale()
    if was_stale:
        prior = process_lock.prior_owner() or {}
        logger.warning(
            "Cache directory was not released cleanly (previous owner pid=%s); "
            "running recovery",
            prior.get("pid", "unknown"),
        )
    return process_lock, was_stale


def enforce_format_version(layout: CacheLayout) -> bool:
    """Wipe the on-disk cache when its segment format version != code.

    ``CACHE_FILE_FORMAT_VERSION`` is the segment message layout / cache-key
    encoding contract (see the constant). A marker file records the version the
    on-disk segments were written under, and a missing or mismatched marker
    drops them before the index rebuild. A cache dir with segments but no marker
    counts as a mismatch.

    Must run while the process lock is held and before recovery / index rebuild
    read the segments. Returns True iff a (re)stamp happened (marker missing or
    mismatched), in which case the segments were just wiped and there is nothing
    to recover.
    """
    try:
        on_disk: Optional[int] = int(layout.marker_path.read_text().strip())
    except (OSError, ValueError):
        # Missing marker (pre-enforcement / fresh dir) or an unparseable one
        # (torn write) -> treat as a mismatch and re-stamp.
        on_disk = None

    if on_disk == CACHE_FILE_FORMAT_VERSION:
        return False

    _wipe_cache_contents(layout, stale_version=on_disk)

    # Stamp the current version. Write to a temp file and atomically replace
    # so a crash mid-write cannot leave a torn marker that spuriously wipes a
    # good cache on the next boot.
    tmp_path = layout.marker_path.with_suffix(".tmp")
    tmp_path.write_text(f"{CACHE_FILE_FORMAT_VERSION}\n")
    os.replace(tmp_path, layout.marker_path)
    return True


def _wipe_cache_contents(layout: CacheLayout, stale_version: Optional[int]) -> None:
    """Delete the version-sensitive on-disk state (the segments), leaving
    the held process lock and marker in place, and recreate an empty
    ``segments`` dir. Logs loudly only when data was actually discarded (a
    fresh dir wipe is a silent no-op). Runs before any mmap is opened, so no
    segment is mapped and the removal is safe on Windows too (issue #5).

    Fail-closed on a partial wipe: ``shutil.rmtree(ignore_errors=True)`` can
    leave files behind without surfacing an error -- an NFS unlink that
    returns ESTALE/EIO, a permission glitch, a handle another process still
    holds.
    """
    segments_dir = layout.segments_dir
    had_data = segments_dir.exists() and any(segments_dir.iterdir())
    if had_data:
        logger.warning(
            "Cache format-version mismatch (on-disk=%s, code=%d): discarding "
            "the incompatible on-disk cache at %s before rebuild.",
            "unversioned" if stale_version is None else stale_version,
            CACHE_FILE_FORMAT_VERSION,
            layout.cache_dir,
        )

    shutil.rmtree(segments_dir, ignore_errors=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    leftover = sorted(segments_dir.glob("seg_*.arrow"))
    if leftover:
        # The caller releases the process lock on the way out, so an aborted
        # init doesn't wedge a retry or the next start.
        raise RuntimeError(
            f"Failed to clear the incompatible cache at {segments_dir}: "
            f"{len(leftover)} segment file(s) survived the wipe "
            f"(e.g. {leftover[0].name}). Refusing to start rather than serve "
            "them; remove the cache directory manually and retry."
        )


def recovery_status(
    layout: CacheLayout, recovered_entries: int, lost_entries: int
) -> RecoveryStatus:
    """Summarize what an unclean restart found, once the index is rebuilt.

    Both counts come from the rebuild rather than from a log kept during normal
    operation: ``recovered_entries`` is what the index actually holds, and
    ``lost_entries`` is the number of segments whose tail was torn -- a write
    interrupted mid-``write_batch``, which is the only way a synchronous write
    can fail to survive.

    That is strictly better than the write-ahead log this replaced
    (biopb/biopb#815 era), which cost ~167 us on *every* cached chunk -- 47% of
    the write -- to report a number that could only ever be 0 or 1, and got it
    wrong in the window between ``sink.flush()`` and its own commit record,
    where a successful write was still counted as lost.

    Byte total is the segment files' on-disk footprint (a ``stat``, no read).
    """
    recovered_bytes = 0
    for seg_file in layout.segments_dir.glob("seg_*.arrow"):
        try:
            recovered_bytes += seg_file.stat().st_size
        except OSError:
            pass

    if lost_entries:
        logger.warning(
            "Cache recovery: %d segment(s) ended in a partial write; those "
            "entries were dropped and will be recomputed on demand.",
            lost_entries,
        )

    return RecoveryStatus(
        recovered_entries=recovered_entries,
        lost_entries=lost_entries,
        recovered_bytes=recovered_bytes,
        lost_bytes=0,
        errors=[],
    )
