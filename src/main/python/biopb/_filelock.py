"""Cross-process file lock — shared, stdlib-only, cross-platform.

An OS-level advisory *exclusive* lock keyed on a lock file, used to serialize an
otherwise-racy check-then-act across independent processes. The motivating case
is ``biopb control start``: the launcher, the installer, and — once the shim
starts the control plane on demand — several racing agent sessions can all invoke
it at once, and without serialization two starters can both see "no pidfile",
both spawn a control, and the bind-loser's parent overwrite or remove the live
winner's pidfile (orphaning a control that ``control stop`` can no longer reach).

The lock is held on an open file descriptor, so the OS releases it automatically
when the fd is closed *or the holder process dies* — there is no stale lock file
to reap (unlike an ``O_CREAT | O_EXCL`` sentinel, which a crashed holder would
leave behind). The lock file itself is created if absent and never removed; its
mere existence conveys nothing.

POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``. Both are polled in
non-blocking mode so the wait honours a timeout — ``flock``'s blocking mode has
no portable timeout, and this keeps one code path for both platforms.

Kept dependency-free and in the core ``biopb`` SDK (next to ``_proc`` /
``_config_sessions``) so both the CLI and biopb-mcp's shim can import it without
dragging in a heavy stack or importing each other.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Iterator


class LockTimeout(TimeoutError):
    """Raised when :func:`file_lock` cannot acquire the lock within its timeout.

    Subclasses ``TimeoutError`` so a caller may catch either, but is distinct so a
    lock-acquisition timeout is never confused with an unrelated ``TimeoutError``
    (e.g. a socket timeout) raised from inside the guarded block.
    """


if sys.platform == "win32":
    import msvcrt

    def _try_acquire(fd: int) -> bool:
        # Lock one byte at offset 0. Windows byte-range locks are mandatory and
        # per-handle, so a second handle (even in the same process) is excluded.
        # LK_NBLCK returns immediately, raising OSError when the byte is held.
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _release(fd: int) -> None:
        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_acquire(fd: int) -> bool:
        # flock locks are per open-file-description, so a second os.open of the
        # same path (another process, or the same one) is denied while held.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _release(fd: int) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(path: Path, timeout: float = 30.0, poll: float = 0.1) -> Iterator[None]:
    """Hold an exclusive cross-process lock on ``path`` for the block's duration.

    Blocks until the lock is acquired or ``timeout`` seconds elapse, in which case
    :class:`LockTimeout` is raised (the caller decides whether that is fatal). The
    parent directory and the lock file are created if absent; the file is never
    removed — the lock lives on the open descriptor and the OS drops it when the
    fd is closed or the process dies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        deadline = time.monotonic() + timeout
        while not _try_acquire(fd):
            if time.monotonic() >= deadline:
                raise LockTimeout(f"could not acquire {path} within {timeout}s")
            time.sleep(poll)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)
