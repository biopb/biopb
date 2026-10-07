"""A child's output, written to a log file that rotates while it grows.

The supervisor gives a child a pipe rather than the file, so the control is the
only writer and can rotate between writes. That bounds everything the child
emits -- ``warnings``, native-library output, tracebacks -- not just its log
records, and the file stays one the control opened itself.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import BinaryIO, Optional

from biopb._config import locations as _locations

logger = logging.getLogger(__name__)


class RotatingLog:
    """An append-only binary log that rotates (``.1`` ... ``.N``) past
    ``max_bytes``. Thread-safe: a child's pump and the supervisor both write.

    Opening rotates a file that is already over the limit. Output is
    never refused: a failed write or rotation drops data rather than raising,
    because the one thing worse than a lost line is a child blocked on its pipe.
    """

    def __init__(
        self,
        path: Path,
        max_bytes: int = _locations.LOG_MAX_BYTES,
        backup_count: int = _locations.LOG_BACKUP_COUNT,
    ) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()
        self._fh: Optional[BinaryIO] = None
        self._size = 0
        self._rotate_at = max_bytes
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _locations.rotate_log(self._path, max_bytes, backup_count)
        self._open()

    def _open(self) -> None:
        self._fh = open(self._path, "ab", buffering=0)  # noqa: SIM115 - held until close()
        self._size = self._fh.tell()

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._fh is None:
                return
            try:
                if self._size >= self._rotate_at:
                    self._rotate()
                self._fh.write(data)
                self._size += len(data)
            except (OSError, ValueError):
                pass

    def _rotate(self) -> None:
        self._fh.close()
        try:
            _locations.rotate_log(self._path, self._max_bytes, self._backup_count)
            self._rotate_at = self._max_bytes
        except OSError as e:
            # Could not move it aside (a reader holds it open on Windows, a
            # read-only directory): keep appending and try again a full limit
            # from here, rather than on every write.
            logger.warning("Cannot rotate %s: %s", self._path, e)
            self._rotate_at = self._size + self._max_bytes
        self._open()

    def close(self) -> None:
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass


def pump(stream: BinaryIO, log: RotatingLog) -> None:
    """Copy ``stream`` into ``log`` until it ends, then close it."""
    try:
        while True:
            data = stream.read1(65536)  # type: ignore[attr-defined]
            if not data:
                break
            log.write(data)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass
