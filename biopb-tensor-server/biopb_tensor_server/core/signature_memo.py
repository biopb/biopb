"""Memo of a content probe, keyed on a file's identity.

A claim that reads file content (an OME-XML, a DICOM header) is a pure function of
the file's bytes, so its result is reusable while the file's stat signature --
(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns) -- is unchanged. A rescan then
costs one ``stat`` per file instead of a read and a parse.

Values must be small: the bound is a count, sized to cover a large catalog's
steady-state rescan. An LRU smaller than the scan evicts every entry before the
next pass reaches it.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

T = TypeVar("T")

Signature = Tuple[int, int, int, int, int]


def file_signature(path: "Path | str") -> Optional[Signature]:
    """Identity of a file's current bytes, or ``None`` if it cannot be stat-ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _key(path: "Path | str", signature: Signature) -> bytes:
    """A 128-bit digest of the path and the whole signature: a fraction of the
    memory of keeping both, and a collision is not a practical concern."""
    return hashlib.blake2b(
        f"{path}\0{signature}".encode("utf-8", "surrogateescape"), digest_size=16
    ).digest()


class SignatureMemo:
    """Bounded LRU of ``compute(path)`` results per ``(path, signature)``.

    ``None`` is a result like any other, so membership decides a hit. A
    ``compute`` that raises caches nothing: a transient I/O error must not stand
    in for the file's content until it changes.
    """

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._entries: OrderedDict[bytes, object] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get(
        self,
        path: "Path | str",
        compute: Callable[[], T],
        signature: Optional[Signature] = None,
        *,
        memoize: bool = True,
    ) -> T:
        """The memoized ``compute()`` for *path*; the file's own stat unless
        *signature* is given. A file that cannot be stat-ed is not memoized, and
        ``memoize=False`` (a file that will not be probed again) is a plain call."""
        if not memoize:
            return compute()
        if signature is None:
            signature = file_signature(path)
        if signature is None:
            return compute()

        key = _key(path, signature)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]  # type: ignore[return-value]

        result = compute()

        with self._lock:
            self._entries[key] = result
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return result
