"""A pool of open file handles keyed by file identity, not by adapter.

An adapter that owns its handle loses it when the adapter is dropped, so
evicting and rebuilding a source pays the file's reopen on the next read. Here
the pool holds the handle: it is keyed by what the file is (path, scene,
content version), so a rebuilt adapter with the same key finds the handle its
predecessor opened.

``checkout`` returns a lease. A handle with a live lease is never closed: an
idle sweep, the cap and ``drop`` all skip it (``drop`` closes it when the last
lease is released), so a reader needs no other fence. Each handle carries its
own lock for readers that must serialize on it.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Hashable, Iterator, Optional

logger = logging.getLogger(__name__)


class PooledHandle:
    """An open handle plus the bookkeeping the pool needs."""

    def __init__(self, key: Hashable, value: Any, close: Callable[[], None]) -> None:
        self.key = key
        self.value = value
        self.lock = threading.Lock()
        self._close = close
        self.leases = 0
        self.last_access = time.monotonic()
        self.doomed = False

    def close(self) -> None:
        try:
            self._close()
        except Exception:
            logger.debug("error closing pooled handle %r", self.key, exc_info=True)


class HandlePool:
    def __init__(self, ttl_seconds: float, max_handles: int, thread_name: str) -> None:
        self.ttl = float(ttl_seconds)
        self.max_handles = int(max_handles)
        self._thread_name = thread_name
        self._handles: Dict[Hashable, PooledHandle] = {}
        self._lock = threading.Lock()
        self._started = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._handles)

    @contextmanager
    def checkout(
        self, key: Hashable, open_fn: Callable[[], Optional[PooledHandle]]
    ) -> Iterator[Optional[PooledHandle]]:
        """Lease the handle for *key*, opening it with *open_fn* on a miss.

        ``open_fn`` returns a :class:`PooledHandle`, or None when the file has
        no usable handle (the lease then yields None). Two threads missing at
        once may both open; the first to publish wins and the other's is closed.
        """
        handle = self._lease(key)
        if handle is None:
            opened = open_fn()
            if opened is None:
                yield None
                return
            handle = self._publish(opened)
        try:
            yield handle
        finally:
            self._release(handle)

    def drop(self, key: Hashable) -> None:
        """Close the handle for *key* now, or at its last lease."""
        with self._lock:
            handle = self._handles.get(key)
            if handle is None:
                return
            if handle.leases:
                handle.doomed = True
                return
            del self._handles[key]
        handle.close()

    def sweep(self) -> None:
        """Close idle handles past the TTL. Never touches a leased handle."""
        now = time.monotonic()
        with self._lock:
            idle = [
                h
                for h in self._handles.values()
                if not h.leases and now - h.last_access > self.ttl
            ]
            for h in idle:
                del self._handles[h.key]
        for h in idle:
            h.close()

    def _lease(self, key: Hashable) -> Optional[PooledHandle]:
        with self._lock:
            handle = self._handles.get(key)
            if handle is not None and not handle.doomed:
                handle.leases += 1
                handle.last_access = time.monotonic()
                return handle
        return None

    def _publish(self, opened: PooledHandle) -> PooledHandle:
        excess = []
        with self._lock:
            existing = self._handles.get(opened.key)
            if existing is not None and not existing.doomed:
                existing.leases += 1
                existing.last_access = time.monotonic()
                winner, loser = existing, opened
            else:
                opened.leases += 1
                self._handles[opened.key] = opened
                winner, loser = opened, None
                excess = self._over_cap()
            self._start_sweeper()
        if loser is not None:
            loser.close()
        for h in excess:
            h.close()
        return winner

    def _over_cap(self) -> list:
        """Unlink the least recently used unleased handles beyond the cap.
        Caller holds ``_lock``; the caller closes them after releasing it."""
        spare = len(self._handles) - self.max_handles
        if spare <= 0:
            return []
        idle = sorted(
            (h for h in self._handles.values() if not h.leases),
            key=lambda h: h.last_access,
        )[:spare]
        for h in idle:
            del self._handles[h.key]
        return idle

    def _release(self, handle: PooledHandle) -> None:
        close = False
        with self._lock:
            handle.leases -= 1
            handle.last_access = time.monotonic()
            if handle.doomed and not handle.leases:
                if self._handles.get(handle.key) is handle:
                    del self._handles[handle.key]
                close = True
        if close:
            handle.close()

    def _start_sweeper(self) -> None:
        if self._started or self.ttl <= 0:
            return
        self._started = True
        threading.Thread(target=self._loop, name=self._thread_name, daemon=True).start()

    def _loop(self) -> None:
        while True:
            time.sleep(max(1.0, min(self.ttl / 4.0, 30.0)))
            try:
                self.sweep()
            except Exception:  # pragma: no cover - the sweeper must never die
                logger.debug("%s sweep failed", self._thread_name, exc_info=True)
