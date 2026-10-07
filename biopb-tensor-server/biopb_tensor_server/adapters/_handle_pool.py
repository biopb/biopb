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

from biopb_tensor_server.adapters._handle_reaper import TtlCeiling

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


class HandlePool(TtlCeiling):
    """Open handles by key, with an idle TTL (``<= 0`` keeps nothing between
    reads) and an LRU cap."""

    def __init__(self, ttl_seconds: float, max_handles: int, thread_name: str) -> None:
        super().__init__(ttl_seconds)
        self.max_handles = int(max_handles)
        self._thread_name = thread_name
        self._handles: Dict[Hashable, PooledHandle] = {}
        self._opening: Dict[Hashable, threading.Lock] = {}
        self._lock = threading.Lock()
        self._started = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._handles)

    @contextmanager
    def checkout(
        self,
        key: Hashable,
        open_fn: Callable[[], Optional[PooledHandle]],
        *,
        persist: bool = True,
    ) -> Iterator[Optional[PooledHandle]]:
        """Lease the handle for *key*, opening it with *open_fn* on a miss.

        ``open_fn`` returns a :class:`PooledHandle`, or None when the file has
        no usable handle (the lease then yields None). Concurrent misses on one
        key open once: the rest wait for the first and lease its handle.
        ``persist=False`` opens a handle for this lease alone and closes it at
        the end, never publishing it.
        """
        if not persist:
            handle = open_fn()
            try:
                yield handle
            finally:
                if handle is not None:
                    handle.close()
            return
        handle = self._lease(key)
        if handle is None:
            opening = self._opening_lock(key)
            try:
                with opening:
                    handle = self._lease(key)
                    if handle is None:
                        opened = open_fn()
                        if opened is not None:
                            handle = self._publish(opened)
            finally:
                with self._lock:
                    if self._opening.get(key) is opening:
                        del self._opening[key]
            if handle is None:
                yield None
                return
        try:
            yield handle
        finally:
            self._release(handle)

    def put(self, opened: PooledHandle) -> None:
        """Pool a handle the caller already opened, as if a checkout had."""
        self._release(self._publish(opened))

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

    def _opening_lock(self, key: Hashable) -> threading.Lock:
        with self._lock:
            return self._opening.setdefault(key, threading.Lock())

    @staticmethod
    def _touch(handle: PooledHandle) -> None:
        handle.leases += 1
        handle.last_access = time.monotonic()

    def _lease(self, key: Hashable) -> Optional[PooledHandle]:
        with self._lock:
            handle = self._handles.get(key)
            if handle is not None and not handle.doomed:
                self._touch(handle)
                return handle
        return None

    def _publish(self, opened: PooledHandle) -> PooledHandle:
        """Pool *opened* leased once; an earlier handle for its key wins and
        *opened* is closed."""
        excess = []
        loser = None
        with self._lock:
            winner = self._handles.get(opened.key)
            if winner is not None and not winner.doomed:
                loser = opened
            else:
                winner = opened
                self._handles[opened.key] = opened
            self._touch(winner)
            if loser is None:
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
            if (handle.doomed or self.ttl <= 0) and not handle.leases:
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
