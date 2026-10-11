"""Background registration of the sources the first scan only claimed.

Registration opens and parses each source's file, which is what makes a first
scan of a large site take hours. The scan therefore commits every claim to the
catalog alone (``Reconciler._commit_pending_claim``) and queues the source here; a
small pool of threads registers them, newest file first, so what a user is most
likely to want is complete soonest. A client that resolves a source before its
turn registers it itself (``Reconciler.ensure_registered``, single-flight), and
the worker finds it done when it gets there.

The pool can be held while the first scan walks (``pause``): registering and
walking contend, and a held pool lets the walk finish, and so the whole catalog
exist as pending rows, before any file is opened.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RegistrationWorker:
    """A pool of threads that register queued sources, newest mtime first.

    *register* is ``Reconciler.ensure_registered``: idempotent, single-flight per
    source, and never raising for a source that failed (it records the failure).
    """

    def __init__(self, register: Callable[[str], bool], workers: int = 4) -> None:
        if workers < 1:
            raise ValueError("a registration worker pool needs at least one thread")
        self._register = register
        self._workers = workers
        self._cond = threading.Condition()
        self._heap: List[Tuple[float, int, str]] = []
        self._queued: Set[str] = set()
        self._seq = 0
        self._paused = False
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for i in range(self._workers):
            thread = threading.Thread(
                target=self._run, name=f"RegistrationWorker-{i}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, join_timeout: float = 5.0) -> None:
        """Stop the pool, waiting at most *join_timeout* in all.

        A worker in the middle of opening a large file cannot be interrupted, so
        one deadline is shared rather than one per thread: they are daemons, and
        what is not done by then is left to die with the process.
        """
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        deadline = time.monotonic() + join_timeout
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._threads = []

    def enqueue(self, source_id: str, mtime: float = 0.0) -> None:
        """Queue *source_id*; a source already queued keeps its place."""
        with self._cond:
            if source_id in self._queued:
                return
            self._queued.add(source_id)
            self._seq += 1
            heapq.heappush(self._heap, (-mtime, self._seq, source_id))
            self._cond.notify()

    def pause(self) -> None:
        """Hold queued sources back. A registration already under way finishes."""
        with self._cond:
            self._paused = True

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def queued(self) -> int:
        with self._cond:
            return len(self._heap)

    def _pop(self) -> Optional[str]:
        with self._cond:
            while self._paused or not self._heap:
                if self._stop.is_set():
                    return None
                self._cond.wait()
            if self._stop.is_set():
                return None
            _, _, source_id = heapq.heappop(self._heap)
            self._queued.discard(source_id)
            return source_id

    def _run(self) -> None:
        while not self._stop.is_set():
            source_id = self._pop()
            if source_id is None:
                return
            try:
                self._register(source_id)
            except Exception:
                logger.exception("background registration of %s failed", source_id)
