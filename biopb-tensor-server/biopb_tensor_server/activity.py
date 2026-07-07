"""Flight-activity + warm-guard tracking for the tensor server.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). Two cheap,
uncontended concerns share one lock (as they did on the server):

- **Activity** -- counts in-flight heavy reads (``do_get``/``warm``) and stamps
  the last time one finished, so the background precache worker can stay off the
  wire while real traffic flows.
- **Warm guard** -- the set of source ids with a "warm" (hydrate-ahead) recall
  in flight, so a second concurrent warm of the same source is rejected rather
  than doubling the disk/recall pressure.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator, Set


class ActivityTracker:
    """Tracks in-flight Flight reads and the warm-in-progress guard set."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight = 0
        self._last_active = 0.0  # time.monotonic() of last read completion
        self._warming: Set[str] = set()

    @contextlib.contextmanager
    def serving_request(self) -> Iterator[None]:
        """Mark a heavy read in flight for its duration (precache idle signal)."""
        with self._lock:
            self._inflight += 1
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1
                self._last_active = time.monotonic()

    def idle_for(self, seconds: float) -> bool:
        """True if no heavy read is in flight and none finished within *seconds*.

        Used by the precache worker to debounce against live traffic.
        """
        with self._lock:
            if self._inflight > 0:
                return False
            return (time.monotonic() - self._last_active) >= seconds

    def begin_warm(self, source_id: str) -> bool:
        """Try to claim the warm slot for *source_id*.

        Returns ``True`` when the slot was free (and is now held), ``False`` when
        a warm of the same source is already in flight -- the caller rejects the
        duplicate.
        """
        with self._lock:
            if source_id in self._warming:
                return False
            self._warming.add(source_id)
            return True

    def end_warm(self, source_id: str) -> None:
        """Release the warm slot for *source_id* (idempotent)."""
        with self._lock:
            self._warming.discard(source_id)

    @property
    def warming(self) -> Set[str]:
        """A snapshot of the source ids currently warming (read-only view)."""
        with self._lock:
            return set(self._warming)
