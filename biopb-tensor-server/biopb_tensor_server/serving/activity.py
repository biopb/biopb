"""Flight-activity tracking for the tensor server.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A): counts in-flight
heavy reads (``do_get``) and stamps the last time one finished, so the
background precache worker can stay off the wire while real traffic flows.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator


class ActivityTracker:
    """Tracks in-flight Flight reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight = 0
        self._last_active = 0.0  # time.monotonic() of last read completion

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
