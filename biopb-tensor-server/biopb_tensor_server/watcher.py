"""Periodic rescan watcher for monitored source directories.

The runtime monitoring model is timer-driven: the watcher emits a RESCAN event at
fixed intervals and leaves filesystem stability checks and discovery diffs to the
SourceManager.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class WatcherEventType(Enum):
    """Watcher event types.

    Only RESCAN is produced by the current runtime. Legacy values remain so stale
    callers or tests can still construct old event objects without import breakage.
    """

    RESCAN = "rescan"
    CREATED = "created"
    DELETED = "deleted"
    MOVED = "moved"
    CLOSED = "closed"


@dataclass
class WatcherEvent:
    """Event emitted by the directory watcher."""

    event_type: WatcherEventType
    path: Path
    old_path: Optional[Path] = None
    is_directory: bool = False

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if isinstance(self.old_path, str):
            self.old_path = Path(self.old_path)


class DirectoryWatcher(ABC):
    """Abstract watcher interface for periodic rescans."""

    @abstractmethod
    def start(self, directories: Set[Path]) -> None:
        """Start monitoring the specified directories."""

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring and release resources."""

    @abstractmethod
    def get_events(self, timeout: float = 0.1) -> List[WatcherEvent]:
        """Return any watcher events available within timeout seconds."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return True when the watcher is active."""


class PeriodicRescanWatcher(DirectoryWatcher):
    """Timer-driven watcher that requests periodic rescans."""

    def __init__(self, rescan_interval: float = 30.0):
        self._rescan_interval = max(0.1, rescan_interval)
        self._directories: Set[Path] = set()
        self._running = False
        self._next_rescan_at: Optional[float] = None

    def start(self, directories: Set[Path]) -> None:
        self._directories = {directory.resolve() for directory in directories}
        self._running = True
        self._next_rescan_at = time.monotonic() + self._rescan_interval
        logger.info(
            "Starting periodic rescan watcher for %s every %.1fs",
            self._directories,
            self._rescan_interval,
        )

    def stop(self) -> None:
        self._running = False
        self._next_rescan_at = None
        logger.debug("Periodic rescan watcher stopped")

    def get_events(self, timeout: float = 0.1) -> List[WatcherEvent]:
        if not self._running:
            return []

        deadline = time.monotonic() + max(timeout, 0.0)
        while self._running:
            now = time.monotonic()
            if self._next_rescan_at is None:
                self._next_rescan_at = now + self._rescan_interval

            if now >= self._next_rescan_at:
                self._next_rescan_at = now + self._rescan_interval
                return [WatcherEvent(WatcherEventType.RESCAN, Path("."))]

            if now >= deadline:
                return []

            sleep_for = min(0.05, deadline - now, self._next_rescan_at - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                return []

        return []

    def is_running(self) -> bool:
        return self._running


def get_watcher(
    watcher_type: str = "auto",
    directories: Optional[Set[Path]] = None,
    debounce_window: float = 1.5,
    poll_interval: float = 30.0,
) -> DirectoryWatcher:
    """Create the configured watcher.

    The runtime supports only periodic rescans. Legacy watcher_type values map to
    the same implementation so existing configs continue to work during migration.
    """
    del directories, debounce_window

    if watcher_type in {"auto", "watchdog", "pollvfs", "periodic", "off"}:
        return PeriodicRescanWatcher(rescan_interval=poll_interval)
    raise ValueError(f"Unknown watcher type: {watcher_type}")
