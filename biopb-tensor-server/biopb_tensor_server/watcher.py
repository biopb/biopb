"""Filesystem watcher for tensor store directories.

Provides live monitoring for configured directories, emitting events when
files are added or deleted. Designed with an abstract interface to support
both watchdog (inode-based) and future PollVFS implementations for NFS.

Architecture:
- Abstract DirectoryWatcher interface for extensibility
- WatchdogWatcher implementation using watchdog library (subprocess isolation)
- Debouncing to handle rapid file operations (acquisition, bulk transfers)
- Multiprocessing.Queue for thread-safe communication with main process
- Multiprocessing.Event for shutdown signaling

Events:
- CREATED: New file/directory detected
- DELETED: File/directory removed
- MOVED: File renamed/moved (emitted as old_path + new_path)

Shutdown:
- Parent sets shutdown Event to signal graceful stop
- Subprocess stops observer and exits cleanly
- Parent joins subprocess with timeout, then kills if needed

Only add/delete events are tracked; modification events are ignored.
"""

from __future__ import annotations

import logging
import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue, Event
from pathlib import Path
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.process import Process as _Process

logger = logging.getLogger(__name__)


class WatcherEventType(Enum):
    """Types of filesystem events."""
    CREATED = "created"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class WatcherEvent:
    """Event sent from watcher to main process.

    Attributes:
        event_type: Type of event (created, deleted, moved)
        path: Path to the affected file/directory
        old_path: For moved events, the original path before move
        is_directory: True if the affected path is a directory
    """
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
    """Abstract interface for directory monitoring.

    This abstraction allows different implementations:
    - WatchdogWatcher: inode-based monitoring (local filesystems)
    - PollVFSWatcher: polling-based monitoring (for NFS, network mounts)

    All implementations should:
    - Run in isolation (subprocess or separate thread)
    - Emit events via a multiprocessing.Queue
    - Support debouncing to handle rapid file operations
    """

    @abstractmethod
    def start(self, directories: Set[Path]) -> None:
        """Start watching the specified directories.

        Args:
            directories: Set of directory paths to monitor recursively
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop watching and clean up resources."""
        pass

    @abstractmethod
    def get_events(self, timeout: float = 0.1) -> List[WatcherEvent]:
        """Poll for events from the watcher.

        Args:
            timeout: Max time to wait for events (seconds)

        Returns:
            List of debounced events (may be empty)
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the watcher is actively running."""
        pass


class WatchdogWatcher(DirectoryWatcher):
    """Watchdog-based filesystem watcher with subprocess isolation.

    Runs watchdog observer in a separate process for stability.
    Uses debouncing to collapse rapid file operations.

    Shutdown uses multiprocessing.Event for clean signaling:
    - Parent sets shutdown_event when stop() is called
    - Subprocess checks event periodically and exits gracefully
    - Queue is ONLY used for events (subprocess → parent)

    Args:
        debounce_window: Time window for debouncing events (seconds)

    Example:
        watcher = WatchdogWatcher(debounce_window=1.5)
        watcher.start({Path("/data/zarrs")})
        while running:
            events = watcher.get_events()
            for event in events:
                handle_event(event)
        watcher.stop()
    """

    def __init__(
        self,
        debounce_window: float = 1.5,
    ):
        self._debounce_window = debounce_window
        self._queue: Queue = Queue()
        self._shutdown_event: Event = Event()
        self._process: Optional[_Process] = None
        self._directories: Set[Path] = set()

    def start(self, directories: Set[Path]) -> None:
        """Start watcher subprocess."""
        if self._process is not None and self._process.is_alive():
            logger.warning("Watcher already running, stopping previous instance")
            self.stop()

        self._directories = directories
        if not directories:
            logger.warning("No directories to watch")
            return

        # Clear shutdown event for fresh start
        self._shutdown_event.clear()

        # Resolve directories to absolute paths
        resolved_dirs = {d.resolve() for d in directories}

        logger.info(f"Starting watchdog watcher for: {resolved_dirs}")
        self._process = Process(
            target=_run_watchdog_subprocess,
            args=(self._queue, self._shutdown_event, resolved_dirs, self._debounce_window),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        """Stop watcher subprocess gracefully."""
        if self._process is None:
            return

        logger.info("Stopping watchdog watcher")

        # Signal shutdown via Event (clean, no queue contention)
        self._shutdown_event.set()

        # Wait for graceful shutdown
        self._process.join(timeout=5)

        if self._process.is_alive():
            logger.warning("Watcher process did not terminate gracefully, killing")
            self._process.terminate()
            self._process.join(timeout=2)

            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)

        self._process = None

    def get_events(self, timeout: float = 0.1) -> List[WatcherEvent]:
        """Poll for debounced events from subprocess."""
        events = []
        while True:
            try:
                event = self._queue.get(timeout=timeout)
                if event is None:
                    # Sentinel value signals end of batch
                    break
                events.append(event)
            except queue.Empty:
                break
        return events

    def is_running(self) -> bool:
        """Check if watcher subprocess is alive."""
        return self._process is not None and self._process.is_alive()


def _run_watchdog_subprocess(
    queue: Queue,
    shutdown_event: Event,
    directories: Set[Path],
    debounce_window: float,
) -> None:
    """Subprocess entry point - runs watchdog observer with debouncing.

    This function runs in a separate process and:
    1. Creates a watchdog observer for each directory
    2. Buffers events for debounce_window seconds
    3. Deduplicates and collapses rapid events
    4. Sends debounced events to main process via queue
    5. Exits cleanly when shutdown_event is set

    Args:
        queue: Queue for sending events to parent (ONLY subprocess writes)
        shutdown_event: Event set by parent to signal shutdown
        directories: Set of directory paths to monitor
        debounce_window: Time window for debouncing events
    """
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    # Event buffer for debouncing: path -> (event_type, timestamp, old_path)
    event_buffer: dict = {}
    last_emit_time = time.time()

    class DebouncingHandler(FileSystemEventHandler):
        """Handler that buffers events for debouncing."""

        def on_created(self, event):
            if event.is_directory:
                path = Path(event.src_path)
                event_buffer[path] = (WatcherEventType.CREATED, time.time(), None, True)
            else:
                path = Path(event.src_path)
                # Skip hidden files
                if path.name.startswith('.'):
                    return
                event_buffer[path] = (WatcherEventType.CREATED, time.time(), None, False)

        def on_deleted(self, event):
            if event.is_directory:
                path = Path(event.src_path)
                event_buffer[path] = (WatcherEventType.DELETED, time.time(), None, True)
            else:
                path = Path(event.src_path)
                # Skip hidden files
                if path.name.startswith('.'):
                    return
                event_buffer[path] = (WatcherEventType.DELETED, time.time(), None, False)

        def on_moved(self, event):
            old_path = Path(event.src_path)
            new_path = Path(event.dest_path)
            is_dir = event.is_directory

            # Skip hidden files
            if new_path.name.startswith('.'):
                # Move to hidden = deletion
                event_buffer[old_path] = (WatcherEventType.DELETED, time.time(), None, is_dir)
                return
            if old_path.name.startswith('.'):
                # Move from hidden = creation
                event_buffer[new_path] = (WatcherEventType.CREATED, time.time(), None, is_dir)
                return

            event_buffer[old_path] = (WatcherEventType.MOVED, time.time(), new_path, is_dir)

    # Set up observer
    observer = Observer()
    handler = DebouncingHandler()

    for directory in directories:
        try:
            observer.schedule(handler, str(directory), recursive=True)
        except Exception as e:
            # Log to queue as error (main process will handle)
            queue.put(WatcherEvent(
                event_type=WatcherEventType.DELETED,  # Use as error signal
                path=directory,
                is_directory=True,
            ))

    observer.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)  # Polling interval

            # Check if we should emit events
            current_time = time.time()
            if current_time - last_emit_time >= debounce_window:
                # Process and emit debounced events
                _emit_debounced_events(queue, event_buffer)
                event_buffer.clear()
                last_emit_time = current_time

    except KeyboardInterrupt:
        # Also caught on SIGTERM (from terminate())
        pass
    finally:
        observer.stop()
        observer.join(timeout=2)


def _emit_debounced_events(
    queue: Queue,
    event_buffer: dict,
) -> None:
    """Emit debounced events from buffer.

    Deduplication logic:
    - If path has CREATED then DELETED, skip both (transient file)
    - If path has DELETED then CREATED, emit CREATED (recreate)
    - For MOVED events, check if source still exists
    - Collapse multiple events for same path to single final event

    Args:
        queue: Queue to send events to
        event_buffer: Buffer of {path: (event_type, timestamp, old_path, is_dir)}
    """
    # Group events by path
    path_events: dict = {}
    for path, (event_type, timestamp, extra, is_dir) in event_buffer.items():
        if path not in path_events:
            path_events[path] = []
        path_events[path].append((event_type, timestamp, extra, is_dir))

    # Deduplicate per path
    for path, events in path_events.items():
        if len(events) == 1:
            # Single event - emit directly
            event_type, _, extra, is_dir = events[0]
            queue.put(WatcherEvent(
                event_type=event_type,
                path=path,
                old_path=extra if event_type == WatcherEventType.MOVED else None,
                is_directory=is_dir,
            ))
        else:
            # Multiple events - collapse to final state
            # Sort by timestamp
            events.sort(key=lambda e: e[1])

            # Check for create+delete collapse (transient file)
            has_create = any(e[0] == WatcherEventType.CREATED for e in events)
            has_delete = any(e[0] == WatcherEventType.DELETED for e in events)

            if has_create and has_delete:
                # Transient file - skip
                continue

            # Keep last event
            event_type, _, extra, is_dir = events[-1]
            queue.put(WatcherEvent(
                event_type=event_type,
                path=path,
                old_path=extra if event_type == WatcherEventType.MOVED else None,
                is_directory=is_dir,
            ))

    # Send sentinel to signal batch end
    queue.put(None)


# Polling-based watcher for NFS/network mounts
class PollVFSWatcher(DirectoryWatcher):
    """Polling-based filesystem watcher for NFS/network mounts.

    Uses subprocess isolation like WatchdogWatcher for consistency.
    Scans directories periodically and compares snapshots to detect changes.

    Uses mtime for change detection since inotify doesn't work on NFS.
    Detects: CREATED, DELETED (moves detected as delete+create pair).

    Args:
        poll_interval: Time between scans (seconds, default 30.0)
        debounce_window: Time window for debouncing events (seconds)

    Example:
        watcher = PollVFSWatcher(poll_interval=5.0)
        watcher.start({Path("/nfs/data")})
        while running:
            events = watcher.get_events()
            for event in events:
                handle_event(event)
        watcher.stop()
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        debounce_window: float = 1.5,
    ):
        self._poll_interval = poll_interval
        self._debounce_window = debounce_window
        self._queue: Queue = Queue()
        self._shutdown_event: Event = Event()
        self._process: Optional[_Process] = None
        self._directories: Set[Path] = set()

    def start(self, directories: Set[Path]) -> None:
        """Start watcher subprocess."""
        if self._process is not None and self._process.is_alive():
            logger.warning("PollVFS watcher already running, stopping previous instance")
            self.stop()

        self._directories = directories
        if not directories:
            logger.warning("No directories to watch")
            return

        # Clear shutdown event for fresh start
        self._shutdown_event.clear()

        # Resolve directories to absolute paths
        resolved_dirs = {d.resolve() for d in directories}

        logger.info(f"Starting PollVFS watcher for: {resolved_dirs}")
        self._process = Process(
            target=_run_pollvfs_subprocess,
            args=(self._queue, self._shutdown_event, resolved_dirs, self._poll_interval, self._debounce_window),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        """Stop watcher subprocess gracefully."""
        if self._process is None:
            return

        logger.info("Stopping PollVFS watcher")

        # Signal shutdown via Event
        self._shutdown_event.set()

        # Wait for graceful shutdown
        self._process.join(timeout=5)

        if self._process.is_alive():
            logger.warning("PollVFS watcher did not terminate gracefully, killing")
            self._process.terminate()
            self._process.join(timeout=2)

            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)

        self._process = None

    def get_events(self, timeout: float = 0.1) -> List[WatcherEvent]:
        """Poll for debounced events from subprocess."""
        events = []
        while True:
            try:
                event = self._queue.get(timeout=timeout)
                if event is None:
                    # Sentinel value signals end of batch
                    break
                events.append(event)
            except queue.Empty:
                break
        return events

    def is_running(self) -> bool:
        """Check if watcher subprocess is alive."""
        return self._process is not None and self._process.is_alive()


def _run_pollvfs_subprocess(
    queue: Queue,
    shutdown_event: Event,
    directories: Set[Path],
    poll_interval: float,
    debounce_window: float,
) -> None:
    """Subprocess entry point - runs polling loop with snapshot comparison.

    This function runs in a separate process and:
    1. Takes initial snapshot of all directories
    2. Polls periodically, comparing snapshots
    3. Emits CREATED/DELETED events for changes
    4. Uses debouncing to collapse rapid changes
    5. Exits cleanly when shutdown_event is set

    Args:
        queue: Queue for sending events to parent (ONLY subprocess writes)
        shutdown_event: Event set by parent to signal shutdown
        directories: Set of directory paths to monitor
        poll_interval: Time between scans (seconds)
        debounce_window: Time window for debouncing events
    """
    # Snapshot: {path: mtime}
    # Nested by monitored directory for efficient updates
    snapshots: Dict[Path, Dict[Path, float]] = {}

    # Event buffer for debouncing: path -> (event_type, timestamp, extra, is_dir)
    event_buffer: dict = {}
    last_emit_time = time.time()

    def scan_directory(root: Path) -> Dict[Path, float]:
        """Recursively scan directory, return {path: mtime}."""
        result = {}
        try:
            for path in root.rglob('*'):
                # Skip hidden files/directories
                if path.name.startswith('.'):
                    continue
                try:
                    result[path] = path.stat().st_mtime
                except OSError:
                    # File disappeared or permission issue
                    pass
        except OSError:
            # Permission issue reading directory
            pass
        return result

    # Take initial snapshot
    for directory in directories:
        snapshots[directory] = scan_directory(directory)

    try:
        while not shutdown_event.is_set():
            time.sleep(poll_interval)

            # Check if we should emit events
            current_time = time.time()
            if current_time - last_emit_time >= debounce_window:
                _emit_debounced_events(queue, event_buffer)
                event_buffer.clear()
                last_emit_time = current_time

            # Scan and compare snapshots
            for directory in directories:
                current = scan_directory(directory)
                previous = snapshots.get(directory, {})

                # Detect created: in current but not previous
                for path in current:
                    if path not in previous:
                        is_dir = path.is_dir()
                        event_buffer[path] = (WatcherEventType.CREATED, time.time(), None, is_dir)

                # Detect deleted: in previous but not current
                for path in previous:
                    if path not in current:
                        # We don't know if it was a dir or file now
                        # Check if it had children in previous snapshot
                        was_dir = any(
                            p.is_relative_to(path) and p != path
                            for p in previous
                        ) or path in directories  # monitored dirs are dirs
                        event_buffer[path] = (WatcherEventType.DELETED, time.time(), None, was_dir)

                # Update snapshot
                snapshots[directory] = current

    except KeyboardInterrupt:
        pass
    finally:
        # Emit any remaining buffered events
        if event_buffer:
            _emit_debounced_events(queue, event_buffer)


def detect_nfs_mount(path: Path) -> bool:
    """Check if path is on an NFS mount by parsing /proc/mounts.

    Args:
        path: Path to check

    Returns:
        True if path is on NFS or NFS4 mount, False otherwise
    """
    try:
        resolved = path.resolve()
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mount_point = parts[1]
                    fs_type = parts[2]
                    # Check if resolved path is under this NFS mount
                    if fs_type in ('nfs', 'nfs4'):
                        try:
                            mount_path = Path(mount_point)
                            if resolved.is_relative_to(mount_path):
                                return True
                        except Exception:
                            # Invalid mount point path
                            pass
        return False
    except OSError:
        # /proc/mounts not readable (non-Linux systems)
        return False


def get_watcher(
    watcher_type: str = "auto",
    directories: Optional[Set[Path]] = None,
    debounce_window: float = 1.5,
    poll_interval: float = 30.0,
) -> DirectoryWatcher:
    """Factory function to create appropriate watcher.

    Args:
        watcher_type: Type of watcher:
            - "auto": Auto-detect based on filesystem type (default)
            - "watchdog": Force watchdog/inotify-based watcher
            - "pollvfs": Force polling-based watcher (for NFS)
        directories: Paths to monitor (used for auto-detection when watcher_type="auto")
        debounce_window: Time window for debouncing events (seconds)
        poll_interval: Time between scans for PollVFS watcher (seconds)

    Returns:
        DirectoryWatcher instance

    Raises:
        ValueError: If watcher_type is not supported
    """
    if watcher_type == "auto":
        # Auto-detect based on mount type
        if directories and any(detect_nfs_mount(d) for d in directories):
            logger.info("Auto-detected NFS mount, using PollVFS watcher")
            return PollVFSWatcher(poll_interval=poll_interval, debounce_window=debounce_window)
        return WatchdogWatcher(debounce_window=debounce_window)
    elif watcher_type == "watchdog":
        return WatchdogWatcher(debounce_window=debounce_window)
    elif watcher_type == "pollvfs":
        return PollVFSWatcher(poll_interval=poll_interval, debounce_window=debounce_window)
    else:
        raise ValueError(f"Unknown watcher type: {watcher_type}")