"""Source lifecycle manager for tensor store catalog.

Coordinates filesystem watcher, discovery protocol, and server catalog updates.
Handles dynamic source registration/unregistration when files are added or
deleted in monitored directories.

Thread Safety:
- All server updates (register/unregister) are serialized via a thread-safe queue
- DiscoveryState mutations are protected by a lock
- Callbacks are invoked from the event processing thread, not watcher subprocess

Architecture:
- SourceManager runs event loop in separate thread
- Events arrive from watcher subprocess via multiprocessing.Queue
- Server updates are batched and applied atomically

Move Handling:
- Moves WITHIN monitored directory: preserve source_id, update path
- Moves OUT OF monitored directory: treat as delete
- Moves INTO monitored directory: treat as create with new source_id
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from biopb_tensor_server.config import SourceConfig
from biopb_tensor_server.discovery import (
    AdapterRegistry,
    DiscoveryState,
    SourceClaim,
    discover_sources,
    get_file_identity,
)
from biopb_tensor_server.watcher import (
    DirectoryWatcher,
    WatcherEvent,
    WatcherEventType,
)

if TYPE_CHECKING:
    from biopb_tensor_server.server import TensorFlightServer

logger = logging.getLogger(__name__)


class SourceManager:
    """Manages dynamic source registration/unregistration.

    Coordinates:
    - Filesystem watcher (subprocess sending events)
    - DiscoveryState (claim tracking with callbacks)
    - TensorFlightServer (register/unregister sources)
    - AdapterRegistry (create adapters for new sources)

    The manager runs an event processing loop in a background thread,
    receiving debounced events from the watcher and updating the catalog.

    Args:
        server: TensorFlightServer instance for source registration
        registry: AdapterRegistry for adapter creation
        discovery_state: DiscoveryState for claim tracking
        watcher: DirectoryWatcher for filesystem events

    Example:
        manager = SourceManager(server, registry, state, watcher)
        manager.start()
        # ... server runs ...
        manager.stop()
    """

    def __init__(
        self,
        server: TensorFlightServer,
        registry: AdapterRegistry,
        discovery_state: DiscoveryState,
        watcher: Optional[DirectoryWatcher],
        monitored_dirs: Set[Path],
        dim_labels: Optional[List[str]] = None,
    ):
        self._server = server
        self._registry = registry
        self._state = discovery_state
        self._watcher = watcher
        self._monitored_dirs = monitored_dirs
        self._dim_labels = dim_labels

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Path tracking for move handling
        # Maps resolved path -> source_id
        self._path_to_source_id: Dict[Path, str] = {}

        # Initialize path tracking from existing claims
        for source_id, claim in self._state.claims.items():
            self._path_to_source_id[claim.primary_path.resolve()] = source_id

        # Set up discovery callbacks
        self._state.on_source_added = self._on_source_added
        self._state.on_source_removed = self._on_source_removed

    def start(self) -> None:
        """Start the event processing loop."""
        if self._watcher is None:
            return  # Static-only mode; no filesystem events to process

        if self._thread is not None and self._thread.is_alive():
            logger.warning("SourceManager already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="SourceManager-EventLoop",
        )
        self._thread.start()
        logger.info("SourceManager started")

    def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("SourceManager stopped")

    def is_running(self) -> bool:
        """Check if the manager is actively processing events."""
        return self._running and (self._thread is not None and self._thread.is_alive())

    def _event_loop(self) -> None:
        """Process watcher events and update catalog.

        This loop runs in a background thread:
        1. Poll watcher for events (debounced by watcher)
        2. Process each event (create/delete/move)
        3. Update DiscoveryState and server catalog
        """
        while self._running:
            try:
                events = self._watcher.get_events(timeout=0.5)

                if events:
                    with self._lock:
                        for event in events:
                            self._process_event(event)

            except Exception as e:
                logger.error(f"Error processing events: {e}", exc_info=True)

            # Small sleep to prevent busy polling
            time.sleep(0.1)

    def _process_event(self, event: WatcherEvent) -> None:
        """Handle a filesystem event.

        Args:
            event: WatcherEvent from the watcher
        """
        try:
            if event.event_type == WatcherEventType.CREATED:
                self._handle_created(event.path, event.is_directory)
            elif event.event_type == WatcherEventType.DELETED:
                self._handle_deleted(event.path, event.is_directory)
            elif event.event_type == WatcherEventType.MOVED:
                self._handle_moved(event.old_path, event.path, event.is_directory)
        except Exception as e:
            logger.error(f"Error handling event {event}: {e}", exc_info=True)

    def _handle_created(self, path: Path, is_directory: bool) -> None:
        """Handle file/directory creation.

        For files: Try to claim as new source.
        For directories: Check if it's a data source, otherwise monitor children.

        Args:
            path: Path to created file/directory
            is_directory: True if path is a directory
        """
        resolved_path = path.resolve()

        # Skip if already claimed
        if self._state.is_path_claimed(resolved_path):
            return

        # Skip hidden paths
        if resolved_path.name.startswith('.'):
            return

        # Get identity for deduplication
        try:
            identity = get_file_identity(resolved_path)
            if identity in self._state.visited_identities:
                return
            self._state.visited_identities.add(identity)
        except OSError:
            return

        # Try to claim
        claims = self._registry.get_claims_for_path(
            resolved_path,
            self._state.visited_identities,
        )

        if claims:
            claim = claims[0]
            if self._dim_labels and claim.dim_labels is None:
                claim.dim_labels = self._dim_labels

            # Add claimed paths to visited identities
            for claimed_path in claim.claimed_paths:
                try:
                    claimed_identity = get_file_identity(claimed_path)
                    self._state.visited_identities.add(claimed_identity)
                except OSError:
                    pass

            # Add to discovery state (callback will register with server)
            self._state.add_claim(claim)
            logger.info(f"Added source: {claim.source_id} at {claim.primary_path}")

    def _handle_deleted(self, path: Path, is_directory: bool) -> None:
        """Handle file/directory deletion.

        For files: Remove the source if path matches a claim.
        For directories: Remove all sources inside the directory.

        Args:
            path: Path to deleted file/directory
            is_directory: True if path is a directory
        """
        resolved_path = path.resolve()

        if is_directory:
            # Cascade deletion - remove all sources inside this directory
            self._handle_directory_deleted(resolved_path)
        else:
            # Single file deletion
            source_id = self._state.get_source_for_path(resolved_path)
            if source_id:
                self._state.remove_claim(resolved_path)
                logger.info(f"Removed source: {source_id} at {path}")

    def _handle_directory_deleted(self, directory: Path) -> None:
        """Remove all sources inside a deleted directory.

        Args:
            directory: Path to deleted directory
        """
        # Find all claims with paths inside this directory
        removed_ids = []
        with self._lock:
            for source_id, claim in list(self._state.claims.items()):
                for claimed_path in claim.claimed_paths:
                    try:
                        if claimed_path.resolve().is_relative_to(directory):
                            # This claim is inside the deleted directory
                            self._state.remove_claim(claim.primary_path)
                            removed_ids.append(source_id)
                            break
                    except OSError:
                        # Path no longer exists
                        pass

        if removed_ids:
            logger.info(f"Removed {len(removed_ids)} sources from deleted directory: {directory}")

    def _handle_moved(
        self,
        old_path: Path,
        new_path: Path,
        is_directory: bool,
    ) -> None:
        """Handle file/directory move.

        Move semantics:
        - Within monitored directory: preserve source_id, update paths
        - Out of monitored directory: treat as delete
        - Into monitored directory: treat as create

        Args:
            old_path: Original path before move
            new_path: New path after move
            is_directory: True if path is a directory
        """
        resolved_old = old_path.resolve()
        resolved_new = new_path.resolve()

        # Check if move is within monitored directories
        old_in_monitored = any(
            resolved_old.is_relative_to(d) for d in self._monitored_dirs
        )
        new_in_monitored = any(
            resolved_new.is_relative_to(d) for d in self._monitored_dirs
        )

        if old_in_monitored and new_in_monitored:
            # Move within monitored area - preserve source_id
            self._handle_move_within(resolved_old, resolved_new, is_directory)
        elif old_in_monitored and not new_in_monitored:
            # Move out of monitored area - treat as delete
            self._handle_deleted(old_path, is_directory)
        elif not old_in_monitored and new_in_monitored:
            # Move into monitored area - treat as create
            self._handle_created(new_path, is_directory)
        else:
            # Move between unmonitored areas - ignore
            pass

    def _handle_move_within(
        self,
        old_path: Path,
        new_path: Path,
        is_directory: bool,
    ) -> None:
        """Handle move within monitored directory - preserve source_id.

        Args:
            old_path: Original resolved path
            new_path: New resolved path
            is_directory: True if path is a directory
        """
        source_id = self._state.get_source_for_path(old_path)

        if source_id:
            claim = self._state.claims.get(source_id)
            if claim:
                # Update claim paths
                new_claimed_paths = set()
                for claimed_path in claim.claimed_paths:
                    try:
                        if claimed_path.resolve().is_relative_to(old_path):
                            # This path moved with the source
                            relative = claimed_path.relative_to(old_path)
                            new_claimed_path = new_path / relative
                            new_claimed_paths.add(new_claimed_path)
                        else:
                            new_claimed_paths.add(claimed_path)
                    except OSError:
                        pass

                # Update claim
                new_claim = SourceClaim(
                    source_type=claim.source_type,
                    primary_path=new_path,
                    claimed_paths=new_claimed_paths,
                    source_id=source_id,  # Preserve source_id
                    dim_labels=claim.dim_labels,
                    extra_config=claim.extra_config,
                )

                # Remove old claim, add new claim (callbacks handle server update)
                self._state.remove_claim(old_path)
                self._state.add_claim(new_claim)

                logger.info(f"Moved source {source_id}: {old_path} -> {new_path}")
        else:
            # No existing claim - treat as create at new location
            self._handle_created(new_path, is_directory)

    def _on_source_added(self, claim: SourceClaim) -> None:
        """Callback when source is added to DiscoveryState.

        Creates adapter and registers with Flight server.

        Args:
            claim: SourceClaim that was added
        """
        try:
            # Create source config from claim
            source_config = SourceConfig(
                type=claim.source_type,
                url=str(claim.primary_path),
                source_id=claim.source_id,
                dim_labels=claim.dim_labels,
                dataset=claim.extra_config.get('dataset'),
            )

            # Get adapter class and create instance
            adapter_cls = self._registry.get_adapter_for_type(claim.source_type)
            if adapter_cls is None:
                logger.error(f"No adapter for type: {claim.source_type}")
                return

            adapter = adapter_cls.create_from_config(source_config)

            # Register with server
            self._server.register_source(claim.source_id, adapter)

            # Update path tracking for move handling
            self._path_to_source_id[claim.primary_path.resolve()] = claim.source_id

            logger.info(f"Registered source with server: {claim.source_id}")

        except Exception as e:
            logger.error(
                f"Failed to create/register source {claim.source_id}: {e}",
                exc_info=True,
            )

    def _on_source_removed(self, source_id: str) -> None:
        """Callback when source is removed from DiscoveryState.

        Unregisters from Flight server.

        Args:
            source_id: ID of removed source
        """
        try:
            self._server.unregister_source(source_id)

            # Remove path tracking
            paths_to_remove = [
                path for path, sid in self._path_to_source_id.items()
                if sid == source_id
            ]
            for path in paths_to_remove:
                self._path_to_source_id.pop(path, None)

            logger.info(f"Unregistered source from server: {source_id}")
        except Exception as e:
            logger.error(
                f"Failed to unregister source {source_id}: {e}",
                exc_info=True,
            )


def create_source_manager(
    server: TensorFlightServer,
    registry: AdapterRegistry,
    watcher: Optional[DirectoryWatcher],
    monitored_sources: Optional[List[SourceConfig]] = None,
    static_sources: Optional[List[SourceConfig]] = None,
) -> Optional[SourceManager]:
    """Create a SourceManager for all configured sources.

    Handles both static sources (explicit config, registered once) and
    monitored sources (filesystem-discovered, kept live via watcher).
    Both paths use the same DiscoveryState/callback machinery.

    Args:
        server: TensorFlightServer instance
        registry: AdapterRegistry for adapter creation
        watcher: DirectoryWatcher for filesystem events (None for static-only)
        monitored_sources: SourceConfig entries with monitor=True
        static_sources: Explicit SourceConfig entries (monitor=False)

    Returns:
        SourceManager if there are any sources, None otherwise
    """
    monitored_sources = monitored_sources or []
    static_sources = static_sources or []

    if not monitored_sources and not static_sources:
        return None

    # Extract monitored directories
    monitored_dirs: Set[Path] = set()
    for source in monitored_sources:
        if source.is_remote:
            logger.warning(f"Cannot monitor remote URL: {source.url}")
            continue

        local_path = source.local_path
        if local_path is None or not local_path.exists():
            logger.warning(f"Cannot monitor non-existent path: {source.url}")
            continue

        if local_path.is_file():
            logger.warning(f"Cannot monitor single file: {source.url}")
            continue

        monitored_dirs.add(local_path)

    if not monitored_dirs and not static_sources:
        logger.warning("No valid sources to serve")
        return None

    # Create discovery state (empty - will be populated after SourceManager is created)
    discovery_state = DiscoveryState()

    # Create source manager FIRST (sets up callbacks on discovery_state)
    manager = SourceManager(
        server=server,
        registry=registry,
        discovery_state=discovery_state,
        watcher=watcher,
        monitored_dirs=monitored_dirs,
        dim_labels=monitored_sources[0].dim_labels if monitored_sources else None,
    )

    # Seed static sources as direct claims (explicit config, no filesystem walk)
    # These are added first so monitored discovery skips paths already claimed.
    for source in static_sources:
        claim = SourceClaim(
            source_type=source.type,
            primary_path=Path(source.url),
            claimed_paths={Path(source.url)},
            source_id=source.source_id,
            dim_labels=source.dim_labels,
            extra_config={'dataset': source.dataset} if source.dataset else {},
        )
        discovery_state.add_claim(claim)  # fires _on_source_added callback

    # Filesystem discovery for monitored sources (callbacks fire here too)
    for source in monitored_sources:
        if source.is_remote:
            continue

        local_path = source.local_path
        if local_path is None:
            continue

        state = discover_sources(
            local_path,
            registry,
            dim_labels=source.dim_labels,
        )

        for claim in state.get_all_claims():
            discovery_state.add_claim(claim)

    return manager