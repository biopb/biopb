"""Source lifecycle manager for the periodic catalog rescan runtime.

Coordinates the periodic rescan watcher, discovery state, server catalog updates,
and metadata database synchronization for monitored local directories.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from biopb_tensor_server.config import SourceConfig
from biopb_tensor_server.discovery import (
    AdapterRegistry,
    DiscoveryState,
    SourceClaim,
    discover_sources,
    get_file_identity,
    is_remote_url,
    should_skip_walk_entry,
)
from biopb_tensor_server.watcher import (
    DirectoryWatcher,
    WatcherEvent,
    WatcherEventType,
)

if TYPE_CHECKING:
    from biopb_tensor_server.server import TensorFlightServer

logger = logging.getLogger(__name__)


@dataclass
class _FailureTracker:
    attempts: int = 0
    next_retry_at: float = 0.0
    last_log_at: float = float("-inf")


class SourceManager:
    """Manages periodic rescans and source lifecycle reconciliation."""

    _retry_backoff_initial = 1.0
    _retry_backoff_max = 60.0
    _failure_log_interval = 30.0

    def __init__(
        self,
        server: TensorFlightServer,
        registry: AdapterRegistry,
        discovery_state: DiscoveryState,
        watcher: Optional[DirectoryWatcher],
        monitored_dirs: Set[Path],
        dim_labels: Optional[List[str]] = None,
        credentials_config: Optional[Any] = None,
        stability_window: float = 30.0,
        probe_open_files: bool = True,
        full_rescan_interval: float = 3600.0,
        stable_rescans_required: int = 0,
        aggressive_dir_pruning: bool = False,
    ):
        self._server = server
        self._registry = registry
        self._state = discovery_state
        self._watcher = watcher
        self._monitored_dirs = monitored_dirs
        self._dim_labels = dim_labels
        self._credentials_config = credentials_config
        self._stability_window = stability_window
        self._probe_open_files = probe_open_files
        self._full_rescan_interval = full_rescan_interval
        self._stable_rescans_required = max(0, stable_rescans_required)
        self._aggressive_dir_pruning = aggressive_dir_pruning

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Background precache hook: called with a source_id when a source is
        # committed *after* start() (the runtime phase). Left None for the
        # initial startup scan -- which is committed before start() -- so the
        # precache worker only warms sources that arrive live. See start().
        self._on_source_committed: Optional[Callable[[str], None]] = None
        self._runtime_phase = False
        # Rescan/reconcile helpers may re-enter other state-mutating helpers.
        # Use an RLock so nested calls on the same thread do not deadlock.
        self._lock = threading.RLock()

        # Path tracking for move handling
        # Maps resolved path -> source_id (str keys for URL support)
        self._path_to_source_id: Dict[str, str] = {}
        # Cached filesystem signatures for stability and change detection.
        # path -> (is_directory, signature_tuple, last_changed_epoch)
        self._entry_state: Dict[str, Tuple[bool, Tuple[Any, ...], float]] = {}
        # path -> consecutive unchanged rescans observed since the last change.
        self._entry_stable_observations: Dict[str, int] = {}
        # path -> whether the current signature still needs one eligible discovery pass.
        self._entry_pending_scan: Dict[str, bool] = {}
        # source_id -> member path signature map used to detect in-place changes.
        self._source_signatures: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
        # source_id -> retry/logging state for repeatedly failing datasets.
        self._failed_sources: Dict[str, _FailureTracker] = {}
        # Rescan bookkeeping for low-overhead subtree pruning.
        self._skipped_stable_dirs: Set[str] = set()
        self._last_full_rescan_at: float = float("-inf")

        # Initialize path tracking from existing claims
        for source_id, claim in self._state.claims.items():
            self._path_to_source_id[claim.primary_path] = source_id

        self._state.on_source_added = None
        self._state.on_source_removed = None

    def start(self) -> None:
        """Start the event processing loop."""
        if self._watcher is None:
            return  # Static-only mode; no filesystem events to process

        if self._thread is not None and self._thread.is_alive():
            logger.warning("SourceManager already running")
            return

        # Everything committed from here on is a live (runtime) addition; the
        # initial startup scan ran before this point and is excluded from the
        # precache hook.
        self._runtime_phase = True
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="SourceManager-EventLoop",
        )
        self._thread.start()
        logger.info("SourceManager started")

    def iter_local_source_mtimes(self) -> List[Tuple[str, float]]:
        """Return ``(source_id, mtime)`` for every currently-registered *local*
        source, for seeding the precache backlog (newest first).

        Remote sources are skipped (no ``os.stat`` mtime), as are any whose path
        can't be stat-ed (e.g. removed between commit and this call).
        """
        # Snapshot under the same lock that guards claim mutations: the watcher's
        # event-loop thread adds/removes claims via _commit_add_claim /
        # _commit_remove_claim, so iterating self._state.claims unlocked could
        # race a concurrent rescan ("dict changed size during iteration"). Copy
        # the few fields we need under the lock, then stat() outside it (I/O).
        with self._lock:
            snapshot = [
                (claim.source_id, claim.primary_path)
                for claim in self._state.claims.values()
                if not claim.is_remote
            ]
        out: List[Tuple[str, float]] = []
        for source_id, primary_path in snapshot:
            try:
                mtime = os.stat(primary_path).st_mtime
            except OSError:
                continue
            out.append((source_id, mtime))
        return out

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
        """Process periodic rescan triggers from the watcher."""
        while self._running:
            try:
                events = self._watcher.get_events(timeout=0.5)

                if events:
                    for event in events:
                        self._process_event(event)

            except Exception as e:
                logger.error(f"Error processing events: {e}", exc_info=True)

            # Small sleep to prevent busy polling
            time.sleep(0.1)

    def _process_event(self, event: WatcherEvent) -> None:
        """Handle a watcher event."""
        try:
            logger.debug(f"Processing event: {event.event_type.value} {event.path}")
            if event.event_type == WatcherEventType.RESCAN:
                self._handle_rescan()
            else:
                logger.debug(
                    "Ignoring unsupported watcher event type: %s",
                    event.event_type.value,
                )
        except Exception as e:
            logger.error(f"Error handling event {event}: {e}", exc_info=True)

    def _handle_rescan(self) -> None:
        """Run one periodic rescan across monitored directories."""
        if not self._monitored_dirs:
            return

        self._cleanup_deleted_monitored_dirs()
        if not self._monitored_dirs:
            return

        force_full_rescan = self._should_force_full_rescan()
        (
            next_state,
            next_stable_observations,
            next_pending_scan,
            skipped_dirs,
        ) = self._refresh_entry_state(force_full=force_full_rescan, publish=False)
        previous_state = self._entry_state
        previous_stable_observations = self._entry_stable_observations
        previous_pending_scan = self._entry_pending_scan
        previous_skipped_dirs = self._skipped_stable_dirs

        self._entry_state = next_state
        self._entry_stable_observations = next_stable_observations
        self._entry_pending_scan = next_pending_scan
        self._skipped_stable_dirs = skipped_dirs

        rescan_succeeded = False
        try:
            discovered_state = DiscoveryState()
            for monitored_dir in sorted(self._monitored_dirs):
                resolved_dir = monitored_dir.resolve()
                if str(resolved_dir) in skipped_dirs:
                    continue
                discover_sources(
                    monitored_dir,
                    self._registry,
                    state=discovered_state,
                    dim_labels=self._dim_labels,
                    path_filter=self._should_scan_path,
                )

            self._preserve_skipped_claims(discovered_state, skipped_dirs)

            unstable_paths = self._get_unstable_paths()
            self._reconcile_discovered_state(discovered_state, unstable_paths)
            rescan_succeeded = True
        finally:
            if not rescan_succeeded:
                self._entry_state = previous_state
                self._entry_stable_observations = previous_stable_observations
                self._entry_pending_scan = previous_pending_scan
                self._skipped_stable_dirs = previous_skipped_dirs

        if force_full_rescan and rescan_succeeded:
            self._last_full_rescan_at = time.time()

    def _cleanup_deleted_monitored_dirs(self) -> None:
        """Remove claims for monitored roots that no longer exist."""
        deleted_dirs = []
        for monitored_dir in sorted(self._monitored_dirs):
            try:
                exists = monitored_dir.exists()
            except OSError:
                exists = False
            if not exists:
                deleted_dirs.append(monitored_dir)

        for deleted_dir in deleted_dirs:
            self._cleanup_deleted_monitored_dir(deleted_dir)

    def _cleanup_deleted_monitored_dir(self, deleted_dir: Path) -> None:
        """Remove sources and cache state for a monitored root that disappeared."""
        removed_source_ids = []
        deleted_root = deleted_dir.resolve(strict=False)

        with self._lock:
            current_claims = list(self._state.claims.items())

        for source_id, claim in current_claims:
            if is_remote_url(claim.primary_path):
                continue
            try:
                claim_path = Path(claim.primary_path).resolve(strict=False)
            except OSError:
                continue
            if claim_path == deleted_root or claim_path.is_relative_to(deleted_root):
                if self._commit_remove_source(source_id):
                    removed_source_ids.append(source_id)

        deleted_root_str = str(deleted_root)
        self._monitored_dirs.discard(deleted_dir)
        self._skipped_stable_dirs.discard(deleted_root_str)

        entry_paths_to_remove = [
            path_str
            for path_str in self._entry_state
            if path_str == deleted_root_str
            or Path(path_str).is_relative_to(deleted_root)
        ]
        for path_str in entry_paths_to_remove:
            self._entry_state.pop(path_str, None)
            self._entry_pending_scan.pop(path_str, None)

        if removed_source_ids:
            logger.warning(
                "Removed %d sources after monitored directory disappeared: %s",
                len(removed_source_ids),
                deleted_dir,
            )
        else:
            logger.warning(
                "Stopped monitoring deleted directory with no active sources: %s",
                deleted_dir,
            )

    def _refresh_entry_state(
        self,
        force_full: bool = False,
        publish: bool = True,
    ) -> Tuple[
        Dict[str, Tuple[bool, Tuple[Any, ...], float]],
        Dict[str, int],
        Dict[str, bool],
        Set[str],
    ]:
        """Refresh cached filesystem signatures for all monitored trees."""
        now = time.time()
        next_state: Dict[str, Tuple[bool, Tuple[Any, ...], float]] = {}
        next_stable_observations: Dict[str, int] = {}
        next_pending_scan: Dict[str, bool] = {}
        skipped_dirs: Set[str] = set()
        # One identity set across all monitored roots for this refresh: breaks
        # directory loops (symlink, Windows junction, hardlink, bind mount) and
        # also stops overlapping roots from walking a shared subtree twice.
        visited_identities: Set[str] = set()
        for monitored_dir in sorted(self._monitored_dirs):
            # An explicitly-configured root is honored unconditionally — even if
            # it is a symlink, hidden, or named like a pruned system dir; the
            # skips only ever apply to entries found *inside* it. That contract is
            # the sum of two pieces here, both load-bearing:
            #   * resolve() — the root reaches _scan_tree_state already
            #     dereferenced, so its is_symlink reads False and the symlink/loop
            #     guard there does not reject a symlinked root (the guard rejects
            #     symlinks by the entry's *own* pre-resolution flag; resolving the
            #     root up front is precisely what exempts it);
            #   * is_root=True — tells _scan_tree_state to skip the
            #     hidden/system/offline policy for this entry only.
            # Change either and a symlinked or oddly-named monitored root silently
            # stops being scanned.
            self._scan_tree_state(
                monitored_dir.resolve(),
                now,
                next_state,
                next_stable_observations,
                next_pending_scan,
                skipped_dirs,
                force_full,
                self._aggressive_dir_pruning,
                visited_identities,
                is_root=True,
            )
        if publish:
            self._entry_state = next_state
            self._entry_stable_observations = next_stable_observations
            self._entry_pending_scan = next_pending_scan
            self._skipped_stable_dirs = skipped_dirs
        return (
            next_state,
            next_stable_observations,
            next_pending_scan,
            skipped_dirs,
        )

    def _scan_tree_state(
        self,
        path: Path,
        now: float,
        next_state: Dict[str, Tuple[bool, Tuple[Any, ...], float]],
        next_stable_observations: Dict[str, int],
        next_pending_scan: Dict[str, bool],
        skipped_dirs: Set[str],
        force_full: bool,
        allow_prune: bool,
        visited_identities: Set[str],
        is_root: bool = False,
    ) -> None:
        """Capture the current filesystem signature state for one subtree."""
        try:
            # Symlink-ness of the *entry itself*, read before resolving — once
            # resolved the path is the canonical target and is never a symlink,
            # so checking the resolved path (the old bug) never fired.
            is_symlink = path.is_symlink()
            resolved_path = path.resolve(strict=False)
            stat_result = resolved_path.stat()
            is_directory = resolved_path.is_dir()
        except OSError:
            return

        # Shared skip policy (hidden / system / cloud-placeholder), identical to
        # the claim walk's. Applied to entries found while walking, never to an
        # explicitly-configured root.
        if not is_root and should_skip_walk_entry(path, is_directory):
            return

        path_str = str(resolved_path)
        signature = self._build_entry_signature(stat_result, is_directory)
        previous_entry = self._entry_state.get(path_str)
        last_changed = self._get_entry_change_time(stat_result, now)
        stable_observations = 0
        if previous_entry is not None and previous_entry[:2] == (
            is_directory,
            signature,
        ):
            last_changed = previous_entry[2]
            stable_observations = (
                self._entry_stable_observations.get(path_str, 0) + 1
            )
            pending_scan = self._entry_pending_scan.get(path_str, False)
        else:
            pending_scan = True
        next_state[path_str] = (is_directory, signature, last_changed)
        next_stable_observations[path_str] = stable_observations
        next_pending_scan[path_str] = pending_scan

        # Only real directories are walked further. Never follow a symlinked
        # directory; and break every *other* kind of loop — Windows junction,
        # hardlink, bind mount, none of which present as a symlink — by
        # filesystem identity, since the symlink flag alone is not enough.
        # (A configured root passes this only because _refresh_entry_state hands
        # it in pre-resolved, so is_symlink is False — see the call site.)
        if not is_directory or is_symlink:
            return
        identity = get_file_identity(resolved_path)
        if identity in visited_identities:
            return
        visited_identities.add(identity)

        if (
            allow_prune
            and
            not force_full
            and previous_entry is not None
            and previous_entry[:2] == (is_directory, signature)
            and not pending_scan
            and now - previous_entry[2] >= self._stability_window
            and stable_observations >= self._stable_rescans_required
        ):
            skipped_dirs.add(path_str)
            self._copy_cached_subtree_entries(
                path_str,
                next_state,
                next_stable_observations,
                next_pending_scan,
            )
            return

        try:
            for child in resolved_path.iterdir():
                self._scan_tree_state(
                    child,
                    now,
                    next_state,
                    next_stable_observations,
                    next_pending_scan,
                    skipped_dirs,
                    force_full,
                    True,
                    visited_identities,
                )
        except OSError:
            return

    def _copy_cached_subtree_entries(
        self,
        root_path_str: str,
        next_state: Dict[str, Tuple[bool, Tuple[Any, ...], float]],
        next_stable_observations: Dict[str, int],
        next_pending_scan: Dict[str, bool],
    ) -> None:
        """Carry forward cached descendants when a stable subtree is skipped."""
        root_path = Path(root_path_str)
        for cached_path, entry in self._entry_state.items():
            if cached_path == root_path_str or cached_path in next_state:
                continue
            try:
                if Path(cached_path).is_relative_to(root_path):
                    next_state[cached_path] = entry
                    next_stable_observations[cached_path] = (
                        self._entry_stable_observations.get(cached_path, 0)
                    )
                    next_pending_scan[cached_path] = self._entry_pending_scan.get(
                        cached_path,
                        False,
                    )
            except OSError:
                continue

    def _should_force_full_rescan(self) -> bool:
        """Return True when a full tree walk should bypass subtree pruning."""
        if self._full_rescan_interval <= 0:
            return False
        return time.time() - self._last_full_rescan_at >= self._full_rescan_interval

    def _build_entry_signature(
        self,
        stat_result: Any,
        is_directory: bool,
    ) -> Tuple[Any, ...]:
        """Build a stable signature tuple for a file or directory."""
        if is_directory:
            return (
                stat_result.st_dev,
                stat_result.st_ino,
                stat_result.st_mtime_ns,
                stat_result.st_ctime_ns,
            )
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    def _get_entry_change_time(self, stat_result: Any, now: float) -> float:
        """Return the best available filesystem change timestamp for an entry."""
        mtime_ns = getattr(stat_result, "st_mtime_ns", None)
        ctime_ns = getattr(stat_result, "st_ctime_ns", None)
        if mtime_ns is not None and ctime_ns is not None:
            return min(now, max(mtime_ns, ctime_ns) / 1_000_000_000)

        mtime = getattr(stat_result, "st_mtime", None)
        ctime = getattr(stat_result, "st_ctime", None)
        if mtime is not None and ctime is not None:
            return min(now, max(float(mtime), float(ctime)))

        return now

    def _should_scan_path(self, path: Path) -> bool:
        """Return True when a path is stable enough to participate in discovery."""
        try:
            resolved_path = path.resolve(strict=False)
        except OSError:
            return False

        if resolved_path.name.startswith("."):
            return False

        if str(resolved_path) in self._skipped_stable_dirs:
            return False

        entry = self._entry_state.get(str(resolved_path))
        if entry is None:
            return False

        age = time.time() - entry[2]
        if age < self._stability_window:
            return False

        stable_observations = self._entry_stable_observations.get(
            str(resolved_path),
            self._stable_rescans_required,
        )
        if stable_observations < self._stable_rescans_required:
            return False

        if not entry[0] and self._probe_open_files:
            if not self._can_open_for_append(resolved_path):
                return False

        self._entry_pending_scan[str(resolved_path)] = False
        return True

    def _can_open_for_append(self, path: Path) -> bool:
        """Best-effort probe that a file is not obviously blocked for append.

        This is not a reliable active-writer detector on POSIX filesystems.
        Stability gating still primarily relies on signature age and repeated
        unchanged rescans rather than this probe alone.
        """
        try:
            with open(path, "a"):
                return True
        except OSError:
            return False

    def _get_unstable_paths(self) -> List[Path]:
        """Return unstable files/directories observed in the latest refresh."""
        now = time.time()
        unstable = []
        for path_str, (_, _, last_changed) in self._entry_state.items():
            if now - last_changed < self._stability_window:
                unstable.append(Path(path_str))
        return unstable

    def _reconcile_discovered_state(
        self,
        discovered_state: DiscoveryState,
        unstable_paths: List[Path],
    ) -> None:
        """Apply add/remove/update diffs between the current and discovered states."""
        with self._lock:
            current_claims = {
                source_id: claim
                for source_id, claim in self._state.claims.items()
                if self._is_monitored_claim(claim)
            }

        discovered_claims = discovered_state.claims
        current_ids = set(current_claims)
        discovered_ids = set(discovered_claims)

        changed_ids: Set[str] = set()
        for source_id in current_ids & discovered_ids:
            new_signatures = self._build_claim_signatures(discovered_claims[source_id])
            existing_signatures = self._source_signatures.get(source_id)
            if existing_signatures is None:
                self._source_signatures[source_id] = new_signatures
                continue
            if existing_signatures != new_signatures:
                changed_ids.add(source_id)

        removed_ids = [
            source_id
            for source_id in sorted(current_ids - discovered_ids | changed_ids)
            if not self._claim_overlaps_unstable(
                current_claims[source_id], unstable_paths
            )
        ]
        added_claims = [
            discovered_claims[source_id]
            for source_id in sorted((discovered_ids - current_ids) | changed_ids)
            if self._should_retry_source(source_id)
        ]

        for source_id in removed_ids:
            self._commit_remove_source(source_id)
        for claim in added_claims:
            self._commit_add_claim(claim)

    def _is_monitored_claim(self, claim: SourceClaim) -> bool:
        """Check if a claim belongs to one of the monitored local roots."""
        if is_remote_url(claim.primary_path):
            return False

        try:
            claim_path = Path(claim.primary_path).resolve(strict=False)
        except OSError:
            return False

        return any(
            claim_path.is_relative_to(monitored_dir)
            for monitored_dir in self._monitored_dirs
        )

    def _claim_overlaps_unstable(
        self,
        claim: SourceClaim,
        unstable_paths: List[Path],
    ) -> bool:
        """Check if any claimed member path falls in an unstable area."""
        for member_path in claim.member_paths:
            try:
                resolved_member = Path(member_path).resolve(strict=False)
            except OSError:
                continue

            for unstable_path in unstable_paths:
                if resolved_member == unstable_path:
                    return True
                if unstable_path.is_dir() and resolved_member.is_relative_to(
                    unstable_path
                ):
                    return True
        return False

    def _build_claim_signatures(
        self,
        claim: SourceClaim,
    ) -> Dict[str, Tuple[Any, ...]]:
        """Collect cached member-path signatures for a claim."""
        signatures: Dict[str, Tuple[Any, ...]] = {}
        for member_path in sorted(claim.member_paths):
            entry = self._entry_state.get(member_path)
            if entry is not None:
                signatures[member_path] = entry[1]
                continue

            try:
                resolved_path = Path(member_path).resolve(strict=False)
                stat_result = resolved_path.stat()
            except OSError:
                continue

            signatures[member_path] = self._build_entry_signature(
                stat_result,
                resolved_path.is_dir(),
            )
        return signatures

    def _preserve_skipped_claims(
        self,
        discovered_state: DiscoveryState,
        skipped_dirs: Set[str],
    ) -> None:
        """Carry forward claims whose subtree was intentionally skipped this cycle."""
        if not skipped_dirs:
            return

        skipped_paths = [Path(path_str) for path_str in sorted(skipped_dirs)]
        with self._lock:
            current_claims = list(self._state.claims.values())

        for claim in current_claims:
            if not self._is_monitored_claim(claim):
                continue
            if claim.source_id in discovered_state.claims:
                continue
            if self._claim_overlaps_skipped_subtree(claim, skipped_paths):
                discovered_state.add_claim(claim, notify=False)

    def _claim_overlaps_skipped_subtree(
        self,
        claim: SourceClaim,
        skipped_dirs: List[Path],
    ) -> bool:
        """Return True when a claim lives in a subtree skipped as unchanged."""
        claim_paths = {claim.primary_path, *claim.member_paths}
        for claim_path_str in claim_paths:
            if is_remote_url(claim_path_str):
                continue
            try:
                claim_path = Path(claim_path_str).resolve(strict=False)
            except OSError:
                continue
            for skipped_dir in skipped_dirs:
                if claim_path == skipped_dir or claim_path.is_relative_to(skipped_dir):
                    return True
        return False

    def _commit_add_claim(self, claim: SourceClaim) -> bool:
        """Register a discovered source, then commit it into confirmed state."""
        if not self._register_source_claim(claim):
            self._record_failed_source_attempt(claim.source_id)
            return False

        with self._lock:
            added = self._state.add_claim(claim, notify=False)
            if not added:
                self._rollback_source_registration(claim.source_id)
                self._record_failed_source_attempt(claim.source_id)
                return False
            self._source_signatures[claim.source_id] = self._build_claim_signatures(
                claim
            )
            self._clear_failed_source_attempt(claim.source_id)

        # Notify the precache worker of live additions only (runtime phase).
        # Best-effort: a hook failure must never abort a source commit.
        if self._runtime_phase and self._on_source_committed is not None:
            try:
                self._on_source_committed(claim.source_id)
            except Exception:
                logger.exception(
                    "precache on_source_committed hook failed for %s",
                    claim.source_id,
                )
        return True

    def _commit_remove_source(self, source_id: str) -> bool:
        """Unregister a confirmed source and then drop it from state."""
        with self._lock:
            claim = self._state.claims.get(source_id)
        if claim is None:
            return False

        if not self._unregister_source_claim(source_id):
            return False

        with self._lock:
            self._state.remove_claim(claim.primary_path, notify=False)
            self._source_signatures.pop(source_id, None)
            self._clear_failed_source_attempt(source_id)
        return True

    def _should_retry_source(self, source_id: str) -> bool:
        """Return True when a failed source is eligible for another add attempt."""
        tracker = self._failed_sources.get(source_id)
        if tracker is None:
            return True
        return time.time() >= tracker.next_retry_at

    def _record_failed_source_attempt(self, source_id: Optional[str]) -> None:
        """Advance retry state for a failing source."""
        if not source_id:
            return

        now = time.time()
        tracker = self._failed_sources.get(source_id)
        if tracker is None:
            tracker = _FailureTracker()
            self._failed_sources[source_id] = tracker

        tracker.attempts += 1
        delay = min(
            self._retry_backoff_initial * (2 ** (tracker.attempts - 1)),
            self._retry_backoff_max,
        )
        tracker.next_retry_at = now + delay

    def _clear_failed_source_attempt(self, source_id: Optional[str]) -> None:
        """Drop retry state after a source reaches a clean steady state."""
        if not source_id:
            return
        self._failed_sources.pop(source_id, None)

    def _log_source_failure(
        self,
        source_id: Optional[str],
        message: str,
        *args: Any,
        exc_info: bool = False,
    ) -> None:
        """Log a source failure no more than once per rate-limit window."""
        if not source_id:
            logger.error(message, *args, exc_info=exc_info)
            return

        now = time.time()
        tracker = self._failed_sources.get(source_id)
        if tracker is None:
            tracker = _FailureTracker()
            self._failed_sources[source_id] = tracker

        if now - tracker.last_log_at >= self._failure_log_interval:
            logger.error(message, *args, exc_info=exc_info)
            tracker.last_log_at = now
            return

        logger.debug(
            "Suppressing repeated failure log for source %s until retry window expires",
            source_id,
        )

    def _register_source_claim(self, claim: SourceClaim) -> bool:
        """Create and register a source, rolling back on partial failure."""
        try:
            source_config = SourceConfig(
                type=claim.source_type,
                url=str(claim.primary_path),
                source_id=claim.source_id,
                dim_labels=claim.dim_labels,
                dataset=claim.extra_config.get("dataset"),
            )

            adapter_cls = self._registry.get_adapter_for_type(claim.source_type)
            if adapter_cls is None:
                self._log_source_failure(
                    claim.source_id,
                    "No adapter for type %s for source %s (%s)",
                    claim.source_type,
                    claim.source_id,
                    claim.primary_path,
                )
                return False

            adapter = adapter_cls.create_from_config(
                source_config, self._credentials_config
            )
        except Exception as e:
            self._log_source_failure(
                claim.source_id,
                "Failed to create adapter for source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
                exc_info=True,
            )
            return False

        registered = False
        try:
            self._server.register_source(claim.source_id, adapter)
            registered = True

            if (
                hasattr(self._server, "_metadata_db")
                and self._server._metadata_db is not None
            ):
                self._server._metadata_db.sync_source_added(claim.source_id, adapter)

            self._path_to_source_id[claim.primary_path] = claim.source_id
            logger.info(f"Registered source with server: {claim.source_id}")
            return True
        except Exception as e:
            self._log_source_failure(
                claim.source_id,
                "Failed to register/sync source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
                exc_info=True,
            )
            if registered:
                self._rollback_source_registration(claim.source_id)
            return False

    def _rollback_source_registration(self, source_id: str) -> None:
        """Best-effort rollback after a partial add failure."""
        try:
            self._server.unregister_source(source_id)
        except Exception:
            logger.error(
                "Rollback failed to unregister source %s", source_id, exc_info=True
            )

        try:
            if (
                hasattr(self._server, "_metadata_db")
                and self._server._metadata_db is not None
            ):
                self._server._metadata_db.sync_source_removed(source_id)
        except Exception:
            logger.error(
                "Rollback failed to remove source %s from metadata DB",
                source_id,
                exc_info=True,
            )

        paths_to_remove = [
            path for path, sid in self._path_to_source_id.items() if sid == source_id
        ]
        for path in paths_to_remove:
            self._path_to_source_id.pop(path, None)

    def _unregister_source_claim(self, source_id: str) -> bool:
        """Remove a source from the server and metadata DB."""
        try:
            self._server.unregister_source(source_id)
            if (
                hasattr(self._server, "_metadata_db")
                and self._server._metadata_db is not None
            ):
                self._server._metadata_db.sync_source_removed(source_id)

            paths_to_remove = [
                path
                for path, sid in self._path_to_source_id.items()
                if sid == source_id
            ]
            for path in paths_to_remove:
                self._path_to_source_id.pop(path, None)

            logger.info(f"Unregistered source from server: {source_id}")
            return True
        except Exception as e:
            logger.error(
                "Failed to unregister source %s: %s",
                source_id,
                e,
                exc_info=True,
            )
            return False


def create_source_manager(
    server: TensorFlightServer,
    registry: AdapterRegistry,
    watcher: Optional[DirectoryWatcher],
    monitored_sources: Optional[List[SourceConfig]] = None,
    static_sources: Optional[List[SourceConfig]] = None,
    credentials_config: Optional[Any] = None,
    stability_window: float = 30.0,
    probe_open_files: bool = True,
    full_rescan_interval: float = 3600.0,
    stable_rescans_required: int = 0,
    aggressive_dir_pruning: bool = False,
) -> Optional[SourceManager]:
    """Create a SourceManager for all configured sources.

    Handles both static sources (explicit config, registered once) and
    monitored sources (filesystem-discovered, kept live via watcher).
    Both paths use the same DiscoveryState/callback machinery.

    Remote sources:
    - Are NOT monitored (no filesystem events)
    - Are registered during initial discovery only
    - Use credentials_config for authentication

    Args:
        server: TensorFlightServer instance
        registry: AdapterRegistry for adapter creation
        watcher: DirectoryWatcher for filesystem events (None for static-only)
        monitored_sources: SourceConfig entries with monitor=True
        static_sources: Explicit SourceConfig entries (monitor=False)
        credentials_config: CredentialsConfig for remote storage authentication

    Returns:
        SourceManager if there are any sources, None otherwise
    """
    monitored_sources = monitored_sources or []
    static_sources = static_sources or []

    if not monitored_sources and not static_sources:
        return None

    # Extract monitored directories (skip remote URLs)
    monitored_dirs: Set[Path] = set()
    for source in monitored_sources:
        if source.is_remote:
            logger.info(f"Remote URL will not be monitored: {source.url}")
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
        credentials_config=credentials_config,
        stability_window=stability_window,
        probe_open_files=probe_open_files,
        full_rescan_interval=full_rescan_interval,
        stable_rescans_required=stable_rescans_required,
        aggressive_dir_pruning=aggressive_dir_pruning,
    )

    # Seed static sources as direct claims (explicit config, no filesystem walk)
    # These are added first so monitored discovery skips paths already claimed.
    for source in static_sources:
        claim = SourceClaim(
            source_type=source.type,
            primary_path=source.url,  # str for URL support
            source_id=source.source_id,
            dim_labels=source.dim_labels,
            extra_config={"dataset": source.dataset} if source.dataset else {},
        )
        manager._commit_add_claim(claim)

    # Bootstrap monitored discovery through the same rescan pipeline used at runtime.
    if monitored_dirs:
        manager._handle_rescan()

    return manager
