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
from stat import S_ISDIR
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from biopb_tensor_server.config import SourceConfig
from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter
from biopb_tensor_server.discovery import (
    AdapterRegistry,
    DiscoveryState,
    SourceClaim,
    _is_offline_placeholder,
    discover_sources_from_entries,
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
        cloud_roots: Optional[Set[Path]] = None,
    ):
        self._server = server
        self._registry = registry
        self._state = discovery_state
        self._watcher = watcher
        self._monitored_dirs = monitored_dirs
        # Resolved roots opted into cloud/synced-folder handling (config cloud=true).
        # Under these, dehydrated entries are admitted and registered as unresolved
        # sources that resolve lazily on first access (cloud-storage phase 2).
        self._cloud_roots: Set[Path] = cloud_roots or set()
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
            # Single traversal: the state walk above already visited every entry and
            # recorded its (resolved path, is_directory) into next_state in DFS
            # parent-first order. Drive the claim phase straight off that snapshot
            # instead of re-walking the filesystem a second time — the duplicate walk
            # was ~96% of the post-#61 rescan syscalls (biopb/biopb#56, item 4).
            # skipped_dirs prunes the stable subtrees the state walk carried forward
            # (their claims are preserved below), exactly as the old per-dir
            # `if ... in skipped_dirs: continue` did for whole roots.
            discovered_state = discover_sources_from_entries(
                (
                    (path_str, entry[0], entry[1])
                    for path_str, entry in next_state.items()
                ),
                self._registry,
                dim_labels=self._dim_labels,
                path_filter=self._should_scan_resolved,
                skipped_dirs=skipped_dirs,
                cloud_filter=self._is_under_cloud_root,
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
                cloud=monitored_dir.resolve() in self._cloud_roots,
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
        dir_entry: Optional[os.DirEntry] = None,
        cloud: bool = False,
    ) -> None:
        """Capture the current filesystem signature state for one subtree.

        ``dir_entry`` is the ``os.DirEntry`` the parent's ``os.scandir`` produced for
        this entry (None for an explicitly-configured root, which the caller hands in
        already resolved). Reusing it is the heart of the #56 per-entry syscall cut:
        its ``d_type`` answers is-symlink without a syscall, its cached ``stat()`` is
        the single stat we take, and a non-symlink child of an already-resolved
        directory is canonical by construction — so we never re-``resolve()`` it.
        Only the rare symlink entry pays a ``resolve()``, to preserve the
        resolved-target keying that ``next_state`` and the loop guard depend on.
        """
        try:
            if dir_entry is not None:
                # is_symlink reads the entry's own d_type (the symlink-ness of the
                # entry itself, never the target); stat() follows the symlink, so it
                # matches the old resolved_path.stat() for the signature/identity.
                is_symlink = dir_entry.is_symlink()
                stat_result = dir_entry.stat()
                # On Windows, DirEntry.stat() always reports st_ino/st_dev/st_nlink
                # as zero (they are expensive to compute), so the cheap cached stat
                # is missing the very fields the entry signature and file identity
                # key on. The claim walk takes a real os.stat (real inode on NTFS),
                # so a zeroed signature here would never match and every rescan would
                # spuriously cycle the source (biopb/biopb#56). Pay one real stat to
                # backfill the identity fields; POSIX DirEntry.stat() already does a
                # real stat, so this only fires on Windows and the syscall-cut win is
                # untouched.
                if stat_result.st_ino == 0:
                    stat_result = os.stat(dir_entry.path)
            else:
                # Root: _refresh_entry_state hands it in pre-resolved, so it is
                # canonical and not itself a symlink.
                is_symlink = False
                stat_result = path.stat()
        except OSError:
            return

        is_directory = S_ISDIR(stat_result.st_mode)
        # Resolve only symlinks. A non-symlink entry under an already-resolved parent
        # is its own canonical path, so re-resolving it (the old per-entry cost, an
        # O(depth) readlink walk) bought nothing (biopb/biopb#56).
        resolved_path = path.resolve(strict=False) if is_symlink else path

        # Shared skip policy (hidden / system / cloud-placeholder), identical to the
        # claim walk's. Pass the stat we already took so the offline-placeholder
        # check does not stat the entry a second time. Never applied to a root.
        # Under a `cloud`-opted root, ``admit_nonresident`` keeps dehydrated files
        # (they are admitted as unresolved sources instead of skipped).
        if not is_root and should_skip_walk_entry(
            path, is_directory, stat_result, admit_nonresident=cloud
        ):
            return

        path_str = str(resolved_path)
        signature = self._build_entry_signature(stat_result, is_directory, cloud=cloud)
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
        # resolved_path is canonical and we already hold its stat — hand both to
        # get_file_identity so it reuses (st_dev, st_ino) instead of re-resolving
        # and re-stat'ing (biopb/biopb#56).
        identity = get_file_identity(resolved_path, stat_result)
        if identity in visited_identities:
            return
        visited_identities.add(identity)

        if (
            allow_prune
            and
            not force_full
            # Never prune a cloud subtree on its mtime signature: cloud mtime is
            # unreliable (doc S1.2), so a "stable" signature is not trustworthy and
            # pruning could permanently hide a newly-archived dataset. A cloud root
            # is always fully walked (cloud-storage phase 2).
            and not cloud
            and previous_entry is not None
            and previous_entry[:2] == (is_directory, signature)
            and not pending_scan
            and now - previous_entry[2] >= self._stability_window
            and stable_observations >= self._stable_rescans_required
            and not self._subtree_has_pending_scan(path_str)
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
            # os.scandir yields DirEntry objects carrying d_type and a cached stat,
            # so each child is processed with one stat and no resolve — the per-entry
            # win (biopb/biopb#56). entry.path is the child under the canonical
            # parent, hence itself canonical for the common non-symlink case.
            with os.scandir(resolved_path) as entries:
                for entry in entries:
                    self._scan_tree_state(
                        Path(entry.path),
                        now,
                        next_state,
                        next_stable_observations,
                        next_pending_scan,
                        skipped_dirs,
                        force_full,
                        True,
                        visited_identities,
                        dir_entry=entry,
                        cloud=cloud,
                    )
        except OSError:
            return

    def _subtree_has_pending_scan(self, root_path_str: str) -> bool:
        """True when any cached descendant still needs a discovery pass.

        A directory's own mtime/ctime signature is blind to writes deep in its
        subtree (appending to a file does not bump any ancestor's mtime), so the
        prune gate cannot rely on the signature alone. `pending_scan` is set on
        any new/changed entry and cleared only when it passes a discovery walk
        (age >= stability_window), so a still-settling or undiscovered descendant
        keeps the flag — which is the signal to keep descending instead of
        freezing the subtree out (biopb/biopb#53). The directory's own flag is
        already covered by the `not pending_scan` clause in the prune gate, so it
        is skipped here.
        """
        root_path = Path(root_path_str)
        for cached_path, pending in self._entry_pending_scan.items():
            if not pending or cached_path == root_path_str:
                continue
            try:
                if Path(cached_path).is_relative_to(root_path):
                    return True
            except OSError:
                continue
        return False

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
        cloud: bool = False,
    ) -> Tuple[Any, ...]:
        """Build a stable signature tuple for a file or directory.

        Under a ``cloud = true`` root the signature is **residency-invariant** --
        keyed on identity (``st_dev``, ``st_ino``) only. Hydrating a placeholder
        (a consented recall) bumps ``st_size``/``st_mtime_ns``/``st_ctime_ns``;
        including those would make the next rescan see the just-resolved source as
        "changed" and destructively remove+re-add it (and re-dehydration/eviction
        would flap it). Archived cloud data is stable and cloud mtime is
        untrustworthy anyway, so identity is the right key there. Non-cloud
        entries keep the full mtime/size-sensitive signature.
        """
        if cloud:
            return (stat_result.st_dev, stat_result.st_ino)
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
        return self._should_scan_resolved(str(resolved_path))

    def _should_scan_resolved(self, resolved_str: str) -> bool:
        """Stability gate for an entry whose resolved path string is already known.

        The snapshot-driven discovery (biopb/biopb#56 item 4) iterates ``next_state``
        keys, which ``_scan_tree_state`` already stored as resolved path strings, so
        the per-entry ``Path.resolve()`` ``_should_scan_path`` would otherwise repeat
        is pure waste. Same predicate and the same load-bearing
        ``_entry_pending_scan`` clear-on-pass side effect (the #53 subtree-pending
        prune gate depends on it) — only the redundant resolve is dropped.
        """
        if os.path.basename(resolved_str).startswith("."):
            return False

        if resolved_str in self._skipped_stable_dirs:
            return False

        entry = self._entry_state.get(resolved_str)
        if entry is None:
            return False

        # Cloud/synced-folder entries bypass the stability machinery entirely
        # (cloud-storage phase 2). Two reasons, both load-bearing:
        #   * the open-for-append probe below opens the file -- a whole-file recall
        #     on a dehydrated placeholder, which is exactly what cloud handling must
        #     avoid; and
        #   * the mtime/ctime age + stable-rescan gate is unreliable on cloud
        #     filesystems (doc S1.2), so a placeholder could never stabilize.
        # Archived dehydrated data is inherently stable (never mid-write), so admit
        # it immediately. The pending-scan clear side effect is preserved.
        if self._is_under_cloud_root(resolved_str):
            self._entry_pending_scan[resolved_str] = False
            return True

        age = time.time() - entry[2]
        if age < self._stability_window:
            return False

        stable_observations = self._entry_stable_observations.get(
            resolved_str,
            self._stable_rescans_required,
        )
        if stable_observations < self._stable_rescans_required:
            return False

        if not entry[0] and self._probe_open_files:
            if not self._can_open_for_append(Path(resolved_str)):
                return False

        self._entry_pending_scan[resolved_str] = False
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

            # The cached-entry path above already carries the cloud-invariant
            # signature (``_scan_tree_state`` built it with ``cloud=``). This
            # re-stat fallback has no cloud context, so recompute it from the
            # cloud roots so hydration/eviction does not flap a resolved source.
            signatures[member_path] = self._build_entry_signature(
                stat_result,
                resolved_path.is_dir(),
                cloud=self._is_under_cloud_root(member_path),
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

    def _is_under_cloud_root(self, path: str) -> bool:
        """True when *path* is a cloud-opted root or lives under one."""
        if not self._cloud_roots:
            return False
        try:
            resolved = Path(path).resolve()
        except OSError:
            return False
        for root in self._cloud_roots:
            if resolved == root or root in resolved.parents:
                return True
        return False

    def _claim_is_unresolved(self, claim: SourceClaim) -> bool:
        """Whether this claim must be registered as an unresolved cloud source.

        Two triggers, both meaning "do not open the content now":
        - the adapter's own ``claim.unresolved`` flag -- a reader adapter (OME-Zarr,
          MicroManager, OME-TIFF, DICOM) recognized the source structurally and
          deferred its sidecar/container read because it was non-resident; or
        - under a ``cloud`` root, the source's content is a non-resident
          placeholder. This catches the content-free *file* formats (NIfTI, CZI,
          single OME-TIFF, ...) whose ``claim()`` never reads bytes but whose
          ``create_from_config`` would hydrate the file to learn its shape.

        Cloud-gated so a normal local source is never marked unresolved (and the
        per-file placeholder stat -- which can false-positive on a resident tiny
        file on some filesystems -- is only consulted for cloud roots).
        """
        if claim.unresolved:
            return True
        if not self._is_under_cloud_root(claim.primary_path):
            return False
        # A directory source (zarr store, ...) is born resolved from its resident
        # sidecars; only a non-resident *content file* forces deferral. Guard on
        # is_file so a directory's st_blocks==0 (macOS APFS) is never a false hit.
        for member in claim.member_paths:
            if is_remote_url(member):
                continue
            member_path = Path(member)
            try:
                if member_path.is_file() and _is_offline_placeholder(member_path):
                    return True
            except OSError:
                continue
        return False

    def _on_source_resolved(self, source_id: str, adapter: Any) -> None:
        """Backfill the metadata DB when an unresolved cloud source resolves.

        ``sync_source_added`` is an INSERT OR REPLACE upsert, so re-syncing the
        now-resolved adapter overwrites the source's NULL shape/dtype row with the
        concrete descriptor. Persistence across restart is phase 3 (file-backed
        DB); here the backfill lives for the process lifetime.
        """
        if (
            hasattr(self._server, "_metadata_db")
            and self._server._metadata_db is not None
        ):
            try:
                self._server._metadata_db.sync_source_added(source_id, adapter)
            except Exception:
                logger.exception(
                    "metadata-DB backfill failed for resolved source %s", source_id
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

            if self._claim_is_unresolved(claim):
                # Cloud-storage phase 2: register a placeholder that resolves
                # lazily on first access (re-claim + create_from_config on the
                # hydrated path) instead of opening the source now.
                adapter = UnresolvedSourceAdapter(
                    source_config,
                    self._registry,
                    credentials_config=self._credentials_config,
                    on_resolved=self._on_source_resolved,
                    cloud_root=self._is_under_cloud_root(claim.primary_path),
                )
            else:
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

    # Resolved roots opted into cloud/synced-folder handling (config cloud=true),
    # across both monitored and static sources. Under a monitored cloud root the
    # walk admits dehydrated entries; for any cloud source the registration path
    # defers a non-resident dataset to lazy resolution (cloud-storage phase 2).
    cloud_roots: Set[Path] = set()
    for source in (*monitored_sources, *static_sources):
        if source.cloud and not source.is_remote:
            local_path = source.local_path
            if local_path is not None:
                cloud_roots.add(local_path)

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
        cloud_roots=cloud_roots,
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
            # A static source explicitly flagged cloud is always deferred: the
            # user said "don't open it eagerly". If it is in fact resident, the
            # first access still resolves it cheaply.
            unresolved=bool(source.cloud),
        )
        manager._commit_add_claim(claim)

    # Bootstrap monitored discovery through the same rescan pipeline used at runtime.
    if monitored_dirs:
        manager._handle_rescan()

    return manager
