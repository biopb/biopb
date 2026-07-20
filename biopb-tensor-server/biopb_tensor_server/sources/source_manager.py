"""Source lifecycle manager for the periodic catalog rescan runtime.

Coordinates the periodic rescan watcher, discovery state, server catalog updates,
and metadata database synchronization for monitored local directories.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from biopb_tensor_server.core.config import SourceConfig, _reroot_catalog_url
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    discover_sources,
    discover_sources_from_entries,
    generate_source_id,
    is_remote_url,
    resolve_local_path,
)
from biopb_tensor_server.sources.reconciler import Reconciler, is_under_cloud_root
from biopb_tensor_server.sources.tree_scanner import EntryState, TreeScanner
from biopb_tensor_server.sources.watcher import (
    DirectoryWatcher,
    WatcherEvent,
    WatcherEventType,
)

if TYPE_CHECKING:
    from biopb_tensor_server.core.metadata_db import MetadataDatabase
    from biopb_tensor_server.serving.server import TensorFlightServer

logger = logging.getLogger(__name__)


# Backoff ceiling for the tensor-server upstream re-list, in rescan ticks
# (biopb/biopb#178). A re-list runs every tick while an upstream is changing or
# failing; while it stays stable the spacing doubles up to this many ticks, so a
# fully-stable upstream settles to re-listing about once an hour at the default
# 30s rescan tick (instead of querying it every 30s forever).
_UPSTREAM_RELIST_MAX_TICKS = 120

# While add_local_source waits for the catalog lock (a rescan is mid-flight), it
# emits a heartbeat this often so the streamed action does not sit silent long
# enough to trip a proxy idle read timeout.
_ADD_SOURCE_ACQUIRE_HEARTBEAT = 5.0


# Virtual scheme stamped on the catalog ``source_url`` of a drag-dropped source.
# It marks the source's *origin* (a runtime drop, not config/discovery) and is the
# key a future "remove dropped source" feature authorizes on: the drop re-root
# guard below only stamps it when the drop is entirely new AND outside every
# monitored root, so its presence means "user-added and nothing will re-add it."
# It is a display-only scheme (like ``cache://``) — never touches ``source_id`` or
# the raw ``_source_url`` used for I/O. Keep in sync with the client tree builders
# that strip it: ``_get_path_parts`` (biopb-mcp ``tensor_browser/_widget.py``) and
# ``getPathParts`` (web ``SourceTree.tsx``).
DND_URL_PREFIX = "dnd://"


def _drop_catalog_url(
    dropped_root: str, primary_path: str, *, mark_dnd: bool = True
) -> str:
    """Catalog ``source_url`` that re-roots a drag-dropped source under the
    dropped item's basename, optionally prefixed with the ``dnd://`` origin scheme.

    A dropped file's real path (``/home/u/data/exp/a.tif``) would otherwise nest
    it deep inside the shared absolute-path tree; re-rooting it at the dropped
    item's basename makes each drop its own root instead (via the shared
    ``_reroot_catalog_url``). When ``mark_dnd`` is set, the ``dnd://`` prefix also
    marks it as removable drop-origin:

        drop /home/u/data/exp.zarr           -> "dnd://exp.zarr"        (own root)
        drop /home/u/data/exp/ (a folder) with
             .../exp/a.tif, .../exp/sub/b.tif -> "dnd://exp/a.tif",
                                                 "dnd://exp/sub/b.tif"

    ``mark_dnd`` is False for a drop that lands under a monitored root: it still
    gets a tidy display root, but no marker, because the periodic rescan will
    re-discover it (with its native url) — so it is not safely removable. The
    marker therefore means exactly "user-added and nothing will re-add it."

    Display-only (never ``source_id``, nor the raw ``_source_url`` the filesystem
    uses); the client tree builders strip the scheme for display. The
    configured-``alias`` re-root shares ``_reroot_catalog_url`` but is always
    scheme-less, so the two re-root paths stay distinguishable.
    """
    dropped_root = str(dropped_root).rstrip("/\\")
    base = os.path.basename(dropped_root) or dropped_root
    rerooted = _reroot_catalog_url(base, dropped_root, primary_path)
    return DND_URL_PREFIX + rerooted if mark_dnd else rerooted


class SourceManager:
    """Drives the periodic rescan and owns the filesystem-scan machinery.

    The confirmed-catalog write path (registration, the discovered/upstream diff,
    add/remove lifecycle, failure-retry state) lives in :class:`Reconciler`, which
    this manager constructs and delegates every catalog mutation to; see that
    class's module docstring for the seam between the two.
    """

    def __init__(
        self,
        server: TensorFlightServer,
        registry: AdapterRegistry,
        discovery_state: DiscoveryState,
        watcher: Optional[DirectoryWatcher],
        monitored_dirs: Set[Path],
        metadata_db: Optional[MetadataDatabase] = None,
        dim_labels: Optional[List[str]] = None,
        credentials_config: Optional[Any] = None,
        stability_window: float = 30.0,
        probe_open_files: bool = True,
        full_rescan_interval: float = 3600.0,
        stable_rescans_required: int = 0,
        aggressive_dir_pruning: bool = False,
        cloud_roots: Optional[Set[Path]] = None,
        monitored_upstreams: Optional[List[SourceConfig]] = None,
    ):
        self._server = server
        # Kept for the runtime add_local_source discovery walk; the confirmed-
        # catalog write path uses the Reconciler's own copy.
        self._registry = registry
        self._watcher = watcher
        self._monitored_dirs = monitored_dirs
        # Resolved roots opted into cloud/synced-folder handling (config cloud=true).
        # Under these, dehydrated entries are admitted and registered as unresolved
        # sources that resolve lazily on first access (cloud-storage phase 2).
        self._cloud_roots: Set[Path] = cloud_roots or set()
        # Monitored tensor-server (bare-host grpc://) upstreams: their catalog is
        # periodically re-listed and reconciled like a directory walk
        # (biopb/biopb#178). Single-source grpc://host/<id> entries are not here --
        # there is nothing to re-list for one fixed source.
        self._monitored_upstreams: List[SourceConfig] = monitored_upstreams or []
        # Per-upstream re-list cadence (keyed by url), counted in rescan ticks (no
        # wall-clock interval -- the rescan tick is the unit). A re-list runs every
        # tick by default; when it finds the source set UNCHANGED the spacing
        # doubles (a stable upstream is skipped for more ticks, up to
        # _UPSTREAM_RELIST_MAX_TICKS, so we are not querying it every tick forever),
        # and any change OR failure resets it to every tick -- so a new source / a
        # recovering upstream is picked up within ~one tick, not after the full
        # backoff. `countdown` ticks down to 0 (re-list due); `period` is the
        # current spacing in ticks.
        self._upstream_relist: Dict[str, Dict[str, int]] = {}
        self._upstream_max_period: int = _UPSTREAM_RELIST_MAX_TICKS
        # Upstreams (by url) whose last re-list failed -- a status signal (the fast
        # retry itself is driven by the period reset above).
        self._failed_upstreams: Set[str] = set()
        self._dim_labels = dim_labels
        self._stability_window = stability_window
        self._probe_open_files = probe_open_files
        self._full_rescan_interval = full_rescan_interval
        self._stable_rescans_required = max(0, stable_rescans_required)
        self._aggressive_dir_pruning = aggressive_dir_pruning
        # The filesystem signature walk (biopb/biopb#278 item B). A pure producer:
        # given the previous caches it returns a fresh ScanSnapshot; this manager
        # owns publishing / rollback / cloud partitioning of that snapshot.
        self._scanner = TreeScanner(
            stability_window=stability_window,
            stable_rescans_required=self._stable_rescans_required,
            aggressive_dir_pruning=aggressive_dir_pruning,
        )

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Background precache hook: called with a source_id when a source is
        # committed *after* the initial scan completes. Gated on
        # ``_initial_scan_done`` so the (possibly large) startup set is routed to
        # the slow precache *backlog* instead of the prompt enqueue -- only
        # sources discovered live (later rescans) warm promptly. See
        # ``_commit_add_claim``.
        self._on_source_committed: Optional[Callable[[str], None]] = None
        # Flipped True at the end of the first successful full rescan. Under
        # progressive discovery that scan runs in the event loop *after* start(),
        # so this -- not "are we past start()" -- is the correct startup/runtime
        # boundary for the precache gate.
        self._initial_scan_done = False
        # Set only for the duration of the boot-tick upstream re-list, when the
        # local walk earlier in the *same* rescan already flipped
        # ``_initial_scan_done`` True (see ``_handle_rescan``). It keeps the
        # startup upstream mirror -- committed after that flip -- routed to the
        # slow backlog instead of the prompt enqueue, exactly as the startup set
        # is meant to be (the mirror is part of the startup catalog regardless of
        # which half of the tick registers it). Event-loop thread only, so a plain
        # flag is safe.
        self._suppress_live_precache = False
        # Best-effort callback fired once, when the first full scan completes
        # (from the event-loop thread). The launcher uses it to seed the precache
        # backlog with the startup set at the moment the catalog is established.
        self._on_initial_scan_complete: Optional[Callable[[], None]] = None

        # Coarse mutex serializing a *whole* catalog-mutation pass. The periodic
        # rescan (event-loop thread) and a runtime ``add_local_source`` (a Flight
        # handler thread) are the only two flows that discover + commit sources;
        # holding this across each keeps the confirmed catalog single-writer
        # without threading add_source through the event loop. Distinct from the
        # Reconciler's fine-grained state RLock that the commit primitives take.
        self._catalog_lock = threading.Lock()

        # Cached filesystem signatures for stability and change detection.
        # path -> EntryState (signature + stability counter + pending-scan flag).
        self._entry_states: Dict[str, EntryState] = {}
        # Rescan bookkeeping for low-overhead subtree pruning.
        self._skipped_stable_dirs: Set[str] = set()
        self._last_full_rescan_at: float = float("-inf")
        # Cloud-subtree entry partition. Cloud (synced-folder) subtrees are walked
        # only on the hourly force_full pass; on the frequent incremental rescans
        # they are skipped entirely. To keep that O(non-cloud), cloud entries live
        # here -- rebuilt only at the end of a successful force_full -- instead of
        # being re-materialized into ``_entry_states``/``next_state`` every cycle
        # (which made every per-entry rescan loop O(whole cloud catalog) and
        # stalled the Flight serving threads via the GIL). Same ``EntryState``
        # shape as ``_entry_states`` (the stability fields inert -- see EntryState).
        # Never mutated on a failed rescan, so the last good snapshot survives a
        # force_full failure (no rollback variable needed).
        self._cloud_entry_states: Dict[str, EntryState] = {}

        # The confirmed-catalog writer (biopb/biopb#278 item B). Owns the live
        # claim set + registration + the discovered/upstream diff; this manager
        # feeds it scan results and delegates every catalog mutation to it. The
        # injected seams (see Reconciler's docstring): ``entry_for`` reads this
        # manager's scan-signature caches, ``notify_source_committed`` applies the
        # precache routing gate this manager owns.
        self._reconciler = Reconciler(
            server=server,
            registry=registry,
            discovery_state=discovery_state,
            metadata_db=metadata_db,
            credentials_config=credentials_config,
            monitored_dirs=monitored_dirs,
            cloud_roots=self._cloud_roots,
            entry_for=self._entry_for,
            notify_source_committed=self._notify_source_committed,
        )

    def start(self) -> None:
        """Start the event processing loop."""
        if self._watcher is None:
            return  # Static-only mode; no filesystem events to process

        if self._thread is not None and self._thread.is_alive():
            logger.warning("SourceManager already running")
            return

        # The background bootstrap scan runs in this event loop (the watcher
        # fires its first rescan immediately). Until that first scan completes,
        # ``_initial_scan_done`` stays False so its sources route to the precache
        # backlog rather than the prompt enqueue.
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="SourceManager-EventLoop",
        )
        self._thread.start()
        logger.info("SourceManager started")

    # --- Startup-protocol seam (biopb/biopb#277 item C) -----------------------
    # The launcher drives startup through these public methods instead of poking
    # the private hook attributes / _handle_rescan / _initial_scan_done. Wiring
    # mirrors the server's set_*_handler injection (see cli.py).

    def set_source_committed_hook(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        """Register the hook called with a ``source_id`` when a source is
        committed *after* the initial scan (a live addition).

        The launcher wires this to the precache worker's prompt-enqueue. Startup
        sources are gated out of it (they route to the slow backlog) until the
        first full scan flips the startup boundary -- see ``_commit_add_claim``.
        """
        self._on_source_committed = callback

    def set_initial_scan_complete_hook(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        """Register the hook fired once when the first full scan completes.

        The launcher uses it to seed the precache backlog from the established
        startup catalog. Fired from the event-loop thread (or from
        :meth:`complete_initial_scan` on a static-only config).
        """
        self._on_initial_scan_complete = callback

    def run_initial_scan(self) -> None:
        """Run the bootstrap scan synchronously (public seam for the launcher).

        Under progressive discovery the bootstrap scan normally runs in the
        event loop after :meth:`start`. When no event loop will drive it -- the
        watcher failed to start but monitored dirs exist -- the launcher calls
        this to run one full rescan inline. Being the first pass, it force-fulls,
        stamps freshness, flips the startup gate, and fires the completion hook,
        exactly as the background path would.
        """
        self._handle_rescan()

    def complete_initial_scan(self) -> None:
        """Advance the startup protocol when there is nothing to walk.

        A static-only config (no monitored dirs) has no bootstrap scan, but the
        startup protocol must still complete: stamp catalog freshness, flip the
        precache startup gate, and fire the first-scan-complete hook. Centralizes
        the completion sequence the launcher previously open-coded against the
        private members (biopb/biopb#277 item C). Idempotent: the hook fires only
        on the transition to done.
        """
        self._last_full_rescan_at = time.time()
        self._server.set_last_full_scan(self._last_full_rescan_at)
        if not self._initial_scan_done:
            self._initial_scan_done = True
            self._fire_initial_scan_complete()

    def iter_local_source_mtimes(self) -> List[Tuple[str, float]]:
        """Return ``(source_id, mtime)`` for every currently-registered *local*
        source, for seeding the precache backlog (newest first).

        Remote sources are skipped (no ``os.stat`` mtime), as are any whose path
        can't be stat-ed (e.g. removed between commit and this call).
        """
        # The Reconciler snapshots claims under its state lock (the watcher's
        # event-loop thread adds/removes claims concurrently); we stat() outside
        # that lock (I/O).
        snapshot = self._reconciler.local_claim_paths()
        out: List[Tuple[str, float]] = []
        for source_id, primary_path in snapshot:
            try:
                mtime = os.stat(primary_path).st_mtime
            except OSError:
                continue
            out.append((source_id, mtime))
        return out

    def stop(self, join_timeout: float = 5) -> None:
        """Stop the event processing loop.

        ``join_timeout`` bounds the wait for the daemon event-loop thread to
        exit. The 5s default suits steady-state callers; graceful shutdown passes
        a short value, because the thread may be blocked inside a *blocking*
        upstream re-list RPC (``_reconcile_one_upstream`` ->
        ``list_upstream_source_ids`` -> Flight ``list_flights``) and, being a
        daemon, does not need a clean join at process exit -- a long wait there
        only burns the shutdown budget (biopb/biopb#300). Deeper cancellation of
        that in-flight upstream RPC (reconciler/remote_tensor/client) is the
        follow-up; it is intentionally not attempted here.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
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
                logger.exception(f"Error processing events: {e}")

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
            logger.exception(f"Error handling event {event}: {e}")

    def _handle_rescan(self) -> None:
        """Run one periodic rescan: walk monitored dirs first, then re-list upstreams.

        Local directory sources are discovered *before* the tensor-server upstream
        re-list so a slow/large upstream (hundreds of mirrored sources, each a
        network round-trip) cannot delay the local catalog from appearing: the
        local walk streams its sources first and the upstream mirror fills in
        behind it on the same tick. (biopb/biopb#178 introduced the re-list; this
        ordering keeps it off the local catalog's critical path -- previously the
        re-list ran first, so on the boot tick local sources surfaced only after
        every upstream source had been registered, minutes later.)

        Precache routing subtlety: on the boot tick the local walk flips
        ``_initial_scan_done`` True *before* the upstream re-list runs, which
        would otherwise make ``_commit_add_claim`` prompt-enqueue the entire
        startup upstream mirror at the precache worker's un-idle-gated live tier
        (hundreds of upstream chunk fetches competing with serving -- the very
        thing this reorder protects the local catalog from). The whole tick is a
        startup tick if the scan had not completed when it began, so the upstream
        mirror it registers is startup set and must route to the slow backlog. We
        suppress the live enqueue across just that re-list.
        """
        # Serialize the whole pass against a concurrent runtime add_local_source
        # (Flight thread) so the two never mutate the confirmed catalog at once.
        with self._catalog_lock:
            startup_tick = not self._initial_scan_done
            self._rescan_monitored_dirs()
            # tensor-server upstream re-list (biopb/biopb#178): adaptive per-upstream
            # cadence -- fast (every tick) while changing/failing, backing off toward
            # full_rescan_interval while a source set stays stable. Runs AFTER the
            # local walk (see docstring).
            if startup_tick:
                self._suppress_live_precache = True
                try:
                    self._reconcile_due_upstreams()
                finally:
                    self._suppress_live_precache = False
            else:
                self._reconcile_due_upstreams()

    def _rescan_monitored_dirs(self) -> None:
        """Walk the monitored directories and reconcile the discovered catalog.

        No-op for an upstream-only config (no monitored dirs); that case's
        freshness signals + first-scan gate are driven by _reconcile_due_upstreams.
        """
        if not self._monitored_dirs:
            return

        self._cleanup_deleted_monitored_dirs()
        if not self._monitored_dirs:
            return

        force_full_rescan = self._should_force_full_rescan()
        # Progressive-discovery freshness signals: while a *full* reconcile runs,
        # the health action reports full_scan_in_progress=True; on success it
        # advances last_full_scan_finished_at. Incremental rescans leave both
        # untouched (they deliberately skip stable/cloud subtrees, so they are
        # not a whole-tree reconcile). Guaranteed reset in the outer finally.
        if force_full_rescan:
            self._server.set_full_scan_in_progress(True)
        try:
            snapshot = self._scanner.scan(
                monitored_dirs=self._monitored_dirs,
                cloud_roots=self._cloud_roots,
                force_full=force_full_rescan,
                prev_entry_states=self._entry_states,
                prev_cloud_entry_states=self._cloud_entry_states,
            )
            next_state = snapshot.entry_states
            skipped_dirs = snapshot.skipped_dirs
            next_cloud = snapshot.cloud_by_path
            previous_state = self._entry_states
            previous_skipped_dirs = self._skipped_stable_dirs

            self._entry_states = next_state
            self._skipped_stable_dirs = skipped_dirs

            rescan_succeeded = False
            try:
                # Progressive population (Option B): on the *first* full scan,
                # register each source the moment the walk claims it instead of
                # batching every add into the end-of-walk reconcile, so the
                # catalog grows within the walk. This is safe only for the first
                # scan -- it starts empty and force-full, so there are no removals
                # to diff and every claim is a pure add. The claim phase already
                # applies the stability gate (path_filter), so deferred/unstable
                # entries are never claimed and therefore never streamed; they
                # are picked up by the next steady-state rescan. The end-of-walk
                # reconcile below still runs and is idempotent for streamed adds.
                stream_first_scan = force_full_rescan and not self._initial_scan_done
                discovered_state = DiscoveryState()
                if stream_first_scan:
                    discovered_state.on_source_added = (
                        self._reconciler._stream_first_scan_add
                    )

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
                        (path_str, entry.is_directory, entry.signature)
                        for path_str, entry in next_state.items()
                    ),
                    self._registry,
                    state=discovered_state,
                    dim_labels=self._dim_labels,
                    path_filter=self._should_scan_resolved,
                    skipped_dirs=skipped_dirs,
                    cloud_by_path=next_cloud,
                )

                self._reconciler._preserve_skipped_claims(
                    discovered_state, skipped_dirs
                )

                unstable_paths = self._get_unstable_paths()
                self._reconciler._reconcile_discovered_state(
                    discovered_state, unstable_paths, force_full=force_full_rescan
                )
                rescan_succeeded = True
            finally:
                if not rescan_succeeded:
                    self._entry_states = previous_state
                    self._skipped_stable_dirs = previous_skipped_dirs

            if force_full_rescan and rescan_succeeded:
                self._last_full_rescan_at = time.time()
                self._server.set_last_full_scan(self._last_full_rescan_at)
                # Partition the just-walked cloud entries out of _entry_states into
                # the cloud partition. This runs only after the force_full claim +
                # reconcile have already seen the full _entry_states (cloud
                # included), so cloud sources reconcile normally here; afterwards
                # _entry_states holds non-cloud only, so the frequent incremental
                # rescans never iterate cloud entries (the GIL-stall fix). next_state
                # is self._entry_states (set above), so popping trims it in place.
                # Only on success -> a failed force_full leaves the previous
                # _cloud_entry_states intact.
                cloud_state: Dict[str, EntryState] = {}
                for path_str, is_cloud in next_cloud.items():
                    if not is_cloud:
                        continue
                    entry = next_state.pop(path_str, None)
                    if entry is not None:
                        cloud_state[path_str] = entry
                self._cloud_entry_states = cloud_state
                # First full scan done: flip the precache gate (live additions
                # now prompt-enqueue) and let the launcher seed the backlog with
                # the established catalog. Fired once, best-effort.
                if not self._initial_scan_done:
                    self._initial_scan_done = True
                    self._fire_initial_scan_complete()
        finally:
            if force_full_rescan:
                self._server.set_full_scan_in_progress(False)

    def _fire_initial_scan_complete(self) -> None:
        """Invoke the first-scan-complete callback, swallowing any error.

        Best-effort, mirroring the precache commit hook: a callback failure must
        never abort or destabilize the rescan that triggered it.
        """
        callback = self._on_initial_scan_complete
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.exception("on_initial_scan_complete callback failed")

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

        for source_id, claim in self._reconciler.claim_items():
            if is_remote_url(claim.primary_path):
                continue
            try:
                claim_path = Path(claim.primary_path).resolve(strict=False)
            except OSError:
                continue
            if (
                claim_path == deleted_root or claim_path.is_relative_to(deleted_root)
            ) and self._reconciler._commit_remove_source(source_id):
                removed_source_ids.append(source_id)

        deleted_root_str = str(deleted_root)
        self._monitored_dirs.discard(deleted_dir)
        self._skipped_stable_dirs.discard(deleted_root_str)

        entry_paths_to_remove = [
            path_str
            for path_str in self._entry_states
            if path_str == deleted_root_str
            or Path(path_str).is_relative_to(deleted_root)
        ]
        for path_str in entry_paths_to_remove:
            self._entry_states.pop(path_str, None)

        # Cloud entries live in the partition, not _entry_states; prune them too.
        cloud_paths_to_remove = [
            path_str
            for path_str in self._cloud_entry_states
            if path_str == deleted_root_str
            or Path(path_str).is_relative_to(deleted_root)
        ]
        for path_str in cloud_paths_to_remove:
            self._cloud_entry_states.pop(path_str, None)

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

    def _should_force_full_rescan(self) -> bool:
        """Return True when a full tree walk should bypass subtree pruning."""
        if self._full_rescan_interval <= 0:
            return False
        return time.time() - self._last_full_rescan_at >= self._full_rescan_interval

    def _entry_for(self, path_str: str) -> Optional[EntryState]:
        """Cached signature entry for a path, from either partition.

        Cloud entries live in ``_cloud_entry_states`` (walked only on force_full),
        non-cloud in ``_entry_states``. Readers that may receive a cloud member path
        outside the force_full walk (signature diff, stability gate) use this so a
        cloud member is found in the partition instead of falling through to a live
        ``Path(member).stat()`` -- a cloud network round-trip.
        """
        entry = self._entry_states.get(path_str)
        if entry is None:
            entry = self._cloud_entry_states.get(path_str)
        return entry

    def _should_scan_resolved(self, resolved_str: str) -> bool:
        """Stability gate for an entry whose resolved path string is already known.

        The snapshot-driven discovery (biopb/biopb#56 item 4) iterates ``next_state``
        keys, which ``TreeScanner._scan_tree_state`` already stored as resolved path strings, so
        a per-entry ``Path.resolve()`` would be pure waste. This carries the
        load-bearing ``pending_scan`` clear-on-pass side effect (the #53
        subtree-pending prune gate depends on it) by mutating the cached
        ``EntryState`` in place.
        """
        if os.path.basename(resolved_str).startswith("."):
            return False

        if resolved_str in self._skipped_stable_dirs:
            return False

        entry = self._entry_for(resolved_str)
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
        #
        # This bypass is also load-bearing for the cloud inode-backfill skip in
        # TreeScanner._scan_tree_state (biopb/biopb#190): under cloud the entry signature
        # degrades to a constant (0, 0), leaving the stability counter meaningless
        # -- safe only because this early-return means that counter is never read
        # for a cloud path. Removing the bypass would make that skip incorrect.
        if self._is_under_cloud_root(resolved_str):
            entry.pending_scan = False
            return True

        age = time.time() - entry.last_changed
        if age < self._stability_window:
            return False

        if entry.stable_observations < self._stable_rescans_required:
            return False

        if (
            not entry.is_directory
            and self._probe_open_files
            and not self._can_open_for_append(Path(resolved_str))
        ):
            return False

        entry.pending_scan = False
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
        for path_str, entry in self._entry_states.items():
            if now - entry.last_changed < self._stability_window:
                unstable.append(Path(path_str))
        return unstable

    def _is_under_cloud_root(self, path: str) -> bool:
        """True when *path* is a cloud-opted root or lives under one.

        Consulted by the stability gate (:meth:`_should_scan_resolved`); shares
        the cloud-membership rule with the Reconciler via the module free
        function so neither object depends on the other.
        """
        return is_under_cloud_root(self._cloud_roots, path)

    def _notify_source_committed(self, source_id: str) -> None:
        """Precache routing gate for a freshly committed source (injected into
        the Reconciler as ``notify_source_committed``).

        Only live additions -- those committed after the initial scan completes,
        and outside the boot-tick upstream re-list guarded by
        ``_suppress_live_precache`` -- are prompt-enqueued; the startup set routes
        to the slow backlog instead. This manager owns the startup/suppress state
        that decides that, so the gate lives here rather than in the Reconciler.
        Best-effort: a hook failure must never abort a source commit.
        """
        if (
            self._initial_scan_done
            and not self._suppress_live_precache
            and self._on_source_committed is not None
        ):
            try:
                self._on_source_committed(source_id)
            except Exception:
                logger.exception(
                    "precache on_source_committed hook failed for %s",
                    source_id,
                )

    def should_warm(self, source_id: str) -> bool:
        """Whether the precache worker may warm *source_id* right now.

        A thin pass-through to the Reconciler's residency policy (kept on this
        manager because the precache worker holds the manager as its warm gate).
        """
        return self._reconciler.should_warm(source_id)

    def remove_dropped_root(
        self, root_url: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Deregister every drag-dropped source at or under a ``dnd://`` branch root.

        The narrow counterpart to :meth:`add_local_source`: it removes ONLY
        drag-dropped sources, identified by the ``dnd://`` scheme their catalog
        ``source_url`` carries. That scheme is stamped by ``_drop_catalog_url``
        only for drops that are new and outside every monitored root -- i.e.
        sources nothing will re-add -- so it is a sound authorization key. A
        source matches when its ``source_url`` equals ``root_url`` or is under it
        (the ``root_url + "/"`` prefix), so one dropped folder's sources go as a
        unit.

        Runs under ``self._catalog_lock`` (like add / rescan) so the confirmed
        catalog stays single-writer. Returns ``(removed_ids, failed)`` where
        ``failed`` is a list of ``(source_id, reason)``.

        Raises ``ValueError`` if ``root_url`` does not carry the ``dnd://``
        scheme -- the authorization boundary: only user-dropped sources are ever
        removable this way. The bare scheme (``dnd://`` with no branch under it)
        is refused for the same reason: ``rstrip("/")`` would collapse it to a
        ``dnd:/`` prefix that matches *every* drop, so it must not resolve to a
        wildcard "remove all drops".
        """
        if not root_url.startswith(DND_URL_PREFIX):
            raise ValueError(
                f"remove_source only removes drag-dropped ({DND_URL_PREFIX}) "
                f"sources; refusing root_url: {root_url!r}"
            )
        if not root_url[len(DND_URL_PREFIX) :].strip("/\\"):
            raise ValueError(
                f"remove_source needs a branch root under {DND_URL_PREFIX}, "
                f"not the bare scheme; refusing root_url: {root_url!r}"
            )

        removed: List[str] = []
        failed: List[Tuple[str, str]] = []
        prefix = root_url.rstrip("/") + "/"
        with self._catalog_lock:
            source_ids = self._reconciler.claim_ids()
            targets = [
                source_id
                for source_id in source_ids
                if (desc := self._descriptor_for(source_id)) is not None
                and (desc.source_url == root_url or desc.source_url.startswith(prefix))
            ]
            for source_id in targets:
                if self._reconciler._commit_remove_source(source_id):
                    removed.append(source_id)
                else:
                    failed.append((source_id, "not present (already removed?)"))
        return removed, failed

    def _reconcile_due_upstreams(self) -> None:
        """Re-list the upstreams that are due this tick (adaptive backoff).

        Each upstream re-lists on the fast rescan tick while it is changing or
        failing, and backs off (period doubles per unchanged re-list, capped at
        full_rescan_interval) while it is stable -- so a stable lab store is not
        queried every 30s forever, yet a new source / a recovered upstream is
        mirrored within ~one tick. When there are no monitored *dirs* this is the
        sole reconcile, so the first pass also drives the progressive-discovery
        freshness signals + first-scan gate the dir path would otherwise own.
        """
        if not self._monitored_upstreams:
            return
        due: List[SourceConfig] = []
        for upstream in self._monitored_upstreams:
            state = self._upstream_relist.setdefault(
                upstream.url, {"period": 1, "countdown": 0}
            )
            state["countdown"] -= 1
            if state["countdown"] <= 0:
                due.append(upstream)
        if not due:
            return

        upstream_only = not self._monitored_dirs
        first_pass = upstream_only and not self._initial_scan_done
        if first_pass:
            self._server.set_full_scan_in_progress(True)
        try:
            for upstream in due:
                self._reconcile_and_reschedule(upstream)
        finally:
            if upstream_only:
                # Each completed pass re-verifies the (remote) catalog -> advance
                # freshness. in_progress / the first-scan gate fire once, on boot.
                self._last_full_rescan_at = time.time()
                self._server.set_last_full_scan(self._last_full_rescan_at)
                if first_pass:
                    self._server.set_full_scan_in_progress(False)
                    self._initial_scan_done = True
                    self._fire_initial_scan_complete()

    def _reconcile_and_reschedule(self, upstream: SourceConfig) -> None:
        """Re-list one upstream and set its next-due period from the outcome."""
        state = self._upstream_relist[upstream.url]
        try:
            changed = self._reconciler._reconcile_one_upstream(upstream)
        except Exception:
            # Failure (unreachable): retry on the fast cadence next tick.
            self._failed_upstreams.add(upstream.url)
            state["period"] = 1
            state["countdown"] = state["period"]
            logger.warning(
                "Upstream re-list failed for %s; keeping its current catalog "
                "(retrying on the next rescan)",
                upstream.url,
                exc_info=True,
            )
            return
        self._failed_upstreams.discard(upstream.url)
        if changed:
            state["period"] = 1  # the catalog moved -> stay fast
        else:
            # Stable -> back off (double, capped at full_rescan_interval).
            state["period"] = min(state["period"] * 2, self._upstream_max_period)
        state["countdown"] = state["period"]

    def _reconcile_upstreams(
        self, upstreams: Optional[List[SourceConfig]] = None
    ) -> None:
        """Re-list each given tensor-server upstream now (ignoring the backoff
        schedule); used by tests and any caller that wants an immediate pass.

        Best-effort per upstream: an unreachable upstream leaves its currently
        mirrored sources in place (no spurious removals) and is marked failed.
        """
        if upstreams is None:
            upstreams = self._monitored_upstreams
        for upstream in upstreams:
            try:
                self._reconciler._reconcile_one_upstream(upstream)
            except Exception:
                self._failed_upstreams.add(upstream.url)
                logger.warning(
                    "Upstream re-list failed for %s; keeping its current catalog "
                    "(will retry on the next rescan)",
                    upstream.url,
                    exc_info=True,
                )
            else:
                self._failed_upstreams.discard(upstream.url)

    def add_local_source(
        self,
        url: str,
        source_type: str = "",
        dim_labels: Optional[List[str]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        """Register ``url`` (a path on the server) as source(s) at runtime.

        Generator, driving the "add_source" Flight action / tensor-browser
        drag-drop. It yields event tuples the caller maps onto the wire:

        - ``("progress", added_count, current_path)`` -- one per source as it
          registers (a running count + the path being scanned),
        - ``("result", added, already_present, failed)`` -- exactly one terminal
          tally (``added`` is a list of descriptors, ``already_present`` a list
          of source_ids, ``failed`` a list of ``(path, reason)``).

        The walk + commit run inline on the CALLING (Flight handler) thread, but
        under ``self._catalog_lock`` -- so they are mutually exclusive with the
        periodic rescan and the confirmed catalog stays single-writer. A dropped
        directory that is not itself a dataset is walked recursively and may
        register several sources; discovery runs into a scratch state so it never
        mutates the confirmed catalog until a claim is committed.

        ``should_cancel()`` is polled between sources: a cancel stops discovery
        but KEEPS everything already committed (registration is not rolled back).

        Whole-request problems raise before the first yield:
        ``FileNotFoundError`` / ``PermissionError`` (server-side path check) or
        ``ValueError`` (remote URL -- runtime add is local-only for now).
        """
        if is_remote_url(url):
            raise ValueError(
                "Runtime source add supports local filesystem paths only; "
                f"got remote URL: {url}"
            )

        # Server-side locality/existence check (belt-and-suspenders; the client
        # already gated on a localhost server, which shares this filesystem).
        real = resolve_local_path(url)
        if not os.path.exists(real):
            raise FileNotFoundError(f"Path not found on server: {url}")
        if not os.access(real, os.R_OK):
            raise PermissionError(f"Path not readable by the server: {url}")
        url = real
        is_dir = os.path.isdir(url)

        added: List[Any] = []
        already_present: List[str] = []
        failed: List[Tuple[str, str]] = []

        # Acquire the catalog lock, heart-beating while a rescan holds it so a
        # long wait does not sit silent long enough to trip a proxy timeout.
        while not self._catalog_lock.acquire(timeout=_ADD_SOURCE_ACQUIRE_HEARTBEAT):
            yield ("progress", 0, "waiting for catalog scan to finish")
        try:
            # Containment check (case 4): if a STRICT ancestor of the drop is
            # already owned by a source, the drop is *inside* that source. The
            # exact-path member dedup in DiscoveryState.add_claim does not catch
            # this (dir sources record only the dir as a member), so reject here.
            owner = self._reconciler._find_containing_source(url)
            if owner is not None:
                failed.append((url, f"already part of source '{owner}'"))
                yield ("result", added, already_present, failed)
                return

            # Is the dropped path itself a dataset (single claim), or a plain
            # folder to recurse into? Probe the root once against a scratch state.
            scratch = DiscoveryState()
            root_claims = self._registry.get_claims_for_path(
                ClaimContext(Path(url)), scratch
            )
            if root_claims:
                claims: List[SourceClaim] = [root_claims[0]]
            elif is_dir:
                # A plain folder is walked recursively (case 5). The large-drop
                # footgun guard lives client-side (the tensor browser confirms
                # before sending an oversized folder): drag-drop is localhost-only,
                # so the client shares this filesystem and can size the tree before
                # any scan is sent. A direct SDK caller passing a path is explicit
                # intent, so the walk is not gated here.
                discover_sources(
                    Path(url),
                    self._registry,
                    scratch,
                    dim_labels=dim_labels,
                )
                claims = list(scratch.claims.values())
            else:
                claims = []

            if not claims:
                reason = (
                    "no supported datasets found under directory"
                    if is_dir
                    else "not a recognized image format"
                )
                failed.append((url, reason))
                yield ("result", added, already_present, failed)
                return

            # Assign identity to every claim up front so the overlap check below
            # can see the whole drop before any of it is committed.
            for claim in claims:
                if source_type:
                    claim.source_type = source_type
                if dim_labels:
                    claim.dim_labels = list(dim_labels)
                if not claim.source_id:
                    claim.source_id = generate_source_id(
                        str(claim.primary_path), claim.source_type
                    )

            # Re-root the drop into its own browser tree root only when it is
            # ENTIRELY NEW. If any claim is already registered, this drop is a
            # rescan of a location already represented in the tree -- e.g. a
            # monitor=false config dir dropped to pick up new files -- so keep the
            # native source_url on the new siblings. Re-rooting them instead would
            # split that one dir's old and new contents across two roots with
            # nothing to reconcile them (a monitor=false dir never rescans).
            reroot = not any(
                self._reconciler.has_claim(claim.source_id) for claim in claims
            )

            for claim in claims:
                already = self._reconciler.has_claim(claim.source_id)
                if already:
                    already_present.append(claim.source_id)
                else:
                    # Re-rooting (own display root) and the ``dnd://`` origin
                    # marker are decoupled: a drop under a monitored root still
                    # gets a tidy display root, but NOT the marker -- the periodic
                    # rescan re-discovers it, so it is not safely removable. Only a
                    # drop outside every monitored root is stamped ``dnd://``, so
                    # the marker stays equivalent to "user-added and nothing will
                    # re-add it" (what Phase 2 removal authorizes on).
                    catalog_url = (
                        _drop_catalog_url(
                            url,
                            claim.primary_path,
                            mark_dnd=not self._reconciler._is_monitored_claim(claim),
                        )
                        if reroot
                        else None
                    )
                    if self._reconciler._commit_add_claim(
                        claim, catalog_url=catalog_url
                    ):
                        desc = self._descriptor_for(claim.source_id)
                        added.append(desc)
                        yield ("progress", len(added), str(claim.primary_path))
                    else:
                        failed.append(
                            (
                                str(claim.primary_path),
                                "could not open or register (see server log)",
                            )
                        )

                if should_cancel is not None and should_cancel():
                    break

            yield ("result", added, already_present, failed)
        finally:
            self._catalog_lock.release()

    def _descriptor_for(self, source_id: str):
        """Fetch the registered source's DataSourceDescriptor (None if missing)."""
        adapter = self._server.sources.get(source_id)
        return adapter.get_source_descriptor() if adapter is not None else None


def create_source_manager(
    server: TensorFlightServer,
    registry: AdapterRegistry,
    watcher: Optional[DirectoryWatcher],
    monitored_sources: Optional[List[SourceConfig]] = None,
    static_sources: Optional[List[SourceConfig]] = None,
    metadata_db: Optional[MetadataDatabase] = None,
    credentials_config: Optional[Any] = None,
    stability_window: float = 30.0,
    probe_open_files: bool = True,
    full_rescan_interval: float = 3600.0,
    stable_rescans_required: int = 0,
    aggressive_dir_pruning: bool = False,
    allow_empty: bool = False,
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
        metadata_db: MetadataDatabase to keep in sync as sources are added/removed
            (None when the feature is disabled)
        credentials_config: CredentialsConfig for remote storage authentication
        allow_empty: When True, build and return an empty SourceManager instead of
            None when there are no (valid) sources. An empty catalog is a valid
            runtime state -- sources can arrive later via runtime add_source
            (napari drag-drop), DoPut uploads, or a monitored dir that fills after
            startup -- so the launcher serves it (health SERVING, empty
            list_flights) rather than refusing to boot (biopb/biopb#515).

    Returns:
        SourceManager if there are any sources (or allow_empty=True), None otherwise
    """
    monitored_sources = monitored_sources or []
    static_sources = static_sources or []

    if not monitored_sources and not static_sources and not allow_empty:
        return None

    # A bare-host tensor-server upstream ("mirror everything") IS monitored -- its
    # catalog is re-listed/reconciled in the background (biopb/biopb#178) -- it just
    # is not *filesystem*-watched. Distinguished here so the log line below is not
    # misleading, and reused for the monitored_upstreams filter.
    from biopb_tensor_server.adapters.remote_tensor import _split_grpc_url

    def _is_bare_host_upstream_url(url: str) -> bool:
        return (
            url.lower().startswith(("grpc://", "grpc+tls://", "grpcs://"))
            and _split_grpc_url(url)[1] is None
        )

    # Extract monitored directories. Remote sources are never filesystem-watched;
    # a bare-host upstream is still monitored via the background re-list, whereas
    # any other remote source (a single-source grpc://host/<id>, or an s3://...
    # entry) is registered statically.
    monitored_dirs: Set[Path] = set()
    for source in monitored_sources:
        if source.is_remote:
            if _is_bare_host_upstream_url(source.url):
                logger.info(
                    "Tensor-server upstream %s: catalog re-listed in the background, "
                    "not filesystem-watched",
                    source.url,
                )
            else:
                logger.info(
                    "Remote source %s is registered statically, not monitored",
                    source.url,
                )
            continue

        local_path = source.local_path
        if local_path is None or not local_path.exists():
            logger.warning(f"Cannot monitor non-existent path: {source.url}")
            continue

        if local_path.is_file():
            logger.warning(f"Cannot monitor single file: {source.url}")
            continue

        monitored_dirs.add(local_path)

    # Tensor-server upstreams (bare-host grpc://) whose catalog is re-listed and
    # reconciled in the background (biopb/biopb#178). Every bare-host upstream
    # qualifies regardless of `monitor`: cli._resolve_serve_sources routes them all
    # here (never to inline static expansion) so a large upstream neither blocks
    # SERVING nor pays a per-source get_descriptor RPC -- `monitor=false` only makes
    # the adaptive cadence back off after the boot-tick reconcile. A single-source
    # grpc://host/<id> entry has nothing to re-list, so it is excluded -- only the
    # bare-host "mirror everything" form qualifies.
    monitored_upstreams = [
        ms
        for ms in monitored_sources
        if ms.is_remote and _is_bare_host_upstream_url(ms.url)
    ]

    # An unreachable monitored upstream contributes no static sources at startup
    # (its bare-host expansion was skipped), but the re-list will populate it once
    # it is reachable -- so it counts as "something to serve" and must not let the
    # server hard-fail to start (#178: require monitor=true for bare-host recovery).
    if not monitored_dirs and not static_sources and not monitored_upstreams:
        if not allow_empty:
            logger.warning("No valid sources to serve")
            return None
        # Empty is a valid runtime state (biopb/biopb#515): fall through and build
        # an empty manager so the server serves an empty catalog and accepts
        # sources added later (runtime add_source, DoPut, a monitored dir that
        # fills after startup).
        logger.warning("No sources configured yet; serving an empty catalog")

    # Resolved roots opted into cloud/synced-folder handling (config cloud=true),
    # across both monitored and static sources. Under a monitored cloud root the
    # walk admits dehydrated entries; for any cloud source the registration path
    # defers a non-resident dataset to lazy resolution (cloud-storage phase 2).
    cloud_roots: Set[Path] = set()
    for source in (*monitored_sources, *static_sources):
        if source.cloud:
            # EXPERIMENTAL: cloud/synced-folder mode (offline placeholders resolved
            # lazily on first access) is not yet stable. Warned once per configured
            # cloud source at startup.
            logger.warning(
                "Source %r uses the EXPERIMENTAL 'cloud' mode (offline/synced-folder "
                "placeholders resolved lazily on first access): its behavior and "
                "config surface may change without notice in a future release.",
                source.url,
            )
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
        metadata_db=metadata_db,
        dim_labels=monitored_sources[0].dim_labels if monitored_sources else None,
        credentials_config=credentials_config,
        stability_window=stability_window,
        probe_open_files=probe_open_files,
        full_rescan_interval=full_rescan_interval,
        stable_rescans_required=stable_rescans_required,
        aggressive_dir_pruning=aggressive_dir_pruning,
        cloud_roots=cloud_roots,
        monitored_upstreams=monitored_upstreams,
    )

    # Seed static sources as direct claims (explicit config, no filesystem walk)
    # These are added first so monitored discovery skips paths already claimed.
    for source in static_sources:
        extra_config = {}
        if source.dataset:
            extra_config["dataset"] = source.dataset
        # credentials_profile is dropped by the claim->SourceConfig rebuild in
        # _register_source_claim; carry it here so a tensor-server proxy's
        # per-upstream token (and any remote source's profile) reaches
        # create_from_config.
        if source.credentials_profile:
            extra_config["credentials_profile"] = source.credentials_profile
        # Store the canonical resolved form of a local config path (the same
        # resolve_local_path the source_id hash, the containment guard, and the
        # drop path all use) so its claim key compares equal to a drop that lands
        # inside it. Otherwise a source configured through a symlink/junction
        # (e.g. /data/current -> /data/2026-07, or a Windows junction / mapped
        # drive) keeps its raw path, so a drop *inside* it -- whose resolved form
        # differs -- evades the "already part of <source>" guard and double-
        # registers. Monitored sources reach the same form via the walk's
        # Path.resolve. A remote URL is left verbatim -- is_remote_url is
        # prefix-based, so a Windows drive letter (C:\...) stays a local path.
        primary_path = source.url
        if not is_remote_url(source.url):
            primary_path = resolve_local_path(source.url)
        claim = SourceClaim(
            source_type=source.type,
            primary_path=primary_path,  # resolved local path; remote URL verbatim
            source_id=source.source_id,
            dim_labels=source.dim_labels,
            extra_config=extra_config,
            # A static source explicitly flagged cloud is always deferred: the
            # user said "don't open it eagerly". If it is in fact resident, the
            # first access still resolves it cheaply.
            unresolved=bool(source.cloud),
        )
        # source._catalog_url is the alias-derived display tree-root for a local
        # source (config.resolve_all_sources), or None. Threaded as the descriptor's
        # source_url override, exactly like the drag-drop re-rooting path.
        manager._reconciler._commit_add_claim(claim, catalog_url=source._catalog_url)

    # Monitored discovery is NOT run synchronously here: under progressive
    # discovery the launcher starts the manager's event loop and the watcher
    # fires the first rescan immediately, so the (possibly slow) bootstrap scan
    # happens in the background while the server already reports SERVING. A
    # static-only config (no monitored_dirs) has nothing to scan -- the launcher
    # drives the first-scan-complete path directly so it still reports a
    # freshness timestamp and seeds the backlog.
    return manager
