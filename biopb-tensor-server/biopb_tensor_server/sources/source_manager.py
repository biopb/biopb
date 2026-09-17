"""Source lifecycle manager for the periodic catalog rescan runtime.

Owns the rescan timer, discovery state, server catalog updates, and metadata
database synchronization for monitored local directories. Monitoring is
timer-driven: every ``rescan_interval`` seconds the whole monitored tree is
walked and diffed, so there are no per-path filesystem events to handle.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    discover_sources,
    discover_sources_from_entries,
    generate_source_id,
    local_path_is_rooted,
    resolve_local_path,
)
from biopb_tensor_server.core.errors import UpstreamConfigError
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.sources.reconciler import Reconciler, is_under_cloud_root
from biopb_tensor_server.sources.resolve import _reroot_catalog_url
from biopb_tensor_server.sources.tree_scanner import (
    EntryState,
    TreeScanner,
    entry_is_quiet,
)

if TYPE_CHECKING:
    from biopb_tensor_server.serving.metadata_db import MetadataDatabase
    from biopb_tensor_server.serving.server import TensorFlightServer

logger = logging.getLogger(__name__)


# Backoff ceiling for the tensor-server upstream re-list, in rescan ticks
# (biopb/biopb#178). A re-list runs every tick while an upstream is changing or
# failing; while it stays stable the spacing doubles up to this many ticks, so a
# fully-stable upstream settles to re-listing about once an hour at the default
# 30s rescan tick (instead of querying it every 30s forever).
_UPSTREAM_RELIST_MAX_TICKS = 120

# How often a still-unreachable upstream re-reports itself. The re-list stays on
# the fast cadence -- an upstream that is merely down recovers on its own, so
# retrying every tick is right -- but repeating the same traceback at that rate
# buries every other line in the log: an overnight outage at the default 30s tick
# is ~1200 of them. Rate-limit the report, not the retry.
_UPSTREAM_FAILURE_LOG_INTERVAL = 300.0

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


@dataclass
class AddSourceTally:
    """The terminal outcome of one ``add_local_source`` drop.

    A record rather than a positional tuple: the categories are all lists of
    ids, so a mis-ordered unpack at one of the yield sites would be silent.
    """

    added: List[str] = field(default_factory=list)
    already_present: List[str] = field(default_factory=list)
    refreshed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)


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
        monitored_dirs: Set[Path],
        rescan_interval: float = 30.0,
        metadata_db: Optional[MetadataDatabase] = None,
        credentials_config: Optional[Any] = None,
        stability_window: float = 30.0,
        full_rescan_interval: float = 3600.0,
        aggressive_dir_pruning: bool = False,
        cloud_roots: Optional[Set[Path]] = None,
        monitored_upstreams: Optional[List[SourceConfig]] = None,
        prune_unseen_days: int = 0,
    ):
        # Collaborators. The registry is kept for ``add_local_source``'s own
        # discovery walk; every confirmed-catalog mutation goes through the
        # Reconciler built at the end of this method, which holds its own copy.
        self._server = server
        self._registry = registry
        self._metadata_db = metadata_db

        # What is watched. Under a cloud root (config ``cloud=true``) dehydrated
        # entries are admitted and registered as unresolved sources that resolve
        # on first access. Upstreams are bare-host ``grpc://`` tensor servers
        # whose catalog is re-listed and reconciled like a directory walk; a
        # single-source ``grpc://host/<id>`` entry is not here, having nothing to
        # re-list.
        self._monitored_dirs = monitored_dirs
        self._cloud_roots: Set[Path] = cloud_roots or set()
        self._monitored_upstreams: List[SourceConfig] = monitored_upstreams or []

        # Scan tuning, and the filesystem signature walk it configures. The
        # scanner is a pure producer: given the previous caches it returns a
        # fresh ScanSnapshot, and this manager publishes, rolls back and
        # partitions that snapshot.
        self._stability_window = stability_window
        self._full_rescan_interval = full_rescan_interval
        self._aggressive_dir_pruning = aggressive_dir_pruning
        self._scanner = TreeScanner(
            stability_window=stability_window,
            aggressive_dir_pruning=aggressive_dir_pruning,
        )

        # Scan caches: path -> EntryState (signature + pending-scan flag).
        # Cloud entries sit in their own partition because
        # cloud subtrees are walked only on the force_full pass -- keeping them
        # out of ``_entry_states`` is what holds every per-entry rescan loop to
        # O(non-cloud). That partition is rebuilt only at the end of a successful
        # force_full, so a failed one leaves the last good snapshot intact.
        self._entry_states: Dict[str, EntryState] = {}
        self._cloud_entry_states: Dict[str, EntryState] = {}
        self._skipped_stable_dirs: Set[str] = set()
        self._last_full_rescan_at: float = float("-inf")

        # Per-upstream re-list cadence, keyed by url and counted in rescan ticks
        # rather than wall clock: ``countdown`` ticks down to 0 (re-list due),
        # ``period`` is the current spacing. An unchanged re-list doubles the
        # period up to ``_upstream_max_period``; any change or failure resets it
        # to every tick. The three failure maps are reporting state -- what was
        # last said about an upstream, so a persistent fault is reported on a
        # window instead of on every tick.
        self._upstream_relist: Dict[str, Dict[str, int]] = {}
        self._upstream_max_period: int = _UPSTREAM_RELIST_MAX_TICKS
        self._failed_upstreams: Set[str] = set()
        self._upstream_config_errors: Dict[str, str] = {}
        self._upstream_failures: Dict[str, Tuple[float, str]] = {}

        # Annotation orphan clock, driven from :meth:`_mark_catalog_complete`.
        # Auto-prune arms only once the process has been up longer than the
        # threshold it would delete on; monotonic, so an NTP correction cannot
        # age the server into deleting.
        self._prune_unseen_days = max(0, prune_unseen_days)
        self._started_at = time.monotonic()

        # The rescan loop: a bare timer on its own thread. ``_stop`` doubles as
        # the loop's wait condition and its wake signal (the ``precache``
        # worker's idiom), so :meth:`stop` returns at once instead of running
        # out the current interval, and no separate running flag has to be kept
        # in step with it. A ``rescan_interval`` of 0 or less means no loop at
        # all (config ``monitor_mode = "off"``); the launcher then drives a
        # single scan itself. A positive one is floored at 0.1s -- a config
        # value below that is almost certainly a mistake, and without a floor
        # it would drive back-to-back full tree walks.
        self._rescan_interval = (
            rescan_interval if rescan_interval <= 0 else max(0.1, rescan_interval)
        )
        self._next_rescan_at: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Startup protocol, and the precache routing gate it drives.
        # ``_initial_scan_done`` flips at the end of the first successful full
        # rescan -- which under progressive discovery runs in the event loop
        # *after* start(), so it, not "are we past start()", is the
        # startup/runtime boundary. Sources committed before it route to the slow
        # precache backlog; after it, ``_on_source_committed`` prompt-enqueues
        # them. ``_suppress_live_precache`` holds that backlog routing across the
        # boot tick's upstream re-list, which the local walk in the same tick has
        # already flipped the boundary for. Event-loop thread only, so plain
        # flags are safe.
        self._initial_scan_done = False
        self._suppress_live_precache = False
        self._on_source_committed: Optional[Callable[[str], None]] = None
        self._on_initial_scan_complete: Optional[Callable[[], None]] = None

        # Coarse mutex serializing a *whole* catalog-mutation pass. The periodic
        # rescan (event-loop thread) and a runtime ``add_local_source`` (a Flight
        # handler thread) are the only two flows that discover + commit sources;
        # holding this across each keeps the confirmed catalog single-writer
        # without threading add_source through the event loop. Distinct from the
        # Reconciler's fine-grained state RLock that the commit primitives take.
        self._catalog_lock = threading.Lock()

        # The confirmed-catalog writer: owns the live claim set, registration and
        # the discovered/upstream diff. This manager feeds it scan results and
        # delegates every catalog mutation to it. Its injected seams read back
        # into here -- ``entry_for`` for the scan caches above,
        # ``notify_source_committed`` for the precache gate.
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
            stability_window=stability_window,
        )

    def start(self) -> None:
        """Start the periodic rescan loop, if there is anything for it to do.

        A no-op when rescanning is disabled (``rescan_interval <= 0``), or
        neither a monitored directory nor a monitored upstream is configured;
        callers check :meth:`is_running` afterward to tell.

        The bootstrap scan runs in this loop, whose first tick fires
        immediately. Until it completes, ``_initial_scan_done`` stays False so
        its sources route to the precache backlog rather than the prompt
        enqueue.
        """
        if self._rescan_interval <= 0:
            return
        if not self._monitored_dirs and not self._monitored_upstreams:
            return

        if self._thread is not None and self._thread.is_alive():
            logger.warning("SourceManager already running")
            return

        self._next_rescan_at = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="SourceManager-EventLoop",
        )
        self._thread.start()
        logger.info(
            "SourceManager started; rescanning every %.1fs", self._rescan_interval
        )

    def run_bootstrap_fallback(self) -> None:
        """Run the startup scan the caller must drive itself, when :meth:`start`
        didn't (call only after checking :meth:`is_running` is False).

        Centralizes the branch a launcher would otherwise have to re-derive from
        ``monitored_dirs`` -- this manager already knows which case it is: a
        monitored tree with rescanning off still needs one synchronous walk
        (:meth:`run_initial_scan`); a static-only config has nothing to walk, so
        the startup protocol is advanced directly (:meth:`complete_initial_scan`).
        """
        if self._monitored_dirs:
            self.run_initial_scan()
        else:
            self.complete_initial_scan()

    # --- Startup-protocol seam ------------------------------------------------
    # The launcher drives startup through these public methods rather than the
    # private hook attributes / _handle_rescan / _initial_scan_done. Wiring
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
        rescan loop after :meth:`start`. When no loop will drive it -- rescanning
        is off but monitored dirs exist -- the launcher calls this to run one
        full rescan inline. Being the first pass, it force-fulls,
        stamps freshness, flips the startup gate, and fires the completion hook,
        exactly as the background path would.
        """
        self._handle_rescan()

    def _mark_catalog_complete(self) -> None:
        """Publish that a full scan just finished, and run the orphan clock.

        The three paths that can complete one -- a forced full rescan, an
        upstream re-list pass, and a static-only config with nothing to walk --
        all end here, which is what lets the clock be driven by the event rather
        than by a timer that would have to re-derive it. "Complete" is not
        checked here; at this point it is held.

        Auto-prune is armed only once this process has been up longer than the
        threshold it would delete on. Until then it cannot have watched anything
        for long enough to conclude absence: at boot an unmounted drive and an
        unreachable proxy upstream look exactly like deleted ones, and their rows
        are already as old as the downtime. A gate on the first scan alone would
        not do -- an upstream that is down at boot is usually still down an hour
        later, and the second completed scan would delete its annotations.
        """
        self._last_full_rescan_at = time.time()
        self._server.set_last_full_scan(self._last_full_rescan_at)
        if self._metadata_db is None:
            return

        try:
            # Always, and before any delete: last_seen_at does not advance while
            # the server is down, so after a week off every annotation looks a
            # week unseen. Pruning first would take out rows whose images are
            # sitting right there.
            self._metadata_db.mark_sources_seen()
            if self._prune_unseen_days > 0 and self._auto_prune_armed():
                cutoff = datetime.now() - timedelta(days=self._prune_unseen_days)
                self._metadata_db.prune_unseen(cutoff)
        except Exception:
            # A scan must not fail over annotation bookkeeping; the next
            # completed scan retries the whole thing.
            logger.exception("Orphan clock update failed")

    def _auto_prune_armed(self) -> bool:
        """Whether this process has watched long enough to conclude absence.

        Uptime, not scan count: the sweep is the only thing that advances
        ``last_seen_at``, so a row can only be *known* unseen for N days if this
        process has been running the sweep for N days. Anything shorter is
        reading a gap the server was not present for as evidence about the
        world -- and a restart is exactly when that gap is largest.

        The cost is that a server restarted more often than the threshold never
        auto-prunes at all. That is the intended trade: these are hand-drawn
        annotations, and the reporting path (``unseen_rois``) stays available
        for a person to act on whenever they like.
        """
        return time.monotonic() - self._started_at >= self._prune_unseen_days * 86400

    def complete_initial_scan(self) -> None:
        """Advance the startup protocol when there is nothing to walk.

        A static-only config (no monitored dirs) has no bootstrap scan, but the
        startup protocol must still complete: stamp catalog freshness, flip the
        precache startup gate, and fire the first-scan-complete hook. Idempotent:
        the hook fires only on the transition to done.
        """
        self._mark_catalog_complete()
        if not self._initial_scan_done:
            self._initial_scan_done = True
            self._fire_initial_scan_complete()

    def iter_local_source_mtimes(self) -> List[Tuple[str, float]]:
        """Return ``(source_id, mtime)`` for every currently-registered *local*
        source, for seeding the precache backlog (newest first).

        Remote sources are skipped (no ``os.stat`` mtime), as are any whose path
        can't be stat-ed (e.g. removed between commit and this call).
        """
        # The Reconciler snapshots claims under its state lock (the rescan
        # thread adds/removes claims concurrently); we stat() outside
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
        only burns the shutdown budget. That in-flight RPC is not cancelled.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None
        logger.info("SourceManager stopped")

    def is_running(self) -> bool:
        """Check if the manager is actively processing events."""
        return self._thread is not None and self._thread.is_alive()

    def _event_loop(self) -> None:
        """Run a rescan every ``rescan_interval`` seconds until stopped.

        The next tick is scheduled from the moment the previous one *finished*,
        so a rescan that outruns the interval does not immediately queue another.
        A failed rescan is logged and the cadence continues; the next pass sees
        the same tree and retries.
        """
        while not self._stop.wait(max(0.0, self._next_rescan_at - time.monotonic())):
            try:
                self._handle_rescan()
            except Exception:
                logger.exception("Rescan failed")
            self._next_rescan_at = time.monotonic() + self._rescan_interval

    def _handle_rescan(self) -> None:
        """Run one periodic rescan: walk monitored dirs first, then re-list upstreams.

        Local directory sources are discovered *before* the tensor-server upstream
        re-list so a slow/large upstream (hundreds of mirrored sources, each a
        network round-trip) cannot delay the local catalog from appearing: the
        local walk streams its sources first and the upstream mirror fills in
        behind it on the same tick.

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
                # Progressive population: on the *first* full scan, register each
                # source the moment the walk claims it rather than batching every
                # add into the end-of-walk reconcile, so the catalog grows within
                # the walk. Safe only for the first scan -- it starts empty and
                # force-full, so there are no removals to diff and every claim is
                # a pure add. The claim phase already applies the stability gate
                # (path_filter), so deferred/unstable entries are never claimed
                # and therefore never streamed; the next steady-state rescan
                # picks them up. The end-of-walk reconcile below still runs and
                # is idempotent for streamed adds.
                stream_first_scan = force_full_rescan and not self._initial_scan_done
                discovered_state = DiscoveryState()
                if stream_first_scan:
                    discovered_state.on_source_added = (
                        self._reconciler._stream_first_scan_add
                    )

                # Single traversal: the state walk above already visited every
                # entry and recorded its (resolved path, is_directory) into
                # next_state in DFS parent-first order, so the claim phase is
                # driven straight off that snapshot rather than re-walking the
                # filesystem. skipped_dirs prunes the stable subtrees the state
                # walk carried forward; their claims are preserved below.
                discovered_state = discover_sources_from_entries(
                    (
                        (path_str, entry.is_directory, entry.signature)
                        for path_str, entry in next_state.items()
                    ),
                    self._registry,
                    state=discovered_state,
                    path_filter=self._should_scan_resolved,
                    skipped_dirs=skipped_dirs,
                    cloud_by_path=next_cloud,
                )

                self._reconciler._preserve_skipped_claims(
                    discovered_state, skipped_dirs
                )

                self._reconciler._reconcile_discovered_state(
                    discovered_state, force_full=force_full_rescan
                )
                rescan_succeeded = True
            finally:
                if not rescan_succeeded:
                    self._entry_states = previous_state
                    self._skipped_stable_dirs = previous_skipped_dirs

            if force_full_rescan and rescan_succeeded:
                self._mark_catalog_complete()
                # Partition the just-walked cloud entries out of _entry_states.
                # Runs only after the force_full claim + reconcile have seen the
                # full _entry_states (cloud included), so cloud sources reconcile
                # normally here; afterwards _entry_states holds non-cloud only and
                # the frequent incremental rescans never iterate cloud entries.
                # next_state is self._entry_states (set above), so popping trims
                # it in place. Only on success -- a failed force_full leaves the
                # previous _cloud_entry_states intact.
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
        """Stability gate: may this path be claimed on this pass?

        Discovery iterates ``next_state`` keys, which ``TreeScanner`` already
        stored as resolved path strings, so a per-entry ``Path.resolve()`` would
        be pure waste. A pure read -- the walk derives ``pending_scan`` from the
        same ``entry_is_quiet`` predicate this applies, so nothing here reaches
        back into the cached record.
        """
        if os.path.basename(resolved_str).startswith("."):
            return False

        if resolved_str in self._skipped_stable_dirs:
            return False

        entry = self._entry_for(resolved_str)
        if entry is None:
            return False

        # Cloud/synced-folder entries bypass the stability machinery entirely
        # (cloud-storage phase 2): the mtime/ctime age is unreliable on cloud
        # filesystems (doc S1.2), so a placeholder could never stabilize.
        # Archived dehydrated data is inherently stable (never mid-write), so
        # admit it immediately.
        #
        # Load-bearing for TreeScanner's cloud inode-backfill skip: under cloud
        # the entry signature degrades to a constant (0, 0), so `last_changed`
        # never advances -- safe only because this early return means it is
        # never read for a cloud path.
        if self._is_under_cloud_root(resolved_str):
            return True

        return entry_is_quiet(entry.last_changed, time.time(), self._stability_window)

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
                if (url := self._catalog_url_for(source_id)) is not None
                and (url == root_url or url.startswith(prefix))
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
                self._mark_catalog_complete()
                if first_pass:
                    self._server.set_full_scan_in_progress(False)
                    self._initial_scan_done = True
                    self._fire_initial_scan_complete()

    def _log_upstream_config_error(self, url: str, exc: Exception) -> None:
        """Report a broken upstream config once, until it changes or clears.

        A config error repeats identically on every re-list, so logging it each
        pass would bury the (actionable) first report under noise -- but staying
        silent forever hides a *different* breakage introduced by the next edit.
        Keying the suppression on the message re-reports whenever the operator
        changes something without fixing it (biopb/biopb#608).
        """
        message = str(exc)
        if self._upstream_config_errors.get(url) == message:
            logger.debug("upstream %s still misconfigured: %s", url, message)
            return
        self._upstream_config_errors[url] = message
        logger.error(
            "Upstream %s is MISCONFIGURED, not unreachable: %s. Its catalog is "
            "kept as-is; correct the configuration and the next re-list picks it "
            "up -- but that is now up to %d rescan ticks away, not the next one, "
            "because a config error cannot fix itself between ticks.",
            url,
            message,
            self._upstream_max_period,
        )

    def _clear_upstream_config_error(self, url: str) -> bool:
        """Note that a previously misconfigured upstream now resolves.

        Returns whether one was cleared, so the caller can restore the fast
        cadence with it. The recovery is logged because the failure was: an
        operator who fixed the config saw an error telling them it could be up to
        an hour before anything happens, and needs the other half of that
        sentence -- that the edit took, and when (biopb/biopb#608).
        """
        message = self._upstream_config_errors.pop(url, None)
        if message is None:
            return False
        logger.warning(
            "Upstream %s is configured correctly again (was: %s); re-listed and "
            "back on the fast cadence.",
            url,
            message,
        )
        return True

    def _log_upstream_unreachable(self, url: str, exc: Exception) -> None:
        """Report an unreachable upstream, at most once per log window.

        Keyed on the message as well as the window so a *changed* failure is never
        held back: "connection refused" becoming "certificate verify failed" is
        the operator's cue that something moved, and it would be worse than
        useless to sit on it for the rest of the window.
        """
        message = f"{type(exc).__name__}: {exc}"
        now = time.time()
        last = self._upstream_failures.get(url)
        if (
            last is not None
            and last[1] == message
            and now - last[0] < _UPSTREAM_FAILURE_LOG_INTERVAL
        ):
            logger.debug("upstream %s still unreachable: %s", url, message)
            return
        self._upstream_failures[url] = (now, message)
        logger.warning(
            "Upstream re-list failed for %s; keeping its current catalog "
            "(retrying on the next rescan; reporting again in at most %.0fs if "
            "this persists)",
            url,
            _UPSTREAM_FAILURE_LOG_INTERVAL,
            exc_info=True,
        )

    def _clear_upstream_failure(self, url: str) -> None:
        """Announce that a previously unreachable upstream answers again.

        Necessary *because* the failures are rate-limited: without it recovery
        reads as the absence of a line that was already mostly absent, so an
        operator watching the log cannot tell a fixed upstream from a suppressed
        one.
        """
        if self._upstream_failures.pop(url, None) is None:
            return
        logger.info("Upstream %s is reachable again; catalog re-listed.", url)

    def _reconcile_and_reschedule(self, upstream: SourceConfig) -> None:
        """Re-list one upstream and set its next-due period from the outcome."""
        state = self._upstream_relist[upstream.url]
        try:
            changed = self._reconciler._reconcile_one_upstream(upstream)
        except UpstreamConfigError as exc:
            # Config, not connectivity: the fast cadence exists so a *recovering*
            # upstream is picked up within a tick, and nothing about an unreadable
            # trust anchor recovers on its own. Back all the way off instead of
            # re-reading the same broken file every tick (biopb/biopb#608); a
            # fixed config is still picked up, one slow tick later.
            self._failed_upstreams.add(upstream.url)
            state["period"] = self._upstream_max_period
            state["countdown"] = state["period"]
            # We never dialed, so any pending unreachable report is stale -- drop
            # it silently rather than let the config fix also announce a recovery
            # of reachability nothing just observed.
            self._upstream_failures.pop(upstream.url, None)
            self._log_upstream_config_error(upstream.url, exc)
            return
        except Exception as exc:
            # Failure (unreachable): retry on the fast cadence next tick.
            self._failed_upstreams.add(upstream.url)
            state["period"] = 1
            state["countdown"] = state["period"]
            self._log_upstream_unreachable(upstream.url, exc)
            return
        self._failed_upstreams.discard(upstream.url)
        self._clear_upstream_failure(upstream.url)
        recovered = self._clear_upstream_config_error(upstream.url)
        if changed or recovered:
            # Recovery is a state change like any other: this upstream was parked
            # at the slow cadence *because* it was broken, so leaving it there
            # once it works would make the next real change up to an hour late.
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
            except UpstreamConfigError as exc:
                self._failed_upstreams.add(upstream.url)
                self._log_upstream_config_error(upstream.url, exc)
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
                self._clear_upstream_config_error(upstream.url)

    def add_local_source(
        self,
        url: str,
        source_type: str = "",
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        """Register ``url`` (a path on the server) as source(s) at runtime.

        Generator, driving the "add_source" Flight action / tensor-browser
        drag-drop. It yields event tuples the caller maps onto the wire:

        - ``("progress", added_count, current_path)`` -- one per source as it
          registers or refreshes (the count is of NEW sources, so a pure re-drop
          advances only the path),
        - ``("result", tally)`` -- exactly one terminal
          :class:`AddSourceTally`.

        A claim that is already registered is **rebuilt**, not skipped: its
        adapter is reconstructed against the file as it is now, which is the
        only thing that resamples the descriptor and the ``content_version``
        that namespaces the chunk cache (biopb/biopb#944). Those source_ids come
        back in ``refreshed`` as well as ``already_present`` -- the latter keeps
        its meaning, "this id was already known", so an older client still reads
        a re-drop as "already present" rather than "nothing happened".

        The rebuild is unconditional: a directory source's signature is the
        directory's own, and that mtime does not move when a member is rewritten
        in place, so for a zarr or TIFF sequence the drop is the only signal.

        Registered sources under the dropped path whose files are GONE are
        deregistered and reported in ``removed``.

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
        ``ValueError`` (a remote URL -- runtime add is local-only for now -- or a
        relative path, which has no anchor on this side of the wire).
        """
        if is_remote_url(url):
            raise ValueError(
                "Runtime source add supports local filesystem paths only; "
                f"got remote URL: {url}"
            )

        # A rootless path has no meaning across the wire: `resolve_local_path`
        # below would complete it from the *server's* cwd, which the caller does
        # not know and did not pick (biopb/biopb#947). Same predicate the config
        # loader uses, for the same reason and from the same place -- and it
        # judges a `file://` url on the path it carries, which `resolve_local_path`
        # then strips, so both url forms land on one identity here too. The
        # drag-drop client always sends a rooted path (Qt's ``QUrl.toLocalFile``),
        # so this only catches a hand-built call.
        if not local_path_is_rooted(url):
            raise ValueError(
                "Runtime source add requires a rooted path; one without a root "
                f"would be completed from the server's own directory: {url}"
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

        tally = AddSourceTally()

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
                tally.failed.append((url, f"already part of source '{owner}'"))
                yield ("result", tally)
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
                )
                claims = list(scratch.claims.values())
            else:
                claims = []

            # Assign identity to every claim up front so the overlap check below
            # can see the whole drop before any of it is committed.
            for claim in claims:
                if source_type:
                    claim.source_type = source_type
                if not claim.source_id:
                    claim.source_id = generate_source_id(
                        str(claim.primary_path), claim.source_type
                    )

            already_ids = {
                claim.source_id
                for claim in claims
                if self._reconciler.has_claim(claim.source_id)
            }

            # Removal half, before the empty-drop bail-out below: dropping a
            # folder whose contents were deleted is exactly how a stale entry
            # gets noticed, and there is nothing to add in that case. A dropped
            # *file* skips the O(catalog) scan -- its own existence was checked
            # above, and it has no descendants that could have vanished.
            tally.removed = (
                self._deregister_vanished_under(
                    url, {claim.source_id for claim in claims}
                )
                if is_dir
                else []
            )

            if not claims:
                if not tally.removed:
                    reason = (
                        "no supported datasets found under directory"
                        if is_dir
                        else "not a recognized image format"
                    )
                    tally.failed.append((url, reason))
                yield ("result", tally)
                return

            # Re-root the drop into its own browser tree root only when it is
            # ENTIRELY NEW. If any claim is already registered, this drop is a
            # rescan of a location already represented in the tree -- e.g. a
            # monitor=false config dir dropped to pick up new files -- so keep the
            # native source_url on the new siblings. Re-rooting them instead would
            # split that one dir's old and new contents across two roots with
            # nothing to reconcile them (a monitor=false dir never rescans).
            reroot = not already_ids

            for claim in claims:
                if claim.source_id in already_ids:
                    tally.already_present.append(claim.source_id)
                    # fresh_signatures: this drop is not the periodic pass, so
                    # the scan cache holds what that pass last saw, not what is
                    # on disk now.
                    if self._reconciler._refresh_claim(claim, fresh_signatures=True):
                        tally.refreshed.append(claim.source_id)
                        yield ("progress", len(tally.added), str(claim.primary_path))
                    else:
                        tally.failed.append(
                            (
                                str(claim.primary_path),
                                "could not rebuild (see server log); the "
                                "previously registered source is still served",
                            )
                        )
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
                        tally.added.append(claim.source_id)
                        yield ("progress", len(tally.added), str(claim.primary_path))
                    else:
                        tally.failed.append(
                            (
                                str(claim.primary_path),
                                "could not open or register (see server log)",
                            )
                        )

                if should_cancel is not None and should_cancel():
                    break

            yield ("result", tally)
        finally:
            self._catalog_lock.release()

    def _deregister_vanished_under(
        self, root: str, discovered_ids: Set[str]
    ) -> List[str]:
        """Deregister sources under ``root`` whose files are no longer there.

        Scoped to the drop, deliberately: the periodic reconcile's diff is
        whole-catalog (``current_ids - discovered_ids``), so running it against a
        subtree walk would deregister every source outside the drop.

        Absence from the walk is not on its own evidence of deletion -- an
        adapter can decline a claim it once made (a sequence directory worn down
        to a single file), and a drop has none of the stability gating the
        periodic path removes under. So a source goes only when its primary path
        is gone from the filesystem, which is the case #944 reports.
        """
        # `root` is what resolve_local_path returned in add_local_source; a
        # second, differently-spelled canonicalization here is what drifts (#947).
        root_path = Path(root)

        removed: List[str] = []
        for source_id, claim in self._reconciler.claim_items():
            if source_id in discovered_ids or is_remote_url(claim.primary_path):
                continue
            # Existence first: it is one stat and rejects nearly every source in
            # the catalog, where resolve() is an lstat per path component.
            if os.path.exists(claim.primary_path):
                continue
            try:
                primary = Path(claim.primary_path).resolve(strict=False)
            except OSError:
                continue
            if not primary.is_relative_to(root_path):
                continue
            if self._reconciler._commit_remove_source(source_id):
                removed.append(source_id)
                logger.info("Deregistered source %s: %s is gone", source_id, primary)
        return removed

    def _catalog_url_for(self, source_id: str) -> Optional[str]:
        """The registered source's catalog ``source_url`` (None if missing)."""
        adapter = self._server.sources.get(source_id)
        return adapter.catalog_url if adapter is not None else None


def create_source_manager(
    server: TensorFlightServer,
    registry: AdapterRegistry,
    monitored_sources: Optional[List[SourceConfig]] = None,
    static_sources: Optional[List[SourceConfig]] = None,
    metadata_db: Optional[MetadataDatabase] = None,
    credentials_config: Optional[Any] = None,
    stability_window: float = 30.0,
    full_rescan_interval: float = 3600.0,
    aggressive_dir_pruning: bool = False,
    prune_unseen_days: int = 0,
    rescan_interval: float = 30.0,
) -> SourceManager:
    """Create a SourceManager for all configured sources.

    Handles both static sources (explicit config, registered once) and monitored
    sources (filesystem-discovered, kept live by the rescan loop). Both paths use
    the same DiscoveryState/callback machinery. Remote sources are never
    filesystem-watched: a bare-host ``grpc://`` upstream is monitored through the
    background catalog re-list, and any other remote source is registered
    statically during initial discovery.

    Always returns a manager. An empty catalog is a valid runtime state --
    sources arrive later through runtime add_source (napari drag-drop), DoPut
    uploads, or a monitored dir that fills after startup -- so a config naming
    no usable source yields an empty manager the launcher serves (health
    SERVING, empty list_flights) rather than a refusal to boot.

    Args:
        server: TensorFlightServer the sources are registered into.
        registry: AdapterRegistry used for claim detection and adapter creation.
        monitored_sources: SourceConfig entries with monitor=True.
        static_sources: Explicit SourceConfig entries (monitor=False).
        metadata_db: the catalog this manager writes as sources are added and
            removed. It is the only writer: the server registers, the reconciler
            catalogues. None leaves registered sources absent from every browse.
        credentials_config: CredentialsConfig for remote storage authentication.
        stability_window: Seconds an entry's signature must be unchanged before
            it is eligible to be claimed.
        full_rescan_interval: Seconds between full tree walks; the rescans in
            between prune stable and cloud subtrees. <= 0 disables force-full.
        aggressive_dir_pruning: Whether the scanner may skip a directory whose
            own signature is unchanged without descending into it.
        prune_unseen_days: Days of absence after which annotations for a missing
            source are auto-pruned; 0 disables auto-prune.
        rescan_interval: Seconds between rescans. <= 0 leaves the manager with
            no rescan loop, so :meth:`SourceManager.start` no-ops and the caller
            drives any scan itself (config ``monitor_mode = "off"``).

    Returns:
        A SourceManager, empty if no source is usable.
    """
    monitored_sources = monitored_sources or []
    static_sources = static_sources or []

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
    # (its bare-host expansion was skipped), but the re-list populates it once it
    # is reachable -- so it counts as "something to serve" and the catalog is not
    # reported as empty on its account.
    if not monitored_dirs and not static_sources and not monitored_upstreams:
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
        monitored_dirs=monitored_dirs,
        rescan_interval=rescan_interval,
        metadata_db=metadata_db,
        credentials_config=credentials_config,
        stability_window=stability_window,
        full_rescan_interval=full_rescan_interval,
        aggressive_dir_pruning=aggressive_dir_pruning,
        cloud_roots=cloud_roots,
        monitored_upstreams=monitored_upstreams,
        prune_unseen_days=prune_unseen_days,
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
        # Same rebuild drops `alias`, which a tensor-server proxy needs as the
        # display authority of its catalog source_url (biopb/biopb#788). The
        # source_id is already namespaced by then, so this is display-only.
        if source.alias:
            extra_config["alias"] = source.alias
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
            extra_config=extra_config,
            # A static source explicitly flagged cloud is always deferred: the
            # user said "don't open it eagerly". If it is in fact resident, the
            # first access still resolves it cheaply.
            unresolved=bool(source.cloud),
        )
        # source._catalog_url is the alias-derived display tree-root for a local
        # source (resolve.resolve_all_sources), or None. Threaded as the descriptor's
        # source_url override, exactly like the drag-drop re-rooting path.
        manager._reconciler._commit_add_claim(claim, catalog_url=source._catalog_url)

    # Monitored discovery is NOT run synchronously here: under progressive
    # discovery the launcher starts the manager's rescan loop, whose first tick
    # fires immediately, so the (possibly slow) bootstrap scan
    # happens in the background while the server already reports SERVING. A
    # static-only config (no monitored_dirs) has nothing to scan -- the launcher
    # drives the first-scan-complete path directly so it still reports a
    # freshness timestamp and seeds the backlog.
    return manager
