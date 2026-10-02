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

from biopb_tensor_server.adapters.remote_tensor import is_bare_host_upstream_url
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    WalkReport,
    discover_sources,
    generate_source_id,
    local_path_is_rooted,
    resolve_local_path,
)
from biopb_tensor_server.core.errors import UpstreamConfigError
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.sources.entry_stat import entry_change_time, entry_is_quiet
from biopb_tensor_server.sources.reconciler import Reconciler, is_under_cloud_root
from biopb_tensor_server.sources.resolve import _alias_catalog_url, _reroot_catalog_url

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
    dropped_root: str, primary_path: str, *, label: Optional[str] = None
) -> str:
    """Catalog ``source_url`` that re-roots a drag-dropped source under its drop's
    label, under the ``dnd://`` origin scheme.

    The label is the dropped item's basename unless the caller picked another (a
    second drop of a same-named folder gets a distinct one, see
    ``SourceManager._unique_drop_label``). The shared ``_reroot_catalog_url`` keeps
    the source's place beneath the drop::

        drop /home/u/data/exp.zarr           -> "dnd://exp.zarr"        (own root)
        drop /home/u/data/exp/ (a folder) with
             .../exp/a.tif, .../exp/sub/b.tif -> "dnd://exp/a.tif",
                                                 "dnd://exp/sub/b.tif"

    The marker means "user-added from outside every configured root, so nothing
    will re-add it": it is what ``remove_source`` authorizes on, and it is stamped
    only on a drop that lands outside every known root and overlaps nothing.

    Display-only (never ``source_id``, nor the raw ``_source_url`` the filesystem
    uses); the client tree builders strip the scheme for display. The
    configured-``alias`` re-root shares ``_reroot_catalog_url`` but is always
    scheme-less, so the two re-root paths stay distinguishable.
    """
    dropped_root = str(dropped_root).rstrip("/\\")
    if label is None:
        label = os.path.basename(dropped_root) or dropped_root
    return DND_URL_PREFIX + _reroot_catalog_url(label, dropped_root, primary_path)


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
    # Offline placeholder files the walk passed over because the drop was not a
    # cloud root; non-zero means the import is incomplete.
    skipped_offline: int = 0


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
        rescan_interval: float = 120.0,
        metadata_db: Optional[MetadataDatabase] = None,
        credentials_config: Optional[Any] = None,
        stability_window: float = 30.0,
        full_rescan_interval: float = 3600.0,
        cloud_roots: Optional[Set[Path]] = None,
        monitored_upstreams: Optional[List[SourceConfig]] = None,
        scan_once_sources: Optional[List[SourceConfig]] = None,
        monitored_aliases: Optional[Dict[Path, str]] = None,
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
        # Monitored roots that could not be listed on the last walk, so a change
        # of state is logged once, not every tick.
        self._unavailable_roots: Set[Path] = set()
        # Resolved monitored root -> configured ``alias``: the display root every
        # source discovered under it is registered beneath.
        self._monitored_aliases: Dict[Path, str] = monitored_aliases or {}
        self._cloud_roots: Set[Path] = cloud_roots or set()
        # Drops from outside every known root: ``dnd://`` label -> the dropped
        # path, and the cloud roots such a drop consented to, so deregistering the
        # drop takes its consent back with it.
        self._dropped_roots: Dict[str, Path] = {}
        self._drop_cloud_consent: Set[Path] = set()
        self._monitored_upstreams: List[SourceConfig] = monitored_upstreams or []
        # Configured directories that are catalogued but not watched
        # (``monitor = false``): each is scanned once, by the first tick, and
        # never again. Consumed by :meth:`_scan_pending_roots`.
        self._scan_once_pending: List[SourceConfig] = scan_once_sources or []
        # The same directories as persistent roots (path -> ``alias``), so a drop
        # inside one is recognized after the first tick has consumed the list.
        self._scan_once_roots: Dict[Path, Optional[str]] = {
            source.local_path: source.alias
            for source in (scan_once_sources or [])
            if source.local_path is not None
        }

        # Scan tuning. Nothing is kept between scans: every rescan walks the
        # monitored roots afresh (the one walker, ``discover_sources``), so a root
        # costs a stat per entry per tick -- the price of having no snapshot to
        # prune against, and why a very large directory should not be monitored.
        # Cloud roots are the exception: walked only on the periodic full pass.
        self._stability_window = stability_window
        self._full_rescan_interval = full_rescan_interval
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
        # in step with it. The loop always runs: the first scan, and the startup
        # protocol it completes, happen on its first tick. The interval is floored
        # at 0.1s -- a config value below that is almost certainly a mistake, and
        # without a floor it would drive back-to-back full tree walks.
        self._rescan_interval = max(0.1, rescan_interval)
        self._next_rescan_at: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Startup protocol, and the precache routing gate it drives.
        # ``_initial_scan_done`` flips at the end of the first tick -- after every
        # configured source (one-shot directories, monitored walk, upstream
        # mirror) has had its first pass -- which runs in the event loop *after*
        # start(), so it, not "are we past start()", is the startup/runtime
        # boundary. Sources committed before it route to the slow precache
        # backlog; after it, ``_on_source_committed`` prompt-enqueues them.
        # Event-loop thread only, so a plain flag is safe.
        self._initial_scan_done = False
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
        # delegates every catalog mutation to it. Its injected seam reads back
        # into here: ``notify_source_committed`` for the precache gate.
        self._reconciler = Reconciler(
            server=server,
            registry=registry,
            discovery_state=discovery_state,
            metadata_db=metadata_db,
            credentials_config=credentials_config,
            monitored_dirs=monitored_dirs,
            cloud_roots=self._cloud_roots,
            notify_source_committed=self._notify_source_committed,
            catalog_url_for=self._monitored_catalog_url,
            stability_window=stability_window,
        )

    def _monitored_catalog_url(self, claim: SourceClaim) -> Optional[str]:
        """Display ``source_url`` for a source discovered under a monitored root
        that has an ``alias``; ``None`` when there is none.

        The innermost aliased root wins. Display-only, like a drop's re-root: the
        source id still hashes the native path.
        """
        if not self._monitored_aliases:
            return None
        path = Path(claim.primary_path)
        best = max(
            (r for r in self._monitored_aliases if r == path or r in path.parents),
            key=lambda r: len(r.parts),
            default=None,
        )
        if best is None:
            return None
        return _alias_catalog_url(
            self._monitored_aliases[best], str(best), claim.primary_path
        )

    def start(self) -> None:
        """Start the rescan loop.

        The first scan runs in this loop, whose first tick fires immediately --
        even for a config with nothing to scan, where that tick just completes
        the startup protocol. Until it does, ``_initial_scan_done`` stays False
        so its sources route to the precache backlog rather than the prompt
        enqueue.
        """
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
        startup catalog. Fired from the event-loop thread.
        """
        self._on_initial_scan_complete = callback

    def _mark_catalog_complete(self) -> None:
        """Publish that a full scan just finished, and run the orphan clock.

        The paths that can complete one -- the first tick, a later forced full
        rescan, and a later upstream re-list pass -- all end here, so the clock is
        driven by the event, not by a timer.

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
        """Advance the startup protocol: stamp catalog freshness, clear
        ``full_scan_in_progress``, flip the precache startup gate, and fire the
        first-scan-complete hook.

        Called at the end of the first tick, whatever it scanned (nothing, for a
        config of static sources only). Idempotent: the hook fires only on the
        transition to done.
        """
        self._mark_catalog_complete()
        self._server.set_full_scan_in_progress(False)
        if not self._initial_scan_done:
            self._initial_scan_done = True
            logger.info(
                "Initial scan complete: %d sources, %.0f s after start",
                len(self._server.sources),
                time.monotonic() - self._started_at,
            )
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
        """Run one tick: the one-shot directories, the monitored walk, then the
        due upstream re-lists.

        The first tick is the first scan. It leaves ``full_scan_in_progress`` set
        (the launcher raised it before :meth:`start`) and ends by completing the
        startup protocol, so the flag clears, the precache gate opens and the
        freshness stamp advances only once every source has had its first pass --
        an unreachable upstream delays that by one failed attempt, not
        indefinitely. Everything registered before then, the upstream mirror
        included, is startup set and routes to the precache backlog.
        """
        # Serialize the whole pass against a concurrent runtime add_local_source
        # (Flight thread) so the two never mutate the confirmed catalog at once.
        with self._catalog_lock:
            try:
                self._scan_pending_roots()
                self._rescan_monitored_dirs()
                # tensor-server upstream re-list (biopb/biopb#178): adaptive
                # per-upstream cadence -- fast (every tick) while changing/failing,
                # backing off toward full_rescan_interval while a source set stays
                # stable.
                self._reconcile_due_upstreams()
            except BaseException:
                if not self._initial_scan_done:
                    # The next tick retries the first scan; until it completes
                    # nothing is scanning, so do not leave the flag claiming so.
                    self._server.set_full_scan_in_progress(False)
                raise
            if not self._initial_scan_done:
                self.complete_initial_scan()

    def _scan_pending_roots(self) -> None:
        """Scan the configured ``monitor = false`` directories, each once.

        Runs ahead of the monitored walk on the first tick, so what it registers
        is startup set (precache backlog, not the live enqueue) and the catalog
        grows behind SERVING exactly as a monitored directory's does. A claim
        registered here sits outside every monitored root, so the rescan's
        removal diff never sees it; a later drop of the directory picks up
        changes, as it does for any root.

        Caller holds ``_catalog_lock``.
        """
        pending, self._scan_once_pending = self._scan_once_pending, []
        for source in pending:
            try:
                self._scan_configured_root(source)
            except Exception:
                # One bad root must not cost the others.
                logger.exception("Could not scan configured directory %s", source.url)

    def _scan_configured_root(self, source: SourceConfig) -> None:
        """Register everything under one configured directory (see
        :meth:`_register_root`)."""
        url = resolve_local_path(source.url)
        if not os.path.isdir(url):
            logger.warning("Configured directory does not exist: %s", source.url)
            return
        # The config-line analogue of a drag-dropped folder becoming its own root:
        # an `alias` re-roots everything found under it. Persistent here, because
        # nothing rescans the root to re-merge it into the shared path tree.
        alias = source.alias
        for event in self._register_root(
            url,
            catalog_url_for=lambda claim: (
                _alias_catalog_url(alias, url, claim.primary_path) if alias else None
            ),
            cloud=source.cloud,
            dataset=source.dataset,
        ):
            if event[0] == "result":
                for path, reason in event[1].failed:
                    logger.warning("Configured directory %s: %s: %s", url, path, reason)

    def _rescan_monitored_dirs(self) -> None:
        """Walk the monitored directories and reconcile the discovered catalog.

        No-op for an upstream-only config (no monitored dirs). On the first tick
        the progress flag and freshness stamp are left to
        :meth:`complete_initial_scan`, which runs after the upstream pass.
        """
        if not self._monitored_dirs:
            return

        force_full_rescan = self._should_force_full_rescan()
        # Progressive-discovery freshness signals: while a *full* reconcile runs,
        # the health action reports full_scan_in_progress=True; on success it
        # advances last_full_scan_finished_at. Incremental rescans leave both
        # untouched (they skip cloud roots, so they are not a whole-catalog
        # reconcile). The first tick's pass is only part of the first scan, so it
        # leaves both to complete_initial_scan; a later one resets them in the
        # outer finally.
        startup = not self._initial_scan_done
        completes_here = force_full_rescan and not startup
        if force_full_rescan:
            self._server.set_full_scan_in_progress(True)
        try:
            # Progressive population: on the *first* full scan, register each
            # source the moment the walk claims it rather than batching every add
            # into the end-of-walk reconcile, so the catalog grows within the
            # walk. Safe only for the first scan -- it starts empty and
            # force-full, so there are no removals to diff and every claim is a
            # pure add. The stability gate runs inside the walk, so unstable
            # entries are never claimed and therefore never streamed; the next
            # rescan picks them up. The end-of-walk reconcile below still runs and
            # is idempotent for streamed adds.
            stream_first_scan = force_full_rescan and startup
            discovered_state = DiscoveryState()
            if stream_first_scan:
                discovered_state.on_source_added = (
                    self._reconciler._stream_first_scan_add
                )

            # One walk per monitored root, into one state (the identity set it
            # carries stops overlapping roots from claiming a subtree twice). A
            # cloud root is enumerated only on the full pass: listing one is
            # expensive and its mtimes are unreliable, so the incremental ticks
            # leave its sources registered (the reconcile scopes them out) rather
            # than re-walking it.
            report = WalkReport()
            for monitored_dir in sorted(self._monitored_dirs):
                # Walked as stored, not resolved again: the root is canonical
                # from config, so its claims are spelled under it however the
                # path has changed since (a migration may have left a link).
                root = monitored_dir
                cloud = root in self._cloud_roots
                if cloud and not force_full_rescan:
                    continue
                if not self._root_is_listable(root):
                    report.declined_dirs.add(str(root))
                    continue
                discover_sources(
                    root,
                    self._registry,
                    discovered_state,
                    path_filter=self._should_claim,
                    admit_nonresident=cloud,
                    cloud_root=cloud,
                    report=report,
                )

            # A directory the walk declined (the stability gate, or the skip
            # policy) is not evidence that what is registered under it is gone.
            self._reconciler._preserve_skipped_claims(
                discovered_state, report.declined_dirs
            )
            self._reconciler._reconcile_discovered_state(
                discovered_state, force_full=force_full_rescan
            )

            if completes_here:
                self._mark_catalog_complete()
        finally:
            if completes_here:
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

    def _root_is_listable(self, root: Path) -> bool:
        """Whether a monitored root can be walked now; logs each change of state.

        An unmounted drive, a share that is down and a deleted directory look the
        same from here, so a root that cannot be listed is not evidence that what
        is registered under it is gone: the caller records it as declined, its
        sources stay, and it is walked again as soon as it is back.
        """
        if root.is_dir():
            if root in self._unavailable_roots:
                self._unavailable_roots.discard(root)
                logger.info("Monitored directory is available again: %s", root)
            return True
        if root not in self._unavailable_roots:
            self._unavailable_roots.add(root)
            logger.warning(
                "Monitored directory is not available; keeping its sources until it "
                "is back: %s",
                root,
            )
        return False

    def _should_force_full_rescan(self) -> bool:
        """Whether a full pass (the only one that walks cloud roots) is due."""
        if self._full_rescan_interval <= 0:
            return False
        return time.time() - self._last_full_rescan_at >= self._full_rescan_interval

    def _should_claim(self, path: Path) -> bool:
        """Stability gate: may this entry be claimed -- and, for a directory,
        entered -- on this pass?

        Claiming a half-written file registers a wrong descriptor, and for a
        format whose type marker is written last (OME-TIFF) a wrong
        ``source_type``, hence a different ``source_id``. So an entry is eligible
        only once it has been unchanged for the stability window, judged by a stat
        taken now (nothing is cached between scans). A pure read of the
        filesystem; the removal shield asks the same ``entry_is_quiet`` question.

        Cloud/synced-folder entries bypass the window entirely (cloud-storage
        phase 2): mtime/ctime age is unreliable there (doc S1.2), so a placeholder
        could never stabilize, and archived dehydrated data is never mid-write.
        """
        if self._is_under_cloud_root(str(path)):
            return True
        try:
            stat_result = os.stat(path)
        except OSError:
            return False
        now = time.time()
        return entry_is_quiet(
            entry_change_time(stat_result, now), now, self._stability_window
        )

    def _is_under_cloud_root(self, path: str) -> bool:
        """True when *path* is a cloud-opted root or lives under one.

        Consulted by the stability gate (:meth:`_should_claim`); shares
        the cloud-membership rule with the Reconciler via the module free
        function so neither object depends on the other.
        """
        return is_under_cloud_root(self._cloud_roots, path)

    def _notify_source_committed(self, source_id: str) -> None:
        """Precache routing gate for a freshly committed source (injected into
        the Reconciler as ``notify_source_committed``).

        Only live additions -- those committed after the initial scan completes --
        are prompt-enqueued; the startup set routes to the slow backlog instead.
        This manager owns the startup state that decides that, so the gate lives
        here rather than in the Reconciler. Best-effort: a hook failure must never
        abort a source commit.
        """
        if self._initial_scan_done and self._on_source_committed is not None:
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
            # Taking the drop away takes its label and its cloud consent too; a
            # later plain drop of the same folder must not inherit either.
            dropped = self._dropped_roots.pop(
                root_url[len(DND_URL_PREFIX) :].strip("/\\"), None
            )
            if dropped is not None and dropped in self._drop_cloud_consent:
                self._drop_cloud_consent.discard(dropped)
                self._cloud_roots.discard(dropped)
        return removed, failed

    def _reconcile_due_upstreams(self) -> None:
        """Re-list the upstreams that are due this tick (adaptive backoff).

        Each upstream re-lists on the fast rescan tick while it is changing or
        failing, and backs off (period doubles per unchanged re-list, capped at
        full_rescan_interval) while it is stable -- so a stable lab store is not
        queried every 30s forever, yet a new source / a recovered upstream is
        mirrored within ~one tick. When there are no monitored *dirs* each pass
        is the whole reconcile, so a later one advances catalog freshness itself.
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

        try:
            for upstream in due:
                self._reconcile_and_reschedule(upstream)
        finally:
            if not self._monitored_dirs and self._initial_scan_done:
                # Each completed pass re-verifies the (remote) catalog -> advance
                # freshness. The first tick leaves that to complete_initial_scan.
                self._mark_catalog_complete()

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
        cloud: bool = False,
    ):
        """Register ``url`` (a path on the server) as source(s) at runtime.

        Generator, driving the "add_source" Flight action / tensor-browser
        drag-drop. It yields event tuples the caller maps onto the wire:

        - ``("progress", added_count, current_path)`` -- one per source as it
          registers or refreshes (the count is of NEW sources, so a pure re-drop
          advances only the path),
        - ``("result", tally)`` -- exactly one terminal
          :class:`AddSourceTally`.

        ``cloud`` treats ``url`` as a cloud / synced folder: offline placeholders
        are admitted and registered unresolved (content read on first access)
        instead of skipped, and multi-file grouping is off. Without it the walk
        skips placeholders and the tally's ``skipped_offline`` says how many.

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

        Where the drop lands decides what it may do. **Inside a known root** -- a
        monitored directory, a configured ``monitor = false`` directory, or an
        earlier drop's root, the root itself included -- it registers and refreshes
        as a rescan of that root would, in that root's display tree (its alias if
        it has one) and with **no new** ``dnd://`` mark; marks already on sources
        stay. **Outside every known root** it becomes a root of its own, with
        a unique ``dnd://`` label, but only if it overlaps nothing already
        registered; otherwise it is refused. Drops are refused until the first scan
        has finished, since both checks need the whole catalog.

        Whole-request problems raise before the first yield:
        ``FileNotFoundError`` / ``PermissionError`` (server-side path check) or
        ``ValueError`` (a remote URL -- runtime add is local-only for now -- a
        relative path, which has no anchor on this side of the wire, a first scan
        still running, or cloud mode asked for inside a known root).
        """
        if not self._initial_scan_done:
            raise ValueError(
                "The catalog is still being indexed; try again when the first scan "
                "has finished"
            )
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

        root_path = Path(url)
        known = self._known_root(root_path)
        if known is not None and cloud and not self._is_under_cloud_root(url):
            # Cloud is a property of a root, set where the root is first added; a
            # subfolder cannot turn it on afterwards, and there would be no drop
            # to deregister to take it back.
            raise ValueError(
                f"Cannot switch cloud mode on inside {known[1]}: it is set where "
                "the folder is first added, or in the config"
            )

        # Acquire the catalog lock, heart-beating while a rescan holds it so a
        # long wait does not sit silent long enough to trip a proxy timeout.
        while not self._catalog_lock.acquire(timeout=_ADD_SOURCE_ACQUIRE_HEARTBEAT):
            yield ("progress", 0, "waiting for catalog scan to finish")
        consented = False
        committed = False
        try:
            # Cloud-ness belongs to the root, not to this call: once a root is
            # cloud, the stability gate, the deferred registration, the precache
            # residency check and the reconcile's cloud scoping all ask the same
            # question of every path under it. So a consented drop records its
            # root, and it stays cloud until that drop is deregistered. A path
            # already under a configured cloud root is cloud whatever the request
            # says.
            if cloud and not self._is_under_cloud_root(url):
                self._cloud_roots.add(root_path)
                self._drop_cloud_consent.add(root_path)
                consented = True
            cloud = self._is_under_cloud_root(url)

            label = None if known is not None else self._unique_drop_label(root_path)
            for event in self._register_root(
                url,
                source_type=source_type,
                should_cancel=should_cancel,
                catalog_url_for=lambda claim: self._drop_catalog_url_for(
                    known, label, url, claim
                ),
                cloud=cloud,
                reject_overlap=known is None,
            ):
                if event[0] == "progress":
                    committed = committed or event[1] > 0
                else:
                    committed = bool(event[1].added)
                yield event
        finally:
            if known is None:
                if committed:
                    self._dropped_roots[label] = root_path
                elif consented:
                    self._drop_cloud_consent.discard(root_path)
                    self._cloud_roots.discard(root_path)
            self._catalog_lock.release()

    def _known_root(self, path: Path) -> Optional[Tuple[str, Path]]:
        """The innermost configured or dropped root that ``path`` is at or under.

        Lexical, on the resolved drop path against roots resolved once: a monitored
        directory, a ``monitor = false`` directory, or the root of an earlier drop.
        Returns ``(kind, root)`` or None.
        """
        roots = [
            *(("monitored", r) for r in self._monitored_dirs),
            *(("scan_once", r) for r in self._scan_once_roots),
            *(("dropped", r) for r in self._dropped_roots.values()),
        ]
        inside = [(kind, root) for kind, root in roots if path.is_relative_to(root)]
        return max(inside, key=lambda kr: len(kr[1].parts), default=None)

    def _unique_drop_label(self, root: Path) -> str:
        """The ``dnd://`` label for a drop from outside every known root.

        The basename, with a counter when another drop already holds it, so two
        same-named folders from different parents are two roots, not one. No ``/``,
        so one label is never a prefix of another's urls.
        """
        base = root.name or str(root)
        label, n = base, 2
        while label in self._dropped_roots:
            label, n = f"{base} ({n})", n + 1
        return label

    def _drop_catalog_url_for(
        self,
        known: Optional[Tuple[str, Path]],
        label: Optional[str],
        url: str,
        claim: SourceClaim,
    ) -> Optional[str]:
        """Display ``source_url`` for a source a drop adds.

        Inside a known root it joins that root's display tree, with no new mark:
        the monitored root's alias, the ``monitor = false`` root's alias, or an
        earlier drop's label without its ``dnd://`` scheme (the clients strip the
        scheme for display, so it sits beside the marked sources; the scheme is the
        removal key, so only those stay removable). With no alias it is None, which
        leaves the plain file url. Outside, the drop's own marked ``dnd://`` root.
        """
        if known is None:
            return _drop_catalog_url(url, claim.primary_path, label=label)
        kind, root = known
        if kind == "monitored":
            return self._monitored_catalog_url(claim)
        if kind == "dropped":
            earlier = next(lab for lab, r in self._dropped_roots.items() if r == root)
            return _reroot_catalog_url(earlier, str(root), claim.primary_path)
        alias = self._scan_once_roots.get(root)
        if alias:
            return _alias_catalog_url(alias, str(root), claim.primary_path)
        return None

    def _register_root(
        self,
        url: str,
        *,
        source_type: str = "",
        should_cancel: Optional[Callable[[], bool]] = None,
        catalog_url_for: Callable[[SourceClaim], Optional[str]],
        cloud: bool = False,
        dataset: Optional[str] = None,
        reject_overlap: bool = False,
    ):
        """Claim everything at or under ``url`` and bring the catalog in line.

        The one primitive a drop and a one-shot configured directory share:
        containment guard, ``discover_sources`` into a scratch state, remove what
        is gone under the root, then per claim refresh-if-known else add. The
        caller holds ``_catalog_lock`` and has checked that ``url`` is a rooted,
        readable local path. Yields the events :meth:`add_local_source` documents.

        ``catalog_url_for`` gives a NEW claim its display ``source_url`` override,
        or None. ``cloud`` scans ``url`` as a cloud root; ``dataset`` fills in an
        HDF5 claim that needs one. ``reject_overlap`` refuses the whole root, before
        anything is removed or committed, when it holds a source that is already
        registered or an existing source lies under it: a root that is its own
        display root must not share sources with another.
        """
        is_dir = os.path.isdir(url)
        tally = AddSourceTally()

        # Containment check (case 4): if a STRICT ancestor of the drop is
        # already owned by a source, the drop is *inside* that source. The
        # exact-path member dedup in DiscoveryState.add_claim does not catch
        # this (dir sources record only the dir as a member), so reject here.
        owner = self._reconciler._find_containing_source(url)
        if owner is not None:
            tally.failed.append((url, f"already part of source '{owner}'"))
            yield ("result", tally)
            return

        # A dataset is claimed in place; a plain folder is walked recursively
        # (case 5). The large-drop footgun guard lives client-side (the tensor
        # browser confirms before sending an oversized folder): drag-drop is
        # localhost-only, so the client shares this filesystem and can size the
        # tree before any scan is sent. A direct SDK caller passing a path is
        # explicit intent, so the walk is not gated here. Discovery runs into a
        # scratch state, so it never mutates the confirmed catalog until a claim
        # is committed.
        report = WalkReport()
        scratch = discover_sources(
            Path(url),
            self._registry,
            DiscoveryState(),
            admit_nonresident=cloud,
            cloud_root=cloud,
            report=report,
        )
        claims: List[SourceClaim] = list(scratch.claims.values())
        tally.skipped_offline = report.offline_files

        # Assign identity to every claim up front so the overlap check below
        # can see the whole drop before any of it is committed.
        for claim in claims:
            if source_type:
                claim.source_type = source_type
            if (
                dataset
                and claim.source_type == "hdf5"
                and claim.extra_config.get("needs_dataset")
            ):
                claim.extra_config["dataset"] = dataset
            if not claim.source_id:
                claim.source_id = generate_source_id(
                    str(claim.primary_path), claim.source_type
                )

        already_ids = {
            claim.source_id
            for claim in claims
            if self._reconciler.has_claim(claim.source_id)
        }

        if reject_overlap:
            root_path = Path(url)
            overlapping = already_ids or any(
                Path(path).is_relative_to(root_path)
                for _, path in self._reconciler.local_claim_paths()
            )
            if overlapping:
                tally.failed.append(
                    (
                        url,
                        "overlaps sources that are already registered; drop a "
                        "narrower path, or remove them first",
                    )
                )
                yield ("result", tally)
                return

        # Removal half, before the empty-drop bail-out below: dropping a
        # folder whose contents were deleted is exactly how a stale entry
        # gets noticed, and there is nothing to add in that case. A dropped
        # *file* skips the O(catalog) scan -- its own existence was checked
        # above, and it has no descendants that could have vanished.
        tally.removed = (
            self._remove_unclaimed_under(
                url, {claim.source_id for claim in claims}, report.declined_dirs
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

        for claim in claims:
            if claim.source_id in already_ids:
                tally.already_present.append(claim.source_id)
                if self._reconciler._refresh_claim(claim):
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
                catalog_url = catalog_url_for(claim)
                if self._reconciler._commit_add_claim(claim, catalog_url=catalog_url):
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

    def _remove_unclaimed_under(
        self, root: str, discovered_ids: Set[str], declined_dirs: Set[str]
    ) -> List[str]:
        """Remove the sources under ``root`` that this walk did not find again.

        Scoped to the root, deliberately: the periodic reconcile's diff is
        whole-catalog (``current_ids - discovered_ids``), so running it against a
        subtree walk would deregister every source outside the drop.

        A source is removed when it is under ``root`` and

        * the rescan will not do it for us -- a source under a monitored root is
          left to the rescan, whose two-scan rule suits a source that is only
          briefly unclaimable; a drop outside every monitored root has no later
          pass, so waiting would mean never;
        * the walk did not decline the directory it lives in (the skip policy
          never entered it, so absence says nothing);
        * its own paths are quiet, the same test the rescan's removal applies; and
        * an adapter does not claim it on a second look. An adapter can briefly
          decline a claim it made (a sidecar being rewritten, a locked header),
          and with no later pass a transient miss would cost a working source.
          The re-probe is one claim per missing source.

        A source whose primary path is gone skips the re-probe: nothing to claim.
        """
        root_path = Path(root)
        declined = [Path(d) for d in declined_dirs]

        removed: List[str] = []
        for source_id, claim in self._reconciler.claim_items():
            if source_id in discovered_ids or is_remote_url(claim.primary_path):
                continue
            # Lexical, so nearly every source in the catalog is rejected
            # without touching the filesystem, and a link is under the root it was
            # found in wherever it points.
            primary = Path(claim.primary_path)
            if not primary.is_relative_to(root_path):
                continue
            if self._reconciler._is_monitored_claim(claim):
                continue
            if any(primary.is_relative_to(d) for d in declined):
                continue
            if not self._reconciler._claim_is_quiet(claim):
                continue
            if os.path.exists(claim.primary_path) and self._claimed_again(claim):
                continue
            if self._reconciler._commit_remove_source(source_id):
                removed.append(source_id)
                logger.info(
                    "Deregistered source %s: no longer found under %s", source_id, root
                )
        return removed

    def _claimed_again(self, claim: SourceClaim) -> bool:
        """Whether an adapter claims ``claim``'s primary path right now."""
        try:
            again = self._registry.get_claims_for_path(
                ClaimContext(
                    Path(claim.primary_path),
                    cloud_root=self._is_under_cloud_root(claim.primary_path),
                ),
                DiscoveryState(),
            )
        except Exception:
            return False
        return any(c.primary_path == claim.primary_path for c in again)

    def _catalog_url_for(self, source_id: str) -> Optional[str]:
        """The registered source's catalog ``source_url`` (None if missing)."""
        adapter = self._server.sources.get(source_id)
        return adapter.catalog_url if adapter is not None else None


def create_source_manager(
    server: TensorFlightServer,
    registry: AdapterRegistry,
    monitored_sources: Optional[List[SourceConfig]] = None,
    static_sources: Optional[List[SourceConfig]] = None,
    scan_once_sources: Optional[List[SourceConfig]] = None,
    metadata_db: Optional[MetadataDatabase] = None,
    credentials_config: Optional[Any] = None,
    stability_window: float = 30.0,
    full_rescan_interval: float = 3600.0,
    prune_unseen_days: int = 0,
    rescan_interval: float = 120.0,
) -> SourceManager:
    """Create a SourceManager for all configured sources.

    Handles static sources (explicit config, registered once, nothing to walk),
    monitored sources (filesystem-discovered, kept live by the rescan loop) and
    one-shot directories (discovered by the first tick, then left alone). All
    three use the same DiscoveryState/callback machinery. Remote sources are never
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
        static_sources: Explicit SourceConfig entries (a typed source, a file, a
            single remote source): nothing to discover, so nothing to walk.
        scan_once_sources: Local directories with ``monitor=False``; each is
            walked once by the first rescan tick and never rescanned.
        metadata_db: the catalog this manager writes as sources are added and
            removed. It is the only writer: the server registers, the reconciler
            catalogues. None leaves registered sources absent from every browse.
        credentials_config: CredentialsConfig for remote storage authentication.
        stability_window: Seconds an entry's signature must be unchanged before
            it is eligible to be claimed.
        full_rescan_interval: Seconds between full passes, the only ones that
            walk a cloud root. <= 0 disables force-full.
        prune_unseen_days: Days of absence after which annotations for a missing
            source are auto-pruned; 0 disables auto-prune.
        rescan_interval: Seconds between rescans (floored at 0.1s).

    Returns:
        A SourceManager, empty if no source is usable.
    """
    monitored_sources = monitored_sources or []
    static_sources = static_sources or []
    scan_once_sources = scan_once_sources or []

    # ``monitored_sources`` holds the watched local directories and the bare-host
    # tensor-server upstreams ("mirror everything", biopb/biopb#178).
    monitored_dirs: Set[Path] = set()
    monitored_aliases: Dict[Path, str] = {}
    monitored_upstreams: List[SourceConfig] = []
    for source in monitored_sources:
        if source.is_remote:
            if is_bare_host_upstream_url(source.url):
                logger.info(
                    "Tensor-server upstream %s: catalog re-listed in the background, "
                    "not filesystem-watched",
                    source.url,
                )
                monitored_upstreams.append(source)
            else:
                # Nothing to re-list or watch; the caller should have registered it.
                logger.warning(
                    "Remote source %s is not a bare-host upstream; not monitored",
                    source.url,
                )
            continue

        local_path = source.local_path
        if local_path is None:
            logger.warning(f"Cannot monitor path: {source.url}")
            continue

        # A path that is not there yet stays: the walk keeps it and picks it up
        # when it appears.
        if local_path.is_file():
            logger.warning(f"Cannot monitor single file: {source.url}")
            continue

        monitored_dirs.add(local_path)
        if source.alias:
            monitored_aliases[local_path] = source.alias

    if (
        not monitored_dirs
        and not static_sources
        and not scan_once_sources
        and not monitored_upstreams
    ):
        logger.info("No sources configured yet; serving an empty catalog")

    # EXPERIMENTAL: cloud/synced-folder mode. The walk admits dehydrated entries,
    # and register placeholder adapters resolved lazily on first access.
    cloud_roots: Set[Path] = set()
    for source in (*monitored_sources, *static_sources, *scan_once_sources):
        if source.cloud:
            # Warned once per configured cloud source at startup.
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
        cloud_roots=cloud_roots,
        monitored_upstreams=monitored_upstreams,
        scan_once_sources=scan_once_sources,
        monitored_aliases=monitored_aliases,
        prune_unseen_days=prune_unseen_days,
    )

    # Seed static sources as direct claims (explicit config, no filesystem walk)
    # These are added first so monitored discovery skips paths already claimed.
    for source in static_sources:
        # claim->SourceConfig drops most configs, so we carry some through to call
        # create_from_config.
        extra_config = {}
        if source.dataset:
            extra_config["dataset"] = source.dataset
        if source.credentials_profile:
            extra_config["credentials_profile"] = source.credentials_profile
        if source.alias:  # display-only
            extra_config["alias"] = source.alias
        # Store the canonical resolved form of a local config path, so Claim can
        # be keyed on it.
        primary_path = source.url  # initial assignment; may be resolved if local
        # A remote URL is left verbatim. Aliasing is intentional because the server may
        # use different address/port but serving the same data.
        if not is_remote_url(source.url):
            primary_path = resolve_local_path(source.url)
        claim = SourceClaim(
            source_type=source.type,
            primary_path=primary_path,
            source_id=source.source_id,
            extra_config=extra_config,
            # A static source explicitly flagged cloud is always deferred: the
            # first access still resolves it cheaply.
            unresolved=bool(source.cloud),
        )
        # source._catalog_url is the alias-derived display tree-root for a local
        # source (resolve.resolve_all_sources), or None.
        manager._reconciler._commit_add_claim(claim, catalog_url=source._catalog_url)

    return manager
