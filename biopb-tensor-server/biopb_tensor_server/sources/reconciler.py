"""Confirmed-catalog reconciliation and source lifecycle (biopb/biopb#278 item B).

The :class:`Reconciler` owns the *confirmed catalog* -- the live ``SourceClaim``
set (``DiscoveryState``), the server registration, the metadata-DB rows, and the
derived indices (path->source_id, per-source signatures, cloud-source ids, and
failed-source retry state). It is the single writer of that catalog, reached from
the three desired-state sources, all of which reduce to the same
add/refresh/remove/register primitives:

  * the periodic filesystem rescan  -> :meth:`_reconcile_discovered_state` /
    :meth:`_preserve_skipped_claims` (fed the walk's discovered claims);
  * a runtime drag-drop / SDK add   -> :meth:`_commit_add_claim` /
    :meth:`_refresh_claim` (driven by ``SourceManager.add_local_source``);
  * a tensor-server upstream re-list -> :meth:`_reconcile_one_upstream`.

``SourceManager`` owns the *rescan machinery* (event loop, the filesystem walk,
the stability gate, the startup protocol) and delegates every catalog mutation
here. The seam is deliberately narrow:

  * SourceManager -> Reconciler: the reconcile/commit calls above, plus four
    claim-snapshot accessors (:meth:`claim_items` / :meth:`claim_ids` /
    :meth:`has_claim` / :meth:`local_claim_paths`) for its cleanup and precache
    reads.
  * Reconciler -> SourceManager: one injected callable,
    ``notify_source_committed`` (the precache routing gate, owned by the startup
    state), plus the shared :class:`~biopb_tensor_server.sources.roots.Roots`.

The coarse single-writer mutex (a runtime add vs the periodic rescan) stays in
``SourceManager`` (``_catalog_lock``); the fine-grained state RLock (``_lock``,
taken by the commit primitives) lives here.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter
from biopb_tensor_server.core.adapter_base import to_catalog_url
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    DiscoveryState,
    SourceClaim,
    _is_offline_placeholder,
    resolve_local_path,
)
from biopb_tensor_server.core.errors import (
    SourceRegistrationError,
    SourceUnresolvedError,
    UpstreamConfigError,
)
from biopb_tensor_server.core.normalize import normalize_adapter
from biopb_tensor_server.core.registration_stats import (
    RegistrationStats,
    SyncCost,
)
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.core.source_registry import close_adapter
from biopb_tensor_server.sources.entry_stat import (
    build_entry_signature,
    entry_change_time,
    entry_is_quiet,
)
from biopb_tensor_server.sources.roots import Roots

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import (
        SourceConfig as _SourceConfig,  # noqa: F401
    )
    from biopb_tensor_server.serving.metadata_db import MetadataDatabase
    from biopb_tensor_server.serving.server import TensorFlightServer

logger = logging.getLogger(__name__)


# Consecutive rescans a monitored source may go undiscovered before it is removed.
_MISSES_BEFORE_REMOVAL = 2

# Locks that serialize one source's registration, refresh and removal.
_REGISTRATION_STRIPES = 256

# Remote/cloud source families are EXPERIMENTAL. Warn once per family per process
# (registration runs per source) so an operator sees the maturity caveat without
# per-source log spam. Keyed by family; the set is only ever added to.
_EXPERIMENTAL_WARNED: set = set()
_EXPERIMENTAL_SOURCE_MESSAGES = {
    "cloud": (
        "Cloud / synced-folder sources (cloud=true, e.g. OneDrive Files "
        "On-Demand) are EXPERIMENTAL: resolve-on-serve and hydrate-ahead behavior "
        "may change."
    ),
    "tensor-server": (
        "Remote tensor-server proxy sources (type=tensor-server) are "
        "EXPERIMENTAL: the caching-passthrough proxy may change."
    ),
    "remote-url": (
        "Remote URL sources (s3://, http(s)://, ...) are EXPERIMENTAL and may change."
    ),
}


def _warn_experimental_source(family: str) -> None:
    """Log a one-time EXPERIMENTAL warning for a remote/cloud source *family*."""
    if family in _EXPERIMENTAL_WARNED:
        return
    _EXPERIMENTAL_WARNED.add(family)
    logger.warning("%s", _EXPERIMENTAL_SOURCE_MESSAGES[family])


@dataclass
class _FailureTracker:
    attempts: int = 0
    next_retry_at: float = 0.0
    last_log_at: float = float("-inf")


class Reconciler:
    """Single-writer of the confirmed source catalog. See the module docstring."""

    _retry_backoff_initial = 1.0
    _retry_backoff_max = 60.0
    _failure_log_interval = 30.0
    # A rebuild failing this many times running is presumed permanent (a member
    # file deleted), not a transient DB hiccup -- remove the source rather than
    # serve stale bytes forever on capped backoff.
    _max_refresh_failures = 5

    def __init__(
        self,
        *,
        server: TensorFlightServer,
        registry: AdapterRegistry,
        discovery_state: DiscoveryState,
        metadata_db: Optional[MetadataDatabase],
        credentials_config: Optional[Any],
        roots: Roots,
        notify_source_committed: Callable[[str], None],
        catalog_url_for: Callable[[SourceClaim], Optional[str]] = lambda claim: None,
        stability_window: float = 30.0,
        registration_stats: bool = False,
    ):
        self._server = server
        self._registry = registry
        self._state = discovery_state
        self._metadata_db = metadata_db
        self._credentials_config = credentials_config
        # Shared with SourceManager; read-only here (monitored-claim scoping, cloud
        # policy).
        self._roots = roots
        # Injected SourceManager seam (see module docstring).
        self._notify_source_committed = notify_source_committed
        # Display root for a newly discovered claim (a monitored root's alias);
        # None leaves the native url.
        self._catalog_url_for = catalog_url_for
        # Quiet period a claim must have had before this reconcile will remove or
        # rebuild it -- the same window, and the same predicate, the claim gate
        # applies on the way in (see ``_claim_is_quiet``).
        self._stability_window = stability_window

        # Fine-grained state RLock: rescan/reconcile helpers re-enter other
        # state-mutating helpers, so nested calls on the same thread must not
        # deadlock. The coarse whole-pass mutex lives in SourceManager.
        self._lock = threading.RLock()

        # source_id -> member path signature map used to detect in-place changes.
        self._source_signatures: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
        # source_id -> retry/logging state for repeatedly failing datasets.
        self._failed_sources: Dict[str, _FailureTracker] = {}
        # source_ids whose primary_path is under a cloud root, maintained at commit
        # time (O(1) per source). Lets the incremental reconcile preserve cloud
        # sources by a hash-set check instead of resolving every cloud member path.
        self._cloud_source_ids: Set[str] = set()
        # source_id -> consecutive rescans that did not find it. A source that was
        # claimed can briefly stop being claimable with its files still in place (a
        # sidecar rewritten in place, a locked header); removing it on the first
        # miss would unregister a working source and rebuild it a tick later.
        self._missed_scans: Dict[str, int] = {}

        # Deferred registration. While ``_defer_registration`` is set a claim is
        # committed to the catalog only (a ``pending`` row) and registered later,
        # by a worker or by the first read that needs it. ``_pending`` maps each
        # such source_id to its display url; ``_pending_failed`` maps those among
        # them whose registration raised to the error (they are retried, but are
        # not waited for).
        self._defer_registration = False
        self._pending_hook: Optional[Callable[[str], None]] = None
        self._pending: Dict[str, Optional[str]] = {}
        self._pending_failed: Dict[str, str] = {}
        # A registration, a refresh and a removal of one source never overlap, so
        # a read that races the worker coalesces onto one parse and a removed
        # source is not registered back. A fixed stripe of locks hashed by
        # source_id: nothing to allocate or clean up, and two sources that share
        # one only wait for each other. Taken before ``self._lock``, never inside
        # it, and never nested across sources.
        self._registration_stripes = tuple(
            threading.RLock() for _ in range(_REGISTRATION_STRIPES)
        )

        # Per-type timings and row sizes of every registration (see its module);
        # collected only when asked for.
        self.stats = RegistrationStats(registration_stats)

        self._state.on_source_added = None
        self._state.on_source_removed = None

    # --- Claim-snapshot accessors (SourceManager cleanup / precache reads) -----
    # Each snapshots under the fine-grained lock so a caller can iterate without
    # racing a concurrent commit ("dict changed size during iteration").

    def claim_items(self) -> List[Tuple[str, SourceClaim]]:
        """A ``(source_id, claim)`` snapshot of the confirmed catalog."""
        with self._lock:
            return list(self._state.claims.items())

    def claim_ids(self) -> List[str]:
        """A snapshot of the confirmed source_ids."""
        with self._lock:
            return list(self._state.claims.keys())

    def has_claim(self, source_id: str) -> bool:
        """Whether *source_id* is currently in the confirmed catalog."""
        with self._lock:
            return source_id in self._state.claims

    def local_claim_paths(self) -> List[Tuple[str, str]]:
        """``(source_id, primary_path)`` for every non-remote confirmed source.

        Includes a source whose registration is still pending.
        """
        with self._lock:
            return [
                (claim.source_id, claim.primary_path)
                for claim in self._state.claims.values()
                if not claim.is_remote
            ]

    def claim_primary_path(self, source_id: str) -> Optional[str]:
        """The primary path of a confirmed source, or None. O(1)."""
        with self._lock:
            claim = self._state.claims.get(source_id)
            return claim.primary_path if claim is not None else None

    # --- Deferred registration ---------------------------------------------

    def set_defer_registration(self, defer: bool) -> None:
        """Commit new claims as pending, catalog row only, from now on (or stop).

        Only claims that register by opening a local file are deferred; see
        :meth:`_deferrable`.
        """
        self._defer_registration = defer

    def set_pending_hook(self, hook: Optional[Callable[[str], None]]) -> None:
        """Call *hook* with each source_id that is committed pending."""
        self._pending_hook = hook

    def pending_count(self) -> int:
        """Sources waiting for their registration to run.

        A source whose registration failed is not counted: it is retried, but
        nothing is waiting on it.
        """
        with self._lock:
            return len(self._pending) - len(self._pending_failed)

    def unregistered_count(self) -> int:
        """Sources in the catalog whose registration has not completed, failed
        ones included."""
        with self._lock:
            return len(self._pending)

    def is_pending(self, source_id: str) -> bool:
        # A dict membership test is atomic, so this takes no lock: it is asked
        # on every registry miss, and must not wait on a scan's commits.
        return source_id in self._pending

    def catalog_url_of(self, source_id: str) -> Optional[str]:
        """The catalog ``source_url`` of a source, registered or still pending
        (None if it is neither)."""
        adapter = self._server.sources.get(source_id)
        if adapter is not None:
            return adapter.catalog_url
        with self._lock:
            if source_id not in self._pending:
                return None
            claim = self._state.claims.get(source_id)
            url = self._pending[source_id]
        if url:
            return url
        return to_catalog_url(str(claim.primary_path)) if claim is not None else None

    def _clear_pending(self, source_id: str) -> None:
        """Forget a source's pending state. Caller holds ``self._lock``."""
        self._pending.pop(source_id, None)
        self._pending_failed.pop(source_id, None)

    def _registration_lock(self, source_id: str) -> threading.RLock:
        return self._registration_stripes[hash(source_id) % _REGISTRATION_STRIPES]

    def _deferrable(self, claim: SourceClaim) -> bool:
        """Whether registering *claim* is the slow, local, file-opening kind.

        A remote proxy registers from a bulk seed and an unresolved cloud source
        opens nothing, so neither gains from waiting.
        """
        return (
            not claim.is_remote
            and claim.source_type != "tensor-server"
            and not self._claim_is_unresolved(claim)
        )

    def _commit_pending_claim(
        self, claim: SourceClaim, catalog_url: Optional[str]
    ) -> bool:
        """Commit *claim* to the catalog only; its registration runs later.

        Everything but the parse happens now: the source has a catalog row
        (``is_resolved`` false, ``pending``, built from the claim), and its claim
        and signatures are in state, so the rescan diff, removal and refresh treat
        it as any confirmed source. It is not in the registry until
        :meth:`ensure_registered` has built its adapter.
        """
        try:
            if self._metadata_db is not None:
                self._metadata_db.sync_pending_source(claim, catalog_url)
        except Exception as e:
            self._log_source_failure(
                claim.source_id,
                "Failed to catalog pending source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
                exc_info=True,
            )
            self._rollback_source_registration(claim.source_id)
            self._record_failed_source_attempt(claim.source_id)
            return False

        signatures = self._build_claim_signatures(claim)
        with self._lock:
            added = self._state.add_claim(claim, notify=False)
            if not added:
                self._rollback_source_registration(claim.source_id)
                self._record_failed_source_attempt(claim.source_id)
                return False
            self._commit_claim_bookkeeping(claim, signatures)
            self._pending[claim.source_id] = catalog_url
        self._on_pending(claim.source_id)
        return True

    def ensure_registered(self, source_id: str) -> bool:
        """Register a pending source now. Returns whether it is registered.

        Safe from any thread and single-flight per source: the worker and a read
        that needs the source at the same moment share one registration. A
        source that is not pending (never was, already registered, removed)
        returns at once. A failed one is not retried inside its backoff window.
        """
        if not self.is_pending(source_id):
            return True
        with self._registration_lock(source_id):
            with self._lock:
                claim = self._state.claims.get(source_id)
                if claim is None or source_id not in self._pending:
                    return source_id not in self._pending
                failed = source_id in self._pending_failed
                catalog_url = self._pending[source_id]
            if failed and not self._should_retry_source(source_id):
                return False

            errors: List[str] = []
            if not self._register_source_claim(
                claim, catalog_url=catalog_url, replace=True, error_sink=errors
            ):
                self._record_failed_source_attempt(source_id)
                self._mark_registration_failed(source_id, errors)
                self.log_summary_if_drained()
                return False

            with self._lock:
                self._clear_pending(source_id)
            self._clear_failed_source_attempt(source_id)
        self._notify_source_committed(source_id)
        self.log_summary_if_drained()
        return True

    def materialize(self, source_id: str) -> None:
        """Register a pending source for a reader that needs it now.

        Returns once the source is registered, or is not pending (unknown,
        removed). Raises why it cannot be: ``SourceRegistrationError`` when its
        registration failed, and an unresolved error when it has not completed.
        """
        if self.ensure_registered(source_id):
            return
        with self._lock:
            if source_id not in self._pending:
                return
            error = self._pending_failed.get(source_id)
        if error:
            raise SourceRegistrationError(source_id, error)
        raise SourceUnresolvedError(
            f"source {source_id!r} is unresolved: its registration has not run yet"
        )

    def log_summary_if_drained(self) -> None:
        """Log the registration cost once nothing is left waiting.

        A failed source is not waiting (see :meth:`pending_count`), so a site
        with sources that cannot register still gets its summary.
        """
        if not self._defer_registration and self.pending_count() == 0:
            self.stats.log_summary()

    def _mark_registration_failed(self, source_id: str, errors: List[str]) -> None:
        """Record why a pending source did not register.

        The row reads ``failed`` with the error, so a client does not wait on a
        source that will not finish; a read of it raises the error.
        """
        message = errors[-1] if errors else "registration failed (see the server log)"
        with self._lock:
            # Only while it is still pending: a removal that got here first has
            # forgotten the source, and ``pending_count`` relies on the failed
            # map being a subset of the pending one.
            if source_id not in self._pending:
                return
            self._pending_failed[source_id] = message
            claim = self._state.claims.get(source_id)
            catalog_url = self._pending[source_id]
        if claim is not None and self._metadata_db is not None:
            try:
                self._metadata_db.sync_pending_source(claim, catalog_url, error=message)
            except Exception:
                logger.exception("could not record the failure of source %s", source_id)

    def failed_pending_due(self) -> List[str]:
        """Pending sources whose registration failed and may be tried again."""
        with self._lock:
            failed = list(self._pending_failed)
        return [sid for sid in failed if self._should_retry_source(sid)]

    def _on_pending(self, source_id: str) -> None:
        """Tell whoever registers pending sources that one is waiting."""
        hook = self._pending_hook
        if hook is not None:
            try:
                hook(source_id)
            except Exception:
                logger.exception("pending-registration hook failed for %s", source_id)

    def _reconcile_discovered_state(
        self,
        discovered_state: DiscoveryState,
        force_full: bool = False,
    ) -> None:
        """Apply add/remove/update diffs between the current and discovered states.

        On an incremental rescan, cloud-root sources are excluded from the
        candidate set: their subtree was not walked, so they are absent from
        ``discovered_state`` and would otherwise be diffed/removed. Excluding them
        by a hash-set check (``_cloud_source_ids``) preserves them untouched -- no
        removal, no signature diff, no per-member ``Path.resolve()`` -- without the
        ``_preserve_skipped_claims`` re-injection loop. On a force_full pass cloud
        sources ARE walked, so they participate in the full reconcile.
        """
        with self._lock:
            current_claims = {
                source_id: claim
                for source_id, claim in self._state.claims.items()
                if self._roots.is_monitored(claim.primary_path)
                and (force_full or source_id not in self._cloud_source_ids)
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

        # Removal: not rediscovered on _MISSES_BEFORE_REMOVAL consecutive scans, and
        # quiet. A claim found again, or no longer a candidate, forfeits its count.
        # A cloud source on an incremental tick is neither: it was not walked, so
        # the tick says nothing about it and its count waits for the next full pass
        # (the only one that can count a miss for it).
        absent_ids = current_ids - discovered_ids
        unwalked_ids = set() if force_full else self._cloud_source_ids
        for source_id in [
            sid
            for sid in self._missed_scans
            if sid not in absent_ids and sid not in unwalked_ids
        ]:
            del self._missed_scans[source_id]
        removed_ids = []
        for source_id in sorted(absent_ids):
            misses = self._missed_scans.get(source_id, 0) + 1
            self._missed_scans[source_id] = misses
            if misses >= _MISSES_BEFORE_REMOVAL and self._claim_is_quiet(
                current_claims[source_id]
            ):
                removed_ids.append(source_id)
        added_claims = [
            discovered_claims[source_id]
            for source_id in sorted(discovered_ids - current_ids)
            if self._should_retry_source(source_id)
        ]
        # A changed source is REBUILT in place rather than removed and re-added:
        # the replacement adapter is registered on top of the live one, so the
        # source is never absent from ListFlights or the catalog and a failed
        # rebuild does not cost a working source. Still gated on stability --
        # rebuilding mid-write reads a half-written file.
        refreshed_ids = [
            source_id
            for source_id in sorted(changed_ids)
            if self._claim_is_quiet(current_claims[source_id])
            and self._should_retry_source(source_id)
        ]

        for source_id in removed_ids:
            self._commit_remove_source(source_id)
        for claim in added_claims:
            self._commit_add_claim(claim)
        for source_id in refreshed_ids:
            self._refresh_claim(discovered_claims[source_id])

    def _claim_is_quiet(self, claim: SourceClaim) -> bool:
        """Has this claim stopped changing long enough to remove or rebuild it?

        The removal/rebuild side of the stability gate, and deliberately the
        *same* predicate (``entry_is_quiet``) the claim gate applies on the way
        in: "quiet enough to claim" and "quiet enough to have stopped existing"
        are one question asked from two sides, and answering it in two places is
        what let them drift apart (biopb/biopb#1042).

        Asks the claim's own member paths, so it is O(members) and needs no walk.
        A member that cannot be stat'd is not churning: it is gone, which is the
        case removal exists to act on. A cloud source bypasses the window, as the
        claim gate does -- a placeholder's mtime is not evidence of a write.
        """
        if claim.source_id in self._cloud_source_ids or self._roots.is_cloud(
            claim.primary_path
        ):
            return True
        now = time.time()
        for member_path in {claim.primary_path, *claim.member_paths}:
            if is_remote_url(member_path):
                continue
            try:
                stat_result = os.stat(member_path)
            except OSError:
                continue
            if not entry_is_quiet(
                entry_change_time(stat_result, now), now, self._stability_window
            ):
                return False
        return True

    def _build_claim_signatures(self, claim: SourceClaim) -> Dict[str, Tuple[Any, ...]]:
        """Stat every member of a claim into a signature map.

        Compared between scans to detect an in-place change. Cloud-root
        membership is a property of the *source* -- every member lives under
        ``claim.primary_path`` -- so it is resolved once and the whole source gets
        one cloud-invariance policy (identity only, so hydration and eviction do
        not flap a resolved source).
        """
        signatures: Dict[str, Tuple[Any, ...]] = {}
        cloud = self._roots.is_cloud(claim.primary_path)
        for member_path in sorted(claim.member_paths):
            try:
                # `stat` follows symlinks, so it lands where an explicit
                # `resolve()` would -- and the mode it returns answers is_dir
                # without a second trip.
                stat_result = Path(member_path).stat()
            except OSError:
                continue
            signatures[member_path] = build_entry_signature(
                stat_result,
                stat.S_ISDIR(stat_result.st_mode),
                cloud=cloud,
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
            if claim.source_id in self._cloud_source_ids:
                # Cloud sources are preserved by the reconcile scoping (excluded
                # from the candidate set on incrementals), not re-injected here.
                # Re-injecting would place them in ``discovered_ids`` while reconcile
                # drops them from ``current_ids`` -> a spurious re-add every cycle.
                # Skipping also retires the per-cloud-claim ``Path.resolve()`` loop.
                continue
            if not self._roots.is_monitored(claim.primary_path):
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
            claim_path = Path(claim_path_str)
            for skipped_dir in skipped_dirs:
                if claim_path == skipped_dir or claim_path.is_relative_to(skipped_dir):
                    return True
        return False

    def _stream_first_scan_add(self, claim: SourceClaim) -> None:
        """Commit a first-scan claim live, as the walk discovers it (Option B).

        Wired as the discovery state's ``on_source_added`` only during the first
        full scan. Routes through ``_commit_add_claim`` so a streamed add gets
        the same server registration, metadata-DB sync, signature bookkeeping,
        and precache gating as a reconcile-driven add.

        Idempotent against a *retried* first scan: a source already committed by
        a prior partial scan (that later failed before flipping
        ``_initial_scan_done``) is skipped, so the duplicate-rollback path in
        ``_commit_add_claim`` -- which would *unregister* it -- is never hit, and
        the end-of-walk reconcile stays a clean no-op.
        """
        with self._lock:
            if claim.source_id in self._state.claims:
                return
        self._commit_add_claim(claim)

    def _commit_add_claim(
        self,
        claim: SourceClaim,
        catalog_seed: Optional[tuple] = None,
        catalog_url: Optional[str] = None,
    ) -> bool:
        """Register a discovered source, then commit it into confirmed state.

        ``catalog_seed`` is forwarded to ``_register_source_claim`` (biopb/biopb#266,
        remote bulk-seed); ``None`` for local sources. ``catalog_url`` overrides the
        descriptor's display ``source_url`` (drag-drop re-rooting, see
        ``_drop_catalog_url``); ``None`` falls back to the display root of the
        monitored directory the claim sits under, if that has one.
        """
        if catalog_url is None:
            catalog_url = self._catalog_url_for(claim)
        if (
            self._defer_registration
            and catalog_seed is None
            and self._deferrable(claim)
        ):
            return self._commit_pending_claim(claim, catalog_url)
        if not self._register_source_claim(
            claim, catalog_seed=catalog_seed, catalog_url=catalog_url
        ):
            self._record_failed_source_attempt(claim.source_id)
            return False

        signatures = self._build_claim_signatures(claim)
        with self._lock:
            added = self._state.add_claim(claim, notify=False)
            if not added:
                self._rollback_source_registration(claim.source_id)
                self._record_failed_source_attempt(claim.source_id)
                return False
            self._commit_claim_bookkeeping(claim, signatures)

        # Route the freshly committed source to the precache worker. The
        # live-vs-startup gate (and the best-effort hook invocation) lives in the
        # injected SourceManager callback, which owns the startup/suppress state.
        self._notify_source_committed(claim.source_id)
        return True

    def _commit_claim_bookkeeping(
        self, claim: SourceClaim, signatures: Dict[str, Tuple[Any, ...]]
    ) -> None:
        """Index a claim that has just been committed. Caller holds ``self._lock``.

        Every index a commit must leave consistent, in one place: the add and
        refresh paths share it so a new one cannot be added to only one of them.
        ``_cloud_source_ids`` is what lets the incremental reconcile preserve a
        cloud-root source by a hash-set check (see _reconcile_discovered_state).
        """
        self._source_signatures[claim.source_id] = signatures
        if self._roots.is_cloud(claim.primary_path):
            self._cloud_source_ids.add(claim.source_id)
        self._clear_failed_source_attempt(claim.source_id)

    def _refresh_claim(self, claim: SourceClaim) -> bool:
        """:meth:`_refresh_claim_locked`, serialized against this source's
        deferred registration and removal."""
        with self._registration_lock(claim.source_id):
            return self._refresh_claim_locked(claim)

    def _refresh_claim_locked(self, claim: SourceClaim) -> bool:
        """Re-register an already-confirmed source against its bytes as they are now.

        A source's descriptor and its ``content_version`` are both sampled in the
        adapter's ``__init__``, so building a new adapter is the only way to
        notice that a file was rewritten in place -- and the version is what
        namespaces the chunk cache, so a refresh that skipped it would go on
        serving the pre-edit bytes (biopb/biopb#944).

        Registration goes through ``replace=True``, which swaps the new adapter
        in over the old one and closes the old one only afterwards; see
        :meth:`_register_source_claim`.
        """
        with self._lock:
            previous = self._state.claims.get(claim.source_id)
        if previous is None:
            return False

        # The display url is the source's, not the rebuild's. Re-deriving it
        # would either hand a dnd:// drop its native file:// url back -- losing
        # its removability, since remove_source authorizes on that scheme -- or
        # stamp the marker onto a monitored source, where it would falsely
        # promise that nothing will re-add it.
        live = self._server.sources.get(claim.source_id)
        if live is not None:
            catalog_url = getattr(live, "_catalog_url", None)
        else:
            catalog_url = self._pending.get(claim.source_id)

        errors: List[str] = []
        if not self._register_source_claim(
            claim, catalog_url=catalog_url, replace=True, error_sink=errors
        ):
            self._record_failed_source_attempt(claim.source_id)
            if self.is_pending(claim.source_id):
                self._mark_registration_failed(claim.source_id, errors)
            tracker = self._failed_sources.get(claim.source_id)
            if tracker is not None and tracker.attempts >= self._max_refresh_failures:
                logger.warning(
                    "Giving up on source %s after %d consecutive failed "
                    "rebuild attempts; removing it instead of continuing to "
                    "serve its stale adapter",
                    claim.source_id,
                    tracker.attempts,
                )
                self._commit_remove_source(claim.source_id)
            return False

        # Outside the lock: this stats every member, and the lock it would
        # otherwise hold also serializes the rescan's reconcile.
        signatures = self._build_claim_signatures(claim)

        with self._lock:
            # Membership moves under a source (a file added to a sequence dir),
            # so the claim is replaced rather than left at what was discovered
            # when it was first registered.
            self._state.remove_claim(previous.primary_path, notify=False)
            # The rebuilt adapter is already live, so state must track the NEW
            # membership even where it overlaps another source's claim; the old
            # membership would describe an adapter that no longer exists. Hence
            # replace_claim rather than add_claim's reject-on-conflict.
            conflicting = self._state.replace_claim(claim)
            if conflicting:
                logger.error(
                    "Refreshed source %s now overlaps another source's claim "
                    "on %s; the rebuilt adapter is serving but those paths "
                    "remain attributed to the other source",
                    claim.source_id,
                    sorted(conflicting),
                )
            self._commit_claim_bookkeeping(claim, signatures)
            # A source that was still pending is registered now: this rebuild is
            # the registration the worker would have run.
            self._clear_pending(claim.source_id)

        logger.info(f"Refreshed source: {claim.source_id}")
        # Warm as a fresh add does, not only when the content_version moved: the
        # cache evicts under LRU, so an unchanged source can still have holes.
        # Cheap when it has none -- an unmoved token leaves every cache key
        # identical, and resolve_chunk_data calls compute_fn only on a miss.
        self._notify_source_committed(claim.source_id)
        return True

    def _commit_remove_source(self, source_id: str) -> bool:
        """:meth:`_commit_remove_source_locked`, serialized against this
        source's deferred registration, which must not register it back."""
        with self._registration_lock(source_id):
            return self._commit_remove_source_locked(source_id)

    def _commit_remove_source_locked(self, source_id: str) -> bool:
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
            self._cloud_source_ids.discard(source_id)
            self._missed_scans.pop(source_id, None)
            self._clear_pending(source_id)
            self._clear_failed_source_attempt(source_id)
        return True

    def _reconcile_one_upstream(self, upstream: SourceConfig) -> bool:
        """Diff one upstream's live source list against the mirrored catalog.

        Returns whether the mirrored set changed (a source was added or removed).

        The diff is the same add/remove model as the filesystem reconcile, but the
        "scan" is a remote ``list_sources()`` and the unit is a source_id (not a
        path signature): desired = the alias-namespaced ids the upstream lists now;
        current = the tensor-server claims already mirrored from this endpoint.
        """
        from biopb.tensor import TensorFlightClient

        from biopb_tensor_server.adapters.remote_tensor import (
            _split_grpc_url,
            fetch_upstream_rows,
            list_upstream_versions,
            resolve_upstream_credentials,
        )
        from biopb_tensor_server.sources.resolve import namespaced_source_id

        endpoint, _ = _split_grpc_url(upstream.url)
        alias = upstream.alias
        # The catalog fetch dials the upstream directly (not through the adapter
        # pool), so it needs the same per-upstream token AND trust anchor the
        # mirrored adapters will use (biopb/biopb#604 item 4) -- otherwise a
        # grpcs:// upstream with a configured CA would still be TOFU-pinned here.
        credentials = resolve_upstream_credentials(upstream, self._credentials_config)

        client = TensorFlightClient(
            endpoint,
            cache_bytes=0,
            token=credentials.token,
            tls_ca_pem=credentials.tls_ca_pem,
            tls_fingerprint=credentials.tls_fingerprint,
        )
        try:
            # A narrow id + indexed_at pass decides everything that follows, so a
            # steady re-list of a six-figure catalog moves two columns, not every
            # source's metadata. Complete (the server-side DuckDB catalog is not
            # truncated like list_sources()), so what it no longer lists is gone.
            versions = list_upstream_versions(client)
            desired = {namespaced_source_id(alias, up): up for up in versions}

            prefix = f"{endpoint}/"
            alias_prefix = f"{alias}__" if alias else None
            with self._lock:
                current = {
                    source_id
                    for source_id, claim in self._state.claims.items()
                    if claim.source_type == "tensor-server"
                    and str(claim.primary_path).startswith(prefix)
                    and (alias_prefix is None or source_id.startswith(alias_prefix))
                }

            added = desired.keys() - current
            # A failed query raised above, leaving the mirrored catalog untouched.
            removed = current - desired.keys()

            for source_id in sorted(removed):
                self._commit_remove_source(source_id)

            # A mirrored source needs its row again only when the upstream
            # re-registered it (indexed_at moved) -- notably unresolved -> resolved
            # -- or when it was never seeded. An unversioned upstream has no
            # indexed_at to compare, so it is re-read every time.
            stale = {}
            for source_id in current & desired.keys():
                adapter = self._server.sources.get(source_id)
                if adapter is None or not hasattr(adapter, "seed_catalog"):
                    continue
                if not adapter.is_current(versions[desired[source_id]].indexed_at):
                    stale[source_id] = adapter

            extra_config = {}
            if upstream.credentials_profile:
                extra_config["credentials_profile"] = upstream.credentials_profile
            # The proxy's display authority; without it every mirrored source_url
            # exposes the upstream host:port instead of the alias (biopb/biopb#788).
            if alias:
                extra_config["alias"] = alias

            # One bounded batch of full rows at a time, so the first sync of a large
            # catalog never holds it whole.
            wanted = [desired[sid] for sid in sorted(added | stale.keys())]
            sizes = {up: versions[up].size for up in wanted}
            for rows in fetch_upstream_rows(client, wanted, sizes):
                for row in rows:
                    source_id = namespaced_source_id(alias, row["source_id"])
                    seed = self._row_to_seed(row)
                    if source_id in added:
                        self._commit_add_claim(
                            SourceClaim(
                                source_type="tensor-server",
                                primary_path=f"{endpoint}/{row['source_id']}",
                                source_id=source_id,
                                extra_config=dict(extra_config),
                            ),
                            catalog_seed=seed,
                        )
                    elif source_id in stale:
                        self._refresh_mirrored_source(source_id, stale[source_id], seed)
        finally:
            self._close_upstream_client(client)

        if added or removed:
            logger.info(
                "Upstream %s re-list: +%d / -%d sources",
                endpoint,
                len(added),
                len(removed),
            )
        # Whether the mirrored set moved -- drives the adaptive re-list cadence.
        return bool(added or removed)

    @staticmethod
    def _row_to_seed(row: dict) -> tuple:
        """(tensors, metadata, is_resolved, source_url, indexed_at) for
        ``seed_catalog``."""
        raw = row.get("metadata_json")
        try:
            metadata = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            metadata = {}
        return (
            row.get("tensors") or [],
            metadata,
            # Whether that row describes a real source yet. True for an upstream
            # predating the column.
            bool(row.get("is_resolved", True)),
            row.get("source_url"),
            row.get("indexed_at"),  # -> proxy content_version (biopb/biopb#178)
        )

    def _refresh_mirrored_source(
        self, source_id: str, adapter: Any, seed: tuple
    ) -> None:
        """Re-seed an already-mirrored source and re-sync its catalog row if the
        seed changed (so a steady re-list does not churn ``indexed_at``)."""
        if adapter.seed_catalog(*seed) and self._metadata_db is not None:
            try:
                self._metadata_db.sync_source_added(source_id, adapter)
            except Exception:
                logger.warning(
                    "failed to refresh mirrored catalog row for %s",
                    source_id,
                    exc_info=True,
                )

    @staticmethod
    def _close_upstream_client(client) -> None:
        # An exception here would replace whatever is propagating out of the
        # caller's try: body -- and a broken channel is exactly when both an
        # upstream failure and a failing close() happen together (biopb/biopb#529).
        try:
            client.close()
        except Exception:
            logger.debug("error closing upstream client", exc_info=True)

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
        if not self._roots.is_cloud(claim.primary_path):
            return False
        return self._claim_has_dehydrated_member(claim)

    def _claim_has_dehydrated_member(self, claim: SourceClaim) -> bool:
        """True when any local member *file* of *claim* is a non-resident placeholder.

        Metadata-only (``os.stat`` via ``_is_offline_placeholder``); never opens
        content, so it cannot itself trigger a cloud recall. Remote members are
        ignored.

        A directory source (zarr store, ...) is born resolved from its resident
        sidecars; only a non-resident *content file* forces deferral. Guard on
        ``is_file`` so a directory's ``st_blocks == 0`` (macOS APFS) is never a
        false hit.
        """
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

    def should_warm(self, source_id: str) -> bool:
        """Whether the precache worker may warm *source_id* right now.

        Registration decides residency once (``_claim_is_unresolved``), but the
        cloud provider (OneDrive Files On-Demand, ...) can re-dehydrate the bytes
        afterwards, and precache has no per-chunk gate -- so a backlog pass would
        trigger exactly the recall the ``cloud = true`` policy exists to prevent
        (#174).

        Ask the adapter, not the claim's ``member_paths``: those are just the
        directory for every dir-claimed format (zarr, ome-zarr, ome-zarr-hcs,
        ndtiff, tiff-sequence, micromanager-legacy), and the placeholder stat is
        ``is_file``-guarded, so a wholly dehydrated store reads as resident
        (biopb/biopb#1035).

        Only sources under a ``cloud`` root are asked, which keeps the bounded
        stat walk off the common path. Returns False when the source is no longer
        registered or its adapter cannot answer -- a gate that cannot see is not
        permission to read.
        """
        with self._lock:
            claim = self._state.claims.get(source_id)
        if claim is None:
            return False
        if not self._roots.is_cloud(claim.primary_path):
            return True
        adapter = self._server.sources.get(source_id)
        if adapter is None:
            return False
        try:
            return bool(adapter.is_resident())
        except Exception:  # noqa: BLE001 -- a gate that cannot see fails closed
            return False

    def _on_source_resolved(self, source_id: str, adapter: Any) -> None:
        """Backfill the metadata DB when an unresolved cloud source resolves.

        ``sync_source_added`` is an INSERT OR REPLACE upsert, so re-syncing the
        now-resolved adapter overwrites the source's NULL shape/dtype row with the
        concrete descriptor. Persistence across restart is phase 3 (file-backed
        DB); here the backfill lives for the process lifetime.
        """
        if self._metadata_db is not None:
            try:
                self._metadata_db.sync_source_added(source_id, adapter)
            except Exception:
                logger.exception(
                    "metadata-DB backfill failed for resolved source %s", source_id
                )

    def _find_containing_source(self, path: str) -> Optional[str]:
        """Return the source_id owning a strict ancestor of ``path``, else None.

        Used by ``add_local_source`` to reject a drop that lands inside an
        already-registered source (case 4). Only strict ancestors count -- an
        exact re-drop of a source's own path is handled as ``already_present``.
        """
        p = Path(resolve_local_path(path))
        with self._lock:
            for ancestor in p.parents:
                owner = self._state.get_source_for_path(str(ancestor))
                if owner is not None:
                    return owner
        return None

    def _warn_if_experimental(self, claim: SourceClaim) -> None:
        """Emit a one-time EXPERIMENTAL warning for remote/cloud source families.

        Cloud/synced-folder, remote tensor-server proxy, and remote-URL sources
        are experimental; classify the claim and warn once per family (see
        ``_warn_experimental_source``). Cheap, stat-free classification: it reads
        ``claim.unresolved`` / the configured cloud roots, never opens a file.
        """
        if claim.source_type == "tensor-server":
            family = "tensor-server"
        elif claim.unresolved or self._roots.is_cloud(claim.primary_path):
            family = "cloud"
        elif is_remote_url(claim.primary_path):
            family = "remote-url"
        else:
            return
        _warn_experimental_source(family)

    @staticmethod
    def _source_config_for(claim: SourceClaim) -> SourceConfig:
        """The config a claim's adapter is created from."""
        return SourceConfig(
            type=claim.source_type,
            url=str(claim.primary_path),
            source_id=claim.source_id,
            credentials_profile=claim.extra_config.get("credentials_profile"),
            alias=claim.extra_config.get("alias"),
        )

    def _register_source_claim(
        self,
        claim: SourceClaim,
        catalog_seed: Optional[tuple] = None,
        catalog_url: Optional[str] = None,
        replace: bool = False,
        error_sink: Optional[List[str]] = None,
    ) -> bool:
        """Create and register a source, rolling back on partial failure.

        ``error_sink`` collects the text of the failure, for a caller that records
        it on the source (a deferred registration leaves it in the row).

        ``replace`` re-registers a source that is already live: the new adapter
        is swapped in on top of the old one and the old one is closed only after
        the swap and the catalog upsert have both succeeded. Ordering matters --
        unregister-then-register would leave the source absent from ListFlights
        and its catalog row deleted for the length of the rebuild, and gone for
        good if the rebuild then failed.

        ``catalog_seed`` (biopb/biopb#266) is an optional
        ``(tensors, metadata, is_resolved, source_url)`` tuple from a bulk upstream
        ``query``; when the adapter supports it (the remote proxy), it is
        applied before ``sync_source_added`` so registration needs no per-source
        upstream RPC. ``catalog_url`` (drag-drop re-rooting) overrides the display
        ``source_url`` on the adapter *before* register/sync so both ListFlights
        and the metadata DB record the re-rooted url.
        """
        self._warn_if_experimental(claim)
        create_s = None  # set only where a file is actually opened
        try:
            source_config = self._source_config_for(claim)

            if self._claim_is_unresolved(claim):
                # Cloud-storage phase 2: register a placeholder that resolves
                # lazily on first access (re-claim + create_from_config on the
                # hydrated path) instead of opening the source now.
                adapter = UnresolvedSourceAdapter(
                    source_config,
                    self._registry,
                    credentials_config=self._credentials_config,
                    on_resolved=self._on_source_resolved,
                    cloud_root=self._roots.is_cloud(claim.primary_path),
                )
            else:
                adapter_cls = self._registry.get_adapter_for_type(claim.source_type)
                if adapter_cls is None:
                    if error_sink is not None:
                        error_sink.append(f"no adapter for type {claim.source_type}")
                    self._log_source_failure(
                        claim.source_id,
                        "No adapter for type %s for source %s (%s)",
                        claim.source_type,
                        claim.source_id,
                        claim.primary_path,
                    )
                    return False

                created_at = time.perf_counter()
                adapter = adapter_cls.create_from_config(
                    source_config, self._credentials_config
                )
                create_s = time.perf_counter() - created_at

                # Bulk-seed the catalog surface so sync_source_added below needs
                # no per-source upstream RPC (biopb/biopb#266). Guarded by the
                # adapter opting in via seed_catalog (only the remote proxy does).
                if catalog_seed is not None and hasattr(adapter, "seed_catalog"):
                    tensors, metadata, is_resolved, source_url, indexed_at = (
                        catalog_seed
                    )
                    adapter.seed_catalog(
                        tensors, metadata, is_resolved, source_url, indexed_at
                    )
        except UpstreamConfigError as e:
            if error_sink is not None:
                error_sink.append(str(e))
            # Same skip, different diagnosis: "failed to create adapter" reads as
            # a transient upstream problem, and this one is an operator edit away
            # from fixed and will fail identically until then (biopb/biopb#608).
            self._log_source_failure(
                claim.source_id,
                "Source %s (%s) is MISCONFIGURED and cannot be registered: %s. "
                "Correct the configuration; retrying will not help.",
                claim.source_id,
                claim.primary_path,
                e,
            )
            return False
        except Exception as e:
            if error_sink is not None:
                error_sink.append(str(e))
            self._log_source_failure(
                claim.source_id,
                "Failed to create adapter for source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
                exc_info=True,
            )
            return False

        # Drag-drop re-rooting: stamp the display-only source_url override before
        # register/sync so ListFlights and the metadata-DB row both carry it.
        if catalog_url:
            adapter._catalog_url = catalog_url

        registered = False
        displaced: Optional[Any] = None
        cost = None
        try:
            # Normalize the axis order here rather than leaning on what
            # register_source hands back (biopb/biopb#596): the catalog row
            # below must describe the same tensors the serve path will hand
            # out, and the wrap is idempotent, so the registry re-applying it
            # is a no-op.
            normalizing_at = time.perf_counter()
            adapter = normalize_adapter(adapter)
            normalize_s = time.perf_counter() - normalizing_at
            if replace:
                adapter, displaced = self._server.swap_source(claim.source_id, adapter)
            else:
                self._server.register_source(claim.source_id, adapter)
            registered = True

            # Raises on failure -> the except below rolls back register_source,
            # so a catalog write error never leaves a source visible in
            # ListFlights but absent from DuckDB. sync_source_added is an upsert,
            # so a replace overwrites the row rather than needing it deleted
            # first -- which is what keeps the source continuously catalogued.
            if self._metadata_db is not None:
                sync = self._metadata_db.sync_source_added
                cost = (
                    sync(claim.source_id, adapter, measure=True)
                    if self.stats.enabled
                    else sync(claim.source_id, adapter)
                )

            if create_s is not None:
                self.stats.record_registration(
                    claim.source_type,
                    create_s=create_s,
                    normalize_s=normalize_s,
                    members=len(claim.member_paths),
                    cost=cost if isinstance(cost, SyncCost) else None,
                )
            if displaced is not None:
                # Only now, and this ordering is the reason `swap` hands the
                # displaced adapter back open rather than closing it: until the
                # catalog upsert above has succeeded, the except below may still
                # restore this adapter and go on serving from it. Draining the
                # reader that resolved it before the swap is close()'s own job,
                # not what the delay buys.
                close_adapter(displaced)
            logger.info(f"Registered source with server: {claim.source_id}")
            return True
        except Exception as e:
            if error_sink is not None:
                error_sink.append(str(e))
            self._log_source_failure(
                claim.source_id,
                "Failed to register/sync source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
                exc_info=True,
            )
            if registered and displaced is not None:
                # A failed REBUILD must not cost the working source: put the
                # adapter that was serving back, and its catalog row with it.
                self._restore_displaced_source(claim.source_id, displaced)
            elif registered and self.is_pending(claim.source_id):
                # A claimed source still waiting to register keeps its claim and
                # its pending row: only the adapter this call put in goes.
                self._server.unregister_source(claim.source_id)
            elif registered:
                self._rollback_source_registration(claim.source_id)
            if self._server.sources.get(claim.source_id) is not adapter:
                # The adapter this call built is not the one serving -- the swap
                # never took, or was undone above -- and nothing else holds it,
                # so its handles are ours to release. Harmless where the
                # rollback already closed it: close() must be safe twice.
                close_adapter(adapter)
            return False

    def _restore_displaced_source(self, source_id: str, displaced: Any) -> None:
        """Put a swapped-out adapter back after a failed replace (best-effort)."""
        try:
            self._server.swap_source(source_id, displaced)
            if self._metadata_db is not None:
                self._metadata_db.sync_source_added(source_id, displaced)
        except Exception:
            logger.exception(
                "Failed to restore the previous adapter for source %s; it is "
                "no longer served",
                source_id,
            )

    def _teardown_source_bookkeeping(self, source_id: str) -> None:
        """Drop a source's catalog row (best-effort).

        The teardown shared by the add-failure rollback
        (:meth:`_rollback_source_registration`) and the confirmed remove
        (:meth:`_unregister_source_claim`): the metadata-DB delete is isolated in
        its own try/except -- a catalog-delete failure is logged, never
        propagated (worst case a leaked row).

        Deliberately does NOT touch the server registration (callers own that,
        with differing abort semantics) nor ``_cloud_source_ids``: the remove
        path discards that under ``self._lock`` in :meth:`_commit_remove_source`,
        while the rollback path -- which runs outside that lock -- discards it
        itself.
        """
        if self._metadata_db is not None:
            try:
                self._metadata_db.sync_source_removed(source_id)
            except Exception:
                logger.exception(
                    "Failed to remove source %s from metadata DB",
                    source_id,
                )

    def _rollback_source_registration(self, source_id: str) -> None:
        """Best-effort rollback after a partial add failure."""
        try:
            self._server.unregister_source(source_id)
        except Exception:
            logger.exception("Rollback failed to unregister source %s", source_id)
        self._teardown_source_bookkeeping(source_id)
        self._cloud_source_ids.discard(source_id)

    def _unregister_source_claim(self, source_id: str) -> bool:
        """Remove a source from the server and metadata DB.

        A server-unregister failure aborts (returns False, leaving the claim in
        state for a later retry). A catalog-delete failure does NOT abort: the
        shared :meth:`_teardown_source_bookkeeping` isolates the
        ``sync_source_removed`` call in its own try/except so the server-side
        unregister still completes. The worst case is a leaked catalog row
        (logged), matching the remove-site log-and-continue policy and the
        pre-raise behavior.
        """
        try:
            self._server.unregister_source(source_id)
        except Exception as e:
            logger.exception(
                "Failed to unregister source %s: %s",
                source_id,
                e,
            )
            return False

        self._teardown_source_bookkeeping(source_id)
        logger.info(f"Unregistered source from server: {source_id}")
        return True
