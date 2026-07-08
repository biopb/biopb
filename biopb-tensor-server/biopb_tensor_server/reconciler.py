"""Confirmed-catalog reconciliation and source lifecycle (biopb/biopb#278 item B).

The :class:`Reconciler` owns the *confirmed catalog* -- the live ``SourceClaim``
set (``DiscoveryState``), the server registration, the metadata-DB rows, and the
derived indices (path->source_id, per-source signatures, cloud-source ids, and
failed-source retry state). It is the single writer of that catalog, reached from
the three desired-state sources, all of which reduce to the same
add/remove/register primitives:

  * the periodic filesystem rescan  -> :meth:`_reconcile_discovered_state` /
    :meth:`_preserve_skipped_claims` (fed the walk's discovered claims);
  * a runtime drag-drop / SDK add   -> :meth:`_commit_add_claim` (driven by
    ``SourceManager.add_local_source``);
  * a tensor-server upstream re-list -> :meth:`_reconcile_one_upstream`.

``SourceManager`` owns the *rescan machinery* (event loop, the filesystem walk
via ``TreeScanner``, the stability gate, the cloud partition, the startup
protocol) and delegates every catalog mutation here. The seam is deliberately
narrow:

  * SourceManager -> Reconciler: the reconcile/commit calls above, plus four
    claim-snapshot accessors (:meth:`claim_items` / :meth:`claim_ids` /
    :meth:`has_claim` / :meth:`local_claim_paths`) for its cleanup and precache
    reads.
  * Reconciler -> SourceManager: two injected callables -- ``entry_for`` (the
    cached filesystem-signature lookup, owned by the scan caches) and
    ``notify_source_committed`` (the precache routing gate, owned by the startup
    state) -- plus the shared, in-place-mutated ``monitored_dirs`` set and the
    immutable ``cloud_roots`` set.

The coarse single-writer mutex (a runtime add vs the periodic rescan) stays in
``SourceManager`` (``_catalog_lock``); the fine-grained state RLock (``_lock``,
taken by the commit primitives) lives here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter
from biopb_tensor_server.config import SourceConfig
from biopb_tensor_server.discovery import (
    AdapterRegistry,
    DiscoveryState,
    SourceClaim,
    _is_offline_placeholder,
    is_remote_url,
    resolve_local_path,
)
from biopb_tensor_server.tree_scanner import EntryState, build_entry_signature

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig as _SourceConfig  # noqa: F401
    from biopb_tensor_server.metadata_db import MetadataDatabase
    from biopb_tensor_server.server import TensorFlightServer

logger = logging.getLogger(__name__)


def is_under_cloud_root(cloud_roots: Set[Path], path: str) -> bool:
    """True when *path* is one of *cloud_roots* or lives under one.

    A free function so both the Reconciler (registration / signature policy) and
    the SourceManager stability gate share one cloud-membership rule without
    either object depending on the other.
    """
    if not cloud_roots:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in cloud_roots)


# Remote/cloud source families are EXPERIMENTAL. Warn once per family per process
# (registration runs per source) so an operator sees the maturity caveat without
# per-source log spam. Keyed by family; the set is only ever added to.
_EXPERIMENTAL_WARNED: set = set()
_EXPERIMENTAL_SOURCE_MESSAGES = {
    "cloud": (
        "Cloud / synced-folder sources (cloud=true, e.g. OneDrive Files "
        "On-Demand) are EXPERIMENTAL: resolve-on-serve and hydrate-ahead behavior "
        "may change. See docs/cloud-storage-support.md."
    ),
    "tensor-server": (
        "Remote tensor-server proxy sources (type=tensor-server) are "
        "EXPERIMENTAL: the caching-passthrough proxy may change. "
        "See docs/remote-tensor-cache.md."
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

    def __init__(
        self,
        *,
        server: TensorFlightServer,
        registry: AdapterRegistry,
        discovery_state: DiscoveryState,
        metadata_db: Optional[MetadataDatabase],
        credentials_config: Optional[Any],
        monitored_dirs: Set[Path],
        cloud_roots: Set[Path],
        entry_for: Callable[[str], Optional[EntryState]],
        notify_source_committed: Callable[[str], None],
    ):
        self._server = server
        self._registry = registry
        self._state = discovery_state
        self._metadata_db = metadata_db
        self._credentials_config = credentials_config
        # Shared with SourceManager (in-place-mutated there on directory delete);
        # read-only here for the monitored-claim scoping.
        self._monitored_dirs = monitored_dirs
        self._cloud_roots = cloud_roots
        # Injected SourceManager seams (see module docstring).
        self._entry_for = entry_for
        self._notify_source_committed = notify_source_committed

        # Fine-grained state RLock: rescan/reconcile helpers re-enter other
        # state-mutating helpers, so nested calls on the same thread must not
        # deadlock. The coarse whole-pass mutex lives in SourceManager.
        self._lock = threading.RLock()

        # Maps resolved path -> source_id (str keys for URL support).
        self._path_to_source_id: Dict[str, str] = {}
        # source_id -> member path signature map used to detect in-place changes.
        self._source_signatures: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
        # source_id -> retry/logging state for repeatedly failing datasets.
        self._failed_sources: Dict[str, _FailureTracker] = {}
        # source_ids whose primary_path is under a cloud root, maintained at commit
        # time (O(1) per source). Lets the incremental reconcile preserve cloud
        # sources by a hash-set check instead of resolving every cloud member path.
        self._cloud_source_ids: Set[str] = set()

        # Initialize path tracking from existing claims.
        for source_id, claim in self._state.claims.items():
            self._path_to_source_id[claim.primary_path] = source_id

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
        """``(source_id, primary_path)`` for every non-remote confirmed source."""
        with self._lock:
            return [
                (claim.source_id, claim.primary_path)
                for claim in self._state.claims.values()
                if not claim.is_remote
            ]

    def _is_under_cloud_root(self, path: str) -> bool:
        """True when *path* is a cloud-opted root or lives under one."""
        return is_under_cloud_root(self._cloud_roots, path)

    def _reconcile_discovered_state(
        self,
        discovered_state: DiscoveryState,
        unstable_paths: List[Path],
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
                if self._is_monitored_claim(claim)
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
        # Cloud-root membership is a property of the *source*, not the individual
        # member: every member lives under ``claim.primary_path``, so they share
        # one cloud status. Resolve it once -- both to skip a redundant
        # ``Path.resolve()`` + roots scan per member, and so the whole source's
        # signatures use one uniform cloud-invariance policy (matching the cached
        # branch, whose signatures ``TreeScanner._scan_tree_state`` built with a per-tree
        # ``cloud`` flag).
        cloud = self._is_under_cloud_root(claim.primary_path)
        for member_path in sorted(claim.member_paths):
            entry = self._entry_for(member_path)
            if entry is not None:
                signatures[member_path] = entry.signature
                continue

            try:
                resolved_path = Path(member_path).resolve(strict=False)
                stat_result = resolved_path.stat()
            except OSError:
                continue

            # The cached-entry path above already carries the cloud-invariant
            # signature. This re-stat fallback has no cloud context, so reuse the
            # per-claim flag so hydration/eviction does not flap a resolved source.
            signatures[member_path] = build_entry_signature(
                stat_result,
                resolved_path.is_dir(),
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
        ``_drop_catalog_url``); ``None`` for the discovery/watcher path.
        """
        if not self._register_source_claim(
            claim, catalog_seed=catalog_seed, catalog_url=catalog_url
        ):
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
            # Track cloud-root sources so the incremental reconcile can preserve
            # them by a hash-set check (see _reconcile_discovered_state).
            if self._is_under_cloud_root(claim.primary_path):
                self._cloud_source_ids.add(claim.source_id)
            self._clear_failed_source_attempt(claim.source_id)

        # Route the freshly committed source to the precache worker. The
        # live-vs-startup gate (and the best-effort hook invocation) lives in the
        # injected SourceManager callback, which owns the startup/suppress state.
        self._notify_source_committed(claim.source_id)
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
            self._cloud_source_ids.discard(source_id)
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
        import json

        from biopb.tensor import TensorFlightClient

        from biopb_tensor_server.adapters.remote_tensor import (
            _resolve_upstream_token,
            _split_grpc_url,
            fetch_upstream_catalog,
            list_upstream_source_ids,
        )
        from biopb_tensor_server.config import _namespaced_source_id

        endpoint, _ = _split_grpc_url(upstream.url)
        alias = upstream.alias
        token = _resolve_upstream_token(upstream, self._credentials_config)

        client = TensorFlightClient(endpoint, cache_bytes=0, token=token)
        try:
            # ONE bulk query_sources fetches every upstream source's id AND its
            # seed data (tensors + metadata), so mirroring is O(1) upstream RPCs
            # instead of one per added source at registration (biopb/biopb#266).
            # Complete: the server-side DuckDB catalog is not truncated like
            # list_sources() (which would both miss sources AND spuriously remove
            # the ones past the cap below).
            rows, complete = fetch_upstream_catalog(client)
            if rows is not None:
                seed_by_up_id = {r["source_id"]: r for r in rows}
                upstream_ids = list(seed_by_up_id.keys())
            else:
                # Legacy upstream without a SQL catalog: id-only enumeration, no
                # seed -> each added source syncs via a live per-source RPC.
                upstream_ids, complete = list_upstream_source_ids(client)
                seed_by_up_id = {}
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()

        desired = {_namespaced_source_id(alias, up_id): up_id for up_id in upstream_ids}

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

        added = set(desired) - current
        # Only remove when the upstream list is COMPLETE: a truncated/incomplete
        # enumeration must never drop a mirrored source it simply failed to see.
        removed = (current - set(desired)) if complete else set()

        for source_id in sorted(removed):
            self._commit_remove_source(source_id)

        def _row_to_seed(row):
            """(tensors, metadata, data_resident, source_url) for seed_catalog, or None."""
            if row is None:
                return None
            raw = row.get("metadata_json")
            try:
                metadata = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                metadata = {}
            return (
                row.get("tensors") or [],
                metadata,
                bool(row.get("data_resident")),
                row.get("source_url"),
            )

        extra_config = {}
        if upstream.credentials_profile:
            extra_config["credentials_profile"] = upstream.credentials_profile
        for source_id in sorted(added):
            up_id = desired[source_id]
            self._commit_add_claim(
                SourceClaim(
                    source_type="tensor-server",
                    primary_path=f"{endpoint}/{up_id}",
                    source_id=source_id,
                    extra_config=dict(extra_config),
                ),
                catalog_seed=_row_to_seed(seed_by_up_id.get(up_id)),
            )

        # Refresh already-mirrored sources from the same bulk result, so an
        # in-place upstream change -- notably unresolved -> resolved (empty ->
        # populated tensors, data_resident false -> true) -- is reflected on the
        # catalog surface without a per-source RPC (biopb/biopb#266). Re-sync the
        # DuckDB row only when the seed actually changed, so a steady re-list does
        # not churn indexed_at.
        for source_id in sorted(current & set(desired)):
            seed = _row_to_seed(seed_by_up_id.get(desired[source_id]))
            if seed is None:
                continue
            adapter = self._server.sources.get(source_id)
            if adapter is None or not hasattr(adapter, "seed_catalog"):
                continue
            if adapter.seed_catalog(*seed) and self._metadata_db is not None:
                try:
                    self._metadata_db.sync_source_added(source_id, adapter)
                except Exception:
                    logger.warning(
                        "failed to refresh mirrored catalog row for %s",
                        source_id,
                        exc_info=True,
                    )

        if added or removed:
            logger.info(
                "Upstream %s re-list: +%d / -%d sources",
                endpoint,
                len(added),
                len(removed),
            )
        # Whether the mirrored set moved -- drives the adaptive re-list cadence.
        return bool(added or removed)

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
        if not self._is_under_cloud_root(claim.primary_path):
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

        Residency is decided once, at registration time (``_claim_is_unresolved``):
        a source whose files were resident then registers as a normal adapter and
        keeps that registration even if the cloud provider (OneDrive Files
        On-Demand, ...) later re-dehydrates the bytes. Precache has no per-chunk
        residency gate, so a later backlog pass would read those bytes and trigger
        a background recall the ``cloud = true`` policy exists to prevent (#174).

        This re-checks residency at warm time, mirroring the registration-path
        rule so the two stay in sync. Only sources under a ``cloud`` root are
        gated -- a normal local source always warms -- and the check is
        metadata-only, so it never recalls content itself. Returns False (skip)
        when any member *file* is now a placeholder, or when the source is no
        longer registered.

        Boundary: like ``_claim_has_dehydrated_member``, the residency check is
        ``is_file``-guarded, so it does not catch re-dehydration of a
        dir-claiming source's *interior* files (ome-zarr, micromanager, ndtiff,
        tiff-sequence, whose ``member_paths`` is just the directory). Those are
        kept safe today by ``UnresolvedSourceAdapter.list_tensor_descriptors``
        returning empty until resolved; closing the post-resolution re-warm path
        is the cloud-storage spec's phase-4 deferral.
        """
        with self._lock:
            claim = self._state.claims.get(source_id)
        if claim is None:
            return False
        if not self._is_under_cloud_root(claim.primary_path):
            return True
        return not self._claim_has_dehydrated_member(claim)

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
                owner = self._state.path_to_source.get(str(ancestor))
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
        elif claim.unresolved or self._is_under_cloud_root(claim.primary_path):
            family = "cloud"
        elif is_remote_url(claim.primary_path):
            family = "remote-url"
        else:
            return
        _warn_experimental_source(family)

    def _register_source_claim(
        self,
        claim: SourceClaim,
        catalog_seed: Optional[tuple] = None,
        catalog_url: Optional[str] = None,
    ) -> bool:
        """Create and register a source, rolling back on partial failure.

        ``catalog_seed`` (biopb/biopb#266) is an optional
        ``(tensors, metadata, data_resident, source_url)`` tuple from a bulk upstream
        ``query_sources``; when the adapter supports it (the remote proxy), it is
        applied before ``sync_source_added`` so registration needs no per-source
        upstream RPC. ``catalog_url`` (drag-drop re-rooting) overrides the display
        ``source_url`` on the adapter *before* register/sync so both ListFlights
        and the metadata DB record the re-rooted url.
        """
        self._warn_if_experimental(claim)
        try:
            source_config = SourceConfig(
                type=claim.source_type,
                url=str(claim.primary_path),
                source_id=claim.source_id,
                dim_labels=claim.dim_labels,
                dataset=claim.extra_config.get("dataset"),
                credentials_profile=claim.extra_config.get("credentials_profile"),
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

                # Bulk-seed the catalog surface so sync_source_added below needs
                # no per-source upstream RPC (biopb/biopb#266). Guarded by the
                # adapter opting in via seed_catalog (only the remote proxy does).
                if catalog_seed is not None and hasattr(adapter, "seed_catalog"):
                    tensors, metadata, data_resident, source_url = catalog_seed
                    adapter.seed_catalog(tensors, metadata, data_resident, source_url)
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

        # Drag-drop re-rooting: stamp the display-only source_url override before
        # register/sync so ListFlights and the metadata-DB row both carry it.
        if catalog_url:
            adapter._catalog_url = catalog_url

        registered = False
        try:
            self._server.register_source(claim.source_id, adapter)
            registered = True

            # Raises on failure -> the except below rolls back register_source,
            # so a catalog write error never leaves a source visible in
            # ListFlights but absent from DuckDB.
            if self._metadata_db is not None:
                self._metadata_db.sync_source_added(claim.source_id, adapter)

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

    def _teardown_source_bookkeeping(self, source_id: str) -> None:
        """Drop a source's catalog row and path-map entries (best-effort).

        The teardown shared by the add-failure rollback
        (:meth:`_rollback_source_registration`) and the confirmed remove
        (:meth:`_unregister_source_claim`): the metadata-DB delete is isolated in
        its own try/except -- a catalog-delete failure is logged, never
        propagated (worst case a leaked row) -- and every ``_path_to_source_id``
        entry pointing at ``source_id`` is dropped so a later re-add/reconcile of
        the same path is not misled by a stale mapping.

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
                logger.error(
                    "Failed to remove source %s from metadata DB",
                    source_id,
                    exc_info=True,
                )

        paths_to_remove = [
            path for path, sid in self._path_to_source_id.items() if sid == source_id
        ]
        for path in paths_to_remove:
            self._path_to_source_id.pop(path, None)

    def _rollback_source_registration(self, source_id: str) -> None:
        """Best-effort rollback after a partial add failure."""
        try:
            self._server.unregister_source(source_id)
        except Exception:
            logger.error(
                "Rollback failed to unregister source %s", source_id, exc_info=True
            )
        self._teardown_source_bookkeeping(source_id)
        self._cloud_source_ids.discard(source_id)

    def _unregister_source_claim(self, source_id: str) -> bool:
        """Remove a source from the server and metadata DB.

        A server-unregister failure aborts (returns False, leaving the claim in
        state for a later retry). A catalog-delete failure does NOT abort: the
        shared :meth:`_teardown_source_bookkeeping` isolates the
        ``sync_source_removed`` call in its own try/except so the server-side
        unregister and the ``_path_to_source_id`` cleanup still complete -- a
        stale path-map entry pointing at an already-unregistered ``source_id``
        would otherwise mislead a later re-add/reconcile of the same path. The
        worst case is a leaked catalog row (logged), matching the remove-site
        log-and-continue policy and the pre-raise behavior.
        """
        try:
            self._server.unregister_source(source_id)
        except Exception as e:
            logger.error(
                "Failed to unregister source %s: %s",
                source_id,
                e,
                exc_info=True,
            )
            return False

        self._teardown_source_bookkeeping(source_id)
        logger.info(f"Unregistered source from server: {source_id}")
        return True
