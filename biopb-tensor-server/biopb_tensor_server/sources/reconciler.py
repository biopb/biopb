"""Confirmed-catalog reconciliation and source lifecycle (biopb/biopb#278 item B).

The :class:`Reconciler` owns the *confirmed catalog* -- the live ``SourceClaim``
set (``DiscoveryState``), the server registration, the metadata-DB rows, and the
derived indices (path->source_id, per-source signatures and cloud-source ids). It is
the single writer of that catalog, reached from
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
    claim-snapshot accessors (:meth:`claim_ids` / :meth:`has_claim` /
    :meth:`local_claim_paths`) for its cleanup and precache reads.
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from biopb_tensor_server.core.adapter_base import to_catalog_url
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    _is_offline_placeholder,
    resolve_local_path,
    source_is_resident,
)
from biopb_tensor_server.core.errors import (
    SourceRegistrationError,
    SourceResolveRetriableError,
    SourceUnresolvedError,
    UpstreamConfigError,
)
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.sources.entry_stat import (
    build_entry_signature,
    entry_change_time,
    entry_is_quiet,
)
from biopb_tensor_server.sources.mirror import MirrorSet
from biopb_tensor_server.sources.pending_rows import PendingRow, PendingRowWriter
from biopb_tensor_server.sources.roots import Root, RootKind, Roots, path_under_root
from biopb_tensor_server.sources.source_registry import close_adapter

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import (
        SourceConfig as _SourceConfig,  # noqa: F401
    )
    from biopb_tensor_server.serving.metadata_db import CatalogRecord, MetadataDatabase
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
        "On-Demand) are EXPERIMENTAL: resolve-on-serve behavior "
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


# How long a persisted source survives without its row being written or its root
# walked, before a restore stops bringing it back.
_RESTORE_MAX_AGE = timedelta(days=30)


def _same_signature(
    held: Dict[str, Tuple[Any, ...]], found: Dict[str, Tuple[Any, ...]]
) -> bool:
    """Whether two member signatures are the same file state.

    A restored signature has no ``st_dev`` (the persisted form drops it, since it
    renumbers across boots), held as ``None``, which matches whatever the device is.
    """
    if held.keys() != found.keys():
        return False
    return all(
        held[path] == found[path]
        or (held[path][0] is None and held[path][1:] == found[path][1:])
        for path in held
    )


def _warn_experimental_source(family: str) -> None:
    """Log a one-time EXPERIMENTAL warning for a remote/cloud source *family*."""
    if family in _EXPERIMENTAL_WARNED:
        return
    _EXPERIMENTAL_WARNED.add(family)
    logger.warning("%s", _EXPERIMENTAL_SOURCE_MESSAGES[family])


class Reconciler:
    """Single-writer of the confirmed source catalog. See the module docstring."""

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
    ):
        self._server = server
        self._registry = registry
        self._state = discovery_state
        self._metadata_db = metadata_db
        self._credentials_config = credentials_config
        # Shared with SourceManager; read-only here (monitored-claim scoping, cloud
        # policy).
        self._roots = roots
        # The roots that are not persisted the catalog has been told of.
        self._catalog_roots: Set[str] = set()
        # The sources mirrored from each upstream root, which are not claims.
        self._mirrors: Dict[Root, MirrorSet] = {}
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
        # source_id -> consecutive rescans that did not find it. A source that was
        # claimed can briefly stop being claimable with its files still in place (a
        # sidecar rewritten in place, a locked header); removing it on the first
        # miss would unregister a working source and rebuild it a tick later.
        self._missed_scans: Dict[str, int] = {}

        # Deferred registration. While ``_defer_registration`` is set a claim is
        # committed to the catalog only (a ``pending`` row) and registered later,
        # by a worker or when a client resolves it. ``_pending`` maps each
        # such source_id to its display url; ``_pending_failed`` maps those among
        # them whose registration raised to the error (they are not waited for,
        # and not retried until the walk finds a new signature for them, a drop
        # refreshes them or a client resolves them).
        self._defer_registration = False
        self._pending_hook: Optional[Callable[[str], None]] = None
        # While a walk streams claims that register later, their pending rows are
        # written in batches by this (None otherwise: one write per claim).
        self._pending_writer: Optional[PendingRowWriter] = None
        # Sources already reported as reachable by two paths (one report each).
        self._warned_shared: Set[str] = set()
        self._pending: Dict[str, Optional[str]] = {}
        self._pending_failed: Dict[str, str] = {}
        # The pending sources that wait for a client, not the pool: a cloud source
        # is downloaded when it is resolved, never in the background. A subset of
        # ``_pending``, disjoint from ``_pending_failed``.
        self._recall: Set[str] = set()
        # The pending sources a restore found resolved: their rows are complete
        # and list at once, and a read registers them (``check_registered``)
        # rather than raising "resolve it", which a client would never ask for.
        # A subset of ``_pending``, disjoint from the other two.
        self._restored: Set[str] = set()
        # A registration, a refresh and a removal of one source never overlap, so
        # a resolve that races the worker coalesces onto one parse and a removed
        # source is not registered back. A fixed stripe of locks hashed by
        # source_id: nothing to allocate or clean up, and two sources that share
        # one only wait for each other. Taken before ``self._lock``, never inside
        # it, and never nested across sources.
        self._registration_stripes = tuple(
            threading.RLock() for _ in range(_REGISTRATION_STRIPES)
        )

        self._state.on_source_added = None
        self._state.on_source_removed = None

    # --- Claim-snapshot accessors (SourceManager cleanup / precache reads) -----
    # Each snapshots under the fine-grained lock so a caller can iterate without
    # racing a concurrent commit ("dict changed size during iteration").

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

        A source whose registration failed is not counted: nothing is waiting on
        it. Nor is a cloud source, which waits for a client to resolve it.
        """
        with self._lock:
            return len(self._pending) - len(self._pending_failed) - len(self._recall)

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
            claim = self._state.claims.get(source_id)
            if source_id not in self._pending:
                # Registered, its adapter let go of: derived as it was built.
                if claim is None or source_id not in self._server.sources:
                    return None
                url = self._catalog_url_for(claim)
            else:
                url = self._pending[source_id]
        if url:
            return url
        return to_catalog_url(str(claim.primary_path)) if claim is not None else None

    def _clear_pending(self, source_id: str) -> None:
        """Forget a source's pending state. Caller holds ``self._lock``."""
        self._pending.pop(source_id, None)
        self._pending_failed.pop(source_id, None)
        self._recall.discard(source_id)
        self._restored.discard(source_id)

    def _registration_lock(self, source_id: str) -> threading.RLock:
        return self._registration_stripes[hash(source_id) % _REGISTRATION_STRIPES]

    def _deferrable(self, claim: SourceClaim) -> bool:
        """Whether registering *claim* is the slow, local, file-opening kind.

        A remote proxy registers from a bulk seed, so it gains nothing from
        waiting. (A cloud source is not deferred but left for a client: see
        ``_claim_is_unresolved``.)
        """
        return not claim.is_remote and claim.source_type != "tensor-server"

    def claims_under(self, root: Path) -> Dict[str, SourceClaim]:
        """A snapshot of the confirmed claims that lie at or under *root*.

        Lexical, on the path each claim is spelled under: a claim belongs to the
        root its walk found it in, whatever a link under it points at. What a
        root's scan compares its walk with, so that it never diffs the claims of
        another root.
        """
        with self._lock:
            claims = list(self._state.claims.items())
        # Claim paths are already spelled lexically, so a prefix test is the same
        # answer as ``Path.is_relative_to`` without building a Path per claim; a
        # remote url never starts with a local root.
        prefix = str(root)
        under = prefix if prefix.endswith(os.sep) else prefix + os.sep
        return {
            source_id: claim
            for source_id, claim in claims
            if claim.primary_path == prefix or claim.primary_path.startswith(under)
        }

    def restore(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Put the claims of the persisted *rows* back, every one waiting to register.

        What a restart keeps of the last run: the claim, its claim-time signature
        and what its row said it was waiting for (see ``_pending``, ``_recall``,
        ``_pending_failed`` and ``_restored``). No file is opened. The first walk
        then verifies each claim like any rescan: an unchanged one is left, a
        changed one refreshed, one that is gone removed.

        Ownership is re-derived against the current roots, never read from the
        row: a claim under no persisted root, or with a member outside them, or
        that cannot be read, is deleted and left for the walk to find again; a row
        whose root or path under it differs is rewritten. A claim that conflicts
        with one already restored is dropped, the first holding it.

        A row last seen, and whose root was last walked, more than
        ``_RESTORE_MAX_AGE`` ago is dropped too: a root that is never reachable
        (a drive that is gone) must not leave its sources listed for ever.

        Returns the counts ``restored``, ``dropped`` and ``rewritten``, and the
        ``queue`` of sources to register in the background.
        """
        plan = []
        dropped: List[str] = []
        cutoff = datetime.now() - _RESTORE_MAX_AGE
        for row in rows:
            source_id = row["source_id"]
            seen = [t for t in (row.get("last_seen"), row.get("last_scanned")) if t]
            restored = (
                None if seen and max(seen) < cutoff else self._restored_claim(row)
            )
            if restored is None:
                dropped.append(source_id)
            else:
                plan.append((row, *restored))

        kept = 0
        rewrites: List[Tuple[str, str, str]] = []
        recall_rows: List[SourceClaim] = []
        queue: List[str] = []
        with self._lock:
            for row, claim, signatures, root in plan:
                source_id = claim.source_id
                if not self._state.add_claim(claim, notify=False):
                    dropped.append(source_id)
                    continue
                kept += 1
                self._source_signatures[source_id] = signatures
                self._pending[source_id] = self._catalog_url_for(claim)
                reason = row["unresolved_reason"]
                if root.cloud or reason == "needs_recall":
                    # Never resolved from a row: see the Cloud section of the design.
                    self._recall.add(source_id)
                    if row["is_resolved"]:
                        recall_rows.append(claim)
                elif reason == "failed":
                    self._pending_failed[source_id] = (
                        row["unresolved_error"]
                        or "registration failed (see the server log)"
                    )
                else:
                    if row["is_resolved"]:
                        self._restored.add(source_id)
                    queue.append(source_id)
                rel = path_under_root(root.url, claim.primary_path)
                if (row["root_id"], row["rel"]) != (root.root_id, rel):
                    rewrites.append((source_id, root.root_id, rel))

        if self._metadata_db is not None:
            self._metadata_db.drop_catalog_rows(dropped)
            self._metadata_db.rewrite_row_roots(rewrites)
            for claim in recall_rows:
                # Listed as the unresolved source it will be, not as a resolved one.
                self._metadata_db.sync_pending_source(
                    claim,
                    self._pending.get(claim.source_id),
                    recall=True,
                    record=self._state_record(claim),
                )
        return {
            # For the caller to queue: a stat of every restored file for its
            # priority would stall a start over a slow share.
            "queue": queue,
            "restored": kept,
            "dropped": len(dropped),
            "rewritten": len(rewrites),
        }

    def _restored_claim(self, row: Dict[str, Any]):
        """``(claim, signatures, root)`` of a persisted row, or None if it cannot be
        kept: unreadable, or no longer under the roots."""
        try:
            primary = row["primary_path"]
            members = list(row["member_paths"] or [primary])
            claim = SourceClaim(
                row["source_type"],
                primary,
                row["source_id"],
                extra_config=json.loads(row["extra_config"] or "{}"),
                member_paths=members,
                unresolved=row["unresolved_reason"] == "needs_recall",
            )
            signatures = {
                path: (None, *values)
                for path, values in json.loads(row["signature"]).items()
            }
        except Exception:
            logger.warning(
                "dropping unreadable catalog row for source %s",
                row.get("source_id"),
                exc_info=True,
            )
            return None
        root = self._roots.containing(Path(primary))
        if root is None or root.kind not in (RootKind.MONITORED, RootKind.SCAN_ONCE):
            return None
        if any(self._roots.containing(Path(m)) is None for m in members):
            return None
        return claim, signatures, root

    def begin_pending_batch(self) -> None:
        """Write the pending rows of the claims that follow in batches, until
        :meth:`end_pending_batch`. For a walk that claims them by the tens of
        thousands; it must not wait on the catalog for each. Only while
        registration is deferred: otherwise a claim registers where it is found
        and has no pending row."""
        write = getattr(self._metadata_db, "sync_pending_sources", None)
        if (
            write is not None
            and self._defer_registration
            and self._pending_writer is None
        ):
            self._pending_writer = PendingRowWriter(write)

    def end_pending_batch(self) -> None:
        """Write what is still buffered and go back to one write per claim."""
        writer, self._pending_writer = self._pending_writer, None
        if writer is not None:
            writer.close()

    def _commit_pending_claim(
        self,
        claim: SourceClaim,
        catalog_url: Optional[str],
        recall: bool = False,
        queue: bool = True,
    ) -> bool:
        """Commit *claim* to the catalog only; its registration runs later.

        *recall* marks a cloud source: its row says ``needs_recall`` and nothing
        queues it, since registering it downloads it and only a client's
        ``resolve`` does that.

        Everything but the parse happens now: the source has a catalog row
        (``is_resolved`` false, ``pending``, built from the claim), and its claim
        and signatures are in state, so the rescan diff, removal and refresh treat
        it as any confirmed source. It is not in the registry until
        :meth:`ensure_registered` has built its adapter.

        The row is written with the claim and its signature when the source has
        one to persist. The signature is taken once, here, before the row: it is
        the one the row persists and the one state keeps, and a registration that
        follows reuses it.

        *queue* False leaves the claim out of the pool: a failed registration,
        which waits for a new signature, a drop or a resolve.
        """
        signatures = self._build_claim_signatures(claim)
        try:
            record = self._catalog_record(claim, signatures=signatures)
            writer = self._pending_writer
            if self._metadata_db is not None:
                if writer is not None:
                    writer.add(PendingRow(claim, catalog_url, recall, record))
                else:
                    self._metadata_db.sync_pending_source(
                        claim, catalog_url, recall=recall, record=record
                    )
        except Exception as e:
            logger.exception(
                "Failed to catalog pending source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
            )
            self._rollback_source_registration(claim.source_id)
            return False

        with self._lock:
            added = self._state.add_claim(claim, notify=False)
            if not added:
                self._rollback_source_registration(claim.source_id)
                return False
            self._source_signatures[claim.source_id] = signatures
            self._pending[claim.source_id] = catalog_url
            if recall:
                self._recall.add(claim.source_id)
        if queue and not recall:
            self._on_pending(claim.source_id)
        return True

    def _commit_failed_claim(
        self, claim: SourceClaim, catalog_url: Optional[str], errors: List[str]
    ) -> None:
        """Keep a claim whose registration failed, as a failed row.

        Without it the claim would not be in state, and the next walk would find
        it new and try again, for as long as the file stays. As a failed row it is
        known: the walk leaves it alone until its signature changes.
        """
        if self._commit_pending_claim(claim, catalog_url, queue=False):
            self._mark_registration_failed(claim.source_id, errors)

    def ensure_registered(self, source_id: str) -> bool:
        """Make sure a claimed source has an adapter. Returns whether it has.

        Safe from any thread and single-flight per source: the worker and a read
        that needs the source at the same moment share one registration. A
        pending source is registered; a source that is claimed and registered
        but has no adapter in the registry is rebuilt from its row (see
        :meth:`_rebuild`). One that is neither claimed nor pending (never was,
        removed) returns ``True`` at once. A failed one is tried again: nothing
        queues it, so this is a client's resolve asking.
        """
        if not self.is_pending(source_id):
            return self._rebuild(source_id)
        with self._registration_lock(source_id):
            with self._lock:
                claim = self._state.claims.get(source_id)
                if claim is None or source_id not in self._pending:
                    return source_id not in self._pending
                recall = source_id in self._recall
                restored = source_id in self._restored
                catalog_url = self._pending[source_id]

            errors: List[str] = []
            # A restored row is taken as it was written: the walk compares the
            # files with its signature and refreshes what changed, as it does for
            # any registered source.
            hydrate = (
                self._metadata_db.read_hydration(source_id)
                if restored and self._metadata_db is not None
                else None
            )
            # A cloud source raises why it could not be opened (retriable or
            # not) instead of recording a failure: it stays ``needs_recall`` and
            # the client that resolved it hears the reason.
            if not self._register_source_claim(
                claim,
                catalog_url=catalog_url,
                replace=True,
                error_sink=errors,
                recall=recall,
                hydrate=hydrate,
            ):
                if not recall:
                    self._mark_registration_failed(source_id, errors)
                return False

            with self._lock:
                self._clear_pending(source_id)
        # A cloud source is not warmed for having been resolved: it is
        # downloaded because a client asked, and it joins no startup backlog.
        if not recall:
            self._notify_source_committed(source_id)
        return True

    def _rebuild(self, source_id: str) -> bool:
        """Give a registered source back the adapter the registry no longer holds.

        The adapter is rebuilt from the source's row, as a restart would, and
        the source stays registered throughout: it is not pending, and it is
        not announced to the precache as a new source. A failure marks the
        source failed (pending again, with the error) and is a client's to retry.
        """
        if self._server.sources.get(source_id) is not None:
            return True
        with self._lock:
            if source_id not in self._state.claims:
                return True
        with self._registration_lock(source_id):
            with self._lock:
                claim = self._state.claims.get(source_id)
                signatures = self._source_signatures.get(source_id)
            if claim is None or self._server.sources.get(source_id) is not None:
                return True
            catalog_url = self._catalog_url_for(claim)
            errors: List[str] = []
            if self._register_source_claim(
                claim,
                catalog_url=catalog_url,
                replace=True,
                error_sink=errors,
                signatures=signatures,
                hydrate=(
                    self._metadata_db.read_hydration(source_id)
                    if self._metadata_db is not None
                    else None
                ),
            ):
                return True
            with self._lock:
                self._pending[source_id] = catalog_url
            self._mark_registration_failed(source_id, errors)
            return False

    def materialize(self, source_id: str) -> None:
        """Register a pending source now, for a client that resolved it.

        Returns once the source is registered, or is not pending (unknown,
        removed). Raises ``SourceRegistrationError`` when its registration failed.
        """
        if self.ensure_registered(source_id):
            return
        self.check_registered(source_id)

    def check_registered(self, source_id: str) -> None:
        """Raise why a read of a source with no adapter cannot be served.

        ``SourceRegistrationError`` when its registration failed, an unresolved
        error while it waits -- either way the client resolves it. Returns for a
        source that is unknown or has an adapter.

        A registered source whose adapter is gone, and a restored one, are the
        exceptions to "never registers": their rows are complete and list as
        resolved, so a read rebuilds the adapter here, once (the registration
        is single-flight), instead of asking a client to resolve it.
        """
        with self._lock:
            restored = source_id in self._restored
        if (restored or not self.is_pending(source_id)) and self.ensure_registered(
            source_id
        ):
            return
        with self._lock:
            if source_id not in self._pending:
                return
            error = self._pending_failed.get(source_id)
            recall = source_id in self._recall
        if error:
            raise SourceRegistrationError(source_id, error)
        what = "download and register" if recall else "register"
        raise SourceUnresolvedError(
            f"source {source_id!r} is unresolved: resolve it to {what} it"
        )

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
                self._metadata_db.sync_pending_source(
                    claim,
                    catalog_url,
                    error=message,
                    record=self._state_record(claim),
                )
            except Exception:
                logger.exception("could not record the failure of source %s", source_id)

    def _on_pending(self, source_id: str) -> None:
        """Tell whoever registers pending sources that one is waiting."""
        hook = self._pending_hook
        if hook is not None:
            try:
                hook(source_id)
            except Exception:
                logger.exception("pending-registration hook failed for %s", source_id)

    def _reconcile_root(
        self,
        snapshot: Dict[str, SourceClaim],
        discovered_state: DiscoveryState,
        recurring: bool,
    ) -> None:
        """Bring one root's claims in line with what its walk found.

        *snapshot* is the root's confirmed claims (:meth:`claims_under`) taken
        BEFORE the walk; the walk has already committed every claim that was new,
        and those are not in it, so what is left to decide is what the walk did
        not find (removal), and what it found unchanged or changed. A claim the
        walk committed is never stat'ed again to be compared with itself.

        *recurring* is a monitored root, which is walked again: a claim must be
        missing on ``_MISSES_BEFORE_REMOVAL`` walks running before it is removed,
        and a claim is removed or rebuilt only once quiet (the stability window).
        A scan-once root has no later walk to correct a miss, so a claim it did
        not find is removed at once, after a second look by its adapter, and the
        stability gate is off.
        """
        discovered_claims = discovered_state.claims
        current_ids = set(snapshot)
        discovered_ids = set(discovered_claims)

        changed_ids: Set[str] = set()
        unchanged_ids: List[str] = []
        for source_id in current_ids & discovered_ids:
            new_signatures = self._build_claim_signatures(discovered_claims[source_id])
            existing_signatures = self._source_signatures.get(source_id)
            if existing_signatures is None:
                self._source_signatures[source_id] = new_signatures
                continue
            if not _same_signature(existing_signatures, new_signatures):
                changed_ids.add(source_id)
            else:
                unchanged_ids.append(source_id)
        # Known and unchanged, but not registered: its row may be waiting on the
        # wrong thing (a file that became resident, or stopped being).
        for source_id in unchanged_ids:
            self._settle_unresolved_claim(discovered_claims[source_id])

        # A claim found again forfeits its count.
        for source_id in current_ids & discovered_ids:
            self._missed_scans.pop(source_id, None)
        self._remove_absent(snapshot, discovered_ids, recurring=recurring)
        # A changed source is REBUILT in place rather than removed and re-added:
        # the replacement adapter is registered on top of the live one, so the
        # source is never absent from ListFlights or the catalog and a failed
        # rebuild does not cost a working source. A recurring root's claim is
        # rebuilt only once quiet: rebuilding mid-write reads a half-written file.
        refreshed_ids = [
            source_id
            for source_id in sorted(changed_ids)
            if not recurring or self._claim_is_quiet(snapshot[source_id])
        ]

        for source_id in refreshed_ids:
            self._refresh_claim(discovered_claims[source_id])

    def _remove_absent(
        self,
        snapshot: Dict[str, SourceClaim],
        found_ids: Set[str],
        *,
        recurring: bool,
    ) -> List[str]:
        """Remove the claims in *snapshot* that a walk did not find again.

        The one removal rule of every scan; what differs is how sure it must be.
        A *recurring* root is walked again, so it waits:

        * a claim must be missing on ``_MISSES_BEFORE_REMOVAL`` walks running;
        * and quiet, the stability window the claim gate applies on the way in, so
          a source being rewritten is not removed for an adapter that declines its
          half-written file.

        A root walked once (a scan-once root, a drop) has no later walk to correct
        a transient decline (a sidecar being rewritten, a locked header), so it
        removes at once, but only if no adapter claims the path again on a second
        look. A claim whose primary path is gone skips the re-probe: nothing to
        claim.

        Returns the ids removed.
        """
        removed: List[str] = []
        for source_id in sorted(set(snapshot) - found_ids):
            claim = snapshot[source_id]
            if recurring:
                misses = self._missed_scans.get(source_id, 0) + 1
                self._missed_scans[source_id] = misses
                if misses < _MISSES_BEFORE_REMOVAL or not self._claim_is_quiet(claim):
                    continue
            elif os.path.exists(claim.primary_path) and self._claimed_again(claim):
                continue
            if self._commit_remove_source(source_id):
                removed.append(source_id)
                logger.info("Deregistered source %s: no longer found", source_id)
        return removed

    def _claimed_again(self, claim: SourceClaim) -> bool:
        """Whether an adapter claims ``claim``'s primary path right now.

        An adapter can briefly decline a claim it made (a sidecar being rewritten,
        a locked header); a root with no later walk would lose a working source to
        that, so a claim is re-probed once before it is removed.
        """
        try:
            again = self._registry.get_claims_for_path(
                ClaimContext(
                    Path(claim.primary_path),
                    cloud_root=self._roots.is_cloud(claim.primary_path),
                ),
                DiscoveryState(),
            )
        except Exception:
            return False
        return any(c.primary_path == claim.primary_path for c in again)

    def _settle_unresolved_claim(self, claim: SourceClaim) -> None:
        """Re-decide what a known, unchanged, unregistered claim is waiting for.

        Waiting for the pool (``pending``) when its content is resident, for a
        client (``needs_recall``) when it is not. A failed row is left alone: it
        leaves that state only by a new signature, a drop or a resolve.
        """
        source_id = claim.source_id
        # The registration lock, so the row is not written over a registration
        # that finishes while this decides.
        with self._registration_lock(source_id):
            with self._lock:
                if source_id not in self._pending or source_id in self._pending_failed:
                    return
                waiting_on_client = source_id in self._recall
                catalog_url = self._pending[source_id]
            if self._claim_is_unresolved(claim) == waiting_on_client:
                return
            recall = not waiting_on_client
            try:
                if self._metadata_db is not None:
                    self._metadata_db.sync_pending_source(
                        claim,
                        catalog_url,
                        recall=recall,
                        record=self._state_record(claim),
                    )
            except Exception:
                logger.exception("could not update the row of source %s", source_id)
                return
            with self._lock:
                if recall:
                    self._recall.add(source_id)
                else:
                    self._recall.discard(source_id)
        if not recall:
            self._on_pending(source_id)

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
        if self._roots.is_cloud(claim.primary_path):
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

    def _catalog_record(
        self,
        claim: SourceClaim,
        signatures: Optional[Dict[str, Tuple[Any, ...]]] = None,
    ) -> Optional[CatalogRecord]:
        """Where *claim*'s source sits in the catalog, and what it persists.

        A source under a root the server knows sits under it: a local source under a
        monitored or scan-once root (cloud roots included) with its claim and
        signature, which a restart rebuilds it from; a drop under that root without
        them, since nothing would rebuild it. Any other has none, and the catalog
        files it under its built-in root. Whether the adapter can also be rebuilt
        is its own ``catalog_payload``.

        The persisted signature drops ``st_dev`` (first element), which can
        renumber across boots and would make every row look changed.

        *signatures* is the claim-time signature, from a caller that has just taken
        it (the pending commit) or that wants the one state holds (a registration
        after the claim was committed pending, which must persist the signature
        taken then, not a new one; a refresh must not, since state holds the old
        file's). None means stat now.
        """
        from biopb_tensor_server.serving.metadata_db import CatalogRecord

        if claim.is_remote:
            return None
        root = self._roots.containing(Path(claim.primary_path))
        if root is None:
            return None
        rel = path_under_root(root.url, claim.primary_path)
        if not root.persisted:
            self._ensure_root(root)
            return CatalogRecord(None, {}, root.root_id, rel)
        if signatures is None:
            signatures = self._build_claim_signatures(claim)
        signature = {path: sig[1:] for path, sig in signatures.items()}
        return CatalogRecord(
            claim=claim,
            signature=signature,
            root_id=root.root_id,
            rel=rel,
            cloud=root.cloud,
        )

    def _ensure_root(self, root: Root) -> None:
        """Have the catalog know a root that is not persisted before a row sits under
        it. Once per root: its id names its url and the alias that namespaces it."""
        if self._metadata_db is None or root.root_id in self._catalog_roots:
            return
        self._metadata_db.ensure_root(root.root_id, root.root_url)
        self._catalog_roots.add(root.root_id)

    def _state_record(self, claim: SourceClaim) -> Optional[CatalogRecord]:
        """The record of a claim already committed, with the signature state holds."""
        return self._catalog_record(claim, self._source_signatures.get(claim.source_id))

    def _preserve_skipped_claims(
        self,
        snapshot: Dict[str, SourceClaim],
        discovered_state: DiscoveryState,
        skipped_dirs: Set[str],
    ) -> None:
        """Carry forward the root's claims whose subtree was intentionally skipped.

        A directory the walk declined (the stability gate, or the skip policy) is
        not evidence that what is registered under it is gone.
        """
        if not skipped_dirs:
            return

        skipped_paths = [Path(path_str) for path_str in sorted(skipped_dirs)]
        for source_id, claim in snapshot.items():
            if source_id in discovered_state.claims:
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

    def _stream_claim_add(self, claim: SourceClaim) -> None:
        """Commit a new claim live, as the walk discovers it.

        Wired as the discovery state's ``on_source_added`` for every walk. A claim
        already in state is left to the end-of-walk reconcile, which compares its
        signature; one that is not is new, and routes through ``_commit_add_claim``
        so a streamed add gets the same server registration, metadata-DB sync,
        signature bookkeeping and precache gating as a reconcile-driven add.

        Idempotent against a *retried* walk: a source already committed by a prior
        partial one is skipped, so the duplicate-rollback path in
        ``_commit_add_claim`` -- which would *unregister* it -- is never hit.
        """
        with self._lock:
            known = self._state.claims.get(claim.source_id)
        if known is not None:
            self.is_held_under_another_path(claim)
            return
        self._commit_add_claim(claim, keep_failed=True)

    def is_held_under_another_path(self, claim: SourceClaim) -> bool:
        """Whether *claim*'s source is already held, spelled by a different path.

        The same file reached by a link or an overlapping root. It stays where it
        is: re-spelling it would move it to another root, and so possibly change
        its url. Warns once.
        """
        with self._lock:
            known = self._state.claims.get(claim.source_id)
        if known is None or known.primary_path == claim.primary_path:
            return False
        self._warn_shared_source(known, claim)
        return True

    def _warn_shared_source(self, known: SourceClaim, found: SourceClaim) -> None:
        """Say once that one file was reached by two paths.

        The id hashes the resolved path, so two spellings with one id are one file:
        a link, or two roots that overlap. The scan assumes neither happens, so
        this is the one place it is noticed. The source stays under the first path.
        """
        if known.source_id in self._warned_shared:
            return
        self._warned_shared.add(known.source_id)
        logger.warning(
            "Source %s is reachable by two paths, %s and %s; it is catalogued under "
            "the first. Roots must not nest or share files, and a link must not lead "
            "into another root.",
            known.source_id,
            known.primary_path,
            found.primary_path,
        )

    def _commit_add_claim(
        self,
        claim: SourceClaim,
        catalog_url: Optional[str] = None,
        keep_failed: bool = False,
    ) -> bool:
        """Register a discovered source, then commit it into confirmed state.

        ``keep_failed`` (the walker's adds): a claim whose registration fails is
        committed anyway, as a failed row, so the walk does not find it new and
        try again. A drop reports the failure to its caller instead, and leaves
        nothing behind.

        ``catalog_url`` overrides the descriptor's display ``source_url``
        (drag-drop re-rooting, see
        ``_drop_catalog_url``); ``None`` falls back to the display root of the
        monitored directory the claim sits under, if that has one.
        """
        if catalog_url is None:
            catalog_url = self._catalog_url_for(claim)
        if self._claim_is_unresolved(claim):
            return self._commit_pending_claim(claim, catalog_url, recall=True)
        if self._defer_registration and self._deferrable(claim):
            return self._commit_pending_claim(claim, catalog_url)
        # Before the parse, and the one the registration persists and state keeps.
        signatures = self._build_claim_signatures(claim)
        errors: List[str] = []
        if not self._register_source_claim(
            claim,
            catalog_url=catalog_url,
            error_sink=errors,
            signatures=signatures,
        ):
            if keep_failed:
                self._commit_failed_claim(claim, catalog_url, errors)
            return False

        with self._lock:
            added = self._state.add_claim(claim, notify=False)
            if not added:
                self._rollback_source_registration(claim.source_id)
                return False
            self._source_signatures[claim.source_id] = signatures

        # Route the freshly committed source to the precache worker. The
        # live-vs-startup gate (and the best-effort hook invocation) lives in the
        # injected SourceManager callback, which owns the startup/suppress state.
        self._notify_source_committed(claim.source_id)
        return True

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
        if self._claim_is_unresolved(claim):
            return self._refresh_recall_claim(claim, previous)

        catalog_url = self._display_url_of(claim.source_id)

        # Before the parse: the new file's identity, not the one state holds, is
        # what the rebuilt row persists and what state keeps afterwards.
        signatures = self._build_claim_signatures(claim)
        errors: List[str] = []
        if not self._register_source_claim(
            claim,
            catalog_url=catalog_url,
            replace=True,
            error_sink=errors,
            signatures=signatures,
        ):
            # A source that was serving keeps its adapter and its row; one that
            # was still pending becomes a failed row. Only the walk removes one.
            if self.is_pending(claim.source_id):
                self._mark_registration_failed(claim.source_id, errors)
            return False

        with self._lock:
            self._replace_claim_locked(claim, previous, signatures)
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

    def _display_url_of(self, source_id: str) -> Optional[str]:
        """The catalog url a source is shown under, registered or still pending.

        The display url is the source's, not a rebuild's. Re-deriving it would
        either hand a dnd:// drop its native file:// url back -- losing its
        removability, since remove_source authorizes on that scheme -- or stamp
        the marker onto a monitored source, where it would falsely promise that
        nothing will re-add it.
        """
        live = self._server.sources.get(source_id)
        if live is not None:
            return getattr(live, "_catalog_url", None)
        if source_id in self._pending:
            return self._pending[source_id]
        claim = self._state.claims.get(source_id)
        if claim is not None and source_id in self._server.sources:
            return self._catalog_url_for(claim)  # registered, adapter let go of
        return None

    def _replace_claim_locked(
        self,
        claim: SourceClaim,
        previous: SourceClaim,
        signatures: Dict[str, Tuple[Any, ...]],
    ) -> None:
        """Put a refreshed source's new claim in state. Caller holds ``self._lock``.

        Membership moves under a source (a file added to a sequence dir), so the
        claim is replaced rather than left at what was discovered when it was
        first registered. State must track the NEW membership even where it
        overlaps another source's claim: the old membership would describe an
        adapter that no longer exists. Hence replace_claim rather than
        add_claim's reject-on-conflict.
        """
        self._state.remove_claim(previous.primary_path, notify=False)
        conflicting = self._state.replace_claim(claim)
        if conflicting:
            logger.error(
                "Refreshed source %s now overlaps another source's claim "
                "on %s; those paths remain attributed to the other source",
                claim.source_id,
                sorted(conflicting),
            )
        self._source_signatures[claim.source_id] = signatures

    def _refresh_recall_claim(self, claim: SourceClaim, previous: SourceClaim) -> bool:
        """Refresh a cloud source that is not resident: back to ``needs_recall``.

        Nothing is opened. Its adapter, if it had been resolved, is dropped, since
        the bytes behind it changed or went away; the row says it needs a recall
        again, and the next ``resolve`` rebuilds it.
        """
        source_id = claim.source_id
        catalog_url = self._display_url_of(source_id)
        signatures = self._build_claim_signatures(claim)
        try:
            if self._metadata_db is not None:
                self._metadata_db.sync_pending_source(
                    claim,
                    catalog_url,
                    recall=True,
                    record=self._catalog_record(claim, signatures=signatures),
                )
            if self._server.sources.get(source_id) is not None:
                self._server.unregister_source(source_id)
        except Exception:
            logger.exception("could not refresh cloud source %s", source_id)
            return False
        with self._lock:
            self._replace_claim_locked(claim, previous, signatures)
            self._pending[source_id] = catalog_url
            self._pending_failed.pop(source_id, None)
            self._recall.add(source_id)
        logger.info(f"Refreshed source: {source_id} (needs recall)")
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
            self._missed_scans.pop(source_id, None)
            self._clear_pending(source_id)
        return True

    def _reconcile_one_upstream(self, root: Root) -> bool:
        """Re-list one upstream's sources: see :meth:`MirrorSet.relist`.

        Returns whether the mirrored set changed (a source was added or removed).
        """
        mirrors = self._mirrors.get(root)
        if mirrors is None:
            _warn_experimental_source("tensor-server")
            mirrors = self._mirrors[root] = MirrorSet(
                root,
                self._server,
                self._metadata_db,
                self.has_claim,
                self._ensure_root,
            )
        return mirrors.relist(self._credentials_config)

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
        ``S_ISREG`` so a directory's ``st_blocks == 0`` (macOS APFS) is never a
        false hit.
        """
        for member in claim.member_paths:
            if is_remote_url(member):
                continue
            member_path = Path(member)
            try:
                st = member_path.stat()
                if stat.S_ISREG(st.st_mode) and _is_offline_placeholder(
                    member_path, st
                ):
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
            return source_is_resident(adapter.source_url)
        except Exception:  # noqa: BLE001 -- a gate that cannot see fails closed
            return False

    def _build_recalled_adapter(
        self, claim: SourceClaim, source_config: SourceConfig
    ) -> Any:
        """Open a cloud source that was left unopened, downloading it if dehydrated.

        Re-runs the claim on the path as it is now: the type recorded at scan
        time was a recall-free guess (a ``.zarr`` provisionally typed ome-zarr may
        be ome-zarr-hcs), and the authoritative one comes from the content. A
        fresh DiscoveryState keeps the scan's consumed paths from suppressing the
        source's own member claims.

        An I/O failure while the sync engine delivers the bytes raises
        ``SourceResolveRetriableError``; a format failure, or no adapter for the
        type, raises ``SourceUnresolvedError`` and will not clear on a retry.
        """
        source_id = claim.source_id
        resolved_type = source_config.type or "unknown"
        try:
            ctx = ClaimContext(
                Path(source_config.url),
                cloud_root=self._roots.is_cloud(claim.primary_path),
            )
            claims = self._registry.get_claims_for_path(ctx, DiscoveryState())
        except OSError as e:
            # Not the claim-time guess: that would launder a network blip into a
            # wrong type.
            raise SourceResolveRetriableError(
                f"source {source_id!r} could not be resolved "
                f"(re-claim recall/IO failed): {e}"
            ) from e
        except Exception as e:  # a non-IO claim error: no claim, keep the guess
            logger.debug("re-claim during resolution failed for %s: %s", source_id, e)
            claims = []
        if claims:
            resolved_type = claims[0].source_type

        adapter_cls = self._registry.get_adapter_for_type(resolved_type)
        if adapter_cls is None:
            raise SourceUnresolvedError(
                f"source {source_id!r} could not be resolved: no adapter "
                f"for type {resolved_type!r}"
            )
        config = SourceConfig(
            url=source_config.url,
            type=resolved_type,
            source_id=source_id,
            credentials_profile=source_config.credentials_profile,
            alias=source_config.alias,
        )
        try:
            return adapter_cls.create_from_config(config, self._credentials_config)
        except SourceUnresolvedError:
            raise
        except OSError as e:
            raise SourceResolveRetriableError(
                f"source {source_id!r} could not be resolved "
                f"(open/hydrate recall/IO failed): {e}"
            ) from e
        except Exception as e:
            raise SourceUnresolvedError(
                f"source {source_id!r} could not be resolved (open/hydrate failed): {e}"
            ) from e

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
        catalog_url: Optional[str] = None,
        replace: bool = False,
        error_sink: Optional[List[str]] = None,
        recall: bool = False,
        signatures: Optional[Dict[str, Tuple[Any, ...]]] = None,
        hydrate: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    ) -> bool:
        """Create and register a source, rolling back on partial failure.

        ``hydrate`` is a restored row's ``(payload, metadata)``: the adapter is
        rebuilt from it when its type can be, and parsed from the claim when not.

        ``signatures`` is the claim's member signature, taken by a caller that
        will keep it in state: it is persisted beside the parse and the caller
        commits the same one, so the members are stat'ed once. Taken before the
        adapter is built, a file that changes during the parse then reads as
        changed on the next scan. Without it a pending source's claim-time
        signature is used (state holds it), else the members are stat'ed now.

        ``recall`` opens a cloud source that was left unopened (see
        :meth:`_build_recalled_adapter`); an error that says why it cannot be
        opened is raised, not recorded.

        ``error_sink`` collects the text of the failure, for a caller that records
        it on the source (a deferred registration leaves it in the row).

        ``replace`` re-registers a source that is already live: the new adapter
        is swapped in on top of the old one and the old one is closed only after
        the swap and the catalog upsert have both succeeded. Ordering matters --
        unregister-then-register would leave the source absent from ListFlights
        and its catalog row deleted for the length of the rebuild, and gone for
        good if the rebuild then failed.

        ``catalog_url`` (drag-drop re-rooting) overrides the display
        ``source_url`` on the adapter *before* register/sync so both ListFlights
        and the metadata DB record the re-rooted url.
        """
        self._warn_if_experimental(claim)
        # Whether the adapter was rebuilt from the row it is already catalogued by.
        from_payload = False
        # Taken before the adapter is built, so a file that changes during the
        # parse is recorded with the identity the parse started from.
        if signatures is None and self.is_pending(claim.source_id):
            signatures = self._source_signatures.get(claim.source_id)
        record = self._catalog_record(claim, signatures)
        try:
            source_config = self._source_config_for(claim)

            if recall:
                adapter = self._build_recalled_adapter(claim, source_config)
            else:
                adapter_cls = self._registry.get_adapter_for_type(claim.source_type)
                if adapter_cls is None:
                    if error_sink is not None:
                        error_sink.append(f"no adapter for type {claim.source_type}")
                    logger.error(
                        "No adapter for type %s for source %s (%s)",
                        claim.source_type,
                        claim.source_id,
                        claim.primary_path,
                    )
                    return False

                adapter = None
                if hydrate is not None:
                    try:
                        adapter = adapter_cls.create_from_payload(
                            source_config, *hydrate, self._credentials_config
                        )
                        from_payload = adapter is not None
                        if from_payload:
                            self._stamp_persisted_version(adapter, claim)
                    except Exception:
                        logger.warning(
                            "could not rebuild source %s from its stored payload; "
                            "parsing it",
                            claim.source_id,
                            exc_info=True,
                        )
                if adapter is None:
                    adapter = adapter_cls.create_from_config(
                        source_config, self._credentials_config
                    )
        except UpstreamConfigError as e:
            if error_sink is not None:
                error_sink.append(str(e))
            # Same skip, different diagnosis: "failed to create adapter" reads as
            # a transient upstream problem, and this one is an operator edit away
            # from fixed and will fail identically until then (biopb/biopb#608).
            logger.error(
                "Source %s (%s) is MISCONFIGURED and cannot be registered: %s. "
                "Correct the configuration; retrying will not help.",
                claim.source_id,
                claim.primary_path,
                e,
            )
            return False
        except Exception as e:
            if recall and isinstance(e, SourceUnresolvedError):
                raise
            if error_sink is not None:
                error_sink.append(str(e))
            logger.exception(
                "Failed to create adapter for source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
            )
            return False

        # Drag-drop re-rooting: stamp the display-only source_url override before
        # register/sync so ListFlights and the metadata-DB row both carry it.
        if catalog_url:
            adapter._catalog_url = catalog_url
        registered = False
        displaced: Optional[Any] = None
        try:
            # What can be rebuilt from its row may be let go when idle: a local
            # file source with a claim. A proxy and an upload are held for as
            # long as they are registered. A cloud source waiting for a recall is
            # pending, not registered, and is never rebuilt here.
            evictable = self._deferrable(claim)
            if replace:
                adapter, displaced = self._server.swap_source(
                    claim.source_id, adapter, evictable
                )
            else:
                self._server.register_source(claim.source_id, adapter, evictable)
            registered = True

            # Raises on failure -> the except below rolls back register_source,
            # so a catalog write error never leaves a source visible in
            # ListFlights but absent from DuckDB. sync_source_added is an upsert,
            # so a replace overwrites the row rather than needing it deleted
            # first -- which is what keeps the source continuously catalogued.
            # A source rebuilt from its own row is already described by it:
            # rewriting a row that can hold megabytes of metadata to say the same
            # thing is most of what a hydration would cost. It is not confirmed
            # either: only its root's walk says the files are as persisted. What
            # the row cannot know is which uploaded fields and label sets are on
            # disk now, which the registration has just attached, so it relists
            # the tensors.
            if self._metadata_db is not None:
                if not from_payload:
                    self._metadata_db.sync_source_added(
                        claim.source_id, adapter, record
                    )
                else:
                    self._relist_tensors(claim.source_id, adapter)

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
            logger.exception(
                "Failed to register/sync source %s (%s): %s",
                claim.source_id,
                claim.primary_path,
                e,
            )
            if registered and displaced is not None:
                # A failed REBUILD must not cost the working source: put the
                # adapter that was serving back. Its row is as it was, since the
                # one write that could have changed it is the last step that
                # can fail (one transaction, nothing after it raises).
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

    def _relist_tensors(self, source_id: str, adapter: Any) -> None:
        """Bring a hydrated source's listed tensors up to what it serves.

        Best-effort: a stale listing is what the row already was, so a failure
        here must not cost the registration the source's pixels.
        """
        try:
            if self._metadata_db.relist_tensors(source_id, adapter):
                logger.info(f"Relisted the tensors of restored source {source_id}")
        except Exception:
            logger.warning(
                "could not relist the tensors of %s", source_id, exc_info=True
            )

    def _stamp_persisted_version(self, adapter: Any, claim: SourceClaim) -> None:
        """Version a source rebuilt from its row by the file state it was parsed at.

        An adapter stamps ``content_version`` from the file's stat when it is built,
        which for a rebuilt one is after the file may have changed. Chunks read
        through the old layout would then be cached under the new version and
        outlive the walk's refresh. The persisted ``mtime_ns:size`` is what a
        registered source holds, so the same read lands under a version the refresh
        replaces. Only a file with a versioned adapter: a directory's persisted
        signature has no size.
        """
        held = (self._source_signatures.get(claim.source_id) or {}).get(
            str(claim.primary_path)
        )
        if held is not None and len(held) == 5 and adapter.content_version is not None:
            adapter._content_version = f"{held[3]}:{held[2]}".encode()

    def _restore_displaced_source(self, source_id: str, displaced: Any) -> None:
        """Put a swapped-out adapter back after a failed replace (best-effort)."""
        try:
            self._server.swap_source(source_id, displaced)
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
        with differing abort semantics).
        """
        writer = self._pending_writer
        if writer is not None:
            # Before the delete, so a buffered row cannot be written back after it.
            writer.discard(source_id)
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
