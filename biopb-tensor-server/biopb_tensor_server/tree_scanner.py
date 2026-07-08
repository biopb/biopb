"""Filesystem signature scanner for the periodic catalog rescan.

Extracted from ``SourceManager`` (biopb/biopb#278 item B): the recursive tree
walk that captures a stat-signature snapshot of the monitored directories,
separated from the catalog reconciliation that consumes it. :class:`TreeScanner`
is a pure producer -- given the previous snapshot and the walk config it returns
an immutable :class:`ScanSnapshot`; the live entry-state caches, the
swap/rollback/partition orchestration, and the post-swap stability gate stay in
``SourceManager`` (which co-owns those caches with the lifecycle code).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR
from typing import Any, Dict, Optional, Set, Tuple

from biopb_tensor_server.discovery import get_file_identity, should_skip_walk_entry

logger = logging.getLogger(__name__)

# Hard backstop against unbounded recursion in the rescan walk. visited_identities
# breaks real directory loops (symlink/junction/hardlink/bind mount) by filesystem
# identity, but under a cloud root the inode is zeroed (biopb/biopb#207) so identity
# degrades to a path hash that never repeats inside a junction loop -> RecursionError,
# which the os.scandir `except OSError` does not catch (biopb/biopb#207 review). This
# cap turns any such loop into a bounded, logged skip. Real microscopy trees are far
# shallower (rarely past ~10 levels), so this only ever fires on a pathological loop.
_MAX_WALK_DEPTH = 64


@dataclass
class EntryState:
    """Cached filesystem snapshot for one walked path (file or directory).

    Collapses the former three parallel path-keyed dicts into one record:

      * ``is_directory`` / ``signature`` / ``last_changed`` -- the change-detection
        signature (identity + mtime/size tuple, plus the last observed change
        epoch);
      * ``stable_observations`` -- consecutive unchanged rescans since the last
        change (the stability-window counter);
      * ``pending_scan`` -- True until the entry passes one eligible discovery
        pass; the #53 subtree-pending prune gate keys on it.

    Under a cloud root the last two fields are inert -- cloud entries bypass the
    stability machinery (``_should_scan_resolved`` short-circuits), so nothing
    reads them; only the signature triplet is meaningful in ``_cloud_entry_states``.
    """

    is_directory: bool
    signature: Tuple[Any, ...]
    last_changed: float
    stable_observations: int = 0
    pending_scan: bool = False


@dataclass(frozen=True)
class ScanSnapshot:
    """Immutable result of one :meth:`TreeScanner.scan`.

    ``entry_states`` is the freshly-walked signature map (cloud + non-cloud
    mixed; ``cloud_by_path`` marks which are under a cloud root, carried to the
    claim phase so cloud-ness is computed once). ``skipped_dirs`` are the roots
    the walk pruned (stable subtrees, cloud subtrees on an incremental, or
    name/system skips) -- the reconcile carries their existing claims forward.
    The dicts are handed off to the caller, which owns them after this returns.
    """

    entry_states: Dict[str, EntryState]
    cloud_by_path: Dict[str, bool]
    skipped_dirs: Set[str]


@dataclass
class _WalkContext:
    """Per-refresh accumulators + invariant flags threaded through the recursive
    signature walk (:meth:`TreeScanner._scan_tree_state`).

    Everything constant across the whole refresh lives here, so the recursion
    only passes the per-entry values (``path``, ``dir_entry``, ``is_root``,
    ``cloud``, ``depth``). ``prev_entry_states`` / ``prev_cloud_entry_states``
    are the previous published snapshot, read for signature continuity (a
    carried ``last_changed`` and the stability counter). ``allow_prune`` is *not*
    here -- it is derived per call from ``is_root`` (the monitored root honors
    the ``aggressive_dir_pruning`` config; descendants always allow pruning).
    """

    now: float
    prev_entry_states: Dict[str, EntryState]
    prev_cloud_entry_states: Dict[str, EntryState]
    next_state: Dict[str, EntryState]
    # path -> whether it is under a cloud root (carried to the claim phase so
    # cloud-ness is computed once, in the walk, not re-derived per entry).
    next_cloud: Dict[str, bool]
    skipped_dirs: Set[str]
    force_full: bool
    visited_identities: Set[str]


def build_entry_signature(
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

    Load-bearing for the cloud inode-backfill skip in ``_scan_tree_state``
    (biopb/biopb#190): because this branch is identity-only, a zeroed cloud
    inode degrades it to a constant ``(0, 0)`` that is still residency-
    invariant and per-path. If you add size/mtime back to the cloud
    signature, you must restore that backfill (and reintroduce the flap).
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


def entry_change_time(stat_result: Any, now: float) -> float:
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


class TreeScanner:
    """Walks the monitored directories and captures a stat-signature snapshot.

    Stateless across calls apart from the immutable walk config: :meth:`scan`
    takes the previous snapshot and returns a fresh :class:`ScanSnapshot`,
    mutating nothing the caller owns. ``SourceManager`` publishes / rolls back /
    partitions that snapshot into its live caches.
    """

    def __init__(
        self,
        *,
        stability_window: float,
        stable_rescans_required: int,
        aggressive_dir_pruning: bool,
    ):
        self._stability_window = stability_window
        self._stable_rescans_required = stable_rescans_required
        self._aggressive_dir_pruning = aggressive_dir_pruning

    def scan(
        self,
        *,
        monitored_dirs: Set[Path],
        cloud_roots: Set[Path],
        force_full: bool,
        prev_entry_states: Dict[str, EntryState],
        prev_cloud_entry_states: Dict[str, EntryState],
    ) -> ScanSnapshot:
        """Walk every monitored tree and return a fresh signature snapshot.

        Does not publish: the returned :class:`ScanSnapshot` is the candidate
        next state, which the caller swaps in (and rolls back on failure).
        """
        ctx = _WalkContext(
            now=time.time(),
            prev_entry_states=prev_entry_states,
            prev_cloud_entry_states=prev_cloud_entry_states,
            next_state={},
            next_cloud={},
            skipped_dirs=set(),
            force_full=force_full,
            # One identity set across all monitored roots for this refresh: breaks
            # directory loops (symlink, Windows junction, hardlink, bind mount) and
            # also stops overlapping roots from walking a shared subtree twice.
            visited_identities=set(),
        )
        for monitored_dir in sorted(monitored_dirs):
            # An explicitly-configured root is honored unconditionally -- even if
            # it is a symlink, hidden, or named like a pruned system dir; the
            # skips only ever apply to entries found *inside* it. resolve() hands
            # the root to _scan_tree_state already dereferenced (so its is_symlink
            # reads False and the symlink/loop guard does not reject a symlinked
            # root), and is_root=True skips the hidden/system/offline policy for
            # this entry only. Change either and a symlinked or oddly-named
            # monitored root silently stops being scanned.
            self._scan_tree_state(
                monitored_dir.resolve(),
                ctx,
                is_root=True,
                cloud=monitored_dir.resolve() in cloud_roots,
            )
        return ScanSnapshot(
            entry_states=ctx.next_state,
            cloud_by_path=ctx.next_cloud,
            skipped_dirs=ctx.skipped_dirs,
        )

    def _scan_tree_state(
        self,
        path: Path,
        ctx: _WalkContext,
        *,
        is_root: bool = False,
        dir_entry: Optional[os.DirEntry] = None,
        cloud: bool = False,
        depth: int = 0,
    ) -> None:
        """Capture the current filesystem signature state for one subtree.

        ``ctx`` carries the refresh-wide accumulators + flags (see ``_WalkContext``);
        only the per-entry values are explicit args. ``allow_prune`` is derived
        below: the monitored root honors the ``aggressive_dir_pruning`` config,
        every descendant always allows subtree pruning.

        ``dir_entry`` is the ``os.DirEntry`` the parent's ``os.scandir`` produced for
        this entry (None for an explicitly-configured root, which the caller hands in
        already resolved). Reusing it is the heart of the #56 per-entry syscall cut:
        its ``d_type`` answers is-symlink without a syscall, its cached ``stat()`` is
        the single stat we take, and a non-symlink child of an already-resolved
        directory is canonical by construction — so we never re-``resolve()`` it.
        Only the rare symlink entry pays a ``resolve()``, to preserve the
        resolved-target keying that ``ctx.next_state`` and the loop guard depend on.
        """
        now = ctx.now
        force_full = ctx.force_full
        # The root's pruning is config-gated; descendants always prune stable
        # subtrees. (Formerly threaded as the ``allow_prune`` param -- root call
        # passed ``self._aggressive_dir_pruning``, recursion passed ``True``.)
        allow_prune = self._aggressive_dir_pruning if is_root else True
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
                #
                # ...EXCEPT under a cloud root, where that backfill os.stat() is a
                # whole-extra network round-trip per entry (the cached DirEntry.stat()
                # is free, served from the directory enumeration; the inode is the one
                # field that costs a round-trip on a synced placeholder). On a OneDrive
                # tree this backfill measured ~59s of startup (biopb/biopb#190,
                # Finding 1). We skip it under cloud, which is correct ONLY because
                # every consumer of the zeroed inode has a cloud-safe degradation --
                # do not break these invariants:
                #
                #   1. Signature. `_build_entry_signature(cloud=True)` is *already*
                #      identity-only `(st_dev, st_ino)` and excludes size/mtime/ctime
                #      on purpose (hydration bumps those -> destructive flap). With a
                #      zeroed inode it degrades to a constant `(0, 0)` per entry. That
                #      is residency-invariant (never flaps on hydrate/evict -- strictly
                #      safer than today) and it is compared *per path_str*, so distinct
                #      cloud paths never collide into one another.
                #   2. Stability gate. Under a constant `(0, 0)` signature the
                #      stability counter is meaningless -- but cloud entries
                #      *bypass* the stability window entirely (`_should_scan_resolved`
                #      returns True immediately for a cloud path), so that counter is
                #      never read under cloud. If that bypass is ever removed, this
                #      skip becomes incorrect.
                #   3. Identity / loop-breaking. `get_file_identity` falls back to a
                #      hash of the *resolved path* when `st_ino == 0` (its FAT32 path),
                #      so `visited_identities` dedup still distinguishes cloud entries
                #      without a real inode. BUT for *non-symlink* reparse points
                #      (Windows junction, bind mount) `(st_dev, st_ino)` is the only
                #      thing that breaks a directory *cycle* (symlinks are skipped
                #      separately), and a non-symlink entry is never resolve()d, so a
                #      junction loop `J -> .` yields an ever-growing path that hashes
                #      to a new identity each descent -- the cycle is never caught.
                #      That is no longer just a lost-dedup (rare aliasing); it would
                #      recurse to RecursionError. The `_MAX_WALK_DEPTH` backstop in
                #      `_scan_tree_state` bounds such a loop to a logged skip instead
                #      of crashing the refresh thread (biopb/biopb#207 review).
                #
                # NOTE: this is a Windows-only effect; on POSIX DirEntry.stat() returns
                # a real inode, so neither the backfill nor this skip ever fires there.
                if stat_result.st_ino == 0 and not cloud:
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
            # The entry EXISTS on disk -- discovery is declining to traverse a
            # system/cloud directory (e.g. OneDrive) or to read an offline
            # placeholder, NOT observing a deletion. Record a skipped *directory*
            # so the reconcile (via _preserve_skipped_claims) carries forward any
            # already-registered claim at or under it, exactly as the cloud/stable
            # skips below do. Without this, a source explicitly registered under
            # such a path is reaped by the very next reconcile as "disappeared":
            # add_source treats a drop as a root, which is exempt from this skip
            # (see _refresh_entry_state's is_root=True), so it happily indexes
            # OneDrive content -- but the monitored-tree walk that would re-find it
            # refuses to descend here, so absence from the walk must not be read as
            # deletion (biopb/biopb#309 drag-drop follow-up).
            #
            # Only directories are recorded. A name-skipped subtree (the #309 case)
            # is bounded -- the walk returns without descending, so it contributes
            # one entry. Offline-placeholder FILES, by contrast, are leaves reached
            # by descending normal directories, so recording them would grow
            # skipped_dirs O(files) (every zero-block/empty/inline file, since
            # _is_offline_placeholder flags st_blocks == 0) and feed that set into
            # _preserve_skipped_claims' per-rescan sort + per-entry Path()/
            # is_relative_to -- the pathlib hot-path cost _copy_cached_subtree_entries
            # was rewritten to string-prefix to avoid. The only source this omits is
            # a single dehydrated placeholder file added directly under a non-cloud
            # root, whose bytes are non-resident (unreadable) anyway.
            if is_directory:
                ctx.skipped_dirs.add(str(resolved_path))
            return

        path_str = str(resolved_path)
        signature = build_entry_signature(stat_result, is_directory, cloud=cloud)
        # Cloud entries live in the cloud partition (walked only on force_full);
        # read the prior signature from there so last_changed stays continuous
        # across the hourly re-walk. Non-cloud reads _entry_states as before.
        previous_entry = (
            ctx.prev_cloud_entry_states if cloud else ctx.prev_entry_states
        ).get(path_str)
        last_changed = entry_change_time(stat_result, now)
        stable_observations = 0
        if previous_entry is not None and (
            previous_entry.is_directory,
            previous_entry.signature,
        ) == (is_directory, signature):
            last_changed = previous_entry.last_changed
            stable_observations = previous_entry.stable_observations + 1
            pending_scan = previous_entry.pending_scan
        else:
            pending_scan = True
        ctx.next_state[path_str] = EntryState(
            is_directory=is_directory,
            signature=signature,
            last_changed=last_changed,
            stable_observations=stable_observations,
            pending_scan=pending_scan,
        )
        # Record cloud-ness once, here, where the walk already knows it (inherited
        # per monitored root, see _refresh_entry_state). The claim phase reads this
        # instead of re-deriving it per entry, so there is a single source of truth
        # for "is this path under a cloud root" -- consistent with the signature
        # above, which is also computed with this same `cloud`.
        ctx.next_cloud[path_str] = cloud

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
        if identity in ctx.visited_identities:
            return
        ctx.visited_identities.add(identity)

        # Cloud subtree: re-walked only on a force_full pass. Enumerating a cloud
        # root is expensive and its mtime signature is unreliable (doc S1.2), so the
        # frequent incremental rescans skip it entirely -- they neither descend nor
        # re-materialize its descendants. Cloud entries persist in
        # ``_cloud_entry_states`` (rebuilt only on force_full); the cloud sources are
        # kept registered across incrementals by the reconcile scoping (see
        # ``_reconcile_discovered_state``), not by carrying entries forward. The first
        # rescan is force_full (last-full == -inf), so a cloud root is still
        # catalogued at startup, and a brand-new cloud dataset surfaces on the next
        # force_full pass. ``skipped_dirs.add`` records the skip (asserted by tests)
        # and prunes the cloud root that was just recorded into ``next_state``.
        if cloud and not force_full:
            ctx.skipped_dirs.add(path_str)
            return

        if (
            allow_prune
            and not force_full
            # Cloud is handled by the dedicated branch above (a cloud subtree never
            # reaches this signature-based prune), so no cloud guard is needed here.
            and previous_entry is not None
            and (previous_entry.is_directory, previous_entry.signature)
            == (is_directory, signature)
            and not pending_scan
            and now - previous_entry.last_changed >= self._stability_window
            and stable_observations >= self._stable_rescans_required
            and not self._subtree_has_pending_scan(path_str, ctx)
        ):
            ctx.skipped_dirs.add(path_str)
            self._copy_cached_subtree_entries(path_str, ctx)
            return

        # Depth backstop: if identity dedup ever fails to catch a directory loop
        # (e.g. a cloud junction whose zeroed inode makes get_file_identity fall back
        # to an ever-growing path hash, biopb/biopb#207 review), the walk would recurse
        # to RecursionError -- which the os.scandir `except OSError` below does NOT
        # catch, so it propagates and kills the refresh thread. Record this entry but
        # do not descend past the cap, turning any such loop into a bounded skip.
        if depth >= _MAX_WALK_DEPTH:
            logger.warning(
                "Rescan walk hit max depth %d at %s; not descending further "
                "(possible directory loop)",
                _MAX_WALK_DEPTH,
                resolved_path,
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
                        ctx,
                        dir_entry=entry,
                        cloud=cloud,
                        depth=depth + 1,
                    )
        except OSError:
            return

    def _subtree_has_pending_scan(self, root_path_str: str, ctx: _WalkContext) -> bool:
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
        # String-prefix, not Path(cached_path).is_relative_to(root_path): same
        # per-entry pathlib-parse cost as _copy_cached_subtree_entries, on the
        # same large carried-forward entry set. Keys are resolved path strings.
        prefix = root_path_str + os.sep
        for cached_path, entry in ctx.prev_entry_states.items():
            if not entry.pending_scan or cached_path == root_path_str:
                continue
            if cached_path.startswith(prefix):
                return True
        return False

    def _copy_cached_subtree_entries(
        self, root_path_str: str, ctx: _WalkContext
    ) -> None:
        """Carry forward cached descendants when a stable subtree is skipped.

        Hot path: this runs every rescan for each skipped (stable **or cloud**)
        root and scans the entire cached entry set. Match descendants with a
        string-prefix test, not ``Path(cached_path).is_relative_to(root_path)``:
        the latter parses a pathlib ``Path`` per entry, and on a large cloud
        catalog (tens of thousands of carried-forward entries, walked only hourly
        but *re-copied* on every 30 s incremental) that per-entry parsing was the
        single dominant cost of the rescan thread -- it held the GIL for tens of
        seconds and starved the Flight serving threads, stalling reads
        (biopb/biopb). Entry keys are resolved path strings produced by the same
        walk, so a prefix test is exact; this mirrors the string-prefix prune the
        claim phase already uses (``discover_sources_from_entries._under``).
        """
        prefix = root_path_str + os.sep
        for cached_path, entry in ctx.prev_entry_states.items():
            if cached_path == root_path_str or cached_path in ctx.next_state:
                continue
            if not cached_path.startswith(prefix):
                continue
            # Carried by reference, sharing the previous-generation instance.
            # WARNING: EntryState is mutable -- `_should_scan_resolved` clears
            # `pending_scan` in place -- so a mutation of a carried record would
            # leak across generations and break the swap-then-rollback isolation
            # `_rescan_monitored_dirs` relies on (the rolled-back "previous" cache
            # would already carry the mutation). This is safe ONLY because a
            # carried entry sits under a root just added to `skipped_dirs`, which
            # the claim walk prunes (`discover_sources_from_entries._under`), so
            # `_should_scan_resolved` never runs on it -- nothing mutates a carried
            # record. If you ever mutate carried EntryStates, or decouple this
            # carry prefix from the skip prefix, copy the record here instead.
            ctx.next_state[cached_path] = entry
