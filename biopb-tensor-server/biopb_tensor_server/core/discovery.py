"""Discovery module for tensor data sources.

Provides a claim-based discovery architecture where each adapter can
"claim" filesystem paths it recognizes. This enables:

1. Extensible format detection - new adapters register and participate
2. Cross-platform file identity tracking (symlink/hardlink safe)
3. Future filesystem monitoring compatibility (DiscoveryState)
4. Remote storage discovery via fsspec (S3, GCS, HTTP)

Key components:
- SourceClaim: Represents a claimed data source (str paths for URL support)
- AdapterRegistry: Registry of all adapter backends with remote claim support
- DiscoveryState: Persistent state for incremental discovery
- discover_sources(): Main discovery function (local + remote)
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

# is_remote_url's canonical home is core.remote (it decides whether a URL needs
# a RemoteStore). Imported here for generate_source_id. Safe at module load:
# core.remote imports only stdlib at module level, so there is no import cycle.
from biopb_tensor_server.core.remote import is_remote_url

if TYPE_CHECKING:
    from biopb_tensor_server.core.adapter_base import SourceAdapter
    from biopb_tensor_server.core.remote import RemoteStore

logger = logging.getLogger(__name__)


def get_file_identity(path: Path, stat_result: Optional[os.stat_result] = None) -> str:
    """Get cross-platform stable file identity.

    Uses device + inode on Unix/NTFS, falls back to path hash on FAT32.

    Args:
        path: Path to get identity for (should already be resolved).
        stat_result: A ``stat`` of ``path`` the caller already holds. When given,
            its ``(st_dev, st_ino)`` are used directly and no syscall is issued —
            the walks stat every entry once for the signature, so re-resolving and
            re-stat'ing here was pure waste (biopb/biopb#56). ``path`` must be the
            resolved path the stat was taken of, so the hash fallback stays stable.

    Returns:
        Stable identity string for deduplication
    """
    try:
        if stat_result is None:
            path = path.resolve()
            stat_result = os.stat(path)

        # st_ino > 0 on Unix and Windows NTFS (sometimes)
        if stat_result.st_ino > 0:
            # Combine device + inode for uniqueness across mount points
            return f"{stat_result.st_dev}:{stat_result.st_ino}"
        # FAT32 or filesystem without stable inodes: hash the path as fallback.
        return _hash_path(path)
    except OSError:
        # Permission issue or broken symlink
        return _hash_path(path)


def _hash_path(path: Path) -> str:
    """Hash path for identity when inode unavailable."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


# Directory names that are never microscopy-data roots but are common, enormous,
# and — fatally on Windows/WSL — full of OneDrive "Files On-Demand" placeholders
# whose content recalls (and can hang) on read. Recursive discovery must never
# descend into them no matter how broad a root it is pointed at; a Windows user
# profile (e.g. /mnt/c/Users/<user>) otherwise stalls the server before it binds.
# Matched case-insensitively against the bare directory name.
_SKIP_DIR_NAMES = frozenset(
    {
        "appdata",
        "$recycle.bin",
        "$winreagent",
        "system volume information",
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "recovery",
        "node_modules",
    }
)

# Windows file-attribute bits marking content that is not resident on local disk
# (cloud placeholder / HSM stub). Defined numerically because the stdlib ``stat``
# module exposes only some of them. Reading such a file triggers an on-demand
# recall that can block indefinitely — OneDrive Files On-Demand over WSL ``drvfs``
# is the motivating case — so discovery skips it rather than let an adapter open
# it to sniff its format.
_FILE_ATTRIBUTE_OFFLINE = 0x00001000
_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_OFFLINE_ATTR_MASK = (
    _FILE_ATTRIBUTE_OFFLINE
    | _FILE_ATTRIBUTE_RECALL_ON_OPEN
    | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)

# Skipping suspected cloud placeholders is best-effort and on by default; the
# POSIX signal (zero allocated blocks) is a heuristic, so allow an escape hatch in
# case a filesystem reports it spuriously and discovery wrongly skips real files.
_SKIP_OFFLINE = os.environ.get("BIOPB_DISCOVERY_SKIP_OFFLINE", "1") != "0"


def _is_skippable_system_dir(name: str) -> bool:
    """True for well-known system/cloud directory names discovery must not enter.

    Covers the fixed names in ``_SKIP_DIR_NAMES`` plus OneDrive roots, which are
    named ``OneDrive`` or ``OneDrive - <Org>``.
    """
    low = name.lower()
    if low in _SKIP_DIR_NAMES:
        return True
    return low == "onedrive" or low.startswith(("onedrive -", "onedrive-"))


def _is_offline_placeholder(
    path: Path, stat_result: Optional[os.stat_result] = None
) -> bool:
    """Best-effort: True when *path*'s content is not resident on local disk.

    ``os.stat`` reads metadata only and does **not** trigger a recall, so a
    placeholder can be detected and skipped before any adapter opens the file.
    Two signals, by platform:

    - Windows: the ``FILE_ATTRIBUTE_OFFLINE`` / ``RECALL_ON_*`` bits in
      ``st_file_attributes``.
    - POSIX/WSL: zero allocated blocks (``st_blocks == 0``) — the content is
      stubbed out. No logical-size floor: a placeholder is indistinguishable
      from a resident/sparse file by stat alone (verified — drvfs reports a
      OneDrive placeholder as a plain ``regular file`` with ``blocks == 0`` and
      no reparse tag or xattr), so we skip *every* zero-block file. Missing a
      benign one (resident tiny file, sparse file, empty file) is harmless;
      opening a real placeholder while offline would block indefinitely, which
      is the failure this guard exists to prevent.

    Never raises; returns ``False`` whenever the signal is unavailable, so a
    normal file is never wrongly skipped on that account.
    """
    if not _SKIP_OFFLINE:
        return False
    st = stat_result
    if st is None:
        try:
            st = path.stat()
        except OSError:
            return False

    attrs = getattr(st, "st_file_attributes", 0)
    if attrs and (attrs & _OFFLINE_ATTR_MASK):
        return True

    st_blocks = getattr(st, "st_blocks", None)
    return st_blocks == 0


def should_skip_walk_entry(
    path: Path,
    is_dir: bool,
    stat_result: Optional[os.stat_result] = None,
    admit_nonresident: bool = False,
) -> bool:
    """Shared per-entry skip policy for discovery tree walks.

    Both traversals over the monitored trees route their skip decision through
    this one predicate — the claim walk (``walk_with_identity_tracking``) and the
    signature/stability scan (``TreeScanner._scan_tree_state``) — so the policy
    cannot drift between them. That drift is exactly what left the signature scan
    descending into OneDrive placeholders the claim walk had already learned to
    prune.

    Decides on the entry's *name* and metadata only — never opens content, so it
    cannot itself trigger a cloud recall:

    - hidden entries (name starts with ``.``);
    - well-known system/cloud directories (``_is_skippable_system_dir``);
    - files whose content is not resident locally (``_is_offline_placeholder``).

    Checked against the supplied (pre-resolution) ``path`` name so a symlink or
    junction named e.g. ``OneDrive`` is pruned by its own name, not its target's.

    Loop protection (symlinks, Windows junctions, hardlinks, bind mounts) is
    **not** handled here — it needs per-walk identity state and is applied at
    each walk's recursion point via ``get_file_identity``.

    ``stat_result``, when supplied, is a stat of ``path`` the caller already holds;
    it is forwarded to the offline-placeholder check so a walk that has already
    stat'd the entry (the state walk) does not stat it a second time
    (biopb/biopb#56).

    ``admit_nonresident`` flips the offline-placeholder rule for a ``cloud``-opted
    root (cloud-storage phase 2): instead of skipping a dehydrated file, the walk
    admits it so ``claim()`` can register it as an *unresolved* source. The
    hidden-entry and system/cloud-directory prunes still apply -- only the
    file-residency skip is lifted, and only under an explicitly configured root.
    """
    name = path.name
    if name.startswith("."):
        return True
    if is_dir:
        return _is_skippable_system_dir(name)
    if admit_nonresident:
        return False
    return _is_offline_placeholder(path, stat_result)


class ClaimContext:
    """Unified path access for claim protocol.

    Wraps local Path or RemoteStore operations with identical interface,
    allowing claim() to work with both filesystem types uniformly.
    """

    def __init__(
        self,
        path: Path | str,
        store: Optional[RemoteStore] = None,
        is_dir: Optional[bool] = None,
        signature: Optional[Tuple] = None,
        cloud_root: bool = False,
        child_listing: Optional[List[str]] = None,
    ):
        self._store = store
        if store is not None:
            # Remote context. The local-only caches/flags below are meaningless
            # here: remote reads go through cheap range requests (no residency or
            # child-listing optimization applies), and a remote path is never a
            # "cloud root" in the placeholder sense, so every probe hits the store.
            self._path = None
            self._remote_path = str(path)
            self._cached_is_dir = None
            self._signature = None
            self._cloud_root = False
            self._child_listing = None
            return

        self._path = Path(path)
        self._remote_path = None
        # When the caller already knows this entry's kind — the snapshot-driven
        # discovery hands in the ``is_directory`` the state walk computed from its
        # single ``DirEntry.stat()`` — cache it so ``is_dir``/``is_file``/``exists``
        # answer without re-stat'ing the entry. Every registered adapter's
        # ``claim()`` opens with an ``is_file()``/``is_dir()`` gate, so each rescan
        # entry was being stat'd once per adapter (~16×) for a fact the walk already
        # held (biopb/biopb#56, items 3+4). ``join()`` sub-contexts get no cache, so
        # structural probes (``.zattrs``, ``zarr.json``, ``NDTiff.index``, …) still
        # stat live.
        self._cached_is_dir = is_dir
        # The entry's content-identity signature (st_dev, st_ino, st_size,
        # st_mtime_ns, st_ctime_ns), as computed by the state walk. Adapters that
        # open the file to sniff content (``_get_ome_metadata_from_tiff``) key a
        # process-wide cache on it so a steady-state rescan re-reads unchanged
        # headers from memory instead of disk (biopb/biopb#56, item 6). Top-level
        # contexts only; ``join()`` sub-contexts carry no signature.
        self._signature = signature
        # True when this entry lives under a ``cloud = true`` root. Lets an
        # adapter's ``claim()`` (and the resolve-time re-claim) suppress
        # content-membership multi-file grouping under cloud regardless of
        # per-file residency -- residency can't gate the resolve path, where the
        # file is already resident.
        self._cloud_root = cloud_root
        # The directory's child paths as the state walk recorded them
        # (snapshot-driven discovery only). When present, ``glob()`` serves
        # single-level name-pattern matches from this list instead of re-reading
        # the directory off disk. A directory-claiming adapter globs its candidate
        # directory up to 6× per rescan cycle (TIFF sequence: ``*.tif``, ``*.tiff``
        # + 4 metadata patterns), and on cloud storage each glob is a directory
        # enumeration round-trip (~0.5-1 s/dir on OneDrive Files-On-Demand) — yet
        # the state walk already enumerated every directory's children once, so the
        # claim phase can reuse that listing instead of re-hitting the filesystem
        # (biopb/biopb#65). ``None`` on live-walk and ``join()`` sub-contexts, which
        # fall back to a real ``glob`` (same discipline as the ``is_dir`` /
        # ``signature`` caches).
        self._child_listing = child_listing

    @property
    def cloud_root(self) -> bool:
        """Whether this path is under a configured ``cloud = true`` root."""
        return self._cloud_root

    def is_dir(self) -> bool:
        """Check if path is directory."""
        if self._store:
            return self._store.isdir(self._remote_path)
        if self._cached_is_dir is not None:
            return self._cached_is_dir
        return self._path.is_dir()

    def is_file(self) -> bool:
        """Check if path is file."""
        if self._store:
            return self._store.isfile(self._remote_path)
        if self._cached_is_dir is not None:
            # The entry exists (it came from a successful stat) and is not a
            # directory ⇒ a file for claim purposes. Differs from ``Path.is_file()``
            # (S_ISREG) only for the rare non-regular entry (socket/fifo/device),
            # which every file-gated adapter rejects at its next extension/content
            # check anyway.
            return not self._cached_is_dir
        return self._path.is_file()

    def exists(self) -> bool:
        """Check if path exists."""
        if self._store:
            return self._store.exists(self._remote_path)
        if self._cached_is_dir is not None:
            return True
        return self._path.exists()

    def read_text(self, subpath: str = "") -> str:
        """Read file contents as text.

        Args:
            subpath: Relative path within this context (empty for current path)
        """
        if self._store:
            target = (
                (self._remote_path + "/" + subpath).lstrip("/")
                if subpath
                else self._remote_path
            )
            return self._store.read_text(target)
        target = self._path / subpath if subpath else self._path
        return target.read_text()

    def join(self, subpath: str) -> ClaimContext:
        """Create context for subpath."""
        if self._store:
            new_path = (
                self._remote_path.rstrip("/") + "/" + subpath
                if self._remote_path
                else subpath
            )
            return ClaimContext(new_path, self._store)
        return ClaimContext(self._path / subpath)

    def glob(self, pattern: str) -> List[ClaimContext]:
        """Find files matching ``pattern`` in this directory (maxdepth 1).

        When a cached child listing is available (snapshot-driven discovery) and
        the pattern is a single directory level — which every directory-claiming
        adapter's claim glob is (``*.tif``, ``metadata.txt``, ``*.companion.ome``,
        …) — the matches are served by ``fnmatch``ing the cached basenames, with no
        filesystem read (biopb/biopb#65). ``fnmatch`` mirrors ``Path.glob``'s
        per-platform case sensitivity (case-sensitive on POSIX, case-insensitive on
        Windows) via ``os.path.normcase``. Multi-level patterns (containing ``/``
        or ``**``) and contexts without a cached listing fall back to a real glob.
        """
        if self._store:
            matches = self._store.find(pattern, maxdepth=1)
            return [ClaimContext(m, self._store) for m in matches]
        if self._child_listing is not None and "/" not in pattern:
            return [
                ClaimContext(Path(child))
                for child in self._child_listing
                if fnmatch.fnmatch(os.path.basename(child), pattern)
            ]
        return [ClaimContext(p) for p in self._path.glob(pattern)]

    @property
    def path_str(self) -> str:
        """Get path as string (for SourceClaim)."""
        if self._store:
            return self._store._join(self._remote_path)
        return str(self._path)

    @property
    def name(self) -> str:
        """Get filename/directory name."""
        if self._store:
            return self._remote_path.rstrip("/").split("/")[-1]
        return self._path.name

    @property
    def parent(self) -> ClaimContext:
        """Get parent directory context."""
        if self._store:
            parent_path = (
                self._remote_path.rsplit("/", 1)[0] if "/" in self._remote_path else ""
            )
            return ClaimContext(parent_path, self._store)
        return ClaimContext(self._path.parent)

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote context."""
        return self._store is not None

    @property
    def store(self) -> Optional[RemoteStore]:
        """Get underlying RemoteStore if remote."""
        return self._store

    @property
    def signature(self) -> Optional[Tuple]:
        """Content-identity signature for this entry, or None if not supplied.

        Set only on the top-level snapshot-driven contexts (the state walk's
        per-entry stat signature); ``None`` on live-walk and ``join()`` contexts,
        which signals content-probe caches to run uncached.
        """
        return self._signature

    def is_resident(self) -> bool:
        """Recall-free: is this path's content local and cheap to read right now?

        This is the per-read residency gate an adapter's ``claim()`` consults
        before opening a sidecar or container: when it returns False the read
        would trigger a whole-file cloud recall (or block offline), so the adapter
        defers and emits an *unresolved* claim instead (cloud-storage phase 2).

        Remote contexts read via cheap range requests, so they are always treated
        as resident -- remote claim behavior is unchanged. A local path is
        resident unless it is an offline cloud placeholder, detected by
        ``_is_offline_placeholder`` (a stat-only check that never opens content).
        """
        if self._store is not None:
            return True
        # The placeholder signal (st_blocks == 0) is a per-file concept; a
        # directory legitimately reports zero blocks on some filesystems (macOS
        # APFS), so treat a directory as resident -- mirrors SourceAdapter
        # .is_resident and should_skip_walk_entry (which gates on `not is_dir`).
        try:
            if self._path.is_dir():
                return True
        except OSError:
            return False
        return not _is_offline_placeholder(self._path)


def walk_with_identity_tracking(
    root: Path,
    visited_identities: Set[str],
    path_filter: Optional[Callable[[Path], bool]] = None,
    should_descend: Optional[Callable[[Path], bool]] = None,
    admit_nonresident: bool = False,
) -> Iterator[Path]:
    """Walk filesystem with cross-platform identity tracking.

    Prevents infinite loops from symlink cycles and duplicate processing
    from hardlinks.

    Args:
        root: Root directory to walk
        visited_identities: Set of already-visited file identities
        path_filter: Optional predicate; an entry is skipped when it returns False
        should_descend: Optional predicate consulted *after* a directory entry has
            been yielded (and the consumer has had a chance to claim it). When it
            returns False the walk does not recurse into that directory. This lets
            the consumer stop the walk from descending below a directory-level
            claim — e.g. a ``.zarr`` store — whose interior files can never produce
            a claim of their own (biopb/biopb#55).

    Yields:
        Paths to files/directories (not yet claimed)
    """
    try:
        for path in root.iterdir():
            try:
                is_dir = path.is_dir()
            except OSError:
                continue  # Broken entry or permission issue

            # Shared skip policy: hidden entries, system/cloud directories
            # (AppData, OneDrive, Windows, …), and offline/placeholder files
            # whose content recalls on read. Pruned by name/metadata so the whole
            # subtree is skipped without a content open that could hang.
            if should_skip_walk_entry(
                path, is_dir, admit_nonresident=admit_nonresident
            ):
                logger.debug("walk: skipping %s", path)
                continue

            if path_filter is not None and not path_filter(path):
                continue

            try:
                identity = get_file_identity(path)
            except OSError:
                continue  # Broken symlink or permission issue

            if identity in visited_identities:
                continue  # Already processed (cycle or hardlink duplicate)
            visited_identities.add(identity)

            yield path

            # Recurse into real directories (not symlinks pointing to dirs).
            # ``should_descend`` runs after the yield, so the consumer has already
            # decided whether to claim this directory; if it claimed it (e.g. a
            # zarr store), skip the subtree instead of probing every chunk file
            # for a claim that can never fire (biopb/biopb#55).
            if (
                is_dir
                and not path.is_symlink()
                and (should_descend is None or should_descend(path))
            ):
                yield from walk_with_identity_tracking(
                    path,
                    visited_identities,
                    path_filter=path_filter,
                    should_descend=should_descend,
                    admit_nonresident=admit_nonresident,
                )
    except OSError:
        # Permission issue reading directory
        pass


class SourceClaim:
    """Represents a claimed data source.

    A claim describes what paths an adapter recognizes and wants to handle.
    Claims can be single-node (one file/dir) or multi-node (multiple files).

    Uses __slots__ for memory efficiency when scanning large directories.

    Attributes:
        source_type: Type identifier ("zarr", "ome-tiff", "hdf5", etc.)
        primary_path: Main entry point for the source (str to support URLs)
        source_id: Unique identifier (auto-generated if None)
        dim_labels: Optional dimension labels
        extra_config: Adapter-specific configuration (e.g., HDF5 dataset path)
        is_remote: Flag indicating if this is a remote source
        unresolved: True when the adapter recognized this source by recall-free
            signals only (a non-resident cloud/synced-folder target) and deferred
            its content read. Such a claim carries no shape/dtype yet; the server
            registers it behind an UnresolvedSourceAdapter and resolves it lazily
            on first access (cloud-storage phase 2).
    """

    __slots__ = (
        "source_type",
        "primary_path",
        "source_id",
        "dim_labels",
        "extra_config",
        "is_remote",
        "member_paths",
        "unresolved",
    )

    def __init__(
        self,
        source_type: str,
        primary_path: Path | str,
        source_id: Optional[str] = None,
        dim_labels: Optional[List[str]] = None,
        extra_config: Optional[dict] = None,
        is_remote: bool = False,
        member_paths: Optional[Set[str] | List[str]] = None,
        unresolved: bool = False,
    ):
        self.source_type = source_type
        self.primary_path = (
            str(primary_path) if isinstance(primary_path, Path) else primary_path
        )
        self.source_id = source_id
        self.dim_labels = dim_labels
        self.extra_config = extra_config if extra_config is not None else {}
        self.is_remote = is_remote
        self.unresolved = unresolved
        normalized_member_paths = {self.primary_path}
        if member_paths is not None:
            normalized_member_paths.update(str(path) for path in member_paths)
        self.member_paths = normalized_member_paths

    def __repr__(self) -> str:
        return (
            f"SourceClaim(source_type={self.source_type!r}, "
            f"primary_path={self.primary_path!r}, "
            f"source_id={self.source_id!r}, "
            f"is_remote={self.is_remote!r}, "
            f"unresolved={self.unresolved!r})"
        )


class AdapterRegistry:
    """Registry of all adapter backends.

    Adapters register themselves and participate in discovery by
    implementing the claim() classmethod with ClaimContext and DiscoveryState.

    Usage:
        registry = AdapterRegistry()
        registry.register(ZarrAdapter, "zarr")
        registry.register(OmeZarrAdapter, ["ome-zarr", "ome-zarr-hcs"])
        ctx = ClaimContext(path)  # or ClaimContext("", store) for remote
        claims = registry.get_claims_for_path(ctx, state)
    """

    def __init__(self):
        self._adapters: List[Type[SourceAdapter]] = []
        self._type_to_adapter: Dict[str, Type[SourceAdapter]] = {}

    def register(
        self,
        cls: Type[SourceAdapter],
        source_type: Optional[str | Iterable[str]] = None,
    ) -> None:
        """Register an adapter class, mapping its source type(s) to it.

        Args:
            cls: SourceAdapter subclass with a claim(ctx, state) method.
            source_type: The type string this adapter serves, or an iterable of
                them when one class serves several (OME-Zarr serves both
                ``ome-zarr`` and ``ome-zarr-hcs``). Recorded here, at
                registration, so ``get_adapter_for_type`` resolves a type
                *before* any path of that type has been claimed -- the
                lazy-resolve / cloud phase-2 flow (``UnresolvedSourceAdapter``)
                depends on that. ``None`` registers a claim-only adapter: it
                participates in discovery probing but is not resolvable by type
                (test doubles that only exercise ``claim()``).
        """
        self._adapters.append(cls)
        if source_type is None:
            return
        types = [source_type] if isinstance(source_type, str) else source_type
        for t in types:
            self._type_to_adapter[t] = cls

    def get_claims_for_path(
        self, ctx: ClaimContext, state: DiscoveryState
    ) -> List[SourceClaim]:
        """Ask adapters to claim this path, stopping at the first winner.

        Adapters' claim(ctx, state) methods are called in registration order;
        the first to return a non-None claim wins and the rest are not probed.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            List with the single winning SourceClaim, or empty if none claims.
        """
        claims = []
        for adapter_cls in self._adapters:
            # Record exactly the paths this adapter consumes during its claim()
            # call (adapters consume via state.try_claim_path) so its members can
            # be attributed without copying the entire consumed-paths set on every
            # probe — that copy was O(entries × adapters × consumed) allocation on
            # the rescan hot path (biopb/biopb#56). primary_path is already in
            # claim.member_paths (SourceClaim.__init__), so only the newly consumed
            # members are folded in.
            recorder: List[str] = []
            state._claim_recorder = recorder
            try:
                claim = adapter_cls.claim(ctx, state)
                if claim is not None:
                    claim.member_paths.update(recorder)
                    claims.append(claim)
                    logger.debug(
                        f"Adapter {adapter_cls.__name__} claimed {ctx.path_str} as {claim.source_type}"
                    )
                    # First claim wins: callers take claims[0] and the registry
                    # order is load-bearing priority, so stop probing the
                    # remaining adapters. On cloud roots their claim() probes are
                    # network round-trips, so this avoids up to 17x the wasted
                    # stat/glob per non-matching entry (biopb/biopb#190).
                    break
            except Exception as e:
                logger.debug(
                    f"Adapter {adapter_cls.__name__} claim() raised exception: {e}"
                )
                continue
            finally:
                state._claim_recorder = None
        return claims

    def get_adapter_for_type(self, source_type: str) -> Optional[Type[SourceAdapter]]:
        """Get adapter class for a source type.

        Args:
            source_type: Source type string

        Returns:
            SourceAdapter subclass or None if not registered
        """
        return self._type_to_adapter.get(source_type)


class DiscoveryState:
    """Persistent state for incremental discovery.

    Maintains bidirectional mappings for efficient source add/remove
    operations. Designed for future filesystem monitoring support.

    Attributes:
        claims: Forward mapping (source_id → SourceClaim)
        path_to_source: Reverse mapping (primary_path → source_id)
        consumed_paths: All paths consumed by any source (Set[str] for URLs)
        visited_identities: File identities already visited
        on_source_added: Callback for source addition events
        on_source_removed: Callback for source removal events
    """

    claims: Dict[str, SourceClaim]
    path_to_source: Dict[str, str]  # Changed from Dict[Path, str]
    source_to_paths: Dict[str, Set[str]]
    consumed_paths: Set[str]  # Changed from Set[Path]
    visited_identities: Set[str]
    on_source_added: Optional[Callable[[SourceClaim], None]]
    on_source_removed: Optional[Callable[[str], None]]

    def __init__(
        self,
        on_source_added: Optional[Callable[[SourceClaim], None]] = None,
        on_source_removed: Optional[Callable[[str], None]] = None,
    ):
        self.claims = {}
        self.path_to_source = {}
        self.source_to_paths = {}
        self.consumed_paths = set()
        self.visited_identities = set()
        self.on_source_added = on_source_added
        self.on_source_removed = on_source_removed
        # Set by AdapterRegistry.get_claims_for_path around a single adapter's
        # claim() call: try_claim_path appends each path it consumes so the
        # registry can attribute members without snapshotting consumed_paths.
        self._claim_recorder: Optional[List[str]] = None

    def try_claim_path(self, path: str | Path, identity: Optional[str] = None) -> bool:
        """Check if path can be claimed and mark it as consumed.

        This is the callback for multi-file source discovery. Adapters call
        this for each path they want to claim. The method handles identity
        tracking and path consumption atomically.

        Args:
            path: Path to claim (local Path or remote URL string)
            identity: Optional pre-computed identity (computed if None)

        Returns:
            True if path is available and now claimed, False if already claimed/visited
        """
        path_str = str(path)
        if path_str in self.consumed_paths:
            return False

        if identity is None:
            try:
                identity = get_file_identity(Path(path_str))
            except OSError:
                identity = _hash_path(Path(path_str))

        if identity not in self.visited_identities:
            self.visited_identities.add(identity)

        self.consumed_paths.add(path_str)
        if self._claim_recorder is not None:
            self._claim_recorder.append(path_str)
        return True

    def add_claim(self, claim: SourceClaim, notify: bool = True) -> bool:
        """Add a claim with callback notification.

        Paths should already be consumed via try_claim_path() during discovery.

        Args:
            claim: SourceClaim to add
            notify: Whether to invoke on_source_added after storing the claim

        Returns:
            True if added, False if path already claimed
        """
        # Generate source_id if not provided
        source_id = claim.source_id or generate_source_id(
            str(claim.primary_path), claim.source_type
        )

        # primary_path is already in claim.member_paths (SourceClaim.__init__).
        member_paths = set(claim.member_paths)

        existing_owner = None
        for path in member_paths:
            owner = self.path_to_source.get(path)
            if owner is not None and owner != source_id:
                existing_owner = owner
                break

        if existing_owner is not None:
            return False

        # Update claim's source_id (important for callbacks)
        claim.source_id = source_id
        claim.member_paths = member_paths

        # Store claim
        self.claims[source_id] = claim
        self.source_to_paths[source_id] = member_paths
        for path in member_paths:
            self.path_to_source[path] = source_id
            self.consumed_paths.add(path)

        # Callback
        if notify and self.on_source_added:
            self.on_source_added(claim)

        return True

    def remove_claim(self, path: str, notify: bool = True) -> Optional[str]:
        """Remove claim by path (for file deletion events).

        Args:
            path: Primary path of the claim to remove (str to support URLs)
            notify: Whether to invoke on_source_removed after removing the claim

        Returns:
            source_id if removed, None if not found
        """
        source_id = self.path_to_source.get(path)
        if source_id is None:
            return None

        claim = self.claims.pop(source_id)
        member_paths = self.source_to_paths.pop(source_id, set(claim.member_paths))
        for member_path in member_paths:
            self.path_to_source.pop(member_path, None)
            self.consumed_paths.discard(member_path)

        # Callback
        if notify and self.on_source_removed:
            self.on_source_removed(source_id)

        return source_id

    def is_path_claimed(self, path: str) -> bool:
        """Check if a path is already part of a claim."""
        return path in self.consumed_paths

    def get_source_for_path(self, path: str) -> Optional[str]:
        """Get source_id that owns this path (reverse lookup)."""
        return self.path_to_source.get(path)

    def get_all_claims(self) -> List[SourceClaim]:
        """Get all claims as a list."""
        return list(self.claims.values())

    def get_paths_for_source(self, source_id: str) -> Set[str]:
        """Get all claimed member paths for a source."""
        return set(self.source_to_paths.get(source_id, set()))


def resolve_local_path(path: str) -> str:
    """Canonical absolute form of a LOCAL filesystem path.

    The single canonicalizer for local-path identity across the server: the
    ``source_id`` hash (``generate_source_id``) and -- in ``source_manager`` --
    the drag-drop containment guard and the static-config seed all reduce a path
    to this form, so the same physical location compares equal however it was
    spelled (symlink / junction / mapped drive / 8.3 / case / trailing sep). The
    monitored walk reaches the same form via ``Path.resolve`` on its root.
    ``Path.resolve`` resolves reparse points on Python 3.8+, so it folds those on
    Windows too.

    Local paths only: a remote URL must NOT be passed here -- ``Path.resolve``
    mangles the scheme (collapsing ``//`` and prepending the cwd); callers gate
    on ``is_remote_url`` first.
    """
    return str(Path(path).resolve())


def generate_source_id(url: str, source_type: str) -> str:
    """Generate deterministic unique source_id from URL.

    Uses SHA-256 hash of the URL to ensure uniqueness while remaining
    deterministic for the same URL.

    Args:
        url: URL or path to the data source
        source_type: Source type prefix (e.g., "zarr", "aics", "ome-zarr")

    Returns:
        Unique source_id like "zarr_a3f2b1c4d5e6"
    """
    if url is None or url == "":
        raise ValueError("Cannot generate source_id from empty URL")

    # Remote URLs must NOT go through Path().resolve(): it treats the URL as a
    # relative POSIX path, collapsing the scheme's "//" and prepending the server's
    # cwd, which makes the id non-deterministic across deployments. Hash the raw URL
    # (trailing slashes stripped so "x.zarr" == "x.zarr/"). Local paths resolve to
    # the canonical absolute path so the same location hashes identically however it
    # was spelled (resolve_local_path).
    key = url.rstrip("/") if is_remote_url(url) else resolve_local_path(url)

    hash_hex = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{source_type}_{hash_hex}"


def _record_claim(
    state: DiscoveryState,
    claims: List[SourceClaim],
    dim_labels: Optional[List[str]],
) -> Optional[SourceClaim]:
    """Finalize the winning claim from ``get_claims_for_path`` into ``state``.

    Applies the default ``dim_labels`` (only when the claim carries none) and
    registers the claim. Shared by every discovery entry point so the
    claim-finalization policy lives in one place. Returns the recorded claim, or
    ``None`` when no adapter claimed the path.
    """
    if not claims:
        return None
    claim = claims[0]
    if claim.dim_labels is None and dim_labels is not None:
        claim.dim_labels = dim_labels
    state.add_claim(claim)
    return claim


def discover_sources(
    root: Path,
    registry: AdapterRegistry,
    state: Optional[DiscoveryState] = None,
    dim_labels: Optional[List[str]] = None,
    path_filter: Optional[Callable[[Path], bool]] = None,
    admit_nonresident: bool = False,
    cloud_root: bool = False,
) -> DiscoveryState:
    """Recursive filesystem discovery with claim protocol.

    Walks the filesystem recursively, asking each registered adapter
    to claim paths it recognizes.

    Args:
        root: Root directory to scan
        registry: Adapter registry for claims
        state: Existing DiscoveryState to update (creates new if None)
        dim_labels: Optional dimension labels to apply to all claims
        admit_nonresident: Under a cloud root, admit dehydrated placeholders
            instead of skipping them.
        cloud_root: Under a cloud root, set ``ClaimContext.cloud_root`` so the
            content-membership adapters (multi-file OME-TIFF / DICOM series) fall
            back to single-file sources instead of grouping -- the same ban the
            monitored rescan applies. Keeps the static one-shot scan of a
            ``monitor=false`` cloud directory consistent with the monitored path.

    Returns:
        DiscoveryState with all discovered sources
    """
    if state is None:
        state = DiscoveryState()

    logger.debug(f"discover_sources: scanning {root}")

    if path_filter is not None and not path_filter(root):
        logger.debug(f"discover_sources: skipping filtered root {root}")
        return state

    # Get identity for root itself
    try:
        root_identity = get_file_identity(root)
        state.visited_identities.add(root_identity)
    except OSError:
        logger.debug(f"discover_sources: cannot get identity for {root}")
        return state

    # Check if root itself is a data source (e.g., a .zarr directory)
    ctx = ClaimContext(root, cloud_root=cloud_root)
    claim = _record_claim(state, registry.get_claims_for_path(ctx, state), dim_labels)
    if claim is not None:
        logger.info(f"discover_sources: root {root} claimed as {claim.source_type}")
        return state  # Root claimed, no need to recurse

    # Walk filesystem
    paths_scanned = 0
    for path in walk_with_identity_tracking(
        root,
        state.visited_identities,
        path_filter=path_filter,
        # Don't descend below a directory the consumer just claimed: everything
        # under a claimed source belongs to that source by construction, so
        # probing interior files (e.g. zarr chunk stores) is pure waste
        # (biopb/biopb#55).
        should_descend=lambda p: not state.is_path_claimed(str(p)),
        admit_nonresident=admit_nonresident,
    ):
        paths_scanned += 1
        path_str = str(path)
        if state.is_path_claimed(path_str):
            continue

        ctx = ClaimContext(path, cloud_root=cloud_root)
        _record_claim(state, registry.get_claims_for_path(ctx, state), dim_labels)

    logger.debug(
        f"discover_sources: scanned {paths_scanned} paths, found {len(state.claims)} sources"
    )
    return state


def discover_sources_from_entries(
    entries: Iterable[Tuple[str, bool, Optional[Tuple]]],
    registry: AdapterRegistry,
    state: Optional[DiscoveryState] = None,
    dim_labels: Optional[List[str]] = None,
    path_filter: Optional[Callable[[str], bool]] = None,
    skipped_dirs: Optional[Set[str]] = None,
    cloud_by_path: Optional[Dict[str, bool]] = None,
) -> DiscoveryState:
    """Claim discovery driven by a pre-built entry snapshot — no filesystem walk.

    The periodic rescan already walks every monitored tree once to capture
    stat-signatures (``TreeScanner._scan_tree_state``). That walk holds everything
    the claim phase needs — each entry's resolved path and whether it is a directory —
    so re-walking the filesystem a second time just to probe adapters is pure
    duplication (it was ~96% of the post-#61 rescan syscalls). This drives the same
    claim protocol as :func:`discover_sources` straight off that snapshot
    (biopb/biopb#56, item 4).

    ``entries`` is an ordered ``(resolved_path_str, is_dir, signature)`` stream in
    **DFS parent-first order** (the order ``TreeScanner._scan_tree_state`` inserts into its state
    dict), which is what lets a directory-level claim or skip prune its whole subtree
    before any interior entry is probed. ``signature`` is the state walk's content
    identity for the entry, carried onto the ``ClaimContext`` so content-probing
    adapters can memoize on it (biopb/biopb#56, item 6); it may be ``None``.

    Behavior is kept identical to a :func:`discover_sources` walk over the same tree:

    - ``skipped_dirs`` (stable subtrees the state walk pruned) and any directory that
      fails ``path_filter`` prune their entire subtree — mirroring how the walk does
      not descend past a filtered/skipped directory. ``skipped_dirs`` descendants are
      carried forward in the snapshot, so without this they would be re-probed; their
      claims are preserved separately (``SourceManager._preserve_skipped_claims``).
    - a claimed directory prunes its subtree (interior zarr chunk files etc. belong to
      it by construction — biopb/biopb#55).
    - ``path_filter`` (the stability gate) receives the already-resolved path string.

    The prune set is maintained as a stack, exploiting the parent-first ordering: a
    prefix is pushed when its subtree must be skipped and popped as soon as an entry
    falls outside it, so the per-entry prune check stays O(1) amortized rather than
    O(entries × prefixes).
    """
    if state is None:
        state = DiscoveryState()

    # Group the snapshot into each directory's recorded children once (O(n) by
    # parent path) so a directory's ClaimContext can serve its claim globs from
    # memory instead of re-reading the directory — the largest remaining per-cycle
    # cost in the claim phase, and on cloud storage a per-glob round-trip
    # (biopb/biopb#65). The state walk emits entries parent-first, so a directory
    # is processed before its children stream in; build the full map up front.
    entries = list(entries)
    children_by_dir: Dict[str, List[str]] = {}
    for path_str, _is_dir, _signature in entries:
        children_by_dir.setdefault(os.path.dirname(path_str), []).append(path_str)

    skipped = skipped_dirs or set()
    cloud_by_path = cloud_by_path or {}
    prune_stack: List[str] = []

    def _under(path_str: str, prefix: str) -> bool:
        return path_str == prefix or path_str.startswith(prefix + os.sep)

    for path_str, is_dir, signature in entries:
        # Drop prune prefixes we have walked out of (DFS contiguity), then skip
        # anything still beneath an active one (a claimed source or a skipped subtree).
        while prune_stack and not _under(path_str, prune_stack[-1]):
            prune_stack.pop()
        if prune_stack:
            continue

        # A stable skipped subtree: prune the root and everything beneath it.
        if path_str in skipped:
            prune_stack.append(path_str)
            continue

        # Consumed as a member of an already-recorded multi-file claim (companion
        # OME, tiff/dicom series siblings) — same skip the walk applies.
        if state.is_path_claimed(path_str):
            continue

        # Stability gate. A directory that is not yet eligible is not descended —
        # exactly as the walk's path_filter short-circuits its recursion.
        if path_filter is not None and not path_filter(path_str):
            if is_dir:
                prune_stack.append(path_str)
            continue

        ctx = ClaimContext(
            Path(path_str),
            is_dir=is_dir,
            signature=signature,
            cloud_root=cloud_by_path.get(path_str, False),
            # Only directories glob (every claim glob is maxdepth-1 over a dir's
            # children); files carry no listing.
            child_listing=children_by_dir.get(path_str) if is_dir else None,
        )
        claim = _record_claim(
            state, registry.get_claims_for_path(ctx, state), dim_labels
        )
        if claim is not None and is_dir:
            prune_stack.append(path_str)

    logger.debug("discover_sources_from_entries: found %d sources", len(state.claims))
    return state


def discover_remote_source(
    url: str,
    registry: AdapterRegistry,
    credentials_config: Optional[Any] = None,
    profile_name: Optional[str] = None,
    state: Optional[DiscoveryState] = None,
    dim_labels: Optional[List[str]] = None,
) -> DiscoveryState:
    """Discover a single remote source using fsspec.

    For remote URLs, we check if the URL itself is a data source.
    Unlike local discovery, we don't recursively scan remote directories
    by default (too slow on large buckets).

    Args:
        url: Remote URL (s3://..., gs://..., etc.)
        registry: Adapter registry for claims
        credentials_config: CredentialsConfig for authentication
        profile_name: Credential profile name to use
        state: Existing DiscoveryState to update (creates new if None)
        dim_labels: Optional dimension labels

    Returns:
        DiscoveryState with discovered remote source
    """
    from biopb_tensor_server.core.remote import RemoteStore

    if state is None:
        state = DiscoveryState()

    logger.debug(f"discover_remote_source: checking {url}")

    # Create RemoteStore for this URL
    store = RemoteStore.from_config(
        url=url,
        credentials_config=credentials_config,
        profile_name=profile_name,
    )

    # Get identity for remote path
    try:
        identity = store.get_identity("")
        if identity in state.visited_identities:
            logger.debug(f"discover_remote_source: {url} already visited")
            return state
        state.visited_identities.add(identity)
    except Exception as e:
        logger.debug(f"discover_remote_source: cannot get identity for {url}: {e}")

    # Check if root URL is a data source
    ctx = ClaimContext("", store)
    claim = _record_claim(state, registry.get_claims_for_path(ctx, state), dim_labels)
    if claim is not None:
        logger.info(f"discover_remote_source: {url} claimed as {claim.source_type}")

    return state
