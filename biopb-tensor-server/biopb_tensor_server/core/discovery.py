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
- discover_sources(): the one filesystem walker (drop, one-shot dir, rescan)
- discover_remote_source(): claim a remote URL
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import queue
import stat as stat_module
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
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


def _is_cloud_dir(name: str) -> bool:
    """True for a OneDrive root: ``OneDrive`` or ``OneDrive - <Org>``."""
    low = name.lower()
    return low == "onedrive" or low.startswith(("onedrive -", "onedrive-"))


def _is_skippable_system_dir(name: str) -> bool:
    """True for well-known system/cloud directory names discovery must not enter.

    Covers the fixed names in ``_SKIP_DIR_NAMES`` plus OneDrive roots, which are
    named ``OneDrive`` or ``OneDrive - <Org>``.
    """
    low = name.lower()
    return low in _SKIP_DIR_NAMES or _is_cloud_dir(name)


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
    """Per-entry skip policy for the discovery walk.

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
    admits it so ``claim()`` can register it as an *unresolved* source, and enters
    OneDrive directories, which a plain walk prunes. The hidden-entry and other
    system-directory prunes still apply; only the cloud skips are lifted, and only
    under an explicitly configured root.
    """
    name = path.name
    if name.startswith("."):
        return True
    if is_dir:
        if admit_nonresident and _is_cloud_dir(name):
            return False
        return _is_skippable_system_dir(name)
    if admit_nonresident:
        return False
    return _is_offline_placeholder(path, stat_result)


# Bound on directory_is_resident's sample -- large enough that a resolved
# cloud source whose data files are all still placeholders is caught by the
# first file checked, small enough that this stays a cheap, recall-free probe
# rather than a full walk.
_RESIDENCY_SAMPLE_LIMIT = 32


def directory_is_resident(root: Path, max_files: int = _RESIDENCY_SAMPLE_LIMIT) -> bool:
    """Best-effort, recall-free: does *root* look locally resident?

    A directory's own stat bits are not a usable signal here -- it can
    legitimately report ``st_blocks == 0`` on some local filesystems (e.g.
    macOS APFS) even when fully resident, which is why ``is_resident()``
    does not apply the file-level check to the directory path itself. This
    instead samples a bounded number of the *files* inside it and applies
    that same check to each, short-circuiting on the first placeholder found.
    A cloud source that has been resolved but not read has every data
    file still dehydrated, so a small sample reliably catches that case; this
    is not a full-tree scan and gives no guarantee for a directory that is
    only partially rehydrated.

    Never raises and never opens file content -- an unreadable directory, or
    one exhausted by the sample cap without finding a placeholder, reads as
    resident, on the same best-effort terms as the rest of this module.
    """
    checked = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Sorted so the sample is deterministic (which files a cap of
            # `max_files` reaches should not depend on directory-entry order).
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not should_skip_walk_entry(Path(dirpath) / d, is_dir=True)
            )
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                if _is_offline_placeholder(Path(dirpath) / name):
                    return False
                checked += 1
                if checked >= max_files:
                    return True
    except OSError:
        return True
    return True


def source_is_resident(source_url: str) -> bool:
    """Best-effort, recall-free: is the source at *source_url* local and cheap to
    read right now?

    A remote url never is. The offline-placeholder signal (``st_blocks == 0``) is
    a per-*file* concept -- :func:`should_skip_walk_entry` only consults it for
    files -- and a directory-based source (zarr, ome-zarr store) legitimately
    reports it on some filesystems (macOS APFS), so a directory is judged by
    :func:`directory_is_resident`, which samples the files inside it.
    """
    if is_remote_url(source_url):
        return False
    path = Path(source_url)
    if path.is_dir():
        return directory_is_resident(path)
    return not _is_offline_placeholder(path)


class ClaimContext(abc.ABC):
    """Unified path access for the claim protocol.

    ``claim()`` implementations probe the filesystem through this seam so they
    work identically over a local ``Path`` or a remote ``RemoteStore``. The
    variance is expressed as a **type**, not a set of mode flags: calling
    ``ClaimContext(...)`` dispatches to one of two concrete shapes (the
    ``pathlib.Path`` idiom -- constructing the base returns a subclass), so each
    operation is a single implementation instead of a remote-vs-local ladder:

    - :class:`RemoteContext` -- every probe hits a ``RemoteStore``.
    - :class:`LiveLocalContext` -- a local ``Path``, probed live each call.

    Sub-contexts from :meth:`join` / :meth:`parent` are of the same shape.
    """

    # Factory dispatch. ``ClaimContext(...)`` picks the concrete shape from its
    # arguments and returns an uninitialized instance of it; Python then calls
    # that subclass's ``__init__`` with the same arguments (so each subclass
    # ``__init__`` accepts the arguments its own call sites pass). Constructing a
    # subclass directly (``LiveLocalContext(path)``) skips the dispatch.
    def __new__(
        cls,
        path: Path | str = "",
        store: Optional[RemoteStore] = None,
        cloud_root: bool = False,
        monitored: bool = False,
    ) -> ClaimContext:
        if cls is not ClaimContext:
            return object.__new__(cls)
        return object.__new__(RemoteContext if store is not None else LiveLocalContext)

    # --- shared flag properties (overridden only by the shapes that differ) ---

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote context."""
        return False

    @property
    def store(self) -> Optional[RemoteStore]:
        """Underlying RemoteStore if remote, else None."""
        return None

    @property
    def cloud_root(self) -> bool:
        """Whether this path is under a configured ``cloud = true`` root."""
        return False

    @property
    def monitored(self) -> bool:
        """Whether this path is under a monitored root, which is walked again
        every rescan. A claim memoizes its content probe only then: a one-shot
        scan never revisits a file."""
        return False

    # --- path operations (each concrete shape implements these) ---
    #
    # Abstract, so the base cannot be instantiated and a concrete shape that
    # forgets an override is rejected at construction (and flagged by type
    # checkers) rather than at first call.

    @abc.abstractmethod
    def is_dir(self) -> bool:
        """Check if path is directory."""

    @abc.abstractmethod
    def is_file(self) -> bool:
        """Check if path is file."""

    @abc.abstractmethod
    def exists(self) -> bool:
        """Check if path exists."""

    @abc.abstractmethod
    def read_text(self, subpath: str = "") -> str:
        """Read file contents as text (``subpath`` empty for the current path)."""

    @abc.abstractmethod
    def open(self, mode: str = "rb") -> IO:
        """Open this path as a stream (``rb`` by default).

        The shape-agnostic read seam: a local shape opens the ``Path``, a remote
        shape opens through its ``RemoteStore``, so an adapter that needs a
        file-like handle (``pydicom.dcmread`` on a header) stays blind to which
        concrete context it got. The returned object is a context manager.
        """

    @abc.abstractmethod
    def join(self, subpath: str) -> ClaimContext:
        """Create a (live) context for ``subpath`` under this one."""

    @abc.abstractmethod
    def glob(self, pattern: str) -> List[ClaimContext]:
        """Find entries matching ``pattern`` in this directory (maxdepth 1)."""

    @property
    @abc.abstractmethod
    def path_str(self) -> str:
        """Get path as string (for SourceClaim)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get filename/directory name."""

    @property
    @abc.abstractmethod
    def parent(self) -> ClaimContext:
        """Get parent directory context."""

    @abc.abstractmethod
    def is_resident(self) -> bool:
        """Recall-free: is this path's content local and cheap to read right now?

        The per-read residency gate an adapter's ``claim()`` consults before
        opening a sidecar or container: when it returns False the read would
        trigger a whole-file cloud recall (or block offline), so the adapter
        defers and emits an *unresolved* claim instead (cloud-storage phase 2).
        """


class RemoteContext(ClaimContext):
    """Claim context backed by a ``RemoteStore``: every probe is a store call.

    The local-only caches/flags do not apply here -- remote reads go through cheap
    range requests (no residency or child-listing optimization), and a remote path
    is never a "cloud root" in the placeholder sense, so every probe hits the
    store. ``cloud_root`` therefore keeps the base default.
    """

    def __init__(self, path: Path | str, store: RemoteStore):
        self._remote_path = str(path)
        self._store = store

    @property
    def is_remote(self) -> bool:
        return True

    @property
    def store(self) -> RemoteStore:
        return self._store

    def is_dir(self) -> bool:
        return self._store.isdir(self._remote_path)

    def is_file(self) -> bool:
        return self._store.isfile(self._remote_path)

    def exists(self) -> bool:
        return self._store.exists(self._remote_path)

    def read_text(self, subpath: str = "") -> str:
        target = (
            (self._remote_path + "/" + subpath).lstrip("/")
            if subpath
            else self._remote_path
        )
        return self._store.read_text(target)

    def open(self, mode: str = "rb") -> IO:
        return self._store.open(self._remote_path, mode=mode)

    def join(self, subpath: str) -> ClaimContext:
        new_path = (
            self._remote_path.rstrip("/") + "/" + subpath
            if self._remote_path
            else subpath
        )
        return RemoteContext(new_path, self._store)

    def glob(self, pattern: str) -> List[ClaimContext]:
        matches = self._store.find(pattern, maxdepth=1)
        return [RemoteContext(m, self._store) for m in matches]

    @property
    def path_str(self) -> str:
        return self._store._join(self._remote_path)

    @property
    def name(self) -> str:
        return self._remote_path.rstrip("/").split("/")[-1]

    @property
    def parent(self) -> ClaimContext:
        parent_path = (
            self._remote_path.rsplit("/", 1)[0] if "/" in self._remote_path else ""
        )
        return RemoteContext(parent_path, self._store)

    def is_resident(self) -> bool:
        # Remote contexts read via cheap range requests, so always resident --
        # remote claim behavior is unchanged.
        return True


class LiveLocalContext(ClaimContext):
    """A local ``Path``, probed live: ``is_dir``/``is_file``/``exists``/``glob``
    read the filesystem each call.

    ``join`` / ``parent`` return the same shape.
    """

    def __init__(
        self, path: Path | str, cloud_root: bool = False, monitored: bool = False
    ):
        self._path = Path(path)
        self._monitored = monitored
        # True when this entry lives under a ``cloud = true`` root. Lets an
        # adapter's ``claim()`` (and the resolve-time re-claim) suppress
        # content-membership multi-file grouping under cloud regardless of
        # per-file residency -- residency can't gate the resolve path, where the
        # file is already resident.
        self._cloud_root = cloud_root

    @property
    def cloud_root(self) -> bool:
        return self._cloud_root

    @property
    def monitored(self) -> bool:
        return self._monitored

    def read_text(self, subpath: str = "") -> str:
        target = self._path / subpath if subpath else self._path
        return target.read_text()

    def open(self, mode: str = "rb") -> IO:
        return self._path.open(mode)

    @property
    def path_str(self) -> str:
        return str(self._path)

    @property
    def name(self) -> str:
        return self._path.name

    def join(self, subpath: str) -> ClaimContext:
        return LiveLocalContext(self._path / subpath)

    @property
    def parent(self) -> ClaimContext:
        return LiveLocalContext(self._path.parent)

    def is_resident(self) -> bool:
        # The placeholder signal (st_blocks == 0) is a per-file concept; a
        # directory legitimately reports zero blocks on some filesystems (macOS
        # APFS), so treat a directory as resident -- mirrors SourceAdapter
        # .is_resident and should_skip_walk_entry (which gates on `not is_dir`).
        # A local file is resident unless it is an offline cloud placeholder
        # (``_is_offline_placeholder``, a stat-only check that never opens content).
        try:
            if self._path.is_dir():
                return True
        except OSError:
            return False
        return not _is_offline_placeholder(self._path)

    def is_dir(self) -> bool:
        return self._path.is_dir()

    def is_file(self) -> bool:
        return self._path.is_file()

    def exists(self) -> bool:
        return self._path.exists()

    def glob(self, pattern: str) -> List[ClaimContext]:
        return [LiveLocalContext(p) for p in self._path.glob(pattern)]


@dataclass
class WalkReport:
    """What a walk declined to look at, for a caller that must tell "not found"
    from "not looked at".

    ``declined_dirs`` are the directories not entered -- by the skip policy or by
    the caller's ``path_filter``; directories only, because a skipped file is a
    leaf and recording every placeholder would make the set O(files).
    ``offline_files`` counts the non-resident placeholder *files* the skip policy
    passed over (it is zero under a cloud root, which admits them), so a caller can
    say "N offline files were skipped" instead of reporting an empty folder.
    ``cloud_dirs`` counts the OneDrive directories pruned by name (never under a
    cloud root), whose files are neither entered nor counted in ``offline_files``.
    """

    declined_dirs: Set[str] = field(default_factory=set)
    offline_files: int = 0
    cloud_dirs: int = 0


# Directory levels a walk descends below its root before it stops. No real
# acquisition tree is this deep; a tree that is has a loop the other guards missed
# (a shortcut a cloud provider exposes as an ordinary directory, a filesystem
# whose inode numbers are synthetic), and every further level is a listing.
MAX_WALK_DEPTH = 64


def _real_dir(path: Path) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return str(path)


def _leads_back_up(real: str, current: str) -> bool:
    """True when the directory ``real`` is ``current`` or one of its ancestors."""
    return current == real or current.startswith(real.rstrip(os.sep) + os.sep)


def _note_skipped_entry(report: Any, path: Path, is_dir: bool) -> None:
    """Record, in *report* (a ``WalkReport`` or anything with its two fields), an
    entry the skip policy passed over. A no-op without a report."""
    if report is None:
        return
    if is_dir:
        report.declined_dirs.add(str(path))
        if _is_cloud_dir(path.name):
            report.cloud_dirs += 1
    elif not path.name.startswith(".") and _is_offline_placeholder(path):
        report.offline_files += 1


def _descent_refused(
    path: Path, child_real: str, current_real: str, depth: int, max_depth: int
) -> bool:
    """Whether the walk must not enter directory *path*: it leads back to its own
    ancestry, or it is already ``max_depth`` levels below the root. Logs why."""
    if _leads_back_up(child_real, current_real):
        logger.warning("walk: not entering %s: it leads back to %s", path, current_real)
        return True
    if depth >= max_depth:
        logger.warning(
            "walk: not entering %s: more than %d levels below the root",
            path,
            max_depth,
        )
        return True
    return False


def walk_with_identity_tracking(
    root: Path,
    visited_identities: Set[str],
    path_filter: Optional[Callable[[Path], bool]] = None,
    should_descend: Optional[Callable[[Path], bool]] = None,
    admit_nonresident: bool = False,
    report: Optional[WalkReport] = None,
    max_depth: int = MAX_WALK_DEPTH,
    _depth: int = 0,
    _root_real: Optional[str] = None,
) -> Iterator[Path]:
    """Walk filesystem with cross-platform identity tracking.

    Three guards stop a walk from running away on a loop or duplicating work: a
    directory that is a symlink is never entered; an entry whose identity
    (device and inode, else resolved path) was already visited is skipped, which
    also drops hardlink duplicates; and a directory that resolves to itself or an
    ancestor (a junction or mount that ``is_symlink`` does not report) is not
    entered. None of the first two helps where inode numbers are synthetic or the
    loop is an ordinary directory, so the walk also stops ``max_depth`` levels
    below the root. A directory refused for either reason is recorded as declined,
    so what is registered below it is not taken for gone.

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
        report: Optional :class:`WalkReport` filled in as the walk goes.
        max_depth: Directory levels to descend below ``root``.

    Yields:
        Paths to files/directories (not yet claimed)
    """
    current_real = _root_real if _root_real is not None else _real_dir(root)
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
                _note_skipped_entry(report, path, is_dir)
                continue

            if path_filter is not None and not path_filter(path):
                if is_dir and report is not None:
                    report.declined_dirs.add(str(path))
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
                child_real = _real_dir(path)
                if not _descent_refused(
                    path, child_real, current_real, _depth, max_depth
                ):
                    yield from walk_with_identity_tracking(
                        path,
                        visited_identities,
                        path_filter=path_filter,
                        should_descend=should_descend,
                        admit_nonresident=admit_nonresident,
                        report=report,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _root_real=child_real,
                    )
                    continue
                if report is not None:
                    report.declined_dirs.add(str(path))
    except OSError:
        # Permission issue reading directory
        pass


class SourceClaim:
    """Represents a claimed data source.

    A claim describes what paths an adapter recognizes and wants to handle.
    Claims can be single-node (one file/dir) or multi-node (multiple files).

    Uses __slots__ for memory efficiency when scanning large directories.

    Attributes:
        source_type: Type identifier ("zarr", "ome-tiff", etc.)
        primary_path: Main entry point for the source (str to support URLs)
        source_id: Unique identifier (auto-generated if None)
        extra_config: Adapter-specific configuration (e.g., credentials_profile, alias)
        is_remote: Flag indicating if this is a remote source
        unresolved: True when the adapter recognized this source by recall-free
            signals only (a non-resident cloud/synced-folder target) and deferred
            its content read. Such a claim carries no shape/dtype yet; the server
            catalogs it as ``needs_recall`` with no adapter and registers it when
            a client resolves it (cloud-storage phase 2).
    """

    __slots__ = (
        "source_type",
        "primary_path",
        "source_id",
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
                lazy-resolve / cloud phase-2 flow (``Reconciler``) depends on
                that. ``None`` registers a claim-only adapter: it
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
        _path_to_source: Reverse mapping (primary_path → source_id)
        consumed_paths: All paths consumed by any source (Set[str] for URLs)
        visited_identities: File identities already visited
        on_source_added: Callback for source addition events
        on_source_removed: Callback for source removal events
    """

    claims: Dict[str, SourceClaim]
    _path_to_source: Dict[str, str]  # Changed from Dict[Path, str]
    _source_to_paths: Dict[str, Set[str]]
    consumed_paths: Set[str]  # Changed from Set[Path]
    visited_identities: Set[str]
    on_source_added: Optional[Callable[[SourceClaim], None]]
    on_source_removed: Optional[Callable[[str], None]]

    def __init__(
        self,
        on_source_added: Optional[Callable[[SourceClaim], None]] = None,
        on_source_removed: Optional[Callable[[str], None]] = None,
        source_type: Optional[str] = None,
    ):
        # A configured path can name its type; every claim found under it then
        # has that type, and so the id that type hashes into.
        self.source_type = source_type or None
        self.claims = {}
        self._path_to_source = {}
        self._source_to_paths = {}
        self.consumed_paths = set()
        self.visited_identities = set()
        self.on_source_added = on_source_added
        self.on_source_removed = on_source_removed
        # Set by AdapterRegistry.get_claims_for_path around a single adapter's
        # claim() call: try_claim_path appends each path it consumes so the
        # registry can attribute members without snapshotting consumed_paths.
        # Per thread, so a parallel walk's probes each record their own.
        self._recorders = threading.local()

    @property
    def _claim_recorder(self) -> Optional[List[str]]:
        return getattr(self._recorders, "value", None)

    @_claim_recorder.setter
    def _claim_recorder(self, value: Optional[List[str]]) -> None:
        self._recorders.value = value

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
        if self.source_type:
            claim.source_type = self.source_type
        # Generate source_id if not provided
        source_id = claim.source_id or generate_source_id(
            str(claim.primary_path), claim.source_type
        )

        # primary_path is already in claim.member_paths (SourceClaim.__init__).
        member_paths = set(claim.member_paths)

        existing_owner = None
        for path in member_paths:
            owner = self._path_to_source.get(path)
            if owner is not None and owner != source_id:
                existing_owner = owner
                break

        if existing_owner is not None:
            return False

        # Update claim's source_id (important for callbacks)
        claim.source_id = source_id
        self._store_claim(claim, member_paths)

        # Callback
        if notify and self.on_source_added:
            self.on_source_added(claim)

        return True

    def _store_claim(
        self,
        claim: SourceClaim,
        member_paths: Set[str],
        skip: Set[str] = frozenset(),
    ) -> None:
        """Write *claim* into the four indices, superseding any entry for its id.

        The conflict policy is the caller's: *skip* names member paths to leave
        attributed to their current owner. Shared so the ownership maps cannot
        end up maintained by one storing path and not the other.
        """
        claim.member_paths = member_paths
        self.claims[claim.source_id] = claim
        self._source_to_paths[claim.source_id] = member_paths
        for path in member_paths - skip:
            self._path_to_source[path] = claim.source_id
            self.consumed_paths.add(path)

    def replace_claim(self, claim: SourceClaim) -> Set[str]:
        """Force a claim into state even where its membership overlaps another.

        Used by a rebuild whose adapter is already live under this source_id
        (the caller already swapped it in) but whose rediscovered membership
        conflicts with another source's claim -- add_claim's reject-on-conflict
        semantics would otherwise leave this source with no claims entry at
        all, out of sync with what is actually being served.

        Paths already owned by a *different* source_id are left with that
        owner rather than stolen; those are returned so the caller can log
        them.

        Args:
            claim: SourceClaim to store, superseding any existing entry for
                its source_id.

        Returns:
            The subset of claim.member_paths still owned by another source_id.
        """
        source_id = claim.source_id
        member_paths = set(claim.member_paths)

        conflicting = {
            path
            for path in member_paths
            if self._path_to_source.get(path) not in (None, source_id)
        }

        self._store_claim(claim, member_paths, skip=conflicting)
        return conflicting

    def remove_claim(self, path: str, notify: bool = True) -> Optional[str]:
        """Remove claim by path (for file deletion events).

        Args:
            path: Primary path of the claim to remove (str to support URLs)
            notify: Whether to invoke on_source_removed after removing the claim

        Returns:
            source_id if removed, None if not found
        """
        source_id = self._path_to_source.get(path)
        if source_id is None:
            return None

        claim = self.claims.pop(source_id)
        member_paths = self._source_to_paths.pop(source_id, set(claim.member_paths))
        for member_path in member_paths:
            self._path_to_source.pop(member_path, None)
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
        return self._path_to_source.get(path)

    def get_all_claims(self) -> List[SourceClaim]:
        """Get all claims as a list."""
        return list(self.claims.values())

    def get_paths_for_source(self, source_id: str) -> Set[str]:
        """Get all claimed member paths for a source."""
        return set(self._source_to_paths.get(source_id, set()))


# ``file://`` is a LOCAL url (see ``is_remote_url``): every adapter that meets one
# strips this prefix and hands the rest to a filesystem reader, so the functions
# below -- which decide the same url's identity -- must strip it the same way. A
# naive prefix strip is deliberate: it is what the adapters do, and a cleverer
# one (url2pathname, percent-decoding) would make a source's id name a different
# file from the one its adapter opens.
_FILE_URL_PREFIX = "file://"


def _as_filesystem_path(path: str) -> str:
    """The filesystem path a local path-or-``file://``-url names."""
    if path.startswith(_FILE_URL_PREFIX):
        return path[len(_FILE_URL_PREFIX) :]
    return path


def resolve_local_path(path: str) -> str:
    """Canonical absolute form of a LOCAL filesystem path or ``file://`` url.

    The single canonicalizer for local-path identity across the server: the
    ``source_id`` hash (``generate_source_id``), ``SourceConfig.local_path``, and
    -- in ``source_manager`` -- the drag-drop containment guard and the
    static-config seed all reduce a path to this form, so the same physical
    location compares equal however it was spelled (``file://`` / symlink /
    junction / mapped drive / 8.3 / case / trailing sep). The monitored walk
    reaches the same form via ``Path.resolve`` on its root. ``Path.resolve``
    resolves reparse points on Python 3.8+, so it folds those on Windows too.

    Stripping ``file://`` here is what makes that url form share one identity
    with the plain path it names. Without it ``Path.resolve`` treated the whole
    url as a relative path, producing ``$PWD/file:/data/x`` -- a location that
    does not exist and an id that moved with the launch directory
    (biopb/biopb#947).

    Local paths only: a remote URL must NOT be passed here -- ``Path.resolve``
    mangles the scheme (collapsing ``//`` and prepending the cwd); callers gate
    on ``is_remote_url`` first.
    """
    return str(Path(_as_filesystem_path(path)).resolve())


def local_path_is_rooted(path: str) -> bool:
    """True if a LOCAL path or ``file://`` url starts from a filesystem root.

    The companion guard to :func:`resolve_local_path`: that function completes a
    rootless path from the *process* cwd, which is never what a config file or a
    wire request meant, so every surface that accepts a path from a user checks
    this first (biopb/biopb#947). One spelling, in one place, because the two
    obvious spellings disagree.

    Deliberately not ``os.path.isabs``: on Windows that returns True for a
    driveless ``/data/x``, which CPython's own source marks "LEGACY BUG" in
    ``ntpath.isabs`` and reserves the right to fix. Two callers spelling this
    check differently would then diverge on a *Python upgrade* rather than on an
    edit -- drift with no diff to notice. ``Path.root`` is stable: rooted
    ``/data/x`` passes on both platforms (it is how this repo's configs and
    fixtures spell a source), while drive-relative ``C:x`` and plain ``data/x``
    do not.

    Judges a ``file://`` url on the path it carries, so the check agrees with
    what :func:`resolve_local_path` will make of it.
    """
    return bool(Path(_as_filesystem_path(path)).root)


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
) -> Optional[SourceClaim]:
    """Finalize the winning claim from ``get_claims_for_path`` into ``state``.

    Registers the claim. Shared by every discovery entry point so the
    claim-finalization policy lives in one place. Returns the recorded claim, or
    ``None`` when no adapter claimed the path.
    """
    if not claims:
        return None
    claim = claims[0]
    state.add_claim(claim)
    return claim


@dataclass
class _DirVisit:
    """What one worker found in one directory, for the scheduler to apply."""

    claims: List[SourceClaim] = field(default_factory=list)
    subdirs: List[tuple] = field(default_factory=list)  # (path, depth, real)
    declined_dirs: Set[str] = field(default_factory=set)
    offline_files: int = 0
    cloud_dirs: int = 0


def _visit_directory(
    directory: Path,
    depth: int,
    current_real: str,
    registry: AdapterRegistry,
    state: DiscoveryState,
    path_filter: Optional[Callable[[Path], bool]],
    admit_nonresident: bool,
    cloud_root: bool,
    monitored: bool,
    max_depth: int,
) -> _DirVisit:
    """List one directory and probe its entries, as the serial walk does.

    One worker owns a directory from its listing to the last probe of its
    entries, so a claim that consumes sibling files (a multi-file source) sees the
    same entries in the same order as the serial walk. Nothing is applied to the
    shared claim table here: the claims come back to the scheduler.
    """
    visit = _DirVisit()
    try:
        entries = list(directory.iterdir())
    except OSError:
        return visit  # Permission issue reading directory

    for path in entries:
        try:
            stat_result = os.stat(path)
        except OSError:
            continue  # Broken entry, broken symlink or permission issue
        is_dir = stat_module.S_ISDIR(stat_result.st_mode)

        if should_skip_walk_entry(
            path, is_dir, stat_result=stat_result, admit_nonresident=admit_nonresident
        ):
            _note_skipped_entry(visit, path, is_dir)
            continue

        if path_filter is not None and not path_filter(path):
            if is_dir:
                visit.declined_dirs.add(str(path))
            continue

        path_str = str(path)
        if not state.is_path_claimed(path_str):
            ctx = ClaimContext(path, cloud_root=cloud_root, monitored=monitored)
            claims = registry.get_claims_for_path(ctx, state)
            if claims:
                visit.claims.append(claims[0])

        if is_dir and not path.is_symlink() and not state.is_path_claimed(path_str):
            child_real = _real_dir(path)
            if not _descent_refused(path, child_real, current_real, depth, max_depth):
                visit.subdirs.append((path, depth + 1, child_real))
                continue
            visit.declined_dirs.add(path_str)
    return visit


def _discover_parallel(
    root: Path,
    registry: AdapterRegistry,
    state: DiscoveryState,
    path_filter: Optional[Callable[[Path], bool]],
    admit_nonresident: bool,
    cloud_root: bool,
    report: Optional[WalkReport],
    monitored: bool,
    threads: int,
    max_depth: int = MAX_WALK_DEPTH,
) -> int:
    """Walk *root* with *threads* workers; the calling thread is the scheduler.

    Workers only read the filesystem and probe; the scheduler alone applies what
    they return (``add_claim``, so the streamed first-scan commit also runs on
    one thread) and queues the subdirectories they report. Work is handed out a
    directory at a time from one queue, so a worker that finishes early takes the
    next waiting directory. A directory that claim-descent prunes is never queued.

    No identity set, unlike the serial walk: two spellings of one location get one
    source id, and a hardlinked or bind-mounted copy is a second source.

    Returns the number of directories visited.
    """
    results: queue.SimpleQueue = queue.SimpleQueue()
    cancelled = threading.Event()

    def task(directory: Path, depth: int, real: str) -> None:
        try:
            if cancelled.is_set():
                results.put((None, None))
                return
            results.put(
                (
                    _visit_directory(
                        directory,
                        depth,
                        real,
                        registry,
                        state,
                        path_filter,
                        admit_nonresident,
                        cloud_root,
                        monitored,
                        max_depth,
                    ),
                    None,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised by the scheduler
            results.put((None, exc))

    visited = 0
    with ThreadPoolExecutor(
        max_workers=threads, thread_name_prefix="discovery-walk"
    ) as pool:
        pool.submit(task, root, 0, _real_dir(root))
        outstanding = 1
        failure: Optional[BaseException] = None
        while outstanding:
            visit, exc = results.get()
            outstanding -= 1
            if exc is not None:
                failure = failure or exc
                cancelled.set()
                continue
            if visit is None or failure is not None:
                continue
            visited += 1
            # Subdirectories first: the commit a claim triggers (stats and a catalog
            # write, on this thread) must not keep idle workers waiting for work.
            for sub in visit.subdirs:
                pool.submit(task, *sub)
                outstanding += 1
            for claim in visit.claims:
                state.add_claim(claim)
            if report is not None:
                report.declined_dirs |= visit.declined_dirs
                report.offline_files += visit.offline_files
                report.cloud_dirs += visit.cloud_dirs
    if failure is not None:
        raise failure
    return visited


def discover_sources(
    root: Path,
    registry: AdapterRegistry,
    state: Optional[DiscoveryState] = None,
    path_filter: Optional[Callable[[Path], bool]] = None,
    admit_nonresident: bool = False,
    cloud_root: bool = False,
    report: Optional[WalkReport] = None,
    monitored: bool = False,
    walk_threads: int = 1,
) -> DiscoveryState:
    """Recursive filesystem discovery with claim protocol.

    The one walker: a drop, a one-shot directory and the periodic rescan of a
    monitored root all come through here. Walks the filesystem recursively,
    asking each registered adapter to claim paths it recognizes, and stops
    descending at a claimed directory. It keeps nothing between calls, so every
    call stats the whole tree under ``root`` that it does not prune.

    Args:
        root: Root directory to scan. Honored unconditionally -- ``path_filter``
            and the skip policy apply only to what is found inside it.
        registry: Adapter registry for claims
        state: Existing DiscoveryState to update (creates new if None)
        path_filter: Entry gate (the rescan's stability window); a directory it
            rejects is not entered.
        report: Filled with what the walk declined (:class:`WalkReport`).
        admit_nonresident: Under a cloud root, admit dehydrated placeholders
            instead of skipping them.
        cloud_root: Under a cloud root, set ``ClaimContext.cloud_root`` so the
            content-membership adapters (multi-file OME-TIFF / DICOM series) fall
            back to single-file sources instead of grouping -- the same ban the
            monitored rescan applies. Keeps the static one-shot scan of a
            ``monitor=false`` cloud directory consistent with the monitored path.
        monitored: The walk is a monitored root's rescan, which visits the same
            files again every tick, so claims memoize their content probes
            (``ClaimContext.monitored``). A one-shot walk does not.
        walk_threads: Above 1, directories are read and probed by that many
            worker threads under one scheduler (:func:`_discover_parallel`), which
            is what a high-latency filesystem (NFS) needs; 1 is the serial walk.

    Returns:
        DiscoveryState with all discovered sources
    """
    if state is None:
        state = DiscoveryState()

    logger.debug(f"discover_sources: scanning {root}")

    # Get identity for root itself
    try:
        root_identity = get_file_identity(root)
        state.visited_identities.add(root_identity)
    except OSError:
        logger.debug(f"discover_sources: cannot get identity for {root}")
        return state

    # Check if root itself is a data source (e.g., a .zarr directory)
    ctx = ClaimContext(root, cloud_root=cloud_root, monitored=monitored)
    claim = _record_claim(state, registry.get_claims_for_path(ctx, state))
    if claim is not None:
        logger.info(f"discover_sources: root {root} claimed as {claim.source_type}")
        return state  # Root claimed, no need to recurse

    if walk_threads > 1:
        dirs = _discover_parallel(
            root,
            registry,
            state,
            path_filter,
            admit_nonresident,
            cloud_root,
            report,
            monitored,
            walk_threads,
        )
        logger.debug(
            f"discover_sources: {dirs} directories, found {len(state.claims)} sources"
        )
        return state

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
        report=report,
    ):
        paths_scanned += 1
        path_str = str(path)
        if state.is_path_claimed(path_str):
            continue

        ctx = ClaimContext(path, cloud_root=cloud_root, monitored=monitored)
        _record_claim(state, registry.get_claims_for_path(ctx, state))

    logger.debug(
        f"discover_sources: scanned {paths_scanned} paths, found {len(state.claims)} sources"
    )
    return state


def discover_remote_source(
    url: str,
    registry: AdapterRegistry,
    credentials_config: Optional[Any] = None,
    profile_name: Optional[str] = None,
    state: Optional[DiscoveryState] = None,
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
    claim = _record_claim(state, registry.get_claims_for_path(ctx, state))
    if claim is not None:
        logger.info(f"discover_remote_source: {url} claimed as {claim.source_type}")

    return state
