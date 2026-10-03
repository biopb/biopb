"""Every root the server knows about, and what is true of each.

A *root* is a place sources come from that is not a single configured source: a
watched directory, a directory walked once, a folder dropped at runtime, or a
bare-host tensor-server upstream. :class:`Roots` is the one record of them, shared
by the :class:`~biopb_tensor_server.sources.source_manager.SourceManager` and the
:class:`~biopb_tensor_server.sources.reconciler.Reconciler`, so "which root is
this path under, is it cloud, what is its display root" is answered in one place.

Reads are lock-free over an immutable snapshot; writes replace the snapshot under
a lock, so the rescan thread can ask while a drop adds or removes a root.

:func:`partition_sources` is where the configured ``[[sources]]`` become roots: the
one place that decides whether an entry is kept live by the manager's rescan loop
(monitored, or an upstream), registered once by its first tick (scan-once: a local
directory, file or typed dataset), or a single remote source registered as it is
(static), and the only one to warn about an entry that is not what it asked for.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple

from biopb_tensor_server.adapters.remote_tensor import is_bare_host_upstream_url
from biopb_tensor_server.core.config import ServerConfig, SourceConfig
from biopb_tensor_server.core.discovery import AdapterRegistry
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.serving.upload_manager import write_dir_under_root
from biopb_tensor_server.sources.resolve import (
    _alias_catalog_url,
    _reroot_catalog_url,
    resolve_all_sources,
)

logger = logging.getLogger(__name__)

# Display-only origin scheme on the ``source_url`` of a drag-dropped source. The
# drop path only stamps it when the drop is entirely new AND outside every known
# root, so its presence means "user-added and nothing will re-add it." It never
# touches ``source_id`` or the raw ``_source_url`` used for I/O. Keep in sync with
# the client tree builders that strip it: ``_get_path_parts`` (biopb-mcp
# ``tensor_browser/_widget.py``) and ``getPathParts`` (web ``SourceTree.tsx``).
DND_URL_PREFIX = "dnd://"

OVERLAP_MESSAGE = (
    "overlaps sources that are already registered; drop a narrower path, or "
    "remove them first"
)


def _drop_catalog_url(
    dropped_root: str, primary_path: str, *, label: Optional[str] = None
) -> str:
    """Catalog ``source_url`` that re-roots a drag-dropped source under its drop's
    label, under the ``dnd://`` origin scheme.

    The label is the dropped item's basename unless the caller picked another (a
    second drop of a same-named folder gets a distinct one, see
    :meth:`Roots.unique_label`). The shared ``_reroot_catalog_url`` keeps the
    source's place beneath the drop::

        drop /home/u/data/exp.zarr           -> "dnd://exp.zarr"        (own root)
        drop /home/u/data/exp/ (a folder) with
             .../exp/a.tif, .../exp/sub/b.tif -> "dnd://exp/a.tif",
                                                 "dnd://exp/sub/b.tif"

    The marker means "user-added from outside every known root, so nothing will
    re-add it": it is what ``remove_source`` authorizes on. Never under a
    monitored or ``monitor = false`` root.

    Display-only; the client tree builders strip the scheme. The configured-alias
    re-root shares ``_reroot_catalog_url`` but is always scheme-less, so the two
    stay distinguishable.
    """
    dropped_root = str(dropped_root).rstrip("/\\")
    if label is None:
        label = os.path.basename(dropped_root) or dropped_root
    return DND_URL_PREFIX + _reroot_catalog_url(label, dropped_root, primary_path)


class RootKind(Enum):
    MONITORED = "monitored"  # watched local directory
    # A local path (directory, file, typed dataset) the first tick registers, then
    # never rescans: a directory is walked, a file or typed dataset claimed in place.
    SCAN_ONCE = "scan_once"
    DROPPED = "dropped"  # added at runtime (drag-drop / register_local_path)
    UPSTREAM = "upstream"  # bare-host tensor server, re-listed rather than walked


_LOCAL_KINDS = (RootKind.MONITORED, RootKind.SCAN_ONCE, RootKind.DROPPED)


@dataclass(frozen=True, eq=False)
class Root:
    """One root. Identity is the object: ``Roots.remove`` takes the one it added."""

    kind: RootKind
    url: str  # the resolved local path, or the upstream's ``grpc[s]://`` url
    alias: Optional[str] = None  # display root for what is found under it
    cloud: bool = False  # offline placeholders admitted, resolved on first access
    label: Optional[str] = None  # DROPPED only: the ``dnd://`` label
    # The config entry a configured root came from: an upstream is re-listed with
    # its credentials profile, a scan-once directory with its HDF5 ``dataset``.
    source: Optional[SourceConfig] = None
    # Derived once: queried per claim on every walk.
    path: Optional[Path] = field(init=False, default=None, repr=False)
    depth: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        if self.kind is not RootKind.UPSTREAM:
            path = Path(self.url)
            object.__setattr__(self, "path", path)
            object.__setattr__(self, "depth", len(path.parts))

    @classmethod
    def from_config(cls, source: SourceConfig, kind: RootKind) -> Root:
        if kind is RootKind.UPSTREAM:
            return cls(kind, source.url, source.alias, source=source)
        return cls(
            kind,
            str(source.local_path),
            source.alias,
            bool(source.cloud),
            source=source,
        )


def _innermost(roots: Iterable[Root]) -> Optional[Root]:
    return max(roots, key=lambda r: r.depth, default=None)


class _Snapshot(NamedTuple):
    """What the queries read, rebuilt whole on every change."""

    roots: Tuple[Root, ...]
    local: Tuple[Root, ...]  # the roots a path can be under
    cloud: Tuple[Path, ...]
    monitored: Tuple[Path, ...]

    @classmethod
    def of(cls, roots: Tuple[Root, ...]) -> _Snapshot:
        return cls(
            roots,
            tuple(r for r in roots if r.kind in _LOCAL_KINDS),
            tuple(r.path for r in roots if r.cloud and r.path is not None),
            tuple(r.path for r in roots if r.kind is RootKind.MONITORED),
        )


class Roots:
    """The set of known roots. See the module docstring."""

    def __init__(self, roots: Iterable[Root] = ()):
        self._lock = threading.Lock()
        self._snap = _Snapshot.of(tuple(roots))
        self._unscanned: set = {
            r for r in self._snap.roots if r.kind is RootKind.SCAN_ONCE
        }

    def __iter__(self):
        return iter(self._snap.roots)

    def __len__(self) -> int:
        return len(self._snap.roots)

    # -- mutation ---------------------------------------------------------

    def add(self, root: Root) -> None:
        with self._lock:
            self._snap = _Snapshot.of((*self._snap.roots, root))
            if root.kind is RootKind.SCAN_ONCE:
                self._unscanned.add(root)

    def remove(self, root: Root) -> None:
        with self._lock:
            self._snap = _Snapshot.of(
                tuple(r for r in self._snap.roots if r is not root)
            )
            self._unscanned.discard(root)

    def take_unscanned(self) -> List[Root]:
        """The scan-once roots not yet handed out, marked as handed out."""
        with self._lock:
            taken = [r for r in self._snap.roots if r in self._unscanned]
            self._unscanned.clear()
        return taken

    # -- queries ----------------------------------------------------------

    def of_kind(self, *kinds: RootKind) -> List[Root]:
        return [r for r in self._snap.roots if r.kind in kinds]

    def by_label(self, label: str) -> Optional[Root]:
        return next((r for r in self._snap.roots if r.label == label), None)

    def unique_label(self, path: Path) -> str:
        """The ``dnd://`` label for a drop from outside every known root.

        The basename, with a counter when another drop already holds it, so two
        same-named folders from different parents are two roots, not one. No ``/``,
        so one label is never a prefix of another's urls.
        """
        base = path.name or str(path)
        label, n = base, 2
        while self.by_label(label) is not None:
            label, n = f"{base} ({n})", n + 1
        return label

    def containing(self, path: Path) -> Optional[Root]:
        """The innermost local root ``path`` is at or under, or None.

        Lexical, on the resolved path against roots resolved once.
        """
        return _innermost(r for r in self._snap.local if path.is_relative_to(r.path))

    def cloud_roots(self) -> frozenset:
        """The paths of the cloud roots, for a consumer that wants the raw set."""
        return frozenset(self._snap.cloud)

    def is_cloud(self, path: str) -> bool:
        """True when *path* is a cloud root or lives under one.

        Lexical: *path* is a claim (or walk) path, spelled under the root it was
        found in, so it is never resolved -- a link under a cloud root is under it
        wherever it points.
        """
        cloud = self._snap.cloud
        if not cloud or is_remote_url(path):
            return False
        claim_path = Path(path)
        return any(claim_path.is_relative_to(root) for root in cloud)

    def is_monitored(self, path: str) -> bool:
        """True when *path* lives under a monitored directory (lexical, local only)."""
        monitored = self._snap.monitored
        if not monitored or is_remote_url(path):
            return False
        claim_path = Path(path)
        return any(claim_path.is_relative_to(root) for root in monitored)

    def display_url(self, claim_path: str) -> Optional[str]:
        """The display ``source_url`` for a source found at ``claim_path``, or None.

        What the root it is under makes of it: a drop's ``dnd://`` label (so the
        source goes with the drop), the innermost aliased monitored root's alias,
        or a ``monitor = false`` root's alias. None leaves the plain file url.
        Display-only: the source id still hashes the native path.
        """
        path = Path(claim_path)
        inside = [r for r in self._snap.local if path.is_relative_to(r.path)]
        root = _innermost(inside)
        if root is None:
            return None
        if root.kind is RootKind.DROPPED:
            return _drop_catalog_url(root.url, claim_path, label=root.label)
        if root.kind is RootKind.SCAN_ONCE:
            # Persistent: nothing rescans the root to re-merge it into the tree.
            alias_root = root if root.alias else None
        else:
            alias_root = _innermost(
                r for r in inside if r.kind is RootKind.MONITORED and r.alias
            )
        if alias_root is None:
            return None
        return _alias_catalog_url(alias_root.alias, alias_root.url, claim_path)

    def check_overlap(
        self,
        path: Path,
        registered_paths: Iterable[str],
        *,
        already_registered: bool = False,
        exclude: Optional[Root] = None,
    ) -> Optional[str]:
        """Why a root at ``path`` may not be added, or None.

        A root that is its own display root must not share sources with another:
        refused when it holds a source that is already registered
        (``already_registered``), when an existing source lies under it
        (``registered_paths``), or when it contains a root already known
        (``exclude`` is the root being added, which is not its own overlap).
        """
        if (
            already_registered
            or any(Path(p).is_relative_to(path) for p in registered_paths)
            or any(
                r is not exclude and r.path is not None and r.path.is_relative_to(path)
                for r in self._snap.roots
            )
        ):
            return OVERLAP_MESSAGE
        return None


def route_source(s: SourceConfig) -> Optional[RootKind]:
    """Where a configured source goes on the serve path: the kind of root it
    becomes, or None for a single remote source, which is registered as it is.
    Logs why when that is not what the entry asked for.

    Every local path is discovered by the manager after the server is SERVING, never
    expanded here: that would walk the tree an extra time before the server binds,
    and crash on a not-yet-mounted directory (biopb/biopb#54).
    """
    if s.is_remote:
        # A bare-host tensor-server upstream ("mirror everything") holds many
        # sources of its own, so it always goes to the manager's background
        # re-list. Every other remote (s3://, ...) names a single source.
        return RootKind.UPSTREAM if is_bare_host_upstream_url(s.url) else None

    path = s.local_path
    if path is None:  # unreachable: a local url always resolves to a path
        return None

    if path.is_file():
        if s.monitor:
            logger.warning(
                "Cannot live-monitor a single file; registering it once instead: %s",
                s.url,
            )
        return RootKind.SCAN_ONCE

    if s.monitor:
        if not path.exists():
            logger.warning(
                "Monitored path does not exist yet; will start monitoring when it "
                "appears: %s",
                s.url,
            )
        return RootKind.MONITORED

    # Not watched, but still registered once. A path that is not there is left to
    # that pass, which warns and skips it.
    return RootKind.SCAN_ONCE


def partition_sources(
    sources: List[SourceConfig],
    registry: Optional[AdapterRegistry] = None,
    *,
    credentials_config=None,
    write_dir: Optional[Path] = None,
) -> Tuple[List[SourceConfig], Roots]:
    """Partition configured sources for the serve path: ``(static, roots)``.

    ``static`` is the single remote sources, expanded here. Everything the manager
    registers after SERVING (watched directories, scan-once paths, upstreams) is a
    root. See :func:`route_source`.
    """
    to_expand: List[SourceConfig] = []
    scan_once: List[SourceConfig] = []
    roots = Roots()

    for s in sources:
        kind = route_source(s)
        if kind is None:
            to_expand.append(s)
        elif kind is RootKind.SCAN_ONCE:
            scan_once.append(s)
        else:
            roots.add(Root.from_config(s, kind))

    # A file or typed dataset listed inside a monitored directory is the rescan's:
    # registering it again here would claim it twice.
    scan_once = [
        s
        for s in scan_once
        if not (
            (s.local_path.is_file() or s.type) and roots.is_monitored(str(s.local_path))
        )
    ]
    for s in scan_once:
        roots.add(Root.from_config(s, RootKind.SCAN_ONCE))

    # tolerant=True so one missing or broken static source is warned-and-skipped
    # rather than killing the server.
    static_sources = resolve_all_sources(
        ServerConfig(sources=to_expand, credentials=credentials_config),
        registry,
        tolerant=True,
    )

    # Upload stores are registered by the upload path, and the adapters decline
    # them if discovery reaches one, so a write_dir inside a scanned directory is
    # not catalogued twice -- but the walk still descends into every store and
    # stats its chunk files, and a store being written keeps its directory busy.
    scanned_dirs = {
        r.path
        for r in roots.of_kind(RootKind.MONITORED, RootKind.SCAN_ONCE)
        if r.kind is RootKind.MONITORED or r.path.is_dir()
    }
    inside = write_dir_under_root(write_dir, scanned_dirs)
    if inside is not None:
        logger.warning(
            "write_dir %s lies inside the source directory %s: its upload stores "
            "are walked whenever that directory is scanned (every rescan, if it "
            "is monitored). Keep write_dir outside every source directory.",
            write_dir,
            inside,
        )

    return static_sources, roots
