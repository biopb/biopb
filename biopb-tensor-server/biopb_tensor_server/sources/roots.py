"""Every root the server knows about, and what is true of each.

A *root* is a place sources come from that is not a single configured source: a
watched directory, a directory walked once, a folder dropped at runtime, or a
bare-host tensor-server upstream. :class:`Roots` is the one record of them, shared
by the :class:`~biopb_tensor_server.sources.source_manager.SourceManager` and the
:class:`~biopb_tensor_server.sources.reconciler.Reconciler`, so "which root is
this path under, is it cloud, what is its display root" is answered in one place.

Reads are lock-free over an immutable snapshot; writes replace the snapshot under
a lock, so the rescan thread can ask while a drop adds or removes a root.

How the configured ``[[sources]]`` become roots is :mod:`~biopb_tensor_server.sources.resolve`.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple

from biopb_tensor_server.core.adapter_base import to_catalog_url
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.remote import is_remote_url

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


def reroot_catalog_url(label: str, root_path: str, primary_path: str) -> str:
    """Re-root ``primary_path`` under ``label``, preserving its position beneath
    ``root_path``. Shared core of the two re-rooting entry points -- drag-drop
    (``roots._drop_catalog_url``, ``label`` = the dropped item's basename) and a
    configured ``alias`` (``label`` = the alias).

    The tensor-browser (and web viewer) build their tree by splitting each
    source's ``source_url`` on ``/``, so ``label`` becomes the top-level root and
    the sub-structure beneath ``root_path`` is preserved under it:

        root /data/exp, primary /data/exp            -> "<label>"
        root /data/exp, primary /data/exp/sub/b.tif  -> "<label>/sub/b.tif"

    Display-only: it feeds the descriptor's ``source_url`` and never the
    ``source_id`` (which hashes the raw path), so a bare virtual path with no
    scheme is fine.
    """
    rel = path_under_root(root_path, primary_path)
    return label if rel == "." else f"{label}/{rel}"


def path_under_root(root_path: str, primary_path: str) -> str:
    """``primary_path`` beneath ``root_path``, forward-slashed; ``.`` when it is the
    root itself or (defensively) not under it, so a url never carries ``../``."""
    try:
        rel = os.path.relpath(str(primary_path), str(root_path)).replace("\\", "/")
    except ValueError:  # different drive on Windows, etc. -- can't relativize
        return "."
    return "." if rel in (".", "") or rel.startswith("../") else rel


def _drop_catalog_url(
    dropped_root: str, primary_path: str, *, label: Optional[str] = None
) -> str:
    """Catalog ``source_url`` that re-roots a drag-dropped source under its drop's
    label, under the ``dnd://`` origin scheme.

    The label is the dropped item's basename unless the caller picked another (a
    second drop of a same-named folder gets a distinct one, see
    :meth:`Roots.unique_label`). The shared ``reroot_catalog_url`` keeps the
    source's place beneath the drop::

        drop /home/u/data/exp.zarr           -> "dnd://exp.zarr"        (own root)
        drop /home/u/data/exp/ (a folder) with
             .../exp/a.tif, .../exp/sub/b.tif -> "dnd://exp/a.tif",
                                                 "dnd://exp/sub/b.tif"

    The marker means "user-added from outside every known root, so nothing will
    re-add it": it is what ``remove_source`` authorizes on. Never under a
    monitored or ``monitor = false`` root.

    Display-only; the client tree builders strip the scheme. The configured-alias
    re-root shares ``reroot_catalog_url`` but is always scheme-less, so the two
    stay distinguishable.
    """
    dropped_root = str(dropped_root).rstrip("/\\")
    if label is None:
        label = os.path.basename(dropped_root) or dropped_root
    return DND_URL_PREFIX + reroot_catalog_url(label, dropped_root, primary_path)


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
    # its credentials profile, a scan-once directory with its ``type``.
    source: Optional[SourceConfig] = None
    # Derived once: queried per claim on every walk.
    path: Optional[Path] = field(init=False, default=None, repr=False)
    depth: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        if self.kind is not RootKind.UPSTREAM:
            path = Path(self.url)
            object.__setattr__(self, "path", path)
            object.__setattr__(self, "depth", len(path.parts))

    @property
    def root_id(self) -> str:
        """Names the root in the catalog table: a hash of its resolved path."""
        return hashlib.sha256(self.url.encode()).hexdigest()[:12]

    @property
    def root_url(self) -> str:
        """What a source's catalog url starts with: the alias, else the file url of
        the root. A drop's ``dnd://`` label is its own and is never persisted."""
        return self.alias or to_catalog_url(self.url)

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


def _under_any(path: str, roots: Tuple[Path, ...]) -> bool:
    """True when the local *path* is at or under any of *roots* (lexical)."""
    if not roots or is_remote_url(path):
        return False
    claim_path = Path(path)
    return any(claim_path.is_relative_to(root) for root in roots)


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

    def is_cloud(self, path: str) -> bool:
        """True when *path* is a cloud root or lives under one.

        Lexical: *path* is a claim (or walk) path, spelled under the root it was
        found in, so it is never resolved -- a link under a cloud root is under it
        wherever it points.
        """
        return _under_any(path, self._snap.cloud)

    def is_monitored(self, path: str) -> bool:
        """True when *path* lives under a monitored directory (lexical, local only)."""
        return _under_any(path, self._snap.monitored)

    def display_url(self, claim_path: str) -> Optional[str]:
        """The display ``source_url`` for a source found at ``claim_path``, or None.

        What the root it is under makes of it: a drop's ``dnd://`` label (so the
        source goes with the drop), else the root's alias. None leaves the plain
        file url. The catalog's view builds the same from a row's root and path
        (``Root.root_url``). Display-only: the source id still hashes the native path.
        """
        root = self.containing(Path(claim_path))
        if root is None:
            return None
        if root.kind is RootKind.DROPPED:
            return _drop_catalog_url(root.url, claim_path, label=root.label)
        if root.alias is None:
            return None
        return reroot_catalog_url(root.alias, root.url, claim_path)

    def persisted(self) -> List[Root]:
        """The roots whose sources the catalog table holds."""
        return self.of_kind(RootKind.MONITORED, RootKind.SCAN_ONCE)

    def check_overlap(
        self,
        path: Path,
        registered_paths: Iterable[str],
        *,
        already_registered: bool = False,
    ) -> Optional[str]:
        """Why a root at ``path`` may not be added, or None.

        A root that is its own display root must not share sources with another:
        refused when it holds a source that is already registered
        (``already_registered``), when an existing source lies under it
        (``registered_paths``), or when it contains a root already known.
        """
        if (
            already_registered
            or any(Path(p).is_relative_to(path) for p in registered_paths)
            or any(
                r.path is not None and r.path.is_relative_to(path)
                for r in self._snap.roots
            )
        ):
            return OVERLAP_MESSAGE
        return None
