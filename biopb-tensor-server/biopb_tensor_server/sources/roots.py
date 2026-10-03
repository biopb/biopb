"""Every root the server knows about, and what is true of each.

A *root* is a place sources come from that is not a single configured source: a
watched directory, a directory walked once, a folder dropped at runtime, or a
bare-host tensor-server upstream. :class:`Roots` is the one record of them, shared
by the :class:`~biopb_tensor_server.sources.source_manager.SourceManager` and the
:class:`~biopb_tensor_server.sources.reconciler.Reconciler`, so "which root is
this path under, is it cloud, what is its display root" is answered in one place.

Reads are lock-free over an immutable snapshot; writes replace the snapshot under
a lock, so the rescan thread can ask while a drop adds or removes a root.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.remote import is_remote_url
from biopb_tensor_server.sources.resolve import (
    _alias_catalog_url,
    _reroot_catalog_url,
)

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
    SCAN_ONCE = (
        "scan_once"  # local path (directory, file, typed dataset) registered once
    )
    DROPPED = "dropped"  # added at runtime (drag-drop / register_local_path)
    UPSTREAM = "upstream"  # bare-host tensor server, re-listed rather than walked


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

    @property
    def path(self) -> Optional[Path]:
        """The local path; None for an upstream."""
        return None if self.kind is RootKind.UPSTREAM else Path(self.url)

    @classmethod
    def from_config(cls, source: SourceConfig, kind: RootKind) -> Root:
        local_path = source.local_path
        if kind is RootKind.UPSTREAM or local_path is None:
            return cls(RootKind.UPSTREAM, source.url, source.alias, source=source)
        return cls(
            kind, str(local_path), source.alias, bool(source.cloud), source=source
        )


class Roots:
    """The set of known roots. See the module docstring."""

    def __init__(self, roots: Iterable[Root] = ()):
        self._lock = threading.Lock()
        self._roots: Tuple[Root, ...] = ()
        self._unscanned: set = set()  # scan-once roots not yet handed out
        for root in roots:
            self.add(root)

    def __iter__(self):
        return iter(self._roots)

    def __len__(self) -> int:
        return len(self._roots)

    # -- mutation ---------------------------------------------------------

    def add(self, root: Root) -> None:
        with self._lock:
            self._roots = (*self._roots, root)
            if root.kind is RootKind.SCAN_ONCE:
                self._unscanned.add(root)

    def remove(self, root: Root) -> None:
        with self._lock:
            self._roots = tuple(r for r in self._roots if r is not root)
            self._unscanned.discard(root)

    def take_unscanned(self) -> List[Root]:
        """The scan-once roots not yet handed out, marked as handed out."""
        with self._lock:
            taken = [r for r in self._roots if r in self._unscanned]
            self._unscanned.clear()
        return taken

    # -- queries ----------------------------------------------------------

    def of_kind(self, *kinds: RootKind) -> List[Root]:
        return [r for r in self._roots if r.kind in kinds]

    def by_label(self, label: str) -> Optional[Root]:
        return next((r for r in self._roots if r.label == label), None)

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

    def _containing(self, path: Path, kinds: Tuple[RootKind, ...]) -> List[Root]:
        return [
            r
            for r in self._roots
            if r.kind in kinds
            and (rp := r.path) is not None
            and path.is_relative_to(rp)
        ]

    def containing(self, path: Path) -> Optional[Root]:
        """The innermost local root ``path`` is at or under, or None.

        Lexical, on the resolved path against roots resolved once.
        """
        inside = self._containing(
            path, (RootKind.MONITORED, RootKind.SCAN_ONCE, RootKind.DROPPED)
        )
        return max(inside, key=lambda r: len(r.path.parts), default=None)

    def cloud_roots(self) -> frozenset:
        """The paths of the cloud roots, for a consumer that wants the raw set."""
        return frozenset(r.path for r in self._roots if r.cloud and r.path)

    def is_cloud(self, path: str) -> bool:
        """True when *path* is a cloud root or lives under one.

        Lexical: *path* is a claim (or walk) path, spelled under the root it was
        found in, so it is never resolved -- a link under a cloud root is under it
        wherever it points.
        """
        cloud = self.cloud_roots()
        if not cloud or is_remote_url(path):
            return False
        claim_path = Path(path)
        return any(claim_path.is_relative_to(root) for root in cloud)

    def is_monitored(self, path: str) -> bool:
        """True when *path* lives under a monitored directory (lexical, local only)."""
        if is_remote_url(path):
            return False
        claim_path = Path(path)
        return bool(self._containing(claim_path, (RootKind.MONITORED,)))

    def display_url(self, claim_path: str) -> Optional[str]:
        """The display ``source_url`` for a source found at ``claim_path``, or None.

        What the root it is under makes of it: a drop's ``dnd://`` label (so the
        source goes with the drop), the innermost aliased monitored root's alias,
        or a ``monitor = false`` root's alias. None leaves the plain file url.
        Display-only: the source id still hashes the native path.
        """
        path = Path(claim_path)
        root = self.containing(path)
        if root is None:
            return None
        if root.kind is RootKind.DROPPED:
            return _drop_catalog_url(root.url, claim_path, label=root.label)
        if root.kind is RootKind.SCAN_ONCE:
            # Persistent: nothing rescans the root to re-merge it into the tree.
            alias_root = root if root.alias else None
        else:
            aliased = [
                r for r in self._containing(path, (RootKind.MONITORED,)) if r.alias
            ]
            alias_root = max(aliased, key=lambda r: len(r.path.parts), default=None)
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
                r is not exclude
                and (rp := r.path) is not None
                and rp.is_relative_to(path)
                for r in self._roots
            )
        ):
            return OVERLAP_MESSAGE
        return None
