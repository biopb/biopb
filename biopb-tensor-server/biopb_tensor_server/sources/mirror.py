"""The sources mirrored from one upstream tensor server.

A bare-host ``tensor-server`` entry is a root (:attr:`RootKind.UPSTREAM`) that is
re-listed rather than walked. What it mirrors is not a set of claims: there is no
file to stat, parse or re-find, and nothing to persist, since the root is rebuilt
from config on every start. A :class:`MirrorSet` holds the root and the version
each of its sources was last synced at, and keeps the catalog in step with the
upstream's own catalog.

A re-list writes catalog rows and builds no adapter. A source's adapter is built
by :meth:`MirrorSet.materialize` on the first read, from the row the catalog holds,
and is then let go of when idle like any other.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from biopb_tensor_server.adapters.remote_tensor import (
    RemoteTensorAdapter,
    _split_grpc_url,
    close_upstream_client,
    content_version_for,
    display_parts,
    fetch_upstream_rows,
    list_upstream_versions,
    localize_tensor_rows,
    mirror_display_url,
    open_upstream_client,
    resolve_upstream_credentials,
)
from biopb_tensor_server.core.errors import SourceUnresolvedError
from biopb_tensor_server.serving.metadata_db import MetadataDatabase, MirroredRow
from biopb_tensor_server.sources.resolve import namespaced_source_id
from biopb_tensor_server.sources.roots import Root

logger = logging.getLogger(__name__)


class MirrorSet:
    """The mirrors of one upstream: its root, and the version of each source."""

    def __init__(
        self,
        root: Root,
        server: Any,
        metadata_db: Optional[MetadataDatabase],
        is_claimed: Callable[[str], bool],
        ensure_root: Callable[[Root], None],
    ):
        """*is_claimed* says whether a source id belongs to a claim, which a mirror
        must not displace. *ensure_root* has the catalog know the root before a row
        sits under it."""
        self.root = root
        self._server = server
        self._metadata_db = metadata_db
        self._is_claimed = is_claimed
        self._ensure_root = ensure_root
        self._endpoint = _split_grpc_url(root.url)[0]
        self._scheme, self._authority = display_parts(self._endpoint, root.alias)
        # source id -> (upstream id, the upstream's ``indexed_at`` it was synced at):
        # all a mirror needs besides its catalog row to be rebuilt.
        self._versions: Dict[str, Tuple[str, object]] = {}
        # The rows themselves, for a set with no catalog to read them back from.
        self._held: Dict[str, Tuple[str, List[dict], bool]] = {}
        self._credentials: Any = None
        self._build_lock = threading.Lock()

    def owns(self, source_id: str) -> bool:
        """Whether *source_id* is one of this upstream's mirrors."""
        return source_id in self._versions

    def unbuilt(self) -> int:
        """How many mirrors the registry has no entry for. One it has let go of is
        still an entry, and counted there."""
        return sum(
            1
            for source_id in list(self._versions)
            if source_id not in self._server.sources
        )

    def _built(self, source_id: str) -> Optional[Any]:
        return self._server.sources.get(source_id)

    def materialize(self, source_id: str) -> bool:
        """Give a mirror its adapter, from the row the catalog holds. Returns whether
        the source has one: False for an id this set does not mirror.

        Whether the source is resolved is the row's to say, and the adapter does not
        change it: a row the upstream has not resolved gets no adapter, and the read
        is refused as unresolved.

        Single-flight, and the adapter is registered evictable, so an idle one is
        let go of and built again by the next read.
        """
        if source_id not in self._versions:
            return False
        with self._build_lock:
            entry = self._versions.get(source_id)
            if entry is None:
                return False
            if self._built(source_id) is not None:
                return True
            if self._metadata_db is not None:
                row = self._metadata_db.read_mirrored(source_id)
            else:
                row = self._held.get(source_id)
            if row is None:
                return False
            url, tensors, resolved = row
            if not resolved:
                raise SourceUnresolvedError(
                    f"source {source_id!r} is not resolved on its upstream "
                    f"{self._endpoint}"
                )
            upstream_id, indexed_at = entry
            adapter = RemoteTensorAdapter(
                source_id,
                self._endpoint,
                upstream_id,
                credentials=self._credentials,
                alias=self.root.alias,
            )
            adapter.restore_catalog(tensors, url, indexed_at)
            self._server.register_source(source_id, adapter, evictable=True)
        return True

    def relist(self, credentials_config: Optional[Any]) -> bool:
        """Bring the mirrors to what the upstream lists now.

        Returns whether the set changed (a source was added or removed), which
        drives the adaptive re-list cadence. Raises when the upstream cannot be
        queried, leaving everything as it was.

        A narrow id + ``indexed_at`` pass decides everything that follows, so a
        steady re-list of a six-figure catalog moves two columns, not every
        source's metadata. It is complete (the upstream's catalog is not
        row-capped), so what it no longer lists is gone. A mirror needs its row
        again only when the upstream re-registered the source (``indexed_at``
        moved), or when it is new; an unversioned upstream has no ``indexed_at``
        to compare, so it is re-read every time.
        """
        alias = self.root.alias
        credentials = resolve_upstream_credentials(self.root.source, credentials_config)
        self._credentials = credentials
        client = open_upstream_client(self._endpoint, credentials)
        added = 0
        try:
            versions = list_upstream_versions(client)
            desired = {namespaced_source_id(alias, up): up for up in versions}

            removed = sorted(self._versions.keys() - desired.keys())
            self._remove(removed)

            new = sorted(desired.keys() - self._versions.keys())
            stale = sorted(
                source_id
                for source_id, (_, held) in self._versions.items()
                if source_id in desired
                and not self._is_current(held, versions[desired[source_id]].indexed_at)
            )
            wanted = [desired[source_id] for source_id in new + stale]
            sizes = {up: versions[up].size for up in wanted}
            # One bounded batch of full rows at a time, so the first sync of a large
            # catalog never holds it whole.
            for rows in fetch_upstream_rows(client, wanted, sizes):
                added += self._apply(rows)
        finally:
            close_upstream_client(client)

        if added or removed:
            logger.info(
                "Upstream %s re-list: +%d / -%d sources",
                self._endpoint,
                added,
                len(removed),
            )
        return bool(added or removed)

    @staticmethod
    def _is_current(held: object, indexed_at: object) -> bool:
        """Whether a source synced at *held* is as the upstream lists it now. An
        unversioned upstream is never current, so it is re-read."""
        version = content_version_for(indexed_at)
        return version is not None and version == content_version_for(held)

    def _apply(self, rows: Sequence[dict]) -> int:
        """Write the catalog rows of *rows* and record their versions. Returns how
        many sources were added.

        A failure leaves nothing recorded and nothing re-seeded, so the next tick
        finds the same sources new or stale and tries again.
        """
        alias = self.root.alias
        versions: Dict[str, Tuple[str, object]] = {}
        by_source: Dict[str, dict] = {}
        entries: List[MirroredRow] = []
        held: Dict[str, Tuple[str, List[dict], bool]] = {}
        added = 0
        for row in rows:
            upstream_id = row["source_id"]
            source_id = namespaced_source_id(alias, upstream_id)
            if source_id not in self._versions:
                if source_id in self._server.sources or self._is_claimed(source_id):
                    logger.warning(
                        "Not mirroring %s from %s: its id is already a source of "
                        "this server. Give the upstream an alias to namespace it.",
                        upstream_id,
                        self._endpoint,
                    )
                    continue
                added += 1
            entry = self._entry(source_id, upstream_id, row)
            entries.append(entry)
            versions[source_id] = (upstream_id, row.get("indexed_at"))
            by_source[source_id] = row
            if self._metadata_db is None:
                held[source_id] = (
                    self._url(upstream_id, row),
                    entry.tensors,
                    entry.is_resolved,
                )

        if self._metadata_db is not None and entries:
            try:
                self._ensure_root(self.root)
                self._metadata_db.sync_mirrored_rows(
                    self.root.root_id, "tensor-server", entries
                )
            except Exception:
                logger.exception(
                    "Failed to catalogue the mirrored sources of %s; they are retried",
                    self._endpoint,
                )
                return 0

        # Under the build lock, so an adapter being built from the row as it was
        # is re-seeded here, not left serving it.
        with self._build_lock:
            self._versions.update(versions)
            self._held.update(held)
            for source_id, row in by_source.items():
                # A built adapter is re-seeded in place: the next read sees the new
                # tensors and version without a rebuild.
                adapter = self._built(source_id)
                if adapter is not None:
                    adapter.seed_catalog(
                        row.get("tensors"),
                        row.get("source_url"),
                        row.get("indexed_at"),
                    )
        return added

    def _url(self, upstream_id: str, row: dict) -> str:
        return mirror_display_url(
            self._scheme, self._authority, upstream_id, row.get("source_url")
        )

    def _entry(self, source_id: str, upstream_id: str, row: dict) -> MirroredRow:
        """The catalog row of a mirror: the upstream's, with its ids made local.

        Placed beneath the root by the url the mirror shows: the upstream's own
        path where it has one, else its id.
        """
        url = self._url(upstream_id, row)
        prefix = self.root.root_url + "/"
        return MirroredRow(
            source_id,
            url[len(prefix) :] if url.startswith(prefix) else row["source_id"],
            row.get("metadata_json") or None,
            # An upstream predating the column has only resolved sources.
            bool(row.get("is_resolved", True)),
            localize_tensor_rows(row.get("tensors"), upstream_id, source_id),
        )

    def _remove(self, source_ids: Sequence[str]) -> None:
        """Drop sources the upstream no longer lists, from the registry and the
        catalog in one transaction."""
        for source_id in source_ids:
            if source_id in self._server.sources:
                self._server.unregister_source(source_id)
            del self._versions[source_id]
            self._held.pop(source_id, None)
        if self._metadata_db is None:
            return
        try:
            self._metadata_db.sync_mirrored_removed(source_ids)
        except Exception:
            logger.exception(
                "Failed to remove %d mirrored sources of %s from metadata DB",
                len(source_ids),
                self._endpoint,
            )
