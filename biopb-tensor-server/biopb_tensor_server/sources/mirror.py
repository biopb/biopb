"""The sources mirrored from one upstream tensor server.

A bare-host ``tensor-server`` entry is a root (:attr:`RootKind.UPSTREAM`) that is
re-listed rather than walked. What it mirrors is not a set of claims: there is no
file to stat, parse or re-find, and nothing to persist, since the root is rebuilt
from config on every start. A :class:`MirrorSet` holds the root and the adapters
registered for its sources, and keeps both and the catalog in step with the
upstream's own catalog.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from biopb_tensor_server.adapters.remote_tensor import (
    RemoteTensorAdapter,
    _split_grpc_url,
    close_upstream_client,
    fetch_upstream_rows,
    list_upstream_versions,
    open_upstream_client,
    resolve_upstream_credentials,
)
from biopb_tensor_server.serving.metadata_db import MetadataDatabase, MirroredRow
from biopb_tensor_server.sources.resolve import namespaced_source_id
from biopb_tensor_server.sources.roots import Root

logger = logging.getLogger(__name__)


class MirrorSet:
    """The mirrors of one upstream: its root, and a source id -> adapter map."""

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
        self._adapters: Dict[str, RemoteTensorAdapter] = {}

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
        client = open_upstream_client(self._endpoint, credentials)
        added = 0
        try:
            versions = list_upstream_versions(client)
            desired = {namespaced_source_id(alias, up): up for up in versions}

            removed = sorted(self._adapters.keys() - desired.keys())
            self._remove(removed)

            new = sorted(desired.keys() - self._adapters.keys())
            stale = sorted(
                source_id
                for source_id, adapter in self._adapters.items()
                if source_id in desired
                and not adapter.is_current(versions[desired[source_id]].indexed_at)
            )
            wanted = [desired[source_id] for source_id in new + stale]
            sizes = {up: versions[up].size for up in wanted}
            # One bounded batch of full rows at a time, so the first sync of a large
            # catalog never holds it whole.
            for rows in fetch_upstream_rows(client, wanted, sizes):
                added += self._apply(rows, credentials)
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

    def _apply(self, rows: Sequence[dict], credentials: Any) -> int:
        """Register the new sources among *rows*, write all their catalog rows
        and re-seed the known ones. Returns how many sources were added.

        A failure leaves the catalog and the registry agreeing: the new adapters
        come back out and nothing is re-seeded, so the next tick finds the same
        sources new or stale and tries again.
        """
        alias = self.root.alias
        fresh: Dict[str, RemoteTensorAdapter] = {}
        known: List[Tuple[RemoteTensorAdapter, dict]] = []
        entries: List[MirroredRow] = []
        for row in rows:
            upstream_id = row["source_id"]
            source_id = namespaced_source_id(alias, upstream_id)
            adapter = self._adapters.get(source_id)
            if adapter is None:
                if self._server.sources.get(source_id) is not None or (
                    self._is_claimed(source_id)
                ):
                    logger.warning(
                        "Not mirroring %s from %s: its id is already a source of "
                        "this server. Give the upstream an alias to namespace it.",
                        upstream_id,
                        self._endpoint,
                    )
                    continue
                adapter = RemoteTensorAdapter(
                    source_id,
                    self._endpoint,
                    upstream_id,
                    credentials=credentials,
                    alias=alias,
                )
                self._seed(adapter, row)
                fresh[source_id] = adapter
            else:
                known.append((adapter, row))
            entries.append(self._entry(adapter, row))

        registered: List[str] = []
        try:
            for source_id, adapter in fresh.items():
                self._server.register_source(source_id, adapter)
                registered.append(source_id)
            if self._metadata_db is not None and entries:
                self._ensure_root(self.root)
                self._metadata_db.sync_mirrored_rows(
                    self.root.root_id, "tensor-server", entries
                )
        except Exception:
            logger.exception(
                "Failed to register mirrored sources of %s; they are retried",
                self._endpoint,
            )
            for source_id in registered:
                self._server.unregister_source(source_id)
            return 0

        self._adapters.update(fresh)
        for adapter, row in known:
            self._seed(adapter, row)
        return len(fresh)

    @staticmethod
    def _seed(adapter: RemoteTensorAdapter, row: dict) -> None:
        adapter.seed_catalog(
            row.get("tensors"), row.get("source_url"), row.get("indexed_at")
        )

    def _entry(self, adapter: RemoteTensorAdapter, row: dict) -> MirroredRow:
        """The catalog row of a mirror: the upstream's, with its ids made local.

        Placed beneath the root by the url the adapter shows: the upstream's own
        path where it has one, else its id.
        """
        url = adapter.display_url(row.get("source_url"))
        prefix = self.root.root_url + "/"
        return MirroredRow(
            adapter.source_id,
            url[len(prefix) :] if url.startswith(prefix) else row["source_id"],
            row.get("metadata_json") or None,
            # An upstream predating the column has only resolved sources.
            bool(row.get("is_resolved", True)),
            adapter.local_tensor_rows(row.get("tensors")),
        )

    def _remove(self, source_ids: Sequence[str]) -> None:
        """Drop sources the upstream no longer lists, from the registry and the
        catalog. One whose unregistering fails stays, and is tried again."""
        gone = []
        for source_id in source_ids:
            try:
                self._server.unregister_source(source_id)
            except Exception:
                logger.exception("Failed to unregister mirrored source %s", source_id)
                continue
            del self._adapters[source_id]
            gone.append(source_id)
        if self._metadata_db is None:
            return
        try:
            self._metadata_db.sync_mirrored_removed(gone)
        except Exception:
            logger.exception(
                "Failed to remove %d mirrored sources of %s from metadata DB",
                len(gone),
                self._endpoint,
            )
