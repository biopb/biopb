"""The catalog as the Tensor Browser shows it, and what keeps it current.

Qt-free, so it runs and tests without a display. It reads through the shared
:class:`biopb.tensor.Connection` and holds the one cache in the picture: the
list of sources the tree is drawn from.

Threading: the watcher re-lists from its own daemon thread and only ever
*rebinds* ``sources`` to a fresh dict, so a reader on the Qt thread sees the
whole old list or the whole new one.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

from biopb.tensor._catalog_rows import SOURCE_ROW_COLUMNS

from .._catalog import CatalogSource, source_from_row, sources_from_rows

logger = logging.getLogger(__name__)

#: Catalogs larger than this filter server-side, in SQL.
SERVER_QUERY_THRESHOLD = 1000

#: The watcher's poll interval backs off from the first to the second while
#: the source count holds, and snaps back on a change.
WATCH_MIN_INTERVAL_S = 2.0
WATCH_MAX_INTERVAL_S = 60.0

_SOURCES_SQL = f"SELECT {SOURCE_ROW_COLUMNS} FROM sources ORDER BY source_id"


class SourceList:
    """The sources the tree shows, re-listed when the server's count moves."""

    def __init__(self, connection) -> None:
        self._conn = connection
        self.sources: Dict[str, CatalogSource] = {}
        # The last health answer, so the paint thread can tell "still indexing"
        # from "empty" without a round trip.
        self.last_health: dict | None = None
        # Called from the watcher thread with the fresh dict after a re-list.
        self.on_changed = None
        self._watch_thread: threading.Thread | None = None
        self._watch_stop = threading.Event()

    def _client(self):
        client = self._conn.client
        if client is None:
            raise RuntimeError("Not connected")
        return client

    @property
    def use_server_query(self) -> bool:
        return len(self.sources) > SERVER_QUERY_THRESHOLD

    def clear(self) -> None:
        self.sources = {}
        self.last_health = None

    def refresh(self) -> Dict[str, CatalogSource]:
        """Re-list the whole catalog from the server."""
        rows = self._client().query_sources(_SOURCES_SQL, format="records")
        self.sources = {s.source_id: s for s in sources_from_rows(rows)}
        return self.sources

    def update_health(self) -> dict | None:
        """Ask the server how it is; kept for :meth:`scan_in_progress`."""
        health = self._client().health_check()
        self.last_health = health if isinstance(health, dict) else None
        return self.last_health

    def scan_in_progress(self) -> bool:
        h = self.last_health
        return bool(h.get("full_scan_in_progress")) if h else False

    def scan_source_count(self) -> int:
        h = self.last_health
        try:
            return int(h.get("source_count") or 0) if h else 0
        except (TypeError, ValueError):
            return 0

    # --- the server's verbs, followed by a re-list ---------------------------

    def resolve(self, source_id: str, *, on_progress=None, should_cancel=None):
        """Resolve a cloud source (downloads it; call off the GUI thread)."""
        row = self._client().resolve(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )
        self.refresh()
        # The row this resolve committed, not the refreshed entry, which a
        # concurrent rescan could have re-registered underneath.
        return source_from_row(row)

    def warm(self, source_id: str, *, on_progress=None, should_cancel=None):
        """Recall a multi-file source's members. Residency is not in the
        catalog, so there is nothing to re-list."""
        return self._client().warm(
            source_id, on_progress=on_progress, should_cancel=should_cancel
        )

    def add(self, path: str, *, on_progress=None, should_cancel=None):
        result = self._client().add_source(
            path, on_progress=on_progress, should_cancel=should_cancel
        )
        self.refresh()
        return result

    def remove(self, root_url: str):
        result = self._client().remove_source(root_url)
        self.refresh()
        return result

    # --- the watcher ---------------------------------------------------------

    def start_watch(
        self,
        min_interval: float = WATCH_MIN_INTERVAL_S,
        max_interval: float = WATCH_MAX_INTERVAL_S,
    ) -> None:
        """Re-list whenever the server's ``source_count`` changes.

        A catalog listed while the server was still indexing then fills itself
        in. A thread, not a ``QTimer``: ``health_check`` blocks. Idempotent.
        """
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(min_interval, max(max_interval, min_interval)),
            name="biopb-source-watch",
            daemon=True,
        )
        self._watch_thread.start()

    def stop_watch(self) -> None:
        self._watch_stop.set()

    def _watch_loop(self, min_interval: float, max_interval: float) -> None:
        """``last_count`` is the count the list was last reconciled against;
        ``None`` while disconnected, so the first connected poll compares
        against what was listed at connect and catches a mid-index partial.

        A source that gains tensors without a new source is not caught: the
        count does not move.
        """
        interval = min_interval
        last_count: int | None = None
        while not self._watch_stop.wait(interval):
            if self._conn.client is None:
                last_count = None
                interval = min_interval
                continue
            try:
                health = self.update_health()
            except Exception:  # noqa: BLE001 - transient; back off
                interval = min(interval * 2, max_interval)
                continue
            count = health.get("source_count") if health else None
            if count is None:
                interval = max_interval
                continue
            if last_count is None:
                last_count = len(self.sources)
            if count != last_count:
                self._relist()
                last_count = count
                interval = min_interval
            else:
                interval = min(interval * 2, max_interval)

    def _relist(self) -> None:
        try:
            sources = self.refresh()
        except Exception:  # noqa: BLE001 - keep watching through a blip
            logger.exception("Source watch: re-list failed")
            return
        logger.info("Source watch: re-listed %d sources", len(sources))
        callback = self.on_changed
        if callback is not None:
            try:
                callback(sources)
            except Exception:  # noqa: BLE001
                logger.exception("Source watch: on_changed hook failed")
