"""Batched catalog rows for claimed sources that are not registered yet.

A walk that claims tens of thousands of sources gives each a ``pending``
catalog row. One ``INSERT`` per row costs about 4 ms (mostly DuckDB's per-statement
work, not the disk), which on a fast filesystem is most of the walk; one
multi-row ``INSERT`` costs about 0.06 ms a row. The walk therefore hands its rows
to a :class:`PendingRowWriter`, whose thread writes them in batches.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from biopb_tensor_server.core.discovery import SourceClaim
    from biopb_tensor_server.serving.metadata_db import CatalogRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingRow:
    """The catalog row of a claimed source: built from the claim, no file opened."""

    claim: SourceClaim
    catalog_url: Optional[str] = None
    #: A cloud source: ``needs_recall`` instead of ``pending``.
    recall: bool = False
    #: The claim and signature to persist with the row; None for a source with no
    #: claim to restore (a drop, a mirror), whose row is volatile.
    record: Optional[CatalogRecord] = None


class PendingRowWriter:
    """Buffers :class:`PendingRow` and writes them from one thread, in batches.

    ``add`` never touches the database, so a walker thread is never held up by
    it. A batch is written when ``batch_size`` rows are waiting or ``interval``
    seconds have passed, so the catalog still fills in while the walk runs.

    A failed batch is logged and dropped. Its claims stay pending, and the
    registration that follows writes the row, so nothing is lost; what is lost is
    only that the claim is not rolled back, as a failed single write would.
    """

    def __init__(
        self,
        write: Callable[[Sequence[PendingRow]], None],
        batch_size: int = 500,
        interval: float = 0.25,
    ) -> None:
        self._write = write
        self._batch_size = batch_size
        self._interval = interval
        self._rows: Dict[str, PendingRow] = {}
        self._cond = threading.Condition()
        # Held for the whole of a write, so ``discard`` can wait one out.
        self._flush_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="pending-row-writer"
        )
        self._thread.start()

    def add(self, row: PendingRow) -> None:
        with self._cond:
            self._rows[row.claim.source_id] = row
            if len(self._rows) >= self._batch_size:
                self._cond.notify()

    def discard(self, source_id: str) -> None:
        """Forget a source's buffered row, and wait out a write that may hold it.

        Called before the source's catalog row is deleted: once this returns the
        row is either in the catalog (and the delete takes it out) or never will
        be, so a removed source cannot be written back afterwards.
        """
        with self._flush_lock:
            with self._cond:
                self._rows.pop(source_id, None)

    def flush(self) -> None:
        with self._flush_lock:
            with self._cond:
                batch: List[PendingRow] = list(self._rows.values())
                self._rows.clear()
            if not batch:
                return
            try:
                self._write(batch)
            except Exception:
                logger.exception(
                    "Could not write %d pending catalog rows; their sources stay "
                    "pending and are catalogued when they register",
                    len(batch),
                )

    def close(self) -> None:
        """Stop the thread and write what is left."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join()
        self.flush()

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._closed and len(self._rows) < self._batch_size:
                    self._cond.wait(self._interval)
                if self._closed:
                    return
            self.flush()
