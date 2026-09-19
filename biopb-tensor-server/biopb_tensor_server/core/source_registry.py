"""Thread-safe registry of the server's live source adapters.

Extracted from ``TensorFlightServer`` (biopb/biopb#278 item A). Owns the
``source_id -> SourceAdapter`` map, the single registration chokepoint (the
slash-free id validation the tensor identity policy requires), and
adapter-lifecycle cleanup -- closing long-lived OS handles on unregister and
shutdown.

The registry is deliberately dict-like (``get``/``__contains__``/``__iter__``/
``values``/``len``) so call sites read naturally, but every access is guarded by
one ``RLock`` -- the same single lock the server used to hold, so lock semantics
are unchanged by the extraction.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from biopb_tensor_server.core.adapter_base import SourceAdapter
from biopb_tensor_server.core.normalize import normalize_adapter

logger = logging.getLogger(__name__)


def close_adapter(adapter: Optional[SourceAdapter]) -> None:
    """Best-effort release of an adapter's resources (e.g. open file handles).

    Public because :meth:`SourceRegistry.swap` hands the displaced adapter back
    unclosed -- it may have to be restored rather than closed, so the choice is
    the caller's -- and whoever swapped then needs this same never-raises close
    for the branch where the replace committed.

    ``SourceAdapter.close()`` is declared on the ABC with a no-op default, so
    this calls it rather than sniffing for it: a wrapper that forwards every
    other method but not ``close`` is then a visible omission in the interface
    instead of a silent skip (biopb/biopb#71). Never raises -- shutdown and
    unregister must not fail on a balky adapter, and the registry also accepts
    non-inheriting test doubles.
    """
    if adapter is None:  # unregister of an id that was never registered
        return
    try:
        adapter.close()
    except Exception:  # cleanup must not fail
        logger.debug("error closing source adapter", exc_info=True)


class SourceRegistry:
    """The server's live ``source_id -> SourceAdapter`` map, thread-safe."""

    def __init__(self, labels_dir: Optional[Path] = None) -> None:
        """*labels_dir* is ``<write_dir>/labels``: where uploaded label sets
        live, one directory per source_id. None on a server with no
        ``write_dir``, which then has no sidecars to attach."""
        self._sources: Dict[str, SourceAdapter] = {}
        self._lock = threading.RLock()
        self._labels_dir = labels_dir

    def register(self, source_id: str, adapter: SourceAdapter) -> SourceAdapter:
        """Register a data source, in canonical axis order.

        Being the single registration chokepoint, this is also where the
        canonical-axis-order guarantee is applied (biopb/biopb#596): the adapter
        is passed through :func:`normalize_adapter`, which wraps it only if its
        axes are not already canonical. **Returns the registered adapter** --
        the same object in the overwhelmingly common compliant case, the wrapper
        otherwise. Callers that keep using the adapter after registering it (to
        sync the catalog row, say) must use the return value, or the catalog
        would describe a different axis order than the serve path.

        Args:
            source_id: Unique identifier for the data source. Must be non-empty
                and slash-free (see Raises).
            adapter: Source adapter for the data source

        Returns:
            The adapter as registered -- normalized if it needed it.

        Raises:
            ValueError: If *source_id* is empty or contains ``"/"``. The tensor
                identity policy (proto/biopb/tensor/descriptor.proto) requires a
                slash-free source_id: the internal chunk-route id is
                ``"source_id/array_id"`` and is decoded by splitting on the first
                ``"/"``, so a ``"/"`` in source_id would make the
                (source_id, array_id) pair undecodable. Auto-generated ids are
                already slash-free; this guards caller-supplied ones. This is the
                single registration chokepoint -- discovery, the source manager,
                uploads, and direct use all funnel through here.
        """
        if not source_id:
            raise ValueError("register: source_id must be non-empty")
        if "/" in source_id:
            raise ValueError(
                f"register: source_id must not contain '/' (got "
                f"{source_id!r}); the chunk-route id source_id/array_id decodes "
                f"by splitting on the first '/'."
            )
        adapter = normalize_adapter(adapter)
        self._attach_sidecar_labels(source_id, adapter)
        with self._lock:
            self._sources[source_id] = adapter
        logger.debug(f"Registered source: {source_id}")
        return adapter

    def _attach_sidecar_labels(self, source_id: str, adapter: SourceAdapter) -> None:
        """Attach the finished uploaded label sets this source has on disk.

        Here, at the one registration chokepoint, because a sidecar is keyed
        by ``source_id`` and no format knows about it (biopb/biopb#1059). A
        sidecar that will not open costs the set, never the source: a source
        is its pixels first.
        """
        if self._labels_dir is None:
            return
        try:
            from biopb_tensor_server.adapters.labels import sidecar_label_sets

            for field, label_set in sidecar_label_sets(
                source_id, self._labels_dir, adapter
            ).items():
                adapter.attach_label_set(field, label_set)
        except Exception:
            logger.warning(
                f"labels: could not attach sidecar sets of {source_id}", exc_info=True
            )

    def register_new(
        self, source_id: str, adapter: SourceAdapter
    ) -> Optional[SourceAdapter]:
        """:meth:`register`, refused if *source_id* is already taken.

        Returns the registered adapter, or ``None`` when the id is held -- by
        anything, a live upload or a sealed one. Atomic, so two concurrent
        creates of one name cannot both be told they own it. The caller still
        owns *adapter* on a refusal and closes it.

        Only an upload's ``create_tensor`` calls this rather than
        :meth:`register` directly: an upload name is caller-chosen, so a
        collision could mean someone else's data getting swapped in under an
        id already handed out. Discovery and the reconciler still call
        :meth:`register` (silent overwrite) because their source_id is
        ``generate_source_id``'s hash of the resolved source URL -- a
        collision there is definitionally the same source being re-registered
        (a rescan, a content update at the same path), never a distinct
        source claiming a name that is not its own.
        """
        with self._lock:
            if source_id in self._sources:
                return None
            return self.register(source_id, adapter)

    def unregister(self, source_id: str) -> Optional[SourceAdapter]:
        """Remove a source and release its adapter's resources.

        Returns the removed adapter (or ``None`` if it was not registered).

        Closed here, where :meth:`swap` hands its displaced adapter back open:
        the id is gone, so there is no rollback that could need this one back.
        The reader that resolved it a moment ago and is still decoding from it
        is ``SourceAdapter.close``'s concern, not this method's -- see
        :meth:`swap` for why that is not what separates the two.
        """
        with self._lock:
            adapter = self._sources.pop(source_id, None)
        close_adapter(adapter)
        logger.debug(f"Unregistered source: {source_id}")
        return adapter

    def swap(
        self, source_id: str, adapter: SourceAdapter
    ) -> Tuple[SourceAdapter, Optional[SourceAdapter]]:
        """Put *adapter* in place of ``source_id``'s current one, atomically.

        Returns ``(registered, displaced)`` -- the adapter as registered (see
        :meth:`register` on normalization) and the one it replaced, or None if
        the id was free. That handback is this method's whole content -- a bare
        :meth:`register` over a live id leaks what it overwrote.

        **The displaced adapter is not closed, because the swap may be undone.**
        A replace that fails after the swap -- the catalog upsert raises, say --
        puts the displaced adapter back and goes on serving from it
        (``Reconciler._restore_displaced_source``), so a registry that closed it
        here would restore a source that is live in ListFlights and broken on
        its first read. Ownership therefore passes to the caller, which either
        closes with :func:`close_adapter` once the replace has committed, or
        restores instead of closing.

        What does *not* separate the two methods is draining in-flight readers.
        :meth:`unregister` closes on the spot under exactly the same condition
        -- a reader that resolved the adapter a moment earlier is still decoding
        from it -- because that is ``SourceAdapter.close``'s own obligation in
        both directions: ``OmeTiffAdapter`` drains on ``_active_reads``, mrc /
        dv / qptiff defer to the reaper while a read is active, czi / nd2 /
        ndtiff / bioio release under ``_io_lock``, and the default holds nothing.
        No caller of either method owes a drain.
        """
        with self._lock:
            displaced = self._sources.get(source_id)
            registered = self.register(source_id, adapter)
        logger.debug(f"Swapped source adapter: {source_id}")
        return registered, displaced

    def get(self, source_id: str) -> Optional[SourceAdapter]:
        """Thread-safe source lookup."""
        with self._lock:
            return self._sources.get(source_id)

    def snapshot(self) -> List[Tuple[str, SourceAdapter]]:
        """Return a stable snapshot of registered sources for iteration."""
        with self._lock:
            return list(self._sources.items())

    def close_all(self) -> None:
        """Release every registered adapter's resources (shutdown).

        Some adapters hold long-lived OS handles (e.g. the OME-TIFF adapter's
        persistent aszarr store). Closing them on shutdown releases those
        handles -- required on Windows, where an open file cannot be deleted
        (otherwise a test's TemporaryDirectory cleanup raises WinError 32).
        """
        with self._lock:
            adapters = list(self._sources.values())
        for adapter in adapters:
            close_adapter(adapter)

    def replace(self, mapping: Dict[str, SourceAdapter]) -> None:
        """Atomically swap the whole map (used by tests to inject fixtures)."""
        with self._lock:
            self._sources = dict(mapping)

    def __contains__(self, source_id: str) -> bool:
        with self._lock:
            return source_id in self._sources

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._sources.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)

    def values(self) -> List[SourceAdapter]:
        with self._lock:
            return list(self._sources.values())
