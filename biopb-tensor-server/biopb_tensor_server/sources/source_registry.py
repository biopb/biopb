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
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.core.adapter_base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.attachments import Attachments
from biopb_tensor_server.core.normalize import normalize_adapter

logger = logging.getLogger(__name__)


def close_adapter(adapter: Optional[SourceAdapter]) -> None:
    """Best-effort release of an adapter's resources (e.g. open file handles).

    Public because :meth:`SourceRegistry.swap` hands the displaced adapter back
    unclosed -- it may have to be restored rather than closed, so the choice is
    the caller's -- and whoever swapped then needs this same never-raises close
    for the branch where the replace committed.

    Closes the adapter only: its attached tensors belong to the registry
    (:meth:`SourceRegistry.attached_to`) and outlive the adapter.

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

    def __init__(self) -> None:
        self._sources: Dict[str, SourceAdapter] = {}
        self._lock = threading.RLock()
        self._pending_check: Optional[Callable[[str], None]] = None
        # Tensors attached to a source -- uploaded fields, label sets -- keyed
        # by source id, not by adapter: an adapter is rebuilt on a refresh, and
        # an upload in flight must keep routing across it.
        self._attachments: Dict[str, Attachments] = {}

    def set_pending_check(self, check: Optional[Callable[[str], None]]) -> None:
        """Wire what :meth:`get_registered` asks when a source is not registered:
        *check* raises why a read of a claimed source cannot be served (it is
        unresolved, or its registration failed) and returns for any other id."""
        self._pending_check = check

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
        with self._lock:
            self._sources[source_id] = adapter
        logger.debug(f"Registered source: {source_id}")
        return adapter

    def unregister(self, source_id: str) -> Optional[SourceAdapter]:
        """Remove a source and release its adapter's resources.

        Returns the removed adapter (or ``None`` if it was not registered).
        Its attached tensors stay: their stores are the server's, on disk, and
        the id may come back; :meth:`close_all` releases them.

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

    def get_registered(self, source_id: str) -> Optional[SourceAdapter]:
        """:meth:`get`, raising when the source is claimed but not registered.

        What a reader of the source's data or tensors uses: a pending source is
        not read until a client resolves it (the ``resolve`` action). Raises why it
        cannot be read -- unresolved while its registration is waiting, a
        registration error once it failed -- and returns None for an unknown
        source. The internal callers that only ask whether it is registered, or
        read its url, use :meth:`get`.
        """
        adapter = self.get(source_id)
        if adapter is None and self._pending_check is not None:
            self._pending_check(source_id)
            adapter = self.get(source_id)  # registered while we checked
        return adapter

    # -- attachments ----------------------------------------------------------

    def attached_to(self, source_id: str) -> Attachments:
        """The attachments of *source_id*, created empty on first ask."""
        with self._lock:
            attachments = self._attachments.get(source_id)
            if attachments is None:
                attachments = self._attachments[source_id] = Attachments(source_id)
            return attachments

    def adopt(self, index: Dict[str, Dict[str, TensorAdapter]]) -> None:
        """Take the attached tensors a previous life left on disk, by source id.

        Run once at boot, so a registration is a dict lookup rather than a
        directory scan. A source not registered yet keeps its tensors here
        until it is.
        """
        for source_id, tensors in index.items():
            self.attached_to(source_id).tensors.update(tensors)

    def attachments(self, source_id: str) -> Dict[str, TensorAdapter]:
        """The tensors attached to *source_id*, whatever state: a copy."""
        with self._lock:
            found = self._attachments.get(source_id)
            return dict(found.tensors) if found is not None else {}

    def attached(self, source_id: str, field: str) -> Optional[TensorAdapter]:
        """The tensor attached at *field* of *source_id*, whatever its state."""
        with self._lock:
            found = self._attachments.get(source_id)
            return found.tensors.get(field) if found is not None else None

    def attachment_snapshot(self) -> List[Tuple[str, Dict[str, TensorAdapter]]]:
        """Every source's attachments, for a sweep that walks them all."""
        with self._lock:
            return [
                (sid, dict(a.tensors))
                for sid, a in self._attachments.items()
                if a.tensors
            ]

    def attach(self, source_id: str, field: str, tensor: TensorAdapter) -> None:
        """Make *tensor* answer for *field* on *source_id*."""
        self.attached_to(source_id).attach(field, tensor)

    def detach(self, source_id: str, field: str) -> Optional[TensorAdapter]:
        """Stop answering for *field*; returns what was attached, or None."""
        return self.attached_to(source_id).detach(field)

    def attachment_changed(self, source_id: str) -> None:
        """Rebuild the source's checked views: an attached tensor's state moved."""
        self.attached_to(source_id).changed()

    # -- reads through the attachments ---------------------------------------

    def catalog_tensors(
        self, source_id: str, adapter: Optional[SourceAdapter] = None
    ) -> List[TensorDescriptor]:
        """A source's tensors as the catalog stores them: the format's own, then
        its attached fields and label sets (:meth:`Attachments.catalog_tensors`).

        *adapter* is the one being described when it is not the registered one
        yet; attachments are the id's either way.
        """
        adapter = adapter if adapter is not None else self.get(source_id)
        return self.attached_to(source_id).catalog_tensors(adapter)

    def resolve_tensor(
        self, source_id: str, tensor_id: Optional[str]
    ) -> Optional[TensorAdapter]:
        """The adapter bound to *tensor_id* of *source_id*, None for an unknown
        source. Raises what :meth:`get_registered` raises."""
        adapter = self.get_registered(source_id)
        if adapter is None:
            return None
        return self.attached_to(source_id).resolve_tensor(adapter, tensor_id)

    def resolve_chunk_adapter(
        self, source_id: str, field: Optional[str]
    ) -> Optional[TensorAdapter]:
        """The adapter that serves a chunk whose route is ``source_id/<field>``,
        None for an unknown source. Raises what :meth:`get_registered` raises."""
        adapter = self.get_registered(source_id)
        if adapter is None:
            return None
        return self.attached_to(source_id).resolve_chunk_adapter(adapter, field)

    def tensor_capability_token(
        self, source_id: str, array_id: Optional[str]
    ) -> Optional[str]:
        """The grant the tensor *array_id* carries, or None (a source's own
        tensors carry none)."""
        with self._lock:
            found = self._attachments.get(source_id)
        return found.capability_token(array_id) if found is not None else None

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
            attached = [
                t for a in self._attachments.values() for t in a.tensors.values()
            ]
        for tensor in attached:
            try:
                tensor.close()
            except Exception:  # cleanup must not fail
                logger.debug("error closing attached tensor", exc_info=True)
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
