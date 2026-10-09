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
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from biopb_tensor_server.core.adapter_base import (
    SourceAdapter,
    TensorAdapter,
    TensorEntry,
)
from biopb_tensor_server.core.attachments import Attachments
from biopb_tensor_server.core.normalize import is_canonical, log_reordering, unwrapped
from biopb_tensor_server.core.weak import weak_or_none

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


class _Slot:
    """One registered source: its adapter, and whether it can be let go.

    An evictable slot (``ref`` set) holds the adapter strongly while in use
    and, once released, only through ``ref``: it lives as long as a reader.
    """

    __slots__ = ("strong", "ref", "last_access")

    def __init__(self, adapter: SourceAdapter, evictable: bool) -> None:
        self.strong: Optional[SourceAdapter] = adapter
        self.ref: Optional[Callable[[], Any]] = (
            weak_or_none(adapter) if evictable else None
        )
        self.last_access = time.monotonic()

    @property
    def adapter(self) -> Optional[SourceAdapter]:
        if self.strong is not None or self.ref is None:
            return self.strong
        return self.ref()


class SourceRegistry:
    """The server's live ``source_id -> SourceAdapter`` map, thread-safe.

    A source registered ``evictable`` is one that can be rebuilt from its
    catalog row (:meth:`Reconciler.ensure_registered`). The registry holds it
    strongly while it is in use and :meth:`release_idle` demotes it to a weak
    reference once it has been idle, so an adapter nobody is reading goes with
    its last reader. The id stays registered throughout (``in``, ``len`` and
    iteration count it); only :meth:`get` stops finding the adapter, and
    :meth:`get_registered` asks the pending check to rebuild it.

    Anything else -- uploads, scratch, a proxy -- is held strongly for as long
    as it is registered.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, _Slot] = {}
        self._lock = threading.RLock()
        self._pending_check: Optional[Callable[[str], None]] = None
        # Tensors attached to a source -- uploaded fields, label sets -- keyed
        # by source id, not by adapter: an adapter is rebuilt on a refresh, and
        # an upload in flight must keep routing across it.
        self._attachments: Dict[str, Attachments] = {}
        self._attachment_listener: Optional[Callable[[str], None]] = None

    def set_attachment_listener(
        self, listener: Optional[Callable[[str], None]]
    ) -> None:
        """Tell *listener* the source id of every source whose attachments changed.

        What follows from a change in what a source lists -- its catalog row --
        is the listener's, so no caller has to remember it. Boot adoption
        (:meth:`adopt`) is not a change and does not call it."""
        self._attachment_listener = listener

    def set_pending_check(self, check: Optional[Callable[[str], None]]) -> None:
        """Wire what :meth:`get_registered` asks when a source is not registered:
        *check* raises why a read of a claimed source cannot be served (it is
        unresolved, or its registration failed) and returns for any other id."""
        self._pending_check = check

    def register(
        self, source_id: str, adapter: SourceAdapter, evictable: bool = False
    ) -> SourceAdapter:
        """Register a data source.

        Canonical axis order (biopb/biopb#596) is the adapter's own business
        (``TensorAdapter``); registering only reports a source that is served
        reordered.

        Args:
            source_id: Unique identifier for the data source. Must be non-empty
                and slash-free (see Raises).
            adapter: Source adapter for the data source
            evictable: Whether the source can be rebuilt once the registry has
                let go of its adapter (see the class docstring).

        Returns:
            The adapter, as passed.

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
        _log_reordering(source_id, adapter)
        with self._lock:
            self._sources[source_id] = _Slot(adapter, evictable)
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
            slot = self._sources.pop(source_id, None)
        adapter = slot.adapter if slot is not None else None
        close_adapter(adapter)
        logger.debug(f"Unregistered source: {source_id}")
        return adapter

    def swap(
        self, source_id: str, adapter: SourceAdapter, evictable: bool = False
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
            displaced = self.get(source_id)
            registered = self.register(source_id, adapter, evictable)
        logger.debug(f"Swapped source adapter: {source_id}")
        return registered, displaced

    def get(self, source_id: str) -> Optional[SourceAdapter]:
        """The source's adapter, or None when it has none: unknown, or evictable
        and let go of. Does not count as use (see :meth:`get_registered`)."""
        return self._lookup(source_id, touch=False)

    def _lookup(self, source_id: str, touch: bool) -> Optional[SourceAdapter]:
        """The adapter of *source_id*; with *touch*, a read of it: stamped, and
        held strongly again if it had been let go."""
        with self._lock:
            slot = self._sources.get(source_id)
            if slot is None:
                return None
            adapter = slot.adapter
            if touch and adapter is not None and slot.ref is not None:
                slot.last_access = time.monotonic()
                slot.strong = adapter
            return adapter

    def release_idle(self, idle_seconds: float, now: Optional[float] = None) -> int:
        """Let go of evictable adapters not read for *idle_seconds*.

        A reader that already has the adapter keeps it alive and nothing else
        does, so a read in flight needs no lease. Returns how many were released.
        """
        if now is None:
            now = time.monotonic()
        released = 0
        with self._lock:
            for slot in self._sources.values():
                if (
                    slot.ref is not None
                    and slot.strong is not None
                    and now - slot.last_access > idle_seconds
                ):
                    slot.strong = None
                    released += 1
        return released

    def get_registered(self, source_id: str) -> Optional[SourceAdapter]:
        """:meth:`get` as a read, raising when the source is claimed but not registered.

        What a reader of the source's data or tensors uses: a pending source is
        not read until a client resolves it (the ``resolve`` action). Raises why it
        cannot be read -- unresolved while its registration is waiting, a
        registration error once it failed -- and returns None for an unknown
        source. A source whose adapter was let go is rebuilt. The internal
        callers that only ask whether it is registered, or read its url, use
        :meth:`get`.
        """
        adapter = self._lookup(source_id, touch=True)
        if adapter is None and self._pending_check is not None:
            self._pending_check(source_id)
            adapter = self._lookup(source_id, touch=True)  # registered while we checked
        return adapter

    def resolve(self, source_id: str) -> Optional[SourceAdapter]:
        """:meth:`get_registered`, for a caller that wants the adapter or nothing:
        None for a source that is unknown, unresolved or failed."""
        try:
            return self.get_registered(source_id)
        except Exception:
            return None

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
        self._notify(source_id)

    def detach(self, source_id: str, field: str) -> Optional[TensorAdapter]:
        """Stop answering for *field*; returns what was attached, or None."""
        removed = self.attached_to(source_id).detach(field)
        if removed is not None:
            self._notify(source_id)
        return removed

    def attachment_changed(self, source_id: str) -> None:
        """Rebuild the source's checked views: an attached tensor's state moved."""
        self.attached_to(source_id).changed()
        self._notify(source_id)

    def _notify(self, source_id: str) -> None:
        if self._attachment_listener is not None:
            self._attachment_listener(source_id)

    # -- reads through the attachments ---------------------------------------

    def catalog_tensors(
        self, source_id: str, adapter: Optional[SourceAdapter] = None
    ) -> List[TensorEntry]:
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

    def tensor_capability_token(
        self, source_id: str, array_id: Optional[str]
    ) -> Optional[str]:
        """The grant the tensor *array_id* carries, or None (a source's own
        tensors carry none)."""
        with self._lock:
            found = self._attachments.get(source_id)
        return found.capability_token(array_id) if found is not None else None

    def snapshot(self) -> List[Tuple[str, SourceAdapter]]:
        """Return a stable snapshot of the sources that have an adapter now."""
        with self._lock:
            return [
                (source_id, adapter)
                for source_id, slot in self._sources.items()
                if (adapter := slot.adapter) is not None
            ]

    def close_all(self) -> None:
        """Release every registered adapter's resources (shutdown).

        Some adapters hold long-lived OS handles (e.g. the OME-TIFF adapter's
        persistent aszarr store). Closing them on shutdown releases those
        handles -- required on Windows, where an open file cannot be deleted
        (otherwise a test's TemporaryDirectory cleanup raises WinError 32).
        """
        with self._lock:
            adapters = self.values()
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
            self._sources = {sid: _Slot(a, False) for sid, a in mapping.items()}

    def __contains__(self, source_id: str) -> bool:
        with self._lock:
            return source_id in self._sources

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._sources))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)

    def values(self) -> List[SourceAdapter]:
        """The adapters that exist now (an evicted source has none)."""
        return [adapter for _, adapter in self.snapshot()]


def _log_reordering(source_id: str, adapter: SourceAdapter) -> None:
    """INFO-log the tensors ``adapter`` will serve reordered, if any.

    Never fails a registration: a duck-typed double, or an adapter that cannot
    list yet (an unresolved cloud source), just reports nothing.
    """
    if not is_canonical(adapter):
        return
    try:
        log_reordering(source_id, unwrapped(type(adapter).list_tensors)(adapter))
    except Exception:
        logger.debug(
            "axis normalization: could not inspect %r", source_id, exc_info=True
        )
