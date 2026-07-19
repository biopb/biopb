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
from typing import Dict, Iterator, List, Optional, Tuple

from biopb_tensor_server.core.base import SourceAdapter

logger = logging.getLogger(__name__)


def _close_adapter(adapter: Optional[SourceAdapter]) -> None:
    """Best-effort release of an adapter's resources (e.g. open file handles).

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
    except Exception:  # pragma: no cover - cleanup must not fail
        logger.debug("error closing source adapter", exc_info=True)


class SourceRegistry:
    """The server's live ``source_id -> SourceAdapter`` map, thread-safe."""

    def __init__(self) -> None:
        self._sources: Dict[str, SourceAdapter] = {}
        self._lock = threading.RLock()

    def register(self, source_id: str, adapter: SourceAdapter) -> None:
        """Register a data source.

        Args:
            source_id: Unique identifier for the data source. Must be non-empty
                and slash-free (see Raises).
            adapter: Source adapter for the data source

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
        with self._lock:
            self._sources[source_id] = adapter
        logger.debug(f"Registered source: {source_id}")

    def unregister(self, source_id: str) -> Optional[SourceAdapter]:
        """Remove a source and release its adapter's resources.

        Returns the removed adapter (or ``None`` if it was not registered).
        """
        with self._lock:
            adapter = self._sources.pop(source_id, None)
        _close_adapter(adapter)
        logger.debug(f"Unregistered source: {source_id}")
        return adapter

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
            _close_adapter(adapter)

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
