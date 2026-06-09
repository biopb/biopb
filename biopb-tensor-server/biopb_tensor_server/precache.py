"""Background pre-cache worker for the tensor server.

When a new source is added to the catalog *after* startup, this worker warms the
file cache for that source at the coarsest pyramid level a client requests on
open (see ``compute_precache_scale_hint``), so the first view a scientist opens
is already warm instead of paying a cold decode+downsample on the critical path.

Design constraints (all best-effort, never fatal to the server):

- **File backend only.** Inert unless the cache is the persistent
  ``ArrowFileBackend``; on a memory backend it drops queued work.
- **Runtime additions only.** The queue is fed by ``SourceManager``'s commit
  hook, which only fires for sources added after ``start()`` -- the initial
  startup scan is excluded.
- **Stays out of the way.** Before each chunk it waits until the Flight server
  has been idle for ``idle_debounce_seconds`` (no in-flight ``do_get``), and it
  re-checks between chunks so a burst of live traffic preempts it at chunk
  granularity. On the locked adapters precache's reads also serialize behind
  live reads through the per-source ``_io_lock``, so it never races a
  non-thread-safe reader.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Optional, Set

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.cache import ArrowFileBackend, CacheManager
from biopb_tensor_server.chunk import compute_precache_scale_hint

if TYPE_CHECKING:
    from biopb_tensor_server.config import PrecacheConfig
    from biopb_tensor_server.server import TensorFlightServer

logger = logging.getLogger(__name__)

# How often to re-check idle/stop while waiting for the server to quiesce.
_POLL_INTERVAL_SECONDS = 0.2


class PrecacheWorker:
    """Daemon thread that warms the file cache for newly-added sources."""

    def __init__(self, server: "TensorFlightServer", config: "PrecacheConfig"):
        self._server = server
        self._cfg = config
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._seen: Set[str] = set()
        self._seen_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("PrecacheWorker already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PrecacheWorker"
        )
        self._thread.start()
        logger.info("PrecacheWorker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("PrecacheWorker stopped")

    # -- producer API (handed to SourceManager._on_source_committed) -------

    def enqueue(self, source_id: str) -> None:
        """Queue a source for warming (non-blocking, deduplicated)."""
        with self._seen_lock:
            if source_id in self._seen:
                return
            self._seen.add(source_id)
        self._queue.put(source_id)

    # -- worker loop -------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                source_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Drop the dedup marker before processing: a commit that arrives
            # while we work should be allowed to re-queue a fresh pass.
            with self._seen_lock:
                self._seen.discard(source_id)
            try:
                self._process_source(source_id)
            except Exception:
                logger.exception("precache: failed for source %s", source_id)

    def _file_backend_active(self) -> bool:
        """True only when the persistent file cache is in use."""
        cache_manager = CacheManager.get_instance()
        return cache_manager is not None and isinstance(
            cache_manager.backend, ArrowFileBackend
        )

    def _wait_until_idle(self) -> bool:
        """Block until the Flight server is idle. Return False if asked to stop."""
        debounce = self._cfg.idle_debounce_seconds
        while not self._stop.is_set():
            if self._server.flight_idle_for(debounce):
                return True
            self._stop.wait(_POLL_INTERVAL_SECONDS)
        return False

    def _process_source(self, source_id: str) -> None:
        # Runtime file-backend gate: the "only run if file-based caching"
        # condition, enforced regardless of config.
        if not self._file_backend_active():
            logger.debug("precache: file backend not active, skipping %s", source_id)
            return
        cache_manager = CacheManager.get_instance()

        source_adapter = self._server._get_source_adapter(source_id)
        if source_adapter is None:
            return
        try:
            descriptors = source_adapter.list_tensor_descriptors()
        except Exception:
            logger.exception(
                "precache: list_tensor_descriptors failed for %s", source_id
            )
            return

        for td in descriptors:
            if self._stop.is_set():
                return
            self._process_tensor(source_adapter, td, cache_manager)

    def _process_tensor(self, source_adapter, td, cache_manager) -> None:
        # The client passes the descriptor's array_id verbatim as tensor_id
        # (TensorFlightClient), so the request we build mirrors get_flight_info.
        tensor_id = td.array_id
        try:
            tensor_adapter = source_adapter.get_tensor_adapter(tensor_id)
        except Exception:
            logger.exception("precache: get_tensor_adapter failed for %s", tensor_id)
            return
        if tensor_adapter is None:
            return
        try:
            base_desc = tensor_adapter.get_tensor_descriptor()
        except Exception:
            logger.exception(
                "precache: get_tensor_descriptor failed for %s", tensor_id
            )
            return

        scale = compute_precache_scale_hint(
            list(base_desc.shape),
            list(base_desc.dim_labels),
            threshold=self._cfg.threshold,
            downscale_factor=self._cfg.downscale_factor,
            pixel_budget_cubic_root=self._cfg.pixel_budget_cubic_root,
        )

        # Build the request descriptor exactly as get_flight_info does, so the
        # read plan's scaled chunk_ids match what the client will fetch.
        request_desc = TensorDescriptor(
            array_id=tensor_id,
            dim_labels=base_desc.dim_labels,
            shape=base_desc.shape,
            chunk_shape=base_desc.chunk_shape,
            dtype=base_desc.dtype,
        )
        request_desc.scale_hint[:] = scale
        request_desc.reduction_method = self._cfg.reduction_method

        try:
            read_plan = tensor_adapter.get_read_plan(request_desc)
        except Exception:
            logger.exception("precache: get_read_plan failed for %s", tensor_id)
            return

        endpoints = read_plan.chunk_endpoints
        warmed = 0
        for ce in endpoints:
            if self._stop.is_set():
                return
            # Debounce + preempt between chunks: wait for the server to be idle
            # before warming each chunk.
            if not self._wait_until_idle():
                return
            try:
                tensor_adapter.resolve_chunk_data(ce.chunk_id, cache_manager)
                warmed += 1
            except Exception as e:
                # One bad chunk shouldn't abort the whole tensor.
                logger.debug("precache: chunk warm failed for %s: %s", tensor_id, e)

        logger.info(
            "precache: warmed %d/%d chunks for %s at scale=%s",
            warmed,
            len(endpoints),
            tensor_id,
            scale,
        )
