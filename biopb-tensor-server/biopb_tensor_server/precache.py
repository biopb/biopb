"""Background pre-cache worker for the tensor server.

This worker warms the file cache so the first view a scientist opens is already
warm instead of paying a cold decode+downsample on the critical path. It warms
the coarsest level of the same pyramid the server advertises on the tensor
descriptor (see ``chunk.build_pyramid_plan``), so the warmed scale always matches
what the client requests on open.

It serves two tiers, in strict priority order:

- **Live tier (primary).** Sources added to the catalog *after* startup, fed by
  ``SourceManager``'s commit hook (``enqueue``). Always warmed.
- **Backlog tier (secondary).** Local sources already present at startup, seeded
  once via ``seed_backlog`` and ordered newest-mtime-first. Drained only when the
  live queue is empty, and bounded so it never evicts live data (see below).

Design constraints (all best-effort, never fatal to the server):

- **File backend only.** Inert unless the cache is the persistent
  ``ArrowFileBackend``; on a memory backend it drops queued work.
- **Stays out of the way.** Before each chunk it waits until the Flight server
  has been idle for ``idle_debounce_seconds`` (no in-flight ``do_get``), and it
  re-checks between chunks so a burst of live traffic preempts it at chunk
  granularity. On the locked adapters precache's reads also serialize behind
  live reads through the per-source ``_io_lock``, so it never races a
  non-thread-safe reader.
- **Backlog never evicts live data.** The file cache evicts globally on every
  write, so the backlog tier gates each chunk on cache fill and stops above
  ``backlog_high_water`` of the cache's ``max_bytes``, and yields the moment a
  live source is enqueued.
"""

from __future__ import annotations

import heapq
import logging
import queue
import threading
from typing import TYPE_CHECKING, List, Optional, Sequence, Set, Tuple

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.cache import ArrowFileBackend, CacheManager
from biopb_tensor_server.chunk import build_pyramid_plan

if TYPE_CHECKING:
    from biopb_tensor_server.config import PrecacheConfig, PyramidConfig
    from biopb_tensor_server.server import TensorFlightServer

logger = logging.getLogger(__name__)

# How often to re-check idle/stop while waiting for the server to quiesce.
_POLL_INTERVAL_SECONDS = 0.2


def _release_persistent_store(tensor_adapter) -> None:
    """Close any persistent file handle the adapter opened while warming.

    aicsimageio's tiff-backed adapters keep a persistent aszarr store open after
    a read, reclaimed only by an idle reaper (``BIOPB_TIFF_STORE_TTL``, default
    300 s). Precache sweeps the whole catalog back-to-back, so those stores never
    go idle and accumulate across every source — the memory exhaustion behind the
    precache OOM. We warm one source at a time, so releasing its store the instant
    we finish bounds the resident store count to one instead of N. Effectively a
    zero-TTL for precache-driven access. No-op for adapters with no persistent
    store (e.g. nd2); best-effort and never raises.
    """
    # Nothing open -> leave adapter state untouched (don't reset _persistent_
    # attempted, which non-tiff sources set to avoid re-probing a tiff store).
    if getattr(tensor_adapter, "_persistent_dask", None) is None:
        return
    close = getattr(tensor_adapter, "_close_persistent_store", None)
    if close is None:
        return
    lock = getattr(tensor_adapter, "_io_lock", None)
    try:
        if lock is not None:
            with lock:
                close()
        else:
            close()
    except Exception:  # pragma: no cover - teardown must never break precache
        logger.debug("precache: persistent-store release failed", exc_info=True)


class PrecacheWorker:
    """Daemon thread that warms the file cache for newly-added and existing
    sources."""

    def __init__(
        self,
        server: "TensorFlightServer",
        config: "PrecacheConfig",
        pyramid_config: Optional["PyramidConfig"] = None,
    ):
        self._server = server
        self._cfg = config
        # Pyramid level definition is shared with the server's advertised
        # TensorDescriptor.pyramid, so the warmed scale == the advertised scale.
        # Defaults to the canonical knobs when a caller omits it.
        if pyramid_config is None:
            from biopb_tensor_server.config import PyramidConfig

            pyramid_config = PyramidConfig()
        self._pyramid_cfg = pyramid_config
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._seen: Set[str] = set()
        self._seen_lock = threading.Lock()
        # Backlog tier: a newest-mtime-first heap of (-mtime, seq, source_id).
        self._backlog: List[Tuple[float, int, str]] = []
        self._backlog_ids: Set[str] = set()
        self._backlog_seq = 0
        self._backlog_lock = threading.Lock()
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

    # -- producer API ------------------------------------------------------

    def enqueue(self, source_id: str) -> None:
        """Queue a live (runtime) source for warming (non-blocking, deduped).

        Handed to ``SourceManager._on_source_committed``.
        """
        with self._seen_lock:
            if source_id in self._seen:
                return
            self._seen.add(source_id)
        self._queue.put(source_id)

    def seed_backlog(self, items: Sequence[Tuple[str, float]]) -> None:
        """Seed the secondary backlog with ``(source_id, mtime)`` pairs.

        Called once at startup with the existing local sources. Items already
        queued in the live tier or the backlog are skipped.
        """
        if not items:
            return
        with self._seen_lock:
            seen_snapshot = set(self._seen)
        added = 0
        with self._backlog_lock:
            for source_id, mtime in items:
                if source_id in self._backlog_ids or source_id in seen_snapshot:
                    continue
                self._backlog_seq += 1
                heapq.heappush(
                    self._backlog, (-mtime, self._backlog_seq, source_id)
                )
                self._backlog_ids.add(source_id)
                added += 1
        logger.info(
            "precache: seeded %d/%d existing sources into backlog",
            added,
            len(items),
        )

    # -- worker loop -------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            # 1. Live tier (primary): always drained first.
            try:
                source_id = self._queue.get_nowait()
            except queue.Empty:
                source_id = None
            if source_id is not None:
                self._process_live(source_id)
                continue

            # 2. Backlog tier (secondary): only on a file backend with headroom.
            if self._backlog_has_items():
                if not self._file_backend_active():
                    self._clear_backlog()
                    continue
                if not self._has_headroom():
                    # Cache is full; warming would evict live data. Nap and
                    # re-check (live eviction may free room later).
                    self._stop.wait(self._cfg.backlog_idle_recheck_seconds)
                    continue
                self._drain_one_backlog()
                continue

            # 3. Idle: block briefly for the next live addition.
            try:
                source_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_live(source_id)

    def _process_live(self, source_id: str) -> None:
        # Drop the dedup marker before processing: a commit that arrives while
        # we work should be allowed to re-queue a fresh pass.
        with self._seen_lock:
            self._seen.discard(source_id)
        try:
            self._process_source(source_id)
        except Exception:
            logger.exception("precache: failed for source %s", source_id)

    def _drain_one_backlog(self) -> None:
        entry = self._pop_backlog()
        if entry is None:
            return
        neg_mtime, source_id = entry
        try:
            preempted = self._process_source(source_id, backlog=True)
        except Exception:
            logger.exception("precache: backlog failed for source %s", source_id)
            preempted = False
        if preempted:
            # Live traffic or a full cache interrupted us; resume this source
            # (newest first) once conditions allow. Re-warm is cheap (hits).
            self._requeue_backlog(source_id, neg_mtime)

    # -- backlog bookkeeping -----------------------------------------------

    def _backlog_has_items(self) -> bool:
        with self._backlog_lock:
            return bool(self._backlog)

    def _pop_backlog(self) -> Optional[Tuple[float, str]]:
        with self._backlog_lock:
            if not self._backlog:
                return None
            neg_mtime, _seq, source_id = heapq.heappop(self._backlog)
            self._backlog_ids.discard(source_id)
            return neg_mtime, source_id

    def _requeue_backlog(self, source_id: str, neg_mtime: float) -> None:
        with self._backlog_lock:
            if source_id in self._backlog_ids:
                return
            self._backlog_seq += 1
            heapq.heappush(self._backlog, (neg_mtime, self._backlog_seq, source_id))
            self._backlog_ids.add(source_id)

    def _clear_backlog(self) -> None:
        with self._backlog_lock:
            self._backlog.clear()
            self._backlog_ids.clear()

    # -- gates -------------------------------------------------------------

    def _file_backend_active(self) -> bool:
        """True only when the persistent file cache is in use."""
        cache_manager = CacheManager.get_instance()
        return cache_manager is not None and isinstance(
            cache_manager.backend, ArrowFileBackend
        )

    def _has_headroom(self) -> bool:
        """True while the file cache is below the backlog high-water mark.

        Keeps the backlog tier from filling the cache to the brim and evicting
        genuinely-hot live data (the cache evicts globally on every write).
        """
        cache_manager = CacheManager.get_instance()
        if cache_manager is None:
            return False
        try:
            st = cache_manager.backend.stats()
        except Exception:
            return False
        if st.max_bytes <= 0:
            return False
        return st.total_bytes < st.max_bytes * self._cfg.backlog_high_water

    def _wait_until_idle(self) -> bool:
        """Block until the Flight server is idle. Return False if asked to stop."""
        debounce = self._cfg.idle_debounce_seconds
        while not self._stop.is_set():
            if self._server.flight_idle_for(debounce):
                return True
            self._stop.wait(_POLL_INTERVAL_SECONDS)
        return False

    # -- per-source warming ------------------------------------------------

    def _process_source(self, source_id: str, backlog: bool = False) -> bool:
        """Warm every tensor of a source. Return True if a backlog pass was
        preempted (and should be re-queued)."""
        # Runtime file-backend gate: the "only run if file-based caching"
        # condition, enforced regardless of config.
        if not self._file_backend_active():
            logger.debug("precache: file backend not active, skipping %s", source_id)
            return False
        cache_manager = CacheManager.get_instance()

        source_adapter = self._server._get_source_adapter(source_id)
        if source_adapter is None:
            return False
        # Skip formats that ship their own multi-resolution pyramid (e.g.
        # well-formed OME-Zarr): they already serve overviews cheaply from
        # native coarse levels, so precache gains little. The client's overview
        # request would also use a synthetic scale that doesn't reuse these
        # anyway, so warming them is wasted I/O.
        if source_adapter.has_native_pyramid():
            logger.debug(
                "precache: skipping well-formed multiscale source %s", source_id
            )
            return False
        try:
            descriptors = source_adapter.list_tensor_descriptors()
        except Exception:
            logger.exception(
                "precache: list_tensor_descriptors failed for %s", source_id
            )
            return False

        for td in descriptors:
            if self._stop.is_set():
                return False
            if self._process_tensor(
                source_adapter, td, cache_manager, backlog=backlog
            ):
                return True  # preempted mid-source
        return False

    def _process_tensor(
        self, source_adapter, td, cache_manager, backlog: bool = False
    ) -> bool:
        """Warm one tensor's coarsest level. Return True if preempted (backlog)."""
        # The client passes the descriptor's array_id verbatim as tensor_id
        # (TensorFlightClient), so the request we build mirrors get_flight_info.
        tensor_id = td.array_id
        try:
            tensor_adapter = source_adapter.get_tensor_adapter(tensor_id)
        except Exception:
            logger.exception("precache: get_tensor_adapter failed for %s", tensor_id)
            return False
        if tensor_adapter is None:
            return False
        try:
            base_desc = tensor_adapter.get_tensor_descriptor()
        except Exception:
            logger.exception(
                "precache: get_tensor_descriptor failed for %s", tensor_id
            )
            return False

        # Warm the coarsest level of the same plan the server advertises (a
        # non-native source: native ones are skipped above), so the warmed
        # chunk_ids are exactly what the client fetches on open.
        cfg = self._pyramid_cfg
        coarsest = build_pyramid_plan(
            list(base_desc.shape),
            list(base_desc.dim_labels),
            reduction_method=cfg.reduction_method,
            threshold=cfg.threshold,
            downscale_factor=cfg.downscale_factor,
            pixel_budget_cubic_root=cfg.pixel_budget_cubic_root,
        )[-1]

        # Build the request descriptor exactly as get_flight_info does, so the
        # read plan's scaled chunk_ids match what the client will fetch.
        request_desc = TensorDescriptor(
            array_id=tensor_id,
            dim_labels=base_desc.dim_labels,
            shape=base_desc.shape,
            chunk_shape=base_desc.chunk_shape,
            dtype=base_desc.dtype,
        )
        request_desc.scale_hint[:] = list(coarsest.scale_hint)
        request_desc.reduction_method = coarsest.reduction_method

        try:
            read_plan = tensor_adapter.get_read_plan(request_desc)
        except Exception:
            logger.exception("precache: get_read_plan failed for %s", tensor_id)
            return False

        endpoints = read_plan.chunk_endpoints
        warmed = 0
        # Only release a store precache itself opens. If another reader already
        # has this source's persistent store open (mid idle-TTL countdown), leave
        # it intact -- don't evict it or reset its TTL out from under that reader.
        store_preexisted = (
            getattr(tensor_adapter, "_persistent_dask", None) is not None
        )
        try:
            for ce in endpoints:
                if self._stop.is_set():
                    return False
                if backlog:
                    # Yield to live work the instant any arrives.
                    if not self._queue.empty():
                        return True
                    # Respect cache headroom: live traffic may have filled it.
                    if not self._has_headroom():
                        return True
                # Debounce + preempt between chunks: wait for the server to be
                # idle before warming each chunk.
                if not self._wait_until_idle():
                    return False
                try:
                    tensor_adapter.resolve_chunk_data(ce.chunk_id, cache_manager)
                    warmed += 1
                except Exception as e:
                    # One bad chunk shouldn't abort the whole tensor.
                    logger.debug(
                        "precache: chunk warm failed for %s: %s", tensor_id, e
                    )

            logger.info(
                "precache: warmed %d/%d chunks for %s at scale=%s%s",
                warmed,
                len(endpoints),
                tensor_id,
                list(coarsest.scale_hint),
                " (backlog)" if backlog else "",
            )
            return False
        finally:
            # Release the store precache opened for this source before moving to
            # the next, so stores can't accumulate across the sweep (the OOM).
            # Skip when the store pre-existed -- that one belongs to another
            # reader. Runs on every exit: completion, preemption, stop, or error.
            if not store_preexisted:
                _release_persistent_store(tensor_adapter)
