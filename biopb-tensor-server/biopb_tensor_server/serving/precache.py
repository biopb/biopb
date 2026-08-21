"""Background pre-cache worker for the tensor server.

This worker warms the file cache so the first view a scientist opens is already
warm instead of paying a cold decode+downsample on the critical path. It warms
the coarsest level of the same pyramid the server advertises on the tensor
descriptor (see ``chunk.build_pyramid_plan``), so the warmed scale always matches
what the client requests on open.

It serves three tiers, in strict priority order:

- **Demand tier (highest).** Triggered by a chunk a client actually read
  (``observe_read``, fed from the server's read paths): warms the rest of that
  tensor's level first, then the source's other tensors, all at the *client's*
  scale and reduction method. This is the only tier with evidence behind it --
  somebody is looking at this source right now -- and the only one whose target
  the server does not have to guess, since the observed read carries the scale.
  A client running its own pyramid policy is therefore warmed correctly without
  the server modelling that policy.
- **Live tier.** Sources added to the catalog *after* startup, fed by
  ``SourceManager``'s commit hook (``enqueue``). Always warmed.
- **Backlog tier (lowest, off by default).** Local sources already present at
  startup, seeded once via ``seed_backlog`` and ordered newest-mtime-first.
  Drained only when the live queue is empty, and bounded so it never evicts live
  data (see below). Disabled by default because it is the tier with no evidence
  behind it: it warms the entire catalog against the chance someone opens some
  of it, and when the footprint exceeds the cache the ``backlog_high_water``
  stop decides *which* sources stay warm by mtime rather than by value. The
  demand tier covers the same ground for sources anyone actually opens.

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
- **Only warms what scaling makes cheap.** A tensor whose coarsest advertised
  level is full resolution is skipped: warming it would cache the source 1:1 to
  save an open nothing. Without this the worker chases a footprint far larger
  than the cache, and the ``backlog_high_water`` stop then decides *which*
  sources stay warm by mtime order rather than by what warming is worth.
"""

from __future__ import annotations

import heapq
import logging
import queue
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Set, Tuple

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.cache import ArrowFileBackend, CacheManager
from biopb_tensor_server.core.chunk import (
    build_pyramid_plan,
    decode_reduction_method,
    decode_scale_info,
    is_scaled_chunk,
    routing_array_id,
)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import PrecacheConfig, PyramidConfig
    from biopb_tensor_server.serving.server import TensorFlightServer

logger = logging.getLogger(__name__)

# How often to re-check idle/stop while waiting for the server to quiesce.
_POLL_INTERVAL_SECONDS = 0.2

# How many (source, scale, method) levels the demand tier remembers warming.
# Sized to cover a browsing session's working set without pinning the memory of
# a warm that the cache has since evicted.
_DEMAND_MEMORY = 512


class PrecacheWorker:
    """Daemon thread that warms the file cache for newly-added and existing
    sources."""

    def __init__(
        self,
        server: TensorFlightServer,
        config: PrecacheConfig,
        pyramid_config: Optional[PyramidConfig] = None,
    ):
        self._server = server
        self._cfg = config
        # Pyramid level definition is shared with the server's advertised
        # TensorDescriptor.pyramid, so the warmed scale == the advertised scale.
        # Defaults to the canonical knobs when a caller omits it.
        if pyramid_config is None:
            from biopb_tensor_server.core.config import PyramidConfig

            pyramid_config = PyramidConfig()
        self._pyramid_cfg = pyramid_config
        self._queue: queue.Queue[str] = queue.Queue()
        self._seen: Set[str] = set()
        self._seen_lock = threading.Lock()
        # Demand tier: chunk_ids clients actually read. Bounded and lossy on
        # purpose -- an observation is a hint about what someone is looking at
        # *now*, so a backlog of stale ones is worth less than the newest few.
        # Nothing is decoded on the producer side; that work belongs off the
        # serving thread.
        self._demand: queue.Queue[bytes] = queue.Queue(
            maxsize=max(1, config.demand_queue_max)
        )
        # (source_id, scale, reduction_method) already warmed, most recent
        # last. Keyed on the triple rather than the chunk_id so re-reading any
        # chunk of a level already warmed for that source is free.
        #
        # Bounded and FIFO-evicted rather than a plain set: the cache evicts
        # globally, so a level warmed hours ago may be long gone. An unbounded
        # set would remember warming it and refuse to ever warm it again, which
        # turns a cache eviction into a permanent cold spot.
        self._demand_done: OrderedDict[Tuple[str, Tuple[int, ...], str], None] = (
            OrderedDict()
        )
        self._demand_lock = threading.Lock()
        # Backlog tier: a newest-mtime-first heap of (-mtime, seq, source_id).
        self._backlog: List[Tuple[float, int, str]] = []
        self._backlog_ids: Set[str] = set()
        self._backlog_seq = 0
        self._backlog_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Residency gate: re-checks at warm time whether a cloud-root source's
        # files are still resident, so we never recall content OneDrive (or
        # another Files-On-Demand provider) has re-dehydrated since registration
        # (#174). Wired from SourceManager.should_warm in cli.py; left None when
        # there is no manager (e.g. static-only deployments), in which case the
        # gate is a no-op and warming proceeds as before.
        self.should_warm: Optional[Callable[[str], bool]] = None

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

    def observe_read(self, chunk_id: bytes) -> None:
        """Record that a client actually read ``chunk_id``.

        Called from the server's read paths (``do_get`` and the localhost
        locate handoff), so it runs on a serving thread while a client waits.
        It therefore does no decoding, no locking beyond the queue's own, and
        never raises: everything the decision needs is recoverable from the
        chunk_id later, on the worker thread.

        Dropping on a full queue is the intended behaviour, not a failure. An
        observation is a guess about what someone is looking at right now; if
        they are moving faster than the worker warms, the newest guesses are
        worth more than a queue of stale ones.
        """
        if not self._cfg.demand_enabled or self._stop.is_set():
            return
        try:
            self._demand.put_nowait(chunk_id)
        except queue.Full:
            pass

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
                heapq.heappush(self._backlog, (-mtime, self._backlog_seq, source_id))
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
            # 0. Demand tier: a client is reading this source right now, so it
            #    outranks both speculative tiers.
            try:
                observed = self._demand.get_nowait()
            except queue.Empty:
                observed = None
            if observed is not None:
                self._process_demand(observed)
                continue

            # 1. Live tier: always drained before the backlog.
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

    def _process_demand(self, chunk_id: bytes) -> None:
        """Warm the siblings of a tensor a client just read, at its scale.

        Only *scaled* reads qualify. A full-resolution read is the computation
        pattern (a dask graph walking the whole array once); warming for it
        would cache bytes nobody re-reads and charge a cache write to a workflow
        that wants throughput. A scaled read is the browsing pattern, and it is
        the one whose next step is predictable.

        The observed tensor is warmed too, ahead of its siblings, and that is
        deliberate: a read of one plane only populates the others when the
        adapter's layout forces it to. ND2 interleaves channels, so decoding one
        decodes all and the rest of the level is already resident; a planar
        layout (separate channel planes / a chunked C axis) decodes exactly what
        was asked for and leaves every other plane cold. Warming the observed
        tensor covers the second case and costs cache hits in the first.
        """
        try:
            if not is_scaled_chunk(chunk_id):
                return
            scale = tuple(int(v) for v in decode_scale_info(chunk_id))
            method = decode_reduction_method(chunk_id)
            array_id = routing_array_id(chunk_id)
        except Exception:
            # A malformed or unfamiliar chunk_id is a hint we simply drop; the
            # read it came from already succeeded.
            logger.debug("precache: undecodable observed chunk_id", exc_info=True)
            return
        if all(v == 1 for v in scale):
            return

        source_id = array_id.split("/")[0]
        key = (source_id, scale, method)
        with self._demand_lock:
            if key in self._demand_done:
                return
            self._demand_done[key] = None
            while len(self._demand_done) > _DEMAND_MEMORY:
                self._demand_done.popitem(last=False)

        try:
            self._process_source(
                source_id,
                scale_hint=list(scale),
                reduction_method=method,
                first_array_id=array_id,
            )
        except Exception:
            logger.exception("precache: demand warm failed for %s", source_id)

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

    def _process_source(
        self,
        source_id: str,
        backlog: bool = False,
        scale_hint: Optional[List[int]] = None,
        reduction_method: Optional[str] = None,
        first_array_id: Optional[str] = None,
    ) -> bool:
        """Warm every tensor of a source. Return True if a backlog pass was
        preempted (and should be re-queued).

        ``scale_hint``/``reduction_method`` override the server's own pyramid
        plan -- the demand tier passes the level a client was observed reading.
        ``first_array_id`` moves one tensor to the front: the one the client is
        actually looking at, whose unread planes matter more than any sibling."""
        # Runtime file-backend gate: the "only run if file-based caching"
        # condition, enforced regardless of config.
        if not self._file_backend_active():
            logger.debug("precache: file backend not active, skipping %s", source_id)
            return False
        # Residency gate (#174): under a cloud root, skip a source whose member
        # files have been re-dehydrated since registration. Reading them would
        # recall content the cloud=true policy is meant to keep offline. Coarse
        # (whole-source) by design, matching the registration-time check; a
        # partially-evicted cloud source is skipped rather than partly recalled.
        if self.should_warm is not None and not self.should_warm(source_id):
            logger.debug(
                "precache: source %s not resident (cloud), skipping warm", source_id
            )
            return False
        cache_manager = CacheManager.get_instance()

        source_adapter = self._server.sources.get(source_id)
        if source_adapter is None:
            return False

        # Skip non-local (remote) sources entirely (biopb/biopb#299). Warming a
        # remote-tensor proxy source would speculatively pull every chunk across
        # the network from the upstream at startup -- costly I/O of questionable
        # value (the real read path caches on demand, and the upstream caches
        # too), and it inflates the local file cache (feeding the slow-restart
        # recovery, #300). It is also unsound today: the proxy does not implement
        # has_native_pyramid(), so a pyramidal upstream would be warmed at a
        # computed coarse level the upstream already serves natively -- caching
        # chunks the native-pyramid skip below is meant to avoid.
        from biopb_tensor_server.core.remote import is_remote_url

        if is_remote_url(source_adapter.source_url or ""):
            logger.debug(
                "precache: skipping non-local source %s (warmed on demand)", source_id
            )
            return False

        try:
            descriptors = source_adapter.list_tensor_descriptors()
        except Exception:
            logger.exception(
                "precache: list_tensor_descriptors failed for %s", source_id
            )
            return False

        if first_array_id is not None:
            descriptors = sorted(
                descriptors, key=lambda d: d.array_id != first_array_id
            )

        for td in descriptors:
            if self._stop.is_set():
                return False
            if self._process_tensor(
                source_adapter,
                td,
                cache_manager,
                backlog=backlog,
                scale_hint=scale_hint,
                reduction_method=reduction_method,
            ):
                return True  # preempted mid-source
        return False

    def _process_tensor(
        self,
        source_adapter,
        td,
        cache_manager,
        backlog: bool = False,
        scale_hint: Optional[List[int]] = None,
        reduction_method: Optional[str] = None,
    ) -> bool:
        """Warm one tensor level. Return True if preempted (backlog).

        Defaults to the coarsest level of the server-advertised plan; the demand
        tier passes an observed ``scale_hint``/``reduction_method`` instead."""
        # The client passes the descriptor's array_id verbatim as tensor_id
        # (TensorFlightClient), so the request we build mirrors get_flight_info.
        tensor_id = td.array_id
        try:
            tensor_adapter = source_adapter.get_tensor_adapter(tensor_id)
        except Exception:
            logger.exception("precache: get_tensor_adapter failed for %s", tensor_id)
            return False
        # Skip a tensor that ships its own multi-resolution pyramid (e.g. a
        # well-formed OME-Zarr image, or a pyramidal qptiff/ndtiff series): it
        # already serves overviews cheaply from its native coarse levels, so
        # warming is wasted I/O. Per-tensor because pyramid support can vary
        # between tensors of one source (a pyramidal main series alongside flat
        # label/macro series).
        try:
            if tensor_adapter.has_native_pyramid():
                logger.debug(
                    "precache: skipping tensor %s (serves overviews natively)",
                    tensor_id,
                )
                return False
        except Exception:
            logger.exception("precache: has_native_pyramid failed for %s", tensor_id)
        try:
            base_desc = tensor_adapter.get_tensor_descriptor()
        except Exception:
            logger.exception("precache: get_tensor_descriptor failed for %s", tensor_id)
            return False

        # An observed level wins over the advertised plan: it is what a client
        # demonstrably reads, whereas the plan is what the server guesses it
        # should. A client with its own pyramid policy only ever agrees with the
        # former. Rank mismatches are dropped rather than coerced -- an override
        # from a different tensor would warm chunk_ids nobody asks for.
        if scale_hint is not None:
            if len(scale_hint) != len(base_desc.shape):
                logger.debug(
                    "precache: observed scale rank %d != tensor rank %d for %s",
                    len(scale_hint),
                    len(base_desc.shape),
                    tensor_id,
                )
                return False
            level_scale = list(scale_hint)
            level_method = reduction_method or self._pyramid_cfg.reduction_method
        else:
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
            level_scale = list(coarsest.scale_hint)
            level_method = coarsest.reduction_method

        # Nothing to precompute when the coarsest level is full resolution:
        # warming caches the source 1:1 and saves an open nothing.
        #
        # Test the planner's scale_hint rather than a pixel threshold of our
        # own -- it is already the answer to "is the spatial extent worth a
        # level?", and tying the gate to it keeps precache from warming
        # chunk_ids no client requests. Biggest effect is on long timelapses:
        # the plan scores Lx*Ly*Lz only, so T is never scaled and a many-frame
        # series is both the costliest warm and the one warming cannot help.
        if all(int(v) == 1 for v in level_scale):
            logger.debug(
                "precache: skipping tensor %s (level is full resolution)",
                tensor_id,
            )
            return False

        # Build the request descriptor exactly as get_flight_info does, so the
        # read plan's scaled chunk_ids match what the client will fetch.
        request_desc = TensorDescriptor(
            array_id=tensor_id,
            dim_labels=base_desc.dim_labels,
            shape=base_desc.shape,
            chunk_shape=base_desc.chunk_shape,
            dtype=base_desc.dtype,
        )
        request_desc.scale_hint[:] = level_scale
        request_desc.reduction_method = level_method

        try:
            # Same budget as the serve path: the grid decides the chunk_ids,
            # so a precache planning on a different budget would warm chunks
            # no read ever asks for.
            read_plan = tensor_adapter.get_read_plan(
                request_desc,
                max_read_block_bytes=self._pyramid_cfg.max_read_block_mb * 1024 * 1024,
            )
        except Exception:
            logger.exception("precache: get_read_plan failed for %s", tensor_id)
            return False

        endpoints = read_plan.chunk_endpoints
        warmed = 0
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
            elif scale_hint is not None and not self._has_headroom():
                # Demand tier: better evidence than the backlog has, but still
                # speculative, so it observes the same ceiling. It does *not*
                # yield to the live queue -- a source someone is reading now
                # outranks one that was merely added.
                logger.debug("precache: demand warm stopped, cache at high water")
                return False
            # Debounce + preempt between chunks: wait for the server to be idle
            # before warming each chunk.
            if not self._wait_until_idle():
                return False
            try:
                tensor_adapter.resolve_chunk_data(ce.chunk_id, cache_manager)
                warmed += 1
            except Exception as e:
                # One bad chunk shouldn't abort the whole tensor.
                logger.debug("precache: chunk warm failed for %s: %s", tensor_id, e)

        logger.info(
            "precache: warmed %d/%d chunks for %s at scale=%s%s",
            warmed,
            len(endpoints),
            tensor_id,
            level_scale,
            " (observed)"
            if scale_hint is not None
            else (" (backlog)" if backlog else ""),
        )
        return False
