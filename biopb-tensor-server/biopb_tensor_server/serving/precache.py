"""Background cache warmer for the tensor server.

Warms the chunks around what a client has *actually read*, so that moving
within a source -- to another plane, or to another tensor of the same file --
does not pay a cold decode. It is demand-driven by construction: an observed
read names the tensor, the scale and the reduction method, so the worker never
has to guess a level.

That is the whole design constraint, and it is why the two speculative tiers
this worker used to run are gone (biopb/biopb#89). Warming a level the server
picked only pays off for a client that happens to want that level, and the
server has no way to know which one that is: a computed pyramid is arithmetic
over the tensor's shape, so it encodes a policy rather than knowledge. Two
clients here already disagree -- the napari path follows the server's 4x plan,
the Viv-based viewer needs strict 2x levels and stops once a plane fits one
tile -- and the guess was wrong for whichever one it was not written for.

Design constraints (all best-effort, never fatal to the server):

- **File backend only.** Inert unless the cache is the persistent
  ``ArrowFileBackend``.
- **Scaled reads only.** A full-resolution read is the computation pattern (a
  dask graph walking the array once); warming for it would charge a cache write
  to a workflow with no re-read to pay it back.
- **Stays out of the way.** Before each chunk it waits until the Flight server
  has been idle for ``idle_debounce_seconds`` (no in-flight ``do_get``), and it
  re-checks between chunks so live traffic preempts it at chunk granularity. On
  the locked adapters its reads also serialize behind live reads through the
  per-source ``_io_lock``, so it never races a non-thread-safe reader.
- **Never evicts live data.** The file cache evicts globally on every write, so
  warming stops above ``high_water`` of the cache's ``max_bytes``.
- **Skips native pyramids.** A tensor that ships its own coarse levels serves
  overviews cheaply already.

What is deliberately *not* covered: the first read of a source nobody has
touched. Nothing predicts it, so it stays cold -- once per source, since the
first read then warms that source's other tensors.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.cache import ArrowFileBackend, CacheManager
from biopb_tensor_server.core.chunk import (
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
        # Demand tier: chunk_ids clients actually read. Bounded and lossy on
        # purpose -- an observation is a hint about what someone is looking at
        # *now*, so a backlog of stale ones is worth less than the newest few.
        # When it overflows the *oldest* entry goes, never the arriving one.
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

    def observe_read(self, chunk_id: bytes) -> None:
        """Record that a client actually read ``chunk_id``.

        Called from the server's read paths (``do_get`` and the localhost
        locate handoff), so it runs on a serving thread while a client waits.
        It therefore does no decoding, no locking beyond the queue's own, and
        never raises: everything the decision needs is recoverable from the
        chunk_id later, on the worker thread.

        Losing observations on a full queue is the intended behaviour, not a
        failure -- but which one is lost matters. A full queue means the client
        is moving faster than the worker warms, so what the queue holds is the
        *stale* guesses and what is arriving is where the client is now.
        Evicting the oldest keeps the tier tracking the client instead of
        pinning it to wherever they were when the worker fell behind.
        """
        if not self._cfg.demand_enabled or self._stop.is_set():
            return
        try:
            self._demand.put_nowait(chunk_id)
            return
        except queue.Full:
            pass
        # One eviction, one retry, then give up: the producer owns a serving
        # thread and must not spin. A racing consumer or producer can take the
        # slot back in between, which only costs this one hint -- and the queue
        # stays bounded either way.
        try:
            self._demand.get_nowait()
        except queue.Empty:
            pass
        try:
            self._demand.put_nowait(chunk_id)
        except queue.Full:
            pass

    # -- worker loop -------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                observed = self._demand.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_demand(observed)

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

    # -- gates -------------------------------------------------------------

    def _file_backend_active(self) -> bool:
        """True only when the persistent file cache is in use."""
        cache_manager = CacheManager.get_instance()
        return cache_manager is not None and isinstance(
            cache_manager.backend, ArrowFileBackend
        )

    def _has_headroom(self) -> bool:
        """True while the file cache is below the high-water mark.

        Keeps speculative warming from filling the cache to the brim and
        evicting genuinely-hot live data (the cache evicts globally on every
        write) -- including the very chunks the triggering read just cached.
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
        return st.total_bytes < st.max_bytes * self._cfg.high_water

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
        scale_hint: List[int],
        reduction_method: str,
        first_array_id: Optional[str] = None,
    ) -> None:
        """Warm every tensor of a source at an observed level.

        ``scale_hint``/``reduction_method`` are the level a client was seen
        reading -- there is no server-chosen fallback, which is the point.
        ``first_array_id`` moves one tensor to the front: the one the client is
        actually looking at, whose unread planes matter more than any sibling."""
        # Runtime file-backend gate: the "only run if file-based caching"
        # condition, enforced regardless of config.
        if not self._file_backend_active():
            logger.debug("precache: file backend not active, skipping %s", source_id)
            return
        # Residency gate (#174): under a cloud root, skip a source whose member
        # files have been re-dehydrated since registration. Reading them would
        # recall content the cloud=true policy is meant to keep offline. Coarse
        # (whole-source) by design, matching the registration-time check; a
        # partially-evicted cloud source is skipped rather than partly recalled.
        if self.should_warm is not None and not self.should_warm(source_id):
            logger.debug(
                "precache: source %s not resident (cloud), skipping warm", source_id
            )
            return
        cache_manager = CacheManager.get_instance()

        source_adapter = self._server.sources.get(source_id)
        if source_adapter is None:
            return

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
                "precache: skipping non-local source %s (read directly)", source_id
            )
            return

        try:
            descriptors = source_adapter.list_tensor_descriptors()
        except Exception:
            logger.exception(
                "precache: list_tensor_descriptors failed for %s", source_id
            )
            return

        if first_array_id is not None:
            descriptors = sorted(
                descriptors, key=lambda d: d.array_id != first_array_id
            )

        for td in descriptors:
            if self._stop.is_set():
                return
            if self._process_tensor(
                source_adapter,
                td,
                cache_manager,
                scale_hint=scale_hint,
                reduction_method=reduction_method,
            ):
                return  # cache filled up; the rest can warm on their own reads

    def _process_tensor(
        self,
        source_adapter,
        td,
        cache_manager,
        scale_hint: List[int],
        reduction_method: str,
    ) -> bool:
        """Warm one tensor at an observed level. True if it stopped early."""
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

        # A sibling of a different rank cannot use this level: warming it would
        # produce chunk_ids nobody asks for. Drop rather than coerce.
        if len(scale_hint) != len(base_desc.shape):
            logger.debug(
                "precache: observed scale rank %d != tensor rank %d for %s",
                len(scale_hint),
                len(base_desc.shape),
                tensor_id,
            )
            return False
        level_scale = list(scale_hint)
        level_method = reduction_method

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
            if not self._has_headroom():
                # Speculative work must never evict what live reads just put in.
                logger.debug("precache: warm stopped, cache at high water")
                return True
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
            "precache: warmed %d/%d chunks for %s at scale=%s",
            warmed,
            len(endpoints),
            tensor_id,
            level_scale,
        )
        return False
