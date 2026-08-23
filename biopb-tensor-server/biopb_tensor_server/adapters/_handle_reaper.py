"""Shared idle-handle reaper for adapters that keep a persistent OS handle.

Most file adapters do **not** need this. ``hdf5`` and ``mrc`` reopen their file
per read (biopb/biopb#71): their open is O(1), so the reopen is unmeasurable
against a 64 MB chunk read and it removes the steady-state pin entirely -- no TTL
to tune, no handle held between reads. That is strictly better whenever it is
affordable, so it is the default.

This reaper is the **opt-in** alternative for the adapters where it is *not*
affordable -- those whose open cost scales with something:

- ``ome-tiff`` -- open is linear in IFD count and unbounded (~615 ms extrapolated
  for a 50k-page whole-slide file), so a reopen-per-read would be a >150%
  regression on exactly the large files the format exists for.
- ``ndtiff`` -- the reopen *unit* is the whole acquisition: ``NDTiffDataset``
  eagerly opens every ``NDTiffStack_*.tif``, so a reopen-per-read would open
  thousands of files to serve one plane.

For those, the handle stays warm between reads and a background reaper closes it
once it has been idle longer than a TTL -- bounding the steady-state pin (the
Windows-undeletable / disk-not-reclaimed effects) rather than eliminating it, at
the cost of one reopen on the next read after a lull.

A pool is bounded on **two** axes, and both are properties of the pool rather
than of the process:

- **How long** one idle handle is kept. This tracks what reopening *that* format
  costs, and the formats differ by four orders of magnitude: an OME-TIFF store
  is ~615 ms to reopen, an ND2 reader ~0.1 ms. A pool whose reopen is nearly
  free wants a short TTL -- what it holds is not a saved open but a warm page
  table (``nd2.read_frame`` hands out a view onto the reader's mmap), and that
  is worth only as long as the read that built it.
- **How many** are kept at once. Without a cap a pool grows with the catalog:
  one warm handle per source touched, never released until each goes idle. The
  cap is enforced LRU on register, so crossing it costs the coldest handle
  rather than one sweep interval of unbounded growth.

``ServerConfig.handle_reaper_ttl`` is a **ceiling** over the first axis, not an
assignment: an operator knows their fd and memory budget, not the reopen cost of
each format, so the knob tightens every pool and never loosens one.

The contract an adapter opts into (see :class:`ReapableHandle`): expose an
``_io_lock`` the reaper can take to fence a close against an in-flight read, an
``_active_reads`` counter for read paths that decode *without* the lock (0 when
reads stay fully under ``_io_lock``), a monotonic ``_persistent_last_access``
stamp, and a ``_release_persistent_handle`` that drops the handle and permits a
later reopen. Register on open, discard on release.
"""

import logging
import threading
import time
import weakref
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Seconds an idle handle is kept warm in a pool whose reopen is *expensive* --
# OME-TIFF stores, NDTiff acquisitions, CZI readers, where the open parses a
# directory whose size tracks the file's. A pool with a cheap reopen passes its
# own, shorter value instead (see the module docstring).
#
# Capped at runtime by ``ServerConfig.handle_reaper_ttl`` (see
# ``set_handle_reaper_ttl`` / ``cli.serve``); this is what applies in a process
# that never loads a ServerConfig (ad-hoc tooling, tests).
DEFAULT_HANDLE_REAPER_TTL = 150.0

# No ceiling until a ServerConfig sets one.
_NO_CEILING = float("inf")

# Every constructed reaper, so one config knob can retune them all at startup.
# Weak, so a reaper built in a test is not pinned for the process lifetime.
_configured_reapers: "weakref.WeakSet[IdleHandleReaper]" = weakref.WeakSet()


def set_handle_reaper_ttl(seconds: float) -> None:
    """Cap how long any pool keeps an idle handle warm.

    A ceiling rather than an assignment. Each pool's own TTL encodes what
    reopening that format costs, which an operator has no way to know and no
    reason to tune; what an operator does know is their fd and memory budget,
    and that is an upper bound on all of them. So this tightens every pool and
    never loosens one -- a pool already shorter than ``seconds`` keeps its value.

    Process-wide policy (one ServerConfig per process), mirroring the other
    module-level startup toggles like ``set_claim_generic_images``. Called from
    ``cli.serve`` before any source registers, so a ``<= 0`` value cleanly
    disables reaping (``register`` becomes a no-op and no thread ever starts).
    """
    for reaper in list(_configured_reapers):
        reaper.set_ceiling(seconds)


@runtime_checkable
class ReapableHandle(Protocol):
    """What the reaper needs from an adapter holding a persistent handle."""

    #: Non-reentrant lock fencing handle open/close against reads.
    _io_lock: threading.Lock
    #: Reads in flight that decode WITHOUT ``_io_lock`` held (0 if reads keep the
    #: lock for their whole duration). Non-zero blocks a reap.
    _active_reads: int
    #: ``time.monotonic()`` of the last read; drives the idle test.
    _persistent_last_access: float

    def _release_persistent_handle(self) -> None:
        """Close the handle and permit a later reopen. Called under ``_io_lock``."""
        ...


class IdleHandleReaper:
    """Closes persistent handles idle longer than a TTL, on one daemon thread.

    One instance per handle pool (e.g. one for OME-TIFF stores, one for NDTiff
    datasets), each with its own TTL and thread name. The thread starts lazily on
    the first :meth:`register` and only when ``ttl_seconds > 0``, so a process
    that opts out (or registers nothing) never spawns it.

    Adapters are held weakly: a dropped adapter falls out of the sweep on its own,
    and the GC finalizer (not the reaper) releases its handle.
    """

    def __init__(
        self,
        ttl_seconds: float,
        thread_name: str,
        *,
        max_handles: int,
    ) -> None:
        """
        Args:
            ttl_seconds: This pool's own idle TTL, sized from what reopening
                its format costs. Capped at runtime by
                :func:`set_handle_reaper_ttl`.
            thread_name: Name of this pool's sweep thread.
            max_handles: Most handles this pool keeps warm at once. Keyword-only
                and required: an unbounded pool grows with the catalog, and the
                right bound depends on what one handle of *this* kind pins --
                one NDTiff dataset holds thousands of open stack files where one
                OME-TIFF store holds a parsed IFD table, so there is no
                defensible shared default to inherit by accident.
        """
        self._pool_ttl = float(ttl_seconds)
        self._ceiling = _NO_CEILING
        self._max_handles = int(max_handles)
        self._thread_name = thread_name
        self._adapters: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._started = False
        _configured_reapers.add(self)

    @property
    def ttl(self) -> float:
        """Effective TTL: this pool's own, capped by the process-wide ceiling."""
        return min(self._pool_ttl, self._ceiling)

    @property
    def enabled(self) -> bool:
        """Whether reaping is active (effective TTL > 0)."""
        return self.ttl > 0

    def set_ttl(self, seconds: float) -> None:
        """Retune this pool's own TTL. Read live by ``register``/``_sweep``, so a
        value applied at startup (before the thread lazily starts) fully takes
        effect; ``<= 0`` disables the pool."""
        self._pool_ttl = float(seconds)

    def set_ceiling(self, seconds: float) -> None:
        """Cap this pool's TTL from process-wide config. See
        :func:`set_handle_reaper_ttl`."""
        self._ceiling = float(seconds)

    def register(self, adapter: ReapableHandle) -> None:
        """Track an adapter that just opened its handle; start the thread if needed.

        Idempotent (a ``WeakSet``), so a reopen after a reap re-registers cleanly.
        A no-op when disabled, so a caller need not branch on the TTL.

        Enforces the pool's cap here rather than leaving it to the sweep: a burst
        that opens handles faster than the sweep interval would otherwise pin all
        of them until the first sweep, which is exactly the case the cap exists
        for.
        """
        if self.ttl <= 0:
            return
        with self._lock:
            self._adapters.add(adapter)
            if not self._started:
                self._started = True
                threading.Thread(
                    target=self._loop,
                    name=self._thread_name,
                    daemon=True,
                ).start()
        self._close_over_cap()

    def _close_over_cap(self) -> None:
        """Close least-recently-used handles until the pool is back at its cap.

        Soft by construction: a handle whose read is in flight is skipped, never
        waited for. Blocking a live read to enforce a bound on *idle* pins would
        be the worse trade, and the next register tries again -- so the cap is a
        ceiling on what sits idle, not an invariant on the instantaneous count.

        Never called with ``_lock`` held: ``_release_persistent_handle`` calls
        back into :meth:`discard`, which takes it.
        """
        with self._lock:
            tracked = list(self._adapters)
        if len(tracked) <= self._max_handles:
            return
        # Oldest first. The adapter that just registered stamped itself on open,
        # so it sorts last and is never the one evicted by its own register.
        tracked.sort(key=lambda adapter: adapter._persistent_last_access)
        for adapter in tracked[: len(tracked) - self._max_handles]:
            if not adapter._io_lock.acquire(blocking=False):
                continue
            try:
                if adapter._active_reads == 0:
                    adapter._release_persistent_handle()
            finally:
                adapter._io_lock.release()

    def discard(self, adapter: ReapableHandle) -> None:
        """Stop tracking an adapter whose handle is now closed."""
        with self._lock:
            self._adapters.discard(adapter)

    def _loop(self) -> None:
        """Sweep at a quarter of the TTL (clamped to [1s, 30s]), forever.

        The interval is recomputed each pass so a retune -- or a ceiling applied
        after the thread started -- takes effect without a restart.
        """
        while True:
            time.sleep(max(1.0, min(self.ttl / 4.0, 30.0)))
            try:
                self._sweep()
            except Exception:  # pragma: no cover - the reaper must never die
                logger.debug("%s sweep failed", self._thread_name, exc_info=True)

    def _sweep(self) -> None:
        """Close every handle idle longer than the TTL. Never blocks a read.

        Split out from :meth:`_loop` so it can be driven deterministically in a
        test without the sleep/thread.
        """
        with self._lock:
            adapters = list(self._adapters)
        now = time.monotonic()
        for adapter in adapters:
            # Cheap idle test first -- both attributes are set before an adapter
            # ever registers, so read them bare (a missing one should fail loudly,
            # not default the reaper into a silent no-op).
            ttl = self.ttl
            if now - adapter._persistent_last_access <= ttl:
                continue
            # Only close when idle AND no lock-free read is mid-flight. A read that
            # holds _io_lock for its duration makes this acquire fail (correct: it
            # is not idle); a lock-free read releases the lock but bumps
            # _active_reads, which the second check honours.
            if adapter._io_lock.acquire(blocking=False):
                try:
                    idle = time.monotonic() - adapter._persistent_last_access
                    if idle > ttl and adapter._active_reads == 0:
                        adapter._release_persistent_handle()
                finally:
                    adapter._io_lock.release()
