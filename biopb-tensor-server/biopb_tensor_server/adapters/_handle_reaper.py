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

# Default seconds an idle persistent handle is kept warm before it is reaped.
# The effective value is set from ``ServerConfig.handle_reaper_ttl`` at server
# startup (see ``set_handle_reaper_ttl`` / ``cli.serve``); this is the fallback
# for any process that never loads a ServerConfig (ad-hoc tooling, tests).
DEFAULT_HANDLE_REAPER_TTL = 150.0

# Every constructed reaper, so one config knob can retune them all at startup.
# Weak, so a reaper built in a test is not pinned for the process lifetime.
_configured_reapers: "weakref.WeakSet[IdleHandleReaper]" = weakref.WeakSet()


def set_handle_reaper_ttl(seconds: float) -> None:
    """Set the idle TTL on every reaper pool (OME-TIFF stores, NDTiff datasets).

    Process-wide policy (one ServerConfig per process), mirroring the other
    module-level startup toggles like ``set_claim_generic_images``. Called from
    ``cli.serve`` before any source registers, so a ``<= 0`` value cleanly
    disables reaping (``register`` becomes a no-op and no thread ever starts).
    """
    for reaper in list(_configured_reapers):
        reaper.set_ttl(seconds)


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

    def __init__(self, ttl_seconds: float, thread_name: str) -> None:
        self._ttl = ttl_seconds
        self._thread_name = thread_name
        self._adapters: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._started = False
        _configured_reapers.add(self)

    @property
    def enabled(self) -> bool:
        """Whether reaping is active (``ttl_seconds > 0``)."""
        return self._ttl > 0

    def set_ttl(self, seconds: float) -> None:
        """Retune the idle TTL. Read live by ``register``/``_sweep``, so a value
        applied at startup (before the thread lazily starts) fully takes effect;
        ``<= 0`` disables the pool."""
        self._ttl = float(seconds)

    def register(self, adapter: ReapableHandle) -> None:
        """Track an adapter that just opened its handle; start the thread if needed.

        Idempotent (a ``WeakSet``), so a reopen after a reap re-registers cleanly.
        A no-op when disabled, so a caller need not branch on the TTL.
        """
        if self._ttl <= 0:
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

    def discard(self, adapter: ReapableHandle) -> None:
        """Stop tracking an adapter whose handle is now closed."""
        with self._lock:
            self._adapters.discard(adapter)

    def _loop(self) -> None:
        """Sweep at a quarter of the TTL (clamped to [1s, 30s]), forever."""
        interval = max(1.0, min(self._ttl / 4.0, 30.0))
        while True:
            time.sleep(interval)
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
            if now - adapter._persistent_last_access <= self._ttl:
                continue
            # Only close when idle AND no lock-free read is mid-flight. A read that
            # holds _io_lock for its duration makes this acquire fail (correct: it
            # is not idle); a lock-free read releases the lock but bumps
            # _active_reads, which the second check honours.
            if adapter._io_lock.acquire(blocking=False):
                try:
                    idle = time.monotonic() - adapter._persistent_last_access
                    if idle > self._ttl and adapter._active_reads == 0:
                        adapter._release_persistent_handle()
                finally:
                    adapter._io_lock.release()
