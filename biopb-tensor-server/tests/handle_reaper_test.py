"""Unit tests for the shared idle-handle reaper (biopb/biopb#71).

The reaper closes a persistent handle once it has been idle past a TTL, without
ever closing one mid-read. These drive :meth:`IdleHandleReaper._sweep` directly
(no sleep, no thread) so the idle/in-flight logic is deterministic; the
lazy-thread-start path is covered separately.
"""

import threading
import time
import weakref

from biopb_tensor_server.adapters._handle_reaper import (
    IdleHandleReaper,
    ReapableHandle,
    set_handle_reaper_ttl,
)


class _FakeReapable:
    """Minimal object satisfying the ReapableHandle contract."""

    def __init__(self):
        self._io_lock = threading.Lock()
        self._active_reads = 0
        self._persistent_last_access = time.monotonic()
        self.released = 0

    def _release_persistent_handle(self) -> None:
        self.released += 1


def _reaper_with(adapter, ttl=10.0):
    """A reaper tracking `adapter` WITHOUT starting its thread (add directly)."""
    reaper = IdleHandleReaper(ttl_seconds=ttl, thread_name="test-reaper")
    reaper._adapters.add(adapter)
    return reaper


def test_fake_satisfies_the_protocol():
    assert isinstance(_FakeReapable(), ReapableHandle)


def test_sweep_releases_an_idle_handle():
    a = _FakeReapable()
    a._persistent_last_access = time.monotonic() - 100  # idle >> ttl
    _reaper_with(a, ttl=10.0)._sweep()
    assert a.released == 1


def test_sweep_keeps_a_recently_used_handle():
    a = _FakeReapable()
    a._persistent_last_access = time.monotonic()  # just read
    _reaper_with(a, ttl=10.0)._sweep()
    assert a.released == 0


def test_sweep_skips_a_lockfree_read_in_flight():
    # Idle by the clock, but a lock-free read is registered -- must not close.
    a = _FakeReapable()
    a._persistent_last_access = time.monotonic() - 100
    a._active_reads = 1
    _reaper_with(a, ttl=10.0)._sweep()
    assert a.released == 0


def test_sweep_skips_when_io_lock_is_held():
    # A read holding _io_lock across its decode makes the non-blocking acquire
    # fail -- the reaper backs off rather than closing under the read.
    a = _FakeReapable()
    a._persistent_last_access = time.monotonic() - 100
    reaper = _reaper_with(a, ttl=10.0)
    with a._io_lock:
        reaper._sweep()
    assert a.released == 0


def test_disabled_reaper_never_tracks_or_starts():
    # ttl <= 0 opts the pool out entirely: register is a no-op, no thread.
    reaper = IdleHandleReaper(ttl_seconds=0.0, thread_name="off")
    assert reaper.enabled is False
    reaper.register(_FakeReapable())
    assert reaper._started is False
    assert len(list(reaper._adapters)) == 0


def test_set_ttl_updates_ttl_and_enabled():
    r = IdleHandleReaper(ttl_seconds=150.0, thread_name="tune")
    assert r.enabled is True
    r.set_ttl(0)  # <= 0 disables the pool
    assert r.enabled is False
    r.set_ttl(60)
    assert r._ttl == 60.0
    assert r.enabled is True


def test_set_handle_reaper_ttl_retunes_every_pool(monkeypatch):
    # The startup hook drives every constructed reaper from one config knob.
    # Isolate to a fresh registry so the real OME-TIFF/NDTiff pools are untouched.
    import biopb_tensor_server.adapters._handle_reaper as hr

    monkeypatch.setattr(hr, "_configured_reapers", weakref.WeakSet())
    r1 = IdleHandleReaper(ttl_seconds=150.0, thread_name="pool-1")
    r2 = IdleHandleReaper(ttl_seconds=150.0, thread_name="pool-2")

    set_handle_reaper_ttl(30.0)
    assert r1._ttl == 30.0 and r2._ttl == 30.0

    set_handle_reaper_ttl(0.0)  # disable everywhere
    assert r1.enabled is False and r2.enabled is False


def test_register_starts_one_daemon_thread_then_discard():
    reaper = IdleHandleReaper(ttl_seconds=300.0, thread_name="lazy-start")
    assert reaper._started is False
    a = _FakeReapable()
    reaper.register(a)
    assert reaper._started is True
    # A second register does not start a second thread.
    live = [t for t in threading.enumerate() if t.name == "lazy-start"]
    assert len(live) == 1
    assert live[0].daemon is True
    reaper.register(_FakeReapable())
    assert len([t for t in threading.enumerate() if t.name == "lazy-start"]) == 1
    reaper.discard(a)
    assert a not in list(reaper._adapters)
