"""Committing a cached chunk from memory while its write lands in background.

A cold read currently waits for its own cache write. Deferring it is safe for a
*cache* -- the batch is correct in memory, and a lost write costs a re-read from
the backend -- and unsafe wherever the cache is the only copy of the data. These
pin that line, and the two failure modes that make the difference invisible:
an unbounded backlog, and a write that fails after the caller was released.
"""

import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from biopb_tensor_server.cache import ArrowFileBackend
from biopb_tensor_server.cache.base import EntryState
from biopb_tensor_server.cache.file_backend import ArrowFileConfig
from biopb_tensor_server.core.adapter_base import pack_chunk_batch


def _batch(seed: int, elements: int = 4096):
    rng = np.random.default_rng(seed)
    return pack_chunk_batch(rng.integers(0, 65535, size=(elements,)).astype("<u2"))


def _backend(directory, deferred_mb=0, total_mb=256):
    return ArrowFileBackend(
        ArrowFileConfig(
            cache_dir=Path(directory),
            max_segment_bytes=1024 * 1024,
            max_total_bytes=total_mb * 1024 * 1024,
            max_deferred_write_bytes=deferred_mb * 1024 * 1024,
        )
    )


@pytest.fixture
def directory():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _store(backend, key, batch):
    """Drive the promise protocol the way ``resolve_chunk_data`` does."""
    return backend.get_or_acquire(key, lambda: (batch, batch.nbytes))


def _hold_the_writer(backend, gate, timeout=10):
    """Stall the background writer WITHOUT stalling inline writes.

    Gating every `_persist_entry` would also gate the backpressure path, which
    runs on the caller's thread -- and that is the path half of these tests are
    checking still works while the writer is stuck.
    """
    original = backend._persist_entry

    def gated(*args, **kwargs):
        if threading.current_thread().name == "biopb-cache-writer":
            gate.wait(timeout)
        return original(*args, **kwargs)

    backend._persist_entry = gated
    return original


# ==============================================================================
# The point: the caller stops waiting for the disk
# ==============================================================================


def test_the_caller_is_released_before_the_write_lands(directory):
    """The entry is READY, and serving it does not wait on the segment."""
    backend = _backend(directory, deferred_mb=64)
    gate = threading.Event()
    original = _hold_the_writer(backend, gate)
    try:
        batch = _batch(1)
        entry = _store(backend, b"k", batch)
        backend.release(b"k")

        # Committed from memory while the write is still blocked.
        assert entry.state == EntryState.READY
        assert entry.data is not None
        assert backend.stats().deferred_write_bytes > 0
        # ... and not yet locatable, so the localhost handoff falls back to
        # do_get rather than mapping a byte range that does not exist.
        assert backend.locate_entry(b"k") is None

        gate.set()
        assert backend.flush_deferred_writes(timeout=30)
        assert backend.locate_entry(b"k") is not None
        assert backend.stats().deferred_write_bytes == 0
    finally:
        gate.set()
        backend._persist_entry = original
        backend.close()


def test_a_deferred_entry_still_reaches_disk(directory):
    """Deferred is a schedule, not a downgrade: it must survive a restart."""
    backend = _backend(directory, deferred_mb=64)
    batch = _batch(2)
    _store(backend, b"k", batch)
    backend.release(b"k")
    assert backend.flush_deferred_writes(timeout=30)
    backend.close()

    reopened = _backend(directory, deferred_mb=64)
    try:
        assert reopened.locate_entry(b"k") is not None
    finally:
        reopened.close()


def test_close_drains_rather_than_sealing_over_the_queue(directory):
    """Without the drain, shutdown loses exactly what a restart looks for."""
    backend = _backend(directory, deferred_mb=64)
    for index in range(8):
        _store(backend, f"k{index}".encode(), _batch(index))
        backend.release(f"k{index}".encode())
    backend.close()  # no explicit flush

    reopened = _backend(directory, deferred_mb=64)
    try:
        for index in range(8):
            assert reopened.locate_entry(f"k{index}".encode()) is not None
    finally:
        reopened.close()


# ==============================================================================
# Backpressure
# ==============================================================================


def test_a_full_budget_writes_inline_instead_of_growing(directory):
    """Reaching the budget must not block and must not queue.

    The degraded behaviour is precisely today's: the caller writes its own
    entry. Anything else here is an unbounded memory leak wearing a queue.
    """
    backend = _backend(directory, deferred_mb=0)  # budget of zero == always inline
    try:
        batch = _batch(3)
        _store(backend, b"k", batch)
        backend.release(b"k")
        # Written on the calling thread, so it is locatable with no drain and
        # no writer thread was ever started.
        assert backend.locate_entry(b"k") is not None
        assert backend._writer_thread is None
        assert backend.stats().deferred_write_bytes == 0
    finally:
        backend.close()


def test_the_queue_never_exceeds_its_budget(directory):
    """Hold the writer still and pile on: queued bytes stay under the cap."""
    backend = _backend(directory, deferred_mb=1)
    gate = threading.Event()
    original = _hold_the_writer(backend, gate)
    try:
        cap = backend._config.max_deferred_write_bytes
        for index in range(64):
            key = f"k{index}".encode()
            _store(backend, key, _batch(index, elements=16384))
            backend.release(key)
            assert backend.stats().deferred_write_bytes <= cap
        # The overflow did not vanish -- it went to disk on the caller's thread.
        assert any(
            backend.locate_entry(f"k{index}".encode()) is not None
            for index in range(64)
        )
    finally:
        gate.set()
        backend.flush_deferred_writes(timeout=30)
        backend._persist_entry = original
        backend.close()


# ==============================================================================
# Failure has to be visible
# ==============================================================================


def test_a_failed_deferred_write_is_counted_not_swallowed(directory):
    """The caller was released long ago, so there is nobody left to raise to.

    The batch is still correct and still served; what is lost is persistence.
    That has to show up somewhere or the cache silently stops persisting and
    the only symptom is that everything gets slower.
    """
    backend = _backend(directory, deferred_mb=64)
    original = backend._persist_entry

    def boom(*_args, **_kwargs):
        raise OSError("no space left on device")

    backend._persist_entry = boom
    try:
        batch = _batch(4)
        entry = _store(backend, b"k", batch)
        backend.release(b"k")
        assert backend.flush_deferred_writes(timeout=30)

        assert backend.stats().deferred_write_failures == 1
        # Still served from memory -- the data was never wrong.
        assert entry.state == EntryState.READY
        assert entry.data is not None
        # ... and honestly reported as not on disk.
        assert backend.locate_entry(b"k") is None
        # The budget was returned, so one bad write does not wedge the queue.
        assert backend.stats().deferred_write_bytes == 0
    finally:
        backend._persist_entry = original
        backend.close()


# ==============================================================================
# Where it must NOT apply
# ==============================================================================


def test_uploads_are_never_deferred(directory):
    """``CacheManager.put`` has no backend to re-read from.

    ``CachedSourceAdapter.get_data`` raises and it serves only chunk_ids that
    were written, so an upload's only copy is what lands in the segment. It must
    not be told the bytes are stored until they are.
    """
    from biopb_tensor_server.cache.manager import CacheManager
    from biopb_tensor_server.core.config import CacheConfig

    manager = CacheManager(
        CacheConfig(
            backend="file",
            file_cache_dir=Path(directory),
            file_max_segment_bytes=1024 * 1024,
            file_max_total_bytes=64 * 1024 * 1024,
            file_deferred_write_mb=64,
        )
    )
    try:
        batch = _batch(5)
        assert manager.backend.SUPPORTS_DEFERRED_WRITES is True
        assert manager.put(b"upload", batch, batch.nbytes) is True
        # On disk the moment put() returns -- no drain, nothing queued.
        assert manager.backend.locate_entry(b"upload") is not None
        assert manager.backend.stats().deferred_write_bytes == 0
    finally:
        manager.close()


def test_read_path_writes_are_deferred_under_the_same_manager(directory):
    """The counterpart: the same backend defers what it is safe to defer."""
    from biopb_tensor_server.cache.manager import CacheManager
    from biopb_tensor_server.core.config import CacheConfig

    manager = CacheManager(
        CacheConfig(
            backend="file",
            file_cache_dir=Path(directory),
            file_max_segment_bytes=1024 * 1024,
            file_max_total_bytes=64 * 1024 * 1024,
            file_deferred_write_mb=64,
        )
    )
    backend = manager.backend
    gate = threading.Event()
    original = _hold_the_writer(backend, gate)
    try:
        batch = _batch(6)
        manager.get_or_acquire(b"read", lambda: (batch, batch.nbytes))
        manager.release(b"read")
        assert backend.stats().deferred_write_bytes > 0
    finally:
        gate.set()
        backend.flush_deferred_writes(timeout=30)
        backend._persist_entry = original
        manager.close()


# ==============================================================================
# Concurrency
# ==============================================================================


def test_concurrent_writers_and_readers_agree(directory):
    """Many keys in flight at once, then everything is on disk exactly once."""
    backend = _backend(directory, deferred_mb=64)
    errors = []
    batches = {f"k{index}".encode(): _batch(index) for index in range(32)}

    def worker(key, batch):
        try:
            entry = backend.get_or_acquire(key, lambda: (batch, batch.nbytes))
            assert entry.data is not None
            backend.release(key)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(key, batch))
        for key, batch in batches.items()
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors, errors
        assert backend.flush_deferred_writes(timeout=60)
        for key in batches:
            assert backend.locate_entry(key) is not None
        assert backend.stats().deferred_write_failures == 0
    finally:
        backend.close()


def test_flush_returns_false_rather_than_hanging(directory):
    """A wedged writer must be observable, not a silent hang at shutdown."""
    backend = _backend(directory, deferred_mb=64)
    gate = threading.Event()
    original = _hold_the_writer(backend, gate, timeout=30)
    try:
        batch = _batch(7)
        _store(backend, b"k", batch)
        backend.release(b"k")
        start = time.perf_counter()
        assert backend.flush_deferred_writes(timeout=0.2) is False
        assert time.perf_counter() - start < 5
    finally:
        gate.set()
        backend.flush_deferred_writes(timeout=30)
        backend._persist_entry = original
        backend.close()


def test_the_graceful_shutdown_path_drains_too(directory):
    """``release_process_lock`` promises the next boot a clean cache.

    It is the fast half of close(): it clears the WAL and drops the process lock
    while leaving handles open, on the grounds that completed writes are already
    flushed. Deferring makes that untrue -- a queued write is held by a daemon
    thread that dies with the process -- so it has to drain, and it has to do so
    before clearing the WAL out from under an in-flight write.
    """
    backend = _backend(directory, deferred_mb=64)
    try:
        for index in range(8):
            key = f"k{index}".encode()
            _store(backend, key, _batch(index))
            backend.release(key)
        backend.release_process_lock()
        assert backend.stats().deferred_write_bytes == 0
        for index in range(8):
            assert backend.locate_entry(f"k{index}".encode()) is not None
    finally:
        backend.close()


def test_a_cold_locate_still_answers_with_a_byte_range(directory):
    """The localhost handoff must not be retired by deferring.

    A cold locate does not simply fall back to do_get: the handler resolves the
    chunk -- materializing AND caching it -- then locates again, and only reports
    "unavailable" if that second locate fails. Deferring makes the second locate
    fail every time, because the entry is committed from memory with no byte
    range yet, so this caller has to wait for the write it just triggered. It is
    the one reader that wants bytes on disk rather than data in hand.
    """
    backend = _backend(directory, deferred_mb=64)
    try:
        batch = _batch(8)
        # Step 1: cold, nothing to locate.
        assert backend.locate_entry(b"k") is None
        # Step 2: the handler resolves, which caches from memory.
        _store(backend, b"k", batch)
        backend.release(b"k")
        # Step 3: without the wait this is still None and the client is sent
        # away; with it, the fast path answers as it did before deferring.
        assert backend.flush_deferred_write(b"k", timeout=30) is True
        assert backend.locate_entry(b"k") is not None
    finally:
        backend.close()


def test_awaiting_a_key_with_no_deferred_write_is_free(directory):
    """True immediately, so the caller need not know whether deferral is on."""
    backend = _backend(directory, deferred_mb=0)
    try:
        batch = _batch(9)
        _store(backend, b"k", batch)
        backend.release(b"k")
        start = time.perf_counter()
        assert backend.flush_deferred_write(b"k", timeout=30) is True
        assert time.perf_counter() - start < 0.5
        assert backend.flush_deferred_write(b"never-stored", timeout=30) is True
    finally:
        backend.close()


# ==============================================================================
# Shutdown when the writer will not stop
# ==============================================================================


def _wedge_mid_write(backend):
    """A writer stuck *inside* a write, after its WAL pending record.

    ``_persist_entry`` logs pending, writes the batch, then logs committed.
    Blocking at the last step is the state the shutdown policy is about: the WAL
    says a write is outstanding, and it is -- so clearing it would erase the one
    signal that makes the next start recover.
    """
    stuck, release = threading.Event(), threading.Event()
    original = backend._wal.log_committed

    def wedged(key):
        stuck.set()
        release.wait(120)
        return original(key)

    backend._wal.log_committed = wedged
    return stuck, release, original


def _wedge_the_writer(backend):
    """A writer thread that never finishes its current write."""
    stuck = threading.Event()
    original = backend._persist_entry

    def wedged(*args, **kwargs):
        if threading.current_thread().name == "biopb-cache-writer":
            stuck.set()
            release.wait(120)
        return original(*args, **kwargs)

    release = threading.Event()
    backend._persist_entry = wedged
    return stuck, release, original


def test_a_wedged_writer_does_not_get_its_segment_closed_underneath_it(directory):
    """The race this policy exists to prevent.

    ``_close_writer`` takes no ``_write_lock``: it detaches the writer and sink
    and closes them, while ``_persist_entry``'s blocking phase runs with
    ``_lock`` released, on references it captured earlier. Closing segments
    while that thread is alive closes a sink out from under an in-flight write.
    So ``close()`` must decline to touch anything when the drain fails.
    """
    backend = _backend(directory, deferred_mb=64)
    stuck, release, original = _wedge_the_writer(backend)
    try:
        _store(backend, b"k", _batch(10))
        backend.release(b"k")
        assert stuck.wait(10), "writer never reached the wedged write"
        open_ids = backend._open_segment_ids()

        backend._stop_writer(timeout=0.2)
        # The thread is still alive, so it must still own its segments.
        assert backend._writer_thread is not None
        assert backend._open_segment_ids() == open_ids
        for segment_id in open_ids:
            assert segment_id in backend._pool_writers
    finally:
        release.set()
        backend._persist_entry = original
        backend.flush_deferred_writes(timeout=30)
        backend.close()


def test_a_failed_drain_leaves_the_evidence_for_the_next_start(directory):
    """WAL pending + process lock held are what trigger recovery.

    Clearing either would tell the next start that this one finished cleanly,
    which is exactly the claim that is false while a writer is still running.
    """
    backend = _backend(directory, deferred_mb=64)
    stuck, release, original = _wedge_mid_write(backend)
    try:
        _store(backend, b"k", _batch(11))
        backend.release(b"k")
        assert stuck.wait(10)

        # close() re-drains with its own bound, so give it a short one: the
        # writer is wedged on purpose and the real bound is 60 s.
        backend.close(drain_timeout=0.2)

        assert backend._wal is not None and backend._wal.has_pending()
        assert backend._process_lock is not None
        assert backend._process_lock.is_held()
        # No sidecar was written for a segment whose index was still moving.
        assert not list(Path(directory).rglob("*.idx"))
    finally:
        release.set()
        backend._wal.log_committed = original
        backend.flush_deferred_writes(timeout=30)
        backend.close()


def test_release_process_lock_declines_rather_than_doing_half_of_it(directory):
    """Its two effects are both claims that are false mid-write."""
    backend = _backend(directory, deferred_mb=64)
    stuck, release, original = _wedge_mid_write(backend)
    try:
        _store(backend, b"k", _batch(12))
        backend.release(b"k")
        assert stuck.wait(10)

        # Short bound: the writer is wedged on purpose, and the point is
        # what this declines to do, not how long it waits first.
        backend.release_process_lock(drain_timeout=0.2)

        assert backend._wal.has_pending()
        assert backend._process_lock.is_held()
    finally:
        release.set()
        backend._wal.log_committed = original
        backend.flush_deferred_writes(timeout=30)
        backend.close()


def test_a_clean_drain_still_clears_everything(directory):
    """The guard must not fire on the ordinary path."""
    backend = _backend(directory, deferred_mb=64)
    for index in range(4):
        key = f"k{index}".encode()
        _store(backend, key, _batch(index))
        backend.release(key)
    backend.close()

    assert not backend._wal.has_pending()
    assert not backend._process_lock.is_held()


def test_a_backend_that_cannot_defer_keeps_its_original_signature():
    """``complete_entry`` must not have widened for backends that never defer.

    ``CacheManager.put`` asks ``SUPPORTS_DEFERRED_WRITES`` before it passes
    ``allow_deferred``, so a backend written against the historical three-
    argument form -- the memory backend, or one outside this tree -- is called
    exactly as it always was.
    """
    import inspect

    from biopb_tensor_server.cache.base import CacheBackend
    from biopb_tensor_server.cache.memory_backend import MemoryCacheBackend

    for klass in (CacheBackend, MemoryCacheBackend):
        assert klass.SUPPORTS_DEFERRED_WRITES is False
        assert list(inspect.signature(klass.complete_entry).parameters) == [
            "self",
            "key",
            "data",
            "size_bytes",
        ], f"{klass.__name__} widened the contract"
    assert ArrowFileBackend.SUPPORTS_DEFERRED_WRITES is True


def test_put_works_against_a_backend_that_does_not_defer(directory):
    """The compatibility branch, exercised rather than reasoned about."""
    from biopb_tensor_server.cache.manager import CacheManager
    from biopb_tensor_server.core.config import CacheConfig

    manager = CacheManager(CacheConfig(backend="memory"))
    try:
        assert manager.backend.SUPPORTS_DEFERRED_WRITES is False
        batch = _batch(13)
        assert manager.put(b"upload", batch, batch.nbytes) is True
    finally:
        manager.close()
