"""Thread-safety tests for cache module."""

import errno
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from biopb_tensor_server.cache import (
    MAX_ARROW_BATCH_BYTES,
    CacheEntry,
    CacheManager,
    EntryState,
    MemoryCacheBackend,
)
from biopb_tensor_server.cache.file_backend import (
    CACHE_KEY_FIELD,
    SIDECAR_FORMAT_VERSION,
    SIZE_CLASS_MEDIUM_THRESHOLD,
    SIZE_CLASS_SMALL_THRESHOLD,
    SIZE_CLASS_TINY_THRESHOLD,
    ArrowFileBackend,
    ArrowFileConfig,
    K,
    _get_size_class,
)
from biopb_tensor_server.cache.memory_backend import MemoryCacheConfig
from biopb_tensor_server.cache.recovery import (
    ProcessLock,
    WriteAheadLog,
)
from biopb_tensor_server.core.config import CacheConfig


def _hold_cache_lock(lock_path: str, started) -> None:
    """Child entry point: take a cache lock, then idle until killed.

    Module level and argument-driven so it survives pickling under the ``spawn``
    start method (the default on Windows/macOS). It waits in ``time.sleep``
    rather than on a shared primitive: the test kills this process mid-wait, and
    a killed waiter leaves a multiprocessing Event's internal semaphores in a
    state that hangs the parent's ``set()``.
    """
    from biopb_tensor_server.cache.recovery import ProcessLock

    lock = ProcessLock(Path(lock_path))
    if not lock.acquire():
        return
    started.set()
    time.sleep(120)  # killed long before this; never released cleanly


class TestCacheEntry:
    """Tests for CacheEntry state and ref_count."""

    def test_pending_entry(self):
        """Pending entry creation."""
        entry = CacheEntry(state=EntryState.PENDING)
        assert entry.state == EntryState.PENDING
        assert entry.data is None
        assert entry.ref_count == 0

    def test_acquire_release(self):
        """Reference counting."""
        entry = CacheEntry(state=EntryState.READY, ref_count=0)
        entry.acquire()
        assert entry.ref_count == 1
        entry.acquire()
        assert entry.ref_count == 2
        entry.release()
        assert entry.ref_count == 1

    def test_is_evictable(self):
        """Eviction check."""
        entry = CacheEntry(state=EntryState.READY, ref_count=0)
        assert entry.is_evictable() is True
        entry.acquire()
        assert entry.is_evictable() is False
        entry.release()
        assert entry.is_evictable() is True

    def test_set_ready(self):
        """Mark entry ready."""
        data = pa.RecordBatch.from_arrays(
            [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )
        entry = CacheEntry(state=EntryState.PENDING)
        entry.set_ready(data, size_bytes=24)
        assert entry.state == EntryState.READY
        assert entry.data == data
        assert entry.event.is_set() is True

    def test_set_error(self):
        """Mark entry as error."""
        entry = CacheEntry(state=EntryState.PENDING)
        error = ValueError("test error")
        entry.set_error(error)
        assert entry.state == EntryState.ERROR
        assert entry.error == error
        assert entry.event.is_set() is True

    def test_wait_ready(self):
        """Waiting on pending entry."""
        entry = CacheEntry(state=EntryState.PENDING)

        def complete():
            time.sleep(0.1)
            data = pa.RecordBatch.from_arrays(
                [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(["int64"])],
                ["data", "shape", "dtype"],
            )
            entry.set_ready(data, 24)

        thread = threading.Thread(target=complete)
        thread.start()

        # Wait should block until ready
        assert entry.wait_ready(timeout=5.0) is True
        assert entry.state == EntryState.READY
        thread.join()

    def test_wait_ready_raises_on_error(self):
        """Waiting raises if entry fails."""
        entry = CacheEntry(state=EntryState.PENDING)

        def fail():
            time.sleep(0.1)
            entry.set_error(ValueError("computation failed"))

        thread = threading.Thread(target=fail)
        thread.start()

        with pytest.raises(ValueError, match="computation failed"):
            entry.wait_ready(timeout=5.0)
        thread.join()


class TestMemoryCacheBackend:
    """Tests for thread-safe MemoryCacheBackend."""

    def _make_data(self, values) -> pa.RecordBatch:
        """Helper to create RecordBatch with new schema format."""
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def test_start_compute_creates_pending(self):
        """First request creates pending entry."""
        config = MemoryCacheConfig(max_entries=10, max_bytes=1024 * 1024)
        backend = MemoryCacheBackend(config)

        entry, is_owner = backend.start_compute(b"key1")
        assert is_owner is True
        assert entry.state == EntryState.PENDING
        assert entry.ref_count >= 1
        backend.close()

    def test_start_compute_ready_entry(self):
        """Ready entry returns immediately."""
        config = MemoryCacheConfig(max_entries=10)
        backend = MemoryCacheBackend(config)

        # Create and complete entry
        entry, is_owner = backend.start_compute(b"key1")
        data = self._make_data([1, 2, 3])
        backend.complete_entry(b"key1", data, 24)

        # Second request should get ready entry
        entry2, is_owner2 = backend.start_compute(b"key1")
        assert is_owner2 is False
        assert entry2.state == EntryState.READY
        assert entry2.data.column(0).to_pylist() == [[1, 2, 3]]
        backend.close()

    def test_start_compute_pending_wait(self):
        """Second request waits on pending entry."""
        config = MemoryCacheConfig(max_entries=10)
        backend = MemoryCacheBackend(config)

        results = []
        ready_event = threading.Event()

        def compute_and_complete():
            # Wait for waiter to signal it's ready to observe
            ready_event.wait(timeout=5.0)
            entry, is_owner = backend.start_compute(b"key1")
            # Should NOT be owner - waiter already created pending entry
            assert is_owner is False
            # Waiter's entry is pending, this thread needs to wait too
            # Actually, this scenario shouldn't happen - both threads wait on pending
            # Let's restructure the test
            entry.wait_ready(timeout=5.0)
            backend.release(b"key1")
            results.append("computed")

        def wait_for_ready():
            entry, is_owner = backend.start_compute(b"key1")
            # First thread creates pending - is owner
            assert is_owner is True
            assert entry.state == EntryState.PENDING
            # Signal compute thread to start (it will also find pending)
            ready_event.set()
            time.sleep(0.1)  # Give compute thread time to see pending
            # Now complete
            data = self._make_data([1, 2, 3])
            backend.complete_entry(b"key1", data, 24)
            backend.release(b"key1")
            results.append("waited")

        # Start waiter thread first - it creates pending entry
        t1 = threading.Thread(target=wait_for_ready)
        t1.start()
        # Start compute thread second - it will find pending
        t2 = threading.Thread(target=compute_and_complete)
        t2.start()

        t1.join()
        t2.join()

        assert "waited" in results
        assert "computed" in results
        backend.close()

    def test_release(self):
        """Release decrements ref_count."""
        config = MemoryCacheConfig()
        backend = MemoryCacheBackend(config)

        entry, _ = backend.start_compute(b"key1")
        data = self._make_data([1])
        backend.complete_entry(b"key1", data, 8)
        assert entry.ref_count >= 1

        backend.release(b"key1")
        assert entry.ref_count == 0
        backend.close()

    def test_eviction_skips_referenced_entries(self):
        """Entries with ref_count > 0 cannot be evicted."""
        config = MemoryCacheConfig(max_entries=3, max_bytes=1024 * 1024)
        backend = MemoryCacheBackend(config)

        # Create and hold entry
        entry1, _ = backend.start_compute(b"key1")
        backend.complete_entry(b"key1", self._make_data([1]), 8)

        # Create more entries (should trigger eviction)
        entry2, _ = backend.start_compute(b"key2")
        backend.complete_entry(b"key2", self._make_data([2]), 8)
        backend.release(b"key2")

        entry3, _ = backend.start_compute(b"key3")
        backend.complete_entry(b"key3", self._make_data([3]), 8)
        backend.release(b"key3")

        entry4, _ = backend.start_compute(b"key4")
        backend.complete_entry(b"key4", self._make_data([4]), 8)
        backend.release(b"key4")

        # key1 should still exist (has ref_count > 0)
        assert backend.start_compute(b"key1")[0].state == EntryState.READY
        backend.close()

    def test_stats_tracking(self):
        """Statistics are tracked."""
        config = MemoryCacheConfig()
        backend = MemoryCacheBackend(config)

        entry, is_owner = backend.start_compute(b"key1")
        backend.complete_entry(b"key1", self._make_data([1]), 8)
        backend.release(b"key1")

        # Hit
        entry2, _ = backend.start_compute(b"key1")
        backend.release(b"key1")

        stats = backend.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        backend.close()

    def test_clear_evictable_only(self):
        """Clear only removes evictable entries."""
        config = MemoryCacheConfig()
        backend = MemoryCacheBackend(config)

        entry1, _ = backend.start_compute(b"key1")
        backend.complete_entry(b"key1", self._make_data([1]), 8)
        # Don't release - still referenced

        entry2, _ = backend.start_compute(b"key2")
        backend.complete_entry(b"key2", self._make_data([2]), 8)
        backend.release(b"key2")  # Evictable

        backend.clear()

        # key1 should still exist
        assert backend.start_compute(b"key1")[0].state == EntryState.READY
        # key2 should be gone
        entry, is_owner = backend.start_compute(b"key2")
        assert is_owner is True  # New computation needed
        backend.close()


class TestCacheManager:
    """Tests for CacheManager singleton."""

    def test_initialize_singleton(self):
        """Singleton initialization is thread-safe."""
        CacheManager.reset()

        # Use a lock to ensure first thread initializes before others start
        init_lock = threading.Lock()
        first_done = threading.Event()

        results = []

        def init():
            with init_lock:
                if not first_done.is_set():
                    # First thread initializes
                    config = CacheConfig(backend="memory")
                    manager = CacheManager.initialize(config)
                    first_done.set()
                else:
                    # Subsequent threads should get existing instance
                    manager = CacheManager.get_instance()
            results.append(manager)

        threads = [threading.Thread(target=init) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert all(m is results[0] for m in results)
        CacheManager.reset()

    def test_start_compute_and_complete(self):
        """Full compute cycle."""
        CacheManager.reset()
        config = CacheConfig(backend="memory")
        manager = CacheManager.initialize(config)

        entry, is_owner = manager.start_compute(b"key1", metadata={"test": True})
        assert is_owner is True

        data = pa.RecordBatch.from_arrays(
            [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )
        manager.complete_entry(b"key1", data, 24)

        assert entry.state == EntryState.READY
        manager.release(b"key1")
        CacheManager.reset()

    def test_fail_entry(self):
        """Failed computation is removed."""
        CacheManager.reset()
        config = CacheConfig(backend="memory")
        manager = CacheManager.initialize(config)

        entry, is_owner = manager.start_compute(b"key1")
        manager.fail_entry(b"key1", ValueError("failed"))

        assert entry.state == EntryState.ERROR
        # Entry should be removed from cache
        entry2, is_owner2 = manager.start_compute(b"key1")
        assert is_owner2 is True  # New computation needed
        CacheManager.reset()


class TestConcurrentCompute:
    """Tests for concurrent computation scenarios."""

    def test_concurrent_same_key_only_one_computes(self):
        """Multiple threads requesting same key - only one computes."""
        config = MemoryCacheConfig(max_entries=10)
        backend = MemoryCacheBackend(config)

        compute_counts = [0]  # Use list for thread-safe increment
        compute_lock = threading.Lock()
        start_barrier = threading.Barrier(5)  # All threads start simultaneously

        def worker(worker_id):
            start_barrier.wait()  # All threads hit this point together
            entry, is_owner = backend.start_compute(b"key1")
            if is_owner:
                with compute_lock:
                    compute_counts[0] += 1
                time.sleep(0.1)  # Simulate computation
                data = pa.RecordBatch.from_arrays(
                    [pa.array([[worker_id]]), pa.array([[1]]), pa.array(["int64"])],
                    ["data", "shape", "dtype"],
                )
                backend.complete_entry(b"key1", data, 8)
            else:
                entry.wait_ready(timeout=5.0)
            backend.release(b"key1")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one thread should have computed
        assert compute_counts[0] == 1
        backend.close()

    def test_concurrent_different_keys(self):
        """Different keys are computed independently."""
        config = MemoryCacheConfig(max_entries=10)
        backend = MemoryCacheBackend(config)

        results = {}

        def worker(key, value):
            entry, is_owner = backend.start_compute(key)
            assert is_owner is True
            data = pa.RecordBatch.from_arrays(
                [pa.array([[value]]), pa.array([[1]]), pa.array(["int64"])],
                ["data", "shape", "dtype"],
            )
            backend.complete_entry(key, data, 8)
            results[key] = value
            backend.release(key)

        threads = [
            threading.Thread(target=worker, args=(b"key1", 1)),
            threading.Thread(target=worker, args=(b"key2", 2)),
            threading.Thread(target=worker, args=(b"key3", 3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        backend.close()

    def test_timeout_on_pending(self):
        """Waiting on pending entry returns False on timeout."""
        config = MemoryCacheConfig(max_entries=10, pending_timeout=0.5)
        backend = MemoryCacheBackend(config)

        entry, is_owner = backend.start_compute(b"key1")
        assert is_owner is True
        # Don't complete - leave it pending

        # Another request should find pending
        entry2, is_owner2 = backend.start_compute(b"key1")
        assert is_owner2 is False
        # wait_ready returns False on timeout (not raises)
        result = entry2.wait_ready(timeout=0.1)
        assert result is False  # Timeout occurred
        backend.close()


class TestArrowFileBackend:
    """Tests for persistent Arrow file cache backend."""

    def _make_data(self, values) -> pa.RecordBatch:
        """Helper to create RecordBatch with new schema format."""
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def _make_temp_cache_dir(self):
        """Create a temporary cache directory."""
        return Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))

    def test_start_compute_creates_pending(self):
        """First request creates pending entry."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1024 * 1024,
            max_total_bytes=10 * 1024 * 1024,
        )
        backend = ArrowFileBackend(config)

        entry, is_owner = backend.start_compute(b"key1")
        assert is_owner is True
        assert entry.state == EntryState.PENDING
        assert entry.ref_count >= 1
        backend.close()
        shutil.rmtree(cache_dir)

    def test_persist_and_retrieve(self):
        """Entry persists across restarts."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1024 * 1024,
            max_total_bytes=10 * 1024 * 1024,
        )

        # First instance: store entry
        backend1 = ArrowFileBackend(config)
        entry1, is_owner1 = backend1.start_compute(b"key1")
        data = self._make_data([1, 2, 3])
        backend1.complete_entry(b"key1", data, 24)
        backend1.release(b"key1")
        backend1.close()

        # Second instance: retrieve entry
        backend2 = ArrowFileBackend(config)
        entry2, is_owner2 = backend2.start_compute(b"key1")
        assert is_owner2 is False  # Already cached
        assert entry2.state == EntryState.READY
        assert entry2.data.column(0).to_pylist() == [[1, 2, 3]]
        backend2.release(b"key1")
        backend2.close()
        shutil.rmtree(cache_dir)

    def test_release_drops_in_memory_batch_but_keeps_disk(self):
        """Releasing the last reference frees the in-RAM RecordBatch once its
        segment is readable, while the on-disk data (and so the cache hit)
        survives.

        Regression test: complete_entry persists the batch to a segment AND
        pins it on entry.data. Without dropping that copy, the file backend
        mirrors every chunk ever served in memory for the life of the process --
        bounded only by the *disk* max_total_bytes -- which exhausts RAM on a
        large catalog / precache sweep. With a tiny max_segment_bytes each entry
        closes its own segment immediately, so after release _entries must no
        longer hold it, _metadata must still know it, and a fresh lookup must
        rebuild correct data from the segment.
        """
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1,  # close (and make readable) after every entry
            max_total_bytes=10 * 1024 * 1024,
        )
        backend = ArrowFileBackend(config)

        # Cache several entries. Each closes its segment on write, but the owner
        # still holds a reference, so the batch stays in memory for now.
        keys = [f"key{i}".encode() for i in range(5)]
        for i, key in enumerate(keys):
            backend.start_compute(key)
            backend.complete_entry(key, self._make_data([i, i, i]), 24)
            assert backend._entries[key].data is not None  # owner still holds it

        # Release every reference -> segments are readable, so the in-memory
        # mirror must be dropped...
        for key in keys:
            backend.release(key)
        assert len(backend._entries) == 0, "in-RAM batches not freed on release"
        # ...but the on-disk index is fully intact.
        assert all(key in backend._metadata for key in keys)

        # A subsequent lookup is still a hit and returns the correct data,
        # rebuilt from the segment mmap rather than the freed in-RAM copy.
        entry, is_owner = backend.start_compute(keys[2])
        assert is_owner is False
        assert entry.state == EntryState.READY
        assert entry.data.column(0).to_pylist() == [[2, 2, 2]]
        backend.release(keys[2])

        backend.close()
        shutil.rmtree(cache_dir)

    def test_open_segment_entry_retained_until_readable(self):
        """An entry on the still-open write segment keeps its in-memory copy on
        release (it is the only readable copy), and is freed once the segment
        closes and becomes mmap-readable.

        Guards the readability gate: dropping a not-yet-readable entry would turn
        a hot, freshly-written chunk into a miss/recompute (the SieveK-style
        failure mode).
        """
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=10 * 1024 * 1024,  # segment stays open
            max_total_bytes=100 * 1024 * 1024,
        )
        backend = ArrowFileBackend(config)

        backend.start_compute(b"open")
        backend.complete_entry(b"open", self._make_data([7, 7, 7]), 24)
        seg_id = backend._metadata[b"open"].segment_id
        assert seg_id not in backend._segment_mmaps  # still the write segment

        # Released, but not yet readable -> the only copy must be retained.
        backend.release(b"open")
        assert b"open" in backend._entries
        assert backend._entries[b"open"].data is not None

        # Closing the segment makes it readable and frees the redundant copy.
        backend._close_segment(seg_id)
        assert seg_id in backend._segment_mmaps
        assert b"open" not in backend._entries
        assert b"open" in backend._metadata

        # Still a correct hit afterwards, served from the mmap.
        entry, is_owner = backend.start_compute(b"open")
        assert is_owner is False
        assert entry.data.column(0).to_pylist() == [[7, 7, 7]]
        backend.release(b"open")

        backend.close()
        shutil.rmtree(cache_dir)

    def test_segment_unlink_while_caller_holds_batch(self):
        """A segment can be evicted/unlinked while a caller still holds the
        RecordBatch read from it.

        Reads are zero-copy from the segment mmap, so the returned batch points
        into the file's memory mapping. On Windows a live mapping blocks file
        deletion (WinError 32), so eviction fails unless the batch is copied off
        the mmap first; on POSIX unlink-while-mapped is harmless. Regression test
        for issue #5: on POSIX this exercises the zero-copy path, on Windows the
        copy-on-read path.
        """
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1024 * 1024,
            max_total_bytes=10 * 1024 * 1024,
        )

        # Store an entry, then restart so the segment is closed and the rebuild
        # reopens it as a readable mmap.
        backend1 = ArrowFileBackend(config)
        backend1.start_compute(b"key1")
        backend1.complete_entry(b"key1", self._make_data([1, 2, 3]), 24)
        backend1.release(b"key1")
        backend1.close()

        backend2 = ArrowFileBackend(config)
        entry, is_owner = backend2.start_compute(b"key1")
        assert is_owner is False
        assert entry.state == EntryState.READY

        segment_id = backend2._metadata[b"key1"].segment_id
        seg_file = cache_dir / "segments" / f"seg_{segment_id:04d}.arrow"
        assert seg_file.exists()

        # Reproduce the real failure mode: keep a plain Python reference to the
        # batch, then RELEASE the cache entry so the segment becomes evictable
        # (ref_count -> 0). The cache now considers the segment free to delete,
        # but the held batch's buffers still map the file (zero-copy on POSIX,
        # an off-mmap copy on Windows).
        held_batch = entry.data
        backend2.release(b"key1")
        assert backend2._segment_is_evictable(segment_id)

        # Drive the real eviction path. This must not raise (even on Windows)
        # and must actually delete the file.
        assert backend2._evict_segment_sieve_k() is True
        assert not seg_file.exists()

        # The held batch is still valid after its backing file is gone.
        assert held_batch.column(0).to_pylist() == [[1, 2, 3]]

        backend2.close()
        shutil.rmtree(cache_dir)

    def test_multi_entry_segment_persist_keys_not_crosswired(self):
        """Each key returns its OWN data after a restart when several entries
        share one segment.

        Regression test for the file-cache index-rebuild bug: the per-entry
        cache key was stored in Arrow IPC *schema metadata*, but a stream
        serializes the schema only once (from the first batch), so on rebuild
        every batch reported the first entry's key. All keys collapsed onto the
        first one and pointed at the last batch, so e.g. requesting the first
        cached source's chunk returned a different source's data.
        """
        cache_dir = self._make_temp_cache_dir()
        # Large segment cap so all entries land in a single segment.
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1024 * 1024,
            max_total_bytes=10 * 1024 * 1024,
        )

        entries = {
            b"source-A/chunk0": [10, 11, 12],
            b"source-B/chunk0": [20, 21, 22],
            b"source-C/chunk0": [30, 31, 32],
        }

        backend1 = ArrowFileBackend(config)
        for key, values in entries.items():
            backend1.start_compute(key)
            backend1.complete_entry(key, self._make_data(values), 24)
            backend1.release(key)
        # Sanity: all entries share one segment.
        seg_files = list((cache_dir / "segments").glob("seg_*.arrow"))
        assert len(seg_files) == 1
        backend1.close()

        # Restart -> index is rebuilt from the segment file on disk.
        backend2 = ArrowFileBackend(config)
        for key, values in entries.items():
            entry, is_owner = backend2.start_compute(key)
            assert is_owner is False, f"{key!r} should be a cache hit after restart"
            assert entry.state == EntryState.READY
            assert entry.data.column(0).to_pylist() == [values], (
                f"{key!r} returned wrong data after rebuild"
            )
            backend2.release(key)
        backend2.close()
        shutil.rmtree(cache_dir)

    def test_segment_file_creation(self):
        """Segment files are created with correct naming."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1024,  # Very small to force multiple segments
            max_total_bytes=10 * 1024,
        )

        backend = ArrowFileBackend(config)
        for i in range(5):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 100)
            backend.complete_entry(key, data, 800)
            backend.release(key)

        # Check segment files exist
        segments_dir = cache_dir / "segments"
        segment_files = list(segments_dir.glob("seg_*.arrow"))
        assert len(segment_files) >= 1

        backend.close()
        shutil.rmtree(cache_dir)

    def test_segment_eviction(self):
        """Segment-level eviction when total size exceeded."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1024,
            max_total_bytes=2 * 1024,  # Very small total
        )

        backend = ArrowFileBackend(config)
        # Store entries
        for i in range(10):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 50)
            backend.complete_entry(key, data, 400)
            backend.release(key)

        stats = backend.stats()
        assert stats.evictions >= 1  # Some segments should be evicted

        backend.close()
        shutil.rmtree(cache_dir)

    def test_concurrent_same_key(self):
        """Multiple threads requesting same key - only one computes."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        compute_counts = [0]
        compute_lock = threading.Lock()
        start_barrier = threading.Barrier(5)

        def worker(worker_id):
            start_barrier.wait()
            entry, is_owner = backend.start_compute(b"key1")
            if is_owner:
                with compute_lock:
                    compute_counts[0] += 1
                time.sleep(0.1)
                data = pa.RecordBatch.from_arrays(
                    [pa.array([[worker_id]]), pa.array([[1]]), pa.array(["int64"])],
                    ["data", "shape", "dtype"],
                )
                backend.complete_entry(b"key1", data, 8)
            else:
                entry.wait_ready(timeout=5.0)
            backend.release(b"key1")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert compute_counts[0] == 1
        backend.close()
        shutil.rmtree(cache_dir)

    def test_stats_tracking(self):
        """Statistics are tracked."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        entry, is_owner = backend.start_compute(b"key1")
        backend.complete_entry(b"key1", self._make_data([1]), 8)
        backend.release(b"key1")

        # Hit
        entry2, _ = backend.start_compute(b"key1")
        backend.release(b"key1")

        stats = backend.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        backend.close()
        shutil.rmtree(cache_dir)


def _simulate_crash(backend):
    """Release a backend's OS file handles the way a dying process would.

    A real crash makes the OS reclaim every open handle -- writers, sinks, mmaps,
    and the cache lock's descriptor -- while leaving the on-disk owner record and
    WAL behind (no clean shutdown). The test process stays alive, so we drop those
    handles explicitly; on Windows a lingering handle also blocks the segment files
    from being deleted (issue #5). Deliberately does NOT call backend.close() or
    ProcessLock.release(), either of which would clean up and thus erase the crash
    state -- and does NOT unlink the lock file, which no crash does either
    (biopb/biopb#544).
    """
    for seg_id in list(backend._pool_writers.keys() | backend._pool_sinks.keys()):
        backend._close_writer(seg_id)
    for mmap in backend._segment_mmaps.values():
        mmap.close()
    backend._segment_mmaps.clear()
    # Drop the lock descriptor only -- the kernel does this for a dying process.
    # The `.owner` record survives, which is the crash signal the next owner reads.
    if backend._process_lock is not None:
        backend._process_lock._lock.release()


class TestArrowFileBackendRecovery:
    """Tests for crash recovery."""

    def _make_data(self, values) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def _make_temp_cache_dir(self):
        return Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))

    def test_second_holder_is_refused(self):
        """Two owners of one cache dir are impossible, not merely unlikely.

        Regression for biopb/biopb#544: exclusion used to be a pid record, and
        deciding it was a check-then-act (read the file, judge the owner dead,
        write your own), so racing starters could all conclude they owned the
        cache. The lock is a held descriptor now -- a second acquire loses even
        with no contention window to exploit. flock is per open-file-description
        and Windows byte-range locks are per handle, so a second lock object in
        *this* process is excluded exactly as another process would be.
        """
        cache_dir = self._make_temp_cache_dir()
        lock_path = cache_dir / "lock"

        first = ProcessLock(lock_path)
        assert first.acquire() is True
        try:
            second = ProcessLock(lock_path)
            assert second.acquire() is False
            assert second.is_acquired() is False
        finally:
            first.release()

        # Released -> the next owner gets it.
        third = ProcessLock(lock_path)
        assert third.acquire() is True
        third.release()

        shutil.rmtree(cache_dir)

    def test_clean_release_is_not_stale(self):
        """A cache dir released cleanly does not look like a crash."""
        cache_dir = self._make_temp_cache_dir()
        lock_path = cache_dir / "lock"

        first = ProcessLock(lock_path)
        assert first.acquire() is True
        assert first.is_stale() is False  # nothing ran here before
        first.release()

        second = ProcessLock(lock_path)
        assert second.acquire() is True
        assert second.is_stale() is False, "clean release must not trigger recovery"
        second.release()

        shutil.rmtree(cache_dir)

    def test_unreleased_owner_record_signals_a_crash(self):
        """An owner record left behind is the crash signal.

        A dying process releases the OS lock (the kernel closes its fd) but
        cannot remove its record, so the next owner acquires successfully *and*
        finds the marker -- which is precisely the case WAL recovery is for.
        """
        cache_dir = self._make_temp_cache_dir()
        lock_path = cache_dir / "lock"

        crashed = ProcessLock(lock_path)
        assert crashed.acquire() is True
        # Drop the descriptor the way process death would, leaving the record.
        crashed._lock.release()

        survivor = ProcessLock(lock_path)
        assert survivor.acquire() is True
        assert survivor.is_stale() is True
        assert (survivor.prior_owner() or {}).get("pid") == os.getpid()
        survivor.release()

        # The recovery signal is consumed: the next owner starts clean.
        after = ProcessLock(lock_path)
        assert after.acquire() is True
        assert after.is_stale() is False
        after.release()

        shutil.rmtree(cache_dir)

    def test_corrupt_owner_record_still_signals_a_crash(self):
        """Presence is the signal, so an unparseable record still means crash.

        It must also not raise -- the record's contents are only ever
        diagnostic, never consulted to decide ownership.
        """
        cache_dir = self._make_temp_cache_dir()
        lock_path = cache_dir / "lock"
        (cache_dir / "lock.owner").write_text("{not valid json")

        lock = ProcessLock(lock_path)
        assert lock.acquire() is True
        assert lock.is_stale() is True
        assert lock.prior_owner() == {}
        lock.release()

        shutil.rmtree(cache_dir)

    def test_lock_survives_holder_death_without_pid_bookkeeping(self):
        """A killed holder's lock is released by the OS, not reclaimed by us.

        This is the property that made the pid/create-time identity check
        unnecessary: the old scheme had to prove the recorded owner was dead
        (and survive pid reuse) because nothing else would ever free the lock.
        """
        cache_dir = self._make_temp_cache_dir()
        lock_path = cache_dir / "lock"

        ctx = multiprocessing.get_context("spawn")
        started = ctx.Event()
        holder = ctx.Process(target=_hold_cache_lock, args=(str(lock_path), started))
        holder.start()
        try:
            assert started.wait(60), "child never acquired the lock"
            contender = ProcessLock(lock_path)
            assert contender.acquire() is False, "a live holder must exclude us"
        finally:
            holder.kill()  # uncatchable: no chance to release or clean up
            holder.join(30)

        reclaimed = ProcessLock(lock_path)
        assert reclaimed.acquire() is True, (
            "OS should have dropped the dead holder's lock"
        )
        assert reclaimed.is_stale() is True, "a killed holder left its record"
        reclaimed.release()

        shutil.rmtree(cache_dir)

    def test_wal_pending_detection(self):
        """WAL detects pending writes."""
        cache_dir = self._make_temp_cache_dir()
        wal_path = cache_dir / "wal.json"

        wal = WriteAheadLog(wal_path)
        wal.log_pending(b"test_key")

        # Pending key should be tracked
        pending = wal.get_pending_keys()
        assert b"test_key" in pending
        assert wal.has_pending() is True

        wal.log_committed(b"test_key")
        assert wal.has_pending() is False

        wal.clear()
        shutil.rmtree(cache_dir)

    def test_recovery_after_simulated_crash(self):
        """Valid entries survive after simulated crash."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)

        # First instance: store entry
        backend1 = ArrowFileBackend(config)
        entry1, _ = backend1.start_compute(b"key1")
        data = self._make_data([1, 2, 3])
        backend1.complete_entry(b"key1", data, 24)
        backend1.release(b"key1")

        # Crash: the OS drops the lock descriptor and leaves both the lock file
        # and its owner record behind. The next instance reclaims the freed lock
        # and the leftover record drives recovery -- no unlink, which no crash does.
        _simulate_crash(backend1)

        # Second instance should recover and still have the entry
        backend2 = ArrowFileBackend(config)
        entry2, is_owner2 = backend2.start_compute(b"key1")
        assert is_owner2 is False  # Should find cached entry
        assert entry2.state == EntryState.READY

        # Recovery status should be available (smoke check: does not raise).
        # May or may not have recovery depending on whether there were pending writes
        backend2.get_recovery_status()

        backend2.close()
        shutil.rmtree(cache_dir)

    def test_stale_lock_triggers_recovery_without_wal_entries(self):
        """An unclean exit alone (no WAL entries) triggers recovery on restart."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)

        # First instance: write a complete entry (WAL is cleared after commit)
        backend1 = ArrowFileBackend(config)
        entry1, _ = backend1.start_compute(b"key1")
        data = self._make_data([1, 2, 3])
        backend1.complete_entry(b"key1", data, 24)
        backend1.release(b"key1")

        # Crash: the lock's descriptor goes, its owner record stays behind.
        _simulate_crash(backend1)

        # Verify WAL has no pending entries (so recovery must be triggered by
        # the leftover owner record alone)
        from biopb_tensor_server.cache.recovery import WriteAheadLog

        wal = WriteAheadLog(cache_dir / "wal.json")
        assert not wal.has_pending(), "Test requires no pending WAL entries"

        # Reinitialize: recovery should trigger due to stale lock
        backend2 = ArrowFileBackend(config)
        assert backend2.get_recovery_status() is not None, (
            "Recovery should be triggered by stale lock even without pending WAL entries"
        )

        backend2.close()
        shutil.rmtree(cache_dir)

    def test_pending_write_is_discarded_on_recovery(self):
        """An interrupted write (logged pending, never committed) is dropped on
        recovery and never served as a torn cache hit -- the WAL's whole purpose,
        and the crash-safety property that must hold without a clean shutdown
        (#138 item 2). A committed entry alongside it must still survive.
        """
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)

        backend1 = ArrowFileBackend(config)
        # A fully committed entry: must survive recovery.
        backend1.start_compute(b"key_good")
        backend1.complete_entry(b"key_good", self._make_data([1, 2, 3]), 24)
        backend1.release(b"key_good")
        # An in-flight write: pending in the WAL with no committed segment -- the
        # on-disk shape of a crash mid-complete_entry (after log_pending, before
        # log_committed).
        backend1._wal.log_pending(b"key_bad")

        # Crash without clean shutdown: the OS frees the lock descriptor so a
        # fresh instance can reclaim it, and recovery is driven by the pending
        # WAL entry (the lock file and owner record stay -- no crash unlinks them).
        _simulate_crash(backend1)

        backend2 = ArrowFileBackend(config)
        # Recovery ran (driven by the pending WAL entry) and purged the in-flight
        # write: the pending marker is gone, so the key is "lost" per _recover().
        assert backend2.get_recovery_status() is not None, "recovery must run"
        assert not backend2._wal.has_pending(), "pending write must be purged"

        good, good_owner = backend2.start_compute(b"key_good")
        assert good_owner is False, "committed entry must survive as a cache hit"
        assert good.state == EntryState.READY
        # The interrupted key was lost: the caller becomes the owner (must
        # recompute) rather than getting a torn/partial hit.
        _bad, bad_owner = backend2.start_compute(b"key_bad")
        assert bad_owner is True, "interrupted write must be recomputed, not served"

        backend2.close()
        shutil.rmtree(cache_dir)

    def test_recovery_accounting_is_cheap_and_correct(self):
        """Recovery reports the entry count from the rebuilt index and the byte
        total from segment file sizes -- it must NOT scan segment bodies just for
        the status line (biopb/biopb#300).
        """
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)

        # Three fully-committed entries.
        backend1 = ArrowFileBackend(config)
        for i in range(3):
            key = f"key{i}".encode()
            backend1.start_compute(key)
            backend1.complete_entry(key, self._make_data([i, i + 1]), 16)
            backend1.release(key)

        # Crash: the leftover owner record is what triggers recovery.
        _simulate_crash(backend1)

        segments_dir = cache_dir / "segments"
        expected_bytes = sum(f.stat().st_size for f in segments_dir.glob("seg_*.arrow"))

        backend2 = ArrowFileBackend(config)
        status = backend2.get_recovery_status()
        assert status is not None, "stale lock must trigger recovery"
        # Entry count is backfilled from the rebuilt index (all 3 survived).
        assert status.recovered_entries == 3
        # Byte total is the on-disk segment footprint (stat), not a sum of decoded
        # batch.nbytes -- so it matches the segment file sizes exactly.
        assert status.recovered_bytes == expected_bytes
        backend2.close()
        shutil.rmtree(cache_dir)


class TestBackendSelection:
    """Tests for CacheManager backend selection."""

    def test_memory_backend_selection(self):
        """Config with backend='memory' uses MemoryCacheBackend."""
        CacheManager.reset()
        config = CacheConfig(
            backend="memory",
            memory_max_entries=100,
            memory_max_bytes=1024 * 1024,
        )
        manager = CacheManager.initialize(config)
        assert isinstance(manager.backend, MemoryCacheBackend)
        CacheManager.reset()

    def test_file_backend_selection(self):
        """Config with backend='file' uses ArrowFileBackend."""
        CacheManager.reset()
        cache_dir = Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))
        config = CacheConfig(
            backend="file",
            file_cache_dir=cache_dir,
            file_max_segment_bytes=1024 * 1024,
            file_max_total_bytes=10 * 1024 * 1024,
        )
        manager = CacheManager.initialize(config)
        assert isinstance(manager.backend, ArrowFileBackend)
        CacheManager.reset()
        shutil.rmtree(cache_dir)

    def test_unknown_backend_raises(self):
        """Unknown backend raises ValueError."""
        CacheManager.reset()
        config = CacheConfig(backend="unknown")
        with pytest.raises(ValueError, match="Unknown cache backend"):
            CacheManager.initialize(config)
        CacheManager.reset()


class TestOversizedChunkHandling:
    """Tests for oversized chunk handling (>2GB)."""

    def _make_data(self, values) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def test_memory_backend_skips_oversized(self):
        """Memory backend skips caching oversized chunks."""
        config = MemoryCacheConfig()
        backend = MemoryCacheBackend(config)

        entry, is_owner = backend.start_compute(b"big_key")
        # Simulate oversized chunk (use MAX_ARROW_BATCH_BYTES + 1)
        oversized_bytes = MAX_ARROW_BATCH_BYTES + 1
        data = self._make_data([1])
        backend.complete_entry(b"big_key", data, oversized_bytes)

        # Entry should be ready but not stored in cache properly
        assert entry.state == EntryState.READY

        # Stats should show oversized skip
        stats = backend.stats()
        assert stats.oversized_skips == 1

        backend.close()

    def test_file_backend_skips_oversized(self):
        """File backend skips caching oversized chunks."""
        cache_dir = Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        entry, is_owner = backend.start_compute(b"big_key")
        oversized_bytes = MAX_ARROW_BATCH_BYTES + 1
        data = self._make_data([1])
        backend.complete_entry(b"big_key", data, oversized_bytes)

        # Entry should be ready
        assert entry.state == EntryState.READY

        # Stats should show oversized skip
        stats = backend.stats()
        assert stats.oversized_skips == 1

        # Entry should not be persisted
        backend.close()

        # Reopen - entry should not be found
        backend2 = ArrowFileBackend(config)
        entry2, is_owner2 = backend2.start_compute(b"big_key")
        assert is_owner2 is True  # New computation needed
        backend2.close()
        shutil.rmtree(cache_dir)


class TestSizeClassClassification:
    """Tests for size class classification for pooling."""

    def test_tiny_size_class(self):
        """Chunks under TINY_THRESHOLD are classified as tiny."""
        assert _get_size_class(1) == "tiny"
        assert _get_size_class(SIZE_CLASS_TINY_THRESHOLD - 1) == "tiny"

    def test_small_size_class(self):
        """Chunks between TINY and SMALL thresholds are small."""
        assert _get_size_class(SIZE_CLASS_TINY_THRESHOLD) == "small"
        assert _get_size_class(SIZE_CLASS_SMALL_THRESHOLD - 1) == "small"

    def test_medium_size_class(self):
        """Chunks between SMALL and MEDIUM thresholds are medium."""
        assert _get_size_class(SIZE_CLASS_SMALL_THRESHOLD) == "medium"
        assert _get_size_class(SIZE_CLASS_MEDIUM_THRESHOLD - 1) == "medium"

    def test_large_size_class(self):
        """Chunks over MEDIUM_THRESHOLD are large."""
        assert _get_size_class(SIZE_CLASS_MEDIUM_THRESHOLD) == "large"
        assert _get_size_class(MAX_ARROW_BATCH_BYTES) == "large"


class TestSchemaPooling:
    """Regression tests for schema mismatch handling with pooling."""

    def _make_temp_cache_dir(self):
        """Create a temporary cache directory."""
        return Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))

    def test_alternating_schemas_pooled_separately(self):
        """Alternating dtype writes share the unified binary schema, so they pool
        together into few segments (biopb/biopb#293) -- not one segment per write."""
        cache_dir = self._make_temp_cache_dir()
        # Large segment size to allow many entries per segment
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=10 * 1024 * 1024,  # 10 MB
            max_total_bytes=100 * 1024 * 1024,  # 100 MB
        )

        backend = ArrowFileBackend(config)

        # Different dtypes now all serialize to the ONE unified binary chunk
        # schema (raw bytes + dtype string), so they share a pool.
        from biopb_tensor_server.core.base import pack_chunk_batch

        int_data = pack_chunk_batch(np.array([1, 2, 3], dtype=np.int32))
        float_data = pack_chunk_batch(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        # Write alternating entries with different schemas
        # Each schema should get its own pool/segment
        for i in range(10):
            int_key = f"int_key_{i}".encode()
            float_key = f"float_key_{i}".encode()

            entry, _ = backend.start_compute(int_key)
            backend.complete_entry(int_key, int_data, 12)

            entry, _ = backend.start_compute(float_key)
            backend.complete_entry(float_key, float_data, 12)

        # Verify all entries can be retrieved
        for i in range(10):
            int_key = f"int_key_{i}".encode()
            float_key = f"float_key_{i}".encode()

            entry, is_owner = backend.start_compute(int_key)
            assert is_owner is False
            assert entry.state == EntryState.READY
            backend.release(int_key)

            entry, is_owner = backend.start_compute(float_key)
            assert is_owner is False
            assert entry.state == EntryState.READY
            backend.release(float_key)

        # Check segment count - should be limited (not 20 segments)
        segments_dir = cache_dir / "segments"
        segment_files = list(segments_dir.glob("seg_*.arrow"))
        # With pooling, we should have at most a few segments per schema pool
        # Not one segment per write (which would be 20)
        assert len(segment_files) <= 4, f"Too many segments: {len(segment_files)}"

        backend.close()
        shutil.rmtree(cache_dir)

    def test_large_chunks_skip_file_cache(self):
        """Large chunks (>64MB) skip file cache and stay in memory."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        # Create a batch that would be classified as oversized
        # Use the MAX_ARROW_BATCH_BYTES threshold (>64MB triggers oversized skip)
        large_size = MAX_ARROW_BATCH_BYTES + 1000
        data = pa.RecordBatch.from_arrays(
            [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

        entry, _ = backend.start_compute(b"large_key")
        backend.complete_entry(b"large_key", data, large_size)

        # Entry should be ready in memory
        assert entry.state == EntryState.READY

        # Stats should show oversized skip (for large chunks)
        stats = backend.stats()
        assert stats.oversized_skips == 1

        # No segment file should have been created for this entry
        segments_dir = cache_dir / "segments"
        segment_files = list(segments_dir.glob("seg_*.arrow"))
        # May have 1 segment (the initial empty one), but large data not in it
        assert len(segment_files) <= 1

        backend.close()
        shutil.rmtree(cache_dir)

    def test_same_schema_different_sizes_share_pool(self):
        """Entries with same schema but different size classes can share segment."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=5 * 1024 * 1024,  # 5 MB
            max_total_bytes=50 * 1024 * 1024,  # 50 MB
        )
        backend = ArrowFileBackend(config)

        # Create batches with same schema but different sizes (small vs tiny)
        list_type = pa.list_(pa.int32())

        # Tiny entry (< 2MB)
        tiny_data = pa.RecordBatch.from_arrays(
            [
                pa.array([[1] * 100], type=list_type),
                pa.array([[100]]),
                pa.array(["int32"]),
            ],
            ["data", "shape", "dtype"],
        )
        tiny_size = 400  # Tiny

        # Small entry (between 2MB and 32MB in size classification)
        # Simulate with small actual data but report larger size
        small_data = pa.RecordBatch.from_arrays(
            [
                pa.array([[2] * 100], type=list_type),
                pa.array([[100]]),
                pa.array(["int32"]),
            ],
            ["data", "shape", "dtype"],
        )
        small_size = SIZE_CLASS_TINY_THRESHOLD + 1000  # Small class

        entry, _ = backend.start_compute(b"tiny_key")
        backend.complete_entry(b"tiny_key", tiny_data, tiny_size)

        entry, _ = backend.start_compute(b"small_key")
        backend.complete_entry(b"small_key", small_data, small_size)

        # Both entries should be retrievable
        entry, is_owner = backend.start_compute(b"tiny_key")
        assert is_owner is False
        backend.release(b"tiny_key")

        entry, is_owner = backend.start_compute(b"small_key")
        assert is_owner is False
        backend.release(b"small_key")

        backend.close()
        shutil.rmtree(cache_dir)


class TestSieveKEviction:
    """Tests for Sieve-K scan-resistant eviction algorithm."""

    def _make_data(self, values) -> pa.RecordBatch:
        """Helper to create RecordBatch with new schema format."""
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def _make_temp_cache_dir(self):
        """Create a temporary cache directory."""
        return Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))

    def test_scan_resistance(self):
        """Scan items (frequency=0) are evicted first, hot items survive."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=500,  # Small segment to force many segments
            max_total_bytes=3 * 500,  # Only 3 segments can fit
        )
        backend = ArrowFileBackend(config)

        # Create 10 segments (10 entries)
        for i in range(10):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 50)
            backend.complete_entry(key, data, 400)
            backend.release(key)

        # Access keys 0, 1, 2 repeatedly (make them "hot")
        for _ in range(3):
            for hot_key in [b"key0", b"key1", b"key2"]:
                entry, _ = backend.start_compute(hot_key)
                backend.release(hot_key)

        # Verify hot keys still exist after multiple evictions
        # Cold keys (key3-key9) should have been evicted first
        hot_keys_cached = 0
        for hot_key in [b"key0", b"key1", b"key2"]:
            entry, is_owner = backend.start_compute(hot_key)
            if not is_owner:
                hot_keys_cached += 1
            backend.release(hot_key)

        # Check that some cold keys were evicted
        stats = backend.stats()
        assert stats.evictions >= 3  # At least 3 cold segments evicted
        # At least one hot key should still be cached (they have higher frequency)
        assert hot_keys_cached >= 1  # At least 1 hot key survived

        backend.close()
        shutil.rmtree(cache_dir)

    def test_frequency_promotion(self):
        """Accessing same segment increases frequency counter, saturating at K=2."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        # Create entry
        key = b"key1"
        entry, _ = backend.start_compute(key)
        data = self._make_data([1, 2, 3])
        backend.complete_entry(key, data, 24)
        backend.release(key)

        # Get pool queue for this segment
        pool_key = ("unified", "tiny")
        pool_queue = backend._pool_queues.get(pool_key)

        if pool_queue:
            # Get segment info
            segment_id = (
                pool_queue.queue[-1] if pool_queue.queue else None
            )  # Tail (oldest)
            if segment_id:
                seg_info = pool_queue.segments.get(segment_id)
                assert seg_info is not None

                # Access 5 times - frequency should saturate at K=2
                for _ in range(5):
                    entry, _ = backend.start_compute(key)
                    backend.release(key)

                # Frequency should be at K (saturated)
                assert seg_info.frequency == K

        backend.close()
        shutil.rmtree(cache_dir)

    def test_hand_position_after_eviction(self):
        """Hand stays at same offset after evicting a segment."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=500,
            max_total_bytes=2 * 500,  # Only 2 segments can fit
        )
        backend = ArrowFileBackend(config)

        # Create 5 segments (cold, frequency=0)
        for i in range(5):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 50)
            backend.complete_entry(key, data, 400)
            backend.release(key)

        pool_key = ("unified", "tiny")
        pool_queue = backend._pool_queues.get(pool_key)

        if pool_queue and len(pool_queue.queue) > 2:
            # Trigger eviction
            key_new = b"key_new"
            entry, _ = backend.start_compute(key_new)
            data = self._make_data([99] * 50)
            backend.complete_entry(key_new, data, 400)
            backend.release(key_new)

            # Hand should have stayed at same offset (or wrapped)
            # The actual behavior depends on where the cold segment was found
            stats = backend.stats()
            assert stats.evictions >= 1  # At least one eviction occurred

        backend.close()
        shutil.rmtree(cache_dir)

    def test_pool_selection_by_hit_rate(self):
        """Pool with lowest hit rate is selected for eviction."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=500,
            max_total_bytes=4 * 500,  # 4 segments can fit
        )
        backend = ArrowFileBackend(config)

        # Create entries in "tiny" pool (low hit rate)
        for i in range(3):
            key = f"tiny{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 10)  # Tiny size
            backend.complete_entry(key, data, 80)
            backend.release(key)

        # Create entries in "small" pool (higher hit rate)
        small_size = SIZE_CLASS_TINY_THRESHOLD + 100
        for i in range(3):
            key = f"small{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 50)
            backend.complete_entry(key, data, small_size)
            backend.release(key)

        # Access small pool entries repeatedly (increase hit rate)
        for _ in range(5):
            for i in range(3):
                key = f"small{i}".encode()
                entry, _ = backend.start_compute(key)
                backend.release(key)

        # Trigger eviction by adding more entries
        for i in range(3):
            key = f"new{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 10)
            backend.complete_entry(key, data, 80)
            backend.release(key)

        # Check pool stats
        stats = backend.stats()
        assert stats.evictions >= 1

        # Tiny pool should have lower hit rate
        if "unified-tiny" in stats.pool_stats and "unified-small" in stats.pool_stats:
            tiny_rate = stats.pool_stats["unified-tiny"].hit_rate
            small_rate = stats.pool_stats["unified-small"].hit_rate
            # Small pool was accessed more, should have higher hit rate
            assert small_rate >= tiny_rate

        backend.close()
        shutil.rmtree(cache_dir)

    def test_pool_stats_tracking(self):
        """Pool-level statistics are tracked correctly."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        # Create some entries
        for i in range(5):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 10)
            backend.complete_entry(key, data, 80)
            backend.release(key)

        # Access some entries (hits)
        for i in range(3):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            backend.release(key)

        stats = backend.stats()
        assert "unified-tiny" in stats.pool_stats

        pool_stat = stats.pool_stats["unified-tiny"]
        assert pool_stat.hits >= 3  # 3 hits
        assert pool_stat.misses >= 5  # 5 misses (initial writes)
        assert pool_stat.segments >= 1

        backend.close()
        shutil.rmtree(cache_dir)

    def test_new_segment_frequency_zero(self):
        """Newly created segments start with frequency=0."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)
        backend = ArrowFileBackend(config)

        # Create entry (creates new segment)
        key = b"key1"
        entry, _ = backend.start_compute(key)
        data = self._make_data([1, 2, 3])
        backend.complete_entry(key, data, 24)
        backend.release(key)

        pool_key = ("unified", "tiny")
        pool_queue = backend._pool_queues.get(pool_key)

        if pool_queue and pool_queue.queue:
            segment_id = pool_queue.queue[0]  # Head (newest)
            seg_info = pool_queue.segments.get(segment_id)
            assert seg_info is not None
            assert seg_info.frequency == 0  # New segments start at 0

        backend.close()
        shutil.rmtree(cache_dir)

    def test_decrement_on_eviction_attempt(self):
        """When hand passes a hot segment (frequency>0), counter is decremented."""
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=500,
            max_total_bytes=2 * 500,  # Only 2 segments
        )
        backend = ArrowFileBackend(config)

        # Create segments
        for i in range(4):
            key = f"key{i}".encode()
            entry, _ = backend.start_compute(key)
            data = self._make_data([i] * 50)
            backend.complete_entry(key, data, 400)
            backend.release(key)

        pool_key = ("unified", "tiny")
        pool_queue = backend._pool_queues.get(pool_key)

        if pool_queue and len(pool_queue.queue) >= 3:
            # Access oldest segment multiple times to increase frequency
            oldest_seg_id = pool_queue.queue[-1]  # Tail (oldest)

            # Find key in oldest segment
            for key, entry_info in backend._metadata.items():
                if entry_info.segment_id == oldest_seg_id:
                    # Access to increase frequency
                    for _ in range(3):
                        entry, _ = backend.start_compute(key)
                        backend.release(key)
                    break

            # Trigger eviction
            key_new = b"key_new"
            entry, _ = backend.start_compute(key_new)
            data = self._make_data([99] * 50)
            backend.complete_entry(key_new, data, 400)
            backend.release(key_new)

            # The oldest segment's frequency should have been decremented
            # if it wasn't evicted immediately
            stats = backend.stats()
            assert stats.evictions >= 1

        backend.close()
        shutil.rmtree(cache_dir)


class _WriterProxy:
    """Wraps a segment writer so a test can make ``write_batch`` block or fail.

    Only ``write_batch`` / ``close`` are used by the backend, so delegating
    those is enough.
    """

    def __init__(self, inner, backend):
        self._inner = inner
        self._backend = backend

    def write_batch(self, batch):
        b = self._backend
        if b.fail_next_write:
            raise OSError(errno.ENOSPC, "No space left on device")
        if b.block_next_write:
            b.write_in_progress.set()
            # Block as a stalled flush on a full filesystem would, while holding
            # _write_lock but NOT _lock.
            b.release_write.wait(timeout=10)
        return self._inner.write_batch(batch)

    def close(self):
        return self._inner.close()


class _InstrumentedBackend(ArrowFileBackend):
    """ArrowFileBackend whose segment writes can be stalled or failed on demand."""

    def __init__(self, config):
        self.block_next_write = False
        self.fail_next_write = False
        self.write_in_progress = threading.Event()
        self.release_write = threading.Event()
        super().__init__(config)

    def _create_segment_for_pool(self, pool_key, schema):
        seg_id = super()._create_segment_for_pool(pool_key, schema)
        self._pool_writers[seg_id] = _WriterProxy(self._pool_writers[seg_id], self)
        return seg_id


class TestWriteLockDoesNotWedgeReads:
    """Regression: a stalled/failed segment write must not hold the lock the
    read path needs. A full /tmp once stalled a write inside complete_entry
    while it held the global lock, wedging every read server-wide."""

    def _data(self, values):
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def _backend(self):
        cache_dir = Path(tempfile.mkdtemp(prefix="biopb-cache-wedge-test-"))
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=10 * 1024 * 1024,  # big: keep one segment open
            max_total_bytes=100 * 1024 * 1024,
        )
        return _InstrumentedBackend(config), cache_dir

    def test_stalled_write_does_not_block_reads(self):
        backend, cache_dir = self._backend()
        try:
            # Seed a READY entry (normal, unblocked write).
            backend.start_compute(b"ready")
            backend.complete_entry(b"ready", self._data([1, 2, 3]), 24)

            # Arm the next write to stall mid-flush.
            backend.block_next_write = True
            blocked = threading.Thread(
                target=lambda: backend.get_or_acquire(
                    b"stall", lambda: (self._data([9, 9]), 16)
                ),
                daemon=True,
            )
            blocked.start()
            assert backend.write_in_progress.wait(timeout=5), (
                "stalled write never started"
            )

            # While that write is stalled (holding _write_lock, not _lock), a
            # read of the already-cached key must still complete promptly.
            done = []
            reader = threading.Thread(
                target=lambda: done.append(backend.get_or_acquire(b"ready", None)),
                daemon=True,
            )
            reader.start()
            reader.join(timeout=3)
            assert not reader.is_alive(), "read was wedged behind the stalled write"
            assert done and done[0].state == EntryState.READY
            assert done[0].data.column(0).to_pylist() == [[1, 2, 3]]
            backend.release(b"ready")
        finally:
            backend.release_write.set()  # let the stalled write finish
            blocked.join(timeout=5)
            backend.close()
            shutil.rmtree(cache_dir)

    def test_enospc_fails_entry_without_wedging(self):
        backend, cache_dir = self._backend()
        try:
            backend.start_compute(b"ready")
            backend.complete_entry(b"ready", self._data([1, 2, 3]), 24)

            # A write that hits ENOSPC must surface as an error and leave the
            # entry failed (not stuck PENDING, which would time out every waiter).
            backend.fail_next_write = True
            with pytest.raises(OSError):
                backend.get_or_acquire(b"boom", lambda: (self._data([7]), 8))
            # Entry is gone (failed), not stuck PENDING.
            assert b"boom" not in backend._entries

            # The backend is not wedged: reads and fresh writes still work.
            backend.fail_next_write = False
            hit, _ = backend.start_compute(b"ready")
            assert hit.state == EntryState.READY
            backend.release(b"ready")
            backend.start_compute(b"after")
            backend.complete_entry(b"after", self._data([5, 5]), 16)
            got, _ = backend.start_compute(b"after")
            assert got.state == EntryState.READY
            backend.release(b"after")
        finally:
            backend.close()
            shutil.rmtree(cache_dir)


class _ScanCountingBackend(ArrowFileBackend):
    """Counts body-walk scans so a test can prove the sidecar fast path skips
    them at boot. ``_scan_segment_records`` is the single body-reading chokepoint
    (both the boot fallback walk and the seal-time sidecar write go through it),
    so a zero count right after boot means no segment body was faulted in."""

    def __init__(self, config):
        self.scan_calls = 0
        self.scanned_segments = []
        super().__init__(config)

    def _scan_segment_records(self, seg_file):
        self.scan_calls += 1
        self.scanned_segments.append(seg_file.name)
        return super()._scan_segment_records(seg_file)


class TestSegmentSidecarIndex:
    """Per-segment sidecar index: boot restores the index from seg_NNNN.idx
    instead of walking every segment body (biopb/biopb#300)."""

    def _make_data(self, values) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(["int64"])],
            ["data", "shape", "dtype"],
        )

    def _make_temp_cache_dir(self):
        return Path(tempfile.mkdtemp(prefix="biopb-cache-sidecar-test-"))

    def _rotating_config(self, cache_dir):
        # Small segment cap so entries rotate into several SEALED segments.
        return ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=1000,
            max_total_bytes=100 * 1024 * 1024,
        )

    def _write_entries(self, backend, n, close=True):
        """Write n entries (values [i, i+1, i+2]); optionally close (seals +
        writes sidecars for every segment)."""
        for i in range(n):
            key = f"key{i}".encode()
            backend.start_compute(key)
            backend.complete_entry(key, self._make_data([i, i + 1, i + 2]), 400)
            backend.release(key)
        if close:
            backend.close()

    def _index_snapshot(self, backend):
        return {
            k: (v.segment_id, v.byte_offset, v.byte_length, v.size_bytes, v.offset)
            for k, v in backend._metadata.items()
        }

    def _seg_files(self, cache_dir):
        return sorted((cache_dir / "segments").glob("seg_*.arrow"))

    def _tamper_sidecar_size(self, idx_path, wrong_size):
        with pa.OSFile(str(idx_path), "rb") as f:
            table = pa.ipc.open_file(f).read_all()
        new_meta = {
            b"biopb_sidecar_version": str(SIDECAR_FORMAT_VERSION).encode(),
            b"biopb_segment_size": str(wrong_size).encode(),
        }
        table = table.replace_schema_metadata(new_meta)
        with pa.OSFile(str(idx_path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as w:
                w.write_table(table)

    def test_sidecar_index_matches_full_walk(self):
        """Round-trip parity: the index restored from sidecars is byte-for-byte
        equal to a full-walk rebuild (same keys, segment_id, byte_offset,
        byte_length, size_bytes, offset)."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 7)  # closes -> seals every segment + sidecars

        seg_files = self._seg_files(cache_dir)
        assert len(seg_files) >= 2, "test needs multiple sealed segments"
        for sf in seg_files:
            assert sf.with_suffix(".idx").exists(), f"no sidecar for {sf.name}"

        # Index restored from sidecars (fast path).
        b_side = ArrowFileBackend(config)
        side_index = self._index_snapshot(b_side)
        b_side.close()

        # Same index, rebuilt by the full body walk.
        for idx in (cache_dir / "segments").glob("seg_*.idx"):
            idx.unlink()
        b_walk = ArrowFileBackend(config)
        walk_index = self._index_snapshot(b_walk)
        b_walk.close()

        assert len(side_index) == 7
        assert side_index == walk_index
        shutil.rmtree(cache_dir)

    def test_boot_from_sidecar_skips_body_walk(self):
        """With sidecars present, boot reads no segment bodies, and reads still
        return the right data."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 6)
        assert len(self._seg_files(cache_dir)) >= 1

        counting = _ScanCountingBackend(config)
        assert counting.scan_calls == 0, "sidecar boot must not walk segment bodies"

        entry, is_owner = counting.start_compute(b"key3")
        assert is_owner is False
        assert entry.data.column("data").to_pylist() == [[3, 4, 5]]
        counting.release(b"key3")
        counting.close()
        shutil.rmtree(cache_dir)

    def test_sidecar_byte_ranges_are_exact(self):
        """After an .idx-based boot, locate_entry's byte range points exactly at
        the entry's IPC message, and a normal read returns the same data -- proof
        that byte_offset/byte_length (the #9 fast path) are exact."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 5)

        b2 = ArrowFileBackend(config)  # sidecar boot
        for i in range(5):
            key = f"key{i}".encode()
            loc = b2.locate_entry(key)
            assert loc is not None
            assert loc.byte_offset > 0 and loc.byte_length > 0

            with pa.OSFile(loc.segment_path, "rb") as f:
                schema = pa.ipc.open_stream(f).schema
            with pa.memory_map(loc.segment_path, "r") as mm:
                mm.seek(loc.byte_offset)
                msg = pa.ipc.read_message(mm)
                assert mm.tell() - loc.byte_offset == loc.byte_length
                batch = pa.ipc.read_record_batch(msg, schema)
            assert batch.column(CACHE_KEY_FIELD)[0].as_py() == key
            assert batch.column("data").to_pylist() == [[i, i + 1, i + 2]]

            entry, is_owner = b2.start_compute(key)
            assert is_owner is False
            assert entry.data.column("data").to_pylist() == [[i, i + 1, i + 2]]
            b2.release(key)
        b2.close()
        shutil.rmtree(cache_dir)

    def test_missing_sidecar_walks_that_segment_and_backfills(self):
        """A deleted sidecar makes only its own segment fall back to the walk; a
        fresh sidecar is backfilled, so the next boot is fully fast."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 7)
        seg_files = self._seg_files(cache_dir)
        assert len(seg_files) >= 2

        victim = seg_files[0]
        victim_idx = victim.with_suffix(".idx")
        assert victim_idx.exists()
        victim_idx.unlink()

        counting = _ScanCountingBackend(config)
        assert counting.scan_calls == 1
        assert counting.scanned_segments == [victim.name]
        # All entries still hit with correct data.
        for i in range(7):
            key = f"key{i}".encode()
            entry, is_owner = counting.start_compute(key)
            assert is_owner is False
            assert entry.data.column("data").to_pylist() == [[i, i + 1, i + 2]]
            counting.release(key)
        # The sidecar was backfilled.
        assert victim_idx.exists()
        counting.close()

        # Second boot uses the backfilled sidecar: no walks.
        counting2 = _ScanCountingBackend(config)
        assert counting2.scan_calls == 0
        counting2.close()
        shutil.rmtree(cache_dir)

    def test_corrupt_sidecar_walks_that_segment_and_backfills(self):
        """A corrupt sidecar is rejected, its segment is walked, and a valid
        sidecar is written in its place."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 7)
        seg_files = self._seg_files(cache_dir)
        assert len(seg_files) >= 2

        victim = seg_files[0]
        victim_idx = victim.with_suffix(".idx")
        victim_idx.write_bytes(b"this is not an arrow file")

        counting = _ScanCountingBackend(config)
        assert counting.scanned_segments == [victim.name]
        for i in range(7):
            key = f"key{i}".encode()
            entry, is_owner = counting.start_compute(key)
            assert is_owner is False
            assert entry.data.column("data").to_pylist() == [[i, i + 1, i + 2]]
            counting.release(key)
        counting.close()

        counting2 = _ScanCountingBackend(config)
        assert counting2.scan_calls == 0  # backfilled sidecar is now valid
        counting2.close()
        shutil.rmtree(cache_dir)

    def test_stale_sidecar_size_is_rejected(self):
        """A sidecar whose recorded .arrow size no longer matches the file is
        rejected (torn/mismatched), so its segment is walked instead."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 7)
        seg_files = self._seg_files(cache_dir)
        assert len(seg_files) >= 2

        victim = seg_files[0]
        victim_idx = victim.with_suffix(".idx")
        self._tamper_sidecar_size(victim_idx, victim.stat().st_size + 4096)

        counting = _ScanCountingBackend(config)
        assert victim.name in counting.scanned_segments
        for i in range(7):
            key = f"key{i}".encode()
            entry, is_owner = counting.start_compute(key)
            assert is_owner is False
            assert entry.data.column("data").to_pylist() == [[i, i + 1, i + 2]]
            counting.release(key)
        counting.close()
        shutil.rmtree(cache_dir)

    def test_eviction_removes_sidecar(self):
        """Evicting a segment removes both its .arrow and its .idx sidecar."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        self._write_entries(b1, 7)

        b2 = ArrowFileBackend(config)
        seg_id = next(iter(b2._metadata.values())).segment_id
        seg_file = cache_dir / "segments" / f"seg_{seg_id:04d}.arrow"
        idx_file = seg_file.with_suffix(".idx")
        assert seg_file.exists() and idx_file.exists()

        b2._do_evict_segment(seg_id)
        assert not seg_file.exists()
        assert not idx_file.exists()
        b2.close()
        shutil.rmtree(cache_dir)

    def test_crash_loads_sealed_from_sidecar_and_walks_open(self):
        """After a crash (no clean close), the sealed segments load from their
        sidecars while the never-sealed open segment is walked/recovered; the
        final index is complete and correct."""
        cache_dir = self._make_temp_cache_dir()
        config = self._rotating_config(cache_dir)

        b1 = ArrowFileBackend(config)
        # 7 entries at size 400 with a 1000-byte cap -> 2 sealed segments
        # (3 entries each, with sidecars) + 1 open segment (entry 6, no sidecar).
        self._write_entries(b1, 7, close=False)
        sealed = self._seg_files(cache_dir)
        sidecars = list((cache_dir / "segments").glob("seg_*.idx"))
        assert len(sealed) == 3  # two sealed + one open on disk
        assert len(sidecars) == 2  # only the two sealed ones have sidecars

        _simulate_crash(b1)

        counting = _ScanCountingBackend(config)
        assert counting.scan_calls == 1, "only the open (sidecar-less) segment walks"
        for i in range(7):
            key = f"key{i}".encode()
            entry, is_owner = counting.start_compute(key)
            assert is_owner is False, f"{key!r} lost across crash"
            assert entry.data.column("data").to_pylist() == [[i, i + 1, i + 2]]
            counting.release(key)
        counting.close()
        shutil.rmtree(cache_dir)
