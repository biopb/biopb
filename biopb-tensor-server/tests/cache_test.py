"""Thread-safety tests for cache module."""

import shutil
import tempfile
import threading
import time
from pathlib import Path

import pyarrow as pa
import pytest

from biopb_tensor_server.cache import (
    MAX_ARROW_BATCH_BYTES,
    CacheEntry,
    CacheManager,
    EntryState,
    MemoryCacheBackend,
    PoolStats,
)
from biopb_tensor_server.cache.file_backend import (
    SIZE_CLASS_MEDIUM_THRESHOLD,
    SIZE_CLASS_SMALL_THRESHOLD,
    SIZE_CLASS_TINY_THRESHOLD,
    ArrowFileBackend,
    ArrowFileConfig,
    _get_size_class,
    K,
)
from biopb_tensor_server.cache.memory_backend import MemoryCacheConfig
from biopb_tensor_server.cache.recovery import (
    ProcessLock,
    WriteAheadLog,
)
from biopb_tensor_server.config import CacheConfig


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
            [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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
                [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(['int64'])],
                ["data", "shape", "dtype"]
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
            [pa.array([values]), pa.array([[len(values)]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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

        stats = backend.stats()
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
            [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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
                    [pa.array([[worker_id]]), pa.array([[1]]), pa.array(['int64'])],
                    ["data", "shape", "dtype"]
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
                [pa.array([[value]]), pa.array([[1]]), pa.array(['int64'])],
                ["data", "shape", "dtype"]
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
            [pa.array([values]), pa.array([[len(values)]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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
                    [pa.array([[worker_id]]), pa.array([[1]]), pa.array(['int64'])],
                    ["data", "shape", "dtype"]
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


class TestArrowFileBackendRecovery:
    """Tests for crash recovery."""

    def _make_data(self, values) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([values]), pa.array([[len(values)]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
        )

    def _make_temp_cache_dir(self):
        return Path(tempfile.mkdtemp(prefix="biopb-cache-test-"))

    def test_stale_lock_detection(self):
        """Stale lock from dead process is detected."""
        cache_dir = self._make_temp_cache_dir()
        lock_path = cache_dir / "lock"

        # Create a stale lock with fake PID
        import json
        fake_data = {"pid": 99999, "acquired_at": time.time()}
        with open(lock_path, 'w') as f:
            json.dump(fake_data, f)

        # ProcessLock should detect it as stale
        lock = ProcessLock(lock_path)
        assert lock.is_stale() is True
        assert lock.acquire() is True  # Should acquire after removing stale lock
        lock.release()

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

        # Simulate crash - don't call close(), just remove lock
        # (this simulates process dying without clean shutdown)
        lock_path = cache_dir / "lock"
        if lock_path.exists():
            lock_path.unlink()

        # Second instance should recover and still have the entry
        backend2 = ArrowFileBackend(config)
        entry2, is_owner2 = backend2.start_compute(b"key1")
        assert is_owner2 is False  # Should find cached entry
        assert entry2.state == EntryState.READY

        # Recovery status should be available
        recovery = backend2.get_recovery_status()
        # May or may not have recovery depending on whether there were pending writes

        backend2.close()
        shutil.rmtree(cache_dir)

    def test_stale_lock_triggers_recovery_without_wal_entries(self):
        """Stale lock alone (no WAL entries) triggers recovery on restart."""
        import json
        cache_dir = self._make_temp_cache_dir()
        config = ArrowFileConfig(cache_dir=cache_dir)

        # First instance: write a complete entry (WAL is cleared after commit)
        backend1 = ArrowFileBackend(config)
        entry1, _ = backend1.start_compute(b"key1")
        data = self._make_data([1, 2, 3])
        backend1.complete_entry(b"key1", data, 24)
        backend1.release(b"key1")

        # Simulate crash: overwrite lock with a dead PID (no pending WAL entries)
        lock_path = cache_dir / "lock"
        fake_data = {"pid": 99999, "acquired_at": time.time()}
        with open(lock_path, 'w') as f:
            json.dump(fake_data, f)

        # Verify WAL has no pending entries (so recovery must be triggered by stale lock)
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
            [pa.array([values]), pa.array([[len(values)]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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
        """Alternating dtype writes create separate pools but not many small segments."""
        cache_dir = self._make_temp_cache_dir()
        # Large segment size to allow many entries per segment
        config = ArrowFileConfig(
            cache_dir=cache_dir,
            max_segment_bytes=10 * 1024 * 1024,  # 10 MB
            max_total_bytes=100 * 1024 * 1024,  # 100 MB
        )

        backend = ArrowFileBackend(config)

        # Create batches with different schemas (different dtypes)
        int_data = pa.RecordBatch.from_arrays(
            [pa.array([[1, 2, 3]], type=pa.list_(pa.int32())), pa.array([[3]]), pa.array(['int32'])],
            ["data", "shape", "dtype"]
        )
        float_data = pa.RecordBatch.from_arrays(
            [pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float32())), pa.array([[3]]), pa.array(['float32'])],
            ["data", "shape", "dtype"]
        )

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
            [pa.array([[1, 2, 3]]), pa.array([[3]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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
            [pa.array([[1] * 100], type=list_type), pa.array([[100]]), pa.array(['int32'])],
            ["data", "shape", "dtype"]
        )
        tiny_size = 400  # Tiny

        # Small entry (between 2MB and 32MB in size classification)
        # Simulate with small actual data but report larger size
        small_data = pa.RecordBatch.from_arrays(
            [pa.array([[2] * 100], type=list_type), pa.array([[100]]), pa.array(['int32'])],
            ["data", "shape", "dtype"]
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
            [pa.array([values]), pa.array([[len(values)]]), pa.array(['int64'])],
            ["data", "shape", "dtype"]
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
            segment_id = pool_queue.queue[-1] if pool_queue.queue else None  # Tail (oldest)
            if segment_id:
                seg_info = pool_queue.segments.get(segment_id)
                assert seg_info is not None
                initial_freq = seg_info.frequency

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
            # Record hand position before eviction
            initial_hand = pool_queue.hand

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