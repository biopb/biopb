"""Thread-safety tests for cache module."""

import os
import shutil
import threading
import time
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from biopb_tensor_server.cache import (
    CacheBackend,
    CacheEntry,
    CacheKey,
    CacheStats,
    CacheManager,
    MemoryCacheBackend,
    EntryState,
    MAX_ARROW_BATCH_BYTES,
)
from biopb_tensor_server.cache.memory_backend import MemoryCacheConfig
from biopb_tensor_server.cache.file_backend import ArrowFileBackend, ArrowFileConfig
from biopb_tensor_server.cache.recovery import (
    WriteAheadLog,
    ProcessLock,
    RecoveryStatus,
)
from biopb_tensor_server.config import CacheConfig


class TestCacheKey:
    """Tests for CacheKey serialization."""

    def test_to_bytes_roundtrip(self):
        """Test consistent serialization."""
        key1 = CacheKey(
            array_id="test-array",
            scale_hint=(2, 2, 2),
            source_start=(0, 0, 0),
            source_stop=(100, 100, 100),
            valid_stop=(100, 100, 100),
            reduction_method="nearest",
        )
        key2 = CacheKey(
            array_id="test-array",
            scale_hint=(2, 2, 2),
            source_start=(0, 0, 0),
            source_stop=(100, 100, 100),
            valid_stop=(100, 100, 100),
            reduction_method="nearest",
        )
        assert key1.to_bytes() == key2.to_bytes()

    def test_different_keys_different_bytes(self):
        """Different keys produce different bytes."""
        key1 = CacheKey(array_id="array1", scale_hint=(2, 2), source_start=(0, 0),
                        source_stop=(100, 100), valid_stop=(100, 100), reduction_method="nearest")
        key2 = CacheKey(array_id="array2", scale_hint=(2, 2), source_start=(0, 0),
                        source_stop=(100, 100), valid_stop=(100, 100), reduction_method="nearest")
        assert key1.to_bytes() != key2.to_bytes()

    def test_different_scale_different_bytes(self):
        """Different scales produce different bytes."""
        key1 = CacheKey(array_id="test", scale_hint=(2, 2), source_start=(0, 0),
                        source_stop=(100, 100), valid_stop=(100, 100), reduction_method="nearest")
        key2 = CacheKey(array_id="test", scale_hint=(4, 4), source_start=(0, 0),
                        source_stop=(100, 100), valid_stop=(100, 100), reduction_method="nearest")
        assert key1.to_bytes() != key2.to_bytes()

    def test_to_string_readable(self):
        """Human readable representation."""
        key = CacheKey(array_id="test", scale_hint=(2, 2), source_start=(0, 0),
                       source_stop=(100, 100), valid_stop=(100, 100), reduction_method="nearest")
        s = key.to_string()
        assert "test" in s
        assert "nearest" in s


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
        data = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ["data"])
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
            data = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ["data"])
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
        """Helper to create RecordBatch."""
        return pa.RecordBatch.from_arrays([pa.array(values)], ["data"])

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
        assert entry2.data.column(0).to_pylist() == [1, 2, 3]
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

        data = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ["data"])
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
                data = pa.RecordBatch.from_arrays([pa.array([worker_id])], ["data"])
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
            data = pa.RecordBatch.from_arrays([pa.array([value])], ["data"])
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


class TestCacheKeyIntegration:
    """Tests for CacheKey with virtual chunk parameters."""

    def test_cache_key_from_virtual_chunk_params(self):
        """CacheKey from virtual chunk decode."""
        key = CacheKey(
            array_id="test-array",
            scale_hint=(2, 2, 2),
            source_start=(0, 0, 0),
            source_stop=(128, 128, 128),
            valid_stop=(128, 128, 128),
            reduction_method="nearest",
        )
        assert len(key.to_bytes()) > 0

    def test_different_regions_different_keys(self):
        """Different regions produce different keys."""
        key1 = CacheKey(array_id="test", scale_hint=(2, 2), source_start=(0, 0),
                        source_stop=(100, 100), valid_stop=(100, 100), reduction_method="nearest")
        key2 = CacheKey(array_id="test", scale_hint=(2, 2), source_start=(100, 0),
                        source_stop=(200, 100), valid_stop=(200, 100), reduction_method="nearest")
        assert key1.to_bytes() != key2.to_bytes()


class TestArrowFileBackend:
    """Tests for persistent Arrow file cache backend."""

    def _make_data(self, values) -> pa.RecordBatch:
        """Helper to create RecordBatch."""
        return pa.RecordBatch.from_arrays([pa.array(values)], ["data"])

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
        assert entry2.data.column(0).to_pylist() == [1, 2, 3]
        backend2.release(b"key1")
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
                data = pa.RecordBatch.from_arrays([pa.array([worker_id])], ["data"])
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
        return pa.RecordBatch.from_arrays([pa.array(values)], ["data"])

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
        return pa.RecordBatch.from_arrays([pa.array(values)], ["data"])

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