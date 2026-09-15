"""Thread-safe cache system for tensor-server computed virtual chunks.

The cache system stores computed scaled-down chunk tensors to avoid
repeated expensive operations. Uses future/promise pattern for
safe concurrent computation in Flight server context.

Modules:
    - types: the value types (entries, retention, stats)
    - manager: the singleton the server holds
    - file_backend: the live, lock-disciplined serving backend
    - bootstrap: the cache directory's layout and its single-threaded boot path
    - segment_index: the on-disk ``.arrow`` / ``.idx`` formats
    - recovery: WAL, process lock, and the Sieve-K pool bookkeeping

Exports:
    - CacheManager: Singleton manager for cache operations
    - CacheEntry: Cached data with state and ref_count
    - CacheStats: Cache statistics for monitoring
    - ChunkLocation: On-disk byte range of a cached chunk (localhost handoff)
    - EntryState: PENDING, READY, or ERROR states
    - RetentionClass: what a miss costs ("cheap" | "normal" | "pinned")
    - ArrowFileBackend: Persistent Arrow file cache backend
    - ArrowFileConfig: Configuration for file backend
    - RecoveryStatus: Result of crash recovery
    - MAX_ARROW_BATCH_BYTES: Maximum batch size threshold for oversized chunk handling
"""

from biopb_tensor_server.cache.bootstrap import CACHE_FILE_FORMAT_VERSION
from biopb_tensor_server.cache.file_backend import ArrowFileBackend, ArrowFileConfig
from biopb_tensor_server.cache.manager import CacheManager
from biopb_tensor_server.cache.recovery import RecoveryStatus
from biopb_tensor_server.cache.types import (
    MAX_ARROW_BATCH_BYTES,
    CacheEntry,
    CacheStats,
    ChunkLocation,
    EntryState,
    PoolStats,
    RetentionClass,
)

__all__ = [
    "CacheEntry",
    "CacheManager",
    "CacheStats",
    "ChunkLocation",
    "EntryState",
    "ArrowFileBackend",
    "ArrowFileConfig",
    "CACHE_FILE_FORMAT_VERSION",
    "RecoveryStatus",
    "MAX_ARROW_BATCH_BYTES",
    "PoolStats",
    "RetentionClass",
]
