"""Thread-safe cache system for tensor-server computed virtual chunks.

The cache system stores computed scaled-down chunk tensors to avoid
repeated expensive operations. Uses future/promise pattern for
safe concurrent computation in Flight server context.

Exports:
    - CacheManager: Singleton manager for cache operations
    - CacheBackend: Abstract interface for cache storage
    - CacheEntry: Cached data with state and ref_count
    - CacheStats: Cache statistics for monitoring
    - ChunkLocation: On-disk byte range of a cached chunk (localhost handoff)
    - EntryState: PENDING, READY, or ERROR states
    - MemoryCacheBackend: In-memory LRU cache backend
    - MemoryCacheConfig: Configuration for memory backend
    - ArrowFileBackend: Persistent Arrow file cache backend
    - ArrowFileConfig: Configuration for file backend
    - RecoveryStatus: Result of crash recovery
    - MAX_ARROW_BATCH_BYTES: Maximum batch size threshold for oversized chunk handling
"""

from biopb_tensor_server.cache.base import (
    MAX_ARROW_BATCH_BYTES,
    CacheBackend,
    CacheEntry,
    CacheStats,
    ChunkLocation,
    EntryState,
    PoolStats,
)
from biopb_tensor_server.cache.file_backend import (
    CACHE_FILE_FORMAT_VERSION,
    ArrowFileBackend,
    ArrowFileConfig,
)
from biopb_tensor_server.cache.manager import CacheManager
from biopb_tensor_server.cache.memory_backend import (
    MemoryCacheBackend,
    MemoryCacheConfig,
)
from biopb_tensor_server.cache.recovery import RecoveryStatus

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheManager",
    "CacheStats",
    "ChunkLocation",
    "EntryState",
    "MemoryCacheBackend",
    "MemoryCacheConfig",
    "ArrowFileBackend",
    "ArrowFileConfig",
    "CACHE_FILE_FORMAT_VERSION",
    "RecoveryStatus",
    "MAX_ARROW_BATCH_BYTES",
    "PoolStats",
]
