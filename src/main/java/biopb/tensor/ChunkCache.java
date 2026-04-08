package biopb.tensor;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Cache for tensor chunks using soft references for memory management.
 *
 * This cache stores chunk data with soft references, allowing the JVM to
 * reclaim memory under pressure. It also tracks approximate memory usage.
 */
public class ChunkCache {

    private final Map<String, SoftReference<CacheEntry>> cache;
    private final AtomicLong currentSize;
    private final long maxSize;

    /**
     * Create a new ChunkCache.
     *
     * @param maxBytes Maximum cache size in bytes
     */
    public ChunkCache(long maxBytes) {
        this.cache = new HashMap<>();
        this.currentSize = new AtomicLong(0);
        this.maxSize = maxBytes;
    }

    /**
     * Get a cached chunk.
     *
     * @param chunkId The chunk identifier
     * @return The cached data, or null if not cached
     */
    public Object get(byte[] chunkId) {
        String key = bytesToHex(chunkId);
        SoftReference<CacheEntry> ref = cache.get(key);
        if (ref == null) {
            return null;
        }
        CacheEntry entry = ref.get();
        if (entry == null) {
            // Reference was cleared by GC
            cache.remove(key);
            return null;
        }
        return entry.data;
    }

    /**
     * Put a chunk in the cache.
     *
     * @param chunkId The chunk identifier
     * @param data The chunk data
     * @param size Approximate size in bytes
     */
    public void put(byte[] chunkId, Object data, long size) {
        String key = bytesToHex(chunkId);
        CacheEntry entry = new CacheEntry(data, size);

        // Simple eviction: remove entries if over limit
        while (currentSize.get() + size > maxSize && !cache.isEmpty()) {
            // Remove oldest entry (simple LRU approximation)
            String oldestKey = cache.keySet().iterator().next();
            SoftReference<CacheEntry> oldRef = cache.remove(oldestKey);
            if (oldRef != null) {
                CacheEntry oldEntry = oldRef.get();
                if (oldEntry != null) {
                    currentSize.addAndGet(-oldEntry.size);
                }
            }
        }

        cache.put(key, new SoftReference<>(entry));
        currentSize.addAndGet(size);
    }

    /**
     * Get current cache size.
     *
     * @return Approximate size in bytes
     */
    public long getCurrentSize() {
        return currentSize.get();
    }

    /**
     * Get maximum cache size.
     *
     * @return Maximum size in bytes
     */
    public long getMaxSize() {
        return maxSize;
    }

    /**
     * Clear the cache.
     */
    public void clear() {
        cache.clear();
        currentSize.set(0);
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    private static class CacheEntry {
        final Object data;
        final long size;

        CacheEntry(Object data, long size) {
            this.data = data;
            this.size = size;
        }
    }
}