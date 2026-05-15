package biopb.tensor;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.grpc.CredentialCallOption;
import org.apache.arrow.memory.BufferAllocator;

/**
 * Thread-local connection pool for TensorFlightClient connections.
 *
 * This pool is used by SerializableTensorImg to reconstruct connections
 * after deserialization. Each thread maintains its own FlightClient for
 * lock-free read access, with automatic cleanup on thread death or JVM shutdown.
 *
 * Design mirrors Python's thread-local connection pool with:
 * - Per-thread FlightClient storage (lock-free reads)
 * - Connection registry for cleanup
 * - Thread death eviction
 * - Shutdown hook for resource cleanup
 */
public final class TensorConnectionPool {

    // Per-thread FlightClient storage (lock-free read)
    private static final ThreadLocal<Map<ConnectionKey, PooledConnection>> THREAD_LOCAL_POOL =
        ThreadLocal.withInitial(HashMap::new);

    // Shared cache for cross-thread cache hits (optional, for imglib2 integration)
    private static final ConcurrentHashMap<ConnectionKey, SoftReference<SharedCache>> SHARED_CACHE_POOL =
        new ConcurrentHashMap<>();

    // Connection registry for cleanup: threadId -> connections
    private static final ConcurrentHashMap<Long, Map<ConnectionKey, PooledConnection>> CONNECTION_REGISTRY =
        new ConcurrentHashMap<>();

    // Lock for registry operations
    private static final Object REGISTRY_LOCK = new Object();

    // Counter for eviction scheduling (evict every N accesses)
    private static final AtomicLong accessCounter = new AtomicLong(0);
    private static final int EVICTION_INTERVAL = 100;

    // Private constructor - all methods are static
    private TensorConnectionPool() {}

    static {
        // Register shutdown hook for cleanup
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            synchronized (REGISTRY_LOCK) {
                for (Map<ConnectionKey, PooledConnection> connections : CONNECTION_REGISTRY.values()) {
                    for (PooledConnection conn : connections.values()) {
                        closeConnectionQuietly(conn);
                    }
                }
                CONNECTION_REGISTRY.clear();
                SHARED_CACHE_POOL.clear();
            }
        }, "TensorConnectionPool-shutdown"));
    }

    /**
     * Get or create a pooled connection for the given location and token.
     *
     * Uses thread-local storage for lock-free reads. New connections are
     * registered for cleanup when the thread dies or JVM shuts down.
     *
     * @param location Flight server location
     * @param token Bearer token for authentication (null disables auth)
     * @param allocator Buffer allocator for the connection
     * @return PooledConnection containing FlightClient and auth option
     */
    public static PooledConnection getConnection(Location location, String token, BufferAllocator allocator) {
        ConnectionKey key = new ConnectionKey(location, token);

        // Fast path: thread-local lookup (no lock)
        Map<ConnectionKey, PooledConnection> localPool = THREAD_LOCAL_POOL.get();
        PooledConnection conn = localPool.get(key);
        if (conn != null) {
            return conn;
        }

        // Slow path: create new connection
        conn = createConnection(key, allocator);
        localPool.put(key, conn);

        // Register for cleanup
        long threadId = Thread.currentThread().getId();
        synchronized (REGISTRY_LOCK) {
            // Periodically evict dead threads
            if (accessCounter.incrementAndGet() % EVICTION_INTERVAL == 0) {
                evictDeadThreads();
            }
            CONNECTION_REGISTRY.computeIfAbsent(threadId, k -> new HashMap<>())
                .put(key, conn);
        }

        return conn;
    }

    private static PooledConnection createConnection(ConnectionKey key, BufferAllocator allocator) {
        FlightClient client = FlightClient.builder(allocator, key.location).build();
        CredentialCallOption authOption = key.token != null && !key.token.isEmpty()
            ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + key.token))
            : null;
        return new PooledConnection(client, authOption);
    }

    /**
     * Close a specific connection for the given location and token.
     *
     * Removes the connection from the current thread's pool and the registry.
     *
     * @param location Flight server location
     * @param token Bearer token for authentication
     */
    public static void closeConnection(Location location, String token) {
        ConnectionKey key = new ConnectionKey(location, token);

        // Remove from thread-local
        Map<ConnectionKey, PooledConnection> localPool = THREAD_LOCAL_POOL.get();
        PooledConnection conn = localPool.remove(key);

        // Remove from registry
        synchronized (REGISTRY_LOCK) {
            long threadId = Thread.currentThread().getId();
            Map<ConnectionKey, PooledConnection> registry = CONNECTION_REGISTRY.get(threadId);
            if (registry != null) {
                registry.remove(key);
            }
        }

        // Close the connection
        if (conn != null) {
            closeConnectionQuietly(conn);
        }
    }

    /**
     * Evict connections from dead threads.
     * Must be called with REGISTRY_LOCK held.
     */
    private static void evictDeadThreads() {
        Set<Long> deadThreads = new HashSet<>();
        for (Long threadId : CONNECTION_REGISTRY.keySet()) {
            if (!isThreadAlive(threadId)) {
                deadThreads.add(threadId);
            }
        }

        for (Long threadId : deadThreads) {
            Map<ConnectionKey, PooledConnection> connections = CONNECTION_REGISTRY.remove(threadId);
            if (connections != null) {
                for (PooledConnection conn : connections.values()) {
                    closeConnectionQuietly(conn);
                }
            }
        }
    }

    /**
     * Check if a thread with the given ID is still alive.
     */
    private static boolean isThreadAlive(long threadId) {
        for (Thread thread : Thread.getAllStackTraces().keySet()) {
            if (thread.getId() == threadId) {
                return thread.isAlive();
            }
        }
        return false;
    }

    /**
     * Close a connection, ignoring any exceptions.
     */
    private static void closeConnectionQuietly(PooledConnection conn) {
        try {
            conn.getClient().close();
        } catch (Exception e) {
            // Ignore close failures
        }
    }

    /**
     * Get or create a shared cache for the given connection key.
     *
     * Shared caches allow cross-thread cache hits for read-heavy workloads.
     *
     * @param key Connection key
     * @param maxBytes Maximum bytes to cache
     * @return SharedCache for the connection
     */
    public static SharedCache getSharedCache(ConnectionKey key, long maxBytes) {
        SoftReference<SharedCache> ref = SHARED_CACHE_POOL.get(key);
        SharedCache cache = ref != null ? ref.get() : null;
        if (cache == null) {
            cache = new SharedCache(maxBytes);
            SHARED_CACHE_POOL.put(key, new SoftReference<>(cache));
        }
        return cache;
    }

    /**
     * Get the total connection count across all threads (for testing/monitoring).
     */
    public static int getConnectionCount() {
        synchronized (REGISTRY_LOCK) {
            int count = 0;
            for (Map<ConnectionKey, PooledConnection> connections : CONNECTION_REGISTRY.values()) {
                count += connections.size();
            }
            return count;
        }
    }

    /**
     * Get the thread count with active connections (for testing/monitoring).
     */
    public static int getThreadCount() {
        synchronized (REGISTRY_LOCK) {
            return CONNECTION_REGISTRY.size();
        }
    }

    /**
     * Clear all pooled connections (for testing).
     */
    public static void clear() {
        THREAD_LOCAL_POOL.remove();
        synchronized (REGISTRY_LOCK) {
            for (Map<ConnectionKey, PooledConnection> connections : CONNECTION_REGISTRY.values()) {
                for (PooledConnection conn : connections.values()) {
                    closeConnectionQuietly(conn);
                }
            }
            CONNECTION_REGISTRY.clear();
            SHARED_CACHE_POOL.clear();
        }
    }

    /**
     * Key for connection lookup based on location and token.
     */
    static final class ConnectionKey {
        final Location location;
        final String token;
        final int hashCode;

        ConnectionKey(Location location, String token) {
            this.location = location;
            this.token = token;
            this.hashCode = Objects.hash(location.toString(), token);
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof ConnectionKey)) return false;
            ConnectionKey other = (ConnectionKey) obj;
            return Objects.equals(location.toString(), other.location.toString())
                && Objects.equals(token, other.token);
        }

        @Override
        public int hashCode() {
            return hashCode;
        }
    }

    /**
     * A pooled connection containing FlightClient and authentication option.
     */
    public static final class PooledConnection {
        private final FlightClient client;
        private final CredentialCallOption authOption;
        private final long creationTime;

        PooledConnection(FlightClient client, CredentialCallOption authOption) {
            this.client = client;
            this.authOption = authOption;
            this.creationTime = System.currentTimeMillis();
        }

        public FlightClient getClient() {
            return client;
        }

        public CredentialCallOption getAuthOption() {
            return authOption;
        }

        public long getCreationTime() {
            return creationTime;
        }
    }

    /**
     * Shared cache for cross-thread cache hits.
     *
     * Uses soft references to allow GC under memory pressure.
     */
    public static final class SharedCache {
        private final Map<String, SoftReference<Object>> cache = new ConcurrentHashMap<>();
        private final AtomicLong totalBytes = new AtomicLong(0);
        private final long maxBytes;

        SharedCache(long maxBytes) {
            this.maxBytes = maxBytes;
        }

        public Object get(String key) {
            SoftReference<Object> ref = cache.get(key);
            return ref != null ? ref.get() : null;
        }

        public void put(String key, Object value, long bytes) {
            // Simple eviction: if over budget, clear old entries
            if (totalBytes.get() + bytes > maxBytes && totalBytes.get() > 0) {
                synchronized (this) {
                    cache.clear();
                    totalBytes.set(0);
                }
            }
            cache.put(key, new SoftReference<>(value));
            totalBytes.addAndGet(bytes);
        }

        public void clear() {
            synchronized (this) {
                cache.clear();
                totalBytes.set(0);
            }
        }

        public long getTotalBytes() {
            return totalBytes.get();
        }
    }
}