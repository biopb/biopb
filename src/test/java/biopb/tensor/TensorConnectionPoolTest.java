package biopb.tensor;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.arrow.flight.Location;
import org.apache.arrow.memory.RootAllocator;
import org.junit.Test;

/**
 * Unit tests for TensorConnectionPool.
 *
 * Tests thread-local connection storage, connection key equality,
 * shared cache operations, and thread death eviction.
 */
public class TensorConnectionPoolTest {

    @Test
    public void testConnectionKeyEquality() {
        Location loc1 = Location.forGrpcInsecure("localhost", 8815);
        Location loc2 = Location.forGrpcInsecure("localhost", 8815);
        Location loc3 = Location.forGrpcInsecure("localhost", 9000);

        TensorConnectionPool.ConnectionKey key1a = new TensorConnectionPool.ConnectionKey(loc1, "token1");
        TensorConnectionPool.ConnectionKey key1b = new TensorConnectionPool.ConnectionKey(loc2, "token1");
        TensorConnectionPool.ConnectionKey key2 = new TensorConnectionPool.ConnectionKey(loc1, "token2");
        TensorConnectionPool.ConnectionKey key3 = new TensorConnectionPool.ConnectionKey(loc3, "token1");

        // Same location and token should be equal
        assertEquals(key1a, key1b);

        // Different token should not be equal
        assertNotEquals(key1a, key2);

        // Different location should not be equal
        assertNotEquals(key1a, key3);

        // Null token should not equal non-null token
        TensorConnectionPool.ConnectionKey keyNull = new TensorConnectionPool.ConnectionKey(loc1, null);
        assertNotEquals(key1a, keyNull);

        // Two null tokens at same location should be equal
        TensorConnectionPool.ConnectionKey keyNull1 = new TensorConnectionPool.ConnectionKey(loc1, null);
        TensorConnectionPool.ConnectionKey keyNull2 = new TensorConnectionPool.ConnectionKey(loc2, null);
        assertEquals(keyNull1, keyNull2);
    }

    @Test
    public void testConnectionKeyHashCode() {
        Location loc1 = Location.forGrpcInsecure("localhost", 8815);
        Location loc2 = Location.forGrpcInsecure("localhost", 8815);

        TensorConnectionPool.ConnectionKey key1 = new TensorConnectionPool.ConnectionKey(loc1, "token");
        TensorConnectionPool.ConnectionKey key2 = new TensorConnectionPool.ConnectionKey(loc2, "token");

        // Equal keys should have same hashCode
        assertEquals(key1.hashCode(), key2.hashCode());

        // HashCode should be consistent
        int hash1 = key1.hashCode();
        int hash2 = key1.hashCode();
        assertEquals(hash1, hash2);
    }

    @Test
    public void testConnectionKeyNullToken() {
        Location loc = Location.forGrpcInsecure("localhost", 8815);

        TensorConnectionPool.ConnectionKey key1 = new TensorConnectionPool.ConnectionKey(loc, null);
        TensorConnectionPool.ConnectionKey key2 = new TensorConnectionPool.ConnectionKey(loc, "");

        // Null and empty token should not be equal (depends on implementation)
        // The implementation uses Objects.equals on token strings
        assertNotEquals(key1, key2);
    }

    @Test
    public void testSharedCacheCreation() {
        Location loc = Location.forGrpcInsecure("localhost", 8815);
        TensorConnectionPool.ConnectionKey key = new TensorConnectionPool.ConnectionKey(loc, null);

        TensorConnectionPool.SharedCache cache = TensorConnectionPool.getSharedCache(key, 1_000_000L);

        assertNotNull(cache);
    }

    @Test
    public void testSharedCachePutAndGet() {
        Location loc = Location.forGrpcInsecure("localhost", 8815);
        TensorConnectionPool.ConnectionKey key = new TensorConnectionPool.ConnectionKey(loc, null);

        TensorConnectionPool.SharedCache cache = TensorConnectionPool.getSharedCache(key, 10_000L);
        cache.clear(); // Start fresh

        // Put a value
        String testKey = "chunk-abc123";
        Object testValue = new byte[100];
        cache.put(testKey, testValue, 100);

        // Get should return the value
        Object retrieved = cache.get(testKey);
        assertNotNull(retrieved);

        // Get for missing key should return null
        Object missing = cache.get("nonexistent");
        assertNull(missing);
    }

    @Test
    public void testSharedCacheEviction() {
        Location loc = Location.forGrpcInsecure("localhost", 8815);
        TensorConnectionPool.ConnectionKey key = new TensorConnectionPool.ConnectionKey(loc, null);

        // Create small cache (1KB max)
        TensorConnectionPool.SharedCache cache = new TensorConnectionPool.SharedCache(1000L);

        // Put values totaling more than max
        cache.put("key1", new byte[500], 500);
        assertEquals(500L, cache.getTotalBytes());

        // Put another value - should trigger eviction
        cache.put("key2", new byte[600], 600);

        // After eviction, cache should be cleared
        // The implementation clears when totalBytes + newBytes > maxBytes
        assertEquals(600L, cache.getTotalBytes()); // Only new entry remains
    }

    @Test
    public void testSharedCacheClear() {
        TensorConnectionPool.SharedCache cache = new TensorConnectionPool.SharedCache(10_000L);

        // Add some entries
        cache.put("key1", "value1", 100);
        cache.put("key2", "value2", 200);

        assertTrue(cache.getTotalBytes() > 0);

        // Clear should reset
        cache.clear();
        assertEquals(0L, cache.getTotalBytes());
        assertNull(cache.get("key1"));
        assertNull(cache.get("key2"));
    }

    @Test
    public void testGetConnectionCreatesNewConnection() {
        // Clear pool before test
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.PooledConnection conn = TensorConnectionPool.getConnection(loc, null, allocator);

            assertNotNull(conn);
            assertNotNull(conn.getClient());
            assertNull(conn.getAuthOption()); // No token
            assertTrue(conn.getCreationTime() > 0);
        }

        // Cleanup
        TensorConnectionPool.clear();
    }

    @Test
    public void testGetConnectionWithToken() {
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.PooledConnection conn = TensorConnectionPool.getConnection(loc, "test-token", allocator);

            assertNotNull(conn);
            assertNotNull(conn.getClient());
            assertNotNull(conn.getAuthOption()); // Should have auth option
        }

        TensorConnectionPool.clear();
    }

    @Test
    public void testGetConnectionReusesThreadLocal() {
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.PooledConnection conn1 = TensorConnectionPool.getConnection(loc, null, allocator);
            TensorConnectionPool.PooledConnection conn2 = TensorConnectionPool.getConnection(loc, null, allocator);

            // Same thread should get same connection
            assertEquals(conn1, conn2);
        }

        TensorConnectionPool.clear();
    }

    @Test
    public void testClearRemovesAllConnections() {
        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            // Create a connection
            TensorConnectionPool.getConnection(loc, null, allocator);

            // Should have at least 1 connection
            assertTrue(TensorConnectionPool.getConnectionCount() >= 1);

            // Clear should remove all
            TensorConnectionPool.clear();

            assertEquals(0, TensorConnectionPool.getConnectionCount());
            assertEquals(0, TensorConnectionPool.getThreadCount());
        }
    }

    @Test
    public void testGetConnectionCount() {
        TensorConnectionPool.clear();

        Location loc1 = Location.forGrpcInsecure("localhost", 8815);
        Location loc2 = Location.forGrpcInsecure("localhost", 9000);

        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            // Create connections for different locations
            TensorConnectionPool.getConnection(loc1, null, allocator);
            TensorConnectionPool.getConnection(loc2, null, allocator);

            // Should have 2 connections (in same thread)
            assertTrue(TensorConnectionPool.getConnectionCount() >= 2);
        }

        TensorConnectionPool.clear();
    }

    @Test
    public void testGetThreadCount() {
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.getConnection(loc, null, allocator);

            // Should have at least 1 thread
            assertTrue(TensorConnectionPool.getThreadCount() >= 1);
        }

        TensorConnectionPool.clear();
    }

    @Test
    public void testCloseConnectionRemovesFromPool() {
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.getConnection(loc, null, allocator);

            int countBefore = TensorConnectionPool.getConnectionCount();
            assertTrue(countBefore >= 1);

            // Close the connection
            TensorConnectionPool.closeConnection(loc, null);

            // Should have fewer connections
            assertTrue(TensorConnectionPool.getConnectionCount() < countBefore);
        }
    }

    @Test
    public void testPooledConnectionGetClient() {
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.PooledConnection conn = TensorConnectionPool.getConnection(loc, null, allocator);

            assertNotNull(conn.getClient());
            // Client should be a FlightClient
            assertTrue(conn.getClient() instanceof org.apache.arrow.flight.FlightClient);
        }

        TensorConnectionPool.clear();
    }

    @Test
    public void testPooledConnectionCreationTime() {
        TensorConnectionPool.clear();

        Location loc = Location.forGrpcInsecure("localhost", 8815);
        long beforeTime = System.currentTimeMillis();

        try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
            TensorConnectionPool.PooledConnection conn = TensorConnectionPool.getConnection(loc, null, allocator);

            long creationTime = conn.getCreationTime();
            assertTrue(creationTime >= beforeTime);
            assertTrue(creationTime <= System.currentTimeMillis());
        }

        TensorConnectionPool.clear();
    }
}