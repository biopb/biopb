package biopb.tensor;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.grpc.CredentialCallOption;
import org.apache.arrow.memory.BufferAllocator;

/**
 * Static singleton connection pool for TensorFlightClient connections.
 *
 * This pool is used by SerializableTensorImg to reconstruct connections
 * after deserialization. Connections are keyed by (location, token) and
 * reused across multiple deserializations.
 *
 * Similar to Python's module-level connection pools for pickled tensors.
 */
public final class TensorConnectionPool {

    private static final TensorConnectionPool INSTANCE = new TensorConnectionPool();

    private final ConcurrentHashMap<ConnectionKey, PooledConnection> connections = new ConcurrentHashMap<>();

    private TensorConnectionPool() {
        // Singleton - private constructor
    }

    /**
     * Get or create a pooled connection for the given location and token.
     *
     * @param location Flight server location
     * @param token Bearer token for authentication (null disables auth)
     * @param allocator Buffer allocator for the connection
     * @return PooledConnection containing FlightClient and auth option
     */
    public static PooledConnection getConnection(Location location, String token, BufferAllocator allocator) {
        ConnectionKey key = new ConnectionKey(location, token);
        return INSTANCE.connections.computeIfAbsent(key, k -> createConnection(k, allocator));
    }

    private static PooledConnection createConnection(ConnectionKey key, BufferAllocator allocator) {
        FlightClient client = FlightClient.builder(allocator, key.location).build();
        CredentialCallOption authOption = key.token != null && !key.token.isEmpty()
            ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + key.token))
            : null;
        return new PooledConnection(client, authOption);
    }

    /**
     * Get the singleton instance's connection count (for testing/monitoring).
     */
    public static int getConnectionCount() {
        return INSTANCE.connections.size();
    }

    /**
     * Clear all pooled connections (for testing).
     */
    public static void clear() {
        INSTANCE.connections.clear();
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
}