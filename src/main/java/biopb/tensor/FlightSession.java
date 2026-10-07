package biopb.tensor;

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.FlightStatusCode;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.flight.grpc.CredentialCallOption;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/**
 * Owns one Flight connection's allocator, authentication and error boundary.
 *
 * <p>Every foreground Flight call enters here, so typed server errors are
 * translated consistently both when an RPC starts and when a streaming action
 * reports an error while its iterator is consumed. It also builds the TLS
 * client from a resolved {@link TlsTrust}.
 */
public final class FlightSession implements AutoCloseable {
    private final BufferAllocator allocator;
    private final FlightClient client;
    private final CredentialCallOption authOption;
    private final Location location;
    private final String token;
    private final TlsTrust trust;

    private static final Gson GSON = new Gson();

    /**
     * Set once the server's Flight protocol shape has been checked. Volatile
     * and checked outside the lock: the probe is idempotent, so a racing pair
     * costing one extra {@code health} call is cheaper than serializing every
     * RPC behind a monitor.
     */
    private volatile boolean protocolChecked;

    /** The server's own {@code health.external_location}, once {@link #ensureProtocol} has read it. */
    private volatile String advertisedLocation;

    public FlightSession(Location location, String token) {
        this(location, token, TlsTrust.NONE);
    }

    /**
     * Open a session whose TLS trust is {@code trust}: its anchor is the only
     * certificate the connection accepts, and its hostname override (when set)
     * is the name matched against the certificate instead of the dialed one.
     * {@link TlsTrust#NONE} leaves the JDK's default trust store in charge.
     * Meaningful only for a {@code grpc+tls://} location.
     */
    public FlightSession(Location location, String token, TlsTrust trust) {
        TlsTrust effective = trust == null ? TlsTrust.NONE : trust;
        if (effective.reresolve()) {
            throw new IllegalArgumentException("an anchored TlsTrust must be made concrete for this "
                    + "location first: TlsTrusts.concrete(location, trust)");
        }
        this.location = location;
        this.token = token;
        this.trust = effective;
        this.allocator = new RootAllocator(Long.MAX_VALUE);
        FlightClient.Builder builder = FlightClient.builder(allocator, location);
        if (effective.rootCerts() != null) {
            builder.trustedCertificates(new java.io.ByteArrayInputStream(effective.rootCerts()));
        }
        if (effective.overrideHostname() != null) {
            builder.overrideHostname(effective.overrideHostname());
        }
        this.client = builder.build();
        this.authOption = token == null || token.isEmpty()
                ? null
                : new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + token));
    }

    public Location location() { return location; }
    public String token() { return token; }
    public TlsTrust trust() { return trust; }
    public BufferAllocator allocator() { return allocator; }
    public FlightClient client() { return client; }
    public CredentialCallOption authOption() { return authOption; }

    /**
     * Refuse a server whose Flight protocol shape is not this SDK's, once per
     * connection.
     *
     * <p>Runs on first use rather than in the constructor, so building a client
     * stays free of I/O -- the arrangement Python's {@code _ClientState.client}
     * property has. Reads the {@code protocol} key from the {@code health}
     * action; a server without it predates the key and speaks v1, which this
     * SDK no longer does. Failing here names the mismatch, where letting a v2
     * request reach a v1 server produces a parse error from the wrong proto,
     * or a chunk this client cannot decode.
     *
     * <p>Deliberately quiet in one case: a capability token cannot reach the
     * catalog tier that {@code health} sits on, and the private call about to
     * be made authorizes itself. Every other way of not stating a protocol --
     * no reply, an empty body, a body that is not JSON, a JSON body without the
     * key -- is refused alike. They are one fact, "this server did not say",
     * and the gate is worth nothing if the quietest server walks through it.
     */
    private void ensureProtocol() {
        if (protocolChecked) {
            return;
        }
        Optional<Map<String, Object>> health;
        try {
            health = health();
        } catch (FlightRuntimeException error) {
            if (error.status().code() == FlightStatusCode.UNAUTHENTICATED
                    || error.status().code() == FlightStatusCode.UNAUTHORIZED) {
                protocolChecked = true;
                return;
            }
            throw TensorErrorMapper.map(error);
        }
        Object external = health.orElse(Collections.emptyMap()).get("external_location");
        if (external instanceof String && !((String) external).isEmpty()) {
            advertisedLocation = (String) external;
        }
        // Anything but a stated v2 is a v1 server. The key postdates that
        // version, so its absence names the version rather than leaving it
        // unknown -- and a server that is not biopb at all is refused here
        // rather than at "Data column value is not binary", from inside a cell
        // load, which is the error this gate exists to replace.
        Object protocol = health.orElse(Collections.emptyMap()).get("protocol");
        int serverVersion = protocol instanceof Number ? ((Number) protocol).intValue() : 1;
        if (serverVersion != WireVersions.FLIGHT_PROTOCOL_VERSION) {
            throw new UnsupportedOperationException(WireVersions.mismatch(
                    "Flight protocol", serverVersion, WireVersions.FLIGHT_PROTOCOL_VERSION,
                    "The server at " + location + " routes requests in another shape."));
        }
        protocolChecked = true;
    }

    /**
     * The address the server says it is reachable at ({@code
     * health.external_location}, biopb/biopb#1158), as Arrow names it
     * ({@code grpc+tls://} for a TLS location), or null if it published none or
     * will not say (a capability token cannot reach {@code health}). Runs the one
     * protocol check if no call has yet.
     */
    String advertisedLocation() {
        ensureProtocol();
        String advertised = advertisedLocation;
        if (advertised == null) {
            return null;
        }
        return advertised.regionMatches(true, 0, "grpcs://", 0, 8)
                ? "grpc+tls://" + advertised.substring(8)
                : advertised;
    }

    /**
     * The server's {@code health} reply, parsed. Absent means it answered
     * nothing at all; an empty map means it answered something that is not
     * JSON. {@link #ensureProtocol} refuses both, and {@link
     * TensorFlightClient#healthCheck} reports the first as {@code UNKNOWN}.
     *
     * <p>Goes straight to the client rather than through {@link #doAction}, so
     * that {@link #ensureProtocol} -- which is the reason this exists -- cannot
     * recurse into itself, and so a caller asking for health does not pay for a
     * second {@code health} round trip to get there.
     */
    Optional<Map<String, Object>> health() {
        Iterator<Result> results = client.doAction(new Action("health", new byte[0]), authOption);
        byte[] body = results.hasNext() ? results.next().getBody() : null;
        if (body == null || body.length == 0) {
            return Optional.empty();
        }
        try {
            Map<String, Object> parsed = GSON.fromJson(new String(body, StandardCharsets.UTF_8),
                    new TypeToken<Map<String, Object>>() {}.getType());
            return Optional.of(parsed == null ? Collections.emptyMap() : parsed);
        } catch (RuntimeException ignored) {
            return Optional.of(Collections.emptyMap());
        }
    }

    public FlightInfo getInfo(FlightDescriptor descriptor) {
        ensureProtocol();
        try {
            return client.getInfo(descriptor, authOption);
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    public FlightStream getStream(Ticket ticket) {
        ensureProtocol();
        try {
            return client.getStream(ticket, authOption);
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    /**
     * Open a DoPut and send {@code root}'s schema; the caller writes the
     * batches and drains {@code listener}.
     */
    public FlightClient.ClientStreamListener startPut(
            FlightDescriptor descriptor, VectorSchemaRoot root, FlightClient.PutListener listener) {
        ensureProtocol();
        try {
            return client.startPut(descriptor, root, listener, authOption);
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    /**
     * The typed exception behind a failure surfaced through a future.
     *
     * <p>A DoPut reports its error on the listener rather than from the call
     * that started it, so the boundary the other methods here draw has to be
     * reachable from the unwrapped cause as well.
     */
    public static RuntimeException mapped(Throwable cause) {
        if (cause instanceof FlightRuntimeException) {
            return TensorErrorMapper.map((FlightRuntimeException) cause);
        }
        if (cause instanceof RuntimeException) {
            return (RuntimeException) cause;
        }
        return new IllegalStateException(cause == null ? "Flight call failed" : cause.getMessage(), cause);
    }

    public Iterator<Result> doAction(Action action) {
        // `health` itself goes through the raw client in ensureProtocol, so
        // this cannot recurse.
        ensureProtocol();
        try {
            return new ErrorMappingIterator(client.doAction(action, authOption));
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    @Override
    public void close() {
        try {
            client.close();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        } finally {
            allocator.close();
        }
    }

    static final class ErrorMappingIterator implements Iterator<Result> {
        private final Iterator<Result> delegate;

        ErrorMappingIterator(Iterator<Result> delegate) { this.delegate = delegate; }

        @Override
        public boolean hasNext() {
            try {
                return delegate.hasNext();
            } catch (FlightRuntimeException error) {
                throw TensorErrorMapper.map(error);
            }
        }

        @Override
        public Result next() {
            try {
                return delegate.next();
            } catch (FlightRuntimeException error) {
                throw TensorErrorMapper.map(error);
            }
        }
    }
}
