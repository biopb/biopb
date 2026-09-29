package biopb.tensor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.arrow.flight.Location;

/**
 * The process-wide cache of {@link FlightSession}s, keyed by where and as whom.
 *
 * <p>A {@link SerializedTensor} is the cross-process handle: a worker is handed
 * one (or many) and reconstructs a lazy image from each. Giving every image its
 * own session makes the channel count scale with the number of tensors
 * deserialized rather than with the number of servers talked to -- and since
 * {@code tensorFromPb} is typed {@link net.imglib2.RandomAccessibleInterval},
 * a caller cannot close one even if it wanted to. Sharing per
 * {@code (location, token)} is what this SDK's own connection pool did before
 * it was retired.
 *
 * <p>Python keys the same idea differently in two ways, and only one of them
 * is a difference this SDK can have. Its client pool
 * ({@code biopb.tensor._pool._get_thread_client}) is <b>per thread</b>, because
 * a pyarrow {@code FlightClient} is not the shared, multiplexing channel a
 * {@code FlightClient} is here; its chunk-cache pool, which is shared across
 * threads, keys on {@code (location, token)} exactly as this does. The token
 * belongs in the key in every one of them: two callers on one server with
 * different capabilities must not share a channel's authorization.
 *
 * <p>The one to carry over is the third component Python's connection key has
 * and this one does not: {@code TlsTrust.key_id}. One process can front two
 * upstreams naming the same {@code host:port} under different configured trust
 * anchors, each with its own hostname override, and keying without it hands one
 * upstream a connection built with the other's trust (biopb/biopb#604 item 4).
 * It is not a gap yet only because {@link FlightSession} does not construct TLS
 * at all -- {@link LocationUris} parses a {@code grpc+tls://} location but
 * nothing configures trust for it. <b>Whoever adds TLS must add it to this key
 * in the same change</b>, or the bug arrives silently with the feature.
 * biopb/biopb#1072 tracks that work and names this as part of it.
 *
 * <p>Sessions live until the JVM exits. That is the point -- there is no
 * reference count to hang a close on, because an image hands its session to an
 * imglib2 cell cache that outlives every call this class can see. The cost is
 * one idle gRPC channel per distinct server, which is what a connection pool
 * is.
 *
 * <p>A caller that owns its connection's lifetime should use
 * {@link TensorFlightClient}, which holds a session of its own and closes it.
 */
final class FlightSessions {

    private static final Map<String, FlightSession> CACHE = new ConcurrentHashMap<>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(FlightSessions::closeAll, "biopb-flight-sessions"));
    }

    private FlightSessions() {}

    /**
     * The session for {@code location} as {@code token}, opening one on first
     * use. The cache owns it; do <b>not</b> close the result.
     */
    static FlightSession shared(Location location, String token) {
        String key = key(location, token);
        return CACHE.computeIfAbsent(key, ignored -> new FlightSession(location, token));
    }

    /**
     * The token is part of the key, not just the location: two callers on one
     * server with different capabilities must not share a channel's
     * authorization. A resolved TLS trust id belongs here too, once there is
     * one -- see the class javadoc.
     */
    private static String key(Location location, String token) {
        return location.getUri().toString() + "\u0000" + (token == null ? "" : token);
    }

    /** Close every cached session. Package-private for tests and the shutdown hook. */
    static void closeAll() {
        for (String key : CACHE.keySet()) {
            FlightSession session = CACHE.remove(key);
            if (session != null) {
                try {
                    session.close();
                } catch (RuntimeException ignored) {
                    // Shutdown is best-effort; one bad channel must not strand the rest.
                }
            }
        }
    }

    /** How many sessions are open. Package-private for tests. */
    static int size() {
        return CACHE.size();
    }
}
