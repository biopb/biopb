package biopb.tensor;

import java.net.URI;

import org.apache.arrow.flight.Location;

/**
 * Parse a tensor-server location string into an Arrow Flight {@link Location}.
 *
 * <p>The location travels on the wire as a URI string (e.g.
 * {@code "grpc://host:port"}, {@code "grpc+tcp://host:port"}). Parsing was
 * forked four ways across the Java client (biopb/biopb#277 item D):
 * {@code SerializableTensorImg} and {@code Utils} used this tolerant
 * URI-based form, while {@code TensorFlightClient} carried a stricter parser
 * (plus an inline copy) that hand-sliced {@code grpc://}/{@code grpc+tcp://}
 * and rejected every other scheme. This is the single home, and it adopts the
 * tolerant contract -- a strict superset of the old grpc-only one -- so a
 * scheme-less {@code host:port} or a {@code grpc+tls://} URI parses instead of
 * throwing.
 */
public final class LocationUris {

    private LocationUris() {}

    /** {@code grpcs://}, the public spelling of Arrow's {@code grpc+tls://}, rewritten to it. */
    static String normalizeScheme(String uri) {
        return uri.regionMatches(true, 0, "grpcs://", 0, 8) ? "grpc+tls://" + uri.substring(8) : uri;
    }

    /** Whether {@code uri} is a TLS address in either spelling. */
    static boolean isTls(String uri) {
        return normalizeScheme(uri).regionMatches(true, 0, "grpc+tls://", 0, 11);
    }

    /**
     * Parse {@code uri} into a Flight {@link Location}.
     *
     * <p>A scheme-less authority (e.g. {@code "host:port"}) defaults to an
     * insecure gRPC location; any explicit scheme Arrow understands
     * ({@code grpc}, {@code grpc+tcp}, {@code grpc+tls}, {@code grpc+unix}) is
     * passed through as-is, and {@code grpcs://} is read as {@code grpc+tls://}.
     */
    public static Location parse(String uri) {
        uri = normalizeScheme(uri);
        try {
            return new Location(URI.create(uri));
        } catch (Exception e) {
            URI parsed = URI.create(uri);
            String scheme = parsed.getScheme();
            if (scheme == null) {
                return Location.forGrpcInsecure(parsed.getHost(), parsed.getPort());
            }
            return new Location(parsed);
        }
    }
}
