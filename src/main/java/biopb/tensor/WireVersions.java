package biopb.tensor;

/**
 * The two protocol versions this SDK speaks, and how to read the server's.
 *
 * <p>These mirror {@code biopb.tensor._wire_version}, which the Python SDK and
 * the tensor server both import so they cannot drift. Java cannot import it, so
 * this is a second copy that has to be kept in step -- the same arrangement as
 * {@link RoiRowCodec} and the ROI row schema.
 *
 * <p>The two are independent, and a mismatch in either is fatal in its own way:
 *
 * <ul>
 *   <li>{@link #FLIGHT_PROTOCOL_VERSION} -- the <b>shape</b> of the protocol:
 *       which descriptors, tickets and put commands the server understands.
 *       v1 routed by a sentinel {@code source_id} in a {@code FlightCmd} and
 *       sniffed ticket prefixes; v2 is the {@code FlightRequest} /
 *       {@code TensorTicket} / {@code PutCommand} oneofs this client sends.
 *       Reported by the {@code health} action and checked once per connection,
 *       so a v1 server is named as such instead of parsing a v2 request as
 *       something else.
 *   <li>{@link #TENSOR_WIRE_PROTOCOL_VERSION} -- the <b>chunk encoding</b>:
 *       v1 was a typed {@code data: list<T>} per chunk, v2 is one binary blob
 *       plus a numpy dtype string (biopb/biopb#293), which is what
 *       {@link ChunkDecoder} reads. Stamped on the schema of every read plan
 *       and checked where a plan becomes an image, because that is the last
 *       point before the bytes are reinterpreted.
 * </ul>
 *
 * <p>Those two are the whole set. A read plan's schema once also carried a
 * {@code tensor_schema_version} release tag, and {@code chunk_locate} a
 * {@code format_version} for the localhost mmap handoff this SDK does not
 * implement; both are gone (biopb/biopb#1070). A version-shaped key beside a
 * real gate reads like a gate -- this client implemented the first one as a
 * compatibility check, which it never was.
 */
final class WireVersions {

    private WireVersions() {}

    /**
     * The newest Flight protocol shape this client speaks. v3 moved what a plan
     * says about its request: {@code FlightInfo.app_metadata} is the whole
     * {@code TensorReadOption}, and the descriptor no longer echoes its scale
     * and method (see {@link PlanRequest}).
     */
    static final int FLIGHT_PROTOCOL_VERSION = 3;

    /** The oldest Flight protocol shape this client still reads. */
    static final int MIN_FLIGHT_PROTOCOL_VERSION = 2;

    /** Schema-metadata key carrying the protocol a plan was written under. */
    static final String FLIGHT_PROTOCOL_METADATA_KEY = "flight_protocol";

    /** Does this client speak the server's Flight protocol shape? */
    static boolean supportsFlight(int serverVersion) {
        return serverVersion >= MIN_FLIGHT_PROTOCOL_VERSION && serverVersion <= FLIGHT_PROTOCOL_VERSION;
    }

    /** The chunk wire encoding this client can decode. */
    static final int TENSOR_WIRE_PROTOCOL_VERSION = 2;

    /** Schema-metadata key carrying the server's chunk encoding version. */
    static final String WIRE_PROTOCOL_METADATA_KEY = "chunk_wire_protocol";

    /** The value stamped under {@code key} on the plan's schema metadata, or null. */
    static String stamp(org.apache.arrow.flight.FlightInfo plan, String key) {
        java.util.Optional<org.apache.arrow.vector.types.pojo.Schema> schema = plan.getSchemaOptional();
        if (!schema.isPresent()) {
            return null;
        }
        java.util.Map<String, String> metadata = schema.get().getCustomMetadata();
        return metadata == null ? null : metadata.get(key);
    }

    /**
     * A version stamp, or 1 when it is absent or unreadable.
     *
     * <p>Absent means v1 in both cases: the key postdates that version, so a
     * server that does not send it is one that predates the key.
     */
    static int stampedVersion(String raw) {
        if (raw == null || raw.isEmpty()) {
            return 1;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException ignored) {
            return 1;
        }
    }

    /** The refusal a Flight protocol mismatch deserves, naming which side to upgrade. */
    static String flightMismatch(int serverVersion, String detail) {
        String stale = serverVersion < MIN_FLIGHT_PROTOCOL_VERSION ? "server" : "client";
        return "Incompatible biopb Flight protocol: the server speaks v" + serverVersion
                + ", this client speaks v" + MIN_FLIGHT_PROTOCOL_VERSION + "-v" + FLIGHT_PROTOCOL_VERSION
                + ". " + detail + " Upgrade the " + stale + " so both sides match.";
    }

    /** The refusal a version mismatch deserves, naming which side to upgrade. */
    static String mismatch(String what, int serverVersion, int clientVersion, String detail) {
        String stale = serverVersion < clientVersion ? "server" : "client";
        return "Incompatible biopb " + what + ": the server speaks v" + serverVersion
                + ", this client speaks v" + clientVersion + ". " + detail
                + " Upgrade the " + stale + " so both sides match.";
    }
}
