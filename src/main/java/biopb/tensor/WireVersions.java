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
 * <p>Neither is the on-disk cache-file format ({@code format_version} in the
 * {@code chunk_locate} reply), which versions the localhost mmap handoff this
 * SDK does not implement.
 */
final class WireVersions {

    private WireVersions() {}

    /** The Flight protocol shape this client speaks. */
    static final int FLIGHT_PROTOCOL_VERSION = 2;

    /** The chunk wire encoding this client can decode. */
    static final int TENSOR_WIRE_PROTOCOL_VERSION = 2;

    /** Schema-metadata key carrying the server's chunk encoding version. */
    static final String WIRE_PROTOCOL_METADATA_KEY = "chunk_wire_protocol";

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

    /** The refusal a version mismatch deserves, naming which side to upgrade. */
    static String mismatch(String what, int serverVersion, int clientVersion, String detail) {
        String stale = serverVersion < clientVersion ? "server" : "client";
        return "Incompatible biopb " + what + ": the server speaks v" + serverVersion
                + ", this client speaks v" + clientVersion + ". " + detail
                + " Upgrade the " + stale + " so both sides match.";
    }
}
