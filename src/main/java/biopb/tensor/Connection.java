package biopb.tensor;

import java.time.Duration;
import java.util.Locale;
import java.util.Map;
import java.util.function.LongConsumer;
import java.util.logging.Logger;

import org.apache.arrow.flight.Location;
import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * One handle on the data plane: finds it, dials it, and says why when it cannot.
 *
 * <p>The Java twin of Python's {@code biopb.tensor.Connection}. It holds the live
 * {@link TensorFlightClient}, the URL it dials and, when it has none, why. It
 * caches nothing else: the catalog, health and sources are the plane's to
 * answer, asked on the client.
 *
 * <p>Where the plane is comes from, in order: {@code $BIOPB_TENSOR_URL}, then the
 * control ({@code docs/discovery-contract.md}), which also brings the plane up.
 * An address from anywhere but the control is dialed with an explicit token or
 * {@code $BIOPB_TENSOR_TOKEN}, never with the control's credential file.
 *
 * <p>{@link #connect} blocks on the network. Not thread-safe to call from two
 * threads at once; readers on other threads see the previous {@link #client}
 * until the new one is ready.
 */
public final class Connection implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(Connection.class.getName());

    /** How long {@link #connect()} waits for a plane it was told is coming up. */
    public static final Duration START_TIMEOUT = Duration.ofSeconds(60);

    static final String NO_CONTROL_MESSAGE = "No biopb control plane is running, so there is no data plane to "
            + "connect to. Run `biopb control start`.";

    private static final long DEFAULT_CACHE_BYTES = 100_000_000L;

    private static final String[] AUTH_MARKERS = {
        "unauthenticat", "unauthoriz", "permission denied", "invalid token", "missing token" };
    private static final String[] UNREACHABLE_MARKERS = {
        "unavailable", "refused", "failed to connect", "deadline", "timed out", "timeout" };

    /**
     * Where first-use trust remembers pins. Process-wide and in memory: pins last
     * until the JVM exits (see {@link TlsPinStore} for why there is no file).
     */
    private static final TlsPinStore PROCESS_PINS = TlsPinStore.inMemory();

    private final DataPlaneDiscovery discovery;
    private final LongConsumer sleeper;

    private volatile TensorFlightClient client;
    private volatile String url;
    private volatile String lastMessage = "";

    public Connection() {
        this(new DataPlaneDiscovery(DiscoveryEnvironment.system()), Connection::sleepMillis);
    }

    Connection(DataPlaneDiscovery discovery, LongConsumer sleeper) {
        this.discovery = discovery;
        this.sleeper = sleeper;
    }

    /** The live client, or null until a {@link #connect} succeeds (and again after one fails). */
    public TensorFlightClient client() {
        return client;
    }

    /** The address last dialed, or null. */
    public String url() {
        return url;
    }

    /** Why the last {@link #connect} failed, phrased for the person who has to fix it; empty on success. */
    public String lastMessage() {
        return lastMessage;
    }

    /** {@link #connect(String, String, Duration)} with the control naming the plane. */
    public boolean connect() {
        return connect(null, null, START_TIMEOUT);
    }

    /**
     * Dial the data plane; true when {@link #client} is ready to use. Never throws.
     *
     * <p>With no {@code url}, {@code $BIOPB_TENSOR_URL} or else the control names
     * the plane. A {@code url} is dialed as given, with {@code token} and nothing
     * else.
     *
     * <p>A plane the control named is waited on through {@code timeout}, since the
     * control may have just started it; one named anywhere else is not, except
     * through a scan it reports as {@code STARTING}.
     *
     * @param url the plane to dial, or null to discover it
     * @param token the bearer token for {@code url}; ignored when discovering
     */
    public boolean connect(String url, String token, Duration timeout) {
        if (url != null) {
            return dial(url, token, Origin.MANUAL, timeout);
        }
        String fromEnv = discovery.environment().trimmed(DataPlaneDiscovery.ENV_TENSOR_URL);
        if (!fromEnv.isEmpty()) {
            return dial(fromEnv, discovery.resolveToken(null, false), Origin.ENV, timeout);
        }
        DataPlaneDiscovery.DataPlane plane = discovery.ensureDataPlane(timeout);
        if (plane == null) {
            this.client = closeAndNull(this.client);
            this.url = null;
            this.lastMessage = NO_CONTROL_MESSAGE;
            return false;
        }
        return dial(plane.url, plane.token, Origin.CONTROL, timeout);
    }

    /** Close the client, if any. */
    @Override
    public void close() {
        this.client = closeAndNull(this.client);
    }

    // ---- dialing --------------------------------------------------------------

    private enum Origin { CONTROL, ENV, MANUAL }

    /** The plane answered but is not {@code SERVING} yet. */
    private static final class Starting extends Exception {
        private static final long serialVersionUID = 1L;

        Starting(String message) {
            super(message);
        }
    }

    private boolean dial(String target, String token, Origin origin, Duration timeout) {
        this.url = target;
        long deadline = System.nanoTime() + timeout.toNanos();
        long interval = 500;
        while (true) {
            try {
                TensorFlightClient opened = open(target, token, origin);
                closeAndNull(this.client);
                this.client = opened;
                this.lastMessage = "";
                return true;
            } catch (Starting starting) {
                this.lastMessage = starting.getMessage();
            } catch (Exception error) { // recorded, not raised
                this.lastMessage = connectErrorMessage(error, target, token, origin);
                // Only the control's plane may still be binding its port.
                if (origin != Origin.CONTROL) {
                    break;
                }
            }
            if (System.nanoTime() >= deadline) {
                break;
            }
            sleeper.accept(interval);
            interval = Math.min(interval * 2, 5_000);
        }
        this.client = closeAndNull(this.client);
        LOGGER.info("data plane at " + target + " not connected: " + this.lastMessage);
        return false;
    }

    /** A client for {@code target} that the plane has answered, and accepted. */
    private TensorFlightClient open(String target, String token, Origin origin) throws Exception {
        DataPlaneDiscovery.TlsAnchor anchor = discovery.dataPlaneTrust(target, origin == Origin.CONTROL);
        Location location = LocationUris.parse(target);
        TlsTrust trust = TlsTrusts.isTlsLocation(location)
                ? TlsTrusts.resolve(location, anchor.caPem, anchor.fingerprint, PROCESS_PINS)
                : TlsTrust.NONE;
        TensorFlightClient opened = new TensorFlightClient(location, DEFAULT_CACHE_BYTES, token, trust);
        try {
            Map<String, Object> health = opened.healthCheck();
            Object status = health.getOrDefault("status", "SERVING");
            if (!"SERVING".equals(status)) {
                throw new Starting(startingMessage(health));
            }
            // health answers anyone; this is the call that checks the token.
            try (VectorSchemaRoot ignored = opened.query("SELECT 1 FROM sources LIMIT 0")) {
                return opened;
            }
        } catch (Exception | Error error) {
            opened.close();
            throw error;
        }
    }

    private static String startingMessage(Map<String, Object> health) {
        StringBuilder message = new StringBuilder("Tensor server is starting -- scanning its data folder; this "
                + "can take a while for large catalogs.");
        StringBuilder bits = new StringBuilder();
        Object sources = health.get("source_count");
        if (sources instanceof Number) {
            bits.append(((Number) sources).longValue()).append(" sources registered so far");
        }
        Object uptime = health.get("uptime_seconds");
        if (uptime instanceof Number) {
            bits.append(bits.length() > 0 ? ", " : "").append("up ").append(((Number) uptime).longValue()).append('s');
        }
        return bits.length() > 0 ? message + " (" + bits + ")" : message.toString();
    }

    /**
     * Why a dial to {@code target} failed, phrased for the person who has to fix
     * it. The fix for a missing token differs by where the address came from: only
     * the control's address was offered its credential file.
     */
    private static String connectErrorMessage(Exception error, String target, String token, Origin origin) {
        // By type first: an unreadable certificate says "Permission denied", which
        // the auth markers below would otherwise claim.
        if (error instanceof LocalTrustException
                || error instanceof TlsPinMismatchException
                || error instanceof TlsCertExpiredException) {
            return error.getMessage();
        }
        String text = (error.getClass().getSimpleName() + ": "
                + (error instanceof org.apache.arrow.flight.FlightRuntimeException
                        ? ((org.apache.arrow.flight.FlightRuntimeException) error).status().code() + ": "
                        : "")
                + error.getMessage()).trim();
        String low = text.toLowerCase(Locale.ROOT);
        if (containsAny(low, AUTH_MARKERS)) {
            if (token != null) {
                return "Authentication failed: the tensor server at " + target + " rejected the token.";
            }
            if (origin == Origin.ENV) {
                return "Authentication required: the tensor server at " + target + " needs a token. Set $"
                        + DataPlaneDiscovery.ENV_TENSOR_TOKEN + ". This endpoint came from $"
                        + DataPlaneDiscovery.ENV_TENSOR_URL + ", so it bypassed the control plane and the "
                        + "control's local credential file was not used for it.";
            }
            if (origin == Origin.MANUAL) {
                return "Authentication required: the tensor server at " + target + " needs a token.";
            }
            return "Authentication required: the tensor server at " + target + " needs a token, but the control "
                    + "plane's credential file held none. Restart the control (`biopb control start`), or set $"
                    + DataPlaneDiscovery.ENV_TENSOR_TOKEN + ".";
        }
        if (containsAny(low, UNREACHABLE_MARKERS)) {
            return "Cannot reach the tensor server at " + target + " -- is it running?";
        }
        return "Could not connect to the tensor server at " + target + ": " + text;
    }

    private static boolean containsAny(String text, String[] markers) {
        for (String marker : markers) {
            if (text.contains(marker)) {
                return true;
            }
        }
        return false;
    }

    private static TensorFlightClient closeAndNull(TensorFlightClient client) {
        if (client != null) {
            try {
                client.close();
            } catch (RuntimeException ignored) {
                // closing a client that is being replaced must not mask the new outcome
            }
        }
        return null;
    }

    private static void sleepMillis(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        }
    }
}
