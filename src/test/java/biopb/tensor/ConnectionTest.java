package biopb.tensor;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.CallHeaders;
import org.apache.arrow.flight.CallStatus;
import org.apache.arrow.flight.FlightProducer;
import org.apache.arrow.flight.FlightServer;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.NoOpFlightProducer;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.flight.auth2.CallHeaderAuthenticator;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.sun.net.httpserver.HttpServer;

/**
 * {@link Connection} and the discovery contract behind it, against a fake
 * control (the JDK's own HTTP server) and a real Flight server, with the
 * environment and the state directory injected.
 */
public class ConnectionTest {

    private Path home;
    private Path state;
    private BufferAllocator allocator;
    private FlightServer plane;
    private HttpServer control;
    private final Map<String, String> vars = new HashMap<>();
    private final AtomicInteger startingPolls = new AtomicInteger();
    private volatile int startingFor = 0;
    private final List<String> ensureTokens = new CopyOnWriteArrayList<>();
    private volatile String planeUrl;

    @Before
    public void setUp() throws IOException {
        home = Files.createTempDirectory("biopb-home");
        vars.put("BIOPB_STATE_HOME", home.toString());
        state = home.resolve("biopb");
        Files.createDirectories(state);
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @After
    public void tearDown() throws Exception {
        if (plane != null) {
            plane.close();
        }
        if (control != null) {
            control.stop(0);
        }
        allocator.close();
        deleteTree(home);
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (java.util.stream.Stream<Path> walk = Files.walk(root)) {
            for (Path path : (Iterable<Path>) walk.sorted(java.util.Comparator.reverseOrder())::iterator) {
                Files.deleteIfExists(path);
            }
        }
    }

    // ---- fakes ------------------------------------------------------------------

    /** The plane: answers {@code health} (STARTING for a while, if asked) and an empty catalog query. */
    private final class Plane extends NoOpFlightProducer {
        @Override
        public void doAction(
                FlightProducer.CallContext context, Action action, FlightProducer.StreamListener<Result> listener) {
            String status = startingPolls.getAndIncrement() < startingFor ? "STARTING" : "SERVING";
            listener.onNext(new Result(("{\"status\":\"" + status + "\",\"protocol\":2,\"source_count\":3}")
                    .getBytes(StandardCharsets.UTF_8)));
            listener.onCompleted();
        }

        @Override
        public void getStream(
                FlightProducer.CallContext context, Ticket ticket, FlightProducer.ServerStreamListener listener) {
            try (IntVector column = new IntVector("one", allocator);
                    VectorSchemaRoot root = VectorSchemaRoot.of(column)) {
                root.setRowCount(0);
                listener.start(root);
                listener.putNext();
                listener.completed();
            }
        }
    }

    private void startPlane(String requireToken) throws Exception {
        FlightServer.Builder builder = FlightServer.builder(
                allocator, Location.forGrpcInsecure("localhost", 0), new Plane());
        if (requireToken != null) {
            builder.headerAuthenticator(new CallHeaderAuthenticator() {
                @Override
                public AuthResult authenticate(CallHeaders headers) {
                    if (!("Bearer " + requireToken).equals(headers.get("authorization"))) {
                        throw CallStatus.UNAUTHENTICATED.withDescription("invalid token").toRuntimeException();
                    }
                    return () -> "user";
                }
            });
        }
        plane = builder.build().start();
        planeUrl = "grpc://localhost:" + plane.getPort();
    }

    /** A control whose {@code /health} and {@code ensure} both name {@link #planeUrl}. */
    private void startControl() throws IOException {
        control = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        control.createContext("/health", exchange -> {
            reply(exchange, 200, "{\"data_plane\":{\"grpc_url\":\"" + planeUrl + "\"}}");
        });
        control.createContext("/api/data_plane/ensure", exchange -> {
            String token = exchange.getRequestHeaders().getFirst("X-Biopb-Token");
            ensureTokens.add(token == null ? "" : token);
            if (!"POST".equals(exchange.getRequestMethod())
                    || exchange.getRequestURI().getQuery() == null
                    || !exchange.getRequestURI().getQuery().startsWith("client_timeout=")) {
                reply(exchange, 400, "{}");
                return;
            }
            reply(exchange, 200, "{\"data_plane\":{\"grpc_url\":\"" + planeUrl + "\"}}");
        });
        control.start();
        vars.put("BIOPB_CONTROL_PORT", String.valueOf(control.getAddress().getPort()));
    }

    private static void reply(com.sun.net.httpserver.HttpExchange exchange, int status, String body)
            throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private Connection connection() {
        return new Connection(new DataPlaneDiscovery(DiscoveryEnvironment.of(vars, home)), millis -> { });
    }

    private DataPlaneDiscovery discovery() {
        return new DataPlaneDiscovery(DiscoveryEnvironment.of(vars, home));
    }

    private void writeCredential(String token) throws IOException {
        Files.write(state.resolve("tensor-server.token"), (token + "\n").getBytes(StandardCharsets.UTF_8));
    }

    // ---- the control names the plane ---------------------------------------------

    @Test
    public void theControlNamesAndStartsThePlaneAndItsCredentialGoesWithIt() throws Exception {
        startPlane("tok");
        startControl();
        writeCredential("tok");

        try (Connection connection = connection()) {
            Assert.assertTrue(connection.lastMessage(), connection.connect());
            Assert.assertNotNull(connection.client());
            Assert.assertEquals(planeUrl, connection.url());
            Assert.assertEquals("", connection.lastMessage());
        }
        // The credential also clears the control's CSRF gate on the POST.
        Assert.assertEquals(Collections.singletonList("tok"), ensureTokens);
    }

    @Test
    public void noControlIsSaidSoAndLeavesNoClient() throws Exception {
        vars.put("BIOPB_CONTROL_PORT", "1"); // nothing listens there
        try (Connection connection = connection()) {
            Assert.assertFalse(connection.connect(null, null, Duration.ofSeconds(2)));
            Assert.assertNull(connection.client());
            Assert.assertNull(connection.url());
            Assert.assertEquals(Connection.NO_CONTROL_MESSAGE, connection.lastMessage());
        }
    }

    @Test
    public void thePublishedRecordLocatesTheControlWhenNoVariableDoes() throws Exception {
        startPlane(null);
        startControl();
        int port = Integer.parseInt(vars.remove("BIOPB_CONTROL_PORT"));
        Files.write(state.resolve("control.json"),
                ("{\"host\":\"127.0.0.1\",\"port\":" + port + ",\"pid\":1}").getBytes(StandardCharsets.UTF_8));

        try (Connection connection = connection()) {
            Assert.assertTrue(connection.lastMessage(), connection.connect());
        }
    }

    @Test
    public void aPlaneStillScanningIsWaitedOnThroughTheTimeout() throws Exception {
        startPlane(null);
        startControl();
        startingFor = 2;

        try (Connection connection = connection()) {
            Assert.assertTrue(connection.lastMessage(), connection.connect());
        }
        Assert.assertTrue("polled " + startingPolls.get(), startingPolls.get() >= 3);
    }

    @Test
    public void aStartingPlaneThatNeverFinishesSaysWhyWhenTheTimeoutRunsOut() throws Exception {
        startPlane(null);
        startControl();
        startingFor = Integer.MAX_VALUE;

        try (Connection connection = connection()) {
            Assert.assertFalse(connection.connect(null, null, Duration.ofMillis(1500)));
            Assert.assertTrue(connection.lastMessage(), connection.lastMessage().contains("is starting"));
            Assert.assertTrue(connection.lastMessage(), connection.lastMessage().contains("3 sources"));
            Assert.assertNull(connection.client());
        }
    }

    // ---- an address that bypassed the control --------------------------------------

    @Test
    public void anAddressFromTheEnvironmentNeverGetsTheCredentialFile() throws Exception {
        startPlane("tok");
        writeCredential("tok");
        vars.put("BIOPB_TENSOR_URL", "grpc://localhost:" + plane.getPort());

        try (Connection connection = connection()) {
            Assert.assertFalse(connection.connect());
            Assert.assertTrue(connection.lastMessage(),
                    connection.lastMessage().contains("bypassed the control plane"));
            Assert.assertNull(connection.client());

            vars.put("BIOPB_TENSOR_TOKEN", "tok");
            Assert.assertTrue(connection.lastMessage(), connection.connect());
        }
        Assert.assertTrue("the control was never asked", ensureTokens.isEmpty());
    }

    @Test
    public void anExplicitUrlIsDialedWithTheTokenGivenAndNothingElse() throws Exception {
        startPlane("tok");
        writeCredential("tok");

        try (Connection connection = connection()) {
            Assert.assertFalse(connection.connect(planeUrl, null, Duration.ofSeconds(5)));
            Assert.assertEquals("Authentication required: the tensor server at " + planeUrl + " needs a token.",
                    connection.lastMessage());

            Assert.assertFalse(connection.connect(planeUrl, "wrong", Duration.ofSeconds(5)));
            Assert.assertTrue(connection.lastMessage(), connection.lastMessage().contains("rejected the token"));

            Assert.assertTrue(connection.lastMessage(), connection.connect(planeUrl, "tok", Duration.ofSeconds(5)));
        }
    }

    @Test
    public void anUnreachableAddressIsNotRetriedUnlessTheControlNamedIt() throws Exception {
        try (Connection connection = connection()) {
            long before = System.nanoTime();
            Assert.assertFalse(connection.connect("grpc://localhost:1", null, Duration.ofSeconds(30)));
            Assert.assertTrue("retried", Duration.ofNanos(System.nanoTime() - before).getSeconds() < 20);
            Assert.assertTrue(connection.lastMessage(), connection.lastMessage().startsWith("Cannot reach"));
        }
    }

    // ---- the contract's files and variables -----------------------------------------

    @Test
    public void theStateDirectoryMustBeAbsolute() {
        vars.put("BIOPB_STATE_HOME", "relative/dir");
        Assert.assertThrows(IllegalArgumentException.class, () -> DiscoveryEnvironment.of(vars, home).stateDir());
        vars.remove("BIOPB_STATE_HOME");
        Assert.assertEquals(home.resolve(".local").resolve("state").resolve("biopb"),
                DiscoveryEnvironment.of(vars, home).stateDir());
    }

    @Test
    public void theControlEndpointIsEnvironmentThenRecordThenDefault() throws Exception {
        Assert.assertEquals("http://127.0.0.1:8813", discovery().controlBaseUrl());

        Files.write(state.resolve("control.json"),
                "{\"host\":\"10.0.0.5\",\"port\":9000}".getBytes(StandardCharsets.UTF_8));
        Assert.assertEquals("http://10.0.0.5:9000", discovery().controlBaseUrl());

        // Each is resolved on its own.
        vars.put("BIOPB_CONTROL_PORT", "9100");
        Assert.assertEquals("http://10.0.0.5:9100", discovery().controlBaseUrl());
        vars.put("BIOPB_CONTROL_HOST", "example.org");
        Assert.assertEquals("http://example.org:9100", discovery().controlBaseUrl());

        // A malformed override falls to the default, never raises.
        vars.put("BIOPB_CONTROL_PORT", "not-a-port");
        Assert.assertEquals(8813, discovery().controlPort());

        // A malformed record is no record.
        vars.remove("BIOPB_CONTROL_PORT");
        vars.remove("BIOPB_CONTROL_HOST");
        Files.write(state.resolve("control.json"), "{not json".getBytes(StandardCharsets.UTF_8));
        Assert.assertEquals("http://127.0.0.1:8813", discovery().controlBaseUrl());
    }

    @Test
    public void aWildcardBindIsDialedOverLoopbackAndAnIpv6LiteralIsBracketed() {
        Assert.assertEquals("http://127.0.0.1:1", DataPlaneDiscovery.connectUrl("0.0.0.0", 1));
        Assert.assertEquals("http://[::1]:1", DataPlaneDiscovery.connectUrl("::", 1));
        Assert.assertEquals("http://[fe80::1]:1", DataPlaneDiscovery.connectUrl("fe80::1", 1));
        Assert.assertEquals("http://host:1", DataPlaneDiscovery.connectUrl("host", 1));
    }

    @Test
    public void theTokenIsExplicitThenEnvironmentThenTheFileOnlyWhenAllowed() throws Exception {
        writeCredential("from-file");
        Assert.assertEquals("from-file", discovery().resolveToken(null, true));
        Assert.assertNull(discovery().resolveToken(null, false));

        vars.put("BIOPB_TENSOR_TOKEN", "from-env");
        Assert.assertEquals("from-env", discovery().resolveToken(null, true));
        Assert.assertEquals("from-env", discovery().resolveToken(null, false));
        Assert.assertEquals("explicit", discovery().resolveToken(" explicit ", true));

        // Blank is none, never "" (an empty Bearer header).
        vars.put("BIOPB_TENSOR_TOKEN", "  ");
        Files.write(state.resolve("tensor-server.token"), "\n".getBytes(StandardCharsets.UTF_8));
        Assert.assertNull(discovery().resolveToken(null, true));
    }

    // ---- TLS anchors ------------------------------------------------------------------

    private byte[] fixture(String name) throws IOException {
        try (InputStream in = ConnectionTest.class.getResourceAsStream("/tls/" + name)) {
            Assert.assertNotNull(name, in);
            return in.readAllBytes();
        }
    }

    @Test
    public void aConfiguredCaWinsOverAFingerprintAndAControlNamedPlaneIgnoresBoth() throws Exception {
        Path ca = home.resolve("ca.pem");
        Files.write(ca, fixture("ca.pem"));
        vars.put("BIOPB_TENSOR_TLS_CA", ca.toString());
        vars.put("BIOPB_TENSOR_TLS_FINGERPRINT", "ab:cd");

        DataPlaneDiscovery.TlsAnchor anchor = discovery().dataPlaneTrust("grpcs://remote.example:8815", false);
        Assert.assertNotNull(anchor.caPem);
        Assert.assertNull("the CA wins", anchor.fingerprint);

        // A stale variable cannot point a local client at an old certificate.
        Files.write(state.resolve("tls-served.json"),
                "{\"8815\":{\"fingerprint\":\"served\"}}".getBytes(StandardCharsets.UTF_8));
        DataPlaneDiscovery.TlsAnchor named = discovery().dataPlaneTrust("grpcs://localhost:8815", true);
        Assert.assertNull(named.caPem);
        Assert.assertEquals("served", named.fingerprint);
    }

    @Test
    public void anUnusableConfiguredCaIsNamed() throws Exception {
        vars.put("BIOPB_TENSOR_TLS_CA", home.resolve("missing.pem").toString());
        TlsConfigException missing = Assert.assertThrows(TlsConfigException.class,
                () -> discovery().configuredTlsAnchor());
        Assert.assertTrue(missing.getMessage(), missing.getMessage().contains("not found"));

        Path junk = home.resolve("junk.pem");
        Files.write(junk, "hello".getBytes(StandardCharsets.UTF_8));
        vars.put("BIOPB_TENSOR_TLS_CA", junk.toString());
        Assert.assertTrue(Assert.assertThrows(TlsConfigException.class,
                () -> discovery().configuredTlsAnchor()).getMessage().contains("is not PEM"));

        Path key = home.resolve("key.pem");
        Files.write(key, "-----BEGIN ENCRYPTED PRIVATE KEY-----\n".getBytes(StandardCharsets.UTF_8));
        vars.put("BIOPB_TENSOR_TLS_CA", key.toString());
        Assert.assertTrue(Assert.assertThrows(TlsConfigException.class,
                () -> discovery().configuredTlsAnchor()).getMessage().contains("passphrase"));
    }

    @Test
    public void aLocalPlanesCertificateIsKnownFromWhatItPublishedThenWhatItMinted() throws Exception {
        String url = "grpcs://localhost:8815";
        Files.write(state.resolve("tls-served.json"),
                "{\"8815\":{\"fingerprint\":\"published\",\"pid\":1}}".getBytes(StandardCharsets.UTF_8));
        Assert.assertEquals("published", discovery().localDataPlaneFingerprint(url));

        // Another port has no entry: fall back to the minted certificate.
        Files.createDirectories(state.resolve("tls"));
        Files.write(state.resolve("tls").resolve("server-cert.pem"), fixture("server.pem"));
        Assert.assertEquals(TlsTrusts.fingerprint(fixture("server.pem")),
                discovery().localDataPlaneFingerprint("grpcs://127.0.0.1:9999"));
    }

    @Test
    public void aLocalTlsPlaneWithNothingToVerifyItAgainstIsAnErrorNotTrustOnFirstUse() {
        LocalTrustException error = Assert.assertThrows(LocalTrustException.class,
                () -> discovery().localDataPlaneFingerprint("grpcs://localhost:8815"));
        Assert.assertTrue(error.getMessage(), error.getMessage().contains("not retried as trust-on-first-use"));
    }

    @Test
    public void plaintextAndRemoteEndpointsHaveNoLocalIdentity() {
        Assert.assertNull(discovery().localDataPlaneFingerprint("grpc://localhost:8815"));
        Assert.assertNull(discovery().localDataPlaneFingerprint("grpcs://remote.example:8815"));
    }

    @Test
    public void aLocalTlsPlaneIsVerifiedAgainstWhatItPublished() throws Exception {
        File cert = copy("server.pem");
        File key = copy("server.key");
        plane = FlightServer.builder(allocator, Location.forGrpcTls("localhost", 0), new Plane())
                .useTls(cert, key)
                .build()
                .start();
        planeUrl = "grpcs://localhost:" + plane.getPort();
        startControl();
        Files.write(state.resolve("tls-served.json"),
                ("{\"" + plane.getPort() + "\":{\"fingerprint\":\"" + TlsTrusts.fingerprint(fixture("server.pem"))
                        + "\"}}").getBytes(StandardCharsets.UTF_8));

        try (Connection connection = connection()) {
            Assert.assertTrue(connection.lastMessage(), connection.connect());
        }
    }

    @Test
    public void aLocalTlsPlaneWithAWrongPublishedFingerprintIsRefusedByName() throws Exception {
        plane = FlightServer.builder(allocator, Location.forGrpcTls("localhost", 0), new Plane())
                .useTls(copy("server.pem"), copy("server.key"))
                .build()
                .start();
        planeUrl = "grpcs://localhost:" + plane.getPort();
        startControl();
        Files.write(state.resolve("tls-served.json"),
                ("{\"" + plane.getPort() + "\":{\"fingerprint\":\"" + TlsTrusts.fingerprint(fixture("othername.pem"))
                        + "\"}}").getBytes(StandardCharsets.UTF_8));

        try (Connection connection = connection()) {
            Assert.assertFalse(connection.connect(null, null, Duration.ofMillis(1500)));
            Assert.assertTrue(connection.lastMessage(), connection.lastMessage().contains("configured fingerprint"));
        }
    }

    @Test
    public void aLocalTlsPlaneWithNoRecordIsReportedAsATrustProblemNotAnAuthOne() throws Exception {
        planeUrl = "grpcs://localhost:" + 1;
        startControl();

        try (Connection connection = connection()) {
            Assert.assertFalse(connection.connect(null, null, Duration.ofMillis(1500)));
            Assert.assertTrue(connection.lastMessage(), connection.lastMessage().contains("nothing on this machine"));
            Assert.assertFalse(connection.lastMessage(), connection.lastMessage().contains("Authentication"));
        }
    }

    private File copy(String name) throws IOException {
        Path target = home.resolve(name);
        Files.write(target, fixture(name));
        return target.toFile();
    }
}
