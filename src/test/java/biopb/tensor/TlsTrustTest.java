package biopb.tensor;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightProducer;
import org.apache.arrow.flight.FlightServer;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.NoOpFlightProducer;
import org.apache.arrow.flight.Result;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * TLS trust against a real {@code grpc+tls://} Flight server.
 *
 * <p>The fixtures are in {@code src/test/resources/tls}, made by the
 * {@code generate.py} beside them: a self-signed leaf for {@code localhost}, one
 * that lists only {@code other.example}, an expired one, and a private CA with a
 * leaf for {@code other.example}.
 */
public class TlsTrustTest {

    private Path dir;
    private BufferAllocator allocator;
    private FlightServer server;

    @Before
    public void setUp() throws IOException {
        dir = Files.createTempDirectory("biopb-tls");
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @After
    public void tearDown() throws Exception {
        if (server != null) {
            server.close();
        }
        allocator.close();
        try (java.util.stream.Stream<Path> files = Files.list(dir)) {
            for (Path file : (Iterable<Path>) files::iterator) {
                Files.deleteIfExists(file);
            }
        }
        Files.deleteIfExists(dir);
    }

    private File fixture(String name) throws IOException {
        Path target = dir.resolve(name);
        if (!Files.exists(target)) {
            try (InputStream in = TlsTrustTest.class.getResourceAsStream("/tls/" + name)) {
                Assert.assertNotNull("missing fixture " + name, in);
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        return target.toFile();
    }

    private byte[] pem(String stem) throws IOException {
        return Files.readAllBytes(fixture(stem + ".pem").toPath());
    }

    /** Serve {@code stem}'s certificate over TLS and answer {@code health}. */
    private Location serve(String stem) throws Exception {
        if (server != null) {
            server.close();
        }
        server = FlightServer.builder(allocator, Location.forGrpcTls("localhost", 0), new HealthProducer())
                .useTls(fixture(stem + ".pem"), fixture(stem + ".key"))
                .build()
                .start();
        return Location.forGrpcTls("localhost", server.getPort());
    }

    private static final class HealthProducer extends NoOpFlightProducer {
        @Override
        public void doAction(
                FlightProducer.CallContext context, Action action, FlightProducer.StreamListener<Result> listener) {
            listener.onNext(new Result("{\"status\":\"SERVING\",\"protocol\":2}".getBytes(StandardCharsets.UTF_8)));
            listener.onCompleted();
        }
    }

    private static Map<String, Object> health(Location location, TlsTrust trust) throws IOException {
        try (TensorFlightClient client = new TensorFlightClient(location, 1_000_000L, null, trust)) {
            return client.healthCheck();
        }
    }

    // ---- the modes --------------------------------------------------------

    @Test
    public void aServerTheJdkDoesNotKnowIsRefusedWithNoTrust() throws Exception {
        Location location = serve("server");
        Assert.assertThrows(RuntimeException.class, () -> health(location, TlsTrust.NONE));
    }

    @Test
    public void aConfiguredCaAnchorsTheConnectionAndStaysOffline() throws Exception {
        Location location = serve("server");
        TlsTrust trust = TlsTrusts.resolve(location, pem("server"), null, null);

        Assert.assertNull("a configured anchor never carries an override", trust.overrideHostname());
        Assert.assertEquals("SERVING", health(location, trust).get("status"));
    }

    @Test
    public void aFingerprintThatMatchesConnects() throws Exception {
        Location location = serve("server");
        String fingerprint = TlsTrusts.fingerprint(pem("server"));
        // Both spellings an operator pastes, any case.
        String colonGrouped = fingerprint.toUpperCase().replaceAll("(..)(?!$)", "$1:");

        TlsTrust trust = TlsTrusts.resolve(location, null, colonGrouped, null);

        Assert.assertEquals("SERVING", health(location, trust).get("status"));
    }

    @Test
    public void aWrongFingerprintIsRefusedOnTheFirstConnect() throws Exception {
        Location location = serve("server");
        String other = TlsTrusts.fingerprint(pem("othername"));

        TlsPinMismatchException error = Assert.assertThrows(TlsPinMismatchException.class,
                () -> TlsTrusts.resolve(location, null, other, null));
        Assert.assertTrue(error.getMessage(), error.getMessage().contains("configured fingerprint"));
    }

    @Test
    public void trustOnFirstUsePinsThenHoldsTheServerToThePin() throws Exception {
        Location location = serve("server");
        TlsPinStore pins = TlsPinStore.inMemory();

        TlsTrust first = TlsTrusts.resolve(location, null, null, pins);
        String key = "localhost:" + server.getPort();
        Assert.assertNotNull("first use pins the certificate", pins.get(key));
        Assert.assertEquals("SERVING", health(location, first).get("status"));

        TlsTrust again = TlsTrusts.resolve(location, null, null, pins);
        Assert.assertEquals(first, again);

        // The same endpoint now presenting another certificate is the SSH
        // "host identification has changed" case.
        pins.put(key, new String(pem("othername"), StandardCharsets.US_ASCII));
        TlsPinMismatchException error = Assert.assertThrows(TlsPinMismatchException.class,
                () -> TlsTrusts.resolve(location, null, null, pins));
        Assert.assertTrue(error.getMessage(), error.getMessage().contains("pinned"));
    }

    @Test
    public void trustOnFirstUseWithoutAStoreIsRefused() throws Exception {
        Location location = serve("server");
        Assert.assertThrows(IllegalArgumentException.class,
                () -> TlsTrusts.resolve(location, null, null, null));
    }

    @Test
    public void aPlaintextLocationNeedsNoTrust() throws Exception {
        Assert.assertSame(TlsTrust.NONE,
                TlsTrusts.resolve(Location.forGrpcInsecure("localhost", 1), null, null, null));
    }

    // ---- the hostname override ---------------------------------------------

    @Test
    public void aPinnedLeafThatOmitsTheDialedNameVerifiesAgainstOneItDoesList() throws Exception {
        Location location = serve("othername"); // dialed as localhost; SAN is other.example
        TlsTrust trust = TlsTrusts.resolve(location, null, null, TlsPinStore.inMemory());

        Assert.assertEquals("other.example", trust.overrideHostname());
        Assert.assertEquals("SERVING", health(location, trust).get("status"));
    }

    @Test
    public void withoutTheOverrideThatCertificateFailsHostnameVerification() throws Exception {
        Location location = serve("othername");
        TlsTrust anchorOnly = TlsTrusts.resolve(location, pem("othername"), null, null);

        Assert.assertNull(anchorOnly.overrideHostname());
        Assert.assertThrows(RuntimeException.class, () -> health(location, anchorOnly));
    }

    @Test
    public void aCaAnchorKeepsTheNameCheck() throws Exception {
        // The leaf is validly issued by the anchor but lists only other.example.
        // Substituting a name here would let any host in that PKI stand in for
        // any other, so the check stays and the connection fails.
        Location location = serve("ca-leaf");
        TlsTrust trust = TlsTrusts.concrete(location, TlsTrusts.anchored(pem("ca")));

        Assert.assertNull(trust.overrideHostname());
        Assert.assertThrows(RuntimeException.class, () -> health(location, trust));
    }

    @Test
    public void anAnchoredTrustResolvesItsOwnNameCheckOnTheConsumersNetwork() throws Exception {
        Location location = serve("othername");
        TlsTrust sent = TlsTrusts.anchored(pem("othername"));
        Assert.assertTrue(sent.reresolve());

        TlsTrust trust = TlsTrusts.concrete(location, sent);

        Assert.assertEquals("other.example", trust.overrideHostname());
        Assert.assertEquals("SERVING", health(location, trust).get("status"));
    }

    @Test
    public void anAnchoredTrustMustBeMadeConcreteBeforeASessionIsOpened() throws Exception {
        Location location = serve("server");
        Assert.assertThrows(IllegalArgumentException.class,
                () -> new FlightSession(location, null, TlsTrusts.anchored(pem("server"))));
    }

    // ---- expiry -------------------------------------------------------------

    @Test
    public void anExpiredServerCertificateIsNamedNotLeftAsAnUnexplainedFailure() throws Exception {
        Location location = serve("expired");
        TlsCertExpiredException error = Assert.assertThrows(TlsCertExpiredException.class,
                () -> TlsTrusts.resolve(location, null, null, TlsPinStore.inMemory()));
        Assert.assertTrue(error.getMessage(), error.getMessage().contains("has expired"));
    }

    @Test
    public void aConfiguredExpiredAnchorIsNamedWithoutTouchingTheNetwork() throws Exception {
        // No server is running: a configured anchor is read offline.
        Location location = Location.forGrpcTls("localhost", 1);
        Assert.assertThrows(TlsCertExpiredException.class,
                () -> TlsTrusts.resolve(location, pem("expired"), null, null));
    }

    // ---- the connection cache -------------------------------------------------

    @Test
    public void twoAnchorsOnOneEndpointAreNotHandedEachOthersConnection() throws Exception {
        Location location = serve("server");
        TlsTrust a = TlsTrusts.resolve(location, pem("server"), null, null);
        TlsTrust b = TlsTrusts.resolve(location, pem("othername"), null, null);
        Assert.assertNotEquals(a.keyId(), b.keyId());

        FlightSession one = FlightSessions.shared(location, "t", a);
        FlightSession two = FlightSessions.shared(location, "t", b);
        try {
            Assert.assertNotSame(one, two);
            Assert.assertSame(one, FlightSessions.shared(location, "t", a));
        } finally {
            FlightSessions.closeAll();
        }
    }
}
