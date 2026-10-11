package biopb.image;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.google.protobuf.Empty;
import com.sun.net.httpserver.HttpServer;

import biopb.tensor.ControlClient;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;

/** The {@code biopb.image} Ops codec and client, over an in-process gRPC server. */
public class OpsClientTest {

    private final List<Server> servers = new ArrayList<>();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private Path home;
    private HttpServer control;

    @Before
    public void setUp() throws IOException {
        home = Files.createTempDirectory("biopb-ops");
    }

    @After
    public void tearDown() throws IOException {
        for (Server server : servers) {
            server.shutdownNow();
        }
        if (control != null) {
            control.stop(0);
        }
        Files.deleteIfExists(home);
    }

    /** {@code echo} returns its argument; {@code pair} two outputs; {@code slow} never answers. */
    private final class Ops extends OpsGrpc.OpsImplBase {
        @Override
        public void describe(Empty request, StreamObserver<OpList> out) {
            out.onNext(OpList.newBuilder()
                    .addOps(OpInfo.newBuilder().setName("echo"))
                    .addOps(OpInfo.newBuilder().setName("pair"))
                    .setFingerprint("f")
                    .build());
            out.onCompleted();
        }

        @Override
        public void call(biopb.image.Call request, StreamObserver<Event> out) {
            switch (request.getOp()) {
                case "echo":
                    out.onNext(Event.newBuilder().setProgress("working").build());
                    out.onNext(Event.newBuilder().putOutputs("result", request.getArgsOrThrow("x")).build());
                    out.onCompleted();
                    break;
                case "pair":
                    out.onNext(Event.newBuilder()
                            .putOutputs("1", ArgCodec.jsonArg("b"))
                            .putOutputs("0", ArgCodec.jsonArg(7))
                            .build());
                    out.onCompleted();
                    break;
                case "stream":
                    out.onNext(Event.newBuilder().putOutputs("result", ArgCodec.jsonArg(1)).build());
                    out.onNext(Event.newBuilder().putOutputs("result", ArgCodec.jsonArg(2)).build());
                    out.onCompleted();
                    break;
                case "slow":
                    ((ServerCallStreamObserver<Event>) out).setOnCancelHandler(() -> cancelled.set(true));
                    break; // never completes: the client has to leave
                case "refuse":
                    out.onError(Status.INVALID_ARGUMENT.withDescription("bad sigma").asRuntimeException());
                    break;
                default:
                    out.onError(Status.NOT_FOUND.withDescription("no op " + request.getOp()).asRuntimeException());
            }
        }
    }

    private String serve(String requireToken) throws IOException {
        ServerInterceptor auth = new ServerInterceptor() {
            @Override
            public <A, B> ServerCall.Listener<A> interceptCall(
                    ServerCall<A, B> call, Metadata headers, ServerCallHandler<A, B> next) {
                String given = headers.get(Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER));
                if (requireToken != null && !("Bearer " + requireToken).equals(given)) {
                    call.close(Status.UNAUTHENTICATED.withDescription("token"), new Metadata());
                    return new ServerCall.Listener<A>() { };
                }
                return next.startCall(call, headers);
            }
        };
        Server server = ServerBuilder.forPort(0)
                .addService(ServerInterceptors.intercept(new Ops(), auth))
                .build()
                .start();
        servers.add(server);
        return "grpc://127.0.0.1:" + server.getPort();
    }

    private static Map<String, Object> args(Object... keyValues) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < keyValues.length; i += 2) {
            out.put((String) keyValues[i], keyValues[i + 1]);
        }
        return out;
    }

    // ---- the codec -----------------------------------------------------------------

    @Test
    public void jsonValuesRoundTrip() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("k", Arrays.asList(1L, 2L));
        nested.put("z", Collections.singletonMap("a", 1L));
        for (Object value : new Object[] { null, true, "text", 3L, 2.5, Arrays.asList(1L, "a", null), nested }) {
            Assert.assertEquals(value, ArgCodec.decodeArg(ArgCodec.encodeArg(value)));
        }
    }

    @Test
    public void jsonKeepsNanAndInf() {
        List<?> out = (List<?>) ArgCodec.decodeArg(ArgCodec.encodeArg(Arrays.asList(Double.NaN, Double.POSITIVE_INFINITY)));
        Assert.assertTrue(Double.isNaN((Double) out.get(0)));
        Assert.assertEquals(Double.POSITIVE_INFINITY, (Double) out.get(1), 0.0);
    }

    @Test
    public void integralNumbersReadAsLongsUnlessAskedNotTo() {
        Arg arg = ArgCodec.encodeArg(6);
        Assert.assertEquals(6L, ArgCodec.decodeArg(arg));
        Assert.assertEquals(6.0, ArgCodec.decodeArg(arg, false));
        Assert.assertEquals(2.5, ArgCodec.decodeArg(ArgCodec.encodeArg(2.5)));
    }

    @Test
    public void primitiveArraysAreJsonLists() {
        Assert.assertEquals(Arrays.asList(0L, 1L, 2L), ArgCodec.decodeArg(ArgCodec.encodeArg(new int[] { 0, 1, 2 })));
        Assert.assertEquals(Arrays.asList(1.5, 2.5), ArgCodec.decodeArg(ArgCodec.encodeArg(new double[] { 1.5, 2.5 })));
    }

    @Test
    public void aValueWithNoJsonFormIsRefused() {
        IllegalArgumentException error = Assert.assertThrows(IllegalArgumentException.class,
                () -> ArgCodec.encodeArg(new Object()));
        Assert.assertTrue(error.getMessage(), error.getMessage().contains("not JSON"));
    }

    private static short[] ramp(int n) {
        short[] values = new short[n];
        for (int i = 0; i < n; i++) {
            values[i] = (short) (i * 7);
        }
        return values;
    }

    @Test
    public void anArrayRoundTripsAsPixelsWithDefaultLabels() {
        RandomAccessibleInterval<UnsignedShortType> array = ArrayImgs.unsignedShorts(ramp(12), 4, 3);
        Arg arg = ArgCodec.encodeArg(array);

        Assert.assertEquals(Arg.KindCase.EAGER, arg.getKindCase());
        Assert.assertEquals(Arrays.asList("Y", "X"), arg.getEager().getDimLabelsList());
        Assert.assertEquals(Arrays.asList(4, 3), arg.getEager().getDimsList());

        RandomAccessibleInterval<?> back = (RandomAccessibleInterval<?>) ArgCodec.decodeArg(arg);
        Assert.assertEquals(4, back.dimension(0));
        Assert.assertEquals(3, back.dimension(1));
        Cursor<UnsignedShortType> expected = array.cursor();
        Cursor<?> actual = back.cursor();
        while (expected.hasNext()) {
            Assert.assertEquals(expected.next().getRealDouble(), ((RealType<?>) actual.next()).getRealDouble(), 0.0);
        }
    }

    @Test
    public void anArraysLabelsAreTheCallersWhenGiven() {
        Arg arg = ArgCodec.encodeArg(ArrayImgs.unsignedShorts(ramp(24), 4, 3, 2), Arrays.asList("X", "Y", "Z"));
        Assert.assertEquals(Arrays.asList("X", "Y", "Z"), arg.getEager().getDimLabelsList());
    }

    @Test
    public void anEmptyArgHasNoValue() {
        IllegalArgumentException error = Assert.assertThrows(IllegalArgumentException.class,
                () -> ArgCodec.decodeArg(Arg.getDefaultInstance()));
        Assert.assertTrue(error.getMessage().contains("empty"));
    }

    // ---- the client ------------------------------------------------------------------

    @Test
    public void describeListsTheOps() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            List<String> names = new ArrayList<>();
            client.describe().getOpsList().forEach(op -> names.add(op.getName()));
            Assert.assertEquals(Arrays.asList("echo", "pair"), names);
        }
    }

    @Test
    public void callReturnsTheSingleOutputAndReportsProgress() throws Exception {
        List<String> seen = new ArrayList<>();
        RandomAccessibleInterval<UnsignedShortType> array = ArrayImgs.unsignedShorts(ramp(6), 3, 2);
        try (OpsClient client = OpsClient.connect(serve(null))) {
            Object out = client.call("echo", args("x", array), seen::add, null);
            Assert.assertTrue(out instanceof RandomAccessibleInterval);
            Assert.assertEquals(3, ((RandomAccessibleInterval<?>) out).dimension(0));
        }
        Assert.assertEquals(Collections.singletonList("working"), seen);
    }

    @Test
    public void callReturnsJsonUnchanged() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            Assert.assertEquals(Collections.singletonMap("sigma", 1.5),
                    client.call("echo", args("x", Collections.singletonMap("sigma", 1.5))));
        }
    }

    @Test
    public void callOrdersSeveralOutputsAsAnArray() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            Assert.assertArrayEquals(new Object[] { 7L, "b" }, (Object[]) client.call("pair", args()));
        }
    }

    @Test
    public void aStreamingOpReturnsOneValuePerEvent() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            Assert.assertEquals(Arrays.asList(1L, 2L), client.call("stream", args()));
        }
    }

    @Test
    public void aRefusedCallIsAnIllegalArgument() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            IllegalArgumentException error = Assert.assertThrows(IllegalArgumentException.class,
                    () -> client.call("refuse", args()));
            Assert.assertEquals("refuse: bad sigma", error.getMessage());
        }
    }

    @Test
    public void anUnknownOpIsNoSuchElement() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            Assert.assertThrows(NoSuchElementException.class, () -> client.call("nope", args()));
        }
    }

    @Test
    public void theTokenIsSent() throws Exception {
        String url = serve("secret");
        try (OpsClient client = OpsClient.connect(url)) {
            IllegalStateException error = Assert.assertThrows(IllegalStateException.class, client::describe);
            Assert.assertTrue(error.getMessage(), error.getMessage().contains("UNAUTHENTICATED"));
        }
        try (OpsClient client = OpsClient.connect(url, "secret", null)) {
            Assert.assertEquals(2, client.describe().getOpsCount());
        }
    }

    @Test
    public void silenceTimesOutAndCancelsTheCallOnTheServer() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null), null, Duration.ofMillis(300))) {
            OpTimeoutException error = Assert.assertThrows(OpTimeoutException.class,
                    () -> client.call("slow", args()));
            Assert.assertTrue(error.getMessage(), error.getMessage().contains("slow"));
        }
        long deadline = System.currentTimeMillis() + 3000;
        while (!cancelled.get() && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        Assert.assertTrue("the server was never told the caller left", cancelled.get());
    }

    @Test
    public void aSinkThatThrowsCancelsTheCall() throws Exception {
        try (OpsClient client = OpsClient.connect(serve(null))) {
            Assert.assertThrows(IllegalStateException.class, () -> client.events("echo", Collections.singletonMap("x", ArgCodec.jsonArg(1)), event -> {
                throw new IllegalStateException("enough");
            }));
        }
    }

    @Test
    public void aUrlNeedsAHostAndAKnownScheme() {
        Assert.assertTrue(Assert.assertThrows(IllegalArgumentException.class,
                () -> OpsClient.makeChannel("grpc://")).getMessage().contains("no host"));
        Assert.assertTrue(Assert.assertThrows(IllegalArgumentException.class,
                () -> OpsClient.makeChannel("http://x:1")).getMessage().contains("grpc:// or grpcs://"));
    }

    // ---- a name goes through the control -----------------------------------------------

    private ControlClient control(String state, String url, String token, String error) throws IOException {
        control = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        control.createContext("/api/algorithms/ensure", exchange -> {
            String query = exchange.getRequestURI().getQuery();
            int status = 200;
            String body;
            if (!"POST".equals(exchange.getRequestMethod()) || !query.contains("name=cellpose")) {
                status = query.contains("name=ghost") ? 404 : 400;
                body = "{\"error\":\"no such entry\"}";
            } else {
                body = "{\"server\":{\"name\":\"cellpose\",\"state\":\"" + state + "\",\"url\":\"" + url
                        + "\",\"token\":" + (token == null ? "null" : "\"" + token + "\"") + ",\"error\":"
                        + (error == null ? "null" : "\"" + error + "\"") + "}}";
            }
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(status, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        control.start();
        Map<String, String> vars = new HashMap<>();
        vars.put("BIOPB_STATE_HOME", home.toString());
        vars.put("BIOPB_CONTROL_PORT", String.valueOf(control.getAddress().getPort()));
        return ControlClient.of(vars, home);
    }

    @Test
    public void aNameIsBroughtUpThroughTheControlAndTakesTheServersToken() throws Exception {
        String url = serve("t");
        try (OpsClient client = OpsClient.connect("cellpose", null, null, control("up", url, "t", null))) {
            Assert.assertEquals(url, client.url());
            Assert.assertEquals(2, client.describe().getOpsCount());
        }
    }

    @Test
    public void anExplicitTokenOverridesTheServersOwn() throws Exception {
        String url = serve("mine");
        try (OpsClient client = OpsClient.connect("cellpose", "mine", null, control("up", url, "theirs", null))) {
            Assert.assertEquals(2, client.describe().getOpsCount());
        }
    }

    @Test
    public void aNameThatIsNotUpIsAnIllegalState() throws Exception {
        ControlClient client = control("error", "", null, "no gpu");
        IllegalStateException error = Assert.assertThrows(IllegalStateException.class,
                () -> OpsClient.connect("cellpose", null, null, client));
        Assert.assertTrue(error.getMessage(), error.getMessage().contains("is error:\nno gpu"));
    }

    @Test
    public void anUnknownNameIsNoSuchElement() throws Exception {
        ControlClient client = control("up", "", null, null);
        Assert.assertThrows(NoSuchElementException.class,
                () -> OpsClient.connect("ghost", null, null, client));
    }
}
