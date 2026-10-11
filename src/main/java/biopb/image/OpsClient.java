package biopb.image;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import com.google.protobuf.Empty;

import biopb.tensor.ControlClient;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.ClientResponseObserver;
import io.grpc.stub.MetadataUtils;

/**
 * A client of an algorithm server: the {@code Ops} protocol over gRPC.
 *
 * <p>The Java twin of Python's {@code biopb.image.OpsClient}. Use {@link
 * #connect(String)} to make one. A call has no deadline: the inactivity timeout
 * bounds the silence between events instead, and leaving a call (an error, a
 * sink that throws, a timeout) cancels it on the server.
 */
public final class OpsClient implements AutoCloseable {

    /**
     * How long {@code connect} may wait for the control to bring a registry entry
     * up: a first start installs, and may pull torch.
     */
    private static final Duration ENSURE_TIMEOUT = Duration.ofSeconds(900);

    /** Ends the event queue: the server closed the stream. */
    private static final Object END = new Object();

    private final String url;
    private final ManagedChannel channel;
    private final Duration inactivityTimeout;
    private final Metadata headers;

    /**
     * A client of the server at {@code url}, a {@code grpc://} or {@code grpcs://}
     * address.
     *
     * @param token bearer token, or null for none
     * @param inactivityTimeout longest silence between events before a call is
     *        given up on, or null for none
     * @throws IllegalArgumentException the URL has no host or another scheme
     */
    public OpsClient(String url, String token, Duration inactivityTimeout) {
        this(url, makeChannel(url), token, inactivityTimeout);
    }

    /**
     * A client over a channel the caller built (to set message size limits,
     * credentials or a proxy). The client owns it: {@link #close} shuts it down.
     */
    public OpsClient(ManagedChannel channel, String token, Duration inactivityTimeout) {
        this(channel.authority(), channel, token, inactivityTimeout);
    }

    private OpsClient(String url, ManagedChannel channel, String token, Duration inactivityTimeout) {
        this.url = url;
        this.channel = channel;
        this.inactivityTimeout = inactivityTimeout;
        this.headers = new Metadata();
        if (token != null && !token.isEmpty()) {
            headers.put(Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER), "Bearer " + token);
        }
    }

    /**
     * A client of the algorithm server at {@code target}: a {@code grpc://} or
     * {@code grpcs://} URL, or the name of a registry entry the control manages.
     * A name is brought up through the control first (installing it if needed)
     * and takes the server's own token; an explicit {@code token} overrides it.
     *
     * @throws java.util.NoSuchElementException a name the control does not know
     * @throws IllegalStateException the entry did not come up, or no control answered
     */
    public static OpsClient connect(String target) {
        return connect(target, null, null);
    }

    /** {@link #connect(String)} with an explicit token and inactivity timeout (either may be null). */
    public static OpsClient connect(String target, String token, Duration inactivityTimeout) {
        return connect(target, token, inactivityTimeout, ControlClient.system());
    }

    /** {@link #connect(String, String, Duration)} asking a given control, for tests and embedding. */
    public static OpsClient connect(String target, String token, Duration inactivityTimeout, ControlClient control) {
        if (!target.contains("://")) {
            Map<String, Object> row = control.ensureAlgorithm(target, ENSURE_TIMEOUT);
            Object state = row.get("state");
            if (!"up".equals(state)) {
                Object error = row.get("error");
                String detail = error == null ? "" : error.toString().trim();
                throw new IllegalStateException("algorithm server '" + target + "' is " + state
                        + (detail.isEmpty() ? "" : ":\n" + detail));
            }
            Object rowToken = row.get("token");
            return new OpsClient(String.valueOf(row.get("url")),
                    token != null && !token.isEmpty() ? token : rowToken == null ? null : rowToken.toString(),
                    inactivityTimeout);
        }
        return new OpsClient(target, token, inactivityTimeout);
    }

    /** The address this client dials. */
    public String url() {
        return url;
    }

    /** Shut the channel down, waiting briefly for calls to finish. */
    @Override
    public void close() {
        channel.shutdown();
        try {
            if (!channel.awaitTermination(5, TimeUnit.SECONDS)) {
                channel.shutdownNow();
            }
        } catch (InterruptedException error) {
            channel.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    /** The ops the server offers, within 10 seconds. */
    public OpList describe() {
        return describe(Duration.ofSeconds(10));
    }

    /** The ops the server offers. */
    public OpList describe(Duration timeout) {
        try {
            return OpsGrpc.newBlockingStub(channel)
                    .withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers))
                    .withDeadlineAfter(timeout.toMillis(), TimeUnit.MILLISECONDS)
                    .describe(Empty.getDefaultInstance());
        } catch (StatusRuntimeException error) {
            throw opError("describe", error);
        }
    }

    /**
     * Hand {@code sink} every event of one call, progress-only ones included,
     * until the server ends the stream. Returns when it does; a {@code sink} that
     * throws ends the call, which is cancelled on the server.
     *
     * @throws OpTimeoutException no event arrived within the inactivity timeout
     * @throws IllegalArgumentException the server refused the arguments
     * @throws NoSuchElementException the server has no such op
     * @throws IllegalStateException any other failure
     */
    public void events(String op, Map<String, Arg> args, Consumer<Event> sink) {
        biopb.image.Call request = biopb.image.Call.newBuilder().setOp(op).putAllArgs(args).build();
        // An Event, the Throwable that ended the call, or END.
        BlockingQueue<Object> items = new LinkedBlockingQueue<>();
        AtomicReference<ClientCallStreamObserver<biopb.image.Call>> handle = new AtomicReference<>();
        OpsGrpc.newStub(channel)
                .withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers))
                .call(request, new ClientResponseObserver<biopb.image.Call, Event>() {
                    @Override
                    public void beforeStart(ClientCallStreamObserver<biopb.image.Call> requestStream) {
                        handle.set(requestStream);
                    }

                    @Override
                    public void onNext(Event event) {
                        items.add(event);
                    }

                    @Override
                    public void onError(Throwable error) {
                        items.add(error);
                    }

                    @Override
                    public void onCompleted() {
                        items.add(END);
                    }
                });
        try {
            while (true) {
                Object item = inactivityTimeout == null
                        ? items.take()
                        : items.poll(inactivityTimeout.toNanos(), TimeUnit.NANOSECONDS);
                if (item == null) {
                    throw new OpTimeoutException(op + ": no word from the server in "
                            + inactivityTimeout.toMillis() / 1000.0 + " s");
                }
                if (item == END) {
                    return;
                }
                if (item instanceof Throwable) {
                    Throwable error = (Throwable) item;
                    throw error instanceof StatusRuntimeException
                            ? opError(op, (StatusRuntimeException) error)
                            : new IllegalStateException(op + ": " + error, error);
                }
                sink.accept((Event) item);
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(op + ": interrupted", error);
        } finally {
            // A stop, a timeout or an error: the server stops computing for a
            // caller that left. A no-op once the call ended.
            ClientCallStreamObserver<biopb.image.Call> call = handle.get();
            if (call != null) {
                call.cancel("caller left", null);
            }
        }
    }

    /** {@link #call(String, Map, Consumer, List)} with no progress sink and default axis labels. */
    public Object call(String op, Map<String, ?> values) {
        return call(op, values, null, null);
    }

    /**
     * Run one op on {@code values}, by name: arrays go as pixels, anything else as
     * JSON (see {@link ArgCodec#encodeArg}).
     *
     * <p>Returns the op's result: its single output; an {@code Object[]} of them
     * in order for several; {@code null} for no output. A streaming op returns a
     * {@code List<Object>} with one such value per event.
     *
     * @param onProgress receives each progress message, or null
     * @param dimLabels labels every array argument's axes, or null for the
     *        defaults ({@link ArgCodec#NDIM_LABELS})
     * @throws IllegalArgumentException the server refused the arguments
     * @throws NoSuchElementException the server has no such op
     * @throws IllegalStateException any other failure
     */
    public Object call(String op, Map<String, ?> values, Consumer<String> onProgress, List<String> dimLabels) {
        Map<String, Arg> args = new LinkedHashMap<>();
        for (Map.Entry<String, ?> value : values.entrySet()) {
            args.put(value.getKey(), ArgCodec.encodeArg(value.getValue(), dimLabels));
        }
        List<Object> results = new ArrayList<>();
        events(op, args, event -> {
            if (event.getOutputsCount() > 0) {
                results.add(result(event));
            } else if (!event.getProgress().isEmpty() && onProgress != null) {
                onProgress.accept(event.getProgress());
            }
        });
        if (results.isEmpty()) {
            return null;
        }
        return results.size() == 1 ? results.get(0) : results;
    }

    private static Object result(Event event) {
        Map<String, Arg> outputs = event.getOutputsMap();
        if (outputs.size() == 1 && outputs.containsKey("result")) {
            return ArgCodec.decodeArg(outputs.get("result"));
        }
        // Numbered outputs in numeric order, then any named ones.
        List<String> keys = new ArrayList<>(outputs.keySet());
        Collections.sort(keys, Comparator
                .comparing((String key) -> !isDigits(key))
                .thenComparingInt(key -> isDigits(key) ? Integer.parseInt(key) : 0)
                .thenComparing(Comparator.naturalOrder()));
        Object[] values = new Object[keys.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = ArgCodec.decodeArg(outputs.get(keys.get(i)));
        }
        return values;
    }

    private static boolean isDigits(String key) {
        if (key.isEmpty() || key.length() > 9) {
            return false;
        }
        for (int i = 0; i < key.length(); i++) {
            if (!Character.isDigit(key.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * A channel to a {@code grpc://} or {@code grpcs://} URL.
     *
     * @throws IllegalArgumentException no host, or another scheme
     */
    public static ManagedChannel makeChannel(String url) {
        int separator = url.indexOf("://");
        String scheme = separator < 0 ? "" : url.substring(0, separator).toLowerCase(java.util.Locale.ROOT);
        String rest = separator < 0 ? url : url.substring(separator + 3);
        int slash = rest.indexOf('/');
        String target = slash < 0 ? rest : rest.substring(0, slash);
        if (target.isEmpty()) {
            throw new IllegalArgumentException("algorithm server URL has no host: '" + url + "'");
        }
        if (scheme.equals("grpcs")) {
            return ManagedChannelBuilder.forTarget(target).useTransportSecurity().build();
        }
        if (scheme.equals("grpc")) {
            return ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        }
        throw new IllegalArgumentException("algorithm server URL must be grpc:// or grpcs://, got '" + url + "'");
    }

    /**
     * The exception a failed call raises: the server's message, typed by what went
     * wrong rather than wrapped in gRPC's own. {@link IllegalArgumentException} for
     * arguments the op refuses, {@link NoSuchElementException} for an op the
     * server lacks, {@link IllegalStateException} otherwise.
     */
    static RuntimeException opError(String op, StatusRuntimeException error) {
        Status status = error.getStatus();
        String description = status.getDescription() == null ? "" : status.getDescription().trim();
        String detail = description.isEmpty() ? status.getCode().name() : description;
        switch (status.getCode()) {
            case INVALID_ARGUMENT:
                return new IllegalArgumentException(op + ": " + detail);
            case NOT_FOUND:
                return new NoSuchElementException(op + ": " + detail);
            default:
                return new IllegalStateException(op + ": " + status.getCode().name() + ": " + detail, error);
        }
    }
}
