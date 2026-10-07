package biopb.tensor;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Arrays;
import java.util.logging.Logger;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Where the data plane is, what credential reaches it and what anchors its TLS:
 * the Java implementation of {@code docs/discovery-contract.md}.
 *
 * <p>Files, environment variables and two HTTP calls to the control, so it
 * mirrors Python's {@code biopb._control} line for line in what it reads and in
 * what order. The order is <b>ask, then guess</b>, and the credential and the TLS
 * anchor both follow the address: an address that routed around the control gets
 * an explicit token or none, and never this machine's credential file.
 */
final class DataPlaneDiscovery {

    private static final Logger LOGGER = Logger.getLogger(DataPlaneDiscovery.class.getName());

    static final String ENV_TENSOR_URL = "BIOPB_TENSOR_URL";
    static final String ENV_TENSOR_TOKEN = "BIOPB_TENSOR_TOKEN";
    static final String ENV_TENSOR_TLS_CA = "BIOPB_TENSOR_TLS_CA";
    static final String ENV_TENSOR_TLS_FINGERPRINT = "BIOPB_TENSOR_TLS_FINGERPRINT";

    private static final String CONTROL_DEFAULT_HOST = "127.0.0.1";
    private static final int CONTROL_DEFAULT_PORT = 8813;
    private static final String CREDENTIAL_FILE = "tensor-server.token";

    /** A plane the control named: where to dial and the token that goes with it. */
    static final class DataPlane {
        final String url;
        final String token;

        DataPlane(String url, String token) {
            this.url = url;
            this.token = token;
        }
    }

    /**
     * What a client trusts of a TLS server: a CA or leaf PEM, or a leaf's SHA-256.
     * At most one is set; neither leaves the client to trust on first use.
     */
    static final class TlsAnchor {
        static final TlsAnchor NONE = new TlsAnchor(null, null);

        final byte[] caPem;
        final String fingerprint;

        TlsAnchor(byte[] caPem, String fingerprint) {
            this.caPem = caPem;
            this.fingerprint = fingerprint;
        }

        boolean isSet() {
            return caPem != null || fingerprint != null;
        }
    }

    private final DiscoveryEnvironment env;
    private final HttpClient http = HttpClient.newHttpClient();

    DataPlaneDiscovery(DiscoveryEnvironment env) {
        this.env = env;
    }

    DiscoveryEnvironment environment() {
        return env;
    }

    // ---- the control ----------------------------------------------------------

    /** Where the control listens, e.g. {@code http://127.0.0.1:8813}. */
    String controlBaseUrl() {
        return connectUrl(controlHost(), controlPort());
    }

    /**
     * {@code BIOPB_CONTROL_HOST}, then the serving control's published record,
     * then 127.0.0.1.
     */
    String controlHost() {
        String fromEnv = env.get("BIOPB_CONTROL_HOST");
        if (fromEnv != null && !fromEnv.isEmpty()) {
            return fromEnv;
        }
        JsonElement host = runtimeRecord().get("host");
        return host != null && host.isJsonPrimitive() && !host.getAsString().isEmpty()
                ? host.getAsString()
                : CONTROL_DEFAULT_HOST;
    }

    /**
     * {@code BIOPB_CONTROL_PORT}, then the published record, then 8813. A
     * malformed override falls to the default rather than raising, so a stray
     * value can never wedge a client that only wants to probe the control.
     */
    int controlPort() {
        String raw = env.get("BIOPB_CONTROL_PORT");
        if (raw != null && !raw.isEmpty()) {
            try {
                return Integer.parseInt(raw.trim());
            } catch (NumberFormatException error) {
                return CONTROL_DEFAULT_PORT;
            }
        }
        JsonElement port = runtimeRecord().get("port");
        if (port != null && port.isJsonPrimitive() && port.getAsJsonPrimitive().isNumber()) {
            double value = port.getAsDouble();
            if (value == Math.rint(value)) {
                return (int) value;
            }
        }
        return CONTROL_DEFAULT_PORT;
    }

    /**
     * The control's published endpoint ({@code control.json}), or an empty object.
     * A hint to probe, not proof: a crashed control leaves its record behind.
     * Deliberately forgiving -- a missing or malformed file means "no record",
     * never an exception, since this sits under every client's first call.
     */
    private JsonObject runtimeRecord() {
        try {
            Path file = env.stateDir().resolve("control.json");
            JsonElement parsed = JsonParser.parseString(new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
            return parsed.isJsonObject() ? parsed.getAsJsonObject() : new JsonObject();
        } catch (IOException | RuntimeException error) {
            return new JsonObject();
        }
    }

    /**
     * A base URL a client on this machine can connect to. A server bound to a
     * wildcard is dialed over loopback, and an IPv6 literal is bracketed so the
     * port suffix stays unambiguous.
     */
    static String connectUrl(String host, int port) {
        String dialed = host;
        switch (host) {
            case "0.0.0.0":
            case "":
                dialed = "127.0.0.1";
                break;
            case "::":
                dialed = "::1";
                break;
            default:
                break;
        }
        if (dialed.indexOf(':') >= 0 && !dialed.startsWith("[")) {
            dialed = "[" + dialed + "]";
        }
        return "http://" + dialed + ":" + port;
    }

    /**
     * The data-plane URL the control publishes ({@code GET /health},
     * unauthenticated, {@code data_plane.grpc_url}), or null if no control
     * answers. Best-effort: an absent, slow or malformed control is "no answer".
     */
    String controlGrpcUrl(Duration timeout) {
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(controlBaseUrl() + "/health"))
                    .timeout(timeout)
                    .GET()
                    .build();
            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                return null;
            }
            return grpcUrlOf(JsonParser.parseString(response.body()));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return null;
        } catch (IOException | RuntimeException error) {
            return null;
        }
    }

    /**
     * Have the control bring its plane up ({@code POST /api/data_plane/ensure}),
     * idempotent on its side; null when no control answers or it could not.
     *
     * <p>{@code timeout} is both this call's HTTP timeout and the {@code
     * client_timeout} the control keeps its own wait under, so a slow start comes
     * back as a verdict rather than as a timeout that looks like no control at all.
     */
    DataPlane ensureDataPlane(Duration timeout) {
        double seconds = timeout.toMillis() / 1000.0;
        String token = resolveToken(null, true);
        try {
            HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(
                            controlBaseUrl() + "/api/data_plane/ensure?client_timeout=" + seconds))
                    .timeout(timeout)
                    .POST(HttpRequest.BodyPublishers.noBody());
            // The token also clears the control's CSRF gate on a POST. Without
            // one (a tokenless local control) the gate falls back to a loopback Host.
            if (token != null) {
                request.header("X-Biopb-Token", token);
            }
            HttpResponse<String> response = http.send(request.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                LOGGER.info("control ensure_data_plane answered " + response.statusCode());
                return null;
            }
            JsonElement parsed = JsonParser.parseString(response.body());
            String url = grpcUrlOf(parsed);
            if (url == null) {
                LOGGER.warning("control answered ensure without a data-plane url");
                return null;
            }
            return answer(url);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return null;
        } catch (IOException | RuntimeException error) {
            LOGGER.info("control ensure_data_plane failed: " + error);
            return null;
        }
    }

    /** The control refused a request: its HTTP status and the message it gave. */
    static final class ControlRefused extends IOException {
        private static final long serialVersionUID = 1L;
        final int status;

        ControlRefused(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    /**
     * The control's JSON answer to {@code method path} with {@code params} as its
     * query string, carrying the control's token when there is one.
     *
     * @throws ControlRefused the control answered with an error status
     * @throws IOException no control answers
     */
    JsonObject controlRequest(String method, String path, java.util.Map<String, String> params, Duration timeout)
            throws IOException {
        StringBuilder query = new StringBuilder();
        for (java.util.Map.Entry<String, String> param : params.entrySet()) {
            query.append(query.length() == 0 ? '?' : '&')
                    .append(java.net.URLEncoder.encode(param.getKey(), StandardCharsets.UTF_8))
                    .append('=')
                    .append(java.net.URLEncoder.encode(param.getValue(), StandardCharsets.UTF_8));
        }
        String token = resolveToken(null, true);
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(controlBaseUrl() + path + query))
                .timeout(timeout);
        if ("POST".equals(method)) {
            request.POST(HttpRequest.BodyPublishers.noBody());
        } else {
            request.GET();
        }
        // The token also clears the control's CSRF gate on a POST. Without one
        // (a tokenless local control) the gate falls back to a loopback Host.
        if (token != null) {
            request.header("X-Biopb-Token", token);
        }
        HttpResponse<String> response;
        try {
            response = http.send(request.build(), HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted", error);
        }
        JsonObject body = new JsonObject();
        try {
            JsonElement parsed = JsonParser.parseString(response.body());
            if (parsed.isJsonObject()) {
                body = parsed.getAsJsonObject();
            }
        } catch (RuntimeException ignored) {
            // an error page that is not JSON: the status says enough
        }
        if (response.statusCode() / 100 != 2) {
            JsonElement message = body.get("error");
            throw new ControlRefused(response.statusCode(), message != null && message.isJsonPrimitive()
                    ? message.getAsString()
                    : "the control answered HTTP " + response.statusCode());
        }
        return body;
    }

    private static String grpcUrlOf(JsonElement payload) {
        if (!payload.isJsonObject()) {
            return null;
        }
        JsonElement plane = payload.getAsJsonObject().get("data_plane");
        if (plane == null || !plane.isJsonObject()) {
            return null;
        }
        JsonElement url = plane.getAsJsonObject().get("grpc_url");
        return url != null && url.isJsonPrimitive() && !url.getAsString().isEmpty() ? url.getAsString() : null;
    }

    /**
     * The plane the control named. The credential file is read here and only
     * here: it is the control's credential for its own plane, so it goes only to
     * an address the control gave.
     */
    DataPlane answer(String url) {
        return new DataPlane(url, resolveToken(null, true));
    }

    // ---- the token ------------------------------------------------------------

    /**
     * The data-plane token: explicit, then {@code BIOPB_TENSOR_TOKEN}, then the
     * credential file in the state directory -- the last only when {@code
     * allowCredentialFile}, which is for an address the control named. Null when
     * nothing yields one; blank is null, never "" (an empty string would be sent
     * as an empty {@code Bearer} header rather than omitted).
     */
    String resolveToken(String explicit, boolean allowCredentialFile) {
        String given = explicit == null ? "" : explicit.trim();
        if (given.isEmpty()) {
            given = env.trimmed(ENV_TENSOR_TOKEN);
        }
        if (!given.isEmpty()) {
            return given;
        }
        if (!allowCredentialFile) {
            return null;
        }
        try {
            String token = new String(Files.readAllBytes(env.stateDir().resolve(CREDENTIAL_FILE)),
                    StandardCharsets.UTF_8).trim();
            return token.isEmpty() ? null : token;
        } catch (IOException | RuntimeException error) {
            return null;
        }
    }

    // ---- TLS ------------------------------------------------------------------

    /**
     * The TLS anchor to dial {@code url} with, given where its address came from.
     *
     * <p>A plane the control named is this machine's own and is identified by the
     * record it published; nothing in the environment overrides that, so a stale
     * variable cannot point a local client at an old certificate. Any other
     * address routed around the control, so an anchor the environment configures
     * wins, and without one the local record (a loopback {@code grpcs://}) or
     * trust on first use decides.
     */
    TlsAnchor dataPlaneTrust(String url, boolean controlNamed) {
        if (!controlNamed) {
            TlsAnchor configured = configuredTlsAnchor();
            if (configured.isSet()) {
                return configured;
            }
        }
        return new TlsAnchor(null, localDataPlaneFingerprint(url));
    }

    /**
     * {@code BIOPB_TENSOR_TLS_CA} (a PEM file) or {@code BIOPB_TENSOR_TLS_FINGERPRINT},
     * else none; the CA wins when both are set.
     *
     * @throws TlsConfigException the CA file is unusable
     */
    TlsAnchor configuredTlsAnchor() {
        String caPath = env.trimmed(ENV_TENSOR_TLS_CA);
        byte[] ca = caPath.isEmpty() ? null : readPem(expandUser(caPath), "$" + ENV_TENSOR_TLS_CA);
        String fingerprint = env.trimmed(ENV_TENSOR_TLS_FINGERPRINT);
        if (ca != null && !fingerprint.isEmpty()) {
            LOGGER.warning("The environment sets both a CA and a fingerprint; the CA is used and the "
                    + "fingerprint is ignored.");
            fingerprint = "";
        }
        return new TlsAnchor(ca, fingerprint.isEmpty() ? null : fingerprint);
    }

    /**
     * Identity of the certificate a <i>local</i> plane serves, as a SHA-256 digest.
     *
     * <p>A loopback {@code grpcs://} plane is this machine's own, so what it serves
     * is knowable here rather than something to accept on first sight. The digest
     * is checked against the certificate the server presents on every connect. A
     * fingerprint rather than the PEM, deliberately: a PEM resolves trust offline,
     * which also skips the hostname-override probe a loopback dial needs.
     *
     * <p>Two sources, in order: what the plane <b>published</b> for this port
     * ({@code tls-served.json}, keyed by port), then the certificate it would have
     * <b>minted</b> ({@code tls/server-cert.pem}). Null for a plaintext or remote
     * endpoint, whose certificate is not on this disk.
     *
     * @throws LocalTrustException a local plane serves TLS and neither source answers:
     *         falling back to trust-on-first-use there would trade a verified
     *         identity for an unverified one where the strong option was meant to apply
     */
    String localDataPlaneFingerprint(String url) {
        String lower = url.toLowerCase();
        if (!(lower.startsWith("grpcs://") || lower.startsWith("grpc+tls://")) || !isLocalUrl(url)) {
            return null;
        }
        Path state = env.stateDir();
        int port = portOf(url);
        if (port > 0) {
            String published = publishedFingerprint(state.resolve("tls-served.json"), port);
            if (published != null) {
                return published;
            }
        }
        Path certPath = state.resolve("tls").resolve("server-cert.pem");
        byte[] pem;
        try {
            pem = Files.readAllBytes(certPath);
        } catch (IOException error) {
            throw new LocalTrustException("The local data plane at " + url + " serves TLS, but nothing on this "
                    + "machine says which certificate: no record for port " + port + " in "
                    + state.resolve("tls-served.json") + ", and no certificate at " + certPath + " ("
                    + error.getMessage() + "). A local plane is verified against what it serves, not pinned from "
                    + "the wire, so this is not retried as trust-on-first-use. Check the state dir is the one the "
                    + "server writes to (BIOPB_STATE_HOME); a plane publishes its record at startup, so restart "
                    + "one that predates this build, or mint the certificate with `biopb-tensor-server cert init`.",
                    error);
        }
        if (new String(pem, StandardCharsets.US_ASCII).trim().isEmpty()) {
            throw new LocalTrustException("The local data plane's TLS certificate at " + certPath + " is empty.");
        }
        try {
            return TlsTrusts.fingerprint(pem);
        } catch (IllegalArgumentException error) {
            throw new LocalTrustException("The local data plane's TLS certificate at " + certPath
                    + " is not readable as PEM (" + error.getMessage() + ").", error);
        }
    }

    private static String publishedFingerprint(Path file, int port) {
        try {
            JsonElement parsed = JsonParser.parseString(new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
            if (!parsed.isJsonObject()) {
                return null;
            }
            JsonElement entry = parsed.getAsJsonObject().get(String.valueOf(port));
            if (entry == null || !entry.isJsonObject()) {
                return null;
            }
            JsonElement value = entry.getAsJsonObject().get("fingerprint");
            return value != null && value.isJsonPrimitive() && !value.getAsString().isEmpty()
                    ? value.getAsString()
                    : null;
        } catch (IOException | RuntimeException error) {
            return null; // an absent or malformed record is "none published"
        }
    }

    static boolean isLocalUrl(String url) {
        try {
            String host = new URI(url).getHost();
            if (host == null) {
                return true;
            }
            String bare = host.startsWith("[") && host.endsWith("]") ? host.substring(1, host.length() - 1) : host;
            return Arrays.asList("localhost", "127.0.0.1", "::1").contains(bare);
        } catch (URISyntaxException error) {
            return false;
        }
    }

    private static int portOf(String url) {
        try {
            return new URI(url).getPort();
        } catch (URISyntaxException error) {
            return -1;
        }
    }

    private Path expandUser(String value) {
        if (value.equals("~") || value.startsWith("~/") || value.startsWith("~\\")) {
            return env.home().resolve(value.length() > 2 ? value.substring(2) : "");
        }
        return Paths.get(value);
    }

    /**
     * {@code path} as PEM, or a {@link TlsConfigException} naming the fault.
     * Checks only what bytes show: readable, non-empty, PEM, not
     * passphrase-protected.
     */
    private static byte[] readPem(Path path, String label) {
        if (Files.isDirectory(path)) {
            throw new TlsConfigException(label + " is a directory, not a PEM file: " + path);
        }
        if (!Files.isRegularFile(path)) {
            throw new TlsConfigException(label + " not found: " + path);
        }
        byte[] data;
        try {
            data = Files.readAllBytes(path);
        } catch (IOException error) {
            throw new TlsConfigException(label + " could not be read: " + path + " (" + error.getMessage() + ").");
        }
        String text = new String(data, StandardCharsets.ISO_8859_1);
        if (text.trim().isEmpty()) {
            throw new TlsConfigException(label + " is empty: " + path);
        }
        if (!text.contains("-----BEGIN")) {
            throw new TlsConfigException(label + " is not PEM: " + path + " (no '-----BEGIN' block). A DER or "
                    + "PKCS#12 file has to be converted first -- `openssl x509 -inform der` for a certificate.");
        }
        if (text.contains("ENCRYPTED PRIVATE KEY") || text.contains("Proc-Type: 4,ENCRYPTED")) {
            throw new TlsConfigException(label + " is passphrase-protected: " + path);
        }
        return data;
    }
}
