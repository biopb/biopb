package biopb.tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

/**
 * The control's algorithm-plane calls, for the SDK's own algorithm client.
 *
 * <p>Where the control listens is the discovery contract's ({@code
 * docs/discovery-contract.md}); this asks it to bring a registry entry up. It
 * is the one control call {@code biopb.image.OpsClient.connect} needs when it is
 * given a name instead of a URL, public only because that client lives in
 * another package. The supported ways to reach the control are {@link Connection}
 * and the {@code biopb} CLI.
 */
public final class ControlClient {

    private static final Gson GSON = new Gson();

    private final DataPlaneDiscovery discovery;

    private ControlClient(DataPlaneDiscovery discovery) {
        this.discovery = discovery;
    }

    /** A client for the control this process's environment and home point at. */
    public static ControlClient system() {
        return new ControlClient(new DataPlaneDiscovery(DiscoveryEnvironment.system()));
    }

    /** A client for the control an explicit environment and home point at; for tests and embedding. */
    public static ControlClient of(Map<String, String> environment, Path home) {
        return new ControlClient(new DataPlaneDiscovery(DiscoveryEnvironment.of(environment, home)));
    }

    /**
     * Bring a script entry up, installing it first if its file changed, and
     * answer its row ({@code name, kind, url, state, error, token, ...}); a url
     * entry is probed. Waits under {@code timeout}: a row still {@code installing}
     * or {@code starting} means ask again.
     *
     * @throws NoSuchElementException the name is unknown
     * @throws IllegalArgumentException the control refused the request
     * @throws IllegalStateException no control answered, or it failed
     */
    public Map<String, Object> ensureAlgorithm(String name, Duration timeout) {
        Map<String, String> params = new LinkedHashMap<>();
        params.put("name", name);
        params.put("client_timeout", String.valueOf(timeout.toMillis() / 1000.0));
        JsonObject answer;
        try {
            answer = discovery.controlRequest("POST", "/api/algorithms/ensure", params,
                    discovery.resolveToken(null, true), timeout);
        } catch (DataPlaneDiscovery.ControlRefused refused) {
            if (refused.status == 404) {
                throw new NoSuchElementException(refused.getMessage());
            }
            if (refused.status == 400) {
                throw new IllegalArgumentException(refused.getMessage());
            }
            throw new IllegalStateException(refused.getMessage(), refused);
        } catch (IOException error) {
            throw new IllegalStateException("no control answered: " + error.getMessage(), error);
        }
        if (!answer.has("server") || !answer.get("server").isJsonObject()) {
            throw new IllegalStateException("the control answered ensure without a server row");
        }
        return GSON.fromJson(answer.get("server"), new TypeToken<Map<String, Object>>() { }.getType());
    }
}
