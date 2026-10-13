package biopb.tensor;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Where trust-on-first-use remembers the certificate it pinned, keyed by
 * {@code host:port}.
 *
 * <p>{@link FileTlsPinStore} is the store {@link Connection} uses: the file the
 * Python SDK keeps under the state directory, so a pin made by either SDK holds
 * for both. {@link #inMemory()} is for a caller that wants pins to last no longer
 * than the process.
 */
public interface TlsPinStore {

    /** The PEM pinned for {@code hostPort}, or null if nothing is. */
    String get(String hostPort);

    /** Record {@code pem} as the pin for {@code hostPort}. */
    void put(String hostPort, String pem);

    /** A store that lives and dies with the process: pins last until the JVM exits. */
    static TlsPinStore inMemory() {
        ConcurrentHashMap<String, String> pins = new ConcurrentHashMap<>();
        return new TlsPinStore() {
            @Override
            public String get(String hostPort) {
                return pins.get(hostPort);
            }

            @Override
            public void put(String hostPort, String pem) {
                pins.put(hostPort, pem);
            }
        };
    }
}
