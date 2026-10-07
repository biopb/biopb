package biopb.tensor;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Where trust-on-first-use remembers the certificate it pinned, keyed by
 * {@code host:port}.
 *
 * <p>Deliberately an interface the caller supplies. Python keeps its pins in a
 * file under the user's state directory; sharing that file would make a pin set
 * by one SDK apply to the other, but it is a cross-language on-disk contract
 * (location and format) that has not been decided, and this SDK does not invent
 * it quietly (biopb/biopb#1070). A caller that wants pins to outlive the process
 * implements this over whatever it keeps state in.
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
