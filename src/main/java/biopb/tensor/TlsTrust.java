package biopb.tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Everything a {@code FlightClient} needs to trust one TLS endpoint -- plain
 * data, resolved once per endpoint by {@link TlsTrusts}.
 *
 * <p>The Java twin of Python's {@code TlsTrust}: a PEM trust anchor, an optional
 * hostname override, and a key id. It maps onto Arrow's {@code
 * FlightClient.Builder} as {@code trustedCertificates(rootCerts)} and {@code
 * overrideHostname(overrideHostname)}; with neither, the JDK's default trust
 * store and the dialed name decide, as they do for any {@code grpc+tls://}
 * location.
 */
public final class TlsTrust {

    /** A plaintext location, or the system trust store: no anchor, no override, no key. */
    public static final TlsTrust NONE = new TlsTrust(null, null, null, false);

    private final byte[] rootCerts;
    private final String overrideHostname;
    private final String keyId;
    private final boolean reresolve;

    TlsTrust(byte[] rootCerts, String overrideHostname, String keyId, boolean reresolve) {
        this.rootCerts = rootCerts == null ? null : rootCerts.clone();
        this.overrideHostname = overrideHostname;
        this.keyId = keyId;
        this.reresolve = reresolve;
    }

    /** The PEM trust anchor, or null to use the system trust store. */
    public byte[] rootCerts() {
        return rootCerts == null ? null : rootCerts.clone();
    }

    /**
     * The name to match against the certificate's SANs instead of the dialed one,
     * or null to verify the dialed name as usual. Set only when the anchor is the
     * certificate the server presents; see {@link TlsTrusts}.
     */
    public String overrideHostname() {
        return overrideHostname;
    }

    /**
     * Which trust decision this is: {@code endpoint | anchor digest | fingerprint}.
     * The connection cache keys on it, so two upstreams that name the same
     * {@code host:port} under different anchors are never handed each other's
     * connection (biopb/biopb#604 item 4). Null for {@link #NONE}.
     */
    public String keyId() {
        return keyId;
    }

    /**
     * An anchor another process verified with, handed over to be applied to
     * whatever name this process dials. {@link TlsTrusts#concrete} turns it into
     * a usable trust.
     */
    boolean reresolve() {
        return reresolve;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof TlsTrust)) {
            return false;
        }
        TlsTrust that = (TlsTrust) other;
        return Arrays.equals(rootCerts, that.rootCerts)
                && Objects.equals(overrideHostname, that.overrideHostname)
                && Objects.equals(keyId, that.keyId)
                && reresolve == that.reresolve;
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(rootCerts), overrideHostname, keyId, reresolve);
    }
}
