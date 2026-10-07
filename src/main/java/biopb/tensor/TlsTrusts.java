package biopb.tensor;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.security.MessageDigest;
import java.security.cert.Certificate;
import java.security.cert.CertificateExpiredException;
import java.security.cert.CertPathValidatorException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.logging.Logger;

import javax.net.ssl.SNIHostName;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLHandshakeException;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManager;
import javax.net.ssl.TrustManagerFactory;
import javax.net.ssl.X509TrustManager;

import org.apache.arrow.flight.Location;

/**
 * Resolve the {@link TlsTrust} for a {@code grpc+tls://} location.
 *
 * <p>The Java twin of Python's {@code biopb.tensor._tls}. A server on a private
 * LAN typically presents a self-signed or private-CA certificate that the JDK's
 * default trust store does not know, so the caller says how to trust it, in one
 * of three modes:
 *
 * <ul>
 *   <li><b>{@code caPem}</b> -- trust exactly these PEM bytes (a private CA, or
 *       the server's own leaf). Never touches the network and never consults the
 *       pin store.
 *   <li><b>{@code expectedFingerprint}</b> -- fetch the presented leaf and
 *       require its SHA-256 to equal this, on <i>every</i> resolve: a wrong
 *       certificate is refused on the first connect, which trust-on-first-use
 *       cannot do. Also bypasses the pin store.
 *   <li><b>neither</b> -- trust on first use, the SSH host-key model: the first
 *       certificate seen for a {@code host:port} is pinned in the {@link
 *       TlsPinStore} and used as the anchor; a different one later is a {@link
 *       TlsPinMismatchException}. Protects against an attacker who arrives
 *       <i>after</i> pinning, not one already in the path at first connect.
 * </ul>
 *
 * <p>Trust and hostname verification are separate checks. An anchor supplies the
 * first; the JDK still matches the <b>dialed</b> name against the certificate's
 * SANs, so a certificate that pins fine still fails every handshake if the
 * server never listed the name clients dial. When the anchor <i>is</i> the
 * certificate the server presents, that second check is redundant -- the
 * presented certificate must be the pinned one, so there is no "different but
 * validly-issued certificate" for a name check to exclude -- and a name the
 * certificate does list is substituted as the {@code overrideHostname}. A CA
 * anchor keeps the name check, which is load-bearing there. The substitution
 * never disables verification: the chain is still checked against the anchor.
 *
 * <p>One thing a plain {@code FlightClient} cannot give you: an expired anchor
 * fails every handshake with no reason the caller can read. {@link
 * TlsCertExpiredException} is raised here instead, where the answer is known.
 */
public final class TlsTrusts {

    private static final Logger LOGGER = Logger.getLogger(TlsTrusts.class.getName());

    private static final String TLS_SCHEME = "grpc+tls";
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    /** An anchor this close to its notAfter is warned about, once per resolve. */
    private static final long EXPIRY_WARN_MS = 30L * 86_400_000L;

    private static final java.util.concurrent.ConcurrentHashMap<String, TlsTrust> ANCHORED =
            new java.util.concurrent.ConcurrentHashMap<>();

    private TlsTrusts() {}

    /**
     * Resolve the trust for {@code location}.
     *
     * @param location the server; a non-TLS location needs no trust and gets
     *        {@link TlsTrust#NONE} with no network
     * @param caPem trust exactly these PEM bytes; wins over the other modes
     * @param expectedFingerprint SHA-256 of the server's certificate DER, hex
     *        (colon-grouped or bare, any case)
     * @param pins where trust-on-first-use remembers pins; needed only when
     *        neither {@code caPem} nor {@code expectedFingerprint} is given
     * @throws TlsPinMismatchException the server presents a certificate that
     *         contradicts the pin or the configured fingerprint
     * @throws TlsCertExpiredException the anchor has expired
     * @throws IllegalArgumentException trust on first use was asked for with no
     *         {@code pins}
     * @throws IOException the server could not be reached to read its certificate
     */
    public static TlsTrust resolve(
            Location location, byte[] caPem, String expectedFingerprint, TlsPinStore pins)
            throws IOException {
        HostPort target = hostPort(location);
        if (target == null) {
            return TlsTrust.NONE;
        }
        String fingerprint = expectedFingerprint == null || expectedFingerprint.trim().isEmpty()
                ? null
                : normalizeFingerprint(expectedFingerprint);
        String keyId = keyId(target.key(), caPem, fingerprint);

        byte[] anchor;
        String override = null;
        Mode mode;
        if (caPem != null && caPem.length > 0) {
            // Offline: staying off the network is the point of configuring an
            // anchor, so this mode also skips the name probe and never carries an
            // override -- the conservative answer, since a configured anchor is
            // usually a real CA whose SAN check is load-bearing.
            mode = Mode.CA;
            anchor = caPem;
        } else {
            if (fingerprint != null) {
                mode = Mode.FINGERPRINT;
                anchor = resolveAgainstFingerprint(target, fingerprint);
            } else {
                if (pins == null) {
                    throw new IllegalArgumentException("trust on first use needs a TlsPinStore, or give "
                            + "caPem or expectedFingerprint; see TlsPinStore.inMemory()");
                }
                mode = Mode.TOFU;
                anchor = resolveTofu(target, pins);
            }
            override = resolveHostnameOverride(target, anchor, mode);
        }
        checkAnchorExpiry(target, anchor, mode);
        return new TlsTrust(anchor, override, keyId, false);
    }

    /**
     * A trust that carries {@code anchorPem} to a consumer to resolve for
     * itself, as a {@code SerializedTensor.tls_anchor} does. No network here, so
     * the sender never has to reach the address the consumer will dial. Null or
     * empty (the sender trusted the system store) is the system store again.
     */
    public static TlsTrust anchored(byte[] anchorPem) {
        if (anchorPem == null || anchorPem.length == 0) {
            return TlsTrust.NONE;
        }
        return new TlsTrust(anchorPem, null, null, true);
    }

    /**
     * {@code trust} as something a client can use for {@code location}.
     *
     * <p>An {@link #anchored} trust is the sender's decision about <i>which
     * certificate</i>; the name check is this process's, because it depends on
     * the name dialed here. When the anchor is the leaf the server presents, a
     * name the certificate does not list is rescued with an override; a CA
     * anchor keeps the SAN check. A probe that cannot run leaves no override, so
     * a working connection never breaks here.
     */
    public static TlsTrust concrete(Location location, TlsTrust trust) {
        if (trust == null || !trust.reresolve()) {
            return trust;
        }
        HostPort target = hostPort(location);
        if (target == null) {
            return TlsTrust.NONE;
        }
        byte[] anchor = trust.rootCerts();
        String key = keyId(target.key(), anchor, null) + "|anchored";
        // Deserializing a tensor lands here once per tensor, ahead of the session
        // cache, and the answer is a pair of TLS handshakes: ask once per endpoint
        // and anchor. Not invalidated, like Python's memo -- a rotated certificate
        // is a restart.
        return ANCHORED.computeIfAbsent(key, ignored -> new TlsTrust(
                anchor, resolveHostnameOverride(target, anchor, Mode.ANCHORED), key, false));
    }

    /** Whether {@code location} is a {@code grpc+tls://} address, so trust applies. */
    public static boolean isTlsLocation(Location location) {
        return hostPort(location) != null;
    }

    /** SHA-256 of a certificate's DER body, lowercase hex: the id a fingerprint names. */
    public static String fingerprint(byte[] pem) {
        try {
            return fingerprintOf(parseCertificates(pem).get(0));
        } catch (GeneralSecurityException | IndexOutOfBoundsException error) {
            throw new IllegalArgumentException("not a PEM certificate", error);
        }
    }

    // ---- modes ------------------------------------------------------------

    private enum Mode { CA, FINGERPRINT, TOFU, ANCHORED }

    private static byte[] resolveAgainstFingerprint(HostPort target, String fingerprint) throws IOException {
        // The configured digest is the whole trust decision, so this does not
        // touch the pin store: writing one would create a second anchor that can
        // later disagree with the config.
        byte[] presented = fetchServerCert(target);
        String actual = fingerprint(presented);
        if (!actual.equals(fingerprint)) {
            throw new TlsPinMismatchException("TLS certificate for " + target.key()
                    + " does not match the configured fingerprint (expected " + fingerprint.substring(0, 16)
                    + ", server presented " + actual.substring(0, 16)
                    + "). If the server's certificate was legitimately rotated, update the configured "
                    + "fingerprint to the new value; otherwise this may be a man-in-the-middle and you "
                    + "should not connect.");
        }
        return presented;
    }

    private static byte[] resolveTofu(HostPort target, TlsPinStore pins) throws IOException {
        byte[] presented = fetchServerCert(target);
        String key = target.key();
        String pinned = pins.get(key);
        if (pinned == null) {
            pins.put(key, new String(presented, StandardCharsets.US_ASCII));
            LOGGER.info("TOFU: pinned certificate for " + key + " (" + fingerprint(presented).substring(0, 16)
                    + "). Remove the pin to re-pin after a legitimate certificate change.");
            return presented;
        }
        byte[] pinnedBytes = pinned.getBytes(StandardCharsets.US_ASCII);
        if (!fingerprint(presented).equals(fingerprint(pinnedBytes))) {
            throw new TlsPinMismatchException("TLS certificate for " + key
                    + " does not match the pinned one (pinned " + fingerprint(pinnedBytes).substring(0, 16)
                    + ", server now " + fingerprint(presented).substring(0, 16)
                    + "). If the server's certificate was legitimately rotated, remove the '" + key
                    + "' pin and reconnect; otherwise this may be a man-in-the-middle and you should "
                    + "not connect.");
        }
        return pinnedBytes;
    }

    // ---- the hostname probe -----------------------------------------------

    /**
     * A name to verify against instead of the dialed one, or null.
     *
     * <p>Handshakes with the anchor and the name check on; success says the name
     * is covered and there is nothing to do. On failure, handshakes again with the
     * chain checked and the name not: if that one succeeds the name was the only
     * thing wrong, and only then -- and only when the anchor is the presented
     * leaf itself -- is a name the certificate does list substituted.
     *
     * <p>Diagnostic-only, with one exception: any failure other than a definite
     * name mismatch leaves this silent and overrideless, so it can never turn a
     * working connection into a broken one. The exception is an expired
     * certificate, which is raised -- that connection is already broken, and this
     * is the only place the reason is legible.
     */
    private static String resolveHostnameOverride(HostPort target, byte[] anchor, Mode mode) {
        try {
            probePeer(target, anchor, true);
            return null; // the dialed name is covered
        } catch (SSLHandshakeException error) {
            // Fall through: a name mismatch, an expiry or an unrelated chain failure.
        } catch (IOException | GeneralSecurityException error) {
            return null;
        }

        X509Certificate presented;
        try {
            presented = probePeer(target, anchor, false);
        } catch (SSLHandshakeException error) {
            if (isExpired(error)) {
                throw new TlsCertExpiredException(expiredMessage(target, mode), error);
            }
            return null;
        } catch (IOException | GeneralSecurityException error) {
            return null;
        }

        // The chain verified without the name check, so the name was the only
        // failure. Is substituting one safe? Only when the anchor is that exact
        // certificate: then no other certificate can satisfy the chain.
        boolean anchorIsLeaf = false;
        try {
            List<X509Certificate> anchors = parseCertificates(anchor);
            anchorIsLeaf = anchors.size() == 1 && java.util.Arrays.equals(anchors.get(0).getEncoded(), presented.getEncoded());
        } catch (GeneralSecurityException ignored) {
            // an unparseable anchor is simply "not a single leaf"
        }
        String override = anchorIsLeaf ? pickOverride(presented) : null;
        if (override != null) {
            LOGGER.info("TLS certificate for " + target.key() + " does not list '" + target.host
                    + "' among its subject-alternative names; verifying against '" + override
                    + "' instead, which it does list. Safe here because the trust anchor is that exact "
                    + "certificate, so no other certificate can satisfy the chain.");
            return override;
        }
        warnNoUsableName(target, anchorIsLeaf, mode);
        return null;
    }

    /** A DNS name over an IP (gRPC matches an IP SAN only when the target looks like one); wildcards skipped. */
    private static String pickOverride(X509Certificate cert) {
        Collection<List<?>> sans;
        try {
            sans = cert.getSubjectAlternativeNames();
        } catch (java.security.cert.CertificateParsingException error) {
            return null;
        }
        if (sans == null) {
            return null;
        }
        String ip = null;
        for (List<?> entry : sans) {
            int type = (Integer) entry.get(0);
            String value = String.valueOf(entry.get(1));
            if (type == 2 && value.indexOf('*') < 0) {
                return value;
            }
            if (type == 7 && ip == null) {
                ip = value;
            }
        }
        return ip;
    }

    private static void warnNoUsableName(HostPort target, boolean anchorIsLeaf, Mode mode) {
        if (!anchorIsLeaf) {
            LOGGER.warning("TLS certificate for " + target.key() + " does not list '" + target.host
                    + "' among its subject-alternative names, so the connection will fail hostname "
                    + "verification. The configured trust anchor for this endpoint is not that "
                    + "certificate itself, so the name check is load-bearing here and is not "
                    + "substituted away: reissue the server's certificate with '" + target.host
                    + "' in its SANs, or dial a name it does list.");
            return;
        }
        LOGGER.warning("TLS certificate for " + target.key() + " does not list '" + target.host
                + "' among its subject-alternative names and carries no name that could be verified "
                + "instead, so the connection will fail hostname verification. Re-mint the server's "
                + "certificate with that name and " + remediation(target, mode)
                + ", or dial a name the certificate does list.");
    }

    // ---- expiry -----------------------------------------------------------

    private static void checkAnchorExpiry(HostPort target, byte[] anchor, Mode mode) {
        List<X509Certificate> certs;
        try {
            certs = parseCertificates(anchor);
        } catch (GeneralSecurityException error) {
            return; // not ours to judge; the handshake will say
        }
        // A configured CA is checked only when it is a single certificate: a bundle
        // holds roots the server may never chain to, so its earliest date says
        // nothing about this connection.
        if (certs.isEmpty() || (mode == Mode.CA && certs.size() != 1)) {
            return;
        }
        Date earliest = null;
        for (X509Certificate cert : certs) {
            if (earliest == null || cert.getNotAfter().before(earliest)) {
                earliest = cert.getNotAfter();
            }
        }
        long remaining = earliest.getTime() - System.currentTimeMillis();
        if (remaining <= 0) {
            throw new TlsCertExpiredException(expiredMessage(target, mode), null);
        }
        if (remaining < EXPIRY_WARN_MS) {
            LOGGER.warning("The TLS certificate for " + target.key() + " expires in " + (remaining / 86_400_000L)
                    + " days (" + earliest.toInstant().toString().substring(0, 10)
                    + "). It will need to be re-minted on the server, " + remediation(target, mode) + ".");
        }
    }

    private static boolean isExpired(Throwable error) {
        for (Throwable t = error; t != null; t = t.getCause()) {
            if (t instanceof CertificateExpiredException) {
                return true;
            }
            if (t instanceof CertPathValidatorException
                    && ((CertPathValidatorException) t).getReason() == CertPathValidatorException.BasicReason.EXPIRED) {
                return true;
            }
            if (t.getCause() == t) {
                break;
            }
        }
        return false;
    }

    private static String remediation(HostPort target, Mode mode) {
        switch (mode) {
            case TOFU:
                return "then remove the '" + target.key() + "' pin";
            case FINGERPRINT:
                return "then update the configured TLS fingerprint to the new certificate";
            default:
                return "then replace the configured CA certificate";
        }
    }

    private static String expiredMessage(HostPort target, Mode mode) {
        return "The TLS certificate for " + target.key() + " has expired. Re-mint the server's certificate "
                + "and " + remediation(target, mode) + ".";
    }

    // ---- handshakes -------------------------------------------------------

    /**
     * Fetch the leaf certificate a server presents, as PEM.
     *
     * <p>An intentionally <i>unverified</i> handshake: there is no anchor yet
     * (that is what first-use trust is bootstrapping). It reads the presented
     * certificate and exchanges no data.
     */
    private static byte[] fetchServerCert(HostPort target) throws IOException {
        TrustManager acceptAnything = new X509TrustManager() {
            @Override
            public void checkClientTrusted(X509Certificate[] chain, String authType) {}

            @Override
            public void checkServerTrusted(X509Certificate[] chain, String authType) {}

            @Override
            public X509Certificate[] getAcceptedIssuers() {
                return new X509Certificate[0];
            }
        };
        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, new TrustManager[] { acceptAnything }, null);
            try (SSLSocket socket = connect(context, target, false)) {
                Certificate[] chain = socket.getSession().getPeerCertificates();
                if (chain.length == 0) {
                    throw new IOException(target.key() + " presented no certificate");
                }
                return toPem((X509Certificate) chain[0]);
            }
        } catch (GeneralSecurityException error) {
            throw new IOException("could not read the certificate " + target.key() + " presents", error);
        }
    }

    /**
     * Handshake with {@code target} trusting only {@code anchor}; return the leaf.
     * The chain is always verified; {@code checkName} selects whether the dialed
     * name is also matched.
     */
    private static X509Certificate probePeer(HostPort target, byte[] anchor, boolean checkName)
            throws IOException, GeneralSecurityException {
        KeyStore store = KeyStore.getInstance(KeyStore.getDefaultType());
        store.load(null, null);
        List<X509Certificate> certs = parseCertificates(anchor);
        for (int i = 0; i < certs.size(); i++) {
            store.setCertificateEntry("anchor-" + i, certs.get(i));
        }
        TrustManagerFactory factory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
        factory.init(store);
        SSLContext context = SSLContext.getInstance("TLS");
        context.init(null, factory.getTrustManagers(), null);
        try (SSLSocket socket = connect(context, target, checkName)) {
            return (X509Certificate) socket.getSession().getPeerCertificates()[0];
        }
    }

    private static SSLSocket connect(SSLContext context, HostPort target, boolean checkName) throws IOException {
        SSLSocket socket = (SSLSocket) context.getSocketFactory().createSocket();
        try {
            socket.connect(new InetSocketAddress(target.host, target.port), CONNECT_TIMEOUT_MS);
            socket.setSoTimeout(CONNECT_TIMEOUT_MS);
            SSLParameters parameters = socket.getSSLParameters();
            parameters.setEndpointIdentificationAlgorithm(checkName ? "HTTPS" : null);
            if (!looksLikeIp(target.host)) {
                parameters.setServerNames(Collections.singletonList(new SNIHostName(target.host)));
            }
            socket.setSSLParameters(parameters);
            socket.startHandshake();
            return socket;
        } catch (IOException | RuntimeException error) {
            try {
                socket.close();
            } catch (IOException ignored) {
                // the handshake's failure is the one to report
            }
            throw error;
        }
    }

    // ---- helpers ----------------------------------------------------------

    private static final class HostPort {
        final String host;
        final int port;

        HostPort(String host, int port) {
            this.host = host;
            this.port = port;
        }

        String key() {
            return host + ":" + port;
        }
    }

    /** Null for any non-TLS scheme: no trust applies. */
    private static HostPort hostPort(Location location) {
        java.net.URI uri = location.getUri();
        if (!TLS_SCHEME.equals(uri.getScheme()) || uri.getHost() == null || uri.getPort() <= 0) {
            return null;
        }
        return new HostPort(uri.getHost(), uri.getPort());
    }

    private static boolean looksLikeIp(String host) {
        return host.indexOf(':') >= 0 || host.matches("\\d{1,3}(\\.\\d{1,3}){3}");
    }

    /**
     * The memo/pool discriminator. The CA is digested over its <b>raw bytes</b>,
     * not through {@link #fingerprint}: a private-CA bundle holds several
     * certificates, and nothing here is a trust decision -- it only has to
     * separate distinct anchors. The full digest, not a prefix, because a
     * collision would hand one upstream's connection to another.
     */
    private static String keyId(String endpoint, byte[] caPem, String fingerprint) {
        String anchor = caPem == null || caPem.length == 0 ? "-" : sha256Hex(caPem);
        return endpoint + "|" + anchor + "|" + (fingerprint == null ? "-" : fingerprint);
    }

    /** Both spellings an operator pastes (colon-grouped and bare), any case. */
    private static String normalizeFingerprint(String value) {
        return value.replace(":", "").replace(" ", "").trim().toLowerCase();
    }

    private static List<X509Certificate> parseCertificates(byte[] pem) throws GeneralSecurityException {
        CertificateFactory factory = CertificateFactory.getInstance("X.509");
        List<X509Certificate> out = new ArrayList<>();
        for (Certificate cert : factory.generateCertificates(new ByteArrayInputStream(pem))) {
            out.add((X509Certificate) cert);
        }
        return out;
    }

    private static String fingerprintOf(X509Certificate cert) throws GeneralSecurityException {
        return sha256Hex(cert.getEncoded());
    }

    private static String sha256Hex(byte[] bytes) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                hex.append(String.format("%02x", b));
            }
            return hex.toString();
        } catch (java.security.NoSuchAlgorithmException error) {
            throw new IllegalStateException(error);
        }
    }

    private static byte[] toPem(X509Certificate cert) throws GeneralSecurityException {
        String body = Base64.getMimeEncoder(64, new byte[] { '\n' }).encodeToString(cert.getEncoded());
        return ("-----BEGIN CERTIFICATE-----\n" + body + "\n-----END CERTIFICATE-----\n")
                .getBytes(StandardCharsets.US_ASCII);
    }
}
