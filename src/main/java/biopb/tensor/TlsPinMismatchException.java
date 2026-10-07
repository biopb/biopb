package biopb.tensor;

/**
 * The server's certificate is not the one this client expected.
 *
 * <p>Raised both for a trust-on-first-use pin that no longer matches and for a
 * configured {@code expectedFingerprint} the presented certificate fails. A
 * legitimate cause is certificate rotation; a malicious one is a
 * man-in-the-middle. Either way the client refuses to connect until the
 * operator confirms, and the message names what to update.
 */
public final class TlsPinMismatchException extends RuntimeException {
    TlsPinMismatchException(String message) {
        super(message);
    }
}
