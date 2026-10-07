package biopb.tensor;

/**
 * The local data plane's TLS certificate could not be used as a trust anchor.
 *
 * <p>A type of its own because layers above classify connect failures by message
 * text, and the likeliest cause here -- an unreadable certificate file --
 * stringifies as "Permission denied", which reads as an authentication failure
 * and would send the reader after a token that has nothing to do with it.
 * Matching the type is exact. The Java twin of Python's {@code LocalTrustError}.
 */
public class LocalTrustException extends RuntimeException {
    LocalTrustException(String message) {
        super(message);
    }

    LocalTrustException(String message, Throwable cause) {
        super(message, cause);
    }
}
