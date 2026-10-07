package biopb.tensor;

/**
 * The server's certificate is past its {@code notAfter}, so no handshake can
 * succeed.
 *
 * <p>A separate failure from {@link TlsPinMismatchException}: the certificate is
 * the expected one, it has simply run out. Raised because the transport would
 * otherwise report it as an unexplained connection failure.
 */
public final class TlsCertExpiredException extends RuntimeException {
    TlsCertExpiredException(String message, Throwable cause) {
        super(message, cause);
    }
}
