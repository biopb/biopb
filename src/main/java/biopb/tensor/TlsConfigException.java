package biopb.tensor;

/**
 * A trust anchor set in the environment ({@code BIOPB_TENSOR_TLS_CA}) cannot be
 * used. A {@link LocalTrustException}, so every layer that reports those by type
 * reports this one the same way.
 */
public final class TlsConfigException extends LocalTrustException {
    TlsConfigException(String message) {
        super(message);
    }
}
