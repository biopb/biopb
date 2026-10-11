package biopb.image;

/**
 * An op call went silent: no event arrived within the client's inactivity
 * timeout. The call is cancelled on the server. The Java twin of Python's
 * {@code TimeoutError} from {@code OpsClient.events}.
 */
public final class OpTimeoutException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    OpTimeoutException(String message) {
        super(message);
    }
}
