package biopb.tensor;

import org.apache.arrow.flight.FlightStatusCode;

/** Base class for a tensor-server error that carried a validated typed payload. */
public class TensorFlightException extends RuntimeException {
    private final FlightStatusCode statusCode;
    private final String reason;

    TensorFlightException(
            String message, FlightStatusCode statusCode, String reason, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
        this.reason = reason;
    }

    public FlightStatusCode getStatusCode() { return statusCode; }
    public String getReason() { return reason; }
}
