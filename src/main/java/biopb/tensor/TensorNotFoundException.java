package biopb.tensor;

import org.apache.arrow.flight.FlightStatusCode;

/** The requested source, tensor field, or chunk no longer exists. */
public final class TensorNotFoundException extends TensorFlightException {
    TensorNotFoundException(String message, String reason, Throwable cause) {
        super(message, FlightStatusCode.NOT_FOUND, reason, cause);
    }
}
