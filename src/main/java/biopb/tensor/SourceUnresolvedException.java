package biopb.tensor;

import org.apache.arrow.flight.FlightStatusCode;

/** The source must be resolved explicitly before it can be described or read. */
public final class SourceUnresolvedException extends TensorFlightException {
    SourceUnresolvedException(String message, Throwable cause) {
        super(message, FlightStatusCode.UNAVAILABLE, null, cause);
    }
}
