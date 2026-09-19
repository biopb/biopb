package biopb.tensor;

import org.apache.arrow.flight.FlightStatusCode;

/** A held read plan names changed content; request a new FlightInfo plan. */
public final class StaleReadPlanException extends TensorFlightException {
    StaleReadPlanException(String message, String reason, Throwable cause) {
        super(message, FlightStatusCode.NOT_FOUND, reason, cause);
    }
}
