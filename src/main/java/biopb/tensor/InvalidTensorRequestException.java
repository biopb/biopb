package biopb.tensor;

import org.apache.arrow.flight.FlightStatusCode;

/** The tensor id, slice, scale, or another read parameter is invalid. */
public final class InvalidTensorRequestException extends TensorFlightException {
    InvalidTensorRequestException(String message, String reason, Throwable cause) {
        super(message, FlightStatusCode.INVALID_ARGUMENT, reason, cause);
    }
}
