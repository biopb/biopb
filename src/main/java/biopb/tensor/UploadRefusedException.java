package biopb.tensor;

import org.apache.arrow.flight.FlightStatusCode;

/** A chunk write or finish reached an upload that is already terminal. */
public final class UploadRefusedException extends TensorFlightException {
    private final String sourceId;
    private final String state;
    private final String detail;

    UploadRefusedException(String message, String reason, String sourceId, String state, String detail, Throwable cause) {
        super(message, FlightStatusCode.CANCELLED, reason, cause);
        this.sourceId = sourceId;
        this.state = state;
        this.detail = detail;
    }

    public String getSourceId() { return sourceId; }
    public String getState() { return state; }
    public String getDetail() { return detail; }
}
