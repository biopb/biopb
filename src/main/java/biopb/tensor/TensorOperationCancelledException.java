package biopb.tensor;

/** Raised when a caller stops consuming a long-running tensor server operation. */
public final class TensorOperationCancelledException extends RuntimeException {
    private final String operation;
    private final String sourceId;

    public TensorOperationCancelledException(String operation, String sourceId) {
        super(operation + "('" + sourceId + "') cancelled by caller");
        this.operation = operation;
        this.sourceId = sourceId;
    }

    public String getOperation() { return operation; }
    public String getSourceId() { return sourceId; }
}
