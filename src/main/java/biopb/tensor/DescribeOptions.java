package biopb.tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Which optional parts a {@link TensorFlightClient#getDescriptor(String,
 * DescribeOptions)} response should fill.
 *
 * <p>The named-boolean face of the wire's {@code TensorReadOption.fields} mask,
 * as Python's {@code get_descriptor(with_*=...)} is: a mistyped path would be a
 * server-side refusal, where a mistyped method is a compile error. Every part
 * is opt-in except the pyramid, which the primary describe consumer reads.
 * Immutable; each {@code with*} returns a copy.
 */
public final class DescribeOptions {
    private final boolean metadata;
    private final boolean pyramid;
    private final boolean readPlan;
    private final boolean residency;
    private final boolean uploadStatus;

    private DescribeOptions(
            boolean metadata, boolean pyramid, boolean readPlan, boolean residency, boolean uploadStatus) {
        this.metadata = metadata;
        this.pyramid = pyramid;
        this.readPlan = readPlan;
        this.residency = residency;
        this.uploadStatus = uploadStatus;
    }

    /** The cheap describe: base descriptor plus the advertised pyramid. */
    public static DescribeOptions defaults() {
        return new DescribeOptions(false, true, false, false, false);
    }

    /** Fill {@code metadata_json}, the full OME tree (megabytes on a per-plane-annotated file). */
    public DescribeOptions withMetadata(boolean wanted) {
        return new DescribeOptions(wanted, pyramid, readPlan, residency, uploadStatus);
    }

    /**
     * Advertise the resolution pyramid. Sizing a native one opens each level's
     * store, so a cloud store pays network I/O per level.
     */
    public DescribeOptions withPyramid(boolean wanted) {
        return new DescribeOptions(metadata, wanted, readPlan, residency, uploadStatus);
    }

    /** Enumerate the per-request chunk endpoints: O(chunks), which a describe has no use for. */
    public DescribeOptions withReadPlan(boolean wanted) {
        return new DescribeOptions(metadata, pyramid, wanted, residency, uploadStatus);
    }

    /**
     * Ask whether the source's bytes are local right now, answered on {@code
     * is_resident}. A bounded stat walk of the source: ask it for a source you
     * are about to read, never in a loop over a listing (biopb/biopb#1048).
     */
    public DescribeOptions withResidency(boolean wanted) {
        return new DescribeOptions(metadata, pyramid, readPlan, wanted, uploadStatus);
    }

    /** Fill {@code upload_status} for a source backed by an upload. Cheap: an in-memory read. */
    public DescribeOptions withUploadStatus(boolean wanted) {
        return new DescribeOptions(metadata, pyramid, readPlan, residency, wanted);
    }

    /** The {@code fields} paths this selects, in the order the server documents them. */
    List<String> paths() {
        List<String> paths = new ArrayList<>();
        if (readPlan) {
            paths.add("endpoints");
        }
        if (metadata) {
            paths.add("metadata_json");
        }
        if (pyramid) {
            paths.add("pyramid");
        }
        if (uploadStatus) {
            paths.add("upload_status");
        }
        if (residency) {
            paths.add("is_resident");
        }
        return paths;
    }
}
