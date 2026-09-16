package biopb.tensor;

import java.util.Collections;
import java.util.List;

/**
 * One entry of a {@code sources} row's {@code tensors} STRUCT[]: a tensor's
 * identity and shape.
 *
 * <p>Enough to enumerate a source's tensors and address one; not enough to plan
 * a read. Describe the tensor (GetFlightInfo) for the transfer grid and the
 * pyramid -- those are absent here rather than present and permanently empty.
 *
 * <p>Immutable. A plain class, not a record: the published artifact targets
 * Java 11.
 *
 * @see CatalogSource
 */
public final class CatalogTensor {

    private final String arrayId;
    private final List<String> dimLabels;
    private final List<Long> shape;
    private final String dtype;

    public CatalogTensor(String arrayId, List<String> dimLabels, List<Long> shape, String dtype) {
        this.arrayId = arrayId == null ? "" : arrayId;
        this.dimLabels = dimLabels == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(new java.util.ArrayList<>(dimLabels));
        this.shape = shape == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(new java.util.ArrayList<>(shape));
        this.dtype = dtype == null ? "" : dtype;
    }

    /** Globally-unique tensor identifier: {@code source_id} or {@code source_id/field}. */
    public String getArrayId() {
        return arrayId;
    }

    /** Semantic axis names, aligned with {@link #getShape()}. */
    public List<String> getDimLabels() {
        return dimLabels;
    }

    /** Full array shape, per dimension. */
    public List<Long> getShape() {
        return shape;
    }

    /** Element dtype, numpy-style (e.g. {@code "uint16"}). */
    public String getDtype() {
        return dtype;
    }

    @Override
    public String toString() {
        return "CatalogTensor{arrayId=" + arrayId + ", shape=" + shape + ", dtype=" + dtype + "}";
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof CatalogTensor)) {
            return false;
        }
        CatalogTensor that = (CatalogTensor) other;
        return arrayId.equals(that.arrayId)
                && dimLabels.equals(that.dimLabels)
                && shape.equals(that.shape)
                && dtype.equals(that.dtype);
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(arrayId, dimLabels, shape, dtype);
    }
}
