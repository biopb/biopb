package biopb.tensor;

import java.util.Collections;
import java.util.List;

/**
 * One {@code sources} catalog row, as the SDK's own struct.
 *
 * <p>The row is the only representation of a source that crosses the wire -- the
 * {@code catalog} flight streams them (SQL over DoGet) and {@code resolve}
 * returns the one it just wrote. {@link TensorFlightClient#sourcesFromRows}
 * builds these from it.
 *
 * <p>A plain class, not the generated {@code DataSourceDescriptor}
 * (biopb/biopb#1032). That message never crossed the wire and each SDK built it
 * for itself, yet every catalog column a client wanted cost a {@code .proto}
 * edit and a {@code buf generate} across all bindings. The "one schema, every
 * language" reason for paying that was already gone: TypeScript hand-rolls the
 * same shape, and the HTTP sidecar decodes rows to plain dicts.
 *
 * <p>Immutable, and Java 11 -- hence getters rather than a record. Package-private:
 * this is the client's own view of a row, not API (biopb/biopb#1032).
 */
final class CatalogSource {

    private final String sourceId;
    private final String sourceUrl;
    private final String sourceType;
    private final List<CatalogTensor> tensors;
    private final boolean resolved;
    private final Boolean dataResident;

    public CatalogSource(
            String sourceId,
            String sourceUrl,
            String sourceType,
            List<CatalogTensor> tensors,
            boolean resolved,
            Boolean dataResident) {
        this.sourceId = sourceId == null ? "" : sourceId;
        this.sourceUrl = sourceUrl == null ? "" : sourceUrl;
        this.sourceType = sourceType == null ? "" : sourceType;
        this.tensors = tensors == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(new java.util.ArrayList<>(tensors));
        this.resolved = resolved;
        this.dataResident = dataResident;
    }

    /** Unique identifier for the data source (the flight identifier). */
    public String getSourceId() {
        return sourceId;
    }

    /** File path, directory, or remote URL. */
    public String getSourceUrl() {
        return sourceUrl;
    }

    /** Source type: {@code "ome-zarr"}, {@code "zarr"}, {@code "hdf5"}, ... */
    public String getSourceType() {
        return sourceType;
    }

    /**
     * Every tensor in this source, as a structural catalog entry.
     *
     * <p>Empty for a source that has not been resolved yet -- and also for one
     * that resolved and had nothing readable in it, which is why
     * {@link #isResolved()} exists and this list's emptiness is not the test.
     */
    public List<CatalogTensor> getTensors() {
        return tensors;
    }

    /**
     * Whether the server has hydrated this source enough to know its tensors.
     *
     * <p>Monotonic -- false to true once, never back -- which is what makes it
     * safe to read off a stored row. Not to be confused with
     * {@link #getDataResident()}.
     */
    public boolean isResolved() {
        return resolved;
    }

    /**
     * Advisory, point-in-time: the content is local and cheap to read <i>right
     * now</i>, or {@code null} when the server did not report it.
     *
     * <p>VOLATILE -- a synced-folder source re-dehydrates under storage pressure
     * -- so a read path wanting certainty asks the server, not this.
     */
    public Boolean getDataResident() {
        return dataResident;
    }

    @Override
    public String toString() {
        return "CatalogSource{sourceId=" + sourceId
                + ", sourceType=" + sourceType
                + ", isResolved=" + resolved
                + ", tensors=" + tensors.size() + "}";
    }
}
