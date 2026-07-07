package biopb.tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import net.imglib2.Interval;
import net.imglib2.img.cell.CellGrid;

import static biopb.tensor.TensorChunkCodec.cellCount;

/**
 * Maps an imglib2 cell (by its interval min) to the endpoint serving the
 * matching chunk, for the aligned-grid lazy fast path shared by both Flight
 * tensor clients (biopb/biopb#277 item D).
 *
 * <p>The clients previously carried three near-identical index classes plus
 * three copies of the grid-alignment validation ({@code SerializableTensorImg}'s
 * {@code EndpointIndex}, and {@code TensorFlightClient}'s {@code EndpointIndex}
 * and {@code SerializedEndpointIndex}). They differed only in what each cell maps
 * to -- a raw {@code FlightEndpoint} vs. a materialized {@code (ticket, bounds)}
 * record -- and in how a {@link ChunkBounds} is read off each input item.
 * {@link #build} takes both as functions, making this the single home for the
 * grid logic.
 *
 * @param <V> the per-cell value the caller looks up (e.g. a {@code FlightEndpoint}
 *            or a serialized endpoint record)
 */
final class ChunkGridIndex<V> {

    private final CellGrid grid;
    private final int[] nominalCellDimensions;
    private final Map<Long, V> entries;

    private ChunkGridIndex(CellGrid grid, int[] nominalCellDimensions, Map<Long, V> entries) {
        this.grid = grid;
        this.nominalCellDimensions = nominalCellDimensions.clone();
        this.entries = entries;
    }

    /** Flat cell index for {@code interval}'s (grid-aligned) min position. */
    long indexFor(Interval interval) {
        long[] gridPosition = new long[interval.numDimensions()];
        for (int axis = 0; axis < interval.numDimensions(); axis++) {
            long min = interval.min(axis);
            int nominalCellDimension = nominalCellDimensions[axis];
            if (min % nominalCellDimension != 0) {
                throw new IllegalStateException("Cell minimum is not aligned to the logical chunk grid");
            }
            gridPosition[axis] = min / nominalCellDimension;
        }
        return grid.getCellGridIndexFlat(gridPosition);
    }

    /** The value mapped to {@code cellIndex}, or {@code null} if none. */
    V get(long cellIndex) {
        return entries.get(cellIndex);
    }

    /**
     * Build an index over {@code items} if -- and only if -- they tile the
     * {@code dims}/{@code cellDimensions} grid exactly: every chunk aligned to
     * and sized as its nominal cell, no duplicates, and the full cell count
     * present. Returns {@code null} when the layout is not a clean, complete
     * grid, signalling the caller to fall back to whole-array materialization.
     *
     * @param boundsFn extracts the {@link ChunkBounds} of an item
     * @param valueFn  maps an item to the value {@link #get} should return
     */
    static <I, V> ChunkGridIndex<V> build(
            List<I> items,
            long[] dims,
            int[] cellDimensions,
            Function<I, ChunkBounds> boundsFn,
            Function<I, V> valueFn) {

        if (dims.length == 0 || items.isEmpty()) {
            return null;
        }

        CellGrid grid = new CellGrid(dims, cellDimensions);
        Map<Long, V> entries = new HashMap<>();
        long[] gridDimensions = grid.getGridDimensions();
        long[] gridPosition = new long[dims.length];
        long[] expectedMin = new long[dims.length];
        int[] expectedDimensions = new int[dims.length];

        for (I item : items) {
            ChunkBounds bounds = boundsFn.apply(item);
            if (bounds.getStartCount() != dims.length || bounds.getStopCount() != dims.length) {
                return null;
            }

            for (int axis = 0; axis < dims.length; axis++) {
                long start = bounds.getStart(axis);
                long stop = bounds.getStop(axis);
                int nominalCellDimension = cellDimensions[axis];
                if (start < 0 || stop < start || nominalCellDimension <= 0) {
                    return null;
                }
                if (start % nominalCellDimension != 0) {
                    return null;
                }
                gridPosition[axis] = start / nominalCellDimension;
                if (gridPosition[axis] >= gridDimensions[axis]) {
                    return null;
                }
            }

            grid.getCellDimensions(gridPosition, expectedMin, expectedDimensions);
            for (int axis = 0; axis < dims.length; axis++) {
                if (expectedMin[axis] != bounds.getStart(axis)) {
                    return null;
                }
                if ((long) expectedDimensions[axis] != bounds.getStop(axis) - bounds.getStart(axis)) {
                    return null;
                }
            }

            long cellIndex = grid.getCellGridIndexFlat(gridPosition);
            if (entries.put(cellIndex, valueFn.apply(item)) != null) {
                return null;
            }
        }

        if (entries.size() != cellCount(gridDimensions)) {
            return null;
        }

        return new ChunkGridIndex<>(grid, cellDimensions, entries);
    }
}
