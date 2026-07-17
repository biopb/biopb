package biopb.tensor;

import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.Ticket;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

/**
 * Client-side reconstruction of a compact-grid {@code GetFlightInfo} response
 * (biopb/biopb#346).
 *
 * <p>When the client opts in with {@link TensorReadOption} {@code compact_grid_ok}
 * and the resolved plan is a regular chunk grid, the server omits the per-chunk
 * endpoint list and instead stamps {@code chunk_array_id} (and the realized
 * {@code slice_hint}) on the response descriptor. A {@code chunk_id} is a pure
 * function of {@code (array_id, bounds)}, so the whole endpoint list is derivable
 * from the grid without the server enumerating it -- O(1) transfer instead of
 * O(n_chunks). This is the Java twin of the server's Python
 * {@code expand_compact_grid}; the two must produce byte-identical chunk_ids.
 *
 * <p>Every consumer routes {@link FlightInfo#getEndpoints()} through
 * {@link #resolveFlightEndpoints}, so the compact/explicit fork lives in one
 * place. An old server (or a non-regular plan) still returns explicit endpoints,
 * handled by the early return.
 */
final class CompactGrid {

    private CompactGrid() {}

    /**
     * The endpoints for a {@code GetFlightInfo} response: the server's own list
     * when it enumerated them, or the arithmetically-regenerated list when the
     * server answered compact (empty endpoints + {@code chunk_array_id} on the
     * descriptor). An empty endpoint list with an unset {@code chunk_array_id} is
     * a genuinely empty tensor, returned as-is.
     */
    static List<FlightEndpoint> resolveFlightEndpoints(FlightInfo info) {
        if (!info.getEndpoints().isEmpty()) {
            return info.getEndpoints();
        }
        TensorDescriptor descriptor;
        try {
            descriptor = TensorDescriptor.parseFrom(info.getDescriptor().getCommand());
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse compact response descriptor", e);
        }
        if (descriptor.getChunkArrayId().isEmpty()) {
            return info.getEndpoints();
        }
        return expandToFlightEndpoints(descriptor);
    }

    /**
     * The regenerated endpoint list for a compact response descriptor, as
     * synthetic {@link FlightEndpoint}s (ticket = {@link TensorTicket} wrapping
     * the {@code chunk_id}; app-metadata = the logical {@link ChunkBounds}) --
     * byte-identical to what the server would have enumerated explicitly.
     */
    static List<FlightEndpoint> expandToFlightEndpoints(TensorDescriptor descriptor) {
        List<FlightEndpoint> endpoints = new ArrayList<>();
        for (Entry entry : expand(descriptor)) {
            TensorTicket ticket = TensorTicket.newBuilder()
                    .setChunkId(ByteString.copyFrom(entry.chunkId))
                    .build();
            endpoints.add(FlightEndpoint.builder(new Ticket(ticket.toByteArray()))
                    .setAppMetadata(entry.bounds.toByteArray())
                    .build());
        }
        return endpoints;
    }

    /** One regenerated chunk: its wire {@code chunk_id} and its logical bounds. */
    private static final class Entry {
        final byte[] chunkId;
        final ChunkBounds bounds;

        Entry(byte[] chunkId, ChunkBounds bounds) {
            this.chunkId = chunkId;
            this.bounds = bounds;
        }
    }

    /**
     * Regenerate the {@code (chunk_id, logical_bounds)} list a regular grid would
     * have enumerated, from a compact response descriptor. Mirrors the server's
     * {@code expand_compact_grid} exactly: {@code slice_hint} carries the realized
     * virtual-coordinate bounds [start, stop), {@code chunk_array_id} is the id
     * the chunk_ids are encoded with, and a present {@code scale_hint} means the
     * chunk_ids are scale-encoded and the logical grid is downsampled.
     */
    private static List<Entry> expand(TensorDescriptor descriptor) {
        int ndim = descriptor.getShapeCount();
        if (!descriptor.hasSliceHint()) {
            throw new IllegalStateException(
                    "Compact response descriptor is missing slice_hint; cannot reconstruct grid");
        }

        boolean scaled = descriptor.getScaleHintCount() > 0;
        long[] logicalChunk = new long[ndim];
        long[] scale = new long[ndim];
        long[] rstart = new long[ndim];
        long[] rstop = new long[ndim];
        long[] vcs = new long[ndim];
        long[] gridCounts = new long[ndim];
        for (int axis = 0; axis < ndim; axis++) {
            logicalChunk[axis] = descriptor.getChunkShape(axis);
            scale[axis] = scaled ? descriptor.getScaleHint(axis) : 1L;
            rstart[axis] = descriptor.getSliceHint().getStart(axis);
            rstop[axis] = descriptor.getSliceHint().getStop(axis);
            // virtual chunk size = logical chunk size * scale (recovers the
            // server's virtual_chunk_size, a multiple of scale).
            vcs[axis] = logicalChunk[axis] * scale[axis];
            gridCounts[axis] = ceilDiv(rstop[axis] - rstart[axis], vcs[axis]);
        }

        String chunkArrayId = descriptor.getChunkArrayId();
        String method = descriptor.getReductionMethod();

        long total = 1L;
        for (long count : gridCounts) {
            total *= count;
        }

        List<Entry> out = new ArrayList<>();
        long[] index = new long[ndim];
        for (long flat = 0; flat < total; flat++) {
            // flat -> per-axis index, row-major (last axis fastest), matching the
            // server's enumeration order. Order is immaterial to the grid index
            // the consumer builds, but keeping it identical eases equivalence.
            long remaining = flat;
            for (int axis = ndim - 1; axis >= 0; axis--) {
                index[axis] = remaining % gridCounts[axis];
                remaining /= gridCounts[axis];
            }

            ChunkBounds.Builder virtual = ChunkBounds.newBuilder();
            long[] vstart = new long[ndim];
            long[] vstop = new long[ndim];
            for (int axis = 0; axis < ndim; axis++) {
                vstart[axis] = rstart[axis] + index[axis] * vcs[axis];
                vstop[axis] = Math.min(vstart[axis] + vcs[axis], rstop[axis]);
            }
            for (int axis = 0; axis < ndim; axis++) {
                virtual.addStart(vstart[axis]);
            }
            for (int axis = 0; axis < ndim; axis++) {
                virtual.addStop(vstop[axis]);
            }
            ChunkBounds virtualBounds = virtual.build();

            ChunkBounds.Builder logical = ChunkBounds.newBuilder();
            byte[] chunkId;
            if (scaled) {
                chunkId = TensorChunkCodec.encodeChunkIdWithScale(
                        chunkArrayId, virtualBounds, scale, method);
                for (int axis = 0; axis < ndim; axis++) {
                    logical.addStart((vstart[axis] - rstart[axis]) / scale[axis]);
                }
                for (int axis = 0; axis < ndim; axis++) {
                    logical.addStop(ceilDiv(vstop[axis] - rstart[axis], scale[axis]));
                }
            } else {
                chunkId = TensorChunkCodec.encodeChunkId(chunkArrayId, virtualBounds);
                for (int axis = 0; axis < ndim; axis++) {
                    logical.addStart(vstart[axis] - rstart[axis]);
                }
                for (int axis = 0; axis < ndim; axis++) {
                    logical.addStop(vstop[axis] - rstart[axis]);
                }
            }
            out.add(new Entry(chunkId, logical.build()));
        }
        return out;
    }

    /** Ceiling division for non-negative {@code a} and positive {@code b}. */
    private static long ceilDiv(long a, long b) {
        return -Math.floorDiv(-a, b);
    }
}
