package biopb.tensor;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;

import com.google.protobuf.InvalidProtocolBufferException;

import net.imglib2.RandomAccess;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Pure chunk-codec helpers shared by the two Flight tensor clients
 * ({@link SerializableTensorImg} and {@link TensorFlightClient}).
 *
 * <p>These translate between the tensor wire messages ({@link TensorDescriptor},
 * {@link ChunkBounds}, {@link TensorTicket}) and the imglib2 value model: a
 * numpy dtype string to an imglib2 {@link NativeType}, chunk byte/cell counts,
 * and scattering a decoded chunk into a {@link RandomAccess}. Every method here
 * was duplicated verbatim in both clients (biopb/biopb#277 item D); this is
 * their single home. Raw bytes to {@code double[]} decoding lives one level
 * down in {@link ChunkDecoder}, and location-URI parsing in {@link LocationUris}.
 */
final class TensorChunkCodec {

    private TensorChunkCodec() {}

    /** Convert a protobuf {@code repeated int64} to a {@code long[]}. */
    static long[] toLongArray(List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    /** Convert a protobuf {@code repeated int64} to an {@code int[]} (overflow-checked). */
    static int[] toIntArray(List<Long> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = Math.toIntExact(values.get(i));
        }
        return out;
    }

    /** Product of grid dimensions -- the number of cells in a chunk grid. */
    static long cellCount(long[] gridDimensions) {
        long count = 1L;
        for (long axisCount : gridDimensions) {
            count *= axisCount;
        }
        return count;
    }

    /** Approximate on-the-wire bytes of one chunk, for sizing the cell cache. */
    static long estimateChunkBytes(TensorDescriptor descriptor) {
        long elements = 1L;
        for (long dim : descriptor.getChunkShapeList()) {
            elements *= Math.max(dim, 1L);
        }
        return elements * bytesPerElement(descriptor.getDtype());
    }

    /** Bytes per element for a numpy dtype string; unknown dtypes assume 4 (float32). */
    static int bytesPerElement(String dtype) {
        String normalized = dtype == null ? "" : dtype.trim().toLowerCase();
        switch (normalized) {
            case "u1":
            case "uint8":
            case "|u1":
                return 1;
            case "<u2":
            case ">u2":
            case "u2":
            case "uint16":
                return 2;
            case "<u4":
            case ">u4":
            case "u4":
            case "uint32":
            case "<f4":
            case ">f4":
            case "f4":
            case "float32":
                return 4;
            case "<f8":
            case ">f8":
            case "f8":
            case "float64":
                return 8;
            default:
                return 4;
        }
    }

    /** imglib2 type for a numpy dtype string; unknown dtypes fall back to float32. */
    static NativeType<?> createType(String dtype) {
        String normalized = dtype == null ? "" : dtype.trim().toLowerCase();
        switch (normalized) {
            case "u1":
            case "uint8":
            case "|u1":
                return new UnsignedByteType();
            case "<u2":
            case ">u2":
            case "u2":
            case "uint16":
                return new UnsignedShortType();
            case "<u4":
            case ">u4":
            case "u4":
            case "uint32":
                return new UnsignedIntType();
            case "<f8":
            case ">f8":
            case "f8":
            case "float64":
                return new DoubleType();
            case "<f4":
            case ">f4":
            case "f4":
            case "float32":
            default:
                return new FloatType();
        }
    }

    /**
     * Encode a {@code chunk_id} from an array_id and bounds, byte-for-byte
     * identical to the server's Python codec (biopb/biopb#346). The client needs
     * this only on the compact-grid read path, where it regenerates chunk_ids
     * arithmetically instead of reading them off explicit endpoints.
     *
     * <p>Format (all integers big-endian, which is {@link ByteBuffer}'s default):
     * {@code [uint32 array_id_len][array_id utf-8][uint16 ndim][int64*ndim
     * start][int64*ndim stop]}.
     */
    static byte[] encodeChunkId(String arrayId, ChunkBounds bounds) {
        byte[] idBytes = arrayId.getBytes(StandardCharsets.UTF_8);
        int ndim = bounds.getStartCount();
        ByteBuffer buffer = ByteBuffer.allocate(4 + idBytes.length + 2 + 16 * ndim);
        buffer.putInt(idBytes.length);
        buffer.put(idBytes);
        buffer.putShort((short) ndim);
        for (int axis = 0; axis < ndim; axis++) {
            buffer.putLong(bounds.getStart(axis));
        }
        for (int axis = 0; axis < ndim; axis++) {
            buffer.putLong(bounds.getStop(axis));
        }
        return buffer.array();
    }

    /**
     * Encode a scaled {@code chunk_id}: the standard bounds encoding followed by
     * {@code [int64*ndim scale][uint16 method_len][method utf-8]}. Matches the
     * server's {@code encode_chunk_id_with_scale}.
     */
    static byte[] encodeChunkIdWithScale(
            String arrayId, ChunkBounds bounds, long[] scale, String method) {
        byte[] base = encodeChunkId(arrayId, bounds);
        byte[] methodBytes = method.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer.allocate(base.length + 8 * scale.length + 2 + methodBytes.length);
        buffer.put(base);
        for (long factor : scale) {
            buffer.putLong(factor);
        }
        buffer.putShort((short) methodBytes.length);
        buffer.put(methodBytes);
        return buffer.array();
    }

    /** Parse a {@link TensorTicket} from an endpoint ticket's bytes. */
    static TensorTicket parseTicket(byte[] bytes) {
        try {
            return TensorTicket.parseFrom(bytes);
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse TensorTicket", e);
        }
    }

    /** Parse {@link ChunkBounds} from an endpoint's app metadata. */
    static ChunkBounds parseChunkBounds(byte[] bytes) {
        try {
            return ChunkBounds.parseFrom(bytes);
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse ChunkBounds", e);
        }
    }

    /**
     * Scatter a row-major decoded chunk into {@code access} at the global
     * position given by {@code bounds}.
     *
     * <p>{@code values} is the chunk's elements in row-major (C) order; its
     * length must equal the product of the chunk's per-axis extents or an
     * {@link IllegalStateException} is thrown.
     */
    static <T extends NativeType<T> & RealType<T>> void writeChunk(
            RandomAccess<T> access,
            ChunkBounds bounds,
            double[] values) {

        long[] start = toLongArray(bounds.getStartList());
        long[] stop = toLongArray(bounds.getStopList());
        long[] chunkShape = new long[start.length];
        long expectedSize = 1L;
        for (int axis = 0; axis < start.length; axis++) {
            chunkShape[axis] = stop[axis] - start[axis];
            expectedSize *= chunkShape[axis];
        }
        if (expectedSize != values.length) {
            throw new IllegalStateException(
                    "Chunk size mismatch: expected " + expectedSize + " values but received " + values.length);
        }

        long[] localPosition = new long[chunkShape.length];
        long[] globalPosition = new long[chunkShape.length];
        for (int index = 0; index < values.length; index++) {
            rowMajorPosition(index, chunkShape, localPosition);
            for (int axis = 0; axis < chunkShape.length; axis++) {
                globalPosition[axis] = start[axis] + localPosition[axis];
            }
            access.setPosition(globalPosition);
            access.get().setReal(values[index]);
        }
    }

    private static void rowMajorPosition(int index, long[] shape, long[] position) {
        long remaining = index;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            position[axis] = remaining % shape[axis];
            remaining /= shape[axis];
        }
    }
}
