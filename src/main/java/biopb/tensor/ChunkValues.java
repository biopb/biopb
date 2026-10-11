package biopb.tensor;

import java.util.List;

import net.imglib2.RandomAccess;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * One chunk's decoded elements, in row-major order.
 *
 * <p>Integer dtypes are held as {@code long[]} and everything else as
 * {@code double[]}: a {@code double} cannot hold an {@code i8}/{@code u8}
 * label id above 2^53, and the one that follows a lost id is a different id
 * that exists (biopb/biopb#1071). Which of the two is a property of the dtype,
 * so it is chosen once per chunk rather than per element.
 */
final class ChunkValues {
    private final double[] reals;
    private final long[] integers;

    private ChunkValues(double[] reals, long[] integers) {
        this.reals = reals;
        this.integers = integers;
    }

    static ChunkValues ofReals(double[] values) {
        return new ChunkValues(values, null);
    }

    static ChunkValues ofIntegers(long[] values) {
        return new ChunkValues(null, values);
    }

    /**
     * Decode a chunk delivered as several rows (one per record batch row, all
     * of one dtype) into one value array, copying each row once.
     *
     * @throws IllegalStateException if the rows disagree on integer vs real
     */
    static ChunkValues decode(List<byte[]> rows, List<String> dtypes) {
        boolean integer = !dtypes.isEmpty() && ChunkDecoder.isInteger(dtypes.get(0));
        int total = 0;
        for (int i = 0; i < rows.size(); i++) {
            if (ChunkDecoder.isInteger(dtypes.get(i)) != integer) {
                throw new IllegalStateException(
                        "Chunk rows mix integer and floating-point dtypes: " + dtypes);
            }
            total += ChunkDecoder.elementCount(rows.get(i), dtypes.get(i));
        }
        if (integer) {
            long[] out = new long[total];
            int offset = 0;
            for (int i = 0; i < rows.size(); i++) {
                offset += ChunkDecoder.decodeChunkIntegers(rows.get(i), dtypes.get(i), out, offset);
            }
            return ofIntegers(out);
        }
        double[] out = new double[total];
        int offset = 0;
        for (int i = 0; i < rows.size(); i++) {
            double[] decoded = ChunkDecoder.decodeChunkBytes(rows.get(i), dtypes.get(i));
            System.arraycopy(decoded, 0, out, offset, decoded.length);
            offset += decoded.length;
        }
        return ofReals(out);
    }

    /** Scatter into {@code access} at {@code bounds}; see {@link TensorChunkCodec}. */
    <T extends NativeType<T> & RealType<T>> void writeTo(RandomAccess<T> access, ChunkBounds bounds) {
        if (integers != null) {
            TensorChunkCodec.writeChunk(access, bounds, integers);
        } else {
            TensorChunkCodec.writeChunk(access, bounds, reals);
        }
    }
}
