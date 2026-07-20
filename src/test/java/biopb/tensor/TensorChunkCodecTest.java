package biopb.tensor;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import net.imglib2.RandomAccess;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Unit tests for {@link TensorChunkCodec}, the chunk-codec helpers that were
 * duplicated verbatim across both Flight clients (biopb/biopb#277 item D).
 * Extracting them to a single package-private class is what makes them directly
 * testable here rather than only through a full server round-trip.
 */
public class TensorChunkCodecTest {

    @Test
    public void createType_maps_numpy_dtypes_to_imglib2_types() {
        Assert.assertTrue(TensorChunkCodec.createType("u1") instanceof UnsignedByteType);
        Assert.assertTrue(TensorChunkCodec.createType("|u1") instanceof UnsignedByteType);
        Assert.assertTrue(TensorChunkCodec.createType("uint16") instanceof UnsignedShortType);
        Assert.assertTrue(TensorChunkCodec.createType(">u2") instanceof UnsignedShortType);
        Assert.assertTrue(TensorChunkCodec.createType("u4") instanceof UnsignedIntType);
        Assert.assertTrue(TensorChunkCodec.createType("float64") instanceof DoubleType);
        Assert.assertTrue(TensorChunkCodec.createType("<f8") instanceof DoubleType);
        // Unknown / float32 both fall back to FloatType.
        Assert.assertTrue(TensorChunkCodec.createType("f4") instanceof FloatType);
        Assert.assertTrue(TensorChunkCodec.createType("something-weird") instanceof FloatType);
        Assert.assertTrue(TensorChunkCodec.createType(null) instanceof FloatType);
    }

    @Test
    public void bytesPerElement_matches_dtype_widths() {
        Assert.assertEquals(1, TensorChunkCodec.bytesPerElement("u1"));
        Assert.assertEquals(2, TensorChunkCodec.bytesPerElement(">u2"));
        Assert.assertEquals(4, TensorChunkCodec.bytesPerElement("u4"));
        Assert.assertEquals(4, TensorChunkCodec.bytesPerElement("<f4"));
        Assert.assertEquals(8, TensorChunkCodec.bytesPerElement("float64"));
        Assert.assertEquals(4, TensorChunkCodec.bytesPerElement("unknown")); // float32 fallback
    }

    @Test
    public void estimateChunkBytes_is_element_count_times_width() {
        TensorDescriptor d = TensorDescriptor.newBuilder()
                .addChunkShape(2).addChunkShape(3)
                .setDtype("u2")
                .build();
        Assert.assertEquals(2L * 3L * 2L, TensorChunkCodec.estimateChunkBytes(d));
    }

    @Test
    public void estimateChunkBytes_treats_zero_axis_as_one() {
        // Guards the Math.max(dim, 1L) clamp, so a 0-length axis doesn't zero the estimate.
        TensorDescriptor d = TensorDescriptor.newBuilder()
                .addChunkShape(0).addChunkShape(4)
                .setDtype("u1")
                .build();
        Assert.assertEquals(4L, TensorChunkCodec.estimateChunkBytes(d));
    }

    @Test
    public void cellCount_is_product_of_grid_dimensions() {
        Assert.assertEquals(24L, TensorChunkCodec.cellCount(new long[] {2, 3, 4}));
        Assert.assertEquals(1L, TensorChunkCodec.cellCount(new long[] {}));
    }

    @Test
    public void toLongArray_and_toIntArray_convert_element_wise() {
        Assert.assertArrayEquals(new long[] {5, 6, 7},
                TensorChunkCodec.toLongArray(Arrays.asList(5L, 6L, 7L)));
        Assert.assertArrayEquals(new int[] {5, 6, 7},
                TensorChunkCodec.toIntArray(Arrays.asList(5L, 6L, 7L)));
    }

    @Test(expected = ArithmeticException.class)
    public void toIntArray_rejects_overflowing_values() {
        TensorChunkCodec.toIntArray(Arrays.asList((long) Integer.MAX_VALUE + 1L));
    }

    @Test
    public void writeChunk_scatters_row_major_values_to_global_positions() {
        ArrayImg<FloatType, ?> img = new ArrayImgFactory<>(new FloatType()).create(2, 2);
        RandomAccess<FloatType> access = img.randomAccess();
        // Chunk covering the whole [2,2] image, row-major values.
        ChunkBounds bounds = ChunkBounds.newBuilder()
                .addStart(0).addStart(0)
                .addStop(2).addStop(2)
                .build();
        TensorChunkCodec.writeChunk(access, bounds, new double[] {1, 2, 3, 4});

        Assert.assertEquals(1.0f, valueAt(access, 0, 0), 0.0f);
        Assert.assertEquals(2.0f, valueAt(access, 0, 1), 0.0f);
        Assert.assertEquals(3.0f, valueAt(access, 1, 0), 0.0f);
        Assert.assertEquals(4.0f, valueAt(access, 1, 1), 0.0f);
    }

    @Test
    public void writeChunk_honors_a_nonzero_chunk_origin() {
        ArrayImg<FloatType, ?> img = new ArrayImgFactory<>(new FloatType()).create(4, 1);
        RandomAccess<FloatType> access = img.randomAccess();
        // A 2x1 chunk placed at x=2.
        ChunkBounds bounds = ChunkBounds.newBuilder()
                .addStart(2).addStart(0)
                .addStop(4).addStop(1)
                .build();
        TensorChunkCodec.writeChunk(access, bounds, new double[] {7, 8});

        Assert.assertEquals(0.0f, valueAt(access, 0, 0), 0.0f);
        Assert.assertEquals(7.0f, valueAt(access, 2, 0), 0.0f);
        Assert.assertEquals(8.0f, valueAt(access, 3, 0), 0.0f);
    }

    @Test(expected = IllegalStateException.class)
    public void writeChunk_rejects_value_count_mismatch() {
        ArrayImg<FloatType, ?> img = new ArrayImgFactory<>(new FloatType()).create(2, 2);
        ChunkBounds bounds = ChunkBounds.newBuilder()
                .addStart(0).addStart(0)
                .addStop(2).addStop(2)
                .build();
        // Expects 4 values, given 3.
        TensorChunkCodec.writeChunk(img.randomAccess(), bounds, new double[] {1, 2, 3});
    }

    private static float valueAt(RandomAccess<? extends NativeType<?>> access, long x, long y) {
        access.setPosition(new long[] {x, y});
        return ((FloatType) access.get()).get();
    }
}
