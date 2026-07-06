package biopb.tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

/**
 * Unit tests for the unified-binary chunk decode (biopb/biopb#293), focused on
 * the dtypes that regressed: float16 (no native Java decode) and uint64 (the
 * unsigned->signed overflow). Exercises the shared {@link ChunkDecoder}, which
 * both {@link SerializableTensorImg} and {@link TensorFlightClient} delegate to.
 */
public class DecodeChunkBytesTest {

    private static double[] decode(byte[] raw, String dtype) {
        return ChunkDecoder.decodeChunkBytes(raw, dtype);
    }

    private static byte[] leShort(int v) {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) v).array();
    }

    @Test
    public void float16_decodes_known_values() throws Exception {
        // 1.0=0x3C00, 2.0=0x4000, 0.5=0x3800, -2.0=0xC000
        Assert.assertEquals(1.0, decode(leShort(0x3C00), "<f2")[0], 0.0);
        Assert.assertEquals(2.0, decode(leShort(0x4000), "<f2")[0], 0.0);
        Assert.assertEquals(0.5, decode(leShort(0x3800), "<f2")[0], 0.0);
        Assert.assertEquals(-2.0, decode(leShort(0xC000), "<f2")[0], 0.0);
    }

    @Test
    public void float16_smallest_subnormal() throws Exception {
        // 0x0001 is the smallest positive subnormal half: 2^-24.
        Assert.assertEquals(Math.pow(2, -24), decode(leShort(0x0001), "<f2")[0], 1e-12);
    }

    @Test
    public void float16_honors_big_endian_byteorder() throws Exception {
        byte[] be = ByteBuffer.allocate(2).order(ByteOrder.BIG_ENDIAN)
                .putShort((short) 0x3C00).array();
        Assert.assertEquals(1.0, decode(be, ">f2")[0], 0.0);
    }

    @Test
    public void float16_falls_back_is_not_used_for_float32() throws Exception {
        // Regression guard: f4 must still read 4 bytes as a float, not 2.
        byte[] b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(3.5f).array();
        double[] out = decode(b, "<f4");
        Assert.assertEquals(1, out.length);
        Assert.assertEquals(3.5, out[0], 0.0);
    }

    @Test
    public void uint64_max_is_unsigned_not_negative_one() throws Exception {
        byte[] allOnes = new byte[8];
        Arrays.fill(allOnes, (byte) 0xFF);
        double v = decode(allOnes, "<u8")[0];
        // Old signed decode gave -1.0; uint64 max is 2^64 - 1 (~1.8446744e19).
        Assert.assertTrue("expected a large positive value, got " + v, v > 1.8e19);
        Assert.assertEquals(Math.pow(2, 64), v, 4096.0); // within one ulp at 2^64
    }

    @Test
    public void int64_stays_signed() throws Exception {
        byte[] allOnes = new byte[8];
        Arrays.fill(allOnes, (byte) 0xFF);
        Assert.assertEquals(-1.0, decode(allOnes, "<i8")[0], 0.0);
    }

    @Test
    public void uint32_is_unsigned() throws Exception {
        byte[] b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(-1).array();
        Assert.assertEquals(4294967295.0, decode(b, "<u4")[0], 0.0); // 2^32 - 1
    }
}
