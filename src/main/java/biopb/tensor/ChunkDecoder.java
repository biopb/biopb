package biopb.tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Decoder for the unified binary chunk wire schema (biopb/biopb#293).
 *
 * <p>Each chunk crosses the wire as an opaque {@code byte[]} (the "data" column)
 * plus a numpy dtype string (the "dtype" column, e.g. {@code "<i2"}, {@code ">i2"},
 * {@code "<f4"}, {@code "|u1"}) that says how to reinterpret the bytes. This logic
 * is shared by both Flight clients ({@link SerializableTensorImg} and
 * {@link TensorFlightClient}) so it has a single definition to keep in sync.
 */
final class ChunkDecoder {

    private ChunkDecoder() {}

    /**
     * Reinterpret a chunk's raw bytes as doubles per its numpy dtype string.
     *
     * <p>The dtype string carries the byte order ('&lt;' little, '&gt;' big) and
     * the element kind/size (e.g. "u2", "&gt;i2", "float32"), so numpy-native and
     * byte-swapped sources both decode correctly. All values are widened to
     * double, matching the img's value model. Unknown dtypes fall back to float32
     * (mirrors {@code SerializableTensorImg.createType}).
     */
    static double[] decodeChunkBytes(byte[] raw, String dtypeStr) {
        String s = dtypeStr == null ? "" : dtypeStr.trim().toLowerCase();
        ByteOrder order = s.startsWith(">") ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
        // The byte-order mark and the spelled-out aliases are folded away by
        // the same normalizer createType and bytesPerElement use, so this
        // decode and the imglib2 type it lands in cannot disagree about a
        // dtype -- which would be a silent reinterpretation of the bytes.
        String body = TensorChunkCodec.normalizeDtype(s);
        char kind = body.isEmpty() ? 'f' : body.charAt(0);   // 'u', 'i' or 'f'
        if (kind != 'u' && kind != 'i') {
            kind = 'f';
        }
        int size = TensorChunkCodec.bytesPerElement(body);   // unknown dtypes assume f4

        ByteBuffer buf = ByteBuffer.wrap(raw).order(order);
        int n = size == 0 ? 0 : raw.length / size;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            switch (kind) {
                case 'u':
                    switch (size) {
                        case 1: out[i] = buf.get() & 0xFF; break;
                        case 2: out[i] = buf.getShort() & 0xFFFF; break;
                        case 4: out[i] = buf.getInt() & 0xFFFFFFFFL; break;
                        default: { // u8: interpret the 64 bits as UNSIGNED
                            long u = buf.getLong();
                            out[i] = u >= 0
                                    ? (double) u
                                    : (u & Long.MAX_VALUE) + 9223372036854775808.0; // + 2^63
                            break;
                        }
                    }
                    break;
                case 'i':
                    switch (size) {
                        case 1: out[i] = buf.get(); break;
                        case 2: out[i] = buf.getShort(); break;
                        case 4: out[i] = buf.getInt(); break;
                        default: out[i] = (double) buf.getLong(); break;
                    }
                    break;
                default: // 'f'
                    out[i] = size == 8
                            ? buf.getDouble()
                            : size == 2 ? halfToFloat(buf.getShort()) : buf.getFloat();
                    break;
            }
        }
        return out;
    }

    /**
     * Convert an IEEE 754 half-precision (float16) bit pattern to a float.
     *
     * <p>Java 11 has no native float16 decode, so unpack the sign / 5-bit exponent /
     * 10-bit mantissa by hand, handling subnormals, Inf and NaN. Used for chunks
     * whose dtype string is {@code f2}/{@code float16}.
     */
    private static float halfToFloat(short half) {
        int bits = half & 0xffff;
        int sign = (bits >>> 15) & 0x1;
        int exp = (bits >>> 10) & 0x1f;
        int mant = bits & 0x3ff;
        int out;
        if (exp == 0) {
            if (mant == 0) {
                out = sign << 31; // +/- zero
            } else {
                // Subnormal half: normalize into a float32 normal.
                exp = 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3ff;
                out = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 0x1f) {
            out = (sign << 31) | (0xff << 23) | (mant << 13); // Inf / NaN
        } else {
            out = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(out);
    }
}
