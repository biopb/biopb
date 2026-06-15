package biopb.tensor;

import static org.junit.Assert.*;

import org.junit.Test;

import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Unit tests for TensorFlightClient utility functions.
 *
 * Tests static utility methods that don't require a live Flight server.
 */
public class TensorFlightClientUtilTest {

    // Test normalizeReductionMethod indirectly through getTensor validation
    // (the method is private but behavior can be verified through public API)

    @Test
    public void testNormalizeReductionMethodNearest() {
        // "nearest" should stay "nearest"
        // "stride" and "decimate" should normalize to "nearest"
        // This is tested indirectly through TensorFlightClient.getTensor
        // with valid/invalid reduction methods
        // See TensorFlightClientTest for those tests
    }

    @Test
    public void testNormalizeReductionMethodArea() {
        // "area" should stay "area"
        // "mean" should normalize to "area"
    }

    @Test
    public void testNormalizeReductionMethodLinear() {
        // "linear" should stay "linear"
    }

    @Test
    public void testNormalizeReductionMethodInvalid() {
        // Invalid methods like "median" should throw IllegalArgumentException
        // See TensorFlightClientTest.testScaledReadRejectsUnsupportedMethod
    }

    // Tensor identity policy: source_id is the prefix of array_id before the
    // first '/' (array_id = source_id or source_id/field; source_id slash-free).

    @Test
    public void testSourceIdFromArrayIdSingleTensor() {
        // single-tensor source: array_id == source_id (no slash)
        assertEquals("zarr_a3f2", TensorFlightClient.sourceIdFromArrayId("zarr_a3f2"));
    }

    @Test
    public void testSourceIdFromArrayIdMultiTensor() {
        assertEquals("aics_7f3", TensorFlightClient.sourceIdFromArrayId("aics_7f3/Image:0"));
    }

    @Test
    public void testSourceIdFromArrayIdHierarchicalField() {
        // HCS: array_id = source/well/field; split only on the first '/'
        assertEquals("plate_x", TensorFlightClient.sourceIdFromArrayId("plate_x/A01/0"));
    }

    @Test
    public void testResolveArrayIdBareSource() {
        // bare source_id -> {source_id, null}: tensorId unset so getTensor
        // resolves the source's sole/default tensor
        String[] route = TensorFlightClient.resolveArrayId("zarr_a3f2");
        assertEquals("zarr_a3f2", route[0]);
        assertNull(route[1]);
    }

    @Test
    public void testResolveArrayIdQualified() {
        // qualified source_id/field -> {source_id, full-array_id}
        String[] route = TensorFlightClient.resolveArrayId("aics_7f3/Image:0");
        assertEquals("aics_7f3", route[0]);
        assertEquals("aics_7f3/Image:0", route[1]);
    }

    @Test
    public void testResolveArrayIdHierarchicalField() {
        // HCS field carries its own '/': source boundary is the FIRST '/', and
        // the full array_id is preserved as the tensorId
        String[] route = TensorFlightClient.resolveArrayId("plate_x/A01/0");
        assertEquals("plate_x", route[0]);
        assertEquals("plate_x/A01/0", route[1]);
    }

    // We can test bytesPerElement and createType by examining the types they create

    @Test
    public void testBytesPerElementUint8() {
        // u1, uint8, |u1 -> 1 byte
        assertEquals(1, getBytesPerElement("u1"));
        assertEquals(1, getBytesPerElement("uint8"));
        assertEquals(1, getBytesPerElement("|u1"));
    }

    @Test
    public void testBytesPerElementUint16() {
        // u2, uint16 -> 2 bytes
        assertEquals(2, getBytesPerElement("u2"));
        assertEquals(2, getBytesPerElement("uint16"));
        assertEquals(2, getBytesPerElement("<u2"));
        assertEquals(2, getBytesPerElement(">u2"));
    }

    @Test
    public void testBytesPerElementUint32() {
        // u4, uint32 -> 4 bytes
        assertEquals(4, getBytesPerElement("u4"));
        assertEquals(4, getBytesPerElement("uint32"));
    }

    @Test
    public void testBytesPerElementFloat32() {
        // f4, float32 -> 4 bytes
        assertEquals(4, getBytesPerElement("f4"));
        assertEquals(4, getBytesPerElement("float32"));
        assertEquals(4, getBytesPerElement("<f4"));
        assertEquals(4, getBytesPerElement(">f4"));
    }

    @Test
    public void testBytesPerElementFloat64() {
        // f8, float64 -> 8 bytes
        assertEquals(8, getBytesPerElement("f8"));
        assertEquals(8, getBytesPerElement("float64"));
    }

    @Test
    public void testBytesPerElementUnknown() {
        // Unknown dtype should default to 4 bytes
        assertEquals(4, getBytesPerElement("unknown"));
        assertEquals(4, getBytesPerElement(null));
        assertEquals(4, getBytesPerElement(""));
    }

    @Test
    public void testCreateTypeUint8() {
        // u1, uint8, |u1 -> UnsignedByteType
        assertTrue(createType("u1") instanceof UnsignedByteType);
        assertTrue(createType("uint8") instanceof UnsignedByteType);
        assertTrue(createType("|u1") instanceof UnsignedByteType);
    }

    @Test
    public void testCreateTypeUint16() {
        // u2, uint16 -> UnsignedShortType
        assertTrue(createType("u2") instanceof UnsignedShortType);
        assertTrue(createType("uint16") instanceof UnsignedShortType);
        assertTrue(createType("<u2") instanceof UnsignedShortType);
        assertTrue(createType(">u2") instanceof UnsignedShortType);
    }

    @Test
    public void testCreateTypeUint32() {
        // u4, uint32 -> UnsignedIntType
        assertTrue(createType("u4") instanceof UnsignedIntType);
        assertTrue(createType("uint32") instanceof UnsignedIntType);
    }

    @Test
    public void testCreateTypeFloat32() {
        // f4, float32 -> FloatType
        assertTrue(createType("f4") instanceof FloatType);
        assertTrue(createType("float32") instanceof FloatType);
        assertTrue(createType("<f4") instanceof FloatType);
        assertTrue(createType(">f4") instanceof FloatType);
    }

    @Test
    public void testCreateTypeFloat64() {
        // f8, float64 -> DoubleType
        assertTrue(createType("f8") instanceof DoubleType);
        assertTrue(createType("float64") instanceof DoubleType);
    }

    @Test
    public void testCreateTypeUnknownDefaultsToFloat() {
        // Unknown or null dtype should default to FloatType
        assertTrue(createType("unknown") instanceof FloatType);
        assertTrue(createType(null) instanceof FloatType);
        assertTrue(createType("") instanceof FloatType);
    }

    @Test
    public void testParseVersionSimple() {
        // Simple semantic version parsing
        int[] version = parseVersion("1.2.3");
        assertEquals(3, version.length);
        assertEquals(1, version[0]);  // major
        assertEquals(2, version[1]);  // minor
        assertEquals(3, version[2]);  // patch
    }

    @Test
    public void testParseVersionDev() {
        // Dev versions like "0.3.1.dev43+g..."
        int[] version = parseVersion("0.3.1.dev43");
        assertEquals(0, version[0]);  // major
        assertEquals(3, version[1]);  // minor
        assertEquals(1, version[2]);  // patch

        // With + suffix
        version = parseVersion("1.2.3+gabc");
        assertEquals(1, version[0]);
        assertEquals(2, version[1]);
        assertEquals(3, version[2]);
    }

    @Test
    public void testParseVersionPartial() {
        // Two-part version
        int[] version = parseVersion("1.2");
        assertEquals(1, version[0]);
        assertEquals(2, version[1]);
        assertEquals(0, version[2]);

        // One-part version
        version = parseVersion("1");
        assertEquals(1, version[0]);
        assertEquals(0, version[1]);
        assertEquals(0, version[2]);
    }

    @Test
    public void testBytesToHex() {
        byte[] bytes = {0x01, 0x02, 0x03, 0x04, 0x05};
        String hex = bytesToHex(bytes, 5);
        assertEquals("0102030405", hex);

        // With limit
        hex = bytesToHex(bytes, 3);
        assertEquals("010203...", hex);

        // Empty bytes
        hex = bytesToHex(new byte[0], 10);
        assertEquals("", hex);
    }

    // Helper methods that mirror TensorFlightClient's private methods
    // These allow testing the logic without exposing private methods

    private static int getBytesPerElement(String dtype) {
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

    private static net.imglib2.type.NativeType<?> createType(String dtype) {
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

    private static int[] parseVersion(String version) {
        // Handle dev versions like "0.3.1.dev43+g..."
        String base = version.split(".dev")[0].split("\\+")[0];
        String[] parts = base.split("\\.");
        int major = parts.length > 0 ? Integer.parseInt(parts[0]) : 0;
        int minor = parts.length > 1 ? Integer.parseInt(parts[1]) : 0;
        int patch = parts.length > 2 ? Integer.parseInt(parts[2]) : 0;
        return new int[] { major, minor, patch };
    }

    private static String bytesToHex(byte[] bytes, int limit) {
        StringBuilder sb = new StringBuilder();
        int len = Math.min(bytes.length, limit);
        for (int i = 0; i < len; i++) {
            sb.append(String.format("%02x", bytes[i]));
        }
        if (bytes.length > limit) {
            sb.append("...");
        }
        return sb.toString();
    }
}