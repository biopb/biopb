package biopb.tensor;

import static org.junit.Assert.assertEquals;

import java.lang.reflect.Method;

import org.junit.Test;

/**
 * Unit tests for {@link TensorFlightClient}'s static helpers -- the ones that
 * need no live Flight server.
 *
 * <p>The dtype helpers this file used to cover live in {@link TensorChunkCodec}
 * and are tested against the real methods in {@link TensorChunkCodecTest}. They
 * were tested here through private copies, which is how the {@code parseVersion}
 * regression below survived: the copy was correct and the production method was
 * not.
 */
public class TensorFlightClientUtilTest {

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
    public void testParseVersionHandlesDevAndBuildMetadataSuffixes() throws Exception {
        // Regression: parseVersion used split("+") -- an invalid regex that
        // throws PatternSyntaxException on a dev version with a "+gHASH"
        // build-metadata suffix.
        assertVersion(parseVersion("1.2.3"), 1, 2, 3);
        assertVersion(parseVersion("0.3.1.dev43"), 0, 3, 1);
        assertVersion(parseVersion("0.3.1.dev43+gabc123"), 0, 3, 1);
        assertVersion(parseVersion("1.2.3+gabc"), 1, 2, 3);
    }

    @Test
    public void testParseVersionFillsMissingParts() throws Exception {
        assertVersion(parseVersion("1.2"), 1, 2, 0);
        assertVersion(parseVersion("1"), 1, 0, 0);
    }

    private static void assertVersion(int[] version, int major, int minor, int patch) {
        assertEquals(major, version[0]);
        assertEquals(minor, version[1]);
        assertEquals(patch, version[2]);
    }

    /** The production method, reached by reflection rather than re-implemented. */
    private static int[] parseVersion(String version) throws Exception {
        Method method = TensorFlightClient.class.getDeclaredMethod("parseVersion", String.class);
        method.setAccessible(true);
        return (int[]) method.invoke(null, version);
    }
}
