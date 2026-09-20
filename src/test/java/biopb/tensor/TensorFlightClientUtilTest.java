package biopb.tensor;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * Unit tests for {@link TensorFlightClient}'s static helpers -- the ones that
 * need no live Flight server.
 *
 * <p>The dtype helpers this file used to cover live in {@link TensorChunkCodec}
 * and are tested against the real methods in {@link TensorChunkCodecTest}.
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
}
