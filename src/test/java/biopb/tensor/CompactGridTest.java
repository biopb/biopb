package biopb.tensor;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.apache.arrow.flight.FlightEndpoint;
import org.junit.Test;

/**
 * Cross-language byte-parity for the compact-grid client reconstruction
 * (biopb/biopb#346). The golden vectors are produced by the server's Python
 * codec ({@code biopb.tensor._chunk_codec}); the Java side must match them
 * byte-for-byte, or a compact response would reconstruct chunk_ids the server
 * cannot resolve.
 */
public class CompactGridTest {

    /** {@code encode_chunk_id("test/img", [0,0,0]..[1,40,50])} from Python. */
    private static final String PLAIN_HEX =
            "00000008746573742f696d670003000000000000000000000000000000000000000000000000000000000000000100000000000000280000000000000032";

    /**
     * {@code encode_chunk_id_with_scale("test/img", [0,0,0]..[2,80,100], (1,2,2),
     * "area")} from Python.
     */
    private static final String SCALED_HEX =
            "00000008746573742f696d670003000000000000000000000000000000000000000000000000000000000000000200000000000000500000000000000064000000000000000100000000000000020000000000000002000461726561";

    private static byte[] fromHex(String hex) {
        int n = hex.length() / 2;
        byte[] out = new byte[n];
        for (int i = 0; i < n; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(2 * i, 2 * i + 2), 16);
        }
        return out;
    }

    @Test
    public void encodeChunkIdMatchesPythonCodec() {
        ChunkBounds bounds = ChunkBounds.newBuilder()
                .addStart(0).addStart(0).addStart(0)
                .addStop(1).addStop(40).addStop(50)
                .build();
        assertArrayEquals(fromHex(PLAIN_HEX), TensorChunkCodec.encodeChunkId("test/img", bounds));
    }

    @Test
    public void encodeScaledChunkIdMatchesPythonCodec() {
        ChunkBounds bounds = ChunkBounds.newBuilder()
                .addStart(0).addStart(0).addStart(0)
                .addStop(2).addStop(80).addStop(100)
                .build();
        byte[] got = TensorChunkCodec.encodeChunkIdWithScale(
                "test/img", bounds, new long[] {1, 2, 2}, "area");
        assertArrayEquals(fromHex(SCALED_HEX), got);
    }

    @Test
    public void expandRegeneratesTheServerGrid() {
        // A 3-plane grid (one z-plane per chunk), no scale -- the Python
        // expand_compact_grid golden for this descriptor is three endpoints with
        // the chunk_ids and logical bounds asserted below.
        TensorDescriptor.Builder descriptor = TensorDescriptor.newBuilder()
                .setArrayId("test/img")
                .setChunkArrayId("test/img")
                .setDtype("<u2")
                .addShape(3).addShape(40).addShape(50)
                .addChunkShape(1).addChunkShape(40).addChunkShape(50);
        descriptor.getSliceHintBuilder()
                .addStart(0).addStart(0).addStart(0)
                .addStop(3).addStop(40).addStop(50);

        List<FlightEndpoint> endpoints = CompactGrid.expandToFlightEndpoints(descriptor.build());
        assertEquals(3, endpoints.size());

        for (int plane = 0; plane < 3; plane++) {
            FlightEndpoint ep = endpoints.get(plane);
            TensorTicket ticket = TensorChunkCodec.parseTicket(ep.getTicket().getBytes());
            ChunkBounds virtual = ChunkBounds.newBuilder()
                    .addStart(plane).addStart(0).addStart(0)
                    .addStop(plane + 1).addStop(40).addStop(50)
                    .build();
            // chunk_id encodes the VIRTUAL bounds; app-metadata is the LOGICAL
            // (output) bounds, which equal virtual for an unscaled read.
            assertArrayEquals(
                    TensorChunkCodec.encodeChunkId("test/img", virtual),
                    ticket.getChunkId().toByteArray());
            ChunkBounds logical = TensorChunkCodec.parseChunkBounds(ep.getAppMetadata());
            assertEquals(Arrays.asList((long) plane, 0L, 0L), logical.getStartList());
            assertEquals(Arrays.asList((long) plane + 1, 40L, 50L), logical.getStopList());
        }
    }
}
