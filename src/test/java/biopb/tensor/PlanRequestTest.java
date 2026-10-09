package biopb.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.Test;

/** What a plan says about the request it answers, whichever protocol wrote it. */
public class PlanRequestTest {

    private static FlightInfo plan(
            TensorDescriptor descriptor, byte[] appMetadata, Map<String, String> schemaMetadata) {
        return new FlightInfo(
                new Schema(new ArrayList<>(), schemaMetadata),
                FlightDescriptor.command(descriptor.toByteArray()),
                new ArrayList<>(), -1, -1, false,
                org.apache.arrow.vector.ipc.message.IpcOption.DEFAULT, appMetadata);
    }

    private static SliceHint slice(long start, long stop) {
        return SliceHint.newBuilder().addStart(start).addStop(stop).build();
    }

    @Test
    public void testAV3PlanIsWhatItsAppMetadataSays() {
        TensorReadOption wanted = TensorReadOption.newBuilder()
                .setArrayId("img")
                .setSliceHint(slice(1, 7))
                .addScaleHint(2L)
                .setReductionMethod("area")
                .build();
        Map<String, String> stamp = new HashMap<>();
        stamp.put(WireVersions.FLIGHT_PROTOCOL_METADATA_KEY, "3");

        TensorReadOption request = PlanRequest.of(plan(
                TensorDescriptor.newBuilder().setArrayId("img").build(), wanted.toByteArray(), stamp));

        assertEquals(wanted, request);
    }

    @Test
    public void testAV2PlanIsPutBackTogether() {
        // The slice alone in app_metadata, the scale and method echoed on the
        // descriptor, and no protocol stamp on the schema.
        TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                .setArrayId("img").addScaleHint(2L).setReductionMethod("area").build();

        TensorReadOption request = PlanRequest.of(
                plan(descriptor, slice(1, 7).toByteArray(), Collections.emptyMap()));

        assertEquals("img", request.getArrayId());
        assertEquals(slice(1, 7), request.getSliceHint());
        assertEquals(Collections.singletonList(2L), request.getScaleHintList());
        assertEquals("area", request.getReductionMethod());
    }

    @Test
    public void testASliceIsNotMisreadAsAV3Request() {
        // A SliceHint's bytes parse as a TensorReadOption with a garbled array_id;
        // the stamp on the schema is what says which this is.
        TensorReadOption request = PlanRequest.of(plan(
                TensorDescriptor.newBuilder().setArrayId("img").build(),
                slice(1, 7).toByteArray(), Collections.emptyMap()));

        assertEquals("img", request.getArrayId());
        assertEquals(slice(1, 7), request.getSliceHint());
    }

    @Test
    public void testAPlanThatRecordedNothingComesBackEmpty() {
        TensorReadOption request = PlanRequest.of(plan(
                TensorDescriptor.newBuilder().setArrayId("img").build(),
                new byte[0], Collections.emptyMap()));

        assertFalse(request.hasSliceHint());
        assertTrue(request.getScaleHintList().isEmpty());
    }

    @Test
    public void testTheFlightProtocolRangeIsTwoThroughThree() {
        assertFalse(WireVersions.supportsFlight(1));
        assertTrue(WireVersions.supportsFlight(2));
        assertTrue(WireVersions.supportsFlight(3));
        assertFalse(WireVersions.supportsFlight(4));
    }

    @Test
    public void testTheMismatchNamesTheSideToUpgrade() {
        assertTrue(WireVersions.flightMismatch(1, "x").contains("Upgrade the server"));
        assertTrue(WireVersions.flightMismatch(4, "x").contains("Upgrade the client"));
        assertTrue(WireVersions.flightMismatch(4, "x").contains("v2-v3"));
    }
}
