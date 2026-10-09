package biopb.tensor;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.vector.ipc.message.IpcOption;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.Test;

import com.google.protobuf.ByteString;

/**
 * Sealed stub tickets (biopb/biopb#1112): a plan is issued as one stub on the
 * descriptor plus a grid index per endpoint, and the client reads a chunk by
 * concatenating the two, which protobuf merges into one {@code ChunkRef}.
 */
public class TicketStubTest {

    private static TicketStub stub() {
        return TicketStub.newBuilder()
                .setIdentity(ByteString.copyFromUtf8("an-identity"))
                .setGrant(ChunkGrant.newBuilder()
                        .addWindowStart(0).addWindowStart(0)
                        .addWindowStop(4).addWindowStop(4)
                        .setExpiresAt(1_000_000L)
                        .setSeal(ByteString.copyFromUtf8("a-seal")))
                .build();
    }

    private static byte[] stubTicket() {
        return TensorTicket.newBuilder()
                .setChunkRef(ChunkRef.newBuilder().setStub(stub()))
                .build()
                .toByteArray();
    }

    private static Ticket indexTicket(int... index) {
        ChunkRef.Builder ref = ChunkRef.newBuilder();
        for (int i : index) {
            ref.addIndex(i);
        }
        return new Ticket(TensorTicket.newBuilder().setChunkRef(ref).build().toByteArray());
    }

    private static FlightInfo plan(TensorDescriptor descriptor, List<FlightEndpoint> endpoints) {
        return new FlightInfo(
                new Schema(new ArrayList<>(), Collections.emptyMap()),
                FlightDescriptor.command(descriptor.toByteArray()),
                endpoints, -1, -1, false, IpcOption.DEFAULT, new byte[0]);
    }

    private static List<FlightEndpoint> endpoints(Ticket... tickets) {
        List<FlightEndpoint> out = new ArrayList<>();
        for (Ticket ticket : tickets) {
            out.add(new FlightEndpoint(ticket));
        }
        return out;
    }

    @Test
    public void testTheStubAndTheIndexJoinIntoOneChunkRef() throws Exception {
        Ticket joined = Imglib2TensorFactory.ticketOf(stubTicket(), indexTicket(1, 2));

        ChunkRef ref = TensorTicket.parseFrom(joined.getBytes()).getChunkRef();
        assertEquals(stub(), ref.getStub());
        assertEquals(java.util.Arrays.asList(1, 2), ref.getIndexList());
    }

    @Test
    public void testAPlanWithNoStubIsReadByItsOwnTickets() {
        Ticket whole = new Ticket(TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFromUtf8("c-0")).build().toByteArray());

        assertEquals(whole, Imglib2TensorFactory.ticketOf(new byte[0], whole));
    }

    @Test
    public void testAPlanIsSealedWhenItsEndpointsAreIndicesUnderAStub() {
        TensorDescriptor stubbed = TensorDescriptor.newBuilder()
                .setArrayId("img").setTicketStub(ByteString.copyFrom(stubTicket())).build();
        TensorDescriptor plain = TensorDescriptor.newBuilder().setArrayId("img").build();

        assertTrue(TensorFlightClient.isSealed(plan(stubbed, endpoints(indexTicket(0, 0)))));
        assertFalse(TensorFlightClient.isSealed(plan(plain, endpoints(indexTicket(0, 0)))));
        // A describe has no endpoints to seal, and the reader would have nothing
        // else to plan with.
        assertFalse(TensorFlightClient.isSealed(plan(stubbed, endpoints())));
    }

    @Test
    public void testASealedReferenceCarriesItsRoiTicket() {
        byte[] roi = TensorTicket.newBuilder()
                .setRoiRead(RoiRead.newBuilder().setArrayId("img")
                        .setGrant(RoiGrant.newBuilder().setSeal(ByteString.copyFromUtf8("r"))))
                .build().toByteArray();
        TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                .setArrayId("img").setRoiTicket(ByteString.copyFrom(roi)).build();
        SerializedTensor pb = SerializedTensor.newBuilder()
                .setLocation("grpc://x:1")
                .setFlightInfo(ByteString.copyFrom(plan(descriptor, endpoints()).serialize()))
                .build();

        assertArrayEquals(roi, TensorFlightClient.roiTicketOf(pb));
    }

    @Test
    public void testAReferenceWithNoRoiTicketHasNone() {
        TensorDescriptor descriptor = TensorDescriptor.newBuilder().setArrayId("img").build();
        SerializedTensor pb = SerializedTensor.newBuilder()
                .setLocation("grpc://x:1")
                .setFlightInfo(ByteString.copyFrom(plan(descriptor, endpoints()).serialize()))
                .build();

        assertNull(TensorFlightClient.roiTicketOf(pb));
    }

    @Test
    public void testASetNameMergesOntoTheSealedRoiTicket() throws Exception {
        byte[] sealed = TensorTicket.newBuilder()
                .setRoiRead(RoiRead.newBuilder().setArrayId("img")
                        .setGrant(RoiGrant.newBuilder().setSeal(ByteString.copyFromUtf8("r"))))
                .build().toByteArray();

        RoiRead read = TensorTicket.parseFrom(
                TensorFlightClient.roiReadTicket("img", "nuclei", sealed)).getRoiRead();

        assertEquals("img", read.getArrayId());
        assertEquals("nuclei", read.getSetName());
        assertEquals("r", read.getGrant().getSeal().toStringUtf8());
    }

    @Test
    public void testNoSetNameReadsEverySetThroughTheTicketAsIs() {
        byte[] sealed = TensorTicket.newBuilder()
                .setRoiRead(RoiRead.newBuilder().setArrayId("img"))
                .build().toByteArray();

        assertArrayEquals(sealed, TensorFlightClient.roiReadTicket("img", "", sealed));
    }

    @Test
    public void testWithoutATicketTheReadNamesTheTensorItself() throws Exception {
        RoiRead read = TensorTicket.parseFrom(
                TensorFlightClient.roiReadTicket("img", "nuclei", null)).getRoiRead();

        assertEquals("img", read.getArrayId());
        assertEquals("nuclei", read.getSetName());
        assertFalse(read.hasGrant());
    }
}
