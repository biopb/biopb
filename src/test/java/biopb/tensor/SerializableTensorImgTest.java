package biopb.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.Test;

import com.google.protobuf.ByteString;

/** Tests that the legacy imglib2 adapter externalizes only SerializedTensor. */
public class SerializableTensorImgTest {

    @Test
    public void testExternalizableRoundTripPreservesTheFlightHandle() throws Exception {
        SerializedTensor expected = handle("grpc+tcp://localhost:8815", "test-token");
        SerializableTensorImg<?> image = new SerializableTensorImg<>(expected, 123_456L, null);

        SerializableTensorImg<?> restored = roundTrip(image);

        assertEquals(expected, restored.getSerializedTensor());
    }

    @Test
    public void testLocationParsingRemainsCompatibleWithSerializedHandles() {
        assertNotNull(LocationUris.parse("grpc://localhost:8815"));
        assertNotNull(LocationUris.parse("grpc+tcp://localhost:8815"));
        assertNotNull(LocationUris.parse("localhost:8815"));
    }

    @Test
    public void testConcurrentFirstAccessBuildsOneDelegate() throws Exception {
        // Reconstruction opens a FlightSession, so a racing first access would
        // open two and orphan one -- with no reference left to close it. The
        // plan here carries its own endpoints, so building the delegate needs
        // no server; what is under test is that it happens once.
        SerializableTensorImg<?> image = new SerializableTensorImg<>(
                plannedHandle("grpc+tcp://localhost:8815"), 1_000L, null);

        int threads = 8;
        java.util.concurrent.CountDownLatch start = new java.util.concurrent.CountDownLatch(1);
        java.util.List<Object> delegates =
                java.util.Collections.synchronizedList(new ArrayList<>());
        java.util.List<Exception> failures =
                java.util.Collections.synchronizedList(new ArrayList<>());
        java.util.List<Thread> workers = new ArrayList<>();
        for (int i = 0; i < threads; i++) {
            Thread worker = new Thread(() -> {
                try {
                    start.await();
                    image.numDimensions();
                    delegates.add(delegateOf(image));
                } catch (Exception error) {
                    failures.add(error);
                }
            });
            worker.start();
            workers.add(worker);
        }
        start.countDown();
        for (Thread worker : workers) {
            worker.join(10_000);
        }

        assertEquals("workers failed: " + failures, 0, failures.size());
        assertEquals(threads, delegates.size());
        // Same instance for every thread: exactly one reconstruction happened.
        for (Object seen : delegates) {
            assertEquals(delegates.get(0), seen);
        }
        image.close();
    }

    @Test
    public void testManyImagesOnOneServerShareOneConnection() throws Exception {
        // The reason the cache exists: a worker is handed many SerializedTensors
        // and reconstructs an image from each. One session per image made the
        // channel count scale with the tensors deserialized -- and tensorFromPb
        // is typed RandomAccessibleInterval, so a caller cannot close one.
        FlightSessions.closeAll();
        try {
            for (int i = 0; i < 20; i++) {
                new SerializableTensorImg<>(
                        plannedHandle("grpc+tcp://localhost:8815"), 1_000L, null).numDimensions();
            }
            assertEquals(1, FlightSessions.size());

            // A different server, and a different token on the same server, are
            // each their own connection: a channel's authorization is not shared.
            new SerializableTensorImg<>(
                    plannedHandle("grpc+tcp://localhost:8816"), 1_000L, null).numDimensions();
            new SerializableTensorImg<>(
                    plannedHandle("grpc+tcp://localhost:8815", "cap-token"), 1_000L, null).numDimensions();
            assertEquals(3, FlightSessions.size());
        } finally {
            FlightSessions.closeAll();
        }
        assertEquals(0, FlightSessions.size());
    }

    private static Object delegateOf(SerializableTensorImg<?> image) throws Exception {
        java.lang.reflect.Field field = SerializableTensorImg.class.getDeclaredField("delegate");
        field.setAccessible(true);
        return field.get(image);
    }

    private static SerializableTensorImg<?> roundTrip(SerializableTensorImg<?> image) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
            output.writeObject(image);
        }
        try (ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (SerializableTensorImg<?>) input.readObject();
        }
    }

    /**
     * A handle whose plan already carries its endpoints, so reconstruction
     * needs no server: the cell image is lazy and a gRPC channel does not dial
     * until a call is made.
     */
    private static SerializedTensor plannedHandle(String location) {
        return plannedHandle(location, "");
    }

    private static SerializedTensor plannedHandle(String location, String token) {
        TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                .setArrayId("test-tensor")
                .addShape(4).addShape(4)
                .addChunkShape(2).addChunkShape(2)
                .setDtype("float32")
                .build();
        java.util.List<org.apache.arrow.flight.FlightEndpoint> endpoints = new ArrayList<>();
        for (long y = 0; y < 4; y += 2) {
            for (long x = 0; x < 4; x += 2) {
                TensorTicket ticket = TensorTicket.newBuilder()
                        .setChunkId(ByteString.copyFromUtf8("c-" + y + "-" + x))
                        .build();
                ChunkBounds bounds = ChunkBounds.newBuilder()
                        .addStart(y).addStart(x)
                        .addStop(y + 2).addStop(x + 2)
                        .build();
                endpoints.add(org.apache.arrow.flight.FlightEndpoint
                        .builder(new org.apache.arrow.flight.Ticket(ticket.toByteArray()))
                        .setAppMetadata(bounds.toByteArray())
                        .build());
            }
        }
        FlightInfo plan = new FlightInfo(
                new Schema(new ArrayList<>(),
                        java.util.Collections.singletonMap("chunk_wire_protocol", "2")),
                FlightDescriptor.command(descriptor.toByteArray()),
                endpoints, -1, -1);
        return SerializedTensor.newBuilder()
                .setLocation(location)
                .setAuthToken(token)
                .setFlightInfo(ByteString.copyFrom(plan.serialize()))
                .build();
    }

    private static SerializedTensor handle(String location, String token) {
        TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                .setArrayId("test-tensor")
                .addShape(4)
                .addShape(4)
                .addChunkShape(2)
                .addChunkShape(2)
                .setDtype("float32")
                .build();
        FlightInfo plan = new FlightInfo(
                new Schema(new ArrayList<>()),
                FlightDescriptor.command(descriptor.toByteArray()),
                new ArrayList<>(), -1, -1);
        return SerializedTensor.newBuilder()
                .setLocation(location)
                .setAuthToken(token)
                .setFlightInfo(ByteString.copyFrom(plan.serialize()))
                .build();
    }
}
