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

    private static SerializableTensorImg<?> roundTrip(SerializableTensorImg<?> image) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
            output.writeObject(image);
        }
        try (ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (SerializableTensorImg<?>) input.readObject();
        }
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
