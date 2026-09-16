package biopb.tensor;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightProducer;
import org.apache.arrow.flight.FlightServer;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.NoOpFlightProducer;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.Assert;
import org.junit.Test;

import com.google.gson.Gson;
import com.google.protobuf.ByteString;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;

public class TensorFlightClientTest {

    @Test
    public void testListSourcesAndTensorLookup() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                @SuppressWarnings("deprecation")
                Map<String, DataSourceDescriptor> sources = client.listSources();
                Assert.assertTrue(sources.containsKey("test-source"));

                DataSourceDescriptor sourceDesc = sources.get("test-source");
                Assert.assertEquals(1, sourceDesc.getTensorsCount());
                Assert.assertEquals("test-tensor", sourceDesc.getTensors(0).getArrayId());
                Assert.assertEquals(Arrays.asList(4L, 4L), sourceDesc.getTensors(0).getShapeList());
            }
        }
    }

    @Test
    public void testCatalogRowsDecodeToStructsCarryingIsResolved() throws Exception {
        // The same rows listSources() reads, through the decoder that is not
        // bounded by the generated message (biopb/biopb#1032).
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                try (VectorSchemaRoot root = client.querySources(
                        "SELECT " + TensorFlightClient.SOURCE_ROW_COLUMNS + " FROM sources")) {
                    List<CatalogSource> sources = TensorFlightClient.sourcesFromRows(root);
                    Assert.assertEquals(1, sources.size());
                    CatalogSource source = sources.get(0);
                    Assert.assertEquals("test-source", source.getSourceId());
                    Assert.assertTrue(source.isResolved());
                    Assert.assertEquals(1, source.getTensors().size());
                    Assert.assertEquals("test-tensor", source.getTensors().get(0).getArrayId());
                    Assert.assertEquals(Arrays.asList(4L, 4L), source.getTensors().get(0).getShape());
                }
            }
        }
    }

    @Test
    public void testMaterializesBaseArrayFromFlightChunks() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-source", "test-tensor");
                Assert.assertEquals(0, server.getTotalChunkRequestCount());

                Assert.assertEquals(4, image.dimension(0));
                Assert.assertEquals(4, image.dimension(1));
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("base-0-0"));
                Assert.assertEquals(1, server.getTotalChunkRequestCount());

                Assert.assertEquals(6.0f, image.getAt(1, 1).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("base-0-0"));
                Assert.assertEquals(1, server.getTotalChunkRequestCount());

                Assert.assertEquals(13.0f, image.getAt(3, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("base-1-0"));
                Assert.assertEquals(2, server.getTotalChunkRequestCount());

                Assert.assertEquals(4.0f, image.getAt(0, 3).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("base-0-1"));
                Assert.assertEquals(3, server.getTotalChunkRequestCount());

                Assert.assertEquals(16.0f, image.getAt(3, 3).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("base-1-1"));
                Assert.assertEquals(4, server.getTotalChunkRequestCount());
            }
        }
    }

    @Test
    public void testScaledReadUsesResponseDescriptor() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                long[] scaleHint = new long[] {2, 2};
                String reductionMethod = "nearest";

                RandomAccessibleInterval<FloatType> scaled = client.getTensor(
                        "test-source", "test-tensor", scaleHint, reductionMethod);
                Assert.assertEquals(0, server.getTotalChunkRequestCount());

                Assert.assertEquals(2, scaled.dimension(0));
                Assert.assertEquals(2, scaled.dimension(1));
                Assert.assertEquals(1.0f, scaled.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-0-0"));
                Assert.assertEquals(1, server.getTotalChunkRequestCount());

                Assert.assertEquals(9.0f, scaled.getAt(1, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-1-0"));
                Assert.assertEquals(2, server.getTotalChunkRequestCount());

                Assert.assertEquals(3.0f, scaled.getAt(0, 1).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-0-1"));
                Assert.assertEquals(3, server.getTotalChunkRequestCount());

                Assert.assertEquals(11.0f, scaled.getAt(1, 1).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-1-1"));
                Assert.assertEquals(4, server.getTotalChunkRequestCount());

                Assert.assertEquals(11.0f, scaled.getAt(1, 1).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-1-1"));
                Assert.assertEquals(4, server.getTotalChunkRequestCount());
                Assert.assertEquals("nearest", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadAcceptsClippedEdgeChunks() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                long[] scaleHint = new long[] {2, 2};
                String reductionMethod = "linear";

                RandomAccessibleInterval<FloatType> scaled = client.getTensor(
                        "test-source", "test-tensor", scaleHint, reductionMethod);
                Assert.assertEquals(0, server.getTotalChunkRequestCount());

                Assert.assertEquals(3, scaled.dimension(0));
                Assert.assertEquals(3, scaled.dimension(1));

                Assert.assertEquals(1.0f, scaled.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-edge-0-0"));
                Assert.assertEquals(1, server.getTotalChunkRequestCount());

                Assert.assertEquals(3.0f, scaled.getAt(0, 2).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-edge-0-1"));
                Assert.assertEquals(2, server.getTotalChunkRequestCount());

                Assert.assertEquals(7.0f, scaled.getAt(2, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-edge-1-0"));
                Assert.assertEquals(3, server.getTotalChunkRequestCount());

                Assert.assertEquals(9.0f, scaled.getAt(2, 2).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-edge-1-1"));
                Assert.assertEquals(4, server.getTotalChunkRequestCount());

                Assert.assertEquals("linear", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledConvenienceDefaultsToNearest() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getTensor(
                        "test-source", "test-tensor", new long[] {2, 2}, null);
                Assert.assertEquals(2, image.dimension(0));
                Assert.assertEquals(2, image.dimension(1));
                Assert.assertEquals("nearest", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadRejectsRankMismatch() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getTensor("test-source", "test-tensor", new long[] {2}, "nearest"));
                Assert.assertTrue(error.getMessage().contains("dimensionality mismatch"));
                Assert.assertNull(server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadRejectsNonPositiveScale() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getTensor("test-source", "test-tensor", new long[] {2, 0}, "nearest"));
                Assert.assertTrue(error.getMessage().contains("must be positive"));
                Assert.assertNull(server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadRejectsUnsupportedMethod() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getTensor("test-source", "test-tensor", new long[] {2, 2}, "median"));
                Assert.assertTrue(error.getMessage().contains("Unsupported reduction method"));
                Assert.assertNull(server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testTensorNotFoundRaises() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getTensor("test-source", "nonexistent"));
                Assert.assertTrue(error.getMessage().contains("not found"));
            }
        }
    }

    @Test
    public void testSourceNotFoundRaises() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getTensor("nonexistent-source", "some-tensor"));
                Assert.assertTrue(error.getMessage().contains("Source not found"));
            }
        }
    }

    @Test
    public void testSerializableTensorImgSerialization() throws Exception {
        // Clear connection pool before test
        TensorConnectionPool.clear();

        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                // Get tensor - should return SerializableTensorImg
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-source", "test-tensor");
                Assert.assertTrue(image instanceof SerializableTensorImg);

                // Verify initial data access
                Assert.assertEquals(4, image.dimension(0));
                Assert.assertEquals(4, image.dimension(1));
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
                int initialRequests = server.getTotalChunkRequestCount();

                // Serialize to bytes
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(baos);
                oos.writeObject(image);
                oos.close();
                byte[] serialized = baos.toByteArray();

                // Deserialize
                java.io.ByteArrayInputStream bais = new java.io.ByteArrayInputStream(serialized);
                java.io.ObjectInputStream ois = new java.io.ObjectInputStream(bais);
                RandomAccessibleInterval<FloatType> deserialized =
                    (RandomAccessibleInterval<FloatType>) ois.readObject();

                // Verify deserialized image dimensions
                Assert.assertEquals(4, deserialized.dimension(0));
                Assert.assertEquals(4, deserialized.dimension(1));

                // Access data - should trigger lazy reconstruction
                Assert.assertEquals(1.0f, deserialized.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(6.0f, deserialized.getAt(1, 1).get(), 0.0001f);

                // Verify connection pool was used
                Assert.assertTrue(TensorConnectionPool.getConnectionCount() > 0);
            }
        }
    }

    @Test
    public void testSerializableTensorImgMultipleDeserialization() throws Exception {
        // Clear connection pool before test
        TensorConnectionPool.clear();

        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-source", "test-tensor");

                // Serialize
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(baos);
                oos.writeObject(image);
                oos.close();
                byte[] serialized = baos.toByteArray();

                // Deserialize multiple times
                for (int i = 0; i < 3; i++) {
                    java.io.ByteArrayInputStream bais = new java.io.ByteArrayInputStream(serialized);
                    java.io.ObjectInputStream ois = new java.io.ObjectInputStream(bais);
                    RandomAccessibleInterval<FloatType> deserialized =
                        (RandomAccessibleInterval<FloatType>) ois.readObject();

                    // Verify each deserialization works
                    Assert.assertEquals(4, deserialized.dimension(0));
                    Assert.assertEquals(1.0f, deserialized.getAt(0, 0).get(), 0.0001f);
                }

                // Connection pool should reuse same connection
                Assert.assertEquals(1, TensorConnectionPool.getConnectionCount());
            }
        }
    }

    @Test
    public void testGetTensorAsPb() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = client.getTensorAsPb("test-source", "test-tensor", null, null, null);

                // Verify descriptor is populated
                Assert.assertEquals("test-tensor", pb.getTensorDescriptor().getArrayId());
                Assert.assertEquals(Arrays.asList(4L, 4L), pb.getTensorDescriptor().getShapeList());
                Assert.assertEquals("float32", pb.getTensorDescriptor().getDtype());
                Assert.assertEquals(Arrays.asList(2L, 2L), pb.getTensorDescriptor().getChunkShapeList());

                // Verify location is populated
                Assert.assertTrue(pb.getLocation().contains("localhost"));

                // Verify endpoints are populated
                Assert.assertEquals(4, pb.getEndpointsCount());
            }
        }
    }

    @Test
    public void testTensorFromPb() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = client.getTensorAsPb("test-source", "test-tensor", null, null, null);

                // Reconstruct array
                RandomAccessibleInterval<FloatType> image = TensorFlightClient.tensorFromPb(pb, 10_000_000L);

                // Verify shape and type
                Assert.assertEquals(4, image.dimension(0));
                Assert.assertEquals(4, image.dimension(1));

                // Verify data values
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(6.0f, image.getAt(1, 1).get(), 0.0001f);
                Assert.assertEquals(16.0f, image.getAt(3, 3).get(), 0.0001f);
            }
        }
    }

    @Test
    public void testTensorPbSerialization() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = client.getTensorAsPb("test-source", "test-tensor", null, null, null);

                // Serialize to bytes
                byte[] serializedBytes = pb.toByteArray();

                // Deserialize
                SerializedTensor pb2 = SerializedTensor.parseFrom(serializedBytes);

                // Reconstruct array from deserialized protobuf
                RandomAccessibleInterval<FloatType> image = TensorFlightClient.tensorFromPb(pb2, 10_000_000L);

                // Verify data is correct
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(16.0f, image.getAt(3, 3).get(), 0.0001f);
            }
        }
    }

    @Test
    public void testGetTensorAsPbWithScaleHint() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                long[] scaleHint = new long[] {2, 2};
                SerializedTensor pb = client.getTensorAsPb("test-source", "test-tensor", null, scaleHint, "nearest");

                // Verify scale_hint in descriptor
                Assert.assertEquals(Arrays.asList(2L, 2L), pb.getTensorDescriptor().getScaleHintList());

                // Reconstruct and verify downscaled shape
                RandomAccessibleInterval<FloatType> image = TensorFlightClient.tensorFromPb(pb, 10_000_000L);
                Assert.assertEquals(2, image.dimension(0));
                Assert.assertEquals(2, image.dimension(1));

                // Verify data values
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(11.0f, image.getAt(1, 1).get(), 0.0001f);
            }
        }
    }

    @Test
    public void testGetUploadStatusFromSerializedTensor() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setUploadStatusSequence("upload-source",
                    status("upload-source", "PENDING", 4, 1));

            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = SerializedTensor.newBuilder()
                        .setTensorDescriptor(TensorDescriptor.newBuilder().setArrayId("upload-source").build())
                        .build();

                Map<String, Object> status = client.getUploadStatus(pb);
                Assert.assertEquals("PENDING", status.get("state"));
                Assert.assertEquals(1.0d, ((Number) status.get("uploaded_chunks")).doubleValue(), 0.0d);
            }
        }
    }

    @Test
    public void testWaitForUploadReadyFromSerializedTensor() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setUploadStatusSequence(
                    "upload-source",
                    status("upload-source", "PENDING", 4, 1),
                    status("upload-source", "READY", 4, 4));

            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = SerializedTensor.newBuilder()
                        .setTensorDescriptor(TensorDescriptor.newBuilder().setArrayId("upload-source").build())
                        .build();

                Map<String, Object> status = client.waitForUploadReady(pb, 100L, 0L);
                Assert.assertEquals("READY", status.get("state"));
                Assert.assertEquals(4.0d, ((Number) status.get("uploaded_chunks")).doubleValue(), 0.0d);
            }
        }
    }

    @Test
    public void testWaitForUploadReadyTimesOut() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setUploadStatusSequence("upload-source",
                    status("upload-source", "PENDING", 4, 1));

            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IOException error = Assert.assertThrows(
                        IOException.class,
                        () -> client.waitForUploadReady("upload-source", 0L, 0L));
                Assert.assertTrue(error.getMessage().contains("Timed out waiting for upload readiness"));
            }
        }
    }

    @Test
    public void testGetUploadStatusRequiresArrayId() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = SerializedTensor.newBuilder()
                        .setTensorDescriptor(TensorDescriptor.newBuilder().build())
                        .build();

                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getUploadStatus(pb));
                Assert.assertTrue(error.getMessage().contains("tensor_descriptor.array_id is required"));
            }
        }
    }

    @Test
    public void testWaitForUploadReadyRaisesOnFailedState() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setUploadStatusSequence("upload-source",
                    status("upload-source", "FAILED", 4, 2));

            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IOException error = Assert.assertThrows(
                        IOException.class,
                        () -> client.waitForUploadReady("upload-source", 100L, 0L));
                Assert.assertTrue(error.getMessage().contains("Upload failed for source 'upload-source'"));
            }
        }
    }

    private static Map<String, Object> status(String sourceId, String state, int expectedChunks, int uploadedChunks) {
        Map<String, Object> status = new HashMap<>();
        status.put("source_id", sourceId);
        status.put("state", state);
        status.put("expected_chunks", expectedChunks);
        status.put("uploaded_chunks", uploadedChunks);
        return status;
    }

    private static class TestFlightServer implements AutoCloseable {
        private final BufferAllocator allocator;
        private final FlightServer server;
        private final TensorTestProducer producer;

        TestFlightServer() throws IOException {
            this.allocator = new RootAllocator(Long.MAX_VALUE);
            this.producer = new TensorTestProducer(allocator);
            this.server = FlightServer.builder(
                    allocator,
                    Location.forGrpcInsecure("localhost", 0),
                    producer)
                    .build()
                    .start();
        }

        int getPort() {
            return server.getPort();
        }

        int getChunkRequestCount(String chunkId) {
            return producer.getChunkRequestCount(chunkId);
        }

        int getTotalChunkRequestCount() {
            return producer.getTotalChunkRequestCount();
        }

        String getLastReductionMethod() {
            return producer.getLastReductionMethod();
        }

        void setUploadStatusSequence(String sourceId, Map<String, Object>... statuses) {
            producer.setUploadStatusSequence(sourceId, Arrays.asList(statuses));
        }

        @Override
        public void close() throws Exception {
            server.close();
            allocator.close();
        }
    }

    private static class TensorTestProducer extends NoOpFlightProducer {
        private final BufferAllocator allocator;
        private final TensorDescriptor baseDescriptor;
        private final org.apache.arrow.vector.types.pojo.Schema schema;
        private final Map<String, float[]> chunkData;
        private final Map<String, AtomicInteger> chunkRequests;
        private final Map<String, List<Map<String, Object>>> uploadStatusSequences;
        private final Map<String, AtomicInteger> uploadStatusCalls;
        private volatile FlightRequest lastCmd;

        TensorTestProducer(BufferAllocator allocator) {
            this.allocator = allocator;

            // Base tensor descriptor
            this.baseDescriptor = TensorDescriptor.newBuilder()
                    .setArrayId("test-tensor")
                    .addDimLabels("y")
                    .addDimLabels("x")
                    .addShape(4)
                    .addShape(4)
                    .addChunkShape(2)
                    .addChunkShape(2)
                    .setDtype("float32")
                    .build();

            this.schema = createSchema(allocator);
            this.chunkData = new HashMap<>();
            this.chunkRequests = new ConcurrentHashMap<>();
            this.uploadStatusSequences = new ConcurrentHashMap<>();
            this.uploadStatusCalls = new ConcurrentHashMap<>();
            chunkData.put("base-0-0", new float[] {1, 2, 5, 6});
            chunkData.put("base-0-1", new float[] {3, 4, 7, 8});
            chunkData.put("base-1-0", new float[] {9, 10, 13, 14});
            chunkData.put("base-1-1", new float[] {11, 12, 15, 16});
            chunkData.put("scaled-0-0", new float[] {1});
            chunkData.put("scaled-0-1", new float[] {3});
            chunkData.put("scaled-1-0", new float[] {9});
            chunkData.put("scaled-1-1", new float[] {11});
            chunkData.put("scaled-edge-0-0", new float[] {1, 2, 4, 5});
            chunkData.put("scaled-edge-0-1", new float[] {3, 6});
            chunkData.put("scaled-edge-1-0", new float[] {7, 8});
            chunkData.put("scaled-edge-1-1", new float[] {9});
        }

        void setUploadStatusSequence(String sourceId, List<Map<String, Object>> statuses) {
            uploadStatusSequences.put(sourceId, statuses);
            uploadStatusCalls.put(sourceId, new AtomicInteger());
        }

        int getChunkRequestCount(String chunkId) {
            AtomicInteger count = chunkRequests.get(chunkId);
            return count == null ? 0 : count.get();
        }

        int getTotalChunkRequestCount() {
            int total = 0;
            for (AtomicInteger count : chunkRequests.values()) {
                total += count.get();
            }
            return total;
        }

        String getLastReductionMethod() {
            if (lastCmd == null || !lastCmd.hasTensorRead()) {
                return null;
            }
            return lastCmd.getTensorRead().getReductionMethod();
        }

        @Override
        public FlightInfo getFlightInfo(FlightProducer.CallContext context, FlightDescriptor descriptor) {
            FlightRequest cmd = parseCmd(descriptor.getCommand());
            lastCmd = cmd;
            TensorReadOption readOpt = cmd.hasTensorRead()
                    ? cmd.getTensorRead()
                    : null;

            // Validate the tensor: "test-tensor" is the sole tensor of "test-source".
            if (readOpt == null
                    || !(readOpt.getArrayId().equals("test-tensor")
                            || readOpt.getArrayId().equals("test-source"))) {
                throw new IllegalArgumentException("Tensor not found: " + (readOpt != null ? readOpt.getArrayId() : "null"));
            }

            // Handle scaled reads
            if (readOpt != null
                    && readOpt.getScaleHintCount() == 2
                    && readOpt.getScaleHint(0) == 2
                    && readOpt.getScaleHint(1) == 2
                    && "linear".equals(readOpt.getReductionMethod())) {

                TensorDescriptor responseDescriptor = TensorDescriptor.newBuilder(baseDescriptor)
                        .clearShape()
                        .clearChunkShape()
                        .addShape(3)
                        .addShape(3)
                        .addChunkShape(2)
                        .addChunkShape(2)
                        .clearScaleHint()
                        .addAllScaleHint(readOpt.getScaleHintList())
                        .setReductionMethod(readOpt.getReductionMethod())
                        .build();
                return new FlightInfo(
                        schema,
                        FlightDescriptor.command(responseDescriptor.toByteArray()),
                        scaledEdgeEndpoints(),
                        -1,
                        -1);
            }

            if (readOpt != null
                    && readOpt.getScaleHintCount() == 2
                    && readOpt.getScaleHint(0) == 2
                    && readOpt.getScaleHint(1) == 2) {

                TensorDescriptor responseDescriptor = TensorDescriptor.newBuilder(baseDescriptor)
                        .clearShape()
                        .clearChunkShape()
                        .addShape(2)
                        .addShape(2)
                        .addChunkShape(1)
                        .addChunkShape(1)
                        .clearScaleHint()
                        .addAllScaleHint(readOpt.getScaleHintList())
                        .setReductionMethod(readOpt.getReductionMethod())
                        .build();
                return new FlightInfo(
                        schema,
                        FlightDescriptor.command(responseDescriptor.toByteArray()),
                        scaledEndpoints(),
                        -1,
                        -1);
            }

            return new FlightInfo(
                    schema,
                    FlightDescriptor.command(baseDescriptor.toByteArray()),
                    baseEndpoints(),
                    -1,
                    -1);
        }

        @Override
        public void doAction(
                FlightProducer.CallContext context,
                Action action,
                FlightProducer.StreamListener<Result> listener) {
            if (!"upload_status".equals(action.getType())) {
                listener.onError(new IllegalArgumentException("Unknown action: " + action.getType()));
                return;
            }

            String sourceId = new String(action.getBody(), StandardCharsets.UTF_8);
            List<Map<String, Object>> sequence = uploadStatusSequences.get(sourceId);
            if (sequence == null || sequence.isEmpty()) {
                sequence = Collections.singletonList(status(sourceId, "UNKNOWN", 0, 0));
            }

            AtomicInteger calls = uploadStatusCalls.computeIfAbsent(sourceId, ignored -> new AtomicInteger());
            int index = Math.min(calls.getAndIncrement(), sequence.size() - 1);
            String json = new Gson().toJson(sequence.get(index));
            listener.onNext(new Result(json.getBytes(StandardCharsets.UTF_8)));
            listener.onCompleted();
        }

        @Override
        public void getStream(
                FlightProducer.CallContext context,
                Ticket ticket,
                FlightProducer.ServerStreamListener listener) {

            TensorTicket tensorTicket = parseTicket(ticket.getBytes());
            if (tensorTicket.hasCatalogQuery()) {
                // The `catalog` flight: the `sources` row(s) the query selects, as
                // the server's DuckDB would stream them (tensors is a LIST<STRUCT>).
                // The WHERE clause is honoured because the client now addresses
                // single rows with one -- a fake that answered every id with its
                // one row would report a missing source as present.
                String sql = tensorTicket.getCatalogQuery().getSql();
                boolean matches = !sql.contains("WHERE source_id = ")
                        || sql.contains("'test-source'");
                try (VectorSchemaRoot root = catalogRoot(matches ? 1 : 0)) {
                    listener.start(root);
                    listener.putNext();
                    listener.completed();
                }
                return;
            }
            String chunkIdStr = tensorTicket.getChunkId().toString(StandardCharsets.UTF_8);
            chunkRequests.computeIfAbsent(chunkIdStr, ignored -> new AtomicInteger()).incrementAndGet();
            float[] values = chunkData.get(chunkIdStr);
            if (values == null) {
                listener.error(new IllegalArgumentException("Unknown chunk: " + chunkIdStr));
                return;
            }

            // Unified binary chunk schema (biopb/biopb#293): data (binary) is the
            // raw little-endian float32 bytes; dtype (utf8) says how to read them.
            byte[] raw = new byte[values.length * 4];
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (float v : values) {
                bb.putFloat(v);
            }

            org.apache.arrow.vector.VarBinaryVector dataVector =
                    new org.apache.arrow.vector.VarBinaryVector("data", allocator);
            dataVector.allocateNew();
            dataVector.setSafe(0, raw);
            dataVector.setValueCount(1);

            org.apache.arrow.vector.VarCharVector dtypeVector =
                    new org.apache.arrow.vector.VarCharVector("dtype", allocator);
            dtypeVector.allocateNew();
            dtypeVector.setSafe(0, "<f4".getBytes(StandardCharsets.UTF_8));
            dtypeVector.setValueCount(1);

            try (VectorSchemaRoot root = VectorSchemaRoot.of(dataVector, dtypeVector)) {
                root.setRowCount(1);
                listener.start(root);
                listener.putNext();
                listener.completed();
            }
        }

        private VectorSchemaRoot catalogRoot(int rows) {
            Field arrayId = new Field("array_id", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
            Field dimLabels = new Field("dim_labels", FieldType.nullable(ArrowType.List.INSTANCE),
                    Collections.singletonList(new Field("item", FieldType.nullable(ArrowType.Utf8.INSTANCE), null)));
            Field shape = new Field("shape", FieldType.nullable(ArrowType.List.INSTANCE),
                    Collections.singletonList(new Field("item", FieldType.nullable(new ArrowType.Int(64, true)), null)));
            Field dtype = new Field("dtype", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
            Field tensorStruct = new Field("item", FieldType.nullable(ArrowType.Struct.INSTANCE),
                    Arrays.asList(arrayId, dimLabels, shape, dtype));
            Schema catalogSchema = new Schema(Arrays.asList(
                    new Field("source_id", FieldType.nullable(ArrowType.Utf8.INSTANCE), null),
                    new Field("source_url", FieldType.nullable(ArrowType.Utf8.INSTANCE), null),
                    new Field("source_type", FieldType.nullable(ArrowType.Utf8.INSTANCE), null),
                    new Field("data_resident", FieldType.nullable(ArrowType.Bool.INSTANCE), null),
                    new Field("is_resolved", FieldType.nullable(ArrowType.Bool.INSTANCE), null),
                    new Field("tensors", FieldType.nullable(ArrowType.List.INSTANCE),
                            Collections.singletonList(tensorStruct))));
            VectorSchemaRoot root = VectorSchemaRoot.create(catalogSchema, allocator);
            root.allocateNew();
            if (rows == 0) {
                root.setRowCount(0);
                return root;
            }
            ((org.apache.arrow.vector.VarCharVector) root.getVector("source_id"))
                    .setSafe(0, "test-source".getBytes(StandardCharsets.UTF_8));
            ((org.apache.arrow.vector.VarCharVector) root.getVector("source_url"))
                    .setSafe(0, "mock://test".getBytes(StandardCharsets.UTF_8));
            ((org.apache.arrow.vector.VarCharVector) root.getVector("source_type"))
                    .setSafe(0, "mock".getBytes(StandardCharsets.UTF_8));
            ((org.apache.arrow.vector.BitVector) root.getVector("data_resident")).setSafe(0, 1);
            ((org.apache.arrow.vector.BitVector) root.getVector("is_resolved")).setSafe(0, 1);
            ListVector tensors = (ListVector) root.getVector("tensors");
            UnionListWriter writer = tensors.getWriter();
            writer.setPosition(0);
            writer.startList();
            org.apache.arrow.vector.complex.writer.BaseWriter.StructWriter sw = writer.struct();
            sw.start();
            sw.varChar("array_id").writeVarChar("test-tensor");
            org.apache.arrow.vector.complex.writer.BaseWriter.ListWriter labels = sw.list("dim_labels");
            labels.startList();
            labels.varChar().writeVarChar("y");
            labels.varChar().writeVarChar("x");
            labels.endList();
            org.apache.arrow.vector.complex.writer.BaseWriter.ListWriter shapeW = sw.list("shape");
            shapeW.startList();
            shapeW.bigInt().writeBigInt(4);
            shapeW.bigInt().writeBigInt(4);
            shapeW.endList();
            sw.varChar("dtype").writeVarChar("float32");
            sw.end();
            writer.endList();
            tensors.setValueCount(1);
            root.setRowCount(1);
            return root;
        }

        private List<FlightEndpoint> baseEndpoints() {
            List<FlightEndpoint> endpoints = new ArrayList<>();
            endpoints.add(endpoint("base-0-0", 0, 0, 2, 2));
            endpoints.add(endpoint("base-0-1", 0, 2, 2, 4));
            endpoints.add(endpoint("base-1-0", 2, 0, 4, 2));
            endpoints.add(endpoint("base-1-1", 2, 2, 4, 4));
            return endpoints;
        }

        private List<FlightEndpoint> scaledEndpoints() {
            List<FlightEndpoint> endpoints = new ArrayList<>();
            endpoints.add(endpoint("scaled-0-0", 0, 0, 1, 1));
            endpoints.add(endpoint("scaled-0-1", 0, 1, 1, 2));
            endpoints.add(endpoint("scaled-1-0", 1, 0, 2, 1));
            endpoints.add(endpoint("scaled-1-1", 1, 1, 2, 2));
            return endpoints;
        }

        private List<FlightEndpoint> scaledEdgeEndpoints() {
            List<FlightEndpoint> endpoints = new ArrayList<>();
            endpoints.add(endpoint("scaled-edge-0-0", 0, 0, 2, 2));
            endpoints.add(endpoint("scaled-edge-0-1", 0, 2, 2, 3));
            endpoints.add(endpoint("scaled-edge-1-0", 2, 0, 3, 2));
            endpoints.add(endpoint("scaled-edge-1-1", 2, 2, 3, 3));
            return endpoints;
        }

        private FlightEndpoint endpoint(String chunkId, long start0, long start1, long stop0, long stop1) {
            TensorTicket ticket = TensorTicket.newBuilder()
                    .setChunkId(ByteString.copyFromUtf8(chunkId))
                    .build();
            ChunkBounds bounds = ChunkBounds.newBuilder()
                    .addStart(start0)
                    .addStart(start1)
                    .addStop(stop0)
                    .addStop(stop1)
                    .build();
            return FlightEndpoint.builder(new Ticket(ticket.toByteArray()))
                    .setAppMetadata(bounds.toByteArray())
                    .build();
        }

        private static org.apache.arrow.vector.types.pojo.Schema createSchema(BufferAllocator allocator) {
            // Unified binary chunk schema (biopb/biopb#293): data (binary), dtype (utf8).
            Field dataField = new Field("data", FieldType.nullable(ArrowType.Binary.INSTANCE), null);
            Field dtypeField = new Field("dtype", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
            return new org.apache.arrow.vector.types.pojo.Schema(Arrays.asList(dataField, dtypeField));
        }

        private static FlightRequest parseCmd(byte[] bytes) {
            try {
                return FlightRequest.parseFrom(bytes);
            } catch (IOException e) {
                throw new IllegalStateException("Failed to parse FlightRequest", e);
            }
        }

        private static TensorTicket parseTicket(byte[] bytes) {
            try {
                return TensorTicket.parseFrom(bytes);
            } catch (IOException e) {
                throw new IllegalStateException("Failed to parse TensorTicket", e);
            }
        }
    }
}
