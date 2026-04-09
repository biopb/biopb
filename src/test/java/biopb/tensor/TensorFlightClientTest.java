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

import org.apache.arrow.flight.Criteria;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightProducer;
import org.apache.arrow.flight.FlightServer;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.NoOpFlightProducer;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.Assert;
import org.junit.Test;

import com.google.protobuf.ByteString;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;

public class TensorFlightClientTest {

    @Test
    public void testListTensorsAndDescriptorLookup() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                List<String> tensorIds = client.listTensors();
                Assert.assertEquals(Collections.singletonList("test-array"), tensorIds);

                TensorDescriptor descriptor = client.getDescriptor("test-array");
                Assert.assertEquals(Arrays.asList(4L, 4L), descriptor.getShapeList());
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getChunkShapeList());
            }
        }
    }

    @Test
    public void testMaterializesBaseArrayFromFlightChunks() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getArray("test-array");
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
                TensorReadOptions readOptions = TensorReadOptions.newBuilder()
                        .addScaleHint(2)
                        .addScaleHint(2)
                        .setReductionMethod("nearest")
                        .build();

                TensorDescriptor descriptor = client.getDescriptor("test-array", readOptions);
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getShapeList());
                Assert.assertEquals(Arrays.asList(1L, 1L), descriptor.getChunkShapeList());
                Assert.assertTrue(descriptor.hasReadOptions());

                RandomAccessibleInterval<FloatType> scaled = client.getArray("test-array", readOptions);
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
                TensorReadOptions readOptions = TensorReadOptions.newBuilder()
                        .addScaleHint(2)
                        .addScaleHint(2)
                        .setReductionMethod("linear")
                        .build();

                TensorDescriptor descriptor = client.getDescriptor("test-array", readOptions);
                Assert.assertEquals(Arrays.asList(3L, 3L), descriptor.getShapeList());
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getChunkShapeList());

                RandomAccessibleInterval<FloatType> scaled = client.getArray("test-array", readOptions);
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

                Assert.assertEquals(9.0f, scaled.getAt(2, 2).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("scaled-edge-1-1"));
                Assert.assertEquals(4, server.getTotalChunkRequestCount());
                Assert.assertEquals("linear", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledDescriptorConvenienceNormalizesAlias() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorDescriptor descriptor = client.getDescriptor("test-array", new long[] {2, 2}, "stride");
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getShapeList());
                Assert.assertEquals(Arrays.asList(1L, 1L), descriptor.getChunkShapeList());
                Assert.assertEquals("nearest", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadOptionsNormalizesMeanAlias() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorReadOptions readOptions = TensorReadOptions.newBuilder()
                        .addScaleHint(2)
                        .addScaleHint(2)
                        .setReductionMethod("mean")
                        .build();

                TensorDescriptor descriptor = client.getDescriptor("test-array", readOptions);
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getShapeList());
                Assert.assertEquals("area", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledConvenienceDefaultsToNearest() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getArray("test-array", new long[] {2, 2}, null);
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
                        () -> client.getArray("test-array", new long[] {2}, "nearest"));
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
                        () -> client.getArray("test-array", new long[] {2, 0}, "nearest"));
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
                        () -> client.getArray("test-array", new long[] {2, 2}, "median"));
                Assert.assertTrue(error.getMessage().contains("Unsupported reduction method"));
                Assert.assertNull(server.getLastReductionMethod());
            }
        }
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
        private volatile TensorDescriptor lastRequestDescriptor;

        TensorTestProducer(BufferAllocator allocator) {
            this.allocator = allocator;
            this.baseDescriptor = TensorDescriptor.newBuilder()
                    .setArrayId("test-array")
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
            if (lastRequestDescriptor == null || !lastRequestDescriptor.hasReadOptions()) {
                return null;
            }
            return lastRequestDescriptor.getReadOptions().getReductionMethod();
        }

        @Override
        public void listFlights(
                FlightProducer.CallContext context,
                Criteria criteria,
                FlightProducer.StreamListener<FlightInfo> listener) {

            listener.onNext(new FlightInfo(
                    schema,
                    FlightDescriptor.command(baseDescriptor.toByteArray()),
                    Collections.singletonList(new FlightEndpoint(new Ticket(new byte[0]))),
                    -1,
                    -1));
            listener.onCompleted();
        }

        @Override
        public FlightInfo getFlightInfo(FlightProducer.CallContext context, FlightDescriptor descriptor) {
            TensorDescriptor requestDescriptor = parseDescriptor(descriptor.getCommand());
            lastRequestDescriptor = requestDescriptor;
            TensorReadOptions readOptions = requestDescriptor.hasReadOptions()
                    ? requestDescriptor.getReadOptions()
                    : null;

                if (readOptions != null
                    && readOptions.getScaleHintCount() == 2
                    && readOptions.getScaleHint(0) == 2
                    && readOptions.getScaleHint(1) == 2
                    && "linear".equals(readOptions.getReductionMethod())) {
                TensorDescriptor responseDescriptor = TensorDescriptor.newBuilder(baseDescriptor)
                    .clearShape()
                    .clearChunkShape()
                    .addShape(3)
                    .addShape(3)
                    .addChunkShape(2)
                    .addChunkShape(2)
                    .setReadOptions(readOptions)
                    .build();
                return new FlightInfo(
                    schema,
                    FlightDescriptor.command(responseDescriptor.toByteArray()),
                    scaledEdgeEndpoints(),
                    -1,
                    -1);
                }

                if (readOptions != null
                    && readOptions.getScaleHintCount() == 2
                    && readOptions.getScaleHint(0) == 2
                    && readOptions.getScaleHint(1) == 2) {
                TensorDescriptor responseDescriptor = TensorDescriptor.newBuilder(baseDescriptor)
                        .clearShape()
                        .clearChunkShape()
                        .addShape(2)
                        .addShape(2)
                        .addChunkShape(1)
                        .addChunkShape(1)
                        .setReadOptions(readOptions)
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
        public void getStream(
                FlightProducer.CallContext context,
                Ticket ticket,
                FlightProducer.ServerStreamListener listener) {

            TensorTicket tensorTicket = parseTicket(ticket.getBytes());
            String chunkId = tensorTicket.getChunkId().toString(StandardCharsets.UTF_8);
            chunkRequests.computeIfAbsent(chunkId, ignored -> new AtomicInteger()).incrementAndGet();
            float[] values = chunkData.get(chunkId);
            if (values == null) {
                listener.error(new IllegalArgumentException("Unknown chunk: " + chunkId));
                return;
            }

            try (Float4Vector vector = new Float4Vector("data", allocator)) {
                vector.allocateNew(values.length);
                for (int i = 0; i < values.length; i++) {
                    vector.setSafe(i, values[i]);
                }
                vector.setValueCount(values.length);

                try (VectorSchemaRoot root = VectorSchemaRoot.of(vector)) {
                    root.setRowCount(values.length);
                    listener.start(root);
                    listener.putNext();
                    listener.completed();
                }
            }
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
            try (Float4Vector vector = new Float4Vector("data", allocator);
                    VectorSchemaRoot root = VectorSchemaRoot.of(vector)) {
                return root.getSchema();
            }
        }

        private static TensorDescriptor parseDescriptor(byte[] bytes) {
            try {
                return TensorDescriptor.parseFrom(bytes);
            } catch (IOException e) {
                throw new IllegalStateException("Failed to parse TensorDescriptor", e);
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