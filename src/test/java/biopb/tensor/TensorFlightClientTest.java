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
import org.apache.arrow.flight.CallStatus;
import org.apache.arrow.flight.ErrorFlightMetadata;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightProducer;
import org.apache.arrow.flight.FlightServer;
import org.apache.arrow.flight.FlightStatusCode;
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
import com.google.gson.reflect.TypeToken;
import com.google.protobuf.ByteString;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;

public class TensorFlightClientTest {

    @Test
    public void testCatalogRowCarriesIsResolvedForCallersToRead() throws Exception {
        // What a caller actually gets: a row. `is_resolved` is a column on it,
        // read without any type this SDK picked -- which is the whole point of
        // biopb/biopb#1032, and what listSources() cannot give you because
        // DataSourceDescriptor has no field for it.
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                try (VectorSchemaRoot root = client.querySources(
                        "SELECT " + TensorFlightClient.SOURCE_ROW_COLUMNS + " FROM sources")) {
                    Assert.assertEquals(1, root.getRowCount());
                    Assert.assertEquals("test-source",
                            String.valueOf(root.getVector("source_id").getObject(0)));
                    Assert.assertEquals(Boolean.TRUE,
                            root.getVector("is_resolved").getObject(0));

                    Object tensors = root.getVector("tensors").getObject(0);
                    Assert.assertTrue(tensors instanceof List);
                    Map<?, ?> tensor = (Map<?, ?>) ((List<?>) tensors).get(0);
                    Assert.assertEquals("test-tensor",
                            String.valueOf(tensor.get("array_id")));
                    // Every tensor of the source is enumerated on the row, which
                    // is the browse surface -- there is no second decode of it
                    // into a message that has no field for `is_resolved`.
                    Assert.assertEquals(Arrays.asList(4L, 4L),
                            ((List<?>) tensor.get("shape")).stream()
                                    .map(dim -> ((Number) dim).longValue())
                                    .collect(java.util.stream.Collectors.toList()));
                }
            }
        }
    }

    @Test
    public void testMaterializesBaseArrayFromFlightChunks() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-tensor");
                // The lazy adapter consumes the FlightInfo returned by planning;
                // accessing cells must not initiate a replacement read plan.
                Assert.assertEquals(1, server.getFlightInfoRequestCount());
                Assert.assertEquals(0, server.getTotalChunkRequestCount());

                Assert.assertEquals(4, image.dimension(0));
                Assert.assertEquals(4, image.dimension(1));
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
                Assert.assertEquals(1, server.getChunkRequestCount("base-0-0"));
                Assert.assertEquals(1, server.getTotalChunkRequestCount());
                Assert.assertEquals(1, server.getFlightInfoRequestCount());

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
                        "test-tensor", scaleHint, reductionMethod);
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
                        "test-tensor", scaleHint, reductionMethod);
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
                        "test-tensor", new long[] {2, 2}, null);
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
                InvalidTensorRequestException error = Assert.assertThrows(
                        InvalidTensorRequestException.class,
                        () -> client.getTensor("test-tensor", new long[] {2}, "nearest"));
                Assert.assertEquals("scale_rank", error.getReason());
                Assert.assertEquals("nearest", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadRejectsNonPositiveScale() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                InvalidTensorRequestException error = Assert.assertThrows(
                        InvalidTensorRequestException.class,
                        () -> client.getTensor("test-tensor", new long[] {2, 0}, "nearest"));
                Assert.assertEquals("scale_not_positive", error.getReason());
                Assert.assertEquals("nearest", server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testScaledReadRejectsUnsupportedMethod() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getTensor("test-tensor", new long[] {2, 2}, "median"));
                Assert.assertTrue(error.getMessage().contains("Unsupported reduction method"));
                Assert.assertNull(server.getLastReductionMethod());
            }
        }
    }

    @Test
    public void testTensorNotFoundRaises() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorNotFoundException error = Assert.assertThrows(
                        TensorNotFoundException.class,
                        () -> client.getTensor("nonexistent"));
                Assert.assertTrue(error.getMessage().contains("not found"));
            }
        }
    }

    @Test
    public void testSourceNotFoundRaises() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorNotFoundException error = Assert.assertThrows(
                        TensorNotFoundException.class,
                        () -> client.getTensor("nonexistent-source/some-tensor"));
                Assert.assertEquals("unknown_field", error.getReason());
            }
        }
    }

    @Test
    public void testSerializableTensorImgSerialization() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                // Get tensor - should return SerializableTensorImg
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-tensor");
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

                // The serialized FlightInfo is consumed directly; it does not
                // issue another read-planning RPC after deserialization.
                Assert.assertEquals(1, server.getFlightInfoRequestCount());
            }
        }
    }

    @Test
    public void testSerializableTensorImgMultipleDeserialization() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-tensor");

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

                // Each reconstruction reads the embedded plan; none replans.
                Assert.assertEquals(1, server.getFlightInfoRequestCount());
            }
        }
    }

    @Test
    public void testGetTensorAsPb() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = client.getTensorAsPb("test-tensor", null, null, null);

                // The plan is the FlightInfo the server answered, carried whole.
                TensorDescriptor descriptor = TensorFlightClient.descriptorOf(pb);
                Assert.assertEquals("test-tensor", descriptor.getArrayId());
                Assert.assertEquals(Arrays.asList(4L, 4L), descriptor.getShapeList());
                Assert.assertEquals("float32", descriptor.getDtype());
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getChunkShapeList());
                Assert.assertEquals(4, TensorFlightClient.flightInfoOf(pb).getEndpoints().size());

                // Verify location is populated
                Assert.assertTrue(pb.getLocation().contains("localhost"));
            }
        }
    }

    @Test
    public void testTensorFromPb() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = client.getTensorAsPb("test-tensor", null, null, null);

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
                SerializedTensor pb = client.getTensorAsPb("test-tensor", null, null, null);

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
                SerializedTensor pb = client.getTensorAsPb("test-tensor", null, scaleHint, "nearest");

                // Verify scale_hint in the plan's descriptor
                TensorDescriptor descriptor = TensorFlightClient.descriptorOf(pb);
                Assert.assertEquals(Arrays.asList(2L, 2L), descriptor.getScaleHintList());

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

    // ---- protocol gates ---------------------------------------------------

    @Test
    public void testRefusesAServerSpeakingAnotherFlightShape() throws Exception {
        // v1 routed by a sentinel source_id in a FlightCmd; sending it a v2
        // FlightRequest gets it parsed as something else. Name the mismatch
        // instead, once per connection, before the first real call.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setProtocolVersion(1);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                UnsupportedOperationException error = Assert.assertThrows(
                        UnsupportedOperationException.class,
                        () -> client.getTensor("test-tensor"));
                Assert.assertTrue(error.getMessage(),
                        error.getMessage().contains("server speaks v1"));
                Assert.assertTrue(error.getMessage(), error.getMessage().contains("Upgrade the server"));
            }
        }
    }

    @Test
    public void testAcceptsAMatchingFlightShapeAndProbesOnlyOnce() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                client.getTensor("test-tensor");
                client.getTensor("test-tensor");
                // The probe is cached: a connection is checked once, not per call.
                Assert.assertEquals(1, server.getHealthRequestCount());
            }
        }
    }

    @Test
    public void testACapabilityTokenIsNotRefusedForFailingTheProbe() throws Exception {
        // health is on the catalog tier, which a per-source capability cannot
        // reach. Refusing it here would lock the narrowest credential out of the
        // SDK entirely; the private call it is about to make authorizes itself.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setHealthUnauthenticated(true);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<FloatType> image = client.getTensor("test-tensor");
                Assert.assertEquals(4, image.dimension(0));
                Assert.assertEquals(1.0f, image.getAt(0, 0).get(), 0.0001f);
            }
        }
    }

    @Test
    public void testRefusesAChunkEncodingItCannotDecode() throws Exception {
        // An unstamped schema is a pre-#293 server: chunks are a typed
        // data: list<T>, which this client reads as "not binary" from inside a
        // cell load. Refuse at the plan, with the reason.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setChunkWireProtocol(null);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                UnsupportedOperationException error = Assert.assertThrows(
                        UnsupportedOperationException.class,
                        () -> client.getTensor("test-tensor"));
                Assert.assertTrue(error.getMessage(), error.getMessage().contains("server speaks v1"));
                Assert.assertTrue(error.getMessage(), error.getMessage().contains("#293"));
                // Refused before any chunk was fetched.
                Assert.assertEquals(0, server.getTotalChunkRequestCount());
            }
        }
    }

    @Test
    public void testRefusesAChunkEncodingFromTheFuture() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setChunkWireProtocol("3");
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                UnsupportedOperationException error = Assert.assertThrows(
                        UnsupportedOperationException.class,
                        () -> client.getTensor("test-tensor"));
                Assert.assertTrue(error.getMessage(), error.getMessage().contains("Upgrade the client"));
            }
        }
    }

    @Test
    public void testAnUnstampedPlanIsRefusedWhereverItCameFrom() throws Exception {
        // A handle that arrived from another process is reconstructed by the
        // same factory, so it meets the same gate -- the reason the check sits
        // where a plan becomes an image rather than at GetFlightInfo.
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SerializedTensor pb = client.getTensorAsPb("test-tensor");
                SerializedTensor unstamped = SerializedTensor.newBuilder(pb)
                        .setFlightInfo(ByteString.copyFrom(new FlightInfo(
                                new org.apache.arrow.vector.types.pojo.Schema(new ArrayList<>()),
                                TensorFlightClient.flightInfoOf(pb).getDescriptor(),
                                TensorFlightClient.flightInfoOf(pb).getEndpoints(),
                                -1, -1).serialize()))
                        .build();
                RandomAccessibleInterval<FloatType> image =
                        TensorFlightClient.tensorFromPb(unstamped, 10_000_000L);
                Assert.assertThrows(UnsupportedOperationException.class, () -> image.dimension(0));
            }
        }
    }

    // ---- cloud path: resolve / warm / getSourceMetadata -------------------
    // These run against a `doAction` fake, which is what the suite lacked: the
    // three calls that drive an unresolved (cloud / synced-folder) source were
    // compiled but never executed here.

    @Test
    public void testResolveReturnsTheCatalogRow() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setSourceResolved(false);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                // A row, not a struct this SDK picked: the caller decodes it the
                // same way it decodes a querySources result (biopb/biopb#1032).
                try (VectorSchemaRoot row = client.resolve("test-source")) {
                    Assert.assertEquals(1, row.getRowCount());
                    Assert.assertEquals("test-source",
                            String.valueOf(row.getVector("source_id").getObject(0)));
                    // Resolution is what flipped it; the terminal row says so.
                    Assert.assertEquals(Boolean.TRUE, row.getVector("is_resolved").getObject(0));

                    Object tensors = row.getVector("tensors").getObject(0);
                    Map<?, ?> tensor = (Map<?, ?>) ((List<?>) tensors).get(0);
                    Assert.assertEquals("test-tensor", String.valueOf(tensor.get("array_id")));
                }
            }
        }
    }

    @Test
    public void testCatalogRowsCarryNoResidency() throws Exception {
        // Residency is asked per call, never stored: the column is gone from
        // SOURCE_ROW_COLUMNS, so a browse cannot report a stale one
        // (biopb/biopb#1035).
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                try (VectorSchemaRoot root = client.querySources(
                        "SELECT " + TensorFlightClient.SOURCE_ROW_COLUMNS + " FROM sources")) {
                    Assert.assertNull(root.getVector("data_resident"));
                }
            }
        }
    }

    @Test
    public void testResolveOutlivesTheStreamItArrivedOn() throws Exception {
        // The row rides an ArrowStreamReader that frees its buffers on close, so
        // resolve() has to hand back a copy. Reading after the call is what would
        // catch a returned view into freed memory.
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                VectorSchemaRoot row = client.resolve("test-source");
                try {
                    Assert.assertEquals("test-source",
                            String.valueOf(row.getVector("source_id").getObject(0)));
                    Assert.assertEquals("mock://test",
                            String.valueOf(row.getVector("source_url").getObject(0)));
                } finally {
                    row.close();
                }
            }
        }
    }

    @Test
    public void testResolveSkipsProgressHeartbeats() throws Exception {
        // Heartbeats keep the connection warm under proxy idle timeouts; only the
        // terminal message carries the row.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setResolveHeartbeats(3);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                try (VectorSchemaRoot row = client.resolve("test-source")) {
                    Assert.assertEquals(1, row.getRowCount());
                }
            }
        }
    }

    @Test
    public void testResolveReportsProgressAndHonorsCancellation() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setResolveHeartbeats(3);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                List<ResolveProgress> progress = new ArrayList<>();
                try (VectorSchemaRoot row = client.resolve("test-source", progress::add, () -> false)) {
                    Assert.assertEquals(1, row.getRowCount());
                }
                Assert.assertEquals(3, progress.size());
                Assert.assertEquals("img.tif", progress.get(0).getTargetName());

                TensorOperationCancelledException error = Assert.assertThrows(
                        TensorOperationCancelledException.class,
                        () -> client.resolve("test-source", ignored -> Assert.fail("must not report after cancellation"),
                                () -> true));
                Assert.assertEquals("resolve", error.getOperation());
                Assert.assertEquals("test-source", error.getSourceId());
            }
        }
    }

    @Test
    public void testResolveWithoutTerminalRowFails() throws Exception {
        // Heartbeats and nothing else: the server closed without a row. That is an
        // error, not an empty result.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setResolveHeartbeats(2);
            server.setResolveSendsRow(false);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IOException error = Assert.assertThrows(
                        IOException.class,
                        () -> client.resolve("test-source"));
                Assert.assertTrue(error.getMessage().contains("no catalog row"));
            }
        }
    }

    @Test
    public void testGetTensorOnUnresolvedSourceSteersToResolve() throws Exception {
        // The Java twin of napari's `_is_unresolved`: a source with no tensors
        // and is_resolved false needs the consented resolve, and the error says
        // so (biopb/biopb#1032).
        try (TestFlightServer server = new TestFlightServer()) {
            server.setSourceHasTensors(false);
            server.setSourceResolved(false);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                SourceUnresolvedException error = Assert.assertThrows(
                        SourceUnresolvedException.class,
                        () -> client.getTensor("test-source"));
                Assert.assertTrue(error.getMessage().contains("resolve"));
            }
        }
    }

    @Test
    public void testGetTensorOnResolvedSourceWithNoTensorsSaysSo() throws Exception {
        // The other half: it resolved, and there was nothing readable in it.
        // Sending this caller at resolve() would point them at an operation that
        // can only succeed and change nothing -- which is what the old
        // `n == 0 -> unresolved` inference did.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setSourceHasTensors(false);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorNotFoundException error = Assert.assertThrows(
                        TensorNotFoundException.class,
                        () -> client.getTensor("test-source"));
                Assert.assertEquals("no_readable_tensors", error.getReason());
            }
        }
    }

    @Test
    public void testWarmReturnsTheTerminalCounts() throws Exception {
        // warm returns a status, not a row: residency is not a durable catalog
        // fact, so these counts exist nowhere else (biopb/biopb#1035).
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                WarmProgress done = client.warm("test-source");
                Assert.assertEquals(2, done.getFilesTotal());
                Assert.assertEquals(2, done.getFilesDone());
                Assert.assertEquals(2048L, done.getBytesDone());
            }
        }
    }

    @Test
    public void testWarmWithoutTerminalStatusFails() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            server.setWarmSendsDone(false);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IOException error = Assert.assertThrows(
                        IOException.class,
                        () -> client.warm("test-source"));
                Assert.assertTrue(error.getMessage().contains("no terminal status"));
            }
        }
    }

    @Test
    public void testWarmReportsProgressAndHonorsCancellation() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                List<WarmProgress> progress = new ArrayList<>();
                WarmProgress done = client.warm("test-source", progress::add, () -> false);
                Assert.assertEquals(1, progress.size());
                Assert.assertEquals(1, progress.get(0).getFilesDone());
                Assert.assertEquals(2, done.getFilesDone());

                TensorOperationCancelledException error = Assert.assertThrows(
                        TensorOperationCancelledException.class,
                        () -> client.warm("test-source", ignored -> Assert.fail("must not report after cancellation"),
                                () -> true));
                Assert.assertEquals("warm", error.getOperation());
            }
        }
    }

    @Test
    public void testGetSourceMetadataReadsTheColumn() throws Exception {
        // The column IS the answer -- filled once at registration, read back from
        // the catalog rather than recomputed (biopb/biopb#253).
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                Map<String, Object> metadata = client.getSourceMetadata("test-source");
                Assert.assertEquals("test_value", metadata.get("test_key"));
            }
        }
    }

    @Test
    public void testGetSourceMetadataOnUnresolvedSourceSteersToResolve() throws Exception {
        // The flag, not an empty tensor list: a source can resolve cleanly and
        // hold nothing readable, and telling *that* caller to resolve sends them
        // at an operation that can only succeed and change nothing
        // (biopb/biopb#1032).
        try (TestFlightServer server = new TestFlightServer()) {
            server.setSourceResolved(false);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalStateException error = Assert.assertThrows(
                        IllegalStateException.class,
                        () -> client.getSourceMetadata("test-source"));
                Assert.assertTrue(error.getMessage().contains("is unresolved"));
                Assert.assertTrue(error.getMessage().contains("resolve('test-source')"));
            }
        }
    }

    @Test
    public void testGetSourceMetadataReturnsEmptyForResolvedSourceWithNone() throws Exception {
        // The other half of the same distinction: resolved, simply no metadata.
        try (TestFlightServer server = new TestFlightServer()) {
            server.setSourceMetadataJson(null);
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                Assert.assertTrue(client.getSourceMetadata("test-source").isEmpty());
            }
        }
    }

    @Test
    public void testGetSourceMetadataUnknownSourceThrows() throws Exception {
        try (TestFlightServer server = new TestFlightServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.getSourceMetadata("nope"));
                Assert.assertTrue(error.getMessage().contains("Source not found"));
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

        int getFlightInfoRequestCount() {
            return producer.getFlightInfoRequestCount();
        }

        int getHealthRequestCount() {
            return producer.healthRequests.get();
        }

        String getLastReductionMethod() {
            return producer.getLastReductionMethod();
        }

        void setUploadStatusSequence(String sourceId, Map<String, Object>... statuses) {
            producer.setUploadStatusSequence(sourceId, Arrays.asList(statuses));
        }

        void setSourceResolved(boolean resolved) {
            producer.sourceResolved = resolved;
        }

        void setSourceMetadataJson(String json) {
            producer.sourceMetadataJson = json;
        }

        void setResolveHeartbeats(int count) {
            producer.resolveHeartbeats = count;
        }

        void setResolveSendsRow(boolean sends) {
            producer.resolveSendsRow = sends;
        }

        void setWarmSendsDone(boolean sends) {
            producer.warmSendsDone = sends;
        }

        void setSourceHasTensors(boolean has) {
            producer.sourceHasTensors = has;
        }

        void setProtocolVersion(int version) {
            producer.protocolVersion = version;
        }

        void setChunkWireProtocol(String version) {
            producer.chunkWireProtocol = version;
        }

        void setHealthUnauthenticated(boolean refuse) {
            producer.healthUnauthenticated = refuse;
        }

        /**
         * Shut the server down, then wait for the producer to be idle before
         * closing the allocator.
         *
         * <p>A cancelled action leaves its handler running -- that is the whole
         * point of cancelling -- and Arrow runs one on its own executor, which
         * {@code server.close()} does not wait for. Closing the allocator out
         * from under a handler that still holds a root reports it as a leak by
         * whichever test happened to cancel.
         */
        @Override
        public void close() throws Exception {
            server.close();
            long deadline = System.currentTimeMillis() + 5_000;
            while (producer.inFlight.get() > 0 && System.currentTimeMillis() < deadline) {
                Thread.sleep(10);
            }
            allocator.close();
        }
    }

    private static class TensorTestProducer extends NoOpFlightProducer {
        private final BufferAllocator allocator;
        private final TensorDescriptor baseDescriptor;
        private final org.apache.arrow.vector.types.pojo.Schema schema;
        private final Map<String, float[]> chunkData;
        private final Map<String, AtomicInteger> chunkRequests;
        private final AtomicInteger flightInfoRequests;
        private final Map<String, List<Map<String, Object>>> uploadStatusSequences;
        private final Map<String, AtomicInteger> uploadStatusCalls;
        private volatile FlightRequest lastCmd;
        // Cloud-path knobs. `sourceResolved` is what a row's is_resolved
        // column says; a resolve flips it, which is the transition the client
        // exists to drive.
        private volatile boolean sourceResolved = true;
        private volatile String sourceMetadataJson = "{\"test_key\": \"test_value\"}";
        private volatile int resolveHeartbeats = 0;
        private volatile boolean resolveSendsRow = true;
        private volatile boolean warmSendsDone = true;
        // An unresolved source lists with no tensors -- but so does one that
        // resolved and held nothing readable, which is the pair #1032 exists
        // to stop conflating.
        private volatile boolean sourceHasTensors = true;
        // The Flight protocol shape this fake claims to speak.
        volatile int protocolVersion = 2;
        final AtomicInteger healthRequests = new AtomicInteger();
        /** Producer calls currently running, so teardown can wait them out. */
        final AtomicInteger inFlight = new AtomicInteger();
        // A capability token cannot read the catalog tier health sits on.
        volatile boolean healthUnauthenticated = false;
        // The chunk encoding this fake stamps; null leaves the schema unstamped,
        // which is how a pre-#293 server presents.
        volatile String chunkWireProtocol = "2";
        // Residency is asked per call, never stored, so the fake counts the
        // asks as well as answering them (biopb/biopb#1035).

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
            this.flightInfoRequests = new AtomicInteger();
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

        int getFlightInfoRequestCount() {
            return flightInfoRequests.get();
        }

        String getLastReductionMethod() {
            if (lastCmd == null || !lastCmd.hasTensorRead()) {
                return null;
            }
            return lastCmd.getTensorRead().getReductionMethod();
        }

        @Override
        public FlightInfo getFlightInfo(FlightProducer.CallContext context, FlightDescriptor descriptor) {
            flightInfoRequests.incrementAndGet();
            FlightRequest cmd = parseCmd(descriptor.getCommand());
            lastCmd = cmd;
            TensorReadOption readOpt = cmd.hasTensorRead()
                    ? cmd.getTensorRead()
                    : null;

            if (!sourceHasTensors) {
                if (!sourceResolved) {
                    throw typedError(FlightStatusCode.UNAVAILABLE,
                            "Source unresolved (open to resolve)", "UNAVAILABLE", null);
                }
                throw typedError(FlightStatusCode.NOT_FOUND,
                        "Source has no readable tensors", "NOT_FOUND", "no_readable_tensors");
            }

            // A source with a registered upload sequence answers with the status
            // on its descriptor -- the poll path, which needs no endpoints.
            if (readOpt != null && uploadStatusSequences.containsKey(readOpt.getArrayId())) {
                TensorDescriptor.Builder d = TensorDescriptor.newBuilder()
                        .setArrayId(readOpt.getArrayId());
                UploadStatus st = nextUploadStatus(readOpt.getArrayId());
                if (st != null) {
                    d.setUploadStatus(st);
                }
                return new FlightInfo(
                        new org.apache.arrow.vector.types.pojo.Schema(new ArrayList<>()),
                        FlightDescriptor.command(d.build().toByteArray()),
                        new ArrayList<>(),
                        -1,
                        -1);
            }

            // Validate the tensor: "test-tensor" is the sole tensor of "test-source".
            if (readOpt == null
                    || !(readOpt.getArrayId().equals("test-tensor")
                            || readOpt.getArrayId().equals("test-source"))) {
                throw typedError(FlightStatusCode.NOT_FOUND,
                        "Tensor not found: " + (readOpt != null ? readOpt.getArrayId() : "null"),
                        "NOT_FOUND", "unknown_field");
            }

            if (readOpt.getScaleHintCount() != 0 && readOpt.getScaleHintCount() != 2) {
                throw typedError(FlightStatusCode.INVALID_ARGUMENT,
                        "Scale hint rank must match tensor rank", "INVALID_ARGUMENT", "scale_rank");
            }
            for (long scale : readOpt.getScaleHintList()) {
                if (scale <= 0) {
                    throw typedError(FlightStatusCode.INVALID_ARGUMENT,
                            "Scale hint must be positive", "INVALID_ARGUMENT", "scale_not_positive");
                }
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
                        planSchema(),
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
                        planSchema(),
                        FlightDescriptor.command(responseDescriptor.toByteArray()),
                        scaledEndpoints(),
                        -1,
                        -1);
            }

            return new FlightInfo(
                    planSchema(),
                    FlightDescriptor.command(baseDescriptor.toByteArray()),
                    baseEndpoints(),
                    -1,
                    -1);
        }

        /**
         * A typed server error, on the wire as the real server puts it there.
         *
         * <p>The transport status is NOT the payload's code: pyarrow has no
         * FlightNotFoundError, so every terminal domain error rides
         * FlightServerError and reaches a client as UNKNOWN, carrying the
         * precise code in the trailer (server.to_flight_error). Only
         * UNAVAILABLE has a class of its own.
         */
        private static RuntimeException typedError(
                FlightStatusCode status, String message, String code, String reason) {
            ErrorFlightMetadata metadata = new ErrorFlightMetadata();
            String payload = reason == null
                    ? "{\"code\":\"" + code + "\"}"
                    : "{\"code\":\"" + code + "\",\"reason\":\"" + reason + "\"}";
            metadata.insert("x-biopb-error-bin", payload.getBytes(StandardCharsets.UTF_8));
            FlightStatusCode wire = status == FlightStatusCode.UNAVAILABLE
                    ? FlightStatusCode.UNAVAILABLE
                    : FlightStatusCode.UNKNOWN;
            return new CallStatus(wire, null, message, metadata).toRuntimeException();
        }

        @Override
        public void doAction(
                FlightProducer.CallContext context,
                Action action,
                FlightProducer.StreamListener<Result> listener) {
            inFlight.incrementAndGet();
            try {
                dispatch(action, listener);
            } finally {
                inFlight.decrementAndGet();
            }
        }

        private void dispatch(Action action, FlightProducer.StreamListener<Result> listener) {
            if ("health".equals(action.getType())) {
                healthRequests.incrementAndGet();
                if (healthUnauthenticated) {
                    listener.onError(CallStatus.UNAUTHENTICATED
                            .withDescription("no catalog access").toRuntimeException());
                    return;
                }
                // Every v2 server answers this, and the SDK probes it once per
                // connection before its first real call.
                listener.onNext(new Result(("{\"status\":\"SERVING\",\"protocol\":"
                        + protocolVersion + "}").getBytes(StandardCharsets.UTF_8)));
                listener.onCompleted();
                return;
            }
            if ("resolve".equals(action.getType())) {
                doResolve(new String(action.getBody(), StandardCharsets.UTF_8), listener);
                return;
            }
            if ("warm".equals(action.getType())) {
                doWarm(listener);
                return;
            }
            // `upload_status` is not an action any more: it rides the descriptor
            // GetFlightInfo returns (biopb/biopb#1048 step 2). See
            // `nextUploadStatus`, which serves the registered sequence there.
            listener.onError(new IllegalArgumentException("Unknown action: " + action.getType()));
        }

        /**
         * The next status in this source's registered sequence, or null when it
         * has none -- which the client reads as UNKNOWN, the same as a source
         * the server does not serve.
         *
         * <p>Advances per call, so a polling test still sees its sequence move.
         */
        private UploadStatus nextUploadStatus(String sourceId) {
            List<Map<String, Object>> sequence = uploadStatusSequences.get(sourceId);
            if (sequence == null || sequence.isEmpty()) {
                return null;
            }
            AtomicInteger calls = uploadStatusCalls.computeIfAbsent(sourceId, ignored -> new AtomicInteger());
            int index = Math.min(calls.getAndIncrement(), sequence.size() - 1);
            Map<String, Object> entry = sequence.get(index);
            String state = String.valueOf(entry.get("state"));
            UploadStatus.Builder b = UploadStatus.newBuilder()
                    .setExpectedChunks(((Number) entry.get("expected_chunks")).longValue())
                    .setUploadedChunks(((Number) entry.get("uploaded_chunks")).longValue());
            Object reason = entry.get("reason");
            if (reason != null) {
                b.setReason(String.valueOf(reason));
            }
            // An UNKNOWN entry means "no upload here", which on the wire is the
            // field being absent rather than a state value.
            if ("UNKNOWN".equals(state)) {
                return null;
            }
            b.setState(UploadStatus.State.valueOf(state));
            return b.build();
        }


        /**
         * The `resolve` action: zero or more progress heartbeats, then one
         * terminal message carrying the source's now-concrete catalog row as
         * an Arrow IPC stream.
         */
        private void doResolve(String sourceId, FlightProducer.StreamListener<Result> listener) {
            for (int i = 0; i < resolveHeartbeats; i++) {
                ResolveStreamMessage beat = ResolveStreamMessage.newBuilder()
                        .setProgress(ResolveProgress.newBuilder()
                                .setElapsedSeconds(i)
                                .setTargetName("img.tif")
                                .setTargetBytes(1024)
                                .build())
                        .build();
                listener.onNext(new Result(beat.toByteArray()));
            }
            if (!resolveSendsRow) {
                // A stream of heartbeats and nothing else: the server closed
                // without a row, which the client must treat as an error.
                listener.onCompleted();
                return;
            }
            // Resolution is what makes the source resolved -- the row the
            // terminal message carries reflects that, and so does a later
            // browse.
            sourceResolved = true;
            byte[] ipc;
            try (VectorSchemaRoot root = catalogRoot(
                    "SELECT " + TensorFlightClient.SOURCE_ROW_COLUMNS + " FROM sources", 1)) {
                java.io.ByteArrayOutputStream sink = new java.io.ByteArrayOutputStream();
                try (org.apache.arrow.vector.ipc.ArrowStreamWriter writer =
                        new org.apache.arrow.vector.ipc.ArrowStreamWriter(
                                root, null, java.nio.channels.Channels.newChannel(sink))) {
                    writer.start();
                    writer.writeBatch();
                    writer.end();
                }
                ipc = sink.toByteArray();
            } catch (Exception e) {
                listener.onError(e);
                return;
            }
            ResolveStreamMessage done = ResolveStreamMessage.newBuilder()
                    .setSourceRow(ByteString.copyFrom(ipc))
                    .build();
            listener.onNext(new Result(done.toByteArray()));
            listener.onCompleted();
        }

        /** The `warm` action: one progress update, then the terminal counts. */
        private void doWarm(FlightProducer.StreamListener<Result> listener) {
            WarmStreamMessage beat = WarmStreamMessage.newBuilder()
                    .setProgress(WarmProgress.newBuilder()
                            .setFilesTotal(2)
                            .setFilesDone(1)
                            .setBytesTotal(2048)
                            .setBytesDone(1024)
                            .setCurrentName("0.0.0")
                            .build())
                    .build();
            listener.onNext(new Result(beat.toByteArray()));
            if (warmSendsDone) {
                WarmStreamMessage done = WarmStreamMessage.newBuilder()
                        .setDone(WarmProgress.newBuilder()
                                .setFilesTotal(2)
                                .setFilesDone(2)
                                .setBytesTotal(2048)
                                .setBytesDone(2048)
                                .build())
                        .build();
                listener.onNext(new Result(done.toByteArray()));
            }
            listener.onCompleted();
        }

        @Override
        public void getStream(
                FlightProducer.CallContext context,
                Ticket ticket,
                FlightProducer.ServerStreamListener listener) {
            inFlight.incrementAndGet();
            try {
                serve(ticket, listener);
            } finally {
                inFlight.decrementAndGet();
            }
        }

        private void serve(Ticket ticket, FlightProducer.ServerStreamListener listener) {
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
                try (VectorSchemaRoot root = catalogRoot(sql, matches ? 1 : 0)) {
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

        /**
         * The `sources` row(s) a query selects, projected to the SELECT list.
         *
         * <p>Projected, not "always every column": getSourceMetadata asks for
         * {@code SELECT is_resolved, metadata_json}, and a fake that answered
         * with the browse columns instead would hand back a root with no
         * metadata_json vector -- which the client reads as "no metadata"
         * rather than failing, so the test would pass without ever exercising
         * the column it exists to check.
         */
        private VectorSchemaRoot catalogRoot(String sql, int rows) {
            List<String> selected = selectedColumns(sql);
            Field arrayId = new Field("array_id", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
            Field dimLabels = new Field("dim_labels", FieldType.nullable(ArrowType.List.INSTANCE),
                    Collections.singletonList(new Field("item", FieldType.nullable(ArrowType.Utf8.INSTANCE), null)));
            Field shape = new Field("shape", FieldType.nullable(ArrowType.List.INSTANCE),
                    Collections.singletonList(new Field("item", FieldType.nullable(new ArrowType.Int(64, true)), null)));
            Field dtype = new Field("dtype", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
            Field tensorStruct = new Field("item", FieldType.nullable(ArrowType.Struct.INSTANCE),
                    Arrays.asList(arrayId, dimLabels, shape, dtype));
            Map<String, Field> columns = new java.util.LinkedHashMap<>();
            columns.put("source_id", new Field("source_id", FieldType.nullable(ArrowType.Utf8.INSTANCE), null));
            columns.put("source_url", new Field("source_url", FieldType.nullable(ArrowType.Utf8.INSTANCE), null));
            columns.put("source_type", new Field("source_type", FieldType.nullable(ArrowType.Utf8.INSTANCE), null));
            columns.put("metadata_json", new Field("metadata_json", FieldType.nullable(ArrowType.Utf8.INSTANCE), null));
            columns.put("data_resident", new Field("data_resident", FieldType.nullable(ArrowType.Bool.INSTANCE), null));
            columns.put("is_resolved", new Field("is_resolved", FieldType.nullable(ArrowType.Bool.INSTANCE), null));
            columns.put("tensors", new Field("tensors", FieldType.nullable(ArrowType.List.INSTANCE),
                    Collections.singletonList(tensorStruct)));

            List<Field> fields = new ArrayList<>();
            for (String name : selected) {
                Field field = columns.get(name);
                if (field == null) {
                    throw new IllegalArgumentException("fake catalog has no column: " + name);
                }
                fields.add(field);
            }
            VectorSchemaRoot root = VectorSchemaRoot.create(new Schema(fields), allocator);
            root.allocateNew();
            if (rows == 0) {
                root.setRowCount(0);
                return root;
            }
            setText(root, "source_id", "test-source");
            setText(root, "source_url", "mock://test");
            setText(root, "source_type", "mock");
            if (sourceMetadataJson != null) {
                setText(root, "metadata_json", sourceMetadataJson);
            }
            setBool(root, "data_resident", true);
            setBool(root, "is_resolved", sourceResolved);
            ListVector tensors = (ListVector) root.getVector("tensors");
            if (tensors != null && !sourceHasTensors) {
                UnionListWriter empty = tensors.getWriter();
                empty.setPosition(0);
                empty.startList();
                empty.endList();
                tensors.setValueCount(1);
            } else if (tensors != null) {
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
            }
            root.setRowCount(1);
            return root;
        }

        /** The column names between SELECT and FROM, in order. */
        private static List<String> selectedColumns(String sql) {
            int select = sql.toUpperCase(java.util.Locale.ROOT).indexOf("SELECT ");
            int from = sql.toUpperCase(java.util.Locale.ROOT).indexOf(" FROM ");
            String list = sql.substring(select + "SELECT ".length(), from);
            List<String> out = new ArrayList<>();
            for (String part : list.split(",")) {
                out.add(part.trim());
            }
            return out;
        }

        private static void setText(VectorSchemaRoot root, String column, String value) {
            org.apache.arrow.vector.VarCharVector vector =
                    (org.apache.arrow.vector.VarCharVector) root.getVector(column);
            if (vector != null) {
                vector.setSafe(0, value.getBytes(StandardCharsets.UTF_8));
            }
        }

        private static void setBool(VectorSchemaRoot root, String column, boolean value) {
            org.apache.arrow.vector.BitVector vector =
                    (org.apache.arrow.vector.BitVector) root.getVector(column);
            if (vector != null) {
                vector.setSafe(0, value ? 1 : 0);
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
            // Unified binary chunk schema (biopb/biopb#293): data (binary), dtype (utf8),
            // stamped with the encoding version the client gates on -- the real
            // server stamps every read plan's schema the same way.
            Field dataField = new Field("data", FieldType.nullable(ArrowType.Binary.INSTANCE), null);
            Field dtypeField = new Field("dtype", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
            return new org.apache.arrow.vector.types.pojo.Schema(
                    Arrays.asList(dataField, dtypeField),
                    Collections.singletonMap("chunk_wire_protocol", "2"));
        }

        /** The read-plan schema, stamped as this fake is currently configured. */
        private org.apache.arrow.vector.types.pojo.Schema planSchema() {
            if ("2".equals(chunkWireProtocol)) {
                return schema;
            }
            java.util.Map<String, String> metadata = new java.util.HashMap<>();
            if (chunkWireProtocol != null) {
                metadata.put("chunk_wire_protocol", chunkWireProtocol);
            }
            return new org.apache.arrow.vector.types.pojo.Schema(schema.getFields(), metadata);
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
