package biopb.tensor;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.CallStatus;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightProducer;
import org.apache.arrow.flight.FlightServer;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.NoOpFlightProducer;
import org.apache.arrow.flight.PutResult;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.Assert;
import org.junit.Test;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import biopb.image.Point;
import biopb.image.ROI;
import biopb.image.RoiAnnotation;
import biopb.image.RoiDeleteResult;
import biopb.image.RoiListResult;
import biopb.image.RoiPruneRequest;
import biopb.image.RoiPruneResult;
import biopb.image.RoiPutResult;
import biopb.image.RoiUnseen;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;

/**
 * Source lifecycle, ROI annotations and uploads against a fake producer.
 *
 * <p>Kept apart from {@link TensorFlightClientTest}, whose fake serves the read
 * path: these are the write- and lifecycle-side surfaces, and each needs a
 * producer that records what it was sent rather than one that answers reads.
 */
public class TensorLifecycleTest {

    // ---- source lifecycle -------------------------------------------------

    @Test
    public void testAddSourceReportsProgressThenTheTally() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                List<AddSourceProgress> progress = new ArrayList<>();
                AddSourceResult result = client.addSource(
                        "/data/plate", "ome-zarr", progress::add, () -> false);

                Assert.assertEquals(Arrays.asList("plate_a", "plate_b"), result.getAddedList());
                Assert.assertEquals(Collections.singletonList("plate_c"), result.getRefreshedList());
                Assert.assertEquals(2, progress.size());
                Assert.assertEquals(1, progress.get(0).getAddedCount());
                // The request reaches the server whole, adapter type included.
                Assert.assertEquals("/data/plate", server.producer.lastAddSource.getUrl());
                Assert.assertEquals("ome-zarr", server.producer.lastAddSource.getSourceType());
            }
        }
    }

    @Test
    public void testAddSourceCancelKeepsWhatRegistered() throws Exception {
        // A cancel is intentional, so it reports an empty tally rather than an
        // error -- and the sources already registered stay registered.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                AddSourceResult result = client.addSource("/data/plate", "", null, () -> true);
                Assert.assertEquals(0, result.getAddedCount());
            }
        }
    }

    @Test
    public void testAddSourceCancelOnTheTerminalKeepsTheTally() throws Exception {
        // The poll runs AFTER a message is consumed, so a cancel landing exactly
        // on the terminal result must not discard a tally already in hand.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                // False for the two heartbeats, true once the result has landed.
                int[] calls = { 0 };
                AddSourceResult result = client.addSource(
                        "/data/plate", "", null, () -> ++calls[0] >= 3);
                Assert.assertEquals(Arrays.asList("plate_a", "plate_b"), result.getAddedList());
            }
        }
    }

    @Test
    public void testCancelStopsTheServerNotJustTheClient() throws Exception {
        // The call is created inside a gRPC CancellableContext, so stopping the
        // loop cancels the RPC and the server sees it -- what Python gets by
        // closing its generator. Without it the server finishes a walk nobody
        // is reading.
        try (TestServer server = new TestServer()) {
            server.producer.addSourceHeartbeats = 2000;
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                java.util.concurrent.atomic.AtomicInteger seen =
                        new java.util.concurrent.atomic.AtomicInteger();
                AddSourceResult result = client.addSource("/data/plate", "",
                        ignored -> seen.incrementAndGet(), () -> seen.get() >= 3);

                Assert.assertEquals(0, result.getAddedCount());
                // The server stopped on its own poll rather than running to 2000.
                long deadline = System.currentTimeMillis() + 10_000;
                while (!server.producer.observedCancel && System.currentTimeMillis() < deadline) {
                    Thread.sleep(20);
                }
                Assert.assertTrue("server never observed the cancel",
                        server.producer.observedCancel);
                Assert.assertTrue("server emitted " + server.producer.emitted.get()
                                + ", i.e. it was not stopped early",
                        server.producer.emitted.get() < 2000);
            }
        }
    }

    @Test
    public void testAddSourceWithoutTerminalResultFails() throws Exception {
        try (TestServer server = new TestServer()) {
            server.producer.addSourceSendsResult = false;
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                IOException error = Assert.assertThrows(IOException.class,
                        () -> client.addSource("/data/plate"));
                Assert.assertTrue(error.getMessage().contains("no terminal result"));
            }
        }
    }

    @Test
    public void testAddSourceOnAnOldServerNamesTheFeature() throws Exception {
        // "Unknown action" is how a server that predates the action answers;
        // the client says which feature is missing, not which RPC failed.
        try (TestServer server = new TestServer()) {
            server.producer.knownActions = Collections.emptySet();
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                UnsupportedOperationException error = Assert.assertThrows(
                        UnsupportedOperationException.class,
                        () -> client.addSource("/data/plate"));
                Assert.assertTrue(error.getMessage().contains("Runtime source registration is unavailable"));
                Assert.assertTrue(error.getMessage().contains("add_source"));
            }
        }
    }

    @Test
    public void testRemoveSourceTakesADndBranch() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RemoveSourceResult result = client.removeSource("dnd://drop-1");
                Assert.assertEquals(Arrays.asList("plate_a", "plate_b"), result.getRemovedList());
                Assert.assertEquals("dnd://drop-1", server.producer.lastRemoveSource.getRootUrl());
            }
        }
    }

    @Test
    public void testRemoveSourceOnAnOldServerNamesTheFeature() throws Exception {
        try (TestServer server = new TestServer()) {
            server.producer.knownActions = Collections.emptySet();
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                UnsupportedOperationException error = Assert.assertThrows(
                        UnsupportedOperationException.class,
                        () -> client.removeSource("dnd://drop-1"));
                Assert.assertTrue(error.getMessage().contains("Source removal is unavailable"));
            }
        }
    }

    // ---- label sets -------------------------------------------------------

    @Test
    public void testLabelSetsIsACatalogQueryOverThePath() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                Assert.assertEquals(
                        Arrays.asList("src_ab12/labels/@ome", "src_ab12/labels/nuclei"),
                        client.labelSets("src_ab12"));
                // The prefix is the image's own path, quoted for the SQL surface.
                Assert.assertTrue(server.producer.lastSql.contains("'src_ab12/labels/'"));
            }
        }
    }

    @Test
    public void testDiscardingALabelSetNamesItByItsArrayId() throws Exception {
        // Removing an uploaded set is discarding its upload; there is no
        // delete verb of its own, and a set is named by its array_id.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                Map<String, Object> status = client.setUploadStatus(
                        "src_ab12/labels/nuclei", UploadStatus.State.DISCARDED, "replaced");
                Assert.assertEquals("src_ab12/labels/nuclei",
                        server.producer.lastSetStatus.getArrayId());
                Assert.assertEquals("replaced", server.producer.lastSetStatus.getReason());
                Assert.assertEquals("DISCARDED", status.get("state"));
            }
        }
    }

    // ---- ROI annotations --------------------------------------------------

    @Test
    public void testListRoisCarriesSetsAndTruncationOffTheSchema() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RoiListResult result = client.listRois("src_ab12");

                Assert.assertEquals(2, result.getRoisCount());
                Assert.assertEquals("roi-1", result.getRois(0).getRoiId());
                Assert.assertEquals("nuclei", result.getRois(0).getSetName());
                Assert.assertEquals(12, (int) result.getRois(0).getPlaneMap().get(0));
                Assert.assertEquals(4.0f, result.getRois(1).getRoi().getPoint().getX(), 1e-6f);

                // truncated and the tensor's sets ride the stream's schema
                // metadata, so they arrive whatever `rois` covers.
                Assert.assertTrue(result.getTruncated());
                Assert.assertEquals(2, result.getSetsCount());
                Assert.assertEquals("nuclei", result.getSets(0).getSetName());
                Assert.assertEquals(17L, result.getSets(0).getCount());
                Assert.assertFalse(result.getSets(0).getReserved());
                Assert.assertTrue(result.getSets(1).getReserved());

                Assert.assertEquals("src_ab12", server.producer.lastRoiRead.getArrayId());
                Assert.assertEquals("", server.producer.lastRoiRead.getSetName());
            }
        }
    }

    @Test
    public void testListRoisNarrowsToOneSet() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                client.listRois("src_ab12", "@ome");
                Assert.assertEquals("@ome", server.producer.lastRoiRead.getSetName());
            }
        }
    }

    @Test
    public void testPutRoisSendsTheRowsAndReadsTheReply() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RoiAnnotation drawn = RoiAnnotation.newBuilder()
                        .setArrayId("src_ab12")
                        .setSetName("nuclei")
                        .setLabel("mitotic")
                        .setRoi(ROI.newBuilder().setPoint(Point.newBuilder().setX(7).setY(8)))
                        .putPlane(0, 3)
                        .build();

                RoiPutResult result = client.putRois("src_ab12", Collections.singletonList(drawn));

                // The reply is the server's, read out of the put's app_metadata.
                Assert.assertEquals(1, result.getStoredCount());
                Assert.assertEquals("minted-1", result.getStored(0).getRoiId());

                // What the server received is the annotation, whole.
                Assert.assertEquals(1, server.producer.putRois.size());
                RoiAnnotation received = server.producer.putRois.get(0);
                Assert.assertEquals("mitotic", received.getLabel());
                Assert.assertEquals(7.0f, received.getRoi().getPoint().getX(), 1e-6f);
                Assert.assertEquals(Integer.valueOf(3), received.getPlaneMap().get(0));
                Assert.assertEquals("src_ab12", server.producer.lastRoiPut.getArrayId());
                Assert.assertFalse(server.producer.lastRoiPut.getCheckRev());
            }
        }
    }

    @Test
    public void testPutRoisPassesCheckRev() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                client.putRois("src_ab12", Collections.emptyList(), true);
                Assert.assertTrue(server.producer.lastRoiPut.getCheckRev());
                Assert.assertTrue(server.producer.putRois.isEmpty());
            }
        }
    }

    @Test
    public void testDeleteRoisSendsTheIds() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RoiDeleteResult result = client.deleteRois(
                        "src_ab12", Arrays.asList("roi-1", "roi-2"), "");
                Assert.assertEquals(Arrays.asList("roi-1", "roi-2"), result.getDeletedList());
                Assert.assertEquals(Arrays.asList("roi-1", "roi-2"), server.producer.deleteIds);
            }
        }
    }

    @Test
    public void testDeleteRoisWithoutIdsDropsAWholeLayer() throws Exception {
        // An empty stream plus a set_name is how a layer is dropped; the client
        // must still open the put so the command reaches the server.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                client.deleteRois("src_ab12", Collections.emptyList(), "nuclei");
                Assert.assertTrue(server.producer.deleteIds.isEmpty());
                Assert.assertEquals("nuclei", server.producer.lastRoiDelete.getSetName());
                Assert.assertEquals("src_ab12", server.producer.lastRoiDelete.getArrayId());
            }
        }
    }

    @Test
    public void testPruneRoisReportsBeforeItDeletes() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RoiPruneResult report = client.pruneRois(30, false);
                Assert.assertEquals(0, report.getDeleted());
                Assert.assertEquals(1, report.getUnseenCount());
                Assert.assertEquals("gone_src", report.getUnseen(0).getArrayId());
                Assert.assertFalse(server.producer.lastPrune.getApply());
                Assert.assertEquals(30, server.producer.lastPrune.getUnseenDays());

                Assert.assertEquals(4, client.pruneRois(30, true).getDeleted());
                Assert.assertTrue(server.producer.lastPrune.getApply());
            }
        }
    }

    // ---- uploads ----------------------------------------------------------

    @Test
    public void testAddTensorEchoesTheServersDescriptor() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorDescriptor descriptor = client.addTensor(
                        "cache://registered_abc123/mine", new long[] {4, 6}, "<u2", new long[] {2, 3},
                        Arrays.asList("y", "x"), "{\"ome\":true}");

                Assert.assertEquals("registered_abc123/mine", descriptor.getArrayId());
                Assert.assertEquals(Arrays.asList(4L, 6L), descriptor.getShapeList());
                Assert.assertEquals(Arrays.asList(2L, 3L), descriptor.getChunkShapeList());
                Assert.assertEquals("<u2", descriptor.getDtype());
                Assert.assertEquals(Arrays.asList("y", "x"), descriptor.getDimLabelsList());
                Assert.assertEquals("{\"ome\":true}", server.producer.lastCreate.getMetadataJson());
            }
        }
    }

    @Test
    public void testAddTensorTakesShapeAndDtypeFromATemplate() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                RandomAccessibleInterval<UnsignedShortType> template =
                        ArrayImgs.unsignedShorts(new short[24], 4, 6);
                TensorDescriptor descriptor = client.addTensor(
                        "cache://registered_abc123/mine", template, new long[] {2, 3}, null, null);

                Assert.assertEquals(Arrays.asList(4L, 6L), descriptor.getShapeList());
                Assert.assertEquals("<u2", server.producer.lastCreate.getDtype());
            }
        }
    }

    @Test
    public void testUploadArrayWritesEveryChunkOnTheGridAndSeals() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                // 4x6 on a 2x3 grid: four chunks, each in row-major order.
                short[] values = new short[24];
                for (int i = 0; i < values.length; i++) {
                    values[i] = (short) (i + 1);
                }
                RandomAccessibleInterval<UnsignedShortType> array =
                        ArrayImgs.unsignedShorts(values, 6, 4);

                TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                        .setArrayId("registered_abc123/mine")
                        .addAllShape(Arrays.asList(6L, 4L))
                        .addAllChunkShape(Arrays.asList(3L, 2L))
                        .setDtype("<u2")
                        .build();

                server.producer.plannedTensor = descriptor;
                Map<String, Object> status = client.uploadArray(descriptor, array);

                Assert.assertEquals(4, server.producer.chunks.size());
                Chunk first = server.producer.chunks.get(0);
                Assert.assertEquals(Arrays.asList(0L, 0L), first.bounds.getStartList());
                Assert.assertEquals(Arrays.asList(3L, 2L), first.bounds.getStopList());
                Chunk last = server.producer.chunks.get(3);
                Assert.assertEquals(Arrays.asList(3L, 2L), last.bounds.getStartList());
                Assert.assertEquals(Arrays.asList(6L, 4L), last.bounds.getStopList());

                // The order each chunk is written in is the order a read
                // scatters it back in, so the chunks the server holds
                // reassemble into the array that was uploaded. That symmetry is
                // the whole contract; an offset convention asserted on its own
                // can be self-consistently wrong.
                assertReassembles(array, server.producer.chunks, 6, 4);

                // Publishing is what marks the source complete, and a
                // whole-array upload does it on the caller's behalf.
                Assert.assertEquals("registered_abc123/mine", server.producer.lastSetStatus.getArrayId());
                Assert.assertEquals("READY", status.get("state"));
                Assert.assertEquals(4.0d, status.get("uploaded_chunks"));
            }
        }
    }

    @Test
    public void testUploadArrayRefusesAShapeMismatch() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                        .setArrayId("registered_abc123/mine")
                        .addAllShape(Arrays.asList(8L, 4L))
                        .addAllChunkShape(Arrays.asList(4L, 2L))
                        .setDtype("<u2")
                        .build();
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.uploadArray(descriptor,
                                ArrayImgs.unsignedShorts(new short[24], 6, 4)));
                Assert.assertTrue(error.getMessage().contains("does not match the declared shape"));
                Assert.assertTrue(server.producer.chunks.isEmpty());
            }
        }
    }

    @Test
    public void testUploadArrayRefusesADtypeMismatch() throws Exception {
        // The chunk column is typed by the declared dtype, so an unchecked
        // mismatch narrows every value and the upload still "succeeds".
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                        .setArrayId("registered_abc123/mine")
                        .addAllShape(Arrays.asList(2L, 2L))
                        .addAllChunkShape(Arrays.asList(2L, 2L))
                        .setDtype("<f4")
                        .build();
                IllegalArgumentException error = Assert.assertThrows(
                        IllegalArgumentException.class,
                        () -> client.uploadArray(descriptor,
                                ArrayImgs.unsignedShorts(new short[4], 2, 2)));
                Assert.assertTrue(error.getMessage().contains("does not match the declared dtype"));
                Assert.assertTrue(server.producer.chunks.isEmpty());
            }
        }
    }

    @Test
    public void testLabelSetSkipsItsEmptyChunks() throws Exception {
        // A label set's unwritten chunk reads back as background, so an all-zero
        // block is not sent at all (biopb/biopb#1059). A `cache:` source has no
        // such fill value, so the skip must not apply to it.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                short[] values = new short[24];
                values[0] = 5; // only the first 3x2 block is labelled
                RandomAccessibleInterval<UnsignedShortType> array =
                        ArrayImgs.unsignedShorts(values, 6, 4);

                server.producer.plannedTensor = labelDescriptor("src_ab12/labels/nuclei");
                client.uploadArray(labelDescriptor("src_ab12/labels/nuclei"), array);
                Assert.assertEquals(1, server.producer.chunks.size());
                Assert.assertEquals(Arrays.asList(0L, 0L),
                        server.producer.chunks.get(0).bounds.getStartList());

                server.producer.chunks.clear();
                server.producer.plannedTensor = labelDescriptor("registered_abc123/mine");
                client.uploadArray(labelDescriptor("registered_abc123/mine"), array);
                Assert.assertEquals(4, server.producer.chunks.size());
            }
        }
    }

    @Test
    public void testUploadArrayReadsACroppedViewAtItsOwnMin() throws Exception {
        // A view's random access is unbounded, so reading it at 0-based
        // coordinates returns real pixels from the wrong place and the upload
        // stores them without complaint. The interval's min is the origin.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                short[] values = new short[12 * 8];
                for (int i = 0; i < values.length; i++) {
                    values[i] = (short) (i + 1);
                }
                RandomAccessibleInterval<UnsignedShortType> whole =
                        ArrayImgs.unsignedShorts(values, 12, 8);
                // The bottom-right 6x4 quadrant, which starts at (6, 4).
                RandomAccessibleInterval<UnsignedShortType> crop = net.imglib2.view.Views.interval(
                        whole, new long[] {6, 4}, new long[] {11, 7});

                TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                        .setArrayId("registered_abc123/mine")
                        .addAllShape(Arrays.asList(6L, 4L))
                        .addAllChunkShape(Arrays.asList(3L, 2L))
                        .setDtype("<u2")
                        .build();
                server.producer.plannedTensor = descriptor;
                client.uploadArray(descriptor, crop);

                Assert.assertEquals(4, server.producer.chunks.size());
                // Chunk (0,0) of the tensor is the crop's own first block --
                // (6,4) of the underlying image, which holds 6*1 + 12*4 + 1.
                Assert.assertEquals(Integer.valueOf(55), server.producer.chunks.get(0).values.get(0));
                assertReassembles(crop, server.producer.chunks, 6, 4);
            }
        }
    }

    @Test
    public void testUploadChunkSendsOneBlock() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                short[] values = new short[24];
                for (int i = 0; i < values.length; i++) {
                    values[i] = (short) (i + 1);
                }
                RandomAccessibleInterval<UnsignedShortType> array =
                        ArrayImgs.unsignedShorts(values, 6, 4);
                TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                        .setArrayId("registered_abc123/mine")
                        .addAllShape(Arrays.asList(6L, 4L))
                        .addAllChunkShape(Arrays.asList(3L, 2L))
                        .setDtype("<u2")
                        .build();

                ChunkBounds bounds = ChunkBounds.newBuilder()
                        .addAllStart(Arrays.asList(3L, 2L))
                        .addAllStop(Arrays.asList(6L, 4L))
                        .build();
                server.producer.plannedTensor = descriptor;
                client.uploadChunk(descriptor, bounds, array);

                Assert.assertEquals(1, server.producer.chunks.size());
                // One block, read out of the interval at its own global
                // coordinates -- so the whole array can be handed to every
                // chunk, and the corner block still carries the corner values.
                Assert.assertEquals(6, server.producer.chunks.get(0).values.size());
                assertBlockMatches(array, server.producer.chunks.get(0));
                // The manual half does NOT seal; that is setUploadStatus's job.
                Assert.assertNull(server.producer.lastSetStatus);
            }
        }
    }

    @Test
    public void testRegisterSourceAnswersTheMintedId() throws Exception {
        // The id is the server's: the client sends a name and gets back
        // something it could not have derived from it.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                Assert.assertEquals("registered_abc123", client.registerSource("plate"));
                Assert.assertEquals("plate", server.producer.lastRegisterSource.getName());
            }
        }
    }

    @Test
    public void testRegisterSourceCarriesTheMetadataVerbatim() throws Exception {
        // Opaque, as on addTensor: the client does not parse or re-encode
        // the OME tree, so what the server stores is what the caller wrote.
        String metadata = "{\"omero\":{\"channels\":[]}}";
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                client.registerSource("plate", metadata);
                Assert.assertEquals(
                        metadata, server.producer.lastRegisterSource.getMetadataJson());
            }
        }
    }

    @Test
    public void testRegisterSourceSendsAnEmptyStringForNoMetadata() throws Exception {
        // Both SDKs have to mean the same thing by omitting the block, and for
        // a proto string field that is "", which the server reads as absent.
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                client.registerSource("");
                Assert.assertEquals("", server.producer.lastRegisterSource.getName());
                Assert.assertEquals(
                        "", server.producer.lastRegisterSource.getMetadataJson());
            }
        }
    }

    @Test
    public void testSetUploadStatusSealsAndReportsTheStatus() throws Exception {
        try (TestServer server = new TestServer()) {
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                Map<String, Object> status = client.setUploadStatus(
                        "registered_abc123/mine", UploadStatus.State.READY, "");
                Assert.assertEquals("registered_abc123/mine", status.get("source_id"));
                Assert.assertEquals("READY", status.get("state"));
            }
        }
    }

    @Test
    public void testUploadRefusalSurfacesAsTheTypedException() throws Exception {
        // A sealed or discarded upload is refused with CANCELLED + an
        // `upload_*` reason; a writer must be able to act on the type, not
        // parse a message (biopb/biopb#1).
        try (TestServer server = new TestServer()) {
            server.producer.refuseChunks = true;
            try (TensorFlightClient client = new TensorFlightClient("localhost", server.getPort())) {
                TensorDescriptor descriptor = TensorDescriptor.newBuilder()
                        .setArrayId("registered_abc123/mine")
                        .addAllShape(Arrays.asList(2L, 2L))
                        .addAllChunkShape(Arrays.asList(2L, 2L))
                        .setDtype("<u2")
                        .build();
                server.producer.plannedTensor = descriptor;
                UploadRefusedException error = Assert.assertThrows(
                        UploadRefusedException.class,
                        () -> client.uploadChunk(descriptor,
                                ChunkBounds.newBuilder()
                                        .addAllStart(Arrays.asList(0L, 0L))
                                        .addAllStop(Arrays.asList(2L, 2L))
                                        .build(),
                                ArrayImgs.unsignedShorts(new short[4], 2, 2)));
                Assert.assertEquals("DISCARDED", error.getState());
                Assert.assertEquals("registered_abc123/mine", error.getSourceId());
            }
        }
    }

    /**
     * The chunks, scattered back through the read path's own decoder, equal the
     * array they came from.
     */
    private static void assertReassembles(
            RandomAccessibleInterval<UnsignedShortType> original,
            List<Chunk> chunks,
            long... dims) {
        net.imglib2.img.array.ArrayImg<UnsignedShortType, ?> rebuilt =
                ArrayImgs.unsignedShorts(dims);
        for (Chunk chunk : chunks) {
            double[] values = new double[chunk.values.size()];
            for (int i = 0; i < values.length; i++) {
                values[i] = chunk.values.get(i);
            }
            TensorChunkCodec.writeChunk(rebuilt.randomAccess(), chunk.bounds, values);
        }
        net.imglib2.RandomAccess<UnsignedShortType> expected = original.randomAccess();
        net.imglib2.RandomAccess<UnsignedShortType> actual = rebuilt.randomAccess();
        for (long y = 0; y < dims[1]; y++) {
            for (long x = 0; x < dims[0]; x++) {
                expected.setPosition(new long[] { original.min(0) + x, original.min(1) + y });
                actual.setPosition(new long[] { x, y });
                Assert.assertEquals("at (" + x + "," + y + ")",
                        expected.get().get(), actual.get().get());
            }
        }
    }

    /** One chunk's values are the block the bounds name, in the read path's order. */
    private static void assertBlockMatches(
            RandomAccessibleInterval<UnsignedShortType> original, Chunk chunk) {
        long[] start = new long[chunk.bounds.getStartCount()];
        long[] extents = new long[start.length];
        for (int axis = 0; axis < start.length; axis++) {
            start[axis] = chunk.bounds.getStart(axis);
            extents[axis] = chunk.bounds.getStop(axis) - start[axis];
        }
        net.imglib2.RandomAccess<UnsignedShortType> access = original.randomAccess();
        long[] local = new long[start.length];
        long[] global = new long[start.length];
        for (int index = 0; index < chunk.values.size(); index++) {
            TensorChunkCodec.rowMajorPosition(index, extents, local);
            for (int axis = 0; axis < start.length; axis++) {
                global[axis] = start[axis] + local[axis];
            }
            access.setPosition(global);
            Assert.assertEquals(access.get().get(), (int) chunk.values.get(index));
        }
    }

    private static TensorDescriptor labelDescriptor(String arrayId) {
        return TensorDescriptor.newBuilder()
                .setArrayId(arrayId)
                .addAllShape(Arrays.asList(6L, 4L))
                .addAllChunkShape(Arrays.asList(3L, 2L))
                .setDtype("<u2")
                .build();
    }

    // ---- the fake ---------------------------------------------------------

    /** One chunk as the server received it. */
    private static final class Chunk {
        final ChunkBounds bounds;
        final List<Integer> values;

        Chunk(ChunkBounds bounds, List<Integer> values) {
            this.bounds = bounds;
            this.values = values;
        }
    }

    private static final class TestServer implements AutoCloseable {
        private final BufferAllocator allocator;
        private final FlightServer server;
        final LifecycleProducer producer;

        TestServer() throws IOException {
            this.allocator = new RootAllocator(Long.MAX_VALUE);
            this.producer = new LifecycleProducer(allocator);
            this.server = FlightServer.builder(
                    allocator, Location.forGrpcInsecure("localhost", 0), producer)
                    .build()
                    .start();
        }

        int getPort() {
            return server.getPort();
        }

        @Override
        public void close() throws Exception {
            server.close();
            allocator.close();
        }
    }

    private static final class LifecycleProducer extends NoOpFlightProducer {
        private final BufferAllocator allocator;

        volatile java.util.Set<String> knownActions = new java.util.HashSet<>(Arrays.asList(
                "add_source", "remove_source", "roi_prune", "add_tensor",
                "register_source", "set_upload_status"));
        volatile RegisterSource lastRegisterSource = null;
        volatile boolean addSourceSendsResult = true;
        volatile int addSourceHeartbeats = 2;
        volatile boolean observedCancel = false;
        final java.util.concurrent.atomic.AtomicInteger emitted =
                new java.util.concurrent.atomic.AtomicInteger();
        volatile boolean refuseChunks = false;
        /**
         * The tensor {@code getFlightInfo} plans. A write asks the server what
         * a chunk is, so a test that uploads has to say what it declared --
         * this fake keeps no catalog.
         */
        volatile TensorDescriptor plannedTensor;

        volatile AddSourceRequest lastAddSource;
        volatile RemoveSourceRequest lastRemoveSource;
        volatile RoiPruneRequest lastPrune;
        volatile TensorDescriptor lastCreate;
        volatile SetUploadStatus lastSetStatus;
        volatile RoiRead lastRoiRead;
        volatile RoiPut lastRoiPut;
        volatile RoiDelete lastRoiDelete;
        volatile String lastSql;
        final List<RoiAnnotation> putRois = new CopyOnWriteArrayList<>();
        final List<String> deleteIds = new CopyOnWriteArrayList<>();
        final List<Chunk> chunks = new CopyOnWriteArrayList<>();

        LifecycleProducer(BufferAllocator allocator) {
            this.allocator = allocator;
        }

        @Override
        public void doAction(
                FlightProducer.CallContext context,
                Action action,
                FlightProducer.StreamListener<Result> listener) {
            try {
                if ("health".equals(action.getType())) {
                    // Answered whatever knownActions says: an older v2 server
                    // missing add_source still has health, and the SDK probes it
                    // before every first call.
                    listener.onNext(new Result(
                            "{\"status\":\"SERVING\",\"protocol\":2}".getBytes(StandardCharsets.UTF_8)));
                    listener.onCompleted();
                    return;
                }
                if (!knownActions.contains(action.getType())) {
                    listener.onError(org.apache.arrow.flight.CallStatus.INTERNAL
                            .withDescription("Unknown action: " + action.getType())
                            .toRuntimeException());
                    return;
                }
                switch (action.getType()) {
                    case "add_source":
                        doAddSource(context, action, listener);
                        break;
                    case "remove_source":
                        lastRemoveSource = RemoveSourceRequest.parseFrom(action.getBody());
                        listener.onNext(new Result(RemoveSourceResult.newBuilder()
                                .addAllRemoved(Arrays.asList("plate_a", "plate_b"))
                                .build().toByteArray()));
                        break;
                    case "roi_prune":
                        lastPrune = RoiPruneRequest.parseFrom(action.getBody());
                        listener.onNext(new Result(RoiPruneResult.newBuilder()
                                .addUnseen(RoiUnseen.newBuilder()
                                        .setArrayId("gone_src")
                                        .setCount(4))
                                .setDeleted(lastPrune.getApply() ? 4 : 0)
                                .build().toByteArray()));
                        break;
                    case "add_tensor":
                        lastCreate = TensorDescriptor.parseFrom(action.getBody());
                        // As the real server answers: the scheme named the
                        // store format and is not part of the tensor's id.
                        listener.onNext(new Result(lastCreate.toBuilder()
                                .setArrayId(lastCreate.getArrayId().replaceFirst("^[a-z]+://", ""))
                                .build().toByteArray()));
                        break;
                    case "register_source":
                        lastRegisterSource = RegisterSource.parseFrom(action.getBody());
                        listener.onNext(new Result(RegisterSourceResult.newBuilder()
                                .setSourceId("registered_abc123")
                                .build().toByteArray()));
                        break;
                    default: // "set_upload_status"
                        lastSetStatus = SetUploadStatus.parseFrom(action.getBody());
                        listener.onNext(new Result(UploadStatus.newBuilder()
                                .setState(lastSetStatus.getState())
                                .setExpectedChunks(chunks.size())
                                .setUploadedChunks(chunks.size())
                                .build().toByteArray()));
                        break;
                }
                listener.onCompleted();
            } catch (Exception error) {
                listener.onError(error);
            }
        }

        private void doAddSource(
                FlightProducer.CallContext context,
                Action action,
                FlightProducer.StreamListener<Result> listener) throws Exception {
            lastAddSource = AddSourceRequest.parseFrom(action.getBody());
            if (addSourceHeartbeats > 2) {
                // The long-walk shape: emit until the client goes away, polling
                // cancellation the way the server's own discovery loop does.
                for (int i = 1; i <= addSourceHeartbeats; i++) {
                    if (context.isCancelled()) {
                        observedCancel = true;
                        return;
                    }
                    emitted.incrementAndGet();
                    listener.onNext(new Result(AddSourceStreamMessage.newBuilder()
                            .setProgress(AddSourceProgress.newBuilder().setAddedCount(i))
                            .build().toByteArray()));
                    Thread.sleep(5);
                }
                return;
            }
            for (int i = 1; i <= 2; i++) {
                listener.onNext(new Result(AddSourceStreamMessage.newBuilder()
                        .setProgress(AddSourceProgress.newBuilder()
                                .setAddedCount(i)
                                .setCurrentPath("/data/plate/" + i))
                        .build().toByteArray()));
            }
            if (!addSourceSendsResult) {
                return;
            }
            listener.onNext(new Result(AddSourceStreamMessage.newBuilder()
                    .setResult(AddSourceResult.newBuilder()
                            .addAllAdded(Arrays.asList("plate_a", "plate_b"))
                            .addAlreadyPresent("plate_c")
                            .addRefreshed("plate_c"))
                    .build().toByteArray()));
        }

        @Override
        public void getStream(
                FlightProducer.CallContext context,
                Ticket ticket,
                FlightProducer.ServerStreamListener listener) {
            TensorTicket parsed;
            try {
                parsed = TensorTicket.parseFrom(ticket.getBytes());
            } catch (Exception error) {
                listener.error(error);
                return;
            }
            if (parsed.hasRoiRead()) {
                lastRoiRead = parsed.getRoiRead();
                serveRois(listener);
                return;
            }
            lastSql = parsed.getCatalogQuery().getSql();
            serveLabelSets(listener);
        }

        /** The label-set query's one projected column. */
        private void serveLabelSets(FlightProducer.ServerStreamListener listener) {
            Schema schema = new Schema(Collections.singletonList(
                    new Field("array_id", FieldType.nullable(ArrowType.Utf8.INSTANCE), null)));
            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
                root.allocateNew();
                VarCharVector ids = (VarCharVector) root.getVector("array_id");
                ids.setSafe(0, "src_ab12/labels/@ome".getBytes(StandardCharsets.UTF_8));
                ids.setSafe(1, "src_ab12/labels/nuclei".getBytes(StandardCharsets.UTF_8));
                root.setRowCount(2);
                listener.start(root);
                listener.putNext();
                listener.completed();
            }
        }

        private void serveRois(FlightProducer.ServerStreamListener listener) {
            List<RoiAnnotation> rois = Arrays.asList(
                    RoiAnnotation.newBuilder()
                            .setRoiId("roi-1")
                            .setArrayId("src_ab12")
                            .setSetName("nuclei")
                            .setRoi(ROI.newBuilder().setPoint(Point.newBuilder().setX(1).setY(2)))
                            .putPlane(0, 12)
                            .setRev(3)
                            .build(),
                    RoiAnnotation.newBuilder()
                            .setRoiId("roi-2")
                            .setArrayId("src_ab12")
                            .setRoi(ROI.newBuilder().setPoint(Point.newBuilder().setX(4).setY(5)))
                            .setDrawnAgainstVersion(ByteString.copyFromUtf8("v1"))
                            .build());
            try (VectorSchemaRoot rows = RoiRowCodec.roisToRoot(rois, allocator)) {
                java.util.Map<String, String> metadata = new java.util.HashMap<>();
                metadata.put("truncated", "True");
                metadata.put("sets", "[{\"set_name\":\"nuclei\",\"count\":17,\"reserved\":false},"
                        + "{\"set_name\":\"@ome\",\"count\":2,\"reserved\":true}]");
                try (VectorSchemaRoot tagged = new VectorSchemaRoot(
                        new Schema(rows.getSchema().getFields(), metadata),
                        rows.getFieldVectors(), rows.getRowCount())) {
                    listener.start(tagged);
                    listener.putNext();
                    listener.completed();
                }
            }
        }

        /**
         * Plan a write the way the server does: chunk bounds on the declared
         * grid, snapped outward to cover the requested slice, each endpoint
         * carrying its bounds <b>relative to the realized origin</b> and an
         * opaque ticket. The ticket here is the absolute bounds, which is all
         * this fake needs to put the chunk where the test can find it.
         */
        @Override
        public FlightInfo getFlightInfo(
                FlightProducer.CallContext context, FlightDescriptor descriptor) {
            TensorReadOption read;
            try {
                read = FlightRequest.parseFrom(descriptor.getCommand()).getTensorRead();
            } catch (InvalidProtocolBufferException error) {
                throw CallStatus.INVALID_ARGUMENT
                        .withDescription("not a FlightRequest").toRuntimeException();
            }
            TensorDescriptor declared = plannedTensor;
            int ndim = declared.getShapeCount();
            long[] shape = new long[ndim];
            long[] chunk = new long[ndim];
            long[] origin = new long[ndim];
            long[] end = new long[ndim];
            for (int axis = 0; axis < ndim; axis++) {
                shape[axis] = declared.getShape(axis);
                chunk[axis] = declared.getChunkShape(axis);
                long from = read.hasSliceHint() ? read.getSliceHint().getStart(axis) : 0;
                long to = read.hasSliceHint() ? read.getSliceHint().getStop(axis) : shape[axis];
                origin[axis] = (from / chunk[axis]) * chunk[axis];
                end[axis] = Math.min(
                        ((to + chunk[axis] - 1) / chunk[axis]) * chunk[axis], shape[axis]);
            }

            List<FlightEndpoint> endpoints = new ArrayList<>();
            long[] start = origin.clone();
            while (true) {
                ChunkBounds.Builder absolute = ChunkBounds.newBuilder();
                ChunkBounds.Builder relative = ChunkBounds.newBuilder();
                for (int axis = 0; axis < ndim; axis++) {
                    long stop = Math.min(start[axis] + chunk[axis], shape[axis]);
                    absolute.addStart(start[axis]).addStop(stop);
                    relative.addStart(start[axis] - origin[axis]).addStop(stop - origin[axis]);
                }
                TensorTicket ticket = TensorTicket.newBuilder()
                        .setChunkId(ByteString.copyFrom(absolute.build().toByteArray()))
                        .build();
                endpoints.add(FlightEndpoint.builder(new Ticket(ticket.toByteArray()))
                        .setAppMetadata(relative.build().toByteArray())
                        .build());
                int axis = ndim - 1;
                for (; axis >= 0; axis--) {
                    start[axis] += chunk[axis];
                    if (start[axis] < end[axis]) {
                        break;
                    }
                    start[axis] = origin[axis];
                }
                if (axis < 0) {
                    break;
                }
            }

            SliceHint.Builder realized = SliceHint.newBuilder();
            for (int axis = 0; axis < ndim; axis++) {
                realized.addStart(origin[axis]).addStop(end[axis]);
            }
            TensorDescriptor response = TensorDescriptor.newBuilder(declared)
                    .setSliceHint(realized)
                    .build();
            return new FlightInfo(
                    new Schema(new ArrayList<>()),
                    FlightDescriptor.command(response.toByteArray()),
                    endpoints,
                    -1,
                    -1);
        }

        @Override
        public Runnable acceptPut(
                FlightProducer.CallContext context,
                FlightStream stream,
                FlightProducer.StreamListener<PutResult> ackStream) {
            return () -> {
                try {
                    PutCommand command = PutCommand.parseFrom(
                            stream.getDescriptor().getCommand());
                    switch (command.getCommandCase()) {
                        case CHUNK_TICKET:
                            acceptChunk(command.getChunkTicket(), stream);
                            break;
                        case ROI_PUT:
                            lastRoiPut = command.getRoiPut();
                            acceptRoiPut(stream, ackStream);
                            break;
                        default:
                            lastRoiDelete = command.getRoiDelete();
                            acceptRoiDelete(stream, ackStream);
                            break;
                    }
                    ackStream.onCompleted();
                } catch (Exception error) {
                    ackStream.onError(error);
                }
            };
        }

        private void acceptChunk(ByteString ticketBytes, FlightStream stream)
                throws InvalidProtocolBufferException {
            // The ticket is this fake's own: the chunk's absolute bounds.
            ChunkBounds bounds = ChunkBounds.parseFrom(
                    TensorTicket.parseFrom(ticketBytes.toByteArray()).getChunkId());
            List<Integer> values = new ArrayList<>();
            while (stream.next()) {
                UInt2Vector data = (UInt2Vector) stream.getRoot().getVector("data");
                for (int row = 0; row < stream.getRoot().getRowCount(); row++) {
                    // UInt2Vector.get hands back a char -- the unsigned 16-bit
                    // value, which is what the tensor declared.
                    values.add((int) data.get(row));
                }
            }
            if (refuseChunks) {
                org.apache.arrow.flight.ErrorFlightMetadata metadata =
                        new org.apache.arrow.flight.ErrorFlightMetadata();
                // No `code`, exactly as upload_manager._refused sends it: both
                // refusal kinds ride one exception class, so the state is the
                // data and the gRPC code is implied by the class.
                metadata.insert("x-biopb-error-bin",
                        ("{\"reason\":\"upload_discarded\",\"source_id\":\""
                                + plannedTensor.getArrayId() + "\",\"state\":\"DISCARDED\","
                                + "\"detail\":\"producer gave up\"}")
                                .getBytes(StandardCharsets.UTF_8));
                throw new org.apache.arrow.flight.CallStatus(
                        org.apache.arrow.flight.FlightStatusCode.CANCELLED, null,
                        "Upload discarded", metadata).toRuntimeException();
            }
            chunks.add(new Chunk(bounds, values));
        }

        private void acceptRoiPut(FlightStream stream, FlightProducer.StreamListener<PutResult> ackStream) {
            RoiPutResult.Builder reply = RoiPutResult.newBuilder();
            int minted = 0;
            while (stream.next()) {
                for (RoiAnnotation roi : RoiRowCodec.roisFromRoot(stream.getRoot())) {
                    putRois.add(roi);
                    reply.addStored(RoiAnnotation.newBuilder(roi)
                            .setRoiId(roi.getRoiId().isEmpty() ? "minted-" + (++minted) : roi.getRoiId())
                            .setRev(roi.getRev() + 1));
                }
            }
            reply(ackStream, reply.build().toByteArray());
        }

        private void acceptRoiDelete(FlightStream stream, FlightProducer.StreamListener<PutResult> ackStream) {
            RoiDeleteResult.Builder reply = RoiDeleteResult.newBuilder();
            while (stream.next()) {
                VarCharVector ids = (VarCharVector) stream.getRoot().getVector("roi_id");
                for (int row = 0; row < stream.getRoot().getRowCount(); row++) {
                    if (ids.isNull(row)) {
                        continue;
                    }
                    String id = new String(ids.get(row), StandardCharsets.UTF_8);
                    deleteIds.add(id);
                    reply.addDeleted(id);
                }
            }
            reply(ackStream, reply.build().toByteArray());
        }

        /**
         * The put's app_metadata. {@code PutResult.metadata} takes the buffer,
         * and {@code onNext} copies it out synchronously, so this owns it only
         * until the ack is on the wire.
         */
        private void reply(FlightProducer.StreamListener<PutResult> ackStream, byte[] body) {
            ArrowBuf buffer = allocator.buffer(body.length);
            buffer.writeBytes(body);
            try (PutResult ack = PutResult.metadata(buffer)) {
                ackStream.onNext(ack);
            }
        }
    }
}
