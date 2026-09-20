package biopb.tensor;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.Assert;
import org.junit.Test;

import com.google.protobuf.ByteString;

import biopb.image.Ellipse;
import biopb.image.Point;
import biopb.image.ROI;
import biopb.image.Rectangle;
import biopb.image.RoiAnnotation;

/**
 * The ROI row schema is the wire contract in both directions, so what this
 * encodes must be exactly what it decodes -- including the parts a
 * column-by-column reimplementation is most likely to drop: the sparse plane
 * pin, an absent {@code drawn_against_version}, and the geometry arm.
 */
public class RoiRowCodecTest {

    @Test
    public void testRoundTripsEveryColumn() {
        RoiAnnotation drawn = RoiAnnotation.newBuilder()
                .setRoiId("roi-1")
                .setArrayId("src_ab12/Image:0")
                .setSetName("nuclei")
                .setLabel("mitotic")
                .setRoi(ROI.newBuilder().setRectangle(Rectangle.newBuilder()
                        .setTopLeft(Point.newBuilder().setX(1.5f).setY(2.5f))
                        .setBottomRight(Point.newBuilder().setX(9.5f).setY(8.25f))))
                .putPlane(0, 12)
                .putPlane(2, 3)
                .setPropsJson("{\"colour\":\"#ff0000\"}")
                .setDrawnAgainstVersion(ByteString.copyFromUtf8("1700000000:4096"))
                .setRev(7)
                .setCreatedAtUnixMs(1_700_000_000_000L)
                .setUpdatedAtUnixMs(1_700_000_001_000L)
                .build();

        try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
                VectorSchemaRoot root = RoiRowCodec.roisToRoot(
                        Collections.singletonList(drawn), allocator)) {
            Assert.assertEquals(1, root.getRowCount());
            List<RoiAnnotation> back = RoiRowCodec.roisFromRoot(root);
            Assert.assertEquals(Collections.singletonList(drawn), back);
        }
    }

    @Test
    public void testAbsentVersionStaysAbsent() {
        // The one nullable column: an annotation drawn against no known version
        // must not come back claiming an empty one, which would read as "drawn
        // against a version that does not match" downstream.
        RoiAnnotation roi = RoiAnnotation.newBuilder()
                .setArrayId("src_ab12")
                .setRoi(ROI.newBuilder().setPoint(Point.newBuilder().setX(4).setY(5)))
                .build();

        try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
                VectorSchemaRoot root = RoiRowCodec.roisToRoot(
                        Collections.singletonList(roi), allocator)) {
            RoiAnnotation back = RoiRowCodec.roisFromRoot(root).get(0);
            Assert.assertFalse(back.hasDrawnAgainstVersion());
            Assert.assertTrue(back.getPlaneMap().isEmpty());
            Assert.assertEquals("", back.getRoiId());
        }
    }

    @Test
    public void testCarriesTheEllipseRotation() {
        // proto3 JSON is the geometry's only representation on the wire, so a
        // float field that round-trips through it is what keeps a fitted
        // ellipse at the angle it was drawn.
        RoiAnnotation roi = RoiAnnotation.newBuilder()
                .setArrayId("src_ab12")
                .setRoi(ROI.newBuilder().setEllipse(Ellipse.newBuilder()
                        .setCenter(Point.newBuilder().setX(10).setY(20))
                        .setRadius(Point.newBuilder().setX(5).setY(3))
                        .setRotation(0.7853982f)))
                .build();

        try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
                VectorSchemaRoot root = RoiRowCodec.roisToRoot(
                        Collections.singletonList(roi), allocator)) {
            RoiAnnotation back = RoiRowCodec.roisFromRoot(root).get(0);
            Assert.assertEquals(0.7853982f, back.getRoi().getEllipse().getRotation(), 1e-7f);
        }
    }

    @Test
    public void testManyRowsKeepTheirOwnPlanePins() {
        // A map column is written per row through one writer; the failure mode
        // worth pinning is rows bleeding into each other.
        List<RoiAnnotation> rois = Arrays.asList(
                RoiAnnotation.newBuilder().setRoiId("a").putPlane(0, 1).build(),
                RoiAnnotation.newBuilder().setRoiId("b").build(),
                RoiAnnotation.newBuilder().setRoiId("c").putPlane(1, 9).putPlane(3, 4).build());

        try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
                VectorSchemaRoot root = RoiRowCodec.roisToRoot(rois, allocator)) {
            List<RoiAnnotation> back = RoiRowCodec.roisFromRoot(root);
            Assert.assertEquals(3, back.size());
            Assert.assertEquals(Collections.singletonMap(0, 1), back.get(0).getPlaneMap());
            Assert.assertTrue(back.get(1).getPlaneMap().isEmpty());
            Assert.assertEquals(2, back.get(2).getPlaneMap().size());
            Assert.assertEquals(Integer.valueOf(9), back.get(2).getPlaneMap().get(1));
            Assert.assertEquals(Integer.valueOf(4), back.get(2).getPlaneMap().get(3));
        }
    }

    @Test
    public void testRefusesAStreamThatIsNotRois() {
        // The message names the request, not an internal error: a caller that
        // pointed a put at the wrong stream can act on it.
        try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
                VectorSchemaRoot root = RoiRowCodec.roiIdsToRoot(
                        Arrays.asList("a", "b"), allocator)) {
            IllegalArgumentException error = Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> RoiRowCodec.roisFromRoot(root));
            Assert.assertTrue(error.getMessage().contains("missing column"));
        }
    }
}
