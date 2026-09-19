package biopb.tensor;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.impl.UnionMapWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;

import biopb.image.ROI;
import biopb.image.RoiAnnotation;

/**
 * The ROI row schema: how annotations travel on the {@code roi} flight.
 *
 * <p>One Arrow row per {@link RoiAnnotation}, the same columns in both
 * directions -- what the server streams on a DoGet is what it accepts on a
 * DoPut -- so a client that can read a set can write one back. The geometry is
 * the {@code biopb.image.ROI} proto as canonical proto3 JSON text, which is
 * also how the server's catalog stores it.
 *
 * <p>The Java twin of {@code biopb.image._roi_rows}; that module is the wire
 * contract's definition, shared there between the Python SDK and the server,
 * and this must track it column for column.
 */
final class RoiRowCodec {

    private RoiRowCodec() {}

    /** The columns an annotation row carries, in order. */
    static final Schema ROI_ROW_SCHEMA = new Schema(Arrays.asList(
            utf8("roi_id"),
            utf8("array_id"),
            utf8("set_name"),
            utf8("label"),
            // biopb.image.ROI as proto3 JSON.
            utf8("geometry"),
            // Sparse plane pin, wire axis index -> index on that axis.
            planeField(),
            utf8("props_json"),
            // Null when the annotation was not drawn against a known version.
            new Field("drawn_against_version", FieldType.nullable(ArrowType.Binary.INSTANCE), null),
            int64("rev"),
            int64("created_at_unix_ms"),
            int64("updated_at_unix_ms")));

    /** The stream of a {@code RoiDelete} put: which ids to remove. */
    static final Schema ROI_ID_SCHEMA = new Schema(Collections.singletonList(utf8("roi_id")));

    private static Field utf8(String name) {
        return new Field(name, FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
    }

    private static Field int64(String name) {
        return new Field(name, FieldType.nullable(new ArrowType.Int(64, true)), null);
    }

    /**
     * {@code map<uint32, uint32>}, laid out as Arrow's canonical map: a
     * non-nullable {@code entries} struct of a non-nullable {@code key} and a
     * {@code value}. The names matter -- pyarrow's own {@code map_} uses them,
     * and this stream is read by a pyarrow server.
     */
    private static Field planeField() {
        ArrowType uint32 = new ArrowType.Int(32, false);
        // A map's key is non-nullable and its value is not: that is Arrow's
        // canonical layout, and pyarrow's `map_` builds exactly this, so a
        // stream built here compares equal to one the server built.
        Field entries = new Field("entries", FieldType.notNullable(ArrowType.Struct.INSTANCE),
                Arrays.asList(
                        new Field("key", FieldType.notNullable(uint32), null),
                        new Field("value", FieldType.nullable(uint32), null)));
        return new Field("plane", FieldType.nullable(new ArrowType.Map(false)),
                Collections.singletonList(entries));
    }

    /**
     * Annotations as rows of {@link #ROI_ROW_SCHEMA}; the <b>caller must
     * close</b> the returned root.
     */
    static VectorSchemaRoot roisToRoot(List<RoiAnnotation> rois, BufferAllocator allocator) {
        VectorSchemaRoot root = VectorSchemaRoot.create(ROI_ROW_SCHEMA, allocator);
        root.allocateNew();
        for (int row = 0; row < rois.size(); row++) {
            RoiAnnotation roi = rois.get(row);
            setUtf8(root, "roi_id", row, roi.getRoiId());
            setUtf8(root, "array_id", row, roi.getArrayId());
            setUtf8(root, "set_name", row, roi.getSetName());
            setUtf8(root, "label", row, roi.getLabel());
            setUtf8(root, "geometry", row, geometryJson(roi.getRoi()));
            setUtf8(root, "props_json", row, roi.getPropsJson());
            VarBinaryVector version = (VarBinaryVector) root.getVector("drawn_against_version");
            if (roi.hasDrawnAgainstVersion()) {
                version.setSafe(row, roi.getDrawnAgainstVersion().toByteArray());
            } else {
                version.setNull(row);
            }
            ((BigIntVector) root.getVector("rev")).setSafe(row, roi.getRev());
            ((BigIntVector) root.getVector("created_at_unix_ms")).setSafe(row, roi.getCreatedAtUnixMs());
            ((BigIntVector) root.getVector("updated_at_unix_ms")).setSafe(row, roi.getUpdatedAtUnixMs());
            writePlane((MapVector) root.getVector("plane"), row, roi.getPlaneMap());
        }
        root.setRowCount(rois.size());
        return root;
    }

    /** One {@code roi_id} column; the <b>caller must close</b> the returned root. */
    static VectorSchemaRoot roiIdsToRoot(List<String> roiIds, BufferAllocator allocator) {
        VectorSchemaRoot root = VectorSchemaRoot.create(ROI_ID_SCHEMA, allocator);
        root.allocateNew();
        for (int row = 0; row < roiIds.size(); row++) {
            setUtf8(root, "roi_id", row, roiIds.get(row));
        }
        root.setRowCount(roiIds.size());
        return root;
    }

    /**
     * The inverse of {@link #roisToRoot}, over one loaded batch.
     *
     * @throws IllegalArgumentException for a stream that is not in the row
     *         schema, or whose geometry is not a {@code biopb.image.ROI}, so a
     *         caller can report the request rather than an internal error.
     */
    static List<RoiAnnotation> roisFromRoot(VectorSchemaRoot root) {
        List<String> missing = new ArrayList<>();
        for (Field field : ROI_ROW_SCHEMA.getFields()) {
            if (root.getVector(field.getName()) == null) {
                missing.add(field.getName());
            }
        }
        if (!missing.isEmpty()) {
            throw new IllegalArgumentException(
                    "ROI rows are missing column(s) " + missing + "; expected the ROI row schema "
                            + ROI_ROW_SCHEMA.getFields());
        }
        List<RoiAnnotation> out = new ArrayList<>(root.getRowCount());
        for (int row = 0; row < root.getRowCount(); row++) {
            RoiAnnotation.Builder roi = RoiAnnotation.newBuilder()
                    .setRoiId(utf8At(root, "roi_id", row))
                    .setArrayId(utf8At(root, "array_id", row))
                    .setSetName(utf8At(root, "set_name", row))
                    .setLabel(utf8At(root, "label", row))
                    .setPropsJson(utf8At(root, "props_json", row))
                    .setRev(int64At(root, "rev", row))
                    .setCreatedAtUnixMs(int64At(root, "created_at_unix_ms", row))
                    .setUpdatedAtUnixMs(int64At(root, "updated_at_unix_ms", row));
            parseGeometry(utf8At(root, "geometry", row), roi);
            VarBinaryVector version = (VarBinaryVector) root.getVector("drawn_against_version");
            if (!version.isNull(row)) {
                roi.setDrawnAgainstVersion(ByteString.copyFrom(version.get(row)));
            }
            roi.putAllPlane(readPlane((MapVector) root.getVector("plane"), row));
            out.add(roi.build());
        }
        return out;
    }

    private static String geometryJson(ROI roi) {
        try {
            return JsonFormat.printer().omittingInsignificantWhitespace().print(roi);
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("ROI geometry cannot be encoded as proto3 JSON", error);
        }
    }

    private static void parseGeometry(String json, RoiAnnotation.Builder roi) {
        String text = json == null || json.isEmpty() ? "{}" : json;
        try {
            JsonFormat.parser().merge(text, roi.getRoiBuilder());
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("ROI "
                    + (roi.getRoiId().isEmpty() ? "<new>" : roi.getRoiId())
                    + ": geometry is not a biopb.image.ROI: " + error.getMessage(), error);
        }
    }

    private static void writePlane(MapVector plane, int row, Map<Integer, Integer> pins) {
        UnionMapWriter writer = plane.getWriter();
        writer.setPosition(row);
        writer.startMap();
        for (Map.Entry<Integer, Integer> pin : pins.entrySet()) {
            writer.startEntry();
            writer.key().uInt4().writeUInt4(pin.getKey());
            writer.value().uInt4().writeUInt4(pin.getValue());
            writer.endEntry();
        }
        writer.endMap();
    }

    /**
     * One row's plane pins.
     *
     * <p>{@link MapVector#getObject} hands back a list of {@code {key, value}}
     * maps, the generic shape Arrow gives every map vector; the proto wants
     * {@code axis -> index}.
     */
    private static Map<Integer, Integer> readPlane(MapVector plane, int row) {
        Map<Integer, Integer> pins = new java.util.LinkedHashMap<>();
        if (plane.isNull(row)) {
            return pins;
        }
        Object entries = plane.getObject(row);
        if (!(entries instanceof List)) {
            return pins;
        }
        for (Object entry : (List<?>) entries) {
            if (!(entry instanceof Map)) {
                continue;
            }
            Map<?, ?> pin = (Map<?, ?>) entry;
            Object key = pin.get(MapVector.KEY_NAME);
            Object value = pin.get(MapVector.VALUE_NAME);
            if (key instanceof Number && value instanceof Number) {
                pins.put(((Number) key).intValue(), ((Number) value).intValue());
            }
        }
        return pins;
    }

    private static void setUtf8(VectorSchemaRoot root, String column, int row, String value) {
        ((VarCharVector) root.getVector(column)).setSafe(row, value.getBytes(StandardCharsets.UTF_8));
    }

    private static String utf8At(VectorSchemaRoot root, String column, int row) {
        FieldVector vector = root.getVector(column);
        return vector.isNull(row) ? "" : new String(((VarCharVector) vector).get(row), StandardCharsets.UTF_8);
    }

    private static long int64At(VectorSchemaRoot root, String column, int row) {
        FieldVector vector = root.getVector(column);
        return vector.isNull(row) ? 0L : ((BigIntVector) vector).get(row);
    }
}
