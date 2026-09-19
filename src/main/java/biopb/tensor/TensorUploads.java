package biopb.tensor;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.PutResult;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.SyncPutListener;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import com.google.protobuf.InvalidProtocolBufferException;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;

import static biopb.tensor.TensorChunkCodec.rowMajorPosition;
import static biopb.tensor.TensorChunkCodec.toLongArray;

/**
 * Tensor declaration and chunk upload over one Flight connection.
 *
 * <p><b>Experimental</b>, with the rest of the upload API.
 *
 * <p>Declare, then fill: {@link #createTensor} returns the server's descriptor
 * for the new source, and that descriptor is what every write takes. The Java
 * twin of {@code biopb.tensor._upload}, minus its dask graph -- an imglib2
 * interval is walked on the calling thread, one chunk per grid cell.
 */
final class TensorUploads {

    private static final Logger LOGGER = Logger.getLogger(TensorUploads.class.getName());

    private final FlightSession session;

    TensorUploads(FlightSession session) {
        this.session = session;
    }

    /** Backs {@link TensorFlightClient#createTensor}; see that method. */
    TensorDescriptor createTensor(
            String sourceName,
            long[] shape,
            String dtype,
            long[] chunkShape,
            List<String> dimLabels,
            String metadataJson) {
        TensorDescriptor.Builder request = TensorDescriptor.newBuilder()
                .setArrayId(sourceName)
                .setDtype(dtype);
        for (long dim : shape) {
            request.addShape(dim);
        }
        for (long chunk : chunkShape == null || chunkShape.length == 0 ? shape : chunkShape) {
            request.addChunkShape(chunk);
        }
        if (dimLabels != null) {
            request.addAllDimLabels(dimLabels);
        }
        if (metadataJson != null && !metadataJson.isEmpty()) {
            request.setMetadataJson(metadataJson);
        }

        Iterator<Result> results = session.doAction(
                new Action("create_tensor", request.build().toByteArray()));
        if (!results.hasNext()) {
            throw new IllegalStateException("create_tensor: server returned no result");
        }
        TensorDescriptor created = parseDescriptor(results.next().getBody());
        LOGGER.info("createTensor: created " + created.getArrayId());
        return created;
    }

    /** Backs {@link TensorFlightClient#uploadArray}; see that method. */
    <T extends NativeType<T> & RealType<T>> Map<String, Object> uploadArray(
            TensorDescriptor descriptor, RandomAccessibleInterval<T> array) {
        long[] shape = toLongArray(descriptor.getShapeList());
        long[] chunkShape = toLongArray(descriptor.getChunkShapeList());
        if (array.numDimensions() != shape.length) {
            throw new IllegalArgumentException("uploadArray: array rank " + array.numDimensions()
                    + " does not match the declared rank " + shape.length + " of " + descriptor.getArrayId());
        }
        for (int axis = 0; axis < shape.length; axis++) {
            if (array.dimension(axis) != shape[axis]) {
                throw new IllegalArgumentException("uploadArray: array shape "
                        + java.util.Arrays.toString(dimensionsOf(array)) + " does not match the declared shape "
                        + java.util.Arrays.toString(shape) + " of " + descriptor.getArrayId());
            }
        }
        // The chunk column is typed by the DECLARED dtype, so a mismatch would
        // narrow silently -- a float array into a uint16 tensor truncates every
        // value, and the upload succeeds.
        String declared = normalizeDtype(descriptor.getDtype());
        String actual = normalizeDtype(numpyDtype(array.getType()));
        if (!declared.equals(actual)) {
            throw new IllegalArgumentException("uploadArray: array dtype " + actual
                    + " does not match the declared dtype " + declared + " of " + descriptor.getArrayId());
        }
        // An all-zero block of a label set is not sent at all: the sidecar's fill
        // value already reads as background, so one labelled frame of a thousand
        // costs one frame (biopb/biopb#1059).
        boolean skipEmpty = isLabelSet(descriptor.getArrayId());

        long[] start = new long[shape.length];
        while (true) {
            long[] stop = new long[shape.length];
            for (int axis = 0; axis < shape.length; axis++) {
                stop[axis] = Math.min(start[axis] + chunkShape[axis], shape[axis]);
            }
            ChunkBounds bounds = boundsOf(start, stop);
            if (!skipEmpty || !isAllZero(array, start, stop)) {
                uploadChunk(descriptor, bounds, array);
            }
            if (!advance(start, chunkShape, shape)) {
                break;
            }
        }
        // Sealing is what marks the source complete, so a whole-array upload does
        // it on the caller's behalf -- it is the one caller that knows, from
        // having written every block itself, that there is nothing more to send.
        return finishUpload(descriptor);
    }

    /** Backs {@link TensorFlightClient#uploadChunk}; see that method. */
    <T extends NativeType<T> & RealType<T>> void uploadChunk(
            TensorDescriptor descriptor, ChunkBounds bounds, RandomAccessibleInterval<T> source) {
        long[] start = toLongArray(bounds.getStartList());
        long[] stop = toLongArray(bounds.getStopList());
        if (start.length != stop.length || start.length != source.numDimensions()) {
            throw new IllegalArgumentException("uploadChunk: bounds rank " + start.length
                    + " does not match the source rank " + source.numDimensions());
        }

        PutCommand command = PutCommand.newBuilder()
                .setChunk(ChunkUpload.newBuilder()
                        .setSourceId(descriptor.getArrayId())
                        .setBounds(bounds)
                        .build())
                .build();

        BufferAllocator allocator = session.allocator();
        // The root takes the column; closing it is what frees the block.
        FieldVector data = encodeBlock(descriptor.getDtype(), source, start, stop, allocator);
        try (VectorSchemaRoot root = VectorSchemaRoot.of(data);
                SyncPutListener reply = new SyncPutListener()) {
            root.setRowCount(data.getValueCount());
            FlightClient.ClientStreamListener writer = session.startPut(
                    FlightDescriptor.command(command.toByteArray()), root, reply);
            writer.putNext();
            writer.completed();
            // Drain the ack before asking for the result: a refusal (an upload
            // that is sealed or discarded) arrives on this stream, and it is
            // what makes this call raise rather than return quietly. A chunk put
            // carries no app_metadata, so a null here is the ordinary case.
            PutResult ack = reply.read();
            if (ack != null) {
                ack.close();
            }
            writer.getResult();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("uploadChunk interrupted", error);
        } catch (java.util.concurrent.ExecutionException error) {
            throw FlightSession.mapped(error.getCause());
        } catch (org.apache.arrow.flight.FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    /** Backs {@link TensorFlightClient#finishUpload}; see that method. */
    Map<String, Object> finishUpload(TensorDescriptor descriptor) {
        FinishUpload request = FinishUpload.newBuilder()
                .setSourceId(descriptor.getArrayId())
                .build();
        Iterator<Result> results = session.doAction(new Action("finish", request.toByteArray()));
        if (!results.hasNext()) {
            throw new IllegalStateException("finish: server returned no result");
        }
        UploadStatus status;
        try {
            status = UploadStatus.parseFrom(results.next().getBody());
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalStateException("finish: server returned no UploadStatus", error);
        }
        LOGGER.info("finishUpload: sealed " + descriptor.getArrayId());
        return statusMap(descriptor.getArrayId(), status);
    }

    /** The {@code get_upload_status} shape, so a seal and a poll agree. */
    static Map<String, Object> statusMap(String sourceId, UploadStatus status) {
        Map<String, Object> out = new HashMap<>();
        out.put("source_id", sourceId);
        out.put("state", status.getState().name());
        out.put("expected_chunks", (double) status.getExpectedChunks());
        out.put("uploaded_chunks", (double) status.getUploadedChunks());
        out.put("reason", status.getReason());
        return out;
    }

    /**
     * Whether {@code arrayId} names a label set rather than a source of its own.
     *
     * <p>The one upload kind whose unwritten chunks are meaningful: a label set
     * is a zarr with fill value 0, so a chunk that never arrives reads back as
     * background and skipping it is free (biopb/biopb#1059). A {@code cache:}
     * source answers a read of an unwritten chunk with "holds no chunk", so the
     * skip must never be a general behaviour -- hence the check on the shape of
     * the id rather than a flag the caller could set on anything.
     */
    static boolean isLabelSet(String arrayId) {
        int slash = arrayId.indexOf('/');
        String head = slash < 0 ? arrayId : arrayId.substring(0, slash);
        return head.indexOf(':') < 0 && arrayId.contains("/labels/");
    }

    /** The numpy dtype string an imglib2 type uploads as. */
    static String numpyDtype(Object type) {
        if (type instanceof net.imglib2.type.numeric.integer.UnsignedByteType) return "|u1";
        if (type instanceof net.imglib2.type.numeric.integer.ByteType) return "|i1";
        if (type instanceof net.imglib2.type.numeric.integer.UnsignedShortType) return "<u2";
        if (type instanceof net.imglib2.type.numeric.integer.ShortType) return "<i2";
        if (type instanceof net.imglib2.type.numeric.integer.UnsignedIntType) return "<u4";
        if (type instanceof net.imglib2.type.numeric.integer.IntType) return "<i4";
        if (type instanceof net.imglib2.type.numeric.integer.UnsignedLongType) return "<u8";
        if (type instanceof net.imglib2.type.numeric.integer.LongType) return "<i8";
        if (type instanceof net.imglib2.type.numeric.real.DoubleType) return "<f8";
        if (type instanceof net.imglib2.type.numeric.real.FloatType) return "<f4";
        throw new IllegalArgumentException(
                "No numpy dtype for imglib2 type " + type.getClass().getName()
                        + "; declare the dtype explicitly");
    }

    private static long[] dimensionsOf(RandomAccessibleInterval<?> array) {
        long[] dims = new long[array.numDimensions()];
        array.dimensions(dims);
        return dims;
    }

    private static ChunkBounds boundsOf(long[] start, long[] stop) {
        ChunkBounds.Builder bounds = ChunkBounds.newBuilder();
        for (int axis = 0; axis < start.length; axis++) {
            bounds.addStart(start[axis]);
            bounds.addStop(stop[axis]);
        }
        return bounds.build();
    }

    /** Step {@code start} to the next cell of the chunk grid; false when past the last. */
    private static boolean advance(long[] start, long[] chunkShape, long[] shape) {
        for (int axis = start.length - 1; axis >= 0; axis--) {
            start[axis] += chunkShape[axis];
            if (start[axis] < shape[axis]) {
                return true;
            }
            start[axis] = 0;
        }
        return false;
    }

    private static <T extends RealType<T>> boolean isAllZero(
            RandomAccessibleInterval<T> array, long[] start, long[] stop) {
        RandomAccess<T> access = array.randomAccess();
        long[] extents = extentsOf(start, stop);
        long count = elementCount(extents);
        long[] local = new long[extents.length];
        long[] global = new long[extents.length];
        for (long index = 0; index < count; index++) {
            rowMajorPosition(index, extents, local);
            for (int axis = 0; axis < extents.length; axis++) {
                global[axis] = start[axis] + local[axis];
            }
            access.setPosition(global);
            if (access.get().getRealDouble() != 0.0) {
                return false;
            }
        }
        return true;
    }

    /**
     * One chunk's elements as the typed Arrow column the server's
     * {@code write_chunk} reads: the flat values in row-major (C) order, typed
     * by the descriptor's dtype -- which is what the store is declared as, so a
     * chunk cannot land under a dtype the tensor does not have.
     *
     * <p>Integer types are read through {@code getIntegerLong()}, not the
     * double accessor: a 64-bit label id is exact here and would not survive
     * the widening.
     */
    private static <T extends RealType<T>> FieldVector encodeBlock(
            String dtype,
            RandomAccessibleInterval<T> source,
            long[] start,
            long[] stop,
            BufferAllocator allocator) {
        long[] extents = extentsOf(start, stop);
        int count = Math.toIntExact(elementCount(extents));
        FieldVector vector = newVector(dtype, allocator);
        vector.setInitialCapacity(count);
        vector.allocateNew();

        RandomAccess<T> access = source.randomAccess();
        long[] local = new long[extents.length];
        long[] global = new long[extents.length];
        for (int index = 0; index < count; index++) {
            rowMajorPosition(index, extents, local);
            for (int axis = 0; axis < extents.length; axis++) {
                global[axis] = start[axis] + local[axis];
            }
            access.setPosition(global);
            setElement(vector, index, access.get());
        }
        vector.setValueCount(count);
        return vector;
    }

    private static FieldVector newVector(String dtype, BufferAllocator allocator) {
        switch (normalizeDtype(dtype)) {
            case "u1": return new UInt1Vector("data", allocator);
            case "i1": return new TinyIntVector("data", allocator);
            case "u2": return new UInt2Vector("data", allocator);
            case "i2": return new SmallIntVector("data", allocator);
            case "u4": return new UInt4Vector("data", allocator);
            case "i4": return new IntVector("data", allocator);
            case "u8": return new UInt8Vector("data", allocator);
            case "i8": return new BigIntVector("data", allocator);
            case "f8": return new Float8Vector("data", allocator);
            case "f4": return new Float4Vector("data", allocator);
            default:
                throw new IllegalArgumentException("Cannot upload dtype '" + dtype + "'");
        }
    }

    private static <T extends RealType<T>> void setElement(FieldVector vector, int index, T value) {
        if (vector instanceof Float4Vector) {
            ((Float4Vector) vector).set(index, value.getRealFloat());
        } else if (vector instanceof Float8Vector) {
            ((Float8Vector) vector).set(index, value.getRealDouble());
        } else {
            long element = value instanceof IntegerType
                    ? ((IntegerType<?>) value).getIntegerLong()
                    : (long) value.getRealDouble();
            if (vector instanceof UInt1Vector) {
                ((UInt1Vector) vector).set(index, (int) element);
            } else if (vector instanceof TinyIntVector) {
                ((TinyIntVector) vector).set(index, (int) element);
            } else if (vector instanceof UInt2Vector) {
                ((UInt2Vector) vector).set(index, (int) element);
            } else if (vector instanceof SmallIntVector) {
                ((SmallIntVector) vector).set(index, (int) element);
            } else if (vector instanceof UInt4Vector) {
                ((UInt4Vector) vector).set(index, (int) element);
            } else if (vector instanceof IntVector) {
                ((IntVector) vector).set(index, (int) element);
            } else if (vector instanceof UInt8Vector) {
                ((UInt8Vector) vector).set(index, element);
            } else {
                ((BigIntVector) vector).set(index, element);
            }
        }
    }

    /** A numpy dtype string reduced to its kind+size, dropping the byte-order mark. */
    private static String normalizeDtype(String dtype) {
        String text = dtype == null ? "" : dtype.trim().toLowerCase();
        if (!text.isEmpty()) {
            char first = text.charAt(0);
            if (first == '<' || first == '>' || first == '|' || first == '=') {
                text = text.substring(1);
            }
        }
        switch (text) {
            case "uint8": return "u1";
            case "int8": return "i1";
            case "uint16": return "u2";
            case "int16": return "i2";
            case "uint32": return "u4";
            case "int32": return "i4";
            case "uint64": return "u8";
            case "int64": return "i8";
            case "float32": return "f4";
            case "float64": return "f8";
            default: return text;
        }
    }

    private static long[] extentsOf(long[] start, long[] stop) {
        long[] extents = new long[start.length];
        for (int axis = 0; axis < start.length; axis++) {
            extents[axis] = stop[axis] - start[axis];
        }
        return extents;
    }

    private static long elementCount(long[] extents) {
        long count = 1L;
        for (long extent : extents) {
            count *= extent;
        }
        return count;
    }

    private static TensorDescriptor parseDescriptor(byte[] bytes) {
        try {
            return TensorDescriptor.parseFrom(bytes);
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalStateException("create_tensor: server returned no TensorDescriptor", error);
        }
    }
}
