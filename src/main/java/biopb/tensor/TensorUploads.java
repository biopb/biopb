package biopb.tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
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

import com.google.protobuf.ByteString;
import com.google.protobuf.FieldMask;
import com.google.protobuf.InvalidProtocolBufferException;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;

import static biopb.tensor.TensorChunkCodec.advanceRowMajor;
import static biopb.tensor.TensorChunkCodec.cellCount;
import static biopb.tensor.TensorChunkCodec.normalizeDtype;
import static biopb.tensor.TensorChunkCodec.parseChunkBounds;
import static biopb.tensor.TensorChunkCodec.toLongArray;

/**
 * Tensor declaration and chunk upload over one Flight connection.
 *
 * <p><b>Experimental</b>, with the rest of the upload API.
 *
 * <p><b>An upload adds a tensor to a source that already exists</b> and
 * creates none. {@link #addTensor} returns the server's descriptor for the new
 * tensor, and that descriptor is what every write takes. The scheme on an {@code array_id} names the store format and nothing
 * else; the answered id carries none. The Java
 * twin of {@code biopb.tensor._upload}, minus its dask graph -- an imglib2
 * interval is walked on the calling thread, one chunk per planned endpoint,
 * each put finished before the next is encoded. Python instead stores the whole
 * array through dask with {@code lock=False} and the chunks go up concurrently,
 * which the server is built for (it counts arrivals into a set keyed by chunk
 * id). Closing that gap is biopb/biopb#1073.
 *
 * <p><b>What a chunk is, is the server's.</b> A write plans through
 * {@code GetFlightInfo} exactly as a read does and sends back the endpoint's
 * ticket, so there is no grid arithmetic on this side to keep in step with the
 * server's ({@code java-tensor-v2.md} parity rule).
 */
final class TensorUploads {

    private static final Logger LOGGER = Logger.getLogger(TensorUploads.class.getName());

    private final FlightSession session;

    TensorUploads(FlightSession session) {
        this.session = session;
    }

    /**
     * Run a single-result {@code doAction} and hand back its body.
     *
     * <p>The upload actions all want the same four lines -- dispatch,
     * refuse an empty stream, take the one result -- and each spelled them
     * again, so a change to how an empty stream reads had three places to
     * reach. The proto parse stays at the call site, because only it knows
     * which message it asked for.
     *
     * @throws IllegalStateException the server answered with no result at all
     */
    private byte[] actionOneResult(String type, byte[] body) {
        Iterator<Result> results = session.doAction(new Action(type, body));
        if (!results.hasNext()) {
            throw new IllegalStateException(type + ": server returned no result");
        }
        return results.next().getBody();
    }

    /** Backs {@link TensorFlightClient#addTensor}; see that method. */
    TensorDescriptor addTensor(
            String arrayId,
            long[] shape,
            String dtype,
            long[] chunkShape,
            List<String> dimLabels,
            String metadataJson) {
        return addTensor(arrayId, shape, dtype, chunkShape, dimLabels, metadataJson, null);
    }

    /** Backs {@link TensorFlightClient#addTensor}; see that method. */
    TensorDescriptor addTensor(
            String arrayId,
            long[] shape,
            String dtype,
            long[] chunkShape,
            List<String> dimLabels,
            String metadataJson,
            Integer ttlSeconds) {
        TensorDescriptor.Builder request = TensorDescriptor.newBuilder()
                .setArrayId(arrayId)
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
        if (ttlSeconds != null) {
            // Set only when asked, so leaving it null means "unset" and not
            // "zero" -- which the server refuses rather than reading as
            // "expire immediately".
            request.setTtlSeconds(ttlSeconds);
        }

        TensorDescriptor created = TensorChunkCodec.parseDescriptor(
                actionOneResult("add_tensor", request.build().toByteArray()));
        LOGGER.info("addTensor: added " + created.getArrayId());
        return created;
    }

    /** Backs {@link TensorFlightClient#uploadArray}; see that method. */
    <T extends NativeType<T> & RealType<T>> Map<String, Object> uploadArray(
            TensorDescriptor descriptor, RandomAccessibleInterval<T> array) {
        long[] shape = toLongArray(descriptor.getShapeList());
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
        // An all-zero block of a label set is not sent at all: its unwritten
        // chunks read back as background, so one labelled frame of a thousand
        // costs one frame (biopb/biopb#1059).
        boolean skipEmpty = isLabelSet(descriptor.getArrayId());

        // One plan for the whole upload; each entry is one chunk, with the
        // ticket that names it.
        for (PlannedChunk chunk : planWrite(descriptor.getArrayId(), null)) {
            putChunk(descriptor, chunk.ticket, chunk.bounds, array, skipEmpty);
        }
        // Publishing is what marks the source complete, so a whole-array upload
        // does it on the caller's behalf -- it is the one caller that knows,
        // from having written every block itself, that there is nothing more to
        // send.
        return setUploadStatus(descriptor.getArrayId(), UploadStatus.State.READY, "");
    }

    /** Backs {@link TensorFlightClient#uploadChunk}; see that method. */
    <T extends NativeType<T> & RealType<T>> void uploadChunk(
            TensorDescriptor descriptor, ChunkBounds bounds, RandomAccessibleInterval<T> source) {
        putChunk(descriptor, plannedTicket(descriptor.getArrayId(), bounds), bounds, source, false);
    }

    /**
     * One endpoint of a write plan: where it goes, and the ticket that says so.
     *
     * <p>{@code bounds} is in the <b>tensor's</b> coordinates. An endpoint's
     * {@code app_metadata} states them relative to the realized region instead
     * -- that is what a reader wants, since it is assembling an array of just
     * that region -- so {@link #planWrite} shifts them back by the realized
     * origin.
     */
    private static final class PlannedChunk {
        final ChunkBounds bounds;
        final byte[] ticket;

        PlannedChunk(ChunkBounds bounds, byte[] ticket) {
            this.bounds = bounds;
            this.ticket = ticket;
        }
    }

    /**
     * The chunks a write must send, as the server plans them.
     *
     * <p>The same {@code GetFlightInfo} a read makes, with the {@code
     * endpoints} mask and no {@code scale_hint}: the mask is the plan alone
     * because the rest of a describe costs I/O a write has no use for. A
     * {@code sliceHint} plans only the box that will be written, snapped
     * outward to the server's grid. Answered while the tensor is still PENDING
     * -- planning is a metadata read -- and idempotent, so an interrupted
     * upload re-plans to resume.
     */
    private List<PlannedChunk> planWrite(String arrayId, SliceHint sliceHint) {
        TensorReadOption.Builder read = TensorReadOption.newBuilder()
                .setArrayId(arrayId)
                .setFields(FieldMask.newBuilder().addPaths("endpoints").build());
        if (sliceHint != null) {
            read.setSliceHint(sliceHint);
        }
        FlightRequest request = FlightRequest.newBuilder().setTensorRead(read.build()).build();
        FlightInfo info = session.getInfo(FlightDescriptor.command(request.toByteArray()));
        if (info.getEndpoints().isEmpty()) {
            throw new IllegalArgumentException(
                    "upload: the server planned no chunks for " + arrayId);
        }
        // The realized region the plan snapped to; its start is the origin the
        // endpoints' bounds are stated against.
        List<Long> origin = TensorChunkCodec.descriptorOf(info).getSliceHint().getStartList();
        List<PlannedChunk> plan = new ArrayList<>();
        for (FlightEndpoint endpoint : info.getEndpoints()) {
            ChunkBounds relative = parseChunkBounds(endpoint.getAppMetadata());
            ChunkBounds.Builder absolute = ChunkBounds.newBuilder();
            for (int axis = 0; axis < relative.getStartCount(); axis++) {
                long shift = origin.isEmpty() ? 0 : origin.get(axis);
                absolute.addStart(shift + relative.getStart(axis));
                absolute.addStop(shift + relative.getStop(axis));
            }
            plan.add(new PlannedChunk(absolute.build(), endpoint.getTicket().getBytes()));
        }
        return plan;
    }

    /**
     * The ticket for the one chunk at {@code bounds}, or a refusal naming the
     * grid.
     *
     * <p>The server snaps a slice outward, so a plan of one endpoint whose
     * bounds are the ones asked for is the only proof that {@code bounds} is a
     * chunk. Refused here rather than sent, because a write of part of a chunk
     * has nowhere to land: the id the planner mints covers the whole cell.
     */
    private byte[] plannedTicket(String arrayId, ChunkBounds bounds) {
        List<PlannedChunk> plan = planWrite(arrayId, SliceHint.newBuilder()
                .addAllStart(bounds.getStartList())
                .addAllStop(bounds.getStopList())
                .build());
        if (plan.size() != 1 || !plan.get(0).bounds.equals(bounds)) {
            StringBuilder snapped = new StringBuilder();
            for (int i = 0; i < Math.min(4, plan.size()); i++) {
                ChunkBounds cell = plan.get(i).bounds;
                snapped.append(i == 0 ? "" : ", ").append(cell.getStartList())
                        .append('-').append(cell.getStopList());
            }
            throw new IllegalArgumentException("uploadChunk: " + bounds.getStartList() + "-"
                    + bounds.getStopList() + " is not one chunk of " + arrayId
                    + "; the server's grid puts it in " + snapped
                    + (plan.size() > 4 ? " ..." : "") + ". Write a chunk of that grid, "
                    + "or use uploadArray.");
        }
        return plan.get(0).ticket;
    }

    /**
     * Send one planned chunk. With {@code skipEmpty} the block is encoded and
     * then dropped unsent if every element was zero.
     *
     * <p>Encoding first and deciding after is what keeps the emptiness test and
     * the upload one traversal rather than two -- and, more to the point, keeps
     * {@link #positionOf}'s handling of a cropped view's min in one loop rather
     * than in two that have to agree.
     */
    private <T extends NativeType<T> & RealType<T>> void putChunk(
            TensorDescriptor descriptor, byte[] ticket, ChunkBounds bounds,
            RandomAccessibleInterval<T> source, boolean skipEmpty) {
        long[] start = toLongArray(bounds.getStartList());
        long[] stop = toLongArray(bounds.getStopList());
        if (start.length != stop.length || start.length != source.numDimensions()) {
            throw new IllegalArgumentException("uploadChunk: bounds rank " + start.length
                    + " does not match the source rank " + source.numDimensions());
        }

        PutCommand command = PutCommand.newBuilder()
                .setChunkTicket(ByteString.copyFrom(ticket))
                .build();

        BufferAllocator allocator = session.allocator();
        // The root takes the column; closing it is what frees the block.
        Block block = encodeBlock(descriptor.getDtype(), source, start, stop, allocator);
        FieldVector data = block.data;
        if (skipEmpty && !block.anyNonZero) {
            data.close();
            return;
        }
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

    /** Backs {@link TensorFlightClient#setUploadStatus}; see that method. */
    Map<String, Object> setUploadStatus(String arrayId, UploadStatus.State state, String reason) {
        SetUploadStatus request = SetUploadStatus.newBuilder()
                .setArrayId(arrayId)
                .setState(state)
                .setReason(reason == null ? "" : reason)
                .build();
        UploadStatus status;
        try {
            status = UploadStatus.parseFrom(
                    actionOneResult("set_upload_status", request.toByteArray()));
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalStateException(
                    "set_upload_status: server returned no UploadStatus", error);
        }
        LOGGER.info("setUploadStatus: " + arrayId + " -> " + state.name());
        return statusMap(arrayId, status);
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
     * <p>The one upload kind whose unwritten chunks are meaningful <i>by
     * declaration</i>: a label set is a zarr with fill value 0, so a chunk that
     * never arrives reads back as background and skipping it is free
     * (biopb/biopb#1059). Every published upload now reads its gaps as zeros, so
     * skipping would be safe for the other kinds too -- but it would also stop
     * reporting them: an all-zero array would upload nothing at all and land
     * READY with {@code uploaded_chunks} at 0.
     */
    static boolean isLabelSet(String arrayId) {
        return TensorFlightClient.sourceIdFromArrayId(arrayId).indexOf(':') < 0
                && arrayId.contains("/" + TensorFlightClient.LABELS_SEGMENT + "/");
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

    /**
     * Where tensor coordinate {@code start + local} sits in {@code source}.
     *
     * <p>The interval's <b>min is the tensor's origin</b>, so a cropped view
     * ({@code Views.interval}) uploads its own content rather than whatever
     * lies at the underlying image's 0. A view's random access is unbounded, so
     * ignoring the min reads real pixels from the wrong place and the upload
     * stores them without complaint.
     */
    private static void positionOf(
            RandomAccessibleInterval<?> source, long[] start, long[] local, long[] into) {
        for (int axis = 0; axis < into.length; axis++) {
            into[axis] = source.min(axis) + start[axis] + local[axis];
        }
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
    private static <T extends RealType<T>> Block encodeBlock(
            String dtype,
            RandomAccessibleInterval<T> source,
            long[] start,
            long[] stop,
            BufferAllocator allocator) {
        long[] extents = extentsOf(start, stop);
        int count = Math.toIntExact(cellCount(extents));
        Column column = Column.of(dtype);
        FieldVector vector = column.newVector(allocator);
        vector.setInitialCapacity(count);
        vector.allocateNew();

        RandomAccess<T> access = source.randomAccess();
        long[] local = new long[extents.length];
        long[] global = new long[extents.length];
        boolean anyNonZero = false;
        for (int index = 0; index < count; index++) {
            positionOf(source, start, local, global);
            access.setPosition(global);
            anyNonZero |= column.set(vector, index, access.get());
            advanceRowMajor(local, extents);
        }
        vector.setValueCount(count);
        return new Block(vector, anyNonZero);
    }

    /** An encoded chunk, and whether any of it was non-zero. */
    private static final class Block {
        final FieldVector data;
        final boolean anyNonZero;

        Block(FieldVector data, boolean anyNonZero) {
            this.data = data;
            this.anyNonZero = anyNonZero;
        }
    }

    /**
     * The Arrow column one dtype uploads as: how to build it and how to write
     * one element into it.
     *
     * <p>Resolved once per block. The alternative -- re-deriving the type from
     * the vector's class per element -- put a ladder of {@code instanceof}
     * checks in the loop that runs once per pixel.
     */
    private enum Column {
        U1 {
            FieldVector newVector(BufferAllocator allocator) { return new UInt1Vector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((UInt1Vector) vector).set(index, (int) element);
                return element != 0L;
            }
        },
        I1 {
            FieldVector newVector(BufferAllocator allocator) { return new TinyIntVector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((TinyIntVector) vector).set(index, (int) element);
                return element != 0L;
            }
        },
        U2 {
            FieldVector newVector(BufferAllocator allocator) { return new UInt2Vector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((UInt2Vector) vector).set(index, (int) element);
                return element != 0L;
            }
        },
        I2 {
            FieldVector newVector(BufferAllocator allocator) { return new SmallIntVector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((SmallIntVector) vector).set(index, (int) element);
                return element != 0L;
            }
        },
        U4 {
            FieldVector newVector(BufferAllocator allocator) { return new UInt4Vector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((UInt4Vector) vector).set(index, (int) element);
                return element != 0L;
            }
        },
        I4 {
            FieldVector newVector(BufferAllocator allocator) { return new IntVector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((IntVector) vector).set(index, (int) element);
                return element != 0L;
            }
        },
        U8 {
            FieldVector newVector(BufferAllocator allocator) { return new UInt8Vector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((UInt8Vector) vector).set(index, element);
                return element != 0L;
            }
        },
        I8 {
            FieldVector newVector(BufferAllocator allocator) { return new BigIntVector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                long element = integerOf(value);
                ((BigIntVector) vector).set(index, element);
                return element != 0L;
            }
        },
        F4 {
            FieldVector newVector(BufferAllocator allocator) { return new Float4Vector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                float element = value.getRealFloat();
                ((Float4Vector) vector).set(index, element);
                return element != 0.0f;
            }
        },
        F8 {
            FieldVector newVector(BufferAllocator allocator) { return new Float8Vector("data", allocator); }
            boolean set(FieldVector vector, int index, RealType<?> value) {
                double element = value.getRealDouble();
                ((Float8Vector) vector).set(index, element);
                return element != 0.0;
            }
        };

        abstract FieldVector newVector(BufferAllocator allocator);

        /** Write one element, reporting whether it was non-zero. */
        abstract boolean set(FieldVector vector, int index, RealType<?> value);

        static Column of(String dtype) {
            switch (normalizeDtype(dtype)) {
                case "u1": return U1;
                case "i1": return I1;
                case "u2": return U2;
                case "i2": return I2;
                case "u4": return U4;
                case "i4": return I4;
                case "u8": return U8;
                case "i8": return I8;
                case "f8": return F8;
                case "f4": return F4;
                default:
                    throw new IllegalArgumentException("Cannot upload dtype '" + dtype + "'");
            }
        }
    }

    /**
     * An integer element, exactly. Not the double accessor: a 64-bit label id
     * would not survive the widening.
     */
    private static long integerOf(RealType<?> value) {
        return value instanceof IntegerType
                ? ((IntegerType<?>) value).getIntegerLong()
                : (long) value.getRealDouble();
    }

    private static long[] extentsOf(long[] start, long[] stop) {
        long[] extents = new long[start.length];
        for (int axis = 0; axis < start.length; axis++) {
            extents[axis] = stop[axis] - start[axis];
        }
        return extents;
    }

}
