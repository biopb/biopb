package biopb.tensor;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.List;

import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.grpc.CredentialCallOption;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.view.Views;
import net.imglib2.cache.img.ReadOnlyCachedCellImgFactory;
import net.imglib2.cache.img.ReadOnlyCachedCellImgOptions;
import net.imglib2.cache.img.SingleCellArrayImg;
import net.imglib2.cache.img.optional.CacheOptions.CacheType;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Externalizable wrapper for tensor images that enables serialization.
 *
 * This class wraps the non-serializable imglib2 CellImg returned by
 * TensorFlightClient.getTensor() and stores connection parameters
 * instead of actual data. After deserialization, it lazily reconstructs
 * the CellImg using a pooled connection.
 *
 * The wrapper is transparent - it implements RandomAccessibleInterval<T>,
 * so existing code using getTensor() can serialize/deserialize tensors
 * without modification.
 *
 * Serialization stores:
 * - Location URI string (server address)
 * - Authentication token
 * - Cache size (bytes)
 * - Source and tensor identifiers
 * - SliceHint (as protobuf bytes)
 * - scale_hint and reduction_method (flattened, as serialized bytes)
 *
 * Deserialization reconstructs:
 * - Pooled FlightClient connection (shared across deserializations)
 * - Lazy CellImg that fetches chunks on-demand
 */
public class SerializableTensorImg<T extends NativeType<T> & RealType<T>>
        implements RandomAccessibleInterval<T>, Externalizable {

    // Serialized state - connection parameters
    private String locationUri;
    private String token;
    private long cacheBytes;
    private String sourceId;
    private String tensorId;
    private byte[] sliceHintBytes;
    private byte[] scaleHintBytes; // Serialized long[] array
    private String reductionMethod;
    private byte[] descriptorBytes;

    // Transient state - reconstructed after deserialization
    private transient RandomAccessibleInterval<T> delegate;
    private transient SliceHint sliceHint;
    private transient long[] scaleHint;
    private transient TensorDescriptor descriptor;

    // Required for Externalizable - no-arg constructor
    public SerializableTensorImg() {
        this.locationUri = null;
        this.token = null;
        this.cacheBytes = 100_000_000L;
        this.sourceId = null;
        this.tensorId = null;
        this.sliceHintBytes = null;
        this.scaleHintBytes = null;
        this.reductionMethod = null;
        this.descriptorBytes = null;
        this.delegate = null;
    }

    /**
     * Constructor for initial creation from TensorFlightClient.
     *
     * @param location        Flight server location
     * @param token           Authentication token (null if none)
     * @param cacheBytes      Cache size in bytes
     * @param sourceId        Data source identifier
     * @param tensorId        Tensor identifier
     * @param sliceHint       Slice hint (null if none)
     * @param scaleHint       Per-dimension scale factors (null if none)
     * @param reductionMethod Reduction method string (null if none)
     * @param descriptor      Response tensor descriptor (contains shape, chunk
     *                        info)
     * @param delegate        The actual RandomAccessibleInterval to wrap
     */
    public SerializableTensorImg(
            Location location,
            String token,
            long cacheBytes,
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod,
            TensorDescriptor descriptor,
            RandomAccessibleInterval<T> delegate) {

        this.locationUri = location.getUri().toString();
        this.token = token;
        this.cacheBytes = cacheBytes;
        this.sourceId = sourceId;
        this.tensorId = tensorId;
        this.sliceHint = sliceHint;
        this.scaleHint = scaleHint;
        this.reductionMethod = reductionMethod;
        this.descriptor = descriptor;
        this.delegate = delegate;

        // Serialize protobuf objects and arrays to bytes
        this.sliceHintBytes = sliceHint != null ? sliceHint.toByteArray() : null;
        this.scaleHintBytes = scaleHint != null ? serializeLongArray(scaleHint) : null;
        this.descriptorBytes = descriptor != null ? descriptor.toByteArray() : null;
    }

    // Externalizable serialization

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeObject(locationUri);
        out.writeObject(token);
        out.writeLong(cacheBytes);
        out.writeUTF(sourceId);
        out.writeUTF(tensorId);
        out.writeObject(sliceHintBytes);
        out.writeObject(scaleHintBytes);
        out.writeObject(reductionMethod);
        out.writeObject(descriptorBytes);
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        locationUri = (String) in.readObject();
        token = (String) in.readObject();
        cacheBytes = in.readLong();
        sourceId = in.readUTF();
        tensorId = in.readUTF();
        sliceHintBytes = (byte[]) in.readObject();
        scaleHintBytes = (byte[]) in.readObject();
        reductionMethod = (String) in.readObject();
        descriptorBytes = (byte[]) in.readObject();

        // Parse protobuf bytes and deserialize arrays
        sliceHint = sliceHintBytes != null ? SliceHint.parseFrom(sliceHintBytes) : null;
        scaleHint = scaleHintBytes != null ? deserializeLongArray(scaleHintBytes) : null;
        descriptor = descriptorBytes != null ? TensorDescriptor.parseFrom(descriptorBytes) : null;

        // Delegate is null - will be reconstructed lazily
        delegate = null;
    }

    // Lazy reconstruction using pooled connection

    private void ensureDelegate() {
        if (delegate == null) {
            delegate = reconstructDelegate();
        }
    }

    @SuppressWarnings("unchecked")
    private RandomAccessibleInterval<T> reconstructDelegate() {
        if (locationUri == null || sourceId == null || tensorId == null) {
            throw new IllegalStateException("Cannot reconstruct tensor: missing connection parameters");
        }

        Location location = parseLocation(locationUri);
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        TensorConnectionPool.PooledConnection conn = TensorConnectionPool.getConnection(location, token, allocator);
        FlightClient client = conn.getClient();
        CredentialCallOption authOption = conn.getAuthOption();

        // Build TensorReadOption with flattened fields
        TensorReadOption.Builder readBuilder = TensorReadOption.newBuilder()
                .setTensorId(tensorId);

        if (sliceHint != null) {
            readBuilder.setSliceHint(sliceHint);
        }
        if (scaleHint != null) {
            for (long s : scaleHint) {
                readBuilder.addScaleHint(s);
            }
        }
        if (reductionMethod != null && !reductionMethod.isEmpty()) {
            readBuilder.setReductionMethod(reductionMethod);
        }

        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId(sourceId)
                .setTensorRead(readBuilder.build())
                .build();

        try {
            FlightInfo info = client.getInfo(
                    org.apache.arrow.flight.FlightDescriptor.command(cmd.toByteArray()),
                    authOption);

            // Use stored descriptor if available, otherwise parse from response
            TensorDescriptor responseDescriptor = descriptor != null
                    ? descriptor
                    : TensorDescriptor.parseFrom(info.getDescriptor().getCommand());

            // Create the array
            T type = (T) createType(responseDescriptor.getDtype());
            long[] dims = toLongArray(responseDescriptor.getShapeList());
            int[] cellDimensions = toIntArray(responseDescriptor.getChunkShapeList());

            EndpointIndex endpointIndex = buildEndpointIndex(responseDescriptor, dims, cellDimensions,
                    info.getEndpoints());

            if (endpointIndex == null) {
                // Materialize all data into an ArrayImg
                return materializeArray(client, authOption, responseDescriptor, type, dims, info.getEndpoints());
            }

            // Create lazy CellImg
            long estimatedChunkBytes = estimateChunkBytes(responseDescriptor);
            long maxCells = Math.max(1L, cacheBytes / Math.max(estimatedChunkBytes, 1L));
            ReadOnlyCachedCellImgOptions options = ReadOnlyCachedCellImgOptions.options()
                    .cellDimensions(cellDimensions)
                    .cacheType(CacheType.BOUNDED)
                    .maxCacheSize(maxCells);

            ReadOnlyCachedCellImgFactory factory = new ReadOnlyCachedCellImgFactory(options);
            RandomAccessibleInterval<T> rai = (RandomAccessibleInterval<T>) factory.create(dims, type,
                    cell -> loadCell(cell, endpointIndex, client, authOption));

            // Apply cropping if needed
            if (sliceHint != null && responseDescriptor.hasSliceHint()) {
                SliceHint realized = responseDescriptor.getSliceHint();
                int ndim = responseDescriptor.getShapeCount();
                long[] cropMin = new long[ndim];
                long[] cropMax = new long[ndim];
                boolean needsCrop = false;
                for (int ax = 0; ax < ndim; ax++) {
                    long reqStart = sliceHint.getStart(ax);
                    long reqStop = sliceHint.getStop(ax);
                    long retStart = realized.getStart(ax);
                    long scale = 1L;
                    // Use scale_hint directly from TensorDescriptor
                    if (responseDescriptor.getScaleHintCount() > ax) {
                        scale = responseDescriptor.getScaleHint(ax);
                    }
                    if (scale > 1L) {
                        cropMin[ax] = (reqStart - retStart) / scale;
                        cropMax[ax] = (reqStop - retStart + scale - 1L) / scale - 1L;
                    } else {
                        cropMin[ax] = reqStart - retStart;
                        cropMax[ax] = reqStop - retStart - 1L;
                    }
                    if (cropMin[ax] != 0 || cropMax[ax] != rai.max(ax)) {
                        needsCrop = true;
                    }
                }
                if (needsCrop) {
                    rai = Views.zeroMin(Views.interval(rai, new FinalInterval(cropMin, cropMax)));
                }
            }

            return rai;

        } catch (Exception e) {
            throw new IllegalStateException("Failed to reconstruct tensor after deserialization", e);
        }
    }

    // Helper methods for long[] serialization

    private static byte[] serializeLongArray(long[] arr) {
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        java.io.DataOutputStream dos = new java.io.DataOutputStream(baos);
        try {
            dos.writeInt(arr.length);
            for (long val : arr) {
                dos.writeLong(val);
            }
            dos.close();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to serialize long array", e);
        }
        return baos.toByteArray();
    }

    private static long[] deserializeLongArray(byte[] bytes) {
        java.io.ByteArrayInputStream bais = new java.io.ByteArrayInputStream(bytes);
        java.io.DataInputStream dis = new java.io.DataInputStream(bais);
        try {
            int len = dis.readInt();
            long[] arr = new long[len];
            for (int i = 0; i < len; i++) {
                arr[i] = dis.readLong();
            }
            dis.close();
            return arr;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to deserialize long array", e);
        }
    }

    // Helper methods (adapted from TensorFlightClient)

    private static Location parseLocation(String uri) {
        try {
            // Use the Location constructor that accepts URI
            return new Location(java.net.URI.create(uri));
        } catch (Exception e) {
            // Fallback for edge cases - parse URI components manually
            java.net.URI parsed = java.net.URI.create(uri);
            String scheme = parsed.getScheme();
            if (scheme == null) {
                // Default to grpc+tcp if no scheme
                return Location.forGrpcInsecure(parsed.getHost(), parsed.getPort());
            }
            // Let Location constructor handle it
            return new Location(parsed);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> materializeArray(
            FlightClient client,
            CredentialCallOption authOption,
            TensorDescriptor descriptor,
            T type,
            long[] dims,
            List<FlightEndpoint> endpoints) {

        ArrayImg<T, ?> image = (ArrayImg<T, ?>) new ArrayImgFactory<>(type).create(dims);
        RandomAccess<T> access = image.randomAccess();

        for (FlightEndpoint endpoint : endpoints) {
            TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
            ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
            double[] values = fetchChunkValues(client, authOption, ticket.getChunkId().toByteArray());
            writeChunk(access, bounds, values);
        }

        return image;
    }

    private static <T extends NativeType<T> & RealType<T>> void loadCell(
            SingleCellArrayImg<T, ?> cell,
            EndpointIndex endpointIndex,
            FlightClient client,
            CredentialCallOption authOption) {

        long cellIndex = endpointIndex.indexFor(cell);
        FlightEndpoint endpoint = endpointIndex.endpoints.get(cellIndex);
        if (endpoint == null) {
            throw new IllegalStateException("No Flight endpoint found for cell index " + cellIndex);
        }

        TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
        ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
        double[] values = fetchChunkValues(client, authOption, ticket.getChunkId().toByteArray());
        writeChunk(cell.randomAccess(), bounds, values);
    }

    private static EndpointIndex buildEndpointIndex(
            TensorDescriptor descriptor,
            long[] dims,
            int[] cellDimensions,
            List<FlightEndpoint> endpoints) {

        if (dims.length == 0 || endpoints.isEmpty()) {
            return null;
        }

        CellGrid grid = new CellGrid(dims, cellDimensions);
        java.util.Map<Long, FlightEndpoint> endpointMap = new java.util.HashMap<>();
        long[] gridDimensions = grid.getGridDimensions();
        long[] gridPosition = new long[dims.length];
        long[] expectedMin = new long[dims.length];
        int[] expectedDimensions = new int[dims.length];

        for (FlightEndpoint endpoint : endpoints) {
            ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
            if (bounds.getStartCount() != dims.length || bounds.getStopCount() != dims.length) {
                return null;
            }

            for (int axis = 0; axis < dims.length; axis++) {
                long start = bounds.getStart(axis);
                long stop = bounds.getStop(axis);
                int nominalCellDimension = cellDimensions[axis];
                if (start < 0 || stop < start || nominalCellDimension <= 0) {
                    return null;
                }
                if (start % nominalCellDimension != 0) {
                    return null;
                }
                gridPosition[axis] = start / nominalCellDimension;
                if (gridPosition[axis] >= gridDimensions[axis]) {
                    return null;
                }
            }

            grid.getCellDimensions(gridPosition, expectedMin, expectedDimensions);
            for (int axis = 0; axis < dims.length; axis++) {
                if (expectedMin[axis] != bounds.getStart(axis)) {
                    return null;
                }
                if ((long) expectedDimensions[axis] != bounds.getStop(axis) - bounds.getStart(axis)) {
                    return null;
                }
            }

            long cellIndex = grid.getCellGridIndexFlat(gridPosition);
            if (endpointMap.put(cellIndex, endpoint) != null) {
                return null;
            }
        }

        if (endpointMap.size() != cellCount(gridDimensions)) {
            return null;
        }

        return new EndpointIndex(grid, cellDimensions, endpointMap);
    }

    private static double[] fetchChunkValues(
            FlightClient client,
            CredentialCallOption authOption,
            byte[] chunkId) {

        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(com.google.protobuf.ByteString.copyFrom(chunkId))
                .build();

        try (org.apache.arrow.flight.FlightStream stream = client.getStream(
                new org.apache.arrow.flight.Ticket(tensorTicket.toByteArray()), authOption)) {
            double[] values = new double[0];
            while (stream.next()) {
                java.util.List<org.apache.arrow.vector.FieldVector> vectors = stream.getRoot().getFieldVectors();
                if (vectors.isEmpty()) {
                    throw new IllegalStateException("Chunk payload did not contain any Arrow vectors");
                }

                // Read "data" column - it's a ListArray with 1 row per chunk
                org.apache.arrow.vector.FieldVector dataVector = stream.getRoot().getVector("data");
                if (dataVector == null) {
                    throw new IllegalStateException("Chunk payload missing 'data' column");
                }

                // Each row is one chunk's data as a list
                int rowCount = stream.getRoot().getRowCount();
                for (int row = 0; row < rowCount; row++) {
                    Object rowObj = dataVector.getObject(row);
                    if (!(rowObj instanceof java.util.List)) {
                        throw new IllegalStateException("Data column value is not a list: " + rowObj.getClass());
                    }
                    java.util.List<?> dataList = (java.util.List<?>) rowObj;
                    int offset = values.length;
                    values = java.util.Arrays.copyOf(values, offset + dataList.size());
                    for (int i = 0; i < dataList.size(); i++) {
                        values[offset + i] = asDouble(dataList.get(i));
                    }
                }
            }
            return values;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to fetch chunk payload", e);
        }
    }

    private static double asDouble(Object value) {
        if (value == null) {
            throw new IllegalStateException("Chunk payload contains null values");
        }
        if (!(value instanceof Number)) {
            throw new IllegalStateException("Chunk payload is not numeric: " + value.getClass().getName());
        }
        return ((Number) value).doubleValue();
    }

    private static <T extends NativeType<T> & RealType<T>> void writeChunk(
            RandomAccess<T> access,
            ChunkBounds bounds,
            double[] values) {

        long[] start = toLongArray(bounds.getStartList());
        long[] stop = toLongArray(bounds.getStopList());
        long[] chunkShape = new long[start.length];
        long expectedSize = 1L;
        for (int axis = 0; axis < start.length; axis++) {
            chunkShape[axis] = stop[axis] - start[axis];
            expectedSize *= chunkShape[axis];
        }
        if (expectedSize != values.length) {
            throw new IllegalStateException(
                    "Chunk size mismatch: expected " + expectedSize + " values but received " + values.length);
        }

        long[] localPosition = new long[chunkShape.length];
        long[] globalPosition = new long[chunkShape.length];
        for (int index = 0; index < values.length; index++) {
            rowMajorPosition(index, chunkShape, localPosition);
            for (int axis = 0; axis < chunkShape.length; axis++) {
                globalPosition[axis] = start[axis] + localPosition[axis];
            }
            access.setPosition(globalPosition);
            access.get().setReal(values[index]);
        }
    }

    private static void rowMajorPosition(int index, long[] shape, long[] position) {
        long remaining = index;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            position[axis] = remaining % shape[axis];
            remaining /= shape[axis];
        }
    }

    private static NativeType<?> createType(String dtype) {
        String normalized = dtype == null ? "" : dtype.trim().toLowerCase();
        switch (normalized) {
            case "u1":
            case "uint8":
            case "|u1":
                return new UnsignedByteType();
            case "<u2":
            case ">u2":
            case "u2":
            case "uint16":
                return new UnsignedShortType();
            case "<u4":
            case ">u4":
            case "u4":
            case "uint32":
                return new UnsignedIntType();
            case "<f8":
            case ">f8":
            case "f8":
            case "float64":
                return new DoubleType();
            case "<f4":
            case ">f4":
            case "f4":
            case "float32":
            default:
                return new FloatType();
        }
    }

    private static long[] toLongArray(java.util.List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static int[] toIntArray(java.util.List<Long> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = java.lang.Math.toIntExact(values.get(i));
        }
        return out;
    }

    private static long cellCount(long[] gridDimensions) {
        long count = 1L;
        for (long axisCount : gridDimensions) {
            count *= axisCount;
        }
        return count;
    }

    private static long estimateChunkBytes(TensorDescriptor descriptor) {
        long elements = 1L;
        for (long dim : descriptor.getChunkShapeList()) {
            elements *= java.lang.Math.max(dim, 1L);
        }
        return elements * bytesPerElement(descriptor.getDtype());
    }

    private static int bytesPerElement(String dtype) {
        String normalized = dtype == null ? "" : dtype.trim().toLowerCase();
        switch (normalized) {
            case "u1":
            case "uint8":
            case "|u1":
                return 1;
            case "<u2":
            case ">u2":
            case "u2":
            case "uint16":
                return 2;
            case "<u4":
            case ">u4":
            case "u4":
            case "uint32":
            case "<f4":
            case ">f4":
            case "f4":
            case "float32":
                return 4;
            case "<f8":
            case ">f8":
            case "f8":
            case "float64":
                return 8;
            default:
                return 4;
        }
    }

    private static TensorTicket parseTicket(byte[] bytes) {
        try {
            return TensorTicket.parseFrom(bytes);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse TensorTicket", e);
        }
    }

    private static ChunkBounds parseChunkBounds(byte[] bytes) {
        try {
            return ChunkBounds.parseFrom(bytes);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse ChunkBounds", e);
        }
    }

    // RandomAccessibleInterval delegation

    @Override
    public long min(int d) {
        ensureDelegate();
        return delegate.min(d);
    }

    @Override
    public long max(int d) {
        ensureDelegate();
        return delegate.max(d);
    }

    @Override
    public int numDimensions() {
        ensureDelegate();
        return delegate.numDimensions();
    }

    @Override
    public long size() {
        ensureDelegate();
        return delegate.size();
    }

    @Override
    public RandomAccess<T> randomAccess() {
        ensureDelegate();
        return delegate.randomAccess();
    }

    @Override
    public RandomAccess<T> randomAccess(Interval interval) {
        ensureDelegate();
        return delegate.randomAccess(interval);
    }

    @Override
    public Cursor<T> cursor() {
        ensureDelegate();
        return delegate.cursor();
    }

    @Override
    public Cursor<T> localizingCursor() {
        ensureDelegate();
        return delegate.localizingCursor();
    }

    @Override
    public Object iterationOrder() {
        ensureDelegate();
        return delegate.iterationOrder();
    }

    @Override
    public long dimension(int d) {
        ensureDelegate();
        return delegate.dimension(d);
    }

    @Override
    public void min(long[] min) {
        ensureDelegate();
        delegate.min(min);
    }

    @Override
    public void max(long[] max) {
        ensureDelegate();
        delegate.max(max);
    }

    @Override
    public void dimensions(long[] dimensions) {
        ensureDelegate();
        delegate.dimensions(dimensions);
    }

    // EndpointIndex inner class (same as TensorFlightClient)

    private static class EndpointIndex {
        final CellGrid grid;
        final int[] nominalCellDimensions;
        final java.util.Map<Long, FlightEndpoint> endpoints;

        EndpointIndex(
                CellGrid grid,
                int[] nominalCellDimensions,
                java.util.Map<Long, FlightEndpoint> endpoints) {
            this.grid = grid;
            this.nominalCellDimensions = nominalCellDimensions.clone();
            this.endpoints = endpoints;
        }

        long indexFor(Interval interval) {
            long[] gridPosition = new long[interval.numDimensions()];
            for (int axis = 0; axis < interval.numDimensions(); axis++) {
                long min = interval.min(axis);
                int nominalCellDimension = nominalCellDimensions[axis];
                if (min % nominalCellDimension != 0) {
                    throw new IllegalStateException("Cell minimum is not aligned to the logical chunk grid");
                }
                gridPosition[axis] = min / nominalCellDimension;
            }
            return grid.getCellGridIndexFlat(gridPosition);
        }
    }
}