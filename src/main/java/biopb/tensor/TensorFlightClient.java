package biopb.tensor;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import biopb.tensor.ChunkBounds;
import biopb.tensor.DataSourceDescriptor;
import biopb.tensor.SliceHint;
import biopb.tensor.TensorDescriptor;
import biopb.tensor.TensorReadOptions;
import biopb.tensor.TensorSelection;
import biopb.tensor.TensorTicket;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import org.apache.arrow.flight.Criteria;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;

import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
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
 * Client for accessing tensors from a TensorFlightServer.
 *
 * This client uses Apache Arrow Flight to discover data sources, request logical
 * read plans, and fetch chunk payloads from a TensorFlightServer. It supports
 * multifield acquisitions where tensors within a source have different shapes.
 *
 * The Java client returns lazy cell-backed images when the logical Flight
 * endpoint layout matches the descriptor chunk grid. In that case, imglib2's
 * internal cell cache is the primary cache for repeated reads.
 *
 * Usage:
 * <pre>
 * TensorFlightClient client = new TensorFlightClient("localhost:8815");
 *
 * // List data sources (each may contain multiple tensors)
 * Map&lt;String, DataSourceDescriptor&gt; sources = client.listSources();
 *
 * // Access a specific tensor within a source
 * RandomAccessibleInterval&lt;UnsignedByteType&gt; arr = client.getTensor("my-source", "tensor-0");
 * long[] pos = {10, 20, 30};
 * UnsignedByteType pixel = arr.getAt(pos);
 * client.close();
 * </pre>
 */
public class TensorFlightClient implements AutoCloseable {

    private static final String DEFAULT_REDUCTION_METHOD = "nearest";

    private final BufferAllocator allocator;
    private final FlightClient client;
    private final Map<String, TensorDescriptor> descriptors;
    private final Map<String, DataSourceDescriptor> sources;
    private final long cacheBytes;

    /**
     * Create a new TensorFlightClient.
     *
     * @param host Server host
     * @param port Server port
     */
    public TensorFlightClient(String host, int port) {
        this(Location.forGrpcInsecure(host, port), 100_000_000L);
    }

    /**
     * Create a new TensorFlightClient with custom cache size.
     *
     * @param host Server host
     * @param port Server port
     * @param cacheBytes Maximum cache size in bytes
     */
    public TensorFlightClient(String host, int port, long cacheBytes) {
        this(Location.forGrpcInsecure(host, port), cacheBytes);
    }

    /**
     * Create a new TensorFlightClient for an Arrow Flight location.
     *
     * @param location Flight server location
     */
    public TensorFlightClient(Location location) {
        this(location, 100_000_000L);
    }

    /**
     * Create a new TensorFlightClient for an Arrow Flight location.
     *
     * @param location Flight server location
     * @param cacheBytes Maximum cache size in bytes
     */
    public TensorFlightClient(Location location, long cacheBytes) {
        this.allocator = new RootAllocator(Long.MAX_VALUE);
        this.client = FlightClient.builder(this.allocator, location).build();
        this.descriptors = new HashMap<>();
        this.sources = new HashMap<>();
        this.cacheBytes = cacheBytes;
    }

    /**
     * List available data sources.
     *
     * Each data source may contain multiple tensors (for multifield acquisitions
     * where tensors have different shapes). The returned DataSourceDescriptor
     * contains full tensor metadata (shape, dtype, chunk_shape) for all tensors.
     *
     * @return Map of source_id to DataSourceDescriptor
     */
    public Map<String, DataSourceDescriptor> listSources() throws IOException {
        Map<String, DataSourceDescriptor> result = new HashMap<>();
        for (FlightInfo info : client.listFlights(Criteria.ALL)) {
            DataSourceDescriptor sourceDesc = DataSourceDescriptor.parseFrom(
                info.getDescriptor().getCommand());
            result.put(sourceDesc.getSourceId(), sourceDesc);
            // Cache tensor descriptors for quick lookup
            for (TensorDescriptor tensorDesc : sourceDesc.getTensorsList()) {
                descriptors.put(tensorDesc.getArrayId(), tensorDesc);
            }
        }
        sources.putAll(result);
        return result;
    }

    /**
     * Get source-level OME/vendor metadata.
     *
     * Fetches metadata via GetFlightInfo for the first tensor in the source,
     * since metadataJson is populated in the response TensorDescriptor.
     *
     * @param sourceId Source identifier
     * @return Parsed metadata as Map, or empty map if no metadata
     */
    public Map<String, Object> getSourceMetadata(String sourceId) throws IOException {
        if (!sources.containsKey(sourceId)) {
            listSources();
        }
        DataSourceDescriptor sourceDesc = sources.get(sourceId);
        if (sourceDesc == null) {
            throw new IllegalArgumentException("Source not found: " + sourceId);
        }
        if (sourceDesc.getTensorsList().isEmpty()) {
            return new HashMap<>();
        }

        // Get metadata from first tensor via GetFlightInfo
        TensorDescriptor firstTensor = sourceDesc.getTensorsList().get(0);
        TensorSelection selection = TensorSelection.newBuilder()
                .setSourceId(sourceId)
                .setTensorId(firstTensor.getArrayId())
                .build();
        FlightInfo info = client.getInfo(FlightDescriptor.command(selection.toByteArray()));
        TensorDescriptor responseDesc = TensorDescriptor.parseFrom(info.getDescriptor().getCommand());

        if (responseDesc.getMetadataJson().isEmpty()) {
            return new HashMap<>();
        }
        return parseMetadataJson(responseDesc.getMetadataJson());
    }

    /**
     * Get a RandomAccessibleInterval for a tensor within a data source.
     *
     * @param sourceId Data source identifier
     * @param tensorId Tensor identifier within the source
     * @param <T> The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId) {

        return getTensor(sourceId, tensorId, null, (TensorReadOptions) null);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor with read options.
     *
     * @param sourceId Data source identifier
     * @param tensorId Tensor identifier within the source
     * @param readOptions Optional request-scoped read options
     * @param <T> The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            TensorReadOptions readOptions) {

        return getTensor(sourceId, tensorId, null, readOptions);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor with slice and read options.
     *
     * @param sourceId Data source identifier
     * @param tensorId Tensor identifier within the source
     * @param sliceHint Optional slice hint
     * @param readOptions Optional request-scoped read options
     * @param <T> The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            TensorReadOptions readOptions) {

        RequestContext context = getTensorContext(sourceId, tensorId, sliceHint, readOptions);
        return createArray(context);
    }

    /**
     * Convenience overload for scaled reads.
     *
     * @param sourceId Data source identifier
     * @param tensorId Tensor identifier within the source
     * @param scaleHint Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @param <T> The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            long[] scaleHint,
            String reductionMethod) {

        TensorReadOptions readOptions = buildReadOptions(scaleHint, reductionMethod);
        return getTensor(sourceId, tensorId, null, readOptions);
    }

    /**
     * Convenience overload for sliced scaled reads.
     *
     * @param sourceId Data source identifier
     * @param tensorId Tensor identifier within the source
     * @param sliceHint Optional slice hint
     * @param scaleHint Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @param <T> The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

        TensorReadOptions readOptions = buildReadOptions(scaleHint, reductionMethod);
        return getTensor(sourceId, tensorId, sliceHint, readOptions);
    }

    @Override
    public void close() {
        try {
            client.close();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            allocator.close();
        }
    }

    private RequestContext getTensorContext(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            TensorReadOptions readOptions) {

        // Ensure sources are loaded
        if (sources.isEmpty()) {
            try {
                listSources();
            } catch (IOException e) {
                throw new IllegalStateException("Failed to list sources", e);
            }
        }

        DataSourceDescriptor sourceDesc = sources.get(sourceId);
        if (sourceDesc == null) {
            throw new IllegalArgumentException("Source not found: " + sourceId);
        }

        // Find tensor descriptor to get shape for validation
        TensorDescriptor baseDescriptor = null;
        for (TensorDescriptor desc : sourceDesc.getTensorsList()) {
            if (desc.getArrayId().equals(tensorId)) {
                baseDescriptor = desc;
                break;
            }
        }
        if (baseDescriptor == null) {
            throw new IllegalArgumentException("Tensor '" + tensorId + "' not found in source '" + sourceId + "'");
        }

        TensorReadOptions normalizedReadOptions = normalizeReadOptions(baseDescriptor, readOptions);

        // Build TensorSelection for the request
        TensorSelection.Builder selectionBuilder = TensorSelection.newBuilder()
            .setSourceId(sourceId)
            .setTensorId(tensorId);

        if (sliceHint != null) {
            selectionBuilder.setSliceHint(sliceHint);
        }
        if (normalizedReadOptions != null) {
            selectionBuilder.setReadOptions(normalizedReadOptions);
        }

        TensorSelection selection = selectionBuilder.build();
        FlightInfo info = client.getInfo(FlightDescriptor.command(selection.toByteArray()));
        TensorDescriptor responseDescriptor = parseDescriptorUnchecked(info.getDescriptor().getCommand());

        // Cache the response descriptor
        descriptors.put(responseDescriptor.getArrayId(), responseDescriptor);

        return new RequestContext(responseDescriptor, info.getEndpoints());
    }

    private static TensorReadOptions buildReadOptions(long[] scaleHint, String reductionMethod) {
        if ((scaleHint == null || scaleHint.length == 0)
                && (reductionMethod == null || reductionMethod.isEmpty())) {
            return null;
        }

        TensorReadOptions.Builder builder = TensorReadOptions.newBuilder();
        if (scaleHint != null) {
            for (long value : scaleHint) {
                builder.addScaleHint(value);
            }
        }
        if (reductionMethod != null && !reductionMethod.isEmpty()) {
            builder.setReductionMethod(reductionMethod);
        }
        return builder.build();
    }

    private static TensorReadOptions normalizeReadOptions(
            TensorDescriptor baseDescriptor,
            TensorReadOptions readOptions) {

        if (readOptions == null) {
            return null;
        }

        int rank = baseDescriptor.getShapeCount();
        if (readOptions.getScaleHintCount() == 0) {
            return readOptions;
        }

        if (readOptions.getScaleHintCount() != rank) {
            throw new IllegalArgumentException(
                    "Scale hint dimensionality mismatch: expected " + rank
                            + ", got " + readOptions.getScaleHintCount());
        }

        TensorReadOptions.Builder builder = TensorReadOptions.newBuilder();
        for (int axis = 0; axis < readOptions.getScaleHintCount(); axis++) {
            long scale = readOptions.getScaleHint(axis);
            if (scale <= 0) {
                throw new IllegalArgumentException(
                        "Scale hint must be positive on axis " + axis + ": " + scale);
            }
            builder.addScaleHint(scale);
        }

        builder.setReductionMethod(normalizeReductionMethod(readOptions.getReductionMethod()));
        return builder.build();
    }

    private static String normalizeReductionMethod(String reductionMethod) {
        String normalized = reductionMethod == null
                ? DEFAULT_REDUCTION_METHOD
                : reductionMethod.trim().toLowerCase();

        if (normalized.isEmpty()) {
            return DEFAULT_REDUCTION_METHOD;
        }

        switch (normalized) {
            case "stride":
            case "decimate":
                return "nearest";
            case "mean":
                return "area";
            case "nearest":
            case "area":
            case "linear":
                return normalized;
            default:
                throw new IllegalArgumentException(
                        "Unsupported reduction method: " + reductionMethod
                                + ". Supported methods: [nearest, area, linear]");
        }
    }

    @SuppressWarnings("unchecked")
    private <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> createArray(
            RequestContext context) {

        T type = (T) createType(context.descriptor.getDtype());
        long[] dims = toLongArray(context.descriptor.getShapeList());
        int[] cellDimensions = toIntArray(context.descriptor.getChunkShapeList());

        EndpointIndex endpointIndex = buildEndpointIndex(context, dims, cellDimensions);
        if (endpointIndex == null) {
            return materializeArray(context);
        }

        long estimatedChunkBytes = estimateChunkBytes(context.descriptor);
        long maxCells = Math.max(1L, cacheBytes / Math.max(estimatedChunkBytes, 1L));
        ReadOnlyCachedCellImgOptions options = ReadOnlyCachedCellImgOptions.options()
                .cellDimensions(cellDimensions)
                .cacheType(CacheType.BOUNDED)
                .maxCacheSize(maxCells);

        ReadOnlyCachedCellImgFactory factory = new ReadOnlyCachedCellImgFactory(options);
        return (RandomAccessibleInterval<T>) factory.create(dims, type,
                cell -> loadCell(cell, endpointIndex));
    }

    @SuppressWarnings("unchecked")
    private <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> materializeArray(
            RequestContext context) {

        T type = (T) createType(context.descriptor.getDtype());
        long[] dims = toLongArray(context.descriptor.getShapeList());
        ArrayImg<T, ?> image = (ArrayImg<T, ?>) new ArrayImgFactory<>(type).create(dims);
        RandomAccess<T> access = image.randomAccess();

        for (FlightEndpoint endpoint : context.endpoints) {
            TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
            ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
            double[] values = fetchChunkValues(ticket.getChunkId().toByteArray());
            writeChunk(access, bounds, values);
        }

        return image;
    }

    private <T extends NativeType<T> & RealType<T>> void loadCell(
            SingleCellArrayImg<T, ?> cell,
            EndpointIndex endpointIndex) {

        long cellIndex = endpointIndex.indexFor(cell);
        FlightEndpoint endpoint = endpointIndex.endpoints.get(cellIndex);
        if (endpoint == null) {
            throw new IllegalStateException("No Flight endpoint found for cell index " + cellIndex);
        }

        TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
        ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
        double[] values = fetchChunkValues(ticket.getChunkId().toByteArray());
        writeChunk(cell.randomAccess(), bounds, values);
    }

    private EndpointIndex buildEndpointIndex(
            RequestContext context,
            long[] dims,
            int[] cellDimensions) {

        if (dims.length == 0 || context.endpoints.isEmpty()) {
            return null;
        }

        CellGrid grid = new CellGrid(dims, cellDimensions);
        Map<Long, FlightEndpoint> endpointMap = new HashMap<>();
        long[] gridDimensions = grid.getGridDimensions();
        long[] gridPosition = new long[dims.length];
        long[] expectedMin = new long[dims.length];
        int[] expectedDimensions = new int[dims.length];

        for (FlightEndpoint endpoint : context.endpoints) {
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

    private double[] fetchChunkValues(byte[] chunkId) {
        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFrom(chunkId))
                .build();

        try (FlightStream stream = client.getStream(new Ticket(tensorTicket.toByteArray()))) {
            double[] values = new double[0];
            while (stream.next()) {
                List<FieldVector> vectors = stream.getRoot().getFieldVectors();
                if (vectors.isEmpty()) {
                    throw new IllegalStateException("Chunk payload did not contain any Arrow vectors");
                }
                FieldVector vector = vectors.get(0);
                int rowCount = stream.getRoot().getRowCount();
                int offset = values.length;
                values = Arrays.copyOf(values, offset + rowCount);
                for (int i = 0; i < rowCount; i++) {
                    values[offset + i] = asDouble(vector.getObject(i));
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

    private static long[] toLongArray(List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static int[] toIntArray(List<Long> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = Math.toIntExact(values.get(i));
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
            elements *= Math.max(dim, 1L);
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

    private static TensorDescriptor parseDescriptorUnchecked(byte[] bytes) {
        try {
            return TensorDescriptor.parseFrom(bytes);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to parse TensorDescriptor", e);
        }
    }

    private static TensorTicket parseTicket(byte[] bytes) {
        try {
            return TensorTicket.parseFrom(bytes);
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse TensorTicket", e);
        }
    }

    private static ChunkBounds parseChunkBounds(byte[] bytes) {
        try {
            return ChunkBounds.parseFrom(bytes);
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Failed to parse ChunkBounds", e);
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

    private static final Gson GSON = new Gson();

    private static Map<String, Object> parseMetadataJson(String json) {
        if (json == null || json.isEmpty()) {
            return new HashMap<>();
        }
        try {
            return GSON.fromJson(json, new TypeToken<Map<String, Object>>() {}.getType());
        } catch (Exception e) {
            Map<String, Object> result = new HashMap<>();
            result.put("raw", json);
            return result;
        }
    }

    private static class EndpointIndex {
        final CellGrid grid;
        final int[] nominalCellDimensions;
        final Map<Long, FlightEndpoint> endpoints;

        EndpointIndex(
                CellGrid grid,
                int[] nominalCellDimensions,
                Map<Long, FlightEndpoint> endpoints) {
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

    private static class RequestContext {
        final TensorDescriptor descriptor;
        final List<FlightEndpoint> endpoints;

        RequestContext(TensorDescriptor descriptor, List<FlightEndpoint> endpoints) {
            this.descriptor = parseDescriptorUnchecked(descriptor.toByteArray());
            this.endpoints = endpoints;
        }
    }
}