package biopb.tensor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.arrow.flight.Criteria;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.flight.grpc.CredentialCallOption;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Schema;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import net.imglib2.FinalInterval;
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
import net.imglib2.view.Views;

/**
 * Client for accessing tensors from a TensorFlightServer.
 *
 * This client uses Apache Arrow Flight to discover data sources, request
 * logical
 * read plans, and fetch chunk payloads from a TensorFlightServer. It supports
 * multifield acquisitions where tensors within a source have different shapes.
 *
 * The Java client returns lazy cell-backed images when the logical Flight
 * endpoint layout matches the descriptor chunk grid. In that case, imglib2's
 * internal cell cache is the primary cache for repeated reads.
 *
 * Usage:
 * 
 * <pre>
 * TensorFlightClient client = new TensorFlightClient("localhost:8815");
 *
 * // List data sources (each may contain multiple tensors)
 * Map&lt;String, DataSourceDescriptor&gt; sources = client.listSources();
 *
 * // Access a specific tensor within a source
 * RandomAccessibleInterval&lt;UnsignedByteType&gt; arr = client.getTensor("my-source", "tensor-0");
 * long[] pos = { 10, 20, 30 };
 * UnsignedByteType pixel = arr.getAt(pos);
 * client.close();
 * </pre>
 */
public class TensorFlightClient implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(TensorFlightClient.class.getName());
    private static final String DEFAULT_REDUCTION_METHOD = "nearest";

    private final BufferAllocator allocator;
    private final FlightClient client;
    private final CredentialCallOption authOption;
    private final Location location;
    private final String token;
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
        this(Location.forGrpcInsecure(host, port), 100_000_000L, null);
    }

    /**
     * Create a new TensorFlightClient with custom cache size.
     *
     * @param host       Server host
     * @param port       Server port
     * @param cacheBytes Maximum cache size in bytes
     */
    public TensorFlightClient(String host, int port, long cacheBytes) {
        this(Location.forGrpcInsecure(host, port), cacheBytes, null);
    }

    /**
     * Create a new TensorFlightClient with authentication token.
     *
     * @param host       Server host
     * @param port       Server port
     * @param cacheBytes Maximum cache size in bytes
     * @param token      Bearer token for authentication (null disables auth)
     */
    public TensorFlightClient(String host, int port, long cacheBytes, String token) {
        this(Location.forGrpcInsecure(host, port), cacheBytes, token);
    }

    /**
     * Create a new TensorFlightClient for an Arrow Flight location.
     *
     * @param location Flight server location
     */
    public TensorFlightClient(Location location) {
        this(location, 100_000_000L, null);
    }

    /**
     * Create a new TensorFlightClient for an Arrow Flight location.
     *
     * @param location   Flight server location
     * @param cacheBytes Maximum cache size in bytes
     */
    public TensorFlightClient(Location location, long cacheBytes) {
        this(location, cacheBytes, null);
    }

    /**
     * Create a new TensorFlightClient for an Arrow Flight location with
     * authentication.
     *
     * @param location   Flight server location
     * @param cacheBytes Maximum cache size in bytes
     * @param token      Bearer token for authentication (null disables auth)
     */
    public TensorFlightClient(Location location, long cacheBytes, String token) {
        LOGGER.info(
                "Connecting to Flight server at " + location + ", cache=" + cacheBytes + "B, auth=" + (token != null));
        this.location = location;
        this.allocator = new RootAllocator(Long.MAX_VALUE);
        this.client = FlightClient.builder(this.allocator, location).build();
        this.token = token;
        this.authOption = (token != null && !token.isEmpty())
                ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + token))
                : null;
        this.descriptors = new HashMap<>();
        this.sources = new HashMap<>();
        this.cacheBytes = cacheBytes;
    }

    /**
     * Get the Flight server location.
     *
     * @return Location used for this client
     */
    public Location getLocation() {
        return location;
    }

    /**
     * Get the authentication token.
     *
     * @return Bearer token (null if no authentication)
     */
    public String getToken() {
        return token;
    }

    /**
     * Get the cache size in bytes.
     *
     * @return Maximum cache size for cell images
     */
    public long getCacheBytes() {
        return cacheBytes;
    }

    /**
     * List available data sources.
     *
     * Each data source may contain multiple tensors (for multifield acquisitions
     * where tensors have different shapes). The returned DataSourceDescriptor
     * contains full tensor metadata (shape, dtype, chunk_shape) for all tensors.
     *
     * Results may be truncated if server has max_list_flights_results configured.
     * Check returned map size vs total_sources in schema metadata for truncation
     * info.
     *
     * @return Map of source_id to DataSourceDescriptor
     */
    public Map<String, DataSourceDescriptor> listSources() throws IOException {
        Map<String, DataSourceDescriptor> result = new HashMap<>();
        boolean truncated = false;
        long totalSources = 0;

        for (FlightInfo info : client.listFlights(Criteria.ALL, authOption)) {
            DataSourceDescriptor sourceDesc = DataSourceDescriptor.parseFrom(
                    info.getDescriptor().getCommand());
            result.put(sourceDesc.getSourceId(), sourceDesc);
            // Cache tensor descriptors for quick lookup
            for (TensorDescriptor tensorDesc : sourceDesc.getTensorsList()) {
                descriptors.put(tensorDesc.getArrayId(), tensorDesc);
            }

            // Check schema metadata for truncation info
            java.util.Optional<Schema> schemaOpt = info.getSchemaOptional();
            if (schemaOpt.isPresent()) {
                Map<String, String> metadata = schemaOpt.get().getCustomMetadata();
                if (metadata != null) {
                    String truncatedStr = metadata.get("truncated");
                    if (truncatedStr != null) {
                        truncated = Boolean.parseBoolean(truncatedStr);
                    }
                    String totalStr = metadata.get("total_sources");
                    if (totalStr != null) {
                        totalSources = Long.parseLong(totalStr);
                    }
                }
            }
        }
        sources.putAll(result);

        if (truncated && totalSources > result.size()) {
            LOGGER.warning("listSources: returned " + result.size() + " of " + totalSources + " sources (truncated)");
        } else {
            LOGGER.info("listSources: returned " + result.size() + " sources");
        }

        return result;
    }

    /**
     * Execute SQL query against server's source metadata database.
     *
     * Returns Arrow VectorSchemaRoot with query results. Schema metadata may
     * contain
     * "total_sources" key if result was truncated.
     *
     * Requires server to have metadata_db.enabled=true in config.
     *
     * @param sql SQL query (e.g., "SELECT source_id FROM sources WHERE source_url
     *            LIKE '%plate%'")
     * @return VectorSchemaRoot with query results (caller must close)
     * @throws IOException If query fails or server does not have metadata database
     *                     enabled
     *
     *                     Example:
     * 
     *                     <pre>
     *                     VectorSchemaRoot result = client.querySources("SELECT source_id, source_type FROM sources");
     *                     System.out.println("Found " + result.getRowCount() + " sources");
     *                     result.close();
     *                     </pre>
     */
    public VectorSchemaRoot querySources(String sql) throws IOException {
        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId("__metadata_query__")
                .setMetadataQuery(MetadataQueryOption.newBuilder()
                        .setSql(sql)
                        .build())
                .build();

        FlightInfo info = client.getInfo(
                FlightDescriptor.command(cmd.toByteArray()), authOption);

        // Check schema metadata for truncation
        java.util.Optional<Schema> schemaOpt = info.getSchemaOptional();
        if (schemaOpt.isPresent()) {
            Map<String, String> metadata = schemaOpt.get().getCustomMetadata();
            if (metadata != null && metadata.containsKey("total_sources")) {
                long total = Long.parseLong(metadata.get("total_sources"));
                String returnedStr = metadata.get("returned_sources");
                long returned = returnedStr != null ? Long.parseLong(returnedStr) : info.getEndpoints().size();
                if (returned < total) {
                    LOGGER.info("querySources: result truncated, returned " + returned + " of " + total + " sources");
                } else {
                    LOGGER.info("querySources: returned " + returned + " sources");
                }
            }
        }

        // Fetch results via doGet
        if (info.getEndpoints().isEmpty()) {
            // Empty result - return empty table
            Schema schema = info.getSchema();
            VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
            root.setRowCount(0);
            return root;
        }

        FlightStream stream = client.getStream(
                info.getEndpoints().get(0).getTicket(), authOption);

        // Materialize all batches using ArrowRecordBatch (Arrow 18 API)
        List<ArrowRecordBatch> batches = new ArrayList<>();
        Schema schema = info.getSchema();
        while (stream.next()) {
            VectorUnloader unloader = new VectorUnloader(stream.getRoot());
            ArrowRecordBatch batch = unloader.getRecordBatch().cloneWithTransfer(allocator);
            batches.add(batch);
        }

        if (batches.isEmpty()) {
            VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
            root.setRowCount(0);
            return root;
        }

        // Concatenate all batches into one VectorSchemaRoot
        return concatenateBatches(schema, batches);
    }

    /**
     * Concatenate ArrowRecordBatch objects into a single VectorSchemaRoot.
     * Uses TransferPair to copy values from each batch into the result.
     */
    private VectorSchemaRoot concatenateBatches(Schema schema, List<ArrowRecordBatch> batches) {
        // Calculate total row count
        int totalRows = 0;
        for (ArrowRecordBatch batch : batches) {
            totalRows += batch.getLength();
        }

        // Create result root with enough capacity
        VectorSchemaRoot result = VectorSchemaRoot.create(schema, allocator);
        result.allocateNew();
        VectorLoader loader = new VectorLoader(result);

        // Load and copy each batch
        int offset = 0;
        for (ArrowRecordBatch batch : batches) {
            loader.load(batch);
            // Copy values from current position to offset position
            for (int i = 0; i < result.getFieldVectors().size(); i++) {
                org.apache.arrow.vector.ValueVector srcVec = result.getVector(i);
                org.apache.arrow.vector.ValueVector dstVec = result.getVector(i);
                // Use slice to get the loaded portion and copy to offset
                // Since loader loads starting at 0, we need to shift values
                for (int row = 0; row < batch.getLength(); row++) {
                    dstVec.copyFromSafe(row, offset + row, srcVec);
                }
            }
            offset += batch.getLength();
            batch.close();
        }

        result.setRowCount(totalRows);
        return result;
    }

    /**
     * Get source-level OME/vendor metadata.
     *
     * Fetches metadata via GetFlightInfo for the first tensor in the source,
     * since metadataJson is populated in the response TensorDescriptor.
     * The server wraps metadata in {"type": ..., "dim_label": [...], "metadata":
     * {...}},
     * this method returns the inner "metadata" dict.
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
        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId(sourceId)
                .setTensorRead(TensorReadOption.newBuilder()
                        .setTensorId(firstTensor.getArrayId())
                        .setWithMetadata(true)
                        .build())
                .build();
        FlightInfo info = client.getInfo(FlightDescriptor.command(cmd.toByteArray()), authOption);
        TensorDescriptor responseDesc = TensorDescriptor.parseFrom(info.getDescriptor().getCommand());

        if (responseDesc.getMetadataJson().isEmpty()) {
            return new HashMap<>();
        }
        // Unwrap to return just the metadata dict
        Map<String, Object> wrapped = parseMetadataJson(responseDesc.getMetadataJson());
        Object metadataValue = wrapped.get("metadata");
        if (metadataValue instanceof Map) {
            return (Map<String, Object>) metadataValue;
        }
        return wrapped;
    }

    /**
     * Get a RandomAccessibleInterval for a tensor within a data source.
     *
     * @param sourceId Data source identifier
     * @param tensorId Tensor identifier within the source
     * @param <T>      The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId) {

        return getTensor(sourceId, tensorId, null, null, null);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor with slice hint.
     *
     * @param sourceId  Data source identifier
     * @param tensorId  Tensor identifier within the source
     * @param sliceHint Optional slice hint
     * @param <T>       The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            SliceHint sliceHint) {

        return getTensor(sourceId, tensorId, sliceHint, null, null);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor with scaled read options.
     *
     * @param sourceId        Data source identifier
     * @param tensorId        Tensor identifier within the source
     * @param scaleHint       Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @param <T>             The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            long[] scaleHint,
            String reductionMethod) {

        return getTensor(sourceId, tensorId, null, scaleHint, reductionMethod);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor with all options.
     *
     * @param sourceId        Data source identifier
     * @param tensorId        Tensor identifier within the source
     * @param sliceHint       Optional slice hint
     * @param scaleHint       Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @param <T>             The pixel type
     * @return SerializableTensorImg containing the requested tensor (implements
     *         RandomAccessibleInterval)
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

        RequestContext context = getTensorContext(sourceId, tensorId, sliceHint, scaleHint, reductionMethod);
        RandomAccessibleInterval<T> rai = createArray(context);

        // Crop to the originally requested region.
        // The server snaps slice_hint outward to lcm-aligned chunk boundaries, so
        // descriptor.shape may be larger than the requested extent.
        if (sliceHint != null && context.descriptor.hasSliceHint()) {
            SliceHint realized = context.descriptor.getSliceHint();
            int ndim = context.descriptor.getShapeCount();
            long[] cropMin = new long[ndim];
            long[] cropMax = new long[ndim];
            boolean needsCrop = false;
            for (int ax = 0; ax < ndim; ax++) {
                long reqStart = sliceHint.getStart(ax);
                long reqStop = sliceHint.getStop(ax);
                long retStart = realized.getStart(ax);
                long scale = 1L;
                // Use scale_hint directly from TensorDescriptor
                if (context.descriptor.getScaleHintCount() > ax) {
                    scale = context.descriptor.getScaleHint(ax);
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

        // Return SerializableTensorImg wrapper for serialization support
        return new SerializableTensorImg<>(location, token, cacheBytes, sourceId, tensorId,
                sliceHint, scaleHint, reductionMethod, context.descriptor, rai);
    }

    /**
     * Get a SerializedTensor protobuf for cross-process transfer.
     *
     * Returns a protobuf containing connection info and chunk tickets
     * for lazy reconstruction. The protobuf can be serialized to bytes
     * and broadcast to worker processes (e.g., Spark), where each worker
     * can call tensorFromPb() to reconstruct a lazy imglib2 array.
     *
     * @param sourceId        Data source identifier
     * @param tensorId        Tensor identifier within the source
     * @param sliceHint       Optional slice hint
     * @param scaleHint       Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @return SerializedTensor protobuf object
     */
    public SerializedTensor getTensorAsPb(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

        LOGGER.fine("getTensorAsPb: sourceId=" + sourceId + ", tensorId=" + tensorId);
        RequestContext context = getTensorContext(sourceId, tensorId, sliceHint, scaleHint, reductionMethod);

        // Serialize endpoints
        List<SerializedEndpoint> serializedEndpoints = new ArrayList<>();
        for (FlightEndpoint endpoint : context.endpoints) {
            TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
            ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
            SerializedEndpoint serializedEp = SerializedEndpoint.newBuilder()
                    .setTicket(ticket)
                    .setChunkBounds(bounds)
                    .build();
            serializedEndpoints.add(serializedEp);
        }

        // Build SerializedTensor
        SerializedTensor.Builder builder = SerializedTensor.newBuilder()
                .setTensorDescriptor(context.descriptor)
                .setLocation(location.getUri().toString())
                .addAllEndpoints(serializedEndpoints);

        if (token != null && !token.isEmpty()) {
            builder.setAuthToken(token);
        }

        if (sliceHint != null) {
            builder.setOriginalSliceHint(sliceHint);
        }

        return builder.build();
    }

    /**
     * Reconstruct a lazy RandomAccessibleInterval from SerializedTensor protobuf.
     *
     * Creates an imglib2 array that fetches chunks from the Flight server
     * independently. Each worker process maintains its own connection pool
     * and cache keyed by (location, authToken).
     *
     * If endpoints field is empty, calls GetFlightInfo on the server
     * to rebuild the endpoint list.
     *
     * @param pb          SerializedTensor protobuf object
     * @param cacheBytes  Maximum cache size in bytes
     * @param <T>         The pixel type
     * @return RandomAccessibleInterval with lazy chunk loading
     */
    public static <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> tensorFromPb(
            SerializedTensor pb,
            long cacheBytes) {

        TensorDescriptor descriptor = pb.getTensorDescriptor();
        T type = (T) createType(descriptor.getDtype());
        long[] dims = toLongArray(descriptor.getShapeList());
        int[] cellDimensions = toIntArray(descriptor.getChunkShapeList());

        // Build endpoint index - if endpoints empty, fetch via GetFlightInfo
        final SerializedTensor pbEffective;
        if (pb.getEndpointsCount() == 0) {
            LOGGER.fine("tensorFromPb: endpoints empty, fetching via GetFlightInfo");
            List<SerializedEndpoint> fetchedEndpoints = fetchEndpointsViaGetFlightInfo(pb);
            SerializedTensor.Builder pbBuilder = SerializedTensor.newBuilder()
                    .setTensorDescriptor(descriptor)
                    .setLocation(pb.getLocation())
                    .setAuthToken(pb.getAuthToken())
                    .addAllEndpoints(fetchedEndpoints);
            if (pb.hasOriginalSliceHint()) {
                pbBuilder.setOriginalSliceHint(pb.getOriginalSliceHint());
            }
            pbEffective = pbBuilder.build();
        } else {
            pbEffective = pb;
        }

        SerializedEndpointIndex endpointIndex = buildSerializedEndpointIndex(pbEffective, dims, cellDimensions);

        if (endpointIndex == null) {
            // Materialize array for non-aligned endpoint layout
            return materializeSerializedArray(pbEffective, cacheBytes);
        }

        long estimatedChunkBytes = estimateChunkBytes(descriptor);
        long maxCells = Math.max(1L, cacheBytes / Math.max(estimatedChunkBytes, 1L));
        ReadOnlyCachedCellImgOptions options = ReadOnlyCachedCellImgOptions.options()
                .cellDimensions(cellDimensions)
                .cacheType(CacheType.BOUNDED)
                .maxCacheSize(maxCells);

        ReadOnlyCachedCellImgFactory factory = new ReadOnlyCachedCellImgFactory(options);
        return (RandomAccessibleInterval<T>) factory.create(dims, type,
                cell -> loadCellFromSerialized(cell, endpointIndex, pbEffective));
    }

    /**
     * Fetch endpoints from server via GetFlightInfo when not provided in SerializedTensor.
     */
    private static List<SerializedEndpoint> fetchEndpointsViaGetFlightInfo(SerializedTensor pb) {
        TensorDescriptor descriptor = pb.getTensorDescriptor();

        // Parse location URI
        Location location = parseLocationUri(pb.getLocation());

        // Build TensorReadOption from descriptor's fields
        TensorReadOption.Builder readBuilder = TensorReadOption.newBuilder()
                .setTensorId(descriptor.getArrayId())
                .setWithMetadata(false);

        if (descriptor.hasSliceHint()) {
            readBuilder.setSliceHint(descriptor.getSliceHint());
        }
        for (long scale : descriptor.getScaleHintList()) {
            readBuilder.addScaleHint(scale);
        }
        if (!descriptor.getReductionMethod().isEmpty()) {
            readBuilder.setReductionMethod(descriptor.getReductionMethod());
        }

        // Build FlightCmd - use array_id as source_id (convention for single-tensor sources)
        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId(descriptor.getArrayId())
                .setTensorRead(readBuilder.build())
                .build();

        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        CredentialCallOption authOption = (pb.getAuthToken() != null && !pb.getAuthToken().isEmpty())
                ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + pb.getAuthToken()))
                : null;

        try (FlightClient client = FlightClient.builder(allocator, location).build()) {
            FlightInfo info = client.getInfo(FlightDescriptor.command(cmd.toByteArray()), authOption);

            List<SerializedEndpoint> endpoints = new ArrayList<>();
            for (FlightEndpoint endpoint : info.getEndpoints()) {
                TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
                ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
                SerializedEndpoint serializedEp = SerializedEndpoint.newBuilder()
                        .setTicket(ticket)
                        .setChunkBounds(bounds)
                        .build();
                endpoints.add(serializedEp);
            }

            LOGGER.fine("fetchEndpointsViaGetFlightInfo: got " + endpoints.size() + " endpoints");
            return endpoints;

        } catch (Exception e) {
            throw new IllegalStateException("Failed to fetch endpoints via GetFlightInfo", e);
        } finally {
            try {
                allocator.close();
            } catch (Exception e) {
                // Ignore allocator close errors
            }
        }
    }

    /**
     * Parse location URI string to Arrow Flight Location.
     */
    private static Location parseLocationUri(String locationStr) {
        if (locationStr.startsWith("grpc://")) {
            String uriPart = locationStr.substring(7);
            int colonIdx = uriPart.lastIndexOf(':');
            if (colonIdx > 0) {
                String host = uriPart.substring(0, colonIdx);
                int port = Integer.parseInt(uriPart.substring(colonIdx + 1));
                return Location.forGrpcInsecure(host, port);
            }
        } else if (locationStr.startsWith("grpc+tcp://")) {
            String uriPart = locationStr.substring(11);
            int colonIdx = uriPart.lastIndexOf(':');
            if (colonIdx > 0) {
                String host = uriPart.substring(0, colonIdx);
                int port = Integer.parseInt(uriPart.substring(colonIdx + 1));
                return Location.forGrpcInsecure(host, port);
            }
        }
        throw new IllegalArgumentException("Unsupported location URI scheme: " + locationStr);
    }

    /**
     * Load a cell from the Flight server using SerializedEndpoint data.
     */
    private static <T extends NativeType<T> & RealType<T>> void loadCellFromSerialized(
            SingleCellArrayImg<T, ?> cell,
            SerializedEndpointIndex endpointIndex,
            SerializedTensor pb) {

        long cellIndex = endpointIndex.indexFor(cell);
        SerializedEndpointData epData = endpointIndex.endpoints.get(cellIndex);
        if (epData == null) {
            throw new IllegalStateException("No endpoint found for cell index " + cellIndex);
        }

        TensorTicket ticket = epData.ticket;
        ChunkBounds bounds = epData.chunkBounds;
        double[] values = fetchChunkValuesStatic(
                pb.getLocation(),
                pb.getAuthToken(),
                ticket.getChunkId().toByteArray());
        writeChunk(cell.randomAccess(), bounds, values);
    }

    /**
     * Build index from SerializedEndpoint list for aligned chunk grid.
     */
    private static SerializedEndpointIndex buildSerializedEndpointIndex(
            SerializedTensor pb,
            long[] dims,
            int[] cellDimensions) {

        if (dims.length == 0 || pb.getEndpointsCount() == 0) {
            return null;
        }

        CellGrid grid = new CellGrid(dims, cellDimensions);
        Map<Long, SerializedEndpointData> endpointMap = new HashMap<>();
        long[] gridDimensions = grid.getGridDimensions();
        long[] gridPosition = new long[dims.length];
        long[] expectedMin = new long[dims.length];
        int[] expectedDimensions = new int[dims.length];

        for (SerializedEndpoint ep : pb.getEndpointsList()) {
            ChunkBounds bounds = ep.getChunkBounds();
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
            if (endpointMap.put(cellIndex, new SerializedEndpointData(ep.getTicket(), ep.getChunkBounds())) != null) {
                return null;
            }
        }

        if (endpointMap.size() != cellCount(gridDimensions)) {
            return null;
        }

        return new SerializedEndpointIndex(grid, cellDimensions, endpointMap);
    }

    /**
     * Materialize array for non-aligned endpoint layout.
     */
    @SuppressWarnings("unchecked")
    private static <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> materializeSerializedArray(
            SerializedTensor pb,
            long cacheBytes) {

        TensorDescriptor descriptor = pb.getTensorDescriptor();
        T type = (T) createType(descriptor.getDtype());
        long[] dims = toLongArray(descriptor.getShapeList());
        ArrayImg<T, ?> image = (ArrayImg<T, ?>) new ArrayImgFactory<>(type).create(dims);
        RandomAccess<T> access = image.randomAccess();

        for (SerializedEndpoint ep : pb.getEndpointsList()) {
            TensorTicket ticket = ep.getTicket();
            ChunkBounds bounds = ep.getChunkBounds();
            double[] values = fetchChunkValuesStatic(
                    pb.getLocation(),
                    pb.getAuthToken(),
                    ticket.getChunkId().toByteArray());
            writeChunk(access, bounds, values);
        }

        // Apply cropping if original_slice_hint present
        if (pb.hasOriginalSliceHint() && descriptor.hasSliceHint()) {
            SliceHint realized = descriptor.getSliceHint();
            SliceHint original = pb.getOriginalSliceHint();
            int ndim = descriptor.getShapeCount();
            long[] cropMin = new long[ndim];
            long[] cropMax = new long[ndim];
            for (int ax = 0; ax < ndim; ax++) {
                long reqStart = original.getStart(ax);
                long reqStop = original.getStop(ax);
                long retStart = realized.getStart(ax);
                long scale = 1L;
                if (descriptor.getScaleHintCount() > ax) {
                    scale = descriptor.getScaleHint(ax);
                }
                if (scale > 1L) {
                    cropMin[ax] = (reqStart - retStart) / scale;
                    cropMax[ax] = (reqStop - retStart + scale - 1L) / scale - 1L;
                } else {
                    cropMin[ax] = reqStart - retStart;
                    cropMax[ax] = reqStop - retStart - 1L;
                }
            }
            return Views.zeroMin(Views.interval(image, new FinalInterval(cropMin, cropMax)));
        }

        return image;
    }

    /**
     * Helper class to store endpoint message objects.
     */
    private static class SerializedEndpointData {
        final TensorTicket ticket;
        final ChunkBounds chunkBounds;

        SerializedEndpointData(TensorTicket ticket, ChunkBounds chunkBounds) {
            this.ticket = ticket;
            this.chunkBounds = chunkBounds;
        }
    }

    /**
     * Index mapping cell positions to serialized endpoint data.
     */
    private static class SerializedEndpointIndex {
        final CellGrid grid;
        final int[] nominalCellDimensions;
        final Map<Long, SerializedEndpointData> endpoints;

        SerializedEndpointIndex(
                CellGrid grid,
                int[] nominalCellDimensions,
                Map<Long, SerializedEndpointData> endpoints) {
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

    /**
     * Fetch chunk values using pooled FlightClient connection.
     */
    private static double[] fetchChunkValuesStatic(String locationStr, String authToken, byte[] chunkId) {
        LOGGER.fine("fetchChunkStatic: chunkId=" + bytesToHex(chunkId, 16));
        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFrom(chunkId))
                .build();

        // Parse location URI
        Location location;
        if (locationStr.startsWith("grpc://")) {
            String uriPart = locationStr.substring(7);
            int colonIdx = uriPart.lastIndexOf(':');
            if (colonIdx > 0) {
                String host = uriPart.substring(0, colonIdx);
                int port = Integer.parseInt(uriPart.substring(colonIdx + 1));
                location = Location.forGrpcInsecure(host, port);
            } else {
                throw new IllegalArgumentException("Invalid location URI: " + locationStr);
            }
        } else if (locationStr.startsWith("grpc+tcp://")) {
            String uriPart = locationStr.substring(11);
            int colonIdx = uriPart.lastIndexOf(':');
            if (colonIdx > 0) {
                String host = uriPart.substring(0, colonIdx);
                int port = Integer.parseInt(uriPart.substring(colonIdx + 1));
                location = Location.forGrpcInsecure(host, port);
            } else {
                throw new IllegalArgumentException("Invalid location URI: " + locationStr);
            }
        } else {
            throw new IllegalArgumentException("Unsupported location URI scheme: " + locationStr);
        }

        // Get pooled client
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        CredentialCallOption authOption = (authToken != null && !authToken.isEmpty())
                ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + authToken))
                : null;

        try (FlightClient client = FlightClient.builder(allocator, location).build()) {
            try (FlightStream stream = client.getStream(new Ticket(tensorTicket.toByteArray()), authOption)) {
                double[] values = new double[0];
                while (stream.next()) {
                    List<FieldVector> vectors = stream.getRoot().getFieldVectors();
                    if (vectors.isEmpty()) {
                        throw new IllegalStateException("Chunk payload did not contain any Arrow vectors");
                    }

                    FieldVector dataVector = stream.getRoot().getVector("data");
                    if (dataVector == null) {
                        throw new IllegalStateException("Chunk payload missing 'data' column");
                    }

                    int rowCount = stream.getRoot().getRowCount();
                    for (int row = 0; row < rowCount; row++) {
                        Object rowObj = dataVector.getObject(row);
                        if (!(rowObj instanceof List)) {
                            throw new IllegalStateException("Data column value is not a list: " + rowObj.getClass());
                        }
                        List<?> dataList = (List<?>) rowObj;
                        int offset = values.length;
                        values = Arrays.copyOf(values, offset + dataList.size());
                        for (int i = 0; i < dataList.size(); i++) {
                            values[offset + i] = asDouble(dataList.get(i));
                        }
                    }
                }
                return values;
            }
        } catch (Exception e) {
            throw new IllegalStateException("Failed to fetch chunk payload", e);
        } finally {
            try {
                allocator.close();
            } catch (Exception e) {
                // Ignore allocator close errors
            }
        }
    }

    @Override
    public void close() {
        LOGGER.info("Closing Flight client");
        try {
            client.close();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            allocator.close();
        }
    }

    /**
     * Check server health status via Flight action.
     *
     * Returns a map with health status information including:
     * - status: "SERVING" or other status string
     * - source_count: number of registered sources
     * - metadata_db_enabled: whether metadata database is enabled
     * - writable: whether server accepts uploads
     * - uptime_seconds: server uptime in seconds
     *
     * @return Map containing health status
     * @throws IOException If action fails
     */
    public Map<String, Object> healthCheck() throws IOException {
        org.apache.arrow.flight.Action action = new org.apache.arrow.flight.Action(
                "health",
                ByteString.EMPTY.toByteArray());

        java.util.Iterator<org.apache.arrow.flight.Result> iter = client.doAction(action, authOption);
        if (iter.hasNext()) {
            org.apache.arrow.flight.Result result = iter.next();
            byte[] body = result.getBody();
            if (body != null && body.length > 0) {
                return GSON.fromJson(new String(body, java.nio.charset.StandardCharsets.UTF_8),
                        new TypeToken<Map<String, Object>>() {
                        }.getType());
            }
        }

        Map<String, Object> unknown = new HashMap<>();
        unknown.put("status", "UNKNOWN");
        return unknown;
    }

    // Note: Upload API (uploadCellImg) not yet implemented for Java client.
    // Use the Python client for upload functionality.

    private RequestContext getTensorContext(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

        LOGGER.fine("getTensor: sourceId=" + sourceId + ", tensorId=" + tensorId);

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

        // Validate scale hint dimensionality if provided
        if (scaleHint != null && scaleHint.length > 0) {
            int rank = baseDescriptor.getShapeCount();
            if (scaleHint.length != rank) {
                throw new IllegalArgumentException(
                        "Scale hint dimensionality mismatch: expected " + rank
                                + ", got " + scaleHint.length);
            }
            for (int axis = 0; axis < scaleHint.length; axis++) {
                if (scaleHint[axis] <= 0) {
                    throw new IllegalArgumentException(
                            "Scale hint must be positive on axis " + axis + ": " + scaleHint[axis]);
                }
            }
        }

        // Normalize reduction method
        String normalizedReductionMethod = normalizeReductionMethod(reductionMethod);

        // Build TensorReadOption with flattened fields
        TensorReadOption.Builder readBuilder = TensorReadOption.newBuilder()
                .setTensorId(tensorId)
                .setWithMetadata(false);

        if (sliceHint != null) {
            readBuilder.setSliceHint(sliceHint);
        }
        if (scaleHint != null) {
            for (long s : scaleHint) {
                readBuilder.addScaleHint(s);
            }
        }
        if (normalizedReductionMethod != null && !normalizedReductionMethod.isEmpty()) {
            readBuilder.setReductionMethod(normalizedReductionMethod);
        }

        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId(sourceId)
                .setTensorRead(readBuilder.build())
                .build();
        FlightInfo info = client.getInfo(FlightDescriptor.command(cmd.toByteArray()), authOption);
        checkSchemaVersion(info);
        TensorDescriptor responseDescriptor = parseDescriptorUnchecked(info.getDescriptor().getCommand());

        // Cache the response descriptor
        descriptors.put(responseDescriptor.getArrayId(), responseDescriptor);

        return new RequestContext(responseDescriptor, info.getEndpoints());
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
        LOGGER.fine("fetchChunk: chunkId=" + bytesToHex(chunkId, 16));
        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFrom(chunkId))
                .build();

        try (FlightStream stream = client.getStream(new Ticket(tensorTicket.toByteArray()), authOption)) {
            double[] values = new double[0];
            while (stream.next()) {
                List<FieldVector> vectors = stream.getRoot().getFieldVectors();
                if (vectors.isEmpty()) {
                    throw new IllegalStateException("Chunk payload did not contain any Arrow vectors");
                }

                // Read "data" column - it's a ListArray with 1 row per chunk
                FieldVector dataVector = stream.getRoot().getVector("data");
                if (dataVector == null) {
                    throw new IllegalStateException("Chunk payload missing 'data' column");
                }

                // Each row is one chunk's data as a list
                int rowCount = stream.getRoot().getRowCount();
                for (int row = 0; row < rowCount; row++) {
                    Object rowObj = dataVector.getObject(row);
                    if (!(rowObj instanceof List)) {
                        throw new IllegalStateException("Data column value is not a list: " + rowObj.getClass());
                    }
                    List<?> dataList = (List<?>) rowObj;
                    int offset = values.length;
                    values = Arrays.copyOf(values, offset + dataList.size());
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

    private static String bytesToHex(byte[] bytes, int limit) {
        StringBuilder sb = new StringBuilder();
        int len = Math.min(bytes.length, limit);
        for (int i = 0; i < len; i++) {
            sb.append(String.format("%02x", bytes[i]));
        }
        if (bytes.length > limit) {
            sb.append("...");
        }
        return sb.toString();
    }

    private static void checkSchemaVersion(FlightInfo info) {
        java.util.Optional<Schema> schemaOpt = info.getSchemaOptional();
        if (!schemaOpt.isPresent()) {
            return;
        }
        Schema schema = schemaOpt.get();
        Map<String, String> metadata = schema.getCustomMetadata();
        if (metadata == null) {
            return;
        }
        String serverVersion = metadata.get("tensor_schema_version");
        if (serverVersion == null || serverVersion.isEmpty()) {
            return;
        }
        String clientVersion = getClientVersion();
        if (clientVersion == null) {
            return;
        }
        int[] serverParsed = parseVersion(serverVersion);
        int[] clientParsed = parseVersion(clientVersion);
        if (clientParsed[0] < serverParsed[0]
                || (clientParsed[0] == serverParsed[0] && clientParsed[1] < serverParsed[1])
                || (clientParsed[0] == serverParsed[0] && clientParsed[1] == serverParsed[1]
                        && clientParsed[2] < serverParsed[2])) {
            LOGGER.warning("Client version " + clientVersion + " is older than server schema version "
                    + serverVersion + ". Consider upgrading biopb client for compatibility.");
        }
    }

    private static String getClientVersion() {
        try {
            return System.getProperty("biopb.version",
                    java.util.jar.Manifest.class.getProtectionDomain().getCodeSource().getLocation().toString());
        } catch (Exception e) {
            // Try package version from manifest
            Package pkg = TensorFlightClient.class.getPackage();
            if (pkg != null && pkg.getImplementationVersion() != null) {
                return pkg.getImplementationVersion();
            }
            return null;
        }
    }

    private static int[] parseVersion(String version) {
        // Handle dev versions like "0.3.1.dev43+g..."
        String base = version.split(".dev")[0].split("+")[0];
        String[] parts = base.split("\\.");
        int major = parts.length > 0 ? Integer.parseInt(parts[0]) : 0;
        int minor = parts.length > 1 ? Integer.parseInt(parts[1]) : 0;
        int patch = parts.length > 2 ? Integer.parseInt(parts[2]) : 0;
        return new int[] { major, minor, patch };
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
            return GSON.fromJson(json, new TypeToken<Map<String, Object>>() {
            }.getType());
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