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
import org.apache.arrow.flight.FlightRuntimeException;
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

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.ReadOnlyCachedCellImgFactory;
import net.imglib2.cache.img.ReadOnlyCachedCellImgOptions;
import net.imglib2.cache.img.SingleCellArrayImg;
import net.imglib2.cache.img.optional.CacheOptions.CacheType;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import static biopb.tensor.TensorChunkCodec.cellCount;
import static biopb.tensor.TensorChunkCodec.createType;
import static biopb.tensor.TensorChunkCodec.estimateChunkBytes;
import static biopb.tensor.TensorChunkCodec.parseChunkBounds;
import static biopb.tensor.TensorChunkCodec.parseTicket;
import static biopb.tensor.TensorChunkCodec.toIntArray;
import static biopb.tensor.TensorChunkCodec.toLongArray;
import static biopb.tensor.TensorChunkCodec.writeChunk;

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
 * <p>A tensor is identified by its globally-unique {@code array_id} (the tensor
 * identity policy; see the top of {@code proto/biopb/tensor/descriptor.proto}):
 * either {@code source_id} for a single-tensor source or {@code source_id/field}
 * for a multi-tensor one. The array_id-first methods ({@link #getTensor(String)},
 * {@link #getDescriptor(String)}, {@link #getPhysicalScale(String)}) take that one
 * identifier; the older {@code (sourceId, tensorId)} overloads remain available.
 *
 * Usage:
 *
 * <pre>
 * TensorFlightClient client = new TensorFlightClient("localhost:8815");
 *
 * // List data sources (each may contain multiple tensors)
 * Map&lt;String, DataSourceDescriptor&gt; sources = client.listSources();
 *
 * // Access a tensor by its array_id ("source_id" or "source_id/field")
 * RandomAccessibleInterval&lt;UnsignedByteType&gt; arr = client.getTensor("my-source/tensor-0");
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
     * The server-side metadata database is mandatory (biopb/biopb#225), so any
     * standard tensor-server supports this. Only an embedded server explicitly
     * constructed without a metadata database rejects the query.
     *
     * @param sql SQL query (e.g., "SELECT source_id FROM sources WHERE source_url
     *            LIKE '%plate%'")
     * @return VectorSchemaRoot with query results (caller must close)
     * @throws IOException If query fails or the server has no metadata database
     *                     attached
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

        // Load each batch into its OWN transient root, then copy its rows into
        // result at the running offset. Loading every batch into result with a
        // single VectorLoader (the previous approach) overwrites row 0 each
        // time, so only the last batch survived -- multi-batch results were
        // silently corrupted.
        int offset = 0;
        for (ArrowRecordBatch batch : batches) {
            try (VectorSchemaRoot batchRoot = VectorSchemaRoot.create(schema, allocator)) {
                new VectorLoader(batchRoot).load(batch);
                int rows = batch.getLength();
                for (int i = 0; i < result.getFieldVectors().size(); i++) {
                    org.apache.arrow.vector.ValueVector srcVec = batchRoot.getVector(i);
                    org.apache.arrow.vector.ValueVector dstVec = result.getVector(i);
                    for (int row = 0; row < rows; row++) {
                        dstVec.copyFromSafe(row, offset + row, srcVec);
                    }
                }
                offset += rows;
            } finally {
                batch.close();
            }
        }

        result.setRowCount(totalRows);
        return result;
    }

    /**
     * Get source-level OME/vendor metadata as a map.
     *
     * @param sourceId Source identifier
     * @return The source's metadata map, or an empty map if it carries none
     * @throws IllegalArgumentException if the source is unknown
     * @throws IllegalStateException    if the source is unresolved (cloud /
     *                                  synced-folder) -- call {@link #resolve}
     *                                  first
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
            // Unresolved (cloud / synced-folder) source: tensors are unknown until
            // resolve. Don't return {} -- that conflates "unresolved" with
            // "resolved, no metadata". Steer to the explicit, consented resolve().
            throw unresolvedSourceError(sourceId);
        }

        // metadata_json is populated on the descriptor GetFlightInfo returns, so
        // we fetch it via the source's first tensor. The server wraps it as
        // {"type": ..., "dim_label": [...], "metadata": {...}}; we return the
        // inner "metadata" map.
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
     * Resolve an unresolved source and return its full {@link DataSourceDescriptor}.
     *
     * <p>An <i>unresolved</i> source is catalogued by URL only -- its
     * shape/dtype/field list are unknown until first access (it lists with
     * {@code data_resident} false and an empty {@code tensors}). The canonical
     * case is a cloud / synced-folder ("Files-On-Demand") source.
     *
     * <p>Resolving asks the server to hydrate it. For a dehydrated placeholder
     * this <b>downloads the whole file</b> -- a recall that can take minutes,
     * consume local disk, and fail when offline -- then reads its real shape,
     * dtype, and field list. This is the heavyweight, <i>consenting</i> operation
     * that catalog browsing ({@link #listSources} / {@link #querySources})
     * deliberately avoids; call it only when you intend to read the data.
     * Afterwards {@link #getTensor} and friends work normally. Idempotent.
     *
     * @param sourceId The source to resolve (e.g. {@code "onedrive_a3f2"})
     * @return The full DataSourceDescriptor with every tensor/field enumerated
     * @throws IOException If the action fails or the server returns no descriptor
     */
    public DataSourceDescriptor resolve(String sourceId) throws IOException {
        // One dedicated, streaming "resolve" action -- the single server entry
        // point that performs the (possibly minutes-long) recall and returns the
        // full descriptor directly. The action streams ResolveStreamMessage
        // progress heartbeats to keep the connection warm under proxy idle
        // timeouts; the terminal message carries the descriptor in its `result`
        // arm. (An empty body / bare serialized descriptor is also accepted for
        // back-compat with a server predating the progress envelope.)
        org.apache.arrow.flight.Action action = new org.apache.arrow.flight.Action(
                "resolve",
                sourceId.getBytes(java.nio.charset.StandardCharsets.UTF_8));

        DataSourceDescriptor desc = null;
        java.util.Iterator<org.apache.arrow.flight.Result> iter = client.doAction(action, authOption);
        while (iter.hasNext()) {
            byte[] body = iter.next().getBody();
            if (body == null || body.length == 0) {
                continue; // legacy empty-body heartbeat (pre-envelope server)
            }
            try {
                ResolveStreamMessage msg = ResolveStreamMessage.parseFrom(body);
                if (msg.getPayloadCase() == ResolveStreamMessage.PayloadCase.RESULT) {
                    desc = msg.getResult();
                } else if (msg.getPayloadCase() == ResolveStreamMessage.PayloadCase.PROGRESS) {
                    continue; // heartbeat (no progress callback on the Java client yet)
                } else {
                    // Legacy server: a non-empty body IS a bare serialized descriptor.
                    desc = DataSourceDescriptor.parseFrom(body);
                }
            } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                desc = DataSourceDescriptor.parseFrom(body); // legacy bare descriptor
            }
        }
        if (desc == null) {
            throw new IOException("resolve('" + sourceId
                    + "') returned no descriptor (server closed the stream without a result)");
        }
        sources.put(sourceId, desc);
        return desc;
    }

    /**
     * Hydrate-ahead: ask the server to recall all of a resolved multi-file
     * source's member files, so later reads are warm and never stall.
     *
     * <p>{@link #resolve} populates a source's metadata but, for a multi-file
     * cloud source (zarr / ome-zarr / ndtiff / tiff-sequence / micromanager),
     * leaves the bulk pixel data dehydrated -- each member file then recalls
     * one-at-a-time, slowly, the first time a read touches it. This walks the
     * source directory server-side and reads every file to force the sync
     * engine's recall; no pixels cross the wire, only progress. It is idempotent
     * (already-resident files are cheap local reads) and a no-op for a
     * single-file source (resolve already recalled it).
     *
     * @param sourceId The (already-resolved) source to warm.
     * @return The terminal {@link WarmProgress} snapshot (files/bytes made
     *         resident; {@code filesTotal == 0} for a no-op source).
     * @throws IOException If the action fails, the server is too old to support
     *         the {@code warm} action, or it returns no terminal status.
     */
    public WarmProgress warm(String sourceId) throws IOException {
        org.apache.arrow.flight.Action action = new org.apache.arrow.flight.Action(
                "warm",
                sourceId.getBytes(java.nio.charset.StandardCharsets.UTF_8));

        WarmProgress done = null;
        java.util.Iterator<org.apache.arrow.flight.Result> iter = client.doAction(action, authOption);
        while (iter.hasNext()) {
            byte[] body = iter.next().getBody();
            if (body == null || body.length == 0) {
                continue;
            }
            WarmStreamMessage msg = WarmStreamMessage.parseFrom(body);
            if (msg.getPayloadCase() == WarmStreamMessage.PayloadCase.DONE) {
                done = msg.getDone();
            }
            // PROGRESS arms are ignored (no progress callback on the Java client yet).
        }
        if (done == null) {
            throw new IOException("warm('" + sourceId
                    + "') returned no terminal status (server closed the stream without a 'done')");
        }
        return done;
    }

    /**
     * Fetch one tensor's {@link TensorDescriptor} by its globally-unique array_id.
     *
     * <p>A tensor is identified by its {@code array_id} alone (see the tensor
     * identity policy at the top of {@code proto/biopb/tensor/descriptor.proto}),
     * so this takes that one identifier rather than a {@code (sourceId, tensorId)}
     * pair. Works even when the source is beyond the (truncatable)
     * {@link #listSources} cap, and the result is cached. A bare {@code source_id}
     * (single-tensor source, or to anchor on a multi-tensor source's default/first
     * tensor) is accepted. To enumerate ALL tensors/scenes of a source, use
     * {@code listSources().get(sourceId).getTensorsList()} -- NOT this method.
     *
     * <p>This is a cheap probe -- it does NOT resolve. On an unresolved (cloud /
     * synced-folder) source it raises an error pointing at {@link #resolve}.
     *
     * @param arrayId Globally-unique tensor id, e.g. {@code "zarr_a3f2"} or
     *                {@code "aics_7f3/Image:0"}
     * @return The TensorDescriptor for that tensor
     */
    public TensorDescriptor getDescriptor(String arrayId) {
        // source_id is the slash-free prefix; the full array_id is the tensor_id.
        return fetchTensorDescriptor(sourceIdFromArrayId(arrayId), arrayId);
    }

    /**
     * Per-dimension physical pixel size + unit for a tensor.
     *
     * <p>Returns a {@link PhysicalScale} whose {@code scale} and {@code unit}
     * arrays are aligned with the tensor's {@code dim_labels} (source axis order),
     * or {@code null} when no physical sizes are known (an older server, or a
     * format that carries none).
     *
     * <p>{@code physical_scale}/{@code physical_unit} are {@code TensorDescriptor}
     * fields the server fills on every {@code GetFlightInfo} (issue #31), so this
     * reads the descriptor a prior {@link #getTensor} already cached -- no extra
     * RPC when it is cached, and it never requests the opt-in {@code metadata_json}
     * field on that same descriptor. (Contrast {@link #getSourceMetadata}, which
     * forces {@code with_metadata} to ship the whole OME tree; do not dig physical
     * sizes out of that -- this is the compact projection meant for display scale.)
     *
     * @param arrayId Globally-unique tensor id ({@code source_id} or
     *                {@code source_id/field}). A bare source id anchors on the
     *                source's default (first) tensor.
     * @return A PhysicalScale, or {@code null} if no physical scale is known
     */
    public PhysicalScale getPhysicalScale(String arrayId) {
        String sourceId = sourceIdFromArrayId(arrayId);
        TensorDescriptor desc = descriptors.get(arrayId);
        if (desc == null) {
            // Don't silently recall a whole cloud file just to read its pixel
            // size: if the source is known-unresolved, steer to resolve().
            DataSourceDescriptor cached = sources.get(sourceId);
            if (cached != null && cached.getTensorsList().isEmpty()) {
                throw unresolvedSourceError(sourceId);
            }
            desc = fetchTensorDescriptor(sourceId, arrayId);
        }
        if (desc.getPhysicalScaleCount() == 0) {
            return null;
        }
        double[] scale = new double[desc.getPhysicalScaleCount()];
        for (int i = 0; i < scale.length; i++) {
            scale[i] = desc.getPhysicalScale(i);
        }
        String[] unit = desc.getPhysicalUnitList().toArray(new String[0]);
        return new PhysicalScale(scale, unit);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor by its globally-unique array_id.
     *
     * @param arrayId Globally-unique tensor id ({@code source_id} or
     *                {@code source_id/field})
     * @param <T>     The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(String arrayId) {
        return getTensor(arrayId, (SliceHint) null);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor (by array_id) with a slice hint.
     *
     * @param arrayId   Globally-unique tensor id ({@code source_id} or
     *                  {@code source_id/field})
     * @param sliceHint Optional slice hint
     * @param <T>       The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String arrayId,
            SliceHint sliceHint) {
        return getTensor(arrayId, sliceHint, null, null);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor (by array_id) with scaled reads.
     *
     * @param arrayId         Globally-unique tensor id ({@code source_id} or
     *                        {@code source_id/field})
     * @param scaleHint       Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @param <T>             The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String arrayId,
            long[] scaleHint,
            String reductionMethod) {
        return getTensor(arrayId, (SliceHint) null, scaleHint, reductionMethod);
    }

    /**
     * Get a RandomAccessibleInterval for a tensor (by array_id) with all options.
     *
     * @param arrayId         Globally-unique tensor id ({@code source_id} or
     *                        {@code source_id/field})
     * @param sliceHint       Optional slice hint
     * @param scaleHint       Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @param <T>             The pixel type
     * @return RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String arrayId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {
        // array_id-first addressing: "source_id/field" routes to source_id with
        // the full array_id as tensor_id; a bare "source_id" leaves tensorId null
        // so getTensorContext resolves the source's sole/default tensor.
        String[] route = resolveArrayId(arrayId);
        return getTensor(route[0], route[1], sliceHint, scaleHint, reductionMethod);
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

        // tensorId may have arrived null (the bare-source_id array_id path);
        // pin it to the server-resolved array_id so the serializable wrapper can
        // re-fetch independently in another process.
        String resolvedTensorId = context.descriptor.getArrayId();

        // Crop to the originally requested region.
        // The server snaps slice_hint outward to lcm-aligned chunk boundaries, so
        // descriptor.shape may be larger than the requested extent.
        if (sliceHint != null && context.descriptor.hasSliceHint()) {
            rai = RegionCrop.cropToRequest(rai, sliceHint, context.descriptor.getSliceHint(),
                    context.descriptor.getScaleHintList());
        }

        // Return SerializableTensorImg wrapper for serialization support
        return new SerializableTensorImg<>(location, token, cacheBytes, sourceId, resolvedTensorId,
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

        ChunkGridIndex<SerializedEndpointData> endpointIndex = ChunkGridIndex.build(
                pbEffective.getEndpointsList(), dims, cellDimensions,
                SerializedEndpoint::getChunkBounds,
                ep -> new SerializedEndpointData(ep.getTicket(), ep.getChunkBounds()));

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

        Location location = LocationUris.parse(pb.getLocation());

        // Build TensorReadOption from descriptor's fields
        TensorReadOption.Builder readBuilder = TensorReadOption.newBuilder()
                .setTensorId(descriptor.getArrayId())
                .setWithMetadata(false)
                .setCompactGridOk(true);

        if (descriptor.hasSliceHint()) {
            readBuilder.setSliceHint(descriptor.getSliceHint());
        }
        for (long scale : descriptor.getScaleHintList()) {
            readBuilder.addScaleHint(scale);
        }
        if (!descriptor.getReductionMethod().isEmpty()) {
            readBuilder.setReductionMethod(descriptor.getReductionMethod());
        }

        // Build FlightCmd. Per the tensor identity policy, the descriptor's
        // array_id is "source_id" or "source_id/field"; the FlightCmd source_id
        // is the slash-free prefix (the tensor_id above carries the full
        // array_id, which the server reduces to the within-source field).
        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId(sourceIdFromArrayId(descriptor.getArrayId()))
                .setTensorRead(readBuilder.build())
                .build();

        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        CredentialCallOption authOption = (pb.getAuthToken() != null && !pb.getAuthToken().isEmpty())
                ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + pb.getAuthToken()))
                : null;

        try (FlightClient client = FlightClient.builder(allocator, location).build()) {
            FlightInfo info = client.getInfo(FlightDescriptor.command(cmd.toByteArray()), authOption);

            // resolveFlightEndpoints regenerates the endpoints if the server
            // answered compact (biopb/biopb#346); an explicit response passes
            // through unchanged.
            List<SerializedEndpoint> endpoints = new ArrayList<>();
            for (FlightEndpoint endpoint : CompactGrid.resolveFlightEndpoints(info)) {
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
     * Load a cell from the Flight server using SerializedEndpoint data.
     */
    private static <T extends NativeType<T> & RealType<T>> void loadCellFromSerialized(
            SingleCellArrayImg<T, ?> cell,
            ChunkGridIndex<SerializedEndpointData> endpointIndex,
            SerializedTensor pb) {

        long cellIndex = endpointIndex.indexFor(cell);
        SerializedEndpointData epData = endpointIndex.get(cellIndex);
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

        if (pb.hasOriginalSliceHint() && descriptor.hasSliceHint()) {
            return RegionCrop.cropToRequest(image, pb.getOriginalSliceHint(), descriptor.getSliceHint(),
                    descriptor.getScaleHintList());
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
     * Fetch chunk values using pooled FlightClient connection.
     */
    private static double[] fetchChunkValuesStatic(String locationStr, String authToken, byte[] chunkId) {
        LOGGER.fine("fetchChunkStatic: chunkId=" + bytesToHex(chunkId, 16));
        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFrom(chunkId))
                .build();

        Location location = LocationUris.parse(locationStr);

        // Get pooled client
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        CredentialCallOption authOption = (authToken != null && !authToken.isEmpty())
                ? new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + authToken))
                : null;

        try (FlightClient client = FlightClient.builder(allocator, location).build()) {
            try (FlightStream stream = client.getStream(new Ticket(tensorTicket.toByteArray()), authOption)) {
                double[] values = new double[0];
                while (stream.next()) {
                    // Unified binary chunk schema (biopb/biopb#293): "data" is one
                    // opaque byte[] per row, "dtype" names how to reinterpret it.
                    FieldVector dataVector = stream.getRoot().getVector("data");
                    FieldVector dtypeVector = stream.getRoot().getVector("dtype");
                    if (dataVector == null || dtypeVector == null) {
                        throw new IllegalStateException("Chunk payload missing 'data'/'dtype' column");
                    }

                    int rowCount = stream.getRoot().getRowCount();
                    for (int row = 0; row < rowCount; row++) {
                        Object rowObj = dataVector.getObject(row);
                        if (!(rowObj instanceof byte[])) {
                            throw new IllegalStateException("Data column value is not binary: "
                                    + (rowObj == null ? "null" : rowObj.getClass()));
                        }
                        Object dtypeObj = dtypeVector.getObject(row);
                        double[] decoded = ChunkDecoder.decodeChunkBytes((byte[]) rowObj,
                                dtypeObj == null ? "" : dtypeObj.toString());
                        int offset = values.length;
                        values = Arrays.copyOf(values, offset + decoded.length);
                        System.arraycopy(decoded, 0, values, offset, decoded.length);
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

    /**
     * Get upload status for a writable source.
     *
     * @param sourceId Source identifier returned by create_source()
     * @return Map containing source_id, state, expected_chunks, and uploaded_chunks
     * @throws IOException If the action fails
     */
    public Map<String, Object> getUploadStatus(String sourceId) throws IOException {
        org.apache.arrow.flight.Action action = new org.apache.arrow.flight.Action(
                "upload_status",
                sourceId.getBytes(java.nio.charset.StandardCharsets.UTF_8));

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
        unknown.put("source_id", sourceId);
        unknown.put("state", "UNKNOWN");
        unknown.put("expected_chunks", 0.0d);
        unknown.put("uploaded_chunks", 0.0d);
        return unknown;
    }

    /**
     * Get upload status for a registration-first SerializedTensor handle.
     *
     * This helper is intended for cache-backed handles returned before upload
     * completion, where tensor_descriptor.array_id is the source identifier.
     *
     * @param pb SerializedTensor handle from a registration-first flow
     * @return Map containing source_id, state, expected_chunks, and uploaded_chunks
     * @throws IOException If the action fails
     */
    public Map<String, Object> getUploadStatus(SerializedTensor pb) throws IOException {
        return getUploadStatus(getUploadSourceId(pb));
    }

    /**
     * Poll upload status until the source reports READY.
     *
     * @param sourceId Source identifier returned by create_source()
     * @param timeoutMillis Maximum time to wait before timing out
     * @param pollIntervalMillis Delay between status checks
     * @return Final upload status map when READY
     * @throws IOException If the upload fails, times out, or the action fails
     */
    public Map<String, Object> waitForUploadReady(
            String sourceId,
            long timeoutMillis,
            long pollIntervalMillis) throws IOException {

        long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
        while (true) {
            Map<String, Object> status = getUploadStatus(sourceId);
            Object stateValue = status.get("state");
            String state = stateValue != null ? stateValue.toString() : "UNKNOWN";
            if ("READY".equals(state)) {
                return status;
            }
            if ("FAILED".equals(state)) {
                throw new IOException("Upload failed for source '" + sourceId + "'");
            }
            if (System.nanoTime() >= deadline) {
                throw new IOException("Timed out waiting for upload readiness for source '" + sourceId + "'");
            }
            try {
                Thread.sleep(Math.max(0L, pollIntervalMillis));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while waiting for upload readiness", e);
            }
        }
    }

    /**
     * Poll upload status until a registration-first SerializedTensor is READY.
     *
     * @param pb SerializedTensor handle from a registration-first flow
     * @param timeoutMillis Maximum time to wait before timing out
     * @param pollIntervalMillis Delay between status checks
     * @return Final upload status map when READY
     * @throws IOException If the upload fails, times out, or the action fails
     */
    public Map<String, Object> waitForUploadReady(
            SerializedTensor pb,
            long timeoutMillis,
            long pollIntervalMillis) throws IOException {
        return waitForUploadReady(getUploadSourceId(pb), timeoutMillis, pollIntervalMillis);
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

        // Ensure sources are loaded; fall back to a direct server fetch when
        // listSources() didn't return this source (e.g. a truncated catalog).
        if (!sources.containsKey(sourceId)) {
            try {
                listSources();
            } catch (IOException e) {
                throw new IllegalStateException("Failed to list sources", e);
            }
        }

        DataSourceDescriptor sourceDesc = sources.get(sourceId);
        if (sourceDesc == null) {
            // Not in the (capped) listing -- probe the server directly so sources
            // beyond the list cap still resolve. Swallow a fetch failure and let
            // the clean "Source not found" below surface (matches the Python
            // client). An unresolved source still lists with empty tensors, so its
            // directive is raised from the tensorId==null branch, not here.
            try {
                TensorDescriptor td = fetchTensorDescriptor(sourceId, tensorId);
                sourceDesc = DataSourceDescriptor.newBuilder()
                        .setSourceId(sourceId)
                        .addTensors(td)
                        .build();
                sources.put(sourceId, sourceDesc);
            } catch (RuntimeException ignored) {
                // fall through to the clean error below
            }
        }
        if (sourceDesc == null) {
            throw new IllegalArgumentException("Source not found: " + sourceId);
        }

        // Resolve a null tensorId (the bare-source_id array_id path).
        if (tensorId == null) {
            int n = sourceDesc.getTensorsCount();
            if (n == 1) {
                tensorId = sourceDesc.getTensors(0).getArrayId();
            } else if (n == 0) {
                throw unresolvedSourceError(sourceId);
            } else {
                throw new IllegalArgumentException(
                        "Source '" + sourceId + "' has multiple tensors (" + n
                                + "); a within-source field must be specified (use \"source_id/field\")");
            }
        }

        // Find tensor descriptor to get shape for validation; fall back to a
        // direct server fetch when the cached source descriptor is stale/partial.
        TensorDescriptor baseDescriptor = null;
        for (TensorDescriptor desc : sourceDesc.getTensorsList()) {
            if (desc.getArrayId().equals(tensorId)) {
                baseDescriptor = desc;
                break;
            }
        }
        if (baseDescriptor == null) {
            // Stale/partial cached descriptor -- probe the server. Swallow a fetch
            // failure and surface the clean "not found" below (matches Python).
            try {
                baseDescriptor = fetchTensorDescriptor(sourceId, tensorId);
            } catch (RuntimeException ignored) {
                // fall through to the clean error below
            }
        }
        if (baseDescriptor == null) {
            throw new IllegalArgumentException(
                    "Tensor '" + tensorId + "' not found in source '" + sourceId + "'");
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
                .setWithMetadata(false)
                .setCompactGridOk(true);

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

        return new RequestContext(
                responseDescriptor, CompactGrid.resolveFlightEndpoints(info, responseDescriptor));
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

        ChunkGridIndex<FlightEndpoint> endpointIndex = ChunkGridIndex.build(
                context.endpoints, dims, cellDimensions,
                ep -> parseChunkBounds(ep.getAppMetadata()),
                ep -> ep);
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
            ChunkGridIndex<FlightEndpoint> endpointIndex) {

        long cellIndex = endpointIndex.indexFor(cell);
        FlightEndpoint endpoint = endpointIndex.get(cellIndex);
        if (endpoint == null) {
            throw new IllegalStateException("No Flight endpoint found for cell index " + cellIndex);
        }

        TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
        ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
        double[] values = fetchChunkValues(ticket.getChunkId().toByteArray());
        writeChunk(cell.randomAccess(), bounds, values);
    }

    private double[] fetchChunkValues(byte[] chunkId) {
        LOGGER.fine("fetchChunk: chunkId=" + bytesToHex(chunkId, 16));
        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFrom(chunkId))
                .build();

        try (FlightStream stream = client.getStream(new Ticket(tensorTicket.toByteArray()), authOption)) {
            double[] values = new double[0];
            while (stream.next()) {
                // Unified binary chunk schema (biopb/biopb#293): "data" is one
                // opaque byte[] per row, "dtype" names how to reinterpret it.
                FieldVector dataVector = stream.getRoot().getVector("data");
                FieldVector dtypeVector = stream.getRoot().getVector("dtype");
                if (dataVector == null || dtypeVector == null) {
                    throw new IllegalStateException("Chunk payload missing 'data'/'dtype' column");
                }

                int rowCount = stream.getRoot().getRowCount();
                for (int row = 0; row < rowCount; row++) {
                    Object rowObj = dataVector.getObject(row);
                    if (!(rowObj instanceof byte[])) {
                        throw new IllegalStateException("Data column value is not binary: "
                                + (rowObj == null ? "null" : rowObj.getClass()));
                    }
                    Object dtypeObj = dtypeVector.getObject(row);
                    double[] decoded = ChunkDecoder.decodeChunkBytes((byte[]) rowObj,
                            dtypeObj == null ? "" : dtypeObj.toString());
                    int offset = values.length;
                    values = Arrays.copyOf(values, offset + decoded.length);
                    System.arraycopy(decoded, 0, values, offset, decoded.length);
                }
            }
            return values;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to fetch chunk payload", e);
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
        // Advisory only: a malformed version string must never fail a read.
        try {
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
        } catch (RuntimeException e) {
            LOGGER.fine("Skipping schema version check: " + e);
        }
    }

    private static String getClientVersion() {
        // Explicit override wins; otherwise fall back to the packaged
        // implementation version. (Do NOT use it as System.getProperty's
        // default -- getProperty never throws, so the manifest fallback below
        // would be dead code and the jar URL would leak in as a "version".)
        String override = System.getProperty("biopb.version");
        if (override != null && !override.isEmpty()) {
            return override;
        }
        Package pkg = TensorFlightClient.class.getPackage();
        if (pkg != null && pkg.getImplementationVersion() != null) {
            return pkg.getImplementationVersion();
        }
        return null;
    }

    private static int[] parseVersion(String version) {
        // Handle dev versions like "0.3.1.dev43+g...". split() takes a regex, so
        // "." and "+" must be escaped (a bare "+" is a dangling-metacharacter
        // error, and "." matches any char).
        String base = version.split("\\.dev")[0].split("\\+")[0];
        String[] parts = base.split("\\.");
        int major = parts.length > 0 ? Integer.parseInt(parts[0]) : 0;
        int minor = parts.length > 1 ? Integer.parseInt(parts[1]) : 0;
        int patch = parts.length > 2 ? Integer.parseInt(parts[2]) : 0;
        return new int[] { major, minor, patch };
    }

    private static TensorDescriptor parseDescriptorUnchecked(byte[] bytes) {
        try {
            return TensorDescriptor.parseFrom(bytes);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to parse TensorDescriptor", e);
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

    private static String getUploadSourceId(SerializedTensor pb) {
        String arrayId = pb.getTensorDescriptor().getArrayId();
        if (arrayId == null || arrayId.isEmpty()) {
            throw new IllegalArgumentException("SerializedTensor tensor_descriptor.array_id is required");
        }
        // Uploaded sources are single-tensor (array_id == source_id), but derive
        // the prefix anyway so this is correct under the identity policy.
        return sourceIdFromArrayId(arrayId);
    }

    /**
     * Derive the source_id from a tensor's array_id.
     *
     * <p>Per the tensor identity policy, array_id is {@code source_id}
     * (single-tensor) or {@code source_id/field} (multi-tensor), and source_id
     * is globally unique and slash-free, so it is the prefix before the first
     * {@code '/'}.
     *
     * <p>Package-private for unit testing.
     */
    static String sourceIdFromArrayId(String arrayId) {
        int slash = arrayId.indexOf('/');
        return slash < 0 ? arrayId : arrayId.substring(0, slash);
    }

    /**
     * Split an array_id into a {@code {sourceId, tensorId}} route for getTensor.
     *
     * <p>A qualified {@code "source_id/field"} routes to
     * {@code {source_id, full-array_id}}; a bare {@code "source_id"} returns
     * {@code {source_id, null}}, leaving the tensorId unset so getTensorContext
     * resolves the source's sole/default tensor.
     *
     * <p>Package-private for unit testing.
     */
    static String[] resolveArrayId(String arrayId) {
        if (arrayId.indexOf('/') >= 0) {
            return new String[] { sourceIdFromArrayId(arrayId), arrayId };
        }
        return new String[] { arrayId, null };
    }

    /**
     * Directive error for reading an unresolved (cloud / synced-folder) source.
     *
     * <p>Shared by every read entry point so the guidance is uniform: name the
     * cure ({@link #resolve}) instead of leaking a bare internal "no tensors",
     * and -- for metadata queries like {@link #getPhysicalScale} -- raise rather
     * than silently recalling (downloading) the whole file. Resolving is the
     * heavyweight, consenting act; reads must not trigger it implicitly.
     */
    private static IllegalStateException unresolvedSourceError(String sourceId) {
        return new IllegalStateException(
                "Source '" + sourceId + "' is unresolved (no tensors listed yet). If "
                        + "this is a cloud / synced-folder source, call resolve('"
                        + sourceId + "') first to download and resolve it, then read it.");
    }

    /**
     * Fetch one tensor's descriptor directly from the server.
     *
     * <p>Backs {@link #getDescriptor} and {@link #getPhysicalScale}. Uses the
     * per-tensor GetFlightInfo RPC, which works even when the source is beyond
     * the (truncatable) {@link #listSources} cap. A null/empty tensorId, or a
     * tensorId equal to the sourceId, anchors on the source's default (first)
     * tensor via the empty-tensor_id path (the server resolves it, #44); a
     * within-source field is sent verbatim. This is a CHEAP probe: it does NOT
     * resolve -- an unresolved (cloud / synced-folder) source is restated as the
     * {@link #unresolvedSourceError} directive steering the caller to
     * {@link #resolve}, rather than triggering a download.
     *
     * <p>The descriptor is cached in {@code descriptors}, keyed by the
     * echoed-back array_id.
     */
    private TensorDescriptor fetchTensorDescriptor(String sourceId, String tensorId) {
        TensorReadOption.Builder readBuilder = TensorReadOption.newBuilder()
                .setWithMetadata(true);
        if (tensorId != null && !tensorId.equals(sourceId)) {
            readBuilder.setTensorId(tensorId);
        }
        FlightCmd cmd = FlightCmd.newBuilder()
                .setSourceId(sourceId)
                .setTensorRead(readBuilder.build())
                .build();
        FlightInfo info;
        try {
            info = client.getInfo(FlightDescriptor.command(cmd.toByteArray()), authOption);
        } catch (FlightRuntimeException exc) {
            // GetFlightInfo no longer resolves on serve: an unresolved (cloud /
            // synced-folder) source refuses with an "unresolved" error instead of
            // silently downloading. Restate it as the shared directive so the
            // caller is pointed at the explicit, consented resolve().
            String msg = exc.getMessage();
            if (msg != null && msg.toLowerCase().contains("unresolved")) {
                throw unresolvedSourceError(sourceId);
            }
            throw exc;
        }
        TensorDescriptor tensorDesc = parseDescriptorUnchecked(info.getDescriptor().getCommand());
        descriptors.put(tensorDesc.getArrayId(), tensorDesc);
        return tensorDesc;
    }

    /**
     * Per-dimension physical pixel size + unit for a tensor, as returned by
     * {@link #getPhysicalScale}. Both arrays are aligned with the tensor's
     * {@code dim_labels} (source axis order).
     */
    public static final class PhysicalScale {
        private final double[] scale;
        private final String[] unit;

        PhysicalScale(double[] scale, String[] unit) {
            this.scale = scale;
            this.unit = unit;
        }

        /** Physical pixel size per dimension, aligned with dim_labels. */
        public double[] getScale() {
            return scale;
        }

        /** Physical unit per dimension, aligned with dim_labels. */
        public String[] getUnit() {
            return unit;
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
