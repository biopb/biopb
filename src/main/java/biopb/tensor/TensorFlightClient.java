package biopb.tensor;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
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
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Schema;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import static biopb.tensor.TensorChunkCodec.toLongArray;

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
 * // Browse the catalog (SQL over the server's DuckDB)
 * VectorSchemaRoot rows = client.querySources("SELECT * FROM sources");
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

    private final FlightSession session;
    private final BufferAllocator allocator;
    private final FlightClient client;
    private final CredentialCallOption authOption;
    private final Location location;
    private final String token;
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
        this.session = new FlightSession(location, token);
        // Temporary aliases keep the legacy façade stable while its internals
        // move method-by-method to the session boundary.
        this.location = session.location();
        this.allocator = session.allocator();
        this.client = session.client();
        this.token = session.token();
        this.authOption = session.authOption();
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

    /** The columns a {@code sources} row carries, as a SELECT list. */
    static final String SOURCE_ROW_COLUMNS =
            "source_id, source_url, source_type, is_resolved, tensors";

    /**
     * List available data sources.
     *
     * @deprecated Use {@link #querySources}, which hands back rows and leaves
     *             the structure to you. The descriptors returned here also
     *             carry no {@code isResolved} -- the generated message has no
     *             field for it (biopb/biopb#1032).
     *             This is a thin wrapper around {@code SELECT ... FROM sources}
     *             that inherits the server's query row cap, so a large catalog
     *             comes back silently truncated -- and a browse is exactly where
     *             that matters.
     *
     * @return Map of source_id to DataSourceDescriptor
     */
    @Deprecated
    public Map<String, DataSourceDescriptor> listSources() throws IOException {
        Map<String, DataSourceDescriptor> result = new HashMap<>();
        try (VectorSchemaRoot root = querySources(
                "SELECT " + SOURCE_ROW_COLUMNS + " FROM sources ORDER BY source_id")) {
            for (DataSourceDescriptor sourceDesc : descriptorsFromRows(root)) {
                result.put(sourceDesc.getSourceId(), sourceDesc);
            }
        }
        LOGGER.info("listSources: returned " + result.size() + " sources");
        return result;
    }

    /**
     * One source's catalog row by id, or {@code null} when nothing answers to it.
     *
     * <p>One addressed catalog row, so a source past the browse cap still
     * resolves. The <b>caller must close</b> the returned root; it is the same
     * type {@link #querySources} hands back, because it is one of its results.
     */
    private VectorSchemaRoot fetchSourceRow(String sourceId) throws IOException {
        VectorSchemaRoot root = querySources(
                "SELECT " + SOURCE_ROW_COLUMNS + " FROM sources WHERE source_id = "
                        + sqlLiteral(sourceId));
        if (root.getRowCount() == 0) {
            root.close();
            return null;
        }
        return root;
    }

    /**
     * The {@code tensors} STRUCT[] of one row, as structural descriptors.
     *
     * <p>Structural only: array_id / dim_labels / shape / dtype. The transfer
     * chunk_shape belongs to the tensor-bound adapter and is answered by
     * GetFlightInfo (biopb/biopb#812), so a row has no column for it.
     */
    private static List<TensorDescriptor> tensorsFromRow(VectorSchemaRoot root, int index) {
        List<TensorDescriptor> out = new ArrayList<>();
        FieldVector tensors = root.getVector("tensors");
        Object list = tensors == null || tensors.isNull(index) ? null : tensors.getObject(index);
        if (!(list instanceof List)) {
            return out;
        }
        for (Object entry : (List<?>) list) {
            if (!(entry instanceof Map)) {
                continue;
            }
            Map<?, ?> t = (Map<?, ?>) entry;
            TensorDescriptor.Builder td = TensorDescriptor.newBuilder()
                    .setArrayId(String.valueOf(t.get("array_id")))
                    .setDtype(t.get("dtype") == null ? "" : String.valueOf(t.get("dtype")));
            Object labels = t.get("dim_labels");
            if (labels instanceof List) {
                for (Object l : (List<?>) labels) {
                    td.addDimLabels(String.valueOf(l));
                }
            }
            Object shape = t.get("shape");
            if (shape instanceof List) {
                for (Object d : (List<?>) shape) {
                    td.addShape(((Number) d).longValue());
                }
            }
            out.add(td.build());
        }
        return out;
    }

    /**
     * Whether the server has hydrated this row's source enough to know its tensors.
     *
     * <p>True for an absent column, the harmless direction for a monotonic flag; a
     * server whose table predates it fails the SELECT outright, so this only covers
     * a caller's own narrower projection.
     */
    private static boolean isResolved(VectorSchemaRoot root, int index) {
        Boolean flag = nullableBool(root.getVector("is_resolved"), index);
        return flag == null || flag;
    }

    /**
     * Rebuild lean {@link DataSourceDescriptor}s from {@code sources} catalog rows.
     *
     * @deprecated There is no replacement: a row is the data structure. Read the
     *             {@link VectorSchemaRoot} {@link #querySources} returns, and
     *             decode it into whatever suits you. This builds the generated
     *             message, which has no field for {@code is_resolved} and cannot
     *             gain one without a regenerate in every language
     *             (biopb/biopb#1032).
     */
    @Deprecated
    public static List<DataSourceDescriptor> descriptorsFromRows(VectorSchemaRoot root) {
        List<DataSourceDescriptor> out = new ArrayList<>();
        FieldVector sourceIds = root.getVector("source_id");
        FieldVector urls = root.getVector("source_url");
        FieldVector types = root.getVector("source_type");
        FieldVector resident = root.getVector("data_resident");
        for (int i = 0; i < root.getRowCount(); i++) {
            DataSourceDescriptor.Builder desc = DataSourceDescriptor.newBuilder()
                    .setSourceId(text(sourceIds, i))
                    .setSourceUrl(text(urls, i))
                    .setSourceType(text(types, i))
                    .setMetadataJson("")
                    .addAllTensors(tensorsFromRow(root, i));
            // No current server sends data_resident. Residency is a per-source
            // read on the descriptor GetFlightInfo returns (biopb/biopb#1048);
            // an older server still sends the column, and this decode answers
            // it identically against that server.
            Boolean res = nullableBool(resident, i);
            if (res != null) {
                desc.setDataResident(res);
            }
            // is_resolved is dropped, and that is the point: there is no field to
            // put it in. Read it off the row.
            out.add(desc.build());
        }
        return out;
    }

    private static String text(FieldVector vector, int index) {
        if (vector == null || vector.isNull(index)) {
            return "";
        }
        return String.valueOf(vector.getObject(index));
    }

    private static Boolean nullableBool(FieldVector vector, int index) {
        if (vector == null || vector.isNull(index)) {
            return null;
        }
        return (Boolean) vector.getObject(index);
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
        // One DoGet on the `catalog` flight: the ticket carries the SQL itself.
        TensorTicket ticket = TensorTicket.newBuilder()
                .setCatalogQuery(CatalogQuery.newBuilder().setSql(sql).build())
                .build();
        Schema schema;
        List<ArrowRecordBatch> batches = new ArrayList<>();
        try (FlightStream stream = session.getStream(new Ticket(ticket.toByteArray()))) {
            schema = stream.getSchema();

            // Truncation is the server's own flag on the stream's schema metadata.
            Map<String, String> metadata = schema.getCustomMetadata();
            if (metadata != null && metadata.containsKey("returned_rows")) {
                String returned = metadata.get("returned_rows");
                if (Boolean.parseBoolean(metadata.get("truncated"))) {
                    LOGGER.info("querySources: result truncated, returned " + returned
                            + " of " + metadata.get("total_rows") + " rows");
                } else {
                    LOGGER.info("querySources: returned " + returned + " rows");
                }
            }

            // Materialize all batches using ArrowRecordBatch (Arrow 18 API); the
            // stream (and its root) is released here, the clones are ours.
            while (stream.next()) {
                VectorUnloader unloader = new VectorUnloader(stream.getRoot());
                ArrowRecordBatch batch = unloader.getRecordBatch().cloneWithTransfer(allocator);
                batches.add(batch);
            }
        } catch (Exception e) {
            for (ArrowRecordBatch batch : batches) {
                batch.close();
            }
            if (e instanceof FlightRuntimeException) {
                throw TensorErrorMapper.map((FlightRuntimeException) e);
            }
            if (e instanceof IOException) {
                throw (IOException) e;
            }
            throw new IOException("querySources failed: " + e.getMessage(), e);
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
     * <p>Source-scoped, and read from the source's own catalog row: this is the
     * metadata the format carries for the whole container. A <i>field's</i> own
     * extras (an OME-Zarr HCS field's OME block, an EMD signal's
     * {@code original_metadata}) are per-tensor and are not merged in here.
     *
     * @param sourceId Source identifier
     * @return The source's metadata map, or an empty map if it carries none
     * @throws IllegalArgumentException if the source is unknown
     * @throws IllegalStateException    if the source is unresolved (cloud /
     *                                  synced-folder) -- call {@link #resolve}
     *                                  first
     */
    public Map<String, Object> getSourceMetadata(String sourceId) throws IOException {
        // The column IS the answer: the server calls the adapter's get_metadata()
        // once at registration to fill it and reads it back from the catalog on
        // the serve path rather than recomputing (biopb/biopb#253).
        String metadataJson = null;
        boolean found = false;
        boolean resolved = false;
        try (VectorSchemaRoot root = querySources(
                "SELECT is_resolved, metadata_json FROM sources WHERE source_id = " + sqlLiteral(sourceId))) {
            if (root.getRowCount() > 0) {
                found = true;
                Boolean isRes = nullableBool(root.getVector("is_resolved"), 0);
                resolved = isRes == null || isRes;
                FieldVector meta = root.getVector("metadata_json");
                metadataJson = meta == null || meta.isNull(0) ? null : String.valueOf(meta.getObject(0));
            }
        }
        if (!found) {
            throw new IllegalArgumentException("Source not found: " + sourceId);
        }
        if (!resolved) {
            // Unresolved (cloud / synced-folder) source: tensors are unknown until
            // resolve. Don't return {} -- that conflates "unresolved" with
            // "resolved, no metadata". Steer to the explicit, consented resolve().
            // The flag, not an empty tensor list -- a source can resolve cleanly
            // and hold nothing readable (biopb/biopb#1032).
            throw unresolvedSourceError(sourceId);
        }
        return parseMetadataJson(metadataJson);
    }

    /** Quote a string for the catalog's SQL surface, which takes no parameters. */
    static String sqlLiteral(String value) {
        return "'" + value.replace("'", "''") + "'";
    }

    /**
     * Resolve an unresolved source and return its {@code sources} catalog row.
     *
     * <p>An <i>unresolved</i> source is catalogued by URL only -- its
     * shape/dtype/field list are unknown until first access (it lists with
     * {@code is_resolved} false and an empty {@code tensors}). The canonical
     * case is a cloud / synced-folder ("Files-On-Demand") source.
     *
     * <p>Resolving asks the server to hydrate it. For a dehydrated placeholder
     * this <b>downloads the whole file</b> -- a recall that can take minutes,
     * consume local disk, and fail when offline -- then reads its real shape,
     * dtype, and field list. This is the heavyweight, <i>consenting</i> operation
     * that catalog browsing ({@link #querySources}) deliberately avoids; call it only when you intend to read the data.
     * Afterwards {@link #getTensor} and friends work normally. Idempotent.
     *
     * @param sourceId The source to resolve (e.g. {@code "onedrive_a3f2"})
     * @return The source's {@code sources} row, with every tensor enumerated --
     *         one {@link VectorSchemaRoot}, the same type {@link #querySources}
     *         returns, which the <b>caller must close</b>. A row, not a type
     *         this SDK picked: every client can already decode one, and what
     *         you decode it into stays yours (biopb/biopb#1032).
     *         <p>Unlike {@link #warm}, which returns a status because residency
     *         is not a durable catalog fact and its file counts exist nowhere
     *         else, this returns the result: resolving is defined by what it
     *         writes to the row.
     * @throws IOException If the action fails or the server returns no row
     */
    public VectorSchemaRoot resolve(String sourceId) throws IOException {
        return resolve(sourceId, null, null);
    }

    /**
     * Resolve an unresolved source with optional progress and cancellation hooks.
     *
     * <p>Both callbacks run on the calling thread after each streamed action
     * message. Returning true from {@code shouldCancel} stops consuming the
     * action stream; the server may finish its recall independently and cache
     * the result for a later call.
     *
     * @param sourceId source to resolve
     * @param onProgress receives server progress heartbeats, or null
     * @param shouldCancel polled once per action message, or null
     * @return the terminal catalog row; caller must close it
     */
    public VectorSchemaRoot resolve(
            String sourceId,
            Consumer<ResolveProgress> onProgress,
            BooleanSupplier shouldCancel) throws IOException {
        // One dedicated, streaming "resolve" action -- the single server entry
        // point that performs the (possibly minutes-long) recall. The action
        // streams ResolveStreamMessage progress heartbeats to keep the
        // connection warm under proxy idle timeouts; the terminal message
        // carries the source's now-concrete catalog row as an Arrow IPC stream,
        // handed back as-is.
        org.apache.arrow.flight.Action action = new org.apache.arrow.flight.Action(
                "resolve",
                sourceId.getBytes(java.nio.charset.StandardCharsets.UTF_8));

        VectorSchemaRoot row = null;
        java.util.Iterator<org.apache.arrow.flight.Result> iter = session.doAction(action);
        try {
            while (iter.hasNext()) {
                byte[] body = iter.next().getBody();
                if (body == null || body.length == 0) {
                    continue;
                }
                ResolveStreamMessage msg = ResolveStreamMessage.parseFrom(body);
                if (shouldCancel != null && shouldCancel.getAsBoolean()) {
                    throw new TensorOperationCancelledException("resolve", sourceId);
                }
                if (msg.getPayloadCase() == ResolveStreamMessage.PayloadCase.PROGRESS) {
                    if (onProgress != null) {
                        onProgress.accept(msg.getProgress());
                    }
                    continue;
                }
                if (msg.getPayloadCase() != ResolveStreamMessage.PayloadCase.SOURCE_ROW) {
                    continue;
                }
                try (ArrowStreamReader reader = new ArrowStreamReader(
                        new ByteArrayInputStream(msg.getSourceRow().toByteArray()), allocator)) {
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot streamed = reader.getVectorSchemaRoot();
                        // Take a copy while the batch is still loaded; the reader
                        // owns those buffers and frees them on close.
                        // the reader owns those buffers and frees them on close
                        // (the same reason querySources clones its batches).
                        if (row != null) {
                            row.close();
                        }
                        row = copyOf(streamed);
                    }
                }
            }
        } catch (Exception e) {
            if (row != null) {
                row.close();
            }
            if (e instanceof FlightRuntimeException) {
                throw TensorErrorMapper.map((FlightRuntimeException) e);
            }
            if (e instanceof TensorOperationCancelledException) {
                throw (TensorOperationCancelledException) e;
            }
            if (e instanceof IOException) {
                throw (IOException) e;
            }
            throw new IOException("resolve failed: " + e.getMessage(), e);
        }
        if (row == null) {
            throw new IOException("resolve('" + sourceId
                    + "') returned no catalog row (server closed the stream without a result)");
        }
        return row;
    }

    /** A standalone copy of {@code src}, owned by this client's allocator. */
    private VectorSchemaRoot copyOf(VectorSchemaRoot src) {
        VectorUnloader unloader = new VectorUnloader(src);
        ArrowRecordBatch batch = unloader.getRecordBatch().cloneWithTransfer(allocator);
        try {
            VectorSchemaRoot out = VectorSchemaRoot.create(src.getSchema(), allocator);
            new VectorLoader(out).load(batch);
            return out;
        } finally {
            batch.close();
        }
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
     * single-file source (resolve already recalled it). A remote-url source --
     * an object store, or a {@code grpc://} mirror -- fails instead: nothing on
     * the serving machine can be made resident (biopb/biopb#1035).
     *
     * @param sourceId The (already-resolved) source to warm.
     * @return The terminal {@link WarmProgress} snapshot (files/bytes made
     *         resident). {@code filesTotal == 0} means the source was local and
     *         had nothing to warm, i.e. single-file; "not applicable" raises.
     * @throws IOException If the action fails, the server is too old to support
     *         the {@code warm} action, or it returns no terminal status.
     */
    public WarmProgress warm(String sourceId) throws IOException {
        return warm(sourceId, null, null);
    }

    /**
     * Warm a source with optional progress and cancellation hooks.
     *
     * @param sourceId source to warm
     * @param onProgress receives non-terminal warm progress, or null
     * @param shouldCancel polled once per action message, or null
     * @return the terminal progress snapshot
     */
    public WarmProgress warm(
            String sourceId,
            Consumer<WarmProgress> onProgress,
            BooleanSupplier shouldCancel) throws IOException {
        org.apache.arrow.flight.Action action = new org.apache.arrow.flight.Action(
                "warm",
                sourceId.getBytes(java.nio.charset.StandardCharsets.UTF_8));

        WarmProgress done = null;
        java.util.Iterator<org.apache.arrow.flight.Result> iter = session.doAction(action);
        while (iter.hasNext()) {
            byte[] body = iter.next().getBody();
            if (body == null || body.length == 0) {
                continue;
            }
            WarmStreamMessage msg = WarmStreamMessage.parseFrom(body);
            if (shouldCancel != null && shouldCancel.getAsBoolean()) {
                throw new TensorOperationCancelledException("warm", sourceId);
            }
            if (msg.getPayloadCase() == WarmStreamMessage.PayloadCase.PROGRESS) {
                if (onProgress != null) {
                    onProgress.accept(msg.getProgress());
                }
            } else if (msg.getPayloadCase() == WarmStreamMessage.PayloadCase.DONE) {
                done = msg.getDone();
            }
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
     * pair. Works even when the source is beyond the server's query row cap, and
     * the result is cached. A bare {@code source_id} (single-tensor source, or to
     * anchor on a multi-tensor source's default/first tensor) is accepted. To
     * enumerate ALL tensors/scenes of a source, read its catalog row's
     * {@code tensors} column -- NOT this method.
     *
     * <p>This is a cheap probe -- it does NOT resolve. On an unresolved (cloud /
     * synced-folder) source it raises an error pointing at {@link #resolve}.
     *
     * @param arrayId Globally-unique tensor id, e.g. {@code "zarr_a3f2"} or
     *                {@code "aics_7f3/Image:0"}
     * @return The TensorDescriptor for that tensor
     */
    public TensorDescriptor getDescriptor(String arrayId) {
        return describe(arrayId, readMask("pyramid"));
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
     * ships the whole OME tree; do not dig physical sizes out of that -- this is
     * the compact projection meant for display scale.)
     *
     * @param arrayId Globally-unique tensor id ({@code source_id} or
     *                {@code source_id/field}). A bare source id anchors on the
     *                source's default (first) tensor.
     * @return A PhysicalScale, or {@code null} if no physical scale is known
     */
    public PhysicalScale getPhysicalScale(String arrayId) {
        TensorDescriptor desc = describe(arrayId, readMask());
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
        return getTensor(sourceIdFromArrayId(arrayId), arrayId, sliceHint, scaleHint, reductionMethod);
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
     * @return lazy RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String sourceId,
            String tensorId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

        String arrayId = tensorId == null || tensorId.isEmpty() ? sourceId : tensorId;
        RequestContext context = planRead(arrayId, sliceHint, scaleHint, reductionMethod);
        RandomAccessibleInterval<T> rai = new Imglib2TensorFactory(session, cacheBytes)
                .create(context.info);

        // Crop to the originally requested region.
        // The server snaps slice_hint outward to lcm-aligned chunk boundaries, so
        // descriptor.shape may be larger than the requested extent.
        if (sliceHint != null && context.descriptor.hasSliceHint()) {
            rai = RegionCrop.cropToRequest(rai, sliceHint, context.descriptor.getSliceHint(),
                    context.descriptor.getScaleHintList());
        }

        // Preserve source compatibility while externalizing only the v2 handle.
        return new SerializableTensorImg<>(serializedTensorOf(context.info), cacheBytes, rai);
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
        String arrayId = tensorId == null || tensorId.isEmpty() ? sourceId : tensorId;
        RequestContext context = planRead(arrayId, sliceHint, scaleHint, reductionMethod);

        // The plan is Arrow's own FlightInfo, carried whole; only where and as
        // whom to read it is ours to add.
        return serializedTensorOf(context.info);
    }

    private SerializedTensor serializedTensorOf(FlightInfo info) {
        SerializedTensor.Builder builder = SerializedTensor.newBuilder()
                .setLocation(location.getUri().toString())
                .setFlightInfo(ByteString.copyFrom(info.serialize()));
        if (token != null && !token.isEmpty()) {
            builder.setAuthToken(token);
        }
        return builder.build();
    }

    /** The plan a SerializedTensor carries: its serialized Arrow FlightInfo. */
    public static FlightInfo flightInfoOf(SerializedTensor pb) {
        try {
            return FlightInfo.deserialize(pb.getFlightInfo().asReadOnlyByteBuffer());
        } catch (IOException | URISyntaxException e) {
            throw new IllegalArgumentException("SerializedTensor.flight_info is not a FlightInfo", e);
        }
    }

    /** The resolved descriptor a SerializedTensor's plan names. */
    public static TensorDescriptor descriptorOf(SerializedTensor pb) {
        return parseDescriptorUnchecked(flightInfoOf(pb).getDescriptor().getCommand());
    }

    /**
     * The lazy imglib2 array a SerializedTensor describes.
     *
     * The one consumer-side helper. The handle is a FlightInfo plus where and
     * as whom to read it. Its plan is consumed directly on first access; only
     * the documented endpoint-less progressive-discovery plan is refreshed.
     *
     * @param pb          SerializedTensor protobuf object
     * @param cacheBytes  Maximum cache size in bytes
     * @param <T>         The pixel type
     * @return RandomAccessibleInterval with lazy chunk loading
     */
    public static <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> tensorFromPb(
            SerializedTensor pb,
            long cacheBytes) {

        return new SerializableTensorImg<>(pb, cacheBytes, null);
    }

    @Override
    public void close() {
        LOGGER.info("Closing Flight client");
        session.close();
    }

    /**
     * Check server health status via Flight action.
     *
     * Returns a map with health status information including:
     * - status: "SERVING" or other status string
     * - source_count: number of registered sources
     * - metadata_db_enabled: whether the server offers a catalog (false means it
     *   serves sources by id alone and every catalog surface refuses)
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

        java.util.Iterator<org.apache.arrow.flight.Result> iter = session.doAction(action);
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
     * @param sourceId Source identifier returned by create_tensor()
     * @return Map containing source_id, state, expected_chunks, uploaded_chunks and reason
     * @throws IOException If the action fails
     */
    public Map<String, Object> getUploadStatus(String sourceId) throws IOException {
        // A describe-only GetFlightInfo, not an action: doAction takes the
        // server-wide token, while the caller waiting on a fast-return result
        // holds a per-source read capability and nothing else (biopb/biopb#1048).
        // This is deliberately a fresh describe: caching the one field whose
        // purpose is freshness would defeat the poll.
        TensorReadOption readOpt = TensorReadOption.newBuilder()
                .setArrayId(sourceId)
                .setFields(readMask("upload_status"))
                .build();
        FlightRequest cmd = FlightRequest.newBuilder().setTensorRead(readOpt).build();

        TensorDescriptor desc;
        try {
            FlightInfo info = session.getInfo(FlightDescriptor.command(cmd.toByteArray()));
            desc = parseDescriptorUnchecked(info.getDescriptor().getCommand());
        } catch (TensorNotFoundException | FlightRuntimeException exc) {
            // An id the server does not serve. UNKNOWN already means "no upload
            // record here", and an unregistered source is the strongest form of
            // that, so it is the same answer rather than a transport error.
            return unknownUploadStatus(sourceId);
        }
        if (!desc.hasUploadStatus()) {
            // Registered, but not an upload -- an ordinary catalog source.
            return unknownUploadStatus(sourceId);
        }
        UploadStatus status = desc.getUploadStatus();
        Map<String, Object> out = new HashMap<>();
        out.put("source_id", sourceId);
        out.put("state", status.getState().name());
        out.put("expected_chunks", (double) status.getExpectedChunks());
        out.put("uploaded_chunks", (double) status.getUploadedChunks());
        out.put("reason", status.getReason());
        return out;
    }

    /** The answer for a source the server tracks no upload for. */
    private static Map<String, Object> unknownUploadStatus(String sourceId) {
        Map<String, Object> unknown = new HashMap<>();
        unknown.put("source_id", sourceId);
        unknown.put("state", "UNKNOWN");
        unknown.put("expected_chunks", 0.0d);
        unknown.put("uploaded_chunks", 0.0d);
        unknown.put("reason", "");
        return unknown;
    }

    // Note: Upload API (uploadCellImg) not yet implemented for Java client.
    // Use the Python client for upload functionality.

    /**
     * Plan one v2 read. The response descriptor, including its transfer grid,
     * belongs to this request and is never cached.
     */
    private RequestContext planRead(
            String arrayId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {
        TensorReadOption.Builder read = TensorReadOption.newBuilder()
                .setArrayId(arrayId)
                .setFields(readMask("endpoints"));
        if (sliceHint != null) {
            read.setSliceHint(sliceHint);
        }
        if (scaleHint != null) {
            for (long scale : scaleHint) {
                read.addScaleHint(scale);
            }
        }
        String normalized = normalizeReductionMethod(reductionMethod);
        if (!normalized.isEmpty()) {
            read.setReductionMethod(normalized);
        }

        FlightRequest request = FlightRequest.newBuilder().setTensorRead(read.build()).build();
        FlightInfo info = session.getInfo(FlightDescriptor.command(request.toByteArray()));
        checkSchemaVersion(info);
        TensorDescriptor descriptor = parseDescriptorUnchecked(info.getDescriptor().getCommand());
        refuseAmbiguousDefault(arrayId, descriptor.getArrayId());
        return new RequestContext(descriptor, info);
    }

    /** Describe one tensor without requesting the O(chunks) endpoint plan. */
    private TensorDescriptor describe(
            String arrayId, com.google.protobuf.FieldMask fields) {
        TensorReadOption read = TensorReadOption.newBuilder()
                .setArrayId(arrayId)
                .setFields(fields)
                .build();
        FlightRequest request = FlightRequest.newBuilder().setTensorRead(read).build();
        FlightInfo info = session.getInfo(FlightDescriptor.command(request.toByteArray()));
        checkSchemaVersion(info);
        return parseDescriptorUnchecked(info.getDescriptor().getCommand());
    }

    /** A bare id is valid only when the catalog can confirm it is unambiguous. */
    private void refuseAmbiguousDefault(String requestedArrayId, String resolvedArrayId) {
        if (requestedArrayId.equals(resolvedArrayId) || requestedArrayId.indexOf('/') >= 0) {
            return;
        }
        try (VectorSchemaRoot row = fetchSourceRow(sourceIdFromArrayId(requestedArrayId))) {
            if (row != null && tensorsFromRow(row, 0).size() > 1) {
                throw new IllegalArgumentException(
                        "Source '" + requestedArrayId + "' has multiple tensors; use a qualified array_id");
            }
        } catch (IOException | FlightRuntimeException ignored) {
            // A capability token cannot browse the catalog. The server's planned
            // response is still usable, and older servers cannot state substitution.
        }
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
     * Legacy routing helper retained for callers migrating to the array_id-only
     * API. New read planning sends the array_id unchanged.
     */
    @Deprecated
    static String[] resolveArrayId(String arrayId) {
        return arrayId.indexOf('/') >= 0
                ? new String[] { sourceIdFromArrayId(arrayId), arrayId }
                : new String[] { arrayId, null };
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
     * A read mask naming the optional parts wanted.
     *
     * <p>The wire takes a {@link com.google.protobuf.FieldMask} (Flight protocol
     * v3). Every part is opt-in, including {@code endpoints} -- the O(chunks)
     * read plan -- so an empty mask is a describe. The {@code with_*} bools this
     * replaced defaulted that most expensive part to on.
     */
    private static com.google.protobuf.FieldMask readMask(String... paths) {
        com.google.protobuf.FieldMask.Builder b = com.google.protobuf.FieldMask.newBuilder();
        for (String path : paths) {
            b.addPaths(path);
        }
        return b.build();
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
        final FlightInfo info;

        RequestContext(TensorDescriptor descriptor, FlightInfo info) {
            this.descriptor = parseDescriptorUnchecked(descriptor.toByteArray());
            this.info = info;
        }
    }
}
