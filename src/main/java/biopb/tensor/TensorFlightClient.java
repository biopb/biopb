package biopb.tensor;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import io.grpc.Context;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.PutResult;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.SyncPutListener;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
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

import biopb.image.RoiAnnotation;
import biopb.image.RoiDeleteResult;
import biopb.image.RoiListResult;
import biopb.image.RoiPruneRequest;
import biopb.image.RoiPruneResult;
import biopb.image.RoiPutResult;
import biopb.image.RoiSetInfo;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;


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
 * identifier; there is no {@code (sourceId, tensorId)} form.
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
    private final Location location;
    private final String token;
    private final long cacheBytes;
    private final TensorUploads uploads;

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
        // Every RPC goes through the session; what is read back out of it here
        // is what this class answers to callers (location / token) and the
        // allocator its Arrow results are owned by.
        this.location = session.location();
        this.allocator = session.allocator();
        this.token = session.token();
        this.cacheBytes = cacheBytes;
        this.uploads = new TensorUploads(session);
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
        VectorSchemaRoot[] row = { null };
        try {
            streamAction("resolve", sourceId.getBytes(StandardCharsets.UTF_8),
                    ResolveStreamMessage.parser(),
                    message -> {
                        if (shouldCancel != null && shouldCancel.getAsBoolean()) {
                            throw new TensorOperationCancelledException("resolve", sourceId);
                        }
                        if (message.getPayloadCase() == ResolveStreamMessage.PayloadCase.PROGRESS) {
                            if (onProgress != null) {
                                onProgress.accept(message.getProgress());
                            }
                        } else if (message.getPayloadCase() == ResolveStreamMessage.PayloadCase.SOURCE_ROW) {
                            VectorSchemaRoot fresh = readSourceRow(message.getSourceRow());
                            if (fresh != null) {
                                closeQuietly(row[0]);
                                row[0] = fresh;
                            }
                        }
                        return true;
                    },
                    null);
        } catch (UncheckedIOException error) {
            closeQuietly(row[0]);
            throw error.getCause();
        } catch (RuntimeException | IOException error) {
            closeQuietly(row[0]);
            throw error;
        }
        if (row[0] == null) {
            throw new IOException("resolve('" + sourceId
                    + "') returned no catalog row (server closed the stream without a result)");
        }
        return row[0];
    }

    /**
     * The catalog row a terminal {@code resolve} message carries.
     *
     * <p>Copied out while the batch is still loaded: the reader owns those
     * buffers and frees them on close, the same reason {@link #querySources}
     * clones its batches.
     */
    private VectorSchemaRoot readSourceRow(ByteString ipc) {
        VectorSchemaRoot row = null;
        try (ArrowStreamReader reader = new ArrowStreamReader(
                new ByteArrayInputStream(ipc.toByteArray()), allocator)) {
            while (reader.loadNextBatch()) {
                VectorSchemaRoot fresh = copyOf(reader.getVectorSchemaRoot());
                closeQuietly(row);
                row = fresh;
            }
        } catch (IOException error) {
            closeQuietly(row);
            throw new UncheckedIOException(error);
        } catch (RuntimeException error) {
            closeQuietly(row);
            throw error;
        }
        return row;
    }

    private static void closeQuietly(VectorSchemaRoot root) {
        if (root != null) {
            root.close();
        }
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
     * @throws IOException If the action fails or it returns no terminal status
     * @throws UnsupportedOperationException If the server predates the
     *         {@code warm} action
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
        WarmProgress[] done = { null };
        streamAction("warm", sourceId.getBytes(StandardCharsets.UTF_8),
                WarmStreamMessage.parser(),
                message -> {
                    if (shouldCancel != null && shouldCancel.getAsBoolean()) {
                        throw new TensorOperationCancelledException("warm", sourceId);
                    }
                    if (message.getPayloadCase() == WarmStreamMessage.PayloadCase.PROGRESS) {
                        if (onProgress != null) {
                            onProgress.accept(message.getProgress());
                        }
                    } else if (message.getPayloadCase() == WarmStreamMessage.PayloadCase.DONE) {
                        done[0] = message.getDone();
                    }
                    return true;
                },
                "Hydrate-ahead is unavailable");
        if (done[0] == null) {
            throw new IOException("warm('" + sourceId
                    + "') returned no terminal status (server closed the stream without a 'done')");
        }
        return done[0];
    }

    // ---- source lifecycle -------------------------------------------------

    /**
     * Register a local path on the SERVER as a served source at runtime.
     *
     * <p>Hands the server a filesystem path (or directory) that it interprets
     * on its <i>own</i> filesystem, and the server routes it through the same
     * claim -&gt; adapter -&gt; catalog pipeline the directory watcher uses. A
     * directory that is not itself a dataset is walked recursively and may
     * register several sources, so this reports a tally rather than one source.
     *
     * @param url Absolute path (or directory) on the server's filesystem
     * @return the terminal {@link AddSourceResult}
     * @throws IOException If the action fails, the server is too old to support
     *         the {@code add_source} action, or it returns no terminal result
     */
    public AddSourceResult addSource(String url) throws IOException {
        return addSource(url, "", null, null);
    }

    /**
     * Register a path on the server, with progress and cancellation hooks.
     *
     * <p>Because a dropped directory's walk has no known size up front, there
     * is no percentage -- progress is a running count of sources registered so
     * far.
     *
     * <p>Cancelling cancels the RPC, which the server observes and stops
     * discovery on; sources already registered stay registered, and this
     * returns an empty tally rather than raising -- the cancel was
     * intentional.
     *
     * @param url Absolute path (or directory) on the server's filesystem
     * @param sourceType Explicit adapter type ({@code "zarr"},
     *        {@code "ome-zarr"}, ...); empty means auto-detect via the
     *        adapters' claim protocol
     * @param onProgress receives one {@link AddSourceProgress} per source as it
     *        registers, or null
     * @param shouldCancel polled once per action message, or null
     * @return the terminal {@link AddSourceResult}: {@code added} /
     *         {@code alreadyPresent} / {@code refreshed} / {@code removed}
     *         source_ids and {@code failed} (path, reason) pairs. Re-adding a
     *         registered path REBUILDS it against the file as it is now -- that
     *         is what {@code refreshed} reports, and it is how a source picks up
     *         an in-place edit. Registration wrote each source's catalog row, so
     *         anything beyond the ids is one {@link #querySources} away.
     */
    public AddSourceResult addSource(
            String url,
            String sourceType,
            Consumer<AddSourceProgress> onProgress,
            BooleanSupplier shouldCancel) throws IOException {
        AddSourceRequest request = AddSourceRequest.newBuilder()
                .setUrl(url)
                .setSourceType(sourceType == null ? "" : sourceType)
                .build();
        AddSourceResult[] result = { null };
        streamAction("add_source", request.toByteArray(), AddSourceStreamMessage.parser(),
                message -> {
                    if (message.getPayloadCase() == AddSourceStreamMessage.PayloadCase.PROGRESS) {
                        if (onProgress != null) {
                            onProgress.accept(message.getProgress());
                        }
                    } else if (message.getPayloadCase() == AddSourceStreamMessage.PayloadCase.RESULT) {
                        result[0] = message.getResult();
                    }
                    // Poll AFTER consuming this message, not before: a cancel
                    // landing exactly on the terminal result must not discard a
                    // completed tally already captured above.
                    return shouldCancel == null || !shouldCancel.getAsBoolean();
                },
                "Runtime source registration is unavailable");
        if (result[0] == null) {
            if (shouldCancel != null && shouldCancel.getAsBoolean()) {
                // A caller-driven cancel breaks before the terminal result;
                // report an empty tally rather than an error.
                return AddSourceResult.getDefaultInstance();
            }
            throw new IOException("addSource('" + url
                    + "') returned no terminal result (server closed the stream without a result)");
        }
        return result[0];
    }

    /**
     * Deregister a drag-dropped source branch on the SERVER at runtime.
     *
     * <p>The narrow counterpart to {@link #addSource}: it removes ONLY
     * drag-dropped sources, which the server identifies by the {@code dnd://}
     * origin scheme on their catalog {@code source_url}. Every source at or
     * under {@code rootUrl} goes as a unit; a non-{@code dnd://} root is
     * refused by the server.
     *
     * @param rootUrl the {@code dnd://} branch root to remove
     * @return {@code removed} source_ids, and {@code failed} entries whose
     *         {@code path} carries the source_id
     */
    public RemoveSourceResult removeSource(String rootUrl) throws IOException {
        RemoveSourceRequest request = RemoveSourceRequest.newBuilder()
                .setRootUrl(rootUrl)
                .build();
        byte[] body = doActionOneResult("remove_source", request.toByteArray(),
                "Source removal is unavailable");
        try {
            return RemoveSourceResult.parseFrom(body);
        } catch (InvalidProtocolBufferException error) {
            throw new IOException("remove_source returned no RemoveSourceResult", error);
        }
    }

    // ---- label sets (biopb-tensor-server/docs/label-tensors.md) -----------

    /**
     * The {@code array_id}s of the label sets served under an image.
     *
     * <p>A label set is an ordinary tensor of its image, named
     * {@code <image array_id>/labels/<name>}, so this is a catalog query over
     * the path and nothing more -- {@link #getTensor} / {@link #getDescriptor}
     * read one like any other tensor.
     *
     * @param imageArrayId the image's array_id
     * @return the sets' array_ids, sorted; empty when the image has none
     */
    public List<String> labelSets(String imageArrayId) throws IOException {
        List<String> sets = new ArrayList<>();
        try (VectorSchemaRoot root = querySources(
                "SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t) WHERE starts_with(t.array_id, "
                        + sqlLiteral(imageArrayId + "/labels/") + ") ORDER BY t.array_id")) {
            FieldVector ids = root.getVector("array_id");
            for (int row = 0; row < root.getRowCount(); row++) {
                if (ids != null && !ids.isNull(row)) {
                    sets.add(String.valueOf(ids.getObject(row)));
                }
            }
        }
        return sets;
    }

    /**
     * Delete an uploaded label set, and the store behind it.
     *
     * <p><b>Experimental</b>, with the rest of the upload API.
     *
     * <p>Only a <i>finished uploaded</i> set: a set the image's own file
     * carries is the file's, and a server-owned one (a name under {@code @}) is
     * the server's. Deleting frees the name at once -- the next set uploaded
     * under it is a distinct tensor with its own cache namespace, so no stale
     * chunk can be served for it.
     *
     * @param arrayId the set's array_id, as {@link #labelSets} reports it
     * @return {@code {"array_id": ..., "deleted": true}}
     */
    public Map<String, Object> deleteLabels(String arrayId) throws IOException {
        byte[] body = doActionOneResult("delete_labels",
                arrayId.getBytes(StandardCharsets.UTF_8),
                "Label set deletion is unavailable");
        return GSON.fromJson(new String(body, StandardCharsets.UTF_8),
                new TypeToken<Map<String, Object>>() {
                }.getType());
    }

    // ---- ROI annotations (biopb-tensor-server/docs/roi-annotations.md) ----

    /**
     * Fetch a tensor's ROI annotations.
     *
     * <p>There is no plane or bbox filter: a client hit-tests and re-renders
     * from the resident set. Annotations are private data, gated by the
     * tensor's source like its pixels, so they are not on the SQL surface.
     *
     * @param arrayId unversioned array_id of the tensor
     * @return the annotations, a {@code truncated} flag, and {@code sets} --
     *         every set on the tensor with its stored row count, whatever
     *         {@code rois} covers
     */
    public RoiListResult listRois(String arrayId) throws IOException {
        return listRois(arrayId, "");
    }

    /**
     * Fetch one layer of a tensor's ROI annotations.
     *
     * @param arrayId unversioned array_id of the tensor
     * @param setName restrict to one layer, and the only way to read a reserved
     *        ({@code @}) set; empty means the client-owned sets
     * @return the annotations, a {@code truncated} flag, and the tensor's sets
     */
    public RoiListResult listRois(String arrayId, String setName) throws IOException {
        TensorTicket ticket = TensorTicket.newBuilder()
                .setRoiRead(RoiRead.newBuilder()
                        .setArrayId(arrayId)
                        .setSetName(setName == null ? "" : setName)
                        .build())
                .build();
        RoiListResult.Builder result = RoiListResult.newBuilder();
        try (FlightStream stream = session.getStream(new Ticket(ticket.toByteArray()))) {
            // `truncated` and the tensor's `sets` ride the stream's schema
            // metadata, so they are read before the first batch.
            Map<String, String> metadata = stream.getSchema().getCustomMetadata();
            if (metadata != null) {
                result.setTruncated(Boolean.parseBoolean(metadata.get("truncated")));
                result.addAllSets(parseRoiSets(metadata.get("sets")));
            }
            while (stream.next()) {
                result.addAllRois(RoiRowCodec.roisFromRoot(stream.getRoot()));
            }
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        } catch (RuntimeException error) {
            throw error;
        } catch (Exception error) {
            throw new IOException("listRois failed: " + error.getMessage(), error);
        }
        return result.build();
    }

    /** The tensor's sets, as the read stream's {@code sets} metadata reports them. */
    private static List<RoiSetInfo> parseRoiSets(String json) {
        List<RoiSetInfo> sets = new ArrayList<>();
        if (json == null || json.isEmpty()) {
            return sets;
        }
        List<Map<String, Object>> entries = GSON.fromJson(json,
                new TypeToken<List<Map<String, Object>>>() {
                }.getType());
        if (entries == null) {
            return sets;
        }
        for (Map<String, Object> entry : entries) {
            Object count = entry.get("count");
            sets.add(RoiSetInfo.newBuilder()
                    .setSetName(String.valueOf(entry.getOrDefault("set_name", "")))
                    .setCount(count instanceof Number ? ((Number) count).longValue() : 0L)
                    .setReserved(Boolean.TRUE.equals(entry.get("reserved")))
                    .build());
        }
        return sets;
    }

    /**
     * Create or update ROI annotations on a tensor, as one batch.
     *
     * <p>Geometry is {@code biopb.image.ROI} in LEVEL-0 pixel coordinates -- a
     * shape drawn on a downsampled level must be scaled up by the caller. Only
     * the 2-D vector arms are accepted (point / rectangle / ellipse / polygon /
     * polyline); a mask or mesh is refused, because instance segmentation
     * belongs in a label tensor.
     *
     * <p>An annotation with an empty {@code roi_id} is created (the server mints
     * a uuid4); one that names an existing id is updated. The batch is applied
     * in a single transaction, last writer wins.
     *
     * @param arrayId unversioned array_id every annotation belongs to
     * @param rois the annotations to store
     * @return {@code stored} (with server-assigned roi_id / rev / timestamps)
     *         and {@code conflicts}
     */
    public RoiPutResult putRois(String arrayId, List<RoiAnnotation> rois) throws IOException {
        return putRois(arrayId, rois, false);
    }

    /**
     * Create or update ROI annotations, optionally conditional on {@code rev}.
     *
     * @param arrayId unversioned array_id every annotation belongs to
     * @param rois the annotations to store
     * @param checkRev make each write conditional on {@code rev} matching what
     *        is stored; mismatches come back in {@code conflicts} and are not
     *        applied, and the rest of the batch still lands
     * @return {@code stored} and {@code conflicts}
     */
    public RoiPutResult putRois(String arrayId, List<RoiAnnotation> rois, boolean checkRev)
            throws IOException {
        PutCommand command = PutCommand.newBuilder()
                .setRoiPut(RoiPut.newBuilder()
                        .setArrayId(arrayId)
                        .setCheckRev(checkRev)
                        .build())
                .build();
        try (VectorSchemaRoot rows = RoiRowCodec.roisToRoot(
                rois == null ? Collections.emptyList() : rois, allocator)) {
            return RoiPutResult.parseFrom(roiPutStream(command, rows));
        } catch (InvalidProtocolBufferException error) {
            throw new IOException("putRois returned no RoiPutResult", error);
        }
    }

    /**
     * Delete ROI annotations.
     *
     * <p>With {@code roiIds}, deletes exactly those. Without, deletes every
     * annotation on the tensor -- narrowed to {@code setName} when given, which
     * is how a whole layer is dropped.
     *
     * @param arrayId unversioned array_id of the tensor
     * @param roiIds the ids to delete, or empty for all
     * @param setName narrow a delete-all to one layer, or empty
     * @return the ids actually removed
     */
    public RoiDeleteResult deleteRois(String arrayId, List<String> roiIds, String setName)
            throws IOException {
        PutCommand command = PutCommand.newBuilder()
                .setRoiDelete(RoiDelete.newBuilder()
                        .setArrayId(arrayId)
                        .setSetName(setName == null ? "" : setName)
                        .build())
                .build();
        try (VectorSchemaRoot rows = RoiRowCodec.roiIdsToRoot(
                roiIds == null ? Collections.emptyList() : roiIds, allocator)) {
            return RoiDeleteResult.parseFrom(roiPutStream(command, rows));
        } catch (InvalidProtocolBufferException error) {
            throw new IOException("deleteRois returned no RoiDeleteResult", error);
        }
    }

    /**
     * Report, and with {@code apply} delete, annotations whose image is gone.
     *
     * <p>An annotation is unseen when the catalog has not held its source for
     * {@code unseenDays} (a row whose source never appeared counts from its
     * creation). Reserved, server-owned sets are never pruned. Requires the
     * server-wide token: orphans have no source to authorize against.
     *
     * @param unseenDays how long a source must have been absent to count
     * @param apply false reports only; true deletes
     * @return the unseen annotations grouped per tensor, and the row count
     *         deleted (0 on a report)
     */
    public RoiPruneResult pruneRois(int unseenDays, boolean apply) throws IOException {
        RoiPruneRequest request = RoiPruneRequest.newBuilder()
                .setUnseenDays(unseenDays)
                .setApply(apply)
                .build();
        byte[] body = doActionOneResult("roi_prune", request.toByteArray(),
                "ROI pruning is unavailable");
        try {
            return RoiPruneResult.parseFrom(body);
        } catch (InvalidProtocolBufferException error) {
            throw new IOException("roi_prune returned no RoiPruneResult", error);
        }
    }

    /**
     * One DoPut on the {@code roi} flight: the command in the descriptor, the
     * rows in the stream, the structured reply in the put's app_metadata.
     */
    private byte[] roiPutStream(PutCommand command, VectorSchemaRoot rows) throws IOException {
        try (SyncPutListener reply = new SyncPutListener()) {
            FlightClient.ClientStreamListener writer = session.startPut(
                    FlightDescriptor.command(command.toByteArray()), rows, reply);
            if (rows.getRowCount() > 0) {
                writer.putNext();
            }
            writer.completed();
            PutResult ack = reply.read();
            if (ack == null) {
                writer.getResult();
                throw new IOException("the server acknowledged the ROI put with no result");
            }
            try {
                ArrowBuf metadata = ack.getApplicationMetadata();
                byte[] body = new byte[(int) metadata.readableBytes()];
                metadata.getBytes(0, body);
                writer.getResult();
                return body;
            } finally {
                ack.close();
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IOException("ROI put interrupted", error);
        } catch (ExecutionException error) {
            throw FlightSession.mapped(error.getCause());
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    /**
     * Fetch one tensor's {@link TensorDescriptor} by its globally-unique array_id.
     *
     * <p>A tensor is identified by its {@code array_id} alone (see the tensor
     * identity policy at the top of {@code proto/biopb/tensor/descriptor.proto}),
     * so this takes that one identifier. Works even when the source is beyond
     * the server's query row cap. One {@code GetFlightInfo} per call -- nothing
     * is cached, because a descriptor is what the server says now. A bare
     * {@code source_id} (single-tensor source, or to
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
     * is one describe -- the cheap projection, which never requests the opt-in
     * {@code metadata_json} field or the O(chunks) endpoint plan.
     * (Contrast {@link #getSourceMetadata}, which
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
        return getTensor(arrayId, null, scaleHint, reductionMethod);
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
     * @return lazy RandomAccessibleInterval containing the requested tensor
     */
    public <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> getTensor(
            String arrayId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

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
     * Get a SerializedTensor protobuf for a whole tensor.
     *
     * @param arrayId Globally-unique tensor id ({@code source_id} or
     *                {@code source_id/field})
     * @return SerializedTensor protobuf object
     */
    public SerializedTensor getTensorAsPb(String arrayId) {
        return getTensorAsPb(arrayId, null, null, null);
    }

    /**
     * Get a SerializedTensor protobuf for cross-process transfer.
     *
     * Returns a protobuf containing connection info and chunk tickets
     * for lazy reconstruction. The protobuf can be serialized to bytes
     * and broadcast to worker processes (e.g., Spark), where each worker
     * can call tensorFromPb() to reconstruct a lazy imglib2 array.
     *
     * @param arrayId         Globally-unique tensor id ({@code source_id} or
     *                        {@code source_id/field})
     * @param sliceHint       Optional slice hint
     * @param scaleHint       Per-dimension scale factors
     * @param reductionMethod Requested reduction method
     * @return SerializedTensor protobuf object
     */
    public SerializedTensor getTensorAsPb(
            String arrayId,
            SliceHint sliceHint,
            long[] scaleHint,
            String reductionMethod) {

        LOGGER.fine("getTensorAsPb: arrayId=" + arrayId);
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
        return TensorChunkCodec.descriptorOf(flightInfoOf(pb));
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
        Map<String, Object> unknown = new HashMap<>();
        unknown.put("status", "UNKNOWN");
        try {
            return session.health().orElse(unknown);
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
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
            desc = TensorChunkCodec.descriptorOf(info);
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
        return TensorUploads.statusMap(sourceId, desc.getUploadStatus());
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

    // ====================
    // Upload API (EXPERIMENTAL) -- thin delegators onto the TensorUploads
    // collaborator. Status polling is a read of one descriptor field and stays
    // above, with the other catalog reads.
    // ====================

    /**
     * Declare a single-tensor source to fill: the first half of an upload.
     *
     * <p><b>Experimental.</b> The upload / writable-source API (tensor
     * creation, chunk upload, and upload-status polling) may change.
     *
     * <p>Declare, then fill. The returned descriptor is the server's echo --
     * array_id, shape, dtype, chunk_shape, dim_labels -- and is what
     * {@link #uploadArray}, {@link #uploadChunk} and {@link #finishUpload}
     * take. A name is taken while its source exists: a second create under it
     * -- pending, finished or discarded -- is refused. Only the server's
     * reclaim sweep frees one, after a discarded upload's {@code upload_ttl}.
     * {@link #finishUpload} is what marks the upload complete.
     *
     * @param sourceName {@code "cache:name"} for cache-backed,
     *        {@code "ome_zarr:name"} for zarr-backed, either prefix with an
     *        empty name for a server-generated one, or
     *        {@code "<image array_id>/labels/<name>"} for a label set of an
     *        image the server already serves -- the one form whose id is the
     *        request's own rather than a minted source_id. A set is
     *        unsigned-integer, spans its image's non-channel axes at full
     *        length, and its all-zero chunks are skipped by
     *        {@link #uploadArray}
     * @param shape the tensor's shape
     * @param dtype the numpy dtype string to store it as (e.g. {@code "<u2"})
     * @param chunkShape the upload grid; null or empty means one chunk
     * @param dimLabels optional dimension labels
     * @param omeMetadataJson optional OME metadata, as a JSON object
     * @return the new source's descriptor
     */
    public TensorDescriptor createTensor(
            String sourceName,
            long[] shape,
            String dtype,
            long[] chunkShape,
            List<String> dimLabels,
            String omeMetadataJson) {
        return uploads.createTensor(sourceName, shape, dtype, chunkShape, dimLabels, omeMetadataJson);
    }

    /**
     * Declare a source shaped like an array you already hold.
     *
     * <p><b>Experimental</b>, with the rest of the upload API. The template's
     * shape and pixel type stand in for the explicit {@code shape}/{@code dtype}
     * of {@link #createTensor(String, long[], String, long[], List, String)};
     * it is the array about to be uploaded, or one shaped like it.
     *
     * @param sourceName as in
     *        {@link #createTensor(String, long[], String, long[], List, String)}
     * @param template the array to be uploaded, or one shaped like it
     * @param chunkShape the upload grid; null or empty means one chunk
     * @param dimLabels optional dimension labels
     * @param omeMetadataJson optional OME metadata, as a JSON object
     * @param <T> the pixel type
     * @return the new source's descriptor
     */
    public <T extends NativeType<T> & RealType<T>> TensorDescriptor createTensor(
            String sourceName,
            RandomAccessibleInterval<T> template,
            long[] chunkShape,
            List<String> dimLabels,
            String omeMetadataJson) {
        long[] shape = new long[template.numDimensions()];
        template.dimensions(shape);
        return uploads.createTensor(sourceName, shape, TensorUploads.numpyDtype(template.getType()),
                chunkShape, dimLabels, omeMetadataJson);
    }

    /**
     * Fill a declared tensor with an array, and seal it.
     *
     * <p><b>Experimental</b>, with the rest of the upload API.
     *
     * <p>{@code array} must match the descriptor's shape; it is walked on the
     * descriptor's chunk grid, every block is sent as one chunk, and the source
     * is finished. An all-zero block of a <i>label set</i> is not sent at all:
     * the store's fill value already reads as background, so one labelled frame
     * of a thousand costs one frame (biopb/biopb#1059).
     *
     * @param descriptor the descriptor {@link #createTensor} returned
     * @param array the array to upload
     * @param <T> the pixel type
     * @return the sealed upload status, as {@link #getUploadStatus} reports it
     * @throws IllegalArgumentException if {@code array} does not match the
     *         declared shape
     * @throws UploadRefusedException if the upload is over -- sealed or
     *         discarded
     */
    public <T extends NativeType<T> & RealType<T>> Map<String, Object> uploadArray(
            TensorDescriptor descriptor, RandomAccessibleInterval<T> array) {
        return uploads.uploadArray(descriptor, array);
    }

    /**
     * Upload one chunk of a declared tensor.
     *
     * <p><b>Experimental</b>, with the rest of the upload API.
     *
     * <p>The manual half of {@link #uploadArray}: a caller writing chunks
     * itself calls this per chunk and {@link #finishUpload} when done. The
     * chunk's elements are read out of {@code source} at {@code bounds} -- in
     * that interval's own global coordinates, so the whole array can be passed
     * for every chunk.
     *
     * @param descriptor the descriptor {@link #createTensor} returned
     * @param bounds chunk start/stop coordinates
     * @param source the array to read the chunk out of
     * @param <T> the pixel type
     * @throws UploadRefusedException if the upload is over -- sealed or
     *         discarded
     */
    public <T extends NativeType<T> & RealType<T>> void uploadChunk(
            TensorDescriptor descriptor, ChunkBounds bounds, RandomAccessibleInterval<T> source) {
        uploads.uploadChunk(descriptor, bounds, source);
    }

    /**
     * Seal an upload: the source is complete and takes no further chunks.
     *
     * <p><b>Experimental</b>, with the rest of the upload API.
     *
     * <p>The only route to READY, which is the state a consumer waiting on this
     * result polls for. {@link #uploadArray}, which writes every chunk itself,
     * calls it for you.
     *
     * @param descriptor the descriptor {@link #createTensor} returned
     * @return the sealed upload status, as {@link #getUploadStatus} reports it
     * @throws UploadRefusedException if the upload was discarded
     */
    public Map<String, Object> finishUpload(TensorDescriptor descriptor) {
        return uploads.finishUpload(descriptor);
    }

    /**
     * Consume a streaming action, handing each non-empty message to
     * {@code onMessage}; returning false from it stops consuming.
     *
     * <p>The loop shared by {@link #resolve} / {@link #warm} /
     * {@link #addSource}: the {@code doAction} call, the empty-body heartbeat
     * skip, the envelope parse, and the old-server {@code "Unknown action"}
     * remap, applied only when {@code unavailableHint} is given.
     *
     * <p><b>Stopping cancels the RPC.</b> The call is created inside a
     * {@link Context.CancellableContext}, which gRPC ties to it, so cancelling
     * the scope cancels the call and the server observes it -- the same thing
     * Python gets by closing its generator, and what lets a server stop a walk
     * it is halfway through rather than finish it for nobody. It happens on
     * every exit, so a caller that throws out of {@code onMessage}
     * (resolve/warm raise on cancel) also releases the server.
     *
     * <p>A message that does not parse as {@code M} is skipped. Python can call
     * that harmless because its SDK refuses a pre-v2 server at connect; this
     * client has no such handshake yet, so against a v1 server the skip is what
     * turns a protocol mismatch into "returned no terminal result". Reported at
     * the caller, which is the best this can do until the health-action
     * {@code protocol} check is ported.
     *
     * <p>Cancellation policy is deliberately NOT decided here: its semantics
     * differ per caller (resolve/warm raise, addSource returns what it has),
     * and the poll must run relative to a consumed message, which only the
     * caller knows the right side of.
     */
    private <M extends com.google.protobuf.Message> void streamAction(
            String type,
            byte[] body,
            com.google.protobuf.Parser<M> parser,
            java.util.function.Predicate<M> onMessage,
            String unavailableHint) throws IOException {
        Action action = new Action(type, body);
        Context.CancellableContext scope = Context.current().withCancellation();
        try {
            java.util.Iterator<Result> results;
            // The call must be created under the scope for gRPC to bind the two;
            // attaching only for that instant keeps it off the caller's thread.
            Context previous = scope.attach();
            try {
                results = session.doAction(action);
            } finally {
                scope.detach(previous);
            }

            while (results.hasNext()) {
                byte[] message = results.next().getBody();
                if (message == null || message.length == 0) {
                    continue; // legacy empty-body heartbeat (a server predating progress)
                }
                M parsed;
                try {
                    parsed = parser.parseFrom(message);
                } catch (InvalidProtocolBufferException ignored) {
                    continue;
                }
                if (!onMessage.test(parsed)) {
                    return;
                }
            }
        } catch (FlightRuntimeException error) {
            throw tooOld(error, type, unavailableHint);
        } finally {
            // Cancelling a stream already drained is a no-op, so this needs no
            // flag for "did we stop early".
            scope.cancel(null);
        }
    }

    /**
     * Run a single-result {@code doAction}, with the same "old server" remap
     * {@link #streamAction} gives the streaming actions.
     */
    private byte[] doActionOneResult(String type, byte[] body, String unavailableHint)
            throws IOException {
        Result result;
        try {
            java.util.Iterator<Result> results = session.doAction(new Action(type, body));
            result = results.hasNext() ? results.next() : null;
        } catch (FlightRuntimeException error) {
            throw tooOld(error, type, unavailableHint);
        }
        if (result == null) {
            throw new IOException(type + " returned no result");
        }
        byte[] answer = result.getBody();
        return answer == null ? new byte[0] : answer;
    }

    /**
     * A feature the server predates, named as such; anything else unchanged.
     *
     * <p>{@code unavailableHint} is the feature-specific lead-in (e.g. "Source
     * removal is unavailable"); without one the Flight error passes through, so
     * an action every server has cannot be misreported as missing.
     */
    private static RuntimeException tooOld(
            FlightRuntimeException error, String type, String unavailableHint) {
        if (unavailableHint != null && String.valueOf(error.getMessage()).contains("Unknown action")) {
            return new UnsupportedOperationException(unavailableHint
                    + ": the tensor server is too old to support the '" + type
                    + "' action. Upgrade the server.", error);
        }
        return error;
    }

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
        TensorDescriptor descriptor = TensorChunkCodec.descriptorOf(info);
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
        return TensorChunkCodec.descriptorOf(info);
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
        // Advisory only, and NOT a compatibility gate: tensor_schema_version is
        // the server package's own release tag, which says nothing about the
        // wire. The two real gates are the health action's `protocol`
        // (FlightSession) and the schema's `chunk_wire_protocol`
        // (Imglib2TensorFactory). A malformed version string must never fail a read.
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
            this.descriptor = descriptor;
            this.info = info;
        }
    }
}
