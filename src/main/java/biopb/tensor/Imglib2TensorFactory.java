package biopb.tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.vector.FieldVector;

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

import static biopb.tensor.TensorChunkCodec.createType;
import static biopb.tensor.TensorChunkCodec.descriptorOf;
import static biopb.tensor.TensorChunkCodec.estimateChunkBytes;
import static biopb.tensor.TensorChunkCodec.parseChunkBounds;
import static biopb.tensor.TensorChunkCodec.toIntArray;
import static biopb.tensor.TensorChunkCodec.toLongArray;
import static biopb.tensor.TensorChunkCodec.writeChunk;

/**
 * Builds lazy imglib2 arrays directly from an immutable Flight read plan.
 *
 * <p>The server's {@link FlightInfo} is the complete read plan: its descriptor
 * provides the realized shape and chunk layout, while each endpoint provides
 * one chunk ticket and bounds. This adapter deliberately does not issue
 * {@code GetFlightInfo}; callers must plan first and hand it the response they
 * intend to read. A clean chunk grid becomes a bounded cached cell image;
 * uncommon layouts fall back to a materialized {@link ArrayImg}.
 */
final class Imglib2TensorFactory {
    private final FlightSession session;
    private final long cacheBytes;

    Imglib2TensorFactory(FlightSession session, long cacheBytes) {
        this.session = session;
        this.cacheBytes = cacheBytes;
    }

    @SuppressWarnings("unchecked")
    <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> create(FlightInfo plan) {
        checkWireProtocol(plan);
        TensorDescriptor descriptor = descriptorOf(plan);
        T type = (T) createType(descriptor.getDtype());
        long[] dims = toLongArray(descriptor.getShapeList());
        int[] cellDimensions = toIntArray(descriptor.getChunkShapeList());

        List<ChunkRef> chunks = chunkRefs(plan);
        ChunkGridIndex<ChunkRef> chunkIndex = ChunkGridIndex.build(
                chunks, dims, cellDimensions, chunk -> chunk.bounds, chunk -> chunk);
        if (chunkIndex == null) {
            return materialize(chunks, type, dims);
        }

        long estimatedChunkBytes = estimateChunkBytes(descriptor);
        long maxCells = Math.max(1L, cacheBytes / Math.max(estimatedChunkBytes, 1L));
        ReadOnlyCachedCellImgOptions options = ReadOnlyCachedCellImgOptions.options()
                .cellDimensions(cellDimensions)
                .cacheType(CacheType.BOUNDED)
                .maxCacheSize(maxCells);

        ReadOnlyCachedCellImgFactory factory = new ReadOnlyCachedCellImgFactory(options);
        return (RandomAccessibleInterval<T>) factory.create(dims, type,
                cell -> loadCell(cell, chunkIndex));
    }

    /**
     * Refuse a plan whose chunk encoding this client cannot read.
     *
     * <p>Here rather than at {@code GetFlightInfo} because this is the one
     * place every plan passes through -- one the client planned itself, and one
     * that arrived serialized in a {@link SerializedTensor} from another
     * process -- and it is the last point before the bytes are reinterpreted.
     * The same reason Python checks in {@code _dask_from_flight_info}.
     *
     * <p>An unstamped schema is a pre-#293 server: the key postdates v1, so its
     * absence names the version rather than leaving it unknown. Refusing here
     * costs an actionable message instead of "Data column value is not binary"
     * from inside a cell load.
     */
    private static void checkWireProtocol(FlightInfo plan) {
        String stamped = null;
        java.util.Optional<org.apache.arrow.vector.types.pojo.Schema> schema = plan.getSchemaOptional();
        if (schema.isPresent()) {
            java.util.Map<String, String> metadata = schema.get().getCustomMetadata();
            if (metadata != null) {
                stamped = metadata.get(WireVersions.WIRE_PROTOCOL_METADATA_KEY);
            }
        }
        int serverVersion = WireVersions.stampedVersion(stamped);
        if (serverVersion != WireVersions.TENSOR_WIRE_PROTOCOL_VERSION) {
            throw new UnsupportedOperationException(WireVersions.mismatch(
                    "tensor wire protocol", serverVersion, WireVersions.TENSOR_WIRE_PROTOCOL_VERSION,
                    "The chunk encoding is a breaking contract (biopb/biopb#293)."));
        }
    }

    /**
     * One chunk of a plan: the ticket to fetch it with and where it lands.
     *
     * <p>Resolved once per plan rather than per cache miss. The endpoint's
     * ticket already carries the {@link TensorTicket} this client sends back
     * unchanged, so parsing it only to rebuild an identical one was work for
     * nothing; holding the bounds here also lets the whole
     * {@link FlightEndpoint} -- its location list and expiration -- be dropped
     * once the index is built, which on a large tensor is the bulk of what the
     * plan retains for the image's lifetime.
     */
    private static final class ChunkRef {
        final Ticket ticket;
        final ChunkBounds bounds;

        ChunkRef(Ticket ticket, ChunkBounds bounds) {
            this.ticket = ticket;
            this.bounds = bounds;
        }
    }

    private static List<ChunkRef> chunkRefs(FlightInfo plan) {
        List<ChunkRef> chunks = new ArrayList<>(plan.getEndpoints().size());
        for (FlightEndpoint endpoint : plan.getEndpoints()) {
            chunks.add(new ChunkRef(endpoint.getTicket(), parseChunkBounds(endpoint.getAppMetadata())));
        }
        return chunks;
    }

    @SuppressWarnings("unchecked")
    private <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> materialize(
            List<ChunkRef> chunks, T type, long[] dims) {
        ArrayImg<T, ?> image = (ArrayImg<T, ?>) new ArrayImgFactory<>(type).create(dims);
        RandomAccess<T> access = image.randomAccess();

        for (ChunkRef chunk : chunks) {
            writeChunk(access, chunk.bounds, fetchChunkValues(chunk.ticket));
        }
        return image;
    }

    private <T extends NativeType<T> & RealType<T>> void loadCell(
            SingleCellArrayImg<T, ?> cell, ChunkGridIndex<ChunkRef> chunkIndex) {
        long cellIndex = chunkIndex.indexFor(cell);
        ChunkRef chunk = chunkIndex.get(cellIndex);
        if (chunk == null) {
            throw new IllegalStateException("No Flight endpoint found for cell index " + cellIndex);
        }
        writeChunk(cell.randomAccess(), chunk.bounds, fetchChunkValues(chunk.ticket));
    }

    private double[] fetchChunkValues(Ticket ticket) {
        try (FlightStream stream = session.getStream(ticket)) {
            double[] values = new double[0];
            while (stream.next()) {
                FieldVector dataVector = stream.getRoot().getVector("data");
                FieldVector dtypeVector = stream.getRoot().getVector("dtype");
                if (dataVector == null || dtypeVector == null) {
                    throw new IllegalStateException("Chunk payload missing 'data'/'dtype' column");
                }

                for (int row = 0; row < stream.getRoot().getRowCount(); row++) {
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
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        } catch (RuntimeException error) {
            // The session already decoded this one (a stale read plan, an
            // unresolved source); rewrapping it as an IllegalStateException
            // would undo exactly the decode this path exists to preserve.
            throw error;
        } catch (Exception error) {
            throw new IllegalStateException("Failed to fetch chunk payload", error);
        }
    }
}
