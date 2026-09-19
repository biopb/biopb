package biopb.tensor;

import java.util.Arrays;

import org.apache.arrow.flight.FlightEndpoint;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.vector.FieldVector;

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

import static biopb.tensor.TensorChunkCodec.createType;
import static biopb.tensor.TensorChunkCodec.estimateChunkBytes;
import static biopb.tensor.TensorChunkCodec.parseChunkBounds;
import static biopb.tensor.TensorChunkCodec.parseTicket;
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
        TensorDescriptor descriptor = descriptorOf(plan);
        T type = (T) createType(descriptor.getDtype());
        long[] dims = toLongArray(descriptor.getShapeList());
        int[] cellDimensions = toIntArray(descriptor.getChunkShapeList());

        ChunkGridIndex<FlightEndpoint> endpointIndex = ChunkGridIndex.build(
                plan.getEndpoints(), dims, cellDimensions,
                endpoint -> parseChunkBounds(endpoint.getAppMetadata()),
                endpoint -> endpoint);
        if (endpointIndex == null) {
            return materialize(plan, type, dims);
        }

        long estimatedChunkBytes = estimateChunkBytes(descriptor);
        long maxCells = Math.max(1L, cacheBytes / Math.max(estimatedChunkBytes, 1L));
        ReadOnlyCachedCellImgOptions options = ReadOnlyCachedCellImgOptions.options()
                .cellDimensions(cellDimensions)
                .cacheType(CacheType.BOUNDED)
                .maxCacheSize(maxCells);

        ReadOnlyCachedCellImgFactory factory = new ReadOnlyCachedCellImgFactory(options);
        return (RandomAccessibleInterval<T>) factory.create(dims, type,
                cell -> loadCell(cell, endpointIndex));
    }

    private static TensorDescriptor descriptorOf(FlightInfo plan) {
        try {
            return TensorDescriptor.parseFrom(plan.getDescriptor().getCommand());
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("FlightInfo descriptor is not a TensorDescriptor", error);
        }
    }

    @SuppressWarnings("unchecked")
    private <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> materialize(
            FlightInfo plan, T type, long[] dims) {
        ArrayImg<T, ?> image = (ArrayImg<T, ?>) new ArrayImgFactory<>(type).create(dims);
        RandomAccess<T> access = image.randomAccess();

        for (FlightEndpoint endpoint : plan.getEndpoints()) {
            TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
            ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
            writeChunk(access, bounds, fetchChunkValues(ticket.getChunkId().toByteArray()));
        }
        return image;
    }

    private <T extends NativeType<T> & RealType<T>> void loadCell(
            SingleCellArrayImg<T, ?> cell, ChunkGridIndex<FlightEndpoint> endpointIndex) {
        long cellIndex = endpointIndex.indexFor(cell);
        FlightEndpoint endpoint = endpointIndex.get(cellIndex);
        if (endpoint == null) {
            throw new IllegalStateException("No Flight endpoint found for cell index " + cellIndex);
        }

        TensorTicket ticket = parseTicket(endpoint.getTicket().getBytes());
        ChunkBounds bounds = parseChunkBounds(endpoint.getAppMetadata());
        writeChunk(cell.randomAccess(), bounds, fetchChunkValues(ticket.getChunkId().toByteArray()));
    }

    private double[] fetchChunkValues(byte[] chunkId) {
        TensorTicket tensorTicket = TensorTicket.newBuilder()
                .setChunkId(ByteString.copyFrom(chunkId))
                .build();

        try (FlightStream stream = session.getStream(new Ticket(tensorTicket.toByteArray()))) {
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
        } catch (Exception error) {
            throw new IllegalStateException("Failed to fetch chunk payload", error);
        }
    }
}
