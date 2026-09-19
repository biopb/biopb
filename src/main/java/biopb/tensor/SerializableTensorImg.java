package biopb.tensor;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightDescriptor;

import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.FieldMask;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Compatibility adapter for Java serialization of a lazy tensor image.
 *
 * <p>The sole cross-process handle is {@link SerializedTensor}. This class is
 * only an imglib2 adapter: its externalized state is exactly a
 * {@code SerializedTensor} protobuf and a local cache budget. It never
 * serializes source IDs, read options, descriptors, or a second bespoke ticket
 * format. New cross-process APIs should pass {@link SerializedTensor} directly.
 */
@Deprecated
public class SerializableTensorImg<T extends NativeType<T> & RealType<T>>
        implements RandomAccessibleInterval<T>, Externalizable, AutoCloseable {

    private byte[] serializedTensorBytes;
    private long cacheBytes;

    private transient RandomAccessibleInterval<T> delegate;
    private transient FlightSession session;

    /** Required for {@link Externalizable}. */
    public SerializableTensorImg() {
        this.cacheBytes = 100_000_000L;
    }

    /** Wrap a Flight v2 handle and an optional already-created local image. */
    public SerializableTensorImg(
            SerializedTensor serializedTensor,
            long cacheBytes,
            RandomAccessibleInterval<T> delegate) {
        if (serializedTensor == null) {
            throw new IllegalArgumentException("SerializedTensor is required");
        }
        this.serializedTensorBytes = serializedTensor.toByteArray();
        this.cacheBytes = cacheBytes;
        this.delegate = delegate;
    }

    /** Return the immutable Flight v2 handle this adapter carries. */
    public SerializedTensor getSerializedTensor() {
        if (serializedTensorBytes == null) {
            throw new IllegalStateException("SerializedTensor is missing");
        }
        try {
            return SerializedTensor.parseFrom(serializedTensorBytes);
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalStateException("SerializedTensor payload is invalid", error);
        }
    }

    @Override
    public void writeExternal(ObjectOutput output) throws IOException {
        if (serializedTensorBytes == null) {
            throw new IOException("SerializedTensor is missing");
        }
        output.writeLong(cacheBytes);
        output.writeInt(serializedTensorBytes.length);
        output.write(serializedTensorBytes);
    }

    @Override
    public void readExternal(ObjectInput input) throws IOException {
        cacheBytes = input.readLong();
        int length = input.readInt();
        if (length < 0 || length > 64 * 1024 * 1024) {
            throw new IOException("Invalid SerializedTensor payload length: " + length);
        }
        serializedTensorBytes = new byte[length];
        input.readFully(serializedTensorBytes);
        delegate = null;
        session = null;
    }

    /**
     * Build the delegate on first access.
     *
     * <p>Synchronized because reconstruction opens a {@link FlightSession}, and
     * every accessor below funnels through here: two threads reading the same
     * image -- the ordinary case once a cell cache is loading chunks in
     * parallel -- would otherwise each open one, and the loser's is orphaned
     * with no reference left to close it.
     */
    private synchronized void ensureDelegate() {
        if (delegate == null) {
            delegate = reconstructDelegate();
        }
    }

    @SuppressWarnings("unchecked")
    private RandomAccessibleInterval<T> reconstructDelegate() {
        SerializedTensor handle = getSerializedTensor();
        if (handle.getLocation().isEmpty()) {
            throw new IllegalArgumentException("SerializedTensor.location is required");
        }

        FlightInfo plan = TensorFlightClient.flightInfoOf(handle);
        session = new FlightSession(
                LocationUris.parse(handle.getLocation()),
                handle.getAuthToken().isEmpty() ? null : handle.getAuthToken());
        if (plan.getEndpoints().isEmpty()) {
            plan = refreshEndpointlessPlan(plan);
        }

        RandomAccessibleInterval<T> image = new Imglib2TensorFactory(session, cacheBytes).create(plan);
        SliceHint requested = requestedSlice(plan);
        TensorDescriptor descriptor = descriptorOf(plan);
        if (requested != null && descriptor.hasSliceHint()) {
            image = RegionCrop.cropToRequest(image, requested, descriptor.getSliceHint(),
                    descriptor.getScaleHintList());
        }
        return image;
    }

    private FlightInfo refreshEndpointlessPlan(FlightInfo plan) {
        TensorDescriptor descriptor = descriptorOf(plan);
        TensorReadOption.Builder read = TensorReadOption.newBuilder()
                .setArrayId(descriptor.getArrayId())
                .setFields(FieldMask.newBuilder().addPaths("endpoints").build());
        if (descriptor.hasSliceHint()) {
            read.setSliceHint(descriptor.getSliceHint());
        }
        read.addAllScaleHint(descriptor.getScaleHintList());
        if (!descriptor.getReductionMethod().isEmpty()) {
            read.setReductionMethod(descriptor.getReductionMethod());
        }
        FlightRequest request = FlightRequest.newBuilder().setTensorRead(read.build()).build();
        return session.getInfo(FlightDescriptor.command(request.toByteArray()));
    }

    private static TensorDescriptor descriptorOf(FlightInfo plan) {
        try {
            return TensorDescriptor.parseFrom(plan.getDescriptor().getCommand());
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("FlightInfo descriptor is not a TensorDescriptor", error);
        }
    }

    private static SliceHint requestedSlice(FlightInfo plan) {
        byte[] metadata = plan.getAppMetadata();
        if (metadata == null || metadata.length == 0) {
            return null;
        }
        try {
            return SliceHint.parseFrom(metadata);
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("FlightInfo.app_metadata is not a SliceHint", error);
        }
    }

    @Override
    public synchronized void close() {
        if (session != null) {
            session.close();
            session = null;
        }
    }

    @Override public long min(int d) { ensureDelegate(); return delegate.min(d); }
    @Override public long max(int d) { ensureDelegate(); return delegate.max(d); }
    @Override public int numDimensions() { ensureDelegate(); return delegate.numDimensions(); }
    @Override public long size() { ensureDelegate(); return delegate.size(); }
    @Override public RandomAccess<T> randomAccess() { ensureDelegate(); return delegate.randomAccess(); }
    @Override public RandomAccess<T> randomAccess(Interval interval) {
        ensureDelegate(); return delegate.randomAccess(interval);
    }
    @Override public Cursor<T> cursor() { ensureDelegate(); return delegate.cursor(); }
    @Override public Cursor<T> localizingCursor() { ensureDelegate(); return delegate.localizingCursor(); }
    @Override public Object iterationOrder() { ensureDelegate(); return delegate.iterationOrder(); }
    @Override public long dimension(int d) { ensureDelegate(); return delegate.dimension(d); }
    @Override public void min(long[] min) { ensureDelegate(); delegate.min(min); }
    @Override public void max(long[] max) { ensureDelegate(); delegate.max(max); }
    @Override public void dimensions(long[] dimensions) { ensureDelegate(); delegate.dimensions(dimensions); }
}
