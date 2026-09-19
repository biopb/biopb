package biopb.tensor;

import java.util.Iterator;

import org.apache.arrow.flight.Action;
import org.apache.arrow.flight.FlightClient;
import org.apache.arrow.flight.FlightDescriptor;
import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStream;
import org.apache.arrow.flight.Location;
import org.apache.arrow.flight.Result;
import org.apache.arrow.flight.Ticket;
import org.apache.arrow.flight.grpc.CredentialCallOption;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

/**
 * Owns one Flight connection's allocator, authentication and error boundary.
 *
 * <p>Every foreground Flight call enters here, so typed server errors are
 * translated consistently both when an RPC starts and when a streaming action
 * reports an error while its iterator is consumed. TLS construction joins this
 * class in the next migration slice; its ownership boundary is established now.
 */
public final class FlightSession implements AutoCloseable {
    private final BufferAllocator allocator;
    private final FlightClient client;
    private final CredentialCallOption authOption;
    private final Location location;
    private final String token;

    public FlightSession(Location location, String token) {
        this.location = location;
        this.token = token;
        this.allocator = new RootAllocator(Long.MAX_VALUE);
        this.client = FlightClient.builder(allocator, location).build();
        this.authOption = token == null || token.isEmpty()
                ? null
                : new CredentialCallOption(headers -> headers.insert("authorization", "Bearer " + token));
    }

    public Location location() { return location; }
    public String token() { return token; }
    public BufferAllocator allocator() { return allocator; }
    public FlightClient client() { return client; }
    public CredentialCallOption authOption() { return authOption; }

    public FlightInfo getInfo(FlightDescriptor descriptor) {
        try {
            return client.getInfo(descriptor, authOption);
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    public FlightStream getStream(Ticket ticket) {
        try {
            return client.getStream(ticket, authOption);
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    public Iterator<Result> doAction(Action action) {
        try {
            return new ErrorMappingIterator(client.doAction(action, authOption));
        } catch (FlightRuntimeException error) {
            throw TensorErrorMapper.map(error);
        }
    }

    @Override
    public void close() {
        try {
            client.close();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        } finally {
            allocator.close();
        }
    }

    static final class ErrorMappingIterator implements Iterator<Result> {
        private final Iterator<Result> delegate;

        ErrorMappingIterator(Iterator<Result> delegate) { this.delegate = delegate; }

        @Override
        public boolean hasNext() {
            try {
                return delegate.hasNext();
            } catch (FlightRuntimeException error) {
                throw TensorErrorMapper.map(error);
            }
        }

        @Override
        public Result next() {
            try {
                return delegate.next();
            } catch (FlightRuntimeException error) {
                throw TensorErrorMapper.map(error);
            }
        }
    }
}
