package biopb.tensor;

import org.apache.arrow.flight.Location;
import org.junit.Assert;
import org.junit.Test;

/**
 * Unit tests for {@link LocationUris#parse}, the single location-string parser
 * that replaced four forked copies (biopb/biopb#277 item D). These pin the
 * tolerant contract -- notably that schemes the old {@code TensorFlightClient}
 * strict parser rejected (a scheme-less authority, {@code grpc+tls://}) now
 * parse instead of throwing.
 */
public class LocationUrisTest {

    @Test
    public void grpc_uri_preserves_host_and_port() {
        Location loc = LocationUris.parse("grpc://localhost:8815");
        Assert.assertNotNull(loc);
        Assert.assertTrue(loc.getUri().toString().contains("localhost"));
        Assert.assertTrue(loc.getUri().toString().contains("8815"));
    }

    @Test
    public void grpc_tcp_uri_is_accepted() {
        Location loc = LocationUris.parse("grpc+tcp://localhost:9000");
        Assert.assertNotNull(loc);
        Assert.assertTrue(loc.getUri().toString().contains("9000"));
    }

    @Test
    public void ip_address_host_is_preserved() {
        Location loc = LocationUris.parse("grpc://127.0.0.1:8815");
        Assert.assertNotNull(loc);
        Assert.assertTrue(loc.getUri().toString().contains("127.0.0.1"));
    }

    @Test
    public void scheme_less_authority_defaults_to_insecure_grpc() {
        // The old strict grpc-only parser threw on this; the tolerant contract
        // treats a bare host:port as an insecure gRPC location.
        Location loc = LocationUris.parse("localhost:8815");
        Assert.assertNotNull(loc);
        Assert.assertTrue(loc.getUri().toString().contains("localhost"));
        Assert.assertTrue(loc.getUri().toString().contains("8815"));
    }

    @Test
    public void grpc_tls_scheme_is_accepted_not_rejected() {
        // Regression guard for the reconciliation: the former TensorFlightClient
        // parser threw "Unsupported location URI scheme" for anything but
        // grpc/grpc+tcp. The tolerant parser must pass grpc+tls through.
        Location loc = LocationUris.parse("grpc+tls://localhost:8816");
        Assert.assertNotNull(loc);
        Assert.assertTrue(loc.getUri().toString().contains("8816"));
    }
}
