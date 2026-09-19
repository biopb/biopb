package biopb.tensor;

import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.apache.arrow.flight.CallStatus;
import org.apache.arrow.flight.ErrorFlightMetadata;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStatusCode;
import org.junit.Assert;
import org.junit.Test;

public class TensorErrorMapperTest {
    @Test public void mapsTypedNotFound() {
        RuntimeException mapped = TensorErrorMapper.map(error(FlightStatusCode.NOT_FOUND, "unknown field", "{\"code\":\"NOT_FOUND\",\"reason\":\"unknown_field\"}"));
        Assert.assertTrue(mapped instanceof TensorNotFoundException);
        Assert.assertEquals("unknown_field", ((TensorFlightException) mapped).getReason());
    }
    @Test public void mapsTypedInvalidReadRequest() {
        RuntimeException mapped = TensorErrorMapper.map(error(FlightStatusCode.INVALID_ARGUMENT, "wrong scale rank", "{\"code\":\"INVALID_ARGUMENT\",\"reason\":\"scale_rank\"}"));
        Assert.assertTrue(mapped instanceof InvalidTensorRequestException);
    }
    @Test public void mapsTypedUnresolvedButLeavesNetworkUnavailableRaw() {
        Assert.assertTrue(TensorErrorMapper.map(error(FlightStatusCode.UNAVAILABLE, "unresolved", "{\"code\":\"UNAVAILABLE\"}")) instanceof SourceUnresolvedException);
        FlightRuntimeException network = error(FlightStatusCode.UNAVAILABLE, "connection refused", null);
        Assert.assertSame(network, TensorErrorMapper.map(network));
    }
    @Test public void mapsStalePlanAndUploadRefusal() {
        Assert.assertTrue(TensorErrorMapper.map(error(FlightStatusCode.NOT_FOUND, "content changed", "{\"code\":\"NOT_FOUND\",\"reason\":\"stale_content_version\"}")) instanceof StaleReadPlanException);
        RuntimeException upload = TensorErrorMapper.map(error(FlightStatusCode.CANCELLED, "upload sealed", "{\"code\":\"CANCELLED\",\"reason\":\"upload_sealed\",\"source_id\":\"cache_a\",\"state\":\"READY\",\"detail\":\"done\"}"));
        Assert.assertTrue(upload instanceof UploadRefusedException);
        Assert.assertEquals("cache_a", ((UploadRefusedException) upload).getSourceId());
    }
    @Test public void leavesMalformedOrMismatchedPayloadRaw() {
        FlightRuntimeException malformed = error(FlightStatusCode.NOT_FOUND, "missing", "not json");
        Assert.assertSame(malformed, TensorErrorMapper.map(malformed));
        FlightRuntimeException mismatch = error(FlightStatusCode.NOT_FOUND, "missing", "{\"code\":\"INVALID_ARGUMENT\"}");
        Assert.assertSame(mismatch, TensorErrorMapper.map(mismatch));
    }
    @Test public void mapsErrorsRaisedDuringActionIteration() {
        FlightRuntimeException error = error(FlightStatusCode.INVALID_ARGUMENT, "bad slice",
                "{\"code\":\"INVALID_ARGUMENT\",\"reason\":\"slice_rank\"}");
        Iterator<org.apache.arrow.flight.Result> failing = new Iterator<org.apache.arrow.flight.Result>() {
            @Override public boolean hasNext() { throw error; }
            @Override public org.apache.arrow.flight.Result next() { throw new AssertionError("unreachable"); }
        };
        Assert.assertThrows(InvalidTensorRequestException.class,
                () -> new FlightSession.ErrorMappingIterator(failing).hasNext());
    }
    private static FlightRuntimeException error(FlightStatusCode code, String message, String payload) {
        ErrorFlightMetadata metadata = null;
        if (payload != null) { metadata = new ErrorFlightMetadata(); metadata.insert("x-biopb-error-bin", payload.getBytes(StandardCharsets.UTF_8)); }
        return new CallStatus(code, null, message, metadata).toRuntimeException();
    }
}
