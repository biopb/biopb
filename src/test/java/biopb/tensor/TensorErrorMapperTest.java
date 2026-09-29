package biopb.tensor;

import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.apache.arrow.flight.CallStatus;
import org.apache.arrow.flight.ErrorFlightMetadata;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStatusCode;
import org.junit.Assert;
import org.junit.Test;

/**
 * The transport status these errors arrive with is the one the real server
 * sends, which is the point: pyarrow has no {@code FlightNotFoundError}, so
 * every terminal domain error rides {@code FlightServerError} and reaches a
 * client as {@code UNKNOWN}. A fake that sent {@code NOT_FOUND} on the wire
 * would let a mapper that cross-checks the two pass here and decode nothing in
 * production.
 */
public class TensorErrorMapperTest {

    /** What `FlightServerError` reaches a Java client as. */
    private static final FlightStatusCode TERMINAL = FlightStatusCode.UNKNOWN;

    @Test public void mapsTypedNotFound() {
        RuntimeException mapped = TensorErrorMapper.map(error(TERMINAL, "unknown field",
                "{\"code\":\"NOT_FOUND\",\"reason\":\"unknown_field\"}"));
        Assert.assertTrue(mapped instanceof TensorNotFoundException);
        Assert.assertEquals("unknown_field", ((TensorFlightException) mapped).getReason());
    }

    @Test public void mapsTypedInvalidReadRequest() {
        RuntimeException mapped = TensorErrorMapper.map(error(TERMINAL, "wrong scale rank",
                "{\"code\":\"INVALID_ARGUMENT\",\"reason\":\"scale_rank\"}"));
        Assert.assertTrue(mapped instanceof InvalidTensorRequestException);
    }

    @Test public void mapsTypedUnresolvedButLeavesNetworkUnavailableRaw() {
        // The one code whose class the server can represent: an unresolved
        // source really does ride FlightUnavailableError.
        Assert.assertTrue(TensorErrorMapper.map(error(FlightStatusCode.UNAVAILABLE, "unresolved",
                "{\"code\":\"UNAVAILABLE\"}")) instanceof SourceUnresolvedException);
        FlightRuntimeException network = error(FlightStatusCode.UNAVAILABLE, "connection refused", null);
        Assert.assertSame(network, TensorErrorMapper.map(network));
    }

    @Test public void mapsStalePlan() {
        Assert.assertTrue(TensorErrorMapper.map(error(TERMINAL, "content changed",
                "{\"code\":\"NOT_FOUND\",\"reason\":\"stale_content_version\"}"))
                instanceof StaleReadPlanException);
    }

    @Test public void mapsAnUploadRefusalThatCarriesNoCode() {
        // upload_manager._refused sends no `code`: both kinds ride one exception
        // class, so the state is the data and the class is implied. Keying the
        // decode on `code` would miss every refusal there is.
        RuntimeException upload = TensorErrorMapper.map(error(FlightStatusCode.CANCELLED, "upload sealed",
                "{\"reason\":\"upload_sealed\",\"source_id\":\"cache_a\",\"state\":\"READY\",\"detail\":\"done\"}"));
        Assert.assertTrue(upload instanceof UploadRefusedException);
        Assert.assertEquals("cache_a", ((UploadRefusedException) upload).getSourceId());
        Assert.assertEquals("READY", ((UploadRefusedException) upload).getState());
        Assert.assertEquals("done", ((UploadRefusedException) upload).getDetail());
    }

    @Test public void leavesMalformedOrForeignPayloadRaw() {
        FlightRuntimeException malformed = error(TERMINAL, "missing", "not json");
        Assert.assertSame(malformed, TensorErrorMapper.map(malformed));
        // A trailer that is JSON but says nothing we understand is not ours.
        FlightRuntimeException foreign = error(TERMINAL, "missing", "{\"detail\":\"something else\"}");
        Assert.assertSame(foreign, TensorErrorMapper.map(foreign));
        // A code we have no class for stays raw rather than becoming a wrong one.
        FlightRuntimeException unknownCode = error(TERMINAL, "boom", "{\"code\":\"INTERNAL\"}");
        Assert.assertSame(unknownCode, TensorErrorMapper.map(unknownCode));
    }

    @Test public void stripsTheTransportTrailerFromTheMessage() {
        // pyarrow appends its own ". Detail: ..."; the message a caller reads is
        // the server's sentence, as it is in Python.
        RuntimeException mapped = TensorErrorMapper.map(error(TERMINAL,
                "Source not found: nope. Detail: Failed",
                "{\"code\":\"NOT_FOUND\",\"reason\":\"unknown_field\"}"));
        Assert.assertEquals("Source not found: nope", mapped.getMessage());
    }

    @Test public void mapsErrorsRaisedDuringActionIteration() {
        FlightRuntimeException error = error(TERMINAL, "bad slice",
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
        if (payload != null) {
            metadata = new ErrorFlightMetadata();
            metadata.insert("x-biopb-error-bin", payload.getBytes(StandardCharsets.UTF_8));
        }
        return new CallStatus(code, null, message, metadata).toRuntimeException();
    }
}
