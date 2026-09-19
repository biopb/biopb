package biopb.tensor;

import java.nio.charset.StandardCharsets;
import java.util.Map;

import org.apache.arrow.flight.ErrorFlightMetadata;
import org.apache.arrow.flight.FlightRuntimeException;
import org.apache.arrow.flight.FlightStatusCode;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/** Decodes validated, typed tensor-server Flight errors. */
public final class TensorErrorMapper {
    private static final Gson GSON = new Gson();

    private TensorErrorMapper() {}

    /** Return a typed SDK exception, or the original error when it is not ours. */
    public static RuntimeException map(FlightRuntimeException error) {
        Payload payload = payload(error.status().metadata());
        FlightStatusCode status = error.status().code();
        if (payload == null || !status.name().equals(payload.code)) return error;

        String message = error.status().description();
        if (message == null || message.isEmpty()) message = error.getMessage();
        switch (status) {
            case NOT_FOUND:
                return "stale_content_version".equals(payload.reason)
                        ? new StaleReadPlanException(message, payload.reason, error)
                        : new TensorNotFoundException(message, payload.reason, error);
            case INVALID_ARGUMENT:
                return new InvalidTensorRequestException(message, payload.reason, error);
            case UNAVAILABLE:
                return new SourceUnresolvedException(message, error);
            case CANCELLED:
                if (payload.reason != null && payload.reason.startsWith("upload_")) {
                    return new UploadRefusedException(message, payload.reason, payload.sourceId, payload.state, payload.detail, error);
                }
                return error;
            default:
                return error;
        }
    }

    private static Payload payload(ErrorFlightMetadata metadata) {
        if (metadata == null) return null;
        for (String key : metadata.keys()) {
            byte[] raw = metadata.getByte(key);
            if (raw == null) continue;
            try {
                Map<String, Object> value = GSON.fromJson(new String(raw, StandardCharsets.UTF_8),
                        new TypeToken<Map<String, Object>>() {}.getType());
                Object code = value.get("code");
                if (code instanceof String) {
                    return new Payload((String) code, string(value, "reason"), string(value, "source_id"),
                            string(value, "state"), string(value, "detail"));
                }
            } catch (RuntimeException ignored) {
                // Metadata may contain unrelated trailers; keep looking.
            }
        }
        return null;
    }

    private static String string(Map<String, Object> value, String key) {
        Object result = value.get(key);
        return result instanceof String ? (String) result : null;
    }

    private static final class Payload {
        private final String code, reason, sourceId, state, detail;
        Payload(String code, String reason, String sourceId, String state, String detail) {
            this.code = code; this.reason = reason; this.sourceId = sourceId; this.state = state; this.detail = detail;
        }
    }
}
