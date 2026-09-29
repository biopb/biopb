package biopb.tensor;

import java.nio.charset.StandardCharsets;
import java.util.Map;

import org.apache.arrow.flight.ErrorFlightMetadata;
import org.apache.arrow.flight.FlightRuntimeException;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/**
 * Decodes the typed payload a tensor-server Flight error carries.
 *
 * <p><b>The payload is the taxonomy, not the transport status.</b> Flight-in-
 * Python exposes only a subset of gRPC's canonical codes as typed exceptions --
 * there is no {@code FlightNotFoundError} -- so the server rides every terminal
 * domain error on {@code FlightServerError} (the coarsest terminal class it
 * has, which reaches a client as {@code UNKNOWN}) and puts the precise code in
 * {@code extra_info}. Cross-checking the two therefore rejects every error
 * worth decoding; this switches on the payload's own {@code code}, which is
 * what the server documents and what the Python client does
 * ({@code _session._addressing_error}).
 */
public final class TensorErrorMapper {
    private static final Gson GSON = new Gson();

    private TensorErrorMapper() {}

    /** Return a typed SDK exception, or the original error when it is not ours. */
    public static RuntimeException map(FlightRuntimeException error) {
        Payload payload = payload(error.status().metadata());
        if (payload == null) {
            return error;
        }

        String message = message(error);

        // An upload refusal is the one payload with no `code`: it rides
        // FlightCancelledError, and the code would restate the class
        // (upload_manager._refused). So it is keyed on the reason prefix --
        // Python also gates on the class, but a refusal moved to another class
        // would then silently stop decoding, which is the failure this whole
        // decode exists to avoid.
        //
        // Only when there is no code, though: the rule stays "switch on the
        // code, refine on the reason", and a coded error whose reason happens
        // to begin `upload_` is not a refusal. biopb/biopb#1070 covers putting
        // the code on the refusal too, which would retire this branch.
        if (payload.code == null) {
            if (payload.reason != null && payload.reason.startsWith("upload_")) {
                return new UploadRefusedException(message, payload.reason, payload.sourceId,
                        payload.state, payload.detail, error);
            }
            return error;
        }
        switch (payload.code) {
            case "NOT_FOUND":
                return "stale_content_version".equals(payload.reason)
                        ? new StaleReadPlanException(message, payload.reason, error)
                        : new TensorNotFoundException(message, payload.reason, error);
            case "INVALID_ARGUMENT":
                return new InvalidTensorRequestException(message, payload.reason, error);
            case "UNAVAILABLE":
                return new SourceUnresolvedException(message, error);
            default:
                return error;
        }
    }

    /**
     * The server's own message, without the transport's trailer.
     *
     * <p>pyarrow appends its own {@code ". Detail: ..."} to whatever the server
     * said; the Python client cuts at the same marker.
     */
    private static String message(FlightRuntimeException error) {
        String message = error.status().description();
        if (message == null || message.isEmpty()) {
            message = error.getMessage();
        }
        if (message == null) {
            return "";
        }
        int detail = message.indexOf(". Detail:");
        return detail < 0 ? message : message.substring(0, detail);
    }

    private static Payload payload(ErrorFlightMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        for (String key : metadata.keys()) {
            byte[] raw = metadata.getByte(key);
            if (raw == null) {
                continue;
            }
            try {
                Map<String, Object> value = GSON.fromJson(new String(raw, StandardCharsets.UTF_8),
                        new TypeToken<Map<String, Object>>() {}.getType());
                if (value == null) {
                    continue;
                }
                // A payload is ours when it says either what went wrong or why;
                // an upload refusal sends only the latter.
                String code = string(value, "code");
                String reason = string(value, "reason");
                if (code != null || reason != null) {
                    return new Payload(code, reason, string(value, "source_id"),
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
