package biopb.tensor;

import java.util.Map;
import java.util.Optional;

import org.apache.arrow.flight.FlightInfo;
import org.apache.arrow.vector.types.pojo.Schema;

import com.google.protobuf.InvalidProtocolBufferException;

/**
 * The request a read plan answers, whichever protocol wrote the plan.
 *
 * <p>A v3 plan carries the whole {@link TensorReadOption} in
 * {@code FlightInfo.app_metadata}. A v2 plan carried only the requested
 * {@link SliceHint} there and echoed the scale and method on its descriptor, so
 * those are put back together here: everything past this class reads one shape,
 * and a v2 server costs one branch. The mirror of Python's {@code _plan_request}.
 *
 * <p>Which protocol wrote the plan is stamped on its schema, not asked of a
 * connection, because a plan travels (a {@link SerializedTensor}). A plan with no
 * stamp is v2; and a SliceHint's bytes parse as a TensorReadOption with a
 * garbled array_id, which is why the stamp is read first.
 */
final class PlanRequest {

    private PlanRequest() {}

    /** The request {@code plan} answers; empty when it recorded nothing. */
    static TensorReadOption of(FlightInfo plan) {
        byte[] raw = plan.getAppMetadata();
        boolean present = raw != null && raw.length > 0;
        try {
            if (writtenUnder(plan) >= 3) {
                return present ? TensorReadOption.parseFrom(raw) : TensorReadOption.getDefaultInstance();
            }
            TensorDescriptor descriptor = TensorChunkCodec.descriptorOf(plan);
            // A record of a request, not one to send, so it carries no field mask.
            TensorReadOption.Builder request = TensorReadOption.getDefaultInstance().toBuilder()
                    .setArrayId(descriptor.getArrayId())
                    .addAllScaleHint(descriptor.getScaleHintList())
                    .setReductionMethod(descriptor.getReductionMethod());
            if (present) {
                request.setSliceHint(SliceHint.parseFrom(raw));
            }
            return request.build();
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("FlightInfo.app_metadata is not the request it claims", error);
        }
    }

    /** The Flight protocol stamped on the plan's schema; 2 when there is none. */
    private static int writtenUnder(FlightInfo plan) {
        Optional<Schema> schema = plan.getSchemaOptional();
        if (schema.isPresent()) {
            Map<String, String> metadata = schema.get().getCustomMetadata();
            if (metadata != null) {
                String stamped = metadata.get(WireVersions.FLIGHT_PROTOCOL_METADATA_KEY);
                if (stamped != null && !stamped.isEmpty()) {
                    return WireVersions.stampedVersion(stamped);
                }
            }
        }
        return 2;
    }
}
