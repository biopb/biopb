package biopb.image;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.protobuf.ListValue;
import com.google.protobuf.NullValue;
import com.google.protobuf.Struct;
import com.google.protobuf.Value;

import biopb.tensor.TensorFlightClient;
import net.imglib2.RandomAccessibleInterval;

/**
 * Ops arguments and results: values and pixels to and from {@link Arg}.
 *
 * <p>An {@code Arg} is pixels (inline, or a reference to a tensor on a plane) or
 * anything else as a protobuf {@link Value}. Both sides of the wire use these: a
 * client encodes what it sends and decodes what comes back, a server the reverse.
 * The Java twin of Python's {@code biopb.image._arg}.
 *
 * <p>JSON values are {@code null}, {@link Boolean}, {@link Number}, {@link
 * String}, a {@link Map} with string keys, and a {@link Iterable} or array of
 * these. A pixel array is a {@link RandomAccessibleInterval}.
 */
public final class ArgCodec {

    private ArgCodec() {}

    /**
     * The axes an unlabelled array of each rank is taken to have, as Python's
     * row-major arrays have them: an interval's dimension 0 is the first label.
     * An interval whose dimension 0 is X (imglib2's usual order) must pass its own
     * labels to {@link #encodeArg(Object, List)}.
     */
    public static final Map<Integer, List<String>> NDIM_LABELS;

    static {
        Map<Integer, List<String>> labels = new LinkedHashMap<>();
        labels.put(2, Collections.unmodifiableList(Arrays.asList("Y", "X")));
        labels.put(3, Collections.unmodifiableList(Arrays.asList("Y", "X", "C")));
        labels.put(4, Collections.unmodifiableList(Arrays.asList("Z", "Y", "X", "C")));
        labels.put(5, Collections.unmodifiableList(Arrays.asList("T", "Z", "Y", "X", "C")));
        NDIM_LABELS = Collections.unmodifiableMap(labels);
    }

    /** The cache a lazily-read reference gets when decoded. */
    private static final long LAZY_CACHE_BYTES = 100_000_000L;

    /**
     * {@code value} as a JSON {@code Arg}.
     *
     * @throws IllegalArgumentException a value of a type with no JSON form
     */
    public static Arg jsonArg(Object value) {
        return Arg.newBuilder().setJson(toValue(value)).build();
    }

    /**
     * A {@link Value} as plain Java: {@code null}, {@link Boolean}, {@link Double}
     * or {@link Long}, {@link String}, {@code Map<String, Object>} and {@code
     * List<Object>}.
     *
     * <p>A protobuf number is always a double. With {@code ints}, an integral one
     * comes back as a {@code Long}, so a count the sender wrote as 6 reads as 6,
     * not 6.0.
     */
    public static Object jsonValue(Value value, boolean ints) {
        switch (value.getKindCase()) {
            case NUMBER_VALUE: {
                double number = value.getNumberValue();
                return ints && number == Math.rint(number) && !Double.isInfinite(number)
                        && Math.abs(number) < 9.2e18 ? (Object) (long) number : (Object) number;
            }
            case STRING_VALUE:
                return value.getStringValue();
            case BOOL_VALUE:
                return value.getBoolValue();
            case STRUCT_VALUE: {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Map.Entry<String, Value> field : value.getStructValue().getFieldsMap().entrySet()) {
                    out.put(field.getKey(), jsonValue(field.getValue(), ints));
                }
                return out;
            }
            case LIST_VALUE: {
                List<Object> out = new ArrayList<>();
                for (Value item : value.getListValue().getValuesList()) {
                    out.add(jsonValue(item, ints));
                }
                return out;
            }
            default:
                return null;
        }
    }

    /**
     * {@code value} as an {@code Arg}: a {@link RandomAccessibleInterval} as inline
     * pixels, anything else as JSON. The array's axes are labelled by {@link
     * #NDIM_LABELS} for its rank.
     */
    public static Arg encodeArg(Object value) {
        return encodeArg(value, null);
    }

    /**
     * {@link #encodeArg(Object)} with {@code dimLabels} naming an array's axes
     * (one per dimension); null takes {@link #NDIM_LABELS}. Ignored for a value
     * that is not an array.
     */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public static Arg encodeArg(Object value, List<String> dimLabels) {
        if (value instanceof RandomAccessibleInterval) {
            RandomAccessibleInterval interval = (RandomAccessibleInterval) value;
            List<String> labels = dimLabels != null ? dimLabels : NDIM_LABELS.get(interval.numDimensions());
            return Arg.newBuilder().setEager(Utils.tensorFromInterval(interval, labels)).build();
        }
        return jsonArg(value);
    }

    /** {@link #decodeArg(Arg, boolean)} reading integral numbers as {@code Long}. */
    public static Object decodeArg(Arg arg) {
        return decodeArg(arg, true);
    }

    /**
     * An {@code Arg} as Java: JSON as plain values (see {@link #jsonValue} for
     * {@code ints}), inline pixels as a {@link RandomAccessibleInterval}, and a
     * reference as the lazy interval it names, read from its plane on demand.
     *
     * @throws IllegalArgumentException an empty {@code Arg}
     */
    public static Object decodeArg(Arg arg, boolean ints) {
        switch (arg.getKindCase()) {
            case JSON:
                return jsonValue(arg.getJson(), ints);
            case EAGER:
                return Utils.intervalFromTensor(arg.getEager());
            case LAZY:
                return TensorFlightClient.tensorFromPb(arg.getLazy(), LAZY_CACHE_BYTES);
            default:
                throw new IllegalArgumentException("an empty Arg has no value");
        }
    }

    // ---- JSON values ---------------------------------------------------------------

    /**
     * Set a {@link Value} from plain JSON types, field by field. Not JSON text:
     * a {@code Value} on the wire is protobuf binary, where {@code number_value}
     * is a double that carries NaN and infinity unchanged -- an argument or a
     * result may legitimately be either.
     */
    private static Value toValue(Object value) {
        Value.Builder out = Value.newBuilder();
        if (value == null) {
            return out.setNullValue(NullValue.NULL_VALUE).build();
        }
        if (value instanceof Boolean) {
            return out.setBoolValue((Boolean) value).build();
        }
        if (value instanceof Number) {
            return out.setNumberValue(((Number) value).doubleValue()).build();
        }
        if (value instanceof CharSequence || value instanceof Character) {
            return out.setStringValue(value.toString()).build();
        }
        if (value instanceof Map) {
            Struct.Builder struct = Struct.newBuilder();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                struct.putFields(String.valueOf(entry.getKey()), toValue(entry.getValue()));
            }
            return out.setStructValue(struct).build();
        }
        if (value instanceof Iterable) {
            ListValue.Builder list = ListValue.newBuilder();
            for (Object item : (Iterable<?>) value) {
                list.addValues(toValue(item));
            }
            return out.setListValue(list).build();
        }
        if (value.getClass().isArray()) {
            ListValue.Builder list = ListValue.newBuilder();
            for (int i = 0, n = Array.getLength(value); i < n; i++) {
                list.addValues(toValue(Array.get(value, i)));
            }
            return out.setListValue(list).build();
        }
        throw new IllegalArgumentException("a value of type " + value.getClass().getSimpleName() + " is not JSON");
    }
}
