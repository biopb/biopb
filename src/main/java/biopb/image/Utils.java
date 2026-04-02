package biopb.image;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.logging.Logger;

import com.google.protobuf.ByteString;

import net.imglib2.view.Views;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.numeric.real.DoubleType;

public final class Utils {
    private Utils() {
        // Prevent instantiation
    }

    private static final Logger LOGGER = Logger.getLogger(Utils.class.getName());

    /**
     * Strip numpy byteorder prefixes from dtype string for backward compatibility.
     *
     * <p>Numpy dtype strings may include byteorder prefixes:
     * <ul>
     *   <li>'&gt;' - big-endian</li>
     *   <li>'&lt;' - little-endian</li>
     *   <li>'|' - native (for types that don't care about endianness)</li>
     *   <li>'=' - native byteorder</li>
     * </ul>
     *
     * <p>These prefixes are stripped since endianness is handled by BinData.endianness field.
     *
     * @param dtype the dtype string potentially containing prefixes
     * @return the dtype string without byteorder prefixes
     */
    private static String stripDtypePrefix(String dtype) {
        while (dtype.length() > 0 && (dtype.charAt(0) == '>' || dtype.charAt(0) == '<' ||
               dtype.charAt(0) == '|' || dtype.charAt(0) == '=')) {
            dtype = dtype.substring(1);
        }
        return dtype;
    }

	private static <T> long getIntervalSize(RandomAccessibleInterval<T> interval) {
        long size = 1L;
        for (int i = 0; i < interval.numDimensions(); i++) {
            if ( interval.dimension(i) > 0 )
                size = size * interval.dimension(i);
        }
        return size;
    }

    /**
     * Serialize a RandomAccessibleInterval to Pixels protobuf with default dimension order "XYZCT".
     *
     * <p>This method converts imglib2 image data to a protobuf format suitable for
     * gRPC transmission. The input interval is assumed to be in imglib2's XYZC dimension
     * order (dimension 0 = X, dimension 1 = Y, dimension 2 = Z, dimension 3 = C).
     *
     * <p>The original data type is preserved. Big-endian byte order is used.
     *
     * <p>Dimension handling:
     * <ul>
     *   <li>2D inputs (X, Y) are promoted to 4D by adding singleton Z and C dimensions</li>
     *   <li>3D inputs (X, Y, Z) are promoted to 4D by adding a singleton C dimension</li>
     *   <li>4D inputs (X, Y, Z, C) are used as-is</li>
     *   <li>5D inputs (X, Y, Z, C, T) are used as-is</li>
     * </ul>
     *
     * @param crop the input RandomAccessibleInterval to serialize
     * @return a Pixels protobuf message containing the serialized image data
     * @throws IllegalArgumentException if the input has more than 5 dimensions
     */
    public static <T extends RealType<T> & NativeType<T> > Pixels SerializeFromInterval(RandomAccessibleInterval<T> crop) {
        return SerializeFromInterval(crop, "XYZCT", null);
    }

    /**
     * Serialize a RandomAccessibleInterval to Pixels protobuf with specified dimension order.
     *
     * <p>This method converts imglib2 image data to a protobuf format suitable for
     * gRPC transmission. The input interval is assumed to be in imglib2's XYZC dimension
     * order (dimension 0 = X, dimension 1 = Y, dimension 2 = Z, dimension 3 = C).
     *
     * <p>The original data type is preserved. Big-endian byte order is used.
     *
     * <p>The dimension_order string describes how bytes are laid out in memory:
     * <ul>
     *   <li>"XYZCT" - X varies fastest (Fortran/memory order), default for imglib2</li>
     *   <li>"CXYZT" - C varies fastest (C order), used by Python/numpy</li>
     * </ul>
     *
     * <p>Dimension handling:
     * <ul>
     *   <li>2D inputs (X, Y) are promoted to 4D by adding singleton Z and C dimensions</li>
     *   <li>3D inputs (X, Y, Z) are promoted to 4D by adding a singleton C dimension</li>
     *   <li>4D inputs (X, Y, Z, C) are used as-is</li>
     *   <li>5D inputs (X, Y, Z, C, T) are used as-is</li>
     * </ul>
     *
     * @param crop the input RandomAccessibleInterval to serialize
     * @param dimensionOrder the dimension order string (e.g., "XYZCT" or "CXYZT"). Must be 5 chars.
     * @return a Pixels protobuf message containing the serialized image data
     * @throws IllegalArgumentException if the input has more than 5 dimensions
     */
    public static <T extends RealType<T> & NativeType<T> > Pixels SerializeFromInterval(RandomAccessibleInterval<T> crop, String dimensionOrder) {
        return SerializeFromInterval(crop, dimensionOrder, null);
    }

    /**
     * Serialize a RandomAccessibleInterval to Pixels protobuf with specified dimension orders.
     *
     * <p>This method converts imglib2 image data to a protobuf format suitable for
     * gRPC transmission. The original data type is preserved.
     *
     * <p>Byte order (endianness) is always big-endian in the output.
     *
     * @param crop the input RandomAccessibleInterval to serialize
     * @param dimensionOrder F-order string for output protobuf (must be 5 chars, e.g., "XYZCT").
     *                       First letter varies fastest in the serialized bytes.
     * @param imglibIndexOrder String describing how imglib2 dimensions map to axis letters.
     *                         First letter = dimension 0, second = dimension 1, etc.
     *                         If null, inferred from input dimensions:
     *                         2D -&gt; "XY", 3D -&gt; "XYZ", 4D -&gt; "XYZC", 5D -&gt; "XYZCT"
     * @return a Pixels protobuf message containing the serialized image data
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public static <T extends RealType<T> & NativeType<T> > Pixels SerializeFromInterval(
            RandomAccessibleInterval<T> crop, String dimensionOrder, String imglibIndexOrder) {

        int nd = crop.numDimensions();

        // Validate dimensionOrder (must be 5 chars, F-order)
        dimensionOrder = dimensionOrder.toUpperCase();
        String validChars = "XYZCT";
        if (dimensionOrder.length() != 5 || !dimensionOrder.chars().allMatch(c -> validChars.indexOf(c) >= 0) ||
            dimensionOrder.chars().distinct().count() != 5) {
            throw new IllegalArgumentException(
                "dimensionOrder must be a permutation of 'XYZCT' (5 chars), got '" + dimensionOrder + "'");
        }

        // Infer or validate imglibIndexOrder
        if (imglibIndexOrder == null) {
            // Default: dimension 0 = first letter, etc.
            if (nd == 2) {
                imglibIndexOrder = "XY";
            } else if (nd == 3) {
                imglibIndexOrder = "XYZ";
            } else if (nd == 4) {
                imglibIndexOrder = "XYZC";
            } else if (nd == 5) {
                imglibIndexOrder = "XYZCT";
            } else {
                throw new IllegalArgumentException("Unsupported number of dimensions: " + nd);
            }
        } else {
            imglibIndexOrder = imglibIndexOrder.toUpperCase();
            if (imglibIndexOrder.length() < 2 || imglibIndexOrder.length() > 5) {
                throw new IllegalArgumentException(
                    "imglibIndexOrder must be 2-5 chars, got '" + imglibIndexOrder + "'");
            }
            if (!imglibIndexOrder.chars().allMatch(c -> validChars.indexOf(c) >= 0)) {
                throw new IllegalArgumentException(
                    "imglibIndexOrder must contain only chars from 'XYZCT', got '" + imglibIndexOrder + "'");
            }
            if (imglibIndexOrder.chars().distinct().count() != imglibIndexOrder.length()) {
                throw new IllegalArgumentException(
                    "imglibIndexOrder must not have duplicate chars, got '" + imglibIndexOrder + "'");
            }
            if (nd != imglibIndexOrder.length()) {
                throw new IllegalArgumentException(
                    "imglibIndexOrder length (" + imglibIndexOrder.length() +
                    ") must match interval dimensions (" + nd + ")");
            }
        }

        // Build size map from input
        java.util.Map<Character, Long> sizes = new java.util.HashMap<>();
        for (int i = 0; i < nd; i++) {
            char axis = imglibIndexOrder.charAt(i);
            sizes.put(axis, crop.dimension(i));
        }
        // Set missing axes to 1
        for (char axis : validChars.toCharArray()) {
            if (!sizes.containsKey(axis)) {
                sizes.put(axis, 1L);
            }
        }

        // Determine dtype from the RealType
        String dtype = getDtypeFromRealType(crop.getType());
        int bytesPerPixel = getBytesPerPixel(dtype);

        // Calculate total size
        long totalSize = getIntervalSize(crop);

        // Build the dimension order for iteration
        // dimensionOrder is F-order: first letter varies fastest
        // We iterate with the outermost loop being the last letter, innermost being first
        // This matches the F-order memory layout

        ByteBuffer buffer = ByteBuffer.allocate((int) (totalSize * bytesPerPixel));

        // Iterate in the order specified by dimensionOrder (F-order)
        // We need to map dimensionOrder letters back to imglib2 dimension indices
        int[] dimIndices = new int[5];  // maps position in dimensionOrder to imglib2 dim index
        long[] dimSizes = new long[5];
        for (int i = 0; i < 5; i++) {
            char axis = dimensionOrder.charAt(i);
            dimSizes[i] = sizes.get(axis);
            int idx = imglibIndexOrder.indexOf(axis);
            dimIndices[i] = idx;  // -1 if not present (singleton)
        }

        // Iterate in F-order (dimensionOrder[0] varies fastest)
        // We iterate from the outermost dimension (last in array) to innermost (first in array)
        // This makes dimensionOrder[0] the innermost loop (fastest varying)
        iterateAndCopy(crop, buffer, dimIndices, dimSizes, 4, new long[nd], dtype);

        // serialize
        BinData bindata = BinData.newBuilder()
                .setData(ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels.Builder pixelsBuilder = Pixels.newBuilder()
                .setDimensionOrder(dimensionOrder)
                .setBindata(bindata)
                .setDtype(dtype)
                .setSizeX(sizes.get('X').intValue())
                .setSizeY(sizes.get('Y').intValue())
                .setSizeZ(sizes.get('Z').intValue())
                .setSizeC(sizes.get('C').intValue())
                .setSizeT(sizes.get('T').intValue());

        return pixelsBuilder.build();
    }

    /**
     * Get numpy-style dtype string from imglib2 RealType.
     */
    private static String getDtypeFromRealType(RealType<?> type) {
        if (type instanceof FloatType) {
            return "f4";
        } else if (type instanceof DoubleType) {
            return "f8";
        } else if (type instanceof UnsignedByteType) {
            return "u1";
        } else if (type instanceof UnsignedShortType) {
            return "u2";
        } else if (type instanceof UnsignedIntType) {
            return "u4";
        } else if (type instanceof ByteType) {
            return "i1";
        } else if (type instanceof ShortType) {
            return "i2";
        } else if (type instanceof IntType) {
            return "i4";
        } else {
            // Default to float32 for unknown types
            return "f4";
        }
    }

    /**
     * Get bytes per pixel for a given dtype.
     */
    private static int getBytesPerPixel(String dtype) {
        // Strip any byteorder prefix
        while (dtype.length() > 0 && (dtype.charAt(0) == '>' || dtype.charAt(0) == '<' ||
               dtype.charAt(0) == '|' || dtype.charAt(0) == '=')) {
            dtype = dtype.substring(1);
        }
        if (dtype.equals("f4") || dtype.equals("u4") || dtype.equals("i4")) {
            return 4;
        } else if (dtype.equals("f8")) {
            return 8;
        } else if (dtype.equals("u2") || dtype.equals("i2")) {
            return 2;
        } else if (dtype.equals("u1") || dtype.equals("i1")) {
            return 1;
        } else {
            return 4;  // Default to float32
        }
    }

    /**
     * Recursively iterate over dimensions in F-order and copy values to buffer.
     * Starts from the outermost dimension and works inward.
     */
    @SuppressWarnings("unchecked")
    private static <T extends RealType<T> & NativeType<T>> void iterateAndCopy(
            RandomAccessibleInterval<T> interval,
            ByteBuffer buffer,
            int[] dimIndices,  // which imglib2 dimension each axis maps to (-1 for singleton)
            long[] dimSizes,    // size of each axis
            int axis,           // current axis being iterated (counts down from 4 to -1)
            long[] pos,         // current position in imglib2 dimensions
            String dtype) {

        if (axis < 0) {
            // All dimensions iterated, copy the value
            T pixel = interval.getAt(pos);
            if (dtype.equals("f4")) {
                buffer.putFloat(pixel.getRealFloat());
            } else if (dtype.equals("f8")) {
                buffer.putDouble(pixel.getRealDouble());
            } else if (dtype.equals("u1")) {
                buffer.put((byte) pixel.getRealFloat());
            } else if (dtype.equals("i1")) {
                buffer.put((byte) pixel.getRealFloat());
            } else if (dtype.equals("u2")) {
                buffer.putShort((short) pixel.getRealFloat());
            } else if (dtype.equals("i2")) {
                buffer.putShort((short) pixel.getRealFloat());
            } else if (dtype.equals("u4")) {
                buffer.putInt((int) pixel.getRealFloat());
            } else if (dtype.equals("i4")) {
                buffer.putInt((int) pixel.getRealFloat());
            } else {
                buffer.putFloat(pixel.getRealFloat());
            }
            return;
        }

        int dimIdx = dimIndices[axis];
        long size = dimSizes[axis];

        if (dimIdx < 0 || size == 1) {
            // Singleton dimension, just recurse
            iterateAndCopy(interval, buffer, dimIndices, dimSizes, axis - 1, pos, dtype);
        } else {
            // Iterate over this dimension
            for (long i = 0; i < size; i++) {
                pos[dimIdx] = i;
                iterateAndCopy(interval, buffer, dimIndices, dimSizes, axis - 1, pos, dtype);
            }
        }
    }

    /**
     * Deserialize a Pixels protobuf message to a RandomAccessibleInterval.
     *
     * <p>This method converts protobuf image data received via gRPC to imglib2 format.
     * The returned interval is in imglib2's XYZC dimension order
     * (dimension 0 = X, dimension 1 = Y, dimension 2 = Z, dimension 3 = C).
     *
     * <p>This is equivalent to calling {@link #DeserializeToInterval(Pixels, String)}
     * with outputIndexOrder="XYZC".
     *
     * @param pixels the protobuf message containing serialized image data
     * @return a RandomAccessibleInterval in XYZC dimension order
     * @throws IllegalArgumentException if the dimension order is invalid or dtype is unsupported
     */
    public static RandomAccessibleInterval<?> DeserializeToInterval(Pixels pixels) {
        return DeserializeToInterval(pixels, "XYZC");
    }

    /**
     * Deserialize a Pixels protobuf message to a RandomAccessibleInterval with specified output index order.
     *
     * <p>This method converts protobuf image data received via gRPC to imglib2 format.
     * The returned interval has the specified dimension order.
     *
     * <p>Supported data types (dtype):
     * <ul>
     *   <li>"f4" or "float32" - 32-bit float</li>
     *   <li>"f8" or "float64" - 64-bit float</li>
     *   <li>"u1" or "uint8" - 8-bit unsigned integer</li>
     *   <li>"u2" or "uint16" - 16-bit unsigned integer</li>
     *   <li>"u4" or "uint32" - 32-bit unsigned integer</li>
     *   <li>"i1" or "int8" - 8-bit signed integer</li>
     *   <li>"i2" or "int16" - 16-bit signed integer</li>
     *   <li>"i4" or "int32" - 32-bit signed integer</li>
     * </ul>
     *
     * <p>The outputIndexOrder specifies which dimensions to include in the output.
     * Dimensions not in outputIndexOrder are squeezed (must be singleton).
     *
     * <p>Byte order (endianness) is read from the BinData field and applied correctly.
     * Dtype prefixes like "&gt;", "&lt;", "|", "=" are automatically stripped for backward
     * compatibility. Note that BinData.endianness is the authoritative source for endianness;
     * a warning is logged if the dtype prefix conflicts with BinData.endianness.
     *
     * @param pixels the protobuf message containing serialized image data
     * @param outputIndexOrder the desired dimension order of the output (2-5 chars, e.g., "XY", "XYZC", "XYZCT").
     *                         Must be a subset permutation of "XYZCT".
     * @return a RandomAccessibleInterval with the specified dimension order
     * @throws IllegalArgumentException if the dimension order is invalid, outputIndexOrder is invalid,
     *                                  a non-singleton dimension is excluded, or dtype is unsupported
     */
    public static RandomAccessibleInterval<?> DeserializeToInterval(Pixels pixels, String outputIndexOrder) {
        String dimOrder = pixels.getDimensionOrder().toUpperCase();

        // Get dimension sizes
        java.util.Map<Character, Long> dims = new java.util.HashMap<>();
        dims.put('X', (long) (pixels.getSizeX() > 0 ? pixels.getSizeX() : 1));
        dims.put('Y', (long) (pixels.getSizeY() > 0 ? pixels.getSizeY() : 1));
        dims.put('Z', (long) (pixels.getSizeZ() > 0 ? pixels.getSizeZ() : 1));
        dims.put('C', (long) (pixels.getSizeC() > 0 ? pixels.getSizeC() : 1));
        dims.put('T', (long) (pixels.getSizeT() > 0 ? pixels.getSizeT() : 1));

        // Validate input dimension_order
        String validChars = "XYZCT";
        if (!dimOrder.chars().allMatch(c -> validChars.indexOf(c) >= 0)) {
            throw new IllegalArgumentException(
                "Invalid dimension order: '" + dimOrder + "' must contain only chars from 'XYZCT'");
        }

        // Validate outputIndexOrder (2-5 chars)
        outputIndexOrder = outputIndexOrder.toUpperCase();
        if (outputIndexOrder.length() < 2 || outputIndexOrder.length() > 5) {
            throw new IllegalArgumentException(
                "outputIndexOrder must be 2-5 chars, got '" + outputIndexOrder + "'");
        }
        if (!outputIndexOrder.chars().allMatch(c -> validChars.indexOf(c) >= 0)) {
            throw new IllegalArgumentException(
                "outputIndexOrder must contain only chars from 'XYZCT', got '" + outputIndexOrder + "'");
        }
        if (outputIndexOrder.chars().distinct().count() != outputIndexOrder.length()) {
            throw new IllegalArgumentException(
                "outputIndexOrder must not have duplicate chars, got '" + outputIndexOrder + "'");
        }

        // Validate: dimensions not in output must be singleton
        for (char axis : validChars.toCharArray()) {
            if (outputIndexOrder.indexOf(axis) < 0 && dims.get(axis) > 1) {
                throw new IllegalArgumentException(
                    "Dimension " + axis + " has size " + dims.get(axis) +
                    " but is not in outputIndexOrder '" + outputIndexOrder + "'. Cannot squeeze non-singleton dimension.");
            }
        }

        String originalDtype = pixels.getDtype();
        String dtype = stripDtypePrefix(originalDtype);

        // Check for endianness conflict
        if (originalDtype.length() > 0) {
            char prefix = originalDtype.charAt(0);
            if (prefix == '<' || prefix == '>') {
                boolean dtypeIsLittleEndian = (prefix == '<');
                boolean bindataIsLittleEndian = (pixels.getBindata().getEndianness() == BinData.Endianness.LITTLE);
                if (dtypeIsLittleEndian != bindataIsLittleEndian) {
                    LOGGER.warning(
                        "Endianness conflict: dtype=" + originalDtype + " indicates " +
                        (dtypeIsLittleEndian ? "little" : "big") + "-endian but " +
                        "BinData.endianness=" + pixels.getBindata().getEndianness().name() + ". " +
                        "Using BinData.endianness as authoritative source."
                    );
                }
            }
        }

        ByteBuffer buffer = ByteBuffer.wrap(pixels.getBindata().getData().toByteArray());
        if (pixels.getBindata().getEndianness() == BinData.Endianness.LITTLE) {
            buffer.order(ByteOrder.LITTLE_ENDIAN);
        } else {
            buffer.order(ByteOrder.BIG_ENDIAN);
        }

        // Build dimensions array matching protobuf's dimension_order (F-order: first letter varies fastest)
        // ArrayImgFactory creates F-order arrays, so dimOrder[0] maps to dimension 0, etc.
        int numDims = dimOrder.length();
        long[] bufferDims = new long[numDims];
        for (int i = 0; i < numDims; i++) {
            bufferDims[i] = dims.get(dimOrder.charAt(i));
        }

        // Create ArrayImg and fill from buffer (F-order iteration matches buffer layout)
        // Transform: squeeze dimensions not in outputIndexOrder, then permute to desired order
        if (dtype.equals("f4") || dtype.equals("float32")) {
            RandomAccessibleInterval<FloatType> interval = new ArrayImgFactory<>(new FloatType()).create(bufferDims);
            for (FloatType p : Views.flatIterable(interval)) {
                p.set(buffer.getFloat());
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("f8") || dtype.equals("float64")) {
            RandomAccessibleInterval<DoubleType> interval = new ArrayImgFactory<>(new DoubleType()).create(bufferDims);
            for (DoubleType p : Views.flatIterable(interval)) {
                p.set(buffer.getDouble());
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("u1") || dtype.equals("uint8")) {
            RandomAccessibleInterval<UnsignedByteType> interval = new ArrayImgFactory<>(new UnsignedByteType()).create(bufferDims);
            for (UnsignedByteType p : Views.flatIterable(interval)) {
                p.set(buffer.get());
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("u2") || dtype.equals("uint16")) {
            RandomAccessibleInterval<UnsignedShortType> interval = new ArrayImgFactory<>(new UnsignedShortType()).create(bufferDims);
            for (UnsignedShortType p : Views.flatIterable(interval)) {
                p.set(buffer.getShort() & 0xFFFF);
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("u4") || dtype.equals("uint32")) {
            RandomAccessibleInterval<UnsignedIntType> interval = new ArrayImgFactory<>(new UnsignedIntType()).create(bufferDims);
            for (UnsignedIntType p : Views.flatIterable(interval)) {
                p.set(buffer.getInt() & 0xFFFFFFFFL);
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("i1") || dtype.equals("int8")) {
            RandomAccessibleInterval<ByteType> interval = new ArrayImgFactory<>(new ByteType()).create(bufferDims);
            for (ByteType p : Views.flatIterable(interval)) {
                p.set(buffer.get());
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("i2") || dtype.equals("int16")) {
            RandomAccessibleInterval<ShortType> interval = new ArrayImgFactory<>(new ShortType()).create(bufferDims);
            for (ShortType p : Views.flatIterable(interval)) {
                p.set(buffer.getShort());
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else if (dtype.equals("i4") || dtype.equals("int32")) {
            RandomAccessibleInterval<IntType> interval = new ArrayImgFactory<>(new IntType()).create(bufferDims);
            for (IntType p : Views.flatIterable(interval)) {
                p.set(buffer.getInt());
            }
            return transformToOutputOrder(interval, dimOrder, outputIndexOrder, dims);

        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dtype);
        }
    }

    /**
     * Transform an interval from protobuf dimension_order to the desired output index order.
     * Step 1: Squeeze dimensions not in outputIndexOrder (must be singleton).
     * Step 2: Permute remaining dimensions to match outputIndexOrder.
     *
     * @param interval the interval in protobuf's dimension_order (F-order)
     * @param dimOrder the protobuf dimension_order string (e.g., "XYZCT")
     * @param outputIndexOrder the desired output dimension order (e.g., "ZYXC")
     * @param dims map of axis sizes
     * @return transformed interval with outputIndexOrder
     */
    @SuppressWarnings("unchecked")
    private static <T> RandomAccessibleInterval<T> transformToOutputOrder(
            RandomAccessibleInterval<T> interval,
            String dimOrder,
            String outputIndexOrder,
            java.util.Map<Character, Long> dims) {

        // Step 1: Squeeze dimensions not in outputIndexOrder
        // We need to track which original dimOrder indices remain after squeezing
        java.util.List<Integer> remainingIndices = new java.util.ArrayList<>();
        for (int i = 0; i < dimOrder.length(); i++) {
            char axis = dimOrder.charAt(i);
            if (outputIndexOrder.indexOf(axis) >= 0) {
                remainingIndices.add(i);
            } else if (dims.get(axis) > 1) {
                throw new IllegalArgumentException(
                    "Dimension " + axis + " has size " + dims.get(axis) +
                    " but is not in outputIndexOrder '" + outputIndexOrder + "'. Cannot squeeze non-singleton dimension.");
            }
            // Singleton dimensions not in outputIndexOrder are implicitly squeezed
        }

        // Apply hyperSlice for each squeezed dimension (process from highest index to lowest)
        RandomAccessibleInterval<T> result = interval;
        java.util.List<Integer> indicesToSqueeze = new java.util.ArrayList<>();
        for (int i = dimOrder.length() - 1; i >= 0; i--) {
            char axis = dimOrder.charAt(i);
            if (outputIndexOrder.indexOf(axis) < 0) {
                indicesToSqueeze.add(i);
            }
        }

        for (int idx : indicesToSqueeze) {
            result = Views.hyperSlice(result, idx, 0);
        }

        // Step 2: Permute to outputIndexOrder
        // After squeezing, remainingIndices[i] tells us which original dimOrder position
        // is now at position i in the squeezed interval
        // We need to rearrange so that outputIndexOrder[j] is at position j

        // Build mapping: for each position in outputIndexOrder, find its current position after squeezing
        int outputDims = outputIndexOrder.length();
        int[] currentPos = new int[outputDims];  // currentPos[j] = where outputIndexOrder[j] currently is
        for (int j = 0; j < outputDims; j++) {
            char axis = outputIndexOrder.charAt(j);
            int originalIdx = dimOrder.indexOf(axis);
            // Find where this original index is now in the squeezed interval
            for (int k = 0; k < remainingIndices.size(); k++) {
                if (remainingIndices.get(k) == originalIdx) {
                    currentPos[j] = k;
                    break;
                }
            }
        }

        // Selection sort: place each outputIndexOrder[j] at position j
        for (int targetDim = 0; targetDim < outputDims; targetDim++) {
            int current = currentPos[targetDim];
            if (current != targetDim) {
                result = Views.permute(result, current, targetDim);
                // Update currentPos: swap the positions
                for (int j = targetDim + 1; j < outputDims; j++) {
                    if (currentPos[j] == targetDim) {
                        currentPos[j] = current;
                    } else if (currentPos[j] == current) {
                        currentPos[j] = targetDim;
                    }
                }
            }
        }

        return result;
    }

}
