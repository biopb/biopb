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
     * <p>The serialization converts all pixel values to 32-bit float (f4) format.
     * Big-endian byte order is used.
     *
     * <p>Dimension handling:
     * <ul>
     *   <li>2D inputs (Y, X) are promoted to 4D by adding singleton Z and C dimensions</li>
     *   <li>3D inputs (Y, X, C) are promoted to 4D by adding a singleton Z dimension</li>
     *   <li>4D inputs are used as-is</li>
     * </ul>
     *
     * @param crop the input RandomAccessibleInterval to serialize
     * @return a Pixels protobuf message containing the serialized image data
     * @throws IllegalArgumentException if the input has more than 4 dimensions
     */
    public static <T extends RealType<T> & NativeType<T> > Pixels SerializeFromInterval(RandomAccessibleInterval<T> crop) {
        return SerializeFromInterval(crop, "XYZCT");
    }

    /**
     * Serialize a RandomAccessibleInterval to Pixels protobuf with specified dimension order.
     *
     * <p>This method converts imglib2 image data to a protobuf format suitable for
     * gRPC transmission. The input interval is assumed to be in imglib2's XYZC dimension
     * order (dimension 0 = X, dimension 1 = Y, dimension 2 = Z, dimension 3 = C).
     *
     * <p>The serialization converts all pixel values to 32-bit float (f4) format.
     * Big-endian byte order is used.
     *
     * <p>The dimension_order string describes how bytes are laid out in memory:
     * <ul>
     *   <li>"XYZCT" - X varies fastest (Fortran/memory order), default for imglib2</li>
     *   <li>"CXYZT" - C varies fastest (C order), used by Python/numpy</li>
     * </ul>
     *
     * <p>Dimension handling:
     * <ul>
     *   <li>2D inputs (Y, X) are promoted to 4D by adding singleton Z and C dimensions</li>
     *   <li>3D inputs (Y, X, C) are promoted to 4D by adding a singleton Z dimension</li>
     *   <li>4D inputs are used as-is</li>
     * </ul>
     *
     * @param crop the input RandomAccessibleInterval to serialize
     * @param dimensionOrder the dimension order string (e.g., "XYZCT" or "CXYZT")
     * @return a Pixels protobuf message containing the serialized image data
     * @throws IllegalArgumentException if the input has more than 4 dimensions
     */
    public static <T extends RealType<T> & NativeType<T> > Pixels SerializeFromInterval(RandomAccessibleInterval<T> crop, String dimensionOrder) {
        int nd = crop.numDimensions();

        if (nd == 2) {
            crop = Views.addDimension(crop, 0, 0);
            crop = Views.addDimension(crop, 0, 0);
        } else if (nd == 3) {
            crop = Views.addDimension(crop, 0, 0);
        } else if (nd != 4) {
            throw new IllegalArgumentException("Unsupported number of dimensions: " + nd);
        }

        // copy to byte array.
        Converter<RealType<T>, FloatType> converter = RealTypeConverters.getConverter(crop.getType(), new FloatType());
        ByteBuffer buffer = ByteBuffer.allocate((int) (getIntervalSize(crop) * Float.BYTES));
        for (T pixel : Views.flatIterable(crop)) {
            FloatType value = new FloatType();
            converter.convert(pixel, value);
            buffer.putFloat(value.get());
        }

        // serialize
        BinData bindata = BinData.newBuilder()
                .setData(ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder(dimensionOrder)
                .setBindata(bindata)
                .setDtype("f4")
                .setSizeX((int) crop.dimension(0))
                .setSizeY((int) crop.dimension(1))
                .setSizeZ((int) crop.dimension(2))
                .setSizeC((int) crop.dimension(3))
                .build();

        return pixels;

    }

    /**
     * Deserialize a Pixels protobuf message to a RandomAccessibleInterval.
     *
     * <p>This method converts protobuf image data received via gRPC to imglib2 format.
     * The returned interval is always in imglib2's XYZC dimension order
     * (dimension 0 = X, dimension 1 = Y, dimension 2 = Z, dimension 3 = C).
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
     * <p>The dimension_order string describes how bytes are laid out in the input buffer.
     * The method handles arbitrary dimension orders by:
     * <ol>
     *   <li>Reading data in the specified order</li>
     *   <li>Permuting axes to imglib2's XYZC convention</li>
     *   <li>Squeezing out singleton T dimensions if present</li>
     * </ol>
     *
     * <p>Byte order (endianness) is read from the BinData field and applied correctly.
     * Dtype prefixes like "&gt;", "&lt;", "|", "=" are automatically stripped for backward
     * compatibility. Note that BinData.endianness is the authoritative source for endianness;
     * a warning is logged if the dtype prefix conflicts with BinData.endianness.
     *
     * @param pixels the protobuf message containing serialized image data
     * @return a RandomAccessibleInterval in XYZC dimension order
     * @throws IllegalArgumentException if the dimension order is invalid or dtype is unsupported
     */
    public static RandomAccessibleInterval<?> DeserializeToInterval(Pixels pixels) {
        String dimOrder = pixels.getDimensionOrder().toUpperCase();

        int dimZ = pixels.getSizeZ() > 0 ? pixels.getSizeZ() : 1;
        int dimY = pixels.getSizeY() > 0 ? pixels.getSizeY() : 1;
        int dimX = pixels.getSizeX() > 0 ? pixels.getSizeX() : 1;
        int dimC = pixels.getSizeC() > 0 ? pixels.getSizeC() : 1;

        // Validate dimension order contains required axes
        if (!dimOrder.contains("X") || !dimOrder.contains("Y") ||
            !dimOrder.contains("Z") || !dimOrder.contains("C")) {
            throw new IllegalArgumentException("Invalid dimension order: " + dimOrder);
        }

        // Get axis positions in the dimension_order string
        // In the dimension_order string, position 0 = fastest varying (first in memory)
        // In imglib2, dimension 0 = fastest varying
        int xPos = dimOrder.indexOf('X');
        int yPos = dimOrder.indexOf('Y');
        int zPos = dimOrder.indexOf('Z');
        int cPos = dimOrder.indexOf('C');

        String originalDtype = pixels.getDtype();
        String dtype = stripDtypePrefix(originalDtype);

        // Check for endianness conflict between dtype prefix and BinData field
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

        // Build dimensions array: for each position in the buffer order, what is the size?
        // dimension_order gives us: position 0 = fastest, position 1 = second, etc.
        // We need to handle dimension orders that may include T (treated as singleton)
        int numDims = dimOrder.length();
        long[] bufferDims = new long[numDims];
        for (int i = 0; i < numDims; i++) {
            char axis = dimOrder.charAt(i);
            if (axis == 'X') bufferDims[i] = dimX;
            else if (axis == 'Y') bufferDims[i] = dimY;
            else if (axis == 'Z') bufferDims[i] = dimZ;
            else if (axis == 'C') bufferDims[i] = dimC;
            else if (axis == 'T') bufferDims[i] = 1; // T is always singleton
            else bufferDims[i] = 1; // Unknown dimension treated as singleton
        }

        if (dtype.equals("f4") || dtype.equals("float32")) {
            // Create interval with dimensions in buffer order
            RandomAccessibleInterval<FloatType> interval = new ArrayImgFactory<>(new FloatType()).create(bufferDims);
            for (FloatType p : Views.flatIterable(interval)) {
                p.set(buffer.getFloat());
            }

            // Permute from buffer order to imglib2 XYZC convention
            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("f8") || dtype.equals("float64")) {
            RandomAccessibleInterval<DoubleType> interval = new ArrayImgFactory<>(new DoubleType()).create(bufferDims);
            for (DoubleType p : Views.flatIterable(interval)) {
                p.set(buffer.getDouble());
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("u1") || dtype.equals("uint8")) {
            RandomAccessibleInterval<UnsignedByteType> interval = new ArrayImgFactory<>(new UnsignedByteType()).create(bufferDims);
            for (UnsignedByteType p : Views.flatIterable(interval)) {
                p.set(buffer.get());
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("u2") || dtype.equals("uint16")) {
            RandomAccessibleInterval<UnsignedShortType> interval = new ArrayImgFactory<>(new UnsignedShortType()).create(bufferDims);
            for (UnsignedShortType p : Views.flatIterable(interval)) {
                p.set(buffer.getShort() & 0xFFFF);
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("u4") || dtype.equals("uint32")) {
            RandomAccessibleInterval<UnsignedIntType> interval = new ArrayImgFactory<>(new UnsignedIntType()).create(bufferDims);
            for (UnsignedIntType p : Views.flatIterable(interval)) {
                p.set(buffer.getInt() & 0xFFFFFFFFL);
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("i1") || dtype.equals("int8")) {
            RandomAccessibleInterval<ByteType> interval = new ArrayImgFactory<>(new ByteType()).create(bufferDims);
            for (ByteType p : Views.flatIterable(interval)) {
                p.set(buffer.get());
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("i2") || dtype.equals("int16")) {
            RandomAccessibleInterval<ShortType> interval = new ArrayImgFactory<>(new ShortType()).create(bufferDims);
            for (ShortType p : Views.flatIterable(interval)) {
                p.set(buffer.getShort());
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else if (dtype.equals("i4") || dtype.equals("int32")) {
            RandomAccessibleInterval<IntType> interval = new ArrayImgFactory<>(new IntType()).create(bufferDims);
            for (IntType p : Views.flatIterable(interval)) {
                p.set(buffer.getInt());
            }

            return permuteToXYZC(interval, xPos, yPos, zPos, cPos, numDims);

        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dtype);
        }

    }

    /**
     * Permute an interval from buffer order (given by axis positions) to XYZC order.
     * Handles 4D or 5D intervals where extra dimensions are singletons.
     */
    @SuppressWarnings("unchecked")
    private static <T> RandomAccessibleInterval<T> permuteToXYZC(
            RandomAccessibleInterval<T> interval, int xPos, int yPos, int zPos, int cPos, int numDims) {

        // If we have more than 4 dimensions, we need to squeeze out singleton dimensions
        // by using hyperSlice on singleton dimensions
        while (interval.numDimensions() > 4) {
            int dimToSqueeze = -1;
            // Find a singleton dimension that is not X, Y, Z, or C
            for (int i = 0; i < interval.numDimensions(); i++) {
                if (i != xPos && i != yPos && i != zPos && i != cPos && interval.dimension(i) == 1) {
                    dimToSqueeze = i;
                    break;
                }
            }
            if (dimToSqueeze < 0) {
                throw new IllegalArgumentException(
                    "Cannot reduce dimensions: no singleton dimension to squeeze");
            }
            interval = Views.hyperSlice(interval, dimToSqueeze, 0);
            // Adjust positions
            if (xPos > dimToSqueeze) xPos--;
            if (yPos > dimToSqueeze) yPos--;
            if (zPos > dimToSqueeze) zPos--;
            if (cPos > dimToSqueeze) cPos--;
        }

        // Now we should have 4 dimensions
        if (interval.numDimensions() != 4) {
            throw new IllegalArgumentException(
                "Expected 4 dimensions after squeezing, got " + interval.numDimensions());
        }

        // Apply permutations to get XYZC order
        RandomAccessibleInterval<T> result = interval;
        int[] currentPos = {xPos, yPos, zPos, cPos};

        for (int targetDim = 0; targetDim < 4; targetDim++) {
            int current = currentPos[targetDim];
            if (current != targetDim) {
                result = Views.permute(result, current, targetDim);
                // Update tracked positions after swap
                for (int j = 0; j < 4; j++) {
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
