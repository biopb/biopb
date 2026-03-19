package biopb.image;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.google.protobuf.ByteString;

import net.imglib2.view.Views;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public final class Utils {
    private Utils() {
        // Prevent instantiation
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
     *   <li>"u1" or "uint8" - 8-bit unsigned integer</li>
     *   <li>"u2" or "uint16" - 16-bit unsigned integer</li>
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
     * Dtype prefixes like "&gt;" (big-endian marker) are automatically stripped.
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

        String dtype = pixels.getDtype();
        if ( dtype.startsWith(">") ) {
            dtype = dtype.substring(1);
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
