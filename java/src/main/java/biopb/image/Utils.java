package biopb.image;

import java.nio.ByteBuffer;

import com.google.protobuf.ByteString;

import net.imglib2.view.Views;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converter;
import net.imglib2.converter.RealTypeConverters;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
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

    public static <T extends RealType<T> & NativeType<T> > Pixels SerializeFromInterval(RandomAccessibleInterval<T> crop) {
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
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("f4")
                .setSizeX((int) crop.dimension(0))
                .setSizeY((int) crop.dimension(1))
                .setSizeZ((int) crop.dimension(2))
                .setSizeC((int) crop.dimension(3))
                .build();
    
        return pixels;
    
    }

    public static RandomAccessibleInterval<?> DeserializeToInterval(Pixels pixels) {
        String dimOrder = pixels.getDimensionOrder();
        if (!dimOrder.equals("XYZCT") && !dimOrder.equals("XYZTC")) {
            throw new IllegalArgumentException("Unsupported dimension order: " + dimOrder);
        }

        int dimZ = pixels.getSizeZ() > 0 ? pixels.getSizeZ() : 1;
        int dimY = pixels.getSizeY() > 0 ? pixels.getSizeY() : 1;
        int dimX = pixels.getSizeX() > 0 ? pixels.getSizeX() : 1;
        int dimC = pixels.getSizeC() > 0 ? pixels.getSizeC() : 1;

        String dtype = pixels.getDtype();
        if ( dtype.startsWith(">") ) {
            dtype = dtype.substring(1);
        }

        ByteBuffer buffer = ByteBuffer.wrap(pixels.getBindata().getData().toByteArray());
        if (dtype.equals("f4") || dtype.equals("float32")) {
            RandomAccessibleInterval<FloatType> interval = new ArrayImgFactory<>(new FloatType()).create(dimX, dimY, dimZ, dimC);
            for (FloatType p : Views.flatIterable(interval)) {
                p.set(buffer.getFloat());
            }

            return interval;
        } else if (dtype.equals("u1") || dtype.equals("uint8")) {
            RandomAccessibleInterval<UnsignedByteType> interval = new ArrayImgFactory<>(new UnsignedByteType()).create(dimX, dimY, dimZ, dimC);
            for (UnsignedByteType p : Views.flatIterable(interval)) {
                p.set(buffer.get());
            }

            return interval;
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dtype);
        }

    }

}