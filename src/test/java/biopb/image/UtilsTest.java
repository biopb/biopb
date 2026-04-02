package biopb.image;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.Test;
import org.junit.Assert;

import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import net.imglib2.RandomAccessibleInterval;

public class UtilsTest {

    private static final float DELTA = 0.0001f;

    @Test
    public void testSerializeDeserialize2DImage() {
        // Create a 2D image (10x10) with float values
        ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
        RandomAccessibleInterval<FloatType> image2D = factory.create(10, 10);

        // Fill with test data
        float value = 0.0f;
        for (FloatType pixel : Views.flatIterable(image2D)) {
            pixel.set(value);
            value += 1.0f;
        }

        // Serialize
        Pixels pixels = Utils.SerializeFromInterval(image2D);

        // Verify dimensions (2D should be converted to 4D with Z=1, C=1)
        Assert.assertEquals(10, pixels.getSizeX());
        Assert.assertEquals(10, pixels.getSizeY());
        Assert.assertEquals(1, pixels.getSizeZ());
        Assert.assertEquals(1, pixels.getSizeC());
        Assert.assertEquals("XYZCT", pixels.getDimensionOrder());
        Assert.assertEquals("f4", pixels.getDtype());

        // Deserialize
        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify data
        value = 0.0f;
        for (Object obj : Views.flatIterable(deserialized)) {
            FloatType pixel = (FloatType) obj;
            Assert.assertEquals(value, pixel.get(), DELTA);
            value += 1.0f;
        }
    }

    @Test
    public void testSerializeDeserialize3DImage() {
        // Create a 3D image (5x5x5) with float values
        ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
        RandomAccessibleInterval<FloatType> image3D = factory.create(5, 5, 5);

        // Fill with test data
        float value = 0.0f;
        for (FloatType pixel : Views.flatIterable(image3D)) {
            pixel.set(value);
            value += 1.0f;
        }

        // Serialize
        Pixels pixels = Utils.SerializeFromInterval(image3D);

        // Verify dimensions (3D should be converted to 4D with C=1)
        Assert.assertEquals(5, pixels.getSizeX());
        Assert.assertEquals(5, pixels.getSizeY());
        Assert.assertEquals(5, pixels.getSizeZ());
        Assert.assertEquals(1, pixels.getSizeC());

        // Deserialize
        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify data
        value = 0.0f;
        for (Object obj : Views.flatIterable(deserialized)) {
            FloatType pixel = (FloatType) obj;
            Assert.assertEquals(value, pixel.get(), DELTA);
            value += 1.0f;
        }
    }

    @Test
    public void testSerializeDeserialize4DImage() {
        // Create a 4D image (4x4x2x3) with float values
        ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
        RandomAccessibleInterval<FloatType> image4D = factory.create(4, 4, 2, 3);

        // Fill with test data
        float value = 0.0f;
        for (FloatType pixel : Views.flatIterable(image4D)) {
            pixel.set(value);
            value += 1.0f;
        }

        // Serialize
        Pixels pixels = Utils.SerializeFromInterval(image4D);

        // Verify dimensions
        Assert.assertEquals(4, pixels.getSizeX());
        Assert.assertEquals(4, pixels.getSizeY());
        Assert.assertEquals(2, pixels.getSizeZ());
        Assert.assertEquals(3, pixels.getSizeC());

        // Deserialize
        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify data
        value = 0.0f;
        for (Object obj : Views.flatIterable(deserialized)) {
            FloatType pixel = (FloatType) obj;
            Assert.assertEquals(value, pixel.get(), DELTA);
            value += 1.0f;
        }
    }

    @Test
    public void testDeserializeUint8() {
        // Create uint8 pixels manually
        int width = 5;
        int height = 5;
        int depth = 1;
        int channels = 1;

        // Create byte data
        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels);
        byte value = 0;
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.put(value);
            value++;
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("u1")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        // Deserialize
        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify data
        value = 0;
        for (Object obj : Views.flatIterable(deserialized)) {
            UnsignedByteType pixel = (UnsignedByteType) obj;
            Assert.assertEquals(value, pixel.get());
            value++;
        }
    }

    @Test
    public void testDeserializeUint8Alias() {
        // Test with "uint8" alias
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.put((byte) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("uint8")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeFloat32Alias() {
        // Test with "float32" alias
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Float.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putFloat((float) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("float32")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeUint16() {
        // Test with "u2" dtype (uint16)
        int width = 4;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Short.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putShort((short) (i * 1000));
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("u2")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify dimensions
        Assert.assertEquals(width, deserialized.dimension(0));
        Assert.assertEquals(height, deserialized.dimension(1));
        Assert.assertEquals(depth, deserialized.dimension(2));
        Assert.assertEquals(channels, deserialized.dimension(3));

        // Verify data
        int expected = 0;
        for (Object obj : Views.flatIterable(deserialized)) {
            net.imglib2.type.numeric.integer.UnsignedShortType pixel =
                (net.imglib2.type.numeric.integer.UnsignedShortType) obj;
            Assert.assertEquals(expected * 1000, pixel.get());
            expected++;
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeUnsupportedDimensionOrder() {
        ByteBuffer buffer = ByteBuffer.allocate(10 * 10 * Float.BYTES);
        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("INVALID")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(1)
                .setBindata(bindata)
                .build();

        Utils.DeserializeToInterval(pixels);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeUnsupportedDtype() {
        // Use a truly unsupported dtype (int64/i8 is not supported)
        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(1)
                .setDtype("i8")  // int64 - not supported
                .build();

        Utils.DeserializeToInterval(pixels);
    }

    @Test
    public void testSerialize5DImage() {
        // Create a 5D image (2x2x2x2x2) - now supported
        ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
        RandomAccessibleInterval<FloatType> image5D = factory.create(2, 2, 2, 2, 2);

        Pixels pixels = Utils.SerializeFromInterval(image5D);

        // Verify dimensions
        Assert.assertEquals(2, pixels.getSizeX());
        Assert.assertEquals(2, pixels.getSizeY());
        Assert.assertEquals(2, pixels.getSizeZ());
        Assert.assertEquals(2, pixels.getSizeC());
        Assert.assertEquals(2, pixels.getSizeT());
    }

    @Test
    public void testDeserializeWithBigEndianPrefix() {
        // Test dtype with ">" prefix (big-endian marker)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.put((byte) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype(">u1")  // Big-endian uint8
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testXYZTCDimensionOrder() {
        // Test deserialization with XYZTC dimension order
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Float.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putFloat((float) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZTC")
                .setBindata(bindata)
                .setDtype("f4")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testCXYZTDimensionOrder() {
        // Test deserialization with CXYZT dimension order (Python default)
        // In CXYZT: C varies fastest, then X, Y, Z
        int width = 4;
        int height = 3;
        int depth = 2;
        int channels = 2;

        // Create buffer in CXYZT order: C fastest, then X, Y, Z
        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Float.BYTES);
        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        buffer.putFloat((float) (z * 1000 + y * 100 + x * 10 + c));
                    }
                }
            }
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("CXYZT")
                .setBindata(bindata)
                .setDtype("f4")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify dimensions are correct (should be in XYZC order after deserialization)
        Assert.assertEquals(width, deserialized.dimension(0));
        Assert.assertEquals(height, deserialized.dimension(1));
        Assert.assertEquals(depth, deserialized.dimension(2));
        Assert.assertEquals(channels, deserialized.dimension(3));
    }

    @Test
    public void testSerializeWithCustomDimensionOrder() {
        // Test serialization with custom dimension order
        ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
        RandomAccessibleInterval<FloatType> image = factory.create(4, 3, 2, 2);

        float value = 0.0f;
        for (FloatType pixel : Views.flatIterable(image)) {
            pixel.set(value);
            value += 1.0f;
        }

        Pixels pixels = Utils.SerializeFromInterval(image, "CXYZT");

        Assert.assertEquals("CXYZT", pixels.getDimensionOrder());
        Assert.assertEquals(4, pixels.getSizeX());
        Assert.assertEquals(3, pixels.getSizeY());
        Assert.assertEquals(2, pixels.getSizeZ());
        Assert.assertEquals(2, pixels.getSizeC());
    }

    @Test
    public void testDeserializeWithPipePrefix() {
        // Test dtype with "|" prefix (numpy native byteorder marker)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.put((byte) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("|u1")  // Pipe prefix from numpy
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeWithLittleEndianPrefix() {
        // Test dtype with "<" prefix (little-endian marker)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Float.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putFloat((float) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.LITTLE)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("<f4")  // Little-endian prefix
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeWithEqualsPrefix() {
        // Test dtype with "=" prefix (numpy native byteorder)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Short.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putShort((short) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("=u2")  // Equals prefix from numpy
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeInt16() {
        // Test with "i2" dtype (int16)
        int width = 4;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Short.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putShort((short) (i * 100));
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("i2")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify dimensions
        Assert.assertEquals(width, deserialized.dimension(0));
        Assert.assertEquals(height, deserialized.dimension(1));
        Assert.assertEquals(depth, deserialized.dimension(2));
        Assert.assertEquals(channels, deserialized.dimension(3));

        // Verify data
        int expected = 0;
        for (Object obj : Views.flatIterable(deserialized)) {
            net.imglib2.type.numeric.integer.ShortType pixel =
                (net.imglib2.type.numeric.integer.ShortType) obj;
            Assert.assertEquals(expected * 100, pixel.get());
            expected++;
        }
    }

    @Test
    public void testDeserializeFloat64() {
        // Test with "f8" dtype (float64/double)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Double.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putDouble((double) i * 0.5);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("f8")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify data
        int expected = 0;
        for (Object obj : Views.flatIterable(deserialized)) {
            net.imglib2.type.numeric.real.DoubleType pixel =
                (net.imglib2.type.numeric.real.DoubleType) obj;
            Assert.assertEquals(expected * 0.5, pixel.get(), 0.0001);
            expected++;
        }
    }

    @Test
    public void testDeserializeInt8() {
        // Test with "i1" dtype (int8)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.put((byte) (i - 4));  // Some negative values
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("i1")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeInt32() {
        // Test with "i4" dtype (int32)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Integer.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putInt(i * 1000);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("i4")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeUint32() {
        // Test with "u4" dtype (uint32)
        int width = 3;
        int height = 3;
        int depth = 1;
        int channels = 1;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Integer.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putInt(i * 1000000);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("u4")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);
        Assert.assertNotNull(deserialized);
    }

    @Test
    public void testDeserializeWithOutputDimOrderZYXC() {
        // Test deserialization with custom outputDimOrder
        int width = 4;
        int height = 3;
        int depth = 2;
        int channels = 2;

        ByteBuffer buffer = ByteBuffer.allocate(width * height * depth * channels * Float.BYTES);
        for (int i = 0; i < width * height * depth * channels; i++) {
            buffer.putFloat((float) i);
        }
        buffer.flip();

        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .setEndianness(BinData.Endianness.BIG)
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setBindata(bindata)
                .setDtype("f4")
                .setSizeX(width)
                .setSizeY(height)
                .setSizeZ(depth)
                .setSizeC(channels)
                .build();

        // Default XYZC order
        RandomAccessibleInterval<?> defaultOrder = Utils.DeserializeToInterval(pixels);
        Assert.assertEquals(width, defaultOrder.dimension(0));  // X
        Assert.assertEquals(height, defaultOrder.dimension(1)); // Y
        Assert.assertEquals(depth, defaultOrder.dimension(2));  // Z
        Assert.assertEquals(channels, defaultOrder.dimension(3)); // C

        // Custom ZYXC order
        RandomAccessibleInterval<?> zyxcOrder = Utils.DeserializeToInterval(pixels, "ZYXC");
        Assert.assertEquals(depth, zyxcOrder.dimension(0));     // Z
        Assert.assertEquals(height, zyxcOrder.dimension(1));    // Y
        Assert.assertEquals(width, zyxcOrder.dimension(2));     // X
        Assert.assertEquals(channels, zyxcOrder.dimension(3));  // C

        // Custom CZYX order
        RandomAccessibleInterval<?> czyxOrder = Utils.DeserializeToInterval(pixels, "CZYX");
        Assert.assertEquals(channels, czyxOrder.dimension(0));  // C
        Assert.assertEquals(depth, czyxOrder.dimension(1));     // Z
        Assert.assertEquals(height, czyxOrder.dimension(2));    // Y
        Assert.assertEquals(width, czyxOrder.dimension(3));     // X
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeInvalidOutputDimOrder() {
        ByteBuffer buffer = ByteBuffer.allocate(10 * 10 * Float.BYTES);
        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(1)
                .setDtype("f4")
                .setBindata(bindata)
                .build();

        Utils.DeserializeToInterval(pixels, "INVALID");
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeOutputDimOrderMissingAxis() {
        // Test that excluding a non-singleton dimension fails
        ByteBuffer buffer = ByteBuffer.allocate(10 * 10 * 3 * Float.BYTES);
        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(3)  // Non-singleton C
                .setDtype("f4")
                .setBindata(bindata)
                .build();

        Utils.DeserializeToInterval(pixels, "XYZ");  // Missing C, C is not singleton
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeOutputDimOrderDuplicateAxis() {
        ByteBuffer buffer = ByteBuffer.allocate(10 * 10 * Float.BYTES);
        BinData bindata = BinData.newBuilder()
                .setData(com.google.protobuf.ByteString.copyFrom(buffer.array()))
                .build();

        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(1)
                .setDtype("f4")
                .setBindata(bindata)
                .build();

        Utils.DeserializeToInterval(pixels, "XYZCC");  // Duplicate C
    }

    @Test
    public void testSerializeUint32LargeValues() {
        // Test serialization of uint32 values beyond float precision threshold (>2^24)
        // This tests the fix for getRealFloat() clipping bug
        int width = 3;
        int height = 3;

        ArrayImgFactory<UnsignedIntType> factory = new ArrayImgFactory<>(new UnsignedIntType());
        RandomAccessibleInterval<UnsignedIntType> image = factory.create(width, height);

        // Use large values beyond float's 24-bit integer precision
        long[] testValues = {
            16_777_217L,      // Just beyond float precision threshold
            300_000_000L,     // Large but representable
            4_000_000_000L    // Near max uint32
        };

        int idx = 0;
        for (UnsignedIntType pixel : Views.flatIterable(image)) {
            pixel.set(testValues[idx % testValues.length]);
            idx++;
        }

        // Serialize
        Pixels pixels = Utils.SerializeFromInterval(image);
        Assert.assertEquals("u4", pixels.getDtype());

        // Deserialize
        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify values are preserved exactly (round-trip test)
        idx = 0;
        for (Object obj : Views.flatIterable(deserialized)) {
            UnsignedIntType pixel = (UnsignedIntType) obj;
            Assert.assertEquals(testValues[idx % testValues.length], pixel.get());
            idx++;
        }
    }

    @Test
    public void testSerializeInt32LargeValues() {
        // Test serialization of int32 values beyond float precision threshold
        int width = 3;
        int height = 3;

        ArrayImgFactory<IntType> factory = new ArrayImgFactory<>(new IntType());
        RandomAccessibleInterval<IntType> image = factory.create(width, height);

        // Use large values beyond float's 24-bit integer precision
        int[] testValues = {
            16_777_217,       // Just beyond float precision threshold
            2_000_000_000,    // Large positive
            -2_000_000_000    // Large negative
        };

        int idx = 0;
        for (IntType pixel : Views.flatIterable(image)) {
            pixel.set(testValues[idx % testValues.length]);
            idx++;
        }

        // Serialize
        Pixels pixels = Utils.SerializeFromInterval(image);
        Assert.assertEquals("i4", pixels.getDtype());

        // Deserialize
        RandomAccessibleInterval<?> deserialized = Utils.DeserializeToInterval(pixels);

        // Verify values are preserved exactly (round-trip test)
        idx = 0;
        for (Object obj : Views.flatIterable(deserialized)) {
            IntType pixel = (IntType) obj;
            Assert.assertEquals(testValues[idx % testValues.length], pixel.get());
            idx++;
        }
    }
}