package biopb.image;

import java.nio.ByteBuffer;

import org.junit.Test;
import org.junit.Assert;

import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
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

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeUnsupportedDimensionOrder() {
        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("INVALID")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(1)
                .build();

        Utils.DeserializeToInterval(pixels);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeserializeUnsupportedDtype() {
        Pixels pixels = Pixels.newBuilder()
                .setDimensionOrder("XYZCT")
                .setSizeX(10)
                .setSizeY(10)
                .setSizeZ(1)
                .setSizeC(1)
                .setDtype("int16")
                .build();

        Utils.DeserializeToInterval(pixels);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSerializeUnsupportedDimensions() {
        // Create a 5D image which is not supported
        ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
        RandomAccessibleInterval<FloatType> image5D = factory.create(2, 2, 2, 2, 2);

        Utils.SerializeFromInterval(image5D);
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
}