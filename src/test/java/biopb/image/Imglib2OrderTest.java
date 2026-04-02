package biopb.image;

import org.junit.Test;
import org.junit.Assert;

import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;
import net.imglib2.RandomAccessibleInterval;

public class Imglib2OrderTest {

    @Test
    public void testImglib2MemoryOrder() {
        // Create a 2x3x4 array and fill with sequential values
        // If dim 0 varies fastest: values go 0,1,2,3,4,5,6,... for positions (0,0,0),(1,0,0),(0,1,0),...
        // If dim 0 varies slowest: values go 0,1,2,3,4,5,6,... for positions (0,0,0),(0,0,1),(0,0,2),...
        ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
        RandomAccessibleInterval<UnsignedByteType> img = factory.create(2, 3, 4);

        // Fill with sequential values using flat iteration (memory order)
        int val = 0;
        for (UnsignedByteType p : Views.flatIterable(img)) {
            p.set(val++);
        }

        // Check specific positions to determine which dimension varies fastest
        // If dim 0 (size 2) varies fastest:
        //   Position (0,0,0) = 0, Position (1,0,0) = 1, Position (0,1,0) = 2, ...
        // If dim 2 (size 4) varies fastest:
        //   Position (0,0,0) = 0, Position (0,0,1) = 1, Position (0,0,2) = 2, ...

        System.out.println("Dimensions: dim0=" + img.dimension(0) + ", dim1=" + img.dimension(1) + ", dim2=" + img.dimension(2));
        System.out.println("Position (0,0,0): " + img.getAt(0, 0, 0).get());
        System.out.println("Position (1,0,0): " + img.getAt(1, 0, 0).get());
        System.out.println("Position (0,1,0): " + img.getAt(0, 1, 0).get());
        System.out.println("Position (0,0,1): " + img.getAt(0, 0, 1).get());

        // If dim 0 varies fastest:
        // - (0,0,0) should be 0
        // - (1,0,0) should be 1  (dim 0 increments first)
        // - (0,1,0) should be 2  (dim 0 wrapped, dim 1 increments)
        // - (0,0,1) should be 6  (dims 0 and 1 wrapped, dim 2 increments)

        // If dim 2 varies fastest (C-order):
        // - (0,0,0) should be 0
        // - (1,0,0) should be 12 (dims 1 and 2 wrapped)
        // - (0,1,0) should be 4  (dim 2 wrapped)
        // - (0,0,1) should be 1  (dim 2 increments first)

        // Test: if dim 0 varies fastest
        Assert.assertEquals("dim 0 fastest: (0,0,0)", 0, img.getAt(0, 0, 0).get());
        Assert.assertEquals("dim 0 fastest: (1,0,0)", 1, img.getAt(1, 0, 0).get());
        Assert.assertEquals("dim 0 fastest: (0,1,0)", 2, img.getAt(0, 1, 0).get());
        Assert.assertEquals("dim 0 fastest: (0,0,1)", 6, img.getAt(0, 0, 1).get());
    }
}