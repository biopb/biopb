package biopb.tensor;

import java.util.Arrays;
import java.util.Collections;

import org.junit.Assert;
import org.junit.Test;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Unit tests for {@link RegionCrop#cropToRequest}, the slice-hint crop
 * arithmetic that was forked three ways across the two Flight clients
 * (biopb/biopb#277 item D). Extraction lets the scale-aware trimming and the
 * no-op fast path be tested directly instead of only through a live server.
 */
public class RegionCropTest {

    /** A w x h image whose pixel (x, y) holds x*10 + y, for identity checks. */
    private static ArrayImg<FloatType, ?> ramp(long w, long h) {
        ArrayImg<FloatType, ?> img = new ArrayImgFactory<>(new FloatType()).create(w, h);
        RandomAccess<FloatType> ra = img.randomAccess();
        for (long x = 0; x < w; x++) {
            for (long y = 0; y < h; y++) {
                ra.setPosition(new long[] {x, y});
                ra.get().set(x * 10 + y);
            }
        }
        return img;
    }

    private static float at(RandomAccessibleInterval<FloatType> rai, long x, long y) {
        RandomAccess<FloatType> ra = rai.randomAccess();
        ra.setPosition(new long[] {x, y});
        return ra.get().get();
    }

    private static SliceHint slice(long sx, long sy, long ex, long ey) {
        return SliceHint.newBuilder()
                .addStart(sx).addStart(sy)
                .addStop(ex).addStop(ey)
                .build();
    }

    @Test
    public void full_request_returns_the_same_instance_unwrapped() {
        ArrayImg<FloatType, ?> img = ramp(4, 4);
        RandomAccessibleInterval<FloatType> result = RegionCrop.cropToRequest(
                img, slice(0, 0, 4, 4), slice(0, 0, 4, 4), Collections.emptyList());
        Assert.assertSame("no-op crop must not wrap the array", img, result);
    }

    @Test
    public void scale_one_trims_the_snapped_overhang() {
        // Server realized [0,4) on each axis but the client asked for [1,3);
        // trim to the 2x2 interior and re-zero the origin.
        ArrayImg<FloatType, ?> img = ramp(4, 4);
        RandomAccessibleInterval<FloatType> result = RegionCrop.cropToRequest(
                img, slice(1, 1, 3, 3), slice(0, 0, 4, 4), Collections.emptyList());

        Assert.assertEquals(2, result.dimension(0));
        Assert.assertEquals(2, result.dimension(1));
        Assert.assertEquals(0, result.min(0)); // zero-min
        Assert.assertEquals(11.0f, at(result, 0, 0), 0.0f); // was img(1,1)
        Assert.assertEquals(21.0f, at(result, 1, 0), 0.0f); // was img(2,1)
        Assert.assertEquals(22.0f, at(result, 1, 1), 0.0f); // was img(2,2)
    }

    @Test
    public void nonzero_realized_origin_is_subtracted() {
        // Realized region starts at global 2; a request for [3,5) maps to local [1,3).
        ArrayImg<FloatType, ?> img = ramp(4, 4);
        RandomAccessibleInterval<FloatType> result = RegionCrop.cropToRequest(
                img, slice(3, 3, 5, 5), slice(2, 2, 6, 6), Collections.emptyList());

        Assert.assertEquals(2, result.dimension(0));
        Assert.assertEquals(11.0f, at(result, 0, 0), 0.0f); // was img(1,1)
        Assert.assertEquals(22.0f, at(result, 1, 1), 0.0f); // was img(2,2)
    }

    @Test
    public void scale_greater_than_one_divides_into_the_downsampled_grid() {
        // Pyramid level with scale 2: request [4,8) in full-res coords maps to
        // local [2,4) on the level-2 array (realized origin 0).
        ArrayImg<FloatType, ?> img = ramp(5, 5);
        RandomAccessibleInterval<FloatType> result = RegionCrop.cropToRequest(
                img, slice(4, 4, 8, 8), slice(0, 0, 10, 10), Arrays.asList(2L, 2L));

        Assert.assertEquals(2, result.dimension(0));
        Assert.assertEquals(2, result.dimension(1));
        Assert.assertEquals(22.0f, at(result, 0, 0), 0.0f); // was img(2,2)
        Assert.assertEquals(33.0f, at(result, 1, 1), 0.0f); // was img(3,3)
    }

    @Test
    public void missing_scale_hint_entries_default_to_one() {
        // Empty scaleHint must behave exactly like scale 1 (no index-out-of-bounds).
        ArrayImg<FloatType, ?> img = ramp(4, 4);
        RandomAccessibleInterval<FloatType> result = RegionCrop.cropToRequest(
                img, slice(1, 1, 3, 3), slice(0, 0, 4, 4), Collections.emptyList());
        Assert.assertEquals(11.0f, at(result, 0, 0), 0.0f);
    }
}
