package biopb.tensor;

import java.util.List;

import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.view.Views;

/**
 * Crop a returned tensor back to the client's originally requested region.
 *
 * <p>The tensor server snaps a slice_hint outward to chunk-aligned (lcm)
 * boundaries, so the array it returns can be larger than what was asked for; the
 * descriptor carries the realized slice so the client can trim the overhang.
 * This scale-aware arithmetic was forked three ways across the two Flight
 * clients -- {@code SerializableTensorImg.reconstructDelegate},
 * {@code TensorFlightClient.getTensor}, and
 * {@code TensorFlightClient.materializeSerializedArray} (biopb/biopb#277 item D).
 * This is its single home.
 */
final class RegionCrop {

    private RegionCrop() {}

    /**
     * Return {@code rai} cropped to the {@code requested} region, given the
     * {@code realized} region the server actually returned and the per-axis
     * {@code scaleHint} (pyramid downsampling factors; missing/short entries mean 1).
     *
     * <p>{@code rai} must be zero-min with as many dimensions as the hints. If the
     * requested region already spans the whole array, {@code rai} is returned
     * unwrapped rather than as an equivalent full-extent view.
     */
    static <T> RandomAccessibleInterval<T> cropToRequest(
            RandomAccessibleInterval<T> rai,
            SliceHint requested,
            SliceHint realized,
            List<Long> scaleHint) {

        int ndim = rai.numDimensions();
        long[] cropMin = new long[ndim];
        long[] cropMax = new long[ndim];
        boolean needsCrop = false;
        for (int ax = 0; ax < ndim; ax++) {
            long reqStart = requested.getStart(ax);
            long reqStop = requested.getStop(ax);
            long retStart = realized.getStart(ax);
            long scale = scaleHint.size() > ax ? scaleHint.get(ax) : 1L;
            if (scale > 1L) {
                cropMin[ax] = (reqStart - retStart) / scale;
                cropMax[ax] = (reqStop - retStart + scale - 1L) / scale - 1L;
            } else {
                cropMin[ax] = reqStart - retStart;
                cropMax[ax] = reqStop - retStart - 1L;
            }
            if (cropMin[ax] != 0 || cropMax[ax] != rai.max(ax)) {
                needsCrop = true;
            }
        }
        if (!needsCrop) {
            return rai;
        }
        return Views.zeroMin(Views.interval(rai, new FinalInterval(cropMin, cropMax)));
    }
}
