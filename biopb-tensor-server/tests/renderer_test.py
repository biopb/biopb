"""Tests for the webapp render path, focused on interleaved RGB(A) samples.

The server render (`ws/render` / `/api/render`) reduces an N-D tensor to a 2-D
Y/X plane and pseudo-colors it -- built for selecting one T/Z/C plane at a time.
An RGB OME-TIFF arrives as a 6-D ``TCZYXS`` block whose trailing ``S`` axis holds
the color components of a single pixel, which must be composited into RGB, not
collapsed to one plane. Before the fix ``extract_yx_slice`` collapsed the ``S``
axis by indexing the Y axis, rendering a 512x3 strip instead of the image.
"""

import numpy as np
import pytest
from biopb_tensor_server.core.axes import plane_axes, samples_axis
from biopb_tensor_server.serving.renderer import (
    clamp_gamma,
    extract_yx_slice,
    render_array_to_image_bytes,
)

RGB_LABELS = ["T", "C", "Z", "Y", "X", "S"]


class TestPlaneAxes:
    """The render plane is read off the canonical wire order (biopb/biopb#596):
    Y and X are the last two axes, behind a trailing samples axis."""

    def test_trailing_yx(self):
        assert plane_axes(["T", "C", "Z", "Y", "X"], (1, 3, 1, 8, 10)) == (3, 4, None)

    def test_samples_pushes_yx_forward(self):
        assert plane_axes(RGB_LABELS, (1, 1, 1, 8, 10, 3)) == (3, 4, 5)

    def test_2d(self):
        assert plane_axes(["Y", "X"], (8, 10)) == (0, 1, None)

    def test_samples_only_where_the_canonical_order_puts_it(self):
        # An S that is not last is not an order this server serves, so it is
        # ignored rather than hunted down -- it reduces to index 0 like any
        # other leading axis.
        assert plane_axes(["S", "Y", "X"], (3, 8, 10)) == (1, 2, None)

    def test_size_gate(self):
        # A trailing "S" that is not 3 or 4 wide is not interleaved colour.
        assert plane_axes(RGB_LABELS, (1, 1, 1, 8, 10, 5)) == (4, 5, None)

    def test_samples_needs_room_for_y_and_x(self):
        assert plane_axes(["Y", "S"], (8, 3)) == (0, 1, None)

    @pytest.mark.parametrize(
        "labels,shape",
        [
            (["Y", "Y", "Y"], (4, 5, 6)),  # the same axis claimed three times
            (["X", "Y"], (4, 5)),  # an order the server does not serve
            ([], (4, 5)),  # no labels at all
            (["Q", "W"], (4, 5)),  # all-unknown labels
            (["Y", "Y", "S"], (4, 5, 3)),  # samples beside a duplicated Y
            (["S", "S", "S"], (3, 3, 3)),  # samples claimed everywhere
        ],
    )
    def test_degenerate_labels_cannot_collide(self, labels, shape):
        # The property that replaced the old collision handling: the indices come
        # from the rank, so no label set can repeat an axis or run out of range.
        y, x, s = plane_axes(labels, shape)
        assert y != x
        assert {y, x} <= set(range(len(shape)))
        if s is not None:
            assert s not in (y, x)
            assert s in range(len(shape))


class TestSamplesAxis:
    def test_detects_rgb_samples_axis(self):
        assert samples_axis(RGB_LABELS, (1, 1, 1, 8, 8, 3)) == 5

    def test_detects_rgba_samples_axis(self):
        assert samples_axis(RGB_LABELS, (1, 1, 1, 8, 8, 4)) == 5

    def test_case_insensitive_label(self):
        assert samples_axis(["y", "x", "s"], (8, 8, 3)) == 2

    def test_size_gated_no_false_positive_on_channel(self):
        # A 3-channel fluorescence stack is C, not samples -- no S label.
        assert samples_axis(["T", "C", "Z", "Y", "X"], (1, 3, 1, 8, 8)) is None

    def test_s_label_wrong_size_is_not_samples(self):
        # An "S" axis that is not 3 or 4 wide is not interleaved RGB(A).
        assert samples_axis(RGB_LABELS, (1, 1, 1, 8, 8, 5)) is None


class TestExtractYxSlice:
    def test_rgb_keeps_samples_axis(self):
        arr = np.zeros((1, 1, 1, 12, 10, 3), np.uint8)
        assert extract_yx_slice(arr, RGB_LABELS).shape == (12, 10, 3)

    def test_rgba_keeps_samples_axis(self):
        arr = np.zeros((1, 1, 1, 12, 10, 4), np.uint8)
        assert extract_yx_slice(arr, RGB_LABELS).shape == (12, 10, 4)

    def test_grayscale_5d_reduces_to_plane(self):
        arr = np.zeros((1, 1, 1, 12, 10), np.uint8)
        assert extract_yx_slice(arr, ["T", "C", "Z", "Y", "X"]).shape == (12, 10)

    def test_2d_passthrough(self):
        arr = np.zeros((12, 10), np.uint8)
        assert extract_yx_slice(arr, ["Y", "X"]).shape == (12, 10)

    def test_leading_axes_reduced_to_first_index(self):
        # A multi-Z stack must be reduced to a single plane (index 0), not a strip.
        arr = np.zeros((1, 1, 16, 12, 10), np.uint8)
        arr[0, 0, 0] = 7  # plane 0
        arr[0, 0, 5] = 99  # a later plane
        out = extract_yx_slice(arr, ["T", "C", "Z", "Y", "X"])
        assert out.shape == (12, 10)
        assert out[0, 0] == 7  # took plane 0

    def test_rgb_orders_yx_before_samples(self):
        # Distinct sizes so a transpose bug would change the shape.
        arr = np.zeros((1, 1, 1, 12, 10, 3), np.uint8)
        assert extract_yx_slice(arr, RGB_LABELS).shape == (12, 10, 3)


class TestExtractYxSliceMalformed:
    """Defensive: adapter-supplied labels must never crash the render.

    These used to need explicit collision handling -- a degenerate set could map
    the samples axis onto Y/X or leave Y/X with no distinct axis, producing a
    repeated / negative transpose axis (ValueError / KeyError). Resolving the
    plane from the rank instead makes that unrepresentable, so these now assert a
    property that holds by construction rather than one defended case by case.
    """

    @pytest.mark.parametrize(
        "labels,shape",
        [
            (["C", "Y", "S"], (3, 8, 10)),  # trailing S, but not 3/4 wide
            (["C", "Y"], (3, 8)),  # no X label at all
            (["Y", "S"], (8, 3)),  # samples but too few spatial axes
            (["S"], (3,)),  # 1-D
            (["X"], ()),  # 0-D
            ([], (4, 5)),  # no labels
            (["Q", "W"], (6, 7)),  # all-unknown labels
            (["X", "Y"], (6, 7)),  # an order the server does not serve
        ],
    )
    def test_no_crash_and_valid_plane(self, labels, shape):
        arr = np.arange(max(1, int(np.prod(shape))), dtype=np.uint8).reshape(shape)
        out = extract_yx_slice(arr, labels)
        assert out.ndim in (2, 3)  # always a usable Y/X (+S) plane
        # And the full render path stays intact.
        _, w, h, _, _ = render_array_to_image_bytes(
            arr=arr, dim_labels=labels, output_format="raw"
        )
        assert w >= 1 and h >= 1


class TestRenderRgb:
    def _gradient_rgb(self, h=64, w=80):
        arr = np.zeros((1, 1, 1, h, w, 3), np.uint8)
        arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]  # R along X
        arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]  # G along Y
        arr[..., 2] = 128  # B constant
        return arr

    def test_rgb_renders_full_plane_not_strip(self):
        arr = self._gradient_rgb()
        img, w, h, _, _ = render_array_to_image_bytes(
            arr=arr, dim_labels=RGB_LABELS, output_format="raw"
        )
        assert (w, h) == (80, 64)
        assert len(img) == 80 * 64 * 3

    def test_rgb_preserves_true_color(self):
        arr = self._gradient_rgb()
        img, w, h, _, _ = render_array_to_image_bytes(
            arr=arr,
            dim_labels=RGB_LABELS,
            output_format="raw",
            percentile_lo=0,
            percentile_hi=100,
        )
        out = np.frombuffer(img, np.uint8).reshape(h, w, 3)
        assert tuple(out[0, 0]) == (0, 0, 128)
        assert tuple(out[0, -1]) == (255, 0, 128)  # R rises left->right
        assert tuple(out[-1, 0]) == (0, 255, 128)  # G rises top->bottom

    def test_rgba_drops_alpha(self):
        arr = np.zeros((1, 1, 1, 20, 24, 4), np.uint8)
        arr[..., :3] = 100
        arr[..., 3] = 255
        img, w, h, _, _ = render_array_to_image_bytes(
            arr=arr, dim_labels=RGB_LABELS, output_format="raw"
        )
        assert (w, h) == (24, 20)
        assert len(img) == 24 * 20 * 3  # RGB, alpha dropped

    def test_rgba_alpha_excluded_from_percentile_stretch(self):
        # An opaque alpha=255 must not inflate the high cutoff: the RGB stretch
        # must be identical whether or not an alpha sample is present. RGB values
        # span 0..200 (max well below the constant 255 alpha), so including alpha
        # in the stats would raise hi and darken the output.
        h, w = 32, 40
        rgb = np.zeros((1, 1, 1, h, w, 3), np.uint8)
        rgb[..., 0] = np.linspace(0, 200, w, dtype=np.uint8)[None, :]
        rgb[..., 1] = 80
        rgb[..., 2] = 40
        rgba = np.concatenate(
            [rgb, np.full((1, 1, 1, h, w, 1), 255, np.uint8)], axis=-1
        )

        out_rgb, _, _, lo_rgb, hi_rgb = render_array_to_image_bytes(
            arr=rgb, dim_labels=RGB_LABELS, output_format="raw"
        )
        out_rgba, _, _, lo_rgba, hi_rgba = render_array_to_image_bytes(
            arr=rgba, dim_labels=RGB_LABELS, output_format="raw"
        )
        assert (lo_rgba, hi_rgba) == (lo_rgb, hi_rgb)
        assert out_rgba == out_rgb

    def test_grayscale_still_pseudocolors(self):
        # Non-RGB single plane keeps the pseudo-color path (green here).
        arr = np.full((1, 1, 1, 16, 20), 128, np.uint8)
        img, w, h, _, _ = render_array_to_image_bytes(
            arr=arr,
            dim_labels=["T", "C", "Z", "Y", "X"],
            output_format="raw",
            color="green",
            percentile_lo=0,
            percentile_hi=100,
        )
        assert (w, h) == (20, 16)
        out = np.frombuffer(img, np.uint8).reshape(h, w, 3)
        # green multiplier -> R=0, B=0, G>0
        assert out[0, 0, 0] == 0 and out[0, 0, 2] == 0 and out[0, 0, 1] > 0


class TestGamma:
    """Gamma reshapes the ramp between the contrast limits, nothing else.

    The viewer offers it because a linear stretch buries dim structure in
    fluorescence data; what matters for correctness is that it moves the
    midtones without moving the endpoints, and that it lands in the same place
    as the browser's shader (exponent on the normalized intensity, before the
    color multiplier).
    """

    LABELS = ["T", "C", "Z", "Y", "X"]

    def _ramp(self):
        # A full 0..255 ramp, so lo/hi land on the endpoints and the only thing
        # gamma can be measured against is the curve in between.
        arr = np.zeros((1, 1, 1, 4, 256), np.uint8)
        arr[..., :] = np.arange(256, dtype=np.uint8)[None, :]
        return arr

    def _render(self, arr, gamma):
        img, w, h, _, _ = render_array_to_image_bytes(
            arr=arr,
            dim_labels=self.LABELS,
            output_format="raw",
            color="white",
            percentile_lo=0,
            percentile_hi=100,
            gamma=gamma,
        )
        return np.frombuffer(img, np.uint8).reshape(h, w, 3)

    def test_default_is_linear(self):
        arr = self._ramp()
        assert self._render(arr, 1.0).tobytes() == self._render(arr, 1.0).tobytes()
        out = self._render(arr, 1.0)
        assert out[0, 128, 0] == 128

    def test_below_one_lifts_the_midtones(self):
        out = self._render(self._ramp(), 0.5)
        # (128/255) ** 0.5 * 255 ~= 181
        assert out[0, 128, 0] == pytest.approx(181, abs=1)

    def test_above_one_pushes_them_down(self):
        out = self._render(self._ramp(), 2.0)
        # (128/255) ** 2 * 255 ~= 64
        assert out[0, 128, 0] == pytest.approx(64, abs=1)

    def test_endpoints_are_fixed(self):
        for gamma in (0.25, 0.5, 2.0, 4.0):
            out = self._render(self._ramp(), gamma)
            assert (out[0, 0, 0], out[0, -1, 0]) == (0, 255)

    def test_applies_before_the_color_multiplier(self):
        # Not after. Gamma on the final RGB would raise the multiplier too, so a
        # half-strength channel would change hue as the slider moved; the
        # browser's shader applies it to the intensity, and this must match.
        arr = np.zeros((1, 1, 1, 4, 3), np.uint8)
        arr[..., 1] = 128
        arr[..., 2] = 255
        img, w, h, _, _ = render_array_to_image_bytes(
            arr=arr,
            dim_labels=self.LABELS,
            output_format="raw",
            color="#804000",  # R at 128/255, G at 64/255, no B
            percentile_lo=0,
            percentile_hi=100,
            gamma=0.5,
        )
        out = np.frombuffer(img, np.uint8).reshape(h, w, 3)
        # sqrt(128/255) * 255 = 181 of intensity, then * (128/255) of red = 91.
        # Gamma applied after the multiplier would give sqrt(128*128/255/255)*255
        # = 128 -- a different, more saturated red.
        assert out[0, 1, 0] == pytest.approx(91, abs=1)

    def test_rgb_samples_keep_their_balance(self):
        # One shared curve across R, G and B: the ratio between samples at the
        # same pixel is what "true color" means here, and gamma must not be
        # applied to one sample and not another.
        h, w = 8, 8
        arr = np.zeros((1, 1, 1, h, w, 3), np.uint8)
        arr[..., 0] = 255
        arr[..., 1] = 64
        arr[..., 2] = 0
        img, ow, oh, _, _ = render_array_to_image_bytes(
            arr=arr,
            dim_labels=RGB_LABELS,
            output_format="raw",
            percentile_lo=0,
            percentile_hi=100,
            gamma=0.5,
        )
        out = np.frombuffer(img, np.uint8).reshape(oh, ow, 3)
        assert out[0, 0, 0] == 255  # the top of the shared stretch stays there
        assert out[0, 0, 2] == 0  # and so does the bottom
        # (64/255) ** 0.5 * 255 ~= 128: lifted, not left alone.
        assert out[0, 0, 1] == pytest.approx(128, abs=1)

    @pytest.mark.parametrize(
        "given,expected",
        [
            (0.0, 0.25),  # not dim -- a uniform white plane
            (-1.0, 0.25),
            (100.0, 4.0),
            (float("nan"), 1.0),
            (float("inf"), 1.0),
            (1.0, 1.0),
        ],
    )
    def test_out_of_range_gamma_is_pulled_back(self, given, expected):
        assert clamp_gamma(given) == expected

    def test_a_rejected_gamma_still_renders(self):
        # The clamp is on the render path, not only in the request model: a
        # gamma of 0 must not reach np.power and turn the plane white.
        out = self._render(self._ramp(), 0.0)
        assert out[0, 128, 0] < 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
