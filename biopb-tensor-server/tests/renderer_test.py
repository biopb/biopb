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
from biopb_tensor_server.serving.renderer import (
    extract_yx_slice,
    render_array_to_image_bytes,
    samples_axis,
)

RGB_LABELS = ["T", "C", "Z", "Y", "X", "S"]


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
    """Defensive: attacker/adapter-supplied labels must never crash the render.

    Degenerate label sets can map the samples axis onto Y/X, leave Y/X with no
    distinct axis, or under-rank the array -- all of which previously produced a
    repeated / negative transpose axis (ValueError / KeyError).
    """

    @pytest.mark.parametrize(
        "labels,shape",
        [
            (["C", "Y", "S"], (3, 8, 10)),  # S claimed as X by fallback + samples
            (["C", "Y"], (3, 8)),  # X falls back onto Y's axis
            (["Y", "S"], (8, 3)),  # samples but too few spatial axes
            (["S"], (3,)),  # 1-D
            (["X"], ()),  # 0-D
            ([], (4, 5)),  # no labels
            (["Q", "W"], (6, 7)),  # all-unknown labels
            (["X", "Y"], (6, 7)),  # swapped order
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
