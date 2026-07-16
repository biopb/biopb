"""Unit tests for the rolling-ball background plugin (biopb_mcp.plugins.rolling_ball).

Pins the ImageJ-faithful behavior that matters to a user: a smooth background is
recovered under bright features (dark features on a bright field with
``light_background``), features survive subtraction, the input dtype is preserved
with ImageJ's integer offset/clip, N-D arrays are processed plane-by-plane, and
the radius→(shrink, ball) buckets match ImageJ. Also checks the plugin is
registered as a ``biopb_mcp.namespace`` entry point so the kernel bootstrap loads
it, and that the static inspector sees it. No kernel/display needed.
"""

from importlib.metadata import entry_points

import numpy as np
import pytest
from skimage.draw import disk

from biopb_mcp.plugins import rolling_ball as rb

RADIUS = 25


def _ramp_with_blobs(shape=(200, 200), centers=((60, 60), (150, 140)), amp=120.0):
    """A smooth linear-ramp background plus bright disk features."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    bg = (30.0 + 0.12 * xx + 0.08 * yy).astype(np.float32)
    img = bg.copy()
    for cy, cx in centers:
        r, c = disk((cy, cx), 7, shape=shape)
        img[r, c] += amp
    return img, bg, centers


class TestBallConstruction:
    @pytest.mark.parametrize(
        "radius,shrink",
        [(5, 1), (10, 1), (11, 2), (30, 2), (31, 4), (100, 4), (101, 8)],
    )
    def test_shrink_buckets_match_imagej(self, radius, shrink):
        ball, s = rb._build_ball(radius)
        assert s == shrink
        # Square, odd-sized patch; center is the tallest point of the hemisphere.
        assert ball.ndim == 2 and ball.shape[0] == ball.shape[1]
        assert ball.shape[0] % 2 == 1
        h = ball.shape[0] // 2
        assert ball[h, h] == ball.max() > 0  # center is the top of the hemisphere
        assert (ball >= 0).all()  # heights, 0 outside the sphere cap


class TestBackgroundEstimate:
    def test_recovers_smooth_background_under_features(self):
        img, bg_true, centers = _ramp_with_blobs()
        est = rb.rolling_ball_background(img.astype(np.uint16), radius=RADIUS)
        assert est.dtype == np.float32
        # In flat regions (away from the blobs) the estimate tracks the true ramp.
        flat = np.ones(img.shape, bool)
        for cy, cx in centers:
            r, c = disk((cy, cx), 14, shape=img.shape)
            flat[r, c] = False
        err = np.abs(est[flat] - bg_true[flat])
        assert err.mean() < 2.0 and err.max() < 6.0

    def test_subtract_flattens_background_keeps_features(self):
        img, _bg, centers = _ramp_with_blobs()
        sub = rb.subtract_background(img.astype(np.uint16), radius=RADIUS)
        assert sub.dtype == np.uint16
        flat = np.ones(img.shape, bool)
        for cy, cx in centers:
            r, c = disk((cy, cx), 14, shape=img.shape)
            flat[r, c] = False
        assert sub[flat].mean() < 2.0  # background essentially removed
        for cy, cx in centers:  # feature peaks survive close to their true amplitude
            assert sub[cy, cx] > 100

    def test_return_background_matches_helper(self):
        img, _bg, _c = _ramp_with_blobs()
        u16 = img.astype(np.uint16)
        a = rb.subtract_background(u16, radius=RADIUS, return_background=True)
        b = rb.rolling_ball_background(u16, radius=RADIUS)
        assert np.array_equal(a, b)


class TestLightBackground:
    def test_dark_features_on_bright_field(self):
        yy, xx = np.mgrid[0:128, 0:128]
        field = (240.0 - 0.05 * xx).astype(np.float32)
        img = field.copy()
        r, c = disk((64, 64), 6, shape=img.shape)
        img[r, c] -= 150.0  # a dark spot
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        sub = rb.subtract_background(u8, radius=RADIUS, light_background=True)
        assert sub.dtype == np.uint8
        # The bright field is retained near white; the dark feature stays dark.
        assert int(np.median(sub)) >= 250
        assert sub[64, 64] < 200


class TestDtypeAndShape:
    def test_float_input_returns_raw_difference(self):
        img, _bg, _c = _ramp_with_blobs()
        f = img.astype(np.float32)
        sub = rb.subtract_background(f, radius=RADIUS)
        bg = rb.rolling_ball_background(f, radius=RADIUS)
        assert sub.dtype == np.float32
        assert np.allclose(sub, f - bg)  # no offset/clip for float

    def test_ndim_applied_per_plane(self):
        img, _bg, _c = _ramp_with_blobs(shape=(96, 96))
        vol = np.stack([img, img * 0 + 40.0], axis=0).astype(np.uint16)
        out = rb.subtract_background(vol, radius=RADIUS)
        assert out.shape == vol.shape and out.dtype == np.uint16
        # The flat second plane subtracts to ~0 everywhere.
        assert out[1].max() <= 1

    def test_rejects_below_2d(self):
        with pytest.raises(ValueError):
            rb.subtract_background(np.arange(10.0), radius=RADIUS)


class TestPluginRegistration:
    def test_registered_as_namespace_entry_point(self):
        eps = entry_points(group="biopb_mcp.namespace")
        by_name = {e.name: e for e in eps}
        assert "rolling-ball" in by_name
        assert by_name["rolling-ball"].value == "biopb_mcp.plugins.rolling_ball"
        # Loading the entry point yields the module; its __all__ is the surface
        # the bootstrap merges into the kernel namespace.
        mod = by_name["rolling-ball"].load()
        assert set(mod.__all__) == {"subtract_background", "rolling_ball_background"}

    def test_static_inspector_lists_it(self):
        from biopb import _kernel_plugins

        names = [row["name"] for row in _kernel_plugins.entry_point_plugins()]
        assert "rolling-ball" in names
