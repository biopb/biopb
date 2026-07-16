"""Rolling-ball background subtraction (Sternberg 1983), the fast ImageJ port.

A faithful Python port of ImageJ/Fiji's ``BackgroundSubtracter`` rolling-ball
algorithm. ``skimage.restoration.rolling_ball`` computes the same morphological
result but rolls a full-radius ball over the full-resolution image
(``O(n·radius²)``); ImageJ — and this port — instead **shrink** the image, roll a
proportionally small ball on it, and **bilinearly enlarge** the background back,
cutting the work by the shrink factor to the fourth power (``~256×`` at radius 50).
That is the whole reason ImageJ is dramatically faster, and this reproduces it.

Exposed in the agent kernel namespace (loaded as a ``biopb_mcp.namespace`` plugin):

- ``subtract_background(image, radius=50, ...)`` — the background-subtracted image.
- ``rolling_ball_background(image, radius=50, ...)`` — just the estimated background.

Both accept a 2-D plane or an N-D array (applied plane-by-plane over the last two
axes), preserve the input dtype (integer results are offset+clipped exactly as
ImageJ does), and take ``light_background`` (dark features on a bright field) and
``do_presmooth`` (a 3×3 mean before rolling, on by default, to resist noise).

Implementation mirrors ImageJ method-for-method: ``_build_ball`` reproduces the
radius→(shrink, arc-trim) buckets and the trimmed-hemisphere patch; ``rollBall``
is a grayscale opening (erode-then-dilate by that non-flat structuring element,
via :mod:`scipy.ndimage`); shrink is a block ``min``; enlarge uses ImageJ's exact
half-pixel interpolation index/weight arrays. Border handling of the morphology
is scipy's edge-replicate, a minor rim approximation vs ImageJ's roll-past-edge —
immaterial to the interior background estimate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion, uniform_filter
from skimage.measure import block_reduce

__all__ = ["subtract_background", "rolling_ball_background"]

# ImageJ default; the radius should exceed the largest foreground feature so the
# ball rolls under it rather than into it.
DEFAULT_RADIUS = 50.0


def _rolling_ball_params(radius: float) -> tuple[int, int]:
    """ImageJ's radius→(shrinkFactor, arcTrimPercent) buckets (RollingBall ctor)."""
    if radius <= 10:
        return 1, 24
    if radius <= 30:
        return 2, 24
    if radius <= 100:
        return 4, 32
    return 8, 40


def _build_ball(radius: float) -> tuple[np.ndarray, int]:
    """Build the rolling-ball patch and its shrink factor (ImageJ buildRollingBall).

    Returns a square float32 array of hemisphere heights (0 outside the sphere
    cap) and the shrink factor the image must be reduced by before rolling. The
    ball is downscaled with the image, so its radius is ``radius/shrinkFactor``.
    """
    shrink, arc_trim = _rolling_ball_params(radius)
    small_r = radius / shrink
    if small_r < 1:
        small_r = 1.0
    rsquare = small_r * small_r
    # (int)(arcTrimPer*smallballradius)/100 — Java truncation then integer divide.
    xtrim = int(arc_trim * small_r) // 100
    # Java Math.round = floor(x + 0.5), NOT numpy's banker's rounding.
    half = int(math.floor(small_r - xtrim + 0.5))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    temp = rsquare - xx * xx - yy * yy
    ball = np.where(temp > 0.0, np.sqrt(np.maximum(temp, 0.0)), 0.0)
    return ball.astype(np.float32), shrink


def _roll_ball(image: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """Roll the ball under the surface: grayscale opening by the ball (ImageJ rollBall).

    ImageJ finds, per ball position, the highest the ball rises while staying below
    the surface (``min`` of surface−ball over the footprint) then paints the ball
    surface (``max`` of that height+ball) — exactly an erosion followed by a
    dilation with the non-flat ``ball`` structuring element. ``mode='nearest'``
    replicates the edge (a small rim approximation vs ImageJ's roll-past-border).
    """
    eroded = grey_erosion(image, structure=ball, mode="nearest")
    return grey_dilation(eroded, structure=ball, mode="nearest")


def _shrink(image: np.ndarray, shrink: int) -> np.ndarray:
    """Down-sample by taking the block minimum (ImageJ shrinkImage).

    Output dims are ``ceil(dim/shrink)``; a padded partial block is filled with
    ``+inf`` so ``min`` ignores it (matching ImageJ's "min over available pixels").
    """
    small = block_reduce(image, (shrink, shrink), func=np.min, cval=np.float32(np.inf))
    return small.astype(np.float32)


def _interp_arrays(
    length: int, s_length: int, shrink: int
) -> tuple[np.ndarray, np.ndarray]:
    """ImageJ makeInterpolationArrays: left-neighbor index + weight for enlarging."""
    i = np.arange(length)
    # (i - shrink/2)/shrink with Java integer (toward-zero) division.
    idx = np.trunc((i - shrink // 2) / shrink).astype(np.intp)
    # ImageJ clamps only the upper edge to s_length-2 (so idx+1 stays in range);
    # the lower clamp to 0 is a no-op for the values Java produces but guards the
    # array access explicitly.
    idx = np.clip(idx, 0, s_length - 2)
    distance = (i + 0.5) / shrink - (idx + 0.5)
    weights = (1.0 - distance).astype(np.float32)
    return idx, weights


def _enlarge(small: np.ndarray, height: int, width: int, shrink: int) -> np.ndarray:
    """Bilinearly enlarge the shrunken background to full size (ImageJ enlargeImage)."""
    s_height, s_width = small.shape
    x_idx, x_w = _interp_arrays(width, s_width, shrink)
    y_idx, y_w = _interp_arrays(height, s_height, shrink)
    # x-interpolate every small row, then y-interpolate the resulting lines.
    lines = small[:, x_idx] * x_w + small[:, x_idx + 1] * (
        1.0 - x_w
    )  # (s_height, width)
    out = lines[y_idx] * y_w[:, None] + lines[y_idx + 1] * (1.0 - y_w)[:, None]
    return out.astype(np.float32)


def _background_2d(
    plane: np.ndarray, radius: float, invert: bool, do_presmooth: bool
) -> np.ndarray:
    """Estimate the background of one 2-D float plane (ImageJ rollingBallFloatBackground).

    Returns the background in the plane's own value space (the ``invert`` for a
    light background is applied and undone internally).
    """
    fp = plane.astype(np.float32, copy=True)
    if invert:
        fp = -fp
    if do_presmooth:
        # Separable 3x3 mean, edge-replicated — matches ImageJ filter3x3(MEAN).
        fp = uniform_filter(fp, size=3, mode="nearest")
    ball, shrink = _build_ball(radius)
    if shrink > 1:
        rolled = _roll_ball(_shrink(fp, shrink), ball)
        bg = _enlarge(rolled, fp.shape[0], fp.shape[1], shrink)
    else:
        bg = _roll_ball(fp, ball)
    if invert:
        bg = -bg
    return bg


def _apply_over_planes(arr: np.ndarray, radius, invert, do_presmooth) -> np.ndarray:
    """Run the 2-D background estimate over the last two axes of an N-D array."""
    if arr.ndim < 2:
        raise ValueError("rolling-ball background needs at least a 2-D image")
    if arr.ndim == 2:
        return _background_2d(arr, radius, invert, do_presmooth)
    out = np.empty(arr.shape, dtype=np.float32)
    for idx in np.ndindex(arr.shape[:-2]):
        out[idx] = _background_2d(arr[idx], radius, invert, do_presmooth)
    return out


def rolling_ball_background(
    image,
    radius: float = DEFAULT_RADIUS,
    *,
    light_background: bool = False,
    do_presmooth: bool = True,
) -> np.ndarray:
    """Estimate the smooth background of ``image`` by rolling a ball under it.

    A fast ImageJ-faithful rolling ball (Sternberg 1983): the background at each
    pixel is the highest a ball of the given ``radius`` reaches while staying below
    the intensity surface. Choose ``radius`` larger than the biggest feature you
    want treated as foreground.

    Args:
        image: 2-D plane, or N-D array processed plane-by-plane over the last two
            axes (numpy array or anything ``np.asarray`` accepts, e.g. a computed
            dask slice).
        radius: Ball radius in pixels (default 50).
        light_background: True for dark features on a bright field (the ball rolls
            over the top instead of under the bottom).
        do_presmooth: Apply a 3×3 mean before rolling (default True) so isolated
            noisy pixels don't poke through the ball.

    Returns:
        The estimated background as a ``float32`` array shaped like ``image``.
    """
    arr = np.asarray(image)
    return _apply_over_planes(arr, radius, bool(light_background), do_presmooth)


def subtract_background(
    image,
    radius: float = DEFAULT_RADIUS,
    *,
    light_background: bool = False,
    do_presmooth: bool = True,
    return_background: bool = False,
) -> np.ndarray:
    """Subtract a rolling-ball background from ``image`` (ImageJ "Subtract Background").

    Estimates the background with :func:`rolling_ball_background` and returns
    ``image − background``. The input dtype is preserved: for integer images the
    result is offset and clipped exactly as ImageJ does (a light background keeps
    its bright field by adding the dtype maximum; values round-half-up and clip to
    the dtype range), and float images are returned as the raw difference.

    Args and ``radius``/``light_background``/``do_presmooth`` are as in
    :func:`rolling_ball_background`. Pass ``return_background=True`` to get the
    background itself instead of the subtracted image (a ``float32`` array).
    """
    arr = np.asarray(image)
    invert = bool(light_background)
    bg = _apply_over_planes(arr, radius, invert, do_presmooth)
    if return_background:
        return bg
    diff = arr.astype(np.float32) - bg
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        # ImageJ: value = orig - bg + offset, clipped, then truncated to int. The
        # offset folds in a light background's bright field (+max) plus the 0.5
        # that makes the truncation round to nearest.
        offset = (float(info.max) if invert else 0.0) + 0.5
        clipped = np.clip(diff + offset, 0.0, float(info.max))
        return clipped.astype(arr.dtype)  # >= 0, so astype truncates == floor
    return diff.astype(arr.dtype)
