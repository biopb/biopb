"""Image rendering for tensor-server.

Provides backend rendering of microscopy images using PIL/Pillow, supporting:
- Percentile-based intensity normalization
- Pseudo-color for fluorescence channels
- In-memory PNG/JPEG/Raw output
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color resolution (mirror of frontend colorUtils.ts)
# ---------------------------------------------------------------------------

PRESET_COLOR_MULTIPLIERS: dict[str, Tuple[float, float, float]] = {
    "auto": (1.0, 1.0, 1.0),  # Placeholder - resolved at render time
    "gray": (1.0, 1.0, 1.0),
    "grayscale": (1.0, 1.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def hex_to_multipliers(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB multipliers."""
    clean_hex = hex_color.replace("#", "")
    # Expand shorthand (e.g., "f00" -> "ff0000")
    if len(clean_hex) == 3:
        clean_hex = clean_hex[0] * 2 + clean_hex[1] * 2 + clean_hex[2] * 2
    r = int(clean_hex[0:2], 16) / 255.0
    g = int(clean_hex[2:4], 16) / 255.0
    b = int(clean_hex[4:6], 16) / 255.0
    return (r, g, b)


def guess_default_color(channel_name: str) -> str:
    """Guess display color from fluorescence channel name.

    Mirrors frontend colorUtils.ts guessDefaultColor().
    """
    name = channel_name.lower()

    # Green fluorescence markers
    if re.search(r"\bgfp\b|\bgreen\b|fitc|egfp|af488|cy2\b", name):
        return "green"

    # Red fluorescence markers
    if re.search(r"\brfp\b|\bred\b|mcherry|tritc|af568|af594|tdtomato|tomato", name):
        return "red"

    # Blue fluorescence markers (typically nuclear)
    if re.search(r"\bdapi\b|\bblue\b|hoechst|uv", name):
        return "blue"

    # Far red / magenta (Cy5 and similar)
    if re.search(r"\bcy5\b|\bmagenta\b|af647|af680|cy5\.5|cy7|ir800|ir700", name):
        return "magenta"

    # Cyan markers (Cy3 and similar)
    if re.search(r"\bcy3\b|\bcyan\b|\bcfp\b|cy3\.5", name):
        return "cyan"

    # Yellow markers
    if re.search(r"\byfp\b|\byellow\b|vfp", name):
        return "yellow"

    # Alexa Fluor series
    if re.search(r"af350|alexa350", name):
        return "blue"
    if re.search(r"af405|alexa405", name):
        return "blue"
    if re.search(r"af488|alexa488", name):
        return "green"
    if re.search(r"af555|alexa555", name):
        return "cyan"
    if re.search(r"af568|alexa568", name):
        return "red"
    if re.search(r"af594|alexa594", name):
        return "red"
    if re.search(r"af647|alexa647", name):
        return "magenta"
    if re.search(r"af680|alexa680", name):
        return "magenta"
    if re.search(r"af700|alexa700", name):
        return "magenta"
    if re.search(r"af750|alexa750", name):
        return "magenta"

    # Dye series
    if re.search(r"dyelight.*350|dl350", name):
        return "blue"
    if re.search(r"dyelight.*488|dl488", name):
        return "green"
    if re.search(r"dyelight.*550|dl550", name):
        return "cyan"
    if re.search(r"dyelight.*594|dl594", name):
        return "red"
    if re.search(r"dyelight.*633|dl633", name):
        return "magenta"
    if re.search(r"dyelight.*650|dl650", name):
        return "magenta"
    if re.search(r"dyelight.*680|dl680", name):
        return "magenta"
    if re.search(r"dyelight.*755|dl755", name):
        return "magenta"

    # Atto dyes
    if re.search(r"atto.*390|a390", name):
        return "blue"
    if re.search(r"atto.*425|a425", name):
        return "blue"
    if re.search(r"atto.*488|a488", name):
        return "green"
    if re.search(r"atto.*550|a550", name):
        return "cyan"
    if re.search(r"atto.*565|a565", name):
        return "red"
    if re.search(r"atto.*594|a594", name):
        return "red"
    if re.search(r"atto.*647|a647|atto647n", name):
        return "magenta"
    if re.search(r"atto.*680|a680", name):
        return "magenta"

    # Generic wavelength-based heuristics
    match = re.search(r"\b(\d{3})\b", name)
    if match:
        wavelength = int(match.group(1))
        if 350 <= wavelength <= 499:
            return "blue"
        if 500 <= wavelength <= 549:
            return "green"
        if 550 <= wavelength <= 569:
            return "cyan"
        if 570 <= wavelength <= 649:
            return "red"
        if wavelength >= 650:
            return "magenta"

    return "gray"


def resolve_color(
    color: str, channel_name: Optional[str] = None
) -> Tuple[float, float, float]:
    """Resolve color string to RGB multipliers.

    Args:
        color: Preset color name ("auto", "green", etc.) or hex string (#rrggbb)
        channel_name: Channel name for "auto" color resolution

    Returns:
        (r, g, b) multipliers in range [0, 1]
    """
    if color == "auto" and channel_name:
        resolved = guess_default_color(channel_name)
        return PRESET_COLOR_MULTIPLIERS.get(resolved, (1.0, 1.0, 1.0))

    if color.startswith("#"):
        return hex_to_multipliers(color)

    return PRESET_COLOR_MULTIPLIERS.get(color.lower(), (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Axis mapping (mirror of frontend buildAxisMap)
# ---------------------------------------------------------------------------

# Recognized axis labels
AXIS_T_LABELS = {"t", "time", "frame", "frames"}
AXIS_Z_LABELS = {"z", "depth", "plane", "planes", "slice"}
AXIS_C_LABELS = {"c", "channel", "channels", "band", "bands"}
AXIS_Y_LABELS = {"y", "height", "row", "rows"}
AXIS_X_LABELS = {"x", "width", "col", "cols", "column", "columns"}
# Interleaved RGB(A) samples axis. aicsimageio labels the samples axis of a
# photometric-RGB image "S" (dims "TCZYXS"); its size is 3 (RGB) or 4 (RGBA).
AXIS_S_LABELS = {"s", "samples"}


def samples_axis(dim_labels: list[str], shape: tuple[int, ...]) -> Optional[int]:
    """Index of an interleaved RGB(A) samples axis, or ``None``.

    Detected by *label* (``S`` / ``samples``) gated on a size of 3 or 4, so a
    size-3 channel or Z axis is never mistaken for color. This axis holds the
    color components of one pixel and must be composited into RGB, not selected
    one-plane-at-a-time like T/Z/C.
    """
    for i, label in enumerate(dim_labels):
        if label.lower() in AXIS_S_LABELS and i < len(shape) and shape[i] in (3, 4):
            return i
    return None


def build_axis_map(dim_labels: list[str]) -> dict[str, Optional[int]]:
    """Map semantic axis names to dimension indices.

    Mirrors frontend buildAxisMap() in tensor-flight-client.
    """
    result: dict[str, Optional[int]] = {
        "t": None,
        "z": None,
        "c": None,
        "y": None,
        "x": None,
    }

    for i, label in enumerate(dim_labels):
        label_lower = label.lower()
        if label_lower in AXIS_T_LABELS:
            result["t"] = i
        elif label_lower in AXIS_Z_LABELS:
            result["z"] = i
        elif label_lower in AXIS_C_LABELS:
            result["c"] = i
        elif label_lower in AXIS_Y_LABELS:
            result["y"] = i
        elif label_lower in AXIS_X_LABELS:
            result["x"] = i

    # Fallback heuristic for unmapped axes
    unassigned = [
        i
        for i, label in enumerate(dim_labels)
        if label.lower()
        not in AXIS_T_LABELS
        | AXIS_Z_LABELS
        | AXIS_C_LABELS
        | AXIS_Y_LABELS
        | AXIS_X_LABELS
    ]

    # Positional fallback: last → X, second-last → Y, third-last → Z
    if result["x"] is None and unassigned:
        result["x"] = unassigned.pop()  # last
    if result["y"] is None and unassigned:
        result["y"] = unassigned.pop()  # second-last
    if result["z"] is None and unassigned:
        result["z"] = unassigned.pop()  # third-last

    return result


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------


def compute_percentile_cutoffs(
    data: np.ndarray,
    lo: float,
    hi: float,
    sample_size: int = 65536,
) -> Tuple[float, float]:
    """Compute percentile cutoffs for intensity normalization.

    Uses sampling for large arrays to improve performance,
    similar to the frontend's approach.
    """
    if data.size == 0:
        return (0.0, 1.0)

    # For small arrays, use full data
    if data.size <= sample_size:
        lo_val = float(np.percentile(data, lo))
        hi_val = float(np.percentile(data, hi))
    else:
        # Sample uniformly across the array
        step = data.size / sample_size
        indices = np.arange(sample_size) * step
        indices = indices.astype(np.int64)
        indices = np.minimum(indices, data.size - 1)  # Clamp to valid range
        sample = data.ravel()[indices]

        # Sort sample and compute percentile positions
        sample.sort()
        lo_idx = int(sample_size * lo / 100.0)
        hi_idx = min(sample_size - 1, int(sample_size * hi / 100.0))

        lo_val = float(sample[lo_idx])
        hi_val = float(sample[hi_idx])

    if lo_val >= hi_val:
        return (0.0, 1.0)

    return (lo_val, hi_val)


def normalize_and_colorize(
    data: np.ndarray,
    lo_val: float,
    hi_val: float,
    color_multipliers: Tuple[float, float, float],
) -> np.ndarray:
    """Normalize to uint8 and apply pseudo-color in one pass, returning RGB (H, W, 3).

    For uint8/uint16 input, builds an RGB LUT (≤256 KB) and does a single indexed
    read — no gray intermediate, no separate colorize pass.
    """
    r_mult, g_mult, b_mult = color_multipliers

    if lo_val >= hi_val:
        return np.zeros((*data.shape, 3), dtype=np.uint8)

    scale = 255.0 / (hi_val - lo_val)

    normalized = data.astype(np.float32)
    normalized -= lo_val
    normalized *= scale
    np.clip(normalized, 0, 255, out=normalized)

    rgb = np.empty((*data.shape, 3), dtype=np.uint8)
    for i, mult in enumerate((r_mult, g_mult, b_mult)):
        if mult == 1.0:
            rgb[:, :, i] = normalized
        elif mult == 0.0:
            rgb[:, :, i] = 0
        else:
            rgb[:, :, i] = normalized * mult

    return rgb


def _resolve_plane_axes(
    ndim: int,
    dim_labels: list[str],
    shape: Tuple[int, ...],
) -> Tuple[int, int, Optional[int]]:
    """Pick distinct ``(y_idx, x_idx, s_idx)`` axes for the display plane.

    Robust to malformed/degenerate label sets (a defensive requirement -- the
    labels are attacker/adapter-supplied, not trusted). Guarantees ``y_idx`` and
    ``x_idx`` are two *distinct*, in-range axes and ``s_idx`` (if not ``None``)
    is a third distinct axis, so the caller's transpose can never see a repeated
    axis. The interleaved-samples axis is honored only when it does not shadow a
    labeled Y/X and at least two other axes remain to serve as Y and X -- so a
    malformed ``"CYS"`` or ``"YS"`` degrades to a plain grayscale plane instead
    of crashing. Assumes ``ndim >= 2`` (the caller guards ``ndim < 2``).
    """
    axis_map = build_axis_map(dim_labels)
    y_lbl = axis_map["y"]
    x_lbl = axis_map["x"]

    s_idx = samples_axis(dim_labels, shape)
    # Drop a samples axis that collides with a labeled Y/X, or that would leave
    # fewer than two axes for the Y/X plane.
    if s_idx is not None and (s_idx in (y_lbl, x_lbl) or ndim < 3):
        s_idx = None

    non_samples = [i for i in range(ndim) if i != s_idx]

    # X: labeled X if usable, else the trailing non-samples axis.
    if x_lbl is not None and x_lbl in non_samples:
        x_idx = x_lbl
    else:
        x_idx = non_samples[-1]

    # Y: labeled Y if usable and distinct from X, else the trailing non-samples
    # axis that is not X.
    if y_lbl is not None and y_lbl in non_samples and y_lbl != x_idx:
        y_idx = y_lbl
    else:
        y_idx = next((i for i in reversed(non_samples) if i != x_idx), x_idx)

    return y_idx, x_idx, s_idx


def extract_yx_slice(
    arr: np.ndarray,
    dim_labels: list[str],
) -> np.ndarray:
    """Reduce a multi-dimensional array to the plane the renderer displays.

    Returns a 2-D ``[Y, X]`` array, or a 3-D ``[Y, X, S]`` array when an
    interleaved RGB(A) samples axis is present (see :func:`samples_axis`). Every
    other axis (T, Z, C, ...) is reduced to its first index -- the slice request
    has already pinned those to a single plane -- and the kept axes are ordered
    ``[Y, X]`` (+ trailing samples), ready for the renderer.
    """
    # Below 2-D there is no Y/X plane to reduce to; promote to a 2-D (1 x N)
    # strip so the renderer always gets at least a plane and never a repeated /
    # negative transpose axis from _resolve_plane_axes.
    if arr.ndim < 2:
        return arr.reshape((1,) * (2 - arr.ndim) + arr.shape)

    y_idx, x_idx, s_idx = _resolve_plane_axes(arr.ndim, dim_labels, arr.shape)

    # Keep Y, X and (if any) the samples axis; reduce every other axis to index
    # 0. Integer indexing drops the reduced axes; the kept axes keep their
    # relative order, which the transpose below normalizes to [Y, X, (S)]. The
    # three kept indices are distinct by construction, so the transpose is safe.
    keep = [y_idx, x_idx] + ([s_idx] if s_idx is not None else [])
    index = tuple(slice(None) if i in keep else 0 for i in range(arr.ndim))
    reduced = arr[index]

    survivors = [i for i in range(arr.ndim) if i in keep]
    new_pos = {orig: p for p, orig in enumerate(survivors)}
    order = [new_pos[y_idx], new_pos[x_idx]]
    if s_idx is not None:
        order.append(new_pos[s_idx])
    return np.transpose(reduced, order)


def normalize_rgb_samples(
    plane: np.ndarray,
    lo_val: float,
    hi_val: float,
) -> np.ndarray:
    """Render an interleaved RGB(A) ``[Y, X, S]`` plane to display RGB ``(H, W, 3)``.

    True color: a *shared* intensity stretch (one ``lo``/``hi`` for all samples)
    is applied so the color balance is preserved, and no pseudo-color multiplier
    is used. RGBA (``S == 4``) drops the alpha channel, which the RGB encoders
    cannot carry.
    """
    rgb = plane[:, :, :3]
    h, w = rgb.shape[:2]
    if lo_val >= hi_val:
        return np.zeros((h, w, 3), dtype=np.uint8)

    scale = 255.0 / (hi_val - lo_val)
    out = rgb.astype(np.float32)
    out -= lo_val
    out *= scale
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def render_array_to_image_bytes(
    arr: np.ndarray,
    dim_labels: list[str],
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    color: str = "auto",
    channel_name: Optional[str] = None,
    output_format: str = "png",
) -> Tuple[bytes, int, int, float, float]:
    """Render numpy array to image bytes using PIL.

    Output formats:
    - "png": PNG with fast compression (compress_level=1)
    - "jpeg": JPEG with quality=90
    - "raw": Raw RGBA bytes (uint8, 4 bytes per pixel) - fastest for localhost

    Returns:
        (image_bytes, width, height, actual_lo_val, actual_hi_val)
    """
    import time

    t0 = time.monotonic()

    # Reduce to the display plane: 2-D [Y, X], or 3-D [Y, X, S] for RGB(A).
    t1 = time.monotonic()
    yx_slice = extract_yx_slice(arr, dim_labels)
    extract_ms = (time.monotonic() - t1) * 1000

    is_rgb = yx_slice.ndim == 3

    # Compute percentile cutoffs. For an interleaved RGB(A) plane a *shared*
    # stretch across the color samples preserves the color balance -- but the
    # alpha sample (RGBA) is excluded from the statistics, else a constant opaque
    # alpha=255 inflates the high cutoff and washes out the RGB.
    t2 = time.monotonic()
    stat_source = yx_slice[..., :3] if is_rgb else yx_slice
    lo_val, hi_val = compute_percentile_cutoffs(
        stat_source, percentile_lo, percentile_hi
    )
    percentile_ms = (time.monotonic() - t2) * 1000

    t3 = time.monotonic()
    if is_rgb:
        # Interleaved RGB(A) samples: composite directly, no pseudo-color.
        rgb = normalize_rgb_samples(yx_slice, lo_val, hi_val)
    else:
        # Single grayscale plane: pseudo-color per the requested display color.
        color_multipliers = resolve_color(color, channel_name)
        rgb = normalize_and_colorize(yx_slice, lo_val, hi_val, color_multipliers)
    colorize_ms = (time.monotonic() - t3) * 1000

    height, width = rgb.shape[:2]

    # Output based on format
    t4 = time.monotonic()
    if output_format.lower() == "raw":
        output_bytes = rgb.tobytes()
    elif output_format.lower() == "jpeg":
        try:
            import simplejpeg
        except ImportError:
            raise ImportError(
                "simplejpeg is required for JPEG rendering. "
                "Install with: pip install simplejpeg"
            )
        output_bytes = simplejpeg.encode_jpeg(
            np.ascontiguousarray(rgb), quality=90, colorspace="RGB"
        )
    else:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for PNG rendering. "
                "Install with: pip install Pillow"
            )
        img = Image.fromarray(rgb, mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=1)
        output_bytes = buffer.getvalue()
    encode_ms = (time.monotonic() - t4) * 1000

    total_ms = (time.monotonic() - t0) * 1000
    logger.debug(
        f"render: shape={height}x{width}, total={total_ms:.1f}ms, "
        f"extract={extract_ms:.1f}ms, percentile={percentile_ms:.1f}ms, "
        f"colorize={colorize_ms:.1f}ms, encode={encode_ms:.1f}ms"
    )

    return (output_bytes, width, height, lo_val, hi_val)
