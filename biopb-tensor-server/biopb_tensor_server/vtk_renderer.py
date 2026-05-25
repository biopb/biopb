"""VTK-based image rendering for tensor-server.

Provides backend rendering of microscopy images using VTK, supporting:
- Percentile-based intensity normalization
- Pseudo-color for fluorescence channels
- In-memory PNG/JPEG output

This is an experimental alternative to frontend Pixi.js rendering.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# VTK imports - lazy loaded to avoid import errors when VTK not installed
_vtk_available = False
_vtk_module = None
_numpy_support = None


def _ensure_vtk():
    """Lazy load VTK modules."""
    global _vtk_available, _vtk_module, _numpy_support
    if _vtk_available:
        return
    try:
        import vtk as _vtk_module
        from vtk.util import numpy_support as _numpy_support
        _vtk_available = True
    except ImportError:
        raise ImportError(
            "VTK is required for backend rendering. "
            "Install with: pip install biopb-tensor-server[vtk-render]"
        )


def is_vtk_available() -> bool:
    """Check if VTK is available without importing."""
    try:
        import vtk  # noqa: F401
        return True
    except ImportError:
        return False


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


def build_axis_map(dim_labels: list[str]) -> dict[str, Optional[int]]:
    """Map semantic axis names to dimension indices.

    Mirrors frontend buildAxisMap() in tensor-flight-client.
    """
    result: dict[str, Optional[int]] = {"t": None, "z": None, "c": None, "y": None, "x": None}

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
    unassigned = [i for i, label in enumerate(dim_labels) if label.lower() not in
                  AXIS_T_LABELS | AXIS_Z_LABELS | AXIS_C_LABELS | AXIS_Y_LABELS | AXIS_X_LABELS]

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


def normalize_to_uint8(
    data: np.ndarray,
    lo_val: float,
    hi_val: float,
) -> np.ndarray:
    """Normalize array values to 0-255 uint8 range.

    Optimized to minimize intermediate array allocations.
    """
    if lo_val >= hi_val:
        return np.zeros(data.shape, dtype=np.uint8)

    scale = 255.0 / (hi_val - lo_val)

    # Use float32 for computation (faster than float64)
    normalized = data.astype(np.float32)
    normalized -= lo_val
    normalized *= scale
    # Clip in-place to avoid another intermediate
    np.clip(normalized, 0, 255, out=normalized)
    return normalized.astype(np.uint8)


def apply_pseudo_color(
    gray: np.ndarray,
    color_multipliers: Tuple[float, float, float],
) -> np.ndarray:
    """Apply pseudo-color multipliers to grayscale image.

    Args:
        gray: 2D uint8 array (H, W)
        color_multipliers: (r_mult, g_mult, b_mult) in range [0, 1]

    Returns:
        RGBA array (H, W, 4) as uint8
    """
    r_mult, g_mult, b_mult = color_multipliers

    height, width = gray.shape
    rgba = np.empty((height, width, 4), dtype=np.uint8)

    # Optimized: avoid float64 intermediate by handling common cases directly
    # For multiplier 0: set to 0
    # For multiplier 1: copy gray directly
    # For other values: need multiplication (still slow but less common)

    rgba[:, :, 0] = gray if r_mult == 1.0 else (0 if r_mult == 0.0 else (gray * r_mult).astype(np.uint8))
    rgba[:, :, 1] = gray if g_mult == 1.0 else (0 if g_mult == 0.0 else (gray * g_mult).astype(np.uint8))
    rgba[:, :, 2] = gray if b_mult == 1.0 else (0 if b_mult == 0.0 else (gray * b_mult).astype(np.uint8))
    rgba[:, :, 3] = 255  # A (full opacity)

    return rgba


def extract_yx_slice(
    arr: np.ndarray,
    dim_labels: list[str],
) -> np.ndarray:
    """Extract 2D Y/X slice from multi-dimensional array.

    Uses axis mapping to find Y and X dimensions.
    Assumes array is already sliced to single T/Z/C indices.
    """
    axis_map = build_axis_map(dim_labels)

    # Find Y and X axes
    y_idx = axis_map["y"]
    x_idx = axis_map["x"]

    # Fallback: last two dimensions
    if y_idx is None:
        y_idx = arr.ndim - 2
    if x_idx is None:
        x_idx = arr.ndim - 1

    # If array is already 2D, return it
    if arr.ndim == 2:
        return arr

    # For 3D+ arrays, squeeze out non-Y/X dimensions that have size 1
    # Only squeeze dimensions that are actually size 1 to avoid errors
    squeeze_dims = tuple(
        i for i in range(arr.ndim)
        if i not in (y_idx, x_idx) and arr.shape[i] == 1
    )
    if squeeze_dims:
        squeezed = np.squeeze(arr, axis=squeeze_dims)
        # Update y_idx and x_idx after squeezing (they shift if earlier dims were removed)
        removed_before_y = sum(1 for d in squeeze_dims if d < y_idx)
        removed_before_x = sum(1 for d in squeeze_dims if d < x_idx)
        y_idx = y_idx - removed_before_y
        x_idx = x_idx - removed_before_x
    else:
        squeezed = arr

    # Ensure we have 2D - take first slice along remaining non-Y/X dimensions
    while squeezed.ndim > 2:
        # Find first non-Y/X dimension and take index 0
        for i in range(squeezed.ndim):
            if i != y_idx and i != x_idx:
                squeezed = squeezed[0]
                # Update indices after removing dimension 0
                if y_idx > 0:
                    y_idx -= 1
                if x_idx > 0:
                    x_idx -= 1
                break

    return squeezed


def render_array_to_image_bytes(
    arr: np.ndarray,
    dim_labels: list[str],
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    color: str = "auto",
    channel_name: Optional[str] = None,
    output_format: str = "png",
) -> Tuple[bytes, int, int, float, float]:
    """Render numpy array to PNG/JPEG bytes using VTK.

    Args:
        arr: Input numpy array (any dtype, may be multi-dimensional)
        dim_labels: Dimension labels for axis mapping
        percentile_lo: Lower percentile for intensity normalization (default 1%)
        percentile_hi: Upper percentile for intensity normalization (default 99%)
        color: Color preset name or hex string
        channel_name: Channel name for auto color resolution
        output_format: "png" or "jpeg"

    Returns:
        (image_bytes, width, height, actual_lo_val, actual_hi_val)
    """
    _ensure_vtk()

    # Extract 2D Y/X slice
    yx_slice = extract_yx_slice(arr, dim_labels)

    # Compute percentile cutoffs
    lo_val, hi_val = compute_percentile_cutoffs(yx_slice, percentile_lo, percentile_hi)

    # Normalize to uint8
    gray = normalize_to_uint8(yx_slice, lo_val, hi_val)

    # Resolve color
    color_multipliers = resolve_color(color, channel_name)

    # Apply pseudo-color
    rgba = apply_pseudo_color(gray, color_multipliers)

    height, width = gray.shape

    # Create VTK image data from RGBA array
    vtk_array = _numpy_support.numpy_to_vtk(
        num_array=rgba.reshape(-1, 4),
        deep=True,
        array_type=_vtk_module.VTK_UNSIGNED_CHAR,
    )
    vtk_array.SetNumberOfComponents(4)

    image_data = _vtk_module.vtkImageData()
    image_data.SetDimensions(width, height, 1)
    image_data.GetPointData().SetScalars(vtk_array)

    # Setup rendering pipeline
    mapper = _vtk_module.vtkImageMapper()
    mapper.SetInputData(image_data)
    mapper.SetColorWindow(255)
    mapper.SetColorLevel(127.5)

    actor = _vtk_module.vtkActor2D()
    actor.SetMapper(mapper)

    renderer = _vtk_module.vtkRenderer()
    renderer.AddActor2D(actor)
    renderer.SetBackground(0.0, 0.0, 0.0)

    render_window = _vtk_module.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(width, height)
    render_window.OffScreenRenderingOn()  # Headless rendering
    render_window.Render()

    # Capture to image
    window_to_image = _vtk_module.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGBA()
    window_to_image.Update()

    # Write to memory using vtkMemoryFile
    if output_format.lower() == "jpeg":
        writer = _vtk_module.vtkJPEGWriter()
    else:
        writer = _vtk_module.vtkPNGWriter()

    # Use memory file for in-memory output
    memory_file = _vtk_module.vtkMemoryFile()
    memory_file.SetArrayName("render_output")
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.SetFileName(memory_file.GetFileName())
    writer.Write()

    # Get bytes from memory file
    output_bytes = memory_file.GetArray().tobytes()

    return (output_bytes, width, height, lo_val, hi_val)


# Alternative simpler approach without full rendering pipeline (PIL fallback)
def render_array_to_image_bytes_simple(
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

    This is a simpler alternative that doesn't require VTK.
    """
    import time
    t0 = time.monotonic()

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL/Pillow is required for simple rendering. "
            "Install with: pip install Pillow"
        )

    # Extract 2D Y/X slice
    t1 = time.monotonic()
    yx_slice = extract_yx_slice(arr, dim_labels)
    extract_ms = (time.monotonic() - t1) * 1000

    # Compute percentile cutoffs
    t2 = time.monotonic()
    lo_val, hi_val = compute_percentile_cutoffs(yx_slice, percentile_lo, percentile_hi)
    percentile_ms = (time.monotonic() - t2) * 1000

    # Normalize to uint8
    t3 = time.monotonic()
    gray = normalize_to_uint8(yx_slice, lo_val, hi_val)
    normalize_ms = (time.monotonic() - t3) * 1000

    # Resolve color
    color_multipliers = resolve_color(color, channel_name)

    # Apply pseudo-color
    t4 = time.monotonic()
    rgba = apply_pseudo_color(gray, color_multipliers)
    colorize_ms = (time.monotonic() - t4) * 1000

    height, width = gray.shape

    # Output based on format
    t5 = time.monotonic()
    if output_format.lower() == "raw":
        # Raw RGBA bytes - no encoding, fastest for localhost
        output_bytes = rgba.tobytes()
    else:
        # Create PIL Image and save to bytes
        img = Image.fromarray(rgba, mode="RGBA")

        buffer = io.BytesIO()
        if output_format.lower() == "jpeg":
            # JPEG doesn't support RGBA, convert to RGB
            img_rgb = img.convert("RGB")
            img_rgb.save(buffer, format="JPEG", quality=90)
        else:
            # Use fast PNG compression (level 1) for speed
            img.save(buffer, format="PNG", compress_level=1)
        output_bytes = buffer.getvalue()
    encode_ms = (time.monotonic() - t5) * 1000

    total_ms = (time.monotonic() - t0) * 1000
    logger.debug(
        f"render_simple: shape={height}x{width}, total={total_ms:.1f}ms, "
        f"extract={extract_ms:.1f}ms, percentile={percentile_ms:.1f}ms, "
        f"normalize={normalize_ms:.1f}ms, colorize={colorize_ms:.1f}ms, encode={encode_ms:.1f}ms"
    )

    return (output_bytes, width, height, lo_val, hi_val)