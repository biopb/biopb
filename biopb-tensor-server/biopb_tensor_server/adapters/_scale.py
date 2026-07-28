"""Shared helpers for adapter physical-scale projection (issue #272).

Small, dependency-free utilities the file adapters use to fold their
already-resident calibration metadata into the compact per-dimension
``physical_scale`` / ``physical_unit`` summary that the tensor-load hot path
carries (see ``TensorAdapter._physical_scale``). Kept here so the
DICOM / TIFF / MicroManager adapters share one implementation of the
label-mapping tail and the unit canonicalisation instead of each reinventing
it. The NIfTI and HDF5 adapters map positionally (their calibration vectors are
already axis-aligned) and so build their vectors directly rather than through
:func:`scale_by_label`.
"""

from typing import Dict, List, Optional, Tuple

# Canonical physical unit the file adapters normalise heterogeneous calibration
# units to, so clients see one consistent spatial unit (OME emits "µm" too).
MICRON = "µm"

# Lower-cased unit string -> micrometres per unit, for canonicalising the
# heterogeneous unit spellings the formats emit (ImageJ "micron", MicroManager
# "um", TIFF ResolutionUnit names, ...). Only length units are listed; a unit we
# cannot place (e.g. "pixel") maps to ``None`` and disables the conversion.
_UNIT_TO_UM: Dict[str, Optional[float]] = {
    "": None,  # sentinel: treated as unknown by unit_to_um
    "px": None,
    "pixel": None,
    "pixels": None,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "nanometers": 1e-3,
    "nanometres": 1e-3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,  # note: distinct code point from the line above (U+03BC vs U+00B5)
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "micrometers": 1.0,
    "micrometres": 1.0,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
    "millimeters": 1e3,
    "millimetres": 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    "centimetre": 1e4,
    "centimeters": 1e4,
    "centimetres": 1e4,
    "m": 1e6,
    "meter": 1e6,
    "metre": 1e6,
    "meters": 1e6,
    "metres": 1e6,
    "inch": 25400.0,
    "inches": 25400.0,
    "in": 25400.0,
    '"': 25400.0,
}


def unit_to_um(unit) -> Optional[float]:
    """Micrometres per ``unit`` for a length-unit string, or ``None``.

    ``None`` for an empty / unknown / non-length unit (e.g. ``"pixel"``), so a
    caller can tell "no usable physical unit" from a real conversion factor.
    """
    if unit is None:
        return None
    return _UNIT_TO_UM.get(str(unit).strip().lower())


def scale_by_label(
    dim_labels,
    value_by_label: Dict[str, Optional[float]],
    unit: str,
) -> Optional[Tuple[List[float], List[str]]]:
    """Fold a ``{lowercased-label: size}`` map onto ``dim_labels``.

    Returns ``(scale, unit)`` -- two lists aligned 1:1 with ``dim_labels`` -- per
    the ``_physical_scale`` contract: a label present with a positive size gets
    that size and ``unit``; every other axis (a label not in the map, or a
    non-positive/None size) gets ``0.0`` / ``""``. Returns ``None`` when no axis
    carries a positive size, so the descriptor fields are left clear rather than
    advertising an all-zero calibration.
    """
    scale: List[float] = []
    units: List[str] = []
    for lab in dim_labels:
        v = value_by_label.get(str(lab).lower())
        if v is not None and v > 0:
            scale.append(float(v))
            units.append(unit)
        else:
            scale.append(0.0)
            units.append("")
    if not any(scale):
        return None
    return scale, units


def axes_scale(axes, dim_labels) -> Optional[Tuple[List[float], List[str]]]:
    """Fold rsciio-style per-axis ``{"scale", "units"}`` dicts onto ``dim_labels``.

    The EM readers (MRC, EMD) hand back one axis dict per dimension already in
    ``dim_labels`` order, each carrying ``scale`` (voxel size) and ``units``.
    Reads them positionally: a positive, parseable ``scale`` keeps its size and
    unit; every other axis gets ``0.0`` / ``""``. Returns ``None`` when the axis
    count doesn't match ``dim_labels`` or no axis carries a positive size, so the
    descriptor fields are left clear rather than advertising an empty calibration.
    """
    labels = list(dim_labels)
    scale: List[float] = []
    unit: List[str] = []
    for ax in axes:
        try:
            v = float(ax.get("scale") or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            scale.append(v)
            unit.append(str(ax.get("units")) if ax.get("units") else "")
        else:
            scale.append(0.0)
            unit.append("")
    if len(scale) != len(labels) or not any(scale):
        return None
    return scale, unit


def mm_summary_scale(summary, dim_labels) -> Optional[Tuple[List[float], List[str]]]:
    """MicroManager summary metadata -> per-dim physical scale, in µm.

    Reads the isotropic in-plane pixel size (``PixelSize_um``) onto the ``x`` /
    ``y`` axes and the z-step (``z-step_um``) onto ``z``; all other axes
    (position / time / channel) get ``0.0`` / ``""``. Shared by the NDTiff and
    legacy-MicroManager adapters, whose summaries carry the same keys. Returns
    ``None`` when ``summary`` is not a dict or carries no positive size.
    """
    if not isinstance(summary, dict):
        return None

    def _first_positive(keys) -> Optional[float]:
        for k in keys:
            if k not in summary:
                continue
            try:
                v = float(summary[k])
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return None

    pixel_um = _first_positive(("PixelSize_um", "PixelSizeUm", "PixelSize_um_"))
    z_um = _first_positive(("z-step_um", "zStep_um", "z_step_um", "Z-step_um"))
    return scale_by_label(dim_labels, {"x": pixel_um, "y": pixel_um, "z": z_um}, MICRON)
