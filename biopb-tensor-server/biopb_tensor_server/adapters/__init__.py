"""Adapter registry and initialization.

This module provides explicit registration of all adapter backends,
making it easy to extend with new formats without modifying cli.py.

Usage:
    from biopb_tensor_server.adapters import get_default_registry
    registry = get_default_registry()
    claims = registry.get_claims_for_path(path, visited)
"""

from biopb_tensor_server.base import BackendAdapter, SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import AdapterRegistry

from .hdf5 import Hdf5Adapter
from .ome_zarr import OmeZarrAdapter
from .tiff import (
    TiffSequenceAdapter,
    MicroManagerLegacyAdapter,
)
from .zarr import ZarrAdapter

# Optional aicsimageio adapters (format-specific subclasses)
try:
    from .aicsimageio import (
        OmeTiffAdapter,
        ZeissAdapter,
        LeicaAdapter,
        NikonAdapter,
        DvAdapter,
        OlympusAdapter,
        AicsImageIoAdapter,
    )
except ImportError:
    OmeTiffAdapter = None  # type: ignore
    ZeissAdapter = None  # type: ignore
    LeicaAdapter = None  # type: ignore
    NikonAdapter = None  # type: ignore
    DvAdapter = None  # type: ignore
    OlympusAdapter = None  # type: ignore
    AicsImageIoAdapter = None  # type: ignore

# Optional medical imaging adapters
try:
    from .dicom import DicomAdapter, DicomSeriesAdapter
except ImportError:
    DicomAdapter = None  # type: ignore
    DicomSeriesAdapter = None  # type: ignore

try:
    from .nifti import NiftiAdapter
except ImportError:
    NiftiAdapter = None  # type: ignore

__all__ = [
    "get_default_registry",
    "AdapterRegistry",
    "BackendAdapter",
    "SourceAdapter",
    "TensorAdapter",
    "ZarrAdapter",
    "Hdf5Adapter",
    "TiffSequenceAdapter",
    "MicroManagerLegacyAdapter",
    "OmeZarrAdapter",
    "OmeTiffAdapter",
    "ZeissAdapter",
    "LeicaAdapter",
    "NikonAdapter",
    "DvAdapter",
    "OlympusAdapter",
    "AicsImageIoAdapter",
    "DicomAdapter",
    "DicomSeriesAdapter",
    "NiftiAdapter",
]


def get_default_registry() -> AdapterRegistry:
    """Get the default adapter registry with all built-in adapters.

    Adapter registration order (by priority/specificity, highest first):
    - OmeTiffAdapter - OME-TIFF files (embedded OME-XML, companion.ome)
    - ZeissAdapter - Zeiss microscopy (CZI, LSM)
    - LeicaAdapter - Leica LIF files
    - NikonAdapter - Nikon ND2 files
    - DvAdapter - DeltaVision DV files
    - OlympusAdapter - Olympus OIF/OIB files
    - AicsImageIoAdapter - Fallback for other aicsimageio-supported formats
    - OmeZarrAdapter - OME-Zarr specific (handles both single images and HCS plates)
    - ZarrAdapter - Generic Zarr fallback
    - MicroManagerLegacyAdapter - Legacy MicroManager datasets with JSON metadata (metadata.txt)
    - TiffSequenceAdapter - Plain TIFF sequences (no metadata)
    - DicomSeriesAdapter - Multi-file DICOM series (directories with same SeriesInstanceUID)
    - DicomAdapter - Single DICOM files (.dcm)
    - NiftiAdapter - NIfTI files (.nii, .nii.gz)
    - Hdf5Adapter - HDF5 files (requires explicit type in config)

    Returns:
        AdapterRegistry with all built-in adapters registered
    """
    registry = AdapterRegistry()

    # Register aicsimageio-based adapters in priority order (most specific first)
    if OmeTiffAdapter is not None:
        registry.register_with_type("ome-tiff", OmeTiffAdapter)
    if ZeissAdapter is not None:
        registry.register_with_type("zeiss", ZeissAdapter)
    if LeicaAdapter is not None:
        registry.register_with_type("leica", LeicaAdapter)
    if NikonAdapter is not None:
        registry.register_with_type("nikon", NikonAdapter)
    if DvAdapter is not None:
        registry.register_with_type("dv", DvAdapter)
    if OlympusAdapter is not None:
        registry.register_with_type("olympus", OlympusAdapter)
    if AicsImageIoAdapter is not None:
        registry.register_with_type("aics", AicsImageIoAdapter)

    # OME-Zarr
    registry.register_with_type("ome-zarr", OmeZarrAdapter)
    registry.register_with_type(
        "ome-zarr-hcs", OmeZarrAdapter
    )  # HCS plates use same adapter

    # TIFF/MicroManager
    registry.register_with_type("micromanager-legacy", MicroManagerLegacyAdapter)
    registry.register_with_type("tiff-sequence", TiffSequenceAdapter)

    # Medical imaging adapters
    if DicomSeriesAdapter is not None:
        registry.register_with_type("dicom-series", DicomSeriesAdapter)
    if DicomAdapter is not None:
        registry.register_with_type("dicom", DicomAdapter)
    if NiftiAdapter is not None:
        registry.register_with_type("nifti", NiftiAdapter)

    registry.register_with_type("hdf5", Hdf5Adapter)

    return registry
