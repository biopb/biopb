"""Adapter registry and initialization.

This module provides explicit registration of all adapter backends,
making it easy to extend with new formats without modifying cli.py.

Usage:
    from biopb_tensor_server.adapters import get_default_registry
    registry = get_default_registry()
    claims = registry.get_claims_for_path(path, visited)
"""

from biopb_tensor_server.core.base import BackendAdapter, SourceAdapter, TensorAdapter
from biopb_tensor_server.core.discovery import AdapterRegistry

from .hdf5 import Hdf5Adapter
from .ome_tiff import OmeTiffAdapter
from .ome_zarr import OmeZarrAdapter
from .remote_tensor import RemoteTensorAdapter
from .tiff import (
    MicroManagerLegacyAdapter,
    TiffSequenceAdapter,
)
from .zarr import ZarrAdapter

# Optional ndtiff adapter (Micro-Manager NDTiff storage format)
try:
    from .ndtiff import NdTiffAdapter
except ImportError:
    NdTiffAdapter = None  # type: ignore

# QPTIFF (Akoya PhenoImager) reads via tifffile (a core dep), but decoding its
# compressed tiles (JPEG/LZW) needs imagecodecs -- gate on it so a slim install
# without imagecodecs simply doesn't claim .qptiff rather than claim-then-error
# at read (biopb/biopb#135).
try:
    import imagecodecs  # noqa: F401

    from .qptiff import QptiffAdapter
except ImportError:
    QptiffAdapter = None  # type: ignore
# Optional electron-microscopy adapters (rosettasciio)
try:
    from .mrc import MrcAdapter
except ImportError:
    MrcAdapter = None  # type: ignore

try:
    from .emd import EmdAdapter
except ImportError:
    EmdAdapter = None  # type: ignore

# Optional bioio adapters (format-specific subclasses)
try:
    from .bioio import (
        AicsImageIoAdapter,
        BioformatsAdapter,
        DvAdapter,
        LeicaAdapter,
        NikonAdapter,
        OlympusAdapter,
        ZeissAdapter,
    )
except ImportError:
    ZeissAdapter = None  # type: ignore
    LeicaAdapter = None  # type: ignore
    NikonAdapter = None  # type: ignore
    DvAdapter = None  # type: ignore
    OlympusAdapter = None  # type: ignore
    BioformatsAdapter = None  # type: ignore
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
    "RemoteTensorAdapter",
    "NdTiffAdapter",
    "OmeTiffAdapter",
    "QptiffAdapter",
    "MrcAdapter",
    "EmdAdapter",
    "ZeissAdapter",
    "LeicaAdapter",
    "NikonAdapter",
    "DvAdapter",
    "OlympusAdapter",
    "BioformatsAdapter",
    "AicsImageIoAdapter",
    "DicomAdapter",
    "DicomSeriesAdapter",
    "NiftiAdapter",
]


def get_default_registry() -> AdapterRegistry:
    """Get the default adapter registry with all built-in adapters.

    Adapter registration order (by priority/specificity, highest first):
    - OmeTiffAdapter - OME-TIFF files (embedded OME-XML, companion.ome)
    - QptiffAdapter - Akoya PhenoImager QPTIFF (.qptiff by extension; tifffile,
      native pyramid)
    - MrcAdapter - MRC electron microscopy (.mrc/.mrcs/.rec/.st/.map; rosettasciio)
    - EmdAdapter - EMD electron microscopy (.emd, NCEM/Velox; rosettasciio)
    - ZeissAdapter - Zeiss microscopy (CZI, LSM)
    - LeicaAdapter - Leica LIF files
    - NikonAdapter - Nikon ND2 files
    - DvAdapter - DeltaVision DV files
    - OlympusAdapter - Olympus OIF/OIB files
    - BioformatsAdapter - Legacy Bio-Formats-only formats (ZVI, ...; requires the
      optional bioformats component)
    - AicsImageIoAdapter - Fallback for other bioio-supported formats
    - OmeZarrAdapter - OME-Zarr specific (handles both single images and HCS plates)
    - ZarrAdapter - Generic Zarr fallback
    - NdTiffAdapter - Micro-Manager NDTiff storage (NDTiff.index + NDTiffStack_*.tif)
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

    # Pure-tifffile OME-TIFF adapter first (no bioio dependency, so always
    # available), so it wins for a local OME-TIFF; a remote/exotic .tif it
    # declines falls through to the generic bioio adapter registered below.
    registry.register_with_type("ome-tiff", OmeTiffAdapter)

    # QPTIFF before the bioio group so it owns .qptiff (bioio would drop the
    # QPTIFF pyramid). Claim is suffix-only -- a QPTIFF saved as .tif is not
    # sniffed (that read is unsafe under cloud and wasteful without caching), so
    # it falls through to bioio unless configured with an explicit type: qptiff.
    # Registration order is claim probe order; callers take claims[0]. See
    # biopb/biopb#135.
    if QptiffAdapter is not None:
        registry.register_with_type("qptiff", QptiffAdapter)
    # Electron-microscopy adapters (rosettasciio), before the bioio group so they
    # own their extensions -- notably .mrc, which no bioio plugin can read
    # (biopb/biopb#94). Registration order is claim probe order; callers take
    # claims[0].
    if MrcAdapter is not None:
        registry.register_with_type("mrc", MrcAdapter)
    if EmdAdapter is not None:
        registry.register_with_type("emd", EmdAdapter)

    # Register bioio-based adapters in priority order (most specific first)
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
    if BioformatsAdapter is not None:
        registry.register_with_type("bioformats", BioformatsAdapter)
    if AicsImageIoAdapter is not None:
        registry.register_with_type("aics", AicsImageIoAdapter)

    # OME-Zarr (specific) before plain Zarr (generic fallback). Order is
    # load-bearing: get_claims_for_path returns claims in registration order and
    # callers take claims[0], so for a .zarr both adapters could claim/defer and
    # OmeZarrAdapter must win. OmeZarrAdapter declines a real plain zarr (no
    # multiscales) and ZarrAdapter declines a real OME-Zarr, so claims[0] lands on
    # the right adapter once the store is resident (e.g. at cloud resolve).
    registry.register_with_type("ome-zarr", OmeZarrAdapter)
    registry.register_with_type(
        "ome-zarr-hcs", OmeZarrAdapter
    )  # HCS plates use same adapter
    registry.register_with_type("zarr", ZarrAdapter)

    # TIFF/MicroManager - NDTiff before Legacy (newer format)
    if NdTiffAdapter is not None:
        registry.register_with_type("ndtiff", NdTiffAdapter)
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

    # Remote tensor server -- a caching passthrough proxy (biopb/biopb#178).
    # Config-only (grpc:// url, like hdf5 it never claims a filesystem path), so
    # its claim() default returns None and it only registers by explicit type.
    registry.register_with_type("tensor-server", RemoteTensorAdapter)

    return registry
