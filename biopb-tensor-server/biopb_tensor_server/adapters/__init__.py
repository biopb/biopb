"""Adapter registry and initialization.

This module provides explicit registration of all adapter backends,
making it easy to extend with new formats without modifying cli.py.

Usage:
    from biopb_tensor_server.adapters import get_default_registry
    registry = get_default_registry()
    claims = registry.get_claims_for_path(path, visited)
"""

from biopb_tensor_server.base import BackendAdapter
from biopb_tensor_server.discovery import AdapterRegistry

from .hdf5 import Hdf5Adapter
from .ome_zarr import OmeZarrAdapter
from .tiff import MultiFileOmeTiffAdapter, OmeTiffAdapter
from .zarr import ZarrAdapter

# Optional aicsimageio adapter
try:
    from .aicsimageio import AicsImageIoAdapter
except ImportError:
    AicsImageIoAdapter = None  # type: ignore

__all__ = [
    'get_default_registry',
    'AdapterRegistry',
    'BackendAdapter',
    'ZarrAdapter',
    'Hdf5Adapter',
    'OmeTiffAdapter',
    'MultiFileOmeTiffAdapter',
    'OmeZarrAdapter',
    'AicsImageIoAdapter',
]


def get_default_registry() -> AdapterRegistry:
    """Get the default adapter registry with all built-in adapters.

    Adapter registration order (by priority/specificity, highest first):
    1. AicsImageIoAdapter - Primary handler (well-maintained, supports CZI, LIF, ND2, DV, LSM, OIF, OIB, XML)
    2. OmeZarrAdapter - OME-Zarr specific (more specific than generic Zarr)
    3. ZarrAdapter - Generic Zarr fallback
    4. MultiFileOmeTiffAdapter - Multi-file OME-TIFF/MicroManager datasets
    5. OmeTiffAdapter - Single-file OME-TIFF only (.ome.tiff/.ome.tif extensions)
    6. Hdf5Adapter - HDF5 files (requires explicit type in config)

    Returns:
        AdapterRegistry with all built-in adapters registered
    """
    registry = AdapterRegistry()

    # Register in priority order (most specific first) with explicit type mapping
    if AicsImageIoAdapter is not None:
        registry.register_with_type("aics", AicsImageIoAdapter)

    registry.register_with_type("ome-zarr", OmeZarrAdapter)
    registry.register_with_type("zarr", ZarrAdapter)
    registry.register_with_type("ome-tiff-multifile", MultiFileOmeTiffAdapter)
    registry.register_with_type("ome-tiff", OmeTiffAdapter)

    registry.register_with_type("hdf5", Hdf5Adapter)

    return registry