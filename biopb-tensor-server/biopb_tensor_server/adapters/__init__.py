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

from .zarr import ZarrAdapter
from .hdf5 import Hdf5Adapter
from .tiff import OmeTiffAdapter, MultiFileOmeTiffAdapter
from .ome_zarr import OmeZarrAdapter

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

    Adapter registration order matters for claim priority:
    - More specific adapters should be registered first
    - OmeZarrAdapter before ZarrAdapter (OME-Zarr is a subset)
    - MultiFileOmeTiffAdapter before OmeTiffAdapter (multi-file is more specific)

    Returns:
        AdapterRegistry with all built-in adapters registered
    """
    registry = AdapterRegistry()

    # Register in priority order (most specific first) with explicit type mapping
    registry.register_with_type("ome-zarr", OmeZarrAdapter)
    registry.register_with_type("zarr", ZarrAdapter)
    registry.register_with_type("ome-tiff-multifile", MultiFileOmeTiffAdapter)
    registry.register_with_type("ome-tiff", OmeTiffAdapter)

    # Optional aicsimageio adapter
    if AicsImageIoAdapter is not None:
        registry.register_with_type("aics", AicsImageIoAdapter)

    registry.register_with_type("hdf5", Hdf5Adapter)

    return registry