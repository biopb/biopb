"""pytest fixtures for biopb_tensor_server tests.

Importable via pytest_plugins mechanism for shared fixtures.
"""

import pytest
import tempfile

from biopb_tensor_server.fixtures import (
    create_multifile_micromanager_dataset,
    create_multifile_ome_dataset,
    create_multiresolution_ome_zarr,
    create_tiled_ome_tiff,
    create_hdf5_dataset,
    create_zarr_array,
)


# =============================================================================
# pytest fixtures using the factory functions
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def multifile_mm_dataset(temp_dir):
    """Complete Micro-Manager multi-file dataset."""
    return create_multifile_micromanager_dataset(temp_dir)


@pytest.fixture
def multifile_mm_dataset_incomplete(temp_dir):
    """Incomplete Micro-Manager multi-file dataset (missing one channel)."""
    return create_multifile_micromanager_dataset(temp_dir, complete=False)


@pytest.fixture
def multifile_ome_dataset(temp_dir):
    """Complete multi-file OME-TIFF dataset with OME-XML metadata."""
    return create_multifile_ome_dataset(temp_dir)


@pytest.fixture
def multifile_ome_dataset_incomplete(temp_dir):
    """Incomplete multi-file OME-TIFF dataset (missing one file)."""
    return create_multifile_ome_dataset(temp_dir, complete=False)


@pytest.fixture
def multires_ome_zarr(temp_dir):
    """Multi-resolution OME-Zarr dataset."""
    return create_multiresolution_ome_zarr(temp_dir)


@pytest.fixture
def tiled_ome_tiff(temp_dir):
    """Tiled OME-TIFF file."""
    return create_tiled_ome_tiff(temp_dir)


@pytest.fixture
def hdf5_dataset(temp_dir):
    """HDF5 dataset with chunked array."""
    return create_hdf5_dataset(temp_dir)


@pytest.fixture
def simple_zarr_array(temp_dir):
    """Simple Zarr array for basic tests."""
    return create_zarr_array(temp_dir)