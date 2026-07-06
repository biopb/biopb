"""Test fixtures for biopb-tensor-server tests.

Imports fixture factory functions from fixtures module and wraps them as pytest fixtures.
"""

import tempfile

import pytest
from biopb_tensor_server.fixtures import (
    create_5d_6d_micromanager_dataset,
    create_companion_ome_dataset,
    create_hdf5_dataset,
    create_multi_series_ome_tiff,
    create_multifile_micromanager_dataset,
    create_multifile_ome_dataset,
    create_multiresolution_ome_zarr,
    create_tiled_ome_tiff,
    create_zarr_array,
)

# =============================================================================
# pytest fixtures using the factory functions
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_upstream_client_pool():
    """Isolate the process-wide upstream client pool (biopb/biopb#266 B1).

    Pooled clients are keyed by ``(endpoint, token)`` and live until eviction or
    process exit, so a client dialed at a random port in one test could be
    handed back in a later test that happens to reuse the port. Clear the pool
    around every test to keep them independent (and to close leaked clients)."""
    from biopb_tensor_server.adapters.remote_tensor import _clear_client_pool

    _clear_client_pool()
    yield
    _clear_client_pool()


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
def multifile_5d_6d_mm_dataset(temp_dir):
    """Full 5D/6D MicroManager dataset with position, time, channel, and z dimensions."""
    return create_5d_6d_micromanager_dataset(
        temp_dir,
        n_positions=2,
        n_times=3,
        n_channels=2,
        n_z=4,
    )


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
def multi_series_ome_tiff(temp_dir):
    """Multi-series OME-TIFF file (multiple fields/positions)."""
    return create_multi_series_ome_tiff(temp_dir)


@pytest.fixture
def companion_ome_dataset(temp_dir):
    """Companion OME dataset with .companion.ome file."""
    return create_companion_ome_dataset(temp_dir)


@pytest.fixture
def hdf5_dataset(temp_dir):
    """HDF5 dataset with chunked array."""
    return create_hdf5_dataset(temp_dir)


@pytest.fixture
def simple_zarr_array(temp_dir):
    """Simple Zarr array for basic tests."""
    return create_zarr_array(temp_dir)
