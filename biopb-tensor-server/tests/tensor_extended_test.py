"""Tests for OmeZarrAdapter and AicsImageIoAdapter (OME-TIFF handling).

Uses synthetic test fixtures from conftest.py - no external data required.
"""

import importlib.util
import os
import tempfile

import numpy as np
import pytest
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr

        # Try to actually use zarr to catch numcodecs compatibility issues
        zarr.open_array
        return True
    except ImportError:
        return False


def _h5py_available() -> bool:
    return importlib.util.find_spec("h5py") is not None


class TestAicsImageIoAdapterEmbeddedMetadata:
    """Tests for AicsImageIoAdapter with embedded OME-XML metadata."""

    @pytest.fixture
    def tiled_ome_tiff(self):
        """Create a tiled OME-TIFF with embedded metadata."""

        import tifffile

        data = np.random.randint(0, 65535, (3, 128, 128), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".ome.tif", delete=False) as f:
            path = f.name

        tifffile.imwrite(
            path,
            data,
            photometric="minisblack",
            tile=(32, 32),
            metadata={"axes": "CYX"},
        )

        yield path

        # Cleanup
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def test_adapter_init_with_embedded_metadata(self, tiled_ome_tiff):
        """Test AicsImageIoAdapter initialization with embedded OME-XML."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        path = tiled_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(path, "test-embedded")

        assert adapter.source_id == "test-embedded"
        # AicsImageIoAdapter is multi-tensor, list descriptors to get shape
        descriptors = adapter.list_tensor_descriptors()
        assert len(descriptors) == 1
        # aicsimageio uses TCZYX dimension order, so shape is (T=1, C=3, Z=1, Y=128, X=128)
        assert descriptors[0].shape == [1, 3, 1, 128, 128]

    def test_get_tensor_descriptor(self, tiled_ome_tiff):
        """Test descriptor with embedded metadata."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        path = tiled_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(path, "test-embedded")

        # For multi-scene AicsImageIoAdapter, use list_tensor_descriptors
        descriptors = adapter.list_tensor_descriptors()
        assert len(descriptors) == 1
        desc = descriptors[0]
        # array_id is the globally-unique source_id/field (identity policy); the
        # scene id "Image:0" is the within-source field.
        assert desc.array_id == "test-embedded/Image:0"
        # aicsimageio uses TCZYX dimension order
        assert list(desc.shape) == [1, 3, 1, 128, 128]
        # dtype is numpy dtype string format (e.g., '<u2' for little-endian uint16)
        assert desc.dtype in ("uint16", "<u2")

    def test_get_metadata_from_embedded_ome_xml(self, tiled_ome_tiff):
        """Test metadata extraction from embedded OME-XML."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        path = tiled_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(path, "test-embedded")

        metadata = adapter.get_metadata()
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

        # Should have OME Image element
        image_key = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image"
        assert image_key in metadata or "Image" in metadata or "images" in metadata


class TestOmeZarrAdapter:
    """Tests for OmeZarrAdapter using synthetic fixtures."""

    @pytest.mark.skipif(
        not _zarr_available(), reason="zarr not available or incompatible numcodecs"
    )
    def test_adapter_init_with_synthetic_zarr(self, multires_ome_zarr):
        """Test OmeZarrAdapter initialization with synthetic OME-Zarr."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode="r")
        arr = root["0"]

        adapter = OmeZarrAdapter(arr, "ome-zarr-test")
        assert adapter.array_id == "ome-zarr-test"

    @pytest.mark.skipif(
        not _zarr_available(), reason="zarr not available or incompatible numcodecs"
    )
    def test_get_tensor_descriptor(self, multires_ome_zarr):
        """Test descriptor returns valid shape and dtype."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode="r")
        arr = root["0"]

        adapter = OmeZarrAdapter(arr, "ome-zarr-test")
        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == "ome-zarr-test"
        assert len(desc.shape) > 0
        assert desc.dtype is not None

    @pytest.mark.skipif(
        not _zarr_available(), reason="zarr not available or incompatible numcodecs"
    )
    def test_ome_metadata(self, multires_ome_zarr):
        """Test OME metadata extraction."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode="r")
        arr = root["0"]

        adapter = OmeZarrAdapter(arr, "ome-zarr-test")

        # Should have multiscales metadata
        assert "multiscales" in adapter.ome_metadata
        assert len(adapter.axes) > 0
        print(f"Axes: {adapter.axes}")
        print(f"Dim labels: {adapter.dim_labels}")

    @pytest.mark.skipif(
        not _zarr_available(), reason="zarr not available or incompatible numcodecs"
    )
    def test_get_metadata(self, multires_ome_zarr):
        """Test get_metadata() returns .zattrs content."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode="r")
        arr = root["0"]

        adapter = OmeZarrAdapter(arr, "ome-zarr-test")
        metadata = adapter.get_metadata()

        # Should return the OME-Zarr .zattrs content
        assert isinstance(metadata, dict)
        assert "multiscales" in metadata

        # Multiscales should have axes and datasets
        multiscales = metadata["multiscales"]
        assert len(multiscales) > 0
        print(f"Multiscales datasets: {multiscales[0].get('datasets', [])}")

    @pytest.mark.skipif(
        not _zarr_available(), reason="zarr not available or incompatible numcodecs"
    )
    def test_level_data_values(self, multires_ome_zarr):
        """Test that different levels have distinguishable values."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode="r")

        # Level 0 should have value 0, level 1 should have value 1, etc.
        for level_idx, level_path in enumerate(level_paths):
            level_name = str(level_idx)
            if level_name in root:
                arr = root[level_name]
                # Check that values match level index (created by fixture)
                data = arr[0:10, 0:10]
                assert data.mean() == level_idx


class TestHdf5Adapter:
    """Tests for Hdf5Adapter using synthetic fixtures."""

    @pytest.mark.skipif(not _h5py_available(), reason="h5py not available")
    def test_adapter_init(self, hdf5_dataset):
        """Test Hdf5Adapter initialization."""
        import h5py

        h5_path, shape, chunks = hdf5_dataset

        with h5py.File(h5_path, "r") as f:
            dataset = f["data"]

            from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter

            adapter = Hdf5Adapter(dataset, "hdf5-test")

            assert adapter.array_id == "hdf5-test"

    @pytest.mark.skipif(not _h5py_available(), reason="h5py not available")
    def test_get_tensor_descriptor(self, hdf5_dataset):
        """Test descriptor returns valid shape and dtype."""
        import h5py

        h5_path, shape, chunks = hdf5_dataset

        with h5py.File(h5_path, "r") as f:
            dataset = f["data"]

            from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter

            adapter = Hdf5Adapter(dataset, "hdf5-test")

            desc = adapter.get_tensor_descriptor()
            assert desc.array_id == "hdf5-test"
            assert tuple(desc.shape) == shape
            assert tuple(desc.chunk_shape) == chunks
