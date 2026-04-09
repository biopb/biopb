"""Tests for MultiFileOmeTiffAdapter and OmeZarrAdapter.

These tests require external test data. Set the environment variable
BIOPB_TEST_DATA_DIR to a directory containing test fixtures:

    export BIOPB_TEST_DATA_DIR=/path/to/test/data

Required fixtures:
- mm_test_data/  (Micro-Manager multi-file OME-TIFF)
- test.ome.zarr/ (OME-Zarr dataset)

Tests will be skipped if fixtures are not found.
"""

import os
import pytest
import numpy as np

from biopb_tensor_server.adapters.tiff import MultiFileOmeTiffAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter


def get_test_data_dir():
    """Get test data directory from environment."""
    return os.environ.get('BIOPB_TEST_DATA_DIR')


class TestMultiFileOmeTiffAdapter:
    """Tests for MultiFileOmeTiffAdapter."""

    @pytest.fixture
    def mm_test_dir(self):
        """Fixture for Micro-Manager test data."""
        test_data_dir = get_test_data_dir()
        if not test_data_dir:
            pytest.skip("BIOPB_TEST_DATA_DIR not set")

        # Try direct path first (dir contains MM files)
        if os.path.isdir(test_data_dir):
            has_mm_files = any(
                f.endswith(('.ome.tif', '.ome.tiff', '_metadata.txt'))
                for f in os.listdir(test_data_dir)
            )
            if has_mm_files:
                return test_data_dir

        # Try subdirectory
        mm_dir = os.path.join(test_data_dir, 'mm_test_data')
        if not os.path.isdir(mm_dir):
            pytest.skip(f"mm_test_data not found in {test_data_dir}")

        return mm_dir

    def test_adapter_init(self, mm_test_dir):
        """Test MultiFileOmeTiffAdapter initialization."""
        adapter = MultiFileOmeTiffAdapter(mm_test_dir, 'mm-test')
        assert adapter.array_id == 'mm-test'
        assert adapter.tiff_file is not None

    def test_get_tensor_descriptor(self, mm_test_dir):
        """Test descriptor returns valid shape and dtype."""
        adapter = MultiFileOmeTiffAdapter(mm_test_dir, 'mm-test')
        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == 'mm-test'
        assert len(desc.shape) > 0
        assert desc.dtype is not None
        print(f"Descriptor: shape={desc.shape}, dtype={desc.dtype}")

    def test_get_chunk_endpoints(self, mm_test_dir):
        """Test chunk endpoint generation."""
        adapter = MultiFileOmeTiffAdapter(mm_test_dir, 'mm-test')
        endpoints = adapter.get_chunk_endpoints()

        assert len(endpoints) > 0
        assert all(hasattr(ep, 'chunk_id') for ep in endpoints)
        assert all(hasattr(ep, 'bounds') for ep in endpoints)
        print(f"Generated {len(endpoints)} chunk endpoints")

    def test_get_chunk_data(self, mm_test_dir):
        """Test chunk data retrieval."""
        adapter = MultiFileOmeTiffAdapter(mm_test_dir, 'mm-test', cache_size=16)
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_data = adapter.get_chunk_data(endpoints[0].chunk_id)
            assert chunk_data is not None
            assert len(chunk_data.columns) == 1
            print(f"First chunk: {len(chunk_data)} elements")

    def test_cache_reuse(self, mm_test_dir):
        """Test that LRU cache is working."""
        adapter = MultiFileOmeTiffAdapter(mm_test_dir, 'mm-test', cache_size=16)
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_id = endpoints[0].chunk_id

            # First call (cache miss)
            data1 = adapter._get_decoded_tile(chunk_id)

            # Second call (cache hit)
            data2 = adapter._get_decoded_tile(chunk_id)

            # Should return same data
            np.testing.assert_array_equal(data1, data2)

    def test_get_metadata(self, mm_test_dir):
        """Test metadata extraction from companion file."""
        adapter = MultiFileOmeTiffAdapter(mm_test_dir, 'mm-test')
        metadata = adapter.get_metadata()

        # Should return metadata dict (Micro-Manager JSON or OME-XML)
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

        # Micro-Manager format has Summary section
        if 'Summary' in metadata:
            summary = metadata['Summary']
            assert 'Width' in summary or 'Height' in summary
            print(f"Micro-Manager metadata: Width={summary.get('Width')}, Height={summary.get('Height')}")
        else:
            # OME-XML format - keys may have namespace prefixes
            print(f"OME-XML metadata keys: {list(metadata.keys())[:5]}")


class TestOmeZarrAdapter:
    """Tests for OmeZarrAdapter."""

    @pytest.fixture
    def ome_zarr_dir(self):
        """Fixture for OME-Zarr test data."""
        test_data_dir = get_test_data_dir()
        if not test_data_dir:
            pytest.skip("BIOPB_TEST_DATA_DIR not set")

        zarr_path = os.path.join(test_data_dir, 'test.ome.zarr')
        if not os.path.isdir(zarr_path):
            pytest.skip(f"test.ome.zarr not found in {test_data_dir}")

        return zarr_path

    def test_adapter_init(self, ome_zarr_dir):
        """Test OmeZarrAdapter initialization."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        assert adapter.array_id == 'ome-zarr-test'

    def test_get_tensor_descriptor(self, ome_zarr_dir):
        """Test descriptor returns valid shape and dtype."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == 'ome-zarr-test'
        assert len(desc.shape) > 0
        assert desc.dtype is not None

    def test_ome_metadata(self, ome_zarr_dir):
        """Test OME metadata extraction."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')

        # Should have multiscales metadata
        assert 'multiscales' in adapter.ome_metadata
        assert len(adapter.axes) > 0
        print(f"Axes: {adapter.axes}")
        print(f"Dim labels: {adapter.dim_labels}")

    def test_get_metadata(self, ome_zarr_dir):
        """Test get_metadata() returns .zattrs content."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        metadata = adapter.get_metadata()

        # Should return the OME-Zarr .zattrs content
        assert isinstance(metadata, dict)
        assert 'multiscales' in metadata

        # Multiscales should have axes and datasets
        multiscales = metadata['multiscales']
        assert len(multiscales) > 0
        print(f"Multiscales datasets: {multiscales[0].get('datasets', [])}")

    def test_get_chunk_endpoints(self, ome_zarr_dir):
        """Test chunk endpoint generation."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test', cache_size=16)
        endpoints = adapter.get_chunk_endpoints()

        assert len(endpoints) > 0
        print(f"Generated {len(endpoints)} chunk endpoints")

    def test_get_chunk_data(self, ome_zarr_dir):
        """Test chunk data retrieval."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test', cache_size=16)
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_data = adapter.get_chunk_data(endpoints[0].chunk_id)
            assert chunk_data is not None
            assert len(chunk_data.columns) == 1

    def test_cache_reuse(self, ome_zarr_dir):
        """Test that LRU cache is working."""
        import zarr
        root = zarr.open_group(ome_zarr_dir, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test', cache_size=16)
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_id = endpoints[0].chunk_id

            # First call (cache miss)
            data1 = adapter._get_chunk_data_cached(chunk_id)

            # Second call (cache hit)
            data2 = adapter._get_chunk_data_cached(chunk_id)

            # Should return same data
            np.testing.assert_array_equal(data1, data2)
