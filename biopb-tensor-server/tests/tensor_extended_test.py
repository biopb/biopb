"""Tests for MultiFileOmeTiffAdapter and OmeZarrAdapter.

Uses synthetic test fixtures from conftest.py - no external data required.
"""

import importlib.util
import os
import tempfile

import numpy as np
import pytest

from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.tiff import MultiFileOmeTiffAdapter, OmeTiffAdapter


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

class TestMultiFileOmeTiffFileListParsing:
    """Tests for _parse_file_list_from_metadata method."""

    @pytest.fixture
    def temp_metadata_dir(self):
        """Create a temporary directory with _metadata.txt for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_parse_raw_ome_xml(self, temp_metadata_dir):
        """Test parsing raw OME-XML format."""
        from pathlib import Path

        import tifffile

        # Create _metadata.txt with OME-XML
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="test">
    <TiffData FirstC="0" FirstT="0" FirstZ="0">
      <UUID FileName="img_001.ome.tif">urn:uuid:abc123</UUID>
    </TiffData>
    <TiffData FirstC="1" FirstT="0" FirstZ="0">
      <UUID FileName="img_002.ome.tif">urn:uuid:def456</UUID>
    </TiffData>
  </Image>
</OME>"""
        metadata_path = Path(temp_metadata_dir) / "_metadata.txt"
        metadata_path.write_text(ome_xml)

        # Create minimal TIFF file to prevent ValueError
        tif_path1 = Path(temp_metadata_dir) / "img_001.ome.tif"
        tif_path2 = Path(temp_metadata_dir) / "img_002.ome.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path1), dummy_data)
        tifffile.imwrite(str(tif_path2), dummy_data)

        # Test parsing
        adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")
        files = adapter._parse_file_list_from_metadata()

        assert files is not None
        assert len(files) == 2
        assert "img_001.ome.tif" in files
        assert "img_002.ome.tif" in files

    def test_parse_json_embedded_ome_xml(self, temp_metadata_dir):
        """Test parsing Micro-Manager JSON format with embedded OME-XML."""
        from pathlib import Path

        import tifffile

        # Create _metadata.txt with JSON format
        ome_xml = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <TiffData>
      <UUID FileName="channel0.ome.tif">urn:uuid:123</UUID>
    </TiffData>
    <TiffData>
      <UUID FileName="channel1.ome.tif">urn:uuid:456</UUID>
    </TiffData>
  </Image>
</OME>"""
        import json
        metadata_content = json.dumps({"OME": ome_xml, "Summary": {"Width": 512}})
        metadata_path = Path(temp_metadata_dir) / "_metadata.txt"
        metadata_path.write_text(metadata_content)

        # Create minimal TIFF files
        tif_path1 = Path(temp_metadata_dir) / "channel0.ome.tif"
        tif_path2 = Path(temp_metadata_dir) / "channel1.ome.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path1), dummy_data)
        tifffile.imwrite(str(tif_path2), dummy_data)

        adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")
        files = adapter._parse_file_list_from_metadata()

        assert files is not None
        assert len(files) == 2
        assert "channel0.ome.tif" in files
        assert "channel1.ome.tif" in files

    def test_parse_pure_json_coords_keys(self, temp_metadata_dir):
        """Test parsing Micro-Manager standard metadata.txt with Coords keys."""
        import json
        from pathlib import Path

        import tifffile

        # Create metadata.txt with Micro-Manager standard format
        metadata_content = json.dumps({
            "Summary": {"Channels": 2},
            "Coords-Default/img_001.tif": {"ChannelIndex": 0},
            "Metadata-Default/img_001.tif": {"UUID": "abc"},
            "Coords-Default/img_002.tif": {"ChannelIndex": 1},
            "Metadata-Default/img_002.tif": {"UUID": "def"},
        })
        metadata_path = Path(temp_metadata_dir) / "metadata.txt"
        metadata_path.write_text(metadata_content)

        # Create TIFF files
        tif_path1 = Path(temp_metadata_dir) / "img_001.tif"
        tif_path2 = Path(temp_metadata_dir) / "img_002.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path1), dummy_data)
        tifffile.imwrite(str(tif_path2), dummy_data)

        adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")
        files = adapter._parse_file_list_from_metadata()

        assert files is not None
        assert len(files) == 2  # Deduplicated
        assert "img_001.tif" in files
        assert "img_002.tif" in files

    def test_parse_missing_metadata(self, temp_metadata_dir):
        """Test that None is returned when no metadata file exists."""
        from pathlib import Path

        import tifffile

        # Create a TIFF file but no _metadata.txt
        tif_path = Path(temp_metadata_dir) / "img_001.ome.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path), dummy_data)

        adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")
        files = adapter._parse_file_list_from_metadata()

        assert files is None

    def test_partial_dataset_warning(self, temp_metadata_dir):
        """Test that missing files are tracked and warning is issued."""
        import warnings
        from pathlib import Path

        import tifffile

        # Create _metadata.txt with OME-XML referencing 3 files
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <TiffData>
      <UUID FileName="file1.ome.tif">urn:uuid:1</UUID>
    </TiffData>
    <TiffData>
      <UUID FileName="file2.ome.tif">urn:uuid:2</UUID>
    </TiffData>
    <TiffData>
      <UUID FileName="file3.ome.tif">urn:uuid:3</UUID>
    </TiffData>
  </Image>
</OME>"""
        metadata_path = Path(temp_metadata_dir) / "_metadata.txt"
        metadata_path.write_text(ome_xml)

        # Create only 2 of the 3 files
        tif_path1 = Path(temp_metadata_dir) / "file1.ome.tif"
        tif_path2 = Path(temp_metadata_dir) / "file2.ome.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path1), dummy_data)
        tifffile.imwrite(str(tif_path2), dummy_data)
        # file3.ome.tif is NOT created

        # Should warn about missing file3
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")

            # Find the specific warning about missing files
            missing_warnings = [
                warning for warning in w
                if "missing files" in str(warning.message).lower()
            ]
            assert len(missing_warnings) >= 1
            assert "file3.ome.tif" in str(missing_warnings[0].message)
            assert len(adapter._missing_files) == 1
            assert "file3.ome.tif" in adapter._missing_files

    def test_shape_adjustment_for_partial_dataset(self, temp_metadata_dir):
        """Test that shape is adjusted for partial datasets."""
        from pathlib import Path

        import tifffile

        # Create _metadata.txt
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <TiffData>
      <UUID FileName="a.ome.tif">urn:uuid:1</UUID>
    </TiffData>
    <TiffData>
      <UUID FileName="b.ome.tif">urn:uuid:2</UUID>
    </TiffData>
  </Image>
</OME>"""
        metadata_path = Path(temp_metadata_dir) / "_metadata.txt"
        metadata_path.write_text(ome_xml)

        # Create only 1 file
        tif_path = Path(temp_metadata_dir) / "a.ome.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path), dummy_data)

        adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")

        # Shape should reflect available data (1 plane)
        desc = adapter.get_tensor_descriptor()
        assert desc.shape[0] == 1  # Only 1 plane available


class TestOmeTiffAdapterEmbeddedMetadata:
    """Tests for OmeTiffAdapter with embedded OME-XML metadata."""

    @pytest.fixture
    def tiled_ome_tiff(self):
        """Create a tiled OME-TIFF with embedded metadata."""
        import tempfile

        import tifffile

        data = np.random.randint(0, 65535, (3, 128, 128), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix='.ome.tif', delete=False) as f:
            path = f.name

        tifffile.imwrite(
            path,
            data,
            photometric='minisblack',
            tile=(32, 32),
            metadata={'axes': 'CYX'},
        )

        yield tifffile.TiffFile(path), path

        # Cleanup
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def test_adapter_init_with_embedded_metadata(self, tiled_ome_tiff):
        """Test OmeTiffAdapter initialization with embedded OME-XML."""
        tf, path = tiled_ome_tiff

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        assert adapter.array_id == 'test-embedded'
        assert adapter.tiff_file is not None
        assert adapter.full_shape == [3, 128, 128]
        assert adapter.chunk_shape == [1, 32, 32]
        assert adapter.dim_labels == ['channel', 'height', 'width']

    def test_get_tensor_descriptor(self, tiled_ome_tiff):
        """Test descriptor with embedded metadata."""
        tf, path = tiled_ome_tiff

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        desc = adapter.get_tensor_descriptor()
        assert desc.array_id == 'test-embedded'
        assert list(desc.shape) == [3, 128, 128]
        assert desc.dtype == 'uint16'

    def test_get_metadata_from_embedded_ome_xml(self, tiled_ome_tiff):
        """Test metadata extraction from embedded OME-XML."""
        tf, path = tiled_ome_tiff

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        metadata = adapter.get_metadata()
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

        # Should have OME Image element
        image_key = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image'
        assert image_key in metadata or 'Image' in metadata

    def test_get_chunk_endpoints(self, tiled_ome_tiff):
        """Test chunk endpoint generation for tiled OME-TIFF."""
        tf, path = tiled_ome_tiff

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        endpoints = adapter.get_chunk_endpoints()
        # Should have 3 planes * 4 tiles per plane (128x128 / 32x32)
        assert len(endpoints) == 3 * 4 * 4  # 48 tiles total

    def test_get_chunk_array(self, tiled_ome_tiff):
        """Test reading chunk array from tiled OME-TIFF."""
        tf, path = tiled_ome_tiff

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        endpoints = adapter.get_chunk_endpoints()
        if endpoints:
            chunk_arr = adapter.get_chunk_array(endpoints[0].chunk_id)
            assert chunk_arr is not None

            # Raw decoded shape from tifffile is (1, tile_h, tile_w, 1)
            # Defensive reshape in resolve_chunk_data handles matching to chunk bounds
            assert chunk_arr.size == 32 * 32

    def test_non_tiled_raises_error(self):
        """Test that non-tiled TIFF raises ValueError."""
        import tempfile

        import tifffile

        # Create non-tiled TIFF
        data = np.zeros((64, 64), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            path = f.name

        tifffile.imwrite(path, data, photometric='minisblack')

        try:
            tf = tifffile.TiffFile(path)

            with pytest.raises(ValueError, match="must be tiled"):
                OmeTiffAdapter(tf, 'test-non-tiled')
        finally:
            tf.close()
            os.unlink(path)

    def test_slice_hint_filters_chunks(self, tiled_ome_tiff):
        """Test that slice_hint correctly filters chunk endpoints."""
        tf, path = tiled_ome_tiff

        from biopb.tensor.descriptor_pb2 import SliceHint

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        # Request only channel 0, top-left quadrant (y: 0-64, x: 0-64)
        slice_hint = SliceHint(start=[0, 0, 0], stop=[1, 64, 64])
        endpoints = adapter.get_chunk_endpoints(slice_hint)

        # Should have only tiles in that region: 1 plane * 2 tiles_y * 2 tiles_x = 4
        assert len(endpoints) == 4


class TestMultiFileOmeTiffAdapter:
    """Tests for MultiFileOmeTiffAdapter using synthetic fixtures."""

    def test_adapter_init_with_synthetic_dataset(self, multifile_ome_dataset):
        """Test MultiFileOmeTiffAdapter initialization with synthetic dataset."""
        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-test')
        assert adapter.array_id == 'mm-test'
        assert adapter.tiff_file is not None

    def test_get_tensor_descriptor(self, multifile_ome_dataset):
        """Test descriptor returns valid shape and dtype."""
        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-test')
        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == 'mm-test'
        assert len(desc.shape) > 0
        assert desc.dtype is not None
        print(f"Descriptor: shape={desc.shape}, dtype={desc.dtype}")

    def test_get_chunk_endpoints(self, multifile_ome_dataset):
        """Test chunk endpoint generation."""
        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-test')
        endpoints = adapter.get_chunk_endpoints()

        assert len(endpoints) > 0
        assert all(hasattr(ep, 'chunk_id') for ep in endpoints)
        assert all(hasattr(ep, 'bounds') for ep in endpoints)
        print(f"Generated {len(endpoints)} chunk endpoints")

    def test_get_chunk_array(self, multifile_ome_dataset):
        """Test chunk array retrieval."""
        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-test')
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_arr = adapter.get_chunk_array(endpoints[0].chunk_id)
            assert chunk_arr is not None
            print(f"First chunk: shape {chunk_arr.shape}")

    def test_cache_reuse(self, multifile_ome_dataset):
        """Test that repeated reads return same data."""
        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-test')
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_id = endpoints[0].chunk_id

            # First call
            arr1 = adapter.get_chunk_array(chunk_id)

            # Second call
            arr2 = adapter.get_chunk_array(chunk_id)

            # Should return same data
            np.testing.assert_array_equal(arr1, arr2)

    def test_get_metadata(self, multifile_ome_dataset):
        """Test metadata extraction from companion file."""
        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-test')
        metadata = adapter.get_metadata()

        # Should return metadata dict (OME-XML or Micro-Manager format)
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

        print(f"Metadata keys: {list(metadata.keys())[:5]}")

    def test_incomplete_dataset_handling(self, multifile_ome_dataset_incomplete):
        """Test handling of incomplete multi-file dataset."""
        import warnings

        dir_path, file_list, metadata = multifile_ome_dataset_incomplete

        # Should warn about missing files but still initialize
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-incomplete')

            assert adapter.array_id == 'mm-incomplete'
            assert len(adapter._missing_files) == 1

            # Shape should reflect available files
            desc = adapter.get_tensor_descriptor()
            assert desc.shape[0] == len(file_list)

    def test_micromanager_format_dataset(self, multifile_mm_dataset):
        """Test adapter with Micro-Manager JSON format metadata."""
        dir_path, file_list, metadata = multifile_mm_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-json-test')
        assert adapter.array_id == 'mm-json-test'

        # Should parse metadata correctly
        parsed_metadata = adapter.get_metadata()
        assert isinstance(parsed_metadata, dict)
        assert 'Summary' in parsed_metadata

        # Verify channels count
        summary = parsed_metadata['Summary']
        assert summary.get('Channels') == 3

    def test_micromanager_dim_labels_use_channel_axis(self, multifile_mm_dataset):
        """Micro-Manager metadata should map leading non-spatial axis to channel."""
        dir_path, file_list, metadata = multifile_mm_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-dimlabels-test')
        desc = adapter.get_tensor_descriptor()

        assert desc.dim_labels[0] == 'c'
        assert desc.dim_labels[-2:] == ['y', 'x']

    def test_data_values_per_channel(self, multifile_mm_dataset):
        """Test that different channels have distinguishable values."""
        dir_path, file_list, metadata = multifile_mm_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-values-test')
        endpoints = adapter.get_chunk_endpoints()

        # Get first chunk from first and second channels
        if len(endpoints) >= 2:
            arr0 = adapter.get_chunk_array(endpoints[0].chunk_id)
            arr1 = adapter.get_chunk_array(endpoints[1].chunk_id)

            # Values should differ (channel 0 = 100, channel 1 = 101)
            # Data values should be different
            assert arr0.mean() != arr1.mean()


class TestOmeZarrAdapter:
    """Tests for OmeZarrAdapter using synthetic fixtures."""

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_adapter_init_with_synthetic_zarr(self, multires_ome_zarr):
        """Test OmeZarrAdapter initialization with synthetic OME-Zarr."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        assert adapter.array_id == 'ome-zarr-test'

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_get_tensor_descriptor(self, multires_ome_zarr):
        """Test descriptor returns valid shape and dtype."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == 'ome-zarr-test'
        assert len(desc.shape) > 0
        assert desc.dtype is not None

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_ome_metadata(self, multires_ome_zarr):
        """Test OME metadata extraction."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')

        # Should have multiscales metadata
        assert 'multiscales' in adapter.ome_metadata
        assert len(adapter.axes) > 0
        print(f"Axes: {adapter.axes}")
        print(f"Dim labels: {adapter.dim_labels}")

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_get_metadata(self, multires_ome_zarr):
        """Test get_metadata() returns .zattrs content."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
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

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_get_chunk_endpoints(self, multires_ome_zarr):
        """Test chunk endpoint generation."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        endpoints = adapter.get_chunk_endpoints()

        assert len(endpoints) > 0
        print(f"Generated {len(endpoints)} chunk endpoints")

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_get_chunk_array(self, multires_ome_zarr):
        """Test chunk array retrieval."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_arr = adapter.get_chunk_array(endpoints[0].chunk_id)
            assert chunk_arr is not None

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_cache_reuse(self, multires_ome_zarr):
        """Test that repeated reads return same data."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-test')
        endpoints = adapter.get_chunk_endpoints()

        if endpoints:
            chunk_id = endpoints[0].chunk_id

            # First call
            arr1 = adapter.get_chunk_array(chunk_id)

            # Second call
            arr2 = adapter.get_chunk_array(chunk_id)

            # Should return same data
            np.testing.assert_array_equal(arr1, arr2)

    @pytest.mark.skipif(
        not _zarr_available(),
        reason="zarr not available or incompatible numcodecs"
    )
    def test_level_data_values(self, multires_ome_zarr):
        """Test that different levels have distinguishable values."""
        import zarr

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')

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

    @pytest.mark.skipif(
        not _h5py_available(),
        reason="h5py not available"
    )
    def test_adapter_init(self, hdf5_dataset):
        """Test Hdf5Adapter initialization."""
        import h5py

        h5_path, shape, chunks = hdf5_dataset

        with h5py.File(h5_path, 'r') as f:
            dataset = f['data']

            from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
            adapter = Hdf5Adapter(dataset, 'hdf5-test')

            assert adapter.array_id == 'hdf5-test'

    @pytest.mark.skipif(
        not _h5py_available(),
        reason="h5py not available"
    )
    def test_get_tensor_descriptor(self, hdf5_dataset):
        """Test descriptor returns valid shape and dtype."""
        import h5py

        h5_path, shape, chunks = hdf5_dataset

        with h5py.File(h5_path, 'r') as f:
            dataset = f['data']

            from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
            adapter = Hdf5Adapter(dataset, 'hdf5-test')

            desc = adapter.get_tensor_descriptor()
            assert desc.array_id == 'hdf5-test'
            assert tuple(desc.shape) == shape
            assert tuple(desc.chunk_shape) == chunks

    @pytest.mark.skipif(
        not _h5py_available(),
        reason="h5py not available"
    )
    def test_get_chunk_endpoints(self, hdf5_dataset):
        """Test chunk endpoint generation."""
        import h5py

        h5_path, shape, chunks = hdf5_dataset

        with h5py.File(h5_path, 'r') as f:
            dataset = f['data']

            from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
            adapter = Hdf5Adapter(dataset, 'hdf5-test')

            endpoints = adapter.get_chunk_endpoints()

            # Should have (100/50) * (100/50) = 4 chunks
            expected_chunks = (shape[0] // chunks[0]) * (shape[1] // chunks[1])
            assert len(endpoints) == expected_chunks

    @pytest.mark.skipif(
        not _h5py_available(),
        reason="h5py not available"
    )
    def test_get_chunk_array(self, hdf5_dataset):
        """Test chunk array retrieval."""
        import h5py

        h5_path, shape, chunks = hdf5_dataset

        with h5py.File(h5_path, 'r') as f:
            dataset = f['data']

            from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
            adapter = Hdf5Adapter(dataset, 'hdf5-test')

            endpoints = adapter.get_chunk_endpoints()

            if endpoints:
                chunk_arr = adapter.get_chunk_array(endpoints[0].chunk_id)
                assert chunk_arr is not None

                # Chunk should be chunks[0] x chunks[1]
                assert chunk_arr.shape == (chunks[0], chunks[1])