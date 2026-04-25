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
import tempfile
import pytest
import numpy as np

from biopb_tensor_server.adapters.tiff import MultiFileOmeTiffAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter


def get_test_data_dir():
    """Get test data directory from environment."""
    return os.environ.get('BIOPB_TEST_DATA_DIR')


class TestMultiFileOmeTiffFileListParsing:
    """Tests for _parse_file_list_from_metadata method."""

    @pytest.fixture
    def temp_metadata_dir(self):
        """Create a temporary directory with _metadata.txt for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_parse_raw_ome_xml(self, temp_metadata_dir):
        """Test parsing raw OME-XML format."""
        import tifffile
        from pathlib import Path

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
        import tifffile
        from pathlib import Path

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
        import tifffile
        import json
        from pathlib import Path

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
        import tifffile
        from pathlib import Path

        # Create a TIFF file but no _metadata.txt
        tif_path = Path(temp_metadata_dir) / "img_001.ome.tif"
        dummy_data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(str(tif_path), dummy_data)

        adapter = MultiFileOmeTiffAdapter(temp_metadata_dir, "test")
        files = adapter._parse_file_list_from_metadata()

        assert files is None

    def test_partial_dataset_warning(self, temp_metadata_dir):
        """Test that missing files are tracked and warning is issued."""
        import tifffile
        import warnings
        from pathlib import Path

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
            assert "file3.ome.tif" in str(w[0].message)
            assert len(adapter._missing_files) == 1
            assert "file3.ome.tif" in adapter._missing_files

    def test_shape_adjustment_for_partial_dataset(self, temp_metadata_dir):
        """Test that shape is adjusted for partial datasets."""
        import tifffile
        from pathlib import Path

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
        import tifffile
        import tempfile

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

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
        adapter = OmeTiffAdapter(tf, 'test-embedded')

        assert adapter.array_id == 'test-embedded'
        assert adapter.tiff_file is not None
        assert adapter.full_shape == [3, 128, 128]
        assert adapter.chunk_shape == [1, 32, 32]
        assert adapter.dim_labels == ['channel', 'height', 'width']

    def test_get_tensor_descriptor(self, tiled_ome_tiff):
        """Test descriptor with embedded metadata."""
        tf, path = tiled_ome_tiff

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
        adapter = OmeTiffAdapter(tf, 'test-embedded')

        desc = adapter.get_tensor_descriptor()
        assert desc.array_id == 'test-embedded'
        assert list(desc.shape) == [3, 128, 128]
        assert desc.dtype == 'uint16'

    def test_get_metadata_from_embedded_ome_xml(self, tiled_ome_tiff):
        """Test metadata extraction from embedded OME-XML."""
        tf, path = tiled_ome_tiff

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
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

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
        adapter = OmeTiffAdapter(tf, 'test-embedded')

        endpoints = adapter.get_chunk_endpoints()
        # Should have 3 planes * 4 tiles per plane (128x128 / 32x32)
        assert len(endpoints) == 3 * 4 * 4  # 48 tiles total

    def test_get_chunk_data(self, tiled_ome_tiff):
        """Test reading chunk data from tiled OME-TIFF."""
        tf, path = tiled_ome_tiff

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
        adapter = OmeTiffAdapter(tf, 'test-embedded')

        endpoints = adapter.get_chunk_endpoints()
        if endpoints:
            chunk_data = adapter.get_chunk_data(endpoints[0].chunk_id)
            assert chunk_data is not None
            assert len(chunk_data.columns) == 1

            # Tile should be 32x32 = 1024 pixels
            arr = chunk_data.column(0).to_numpy()
            assert arr.shape == (1024,)

    def test_non_tiled_raises_error(self):
        """Test that non-tiled TIFF raises ValueError."""
        import tifffile
        import tempfile

        # Create non-tiled TIFF
        data = np.zeros((64, 64), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            path = f.name

        tifffile.imwrite(path, data, photometric='minisblack')

        try:
            tf = tifffile.TiffFile(path)

            from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
            with pytest.raises(ValueError, match="must be tiled"):
                OmeTiffAdapter(tf, 'test-non-tiled')
        finally:
            tf.close()
            os.unlink(path)

    def test_slice_hint_filters_chunks(self, tiled_ome_tiff):
        """Test that slice_hint correctly filters chunk endpoints."""
        tf, path = tiled_ome_tiff

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
        from biopb.tensor.descriptor_pb2 import SliceHint

        adapter = OmeTiffAdapter(tf, 'test-embedded')

        # Request only channel 0, top-left quadrant (y: 0-64, x: 0-64)
        slice_hint = SliceHint(start=[0, 0, 0], stop=[1, 64, 64])
        endpoints = adapter.get_chunk_endpoints(slice_hint)

        # Should have only tiles in that region: 1 plane * 2 tiles_y * 2 tiles_x = 4
        assert len(endpoints) == 4


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
