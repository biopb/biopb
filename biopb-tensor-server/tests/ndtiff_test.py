"""Tests for NDTiff adapter for Micro-Manager NDTiff storage format.

Tests cover:
- Claim detection (local and remote)
- Descriptor generation
- Data access via dask array
- Server/client roundtrip
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from biopb.tensor import TensorFlightClient
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.discovery import ClaimContext, DiscoveryState


def _ndtiff_available() -> bool:
    """Check if ndtiff is available."""
    try:
        import ndtiff  # noqa: F401  # availability probe

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _ndtiff_available(), reason="ndtiff not installed")
class TestNdTiffClaim:
    """Tests for NDTiff claim detection."""

    def test_claim_local_ndtiff_directory(self):
        """Claim detects NDTiff.index file."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NDTiff.index file
            index_path = Path(tmpdir) / "NDTiff.index"
            index_path.write_text("{}")  # Minimal JSON

            # Create dummy TIFF files
            for i in range(2):
                tiff_path = Path(tmpdir) / f"NDTiffStack_{i}.tif"
                tiff_path.write_bytes(b"")

            ctx = ClaimContext(Path(tmpdir))
            state = DiscoveryState()

            claim = NdTiffAdapter.claim(ctx, state)

            assert claim is not None
            assert claim.source_type == "ndtiff"
            assert claim.primary_path == str(tmpdir)
            assert claim.is_remote == False

    def test_claim_rejects_non_ndtiff_directory(self):
        """Claim returns None for directories without NDTiff.index."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some TIFF files but no NDTiff.index
            for i in range(3):
                tiff_path = Path(tmpdir) / f"image_{i}.tif"
                tiff_path.write_bytes(b"")

            ctx = ClaimContext(Path(tmpdir))
            state = DiscoveryState()

            claim = NdTiffAdapter.claim(ctx, state)

            assert claim is None

    def test_claim_rejects_files(self):
        """Claim only works on directories."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "NDTiff.index"
            file_path.write_text("{}")

            ctx = ClaimContext(file_path)
            state = DiscoveryState()

            claim = NdTiffAdapter.claim(ctx, state)

            assert claim is None


@pytest.mark.skipif(not _ndtiff_available(), reason="ndtiff not installed")
class TestNdTiffAdapterDescriptor:
    """Tests for NdTiffAdapter descriptor generation."""

    def test_descriptor_from_mock_dataset(self):
        """Test descriptor with mock NDTiffDataset."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        # Mock NDTiffDataset
        mock_dataset = MagicMock()
        mock_axes = MagicMock()
        mock_axes.keys.return_value = [
            "position",
            "time",
            "channel",
            "z",
            "row",
            "column",
        ]
        mock_dataset.axes = mock_axes

        # Mock dask array
        mock_dask = MagicMock()
        mock_dask.shape = (2, 3, 2, 5, 64, 64)
        mock_dask.dtype = np.dtype("uint16")
        mock_dataset.as_array.return_value = mock_dask

        adapter = NdTiffAdapter(
            dataset=mock_dataset,
            source_id="test-ndtiff",
            source_url="/test/path",
            dim_labels=None,
        )

        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == "test-ndtiff"
        assert list(desc.shape) == [2, 3, 2, 5, 64, 64]
        # dtype is string representation - can be either 'uint16' or '<u2'
        assert desc.dtype in ("uint16", "<u2")
        assert list(desc.dim_labels) == ["p", "t", "c", "z", "y", "x"]

    def test_descriptor_with_custom_dim_labels(self):
        """Test descriptor with custom dim_labels override."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        mock_dataset = MagicMock()
        mock_axes = MagicMock()
        mock_axes.keys.return_value = ["time", "channel", "z", "row", "column"]
        mock_dataset.axes = mock_axes

        mock_dask = MagicMock()
        mock_dask.shape = (10, 3, 5, 64, 64)
        mock_dask.dtype = np.dtype("uint8")
        mock_dataset.as_array.return_value = mock_dask

        adapter = NdTiffAdapter(
            dataset=mock_dataset,
            source_id="test-ndtiff",
            source_url="/test/path",
            dim_labels=["t", "c", "z", "y", "x"],  # Custom labels
        )

        desc = adapter.get_tensor_descriptor()

        assert list(desc.dim_labels) == ["t", "c", "z", "y", "x"]

    def test_chunk_shape_is_2d_planes(self):
        """Chunk shape should be one 2D plane per chunk."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        mock_dataset = MagicMock()
        mock_axes = MagicMock()
        mock_axes.keys.return_value = ["position", "channel", "row", "column"]
        mock_dataset.axes = mock_axes

        mock_dask = MagicMock()
        mock_dask.shape = (4, 3, 256, 256)
        mock_dask.dtype = np.dtype("uint16")
        mock_dataset.as_array.return_value = mock_dask

        adapter = NdTiffAdapter(
            dataset=mock_dataset,
            source_id="test-ndtiff",
            source_url="/test/path",
        )

        desc = adapter.get_tensor_descriptor()

        # Chunk shape: (1, 1, Y, X) - one 2D plane
        assert list(desc.chunk_shape) == [1, 1, 256, 256]


@pytest.mark.skipif(not _ndtiff_available(), reason="ndtiff not installed")
class TestNdTiffGetData:
    """Tests for NdTiffAdapter get_data method."""

    def test_get_data_slices_dask_array(self):
        """get_data slices the dask array and computes."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        mock_dataset = MagicMock()
        mock_axes = MagicMock()
        mock_axes.keys.return_value = ["channel", "row", "column"]
        mock_dataset.axes = mock_axes

        # Mock dask array with actual data
        mock_dask = MagicMock()
        mock_dask.shape = (3, 64, 64)
        mock_dask.dtype = np.dtype("uint8")

        # Create slice result
        slice_result = np.arange(20 * 20, dtype=np.uint8).reshape(20, 20)
        mock_slice = MagicMock()
        mock_slice.compute.return_value = slice_result
        mock_dask.__getitem__ = MagicMock(return_value=mock_slice)

        mock_dataset.as_array.return_value = mock_dask

        adapter = NdTiffAdapter(
            dataset=mock_dataset,
            source_id="test-ndtiff",
            source_url="/test/path",
        )

        bounds = ChunkBounds(start=[1, 10, 10], stop=[2, 30, 30])
        data = adapter.get_data(bounds)

        # Verify slice was called
        mock_dask.__getitem__.assert_called_once()
        assert data.shape == (20, 20)
        assert data.dtype == np.uint8


@pytest.mark.skipif(not _ndtiff_available(), reason="ndtiff not installed")
class TestNdTiffServerClient:
    """Integration tests for NDTiff adapter with server/client."""

    def test_server_client_roundtrip(self):
        """Test full roundtrip: register adapter, connect, get_tensor."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        # Mock dataset with actual-like behavior
        mock_dataset = MagicMock()
        mock_axes = MagicMock()
        mock_axes.keys.return_value = ["channel", "row", "column"]
        mock_dataset.axes = mock_axes

        # Create actual numpy data for roundtrip
        full_data = np.arange(3 * 32 * 32, dtype=np.uint8).reshape(3, 32, 32)

        mock_dask = MagicMock()
        mock_dask.shape = (3, 32, 32)
        mock_dask.dtype = np.dtype("uint8")

        def mock_getitem(self, slices):
            result = MagicMock()
            # Compute returns sliced data
            arr_slice = full_data[slices]
            result.compute.return_value = arr_slice
            return result

        mock_dask.__getitem__ = mock_getitem
        mock_dataset.as_array.return_value = mock_dask
        mock_dataset.summary_metadata = {"test": "metadata"}

        adapter = NdTiffAdapter(
            dataset=mock_dataset,
            source_id="ndtiff-test",
            source_url="/test/path",
        )

        # Start server
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ndtiff-test", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # Get tensor - for single-tensor source, tensor_id = source_id
            arr = client.get_tensor("ndtiff-test", "ndtiff-test")

            # Verify shape
            assert arr.shape == (3, 32, 32)

            # Compute and verify data
            data = arr.compute()
            assert data.shape == (3, 32, 32)
            np.testing.assert_array_equal(data, full_data)

            client.close()
        finally:
            server.shutdown()


class TestRemoteNdTiffFileIO:
    """Tests for RemoteNdTiffFileIO wrapper."""

    def test_path_join_function(self):
        """Test path joining logic."""
        from biopb_tensor_server.adapters.ndtiff import RemoteNdTiffFileIO

        mock_store = MagicMock()
        file_io = RemoteNdTiffFileIO(mock_store)

        # Test basic join
        result = file_io.path_join_function("path/a", "b")
        assert result == "path/a/b"

        # Test join with empty first component
        result = file_io.path_join_function("", "file.tif")
        assert result == "file.tif"

        # Test join with trailing slash
        result = file_io.path_join_function("path/", "file.tif")
        assert result == "path/file.tif"


@pytest.mark.skipif(not _ndtiff_available(), reason="ndtiff not installed")
class TestNdTiffCreateFromConfig:
    """Tests for NdTiffAdapter.create_from_config."""

    def test_create_from_local_config(self):
        """Test creating adapter from local SourceConfig."""
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter
        from biopb_tensor_server.config import SourceConfig

        with patch("ndtiff.NDTiffDataset") as mock_ndtiff:
            # Mock dataset
            mock_dataset = MagicMock()
            mock_axes = MagicMock()
            mock_axes.keys.return_value = ["row", "column"]
            mock_dataset.axes = mock_axes

            mock_dask = MagicMock()
            mock_dask.shape = (64, 64)
            mock_dask.dtype = np.dtype("uint8")
            mock_dataset.as_array.return_value = mock_dask

            mock_ndtiff.return_value = mock_dataset

            source = SourceConfig(
                type="ndtiff",
                url="/test/ndtiff/path",
                source_id="test-source",
            )

            adapter = NdTiffAdapter.create_from_config(source)

            assert adapter.source_id == "test-source"
            mock_ndtiff.assert_called_once_with("/test/ndtiff/path")
