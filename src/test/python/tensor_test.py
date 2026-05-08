"""Tests for TensorFlightClient using the new multifield API."""

import threading
import time
import tempfile
import os
import pytest
import numpy as np

from biopb.tensor import (
    TensorFlightClient,
    TensorReadOptions,
)
from biopb_tensor_server import TensorFlightServer, ZarrAdapter


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr
        zarr.open_array
        return True
    except ImportError:
        return False


class TestTensorFlightClient:
    """Tests for TensorFlightClient with new multifield API."""

    @pytest.fixture
    def server_client(self):
        """Start server and create client."""
        if not _zarr_available():
            pytest.skip("zarr not available")

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(128, 128), chunks=(64, 64), dtype='uint8')

            # Set known values per chunk
            arr[:64, :64] = 10
            arr[:64, 64:] = 20
            arr[64:, :64] = 30
            arr[64:, 64:] = 40

            zarr_arr = zarr.open_array(zarr_path, mode='r')
            adapter = ZarrAdapter(zarr_arr, 'test-tensor', ['y', 'x'])

            server = TensorFlightServer('grpc://localhost:8890')
            server.register_source('test-tensor', adapter)

            # Start server in background
            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)  # Wait for server

            try:
                client = TensorFlightClient('grpc://localhost:8890', cache_bytes=10_000_000)
                yield client
                client.close()
            finally:
                server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_list_sources(self, server_client):
        """Test listing sources."""
        sources = server_client.list_sources()
        assert 'test-tensor' in sources

        # DataSourceDescriptor contains tensor metadata
        source_desc = sources['test-tensor']
        assert len(source_desc.tensors) == 1
        assert source_desc.tensors[0].array_id == 'test-tensor'
        assert list(source_desc.tensors[0].shape) == [128, 128]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_shape(self, server_client):
        """Test tensor shape retrieval."""
        darr = server_client.get_tensor('test-tensor', 'test-tensor')
        assert darr.shape == (128, 128)
        assert darr.dtype == np.uint8

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_read_chunks(self, server_client):
        """Test reading different chunks."""
        darr = server_client.get_tensor('test-tensor', 'test-tensor')

        # Top-left chunk
        data = darr[:64, :64].compute()
        assert data.mean() == 10.0

        # Top-right chunk
        data = darr[:64, 64:].compute()
        assert data.mean() == 20.0

        # Bottom-left chunk
        data = darr[64:, :64].compute()
        assert data.mean() == 30.0

        # Bottom-right chunk
        data = darr[64:, 64:].compute()
        assert data.mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_cache_reuse(self, server_client):
        """Test that cache is reused."""
        darr = server_client.get_tensor('test-tensor', 'test-tensor')

        # First read
        data1 = darr[:64, :64].compute()

        # Check cache has entries
        cache_info = server_client.cache_info()
        initial_bytes = cache_info["size_bytes"]

        # Second read - should hit cache
        data2 = darr[:64, :64].compute()

        # size_bytes should be same (cache hit)
        assert server_client.cache_info()["size_bytes"] == initial_bytes

        # Data should be identical
        np.testing.assert_array_equal(data1, data2)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_stride_view(self, server_client):
        """Test explicit per-call scaled reads using stride downsampling."""
        darr = server_client.get_tensor(
            'test-tensor',
            'test-tensor',
            scale_hint=[2, 2],
            reduction_method='stride',
        )

        assert darr.shape == (64, 64)
        assert darr.dtype == np.uint8

        data = darr.compute()
        assert data[:32, :32].mean() == 10.0
        assert data[:32, 32:].mean() == 20.0
        assert data[32:, :32].mean() == 30.0
        assert data[32:, 32:].mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_nearest_view(self, server_client):
        """Test visualization-oriented nearest downsampling."""
        darr = server_client.get_tensor(
            'test-tensor',
            'test-tensor',
            scale_hint=[2, 2],
            reduction_method='nearest',
        )

        assert darr.shape == (64, 64)
        assert darr.dtype == np.uint8

        data = darr.compute()
        assert data[:32, :32].mean() == 10.0
        assert data[:32, 32:].mean() == 20.0
        assert data[32:, :32].mean() == 30.0
        assert data[32:, 32:].mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_mean_view(self, server_client):
        """Test explicit read_options-based mean downsampling."""
        darr = server_client.get_tensor(
            'test-tensor',
            'test-tensor',
            read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='mean'),
        )

        assert darr.shape == (64, 64)
        assert darr.dtype == np.uint8

        data = darr.compute()
        assert data[:32, :32].mean() == 10.0
        assert data[:32, 32:].mean() == 20.0
        assert data[32:, :32].mean() == 30.0
        assert data[32:, 32:].mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_area_view(self, server_client):
        """Test visualization-oriented area downsampling."""
        darr = server_client.get_tensor(
            'test-tensor',
            'test-tensor',
            read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='area'),
        )

        assert darr.shape == (64, 64)
        assert darr.dtype == np.uint8

        data = darr.compute()
        assert data[:32, :32].mean() == 10.0
        assert data[:32, 32:].mean() == 20.0
        assert data[32:, :32].mean() == 30.0
        assert data[32:, 32:].mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_mean_preserves_dtype_with_rounding(self):
        """Test mean downsampling preserves dtype with integer-safe rounding."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'mean-preserve.zarr')
            source = np.array([
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [8, 9, 12, 13],
                [10, 11, 14, 15],
            ], dtype=np.uint8)
            arr = zarr.open_array(zarr_path, mode='w', shape=(4, 4), chunks=(2, 2), dtype='uint8')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'mean-preserve', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8893')
            server.register_source('mean-preserve', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8893', cache_bytes=10_000_000) as client:
                    darr = client.get_tensor(
                        'mean-preserve',
                        'mean-preserve',
                        read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='mean'),
                    )
                    assert darr.dtype == np.uint8
                    np.testing.assert_array_equal(
                        darr.compute(),
                        np.array([[2, 6], [10, 14]], dtype=np.uint8),
                    )
            finally:
                server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_linear_view(self):
        """Test linear interpolation downsampling."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'linear.zarr')
            source = np.array([
                [0, 10, 40, 90],
                [20, 30, 60, 110],
                [80, 90, 120, 170],
                [180, 190, 220, 255],
            ], dtype=np.uint8)
            arr = zarr.open_array(zarr_path, mode='w', shape=(4, 4), chunks=(2, 2), dtype='uint8')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'linear', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8894')
            server.register_source('linear', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8894', cache_bytes=10_000_000) as client:
                    darr = client.get_tensor(
                        'linear',
                        'linear',
                        read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='linear'),
                    )
                    assert darr.dtype == np.uint8
                    np.testing.assert_array_equal(
                        darr.compute(),
                        np.array([[15, 75], [135, 191]], dtype=np.uint8),
                    )
            finally:
                server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_non_divisible_scaled_nearest_view(self):
        """Test nearest downsampling returns ceil-sized output for edge chunks."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'nearest-edge.zarr')
            source = np.arange(25, dtype=np.uint8).reshape(5, 5)
            arr = zarr.open_array(zarr_path, mode='w', shape=(5, 5), chunks=(3, 3), dtype='uint8')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'nearest-edge', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8892')
            server.register_source('nearest-edge', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8892', cache_bytes=10_000_000) as client:
                    darr = client.get_tensor('nearest-edge', 'nearest-edge', scale_hint=[2, 2], reduction_method='nearest')
                    assert darr.shape == (3, 3)
                    np.testing.assert_array_equal(darr.compute(), source[::2, ::2])
            finally:
                server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_tensor_not_found_raises(self, server_client):
        """Test that requesting non-existent tensor raises error."""
        with pytest.raises(ValueError, match="Tensor 'nonexistent' not found"):
            server_client.get_tensor('test-tensor', 'nonexistent')

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_source_not_found_raises(self, server_client):
        """Test that requesting non-existent source raises error."""
        with pytest.raises(ValueError, match="Source not found"):
            server_client.get_tensor('nonexistent-source', 'some-tensor')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])