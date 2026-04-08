"""Tests for TensorFlight server and client."""

import numpy as np
import zarr
import tempfile
import os
import threading
import time
import pytest

from biopb.tensor import (
    ZarrAdapter,
    TensorFlightServer,
    TensorFlightClient,
    TensorDescriptor,
    SliceHint,
    TensorReadOptions,
)
from biopb.tensor import adapter as tensor_adapter
from biopb.tensor.config import parse_config


class TestZarrAdapter:
    """Tests for ZarrAdapter."""

    def test_get_tensor_descriptor(self):
        """Test descriptor generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 200), chunks=(50, 100), dtype='uint16')

            adapter = ZarrAdapter(arr, 'test-array', ['y', 'x'])
            desc = adapter.get_tensor_descriptor()

            assert desc.array_id == 'test-array'
            assert list(desc.shape) == [100, 200]
            assert list(desc.chunk_shape) == [50, 100]
            assert desc.dtype == '<u2'
            assert list(desc.dim_labels) == ['y', 'x']

    def test_get_chunk_endpoints(self):
        """Test chunk endpoint generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])
            endpoints = adapter.get_chunk_endpoints()

            # 2x2 chunks
            assert len(endpoints) == 4

            # Check first chunk bounds
            assert list(endpoints[0].bounds.start) == [0, 0]
            assert list(endpoints[0].bounds.stop) == [50, 50]

    def test_get_chunk_endpoints_with_slice_hint(self):
        """Test slice hint filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])

            # Request only top-right quadrant
            slice_hint = SliceHint(start=[0, 50], stop=[50, 100])
            endpoints = adapter.get_chunk_endpoints(slice_hint)

            # Should only return chunks that intersect [0:50, 50:100]
            # Chunks: (0,0)=[0:50,0:50], (0,1)=[0:50,50:100], (1,0)=[50:100,0:50], (1,1)=[50:100,50:100]
            # Only (0,1) fully intersects
            assert len(endpoints) == 1
            assert list(endpoints[0].bounds.start) == [0, 50]

    def test_get_chunk_data(self):
        """Test chunk data retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            # Set known values
            arr[:50, :50] = 1
            arr[:50, 50:] = 2

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])
            endpoints = adapter.get_chunk_endpoints()

            # Get first chunk data
            batch = adapter.get_chunk_data(endpoints[0].chunk_id)
            data = batch.column(0).to_numpy().reshape(50, 50)

            assert data.mean() == 1.0

            # Get second chunk data
            batch = adapter.get_chunk_data(endpoints[1].chunk_id)
            data = batch.column(0).to_numpy().reshape(50, 50)

            assert data.mean() == 2.0


class TestTensorFlightServer:
    """Tests for TensorFlightServer."""

    def test_register_and_list_tensors(self):
        """Test tensor registration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(64, 64), chunks=(32, 32), dtype='uint8')

            adapter = ZarrAdapter(arr, 'tensor1', ['y', 'x'])

            server = TensorFlightServer('grpc://localhost:8899')
            server.register_tensor('tensor1', adapter)

            # Should be registered
            assert 'tensor1' in server._tensors

            server.unregister_tensor('tensor1')
            assert 'tensor1' not in server._tensors


class TestTensorConfig:
    """Tests for tensor server config parsing."""

    def test_parse_compute_backend_options(self):
        config = parse_config({
            'server': {
                'host': '127.0.0.1',
                'port': 9000,
            },
            'compute': {
                'backend': 'gpu',
                'gpu_min_input_mb': 8,
                'gpu_min_linear_input_mb': 3,
                'gpu_memory_safety_factor': 6,
                'gpu_min_merged_chunks': 5,
            },
            'sources': [],
        })

        assert config.compute_backend == 'gpu'
        assert config.gpu_min_input_mb == 8.0
        assert config.gpu_min_linear_input_mb == 3.0
        assert config.gpu_memory_safety_factor == 6
        assert config.gpu_min_merged_chunks == 5


class TestComputeBackendSelection:
    """Tests for internal CPU/GPU backend heuristics."""

    def test_nearest_prefers_cpu(self, monkeypatch):
        monkeypatch.setattr(tensor_adapter, '_HAS_CUPY', True)
        monkeypatch.setattr(tensor_adapter, '_get_gpu_free_bytes', lambda: 1 << 30)
        monkeypatch.delenv('BIOPB_TENSOR_FORCE_BACKEND', raising=False)

        backend = tensor_adapter._select_compute_backend(
            source_shape=(4096, 4096),
            dtype=np.dtype('uint8'),
            reduction_method='nearest',
            scale_hint=(4, 4),
            merged_chunk_count=8,
        )

        assert backend == 'cpu'

    def test_large_linear_prefers_gpu(self, monkeypatch):
        monkeypatch.setattr(tensor_adapter, '_HAS_CUPY', True)
        monkeypatch.setattr(tensor_adapter, 'cupy_ndimage', object())
        monkeypatch.setattr(tensor_adapter, '_get_gpu_free_bytes', lambda: 1 << 30)
        monkeypatch.delenv('BIOPB_TENSOR_FORCE_BACKEND', raising=False)

        backend = tensor_adapter._select_compute_backend(
            source_shape=(4096, 4096),
            dtype=np.dtype('uint16'),
            reduction_method='linear',
            scale_hint=(4, 4),
            merged_chunk_count=8,
        )

        assert backend == 'gpu'

    def test_force_cpu_override(self, monkeypatch):
        monkeypatch.setattr(tensor_adapter, '_HAS_CUPY', True)
        monkeypatch.setattr(tensor_adapter, '_get_gpu_free_bytes', lambda: 1 << 30)
        monkeypatch.setenv('BIOPB_TENSOR_FORCE_BACKEND', 'cpu')

        backend = tensor_adapter._select_compute_backend(
            source_shape=(8192, 8192),
            dtype=np.dtype('uint16'),
            reduction_method='area',
            scale_hint=(4, 4),
            merged_chunk_count=16,
        )

        assert backend == 'cpu'

    def test_force_gpu_falls_back_without_cupy(self, monkeypatch):
        monkeypatch.setattr(tensor_adapter, '_HAS_CUPY', False)
        monkeypatch.setenv('BIOPB_TENSOR_FORCE_BACKEND', 'gpu')

        backend = tensor_adapter._select_compute_backend(
            source_shape=(8192, 8192),
            dtype=np.dtype('uint16'),
            reduction_method='area',
            scale_hint=(4, 4),
            merged_chunk_count=16,
        )

        assert backend == 'cpu'

    def test_configure_compute_backend_updates_thresholds(self, monkeypatch):
        monkeypatch.delenv('BIOPB_TENSOR_FORCE_BACKEND', raising=False)
        original = tensor_adapter.get_compute_backend_options()

        try:
            tensor_adapter.configure_compute_backend(
                force_backend='gpu',
                gpu_min_input_bytes=123,
                gpu_min_linear_input_bytes=45,
                gpu_memory_safety_factor=7,
                gpu_min_merged_chunks=9,
            )
            options = tensor_adapter.get_compute_backend_options()
            assert options.force_backend == 'gpu'
            assert options.gpu_min_input_bytes == 123
            assert options.gpu_min_linear_input_bytes == 45
            assert options.gpu_memory_safety_factor == 7
            assert options.gpu_min_merged_chunks == 9
        finally:
            tensor_adapter.configure_compute_backend(**original.__dict__)


class TestTensorFlightClient:
    """Integration tests for TensorFlightClient."""

    @pytest.fixture
    def server_client(self):
        """Start server and create client."""
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
            server.register_tensor('test-tensor', adapter)

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

    def test_list_tensors(self, server_client):
        """Test listing tensors."""
        tensors = server_client.list_tensors()
        assert 'test-tensor' in tensors

    def test_get_array_shape(self, server_client):
        """Test array shape retrieval."""
        darr = server_client.get_array('test-tensor')
        assert darr.shape == (128, 128)
        assert darr.dtype == np.uint8

    def test_read_chunks(self, server_client):
        """Test reading different chunks."""
        darr = server_client.get_array('test-tensor')

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

    def test_cache_reuse(self, server_client):
        """Test that cache is reused."""
        darr = server_client.get_array('test-tensor')

        # First read
        data1 = darr[:64, :64].compute()

        # Check cache has entries
        cache_info = server_client._cache
        # cachey Cache has a .nbytes attribute for total bytes
        initial_bytes = cache_info.nbytes

        # Second read - should hit cache
        data2 = darr[:64, :64].compute()

        # nbytes should be same (cache hit)
        assert cache_info.nbytes == initial_bytes

        # Data should be identical
        np.testing.assert_array_equal(data1, data2)

    def test_scaled_stride_view(self, server_client):
        """Test explicit per-call scaled reads using stride downsampling."""
        darr = server_client.get_array(
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

    def test_scaled_nearest_view(self, server_client):
        """Test visualization-oriented nearest downsampling alias."""
        darr = server_client.get_array(
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

    def test_scaled_mean_view(self, server_client):
        """Test explicit read_options-based mean downsampling."""
        darr = server_client.get_array(
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

    def test_scaled_area_view(self, server_client):
        """Test visualization-oriented area downsampling."""
        darr = server_client.get_array(
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

    def test_scaled_mean_view_rounds_and_preserves_dtype(self):
        """Test mean downsampling preserves dtype with integer-safe rounding."""
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
            server.register_tensor('mean-preserve', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8893', cache_bytes=10_000_000) as client:
                    darr = client.get_array(
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

    def test_scaled_linear_view(self):
        """Test linear interpolation downsampling for visualization."""
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
            server.register_tensor('linear', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8894', cache_bytes=10_000_000) as client:
                    darr = client.get_array(
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

    def test_scaled_view_merges_small_chunks(self):
        """Test scaled reads when source chunks must be merged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'merged.zarr')
            source = np.arange(8 * 8, dtype=np.uint16).reshape(8, 8)
            arr = zarr.open_array(zarr_path, mode='w', shape=(8, 8), chunks=(1, 4), dtype='uint16')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'merged', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8891')
            server.register_tensor('merged', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8891', cache_bytes=10_000_000) as client:
                    darr = client.get_array('merged', scale_hint=[2, 2], reduction_method='stride')
                    np.testing.assert_array_equal(darr.compute(), source[::2, ::2])
            finally:
                server.shutdown()

    def test_non_divisible_scaled_view_raises(self):
        """Test unsupported chunk/scale combinations fail fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'invalid.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(12, 12), chunks=(6, 6), dtype='uint8')
            arr[:] = 1

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'invalid', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8892')
            server.register_tensor('invalid', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8892', cache_bytes=10_000_000) as client:
                    with pytest.raises(Exception):
                        client.get_array('invalid', scale_hint=[4, 4], reduction_method='stride')
            finally:
                server.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])