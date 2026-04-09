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
    OmeZarrAdapter,
    TensorFlightServer,
    TensorFlightClient,
    TensorDescriptor,
    SliceHint,
    TensorReadOptions,
)
from biopb.tensor import base as tensor_adapter
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

    def test_non_divisible_scaled_nearest_view(self):
        """Test nearest downsampling returns ceil-sized output for edge chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'nearest-edge.zarr')
            source = np.arange(25, dtype=np.uint8).reshape(5, 5)
            arr = zarr.open_array(zarr_path, mode='w', shape=(5, 5), chunks=(3, 3), dtype='uint8')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'nearest-edge', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8892')
            server.register_tensor('nearest-edge', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8892', cache_bytes=10_000_000) as client:
                    darr = client.get_array('nearest-edge', scale_hint=[2, 2], reduction_method='nearest')
                    assert darr.shape == (3, 3)
                    np.testing.assert_array_equal(darr.compute(), source[::2, ::2])
            finally:
                server.shutdown()

    def test_non_divisible_scaled_area_slice_uses_edge_padding(self):
        """Test area downsampling pads from the slice edge rather than reading past it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'area-edge.zarr')
            source = np.arange(8, dtype=np.uint8).reshape(1, 8)
            arr = zarr.open_array(zarr_path, mode='w', shape=(1, 8), chunks=(1, 3), dtype='uint8')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'area-edge', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8895')
            server.register_tensor('area-edge', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8895', cache_bytes=10_000_000) as client:
                    darr = client.get_array(
                        'area-edge',
                        slice_hint=(slice(0, 1), slice(1, 6)),
                        read_options=TensorReadOptions(scale_hint=[1, 2], reduction_method='area'),
                    )
                    assert darr.shape == (1, 3)
                    np.testing.assert_array_equal(
                        darr.compute(),
                        np.array([[2, 4, 5]], dtype=np.uint8),
                    )
            finally:
                server.shutdown()

    def test_non_divisible_scaled_linear_view(self):
        """Test linear downsampling uses ceil-sized output with padded edge support."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'linear-edge.zarr')
            source = np.array([[0, 10, 20, 30, 40]], dtype=np.uint8)
            arr = zarr.open_array(zarr_path, mode='w', shape=(1, 5), chunks=(1, 3), dtype='uint8')
            arr[:] = source

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'linear-edge', ['y', 'x'])
            server = TensorFlightServer('grpc://localhost:8896')
            server.register_tensor('linear-edge', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient('grpc://localhost:8896', cache_bytes=10_000_000) as client:
                    darr = client.get_array(
                        'linear-edge',
                        read_options=TensorReadOptions(scale_hint=[1, 2], reduction_method='linear'),
                    )
                    assert darr.shape == (1, 3)
                    np.testing.assert_array_equal(
                        darr.compute(),
                        np.array([[5, 25, 40]], dtype=np.uint8),
                    )
            finally:
                server.shutdown()


class TestGetScaledReadPlan:
    """Tests for BackendAdapter.get_scaled_read_plan default implementation."""

    def test_default_uses_virtual_scaling(self):
        """Test that default get_scaled_read_plan uses virtual scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            arr[:] = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'test', ['y', 'x'])

            # Test that default implementation produces virtual scaling plan
            plan = adapter.get_scaled_read_plan(
                scale_hint=(2, 2),
                slice_hint=None,
                read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='nearest'),
            )

            # Should have virtual chunks (2x2 grid = 4 chunks)
            assert len(plan.chunk_endpoints) == 4
            assert list(plan.descriptor.shape) == [50, 50]
            assert list(plan.descriptor.chunk_shape) == [25, 25]  # Virtual chunk shape

    def test_virtual_scaling_with_slice(self):
        """Test virtual scaling with slice hint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            arr[:] = 1

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'test', ['y', 'x'])

            plan = adapter.get_scaled_read_plan(
                scale_hint=(2, 2),
                slice_hint=SliceHint(start=[10, 10], stop=[50, 50]),
                read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='nearest'),
            )

            # Shape should be (40/2, 40/2) = (20, 20)
            assert list(plan.descriptor.shape) == [20, 20]


class TestOmeZarrPrecompute:
    """Tests for OmeZarrAdapter precomputed level support."""

    def test_find_level_for_scale(self):
        """Test _find_level_for_scale parses multiscales correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import json

            # Create OME-Zarr structure with multiscales
            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            os.makedirs(zarr_path)

            # Create .zattrs with multiscales
            zattrs = {
                'multiscales': [{
                    'name': 'test',
                    'axes': [{'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
                    'datasets': [
                        {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1]}]},
                        {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2, 2]}]},
                        {'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4, 4]}]},
                    ]
                }]
            }
            with open(os.path.join(zarr_path, '.zattrs'), 'w') as f:
                json.dump(zattrs, f)

            # Create level arrays
            import zarr
            zarr.open_array(os.path.join(zarr_path, '0'), mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            zarr.open_array(os.path.join(zarr_path, '1'), mode='w', shape=(50, 50), chunks=(25, 25), dtype='uint8')
            zarr.open_array(os.path.join(zarr_path, '2'), mode='w', shape=(25, 25), chunks=(12, 12), dtype='uint8')

            # Open base level and create adapter
            base_arr = zarr.open_array(os.path.join(zarr_path, '0'), mode='r')
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Test finding levels
            assert adapter._find_level_for_scale((1, 1)) == '0'
            assert adapter._find_level_for_scale((2, 2)) == '1'
            assert adapter._find_level_for_scale((4, 4)) == '2'
            assert adapter._find_level_for_scale((3, 3)) is None  # No match

    def test_get_scaled_read_plan_precompute(self):
        """Test get_scaled_read_plan with precompute method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            import zarr

            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            os.makedirs(zarr_path)

            zattrs = {
                'multiscales': [{
                    'name': 'test',
                    'axes': [{'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
                    'datasets': [
                        {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1]}]},
                        {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2, 2]}]},
                    ]
                }]
            }
            with open(os.path.join(zarr_path, '.zattrs'), 'w') as f:
                json.dump(zattrs, f)

            # Create and populate level arrays
            arr0 = zarr.open_array(os.path.join(zarr_path, '0'), mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            arr1 = zarr.open_array(os.path.join(zarr_path, '1'), mode='w', shape=(50, 50), chunks=(25, 25), dtype='uint8')
            arr0[:] = 1
            arr1[:] = 2

            base_arr = zarr.open_array(os.path.join(zarr_path, '0'), mode='r')
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Request with precompute method
            plan = adapter.get_scaled_read_plan(
                scale_hint=(2, 2),
                slice_hint=None,
                read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='precompute'),
            )

            # Should return level 1's shape
            assert list(plan.descriptor.shape) == [50, 50]
            assert list(plan.descriptor.chunk_shape) == [25, 25]

    def test_precompute_no_match_raises(self):
        """Test that precompute raises error when no matching level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            import zarr

            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            os.makedirs(zarr_path)

            zattrs = {
                'multiscales': [{
                    'name': 'test',
                    'axes': [{'name': 'y'}, {'name': 'x'}],
                    'datasets': [
                        {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1]}]},
                        {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2, 2]}]},
                    ]
                }]
            }
            with open(os.path.join(zarr_path, '.zattrs'), 'w') as f:
                json.dump(zattrs, f)

            zarr.open_array(os.path.join(zarr_path, '0'), mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            zarr.open_array(os.path.join(zarr_path, '1'), mode='w', shape=(50, 50), chunks=(25, 25), dtype='uint8')

            base_arr = zarr.open_array(os.path.join(zarr_path, '0'), mode='r')
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Request with non-matching scale
            with pytest.raises(ValueError, match="No precomputed level matching"):
                adapter.get_scaled_read_plan(
                    scale_hint=(3, 3),
                    slice_hint=None,
                    read_options=TensorReadOptions(scale_hint=[3, 3], reduction_method='precompute'),
                )

    def test_precompute_falls_back_to_virtual_for_other_methods(self):
        """Test that non-precompute methods use virtual scaling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            import zarr

            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            os.makedirs(zarr_path)

            zattrs = {
                'multiscales': [{
                    'name': 'test',
                    'axes': [{'name': 'y'}, {'name': 'x'}],
                    'datasets': [
                        {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1]}]},
                        {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [2, 2]}]},
                    ]
                }]
            }
            with open(os.path.join(zarr_path, '.zattrs'), 'w') as f:
                json.dump(zattrs, f)

            zarr.open_array(os.path.join(zarr_path, '0'), mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            zarr.open_array(os.path.join(zarr_path, '1'), mode='w', shape=(50, 50), chunks=(25, 25), dtype='uint8')

            base_arr = zarr.open_array(os.path.join(zarr_path, '0'), mode='r')
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Request with nearest method (not precompute)
            plan = adapter.get_scaled_read_plan(
                scale_hint=(2, 2),
                slice_hint=None,
                read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='nearest'),
            )

            # Should use virtual scaling (shape based on base / scale)
            assert list(plan.descriptor.shape) == [50, 50]
            # Virtual chunk shape is base_chunk / scale = 50/2 = 25
            assert list(plan.descriptor.chunk_shape) == [25, 25]


class TestSliceConversion:
    """Tests for slice coordinate conversion."""

    def test_convert_slice_to_level(self):
        """Test slice conversion from base to level coordinates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            import zarr

            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            os.makedirs(zarr_path)

            zattrs = {
                'multiscales': [{
                    'datasets': [
                        {'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1, 1]}]},
                        {'path': '1', 'coordinateTransformations': [{'type': 'scale', 'scale': [4, 2]}]},
                    ]
                }]
            }
            with open(os.path.join(zarr_path, '.zattrs'), 'w') as f:
                json.dump(zattrs, f)

            zarr.open_array(os.path.join(zarr_path, '0'), mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            zarr.open_array(os.path.join(zarr_path, '1'), mode='w', shape=(25, 50), chunks=(12, 25), dtype='uint8')

            base_arr = zarr.open_array(os.path.join(zarr_path, '0'), mode='r')
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Test slice conversion
            level_slice = adapter._convert_slice_to_level(
                SliceHint(start=[10, 20], stop=[50, 60]),
                (4, 2)
            )

            assert list(level_slice.start) == [2, 10]  # 10//4=2, 20//2=10
            assert list(level_slice.stop) == [12, 30]  # 50//4=12, 60//2=30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])