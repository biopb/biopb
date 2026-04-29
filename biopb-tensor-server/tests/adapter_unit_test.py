"""Server-side unit tests for adapters, config, and compute backend.

Tests for adapter behavior, configuration parsing, compute backend selection,
and scaled read planning.
"""

import os
import tempfile
import pytest
import numpy as np

from biopb.tensor import (
    SliceHint,
    TensorReadOptions,
)
from biopb_tensor_server import ZarrAdapter, OmeZarrAdapter
from biopb_tensor_server.config import parse_config
from biopb_tensor_server import base as tensor_adapter


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr
        zarr.open_array
        return True
    except ImportError:
        return False


class TestZarrAdapter:
    """Tests for ZarrAdapter."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_descriptor(self):
        """Test descriptor generation."""
        import zarr

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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_chunk_endpoints(self):
        """Test chunk endpoint generation."""
        import zarr

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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_chunk_endpoints_with_slice_hint(self):
        """Test slice hint filtering."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])

            # Request only top-right quadrant
            slice_hint = SliceHint(start=[0, 50], stop=[50, 100])
            endpoints = adapter.get_chunk_endpoints(slice_hint)

            # Should only return chunks that intersect [0:50, 50:100]
            # Only (0,1) fully intersects
            assert len(endpoints) == 1
            assert list(endpoints[0].bounds.start) == [0, 50]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_chunk_data(self):
        """Test chunk data retrieval."""
        import zarr

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


class TestGetScaledReadPlan:
    """Tests for BackendAdapter.get_scaled_read_plan default implementation."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_default_uses_virtual_scaling(self):
        """Test that default get_scaled_read_plan uses virtual scaling."""
        import zarr

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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_virtual_scaling_with_slice(self):
        """Test virtual scaling with slice hint."""
        import zarr

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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_find_level_for_scale(self):
        """Test _find_level_for_scale parses multiscales correctly."""
        import json
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create OME-Zarr structure with multiscales
            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            root = zarr.open_group(zarr_path, mode='w')

            # Create level arrays
            root.create_dataset('0', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            root.create_dataset('1', shape=(50, 50), chunks=(25, 25), dtype='uint8')
            root.create_dataset('2', shape=(25, 25), chunks=(12, 12), dtype='uint8')

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

            # Open base level and create adapter
            root = zarr.open_group(zarr_path, mode='r')
            base_arr = root['0']
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Test finding levels
            assert adapter._find_level_for_scale((1, 1)) == '0'
            assert adapter._find_level_for_scale((2, 2)) == '1'
            assert adapter._find_level_for_scale((4, 4)) == '2'
            assert adapter._find_level_for_scale((3, 3)) is None  # No match

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_scaled_read_plan_precompute(self):
        """Test get_scaled_read_plan with precompute method."""
        import json
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            root = zarr.open_group(zarr_path, mode='w')

            # Create and populate level arrays
            arr0 = root.create_dataset('0', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            arr1 = root.create_dataset('1', shape=(50, 50), chunks=(25, 25), dtype='uint8')
            arr0[:] = 1
            arr1[:] = 2

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

            root = zarr.open_group(zarr_path, mode='r')
            base_arr = root['0']
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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_precompute_no_match_raises(self):
        """Test that precompute raises error when no matching level."""
        import json
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            root = zarr.open_group(zarr_path, mode='w')

            root.create_dataset('0', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            root.create_dataset('1', shape=(50, 50), chunks=(25, 25), dtype='uint8')

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

            root = zarr.open_group(zarr_path, mode='r')
            base_arr = root['0']
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Request with non-matching scale
            with pytest.raises(ValueError, match="No precomputed level matching"):
                adapter.get_scaled_read_plan(
                    scale_hint=(3, 3),
                    slice_hint=None,
                    read_options=TensorReadOptions(scale_hint=[3, 3], reduction_method='precompute'),
                )

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_precompute_falls_back_to_virtual_for_other_methods(self):
        """Test that non-precompute methods use virtual scaling."""
        import json
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            root = zarr.open_group(zarr_path, mode='w')

            root.create_dataset('0', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            root.create_dataset('1', shape=(50, 50), chunks=(25, 25), dtype='uint8')

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

            root = zarr.open_group(zarr_path, mode='r')
            base_arr = root['0']
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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_convert_slice_to_level(self):
        """Test slice conversion from base to level coordinates."""
        import json
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.ome.zarr')
            root = zarr.open_group(zarr_path, mode='w')

            root.create_dataset('0', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            root.create_dataset('1', shape=(25, 50), chunks=(12, 25), dtype='uint8')

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

            root = zarr.open_group(zarr_path, mode='r')
            base_arr = root['0']
            adapter = OmeZarrAdapter(base_arr, 'test')

            # Test slice conversion
            level_slice = adapter._convert_slice_to_level(
                SliceHint(start=[10, 20], stop=[50, 60]),
                (4, 2)
            )

            assert list(level_slice.start) == [2, 10]  # 10//4=2, 20//2=10
            assert list(level_slice.stop) == [12, 30]  # 50//4=12, 60//2=30