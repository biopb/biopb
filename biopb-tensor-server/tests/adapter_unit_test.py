"""Server-side unit tests for adapters, config, and compute backend.

Tests for adapter behavior, configuration parsing, compute backend selection,
and scaled read planning.
"""

import os
import tempfile

import numpy as np
import pytest
from biopb.tensor import (
    SliceHint,
    TensorDescriptor,
)
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server import (
    OmeZarrAdapter,
    ZarrAdapter,
    base as tensor_adapter,
    downsample as _ds,
)
from biopb_tensor_server.config import parse_config


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

    def test_parse_periodic_monitor_settings(self):
        config = parse_config({
            'server': {
                'monitor_mode': 'periodic',
                'rescan_interval': 12,
                'full_rescan_interval': 120,
                'stability_window': 45,
                'stable_rescans_required': 2,
                'probe_open_files': False,
                'aggressive_dir_pruning': True,
            },
            'sources': [],
        })

        assert config.monitor_mode == 'periodic'
        assert config.rescan_interval == 12.0
        assert config.full_rescan_interval == 120.0
        assert config.stability_window == 45.0
        assert config.stable_rescans_required == 2
        assert config.probe_open_files is False
        assert config.aggressive_dir_pruning is True

    def test_parse_legacy_monitor_aliases(self):
        config = parse_config({
            'server': {
                'watcher_type': 'off',
                'poll_interval': 9,
            },
            'sources': [],
        })

        assert config.monitor_mode == 'off'
        assert config.rescan_interval == 9.0
        assert config.full_rescan_interval == 3600.0
        assert config.stability_window == 30.0
        assert config.stable_rescans_required == 0
        assert config.probe_open_files is True
        assert config.aggressive_dir_pruning is False


class TestComputeBackendSelection:
    """Tests for internal CPU/GPU backend heuristics."""

    def test_nearest_prefers_cpu(self, monkeypatch):
        monkeypatch.setattr(_ds, '_HAS_CUPY', True)
        monkeypatch.setattr(_ds, '_get_gpu_free_bytes', lambda: 1 << 30)
        monkeypatch.delenv('BIOPB_TENSOR_FORCE_BACKEND', raising=False)

        backend = _ds._select_compute_backend(
            source_shape=(4096, 4096),
            dtype=np.dtype('uint8'),
            reduction_method='nearest',
            scale_hint=(4, 4),
            merged_chunk_count=8,
        )

        assert backend == 'cpu'

    def test_large_linear_prefers_gpu(self, monkeypatch):
        monkeypatch.setattr(_ds, '_HAS_CUPY', True)
        monkeypatch.setattr(_ds, 'cupy_ndimage', object())
        monkeypatch.setattr(_ds, '_get_gpu_free_bytes', lambda: 1 << 30)
        monkeypatch.delenv('BIOPB_TENSOR_FORCE_BACKEND', raising=False)

        backend = _ds._select_compute_backend(
            source_shape=(4096, 4096),
            dtype=np.dtype('uint16'),
            reduction_method='linear',
            scale_hint=(4, 4),
            merged_chunk_count=8,
        )

        assert backend == 'gpu'

    def test_force_cpu_override(self, monkeypatch):
        monkeypatch.setattr(_ds, '_HAS_CUPY', True)
        monkeypatch.setattr(_ds, '_get_gpu_free_bytes', lambda: 1 << 30)
        monkeypatch.setenv('BIOPB_TENSOR_FORCE_BACKEND', 'cpu')

        backend = _ds._select_compute_backend(
            source_shape=(8192, 8192),
            dtype=np.dtype('uint16'),
            reduction_method='area',
            scale_hint=(4, 4),
            merged_chunk_count=16,
        )

        assert backend == 'cpu'

    def test_force_gpu_falls_back_without_cupy(self, monkeypatch):
        monkeypatch.setattr(_ds, '_HAS_CUPY', False)
        monkeypatch.setenv('BIOPB_TENSOR_FORCE_BACKEND', 'gpu')

        backend = _ds._select_compute_backend(
            source_shape=(8192, 8192),
            dtype=np.dtype('uint16'),
            reduction_method='area',
            scale_hint=(4, 4),
            merged_chunk_count=16,
        )

        assert backend == 'cpu'

    def test_configure_compute_backend_updates_thresholds(self, monkeypatch):
        from biopb_tensor_server import get_compute_backend_options, configure_compute_backend
        monkeypatch.delenv('BIOPB_TENSOR_FORCE_BACKEND', raising=False)
        original = get_compute_backend_options()

        try:
            configure_compute_backend(
                force_backend='gpu',
                gpu_min_input_bytes=123,
                gpu_min_linear_input_bytes=45,
                gpu_memory_safety_factor=7,
                gpu_min_merged_chunks=9,
            )
            options = get_compute_backend_options()
            assert options.force_backend == 'gpu'
            assert options.gpu_min_input_bytes == 123
            assert options.gpu_min_linear_input_bytes == 45
            assert options.gpu_memory_safety_factor == 7
            assert options.gpu_min_merged_chunks == 9
        finally:
            configure_compute_backend(**original.__dict__)


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
            request_desc = TensorDescriptor(
                array_id='test',
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype='uint8',
                scale_hint=[2, 2],
                reduction_method='nearest',
            )
            plan = adapter.get_read_plan(request_desc)

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

            request_desc = TensorDescriptor(
                array_id='test',
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype='uint8',
                slice_hint=SliceHint(start=[10, 10], stop=[50, 50]),
                scale_hint=[2, 2],
                reduction_method='nearest',
            )
            plan = adapter.get_read_plan(request_desc)

            # Slice [10,10]->[50,50] intersects chunk [0,50]x[0,50].
            # Realized (snapped) source bounds = [0,0]->[50,50] -> shape (50/2, 50/2) = (25, 25)
            assert list(plan.descriptor.shape) == [25, 25]


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
            request_desc = TensorDescriptor(
                array_id='test',
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype='uint8',
                scale_hint=[2, 2],
                reduction_method='precompute',
            )
            plan = adapter.get_read_plan(request_desc)

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
            request_desc = TensorDescriptor(
                array_id='test',
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype='uint8',
                scale_hint=[3, 3],
                reduction_method='precompute',
            )
            with pytest.raises(ValueError, match="No precomputed level matching"):
                adapter.get_read_plan(request_desc)

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
            request_desc = TensorDescriptor(
                array_id='test',
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype='uint8',
                scale_hint=[2, 2],
                reduction_method='nearest',
            )
            plan = adapter.get_read_plan(request_desc)

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


class TestGetData:
    """Tests for get_data method across adapters."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_aligned(self):
        """Test ZarrAdapter get_data with aligned bounds."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            arr[:] = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'test', ['y', 'x'])

            # Aligned bounds
            bounds = ChunkBounds(start=[0, 0], stop=[50, 50])
            data = adapter.get_data(bounds)

            assert data.shape == (50, 50)
            assert data.dtype == np.uint8
            # Verify by slicing the full array directly
            full_arr = zarr.open_array(zarr_path, mode='r')
            expected = full_arr[0:50, 0:50]
            np.testing.assert_array_equal(data, expected)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_non_aligned(self):
        """Test ZarrAdapter get_data with non-aligned bounds."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')
            arr[:] = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

            adapter = ZarrAdapter(zarr.open_array(zarr_path, mode='r'), 'test', ['y', 'x'])

            # Non-aligned bounds
            bounds = ChunkBounds(start=[10, 20], stop=[30, 40])
            data = adapter.get_data(bounds)

            assert data.shape == (20, 20)
            # Verify by slicing the full array
            full_arr = zarr.open_array(zarr_path, mode='r')
            expected = full_arr[10:30, 20:40]
            np.testing.assert_array_equal(data, expected)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_out_of_bounds(self):
        """Test ZarrAdapter get_data raises on out-of-bounds."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])

            # Out of bounds
            bounds = ChunkBounds(start=[0, 0], stop=[150, 100])
            with pytest.raises(ValueError, match="exceeds shape"):
                adapter.get_data(bounds)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_negative_start(self):
        """Test ZarrAdapter get_data raises on negative start."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])

            # Negative start
            bounds = ChunkBounds(start=[-10, 0], stop=[50, 50])
            with pytest.raises(ValueError, match="is negative"):
                adapter.get_data(bounds)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_dimensionality_mismatch(self):
        """Test ZarrAdapter get_data raises on wrong dimensionality."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, 'test.zarr')
            arr = zarr.open_array(zarr_path, mode='w', shape=(100, 100), chunks=(50, 50), dtype='uint8')

            adapter = ZarrAdapter(arr, 'test', ['y', 'x'])

            # Wrong dimensionality
            bounds = ChunkBounds(start=[0], stop=[50])
            with pytest.raises(ValueError, match="dimensionality mismatch"):
                adapter.get_data(bounds)