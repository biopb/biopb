"""Integration tests for adapters with server, client, and dask compute.

Tests the full pipeline from adapter registration through client access
to dask compute for each adapter type.
"""

import importlib.util
import threading
import time

import numpy as np
import pytest
from biopb.tensor import (
    TensorFlightClient,
    TensorReadOptions,
)

from biopb_tensor_server import TensorFlightServer


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr
        zarr.open_array
        return True
    except ImportError:
        return False


def _h5py_available() -> bool:
    """Check if h5py is available."""
    return importlib.util.find_spec("h5py") is not None


class TestZarrIntegration:
    """Integration tests for ZarrAdapter with server/client."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_server_client_roundtrip(self, simple_zarr_array):
        """Test basic server -> client -> dask compute workflow."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        # Open array and create adapter
        arr = zarr.open_array(zarr_path, mode='r')
        adapter = ZarrAdapter(arr, 'zarr-integration', ['y', 'x'])

        # Start server
        server = TensorFlightServer('grpc://localhost:8899')
        server.register_source('zarr-integration', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            # Connect client and compute
            client = TensorFlightClient('grpc://localhost:8899', cache_bytes=10_000_000)

            # List sources
            sources = client.list_sources()
            assert 'zarr-integration' in sources

            # Get tensor (source_id matches tensor_id for single-tensor sources)
            darr = client.get_tensor('zarr-integration', 'zarr-integration')
            assert darr.shape == shape
            assert darr.dtype == np.uint8

            # Compute data
            data = darr.compute()
            assert data.shape == shape

            client.close()
        finally:
            server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_read_integration(self, simple_zarr_array):
        """Test scaled reads through server/client."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        arr = zarr.open_array(zarr_path, mode='r')
        adapter = ZarrAdapter(arr, 'zarr-scaled', ['y', 'x'])

        server = TensorFlightServer('grpc://localhost:8898')
        server.register_source('zarr-scaled', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8898', cache_bytes=10_000_000)

            # Test stride downsampling
            darr = client.get_tensor('zarr-scaled', 'zarr-scaled', scale_hint=[2, 2], reduction_method='stride')
            expected_shape = (shape[0] // 2, shape[1] // 2)
            assert darr.shape == expected_shape

            data = darr.compute()
            assert data.shape == expected_shape

            client.close()
        finally:
            server.shutdown()


class TestOmeZarrIntegration:
    """Integration tests for OmeZarrAdapter with server/client."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_precompute_level_access(self, multires_ome_zarr):
        """Test accessing precomputed pyramid levels."""
        import zarr

        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-integration')

        server = TensorFlightServer('grpc://localhost:8897')
        server.register_source('ome-zarr-integration', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8897', cache_bytes=10_000_000)

            # Test precompute method for scale 2
            darr = client.get_tensor('ome-zarr-integration', 'ome-zarr-integration',
                read_options=TensorReadOptions(scale_hint=[2, 2], reduction_method='precompute')
            )

            # Level 1 shape should be base_shape / 2
            base_shape = (256, 256)
            expected_shape = (base_shape[0] // 2, base_shape[1] // 2)
            assert darr.shape == expected_shape

            data = darr.compute()
            assert data.shape == expected_shape

            # Level 1 data should have value 1 (set by fixture)
            assert data.mean() == 1

            client.close()
        finally:
            server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_virtual_scaling_with_ome_zarr(self, multires_ome_zarr):
        """Test virtual scaling when no matching precomputed level."""
        import zarr

        from biopb_tensor_server import OmeZarrAdapter

        zarr_path, level_paths, zattrs = multires_ome_zarr

        root = zarr.open_group(zarr_path, mode='r')
        arr = root['0']

        adapter = OmeZarrAdapter(arr, 'ome-zarr-virtual')

        server = TensorFlightServer('grpc://localhost:8896')
        server.register_source('ome-zarr-virtual', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8896', cache_bytes=10_000_000)

            # Request scale 3 - no matching level, should use virtual scaling
            darr = client.get_tensor('ome-zarr-virtual', 'ome-zarr-virtual', scale_hint=[3, 3], reduction_method='nearest')

            # Virtual scaling gives ceil(shape/3)
            base_shape = (256, 256)
            expected_shape = (
                (base_shape[0] + 2) // 3,  # ceil division
                (base_shape[1] + 2) // 3
            )
            assert darr.shape == expected_shape

            data = darr.compute()
            assert data.shape == expected_shape

            client.close()
        finally:
            server.shutdown()


class TestOmeTiffIntegration:
    """Integration tests for OmeTiffAdapter with server/client."""

    def test_tiled_tiff_read(self, tiled_ome_tiff):
        """Test reading from tiled OME-TIFF through server."""
        import tifffile

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter

        tiff_path, shape, tile_info = tiled_ome_tiff

        tf = tifffile.TiffFile(tiff_path)
        adapter = OmeTiffAdapter(tf, 'ome-tiff-integration')

        server = TensorFlightServer('grpc://localhost:8895')
        server.register_source('ome-tiff-integration', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8895', cache_bytes=10_000_000)

            darr = client.get_tensor('ome-tiff-integration', 'ome-tiff-integration')
            assert darr.shape == shape
            assert darr.dtype == np.uint16

            # Read specific region
            data = darr[:1, :32, :32].compute()
            assert data.shape == (1, 32, 32)

            # Value should be 1 (plane 0 = value 1 set by fixture)
            assert data.mean() == 1

            client.close()
        finally:
            server.shutdown()
            tf.close()

    def test_channel_access(self, tiled_ome_tiff):
        """Test accessing different channels through server."""
        import tifffile

        from biopb_tensor_server.adapters.tiff import OmeTiffAdapter

        tiff_path, shape, tile_info = tiled_ome_tiff

        tf = tifffile.TiffFile(tiff_path)
        adapter = OmeTiffAdapter(tf, 'ome-tiff-channels')

        server = TensorFlightServer('grpc://localhost:8894')
        server.register_source('ome-tiff-channels', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8894', cache_bytes=10_000_000)

            darr = client.get_tensor('ome-tiff-channels', 'ome-tiff-channels')

            # Read channel 0 (value = 1)
            data0 = darr[0:1].compute()
            assert data0.mean() == 1

            # Read channel 1 (value = 2)
            data1 = darr[1:2].compute()
            assert data1.mean() == 2

            # Read channel 2 (value = 3)
            data2 = darr[2:3].compute()
            assert data2.mean() == 3

            client.close()
        finally:
            server.shutdown()
            tf.close()


class TestMultiFileOmeTiffIntegration:
    """Integration tests for MultiFileOmeTiffAdapter with server/client."""

    def test_multifile_read(self, multifile_ome_dataset):
        """Test reading from multi-file OME-TIFF through server."""
        from biopb_tensor_server.adapters.tiff import MultiFileOmeTiffAdapter

        dir_path, file_list, metadata = multifile_ome_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'multifile-integration')

        server = TensorFlightServer('grpc://localhost:8893')
        server.register_source('multifile-integration', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8893', cache_bytes=10_000_000)

            darr = client.get_tensor('multifile-integration', 'multifile-integration')

            # Verify shape - should have n_files planes
            n_files = len(file_list)
            assert darr.shape[0] == n_files

            # Read first plane (use full slice to match chunk boundaries)
            data = darr.compute()
            assert data.shape[0] == n_files

            # First plane should have value 1 (set by fixture)
            assert data[0].mean() == 1

            client.close()
        finally:
            server.shutdown()

    def test_micromanager_multifile_read(self, multifile_mm_dataset):
        """Test reading from Micro-Manager multi-file dataset."""
        from biopb_tensor_server.adapters.tiff import MultiFileOmeTiffAdapter

        dir_path, file_list, metadata = multifile_mm_dataset

        adapter = MultiFileOmeTiffAdapter(dir_path, 'mm-integration')

        server = TensorFlightServer('grpc://localhost:8892')
        server.register_source('mm-integration', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8892', cache_bytes=10_000_000)

            darr = client.get_tensor('mm-integration', 'mm-integration')

            # Should have 3 channels (set by fixture)
            assert darr.shape[0] == 3

            # Read all channels
            data = darr.compute()
            assert data.shape[0] == 3

            client.close()
        finally:
            server.shutdown()


class TestHdf5Integration:
    """Integration tests for Hdf5Adapter with server/client."""

    @pytest.mark.skipif(not _h5py_available(), reason="h5py not available")
    def test_hdf5_read(self, hdf5_dataset):
        """Test reading from HDF5 through server."""
        import h5py

        from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter

        h5_path, shape, chunks = hdf5_dataset

        # HDF5 adapter needs the dataset object, so we need to keep file open
        with h5py.File(h5_path, 'r') as f:
            dataset = f['data']
            adapter = Hdf5Adapter(dataset, 'hdf5-integration')

            server = TensorFlightServer('grpc://localhost:8891')
            server.register_source('hdf5-integration', adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                client = TensorFlightClient('grpc://localhost:8891', cache_bytes=10_000_000)

                darr = client.get_tensor('hdf5-integration', 'hdf5-integration')
                assert darr.shape == shape

                data = darr.compute()
                assert data.shape == shape

                client.close()
            finally:
                server.shutdown()


class TestCacheIntegration:
    """Integration tests for cache behavior across adapters."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_cache_hit_multiple_reads(self, simple_zarr_array):
        """Test that repeated reads hit cache."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        arr = zarr.open_array(zarr_path, mode='r')
        adapter = ZarrAdapter(arr, 'cache-test', ['y', 'x'])

        server = TensorFlightServer('grpc://localhost:8890')
        server.register_source('cache-test', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8890', cache_bytes=10_000_000)

            darr = client.get_tensor('cache-test', 'cache-test')

            # First read
            data1 = darr[:chunks[0], :chunks[1]].compute()

            # Check cache state via cache_info()
            info1 = client.cache_info()
            initial_bytes = info1["size_bytes"]

            # Second read - same region
            data2 = darr[:chunks[0], :chunks[1]].compute()

            # Cache should have same size (hit, no new data)
            info2 = client.cache_info()
            assert info2["size_bytes"] == initial_bytes

            # Data should be identical
            np.testing.assert_array_equal(data1, data2)

            client.close()
        finally:
            server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_different_regions_different_cache_entries(self, simple_zarr_array):
        """Test that different regions create different cache entries."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        arr = zarr.open_array(zarr_path, mode='r')
        adapter = ZarrAdapter(arr, 'cache-regions', ['y', 'x'])

        server = TensorFlightServer('grpc://localhost:8889')
        server.register_source('cache-regions', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient('grpc://localhost:8889', cache_bytes=10_000_000)

            darr = client.get_tensor('cache-regions', 'cache-regions')

            # Read first chunk
            data1 = darr[:chunks[0], :chunks[1]].compute()
            nbytes1 = client.cache_info()["size_bytes"]

            # Read second chunk (different region)
            data2 = darr[chunks[0]:chunks[0]*2, chunks[1]:chunks[1]*2].compute()
            nbytes2 = client.cache_info()["size_bytes"]

            # Cache should have grown
            assert nbytes2 > nbytes1

            client.close()
        finally:
            server.shutdown()


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_concurrent_reads_same_region(self, simple_zarr_array):
        """Test concurrent reads of same region hit same cache."""
        import concurrent.futures

        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        arr = zarr.open_array(zarr_path, mode='r')
        adapter = ZarrAdapter(arr, 'concurrent-test', ['y', 'x'])

        server = TensorFlightServer('grpc://localhost:8888')
        server.register_source('concurrent-test', adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        results = []

        def read_region(client_id):
            client = TensorFlightClient('grpc://localhost:8888', cache_bytes=10_000_000)
            darr = client.get_tensor('concurrent-test', 'concurrent-test')
            data = darr[:chunks[0], :chunks[1]].compute()
            results.append((client_id, data.mean()))
            client.close()
            return data

        try:
            # Run multiple concurrent reads
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(read_region, i) for i in range(4)]
                datas = [f.result() for f in concurrent.futures.as_completed(futures)]

            # All should return same data
            for data in datas:
                np.testing.assert_array_equal(datas[0], data)

            server.shutdown()
        except Exception:
            server.shutdown()
            raise