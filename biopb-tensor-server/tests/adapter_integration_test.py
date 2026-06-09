"""Integration tests for adapters with server, client, and dask compute.

Tests the full pipeline from adapter registration through client access
to dask compute for each adapter type.
"""

import importlib.util
import tempfile
import threading
import time

from pathlib import Path

import numpy as np
import pytest
from biopb.tensor import (
    TensorFlightClient,
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


def _bioformats_available() -> bool:
    """Check if bioformats_jar is available for companion.ome support."""
    try:
        import bioformats_jar

        return True
    except ImportError:
        return False


class TestZarrIntegration:
    """Integration tests for ZarrAdapter with server/client."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_server_client_roundtrip(self, simple_zarr_array):
        """Test basic server -> client -> dask compute workflow."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        # Open array and create adapter
        arr = zarr.open_array(zarr_path, mode="r")
        adapter = ZarrAdapter(arr, "zarr-integration", ["y", "x"])

        # Start server
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("zarr-integration", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            # Connect client and compute
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # List sources
            sources = client.list_sources()
            assert "zarr-integration" in sources

            # Get tensor (source_id matches tensor_id for single-tensor sources)
            darr = client.get_tensor("zarr-integration", "zarr-integration")
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

        arr = zarr.open_array(zarr_path, mode="r")
        adapter = ZarrAdapter(arr, "zarr-scaled", ["y", "x"])

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("zarr-scaled", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # Test stride downsampling
            darr = client.get_tensor(
                "zarr-scaled",
                "zarr-scaled",
                scale_hint=[2, 2],
                reduction_method="stride",
            )
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

        root = zarr.open_group(zarr_path, mode="r")
        arr = root["0"]

        adapter = OmeZarrAdapter(arr, "ome-zarr-integration")

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome-zarr-integration", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # Test precompute method for scale 2
            darr = client.get_tensor(
                "ome-zarr-integration",
                "ome-zarr-integration",
                scale_hint=[2, 2],
                reduction_method="precompute",
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

        root = zarr.open_group(zarr_path, mode="r")
        arr = root["0"]

        adapter = OmeZarrAdapter(arr, "ome-zarr-virtual")

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome-zarr-virtual", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # Request scale 3 - no matching level, should use virtual scaling
            darr = client.get_tensor(
                "ome-zarr-virtual",
                "ome-zarr-virtual",
                scale_hint=[3, 3],
                reduction_method="nearest",
            )

            # Virtual scaling gives ceil(shape/3)
            base_shape = (256, 256)
            expected_shape = (
                (base_shape[0] + 2) // 3,  # ceil division
                (base_shape[1] + 2) // 3,
            )
            assert darr.shape == expected_shape

            data = darr.compute()
            assert data.shape == expected_shape

            client.close()
        finally:
            server.shutdown()


    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_physical_scale_summary_on_descriptor(self, temp_dir):
        """The per-dim physical-scale summary rides the descriptor that
        get_tensor (with_metadata=False) fetches, so the common path needs no
        full-OME round trip (issue #31). It must be empty in list_flights."""
        import json
        import os

        import zarr

        from biopb_tensor_server import OmeZarrAdapter

        # OME-Zarr with real physical units (the fixture uses relative scales).
        zarr_path = os.path.join(temp_dir, "phys.ome.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_dataset("0", shape=(64, 64), chunks=(32, 32), dtype="uint8")
        zattrs = {
            "multiscales": [{
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [
                        {"type": "scale", "scale": [0.5, 0.25]}]},
                ],
            }]
        }
        with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
            json.dump(zattrs, f)

        root = zarr.open_group(zarr_path, mode="r")
        adapter = OmeZarrAdapter(root["0"], "phys")

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("phys", adapter)
        server.mark_ready()
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

            # list_flights stays lean: no physical scale advertised there.
            sources = client.list_sources()
            listed = sources["phys"].tensors[0]
            assert not listed.physical_scale

            # A normal get_tensor (with_metadata=False) populates the cached
            # descriptor's summary; get_physical_scale reads it with no extra
            # full-OME fetch.
            client.get_tensor("phys", "phys")
            scale, unit = client.get_physical_scale("phys", "phys")
            assert list(scale) == [0.5, 0.25]
            assert list(unit) == ["micrometer", "micrometer"]

            client.close()
        finally:
            server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_plain_zarr_has_no_physical_scale(self, simple_zarr_array):
        """A plain Zarr source advertises no physical scale -> client None."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array
        arr = zarr.open_array(zarr_path, mode="r")
        adapter = ZarrAdapter(arr, "plain", ["y", "x"])

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("plain", adapter)
        server.mark_ready()
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
            client.get_tensor("plain", "plain")
            assert client.get_physical_scale("plain", "plain") is None
            client.close()
        finally:
            server.shutdown()


class TestOmeTiffIntegration:
    """Integration tests for OME-TIFF files (now handled by AicsImageIoAdapter)."""

    def test_tiled_tiff_read(self, tiled_ome_tiff):
        """Test reading from tiled OME-TIFF through server."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        tiff_path, fixture_shape, tile_info = tiled_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(tiff_path, "ome-tiff-integration")

        # Get scene_id for tensor access
        descriptors = adapter.list_tensor_descriptors()
        scene_id = descriptors[0].array_id

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome-tiff-integration", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # tensor_id is the scene_id (e.g., 'Image:0')
            darr = client.get_tensor("ome-tiff-integration", scene_id)

            # AICSImage uses TCZYX dimension order, so shape is (T, C, Z, Y, X)
            # Original fixture creates (C, Y, X) = (3, 128, 128) with CYX axes
            # aicsimageio interprets this as T=1, C=3, Z=1, Y=128, X=128
            expected_shape = (1, 3, 1, 128, 128)
            assert darr.shape == expected_shape
            assert darr.dtype == np.uint16

            # Read specific region (slice across C dimension)
            data = darr[:1, :1, :1, :32, :32].compute()
            assert data.shape == (1, 1, 1, 32, 32)

            # Channel 0 should have value 1 (plane 0 = value 1 set by fixture)
            assert data[0, 0, 0].mean() == 1

            client.close()
        finally:
            server.shutdown()

    def test_channel_access(self, tiled_ome_tiff):
        """Test accessing different channels through server."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        tiff_path, fixture_shape, tile_info = tiled_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(tiff_path, "ome-tiff-channels")

        # Get scene_id for tensor access
        descriptors = adapter.list_tensor_descriptors()
        scene_id = descriptors[0].array_id

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome-tiff-channels", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # tensor_id is the scene_id (e.g., 'Image:0')
            darr = client.get_tensor("ome-tiff-channels", scene_id)

            # Read channel 0 (value = 1) - C axis is index 1 in TCZYX
            data0 = darr[0:1, 0:1].compute()
            assert data0[0, 0].mean() == 1

            # Read channel 1 (value = 2)
            data1 = darr[0:1, 1:2].compute()
            assert data1[0, 0].mean() == 2

            # Read channel 2 (value = 3)
            data2 = darr[0:1, 2:3].compute()
            assert data2[0, 0].mean() == 3

            client.close()
        finally:
            server.shutdown()


class TestMultiSeriesOmeTiffIntegration:
    """Integration tests for multi-series OME-TIFF with server/client (now handled by AicsImageIoAdapter)."""

    def test_multi_series_list_tensors(self, multi_series_ome_tiff):
        """Test listing all series as tensors in multi-series OME-TIFF."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(tiff_path, "multi-series-test")

        # List all tensors (series)
        descriptors = adapter.list_tensor_descriptors()
        assert len(descriptors) == series_info["n_series"]

        # Each descriptor should have a unique array_id
        array_ids = [d.array_id for d in descriptors]
        assert len(set(array_ids)) == len(array_ids)

    def test_multi_series_tensor_access(self, multi_series_ome_tiff):
        """Test accessing specific series via get_tensor_adapter."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(tiff_path, "multi-series-access")

        # Get actual scene IDs from adapter
        descriptors = adapter.list_tensor_descriptors()
        scene_ids = [d.array_id for d in descriptors]

        # Access each series and verify data
        for scene_id in scene_ids:
            series_adapter = adapter.get_tensor_adapter(scene_id)
            desc = series_adapter.get_tensor_descriptor()

            # Verify shape matches expected (aicsimageio uses TCZYX order)
            assert len(desc.shape) == 5  # T, C, Z, Y, X
            assert desc.shape[3] == 64  # height
            assert desc.shape[4] == 64  # width

    def test_multi_series_server_client(self, multi_series_ome_tiff):
        """Test reading from multi-series OME-TIFF through server."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(tiff_path, "multi-series-server")

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("multi-series-server", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            # List sources
            sources = client.list_sources()
            assert "multi-series-server" in sources

            # Get actual scene IDs
            descriptors = adapter.list_tensor_descriptors()
            first_scene_id = descriptors[0].array_id

            # Get first series tensor - tensor_id is the scene_id (e.g., 'Image:0')
            darr = client.get_tensor("multi-series-server", first_scene_id)

            # Read data
            data = darr.compute()
            # aicsimageio uses TCZYX order: (T, C, Z, Y, X)
            # Fixture uses CYX axes, so C=2, Z=1
            assert data.shape[1] == 2  # C planes per series (from CYX first axis)

            # First C plane of first series should have value 1
            assert data[0, 0, 0].mean() == 1

            client.close()
        finally:
            server.shutdown()

    def test_lazy_tile_loading(self, multi_series_ome_tiff):
        """Test that aicsimageio provides tile-level lazy loading."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter
        from biopb.tensor.ticket_pb2 import ChunkBounds

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = AicsImageIoAdapter.create_from_url(tiff_path, "lazy-tile-test")

        # Get actual scene IDs
        descriptors = adapter.list_tensor_descriptors()
        first_scene_id = descriptors[0].array_id

        # Get first series adapter
        series_adapter = adapter.get_tensor_adapter(first_scene_id)

        # Read a small tile region (TCZYX order)
        bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 32, 32])
        data = series_adapter.get_data(bounds)

        assert data.shape == (1, 1, 1, 32, 32)
        assert data[0, 0, 0].mean() == 1  # First plane has value 1


class TestCompanionOmeIntegration:
    """Integration tests for companion OME files (now handled by AicsImageIoAdapter)."""

    @pytest.mark.skipif(
        not _bioformats_available(), reason="bioformats_jar not available"
    )
    def test_companion_claim(self, companion_ome_dataset):
        """Test claiming companion.ome file."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter
        from pathlib import Path

        companion_path, tiff_files, metadata_info = companion_ome_dataset

        # Test claim method
        from biopb_tensor_server.discovery import ClaimContext, DiscoveryState

        ctx = ClaimContext(Path(companion_path))
        state = DiscoveryState()
        claim = AicsImageIoAdapter.claim(ctx, state)
        assert claim is not None
        assert claim.source_type == "aics"
        # Primary path is the companion file itself for companion.ome claims
        assert str(claim.primary_path) == companion_path
        # All TIFF files should be in consumed paths (tracked via try_claim_path)
        for tiff_file in tiff_files:
            assert tiff_file in state.consumed_paths

    @pytest.mark.skipif(
        not _bioformats_available(), reason="bioformats_jar not available"
    )
    def test_companion_data_access(self, companion_ome_dataset):
        """Test reading data from companion OME dataset."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter
        from biopb.tensor.ticket_pb2 import ChunkBounds

        companion_path, tiff_files, metadata_info = companion_ome_dataset

        # Create adapter from companion file
        adapter = AicsImageIoAdapter.create_from_url(companion_path, "companion-test")

        # Get first series
        descriptors = adapter.list_tensor_descriptors()
        assert len(descriptors) > 0

        # Read data
        desc = descriptors[0]
        series_adapter = adapter.get_tensor_adapter(desc.array_id)

        bounds = ChunkBounds(start=[0, 0, 0], stop=[metadata_info["n_files"], 64, 64])
        data = series_adapter.get_data(bounds)

        assert data.shape[0] == metadata_info["n_files"]
        # First file has value 1
        assert data[0].mean() == 1


class TestHdf5Integration:
    """Integration tests for Hdf5Adapter with server/client."""

    @pytest.mark.skipif(not _h5py_available(), reason="h5py not available")
    def test_hdf5_read(self, hdf5_dataset):
        """Test reading from HDF5 through server."""
        import h5py

        from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter

        h5_path, shape, chunks = hdf5_dataset

        # HDF5 adapter needs the dataset object, so we need to keep file open
        with h5py.File(h5_path, "r") as f:
            dataset = f["data"]
            adapter = Hdf5Adapter(dataset, "hdf5-integration")

            server = TensorFlightServer("grpc://localhost:0")
            server.register_source("hdf5-integration", adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                client = TensorFlightClient(
                    f"grpc://localhost:{server.port}", cache_bytes=10_000_000
                )

                darr = client.get_tensor("hdf5-integration", "hdf5-integration")
                assert darr.shape == shape

                data = darr.compute()
                assert data.shape == shape

                client.close()
            finally:
                server.shutdown()


class TestCacheIntegration:
    """Integration tests for cache behavior across adapters."""

    @pytest.fixture(autouse=True)
    def _enable_local_cache(self, monkeypatch):
        # The client disables its per-process chunk cache for localhost servers
        # by default (the server already caches its data). These tests exercise
        # the client cache directly against a localhost server, so opt back in.
        monkeypatch.setenv("BIOPB_CACHE_LOCAL", "1")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_cache_hit_multiple_reads(self, simple_zarr_array):
        """Test that repeated reads hit cache."""
        import zarr

        from biopb_tensor_server import ZarrAdapter

        zarr_path, shape, chunks = simple_zarr_array

        arr = zarr.open_array(zarr_path, mode="r")
        adapter = ZarrAdapter(arr, "cache-test", ["y", "x"])

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("cache-test", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            darr = client.get_tensor("cache-test", "cache-test")

            # First read
            data1 = darr[: chunks[0], : chunks[1]].compute()

            # Check cache state via cache_info()
            info1 = client.cache_info()
            initial_bytes = info1["size_bytes"]

            # Second read - same region
            data2 = darr[: chunks[0], : chunks[1]].compute()

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

        arr = zarr.open_array(zarr_path, mode="r")
        adapter = ZarrAdapter(arr, "cache-regions", ["y", "x"])

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("cache-regions", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)

            darr = client.get_tensor("cache-regions", "cache-regions")

            # Read first chunk
            data1 = darr[: chunks[0], : chunks[1]].compute()
            nbytes1 = client.cache_info()["size_bytes"]

            # Read second chunk (different region)
            data2 = darr[chunks[0] : chunks[0] * 2, chunks[1] : chunks[1] * 2].compute()
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

        arr = zarr.open_array(zarr_path, mode="r")
        adapter = ZarrAdapter(arr, "concurrent-test", ["y", "x"])

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("concurrent-test", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        results = []

        def read_region(client_id):
            client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=10_000_000)
            darr = client.get_tensor("concurrent-test", "concurrent-test")
            data = darr[: chunks[0], : chunks[1]].compute()
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


class TestAicsImageIoAdapterClaim:
    """Tests for AicsImageIoAdapter.claim() scope limits."""

    def test_claim_generic_files_rejected(self):
        """Generic file types (txt, csv, cfg) should not be claimed."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter
        from biopb_tensor_server.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various generic file types that bioformats supports
            for ext in [".txt", ".csv", ".cfg", ".htm", ".html", ".db", ".dat", ".bin"]:
                path = Path(tmpdir) / f"test{ext}"
                path.write_text("not an image file")

                ctx = ClaimContext(path)
                state = DiscoveryState()
                claim = AicsImageIoAdapter.claim(ctx, state)

                assert claim is None, f"AicsImageIoAdapter should not claim {ext} files"

    def test_claim_image_files_accepted(self):
        """Standard image file types should be claimed."""
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter
        from biopb_tensor_server.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test standard image formats (extensions only, content doesn't matter for claim)
            for ext in [".png", ".jpg", ".gif", ".bmp"]:
                path = Path(tmpdir) / f"test{ext}"
                path.write_bytes(b"\x00")  # Dummy content

                ctx = ClaimContext(path)
                state = DiscoveryState()
                claim = AicsImageIoAdapter.claim(ctx, state)

                assert claim is not None, f"AicsImageIoAdapter should claim {ext} files"
                assert claim.source_type == "aics"
