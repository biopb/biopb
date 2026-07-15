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

        _ = zarr.open_array
        return True
    except ImportError:
        return False


def _h5py_available() -> bool:
    """Check if h5py is available."""
    return importlib.util.find_spec("h5py") is not None


# Directory of real vendor samples (CZI/ND2/LIF) for the fixture-gated read
# tests. BIOPB_TEST_VENDOR_DIR overrides the checked-in default.
_VENDOR_FIXTURE_DIR = Path(__file__).parent / "data" / "vendor"


def _vendor_fixture(ext: str):
    """First sample file with the given extension, or None if none provided.

    Looks in $BIOPB_TEST_VENDOR_DIR (if set) then tests/data/vendor/. Returns a
    path string so a test can read through the real reader plugin, or None so it
    self-skips when no sample is present.
    """
    import os

    roots = []
    env = os.environ.get("BIOPB_TEST_VENDOR_DIR")
    if env:
        roots.append(Path(env))
    roots.append(_VENDOR_FIXTURE_DIR)
    for root in roots:
        if root.is_dir():
            hits = sorted(root.glob(f"*{ext}"))
            if hits:
                return str(hits[0])
    return None


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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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
    def test_big_endian_source_roundtrip(self):
        """A big-endian source must serve and reconstruct with correct values.

        Arrow cannot hold byte-swapped buffers, so pa.array() on a '>i2' chunk
        used to raise ArrowNotImplementedError ("Byte-swapped arrays not
        supported") -- breaking every read of a big-endian source (FITS is
        big-endian by spec). The server now normalizes to native order at
        serialization; the client must still get identical values.
        """
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "be.zarr")
            # Big-endian int16 array with distinctive values.
            src = (np.arange(64 * 64, dtype="<i2").reshape(64, 64) - 5000).astype(">i2")
            za = zarr.open_array(
                zarr_path, mode="w", shape=(64, 64), chunks=(32, 32), dtype=">i2"
            )
            za[:] = src
            assert za.dtype.byteorder == ">"  # genuinely non-native on this host

            from biopb_tensor_server import ZarrAdapter

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "be", ["y", "x"]
            )
            server = TensorFlightServer("grpc://localhost:0")
            server.register_source("be", adapter)
            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                client = TensorFlightClient(
                    f"grpc://localhost:{server.port}", cache_bytes=10_000_000
                )
                darr = client.get_tensor("be")
                data = darr.compute()  # raised ArrowNotImplementedError before the fix
                # Values are preserved exactly (compared in native order).
                np.testing.assert_array_equal(data.astype("<i2"), src.astype("<i2"))
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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [0.5, 0.25]}
                            ],
                        },
                    ],
                }
            ]
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
    """Integration tests for OME-TIFF files (now handled by OmeTiffAdapter)."""

    def test_tiled_tiff_read(self, tiled_ome_tiff):
        """Test reading from tiled OME-TIFF through server."""
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        tiff_path, fixture_shape, tile_info = tiled_ome_tiff

        adapter = OmeTiffAdapter(tiff_path, "ome-tiff-integration")

        # Get scene_id for tensor access
        descriptors = adapter.list_tensor_descriptors()
        scene_id = descriptors[0].array_id

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome-tiff-integration", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

            # tensor_id is the scene_id (e.g., 'Image:0')
            darr = client.get_tensor("ome-tiff-integration", scene_id)

            # OME-TIFF uses TCZYX dimension order, so shape is (T, C, Z, Y, X)
            # Original fixture creates (C, Y, X) = (3, 128, 128) with CYX axes
            # interpreted as T=1, C=3, Z=1, Y=128, X=128
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
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        tiff_path, fixture_shape, tile_info = tiled_ome_tiff

        adapter = OmeTiffAdapter(tiff_path, "ome-tiff-channels")

        # Get scene_id for tensor access
        descriptors = adapter.list_tensor_descriptors()
        scene_id = descriptors[0].array_id

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome-tiff-channels", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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

    def test_read_path_never_parses_ome_model(self, tmp_path):
        """biopb/biopb#213: end-to-end, GetFlightInfo + DoGet over a registered
        OME-TIFF must not trigger a heavy bioio (BioImage) reader parse.

        A tripwire is installed on the registered adapter's ``_bio_image`` AFTER
        registration, then a full client round-trip drives GetFlightInfo (via the
        #350 ``plan_flight_info`` seam, which also fills the physical scale) and
        DoGet. Physical scale carries real units, proving the whole chain --
        descriptor, physical scale, and pixels -- is served from tifffile.
        """
        import tifffile
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        class _Tripwire:
            def __getattr__(self, name):
                raise AssertionError(
                    f"heavy reader (BioImage) touched on read path: .{name}"
                )

        path = str(tmp_path / "e2e.ome.tif")
        data = np.zeros((3, 64, 64), np.uint16)
        for c in range(3):
            data[c] = c + 1
        tifffile.imwrite(
            path,
            data,
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": 0.325,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.325,
                "PhysicalSizeYUnit": "µm",
            },
        )
        adapter = OmeTiffAdapter(path, "ome213")
        array_id = adapter.list_tensor_descriptors()[0].array_id  # registration

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("ome213", adapter)
        server.mark_ready()
        adapter._bio_image = _Tripwire()  # any OME parse from here is a failure
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
            darr = client.get_tensor(array_id)  # GetFlightInfo (array_id addressing)
            assert darr.shape == (1, 3, 1, 64, 64)

            scale, unit = client.get_physical_scale(array_id)
            assert list(scale) == [0.0, 0.0, 0.0, 0.325, 0.325]
            assert list(unit) == ["", "", "", "µm", "µm"]

            data = darr[:1, 1:2, :1, :32, :32].compute()  # DoGet
            assert data.shape == (1, 1, 1, 32, 32)
            assert (data == 2).all()  # channel 1 -> value 2
            client.close()
        finally:
            server.shutdown()


class TestMultiSeriesOmeTiffIntegration:
    """Integration tests for multi-series OME-TIFF with server/client (now handled by OmeTiffAdapter)."""

    def test_multi_series_list_tensors(self, multi_series_ome_tiff):
        """Test listing all series as tensors in multi-series OME-TIFF."""
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = OmeTiffAdapter(tiff_path, "multi-series-test")

        # List all tensors (series)
        descriptors = adapter.list_tensor_descriptors()
        assert len(descriptors) == series_info["n_series"]

        # Each descriptor should have a unique array_id
        array_ids = [d.array_id for d in descriptors]
        assert len(set(array_ids)) == len(array_ids)

    def test_multi_series_tensor_access(self, multi_series_ome_tiff):
        """Test accessing specific series via get_tensor_adapter."""
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = OmeTiffAdapter(tiff_path, "multi-series-access")

        # Get actual scene IDs from adapter
        descriptors = adapter.list_tensor_descriptors()
        scene_ids = [d.array_id for d in descriptors]

        # Access each series and verify data
        for scene_id in scene_ids:
            series_adapter = adapter.get_tensor_adapter(scene_id)
            desc = series_adapter.get_tensor_descriptor()

            # Verify shape matches expected (OME-TIFF uses TCZYX order)
            assert len(desc.shape) == 5  # T, C, Z, Y, X
            assert desc.shape[3] == 64  # height
            assert desc.shape[4] == 64  # width

    def test_multi_series_server_client(self, multi_series_ome_tiff):
        """Test reading from multi-series OME-TIFF through server."""
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = OmeTiffAdapter(tiff_path, "multi-series-server")

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("multi-series-server", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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
            # OME-TIFF uses TCZYX order: (T, C, Z, Y, X)
            # Fixture uses CYX axes, so C=2, Z=1
            assert data.shape[1] == 2  # C planes per series (from CYX first axis)

            # First C plane of first series should have value 1
            assert data[0, 0, 0].mean() == 1

            client.close()
        finally:
            server.shutdown()

    def test_lazy_tile_loading(self, multi_series_ome_tiff):
        """Test that the OME-TIFF adapter provides tile-level lazy loading."""
        from biopb.tensor.ticket_pb2 import ChunkBounds
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        tiff_path, fixture_series_names, series_info = multi_series_ome_tiff

        adapter = OmeTiffAdapter(tiff_path, "lazy-tile-test")

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
    """`.companion.ome` support was dropped when OmeTiffAdapter went pure-tifffile
    (it was bioformats-only, with no tifffile path). It is no longer claimed."""

    def test_companion_ome_is_not_claimed(self, companion_ome_dataset):
        # The pure-tifffile OmeTiffAdapter declines .companion.ome (only tifffile
        # embedded-OME-XML is supported); the generic AicsImageIoAdapter also
        # excludes it. So a companion sidecar is claimed by no one.
        from pathlib import Path

        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        companion_path, _tiff_files, _metadata_info = companion_ome_dataset
        ctx = ClaimContext(Path(companion_path))
        assert OmeTiffAdapter.claim(ctx, DiscoveryState()) is None


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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )

            darr = client.get_tensor("cache-regions", "cache-regions")

            # Read first chunk
            darr[: chunks[0], : chunks[1]].compute()
            nbytes1 = client.cache_info()["size_bytes"]

            # Read second chunk (different region)
            darr[chunks[0] : chunks[0] * 2, chunks[1] : chunks[1] * 2].compute()
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
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
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


class TestBioioReadPath:
    """Exercise the bioio *read* path, not just claim() routing.

    Reading through bioio needs the matching ``bioio-*`` reader plugin installed;
    a missing one surfaces here as ``UnsupportedFileFormatError`` rather than
    silently at runtime. In particular ``bioio-ome-tiff`` claims ONLY OME-TIFF,
    so plain TIFF and Zeiss LSM depend on ``bioio-tifffile`` -- the coverage gap
    that let that plugin get dropped from the ``[aics]`` extra once.
    """

    def _read_full(self, path, adapter_cls=None):
        """Register a file as a bioio source and read its first tensor whole.

        ``adapter_cls`` selects the concrete adapter to read through -- the
        generic ``AicsImageIoAdapter`` by default, or a vendor subclass
        (``ZeissAdapter`` etc.) to exercise the exact class discovery would
        route a real file to. The read itself lives on the shared base, so the
        subclass only matters for provenance; the companion claim assertion
        (``_assert_claims``) is what pins the routing.
        """
        from bioio import BioImage
        from biopb.tensor.ticket_pb2 import ChunkBounds
        from biopb_tensor_server.adapters.bioio import AicsImageIoAdapter

        adapter_cls = adapter_cls or AicsImageIoAdapter
        src = adapter_cls(
            BioImage(path), scene_index=None, source_id="s", source_url=path
        )
        descs = src.list_tensor_descriptors()
        assert descs, "bioio produced no tensor descriptors"
        scene = src.get_tensor_adapter(descs[0].array_id)
        desc = scene.get_tensor_descriptor()
        data = scene.get_data(
            ChunkBounds(start=[0] * len(desc.shape), stop=list(desc.shape))
        )
        return desc, data

    @staticmethod
    def _assert_claims(path, adapter_cls, source_type):
        """The vendor adapter claims the extension and tags the right type."""
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        claim = adapter_cls.claim(ClaimContext(Path(path)), DiscoveryState())
        assert claim is not None, f"{adapter_cls.__name__} did not claim {path}"
        assert claim.source_type == source_type

    def test_plain_tiff_read_via_bioio(self, tmp_path):
        """Plain (non-OME) TIFF reads through bioio-tifffile (the generic
        AicsImageIoAdapter fallback path)."""
        import tifffile

        arr = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
        path = str(tmp_path / "plain.tif")
        tifffile.imwrite(path, arr)

        desc, data = self._read_full(path)
        assert tuple(data.shape) == tuple(desc.shape)  # data matches its descriptor
        assert data.dtype == np.uint16
        # Pixels round-trip. bioio may permute or insert singleton axes, so
        # compare value multisets rather than a fixed layout.
        np.testing.assert_array_equal(np.sort(data.ravel()), np.sort(arr.ravel()))

    def test_lsm_read_via_bioio(self, tmp_path):
        """Zeiss .lsm (a TIFF variant) reads through bioio-tifffile -- the path
        ZeissAdapter takes for .lsm. Guards the bioio-tifffile dependency."""
        import tifffile

        arr = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
        path = str(tmp_path / "img.lsm")
        tifffile.imwrite(path, arr)

        desc, data = self._read_full(path)
        assert tuple(data.shape) == tuple(desc.shape)
        np.testing.assert_array_equal(np.sort(data.ravel()), np.sort(arr.ravel()))

    def test_dv_read_via_bioio(self, tmp_path):
        """DeltaVision .dv reads through bioio-dv (its ``mrc`` backend), the
        path DvAdapter takes. Guards the bioio-dv dependency.

        DV is the one true vendor format here that a Python lib can *write*
        (``mrc.imwrite``, which bioio-dv also reads through), so it gets real,
        self-contained read-path coverage in CI -- CZI/ND2/LIF have no open
        writer and are fixture-gated below.
        """
        pytest.importorskip("bioio_dv")
        mrc = pytest.importorskip("mrc")  # bioio-dv's reader backend
        from biopb_tensor_server.adapters.bioio import DvAdapter

        # A Z-stack keeps the mrc header at C=1 (a bare 3-D block is otherwise
        # mis-read as 3 channels with mismatched coordinate metadata).
        arr = np.arange(5 * 16 * 16, dtype=np.int16).reshape(5, 16, 16)
        path = str(tmp_path / "sample.dv")
        mrc.imwrite(path, arr)

        self._assert_claims(path, DvAdapter, "dv")
        desc, data = self._read_full(path, DvAdapter)
        assert tuple(data.shape) == tuple(desc.shape)
        assert desc.dtype == np.dtype(np.int16).str
        np.testing.assert_array_equal(np.sort(data.ravel()), np.sort(arr.ravel()))

    # ---- Fixture-gated true-vendor formats (no open writer) ----------------
    #
    # CZI, ND2 and LIF have no Python library that writes a file their bioio
    # plugin will read back faithfully (a synthesized CZI, for instance, lacks
    # the Scenes / Channel / Pixels metadata bioio-czi's OME transform demands
    # and fails validation), so these can only be exercised against a real
    # sample. Drop tiny samples in a directory and point BIOPB_TEST_VENDOR_DIR
    # at it (or the default tests/data/vendor/) and the matching test reads one
    # through its adapter; otherwise it self-skips. This is the hook the planned
    # installer-wheel-coverage CI provisions fixtures for (issue #361).

    @pytest.mark.parametrize(
        "plugin, ext, source_type",
        [
            ("bioio_czi", ".czi", "zeiss"),
            ("bioio_nd2", ".nd2", "nikon"),
            ("bioio_lif", ".lif", "leica"),
        ],
    )
    def test_vendor_fixture_read_via_bioio(self, plugin, ext, source_type):
        """A real CZI/ND2/LIF sample claims + reads through its adapter.

        Skips cleanly when the plugin is absent (slim install) or no sample is
        provisioned, so it never fails spuriously -- but catches a dropped
        plugin the moment a fixture is present.
        """
        pytest.importorskip(plugin)
        from biopb_tensor_server.adapters.bioio import (
            LeicaAdapter,
            NikonAdapter,
            ZeissAdapter,
        )

        adapter_cls = {
            "zeiss": ZeissAdapter,
            "nikon": NikonAdapter,
            "leica": LeicaAdapter,
        }[source_type]

        path = _vendor_fixture(ext)
        if path is None:
            pytest.skip(
                f"no *{ext} sample found (set BIOPB_TEST_VENDOR_DIR or add one "
                f"to {_VENDOR_FIXTURE_DIR})"
            )

        self._assert_claims(path, adapter_cls, source_type)
        desc, data = self._read_full(path, adapter_cls)
        # Descriptor and read agree, the read is non-trivial, and dtype matches.
        assert tuple(data.shape) == tuple(desc.shape)
        assert data.size > 0
        assert data.dtype.str == desc.dtype


class TestAicsImageIoAdapterClaim:
    """Tests for AicsImageIoAdapter.claim() scope limits."""

    def test_claim_generic_files_rejected(self):
        """Generic file types (txt, csv, cfg) should not be claimed."""
        from biopb_tensor_server.adapters.bioio import AicsImageIoAdapter
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various generic file types that bioformats supports
            for ext in [".txt", ".csv", ".cfg", ".htm", ".html", ".db", ".dat", ".bin"]:
                path = Path(tmpdir) / f"test{ext}"
                path.write_text("not an image file")

                ctx = ClaimContext(path)
                state = DiscoveryState()
                claim = AicsImageIoAdapter.claim(ctx, state)

                assert claim is None, f"AicsImageIoAdapter should not claim {ext} files"

    def test_microscopy_files_always_claimed(self):
        """Microscopy/scientific extensions are claimed regardless of the flag."""
        from biopb_tensor_server.adapters.bioio import AicsImageIoAdapter
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            # NOTE: .mrc/.mrcs are intentionally absent -- no bioio plugin can read
            # a plain cryo-EM MRC, so they are owned by MrcAdapter now, not this
            # generic bioio adapter (biopb/biopb#94; see test_mrc_not_claimed).
            for ext in [".tif", ".tiff", ".ims", ".fits", ".nrrd"]:
                path = Path(tmpdir) / f"test{ext}"
                path.write_bytes(b"\x00")  # Dummy content

                ctx = ClaimContext(path)
                state = DiscoveryState()
                claim = AicsImageIoAdapter.claim(ctx, state)

                assert claim is not None, f"AicsImageIoAdapter should claim {ext} files"
                assert claim.source_type == "aics"

    def test_mrc_not_claimed(self):
        """MRC is NOT claimed by the generic bioio adapter (biopb/biopb#94).

        No installed bioio plugin can read a plain cryo-EM MRC, so claiming it
        here produced a claim-then-error. MRC is owned by MrcAdapter, which is
        registered ahead of the bioio adapters.
        """
        from biopb_tensor_server.adapters.bioio import AicsImageIoAdapter
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in [".mrc", ".mrcs"]:
                path = Path(tmpdir) / f"test{ext}"
                path.write_bytes(b"\x00")
                claim = AicsImageIoAdapter.claim(ClaimContext(path), DiscoveryState())
                assert claim is None, f"bioio must not claim {ext}"

    def test_generic_images_rejected_by_default(self):
        """Generic raster/video must NOT be claimed during discovery by default
        (biopb/biopb#40) — they flood the catalog with screenshots/icons/movies."""
        from biopb_tensor_server.adapters.bioio import AicsImageIoAdapter
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".mp4", ".mov"]:
                path = Path(tmpdir) / f"test{ext}"
                path.write_bytes(b"\x00")

                ctx = ClaimContext(path)
                state = DiscoveryState()
                claim = AicsImageIoAdapter.claim(ctx, state)

                assert claim is None, (
                    f"AicsImageIoAdapter should not claim generic {ext} by default"
                )

    def test_generic_images_claimed_when_opted_in(self):
        """With claim_generic_images enabled, generic raster/video are claimed."""
        from biopb_tensor_server.adapters.bioio import (
            AicsImageIoAdapter,
            set_claim_generic_images,
        )
        from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState

        set_claim_generic_images(True)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                for ext in [".png", ".jpg", ".gif", ".bmp", ".mp4"]:
                    path = Path(tmpdir) / f"test{ext}"
                    path.write_bytes(b"\x00")

                    ctx = ClaimContext(path)
                    state = DiscoveryState()
                    claim = AicsImageIoAdapter.claim(ctx, state)

                    assert claim is not None, (
                        f"AicsImageIoAdapter should claim {ext} when opted in"
                    )
                    assert claim.source_type == "aics"
        finally:
            # Restore the process-wide default so other tests are unaffected.
            set_claim_generic_images(False)
