"""Server-backed round-trip tests for TensorFlightClient.

These spin up a real TensorFlightServer (in-process, daemon thread) and exercise
the client against it, so they need the server package installed. They live here
-- with the server package, whose CI installs it -- rather than under
src/test/python/, where the client-only CI job installs no server and silently
skipped them (biopb/biopb#579).
"""

import os
import tempfile
import threading
import time

import numpy as np
import pytest
from biopb.tensor import TensorFlightClient
from biopb_tensor_server import TensorFlightServer, ZarrAdapter

from tests import catalog_server, register_and_catalog


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr

        _ = zarr.open_array
        return True
    except ImportError:
        return False


class TestTensorFlightClientRoundTrip:
    """Client <-> live server round trips with the multifield API."""

    @pytest.fixture
    def server_client(self):
        """Start server and create client."""
        if not _zarr_available():
            pytest.skip("zarr not available")

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(128, 128), chunks=(64, 64), dtype="uint8"
            )

            # Set known values per chunk
            arr[:64, :64] = 10
            arr[:64, 64:] = 20
            arr[64:, :64] = 30
            arr[64:, 64:] = 40

            zarr_arr = zarr.open_array(zarr_path, mode="r")
            adapter = ZarrAdapter(zarr_arr, "test-tensor", ["y", "x"])

            server = catalog_server("grpc://localhost:8890")
            register_and_catalog(server, "test-tensor", adapter)

            # Start server in background
            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)  # Wait for server

            try:
                client = TensorFlightClient(
                    "grpc://localhost:8890", cache_bytes=10_000_000
                )
                yield client
                client.close()
            finally:
                server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_list_sources(self, server_client):
        """Test listing sources."""
        sources = server_client.list_sources()
        assert "test-tensor" in sources

        # DataSourceDescriptor contains tensor metadata
        source_desc = sources["test-tensor"]
        assert len(source_desc.tensors) == 1
        assert source_desc.tensors[0].array_id == "test-tensor"
        assert list(source_desc.tensors[0].shape) == [128, 128]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_source_answers_one_id(self, server_client):
        """The addressed counterpart of list_sources: same descriptor, one row."""
        desc = server_client.get_source("test-tensor")
        assert desc is not None
        assert desc.source_id == "test-tensor"
        assert [t.array_id for t in desc.tensors] == ["test-tensor"]
        assert list(desc.tensors[0].shape) == [128, 128]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_source_is_none_for_an_unknown_id(self, server_client):
        assert server_client.get_source("no-such-source") is None

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_the_catalog_is_never_snapshotted_client_side(self, server_client):
        """Neither call caches anything, source-keyed or otherwise.

        A local copy of the catalog answered "what does this server hold?",
        which is a question only the server can answer -- and the answer went
        stale the moment anything registered. The per-array_id addressing cache
        that once survived this is gone too: it had the same problem one level
        down, and no invalidation at all.
        """
        client = server_client
        client.list_sources()
        client.get_source("test-tensor")

        assert not hasattr(client._catalog._state, "sources")
        assert not hasattr(client._catalog._state, "descriptors")
        assert not hasattr(client, "_descriptors")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_shape(self, server_client):
        """Test tensor shape retrieval."""
        darr = server_client.get_tensor("test-tensor")
        assert darr.shape == (128, 128)
        assert darr.dtype == np.uint8

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_read_chunks(self, server_client):
        """Test reading different chunks."""
        darr = server_client.get_tensor("test-tensor")

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
        darr = server_client.get_tensor("test-tensor")

        # First read
        data1 = darr[:64, :64].compute()

        # Check cache has entries
        cache_info = server_client.cache_info()
        initial_bytes = cache_info["size_bytes"]

        # Second read - should hit cache
        data2 = darr[:64, :64].compute()

        np.testing.assert_array_equal(data1, data2)
        assert server_client.cache_info()["size_bytes"] == initial_bytes

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_a_catalog_source_has_no_upload_status(self, server_client):
        """A source nobody uploaded reports UNKNOWN, not PENDING (biopb/biopb#109).

        "test-tensor" is a registered on-disk source, so the server has no
        upload record for it. UNKNOWN is what a caller's poll loop stops on
        at once, rather than waiting out a timeout on a state no poll moves.
        """
        assert server_client.get_upload_status("test-tensor")["state"] == "UNKNOWN"

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_scaled_stride_view(self, server_client):
        """Test explicit per-call scaled reads using stride downsampling."""
        darr = server_client.get_tensor(
            "test-tensor",
            scale_hint=[2, 2],
            reduction_method="stride",
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
            "test-tensor",
            scale_hint=[2, 2],
            reduction_method="nearest",
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
            "test-tensor",
            scale_hint=[2, 2],
            reduction_method="mean",
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
            "test-tensor",
            scale_hint=[2, 2],
            reduction_method="area",
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
            zarr_path = os.path.join(tmpdir, "mean-preserve.zarr")
            source = np.array(
                [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15],
                ],
                dtype=np.uint8,
            )
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(4, 4), chunks=(2, 2), dtype="uint8"
            )
            arr[:] = source

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "mean-preserve", ["y", "x"]
            )
            server = TensorFlightServer("grpc://localhost:8893")
            server.register_source("mean-preserve", adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient(
                    "grpc://localhost:8893", cache_bytes=10_000_000
                ) as client:
                    darr = client.get_tensor(
                        "mean-preserve",
                        scale_hint=[2, 2],
                        reduction_method="mean",
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
            zarr_path = os.path.join(tmpdir, "linear.zarr")
            source = np.array(
                [
                    [0, 10, 40, 90],
                    [20, 30, 60, 110],
                    [80, 90, 120, 170],
                    [180, 190, 220, 255],
                ],
                dtype=np.uint8,
            )
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(4, 4), chunks=(2, 2), dtype="uint8"
            )
            arr[:] = source

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "linear", ["y", "x"]
            )
            server = TensorFlightServer("grpc://localhost:8894")
            server.register_source("linear", adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient(
                    "grpc://localhost:8894", cache_bytes=10_000_000
                ) as client:
                    darr = client.get_tensor(
                        "linear",
                        scale_hint=[2, 2],
                        reduction_method="linear",
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
            zarr_path = os.path.join(tmpdir, "nearest-edge.zarr")
            source = np.arange(25, dtype=np.uint8).reshape(5, 5)
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(5, 5), chunks=(3, 3), dtype="uint8"
            )
            arr[:] = source

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "nearest-edge", ["y", "x"]
            )
            server = TensorFlightServer("grpc://localhost:8892")
            server.register_source("nearest-edge", adapter)

            server_thread = threading.Thread(target=server.serve, daemon=True)
            server_thread.start()
            time.sleep(1)

            try:
                with TensorFlightClient(
                    "grpc://localhost:8892", cache_bytes=10_000_000
                ) as client:
                    darr = client.get_tensor(
                        "nearest-edge",
                        scale_hint=[2, 2],
                        reduction_method="nearest",
                    )
                    assert darr.shape == (3, 3)
                    np.testing.assert_array_equal(darr.compute(), source[::2, ::2])
            finally:
                server.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_tensor_not_found_raises(self, server_client):
        """Test that requesting non-existent tensor raises error."""
        # The server's wording, not a client-side reconstruction of it: it
        # knows this source exists and has no such field, which the client would
        # need a catalog round trip on every read to learn.
        with pytest.raises(ValueError, match="no field 'nonexistent'"):
            server_client.get_tensor("test-tensor/nonexistent")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_source_not_found_raises(self, server_client):
        """Test that requesting non-existent source raises error."""
        with pytest.raises(ValueError, match="Source not found"):
            server_client.get_tensor("nonexistent-source/some-tensor")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_pb(self, server_client, transfer_target):
        """Test get_tensor_pb returns SerializedTensor protobuf."""
        from biopb.tensor.serialized_pb2 import SerializedTensor

        # Four endpoints over the 128x128 uint8 fixture; at the default target
        # it is a single chunk (biopb/biopb#809) and the endpoint list under
        # test would have one entry.
        transfer_target(64 * 64)

        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        pb = server_client.get_tensor_pb("test-tensor")

        # Verify it's a SerializedTensor
        assert isinstance(pb, SerializedTensor)

        # The plan is the FlightInfo the server answered, carried whole.
        info = flight.FlightInfo.deserialize(pb.flight_info)
        desc = TensorDescriptor.FromString(info.descriptor.command)
        assert desc.array_id == "test-tensor"
        assert list(desc.shape) == [128, 128]
        # dtype may be uint8 or |u1 depending on server
        assert desc.dtype in ("uint8", "|u1")
        assert list(desc.chunk_shape) == [64, 64]
        assert len(info.endpoints) == 4

        # Verify location is populated
        assert pb.location == "grpc://localhost:8890"

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_tensor_from_pb(self, server_client):
        """Test tensor_from_pb reconstructs dask array."""
        pb = server_client.get_tensor_pb("test-tensor")

        # Reconstruct array
        darr = TensorFlightClient.tensor_from_pb(pb)

        # Verify shape and dtype
        assert darr.shape == (128, 128)
        assert darr.dtype == np.uint8

        # Verify data values match direct get_tensor
        direct_darr = server_client.get_tensor("test-tensor")

        # Top-left chunk
        np.testing.assert_array_equal(
            darr[:64, :64].compute(), direct_darr[:64, :64].compute()
        )
        assert darr[:64, :64].compute().mean() == 10.0

        # Bottom-right chunk
        np.testing.assert_array_equal(
            darr[64:, 64:].compute(), direct_darr[64:, 64:].compute()
        )
        assert darr[64:, 64:].compute().mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_tensor_pb_serialization(self, server_client):
        """Test SerializedTensor can be serialized to bytes and reconstructed."""
        pb = server_client.get_tensor_pb("test-tensor")

        # Serialize to bytes
        serialized_bytes = pb.SerializeToString()

        # Deserialize
        from biopb.tensor.serialized_pb2 import SerializedTensor

        pb2 = SerializedTensor.FromString(serialized_bytes)

        # Reconstruct array from deserialized protobuf
        darr = TensorFlightClient.tensor_from_pb(pb2)

        # Verify data is correct
        assert darr[:64, :64].compute().mean() == 10.0
        assert darr[64:, 64:].compute().mean() == 40.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_pb_with_slice_hint(self, server_client):
        """Test get_tensor_pb with slice_hint cropping."""
        pb = server_client.get_tensor_pb(
            "test-tensor",
            slice_hint=(slice(0, 64), slice(0, 64)),  # Top-left quadrant
        )

        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import SliceHint

        # The requested slice rides the plan's app_metadata.
        info = flight.FlightInfo.deserialize(pb.flight_info)
        requested = SliceHint.FromString(info.app_metadata)
        assert list(requested.start) == [0, 0]
        assert list(requested.stop) == [64, 64]

        # Reconstruct and verify cropping
        darr = TensorFlightClient.tensor_from_pb(pb)
        assert darr.shape == (64, 64)
        assert darr.compute().mean() == 10.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_pb_with_scale_hint(self, server_client):
        """Test get_tensor_pb with scale_hint."""
        pb = server_client.get_tensor_pb(
            "test-tensor",
            scale_hint=[2, 2],
            reduction_method="nearest",
        )

        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import TensorDescriptor

        # Verify scale_hint in the plan's descriptor
        info = flight.FlightInfo.deserialize(pb.flight_info)
        assert list(
            TensorDescriptor.FromString(info.descriptor.command).scale_hint
        ) == [2, 2]

        # Reconstruct and verify downscaled shape
        darr = TensorFlightClient.tensor_from_pb(pb)
        assert darr.shape == (64, 64)

        # Verify data values
        assert darr[:32, :32].compute().mean() == 10.0
        assert darr[32:, 32:].compute().mean() == 40.0
