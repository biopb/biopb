"""Tests for TensorFlightClient using the new multifield API."""

import os
import tempfile
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest
from biopb.tensor import (
    TensorFlightClient,
)
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.serialized_pb2 import SerializedTensor

try:
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    _server_available = True
except ImportError:
    _server_available = False

_needs_server = pytest.mark.skipif(
    not _server_available, reason="biopb-tensor-server not installed"
)


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr

        _ = zarr.open_array
        return True
    except ImportError:
        return False


class TestTensorFlightClient:
    """Tests for TensorFlightClient with new multifield API."""

    @pytest.fixture
    def server_client(self):
        """Start server and create client."""
        if not _server_available:
            pytest.skip("biopb-tensor-server not installed")
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

            server = TensorFlightServer("grpc://localhost:8890")
            server.register_source("test-tensor", adapter)

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
    def test_get_tensor_shape(self, server_client):
        """Test tensor shape retrieval."""
        darr = server_client.get_tensor("test-tensor", "test-tensor")
        assert darr.shape == (128, 128)
        assert darr.dtype == np.uint8

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_read_chunks(self, server_client):
        """Test reading different chunks."""
        darr = server_client.get_tensor("test-tensor", "test-tensor")

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
        darr = server_client.get_tensor("test-tensor", "test-tensor")

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
            "test-tensor",
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

    @_needs_server
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

    @_needs_server
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

    @_needs_server
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
        with pytest.raises(ValueError, match="Tensor 'nonexistent' not found"):
            server_client.get_tensor("test-tensor", "nonexistent")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_source_not_found_raises(self, server_client):
        """Test that requesting non-existent source raises error."""
        with pytest.raises(ValueError, match="Source not found"):
            server_client.get_tensor("nonexistent-source", "some-tensor")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_pb(self, server_client):
        """Test get_tensor_pb returns SerializedTensor protobuf."""
        from biopb.tensor.serialized_pb2 import SerializedTensor

        pb = server_client.get_tensor_pb("test-tensor", "test-tensor")

        # Verify it's a SerializedTensor
        assert isinstance(pb, SerializedTensor)

        # Verify descriptor is populated
        assert pb.tensor_descriptor.array_id == "test-tensor"
        assert list(pb.tensor_descriptor.shape) == [128, 128]
        # dtype may be uint8 or |u1 depending on server
        assert pb.tensor_descriptor.dtype in ("uint8", "|u1")
        assert list(pb.tensor_descriptor.chunk_shape) == [64, 64]

        # Verify location is populated
        assert pb.location == "grpc://localhost:8890"

        # Verify endpoints are populated
        assert len(pb.endpoints) == 4  # 4 chunks

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_tensor_from_pb(self, server_client):
        """Test tensor_from_pb reconstructs dask array."""
        pb = server_client.get_tensor_pb("test-tensor", "test-tensor")

        # Reconstruct array
        darr = TensorFlightClient.tensor_from_pb(pb)

        # Verify shape and dtype
        assert darr.shape == (128, 128)
        assert darr.dtype == np.uint8

        # Verify data values match direct get_tensor
        direct_darr = server_client.get_tensor("test-tensor", "test-tensor")

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
        pb = server_client.get_tensor_pb("test-tensor", "test-tensor")

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
            "test-tensor",
            slice_hint=(slice(0, 64), slice(0, 64)),  # Top-left quadrant
        )

        # Verify original_slice_hint is populated
        assert pb.HasField("original_slice_hint")
        assert list(pb.original_slice_hint.start) == [0, 0]
        assert list(pb.original_slice_hint.stop) == [64, 64]

        # Reconstruct and verify cropping
        darr = TensorFlightClient.tensor_from_pb(pb)
        assert darr.shape == (64, 64)
        assert darr.compute().mean() == 10.0

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_pb_with_scale_hint(self, server_client):
        """Test get_tensor_pb with scale_hint."""
        pb = server_client.get_tensor_pb(
            "test-tensor",
            "test-tensor",
            scale_hint=[2, 2],
            reduction_method="nearest",
        )

        # Verify scale_hint in descriptor
        assert list(pb.tensor_descriptor.scale_hint) == [2, 2]

        # Reconstruct and verify downscaled shape
        darr = TensorFlightClient.tensor_from_pb(pb)
        assert darr.shape == (64, 64)

        # Verify data values
        assert darr[:32, :32].compute().mean() == 10.0
        assert darr[32:, 32:].compute().mean() == 40.0

    def test_get_upload_status_pb_uses_tensor_descriptor_array_id(self):
        client = TensorFlightClient("grpc://localhost:8890", cache_bytes=10_000_000)
        # Upload lifecycle lives in the UploadSession collaborator (#278 item C),
        # so the _pb conveniences resolve status through client._upload -- mock there.
        client._upload.get_upload_status = Mock(
            return_value={
                "source_id": "cache_test",
                "state": "PENDING",
                "expected_chunks": 4,
                "uploaded_chunks": 1,
            }
        )

        pb = SerializedTensor(tensor_descriptor=TensorDescriptor(array_id="cache_test"))

        try:
            status = client.get_upload_status_pb(pb)
        finally:
            client.close()

        client._upload.get_upload_status.assert_called_once_with("cache_test")
        assert status["state"] == "PENDING"

    def test_wait_for_upload_ready_pb_returns_when_ready(self):
        client = TensorFlightClient("grpc://localhost:8890", cache_bytes=10_000_000)
        client._upload.get_upload_status = Mock(
            side_effect=[
                {
                    "source_id": "cache_test",
                    "state": "PENDING",
                    "expected_chunks": 4,
                    "uploaded_chunks": 1,
                },
                {
                    "source_id": "cache_test",
                    "state": "READY",
                    "expected_chunks": 4,
                    "uploaded_chunks": 4,
                },
            ]
        )

        pb = SerializedTensor(tensor_descriptor=TensorDescriptor(array_id="cache_test"))

        try:
            status = client.wait_for_upload_ready_pb(
                pb,
                timeout_seconds=0.1,
                poll_interval_seconds=0.0,
            )
        finally:
            client.close()

        assert status["state"] == "READY"
        assert client._upload.get_upload_status.call_count == 2

    def test_wait_for_upload_ready_pb_times_out(self):
        client = TensorFlightClient("grpc://localhost:8890", cache_bytes=10_000_000)
        client._upload.get_upload_status = Mock(
            return_value={
                "source_id": "cache_test",
                "state": "PENDING",
                "expected_chunks": 4,
                "uploaded_chunks": 1,
            }
        )

        pb = SerializedTensor(tensor_descriptor=TensorDescriptor(array_id="cache_test"))

        try:
            with pytest.raises(
                TimeoutError, match="Timed out waiting for upload readiness"
            ):
                client.wait_for_upload_ready_pb(
                    pb,
                    timeout_seconds=0.0,
                    poll_interval_seconds=0.0,
                )
        finally:
            client.close()

    def test_get_upload_status_pb_requires_array_id(self):
        client = TensorFlightClient("grpc://localhost:8890", cache_bytes=10_000_000)
        pb = SerializedTensor(tensor_descriptor=TensorDescriptor())

        try:
            with pytest.raises(
                ValueError, match="tensor_descriptor.array_id is required"
            ):
                client.get_upload_status_pb(pb)
        finally:
            client.close()

    def test_wait_for_upload_ready_pb_raises_on_failed_state(self):
        client = TensorFlightClient("grpc://localhost:8890", cache_bytes=10_000_000)
        client._upload.get_upload_status = Mock(
            return_value={
                "source_id": "cache_test",
                "state": "FAILED",
                "expected_chunks": 4,
                "uploaded_chunks": 2,
            }
        )

        pb = SerializedTensor(tensor_descriptor=TensorDescriptor(array_id="cache_test"))

        try:
            with pytest.raises(
                RuntimeError, match="Upload failed for source 'cache_test'"
            ):
                client.wait_for_upload_ready_pb(
                    pb,
                    timeout_seconds=0.1,
                    poll_interval_seconds=0.0,
                )
        finally:
            client.close()


class TestQuerySourcesFormat:
    """query_sources output-format conversion (server-free).

    Exercises TensorFlightClient._format_query_result directly. The default
    stays 'arrow' (pyarrow.Table) for backward compatibility; 'pandas' and
    'records' are opt-in conveniences.
    """

    @staticmethod
    def _table():
        import pyarrow as pa

        return pa.table(
            {
                "source_id": ["a", "b"],
                # list column mirrors the real `shape_summary` catalog field
                "shape_summary": [[1, 4, 5734, 5734], [1, 5, 7616, 7616]],
            }
        )

    def test_default_format_is_arrow(self):
        # Backward-compat guard: the historical pyarrow.Table return must stay
        # the default so existing Arrow consumers don't break.
        import inspect

        default = (
            inspect.signature(TensorFlightClient.query_sources)
            .parameters["format"]
            .default
        )
        assert default == "arrow"

    def test_arrow_passthrough_is_same_object(self):
        t = self._table()
        assert TensorFlightClient._format_query_result(t, "arrow") is t

    def test_pandas_format_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        t = self._table()
        out = TensorFlightClient._format_query_result(t, "pandas")
        assert isinstance(out, pd.DataFrame)
        assert list(out["source_id"]) == ["a", "b"]
        assert list(out["shape_summary"].iloc[0]) == [1, 4, 5734, 5734]

    def test_pandas_string_nulls_become_none_not_nan(self):
        # issue #47: a NULL in a *string* column (e.g. metadata_json) coerces
        # to a float NaN under Arrow->pandas, and NaN is *truthy* -- so the
        # obvious `if row.metadata_json:` guard passes and json.loads() then
        # blows up on a float. We normalize string-column nulls to None.
        pytest.importorskip("pandas")
        import pyarrow as pa

        t = pa.table(
            {
                # mixed: one row has metadata, one is NULL -> Arrow `string`
                # col with nulls (the sharp case; an all-NULL page is already
                # None because it becomes an Arrow `null`-typed column).
                "source_id": ["a", "b"],
                "metadata_json": pa.array(['{"x": 1}', None], type=pa.string()),
            }
        )
        out = TensorFlightClient._format_query_result(t, "pandas")
        missing = out["metadata_json"].iloc[1]
        assert missing is None
        assert not missing  # falsy, so `if row.metadata_json:` is skipped
        assert out["metadata_json"].iloc[0] == '{"x": 1}'

    def test_pandas_numeric_nan_is_untouched(self):
        # The fix targets string columns by Arrow schema, so a genuine NaN in a
        # real float column must survive (we only normalize text nulls).
        pytest.importorskip("pandas")
        import math

        import pyarrow as pa

        t = pa.table({"score": pa.array([1.5, None], type=pa.float64())})
        out = TensorFlightClient._format_query_result(t, "pandas")
        assert math.isnan(out["score"].iloc[1])

    def test_records_returns_list_of_dicts(self):
        t = self._table()
        out = TensorFlightClient._format_query_result(t, "records")
        assert out == [
            {"source_id": "a", "shape_summary": [1, 4, 5734, 5734]},
            {"source_id": "b", "shape_summary": [1, 5, 7616, 7616]},
        ]

    def test_unknown_format_rejected_before_network(self):
        # Validated at the top of query_sources (now on CatalogClient, #278 item
        # C), so a bad format fails fast without a server / connection.
        from biopb.tensor._session import CatalogClient, _ClientState

        client = TensorFlightClient.__new__(TensorFlightClient)
        client._catalog = CatalogClient(
            _ClientState(
                client=None, call_options=None, location="", token=None, cache_bytes=0
            )
        )
        with pytest.raises(ValueError, match="unknown format"):
            client.query_sources("SELECT 1", format="polars")


class TestDescriptorKey:
    """_descriptor_key normalization (server-free, issue #45).

    The descriptor cache key must be source-unique and must collapse the bare
    and source-qualified array_id forms (which different RPCs emit) for the same
    tensor onto a single key.
    """

    def test_bare_and_qualified_forms_collapse_to_one_key(self):
        # An old data endpoint returns "src/Image:0"; list_sources returns the
        # bare "Image:0". Both are the same tensor -> one key.
        assert TensorFlightClient._descriptor_key(
            "src", "Image:0"
        ) == TensorFlightClient._descriptor_key("src", "src/Image:0")

    def test_same_bare_id_different_sources_do_not_collide(self):
        # Two aicsimageio files both name their tensor "Image:0"; keys must differ.
        assert TensorFlightClient._descriptor_key(
            "aics_aaa", "Image:0"
        ) != TensorFlightClient._descriptor_key("aics_bbb", "Image:0")

    def test_only_own_source_prefix_is_stripped(self):
        # A leading prefix that isn't this source's id is left intact.
        assert TensorFlightClient._descriptor_key("src", "other/Image:0") == (
            "src",
            "other/Image:0",
        )


class TestGetPhysicalScale:
    """get_physical_scale reads the descriptor summary (server-free).

    Exercises the client accessor for the per-dim physical-scale summary the
    server folds onto the descriptor (issue #31), driving the descriptor /
    source caches directly so no connection is needed.
    """

    @staticmethod
    def _client():
        # Build without __init__ (no connection): wire the shared state + the
        # CatalogClient collaborator that now owns get_physical_scale (#278 item
        # C). The method only touches the in-memory descriptor cache and (on a
        # miss) the catalog's _fetch_tensor_descriptor, which we stub there.
        from biopb.tensor._session import CatalogClient, ChunkFetcher, _ClientState

        client = TensorFlightClient.__new__(TensorFlightClient)
        state = _ClientState(
            client=None, call_options=None, location="", token=None, cache_bytes=0
        )
        client._state = state
        client._catalog = CatalogClient(state)
        client._fetcher = ChunkFetcher(state, client._catalog)
        client._catalog._fetch_tensor_descriptor = Mock()
        return client

    @staticmethod
    def _desc(array_id, scale=None, unit=None):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

        desc = TensorDescriptor(array_id=array_id, dim_labels=["z", "y", "x"])
        if scale is not None:
            desc.physical_scale[:] = scale
            desc.physical_unit[:] = unit
        return desc, DataSourceDescriptor

    def test_reads_cached_descriptor_without_rpc(self):
        # A descriptor cached by a prior get_tensor() carries the summary, so
        # get_physical_scale returns it with no extra fetch.
        client = self._client()
        desc, _ = self._desc(
            "t1", [2.0, 0.325, 0.325], ["micrometer", "micrometer", "micrometer"]
        )
        client._descriptors[client._descriptor_key("src", "t1")] = desc

        # Addressed by the qualified array_id; cache hit -> no fetch.
        scale, unit = client.get_physical_scale("src/t1")
        assert scale == [2.0, 0.325, 0.325]
        assert unit == ["micrometer", "micrometer", "micrometer"]
        client._catalog._fetch_tensor_descriptor.assert_not_called()

    def test_none_when_summary_empty(self):
        # Old server / no physical sizes -> empty repeated field -> None.
        client = self._client()
        desc, _ = self._desc("t1")  # no physical_scale set
        client._descriptors[client._descriptor_key("src", "t1")] = desc

        assert client.get_physical_scale("src/t1") is None

    def test_fetches_descriptor_when_not_cached(self):
        # Not in the descriptor cache: a GetFlightInfo fetch (stubbed here via
        # _fetch_tensor_descriptor) supplies the summary. A bare source id
        # resolves the source's default tensor. No get_source / _sources fallback
        # (removed with the array_id-keyed accessor, #75).
        client = self._client()
        desc, _ = self._desc("t1", [1.0, 0.5, 0.5], ["", "micrometer", "micrometer"])
        client._catalog._fetch_tensor_descriptor.return_value = desc

        scale, unit = client.get_physical_scale("src")  # bare source id -> default
        assert scale == [1.0, 0.5, 0.5]
        assert unit == ["", "micrometer", "micrometer"]
        client._catalog._fetch_tensor_descriptor.assert_called_once_with("src", None)

    def test_fetch_error_propagates(self):
        # A real fetch failure (server unreachable, source not found) must NOT be
        # swallowed into None: that would make it indistinguishable from "no
        # physical scale recorded". Only a fetched descriptor with an empty
        # summary yields None (test_none_when_summary_empty).
        client = self._client()
        client._catalog._fetch_tensor_descriptor.side_effect = ConnectionError(
            "unreachable"
        )

        with pytest.raises(ConnectionError):
            client.get_physical_scale("src")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
