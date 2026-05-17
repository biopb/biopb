"""Integration tests for biopb.image gRPC services."""

import os
import time
import numpy as np
import grpc
import grpc_health.v1.health_pb2 as health_pb2
import grpc_health.v1.health_pb2_grpc as health_pb2_grpc
import biopb.image as proto
from google.protobuf.empty_pb2 import Empty
from biopb.image.utils import serialize_from_numpy_to_image_data, deserialize_image_data


# gRPC channel options for large messages (no compression to avoid decompress errors)
_GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.max_send_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.enable_compression", 0),  # Disable compression for large data tests
]


def create_channel(server_addr: str) -> grpc.Channel:
    """Create gRPC channel with large message limits."""
    return grpc.insecure_channel(server_addr, options=_GRPC_OPTIONS)


class TestHealthCheck:
    """Tests for gRPC health check endpoint."""

    def test_health_serving(self, mock_server: str):
        """Server should report SERVING status."""
        channel = grpc.insecure_channel(mock_server)
        stub = health_pb2_grpc.HealthStub(channel)
        request = health_pb2.HealthCheckRequest()
        response = stub.Check(request, timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING


class TestObjectDetection:
    """Tests for ObjectDetection service."""

    def test_detection_eager_small_image(self, grpc_channel: grpc.Channel):
        """RunDetection with small eager image."""
        # Create test image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_data = serialize_from_numpy_to_image_data(image, dim_labels=["Y", "X", "C"])

        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        stub = proto.ObjectDetectionStub(grpc_channel)
        response = stub.RunDetection(request, timeout=30)

        # Mock returns random detections
        assert len(response.detections) > 0
        assert len(response.detections) <= 50  # Mock generates 5-50

    def test_detection_various_sizes(self, grpc_channel: grpc.Channel):
        """RunDetection with various image sizes."""
        stub = proto.ObjectDetectionStub(grpc_channel)

        sizes = [(256, 256), (512, 512), (1024, 1024), (373, 372)]  # Non-square

        for height, width in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image_data = serialize_from_numpy_to_image_data(image)
            request = proto.DetectionRequest(image_data=image_data)

            response = stub.RunDetection(request, timeout=30)
            assert len(response.detections) > 0


class TestProcessImageEager:
    """Tests for ProcessImage service with eager data."""

    def test_run_echo_eager(self, grpc_channel: grpc.Channel):
        """Run with echo mode returns input image."""
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        image_data = serialize_from_numpy_to_image_data(image, dim_labels=["Y", "X"])

        request = proto.ProcessRequest(image_data=image_data, op_name="mock_echo")

        stub = proto.ProcessImageStub(grpc_channel)
        response = stub.Run(request, timeout=30)

        # Decode response
        result = deserialize_image_data(response.image_data)

        # Echo mode returns same image
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    def test_run_random_mode(self, grpc_channel: grpc.Channel):
        """Run with random mode returns random labels."""
        image = np.zeros((512, 512), dtype=np.uint8)
        image_data = serialize_from_numpy_to_image_data(image)

        request = proto.ProcessRequest(image_data=image_data, op_name="mock_random")

        stub = proto.ProcessImageStub(grpc_channel)
        response = stub.Run(request, timeout=30)

        result = deserialize_image_data(response.image_data)
        assert isinstance(result, np.ndarray)
        assert result.shape[:2] == image.shape[:2]  # Same spatial dimensions


class TestProcessImageLazy:
    """Tests for ProcessImage service with lazy data handling."""

    def test_run_large_image_returns_lazy(self, mock_server: str):
        """Large images (>64MB) should return SerializedTensor."""
        # Create image larger than eager threshold (>64MB) but within gRPC limits
        # 8192 x 2049 x float32 = ~64.004MB - just above threshold
        large_image = np.random.rand(8192, 2049).astype(np.float32)
        nbytes = large_image.nbytes
        assert nbytes > 64 * 1024 * 1024, f"Image too small: {nbytes} bytes"

        image_data = serialize_from_numpy_to_image_data(large_image)

        request = proto.ProcessRequest(image_data=image_data, op_name="mock_echo")

        # Use channel with large message limits and no compression
        channel = create_channel(mock_server)
        stub = proto.ProcessImageStub(channel)
        response = stub.Run(request, timeout=60)

        # Response should have lazy_data (SerializedTensor)
        assert response.image_data.HasField("lazy_data"), "Large image should return lazy data"

        # Check SerializedTensor structure
        serialized = response.image_data.lazy_data
        assert serialized.location.startswith("grpc://")
        assert len(serialized.endpoints) > 0

    def test_lazy_location_is_localhost(self, mock_server: str):
        """For --local mode without explicit location, tensor location defaults to localhost."""
        large_image = np.random.rand(8192, 2049).astype(np.float32)
        image_data = serialize_from_numpy_to_image_data(large_image)

        request = proto.ProcessRequest(image_data=image_data, op_name="mock_echo")

        channel = create_channel(mock_server)
        stub = proto.ProcessImageStub(channel)
        response = stub.Run(request, timeout=60)

        assert response.image_data.HasField("lazy_data")
        serialized = response.image_data.lazy_data
        # Mock server fixture uses --tensor-external-location grpc://127.0.0.1:8817
        # If not specified, --local mode would default to grpc://localhost:8817
        assert serialized.location.startswith("grpc://"), \
            f"Location should be grpc:// URL, got {serialized.location}"
        assert "8817" in serialized.location, \
            f"Location should include tensor port 8817, got {serialized.location}"

    def test_local_mode_defaults_to_localhost_location(self):
        """Regression test: --local mode without explicit location uses localhost."""
        import subprocess
        import tempfile
        import shutil

        cache_dir = tempfile.mkdtemp(prefix="biopb-test-default-loc-")

        # Start server with --local but WITHOUT --tensor-external-location
        proc = subprocess.Popen([
            "python", "-m", "biopb_image_base.mock_servicer",
            "--port", "50053",
            "--local",
            "--cache-dir", cache_dir,
            "--cache-size", "1GB",
            "--tensor-port", "8819",
        ], env={**os.environ, "PYTHONPATH": "/home/jiyu/work/biopb/biopb-image-server/src", "BIOPB_LOG_LEVEL": "WARNING"})

        # Wait for server
        time.sleep(3)

        try:
            large_image = np.random.rand(8192, 2049).astype(np.float32)
            image_data = serialize_from_numpy_to_image_data(large_image)
            request = proto.ProcessRequest(image_data=image_data, op_name="mock_echo")

            options = [
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.enable_compression", 0),
            ]
            channel = grpc.insecure_channel("127.0.0.1:50053", options=options)
            stub = proto.ProcessImageStub(channel)
            response = stub.Run(request, timeout=60)

            assert response.image_data.HasField("lazy_data")
            serialized = response.image_data.lazy_data
            # Default for --local should be localhost, NOT 0.0.0.0
            assert serialized.location == "grpc://localhost:8819", \
                f"--local mode without explicit location should use localhost, got {serialized.location}"
        finally:
            proc.terminate()
            proc.wait()
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_lazy_data_fetchable(self, mock_server: str):
        """Lazy data should be fetchable from tensor server."""
        # Create large image (>64MB to trigger lazy)
        large_image = np.random.rand(8192, 2049).astype(np.float32)
        image_data = serialize_from_numpy_to_image_data(large_image)

        request = proto.ProcessRequest(image_data=image_data, op_name="mock_echo")

        # Use channel with large message limits and no compression
        channel = create_channel(mock_server)
        stub = proto.ProcessImageStub(channel)
        response = stub.Run(request, timeout=60)

        # Fetch lazy data using TensorFlightClient
        if response.image_data.HasField("lazy_data"):
            from biopb.tensor.client import TensorFlightClient

            serialized = response.image_data.lazy_data
            client = TensorFlightClient(serialized.location)

            # Reconstruct as dask array
            arr = client.tensor_from_pb(serialized)

            # Compute to get numpy
            result = arr.compute()

            # Verify shape matches
            assert result.shape == large_image.shape
            assert result.dtype == large_image.dtype


class TestStreaming:
    """Tests for streaming endpoints."""

    def test_run_stream_single_chunk(self, grpc_channel: grpc.Channel):
        """RunStream with single image chunk."""
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        image_data = serialize_from_numpy_to_image_data(image)

        def request_generator():
            yield proto.ProcessRequest(image_data=image_data)

        stub = proto.ProcessImageStub(grpc_channel)
        responses = list(stub.RunStream(request_generator(), timeout=30))

        assert len(responses) == 1
        result = deserialize_image_data(responses[0].image_data)
        assert result.shape == image.shape

    def test_run_stream_accumulates(self, grpc_channel: grpc.Channel):
        """RunStream accumulates multiple image chunks."""
        # Send two image chunks that get concatenated
        image1 = np.random.randint(0, 255, (128, 256), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (128, 256), dtype=np.uint8)

        def request_generator():
            yield proto.ProcessRequest(image_data=serialize_from_numpy_to_image_data(image1))
            yield proto.ProcessRequest(image_data=serialize_from_numpy_to_image_data(image2))

        stub = proto.ProcessImageStub(grpc_channel)
        responses = list(stub.RunStream(request_generator(), timeout=30))

        assert len(responses) == 1
        result = deserialize_image_data(responses[0].image_data)
        # Mock accumulates along axis 0
        assert result.shape[0] >= image1.shape[0] + image2.shape[0]


class TestGetOpNames:
    """Tests for GetOpNames endpoint."""

    def test_get_op_names(self, grpc_channel: grpc.Channel):
        """GetOpNames returns available operations."""
        stub = proto.ProcessImageStub(grpc_channel)
        response = stub.GetOpNames(Empty(), timeout=10)

        assert "mock_echo" in response.names
        assert "mock_random" in response.names