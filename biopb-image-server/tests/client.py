"""Test script for biopb gRPC services.

Tests ObjectDetection and ProcessImage services with both eager and lazy data.
"""

import logging
import time
from pathlib import Path

import biopb.image as proto
import grpc
import imageio.v2 as imageio
import numpy as np
import typer
from grpc_health.v1 import health_pb2, health_pb2_grpc

from biopb_image_base.common import _AUTH_HEADER_KEY
from biopb_image_base.logging_config import setup_logging

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)


def test_health(channel: grpc.Channel, metadata: tuple) -> bool:
    """Test the health check endpoint."""
    try:
        stub = health_pb2_grpc.HealthStub(channel)
        request = health_pb2.HealthCheckRequest()
        response = stub.Check(request, metadata=metadata, timeout=5)
        status = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
        print(f"Health check: {status}")
        return response.status == health_pb2.HealthCheckResponse.SERVING
    except grpc.RpcError as e:
        logger.error(f"Health check failed: {e}")
        return False


def construct_request(image: np.ndarray, use_lazy: bool = False) -> proto.DetectionRequest:
    """Construct a DetectionRequest from numpy array."""
    from biopb.image.utils import serialize_from_numpy_to_image_data

    image_data = serialize_from_numpy_to_image_data(image, dim_labels=["Y", "X", "C"])

    return proto.DetectionRequest(
        image_data=image_data,
        detection_settings=proto.DetectionSettings(
            scaling_hint=1.0,
        ),
    )


@app.command()
def main(
    port: int = 50051,
    ip: str = "127.0.0.1",
    token: str = "",
    image_path: Path = Path(__file__).parent / "test_image.png",
    debug: bool = False,
    health: bool = True,
    mode: str = "echo",
):
    """Test biopb.image gRPC server."""
    setup_logging("DEBUG" if debug else "INFO")

    SERVER = f"{ip}:{port}"
    METADATA = ((_AUTH_HEADER_KEY, "Bearer " + token.strip()),) if token else ()
    logger.info(f"Testing server at {SERVER}")

    # Test health check first
    if health:
        logger.info("Testing health check...")
        with grpc.insecure_channel(SERVER) as channel:
            if not test_health(channel, METADATA):
                logger.error("Health check failed, server may not be ready")
                return

    test_image = imageio.imread(image_path)
    logger.info(f"Loaded image {image_path} with shape {test_image.shape}")

    def _test_with_image(image, label: str = ""):
        """Test ObjectDetection and ProcessImage with an image."""
        logger.info(f"Testing image {label or 'default'}: shape={image.shape}")
        start_time = time.perf_counter()

        # Test ObjectDetection service
        try:
            with grpc.insecure_channel(SERVER) as channel:
                stub = proto.ObjectDetectionStub(channel)
                response = stub.RunDetection(
                    construct_request(image),
                    metadata=METADATA,
                    timeout=30,
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(
                f"  ObjectDetection: {len(response.detections)} detections "
                f"in {elapsed_ms:.1f}ms"
            )

        except grpc.RpcError as e:
            logger.error(f"ObjectDetection call failed: {e}")
            return False

        # Test ProcessImage service
        start_time = time.perf_counter()
        try:
            with grpc.insecure_channel(SERVER) as channel:
                stub = proto.ProcessImageStub(channel)
                request = proto.ProcessRequest(
                    image_data=proto.ImageData(
                        eager_data=proto.Tensor(
                            bindata=proto.BinData(
                                data=image.tobytes(),
                                endianness=1 if image.dtype.byteorder == '<' else 0,
                            ),
                            dtype=image.dtype.str.replace('|', '').replace('<', '').replace('>', ''),
                            dims=list(image.shape),
                            dim_labels=["Y", "X", "C"] if image.ndim == 3 else ["Y", "X"],
                        )
                    ),
                    op_name=mode,
                )
                response = stub.Run(
                    request,
                    metadata=METADATA,
                    timeout=30,
                )

            # Decode response
            from biopb.image.utils import deserialize_image_data
            result = deserialize_image_data(response.image_data)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if isinstance(result, np.ndarray):
                print(f"  ProcessImage: {result.shape} in {elapsed_ms:.1f}ms (eager)")
            else:
                print(f"  ProcessImage: {result.shape} in {elapsed_ms:.1f}ms (lazy dask)")
            return True

        except grpc.RpcError as e:
            logger.error(f"ProcessImage call failed: {e}")
            return False

    # Test with different image sizes
    results = []
    results.append(_test_with_image(test_image, "original"))

    cropped = test_image[:373, :372]
    results.append(_test_with_image(cropped, "cropped"))

    padded = np.pad(test_image, [[0, 128], [0, 128], [0, 0]] if test_image.ndim == 3 else [[0, 128], [0, 128]])
    results.append(_test_with_image(padded, "padded"))

    if all(results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        raise SystemExit(1)


@app.command()
def streaming(
    port: int = 50051,
    ip: str = "127.0.0.1",
    token: str = "",
    image_path: Path = Path(__file__).parent / "test_image.png",
    debug: bool = False,
    iterations: int = 4,
):
    """Test streaming endpoints."""
    setup_logging("DEBUG" if debug else "INFO")

    SERVER = f"{ip}:{port}"
    METADATA = ((_AUTH_HEADER_KEY, "Bearer " + token.strip()),) if token else ()

    test_image = imageio.imread(image_path)

    def _stream_messages(image, n: int = 4):
        """Generate streaming request messages."""
        from biopb.image.utils import serialize_from_numpy_to_image_data
        yield proto.ProcessRequest(
            image_data=serialize_from_numpy_to_image_data(image),
        )
        for _ in range(n - 1):
            yield proto.ProcessRequest()

    logger.info(f"Testing streaming with {iterations} iterations")

    try:
        with grpc.insecure_channel(SERVER) as channel:
            stub = proto.ProcessImageStub(channel)
            response_count = 0
            start_time = time.perf_counter()

            for response in stub.RunStream(
                _stream_messages(test_image, iterations),
                metadata=METADATA,
                timeout=60,
            ):
                response_count += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                print(f"  Response {response_count}: received in {elapsed_ms:.1f}ms")

        print("\nStreaming test passed!")

    except grpc.RpcError as e:
        logger.error(f"Streaming test failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    app()