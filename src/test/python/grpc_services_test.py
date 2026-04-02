import os
import pytest

import grpc
import numpy as np

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy


# ---------------------------------------------------------------------------
# shared channel fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def service_channel():
    """Return an open channel to the running combined endpoint.

    By default we assume an insecure gRPC endpoint on localhost port 50051.
    For remote access the address may be given as a URL using either
    ``http://`` (insecure) or ``https://`` (TLS) scheme.  If the latter is
    used the channel will be created with ``grpc.ssl_channel_credentials``.

    ``SERVICE_SERVER`` holds the address/URL.  A separate ``SSL_CA_CERT``
    environment variable may point to a PEM file containing the root CA used
    by the proxy; if omitted the system default roots are used.
    """
    raw = os.environ.get("SERVICE_SERVER", "127.0.0.1:50051")

    # interpret URL-like values
    if "://" in raw:
        from urllib.parse import urlparse
        parts = urlparse(raw)
        scheme = parts.scheme.lower()
        hostport = parts.netloc
    else:
        scheme = ""
        hostport = raw

    if scheme in ("https",):
        ca_path = os.environ.get("SSL_CA_CERT")
        if ca_path:
            with open(ca_path, "rb") as f:
                creds = grpc.ssl_channel_credentials(root_certificates=f.read())
        else:
            creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(hostport, creds)
    else:
        # treat anything else as insecure (http or plain host:port)
        channel = grpc.insecure_channel(hostport)

    try:
        grpc.channel_ready_future(channel).result(timeout=3)
    except grpc.FutureTimeoutError:
        pytest.skip(f"gRPC endpoint {raw} not reachable")
    yield channel
    channel.close()


# ---------------------------------------------------------------------------
# optional health check (if server implements the standard health service)
# ---------------------------------------------------------------------------

try:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    _health_available = True
except ImportError:
    _health_available = False


def test_grpc_health(service_channel):
    if not _health_available:
        pytest.skip("grpcio-health-checking package not installed")

    stub = health_pb2_grpc.HealthStub(service_channel)
    req = health_pb2.HealthCheckRequest(service="")
    try:
        resp = stub.Check(req, timeout=2)
    except grpc.RpcError as e:
        pytest.skip(f"Health service not available: {e.code()}")
    assert resp.status == health_pb2.HealthCheckResponse.SERVING


# ---------------------------------------------------------------------------
# functional tests for each protocol
# ---------------------------------------------------------------------------

def _make_small_image(shape=(1, 8, 8, 1)):
    arr = (np.random.rand(*shape) * 255).astype("uint8")
    return serialize_from_numpy(arr)


def test_process_run(service_channel):
    stub = proto.ProcessImageStub(service_channel)
    req = proto.ProcessRequest(
        image_data=proto.ImageData(pixels=_make_small_image()),
        op_name="test",
    )
    try:
        resp = stub.Run(req, timeout=5)
    except grpc.RpcError as e:
        pytest.skip(f"ProcessImage.Run RPC failed: {e.code()}")
    assert isinstance(resp, proto.ProcessResponse)
    assert resp.image_data.pixels
    # verify we can round-trip the returned pixels
    out = deserialize_to_numpy(resp.image_data.pixels)
    assert out.ndim >= 3


def test_detection_run(service_channel):
    stub = proto.ObjectDetectionStub(service_channel)
    req = proto.DetectionRequest(
        image_data=proto.ImageData(pixels=_make_small_image()),
    )
    try:
        resp = stub.RunDetection(req, timeout=5)
    except grpc.RpcError as e:
        pytest.skip(f"ObjectDetection.RunDetection RPC failed: {e.code()}")
    assert isinstance(resp, proto.DetectionResponse)


@pytest.mark.skip(reason="RunDetectionStream not implemented yet")
def test_detection_stream(service_channel):
    stub = proto.ObjectDetectionStub(service_channel)

    def gen_requests():
        for _ in range(2):
            yield proto.DetectionRequest(
                image_data=proto.ImageData(pixels=_make_small_image()),
            )

    try:
        responses = stub.RunDetectionStream(gen_requests(), timeout=10)
    except grpc.RpcError as e:
        pytest.skip(f"ObjectDetection.RunDetectionStream RPC failed: {e.code()}")

    count = 0
    for r in responses:
        count += 1
        assert isinstance(r, proto.DetectionResponse)
    assert count >= 1
