"""Pytest configuration for biopb-image-server tests."""

import subprocess
import time
import os
import tempfile
import shutil
from typing import Generator

import grpc
import pytest


@pytest.fixture(scope="session")
def mock_server() -> Generator[str, None, None]:
    """Start mock server for testing.

    Returns server address (e.g., "127.0.0.1:50051").
    """
    # Create temp cache directory
    cache_dir = tempfile.mkdtemp(prefix="biopb-test-cache-")

    # Check if mock_servicer module is available
    try:
        import biopb_image_base.mock_servicer
    except ImportError:
        pytest.skip("biopb_image_base.mock_servicer not available")

    # Start mock server as subprocess
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "biopb_image_base.mock_servicer",
            "--port",
            "50051",
            "--local",
            "--cache-dir",
            cache_dir,
            "--cache-size",
            "1GB",
            "--tensor-port",
            "8817",
            "--tensor-external-location",
            "grpc://127.0.0.1:8817",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "BIOPB_LOG_LEVEL": "WARNING"},
    )

    # Wait for server to be ready
    server_addr = "127.0.0.1:50051"
    max_wait = 10

    for _ in range(max_wait):
        try:
            channel = grpc.insecure_channel(server_addr)
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest()
            stub.Check(request, timeout=2)
            break
        except Exception:
            pass
        time.sleep(1)

    yield server_addr

    # Cleanup: terminate server and remove cache
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Clean cache directory
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def grpc_channel(mock_server: str) -> grpc.Channel:
    """Create gRPC channel to mock server."""
    return grpc.insecure_channel(mock_server)