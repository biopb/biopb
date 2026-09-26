"""Pytest configuration for biopb-image-runtime tests."""

import os
import socket
import subprocess
import sys
import textwrap
import time
from typing import Generator

import grpc
import pytest

_OPS_SERVER = """
import numpy as np
from biopb_image_base import Tensor, op, serve


@op(description="Echo the image back", labels=["mock"])
def mock_echo(image: Tensor("YX")):
    return image


@op(description="Echo the image back, by reference", input="lazy")
def mock_echo_lazy(image: Tensor("YX")):
    return image


@op(description="Random labels", labels=["mock"])
def mock_random(image: Tensor("YX"), seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, image.shape).astype(np.uint16)


@op(description="Mean intensity, and the image")
def mock_stats(image: Tensor("YX"), scale: float = 1.0):
    return {"mean": float(image.mean()) * scale}, image


if __name__ == "__main__":
    serve()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def ops_server(tmp_path_factory) -> Generator[str, None, None]:
    """An Ops server from a server file, returning large results through its
    embedded tensor server. Yields its address."""
    root = tmp_path_factory.mktemp("ops-server")
    script = root / "server.py"
    script.write_text(textwrap.dedent(_OPS_SERVER))
    port, tensor_port = _free_port(), _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--port",
            str(port),
            "--cache-dir",
            str(root / "cache"),
            "--cache-size",
            "1GB",
            "--tensor-port",
            str(tensor_port),
            "--tensor-external-location",
            f"grpc://127.0.0.1:{tensor_port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "BIOPB_LOG_LEVEL": "WARNING"},
    )
    address = f"127.0.0.1:{port}"
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    channel = grpc.insecure_channel(address)
    try:
        for _ in range(60):
            try:
                health_pb2_grpc.HealthStub(channel).Check(
                    health_pb2.HealthCheckRequest(), timeout=1
                )
                break
            except grpc.RpcError:
                time.sleep(0.5)
        yield address
    finally:
        channel.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
