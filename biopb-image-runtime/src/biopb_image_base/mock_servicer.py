"""Mock servicer for biopb.image gRPC services.

Provides echo/random responses for testing infrastructure without real ML models.
Useful for:
- Testing gRPC connectivity and health checks
- Benchmarking network throughput
- Development and CI environments
"""

import logging
import random
from typing import Optional

import biopb.image as proto
import dask.array as da
import numpy as np
import typer

from biopb_image_base.common import (
    BiopbServicerBase,
    RequestLogger,
    abort_invalid_argument,
    decode_image_data,
    return_lazy_or_eager,
)
from biopb_image_base.logging_config import get_log_level_from_env
from biopb_image_base.server import run_server

logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)


class MockServicer(BiopbServicerBase):
    """Mock implementation of ObjectDetection and ProcessImage services.

    Returns random/echo responses for testing.
    """

    def __init__(
        self,
        mode: str = "echo",  # "echo" or "random"
    ):
        super().__init__(use_lock=False)
        self._mode = mode

    def RunDetection(self, request, context):
        """Return random detections."""
        with self._server_context(context):
            with RequestLogger("RunDetection", request.ByteSize()) as log:
                # Get image dimensions if provided
                image_data = request.image_data
                if image_data is not None:
                    try:
                        img = decode_image_data(image_data)
                        # Handle both numpy and dask arrays
                        if hasattr(img, 'shape'):
                            shape = img.shape
                        else:
                            shape = (512, 512)  # fallback
                        height = shape[0] if len(shape) > 0 else 512
                        width = shape[1] if len(shape) > 1 else 512
                    except Exception:
                        height, width = 512, 512
                else:
                    height, width = 512, 512

                # Generate random detections
                num_detections = random.randint(5, 50)
                detections = []

                for _ in range(num_detections):
                    # Random bounding box
                    cx = random.uniform(0, width)
                    cy = random.uniform(0, height)
                    size = random.uniform(10, min(100, min(width, height) / 4))

                    roi = proto.ROI(
                        rectangle=proto.Rectangle(
                            top_left=proto.Point(x=cx - size/2, y=cy - size/2),
                            bottom_right=proto.Point(x=cx + size/2, y=cy + size/2),
                        )
                    )

                    score = random.uniform(0.5, 1.0)
                    detections.append(proto.ScoredROI(roi=roi, score=score))

                response = proto.DetectionResponse(detections=detections)
                log.response_size = response.ByteSize()

                logger.info(f"Mock RunDetection: returning {num_detections} detections")
                return response

    def Run(self, request, context):
        """Return input image (echo) or random labels mask."""
        with self._server_context(context):
            with RequestLogger("Run", request.ByteSize()) as log:
                image_data = request.image_data

                if image_data is None:
                    abort_invalid_argument(context, "image_data required")

                img = decode_image_data(image_data)

                if self._mode == "echo":
                    # Return input as-is
                    result = img
                    logger.info("Mock Run (echo): returning input image")
                else:
                    # Return random labels mask
                    if isinstance(img, np.ndarray):
                        result = np.random.randint(0, 100, size=img.shape[:2], dtype=np.int32)
                    else:
                        # Dask array - create random labels
                        result = da.random.randint(
                            0, 100, size=img.shape[:2], dtype=np.int32
                        )
                    logger.info("Mock Run (random): returning random labels mask")

                # Return as ImageData
                response = proto.ProcessResponse(
                    image_data=return_lazy_or_eager(
                        result,
                        tensor_cache=self._tensor_cache,
                        dim_labels=["Y", "X"] if result.ndim == 2 else None,
                    )
                )
                log.response_size = response.ByteSize()

                return response

    def RunStream(self, request_iterator, context):
        """Handle streaming process requests."""
        accumulated = None

        for request in request_iterator:
            if request.image_data is not None:
                img = decode_image_data(request.image_data)
                if accumulated is None:
                    accumulated = img
                else:
                    # Simple concatenation for streaming test
                    if isinstance(accumulated, np.ndarray) and isinstance(img, np.ndarray):
                        accumulated = np.concatenate([accumulated, img], axis=0)

        if accumulated is None:
            abort_invalid_argument(context, "No image data in stream")

        response = proto.ProcessResponse(
            image_data=return_lazy_or_eager(
                accumulated,
                tensor_cache=self._tensor_cache,
            )
        )
        yield response

    def GetOpNames(self, _request, _context):
        """Return supported operation names."""
        return proto.OpNames(
            names=["mock_echo", "mock_random"],
            op_schemas={
                "mock_echo": proto.OpSchema(
                    description="Echo mode - returns input image unchanged",
                    labels=["mock", "echo"],
                ),
                "mock_random": proto.OpSchema(
                    description="Random mode - returns random labels mask",
                    labels=["mock", "random"],
                ),
            }
        )


@app.command()
def main(
    port: int = 50051,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool = False,
    debug: bool = False,
    mode: str = "echo",
    cache_dir: Optional[str] = None,
    cache_size: str = "32GB",
    tensor_port: int = 8817,
    tensor_external_location: Optional[str] = None,
):
    """Run the mock biopb.image service.

    Args:
        port: gRPC port for image service
        ip: IP to bind
        local: Use local credentials (no external access)
        token: Enable token authentication
        debug: Enable debug logging
        mode: Mock mode - "echo" or "random"
        cache_dir: Directory for embedded tensor cache (enables lazy data)
        cache_size: Cache size for embedded tensor server (e.g., "32GB")
        tensor_port: Port for embedded tensor Flight server (default 8817)
        tensor_external_location: External URL for tensor server (e.g., "grpc://hostname:8817")
    """
    # Validate configuration before starting
    if cache_dir is not None and not local and ip == "0.0.0.0" and not tensor_external_location:
        typer.echo(
            "ERROR: --tensor-external-location is required when binding to 0.0.0.0 "
            "with embedded cache enabled (not --local mode).\n"
            "Set it to the externally reachable address "
            "(e.g., 'grpc://hostname:8817')",
            err=True,
        )
        raise typer.Exit(1)

    # Create servicer
    servicer = MockServicer(mode=mode)

    # Determine log level
    level = get_log_level_from_env() or ("DEBUG" if debug else "INFO")

    run_server(
        servicer=servicer,
        port=port,
        ip=ip,
        local=local,
        token=token,
        log_level=level,
        cache_dir=cache_dir,
        cache_size=cache_size,
        tensor_port=tensor_port,
        tensor_external_location=tensor_external_location,
    )


if __name__ == "__main__":
    app()
