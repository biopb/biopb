import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Optional, Tuple

import biopb.image as proto
import grpc
import numpy as np
import pandas as pd
from biopb.image.utils import (
    deserialize_image_data,
    serialize_from_numpy_to_image_data,
)
from google.protobuf import empty_pb2, struct_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc
from napari.qt.threading import thread_worker

from ._chunking import FULL_ORDER, IterationSpec, _data_iterator

from ._config import load_config
from ._render import _adjust_response_offset, _generate_label
from ._typing import napari_data

logger = logging.getLogger(__name__)

# Sentinel value for signaling gRPC call start (for progress bar)
CALL_START = "call_start"

# Regex for parsing server URL with optional scheme prefix and path/label filter
_URL_PATTERN = re.compile(
    r"^(?:(https?)://)?([^/]+)(?:/([^/]+))?$"  # Optional http/https scheme, host:port, optional path
)

# Regex for validating hostname:port format (the "rest" part after scheme)
_HOST_PORT_PATTERN = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?:(\d+)$"
)


def _parse_server_url(
    server_url: str,
) -> Tuple[str, int, Optional[str], Optional[str]]:
    """Parse server URL extracting host, port, scheme, and optional label filter.

    Args:
        server_url: URL in one of these formats:
            - "https://hostname:port/label" (explicit HTTPS with label filter)
            - "http://hostname:port/label" (explicit HTTP with label filter)
            - "hostname:port/label" (auto-detect scheme, with label filter)
            - "hostname:port" (auto-detect, no filter)
            - "hostname" (auto-detect, defaults to port 443, no filter)

    Returns:
        Tuple of (host, port, explicit_scheme, label_filter) where:
        - explicit_scheme is "HTTP", "HTTPS", or None (meaning auto-detect)
        - label_filter is the path component (e.g., "filter") or None

    Raises:
        ValueError: If URL format is invalid
    """
    match = _URL_PATTERN.match(server_url.strip())
    if not match:
        raise ValueError(
            f"Invalid server URL format: '{server_url}'. "
            "Expected 'http://hostname:port', 'https://hostname:port', 'hostname:port', or 'hostname'"
        )

    explicit_scheme = match.group(1)  # None, "http", or "https"
    host_port = match.group(2)
    label_filter = match.group(3)  # None or path component like "filter"

    # Parse host:port from the host_port part
    host_port_match = _HOST_PORT_PATTERN.match(host_port)
    if host_port_match:
        port = int(host_port_match.group(1))
        host = host_port.rsplit(":", 1)[0]
        return host, port, explicit_scheme, label_filter

    # Check if it's a hostname without port (will default to 443)
    if re.match(
        r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$",
        host_port,
    ):
        return host_port, 443, explicit_scheme, label_filter

    raise ValueError(
        f"Invalid server URL format: '{server_url}'. "
        "Use format 'hostname:port' (e.g., 'localhost:50051') or 'hostname' (defaults to port 443)"
    )


def _get_label_filter(server_url: str) -> Optional[str]:
    """Extract label filter from server URL path component.

    Args:
        server_url: URL like "localhost:50051/filter" or "hostname:port"

    Returns:
        Label filter string (e.g., "filter") or None if no path component

    Raises:
        ValueError: If URL format is invalid
    """
    _, _, _, label_filter = _parse_server_url(server_url)
    return label_filter


def dict_to_struct(d: dict) -> struct_pb2.Struct:
    """Convert Python dict to protobuf Struct."""
    s = struct_pb2.Struct()
    s.update(d)
    return s


def struct_to_dict(s: struct_pb2.Struct) -> dict:
    """Convert protobuf Struct to Python dict."""
    return dict(s)


def _parse_annotation_tsv(annotation: str) -> pd.DataFrame:
    """Parse TSV annotation string to DataFrame.

    Args:
        annotation: TSV-formatted string with header row

    Returns:
        DataFrame with annotation data, empty DataFrame if no data

    Note:
        Handles comment lines (starting with #) and duplicate header rows.
        Comment lines are skipped before parsing.
        All rows matching header columns are filtered out.
    """
    if not annotation:
        return pd.DataFrame()

    # Split into lines and filter out comment lines (starting with #)
    lines = annotation.strip().split("\n")
    data_lines = [line for line in lines if not line.strip().startswith("#")]

    if not data_lines:
        return pd.DataFrame()

    # Join filtered lines and parse
    cleaned_tsv = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(cleaned_tsv), delimiter="\t")

    # Filter out rows where values match header columns (duplicate header rows)
    if len(df) > 0:
        header_values = df.columns.astype(str).tolist()
        mask = df.apply(
            lambda row: row.astype(str).tolist() == header_values, axis=1
        )
        df = df[~mask].reset_index(drop=True)

    return df


def _encode_image(
    image: np.ndarray, np_index_order: str = "YXC", z_ratio: float = 1.0
) -> proto.ImageData:
    """Encode numpy image array to protobuf ImageData format.

    Args:
        image: Input image array (must match np_index_order specification)
        np_index_order: Axis order string (e.g., "YXC", "ZYXC", "YX", "ZYX")
        z_ratio: Z aspect ratio for 3D images

    Returns:
        protobuf ImageData object

    Raises:
        ValueError: If image dimensions don't match np_index_order
    """
    expected_ndim = len(np_index_order)
    if image.ndim != expected_ndim:
        raise ValueError(
            f"Image must have {expected_ndim} dimensions for np_index_order '{np_index_order}'. "
            f"Got shape {image.shape} with {image.ndim} dimensions"
        )

    # Add batch dimension (B, ...) for processing
    image = image[None, ...]

    # Convert axis order string to list of individual labels
    dim_labels = list(np_index_order)

    image_data = serialize_from_numpy_to_image_data(
        image,
        dim_labels=["B"] + dim_labels,
    )

    return image_data


def _object_detection_build_request(
    image: np.ndarray, settings: dict
) -> proto.DetectionRequest:
    """Serialize a np image array as ImageData protobuf."""
    # Determine np_index_order from image shape
    # Object detection expects YXC (2D) or ZYXC (3D)
    if image.ndim == 3:
        np_index_order = "YXC"
    elif image.ndim == 4:
        np_index_order = "ZYXC"
    else:
        raise ValueError(
            f"Object detection image must have 3 or 4 dimensions. Got {image.ndim}"
        )

    image_data = _encode_image(
        image,
        np_index_order=np_index_order,
        z_ratio=settings["Z Aspect Ratio"],
    )

    request = proto.DetectionRequest(
        image_data=image_data,
        detection_settings=_get_detection_settings(settings),
    )

    return request


def _get_grpc_channel(settings: dict):
    """Create gRPC channel based on server URL.

    Scheme is determined from URL format:
        - "http://hostname:port" → insecure channel
        - "https://hostname:port" → secure channel
        - "hostname:port" → auto (HTTPS for port 443, HTTP otherwise)
        - "hostname" → auto (defaults to port 443, uses HTTPS)

    Args:
        settings: Widget settings dict with 'Server' key

    Returns:
        gRPC channel (secure or insecure based on URL scheme)

    Raises:
        ValueError: If server URL is invalid
    """
    server_url = settings["Server"]

    try:
        host, port, explicit_scheme, _ = _parse_server_url(server_url)
    except ValueError:
        raise ValueError(
            f"Invalid server URL: '{server_url}'. "
            "Use format 'hostname:port' (e.g., 'localhost:50051') or include scheme like 'http://localhost:50051'"
        )

    config = load_config()
    max_msg_size = config["grpc"]["max_message_size_mb"]
    max_msg_bytes = 1024 * 1024 * max_msg_size

    # Determine scheme: explicit from URL, or auto-detect based on port
    if explicit_scheme:
        scheme = explicit_scheme.upper()
    else:
        scheme = "HTTPS" if port == 443 else "HTTP"

    # Build target address (host:port format for gRPC)
    target = f"{host}:{port}"

    if scheme == "HTTPS":
        return grpc.secure_channel(
            target=target,
            credentials=grpc.ssl_channel_credentials(),
            options=[("grpc.max_receive_message_length", max_msg_bytes)],
        )
    else:
        return grpc.insecure_channel(
            target=target,
            options=[("grpc.max_receive_message_length", max_msg_bytes)],
        )


def check_server_health(
    settings: dict, timeout: Optional[float] = None
) -> bool:
    """Check if the gRPC server is healthy and ready to serve requests.

    Args:
        settings: Widget settings dict (must contain 'Server' and 'Scheme')
        timeout: Timeout in seconds for the health check. If None, uses config value.

    Returns:
        True if server is healthy, False otherwise
    """
    if timeout is None:
        config = load_config()
        timeout = config["timeout"]["health_check"]

    try:
        with _get_grpc_channel(settings) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest()
            response = stub.Check(request, timeout=timeout)
            return response.status == health_pb2.HealthCheckResponse.SERVING
    except Exception as e:
        logger.debug("Health check failed: %s", e, exc_info=True)
        return False


def get_op_names(
    settings: dict, timeout: Optional[float] = None
) -> proto.OpNames:
    """Query server for available operations and their schemas.

    Args:
        settings: Widget settings dict (must contain 'Server' and 'Scheme')
        timeout: Timeout in seconds for the request. If None, uses config value.

    Returns:
        OpNames proto containing operation names and their schemas

    Raises:
        Exception: If connection or query fails
    """
    if timeout is None:
        config = load_config()
        timeout = config["timeout"]["get_op_names"]

    with _get_grpc_channel(settings) as channel:
        stub = proto.ProcessImageStub(channel)
        response = stub.GetOpNames(empty_pb2.Empty(), timeout=timeout)
        logger.debug("Received %d ops from server", len(response.names))

        return response


def _get_detection_settings(settings: dict):
    """Convert widget settings to DetectionSettings protobuf."""
    nms_values = {
        "Off": 0.0,
        "Iou-0.2": 0.2,
        "Iou-0.4": 0.4,
        "Iou-0.6": 0.6,
        "Iou-0.8": 0.8,
    }
    nms_iou = nms_values[settings["NMS"]]

    return proto.DetectionSettings(
        min_score=settings["Min Score"],
        nms_iou=nms_iou,
        cell_diameter_hint=settings["Size Hint"],
    )


@thread_worker
def grpc_object_detection(
    image_data: napari_data,
    settings: dict,
    grid_positions: list,
    abort_event: Optional[threading.Event] = None,
    future_container: Optional[dict] = None,
) -> Generator[np.ndarray, None, None]:
    """Run object detection on image data via gRPC.

    Args:
        image_data: Input image(s) as dask array or numpy array
        settings: Widget settings dict
        grid_positions: List of slice tuples for patch positions
        abort_event: Optional threading.Event to signal cancellation
        future_container: Optional dict to store active gRPC future for direct cancellation

    Yields:
        None for progress updates, then label array for each image

    Raises:
        ValueError: If image dimensions don't match expected format
    """
    is3d = settings["3D"]
    expected_ndim = 5 if is3d else 4
    if image_data.ndim != expected_ndim:
        raise ValueError(
            f"For {'3D' if is3d else '2D'} mode, image data must have {expected_ndim} dimensions "
            f"(batch, {'z,' if is3d else ''}y, x, channel). Got shape {image_data.shape} with {image_data.ndim} dimensions"
        )

    # Get timeout from config
    config = load_config()
    timeout = config["timeout"]["detection_3d" if is3d else "detection_2d"]

    server = settings["Server"]
    logger.info("Starting object detection on %s", server)

    # call server
    with _get_grpc_channel(settings) as channel:
        stub = proto.ObjectDetectionStub(channel)

        for image in image_data:
            # start with an empty response
            response = proto.DetectionResponse()

            for grid in grid_positions:
                # Check for abort before processing each patch
                if abort_event is not None and abort_event.is_set():
                    logger.info("Object detection aborted by user")
                    return

                logger.debug("Processing patch %s", grid)

                patch = np.array(image.__getitem__(grid))

                request = _object_detection_build_request(patch, settings)

                # Signal call start for progress bar
                yield CALL_START

                future = stub.RunDetection.future(request)

                # Store future in container for direct cancellation from UI thread
                if future_container is not None:
                    future_container["active"] = future

                # Poll for abort while waiting for response
                while not future.done():
                    if abort_event is not None and abort_event.is_set():
                        future.cancel()
                        if future_container is not None:
                            future_container["active"] = None
                        logger.info("gRPC RunDetection call cancelled")
                        return
                    time.sleep(0.05)

                # Clear future reference after call completes
                if future_container is not None:
                    future_container["active"] = None

                try:
                    patch_response = future.result(timeout=timeout)
                except grpc.FutureCancelledError:
                    logger.info("gRPC RunDetection call was cancelled")
                    return
                patch_response = _adjust_response_offset(patch_response, grid)

                logger.debug(
                    "Detected %d cells in patch",
                    len(patch_response.detections),
                )

                response.MergeFrom(patch_response)

                yield

            logger.info("Detected %d cells in image", len(response.detections))

            yield _generate_label(
                response, np.zeros(image_data.shape[1:-1], dtype="uint16")
            )


def _process_single_chunk(
    chunk: np.ndarray,
    position: dict,
    stub: proto.ProcessImageStub,
    iter_spec: IterationSpec,
    op_name: str,
    kwargs: dict,
    timeout: float,
) -> Tuple[Optional[np.ndarray], dict, pd.DataFrame]:
    """Process a single chunk via gRPC (for ThreadPoolExecutor).

    Args:
        chunk: Image chunk array
        position: Dict mapping numeric dim indices to index values
        stub: gRPC stub (thread-safe)
        iter_spec: IterationSpec with axis_order
        op_name: Operation name
        kwargs: Operation kwargs
        timeout: Request timeout in seconds

    Returns:
        Tuple of (output, position, annotation_data)
    """
    # Convert axis order to dim_labels list
    dim_labels = list(iter_spec.axis_order)
    image_data = serialize_from_numpy_to_image_data(
        chunk, dim_labels=dim_labels
    )

    request = proto.ProcessRequest(
        image_data=image_data,
        op_name=op_name,
        kwargs=dict_to_struct(kwargs),
    )

    future = stub.Run.future(request)

    # Wait for completion (no abort polling here - handled at higher level)
    response = future.result(timeout=timeout)

    # Parse annotation
    annotation_data = _parse_annotation_tsv(response.annotation)

    # Check if response has image data
    has_image = response.image_data is not None

    if has_image:
        output = deserialize_image_data(response.image_data)
        # Ensure numpy array (dask arrays are computed)
        if hasattr(output, "compute"):
            output = output.compute()
        output = output.astype(output.dtype.type)

        # Validate: all iter dims must be singleton in output
        for axis_name in iter_spec.iter_dims:
            full_idx = FULL_ORDER.index(axis_name)
            if output.shape[full_idx] != 1:
                raise ValueError(
                    f"Operation changed iterated dimension '{axis_name}' "
                    f"from 1 to {output.shape[full_idx]}. "
                    f"Cannot tile results when operation modifies iterated dimensions."
                )

        logger.debug(
            "Processed chunk at position %s, output shape: %s",
            position,
            output.shape,
        )
    else:
        output = None
        logger.debug("Response at position %s has no image data", position)

    return output, position, annotation_data


@thread_worker
def grpc_process_image(
    image_data: napari_data,
    settings: dict,
    iter_spec: IterationSpec,
    abort_event: Optional[threading.Event] = None,
    future_container: Optional[dict] = None,
) -> Generator[Tuple[Optional[np.ndarray], dict, dict], None, None]:
    """Run image processing via gRPC with dimensional iteration.

    Args:
        image_data: Input image(s) as dask array or numpy array (raw, not reshaped)
        settings: Widget settings dict (includes 'Op' and kwargs)
        iter_spec: IterationSpec with iter_dims (set of axis names) and axis_order
        abort_event: Optional threading.Event to signal cancellation
        future_container: Optional dict to store active gRPC future for direct cancellation

    Yields:
        Tuple of (result_chunk, position, annotation) for each iteration
        - result_chunk: Processed image array (5D TZCYX) or None if empty
        - position: Dict mapping numeric dim indices to index values
        - annotation: Dict of column->values for table display

    Raises:
        ValueError: If operation changes iterated dimensions from singleton
    """
    # Get timeout and concurrency from config
    config = load_config()
    timeout = config["timeout"][
        "detection_3d" if "Z" in iter_spec.axis_order else "detection_2d"
    ]
    max_concurrent = config["grpc"].get("max_concurrent_calls", 4)

    server = settings["Server"]
    logger.info(
        "Starting image processing on %s with max %d concurrent calls",
        server,
        max_concurrent,
    )

    # Extract op_name and kwargs from settings
    op_name = settings.get("Op", "")
    kwargs = _extract_kwargs(settings)

    # call server
    with _get_grpc_channel(settings) as channel:
        stub = proto.ProcessImageStub(channel)

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}

            # Submit all chunks for processing
            for position, chunk in _data_iterator(
                image_data, iter_spec.iter_dims, iter_spec.axis_order
            ):
                # Check for abort before submitting
                if abort_event is not None and abort_event.is_set():
                    logger.info("Image processing aborted during submission")
                    # Cancel any already-submitted futures
                    for f in futures:
                        f.cancel()
                    return

                chunk = np.array(chunk)

                # Submit to executor
                future = executor.submit(
                    _process_single_chunk,
                    chunk,
                    position,
                    stub,
                    iter_spec,
                    op_name,
                    kwargs,
                    timeout,
                )
                futures[future] = position

                # Signal call start for progress bar (one per chunk)
                yield CALL_START

            # Store all futures in container for cancellation from UI thread
            if future_container is not None:
                future_container["active"] = set(futures.keys())

            # Collect results as they complete
            for future in as_completed(futures):
                # Check for abort during collection
                if abort_event is not None and abort_event.is_set():
                    logger.info("Image processing aborted during collection")
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    if future_container is not None:
                        future_container["active"] = None
                    return

                # Remove completed future from active set
                if future_container is not None:
                    active_set = future_container.get("active", set())
                    active_set.discard(future)
                    future_container["active"] = active_set

                try:
                    output, position, annotation_data = future.result()
                except grpc.FutureCancelledError:
                    logger.info("gRPC Run call was cancelled")
                    return
                except Exception as e:
                    logger.error(
                        "Error processing chunk at position %s: %s",
                        futures[future],
                        e,
                    )
                    # Cancel remaining futures on error
                    for f in futures:
                        f.cancel()
                    if future_container is not None:
                        future_container["active"] = None
                    raise

                yield output, position, annotation_data

            # Clear future reference after all complete
            if future_container is not None:
                future_container["active"] = None


def _extract_kwargs(settings: dict) -> dict:
    """Extract kwargs from widget settings.

    Settings keys matching known op kwarg names are extracted.
    Non-op-related keys (Image, 3D, Server, Status, Op) are excluded.
    """
    exclude_keys = {"Image", "3D", "Server", "Status", "Op"}
    kwargs = {}
    for key, value in settings.items():
        if key not in exclude_keys:
            kwargs[key] = value
    return kwargs
