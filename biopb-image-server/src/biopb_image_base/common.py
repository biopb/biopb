"""Common utilities for biopb gRPC services.

Provides:
- Image encoding/decoding (supporting both eager and lazy data)
- Token authentication interceptor
- Base servicer class with error handling
- Lazy data handling utilities for co-deployed tensor server
"""

import logging
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

import biopb.image as proto
import grpc
import numpy as np
import dask.array as da
from biopb.image.utils import deserialize_image_data, serialize_from_numpy_to_image_data
from google.protobuf.json_format import MessageToDict

if TYPE_CHECKING:
    from biopb_image_base.server import EmbeddedTensorCache

_AUTH_HEADER_KEY = "authorization"
_MAX_MSG_SIZE = 1024 * 1024 * 128  # 128MB
_MAX_EAGER_SIZE = 1024 * 1024 * 64  # 64MB - threshold for returning lazy data

logger = logging.getLogger(__name__)


# =============================================================================
# Request Logging Context
# =============================================================================


class RequestLogger:
    """Context manager for request logging with timing.

    Usage:
        with RequestLogger("RunDetection", request.ByteSize()) as log:
            # ... process request ...
            log.response_size = response.ByteSize()
    """

    def __init__(self, method: str, request_size: int = 0):
        self.method = method
        self.request_size = request_size
        self.response_size = 0
        self.start_time = time.perf_counter()
        # Use biopb_image_base hierarchy so logging_config applies
        self._logger = logging.getLogger(f"biopb_image_base.request.{method}")

    def __enter__(self):
        self._logger.debug(
            f"Request started: {self.method}, size={self.request_size} bytes"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            self._logger.error(
                f"Request failed: {self.method}, "
                f"latency={elapsed_ms:.2f}ms, error={exc_val}"
            )
        else:
            self._logger.info(
                f"Request completed: {self.method}, "
                f"latency={elapsed_ms:.2f}ms, response_size={self.response_size} bytes"
            )

        return False  # Don't suppress exceptions


# =============================================================================
# Image Utilities
# =============================================================================


def decode_image_data(image_data: proto.ImageData) -> Union[np.ndarray, da.Array]:
    """Decode protobuf ImageData to numpy or dask array.

    Handles both eager (inline) and lazy (SerializedTensor) data.
    Also supports legacy Pixels field for backward compatibility.

    Args:
        image_data: Protobuf ImageData message

    Returns:
        Numpy array (eager) or Dask array (lazy)

    Raises:
        ValueError: If image has unsupported dimensions
    """
    result = deserialize_image_data(image_data)

    if result.ndim < 2:
        raise ValueError("Image data must be at least 2D")
    if result.ndim >= 5:
        raise ValueError("Image data has more than 5 dimensions")

    return result


def encode_image(image: np.ndarray, dim_labels: Optional[list] = None) -> proto.ImageData:
    """Encode numpy array to protobuf ImageData (eager).

    Args:
        image: Numpy array
        dim_labels: Optional dimension labels (e.g., ["Y", "X", "C"])

    Returns:
        Protobuf ImageData message with eager_data
    """
    return serialize_from_numpy_to_image_data(image, dim_labels=dim_labels)


def return_lazy_or_eager(
    result: Union[np.ndarray, da.Array],
    tensor_cache: Optional["EmbeddedTensorCache"] = None,
    dim_labels: Optional[list] = None,
    max_eager_size: int = _MAX_EAGER_SIZE,
) -> proto.ImageData:
    """Return result as eager or lazy depending on size.

    For large results or dask arrays, uploads to tensor server and returns
    SerializedTensor reference. For small numpy arrays, returns inline data.

    Args:
        result: Numpy array or Dask array to return
        tensor_cache: EmbeddedTensorCache for lazy uploads (required for lazy returns)
        dim_labels: Optional dimension labels
        max_eager_size: Threshold in bytes for returning lazy data

    Returns:
        ImageData with either eager_data (inline) or lazy_data (SerializedTensor)

    Raises:
        ValueError: If tensor_cache not provided for lazy result
    """
    is_lazy = isinstance(result, da.Array)
    nbytes = result.nbytes if hasattr(result, 'nbytes') else 0

    if is_lazy or nbytes > max_eager_size:
        if tensor_cache is None:
            raise ValueError(
                "tensor_cache required for returning lazy data. "
                "Enable embedded cache with --cache-dir option."
            )

        logger.info(f"Uploading lazy result ({nbytes} bytes) to tensor server")

        # Upload to embedded tensor cache
        source_id = tensor_cache.create_source(
            result,
            source_name="cache:",
            dim_labels=dim_labels,
        )

        # Get SerializedTensor reference with external location
        serialized = tensor_cache.to_serialized_tensor(source_id)
        return proto.ImageData(lazy_data=serialized)
    else:
        return serialize_from_numpy_to_image_data(result, dim_labels=dim_labels)


def parse_kwargs(request, defaults: dict) -> dict:
    """Merge request kwargs with defaults.

    Args:
        request: DetectionRequest or ProcessRequest with optional kwargs field
        defaults: Dictionary of default parameter values

    Returns:
        Dictionary with defaults overridden by any kwargs from the request
    """
    kwargs = defaults.copy()
    if request.HasField("kwargs"):
        # Struct.fields is a map<string, Value>, so we iterate through it directly
        # This converts protobuf Value types to Python native types
        for key, value in request.kwargs.fields.items():
            # Convert Value to Python native type
            if value.HasField("number_value"):
                kwargs[key] = value.number_value
            elif value.HasField("string_value"):
                kwargs[key] = value.string_value
            elif value.HasField("bool_value"):
                kwargs[key] = value.bool_value
            elif value.HasField("list_value"):
                # Convert ListValue to Python list
                kwargs[key] = [
                    item.number_value if item.HasField("number_value")
                    else item.string_value if item.HasField("string_value")
                    else item.bool_value if item.HasField("bool_value")
                    else None
                    for item in value.list_value.values
                ]
            elif value.HasField("struct_value"):
                # Convert nested Struct to Python dict
                kwargs[key] = {
                    k: v.number_value if v.HasField("number_value")
                    else v.string_value if v.HasField("string_value")
                    else v.bool_value if v.HasField("bool_value")
                    else None
                    for k, v in value.struct_value.fields.items()
                }
    return kwargs


def ensure_eager(image: Union[np.ndarray, da.Array]) -> np.ndarray:
    """Ensure image is a numpy array (eager), not a dask array (lazy).

    Args:
        image: Numpy array or Dask array

    Returns:
        Numpy array

    Raises:
        ValueError: If image is a lazy dask array
    """
    if isinstance(image, da.Array):
        raise ValueError(
            "Lazy data (dask array) not supported. "
            "Services currently only process eager data. "
            f"Received lazy array with shape {image.shape}, dtype {image.dtype}."
        )
    return image


def validate_kwargs(kwargs: dict, schema: dict) -> list[str]:
    """Validate kwargs against a schema.

    Args:
        kwargs: Dictionary of parameter values to validate
        schema: Dictionary defining validation rules for each parameter.
            Each key is a parameter name, value is a dict with:
            - "type": expected type ("array", "number", "int", "bool", "string")
            - "item_type": for arrays, expected type of items
            - "min_length", "max_length": for arrays, length constraints
            - "minimum", "maximum": for numeric types, value constraints
            - "description": optional description for error messages

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for key, rules in schema.items():
        if key not in kwargs:
            continue

        value = kwargs[key]
        expected_type = rules.get("type")

        if expected_type == "array":
            if not isinstance(value, list):
                errors.append(f"{key}: expected array, got {type(value).__name__}")
                continue

            item_type = rules.get("item_type")
            if item_type:
                for i, item in enumerate(value):
                    if item_type == "int":
                        # Accept int or float that is an integer value
                        if isinstance(item, float) and item == int(item):
                            # Valid - float representing an integer (e.g., from protobuf)
                            pass
                        elif not isinstance(item, int):
                            errors.append(f"{key}[{i}]: expected int, got {type(item).__name__}")
                    elif item_type == "number" and not isinstance(item, (int, float)):
                        errors.append(f"{key}[{i}]: expected number, got {type(item).__name__}")

            min_len = rules.get("min_length")
            max_len = rules.get("max_length")
            if min_len and len(value) < min_len:
                errors.append(f"{key}: minimum length {min_len}, got {len(value)}")
            if max_len and len(value) > max_len:
                errors.append(f"{key}: maximum length {max_len}, got {len(value)}")

        elif expected_type == "number":
            if not isinstance(value, (int, float)):
                errors.append(f"{key}: expected number, got {type(value).__name__}")
            else:
                minimum = rules.get("minimum")
                maximum = rules.get("maximum")
                if minimum is not None and value < minimum:
                    errors.append(f"{key}: minimum {minimum}, got {value}")
                if maximum is not None and value > maximum:
                    errors.append(f"{key}: maximum {maximum}, got {value}")

        elif expected_type == "int":
            # Accept int or float that is an integer value
            if isinstance(value, int):
                pass  # valid
            elif isinstance(value, float) and value == int(value):
                value = int(value)  # Convert to int for range checks
            else:
                errors.append(f"{key}: expected int, got {type(value).__name__}")
                continue  # Skip range checks if type is wrong

            minimum = rules.get("minimum")
            maximum = rules.get("maximum")
            if minimum is not None and value < minimum:
                errors.append(f"{key}: minimum {minimum}, got {value}")
            if maximum is not None and value > maximum:
                errors.append(f"{key}: maximum {maximum}, got {value}")

        elif expected_type == "bool":
            if not isinstance(value, bool):
                errors.append(f"{key}: expected bool, got {type(value).__name__}")

        elif expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"{key}: expected string, got {type(value).__name__}")

    return errors


def abort_invalid_argument(context: grpc.ServicerContext, message: str) -> None:
    """Abort the current RPC with INVALID_ARGUMENT.

    Centralizes argument validation failures so servicers do not need to
    reference gRPC status codes directly.
    """
    context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)


# =============================================================================
# Authentication
# =============================================================================


class TokenValidationInterceptor(grpc.ServerInterceptor):
    """gRPC interceptor for Bearer token authentication."""

    def __init__(self, token: Optional[str]):
        def abort(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)
        self.token = token

    def intercept_service(self, continuation, handler_call_details):
        # Allow health checks without authentication
        method = handler_call_details.method
        if method and "grpc.health.v1.Health" in method:
            return continuation(handler_call_details)

        expected_metadata = (_AUTH_HEADER_KEY, f"Bearer {self.token}")
        if (
            self.token is None
            or expected_metadata in handler_call_details.invocation_metadata
        ):
            return continuation(handler_call_details)
        else:
            return self._abort_handler


# =============================================================================
# Base Servicer
# =============================================================================


class BiopbServicerBase(proto.ObjectDetectionServicer, proto.ProcessImageServicer):
    """Base class for biopb servicers with error handling and logging.

    Provides:
    - Optional thread-safe request handling via lock
    - Error handling with proper gRPC status codes and correlation IDs
    - Full traceback logging for all errors

    Subclasses should implement RunDetection and Run methods.

    Args:
        use_lock: If True, serialize requests with a lock. Default True for
            backwards compatibility. Set False for thread-safe models.
        tensor_cache: Optional EmbeddedTensorCache for lazy result handling.
    """

    def __init__(
        self,
        use_lock: bool = True,
        tensor_cache: Optional["EmbeddedTensorCache"] = None,
    ):
        self._lock = threading.RLock() if use_lock else None
        self._use_lock = use_lock
        self._tensor_cache = tensor_cache

    @contextmanager
    def _server_context(self, context):
        """Context manager for request handling with error handling.

        Usage:
            def RunDetection(self, request, context):
                with self._server_context(context):
                    # ... process request ...
                    return response
        """
        try:
            if self._use_lock:
                with self._lock:
                    yield
            else:
                yield

        # Let gRPC abort exceptions propagate (avoid double-abort)
        except grpc.RpcError:
            raise

        except ValueError as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"[{error_id}] Invalid argument: {e}")
            logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{repr(e)} (error_id: {error_id})",
            )

        except NotImplementedError as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"[{error_id}] Not implemented: {e}")
            logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"{repr(e)} (error_id: {error_id})",
            )

        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"[{error_id}] Prediction failed: {e}")
            logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")

            # Check for CUDA errors and log helpful message
            error_str = str(e).lower()
            if "cuda" in error_str or "gpu" in error_str:
                if "out of memory" in error_str:
                    logger.warning(
                        f"[{error_id}] CUDA out of memory error. Consider: "
                        "1) reducing image size, 2) clearing GPU cache, "
                        "3) using smaller batch sizes"
                    )
                elif "device" in error_str or "illegal" in error_str:
                    logger.warning(
                        f"[{error_id}] CUDA device error detected. GPU state may be corrupted. "
                        "Service restart may be required."
                    )

            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Prediction failed with error: {repr(e)} (error_id: {error_id})",
            )

    def RunDetectionStream(self, request_iterator, context):
        """Handle streaming detection requests.

        Accumulates request data from the stream and calls RunDetection.
        """
        request = proto.DetectionRequest()

        for next_request in request_iterator:
            if next_request.image_data is not None:
                if next_request.image_data.HasField("pixels"):
                    request.image_data.pixels.CopyFrom(next_request.image_data.pixels)
                if next_request.image_data.HasField("eager_data"):
                    request.image_data.eager_data.CopyFrom(next_request.image_data.eager_data)
                if next_request.image_data.HasField("lazy_data"):
                    request.image_data.lazy_data.CopyFrom(next_request.image_data.lazy_data)
                if next_request.image_data.HasField("image_annotation"):
                    request.image_data.image_annotation.CopyFrom(
                        next_request.image_data.image_annotation
                    )

            if next_request.HasField("detection_settings"):
                request.detection_settings.CopyFrom(next_request.detection_settings)

            if request.image_data is not None and (
                request.image_data.HasField("pixels") or
                request.image_data.HasField("eager_data") or
                request.image_data.HasField("lazy_data")
            ):
                yield self.RunDetection(request, context)