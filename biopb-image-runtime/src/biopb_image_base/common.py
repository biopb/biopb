"""Common pieces of a biopb gRPC server: message limits, the token interceptor,
and the translation of a handler's exceptions into gRPC status codes."""

from __future__ import annotations

import logging
import sys
import traceback
import uuid
from contextlib import contextmanager
from typing import Optional

import grpc

_AUTH_HEADER_KEY = "authorization"
_MAX_MSG_SIZE = 1024 * 1024 * 128  # 128MB
_MAX_EAGER_SIZE = 1024 * 1024 * 64  # 64MB - threshold for returning lazy data

logger = logging.getLogger(__name__)


def _is_dask_array(arr) -> bool:
    """True if *arr* is a dask array, without importing dask to find out.

    dask comes with the ``[lazy]`` extra. A dask array cannot exist in a process
    that never imported ``dask.array``, so its absence is a definitive "no".
    """
    da_mod = sys.modules.get("dask.array")
    return da_mod is not None and isinstance(arr, da_mod.Array)


def _pyarrow_available() -> bool:
    """True if pyarrow can be imported.

    By-reference (``array_id``) input and the embedded tensor cache are built on
    Arrow Flight and need pyarrow. On builds for old CPUs without SSE4.2/AVX,
    pyarrow is removed (its wheels SIGILL on import there -- see
    cellpose/BUILD_NO_SSE42.md in biopb-server). find_spec only locates the
    module; it does not import it, so this is safe even on a CPU that cannot run
    pyarrow.
    """
    import importlib.util

    return importlib.util.find_spec("pyarrow") is not None


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
# Error translation
# =============================================================================


@contextmanager
def server_context(context: grpc.ServicerContext, lock):
    """Run a handler's body under *lock*, translating its exceptions.

    ``ValueError`` becomes ``INVALID_ARGUMENT``, ``NotImplementedError``
    ``UNIMPLEMENTED``, and anything else ``INTERNAL``; each is logged with its
    traceback under an error id the client sees. A status the handler set with
    ``context.abort`` passes through unchanged. For a streaming handler, wrap
    the ``yield from`` so the lock and the translation span the whole stream.
    """
    try:
        with lock:
            yield

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
        # context.abort() raises a bare Exception after gRPC has recorded the
        # status; re-raise it so an explicit abort is not downgraded to INTERNAL.
        if context.code() is not None:
            raise

        error_id = uuid.uuid4().hex[:8]
        logger.error(f"[{error_id}] Prediction failed: {e}")
        logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")

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
