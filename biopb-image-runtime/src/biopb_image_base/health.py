"""Health check service for biopb services.

Implements the standard gRPC health checking protocol.
"""

import logging
import threading
from typing import Callable, Optional

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

logger = logging.getLogger(__name__)


class HealthServicer(health_pb2_grpc.HealthServicer):
    """Standard gRPC health check servicer.

    Implements the grpc.health.v1.Health service for standard
    health checking protocol compatible with Kubernetes, etc.
    """

    def __init__(self, readiness_check: Optional[Callable[[], bool]] = None):
        """Initialize health servicer.

        Args:
            readiness_check: Optional callable that returns True if service is ready.
                           Called during health checks to determine serving status.
        """
        self._status = health_pb2.HealthCheckResponse.SERVING
        self._readiness_check = readiness_check
        self._lock = threading.Lock()
        self._watchers: dict = {}
        logger.info("Health check servicer initialized")

    def Check(
        self, request: health_pb2.HealthCheckRequest, context: grpc.ServicerContext
    ) -> health_pb2.HealthCheckResponse:
        """Check health status of the service.

        Args:
            request: Health check request (service name optional)
            context: gRPC context

        Returns:
            HealthCheckResponse with current status
        """
        with self._lock:
            # If there's a readiness check, use it
            if self._readiness_check is not None:
                try:
                    if self._readiness_check():
                        status = health_pb2.HealthCheckResponse.SERVING
                    else:
                        status = health_pb2.HealthCheckResponse.NOT_SERVING
                except Exception as e:
                    logger.error(f"Readiness check failed: {e}")
                    status = health_pb2.HealthCheckResponse.UNKNOWN
            else:
                status = self._status

        return health_pb2.HealthCheckResponse(status=status)

    def Watch(
        self, request: health_pb2.HealthCheckRequest, context: grpc.ServicerContext
    ):
        """Watch for health status changes (streaming).

        Note: Basic implementation that sends current status once.
        """
        # Simple implementation - send current status
        yield self.Check(request, context)

    def set_serving(self, serving: bool = True):
        """Set the serving status manually.

        Args:
            serving: True if serving, False if not serving
        """
        with self._lock:
            self._status = (
                health_pb2.HealthCheckResponse.SERVING
                if serving
                else health_pb2.HealthCheckResponse.NOT_SERVING
            )
        logger.info(f"Health status set to {'SERVING' if serving else 'NOT_SERVING'}")

    def set_not_serving(self):
        """Mark service as not serving."""
        self.set_serving(False)


def add_health_servicer(
    server: grpc.Server, readiness_check: Optional[Callable[[], bool]] = None
) -> HealthServicer:
    """Add health check servicer to a gRPC server.

    Args:
        server: gRPC server instance
        readiness_check: Optional callable for readiness checks

    Returns:
        The HealthServicer instance for status updates
    """
    servicer = HealthServicer(readiness_check)
    health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
    logger.info("Health check service registered")
    return servicer
