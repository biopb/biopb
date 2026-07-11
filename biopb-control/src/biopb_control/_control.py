"""The control's loopback control API — stdlib HTTP, no web framework.

A tiny :class:`http.server.ThreadingHTTPServer` is all Layer-2 core needs: two
JSON endpoints a client uses to learn whether the data plane is up and to ask
the control to bring it up. Kept on the stdlib so the lean control pulls in no
starlette/uvicorn/httpx yet — those arrive with the Layer-3 single-origin front,
which will replace this module with a real ASGI app on the same port.

Endpoints:

- ``GET  /health``            -> ``{"control": "ok", "data_plane": {...}}``
- ``POST /data_plane/ensure`` -> ensure the plane is up (bounded wait), then the
                                 snapshot; this is what ``biopb-mcp`` calls in
                                 place of shelling out ``biopb server start``.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ._supervisor import DataPlaneSupervisor

logger = logging.getLogger(__name__)

# Headroom (seconds) between the server's ensure wait and the client's HTTP
# timeout: the server must send its verdict BEFORE the client's urlopen times
# out, else the client treats a working-but-slow control plane as unreachable.
_RESPONSE_MARGIN = 5.0
_MIN_ENSURE_WAIT = 1.0


def _bounded_ensure_wait(ensure_timeout: float, client_timeout: float) -> float:
    """How long ``/data_plane/ensure`` should wait for the plane to come up.

    Bounded strictly below the client's HTTP timeout (by ``_RESPONSE_MARGIN``) so
    the server always answers first; also capped by the server's own configured
    ``ensure_timeout``, and floored at ``_MIN_ENSURE_WAIT``. A missing/invalid
    client hint (``<= 0``) falls back to the configured ``ensure_timeout`` (the
    client then relies on its own urlopen timeout being generous).
    """
    if client_timeout <= 0:
        return ensure_timeout
    return max(_MIN_ENSURE_WAIT, min(ensure_timeout, client_timeout - _RESPONSE_MARGIN))


class _ControlHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the supervisor the handler dispatches to."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, supervisor: DataPlaneSupervisor, ensure_timeout: float):
        super().__init__(addr, _ControlHandler)
        self.supervisor = supervisor
        self.ensure_timeout = ensure_timeout


class _ControlHandler(BaseHTTPRequestHandler):
    server_version = "biopb-control/control"

    # Route access logs through the module logger (debug) instead of stderr.
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        logger.debug("control-api %s - %s", self.address_string(), fmt % args)

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _supervisor(self) -> DataPlaneSupervisor:
        return self.server.supervisor  # type: ignore[attr-defined]

    def do_GET(self):  # noqa: N802 - stdlib hook name
        if self.path in ("/health", "/healthz"):
            self._write_json(
                200, {"control": "ok", "data_plane": self._supervisor.snapshot()}
            )
        elif self.path == "/":
            self._write_json(200, {"control": "ok", "service": "biopb control plane"})
        else:
            self._write_json(404, {"error": "not found", "path": self.path})

    def do_POST(self):  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path == "/data_plane/ensure":
            self._drain_body()
            sup = self._supervisor
            # The client passes ?client_timeout=<its HTTP timeout>; cap our wait
            # below it so we return a verdict before the client gives up (and
            # wrongly treats a slow-but-working control plane as unreachable).
            try:
                client_timeout = float(
                    parse_qs(parsed.query).get("client_timeout", ["0"])[0]
                )
            except ValueError:
                client_timeout = 0.0
            wait = _bounded_ensure_wait(
                self.server.ensure_timeout,  # type: ignore[attr-defined]
                client_timeout,
            )
            # ensure()/_spawn_locked count a spawn failure toward the backoff and
            # do not raise, but wrap defensively so any unexpected error still
            # returns a clean JSON verdict (with the snapshot that reflects the
            # counted failure) rather than an unhandled 500.
            try:
                sup.ensure()
                sup.wait_until_up(wait)
                self._write_json(200, {"data_plane": sup.snapshot()})
            except Exception as exc:  # noqa: BLE001 - report, never crash the handler
                logger.exception("data_plane/ensure failed")
                self._write_json(500, {"error": str(exc), "data_plane": sup.snapshot()})
        else:
            self._write_json(404, {"error": "not found", "path": self.path})

    def _drain_body(self) -> None:
        # Consume any request body so the connection can be reused / closed
        # cleanly, even though these endpoints take no parameters yet.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)


def serve_control_api(
    host: str,
    port: int,
    supervisor: DataPlaneSupervisor,
    ensure_timeout: float,
) -> tuple[_ControlHTTPServer, threading.Thread]:
    """Start the control API on ``host:port`` in a background thread.

    Returns the server and its thread; the caller shuts it down with
    ``server.shutdown()`` on teardown. Binds eagerly so a port clash surfaces
    to the caller (a control plane already running) rather than in a detached thread.
    """
    server = _ControlHTTPServer((host, port), supervisor, ensure_timeout)
    thread = threading.Thread(
        target=server.serve_forever, name="control-api", daemon=True
    )
    thread.start()
    logger.info("Control API listening on http://%s:%d", host, port)
    return server, thread
