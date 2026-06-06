"""Tests for the ``health`` action's STARTING -> SERVING readiness signal.

The Flight port binds (and gRPC starts serving) in ``TensorFlightServer.__init__``,
*before* the startup path scans/registers the data folder. Until that scan
finishes and ``mark_ready()`` is called, the ``health`` action must report
``STARTING`` so a connecting client can tell "booting" apart from "down" and wait
instead of timing out. Once ready it reports ``SERVING``.
"""

import json
import threading
import time

from pyarrow import flight

from biopb_tensor_server.server import TensorFlightServer


def _health(server) -> dict:
    """Invoke the in-process health handler and decode its JSON payload."""
    (raw,) = list(server.do_action(None, flight.Action("health", b"")))
    return json.loads(bytes(raw))


def test_health_reports_starting_until_marked_ready():
    """Status flips STARTING -> SERVING only after mark_ready()."""
    server = TensorFlightServer("grpc://localhost:0")

    assert server.is_ready is False
    assert _health(server)["status"] == "STARTING"

    server.mark_ready()

    assert server.is_ready is True
    assert _health(server)["status"] == "SERVING"


def test_health_payload_shape_unchanged():
    """The readiness change leaves the other health fields intact."""
    server = TensorFlightServer("grpc://localhost:0")
    server.mark_ready()

    payload = _health(server)
    assert payload["status"] == "SERVING"
    for key in (
        "source_count",
        "metadata_db_enabled",
        "writable",
        "uptime_seconds",
    ):
        assert key in payload


def test_health_starting_over_the_wire_before_ready():
    """A real client sees STARTING while the server is still 'scanning'.

    The port is reachable as soon as the server is constructed/served, even
    though mark_ready() has not been called yet -- this is the whole point of
    the signal.
    """
    server = TensorFlightServer("grpc://localhost:0")
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(0.8)
    try:
        client = flight.FlightClient(f"grpc://localhost:{server.port}")

        (raw,) = list(client.do_action(flight.Action("health", b"")))
        assert json.loads(raw.body.to_pybytes())["status"] == "STARTING"

        server.mark_ready()

        (raw,) = list(client.do_action(flight.Action("health", b"")))
        assert json.loads(raw.body.to_pybytes())["status"] == "SERVING"
    finally:
        server.shutdown()
