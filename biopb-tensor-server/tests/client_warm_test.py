"""Unit tests for the SDK ``TensorFlightClient.warm()`` (hydrate-ahead).

Exercise the client's parse of the streaming ``warm`` do_action without a live
server: ``do_action`` is stubbed to replay a fixed stream of ``WarmStreamMessage``
bodies, and the client is built with ``object.__new__`` so no connection opens.
The end-to-end server-side recall is covered by the server suite
(``cloud_phase2_test.py::TestWarmAction``).
"""

import pytest
import pyarrow.flight as flight

from biopb.tensor.client import ResolveCancelled, TensorFlightClient
from biopb.tensor.descriptor_pb2 import WarmProgress, WarmStreamMessage


def _progress_body(files_done, files_total=3, bytes_done=0, name="chunk"):
    return WarmStreamMessage(
        progress=WarmProgress(
            files_total=files_total,
            files_done=files_done,
            bytes_done=bytes_done,
            current_name=name,
        )
    ).SerializeToString()


def _done_body(files_total=3, files_done=3, bytes_total=300, bytes_done=300):
    return WarmStreamMessage(
        done=WarmProgress(
            files_total=files_total,
            files_done=files_done,
            bytes_total=bytes_total,
            bytes_done=bytes_done,
        )
    ).SerializeToString()


def _bare_client():
    client = object.__new__(TensorFlightClient)
    client._sources = {}
    client._descriptors = {}
    client._call_options = None
    return client


class _Buf:
    def __init__(self, body):
        self._body = body

    def to_pybytes(self):
        return self._body


class _FakeResult:
    def __init__(self, body):
        self.body = _Buf(body)


class _FakeFlight:
    """Replays a fixed stream of Results from do_action, or raises on iteration."""

    def __init__(self, results=None, raise_exc=None):
        self._results = results or []
        self._raise = raise_exc
        self.action = None

    def do_action(self, action, options=None):
        self.action = action
        if self._raise is not None:
            raise self._raise
        return iter(self._results)


class TestWarm:
    def test_progress_reported_then_terminal_done_returned(self):
        client = _bare_client()
        client._client = _FakeFlight(
            [
                _FakeResult(_progress_body(0, name="a")),
                _FakeResult(_progress_body(1, bytes_done=100, name="b")),
                _FakeResult(_done_body()),
            ]
        )
        seen = []

        out = client.warm("cloud_x", on_progress=seen.append)

        assert client._client.action.type == "warm"
        assert bytes(client._client.action.body) == b"cloud_x"
        assert out.files_done == 3 and out.bytes_done == 300  # terminal `done`
        assert [p.files_done for p in seen] == [0, 1]  # progress arms only
        assert seen[1].current_name == "b"

    def test_should_cancel_raises_resolve_cancelled(self):
        client = _bare_client()
        client._client = _FakeFlight(
            [_FakeResult(_progress_body(0)), _FakeResult(_done_body())]
        )
        with pytest.raises(ResolveCancelled):
            client.warm("cloud_x", should_cancel=lambda: True)

    def test_empty_bodies_skipped(self):
        client = _bare_client()
        client._client = _FakeFlight(
            [_FakeResult(b""), _FakeResult(_done_body(files_total=0, files_done=0))]
        )
        out = client.warm("cloud_x")
        assert out.files_total == 0  # no-op source terminal

    def test_old_server_unknown_action_maps_to_clear_error(self):
        client = _bare_client()
        client._client = _FakeFlight(
            raise_exc=flight.FlightServerError("Unknown action: warm")
        )
        with pytest.raises(RuntimeError, match="too old"):
            client.warm("cloud_x")

    def test_other_flight_error_propagates(self):
        client = _bare_client()
        client._client = _FakeFlight(
            raise_exc=flight.FlightServerError("boom something else")
        )
        with pytest.raises(flight.FlightServerError, match="boom"):
            client.warm("cloud_x")

    def test_no_terminal_done_raises(self):
        client = _bare_client()
        client._client = _FakeFlight(
            [_FakeResult(_progress_body(0)), _FakeResult(_progress_body(1))]
        )
        with pytest.raises(RuntimeError, match="no terminal status"):
            client.warm("cloud_x")
