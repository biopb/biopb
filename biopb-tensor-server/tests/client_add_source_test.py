"""Unit tests for the SDK ``TensorFlightClient.add_source()`` streaming parse.

Exercise the client's parse of the streaming ``add_source`` do_action without a
live server: ``do_action`` is stubbed to replay a fixed stream of
``AddSourceStreamMessage`` bodies, and the client is built with ``object.__new__``
so no connection opens. The end-to-end server round-trip is covered by
``add_source_test.py::TestAddSourceRoundtrip``; this file pins the envelope
mapping, the old-server remap, and the cancel semantics (issue #4).
"""

import pyarrow.flight as flight
import pytest
from biopb.tensor.client import TensorFlightClient
from biopb.tensor.descriptor_pb2 import (
    AddSourceProgress,
    AddSourceResult,
    AddSourceStreamMessage,
    DataSourceDescriptor,
)


def _progress_body(count, path=""):
    return AddSourceStreamMessage(
        progress=AddSourceProgress(added_count=count, current_path=path)
    ).SerializeToString()


def _result_body(added=(), already=(), failed=()):
    r = AddSourceResult(already_present=list(already))
    r.added.extend(added)
    for path, reason in failed:
        r.failed.add(path=path, reason=reason)
    return AddSourceStreamMessage(result=r).SerializeToString()


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
    """Replays a fixed stream of Results from do_action, or raises on the call."""

    def __init__(self, results=None, raise_exc=None):
        self._results = results or []
        self._raise = raise_exc
        self.action = None

    def do_action(self, action, options=None):
        self.action = action
        if self._raise is not None:
            raise self._raise
        return iter(self._results)


def _flip_after(n):
    """A should_cancel that returns False for the first *n* polls, then True.

    The poll now runs *after* each message is consumed, so ``_flip_after(k)``
    models a cancel that fires just as the (k+1)-th message is taken.
    """
    calls = {"i": 0}

    def should_cancel():
        calls["i"] += 1
        return calls["i"] > n

    return should_cancel


class TestAddSource:
    def test_terminal_result_returned(self):
        client = _bare_client()
        added = DataSourceDescriptor(source_id="s1")
        client._client = _FakeFlight(
            [_FakeResult(_result_body(added=[added], already=["s0"]))]
        )

        out = client.add_source("/drop")

        assert client._client.action.type == "add_source"
        assert [d.source_id for d in out.added] == ["s1"]
        assert list(out.already_present) == ["s0"]

    def test_progress_envelopes_reported_then_terminal_taken(self):
        client = _bare_client()
        seen = []
        client._client = _FakeFlight(
            [
                _FakeResult(_progress_body(1, "/d/a.zarr")),
                _FakeResult(_progress_body(2, "/d/b.zarr")),
                _FakeResult(_result_body(added=[])),
            ]
        )

        client.add_source("/drop", on_progress=seen.append)

        assert [p.added_count for p in seen] == [1, 2]
        assert seen[0].current_path == "/d/a.zarr"

    def test_unknown_action_maps_to_runtimeerror(self):
        client = _bare_client()
        client._client = _FakeFlight(
            raise_exc=flight.FlightServerError("Unknown action 'add_source'")
        )
        with pytest.raises(RuntimeError, match="too old"):
            client.add_source("/drop")

    def test_no_terminal_result_raises(self):
        client = _bare_client()
        client._client = _FakeFlight([_FakeResult(_progress_body(1, "/d/a.zarr"))])
        with pytest.raises(RuntimeError, match="no terminal result"):
            client.add_source("/drop")

    def test_cancel_on_terminal_still_returns_tally(self):
        # #4: a cancel landing exactly on the terminal ``result`` must NOT discard
        # the completed tally. The poll flips True only after the progress
        # message, i.e. as the terminal is consumed -- the old top-of-loop poll
        # would have broken before capturing it and returned an empty tally.
        client = _bare_client()
        added = DataSourceDescriptor(source_id="s1")
        client._client = _FakeFlight(
            [
                _FakeResult(_progress_body(1, "/d/s1")),
                _FakeResult(_result_body(added=[added], already=["s0"])),
            ]
        )

        out = client.add_source("/drop", should_cancel=_flip_after(1))

        assert [d.source_id for d in out.added] == ["s1"]
        assert list(out.already_present) == ["s0"]

    def test_cancel_mid_walk_returns_empty_tally(self):
        # A genuine mid-walk cancel (before the terminal ever arrives) returns an
        # empty tally rather than raising; sources already registered surface
        # later via the watcher re-list.
        client = _bare_client()
        client._client = _FakeFlight(
            [
                _FakeResult(_progress_body(1, "/d/a.zarr")),
                _FakeResult(_progress_body(2, "/d/b.zarr")),
            ]
        )

        out = client.add_source("/drop", should_cancel=_flip_after(1))

        assert list(out.added) == [] and list(out.already_present) == []
