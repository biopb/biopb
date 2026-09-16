"""Unit tests for the SDK ``TensorFlightClient.resolve()`` and the directive
error that steers callers to it (cloud-storage phase 2, agent-facing API).

These exercise the resolve trigger and its caches without a live Flight server:
``_fetch_tensor_descriptor`` (the per-tensor GetFlightInfo) and ``list_sources``
are stubbed, and the client is built with ``object.__new__`` so no connection is
opened. The end-to-end resolve-on-serve path is covered by the server suite.
"""

import pyarrow as pa
import pytest
from biopb.tensor.client import ResolveCancelled, TensorFlightClient
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    ResolveProgress,
    ResolveStreamMessage,
    TensorDescriptor,
)


def _progress_body(elapsed, name="img.tif", nbytes=0):
    return ResolveStreamMessage(
        progress=ResolveProgress(
            elapsed_seconds=elapsed, target_name=name, target_bytes=nbytes
        )
    ).SerializeToString()


def _source_row(source_id, array_ids=(), resident=True):
    """One ``sources`` catalog row, Arrow IPC -- what the terminal message is."""
    table = pa.table(
        {
            "source_id": [source_id],
            "source_url": [f"file:///{source_id}"],
            "source_type": ["ome-zarr"],
            "data_resident": [resident],
            "tensors": [
                [
                    {
                        "array_id": aid,
                        "dim_labels": ["y", "x"],
                        "shape": [4, 4],
                        "dtype": "<f4",
                    }
                    for aid in array_ids
                ]
            ],
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _result_body(source_id, array_ids=(), resident=True):
    return ResolveStreamMessage(
        source_row=_source_row(source_id, array_ids, resident)
    ).SerializeToString()


def _bare_client():
    from biopb.tensor._session import CatalogClient, ChunkFetcher, _ClientState

    # add_source / resolve / warm now live on CatalogClient (#278 item C); build
    # the shared state + collaborators (no connection) and inject the fake flight
    # at ``client._state.client`` where the catalog reads it.
    client = object.__new__(TensorFlightClient)
    state = _ClientState(
        raw_client=None,
        call_options=None,
        location="",
        token=None,
        cache_bytes=0,
        protocol_checked=True,
    )
    client._state = state
    client._catalog = CatalogClient(state)
    client._fetcher = ChunkFetcher(state, client._catalog)
    return client


def _resolved_tensor(array_id):
    return TensorDescriptor(array_id=array_id, shape=[4, 4], dtype="<f4")


class _Buf:
    def __init__(self, body: bytes):
        self._body = body

    def to_pybytes(self) -> bytes:
        return self._body


class _FakeResult:
    """Mimics a pyarrow.flight.Result: ``result.body.to_pybytes()``."""

    def __init__(self, body: bytes):
        self.body = _Buf(body)


class _FakeFlight:
    """Records the action and replays a fixed stream of Results from do_action."""

    def __init__(self, results):
        self._results = results
        self.action = None

    def do_action(self, action, options=None):
        self.action = action
        return iter(self._results)


class TestResolve:
    def test_returns_the_full_row_from_resolve_action(self):
        # resolve() makes a single streaming `resolve` do_action and returns the
        # terminal row directly -- ALL fields, no list_sources, no cap.
        client = _bare_client()
        client._state.client = _FakeFlight(
            [_FakeResult(_result_body("cloud_x", ["cloud_x/f0", "cloud_x/f1"]))]
        )

        out = client.resolve("cloud_x")

        assert client._state.client.action.type == "resolve"
        assert bytes(client._state.client.action.body) == b"cloud_x"
        assert out.source_id == "cloud_x"
        assert out.source_url == "file:///cloud_x"
        assert out.data_resident is True
        assert len(out.tensors) == 2  # complete field set, never truncated
        assert client._sources["cloud_x"] is out  # cache seeded for reuse
        # The per-tensor cache is seeded too, so a following read needs no probe.
        assert set(client._descriptors) == {"cloud_x/f0", "cloud_x/f1"}

    def test_progress_envelopes_reported_then_terminal_taken(self):
        # Progress heartbeats (a ResolveStreamMessage `progress` arm) feed
        # on_progress; the terminal `source_row` arm carries the catalog row.
        client = _bare_client()
        client._state.client = _FakeFlight(
            [
                _FakeResult(_progress_body(0.0, "img.tif", 1024)),
                _FakeResult(_progress_body(0.5, "img.tif", 1024)),
                _FakeResult(_result_body("cloud_x", ["cloud_x"])),
            ]
        )
        seen = []

        out = client.resolve("cloud_x", on_progress=seen.append)

        assert [t.array_id for t in out.tensors] == ["cloud_x"]
        assert [round(p.elapsed_seconds, 1) for p in seen] == [0.0, 0.5]
        assert seen[0].target_name == "img.tif" and seen[0].target_bytes == 1024

    def test_empty_heartbeats_are_skipped(self):
        # An empty body is a heartbeat, not a terminal.
        client = _bare_client()
        client._state.client = _FakeFlight(
            [
                _FakeResult(b""),
                _FakeResult(b""),
                _FakeResult(_result_body("cloud_x", ["cloud_x"])),
            ]
        )

        out = client.resolve("cloud_x")

        assert [t.array_id for t in out.tensors] == ["cloud_x"]

    def test_should_cancel_raises_resolve_cancelled(self):
        # should_cancel polled per received message; True stops the stream and
        # raises ResolveCancelled rather than returning a descriptor.
        client = _bare_client()
        client._state.client = _FakeFlight(
            [
                _FakeResult(_progress_body(0.1)),
                _FakeResult(_result_body("cloud_x", ["cloud_x"])),
            ]
        )
        with pytest.raises(ResolveCancelled):
            client.resolve("cloud_x", should_cancel=lambda: True)
        assert "cloud_x" not in client._sources  # nothing cached on cancel

    def test_no_terminal_result_raises(self):
        # A stream of only heartbeats (server closed without a row) is an error,
        # not a silent empty descriptor.
        client = _bare_client()
        client._state.client = _FakeFlight(
            [_FakeResult(_progress_body(0.0)), _FakeResult(_progress_body(0.1))]
        )
        with pytest.raises(RuntimeError, match="no catalog row"):
            client.resolve("cloud_x")


class TestUnresolvedDirectiveError:
    def test_get_tensor_context_points_at_resolve(self):
        # A bare get_tensor() on an unresolved (empty-tensors) source must fail
        # with a directive message naming client.resolve(), not a bare "no tensors".
        client = _bare_client()
        client._sources = {
            "cloud_x": DataSourceDescriptor(source_id="cloud_x")  # no tensors
        }
        with pytest.raises(ValueError) as exc:
            client._get_tensor_context("cloud_x")
        msg = str(exc.value)
        assert "unresolved" in msg
        assert "client.resolve('cloud_x')" in msg


class TestSourceMetadataUnresolvedGuard:
    """F2 (#108): get_source_metadata must steer to resolve(), not return {}."""

    def test_unresolved_source_raises_instead_of_returning_empty(self, monkeypatch):
        # An unresolved source's row has an empty `tensors` list; the old
        # behavior returned {}, conflating "unresolved" with "resolved, no
        # metadata". It must instead raise the directive error -- and without
        # any GetFlightInfo recall.
        import pyarrow as pa

        client = _bare_client()
        monkeypatch.setattr(
            client._catalog,
            "_query_table",
            lambda sql: pa.table({"tensors": [[]], "metadata_json": [None]}),
        )
        recalled = []
        client._state.client = type(
            "FakeFlight",
            (),
            {"get_flight_info": lambda *a, **k: recalled.append(a)},
        )()
        with pytest.raises(ValueError) as exc:
            client.get_source_metadata("cloud_x")
        msg = str(exc.value)
        assert "unresolved" in msg
        assert "client.resolve('cloud_x')" in msg
        assert recalled == []  # no GetFlightInfo / download was triggered


class TestPhysicalScaleUnresolvedGuard:
    """F1: get_physical_scale must not silently recall a whole cloud file."""

    def test_unresolved_source_raises_instead_of_recalling(self, monkeypatch):
        client = _bare_client()
        client._sources = {
            "cloud_x": DataSourceDescriptor(source_id="cloud_x")  # unresolved
        }
        recalled = []
        monkeypatch.setattr(
            client._catalog,
            "_fetch_tensor_descriptor",
            lambda *a, **k: recalled.append(a) or _resolved_tensor("cloud_x"),
        )
        with pytest.raises(ValueError, match="client.resolve"):
            client.get_physical_scale("cloud_x")
        assert recalled == []  # no GetFlightInfo / download was triggered

    def test_resolved_source_still_fetches_scale(self, monkeypatch):
        # A resolved source is unaffected: the cheap one-shot fetch still runs.
        client = _bare_client()
        td = TensorDescriptor(
            array_id="r",
            shape=[2, 2],
            dtype="<f4",
            physical_scale=[0.5, 0.25],
            physical_unit=["um", "um"],
        )
        client._sources = {"r": DataSourceDescriptor(source_id="r", tensors=[td])}
        monkeypatch.setattr(
            client._catalog, "_fetch_tensor_descriptor", lambda *a, **k: td
        )
        scale, unit = client.get_physical_scale("r")
        assert scale == [0.5, 0.25]
        assert unit == ["um", "um"]
