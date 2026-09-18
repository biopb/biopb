"""Client-only unit tests for TensorFlightClient (no live server).

Server-backed round-trip tests that spin up a real TensorFlightServer live in
biopb-tensor-server/tests/client_flight_test.py -- alongside the server package,
whose CI installs it. They were silently skipped here because the client-only CI
job installs no server (biopb/biopb#579).
"""

import pickle
from unittest.mock import Mock

import pytest
from biopb.tensor import (
    TensorFlightClient,
)
from biopb.tensor.descriptor_pb2 import TensorDescriptor


def _offline_client(raw_client=None):
    """A wired TensorFlightClient with no connection opened.

    Built without ``__init__`` so no socket is created: the shared state plus the
    collaborators (#278 item C). ``protocol_checked`` skips the health probe a
    real connect would run.
    """
    from biopb.tensor._session import CatalogClient, ChunkFetcher, _ClientState

    client = TensorFlightClient.__new__(TensorFlightClient)
    state = _ClientState(
        raw_client=raw_client,
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


class TestQuerySourcesFormat:
    """query_sources output-format conversion (server-free).

    Exercises TensorFlightClient._format_query_result directly. The default
    stays 'arrow' (pyarrow.Table) for backward compatibility; 'pandas' and
    'records' are opt-in conveniences.
    """

    @staticmethod
    def _table():
        import pyarrow as pa

        return pa.table(
            {
                "source_id": ["a", "b"],
                # list column mirrors the real `shape_summary` catalog field
                "shape_summary": [[1, 4, 5734, 5734], [1, 5, 7616, 7616]],
            }
        )

    def test_default_format_is_arrow(self):
        # Backward-compat guard: the historical pyarrow.Table return must stay
        # the default so existing Arrow consumers don't break.
        import inspect

        default = (
            inspect.signature(TensorFlightClient.query_sources)
            .parameters["format"]
            .default
        )
        assert default == "arrow"

    def test_arrow_passthrough_is_same_object(self):
        t = self._table()
        assert TensorFlightClient._format_query_result(t, "arrow") is t

    def test_pandas_format_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        t = self._table()
        out = TensorFlightClient._format_query_result(t, "pandas")
        assert isinstance(out, pd.DataFrame)
        assert list(out["source_id"]) == ["a", "b"]
        assert list(out["shape_summary"].iloc[0]) == [1, 4, 5734, 5734]

    def test_pandas_string_nulls_become_none_not_nan(self):
        # issue #47: a NULL in a *string* column (e.g. metadata_json) coerces
        # to a float NaN under Arrow->pandas, and NaN is *truthy* -- so the
        # obvious `if row.metadata_json:` guard passes and json.loads() then
        # blows up on a float. We normalize string-column nulls to None.
        pytest.importorskip("pandas")
        import pyarrow as pa

        t = pa.table(
            {
                # mixed: one row has metadata, one is NULL -> Arrow `string`
                # col with nulls (the sharp case; an all-NULL page is already
                # None because it becomes an Arrow `null`-typed column).
                "source_id": ["a", "b"],
                "metadata_json": pa.array(['{"x": 1}', None], type=pa.string()),
            }
        )
        out = TensorFlightClient._format_query_result(t, "pandas")
        missing = out["metadata_json"].iloc[1]
        assert missing is None
        assert not missing  # falsy, so `if row.metadata_json:` is skipped
        assert out["metadata_json"].iloc[0] == '{"x": 1}'

    def test_pandas_numeric_nan_is_untouched(self):
        # The fix targets string columns by Arrow schema, so a genuine NaN in a
        # real float column must survive (we only normalize text nulls).
        pytest.importorskip("pandas")
        import math

        import pyarrow as pa

        t = pa.table({"score": pa.array([1.5, None], type=pa.float64())})
        out = TensorFlightClient._format_query_result(t, "pandas")
        assert math.isnan(out["score"].iloc[1])

    def test_records_returns_list_of_dicts(self):
        t = self._table()
        out = TensorFlightClient._format_query_result(t, "records")
        assert out == [
            {"source_id": "a", "shape_summary": [1, 4, 5734, 5734]},
            {"source_id": "b", "shape_summary": [1, 5, 7616, 7616]},
        ]

    def test_unknown_format_rejected_before_network(self):
        # Validated at the top of query_sources (now on CatalogClient, #278 item
        # C), so a bad format fails fast without a server / connection.
        client = _offline_client()
        with pytest.raises(ValueError, match="unknown format"):
            client.query_sources("SELECT 1", format="polars")


class TestGetPhysicalScale:
    """get_physical_scale describes the tensor, every call.

    Exercises the client accessor for the per-dim physical-scale summary the
    server folds onto the descriptor (issue #31), stubbing the fetch so no
    connection is needed.
    """

    @staticmethod
    def _client():
        # get_physical_scale lives on CatalogClient (#278 item C) and reaches
        # the server only through _fetch_tensor_descriptor, stubbed here.
        client = _offline_client()
        client._catalog._fetch_tensor_descriptor = Mock()
        return client

    @staticmethod
    def _desc(array_id, scale=None, unit=None):
        desc = TensorDescriptor(array_id=array_id, dim_labels=["z", "y", "x"])
        if scale is not None:
            desc.physical_scale[:] = scale
            desc.physical_unit[:] = unit
        return desc

    def test_it_asks_the_server_every_time(self):
        # physical_scale is a GetFlightInfo field the catalog leaves empty, so
        # only a fetched descriptor can answer it.
        client = self._client()
        client._catalog._fetch_tensor_descriptor.return_value = self._desc(
            "src/t1", [2.0, 0.325, 0.325], ["micrometer"] * 3
        )

        for _ in range(3):
            assert client.get_physical_scale("src/t1")[0] == [2.0, 0.325, 0.325]
        assert client._catalog._fetch_tensor_descriptor.call_count == 3

    def test_none_when_summary_empty(self):
        # Old server / no physical sizes -> empty repeated field -> None.
        client = self._client()
        client._catalog._fetch_tensor_descriptor.return_value = self._desc("src/t1")

        assert client.get_physical_scale("src/t1") is None

    def test_a_bare_id_fetches_the_sources_default_tensor(self):
        # The property this holds that the test above does not: a bare source id
        # is passed through to the server, which answers with the source's
        # default tensor (#44), and the compact mask is what goes on the wire.
        client = self._client()
        desc = self._desc("t1", [1.0, 0.5, 0.5], ["", "micrometer", "micrometer"])
        client._catalog._fetch_tensor_descriptor.return_value = desc

        scale, unit = client.get_physical_scale("src")  # bare source id -> default
        assert scale == [1.0, 0.5, 0.5]
        assert unit == ["", "micrometer", "micrometer"]
        # physical scale is a compact probe: it fetches the structural descriptor
        # only, never the opt-in OME tree (so it needs no metadata catalog).
        client._catalog._fetch_tensor_descriptor.assert_called_once_with(
            "src", with_metadata=False
        )

    def test_fetch_error_propagates(self):
        # A real fetch failure (server unreachable, source not found) must NOT be
        # swallowed into None: that would make it indistinguishable from "no
        # physical scale recorded". Only a fetched descriptor with an empty
        # summary yields None (test_none_when_summary_empty).
        client = self._client()
        client._catalog._fetch_tensor_descriptor.side_effect = ConnectionError(
            "unreachable"
        )

        with pytest.raises(ConnectionError):
            client.get_physical_scale("src")


class TestGetDescriptorFieldMasks:
    """get_descriptor sends a describe-shaped field mask (#563).

    Every optional part is opt-in, so the mask on the wire *is* the request --
    nothing is implied by omission any more. get_descriptor is a describe (the
    stable per-tensor facts, not a read), so it asks for the pyramid its
    consumer reads and nothing else: no `endpoints` (the per-request plan it
    would discard), no `metadata_json` (the heavy OME tree).

    Guarded exactly, because both defaults are silent when wrong. A revert of
    the metadata opt-in would quietly ship megabytes per call, and an
    `endpoints` path creeping back in would turn every describe into an
    O(chunks) enumeration.
    """

    @staticmethod
    def _client_capturing_read_opt():
        # Build without __init__ (no connection); mock the flight client so we can
        # decode the FlightRequest the descriptor probe puts on the wire.
        client = _offline_client(raw_client=Mock())
        state = client._state
        # get_flight_info returns a FlightInfo whose descriptor.command is a
        # serialized TensorDescriptor (what _fetch_tensor_descriptor parses back).
        info = Mock()
        info.descriptor.command = TensorDescriptor(
            array_id="src/A2"
        ).SerializeToString()
        state.client.get_flight_info.return_value = info
        return client, state

    @staticmethod
    def _sent_read_opt(state):
        from biopb.tensor.descriptor_pb2 import FlightRequest

        fd = state.client.get_flight_info.call_args.args[0]
        return FlightRequest.FromString(fd.command).tensor_read

    def test_defaults_are_describe_shaped(self):
        client, state = self._client_capturing_read_opt()
        client.get_descriptor("src/A2")
        assert set(self._sent_read_opt(state).fields.paths) == {"pyramid"}

    def test_with_metadata_opt_in_is_forwarded(self):
        client, state = self._client_capturing_read_opt()
        client.get_descriptor("src/A2", with_metadata=True)
        assert "metadata_json" in set(self._sent_read_opt(state).fields.paths)

    def test_residency_is_never_asked_for_by_default(self):
        """The expensive one. It is a stat walk of the source, so a describe
        that did not ask must not get it -- and must not pay for it."""
        client, state = self._client_capturing_read_opt()
        client.get_descriptor("src/A2")
        assert "is_resident" not in set(self._sent_read_opt(state).fields.paths)

    def test_residency_opt_in_is_forwarded(self):
        client, state = self._client_capturing_read_opt()
        client.get_descriptor("src/A2", with_residency=True)
        assert "is_resident" in set(self._sent_read_opt(state).fields.paths)

    def test_a_describe_never_asks_for_the_read_plan(self):
        """`endpoints` is the O(chunks) enumeration. Under the bools this
        replaced it was ON unless explicitly disabled, so a describe had to
        remember to opt *out*; the default now costs nothing."""
        client, state = self._client_capturing_read_opt()
        client.get_descriptor("src/A2")
        assert "endpoints" not in set(self._sent_read_opt(state).fields.paths)


class TestDescriptorsAreNotCached:
    """The SDK stores no descriptor; every describe is a round trip."""

    @staticmethod
    def _client(response: TensorDescriptor):
        client = _offline_client(raw_client=Mock())
        info = Mock()
        info.descriptor.command = response.SerializeToString()
        client._state.client.get_flight_info.return_value = info
        return client

    @staticmethod
    def _fat_descriptor():
        from biopb.tensor.descriptor_pb2 import PyramidLevel

        desc = TensorDescriptor(
            array_id="src/A2",
            dim_labels=["z", "y", "x"],
            shape=[8, 64, 64],
            chunk_shape=[1, 64, 64],
            dtype="uint16",
            metadata_json='{"metadata": {"big": "' + "x" * 4096 + '"}}',
        )
        desc.physical_scale[:] = [2.0, 0.325, 0.325]
        desc.physical_unit[:] = ["micrometer"] * 3
        desc.pyramid.append(PyramidLevel(scale_hint=[1, 1, 1], reduction_method="area"))
        desc.pyramid.append(PyramidLevel(scale_hint=[1, 4, 4], reduction_method="area"))
        return desc

    def test_caller_gets_the_fields_it_asked_for(self):
        client = self._client(self._fat_descriptor())

        returned = client.get_descriptor("src/A2", with_metadata=True)

        assert returned.metadata_json  # the mask is honoured on the return value
        assert len(returned.pyramid) == 2

    def test_there_is_nowhere_to_cache_one(self):
        # Gone, not merely unused: nothing can quietly start writing to it.
        client = self._client(self._fat_descriptor())

        client.get_descriptor("src/A2", with_metadata=True)

        assert not hasattr(client, "_descriptors")
        assert not hasattr(client._state, "descriptors")

    def test_every_describe_round_trips(self):
        client = self._client(self._fat_descriptor())

        for _ in range(3):
            client.get_descriptor("src/A2")

        assert client._state.raw_client.get_flight_info.call_count == 3

    def test_masked_fetch_does_not_poison_a_later_full_fetch(self):
        # The regression #795 asks for: a pyramid-less fetch first, then a
        # default one. Every get_descriptor round-trips, so the second caller
        # sees the pyramid regardless of what the first one asked for.
        client = self._client(self._fat_descriptor())

        client.get_descriptor("src/A2", with_pyramid=False)
        second = client.get_descriptor("src/A2")

        assert len(second.pyramid) == 2


class TestResolveDescriptorAddressing:
    """The read path's addressing refusals, read off the catalog row."""

    @staticmethod
    def _client(row):
        client = _offline_client(raw_client=Mock())
        client._catalog._source_tensors_row = Mock(return_value=row)
        # The row answers every case here; reaching the probe is the failure.
        client._catalog._fetch_tensor_descriptor = Mock(
            side_effect=AssertionError("the row should have answered")
        )
        return client

    @staticmethod
    def _row(*array_ids, is_resolved=True):
        return {
            "is_resolved": is_resolved,
            "tensors": [
                {
                    "array_id": aid,
                    "dim_labels": ["y", "x"],
                    "shape": [4, 4],
                    "dtype": "uint8",
                }
                for aid in array_ids
            ],
        }

    def test_unresolved_steers_to_resolve(self):
        client = self._client(self._row(is_resolved=False))

        with pytest.raises(ValueError, match=r"call client\.resolve"):
            client._catalog._resolve_descriptor("cloud_x")

    def test_resolved_but_empty_does_not_steer_to_resolve(self):
        # A source can resolve cleanly and hold nothing readable; the flag
        # distinguishes that from unresolved, an empty tensor list cannot.
        client = self._client(self._row())

        with pytest.raises(ValueError, match="no readable tensors") as exc:
            client._catalog._resolve_descriptor("empty_x")
        assert "client.resolve(" not in str(exc.value)

    def test_bare_id_on_a_multi_tensor_source_is_refused(self):
        client = self._client(self._row("m/f0", "m/f1"))

        with pytest.raises(ValueError, match="multiple tensors"):
            client._catalog._resolve_descriptor("m")

    def test_a_qualified_id_resolves_off_the_row(self):
        client = self._client(self._row("m/f0", "m/f1"))

        desc = client._catalog._resolve_descriptor("m/f1")

        assert desc.array_id == "m/f1"
        assert list(desc.shape) == [4, 4]

    def test_a_bare_id_on_a_single_tensor_source_resolves(self):
        client = self._client(self._row("solo"))

        assert client._catalog._resolve_descriptor("solo").array_id == "solo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestUploadRefused:
    """The one typed exception a write path raises (biopb/biopb#1048 step 7)."""

    def test_it_survives_a_trip_through_a_worker(self):
        """Raised on a dask worker, it comes back as itself with its fields."""
        from biopb.tensor import UploadRefused

        exc = UploadRefused("cache_x", "DISCARDED", "job died")
        back = pickle.loads(pickle.dumps(exc))
        assert isinstance(back, UploadRefused)
        assert (back.source_id, back.state, back.reason) == (
            "cache_x",
            "DISCARDED",
            "job died",
        )
        assert "job died" in str(back)

    def test_it_is_read_off_extra_info_not_the_message(self):
        import json

        import pyarrow.flight as flight
        from biopb.tensor._upload import _refused_from

        info = {
            "code": "CANCELLED",
            "reason": "upload_sealed",
            "state": "READY",
            "source_id": "cache_x",
            "detail": "",
        }
        exc = flight.FlightCancelledError(
            "whatever the message says", json.dumps(info).encode()
        )
        refused = _refused_from(exc)
        assert refused is not None
        assert refused.state == "READY"
        assert refused.source_id == "cache_x"

    def test_another_cancelled_call_passes_through(self):
        """Only an upload refusal is translated; a cancel from anything else is
        not this module's to reinterpret."""
        import pyarrow.flight as flight
        from biopb.tensor._upload import _refused_from

        assert _refused_from(flight.FlightCancelledError("cancelled")) is None
        assert _refused_from(flight.FlightCancelledError("x", b"not json")) is None
        assert (
            _refused_from(flight.FlightCancelledError("x", b'{"reason": "other"}'))
            is None
        )


class TestCreateTensorGrid:
    def test_a_dask_template_supplies_its_grid(self):
        import dask.array as da
        from biopb.tensor._upload import _uniform_chunk_shape

        arr = da.zeros((10, 6), chunks=(4, 6))  # ragged trailing chunk on axis 0
        assert _uniform_chunk_shape(arr) == (4, 6)

    def test_an_irregular_chunking_yields_one_grid(self):
        import dask.array as da
        from biopb.tensor._upload import _uniform_chunk_shape

        arr = da.zeros((10,), chunks=((3, 5, 2),))
        assert _uniform_chunk_shape(arr) == (5,)
