"""A malformed slice/scale hint is the caller's mistake, not a server bug.

``core.chunk``'s request validators raised bare ``ValueError``s from deep inside
read planning, where ``get_flight_info``'s catch-all turned them into
``FlightInternalError("Metadata error for ...")``. Three things were wrong with
that: it blamed the server, it named the wrong subsystem, and ``INTERNAL`` is the
one code a client is entitled to retry -- so a request that can never succeed
looked like one worth sending again.

They are :class:`InvalidReadRequest` now, and the boundary maps them like the
rest of the typed taxonomy: a terminal ``FlightServerError`` carrying
``INVALID_ARGUMENT`` in ``extra_info``.
"""

import json

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import FlightRequest, SliceHint, TensorReadOption
from biopb_tensor_server.core.chunk import (
    normalized_scale_hint,
    normalized_slice_bounds,
)
from biopb_tensor_server.core.errors import (
    InvalidReadRequest,
    TensorResolutionError,
)
from google.protobuf.field_mask_pb2 import FieldMask


class TestTheType:
    def test_it_is_part_of_the_terminal_client_error_taxonomy(self):
        assert issubclass(InvalidReadRequest, TensorResolutionError)

    def test_it_is_still_a_value_error(self):
        """Every ``except ValueError`` guard written before the taxonomy has to
        keep catching it -- the same contract the rest of the tree keeps."""
        assert issubclass(InvalidReadRequest, ValueError)

    def test_it_carries_the_canonical_code(self):
        assert InvalidReadRequest.grpc_code == "INVALID_ARGUMENT"


class TestTheValidators:
    """The reason slugs, which are the machine-readable half of the answer."""

    @pytest.mark.parametrize(
        ("scale_hint", "reason"),
        [
            ((2, 2, 2), "scale_rank"),
            ((2, 0), "scale_not_positive"),
            ((2, -1), "scale_not_positive"),
        ],
    )
    def test_scale_hint(self, scale_hint, reason):
        with pytest.raises(InvalidReadRequest) as exc:
            normalized_scale_hint((4, 4), scale_hint)
        assert exc.value.reason == reason

    @pytest.mark.parametrize(
        ("start", "stop", "reason"),
        [
            ([0], [4], "slice_rank"),
            ([-1, 0], [4, 4], "slice_negative"),
            ([3, 0], [1, 4], "slice_inverted"),
            ([0, 0], [9, 4], "slice_out_of_range"),
        ],
    )
    def test_slice_bounds(self, start, stop, reason):
        with pytest.raises(InvalidReadRequest) as exc:
            normalized_slice_bounds((4, 4), SliceHint(start=start, stop=stop))
        assert exc.value.reason == reason

    def test_a_valid_hint_is_untouched(self):
        assert normalized_scale_hint((4, 4), (2, 2)) == (2, 2)
        assert normalized_slice_bounds(
            (4, 4), SliceHint(start=[0, 0], stop=[2, 2])
        ) == (
            (0, 0),
            (2, 2),
        )


def _zarr_source(writable_server, tmp_path, name="plain"):
    import zarr
    from biopb_tensor_server.adapters.zarr import ZarrAdapter

    path = tmp_path / f"{name}.zarr"
    arr = zarr.open_array(
        str(path), mode="w", shape=(4, 4), chunks=(2, 2), dtype="uint8"
    )
    arr[:] = 3
    writable_server.register_source(
        name, ZarrAdapter(zarr.open_array(str(path), mode="r"), name, ["y", "x"])
    )
    return name


def _get_info(client, sid, *, fields=("endpoints",), **read_opt):
    cmd = FlightRequest(
        tensor_read=TensorReadOption(
            array_id=sid, fields=FieldMask(paths=list(fields)), **read_opt
        )
    )
    fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
    return client._state.client.get_flight_info(fd)


class TestOverTheWire:
    def test_a_bad_scale_hint_is_terminal_not_internal(
        self, client, writable_server, tmp_path
    ):
        """``FlightInternalError`` is 'server bug, retrying may help'. Neither
        half is true here, and the class is what a client without the typed
        codes switches on."""
        sid = _zarr_source(writable_server, tmp_path)
        with pytest.raises(flight.FlightServerError) as exc:
            _get_info(client, sid, scale_hint=[2, 2, 2])
        assert json.loads(exc.value.extra_info) == {
            "code": "INVALID_ARGUMENT",
            "reason": "scale_rank",
        }

    def test_a_slice_past_the_shape_is_terminal_not_internal(
        self, client, writable_server, tmp_path
    ):
        sid = _zarr_source(writable_server, tmp_path)
        with pytest.raises(flight.FlightServerError) as exc:
            _get_info(client, sid, slice_hint=SliceHint(start=[0, 0], stop=[99, 4]))
        assert json.loads(exc.value.extra_info) == {
            "code": "INVALID_ARGUMENT",
            "reason": "slice_out_of_range",
        }

    def test_it_no_longer_claims_to_be_a_metadata_error(
        self, client, writable_server, tmp_path
    ):
        """The old wrapper named the subsystem that happened to share the try
        block, which sent anyone reading the message to the wrong place."""
        sid = _zarr_source(writable_server, tmp_path)
        with pytest.raises(flight.FlightError) as exc:
            _get_info(client, sid, scale_hint=[2, 2, 2])
        assert "Metadata error" not in str(exc.value)
        assert "Scale hint dimensionality mismatch" in str(exc.value)

    def test_a_real_metadata_failure_still_reports_as_one(
        self, client, writable_server, tmp_path, monkeypatch
    ):
        """The narrowing must not cost the arm its actual job."""
        sid = _zarr_source(writable_server, tmp_path)

        def boom(*a, **k):
            raise ValueError("catalog exploded")

        assert writable_server._metadata_db is not None  # else nothing is patched
        monkeypatch.setattr(writable_server._metadata_db, "get_metadata_json", boom)
        with pytest.raises(flight.FlightInternalError, match="catalog exploded") as exc:
            _get_info(client, sid, fields=["metadata_json"])
        assert "Metadata error" in str(exc.value)

    def test_a_valid_request_still_plans(self, client, writable_server, tmp_path):
        sid = _zarr_source(writable_server, tmp_path)
        assert np.asarray(client.get_tensor(sid))[0, 0] == 3


class TestTheSdkSees:
    def test_the_python_client_restates_it_as_a_value_error(
        self, client, writable_server, tmp_path
    ):
        """The consequence the client-side taxonomy already had wired up: it
        switches on the canonical code, so this became a clean ValueError the
        moment the server stopped calling it INTERNAL."""
        sid = _zarr_source(writable_server, tmp_path)
        with pytest.raises(ValueError, match="Scale hint dimensionality mismatch"):
            client.get_tensor(sid, scale_hint=[2, 2, 2])
