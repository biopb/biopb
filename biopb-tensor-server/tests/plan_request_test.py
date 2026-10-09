"""What a plan says about the request it answers (Flight protocol v3).

A v3 plan's ``FlightInfo.app_metadata`` is the whole ``TensorReadOption`` it was
asked with, and its descriptor no longer echoes the request's scale and method.
A v2 plan carried only the requested slice there and echoed the rest. A new SDK
reads both; an older one refuses a v3 server by its protocol check.
"""

import json
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from biopb.tensor import TensorFlightClient
from biopb.tensor._session import (
    _check_flight_protocol,
    _crop_slices,
    _plan_request,
)
from biopb.tensor._wire_version import (
    FLIGHT_PROTOCOL_METADATA_KEY,
    FLIGHT_PROTOCOL_VERSION,
    SUPPORTED_FLIGHT_PROTOCOLS,
    WIRE_PROTOCOL_METADATA_KEY,
)
from biopb.tensor.descriptor_pb2 import (
    SliceHint,
    TensorDescriptor,
    TensorReadOption,
)
from biopb_tensor_server import TensorFlightServer

from tests import catalog_server, grid_source, register_and_catalog, serve

SHAPE = (64, 64)


def _data():
    return np.arange(SHAPE[0] * SHAPE[1], dtype=np.uint8).reshape(SHAPE)


def _plan(client, *args, **kwargs):
    return client._fetcher._plan_read(*args, **kwargs)


# ---------------------------------------------------------------------- v3 shape


@pytest.fixture
def served(tmp_path, transfer_target):
    transfer_target(256)  # one 16x16 uint8 block per chunk
    server = catalog_server("localhost:0")
    register_and_catalog(server, "img", grid_source(tmp_path)[1])
    server.mark_ready()
    serve(server)
    client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=0)
    yield server, client
    client.close()
    server.shutdown()


class TestV3Plan:
    def test_the_server_speaks_v3(self, served):
        _, client = served
        assert client._state.client and client._state.server_health["protocol"] == 3
        assert FLIGHT_PROTOCOL_VERSION == 3

    def test_the_plan_records_the_whole_request(self, served):
        _, client = served
        info = _plan(
            client,
            "img",
            (slice(3, 40), slice(5, 50)),
            scale_hint=(2, 2),
            reduction_method="nearest",
        )
        request = TensorReadOption.FromString(info.app_metadata)
        assert request.array_id == "img"
        assert list(request.slice_hint.start) == [3, 5]
        assert list(request.slice_hint.stop) == [40, 50]
        assert list(request.scale_hint) == [2, 2]
        assert request.reduction_method == "nearest"
        assert "endpoints" in request.fields.paths

    def test_an_unsliced_plan_still_records_it(self, served):
        _, client = served
        info = _plan(client, "img")
        assert TensorReadOption.FromString(info.app_metadata).array_id == "img"

    def test_the_descriptor_does_not_echo_the_request(self, served):
        _, client = served
        info = _plan(client, "img", scale_hint=(2, 2), reduction_method="area")
        descriptor = TensorDescriptor.FromString(info.descriptor.command)
        assert not descriptor.scale_hint
        assert not descriptor.reduction_method
        # What the server decided is still there: the realized region and the
        # logical shape.
        assert list(descriptor.shape) == [32, 32]

    def test_the_realized_slice_stays_on_the_descriptor(self, served):
        _, client = served
        info = _plan(client, "img", (slice(3, 40), slice(5, 50)))
        descriptor = TensorDescriptor.FromString(info.descriptor.command)
        assert descriptor.HasField("slice_hint")
        assert list(descriptor.slice_hint.start) == [0, 0]

    def test_the_schema_says_which_protocol_wrote_it(self, served):
        _, client = served
        info = _plan(client, "img")
        assert info.schema.metadata[FLIGHT_PROTOCOL_METADATA_KEY.encode()] == b"3"

    def test_a_request_reads_back_through_a_serialized_plan(self, served):
        """The plan travels as bytes and says how to read itself."""
        _, client = served
        pb = client.get_tensor(
            "img", (slice(3, 40), slice(5, 50)), scale_hint=(1, 1), output="pb"
        )
        out = TensorFlightClient.tensor_from_pb(pb).compute()
        np.testing.assert_array_equal(out, _data()[3:40, 5:50])

    def test_a_scaled_slice_crops_by_the_recorded_scale(self, served):
        _, client = served
        darr = client.get_tensor(
            "img",
            (slice(8, 40), slice(8, 40)),
            scale_hint=(2, 2),
            reduction_method="nearest",
        )
        assert darr.shape == (16, 16)
        pb = client.get_tensor(
            "img",
            (slice(8, 40), slice(8, 40)),
            scale_hint=(2, 2),
            reduction_method="nearest",
            output="pb",
        )
        np.testing.assert_array_equal(
            TensorFlightClient.tensor_from_pb(pb).compute(), darr.compute()
        )

    def test_an_endpointless_handle_replays_its_request(self, served):
        """A describe-shaped handle is planned by replaying what it recorded."""
        from biopb.tensor._session import _refetch_flight_info

        server, client = served
        info = _plan(client, "img", (slice(3, 40), slice(5, 50)), scale_hint=(1, 1))
        request = _plan_request(info)
        descriptor = TensorDescriptor.FromString(info.descriptor.command)
        again = _refetch_flight_info(
            descriptor, f"grpc://localhost:{server.port}", None, None, request
        )
        assert len(again.endpoints) == len(info.endpoints)
        assert (
            TensorDescriptor.FromString(again.descriptor.command).slice_hint
            == descriptor.slice_hint
        )

    def test_a_handle_built_by_hand_records_nothing_and_still_plans(self, served):
        from biopb.tensor._session import _refetch_flight_info

        server, _ = served
        again = _refetch_flight_info(
            TensorDescriptor(array_id="img"),
            f"grpc://localhost:{server.port}",
            None,
        )
        assert len(again.endpoints) == 16


# ---------------------------------------------------------------- reading a v2 plan


def _v2_info(slice_hint=None, scale=None, method=""):
    """A FlightInfo as a v2 server wrote it: the slice in app_metadata, the scale
    and method echoed on the descriptor, no protocol stamp on the schema."""
    descriptor = TensorDescriptor(array_id="img", shape=[8, 8], dtype="|u1")
    if scale:
        descriptor.scale_hint[:] = scale
    descriptor.reduction_method = method
    return flight.FlightInfo(
        schema=pa.schema([]).with_metadata({WIRE_PROTOCOL_METADATA_KEY: "2"}),
        descriptor=flight.FlightDescriptor.for_command(descriptor.SerializeToString()),
        endpoints=[],
        total_records=-1,
        total_bytes=-1,
        app_metadata=slice_hint.SerializeToString() if slice_hint else b"",
    )


class TestPlanRequest:
    def test_a_v2_plan_is_put_back_together(self):
        info = _v2_info(
            SliceHint(start=[1, 2], stop=[7, 8]), scale=(2, 2), method="area"
        )
        request = _plan_request(info)
        assert request.array_id == "img"
        assert list(request.slice_hint.start) == [1, 2]
        assert list(request.slice_hint.stop) == [7, 8]
        assert list(request.scale_hint) == [2, 2]
        assert request.reduction_method == "area"

    def test_a_v2_plan_without_a_slice_has_none(self):
        request = _plan_request(_v2_info(scale=(2, 2), method="nearest"))
        assert not request.HasField("slice_hint")
        assert list(request.scale_hint) == [2, 2]

    def test_a_plan_built_by_hand_records_nothing(self):
        request = _plan_request(_v2_info())
        assert not request.HasField("slice_hint")
        assert not request.scale_hint

    def test_a_v3_plan_is_what_app_metadata_says(self):
        wanted = TensorReadOption(array_id="img")
        wanted.slice_hint.start[:] = [1, 2]
        wanted.slice_hint.stop[:] = [7, 8]
        wanted.scale_hint[:] = [2, 2]
        info = flight.FlightInfo(
            schema=pa.schema([]).with_metadata({FLIGHT_PROTOCOL_METADATA_KEY: "3"}),
            descriptor=flight.FlightDescriptor.for_command(
                TensorDescriptor(array_id="img").SerializeToString()
            ),
            endpoints=[],
            total_records=-1,
            total_bytes=-1,
            app_metadata=wanted.SerializeToString(),
        )
        assert _plan_request(info) == wanted

    def test_a_slice_bytes_are_not_misread_as_a_v3_request(self):
        """The reason the protocol is stamped on the schema: a SliceHint's bytes
        would parse as a TensorReadOption with a garbled array_id."""
        info = _v2_info(SliceHint(start=[1, 2], stop=[7, 8]))
        assert list(_plan_request(info).slice_hint.stop) == [7, 8]

    def test_the_crop_uses_the_requests_scale(self):
        descriptor = TensorDescriptor(array_id="img", shape=[16, 16])
        descriptor.slice_hint.start[:] = [0, 0]
        descriptor.slice_hint.stop[:] = [64, 64]
        request = TensorReadOption()
        request.slice_hint.start[:] = [8, 8]
        request.slice_hint.stop[:] = [40, 40]
        request.scale_hint[:] = [2, 2]
        assert _crop_slices(descriptor, request, 2) == (slice(4, 20), slice(4, 20))

    def test_no_requested_slice_is_no_crop(self):
        descriptor = TensorDescriptor(array_id="img")
        descriptor.slice_hint.start[:] = [0, 0]
        descriptor.slice_hint.stop[:] = [64, 64]
        assert _crop_slices(descriptor, TensorReadOption(), 2) is None
        assert _crop_slices(descriptor, None, 2) is None


# ------------------------------------------------------------ a v2 server, for real


class _V2Server(TensorFlightServer):
    """A server that answers the way a v2 one did: protocol 2 on health, no
    ``ticket_stubs``, and plans with the slice alone in ``app_metadata``, the
    scale and method echoed on the descriptor, and no protocol stamp."""

    def do_action(self, context, action):
        for result in super().do_action(context, action):
            if action.type == "health":
                health = json.loads(result)
                health["protocol"] = 2
                health.pop("ticket_stubs", None)
                yield json.dumps(health).encode()
            else:
                yield result

    def get_flight_info(self, context, descriptor):
        info = super().get_flight_info(context, descriptor)
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            return info
        request = TensorReadOption.FromString(info.app_metadata)
        desc = TensorDescriptor.FromString(info.descriptor.command)
        desc.scale_hint[:] = request.scale_hint
        desc.reduction_method = request.reduction_method
        desc.ClearField("ticket_stub")
        desc.ClearField("roi_ticket")
        return flight.FlightInfo(
            schema=info.schema.remove_metadata().with_metadata(
                {WIRE_PROTOCOL_METADATA_KEY: "2"}
            ),
            descriptor=flight.FlightDescriptor.for_command(desc.SerializeToString()),
            endpoints=info.endpoints,
            total_records=-1,
            total_bytes=-1,
            app_metadata=(
                request.slice_hint.SerializeToString()
                if request.HasField("slice_hint")
                else b""
            ),
        )


@pytest.fixture
def legacy(tmp_path, transfer_target):
    from biopb_tensor_server.serving.metadata_db import MetadataDatabase

    transfer_target(256)
    server = _V2Server("localhost:0", metadata_db=MetadataDatabase())
    register_and_catalog(server, "img", grid_source(tmp_path)[1])
    server.mark_ready()
    serve(server)
    client = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=0)
    yield server, client
    client.close()
    server.shutdown()


class TestAgainstAV2Server:
    """The new SDK works with the old server."""

    def test_the_server_really_answers_in_the_old_shape(self, legacy):
        _, client = legacy
        info = _plan(client, "img", (slice(3, 40), slice(5, 50)), scale_hint=(1, 1))
        assert FLIGHT_PROTOCOL_METADATA_KEY.encode() not in (info.schema.metadata or {})
        assert list(SliceHint.FromString(info.app_metadata).stop) == [40, 50]

    def test_it_is_accepted(self, legacy):
        _, client = legacy
        assert client._state.client
        assert client._state.server_health["protocol"] == 2
        assert 2 in SUPPORTED_FLIGHT_PROTOCOLS

    def test_a_whole_read(self, legacy):
        _, client = legacy
        np.testing.assert_array_equal(client.get_tensor("img").compute(), _data())

    def test_a_sliced_read_crops(self, legacy):
        _, client = legacy
        out = client.get_tensor("img", (slice(3, 40), slice(5, 50))).compute()
        np.testing.assert_array_equal(out, _data()[3:40, 5:50])

    def test_a_scaled_sliced_read_crops_by_the_echoed_scale(self, legacy):
        _, client = legacy
        out = client.get_tensor(
            "img",
            (slice(8, 40), slice(8, 40)),
            scale_hint=(2, 2),
            reduction_method="nearest",
        ).compute()
        assert out.shape == (16, 16)

    def test_a_serialized_plan_reads_back(self, legacy):
        _, client = legacy
        pb = client.get_tensor("img", (slice(3, 40), slice(5, 50)), output="pb")
        np.testing.assert_array_equal(
            TensorFlightClient.tensor_from_pb(pb).compute(), _data()[3:40, 5:50]
        )

    def test_it_is_planned_as_full_tickets(self, legacy):
        """No ``ticket_stubs`` on its health, so none is asked for."""
        _, client = legacy
        info = _plan(client, "img")
        assert not TensorDescriptor.FromString(info.descriptor.command).ticket_stub


# ------------------------------------------------------------------ the check itself


def _client_with_health(protocol):
    client = Mock()
    client.do_action.return_value = iter(
        [
            Mock(
                body=Mock(
                    to_pybytes=lambda: json.dumps({"protocol": protocol}).encode()
                )
            )
        ]
    )
    return client


class TestProtocolCheck:
    @pytest.mark.parametrize("version", sorted(SUPPORTED_FLIGHT_PROTOCOLS))
    def test_a_supported_version_passes(self, version):
        _check_flight_protocol(_client_with_health(version), None, "grpc://x:1")

    def test_a_v1_server_is_named_stale(self):
        with pytest.raises(RuntimeError, match="Upgrade the server"):
            _check_flight_protocol(_client_with_health(1), None, "grpc://x:1")

    def test_a_newer_server_is_named_as_the_clients_to_upgrade(self):
        with pytest.raises(RuntimeError, match="Upgrade the client"):
            _check_flight_protocol(_client_with_health(99), None, "grpc://x:1")

    def test_the_server_reports_v3(self, served):
        _, client = served
        assert client._state.client  # runs the health check
        assert client._state.server_health["protocol"] == FLIGHT_PROTOCOL_VERSION
