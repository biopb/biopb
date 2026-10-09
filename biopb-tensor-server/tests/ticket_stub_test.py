"""Sealed ticket stubs (biopb/biopb#1112).

A plan is issued as one stub on the descriptor plus an index per endpoint; the
stub carries a seal that reads exactly the plan's chunks, in place of a bearer
token. These cover the codec that completes a stub into the chunk_id it replaces,
the seal, and the server and proxy that issue and redeem them.
"""

import threading
import time

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb.image import ROI, Point, Polygon
from biopb.image.annotation_pb2 import RoiAnnotation
from biopb.tensor import TensorFlightClient
from biopb.tensor._session import _parse_flight_endpoints
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import (
    ChunkBounds,
    ChunkGrant,
    ChunkRef,
    RoiGrant,
    RoiRead,
    TensorTicket,
    TicketStub,
)
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.core.adapter_base import _get_read_plan
from biopb_tensor_server.core.chunk import (
    encode_proxy_identity,
    envelope_inner_is_ticket,
    expand_identity,
    identity_array_id,
    peel_proxy_envelope,
)
from biopb_tensor_server.core.ticket_seal import (
    SEAL_KEY_FILE,
    TicketSealer,
    load_seal_key,
)

from tests import catalog_server, register_and_catalog

SERVER_TOKEN = "server-secret"
SHAPE = (64, 64)


# ------------------------------------------------------------------- the codec


def _plan(shape, chunk, slice_=None, scale=None, version=None):
    base = TensorDescriptor(
        array_id="zarr_ab/Image:0",
        dim_labels=list("zyx"[-len(shape) :]),
        shape=list(shape),
        chunk_shape=list(chunk),
        dtype="<u2",
    )
    req = TensorDescriptor()
    req.CopyFrom(base)
    if slice_:
        req.slice_hint.start[:] = slice_[0]
        req.slice_hint.stop[:] = slice_[1]
    if scale:
        req.scale_hint[:] = scale
        req.reduction_method = "area"
    return _get_read_plan(base, req, tuple(chunk), version)


class TestIdentity:
    """``expand_identity(plan.identity, endpoint.index)`` is the minted chunk_id."""

    @pytest.mark.parametrize(
        "shape,chunk,slice_,scale,version",
        [
            ((10, 300, 500), (2, 64, 64), None, None, None),
            ((10, 300, 500), (2, 64, 64), ((1, 17, 100), (9, 280, 470)), None, b"1:2"),
            (
                (10, 300, 500),
                (2, 64, 64),
                ((0, 5, 5), (10, 300, 500)),
                (1, 4, 4),
                b"3:4",
            ),
            ((7, 301, 509), (3, 50, 50), ((2, 3, 7), (6, 250, 400)), (1, 2, 2), None),
        ],
    )
    def test_expansion_is_the_minted_chunk_id(
        self, shape, chunk, slice_, scale, version
    ):
        plan = _plan(shape, chunk, slice_, scale, version)
        assert plan.chunk_endpoints
        for ce in plan.chunk_endpoints:
            assert expand_identity(plan.identity, ce.index) == ce.chunk_id

    def test_the_window_is_the_plans_chunks_on_the_tensors_grid(self):
        plan = _plan((10, 300, 500), (2, 64, 64), ((1, 17, 100), (9, 280, 470)))
        start, stop = plan.window
        indices = {ce.index for ce in plan.chunk_endpoints}
        assert indices == {
            (z, y, x)
            for z in range(start[0], stop[0])
            for y in range(start[1], stop[1])
            for x in range(start[2], stop[2])
        }

    def test_one_chunk_reads_the_same_bytes_whichever_plan_reached_it(self):
        """The identity names the tensor, not the request: two windows over the
        same tensor share it, so a cache keyed on it does not fragment by slice."""
        whole = _plan((10, 300, 500), (2, 64, 64))
        part = _plan((10, 300, 500), (2, 64, 64), ((2, 70, 70), (6, 250, 400)))
        assert whole.identity == part.identity
        by_index = {ce.index: ce.chunk_id for ce in whole.chunk_endpoints}
        for ce in part.chunk_endpoints:
            assert by_index[ce.index] == ce.chunk_id

    @pytest.mark.parametrize("index", [(0,), (0, 0, 0, 0), (0, 0, 99), (-1, 0, 0)])
    def test_an_index_off_the_grid_is_refused(self, index):
        plan = _plan((10, 300, 500), (2, 64, 64))
        with pytest.raises(ValueError):
            expand_identity(plan.identity, index)

    def test_a_foreign_identity_is_refused(self):
        for junk in (b"", b"\x00junk", b"\xfd\x00"):
            with pytest.raises(ValueError):
                expand_identity(junk, (0, 0, 0))

    def test_the_array_id_comes_without_completing_it(self):
        plan = _plan((10, 300, 500), (2, 64, 64))
        assert identity_array_id(plan.identity) == "zarr_ab/Image:0"


class TestProxyIdentity:
    def test_expands_to_an_envelope_forwarding_the_upstream_ticket(self):
        up = _plan((10, 300, 500), (2, 64, 64))
        identity = encode_proxy_identity(up.identity, "lab__img", b"iat:1")
        assert identity_array_id(identity) == "lab__img"

        chunk_id = expand_identity(identity, (1, 2, 3))
        assert envelope_inner_is_ticket(chunk_id)
        route, _epoch, cv, inner = peel_proxy_envelope(chunk_id)
        assert (route, cv) == ("lab__img", b"iat:1")

        # What it forwards upstream is the upstream's identity and the index,
        # and nothing that changes with every plan.
        forwarded = TensorTicket.FromString(inner).chunk_ref
        assert forwarded.stub.identity == up.identity
        assert list(forwarded.index) == [1, 2, 3]
        assert not forwarded.stub.HasField("grant")

    def test_the_envelope_is_stable_so_it_can_key_a_cache(self):
        up = _plan((10, 300, 500), (2, 64, 64))
        identity = encode_proxy_identity(up.identity, "lab__img", None)
        assert expand_identity(identity, (0, 1, 2)) == expand_identity(
            identity, (0, 1, 2)
        )


# --------------------------------------------------------------------- the seal


class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class TestSeal:
    IDENTITY = b"\xfdan identity"

    def _sealer(self, **kw):
        return TicketSealer(b"k" * 32, clock=kw.pop("clock", _Clock()), **kw)

    def test_a_seal_covers_its_window_and_nothing_outside_it(self):
        sealer = self._sealer()
        grant = sealer.grant_chunks(self.IDENTITY, (1, 2), (3, 4))
        assert sealer.covers(self.IDENTITY, grant, (1, 2))
        assert sealer.covers(self.IDENTITY, grant, (2, 3))
        assert not sealer.covers(self.IDENTITY, grant, (3, 3))
        assert not sealer.covers(self.IDENTITY, grant, (2, 4))
        assert not sealer.covers(self.IDENTITY, grant, (0, 2))
        assert not sealer.covers(self.IDENTITY, grant, (1, 2, 0))

    def test_it_is_bound_to_the_identity(self):
        sealer = self._sealer()
        grant = sealer.grant_chunks(self.IDENTITY, (0, 0), (9, 9))
        assert not sealer.covers(b"\xfdanother", grant, (1, 1))

    def test_a_widened_window_breaks_it(self):
        sealer = self._sealer()
        grant = sealer.grant_chunks(self.IDENTITY, (0, 0), (2, 2))
        widened = ChunkGrant()
        widened.CopyFrom(grant)
        widened.window_stop[:] = [9, 9]
        assert not sealer.covers(self.IDENTITY, widened, (5, 5))

    def test_a_pushed_out_expiry_breaks_it(self):
        clock = _Clock()
        sealer = self._sealer(clock=clock, ttl=60)
        grant = sealer.grant_chunks(self.IDENTITY, (0,), (4,))
        forged = ChunkGrant()
        forged.CopyFrom(grant)
        forged.expires_at += 10**6
        clock.now += 120
        assert not sealer.covers(self.IDENTITY, forged, (1,))

    def test_it_expires(self):
        clock = _Clock()
        sealer = self._sealer(clock=clock, ttl=60)
        grant = sealer.grant_chunks(self.IDENTITY, (0,), (4,))
        assert sealer.covers(self.IDENTITY, grant, (1,))
        clock.now += 61
        assert not sealer.covers(self.IDENTITY, grant, (1,))

    def test_a_zero_ttl_never_expires(self):
        clock = _Clock()
        sealer = self._sealer(clock=clock, ttl=0)
        grant = sealer.grant_chunks(self.IDENTITY, (0,), (4,))
        clock.now += 10**9
        assert grant.expires_at == 0
        assert sealer.covers(self.IDENTITY, grant, (1,))

    def test_another_key_revokes_it(self):
        grant = self._sealer().grant_chunks(self.IDENTITY, (0,), (4,))
        other = TicketSealer(b"z" * 32, clock=_Clock())
        assert not other.covers(self.IDENTITY, grant, (1,))

    def test_an_unsealed_grant_covers_nothing(self):
        assert not self._sealer().covers(self.IDENTITY, ChunkGrant(), (0,))

    def test_a_key_too_short_to_be_one_is_refused(self):
        with pytest.raises(ValueError):
            TicketSealer(b"short")


class TestRoiSeal:
    def test_it_covers_the_tensor_it_was_made_for(self):
        sealer = TicketSealer(b"k" * 32, clock=_Clock())
        grant = sealer.grant_rois("src/img", b"v1")
        assert sealer.covers_rois("src/img", grant, b"v1")
        assert not sealer.covers_rois("src/other", grant, b"v1")

    def test_a_reused_name_does_not_inherit_it(self):
        """The tensor that now has the name is not the one the seal was made for."""
        sealer = TicketSealer(b"k" * 32, clock=_Clock())
        grant = sealer.grant_rois("scratch/@fields/x", b"v1")
        assert not sealer.covers_rois("scratch/@fields/x", grant, b"v2")
        assert not sealer.covers_rois("scratch/@fields/x", grant, None)

    def test_an_unversioned_tensor_is_sealed_as_unversioned(self):
        sealer = TicketSealer(b"k" * 32, clock=_Clock())
        grant = sealer.grant_rois("src/img", None)
        assert sealer.covers_rois("src/img", grant, None)

    def test_it_expires(self):
        clock = _Clock()
        sealer = TicketSealer(b"k" * 32, clock=clock, ttl=60)
        grant = sealer.grant_rois("src/img", b"v")
        clock.now += 61
        assert not sealer.covers_rois("src/img", grant, b"v")

    def test_an_unsealed_grant_covers_nothing(self):
        sealer = TicketSealer(b"k" * 32)
        assert not sealer.covers_rois("src/img", RoiGrant(), None)


class TestKeyFile:
    @pytest.fixture(autouse=True)
    def _state(self, tmp_path, monkeypatch):
        from biopb._config import locations

        monkeypatch.setenv(locations._TREE_ENV_STATE, str(tmp_path / "state"))
        return tmp_path / "state" / "biopb"

    def test_it_is_made_once_and_kept(self, _state):
        first = load_seal_key()
        assert len(first) >= 16
        assert (_state / SEAL_KEY_FILE).exists()
        assert load_seal_key() == first

    def test_a_restart_keeps_its_seals(self, _state):
        grant = TicketSealer(load_seal_key()).grant_chunks(b"\xfdid", (0,), (4,))
        assert TicketSealer(load_seal_key()).covers(b"\xfdid", grant, (1,))

    def test_deleting_it_revokes_every_seal(self, _state):
        grant = TicketSealer(load_seal_key()).grant_chunks(b"\xfdid", (0,), (4,))
        (_state / SEAL_KEY_FILE).unlink()
        assert not TicketSealer(load_seal_key()).covers(b"\xfdid", grant, (1,))

    def test_a_damaged_file_is_replaced_not_trusted(self, _state):
        load_seal_key()
        (_state / SEAL_KEY_FILE).write_text("not hex")
        assert len(load_seal_key()) >= 16

    def test_an_unwritable_state_tree_does_not_stop_the_server(
        self, _state, monkeypatch, caplog
    ):
        def refuse(*_a, **_k):
            raise PermissionError("read-only")

        monkeypatch.setattr("biopb._security.credentials.write_credential", refuse)
        assert len(load_seal_key()) >= 16
        assert "could not persist" in caplog.text


class TestSealTtlConfig:
    def test_it_defaults_to_the_sealers_own(self):
        from biopb_tensor_server.core.config import parse_config
        from biopb_tensor_server.core.ticket_seal import DEFAULT_SEAL_TTL

        assert parse_config({}).seal_ttl == DEFAULT_SEAL_TTL

    def test_it_is_read_from_the_server_section(self):
        from biopb_tensor_server.core.config import parse_config

        assert parse_config({"server": {"seal_ttl": 90}}).seal_ttl == 90.0
        assert parse_config({"server": {"seal_ttl": 0}}).seal_ttl == 0.0

    def test_the_server_applies_it(self):
        server = catalog_server("localhost:0", seal_ttl=600)
        try:
            grant = server._sealer.grant_chunks(b"\xfdid", (0,), (1,))
            assert 590 < grant.expires_at - time.time() <= 600
        finally:
            server.shutdown()

    def test_zero_never_expires(self):
        server = catalog_server("localhost:0", seal_ttl=0)
        try:
            assert server._sealer.grant_chunks(b"\xfdid", (0,), (1,)).expires_at == 0
        finally:
            server.shutdown()

    def test_the_cli_hands_the_configured_value_to_the_server(self, tmp_path):
        from biopb_tensor_server import cli
        from biopb_tensor_server.core.config import parse_config

        cfg = parse_config(
            {
                "server": {"seal_ttl": 600, "writable": False},
                "cache": {"file_cache_dir": str(tmp_path / "cache")},
            }
        )
        server, _, _ = cli._setup_flight_server(cfg, port=0)
        try:
            grant = server._sealer.grant_chunks(b"\xfdid", (0,), (1,))
            assert 590 < grant.expires_at - time.time() <= 600
        finally:
            server.shutdown()


# --------------------------------------------------------------- the live server


def _serve(server):
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1)


def _zarr_source(tmp_path, name="img"):
    import zarr

    data = np.arange(SHAPE[0] * SHAPE[1], dtype=np.uint8).reshape(SHAPE)
    arr = zarr.open_array(
        str(tmp_path / f"{name}.zarr"),
        mode="w",
        shape=SHAPE,
        chunks=(16, 16),
        dtype="uint8",
    )
    arr[:] = data
    return data, ZarrAdapter(
        zarr.open_array(str(tmp_path / f"{name}.zarr"), mode="r"), name, ["y", "x"]
    )


@pytest.fixture
def guarded(tmp_path, transfer_target):
    """A server that requires its token, serving a 4x4-chunk tensor."""
    transfer_target(256)  # one 16x16 uint8 block per chunk
    data, adapter = _zarr_source(tmp_path)
    server = catalog_server("localhost:0", token=SERVER_TOKEN)
    register_and_catalog(server, "img", adapter)
    server.mark_ready()
    _serve(server)
    client = TensorFlightClient(
        f"grpc://localhost:{server.port}", token=SERVER_TOKEN, cache_bytes=0
    )
    yield server, client, data
    client.close()
    server.shutdown()


def _bare(server):
    """A connection with no credential at all."""
    return flight.FlightClient(f"grpc://localhost:{server.port}")


def _stub_of(pb):
    info = flight.FlightInfo.deserialize(pb.flight_info)
    desc = TensorDescriptor.FromString(info.descriptor.command)
    return info, desc, TensorTicket.FromString(desc.ticket_stub).chunk_ref.stub


def _chunk_ticket(desc, endpoint):
    """What a client sends to read one endpoint: the two halves, joined."""
    return desc.ticket_stub + endpoint.ticket.ticket


class TestIssue:
    def test_health_says_it_issues_stubs(self, guarded):
        _, client, _ = guarded
        assert client._state.ticket_stubs is True

    def test_a_read_plan_comes_as_a_stub_and_indices(self, guarded):
        _, client, _ = guarded
        pb = client.get_tensor("img", output="pb")
        info, desc, stub = _stub_of(pb)
        assert stub.identity and stub.grant.seal
        assert len(info.endpoints) == 16
        for ep in info.endpoints:
            ref = TensorTicket.FromString(ep.ticket.ticket).chunk_ref
            assert not ref.HasField("stub")
            assert len(ref.index) == 2
            # The bounds still ride on the endpoint for a client to place it by.
            assert ChunkBounds.FromString(ep.app_metadata).stop

    def test_the_reference_leaves_without_the_connections_token(self, guarded):
        _, client, data = guarded
        pb = client.get_tensor("img", output="pb")
        assert pb.auth_token == ""
        # ...and still reads, as whoever holds it.
        out = TensorFlightClient.tensor_from_pb(pb).compute()
        np.testing.assert_array_equal(out, data)

    def test_a_sliced_reference_reads_its_slice(self, guarded):
        _, client, data = guarded
        pb = client.get_tensor("img", (slice(10, 40), slice(5, 50)), output="pb")
        assert pb.auth_token == ""
        out = TensorFlightClient.tensor_from_pb(pb).compute()
        np.testing.assert_array_equal(out, data[10:40, 5:50])

    def test_a_scaled_reference_reads_scaled(self, guarded):
        _, client, data = guarded
        pb = client.get_tensor("img", scale_hint=(2, 2), output="pb")
        assert pb.auth_token == ""
        out = TensorFlightClient.tensor_from_pb(pb).compute()
        assert out.shape == (32, 32)

    def test_the_connections_own_array_still_reads(self, guarded):
        _, client, data = guarded
        np.testing.assert_array_equal(client.get_tensor("img").compute(), data)

    def test_a_plan_not_asked_to_stub_is_issued_whole(self, guarded):
        server, _, _ = guarded
        from biopb.tensor._session import _read_option, _tensor_read_cmd

        cmd = _tensor_read_cmd("img", _read_option(endpoints=True))
        info = TensorFlightClient(
            f"grpc://localhost:{server.port}", token=SERVER_TOKEN
        )._client.get_flight_info(
            flight.FlightDescriptor.for_command(cmd.SerializeToString()),
            options=flight.FlightCallOptions(
                headers=[(b"authorization", f"Bearer {SERVER_TOKEN}".encode())]
            ),
        )
        desc = TensorDescriptor.FromString(info.descriptor.command)
        assert not desc.ticket_stub
        assert all(
            TensorTicket.FromString(ep.ticket.ticket).WhichOneof("payload")
            == "chunk_id"
            for ep in info.endpoints
        )


class TestNativeLevels:
    """A precomputed read's chunks route by their level's own array_id."""

    def test_a_native_level_reads_through_a_stub(self, tmp_path, transfer_target):
        import zarr
        from biopb_tensor_server import OmeZarrAdapter
        from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr

        zpath, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path), n_levels=3, base_shape=(256, 256), chunk_size=(64, 64)
        )
        root = zarr.open_group(zpath, mode="r")
        server = catalog_server("localhost:0", token=SERVER_TOKEN)
        register_and_catalog(server, "ome", OmeZarrAdapter(root["0"], "ome"))
        server.mark_ready()
        _serve(server)
        client = TensorFlightClient(
            f"grpc://localhost:{server.port}", token=SERVER_TOKEN, cache_bytes=0
        )
        try:
            level = np.asarray(root["1"][:])
            pb = client.get_tensor(
                "ome", scale_hint=(2, 2), reduction_method="precompute", output="pb"
            )
            assert pb.auth_token == ""
            info, desc, stub = _stub_of(pb)
            assert stub.grant.seal
            assert identity_array_id(stub.identity).endswith("/1")
            np.testing.assert_array_equal(
                TensorFlightClient.tensor_from_pb(pb).compute(), level
            )
        finally:
            client.close()
            server.shutdown()


class TestNativeLevelsThroughAMirror:
    """A mirror routes a native level's chunks to its own adapter and forwards
    them verbatim, so the level's own ``array_id`` need never be read."""

    @pytest.fixture
    def mirrored_ome(self, tmp_path):
        import zarr
        from biopb_tensor_server import OmeZarrAdapter
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
        from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr

        zpath, _, _ = create_multiresolution_ome_zarr(
            str(tmp_path), n_levels=3, base_shape=(256, 256), chunk_size=(64, 64)
        )
        root = zarr.open_group(zpath, mode="r")
        upstream = catalog_server("localhost:0")
        register_and_catalog(upstream, "ome", OmeZarrAdapter(root["0"], "ome"))
        upstream.mark_ready()
        _serve(upstream)
        proxy = catalog_server("localhost:0")
        register_and_catalog(
            proxy,
            "lab__ome",
            RemoteTensorAdapter(
                source_id="lab__ome",
                upstream_location=f"grpc://localhost:{upstream.port}",
                upstream_source_id="ome",
            ),
        )
        proxy.mark_ready()
        _serve(proxy)
        client = TensorFlightClient(f"grpc://localhost:{proxy.port}", cache_bytes=0)
        yield upstream, client, np.asarray(root["1"][:])
        client.close()
        proxy.shutdown()
        upstream.shutdown()

    def _read(self, client):
        return client.get_tensor(
            "lab__ome", scale_hint=(2, 2), reduction_method="precompute"
        ).compute()

    def test_by_stub(self, mirrored_ome):
        _, client, level = mirrored_ome
        assert client._state.ticket_stubs
        np.testing.assert_array_equal(self._read(client), level)

    def test_by_full_ticket(self, mirrored_ome):
        _, client, level = mirrored_ome
        assert client._state.client
        client._state.server_health["ticket_stubs"] = False
        np.testing.assert_array_equal(self._read(client), level)

    def test_the_reference_is_sealed(self, mirrored_ome):
        _, client, level = mirrored_ome
        pb = client.get_tensor(
            "lab__ome", scale_hint=(2, 2), reduction_method="precompute", output="pb"
        )
        _, _, stub = _stub_of(pb)
        assert pb.auth_token == "" and stub.grant.seal
        np.testing.assert_array_equal(
            TensorFlightClient.tensor_from_pb(pb).compute(), level
        )


class TestRedeem:
    def _first(self, guarded):
        _, client, _ = guarded
        info, desc, stub = _stub_of(client.get_tensor("img", output="pb"))
        return info, desc, stub

    def test_the_seal_reads_a_chunk_with_no_token(self, guarded):
        server, _, data = guarded
        info, desc, _ = self._first(guarded)
        ep = info.endpoints[0]
        bounds = ChunkBounds.FromString(ep.app_metadata)
        table = _bare(server).do_get(flight.Ticket(_chunk_ticket(desc, ep))).read_all()
        assert table.num_rows == 1
        raw = table.column("data")[0].as_py()
        got = np.frombuffer(raw, dtype=np.uint8)
        expected = data[
            bounds.start[0] : bounds.stop[0], bounds.start[1] : bounds.stop[1]
        ]
        np.testing.assert_array_equal(got.reshape(expected.shape), expected)

    def test_the_same_chunk_without_a_seal_is_refused(self, guarded):
        server, _, _ = guarded
        info, desc, stub = self._first(guarded)
        bare_stub = TensorTicket(
            chunk_ref=ChunkRef(stub=TicketStub(identity=stub.identity))
        ).SerializeToString()
        with pytest.raises(flight.FlightUnauthenticatedError):
            _bare(server).do_get(
                flight.Ticket(bare_stub + info.endpoints[0].ticket.ticket)
            ).read_all()

    def test_the_plan_does_not_read_beyond_its_window(self, guarded):
        server, client, _ = guarded
        pb = client.get_tensor("img", (slice(0, 16), slice(0, 16)), output="pb")
        info, desc, stub = _stub_of(pb)
        assert len(info.endpoints) == 1
        outside = TensorTicket(chunk_ref=ChunkRef(index=[3, 3])).SerializeToString()
        with pytest.raises(flight.FlightUnauthenticatedError):
            _bare(server).do_get(flight.Ticket(desc.ticket_stub + outside)).read_all()

    def test_a_tampered_seal_is_refused(self, guarded):
        server, _, _ = guarded
        info, _, stub = self._first(guarded)
        forged = ChunkGrant()
        forged.CopyFrom(stub.grant)
        forged.window_stop[:] = [99, 99]
        ticket = (
            TensorTicket(
                chunk_ref=ChunkRef(
                    stub=TicketStub(identity=stub.identity, grant=forged)
                )
            ).SerializeToString()
            + info.endpoints[0].ticket.ticket
        )
        with pytest.raises(flight.FlightUnauthenticatedError):
            _bare(server).do_get(flight.Ticket(ticket)).read_all()

    def test_a_seal_expires(self, guarded):
        server, _, _ = guarded
        info, desc, _ = self._first(guarded)
        server._sealer._clock = lambda: time.time() + 10 * 24 * 3600
        with pytest.raises(flight.FlightUnauthenticatedError, match="expired"):
            _bare(server).do_get(
                flight.Ticket(_chunk_ticket(desc, info.endpoints[0]))
            ).read_all()

    def test_a_new_key_revokes_every_seal_outstanding(self, guarded):
        server, _, _ = guarded
        info, desc, _ = self._first(guarded)
        server._sealer = TicketSealer(b"n" * 32)
        with pytest.raises(flight.FlightUnauthenticatedError):
            _bare(server).do_get(
                flight.Ticket(_chunk_ticket(desc, info.endpoints[0]))
            ).read_all()

    def test_the_bearer_token_still_reads_a_stale_seal(self, guarded):
        """Full access is checked first, so an expired seal costs a token holder
        nothing."""
        server, _, _ = guarded
        info, desc, _ = self._first(guarded)
        server._sealer._clock = lambda: time.time() + 10 * 24 * 3600
        authed = flight.FlightClient(f"grpc://localhost:{server.port}")
        opts = flight.FlightCallOptions(
            headers=[(b"authorization", f"Bearer {SERVER_TOKEN}".encode())]
        )
        table = authed.do_get(
            flight.Ticket(_chunk_ticket(desc, info.endpoints[0])), options=opts
        ).read_all()
        assert table.num_rows == 1

    def test_a_seal_does_not_open_actions_or_the_catalog(self, guarded):
        server, _, _ = guarded
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(_bare(server).do_action(flight.Action("cache_stats", b"")))

    def test_a_seal_does_not_plan(self, guarded):
        """Planning is the capability's, not the seal's: a holder cannot ask for
        a plan at another slice."""
        server, _, _ = guarded
        from biopb.tensor._session import _read_option, _tensor_read_cmd

        cmd = _tensor_read_cmd("img", _read_option(endpoints=True))
        with pytest.raises(flight.FlightUnauthenticatedError):
            _bare(server).get_flight_info(
                flight.FlightDescriptor.for_command(cmd.SerializeToString())
            )

    def test_chunk_locate_takes_the_seal_too(self, guarded):
        server, _, _ = guarded
        info, desc, _ = self._first(guarded)
        results = list(
            _bare(server).do_action(
                flight.Action("chunk_locate", _chunk_ticket(desc, info.endpoints[0]))
            )
        )
        assert results  # an answer, not a refusal ("available" may be false)


class TestClientKeys:
    def test_a_chunk_keys_the_same_in_every_plan(self, guarded):
        """The client caches by key, and a grant changes with each plan: the key
        carries the identity and the index and never the grant."""
        _, client, _ = guarded
        a = client.get_tensor("img", (slice(0, 48), slice(0, 48)), output="pb")
        b = client.get_tensor("img", (slice(16, 64), slice(16, 64)), output="pb")
        keys = {}
        for pb in (a, b):
            info = flight.FlightInfo.deserialize(pb.flight_info)
            chunks, bounds, grant = _parse_flight_endpoints(info)
            assert grant is not None
            keys[id(pb)] = {
                (tuple(bd.start), tuple(bd.stop)): key
                for key, bd in zip(chunks, bounds, strict=True)
            }
        shared = set(keys[id(a)]) & set(keys[id(b)])
        assert shared
        # Bounds are plan-relative, so match on the key itself.
        assert set(keys[id(a)].values()) & set(keys[id(b)].values())

    def test_the_grants_differ_but_the_keys_do_not(self, guarded):
        _, client, _ = guarded
        infos = [
            flight.FlightInfo.deserialize(
                client.get_tensor("img", output="pb").flight_info
            )
            for _ in range(2)
        ]
        first = _parse_flight_endpoints(infos[0])
        second = _parse_flight_endpoints(infos[1])
        assert sorted(first[0]) == sorted(second[0])


class TestSealedRois:
    def _put(self, client):
        client.put_rois(
            "img",
            [
                RoiAnnotation(
                    label="nucleus",
                    roi=ROI(
                        polygon=Polygon(
                            points=[Point(x=1, y=2), Point(x=10, y=2), Point(x=5, y=9)]
                        )
                    ),
                )
            ],
        )

    def test_a_reference_reads_its_tensors_annotations(self, guarded):
        server, client, _ = guarded
        self._put(client)
        pb = client.get_tensor("img", output="pb")
        ticket = TensorFlightClient.roi_ticket_from_pb(pb)
        assert ticket

        tokenless = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=0)
        got = tokenless.list_rois("img", roi_ticket=ticket)
        assert [r.label for r in got.rois] == ["nucleus"]
        tokenless.close()

    def test_without_it_the_annotations_stay_private(self, guarded):
        server, client, _ = guarded
        self._put(client)
        tokenless = TensorFlightClient(f"grpc://localhost:{server.port}", cache_bytes=0)
        with pytest.raises(flight.FlightUnauthenticatedError):
            tokenless.list_rois("img")
        tokenless.close()

    def test_the_ticket_does_not_read_another_tensor(self, guarded):
        server, client, _ = guarded
        pb = client.get_tensor("img", output="pb")
        ticket = TensorFlightClient.roi_ticket_from_pb(pb)
        grant = TensorTicket.FromString(ticket).roi_read.grant
        forged = TensorTicket(
            roi_read=RoiRead(array_id="someone_else", grant=grant)
        ).SerializeToString()
        with pytest.raises(flight.FlightUnauthenticatedError):
            _bare(server).do_get(flight.Ticket(forged)).read_all()

    def test_a_replaced_tensor_does_not_inherit_the_grant(self, guarded):
        server, client, _ = guarded
        pb = client.get_tensor("img", output="pb")
        ticket = TensorFlightClient.roi_ticket_from_pb(pb)
        adapter = server.sources.get_registered("img")
        adapter._content_version = b"a-different-tensor"
        try:
            with pytest.raises(flight.FlightUnauthenticatedError):
                _bare(server).do_get(flight.Ticket(ticket)).read_all()
        finally:
            adapter._content_version = None


# ------------------------------------------------------------------ the mirror


class TestMirror:
    @pytest.fixture
    def mirrored(self, tmp_path, transfer_target):
        """A tokened upstream, mirrored by a tokened proxy."""
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        transfer_target(256)
        data, adapter = _zarr_source(tmp_path)
        upstream = catalog_server("localhost:0", token="upstream-secret")
        register_and_catalog(upstream, "img", adapter)
        upstream.mark_ready()
        _serve(upstream)

        proxy = catalog_server("localhost:0", token=SERVER_TOKEN)
        register_and_catalog(
            proxy,
            "lab__img",
            RemoteTensorAdapter(
                source_id="lab__img",
                upstream_location=f"grpc://localhost:{upstream.port}",
                upstream_source_id="img",
                token="upstream-secret",
            ),
        )
        proxy.mark_ready()
        _serve(proxy)
        client = TensorFlightClient(
            f"grpc://localhost:{proxy.port}", token=SERVER_TOKEN, cache_bytes=0
        )
        yield upstream, proxy, client, data
        client.close()
        proxy.shutdown()
        upstream.shutdown()

    def test_the_proxy_issues_a_stub_over_the_upstreams(self, mirrored):
        _, proxy, client, _ = mirrored
        pb = client.get_tensor("lab__img", output="pb")
        info, desc, stub = _stub_of(pb)
        assert pb.auth_token == ""
        assert stub.grant.seal
        # Sealed by the proxy's key, so the upstream's cannot be what it holds.
        assert identity_array_id(stub.identity) == "lab__img"
        assert len(info.endpoints) == 16

    def test_a_tokenless_holder_reads_through_the_proxy(self, mirrored):
        _, _, client, data = mirrored
        pb = client.get_tensor("lab__img", output="pb")
        np.testing.assert_array_equal(
            TensorFlightClient.tensor_from_pb(pb).compute(), data
        )

    def test_a_sliced_read_through_the_proxy(self, mirrored):
        _, _, client, data = mirrored
        pb = client.get_tensor("lab__img", (slice(5, 33), slice(20, 60)), output="pb")
        np.testing.assert_array_equal(
            TensorFlightClient.tensor_from_pb(pb).compute(), data[5:33, 20:60]
        )

    def test_the_proxys_cache_key_is_its_own_plan_independent_envelope(self, mirrored):
        _, _, client, _ = mirrored
        a = client.get_tensor("lab__img", (slice(0, 48), slice(0, 48)), output="pb")
        b = client.get_tensor("lab__img", (slice(16, 64), slice(16, 64)), output="pb")
        ka = set(
            _parse_flight_endpoints(flight.FlightInfo.deserialize(a.flight_info))[0]
        )
        kb = set(
            _parse_flight_endpoints(flight.FlightInfo.deserialize(b.flight_info))[0]
        )
        assert ka & kb

    def test_the_upstream_is_asked_by_identity_and_index(self, mirrored, monkeypatch):
        """Not a full ticket per chunk: the proxy forwards the upstream's own
        stub, unsealed, and the upstream redeems it on the proxy's token."""
        upstream, _, client, data = mirrored
        arms = []
        real = upstream._open_chunk_ticket

        def spy(context, ticket):
            arms.append(ticket.WhichOneof("payload"))
            return real(context, ticket)

        monkeypatch.setattr(upstream, "_open_chunk_ticket", spy)
        pb = client.get_tensor("lab__img", output="pb")
        TensorFlightClient.tensor_from_pb(pb).compute()
        assert arms and set(arms) == {"chunk_ref"}

    def test_the_connection_still_reads_its_own_array(self, mirrored):
        _, _, client, data = mirrored
        np.testing.assert_array_equal(client.get_tensor("lab__img").compute(), data)
