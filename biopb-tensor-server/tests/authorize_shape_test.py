"""Who may reach what: full access, and the narrow grant beside it.

The model (biopb/biopb#1048) is the one the project started with -- a single
server token meaning full access -- plus one exception: a capability token
granting a holder with no server token access to *one* object, for *reads*.

Two properties carry the whole thing, and both are reversals of what the code
did before:

- the server token is checked **first** and opens everything, so a capability
  *adds* access rather than replacing it;
- a capability covers reads only, so writes, ``resolve`` and ``warm`` take full
  access whatever grant the tensor carries.

A grant sits on a *tensor*, never on the source it hangs off: one source is
shared by uploads with different producers.
"""

import json

import pyarrow.flight as flight
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket
from biopb_tensor_server.core.chunk import encode_chunk_id
from biopb_tensor_server.serving.server import (
    READ_ANNOTATIONS,
    READ_PIXELS,
    TensorFlightServer,
)

SERVER_TOKEN = "server-token"
CAPABILITY = "capability-token"


class _Adapter:
    """Minimal source double: an id, and the grants its tensors carry.

    *tensor_tokens* is keyed by full ``array_id`` -- what
    ``SourceAdapter.tensor_capability_token`` answers off its attachment index.
    A source has no token of its own to offer, which is the point; the one set
    in :class:`TestASourceCannotGateWhatIsAttachedToIt` is there to show it is
    never consulted.
    """

    source_type = "zarr"

    def __init__(self, source_id, tensor_tokens=None):
        self.source_id = source_id
        self._tensor_tokens = dict(tensor_tokens or {})

    def tensor_capability_token(self, array_id):
        return self._tensor_tokens.get(array_id)


class _Middleware:
    def __init__(self, token):
        self.token = token


class _Context:
    """A call context carrying a presented Bearer token, or none."""

    def __init__(self, token):
        self._mw = _Middleware(token)

    def get_middleware(self, key):
        return self._mw if key == "auth" else None


#: A grant on one *tensor* of an otherwise ungated source. The uploaded-
#: intermediate shape: many results share one source and each has its own
#: producer, so the grant cannot sit on the source they share.
TENSOR_CAPABILITY = "tensor-capability-token"

#: The gated object throughout: one tensor of a source that gates nothing else.
GATED = "gated/@fields/result"


def _server(token):
    server = TensorFlightServer("grpc://localhost:0", token=token)
    server.sources.register("open", _Adapter("open"))
    server.sources.register(
        "gated", _Adapter("gated", tensor_tokens={GATED: CAPABILITY})
    )
    server.sources.register(
        "shared",
        _Adapter(
            "shared",
            tensor_tokens={"shared/@fields/mine": TENSOR_CAPABILITY},
        ),
    )
    return server


@pytest.fixture
def guarded():
    """A server with a server-wide token configured (the remote-mode shape)."""
    server = _server(SERVER_TOKEN)
    yield server
    server.sources.close_all()


@pytest.fixture
def local():
    """A server with no server-wide token (local mode, and the embedded cache)."""
    server = _server(None)
    yield server
    server.sources.close_all()


class TestFullAccess:
    """``_authorize``: the server token, or local mode."""

    def test_the_server_token_passes(self, guarded):
        guarded._authorize(_Context(SERVER_TOKEN))

    def test_a_wrong_token_is_refused(self, guarded):
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize(_Context("nope"))

    def test_no_token_is_refused(self, guarded):
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize(_Context(None))

    def test_a_capability_does_not_reach_it(self, guarded):
        """Actions are the control surface, and a capability is not a key to it.

        The concrete case is ``warm``: a grant meaning "read this one tensor"
        must not authorize an operation whose cost lands on every other source
        (biopb/biopb#1043).
        """
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize(_Context(CAPABILITY))

    def test_local_mode_is_open(self, local):
        local._authorize(_Context(None))


class TestNarrowGrant:
    """``_authorize_read``: full access, or a capability covering this read."""

    def test_the_capability_opens_its_own_tensor(self, guarded):
        guarded._authorize_read(_Context(CAPABILITY), GATED, READ_PIXELS)

    def test_it_covers_annotations_too(self, guarded):
        """One token, both reads -- today. The actions are named so that can
        stop being true without touching a call site."""
        guarded._authorize_read(_Context(CAPABILITY), GATED, READ_ANNOTATIONS)

    def test_the_capability_does_not_open_another_source(self, guarded):
        """A grant names an object. ``open`` carries none, so the ordinary rule
        applies -- and a capability is not the server token."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context(CAPABILITY), "open", READ_PIXELS)

    def test_the_server_token_opens_a_gated_tensor(self, guarded):
        """The server token is checked first, so presenting it to a
        capability-bearing tensor opens it rather than being refused as the
        wrong capability -- the operator is never locked out of their own
        server."""
        guarded._authorize_read(_Context(SERVER_TOKEN), GATED, READ_PIXELS)

    def test_a_source_without_a_grant_follows_the_ordinary_rule(self, guarded):
        guarded._authorize_read(_Context(SERVER_TOKEN), "open", READ_PIXELS)
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context(None), "open", READ_PIXELS)

    def test_an_unknown_source_is_not_a_backdoor(self, guarded):
        """No adapter means no grant, which means the ordinary rule -- not an
        exemption. Registration races must not read as permission."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context(None), "missing", READ_PIXELS)

    def test_a_wrong_capability_is_refused_not_fallen_through(self, guarded):
        """Presenting the wrong token for a gated tensor is a refusal, not a
        miss that then consults the server-wide rule."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context("wrong"), GATED, READ_PIXELS)


class TestLocalMode:
    """Local mode removes the server-wide gate, not every gate."""

    def test_a_gated_tensor_stays_gated(self, local):
        """The embedded result cache's whole model: it mints capabilities on a
        server with no server-wide token, so if local mode opened everything
        the capability would mean nothing.
        """
        with pytest.raises(flight.FlightUnauthenticatedError):
            local._authorize_read(_Context(None), GATED, READ_PIXELS)

    def test_the_capability_still_opens_it(self, local):
        local._authorize_read(_Context(CAPABILITY), GATED, READ_PIXELS)

    def test_an_ungated_source_is_open(self, local):
        local._authorize_read(_Context(None), "open", READ_PIXELS)


class TestATensorCarriesItsOwnGrant:
    """The half a shared source needs: a grant that covers one tensor of it.

    An uploaded intermediate has a producer, and the source it lands on is
    shared with every other upload. A grant on the source would open all of
    them, so the tensor carries its own.
    """

    def test_it_opens_the_tensor_it_names(self, guarded):
        guarded._authorize_read(
            _Context(TENSOR_CAPABILITY), "shared/@fields/mine", READ_PIXELS
        )

    def test_it_does_not_open_a_sibling(self, guarded):
        """The whole point. Both tensors live on one source, so a grant that
        leaked across them would be a source-level grant wearing a tensor's
        name."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(
                _Context(TENSOR_CAPABILITY), "shared/@fields/theirs", READ_PIXELS
            )

    def test_it_does_not_open_the_source_itself(self, guarded):
        """A bare source_id names the source's default tensor, which is not the
        granted one."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context(TENSOR_CAPABILITY), "shared", READ_PIXELS)

    def test_the_server_token_still_opens_it(self, guarded):
        """A capability adds access; it never takes the operator's away."""
        guarded._authorize_read(
            _Context(SERVER_TOKEN), "shared/@fields/mine", READ_PIXELS
        )

    def test_an_ungated_sibling_follows_the_ordinary_rule(self, guarded):
        """One gated tensor does not gate the source: its siblings are as open
        as the source is, which is what keeps a shared scratch source usable."""
        guarded._authorize_read(
            _Context(SERVER_TOKEN), "shared/@fields/theirs", READ_PIXELS
        )
        assert (
            guarded._grants(SERVER_TOKEN, READ_PIXELS, "shared/@fields/theirs") is None
        )

    def test_a_sibling_of_the_gated_tensor_is_not_gated(self, guarded):
        """A grant names one tensor, so the source it hangs off keeps whatever
        rule it had -- here the ordinary one."""
        assert guarded._grants(SERVER_TOKEN, READ_PIXELS, "gated/0") is None


class TestASourceCannotGateWhatIsAttachedToIt:
    """Only a tensor carries a grant, and only the attachment index is read.

    A source is shared -- every uploaded result lands on one scratch source --
    so a token at source scope would open every producer's result to whoever
    holds one of them. Nothing consults it, which is why the field it would
    live in is declared on ``TensorAdapter`` and not on ``SourceAdapter``.
    """

    def test_a_token_set_on_the_source_gates_nothing(self, guarded):
        adapter = guarded.sources.get("open")
        adapter.capability_token = "source-token"

        assert guarded._grants("source-token", READ_PIXELS, "open") is None
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context("source-token"), "open", READ_PIXELS)

    def test_and_the_server_token_still_reads_it(self, guarded):
        """The other half: a stray source token does not lock the operator out
        of a source that is otherwise open."""
        guarded.sources.get("open").capability_token = "source-token"

        guarded._authorize_read(_Context(SERVER_TOKEN), "open", READ_PIXELS)


class TestChunkLocateIsARead:
    """The localhost fast path is gated like the read it replaces.

    ``chunk_locate`` hands back a path and a byte offset that the client then
    mmaps, so it *is* the read -- nothing downstream can gate it. It sits among
    the actions only because Flight has no other verb for handing back a byte
    range, and treating it as control was wrong twice over: a capability holder
    was refused the fast path on every chunk, and on a server with no
    server-wide token ``_authorize`` let anyone through.
    """

    @staticmethod
    def _locate(server, token, array_id):
        ticket = TensorTicket(
            chunk_id=encode_chunk_id(array_id, ChunkBounds(start=[0], stop=[1]))
        )
        action = flight.Action("chunk_locate", ticket.SerializeToString())
        return list(server.do_action(_Context(token), action))

    def test_a_stranger_is_refused(self, guarded):
        with pytest.raises(flight.FlightUnauthenticatedError):
            self._locate(guarded, None, GATED)

    def test_the_wrong_token_is_refused(self, guarded):
        with pytest.raises(flight.FlightUnauthenticatedError):
            self._locate(guarded, "wrong", GATED)

    def test_the_capability_holder_gets_past_the_gate(self, guarded):
        """Past the gate is the assertion: the chunk is not cached here, so the
        answer is an ordinary miss the client falls back from. Before this it
        was a refusal, paid on every chunk."""
        body = self._locate(guarded, CAPABILITY, GATED)

        assert json.loads(bytes(body[0])) == {"available": False}

    def test_a_grant_on_one_tensor_does_not_locate_a_sibling(self, guarded):
        with pytest.raises(flight.FlightUnauthenticatedError):
            self._locate(guarded, CAPABILITY, "gated/0")


class TestHealthIsUngated:
    """The liveness probe answers anyone, the way an HTTP server answers
    ``/healthz``.

    A caller that cannot yet authenticate still has to be able to ask: an
    orchestrator, a readiness gate, an SDK checking the protocol before its
    first call. What it reports is status, protocol and capability flags --
    never a source name or a path.
    """

    def _health(self, server, token):
        action = flight.Action("health", b"")
        body = next(iter(server.do_action(_Context(token), action)))
        return json.loads(bytes(body))

    def test_a_server_with_a_token_still_answers_a_stranger(self, guarded):
        assert self._health(guarded, None)["status"] in ("SERVING", "STARTING")

    def test_it_carries_the_protocol_a_client_checks_before_authenticating(
        self, guarded
    ):
        """What the SDK reads to refuse a version-skewed server; gating it made
        that check unreachable exactly when it is most useful."""
        assert self._health(guarded, None)["protocol"] >= 2

    def test_every_other_action_still_takes_full_access(self, guarded):
        """Ungating health is not a crack in the rule it is an exception to."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(guarded.do_action(_Context(None), flight.Action("cache_stats", b"")))


class TestGrantsSeam:
    """``_grants`` is where a grant table or a signed token would land
    (biopb/biopb#1048), so its three-valued answer is the contract."""

    def test_no_grant_on_the_source_is_none_not_false(self, guarded):
        """``None`` means "this object opted into nothing", which the caller
        turns into the ordinary rule. Collapsing it to ``False`` would make
        every ungated source unreadable."""
        assert guarded._grants(SERVER_TOKEN, READ_PIXELS, "open") is None

    def test_a_matching_grant_is_true(self, guarded):
        assert guarded._grants(CAPABILITY, READ_PIXELS, GATED) is True

    def test_a_mismatched_token_is_false(self, guarded):
        assert guarded._grants("wrong", READ_PIXELS, GATED) is False

    def test_an_action_outside_the_grant_is_false(self, guarded):
        """The right token for the right tensor, asked about something it does
        not cover. This is what keeps `warm` out even if a call site ever asked
        `_authorize_read` about it."""
        assert guarded._grants(CAPABILITY, "warm", GATED) is False
