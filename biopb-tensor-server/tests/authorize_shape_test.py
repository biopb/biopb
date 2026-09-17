"""Who may reach what: full access, and the narrow grant beside it.

The model (biopb/biopb#1048) is the one the project started with -- a single
server token meaning full access -- plus one exception: a capability token
granting a holder with no server token access to *one* object, for *reads*.

Two properties carry the whole thing, and both are reversals of what the code
did before:

- the server token is checked **first** and opens everything, so a capability
  *adds* access rather than replacing it;
- a capability covers reads only, so writes, ``resolve`` and ``warm`` take full
  access whatever token the source carries.
"""

import pyarrow.flight as flight
import pytest
from biopb_tensor_server.serving.server import (
    READ_ANNOTATIONS,
    READ_PIXELS,
    TensorFlightServer,
)

SERVER_TOKEN = "server-token"
CAPABILITY = "capability-token"


class _Adapter:
    """Minimal source double: an id and whether it carries a grant."""

    source_type = "zarr"

    def __init__(self, source_id, capability_token=None):
        self.source_id = source_id
        self.capability_token = capability_token


class _Middleware:
    def __init__(self, token):
        self.token = token


class _Context:
    """A call context carrying a presented Bearer token, or none."""

    def __init__(self, token):
        self._mw = _Middleware(token)

    def get_middleware(self, key):
        return self._mw if key == "auth" else None


def _server(token):
    server = TensorFlightServer("grpc://localhost:0", token=token)
    server.sources.register("open", _Adapter("open"))
    server.sources.register("gated", _Adapter("gated", capability_token=CAPABILITY))
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

    def test_the_capability_opens_its_own_source(self, guarded):
        guarded._authorize_read(_Context(CAPABILITY), "gated", READ_PIXELS)

    def test_it_covers_annotations_too(self, guarded):
        """One token, both reads -- today. The actions are named so that can
        stop being true without touching a call site."""
        guarded._authorize_read(_Context(CAPABILITY), "gated", READ_ANNOTATIONS)

    def test_the_capability_does_not_open_another_source(self, guarded):
        """A grant names an object. ``open`` carries none, so the ordinary rule
        applies -- and a capability is not the server token."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context(CAPABILITY), "open", READ_PIXELS)

    def test_the_server_token_opens_a_gated_source(self, guarded):
        """The reversal. The capability used to be checked first and returned,
        so presenting the server token to a capability-bearing source was a
        refusal -- the operator locked out of their own server."""
        guarded._authorize_read(_Context(SERVER_TOKEN), "gated", READ_PIXELS)

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
        """Presenting the wrong token for a gated source is a refusal, not a
        miss that then consults the server-wide rule."""
        with pytest.raises(flight.FlightUnauthenticatedError):
            guarded._authorize_read(_Context("wrong"), "gated", READ_PIXELS)


class TestLocalMode:
    """Local mode removes the server-wide gate, not every gate."""

    def test_a_gated_source_stays_gated(self, local):
        """The embedded result cache's whole model: it mints capabilities on a
        server with no server-wide token, so if local mode opened everything
        the capability would mean nothing.
        """
        with pytest.raises(flight.FlightUnauthenticatedError):
            local._authorize_read(_Context(None), "gated", READ_PIXELS)

    def test_the_capability_still_opens_it(self, local):
        local._authorize_read(_Context(CAPABILITY), "gated", READ_PIXELS)

    def test_an_ungated_source_is_open(self, local):
        local._authorize_read(_Context(None), "open", READ_PIXELS)


class TestGrantsSeam:
    """``_grants`` is where a grant table or a signed token would land
    (biopb/biopb#1048), so its three-valued answer is the contract."""

    def test_no_grant_on_the_source_is_none_not_false(self, guarded):
        """``None`` means "this object opted into nothing", which the caller
        turns into the ordinary rule. Collapsing it to ``False`` would make
        every ungated source unreadable."""
        assert guarded._grants(SERVER_TOKEN, READ_PIXELS, "open") is None

    def test_a_matching_grant_is_true(self, guarded):
        assert guarded._grants(CAPABILITY, READ_PIXELS, "gated") is True

    def test_a_mismatched_token_is_false(self, guarded):
        assert guarded._grants("wrong", READ_PIXELS, "gated") is False

    def test_an_action_outside_the_grant_is_false(self, guarded):
        """The right token for the right source, asked about something it does
        not cover. This is what keeps `warm` out even if a call site ever asked
        `_authorize_read` about it."""
        assert guarded._grants(CAPABILITY, "warm", "gated") is False
