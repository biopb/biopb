"""Tests for the shared endpoint-header policy (_endpoint.py).

The point of the module is that both OpenAI-compatible clients in this package
read one answer, so these test the answer; the wiring tests live beside each
client (`test_mcp_model.py`, `agentbench/test_models.py`).
"""

from biopb_mcp import _endpoint

OPENCODE = "https://opencode.ai/zen/go/v1"


class TestKnownGateways:
    def test_a_known_gateway_is_configured_without_being_asked(self):
        # The whole point of the table: a user who has never heard of the
        # header still sends it.
        assert _endpoint.extra_headers(OPENCODE, session="s-1") == {
            "x-opencode-session": "s-1"
        }

    def test_a_subdomain_is_the_same_gateway(self):
        assert _endpoint.extra_headers(
            "https://eu.opencode.ai/zen/v1", session="s-1"
        ) == {"x-opencode-session": "s-1"}

    def test_an_unknown_endpoint_gets_nothing(self):
        # Sending a vendor's header to a vendor that did not ask for it is not
        # harmless -- it is a session id handed to someone else.
        assert _endpoint.extra_headers("https://api.openai.com/v1", session="s-1") == {}
        assert _endpoint.extra_headers("http://localhost:11434/v1", session="s-1") == {}

    def test_a_lookalike_host_is_not_the_gateway(self):
        # Suffix matching is on a dot boundary, so a host that merely starts
        # with the gateway's name is not handed its session id.
        assert (
            _endpoint.extra_headers("https://opencode.ai.evil.test/v1", session="s")
            == {}
        )


class TestConfiguredHeaders:
    def test_a_user_can_name_a_gateway_we_have_never_heard_of(self):
        assert _endpoint.extra_headers(
            "https://gw.test/v1",
            ["X-Trace-Session: {session}", "X-Team: imaging"],
            session="s-2",
        ) == {"X-Trace-Session": "s-2", "X-Team": "imaging"}

    def test_configured_overrides_the_default_for_the_same_header(self):
        # Case-insensitively, or the override lands beside the header it meant
        # to replace and both go on the wire.
        assert _endpoint.extra_headers(
            OPENCODE, ["X-OpenCode-Session: fixed-id"], session="s-3"
        ) == {"X-OpenCode-Session": "fixed-id"}

    def test_an_empty_value_removes_a_default(self):
        # The escape hatch: a default that starts doing harm can be turned off
        # from the config, without waiting for a release that drops the row.
        assert (
            _endpoint.extra_headers(OPENCODE, ["x-opencode-session:"], session="s")
            == {}
        )

    def test_a_malformed_line_is_skipped_not_raised(self):
        # Hand-edited config. One bad line must not take the chat pane down.
        assert _endpoint.extra_headers(
            "https://gw.test/v1", ["not a header", "", "X-Ok: 1"], session="s"
        ) == {"X-Ok": "1"}


class TestSession:
    def test_a_header_with_no_session_to_name_is_dropped(self):
        # Blank is worse than absent: a gateway checking for the header sees it
        # and reads an empty conversation id.
        assert _endpoint.extra_headers(OPENCODE, session="") == {}

    def test_ids_are_unique_and_carry_nothing_about_the_session(self):
        ids = {_endpoint.new_session_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(i.startswith("biopb-") for i in ids)


class TestEnvHeaders:
    def test_newline_separated_because_a_value_may_contain_anything_else(self):
        assert _endpoint.parse_env_headers(
            "X-A: 1\n  X-B: Mon, 01 Jan 2026 00:00:00 GMT  \n\n"
        ) == ("X-A: 1", "X-B: Mon, 01 Jan 2026 00:00:00 GMT")

    def test_unset_is_no_headers(self):
        assert _endpoint.parse_env_headers("") == ()
