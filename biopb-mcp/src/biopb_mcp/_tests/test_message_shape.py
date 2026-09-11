"""Tests for the shared conversation-shape policy (_message_shape.py).

The point of the module is that both clients report a rejection the same way,
so these test the answer; the wiring tests live beside each client
(`test_mcp_model.py`, `agentbench/test_conversation.py`).
"""

from biopb_mcp._message_shape import describe_message, describe_messages


class TestWhatItShows:
    def test_role_keys_and_tool_call_count(self):
        line = describe_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a"}, {"id": "b"}],
                "reasoning_content": "...",
            }
        )
        assert line == (
            "assistant keys=[content,reasoning_content,tool_calls] tool_calls=2"
        )

    def test_a_message_without_tool_calls_does_not_say_zero(self):
        assert describe_message({"role": "user", "content": "go"}) == (
            "user      keys=[content]"
        )

    def test_only_the_tail_of_a_long_thread(self):
        # The provider's own words sit above this; a hundred lines of shape
        # would bury them.
        many = [{"role": "user", "content": str(i)} for i in range(25)]
        out = describe_messages(many)
        assert "last 10 of 25 messages sent" in out
        assert len(out.splitlines()) == 11


class TestWhatItNeverShows:
    def test_content_never_appears(self):
        # The line that makes the shape safe to put in front of a user.
        out = describe_messages(
            [
                {"role": "user", "content": "my unpublished experiment"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"arguments": '{"path": "patient_07.nd2"}'}}
                    ],
                },
            ]
        )
        assert "my unpublished experiment" not in out
        assert "patient_07.nd2" not in out
        assert "keys=[content]" in out and "tool_calls=1" in out

    def test_an_image_payload_shows_as_a_key(self):
        out = describe_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;b"}}
                    ],
                }
            ]
        )
        assert "data:image" not in out
        assert "keys=[content]" in out


class TestItNeverRaises:
    """It runs while an error is being built, so a surprise in the payload
    must not replace the provider's words with a traceback from the code
    describing them."""

    def test_an_empty_thread_says_so(self):
        assert describe_messages([]) == "  no messages were sent."

    def test_a_message_that_is_not_a_mapping(self):
        assert "<str>" in describe_messages(["junk"])

    def test_a_role_that_is_missing(self):
        # `f"{None:<9}"` raises; the role is not always a string.
        assert "None" in describe_message({"content": "x"})
