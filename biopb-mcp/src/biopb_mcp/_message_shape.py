"""The shape of the conversation we sent, for an error someone has to read.

A provider names the message it dislikes, never which one it was.
``reasoning_content in the thinking mode must be passed back`` says a turn was
missing a field; it does not say which turn, or what the turns around it
carried (biopb/biopb#975). Both clients here re-send the whole thread on every
call -- the built-in chat loop (``mcp/_model.py``) and the agentbench harness
(``_tests/agentbench``) -- so the thread they built is the only thing that
identifies the offending turn, and reproducing it costs a paid run.

Roles, the keys on each message, and tool-call counts. **Never content**, and
so never an image, a tool argument or anything the user typed: that line is
what makes the shape safe to put in an error, and it is why the keys are worth
showing at all -- a rejection of this class is about which keys are present.

One definition for both, like :mod:`_endpoint` and :mod:`_provider_echo`
(biopb/biopb#990).
"""

from collections.abc import Mapping

#: Messages to show. A rejection is about the tail of the thread, and the whole
#: of a long one would bury the provider's own words above it.
TAIL = 10


def describe_message(message) -> str:
    """One message as role + keys + tool-call count. No content."""
    if not isinstance(message, Mapping):
        return f"<{type(message).__name__}>"
    calls = len(message.get("tool_calls") or ())
    keys = ",".join(sorted(str(k) for k in message if k != "role"))
    return f"{str(message.get('role')):<9} keys=[{keys}]" + (
        f" tool_calls={calls}" if calls else ""
    )


def describe_messages(messages, tail: int = TAIL) -> str:
    """The tail of *messages*, one line each, indented to sit under an error.

    Total by construction: this runs while an error is being built, so a
    surprise in the payload must not replace the provider's words with a
    traceback from the code describing them.
    """
    try:
        shown = list(messages)[-tail:] if tail else list(messages)
        if not shown:
            return "  no messages were sent."
        lines = "\n    ".join(describe_message(m) for m in shown)
        return f"  last {len(shown)} of {len(messages)} messages sent:\n    {lines}"
    except Exception:  # pragma: no cover - the error being reported is the point
        return "  (could not describe the messages sent)"
