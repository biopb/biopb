"""Fields a provider puts on an assistant turn and demands back with it.

Two clients in this package hold a conversation and re-send it whole every turn
-- the built-in chat loop (``mcp/_chat.py``) and the agentbench harness
(``_tests/agentbench``) -- and both rebuild the assistant turn from the parts
they happen to care about, which silently drops everything else the provider
put there. A thinking model rejects the *whole request* when its own
``reasoning_content`` does not come back on that same message, so the drop
fails a turn later than its cause and the thread stays broken until it is
reset (biopb/biopb#975).

One definition for both, like :mod:`_endpoint`: a field learned on one side is
echoed by the other, rather than only where it was found.
"""

from collections.abc import Mapping
from typing import Any

#: Keys a provider may return on an assistant message and require back on the
#: next request. Tried in order; the first one *present* is carried under its
#: own name, because a provider spelling it ``reasoning`` will not accept
#: ``reasoning_content``.
ECHOED_FIELDS = ("reasoning_content", "reasoning")


def echoed_fields(message: Any) -> dict:
    """What *message* carries that has to be sent back with it.

    Takes either a mapping or an object: the chat loop's adapter returns the
    provider's message dict whole, the harness holds SDK models.
    """
    for key in ECHOED_FIELDS:
        value = (
            message.get(key)
            if isinstance(message, Mapping)
            else getattr(message, key, None)
        )
        if value is not None:
            return {key: value}
    return {}
