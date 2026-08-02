"""MCP tool schemas ⇄ the function-calling shape a chat model expects.

The agent under test is **not** an MCP client. §5a puts it outside the family
that wrote these skills, which in practice means a hosted non-Anthropic model
reached over the OpenAI-compatible chat-completions API — and that API speaks
`tools=[{"type": "function", ...}]`, not MCP.

So something has to translate, and this is it. Deliberately thin: the schemas
that go out are **the server's own**, not a restatement of them. If
`execute_code` renames a parameter, an agent driven through here starts getting
it wrong, which is the point of running against a real session at all.

The one liberty taken is dropping keys some providers reject in a function
parameter schema (`$schema`, `title` at the root). That is a transport
accommodation, not an edit to the contract — :func:`to_function_tools` keeps
every property and every ``required`` entry exactly as the server sent them,
and ``test_bridge`` pins that.
"""

from __future__ import annotations

import json
from typing import Any

#: Keys a provider may reject at the root of a parameters schema. Neither
#: carries meaning for the call itself.
_STRIPPED_ROOT_KEYS = ("$schema", "title")


def to_function_tools(tools) -> list[dict]:
    """MCP :class:`~._session.ToolSpec` list -> chat-completions ``tools``."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": _parameters(tool.input_schema),
            },
        }
        for tool in tools
    ]


def _parameters(schema: dict | None) -> dict:
    """The server's schema, minus keys that are noise on this wire.

    A tool with no inputs still needs an object schema with an empty
    ``properties``: several providers reject a bare ``{}`` or omit the argument
    entirely, and biopb-mcp has three no-argument tools.
    """
    out = dict(schema or {})
    for key in _STRIPPED_ROOT_KEYS:
        out.pop(key, None)
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


def parse_arguments(raw: Any) -> dict:
    """A model's ``arguments`` as a dict, whatever shape it arrived in.

    Providers send a JSON *string*; some models send malformed JSON, and that
    is a real failure mode worth surfacing rather than crashing on — a run that
    could not form a tool call is a finding about the model, and the trace is
    where it belongs. An unparseable payload comes back as a single
    ``__malformed__`` key so the caller can record it and carry on.
    """
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"__malformed__": str(raw)}
    return parsed if isinstance(parsed, dict) else {"__malformed__": str(raw)}
