"""What a session advertises before it exists: a snapshot the stdio shim serves.

The shim answers ``initialize`` and the list requests itself, so a client that
connects and never calls a tool never costs a session child. Those answers must
be the child's own, so they are generated from the live FastMCP server into
``_surface.json`` and shipped; a test fails when the two drift. Regenerate with
``python -m biopb_mcp.mcp._surface``.

Loading the snapshot imports only ``mcp.types``, which the shim holds anyway;
building it imports the whole server.
"""

import json
from importlib import resources

from mcp import types

_FILE = "_surface.json"


class Surface:
    """The advertised capabilities and the four lists, as ``mcp.types`` models."""

    def __init__(self, data: dict):
        self.capabilities = types.ServerCapabilities.model_validate(
            data["capabilities"]
        )
        self.tools = [types.Tool.model_validate(t) for t in data["tools"]]
        self.resources = [types.Resource.model_validate(r) for r in data["resources"]]
        self.resource_templates = [
            types.ResourceTemplate.model_validate(r) for r in data["resource_templates"]
        ]
        self.prompts = [types.Prompt.model_validate(p) for p in data["prompts"]]


def load() -> Surface:
    """The shipped snapshot."""
    text = resources.files(__package__).joinpath(_FILE).read_text(encoding="utf-8")
    return Surface(json.loads(text))


def build() -> dict:
    """The snapshot as the live server would answer it, JSON-ready."""
    import anyio

    from . import _server  # noqa: F401 - registers the tools on `mcp`
    from ._app import mcp

    def dump(models):
        return [
            m.model_dump(mode="json", by_alias=True, exclude_none=True) for m in models
        ]

    async def lists():
        return (
            await mcp.list_tools(),
            await mcp.list_resources(),
            await mcp.list_resource_templates(),
            await mcp.list_prompts(),
        )

    tools, res, templates, prompts = anyio.run(lists)
    options = mcp._mcp_server.create_initialization_options()
    return {
        "capabilities": options.capabilities.model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
        "tools": dump(tools),
        "resources": dump(res),
        "resource_templates": dump(templates),
        "prompts": dump(prompts),
    }


def render(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    path = resources.files(__package__).joinpath(_FILE)
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(render(build()))
    print(f"wrote {path}")
