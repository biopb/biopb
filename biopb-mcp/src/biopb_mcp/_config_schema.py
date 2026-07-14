"""JSON Schema for the biopb-mcp config file, generated from the dataclasses.

Mirrors the tensor server's ``config_schema`` but for biopb-mcp's own (separate)
config -- nothing merges; the two share only the *machinery*
(:mod:`biopb._config_schema`). The flat section dataclasses in
:mod:`biopb_mcp._config` and their ``_CONSTRAINTS`` table are the single source
of truth; this re-projects them as a JSON Schema (Draft 2020-12) describing
``mcp-config.json``, so a config generator, editor autocomplete, and a
schema-driven admin editor share one definition without drift.

Scalar fields come from :func:`biopb._config_schema.dataclass_section`; the few
list-valued fields (grid tile vectors, the server/origin lists) are added here as
the package-specific array parts, exactly as the tensor composer adds its
``sources`` / ``credentials`` arrays.

Emit it with ``python -m biopb_mcp._config_schema`` (or via the control's config
endpoint).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from biopb._config_schema import dataclass_section, json_type

from biopb_mcp._config import _CONSTRAINTS, _SECTION_CLASSES

SCHEMA_ID = "https://biopb.org/schemas/mcp-config.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _array_property(
    default_list: List[Any], help_text: Optional[str]
) -> Dict[str, Any]:
    """Project a list-valued field. Item type is inferred from the default's
    first element (grid vectors are integers; the empty server/origin lists
    default to string items)."""
    item_type = json_type(default_list[0]) if default_list else "string"
    prop: Dict[str, Any] = {
        "type": "array",
        "items": {"type": item_type},
        "default": list(default_list),
    }
    if help_text:
        prop["description"] = help_text
    return prop


def _section_schema(cls: Any) -> Dict[str, Any]:
    """One section object: scalar fields via the shared engine + list fields."""
    props = dataclass_section(cls, _CONSTRAINTS.get(cls.__name__, {}))
    inst = cls()
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        value = getattr(inst, f.name)
        if isinstance(value, list):
            props[f.name] = _array_property(value, f.metadata.get("help"))
    return {"type": "object", "additionalProperties": True, "properties": props}


def build_mcp_config_schema() -> Dict[str, Any]:
    """Build the biopb-mcp config JSON Schema from the config dataclasses.

    Returns a Draft 2020-12 schema describing ``mcp-config.json``, with a property
    for every key the config reads plus value bounds/enums for the constrained
    ones. Sections keep ``additionalProperties: true`` (values are enforced and
    keys documented without erroring on a forward-unknown key).
    """
    properties = {
        section: _section_schema(cls) for section, cls in _SECTION_CLASSES.items()
    }
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        "title": "biopb-mcp configuration",
        "description": (
            "Configuration for biopb-mcp (the agent client: napari plugin + MCP "
            "server). Generated from the config dataclasses + _CONSTRAINTS table "
            "(biopb/biopb#34); the key set and value bounds match what biopb-mcp "
            "reads and enforces."
        ),
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build_mcp_config_schema(), indent=2))
