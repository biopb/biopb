"""Shared JSON-Schema projection for config dataclasses, in core ``biopb``.

Both config-bearing packages describe their on-disk config with a JSON Schema
generated from dataclasses + a ``_CONSTRAINTS`` table, so a config editor and the
runtime known-key set have one source of truth (biopb/biopb#34). The *shape* of
each package's config differs -- the tensor server has bespoke ``sources`` /
``credentials`` arrays and a few fields whose wire key diverges from the
dataclass; biopb-mcp is flat scalar sections plus its own array fields -- but the
**per-field projection** is identical: a scalar field becomes ``{type, default?,
description(help), constraint?, bounds…}`` by the exact same rules.

That identical core lives here so neither package re-implements it (and
biopb-mcp, which cannot import biopb-tensor-server -- not on PyPI -- reuses it
the same way it already shares :mod:`biopb._config.constraints`). Each package
keeps its own *composer* that calls :func:`dataclass_section` for its scalar
sections and adds whatever bespoke array/alias parts it has.

Deliberately stdlib-only, like the sibling :mod:`biopb._config.constraints` and
:mod:`biopb._config.locations`: it duck-types the constraint objects
(``to_json_schema`` / ``describe``) and never imports the constraint classes, so
it pulls in none of the heavy adapter/discovery machinery.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional


def json_type(value: Any) -> str:
    """The JSON Schema ``type`` for a Python default value.

    ``bool`` is an ``int`` subclass, so it is checked first. A ``None`` default
    (e.g. an unset path) is a string on the wire.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def scalar_property(
    value: Any,
    help_text: Optional[str] = None,
    constraint: Any = None,
) -> Dict[str, Any]:
    """Project one scalar config field to a JSON Schema property.

    - ``default`` echoes the dataclass default (omitted for ``None``; non
      JSON-native values such as ``Path`` become ``str``).
    - ``description`` is the field's ``metadata["help"]``.
    - a *constraint* contributes its bounds/enum keywords; its human rule goes in
      a separate ``constraint`` key, which also carries case-insensitive enums.
    """
    prop: Dict[str, Any] = {"type": json_type(value)}
    if value is not None:
        prop["default"] = (
            value if isinstance(value, (bool, int, float, str)) else str(value)
        )
    if help_text:
        prop["description"] = help_text
    if constraint is not None:
        prop.update(constraint.to_json_schema())
        prop["constraint"] = constraint.describe()
    return prop


def dataclass_section(
    cls: Any,
    constraints: Optional[Dict[str, Any]] = None,
    *,
    default_instance: Any = None,
    key_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Project a dataclass's **scalar** fields to ``{on-disk key: property}``.

    Nested dataclass and list fields are skipped (the caller composes those).
    ``constraints`` is ``_CONSTRAINTS[cls.__name__]``; ``key_map`` remaps field
    names whose wire key differs; ``default_instance`` reuses a built default.
    """
    inst = default_instance if default_instance is not None else cls()
    constraints = constraints or {}
    key_map = key_map or {}
    out: Dict[str, Dict[str, Any]] = {}
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        value = getattr(inst, f.name)
        if dataclasses.is_dataclass(value) or isinstance(value, list):
            continue
        key = key_map.get(f.name, f.name)
        out[key] = scalar_property(
            value, f.metadata.get("help"), constraints.get(f.name)
        )
    return out
