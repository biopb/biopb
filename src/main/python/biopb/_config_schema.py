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
the same way it already shares :mod:`biopb._config_constraints`). Each package
keeps its own *composer* that calls :func:`dataclass_section` for its scalar
sections and adds whatever bespoke array/alias parts it has.

Deliberately stdlib-only, like the sibling :mod:`biopb._config_constraints` and
:mod:`biopb._config_location`: it duck-types the constraint objects
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

    - ``default`` echoes the dataclass default so an editor can render the
      effective value of an omitted key (a default-true boolean shows checked);
      a ``None`` default is omitted ("unset", not a meaningful echo). Non
      JSON-native defaults (e.g. a ``Path``) are coerced to ``str`` to match
      their ``"string"`` wire type and stay JSON-serializable.
    - ``description`` is prose, from the field's ``metadata["help"]`` -- the
      single source of truth, so there is no second doc table to drift.
    - a *constraint* (any object exposing ``to_json_schema`` / ``describe``)
      contributes its bounds/enum keywords, and its human rule rides a separate
      ``constraint`` key so ``description`` stays pure prose. The
      case-insensitive-enum accepted set (which ``to_json_schema`` deliberately
      keeps out of a hard ``enum``) survives there.
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

    Nested dataclass and list fields are skipped -- they are their own
    sections/arrays and belong to the caller's bespoke composer. ``constraints``
    is the per-class ``{field name: constraint}`` map (the value at
    ``_CONSTRAINTS[cls.__name__]``). ``key_map`` remaps the few field names whose
    wire key diverges from the dataclass field (identity otherwise);
    ``default_instance`` lets a caller reuse an already-built default instance.
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
