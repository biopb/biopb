"""JSON Schema for the tensor-server config file, generated from ``_CONSTRAINTS``.

The validation rules in :mod:`biopb_tensor_server.config` (the ``_CONSTRAINTS``
table of :class:`~biopb_tensor_server.config._Range` / ``_Enum`` per dataclass
field) are the single source of truth for what a config value may be. This
module re-projects that table as a JSON Schema (Draft 2020-12) describing the
*on-disk* config file, so the same constraints feed three consumers without
drift: a config generator, editor autocomplete (via ``$schema``), and optional
pre-flight validation of generated JSON (biopb/biopb#34).

Nothing here is hand-maintained per field: bounds and enums come straight from
``_CONSTRAINTS``, types/defaults from the dataclasses. The only declarative bit
is ``_ONDISK_OVERRIDES`` -- the handful of fields whose on-disk key or section
differs from the dataclass field (``parse_config`` reads ``[compute]`` into
``ServerConfig`` and converts ``cache.*_mb``/``*_gb`` to byte fields). That map
is small and changes only when the *file layout* changes, which is rare; the
drift-prone part (the value constraints) stays generated.

Emit it with ``biopb-tensor-server config-schema``.
"""

from __future__ import annotations

import typing
from typing import Any, Dict, Optional, Tuple

from biopb_tensor_server.config import (
    _CONSTRAINTS,
    _SECTION_FOR,
    CacheConfig,
    MetadataDbConfig,
    PrecacheConfig,
    PyramidConfig,
    ServerConfig,
    SourceConfig,
)

SCHEMA_ID = "https://biopb.org/schemas/tensor-server-config.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# (ClassName, dataclass field) -> (on-disk section, on-disk key) for the fields
# whose wire form diverges from the dataclass. Everything else maps to
# (_SECTION_FOR[class], field-name). See parse_config:
#   - compute_backend / gpu_* live on ServerConfig but are read from [compute];
#   - cache.max_entries/max_bytes feed memory_max_*; file_max_*_mb/_gb are
#     converted to the *_bytes fields (the >= 1 bound stays sensible in MB/GB).
_ONDISK_OVERRIDES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("ServerConfig", "compute_backend"): ("compute", "backend"),
    ("ServerConfig", "gpu_min_input_mb"): ("compute", "gpu_min_input_mb"),
    ("ServerConfig", "gpu_min_linear_input_mb"): ("compute", "gpu_min_linear_input_mb"),
    ("ServerConfig", "gpu_memory_safety_factor"): (
        "compute",
        "gpu_memory_safety_factor",
    ),
    ("ServerConfig", "gpu_min_merged_chunks"): ("compute", "gpu_min_merged_chunks"),
    ("CacheConfig", "memory_max_entries"): ("cache", "max_entries"),
    ("CacheConfig", "memory_max_bytes"): ("cache", "max_bytes"),
    ("CacheConfig", "file_max_segment_bytes"): ("cache", "file_max_segment_mb"),
    ("CacheConfig", "file_max_total_bytes"): ("cache", "file_max_total_gb"),
}

# Default instances give each field's runtime value -> its JSON type, without
# re-stating the types. Defaults are valid, so construction emits no warnings.
_DEFAULT_INSTANCES = {
    "CacheConfig": CacheConfig(),
    "PyramidConfig": PyramidConfig(),
    "PrecacheConfig": PrecacheConfig(),
    "MetadataDbConfig": MetadataDbConfig(),
    "ServerConfig": ServerConfig(),
}


def _json_type(value: Any) -> str:
    # bool is an int subclass; check it first.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def ondisk_location(class_name: str, field: str) -> Tuple[str, str]:
    """The (section, key) a dataclass field is written as on disk."""
    override = _ONDISK_OVERRIDES.get((class_name, field))
    if override is not None:
        return override
    return _SECTION_FOR.get(class_name, class_name), field


def _property_schema(class_name: str, field: str, constraint) -> Dict[str, Any]:
    inst = _DEFAULT_INSTANCES[class_name]
    prop: Dict[str, Any] = {"type": _json_type(getattr(inst, field))}
    prop.update(constraint.to_json_schema())
    # describe() carries the human-readable rule (and, for case-insensitive
    # enums where to_json_schema() emits nothing, the accepted set) so the
    # description is useful for autocomplete even when validation stays lenient.
    prop["description"] = constraint.describe()
    return prop


def _source_type_enum() -> Optional[list]:
    """The accepted source ``type`` values, lifted from SourceConfig's Literal."""
    try:
        hints = typing.get_type_hints(SourceConfig)
        # type: Optional[Literal[...]]  -> args are (Literal[...], NoneType)
        for arg in typing.get_args(hints["type"]):
            literals = typing.get_args(arg)
            if literals:
                return list(literals)
    except Exception:
        pass
    return None


def _sources_schema() -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "object",
        "required": ["url"],
        "additionalProperties": True,
        "properties": {
            "url": {"type": "string", "description": "Path or URL to the data source."},
            "source_id": {"type": "string"},
            "dataset": {"type": "string", "description": "HDF5 dataset path."},
            "dim_labels": {"type": "array", "items": {"type": "string"}},
            "monitor": {
                "type": "boolean",
                "description": "Watch this local directory for add/delete events.",
            },
            "cloud": {
                "type": "boolean",
                "description": "Treat as a cloud/synced root; admit offline placeholders.",
            },
            "credentials_profile": {"type": "string"},
        },
    }
    type_enum = _source_type_enum()
    type_prop: Dict[str, Any] = {
        "type": ["string", "null"],
        "description": "Storage type; auto-detected for local files when omitted.",
    }
    if type_enum:
        type_prop["enum"] = list(type_enum) + [None]
    item["properties"]["type"] = type_prop
    return {"type": "array", "items": item}


def build_config_schema() -> Dict[str, Any]:
    """Build the tensor-server config JSON Schema from ``_CONSTRAINTS``.

    Returns a Draft 2020-12 schema describing the on-disk config file. Sections
    keep ``additionalProperties: true`` -- the schema enforces the *constrained*
    values (the dangerous ones #34 exists to catch) and documents them for
    autocomplete, without trying to be a closed dictionary of every key.
    """
    sections: Dict[str, Dict[str, Any]] = {}
    for class_name, constraints in _CONSTRAINTS.items():
        for field, constraint in constraints.items():
            section, key = ondisk_location(class_name, field)
            sect = sections.setdefault(
                section,
                {"type": "object", "additionalProperties": True, "properties": {}},
            )
            sect["properties"][key] = _property_schema(class_name, field, constraint)

    properties: Dict[str, Any] = dict(sections)
    properties["sources"] = _sources_schema()
    properties["credentials"] = {"type": "object", "additionalProperties": True}

    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        "title": "biopb tensor server configuration",
        "description": (
            "Configuration for biopb-tensor-server (the Arrow Flight data "
            "plane). Generated from the server's _CONSTRAINTS validation table "
            "(biopb/biopb#34); value bounds and enums here match what the server "
            "enforces at startup."
        ),
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }


# Stable set of every (section, key) the schema constrains -- used by the drift
# guard test to assert the schema covers _CONSTRAINTS without re-walking it.
def constrained_ondisk_keys() -> Dict[Tuple[str, str], Any]:
    out: Dict[Tuple[str, str], Any] = {}
    for class_name, constraints in _CONSTRAINTS.items():
        for field, constraint in constraints.items():
            out[ondisk_location(class_name, field)] = constraint
    return out
