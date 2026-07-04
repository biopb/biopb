"""JSON Schema for the tensor-server config file, generated from the dataclasses.

The config dataclasses in :mod:`biopb_tensor_server.config` and their
``_CONSTRAINTS`` validation table (the ``_Range`` / ``_Enum`` rules enforced in
each ``__post_init__``) are the single source of truth for the config. This
module re-projects them as a JSON Schema (Draft 2020-12) describing the
*on-disk* config file, so one definition feeds three consumers without drift:

- a config generator and editor autocomplete (via ``$schema``);
- optional pre-flight validation of generated JSON;
- the **server's own unknown-key warning** -- ``config._warn_unknown_config_keys``
  derives its known-key set by walking this schema's properties, so there is no
  separate hardcoded key table to drift from the parser (biopb/biopb#34, #234).

Almost nothing is hand-maintained per field: the key set + types come from the
dataclasses, bounds/enums from ``_CONSTRAINTS``. The two declarative pieces are
``_ONDISK_OVERRIDES`` (the few fields whose wire form differs --
``cache.*_mb``/``*_gb`` convert to byte fields) and
``_DEPRECATED_ALIASES`` (legacy keys the parser still accepts but that aren't
dataclass fields). Both are small and change only when the *file layout* does.

Emit it with ``biopb-tensor-server config-schema``.
"""

from __future__ import annotations

import dataclasses
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
from biopb_tensor_server.remote import CredentialProfile

SCHEMA_ID = "https://biopb.org/schemas/tensor-server-config.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Classes whose *scalar* fields populate the top-level config sections. Each
# field is routed to its on-disk (section, key) via ondisk_location(); nested
# dataclass / list fields (cache, pyramid, ..., sources) are their own sections
# and are skipped during enumeration.
_SECTION_CLASSES = (
    ServerConfig,
    CacheConfig,
    PyramidConfig,
    PrecacheConfig,
    MetadataDbConfig,
)

# (ClassName, dataclass field) -> (on-disk section, on-disk key) for the fields
# whose wire form diverges from the dataclass. Everything else maps to
# (_SECTION_FOR[class], field-name). See parse_config:
#   - cache.max_entries/max_bytes feed memory_max_*; file_max_*_mb/_gb are
#     converted to the *_bytes fields (the >= 1 bound stays sensible in MB/GB).
_ONDISK_OVERRIDES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("CacheConfig", "memory_max_entries"): ("cache", "max_entries"),
    ("CacheConfig", "memory_max_bytes"): ("cache", "max_bytes"),
    ("CacheConfig", "file_max_segment_bytes"): ("cache", "file_max_segment_mb"),
    ("CacheConfig", "file_max_total_bytes"): ("cache", "file_max_total_gb"),
}

# Legacy / back-compat keys parse_config still accepts but that are NOT dataclass
# fields, so introspection cannot find them. Declared here as deprecated
# properties so the published schema documents them AND the runtime known-key set
# (derived from this schema) stays quiet on a valid back-compat config -- this is
# what lets us delete #234's hardcoded _KNOWN_* tables. {section: {key: (type,
# note)}}. The pyramid knobs still honored under [precache], the source `path`
# alias, and the deprecated metadata_db.enabled flag are handled in build_*.
_DEPRECATED_ALIASES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "server": {
        "watcher_type": ("string", "Deprecated alias for monitor_mode."),
        "poll_interval": ("number", "Deprecated alias for rescan_interval."),
    },
    "metadata_db": {
        # Removed (biopb/biopb#225): the metadata DB is mandatory. Kept in the
        # schema as a tolerated no-op so an old config with the flag is warned by
        # parse_config, not flagged as an unknown key here.
        "enabled": (
            "boolean",
            "Removed (biopb/biopb#225): the metadata database is mandatory "
            "(always on); this flag is ignored. Drop it from your config.",
        ),
    },
}

# Default instances give each field's runtime value -> its JSON type. Defaults
# are valid, so construction emits no warnings.
_DEFAULT_INSTANCES = {cls.__name__: cls() for cls in _SECTION_CLASSES}


def _json_type(value: Any) -> str:
    # bool is an int subclass; check it first. None (e.g. write_dir) is a path
    # string on the wire.
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


def _empty_section() -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": True, "properties": {}}


def _scalar_property(class_name: str, field: str) -> Dict[str, Any]:
    inst = _DEFAULT_INSTANCES[class_name]
    value = getattr(inst, field)
    prop: Dict[str, Any] = {"type": _json_type(value)}
    # The dataclass default, so a config editor can render the effective value of
    # an omitted key (e.g. show a default-true boolean checked) and only write a
    # key when it diverges. None (e.g. write_dir) is left out -- it is "unset",
    # not a meaningful default to echo. Non-JSON-native defaults (e.g. a Path
    # file_cache_dir) are coerced to str to match their "string" wire type and
    # stay JSON-serializable (the schema is sent over HTTP by GET /api/config).
    if value is not None:
        prop["default"] = (
            value if isinstance(value, (bool, int, float, str)) else str(value)
        )
    constraint = _CONSTRAINTS.get(class_name, {}).get(field)
    if constraint is not None:
        prop.update(constraint.to_json_schema())
        # describe() carries the rule (and, for case-insensitive enums where
        # to_json_schema() emits nothing, the accepted set) so the description
        # is useful for autocomplete even when validation stays lenient.
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
    # Curated, but enumerates every public SourceConfig field (asserted by the
    # drift-guard test) plus the deprecated `path` alias.
    item: Dict[str, Any] = {
        "type": "object",
        "required": ["url"],
        "additionalProperties": True,
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Path or URL to the data source. Local paths are stable; "
                    "remote URLs (s3://, http(s)://, grpc://) are experimental."
                ),
            },
            "source_id": {"type": "string"},
            "dataset": {"type": "string", "description": "HDF5 dataset path."},
            "dim_labels": {"type": "array", "items": {"type": "string"}},
            "monitor": {
                "type": "boolean",
                "description": "Watch this local directory for add/delete events.",
            },
            "cloud": {
                "type": "boolean",
                "description": (
                    "(experimental) Treat as a cloud/synced root; admit offline "
                    "placeholders resolved lazily on first access."
                ),
            },
            "credentials_profile": {
                "type": "string",
                "description": (
                    "(experimental) Credential profile for a remote-URL source "
                    "(s3://, http(s)://, ...)."
                ),
            },
            "alias": {
                "type": "string",
                "description": (
                    "(experimental) Namespace prefix for an upstream 'tensor-server' "
                    "source (<alias>__<upstream_source_id>); must be slash-free."
                ),
            },
            "path": {
                "type": "string",
                "description": "Deprecated alias for url.",
                "deprecated": True,
            },
        },
    }
    type_enum = _source_type_enum()
    type_prop: Dict[str, Any] = {
        "type": ["string", "null"],
        "description": (
            "Storage type; auto-detected for local files when omitted. "
            "('tensor-server' is experimental.)"
        ),
    }
    if type_enum:
        type_prop["enum"] = list(type_enum) + [None]
    item["properties"]["type"] = type_prop
    return {"type": "array", "items": item}


def _credentials_schema() -> Dict[str, Any]:
    # Profile keys come from the CredentialProfile dataclass (all string-valued).
    profile_props = {
        f.name: {"type": "string"} for f in dataclasses.fields(CredentialProfile)
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "default_profile": {"type": "string"},
            "profiles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["name"],
                    "properties": profile_props,
                },
            },
        },
    }


def build_config_schema() -> Dict[str, Any]:
    """Build the tensor-server config JSON Schema from the config dataclasses.

    Returns a Draft 2020-12 schema describing the on-disk config file, with a
    property for **every** key the parser reads (so it doubles as the known-key
    set) plus value bounds/enums for the constrained ones. Sections keep
    ``additionalProperties: true`` -- the published schema enforces values and
    documents keys without erroring on unknown ones during the migration window;
    the server's runtime warning is what flags unknown keys (warn-only).
    """
    sections: Dict[str, Dict[str, Any]] = {}

    # 1. Every scalar dataclass field -> a property under its on-disk section.
    for cls in _SECTION_CLASSES:
        cname = cls.__name__
        inst = _DEFAULT_INSTANCES[cname]
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            value = getattr(inst, f.name)
            if dataclasses.is_dataclass(value) or isinstance(value, list):
                continue  # a nested section, handled on its own pass
            section, key = ondisk_location(cname, f.name)
            sect = sections.setdefault(section, _empty_section())
            sect["properties"][key] = _scalar_property(cname, f.name)

    # 2. Deprecated scalar aliases that aren't dataclass fields.
    for section, aliases in _DEPRECATED_ALIASES.items():
        sect = sections.setdefault(section, _empty_section())
        for key, (jtype, note) in aliases.items():
            sect["properties"][key] = {
                "type": jtype,
                "description": note,
                "deprecated": True,
            }

    # 3. The pyramid knobs are still honored under [precache] for back-compat;
    #    mirror them there as deprecated copies of the canonical [pyramid] props.
    for key, prop in sections["pyramid"]["properties"].items():
        mirrored = dict(prop)
        mirrored["deprecated"] = True
        base = prop.get("description", "")
        mirrored["description"] = (
            f"Deprecated under [precache]; set it under [pyramid]. {base}".strip()
        )
        sections["precache"]["properties"][key] = mirrored

    properties: Dict[str, Any] = dict(sections)
    properties["sources"] = _sources_schema()
    properties["credentials"] = _credentials_schema()

    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        "title": "biopb tensor server configuration",
        "description": (
            "Configuration for biopb-tensor-server (the Arrow Flight data "
            "plane). Generated from the server's config dataclasses + "
            "_CONSTRAINTS validation table (biopb/biopb#34); the key set and "
            "value bounds here match what the server reads and enforces."
        ),
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }


def known_config_keys(
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[set, Dict[str, set], set, set]:
    """The known-key sets the server's unknown-key warning uses, derived from
    the schema so there is no separate hardcoded table.

    Returns ``(sections, section_keys, source_keys, profile_keys)``:
    - ``sections``: the valid top-level section names (incl. ``sources``);
    - ``section_keys``: ``{section: {keys}}`` for the object sections;
    - ``source_keys``: the keys allowed on a ``[[sources]]`` item;
    - ``profile_keys``: the keys allowed on a ``[credentials.profiles]`` item.
    """
    if schema is None:
        schema = build_config_schema()
    props = schema["properties"]
    sections = set(props)
    section_keys = {
        name: set(spec.get("properties", {}))
        for name, spec in props.items()
        if spec.get("type") == "object"
    }
    source_keys = set(props["sources"]["items"]["properties"])
    profile_keys = set(
        props["credentials"]["properties"]["profiles"]["items"]["properties"]
    )
    return sections, section_keys, source_keys, profile_keys


# Stable set of every (section, key) the schema *constrains* -- used by the
# drift guard test to assert the schema covers _CONSTRAINTS.
def constrained_ondisk_keys() -> Dict[Tuple[str, str], Any]:
    out: Dict[Tuple[str, str], Any] = {}
    for class_name, constraints in _CONSTRAINTS.items():
        for field, constraint in constraints.items():
            out[ondisk_location(class_name, field)] = constraint
    return out
