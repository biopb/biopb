"""Tests for the generated biopb-mcp config JSON Schema (biopb/biopb#34).

The schema is projected from the flat config dataclasses + their ``_CONSTRAINTS``
table, so the key property to guard is that it stays in sync: every constrained
value the config enforces is reflected, and the shipped defaults validate.
"""

import dataclasses

import pytest
from jsonschema import Draft202012Validator

from biopb_mcp._config import _CONSTRAINTS, _SECTION_CLASSES, DEFAULT_CONFIG
from biopb_mcp._config_schema import build_mcp_config_schema


@pytest.fixture(scope="module")
def schema():
    return build_mcp_config_schema()


@pytest.fixture(scope="module")
def validator(schema):
    return Draft202012Validator(schema)


def test_schema_is_valid_draft202012(schema):
    Draft202012Validator.check_schema(schema)


def test_defaults_validate(validator):
    """The shipped DEFAULT_CONFIG must pass its own schema."""
    assert list(validator.iter_errors(DEFAULT_CONFIG)) == []


def test_every_section_present(schema):
    for section in _SECTION_CLASSES:
        assert section in schema["properties"], f"missing section {section!r}"


def test_every_scalar_field_is_a_property(schema):
    """Drift guard: every scalar/list dataclass field appears in the schema."""
    for section, cls in _SECTION_CLASSES.items():
        props = schema["properties"][section]["properties"]
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            assert f.name in props, f"{section}.{f.name} missing from schema"


def test_scalar_defaults_and_help_emitted(schema):
    for section, cls in _SECTION_CLASSES.items():
        props = schema["properties"][section]["properties"]
        inst = cls()
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            prop = props[f.name]
            # help -> description (every field carries one)
            assert prop.get("description"), f"{section}.{f.name} has no description"
            value = getattr(inst, f.name)
            if not isinstance(value, list):
                assert prop.get("default") == value, f"{section}.{f.name} default drift"


def test_every_constraint_reflected(schema):
    """Each (class, field) in _CONSTRAINTS is a schema property with matching bounds."""
    for section, cls in _SECTION_CLASSES.items():
        for field_name, constraint in _CONSTRAINTS.get(cls.__name__, {}).items():
            prop = schema["properties"][section]["properties"][field_name]
            js = constraint.to_json_schema()
            for kw, val in js.items():
                assert prop.get(kw) == val, f"{section}.{field_name} {kw} mismatch"
            # the human rule rides its own key so description stays prose
            assert prop.get("constraint")


def test_list_fields_are_arrays(schema):
    grid = schema["properties"]["grid"]["properties"]
    assert grid["size_2d"]["type"] == "array"
    assert grid["size_2d"]["items"]["type"] == "integer"
    services = schema["properties"]["services"]["properties"]
    assert services["process_image_servers"]["type"] == "array"
    assert services["process_image_servers"]["items"]["type"] == "string"


@pytest.mark.parametrize(
    "cfg",
    [
        {"transport": {"port": 8080}},
        {"transport": {"kind": "http"}},
        {"dask": {"scheduler": "threads"}},
        {"pyramid": {"downscale_factor": 2}},
        {"future_unknown": {"knob": 1}},  # additionalProperties: true
    ],
)
def test_accepts_valid(validator, cfg):
    assert list(validator.iter_errors(cfg)) == [], f"unexpectedly rejected: {cfg}"


@pytest.mark.parametrize(
    "cfg",
    [
        {"transport": {"port": 0}},
        {"transport": {"port": 70000}},
        {"transport": {"kind": "websocket"}},
        {"dask": {"scheduler": "bogus"}},
        {"pyramid": {"downscale_factor": 1}},
    ],
)
def test_rejects_invalid(validator, cfg):
    assert list(validator.iter_errors(cfg)), f"should have been rejected: {cfg}"
