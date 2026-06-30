"""Tests for the generated config JSON Schema (biopb/biopb#34).

The schema is *projected from* the server's ``_CONSTRAINTS`` validation table,
so the key property to guard is that it stays in sync: every constrained value
the server enforces at startup must be reflected in the schema, and the schema
must accept what the server accepts and reject what the server rejects.
"""

import jsonschema
import pytest
from biopb_tensor_server.config import _CONSTRAINTS, _Enum, _Range
from biopb_tensor_server.config_schema import (
    build_config_schema,
    constrained_ondisk_keys,
    ondisk_location,
)
from jsonschema import Draft202012Validator


@pytest.fixture(scope="module")
def schema():
    return build_config_schema()


@pytest.fixture(scope="module")
def validator(schema):
    return Draft202012Validator(schema)


def _section_props(schema, section):
    return schema["properties"][section]["properties"]


def test_schema_is_valid_draft202012(schema):
    """The emitted document is itself a well-formed JSON Schema."""
    Draft202012Validator.check_schema(schema)


def test_top_level_sections_present(schema):
    props = schema["properties"]
    for section in (
        "server",
        "compute",
        "cache",
        "pyramid",
        "precache",
        "metadata_db",
        "sources",
        "credentials",
    ):
        assert section in props, f"missing top-level section {section!r}"


def test_every_constraint_is_reflected(schema):
    """Drift guard: each (class, field) in _CONSTRAINTS appears in the schema
    under its on-disk (section, key), with bounds/enum matching the constraint."""
    for class_name, constraints in _CONSTRAINTS.items():
        for field, constraint in constraints.items():
            section, key = ondisk_location(class_name, field)
            section_props = _section_props(schema, section)
            assert key in section_props, (
                f"{class_name}.{field} -> [{section}].{key} missing from schema"
            )
            prop = section_props[key]
            if isinstance(constraint, _Range):
                if constraint.min is not None:
                    assert prop.get("minimum") == constraint.min
                else:
                    assert "minimum" not in prop
                if constraint.max is not None:
                    assert prop.get("maximum") == constraint.max
                else:
                    assert "maximum" not in prop
            elif isinstance(constraint, _Enum):
                if constraint.case_insensitive:
                    # Lenient: no hard enum, but the accepted set is documented.
                    assert "enum" not in prop
                    assert prop.get("description")
                else:
                    assert set(prop["enum"]) == set(constraint._display)


def test_constrained_keys_helper_matches_schema(schema):
    """constrained_ondisk_keys() (used by the drift guard) lines up with the
    actual schema properties."""
    for section, key in constrained_ondisk_keys():
        assert key in _section_props(schema, section)


def test_installer_default_config_validates(validator):
    """The config the installer writes must pass its own schema."""
    cfg = {
        "server": {"host": "127.0.0.1", "port": 8815, "aggressive_dir_pruning": True},
        "cache": {
            "backend": "file",
            "file_max_segment_mb": 256,
            "file_max_total_gb": 128,
        },
        "sources": [{"url": "/data", "monitor": True}],
    }
    assert list(validator.iter_errors(cfg)) == []


@pytest.mark.parametrize(
    "cfg",
    [
        {"pyramid": {"downscale_factor": 2}},  # boundary (min)
        {"precache": {"backlog_high_water": 0.0}},  # boundary (min)
        {"precache": {"backlog_high_water": 1.0}},  # boundary (max)
        {"server": {"port": 65535}},  # boundary (max)
        {"server": {"log_level": "info"}},  # case-insensitive enum stays lenient
        {"cache": {"backend": "memory"}},
        {"compute": {"backend": "gpu"}},
        {"sources": [{"url": "/d", "type": "ome-zarr"}]},
        {"server": {"future_unknown_knob": 7}},  # additionalProperties: true
    ],
)
def test_accepts_valid(validator, cfg):
    assert list(validator.iter_errors(cfg)) == [], f"unexpectedly rejected: {cfg}"


@pytest.mark.parametrize(
    "cfg",
    [
        {"server": {"port": 0}},
        {"server": {"port": 70000}},
        {"cache": {"backend": "bogus"}},
        {"compute": {"backend": "bogus"}},  # case-sensitive enum
        {"pyramid": {"downscale_factor": 1}},  # silently single-level before
        {"pyramid": {"downscale_factor": 0}},  # ZeroDivisionError before
        {"pyramid": {"pixel_budget_cubic_root": 0}},  # infinite loop before
        {"precache": {"backlog_high_water": 2}},
        {"cache": {"file_max_total_gb": 0}},
        {"sources": [{"type": "zarr"}]},  # missing required url
        {"sources": [{"url": "/d", "type": "madeup"}]},  # type not in enum
    ],
)
def test_rejects_invalid(validator, cfg):
    assert list(validator.iter_errors(cfg)), f"should have been rejected: {cfg}"


def test_schema_matches_dataclass_validation_on_known_bad():
    """A value the schema rejects is also one the dataclass validator flags --
    the two share _CONSTRAINTS, so they must agree. Spot-checked on the pyramid
    downscale_factor=1 case (its own _Range(min=2))."""
    c = _CONSTRAINTS["PyramidConfig"]["downscale_factor"]
    assert isinstance(c, _Range) and c.min == 2
    assert not c.ok(1)  # dataclass side rejects
    # schema side rejects (covered by test_rejects_invalid); this ties them.


def test_jsonschema_importable():
    """The schema test relies on jsonschema being in the [test] extra."""
    assert jsonschema.__name__ == "jsonschema"
