"""Tests for the generated config JSON Schema (biopb/biopb#34).

The schema is *projected from* the server's ``_CONSTRAINTS`` validation table,
so the key property to guard is that it stays in sync: every constrained value
the server enforces at startup must be reflected in the schema, and the schema
must accept what the server accepts and reject what the server rejects.
"""

import dataclasses

import jsonschema
import pytest
from biopb_tensor_server.config import (
    _CONSTRAINTS,
    CacheConfig,
    MetadataDbConfig,
    PrecacheConfig,
    PyramidConfig,
    ServerConfig,
    SourceConfig,
    _Enum,
    _Range,
)
from biopb_tensor_server.config_schema import (
    build_config_schema,
    constrained_ondisk_keys,
    known_config_keys,
    ondisk_location,
)
from biopb_tensor_server.remote import CredentialProfile
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


def test_scalar_defaults_emitted_and_match_dataclasses(schema):
    """Each scalar section field echoes its dataclass default (so a config editor
    can render the effective value of an omitted key); None defaults are omitted."""
    for cls in (
        ServerConfig,
        CacheConfig,
        PyramidConfig,
        PrecacheConfig,
        MetadataDbConfig,
    ):
        inst = cls()
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            value = getattr(inst, f.name)
            if dataclasses.is_dataclass(value) or isinstance(value, list):
                continue
            section, key = ondisk_location(cls.__name__, f.name)
            prop = _section_props(schema, section)[key]
            if value is None:
                assert "default" not in prop, (
                    f"{section}.{key} should omit null default"
                )
            else:
                expected = (
                    value if isinstance(value, (bool, int, float, str)) else str(value)
                )
                assert prop.get("default") == expected, f"{section}.{key} default drift"


def test_top_level_sections_present(schema):
    props = schema["properties"]
    for section in (
        "server",
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
    """The config the installer writes must pass its own schema.

    Mirrors install.sh / biopb-engine.ps1: a fresh install writes cache
    file_max_total_gb=32 and a single source that is either monitored (a user
    data dir) or unmonitored (the static, seeded sample bundle).
    """
    base = {
        "server": {"host": "127.0.0.1", "port": 8815, "aggressive_dir_pruning": True},
        "cache": {
            "backend": "file",
            "file_max_segment_mb": 256,
            "file_max_total_gb": 32,
        },
    }
    # A user data dir is watched; the static sample bundle is not.
    for monitor in (True, False):
        cfg = {**base, "sources": [{"url": "/data", "monitor": monitor}]}
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
        {"sources": [{"url": "/d", "type": "ome-zarr"}]},
        {"server": {"future_unknown_knob": 7}},  # additionalProperties: true
        # Removed [compute] section: tolerated by the schema (root
        # additionalProperties), warn-and-ignore at parse time.
        {"compute": {"backend": "gpu"}},
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


# --- known-key set (the schema is now also the source for #234's warning) -----


def _public_scalar_fields(cls):
    """Public dataclass fields that are scalar config keys (not nested
    sections / source lists)."""
    inst = cls()
    out = []
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        value = getattr(inst, f.name)
        if dataclasses.is_dataclass(value) or isinstance(value, list):
            continue
        out.append(f.name)
    return out


@pytest.mark.parametrize(
    "cls", [ServerConfig, CacheConfig, PyramidConfig, PrecacheConfig, MetadataDbConfig]
)
def test_known_keys_cover_every_dataclass_field(schema, cls):
    """Drift guard for the key set: every scalar field the parser can read is a
    schema property under its on-disk section, so the runtime warning (derived
    from the schema) never mis-warns on a valid key."""
    _, section_keys, _, _ = known_config_keys(schema)
    for field in _public_scalar_fields(cls):
        section, key = ondisk_location(cls.__name__, field)
        assert key in section_keys.get(section, set()), (
            f"{cls.__name__}.{field} -> [{section}].{key} missing from schema"
        )


def test_source_keys_cover_sourceconfig_fields(schema):
    _, _, source_keys, _ = known_config_keys(schema)
    for f in dataclasses.fields(SourceConfig):
        if f.name.startswith("_"):
            continue
        assert f.name in source_keys, f"source field {f.name} missing from schema"
    assert "path" in source_keys  # deprecated alias declared


def test_profile_keys_match_credentialprofile(schema):
    _, _, _, profile_keys = known_config_keys(schema)
    expected = {f.name for f in dataclasses.fields(CredentialProfile)}
    assert profile_keys == expected


def test_legacy_aliases_present_and_marked_deprecated(schema):
    """The aliases #234 accepted silently are now declared (and deprecated) so
    the schema documents them and the warning stays quiet on back-compat configs."""
    server = _section_props(schema, "server")
    assert server["watcher_type"]["deprecated"] is True
    assert server["poll_interval"]["deprecated"] is True
    precache = _section_props(schema, "precache")
    assert precache["downscale_factor"]["deprecated"] is True
    # the back-compat pyramid knob keeps its bound under [precache] too
    assert precache["downscale_factor"]["minimum"] == 2
    assert schema["properties"]["sources"]["items"]["properties"]["path"]["deprecated"]
    assert _section_props(schema, "metadata_db")["enabled"]["deprecated"] is True


def test_runtime_warning_uses_schema_keys():
    """config._warn_unknown_config_keys derives its sets from the schema."""
    from biopb_tensor_server.config import _known_config_keys

    sections, section_keys, source_keys, profile_keys = _known_config_keys()
    assert (sections, section_keys, source_keys, profile_keys) == known_config_keys()
