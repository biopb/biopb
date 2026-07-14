"""Unit tests for the shared config-schema projection primitives.

:mod:`biopb._config_schema` is the per-field JSON-Schema projection reused by
both biopb-tensor-server and biopb-mcp (biopb#34). Its behavior is pinned here
in the core package; each consumer's composer is tested in its own package.
"""

import dataclasses

from biopb._config_constraints import Enum, Range
from biopb._config_schema import dataclass_section, json_type, scalar_property


class TestJsonType:
    def test_bool_before_int(self):
        # bool is an int subclass; it must map to "boolean", not "integer".
        assert json_type(True) == "boolean"
        assert json_type(3) == "integer"
        assert json_type(1.5) == "number"
        assert json_type("x") == "string"
        assert json_type(None) == "string"


class TestScalarProperty:
    def test_type_and_default(self):
        assert scalar_property(5) == {"type": "integer", "default": 5}
        assert scalar_property(True) == {"type": "boolean", "default": True}

    def test_none_default_omitted(self):
        assert scalar_property(None) == {"type": "string"}

    def test_non_json_native_default_stringified(self):
        from pathlib import Path

        p = scalar_property(Path("/tmp/x"))
        assert p == {"type": "string", "default": "/tmp/x"}

    def test_help_becomes_description(self):
        p = scalar_property(1, "how many")
        assert p["description"] == "how many"

    def test_range_contributes_bounds_and_constraint_key(self):
        p = scalar_property(4, "step", Range(min=2))
        assert p["minimum"] == 2
        assert p["constraint"] == "a number >= 2"
        # description stays pure prose (the rule lives on its own key).
        assert p["description"] == "step"

    def test_case_insensitive_enum_emits_no_hard_enum(self):
        p = scalar_property(
            "INFO", None, Enum({"INFO", "DEBUG"}, case_insensitive=True)
        )
        assert "enum" not in p
        # the accepted set is still surfaced via the constraint key.
        assert "one of:" in p["constraint"]

    def test_case_sensitive_enum_emits_enum(self):
        p = scalar_property("http", None, Enum({"http", "stdio"}))
        assert set(p["enum"]) == {"http", "stdio"}


class TestDataclassSection:
    def test_scalar_fields_projected_with_help_and_constraint(self):
        @dataclasses.dataclass
        class C:
            port: int = 8815
            name: str = field_with_help("srv")
            _private: int = 0

        props = dataclass_section(C, {"port": Range(min=1, max=65535)})
        assert props["port"]["minimum"] == 1 and props["port"]["maximum"] == 65535
        assert props["name"]["description"] == "srv"
        # underscore-prefixed fields are internal, not config keys.
        assert "_private" not in props

    def test_nested_and_list_fields_skipped(self):
        @dataclasses.dataclass
        class Inner:
            a: int = 1

        @dataclasses.dataclass
        class C:
            scalar: int = 1
            servers: list = dataclasses.field(default_factory=list)
            inner: Inner = dataclasses.field(default_factory=Inner)

        props = dataclass_section(C)
        assert set(props) == {"scalar"}

    def test_key_map_remaps_wire_key(self):
        @dataclasses.dataclass
        class C:
            memory_max_bytes: int = 100

        props = dataclass_section(C, key_map={"memory_max_bytes": "max_bytes"})
        assert "max_bytes" in props and "memory_max_bytes" not in props


def field_with_help(help_text: str):
    return dataclasses.field(default="", metadata={"help": help_text})
