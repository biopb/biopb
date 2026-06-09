"""Tests for the tensor-browser metadata-emptiness helpers.

``_is_empty_for_display`` / ``_filter_empty_metadata`` decide which OME metadata
fields the MetadataDialog shows. They are pure functions (no viewer/Qt context
needed beyond importing the module), so this suite runs on every platform —
unlike the viewer-dependent ``test_tensor_browser_widget.py``.
"""

from biopb_mcp.tensor_browser._widget import (
    _filter_empty_metadata,
    _is_empty_for_display,
)


class TestIsEmptyForDisplay:
    def test_none_is_empty(self):
        assert _is_empty_for_display(None) is True

    def test_empty_and_blank_strings_are_empty(self):
        # The str branch (returns bool(isinstance(str) and not strip())).
        assert _is_empty_for_display("") is True
        assert _is_empty_for_display("   ") is True
        assert _is_empty_for_display("\t\n") is True

    def test_nonblank_string_is_not_empty(self):
        assert _is_empty_for_display("x") is False
        assert _is_empty_for_display("  hi  ") is False

    def test_non_string_scalars_are_not_empty(self):
        # Non-str values fall through the str branch to bool(False) -> False,
        # so zero / False are kept (they are meaningful metadata values).
        assert _is_empty_for_display(0) is False
        assert _is_empty_for_display(False) is False
        assert _is_empty_for_display(1.5) is False

    def test_empty_containers_are_empty(self):
        assert _is_empty_for_display([]) is True
        assert _is_empty_for_display({}) is True

    def test_container_of_only_empties_is_empty(self):
        # Recurses: a list/dict whose every value is itself empty is empty.
        assert _is_empty_for_display(["", None, []]) is True
        assert _is_empty_for_display({"a": "", "b": None, "c": {}}) is True

    def test_container_with_any_content_is_not_empty(self):
        assert _is_empty_for_display(["", "x"]) is False
        assert _is_empty_for_display({"a": "", "b": "x"}) is False

    def test_nested_content_is_not_empty(self):
        assert _is_empty_for_display({"a": {"b": ["", "deep"]}}) is False


class TestFilterEmptyMetadata:
    def test_empty_or_falsy_input_returns_empty_dict(self):
        assert _filter_empty_metadata({}) == {}
        assert _filter_empty_metadata(None) == {}

    def test_drops_empty_keeps_meaningful(self):
        meta = {
            "blank": "",
            "whitespace": "   ",
            "none": None,
            "empty_list": [],
            "empty_dict": {},
            "name": "cells.nd2",
            "zero": 0,
            "nested": {"inner": "value"},
        }
        assert _filter_empty_metadata(meta) == {
            "name": "cells.nd2",
            "zero": 0,
            "nested": {"inner": "value"},
        }
