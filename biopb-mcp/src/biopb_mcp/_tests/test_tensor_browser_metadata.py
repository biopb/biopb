"""Tests for the tensor-browser metadata-emptiness helpers.

``_is_empty_for_display`` / ``_filter_empty_metadata`` decide which OME metadata
fields the MetadataDialog shows. They are pure functions (no viewer/Qt context
needed beyond importing the module), so this suite runs on every platform —
unlike the viewer-dependent ``test_tensor_browser_widget.py``.
"""

from types import SimpleNamespace

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

from biopb_mcp.tensor_browser._widget import (
    _build_tree,
    _filter_empty_metadata,
    _get_path_parts,
    _is_empty_for_display,
    _residency_state,
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


class TestGetPathParts:
    """``_get_path_parts`` feeds the folder tree; it must split POSIX and
    Windows separators alike so a Windows-indexed catalog (backslash
    ``source_url``) does not collapse into a flat list (issue: browser shows a
    flat list of full-path leaves on Windows)."""

    def test_empty_url(self):
        assert _get_path_parts("") == []
        assert _get_path_parts(None) == []  # type: ignore[arg-type]

    def test_posix_path(self):
        assert _get_path_parts("/data/cells/img.tif") == [
            "data",
            "cells",
            "img.tif",
        ]

    def test_posix_file_uri(self):
        assert _get_path_parts("file:///data/cells/img.tif") == [
            "data",
            "cells",
            "img.tif",
        ]

    def test_windows_path_splits_on_backslash(self):
        # The regression: urlparse reads "C:" as a scheme and leaves the path
        # backslash-separated; splitting on "/" alone yielded one giant leaf.
        assert _get_path_parts(
            r"C:\Users\me\Pictures\Screenshots 1\shot.png"
        ) == ["Users", "me", "Pictures", "Screenshots 1", "shot.png"]

    def test_windows_drive_root_file(self):
        assert _get_path_parts(r"C:\img.tif") == ["img.tif"]

    def test_drive_letter_token_is_dropped(self):
        # Lowercase drive, forward slashes -> urlparse keeps "c:" as the first
        # path token; it must be stripped as a drive, not kept as a folder.
        assert _get_path_parts("c:/data/img.tif") == ["data", "img.tif"]

    def test_mixed_separators(self):
        assert _get_path_parts(r"C:\data/cells\img.tif") == [
            "data",
            "cells",
            "img.tif",
        ]


def _src(source_id, source_url):
    """Minimal stand-in for DataSourceDescriptor (``_build_tree`` reads only
    ``source_id`` and ``source_url``)."""
    return SimpleNamespace(source_id=source_id, source_url=source_url)


class TestBuildTreeWindowsPaths:
    def test_windows_paths_build_folder_hierarchy(self):
        # Two images under a shared Windows folder must nest under it, not sit
        # as two flat full-path leaves at the root.
        sources = {
            "a": _src("a", r"C:\Users\me\Pictures\one.png"),
            "b": _src("b", r"C:\Users\me\Pictures\two.png"),
        }
        root = _build_tree(sources)

        # Single flattened top-level folder ("Users/me/Pictures") with the two
        # images as leaves, rather than two leaves directly under root.
        assert len(root.children) == 1
        folder = root.children[0]
        assert folder.node_type == "folder"
        assert folder.name == "Users/me/Pictures"
        leaf_names = sorted(c.name for c in folder.children)
        assert leaf_names == ["one.png", "two.png"]
        assert all(c.node_type == "source" for c in folder.children)


class TestResidencyState:
    """Tri-state residency indicator: resident / remote / unknown (unset)."""

    def test_unset_is_unknown(self):
        # Old server (field absent) -> None, so the UI shows no indicator.
        src = DataSourceDescriptor(source_id="s")
        assert src.HasField("data_resident") is False
        assert _residency_state(src) is None

    def test_true_is_resident(self):
        src = DataSourceDescriptor(source_id="s", data_resident=True)
        assert _residency_state(src) == "resident"

    def test_false_is_remote(self):
        src = DataSourceDescriptor(source_id="s", data_resident=False)
        # Explicitly set false (present) -> remote, not unknown.
        assert src.HasField("data_resident") is True
        assert _residency_state(src) == "remote"
