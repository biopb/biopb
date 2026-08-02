"""The frontmatter contract itself: coercion, the patterns, the emitted shape."""

from __future__ import annotations

import pytest

from ._schema import (
    CURRENT_SPEC_VERSION,
    KEBAB,
    REQUIRED_SECTIONS,
    SEMVER,
    SkillEntry,
    coerce_list,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ([], []),
        (["a", "b"], ["a", "b"]),
        ("a, b", ["a", "b"]),  # comma-separated string
        ("a,,  b , ", ["a", "b"]),  # empty fragments dropped, whitespace stripped
        ("solo", ["solo"]),
        (7, [7]),  # a bare scalar becomes a one-item list
    ],
)
def test_coerce_list_absorbs_authoring_variation(value, expected):
    assert coerce_list(value) == expected


def test_coerce_list_does_not_split_a_list_members():
    # A YAML list is taken as authored; only the string form is split.
    assert coerce_list(["a, b"]) == ["a, b"]


@pytest.mark.parametrize("v", ["0.0.0", "1.0.0", "10.20.30"])
def test_semver_accepts(v):
    assert SEMVER.match(v)


@pytest.mark.parametrize("v", ["1.0", "1", "v1.0.0", "1.0.0-rc1", "1.0.0.0", ""])
def test_semver_rejects(v):
    assert not SEMVER.match(v)


@pytest.mark.parametrize("s", ["a", "skill", "write-a-skill", "b2-c3"])
def test_kebab_accepts(s):
    assert KEBAB.match(s)


@pytest.mark.parametrize(
    "s", ["Skill", "write_a_skill", "-lead", "trail-", "double--dash", "", "has space"]
)
def test_kebab_rejects(s):
    assert not KEBAB.match(s)


def test_required_sections_are_normalized_form():
    # h2_sections() lowercases and strips punctuation before comparing, so the
    # constants must already be in that form or nothing would ever match.
    for s in REQUIRED_SECTIONS:
        assert s == s.lower().strip(" .:")


def test_entry_to_dict_is_complete_and_ordered():
    entry = SkillEntry(
        id="x",
        title="T",
        description="D",
        tags=["t"],
        version="1.0.0",
        spec_version=CURRENT_SPEC_VERSION,
        checklist=["viewer"],
    )
    d = entry.to_dict()
    assert list(d) == [
        "id",
        "title",
        "description",
        "tags",
        "version",
        "spec_version",
        "checklist",
    ]
    assert d["checklist"] == ["viewer"]


def test_the_entry_carries_no_fetch_fields():
    # url/sha256 said where to fetch a body and how to verify it. Skills ship
    # with the package now, so re-adding either would mean the fetch came back.
    fields = SkillEntry.__dataclass_fields__
    assert "url" not in fields
    assert "sha256" not in fields
