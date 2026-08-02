"""The validator that gates every skills change, exercised on synthetic trees.

Two things are being pinned: which malformations are fatal (an error the author
sees in their PR) versus tolerable (a warning that still ships), and that the
entry it yields is canonical regardless of how the file was authored.
"""

from __future__ import annotations

import pytest

from ._schema import CURRENT_SPEC_VERSION, REQUIRED_SECTIONS
from ._validate import first_h1, h2_sections, validate
from .conftest import make_body, write_skill


def messages(rep) -> str:
    return "\n".join(rep.errors + rep.warnings)


# --- the happy path -------------------------------------------------------


def test_wellformed_skill_yields_one_clean_entry(skill_factory, skills_dir):
    skill_factory("my-skill")
    entries, rep = validate(skills_dir)
    assert rep.errors == []
    assert [e.id for e in entries] == ["my-skill"]
    assert entries[0].spec_version == CURRENT_SPEC_VERSION


def test_tags_are_lowercased_and_a_string_is_split(skills_dir):
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter=(
            "description: A sentence.\ntitle: T\nversion: 1.0.0\ntags: Measurement, QC\n"
        ),
    )
    entries, rep = validate(skills_dir)
    assert rep.errors == []
    assert entries[0].tags == ["measurement", "qc"]


def test_id_defaults_to_the_filename_stem(skills_dir):
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter="description: A sentence.\ntitle: T\nversion: 1.0.0\n",
    )
    entries, rep = validate(skills_dir)
    assert rep.errors == []
    assert entries[0].id == "my-skill"


# --- what is fatal --------------------------------------------------------


def test_id_disagreeing_with_the_filename_is_an_error(skills_dir):
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter=(
            "id: other-skill\ndescription: A sentence.\ntitle: T\nversion: 1.0.0\n"
        ),
    )
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "must equal filename stem" in messages(rep)


def test_missing_description_is_an_error(skills_dir):
    write_skill(skills_dir, "my-skill", frontmatter="title: T\nversion: 1.0.0\n")
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "missing required field: description" in messages(rep)


def test_non_semver_version_is_an_error(skills_dir):
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter="description: A sentence.\ntitle: T\nversion: 1.0\n",
    )
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "MAJOR.MINOR.PATCH" in messages(rep)


@pytest.mark.parametrize("dropped", REQUIRED_SECTIONS)
def test_each_required_section_is_individually_required(skills_dir, dropped):
    keep = tuple(s for s in REQUIRED_SECTIONS if s != dropped)
    write_skill(skills_dir, "my-skill", body=make_body(sections=keep))
    entries, rep = validate(skills_dir)
    assert entries == []
    assert dropped in messages(rep)


def test_missing_frontmatter_is_an_error(skills_dir):
    write_skill(skills_dir, "my-skill", raw="# Just a heading\n\nProse.\n")
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "frontmatter" in messages(rep)


def test_unparseable_yaml_is_an_error_not_a_crash(skills_dir):
    write_skill(skills_dir, "my-skill", raw="---\ndescription: [unclosed\n---\n\n# H\n")
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "YAML parse error" in messages(rep)


def test_scalar_frontmatter_is_an_error(skills_dir):
    write_skill(skills_dir, "my-skill", raw="---\njust a string\n---\n\n# H\n")
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "not a mapping" in messages(rep)


def test_non_kebab_filename_is_an_error(skills_dir):
    write_skill(skills_dir, "My_Skill")
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "kebab-case" in messages(rep)


def test_empty_body_is_an_error(skills_dir):
    write_skill(skills_dir, "my-skill", body="")
    entries, rep = validate(skills_dir)
    assert entries == []
    assert "empty body" in messages(rep)


def test_ids_are_unique_because_they_are_pinned_to_the_filename(skills_dir):
    """Two files cannot collide on `id`: one of them must disagree with its own
    stem, and that is caught first. The duplicate-id branch in validate() is a
    backstop, not the mechanism -- this pins the mechanism."""
    write_skill(skills_dir, "a-skill")
    write_skill(
        skills_dir,
        "b-skill",
        frontmatter="id: a-skill\ndescription: A sentence.\ntitle: T\nversion: 1.0.0\n",
    )
    entries, rep = validate(skills_dir)
    assert [e.id for e in entries] == ["a-skill"]
    assert "must equal filename stem" in messages(rep)


# --- what is merely a warning --------------------------------------------


def test_missing_title_is_inferred_from_the_h1_and_only_warns(skills_dir):
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter="description: A sentence.\nversion: 1.0.0\n",
        body=make_body(h1="Inferred Title"),
    )
    entries, rep = validate(skills_dir)
    assert rep.errors == []
    assert entries[0].title == "Inferred Title"
    assert "title inferred" in messages(rep)


def test_title_falls_back_to_the_stem_when_there_is_no_h1(skills_dir):
    body = "\n".join(f"## {s.title()}\n\nProse.\n" for s in REQUIRED_SECTIONS)
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter="description: A sentence.\nversion: 1.0.0\n",
        body=body,
    )
    entries, rep = validate(skills_dir)
    assert rep.errors == []
    assert entries[0].title == "My Skill"
    assert "no H1 heading" in messages(rep)


def test_future_spec_version_warns_and_is_clamped(skills_dir):
    write_skill(
        skills_dir,
        "my-skill",
        frontmatter=(
            f"description: A sentence.\ntitle: T\nversion: 1.0.0\n"
            f"spec_version: {CURRENT_SPEC_VERSION + 5}\n"
        ),
    )
    entries, rep = validate(skills_dir)
    assert rep.errors == []
    assert entries[0].spec_version == CURRENT_SPEC_VERSION


# --- traversal ------------------------------------------------------------


@pytest.mark.parametrize("stem", ["README", "ROADMAP"])
def test_prose_docs_beside_the_skills_are_skipped(skills_dir, stem, skill_factory):
    skill_factory("my-skill")
    write_skill(skills_dir, stem, raw="# Not a skill\n\nNo frontmatter here.\n")
    entries, rep = validate(skills_dir)
    assert [e.id for e in entries] == ["my-skill"]
    assert rep.errors == []


def test_one_bad_file_does_not_suppress_the_others(skills_dir):
    """Regression: errors were accumulated globally and checked with a bare
    `if errors:`, so every file after the first bad one silently produced no
    entry -- and reported nothing about itself."""
    write_skill(skills_dir, "a-bad", frontmatter="title: T\nversion: 1.0.0\n")
    write_skill(skills_dir, "z-good")
    entries, rep = validate(skills_dir)
    assert [e.id for e in entries] == ["z-good"]
    assert len(rep.errors) == 1


def test_validate_is_repeatable_within_one_process(skills_dir, skill_factory):
    """The diagnostics must not be module state: two runs, same answer."""
    skill_factory("my-skill")
    first = validate(skills_dir)
    second = validate(skills_dir)
    assert first[1].errors == second[1].errors == []
    assert [e.id for e in first[0]] == [e.id for e in second[0]]


# --- helpers --------------------------------------------------------------


@pytest.mark.parametrize(
    "heading, normalized",
    [
        ("## When NOT to use", "when not to use"),
        ("## when not to use.", "when not to use"),
        ("## When  NOT   to use :", "when not to use"),
    ],
)
def test_h2_sections_normalizes_case_punctuation_and_whitespace(heading, normalized):
    assert h2_sections(f"{heading}\n\ntext\n") == {normalized}


def test_h2_sections_ignores_deeper_and_shallower_headings():
    body = "# H1\n\n## Steps\n\n### Substep\n\ntext\n"
    assert h2_sections(body) == {"steps"}


def test_first_h1_returns_none_when_absent():
    assert first_h1("## Steps\n\ntext\n") is None


def test_the_old_key_name_is_rejected_on_the_authors_path(skills_dir):
    """`requires:` was renamed to `checklist:`. The runtime reader still accepts the
    old name so a user's own older skill keeps its list, but a shipped one has
    an author and a PR, and two spellings in the catalog is how a grammar rots.
    """
    write_skill(
        skills_dir,
        "old-key",
        frontmatter="description: A sentence.\ntitle: T\nversion: 1.0.0\nrequires: [viewer]\n",
    )
    _, rep = validate(skills_dir)
    assert any("renamed to `checklist:`" in m for m in rep.errors), rep.errors
