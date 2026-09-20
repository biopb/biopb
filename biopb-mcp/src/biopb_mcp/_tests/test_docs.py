"""The knowledge store: ids, the index, reading, writing (`mcp/_docs.py`).

These exercise the runtime reader and writer against a temporary store. The
shipped seed has its own suite (`_tests/docs/`), which asks a different
question: whether what ships is coherent.
"""

from __future__ import annotations

import pytest

from biopb_mcp.mcp import _docs


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A shipped tier and a local tier, both empty, both redirected here."""
    shipped = tmp_path / "shipped"
    local = tmp_path / "local"
    shipped.mkdir()
    monkeypatch.setattr(_docs, "_shipped_root", lambda: shipped)
    monkeypatch.setattr(_docs, "local_dir", lambda: local)
    return shipped, local


def ship(store, doc_id: str, body: str = "# Title\n\nProse.\n", **front):
    shipped, _ = store
    path = shipped / f"{doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"{k}: {v}\n" for k, v in front.items())
    path.write_text(f"---\n{lines}---\n\n{body}" if lines else body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Ids
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "doc_id",
    ["kernel", "my-doc", "a.b", "internals/cache", "a/b/c", "_banked"],
)
def test_an_id_may_be_a_path(doc_id):
    assert _docs.valid_id(doc_id)


@pytest.mark.parametrize(
    "doc_id",
    ["", " ", "..", "../evil", "/abs", "a//b", "a/", "-lead", "a b", "a\\b", "a\x00b"],
)
def test_an_id_that_could_escape_the_tier_is_not_an_id(doc_id):
    """The id becomes a path, so it is validated rather than sanitised: a
    rewrite would silently write somewhere the caller did not name."""
    assert not _docs.valid_id(doc_id)


# --------------------------------------------------------------------------- #
# The shipped tier
# --------------------------------------------------------------------------- #
def test_a_banked_doc_is_unlisted_but_readable(store):
    """`_` is the release's decision not to list, and nothing more: the file
    ships and `read_doc` returns it (knowledge.md §2)."""
    ship(store, "served")
    ship(store, "_banked")
    assert _docs.shipped_ids() == ["served"]
    assert _docs.describe("_banked") is not None
    assert _docs.read_doc("_banked").startswith("_banked — shipped")


def test_a_banked_directory_is_unlisted_too(store):
    ship(store, "kept/one")
    ship(store, "_draft/two")
    assert _docs.shipped_ids() == ["kept/one"]
    assert _docs.describe("_draft/two") is not None


def test_a_hand_written_entry_lists_a_banked_doc(store):
    """One index line promotes it -- which is what keeps banking cheap to undo."""
    ship(store, "index", "- _banked: worth a look after all\n")
    ship(store, "_banked")
    rendered = _docs.render_index()
    assert "- _banked: worth a look after all" in rendered
    assert "(missing)" not in rendered


def test_a_dotfile_is_not_a_doc(store):
    shipped, _ = store
    (shipped / ".DS_Store.md").write_text("junk\n", encoding="utf-8")
    assert _docs.shipped_ids() == []


def test_the_seed_index_is_not_a_doc(store):
    """It is the index, so it must not also appear in the *New shipped* tail."""
    ship(store, "index", "# seed\n")
    ship(store, "real")
    assert _docs.shipped_ids() == ["real"]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_a_file_with_no_frontmatter_still_loads(store):
    ship(store, "bare", "# A title\n\nThe first sentence.\n")
    meta = _docs.describe("bare")
    assert meta["title"] == "A title"
    assert meta["description"] == "The first sentence."


def test_packages_is_read_as_a_list(store):
    ship(store, "needs", packages="[pystackreg~=0.2.8, skan~=0.13.1]")
    assert _docs.describe("needs")["packages"] == ["pystackreg~=0.2.8", "skan~=0.13.1"]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_doc_puts_the_origin_in_a_header(store):
    ship(store, "one")
    assert _docs.read_doc("one").splitlines()[0].startswith("one — shipped")


def test_a_local_doc_shadows_the_shipped_one(store):
    ship(store, "one", "# Shipped\n")
    _docs.write_doc("one", body="# Mine\n")
    text = _docs.read_doc("one")
    assert "shadows shipped" in text.splitlines()[0]
    assert "# Mine" in text and "# Shipped" not in text


def test_an_unknown_id_says_so_rather_than_raising(store):
    assert "No doc 'nope'" in _docs.read_doc("nope")


def test_a_shadow_of_a_shipped_doc_keeps_its_frontmatter(store):
    """The store classifies nothing, so a shadow is the shipped text edited and
    nothing about it changes but the origin."""
    ship(store, "ref", "# Ref\n\nProse.\n", description="the ref")
    _docs.write_doc("ref", old="Prose.", new="Prose, edited.")
    meta = _docs.describe("ref")
    assert meta["description"] == "the ref"
    assert meta["origin"] == "local" and meta["shadows_shipped"]


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
def test_the_index_is_seeded_from_the_package_on_first_run(store):
    ship(store, "index", "# seed\n\n- a: hook\n")
    _, local = store
    assert "# seed" in _docs.index_text()
    assert (local / "index.md").read_text(encoding="utf-8").startswith("# seed")


def test_an_entry_with_no_file_is_marked_missing(store):
    ship(store, "index", "- gone: hook\n")
    assert "- gone: hook (missing)" in _docs.render_index()


def test_a_shadowed_entry_is_marked_a_local_copy(store):
    ship(store, "index", "- one: hook\n")
    ship(store, "one")
    _docs.write_doc("one", body="# Mine\n")
    assert "- one: hook (local copy)" in _docs.render_index()


def test_a_shipped_doc_the_index_never_mentions_reaches_the_tail(store):
    ship(store, "index", "- one: hook\n")
    ship(store, "one")
    ship(store, "two")
    assert "New shipped docs: two" in _docs.render_index()


def test_an_ignored_doc_is_not_in_the_tail_and_is_still_readable(store):
    ship(store, "index", "- one: hook\n\nignored: two\n")
    ship(store, "one")
    ship(store, "two")
    rendered = _docs.render_index()
    assert "New shipped docs" not in rendered
    assert _docs.read_doc("two").startswith("two — shipped")


def test_a_collection_line_is_kept_verbatim(store):
    """Reserved for knowledge.md §9: it names no file today and must not be
    reported as missing, or a seed written for a later release reads as broken."""
    ship(store, "index", "- internals/: the engine's own design docs\n")
    rendered = _docs.render_index()
    assert "- internals/: the engine's own design docs" in rendered
    assert "(missing)" not in rendered


def test_a_prose_bullet_in_entry_shape_is_an_entry(store):
    """The one bullet shape the loader claims. Documented in the seed preamble
    and the handshake header rather than guessed around."""
    ship(store, "index", "- Remember: check the scale\n")
    assert "- Remember: check the scale (missing)" in _docs.render_index()
    assert _docs.index_entry_count("- Remember: x\n- _banked: y\n") == 2


def test_prose_and_headings_pass_through(store):
    ship(store, "index", "# Title\n\nSome prose.\n\n## A heading\n\n- **Note**: bold\n")
    rendered = _docs.render_index()
    for line in ("# Title", "Some prose.", "## A heading", "- **Note**: bold"):
        assert line in rendered


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_a_new_doc_is_filed_in_the_index(store):
    ship(store, "index", "# docs\n")
    result = _docs.write_doc("mine", body="# Mine\n\nWhat it does.\n")
    assert "Filed in the index" in result
    assert "- mine: What it does." in _docs.index_text()
    assert _docs._UNFILED in _docs.index_text()


def test_a_second_write_does_not_file_it_again(store):
    ship(store, "index", "# docs\n")
    _docs.write_doc("mine", body="# Mine\n")
    _docs.write_doc("mine", body="# Mine again\n")
    assert _docs.index_text().count("- mine:") == 1


def test_shadowing_a_listed_shipped_doc_does_not_add_a_second_entry(store):
    ship(store, "index", "- one: hook\n")
    ship(store, "one")
    _docs.write_doc("one", old="# Title", new="# Mine")
    assert _docs.index_text().count("- one:") == 1


def test_a_write_to_a_shipped_id_leaves_the_shipped_file_alone(store):
    path = ship(store, "one", "# Shipped\n\nBody.\n")
    _docs.write_doc("one", old="Body.", new="Edited.")
    assert "Body." in path.read_text(encoding="utf-8")
    assert "Edited." in _docs.read_doc("one")


def test_a_replace_that_matches_nothing_changes_nothing(store):
    ship(store, "one")
    assert "not in" in _docs.write_doc("one", old="absent", new="x")
    assert _docs.describe("one")["origin"] == "shipped"


def test_a_replace_that_matches_twice_is_refused(store):
    ship(store, "one", "# T\n\nsame\nsame\n")
    assert "appears 2 times" in _docs.write_doc("one", old="same", new="x")


def test_the_result_is_a_diff(store):
    ship(store, "one", "# T\n\nbefore\n")
    result = _docs.write_doc("one", old="before", new="after")
    assert "-before" in result and "+after" in result


def test_a_body_over_the_cap_is_refused(store):
    body = "\n".join(f"line {n}" for n in range(_docs.MAX_BODY_LINES + 1))
    assert "the cap is" in _docs.write_doc("long", body=body)
    assert _docs.describe("long") is None


def test_an_index_over_the_entry_cap_is_refused(store):
    entries = "\n".join(f"- doc{n}: hook" for n in range(_docs.MAX_INDEX_ENTRIES + 1))
    assert "entries; the cap is" in _docs.write_doc(_docs.INDEX_ID, body=entries)


def test_writing_the_index_does_not_file_the_index(store):
    result = _docs.write_doc(_docs.INDEX_ID, body="# docs\n\n- a: hook\n")
    assert "Filed in the index" not in result
    assert "- index:" not in _docs.index_text()


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "Pass body"),
        ({"old": "a"}, "Pass body"),
        ({"body": "x", "old": "a", "new": "b"}, "not both"),
    ],
)
def test_the_two_forms_are_not_mixed(store, kwargs, expected):
    assert expected in _docs.write_doc("one", **kwargs)


def test_a_collection_id_is_reserved(store):
    assert "not a doc" in _docs.write_doc("internals/", body="x")


def test_an_invalid_id_is_refused_by_name(store):
    assert "not a doc id" in _docs.write_doc("../escape", body="x")


def test_editing_a_doc_that_does_not_exist_says_to_create_it(store):
    assert "Pass body to create it" in _docs.write_doc("nope", old="a", new="b")


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def test_the_status_line_reports_the_resolved_dir(store):
    _, local = store
    assert str(local) in _docs.local_dir_status()
    assert "not created yet" in _docs.local_dir_status()
    _docs.write_doc("mine", body="# Mine\n")
    assert "1 local doc" in _docs.local_dir_status()


def test_the_status_line_names_a_legacy_skills_dir(store):
    """Nothing reads `~/.config/biopb/skills` any more, so a user who had files
    there would otherwise find them silently gone."""
    _, local = store
    local.mkdir(parents=True, exist_ok=True)
    legacy = local.parent / "skills"
    legacy.mkdir()
    (legacy / "old.md").write_text("# old\n", encoding="utf-8")
    assert "legacy skills dir" in _docs.local_dir_status()
