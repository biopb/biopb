"""Tests for mcp/_skills.py — curated skills discovery + retrieval.

Fully hermetic, and now trivially so: skills ship as package data and nothing is
fetched, so there is no network to patch. Both sources are redirected into
``tmp_path`` — the shipped set via ``_data_dir``, the user's via
``skills_local_dir`` — so no test can read the real
``~/.config/biopb/skills``. Exercises the two-source merge, the tolerant
readers, frontmatter stripping, and the matcher.
"""

import pytest

from biopb_mcp.mcp import _skills

# The real shipped tree, restored only by the tests that are *about* it. Patched
# out by default: it changes on its own schedule (adding a skill is a content
# change, not a code change), and every exact-id assertion would break with it.
_REAL_DATA_DIR = _skills._data_dir


@pytest.fixture
def mock_home(monkeypatch, tmp_path):
    """Redirect ~ so nothing resolves into the real config tree."""
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture
def skills_cfg(monkeypatch, tmp_path):
    """Control config without touching the real CONFIG singleton / disk, and
    point both skill sources at empty tmp trees.

    ``skills_local_dir`` names a path that does not exist yet, so the default
    suite never reads a real ``~/.config/biopb/skills`` (nor whatever
    ``$BIOPB_CONFIG_HOME`` points at on CI). The ``local_skills`` fixture creates
    it for the tests that want local files; ``_ship`` fills the shipped one.
    """
    cfg = {
        "skills_enabled": True,
        "skills_local_dir": str(tmp_path / "local-skills"),
        # Off by default here so these tests stay about the *skill* sources. It
        # also keeps them hermetic: the plugin scan resolves
        # ``~/.config/biopb/kernel``, and `mcp_plugin_dir` honours
        # ``$BIOPB_CONFIG_HOME`` ahead of the patched home — which CI sets — so a
        # default-on scan would read whatever the machine happens to have
        # seeded. The `plugin_catalog` fixture turns it on against a tmp dir.
        "skills_index_plugins": False,
        "namespace_enabled": True,
    }

    def fake_setting(path, default=None):
        return cfg.get(path.rsplit(".", 1)[1], default)

    empty = tmp_path / "shipped"
    empty.mkdir(exist_ok=True)
    monkeypatch.setattr(_skills, "_setting", fake_setting)
    monkeypatch.setattr(_skills, "_data_dir", lambda: empty)
    return cfg


@pytest.fixture
def real_skills(skills_cfg, monkeypatch):
    """Restore the shipped tree, for the tests that are about it."""
    monkeypatch.setattr(_skills, "_data_dir", _REAL_DATA_DIR)


@pytest.fixture
def local_skills(skills_cfg):
    """The configured local-skills dir, created."""
    import pathlib

    d = pathlib.Path(skills_cfg["skills_local_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ship(monkeypatch, tmp_path, skills):
    """Write *skills* (dicts) as real .md files and ship them.

    Real files through the real reader, rather than a stubbed entry list: the
    frontmatter parse is on the path under test, so a matcher fixture would
    otherwise pass against a shape the loader never produces.
    """
    d = tmp_path / "shipped"
    d.mkdir(exist_ok=True)
    for s in skills:
        fm = [f"id: {s['id']}"]
        if "title" in s:
            fm.append(f"title: {s['title']}")
        if "description" in s:
            fm.append(f"description: {s['description']}")
        if s.get("tags"):
            fm.append("tags: [" + ", ".join(s["tags"]) + "]")
        (d / f"{s['id']}.md").write_text(
            "---\n" + "\n".join(fm) + "\n---\n\n# Body\n\nprose\n", encoding="utf-8"
        )
    monkeypatch.setattr(_skills, "_data_dir", lambda: d)
    return d


# --------------------------------------------------------------------------- #
# Tolerant readers
# --------------------------------------------------------------------------- #
def test_strip_frontmatter():
    body = "---\nid: x\ntitle: T\n---\n\n# Heading\n\ntext"
    assert _skills._strip_frontmatter(body) == "# Heading\n\ntext"
    # no frontmatter -> unchanged (except leading whitespace)
    assert _skills._strip_frontmatter("# H\n") == "# H\n"


def test_parse_frontmatter_tolerates_what_it_cannot_read():
    fm = _skills._parse_frontmatter(
        "---\n"
        'title: "Quoted title"\n'
        "tags: [a, 'b']\n"
        "nested:\n  key: value\n"  # unsupported shape: skipped, not fatal
        "# a comment\n"
        "version: 1.0.0\n"
        "---\n\nbody\n"
    )
    assert fm["title"] == "Quoted title"
    assert fm["tags"] == ["a", "b"]
    assert fm["version"] == "1.0.0"
    assert "key" not in fm  # the nested value never becomes a top-level field


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_find_skills_empty_when_both_sources_are_empty(mock_home, skills_cfg):
    # No shipped files, no local dir: discovery returns nothing rather than
    # raising.
    assert _skills.list_skills("") == []


def test_find_skills_query_filters(mock_home, skills_cfg, monkeypatch, tmp_path):
    _ship(
        monkeypatch,
        tmp_path,
        [
            {
                "id": "segment-cells",
                "title": "Segment cells",
                "description": "instance labels",
                "tags": ["segmentation"],
            },
            {"id": "measure", "title": "Measure", "description": "a table"},
        ],
    )
    assert [s["id"] for s in _skills.list_skills("cells")] == ["segment-cells"]
    assert [s["id"] for s in _skills.list_skills("segmentation")] == ["segment-cells"]
    assert _skills.list_skills("no-such-term-xyz") == []


def test_find_skills_disabled_returns_empty(local_skills, skills_cfg, mock_home):
    # The master switch governs the whole subsystem -- including local files,
    # which are otherwise reachable with no configuration at all.
    (local_skills / "mine.md").write_text(
        "---\ndescription: local one\n---\n\n# Mine\n", encoding="utf-8"
    )
    assert [s["id"] for s in _skills.list_skills("")] == ["mine"]

    skills_cfg["skills_enabled"] = False
    assert _skills.list_skills("") == []


def test_shipped_skill_entry_is_built_from_its_frontmatter(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    # There is no catalog index: everything the agent sees about a shipped skill
    # is read out of the file itself.
    d = _ship(monkeypatch, tmp_path, [{"id": "x", "title": "T", "description": "d"}])
    (d / "x.md").write_text(
        "---\n"
        "id: x\n"
        "title: T\n"
        "description: d\n"
        "tags: [a, b]\n"
        "version: 2.1.0\n"
        "checklist: [viewer, pkg:biopb-mcp>=0.13.0]\n"
        "---\n\n# T\n\nbody\n",
        encoding="utf-8",
    )
    (found,) = _skills.list_skills("")
    assert found["tags"] == ["a", "b"]
    assert found["version"] == "2.1.0"
    assert found["checklist"] == ["viewer", "pkg:biopb-mcp>=0.13.0"]
    assert found["origin"] == "catalog"
    assert found["uri"] == "skill://x"


def test_the_old_key_name_still_reads_for_a_users_own_skill(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    # `requires:` was this key's name, and the tolerant reader is on the agent's
    # path: a local skill written before the rename keeps its list rather than
    # silently resolving nothing. The strict validator rejects it, so nothing
    # shipped can drift back to the old spelling.
    d = _ship(monkeypatch, tmp_path, [{"id": "y", "title": "T", "description": "d"}])
    (d / "y.md").write_text(
        "---\nid: y\ntitle: T\ndescription: d\nrequires: [viewer]\n"
        "---\n\n# T\n\nbody\n",
        encoding="utf-8",
    )
    (found,) = _skills.list_skills("")
    assert found["checklist"] == ["viewer"]


def test_an_empty_shipped_set_warns_because_it_is_always_a_bug(
    mock_home, skills_cfg, monkeypatch, caplog
):
    # Nothing legitimate produces zero shipped skills -- the catalog is package
    # data, so the realistic cause is a packaging regression that keeps the .py
    # files and drops the .md ones. Still not raised (this is the agent's path),
    # but not silent either: before skills shipped, an empty result meant
    # "offline", and quietly returning [] was the right answer.
    monkeypatch.setattr(_skills, "_warned_empty", False)
    with caplog.at_level("WARNING"):
        assert _skills.list_skills("") == []
    assert "packaging problem" in caplog.text


def test_the_empty_warning_does_not_repeat(mock_home, skills_cfg, monkeypatch, caplog):
    # load_catalog() runs on every list_skills; a broken install must not fill
    # the session log.
    monkeypatch.setattr(_skills, "_warned_empty", False)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            _skills.list_skills("")
    assert caplog.text.count("packaging problem") == 1


def test_files_present_but_none_usable_warns_too(
    mock_home, skills_cfg, monkeypatch, tmp_path, caplog
):
    # A different cause with the same consequence, so the same severity.
    d = tmp_path / "shipped"
    (d / "broken.md").write_bytes(b"\xff\xfe\x00 not utf-8")
    monkeypatch.setattr(_skills, "_warned_empty", False)
    with caplog.at_level("WARNING"):
        assert _skills.list_skills("") == []
    assert "none usable" in caplog.text


def test_unreadable_shipped_file_is_skipped_not_fatal(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    d = _ship(monkeypatch, tmp_path, [{"id": "good", "description": "d"}])
    (d / "undecodable.md").write_bytes(b"\xff\xfe\x00 not utf-8")
    (d / "_private.md").write_text("# Private\n\nprose\n", encoding="utf-8")
    (d / "notes.txt").write_text("not markdown", encoding="utf-8")
    assert [s["id"] for s in _skills.list_skills("")] == ["good"]


# --------------------------------------------------------------------------- #
# The shipped set
#
# That it *loads* -- not that it retrieves. Whether the real descriptions answer
# real phrasings is _tests/skills/test_retrieval.py, beside the rest of the
# authoring gate; this file stops at the loader and the matcher.
# --------------------------------------------------------------------------- #
def test_the_shipped_skills_load(mock_home, real_skills):
    # The package always answers list_skills with something -- there is no
    # fetch to fail and no cache to be cold.
    ids = [s["id"] for s in _skills.list_skills("")]
    assert "write-a-skill" in ids, ids
    assert all(s["origin"] == "catalog" for s in _skills.list_skills(""))


def test_every_shipped_skill_has_a_readable_body(mock_home, real_skills):
    for s in _skills.list_skills(""):
        body = _skills.get_skill_body(s["id"])
        assert body.startswith("# "), s["id"]
        assert not body.lstrip().startswith("---")  # frontmatter stripped


# --------------------------------------------------------------------------- #
# Body retrieval
# --------------------------------------------------------------------------- #
def test_get_body_unknown_id(mock_home, skills_cfg):
    msg = _skills.get_skill_body("does-not-exist")
    assert "No skill" in msg


def test_get_body_of_a_shipped_skill_strips_frontmatter(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    _ship(monkeypatch, tmp_path, [{"id": "x", "title": "T", "description": "d"}])
    assert _skills.get_skill_body("x") == "# Body\n\nprose\n"


# --------------------------------------------------------------------------- #
# Local (user-authored) skills
# --------------------------------------------------------------------------- #
def test_local_skill_is_discovered_with_frontmatter(local_skills, mock_home):
    (local_skills / "my-workflow.md").write_text(
        "---\n"
        "id: my-workflow\n"
        "title: My workflow\n"
        "description: does the thing\n"
        "tags: [segmentation, workflow]\n"
        "checklist: [viewer]\n"
        "version: 1.2.0\n"
        "---\n"
        "\n# My workflow\n\nSteps here.\n",
        encoding="utf-8",
    )
    (found,) = _skills.list_skills("")
    assert found["id"] == "my-workflow"
    assert found["title"] == "My workflow"
    assert found["description"] == "does the thing"
    assert found["tags"] == ["segmentation", "workflow"]
    assert found["checklist"] == ["viewer"]
    assert found["version"] == "1.2.0"
    assert found["origin"] == "local"
    assert found["updated"]  # from the file mtime
    assert found["uri"] == "skill://my-workflow"


def test_local_skill_without_frontmatter_still_loads(local_skills, mock_home):
    # Requiring well-formed frontmatter would be the friction that kills the
    # feature: id falls back to the stem, title to the H1, description to the
    # first prose line.
    (local_skills / "bare-notes.md").write_text(
        "# Bare notes\n\nHow I do the thing.\n", encoding="utf-8"
    )
    (found,) = _skills.list_skills("")
    assert found["id"] == "bare-notes"
    assert found["title"] == "Bare notes"
    assert found["description"] == "How I do the thing."


def test_local_skill_bad_file_is_skipped_not_fatal(local_skills, mock_home):
    (local_skills / "good.md").write_text("# Good\n\nprose\n", encoding="utf-8")
    (local_skills / "undecodable.md").write_bytes(b"\xff\xfe\x00 not utf-8")
    (local_skills / "_private.md").write_text("# Private\n\nprose\n", encoding="utf-8")
    (local_skills / "notes.txt").write_text("not markdown", encoding="utf-8")
    assert [s["id"] for s in _skills.list_skills("")] == ["good"]


def test_local_skill_shadows_shipped_entry(
    local_skills, mock_home, monkeypatch, tmp_path
):
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "shared", "title": "Curated", "description": "the shipped one"}],
    )
    (local_skills / "shared.md").write_text(
        "---\ndescription: my edited copy\n---\n\n# Mine\n\nbody\n", encoding="utf-8"
    )
    (found,) = _skills.list_skills("")
    assert found["description"] == "my edited copy"
    assert found["origin"] == "local"


def test_local_and_shipped_skills_merge(local_skills, mock_home, monkeypatch, tmp_path):
    # Local files are a second *source*, not an override tier: the shipped set
    # must not hide them, and they must not hide it.
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "curated", "title": "Curated", "description": "d"}],
    )
    (local_skills / "mine.md").write_text("# Mine\n\nprose\n", encoding="utf-8")
    found = {s["id"]: s["origin"] for s in _skills.list_skills("")}
    assert found == {"curated": "catalog", "mine": "local"}


def test_local_body_is_read_fresh_every_time(local_skills, mock_home):
    # An edit is live on the next read, which is what makes the authoring loop
    # usable -- and, since skills otherwise arrive only with a release, what
    # makes the local dir a real escape hatch.
    path = local_skills / "draft.md"
    path.write_text("---\ndescription: d\n---\n\n# Draft\n\nfirst\n", encoding="utf-8")
    assert _skills.get_skill_body("draft") == "# Draft\n\nfirst\n"

    path.write_text("---\ndescription: d\n---\n\n# Draft\n\nsecond\n", encoding="utf-8")
    assert _skills.get_skill_body("draft") == "# Draft\n\nsecond\n"

    # Deleting the file retracts the skill on the next call, for the same reason.
    path.unlink()
    assert _skills.list_skills("") == []


def test_local_body_gone_at_read_time_reports_instead_of_raising(
    skills_cfg, monkeypatch, tmp_path
):
    # The narrow race the fresh-read design leaves open: catalogued by a scan,
    # then removed before the body read. Must degrade to a message, not raise.
    entry = {
        "id": "gone",
        "title": "Gone",
        "description": "d",
        "tags": [],
        "version": "",
        "checklist": [],
        "updated": "",
        "origin": "local",
        "_path": str(tmp_path / "gone.md"),
    }
    monkeypatch.setattr(_skills, "load_catalog", lambda: [entry])
    assert "could not be read" in _skills.get_skill_body("gone")


def test_local_dir_defaults_under_config_dir(monkeypatch, tmp_path):
    # An unset skills_local_dir resolves to ~/.config/biopb/skills. CI sets
    # BIOPB_CONFIG_HOME, which config_dir() honors first, so isolate both.
    import pathlib

    monkeypatch.delenv("BIOPB_CONFIG_HOME", raising=False)
    monkeypatch.delenv("BIOPB_CONFIG_DIR", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(_skills, "_setting", lambda path, default=None: "")
    assert _skills._local_dir() == tmp_path / ".config" / "biopb" / "skills"


# --------------------------------------------------------------------------- #
# Retrieval — matcher semantics
#
# A skill nobody retrieves is absent, so what `query` means is a contract in its
# own right. These use synthetic skills: they are about the matcher, and must
# not move when a shipped description is reworded. Whether a real description
# answers the phrasings a user would type is the other half, and lives in
# _tests/skills/ beside the rest of the authoring gate.
# --------------------------------------------------------------------------- #
def test_every_term_must_match_not_just_one(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    # AND, not OR: a skill matching only half the query is not a hit, or every
    # multi-word query would drag in most of the catalog.
    _ship(
        monkeypatch,
        tmp_path,
        [
            {"id": "a", "title": "Stitch tiles", "description": "a mosaic"},
            {"id": "b", "title": "Stitch nothing", "description": "unrelated"},
        ],
    )
    assert [s["id"] for s in _skills.list_skills("stitch mosaic")] == ["a"]


def test_terms_match_in_any_order_and_across_fields(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    # The whole point of the change: "stitch" in the title and "tiles" in the
    # description is a hit, which whole-query substring matching could not do.
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "a", "title": "Stitch a grid", "description": "overlapping tiles"}],
    )
    assert [s["id"] for s in _skills.list_skills("stitch tiles")] == ["a"]
    assert [s["id"] for s in _skills.list_skills("tiles stitch")] == ["a"]


def test_query_matches_the_skill_id(mock_home, skills_cfg, monkeypatch, tmp_path):
    # Naming the skill is the most specific request there is; matching only
    # prose would miss it.
    _ship(
        monkeypatch,
        tmp_path,
        [
            {
                "id": "flatfield-and-stitch",
                "title": "Correct and join",
                "description": "d",
            }
        ],
    )
    assert [s["id"] for s in _skills.list_skills("flatfield")] == [
        "flatfield-and-stitch"
    ]


def test_query_is_case_insensitive(mock_home, skills_cfg, monkeypatch, tmp_path):
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "a", "title": "Segment Nuclei", "description": "DAPI channel"}],
    )
    assert [s["id"] for s in _skills.list_skills("SEGMENT dapi")] == ["a"]


def test_terms_match_inside_words(mock_home, skills_cfg, monkeypatch, tmp_path):
    # Substring, not token: "measure" is meant to find "measurements".
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "a", "title": "Object measurements", "description": "d"}],
    )
    assert [s["id"] for s in _skills.list_skills("measure")] == ["a"]


def test_matching_a_tag_retrieves(mock_home, skills_cfg, monkeypatch, tmp_path):
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "a", "title": "T", "description": "d", "tags": ["quantification"]}],
    )
    assert [s["id"] for s in _skills.list_skills("quantification")] == ["a"]


def test_a_whole_phrase_that_matched_before_still_matches(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    # Term-wise matching only ever widens: a whole-query substring hit implies
    # every term hits, so nothing that used to retrieve can stop retrieving.
    _ship(
        monkeypatch,
        tmp_path,
        [{"id": "a", "title": "T", "description": "segment nuclei in 3d"}],
    )
    assert [s["id"] for s in _skills.list_skills("segment nuclei")] == ["a"]


def test_unmatched_query_returns_empty_not_everything(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    _ship(monkeypatch, tmp_path, [{"id": "a", "title": "T", "description": "d"}])
    assert _skills.list_skills("stitch") == []


def test_empty_and_whitespace_query_return_the_whole_catalog(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    _ship(
        monkeypatch,
        tmp_path,
        [
            {"id": "a", "title": "A", "description": "d"},
            {"id": "b", "title": "B", "description": "d"},
        ],
    )
    assert len(_skills.list_skills("")) == 2
    assert len(_skills.list_skills("   ")) == 2


def test_results_are_sorted_by_title(mock_home, skills_cfg, monkeypatch, tmp_path):
    # Case-insensitively, so a lowercase title does not sort after every
    # capitalised one.
    _ship(
        monkeypatch,
        tmp_path,
        [
            {"id": "c", "title": "zebra", "description": "shared"},
            {"id": "a", "title": "Apple", "description": "shared"},
            {"id": "b", "title": "mango", "description": "shared"},
        ],
    )
    assert [s["title"] for s in _skills.list_skills("shared")] == [
        "Apple",
        "mango",
        "zebra",
    ]


# --------------------------------------------------------------------------- #
# Kernel plugins as catalog rows (#92 discovery)
# --------------------------------------------------------------------------- #
@pytest.fixture
def plugin_catalog(skills_cfg, monkeypatch, tmp_path):
    """Turn plugin indexing on, pointed at a tmp plugin dir this test owns.

    Returns the dir, so a test writes ``*.py`` into it and calls
    ``list_skills``. ``mcp_plugin_dir`` is patched at its source rather than via
    ``$HOME``, because it reads ``$BIOPB_CONFIG_HOME`` first.
    """
    import biopb._locations as locations

    skills_cfg["skills_index_plugins"] = True
    d = tmp_path / "kernel-plugins"
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(locations, "mcp_plugin_dir", lambda: d)
    return d


def _plugin(d, name, doc):
    (d / f"{name}.py").write_text(
        f'"""{doc}"""\n\n\ndef go():\n    return 1\n', encoding="utf-8"
    )


def test_a_plugin_is_discoverable_by_its_docstring(mock_home, plugin_catalog):
    _plugin(plugin_catalog, "rolling_ball", "Rolling-ball background subtraction.")
    (row,) = _skills.list_skills(["rolling"])
    assert row["id"] == "rolling_ball"
    assert row["kind"] == "plugin"
    assert row["description"] == "Rolling-ball background subtraction."


def test_a_plugin_row_offers_a_handle_and_no_skill_uri(mock_home, plugin_catalog):
    """The two are used differently, so they must not look alike: an agent that
    reads `skill://rolling_ball` gets nothing, and one that calls a skill's id
    as a namespace handle gets a NameError."""
    _plugin(plugin_catalog, "rolling_ball", "Rolling-ball background subtraction.")
    (row,) = _skills.list_skills(["rolling"])
    assert row["handle"] == "rolling_ball"
    assert "uri" not in row


def test_a_skill_row_still_carries_its_uri_and_kind(
    mock_home, skills_cfg, monkeypatch, tmp_path
):
    _ship(monkeypatch, tmp_path, [{"id": "flatfield", "description": "even out"}])
    (row,) = _skills.list_skills(["flatfield"])
    assert row["kind"] == "skill"
    assert row["uri"] == "skill://flatfield"
    assert "handle" not in row


def test_the_opening_paragraph_is_searched_not_just_the_first_line(
    mock_home, plugin_catalog
):
    """A module's first line names the subject and often not the verb, so a
    two-word query would miss it on the second term."""
    _plugin(
        plugin_catalog,
        "chunked_label",
        "Connected components on a chunked dask array.\n\n"
        "Labels are linked across every chunk seam.\n\n"
        "Implementation notes nobody should retrieve on: pixiedust.",
    )
    assert [s["id"] for s in _skills.list_skills(["seam"])] == ["chunked_label"]
    # ...and it stops at the *first* blank line after that, so the rest of a long
    # docstring is not dumped into the haystack.
    assert _skills.list_skills(["pixiedust"]) == []


def test_an_undocumented_plugin_is_still_listed(mock_home, plugin_catalog):
    """Excluding it would hide a working third-party tool entirely. The row says
    plainly that it has to be inspected."""
    (plugin_catalog / "mystery.py").write_text("def go():\n    return 1\n")
    (row,) = _skills.list_skills(["mystery"])
    assert row["kind"] == "plugin"
    assert "inspect_object" in row["description"]


def test_an_entry_point_installed_since_startup_is_listed(
    mock_home, plugin_catalog, monkeypatch
):
    """The entry-point scan is read fresh, like every other catalog source.

    `entry_points()` re-scans in a live process, so `pip install`ing a plugin
    package is visible without restarting this one -- and the next
    `restart_kernel` *will* bind it, because a new kernel is a new process.
    Caching the scan here made `list_skills` report a set `server_status`'s
    `## Kernel plugins` had already picked up, so the two disagreed about the
    same plugin.
    """
    import biopb._kernel_plugins as kp

    installed: list[dict] = []
    monkeypatch.setattr(kp, "entry_point_plugins", lambda: list(installed))

    assert _skills.list_skills(["labshop"]) == []

    installed.append({"name": "labshop", "dist": "labshop-tools 1.0"})

    (row,) = _skills.list_skills(["labshop"])
    assert row["kind"] == "plugin"
    assert row["handle"] == "labshop"


def test_private_files_are_not_plugins(mock_home, plugin_catalog):
    _plugin(plugin_catalog, "_helpers", "Not a plugin.")
    assert _skills.list_skills([]) == []


def test_plugins_are_off_when_the_catalog_is(mock_home, plugin_catalog, skills_cfg):
    """`--bench-skills=false` sets `skills_enabled: false` and the run then
    asserts the catalog came back empty. A plugin row surviving that would fail
    the assertion *and* hand the ablated arm back a form of discovery."""
    _plugin(plugin_catalog, "rolling_ball", "Rolling-ball background subtraction.")
    skills_cfg["skills_enabled"] = False
    assert _skills.list_skills([]) == []


def test_plugins_are_off_when_they_will_not_be_loaded(
    mock_home, plugin_catalog, skills_cfg
):
    """Nothing binds them into the namespace, so the handle would not resolve."""
    _plugin(plugin_catalog, "rolling_ball", "Rolling-ball background subtraction.")
    skills_cfg["namespace_enabled"] = False
    assert _skills.list_skills([]) == []


def test_a_skill_shadows_a_plugin_of_the_same_name(
    mock_home, plugin_catalog, monkeypatch, tmp_path
):
    """One id, one row — otherwise `skill://<id>` is ambiguous. The reviewed
    artifact wins."""
    _plugin(plugin_catalog, "flatfield", "A plugin that happens to share the name.")
    _ship(monkeypatch, tmp_path, [{"id": "flatfield", "description": "the skill"}])
    (row,) = _skills.list_skills(["flatfield"])
    assert row["kind"] == "skill"


def test_skills_sort_before_plugins(mock_home, plugin_catalog, monkeypatch, tmp_path):
    """A curated procedure is the better answer when both match, and that should
    not depend on the alphabet."""
    _plugin(plugin_catalog, "aaa_plugin", "shared term here.")
    _ship(monkeypatch, tmp_path, [{"id": "zzz-skill", "description": "shared term"}])
    assert [s["kind"] for s in _skills.list_skills(["shared"])] == ["skill", "plugin"]


def test_reading_a_plugin_as_a_skill_body_explains_itself(mock_home, plugin_catalog):
    """Reachable, because list_skills lists plugins beside skills and reading
    `skill://<id>` is the habit that row sits next to."""
    _plugin(plugin_catalog, "rolling_ball", "Rolling-ball background subtraction.")
    body = _skills.get_skill_body("rolling_ball")
    assert "not a skill" in body
    assert "inspect_object('rolling_ball')" in body


def test_an_unreadable_plugin_dir_is_not_fatal(mock_home, plugin_catalog, monkeypatch):
    import biopb._kernel_plugins as kp

    monkeypatch.setattr(kp, "startup_files", lambda *a, **k: 1 / 0)
    assert _skills.list_skills([]) == []
