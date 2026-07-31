"""Tests for mcp/_skills.py — curated skills discovery + retrieval.

Fully hermetic: the network is always patched, the cache dir is redirected to a
tmp path via ``Path.home``, and ``skills_local_dir`` points into ``tmp_path`` so
no test can read a real ``~/.config/biopb/skills``. Exercises the fail-open
resolution chain (network → disk cache → bundled snapshot), the local-skills
source merged beside it, the tolerant readers, sha integrity, and frontmatter
stripping.
"""

import hashlib
import json

import pytest

from biopb_mcp.mcp import _skills

CATALOG_URL = "https://example.test/skills/catalog.json"

# The shipped snapshot is data that changes on its own schedule (a refresh is a
# data drop, not a code change), so it is patched out by default and restored
# only for the tests that are *about* it -- otherwise every exact-id assertion
# would break the next time a skill is bundled.
_REAL_BUNDLE_TEXT = _skills._bundle_text


@pytest.fixture
def mock_home(monkeypatch, tmp_path):
    """Redirect ~ so the skills cache dir lands under a tmp path."""
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture
def skills_cfg(monkeypatch, tmp_path):
    """Control config without touching the real CONFIG singleton / disk, and
    reset the module-level TTL cache so tests don't leak into each other.

    ``skills_local_dir`` points at a tmp path that does not exist yet, so the
    default suite never reads a real ``~/.config/biopb/skills`` (nor whatever
    ``$XDG_CONFIG_HOME`` points at on CI). The ``local_skills`` fixture creates
    it for the tests that want local files.
    """
    cfg = {
        "skills_enabled": True,
        "skills_catalog_url": "",
        "skills_cache_ttl": 3600,
        "skills_local_dir": str(tmp_path / "local-skills"),
    }

    def fake_setting(path, default=None):
        return cfg.get(path.rsplit(".", 1)[1], default)

    monkeypatch.setattr(_skills, "_setting", fake_setting)
    monkeypatch.setattr(_skills, "_bundle_text", lambda name: None)
    _skills._cache["skills"] = None
    _skills._cache["at"] = 0.0
    return cfg


@pytest.fixture
def real_bundle(skills_cfg, monkeypatch):
    """Restore the shipped snapshot, for the tests that are about it."""
    monkeypatch.setattr(_skills, "_bundle_text", _REAL_BUNDLE_TEXT)


@pytest.fixture
def local_skills(skills_cfg):
    """The configured local-skills dir, created."""
    import pathlib

    d = pathlib.Path(skills_cfg["skills_local_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _offline(monkeypatch):
    def boom(url, timeout):
        raise OSError("offline (test)")

    monkeypatch.setattr(_skills, "_http_get", boom)


def _catalog_bytes(skills):
    return json.dumps({"catalog_version": 1, "skills": skills}).encode()


# --------------------------------------------------------------------------- #
# Tolerant reader
# --------------------------------------------------------------------------- #
def test_accept_catalog_skips_bad_entries_keeps_good():
    raw = _catalog_bytes(
        [
            {"id": "ok", "description": "good"},
            {"id": "no-desc"},  # dropped: missing description
            {"description": "no id"},  # dropped: missing id
            "not-a-dict",  # dropped
            {"id": "coerce", "description": "d", "tags": "x", "requires": None},
        ]
    )
    parsed = _skills._accept_catalog(raw)
    assert [e["id"] for e in parsed] == ["ok", "coerce"]
    # bad-typed optionals coerce to []
    assert parsed[1]["tags"] == []
    assert parsed[1]["requires"] == []


def test_accept_catalog_rejects_unknown_future_version():
    with pytest.raises(ValueError, match="newer than supported"):
        _skills._accept_catalog(
            _catalog_bytes([]).replace(
                b'"catalog_version": 1', b'"catalog_version": 999'
            )
        )


def test_accept_catalog_rejects_non_object():
    with pytest.raises(ValueError, match="not a JSON object"):
        _skills._accept_catalog(b"[1, 2, 3]")


def test_strip_frontmatter():
    body = "---\nid: x\ntitle: T\n---\n\n# Heading\n\ntext"
    assert _skills._strip_frontmatter(body) == "# Heading\n\ntext"
    # no frontmatter -> unchanged (except leading whitespace)
    assert _skills._strip_frontmatter("# H\n") == "# H\n"


# --------------------------------------------------------------------------- #
# Discovery — bundle fallback
# --------------------------------------------------------------------------- #
def test_find_skills_empty_when_every_source_is_empty(
    mock_home, skills_cfg, monkeypatch
):
    # Network down, no catalog URL, no bundle, no local dir: discovery returns
    # nothing rather than raising.
    _offline(monkeypatch)
    assert _skills.find_skills("") == []


def test_find_skills_query_filters(mock_home, skills_cfg, monkeypatch):
    skills_cfg["skills_catalog_url"] = CATALOG_URL
    monkeypatch.setattr(
        _skills,
        "_http_get",
        lambda url, timeout: _catalog_bytes(
            [
                {
                    "id": "segment-cells",
                    "title": "Segment cells",
                    "description": "instance labels",
                    "tags": ["segmentation"],
                },
                {"id": "measure", "title": "Measure", "description": "a table"},
            ]
        ),
    )
    assert [s["id"] for s in _skills.find_skills("cells")] == ["segment-cells"]
    assert [s["id"] for s in _skills.find_skills("segmentation")] == ["segment-cells"]
    assert _skills.find_skills("no-such-term-xyz") == []


def test_find_skills_disabled_returns_empty(
    local_skills, skills_cfg, mock_home, monkeypatch
):
    # The master switch governs the whole subsystem -- including local files,
    # which are otherwise reachable with no network at all.
    _offline(monkeypatch)
    (local_skills / "mine.md").write_text(
        "---\ndescription: local one\n---\n\n# Mine\n", encoding="utf-8"
    )
    assert [s["id"] for s in _skills.find_skills("")] == ["mine"]

    skills_cfg["skills_enabled"] = False
    assert _skills.find_skills("") == []


# --------------------------------------------------------------------------- #
# Discovery — network success + disk cache
# --------------------------------------------------------------------------- #
def test_network_catalog_then_disk_cache(mock_home, skills_cfg, monkeypatch):
    skills_cfg["skills_catalog_url"] = CATALOG_URL
    payload = _catalog_bytes(
        [{"id": "net", "title": "Net", "description": "from network"}]
    )

    monkeypatch.setattr(_skills, "_http_get", lambda url, timeout: payload)
    got = _skills.find_skills("")
    assert [s["id"] for s in got] == ["net"]

    # Now go offline and bust the TTL cache: the on-disk copy must be used
    # (a stale cache beats nothing), NOT the bundle.
    _offline(monkeypatch)
    skills_cfg["skills_cache_ttl"] = 0
    _skills._cache["skills"] = None
    got2 = _skills.find_skills("")
    assert [s["id"] for s in got2] == ["net"]


def test_ttl_cache_avoids_refetch(mock_home, skills_cfg, monkeypatch):
    skills_cfg["skills_catalog_url"] = CATALOG_URL
    calls = {"n": 0}

    def counting_get(url, timeout):
        calls["n"] += 1
        return _catalog_bytes([{"id": "net", "description": "d"}])

    monkeypatch.setattr(_skills, "_http_get", counting_get)
    _skills.find_skills("")
    _skills.find_skills("")  # within TTL -> no second fetch
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Body retrieval
# --------------------------------------------------------------------------- #
def test_bundle_gives_a_fresh_offline_install_the_meta_skill(
    mock_home, skills_cfg, real_bundle, monkeypatch
):
    # An install that has never reached the network still gets the skill that
    # teaches skill-writing -- which is also what makes the local-skills dir
    # discoverable to a user who has no other skills yet.
    _offline(monkeypatch)
    assert [s["id"] for s in _skills.find_skills("")] == ["write-a-skill"]
    body = _skills.get_skill_body("write-a-skill")
    assert body.startswith("# Write a new biopb skill file")
    assert not body.lstrip().startswith("---")  # frontmatter stripped


def test_every_bundled_skill_is_retrievable_by_its_own_name(
    mock_home, skills_cfg, real_bundle, monkeypatch
):
    # The offline agent's first move is a query, not a bare listing, so the
    # test above (find_skills("")) does not cover the path it actually takes.
    # Stated over whatever the bundle holds rather than naming write-a-skill:
    # the bundle is deliberately meta-skills only, and an invariant costs the
    # same one assertion today while covering a second one for free.
    _offline(monkeypatch)
    for s in _skills.load_catalog():
        query = s["id"].replace("-", " ")
        assert [r["id"] for r in _skills.find_skills(query)] == [s["id"]]


def test_bundle_body_sha_matches_its_catalog(mock_home, skills_cfg, real_bundle):
    # The snapshot is refreshed by copying files in; a body updated without its
    # catalog entry (or vice versa) would ship a sha that verifies nothing.
    for entry in _skills.load_catalog():
        raw = _skills._bundle_text(f"{entry['id']}.md")
        assert raw is not None, f"{entry['id']} in bundle catalog but no body"
        assert hashlib.sha256(raw.encode()).hexdigest() == entry["sha256"]


def test_get_body_unknown_id(mock_home, skills_cfg, monkeypatch):
    _offline(monkeypatch)
    msg = _skills.get_skill_body("does-not-exist")
    assert "No skill" in msg


def test_get_body_network_verifies_and_caches(mock_home, skills_cfg, monkeypatch):
    body_url = "https://example.test/skills/net.md"
    raw_body = b"---\nid: net\n---\n\n# Net\n\nbody text"
    sha = hashlib.sha256(raw_body).hexdigest()
    catalog = _catalog_bytes(
        [{"id": "net", "description": "d", "url": body_url, "sha256": sha}]
    )
    skills_cfg["skills_catalog_url"] = CATALOG_URL

    def get(url, timeout):
        return catalog if url == CATALOG_URL else raw_body

    monkeypatch.setattr(_skills, "_http_get", get)
    out = _skills.get_skill_body("net")
    assert out == "# Net\n\nbody text"

    # Cached under the sha; a second read hits the cache even offline.
    cached = _skills._cache_dir() / "bodies" / f"{sha}.md"
    assert cached.exists()
    _offline(monkeypatch)
    assert _skills.get_skill_body("net") == "# Net\n\nbody text"


# --------------------------------------------------------------------------- #
# Local (user-authored) skills
# --------------------------------------------------------------------------- #
def test_local_skill_is_discovered_with_frontmatter(
    local_skills, mock_home, skills_cfg, monkeypatch
):
    _offline(monkeypatch)
    (local_skills / "my-workflow.md").write_text(
        "---\n"
        "id: my-workflow\n"
        "title: My workflow\n"
        "description: does the thing\n"
        "tags: [segmentation, workflow]\n"
        "requires: [viewer]\n"
        "version: 1.2.0\n"
        "---\n"
        "\n# My workflow\n\nSteps here.\n",
        encoding="utf-8",
    )
    (found,) = _skills.find_skills("")
    assert found["id"] == "my-workflow"
    assert found["title"] == "My workflow"
    assert found["description"] == "does the thing"
    assert found["tags"] == ["segmentation", "workflow"]
    assert found["requires"] == ["viewer"]
    assert found["version"] == "1.2.0"
    assert found["origin"] == "local"
    assert found["updated"]  # from the file mtime
    assert found["uri"] == "skill://my-workflow"


def test_local_skill_without_frontmatter_still_loads(
    local_skills, mock_home, skills_cfg, monkeypatch
):
    # Requiring well-formed frontmatter would be the friction that kills the
    # feature: id falls back to the stem, title to the H1, description to the
    # first prose line.
    _offline(monkeypatch)
    (local_skills / "bare-notes.md").write_text(
        "# Bare notes\n\nHow I do the thing.\n", encoding="utf-8"
    )
    (found,) = _skills.find_skills("")
    assert found["id"] == "bare-notes"
    assert found["title"] == "Bare notes"
    assert found["description"] == "How I do the thing."


def test_local_skill_bad_file_is_skipped_not_fatal(
    local_skills, mock_home, skills_cfg, monkeypatch
):
    _offline(monkeypatch)
    (local_skills / "good.md").write_text("# Good\n\nprose\n", encoding="utf-8")
    (local_skills / "undecodable.md").write_bytes(b"\xff\xfe\x00 not utf-8")
    (local_skills / "_private.md").write_text("# Private\n\nprose\n", encoding="utf-8")
    (local_skills / "notes.txt").write_text("not markdown", encoding="utf-8")
    assert [s["id"] for s in _skills.find_skills("")] == ["good"]


def test_local_skill_shadows_catalog_entry(
    local_skills, mock_home, skills_cfg, monkeypatch
):
    skills_cfg["skills_catalog_url"] = CATALOG_URL
    monkeypatch.setattr(
        _skills,
        "_http_get",
        lambda url, timeout: _catalog_bytes(
            [{"id": "shared", "title": "Curated", "description": "from the catalog"}]
        ),
    )
    (local_skills / "shared.md").write_text(
        "---\ndescription: my edited copy\n---\n\n# Mine\n\nbody\n", encoding="utf-8"
    )
    (found,) = _skills.find_skills("")
    assert found["description"] == "my edited copy"
    assert found["origin"] == "local"


def test_local_and_catalog_skills_merge(
    local_skills, mock_home, skills_cfg, monkeypatch
):
    # Local files are a second *source*, not a fourth fallback tier: a reachable
    # catalog must not hide them, and they must not hide it.
    skills_cfg["skills_catalog_url"] = CATALOG_URL
    monkeypatch.setattr(
        _skills,
        "_http_get",
        lambda url, timeout: _catalog_bytes(
            [{"id": "curated", "title": "Curated", "description": "d"}]
        ),
    )
    (local_skills / "mine.md").write_text("# Mine\n\nprose\n", encoding="utf-8")
    found = {s["id"]: s["origin"] for s in _skills.find_skills("")}
    assert found == {"curated": "catalog", "mine": "local"}


def test_local_body_is_read_fresh_every_time(
    local_skills, mock_home, skills_cfg, monkeypatch
):
    # No sha, no body cache: an edit is live on the next read, which is what
    # makes the authoring loop usable.
    _offline(monkeypatch)
    path = local_skills / "draft.md"
    path.write_text("---\ndescription: d\n---\n\n# Draft\n\nfirst\n", encoding="utf-8")
    assert _skills.get_skill_body("draft") == "# Draft\n\nfirst\n"

    path.write_text("---\ndescription: d\n---\n\n# Draft\n\nsecond\n", encoding="utf-8")
    assert _skills.get_skill_body("draft") == "# Draft\n\nsecond\n"

    # Deleting the file retracts the skill on the next call, for the same reason.
    path.unlink()
    assert _skills.find_skills("") == []


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
        "requires": [],
        "updated": "",
        "url": "",
        "sha256": "",
        "origin": "local",
        "_path": str(tmp_path / "gone.md"),
    }
    monkeypatch.setattr(_skills, "load_catalog", lambda **kw: [entry])
    assert "could not be read" in _skills.get_skill_body("gone")


def test_local_dir_defaults_under_config_dir(monkeypatch, tmp_path):
    # An unset skills_local_dir resolves to ~/.config/biopb/skills. CI sets
    # XDG_CONFIG_HOME, which config_dir() honors first, so isolate both.
    import pathlib

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("BIOPB_CONFIG_DIR", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(_skills, "_setting", lambda path, default=None: "")
    assert _skills._local_dir() == tmp_path / ".config" / "biopb" / "skills"


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
# Concurrent-cache safety: atomic writes + sha-verified reads
# --------------------------------------------------------------------------- #
def test_atomic_write_replaces_and_leaves_no_temp(mock_home):
    target = _skills._cache_dir() / "catalog.json"
    _skills._atomic_write(target, b'{"catalog_version": 1, "skills": []}')
    assert target.read_bytes() == b'{"catalog_version": 1, "skills": []}'
    # Overwrite (last-writer-wins), still atomic, no stray temp left behind.
    _skills._atomic_write(target, b"second")
    assert target.read_bytes() == b"second"
    leftovers = [
        p.name
        for p in target.parent.iterdir()
        if p.name.startswith(".tmp-") or p.suffix == ".part"
    ]
    assert leftovers == []


def test_corrupt_cached_body_is_not_trusted_and_refetched(
    mock_home, skills_cfg, monkeypatch
):
    # A truncated/corrupt cached body (e.g. a crash mid-write, or a concurrent
    # session) must be treated as a miss (sha re-verified on read), not returned.
    body_url = "https://example.test/skills/net.md"
    raw_body = b"---\nid: net\n---\n\n# Net\n\nbody text"
    sha = hashlib.sha256(raw_body).hexdigest()
    catalog = _catalog_bytes(
        [{"id": "net", "description": "d", "url": body_url, "sha256": sha}]
    )
    skills_cfg["skills_catalog_url"] = CATALOG_URL

    bodies = _skills._cache_dir() / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    (bodies / f"{sha}.md").write_bytes(b"CORRUPT PARTIAL")  # wrong bytes for this sha

    fetches = {"body": 0}

    def get(url, timeout):
        if url == CATALOG_URL:
            return catalog
        fetches["body"] += 1
        return raw_body

    monkeypatch.setattr(_skills, "_http_get", get)
    out = _skills.get_skill_body("net")
    assert out == "# Net\n\nbody text"  # correct content, not the corrupt bytes
    assert fetches["body"] == 1  # the corrupt cache was rejected, so it refetched
    # ...and the cache was repaired with the correct bytes.
    assert (bodies / f"{sha}.md").read_bytes() == raw_body


# --------------------------------------------------------------------------- #
# Retrieval — matcher semantics
#
# A skill nobody retrieves is absent, so what `query` means is a contract in its
# own right. These use a synthetic catalog: they are about the matcher, and must
# not move when a published description is reworded.
#
# Whether a real description answers the phrasings a user would type is the other
# half, and it is not testable here -- the published skills live in biopb-site,
# on its own release cadence, so a copy of them kept in this repo would go stale
# green. That gate belongs in biopb-site's suite, against the real skills/*.md.
# --------------------------------------------------------------------------- #
def _serve(monkeypatch, skills_cfg, skills):
    """Point discovery at an in-memory catalog."""
    skills_cfg["skills_catalog_url"] = CATALOG_URL
    monkeypatch.setattr(
        _skills, "_http_get", lambda url, timeout: _catalog_bytes(skills)
    )


def test_every_term_must_match_not_just_one(mock_home, skills_cfg, monkeypatch):
    # AND, not OR: a skill matching only half the query is not a hit, or every
    # multi-word query would drag in most of the catalog.
    _serve(
        monkeypatch,
        skills_cfg,
        [
            {"id": "a", "title": "Stitch tiles", "description": "a mosaic"},
            {"id": "b", "title": "Stitch nothing", "description": "unrelated"},
        ],
    )
    assert [s["id"] for s in _skills.find_skills("stitch mosaic")] == ["a"]


def test_terms_match_in_any_order_and_across_fields(mock_home, skills_cfg, monkeypatch):
    # The user's word order is not the author's. "tiles stitch" is the same
    # request as "stitch tiles", and the two terms may land in different fields.
    _serve(
        monkeypatch,
        skills_cfg,
        [{"id": "a", "title": "Stitch a grid", "description": "overlapping tiles"}],
    )
    assert [s["id"] for s in _skills.find_skills("tiles stitch")] == ["a"]


def test_query_matches_the_skill_id(mock_home, skills_cfg, monkeypatch):
    # Naming the skill is the most specific request there is; hyphens in the id
    # must not hide it.
    _serve(
        monkeypatch,
        skills_cfg,
        [{"id": "flatfield-and-stitch-tiles", "title": "Correct", "description": "d"}],
    )
    assert len(_skills.find_skills("flatfield")) == 1
    assert len(_skills.find_skills("flatfield tiles")) == 1


def test_query_is_case_insensitive(mock_home, skills_cfg, monkeypatch):
    _serve(
        monkeypatch,
        skills_cfg,
        [{"id": "a", "title": "Score", "description": "F1 at matched IoU"}],
    )
    assert len(_skills.find_skills("f1 iou")) == 1
    assert len(_skills.find_skills("F1 IOU")) == 1


def test_terms_match_inside_words(mock_home, skills_cfg, monkeypatch):
    # Deliberate: "measure" should reach "measurements". Stemming is not worth
    # its weight at catalog scale, and the cost is only over-matching.
    _serve(
        monkeypatch,
        skills_cfg,
        [{"id": "a", "title": "Report measurements", "description": "d"}],
    )
    assert len(_skills.find_skills("measure")) == 1


def test_matching_a_tag_retrieves(mock_home, skills_cfg, monkeypatch):
    _serve(
        monkeypatch,
        skills_cfg,
        [{"id": "a", "title": "T", "description": "d", "tags": ["segmentation", "qc"]}],
    )
    assert len(_skills.find_skills("qc")) == 1


def test_a_whole_phrase_that_matched_before_still_matches(
    mock_home, skills_cfg, monkeypatch
):
    # Term matching only ever widens the result set: anything that hit as one
    # substring necessarily has every term present.
    _serve(
        monkeypatch,
        skills_cfg,
        [{"id": "a", "title": "Correct illumination and stitch", "description": "d"}],
    )
    assert len(_skills.find_skills("illumination and stitch")) == 1


def test_unmatched_query_returns_empty_not_everything(
    mock_home, skills_cfg, monkeypatch
):
    _serve(monkeypatch, skills_cfg, [{"id": "a", "title": "T", "description": "d"}])
    assert _skills.find_skills("deconvolution") == []


def test_empty_and_whitespace_query_return_the_whole_catalog(
    mock_home, skills_cfg, monkeypatch
):
    _serve(
        monkeypatch,
        skills_cfg,
        [
            {"id": "a", "title": "A", "description": "d"},
            {"id": "b", "title": "B", "description": "d"},
        ],
    )
    assert len(_skills.find_skills("")) == 2
    assert len(_skills.find_skills("   ")) == 2


def test_results_are_sorted_by_title(mock_home, skills_cfg, monkeypatch):
    _serve(
        monkeypatch,
        skills_cfg,
        [
            {"id": "c", "title": "zebra", "description": "shared"},
            {"id": "a", "title": "Apple", "description": "shared"},
            {"id": "b", "title": "mango", "description": "shared"},
        ],
    )
    assert [s["title"] for s in _skills.find_skills("shared")] == [
        "Apple",
        "mango",
        "zebra",
    ]
