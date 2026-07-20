"""Tests for mcp/_skills.py — curated skills discovery + retrieval.

Fully hermetic: the network is always patched, and the cache dir is redirected
to a tmp path via ``Path.home``. Exercises the fail-open resolution chain
(network → disk cache → bundled snapshot), the tolerant catalog reader, sha
integrity, and frontmatter stripping.
"""

import hashlib
import json

import pytest

from biopb_mcp.mcp import _skills

CATALOG_URL = "https://example.test/skills/catalog.json"


@pytest.fixture
def mock_home(monkeypatch, tmp_path):
    """Redirect ~ so the skills cache dir lands under a tmp path."""
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture
def skills_cfg(monkeypatch):
    """Control config without touching the real CONFIG singleton / disk, and
    reset the module-level TTL cache so tests don't leak into each other."""
    cfg = {"skills_enabled": True, "skills_catalog_url": "", "skills_cache_ttl": 3600}

    def fake_setting(path, default=None):
        return cfg.get(path.rsplit(".", 1)[1], default)

    monkeypatch.setattr(_skills, "_setting", fake_setting)
    _skills._cache["skills"] = None
    _skills._cache["at"] = 0.0
    return cfg


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
def test_find_skills_falls_back_to_bundle_offline(mock_home, skills_cfg, monkeypatch):
    _offline(monkeypatch)  # network down, catalog_url empty -> bundle
    skills = _skills.find_skills("")
    ids = {s["id"] for s in skills}
    assert {"load-tensor-source", "measure-labels", "segment-nuclei"} <= ids
    assert all(s["uri"] == f"skill://{s['id']}" for s in skills)


def test_find_skills_query_filters(mock_home, skills_cfg, monkeypatch):
    _offline(monkeypatch)
    assert [s["id"] for s in _skills.find_skills("nuclei")] == ["segment-nuclei"]
    assert _skills.find_skills("segmentation")  # tag match
    assert _skills.find_skills("no-such-term-xyz") == []


def test_find_skills_disabled_returns_empty(mock_home, skills_cfg, monkeypatch):
    _offline(monkeypatch)
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
def test_get_body_bundle_fallback_strips_frontmatter(
    mock_home, skills_cfg, monkeypatch
):
    _offline(monkeypatch)
    body = _skills.get_skill_body("load-tensor-source")
    assert not body.lstrip().startswith("---")
    assert "Load a tensor source" in body


def test_bundle_body_sha_matches_catalog(mock_home, skills_cfg, monkeypatch):
    _offline(monkeypatch)
    entry = next(s for s in _skills.load_catalog() if s["id"] == "load-tensor-source")
    raw = _skills._bundle_text("load-tensor-source.md").encode()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]


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
