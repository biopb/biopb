"""JSON-only config loading, and how a pre-#34 TOML is turned away
(biopb/biopb#34).

The legacy TOML read path is gone: a ``.toml`` config is rejected without being
parsed, and both that refusal and a plain JSON syntax error must name
``biopb server migrate-config`` -- a parse error is the only place a user
learns the format changed. ``find_config`` still *sees* a legacy file, so the
failure is "migrate this" rather than "no config here".
"""

import json
import logging
import re
from pathlib import Path

import pytest
from biopb_tensor_server.core.config import (
    CANONICAL_CONFIG_NAME,
    LEGACY_CONFIG_NAME,
    find_config,
    generate_source_id,
    load_config,
    parse_config,
    read_legacy_toml,
)

# A config exercising server scalars, a nested table (cache), and a [[sources]]
# array -- the three shapes that differ syntactically between TOML and JSON.
_TOML = """
[server]
host = "127.0.0.1"
port = 9000

[cache]
backend = "memory"
max_bytes = 123456789

[[sources]]
type = "zarr"
url = "/data/a.zarr"
dim_labels = ["z", "y", "x"]
"""

_JSON = {
    "server": {"host": "127.0.0.1", "port": 9000},
    "cache": {"backend": "memory", "max_bytes": 123456789},
    "sources": [
        {
            "type": "zarr",
            "url": "/data/a.zarr",
            "dim_labels": ["z", "y", "x"],
        }
    ],
}


def _assert_expected(cfg):
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.cache.backend == "memory"
    assert cfg.cache.memory_max_bytes == 123456789
    assert len(cfg.sources) == 1
    src = cfg.sources[0]
    assert src.type == "zarr"
    # source_id is derived from the URL, not user-assigned (biopb/biopb#308).
    assert src.source_id == generate_source_id("/data/a.zarr", "zarr")
    assert src.dim_labels == ["z", "y", "x"]


def test_json_config_loads(tmp_path):
    json_path = tmp_path / "biopb.json"
    json_path.write_text(json.dumps(_JSON))
    _assert_expected(load_config(json_path))


def test_extensionless_file_is_read_as_json(tmp_path):
    # An unconventionally-named config still loads -- JSON is assumed, not sniffed.
    p = tmp_path / "config"
    p.write_text(json.dumps(_JSON))
    _assert_expected(load_config(p))


def test_toml_config_is_rejected_with_migration_hint(tmp_path):
    toml_path = tmp_path / "biopb.toml"
    toml_path.write_text(_TOML)
    with pytest.raises(ValueError) as exc:
        load_config(toml_path)
    assert "migrate-config" in str(exc.value)
    assert re.search(re.escape(str(toml_path)), str(exc.value))


def test_toml_content_under_a_json_name_reports_the_migration_hint(tmp_path):
    # The extension lies but the bytes are TOML: it fails as invalid JSON, and
    # that message is the user's only clue that the format changed.
    p = tmp_path / "biopb.json"
    p.write_text(_TOML)
    with pytest.raises(ValueError) as exc:
        load_config(p)
    assert "migrate-config" in str(exc.value)


def test_invalid_json_raises_value_error_naming_file(tmp_path):
    p = tmp_path / "biopb.json"
    p.write_text("{not valid json")
    # `match` is a regex; re.escape so a Windows path (backslashes -> escapes
    # like \U) is matched literally.
    with pytest.raises(ValueError, match=re.escape(str(p))):
        load_config(p)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_read_legacy_toml_still_parses_for_migration(tmp_path):
    # The one surviving TOML reader: `biopb server migrate-config`'s input side.
    toml_path = tmp_path / "biopb.toml"
    toml_path.write_text(_TOML)
    data = read_legacy_toml(toml_path)
    assert data["server"]["port"] == 9000
    assert data["sources"][0]["url"] == "/data/a.zarr"


def test_find_config_prefers_json(tmp_path):
    (tmp_path / LEGACY_CONFIG_NAME).write_text(_TOML)
    (tmp_path / CANONICAL_CONFIG_NAME).write_text(json.dumps(_JSON))
    assert find_config(tmp_path) == tmp_path / CANONICAL_CONFIG_NAME


def test_find_config_warns_when_both_exist(tmp_path, caplog):
    (tmp_path / LEGACY_CONFIG_NAME).write_text(_TOML)
    (tmp_path / CANONICAL_CONFIG_NAME).write_text(json.dumps(_JSON))
    with caplog.at_level(logging.WARNING):
        find_config(tmp_path)
    assert any(
        "biopb.toml" in r.message and "ignoring" in r.message.lower()
        for r in caplog.records
    )


def test_find_config_does_not_warn_with_single_file(tmp_path, caplog):
    (tmp_path / CANONICAL_CONFIG_NAME).write_text(json.dumps(_JSON))
    with caplog.at_level(logging.WARNING):
        find_config(tmp_path)
    assert not caplog.records


def test_find_config_returns_legacy_toml_with_migration_warning(tmp_path, caplog):
    # A lone legacy config is still handed back -- returning the (absent)
    # canonical name instead would read as "no config at all", and every caller
    # defaults around that silently.
    (tmp_path / LEGACY_CONFIG_NAME).write_text(_TOML)
    with caplog.at_level(logging.WARNING):
        found = find_config(tmp_path)
    assert found == tmp_path / LEGACY_CONFIG_NAME
    assert any("migrate-config" in r.message for r in caplog.records)


def test_find_config_defaults_to_canonical_when_absent(tmp_path):
    assert find_config(tmp_path) == tmp_path / CANONICAL_CONFIG_NAME
    assert isinstance(find_config(tmp_path), Path)


# --- metadata_db.enabled deprecation (biopb/biopb#225) -----------------------


def test_metadata_db_enabled_true_warns_removed(caplog):
    """The removed flag (biopb/biopb#225) is ignored with a warning even when on.

    The config carries no `enabled` attribute anymore; the DB is always on."""
    with caplog.at_level(logging.WARNING):
        cfg = parse_config({"metadata_db": {"enabled": True}})
    assert not hasattr(cfg.metadata_db, "enabled")  # field removed
    msgs = [r.message for r in caplog.records if "metadata_db.enabled" in r.message]
    assert msgs and any("#225" in m for m in msgs)


def test_metadata_db_enabled_false_warns_now_on_anyway(caplog):
    """`enabled = false` is the notable case: the DB comes up ON regardless, so
    the warning says the flag is no longer honored and names the SQL catalog."""
    with caplog.at_level(logging.WARNING):
        cfg = parse_config({"metadata_db": {"enabled": False}})
    assert not hasattr(cfg.metadata_db, "enabled")
    msgs = [r.message for r in caplog.records if "metadata_db.enabled" in r.message]
    assert msgs and any("no longer honored" in m.lower() for m in msgs)
    assert any("query_sources" in m for m in msgs)


def test_metadata_db_absent_does_not_warn(caplog):
    """The default path (flag omitted) stays silent -- the DB is always on."""
    with caplog.at_level(logging.WARNING):
        cfg = parse_config({})
    assert cfg.metadata_db is not None
    assert not any("metadata_db.enabled" in r.message for r in caplog.records)


# --- sources.source_id deprecation (biopb/biopb#308) -------------------------


def test_explicit_source_id_is_ignored_and_warns(caplog):
    """An explicit `source_id` no longer overrides the URL-derived id: honoring
    it let two configs aim the same bytes at two catalog rows (biopb/biopb#308).
    It is dropped with a warning, and the id falls back to the URL hash."""
    with caplog.at_level(logging.WARNING):
        cfg = parse_config(
            {"sources": [{"type": "zarr", "url": "/data/a.zarr", "source_id": "a"}]}
        )
    (src,) = cfg.sources
    assert src.source_id == generate_source_id("/data/a.zarr", "zarr")
    assert src.source_id != "a"
    msgs = [r.message for r in caplog.records if "sources.source_id" in r.message]
    assert msgs and any("#308" in m for m in msgs)


def test_source_id_absent_does_not_warn(caplog):
    """A config that never sets `source_id` stays silent."""
    with caplog.at_level(logging.WARNING):
        cfg = parse_config({"sources": [{"type": "zarr", "url": "/data/a.zarr"}]})
    assert cfg.sources[0].source_id == generate_source_id("/data/a.zarr", "zarr")
    assert not any("sources.source_id" in r.message for r in caplog.records)
