"""JSON-only config loading (biopb/biopb#34).

JSON is the only config format, and nothing in the tree knows the word TOML any
more -- not the reader, not ``find_config``, not the installers. A leftover
``biopb.toml`` from a pre-#34 install is an ordinary unrelated file.
"""

import json
import logging
import re
from pathlib import Path

import pytest
from biopb_tensor_server.core.config import (
    CANONICAL_CONFIG_NAME,
    find_config,
    generate_source_id,
    load_config,
    parse_config,
)

_JSON = {
    "server": {"log_level": "DEBUG"},
    "cache": {"file_max_segment_mb": 128},
    "sources": [
        {
            "type": "zarr",
            "url": "/data/a.zarr",
        }
    ],
}


def _assert_expected(cfg):
    assert cfg.log_level == "DEBUG"
    assert cfg.cache.file_max_segment_bytes == 128 * 1024 * 1024
    assert len(cfg.sources) == 1
    src = cfg.sources[0]
    assert src.type == "zarr"
    # source_id is derived from the URL, not user-assigned (biopb/biopb#308).
    assert src.source_id == generate_source_id("/data/a.zarr", "zarr")


def test_json_config_loads(tmp_path):
    json_path = tmp_path / "biopb.json"
    json_path.write_text(json.dumps(_JSON))
    _assert_expected(load_config(json_path))


def test_extensionless_file_is_read_as_json(tmp_path):
    # An unconventionally-named config still loads -- JSON is assumed, not sniffed.
    p = tmp_path / "config"
    p.write_text(json.dumps(_JSON))
    _assert_expected(load_config(p))


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


def test_find_config_returns_the_json_beside_a_leftover_toml(tmp_path, caplog):
    """A pre-#34 ``biopb.toml`` is not config -- it is just a file in the dir.

    Nothing reads it, nothing warns about it, and it does not shadow or shift
    the canonical name. The one thing it must not do is make ``find_config``
    answer anything other than ``biopb.json``.
    """
    (tmp_path / "biopb.toml").write_text("[server]\nport = 9000\n")
    (tmp_path / CANONICAL_CONFIG_NAME).write_text(json.dumps(_JSON))
    with caplog.at_level(logging.WARNING):
        assert find_config(tmp_path) == tmp_path / CANONICAL_CONFIG_NAME
    assert not caplog.records


def test_find_config_ignores_a_lone_toml(tmp_path, caplog):
    # No JSON at all: the answer is still the canonical name, so the caller
    # seeds a fresh config rather than being handed a file nothing can read.
    (tmp_path / "biopb.toml").write_text("[server]\nport = 9000\n")
    with caplog.at_level(logging.WARNING):
        assert find_config(tmp_path) == tmp_path / CANONICAL_CONFIG_NAME
    assert not caplog.records


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


def test_retired_dim_labels_key_warns_and_is_ignored(tmp_path, caplog):
    """A config that still sets ``dim_labels`` loads; the key is ignored."""
    import logging

    payload = {
        "server": {"log_level": "DEBUG"},
        "sources": [
            {"type": "zarr", "url": "/data/a.zarr", "dim_labels": ["z", "y", "x"]}
        ],
    }
    p = tmp_path / "biopb.json"
    p.write_text(json.dumps(payload))
    with caplog.at_level(logging.WARNING):
        cfg = load_config(p)
    assert len(cfg.sources) == 1
    assert not hasattr(cfg.sources[0], "dim_labels")
    assert "Unknown config key `dim_labels`" in caplog.text
