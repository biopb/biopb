"""A source url must be absolute; a relative one is refused (biopb/biopb#947).

``Path.resolve()`` -- which ``local_path`` and the ``source_id`` hash both run --
anchors a relative path on the process cwd. A tensor server is not launched from
a shell someone chose: the control plane, systemd and a container entrypoint each
leave a different cwd, so a relative url names different data per launch, under a
different ``source_id``, which in turn strands the ROI annotations keyed to it.
The one place it appears to work is the author's own terminal, sitting in the
directory they wrote the config in -- exactly where the bug cannot be observed.

There is no anchor worth guessing at, so such an entry is not served. The split
follows biopb/biopb#608: a supervised start drops the one bad line loudly and
serves the rest, while ``validate`` -- asked outright whether the file is good --
fails on it.

Every test here runs from a cwd where the relative path WOULD have resolved, so
a regression cannot pass by accident.
"""

import json
import logging
from pathlib import Path

import pytest
from biopb_tensor_server.core.config import (
    SourceConfig,
    load_config,
    parse_config,
    validate_config_dict,
)


def _write_config(dirpath: Path, url: str, key: str = "url") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / "biopb.json"
    path.write_text(json.dumps({"sources": [{key: url, "type": "zarr"}]}))
    return path


class TestRelativeUrlIsNotServed:
    """The load path drops the entry and comes up with what is left."""

    def test_relative_source_is_dropped(self, tmp_path, monkeypatch):
        config = _write_config(tmp_path / "project", "data/plate3.zarr")
        (tmp_path / "project" / "data" / "plate3.zarr").mkdir(parents=True)
        monkeypatch.chdir(tmp_path / "project")  # where it would have resolved

        cfg = load_config(config)

        assert cfg.sources == []

    def test_the_drop_is_logged_as_an_error_naming_the_url(self, caplog):
        """It never comes right on its own, so it is not a warning.

        biopb/biopb#608's volume rule: a config that cannot resolve until someone
        edits it is louder than a path that may yet be mounted.
        """
        with caplog.at_level(logging.ERROR):
            parse_config({"sources": [{"url": "data/plate3.zarr"}]})

        assert "data/plate3.zarr" in caplog.text
        assert "NOT SERVING" in caplog.text

    def test_one_bad_line_does_not_take_the_others_down(self, tmp_path):
        """The reason this is a skip and not a refusal to start."""
        config = tmp_path / "biopb.json"
        config.write_text(
            json.dumps(
                {
                    "sources": [
                        {"url": "data/plate3.zarr"},
                        {"url": "/data/plate4.zarr"},
                    ]
                }
            )
        )

        cfg = load_config(config)

        assert [s.url for s in cfg.sources] == ["/data/plate4.zarr"]

    def test_legacy_path_key_is_refused_too(self, tmp_path, monkeypatch):
        """`path` is the pre-#34 alias for `url`; it goes through the same door."""
        config = _write_config(tmp_path / "project", "data/plate3.zarr", key="path")
        monkeypatch.chdir(tmp_path / "project")

        assert load_config(config).sources == []

    def test_the_config_directory_is_not_an_anchor_either(self, tmp_path, monkeypatch):
        """Resolving against the config file was considered and rejected.

        A config directory is not usually where the data sits, so anchoring there
        would only trade one surprising directory for another.
        """
        config = _write_config(tmp_path / "project", "data/plate3.zarr")
        (tmp_path / "project" / "data" / "plate3.zarr").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        assert load_config(config).sources == []


class TestAbsoluteUrlIsUntouched:
    def test_absolute_url_is_left_alone(self, tmp_path, monkeypatch):
        config = _write_config(tmp_path / "project", "/data/plate3.zarr")
        monkeypatch.chdir(tmp_path)

        assert load_config(config).sources[0].url == "/data/plate3.zarr"

    def test_home_relative_url_expands(self, tmp_path, monkeypatch):
        """`~` names one directory unambiguously; it used to become `$PWD/~/...`."""
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        config = _write_config(tmp_path / "project", "~/plate3.zarr")
        monkeypatch.chdir(tmp_path)

        assert load_config(config).sources[0].url == str(
            tmp_path / "home" / "plate3.zarr"
        )

    def test_remote_url_is_left_alone(self, tmp_path, monkeypatch):
        """`Path` would mangle a scheme's `//` and prepend the cwd."""
        cfg_dir = tmp_path / "project"
        cfg_dir.mkdir()
        config = cfg_dir / "biopb.json"
        config.write_text(
            json.dumps(
                {"sources": [{"url": "grpc://lab:8815", "type": "tensor-server"}]}
            )
        )
        monkeypatch.chdir(tmp_path)

        assert load_config(config).sources[0].url == "grpc://lab:8815"


class TestSourceConfigBackstop:
    """The invariant `local_path` and `source_id` rely on, held at construction."""

    def test_relative_url_is_refused(self):
        with pytest.raises(ValueError, match="absolute path"):
            SourceConfig(url="data/plate3.zarr")

    def test_absolute_url_is_accepted(self):
        assert SourceConfig(url="/data/plate3.zarr").url == "/data/plate3.zarr"


class TestValidateFailsHard:
    """`validate` and the admin form are the strict end: reported, per field."""

    def test_relative_url_is_reported(self):
        problems = validate_config_dict({"sources": [{"url": "data/plate3.zarr"}]})

        assert len(problems) == 1
        assert "data/plate3.zarr" in problems[0]["message"]

    def test_the_problem_carries_a_field_path_the_form_can_mark(self):
        """A root-level ([]) problem marks nothing in the admin sidebar."""
        problems = validate_config_dict({"sources": [{"url": "data/plate3.zarr"}]})

        assert problems[0]["path"] == ["sources", "url"]

    def test_absolute_and_remote_urls_report_nothing(self):
        assert (
            validate_config_dict(
                {
                    "sources": [
                        {"url": "/data/plate3.zarr"},
                        {"url": "grpc://lab:8815", "type": "tensor-server"},
                    ]
                }
            )
            == []
        )
