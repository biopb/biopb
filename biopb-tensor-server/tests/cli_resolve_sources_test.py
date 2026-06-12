"""Tests for the serve-path source partitioning (biopb/biopb#54).

`_resolve_serve_sources` splits configured `[[sources]]` entries into
(static_sources, monitored_sources) without the old behavior of running a full
discovery walk over `monitor = true` directories at startup. The walk was pure
waste (its results were discarded by the overlap filter) and it crashed startup
on a not-yet-mounted monitored directory. These tests pin the new behavior:

- a missing monitored dir is tolerated (not expanded, no crash);
- a monitored dir is never handed to `discover_sources`;
- a single-file `monitor = true` entry becomes a static source;
- the overlap filter still drops non-monitored expansions under a monitored root;
- remote `monitor = true` stays both static and monitored;
- `resolve_all_sources(sources=..., tolerant=...)` expands only the given subset
  and skips unresolvable entries only when asked.
"""

import os

import numpy as np
import pytest
import tifffile

import biopb_tensor_server.config as config_mod
from biopb_tensor_server.cli import _resolve_serve_sources
from biopb_tensor_server.config import (
    ServerConfig,
    SourceConfig,
    resolve_all_sources,
)


def _write_tiff(path: str) -> None:
    data = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
    tifffile.imwrite(path, data)


def _config(*sources: SourceConfig) -> ServerConfig:
    return ServerConfig(sources=list(sources))


class TestResolveServeSources:
    def test_missing_monitored_dir_does_not_crash(self, tmp_path):
        """A not-yet-mounted monitored dir is kept for monitoring, not expanded.

        Against the old code this raised `ValueError: Path does not exist`
        (config.py) and killed startup (biopb/biopb#54 defect 1).
        """
        missing = tmp_path / "nfs_root_not_mounted_yet"
        cfg = _config(SourceConfig(url=str(missing), monitor=True))

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert static_sources == []
        assert len(monitored_sources) == 1
        assert monitored_sources[0].url == str(missing)

    def test_monitored_dir_is_not_expanded(self, tmp_path, monkeypatch):
        """A monitored directory is never handed to discover_sources.

        The discarded walk is the cause of the triple-traversal / pre-bind
        latency (biopb/biopb#54 defect 2). Spy on discover_sources to prove the
        monitored root never reaches it.
        """
        root = tmp_path / "monitored"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), monitor=True))

        seen_urls = []
        real_discover = config_mod.discover_sources

        def spy(source, registry=None):
            seen_urls.append(source.url)
            return real_discover(source, registry)

        monkeypatch.setattr(config_mod, "discover_sources", spy)

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert str(root) not in seen_urls  # the dir was not walked
        assert static_sources == []
        assert [s.url for s in monitored_sources] == [str(root)]

    def test_single_file_monitor_becomes_static(self, tmp_path):
        """A monitor=true entry pointing at a FILE is registered statically.

        Previously the file path entered the monitored-dirs filter set, its own
        expansion was dropped by the overlap filter, and create_source_manager
        refused to monitor a file -- so the source vanished. Now it is static.
        """
        tiff = tmp_path / "single.tif"
        _write_tiff(str(tiff))
        cfg = _config(SourceConfig(url=str(tiff), monitor=True))

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert monitored_sources == []
        assert len(static_sources) == 1
        assert static_sources[0].local_path == tiff.resolve()

    def test_non_monitored_under_monitored_root_is_filtered(self, tmp_path):
        """A non-monitored entry whose expansion lands under a monitored root
        is dropped from static_sources (overlap filter still applies)."""
        root = tmp_path / "monitored"
        root.mkdir()
        tiff = root / "inside.tif"
        _write_tiff(str(tiff))

        cfg = _config(
            SourceConfig(url=str(root), monitor=True),
            SourceConfig(url=str(tiff)),  # non-monitored, but under root
        )

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == [str(root)]
        assert static_sources == []  # the inside.tif expansion is filtered out

    def test_non_monitored_outside_monitored_root_survives(self, tmp_path):
        """A non-monitored entry outside every monitored root stays static."""
        root = tmp_path / "monitored"
        root.mkdir()
        _write_tiff(str(root / "inside.tif"))

        outside = tmp_path / "outside.tif"
        _write_tiff(str(outside))

        cfg = _config(
            SourceConfig(url=str(root), monitor=True),
            SourceConfig(url=str(outside)),
        )

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == [str(root)]
        assert [s.local_path for s in static_sources] == [outside.resolve()]

    def test_remote_monitor_is_static_and_monitored(self, tmp_path):
        """Remote monitor=true entries keep current behavior: registered
        statically AND passed through as monitored (the manager logs the
        no-monitor notice)."""
        remote = SourceConfig(url="s3://bucket/data.zarr", type="zarr", monitor=True)
        cfg = _config(remote)

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == ["s3://bucket/data.zarr"]
        assert [s.url for s in static_sources] == ["s3://bucket/data.zarr"]

    def test_missing_static_source_is_skipped_not_fatal(self, tmp_path):
        """A missing non-monitored source is warned-and-skipped; the rest serve.

        (biopb/biopb#54 extension: the same ValueError that crashed on a missing
        monitored dir also crashed on a missing static path.)
        """
        good = tmp_path / "good.tif"
        _write_tiff(str(good))
        missing = tmp_path / "typo.tif"

        cfg = _config(
            SourceConfig(url=str(missing)),  # does not exist
            SourceConfig(url=str(good)),
        )

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert monitored_sources == []
        assert [s.local_path for s in static_sources] == [good.resolve()]


class TestResolveAllSourcesOverrides:
    def test_sources_override_expands_only_the_subset(self, tmp_path):
        """The `sources=` keyword expands only the passed entries, ignoring the
        rest of config.sources."""
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _write_tiff(str(a))
        _write_tiff(str(b))

        src_a = SourceConfig(url=str(a))
        src_b = SourceConfig(url=str(b))
        cfg = _config(src_a, src_b)

        only_a = resolve_all_sources(cfg, sources=[src_a])
        assert [s.local_path for s in only_a] == [a.resolve()]

        # No-arg call is unchanged: expands every config source.
        both = resolve_all_sources(cfg)
        assert {s.local_path for s in both} == {a.resolve(), b.resolve()}

    def test_tolerant_skips_unresolvable_entry(self, tmp_path):
        """tolerant=True skips a source that fails to resolve; the default
        (False) re-raises -- so validate/list keep failing loudly."""
        good = tmp_path / "good.tif"
        _write_tiff(str(good))
        missing = SourceConfig(url=str(tmp_path / "missing.tif"))
        good_src = SourceConfig(url=str(good))
        cfg = _config(missing, good_src)

        resolved = resolve_all_sources(cfg, tolerant=True)
        assert [s.local_path for s in resolved] == [good.resolve()]

        with pytest.raises(ValueError):
            resolve_all_sources(cfg, tolerant=False)
