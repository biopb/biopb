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

import biopb_tensor_server.config as config_mod
import numpy as np
import pytest
import tifffile
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

    def test_monitored_bare_host_upstream_is_not_expanded(self, monkeypatch):
        """A monitored bare-host ``grpc://host:port`` tensor-server upstream is
        routed to monitored_sources ONLY -- never expanded into static sources.

        Inline expansion would run one blocking upstream RPC per mirrored source
        before mark_ready(), stalling startup for a large upstream. The
        SourceManager's background re-list owns discovering its sources instead,
        so _resolve_serve_sources must not touch the network at all here.
        """
        import biopb_tensor_server.config as cfg_mod

        def _boom(*_a, **_k):  # pragma: no cover - must never be reached
            raise AssertionError("upstream must not be enumerated at startup")

        monkeypatch.setattr(
            "biopb_tensor_server.adapters.remote_tensor.list_upstream_source_ids",
            _boom,
        )
        # Guard the import site used inside _discover_tensor_server too.
        monkeypatch.setattr(cfg_mod, "_discover_tensor_server", _boom, raising=False)

        upstream = SourceConfig(url="grpc://host:8815", alias="hpc", monitor=True)
        cfg = _config(upstream)

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == ["grpc://host:8815"]
        assert static_sources == []

    def test_unmonitored_bare_host_upstream_is_not_expanded(self, monkeypatch):
        """A bare-host ``grpc://host:port`` upstream with ``monitor=false`` is ALSO
        routed to the background re-list -- never inline-expanded into static
        sources (biopb/biopb#178 regression).

        Inline static expansion registered every mirrored source through a blocking
        per-source ``get_descriptor`` RPC before ``mark_ready()``, so a large
        upstream both stalled SERVING (~1h for a few hundred OME-TIFF proxies) and
        skipped the bulk-seed fast path. A bare-host upstream always mirrors via the
        seeded reconcile; ``monitor=false`` only tunes the re-list cadence, so
        ``_resolve_serve_sources`` must not touch the network here either.
        """
        import biopb_tensor_server.config as cfg_mod

        def _boom(*_a, **_k):  # pragma: no cover - must never be reached
            raise AssertionError("upstream must not be enumerated at startup")

        monkeypatch.setattr(
            "biopb_tensor_server.adapters.remote_tensor.list_upstream_source_ids",
            _boom,
        )
        monkeypatch.setattr(cfg_mod, "_discover_tensor_server", _boom, raising=False)

        upstream = SourceConfig(url="grpc://host:8815", alias="hpc", monitor=False)
        cfg = _config(upstream)

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == ["grpc://host:8815"]
        assert static_sources == []

    def test_monitored_single_source_upstream_is_static(self):
        """A monitored single-source ``grpc://host:port/<id>`` names exactly one
        upstream source (nothing to re-list), so it is still registered as a
        static source (expanded without any upstream RPC)."""
        upstream = SourceConfig(url="grpc://host:8815/raw", alias="hpc", monitor=True)
        cfg = _config(upstream)

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == ["grpc://host:8815/raw"]
        assert [s.url for s in static_sources] == ["grpc://host:8815/raw"]
        # Namespaced under the alias by the single-source expansion path.
        assert static_sources[0].source_id == "hpc__raw"

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

    def test_cloud_without_monitor_is_static_not_monitored(self, tmp_path):
        """`cloud = true` no longer forces monitoring: with monitor unset (false),
        a cloud directory is scanned once via the static-expand path, exactly like
        any other monitor=false directory. The monitor flag is the only switch."""
        root = tmp_path / "cloudroot"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))

        cfg = _config(SourceConfig(url=str(root), cloud=True, monitor=False))

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert monitored_sources == []  # cloud alone does NOT monitor anymore
        assert [s.local_path for s in static_sources] == [
            (root / "image.tif").resolve()
        ]
        # cloud gating rides along: the expanded source keeps cloud=True so it is
        # still deferred as an unresolved source downstream.
        assert all(s.cloud for s in static_sources)

    def test_cloud_with_monitor_is_monitored_not_expanded(self, tmp_path, monkeypatch):
        """`cloud = true, monitor = true` follows the same monitored path as any
        monitored directory: routed to monitored_sources, never pre-walked."""
        root = tmp_path / "cloudroot"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), cloud=True, monitor=True))

        seen_urls = []
        real_discover = config_mod.discover_sources

        def spy(source, registry=None):
            seen_urls.append(source.url)
            return real_discover(source, registry)

        monkeypatch.setattr(config_mod, "discover_sources", spy)

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert str(root) not in seen_urls  # not pre-walked
        assert static_sources == []
        assert [s.url for s in monitored_sources] == [str(root)]


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


class TestAliasTreeRoot:
    """A local source's ``alias`` re-roots it (and everything under a configured
    folder) into its own catalog tree root -- the config-line analogue of a
    drag-dropped folder becoming its own root (see add_source_test's drop cases).
    The override is display-only and honored on the static/expand path only.
    """

    def test_alias_catalog_url_single_source_is_bare_root(self):
        from biopb_tensor_server.config import _alias_catalog_url

        # Configured entry IS the source (file / dataset dir): alias is the root.
        assert _alias_catalog_url("exp", "/data/exp.zarr", "/data/exp.zarr") == "exp"

    def test_alias_catalog_url_preserves_subtree(self):
        from biopb_tensor_server.config import _alias_catalog_url

        assert _alias_catalog_url("exp", "/data/exp", "/data/exp/a.tif") == "exp/a.tif"
        assert (
            _alias_catalog_url("exp", "/data/exp", "/data/exp/sub/b.tif")
            == "exp/sub/b.tif"
        )

    def test_alias_catalog_url_non_relativizable_is_bare_root(self):
        from biopb_tensor_server.config import _alias_catalog_url

        # Primary not under the root (defensive) -> alias-only root, never "../".
        assert _alias_catalog_url("exp", "/data/exp", "/elsewhere/x.tif") == "exp"

    def test_single_file_alias_sets_catalog_url(self, tmp_path):
        f = tmp_path / "img.tif"
        _write_tiff(str(f))
        cfg = _config(SourceConfig(url=str(f), alias="myroot"))

        resolved = resolve_all_sources(cfg)

        assert len(resolved) == 1
        assert resolved[0]._catalog_url == "myroot"

    def test_folder_alias_reroots_children_under_alias(self, tmp_path):
        root = tmp_path / "acquisition"
        root.mkdir()
        (root / "sub").mkdir()
        _write_tiff(str(root / "a.tif"))
        _write_tiff(str(root / "sub" / "b.tif"))
        cfg = _config(SourceConfig(url=str(root), alias="exp"))

        resolved = resolve_all_sources(cfg)

        assert sorted(s._catalog_url for s in resolved) == [
            "exp/a.tif",
            "exp/sub/b.tif",
        ]

    def test_no_alias_leaves_catalog_url_none(self, tmp_path):
        f = tmp_path / "img.tif"
        _write_tiff(str(f))
        cfg = _config(SourceConfig(url=str(f)))

        resolved = resolve_all_sources(cfg)

        assert resolved[0]._catalog_url is None

    def test_remote_alias_is_not_a_tree_root(self):
        """On a remote (non-tensor-server) source the alias keeps its proxy /
        namespace meaning -- it is NOT turned into a display tree-root override."""
        cfg = _config(SourceConfig(url="s3://bucket/k.zarr", type="zarr", alias="x"))

        resolved = resolve_all_sources(cfg)

        assert resolved[0]._catalog_url is None
        assert resolved[0].alias == "x"  # untouched

    def test_monitored_dir_alias_is_ignored_with_warning(self, tmp_path, caplog):
        """A monitored *directory* re-merges into the shared path tree on rescan,
        so its alias tree-root cannot hold: it is dropped with a warning and never
        applied (the monitored entry is not expanded, so catalog_url stays None)."""
        root = tmp_path / "watched"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), alias="live", monitor=True))

        with caplog.at_level("WARNING", logger="biopb_tensor_server.cli"):
            static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert [s.url for s in monitored_sources] == [str(root)]
        assert all(s._catalog_url is None for s in monitored_sources)
        assert any("alias" in r.message and "live" in r.message for r in caplog.records)

    def test_monitored_single_file_alias_is_honored(self, tmp_path):
        """A ``monitor=true`` single *file* cannot be live-monitored, so it is
        registered static -- and being static, its alias tree-root IS honored (it
        is never rescanned). No ignore-warning applies to it."""
        f = tmp_path / "img.tif"
        _write_tiff(str(f))
        cfg = _config(SourceConfig(url=str(f), alias="solo", monitor=True))

        static_sources, monitored_sources = _resolve_serve_sources(cfg)

        assert monitored_sources == []
        assert [s._catalog_url for s in static_sources] == ["solo"]
