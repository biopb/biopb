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

import biopb_tensor_server.sources.resolve as resolve_mod
import numpy as np
import pytest
import tifffile
from biopb_tensor_server.core.config import (
    ServerConfig,
    SourceConfig,
)
from biopb_tensor_server.sources.resolve import resolve_all_sources
from biopb_tensor_server.sources.roots import RootKind, partition_sources


def _write_tiff(path: str) -> None:
    data = np.random.randint(0, 255, (64, 64), dtype=np.uint16)
    tifffile.imwrite(path, data)


def _config(*sources: SourceConfig) -> ServerConfig:
    return ServerConfig(sources=list(sources))


def _resolve_serve_sources(cfg: ServerConfig, registry=None):
    """``(static, upstreams, monitored, scan_once)`` as config entries."""
    static, roots = partition_sources(
        cfg.sources,
        registry,
        credentials_config=cfg.credentials,
        write_dir=cfg.write_dir,
    )

    def sources(kind):
        return [r.source for r in roots.of_kind(kind)]

    return (
        static,
        sources(RootKind.UPSTREAM),
        sources(RootKind.MONITORED),
        sources(RootKind.SCAN_ONCE),
    )


class TestResolveServeSources:
    def test_missing_monitored_dir_does_not_crash(self, tmp_path):
        """A not-yet-mounted monitored dir is kept for monitoring, not expanded.

        Against the old code this raised `ValueError: Path does not exist`
        (config.py) and killed startup (biopb/biopb#54 defect 1).
        """
        missing = tmp_path / "nfs_root_not_mounted_yet"
        cfg = _config(SourceConfig(url=str(missing), monitor=True))

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

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
        real_discover = resolve_mod.discover_sources

        def spy(source, registry=None):
            seen_urls.append(source.url)
            return real_discover(source, registry)

        monkeypatch.setattr(resolve_mod, "discover_sources", spy)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert str(root) not in seen_urls  # the dir was not walked
        assert static_sources == []
        assert [s.url for s in monitored_sources] == [str(root)]

    def test_single_file_monitor_is_registered_once(self, tmp_path):
        """A monitor=true entry pointing at a FILE cannot be watched, so it is
        registered once, as a scan-once root.

        Previously the file path entered the monitored-dirs filter set, its own
        expansion was dropped by the overlap filter, and the file vanished.
        """
        tiff = tmp_path / "single.tif"
        _write_tiff(str(tiff))
        cfg = _config(SourceConfig(url=str(tiff), monitor=True))

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert monitored_sources == []
        assert static_sources == []
        assert [s.local_path for s in scan_once_sources] == [tiff.resolve()]

    def test_unwatched_directory_is_scanned_once_not_expanded(
        self, tmp_path, monkeypatch
    ):
        """A monitor=false directory is handed to the manager, never walked here."""
        root = tmp_path / "plain"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), monitor=False, alias="lab"))

        seen_urls = []
        real_discover = resolve_mod.discover_sources

        def spy(source, registry=None, credentials_config=None):
            seen_urls.append(source.url)
            return real_discover(source, registry, credentials_config)

        monkeypatch.setattr(resolve_mod, "discover_sources", spy)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert str(root) not in seen_urls
        assert static_sources == [] and monitored_sources == []
        assert [s.url for s in scan_once_sources] == [str(root)]
        assert scan_once_sources[0].alias == "lab"

    def test_typed_directory_is_registered_once(self, tmp_path):
        """A directory given an explicit type is claimed in place, as one root."""
        root = tmp_path / "plate.zarr"
        root.mkdir()
        cfg = _config(SourceConfig(url=str(root), type="zarr"))

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert static_sources == []
        assert [s.url for s in scan_once_sources] == [str(root)]

    def test_missing_unwatched_path_is_left_to_the_scan_once_pass(self, tmp_path):
        """A path that is not there is still a root: the manager's pass warns and
        skips it, so a not-yet-mounted path never stops startup."""
        cfg = _config(SourceConfig(url=str(tmp_path / "gone"), monitor=False))

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert static_sources == []
        assert [s.url for s in scan_once_sources] == [str(tmp_path / "gone")]

    def test_non_monitored_under_monitored_root_is_filtered(self, tmp_path):
        """A non-monitored file inside a monitored root is the rescan's, so it is
        not also registered once."""
        root = tmp_path / "monitored"
        root.mkdir()
        tiff = root / "inside.tif"
        _write_tiff(str(tiff))

        cfg = _config(
            SourceConfig(url=str(root), monitor=True),
            SourceConfig(url=str(tiff)),  # non-monitored, but under root
        )

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert [s.url for s in monitored_sources] == [str(root)]
        assert static_sources == [] and scan_once_sources == []

    def test_non_monitored_outside_monitored_root_survives(self, tmp_path):
        """A non-monitored entry outside every monitored root stays its own root."""
        root = tmp_path / "monitored"
        root.mkdir()
        _write_tiff(str(root / "inside.tif"))

        outside = tmp_path / "outside.tif"
        _write_tiff(str(outside))

        cfg = _config(
            SourceConfig(url=str(root), monitor=True),
            SourceConfig(url=str(outside)),
        )

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert [s.url for s in monitored_sources] == [str(root)]
        assert [s.local_path for s in scan_once_sources] == [outside.resolve()]

    def test_remote_monitor_is_static_only(self, tmp_path):
        """A remote monitor=true entry (not a bare-host upstream) has nothing to
        watch or re-list, so it is registered statically and only that."""
        remote = SourceConfig(url="s3://bucket/data.zarr", type="zarr", monitor=True)
        cfg = _config(remote)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert monitored_sources == []
        assert [s.url for s in static_sources] == ["s3://bucket/data.zarr"]

    def test_monitored_bare_host_upstream_is_not_expanded(self, monkeypatch):
        """A monitored bare-host ``grpc://host:port`` tensor-server upstream is
        routed to upstream_sources ONLY -- never expanded into static sources.

        Inline expansion would run one blocking upstream RPC per mirrored source
        before mark_ready(), stalling startup for a large upstream. The
        SourceManager's background re-list owns discovering its sources instead,
        so _resolve_serve_sources must not touch the network at all here.
        """

        def _boom(*_a, **_k):  # pragma: no cover - must never be reached
            raise AssertionError("upstream must not be enumerated at startup")

        monkeypatch.setattr(
            "biopb_tensor_server.sources.resolve.list_upstream_source_ids",
            _boom,
        )
        # Guard the expansion entry point too. No raising=False: if
        # _discover_tensor_server is renamed/removed, this setattr must fail loudly
        # rather than silently create a dead attribute and let the test pass blind.
        monkeypatch.setattr(resolve_mod, "_discover_tensor_server", _boom)

        upstream = SourceConfig(url="grpc://host:8815", alias="hpc", monitor=True)
        cfg = _config(upstream)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert [s.url for s in upstream_sources] == ["grpc://host:8815"]
        assert monitored_sources == []
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

        def _boom(*_a, **_k):  # pragma: no cover - must never be reached
            raise AssertionError("upstream must not be enumerated at startup")

        monkeypatch.setattr(
            "biopb_tensor_server.sources.resolve.list_upstream_source_ids",
            _boom,
        )
        monkeypatch.setattr(resolve_mod, "_discover_tensor_server", _boom)

        upstream = SourceConfig(url="grpc://host:8815", alias="hpc", monitor=False)
        cfg = _config(upstream)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert [s.url for s in upstream_sources] == ["grpc://host:8815"]
        assert monitored_sources == []
        assert static_sources == []

    def test_monitored_single_source_upstream_is_static(self):
        """A monitored single-source ``grpc://host:port/<id>`` names exactly one
        upstream source (nothing to re-list), so it is still registered as a
        static source (expanded without any upstream RPC)."""
        upstream = SourceConfig(url="grpc://host:8815/raw", alias="hpc", monitor=True)
        cfg = _config(upstream)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert monitored_sources == []  # nothing to watch or re-list
        assert [s.url for s in static_sources] == ["grpc://host:8815/raw"]
        # Namespaced under the alias by the single-source expansion path.
        assert static_sources[0].source_id == "hpc__raw"

    def test_a_missing_path_does_not_stop_the_rest(self, tmp_path):
        """A missing non-monitored path is warned-and-skipped later; the rest serve.

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

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert monitored_sources == []
        assert [s.local_path for s in scan_once_sources] == [
            missing.resolve(),
            good.resolve(),
        ]

    def test_cloud_without_monitor_is_scanned_once_not_monitored(
        self, tmp_path, monkeypatch
    ):
        """`cloud = true` no longer forces monitoring: with monitor unset (false),
        a cloud directory is scanned once, exactly like any other monitor=false
        directory -- by the manager, after SERVING, not walked here. The monitor
        flag is the only switch."""
        root = tmp_path / "cloudroot"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), cloud=True, monitor=False))

        seen_urls = []
        real_discover = resolve_mod.discover_sources

        def spy(source, registry=None, credentials_config=None):
            seen_urls.append(source.url)
            return real_discover(source, registry, credentials_config)

        monkeypatch.setattr(resolve_mod, "discover_sources", spy)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert monitored_sources == []  # cloud alone does NOT monitor anymore
        assert static_sources == []
        assert str(root) not in seen_urls  # not walked before the server binds
        # The cloud flag rides along so the scan admits placeholders.
        assert [(s.url, s.cloud) for s in scan_once_sources] == [(str(root), True)]

    def test_cloud_with_monitor_is_monitored_not_expanded(self, tmp_path, monkeypatch):
        """`cloud = true, monitor = true` follows the same monitored path as any
        monitored directory: routed to monitored_sources, never pre-walked."""
        root = tmp_path / "cloudroot"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), cloud=True, monitor=True))

        seen_urls = []
        real_discover = resolve_mod.discover_sources

        def spy(source, registry=None):
            seen_urls.append(source.url)
            return real_discover(source, registry)

        monkeypatch.setattr(resolve_mod, "discover_sources", spy)

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

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

    def test_a_broken_trust_anchor_is_reported_as_config_not_a_missing_path(
        self, tmp_path, caplog
    ):
        """The tolerant path still skips, but says what actually happened (#608).

        A configured trust anchor that cannot be read means the server is coming
        up WITHOUT the stronger trust the operator asked for -- worth an error,
        not the same warning a missing static file gets.
        """
        from biopb_tensor_server.core.remote import CredentialProfile, CredentialsConfig

        good = tmp_path / "good.tif"
        _write_tiff(str(good))
        good_src = SourceConfig(url=str(good))
        # Bare-host form: the credentials resolve happens while expanding it.
        upstream = SourceConfig(
            url="grpcs://lab-store:8815", credentials_profile="lab-store"
        )
        cfg = ServerConfig(
            sources=[upstream, good_src],
            credentials=CredentialsConfig(
                default_profile=None,
                profiles=[
                    CredentialProfile(
                        name="lab-store",
                        storage_type="biopb-tensor",
                        tls_ca_file=str(tmp_path / "typo.pem"),
                    )
                ],
            ),
        )

        with caplog.at_level("ERROR"):
            resolved = resolve_all_sources(cfg, tolerant=True)

        # The healthy source is still served; the broken one is named loudly.
        assert [s.local_path for s in resolved] == [good.resolve()]
        assert "NOT SERVING" in caplog.text
        assert "tls_ca_file" in caplog.text


class TestAliasTreeRoot:
    """A local source's ``alias`` re-roots it (and everything under a configured
    folder) into its own catalog tree root -- the config-line analogue of a
    drag-dropped folder becoming its own root (see add_source_test's drop cases).
    The override is display-only and honored on the static/expand path only.
    """

    def test_alias_catalog_url_single_source_is_bare_root(self):
        from biopb_tensor_server.sources.resolve import _alias_catalog_url

        # Configured entry IS the source (file / dataset dir): alias is the root.
        assert _alias_catalog_url("exp", "/data/exp.zarr", "/data/exp.zarr") == "exp"

    def test_alias_catalog_url_preserves_subtree(self):
        from biopb_tensor_server.sources.resolve import _alias_catalog_url

        assert _alias_catalog_url("exp", "/data/exp", "/data/exp/a.tif") == "exp/a.tif"
        assert (
            _alias_catalog_url("exp", "/data/exp", "/data/exp/sub/b.tif")
            == "exp/sub/b.tif"
        )

    def test_alias_catalog_url_non_relativizable_is_bare_root(self):
        from biopb_tensor_server.sources.resolve import _alias_catalog_url

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

    def test_monitored_dir_keeps_its_alias_for_the_walk_to_apply(
        self, tmp_path, caplog
    ):
        """A monitored directory is not expanded here; its alias travels with it
        and the walk's registrations apply it (see source_manager_test)."""
        root = tmp_path / "watched"
        root.mkdir()
        _write_tiff(str(root / "image.tif"))
        cfg = _config(SourceConfig(url=str(root), alias="live", monitor=True))

        with caplog.at_level("WARNING", logger="biopb_tensor_server.sources.roots"):
            static_sources, upstream_sources, monitored_sources, scan_once_sources = (
                _resolve_serve_sources(cfg)
            )

        assert [(s.url, s.alias) for s in monitored_sources] == [(str(root), "live")]
        assert not caplog.records

    def test_monitored_single_file_alias_is_honored(self, tmp_path):
        """A ``monitor=true`` single *file* cannot be live-monitored, so it is
        registered once -- and its alias tree-root IS honored (it is never
        rescanned). No ignore-warning applies to it."""
        f = tmp_path / "img.tif"
        _write_tiff(str(f))
        cfg = _config(SourceConfig(url=str(f), alias="solo", monitor=True))

        static_sources, upstream_sources, monitored_sources, scan_once_sources = (
            _resolve_serve_sources(cfg)
        )

        assert monitored_sources == []
        assert [s.alias for s in scan_once_sources] == ["solo"]


class TestWriteDirInsideASourceDirectory:
    """upload stores are declined by the claims, but the walk still goes through
    them, so a write_dir inside a scanned directory is worth saying so."""

    @staticmethod
    def _warnings(caplog):
        return [r.message for r in caplog.records if "write_dir" in r.message]

    @pytest.mark.parametrize("monitor", [True, False])
    def test_a_write_dir_inside_a_source_directory_warns(
        self, tmp_path, caplog, monitor
    ):
        root = tmp_path / "data"
        (root / "uploads").mkdir(parents=True)
        cfg = ServerConfig(
            sources=[SourceConfig(url=str(root), monitor=monitor)],
            write_dir=root / "uploads",
        )

        _resolve_serve_sources(cfg)

        assert len(self._warnings(caplog)) == 1

    def test_a_write_dir_outside_every_source_directory_is_quiet(
        self, tmp_path, caplog
    ):
        root = tmp_path / "data"
        root.mkdir()
        cfg = ServerConfig(
            sources=[SourceConfig(url=str(root), monitor=True)],
            write_dir=tmp_path / "uploads",
        )

        _resolve_serve_sources(cfg)

        assert self._warnings(caplog) == []
