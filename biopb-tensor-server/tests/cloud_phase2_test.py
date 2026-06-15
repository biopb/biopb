"""Cloud-storage phase 2: unresolved adapter + lazy resolution + cloud opt-in.

Covers the recall-free claim contract (each adapter avoids byte reads when its
target is not resident), the UnresolvedSourceAdapter proxy and its resolve-on-
serve hook, and the source-manager wiring that registers cloud sources unresolved
and backfills the metadata DB once they resolve.

Residency is simulated by patching ``_is_offline_placeholder`` -- the same
stat-only signal the real code uses -- so a real on-disk store can stand in for a
dehydrated cloud placeholder without a special filesystem.
"""

import json
import os
import tempfile

import pytest

from biopb_tensor_server import discovery
from biopb_tensor_server import source_manager as sm_mod
from biopb_tensor_server.adapters import dicom as dicom_mod
from biopb_tensor_server.adapters import tiff as tiff_mod
from biopb_tensor_server.config import SourceConfig, parse_config
from biopb_tensor_server.discovery import (
    ClaimContext,
    DiscoveryState,
    SourceClaim,
    should_skip_walk_entry,
)


def _zarr_available():
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def force_nonresident(monkeypatch):
    """Make every local path look like a dehydrated cloud placeholder.

    Patches the per-module ``_is_offline_placeholder`` bindings so both
    ``ClaimContext.is_resident`` (via discovery) and the direct callers in the
    tiff/dicom adapters see non-residency.
    """
    fake = lambda path, stat_result=None: True  # noqa: E731
    monkeypatch.setattr(discovery, "_is_offline_placeholder", fake)
    monkeypatch.setattr(tiff_mod, "_is_offline_placeholder", fake)
    monkeypatch.setattr(dicom_mod, "_is_offline_placeholder", fake)
    monkeypatch.setattr(sm_mod, "_is_offline_placeholder", fake)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestCloudConfig:
    def test_cloud_flag_parses(self):
        cfg = parse_config({"sources": [{"url": "/data/cloud/", "cloud": True}]})
        assert cfg.sources[0].cloud is True

    def test_cloud_defaults_false(self):
        cfg = parse_config({"sources": [{"url": "/data/local/"}]})
        assert cfg.sources[0].cloud is False


# --------------------------------------------------------------------------- #
# Residency primitive + walk admission
# --------------------------------------------------------------------------- #


class TestClaimContextResidency:
    def test_directory_is_resident(self, tmp_path):
        # A directory legitimately reports st_blocks == 0 on some filesystems;
        # never flag it (mirrors the phase-1 SourceAdapter.is_resident fix).
        assert ClaimContext(tmp_path).is_resident() is True

    def test_resident_file(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world payload")
        assert ClaimContext(f).is_resident() is True

    def test_placeholder_file_not_resident(self, tmp_path, monkeypatch):
        f = tmp_path / "x.bin"
        f.write_bytes(b"content")
        monkeypatch.setattr(
            discovery, "_is_offline_placeholder", lambda p, s=None: True
        )
        assert ClaimContext(f).is_resident() is False


class TestShouldSkipAdmit:
    def test_admit_lifts_file_residency_skip(self, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(
            discovery, "_is_offline_placeholder", lambda p, s=None: True
        )
        p = Path("/data/cloud/img.dcm")
        # Default: a non-resident file is skipped.
        assert should_skip_walk_entry(p, is_dir=False) is True
        # Cloud root: admitted.
        assert should_skip_walk_entry(p, is_dir=False, admit_nonresident=True) is False

    def test_admit_still_prunes_hidden_and_system(self, monkeypatch):
        from pathlib import Path

        # Hidden entries and system/cloud dirs are pruned even under admit.
        assert (
            should_skip_walk_entry(
                Path("/data/.hidden"), is_dir=False, admit_nonresident=True
            )
            is True
        )
        assert (
            should_skip_walk_entry(
                Path("/home/u/OneDrive"), is_dir=True, admit_nonresident=True
            )
            is True
        )


# --------------------------------------------------------------------------- #
# Content-free adapters: the recall-free guard
# --------------------------------------------------------------------------- #


class _RaisingReadCtx(ClaimContext):
    """A ClaimContext whose content reads explode, proving claim() never reads."""

    def read_text(self, subpath: str = "") -> str:
        raise AssertionError("claim() must not read content for this adapter")


class TestContentFreeClaimsDoNotRead:
    """Extension/structure-only adapters recognize a source without any byte read.

    A regression guard: if a future edit reintroduces a content read into one of
    these claim()s, the raising context turns it into a hard failure.
    """

    @pytest.mark.parametrize(
        "filename, source_type",
        [
            ("scan.nii", "nifti"),
            ("scan.nii.gz", "nifti"),
            ("img.czi", "zeiss"),
            ("img.lsm", "zeiss"),
            ("img.lif", "leica"),
            ("img.nd2", "nikon"),
            ("img.dv", "dv"),
            ("img.oif", "olympus"),
        ],
    )
    def test_extension_only_adapters_claim_without_reading(
        self, tmp_path, filename, source_type
    ):
        from biopb_tensor_server.adapters import get_default_registry

        f = tmp_path / filename
        f.write_bytes(b"\x00\x01\x02\x03")
        ctx = _RaisingReadCtx(f)
        state = DiscoveryState()
        registry = get_default_registry()
        claims = registry.get_claims_for_path(ctx, state)
        assert claims, f"{filename} should be claimed"
        assert claims[0].source_type == source_type

    def test_ndtiff_claims_by_index_existence_without_reading(self, tmp_path):
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        d = tmp_path / "acq"
        d.mkdir()
        (d / "NDTiff.index").write_bytes(b"binary-index")
        (d / "NDTiffStack_1.tif").write_bytes(b"tiff")
        ctx = _RaisingReadCtx(d)
        state = DiscoveryState()
        claim = NdTiffAdapter.claim(ctx, state)
        assert claim is not None
        assert claim.source_type == NdTiffAdapter.SOURCE_TYPE


# --------------------------------------------------------------------------- #
# Reader adapters: residency-guarded defer branch
# --------------------------------------------------------------------------- #


class TestReaderDeferBranches:
    """The 5 content-reading adapters defer (claim unresolved) when non-resident."""

    def test_ome_zarr_defers_without_parsing_zattrs(self, tmp_path, force_nonresident):
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        store = tmp_path / "img.zarr"
        store.mkdir()
        # Garbage .zattrs: proves the defer branch returns before json.loads.
        (store / ".zattrs").write_text("}{ not json")
        claim = OmeZarrAdapter.claim(_RaisingReadCtx(store), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is True
        assert claim.source_type == "ome-zarr"

    def test_ome_zarr_resident_still_resolves_subtype(self, tmp_path):
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        store = tmp_path / "img.zarr"
        store.mkdir()
        (store / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": []}]}))
        claim = OmeZarrAdapter.claim(ClaimContext(store), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is False
        assert claim.source_type == "ome-zarr"

    def test_micromanager_defers_without_parsing_metadata(
        self, tmp_path, force_nonresident
    ):
        from biopb_tensor_server.adapters.tiff import MicroManagerLegacyAdapter

        d = tmp_path / "mm"
        d.mkdir()
        (d / "metadata.txt").write_text("}{ not json")  # would fail to parse
        (d / "img_channel000.tif").write_bytes(b"tiff")
        claim = MicroManagerLegacyAdapter.claim(ClaimContext(d), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is True
        assert claim.source_type == "micromanager-legacy"

    def test_dicom_single_defers_without_dcmread(self, tmp_path, force_nonresident):
        from biopb_tensor_server.adapters.dicom import DicomAdapter

        f = tmp_path / "slice.dcm"
        f.write_bytes(b"not a real dicom")  # would fail dcmread
        claim = DicomAdapter.claim(ClaimContext(f), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is True
        assert claim.source_type == "dicom"

    def test_dicom_series_defers_without_scanning_every_slice(
        self, tmp_path, force_nonresident
    ):
        from biopb_tensor_server.adapters.dicom import DicomSeriesAdapter

        d = tmp_path / "series"
        d.mkdir()
        for i in range(3):
            (d / f"{i}.dcm").write_bytes(b"not a real dicom")
        claim = DicomSeriesAdapter.claim(ClaimContext(d), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is True
        assert claim.source_type == "dicom-series"

    def test_ome_tiff_sniff_skipped_when_nonresident(self, tmp_path, force_nonresident):
        # A non-resident .tif: OmeTiffAdapter declines (skips the IFD sniff) so the
        # extension-only generic AICS adapter claims it instead as an image.
        from biopb_tensor_server.adapters.aicsimageio import (
            AicsImageIoAdapter,
            OmeTiffAdapter,
        )

        f = tmp_path / "img.tif"
        f.write_bytes(b"II*\x00not-a-real-tiff")
        assert OmeTiffAdapter.claim(_RaisingReadCtx(f), DiscoveryState()) is None
        generic = AicsImageIoAdapter.claim(ClaimContext(f), DiscoveryState())
        assert generic is not None


# --------------------------------------------------------------------------- #
# UnresolvedSourceAdapter proxy
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestUnresolvedProxy:
    def _make_proxy(self, url, source_type="ome-zarr", on_resolved=None):
        from biopb_tensor_server.adapters import get_default_registry
        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

        cfg = SourceConfig(url=url, type=source_type, source_id="s1")
        return UnresolvedSourceAdapter(
            cfg, get_default_registry(), on_resolved=on_resolved
        )

    def test_catalog_surface_is_empty_and_not_resident(self):
        proxy = self._make_proxy("/data/cloud/x.zarr")
        assert proxy.list_tensor_descriptors() == []
        assert proxy.is_resident() is False
        assert proxy.has_native_pyramid() is False
        desc = proxy.get_source_descriptor()
        assert list(desc.tensors) == []
        assert desc.data_resident is False
        assert desc.source_type == "ome-zarr"
        assert proxy.is_resolved is False

    def test_resolves_on_serve_and_delegates(self):
        import zarr

        with tempfile.TemporaryDirectory() as d:
            zpath = os.path.join(d, "img.zarr")
            zarr.open_array(
                zpath, mode="w", shape=(64, 128), chunks=(32, 64), dtype="uint16"
            )
            fired = {}
            proxy = self._make_proxy(
                zpath,
                source_type="ome-zarr",  # provisional guess; real type is "zarr"
                on_resolved=lambda sid, ad: fired.update(sid=sid, type=ad._source_type),
            )
            ta = proxy.get_tensor_adapter("s1")
            desc = ta.get_tensor_descriptor()
            assert list(desc.shape) == [64, 128]
            assert proxy.is_resolved is True
            # The authoritative type came from re-probing the hydrated content.
            assert fired == {"sid": "s1", "type": "zarr"}
            # Catalog surface now delegates to the resolved adapter.
            assert [list(t.shape) for t in proxy.list_tensor_descriptors()] == [
                [64, 128]
            ]
            assert proxy.is_resident() is True

    def test_resolves_once_under_repeated_serve(self):
        import zarr

        with tempfile.TemporaryDirectory() as d:
            zpath = os.path.join(d, "img.zarr")
            zarr.open_array(zpath, mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")
            calls = []
            proxy = self._make_proxy(
                zpath, on_resolved=lambda sid, ad: calls.append(sid)
            )
            proxy.get_tensor_adapter("s1")
            first = proxy._resolved
            proxy.get_tensor_adapter("s1")
            assert proxy._resolved is first  # not rebuilt
            assert calls == ["s1"]  # on_resolved fired exactly once

    def test_resolution_failure_raises_source_unresolved(self):
        from biopb_tensor_server.errors import SourceUnresolvedError

        proxy = self._make_proxy("/nonexistent/path.zarr", source_type="ome-zarr")
        with pytest.raises(SourceUnresolvedError):
            proxy.get_tensor_adapter("s1")


# --------------------------------------------------------------------------- #
# Source-manager registration decision
# --------------------------------------------------------------------------- #


class _FakeMetadataDb:
    def __init__(self):
        self.added = []

    def sync_source_added(self, source_id, adapter):
        self.added.append((source_id, adapter))

    def sync_source_removed(self, source_id):
        pass


class _FakeServer:
    def __init__(self):
        self.registered = {}
        self._metadata_db = _FakeMetadataDb()

    def register_source(self, source_id, adapter):
        self.registered[source_id] = adapter

    def unregister_source(self, source_id):
        self.registered.pop(source_id, None)


def _make_manager(server, cloud_roots=None, monitored=None):
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.source_manager import SourceManager

    return SourceManager(
        server=server,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        watcher=None,
        monitored_dirs=monitored or set(),
        cloud_roots=cloud_roots or set(),
    )


class TestUnresolvedDecision:
    def test_flagged_claim_is_unresolved(self, tmp_path):
        mgr = _make_manager(_FakeServer())
        claim = SourceClaim("ome-zarr", str(tmp_path / "x.zarr"), unresolved=True)
        assert mgr._claim_is_unresolved(claim) is True

    def test_resident_claim_outside_cloud_root_is_not_unresolved(self, tmp_path):
        mgr = _make_manager(_FakeServer())
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f))
        assert mgr._claim_is_unresolved(claim) is False

    def test_nonresident_file_under_cloud_root_is_unresolved(
        self, tmp_path, force_nonresident
    ):
        # Content-free file adapter (no flag), but its file content is a cloud
        # placeholder under a cloud root -> deferred so it is not opened eagerly.
        mgr = _make_manager(_FakeServer(), cloud_roots={tmp_path.resolve()})
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f))
        assert mgr._claim_is_unresolved(claim) is True

    def test_directory_member_not_false_flagged(self, tmp_path, force_nonresident):
        # Even with every path looking non-resident, a directory primary_path is
        # not flagged via the member check (is_file guard) -- only the adapter
        # flag would defer a dir-format source. Guards the macOS APFS dir case.
        mgr = _make_manager(_FakeServer(), cloud_roots={tmp_path.resolve()})
        store = tmp_path / "img.zarr"
        store.mkdir()
        claim = SourceClaim("ome-zarr", str(store))  # no unresolved flag, dir member
        assert mgr._claim_is_unresolved(claim) is False


# --------------------------------------------------------------------------- #
# End-to-end through the source manager
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestCloudRegistrationEndToEnd:
    def test_unresolved_registration_then_resolve_backfills_db(
        self, tmp_path, monkeypatch
    ):
        import zarr

        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

        store = tmp_path / "img.zarr"
        zarr.open_array(
            str(store), mode="w", shape=(32, 48), chunks=(16, 24), dtype="uint16"
        )

        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})

        # Register an ome-zarr claim flagged unresolved (as the defer branch would).
        claim = SourceClaim(
            "ome-zarr", str(store), source_id="cloud1", unresolved=True
        )
        assert mgr._register_source_claim(claim) is True

        # Registered as the proxy; catalog row carries NULL shape (empty tensors).
        adapter = server.registered["cloud1"]
        assert isinstance(adapter, UnresolvedSourceAdapter)
        assert adapter.list_tensor_descriptors() == []
        assert server._metadata_db.added[-1][0] == "cloud1"
        assert adapter.get_source_descriptor().data_resident is False

        # First serve resolves -> backfills the DB with the concrete descriptor.
        ta = adapter.get_tensor_adapter("cloud1")
        assert list(ta.get_tensor_descriptor().shape) == [32, 48]
        # on_resolved fired a second sync_source_added (the upsert backfill).
        assert [sid for sid, _ in server._metadata_db.added].count("cloud1") == 2
        resolved_adapter = server._metadata_db.added[-1][1]
        assert resolved_adapter.list_tensor_descriptors()[0].shape == [32, 48]


# --------------------------------------------------------------------------- #
# Precache safety
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestPrecacheSkipsUnresolved:
    def test_unresolved_source_never_reaches_descriptor_hook(self, monkeypatch):
        from biopb_tensor_server.adapters import get_default_registry
        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

        cfg = SourceConfig(url="/data/cloud/x.zarr", type="ome-zarr", source_id="s1")
        proxy = UnresolvedSourceAdapter(cfg, get_default_registry())

        # If precache (which loops list_tensor_descriptors) ever resolved the
        # source, it would hydrate cloud data in the background. Guard: an empty
        # tensor list means it skips before reaching the resolving get_tensor_adapter.
        def _boom(*a, **k):
            raise AssertionError("precache must not resolve an unresolved source")

        monkeypatch.setattr(proxy, "get_tensor_adapter", _boom)
        assert proxy.list_tensor_descriptors() == []
        assert proxy.is_resolved is False

    def test_real_precache_worker_does_not_resolve_unresolved(self, monkeypatch):
        # Drives the actual PrecacheWorker._process_source: the empty tensor list
        # makes it return before the resolving get_tensor_adapter, so the cloud
        # source is never hydrated by background warming.
        from biopb_tensor_server.adapters import get_default_registry
        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter
        from biopb_tensor_server.config import PrecacheConfig
        from biopb_tensor_server.precache import PrecacheWorker

        cfg = SourceConfig(url="/data/cloud/x.zarr", type="ome-zarr", source_id="s1")
        proxy = UnresolvedSourceAdapter(cfg, get_default_registry())

        def _boom(*a, **k):
            raise AssertionError("precache must not resolve an unresolved source")

        monkeypatch.setattr(proxy, "get_tensor_adapter", _boom)

        class _Srv:
            def _get_source_adapter(self, sid):
                return proxy if sid == "s1" else None

        worker = PrecacheWorker(_Srv(), PrecacheConfig())
        # Past the file-backend gate so the real source-processing logic runs.
        monkeypatch.setattr(worker, "_file_backend_active", lambda: True)
        assert worker._process_source("s1") is False
        assert proxy.is_resolved is False


# --------------------------------------------------------------------------- #
# Cloud rescan gating (the watcher path)
# --------------------------------------------------------------------------- #


class TestCloudRescanGating:
    """A cloud root is full-scanned with no mtime stability gate and no open-probe.

    Drives the real ``_handle_rescan`` pipeline (not ``_register_source_claim``
    directly) over a simulated dehydrated dataset, proving the dehydrated content
    is registered unresolved without ever being opened (no recall).
    """

    def test_rescan_registers_unresolved_without_opening_content(
        self, tmp_path, force_nonresident, monkeypatch
    ):
        root = tmp_path / "cloudroot"
        root.mkdir()
        store = root / "img.zarr"
        store.mkdir()
        # A recognizable OME-Zarr store; .zattrs content is irrelevant because the
        # (simulated) non-residency makes claim() defer before reading it.
        (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (store / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": []}]}))

        server = _FakeServer()
        mgr = _make_manager(
            server, cloud_roots={root.resolve()}, monitored={root}
        )

        # The open-for-append probe would recall a placeholder; the cloud gate must
        # bypass it entirely. Make it explode if ever reached during the rescan.
        def _no_probe(path):
            raise AssertionError(f"cloud rescan must not open-probe {path}")

        monkeypatch.setattr(mgr, "_can_open_for_append", _no_probe)

        mgr._handle_rescan()

        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

        assert len(server.registered) == 1
        adapter = next(iter(server.registered.values()))
        assert isinstance(adapter, UnresolvedSourceAdapter)
        assert adapter.list_tensor_descriptors() == []
        assert adapter.get_source_descriptor().data_resident is False
        # Catalogued in the DB with an empty (NULL-shape) row.
        assert server._metadata_db.added

    def test_noncloud_root_defers_fresh_dataset_via_stability_window(
        self, tmp_path, force_nonresident
    ):
        # Same layout but NOT a cloud root. The freshly-created store is within the
        # stability window (mtime ~= now), and a non-cloud entry does NOT bypass it,
        # so nothing is registered this rescan -- the contrast that shows the cloud
        # gate is what admits a fresh dehydrated dataset immediately (test above).
        root = tmp_path / "plainroot"
        root.mkdir()
        store = root / "img.zarr"
        store.mkdir()
        (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (store / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": []}]}))

        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots=set(), monitored={root})
        mgr._handle_rescan()
        assert server.registered == {}
