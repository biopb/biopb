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

    def test_dicom_series_not_grouped_under_cloud(self, tmp_path):
        # DICOM-series membership is content-derived (SeriesInstanceUID per slice)
        # and a dir can hold several series, so under a cloud root we do NOT group:
        # the series adapter bows out (returns None) and each .dcm is claimed
        # single-file by DicomAdapter instead. Gated on ctx.cloud_root (not
        # residency) so it also holds at resolve, when slices are resident.
        from biopb_tensor_server.adapters.dicom import DicomSeriesAdapter

        d = tmp_path / "series"
        d.mkdir()
        for i in range(3):
            (d / f"{i}.dcm").write_bytes(b"not a real dicom")
        ctx = _RaisingReadCtx(d, cloud_root=True)
        assert DicomSeriesAdapter.claim(ctx, DiscoveryState()) is None

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

    def test_serve_surface_refuses_until_resolved(self):
        # get_tensor_adapter (the GetFlightInfo / DoGet path) must NEVER resolve
        # on its own -- it refuses with SourceUnresolvedError until resolve() has
        # run, so the only thing that downloads is the explicit resolve action.
        from biopb_tensor_server.errors import SourceUnresolvedError

        import zarr

        with tempfile.TemporaryDirectory() as d:
            zpath = os.path.join(d, "img.zarr")
            zarr.open_array(zpath, mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")
            proxy = self._make_proxy(zpath)
            with pytest.raises(SourceUnresolvedError):
                proxy.get_tensor_adapter("s1")
            assert proxy.is_resolved is False  # the refusal did not hydrate
            # After an explicit resolve the serve surface delegates normally.
            proxy.resolve()
            ta = proxy.get_tensor_adapter("s1")
            assert list(ta.get_tensor_descriptor().shape) == [8, 8]

    def test_resolve_hydrates_and_delegates(self):
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
            # resolve() returns the full, now-resolved descriptor directly.
            desc = proxy.resolve()
            assert [list(t.shape) for t in desc.tensors] == [[64, 128]]
            assert desc.data_resident is True
            assert proxy.is_resolved is True
            # The authoritative type came from re-probing the hydrated content.
            assert fired == {"sid": "s1", "type": "zarr"}
            # Catalog surface now delegates to the resolved adapter.
            assert [list(t.shape) for t in proxy.list_tensor_descriptors()] == [
                [64, 128]
            ]
            assert proxy.is_resident() is True

    def test_resolves_once_under_repeated_resolve(self):
        import zarr

        with tempfile.TemporaryDirectory() as d:
            zpath = os.path.join(d, "img.zarr")
            zarr.open_array(zpath, mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")
            calls = []
            proxy = self._make_proxy(
                zpath, on_resolved=lambda sid, ad: calls.append(sid)
            )
            proxy.resolve()
            first = proxy._resolved
            proxy.resolve()
            assert proxy._resolved is first  # not rebuilt
            assert calls == ["s1"]  # on_resolved fired exactly once

    def test_resolution_failure_raises_source_unresolved(self):
        from biopb_tensor_server.errors import SourceUnresolvedError

        proxy = self._make_proxy("/nonexistent/path.zarr", source_type="ome-zarr")
        with pytest.raises(SourceUnresolvedError):
            proxy.resolve()


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

        # An explicit resolve -> backfills the DB with the concrete descriptor.
        desc = adapter.resolve()
        assert [list(t.shape) for t in desc.tensors] == [[32, 48]]
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


# --------------------------------------------------------------------------- #
# Server `resolve` action (streaming do_action)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestResolveAction:
    """The dedicated streaming `resolve` do_action: the SOLE resolution entry
    point. Emits empty-body heartbeats while the recall runs, then one non-empty
    terminal Result carrying the full DataSourceDescriptor."""

    def _server(self, source_id, adapter):
        from biopb_tensor_server.server import TensorFlightServer

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source(source_id, adapter)
        return server

    def test_resolve_action_streams_full_descriptor(self):
        import zarr
        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor
        from biopb_tensor_server.adapters import get_default_registry
        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

        with tempfile.TemporaryDirectory() as d:
            zpath = os.path.join(d, "img.zarr")
            zarr.open_array(
                zpath, mode="w", shape=(16, 24), chunks=(8, 12), dtype="uint8"
            )
            cfg = SourceConfig(url=zpath, type="ome-zarr", source_id="cloud1")
            proxy = UnresolvedSourceAdapter(cfg, get_default_registry())
            server = self._server("cloud1", proxy)

            action = flight.Action("resolve", b"cloud1")
            # do_action yields raw bytes (the Flight framework wraps each in a Result).
            bodies = [bytes(r) for r in server.do_action(None, action)]
            terminal = [b for b in bodies if b]
            assert len(terminal) == 1  # exactly one descriptor
            desc = DataSourceDescriptor.FromString(terminal[0])
            assert desc.source_id == "cloud1"
            assert [list(t.shape) for t in desc.tensors] == [[16, 24]]
            assert desc.data_resident is True
            assert proxy.is_resolved is True

    def test_resolve_action_emits_heartbeats_during_long_recall(self, monkeypatch):
        # With a tiny heartbeat interval and a slow resolve, the stream must carry
        # empty-body keep-alives BEFORE the terminal descriptor -- this is what
        # keeps a minutes-long recall under a proxy's idle read timeout.
        import time as _time
        import pyarrow.flight as flight
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor
        from biopb_tensor_server import server as server_mod

        monkeypatch.setattr(server_mod, "_RESOLVE_HEARTBEAT_SECONDS", 0.01)

        class _SlowAdapter:
            def resolve(self):
                _time.sleep(0.06)  # ~6 heartbeat intervals
                return DataSourceDescriptor(source_id="slow")

        server = self._server("slow", _SlowAdapter())
        action = flight.Action("resolve", b"slow")
        bodies = [bytes(r) for r in server.do_action(None, action)]

        assert bodies.count(b"") >= 1  # at least one heartbeat
        assert bodies[-1] != b""  # terminal is the descriptor
        assert DataSourceDescriptor.FromString(bodies[-1]).source_id == "slow"

    def test_resolve_action_unknown_source_errors(self):
        import pyarrow.flight as flight

        server = self._server("cloud1", _SlowSentinel())
        action = flight.Action("resolve", b"missing")
        with pytest.raises(flight.FlightServerError, match="Source not found"):
            list(server.do_action(None, action))


class _SlowSentinel:
    """A registered placeholder so the server has a (different) source; the
    `resolve` test above targets a *missing* id to exercise the not-found path."""

    token = None


# --------------------------------------------------------------------------- #
# Change A: cloud-root flag plumbing (ClaimContext -> claim() -> resolve)
# --------------------------------------------------------------------------- #


class TestCloudRootFlag:
    def test_claim_context_carries_cloud_root(self, tmp_path):
        assert ClaimContext(tmp_path, cloud_root=True).cloud_root is True
        assert ClaimContext(tmp_path).cloud_root is False

    def test_discover_from_entries_sets_cloud_root_per_entry(self, tmp_path):
        # The cloud_filter callback decides cloud-ness per path; the flag must
        # reach the adapter's claim() via its ClaimContext.
        from biopb_tensor_server.discovery import (
            AdapterRegistry,
            discover_sources_from_entries,
        )

        seen = {}

        class _Recorder:
            @classmethod
            def claim(cls, ctx, state):
                seen[ctx.path_str] = ctx.cloud_root
                return None

        registry = AdapterRegistry()
        registry.register_with_type("recorder", _Recorder)

        cloud_dir = str(tmp_path / "cloud")
        plain_dir = str(tmp_path / "plain")
        entries = [(cloud_dir, True, None), (plain_dir, True, None)]
        discover_sources_from_entries(
            entries,
            registry,
            cloud_filter=lambda p: p == cloud_dir,
        )
        assert seen[cloud_dir] is True
        assert seen[plain_dir] is False


# --------------------------------------------------------------------------- #
# Change B: single-file fallback ban for content-membership formats under cloud
# --------------------------------------------------------------------------- #


class TestCloudMultiFileBan:
    """Under a cloud root, OME-TIFF and DICOM-series do not group: each file
    falls back to its own single-file source. The gate is ctx.cloud_root (not
    residency) so it also holds at resolve, when the files are resident."""

    def test_ome_tiff_does_not_group_under_cloud(self, tmp_path):
        # Even a *resident* .tif must not be sniffed/grouped under cloud -- the
        # raising ctx proves no content read happens.
        from biopb_tensor_server.adapters.aicsimageio import OmeTiffAdapter

        f = tmp_path / "img.tif"
        f.write_bytes(b"II*\x00real-bytes-but-cloud")
        assert (
            OmeTiffAdapter.claim(_RaisingReadCtx(f, cloud_root=True), DiscoveryState())
            is None
        )

    def test_companion_ome_skipped_under_cloud(self, tmp_path):
        from biopb_tensor_server.adapters.aicsimageio import OmeTiffAdapter

        f = tmp_path / "set.companion.ome"
        f.write_text("<OME/>")
        assert (
            OmeTiffAdapter.claim(_RaisingReadCtx(f, cloud_root=True), DiscoveryState())
            is None
        )

    def test_cloud_tif_dir_yields_single_file_sources(self, tmp_path):
        # The whole-registry behavior: a dir of .tif files under cloud produces
        # one single-file (generic aics) claim per .tif, never a grouped set.
        from biopb_tensor_server.adapters import get_default_registry

        registry = get_default_registry()
        claims = []
        for i in range(3):
            f = tmp_path / f"plane_{i}.tif"
            f.write_bytes(b"II*\x00")
            c = registry.get_claims_for_path(
                ClaimContext(f, cloud_root=True), DiscoveryState()
            )
            claims.append(c[0] if c else None)
        assert all(c is not None for c in claims)
        assert all(c.source_type == "aics" for c in claims)
        # Each is its own primary_path (single-file), no multi-member grouping.
        assert all(c.member_paths == {c.primary_path} for c in claims)

    def test_dicom_series_bows_out_to_single_file_under_cloud(
        self, tmp_path, force_nonresident
    ):
        from biopb_tensor_server.adapters.dicom import (
            DicomAdapter,
            DicomSeriesAdapter,
        )

        d = tmp_path / "series"
        d.mkdir()
        files = [d / f"{i}.dcm" for i in range(3)]
        for f in files:
            f.write_bytes(b"not a real dicom")
        # The series adapter bows out under cloud (no grouping)...
        assert (
            DicomSeriesAdapter.claim(ClaimContext(d, cloud_root=True), DiscoveryState())
            is None
        )
        # ...and each slice is instead claimed single-file, deferred unresolved by
        # DicomAdapter's own residency gate (force_nonresident makes it defer).
        claim = DicomAdapter.claim(ClaimContext(files[0], cloud_root=True), DiscoveryState())
        assert claim is not None
        assert claim.source_type == "dicom"
        assert claim.unresolved is True


# --------------------------------------------------------------------------- #
# Change C: dir-claiming records the directory as the only member
# --------------------------------------------------------------------------- #


class TestDirClaimingMembership:
    def test_tiff_sequence_member_is_dir_only(self, tmp_path):
        import numpy as np
        import tifffile

        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        # Plain numbered sequence (img_*/OME/MicroManager names are excluded by
        # _group_tiff_sequence on purpose).
        for i in range(3):
            tifffile.imwrite(
                tmp_path / f"frame_{i:03d}.tif", np.zeros((4, 4), dtype="uint8")
            )
        state = DiscoveryState()
        claim = TiffSequenceAdapter.claim(ClaimContext(tmp_path), state)
        assert claim is not None
        assert claim.member_paths == {str(tmp_path)}

    def test_ndtiff_member_is_dir_plus_index(self, tmp_path):
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        (tmp_path / "NDTiff.index").write_bytes(b"idx")
        (tmp_path / "NDTiffStack_1.tif").write_bytes(b"tif")
        state = DiscoveryState()
        claim = NdTiffAdapter.claim(ClaimContext(tmp_path), state)
        assert claim is not None
        # Dir + the recall-free index marker; never the interior stack TIFFs.
        assert str(tmp_path / "NDTiffStack_1.tif") not in claim.member_paths

    def test_zarr_zattrs_only_defers_without_reading_under_nonresident(
        self, tmp_path, force_nonresident
    ):
        # ZarrAdapter's .zattrs-only branch now has the same residency gate as
        # OmeZarrAdapter: a non-resident .zattrs defers without a content read.
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        store = tmp_path / "plain.zarr"
        store.mkdir()
        (store / ".zattrs").write_text("}{ not json")  # would explode if read
        claim = ZarrAdapter.claim(_RaisingReadCtx(store), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is True
        assert claim.source_type == "zarr"


# --------------------------------------------------------------------------- #
# Change D: cloud signatures are residency-invariant
# --------------------------------------------------------------------------- #


class TestCloudSignatureInvariance:
    def test_cloud_file_signature_is_identity_only(self, tmp_path):
        mgr = _make_manager(_FakeServer())
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        st = f.stat()
        cloud_sig = mgr._build_entry_signature(st, is_directory=False, cloud=True)
        plain_sig = mgr._build_entry_signature(st, is_directory=False, cloud=False)
        assert cloud_sig == (st.st_dev, st.st_ino)
        assert plain_sig != cloud_sig  # plain carries size/mtime/ctime

    def test_hydration_does_not_change_cloud_signature(self, tmp_path):
        # Simulate a recall: size + mtime/ctime change. Under cloud the signature
        # is keyed on identity (dev/ino) only, so it is unchanged -> the rescan
        # will not put the just-resolved source in changed_ids.
        import os as _os

        mgr = _make_manager(_FakeServer())
        f = tmp_path / "x.bin"
        f.write_bytes(b"placeholder-stub")
        before = mgr._build_entry_signature(f.stat(), is_directory=False, cloud=True)
        # "Hydrate": grow the file and bump times.
        f.write_bytes(b"hydrated-full-content-now-much-larger")
        _os.utime(f, None)
        after = mgr._build_entry_signature(f.stat(), is_directory=False, cloud=True)
        assert before == after
        # A non-cloud signature WOULD change on the same hydration.
        assert mgr._build_entry_signature(
            f.stat(), is_directory=False, cloud=False
        ) != before


# --------------------------------------------------------------------------- #
# Change E: resolve error surfacing (retriable vs permanent)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestResolveErrorSurfacing:
    def _make_proxy(self, url, source_type="ome-zarr"):
        from biopb_tensor_server.adapters import get_default_registry
        from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter

        cfg = SourceConfig(url=url, type=source_type, source_id="s1")
        return UnresolvedSourceAdapter(cfg, get_default_registry())

    def test_reclaim_oserror_surfaces_retriable(self, monkeypatch):
        # An OSError while re-claiming (recall/IO) must surface as retriable,
        # NOT silently degrade to the claim-time guessed type.
        from biopb_tensor_server.errors import SourceResolveRetriableError

        proxy = self._make_proxy("/data/cloud/x.zarr")

        def _boom(ctx, state):
            raise OSError("recall failed")

        monkeypatch.setattr(proxy._registry, "get_claims_for_path", _boom)
        with pytest.raises(SourceResolveRetriableError):
            proxy.resolve()

    def test_create_from_config_oserror_is_retriable(self, monkeypatch, tmp_path):
        import zarr

        from biopb_tensor_server.errors import SourceResolveRetriableError

        zpath = str(tmp_path / "img.zarr")
        zarr.open_array(zpath, mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")
        proxy = self._make_proxy(zpath, source_type="ome-zarr")

        # Force the open/hydrate step (create_from_config of the resolved adapter
        # class) to raise an OSError -> retriable, not permanent. A bare zarr array
        # (.zarray present) resolves via ZarrAdapter: OmeZarrAdapter declines a
        # non-multiscales store, so the re-claim refines the guessed "ome-zarr" to
        # the authoritative "zarr".
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        def _boom(cls, cfg, creds=None):
            raise OSError("disk vanished mid-recall")

        monkeypatch.setattr(ZarrAdapter, "create_from_config", classmethod(_boom))
        with pytest.raises(SourceResolveRetriableError):
            proxy.resolve()

    def test_permanent_failure_is_plain_unresolved(self):
        # A nonexistent path: re-claim yields nothing and create_from_config
        # fails permanently (no retriable cause) -> plain SourceUnresolvedError,
        # and NOT the retriable subclass.
        from biopb_tensor_server.errors import (
            SourceResolveRetriableError,
            SourceUnresolvedError,
        )

        proxy = self._make_proxy("/nonexistent/path.zarr", source_type="zarr")
        with pytest.raises(SourceUnresolvedError) as exc_info:
            proxy.resolve()
        assert not isinstance(exc_info.value, SourceResolveRetriableError)


# --------------------------------------------------------------------------- #
# Plain Zarr enabled: OME-Zarr keeps priority; bare arrays resolve to zarr
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestZarrOmeZarrPriority:
    """ZarrAdapter is registered after OmeZarrAdapter. Order is load-bearing
    (callers take claims[0]); a real OME-Zarr stays ome-zarr, a bare zarr array
    resolves to plain zarr, and under cloud both defer with OME-Zarr winning."""

    def _registry(self):
        from biopb_tensor_server.adapters import get_default_registry

        return get_default_registry()

    def test_registry_orders_ome_zarr_before_zarr(self):
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        adapters = self._registry()._adapters
        assert OmeZarrAdapter in adapters and ZarrAdapter in adapters
        assert adapters.index(OmeZarrAdapter) < adapters.index(ZarrAdapter)

    def test_real_ome_zarr_claimed_as_ome_zarr(self, tmp_path):
        import zarr

        store = tmp_path / "img.zarr"
        g = zarr.open_group(str(store), mode="w")
        g.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
        g.create_dataset("0", shape=(4, 4), chunks=(4, 4), dtype="uint8")

        claims = self._registry().get_claims_for_path(
            ClaimContext(store), DiscoveryState()
        )
        assert claims and claims[0].source_type == "ome-zarr"

    def test_bare_zarr_array_claimed_as_zarr(self, tmp_path):
        import zarr

        store = tmp_path / "arr.zarr"
        zarr.open_array(str(store), mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")

        claims = self._registry().get_claims_for_path(
            ClaimContext(store), DiscoveryState()
        )
        assert claims and claims[0].source_type == "zarr"
        assert claims[0].unresolved is False  # resident -> not deferred

    def test_nonresident_bare_array_defers_as_zarr_not_ome_zarr(
        self, tmp_path, force_nonresident
    ):
        # A top-level .zarray makes this a definite plain array: OmeZarrAdapter
        # declines (recall-free), ZarrAdapter defers it, so claims[0] is the
        # certain "zarr" -- not a provisional "ome-zarr" guess. Reads explode to
        # prove neither adapter opened content.
        store = tmp_path / "arr.zarr"
        store.mkdir()
        (store / ".zarray").write_text("ignored")

        claims = self._registry().get_claims_for_path(
            _RaisingReadCtx(store), DiscoveryState()
        )
        assert claims and claims[0].source_type == "zarr"
        assert claims[0].unresolved is True

    def test_nonresident_zattrs_only_defers_as_ome_zarr(
        self, tmp_path, force_nonresident
    ):
        # Only .zattrs (no .zarray): both adapters defer, and OmeZarr's priority
        # wins claims[0]. Resolution refines it once the store is resident.
        store = tmp_path / "grp.zarr"
        store.mkdir()
        (store / ".zattrs").write_text("}{ not json")

        claims = self._registry().get_claims_for_path(
            _RaisingReadCtx(store), DiscoveryState()
        )
        assert claims and claims[0].source_type == "ome-zarr"
        assert claims[0].unresolved is True
