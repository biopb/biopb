"""Cloud-storage phase 2: unresolved sources + lazy resolution + cloud opt-in.

Covers the recall-free claim contract (each adapter avoids byte reads when its
target is not resident), and the source-manager wiring that catalogs a cloud
source as ``needs_recall`` with no adapter and registers it, backfilling the
metadata DB, when a client resolves it.

Residency is simulated by patching ``_is_offline_placeholder`` -- the same
stat-only signal the real code uses -- so a real on-disk store can stand in for a
dehydrated cloud placeholder without a special filesystem.
"""

import json
import os
import tempfile
import threading
import time

import pytest
from biopb_tensor_server.adapters import tiff as tiff_mod
from biopb_tensor_server.core import discovery
from biopb_tensor_server.core.adapter_base import SourceAdapter
from biopb_tensor_server.core.config import SourceConfig, parse_config
from biopb_tensor_server.core.discovery import (
    ClaimContext,
    DiscoveryState,
    LiveLocalContext,
    SourceClaim,
    should_skip_walk_entry,
)
from biopb_tensor_server.core.errors import SourceUnresolvedError
from biopb_tensor_server.sources import reconciler as rec_mod
from biopb_tensor_server.sources.entry_stat import build_entry_signature

from tests import catalog_server, make_manager


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
    monkeypatch.setattr(rec_mod, "_is_offline_placeholder", fake)


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


class _RaisingReadCtx(LiveLocalContext):
    """A local ClaimContext whose content reads explode, proving claim() never reads."""

    def read_text(self, subpath: str = "") -> str:
        raise AssertionError("claim() must not read content for this adapter")

    def open(self, mode: str = "rb") -> object:
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
            ("img.tif", "tiff"),
            ("img.lsm", "lsm"),
            ("scan.nii.gz", "nifti"),
            ("img.czi", "czi"),
            ("img.lif", "lif"),
            ("img.nd2", "nd2"),
            ("img.dv", "deltavision"),
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

    @pytest.mark.parametrize(
        "filename, source_type",
        [
            ("img.tif", "tiff"),
            ("img.lsm", "lsm"),
            ("img.czi", "czi"),
            ("img.lif", "lif"),
            ("img.dv", "deltavision"),
            ("img.nd2", "nd2"),
        ],
    )
    def test_native_adapters_claim_a_dehydrated_placeholder(
        self, tmp_path, force_nonresident, filename, source_type
    ):
        """The native claims stay definite for a non-resident file.

        They read nothing, so they cannot recall it, and deferring is
        ``_claim_is_unresolved``'s job. Declining here would also outlive the
        placeholder: the resolve-time re-claim still carries
        ``cloud_root=True``, so a hydrated file would never reach the native
        adapter (biopb/biopb#799).
        """
        from biopb_tensor_server.adapters import get_default_registry

        f = tmp_path / filename
        f.write_bytes(b"II*\x00")
        claims = get_default_registry().get_claims_for_path(
            _RaisingReadCtx(f, cloud_root=True), DiscoveryState()
        )
        assert [c.source_type for c in claims] == [source_type]
        assert claims[0].unresolved is False

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

    def test_qptiff_claims_by_extension_without_opening_file(
        self, tmp_path, monkeypatch
    ):
        # Suffix-only claim (biopb/biopb#135): a .qptiff is claimed recall-free --
        # even under a cloud root, where a byte read recalls the whole placeholder.
        # The dropped .tif vendor-XML sniff opened the file with tifffile.TiffFile
        # (not ctx.read_text, so _RaisingReadCtx alone wouldn't catch it), so make
        # TiffFile explode to prove claim() never reaches it. Deferring a
        # non-resident file is the manager's job (_claim_is_unresolved), not
        # claim()'s, so the claim itself is definite here -- like CZI/NIfTI.
        import tifffile
        from biopb_tensor_server.adapters.qptiff import QptiffAdapter

        def _boom(*args, **kwargs):
            raise AssertionError("claim() must not open the file (recall under cloud)")

        monkeypatch.setattr(tifffile, "TiffFile", _boom)

        f = tmp_path / "slide.qptiff"
        f.write_bytes(b"II*\x00not-a-real-qptiff")
        claim = QptiffAdapter.claim(
            _RaisingReadCtx(f, cloud_root=True), DiscoveryState()
        )
        assert claim is not None
        assert claim.source_type == "qptiff"
        assert claim.unresolved is False

    def test_qptiff_declines_tif_recall_free_under_cloud(self, tmp_path, monkeypatch):
        # The other half of the suffix-only policy: a .tif QPTIFF is NOT sniffed,
        # so QptiffAdapter declines it without opening the file (it falls through
        # to the generic bioio adapter). Guards that no sniff creeps back in.
        import tifffile
        from biopb_tensor_server.adapters.qptiff import QptiffAdapter

        def _boom(*args, **kwargs):
            raise AssertionError("declining a .tif must not open the file")

        monkeypatch.setattr(tifffile, "TiffFile", _boom)

        f = tmp_path / "slide.tif"
        f.write_bytes(b"II*\x00not-a-real-qptiff")
        claim = QptiffAdapter.claim(
            _RaisingReadCtx(f, cloud_root=True), DiscoveryState()
        )
        assert claim is None


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
        from biopb_tensor_server.adapters.bioio import AicsImageIoAdapter
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        f = tmp_path / "img.tif"
        f.write_bytes(b"II*\x00not-a-real-tiff")
        assert OmeTiffAdapter.claim(_RaisingReadCtx(f), DiscoveryState()) is None
        generic = AicsImageIoAdapter.claim(ClaimContext(f), DiscoveryState())
        assert generic is not None

    def test_tiff_sequence_defers_under_cloud(self, tmp_path):
        # TiffSequenceAdapter.__init__ opens EVERY file in the sequence to validate
        # dimensions; under a cloud root those opens recall the whole sequence at
        # startup and wedge health at STARTING (biopb/biopb#173). The claim records
        # only the directory, so the per-file residency gate in _claim_is_unresolved
        # cannot catch it -- the adapter must defer itself. Gated on cloud_root (not
        # residency) so the deferral also holds at resolve. claim() is metadata-free
        # (it only globs/templates names), so _RaisingReadCtx proves no byte read.
        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        d = tmp_path / "seq"
        d.mkdir()
        for i in range(30):  # clears the claim floor (_MIN_TIFF_FILES)
            (d / f"frame{i:03d}.tif").write_bytes(b"II*\x00not-a-real-tiff")
        claim = TiffSequenceAdapter.claim(
            _RaisingReadCtx(d, cloud_root=True), DiscoveryState()
        )
        assert claim is not None
        assert claim.unresolved is True
        assert claim.source_type == "tiff-sequence"

    def test_tiff_sequence_resolves_eagerly_outside_cloud(self, tmp_path):
        # Outside a cloud root the sequence claims as before (resolved), so the
        # ordinary local fast path is unchanged.
        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        d = tmp_path / "seq"
        d.mkdir()
        for i in range(30):  # clears the claim floor (_MIN_TIFF_FILES)
            (d / f"frame{i:03d}.tif").write_bytes(b"II*\x00not-a-real-tiff")
        claim = TiffSequenceAdapter.claim(ClaimContext(d), DiscoveryState())
        assert claim is not None
        assert claim.unresolved is False
        assert claim.source_type == "tiff-sequence"


class _FakeMetadataDb:
    def __init__(self):
        self.added = []
        self.pending = []  # (source_id, recall) of each pending row written

    def sync_source_added(self, source_id, adapter, record=None):
        self.added.append((source_id, adapter))

    def sync_pending_source(
        self, claim, catalog_url=None, error=None, recall=False, record=None
    ):
        self.pending.append((claim.source_id, recall))

    def sync_pending_sources(self, rows):
        for row in rows:
            self.sync_pending_source(row.claim, row.catalog_url, recall=row.recall)

    def sync_source_removed(self, source_id):
        pass


class _FakeServer:
    def __init__(self):
        self.registered = {}
        self._metadata_db = _FakeMetadataDb()
        # The reconciler reaches adapters through `sources`, the real server's
        # registry; `registered` is this fake's own record of the same calls.
        self.sources = self.registered

    def register_source(self, source_id, adapter):
        self.registered[source_id] = adapter

    def swap_source(self, source_id, adapter):
        displaced = self.registered.get(source_id)
        self.registered[source_id] = adapter
        return adapter, displaced

    def unregister_source(self, source_id):
        self.registered.pop(source_id, None)

    def set_full_scan_in_progress(self, in_progress):
        pass

    def set_last_full_scan(self, timestamp):
        pass


def _make_manager(server, cloud_roots=None, monitored=None):
    from biopb_tensor_server.adapters import get_default_registry

    return make_manager(
        server=server,
        metadata_db=server._metadata_db,
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        monitored_dirs=monitored or set(),
        cloud_roots=cloud_roots or set(),
    )


class TestUnresolvedDecision:
    def test_flagged_claim_is_unresolved(self, tmp_path):
        mgr = _make_manager(_FakeServer())
        claim = SourceClaim("ome-zarr", str(tmp_path / "x.zarr"), unresolved=True)
        assert mgr._reconciler._claim_is_unresolved(claim) is True

    def test_resident_claim_outside_cloud_root_is_not_unresolved(self, tmp_path):
        mgr = _make_manager(_FakeServer())
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f))
        assert mgr._reconciler._claim_is_unresolved(claim) is False

    def test_nonresident_file_under_cloud_root_is_unresolved(
        self, tmp_path, force_nonresident
    ):
        # Content-free file adapter (no flag), but its file content is a cloud
        # placeholder under a cloud root -> deferred so it is not opened eagerly.
        mgr = _make_manager(_FakeServer(), cloud_roots={tmp_path.resolve()})
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f))
        assert mgr._reconciler._claim_is_unresolved(claim) is True

    def test_directory_member_not_false_flagged(self, tmp_path, force_nonresident):
        # Even with every path looking non-resident, a directory primary_path is
        # not flagged via the member check (is_file guard) -- only the adapter
        # flag would defer a dir-format source. Guards the macOS APFS dir case.
        mgr = _make_manager(_FakeServer(), cloud_roots={tmp_path.resolve()})
        store = tmp_path / "img.zarr"
        store.mkdir()
        claim = SourceClaim("ome-zarr", str(store))  # no unresolved flag, dir member
        assert mgr._reconciler._claim_is_unresolved(claim) is False


class _ResidencyAdapter:
    """Stand-in carrying the *real* ``is_resident()`` off the adapter base.

    Not a canned bool: the gate's failure mode is asking a weaker question than
    the adapter would, so the test runs the adapter's own answer.
    """

    def __init__(self, source_url):
        self._source_url = str(source_url)

    def is_resident(self):
        return SourceAdapter.is_resident(self)


class TestShouldWarm:
    """Residency gate the precache worker consults before warming (#174).

    Asks the registered adapter, live, so a source that re-dehydrates after
    registration is skipped instead of recalled on a later backlog pass.
    """

    def _register(self, mgr, claim, server=None):
        mgr._reconciler._state.claims[claim.source_id] = claim
        if server is not None:
            server.register_source(
                claim.source_id, _ResidencyAdapter(claim.primary_path)
            )

    def test_unknown_source_not_warmed(self, tmp_path):
        mgr = _make_manager(_FakeServer())
        assert mgr.should_warm("nope") is False

    def test_local_source_outside_cloud_root_always_warms(self, tmp_path):
        # No adapter registered, and none needed: outside a cloud root the gate
        # short-circuits rather than paying for a stat walk per source.
        mgr = _make_manager(_FakeServer())
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f), source_id="s1")
        self._register(mgr, claim)
        assert mgr.should_warm("s1") is True

    def test_resident_cloud_source_warms(self, tmp_path):
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f), source_id="s1")
        self._register(mgr, claim, server)
        assert mgr.should_warm("s1") is True

    def test_rehydrated_cloud_source_skipped(self, tmp_path, force_nonresident):
        # Registered as a normal adapter while resident, then OneDrive evicted the
        # bytes: should_warm now returns False so the warm read never recalls them.
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f), source_id="s1")
        self._register(mgr, claim, server)
        assert mgr.should_warm("s1") is False

    def test_dehydrated_directory_source_skipped(self, tmp_path, force_nonresident):
        """A zarr store claims the *directory*, so ``member_paths`` is just that
        directory and an ``is_file``-guarded member check cannot see a
        placeholder inside it -- the whole store reads as resident and precache
        recalls it, against #174's policy. The adapter samples the interior.
        """
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})
        store = tmp_path / "img.zarr"
        store.mkdir()
        (store / ".zattrs").write_text("{}")
        (store / "0.0").write_bytes(b"chunk")
        claim = SourceClaim("ome-zarr", str(store), source_id="s1")
        self._register(mgr, claim, server)
        # The member check still says "resident" -- it is looking at a directory.
        assert mgr._reconciler._claim_has_dehydrated_member(claim) is False
        assert mgr.should_warm("s1") is False

    def test_unregistered_adapter_is_not_permission_to_warm(self, tmp_path):
        # A claim without a live adapter: nothing can answer, so the gate stays
        # shut rather than defaulting open.
        mgr = _make_manager(_FakeServer(), cloud_roots={tmp_path.resolve()})
        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        claim = SourceClaim("nifti", str(f), source_id="s1")
        self._register(mgr, claim)
        assert mgr.should_warm("s1") is False


# --------------------------------------------------------------------------- #
# End-to-end through the source manager
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestCloudRegistrationEndToEnd:
    def test_unresolved_claim_is_cataloged_unopened_then_resolve_registers_it(
        self, tmp_path, monkeypatch
    ):
        import zarr

        store = tmp_path / "img.zarr"
        zarr.open_array(
            str(store), mode="w", shape=(32, 48), chunks=(16, 24), dtype="uint16"
        )

        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})
        reconciler = mgr._reconciler

        # An ome-zarr claim flagged unresolved (as the defer branch would).
        claim = SourceClaim("ome-zarr", str(store), source_id="cloud1", unresolved=True)
        assert reconciler._commit_add_claim(claim) is True

        # A catalog row that says needs_recall and a claim: no adapter, nothing
        # opened, nothing queued for the background pool.
        assert server.registered == {}
        assert server._metadata_db.pending == [("cloud1", True)]
        assert server._metadata_db.added == []
        assert reconciler.is_pending("cloud1")
        assert reconciler.unregistered_count() == 1
        assert reconciler.pending_count() == 0  # waits for a client, not the pool
        with pytest.raises(SourceUnresolvedError, match="download"):
            reconciler.check_registered("cloud1")

        # An explicit resolve builds the real adapter and writes the concrete row.
        reconciler.materialize("cloud1")
        adapter = server.registered["cloud1"]
        assert [list(t.shape) for t in adapter.list_tensor_descriptors()] == [[32, 48]]
        assert [sid for sid, _ in server._metadata_db.added] == ["cloud1"]
        assert not reconciler.is_pending("cloud1")
        assert reconciler.unregistered_count() == 0

    def test_a_refresh_sends_a_resolved_cloud_source_back_to_needs_recall(
        self, tmp_path
    ):
        import zarr

        store = tmp_path / "img.zarr"
        zarr.open_array(
            str(store), mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8"
        )
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})
        reconciler = mgr._reconciler
        claim = SourceClaim("ome-zarr", str(store), source_id="cloud1", unresolved=True)
        reconciler._commit_add_claim(claim)
        reconciler.materialize("cloud1")
        assert "cloud1" in server.registered

        # The bytes behind it changed: nothing is reopened, the adapter is
        # dropped, and the row asks for a recall again.
        assert reconciler._refresh_claim_locked(claim) is True

        assert server.registered == {}
        assert server._metadata_db.pending[-1] == ("cloud1", True)
        assert "cloud1" in reconciler._recall
        assert reconciler.is_pending("cloud1")


# --------------------------------------------------------------------------- #
# Precache safety
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestPrecacheSkipsUnresolved:
    def test_unresolved_source_is_never_queued_for_registration(self, tmp_path):
        # The background pool registers what it is told of. A cloud source is
        # registered by downloading it, which only a client's resolve may do.
        store = tmp_path / "img.zarr"
        store.mkdir()
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={tmp_path.resolve()})
        queued = []
        mgr._reconciler.set_pending_hook(queued.append)

        claim = SourceClaim("ome-zarr", str(store), source_id="cloud1", unresolved=True)
        assert mgr._reconciler._commit_add_claim(claim) is True

        assert queued == []

    def test_real_precache_worker_skips_a_source_with_no_adapter(self, monkeypatch):
        # Drives the actual PrecacheWorker._process_source: a cloud source has no
        # adapter until it resolves, so background warming finds nothing to read.
        from biopb_tensor_server.core.config import PrecacheConfig
        from biopb_tensor_server.serving.precache import PrecacheWorker

        class _Registry:
            def get(self, sid):
                return None

        class _Srv:
            sources = _Registry()

        worker = PrecacheWorker(_Srv(), PrecacheConfig())
        # Past the cache gate so the real source-processing logic runs.
        monkeypatch.setattr(worker, "_cache_active", lambda: True)
        assert worker._process_source("s1") is False


# --------------------------------------------------------------------------- #
# Cloud rescan gating (the periodic rescan path)
# --------------------------------------------------------------------------- #


class TestCloudRescanGating:
    """A cloud root is walked only on a ``force_full`` rescan -- with no mtime
    stability gate and no open-probe when it IS walked.

    Drives the real ``_handle_rescan`` pipeline (not ``_register_source_claim``
    directly) over a simulated dehydrated dataset, proving the dehydrated content
    is registered unresolved without ever being opened (no recall), and that an
    incremental (non-force_full) rescan silently skips the cloud subtree while
    preserving its claims.
    """

    def test_rescan_registers_unresolved_without_opening_content(
        self, tmp_path, force_nonresident
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
        mgr = _make_manager(server, cloud_roots={root.resolve()}, monitored={root})

        # The cloud gate bypasses the stability machinery outright: a placeholder's
        # mtime is untrustworthy, so it could never age into eligibility. Set a
        # window no local entry could satisfy -- the cloud source must still register.
        mgr._stability_window = 10**9

        mgr._handle_rescan()

        # Cataloged as needs_recall without being opened: no adapter, no read.
        assert server.registered == {}
        assert len(mgr._reconciler._recall) == 1
        assert [recall for _, recall in server._metadata_db.pending] == [True]
        assert server._metadata_db.added == []

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

    @staticmethod
    def _spy_walks(monkeypatch):
        """Record every root the rescan hands to the walker."""
        from biopb_tensor_server.sources import source_manager as sm

        walked = []
        real = sm.discover_sources

        def spy(root, *args, **kwargs):
            walked.append(str(root))
            return real(root, *args, **kwargs)

        monkeypatch.setattr(sm, "discover_sources", spy)
        return walked

    def test_incremental_rescan_skips_cloud_force_full_rewalks(
        self, tmp_path, force_nonresident, monkeypatch
    ):
        # A cloud subtree is walked only on a force_full pass: the first rescan
        # (last-full == -inf) is force_full and registers it; a subsequent
        # incremental rescan does not walk the cloud root and leaves the claim
        # untouched; a later force_full rescan walks it again.
        root = tmp_path / "cloudroot"
        root.mkdir()
        store = root / "img.zarr"
        store.mkdir()
        (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (store / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": []}]}))

        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={root.resolve()}, monitored={root})
        root_key = str(root.resolve())
        walked = self._spy_walks(monkeypatch)

        # First rescan: force_full -> cloud walked and registered.
        mgr._handle_rescan()
        assert len(mgr._reconciler._recall) == 1
        sid = next(iter(mgr._reconciler._recall))
        assert walked == [root_key]

        # Incremental (non-force_full) rescan: cloud root not walked, claim kept.
        monkeypatch.setattr(mgr, "_should_force_full_rescan", lambda: False)
        mgr._handle_rescan()
        assert walked == [root_key]
        assert set(mgr._reconciler._recall) == {
            sid
        }  # preserved, not torn down/re-added

        # force_full rescan: cloud walked again, claim stable.
        monkeypatch.setattr(mgr, "_should_force_full_rescan", lambda: True)
        mgr._handle_rescan()
        assert walked == [root_key, root_key]
        assert set(mgr._reconciler._recall) == {sid}

    @staticmethod
    def _make_cloud_store(root, name="img.zarr"):
        store = root / name
        store.mkdir(parents=True, exist_ok=True)
        (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (store / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": []}]}))
        return store

    def test_incremental_does_no_cloud_work(
        self, tmp_path, force_nonresident, monkeypatch
    ):
        # After the startup force_full, an incremental rescan neither walks the
        # cloud root nor touches its source: it stays registered, and the
        # reconcile scopes it out by id rather than diffing it.
        root = tmp_path / "cloudroot"
        root.mkdir()
        self._make_cloud_store(root)
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={root.resolve()}, monitored={root})

        mgr._handle_rescan()  # force_full
        sid = next(iter(mgr._reconciler._recall))
        assert sid in mgr._reconciler._cloud_source_ids

        walked = self._spy_walks(monkeypatch)
        monkeypatch.setattr(mgr, "_should_force_full_rescan", lambda: False)

        mgr._handle_rescan()

        assert walked == []  # the cloud root was not enumerated
        assert set(mgr._reconciler._recall) == {sid}  # preserved

    def test_incremental_preserves_cloud_without_diff_or_churn(
        self, tmp_path, force_nonresident, monkeypatch
    ):
        # On an incremental, a cloud source is never signature-diffed (so no live
        # member stat) and never re-added (no metadata-DB churn) -- it is simply
        # left untouched by the reconcile scoping.
        root = tmp_path / "cloudroot"
        root.mkdir()
        self._make_cloud_store(root)
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={root.resolve()}, monitored={root})

        mgr._handle_rescan()  # force_full registers + partitions
        sid = next(iter(mgr._reconciler._recall))
        added_before = len(server._metadata_db.pending)

        orig_sig = mgr._reconciler._build_claim_signatures

        def guard(claim):
            assert claim.source_id not in mgr._reconciler._cloud_source_ids, (
                "cloud source must not be signature-diffed on an incremental"
            )
            return orig_sig(claim)

        monkeypatch.setattr(mgr._reconciler, "_build_claim_signatures", guard)
        monkeypatch.setattr(mgr, "_should_force_full_rescan", lambda: False)

        mgr._handle_rescan()

        assert set(mgr._reconciler._recall) == {sid}
        assert len(server._metadata_db.pending) == added_before  # no re-add churn

    def test_new_cloud_dataset_surfaces_only_on_force_full(
        self, tmp_path, force_nonresident, monkeypatch
    ):
        # A dataset added under the cloud root after startup is invisible to the
        # frequent incrementals (they skip the cloud subtree) and surfaces only on
        # the next force_full re-walk, which also rebuilds the partition.
        root = tmp_path / "cloudroot"
        root.mkdir()
        self._make_cloud_store(root, "img.zarr")
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={root.resolve()}, monitored={root})

        mgr._handle_rescan()  # force_full -> 1 source
        assert len(mgr._reconciler._recall) == 1

        self._make_cloud_store(root, "img2.zarr")  # new dataset under the cloud root

        monkeypatch.setattr(mgr, "_should_force_full_rescan", lambda: False)
        mgr._handle_rescan()  # incremental: cloud skipped -> not discovered yet
        assert len(mgr._reconciler._recall) == 1

        monkeypatch.setattr(mgr, "_should_force_full_rescan", lambda: True)
        mgr._handle_rescan()  # force_full: re-walk surfaces it, partition rebuilt
        assert len(mgr._reconciler._recall) == 2
        assert all(
            sid in mgr._reconciler._cloud_source_ids for sid in mgr._reconciler._recall
        )


# --------------------------------------------------------------------------- #
# Server `resolve` action (streaming do_action)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestResolveAction:
    """The dedicated streaming `resolve` do_action: the SOLE resolution entry
    point. Emits ``ResolveStreamMessage`` progress heartbeats while the recall
    runs, then one terminal message carrying the source's now-concrete catalog
    row in its ``source_row`` arm."""

    def _server(self, zpath, source_id="cloud1"):
        """A catalogued server whose reconciler holds an unresolved cloud claim,
        which is what a SourceManager leaves behind for it: a ``needs_recall`` row
        and a claim, with no adapter."""
        from pathlib import Path

        from biopb_tensor_server.adapters import get_default_registry

        server = catalog_server("localhost:0")
        mgr = make_manager(
            server=server,
            metadata_db=server.metadata_db,
            registry=get_default_registry(),
            discovery_state=DiscoveryState(),
            cloud_roots={Path(zpath).parent},
        )
        server.set_resolve_handler(mgr.resolve_source)
        claim = SourceClaim("ome-zarr", zpath, source_id=source_id, unresolved=True)
        assert mgr._reconciler._commit_add_claim(claim) is True
        return server, mgr

    @staticmethod
    def _zarr(d):
        import zarr

        zpath = os.path.join(d, "img.zarr")
        zarr.open_array(zpath, mode="w", shape=(16, 24), chunks=(8, 12), dtype="uint8")
        return zpath

    @staticmethod
    def _parse(bodies):
        from biopb.tensor.descriptor_pb2 import ResolveStreamMessage

        msgs = [ResolveStreamMessage.FromString(b) for b in bodies]
        return msgs, [m.WhichOneof("payload") for m in msgs]

    @staticmethod
    def _rows(msg):
        import pyarrow as pa

        return pa.ipc.open_stream(msg.source_row).read_all().to_pylist()

    def test_resolve_action_streams_the_backfilled_catalog_row(self):
        import pyarrow.flight as flight
        from biopb.tensor._catalog_rows import SOURCE_ROW_COLUMNS

        with tempfile.TemporaryDirectory() as d:
            server, _ = self._server(self._zarr(d))
            db = server.metadata_db
            (placeholder,) = db.query(
                "SELECT is_resolved, unresolved_reason, tensors FROM sources"
            ).to_pylist()
            assert placeholder == {
                "is_resolved": False,
                "unresolved_reason": "needs_recall",
                "tensors": [],
            }
            assert server.sources.get("cloud1") is None

            action = flight.Action("resolve", b"cloud1")
            # do_action yields raw bytes (the Flight framework wraps each in a Result).
            bodies = [bytes(r) for r in server.do_action(None, action)]
            msgs, kinds = self._parse(bodies)
            terminal = [
                m for m, k in zip(msgs, kinds, strict=True) if k == "source_row"
            ]
            assert len(terminal) == 1  # exactly one terminal row
            assert kinds[-1] == "source_row"
            (row,) = self._rows(terminal[0])
            assert row["source_id"] == "cloud1"
            assert [t["shape"] for t in row["tensors"]] == [[16, 24]]
            assert row["is_resolved"] is True
            assert "data_resident" not in row  # never in a row
            assert server.sources.get("cloud1") is not None

            # It IS the catalog row, not a second encoding built beside it.
            browsed = db.query(
                f"SELECT {SOURCE_ROW_COLUMNS} FROM sources WHERE source_id = 'cloud1'"
            ).to_pylist()
            assert browsed == [row]

    def test_resolve_action_emits_heartbeats_during_long_recall(self, monkeypatch):
        # The stream must carry progress keep-alives BEFORE the terminal row --
        # this is what keeps a minutes-long recall under a proxy's idle read
        # timeout, and the elapsed field lets a client show progress.
        #
        # Deterministic drive (no wall-clock race, see issue #177): the recall
        # blocks on an Event we only set AFTER pulling the first stream item.
        # Because the worker thread is parked in it until then, the server's
        # heartbeat loop is guaranteed to time out its join at least once and emit
        # a progress message first -- independent of runner load.
        import pyarrow.flight as flight
        from biopb_tensor_server.serving import server as server_mod

        monkeypatch.setattr(server_mod, "_RESOLVE_HEARTBEAT_SECONDS", 0.01)

        release = threading.Event()

        with tempfile.TemporaryDirectory() as d:
            server, mgr = self._server(self._zarr(d), "slow")
            real = mgr._reconciler._build_recalled_adapter

            def slow_recall(claim, source_config):
                # Park until the test has observed the first heartbeat, then
                # finish. Bounded wait so a broken test can never hang the drain.
                release.wait(timeout=5.0)
                return real(claim, source_config)

            monkeypatch.setattr(mgr._reconciler, "_build_recalled_adapter", slow_recall)
            action = flight.Action("resolve", b"slow")

            stream = server.do_action(None, action)
            try:
                # Worker is parked in the recall -> the first item is a heartbeat.
                first = bytes(next(stream))
            finally:
                release.set()  # let the recall complete, then drain the rest
            bodies = [first] + [bytes(r) for r in stream]
            msgs, kinds = self._parse(bodies)

            assert kinds[0] == "progress"  # deterministically, a heartbeat leads
            assert kinds.count("progress") >= 1  # at least one heartbeat
            assert kinds[-1] == "source_row"  # terminal is the catalog row
            assert self._rows(msgs[-1])[0]["source_id"] == "slow"
            # the heartbeat names what is being recalled
            assert msgs[0].progress.target_name == "img.zarr"
            # progress heartbeats carry a monotonically non-decreasing elapsed clock
            elapsed = [
                m.progress.elapsed_seconds
                for m, k in zip(msgs, kinds, strict=True)
                if k == "progress"
            ]
            assert elapsed == sorted(elapsed)
            assert elapsed[-1] >= 0.0

    @pytest.mark.parametrize(
        "exc_type, flight_error",
        [
            ("retriable", "FlightUnavailableError"),
            ("permanent", "FlightInternalError"),
        ],
    )
    def test_a_resolve_that_does_not_hydrate_raises_and_syncs_nothing(
        self, exc_type, flight_error, monkeypatch
    ):
        """A resolve that did not hydrate registers nothing: the row stays
        ``needs_recall`` as the scan wrote it, and the source is not marked failed
        (a retry may succeed once the bytes arrive)."""
        import pyarrow.flight as flight
        from biopb_tensor_server.core.errors import (
            SourceResolveRetriableError,
            SourceUnresolvedError,
        )

        err = (
            SourceResolveRetriableError("recall failed")
            if exc_type == "retriable"
            else SourceUnresolvedError("unsupported type")
        )

        def wont_hydrate(claim, source_config):
            raise err

        with tempfile.TemporaryDirectory() as d:
            server, mgr = self._server(self._zarr(d))
            monkeypatch.setattr(
                mgr._reconciler, "_build_recalled_adapter", wont_hydrate
            )

            action = flight.Action("resolve", b"cloud1")
            with pytest.raises(getattr(flight, flight_error)):
                list(server.do_action(None, action))

            # Nothing registered, and the placeholder row is untouched.
            assert server.sources.get("cloud1") is None
            (row,) = server.metadata_db.query(
                "SELECT tensors, unresolved_reason FROM sources "
                "WHERE source_id = 'cloud1'"
            ).to_pylist()
            assert row == {"tensors": [], "unresolved_reason": "needs_recall"}
            assert mgr._reconciler.is_pending("cloud1")
            assert "cloud1" not in mgr._reconciler._pending_failed

    def test_resolve_action_without_a_handler_is_not_enabled(self):
        import pyarrow.flight as flight

        server = catalog_server("localhost:0")  # nothing injected a resolver
        action = flight.Action("resolve", b"cloud1")
        with pytest.raises(flight.FlightServerError, match="not enabled"):
            list(server.do_action(None, action))

    def test_resolve_action_unknown_source_errors(self):
        import pyarrow.flight as flight

        # A server that resolves, asked for an id it has never heard of.
        server = catalog_server("localhost:0")
        server.set_resolve_handler(lambda source_id, on_target: False)
        action = flight.Action("resolve", b"missing")
        with pytest.raises(flight.FlightServerError, match="Source not found"):
            list(server.do_action(None, action))


# --------------------------------------------------------------------------- #
# Server `warm` action (streaming do_action): hydrate-ahead a resolved source
# --------------------------------------------------------------------------- #


class _DirAdapter:
    """Minimal registered adapter exposing only ``_source_url`` -- all `warm`
    needs (it walks that directory and reads files; format-agnostic)."""

    capability_token = None

    def __init__(self, url):
        self._source_url = url

    @property
    def source_url(self):
        return self._source_url


class _Ctx:
    """Fake ServerCallContext: counts ``is_cancelled`` polls; can flip True after
    a fixed number so a test can exercise the cooperative-cancel break."""

    def __init__(self, cancel_after=None):
        self.calls = 0
        self._cancel_after = cancel_after

    def is_cancelled(self):
        self.calls += 1
        if self._cancel_after is None:
            return False
        return self.calls > self._cancel_after

    def get_middleware(self, name):
        return None  # no bearer presented; the servers here have no token


class TestWarmAction:
    """The dedicated streaming `warm` do_action: server-side hydrate-ahead. It
    walks the resolved source directory and reads every file (forcing recall),
    emitting ``WarmStreamMessage`` progress, then one terminal ``done``."""

    def _server(self, source_id, adapter):
        # Catalog-less and registry-only: warm walks the source directory, not
        # the catalog, and _DirAdapter is a format-agnostic stub with no
        # metadata to catalogue anyway. The residency test below wires a real
        # catalog because that one IS about the row.
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer("localhost:0")
        server.register_source(source_id, adapter)
        return server

    @staticmethod
    def _parse(bodies):
        from biopb.tensor.descriptor_pb2 import WarmStreamMessage

        msgs = [WarmStreamMessage.FromString(b) for b in bodies]
        return msgs, [m.WhichOneof("payload") for m in msgs]

    @staticmethod
    def _no_throttle(monkeypatch):
        # Force a progress message at every file/block boundary (the default
        # 0.5s throttle would suppress them for tiny fast files).
        from biopb_tensor_server.serving import server as server_mod

        monkeypatch.setattr(server_mod, "_WARM_PROGRESS_MIN_INTERVAL", -1.0)

    def _make_files(self, root, sizes):
        paths = []
        for name, size in sizes.items():
            p = os.path.join(root, name)
            with open(p, "wb") as fh:
                fh.write(b"\xa5" * size)
            paths.append(p)
        return paths

    def test_warm_writes_nothing_to_the_catalog(self, tmp_path):
        """Warm touches no catalog at all: residency is not stored, so there is
        nothing for it to correct, and the `directory_is_resident()` walk that
        used to run here was pure cost (biopb/biopb#1035)."""
        import pyarrow.flight as flight

        class _Forbidden:
            annotations_persisted = False
            store_path = None

            def __getattr__(self, name):
                raise AssertionError(f"warm touched the catalog: {name}")

        root = str(tmp_path / "src")
        os.makedirs(root)
        self._make_files(root, {"a.bin": 8})

        server = self._server("s1", _DirAdapter(root))
        server._metadata_db = _Forbidden()

        bodies = [
            bytes(r) for r in server.do_action(_Ctx(), flight.Action("warm", b"s1"))
        ]
        _msgs, kinds = self._parse(bodies)
        assert kinds[-1] == "done"

    def test_the_catalog_cannot_refresh_residency_at_all(self):
        """It wrote `data_resident`, plus an `is_resolved` already true on the
        one path that called it."""
        from biopb_tensor_server.serving.metadata_db import MetadataDatabase

        assert not hasattr(MetadataDatabase, "refresh_residency")

    def test_warm_streams_progress_and_terminal_done(self, tmp_path, monkeypatch):
        import pyarrow.flight as flight

        self._no_throttle(monkeypatch)
        root = str(tmp_path / "src")
        os.makedirs(root)
        sizes = {"a.bin": 50, "b.bin": 10, "c.bin": 30}
        self._make_files(root, sizes)
        total = sum(sizes.values())

        server = self._server("s1", _DirAdapter(root))
        action = flight.Action("warm", b"s1")
        bodies = [bytes(r) for r in server.do_action(_Ctx(), action)]
        msgs, kinds = self._parse(bodies)

        assert kinds[-1] == "done"  # exactly one terminal, last
        assert kinds.count("done") == 1
        done = msgs[-1].done
        assert done.files_total == 3
        assert done.files_done == 3  # every file read
        assert done.bytes_total == total
        assert done.bytes_done == total  # every byte recalled
        # progress arms report a monotonically non-decreasing files_done
        prog_files = [
            m.progress.files_done
            for m, k in zip(msgs, kinds, strict=True)
            if k == "progress"
        ]
        assert prog_files == sorted(prog_files)
        # guard cleaned up
        assert server.activity.warming == set()

    def test_warm_orders_files_ascending_by_size(self, tmp_path, monkeypatch):
        import pyarrow.flight as flight
        from biopb_tensor_server.serving import server as server_mod

        self._no_throttle(monkeypatch)
        root = str(tmp_path / "src")
        os.makedirs(root)
        self._make_files(root, {"big": 90, "small": 10, "mid": 40})

        submitted = []
        original_submit = server_mod.ThreadPoolExecutor.submit

        def record_submit(executor, fn, *args, **kwargs):
            submitted.append(os.path.basename(args[0]))
            return original_submit(executor, fn, *args, **kwargs)

        monkeypatch.setattr(server_mod.ThreadPoolExecutor, "submit", record_submit)
        server = self._server("s2", _DirAdapter(root))
        list(server.do_action(_Ctx(), flight.Action("warm", b"s2")))

        # Completion order is intentionally concurrent and may vary. The
        # scheduler must still feed the pool in the coarsest-first order.
        assert submitted == ["small", "mid", "big"]

    def test_warm_reads_files_concurrently_with_bounded_workers(
        self, tmp_path, monkeypatch
    ):
        import pyarrow.flight as flight
        from biopb_tensor_server.serving import server as server_mod

        self._no_throttle(monkeypatch)
        root = str(tmp_path / "src")
        os.makedirs(root)
        worker_count = server_mod._WARM_MAX_WORKERS
        sizes = {f"f{i}.bin": 8 for i in range(worker_count + 2)}
        self._make_files(root, sizes)

        active = [0]
        max_active = [0]
        active_lock = threading.Lock()
        real_open = open

        class TrackedFile:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __enter__(self):
                self._wrapped.__enter__()
                return self

            def __exit__(self, *exc_info):
                try:
                    return self._wrapped.__exit__(*exc_info)
                finally:
                    with active_lock:
                        active[0] -= 1

            def readinto(self, buf):
                # Hold each read briefly so all initially scheduled workers have
                # an opportunity to overlap before the first one completes.
                time.sleep(0.05)
                return self._wrapped.readinto(buf)

        def tracked_open(*args, **kwargs):
            wrapped = real_open(*args, **kwargs)
            with active_lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            return TrackedFile(wrapped)

        monkeypatch.setattr(server_mod, "open", tracked_open, raising=False)
        server = self._server("s-concurrent", _DirAdapter(root))
        bodies = [
            bytes(r)
            for r in server.do_action(_Ctx(), flight.Action("warm", b"s-concurrent"))
        ]
        msgs, _kinds = self._parse(bodies)

        assert max_active[0] > 1
        assert max_active[0] <= server_mod._WARM_MAX_WORKERS
        assert msgs[-1].done.files_done == len(sizes)
        assert msgs[-1].done.bytes_done == sum(sizes.values())

    def test_warm_walk_is_recursive(self, tmp_path, monkeypatch):
        import pyarrow.flight as flight

        self._no_throttle(monkeypatch)
        root = str(tmp_path / "store")
        nested = os.path.join(root, "0", "0")
        os.makedirs(nested)
        self._make_files(root, {".zattrs": 20})
        self._make_files(nested, {"chunk": 64})  # interior chunk file

        server = self._server("s3", _DirAdapter(root))
        bodies = [
            bytes(r) for r in server.do_action(_Ctx(), flight.Action("warm", b"s3"))
        ]
        msgs, kinds = self._parse(bodies)
        done = msgs[-1].done
        assert done.files_total == 2  # nested file counted
        assert done.bytes_total == 84
        assert done.bytes_done == 84

    def test_warm_single_file_source_is_noop(self, tmp_path):
        import pyarrow.flight as flight

        f = tmp_path / "one.tif"
        f.write_bytes(b"x" * 100)
        server = self._server("s4", _DirAdapter(str(f)))  # _source_url is a FILE
        bodies = [
            bytes(r) for r in server.do_action(_Ctx(), flight.Action("warm", b"s4"))
        ]
        msgs, kinds = self._parse(bodies)
        assert kinds == ["done"]
        assert msgs[-1].done.files_total == 0  # nothing to warm

    def test_warm_refuses_a_remote_source(self):
        """A remote url has no local tree to recall into, so warm fails loudly
        rather than answering `files_total == 0` -- which a local single-file
        source already uses to mean "nothing left to warm" (biopb/biopb#1035).
        """
        import pyarrow.flight as flight

        server = self._server("s9", _DirAdapter("grpc://lab/img.zarr"))
        with pytest.raises(flight.FlightServerError, match="remote"):
            list(server.do_action(_Ctx(), flight.Action("warm", b"s9")))

    def test_warm_names_the_scheme_and_where_to_go(self):
        # Actionable: which source, why not here, and where to warm it.
        import pyarrow.flight as flight

        server = self._server("s10", _DirAdapter("s3://bucket/img.zarr"))
        with pytest.raises(flight.FlightServerError) as caught:
            list(server.do_action(_Ctx(), flight.Action("warm", b"s10")))
        message = str(caught.value)
        assert "s10" in message
        assert "s3" in message
        assert "server that holds the data" in message

    def test_warm_still_no_ops_on_a_local_single_file_source(self, tmp_path):
        # The refusal must not swallow this: `files_total == 0` is how a client
        # (the web SPA) learns a source is single-file.
        import pyarrow.flight as flight

        f = tmp_path / "scan.nii"
        f.write_bytes(b"payload")
        server = self._server("s11", _DirAdapter(str(f)))
        bodies = [
            bytes(r) for r in server.do_action(_Ctx(), flight.Action("warm", b"s11"))
        ]
        msgs, kinds = self._parse(bodies)
        assert kinds == ["done"]
        assert msgs[-1].done.files_total == 0

    def test_warm_cancel_stops_early_with_partial_done(self, tmp_path, monkeypatch):
        import pyarrow.flight as flight

        self._no_throttle(monkeypatch)
        root = str(tmp_path / "src")
        os.makedirs(root)
        self._make_files(root, {f"f{i}.bin": 8 for i in range(12)})

        server = self._server("s5", _DirAdapter(root))
        # Flip cancelled True after a couple of polls -> break mid-recall.
        ctx = _Ctx(cancel_after=2)
        bodies = [bytes(r) for r in server.do_action(ctx, flight.Action("warm", b"s5"))]
        msgs, kinds = self._parse(bodies)
        assert kinds[-1] == "done"  # still emits a terminal snapshot
        done = msgs[-1].done
        assert done.files_done < done.files_total  # did not finish all 12
        assert server.activity.warming == set()  # guard released even on cancel

    def test_warm_rejects_concurrent_warm_of_same_source(self, tmp_path):
        import pyarrow.flight as flight

        root = str(tmp_path / "src")
        os.makedirs(root)
        self._make_files(root, {"a.bin": 8})
        server = self._server("s6", _DirAdapter(root))
        server.activity.begin_warm("s6")  # simulate an in-flight warm
        with pytest.raises(flight.FlightServerError, match="already in progress"):
            list(server.do_action(_Ctx(), flight.Action("warm", b"s6")))

    def test_warm_unknown_source_errors(self):
        import pyarrow.flight as flight

        server = self._server("s7", _DirAdapter("/nonexistent"))
        with pytest.raises(flight.FlightServerError, match="Source not found"):
            list(server.do_action(_Ctx(), flight.Action("warm", b"missing")))


# --------------------------------------------------------------------------- #
# Change A: cloud-root flag plumbing (ClaimContext -> claim() -> resolve)
# --------------------------------------------------------------------------- #


class TestCloudRootFlag:
    def test_claim_context_carries_cloud_root(self, tmp_path):
        assert ClaimContext(tmp_path, cloud_root=True).cloud_root is True
        assert ClaimContext(tmp_path).cloud_root is False

    def test_discover_sources_sets_cloud_root_on_every_probe(self, tmp_path):
        # The walk's ``cloud_root`` must reach each adapter's claim() through its
        # ClaimContext -- for the root and for every entry found under it.
        from biopb_tensor_server.core.discovery import AdapterRegistry, discover_sources

        seen = {}

        class _Recorder:
            @classmethod
            def claim(cls, ctx, state):
                seen[ctx.path_str] = ctx.cloud_root
                return None

        registry = AdapterRegistry()
        registry.register(_Recorder, "recorder")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.bin").write_text("x")

        discover_sources(tmp_path, registry, cloud_root=True, admit_nonresident=True)

        assert seen and all(seen.values())
        discover_sources(tmp_path, registry)
        assert not any(seen.values())


# --------------------------------------------------------------------------- #
# Drag-drop of a cloud folder (biopb/biopb#310)
# --------------------------------------------------------------------------- #


def _drop(mgr, path, **kwargs):
    mgr.complete_initial_scan()  # drops wait for the first scan
    result = None
    for event in mgr.add_local_source(str(path), **kwargs):
        if event[0] == "result":
            result = event[1]
    return result


class TestDropCloudFolder:
    """A drop is a root like any other: ``cloud`` is a property of it."""

    @staticmethod
    def _folder(tmp_path):
        folder = tmp_path / "synced"
        folder.mkdir()
        (folder / "scan.nii").write_bytes(b"payload")
        return folder

    def test_without_cloud_placeholders_are_skipped_and_counted(
        self, tmp_path, force_nonresident
    ):
        folder = self._folder(tmp_path)
        server = _FakeServer()
        mgr = _make_manager(server)

        result = _drop(mgr, folder)

        assert result.added == []
        assert result.skipped_offline == 1
        assert server.registered == {}

    def test_consented_drop_catalogs_placeholders_as_needs_recall(
        self, tmp_path, force_nonresident
    ):
        folder = self._folder(tmp_path)
        server = _FakeServer()
        mgr = _make_manager(server)

        result = _drop(mgr, folder, cloud=True)

        assert len(result.added) == 1
        assert result.skipped_offline == 0
        (sid,) = result.added
        assert sid not in server.registered  # a row and a claim, no adapter
        assert server._metadata_db.pending == [(sid, True)]

    def test_the_root_stays_cloud_for_the_later_checks(
        self, tmp_path, force_nonresident
    ):
        folder = self._folder(tmp_path)
        mgr = _make_manager(_FakeServer())

        _drop(mgr, folder, cloud=True)

        assert mgr._roots.is_cloud(str(folder / "scan.nii"))
        assert mgr._reconciler._roots.is_cloud(str(folder / "scan.nii"))

    def test_deregistering_the_drop_takes_the_cloud_root_with_it(
        self, tmp_path, force_nonresident
    ):
        folder = self._folder(tmp_path)
        mgr = _make_manager(_FakeServer())
        _drop(mgr, folder, cloud=True)
        assert mgr._roots.is_cloud(str(folder / "scan.nii"))

        mgr.remove_dropped_root("dnd://synced")

        assert not mgr._roots.is_cloud(str(folder / "scan.nii"))
        assert not mgr._reconciler._roots.is_cloud(str(folder / "scan.nii"))

    def test_a_configured_cloud_root_survives_a_drop_being_deregistered(
        self, tmp_path, force_nonresident
    ):
        root = tmp_path / "configured"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / "scan.nii").write_bytes(b"payload")
        mgr = _make_manager(_FakeServer(), cloud_roots={root.resolve()})
        _drop(mgr, sub)

        mgr.remove_dropped_root("dnd://sub")

        assert mgr._roots.is_cloud(str(sub / "scan.nii"))

    def test_a_drop_under_a_configured_cloud_root_needs_no_flag(
        self, tmp_path, force_nonresident
    ):
        root = tmp_path / "configured"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / "scan.nii").write_bytes(b"payload")
        server = _FakeServer()
        mgr = _make_manager(server, cloud_roots={root.resolve()})

        result = _drop(mgr, sub)  # no cloud flag

        assert len(result.added) == 1
        assert server._metadata_db.pending == [(result.added[0], True)]

    def test_a_resident_folder_reports_nothing_skipped_and_is_not_cloud(self, tmp_path):
        folder = self._folder(tmp_path)
        mgr = _make_manager(_FakeServer())

        result = _drop(mgr, folder)

        # (The payload is not a real NIfTI, so registration may fail; what is
        # pinned is that the walk skipped nothing and the root was not made cloud.)
        assert result.skipped_offline == 0
        assert not mgr._roots.is_cloud(str(folder))


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
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        f = tmp_path / "img.tif"
        f.write_bytes(b"II*\x00real-bytes-but-cloud")
        assert (
            OmeTiffAdapter.claim(_RaisingReadCtx(f, cloud_root=True), DiscoveryState())
            is None
        )

    def test_companion_ome_skipped_under_cloud(self, tmp_path):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        f = tmp_path / "set.companion.ome"
        f.write_text("<OME/>")
        assert (
            OmeTiffAdapter.claim(_RaisingReadCtx(f, cloud_root=True), DiscoveryState())
            is None
        )

    def test_cloud_tif_dir_yields_single_file_sources(self, tmp_path):
        # The whole-registry behavior: a dir of .tif files under cloud produces
        # one single-file claim per .tif, never a grouped set. The native
        # tifffile adapter owns that claim (it reads nothing to make it); a
        # dehydrated placeholder is deferred by the source manager, not here.
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
        assert all(c.source_type == "tiff" for c in claims)
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
        claim = DicomAdapter.claim(
            ClaimContext(files[0], cloud_root=True), DiscoveryState()
        )
        assert claim is not None
        assert claim.source_type == "dicom"
        assert claim.unresolved is True

    def test_static_discover_path_applies_multifile_ban(
        self, tmp_path, force_nonresident, monkeypatch
    ):
        # End-to-end through the *static* one-shot discover path (a monitor=false
        # cloud directory): it must hand cloud_root=True to the claim probes so the
        # multi-file ban fires there too, not only on the monitored rescan. Spy on
        # the series adapter to capture the cloud_root it is handed.
        from biopb_tensor_server.adapters.dicom import DicomSeriesAdapter
        from biopb_tensor_server.sources.resolve import discover_sources

        d = tmp_path / "series"
        d.mkdir()
        for i in range(3):
            (d / f"{i}.dcm").write_bytes(b"not a real dicom")

        seen_cloud_root = []
        real = DicomSeriesAdapter.claim.__func__

        def spy(cls, ctx, state):
            seen_cloud_root.append(ctx.cloud_root)
            return real(cls, ctx, state)

        monkeypatch.setattr(DicomSeriesAdapter, "claim", classmethod(spy))

        results = discover_sources(SourceConfig(url=str(d), cloud=True, monitor=False))

        # The static path handed cloud_root=True to the series adapter (the plumbing)
        assert seen_cloud_root and all(seen_cloud_root)
        # ...so it bowed out and each slice became its own deferred single-file
        # source, with cloud propagated to the expanded configs.
        assert len(results) == 3
        assert {r.type for r in results} == {"dicom"}
        assert all(r.cloud for r in results)


# --------------------------------------------------------------------------- #
# Change C: dir-claiming records the directory as the only member
# --------------------------------------------------------------------------- #


class TestDirClaimingMembership:
    def test_tiff_sequence_member_is_dir_only(self, tmp_path):
        import numpy as np
        import tifffile
        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        # Plain numbered sequence (img_*/OME/MicroManager names are excluded by
        # _group_tiff_sequence on purpose). 30 frames clears the claim floor.
        for i in range(30):
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
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        st = f.stat()
        cloud_sig = build_entry_signature(st, is_directory=False, cloud=True)
        plain_sig = build_entry_signature(st, is_directory=False, cloud=False)
        assert cloud_sig == (st.st_dev, st.st_ino)
        assert plain_sig != cloud_sig  # plain carries size/mtime/ctime

    def test_hydration_does_not_change_cloud_signature(self, tmp_path):
        # Simulate a recall: size + mtime/ctime change. Under cloud the signature
        # is keyed on identity (dev/ino) only, so it is unchanged -> the rescan
        # will not put the just-resolved source in changed_ids.
        import os as _os

        f = tmp_path / "x.bin"
        f.write_bytes(b"placeholder-stub")
        before = build_entry_signature(f.stat(), is_directory=False, cloud=True)
        # "Hydrate": grow the file and bump times.
        f.write_bytes(b"hydrated-full-content-now-much-larger")
        _os.utime(f, None)
        after = build_entry_signature(f.stat(), is_directory=False, cloud=True)
        assert before == after
        # A non-cloud signature WOULD change on the same hydration.
        assert (
            build_entry_signature(f.stat(), is_directory=False, cloud=False) != before
        )


# --------------------------------------------------------------------------- #
# Change E: resolve error surfacing (retriable vs permanent)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestResolveErrorSurfacing:
    def _recall(self, path, source_type="ome-zarr"):
        """A manager holding one unresolved cloud claim, and its reconciler."""
        server = _FakeServer()
        mgr = _make_manager(server)
        reconciler = mgr._reconciler
        claim = SourceClaim(source_type, path, source_id="s1", unresolved=True)
        assert reconciler._commit_add_claim(claim) is True
        return server, reconciler

    def test_reclaim_oserror_surfaces_retriable(self, monkeypatch):
        # An OSError while re-claiming (recall/IO) must surface as retriable,
        # NOT silently degrade to the claim-time guessed type.
        from biopb_tensor_server.core.errors import SourceResolveRetriableError

        server, reconciler = self._recall("/data/cloud/x.zarr")

        def _boom(ctx, state):
            raise OSError("recall failed")

        monkeypatch.setattr(reconciler._registry, "get_claims_for_path", _boom)
        with pytest.raises(SourceResolveRetriableError):
            reconciler.materialize("s1")
        assert server.registered == {}
        assert "s1" in reconciler._recall  # still waiting for a retry

    def test_create_from_config_oserror_is_retriable(self, monkeypatch, tmp_path):
        import zarr
        from biopb_tensor_server.core.errors import SourceResolveRetriableError

        zpath = str(tmp_path / "img.zarr")
        zarr.open_array(zpath, mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8")
        server, reconciler = self._recall(zpath)

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
            reconciler.materialize("s1")
        assert "s1" in reconciler._recall

    def test_permanent_failure_is_plain_unresolved(self):
        # A nonexistent path: re-claim yields nothing and create_from_config
        # fails permanently (no retriable cause) -> plain SourceUnresolvedError,
        # and NOT the retriable subclass.
        from biopb_tensor_server.core.errors import (
            SourceResolveRetriableError,
            SourceUnresolvedError,
        )

        server, reconciler = self._recall("/nonexistent/path.zarr", "zarr")
        with pytest.raises(SourceUnresolvedError) as exc_info:
            reconciler.materialize("s1")
        assert not isinstance(exc_info.value, SourceResolveRetriableError)
        assert "s1" not in reconciler._pending_failed


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
        zarr.open_array(
            str(store), mode="w", shape=(8, 8), chunks=(4, 4), dtype="uint8"
        )

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
