"""The DICOM claims' memoized header probe."""

import biopb_tensor_server.adapters.dicom as dicom_mod
import pydicom
import pytest
from biopb_tensor_server.adapters.dicom import DicomAdapter, DicomSeriesAdapter
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState
from pydicom.data import get_testdata_file
from pydicom.uid import generate_uid


@pytest.fixture(autouse=True)
def _clear_memo():
    dicom_mod._HEADER_MEMO.clear()
    yield
    dicom_mod._HEADER_MEMO.clear()


@pytest.fixture
def reads(monkeypatch):
    """Count header parses."""
    calls = []
    real = pydicom.dcmread

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", counting)
    return calls


def _slices(directory, series_sizes):
    """One synthetic slice per file, grouped into series of the given sizes."""
    base = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    n = 0
    for size in series_sizes:
        uid = generate_uid()
        for _ in range(size):
            ds = base.copy()
            ds.SeriesInstanceUID = uid
            ds.SOPInstanceUID = generate_uid()
            ds.save_as(directory / f"s{n:03d}.dcm")
            n += 1


def _claim(directory, monitored=True):
    state = DiscoveryState()
    ctx = ClaimContext(directory, monitored=monitored)
    return DicomSeriesAdapter.claim(ctx, state), state


def test_series_claim_takes_the_largest_series(tmp_path):
    _slices(tmp_path, [3, 1, 2])
    claim, state = _claim(tmp_path)
    assert claim.source_type == "dicom-series"
    assert claim.extra_config["num_slices"] == 3
    assert state.is_path_claimed(str(tmp_path / "s000.dcm"))
    assert not state.is_path_claimed(str(tmp_path / "s003.dcm"))


def test_a_rescan_reads_no_headers(tmp_path, reads):
    _slices(tmp_path, [4])
    del reads[:]
    first, _ = _claim(tmp_path)
    assert len(reads) == 4
    again, _ = _claim(tmp_path)
    assert len(reads) == 4
    assert again.extra_config == first.extra_config


def test_a_changed_slice_is_read_again(tmp_path, reads):
    _slices(tmp_path, [3])
    _claim(tmp_path)
    path = tmp_path / "s001.dcm"
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = generate_uid()  # leaves a 2-slice series
    ds.save_as(path)
    del reads[:]
    claim, _ = _claim(tmp_path)
    assert len(reads) == 1  # only the changed file
    assert claim.extra_config["num_slices"] == 2


def test_a_file_that_is_not_dicom_is_remembered_as_such(tmp_path, reads):
    _slices(tmp_path, [2])
    (tmp_path / "junk.dcm").write_bytes(b"not dicom at all")
    claim, _ = _claim(tmp_path)
    assert claim.extra_config["num_slices"] == 2
    del reads[:]
    _claim(tmp_path)
    assert reads == []  # the junk file's verdict is memoized too


def test_an_unreadable_slice_is_not_memoized(tmp_path, monkeypatch):
    _slices(tmp_path, [3])
    real = pydicom.dcmread
    state = {"fail": True}

    def flaky(path, *args, **kwargs):
        if state["fail"] and str(path).endswith("s001.dcm"):
            raise OSError("transient")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", flaky)
    claim, _ = _claim(tmp_path)
    assert claim.extra_config["num_slices"] == 2  # the slice that could not be read
    state["fail"] = False
    claim, _ = _claim(tmp_path)
    assert claim.extra_config["num_slices"] == 3


def test_single_file_claim_shares_the_memo(tmp_path, reads):
    _slices(tmp_path, [1])
    del reads[:]
    p = tmp_path / "s000.dcm"
    claim = DicomAdapter.claim(ClaimContext(p, monitored=True), DiscoveryState())
    assert claim.source_type == "dicom"
    assert len(reads) == 1
    DicomAdapter.claim(ClaimContext(p, monitored=True), DiscoveryState())
    assert len(reads) == 1


def test_only_a_monitored_root_is_memoized(tmp_path, reads):
    _slices(tmp_path, [3])
    del reads[:]
    _claim(tmp_path, monitored=False)
    _claim(tmp_path, monitored=False)
    assert len(reads) == 6  # read every time
    assert len(dicom_mod._HEADER_MEMO) == 0
    _claim(tmp_path, monitored=True)
    assert len(dicom_mod._HEADER_MEMO) == 3


def test_single_file_claim_declines_non_dicom(tmp_path):
    p = tmp_path / "junk.dcm"
    p.write_bytes(b"nope")
    assert DicomAdapter.claim(ClaimContext(p), DiscoveryState()) is None
