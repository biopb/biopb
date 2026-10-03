"""The OME-TIFF claim's probe goes through the shared memo (see
signature_memo_test.py for the memo's own behavior): an unchanged file is not
probed again, and a claim that will not be repeated leaves nothing behind."""

from pathlib import Path

import biopb_tensor_server.adapters.ome_tiff as ome_tiff_mod
import pytest
from biopb_tensor_server.adapters.ome_tiff import _get_ome_files

SIG_A = (1, 100, 2048, 111, 111)
SIG_B = (1, 100, 4096, 222, 222)  # same inode, file grew + newer mtime


@pytest.fixture(autouse=True)
def _clear_cache():
    ome_tiff_mod._OME_PROBE_MEMO.clear()
    yield
    ome_tiff_mod._OME_PROBE_MEMO.clear()


@pytest.fixture
def probe(monkeypatch):
    """Replace the real probe with a counter returning a scripted value."""
    calls = []
    value = {"ret": ("a.tif",)}

    def fake(path):
        calls.append(str(path))
        return value["ret"]

    monkeypatch.setattr(ome_tiff_mod, "_probe_ome_files", fake)
    return calls, value


def test_an_unchanged_file_is_probed_once(probe):
    calls, _ = probe
    p = Path("/data/img.tif")
    assert _get_ome_files(p, SIG_A) == ("a.tif",)
    assert _get_ome_files(p, SIG_A) == ("a.tif",)
    assert calls == [str(p)]
    _get_ome_files(p, SIG_B)  # a byte change forces a re-probe
    assert len(calls) == 2


def test_no_ome_xml_is_remembered_too(probe):
    calls, value = probe
    value["ret"] = None
    p = Path("/data/plain.tif")
    assert _get_ome_files(p, SIG_A) is None
    assert _get_ome_files(p, SIG_A) is None
    assert calls == [str(p)]


def test_memoize_false_probes_every_time_and_keeps_nothing(probe):
    calls, _ = probe
    p = Path("/data/img.tif")
    _get_ome_files(p, SIG_A, memoize=False)
    _get_ome_files(p, SIG_A, memoize=False)
    assert len(calls) == 2
    assert len(ome_tiff_mod._OME_PROBE_MEMO) == 0


def test_a_probe_that_cannot_read_is_not_memoized(tmp_path, monkeypatch):
    p = tmp_path / "s.ome.tif"
    p.write_bytes(b"x")
    state = {"fail": True}

    def flaky(path):
        if state["fail"]:
            raise OSError("transient")
        return ("a.tif",)

    monkeypatch.setattr(ome_tiff_mod, "_probe_ome_files", flaky)
    assert _get_ome_files(p) is None
    state["fail"] = False
    assert _get_ome_files(p) == ("a.tif",)
