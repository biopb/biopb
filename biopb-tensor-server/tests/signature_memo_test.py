"""SignatureMemo: a content probe reused while the file's identity is unchanged."""

import biopb_tensor_server.adapters.ome_tiff as ome_tiff_mod
import pytest
from biopb_tensor_server.core.signature_memo import SignatureMemo, file_signature

SIG_A = (1, 100, 2048, 111, 111)
SIG_B = (1, 100, 4096, 222, 222)


def counting(value):
    calls = []

    def compute():
        calls.append(1)
        return value

    return calls, compute


def test_hit_skips_compute():
    memo = SignatureMemo(10)
    calls, compute = counting("x")
    assert memo.get("/a", compute, SIG_A) == "x"
    assert memo.get("/a", compute, SIG_A) == "x"
    assert len(calls) == 1


def test_changed_signature_recomputes():
    memo = SignatureMemo(10)
    calls, compute = counting("x")
    memo.get("/a", compute, SIG_A)
    memo.get("/a", compute, SIG_B)
    assert len(calls) == 2


def test_paths_sharing_a_signature_are_independent():
    memo = SignatureMemo(10)
    assert memo.get("/a", lambda: "a", SIG_A) == "a"
    assert memo.get("/b", lambda: "b", SIG_A) == "b"
    assert memo.get("/a", lambda: "stale", SIG_A) == "a"


def test_none_is_a_result():
    memo = SignatureMemo(10)
    calls, compute = counting(None)
    assert memo.get("/a", compute, SIG_A) is None
    assert memo.get("/a", compute, SIG_A) is None
    assert len(calls) == 1


def test_a_file_that_cannot_be_statted_is_not_memoized(tmp_path):
    memo = SignatureMemo(10)
    calls, compute = counting("x")
    gone = tmp_path / "gone"
    memo.get(gone, compute)
    memo.get(gone, compute)
    assert len(calls) == 2
    assert len(memo) == 0


def test_own_stat_is_the_default_signature(tmp_path):
    memo = SignatureMemo(10)
    p = tmp_path / "f"
    p.write_bytes(b"1")
    calls, compute = counting("x")
    memo.get(p, compute)
    memo.get(p, compute)
    assert len(calls) == 1
    p.write_bytes(b"22")  # size changes the signature
    memo.get(p, compute)
    assert len(calls) == 2
    assert file_signature(p) is not None


def test_a_compute_that_raises_caches_nothing():
    memo = SignatureMemo(10)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("nfs hiccup")
        return "ok"

    with pytest.raises(OSError):
        memo.get("/a", flaky, SIG_A)
    assert memo.get("/a", flaky, SIG_A) == "ok"
    assert len(memo) == 1


def test_bounded_lru_evicts_the_oldest():
    memo = SignatureMemo(2)
    calls, compute = counting("x")
    for name in ("/a", "/b", "/a", "/c"):  # /a is touched, so /b goes
        memo.get(name, compute, SIG_A)
    assert len(memo) == 2
    n = len(calls)
    memo.get("/a", compute, SIG_A)
    assert len(calls) == n  # still cached
    memo.get("/b", compute, SIG_A)
    assert len(calls) == n + 1  # evicted


def test_ome_probe_that_cannot_read_is_not_memoized(tmp_path, monkeypatch):
    ome_tiff_mod._OME_PROBE_MEMO.clear()
    p = tmp_path / "s.ome.tif"
    p.write_bytes(b"x")
    state = {"fail": True}

    def probe(path):
        if state["fail"]:
            raise OSError("transient")
        return ("a.tif",)

    monkeypatch.setattr(ome_tiff_mod, "_probe_ome_files", probe)
    assert ome_tiff_mod._get_ome_files(p) is None
    state["fail"] = False
    assert ome_tiff_mod._get_ome_files(p) == ("a.tif",)
    ome_tiff_mod._OME_PROBE_MEMO.clear()
