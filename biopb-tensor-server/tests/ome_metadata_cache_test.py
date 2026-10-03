"""Signature-keyed memoization of the OME-XML probe (biopb/biopb#56 item 6).

`_get_ome_files` opens every monitored .tif just to
learn whether it carries OME-XML — the dominant cost of the post-#63 steady-state
claim phase. The result is a pure function of the file's bytes, so it is cached on
the state walk's content-identity signature. These tests pin: a repeat with the same
signature does not reopen the file; a changed signature re-probes; a cached ``None``
("no OME-XML") still counts as a hit; ``signature=None`` (live walk) never caches;
and the cache stays bounded.
"""

from pathlib import Path

import biopb_tensor_server.adapters.ome_tiff as ome_tiff_mod
import pytest
from biopb_tensor_server.adapters.ome_tiff import _get_ome_files


@pytest.fixture(autouse=True)
def _clear_cache():
    ome_tiff_mod._OME_PROBE_MEMO.clear()
    yield
    ome_tiff_mod._OME_PROBE_MEMO.clear()


@pytest.fixture
def counting_probe(monkeypatch):
    """Replace the real tifffile probe with a counter returning a scripted value."""
    calls = {"paths": []}
    value = {"ret": ("a.tif",)}

    def fake(path):
        calls["paths"].append(str(path))
        return value["ret"]

    monkeypatch.setattr(ome_tiff_mod, "_probe_ome_files", fake)
    return calls, value


SIG_A = (1, 100, 2048, 111, 111)
SIG_B = (1, 100, 4096, 222, 222)  # same inode, file grew + newer mtime


def test_same_signature_hits_cache(counting_probe):
    calls, _ = counting_probe
    p = Path("/data/img.tif")

    assert _get_ome_files(p, SIG_A) == ("a.tif",)
    assert _get_ome_files(p, SIG_A) == ("a.tif",)

    assert calls["paths"] == [str(p)]  # probed exactly once


def test_changed_signature_reprobes(counting_probe):
    calls, _ = counting_probe
    p = Path("/data/img.tif")

    _get_ome_files(p, SIG_A)
    _get_ome_files(p, SIG_B)

    assert calls["paths"] == [str(p), str(p)]  # byte change forced a re-probe


def test_cached_none_is_a_hit(counting_probe):
    calls, value = counting_probe
    value["ret"] = None  # this tiff has no embedded OME-XML
    p = Path("/data/plain.tif")

    assert _get_ome_files(p, SIG_A) is None
    assert _get_ome_files(p, SIG_A) is None

    assert calls["paths"] == [str(p)]  # None is cached, not re-probed


def test_signature_none_never_caches(counting_probe):
    calls, _ = counting_probe
    p = Path("/data/img.tif")

    _get_ome_files(p, None)
    _get_ome_files(p, None)

    assert calls["paths"] == [str(p), str(p)]  # uncached: probed every time
    assert len(ome_tiff_mod._OME_PROBE_MEMO) == 0


def test_distinct_paths_are_independent(counting_probe):
    calls, _ = counting_probe
    a, b = Path("/data/a.tif"), Path("/data/b.tif")

    _get_ome_files(a, SIG_A)
    _get_ome_files(b, SIG_A)
    _get_ome_files(a, SIG_A)
    _get_ome_files(b, SIG_A)

    assert calls["paths"] == [str(a), str(b)]  # each probed once, then cached


def test_cache_is_bounded(counting_probe, monkeypatch):
    calls, _ = counting_probe
    monkeypatch.setattr(ome_tiff_mod._OME_PROBE_MEMO, "max_entries", 4)

    for i in range(6):
        _get_ome_files(Path(f"/data/f{i}.tif"), (1, i, 0, 0, 0))

    assert len(ome_tiff_mod._OME_PROBE_MEMO) == 4  # capped

    # The two oldest were evicted, so re-requesting f0 re-probes.
    n_before = len(calls["paths"])
    _get_ome_files(Path("/data/f0.tif"), (1, 0, 0, 0, 0))
    assert len(calls["paths"]) == n_before + 1
