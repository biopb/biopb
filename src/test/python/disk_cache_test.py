"""Tests for the client-side on-disk chunk cache (docs/client-disk-cache.md)."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from biopb.tensor import _diskcache as dc


def unified_batch(arr: np.ndarray) -> pa.RecordBatch:
    """A batch in the server's unified cache schema: [data, shape, dtype]."""
    return pa.RecordBatch.from_arrays(
        [
            pa.array([arr.tobytes()], type=pa.binary()),
            pa.array([list(arr.shape)], type=pa.list_(pa.int64())),
            pa.array([arr.dtype.str], type=pa.string()),
        ],
        names=["data", "shape", "dtype"],
    )


def decode(batch: pa.RecordBatch) -> np.ndarray:
    dtype = np.dtype(batch.column("dtype")[0].as_py())
    shape = tuple(batch.column("shape").to_pylist()[0])
    buf = batch.column("data").buffers()[2]
    return np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


@pytest.fixture
def settings(tmp_path) -> dc.Settings:
    return dc.Settings(root=tmp_path, budget=10**9, ttl=7 * 24 * 3600, min_free=0)


LOC = "grpc://remote-host:8815"


def age(path: Path, seconds: float) -> None:
    """Backdate a file's mtime by *seconds*."""
    t = time.time() - seconds
    os.utime(path, (t, t))


# --- settings ------------------------------------------------------------- #


def test_disabled_by_default():
    assert dc.load_settings(env={}) is None


def test_zero_budget_is_disabled():
    assert dc.load_settings(env={dc.ENV_BUDGET: "0"}) is None


def test_budget_enables_and_parses_units(tmp_path):
    s = dc.load_settings(env={dc.ENV_BUDGET: "2GiB", dc.ENV_DIR: str(tmp_path)})
    assert s is not None
    assert s.budget == 2 * 1024**3
    assert s.root == tmp_path
    assert s.ttl == dc._TTL_DEFAULT


def test_unparseable_budget_disables_rather_than_raising():
    assert dc.load_settings(env={dc.ENV_BUDGET: "not-a-size"}) is None


def test_sweep_threshold_scales_with_a_small_budget():
    """A small budget must be swept proportionally, not overrun by a fixed 64 MiB."""
    small = dc.Settings(root=Path("/x"), budget=1024, ttl=1.0, min_free=0)
    big = dc.Settings(root=Path("/x"), budget=64 * 1024**3, ttl=1.0, min_free=0)
    assert small.sweep_after == dc._SWEEP_AFTER_BYTES_MIN
    assert big.sweep_after == big.budget // 16


# --- keying --------------------------------------------------------------- #


def test_distinct_chunk_ids_get_distinct_paths(settings):
    a = dc.chunk_path(settings.root, LOC, b"chunk-a")
    b = dc.chunk_path(settings.root, LOC, b"chunk-b")
    assert a != b


def test_same_chunk_id_on_two_servers_does_not_collide(settings):
    """chunk_id embeds array_id, which is server-local -- not server-unique.

    In-process this is covered incidentally because the pools are keyed
    (location, token); a shared directory has to put the location back.
    """
    cid = b"identical-chunk-id"
    one = dc.chunk_path(settings.root, "grpc://host-a:8815", cid)
    two = dc.chunk_path(settings.root, "grpc://host-b:8815", cid)
    assert one != two


def test_content_version_header_changes_the_key(settings):
    """A re-registered source mints version-wrapped ids, so its entries are new.

    Mirrors the biopb/biopb#178 wrapper: ``[0xFF][fmt][uint32 len][cv]`` ahead of
    the legacy chunk_id. The client never parses it -- it just has to not collide.
    """
    inner = b"\x00\x00\x00\x03abc" + b"bounds"
    v1 = b"\xff\x01\x00\x00\x00\x02v1" + inner
    v2 = b"\xff\x01\x00\x00\x00\x02v2" + inner
    assert dc.chunk_path(settings.root, LOC, v1) != dc.chunk_path(
        settings.root, LOC, v2
    )
    assert dc.chunk_path(settings.root, LOC, inner) != dc.chunk_path(
        settings.root, LOC, v1
    )


# --- read / write --------------------------------------------------------- #


def test_miss_on_empty_cache(settings):
    assert dc.read_batch(settings, LOC, b"absent") is None


@pytest.mark.parametrize("dtype", ["<u2", ">u2", "<f4", "|u1"])
def test_roundtrip_preserves_values_and_endianness(settings, dtype):
    """Endianness rides in the dtype string, as it does on every other path."""
    arr = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    assert dc.write_batch(settings, LOC, b"c", unified_batch(arr)) is not None
    got = dc.read_batch(settings, LOC, b"c")
    assert np.array_equal(decode(got), arr)
    assert decode(got).dtype == np.dtype(dtype)


def test_write_returns_a_batch_backed_by_the_file(settings):
    """The stored batch is what the caller keeps -- a view onto the mapping, not
    the private in-memory buffer it arrived in."""
    arr = np.arange(16, dtype="u1")
    stored = dc.write_batch(settings, LOC, b"c", unified_batch(arr))
    assert stored is not None
    assert np.array_equal(decode(stored), arr)


def test_read_survives_the_mapping_handle_being_closed(settings):
    """read_batch closes its own MemoryMappedFile; Arrow refcounts the mapping,
    so the decoded array must stay valid (and correct) afterwards."""
    arr = np.arange(1024, dtype="u4")
    dc.write_batch(settings, LOC, b"c", unified_batch(arr))
    got = decode(dc.read_batch(settings, LOC, b"c"))
    assert np.array_equal(got, arr)  # faults pages in after mm.close()


def test_write_is_atomic_leaving_no_temp_files(settings):
    dc.write_batch(settings, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    assert not list(settings.root.rglob("*.tmp"))


def test_corrupt_file_reads_as_a_miss_and_is_removed(settings):
    """Recovery is "unlink whatever does not parse" -- the caller just refetches."""
    dc.write_batch(settings, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    path = dc.chunk_path(settings.root, LOC, b"c")
    path.write_bytes(b"not an arrow stream")
    assert dc.read_batch(settings, LOC, b"c") is None
    assert not path.exists()


def test_write_failure_returns_none_rather_than_raising(settings, monkeypatch):
    """A full or read-only filesystem degrades to the in-memory path."""

    def boom(*a, **k):
        raise OSError("ENOSPC")

    monkeypatch.setattr(dc.os, "rename", boom)
    out = dc.write_batch(settings, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    assert out is None


# --- TTL ------------------------------------------------------------------ #


def test_expired_entry_reads_as_a_miss_and_is_unlinked(tmp_path):
    s = dc.Settings(root=tmp_path, budget=10**9, ttl=1.0, min_free=0)
    dc.write_batch(s, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    path = dc.chunk_path(s.root, LOC, b"c")
    age(path, 10)
    assert dc.read_batch(s, LOC, b"c") is None
    assert not path.exists()


def test_sweep_expires_entries_nobody_reads(tmp_path):
    """Lazy TTL only reclaims what someone touches, and a single-pass scan's
    garbage is never touched again -- so the sweep must apply the TTL too."""
    s = dc.Settings(root=tmp_path, budget=10**9, ttl=1.0, min_free=0)
    for i in range(5):
        dc.write_batch(s, LOC, f"c{i}".encode(), unified_batch(np.zeros(8, dtype="u1")))
        age(dc.chunk_path(s.root, LOC, f"c{i}".encode()), 10)
    assert dc.maybe_sweep(s)
    assert not list(s.root.rglob(f"*{dc._SUFFIX}"))


def test_a_generous_ttl_does_not_evict_within_a_session(settings):
    """The TTL collects abandoned datasets; it must not manage hit rate."""
    dc.write_batch(settings, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    age(dc.chunk_path(settings.root, LOC, b"c"), 3 * 3600)  # a long session
    assert dc.read_batch(settings, LOC, b"c") is not None


# --- recency -------------------------------------------------------------- #


def test_stale_mtime_is_bumped_on_read(settings):
    dc.write_batch(settings, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    path = dc.chunk_path(settings.root, LOC, b"c")
    age(path, dc._MTIME_BUMP_AFTER + 60)
    before = path.stat().st_mtime
    dc.read_batch(settings, LOC, b"c")
    assert path.stat().st_mtime > before


def test_fresh_mtime_is_not_rewritten_on_every_hit(settings):
    """Hand-rolled relatime: one metadata write per chunk per hour, not per hit."""
    dc.write_batch(settings, LOC, b"c", unified_batch(np.zeros(8, dtype="u1")))
    path = dc.chunk_path(settings.root, LOC, b"c")
    before = path.stat().st_mtime
    for _ in range(3):
        dc.read_batch(settings, LOC, b"c")
    assert path.stat().st_mtime == before


# --- eviction ------------------------------------------------------------- #


def write_aged(s: dc.Settings, name: bytes, nbytes: int, seconds_old: float) -> Path:
    dc.write_batch(s, LOC, name, unified_batch(np.zeros(nbytes, dtype="u1")))
    path = dc.chunk_path(s.root, LOC, name)
    age(path, seconds_old)
    return path


def test_sweep_evicts_oldest_first_until_under_budget(tmp_path):
    s = dc.Settings(root=tmp_path, budget=45_000, ttl=10**9, min_free=0)
    paths = [write_aged(s, f"c{i}".encode(), 10_000, 1000 - i) for i in range(10)]
    assert dc.maybe_sweep(s)
    survivors = [i for i, p in enumerate(paths) if p.exists()]
    # The newest survive, contiguously from the end.
    assert survivors == list(range(10 - len(survivors), 10))
    assert 0 < len(survivors) < 10


def test_sweep_is_a_noop_under_budget(tmp_path):
    s = dc.Settings(root=tmp_path, budget=10**9, ttl=10**9, min_free=0)
    paths = [write_aged(s, f"c{i}".encode(), 1000, 100) for i in range(5)]
    assert dc.maybe_sweep(s)
    assert all(p.exists() for p in paths)


def test_a_reread_chunk_outlives_a_scan_that_never_revisits(tmp_path):
    """Scan resistance, the property that makes oldest-first the right policy.

    A single-pass scan's chunks keep their original write mtime forever, while a
    revisited chunk carries a fresh one -- so the scan's own garbage is always
    evicted first and an interactive working set survives it.
    """
    s = dc.Settings(root=tmp_path, budget=25_000, ttl=10**9, min_free=0)
    hot = write_aged(s, b"hot", 10_000, 5000)  # written long ago...
    scan = [write_aged(s, f"s{i}".encode(), 10_000, 1000 - i) for i in range(5)]

    dc.read_batch(s, LOC, b"hot")  # ...but revisited now, so its mtime is fresh
    assert dc.maybe_sweep(s)

    assert hot.exists()
    assert sum(p.exists() for p in scan) < len(scan)


def test_free_space_floor_evicts_even_when_under_budget(tmp_path, monkeypatch):
    """Whatever budget a user picks will be wrong on someone's laptop."""
    s = dc.Settings(root=tmp_path, budget=10**9, ttl=10**9, min_free=50_000)
    paths = [write_aged(s, f"c{i}".encode(), 10_000, 1000 - i) for i in range(5)]
    monkeypatch.setattr(dc, "_free_bytes", lambda _p: 20_000)
    assert dc.maybe_sweep(s)
    assert not all(p.exists() for p in paths)


def test_unlinked_entry_stays_readable_through_a_live_mapping(settings):
    """The property that lets the sweeper run with no reader coordination:
    unlink-while-mapped keeps the pages valid until the last munmap."""
    arr = np.arange(4096, dtype="u4")
    dc.write_batch(settings, LOC, b"c", unified_batch(arr))
    held = decode(dc.read_batch(settings, LOC, b"c"))
    dc.chunk_path(settings.root, LOC, b"c").unlink()
    assert np.array_equal(held, arr)
    assert dc.read_batch(settings, LOC, b"c") is None  # a later reader just misses


# --- sweep coordination --------------------------------------------------- #


def test_only_one_holder_sweeps_at_a_time(settings):
    """N processes must not all scan-and-evict at once, or they over-eviction by
    a factor of N -- each seeing the same pre-sweep total."""
    from biopb._lifecycle.file_lock import ExclusiveFileLock

    settings.root.mkdir(parents=True, exist_ok=True)
    other = ExclusiveFileLock(settings.root / dc._LOCK_NAME)
    assert other.acquire(timeout=0.0)
    try:
        assert dc.maybe_sweep(settings) is False
    finally:
        other.release()
    assert dc.maybe_sweep(settings) is True


def test_write_accounting_triggers_a_sweep(tmp_path, monkeypatch):
    s = dc.Settings(root=tmp_path, budget=1024, ttl=10**9, min_free=0)
    dc.reset_write_accounting()
    calls = []
    monkeypatch.setattr(dc, "maybe_sweep", lambda st: calls.append(st))

    dc.note_written(s, s.sweep_after // 2)
    assert calls == []
    dc.note_written(s, s.sweep_after)
    assert len(calls) == 1


def test_sweep_tolerates_a_missing_root(tmp_path):
    s = dc.Settings(root=tmp_path / "never-created", budget=1, ttl=1, min_free=0)
    dc.maybe_sweep(s)  # must not raise


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX statvfs")
def test_free_bytes_reports_a_plausible_number(tmp_path):
    free = dc._free_bytes(tmp_path)
    assert free is not None and free > 0


# --- the _pool gate ------------------------------------------------------- #
#
# The remote-only rule is the invariant that keeps this path from regressing
# localhost, where the server's own segment is already warm.


@pytest.fixture
def pool(tmp_path, monkeypatch):
    from biopb.tensor import _pool

    monkeypatch.setenv(dc.ENV_BUDGET, "1GiB")
    monkeypatch.setenv(dc.ENV_DIR, str(tmp_path))
    _pool._disk_cache_settings.cache_clear()
    yield _pool
    _pool._disk_cache_settings.cache_clear()


@pytest.mark.parametrize(
    "location", ["grpc://localhost:8815", "grpc://127.0.0.1:8815", "grpc://[::1]:8815"]
)
def test_localhost_never_uses_the_disk_cache(pool, location):
    """Writing our own copy would cost several times the ~1 ms miss it saves."""
    assert pool._disk_cache_for(location) is None


def test_remote_uses_the_disk_cache(pool, tmp_path):
    s = pool._disk_cache_for("grpc://remote-host.invalid:8815")
    assert s is not None and s.root == tmp_path


def test_remote_is_off_when_no_budget_is_configured(pool, monkeypatch):
    monkeypatch.delenv(dc.ENV_BUDGET)
    pool._disk_cache_settings.cache_clear()
    assert pool._disk_cache_for("grpc://remote-host.invalid:8815") is None


def test_fork_reset_drops_the_memoized_settings(pool):
    assert pool._disk_cache_for("grpc://remote-host.invalid:8815") is not None
    pool._reset_pools_after_fork()
    assert pool._disk_cache_settings.cache_info().currsize == 0
