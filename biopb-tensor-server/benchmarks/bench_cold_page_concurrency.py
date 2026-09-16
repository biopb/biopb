"""Benchmark: what does a page-cache-cold cache HIT cost concurrent readers?

The cache serves a hit by touching bytes inside a ``pa.memory_map()``'d
segment file (``ArrowFileBackend._read_batch_from_segment``). If the pages
are not resident, the touch is a synchronous page fault that CPython does not
release the GIL around. The question is who that fault stalls, on two paths:

- **do_get** (plain gRPC): the SERVER faults. ``start_compute`` holds the one
  backend-wide ``ArrowFileBackend._lock`` across ``_hydrate_hit`` ->
  ``_read_batch_from_segment``, so whatever faults inside that call blocks
  every other reader's lock acquisition, for any client in any process.
  Measured (probe, 2026-09-16): that is only the IPC metadata + schema pages
  -- ~200 KiB, ~1 ms on a SATA SSD -- because ``pa.ipc.read_message`` on a
  memory map is zero-copy; the 6 MiB body is a slice of the mapping and
  faults later, inside Flight's C++ serializer (``do_get`` returns a
  ``RecordBatchStream``; no Python touches the bytes), with the GIL released
  and no lock held. So concurrent cold hits parallelize down to the disk.
- **cachefile** (localhost handoff, issue #9): ``chunk_locate`` is a dict
  lookup; the CLIENT mmaps the segment and faults in its own process.

Two modes, both with warm readers as a real ``ProcessPoolExecutor`` (spawn;
``concurrent_read_test.py``'s convention), so a shared GIL never explains a
cross-process result:

- ``--mode warm-stall``: N workers hammer page-cache-warm chunks in their own
  segments while the parent issues one cold read; report the warm latency
  spike overlapping the cold read. Needs one chunk per segment (the cap is
  set just *below* one entry's size, since rollover is checked after the
  write) so the eviction can't touch the warm entries.
- ``--mode cold-burst``: N workers each fire one cold read at once, released
  from a barrier, with production 64 MB segment packing; report burst wall
  vs single cold latency as a serialization fraction (1 = serial, 0 =
  parallel), plus each worker's disk bytes, a readahead-spillover residency
  check (``mincore``) on the segment-mates, and the segment file's raw
  O_DIRECT ceiling -- if the burst runs near that ceiling the fraction
  reflects the disk, not the lock, and the run says so.

Setup gotchas that produce silently-wrong numbers, all handled here:

- A segment still open for writing serves its chunks from the in-memory
  ``_entries`` mirror (heap copies) and never touches the file; only a
  sealed segment is mmap-served. Cold-burst seals via ``_close_segment``
  after warm-up and asserts the mirror count is 0.
- ``posix_fadvise(DONTNEED)`` cannot evict pages a live mapping pins, and
  does not reliably evict dirty pages: drop the server mmap first, then
  ``fsync``, then ``DONTNEED``. Every round verifies real disk bytes.
- On the dev box root is one LVM volume across a SATA SSD and an NVMe; which
  disk a file lands on follows its ext4 block group, so ``--scratch-dir`` is a
  hint and the printed O_DIRECT ceiling is the truth.
- A fresh worker's first ``.compute()`` pays one-off costs; workers are
  primed with a discarded read before anything is timed.

Run:
    python benchmarks/bench_cold_page_concurrency.py --mode warm-stall [--n-warm 4]
    python benchmarks/bench_cold_page_concurrency.py --mode cold-burst [--n-burst 4]
        [--chunk-mib 6] [--rounds 5] [--transfer both] [--scratch-dir /var/tmp]
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Process-pool worker globals. Must be module-level so `spawn` can pickle
# references to them; each worker process gets its own copy.
# ---------------------------------------------------------------------------
_PROCESS_WORKER_CLIENT = None
_PROCESS_BARRIER = None


def _init_process_worker(location: str, use_cachefile: bool, barrier=None) -> None:
    """Runs once per worker process, before any task: pick the transfer path.

    A `spawn`ed worker is a fresh interpreter -- the env var must be set
    before `biopb.tensor.client` is first imported in *this* process, not
    reloaded like the parent does, since nothing has imported it yet here.

    ``barrier`` (cold-burst mode) has to arrive through ``initargs``: a
    multiprocessing Barrier can only be shared by inheritance, so passing it
    as a task argument would fail to pickle.
    """
    if use_cachefile:
        os.environ.pop("BIOPB_CACHEFILE_TRANSFER_DISABLED", None)
    else:
        os.environ["BIOPB_CACHEFILE_TRANSFER_DISABLED"] = "1"

    from biopb.tensor import TensorFlightClient

    global _PROCESS_WORKER_CLIENT, _PROCESS_BARRIER
    _PROCESS_WORKER_CLIENT = TensorFlightClient(location, cache_bytes=0)
    _PROCESS_WORKER_CLIENT.list_sources()
    _PROCESS_BARRIER = barrier
    atexit.register(_cleanup_worker_client)


def _cleanup_worker_client() -> None:
    global _PROCESS_WORKER_CLIENT
    if _PROCESS_WORKER_CLIENT is not None:
        try:
            _PROCESS_WORKER_CLIENT.close()
        except Exception:
            pass
        _PROCESS_WORKER_CLIENT = None


def _process_worker_pid(_: int) -> int:
    """Forces worker startup so it isn't timed inside a round."""
    return os.getpid()


def _process_cachefile_used(location: str):
    """Whether THIS worker's client actually took the cachefile fast path.

    The parent's ``_pool`` never probes in modes where every read runs in a
    worker, so its capability cache reads ``None`` there -- ask a worker.
    """
    import biopb.tensor._pool as pool

    return pool._cachefile_support.get(location)


def _raw_read_ceiling_mb_s(path: str) -> float:
    """Cold sequential O_DIRECT read of ``path``: the device's ceiling for it.

    Gives the number a burst's aggregate throughput has to be read against:
    if the single cold read already runs near this, a burst cannot
    parallelize no matter what the lock does, and the serialization fraction
    says nothing about the lock. Which physical disk backs a file is LVM's
    choice on the dev box (root spans a SATA SSD and an NVMe), so it is
    measured per run rather than assumed.
    """
    import mmap as _mmap

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = _mmap.mmap(-1, 1 << 20)  # page-aligned, as O_DIRECT requires
    total = 0
    t0 = time.perf_counter()
    try:
        while True:
            n = os.readv(fd, [buf])
            if n <= 0:
                break
            total += n
    finally:
        os.close(fd)
        buf.close()
    elapsed = time.perf_counter() - t0
    return (total / 2**20) / elapsed if elapsed else float("inf")


def _process_warm_loop(arr, deadline: float):
    """Read ``arr`` repeatedly until wall-clock ``deadline`` (``time.time()``).

    Wall clock, not ``perf_counter``, so timestamps are comparable against the
    parent process's cold-read window -- ``perf_counter``'s reference point
    is undefined across separate processes even where it happens to be
    ``CLOCK_MONOTONIC``-backed.
    """
    samples = []
    while time.time() < deadline:
        t0 = time.time()
        arr.compute()
        dt = time.time() - t0
        samples.append((t0, dt))
    return samples


def _process_cold_read(arr, use_barrier: bool):
    """One read of ``arr``, optionally released from the shared barrier.

    Returns ``(t0, dt, disk_bytes)``: wall-clock start, latency, and this
    worker's own ``/proc/self/io`` read delta. The delta matters for the
    cachefile path, where the fault -- and so the disk I/O -- happens in the
    worker, invisible to the parent's counter.
    """
    if use_barrier:
        _PROCESS_BARRIER.wait(timeout=30)
    before = _read_bytes()
    t0 = time.time()
    arr.compute()
    dt = time.time() - t0
    return (t0, dt, _read_bytes() - before)


def _read_bytes() -> int:
    with open("/proc/self/io") as handle:
        for line in handle:
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    return 0


def _resident_fraction(path: str, offset: int, length: int) -> float:
    """Fraction of ``[offset, offset+length)`` of ``path`` resident in page cache.

    ``mincore`` over a throwaway read-only mapping: mapping does not fault
    anything in, and it is unmapped before returning so it can't pin pages
    against a later ``DONTNEED``.
    """
    import ctypes
    import mmap as _mmap

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    page = _mmap.PAGESIZE
    start = offset - offset % page
    span = offset + length - start
    npages = (span + page - 1) // page
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, span, _mmap.PROT_READ, _mmap.MAP_SHARED, fd, start)
    finally:
        os.close(fd)
    if addr is None or addr == 2**64 - 1:
        raise OSError(ctypes.get_errno(), "mmap failed")
    vec = (ctypes.c_ubyte * npages)()
    try:
        if libc.mincore(addr, span, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
    finally:
        libc.munmap(addr, span)
    return sum(v & 1 for v in vec) / npages


def _build_source(root: Path, source_id: str, chunk_bytes: int):
    import zarr

    edge = max(64, int((chunk_bytes / 2) ** 0.5))  # square uint16 chunk
    shape = (edge, edge)
    zpath = root / source_id / "test.zarr"
    arr = zarr.open_array(
        str(zpath), mode="w", shape=shape, chunks=shape, dtype="uint16"
    )
    rng = np.random.default_rng(abs(hash(source_id)) % (2**32))
    arr[:] = rng.integers(0, 65535, size=shape, dtype="uint16")
    return zpath, shape


def _start_server(cache_dir: Path, max_segment_bytes: int | None):
    """``max_segment_bytes=None`` keeps CacheConfig's production default (64 MB)."""
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.core.config import CacheConfig

    config_kwargs = {
        "file_cache_dir": cache_dir / "server-cache",
        "file_max_total_bytes": 64 * 1024**3,
    }
    if max_segment_bytes is not None:
        config_kwargs["file_max_segment_bytes"] = max_segment_bytes

    CacheManager.reset()
    CacheManager.initialize(CacheConfig(**config_kwargs))
    server = TensorFlightServer("grpc://localhost:0")
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(0.5)
    return server


def _switch_transfer_mode(use_cachefile: bool):
    """(Re)read ``BIOPB_CACHEFILE_TRANSFER_DISABLED`` in THIS (parent) process.

    Mirrors ``bench_cachefile.py``'s ``_fresh_client``: the per-location
    capability cache (``_pool._cachefile_support``) and the env-var gate are
    both read once and then cached at import time, so switching path for an
    already-imported process needs a reload, not just an env var write. Only
    affects the parent -- worker processes read the env var fresh at their
    own (first) import, set by ``_init_process_worker``.
    """
    if use_cachefile:
        os.environ.pop("BIOPB_CACHEFILE_TRANSFER_DISABLED", None)
    else:
        os.environ["BIOPB_CACHEFILE_TRANSFER_DISABLED"] = "1"
    import biopb.tensor._pool as pool
    import biopb.tensor.client as client_mod

    importlib.reload(pool)
    importlib.reload(client_mod)
    return client_mod.TensorFlightClient, pool


def _segments_for_keys(backend, keys) -> set:
    with backend._lock:
        return {backend._metadata[key].segment_id for key in keys}


def _evict_page_cache(backend, keys) -> None:
    """Drop the mmap AND the OS page cache pages backing ``keys``' segments.

    A "chunk" as requested can be split across several transfer-chunk cache
    entries (``PREFERRED_ARROW_BATCH_BYTES``/``MAX_ARROW_BATCH_BYTES`` cap the
    grid, see ``bench_cachefile.py`` and the 8 MiB-cap notes), so a source
    maps to a *set* of keys/segments, not necessarily one.

    Mirrors what ``ArrowFileBackend._maybe_release_cold_mmaps`` does on a real
    idle-segment sweep, plus the ``posix_fadvise`` half that actually reclaims
    pages -- ``fadvise`` cannot reclaim while a live mapping still maps them
    (same gotcha as ``bench_plane_latency.py``'s ``_evict``), so the mmap drop
    must happen first, under the lock, before the fadvise call. This also
    evicts the cachefile scenario's pages: page cache is machine-wide, not
    per-process, so a worker process's independent mmap of the same file
    faults too.
    """
    segment_ids = set()
    with backend._lock:
        for key in keys:
            entry_info = backend._metadata[key]
            segment_id = entry_info.segment_id
            seg_info = backend._segment_info(segment_id)
            backend._forget_segment_mmap(segment_id)
            if seg_info is not None:
                seg_info.mmap_released = True
            segment_ids.add(segment_id)
    for segment_id in segment_ids:
        seg_path = backend._segment_path(segment_id)
        fd = os.open(str(seg_path), os.O_RDONLY)
        try:
            # A segment freshly written by this same process can still have
            # dirty pages pending writeback; DONTNEED does not reliably evict
            # a dirty page (the source-file eviction in bench_plane_latency.py
            # never hits this because it never writes what it evicts).
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _run_scenario(
    name: str,
    use_cachefile: bool,
    location: str,
    cold_id: str,
    warm_ids: list,
    backend,
    cold_keys: set,
    rounds: int,
    warm_loop_s: float,
) -> dict:
    print(f"\n=== scenario (process-pool warm readers): {name} ===")

    ColdClient, pool = _switch_transfer_mode(use_cachefile)

    # Build the warm dask arrays once in the parent (picklable graphs, not
    # live connections -- see concurrent_read_test.py's identical pattern),
    # then hand them to worker processes each round.
    build_clients = {w: ColdClient(location, cache_bytes=0) for w in warm_ids}
    warm_arrays = {w: build_clients[w].get_tensor(w) for w in warm_ids}
    for c in build_clients.values():
        c.close()

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=len(warm_ids),
        mp_context=ctx,
        initializer=_init_process_worker,
        initargs=(location, use_cachefile),
    )
    try:
        # Force worker startup so it isn't timed inside round 0.
        list(executor.map(_process_worker_pid, range(len(warm_ids))))
        # Prime each worker with one real, discarded read: the FIRST
        # `.compute()` in a fresh worker pays one-off costs (dask scheduler
        # thread-pool lazy init, connection ramp-up) that otherwise land in
        # a short baseline window and inflate it -- confirmed by a 2s-window
        # probe showing a 0.3s-window "isolated" baseline was ~2.5x too high.
        list(
            executor.map(
                _process_warm_loop,
                [warm_arrays[w] for w in warm_ids],
                [time.time() + 0.1] * len(warm_ids),
            )
        )

        # ---- two baselines, to separate two different effects ----
        # (a) isolated: ONE worker process, nothing else running -- the
        # cleanest "no contention at all" number.
        isolated_deadline = time.time() + 1.0
        isolated_samples = [
            dt
            for (_, dt) in executor.submit(
                _process_warm_loop, warm_arrays[warm_ids[0]], isolated_deadline
            ).result()
        ]
        isolated_p50_ms = (
            statistics.median(isolated_samples) * 1e3 if isolated_samples else 0.0
        )
        print(
            f"isolated baseline (1 worker, no other load): p50={isolated_p50_ms:.2f}ms "
            f"n={len(isolated_samples)}"
        )

        # (b) steady-state: all N workers hammering concurrently, but no cold
        # read yet -- this is what "during a cold read" gets compared to
        # below, since that also has all N workers running. Any gap between
        # (a) and (b) is ordinary N-way dispatch-queueing cost (already
        # documented in the localhost-read-path-ceilings note), not the cold
        # read's doing.
        baseline_deadline = time.time() + 1.5
        baseline_futures = [
            executor.submit(_process_warm_loop, warm_arrays[w], baseline_deadline)
            for w in warm_ids
        ]
        baseline = [dt for f in baseline_futures for (_, dt) in f.result()]
        baseline_p50_ms = statistics.median(baseline) * 1e3
        print(
            f"steady-state baseline ({len(warm_ids)} workers, no cold read): "
            f"p50={baseline_p50_ms:.2f}ms  n={len(baseline)}"
        )

        # ---- the experiment ----
        cold_latencies_ms = []
        overlap_latencies_ms = []
        worst_overlap_ms = []
        not_cold_rounds = 0

        for round_idx in range(rounds):
            _evict_page_cache(backend, cold_keys)

            # A little slack past warm_loop_s so the cold read's window is
            # guaranteed to land fully inside the workers' loop, not clipped
            # at the end.
            loop_deadline = time.time() + warm_loop_s + 0.3
            futures = [
                executor.submit(_process_warm_loop, warm_arrays[w], loop_deadline)
                for w in warm_ids
            ]
            time.sleep(0.05)  # let warm loops reach steady state

            before_io = _read_bytes()
            cold_client = ColdClient(location, cache_bytes=0)
            cold_array = cold_client.get_tensor(cold_id)
            t0 = time.time()
            cold_array.compute()
            cold_latency = time.time() - t0
            cold_window = (t0, t0 + cold_latency)
            cold_client.close()
            disk_read = _read_bytes() - before_io
            if disk_read == 0:
                not_cold_rounds += 1

            per_worker_samples = [f.result() for f in futures]
            overlapping = [
                dt
                for samples in per_worker_samples
                for (start, dt) in samples
                if cold_window[0] <= start <= cold_window[1]
            ]
            cold_latencies_ms.append(cold_latency * 1e3)
            overlap_latencies_ms.extend(x * 1e3 for x in overlapping)

            worst = max(overlapping) * 1e3 if overlapping else 0.0
            worst_overlap_ms.append(worst)
            print(
                f"round {round_idx}: cold={cold_latency * 1e3:8.2f}ms "
                f"disk_read={disk_read / 2**20:6.1f}MiB "
                f"warm-reqs-overlapping-cold={len(overlapping):3d} "
                f"worst-overlap={worst:8.2f}ms"
                + ("  NOT-COLD (fadvise didn't take)" if disk_read == 0 else "")
            )
    finally:
        executor.shutdown(wait=True)

    result = {
        "name": name,
        "isolated_p50_ms": isolated_p50_ms,
        "baseline_p50_ms": baseline_p50_ms,
        "cold_median_ms": statistics.median(cold_latencies_ms),
        "overlap_median_ms": (
            statistics.median(overlap_latencies_ms) if overlap_latencies_ms else None
        ),
        "overlap_n": len(overlap_latencies_ms),
        # The median-of-overlapping stat dilutes fast: most "overlapping"
        # calls (by wall clock) never actually contended for _lock -- only
        # whichever caller was mid-acquire at the instant the cold reader
        # took it. The per-round WORST overlap is the cleaner signal here,
        # especially with process-pool's coarser per-worker sampling rate.
        "worst_overlap_max_ms": max(worst_overlap_ms) if worst_overlap_ms else 0.0,
        "worst_overlap_median_ms": (
            statistics.median(worst_overlap_ms) if worst_overlap_ms else 0.0
        ),
        "not_cold_rounds": not_cold_rounds,
    }

    if use_cachefile:
        used = pool._cachefile_support.get(location)
        print(f"(cachefile fast path engaged for the cold/parent client: {used})")
        result["cachefile_used"] = used

    print()
    print(f"cold read latency:               median={result['cold_median_ms']:.2f}ms")
    print(f"isolated baseline (1 worker):     p50={isolated_p50_ms:.2f}ms")
    print(
        f"steady-state baseline ({len(warm_ids)} workers):  p50={baseline_p50_ms:.2f}ms"
    )
    if result["overlap_median_ms"] is not None:
        ratio = result["overlap_median_ms"] / baseline_p50_ms
        worst_ratio = result["worst_overlap_max_ms"] / baseline_p50_ms
        print(
            f"warm latency DURING a cold read: median={result['overlap_median_ms']:.2f}ms "
            f"({ratio:.1f}x baseline, n={result['overlap_n']})"
        )
        print(
            f"worst single overlapping call:   max={result['worst_overlap_max_ms']:.2f}ms "
            f"({worst_ratio:.1f}x baseline), per-round median of worsts="
            f"{result['worst_overlap_median_ms']:.2f}ms"
        )
    else:
        print(
            "no warm request ever overlapped a cold read -- raise "
            "--warm-loop-s or --rounds"
        )
    if not_cold_rounds:
        print(
            f"WARNING: {not_cold_rounds}/{rounds} rounds read 0 bytes from disk "
            "-- eviction may not have taken; treat those rounds' numbers as a "
            "warm read, not a cold one"
        )

    return result


def _run_cold_burst(
    name: str,
    use_cachefile: bool,
    location: str,
    burst_ids: list,
    backend,
    keys_for_source: dict,
    rounds: int,
) -> dict:
    """N simultaneous cold cache hits, one per worker process, barrier-released.

    The single-cold baseline is measured through the same pool (one worker,
    no barrier), so it pays exactly the per-path costs the burst does.
    Verification: the burst must read ~N x one chunk from disk, summed over
    the parent's counter (do_get faults in the server, which lives here) and
    every worker's own counter (cachefile faults in the worker).
    """
    print(f"\n=== scenario (cold-burst, {len(burst_ids)} workers): {name} ===")

    ColdClient, pool = _switch_transfer_mode(use_cachefile)
    build_clients = {b: ColdClient(location, cache_bytes=0) for b in burst_ids}
    arrays = {b: build_clients[b].get_tensor(b) for b in burst_ids}
    for c in build_clients.values():
        c.close()

    all_keys = {k for b in burst_ids for k in keys_for_source[b]}
    n = len(burst_ids)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n + 1)
    executor = ProcessPoolExecutor(
        max_workers=n,
        mp_context=ctx,
        initializer=_init_process_worker,
        initargs=(location, use_cachefile, barrier),
    )
    single_ms, burst_wall_ms, worker_p50_ms, worker_max_ms = [], [], [], []
    single_disk, burst_disk = [], []
    disk_ok_rounds = 0
    spillover = None
    cachefile_used = None
    try:
        list(executor.map(_process_worker_pid, range(n)))
        # Prime: one discarded warm read per worker (dask/connection one-offs).
        list(
            executor.map(
                _process_cold_read, [arrays[b] for b in burst_ids], [False] * n
            )
        )

        segment_paths = {
            str(backend._segment_path(sid))
            for sid in _segments_for_keys(backend, all_keys)
        }
        ceiling_mb_s = statistics.mean(_raw_read_ceiling_mb_s(p) for p in segment_paths)
        print(
            f"raw O_DIRECT read ceiling of the segment file(s): {ceiling_mb_s:.0f} MB/s"
        )

        for round_idx in range(rounds):
            # ---- single cold read, alone ----
            _evict_page_cache(backend, all_keys)
            parent_before = _read_bytes()
            _, s_dt, s_disk = executor.submit(
                _process_cold_read, arrays[burst_ids[0]], False
            ).result()
            s_disk += _read_bytes() - parent_before
            single_ms.append(s_dt * 1e3)

            if round_idx == 0:
                # Readahead spillover: after faulting chunk 0 alone, how much
                # of its segment-mates is already resident?
                with backend._lock:
                    infos = {
                        b: [backend._metadata[k] for k in keys_for_source[b]]
                        for b in burst_ids
                    }
                seg0 = {i.segment_id for i in infos[burst_ids[0]]}
                fracs = []
                for b in burst_ids[1:]:
                    for info in infos[b]:
                        if info.segment_id in seg0:
                            fracs.append(
                                _resident_fraction(
                                    str(backend._segment_path(info.segment_id)),
                                    info.byte_offset,
                                    info.byte_length,
                                )
                            )
                spillover = statistics.mean(fracs) if fracs else None

            # ---- burst: all N at once ----
            _evict_page_cache(backend, all_keys)
            parent_before = _read_bytes()
            futures = [
                executor.submit(_process_cold_read, arrays[b], True) for b in burst_ids
            ]
            barrier.wait(timeout=30)
            results = [f.result() for f in futures]
            parent_disk = _read_bytes() - parent_before

            starts = [t0 for (t0, _, _) in results]
            ends = [t0 + dt for (t0, dt, _) in results]
            lats = sorted(dt * 1e3 for (_, dt, _) in results)
            wall = (max(ends) - min(starts)) * 1e3
            total_disk = parent_disk + sum(d for (_, _, d) in results)
            expected_disk = n * (s_disk if s_disk else 0)
            disk_ok = s_disk > 0 and total_disk >= 0.8 * expected_disk
            disk_ok_rounds += disk_ok

            burst_wall_ms.append(wall)
            worker_p50_ms.append(statistics.median(lats))
            worker_max_ms.append(lats[-1])
            single_disk.append(s_disk)
            burst_disk.append(total_disk)
            print(
                f"round {round_idx}: single={s_dt * 1e3:7.2f}ms  burst_wall={wall:7.2f}ms  "
                f"worker p50={statistics.median(lats):7.2f}ms max={lats[-1]:7.2f}ms  "
                f"disk={total_disk / 2**20:5.1f}MiB (expect ~{expected_disk / 2**20:.1f})"
                + ("" if disk_ok else "  NOT-COLD")
            )

        if use_cachefile:
            cachefile_used = executor.submit(_process_cachefile_used, location).result()
    finally:
        executor.shutdown(wait=True)

    single = statistics.median(single_ms)
    wall = statistics.median(burst_wall_ms)
    # 0 = burst took one single latency (fully parallel); 1 = N singles (serial).
    serial_fraction = (wall - single) / ((n - 1) * single) if n > 1 else float("nan")
    single_mb_s = (
        (statistics.median(single_disk) / 2**20) / (single / 1e3) if single else 0.0
    )
    burst_mb_s = (statistics.median(burst_disk) / 2**20) / (wall / 1e3) if wall else 0.0
    bandwidth_bound = burst_mb_s >= 0.8 * ceiling_mb_s
    result = {
        "name": name,
        "n": n,
        "single_ms": single,
        "burst_wall_ms": wall,
        "worker_p50_ms": statistics.median(worker_p50_ms),
        "worker_max_ms": statistics.median(worker_max_ms),
        "serial_fraction": serial_fraction,
        "single_mb_s": single_mb_s,
        "burst_mb_s": burst_mb_s,
        "ceiling_mb_s": ceiling_mb_s,
        "bandwidth_bound": bandwidth_bound,
        "spillover": spillover,
        "disk_ok_rounds": disk_ok_rounds,
        "cachefile_used": cachefile_used,
    }

    print()
    print(f"single cold read:        median={single:.2f}ms  ({single_mb_s:.0f} MB/s)")
    print(
        f"burst of {n}:              wall median={wall:.2f}ms  "
        f"(serial would be ~{n * single:.0f}ms, parallel ~{single:.0f}ms)  "
        f"aggregate {burst_mb_s:.0f} MB/s vs device ceiling {ceiling_mb_s:.0f} MB/s"
    )
    print(
        f"per-worker latency:      p50={result['worker_p50_ms']:.2f}ms  "
        f"max={result['worker_max_ms']:.2f}ms"
    )
    print(f"serialization fraction:  {serial_fraction:.2f}  (0=parallel, 1=serial)")
    if bandwidth_bound:
        print(
            "  -> BANDWIDTH-BOUND: the burst saturates the device, so this "
            "fraction reflects the disk, not the lock. Use a faster scratch "
            "dir (--scratch-dir) or smaller chunks to make the test decisive."
        )
    if spillover is not None:
        print(
            f"readahead spillover:     segment-mates {spillover * 100:.1f}% resident after one cold read"
        )
    if use_cachefile:
        print(f"(cachefile fast path engaged in workers: {cachefile_used})")
    if disk_ok_rounds < rounds:
        print(
            f"WARNING: {rounds - disk_ok_rounds}/{rounds} burst rounds read less "
            "disk than N chunks -- eviction may not have taken for every worker"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["warm-stall", "cold-burst"],
        default="warm-stall",
        help=(
            "warm-stall: one cold read while N workers hammer warm chunks in "
            "their own segments. cold-burst: N workers each fire one cold read "
            "simultaneously, with production segment packing"
        ),
    )
    parser.add_argument(
        "--n-warm", type=int, default=4, help="warm-stall: warm reader processes"
    )
    parser.add_argument(
        "--n-burst", type=int, default=4, help="cold-burst: simultaneous cold readers"
    )
    parser.add_argument(
        "--chunk-mib",
        type=float,
        default=6,
        help=(
            "zarr chunk size per source. Kept under the transfer-chunk cap "
            "(~8 MiB, PREFERRED_ARROW_BATCH_BYTES) by default so one source "
            "maps to exactly one cache entry; larger values still work but "
            "split a source across several entries, evicted together"
        ),
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help=(
            "where sources and the server cache are created (default: the "
            "system temp dir). On the dev box root is one LVM volume spanning a SATA "
            "SSD and an NVMe, so the device a file lands on follows the "
            "directory: /tmp measured ~600 MB/s, /var/tmp ~2 GB/s. The run "
            "prints the actual O_DIRECT ceiling of its segment file."
        ),
    )
    parser.add_argument(
        "--warm-loop-s",
        type=float,
        default=0.5,
        help="warm-stall: how long each round's warm worker-process loop keeps running",
    )
    parser.add_argument(
        "--transfer",
        choices=["do_get", "cachefile", "both"],
        default="both",
        help="which read path(s) to exercise for the warm/cold requests",
    )
    args = parser.parse_args()
    burst = args.mode == "cold-burst"

    # Warm-up always goes over the plain socket path -- path doesn't matter
    # for populating the cache, and forcing it keeps setup deterministic.
    TensorFlightClient, _pool = _switch_transfer_mode(use_cachefile=False)

    from biopb_tensor_server import ZarrAdapter
    from biopb_tensor_server.cache import CacheManager

    tmp = Path(tempfile.mkdtemp(prefix="coldpage-bench-", dir=args.scratch_dir))
    n_sources = args.n_burst if burst else args.n_warm + 1
    source_ids = [f"src{i}" for i in range(n_sources)]
    chunk_bytes = int(args.chunk_mib * 1024 * 1024)

    for sid in source_ids:
        _build_source(tmp, sid, chunk_bytes)

    # cold-burst: production packing (64 MB segments) -- no warm readers to
    # shield from the eviction, so let chunks share segments the way a real
    # cache does. warm-stall: exactly one entry per segment, so evicting the
    # "cold" segment can't touch entries the warm workers are reading; rollover
    # triggers when a just-written segment's size is already >= the cap
    # (checked AFTER the write, see ArrowFileBackend.complete_entry), so the
    # cap must sit strictly below one entry's serialized size, not above it.
    max_segment_bytes = None if burst else int(chunk_bytes * 0.95)
    server = _start_server(tmp, max_segment_bytes=max_segment_bytes)
    location = f"grpc://localhost:{server.port}"

    try:
        import zarr

        for sid in source_ids:
            zpath = tmp / sid / "test.zarr"
            server.register_source(
                sid, ZarrAdapter(zarr.open_array(str(zpath), mode="r"), sid, ["y", "x"])
            )

        backend = CacheManager.get_instance()._backend

        # Warm every source once; diff the cache index each time to learn
        # which keys belong to which source. A source can map to more than
        # one cache entry once its chunk exceeds the transfer-chunk cap.
        keys_before = set(backend._metadata.keys())
        keys_for_source = {}
        for sid in source_ids:
            c = TensorFlightClient(location, cache_bytes=0)
            try:
                c.get_tensor(sid).compute()
            finally:
                c.close()
            keys_after = set(backend._metadata.keys())
            new_keys = keys_after - keys_before
            assert new_keys, f"warming {sid!r} added no cache entries"
            keys_for_source[sid] = new_keys
            keys_before = keys_after

        modes = {"do_get": False, "cachefile": True}
        if args.transfer != "both":
            modes = {args.transfer: modes[args.transfer]}

        if burst:
            # Seal the still-open write segment(s). Until a segment seals
            # (64 MB cap reached) its chunks are served from the in-memory
            # RecordBatch mirror in `_entries` -- heap copies that never touch
            # the file -- so an "evicted" read would silently come from RAM
            # (0 disk bytes, pages still non-resident afterwards). Sealing is
            # what `_close_segment` does at rollover: open the mmap, drop the
            # mirrors. Same end state production reaches, just now.
            open_segments = list(backend._pool_writers)
            with backend._write_lock:
                for sid in open_segments:
                    backend._close_segment(sid)
            by_segment = {}
            for sid in source_ids:
                for seg in _segments_for_keys(backend, keys_for_source[sid]):
                    by_segment.setdefault(seg, []).append(sid)
            packing = ", ".join(
                f"seg{seg}:[{' '.join(s)}]" for seg, s in sorted(by_segment.items())
            )
            print(
                f"mode=cold-burst n_burst={args.n_burst} chunk={args.chunk_mib:.0f}MiB "
                f"rounds={args.rounds} transfer={args.transfer} scratch={tmp}\n"
                f"segment packing (production 64MB cap): {packing}\n"
                f"sealed {len(open_segments)} open segment(s); live in-memory mirrors "
                f"now: {len(backend._entries)} (must be 0 for reads to hit the mmap)"
            )
            results = [
                _run_cold_burst(
                    name,
                    uc,
                    location,
                    source_ids,
                    backend,
                    keys_for_source,
                    args.rounds,
                )
                for name, uc in modes.items()
            ]
            if len(results) > 1:
                print("\n=== summary (cold-burst, separate OS processes) ===")
                for r in results:
                    print(
                        f"{r['name']:10} single={r['single_ms']:7.2f}ms  "
                        f"burst_wall={r['burst_wall_ms']:7.2f}ms  "
                        f"worker_max={r['worker_max_ms']:7.2f}ms  "
                        f"serial_fraction={r['serial_fraction']:.2f}  "
                        f"burst {r['burst_mb_s']:.0f}/{r['ceiling_mb_s']:.0f} MB/s"
                        + ("  BANDWIDTH-BOUND" if r["bandwidth_bound"] else "")
                    )
                print(
                    "\nserial_fraction ~1 would mean the N cold faults ran one after "
                    "another; ~0 that they overlapped. Under _lock only the IPC "
                    "metadata pages fault (~200 KiB, ~1 ms); the body faults in "
                    "Flight's C++ serializer off-lock, so do_get is expected to "
                    "overlap too. Workers finishing together (not staggered by one "
                    "single-latency each) is the overlap signature; a BANDWIDTH-BOUND "
                    "flag means the residual fraction is the disk's, not the lock's."
                )
            return

        cold_id, *warm_ids = source_ids
        cold_keys = keys_for_source[cold_id]
        warm_keys = {k for w in warm_ids for k in keys_for_source[w]}

        cold_segments = _segments_for_keys(backend, cold_keys)
        warm_segments = _segments_for_keys(backend, warm_keys)
        assert not (cold_segments & warm_segments), (
            "cold source shares a segment with a warm source -- lower "
            "--chunk-mib so file_max_segment_bytes actually forces one "
            "chunk per segment"
        )

        print(
            f"mode=warm-stall n_warm={args.n_warm} chunk={args.chunk_mib:.0f}MiB "
            f"rounds={args.rounds} transfer={args.transfer} "
            f"(warm readers: separate processes)"
        )

        results = []
        for name, use_cachefile in modes.items():
            result = _run_scenario(
                name,
                use_cachefile,
                location,
                cold_id,
                warm_ids,
                backend,
                cold_keys,
                args.rounds,
                args.warm_loop_s,
            )
            results.append(result)

        if len(results) > 1:
            print("\n=== summary (warm readers are separate OS processes) ===")
            for r in results:
                ratio = (
                    r["overlap_median_ms"] / r["baseline_p50_ms"]
                    if r["overlap_median_ms"] is not None
                    else float("nan")
                )
                worst_ratio = r["worst_overlap_max_ms"] / r["baseline_p50_ms"]
                print(
                    f"{r['name']:10} cold_median={r['cold_median_ms']:8.2f}ms  "
                    f"isolated_p50={r['isolated_p50_ms']:6.2f}ms  "
                    f"steady_p50={r['baseline_p50_ms']:6.2f}ms  "
                    f"warm-during-cold(median)={ratio:.1f}x  "
                    f"worst-single-call={worst_ratio:.1f}x"
                )
            do_get_r = next((r for r in results if r["name"] == "do_get"), None)
            cachefile_r = next((r for r in results if r["name"] == "cachefile"), None)
            if do_get_r and cachefile_r:
                do_get_extra_ms = (
                    do_get_r["worst_overlap_max_ms"] - do_get_r["baseline_p50_ms"]
                )
                cachefile_extra_ms = (
                    cachefile_r["worst_overlap_max_ms"] - cachefile_r["baseline_p50_ms"]
                )
                print(
                    f"\nworst-case extra latency above steady-state baseline: "
                    f"do_get +{do_get_extra_ms:.1f}ms, cachefile +{cachefile_extra_ms:.1f}ms."
                )
            print(
                "do_get's extra reaches separate processes, so a shared GIL with "
                "the cold caller cannot explain it. Its lock-held part is small: "
                "only the IPC metadata pages fault under _lock (~1 ms); the rest "
                "is CPU/GIL contention from serializing the cold chunk. "
                "cachefile's residual cannot be a lock effect at all (no server "
                "lock on that path) -- treat it as OS/disk-level contention, not "
                "perfect isolation."
            )

    finally:
        server.shutdown()
        CacheManager.reset()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
