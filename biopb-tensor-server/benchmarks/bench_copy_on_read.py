"""Benchmark: the per-hit Windows copy-off-mmap term in the file cache (#572).

Background: biopb/biopb#572. On Windows the file cache copies *every cached
chunk out of its mmap on every read hit* -- ``_read_batch_from_segment`` calls
``_copy_batch_off_mmap`` (an IPC round-trip) so a live mapping can't block
segment unlink during eviction (issue #5). POSIX keeps the read zero-copy. The
issue notes there is no Windows measurement of this term; this script provides
one.

It isolates the term the way #572 asks for: drive the exact cache-hit read path
(``ArrowFileBackend._read_batch_from_segment``) against pre-sealed, mmap-backed
segments, once with ``_copy_on_read`` forced off (the POSIX zero-copy read) and
once forced on (the Windows copy). Everything else on the path -- offset seek,
message decode, key-column strip -- is identical between the two, so the delta
is the copy alone. Forcing the flag off is unsafe in production (a held batch
would block eviction unlink on Windows) but exactly right for a measurement.

Dependencies: the biopb_tensor_server package (run in the repo venv). No server,
no client, no network -- just the backend.

Run:
    python benchmarks/bench_copy_on_read.py
    python benchmarks/bench_copy_on_read.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from biopb_tensor_server.cache.file_backend import ArrowFileBackend, ArrowFileConfig
from biopb_tensor_server.core.adapter_base import pack_chunk_batch

KIB = 1024
MIB = 1024 * 1024

# Bracket the Windows allocation cliff #572 reports (a jump at 1 MiB, then a flat
# 3.2-3.6x penalty above it) so we see whether the copy term tracks it, and hold
# the top end at MAX_ARROW_BATCH_BYTES (64 MiB) -- the worst per-hit copy.
SIZES = [
    256 * KIB,
    512 * KIB,
    1 * MIB,
    2 * MIB,
    4 * MIB,
    8 * MIB,
    16 * MIB,
    32 * MIB,
    64 * MIB,
]

# Distinct chunks per size, so a read round-robins across several sealed
# segments rather than re-hitting one hot mmap. Kept small: with max_segment
# set to one chunk, this is also the segment count, and we stay under the
# cold-mmap-release threshold so no segment is unmapped mid-run.
CHUNKS_PER_SIZE = 6


def _reps_for(nbytes: int) -> int:
    """Fewer reps for big buffers -- the effect is large and time is not free."""
    if nbytes <= 4 * MIB:
        return 120
    if nbytes <= 16 * MIB:
        return 60
    return 30


def _new_backend(cache_dir: Path, chunk_bytes: int) -> ArrowFileBackend:
    """A fresh backend whose segments seal after a single chunk.

    ``max_segment_bytes = chunk_bytes`` makes every write cross the seal
    threshold, so each chunk lands on its own sealed, mmap-readable segment --
    the state a real cache hit re-reads from. ``max_total_bytes`` is set far
    above the working set so eviction never fires and can't perturb timings.
    """
    config = ArrowFileConfig(
        cache_dir=cache_dir,
        max_segment_bytes=chunk_bytes,
        max_total_bytes=64 * (64 * MIB),  # >> working set: no eviction
    )
    return ArrowFileBackend(config)


def _seed(backend: ArrowFileBackend, nbytes: int, n_chunks: int) -> list[bytes]:
    """Write ``n_chunks`` sealed chunks of ``nbytes`` and return their keys.

    Uses the public write path (``get_or_acquire`` + ``release``), so the
    entries reach the same sealed-segment / dropped-in-RAM-mirror state a served
    chunk does. Keys that don't come back mmap-readable are dropped from the
    returned list, so the measured loop only ever hits the segment path.
    """
    # One element short of a full 1-D uint8 vector, reshaped so pack has a real
    # shape to carry. Content is irrelevant to the copy cost; a ramp avoids an
    # all-zero page the OS might treat specially.
    n = nbytes
    base = np.arange(n % 251, dtype=np.uint8)
    tile = np.resize(np.arange(251, dtype=np.uint8), n)
    _ = base  # (kept for clarity; tile is the payload)
    keys: list[bytes] = []
    for i in range(n_chunks):
        arr = tile.copy()
        arr[0] = i & 0xFF  # make each chunk distinct
        key = f"bench-{nbytes}-{i}".encode()

        def compute_fn(_arr=arr):
            return pack_chunk_batch(_arr), int(_arr.nbytes)

        entry = backend.get_or_acquire(key, compute_fn)
        # Drop our reference immediately; on a sealed+mmap segment this also
        # drops the in-RAM mirror, forcing the next read through the segment.
        backend.release(key)
        del entry

        if backend._read_batch_from_segment(key) is not None:
            keys.append(key)
    return keys


def _time_reads(
    backend: ArrowFileBackend, keys: list[bytes], copy_on_read: bool, reps: int
) -> float:
    """Median ms per ``_read_batch_from_segment`` hit at the given copy setting.

    Round-robins across ``keys`` so no single mmap stays artificially hot, and
    reports the median over ``reps`` full passes to shrug off scheduler noise.
    """
    backend._copy_on_read = copy_on_read
    # Warm: first touch faults the mapping in; we want steady-state cost.
    for key in keys:
        assert backend._read_batch_from_segment(key) is not None

    per_read: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for key in keys:
            batch = backend._read_batch_from_segment(key)
        dt = (time.perf_counter() - t0) / len(keys)
        per_read.append(dt)
        del batch
    return st.median(per_read) * 1e3


def _bench_size(nbytes: int, tmp_root: Path) -> dict:
    cache_dir = tmp_root / f"cache_{nbytes}"
    backend = _new_backend(cache_dir, nbytes)
    try:
        keys = _seed(backend, nbytes, CHUNKS_PER_SIZE)
        if not keys:
            return {"bytes": nbytes, "error": "no mmap-readable chunks seeded"}
        reps = _reps_for(nbytes)
        zero = _time_reads(backend, keys, copy_on_read=False, reps=reps)
        copy = _time_reads(backend, keys, copy_on_read=True, reps=reps)
    finally:
        backend.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    mb = nbytes / 1e6
    return {
        "bytes": nbytes,
        "mib": nbytes / MIB,
        "chunks": len(keys),
        "reps": reps,
        "zerocopy_ms": zero,
        "copy_ms": copy,
        "copy_overhead_ms": copy - zero,
        "ratio": (copy / zero) if zero else None,
        "zerocopy_mb_s": (mb / (zero / 1e3)) if zero else None,
        "copy_mb_s": (mb / (copy / 1e3)) if copy else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write raw results to this file")
    args = ap.parse_args()

    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "sys_platform": sys.platform,
    }
    print(f"platform : {env['platform']}")
    print(
        f"versions : py {env['python']}  np {env['numpy']}  pa {env['pyarrow']}"
        f"  ({env['sys_platform']})"
    )
    print(
        "\nmeasuring _read_batch_from_segment hit cost, copy_on_read off vs on\n"
        "(off = POSIX zero-copy read, on = the Windows per-hit copy this issue is about)\n"
    )
    hdr = (
        f"{'MiB':>7} | {'zerocopy':>9} | {'copy':>9} | {'overhead':>9} | "
        f"{'ratio':>6} | {'copy MB/s':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    tmp_root = Path(tempfile.mkdtemp(prefix="copy_on_read_bench_"))
    results = []
    try:
        for nbytes in SIZES:
            r = _bench_size(nbytes, tmp_root)
            results.append(r)
            if "error" in r:
                print(f"{nbytes / MIB:7.2f} | {r['error']}")
                continue
            print(
                f"{r['mib']:7.2f} | {r['zerocopy_ms']:8.3f}m | {r['copy_ms']:8.3f}m | "
                f"{r['copy_overhead_ms']:8.3f}m | {r['ratio']:5.2f}x | "
                f"{r['copy_mb_s']:9.0f}"
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    ok = [r for r in results if "ratio" in r and r["ratio"]]
    if ok:
        worst = max(ok, key=lambda r: r["ratio"])
        big = max(ok, key=lambda r: r["bytes"])
        print(
            f"\nworst ratio : {worst['ratio']:.2f}x at {worst['mib']:.2f} MiB"
            f"  (+{worst['copy_overhead_ms']:.3f} ms/hit)"
        )
        print(
            f"64 MiB copy : +{big['copy_overhead_ms']:.3f} ms/hit"
            f"  ({big['copy_mb_s']:.0f} MB/s ceiling with the copy)"
        )

    if args.json:
        args.json.write_text(json.dumps({"env": env, "results": results}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
