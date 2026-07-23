"""Locate the fresh-allocation cliff for chunk-sized buffers, per platform.

Background: biopb/biopb#571. Localhost chunk reads allocate a fresh result
buffer per read (``_array_from_unified_batch``). Above the allocator's
large-block threshold, every allocation is a fresh kernel mapping whose pages
must be faulted in and zeroed on first touch, and every free returns them --
so the cost is paid again on the next read. Buffer reuse avoids it.

On 64-bit glibc that threshold is 32 MiB (``DEFAULT_MMAP_THRESHOLD_MAX``,
``4*1024*1024*sizeof(long)``). Other allocators differ: the Windows heap sends
large blocks straight to ``VirtualAlloc`` at a far lower threshold, and macOS
libmalloc keeps a large-object cache. This script locates the boundary wherever
it happens to be, so the portability claim in #571 rests on measurement rather
than on reasoning about one libc.

Dependencies: numpy only. Run on each platform under test:

    python scripts/alloc_cliff_bench.py
    python scripts/alloc_cliff_bench.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics as st
import sys
import time

import numpy as np

KIB = 1024
MIB = 1024 * 1024

# Log-ish sweep with fine steps bracketing the candidate thresholds: the
# Windows heap's VirtualAlloc cutoff (low, sub-MiB) and glibc's 32 MiB cap.
SIZES = [
    256 * KIB,
    512 * KIB,
    1 * MIB,
    2 * MIB,
    4 * MIB,
    8 * MIB,
    16 * MIB,
    24 * MIB,
    30 * MIB,
    31 * MIB,
    32 * MIB - 4096,
    32 * MIB,
    32 * MIB + 4096,
    33 * MIB,
    36 * MIB,
    48 * MIB,
    64 * MIB,
    96 * MIB,
    128 * MIB,
]


def _reps_for(nbytes: int) -> int:
    """Fewer reps for big buffers -- the effect is large and time is not free."""
    if nbytes <= 4 * MIB:
        return 60
    if nbytes <= 32 * MIB:
        return 40
    return 20


def fresh(nbytes: int, reps: int) -> tuple[float, float]:
    """Median ms to copy into a newly allocated buffer (allocation + fault-in).

    Deliberately uses the same ``np.copyto`` as :func:`reused` so the allocation
    is the *only* difference between the two -- ``ndarray.copy()`` takes a
    different routine internally and would make the pair incomparable.
    """
    src = np.ones(nbytes, dtype=np.uint8)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        dst = np.empty(nbytes, dtype=np.uint8)
        np.copyto(dst, src)
        ts.append(time.perf_counter() - t)
        del dst
    return st.median(ts) * 1e3, min(ts) * 1e3


def reused(nbytes: int, reps: int) -> tuple[float, float]:
    """Median ms to copy into a buffer allocated once and reused."""
    src = np.ones(nbytes, dtype=np.uint8)
    dst = np.empty(nbytes, dtype=np.uint8)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        np.copyto(dst, src)
        ts.append(time.perf_counter() - t)
    return st.median(ts) * 1e3, min(ts) * 1e3


def describe_platform() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
    }
    libc, libc_ver = platform.libc_ver()
    if libc:
        info["libc"] = f"{libc} {libc_ver}".strip()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = ap.parse_args()

    info = describe_platform()
    print("=== allocation-cliff benchmark (biopb/biopb#571) ===")
    for k, v in info.items():
        print(f"  {k:14s} {v}")

    # Report min alongside median. On a memory-pressured host the kernel can
    # reclaim pages of the long-lived reused buffer between iterations, which
    # inflates its median and understates the benefit of reuse; the min is the
    # uncontended best case and is the number to compare.
    print("\n     bytes       MiB   fresh med/min      reused med/min   ratio(min)")
    rows = []
    for n in SIZES:
        reps = _reps_for(n)
        fm, fn_ = fresh(n, reps)
        rm, rn = reused(n, reps)
        ratio = fn_ / rn if rn else float("nan")
        rows.append(
            {
                "bytes": n,
                "mib": n / MIB,
                "fresh_ms": fm,
                "fresh_min_ms": fn_,
                "reused_ms": rm,
                "reused_min_ms": rn,
                "ratio": ratio,
                "ratio_median": (fm / rm if rm else float("nan")),
            }
        )
        print(
            f"{n:10d}  {n / MIB:8.2f}  {fm:7.3f} {fn_:7.3f} ms  "
            f"{rm:7.3f} {rn:7.3f} ms   {ratio:6.2f}x"
        )

    # The cliff is the largest jump in ratio between adjacent sizes.
    jumps = [
        (rows[i]["ratio"] - rows[i - 1]["ratio"], rows[i - 1], rows[i])
        for i in range(1, len(rows))
    ]
    delta, lo, hi = max(jumps, key=lambda j: j[0])
    worst = max(rows, key=lambda r: r["ratio"])
    print(
        f"\nlargest ratio jump: {hi['ratio']:.2f}x between {lo['mib']:.2f} MiB "
        f"and {hi['mib']:.2f} MiB (delta {delta:.2f})"
    )
    print(f"worst ratio overall: {worst['ratio']:.2f}x at {worst['mib']:.2f} MiB")
    if worst["ratio"] < 1.5:
        print("=> no pronounced cliff on this platform; reuse buys little here")
    else:
        print(f"=> reuse is worth up to {worst['ratio']:.1f}x on this platform")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"platform": info, "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
