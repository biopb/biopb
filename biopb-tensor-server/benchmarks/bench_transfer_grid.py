"""Benchmark: transfer-grid sizing vs the adapter's native read grid (#684, #793).

#793 stopped serving the adapter's file/dask block geometry as the public Flight
transfer grid and coalesces small native blocks toward a preferred byte target
instead. That trades bulk-read throughput against partial-read cost: bounds snap
outward to the transfer grid, so a wider grid makes every partial read wider too.
This measures both sides of that trade, including the **native** grid (pre-#793
behaviour), which is the baseline a coalesced-vs-coalesced comparison cannot show.

Run:
    python benchmarks/bench_transfer_grid.py [sweep|transport] [rounds]

Modes:
  sweep     - native / 8 / 16 / 32 MiB targets; one-plane and full-stack timings.
              Variants are interleaved round-robin so thermal and page-cache
              drift hits all of them equally instead of accumulating against
              whichever ran last.
  transport - which transport each read actually uses, and what a coalesced
              chunk costs on each. Reports cold, server-warm/client-cold
              (chunk_locate + mmap) and client-warm (in-process cache).

              The server-warm/client-cold column decides whether mmap over-fetch
              is *lazy* -- map N planes, touch one, and the other N-1 never
              fault in. If it is, over-read on the fast path is ~free and the
              transfer grid can be sized for the cold case alone. A column that
              grows with chunk size says otherwise.

              What the *cold* column measures depends on the server's cache
              backend, so read it together with the reported hits/attempts.
              Against the file backend the server must decode and write the
              whole segment before any of it can be located, so a cold read
              registers an mmap hit but still pays full chunk production; with
              no segment to locate (e.g. a memory backend) the client falls back
              to do_get instead. Either way the cost scales with chunk size --
              lazy faulting saves the *transport*, not the production.
"""

import gc
import importlib
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# The #684 geometry: a 320-plane stack whose adapter advertises one plane per
# block, which is what produced 320 endpoints before #793.
SHAPE = (1, 1, 320, 960, 1000)
NATIVE_CHUNKS = (1, 1, 1, 960, 1000)
LABELS = ["t", "c", "z", "y", "x"]

MIB = 1024 * 1024
VARIANTS = [
    ("native (pre-#793)", "native"),
    ("8 MiB (#793)", 8 * MIB),
    ("16 MiB", 16 * MIB),
    ("32 MiB", 32 * MIB),
]

# Cache-file fast-path attempts/hits, reset immediately before each timed read.
PATH = {"try": 0, "hit": 0}


def _make_store(tmpdir: str) -> str:
    """Sparse zarr at the issue geometry.

    Sparse keeps setup cheap, but it also means over-reading costs no decode and
    no real IO -- a bias that favours coalescing. Point ``BIOPB_BENCH_ZARR`` at a
    dense or compressed store to price the over-read honestly.
    """
    override = os.environ.get("BIOPB_BENCH_ZARR")
    if override:
        return override

    import zarr

    path = str(Path(tmpdir) / "stack.zarr")
    arr = zarr.open_array(
        path, mode="w", shape=SHAPE, chunks=NATIVE_CHUNKS, dtype="uint16"
    )
    arr[0, 0, 0] = 7
    return path


def _set_grid_policy(mode) -> None:
    """Force the server's transfer-grid choice.

    Adapters reach the sizing policy through ``default_transfer_chunk_shape``,
    which resolves ``compute_transfer_chunk_size`` as a module global of
    ``core.chunk`` -- so patching it there is enough, and reaches every adapter
    that did not declare a grid of its own (biopb/biopb#809).
    """
    import biopb_tensor_server.core.chunk as chunk

    real = getattr(chunk, "_real_compute_transfer_chunk_size", None)
    if real is None:
        real = chunk.compute_transfer_chunk_size
        chunk._real_compute_transfer_chunk_size = real

    if mode == "native":
        # Pre-#793: the advertised block was served verbatim unless it exceeded
        # the Arrow ceiling, which a one-plane block never does.
        def policy(native, shape, dtype, labels, **kwargs):
            return tuple(
                min(int(c), int(s)) for c, s in zip(native, shape, strict=True)
            )
    else:

        def policy(native, shape, dtype, labels, **kwargs):
            return real(native, shape, dtype, labels, preferred_bytes=mode)

    chunk.compute_transfer_chunk_size = policy


def _start_server(zarr_path: str):
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.core.config import CacheConfig

    cache_dir = Path(tempfile.mkdtemp()) / "cache"
    CacheManager.reset()
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=str(cache_dir)))

    server = TensorFlightServer("grpc://localhost:0")
    server.register_source(
        "stack", ZarrAdapter(zarr.open_array(zarr_path, mode="r"), "stack", LABELS)
    )
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1.0)
    return server


def _fresh_client(location: str, cache_bytes: int = 64_000_000):
    """A client with no inherited chunk cache.

    ``_pool``'s caches and connection pool are module-global keyed by
    ``(location, token)``, so ``client.close()`` does not drop them and a new
    client re-reads straight out of the previous one's cache. Reloading the
    module is what actually makes the next read cold. Re-instates the fast-path
    counter, which the reload discards.
    """
    import biopb.tensor._pool as pool

    importlib.reload(pool)
    import biopb.tensor.client as client_mod

    importlib.reload(client_mod)

    original = pool._try_cachefile_transfer

    def counting(*args, **kwargs):
        PATH["try"] += 1
        result = original(*args, **kwargs)
        if result is not None:
            PATH["hit"] += 1
        return result

    pool._try_cachefile_transfer = counting
    return pool, client_mod.TensorFlightClient(location, cache_bytes=cache_bytes)


def _grid_of(darr):
    return tuple(int(x) for x in darr.chunksize)


def _endpoints(grid):
    return int(np.prod([int(np.ceil(s / c)) for s, c in zip(SHAPE, grid, strict=True)]))


def _time_ms(fn) -> float:
    gc.collect()
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def _fmt(xs) -> str:
    return f"{statistics.median(xs):.2f} [{min(xs):.2f}-{max(xs):.2f}]"


def sweep(zarr_path: str, rounds: int) -> None:
    acc = {name: {"plane": [], "full": [], "grid": None} for name, _ in VARIANTS}

    for rnd in range(rounds):
        for name, mode in VARIANTS:
            _set_grid_policy(mode)
            server = _start_server(zarr_path)
            location = f"grpc://localhost:{server.port}"
            try:
                _, client = _fresh_client(location)
                darr = client.get_tensor("stack")
                acc[name]["grid"] = _grid_of(darr)
                darr[0, 0, 8].compute()  # warm server/JIT on an unrelated plane
                client.close()

                # A plane the warm-up did not touch, so its chunk is cold on the
                # server as well as on the client.
                plane = 160 + rnd
                _, client = _fresh_client(location)
                darr = client.get_tensor("stack")
                acc[name]["plane"].append(
                    _time_ms(lambda: darr[0, 0, plane].compute())  # noqa: B023
                )
                client.close()

                _, client = _fresh_client(location)
                darr = client.get_tensor("stack")
                acc[name]["full"].append(_time_ms(darr.compute) / 1e3)
                client.close()
            finally:
                server.shutdown()
                time.sleep(0.3)
        print(f"round {rnd + 1}/{rounds}", flush=True)

    print()
    print(
        f"{'variant':>18} {'transfer grid':>22} {'endpoints':>10} "
        f"{'1 plane ms med[min-max]':>24} {'full s med[min-max]':>22}"
    )
    for name, _ in VARIANTS:
        v = acc[name]
        print(
            f"{name:>18} {str(v['grid']):>22} {_endpoints(v['grid']):>10} "
            f"{_fmt(v['plane']):>24} {_fmt(v['full']):>22}"
        )


def transport(zarr_path: str, rounds: int) -> None:
    acc = {
        name: {"cold": [], "mmap": [], "warm": [], "grid": None, "path": ""}
        for name, _ in VARIANTS
    }

    for rnd in range(rounds):
        plane = 160 + rnd
        for name, mode in VARIANTS:
            _set_grid_policy(mode)
            server = _start_server(zarr_path)
            location = f"grpc://localhost:{server.port}"
            try:
                _, client = _fresh_client(location)
                darr = client.get_tensor("stack")
                acc[name]["grid"] = _grid_of(darr)
                darr[0, 0, 8].compute()
                client.close()

                # Cold: nothing cached anywhere for this plane's chunk.
                _, client = _fresh_client(location)
                darr = client.get_tensor("stack")
                PATH["try"] = PATH["hit"] = 0
                acc[name]["cold"].append(
                    _time_ms(lambda: darr[0, 0, plane].compute())  # noqa: B023
                )
                cold_path = (PATH["try"], PATH["hit"])
                client.close()

                # Server-warm, client-cold: the segment now exists on disk, so
                # chunk_locate succeeds and the client maps it.
                _, client = _fresh_client(location)
                darr = client.get_tensor("stack")
                PATH["try"] = PATH["hit"] = 0
                acc[name]["mmap"].append(
                    _time_ms(lambda: darr[0, 0, plane].compute())  # noqa: B023
                )
                mmap_path = (PATH["try"], PATH["hit"])

                # Client-warm: same client, served from the in-process cache.
                acc[name]["warm"].append(
                    _time_ms(lambda: darr[0, 0, plane].compute())  # noqa: B023
                )
                client.close()

                acc[name]["path"] = (
                    f"cold {cold_path[1]}/{cold_path[0]}, "
                    f"warm {mmap_path[1]}/{mmap_path[0]}"
                )
            finally:
                server.shutdown()
                time.sleep(0.3)
        print(f"round {rnd + 1}/{rounds}", flush=True)

    print()
    print(
        "mmap hits/attempts per variant; a cold read that misses fell back to do_get."
    )
    print("'cold' = first touch, incl. server-side chunk production (see docstring).")
    print()
    print(
        f"{'variant':>18} {'transfer grid':>22} {'cold ms':>22} "
        f"{'server-warm mmap ms':>22} {'client-warm ms':>20} {'mmap hits/tries':>20}"
    )
    for name, _ in VARIANTS:
        v = acc[name]
        print(
            f"{name:>18} {str(v['grid']):>22} {_fmt(v['cold']):>22} "
            f"{_fmt(v['mmap']):>22} {_fmt(v['warm']):>20} {v['path']:>20}"
        )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if mode not in ("sweep", "transport"):
        sys.exit("usage: bench_transfer_grid.py [sweep|transport] [rounds]")

    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = _make_store(tmp)
        print(f"geometry {SHAPE}, native blocks {NATIVE_CHUNKS}")
        (sweep if mode == "sweep" else transport)(zarr_path, rounds)


if __name__ == "__main__":
    main()
