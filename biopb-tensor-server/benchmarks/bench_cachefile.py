"""Benchmark: localhost cache-file mmap fast path vs the do_get socket (#9).

Mirrors the measurement in issue #9 (exp_11/gt.tif, ~63 MB chunks) but for the
new cache-file handoff that replaced shm. Isolates *transport*: the server's
file cache is warmed first, then the same chunks are re-fetched with the client
cache off, so we measure only how the already-decoded, page-cache-warm bytes
reach the client.

Run:
    python benchmarks/bench_cachefile.py [path-to.tif]

Compares two client modes against an identical file-backed server:
  - cachefile : chunk_locate + mmap (default localhost fast path)
  - socket    : BIOPB_CACHEFILE_TRANSFER_DISABLED=1 -> plain do_get
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/jiyu/data/exp_11/gt.tif"
CHUNK_Z = 23  # ~63 MB per chunk for 1024x1344 uint16, matching the issue


def _load_to_zarr(tmp: str):
    import tifffile
    import zarr
    arr = np.asarray(tifffile.imread(DATA))
    arr = np.squeeze(arr)
    assert arr.ndim == 3, f"expected 3D (Z,Y,X), got {arr.shape}"
    zpath = str(Path(tmp) / "gt.zarr")
    z = zarr.open_array(
        zpath, mode="w", shape=arr.shape,
        chunks=(CHUNK_Z, arr.shape[1], arr.shape[2]), dtype=arr.dtype,
    )
    z[:] = arr
    return zpath, arr.shape, arr.dtype


def _start_server(zpath):
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig

    cache_dir = Path(tempfile.mkdtemp()) / "cache"
    CacheManager.reset()
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=str(cache_dir)))
    server = TensorFlightServer("grpc://localhost:0")
    server.register_source("gt", ZarrAdapter(zarr.open_array(zpath, mode="r"), "gt", ["z", "y", "x"]))
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1.0)
    return server


def _fresh_client(location):
    # Re-import per run so the per-location capability cache starts empty and the
    # env-var gate is re-read; client cache off to force a transport every fetch.
    import importlib

    import biopb.tensor.client as c
    importlib.reload(c)
    return c, c.TensorFlightClient(location, cache_bytes=0)


def _bench(location, disable_cachefile, shape, trials=4):
    if disable_cachefile:
        os.environ["BIOPB_CACHEFILE_TRANSFER_DISABLED"] = "1"
    else:
        os.environ.pop("BIOPB_CACHEFILE_TRANSFER_DISABLED", None)

    cmod, client = _fresh_client(location)
    darr = client.get_tensor("gt")
    Z = shape[0]
    chunk_starts = list(range(0, Z, CHUNK_Z))
    mb = (CHUNK_Z * shape[1] * shape[2] * 2) / 1e6

    # Warm the server cache once (decode + write segments).
    _ = darr.compute(scheduler="threads")

    # Per-chunk cold transport: client cache off, server cache warm.
    per_chunk = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for zs in chunk_starts:
            ze = min(zs + CHUNK_Z, Z)
            darr[zs:ze].compute(scheduler="threads")
        per_chunk.append((time.perf_counter() - t0) / len(chunk_starts))

    # Full frame-by-frame scan (280 planes; each fetches its containing chunk).
    scans = []
    for _ in range(3):
        t0 = time.perf_counter()
        for z in range(Z):
            darr[z:z + 1].compute(scheduler="threads")
        scans.append(time.perf_counter() - t0)

    used = (not disable_cachefile) and (cmod._cachefile_support.get(location) is True)
    client.close()
    pc = np.array(per_chunk)
    return {
        "per_chunk_ms": (pc.mean() * 1e3, pc.std() * 1e3, mb / pc.mean()),
        "scan_s": (np.mean(scans), np.std(scans)),
        "cachefile_used": used,
    }


def main():
    tmp = tempfile.mkdtemp()
    print(f"loading {DATA} -> zarr (chunk Z={CHUNK_Z}) ...")
    zpath, shape, dtype = _load_to_zarr(tmp)
    print(f"shape={shape} dtype={dtype}")
    server = _start_server(zpath)
    loc = f"grpc://localhost:{server.port}"
    try:
        sock = _bench(loc, disable_cachefile=True, shape=shape)
        cf = _bench(loc, disable_cachefile=False, shape=shape)
    finally:
        server.shutdown()

    for name, r in (("socket  ", sock), ("cachefile", cf)):
        m, s, bw = r["per_chunk_ms"]
        sm, ss = r["scan_s"]
        print(f"\n[{name}] cachefile_used={r['cachefile_used']}")
        print(f"  per-chunk cold : {m:7.1f} +/- {s:4.1f} ms  ({bw:6.0f} MB/s)")
        print(f"  280-frame scan : {sm:7.3f} +/- {ss:5.3f} s")

    sm, ss, _ = sock["per_chunk_ms"]
    cm, cs, _ = cf["per_chunk_ms"]
    print(f"\nper-chunk speedup (socket/cachefile): {sm / cm:.2f}x")
    print(f"scan speedup:                         {sock['scan_s'][0] / cf['scan_s'][0]:.2f}x")


if __name__ == "__main__":
    main()
