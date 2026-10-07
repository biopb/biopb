"""Benchmark an evicted OME-TIFF read with and without a kept handle (#1284).

Run from the repository root:

    uv run --no-sync python \
      biopb-tensor-server/benchmarks/bench_evicted_read.py FILE [CYCLES]

An evicted source is rebuilt from its catalog row. ``A`` reopens the file on the
first read, as an adapter does today. ``B`` moves the previous scene adapter's
open tifffile handle, aszarr store and zarr array onto the rebuilt one, which is
what a handle pool keyed by the file's ``(ino, size, mtime_ns, ctime_ns)`` would
hand out. ``C`` is a resident adapter's read, the floor for ``B``. The pooled
bytes are checked against the first read. Warm page cache; use a large file,
where the reopen (page-tag parse) dominates.
"""

import gc
import os
import statistics
import sys
import time

import numpy as np
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import OmeTiffAdapter
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import SourceClaim
from biopb_tensor_server.serving.metadata_db import CatalogRecord, MetadataDatabase

HANDLE_ATTRS = (
    "_persistent_zarr",
    "_persistent_axes",
    "_persistent_store",
    "_persistent_tiff",
)


def fds():
    return len(os.listdir("/proc/self/fd"))


def read_small(scene):
    shape = list(scene.get_tensor_descriptor().shape)
    stop = [min(s, 64) for s in shape]
    return np.asarray(scene.get_data(ChunkBounds(start=[0] * len(shape), stop=stop)))


def take_handle(scene):
    """Detach the open handle from *scene*, so its finalizer will not close it."""
    handle = {a: getattr(scene, a) for a in HANDLE_ATTRS}
    for a in HANDLE_ATTRS:
        setattr(scene, a, None)
    scene._persistent_attempted = False
    return handle


def give_handle(scene, handle):
    for a, v in handle.items():
        setattr(scene, a, v)
    scene._persistent_attempted = True
    scene._persistent_last_access = time.monotonic()


def stat_signature(path):
    st = os.stat(path)
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def summary(label, xs):
    xs = sorted(xs)
    p95 = xs[max(int(len(xs) * 0.95) - 1, 0)]
    print(
        f"{label:34s} {statistics.median(xs) * 1e3:8.2f} ms median {p95 * 1e3:8.2f} p95"
    )


def main(path, cycles):
    config = SourceConfig(url=path, source_id="s0")
    adapter = OmeTiffAdapter.create_from_config(config)
    array_id = adapter.list_tensor_descriptors()[0].array_id
    db = MetadataDatabase()
    db.sync_roots([("r", "file:///")])
    claim = SourceClaim(adapter.source_type, path, "src", member_paths=[path])
    db.sync_source_added(
        "src", adapter, CatalogRecord(claim, {path: (1, 2, 3, 4)}, "r", "x")
    )
    resident = adapter.get_tensor_adapter(array_id)
    reference = read_small(resident)
    print(
        f"{os.path.getsize(path) / 1e9:.2f} GB, read {reference.shape} {reference.dtype}"
    )

    evicted = []
    for _ in range(cycles):
        t = time.perf_counter()
        payload, metadata = db.read_hydration("src")
        rebuilt = OmeTiffAdapter.create_from_payload(config, payload, metadata, None)
        out = read_small(rebuilt.get_tensor_adapter(array_id))
        evicted.append(time.perf_counter() - t)
        assert np.array_equal(out, reference)
        del rebuilt
        gc.collect()
    summary("A evicted, handle reopened", evicted)

    pool = take_handle(resident)
    signature = stat_signature(path)
    fds_before = fds()
    pooled = []
    for _ in range(cycles):
        t = time.perf_counter()
        payload, metadata = db.read_hydration("src")
        rebuilt = OmeTiffAdapter.create_from_payload(config, payload, metadata, None)
        scene = rebuilt.get_tensor_adapter(array_id)
        assert stat_signature(path) == signature
        give_handle(scene, pool)
        out = read_small(scene)
        pooled.append(time.perf_counter() - t)
        assert np.array_equal(out, reference)
        pool = take_handle(scene)
        del scene, rebuilt
        gc.collect()
    summary("B evicted, handle pooled", pooled)
    print(f"descriptors over {cycles} pooled cycles: {fds_before} -> {fds()}")

    give_handle(resident, pool)
    floor = []
    for _ in range(cycles):
        t = time.perf_counter()
        read_small(resident)
        floor.append(time.perf_counter() - t)
    summary("C resident adapter", floor)
    db.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
