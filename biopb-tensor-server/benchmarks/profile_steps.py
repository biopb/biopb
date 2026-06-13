"""Per-step profile of a single ~63 MB warm chunk read: cache-file vs socket.

Warms the server's file cache, captures real chunk_ids, then times each
sub-step of both read paths in isolation (warm page cache, no client cache).
"""

import json
import statistics as st
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

import biopb.tensor.client as cmod
from biopb.tensor.client import (
    TensorFlightClient,
    _array_from_unified_batch,
    _get_worker_resources,
)
from biopb.tensor.ticket_pb2 import TensorTicket

CHUNK_Z = 23  # ~63 MB for 1024x1344 uint16


def _setup():
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig

    tmp = tempfile.mkdtemp()
    shape = (CHUNK_Z * 4, 1024, 1344)
    arr = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 997).reshape(shape).astype(np.uint16)
    zpath = str(Path(tmp) / "d.zarr")
    z = zarr.open_array(zpath, mode="w", shape=shape, chunks=(CHUNK_Z, shape[1], shape[2]), dtype=arr.dtype)
    z[:] = arr
    CacheManager.reset()
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=str(Path(tmp) / "cache")))
    server = TensorFlightServer("grpc://localhost:0")
    server.register_source("d", ZarrAdapter(zarr.open_array(zpath, mode="r"), "d", ["z", "y", "x"]))
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1.0)
    return server, shape


def _capture_chunk_ids(location, shape):
    """Warm server cache + grab real chunk_ids by spying on the fetch fn."""
    captured = []
    orig = cmod._fetch_chunk_distributed

    def spy(loc, token, chunk_id, *a, **k):
        captured.append(chunk_id)
        return orig(loc, token, chunk_id, *a, **k)

    cmod._fetch_chunk_distributed = spy
    try:
        client = TensorFlightClient(location, cache_bytes=0)
        client.get_tensor("d").compute(scheduler="threads")
        client.close()
    finally:
        cmod._fetch_chunk_distributed = orig
    # de-dup preserving order
    seen, ids = set(), []
    for cid in captured:
        if cid not in seen:
            seen.add(cid); ids.append(cid)
    return ids


def _t(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.mean(ts), st.stdev(ts) if len(ts) > 1 else 0.0


def main():
    server, shape = _setup()
    loc = f"grpc://localhost:{server.port}"
    try:
        ids = _capture_chunk_ids(loc, shape)
        cid = ids[0]
        client, _, opts = _get_worker_resources(loc, None, 0)
        N = 30
        mb = CHUNK_Z * shape[1] * shape[2] * 2 / 1e6
        print(f"chunk ~{mb:.0f} MB, {N} trials each\n")

        # ---- cache-file sub-steps ----
        def locate():
            a = flight.Action("chunk_locate", TensorTicket(chunk_id=cid).SerializeToString())
            return json.loads(next(client.do_action(a, options=opts)).body.to_pybytes())

        info = locate()

        def map_open():
            mm = pa.memory_map(info["segment_path"], "r"); mm.close()

        def full_read():
            mm = pa.memory_map(info["segment_path"], "r")
            try:
                schema = pa.ipc.open_stream(mm).schema
                mm.seek(info["byte_offset"])
                batch = pa.ipc.read_record_batch(pa.ipc.read_message(mm), schema)
                _array_from_unified_batch(batch)
            finally:
                mm.close()

        def read_no_copy():
            mm = pa.memory_map(info["segment_path"], "r")
            try:
                schema = pa.ipc.open_stream(mm).schema
                mm.seek(info["byte_offset"])
                batch = pa.ipc.read_record_batch(pa.ipc.read_message(mm), schema)
                _ = batch.column("data").buffers()[2]  # view only, no .copy()
            finally:
                mm.close()

        l_m, l_s = _t(locate, N)
        mo_m, mo_s = _t(map_open, N)
        rn_m, rn_s = _t(read_no_copy, N)
        fr_m, fr_s = _t(full_read, N)
        print("CACHE-FILE per chunk:")
        print(f"  chunk_locate RPC        : {l_m:7.2f} +/- {l_s:5.2f} ms")
        print(f"  mmap open+close         : {mo_m:7.2f} +/- {mo_s:5.2f} ms")
        print(f"  read view (no copy)     : {rn_m:7.2f} +/- {rn_s:5.2f} ms")
        print(f"  read + .copy() (full)   : {fr_m:7.2f} +/- {fr_s:5.2f} ms")
        print(f"  => copy() alone ~       : {fr_m - rn_m:7.2f} ms")
        print(f"  TOTAL (locate+full)     : {l_m + fr_m:7.2f} ms\n")

        # ---- socket sub-steps ----
        def do_get_readall():
            r = client.do_get(flight.Ticket(TensorTicket(chunk_id=cid).SerializeToString()), options=opts)
            return r.read_all()

        tbl = do_get_readall()

        def materialize():
            t = do_get_readall()
            a = t.column("data").to_numpy()[0]
            a.reshape(tuple(t.column("shape").to_pylist()[0]))

        dg_m, dg_s = _t(do_get_readall, N)
        full_m, full_s = _t(materialize, N)
        print("SOCKET per chunk:")
        print(f"  do_get + read_all       : {dg_m:7.2f} +/- {dg_s:5.2f} ms")
        print(f"  + to_numpy materialize  : {full_m:7.2f} +/- {full_s:5.2f} ms (total)")
        print(f"  => materialize alone ~  : {full_m - dg_m:7.2f} ms\n")

        print(f"end-to-end: cachefile {l_m + fr_m:.1f} ms  vs  socket {full_m:.1f} ms"
              f"  -> {full_m / (l_m + fr_m):.2f}x")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
