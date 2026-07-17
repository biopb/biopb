"""Vectorized compact-grid reconstruction (biopb/biopb#346, PR-D).

The compact read path builds the dask array straight from the descriptor with
numpy (``compact_grid_arrays`` + a numpy-backed ``GridChunkDep``) instead of the
per-chunk Python loop over ``ChunkBounds`` + ``chunk_map`` dict. These tests pin:

- ``compact_grid_arrays`` is byte-identical to the reference ``expand_compact_grid``
  (same chunk_ids and logical bounds, same order) across plain / sliced / scaled /
  precompute / 3-D grids -- so the fast path can never diverge from the list path;
- ``GridChunkDep`` indexes correctly and pickles (it rides to distributed workers);
- over a real Flight socket, a full read, a culled single-plane read, and a
  multi-process distributed read all return byte-identical data, and a culled read
  touches only the blocks it reads (O(blocks read), not O(blocks total)).
"""

import threading
import time

import numpy as np
import pytest
from biopb.tensor._chunk_codec import compact_grid_arrays, expand_compact_grid
from biopb.tensor._pool import GridChunkDep
from biopb.tensor.descriptor_pb2 import TensorDescriptor


def _desc(shape, chunk, rstart, rstop, scale=None, method="", chunk_aid="s"):
    d = TensorDescriptor()
    d.array_id = "s"
    d.chunk_array_id = chunk_aid
    d.dtype = "uint16"
    d.shape.extend(shape)
    d.chunk_shape.extend(chunk)
    d.slice_hint.start.extend(rstart)
    d.slice_hint.stop.extend(rstop)
    if scale:
        d.scale_hint.extend(scale)
        d.reduction_method = method
    return d


_CASES = {
    "plain": _desc([200, 150], [64, 64], [0, 0], [200, 150]),
    "sliced": _desc([64, 150], [64, 64], [64, 0], [128, 150]),
    "scaled": _desc(
        [100, 75], [32, 32], [0, 0], [200, 150], scale=[2, 2], method="mean"
    ),
    "precompute": _desc(
        [50, 50],
        [32, 32],
        [0, 0],
        [100, 100],
        scale=[2, 2],
        method="max",
        chunk_aid="s/1",
    ),
    "3d": _desc([40, 64, 64], [1, 64, 64], [0, 0, 0], [40, 64, 64]),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_vectorized_matches_reference(name):
    """compact_grid_arrays reproduces expand_compact_grid exactly."""
    d = _CASES[name]
    ref = expand_compact_grid(d)
    ref_ids = [cid for cid, _ in ref]
    ref_lstart = np.array([[int(x) for x in b.start] for _, b in ref])
    ref_lstop = np.array([[int(x) for x in b.stop] for _, b in ref])

    g = compact_grid_arrays(d)
    assert list(g.chunk_ids) == ref_ids  # byte-for-byte
    assert np.array_equal(g.lstarts, ref_lstart)
    assert np.array_equal(g.lstops, ref_lstop)
    # grid_chunks tiles the realized (post-downsample) extent
    for a in range(len(d.shape)):
        ext = d.slice_hint.stop[a] - d.slice_hint.start[a]
        if d.scale_hint:
            ext = -(-ext // d.scale_hint[a])
        assert sum(g.grid_chunks[a]) == ext


def test_gridchunkdep_indexing_and_pickle():
    import pickle

    g = compact_grid_arrays(_CASES["plain"])
    numblocks = tuple(len(c) for c in g.grid_chunks)
    dep = GridChunkDep(g.chunk_ids, g.lstarts, g.lstops, numblocks)

    # __getitem__ gathers the C-order row for each block index.
    for lin, (i, j) in enumerate(
        (i, j) for i in range(numblocks[0]) for j in range(numblocks[1])
    ):
        cid, start, stop = dep[(i, j)]
        assert cid == g.chunk_ids[lin]
        assert np.array_equal(start, g.lstarts[lin])
        assert np.array_equal(stop, g.lstops[lin])

    dep2 = pickle.loads(pickle.dumps(dep))
    assert dep2[(1, 1)][0] == dep[(1, 1)][0]


def _serve(source_id, adapter):
    from biopb_tensor_server import TensorFlightServer

    server = TensorFlightServer("grpc://localhost:0")
    server.register_source(source_id, adapter)
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1)
    return server


def _chunked_zarr(tmp_path):
    import zarr

    path = str(tmp_path / "a.zarr")
    za = zarr.open_array(
        path, mode="w", shape=(200, 150), chunks=(64, 64), dtype="uint16"
    )
    for i in range(4):
        for j in range(3):
            za[i * 64 : min((i + 1) * 64, 200), j * 64 : min((j + 1) * 64, 150)] = (
                i * 10 + j + 1
            )
    return path, zarr.open_array(path, mode="r")[:]


class TestVectorizedRealSocket:
    def test_full_read_correct(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server import ZarrAdapter

        path, expected = _chunked_zarr(tmp_path)
        server = _serve(
            "s", ZarrAdapter(zarr.open_array(path, mode="r"), "s", ["y", "x"])
        )
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=0
            )
            out = client.get_tensor("s").compute()
            assert np.array_equal(out, expected)
            client.close()
        finally:
            server.shutdown()

    def test_culled_read_touches_only_blocks_read(self, monkeypatch, tmp_path):
        """A single-block slice must index only the blocks it reads -- the whole
        point of the numpy-backed dep is O(blocks read), not O(blocks total)."""
        zarr = pytest.importorskip("zarr")
        from biopb.tensor import _pool
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server import ZarrAdapter

        seen = []
        real_getitem = _pool.GridChunkDep.__getitem__
        monkeypatch.setattr(
            _pool.GridChunkDep,
            "__getitem__",
            lambda self, idx: (seen.append(idx), real_getitem(self, idx))[1],
        )

        path, expected = _chunked_zarr(tmp_path)
        server = _serve(
            "s", ZarrAdapter(zarr.open_array(path, mode="r"), "s", ["y", "x"])
        )
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=0
            )
            arr = client.get_tensor("s")  # 4x3 = 12 blocks
            seen.clear()
            out = arr[0:10, 0:10].compute()  # lands in exactly one block
            assert np.array_equal(out, expected[0:10, 0:10])
            assert set(seen) == {(0, 0)}, (
                f"culled read indexed {set(seen)}, want just (0,0)"
            )
            client.close()
        finally:
            server.shutdown()

    def test_distributed_multiprocess_read(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        pytest.importorskip("distributed")
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server import ZarrAdapter
        from distributed import Client, LocalCluster

        path, expected = _chunked_zarr(tmp_path)
        server = _serve(
            "s", ZarrAdapter(zarr.open_array(path, mode="r"), "s", ["y", "x"])
        )
        cluster = LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
        )
        dclient = Client(cluster)
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=0
            )
            arr = client.get_tensor("s")
            assert np.array_equal(arr.compute(), expected)
            assert np.array_equal(
                arr[10:150, 20:140].compute(), expected[10:150, 20:140]
            )
            client.close()
        finally:
            dclient.close()
            cluster.close()
            server.shutdown()
