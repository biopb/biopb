"""Unit tests for the lazy chunk-fetch dask-array builder in biopb.tensor._pool.

Focus is the regular-grid path of ``_build_dask_array_from_chunk_map``: it emits
a *single* ``Blockwise`` layer, and each block's ``chunk_id``/bounds ride **per
block** via a ``BlockwiseDepDict`` rather than as one whole-array literal
broadcast to every task. That keeps a partial read's graph *size* O(blocks read)
instead of O(blocks total) -- slicing one plane out of an N-chunk tensor must not
drag all N chunk ids through the graph (biopb/biopb#278).

The leaf fetch (``_fetch_chunk_distributed``) is stubbed so these run without a
live Flight server; both the regular and fallback builders route through it.
"""

import pickle

import biopb.tensor._pool as pool
import dask
import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds


def _bounds(start, stop):
    return ChunkBounds(start=list(start), stop=list(stop))


def _regular_chunk_map(n, chunk=4):
    """A regular grid of ``n`` chunks tiled along axis 0.

    Each chunk is ``(chunk, chunk, chunk)`` and its 16-byte chunk_id encodes its
    axis-0 index in the first byte, so a computed block can be checked against
    the index it should have come from.
    """
    m = {}
    for i in range(n):
        cid = bytes([i % 256]) + b"\x00" * 15
        m[(i, 0, 0)] = (
            cid,
            _bounds((i * chunk, 0, 0), ((i + 1) * chunk, chunk, chunk)),
        )
    return m, (n, 1, 1), (n * chunk, chunk, chunk)


def _build(chunk_map, grid, shape, dtype=np.uint16):
    return pool._build_dask_array_from_chunk_map(
        chunk_map, grid, shape, dtype, "grpc://loc", None, 1 << 30, None
    )


@pytest.fixture
def stub_fetch(monkeypatch):
    """Stub the leaf fetch: return a marker-filled block and record chunk ids.

    The block is filled with ``chunk_id[0]`` (its axis-0 index for the maps
    built here), so callers can assert both *which* chunks were fetched and that
    each block landed in the right place. Patched on the ``_pool`` module, where
    both builders resolve the name at call time (so it survives an array pickle
    round-trip too).
    """
    calls = []

    def fake(
        location,
        token,
        chunk_id,
        bounds_start,
        bounds_stop,
        cache_bytes,
        schema_metadata=None,
    ):
        calls.append(chunk_id)
        shape = tuple(
            stop - start for start, stop in zip(bounds_start, bounds_stop, strict=True)
        )
        return np.full(shape, chunk_id[0], dtype=np.uint16)

    monkeypatch.setattr(pool, "_fetch_chunk_distributed", fake)
    return calls


class TestRegularGridBuilder:
    def test_empty_chunk_map_raises(self):
        with pytest.raises(ValueError, match="No chunks"):
            _build({}, (0,), (0,))

    def test_single_blockwise_layer(self):
        m, grid, shape = _regular_chunk_map(50)
        arr = _build(m, grid, shape)
        # The whole point: one layer, so graph optimization is O(1) not O(chunks).
        assert len(arr.__dask_graph__().layers) == 1

    def test_shape_dtype_and_chunks(self):
        m, grid, shape = _regular_chunk_map(6)
        arr = _build(m, grid, shape)
        assert arr.shape == (24, 4, 4)
        assert arr.dtype == np.uint16
        assert arr.chunks == ((4,) * 6, (4,), (4,))

    def test_full_compute_places_each_block_correctly(self, stub_fetch):
        n = 6
        m, grid, shape = _regular_chunk_map(n)
        arr = _build(m, grid, shape)
        out = arr.compute(scheduler="synchronous")
        for i in range(n):
            block = out[i * 4 : (i + 1) * 4]
            assert (block == i).all(), (i, np.unique(block))
        assert len(stub_fetch) == n

    def test_partial_read_fetches_only_selected_chunk(self, stub_fetch):
        m, grid, shape = _regular_chunk_map(100)
        arr = _build(m, grid, shape)
        # Read one plane that lies entirely within chunk index 3.
        plane = arr[12:16].compute(scheduler="synchronous")
        assert (plane == 3).all()
        # Culling must drop the other 99 chunks' tasks -> exactly one fetch.
        assert stub_fetch == [m[(3, 0, 0)][0]]

    def test_transpose_then_crop_is_lazy_and_correct(self, stub_fetch):
        m, grid, shape = _regular_chunk_map(6)
        arr = _build(m, grid, shape)
        # Move axis 0 last, then crop it to the span of chunk index 1.
        view = arr.transpose((2, 1, 0))[:, :, 4:8]
        assert view.shape == (4, 4, 4)
        out = view.compute(scheduler="synchronous")
        assert (out == 1).all()
        assert stub_fetch == [m[(1, 0, 0)][0]]

    def test_pickle_round_trip_preserves_and_computes(self, stub_fetch):
        m, grid, shape = _regular_chunk_map(8)
        arr = _build(m, grid, shape)
        restored = pickle.loads(pickle.dumps(arr))
        assert restored.shape == arr.shape
        assert restored.chunks == arr.chunks
        out = restored[20:24].compute(scheduler="synchronous")
        assert (out == 5).all()


class TestGraphSizeScaling:
    """Regression guard: partial-read graph size must not grow with array size."""

    @staticmethod
    def _single_plane_graph_bytes(n):
        m, grid, shape = _regular_chunk_map(n)
        arr = _build(m, grid, shape)
        (opt,) = dask.optimize(arr[0:4])  # cull to a single plane
        graph = opt.__dask_graph__()
        graph = graph.to_dict() if hasattr(graph, "to_dict") else dict(graph)
        return len(pickle.dumps(graph)), len(graph)

    def test_single_plane_graph_is_constant_size(self):
        small_bytes, small_tasks = self._single_plane_graph_bytes(8)
        large_bytes, large_tasks = self._single_plane_graph_bytes(4000)

        # Task count culls to O(1) regardless of array size.
        assert small_tasks == large_tasks

        # And so do the *bytes*: the per-block DepDict means a single-plane read
        # of a 4000-chunk array carries one chunk id, not 4000. A broadcast
        # whole-array id_map would blow this up ~100x (measured ~141 KB), so a
        # generous ceiling still catches a regression decisively.
        assert large_bytes < 4096, large_bytes
        assert large_bytes < small_bytes * 2, (small_bytes, large_bytes)


class TestRaggedFallback:
    """The ``da.block``-of-``from_delayed`` fallback for non-regular grids.

    Forced here by stubbing ``_regular_grid_chunks`` to ``None`` so the fallback
    branch is exercised on a grid ``da.block`` can assemble, independent of
    whether real server grids ever trigger it.
    """

    def test_fallback_builds_correct_array(self, stub_fetch, monkeypatch):
        monkeypatch.setattr(pool, "_regular_grid_chunks", lambda *a, **k: None)
        n = 5
        m, grid, shape = _regular_chunk_map(n)
        arr = _build(m, grid, shape)
        assert arr.shape == (20, 4, 4)
        out = arr.compute(scheduler="synchronous")
        for i in range(n):
            assert (out[i * 4 : (i + 1) * 4] == i).all()
