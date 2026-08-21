"""Composing a scaled chunk out of the full-resolution chunks under it.

Two things have to hold or the feature is not worth having. It must produce the
*same bytes* as reading the extent and calling ``downsample_block`` -- otherwise
a chunk's value depends on a config flag -- and it must not be able to deadlock
the cache, whose promise graph it is now adding edges to.
"""

import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.cache import ArrowFileBackend
from biopb_tensor_server.cache.file_backend import ArrowFileConfig
from biopb_tensor_server.core import compose as compose_module
from biopb_tensor_server.core.adapter_base import TensorAdapter, unpack_chunk_array
from biopb_tensor_server.core.chunk import encode_chunk_id, encode_chunk_id_with_scale
from biopb_tensor_server.core.compose import (
    can_compose,
    compose_scaled_chunk,
    covering_chunks,
    is_grid_aligned,
)
from biopb_tensor_server.core.downsample import downsample_block

# ==============================================================================
# Equivalence with downsample_block
# ==============================================================================

# Geometry is where a composed reduction can differ from a whole-extent one, so
# these are chosen to break it rather than to be typical:
#  - "divides"    the easy case, scale divides the transfer extent (1024-style)
#  - "indivisible" scale does NOT divide it (1182-style), so a block straddles
#                  two chunks and each contributes part of the sum
#  - "ragged"     the tensor ends mid-block, so downsample_block's edge pad and
#                 full-block divisor have to be reproduced
#  - "corner"     ragged on both spatial axes at once, so the pad has a corner
#  - "offset"     an extent that does not start at the origin
#  - "aniso"      a different scale per axis, including a reduced C
_GEOMETRY = {
    "divides": (
        (1, 4, 1, 64, 64),
        (0, 0, 0, 0, 0),
        None,
        (1, 4, 1, 8, 8),
        (1, 1, 1, 4, 4),
    ),
    "indivisible": (
        (1, 3, 1, 60, 60),
        (0, 0, 0, 0, 0),
        None,
        (1, 3, 1, 6, 6),
        (1, 1, 1, 4, 4),
    ),
    "ragged": (
        (1, 2, 1, 30, 30),
        (0, 0, 0, 0, 0),
        None,
        (1, 2, 1, 8, 8),
        (1, 1, 1, 4, 4),
    ),
    "corner": (
        (1, 1, 1, 30, 26),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 7, 7),
        (1, 1, 1, 4, 4),
    ),
    "offset": (
        (1, 2, 1, 64, 64),
        (0, 0, 0, 16, 16),
        (1, 2, 1, 48, 48),
        (1, 2, 1, 8, 8),
        (1, 1, 1, 2, 2),
    ),
    "aniso": (
        (1, 4, 1, 32, 48),
        (0, 0, 0, 0, 0),
        None,
        (1, 4, 1, 8, 8),
        (1, 2, 1, 4, 8),
    ),
}


def _case(name):
    shape, start, stop, grid, scale = _GEOMETRY[name]
    return shape, start, (stop or shape), grid, scale


def _random(shape, dtype, seed=0):
    info = np.iinfo(np.dtype(dtype))
    rng = np.random.default_rng(seed)
    return rng.integers(info.min, info.max + 1, size=shape).astype(dtype)


def _reader(data):
    def fetch(chunk_start, chunk_stop):
        return data[
            tuple(slice(a, b) for a, b in zip(chunk_start, chunk_stop, strict=True))
        ].copy()

    return fetch


@pytest.mark.parametrize("geometry", sorted(_GEOMETRY))
@pytest.mark.parametrize("dtype", ["<u2", "|u1", "<i2"])
@pytest.mark.parametrize("method", ["area", "nearest"])
def test_composed_chunk_is_byte_identical(geometry, dtype, method):
    """Composing must not change a single value.

    The flag that turns composing on is a cache-capacity decision; it must not
    also be a decision about what the data is.
    """
    shape, start, stop, grid, scale = _case(geometry)
    data = _random(shape, dtype)

    composed = compose_scaled_chunk(
        _reader(data), start, stop, shape, grid, scale, method, dtype
    )
    direct = downsample_block(
        data[tuple(slice(a, b) for a, b in zip(start, stop, strict=True))],
        scale,
        method,
    )

    assert composed is not None
    assert composed.dtype == direct.dtype
    assert composed.shape == direct.shape
    assert np.array_equal(composed, direct)


def test_covering_chunks_tile_the_extent_exactly():
    """Every source element is fetched once -- no gap, no overlap."""
    shape, start, stop, grid, _ = _case("divides")
    seen = np.zeros(shape, dtype=np.int32)
    for chunk_start, chunk_stop in covering_chunks(start, stop, grid, shape):
        seen[
            tuple(slice(a, b) for a, b in zip(chunk_start, chunk_stop, strict=True))
        ] += 1
    covered = seen[tuple(slice(a, b) for a, b in zip(start, stop, strict=True))]
    assert covered.min() == 1 and covered.max() == 1
    assert seen.sum() == covered.size


# ==============================================================================
# What it refuses -- every one of these must fall back, not approximate
# ==============================================================================


def test_float_area_is_refused_because_staged_means_do_not_reassociate():
    """The float path rounds per axis, so chunk order would change the result."""
    assert not can_compose("<f4", (1, 1, 1, 2, 2), "area")
    assert not can_compose("<f8", (1, 1, 1, 2, 2), "area")
    # nearest picks elements; it never adds, so float is fine there.
    assert can_compose("<f4", (1, 1, 1, 2, 2), "nearest")


def test_non_dyadic_area_is_refused():
    """A scale of 3 rounds at each stage; one closing divide is not the same."""
    assert can_compose("<u2", (1, 1, 1, 4, 4), "area")
    assert not can_compose("<u2", (1, 1, 1, 3, 3), "area")


def test_precompute_is_refused():
    """ "precompute" names a native on-disk level; there is nothing to reduce."""
    assert not can_compose("<u2", (1, 1, 1, 2, 2), "precompute")


def test_misaligned_extent_is_refused():
    """An extent off the transfer grid would need chunks reaching outside it.

    Those exist and could be clipped, but they are not the chunk_ids a
    full-resolution read asks for, so caching them would not help anyone.
    """
    shape, grid = (1, 1, 1, 64, 64), (1, 1, 1, 8, 8)
    assert is_grid_aligned((0, 0, 0, 0, 0), (1, 1, 1, 32, 32), grid, shape)
    # A start off the grid.
    assert not is_grid_aligned((0, 0, 0, 4, 0), (1, 1, 1, 36, 32), grid, shape)
    # A stop off the grid that is not the tensor's own end.
    assert not is_grid_aligned((0, 0, 0, 0, 0), (1, 1, 1, 30, 32), grid, shape)
    # ... but the tensor's end IS aligned: the grid's last chunk is short.
    assert is_grid_aligned((0, 0, 0, 0, 0), (1, 1, 1, 64, 60), grid, (1, 1, 1, 64, 60))


def test_refusal_returns_none_rather_than_an_approximation():
    shape, start, stop, grid, scale = _case("divides")
    data = np.random.default_rng(0).random(shape).astype("<f4")
    assert (
        compose_scaled_chunk(
            _reader(data), start, stop, shape, grid, scale, "area", "<f4"
        )
        is None
    )


# ==============================================================================
# Through a real adapter and a real file-backed cache
# ==============================================================================


class _ArrayAdapter(TensorAdapter):
    """Minimal adapter over an in-memory array, counting its backend reads."""

    def __init__(self, data, grid):
        self._data = data
        self._grid = tuple(grid)
        self.source_id = "src"
        self._tensor_name = "t0"  # array_id is a property: source_id/tensor_name
        self.reads = []

    def get_tensor_descriptor(self):
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=list("TCZYX"),
            shape=list(self._data.shape),
            chunk_shape=list(self._grid),
            dtype=self._data.dtype.str,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        start = tuple(int(v) for v in bounds.start)
        stop = tuple(int(v) for v in bounds.stop)
        self.reads.append((start, stop))
        return self._data[
            tuple(slice(a, b) for a, b in zip(start, stop, strict=True))
        ].copy()

    @classmethod
    def create_from_config(cls, source, credentials_config=None):
        raise NotImplementedError

    def get_metadata(self) -> dict:
        return {}


class _Manager:
    """The slice of CacheManager resolve_chunk_data actually uses."""

    def __init__(self, backend, compose=False):
        self.backend = backend
        self.compose_scaled_reads = compose

    def get_or_acquire(self, key, compute_fn):
        return self.backend.get_or_acquire(key, compute_fn)

    def release(self, key):
        return self.backend.release(key)


def _composing(manager):
    """The same cache, seen as one configured to compose.

    The policy rides on the manager rather than on a ``resolve_chunk_data``
    argument, so that signature stays what the docs publish and an out-of-tree
    adapter overriding it keeps working.
    """
    return _Manager(manager.backend, compose=True)


@pytest.fixture
def cache():
    directory = tempfile.mkdtemp()
    backend = ArrowFileBackend(
        ArrowFileConfig(
            cache_dir=Path(directory),
            max_segment_bytes=4 * 1024 * 1024,
            max_total_bytes=128 * 1024 * 1024,
        )
    )
    try:
        yield _Manager(backend)
    finally:
        backend.close()
        shutil.rmtree(directory, ignore_errors=True)


def _fixture_adapter():
    shape, _, _, grid, _ = _case("divides")
    return _ArrayAdapter(_random(shape, "<u2", seed=7), grid), shape, grid


def _scaled_id(adapter, shape, scale, method="area"):
    bounds = ChunkBounds(start=[0] * len(shape), stop=list(shape))
    return encode_chunk_id_with_scale(adapter.array_id, bounds, scale, method)


def test_composing_leaves_the_full_resolution_chunks_in_the_cache(cache):
    """The whole point: a scaled read pays for the source pixels once."""
    adapter, shape, grid = _fixture_adapter()
    scale = (1, 1, 1, 4, 4)
    chunk_id = _scaled_id(adapter, shape, scale)

    adapter.resolve_chunk_data(chunk_id, _composing(cache))

    # Every full-resolution chunk under the extent is now serveable by its own
    # chunk_id, which is what a later full-resolution read will ask for.
    for chunk_start, chunk_stop in covering_chunks((0,) * 5, shape, grid, shape):
        raw = encode_chunk_id(
            adapter.array_id,
            ChunkBounds(start=list(chunk_start), stop=list(chunk_stop)),
        )
        assert cache.backend.locate_entry(raw) is not None

    # ... and serving them costs no further backend reads.
    before = len(adapter.reads)
    for chunk_start, chunk_stop in covering_chunks((0,) * 5, shape, grid, shape):
        raw = encode_chunk_id(
            adapter.array_id,
            ChunkBounds(start=list(chunk_start), stop=list(chunk_stop)),
        )
        adapter.resolve_chunk_data(raw, cache)
    assert len(adapter.reads) == before


def test_not_composing_reads_the_extent_and_keeps_none_of_it(cache):
    """The behaviour composing replaces, pinned so the difference stays visible."""
    adapter, shape, grid = _fixture_adapter()
    chunk_id = _scaled_id(adapter, shape, (1, 1, 1, 4, 4))

    adapter.resolve_chunk_data(chunk_id, cache)

    assert adapter.reads == [((0,) * 5, tuple(shape))]  # one read of the extent
    for chunk_start, chunk_stop in covering_chunks((0,) * 5, shape, grid, shape):
        raw = encode_chunk_id(
            adapter.array_id,
            ChunkBounds(start=list(chunk_start), stop=list(chunk_stop)),
        )
        assert cache.backend.locate_entry(raw) is None


def test_composed_and_direct_agree_through_the_adapter(cache):
    """Same assertion as the unit case, but over the real cache round trip."""
    adapter, shape, _ = _fixture_adapter()
    scale = (1, 1, 1, 4, 4)
    chunk_id = _scaled_id(adapter, shape, scale)

    composed = unpack_chunk_array(
        adapter.resolve_chunk_data(chunk_id, _composing(cache))
    )
    direct = downsample_block(adapter._data, scale, "area")
    assert np.array_equal(composed, direct)


def test_composing_is_off_unless_asked(cache):
    """Default off: the capacity cost is the caller's decision to make."""
    adapter, shape, grid = _fixture_adapter()
    adapter.resolve_chunk_data(_scaled_id(adapter, shape, (1, 1, 1, 4, 4)), cache)
    assert adapter.reads == [((0,) * 5, tuple(shape))]


def test_memory_backend_does_not_compose():
    """Raw chunks are only cached on ArrowFileBackend.

    Composing against a backend that drops them fetches the same bytes and keeps
    none of them -- all of the cost, none of the point.
    """
    from biopb_tensor_server.cache.memory_backend import (
        MemoryCacheBackend,
        MemoryCacheConfig,
    )

    adapter, shape, _ = _fixture_adapter()
    manager = _Manager(MemoryCacheBackend(MemoryCacheConfig()))
    adapter.resolve_chunk_data(
        _scaled_id(adapter, shape, (1, 1, 1, 4, 4)), _composing(manager)
    )
    assert adapter.reads == [((0,) * 5, tuple(shape))]


def test_composed_ids_are_the_ids_a_full_resolution_plan_asks_for(cache):
    """The invariant the whole feature rests on: one transfer grid per tensor.

    Composing is only worth anything because the chunks it leaves behind are the
    ones a later full-resolution read looks up. That holds because the composer
    and ``get_read_plan`` both take the grid from ``get_transfer_chunk_size``.
    Nothing else enforces it, and a second opinion about the grid anywhere in
    the server turns the cache fill into dead weight -- written, never read.
    (biopb/biopb#812 is that bug in its other form: the catalog publishing a
    grid no read uses.)
    """
    adapter, shape, grid = _fixture_adapter()
    adapter.resolve_chunk_data(
        _scaled_id(adapter, shape, (1, 1, 1, 4, 4)), _composing(cache)
    )

    request = TensorDescriptor(
        array_id=adapter.array_id,
        dim_labels=list("TCZYX"),
        shape=list(shape),
        dtype=adapter._data.dtype.str,
    )
    planned = {
        endpoint.chunk_id for endpoint in adapter.get_read_plan(request).chunk_endpoints
    }
    assert planned, "a full-resolution plan should have endpoints"
    for chunk_id in planned:
        assert cache.backend.locate_entry(chunk_id) is not None, (
            "a full-resolution read would miss on a chunk composing just wrote"
        )


# ==============================================================================
# Re-entrancy
# ==============================================================================


def test_compose_refuses_inside_a_composition(cache):
    """The acyclicity guard, asserted rather than assumed.

    Composition adds scaled -> raw edges to the cache's promise graph. Raw keys
    wait on nothing, so the graph stays bipartite and cannot cycle -- but only
    while nothing composes from a composed chunk. This is that invariant, held
    per-thread because the inner fetches run inline on the calling thread.
    """
    adapter, shape, _ = _fixture_adapter()
    args = (
        _scaled_id(adapter, shape, (1, 1, 1, 4, 4)),
        ChunkBounds(start=[0] * 5, stop=list(shape)),
        (1, 1, 1, 4, 4),
        "area",
        _composing(cache),
    )
    with compose_module.descending():
        assert adapter._compose_scaled_chunk(*args) is None
    # ... and outside that scope the same call composes.
    assert adapter._compose_scaled_chunk(*args) is not None


def test_a_thread_can_opt_out(cache):
    """``without_composition`` is how precache declines, at its own call site."""
    adapter, shape, grid = _fixture_adapter()
    with compose_module.without_composition():
        adapter.resolve_chunk_data(
            _scaled_id(adapter, shape, (1, 1, 1, 4, 4)), _composing(cache)
        )
    assert adapter.reads == [((0,) * 5, tuple(shape))]  # one read of the extent


def test_the_published_signature_still_works(cache):
    """An adapter written against the documented two-argument form must serve.

    ``resolve_chunk_data(chunk_id, cache_manager)`` is quoted in
    docs/remote-tensor-cache.md and docs/volume-rendering.md and listed in
    ``_TENSOR_SCOPED_API``. Composing must not have widened it: an out-of-tree
    adapter overriding it as documented would raise TypeError on the first
    do_get, which is a server-side 500 for every read of that source.
    """
    import inspect

    from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from biopb_tensor_server.core.normalize import NormalizingAdapter

    expected = ["self", "chunk_id", "cache_manager"]
    for klass in (
        TensorAdapter,
        CachedSourceAdapter,
        RemoteTensorAdapter,
        NormalizingAdapter,
    ):
        parameters = list(inspect.signature(klass.resolve_chunk_data).parameters)
        assert parameters == expected, f"{klass.__name__} widened the contract"


def test_neither_cache_lock_is_held_while_a_chunk_is_computed(cache):
    """``compute_fn`` must run outside ``_lock`` and ``_write_lock``.

    Both are plain, non-reentrant ``threading.Lock``. Composing calls back into
    ``get_or_acquire`` from inside a ``compute_fn``, so if either lock were ever
    held across that call the server would deadlock against itself rather than
    slow down. Nothing else pins that, and the two are only a few lines apart.
    """
    backend = cache.backend
    observed = {}

    def compute_fn():
        observed["lock"] = backend._lock.acquire(blocking=False)
        if observed["lock"]:
            backend._lock.release()
        observed["write_lock"] = backend._write_lock.acquire(blocking=False)
        if observed["write_lock"]:
            backend._write_lock.release()
        arr = np.zeros((2, 2), dtype=np.uint16)
        return pa.RecordBatch.from_arrays([pa.array([1])], names=["x"]), arr.nbytes

    try:
        backend.get_or_acquire(b"probe", compute_fn)
    except Exception:  # a schema mismatch on commit is fine; the probe ran
        pass
    assert observed == {"lock": True, "write_lock": True}


def test_concurrent_composers_of_the_same_chunk_all_complete(cache):
    """One computes, the rest wait on its promise; none deadlock or diverge."""
    adapter, shape, _ = _fixture_adapter()
    chunk_id = _scaled_id(adapter, shape, (1, 1, 1, 4, 4))
    results, errors = [], []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        try:
            results.append(
                unpack_chunk_array(
                    adapter.resolve_chunk_data(chunk_id, _composing(cache))
                )
            )
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not any(thread.is_alive() for thread in threads), "composer deadlocked"
    assert not errors, errors
    assert len(results) == 4
    for result in results[1:]:
        assert np.array_equal(result, results[0])
