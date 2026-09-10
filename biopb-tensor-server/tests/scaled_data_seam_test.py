"""The get_scaled_data seam (biopb/biopb#640 phase 1).

`resolve_chunk_data` used to read the whole extent and reduce it inline. It now
goes through one overridable call, so an adapter whose reader can deliver the
extent in pieces can fold each piece as it arrives and never hold the
full-resolution extent. The default is exactly the old inline behavior, which is
what makes the seam land without touching any adapter.
"""

import tempfile

import numpy as np
import pytest
import zarr
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core import cache_source as _cs, downsample as _ds
from biopb_tensor_server.core.adapter_base import _TENSOR_SCOPED_API, TensorAdapter
from biopb_tensor_server.core.chunk import (
    cache_key_for_chunk_id,
    encode_chunk_id_with_scale,
)
from biopb_tensor_server.core.chunk_batch import unpack_chunk_array
from biopb_tensor_server.core.config import CacheConfig


def _bounds(start, stop):
    return ChunkBounds(start=list(start), stop=list(stop))


@pytest.fixture
def adapter():
    with tempfile.TemporaryDirectory() as tmp:
        src = (np.arange(64 * 64, dtype=np.uint16) % 4093).reshape(64, 64)
        arr = zarr.open_array(
            f"{tmp}/a.zarr", mode="w", shape=(64, 64), chunks=(32, 32), dtype="uint16"
        )
        arr[:] = src
        yield ZarrAdapter(zarr.open_array(f"{tmp}/a.zarr", mode="r"), "src", ["y", "x"])


@pytest.fixture
def counted(adapter, monkeypatch):
    """`adapter`, counting the reads its scaled path issues."""
    reads = []
    original = adapter.get_data

    def counting(bounds):
        reads.append((tuple(bounds.start), tuple(bounds.stop)))
        return original(bounds)

    monkeypatch.setattr(adapter, "get_data", counting)
    return adapter, reads


def _set_grid(monkeypatch, adapter, grid, block=None):
    """Set the transfer grid, and what the backend is quantized to.

    ``block=None`` drives the unquantized path, where the tile is the grid. The
    fixture is a real ``ZarrAdapter`` whose store would otherwise floor the tile
    at its own 16x16 chunk -- correct behaviour, but it would mask the grid these
    cases are choosing.
    """
    monkeypatch.setattr(adapter, "get_transfer_chunk_size", lambda: grid)
    monkeypatch.setattr(type(adapter), "read_block_shape", property(lambda self: block))


class TestDefaultIsTodaysBehaviour:
    """The default must be indistinguishable from read-then-downsample.

    Every adapter inherits it, so a difference here is a difference in served
    pixels for every format at once.
    """

    @pytest.mark.parametrize("method", ["area", "nearest"])
    @pytest.mark.parametrize("scale", [(2, 2), (4, 4), (8, 1)])
    def test_matches_read_then_downsample(self, adapter, method, scale):
        bounds = _bounds((0, 0), (64, 64))

        fused = adapter.get_scaled_data(bounds, scale, method)
        expected = _ds.downsample_block(adapter.get_data(bounds), scale, method)

        assert fused.dtype == expected.dtype
        assert fused.shape == expected.shape
        assert np.array_equal(fused, expected)

    def test_shape_is_ceil_div_including_the_ragged_edge(self, adapter):
        """Contract 1: padding is the default's job, not the caller's.

        A ragged extent still yields ceil(extent / scale) -- the edge block is
        edge-replicated and divided by the FULL block size, so an override that
        trims instead would differ exactly at a tensor boundary.
        """
        out = adapter.get_scaled_data(_bounds((0, 0), (30, 30)), (4, 4), "area")
        assert out.shape == (8, 8)

    def test_dtype_is_the_input_dtype(self, adapter):
        out = adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (2, 2), "area")
        assert out.dtype == np.dtype("uint16")


class TestResolveChunkDataGoesThroughTheSeam:
    """The wiring: a scaled read must reach the override, an unscaled one must not."""

    def test_scaled_read_calls_get_scaled_data(self, adapter, monkeypatch):
        seen = {}
        original = adapter.get_scaled_data

        def spy(bounds, scale_hint, reduction_method, cache_manager=None):
            seen["args"] = (tuple(scale_hint), reduction_method)
            return original(bounds, scale_hint, reduction_method, cache_manager)

        monkeypatch.setattr(adapter, "get_scaled_data", spy)
        chunk_id = encode_chunk_id_with_scale(
            "src", _bounds((0, 0), (64, 64)), (4, 4), "area"
        )

        adapter.resolve_chunk_data(chunk_id)

        # The scale_hint and the method both come off the chunk_id, so an
        # override sees what the client asked for rather than a default.
        assert seen["args"] == ((4, 4), "area")

    def test_unscaled_read_does_not(self, adapter, monkeypatch):
        from biopb_tensor_server.core.chunk import encode_chunk_id

        def explode(*args, **kwargs):
            raise AssertionError("unscaled read must not reach get_scaled_data")

        monkeypatch.setattr(adapter, "get_scaled_data", explode)
        adapter.resolve_chunk_data(encode_chunk_id("src", _bounds((0, 0), (32, 32))))

    @pytest.mark.parametrize("method", ["area", "nearest"])
    def test_an_override_reaches_the_wire(self, adapter, monkeypatch, method):
        """What the override returns is what the client is served.

        The point of the seam: a fused adapter's pixels must not be re-reduced,
        re-padded, or otherwise post-processed on the way out.
        """
        sentinel = np.full((16, 16), 7, dtype=np.uint16)
        monkeypatch.setattr(adapter, "get_scaled_data", lambda *a, **k: sentinel)
        chunk_id = encode_chunk_id_with_scale(
            "src", _bounds((0, 0), (64, 64)), (4, 4), method
        )

        served = unpack_chunk_array(adapter.resolve_chunk_data(chunk_id))

        assert np.array_equal(served.reshape(sentinel.shape), sentinel)


class TestSeamIsDeclared:
    def test_is_part_of_the_declared_tensor_api(self):
        """The import-time API assertion has to know about it.

        core/adapter_base.py asserts TensorAdapter's public API equals
        _TENSOR_SCOPED_API at import; a new public method that is not classified
        fails the import, not a test.
        """
        assert "get_scaled_data" in _TENSOR_SCOPED_API
        assert hasattr(TensorAdapter, "get_scaled_data")


class TestStreamingDefault:
    """The streaming default (biopb/biopb#640 phase 2).

    The default reduces an extent unit by unit, so peak residency is one unit
    rather than the extent -- which is what lets a scaled chunk be planned
    without a memory ceiling shaping it. Streaming changes when bytes are read
    and in what shape, never which pixel comes out, so bit-identity is what is
    pinned here.

    The tile is the transfer grid rather than a byte budget, because the grid is
    the tiling the read plan and the chunk cache already use. It is the same
    tile for every backend; what a particular reader would rather have is a
    phase-2 adapter override, not a branch here.
    """

    @pytest.mark.parametrize("method", ["area", "nearest"])
    @pytest.mark.parametrize(
        "scale,stop",
        [
            ((2, 2), (64, 64)),  # aligned
            ((4, 4), (64, 64)),
            ((8, 8), (64, 64)),
            ((4, 4), (30, 30)),  # ragged on BOTH axes -- the edge pad path
            ((4, 4), (62, 64)),  # ragged only on rows
            ((1, 4), (64, 64)),  # row scale 1
            ((8, 1), (64, 64)),  # column scale 1
            ((8, 8), (64, 64)),  # unit 12 does not divide by the scale
        ],
    )
    def test_streamed_is_bit_identical(self, counted, monkeypatch, method, scale, stop):
        """The invariant, through the seam, with a grid small enough to force units.

        The ragged cases matter most: the extent's own end is the one place
        ``downsample_block``'s edge-replicate pad applies, and the unit that ends
        there has to carry it.
        """
        adapter, reads = counted
        bounds = _bounds((0, 0), stop)
        expected = _ds.downsample_block(adapter.get_data(bounds), scale, method)
        reads.clear()

        _set_grid(monkeypatch, adapter, (12, 12))
        actual = adapter.get_scaled_data(bounds, scale, method)

        assert len(reads) > 1, "grid should have forced more than one unit"
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert np.array_equal(actual, expected)

    def test_units_tile_the_extent_without_overlap(self, counted, monkeypatch):
        """Checked on the reads, not only the output.

        A unit that re-read a region would still produce correct-looking pixels
        for ``nearest``, and for ``area`` would double-count silently.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (8, 8), "area")

        seen = np.zeros((64, 64), dtype=int)
        for start, stop in reads:
            seen[start[0] : stop[0], start[1] : stop[1]] += 1
        assert (seen == 1).all(), "units must tile without gap or overlap"

    def test_a_grid_below_the_scale_grows_to_a_whole_block(self, counted, monkeypatch):
        """A tile smaller than one reduction block cannot be read on its own.

        It would reduce a fraction of a block and land a different pixel, so the
        tile rounds up to the block rather than the grid being taken literally.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (4, 4))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (8, 8), "area")

        assert reads, "expected at least one read"
        for start, stop in reads:
            assert stop[0] - start[0] == 8, "tile did not grow to a whole block"
            assert start[0] % 8 == 0, "tile boundary landed inside a block"

    def test_a_block_above_the_grid_floors_the_tile(self, counted, monkeypatch):
        """The dividing case: a tile inside a stored block re-reads that block.

        Measured at 8-11x on a page-mode OME-TIFF and ~3x on a coarsely chunked
        OME-Zarr, which is the whole reason the floor exists.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16), block=(32, 32))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (8, 8), "area")

        assert reads, "expected at least one read"
        for start, stop in reads:
            assert stop[0] - start[0] == 32, "tile was not raised to the block"
            assert start[0] % 32 == 0, "tile boundary landed inside a stored block"

    def test_a_block_below_the_grid_is_the_identity(self, counted, monkeypatch):
        """The coalescing case: the grid is already a whole multiple of it."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16), block=(8, 8))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (8, 8), "area")

        assert {stop[0] - start[0] for start, stop in reads} == {16}

    def test_the_real_store_declares_its_own_chunk(self, adapter):
        """Not monkeypatched: the fixture's zarr array reports its chunk grid."""
        assert adapter.read_block_shape == adapter.zarr_array.chunks

    def test_result_is_owned_even_for_nearest(self, counted, monkeypatch):
        """Contract 4. ``downsample_block`` hands back a strided VIEW for nearest.

        A view reaching the chunk cache would pin its full-resolution base, which
        is the memory streaming exists to bound.
        """
        adapter, _ = counted
        _set_grid(monkeypatch, adapter, (16, 16))

        out = adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "nearest")

        assert out.base is None

    def test_extent_inside_one_unit_reads_once(self, counted, monkeypatch):
        """Streaming an extent that already fits is pure overhead."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (64, 64))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area")

        assert len(reads) == 1

    def test_streaming_needs_no_opt_in(self, counted, monkeypatch):
        """It is the default: no flag, and nothing for an adapter to set."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area")

        assert len(reads) > 1
        assert not hasattr(TensorAdapter, "BANDED_SCALED_READ")

    def test_float_area_streams_like_any_other(self, monkeypatch):
        """Staged float means used to be the one reduction that could not stream.

        They neither commute nor associate, so no accumulator can carry them
        across units -- but whole blocks inside a unit need no accumulator, and
        that is what block-aligned units guarantee. There is no longer any
        reduction that has to read the extent whole.
        """
        with tempfile.TemporaryDirectory() as tmp:
            src = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 997.0
            arr = zarr.open_array(
                f"{tmp}/f.zarr",
                mode="w",
                shape=(64, 64),
                chunks=(16, 16),
                dtype="float32",
            )
            arr[:] = src
            adapter = ZarrAdapter(
                zarr.open_array(f"{tmp}/f.zarr", mode="r"), "src", ["y", "x"]
            )
            _set_grid(monkeypatch, adapter, (16, 16))
            reads = []
            original = adapter.get_data
            adapter.get_data = lambda b: (
                reads.append((tuple(b.start), tuple(b.stop))),
                original(b),
            )[1]
            bounds = _bounds((0, 0), (64, 64))

            out = adapter.get_scaled_data(bounds, (4, 4), "area")

            assert len(reads) > 1, "a float area must stream, not read whole"
            assert out.dtype == np.dtype("float32")
            assert np.array_equal(out, _ds.downsample_block(src, (4, 4), "area"))


class TestCacheSourcedUnits:
    """A scaled read may source its extent from the cache (biopb/biopb#640).

    The full-resolution chunks under a scaled read are ordinary cache entries:
    the extent snaps to ``transfer x scale``, so it tiles exactly into the
    absolute transfer grid the unscaled plan mints chunk_ids on. Where they are
    all warm, decoding the source again is work already done -- and the
    ``read_block_shape`` floor, which speaks for the backend, stops applying.

    Read-only by design: a miss reads the source and stores nothing, so a
    scaled read cannot evict the warm set to deliver one coarse chunk. What it
    trades is two per-byte rates -- the cache holds decoded bytes, the store
    holds what it holds -- which is why it is opted into per deployment.
    """

    @staticmethod
    def _warm_level_zero(adapter, cache):
        """Warm every full-resolution chunk exactly as an unscaled read does.

        Through ``get_read_plan`` rather than by minting ids here: the probe has
        to find the keys the *plan* stored under, and a test that minted its own
        would pass just as happily against a key scheme the server never writes.
        """
        plan = adapter.get_read_plan(TensorDescriptor())
        for endpoint in plan.chunk_endpoints:
            adapter.resolve_chunk_data(endpoint.chunk_id, cache)
        return [endpoint.chunk_id for endpoint in plan.chunk_endpoints]

    @pytest.mark.parametrize("content_version", [None, b"v1"])
    def test_cache_probe_keys_match_the_read_plan(
        self, adapter, monkeypatch, content_version
    ):
        """The probe and unscaled plan must address identical cache entries."""
        _set_grid(monkeypatch, adapter, (16, 16))
        monkeypatch.setattr(adapter, "_content_version", content_version)
        base_desc = adapter.get_tensor_descriptor()
        plan = adapter.get_read_plan(TensorDescriptor())

        expected = {
            tuple(endpoint.bounds.start): cache_key_for_chunk_id(endpoint.chunk_id)
            for endpoint in plan.chunk_endpoints
        }
        actual = dict(
            _cs.chunk_cache_keys(
                base_desc, adapter.content_version, (0, 0), (64, 64), (16, 16)
            )
        )

        assert actual == expected

    @staticmethod
    def _units(adapter, monkeypatch):
        """Record the extent each cache-sourced unit covers, by either route."""
        seen = []
        assemble = _cs.assemble_from_cache
        borrow = _cs.borrow_cached_unit

        def assemble_spy(cache_manager, keys, start, stop, *rest):
            seen.append((tuple(start), tuple(stop)))
            return assemble(cache_manager, keys, start, stop, *rest)

        def borrow_spy(cache_manager, keys, start, stop, *rest):
            seen.append((tuple(start), tuple(stop)))
            return borrow(cache_manager, keys, start, stop, *rest)

        monkeypatch.setattr(_cs, "assemble_from_cache", assemble_spy)
        monkeypatch.setattr(_cs, "borrow_cached_unit", borrow_spy)
        return seen

    def test_a_warm_extent_never_touches_the_source(self, counted, monkeypatch, cache):
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        self._warm_level_zero(adapter, cache)
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert reads == [], "a warm extent must not re-read the source"
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)

    @pytest.mark.parametrize("method", ["area", "nearest"])
    @pytest.mark.parametrize(
        "grid,scale",
        [
            ((16, 16), (4, 4)),  # a chunk is a whole number of blocks
            ((24, 24), (4, 4)),  # chunks ragged against the tensor's own end
            ((16, 16), (32, 32)),  # a block spans two chunks, so the unit does
        ],
    )
    def test_assembled_is_bit_identical(
        self, counted, monkeypatch, cache, method, grid, scale
    ):
        """Sourcing is not a decision about what the data is.

        The ragged grid matters: its last chunk on each axis is short, and the
        cached entry is short in exactly the same way -- which is what makes the
        bounds derived here the plan's own.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, grid)
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), scale, method)
        self._warm_level_zero(adapter, cache)
        reads.clear()

        out = adapter.get_scaled_data(bounds, scale, method, cache)

        assert reads == []
        assert np.array_equal(out, expected)

    def test_an_extent_that_ends_mid_chunk_declines(self, counted, monkeypatch, cache):
        """A partial chunk keys differently from the whole one that is cached.

        No read plan asks for such an extent -- a scaled chunk's bounds snap to
        ``transfer x scale`` -- so this is a guard, not a case to support: the
        alternative is a probe that misses on every entry.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        self._warm_level_zero(adapter, cache)
        bounds = _bounds((0, 0), (30, 30))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert reads, "an extent off the chunk grid must go to the source"
        assert np.array_equal(out, expected)

    def test_an_averaging_reduction_keeps_the_callers_unit(
        self, counted, monkeypatch, cache
    ):
        """`area` runs prod(scale) strided adds per unit whatever its size, so
        more units is the same bytes over more, smaller numpy calls -- measured
        197 ms against 82. A 64x64 read block quantizes this extent into ONE
        unit, and the cache path assembles that unit from its 16 chunks."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16), block=(64, 64))
        bounds = _bounds((0, 0), (64, 64))
        self._warm_level_zero(adapter, cache)
        units = self._units(adapter, monkeypatch)
        reads.clear()

        adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert units == [((0, 0), (64, 64))]
        assert reads == []

    def test_a_pick_drops_the_unit_to_the_chunk_grid(self, counted, monkeypatch, cache):
        """`nearest` needs none of what it skips, so the smallest unit that
        tiles into whole chunks wins: a unit that is one chunk is one mapping
        the pick reads straight out of (``TestBorrowedUnits``), where a wide one
        would land as strided rows in a buffer. The read_block floor does not
        apply -- it speaks for a source read this path does not make."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16), block=(64, 64))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "nearest")
        self._warm_level_zero(adapter, cache)
        units = self._units(adapter, monkeypatch)
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "nearest", cache)

        assert len(units) == 16
        assert {(hi[0] - lo[0], hi[1] - lo[1]) for lo, hi in units} == {(16, 16)}
        assert reads == []
        assert np.array_equal(out, expected)

    def test_a_unit_spanning_several_chunks_is_assembled_from_all_of_them(
        self, counted, monkeypatch, cache
    ):
        """The k x k case: one unit, sixteen entries, bit-identical."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16), block=(64, 64))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        self._warm_level_zero(adapter, cache)
        acquired = []
        original = cache.try_acquire
        monkeypatch.setattr(
            cache,
            "try_acquire",
            lambda key, touch=True: (acquired.append(key), original(key, touch))[1],
        )
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert len(acquired) == 16
        assert reads == []
        assert np.array_equal(out, expected)

    def test_a_cold_extent_reads_the_source_and_stores_nothing(
        self, counted, monkeypatch, cache
    ):
        """The probe is read-only: a scaled read must not populate the
        full-resolution grid it looked for."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        plan = adapter.get_read_plan(TensorDescriptor())
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area", cache)

        assert len(reads) > 1, "a cold extent still streams from the source"
        assert not any(
            cache.contains(cache_key_for_chunk_id(endpoint.chunk_id))
            for endpoint in plan.chunk_endpoints
        )

    def test_one_missing_chunk_declines_the_whole_extent(
        self, counted, monkeypatch, cache
    ):
        """All-or-nothing per extent. A partial mix would read the misses one
        chunk at a time, which is exactly what the read_block floor prevents."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        chunk_ids = self._warm_level_zero(adapter, cache)
        cache.remove(cache_key_for_chunk_id(chunk_ids[-1]))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert reads, "one cold chunk must send the whole extent to the source"
        assert np.array_equal(out, expected)

    def test_an_eviction_after_the_probe_falls_back_per_unit(
        self, counted, monkeypatch, cache
    ):
        """A probe is a decision, not a promise."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        self._warm_level_zero(adapter, cache)
        reads.clear()
        # Everything is present to `contains` and gone by the time it is read.
        monkeypatch.setattr(cache, "try_acquire", lambda key, touch=True: None)

        out = adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert reads, "the fallback must reach the source"
        assert np.array_equal(out, expected)

    def test_a_scale_that_rounds_off_the_grid_declines(
        self, counted, monkeypatch, cache
    ):
        """A non-dyadic scale_hint raises the unit to a whole reduction block --
        16 -> 18 -- which no longer tiles into chunks. Left to the source."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        self._warm_level_zero(adapter, cache)
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (3, 3), "area")
        reads.clear()

        out = adapter.get_scaled_data(bounds, (3, 3), "area", cache)

        assert reads, "an undecomposable unit must not be sourced from the cache"
        assert np.array_equal(out, expected)

    def test_resolve_chunk_data_hands_the_cache_down(self, counted, monkeypatch, cache):
        """The wiring, through the real caller.

        `resolve_chunk_data` is the only production caller, so a handle it does
        not forward makes every test above a test of an unreachable path.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        self._warm_level_zero(adapter, cache)
        chunk_id = encode_chunk_id_with_scale(
            "src", _bounds((0, 0), (64, 64)), (4, 4), "area"
        )
        reads.clear()

        served = unpack_chunk_array(adapter.resolve_chunk_data(chunk_id, cache))

        assert reads == [], "a warm extent must not re-read the source"
        expected = _ds.downsample_block(
            adapter.get_data(_bounds((0, 0), (64, 64))), (4, 4), "area"
        )
        assert np.array_equal(served.reshape(expected.shape), expected)

    def test_sourcing_does_not_credit_the_chunks_it_read(
        self, counted, monkeypatch, cache
    ):
        """The unintended half of on-by-default, closed.

        Warming the OS page cache for the full-resolution chunks is the point;
        making them look hot to the eviction policy is not. One coarse read
        covers every chunk of the source, so a scaled read must not be able to
        decide what stays cached.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        self._warm_level_zero(adapter, cache)
        touched = []
        original = cache.try_acquire
        monkeypatch.setattr(
            cache,
            "try_acquire",
            lambda key, touch=True: (touched.append(touch), original(key, touch))[1],
        )
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area", cache)

        assert reads == [], "expected the cache-sourced path"
        assert touched and not any(touched), "every probe read must be untouched"

    def test_it_is_on_by_default(self, counted, monkeypatch, tmp_path):
        """Inherited rather than opted into: a coarse read is usually a pretext
        for full-resolution work, and sourcing it this way leaves that work's
        pages resident."""
        adapter, reads = counted
        manager = CacheManager(
            CacheConfig(backend="file", file_cache_dir=tmp_path / "cache")
        )
        try:
            assert manager.source_scaled_reads is True
            _set_grid(monkeypatch, adapter, (16, 16))
            self._warm_level_zero(adapter, manager)
            reads.clear()

            adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area", manager)

            assert reads == []
        finally:
            manager.close()

    def test_it_can_be_turned_off(self, counted, monkeypatch, tmp_path):
        """The escape hatch still reads the source, warm cache or not."""
        adapter, reads = counted
        manager = CacheManager(
            CacheConfig(
                backend="file",
                file_cache_dir=tmp_path / "cache",
                source_scaled_reads=False,
            )
        )
        try:
            _set_grid(monkeypatch, adapter, (16, 16))
            self._warm_level_zero(adapter, manager)
            bounds = _bounds((0, 0), (64, 64))
            expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
            reads.clear()

            out = adapter.get_scaled_data(bounds, (4, 4), "area", manager)

            assert len(reads) > 1
            assert np.array_equal(out, expected)
        finally:
            manager.close()

    def test_no_cache_is_the_path_it_always_was(self, counted, monkeypatch):
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "area")

        assert len(reads) > 1
        assert np.array_equal(out, expected)

    def test_the_memory_backend_leaves_nothing_to_source_from(
        self, counted, monkeypatch, tmp_path
    ):
        """Documented, not incidental: `resolve_chunk_data` caches unscaled
        chunks only on the file backend, so a level-0 read against the memory
        backend stores nothing and the probe costs one index lookup."""
        adapter, reads = counted
        manager = CacheManager(CacheConfig(backend="memory", source_scaled_reads=True))
        try:
            _set_grid(monkeypatch, adapter, (16, 16))
            self._warm_level_zero(adapter, manager)
            reads.clear()

            adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area", manager)

            assert len(reads) > 1
        finally:
            manager.close()


class TestBorrowedUnits:
    """A pick reduces out of the segment mapping instead of a copy of it.

    ``nearest`` sizes its unit at one chunk, so the unit *is* one cache entry:
    copying it into a buffer moves prod(scale) times the bytes the pick then
    reads. Borrowing skips that -- at the price of a view whose validity ends
    with the entry, which is what most of this class is about.
    """

    _warm_level_zero = staticmethod(TestCacheSourcedUnits._warm_level_zero)

    @staticmethod
    def _refcounts(cache, monkeypatch):
        """(acquires, releases) the read makes through the manager."""
        acquired, released = [], []
        acquire, release = cache.try_acquire, cache.release

        def acquire_spy(key, touch=True):
            entry = acquire(key, touch)
            if entry is not None:
                acquired.append(key)
            return entry

        def release_spy(key):
            released.append(key)
            return release(key)

        monkeypatch.setattr(cache, "try_acquire", acquire_spy)
        monkeypatch.setattr(cache, "release", release_spy)
        return acquired, released

    def test_a_pick_never_copies_the_unit(self, counted, monkeypatch, cache):
        """The optimization itself: sixteen units, sixteen borrows, no buffer.

        Scale 2 so the shapes separate -- a unit is 16x16 full-resolution and
        the output is 32x32, so the only 16x16 allocation there could be is the
        unit copy this removes.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (2, 2), "nearest")
        self._warm_level_zero(adapter, cache)
        borrowed, assembled = [], []
        borrow = _cs.borrow_cached_unit
        monkeypatch.setattr(
            _cs,
            "borrow_cached_unit",
            lambda *a: (borrowed.append(a[2]), borrow(*a))[1],
        )
        monkeypatch.setattr(
            _cs, "assemble_from_cache", lambda *a: assembled.append(a[2])
        )
        allocated = []
        empty = np.empty
        monkeypatch.setattr(
            np,
            "empty",
            lambda shape, *a, **k: (allocated.append(shape), empty(shape, *a, **k))[1],
        )
        reads.clear()

        out = adapter.get_scaled_data(bounds, (2, 2), "nearest", cache)

        assert len(borrowed) == 16
        assert assembled == []
        assert (16, 16) not in allocated, "a borrowed unit needs no buffer"
        assert reads == []
        assert np.array_equal(out, expected)

    def test_the_reduced_array_outlives_the_entry(self, counted, monkeypatch, cache):
        """Contract 4, and the one that would hand back freed pages.

        The single-unit shortcut returns the reduction itself, and a pick is a
        strided *view* of what it reduced -- of the segment mapping, here. It
        has to be materialised before the entry is released.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (64, 64))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "nearest")
        self._warm_level_zero(adapter, cache)
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "nearest", cache)

        assert reads == [], "expected the borrowed path"
        assert out.base is None and out.flags.owndata, "a view onto a released entry"
        cache.clear()
        assert np.array_equal(out, expected)

    def test_a_buffered_pick_is_not_copied_on_the_way_out(
        self, counted, monkeypatch, cache
    ):
        """The copy is owed by the *unit*, not by the read's mode.

        A grid below the reduction block rounds the pick's unit up past one
        chunk, so it is buffered rather than borrowed -- and that buffer is the
        closure's own, which outlives the read. Materialising here would copy a
        pick that was already safe to return, which is what asking
        ``source_scaled_reads`` instead of the unit would do.

        The grid is narrow on one axis and wide on the other so the pick is
        genuinely strided: a unit of one reduction block reduces to a single
        element, which numpy calls contiguous, and nothing would be copied
        either way.
        """
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (2, 16))
        bounds = _bounds((0, 0), (4, 16))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "nearest")
        self._warm_level_zero(adapter, cache)
        attempts = []
        borrow = _cs.borrow_cached_unit

        def spy(*args):
            lent = borrow(*args)
            attempts.append(lent)
            return lent

        monkeypatch.setattr(_cs, "borrow_cached_unit", spy)
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "nearest", cache)

        assert attempts and all(lent is None for lent in attempts), (
            "a unit of several chunks must decline the borrow"
        )
        assert reads == [], "expected the cache-sourced path"
        assert out.base is not None, "a buffered pick must not be copied"
        assert np.array_equal(out, expected)

    @pytest.mark.parametrize("grid", [(16, 16), (64, 64)])
    def test_every_borrowed_entry_is_released(self, counted, monkeypatch, cache, grid):
        """Streamed and single-unit alike: an entry held past the reduction
        would pin its segment against eviction for the life of the server."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, grid)
        self._warm_level_zero(adapter, cache)
        acquired, released = self._refcounts(cache, monkeypatch)
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "nearest", cache)

        assert reads == [], "expected the borrowed path"
        assert acquired and released == acquired

    def test_a_failed_reduction_releases_too(self, counted, monkeypatch, cache):
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (64, 64))
        self._warm_level_zero(adapter, cache)
        acquired, released = self._refcounts(cache, monkeypatch)
        monkeypatch.setattr(
            "biopb_tensor_server.core.adapter_base.downsample_block",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "nearest", cache)

        assert acquired and released == acquired

    def test_a_unit_of_several_chunks_is_assembled_as_before(
        self, counted, monkeypatch, cache
    ):
        """Borrowing needs the unit to *be* one entry. A grid below the
        reduction block rounds the unit up past it, and that unit is pasted
        together exactly as an averaging one is."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (2, 2))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "nearest")
        self._warm_level_zero(adapter, cache)
        assembled = []
        assemble = _cs.assemble_from_cache
        monkeypatch.setattr(
            _cs,
            "assemble_from_cache",
            lambda *a: (assembled.append(a[2]), assemble(*a))[1],
        )
        reads.clear()

        out = adapter.get_scaled_data(bounds, (4, 4), "nearest", cache)

        assert len(assembled) == 16 * 16, "a 4x4 unit over a 2x2 grid"
        assert reads == []
        assert np.array_equal(out, expected)

    def test_an_eviction_between_the_probe_and_the_borrow_reads_the_source(
        self, counted, monkeypatch, cache
    ):
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        bounds = _bounds((0, 0), (64, 64))
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "nearest")
        self._warm_level_zero(adapter, cache)
        reads.clear()
        monkeypatch.setattr(cache, "try_acquire", lambda key, touch=True: None)

        out = adapter.get_scaled_data(bounds, (4, 4), "nearest", cache)

        assert reads, "the fallback must reach the source"
        assert np.array_equal(out, expected)

    def test_an_averaging_reduction_does_not_borrow(self, counted, monkeypatch, cache):
        """`area` reads every byte anyway, and its edge units go through the
        padding and strided-add kernels -- not worth handing those a read-only
        view of a cached chunk for one memcpy of the ~4% it costs."""
        adapter, reads = counted
        _set_grid(monkeypatch, adapter, (16, 16))
        self._warm_level_zero(adapter, cache)
        borrowed = []
        monkeypatch.setattr(_cs, "borrow_cached_unit", lambda *a: borrowed.append(a[2]))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area", cache)

        assert borrowed == []
        assert reads == []
