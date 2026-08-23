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
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.core import downsample as _ds
from biopb_tensor_server.core.adapter_base import (
    _TENSOR_SCOPED_API,
    TensorAdapter,
    unpack_chunk_array,
)
from biopb_tensor_server.core.chunk import encode_chunk_id_with_scale


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

        def spy(bounds, scale_hint, reduction_method):
            seen["args"] = (tuple(scale_hint), reduction_method)
            return original(bounds, scale_hint, reduction_method)

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

    @pytest.fixture
    def counted(self, adapter, monkeypatch):
        """`adapter`, counting the reads its scaled path issues."""
        reads = []
        original = adapter.get_data

        def counting(bounds):
            reads.append((tuple(bounds.start), tuple(bounds.stop)))
            return original(bounds)

        monkeypatch.setattr(adapter, "get_data", counting)
        return adapter, reads

    @staticmethod
    def _grid(monkeypatch, adapter, grid):
        """Set the transfer grid, which is the tile the stream reads in."""
        monkeypatch.setattr(adapter, "get_transfer_chunk_size", lambda: grid)

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

        self._grid(monkeypatch, adapter, (12, 12))
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
        self._grid(monkeypatch, adapter, (16, 16))
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
        self._grid(monkeypatch, adapter, (4, 4))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (8, 8), "area")

        assert reads, "expected at least one read"
        for start, stop in reads:
            assert stop[0] - start[0] == 8, "tile did not grow to a whole block"
            assert start[0] % 8 == 0, "tile boundary landed inside a block"

    def test_result_is_owned_even_for_nearest(self, counted, monkeypatch):
        """Contract 4. ``downsample_block`` hands back a strided VIEW for nearest.

        A view reaching the chunk cache would pin its full-resolution base, which
        is the memory streaming exists to bound.
        """
        adapter, _ = counted
        self._grid(monkeypatch, adapter, (16, 16))

        out = adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "nearest")

        assert out.base is None

    def test_extent_inside_one_unit_reads_once(self, counted, monkeypatch):
        """Streaming an extent that already fits is pure overhead."""
        adapter, reads = counted
        self._grid(monkeypatch, adapter, (64, 64))
        reads.clear()

        adapter.get_scaled_data(_bounds((0, 0), (64, 64)), (4, 4), "area")

        assert len(reads) == 1

    def test_streaming_needs_no_opt_in(self, counted, monkeypatch):
        """It is the default: no flag, and nothing for an adapter to set."""
        adapter, reads = counted
        self._grid(monkeypatch, adapter, (16, 16))
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
            self._grid(monkeypatch, adapter, (16, 16))
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
