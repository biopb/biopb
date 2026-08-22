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
