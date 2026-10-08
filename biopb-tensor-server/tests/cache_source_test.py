"""``core.cache_source``: the keys a scaled read probes with (biopb/biopb#980).

The cluster used to be methods on ``TensorAdapter``, inherited by every adapter
and every delegating wrapper. It is module functions now, and the one piece of
adapter state it read -- ``content_version`` -- is an argument. What that could
break is silent: a key that differs by one byte misses every entry, the read
falls back to the source, and the pixels stay right. So these compare the probe's
keys against the ones an unscaled read actually *stored* under, rather than
against a second minting of the same expression.
"""

import tempfile

import numpy as np
import pytest
import zarr
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.core import (
    adapter_base as _ab,
    cache_source as _cs,
    downsample as _ds,
)
from biopb_tensor_server.core.chunk import cache_key_for_chunk_id

from tests.scaled_data_seam_test import _set_grid

GRID = (16, 16)


def _zarr(tmp, shape, labels):
    """A zarr adapter over ``shape``, labelled ``labels``, on GRID."""
    path = f"{tmp}/{'-'.join(labels)}.zarr"
    arr = zarr.open_array(path, mode="w", shape=shape, chunks=GRID, dtype="uint16")
    arr[:] = (np.arange(int(np.prod(shape)), dtype=np.uint16) % 4093).reshape(shape)
    return ZarrAdapter(zarr.open_array(path, mode="r"), "src", list(labels))


@pytest.fixture
def adapter():
    with tempfile.TemporaryDirectory() as tmp:
        yield _zarr(tmp, (64, 64), ("y", "x"))


def _plan_keys(adapter):
    """The cache keys an unscaled read of the whole tensor stores under."""
    plan = adapter.get_read_plan(TensorDescriptor())
    return {cache_key_for_chunk_id(e.chunk_id) for e in plan.chunk_endpoints}


def _probe_keys(adapter, start, stop, grid=GRID):
    return {
        key
        for _, key in _cs.chunk_cache_keys(
            adapter.get_tensor_descriptor(), adapter.content_version, start, stop, grid
        )
    }


class TestTheProbeKeysAreThePlansKeys:
    """The probe asks ``contains`` about keys it minted itself. If they are not
    the plan's own, every scaled read declines and nothing says so."""

    @pytest.mark.parametrize("content_version", [None, b"v2"])
    def test_the_source_keys_as_its_plan_does(
        self, adapter, monkeypatch, content_version
    ):
        monkeypatch.setattr(adapter, "_content_version", content_version)
        _set_grid(monkeypatch, adapter, GRID)

        assert _probe_keys(adapter, (0, 0), (64, 64)) == _plan_keys(adapter)

    def test_the_version_reaches_the_key(self, adapter, monkeypatch):
        """``content_version`` travels as an argument now rather than off
        ``self``, and it is the one the extraction could have dropped: a probe
        that forgets it still mints keys, just never onto anything the plan
        wrote."""
        monkeypatch.setattr(adapter, "_content_version", b"v2")
        _set_grid(monkeypatch, adapter, GRID)
        descriptor = adapter.get_tensor_descriptor()

        versioned = _probe_keys(adapter, (0, 0), (64, 64))
        forgotten = {
            key
            for _, key in _cs.chunk_cache_keys(descriptor, None, (0, 0), (64, 64), GRID)
        }

        assert len(forgotten) == len(versioned)
        assert not (forgotten & versioned)

    def test_the_extent_is_keyed_chunk_by_chunk(self, adapter, monkeypatch):
        """A sub-extent probes only the chunks under it, on the absolute grid --
        not a grid rebased on the extent, which would key nothing that exists."""
        _set_grid(monkeypatch, adapter, GRID)

        assert _probe_keys(adapter, (32, 32), (64, 64)) < _plan_keys(adapter)
        assert len(_probe_keys(adapter, (32, 32), (64, 64))) == 4


class TestAPermutedAdapter:
    """A non-canonical adapter plans, keys and reads in canonical order, so the
    cache-sourced probe mints the same chunk_ids the plan did."""

    @pytest.fixture
    def permuted(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _zarr(tmp, (32, 64), ("x", "y"))
            assert adapter._axis_perm() is not None
            yield adapter

    def test_the_probe_runs_on_the_canonical_descriptor_and_the_adapters_version(
        self, permuted, cache, monkeypatch
    ):
        adapter = permuted
        monkeypatch.setattr(adapter, "_content_version", b"v2")
        seen = []
        sourced = _cs.cache_sourced_units

        def spy(cache_manager, descriptor, version, *rest):
            seen.append((descriptor, version))
            return sourced(cache_manager, descriptor, version, *rest)

        # The call site imported the name, so this is where it is bound.
        monkeypatch.setattr(_ab, "cache_sourced_units", spy)
        bounds = ChunkBounds(start=[0, 0], stop=[64, 32])

        adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert len(seen) == 1
        descriptor, content_version = seen[0]
        assert list(descriptor.dim_labels) == ["y", "x"]
        assert content_version == adapter.content_version == b"v2"

    def test_a_warm_cache_serves_a_scaled_read(self, permuted, cache, monkeypatch):
        adapter = permuted
        monkeypatch.setattr(adapter, "get_transfer_chunk_size", lambda: (32, 16))
        monkeypatch.setattr(
            type(adapter), "_native_read_block_shape", property(lambda self: None)
        )
        for endpoint in adapter.get_read_plan(TensorDescriptor()).chunk_endpoints:
            adapter.resolve_chunk_data(endpoint.chunk_id, cache)
        bounds = ChunkBounds(start=[0, 0], stop=[64, 32])
        expected = _ds.downsample_block(adapter.get_data(bounds), (4, 4), "area")
        reads = []
        native = adapter._read_native
        monkeypatch.setattr(
            adapter,
            "_read_native",
            lambda b: (reads.append(tuple(b.start)), native(b))[1],
        )

        out = adapter.get_scaled_data(bounds, (4, 4), "area", cache)

        assert not reads, "the probe hits the warm chunks"
        assert np.array_equal(out, expected)
