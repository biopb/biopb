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
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core import (
    adapter_base as _ab,
    cache_source as _cs,
    downsample as _ds,
)
from biopb_tensor_server.core.chunk import cache_key_for_chunk_id
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.core.normalize import NormalizingAdapter, normalize_adapter

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


@pytest.fixture
def cache(tmp_path):
    manager = CacheManager(
        CacheConfig(
            backend="file",
            file_cache_dir=tmp_path / "cache",
            source_scaled_reads=True,
        )
    )
    try:
        yield manager
    finally:
        manager.close()


def _on_grid(monkeypatch, adapter, grid=GRID):
    """Plan and probe on the same transfer grid, unquantized.

    The fixture's store would otherwise floor the streaming tile at its own
    chunk, and the grid is what both the plan's chunk_ids and the probe's keys
    are minted on -- a test that let them differ would be testing the mismatch.
    """
    monkeypatch.setattr(adapter, "get_transfer_chunk_size", lambda: grid)
    monkeypatch.setattr(type(adapter), "read_block_shape", property(lambda self: None))


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
        _on_grid(monkeypatch, adapter)

        assert _probe_keys(adapter, (0, 0), (64, 64)) == _plan_keys(adapter)

    def test_the_version_reaches_the_key(self, adapter, monkeypatch):
        """``content_version`` travels as an argument now rather than off
        ``self``, and it is the one the extraction could have dropped: a probe
        that forgets it still mints keys, just never onto anything the plan
        wrote."""
        monkeypatch.setattr(adapter, "_content_version", b"v2")
        _on_grid(monkeypatch, adapter)
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
        _on_grid(monkeypatch, adapter)

        assert _probe_keys(adapter, (32, 32), (64, 64)) < _plan_keys(adapter)
        assert len(_probe_keys(adapter, (32, 32), (64, 64))) == 4


class TestADelegatingWrapper:
    """``NormalizingAdapter`` inherits ``get_scaled_data`` and delegates the
    reads under it, so the extraction has to leave it on the same path it was on.
    """

    @pytest.fixture
    def wrapped(self):
        """A permuting wrapper: ``normalize_adapter`` hands back anything
        already canonical unchanged, so a wrapper is always a permuting one."""
        with tempfile.TemporaryDirectory() as tmp:
            inner = _zarr(tmp, (32, 64), ("x", "y"))
            wrapper = normalize_adapter(inner)
            assert isinstance(wrapper, NormalizingAdapter)
            yield wrapper, inner

    def test_the_probe_runs_on_the_wrapper_and_the_delegates_version(
        self, wrapped, cache, monkeypatch
    ):
        """What the mixin used to read off ``self`` now travels as arguments:
        the wrapper's own (canonical) descriptor, and the version its delegate
        owns."""
        wrapper, inner = wrapped
        monkeypatch.setattr(inner, "_content_version", b"v2")
        seen = []
        sourced = _cs.cache_sourced_units

        def spy(cache_manager, descriptor, version, *rest):
            seen.append((descriptor, version))
            return sourced(cache_manager, descriptor, version, *rest)

        # The call site imported the name, so this is where it is bound.
        monkeypatch.setattr(_ab, "cache_sourced_units", spy)
        bounds = ChunkBounds(start=[0, 0], stop=[64, 32])

        wrapper.get_scaled_data(bounds, (4, 4), "area", cache)

        assert len(seen) == 1
        descriptor, content_version = seen[0]
        assert list(descriptor.dim_labels) == ["y", "x"], "the wrapper's own order"
        assert content_version == inner.content_version == b"v2"

    def test_a_permuted_wrapper_declines_and_reads_the_source(
        self, wrapped, cache, monkeypatch
    ):
        """Today's behaviour, pinned rather than endorsed (biopb/biopb#986).

        The plan carries the *delegate's* chunk_ids verbatim, so a warm cache
        holds native-order bounds; the probe mints canonical ones off the
        wrapper's descriptor. The two never meet, the read falls back to the
        source, and the pixels are right anyway -- which is why nothing has
        noticed. Fixing it belongs with whoever reconciles the two orders.
        """
        wrapper, inner = wrapped
        monkeypatch.setattr(inner, "get_transfer_chunk_size", lambda: (16, 32))
        monkeypatch.setattr(wrapper, "get_transfer_chunk_size", lambda: (32, 16))
        monkeypatch.setattr(
            type(inner), "read_block_shape", property(lambda self: None)
        )
        for endpoint in wrapper.get_read_plan(TensorDescriptor()).chunk_endpoints:
            wrapper.resolve_chunk_data(endpoint.chunk_id, cache)
        bounds = ChunkBounds(start=[0, 0], stop=[64, 32])
        expected = _ds.downsample_block(wrapper.get_data(bounds), (4, 4), "area")
        reads = []
        native = inner.get_data
        monkeypatch.setattr(
            inner, "get_data", lambda b: (reads.append(tuple(b.start)), native(b))[1]
        )

        out = wrapper.get_scaled_data(bounds, (4, 4), "area", cache)

        assert reads, "the probe misses, so the source is read"
        assert np.array_equal(out, expected)
