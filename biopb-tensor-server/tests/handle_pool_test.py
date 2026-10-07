"""The file-identity handle pool (#1284 item 2)."""

import numpy as np
import pytest
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import OmeTiffAdapter, ome_tiff as ome_tiff_module
from biopb_tensor_server.adapters._handle_pool import HandlePool, PooledHandle
from biopb_tensor_server.core.config import SourceConfig


class _Opener:
    def __init__(self, key):
        self.key = key
        self.opened = 0
        self.closed = 0

    def __call__(self):
        self.opened += 1
        return PooledHandle(self.key, object(), self._close)

    def _close(self):
        self.closed += 1


@pytest.fixture
def pool():
    return HandlePool(ttl_seconds=0.0, max_handles=2, thread_name="test-pool")


def test_a_second_checkout_reuses_the_handle(pool):
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn) as first:
        pass
    with pool.checkout("a", open_fn) as second:
        assert second is first
    assert open_fn.opened == 1


def test_a_leased_handle_survives_sweep_cap_and_drop_until_released(pool):
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn):
        pool.sweep()
        pool.drop("a")
        assert open_fn.closed == 0
    assert open_fn.closed == 1
    assert len(pool) == 0


def test_sweep_closes_idle_handles(pool):
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn):
        pass
    pool.sweep()
    assert open_fn.closed == 1


def test_cap_closes_the_least_recently_used_idle_handle(pool):
    openers = {k: _Opener(k) for k in "abc"}
    for key, open_fn in openers.items():
        with pool.checkout(key, open_fn):
            pass
    assert len(pool) == 2
    assert openers["a"].closed == 1
    assert openers["c"].closed == 0


def test_a_failed_open_yields_no_handle(pool):
    with pool.checkout("a", lambda: None) as handle:
        assert handle is None


def test_a_rebuilt_adapter_reuses_its_predecessors_handle(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOPB_OMETIFF_HANDLE_POOL", "1")
    path = tmp_path / "a.ome.tif"
    data = np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "ZYX"})
    config = SourceConfig(url=str(path), source_id="s0")

    def read():
        source = OmeTiffAdapter.create_from_config(config)
        scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)
        out = scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 3, 32, 32]))
        return source, out

    first, out = read()
    np.testing.assert_array_equal(out.reshape(data.shape), data)
    pool = ome_tiff_module._store_pool
    handles = list(pool._handles.values())
    del first
    _, again = read()
    np.testing.assert_array_equal(again.reshape(data.shape), data)
    assert list(pool._handles.values()) == handles

    source, _ = read()
    source.close()
    assert len(pool) == 0
