"""The file-identity handle pool (#1284 item 2)."""

import time

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
    return HandlePool(ttl_seconds=60.0, max_handles=2, thread_name="test-pool")


def test_a_second_checkout_reuses_the_handle(pool):
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn) as first:
        pass
    with pool.checkout("a", open_fn) as second:
        assert second is first
    assert open_fn.opened == 1


def test_a_leased_handle_survives_sweep_cap_and_drop_until_released(pool):
    open_fn = _Opener("a")
    pool.set_ttl(0.001)
    with pool.checkout("a", open_fn):
        time.sleep(0.01)
        pool.sweep()
        pool.drop("a")
        assert open_fn.closed == 0
    assert open_fn.closed == 1
    assert len(pool) == 0


def test_sweep_closes_idle_handles(pool):
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn):
        pass
    pool.set_ttl(0.001)
    time.sleep(0.01)
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


def test_a_rebuilt_adapter_reuses_its_predecessors_handle(tmp_path):
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
    key = first.get_tensor_adapter(
        first.list_tensor_descriptors()[0].array_id
    )._pool_key()
    handle = pool._handles[key]
    del first
    _, again = read()
    np.testing.assert_array_equal(again.reshape(data.shape), data)
    assert pool._handles[key] is handle

    source, _ = read()
    source.close()
    assert key not in pool._handles


def test_a_pool_with_no_ttl_keeps_nothing_between_reads():
    pool = HandlePool(ttl_seconds=0.0, max_handles=2, thread_name="off")
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn):
        pass
    assert open_fn.closed == 1
    assert len(pool) == 0


def test_the_process_ceiling_caps_the_pool_ttl():
    from biopb_tensor_server.adapters._handle_reaper import set_handle_reaper_ttl

    pool = HandlePool(ttl_seconds=100.0, max_handles=2, thread_name="cap")
    try:
        set_handle_reaper_ttl(5.0)
        assert pool.ttl == 5.0
    finally:
        set_handle_reaper_ttl(float("inf"))


def test_concurrent_misses_on_one_key_open_once(pool):
    import threading

    open_fn = _Opener("a")
    slow = lambda: (time.sleep(0.05), open_fn())[1]  # noqa: E731
    leased = []

    def work():
        with pool.checkout("a", slow) as handle:
            leased.append(handle)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert open_fn.opened == 1
    assert len({id(h) for h in leased}) == 1


def test_a_non_persistent_checkout_closes_at_the_end_and_is_not_pooled(pool):
    open_fn = _Opener("a")
    with pool.checkout("a", open_fn, persist=False) as handle:
        assert handle is not None
        assert len(pool) == 0
    assert open_fn.closed == 1
