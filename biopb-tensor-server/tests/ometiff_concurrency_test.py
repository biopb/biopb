"""Concurrency of the OmeTiffAdapter read path.

Reads hold a lease on the pooled store, so it is never closed mid-read. Two modes,
selected by ``BIOPB_OMETIFF_PARALLEL_READ`` (default off):

- **Default** -- readers of one store serialize on its handle lock.
- **Opt-in lock-free** -- readers decode without it (tifffile serializes the raw
  seek+read on its own handle lock; the tile decode is per-tile into a fresh
  buffer).

These tests pin that reads are correct under heavy concurrency in both modes, and
that the flag genuinely switches the lock-holding behavior during decode.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import OmeTiffAdapter
from biopb_tensor_server.adapters.ome_tiff import _store_pool
from biopb_tensor_server.fixtures import create_multi_series_ome_tiff


def test_concurrent_reads_are_correct(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOPB_OMETIFF_PARALLEL_READ", "1")
    # Each series is filled with series_idx*100 + plane + 1; shape TCZYX =
    # (1, 2, 1, 64, 64), so series k channel c corner == k*100 + c + 1.
    path, _, _ = create_multi_series_ome_tiff(
        str(tmp_path), n_series=3, series_shape=(2, 64, 64)
    )
    source = OmeTiffAdapter(path, "conc")
    fields = [d.array_id.split("/", 1)[1] for d in source.list_tensors()]
    scenes = [source.get_tensor_adapter(f) for f in fields]

    def read(task):
        k, c = task
        bounds = ChunkBounds(start=[0, c, 0, 0, 0], stop=[1, c + 1, 1, 64, 64])
        arr = np.asarray(scenes[k].get_data(bounds))
        return int(arr.ravel()[0]) == k * 100 + c + 1

    tasks = [(k, c) for k in range(3) for c in range(2)] * 100  # 600 reads
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(read, tasks))

    assert all(results)
    assert all(h.leases == 0 for h in _store_pool._handles.values())
    source.close()


def test_default_reads_are_correct_and_serialized(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOPB_OMETIFF_PARALLEL_READ", raising=False)
    path, _, _ = create_multi_series_ome_tiff(
        str(tmp_path), n_series=3, series_shape=(2, 64, 64)
    )
    source = OmeTiffAdapter(path, "serial")
    fields = [d.array_id.split("/", 1)[1] for d in source.list_tensors()]
    scenes = [source.get_tensor_adapter(f) for f in fields]

    def read(task):
        k, c = task
        bounds = ChunkBounds(start=[0, c, 0, 0, 0], stop=[1, c + 1, 1, 64, 64])
        arr = np.asarray(scenes[k].get_data(bounds))
        return int(arr.ravel()[0]) == k * 100 + c + 1

    tasks = [(k, c) for k in range(3) for c in range(2)] * 50  # 300 reads
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(read, tasks))

    assert all(results)
    assert all(h.leases == 0 for h in _store_pool._handles.values())
    source.close()


def _parked_read(scene):
    """Start a read of *scene* parked inside its decode; return (thread, release)."""
    entered = threading.Event()
    release = threading.Event()
    orig_read_region = scene._read_region

    def parked(za, axes, slices):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return orig_read_region(za, axes, slices)

    scene._read_region = parked
    bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 64, 64])
    t = threading.Thread(target=lambda: scene.get_data(bounds))
    t.start()
    assert entered.wait(5), "read never reached _read_region"
    return t, release, bounds


def _scene(tmp_path, source_id):
    path, _, _ = create_multi_series_ome_tiff(
        str(tmp_path), n_series=1, series_shape=(2, 64, 64)
    )
    source = OmeTiffAdapter(path, source_id)
    field = source.list_tensors()[0].array_id.split("/", 1)[1]
    return source, source.get_tensor_adapter(field)


def test_a_read_in_flight_keeps_its_store_open(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOPB_OMETIFF_PARALLEL_READ", raising=False)
    source, scene = _scene(tmp_path, "leased")
    t, release, _ = _parked_read(scene)
    try:
        handle = _store_pool._handles[scene._pool_key()]
        assert handle.leases == 1
        _store_pool.sweep()
        source.close()  # drops the key: closes at the last lease, not now
        assert scene._pool_key() in _store_pool._handles
    finally:
        release.set()
        t.join(5)
    assert scene._pool_key() not in _store_pool._handles


def test_lock_free_reads_decode_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOPB_OMETIFF_PARALLEL_READ", "1")
    source, scene = _scene(tmp_path, "lockfree")
    t, release, bounds = _parked_read(scene)
    try:
        assert scene.get_data(bounds) is not None  # not blocked by the parked read
    finally:
        release.set()
        t.join(5)
    source.close()


def test_default_reads_of_one_store_serialize(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOPB_OMETIFF_PARALLEL_READ", raising=False)
    source, scene = _scene(tmp_path, "serial")
    t, release, bounds = _parked_read(scene)
    second = threading.Thread(target=lambda: scene.get_data(bounds))
    second.start()
    try:
        second.join(0.3)
        assert second.is_alive(), "second read ran while the first held the store"
    finally:
        release.set()
        t.join(5)
        second.join(5)
    assert not second.is_alive()
    source.close()
