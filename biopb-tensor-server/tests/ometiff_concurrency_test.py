"""Concurrency of the OmeTiffAdapter read path.

The scene read holds ``_io_lock`` only to acquire the store + register the read as
in-flight, then decodes WITHOUT the lock (tifffile serializes the raw seek+read on
its own handle lock; the tile decode is per-tile into a fresh buffer). These tests
pin that reads are (a) correct under heavy concurrency and (b) genuinely lock-free
during decode, with the ``_active_reads`` counter guarding teardown.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import OmeTiffAdapter
from biopb_tensor_server.fixtures import create_multi_series_ome_tiff


def test_concurrent_reads_are_correct(tmp_path):
    # Each series is filled with series_idx*100 + plane + 1; shape TCZYX =
    # (1, 2, 1, 64, 64), so series k channel c corner == k*100 + c + 1.
    path, _, _ = create_multi_series_ome_tiff(
        str(tmp_path), n_series=3, series_shape=(2, 64, 64)
    )
    source = OmeTiffAdapter(path, "conc")
    fields = [d.array_id.split("/", 1)[1] for d in source.list_tensor_descriptors()]
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
    # Every in-flight read was accounted for -- the counter returns to zero.
    for scene in scenes:
        assert scene._active_reads == 0
    source.close()


def test_read_does_not_hold_io_lock_during_decode(tmp_path):
    # Park a read inside _read_region and prove _io_lock is free while it runs and
    # that the read is counted in-flight -- i.e. the decode is not serialized.
    path, _, _ = create_multi_series_ome_tiff(
        str(tmp_path), n_series=1, series_shape=(2, 64, 64)
    )
    source = OmeTiffAdapter(path, "lockfree")
    field = source.list_tensor_descriptors()[0].array_id.split("/", 1)[1]
    scene = source.get_tensor_adapter(field)

    entered = threading.Event()
    release = threading.Event()
    orig_read_region = scene._read_region

    def parked(za, axes, slices):
        entered.set()
        assert release.wait(5)
        return orig_read_region(za, axes, slices)

    scene._read_region = parked

    bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 64, 64])
    t = threading.Thread(target=lambda: scene.get_data(bounds))
    t.start()
    try:
        assert entered.wait(5), "read never reached _read_region"
        # The read is parked mid-decode: _io_lock must be acquirable (not held)...
        acquired = scene._io_lock.acquire(blocking=False)
        assert acquired, "_io_lock held during decode -- read is not lock-free"
        scene._io_lock.release()
        # ...and it must be registered as in-flight so the reaper won't close it.
        assert scene._active_reads == 1
    finally:
        release.set()
        t.join(5)

    assert scene._active_reads == 0
    source.close()
