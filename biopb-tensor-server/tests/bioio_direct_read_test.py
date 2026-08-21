import threading
from types import SimpleNamespace

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.bioio import NikonAdapter

nd2 = pytest.importorskip("nd2")


class _ZeroFrame:
    """Shape/dtype only -- the grid tests never touch pixels, and materializing
    the 4096x4096x4 geometry they need would cost 134 MB for nothing."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeDaskArray:
    def __init__(self, data: np.ndarray, chunks: tuple[tuple[int, ...], ...]):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return SimpleNamespace(compute=lambda: self._data[item].copy())


class _FakeBioImage:
    def __init__(self, data: np.ndarray, labels: str, chunks):
        self.dask_data = _FakeDaskArray(data, chunks)
        self.dims = SimpleNamespace(order=labels)
        self._xarray_dask_data = self.dask_data
        self._dims = self.dims
        self.reader = SimpleNamespace(
            _xarray_dask_data=self.dask_data,
            _dims=self.dims,
        )

    def set_scene(self, _scene_index: int) -> None:
        pass


class _FakeND2File:
    frames: list[np.ndarray] = []
    loop_indices_value: tuple[dict[str, int], ...] = ()
    loop_indices_accesses = 0
    closed = False

    def __init__(self, _path: str):
        type(self).closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        type(self).closed = True

    def read_frame(self, frame_index: int) -> np.ndarray:
        return self.frames[frame_index]

    @property
    def loop_indices(self):
        type(self).loop_indices_accesses += 1
        return self.loop_indices_value


def _adapter(tmp_path, data, labels, chunks):
    path = tmp_path / "source.nd2"
    path.touch()
    return NikonAdapter(
        _FakeBioImage(data, labels, chunks),
        scene_index=1,
        source_id="source",
        source_url=str(path),
    )


def test_nikon_direct_read_maps_scene_t_and_z_and_copies_crop(tmp_path, monkeypatch):
    shape = (2, 2, 3, 4, 5)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    frames = []
    loops = []
    for position in range(2):
        for time in range(shape[0]):
            for z in range(shape[2]):
                loops.append({"P": position, "T": time, "Z": z})
                frames.append(data[time, :, z].copy() + position * 10_000)
    _FakeND2File.frames = frames
    _FakeND2File.loop_indices_value = tuple(loops)
    _FakeND2File.loop_indices_accesses = 0
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)

    adapter = _adapter(
        tmp_path,
        data + 10_000,
        "TCZYX",
        ((2,), (2,), (3,), (4,), (5,)),
    )
    assert adapter._dask_data is None
    assert adapter._scene_descriptor is not None
    assert adapter._bio_image._xarray_dask_data is None
    assert adapter._bio_image.reader._xarray_dask_data is None
    adapter._tensor_name = "scene-1"
    assert adapter.get_tensor_descriptor().array_id == "source/scene-1"
    bounds = ChunkBounds(start=[0, 1, 1, 1, 2], stop=[2, 2, 3, 4, 5])

    actual = adapter.get_data(bounds)
    adapter.get_data(bounds)

    expected = (data + 10_000)[0:2, 1:2, 1:3, 1:4, 2:5]
    assert np.array_equal(actual, expected)
    assert actual.flags.owndata
    assert _FakeND2File.closed
    assert _FakeND2File.loop_indices_accesses == 1
    # The frame ND2 hands back -- all channels, whole Y/X, one T and one Z --
    # seeds the transfer grid; the grid is whole frames of it (biopb/biopb#809).
    grid = list(adapter.get_tensor_descriptor().chunk_shape)
    assert [grid[1], grid[3], grid[4]] == [2, 4, 5]


def test_nikon_direct_read_preserves_rgb_samples(tmp_path, monkeypatch):
    shape = (1, 1, 1, 3, 4, 3)
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    _FakeND2File.frames = [data[0, :, 0].copy()]
    _FakeND2File.loop_indices_value = ({"P": 1},)
    _FakeND2File.loop_indices_accesses = 0
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)
    adapter = _adapter(
        tmp_path,
        data,
        "TCZYXS",
        ((1,), (1,), (1,), (3,), (4,), (3,)),
    )

    actual = adapter.get_data(
        ChunkBounds(start=[0, 0, 0, 1, 1, 1], stop=[1, 1, 1, 3, 4, 3])
    )

    assert np.array_equal(actual, data[:, :, :, 1:3, 1:4, 1:3])


def test_nikon_unrepresented_sequence_loop_falls_back_to_bioio(tmp_path, monkeypatch):
    # Two frames of an unknown loop collapse onto the same (T, Z), so the
    # direct path cannot tell them apart. The zeroed frames stand in for the
    # arbitrary pick it must not make; BioIO's array carries the real pixels.
    data = np.arange(2 * 2, dtype=np.uint8).reshape(1, 1, 1, 2, 2) + 1
    _FakeND2File.frames = [np.zeros((1, 1, 2, 2), np.uint8)] * 2
    _FakeND2File.loop_indices_value = ({"P": 1, "M": 0}, {"P": 1, "M": 1})
    _FakeND2File.loop_indices_accesses = 0
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)
    adapter = _adapter(
        tmp_path,
        data,
        "TCZYX",
        ((1,), (1,), (1,), (2,), (2,)),
    )

    bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 2, 2])
    actual = adapter.get_data(bounds)

    assert np.array_equal(actual, data)
    # The ambiguity verdict caches too: no second walk of the whole file.
    adapter.get_data(bounds)
    assert _FakeND2File.loop_indices_accesses == 1


def test_nikon_missing_tz_coordinate_falls_back_to_bioio(tmp_path, monkeypatch):
    data = np.arange(2 * 2 * 2, dtype=np.uint8).reshape(2, 1, 1, 2, 2)
    _FakeND2File.frames = [data[0, :, 0]]
    _FakeND2File.loop_indices_value = ({"P": 1, "T": 0},)
    _FakeND2File.loop_indices_accesses = 0
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)
    adapter = _adapter(
        tmp_path,
        data,
        "TCZYX",
        ((1, 1), (1,), (1,), (2,), (2,)),
    )

    actual = adapter.get_data(ChunkBounds(start=[1, 0, 0, 0, 0], stop=[2, 1, 1, 2, 2]))

    assert np.array_equal(actual, data[1:2])


def test_nikon_metadata_failure_degrades_to_empty(tmp_path):
    image = _FakeBioImage(
        np.zeros((1, 1, 1, 2, 2), dtype=np.uint8),
        "TCZYX",
        ((1,), (1,), (1,), (2,), (2,)),
    )
    adapter = NikonAdapter(image, scene_index=None, source_id="source")

    assert adapter.get_metadata() == {}
    assert adapter._physical_scale() is None


def test_scene_switch_does_not_race_a_concurrent_read():
    """A descriptor listing must not repoint the scene under a running read.

    list_tensor_descriptors walks every scene, leaving BioIO bound to the last
    one. A reader bound to scene 0 that does not hold _io_lock across its own
    set_scene and array access would come back with scene 1's pixels.
    """
    import time

    scenes = {
        0: np.full((1, 1, 1, 2, 2), 7, np.uint8),
        1: np.full((1, 1, 1, 2, 2), 9, np.uint8),
    }
    chunks = ((1,), (1,), (1,), (2,), (2,))
    reader_is_mid_read = threading.Event()
    reader_thread = None

    class _SceneSwitchingBioImage(_FakeBioImage):
        def __init__(self):
            super().__init__(scenes[0], "TCZYX", chunks)
            self.scenes = ["s0", "s1"]
            self.ome_metadata = None

        def set_scene(self, scene_index):
            index = (
                self.scenes.index(scene_index)
                if isinstance(scene_index, str)
                else scene_index
            )
            self.dask_data = _FakeDaskArray(scenes[index], chunks)
            if threading.current_thread() is reader_thread:
                # Widen the window between binding this scene's lazy array and
                # the caller reading it, where an unlocked lister interleaves
                # and leaves the binding on its own last scene.
                reader_is_mid_read.set()
                time.sleep(0.2)

    image = _SceneSwitchingBioImage()
    source = NikonAdapter(image, scene_index=None, source_id="source")
    reader = NikonAdapter(
        image, scene_index=0, source_id="source", io_lock=source._io_lock
    )

    observed = []

    def _read():
        observed.append(
            reader._get_data_via_bioio(
                ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 2, 2])
            )
        )

    def _list():
        reader_is_mid_read.wait(timeout=5)
        source.list_tensor_descriptors()

    reader_thread = threading.Thread(target=_read)
    threads = [reader_thread, threading.Thread(target=_list)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads), "deadlocked"
    assert observed and np.array_equal(observed[0], scenes[0])


def test_component_axes_are_never_split(tmp_path, monkeypatch):
    """C and S live inside the pixel, so a chunk always holds all of them.

    ND2 has no planar variant: every frame is `(Y, X, channel, RGB component)`,
    so one channel's bytes are `itemsize` of every `componentCount * itemsize`.
    A per-channel chunk faults in every page the others occupy and discards what
    it did not ask for (biopb/biopb#806). There is nothing to detect and no
    probe -- the layout is a property of the format.
    """
    data = _ZeroFrame((1, 4, 1, 4096, 4096), np.dtype("uint16"))
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)

    adapter = _adapter(tmp_path, data, "TCZYX", ((1,), (4,), (1,), (4096,), (4096,)))
    grid = list(adapter.get_tensor_descriptor().chunk_shape)

    # All four channels in one chunk, and a square plane: Y and X are coupled,
    # so a tile read touches one chunk rather than one per band it crosses.
    assert grid == [1, 4, 1, 1024, 1024]

    from biopb_tensor_server.core.chunk import (
        PREFERRED_ARROW_BATCH_BYTES,
        estimate_chunk_bytes,
    )

    assert estimate_chunk_bytes(tuple(grid), "<u2") <= PREFERRED_ARROW_BATCH_BYTES


def test_rgb_samples_are_never_split(tmp_path, monkeypatch):
    """The RGB axis is the same story: nd2 puts it inside the pixel too.

    ``components_per_channel == 3`` makes nd2 report an ``S`` axis rather than
    folding the components into ``C`` -- a two-channel RGB file is ``C=2, S=3``,
    six interleaved components per pixel -- so both axes have to stay whole.
    """
    data = _ZeroFrame((1, 2, 1, 2048, 2048, 3), np.dtype("uint8"))
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)

    adapter = _adapter(
        tmp_path,
        data,
        "TCZYXS",
        ((1,), (2,), (1,), (2048,), (2048,), (3,)),
    )
    grid = list(adapter.get_tensor_descriptor().chunk_shape)

    assert grid[1] == 2 and grid[5] == 3  # C and S whole
    assert grid[3] == grid[4]  # plane square


def test_single_component_nd2_uses_the_backend_block(tmp_path, monkeypatch):
    """``componentCount == 1`` has no interleaved run to protect.

    nd2 drops size-1 axes, so a mono single-channel file has neither C nor S and
    the backend's own block is the better seed.
    """
    data = _ZeroFrame((1, 1, 1, 4096, 4096), np.dtype("uint16"))
    monkeypatch.setattr(nd2, "ND2File", _FakeND2File)

    adapter = _adapter(tmp_path, data, "TCZYX", ((1,), (1,), (1,), (4096,), (4096,)))
    grid = list(adapter.get_tensor_descriptor().chunk_shape)
    assert grid[3] == grid[4]  # still a coupled, square plane
    assert grid[1] == 1
