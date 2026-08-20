from types import SimpleNamespace

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.bioio import NikonAdapter

nd2 = pytest.importorskip("nd2")


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
    assert list(adapter.get_tensor_descriptor().chunk_shape) == [1, 2, 1, 4, 5]


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
