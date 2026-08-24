"""Every adapter classifies whether it can stride its own read.

``get_decimated_data`` is the whole of the fused ``nearest`` path
(biopb/biopb#640): a pick needs none of the elements it skips, so a backend
that can express the stride never reads them. Declining is free and correct --
the caller reads the extent and strides it, which is what it did before -- so
the failure this guards is the same one ``adapter_read_block_test`` guards: a
new adapter inheriting the base ``None`` without anyone deciding it should.

The two lists are near-inverses of the read-block ones, and that is the point:
a backend with a read block has to decode the whole block to hand back any of
it, so a stride inside it saves nothing to read. Where the lists disagree, the
reason is stated.
"""

import inspect

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.downsample import downsample_block

# A strided read costs what it returns, not what it spans.
DECIMATING = {
    "MrcAdapter": "memmap indexing computes byte offsets; the copy shrinks",
    "NiftiAdapter": "nibabel fileslice plans the read from the slice itself",
    "NikonAdapter": "the step selects frames on T/Z and strides the mmap view",
}

# Reads the extent and strides it -- the base's own path.
NOT_DECIMATING = {
    "ZarrAdapter": "a chunk decodes whole; a stride inside it saves no read",
    "OmeZarrAdapter": "inherits ZarrAdapter",
    "_HcsFieldAdapter": "inherits ZarrAdapter",
    "_QptiffLevelAdapter": "inherits ZarrAdapter",
    "Hdf5Adapter": "h5py hyperslab still reads whole chunks",
    "OmeTiffAdapter": "a page decodes whole (aszarr chunkmode='page')",
    "_TifffileAdapterBase": "inherits OmeTiffAdapter",
    "TiffAdapter": "inherits OmeTiffAdapter",
    "LsmAdapter": "inherits OmeTiffAdapter",
    "TiffSequenceAdapter": "a strile decodes whole",
    "MicroManagerLegacyAdapter": "a strile decodes whole",
    "DicomAdapter": "a frame decodes whole",
    "DicomSeriesAdapter": "a slice file decodes whole",
    "NdTiffAdapter": "the dask block behind the plane materialises whole",
    "EmdAdapter": "the dask block materialises whole",
    "_BioioAdapterBase": "the dask block a slice materialises",
    "ZeissAdapter": "inherits _BioioAdapterBase",
    "LeicaAdapter": "inherits _BioioAdapterBase",
    "DvAdapter": "inherits _BioioAdapterBase",
    "OlympusAdapter": "inherits _BioioAdapterBase",
    "BioformatsAdapter": "inherits _BioioAdapterBase",
    "AicsImageIoAdapter": "inherits _BioioAdapterBase",
    # Unquantized in adapter_read_block_test, and still not decimating:
    "CziAdapter": "libCZI takes a zoom, not a stride -- a separate mechanism",
    "RemoteTensorAdapter": "ChunkBounds carries no step; the scale rides upstream",
    "CachedSourceAdapter": "get_data raises; served by chunk_id only",
    "QptiffAdapter": "source-level; serves levels through _QptiffLevelAdapter",
    "TensorAdapter": "the base class itself",
}


def _adapters():
    import biopb_tensor_server.adapters as pkg

    seen = {}
    for name in dir(pkg):
        obj = getattr(pkg, name)
        if inspect.isclass(obj) and issubclass(obj, TensorAdapter):
            seen[obj.__name__] = obj
    for module in pkg.__dict__.values():
        if inspect.ismodule(module):
            for obj in vars(module).values():
                if inspect.isclass(obj) and issubclass(obj, TensorAdapter):
                    seen[obj.__name__] = obj
    return seen


def test_every_adapter_is_classified():
    classified = set(DECIMATING) | set(NOT_DECIMATING)
    found = set(_adapters())
    assert found - classified == set(), (
        f"unclassified adapters: {sorted(found - classified)} -- implement "
        "get_decimated_data and add them to DECIMATING, or add them to "
        "NOT_DECIMATING with the reason a stride buys them nothing"
    )


def test_the_lists_are_disjoint_and_real():
    assert not (set(DECIMATING) & set(NOT_DECIMATING))
    stale = (
        (set(DECIMATING) | set(NOT_DECIMATING)) - set(_adapters()) - {"TensorAdapter"}
    )
    assert not stale, f"listed but no longer present: {sorted(stale)}"


@pytest.mark.parametrize("name", sorted(NOT_DECIMATING))
def test_non_decimating_adapters_inherit_the_base(name):
    """Not merely absent: no class between here and the base may declare it."""
    cls = _adapters().get(name)
    if cls is None:
        pytest.skip(f"{name} not importable in this environment")
    owner = next((k for k in cls.__mro__ if "get_decimated_data" in vars(k)), None)
    assert owner is TensorAdapter, f"{name} declares get_decimated_data"


@pytest.mark.parametrize("name", sorted(DECIMATING))
def test_decimating_adapters_declare_it(name):
    cls = _adapters().get(name)
    if cls is None:
        pytest.skip(f"{name} not importable in this environment")
    owner = next((k for k in cls.__mro__ if "get_decimated_data" in vars(k)), None)
    assert owner is not None and owner is not TensorAdapter, (
        f"{name} inherits the base None but is listed as decimating"
    )


def test_the_base_declines():
    """The default is to decline, which is what keeps a new adapter correct."""
    assert TensorAdapter.get_decimated_data(None, None, None) is None


class TestDecimatedEqualsReadThenStride:
    """The contract, on real adapters: same pixels, same shape, owned array.

    ``downsample_block(data, scale, "nearest")`` is ``data[::scale]``, so this
    is the whole correctness argument for the fused path -- there is no
    accumulator and no dtype promotion to reason about, unlike ``area``.
    """

    @staticmethod
    def _check(adapter, start, stop, step):
        bounds = ChunkBounds(start=list(start), stop=list(stop))
        expected = downsample_block(adapter.get_data(bounds), tuple(step), "nearest")
        picked = adapter.get_decimated_data(bounds, tuple(step))

        assert picked is not None
        assert picked.shape == expected.shape
        assert picked.dtype == expected.dtype
        assert np.array_equal(picked, expected)
        # Rule 3: owned. A view onto a mapping outlives nothing safely.
        assert picked.base is None
        return picked

    def test_mrc(self, tmp_path):
        pytest.importorskip("rsciio")
        from biopb_tensor_server.adapters.mrc import MrcAdapter
        from biopb_tensor_server.core.config import SourceConfig

        from .mrc_test import create_synthetic_mrc

        path = tmp_path / "volume.mrc"
        create_synthetic_mrc(path, shape=(6, 12, 14), dtype=np.float32)
        adapter = MrcAdapter.create_from_config(SourceConfig(url=str(path)))
        try:
            self._check(adapter, (0, 0, 0), (6, 12, 14), (2, 3, 4))
            # An extent that is not a multiple of the step, off the origin.
            self._check(adapter, (1, 2, 3), (6, 11, 14), (2, 2, 5))
        finally:
            adapter.close()

    def test_nifti(self, tmp_path):
        nib = pytest.importorskip("nibabel")
        from biopb_tensor_server.adapters.nifti import NiftiAdapter

        from .nifti_test import create_synthetic_nifti

        path = tmp_path / "volume.nii"
        create_synthetic_nifti(path, shape=(16, 12, 8), dtype=np.float32)
        adapter = NiftiAdapter(nib.load(str(path)), "nifti")
        try:
            self._check(adapter, (0, 0, 0), (16, 12, 8), (4, 3, 2))
            self._check(adapter, (3, 1, 0), (15, 12, 7), (5, 4, 3))
        finally:
            adapter.close()

    def test_nd2(self, tmp_path, monkeypatch):
        nd2 = pytest.importorskip("nd2")

        from .bioio_direct_read_test import _adapter, _FakeND2File

        shape = (2, 2, 6, 9, 11)  # T C Z Y X
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        frames, loops = [], []
        for position in range(2):
            for time in range(shape[0]):
                for z in range(shape[2]):
                    loops.append({"P": position, "T": time, "Z": z})
                    frames.append(data[time, :, z].copy() + position * 10_000)
        _FakeND2File.frames = frames
        _FakeND2File.loop_indices_value = tuple(loops)
        monkeypatch.setattr(nd2, "ND2File", _FakeND2File)

        adapter = _adapter(
            tmp_path, data + 10_000, "TCZYX", ((2,), (2,), (6,), (9,), (11,))
        )
        try:
            # A step on Z picks frames; one on Y/X strides inside them.
            self._check(adapter, (0, 0, 0, 0, 0), shape, (1, 1, 2, 3, 4))
            self._check(adapter, (0, 1, 1, 2, 3), (2, 2, 5, 9, 10), (2, 1, 2, 4, 3))
        finally:
            adapter.close()

    def test_nd2_reads_only_the_frames_the_step_lands_on(self, tmp_path, monkeypatch):
        """The half a pixel comparison cannot see: frames never decoded.

        Striding inside a frame saves a memcpy; stepping over one saves the
        decode entirely, and on a deep Z stack that is most of the read.
        """
        nd2 = pytest.importorskip("nd2")

        from .bioio_direct_read_test import _adapter, _FakeND2File

        shape = (1, 1, 8, 4, 4)
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        frames, loops = [], []
        for position in range(2):
            for z in range(shape[2]):
                loops.append({"P": position, "T": 0, "Z": z})
                frames.append(data[0, :, z].copy() + position * 10_000)
        _FakeND2File.frames = frames
        _FakeND2File.loop_indices_value = tuple(loops)
        monkeypatch.setattr(nd2, "ND2File", _FakeND2File)

        adapter = _adapter(
            tmp_path, data + 10_000, "TCZYX", ((1,), (1,), (8,), (4,), (4,))
        )
        read = []
        original = _FakeND2File.read_frame
        monkeypatch.setattr(
            _FakeND2File,
            "read_frame",
            lambda self, index: (read.append(index), original(self, index))[1],
        )
        try:
            bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=list(shape))
            adapter.get_decimated_data(bounds, (1, 1, 4, 2, 2))
            assert len(read) == 2, f"decoded {len(read)} frames for a scale-4 Z step"
        finally:
            adapter.close()


class TestTheSeam:
    """``get_scaled_data`` routes ``nearest`` here, and only ``nearest``."""

    @pytest.fixture
    def adapter(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        from biopb_tensor_server import ZarrAdapter

        src = (np.arange(64 * 64, dtype=np.uint16) % 4093).reshape(64, 64)
        store = str(tmp_path / "a.zarr")
        arr = zarr.open_array(
            store, mode="w", shape=(64, 64), chunks=(32, 32), dtype="uint16"
        )
        arr[:] = src
        return ZarrAdapter(zarr.open_array(store, mode="r"), "src", ["y", "x"])

    @staticmethod
    def _bounds():
        return ChunkBounds(start=[0, 0], stop=[64, 64])

    def test_nearest_asks_the_adapter_first(self, adapter, monkeypatch):
        seen = {}

        def decimated(bounds, step):
            seen["step"] = step
            return downsample_block(adapter.get_data(bounds), step, "nearest").copy()

        monkeypatch.setattr(adapter, "get_decimated_data", decimated)
        out = adapter.get_scaled_data(self._bounds(), (4, 4), "nearest")

        assert seen["step"] == (4, 4)
        assert np.array_equal(
            out, downsample_block(adapter.get_data(self._bounds()), (4, 4), "nearest")
        )

    def test_area_never_asks(self, adapter, monkeypatch):
        """``area`` must visit every element, so a pick would be a wrong answer."""

        def explode(bounds, step):
            raise AssertionError("area must not reach get_decimated_data")

        monkeypatch.setattr(adapter, "get_decimated_data", explode)
        adapter.get_scaled_data(self._bounds(), (4, 4), "area")

    def test_an_unscaled_nearest_does_not_ask(self, adapter, monkeypatch):
        """Step 1 is the extent itself -- nothing to skip, nothing to gain."""

        def explode(bounds, step):
            raise AssertionError("scale 1 must not reach get_decimated_data")

        monkeypatch.setattr(adapter, "get_decimated_data", explode)
        adapter.get_scaled_data(self._bounds(), (1, 1), "nearest")

    def test_declining_leaves_the_streamed_answer_unchanged(self, adapter):
        """The default: the same pixels the tiled path produces."""
        assert adapter.get_decimated_data(self._bounds(), (4, 4)) is None
        out = adapter.get_scaled_data(self._bounds(), (4, 4), "nearest")
        assert np.array_equal(
            out, downsample_block(adapter.get_data(self._bounds()), (4, 4), "nearest")
        )
