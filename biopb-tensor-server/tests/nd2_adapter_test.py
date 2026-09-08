"""Phase 3 coverage for the native Nikon ND2 adapter (biopb/biopb#799).

``nd2`` ships no writer and there is no small real ND2 sample checked in
(see ``tests/bioio_direct_read_test.py`` for the same constraint on BioIO's
``NikonAdapter``), so these tests drive the adapter against a fake
``nd2.ND2File`` double that reproduces the properties this module reads:
``sizes``, ``dtype``, ``voxel_size()``, ``ome_metadata()``, ``loop_indices``
and ``read_frame``.
"""

from pathlib import Path

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import (
    ClaimContext,
    DiscoveryState,
    LiveLocalContext,
)

pytest.importorskip("nd2")

import nd2  # noqa: E402
from biopb_tensor_server.adapters import (  # noqa: E402
    Nd2Adapter,
    get_default_registry,
    nd2 as nd2_module,  # noqa: E402
)
from biopb_tensor_server.adapters.nd2 import read_layout  # noqa: E402


class _VoxelSize:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeOme:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="json"):
        return self._payload


_DEFAULT_DTYPE = np.dtype("<u2")


class _FakeND2File:
    """A minimal double for the ``nd2.ND2File`` surface this module reads.

    ``sizes`` is (P, T, Z, C, Y, X) by default -- every loop axis present, so
    the frame-index / decimation paths all get exercised. Pass ``sizes`` to
    drop axes, matching how a real single-position or single-timepoint file
    reports fewer of them (``ND2File.sizes`` omits size-1 axes).
    """

    closed = False
    opens = 0

    def __init__(
        self,
        path,
        sizes=None,
        dtype=_DEFAULT_DTYPE,
        voxel=(0.1, 0.2, 0.5),
        loop_indices=None,
        ome_payload=None,
    ):
        type(self).opens += 1
        self.path = path
        self._sizes = sizes or {"P": 2, "T": 3, "Z": 2, "C": 2, "Y": 4, "X": 5}
        self._dtype = dtype
        self._voxel = voxel
        self._loop_indices = loop_indices
        self._ome_payload = ome_payload if ome_payload is not None else {"images": []}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True

    @property
    def sizes(self):
        return dict(self._sizes)

    @property
    def dtype(self):
        return self._dtype

    def voxel_size(self, channel=0):
        return _VoxelSize(*self._voxel)

    def ome_metadata(self, **kwargs):
        return _FakeOme(self._ome_payload)

    @property
    def loop_indices(self):
        if self._loop_indices is not None:
            return self._loop_indices
        present = [axis for axis in ("P", "T", "Z") if axis in self._sizes]
        ranges = [range(self._sizes[axis]) for axis in present]
        from itertools import product

        return tuple(
            dict(zip(present, combo, strict=True)) for combo in product(*ranges)
        ) or ({},)

    def read_frame(self, frame_index):
        coords = self.loop_indices[frame_index]
        c = self._sizes.get("C", 1)
        y, x = self._sizes["Y"], self._sizes["X"]
        base = (
            coords.get("P", 0) * 1000
            + coords.get("T", 0) * 100
            + coords.get("Z", 0) * 10
        )
        frame = np.zeros((c, y, x), dtype=self._dtype)
        for channel in range(c):
            frame[channel] = base + channel
        return frame


def _install_fake(monkeypatch, **kwargs):
    def factory(path):
        return _FakeND2File(path, **kwargs)

    monkeypatch.setattr(nd2, "ND2File", factory)
    return factory


def _source(path, source_id="nd2", **kwargs):
    return SourceConfig(url=str(path), type="nd2", source_id=source_id, **kwargs)


class _RemoteCtx(LiveLocalContext):
    @property
    def is_remote(self) -> bool:
        return True


def _expected(sizes, present):
    shape = tuple(
        sizes[axis] for axis in ("P", "T", "Z", "C", "Y", "X") if axis in sizes
    )
    labels = [axis for axis in ("P", "T", "Z", "C", "Y", "X") if axis in sizes]
    out = np.zeros(shape, dtype=np.uint16)
    from itertools import product

    ranges = [range(sizes[axis]) for axis in present]
    for combo in product(*ranges):
        by_axis = dict(zip(present, combo, strict=True))
        base = (
            by_axis.get("P", 0) * 1000
            + by_axis.get("T", 0) * 100
            + by_axis.get("Z", 0) * 10
        )
        index = tuple(
            by_axis[axis] if axis in present else slice(None) for axis in labels
        )
        c_pos = labels.index("C")
        for c in range(sizes.get("C", 1)):
            idx = list(index)
            idx[c_pos] = c
            out[tuple(idx)] = base + c
    return out, labels


def test_local_nd2_claims_natively_and_reads_through_nd2_package(tmp_path, monkeypatch):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(monkeypatch)

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(path), DiscoveryState())
    assert [c.source_type for c in claims] == ["nd2"]

    source = registry.get_adapter_for_type("nd2").create_from_config(_source(path))
    assert isinstance(source, Nd2Adapter)

    descriptors = source.list_tensor_descriptors()
    assert len(descriptors) == 1
    desc = descriptors[0]
    assert list(desc.dim_labels) == ["P", "T", "Z", "C", "Y", "X"]
    assert list(desc.shape) == [2, 3, 2, 2, 4, 5]
    assert desc.dtype == np.dtype("<u2").str

    expected, _ = _expected(
        dict(zip(desc.dim_labels, desc.shape, strict=True)), ("P", "T", "Z")
    )
    whole = source.get_data(ChunkBounds(start=[0] * 6, stop=list(desc.shape)))
    np.testing.assert_array_equal(whole, expected)


def test_interior_crop_reads_only_the_requested_window(tmp_path, monkeypatch):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(monkeypatch)
    source = Nd2Adapter.create_from_config(_source(path))
    sizes = dict(zip(source.dim_labels, source._layout.shape, strict=True))
    expected, _ = _expected(sizes, ("P", "T", "Z"))

    bounds = ChunkBounds(start=[0, 1, 0, 0, 1, 1], stop=[2, 3, 2, 2, 3, 4])
    got = source.get_data(bounds)
    np.testing.assert_array_equal(got, expected[0:2, 1:3, 0:2, 0:2, 1:3, 1:4])


def test_decimated_read_matches_a_strided_slice_of_the_full_read(tmp_path, monkeypatch):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(monkeypatch)
    source = Nd2Adapter.create_from_config(_source(path))
    sizes = dict(zip(source.dim_labels, source._layout.shape, strict=True))
    expected, _ = _expected(sizes, ("P", "T", "Z"))

    bounds = ChunkBounds(start=[0] * 6, stop=list(source._layout.shape))
    step = (1, 2, 1, 1, 1, 2)
    decimated = source.get_decimated_data(bounds, step)
    np.testing.assert_array_equal(decimated, expected[:, ::2, :, :, :, ::2])


def test_single_position_file_drops_the_p_axis(tmp_path, monkeypatch):
    """``ND2File.sizes`` omits size-1 axes; the descriptor follows."""
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(monkeypatch, sizes={"T": 3, "Z": 2, "C": 1, "Y": 4, "X": 5})
    source = Nd2Adapter.create_from_config(_source(path))
    desc = source.get_tensor_descriptor()
    assert "P" not in desc.dim_labels
    assert list(desc.dim_labels) == ["T", "Z", "C", "Y", "X"]

    whole = source.get_data(ChunkBounds(start=[0] * 5, stop=list(desc.shape)))
    assert whole.shape == tuple(desc.shape)


def test_ambiguous_loop_coordinate_is_rejected_at_registration(tmp_path, monkeypatch):
    """Two frames sharing a (P, T, Z) coordinate is an error, not a guess --
    there is no BioIO fallback left to hand it to."""
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    collision = ({"P": 0, "T": 0}, {"P": 0, "T": 0})
    _install_fake(
        monkeypatch,
        sizes={"P": 1, "T": 1, "C": 1, "Y": 2, "X": 2},
        loop_indices=collision,
    )
    with pytest.raises(ValueError, match="not addressable"):
        read_layout(str(path))


def test_physical_scale_reports_nd2_voxel_size(tmp_path, monkeypatch):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(monkeypatch, voxel=(0.11, 0.22, 0.33))
    source = Nd2Adapter.create_from_config(_source(path))
    scale, unit = source._physical_scale()
    labels = source.dim_labels
    assert scale[labels.index("X")] == pytest.approx(0.11)
    assert scale[labels.index("Y")] == pytest.approx(0.22)
    assert scale[labels.index("Z")] == pytest.approx(0.33)
    assert unit[labels.index("X")] == "µm"


def test_get_metadata_returns_the_ome_summary(tmp_path, monkeypatch):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(monkeypatch, ome_payload={"images": [{"id": "Image:0"}]})
    source = Nd2Adapter.create_from_config(_source(path))
    assert source.get_metadata() == {"images": [{"id": "Image:0"}]}


def test_component_axes_are_never_split_across_a_chunk(tmp_path, monkeypatch):
    """C sits below X in the interleaved frame (biopb/biopb#806): the transfer
    grid must never divide it."""
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _install_fake(
        monkeypatch,
        sizes={"T": 50, "C": 4, "Y": 512, "X": 512},
        dtype=np.dtype("<u2"),
    )
    source = Nd2Adapter.create_from_config(_source(path))
    desc = source.get_tensor_descriptor()
    c_index = list(desc.dim_labels).index("C")
    assert desc.chunk_shape[c_index] == desc.shape[c_index]


def test_reader_reused_across_reads_and_closed_explicitly(tmp_path, monkeypatch):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")
    _FakeND2File.opens = 0  # class-level counter; reset for this test's count
    _install_fake(monkeypatch, sizes={"T": 2, "Y": 2, "X": 2})
    source = Nd2Adapter.create_from_config(_source(path))
    # One open for read_layout's probe.
    assert _FakeND2File.opens == 1

    bounds = ChunkBounds(start=[0, 0, 0], stop=[2, 2, 2])
    source.get_data(bounds)
    source.get_data(bounds)
    # The persistent reader is reused across reads, not reopened each time.
    assert _FakeND2File.opens == 2
    source.close()
    assert source._persistent_reader is None


def test_remote_source_is_refused_rather_than_rerouted(tmp_path):
    """Discovery never routes a remote ND2 here -- ``claim`` declines it."""
    path = Path(tmp_path) / "img.nd2"
    path.write_bytes(b"not a real nd2 file")

    assert Nd2Adapter.claim(_RemoteCtx(path), DiscoveryState()) is None

    source = SourceConfig(url="s3://bucket/img.nd2", type="nd2", source_id="remote")
    with pytest.raises(ValueError, match="only supports local files"):
        Nd2Adapter.create_from_config(source)


def test_remote_nd2_is_claimed_by_the_bioio_adapter(tmp_path):
    """The hand-off is the registry's, not an import: BioIO claims what this
    adapter declines, under its own unchanged ``"nikon"`` type string."""
    pytest.importorskip("bioio_nd2")
    path = Path(tmp_path) / "img.nd2"
    path.write_bytes(b"not a real nd2 file")

    claims = get_default_registry().get_claims_for_path(
        _RemoteCtx(path), DiscoveryState()
    )
    assert [c.source_type for c in claims] == ["nikon"]


def test_module_does_not_import_bioio():
    """No fallback path, so no BioIO dependency in this module."""
    source = Path(nd2_module.__file__).read_text()
    assert "adapters.bioio" not in source
    assert "import bioio" not in source


def test_claim_is_definite_under_a_cloud_root(tmp_path):
    path = tmp_path / "img.nd2"
    path.write_bytes(b"\x00")

    claim = Nd2Adapter.claim(LiveLocalContext(path, cloud_root=True), DiscoveryState())
    assert claim is not None
    assert claim.source_type == "nd2"
    assert claim.unresolved is False


def test_claim_does_not_open_the_file(tmp_path):
    class _RaisingReadCtx(LiveLocalContext):
        def read_text(self, subpath: str = "") -> str:
            raise AssertionError("claim() must not read content")

        def open(self, mode: str = "rb") -> object:
            raise AssertionError("claim() must not open the file")

    path = tmp_path / "not-really.nd2"
    path.write_bytes(b"\x00\x01\x02\x03")

    claim = Nd2Adapter.claim(_RaisingReadCtx(path), DiscoveryState())
    assert claim is not None and claim.source_type == "nd2"
