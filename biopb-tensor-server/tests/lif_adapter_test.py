"""Phase 3 coverage for the native Leica LIF adapter (biopb/biopb#799)."""

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

pytest.importorskip("readlif")

from biopb_tensor_server.adapters import (  # noqa: E402
    LifAdapter,
    get_default_registry,
    lif as lif_module,  # noqa: E402
)
from biopb_tensor_server.core.errors import TensorNotFound  # noqa: E402
from biopb_tensor_server.fixtures import create_leica_lif  # noqa: E402


class _RemoteCtx(LiveLocalContext):
    """A claim context that reports itself remote, as a RemoteContext does."""

    @property
    def is_remote(self) -> bool:
        return True


def _source(path, source_id="lif", **kwargs):
    return SourceConfig(url=str(path), type="lif", source_id=source_id, **kwargs)


def _native(path, **kwargs):
    return LifAdapter.create_from_config(_source(path, **kwargs))


def test_local_lif_claims_natively_and_reads_through_readlif(tmp_path):
    path, expected = create_leica_lif(
        str(tmp_path), n_t=2, n_z=5, n_c=2, image_shape=(24, 32), bit_depth=16
    )

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(Path(path)), DiscoveryState())
    assert [claim.source_type for claim in claims] == ["lif"]

    source = registry.get_adapter_for_type("lif").create_from_config(_source(path))
    assert isinstance(source, LifAdapter)

    descriptors = source.list_tensor_descriptors()
    assert len(descriptors) == 1
    assert list(descriptors[0].dim_labels) == ["T", "C", "Z", "Y", "X"]
    assert list(descriptors[0].shape) == [2, 2, 5, 24, 32]
    # The listing is structural: the grid is the bound image's (biopb/biopb#812).
    assert list(descriptors[0].chunk_shape) == []

    image = source.get_tensor_adapter(descriptors[0].array_id)
    # readlif has no ROI -- the native unit is one whole plane.
    grid = list(image.get_tensor_descriptor().chunk_shape)
    assert grid[-2:] == [24, 32]

    whole = image.get_data(ChunkBounds(start=[0, 0, 0, 0, 0], stop=[2, 2, 5, 24, 32]))
    np.testing.assert_array_equal(whole, expected)


def test_interior_crop_reads_only_the_requested_window(tmp_path):
    path, expected = create_leica_lif(str(tmp_path), n_c=2, n_z=4, image_shape=(24, 32))
    source = _native(path)
    image = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    bounds = ChunkBounds(start=[0, 1, 1, 5, 7], stop=[1, 2, 3, 20, 30])
    np.testing.assert_array_equal(
        image.get_data(bounds), expected[0:1, 1:2, 1:3, 5:20, 7:30]
    )


def test_decimated_read_matches_a_strided_slice_of_the_full_read(tmp_path):
    path, expected = create_leica_lif(
        str(tmp_path), n_t=2, n_z=6, n_c=2, image_shape=(24, 32)
    )
    source = _native(path)
    image = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[2, 2, 6, 24, 32])
    step = (1, 1, 2, 2, 3)
    decimated = image.get_decimated_data(bounds, step)
    np.testing.assert_array_equal(decimated, expected[:, :, ::2, ::2, ::3])


def test_eight_bit_image_reads_as_uint8(tmp_path):
    path, expected = create_leica_lif(
        str(tmp_path), n_t=1, n_z=3, n_c=1, image_shape=(16, 16), bit_depth=8
    )
    source = _native(path)
    descriptor = source.list_tensor_descriptors()[0]
    assert descriptor.dtype == np.dtype(np.uint8).str

    image = source.get_tensor_adapter(descriptor.array_id)
    whole = image.get_data(ChunkBounds(start=[0] * 5, stop=list(descriptor.shape)))
    np.testing.assert_array_equal(whole, expected)


def test_physical_scale_reports_readlif_pixel_size(tmp_path):
    path, _ = create_leica_lif(str(tmp_path), n_z=3, image_shape=(16, 16))
    source = _native(path)
    image = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)
    scale, unit = image._physical_scale()
    labels = image.dim_labels
    for axis, label in enumerate(labels):
        if label in ("X", "Y", "Z"):
            assert scale[axis] > 0
            assert unit[axis] == "µm"


def test_unknown_image_field_raises_tensor_not_found(tmp_path):
    path, _ = create_leica_lif(str(tmp_path), n_z=2, image_shape=(8, 8))
    source = _native(path)
    with pytest.raises(TensorNotFound):
        source.get_tensor_adapter("Image:99")


def test_remote_source_is_refused_rather_than_rerouted(tmp_path):
    """Discovery never routes a remote LIF here -- ``claim`` declines it."""
    path = Path(tmp_path) / "img.lif"
    path.write_bytes(b"not a real lif file")

    assert LifAdapter.claim(_RemoteCtx(path), DiscoveryState()) is None

    source = SourceConfig(url="s3://bucket/img.lif", type="lif", source_id="remote")
    with pytest.raises(ValueError, match="only supports local files"):
        LifAdapter.create_from_config(source)


def test_remote_lif_is_claimed_by_the_bioio_adapter(tmp_path):
    """The hand-off is the registry's, not an import: BioIO claims what this
    adapter declines, under its own unchanged ``"leica"`` type string."""
    pytest.importorskip("bioio_lif")
    path = Path(tmp_path) / "img.lif"
    path.write_bytes(b"not a real lif file")

    claims = get_default_registry().get_claims_for_path(
        _RemoteCtx(path), DiscoveryState()
    )
    assert [c.source_type for c in claims] == ["leica"]


def test_module_does_not_import_bioio():
    """No fallback path, so no BioIO dependency in this module."""
    source = Path(lif_module.__file__).read_text()
    assert "adapters.bioio" not in source
    assert "import bioio" not in source


def test_claim_is_definite_under_a_cloud_root(tmp_path):
    """The claim reads nothing, so a cloud placeholder is claimed, not declined."""
    path, _ = create_leica_lif(str(tmp_path), n_z=1, image_shape=(4, 4))

    claim = LifAdapter.claim(
        LiveLocalContext(Path(path), cloud_root=True), DiscoveryState()
    )
    assert claim is not None
    assert claim.source_type == "lif"
    assert claim.unresolved is False


def test_claim_does_not_open_the_file(tmp_path):
    """Not even a stat-and-open sniff: the claim is extension-only."""

    class _RaisingReadCtx(LiveLocalContext):
        def read_text(self, subpath: str = "") -> str:
            raise AssertionError("claim() must not read content")

        def open(self, mode: str = "rb") -> object:
            raise AssertionError("claim() must not open the file")

    path = tmp_path / "not-really.lif"
    path.write_bytes(b"\x00\x01\x02\x03")

    claim = LifAdapter.claim(_RaisingReadCtx(path), DiscoveryState())
    assert claim is not None and claim.source_type == "lif"


def test_unopenable_file_raises_instead_of_falling_back(tmp_path):
    path = tmp_path / "corrupt.lif"
    path.write_bytes(b"not a lif file at all")

    with pytest.raises(Exception) as excinfo:
        LifAdapter.create_from_config(_source(path))
    assert not isinstance(excinfo.value, AssertionError)
