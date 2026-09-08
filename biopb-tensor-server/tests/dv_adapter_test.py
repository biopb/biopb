"""Phase 3 coverage for the native DeltaVision DV adapter (biopb/biopb#799)."""

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

pytest.importorskip("mrc")

from biopb_tensor_server.adapters import (  # noqa: E402
    DeltaVisionAdapter,
    dv as dv_module,  # noqa: E402
    get_default_registry,
)
from biopb_tensor_server.fixtures import create_deltavision_dv  # noqa: E402


class _RemoteCtx(LiveLocalContext):
    """A claim context that reports itself remote, as a RemoteContext does."""

    @property
    def is_remote(self) -> bool:
        return True


def _source(path, source_id="dv", **kwargs):
    return SourceConfig(
        url=str(path), type="deltavision", source_id=source_id, **kwargs
    )


def _native(path, **kwargs):
    return DeltaVisionAdapter.create_from_config(_source(path, **kwargs))


def test_local_dv_claims_natively_and_reads_through_mrc_dvfile(tmp_path):
    path, expected = create_deltavision_dv(
        str(tmp_path), n_z=6, image_shape=(20, 24), dtype=np.uint16
    )

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(Path(path)), DiscoveryState())
    assert [claim.source_type for claim in claims] == ["deltavision"]

    source = registry.get_adapter_for_type("deltavision").create_from_config(
        _source(path)
    )
    assert isinstance(source, DeltaVisionAdapter)

    descriptors = source.list_tensor_descriptors()
    assert len(descriptors) == 1
    desc = descriptors[0]
    # Native loop order for a single-channel Z-stack (see mrc.DVFile.axes).
    assert list(desc.dim_labels) == ["C", "T", "Z", "Y", "X"]
    assert list(desc.shape) == [1, 1, 6, 20, 24]
    assert desc.dtype == np.dtype(np.uint16).str

    whole = source.get_data(ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 6, 20, 24]))
    np.testing.assert_array_equal(whole, expected.transpose(1, 0, 2, 3, 4))


def test_interior_crop_reads_only_the_requested_window(tmp_path):
    path, expected = create_deltavision_dv(str(tmp_path), n_z=8, image_shape=(16, 20))
    source = _native(path)
    expected = expected.transpose(1, 0, 2, 3, 4)  # -> C,T,Z,Y,X

    bounds = ChunkBounds(start=[0, 0, 2, 3, 4], stop=[1, 1, 6, 12, 15])
    np.testing.assert_array_equal(
        source.get_data(bounds), expected[0:1, 0:1, 2:6, 3:12, 4:15]
    )


def test_decimated_read_matches_a_strided_slice_of_the_full_read(tmp_path):
    path, expected = create_deltavision_dv(str(tmp_path), n_z=8, image_shape=(16, 20))
    source = _native(path)
    expected = expected.transpose(1, 0, 2, 3, 4)

    bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 8, 16, 20])
    step = (1, 1, 2, 2, 3)
    decimated = source.get_decimated_data(bounds, step)
    np.testing.assert_array_equal(decimated, expected[:, :, ::2, ::2, ::3])


def test_physical_scale_reports_dv_header_pixel_spacing(tmp_path):
    path, _ = create_deltavision_dv(str(tmp_path), n_z=2, image_shape=(8, 8))
    source = _native(path)
    scale, unit = source._physical_scale()
    labels = source.dim_labels
    for axis, label in enumerate(labels):
        if label == "X" or label == "Y":
            assert scale[axis] == pytest.approx(1.0)
            assert unit[axis] == "µm"


def test_remote_source_is_refused_rather_than_rerouted(tmp_path):
    """Discovery never routes a remote DV here -- ``claim`` declines it."""
    path = Path(tmp_path) / "img.dv"
    path.write_bytes(b"not a real dv file")

    assert DeltaVisionAdapter.claim(_RemoteCtx(path), DiscoveryState()) is None

    source = SourceConfig(
        url="s3://bucket/img.dv", type="deltavision", source_id="remote"
    )
    with pytest.raises(ValueError, match="only supports local files"):
        DeltaVisionAdapter.create_from_config(source)


def test_remote_dv_is_claimed_by_the_bioio_adapter(tmp_path):
    """The hand-off is the registry's, not an import: BioIO claims what this
    adapter declines, and keeps its own (unchanged) ``"dv"`` type string."""
    path = Path(tmp_path) / "img.dv"
    path.write_bytes(b"not a real dv file")

    claims = get_default_registry().get_claims_for_path(
        _RemoteCtx(path), DiscoveryState()
    )
    assert [c.source_type for c in claims] == ["dv"]


def test_module_does_not_import_bioio():
    """No fallback path, so no BioIO dependency in this module."""
    source = Path(dv_module.__file__).read_text()
    assert "adapters.bioio" not in source
    assert "import bioio" not in source


def test_claim_is_definite_under_a_cloud_root(tmp_path):
    """The claim reads nothing, so a cloud placeholder is claimed, not declined."""
    path, _ = create_deltavision_dv(str(tmp_path), n_z=1, image_shape=(4, 4))

    claim = DeltaVisionAdapter.claim(
        LiveLocalContext(Path(path), cloud_root=True), DiscoveryState()
    )
    assert claim is not None
    assert claim.source_type == "deltavision"
    assert claim.unresolved is False


def test_claim_does_not_open_the_file(tmp_path):
    """Not even a stat-and-open sniff: the claim is extension-only."""

    class _RaisingReadCtx(LiveLocalContext):
        def read_text(self, subpath: str = "") -> str:
            raise AssertionError("claim() must not read content")

        def open(self, mode: str = "rb") -> object:
            raise AssertionError("claim() must not open the file")

    path = tmp_path / "not-really.dv"
    path.write_bytes(b"\x00\x01\x02\x03")

    claim = DeltaVisionAdapter.claim(_RaisingReadCtx(path), DiscoveryState())
    assert claim is not None and claim.source_type == "deltavision"


def test_unopenable_file_raises_instead_of_falling_back(tmp_path):
    path = tmp_path / "corrupt.dv"
    path.write_bytes(b"not a dv file at all, no dvid magic")

    with pytest.raises(Exception) as excinfo:
        DeltaVisionAdapter.create_from_config(_source(path))
    assert not isinstance(excinfo.value, AssertionError)


def test_close_releases_the_mapping_and_a_later_read_reopens(tmp_path):
    path, expected = create_deltavision_dv(str(tmp_path), n_z=4, image_shape=(8, 8))
    source = _native(path)
    expected = expected.transpose(1, 0, 2, 3, 4)

    bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 4, 8, 8])
    first = source.get_data(bounds)
    source.close()
    assert source._persistent_handle is None
    second = source.get_data(bounds)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
