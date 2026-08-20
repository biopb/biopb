"""Phase 1 coverage for the native plain-TIFF adapter."""

from pathlib import Path

import numpy as np
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import TiffAdapter, get_default_registry
from biopb_tensor_server.adapters.tifffile_adapter import _mapped_axes
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState


def test_plain_tiff_claims_as_tiff_and_reads_with_tifffile(tmp_path):
    data = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
    path = Path(tmp_path) / "plain.tif"
    tifffile.imwrite(path, data)

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(path), DiscoveryState())
    assert [claim.source_type for claim in claims] == ["tiff"]

    adapter_cls = registry.get_adapter_for_type("tiff")
    source = adapter_cls.create_from_config(
        SourceConfig(url=str(path), type="tiff", source_id="plain")
    )
    assert isinstance(source, TiffAdapter)

    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == ["T", "C", "Z", "Y", "X"]
    assert list(descriptor.shape) == [1, 1, 5, 8, 9]
    assert source.get_metadata() == {"shape": [5, 8, 9]}

    scene = source.get_tensor_adapter(descriptor.array_id)
    bounds = ChunkBounds(start=[0, 0, 2, 3, 4], stop=[1, 1, 5, 7, 9])
    actual = scene.get_data(bounds)
    np.testing.assert_array_equal(actual, data[2:5, 3:7, 4:9][None, None])


def test_configured_dim_labels_stay_native(tmp_path):
    data = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
    path = Path(tmp_path).joinpath("configured.tif")
    tifffile.imwrite(path, data)

    source = TiffAdapter.create_from_config(
        SourceConfig(
            url=str(path),
            type="tiff",
            source_id="configured",
            dim_labels=["z", "y", "x"],
        )
    )
    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == ["z", "y", "x"]
    assert list(descriptor.shape) == [5, 8, 9]

    scene = source.get_tensor_adapter(descriptor.array_id)
    bounds = ChunkBounds(start=[1, 2, 3], stop=[4, 7, 9])
    actual = scene.get_data(bounds)
    np.testing.assert_array_equal(actual, data[1:4, 2:7, 3:9])


def test_unsupported_tiff_declines_native_claim_and_falls_through(tmp_path):
    data = np.zeros((2, 2, 2, 2, 8, 9), dtype=np.uint8)
    path = Path(tmp_path) / "unsupported.tif"
    tifffile.imwrite(path, data, metadata={"axes": "QQQQYX"})

    claims = get_default_registry().get_claims_for_path(
        ClaimContext(path), DiscoveryState()
    )
    assert [claim.source_type for claim in claims] == ["aics"]


def test_named_position_and_mosaic_axes_are_not_relabelled():
    assert _mapped_axes("PTZYX", (2, 3, 4, 8, 9)) == "PTZYX"
    assert _mapped_axes("MYXC", (2, 8, 9, 3)) == "MYXC"


def test_named_axes_are_claimed_natively_and_kept_in_descriptor(tmp_path):
    cases = [
        (
            "position",
            "PTZYX",
            (2, 3, 4, 8, 9),
            ["P", "T", "C", "Z", "Y", "X"],
            [2, 3, 1, 4, 8, 9],
        ),
        (
            "mosaic",
            "MYXC",
            (2, 8, 9, 3),
            ["M", "T", "C", "Z", "Y", "X"],
            [2, 1, 3, 1, 8, 9],
        ),
    ]
    registry = get_default_registry()

    for name, axes, shape, labels, expected_shape in cases:
        path = tmp_path.joinpath(f"{name}.tif")
        tifffile.imwrite(path, np.zeros(shape, dtype=np.uint8), metadata={"axes": axes})

        claims = registry.get_claims_for_path(ClaimContext(path), DiscoveryState())
        assert [claim.source_type for claim in claims] == ["tiff"]

        source = registry.get_adapter_for_type("tiff").create_from_config(
            SourceConfig(url=str(path), type="tiff", source_id=name)
        )
        descriptor = source.list_tensor_descriptors()[0]
        assert list(descriptor.dim_labels) == labels
        assert list(descriptor.shape) == expected_shape
