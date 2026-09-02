"""Phase 1 coverage for the native plain-TIFF adapter."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import TiffAdapter, get_default_registry
from biopb_tensor_server.adapters.tifffile_adapter import (
    _PERSISTENT_PAGE_THRESHOLD,
    _mapped_axes,
)
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
    assert scene._persistent_tiff is None
    assert scene._persistent_store is None


def test_native_store_persistence_starts_at_page_threshold(tmp_path):
    path = Path(tmp_path) / "threshold.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint8))

    source = TiffAdapter.create_from_config(
        SourceConfig(url=str(path), type="tiff", source_id="threshold")
    )
    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    assert _PERSISTENT_PAGE_THRESHOLD == 16_384
    scene._tifffile_page_count = _PERSISTENT_PAGE_THRESHOLD - 1
    assert scene._should_persist_store() is False
    scene._tifffile_page_count = _PERSISTENT_PAGE_THRESHOLD
    assert scene._should_persist_store() is True


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


def test_empty_dim_labels_are_not_an_override(tmp_path):
    """An unset protobuf repeated field arrives as ``[]``, not ``None``.

    Reading that as "the caller specified labels" fails the rank check for every
    series, leaving no descriptors and turning a readable TIFF into
    "cannot read TIFF source" -- which is how runtime registration of every
    plain TIFF broke while the config path, which passes ``None``, kept working.
    """
    data = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
    path = Path(tmp_path).joinpath("unlabelled.tif")
    tifffile.imwrite(path, data)

    source = TiffAdapter.create_from_config(
        SourceConfig(url=str(path), type="tiff", source_id="empty", dim_labels=[])
    )
    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == ["T", "C", "Z", "Y", "X"]
    assert list(descriptor.shape) == [1, 1, 5, 8, 9]


def test_unknown_axes_claim_and_read_natively(tmp_path):
    shape = (2, 2, 2, 2, 8, 9)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    path = Path(tmp_path) / "unknown.tif"
    tifffile.imwrite(path, data, metadata={"axes": "QQQQYX"})

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(path), DiscoveryState())
    assert [claim.source_type for claim in claims] == ["tiff"]

    source = registry.get_adapter_for_type("tiff").create_from_config(
        SourceConfig(url=str(path), type="tiff", source_id="unknown")
    )
    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == list("QQQQYX")
    assert list(descriptor.shape) == list(shape)

    scene = source.get_tensor_adapter(descriptor.array_id)
    bounds = ChunkBounds(start=[0] * len(shape), stop=list(shape))
    np.testing.assert_array_equal(scene.get_data(bounds), data)


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


@pytest.mark.parametrize(
    "filename, source_type",
    [("invalid.tif", "tiff"), ("invalid.lsm", "lsm")],
)
def test_malformed_native_claims_then_initialization_raises(
    tmp_path, filename, source_type
):
    path = Path(tmp_path).joinpath(filename)
    path.write_bytes(b"\x00\x01\x02\x03")

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(path), DiscoveryState())
    assert [claim.source_type for claim in claims] == [source_type]

    adapter_cls = registry.get_adapter_for_type(source_type)
    with pytest.raises(ValueError, match="cannot read TIFF source") as exc_info:
        adapter_cls.create_from_config(
            SourceConfig(url=str(path), type=source_type, source_id="invalid")
        )
    assert exc_info.value.__cause__ is not None


def test_native_init_propagates_oserror(tmp_path, monkeypatch):
    path = Path(tmp_path) / "unreadable.tif"
    path.write_bytes(b"placeholder")
    failure = OSError("disk unavailable")

    def fail(_adapter):
        raise failure

    monkeypatch.setattr(TiffAdapter, "_tifffile_descriptors", fail)
    with pytest.raises(OSError) as exc_info:
        TiffAdapter(str(path), "unreadable")
    assert exc_info.value is failure


def test_native_init_chains_non_oserror(tmp_path, monkeypatch):
    path = Path(tmp_path) / "broken.tif"
    path.write_bytes(b"placeholder")
    failure = RuntimeError("descriptor parser failed")

    def fail(_adapter):
        raise failure

    monkeypatch.setattr(TiffAdapter, "_tifffile_descriptors", fail)
    with pytest.raises(ValueError, match="cannot read TIFF source") as exc_info:
        TiffAdapter(str(path), "broken")
    assert exc_info.value.__cause__ is failure


class _RemoteClaimStore:
    def isfile(self, path=""):
        return True

    def isdir(self, path=""):
        return False

    def exists(self, path=""):
        return True

    def open(self, path="", mode="rb"):
        raise AssertionError("remote native claim must not read content")

    def find(self, pattern="*", maxdepth=None, withdirs=False):
        return []

    def _join(self, path):
        return f"remote://{path}"


@pytest.mark.parametrize(
    "filename, source_type",
    [("remote.tif", "aics"), ("remote.lsm", "zeiss")],
)
def test_remote_tiff_and_lsm_fall_back_without_sniff(filename, source_type):
    claims = get_default_registry().get_claims_for_path(
        ClaimContext(filename, _RemoteClaimStore()), DiscoveryState()
    )
    assert [claim.source_type for claim in claims] == [source_type]
