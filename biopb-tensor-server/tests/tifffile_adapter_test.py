"""Phase 1 coverage for the native plain-TIFF adapter."""

from pathlib import Path

import numpy as np
import tifffile
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import TiffAdapter, get_default_registry
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState


def test_plain_tiff_claims_as_aics_but_reads_with_tifffile(tmp_path):
    data = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
    path = Path(tmp_path) / "plain.tif"
    tifffile.imwrite(path, data)

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(path), DiscoveryState())
    assert [claim.source_type for claim in claims] == ["aics"]

    adapter_cls = registry.get_adapter_for_type("aics")
    source = adapter_cls.create_from_config(
        SourceConfig(url=str(path), type="aics", source_id="plain")
    )
    assert isinstance(source, TiffAdapter)

    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == ["T", "C", "Z", "Y", "X"]
    assert list(descriptor.shape) == [1, 1, 5, 8, 9]

    scene = source.get_tensor_adapter(descriptor.array_id)
    bounds = ChunkBounds(start=[0, 0, 2, 3, 4], stop=[1, 1, 5, 7, 9])
    actual = scene.get_data(bounds)
    np.testing.assert_array_equal(actual, data[2:5, 3:7, 4:9][None, None])
