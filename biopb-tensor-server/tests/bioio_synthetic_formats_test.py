"""Read-path coverage for the bioio-backed vendor formats.

LSM, LIF, CZI and DV have no checked-in sample data, so each is synthesized
(biopb_tensor_server.fixtures) and driven through the real adapter: claim,
descriptor, pixels. These are the formats whose reads still go through bioio's
dask array, so a regression here is a regression in the path documented in
docs/dask-bypass-benchmarks.md.
"""

from pathlib import Path

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState
from biopb_tensor_server.fixtures import (
    create_deltavision_dv,
    create_leica_lif,
    create_zeiss_czi,
    create_zeiss_lsm,
)

# (fixture factory, required plugin module, expected source_type, scene count).
# LSM carries two scenes: the image and the reduced thumbnail series tifffile
# builds from the interleaved pages. The image is always scene 0.
FORMATS = [
    pytest.param(create_zeiss_lsm, "bioio_tifffile", "zeiss", 2, id="lsm"),
    pytest.param(create_leica_lif, "bioio_lif", "leica", 1, id="lif"),
    pytest.param(create_zeiss_czi, "bioio_czi", "zeiss", 1, id="czi"),
    pytest.param(create_deltavision_dv, "bioio_dv", "dv", 1, id="dv"),
]


def _build(factory, plugin, tmp_path):
    pytest.importorskip(plugin)
    path, expected = factory(str(tmp_path))
    return path, expected


@pytest.mark.parametrize("factory,plugin,source_type,n_scenes", FORMATS)
def test_synthetic_source_is_claimed_by_its_adapter(
    factory, plugin, source_type, n_scenes, tmp_path
):
    path, _ = _build(factory, plugin, tmp_path)

    claims = get_default_registry().get_claims_for_path(
        ClaimContext(Path(path)), DiscoveryState()
    )

    assert [c.source_type for c in claims] == [source_type]


@pytest.mark.parametrize("factory,plugin,source_type,n_scenes", FORMATS)
def test_synthetic_source_reads_back_its_pixels(
    factory, plugin, source_type, n_scenes, tmp_path
):
    path, expected = _build(factory, plugin, tmp_path)
    adapter_cls = get_default_registry().get_adapter_for_type(source_type)
    source = adapter_cls.create_from_config(
        SourceConfig(url=path, type=source_type, source_id="synthetic")
    )

    descriptors = source.list_tensor_descriptors()
    assert len(descriptors) == n_scenes
    assert list(descriptors[0].shape) == list(expected.shape)
    assert descriptors[0].dtype == expected.dtype.str

    scene = source.get_tensor_adapter(descriptors[0].array_id)
    actual = scene.get_data(
        ChunkBounds(start=[0] * expected.ndim, stop=list(expected.shape))
    )

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("factory,plugin,source_type,n_scenes", FORMATS)
def test_synthetic_source_reads_an_interior_crop(
    factory, plugin, source_type, n_scenes, tmp_path
):
    """A sub-chunk read must land on the requested coordinates, not the origin."""
    path, expected = _build(factory, plugin, tmp_path)
    adapter_cls = get_default_registry().get_adapter_for_type(source_type)
    source = adapter_cls.create_from_config(
        SourceConfig(url=path, type=source_type, source_id="synthetic")
    )
    descriptors = source.list_tensor_descriptors()
    scene = source.get_tensor_adapter(descriptors[0].array_id)

    # Last plane of the last channel, cropped away from the origin in Y and X.
    stop = list(expected.shape)
    start = [0] * len(stop)
    start[1], start[2] = stop[1] - 1, stop[2] - 1  # last C, last Z
    start[3], start[4] = stop[3] // 2, stop[4] // 2

    actual = scene.get_data(ChunkBounds(start=start, stop=stop))

    assert np.array_equal(
        actual, expected[tuple(slice(a, b) for a, b in zip(start, stop, strict=True))]
    )
