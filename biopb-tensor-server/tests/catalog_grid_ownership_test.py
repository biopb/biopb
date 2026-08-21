"""The catalog lists structure; the tensor-bound adapter owns the read plan.

``chunk_shape`` is the transfer grid a read is issued on, and it can depend on
facts that only exist once a *specific* tensor is selected -- that scene's own
Dask chunks, its dimension labels, its dtype, its native pyramid level, and the
request's scale. A source lists every tensor without binding any of them, so any
grid it names there is a guess about a scene it never selected, published to
every client as fact (biopb/biopb#812).

So the boundary is: ``SourceAdapter.list_tensor_descriptors`` -> structural entry
(array_id / dim_labels / shape / dtype), and ``TensorAdapter.get_tensor_descriptor``
-> the full serving descriptor, reached only through ``get_tensor_adapter``.
These tests hold both halves, at the adapter, the catalog, and the wire.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from biopb.tensor.descriptor_pb2 import (
    DataSourceDescriptor,
    TensorDescriptor,
    TensorReadOption,
)
from biopb_tensor_server.adapters.bioio import ZeissAdapter
from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.core.adapter_base import SourceAdapter, catalog_entry
from biopb_tensor_server.core.config import PyramidConfig, SourceConfig
from biopb_tensor_server.core.metadata_db import MetadataDatabase

# --- the invariant, over the real adapters ----------------------------------


def _sources(temp_dir, simple_zarr_array, hdf5_dataset, multires_ome_zarr):
    """One live adapter per family that has a distinct listing path."""
    from biopb_tensor_server.fixtures import create_multi_series_ome_tiff

    multi_series = create_multi_series_ome_tiff(temp_dir, n_series=3)[0]
    return {
        "zarr": ZarrAdapter.create_from_config(
            SourceConfig(url=simple_zarr_array[0], type="zarr", source_id="z")
        ),
        "hdf5": Hdf5Adapter.create_from_config(
            SourceConfig(
                url=hdf5_dataset[0], type="hdf5", source_id="h", dataset="data"
            )
        ),
        "ome-zarr": OmeZarrAdapter.create_from_config(
            SourceConfig(url=multires_ome_zarr[0], type="ome-zarr", source_id="oz")
        ),
        "ome-tiff": OmeTiffAdapter(multi_series, "ot"),
    }


@pytest.fixture
def live_sources(temp_dir, simple_zarr_array, hdf5_dataset, multires_ome_zarr):
    return _sources(temp_dir, simple_zarr_array, hdf5_dataset, multires_ome_zarr)


@pytest.mark.parametrize("family", ["zarr", "hdf5", "ome-zarr", "ome-tiff"])
def test_listing_is_structural_and_binding_answers_the_grid(live_sources, family):
    """Both halves at once, on a real adapter of each listing shape.

    Single-tensor (zarr, hdf5), multi-tensor built from one shared expression
    (ome-zarr), and multi-tensor handing the scene the object it listed
    (ome-tiff) -- the three ways a listing is produced in this registry.
    """
    source = live_sources[family]
    entries = source.list_tensor_descriptors()
    assert entries

    for entry in entries:
        # Structural: enough to enumerate and address...
        assert entry.array_id
        assert list(entry.shape)
        assert entry.dtype
        # ...and no read plan.
        assert list(entry.chunk_shape) == []

        # The bound tensor answers for the grid, sized to its own shape.
        served = source.get_tensor_adapter(entry.array_id).get_tensor_descriptor()
        assert served.array_id == entry.array_id
        assert list(served.shape) == list(entry.shape)
        grid = list(served.chunk_shape)
        assert grid
        assert all(1 <= g <= dim for g, dim in zip(grid, entry.shape, strict=True))


def test_source_descriptor_strips_a_grid_the_listing_leaked(live_sources):
    """The base class enforces it, not each adapter's good behaviour.

    ``get_source_descriptor`` is the only path into the DuckDB row and into the
    adapter-fallback ListFlights, so an adapter that still names a grid cannot
    reach a client through it.
    """

    class _LeakyAdapter(SourceAdapter):
        source_id = "leaky"
        _source_url = "/data/leaky.zarr"
        _catalog_url = "file:///data/leaky.zarr"
        _source_type = "zarr"

        @classmethod
        def create_from_config(cls, source, credentials_config=None):
            raise NotImplementedError

        def get_metadata(self):
            return {}

        def is_resident(self):
            return True

        def list_tensor_descriptors(self):
            return [
                TensorDescriptor(
                    array_id="leaky",
                    dim_labels=["y", "x"],
                    shape=[64, 64],
                    chunk_shape=[16, 16],
                    dtype="uint8",
                )
            ]

    desc = _LeakyAdapter().get_source_descriptor()
    (tensor,) = desc.tensors
    assert list(tensor.shape) == [64, 64]  # structure survives
    assert tensor.dtype == "uint8"
    assert list(tensor.chunk_shape) == []  # the read plan does not


def test_catalog_entry_keeps_structure_and_drops_every_serving_field():
    """The projection itself: what a catalog entry is, stated once."""
    from biopb.tensor.descriptor_pb2 import PyramidLevel, SliceHint

    full = TensorDescriptor(
        array_id="s/A1",
        dim_labels=["z", "y", "x"],
        shape=[8, 64, 64],
        chunk_shape=[1, 64, 64],
        dtype="uint16",
        metadata_json='{"metadata": {}}',
        reduction_method="area",
        slice_hint=SliceHint(start=[0, 0, 0], stop=[1, 8, 8]),
        scale_hint=[1, 4, 4],
    )
    full.physical_scale[:] = [2.0, 0.325, 0.325]
    full.physical_unit[:] = ["micrometer"] * 3
    full.pyramid.append(PyramidLevel(scale_hint=[1, 1, 1], reduction_method="area"))

    entry = catalog_entry(full)

    assert entry.array_id == "s/A1"
    assert list(entry.dim_labels) == ["z", "y", "x"]
    assert list(entry.shape) == [8, 64, 64]
    assert entry.dtype == "uint16"
    assert list(entry.chunk_shape) == []
    assert list(entry.pyramid) == []
    assert list(entry.physical_scale) == []
    assert entry.metadata_json == ""
    assert not entry.HasField("slice_hint")
    assert list(entry.scale_hint) == []
    assert entry.reduction_method == ""


# --- no scene-0 state may leak into another scene ---------------------------


class _FakeDask:
    def __init__(self, shape, dtype, chunks):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.chunks = chunks


class _MultiSceneBioImage:
    """Three scenes that disagree on shape, labels, dtype AND backend block.

    The point of the fake: a listing path that sizes every scene from whatever
    it happened to have bound produces the same grid three times, and each one
    is right for at most one scene.
    """

    _SCENES = {
        "A1": (
            (1, 2, 1, 512, 512),
            "<u2",
            "TCZYX",
            ((1,), (1,) * 2, (1,), (512,), (512,)),
        ),
        "A2": (
            (4, 1, 8, 256, 256),
            "|u1",
            "TCZYX",
            ((1,) * 4, (1,), (1,) * 8, (128,) * 2, (128,) * 2),
        ),
        "A3": ((300, 300, 3), "|u1", "YXS", ((100,) * 3, (100,) * 3, (3,))),
    }

    def __init__(self):
        self.scenes = list(self._SCENES)
        self._bind("A1")
        self.reader = SimpleNamespace(_xarray_dask_data=None, _dims=None)
        self._xarray_dask_data = None

    def _bind(self, scene):
        shape, dtype, order, chunks = self._SCENES[scene]
        self.dask_data = _FakeDask(shape, dtype, chunks)
        self.dims = SimpleNamespace(order=order)

    def set_scene(self, scene):
        self._bind(self.scenes[scene] if isinstance(scene, int) else scene)

    @property
    def ome_metadata(self):
        # Force the scene-switching listing path: the scenes are not one
        # canonical 5-D family, which is exactly when OME shapes cannot describe
        # them.
        raise NotImplementedError


def _multi_scene_source():
    # dim_labels=None: each scene reports its own axis order, which is half the
    # per-scene state a shared grid would flatten.
    return ZeissAdapter(
        _MultiSceneBioImage(), scene_index=None, source_id="multi", dim_labels=None
    )


def test_no_scene_state_leaks_into_a_sibling_scene():
    """Each scene's grid is derived from that scene's own facts, only."""
    source = _multi_scene_source()
    entries = source.list_tensor_descriptors()
    assert [e.array_id for e in entries] == ["multi/A1", "multi/A2", "multi/A3"]

    served = {}
    for entry in entries:
        assert list(entry.chunk_shape) == []
        desc = source.get_tensor_adapter(entry.array_id).get_tensor_descriptor()
        served[entry.array_id] = desc

        scene = entry.array_id.split("/")[1]
        shape, dtype, order, chunks = _MultiSceneBioImage._SCENES[scene]
        # Answered against this scene's own shape / labels / dtype ...
        assert list(desc.shape) == list(shape)
        assert list(desc.dim_labels) == list(order)
        assert desc.dtype == np.dtype(dtype).str
        grid = list(desc.chunk_shape)
        assert len(grid) == len(shape)
        assert all(1 <= g <= dim for g, dim in zip(grid, shape, strict=True))
        # ... and aligned to this scene's own backend block, not a sibling's.
        native = [max(c) for c in chunks]
        assert all(
            g % n == 0 or g == dim
            for g, n, dim in zip(grid, native, shape, strict=True)
        )

    # The ranks alone would already be wrong if one scene's answer were reused.
    assert {len(d.chunk_shape) for d in served.values()} == {5, 3}


def test_source_level_descriptor_binds_the_default_scene():
    """A source that also fills the tensor role must bind, not read its listing.

    Reading ``list_tensor_descriptors()[0]`` back used to be how this answered;
    that entry now carries no grid, so answering from it would hand the read
    planner an empty one.
    """
    source = _multi_scene_source()

    desc = source.get_tensor_descriptor()

    assert desc.array_id == "multi/A1"
    assert list(desc.shape) == [1, 2, 1, 512, 512]
    assert list(desc.chunk_shape)
    assert (
        desc.chunk_shape
        == source.get_tensor_adapter("multi/A1").get_tensor_descriptor().chunk_shape
    )


# --- the wire: ListFlights lean, GetFlightInfo authoritative -----------------


def test_list_flights_is_structural_and_get_flight_info_carries_the_grid(
    multires_ome_zarr,
):
    """End to end through the server, on one source, both surfaces.

    The pair is the contract: a client that wants to plan a read describes the
    tensor, and gets the grid of the adapter the server bound to serve it.
    """
    from biopb_tensor_server import TensorFlightServer

    adapter = OmeZarrAdapter.create_from_config(
        SourceConfig(url=multires_ome_zarr[0], type="ome-zarr", source_id="oz")
    )
    db = MetadataDatabase()
    db.sync_source_added("oz", adapter)
    server = TensorFlightServer(location="grpc://localhost:0", metadata_db=db)
    server.sources.replace({"oz": adapter})

    (info,) = list(server.list_flights(None, b""))
    listed = DataSourceDescriptor.FromString(info.descriptor.command)
    (entry,) = listed.tensors
    assert list(entry.shape)
    assert list(entry.chunk_shape) == []

    tensor_adapter = adapter.get_tensor_adapter(entry.array_id)
    plan = tensor_adapter.plan_flight_info(
        TensorReadOption(tensor_id=entry.array_id), PyramidConfig()
    )
    grid = list(plan.descriptor.chunk_shape)
    assert grid == list(tensor_adapter.get_transfer_chunk_size())
    assert grid


def test_catalog_round_trip_never_reintroduces_a_grid(multires_ome_zarr):
    """query_sources / list_source_descriptors answer structure, nothing more."""
    adapter = OmeZarrAdapter.create_from_config(
        SourceConfig(url=multires_ome_zarr[0], type="ome-zarr", source_id="oz")
    )
    db = MetadataDatabase()
    db.sync_source_added("oz", adapter)

    descriptors, _ = db.list_source_descriptors()
    (entry,) = descriptors[0].tensors
    assert list(entry.shape)
    assert list(entry.chunk_shape) == []

    ((_, struct_type, *_),) = [
        row
        for row in db._get_cursor().execute("DESCRIBE sources").fetchall()
        if row[0] == "tensors"
    ]
    assert "chunk_shape" not in struct_type.lower()
    assert "shape" in struct_type.lower()
