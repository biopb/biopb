"""The bioio-backed adapters, LIF and DeltaVision rebuilt from their payload.

Each is checked with ``assert_hydrates_equivalently``: the adapter rebuilt from what
its registration stored serves the parsed adapter's listing, descriptors, grid,
scale, metadata and pixels, and opens nothing while it is rebuilt.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from biopb_tensor_server.adapters import bioio as bioio_module
from biopb_tensor_server.adapters.bioio import (
    AicsImageIoAdapter,
    BioformatsAdapter,
    DvAdapter,
    LeicaAdapter,
    NikonAdapter,
    OlympusAdapter,
    ZeissAdapter,
)
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.fixtures import (
    create_deltavision_dv,
    create_leica_lif,
    create_zeiss_czi,
    create_zeiss_lsm,
)

from tests.payload_equivalence import assert_hydrates_equivalently


def _source(path, adapter_cls):
    return SourceConfig(url=str(path), type=adapter_cls.SOURCE_TYPE, source_id="src")


def _as_stored(adapter):
    """*adapter* reporting its metadata the way its row keeps it: without ``rois``,
    which the annotation store owns once a registration has imported them."""
    # Once: BioIO numbers the OME ids it generates from a counter, so two reads of
    # one parsed adapter already disagree on them, which is not what is under test.
    stored = {k: v for k, v in adapter.get_metadata().items() if k != "rois"}
    adapter.get_metadata = lambda: stored
    return adapter


class TestThroughBioio:
    """A real file, read by the BioIO plugin the vendor adapter is named for."""

    @pytest.mark.parametrize(
        "factory,plugin,adapter_cls",
        [
            pytest.param(create_zeiss_czi, "bioio_czi", ZeissAdapter, id="zeiss-czi"),
            pytest.param(create_zeiss_lsm, "bioio_czi", ZeissAdapter, id="zeiss-lsm"),
            pytest.param(create_leica_lif, "bioio_lif", LeicaAdapter, id="leica"),
            pytest.param(create_deltavision_dv, "bioio_dv", DvAdapter, id="dv"),
        ],
    )
    def test_a_rebuilt_source_is_the_parsed_one(
        self, factory, plugin, adapter_cls, tmp_path, monkeypatch
    ):
        pytest.importorskip("bioio")
        pytest.importorskip(plugin)
        import bioio

        path, _ = factory(str(tmp_path))
        source = _source(path, adapter_cls)
        parsed = _as_stored(adapter_cls.create_from_config(source))

        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(bioio, "BioImage")]
        )

    def test_a_plain_tiff_through_the_generic_adapter(self, tmp_path, monkeypatch):
        pytest.importorskip("bioio")
        tifffile = pytest.importorskip("tifffile")
        import bioio

        path = tmp_path / "plain.tif"
        tifffile.imwrite(path, np.arange(3 * 8 * 8, dtype="uint16").reshape(3, 8, 8))
        source = _source(path, AicsImageIoAdapter)
        parsed = _as_stored(AicsImageIoAdapter.create_from_config(source))

        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(bioio, "BioImage")]
        )

    def test_the_stored_payload_is_json(self, tmp_path):
        pytest.importorskip("bioio_czi")
        path, _ = create_zeiss_lsm(str(tmp_path))
        parsed = ZeissAdapter.create_from_config(_source(path, ZeissAdapter))
        payload = parsed.catalog_payload()
        assert json.loads(json.dumps(payload)) == payload
        assert [e["field"] for e in payload["listing"]] == [
            e.array_id.split("/", 1)[1] for e in parsed.list_tensors()
        ]

    def test_a_scene_adapter_has_no_payload(self, tmp_path):
        pytest.importorskip("bioio_czi")
        path, _ = create_zeiss_lsm(str(tmp_path))
        parsed = ZeissAdapter.create_from_config(_source(path, ZeissAdapter))
        scene = parsed.get_tensor_adapter(parsed.list_tensors()[0].array_id)
        assert scene.catalog_payload() is None

    def test_a_source_with_too_many_scenes_stores_none(self, tmp_path, monkeypatch):
        pytest.importorskip("bioio_czi")
        path, _ = create_zeiss_lsm(str(tmp_path))
        parsed = ZeissAdapter.create_from_config(_source(path, ZeissAdapter))
        monkeypatch.setattr(bioio_module, "_PAYLOAD_MAX_SCENES", 1)
        assert len(parsed.list_tensors()) == 2
        assert parsed.catalog_payload() is None

    def test_a_remote_source_is_parsed(self):
        source = SourceConfig(url="s3://bucket/a.czi", type="zeiss", source_id="src")
        assert ZeissAdapter.create_from_payload(source, {}, {}, None) is None


class _FakeOme:
    """The OME model BioIO exposes: images with a pixel size, and a dump."""

    def __init__(self, scenes):
        self.images = [
            SimpleNamespace(
                pixels=SimpleNamespace(
                    size_t=a.shape[0],
                    size_c=a.shape[1],
                    size_z=a.shape[2],
                    size_y=a.shape[3],
                    size_x=a.shape[4],
                    physical_size_x=0.25,
                    physical_size_x_unit=SimpleNamespace(value="µm"),
                    physical_size_y=0.25,
                    physical_size_y_unit=SimpleNamespace(value="µm"),
                    physical_size_z=1.5 * (i + 1),
                    physical_size_z_unit=SimpleNamespace(value="µm"),
                )
            )
            for i, a in enumerate(scenes.values())
        ]

    with_rois = False

    def model_dump(self, mode=None):
        dump = {"images": [{"id": f"Image:{i}"} for i in range(len(self.images))]}
        if self.with_rois:
            dump["images"][0]["roi_refs"] = [{"id": "ROI:0"}]
            dump["rois"] = [
                {
                    "id": "ROI:0",
                    "union": {
                        "rectangles": [
                            {
                                "id": "Shape:0",
                                "x": 2.0,
                                "y": 3.0,
                                "width": 4.0,
                                "height": 5.0,
                            }
                        ]
                    },
                }
            ]
        return dump


class _FakeBioImage:
    """A BioImage over two in-memory scenes. Counts how often it is opened."""

    opened = 0

    def __init__(self, url, **kwargs):
        import dask.array as da

        type(self).opened += 1
        rng = np.random.default_rng(7)
        self._arrays = {
            "A": rng.integers(0, 1000, (2, 2, 3, 16, 24)).astype("uint16"),
            "B": rng.integers(0, 1000, (1, 2, 5, 16, 24)).astype("uint16"),
        }
        self._da = da
        self.scenes = tuple(self._arrays)
        self._scene = "A"
        self.ome_metadata = _FakeOme(self._arrays)
        self.metadata = self.ome_metadata
        self.dims = SimpleNamespace(order="TCZYX")
        self.reader = SimpleNamespace()

    def set_scene(self, scene):
        self._scene = self.scenes[scene] if isinstance(scene, int) else scene

    @property
    def dask_data(self):
        return self._da.from_array(self._arrays[self._scene], chunks=(1, 1, 1, 16, 24))


class TestSharedBase:
    """The vendor adapters with no synthetic file to read, over a fake BioImage.

    The scene, grid, scale and metadata logic is the base class's, shared by all
    of them; this runs it under each class that inherits it.
    """

    @pytest.mark.parametrize(
        "adapter_cls",
        [NikonAdapter, OlympusAdapter, BioformatsAdapter, AicsImageIoAdapter],
    )
    def test_a_rebuilt_source_is_the_parsed_one(
        self, adapter_cls, tmp_path, monkeypatch
    ):
        pytest.importorskip("bioio")
        import bioio

        monkeypatch.setattr(bioio, "BioImage", _FakeBioImage)
        # Not created: a source with no file takes the BioIO read path.
        path = tmp_path / "scan.bin"
        source = _source(path, adapter_cls)
        parsed = _as_stored(adapter_cls.create_from_config(source))
        _FakeBioImage.opened = 0

        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(bioio, "BioImage")]
        )

    def test_nothing_is_opened_until_a_read(self, tmp_path, monkeypatch):
        pytest.importorskip("bioio")
        import bioio
        from biopb.tensor.ticket_pb2 import ChunkBounds

        from tests.payload_equivalence import stored

        monkeypatch.setattr(bioio, "BioImage", _FakeBioImage)
        source = _source(tmp_path / "scan.bin", OlympusAdapter)
        parsed = OlympusAdapter.create_from_config(source)
        payload, metadata = stored(parsed)
        _FakeBioImage.opened = 0

        rebuilt = OlympusAdapter.create_from_payload(source, payload, metadata, None)
        entries = rebuilt.list_tensors()
        scene = rebuilt.get_tensor_adapter(entries[1].array_id)
        scene.get_tensor_descriptor()
        scene._physical_scale()
        rebuilt.get_metadata()
        assert _FakeBioImage.opened == 0

        shape = list(scene.get_tensor_descriptor().shape)
        scene.get_data(ChunkBounds(start=[0] * 5, stop=shape))
        assert _FakeBioImage.opened == 1  # the first read opens it, once
        scene.get_data(ChunkBounds(start=[0] * 5, stop=shape))
        assert _FakeBioImage.opened == 1


class TestEmbeddedRois:
    """The row's metadata has no ``rois`` (the store owns them), so a rebuilt adapter
    must not re-import from it: that would wipe the imported set."""

    def _sources(self, tmp_path, monkeypatch, with_rois):
        pytest.importorskip("bioio")
        import bioio

        from tests.payload_equivalence import stored

        _FakeOme.with_rois = with_rois
        monkeypatch.setattr(bioio, "BioImage", _FakeBioImage)
        source = _source(tmp_path / "scan.bin", OlympusAdapter)
        parsed = OlympusAdapter.create_from_config(source)
        payload, metadata = stored(parsed)
        return parsed, source, payload, metadata

    def _tensors(self, adapter):
        return [(t.array_id, list(t.dim_labels)) for t in adapter.list_tensors()]

    def test_a_file_with_rois_is_read_for_them_when_asked(self, tmp_path, monkeypatch):
        parsed, source, payload, metadata = self._sources(tmp_path, monkeypatch, True)
        assert payload["has_rois"] is True
        assert "rois" not in metadata
        expected = parsed.registration_record(self._tensors(parsed)).rois
        assert expected  # the fixture has a rectangle on the first image

        rebuilt = OlympusAdapter.create_from_payload(source, payload, metadata, None)
        _FakeBioImage.opened = 0
        got = rebuilt.registration_record(self._tensors(rebuilt)).rois

        assert {k: [a.SerializeToString() for a in v] for k, v in got.items()} == {
            k: [a.SerializeToString() for a in v] for k, v in expected.items()
        }
        assert _FakeBioImage.opened == 1  # looked in the file, which is the point

    def test_a_file_without_rois_is_not_opened_to_find_none(
        self, tmp_path, monkeypatch
    ):
        parsed, source, payload, metadata = self._sources(tmp_path, monkeypatch, False)
        assert payload["has_rois"] is False

        rebuilt = OlympusAdapter.create_from_payload(source, payload, metadata, None)
        _FakeBioImage.opened = 0
        got = rebuilt.registration_record(self._tensors(rebuilt)).rois

        assert got == {}
        assert _FakeBioImage.opened == 0

    def test_a_rebuilt_adapter_keeps_the_flag_through_a_second_registration(
        self, tmp_path, monkeypatch
    ):
        from tests.payload_equivalence import stored

        parsed, source, payload, metadata = self._sources(tmp_path, monkeypatch, True)
        rebuilt = OlympusAdapter.create_from_payload(source, payload, metadata, None)

        assert rebuilt.catalog_payload()["has_rois"] is True
        _, again = stored(rebuilt)
        assert again == metadata


class TestLif:
    def test_a_rebuilt_source_is_the_parsed_one_without_opening_the_file(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("readlif")
        import readlif.reader
        from biopb_tensor_server.adapters.lif import LifAdapter

        path, _ = create_leica_lif(str(tmp_path))
        source = _source(path, LifAdapter)
        parsed = LifAdapter.create_from_config(source)

        assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[(readlif.reader, "LifFile")],
        )

    def test_the_layout_survives_json(self, tmp_path):
        pytest.importorskip("readlif")
        from biopb_tensor_server.adapters.lif import _LifLayout, read_layout

        path, _ = create_leica_lif(str(tmp_path))
        layout = read_layout(path)
        payload = json.loads(json.dumps(layout.to_payload()))
        assert _LifLayout.from_payload(payload, path) == layout

    def test_a_value_json_cannot_carry_stores_no_payload(self, tmp_path):
        pytest.importorskip("readlif")
        from biopb_tensor_server.adapters.lif import LifAdapter

        path, _ = create_leica_lif(str(tmp_path))
        parsed = LifAdapter.create_from_config(_source(path, LifAdapter))
        parsed._layout.image_list[0]["settings"]["odd"] = object()
        assert parsed.catalog_payload() is None

    def test_an_image_adapter_has_no_payload(self, tmp_path):
        pytest.importorskip("readlif")
        from biopb_tensor_server.adapters.lif import LifAdapter

        path, _ = create_leica_lif(str(tmp_path))
        parsed = LifAdapter.create_from_config(_source(path, LifAdapter))
        image = parsed.get_tensor_adapter(parsed.list_tensors()[0].array_id)
        assert image.catalog_payload() is None


class TestDeltaVision:
    def test_a_rebuilt_source_is_the_parsed_one_without_opening_the_file(
        self, tmp_path, monkeypatch
    ):
        mrc = pytest.importorskip("mrc")
        from biopb_tensor_server.adapters.dv import DeltaVisionAdapter

        path, _ = create_deltavision_dv(str(tmp_path))
        source = _source(path, DeltaVisionAdapter)
        parsed = DeltaVisionAdapter.create_from_config(source)

        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(mrc, "DVFile")]
        )

    def test_the_payload_is_json(self, tmp_path):
        pytest.importorskip("mrc")
        from biopb_tensor_server.adapters.dv import DeltaVisionAdapter

        path, _ = create_deltavision_dv(str(tmp_path))
        parsed = DeltaVisionAdapter.create_from_config(
            _source(path, DeltaVisionAdapter)
        )
        payload = parsed.catalog_payload()
        assert json.loads(json.dumps(payload)) == payload
