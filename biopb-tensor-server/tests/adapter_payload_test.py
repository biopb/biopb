"""Adapter payloads: what ``catalog_payload`` persists, and that it rebuilds.

Stage 2 will build an adapter from a persisted row without parsing the file, so
each payload has to be enough for that today. These tests rebuild an adapter's
layout from the JSON round trip of its payload and require the same tensors,
grid and scale as the parsed one.
"""

import json

import numpy as np
import pytest
from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
from biopb_tensor_server.core.config import SourceConfig

from tests import ome_mask_test


def _round_trip(payload):
    return json.loads(json.dumps(payload))


def _descriptors(adapter):
    """Everything a read plan takes from a source: its listing, and each tensor's
    grid and scale."""
    out = []
    for entry in adapter.list_tensors():
        scene = adapter.get_tensor_adapter(entry.array_id)
        d = scene.get_tensor_descriptor()
        out.append(
            (
                entry.array_id,
                list(d.dim_labels),
                list(d.shape),
                list(d.chunk_shape),
                d.dtype,
                scene._physical_scale(),
            )
        )
    return out


class TestOmeTiff:
    def _masked(self, tmp_path):
        bitmap = np.zeros((4, 4), dtype=np.uint8)
        bitmap[1:3, 1:3] = 1
        raw = np.packbits(bitmap.flatten(), bitorder="big").tobytes()
        return OmeTiffAdapter(
            ome_mask_test.TestFastMetadataRealBitmap()._write(tmp_path, raw), "src1"
        )

    def test_a_file_with_a_mask_reports_rois_and_the_label_tensor(self, tmp_path):
        adapter = self._masked(tmp_path)
        payload = adapter.catalog_payload()
        assert payload["has_rois"] is True
        (mask,) = payload["masks"]
        assert mask["field"] == "Image:0/@labels/@ome"
        assert mask["parent_array_id"] == "src1/Image:0"

        # The descriptor is the one the label adapter would serve.
        label = adapter.get_embedded_labels()[mask["field"]].get_tensor_descriptor()
        assert mask["dim_labels"] == list(label.dim_labels)
        assert mask["shape"] == list(label.shape)

    def test_building_the_payload_does_not_consume_the_mask_bitmaps(self, tmp_path):
        adapter = self._masked(tmp_path)
        adapter.catalog_payload()
        assert adapter._mask_payloads_transferred is False
        adapter.get_embedded_labels()
        assert adapter._mask_payloads_transferred is True

    def test_a_plain_file_reports_neither(self, tmp_path):
        import tifffile

        path = str(tmp_path / "plain.ome.tiff")
        tifffile.imwrite(path, np.zeros((2, 8, 8), "uint8"), ome=True)
        payload = OmeTiffAdapter(path, "src1").catalog_payload()
        assert payload["has_rois"] is False
        assert payload["masks"] == []

    def test_the_payload_is_json(self, tmp_path):
        assert _round_trip(self._masked(tmp_path).catalog_payload())

    def test_a_scene_adapter_has_no_payload(self, tmp_path):
        adapter = self._masked(tmp_path)
        scene = adapter.get_tensor_adapter(adapter.list_tensors()[0].array_id)
        assert scene.catalog_payload() is None


class TestNd2:
    @pytest.fixture
    def nd2_adapter(self, tmp_path, monkeypatch):
        pytest.importorskip("nd2")
        from tests.nd2_adapter_test import _install_fake

        _install_fake(monkeypatch)
        path = tmp_path / "img.nd2"
        path.write_bytes(b"\x00")
        from biopb_tensor_server.adapters import Nd2Adapter

        return Nd2Adapter.create_from_config(
            SourceConfig(url=str(path), type="nd2", source_id="nd2")
        )

    def test_the_layout_survives_json(self, nd2_adapter):
        from biopb_tensor_server.adapters.nd2 import _Nd2Layout

        original = nd2_adapter._layout
        payload = _round_trip(nd2_adapter.catalog_payload())["layout"]
        rebuilt = _Nd2Layout.from_payload(payload, original.ome_summary)
        assert rebuilt == original

    def test_the_metadata_is_not_in_the_payload(self, nd2_adapter):
        assert "ome_summary" not in nd2_adapter.catalog_payload()["layout"]

    def test_a_rebuilt_adapter_serves_the_same_tensors(self, nd2_adapter):
        from biopb_tensor_server.adapters import Nd2Adapter
        from biopb_tensor_server.adapters.nd2 import _Nd2Layout

        payload = _round_trip(nd2_adapter.catalog_payload())["layout"]
        rebuilt = Nd2Adapter(
            nd2_adapter._url,
            nd2_adapter.source_id,
            layout=_Nd2Layout.from_payload(
                payload, nd2_adapter.registration_record([], import_rois=False).metadata
            ),
        )
        assert _descriptors(rebuilt) == _descriptors(nd2_adapter)
        assert (
            rebuilt.registration_record([], import_rois=False).metadata
            == nd2_adapter.registration_record([], import_rois=False).metadata
        )

    def test_a_position_adapter_has_no_payload(self, nd2_adapter):
        position = nd2_adapter.get_tensor_adapter("P:0")
        assert position.catalog_payload() is None


class TestCzi:
    @pytest.fixture
    def czi_adapter(self, tmp_path):
        pytest.importorskip("pylibCZIrw")
        from biopb_tensor_server.adapters import CziAdapter
        from biopb_tensor_server.fixtures import create_zeiss_czi

        path, _ = create_zeiss_czi(
            str(tmp_path), n_t=2, n_c=2, n_z=3, image_shape=(24, 32)
        )
        return CziAdapter.create_from_config(
            SourceConfig(url=str(path), type="czi", source_id="czi")
        )

    def test_the_layout_survives_json(self, czi_adapter):
        from biopb_tensor_server.adapters.czi import _CziLayout

        original = czi_adapter._layout
        payload = _round_trip(czi_adapter.catalog_payload())["layout"]
        assert _CziLayout.from_payload(payload, original.information) == original

    def test_the_metadata_is_not_in_the_payload(self, czi_adapter):
        assert "information" not in czi_adapter.catalog_payload()["layout"]

    def test_a_rebuilt_adapter_serves_the_same_tensors(self, czi_adapter):
        from biopb_tensor_server.adapters import CziAdapter
        from biopb_tensor_server.adapters.czi import _CziLayout

        payload = _round_trip(czi_adapter.catalog_payload())["layout"]
        rebuilt = CziAdapter(
            czi_adapter._url,
            czi_adapter.source_id,
            layout=_CziLayout.from_payload(
                payload, czi_adapter.registration_record([], import_rois=False).metadata
            ),
        )
        assert _descriptors(rebuilt) == _descriptors(czi_adapter)
        assert (
            rebuilt.registration_record([], import_rois=False).metadata
            == czi_adapter.registration_record([], import_rois=False).metadata
        )

    def test_a_scene_adapter_has_no_payload(self, czi_adapter):
        scene = czi_adapter.get_tensor_adapter(czi_adapter.list_tensors()[0].array_id)
        assert scene.catalog_payload() is None
