"""The tifffile family (OME-TIFF, plain TIFF, LSM) rebuilt from its payload.

Each is rebuilt without opening the file and then serves what the parsed adapter
serves; a file with ROIs keeps its metadata, ROIs and ``@ome`` labels lazy.
"""

import json

import numpy as np
import pytest
from biopb_tensor_server.adapters import LsmAdapter, OmeTiffAdapter, TiffAdapter
from biopb_tensor_server.core.config import SourceConfig

from tests import ome_mask_test
from tests.payload_equivalence import (
    assert_hydrates_equivalently,
    forbid_opens,
    hydrate,
    stored,
)

tifffile = pytest.importorskip("tifffile")


def _source(path, type_):
    return SourceConfig(url=str(path), type=type_, source_id="src")


def _check(cls, path, type_, monkeypatch):
    source = _source(path, type_)
    parsed = cls.create_from_config(source)
    rebuilt = assert_hydrates_equivalently(
        parsed, source, monkeypatch=monkeypatch, opens=[(tifffile, "TiffFile")]
    )
    return parsed, rebuilt


class TestPlainTiff:
    def test_a_grayscale_tiff(self, tmp_path, monkeypatch):
        path = tmp_path / "a.tif"
        tifffile.imwrite(
            str(path), np.arange(3 * 16 * 24, dtype="uint16").reshape(3, 16, 24)
        )
        _check(TiffAdapter, path, "tiff", monkeypatch)

    def test_a_tiff_with_a_resolution_and_imagej_metadata(self, tmp_path, monkeypatch):
        path = tmp_path / "ij.tif"
        tifffile.imwrite(
            str(path),
            np.zeros((4, 16, 16), "uint8"),
            imagej=True,
            resolution=(2.0, 2.0),
            metadata={"axes": "ZYX", "unit": "um", "spacing": 0.5},
        )
        parsed, rebuilt = _check(TiffAdapter, path, "tiff", monkeypatch)
        assert rebuilt.get_tensor_adapter("src/Image:0")._physical_scale() is not None
        assert rebuilt.get_metadata() == parsed.get_metadata()

    def test_an_rgb_tiff(self, tmp_path, monkeypatch):
        path = tmp_path / "rgb.tif"
        tifffile.imwrite(str(path), np.zeros((16, 16, 3), "uint8"), photometric="rgb")
        _check(TiffAdapter, path, "tiff", monkeypatch)

    def test_a_tiff_with_two_series(self, tmp_path, monkeypatch):
        path = tmp_path / "two.tif"
        with tifffile.TiffWriter(str(path)) as tw:
            tw.write(np.ones((2, 16, 16), "uint16"))
            tw.write(np.full((3, 8, 8), 7, "uint16"))
        parsed, rebuilt = _check(TiffAdapter, path, "tiff", monkeypatch)
        assert len(rebuilt.list_tensors()) == 2

    def test_the_rebuilt_source_stores_again_without_opening_the_file(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "a.tif"
        tifffile.imwrite(str(path), np.zeros((2, 8, 8), "uint16"))
        source = _source(path, "tiff")
        rebuilt = hydrate(TiffAdapter.create_from_config(source), source)

        forbid_opens(monkeypatch, (tifffile, "TiffFile"))
        payload, _ = stored(rebuilt)  # what sync_source_added does on hydration

        assert payload["scenes"] and "physical_scale" in payload


class TestLsm:
    def test_an_lsm(self, tmp_path, monkeypatch):
        # The synthetic LSM of ``fixtures.create_zeiss_lsm`` is not readable by the
        # installed tifffile (its LSM bits-per-sample fix fails), so this is a plain
        # TIFF under the ``.lsm`` name: the class's own descriptor, scale and
        # metadata paths, though not a real file's CZ_LSMINFO calibration.
        path = tmp_path / "a.lsm"
        tifffile.imwrite(
            str(path),
            np.arange(4 * 16 * 16, dtype="uint16").reshape(4, 16, 16),
            photometric="minisblack",
        )
        parsed, rebuilt = _check(LsmAdapter, path, "lsm", monkeypatch)
        assert rebuilt.get_metadata() == parsed.get_metadata()


class TestOmeTiff:
    def test_a_file_with_physical_sizes(self, tmp_path, monkeypatch):
        path = tmp_path / "scaled.ome.tif"
        tifffile.imwrite(
            str(path),
            np.zeros((3, 16, 16), "uint16"),
            ome=True,
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeZ": 2.0,
            },
        )
        parsed, rebuilt = _check(OmeTiffAdapter, path, "ome-tiff", monkeypatch)
        scale = rebuilt.get_tensor_adapter("src/Image:0")._physical_scale()
        assert scale is not None and 0.5 in scale[0]

    def test_a_multi_series_tiled_file(self, tmp_path, monkeypatch):
        from biopb_tensor_server.fixtures import create_multi_series_ome_tiff

        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        parsed, rebuilt = _check(OmeTiffAdapter, path, "ome-tiff", monkeypatch)
        assert len(rebuilt.list_tensors()) == 3

    def test_a_tiled_file(self, tmp_path, monkeypatch):
        from biopb_tensor_server.fixtures import create_tiled_ome_tiff

        path, _, _ = create_tiled_ome_tiff(str(tmp_path))
        _check(OmeTiffAdapter, path, "ome-tiff", monkeypatch)

    def test_a_file_without_rois_never_opens_the_file_to_store_again(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "plain.ome.tif"
        tifffile.imwrite(str(path), np.zeros((2, 8, 8), "uint8"), ome=True)
        source = _source(path, "ome-tiff")
        rebuilt = hydrate(OmeTiffAdapter.create_from_config(source), source)

        forbid_opens(monkeypatch, (tifffile, "TiffFile"))
        stored(rebuilt)  # get_metadata, get_embedded_rois, payload, release
        assert rebuilt.get_embedded_labels() == {}

    def test_a_file_with_a_mask_keeps_its_rois_and_labels(self, tmp_path, monkeypatch):
        bitmap = np.zeros((4, 4), dtype=np.uint8)
        bitmap[1:3, 1:3] = 1
        raw = np.packbits(bitmap.flatten(), bitorder="big").tobytes()
        path = ome_mask_test.TestFastMetadataRealBitmap()._write(tmp_path, raw)
        source = _source(path, "ome-tiff")
        parsed = OmeTiffAdapter.create_from_config(source)
        payload, row_metadata = stored(parsed)
        assert payload["has_rois"] is True and payload["masks"]

        monkeypatch.undo()
        with pytest.MonkeyPatch.context() as m:
            forbid_opens(m, (tifffile, "TiffFile"))
            rebuilt = OmeTiffAdapter.create_from_payload(
                source, json.loads(json.dumps(payload)), row_metadata, None
            )
        assert rebuilt.catalog_payload() == parsed.catalog_payload()

        # The ROIs and the label tensor come from the file, when asked.
        tensors = [(t.array_id, list(t.dim_labels)) for t in rebuilt.list_tensors()]
        rois, _ = rebuilt.get_embedded_rois(rebuilt.get_metadata(), tensors)
        expected, _ = parsed.get_embedded_rois(parsed.get_metadata(), tensors)
        assert {k: len(v) for k, v in rois.items()} == {
            k: len(v) for k, v in expected.items()
        }
        assert set(rebuilt.get_embedded_labels()) == set(parsed.get_embedded_labels())

    def test_a_payload_without_the_scale_is_parsed_instead(self, tmp_path):
        path = tmp_path / "plain.ome.tif"
        tifffile.imwrite(str(path), np.zeros((2, 8, 8), "uint8"), ome=True)
        source = _source(path, "ome-tiff")
        payload, metadata = stored(OmeTiffAdapter.create_from_config(source))
        del payload["physical_scale"]

        assert OmeTiffAdapter.create_from_payload(source, payload, metadata) is None
