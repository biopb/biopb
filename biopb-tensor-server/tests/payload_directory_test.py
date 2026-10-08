"""Directory and multi-file sources rebuilt from their payload.

Zarr, OME-Zarr (an image and an HCS plate), NDTiff, TIFF sequences and legacy
Micro-Manager datasets: each is rebuilt from the row it stored, without opening
its store or files, and serves what the parsed adapter serves.
"""

import json
import os

import numpy as np
import pytest
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.fixtures import (
    create_multiresolution_ome_zarr,
    create_zarr_array,
)

from tests.payload_equivalence import assert_hydrates_equivalently


def _source(url, kind):
    return SourceConfig(url=str(url), type=kind, source_id="src")


class TestZarr:
    def test_a_rebuilt_array_is_the_parsed_one_and_opens_on_the_first_read(
        self, tmp_path, monkeypatch
    ):
        import zarr
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        path, _, _ = create_zarr_array(str(tmp_path), shape=(64, 96), chunks=(16, 32))
        source = _source(path, "zarr")
        parsed = ZarrAdapter.create_from_config(source)

        rebuilt = assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[(zarr, "open_array"), (zarr, "open_group")],
        )

        assert rebuilt.content_version == parsed.content_version
        assert rebuilt._source_url == parsed._source_url

    def test_a_level_or_field_bound_to_a_tensor_has_no_payload(self, tmp_path):
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        path, _, _ = create_zarr_array(str(tmp_path))
        adapter = ZarrAdapter.create_from_config(_source(path, "zarr"))
        adapter._tensor_name = "0"
        assert adapter.catalog_payload() is None


class TestOmeZarrImage:
    def test_a_multiscale_image_is_rebuilt_with_its_pyramid_levels_sized(
        self, tmp_path, monkeypatch
    ):
        import zarr
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        path, _, _ = create_multiresolution_ome_zarr(str(tmp_path), n_levels=3)
        source = _source(path, "ome-zarr")
        parsed = OmeZarrAdapter.create_from_config(source)
        assert parsed.get_native_pyramid_levels()

        rebuilt = assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[
                (zarr, "open_array"),
                (zarr, "open_group"),
                (zarr, "DirectoryStore"),
            ],
        )

        assert (
            rebuilt.registration_record([], import_rois=False).metadata
            == parsed.registration_record([], import_rois=False).metadata
        )
        assert rebuilt.dim_labels == parsed.dim_labels
        assert rebuilt._source_type == parsed._source_type == "ome-zarr"

    def test_the_levels_are_not_opened_to_size_the_pyramid(self, tmp_path, monkeypatch):
        import zarr
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        path, _, _ = create_multiresolution_ome_zarr(str(tmp_path), n_levels=3)
        source = _source(path, "ome-zarr")
        parsed = OmeZarrAdapter.create_from_config(source)
        expected = parsed.get_native_pyramid_levels()
        from tests.payload_equivalence import stored

        payload, metadata = stored(parsed)
        rebuilt = OmeZarrAdapter.create_from_payload(source, payload, metadata)

        with monkeypatch.context() as m:
            m.setattr(zarr, "open_array", lambda *a, **k: pytest.fail("opened a level"))
            levels = rebuilt.get_native_pyramid_levels()

        assert levels == expected

    def test_a_row_without_zattrs_is_parsed_instead(self, tmp_path):
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        from tests.payload_equivalence import stored

        path, _, _ = create_multiresolution_ome_zarr(str(tmp_path), n_levels=2)
        source = _source(path, "ome-zarr")
        payload, _ = stored(OmeZarrAdapter.create_from_config(source))
        assert OmeZarrAdapter.create_from_payload(source, payload, {}) is None


def _write_plate(root, wells=("A01", "A02", "B01"), fields=2, shape=(2, 16, 16)):
    """An HCS plate: ``<well>/<field>/0`` arrays under a plate ``.zattrs``."""
    import zarr

    os.makedirs(root, exist_ok=True)
    plate = {
        "plate": {
            "rows": [{"name": "A"}, {"name": "B"}],
            "columns": [{"name": "1"}, {"name": "2"}],
            "wells": [{"path": w} for w in wells],
        }
    }
    with open(os.path.join(root, ".zattrs"), "w") as f:
        json.dump(plate, f)
    multiscales = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 0.5, 0.5]}
                        ],
                    }
                ],
            }
        ],
        "omero": {"channels": [{"label": "dapi"}, {"label": "gfp"}]},
    }
    for i, well in enumerate(wells):
        well_dir = os.path.join(root, well)
        os.makedirs(well_dir, exist_ok=True)
        with open(os.path.join(well_dir, ".zattrs"), "w") as f:
            json.dump(
                {"well": {"images": [{"path": str(k)} for k in range(fields)]}}, f
            )
        for k in range(fields):
            field_dir = os.path.join(well_dir, str(k))
            os.makedirs(field_dir, exist_ok=True)
            with open(os.path.join(field_dir, ".zattrs"), "w") as f:
                json.dump(multiscales, f)
            arr = zarr.open_array(
                os.path.join(field_dir, "0"),
                mode="w",
                shape=shape,
                chunks=(1, 8, 8),
                dtype="uint16",
            )
            arr[:] = (
                np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape) + i + k
            )
    return root


class TestOmeZarrPlate:
    def test_a_plate_is_rebuilt_without_reading_a_well(self, tmp_path, monkeypatch):
        import zarr
        from biopb_tensor_server.adapters import ome_zarr
        from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter

        root = _write_plate(str(tmp_path / "plate.zarr"))
        source = _source(root, "ome-zarr")
        parsed = OmeZarrAdapter.create_from_config(source)
        assert parsed._is_hcs_plate and len(parsed.list_tensors()) == 6

        rebuilt = assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[
                (zarr, "open_array"),
                (zarr, "open_group"),
                (zarr, "DirectoryStore"),
                (OmeZarrAdapter, "_read_zattrs_at"),
                (ome_zarr.json, "load"),
            ],
        )

        assert rebuilt._is_hcs_plate
        assert rebuilt._source_type == "ome-zarr-hcs"
        assert (
            rebuilt.registration_record([], import_rois=False).metadata
            == parsed.registration_record([], import_rois=False).metadata
        )
        assert rebuilt.channel_names == ["dapi", "gfp"]


class TestNdTiff:
    def test_a_rebuilt_adapter_opens_the_acquisition_on_its_first_read(
        self, tmp_path, monkeypatch
    ):
        ndtiff = pytest.importorskip("ndtiff")
        from biopb_tensor_server.adapters.ndtiff import NdTiffAdapter

        from tests.ndtiff_test import _FakeDataset

        data = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
        opens = []
        monkeypatch.setattr(
            ndtiff, "NDTiffDataset", lambda *a, **k: _FakeDataset(data, opens)
        )
        source = _source(tmp_path, "ndtiff")
        parsed = NdTiffAdapter.create_from_config(source)
        assert len(opens) == 1

        rebuilt = assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(ndtiff, "NDTiffDataset")]
        )

        # The parse opened one and pooled it; the rebuilt adapter opens none, and
        # its first read goes through the one the parse left.
        assert len(opens) == 1
        assert rebuilt._reopen is not None


class TestTiffSequence:
    def _sequence(self, tmp_path, **kw):
        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        from tests.tiff_sequence_test import _write_seq

        _write_seq(tmp_path, **kw)
        source = _source(tmp_path, "tiff-sequence")
        return source, TiffSequenceAdapter.create_from_config(source)

    def test_a_rebuilt_sequence_opens_no_member(self, tmp_path, monkeypatch):
        import tifffile

        source, parsed = self._sequence(tmp_path)

        rebuilt = assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(tifffile, "TiffFile")]
        )

        assert (
            rebuilt.registration_record([], import_rois=False).metadata
            == parsed.registration_record([], import_rois=False).metadata
        )
        assert rebuilt._physical_scale() == parsed._physical_scale()

    def test_multipage_members_keep_their_page_axis(self, tmp_path, monkeypatch):
        import tifffile
        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        from tests.tiff_sequence_test import _N

        for i in range(1, _N + 1):
            data = (np.arange(3 * 8 * 8, dtype=np.uint16) + i).reshape(3, 8, 8)
            tifffile.imwrite(
                str(tmp_path / f"s1-{i:04d}_bf.tif"), data, photometric="minisblack"
            )
        source = _source(tmp_path, "tiff-sequence")
        parsed = TiffSequenceAdapter.create_from_config(source)
        assert parsed.dim_labels == ["i", "z", "y", "x"]

        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(tifffile, "TiffFile")]
        )

    def test_a_file_left_out_of_the_stack_is_still_listed(self, tmp_path, monkeypatch):
        import tifffile
        from biopb_tensor_server.adapters.tiff import TiffSequenceAdapter

        from tests.tiff_sequence_test import _write_seq, _write_tiff

        _write_seq(tmp_path)
        _write_tiff(tmp_path / "s1-9999_bf.tif", shape=(2, 8, 8))  # another page count
        source = _source(tmp_path, "tiff-sequence")
        parsed = TiffSequenceAdapter.create_from_config(source)
        assert parsed.registration_record([], import_rois=False).metadata.get(
            "unstacked_files"
        )

        rebuilt = assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(tifffile, "TiffFile")]
        )

        assert rebuilt.registration_record([], import_rois=False).metadata[
            "unstacked_files"
        ] == ["s1-9999_bf.tif"]


def _write_micromanager(root, positions=1, frames=2, channels=3, slices=2):
    """A legacy Micro-Manager dataset: one TIFF per plane and a ``metadata.txt``."""
    import tifffile

    meta = {
        "Summary": {
            "Positions": positions,
            "Frames": frames,
            "Channels": channels,
            "Slices": slices,
            "AxisOrder": ["time", "channel", "z", "position"],
            "PixelSize_um": 0.65,
        }
    }
    n = 0
    for p in range(positions):
        for t in range(frames):
            for c in range(channels):
                for z in range(slices):
                    name = (
                        f"img_channel{c:03d}_position{p:03d}_time{t:09d}_z{z:03d}.tif"
                    )
                    plane = (np.arange(16 * 24, dtype=np.uint16) + n).reshape(16, 24)
                    tifffile.imwrite(
                        os.path.join(root, name), plane, photometric="minisblack"
                    )
                    meta[f"Coords-{name}"] = {
                        "PositionIndex": p,
                        "TimeIndex": t,
                        "ChannelIndex": c,
                        "SliceIndex": z,
                    }
                    n += 1
    with open(os.path.join(root, "metadata.txt"), "w") as f:
        json.dump(meta, f)
    return str(root)


class TestMicroManagerLegacy:
    @pytest.mark.parametrize("positions", [1, 2])
    def test_a_rebuilt_dataset_reads_no_metadata_file_and_opens_no_plane(
        self, tmp_path, monkeypatch, positions
    ):
        import tifffile
        from biopb_tensor_server.adapters import tiff
        from biopb_tensor_server.adapters.tiff import MicroManagerLegacyAdapter

        path = _write_micromanager(tmp_path, positions=positions)
        source = _source(path, "micromanager-legacy")
        parsed = MicroManagerLegacyAdapter.create_from_config(source)

        rebuilt = assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[
                (tifffile, "TiffFile"),
                (tiff.json, "load"),
                (MicroManagerLegacyAdapter, "_find_metadata_file"),
            ],
        )

        assert rebuilt._coord_map == parsed._coord_map
        assert rebuilt._index_to_file == parsed._index_to_file

    def test_a_row_without_metadata_is_parsed_instead(self, tmp_path):
        from biopb_tensor_server.adapters.tiff import MicroManagerLegacyAdapter

        from tests.payload_equivalence import stored

        path = _write_micromanager(tmp_path)
        source = _source(path, "micromanager-legacy")
        payload, _ = stored(MicroManagerLegacyAdapter.create_from_config(source))
        assert (
            MicroManagerLegacyAdapter.create_from_payload(source, payload, {}) is None
        )


class TestRestoreHydratesFromThePayload:
    """Through the restore: a restarted server serves these without opening them."""

    def test_a_restored_sequence_and_zarr_register_without_opening_their_files(
        self, tmp_path, monkeypatch
    ):
        import tifffile

        from tests.restore_test import _Run
        from tests.tiff_sequence_test import _write_seq

        first = _Run(tmp_path)
        (first.monitored / "seq").mkdir()
        _write_seq(first.monitored / "seq")
        first.manager._handle_rescan()
        ids = sorted(first.rows())
        first.stop()
        assert len(ids) == 1

        run = _Run(tmp_path)
        run.restore()
        with monkeypatch.context() as m:
            m.setattr(
                tifffile, "TiffFile", lambda *a, **k: pytest.fail("opened a file")
            )
            adapter = run.server.sources.get_registered(ids[0])
        assert adapter is not None
        assert run.rows()[ids[0]]["is_resolved"]
        run.stop()
