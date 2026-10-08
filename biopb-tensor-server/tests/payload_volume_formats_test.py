"""mrc, emd, qptiff, dicom and nifti rebuilt from their payload.

Each is rebuilt from what its registration stored and has to serve what the parsed
adapter serves (``assert_hydrates_equivalently``), with the reader that parsed the
file refused for the length of the rebuild.
"""

import numpy as np
import pytest
from biopb_tensor_server.core.config import SourceConfig

from tests import payload_equivalence
from tests.payload_equivalence import assert_hydrates_equivalently, hydrate


@pytest.fixture(autouse=True)
def _metadata_as_the_row_holds_it(monkeypatch):
    """Compare metadata as the catalog row stores it.

    A restore serves the row's metadata, which went through the row's JSON encoder
    (a pydicom ``PersonName`` becomes a string there), so the parsed adapter's raw
    ``get_metadata()`` is compared after the same round trip.
    """
    import json

    from biopb_tensor_server.serving.metadata_db import NumpyEncoder

    original = payload_equivalence.snapshot

    def snapshot(adapter, **kwargs):
        out = original(adapter, **kwargs)
        out["metadata"] = json.loads(json.dumps(out["metadata"], cls=NumpyEncoder))
        return out

    monkeypatch.setattr(payload_equivalence, "snapshot", snapshot)


def _row_view(metadata):
    import json

    from biopb_tensor_server.serving.metadata_db import NumpyEncoder

    return json.loads(json.dumps(metadata, cls=NumpyEncoder))


class TestMrc:
    def _parsed(self, tmp_path):
        pytest.importorskip("rsciio")
        from biopb_tensor_server.adapters import MrcAdapter

        from tests.mrc_test import create_synthetic_mrc

        path = tmp_path / "vol.mrc"
        create_synthetic_mrc(path, shape=(4, 8, 8), cell=(20.0, 20.0, 40.0))
        source = SourceConfig(url=str(path), type="mrc", source_id="src")
        return MrcAdapter.create_from_config(source), source

    def test_a_rebuilt_adapter_is_the_parsed_one_without_the_reader(
        self, tmp_path, monkeypatch
    ):
        import rsciio.mrc

        parsed, source = self._parsed(tmp_path)
        assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[(rsciio.mrc, "file_reader")],
        )

    def test_the_calibration_survives(self, tmp_path):
        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        assert rebuilt._physical_scale() == parsed._physical_scale()
        assert rebuilt._physical_scale() is not None

    def test_the_mapping_is_not_probed_or_held(self, tmp_path):
        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        assert rebuilt._persistent_map is None
        data = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
        from biopb.tensor.ticket_pb2 import ChunkBounds

        got = rebuilt.get_data(ChunkBounds(start=[1, 0, 0], stop=[3, 8, 8]))
        assert np.array_equal(got, data[1:3])


class TestEmd:
    def _parsed(self, tmp_path):
        pytest.importorskip("rsciio")
        pytest.importorskip("h5py")
        from biopb_tensor_server.adapters import EmdAdapter

        from tests.emd_test import create_synthetic_emd

        path = tmp_path / "data.emd"
        create_synthetic_emd(path)
        source = SourceConfig(url=str(path), type="emd", source_id="src")
        return EmdAdapter.create_from_config(source), source

    def test_a_rebuilt_adapter_is_the_parsed_one_without_the_reader(
        self, tmp_path, monkeypatch
    ):
        import rsciio.emd

        parsed, source = self._parsed(tmp_path)
        assert_hydrates_equivalently(
            parsed,
            source,
            monkeypatch=monkeypatch,
            opens=[(rsciio.emd, "file_reader")],
        )

    def test_a_signals_own_metadata_is_served_without_the_file(
        self, tmp_path, monkeypatch
    ):
        import rsciio.emd

        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        field = f"{source.source_id}/0"
        with monkeypatch.context() as m:
            m.setattr(rsciio.emd, "file_reader", lambda *a, **k: 1 / 0)
            tensor = rebuilt.get_tensor_adapter(field)
            assert (
                tensor.get_tensor_metadata()
                == parsed.get_tensor_adapter(field).get_tensor_metadata()
            )
            assert tensor.get_tensor_descriptor().chunk_shape
            assert tensor._physical_scale() is not None

    def test_the_file_is_read_once_for_all_signals(self, tmp_path, monkeypatch):
        import rsciio.emd
        from biopb.tensor.ticket_pb2 import ChunkBounds

        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        calls = []
        real = rsciio.emd.file_reader
        monkeypatch.setattr(
            rsciio.emd, "file_reader", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        tensor = rebuilt.get_tensor_adapter(f"{source.source_id}/0")
        shape = list(tensor.get_tensor_descriptor().shape)
        for _ in range(2):
            tensor.get_data(ChunkBounds(start=[0] * len(shape), stop=shape))
        assert len(calls) == 1


class TestQptiff:
    def _parsed(self, tmp_path):
        pytest.importorskip("imagecodecs")
        pytest.importorskip("tifffile")
        from biopb_tensor_server.adapters import QptiffAdapter

        from tests.qptiff_test import create_synthetic_qptiff

        path = tmp_path / "slide.qptiff"
        create_synthetic_qptiff(path, n_channels=3, base=512, n_levels=3)
        source = SourceConfig(url=str(path), type="qptiff", source_id="src")
        return QptiffAdapter.create_from_config(source), source

    def test_a_rebuilt_adapter_is_the_parsed_one_without_opening_the_file(
        self, tmp_path, monkeypatch
    ):
        import tifffile

        parsed, source = self._parsed(tmp_path)
        rebuilt = assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(tifffile, "TiffFile")]
        )
        rebuilt.close()

    def test_the_native_pyramid_and_scale_come_from_the_row(
        self, tmp_path, monkeypatch
    ):
        import tifffile

        parsed, source = self._parsed(tmp_path)
        want_levels = [lv.shape[:] for lv in parsed.get_native_pyramid_levels()]
        rebuilt = hydrate(parsed, source)
        with monkeypatch.context() as m:
            m.setattr(tifffile, "TiffFile", lambda *a, **k: 1 / 0)
            got = rebuilt.get_native_pyramid_levels()
            assert [lv.shape[:] for lv in got] == want_levels
            assert len(got) == 3
            assert rebuilt.has_native_pyramid()
            assert rebuilt._find_level_for_scale((1, 2, 2)) == 1
            assert rebuilt.catalog_payload() is not None
            assert (
                rebuilt.registration_record([], import_rois=False).metadata["channels"][
                    0
                ]
                == "DAPI"
            )

    def test_a_level_adapter_reads_through_a_handle_opened_on_demand(self, tmp_path):
        from biopb.tensor.ticket_pb2 import ChunkBounds

        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        level = rebuilt.get_tensor_adapter("1")
        shape = list(level.get_tensor_descriptor().shape)
        want = parsed.get_tensor_adapter("1").get_data(
            ChunkBounds(start=[0] * len(shape), stop=shape)
        )
        got = level.get_data(ChunkBounds(start=[0] * len(shape), stop=shape))
        assert np.array_equal(got, want)
        rebuilt.close()
        parsed.close()


class TestDicom:
    def _file(self, tmp_path, **kwargs):
        pytest.importorskip("pydicom")
        from biopb_tensor_server.adapters import DicomAdapter

        from tests.dicom_test import create_synthetic_dicom

        path = tmp_path / "one.dcm"
        create_synthetic_dicom(
            path,
            rows=24,
            cols=32,
            pixel_spacing=(0.5, 0.25),
            patient_name="Rebuilt",
            **kwargs,
        )
        source = SourceConfig(url=str(path), type="dicom", source_id="src")
        return DicomAdapter.create_from_config(source), source

    @pytest.mark.parametrize("multi_frame", [0, 5])
    def test_a_rebuilt_file_is_the_parsed_one_without_the_reader(
        self, tmp_path, monkeypatch, multi_frame
    ):
        import pydicom

        parsed, source = self._file(tmp_path, multi_frame=multi_frame)
        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(pydicom, "dcmread")]
        )

    def test_the_scale_and_metadata_come_from_the_row(self, tmp_path):
        parsed, source = self._file(tmp_path)
        rebuilt = hydrate(parsed, source)
        assert rebuilt._physical_scale() == parsed._physical_scale()
        assert rebuilt._physical_scale() is not None
        assert rebuilt.registration_record([], import_rois=False).metadata == _row_view(
            parsed.registration_record([], import_rois=False).metadata
        )

    def test_a_remote_file_is_parsed(self):
        from biopb_tensor_server.adapters import DicomAdapter

        source = SourceConfig(url="s3://bucket/one.dcm", type="dicom", source_id="s")
        assert DicomAdapter.create_from_payload(source, {}, {}) is None


class TestDicomSeries:
    def _series(self, tmp_path, n=4):
        pytest.importorskip("pydicom")
        from biopb_tensor_server.adapters import DicomSeriesAdapter
        from pydicom.uid import generate_uid

        from tests.dicom_test import create_synthetic_dicom

        series = tmp_path / "series"
        series.mkdir()
        uid = generate_uid()
        # Written out of order, so the sort (by InstanceNumber) is what orders them.
        for i in reversed(range(n)):
            create_synthetic_dicom(
                series / f"slice_{i}.dcm",
                rows=16,
                cols=20,
                series_uid=uid,
                instance_number=i,
                slice_location=float(i),
                pixel_spacing=(0.5, 0.5),
                pixel_data=np.full((16, 20), i * 10, dtype="uint16"),
            )
        source = SourceConfig(url=str(series), type="dicom-series", source_id="src")
        return DicomSeriesAdapter.create_from_config(source), source

    def test_a_rebuilt_series_is_the_parsed_one_without_reading_a_header(
        self, tmp_path, monkeypatch
    ):
        import pydicom

        parsed, source = self._series(tmp_path)
        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(pydicom, "dcmread")]
        )

    def test_the_slice_order_is_the_row_s(self, tmp_path):
        parsed, source = self._series(tmp_path)
        rebuilt = hydrate(parsed, source)
        assert [f.name for f in rebuilt.dicom_files] == [
            f.name for f in parsed.dicom_files
        ]
        assert rebuilt.dicom_files[0].name == "slice_0.dcm"
        assert rebuilt._physical_scale() == parsed._physical_scale()
        assert rebuilt.registration_record([], import_rois=False).metadata == _row_view(
            parsed.registration_record([], import_rois=False).metadata
        )


class TestNifti:
    def _parsed(self, tmp_path, shape=(8, 6, 4), pixdim=(1.0, 0.5, 0.5, 2.0)):
        pytest.importorskip("nibabel")
        from biopb_tensor_server.adapters import NiftiAdapter

        from tests.nifti_test import create_synthetic_nifti

        path = tmp_path / "vol.nii"
        create_synthetic_nifti(path, shape=shape, pixdim=pixdim)
        source = SourceConfig(url=str(path), type="nifti", source_id="src")
        return NiftiAdapter.create_from_config(source), source

    @pytest.mark.parametrize("shape", [(8, 6, 4), (8, 6, 4, 3)])
    def test_a_rebuilt_adapter_is_the_parsed_one_without_loading_the_image(
        self, tmp_path, monkeypatch, shape
    ):
        import nibabel

        parsed, source = self._parsed(tmp_path, shape=shape)
        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(nibabel, "load")]
        )

    def test_the_image_loads_once_on_the_first_read(self, tmp_path, monkeypatch):
        import nibabel
        from biopb.tensor.ticket_pb2 import ChunkBounds

        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        loads = []
        real = nibabel.load
        monkeypatch.setattr(
            nibabel, "load", lambda *a, **k: loads.append(1) or real(*a, **k)
        )
        assert rebuilt._physical_scale() == parsed._physical_scale()
        assert rebuilt.registration_record([], import_rois=False).metadata == _row_view(
            parsed.registration_record([], import_rois=False).metadata
        )
        assert loads == []
        shape = [4, 6, 8]
        for _ in range(2):
            rebuilt.get_data(ChunkBounds(start=[0, 0, 0], stop=shape))
        assert len(loads) == 1

    def test_a_closed_rebuilt_source_does_not_load_again(self, tmp_path):
        from biopb.tensor.ticket_pb2 import ChunkBounds

        parsed, source = self._parsed(tmp_path)
        rebuilt = hydrate(parsed, source)
        rebuilt.close()
        with pytest.raises(RuntimeError, match="closed"):
            rebuilt.get_data(ChunkBounds(start=[0, 0, 0], stop=[4, 6, 8]))

    def test_a_remote_file_is_parsed(self):
        from biopb_tensor_server.adapters import NiftiAdapter

        source = SourceConfig(url="s3://bucket/v.nii.gz", type="nifti", source_id="s")
        assert NiftiAdapter.create_from_payload(source, {}, {}) is None
