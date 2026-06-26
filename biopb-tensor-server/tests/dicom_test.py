"""Tests for DICOM adapters."""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from biopb_tensor_server.adapters.dicom import (
    DicomAdapter,
    DicomSeriesAdapter,
    _derive_orientation_from_iop,
)
from biopb_tensor_server.discovery import ClaimContext, DiscoveryState

# Every test here builds/reads DICOM via pydicom (the [dicom]/[medical] extra);
# skip the whole module when it is not installed rather than erroring.
pytest.importorskip("pydicom")


def create_synthetic_dicom(
    path: Path,
    rows: int = 64,
    cols: int = 64,
    pixel_data: Optional[np.ndarray] = None,
    series_uid: Optional[str] = None,
    instance_number: Optional[int] = None,
    slice_location: Optional[float] = None,
    pixel_spacing: Optional[tuple] = None,
    bits_stored: int = 16,
    pixel_representation: int = 0,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    rescale_slope: Optional[float] = None,
    rescale_intercept: Optional[float] = None,
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    multi_frame: int = 0,
):
    """Create a synthetic DICOM file for testing.

    Args:
        path: Where to save the DICOM file
        rows: Number of rows (height)
        cols: Number of columns (width)
        pixel_data: Optional pixel data array. If None, creates random data.
        series_uid: Optional SeriesInstanceUID
        instance_number: Optional InstanceNumber
        slice_location: Optional SliceLocation
        pixel_spacing: Optional PixelSpacing tuple
        bits_stored: BitsStored value (8, 16, or 32)
        pixel_representation: 0 for unsigned, 1 for signed
        window_center: Optional WindowCenter
        window_width: Optional WindowWidth
        rescale_slope: Optional RescaleSlope
        rescale_intercept: Optional RescaleIntercept
        patient_id: Optional PatientID
        patient_name: Optional PatientName
        multi_frame: Number of frames (0 for single frame)
    """
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid

    # Create file meta info
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian

    # Create dataset
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b'\x00' * 128)

    # Set required DICOM tags
    ds.PatientName = patient_name or 'Test Patient'
    ds.PatientID = patient_id or 'TEST001'
    ds.Modality = 'CT'
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # Image dimensions
    ds.Rows = rows
    ds.Columns = cols

    if multi_frame > 0:
        ds.NumberOfFrames = multi_frame
        total_pixels = rows * cols * multi_frame
    else:
        total_pixels = rows * cols

    # Pixel data type
    ds.BitsStored = bits_stored
    ds.BitsAllocated = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = pixel_representation
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'

    # Create pixel data
    if pixel_data is None:
        if bits_stored <= 8:
            dtype = 'uint8' if pixel_representation == 0 else 'int8'
        elif bits_stored <= 16:
            dtype = 'uint16' if pixel_representation == 0 else 'int16'
        else:
            dtype = 'uint32' if pixel_representation == 0 else 'int32'
        pixel_data = np.random.randint(0, 1000, size=total_pixels, dtype=dtype)
        if multi_frame > 0:
            pixel_data = pixel_data.reshape((multi_frame, rows, cols))
        else:
            pixel_data = pixel_data.reshape((rows, cols))

    ds.PixelData = pixel_data.tobytes()

    # Optional tags
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    if slice_location is not None:
        ds.SliceLocation = slice_location
    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept

    # Spatial orientation (axial by default)
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, slice_location or 0]
    ds.SliceThickness = 1.0

    # Save
    ds.save_as(str(path))


class TestDicomEncoding:
    """Tests for DICOM orientation derivation."""

    def test_derive_orientation_axial(self):
        iop = [1, 0, 0, 0, 1, 0]
        orientation = _derive_orientation_from_iop(iop)
        assert orientation == "axial"

    def test_derive_orientation_sagittal(self):
        # Sagittal: slice normal along x-axis
        iop = [0, 1, 0, 0, 0, 1]
        orientation = _derive_orientation_from_iop(iop)
        assert orientation == "sagittal"

    def test_derive_orientation_coronal(self):
        # Coronal: slice normal along y-axis
        iop = [1, 0, 0, 0, 0, 1]
        orientation = _derive_orientation_from_iop(iop)
        assert orientation == "coronal"


class TestDicomAdapterClaim:
    """Tests for DicomAdapter.claim()."""

    def test_claim_valid_dicom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'test.dcm'
            create_synthetic_dicom(dcm_path)

            ctx = ClaimContext(dcm_path)
            state = DiscoveryState()
            claim = DicomAdapter.claim(ctx, state)

            assert claim is not None
            assert claim.source_type == "dicom"
            assert claim.primary_path == str(dcm_path)

    def test_claim_non_dicom_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-DICOM file
            txt_path = Path(tmpdir) / 'test.txt'
            txt_path.write_text('not a dicom file')

            ctx = ClaimContext(txt_path)
            state = DiscoveryState()
            claim = DicomAdapter.claim(ctx, state)

            assert claim is None

    def test_claim_dicom_without_pixel_data(self):
        """Test that DICOM files without image tags are not claimed."""
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'no_pixel.dcm'

            # Create DICOM without image-related tags (no Rows/Columns)
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.9'  # Basic Text SR
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'

            ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b'\x00' * 128)
            ds.PatientName = 'Test'
            ds.PatientID = 'TEST'
            ds.Modality = 'SR'  # Structured Report (no pixel data)
            # No Rows, Columns, or pixel data tags!
            ds.save_as(str(dcm_path))

            ctx = ClaimContext(dcm_path)
            state = DiscoveryState()
            claim = DicomAdapter.claim(ctx, state)

            assert claim is None

    def test_claim_directory(self):
        """Directories should not be claimed by DicomAdapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            ctx = ClaimContext(dir_path)
            state = DiscoveryState()
            claim = DicomAdapter.claim(ctx, state)

            assert claim is None


class TestDicomAdapter:
    """Tests for DicomAdapter functionality."""

    def test_init_and_shape(self):
        import pydicom

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'test.dcm'
            rows, cols = 128, 64
            create_synthetic_dicom(dcm_path, rows=rows, cols=cols)

            ds = pydicom.dcmread(str(dcm_path))
            adapter = DicomAdapter(ds, 'test_source')

            assert adapter._shape == (rows, cols)
            assert adapter._dtype == 'uint16'
            assert adapter.dim_labels == ['y', 'x']

    def test_multiframe_dicom(self):
        import pydicom

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'multiframe.dcm'
            rows, cols = 32, 32
            num_frames = 10
            create_synthetic_dicom(dcm_path, rows=rows, cols=cols, multi_frame=num_frames)

            ds = pydicom.dcmread(str(dcm_path))
            adapter = DicomAdapter(ds, 'test_multiframe')

            assert adapter._shape == (num_frames, rows, cols)
            assert adapter._is_multiframe
            assert adapter.dim_labels == ['frame', 'y', 'x']

    def test_get_tensor_descriptor(self):
        import pydicom

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'test.dcm'
            create_synthetic_dicom(dcm_path)

            ds = pydicom.dcmread(str(dcm_path))
            adapter = DicomAdapter(ds, 'test_source')

            desc = adapter.get_tensor_descriptor()
            assert desc.array_id == 'test_source'
            assert len(desc.shape) == 2
            assert desc.dtype == 'uint16'


    def test_get_metadata(self):
        import pydicom

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'test.dcm'
            create_synthetic_dicom(
                dcm_path,
                pixel_spacing=(0.5, 0.5),
                window_center=127.5,
                window_width=255.0,
                rescale_slope=1.0,
                rescale_intercept=-1024.0,
                patient_id='PATIENT123',
                patient_name='John Doe',
            )

            ds = pydicom.dcmread(str(dcm_path))
            adapter = DicomAdapter(ds, 'test_source')

            metadata = adapter.get_metadata()

            assert metadata['format'] == 'dicom'
            assert 'tags' in metadata
            assert 'spatial' in metadata
            assert 'patient' in metadata

            # Check spatial info
            assert metadata['spatial']['pixel_spacing_mm'] == [0.5, 0.5]
            assert metadata['spatial']['orientation'] == 'axial'

            # Check patient info
            assert metadata['patient']['PatientID'] == 'PATIENT123'

    def test_signed_pixel_data(self):
        import pydicom

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'signed.dcm'
            create_synthetic_dicom(dcm_path, bits_stored=16, pixel_representation=1)

            ds = pydicom.dcmread(str(dcm_path))
            adapter = DicomAdapter(ds, 'test_source')

            assert adapter._dtype == 'int16'


class TestDicomSeriesAdapterClaim:
    """Tests for DicomSeriesAdapter.claim()."""

    def test_claim_valid_series(self):
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            series_uid = generate_uid()

            # Create multiple DICOM files with same SeriesInstanceUID
            for i in range(5):
                dcm_path = Path(tmpdir) / f'slice_{i}.dcm'
                create_synthetic_dicom(
                    dcm_path,
                    series_uid=series_uid,
                    instance_number=i,
                    slice_location=float(i),
                )

            ctx = ClaimContext(Path(tmpdir))
            state = DiscoveryState()
            claim = DicomSeriesAdapter.claim(ctx, state)

            assert claim is not None
            assert claim.source_type == "dicom-series"
            assert claim.primary_path == str(Path(tmpdir))
            # Should claim directory + all DICOM files (now tracked in state)
            assert len(state.consumed_paths) >= 6  # dir + 5 files

    def test_claim_single_dicom_directory(self):
        """Directory with only one DICOM should not be claimed as series."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / 'single.dcm'
            create_synthetic_dicom(dcm_path)

            ctx = ClaimContext(Path(tmpdir))
            state = DiscoveryState()
            claim = DicomSeriesAdapter.claim(ctx, state)

            assert claim is None

    def test_claim_mixed_series(self):
        """Directory with DICOMs from different series."""
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            series_uid1 = generate_uid()
            series_uid2 = generate_uid()

            # Create 2 files from series 1
            for i in range(2):
                dcm_path = Path(tmpdir) / f'series1_{i}.dcm'
                create_synthetic_dicom(dcm_path, series_uid=series_uid1)

            # Create 1 file from series 2
            dcm_path = Path(tmpdir) / 'series2_0.dcm'
            create_synthetic_dicom(dcm_path, series_uid=series_uid2)

            ctx = ClaimContext(Path(tmpdir))
            state = DiscoveryState()
            claim = DicomSeriesAdapter.claim(ctx, state)

            # Should still claim the first series with 2+ files
            assert claim is not None
            # Should only claim files from one series
            assert claim.extra_config['num_slices'] == 2


class TestDicomSeriesAdapter:
    """Tests for DicomSeriesAdapter functionality."""

    def test_init_and_shape(self):
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            series_uid = generate_uid()
            num_slices = 10
            rows, cols = 64, 32

            for i in range(num_slices):
                dcm_path = Path(tmpdir) / f'slice_{i}.dcm'
                create_synthetic_dicom(
                    dcm_path,
                    rows=rows,
                    cols=cols,
                    series_uid=series_uid,
                    instance_number=i,
                )

            adapter = DicomSeriesAdapter(tmpdir, 'test_series')

            assert adapter._shape == (num_slices, rows, cols)
            assert adapter._num_slices == num_slices
            assert adapter.dim_labels == ['z', 'y', 'x']

    def test_series_sorting_by_instance_number(self):
        import pydicom
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            series_uid = generate_uid()

            # Create files with non-sequential instance numbers
            instance_numbers = [10, 5, 20, 1, 15]
            for i, inst_num in enumerate(instance_numbers):
                dcm_path = Path(tmpdir) / f'file_{i}.dcm'
                create_synthetic_dicom(
                    dcm_path,
                    series_uid=series_uid,
                    instance_number=inst_num,
                )

            adapter = DicomSeriesAdapter(tmpdir, 'test_series')

            # Files should be sorted by InstanceNumber
            expected_order = sorted(instance_numbers)
            for i, dcm_path in enumerate(adapter.dicom_files):
                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                assert ds.InstanceNumber == expected_order[i]

    def test_get_tensor_descriptor(self):
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            series_uid = generate_uid()
            num_slices = 5
            rows, cols = 32, 32

            for i in range(num_slices):
                dcm_path = Path(tmpdir) / f'slice_{i}.dcm'
                create_synthetic_dicom(
                    dcm_path,
                    rows=rows,
                    cols=cols,
                    series_uid=series_uid,
                )

            adapter = DicomSeriesAdapter(tmpdir, 'test_series')

            desc = adapter.get_tensor_descriptor()
            assert desc.array_id == 'test_series'
            assert desc.shape == [num_slices, rows, cols]
            assert desc.chunk_shape == [1, rows, cols]


    def test_get_metadata(self):
        from pydicom.uid import generate_uid

        with tempfile.TemporaryDirectory() as tmpdir:
            series_uid = generate_uid()
            num_slices = 5

            for i in range(num_slices):
                dcm_path = Path(tmpdir) / f'slice_{i}.dcm'
                create_synthetic_dicom(
                    dcm_path,
                    series_uid=series_uid,
                    pixel_spacing=(0.5, 0.5),
                    patient_id='PATIENT123',
                )

            adapter = DicomSeriesAdapter(tmpdir, 'test_series')

            metadata = adapter.get_metadata()

            assert metadata['format'] == 'dicom'
            assert 'series' in metadata
            assert metadata['series']['num_slices'] == num_slices
            assert metadata['spatial']['pixel_spacing_mm'] == [0.5, 0.5]
            assert metadata['patient']['PatientID'] == 'PATIENT123'
