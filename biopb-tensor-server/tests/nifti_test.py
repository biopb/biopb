"""Tests for NIfTI adapter."""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from biopb_tensor_server.adapters.nifti import NiftiAdapter
from biopb_tensor_server.discovery import SourceClaim


def create_synthetic_nifti(
    path: Path,
    shape: tuple = (64, 64, 32),
    dtype: np.dtype = np.float32,
    affine: Optional[np.ndarray] = None,
    pixdim: Optional[tuple] = None,
    intent_code: int = 0,
):
    """Create a synthetic NIfTI file for testing.

    Args:
        path: Where to save the NIfTI file
        shape: Shape of the data array
        dtype: Data type
        affine: Optional 4x4 affine transformation matrix
        pixdim: Optional pixel dimensions tuple (qfac, dx, dy, dz, dt, du, dv, dw)
        intent_code: NIfTI intent code
    """
    import nibabel as nib

    # Create data array
    data = np.random.randn(*shape).astype(dtype)

    # Default affine (identity with voxel spacing)
    if affine is None:
        if pixdim is not None and len(pixdim) >= 4:
            dx, dy, dz = pixdim[1], pixdim[2], pixdim[3]
        else:
            dx, dy, dz = 1.0, 1.0, 1.0
        affine = np.array([
            [dx, 0, 0, 0],
            [0, dy, 0, 0],
            [0, 0, dz, 0],
            [0, 0, 0, 1],
        ])

    # Create NIfTI image
    img = nib.Nifti1Image(data, affine)

    # Set header fields
    header = img.header
    if pixdim is not None:
        header['pixdim'] = list(pixdim) + [0] * (8 - len(pixdim))
    header['intent_code'] = intent_code
    # nibabel xyzt_units encoding: spatial + temporal (2=mm, 8=sec)
    # For 4D+ data, include time units; for 3D, just spatial
    if len(shape) >= 4:
        header['xyzt_units'] = 10  # mm + seconds
    else:
        header['xyzt_units'] = 2  # mm only
    header['descrip'] = 'test_nifti'

    nib.save(img, str(path))


class TestNiftiAdapterClaim:
    """Tests for NiftiAdapter.claim()."""

    def test_claim_nii_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test.nii'
            create_synthetic_nifti(nii_path)

            visited = set()
            claim = NiftiAdapter.claim(nii_path, visited)

            assert claim is not None
            assert claim.source_type == "nifti"
            assert claim.primary_path == nii_path

    def test_claim_nii_gz_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test.nii.gz'
            create_synthetic_nifti(nii_path)

            visited = set()
            claim = NiftiAdapter.claim(nii_path, visited)

            assert claim is not None
            assert claim.source_type == "nifti"

    def test_claim_non_nifti_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / 'test.txt'
            txt_path.write_text('not a nifti file')

            visited = set()
            claim = NiftiAdapter.claim(txt_path, visited)

            assert claim is None

    def test_claim_directory(self):
        """Directories should not be claimed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            visited = set()
            claim = NiftiAdapter.claim(dir_path, visited)

            assert claim is None


class TestNiftiAdapter:
    """Tests for NiftiAdapter functionality."""

    def test_init_and_shape_3d(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test.nii'
            shape = (32, 64, 128)
            create_synthetic_nifti(nii_path, shape=shape)

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            assert adapter._shape == shape
            # NIfTI always reports float64 (scaled data with slope/intercept)
            assert adapter._dtype == 'float64'
            assert adapter.dim_labels == ['z', 'y', 'x']

    def test_init_and_shape_4d(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test_4d.nii'
            shape = (10, 32, 64, 128)  # Time + 3D spatial
            create_synthetic_nifti(nii_path, shape=shape)

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            assert adapter._shape == shape
            # Should detect time series
            assert adapter.dim_labels == ['t', 'z', 'y', 'x']

    def test_get_tensor_descriptor(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test.nii'
            shape = (32, 32, 32)
            create_synthetic_nifti(nii_path, shape=shape)

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            desc = adapter.get_tensor_descriptor()
            assert desc.array_id == 'test_source'
            assert list(desc.shape) == list(shape)
            # NIfTI always reports float64 (scaled data with slope/intercept)
            assert desc.dtype == 'float64'
            # Single chunk
            assert list(desc.chunk_shape) == list(shape)

    
    def test_get_metadata_affine(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test.nii'

            # Custom affine matrix
            affine = np.array([
                [2.0, 0, 0, -64],
                [0, 2.0, 0, -64],
                [0, 0, 3.0, -48],
                [0, 0, 0, 1],
            ])
            create_synthetic_nifti(nii_path, affine=affine)

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            metadata = adapter.get_metadata()

            assert metadata['format'] == 'nifti'
            assert 'spatial' in metadata
            assert 'affine_matrix' in metadata['spatial']

            # Check affine matches
            retrieved_affine = np.array(metadata['spatial']['affine_matrix'])
            np.testing.assert_array_almost_equal(retrieved_affine, affine)

    def test_get_metadata_voxel_size(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test.nii'

            # Custom voxel sizes (pixdim: qfac, dx, dy, dz)
            pixdim = (-1.0, 1.5, 1.5, 2.0)
            create_synthetic_nifti(nii_path, pixdim=pixdim)

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            metadata = adapter.get_metadata()

            assert metadata['spatial']['voxel_size_mm'] == [1.5, 1.5, 2.0]
            assert metadata['spatial']['units'] == 'mm'

    def test_dtype_conversion_uint8(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test_uint8.nii'
            shape = (16, 16, 16)
            data = np.random.randint(0, 255, size=shape, dtype='uint8')
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, str(nii_path))

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            # NIfTI always reports float64 (scaled data with slope/intercept)
            assert adapter._dtype == 'float64'

    def test_dtype_conversion_int16(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test_int16.nii'
            shape = (16, 16, 16)
            data = np.random.randint(-1000, 1000, size=shape, dtype='int16')
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, str(nii_path))

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            # NIfTI always reports float64 (scaled data with slope/intercept)
            assert adapter._dtype == 'float64'

    def test_intent_code_interpretation(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test_intent.nii'
            create_synthetic_nifti(nii_path, intent_code=1002)  # Label

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            metadata = adapter.get_metadata()
            assert metadata['header']['intent'] == 'label'


class TestNiftiAdapterIntegration:
    """Integration tests for NIfTI adapter with nibabel."""

    def test_5d_nifti(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test_5d.nii'
            shape = (5, 10, 32, 32, 32)  # Vector + time + 3D
            create_synthetic_nifti(nii_path, shape=shape)

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_5d')

            assert len(adapter._shape) == 5
            assert adapter.dim_labels == ['v', 't', 'z', 'y', 'x']

    def test_header_fields_preserved(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            nii_path = Path(tmpdir) / 'test_fields.nii'

            data = np.zeros((16, 16, 16), dtype='float32')
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)

            header = img.header
            header['cal_min'] = 0.0
            header['cal_max'] = 100.0
            header['descrip'] = 'test_description'
            header['aux_file'] = 'aux.txt'

            nib.save(img, str(nii_path))

            img = nib.load(str(nii_path))
            adapter = NiftiAdapter(img, 'test_source')

            metadata = adapter.get_metadata()

            assert 'header' in metadata
            assert metadata['header']['cal_min'] == 0.0
            assert metadata['header']['cal_max'] == 100.0
            # descrip and aux_file should be in header