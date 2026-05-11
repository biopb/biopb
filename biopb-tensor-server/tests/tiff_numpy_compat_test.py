"""Regression test for numpy 2.0 / tifffile compatibility.

Tests that MultiFileOmeTiffAdapter.get_data() works with the pinned
numpy/tifffile versions by verifying tifffile TiffSequence.asarray()
doesn't fail with 'numpy.ndarray' object has no attribute 'newbyteorder'.

This bug occurred with tifffile < 2024.8.10 and numpy >= 2.0.
We pin numpy < 2.0 to maintain compatibility with aicsimageio.
"""

import numpy as np
import pytest

from biopb_tensor_server.adapters.tiff import MultiFileOmeTiffAdapter
from biopb.tensor.ticket_pb2 import ChunkBounds


class TestTiffNumpyCompat:
    """Regression tests for tifffile/numpy compatibility."""

    def test_multifile_tiff_sequence_get_data(self, multifile_ome_dataset):
        """Test that TiffSequence.asarray() works with current numpy version.

        This is a regression test for the bug where tifffile < 2024.8.10
        called ndarray.newbyteorder() which was removed in numpy 2.0.
        """
        dir_path, file_list, metadata = multifile_ome_dataset

        # Create adapter for multi-file dataset
        adapter = MultiFileOmeTiffAdapter(dir_path, 'test-multifile')

        # Get descriptor to understand shape
        desc = adapter.get_tensor_descriptor()

        # Request a slice of the data - this triggers TiffSequence.asarray()
        bounds = ChunkBounds(
            start=[0, 0, 0],
            stop=list(desc.shape),
        )

        # This should NOT raise 'numpy.ndarray' object has no attribute 'newbyteorder'
        data = adapter.get_data(bounds)

        # Verify we got valid data
        assert isinstance(data, np.ndarray)
        assert data.shape == tuple(desc.shape)

    def test_numpy_version_constraint(self):
        """Verify numpy is pinned to < 2.0 for tifffile/aicsimageio compatibility."""
        from packaging.version import Version

        numpy_version = Version(np.__version__)

        # We pin numpy < 2.0 to avoid ndarray.newbyteorder() removal
        # which breaks tifffile < 2024.8.10 and aicsimageio
        assert numpy_version < Version("2.0.0"), (
            f"numpy {numpy_version} is incompatible with tifffile/aicsimageio. "
            f"Pin to numpy < 2.0 in pyproject.toml."
        )