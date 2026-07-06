"""Tests for the unresolved-descriptor robustness contract (cloud-storage phase 1).

Covers:
- `require_resolved` / `SourceUnresolvedError` at the read-planning boundary,
- the `is_resident()` residency gate and the `data_resident` descriptor copy.
"""

import os
import tempfile

import pytest
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb_tensor_server.base import _get_read_plan, require_resolved
from biopb_tensor_server.errors import SourceUnresolvedError


def _zarr_available():
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


class TestRequireResolved:
    """An unresolved descriptor must fail legibly, not crash deep in the planner."""

    def test_subclasses_value_error(self):
        # ValueError subclass so existing `except ValueError` guards still catch it.
        assert issubclass(SourceUnresolvedError, ValueError)

    def test_require_resolved_raises_on_missing_shape_and_dtype(self):
        desc = TensorDescriptor(array_id="x")  # no shape, no dtype
        with pytest.raises(SourceUnresolvedError) as exc:
            require_resolved(desc)
        assert "unresolved" in str(exc.value)

    def test_require_resolved_raises_on_missing_dtype_only(self):
        desc = TensorDescriptor(array_id="x", shape=[10, 20])  # shape but no dtype
        with pytest.raises(SourceUnresolvedError):
            require_resolved(desc)

    def test_require_resolved_passes_when_resolved(self):
        desc = TensorDescriptor(array_id="x", shape=[10, 20], dtype="uint16")
        require_resolved(desc)  # does not raise

    def test_get_read_plan_raises_source_unresolved(self):
        # The boundary the doc names: empty descriptor -> clean SourceUnresolvedError,
        # not a raw numpy/ValueError from dtype('')/ndim logic.
        base_desc = TensorDescriptor(array_id="x")
        request_desc = TensorDescriptor(array_id="x")
        with pytest.raises(SourceUnresolvedError):
            _get_read_plan(base_desc, request_desc, chunk_size=())


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestResolvedAdapterRegression:
    """A normal resolved source still plans, builds a schema, and reports resident."""

    def _make_adapter(self, tmpdir):
        import zarr
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        zarr_path = os.path.join(tmpdir, "test.zarr")
        arr = zarr.open_array(
            zarr_path, mode="w", shape=(100, 200), chunks=(50, 100), dtype="uint16"
        )
        return ZarrAdapter(arr, "test-array", ["y", "x"])

    def test_read_plan_and_schema_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = self._make_adapter(tmpdir)
            plan = adapter.get_read_plan(adapter.get_tensor_descriptor())
            assert list(plan.descriptor.shape) == [100, 200]
            schema = adapter.get_arrow_schema()
            assert schema is not None

    def test_local_source_is_resident(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = self._make_adapter(tmpdir)
            assert adapter.is_resident() is True
            assert adapter.get_source_descriptor().data_resident is True


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestRemoteSourceResidency:
    """A remote (fsspec) source is never resident until materialized locally."""

    def test_remote_url_not_resident(self):
        import zarr
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(10, 10), chunks=(5, 5), dtype="uint8"
            )
            adapter = ZarrAdapter(arr, "remote-array", ["y", "x"])
            # Simulate a remote source by overriding the URL the residency check reads.
            adapter._source_url = "s3://bucket/remote.zarr"

            assert adapter.is_resident() is False
            assert adapter.get_source_descriptor().data_resident is False
