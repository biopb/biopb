"""Tests for the tifffile-direct OME-TIFF descriptor fast path (biopb/biopb#168).

Registering a large OME-TIFF used to materialize the AICSImage OME model just to
learn shape/dims/dtype/scenes for the catalog row -- the ~97 s/dataset that
dominated server startup. The fast path derives the same descriptor straight
from tifffile (~6 s) and defers the AICSImage parse to the first read.

These tests lock the two things that make that safe:
1. Parity -- the tifffile descriptor is byte-identical to the aicsimageio one
   (shape/dims/dtype AND the scene ids that form the catalog array_ids).
2. The AICSImage OME parse is genuinely gone from the registration path, while
   reads still resolve the right scene (via the cached descriptor order ->
   set_scene(int)).
"""

import numpy as np
import pytest
from biopb_tensor_server.adapters.aicsimageio import (
    AicsImageIoAdapter,
    _ome_scene_ids,
    _tczyx_shape,
)
from biopb_tensor_server.fixtures import (
    create_multi_series_ome_tiff,
    create_tiled_ome_tiff,
)


class TestTczyxShape:
    @pytest.mark.parametrize(
        "shape,axes,expected",
        [
            ((2, 3, 16, 16), "CZYX", [1, 2, 3, 16, 16]),
            ((5, 16, 16), "ZYX", [1, 1, 5, 16, 16]),
            ((16, 16), "YX", [1, 1, 1, 16, 16]),
            ((4, 2, 3, 16, 16), "TCZYX", [4, 2, 3, 16, 16]),
            ((3, 16, 16), "CYX", [1, 3, 1, 16, 16]),
        ],
    )
    def test_canonical_axes_map_to_tczyx(self, shape, axes, expected):
        assert _tczyx_shape(shape, axes) == expected

    @pytest.mark.parametrize(
        "shape,axes",
        [
            ((16, 16, 3), "YXS"),  # RGB samples -> aicsimageio folds specially
            ((4, 16, 16), "QYX"),  # unknown axis
            ((2, 16, 16), "CZYX"),  # axes/shape length mismatch
            ((16, 16), ""),  # no axes
        ],
    )
    def test_non_canonical_axes_signal_fallback(self, shape, axes):
        assert _tczyx_shape(shape, axes) is None


class TestOmeSceneIds:
    def test_extracts_image_ids_in_order(self):
        xml = '<OME><Image ID="Image:0"/><Image ID="Image:1"/></OME>'
        assert _ome_scene_ids(xml, 2) == ["Image:0", "Image:1"]

    def test_handles_name_before_id_and_namespace(self):
        xml = '<ome:Image Name="f" ID="Image:7"></ome:Image>'
        assert _ome_scene_ids(xml, 1) == ["Image:7"]

    def test_count_mismatch_falls_back_to_positional(self):
        # Only one ID parsed but tifffile reports two series -> positional.
        xml = '<OME><Image ID="Image:0"/></OME>'
        assert _ome_scene_ids(xml, 2) == ["Image:0", "Image:1"]

    def test_no_xml_falls_back_to_positional(self):
        assert _ome_scene_ids(None, 3) == ["Image:0", "Image:1", "Image:2"]


class _Tripwire:
    """Stand-in for AICSImage that fails on ANY attribute access.

    Installed in place of ``adapter._aics_image`` to prove the registration
    (descriptor) path never touches aicsimageio -- the whole point of the fix.
    """

    def __getattr__(self, name):
        raise AssertionError(f"AICSImage accessed during registration: .{name}")


class TestFastPathParity:
    def _aics_truth(self, path):
        """The descriptor aicsimageio would produce: (scene_id, shape, dtype)."""
        from aicsimageio import AICSImage

        img = AICSImage(path)
        out = []
        for i, scene in enumerate(img.scenes):
            img.set_scene(i)
            out.append((scene, list(img.shape), img.dask_data.dtype.str))
        return out

    def _fast(self, adapter):
        descs = adapter._tifffile_descriptors()
        assert descs is not None, "fast path should apply to this OME-TIFF"
        return [
            (d.array_id.split("/", 1)[1], list(d.shape), d.dtype) for d in descs
        ]

    def test_single_series_parity(self, tmp_path):
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(3, 64, 64))
        adapter = AicsImageIoAdapter.create_from_url(path, "single")
        assert self._fast(adapter) == self._aics_truth(path)

    def test_multi_series_parity(self, tmp_path):
        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        adapter = AicsImageIoAdapter.create_from_url(path, "multi")
        assert self._fast(adapter) == self._aics_truth(path)

    def test_list_descriptors_does_not_parse_aicsimageio(self, tmp_path):
        # The registration path must build the catalog row without ever touching
        # AICSImage (the ome-types parse is the startup cost being removed).
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = AicsImageIoAdapter.create_from_url(path, "noparse")
        adapter._aics_image = _Tripwire()

        descriptors = adapter.list_tensor_descriptors()

        assert len(descriptors) == 1
        assert descriptors[0].array_id == "noparse/Image:0"
        assert list(descriptors[0].shape) == [1, 2, 1, 32, 32]
        assert list(descriptors[0].dim_labels) == list("TCZYX")
        # Cached, and still served from cache without parsing.
        assert adapter._cached_descriptors is descriptors
        assert adapter.list_tensor_descriptors() is descriptors


class TestSceneResolutionAndReads:
    def test_scene_index_resolves_from_cache(self, tmp_path):
        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        adapter = AicsImageIoAdapter.create_from_url(path, "idx")
        descriptors = adapter.list_tensor_descriptors()
        fields = [d.array_id.split("/", 1)[1] for d in descriptors]

        for i, field in enumerate(fields):
            assert adapter._scene_index_for_field(field) == i
        with pytest.raises(ValueError):
            adapter._scene_index_for_field("Image:999")

    def test_reads_select_correct_series_after_fast_path(self, tmp_path):
        # The cached descriptor order must map to the right AICSImage scene at
        # read time (cache index -> set_scene(int)). The fixture writes each
        # series filled with series_idx*100 + plane + 1, so series k plane 0 == k*100+1.
        path, _, _ = create_multi_series_ome_tiff(
            str(tmp_path), n_series=3, series_shape=(2, 32, 32)
        )
        adapter = AicsImageIoAdapter.create_from_url(path, "reads")
        descriptors = adapter.list_tensor_descriptors()

        from biopb.tensor.ticket_pb2 import ChunkBounds

        for k, desc in enumerate(descriptors):
            field = desc.array_id.split("/", 1)[1]
            scene = adapter.get_tensor_adapter(field)
            # shape is TCZYX = (1, 2, 1, 32, 32); read the (0,0,0,0,0) corner.
            bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 1, 1])
            val = np.asarray(scene.get_data(bounds)).ravel()[0]
            assert val == k * 100 + 1, f"series {k} returned {val}"


class TestFallback:
    def test_remote_url_falls_back(self, tmp_path):
        # Build on a local file (no S3 round-trip at construction), then point the
        # source_url at a remote URL to exercise the no-local-handle gate.
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = AicsImageIoAdapter.create_from_url(path, "remote")
        adapter._source_url = "s3://bucket/x.ome.tif"
        assert adapter._tifffile_descriptors() is None

    def test_custom_dim_labels_fall_back(self, tmp_path):
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = AicsImageIoAdapter.create_from_url(
            path, "customdims", dim_labels=["C", "Y", "X"]
        )
        assert adapter._tifffile_descriptors() is None

    def test_plain_non_ome_tiff_falls_back(self, tmp_path):
        import tifffile

        plain = tmp_path / "plain.tif"
        tifffile.imwrite(str(plain), np.zeros((16, 16), np.uint8))  # no OME-XML
        adapter = AicsImageIoAdapter.create_from_url(str(plain), "plain")
        assert adapter._tifffile_descriptors() is None
