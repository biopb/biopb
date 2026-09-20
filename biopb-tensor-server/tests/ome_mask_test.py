"""Rasterizing OME-embedded masks into the ``@ome`` label set (biopb/biopb#1059
step 4). Companion to ``ome_roi_import_test.py``, which covers every OTHER
OME-XML shape kind -- a mask is the one kind that import drops, and this is
where it becomes pixels instead.

``masks_by_image`` / ``_bitmap`` / ``RasterizedMaskAdapter.get_data`` are pure
enough to test directly against dicts shaped like an ome-types dump, the same
style ``ome_roi_import_test.py`` uses. One test builds a REAL OME-TIFF (via
tifffile) with a genuinely non-UTF-8 mask bitmap, because that is exactly the
case ``_fast_ome_metadata``'s ``model_dump(mode="json")`` used to crash on.
"""

import base64
import bz2
import zlib

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.adapters.ome_masks import (
    RasterizedMaskAdapter,
    masks_by_image,
    strip_mask_bindata,
)
from biopb_tensor_server.core.errors import WriteNotSupportedError

DIMS = ["T", "Z", "Y", "X"]


def _bin_data(bitmap: np.ndarray, compression: str = "none") -> dict:
    raw = np.packbits(bitmap.astype(np.uint8).flatten(), bitorder="big").tobytes()
    if compression == "zlib":
        raw = zlib.compress(raw)
    elif compression == "bzip2":
        raw = bz2.compress(raw)
    return {"value": base64.b64encode(raw).decode("ascii"), "compression": compression}


def _mask(x, y, w, h, bitmap, *, compression="none", **pins):
    d = {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "bin_data": _bin_data(bitmap, compression),
    }
    d.update(pins)
    return d


def _meta(images_and_masks):
    """``{image_id: [mask_dict, ...]}`` as a metadata dict, roi_refs in order."""
    images, rois = [], []
    for image_id, masks in images_and_masks.items():
        refs = []
        for i, mask in enumerate(masks):
            roi_id = f"ROI:{image_id}:{i}"
            rois.append({"id": roi_id, "union": {"masks": [mask]}})
            refs.append({"id": roi_id})
        images.append({"id": image_id, "roi_refs": refs})
    return {"images": images, "rois": rois}


class TestMasksByImage:
    def test_label_is_the_one_based_roi_refs_position(self):
        bmp = np.ones((2, 2))
        meta = _meta({"Image:0": [_mask(0, 0, 2, 2, bmp), _mask(0, 0, 2, 2, bmp)]})
        shapes = masks_by_image(meta, {"Image:0": ("src/Image:0", DIMS)})["src/Image:0"]
        assert [s.label for s in shapes] == [1, 2]

    def test_unreferenced_or_unmatched_image_contributes_nothing(self):
        meta = _meta({"Image:0": [_mask(0, 0, 2, 2, np.ones((2, 2)))]})
        assert masks_by_image(meta, {}) == {}
        assert masks_by_image(meta, {"Image:99": ("src/Image:99", DIMS)}) == {}

    def test_a_roi_with_no_mask_shapes_is_skipped(self):
        meta = {
            "images": [{"id": "Image:0", "roi_refs": [{"id": "ROI:0"}]}],
            "rois": [{"id": "ROI:0", "union": {"points": [{"x": 1, "y": 1}]}}],
        }
        assert masks_by_image(meta, {"Image:0": ("src/Image:0", DIMS)}) == {}

    def test_an_unreadable_mask_is_dropped_not_fatal(self):
        bad = {
            "x": "not-a-number",
            "y": 0,
            "width": 2,
            "height": 2,
            "bin_data": {"value": "!!"},
        }
        good = _mask(0, 0, 2, 2, np.ones((2, 2)))
        meta = _meta({"Image:0": [bad, good]})
        shapes = masks_by_image(meta, {"Image:0": ("src/Image:0", DIMS)})["src/Image:0"]
        assert len(shapes) == 1
        assert shapes[0].label == 2  # position preserved even though [0] was dropped


class TestRasterizedMaskAdapter:
    def _adapter(self, masks, dim_labels=("Y", "X"), shape=(4, 4)):
        return RasterizedMaskAdapter(
            "src",
            "Image:0/labels/@ome",
            dim_labels=list(dim_labels),
            shape=list(shape),
            masks=masks,
            parent_array_id="src/Image:0",
            content_version=b"v1",
        )

    def _shapes(self, meta, image="Image:0", dims=("Y", "X")):
        return masks_by_image(meta, {image: (f"src/{image}", list(dims))})[
            f"src/{image}"
        ]

    def test_overlap_the_later_roi_wins(self):
        bmp1 = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        bmp2 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        meta = _meta({"Image:0": [_mask(0, 0, 4, 4, bmp1), _mask(0, 0, 4, 4, bmp2)]})
        adapter = self._adapter(self._shapes(meta))
        out = adapter.get_data(ChunkBounds(start=[0, 0], stop=[4, 4]))
        assert out[0, 0] == 1
        assert out[1, 1] == 2  # the overlap cell: mask 2 painted after mask 1
        assert out[0, 2] == 0

    def test_zlib_and_bzip2_compression_round_trip(self):
        bmp = np.eye(4, dtype=np.uint8)
        for compression in ("zlib", "bzip2"):
            meta = _meta({"Image:0": [_mask(0, 0, 4, 4, bmp, compression=compression)]})
            adapter = self._adapter(self._shapes(meta))
            out = adapter.get_data(ChunkBounds(start=[0, 0], stop=[4, 4]))
            assert (out > 0).astype(np.uint8).tolist() == bmp.tolist()

    @pytest.mark.parametrize("compression", ("zlib", "bzip2"))
    def test_corrupt_compressed_mask_is_skipped_not_fatal(self, compression):
        bad = _mask(0, 0, 2, 2, np.ones((2, 2)), compression=compression)
        bad["bin_data"]["value"] = base64.b64encode(b"not a valid stream").decode(
            "ascii"
        )
        good = _mask(2, 2, 2, 2, np.ones((2, 2)))
        adapter = self._adapter(self._shapes(_meta({"Image:0": [bad, good]})))

        out = adapter.get_data(ChunkBounds(start=[0, 0], stop=[4, 4]))

        assert (out[:2, :2] == 0).all()
        assert (out[2:, 2:] == 2).all()
        # The failed decode is memoized; another intersecting chunk stays safe.
        assert (adapter.get_data(ChunkBounds(start=[0, 0], stop=[2, 2])) == 0).all()

    def test_a_partial_chunk_reads_only_its_own_region(self):
        bmp = np.ones((4, 4))
        meta = _meta({"Image:0": [_mask(1, 1, 2, 2, bmp)]})  # bbox = [1,3) x [1,3)
        adapter = self._adapter(self._shapes(meta), shape=(6, 6))
        out = adapter.get_data(ChunkBounds(start=[0, 0], stop=[3, 3]))
        assert out[1, 1] == 1 and out[1, 2] == 1 and out[2, 1] == 1 and out[2, 2] == 1
        assert out[0, 0] == 0
        out2 = adapter.get_data(ChunkBounds(start=[3, 3], stop=[6, 6]))
        assert (out2 == 0).all()  # bbox ends exactly at 3, this chunk starts there

    def test_y_x_are_found_by_label_not_position(self):
        """An interleaved RGB(A) source keeps its trailing samples axis (``S``)
        here -- ``label_extent`` drops only the channel axis -- so Y/X are NOT
        reliably the last two axes, and locating them positionally paints the
        wrong plane entirely."""
        bmp = np.zeros((4, 4))
        bmp[1:3, 1:3] = 1
        meta = _meta({"Image:0": [_mask(0, 0, 4, 4, bmp)]})
        adapter = self._adapter(
            self._shapes(meta, dims=("Z", "Y", "X", "S")),
            dim_labels=("Z", "Y", "X", "S"),
            shape=(1, 4, 4, 3),
        )
        out = adapter.get_data(ChunkBounds(start=[0, 0, 0, 0], stop=[1, 4, 4, 3]))
        assert out.shape == (1, 4, 4, 3)
        assert out[0, 2, 2, 0] == 1
        assert out[0, 0, 0, 0] == 0
        assert (out[0, 2, 2, :] == 1).all()  # broadcasts across the samples axis

    def test_z_pin_applies_only_to_its_own_plane(self):
        bmp = np.ones((3, 3))
        meta = _meta({"Image:0": [_mask(0, 0, 3, 3, bmp, the_z=1)]})
        adapter = self._adapter(
            self._shapes(meta, dims=("Z", "Y", "X")),
            dim_labels=("Z", "Y", "X"),
            shape=(3, 3, 3),
        )
        out = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 3, 3]))
        assert out[0].sum() == 0
        assert (out[1] > 0).all()
        assert out[2].sum() == 0
        # a chunk that does not include the pinned plane holds nothing at all
        out2 = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[1, 3, 3]))
        assert (out2 == 0).all()

    def test_unpinned_axis_broadcasts_the_mask_to_every_plane(self):
        bmp = np.ones((2, 2))
        meta = _meta({"Image:0": [_mask(0, 0, 2, 2, bmp)]})  # no the_t pin
        adapter = self._adapter(
            self._shapes(meta, dims=("T", "Y", "X")),
            dim_labels=("T", "Y", "X"),
            shape=(3, 2, 2),
        )
        out = adapter.get_data(ChunkBounds(start=[0, 0, 0], stop=[3, 2, 2]))
        assert (out == 1).all()

    def test_read_only(self):
        adapter = self._adapter([])
        with pytest.raises(WriteNotSupportedError):
            adapter.put_chunk(
                ChunkBounds(start=[0, 0], stop=[1, 1]), None, (1, 1), "<u4"
            )

    def test_descriptor_and_metadata(self):
        adapter = self._adapter([])
        desc = adapter.get_tensor_descriptor()
        assert desc.array_id == "src/Image:0/labels/@ome"
        assert desc.dtype == "<u4"
        assert list(desc.shape) == [4, 4]
        meta = adapter.get_tensor_metadata()
        assert meta["image-label"]["source"]["image"] == "src/Image:0"

    def test_pyramid_is_always_nearest(self):
        from biopb_tensor_server.core.config import PyramidConfig

        adapter = self._adapter([], shape=(64, 64))
        desc = adapter.get_tensor_descriptor()
        levels = adapter._advertised_pyramid(
            desc, PyramidConfig(reduction_method="area")
        )
        assert levels  # a 64x64 extent has at least one computed level
        assert all(level.reduction_method == "nearest" for level in levels)


class TestStripMaskBindata:
    def test_drops_only_bin_data_value(self):
        meta = _meta({"Image:0": [_mask(0, 0, 2, 2, np.ones((2, 2)))]})
        stripped = strip_mask_bindata(meta)
        mask = stripped["rois"][0]["union"]["masks"][0]
        assert "value" not in mask["bin_data"]
        assert mask["bin_data"]["compression"] == "none"  # everything else survives
        assert mask["x"] == 0 and mask["width"] == 2

    def test_a_non_mask_roi_is_untouched(self):
        meta = {
            "images": [{"id": "Image:0", "roi_refs": [{"id": "ROI:0"}]}],
            "rois": [{"id": "ROI:0", "union": {"points": [{"x": 1, "y": 1}]}}],
        }
        assert strip_mask_bindata(meta) is meta  # the same object, per the docstring

    def test_no_rois_at_all_is_a_no_op(self):
        meta = {"creator": "tifffile"}
        assert strip_mask_bindata(meta) is meta

    def test_original_metadata_is_not_mutated(self):
        meta = _meta({"Image:0": [_mask(0, 0, 2, 2, np.ones((2, 2)))]})
        strip_mask_bindata(meta)
        assert "value" in meta["rois"][0]["union"]["masks"][0]["bin_data"]


class TestFastMetadataRealBitmap:
    """The bug this step's fix closes: a real (non-UTF-8) mask bitmap used to
    crash the WHOLE fast metadata parse, silently costing a file every piece
    of its OME metadata -- not just the mask."""

    def _write(self, tmp_path, raw: bytes):
        import tifffile

        h, w = 4, 4
        b64_len = len(raw)
        xml = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
            'UUID="urn:uuid:11111111-1111-1111-1111-111111111111">'
            '<Image ID="Image:0" Name="img">'
            '<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8" '
            f'SizeX="{w}" SizeY="{h}" SizeZ="1" SizeC="1" SizeT="1">'
            '<Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
            '<TiffData FirstC="0" FirstZ="0" FirstT="0" IFD="0" PlaneCount="1"/>'
            "</Pixels>"
            '<ROIRef ID="ROI:0"/>'
            "</Image>"
            '<ROI ID="ROI:0"><Union>'
            f'<Mask ID="Shape:0:0" X="0" Y="0" Width="{w}" Height="{h}">'
            f'<BinData Length="{b64_len}" Compression="none" BigEndian="false">'
            f"{base64.b64encode(raw).decode('ascii')}</BinData>"
            "</Mask></Union></ROI></OME>"
        )
        p = tmp_path / "mask.ome.tif"
        tifffile.imwrite(
            str(p), np.zeros((h, w), np.uint8), description=xml, metadata=None
        )
        return str(p)

    def test_a_non_utf8_bitmap_does_not_blank_the_whole_metadata(self, tmp_path):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        raw = bytes([0xFF, 0x00, 0xFE, 0x80, 0x01, 0x00])  # not valid UTF-8
        path = self._write(tmp_path, raw)
        adapter = OmeTiffAdapter(path, "src1")
        metadata = adapter.get_metadata()
        assert metadata  # used to come back {} entirely
        assert metadata["images"][0]["id"] == "Image:0"
        mask = metadata["rois"][0]["union"]["masks"][0]
        assert base64.b64decode(mask["bin_data"]["value"]) == raw

    def test_get_embedded_labels_end_to_end(self, tmp_path):
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        raw_bitmap = np.zeros((4, 4), dtype=np.uint8)
        raw_bitmap[1:3, 1:3] = 1
        raw = np.packbits(raw_bitmap.flatten(), bitorder="big").tobytes()
        path = self._write(tmp_path, raw)
        adapter = OmeTiffAdapter(path, "src1")

        sets = adapter.get_embedded_labels()
        assert list(sets.keys()) == ["Image:0/labels/@ome"]
        label_set = sets["Image:0/labels/@ome"]
        assert label_set.array_id == "src1/Image:0/labels/@ome"
        desc = label_set.get_tensor_descriptor()
        out = label_set.get_data(
            ChunkBounds(start=[0] * len(desc.shape), stop=list(desc.shape))
        )
        assert out[tuple([0] * (out.ndim - 2) + [2, 2])] == 1
        assert out[tuple([0] * (out.ndim - 2) + [0, 0])] == 0

        # And through the base SourceAdapter machinery: extent must match.
        assert "Image:0/labels/@ome" in adapter.label_sets

        adapter.release_registration_cache()

        # The label adapter keeps the decoded bitmap, while the source retains
        # neither the base64 payload nor its parsed duplicate.
        assert base64.b64encode(raw).decode("ascii") not in adapter._reduced_ome_xml
        assert adapter._parsed_metadata is None
        assert adapter._parsed_metadata_probed is False
        for scene in adapter._tensor_adapters.values():
            assert base64.b64encode(raw).decode("ascii") not in scene._reduced_ome_xml
        cached_label_set = adapter.label_sets["Image:0/labels/@ome"]
        assert (
            cached_label_set.get_data(
                ChunkBounds(start=[0] * len(desc.shape), stop=list(desc.shape))
            )[tuple([0] * (out.ndim - 2) + [2, 2])]
            == 1
        )
        metadata_after_release = adapter.get_metadata()
        mask_after_release = metadata_after_release["rois"][0]["union"]["masks"][0]
        assert mask_after_release["bin_data"]["value"] == ""

    def test_release_survives_a_reduced_xml_the_stripper_cannot_parse(
        self, tmp_path, monkeypatch
    ):
        """release_registration_cache() is documented to never raise. A reduced
        XML the mask stripper's ET.fromstring rejects must not abort the raw-XML
        drop or the cascade to scene adapters below it -- it is left un-redacted
        instead (biopb/biopb#1081)."""
        import xml.etree.ElementTree as ET

        import biopb_tensor_server.adapters.ome_tiff as ome_tiff_module
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        raw_bitmap = np.zeros((4, 4), dtype=np.uint8)
        raw_bitmap[1:3, 1:3] = 1
        raw = np.packbits(raw_bitmap.flatten(), bitorder="big").tobytes()
        path = self._write(tmp_path, raw)
        adapter = OmeTiffAdapter(path, "src1")
        adapter.get_embedded_labels()  # sets _mask_payloads_transferred

        def _broken_strip(ome_xml):
            raise ET.ParseError("boom")

        monkeypatch.setattr(
            ome_tiff_module, "_strip_mask_bindata_payloads", _broken_strip
        )

        adapter.release_registration_cache()  # must not raise

        assert adapter._raw_ome_xml is None
        assert adapter._raw_ome_xml_released is True
        # Left un-redacted: the strip that would have dropped it never ran.
        assert base64.b64encode(raw).decode("ascii") in adapter._reduced_ome_xml

    def test_get_metadata_parses_the_ome_xml_only_once(self, tmp_path, monkeypatch):
        """get_embedded_labels() calls get_metadata() internally, and so does
        the registration path (metadata_db.py) -- the parsed dict is cached so
        that doesn't cost a second ome-types parse."""
        import biopb_tensor_server.adapters.ome_tiff as ome_tiff_module
        from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter

        path = self._write(tmp_path, b"\x00" * 2)
        adapter = OmeTiffAdapter(path, "src1")

        calls = []
        original = ome_tiff_module._fast_ome_metadata

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(ome_tiff_module, "_fast_ome_metadata", counting)

        adapter.get_metadata()
        adapter.get_embedded_labels()  # calls self.get_metadata() again internally
        adapter.get_metadata()

        assert len(calls) == 1
