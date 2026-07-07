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
from biopb_tensor_server.adapters.ome_tiff import (
    _STRIP_PER_PLANE,
    OmeTiffAdapter,
    _fast_ome_metadata,
    _ome_axes_shape,
    _ome_scene_ids,
    _tczyx_shape,
)
from biopb_tensor_server.fixtures import (
    create_multi_series_ome_tiff,
    create_multifile_embedded_ome_tiff,
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


class TestOmeAxesShape:
    @pytest.mark.parametrize(
        "shape,axes,dims,expected",
        [
            ((3, 16, 16), "CYX", "TCZYX", [1, 3, 1, 16, 16]),  # canonical
            ((4, 2, 3, 16, 16), "TCZYX", "TCZYX", [4, 2, 3, 16, 16]),
            ((16, 16, 3), "YXS", "TCZYXS", [1, 1, 1, 16, 16, 3]),  # RGB samples
            ((12, 10, 4), "YXS", "TCZYXS", [1, 1, 1, 12, 10, 4]),  # RGBA samples
        ],
    )
    def test_maps_canonical_and_samples(self, shape, axes, dims, expected):
        got = _ome_axes_shape(shape, axes)
        assert got is not None
        got_dims, got_shape = got
        assert "".join(got_dims) == dims
        assert got_shape == expected

    @pytest.mark.parametrize(
        "shape,axes",
        [
            ((4, 16, 16), "QYX"),  # unknown (non-OME) axis
            ((2, 16, 16), "CZYX"),  # axes/shape length mismatch
            ((16, 16), ""),  # no axes
        ],
    )
    def test_declines_non_ome_axes(self, shape, axes):
        assert _ome_axes_shape(shape, axes) is None


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
        return [(d.array_id.split("/", 1)[1], list(d.shape), d.dtype) for d in descs]

    def test_single_series_parity(self, tmp_path):
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(3, 64, 64))
        adapter = OmeTiffAdapter(path, "single")
        assert self._fast(adapter) == self._aics_truth(path)

    def test_multi_series_parity(self, tmp_path):
        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        adapter = OmeTiffAdapter(path, "multi")
        assert self._fast(adapter) == self._aics_truth(path)

    def test_multifile_embedded_ome_parity(self, tmp_path):
        # Multi-file OME-TIFF: the master's local IFD count (1) is LESS than the
        # OME-declared SizeC (3) -- the shape must come from the OME Size* across
        # sibling files, not the local file. tifffile assembles the siblings into
        # one OME series, so the fast path's shape must still match aicsimageio.
        path, names, full_shape = create_multifile_embedded_ome_tiff(
            str(tmp_path), n_files=3
        )
        adapter = OmeTiffAdapter(path, "mfembed")

        fast = self._fast(adapter)
        assert fast == self._aics_truth(path)
        # Guard the specific invariant the reviewer flagged: full SizeC, not 1.
        assert fast[0][1] == list(full_shape)  # (1, 3, 1, h, w)
        assert len(names) == 3

    def test_list_descriptors_does_not_parse_aicsimageio(self, tmp_path):
        # The registration path must build the catalog row without ever touching
        # AICSImage (the ome-types parse is the startup cost being removed).
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = OmeTiffAdapter(path, "noparse")
        adapter._aics_image = _Tripwire()

        descriptors = adapter.list_tensor_descriptors()

        assert len(descriptors) == 1
        assert descriptors[0].array_id == "noparse/Image:0"
        assert list(descriptors[0].shape) == [1, 2, 1, 32, 32]
        assert list(descriptors[0].dim_labels) == list("TCZYX")
        # Cached, and still served from cache without parsing.
        assert adapter._cached_descriptors is descriptors
        assert adapter.list_tensor_descriptors() is descriptors


class TestRgbSamplesDescriptor:
    """An RGB (YXS / photometric-rgb) OME-TIFF must yield a descriptor whose
    dim_labels and shape agree in length.

    aicsimageio folds interleaved samples into a trailing ``S`` axis, so it
    reports dims.order "TCZYXS" (6) with dask shape ``(1,1,1,H,W,3)``. The
    OME-metadata fast path in ``list_tensor_descriptors`` builds a canonical 5-D
    TCZYX shape from ``Pixels`` -- if it paired that 5-D shape with the 6 labels
    the descriptor would be malformed and ``get_flight_info`` would reject every
    slice as a dimensionality mismatch (an RGB sample dataset failing to open in
    the webapp). The fast path must defer such sources to the scene-switching
    path, exactly as ``_tczyx_shape`` rejects the ``S`` axis.
    """

    def _write_rgb_ome_tiff(self, tmp_path, h=24, w=32):
        import tifffile

        path = str(tmp_path / "rgb.ome.tif")
        data = np.zeros((h, w, 3), dtype=np.uint8)
        data[..., 0] = 10
        data[..., 1] = 20
        data[..., 2] = 30
        tifffile.imwrite(
            path, data, ome=True, photometric="rgb", metadata={"axes": "YXS"}
        )
        return path

    def test_tifffile_maps_rgb_natively(self, tmp_path):
        # RGB (YXS) is tifffile-native (biopb/biopb#213 follow-up): the descriptor
        # path maps the trailing samples axis to a 6-D TCZYXS descriptor.
        path = self._write_rgb_ome_tiff(tmp_path)
        adapter = OmeTiffAdapter(path, "rgb")
        descs = adapter._tifffile_descriptors()
        assert descs is not None
        assert list(descs[0].dim_labels) == list("TCZYXS")
        assert list(descs[0].shape) == [1, 1, 1, 24, 32, 3]

    def test_descriptor_labels_and_shape_agree(self, tmp_path):
        path = self._write_rgb_ome_tiff(tmp_path)
        adapter = OmeTiffAdapter(path, "rgb")

        descriptors = adapter.list_tensor_descriptors()

        assert len(descriptors) == 1
        d = descriptors[0]
        assert len(d.dim_labels) == len(d.shape), (
            f"label/shape length mismatch: {list(d.dim_labels)} vs {list(d.shape)}"
        )
        # Interleaved samples become a trailing S axis (TCZYXS), the layout the
        # webapp renderer expects.
        assert list(d.dim_labels) == list("TCZYXS")
        assert list(d.shape) == [1, 1, 1, 24, 32, 3]

    def test_slice_round_trips_through_get_flight_info(self, tmp_path):
        # End-to-end guard on the exact path that was failing: get_flight_info
        # validates the client's SliceHint against the descriptor dim count.
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server.server import TensorFlightServer

        path = self._write_rgb_ome_tiff(tmp_path)
        adapter = OmeTiffAdapter(path, "rgb")
        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("rgb", adapter)
        server.mark_ready()
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
            darr = client.get_tensor("rgb/Image:0")
            assert darr.shape == (1, 1, 1, 24, 32, 3)
            sub = np.asarray(darr[0, 0, 0, :, :, :])
            assert sub.shape == (24, 32, 3)
        finally:
            server.shutdown()


class TestSceneResolutionAndReads:
    def test_scene_index_resolves_from_cache(self, tmp_path):
        path, _, _ = create_multi_series_ome_tiff(str(tmp_path), n_series=3)
        adapter = OmeTiffAdapter(path, "idx")
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
        adapter = OmeTiffAdapter(path, "reads")
        descriptors = adapter.list_tensor_descriptors()

        from biopb.tensor.ticket_pb2 import ChunkBounds

        for k, desc in enumerate(descriptors):
            field = desc.array_id.split("/", 1)[1]
            scene = adapter.get_tensor_adapter(field)
            # shape is TCZYX = (1, 2, 1, 32, 32); read the (0,0,0,0,0) corner.
            bounds = ChunkBounds(start=[0, 0, 0, 0, 0], stop=[1, 1, 1, 1, 1])
            val = np.asarray(scene.get_data(bounds)).ravel()[0]
            assert val == k * 100 + 1, f"series {k} returned {val}"


_OME_WITH_PLANES = (
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
    '<Image ID="Image:0" Name="m"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
    'Type="uint16" SizeX="16" SizeY="16" SizeZ="3" SizeC="2" SizeT="1" '
    'PhysicalSizeX="0.1" PhysicalSizeXUnit="µm">'
    '<Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="1"/>'
    '<Channel ID="Channel:0:1" Name="GFP" SamplesPerPixel="1"/>'
    '<Plane TheZ="0" TheC="0" TheT="0" DeltaT="0.0"/>'
    '<TiffData FirstZ="0" FirstC="0" FirstT="0" IFD="0" PlaneCount="1"/>'
    '<Plane TheZ="1" TheC="0" TheT="0" DeltaT="0.1"/>'
    '<TiffData FirstZ="1" FirstC="0" FirstT="0" IFD="1">'
    '<UUID FileName="f.tif">urn:uuid:00000000-0000-0000-0000-000000000001</UUID>'
    "</TiffData>"
    "</Pixels></Image></OME>"
)


class TestStripPerPlane:
    def test_strips_self_closing_and_uuid_tiffdata(self):
        reduced = _STRIP_PER_PLANE.sub("", _OME_WITH_PLANES)
        assert "<Plane" not in reduced
        assert "<TiffData" not in reduced
        # Structural elements survive untouched.
        assert "<Channel" in reduced and "PhysicalSizeX" in reduced

    def test_keeps_xml_when_no_per_plane_elements(self):
        xml = '<OME><Image ID="Image:0"/></OME>'
        assert _STRIP_PER_PLANE.sub("", xml) == xml

    def test_strips_tiffdata_wrapping_self_closing_uuid(self):
        # Some MMStacks emit `<UUID FileName="..."/>` (no urn body). The strip must
        # consume the whole <TiffData>...</TiffData>, not stop at the child's `/>`
        # and orphan the close tag -- which produced malformed XML (biopb/biopb#193).
        xml = (
            '<Pixels><TiffData FirstZ="0" IFD="0">'
            '<UUID FileName="f.ome.tif"/>'
            '</TiffData><Channel ID="c0"/></Pixels>'
        )
        reduced = _STRIP_PER_PLANE.sub("", xml)
        assert reduced == '<Pixels><Channel ID="c0"/></Pixels>'
        # No orphaned open/close fragments left behind.
        assert "TiffData" not in reduced

    def test_strips_plane_wrapping_child_element(self):
        xml = (
            '<Pixels><Plane TheZ="0" TheC="0" TheT="0">'
            '<AnnotationRef ID="Annotation:0"/>'
            '</Plane><Channel ID="c0"/></Pixels>'
        )
        assert _STRIP_PER_PLANE.sub("", xml) == '<Pixels><Channel ID="c0"/></Pixels>'

    def test_strips_namespaced_per_plane_elements(self):
        xml = (
            '<ns:Pixels><ns:TiffData IFD="0">'
            '<ns:UUID FileName="f"/></ns:TiffData>'
            '<ns:Channel ID="c0"/></ns:Pixels>'
        )
        reduced = _STRIP_PER_PLANE.sub("", xml)
        assert reduced == '<ns:Pixels><ns:Channel ID="c0"/></ns:Pixels>'

    def test_does_not_strip_unrelated_lookalike_element(self):
        # \b after the captured name must not let a longer element name match.
        xml = '<Pixels><TiffDataset Foo="1"/><Channel ID="c0"/></Pixels>'
        assert _STRIP_PER_PLANE.sub("", xml) == xml

    def test_many_self_closing_planes_strip_in_linear_time(self):
        # Regression guard (biopb/biopb#193): a real MMStack carries one
        # self-closing <Plane/> per frame (10k+). An open/close form that scanned
        # `.*?</Plane>` to EOF for each self-closing plane was O(n^2) (~87 s on a
        # 10k-plane file). The conditional self-closing branch keeps it linear, so
        # this large input must strip near-instantly, not in seconds.
        import time

        planes = "".join(
            f'<Plane TheZ="0" TheC="0" TheT="{i}" DeltaT="{i}.0"/>' for i in range(8000)
        )
        xml = f'<Pixels>{planes}<Channel ID="c0"/></Pixels>'
        t = time.monotonic()
        reduced = _STRIP_PER_PLANE.sub("", xml)
        elapsed = time.monotonic() - t
        assert reduced == '<Pixels><Channel ID="c0"/></Pixels>'
        assert elapsed < 2.0, f"strip took {elapsed:.1f}s (catastrophic backtracking?)"

    def test_fast_metadata_parses_self_closing_uuid_tiffdata(self):
        # End-to-end: the reduced XML must be well-formed so ome-types parses it
        # instead of falling back to the ~105 s authoritative parse (#193).
        ome = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image ID="Image:0" Name="m"><Pixels ID="Pixels:0" '
            'DimensionOrder="XYZCT" Type="uint16" SizeX="16" SizeY="16" '
            'SizeZ="1" SizeC="1" SizeT="1">'
            '<Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="1"/>'
            '<TiffData FirstZ="0" FirstC="0" FirstT="0" IFD="0" PlaneCount="1">'
            '<UUID FileName="f.ome.tif"/>'
            "</TiffData>"
            "</Pixels></Image></OME>"
        )
        md = _fast_ome_metadata(ome)
        assert md is not None
        px = md["images"][0]["pixels"]
        assert px["tiff_data_blocks"] == []
        assert [c["name"] for c in px["channels"]] == ["DAPI"]


class TestFastMetadata:
    def test_drops_planes_keeps_structure(self):
        md = _fast_ome_metadata(_OME_WITH_PLANES)
        assert md is not None
        px = md["images"][0]["pixels"]
        # Per-plane arrays dropped (the accuracy trade); structure preserved.
        assert px["planes"] == []
        assert px["tiff_data_blocks"] == []
        assert px["size_z"] == 3
        assert px["physical_size_x"] == 0.1
        assert [c["name"] for c in px["channels"]] == ["DAPI", "GFP"]

    def test_parity_with_full_parse_except_planes(self):
        # The fast dict must equal the full ome-types model_dump with only the
        # per-plane arrays zeroed out.
        from ome_types import from_xml

        full = from_xml(_OME_WITH_PLANES).model_dump(mode="json")
        fast = _fast_ome_metadata(_OME_WITH_PLANES)

        def _zero_planes(md):
            for im in md.get("images", []):
                im["pixels"]["planes"] = []
                im["pixels"]["tiff_data_blocks"] = []
            return md

        assert _zero_planes(full) == fast

    def test_malformed_returns_none(self):
        assert _fast_ome_metadata("<OME><not valid") is None

    def test_get_metadata_does_not_parse_aicsimageio(self, tmp_path):
        # get_metadata runs at registration (metadata-DB sync); it must build the
        # dict from the stripped OME-XML without touching AICSImage.
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(3, 64, 64))
        adapter = OmeTiffAdapter(path, "md")

        hits = []

        class Recorder:
            @property
            def ome_metadata(self):
                hits.append("ome_metadata")
                return None

            def __getattr__(self, n):
                hits.append(n)
                return None

        adapter._aics_image = Recorder()
        md = adapter.get_metadata()

        assert hits == [], f"AICSImage accessed in get_metadata: {hits}"
        assert "images" in md and md["images"][0]["pixels"]["planes"] == []

    def test_raw_ome_xml_cached_across_descriptor_and_metadata(self, tmp_path):
        # The descriptor fast path populates the OME-XML cache so get_metadata
        # does not reopen the file.
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = OmeTiffAdapter(path, "cache")
        assert adapter._raw_ome_xml_probed is False

        adapter.list_tensor_descriptors()  # descriptor fast path
        assert adapter._raw_ome_xml_probed is True
        assert adapter._raw_ome_xml  # the embedded OME-XML string
        assert adapter._local_ome_xml() == adapter._raw_ome_xml


class TestFallback:
    def test_remote_url_falls_back(self, tmp_path):
        # Build on a local file (no S3 round-trip at construction), then point the
        # source_url at a remote URL to exercise the no-local-handle gate.
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = OmeTiffAdapter(path, "remote")
        adapter._source_url = "s3://bucket/x.ome.tif"
        assert adapter._tifffile_descriptors() is None

    def test_custom_dim_labels_fall_back(self, tmp_path):
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
        adapter = OmeTiffAdapter(path, "customdims", dim_labels=["C", "Y", "X"])
        assert adapter._tifffile_descriptors() is None

    def test_plain_non_ome_tiff_falls_back(self, tmp_path):
        import tifffile

        plain = tmp_path / "plain.tif"
        tifffile.imwrite(str(plain), np.zeros((16, 16), np.uint8))  # no OME-XML
        adapter = OmeTiffAdapter(str(plain), "plain")
        assert adapter._tifffile_descriptors() is None


class TestOmeParseDedup:
    """The OME parse is shared across aicsimageio's redundant probes (#192)."""

    def test_get_ome_is_memoized(self):
        from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

        get_ome = OmeTiffReader.__dict__["_get_ome"]
        assert getattr(get_ome.__func__, "_biopb_dedup", False), (
            "OmeTiffReader._get_ome should be wrapped by _install_ome_parse_dedup"
        )

    def test_construction_parses_ome_xml_once_not_three_times(self, tmp_path):
        # Stock aicsimageio parses the OME-XML 3x per OME-TIFF construction
        # (determine_reader probe + reader _is_supported probe + stored _ome).
        # The dedup memo collapses them to a single from_xml.
        import aicsimageio.readers.ome_tiff_reader as omr
        from aicsimageio import AICSImage
        from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

        # Drop any shared-cache entry so this fixture's XML is a real miss.
        OmeTiffReader.__dict__["_get_ome"].__func__.cache_clear()

        calls = {"n": 0}
        real_from_xml = omr.from_xml

        def counting_from_xml(*args, **kwargs):
            calls["n"] += 1
            return real_from_xml(*args, **kwargs)

        omr.from_xml = counting_from_xml
        try:
            path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 32, 32))
            img = AICSImage(path)
            assert type(img.reader).__name__ == "OmeTiffReader"
            assert len(img.scenes) == 1  # force the model to be fully built
        finally:
            omr.from_xml = real_from_xml

        assert calls["n"] == 1, f"expected a single OME parse, got {calls['n']}"


class TestReadPathTifffileAuthoritative:
    """The OME-TIFF read path is pure tifffile (biopb/biopb#213 + detachment).

    ``OmeTiffAdapter`` has no aicsimageio dependency at all -- descriptors, reads,
    metadata, and physical scale come from tifffile / the embedded OME-XML.
    Correctness is asserted as internal consistency of the tifffile store (data
    matches its descriptor; sub-regions are correct; ragged gaps are
    deterministic), and that the adapter never grows an ``_aics_image``.
    """

    def _scene(self, path, source_id, field="Image:0"):
        """Registered source adapter + its scene adapter for ``field``."""
        adapter = OmeTiffAdapter(path, source_id)
        descriptors = adapter.list_tensor_descriptors()
        assert descriptors, "expected tifffile descriptors for a local OME-TIFF"
        scene = adapter.get_tensor_adapter(field)
        assert scene._tifffile_descriptor is not None
        return adapter, scene

    def test_read_path_has_no_aicsimageio(self, tmp_path):
        # Core lock post-detachment: exercise the whole read path -- plan_flight_info
        # (the #350 GetFlightInfo seam: descriptor + pyramid + physical scale) and
        # get_data (DoGet) -- and assert the adapter never holds an aicsimageio image.
        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb.tensor.ticket_pb2 import ChunkBounds
        from biopb_tensor_server.config import PyramidConfig

        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(3, 64, 64))
        _, scene = self._scene(path, "pure")
        assert not hasattr(scene, "_aics_image")

        plan = scene.plan_flight_info(TensorReadOption(), PyramidConfig())
        assert list(plan.descriptor.shape) == [1, 3, 1, 64, 64]
        assert len(plan.descriptor.pyramid) >= 1  # computed pyramid

        bounds = ChunkBounds(start=[0, 1, 0, 0, 0], stop=[1, 2, 1, 64, 64])
        arr = np.asarray(scene.get_data(bounds))
        assert arr.shape == (1, 1, 1, 64, 64)
        assert (arr == 2).all()  # channel 1 -> fixture value 2

        assert list(scene.get_tensor_descriptor().shape) == [1, 3, 1, 64, 64]
        assert scene._physical_scale() is None  # fixture carries no PhysicalSize

    def test_descriptor_data_self_consistency_multi_series(self, tmp_path):
        # For each scene the data matches its advertised descriptor (shape/dtype)
        # and the known per-series fill lands at the right position -- locking
        # correct scene->series mapping and TCZYX axis order.
        from biopb.tensor.ticket_pb2 import ChunkBounds

        path, _, _ = create_multi_series_ome_tiff(
            str(tmp_path), n_series=3, series_shape=(2, 32, 32)
        )
        adapter = OmeTiffAdapter(path, "consist")
        descriptors = adapter.list_tensor_descriptors()

        for k, desc in enumerate(descriptors):
            field = desc.array_id.split("/", 1)[1]
            scene = adapter.get_tensor_adapter(field)
            arr = np.asarray(
                scene.get_data(
                    ChunkBounds(start=[0, 0, 0, 0, 0], stop=list(desc.shape))
                )
            )
            assert arr.shape == tuple(desc.shape)
            assert arr.dtype.str == desc.dtype
            for p in range(arr.shape[1]):  # fixture: series k, plane p -> k*100+p+1
                assert arr[0, p, 0, 0, 0] == k * 100 + p + 1

    def test_ragged_gap_is_internally_consistent(self, tmp_path):
        # A sparse OME-TIFF (SizeC=3, channel 1 absent) reads to the declared shape;
        # present channels carry their true values and the ragged gap is filled
        # DETERMINISTICALLY (repeat reads identical) -- internal consistency of the
        # tifffile store, not a specific gap value.
        from biopb.tensor.ticket_pb2 import ChunkBounds

        path, present = self._write_sparse_multifile_ome_tiff(str(tmp_path))
        adapter = OmeTiffAdapter(path, "ragged")
        descriptors = adapter.list_tensor_descriptors()
        assert descriptors

        desc = descriptors[0]
        full = ChunkBounds(start=[0, 0, 0, 0, 0], stop=list(desc.shape))
        arr = np.asarray(adapter.get_tensor_adapter("Image:0").get_data(full))
        assert arr.shape == tuple(desc.shape)  # (1, 3, 1, 16, 16) -- gap included
        for c, val in present.items():
            assert arr[0, c, 0, 0, 0] == val
        arr2 = np.asarray(adapter.get_tensor_adapter("Image:0").get_data(full))
        assert np.array_equal(arr, arr2)  # ragged fill is deterministic

    def test_rgb_reads_via_tifffile_store(self, tmp_path):
        # RGB (YXS) is tifffile-native end-to-end: the scene serves a TCZYXS block.
        import tifffile
        from biopb.tensor.ticket_pb2 import ChunkBounds

        p = str(tmp_path / "rgb.ome.tif")
        data = np.zeros((24, 32, 3), np.uint8)
        data[..., 0], data[..., 1], data[..., 2] = 10, 20, 30
        tifffile.imwrite(p, data, ome=True, photometric="rgb", metadata={"axes": "YXS"})
        _, scene = self._scene(p, "rgb")
        desc = scene.get_tensor_descriptor()
        assert list(desc.dim_labels) == list("TCZYXS")
        arr = np.asarray(
            scene.get_data(ChunkBounds(start=[0] * 6, stop=list(desc.shape)))
        )
        assert arr.shape == (1, 1, 1, 24, 32, 3)
        assert arr[0, 0, 0, 0, 0, 0] == 10 and arr[0, 0, 0, 0, 0, 2] == 30

    def test_physical_scale_from_stripped_xml(self, tmp_path):
        path = self._write_ome_tiff_with_physical_sizes(
            str(tmp_path / "phys.ome.tif"), psx=0.325, psy=0.325, psz=2.0
        )
        _, scene = self._scene(path, "phys")
        scale, unit = scene._physical_scale()
        assert scale == [0.0, 0.0, 2.0, 0.325, 0.325]  # TCZYX
        assert unit == ["", "", "µm", "µm", "µm"]

    def test_physical_scale_missing_unit_defaults_to_micron(self, tmp_path):
        # tifffile always stamps a unit, so inject an OME-XML that omits it to lock
        # the "µm" default (the OME spec default).
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(1, 16, 16))
        _, scene = self._scene(path, "nounit")
        scene._raw_ome_xml = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
            'Type="uint16" SizeX="16" SizeY="16" SizeZ="1" SizeC="1" SizeT="1" '
            'PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
            '<Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
            "</Pixels></Image></OME>"
        )
        scene._raw_ome_xml_probed = True
        scale, unit = scene._physical_scale()
        assert scale == [0.0, 0.0, 0.0, 0.5, 0.5]  # Z absent -> 0.0
        assert unit == ["", "", "", "µm", "µm"]

    def test_no_physical_scale_returns_none(self, tmp_path):
        # An OME-TIFF with no PhysicalSize* has no known scale -> None.
        path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 16, 16))
        _, scene = self._scene(path, "nophys")
        assert scene._physical_scale() is None

    # ---- local fixture writers ----

    @staticmethod
    def _write_ome_tiff_with_physical_sizes(path, psx, psy, psz):
        import tifffile

        tifffile.imwrite(
            path,
            np.zeros((2, 16, 16), np.uint16),
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": psx,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": psy,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": psz,
                "PhysicalSizeZUnit": "µm",
            },
        )
        return path

    @staticmethod
    def _write_sparse_multifile_ome_tiff(tmpdir, present=None):
        """Multi-file embedded OME-TIFF declaring SizeC=3 with channel 1 missing.

        Returns (master_path, {channel_index: fill_value}) for the PRESENT
        channels; the absent channel is a ragged gap tifffile fills
        deterministically. Mirrors ``create_multifile_embedded_ome_tiff`` but omits
        one channel's TiffData/sibling to model a sparse acquisition.
        """
        import uuid as _uuid
        from pathlib import Path as _Path

        import tifffile

        h, w = 16, 16
        present = present or {0: 10, 2: 30}
        first = min(present)
        uuids = {c: f"urn:uuid:{_uuid.UUID(int=c + 1)}" for c in present}
        names = {c: f"sparse_{c}.ome.tif" for c in present}
        tiff_data = "".join(
            f'<TiffData FirstC="{c}" FirstZ="0" FirstT="0" IFD="0" PlaneCount="1">'
            f'<UUID FileName="{names[c]}">{uuids[c]}</UUID></TiffData>'
            for c in present
        )
        channels = "".join(
            f'<Channel ID="Channel:0:{c}" SamplesPerPixel="1"/>' for c in range(3)
        )
        master_xml = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
            f'UUID="{uuids[first]}"><Image ID="Image:0" Name="sparse">'
            '<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" '
            f'SizeX="{w}" SizeY="{h}" SizeZ="1" SizeC="3" SizeT="1">'
            f"{channels}{tiff_data}</Pixels></Image></OME>"
        )
        for c, val in present.items():
            xml = (
                master_xml
                if c == first
                else (
                    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
                    f'UUID="{uuids[c]}"><BinaryOnly UUID="{uuids[first]}" '
                    f'MetadataFile="{names[first]}"/></OME>'
                )
            )
            with tifffile.TiffWriter(str(_Path(tmpdir) / names[c])) as wr:
                wr.write(
                    np.full((h, w), val, np.uint16), description=xml, metadata=None
                )
        return str(_Path(tmpdir) / names[first]), present
