"""Phase 2 coverage for the native CZI adapter (biopb/biopb#799)."""

from pathlib import Path

import numpy as np
import pytest
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.core.discovery import (
    ClaimContext,
    DiscoveryState,
    LiveLocalContext,
)
from biopb_tensor_server.core.downsample import downsample_block

pytest.importorskip("pylibCZIrw")

from biopb_tensor_server.adapters import (  # noqa: E402
    CziAdapter,
    czi as czi_module,  # noqa: E402
    get_default_registry,
)
from biopb_tensor_server.adapters.czi import (  # noqa: E402
    _plane_axes,
    read_layout,
)
from biopb_tensor_server.fixtures import (  # noqa: E402
    create_zeiss_czi,
    create_zeiss_czi_scenes,
)


class _RemoteCtx(LiveLocalContext):
    """A claim context that reports itself remote, as a RemoteContext does."""

    @property
    def is_remote(self) -> bool:
        return True


def _source(path, source_id="czi", **kwargs):
    return SourceConfig(url=str(path), type="czi", source_id=source_id, **kwargs)


def _native(path, **kwargs):
    return CziAdapter.create_from_config(_source(path, **kwargs))


def test_local_czi_claims_natively_and_reads_through_libczi(tmp_path):
    path, expected = create_zeiss_czi(
        str(tmp_path), n_t=2, n_c=2, n_z=3, image_shape=(24, 32)
    )

    registry = get_default_registry()
    claims = registry.get_claims_for_path(ClaimContext(Path(path)), DiscoveryState())
    assert [claim.source_type for claim in claims] == ["czi"]

    source = registry.get_adapter_for_type("czi").create_from_config(_source(path))
    assert isinstance(source, CziAdapter)

    descriptors = source.list_tensor_descriptors()
    assert len(descriptors) == 1
    assert list(descriptors[0].dim_labels) == ["T", "C", "Z", "Y", "X"]
    assert list(descriptors[0].shape) == [2, 2, 3, 24, 32]
    # The listing is structural: the grid is the bound scene's (biopb/biopb#812).
    assert list(descriptors[0].chunk_shape) == []

    scene = source.get_tensor_adapter(descriptors[0].array_id)
    # One plane is the unit libCZI decodes; the transfer grid is built from
    # whole planes rather than being one (biopb/biopb#809).
    grid = list(scene.get_tensor_descriptor().chunk_shape)
    assert grid[-2:] == [24, 32]
    assert all(g <= s for g, s in zip(grid, [2, 2, 3, 24, 32], strict=True))

    whole = scene.get_data(ChunkBounds(start=[0, 0, 0, 0, 0], stop=[2, 2, 3, 24, 32]))
    np.testing.assert_array_equal(whole, expected)


def test_interior_crop_reads_only_the_requested_window(tmp_path):
    path, expected = create_zeiss_czi(str(tmp_path), n_c=2, n_z=3, image_shape=(24, 32))
    source = _native(path)
    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    bounds = ChunkBounds(start=[0, 1, 2, 5, 7], stop=[1, 2, 3, 20, 30])
    np.testing.assert_array_equal(
        scene.get_data(bounds), expected[0:1, 1:2, 2:3, 5:20, 7:30]
    )


def test_each_scene_reads_from_its_own_bounding_rectangle(tmp_path):
    """Scene rectangles are absolute CZI coordinates, not per-scene origins.

    BioIO advertises the *document's* bounding rectangle as the shape of every
    scene past the first, so its catalog row and its own reads disagree for
    this layout. The native descriptors are the scene rectangles throughout.
    """
    path, expected = create_zeiss_czi_scenes(str(tmp_path), n_scenes=2)
    source = _native(path)

    descriptors = source.list_tensor_descriptors()
    assert [descriptor.array_id.split("/")[-1] for descriptor in descriptors] == [
        "Scene:0",
        "Scene:1",
    ]
    assert [list(descriptor.shape) for descriptor in descriptors] == [
        list(expected[0].shape),
        list(expected[1].shape),
    ]

    for index, descriptor in enumerate(descriptors):
        scene = source.get_tensor_adapter(descriptor.array_id)
        shape = list(descriptor.shape)
        whole = scene.get_data(ChunkBounds(start=[0] * 5, stop=shape))
        np.testing.assert_array_equal(whole, expected[index])

        crop = scene.get_data(
            ChunkBounds(start=[0, 1, 1, 3, 5], stop=[1, 2, 2, 11, 17])
        )
        np.testing.assert_array_equal(crop, expected[index][0:1, 1:2, 1:2, 3:11, 5:17])


def test_unknown_scene_is_rejected(tmp_path):
    from biopb_tensor_server.core.errors import TensorNotFound

    path, _ = create_zeiss_czi(str(tmp_path), n_c=1, n_z=1, image_shape=(8, 8))
    source = _native(path)

    with pytest.raises(TensorNotFound):
        source.get_tensor_adapter("Scene:7")


def test_source_level_adapter_refuses_to_serve_pixels(tmp_path):
    path, _ = create_zeiss_czi(str(tmp_path), n_c=1, n_z=1, image_shape=(8, 8))
    source = _native(path)

    with pytest.raises(ValueError, match="source-level"):
        source.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 1, 8, 8]))


def test_scaling_items_become_the_physical_scale(tmp_path):
    path, _ = create_zeiss_czi(
        str(tmp_path),
        n_c=1,
        n_z=2,
        image_shape=(8, 8),
        pixel_size_um=(0.2, 0.3, 1.5),
    )
    source = _native(path)
    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    scale, unit = scene._physical_scale()
    # dim order is T, C, Z, Y, X; only the spatial axes carry a size.
    assert scale == pytest.approx([0.0, 0.0, 1.5, 0.3, 0.2])
    assert unit == ["", "", "µm", "µm", "µm"]


def test_descriptor_carries_the_physical_scale(tmp_path):
    path, _ = create_zeiss_czi(
        str(tmp_path),
        n_c=1,
        n_z=2,
        image_shape=(8, 8),
        pixel_size_um=(0.2, 0.3, 1.5),
    )
    source = _native(path)
    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    descriptor = scene.get_tensor_descriptor()
    scene._fill_physical_scale(descriptor)
    assert list(descriptor.physical_scale) == pytest.approx([0.0, 0.0, 1.5, 0.3, 0.2])


def test_configured_dim_labels_rename_axes_without_reordering(tmp_path):
    path, expected = create_zeiss_czi(str(tmp_path), n_c=2, n_z=3, image_shape=(16, 16))
    source = CziAdapter.create_from_config(
        _source(path, dim_labels=["time", "chan", "depth", "row", "col"])
    )

    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == ["time", "chan", "depth", "row", "col"]
    assert list(descriptor.shape) == [1, 2, 3, 16, 16]

    scene = source.get_tensor_adapter(descriptor.array_id)
    np.testing.assert_array_equal(
        scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 2, 3, 16, 16])), expected
    )


def test_file_url_is_read_as_a_local_path(tmp_path):
    """``file://`` is a local URL here, but libCZI takes a filesystem path."""
    path, expected = create_zeiss_czi(str(tmp_path), n_c=1, n_z=2, image_shape=(8, 8))
    source = CziAdapter.create_from_config(
        SourceConfig(url=f"file://{path}", type="czi", source_id="file-url")
    )
    assert isinstance(source, CziAdapter)

    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)
    np.testing.assert_array_equal(
        scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 2, 8, 8])), expected
    )


def test_wrong_rank_dim_labels_are_reported_not_silently_dropped(tmp_path, caplog):
    path, _ = create_zeiss_czi(str(tmp_path), n_c=1, n_z=2, image_shape=(8, 8))
    with caplog.at_level("WARNING"):
        source = CziAdapter.create_from_config(
            _source(path, dim_labels=["z", "y", "x"])
        )

    assert list(source.list_tensor_descriptors()[0].dim_labels) == [
        "T",
        "C",
        "Z",
        "Y",
        "X",
    ]
    assert "ignoring 3 configured dim_labels" in caplog.text


def test_metadata_is_the_image_information_subtree(tmp_path):
    path, _ = create_zeiss_czi(str(tmp_path), n_c=2, n_z=3, image_shape=(8, 8))
    metadata = _native(path).get_metadata()

    assert metadata["Image"]["SizeC"] == "2"
    assert metadata["Image"]["SizeZ"] == "3"


def test_reader_stays_warm_between_reads_and_closes_on_release(tmp_path):
    path, _ = create_zeiss_czi(str(tmp_path), n_c=1, n_z=2, image_shape=(8, 8))
    source = _native(path)
    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)

    assert scene._persistent_reader is None
    scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 1, 8, 8]))
    warm = scene._persistent_reader
    assert warm is not None

    scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 2, 8, 8]))
    assert scene._persistent_reader is warm

    source.close()
    assert scene._persistent_reader is None

    # A released reader reopens rather than staying broken.
    scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 1, 8, 8]))
    assert scene._persistent_reader is not None
    source.close()


def test_idle_reader_is_reaped(tmp_path):
    path, _ = create_zeiss_czi(str(tmp_path), n_c=1, n_z=1, image_shape=(8, 8))
    source = _native(path)
    scene = source.get_tensor_adapter(source.list_tensor_descriptors()[0].array_id)
    scene.get_data(ChunkBounds(start=[0] * 5, stop=[1, 1, 1, 8, 8]))
    assert scene._persistent_reader is not None

    scene._persistent_last_access -= czi_module._reader_reaper.ttl + 1
    czi_module._reader_reaper._sweep()
    assert scene._persistent_reader is None


@pytest.mark.parametrize(
    "bounding_box, axes, sizes",
    [
        (
            {"T": (0, 3), "C": (0, 2), "Z": (0, 4), "X": (0, 8), "Y": (0, 8)},
            ("T", "C", "Z"),
            {"T": 3, "C": 2, "Z": 4},
        ),
        # Absent canonical axes are size 1, not missing -- an ordinary document
        # keeps the T/C/Z/Y/X shape BioIO reports.
        (
            {"C": (0, 2), "X": (0, 8), "Y": (0, 8)},
            ("T", "C", "Z"),
            {"T": 1, "C": 2, "Z": 1},
        ),
        # A singleton exotic axis is not carried: libCZI's own default plane
        # coordinates include a dimension only at size > 1.
        (
            {"H": (0, 1), "Z": (0, 2), "X": (0, 8), "Y": (0, 8)},
            ("T", "C", "Z"),
            {"T": 1, "C": 1, "Z": 2},
        ),
        # A varying one keeps its own name, ahead of the canonical axes.
        (
            {"H": (0, 3), "Z": (0, 2), "X": (0, 8), "Y": (0, 8)},
            ("H", "T", "C", "Z"),
            {"H": 3, "T": 1, "C": 1, "Z": 2},
        ),
        # Several, outermost first.
        (
            {"H": (0, 2), "V": (0, 4), "X": (0, 8), "Y": (0, 8)},
            ("V", "H", "T", "C", "Z"),
            {"V": 4, "H": 2, "T": 1, "C": 1, "Z": 1},
        ),
        # The extents are counts, not index ranges -- libCZI reports (0, n)
        # whatever the file's own coordinates are -- so the start carries no
        # information and is not read.
        (
            {"Z": (2, 5), "X": (0, 8), "Y": (0, 8)},
            ("T", "C", "Z"),
            {"T": 1, "C": 1, "Z": 3},
        ),
    ],
)
def test_plane_axes_carries_exotic_dimensions_under_their_own_names(
    bounding_box, axes, sizes
):
    assert _plane_axes(bounding_box) == (axes, sizes)


def test_mixed_pixel_types_raise_rather_than_defer(tmp_path):
    """BioIO cannot represent this either, so deferring would only obscure it.

    Its reader reshapes every channel into one array and fails with "cannot
    reshape array of size N", raised from inside the dask graph.
    """
    from pylibCZIrw import czi as pyczi

    path = Path(tmp_path) / "mixed.czi"
    with pyczi.create_czi(str(path)) as writer:
        writer.write(np.zeros((8, 8, 1), np.uint16), plane={"C": 0, "Z": 0, "T": 0})
        writer.write(np.zeros((8, 8, 3), np.uint8), plane={"C": 1, "Z": 0, "T": 0})

    with pytest.raises(ValueError, match="mixes pixel types"):
        read_layout(str(path))


def test_rgb_document_reads_natively_with_a_samples_axis(tmp_path):
    """A Bgr* document gets a trailing samples axis, in the file's own order.

    libCZI hands back the three samples as stored and BioIO reports the
    identical values, so neither reader reorders them.
    """
    from pylibCZIrw import czi as pyczi

    path = Path(tmp_path) / "rgb.czi"
    expected = np.zeros((6, 8, 3), dtype=np.uint8)
    expected[..., 0], expected[..., 1], expected[..., 2] = 11, 22, 33
    with pyczi.create_czi(str(path)) as writer:
        writer.write(expected, plane={"C": 0, "Z": 0, "T": 0})

    source = _native(path)
    descriptor = source.list_tensor_descriptors()[0]
    assert list(descriptor.dim_labels) == ["T", "C", "Z", "Y", "X", "S"]
    assert list(descriptor.shape) == [1, 1, 1, 6, 8, 3]
    assert descriptor.dtype == np.dtype(np.uint8).str

    scene = source.get_tensor_adapter(descriptor.array_id)
    whole = scene.get_data(ChunkBounds(start=[0] * 6, stop=[1, 1, 1, 6, 8, 3]))
    np.testing.assert_array_equal(whole[0, 0, 0], expected)

    # A crop still addresses Y/X through the ROI and slices samples.
    crop = scene.get_data(
        ChunkBounds(start=[0, 0, 0, 2, 3, 1], stop=[1, 1, 1, 5, 7, 3])
    )
    np.testing.assert_array_equal(crop[0, 0, 0], expected[2:5, 3:7, 1:3])


def test_remote_source_is_refused_rather_than_rerouted(tmp_path):
    """Discovery never routes a remote CZI here -- ``claim`` declines it."""
    path = Path(tmp_path) / "img.czi"
    path.write_bytes(b"ZISRAWFILE")

    assert CziAdapter.claim(_RemoteCtx(path), DiscoveryState()) is None

    source = SourceConfig(url="s3://bucket/img.czi", type="czi", source_id="remote")
    with pytest.raises(ValueError, match="only supports local files"):
        CziAdapter.create_from_config(source)


def test_remote_czi_is_claimed_by_the_bioio_adapter(tmp_path):
    """The hand-off is the registry's, not an import: BioIO claims what this
    adapter declines."""
    pytest.importorskip("bioio_czi")
    path = Path(tmp_path) / "img.czi"
    path.write_bytes(b"ZISRAWFILE")

    claims = get_default_registry().get_claims_for_path(
        _RemoteCtx(path), DiscoveryState()
    )
    assert [c.source_type for c in claims] == ["zeiss"]


def test_module_does_not_import_bioio():
    """No fallback path, so no BioIO dependency in this module."""
    source = Path(czi_module.__file__).read_text()
    assert "adapters.bioio" not in source
    assert "import bioio" not in source


def test_claim_is_definite_under_a_cloud_root(tmp_path):
    """The claim reads nothing, so a cloud placeholder is claimed, not declined.

    Deferring a non-resident file belongs to the source manager
    (``_claim_is_unresolved``). Declining here would also outlive the
    placeholder: the resolve-time re-claim still carries ``cloud_root=True``,
    so a hydrated file would never reach this adapter.
    """
    path, _ = create_zeiss_czi(str(tmp_path), n_c=1, n_z=1, image_shape=(8, 8))

    claim = CziAdapter.claim(
        LiveLocalContext(Path(path), cloud_root=True), DiscoveryState()
    )
    assert claim is not None
    assert claim.source_type == "czi"
    assert claim.unresolved is False


def test_claim_does_not_open_the_file(tmp_path):
    """Not even a stat-and-open sniff: the claim is extension-only."""

    class _RaisingReadCtx(LiveLocalContext):
        def read_text(self, subpath: str = "") -> str:
            raise AssertionError("claim() must not read content")

        def open(self, mode: str = "rb") -> object:
            raise AssertionError("claim() must not open the file")

    path = tmp_path / "not-really.czi"
    path.write_bytes(b"\x00\x01\x02\x03")

    claim = CziAdapter.claim(_RaisingReadCtx(path), DiscoveryState())
    assert claim is not None and claim.source_type == "czi"


def test_unopenable_file_raises_instead_of_falling_back(tmp_path):
    """A file libCZI cannot open is an error, not a slow path.

    BioIO reads CZI through this same pylibCZIrw (`use_aicspylibczi=False`), so
    the fallback cannot read what the probe could not.
    """
    path = tmp_path / "corrupt.czi"
    path.write_bytes(b"not a czi at all")

    with pytest.raises(Exception) as excinfo:
        CziAdapter.create_from_config(_source(path))
    assert not isinstance(excinfo.value, AssertionError)


class TestZoomServesNearest:
    """``get_scaled_data`` routes ``nearest`` to libCZI's ``zoom=`` (#640).

    The gate is divisibility, not powers of two: ``zoom=`` returns
    ``floor(extent / factor)`` where the contract requires ``ceil``, so the two
    agree exactly when the factor divides the extent. Every assertion here is
    bit-identity against the default path, which is the only thing that makes
    swapping in a different decoder legitimate.
    """

    @staticmethod
    def _bounds(start, stop):
        return ChunkBounds(start=list(start), stop=list(stop))

    def _adapter(self, tmp_path, shape=(120, 120), **kw):
        path, expected = create_zeiss_czi(
            str(tmp_path), n_t=1, n_c=1, n_z=1, image_shape=shape, **kw
        )
        source = _native(path)
        descriptor = source.list_tensor_descriptors()[0]
        return source.get_tensor_adapter(descriptor.array_id.split("/")[-1]), expected

    @pytest.mark.parametrize("factor", [2, 3, 4, 5, 6, 8])
    def test_zoom_is_bit_identical_to_the_default(self, tmp_path, factor):
        """Non-dyadic factors included -- 3 and 5 divide 120 and are exact."""
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 1, 120, 120))
        scale = (1, 1, 1, factor, factor)

        # Without this the assertions below pass on the fallback path too.
        assert adapter._zoom_factor(bounds, scale, "nearest") == factor
        fused = adapter.get_scaled_data(bounds, scale, "nearest")
        expected = downsample_block(adapter.get_data(bounds), scale, "nearest")

        assert fused.shape == expected.shape
        assert fused.dtype == expected.dtype
        assert np.array_equal(fused, expected)

    def test_an_indivisible_extent_falls_back(self, tmp_path, monkeypatch):
        """The one-off shape: ``floor`` and ``ceil`` disagree, so decline.

        Asserted through the fallback actually running, because a silent
        one-row-short result is exactly what this gate exists to prevent.
        """
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 1, 118, 118))
        scale = (1, 1, 1, 8, 8)
        assert adapter._zoom_factor(bounds, scale, "nearest") is None

        out = adapter.get_scaled_data(bounds, scale, "nearest")
        assert out.shape[-2:] == (15, 15)  # ceil(118 / 8), not floor
        assert np.array_equal(
            out, downsample_block(adapter.get_data(bounds), scale, "nearest")
        )

    def test_area_never_zooms(self, tmp_path):
        """``zoom=`` picks one pixel per block: 100% of pixels differ from area."""
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 1, 120, 120))
        scale = (1, 1, 1, 4, 4)
        assert adapter._zoom_factor(bounds, scale, "area") is None
        assert np.array_equal(
            adapter.get_scaled_data(bounds, scale, "area"),
            downsample_block(adapter.get_data(bounds), scale, "area"),
        )

    def test_a_scaled_plane_axis_declines(self, tmp_path):
        """Reducing across reads is the default path's job, not one read's."""
        path, _ = create_zeiss_czi(
            str(tmp_path), n_t=1, n_c=1, n_z=4, image_shape=(120, 120)
        )
        source = _native(path)
        descriptor = source.list_tensor_descriptors()[0]
        adapter = source.get_tensor_adapter(descriptor.array_id.split("/")[-1])
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 4, 120, 120))

        assert adapter._zoom_factor(bounds, (1, 1, 2, 4, 4), "nearest") is None
        assert adapter._zoom_factor(bounds, (1, 1, 1, 4, 4), "nearest") == 4

    def test_unequal_y_and_x_factors_decline(self, tmp_path):
        """One read takes one zoom; two factors are two reductions."""
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 1, 120, 120))
        assert adapter._zoom_factor(bounds, (1, 1, 1, 4, 2), "nearest") is None
        assert adapter._zoom_factor(bounds, (1, 1, 1, 1, 1), "nearest") is None

    def test_an_interior_window_zooms_from_its_own_origin(self, tmp_path):
        """The pick is relative to the ROI, so an off-origin chunk is exact."""
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 7, 23), (1, 1, 1, 103, 119))
        scale = (1, 1, 1, 8, 8)

        assert adapter._zoom_factor(bounds, scale, "nearest") == 8
        assert np.array_equal(
            adapter.get_scaled_data(bounds, scale, "nearest"),
            downsample_block(adapter.get_data(bounds), scale, "nearest"),
        )

    def test_a_shape_disagreement_falls_back_rather_than_failing(
        self, tmp_path, monkeypatch
    ):
        """If libCZI ever rounds differently, serve correct pixels anyway."""
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 1, 120, 120))
        scale = (1, 1, 1, 4, 4)

        reader = adapter._acquire_reader()
        reader_type = type(reader)
        real_read = reader_type.read

        def truncating(self, **kw):
            plane = real_read(self, **kw)
            return plane[:-1, :-1] if kw.get("zoom") is not None else plane

        monkeypatch.setattr(reader_type, "read", truncating)
        out = adapter.get_scaled_data(bounds, scale, "nearest")
        assert np.array_equal(
            out, downsample_block(adapter.get_data(bounds), scale, "nearest")
        )

    def test_the_zoom_reaches_libczi(self, tmp_path, monkeypatch):
        """The wiring, not the arithmetic: one zoomed read per plane, no more."""
        adapter, _ = self._adapter(tmp_path)
        bounds = self._bounds((0, 0, 0, 0, 0), (1, 1, 1, 120, 120))

        zooms = []
        reader = adapter._acquire_reader()
        real_read = reader.read
        monkeypatch.setattr(
            type(reader),
            "read",
            lambda self, **kw: (zooms.append(kw.get("zoom")), real_read(**kw))[1],
        )

        adapter.get_scaled_data(bounds, (1, 1, 1, 8, 8), "nearest")
        assert zooms == [1.0 / 8]

        zooms.clear()
        adapter.get_scaled_data(bounds, (1, 1, 1, 8, 8), "area")
        assert zooms == [None]
