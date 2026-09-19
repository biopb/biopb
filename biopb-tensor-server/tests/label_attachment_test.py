"""Label sets are tensors of their image (biopb/biopb#1059 step 2).

A source answers for them through the base API -- ``label_sets``,
``resolve_tensor``, ``resolve_chunk_adapter`` -- so the format's own listing
and routing stay untouched, the catalog lists them after the image tensors, and
the serve path reaches them by ``array_id`` like any tensor. Two origins here:
an OME-Zarr's NGFF ``labels/`` group (read by the format) and a finished
sidecar under ``<write_dir>/labels/<source_id>/`` (attached by the registry).
"""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr
from biopb.tensor import TensorFlightClient
from biopb_tensor_server.adapters.labels import (
    LabelSetAdapter,
    sidecar_attrs,
    sidecar_dir,
)
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, with_upload_state
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.config import CacheConfig, SourceConfig
from biopb_tensor_server.core.errors import TensorNotFound, WriteNotSupportedError
from biopb_tensor_server.core.labels import label_field, split_label_field
from biopb_tensor_server.core.normalize import NormalizingAdapter
from biopb_tensor_server.core.source_registry import SourceRegistry
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr

from tests import catalog_server

SHAPE = (64, 64)
CHUNK = (32, 32)


def _ngff_label_attrs(axes=("y", "x"), levels=2, extra=None):
    datasets = [
        {
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": [2**i] * len(axes)}
            ],
        }
        for i in range(levels)
    ]
    attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [{"name": a, "type": "space"} for a in axes],
                "datasets": datasets,
            }
        ],
        "image-label": {
            "version": "0.4",
            "colors": [{"label-value": 7, "rgba": [255, 0, 0, 255]}],
        },
    }
    attrs.update(extra or {})
    return attrs


def _write_label_group(
    group: Path,
    *,
    dtype="uint32",
    axes=("y", "x"),
    levels=2,
    fill=7,
    extra=None,
    shape=SHAPE,
):
    """An NGFF label image at *group*: ``levels`` arrays plus its ``.zattrs``."""
    g = zarr.open_group(str(group), mode="w")
    for i in range(levels):
        shape = tuple(s // 2**i for s in shape)
        arr = g.create_dataset(str(i), shape=shape, chunks=CHUNK, dtype=dtype)
        arr[:] = fill if i == 0 else fill + i
    (group / ".zattrs").write_text(json.dumps(_ngff_label_attrs(axes, levels, extra)))
    return group


@pytest.fixture
def image(tmp_path):
    """An OME-Zarr image with a ``labels/`` group: ``nuclei`` (two levels),
    ``flat`` (one level), ``xy`` (non-canonical axes) and ``bad`` (float)."""
    zarr_path, _, _ = create_multiresolution_ome_zarr(
        str(tmp_path / "img"), n_levels=2, base_shape=SHAPE, chunk_size=CHUNK
    )
    root = Path(zarr_path)
    labels = root / "labels"
    labels.mkdir()
    (labels / ".zattrs").write_text(
        json.dumps({"labels": ["nuclei", "flat", "xy", "bad"]})
    )
    _write_label_group(labels / "nuclei")
    _write_label_group(labels / "flat", levels=1, fill=5)
    _write_label_group(labels / "xy", axes=("x", "y"), levels=1, fill=9)
    _write_label_group(labels / "bad", dtype="float32")
    return root


def _adapter(root, source_id="oz1"):
    return OmeZarrAdapter.create_from_config(
        SourceConfig(url=str(root), type="ome-zarr", source_id=source_id)
    )


class TestTheFieldShape:
    def test_split(self):
        assert split_label_field("labels/nuclei") == ("", "nuclei", None)
        assert split_label_field("A/1/labels/nuclei/2") == ("A/1", "nuclei", "2")
        assert split_label_field("labels/nuclei").set_field == "labels/nuclei"
        assert split_label_field("labels") is None
        assert split_label_field("A/1") is None
        assert split_label_field(None) is None

    def test_compose(self):
        assert label_field("", "n") == "labels/n"
        assert label_field("A/1", "n") == "A/1/labels/n"


class TestANativeSetIsATensorOfItsImage:
    def test_listed_after_the_image_and_readable_by_id(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))

        ids = [t.array_id for t in catalog_tensors(adapter)]
        assert ids[0] == "oz1"
        assert set(ids[1:]) == {"oz1/labels/nuclei", "oz1/labels/flat", "oz1/labels/xy"}

        nuclei = adapter.resolve_tensor("oz1/labels/nuclei")
        assert isinstance(nuclei, LabelSetAdapter)
        assert nuclei.array_id == "oz1/labels/nuclei"
        assert adapter.resolve_tensor("labels/nuclei") is nuclei
        desc = nuclei.get_tensor_descriptor()
        assert (list(desc.shape), desc.dtype) == ([64, 64], "<u4")

    def test_a_float_set_is_skipped_and_the_image_still_registers(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        assert "labels/bad" not in adapter.label_sets
        assert adapter.resolve_tensor("oz1").array_id == "oz1"

    def test_an_unknown_set_is_the_formats_miss(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        with pytest.raises(TensorNotFound):
            adapter.resolve_tensor("labels/nope")

    def test_the_formats_own_routing_is_untouched(self, image):
        raw = _adapter(image)
        adapter = SourceRegistry().register("oz1", raw)
        assert [t.array_id for t in raw.list_tensor_descriptors()] == ["oz1"]
        assert adapter.resolve_tensor(None).array_id == "oz1"
        assert adapter.resolve_chunk_adapter("1").array_id == "oz1/1"

    def test_a_native_level_routes_under_the_set(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        nuclei = adapter.resolve_tensor("labels/nuclei")

        level = adapter.resolve_chunk_adapter("labels/nuclei/1")
        assert level.array_id == "oz1/labels/nuclei/1"
        assert list(level.get_tensor_descriptor().shape) == [32, 32]
        assert level.content_version == nuclei.content_version
        assert adapter.resolve_chunk_adapter("labels/nuclei") is nuclei

    def test_content_version_is_the_images(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        nuclei = adapter.resolve_tensor("labels/nuclei")
        assert nuclei.content_version == adapter.content_version
        assert nuclei.content_version is not None

    def test_the_ladder_is_native_or_nearest_never_area(self, image):
        from biopb_tensor_server.core.config import PyramidConfig

        adapter = SourceRegistry().register("oz1", _adapter(image))
        cfg = PyramidConfig(reduction_method="area")
        nuclei = adapter.resolve_tensor("labels/nuclei")
        native = nuclei._advertised_pyramid(nuclei.get_tensor_descriptor(), cfg)
        assert all(
            lvl.reduction_method == "precompute" and lvl.native for lvl in native
        )

        flat = adapter.resolve_tensor("labels/flat")
        computed = flat._advertised_pyramid(flat.get_tensor_descriptor(), cfg)
        assert computed and all(lvl.reduction_method == "nearest" for lvl in computed)

    def test_metadata_names_the_image(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        meta = adapter.resolve_tensor("labels/nuclei").get_tensor_metadata()
        assert meta["image-label"]["source"] == {"image": "oz1"}
        assert meta["image-label"]["colors"][0]["label-value"] == 7
        assert meta["multiscales"][0]["datasets"][1]["path"] == "1"

    def test_a_set_is_read_only(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        with pytest.raises(WriteNotSupportedError):
            adapter.resolve_tensor("labels/nuclei").put_chunk(None, None, (), None)

    def test_a_non_canonical_set_is_normalized_like_any_tensor(self, image):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        xy = adapter.resolve_tensor("labels/xy")
        assert isinstance(xy, NormalizingAdapter)
        assert list(xy.get_tensor_descriptor().dim_labels) == ["y", "x"]
        assert [
            t.dim_labels[0]
            for t in catalog_tensors(adapter)
            if t.array_id.endswith("/xy")
        ] == ["y"]


class TestOverTheWire:
    @pytest.fixture
    def served(self, image, tmp_path):
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        server = catalog_server(location="grpc://localhost:0")
        registered = server.sources.register("oz1", _adapter(image))
        server.metadata_db.sync_source_added("oz1", registered)
        server.mark_ready()
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            yield server, TensorFlightClient(f"grpc://localhost:{server.port}")
        finally:
            server.shutdown()
            CacheManager.reset()

    def test_catalog_lists_the_sets_after_the_image(self, served):
        server, client = served
        rows = (
            server.metadata_db.query(
                "SELECT list_transform(tensors, t -> t.array_id) FROM sources "
                "WHERE source_id = 'oz1'"
            )
            .column(0)
            .to_pylist()[0]
        )
        assert rows[0] == "oz1"
        assert "oz1/labels/nuclei" in rows
        assert client.get_source("oz1").tensors[0].array_id == "oz1"

    def test_read_the_set_and_its_native_level(self, served):
        _, client = served
        full = client.get_tensor("oz1/labels/nuclei")
        assert full.shape == SHAPE and full.dtype == np.uint32
        assert full[:8, :8].compute().tolist() == np.full((8, 8), 7).tolist()

        desc = client.get_descriptor(
            "oz1/labels/nuclei", with_metadata=True, with_pyramid=True
        )
        assert [lvl.reduction_method for lvl in desc.pyramid] == [
            "precompute",
            "precompute",
        ]
        meta = json.loads(desc.metadata_json)["metadata"]
        assert meta["image-label"]["source"] == {"image": "oz1"}

        coarse = client.get_tensor(
            "oz1/labels/nuclei", scale_hint=[2, 2], reduction_method="precompute"
        )
        assert coarse[:4, :4].compute().max() == 8


class TestASidecarIsAttachedAtRegistration:
    def _sidecar(
        self,
        labels_dir,
        source_id,
        name,
        *,
        state=None,
        image_field="",
        token=b"\x01\x02",
        shape=SHAPE,
        axes=("y", "x"),
    ):
        group = sidecar_dir(labels_dir, source_id) / f"{name}.zarr"
        group.parent.mkdir(parents=True, exist_ok=True)
        extra = sidecar_attrs(image_field, token)
        if state is not None:
            extra = with_upload_state(extra, state)
        _write_label_group(group, levels=1, fill=4, extra=extra, shape=shape, axes=axes)
        return group

    def test_a_sidecar_that_does_not_span_its_image_is_skipped(self, image, tmp_path):
        """The upload refuses such a set at create; the read checks again for
        what reached the directory by other means."""
        labels_dir = tmp_path / "w" / "labels"
        self._sidecar(labels_dir, "oz1", "small", shape=(32, 32))
        self._sidecar(
            labels_dir, "oz1", "extra", shape=(2, 64, 64), axes=("z", "y", "x")
        )
        self._sidecar(labels_dir, "oz1", "orphan", image_field="Image:9")
        self._sidecar(labels_dir, "oz1", "fits")

        adapter = SourceRegistry(labels_dir=labels_dir).register("oz1", _adapter(image))

        sidecars = {
            f
            for f in adapter.label_sets
            if f.split("/")[-1] not in ("nuclei", "flat", "xy")
        }
        assert sidecars == {"labels/fits"}

    def test_extent_mismatch_says_why(self):
        from biopb_tensor_server.adapters.labels import extent_mismatch

        assert (
            extent_mismatch(["y", "x"], [64, 64], ["c", "y", "x"], [3, 64, 64]) is None
        )
        assert (
            extent_mismatch(
                ["Z", "y", "x"], [2, 64, 64], ["depth", "y", "x"], [2, 64, 64]
            )
            is None
        )
        assert "shape" in extent_mismatch(["y", "x"], [32, 64], ["y", "x"], [64, 64])
        assert "axes" in extent_mismatch(
            ["y", "x"], [64, 64], ["z", "y", "x"], [2, 64, 64]
        )
        assert "axes" in extent_mismatch(
            ["c", "y", "x"], [3, 64, 64], ["c", "y", "x"], [3, 64, 64]
        )

    def test_a_ready_sidecar_is_a_set_a_pending_one_is_not(self, image, tmp_path):
        labels_dir = tmp_path / "w" / "labels"
        self._sidecar(labels_dir, "oz1", "mine")
        self._sidecar(labels_dir, "oz1", "half", state=UPLOAD_PENDING)
        self._sidecar(labels_dir, "other", "theirs")

        adapter = SourceRegistry(labels_dir=labels_dir).register("oz1", _adapter(image))

        assert "labels/mine" in adapter.label_sets
        assert "labels/half" not in adapter.label_sets
        assert "labels/theirs" not in adapter.label_sets
        mine = adapter.resolve_tensor("oz1/labels/mine")
        assert mine.content_version == b"\x01\x02"
        assert mine.get_tensor_metadata()["image-label"]["source"] == {"image": "oz1"}
        assert "biopb" not in mine.get_tensor_metadata()

    def test_no_labels_dir_means_no_sidecars(self, image, tmp_path):
        self._sidecar(tmp_path / "w" / "labels", "oz1", "mine")
        adapter = SourceRegistry().register("oz1", _adapter(image))
        assert "labels/mine" not in adapter.label_sets

    def test_attach_and_detach_by_hand(self, image, tmp_path):
        adapter = SourceRegistry().register("oz1", _adapter(image))
        group = self._sidecar(tmp_path / "w" / "labels", "oz1", "late")
        from biopb_tensor_server.adapters.labels import open_label_set

        late = open_label_set(
            group,
            source_id="oz1",
            field="labels/late",
            content_version=b"x",
            parent_array_id="oz1",
        )
        adapter.attach_label_set("labels/late", late)
        assert adapter.resolve_tensor("labels/late").array_id == "oz1/labels/late"
        assert adapter.detach_label_set("labels/late") is not None
        with pytest.raises(TensorNotFound):
            adapter.resolve_tensor("labels/late")

    def test_the_server_wires_its_write_dir(self, image, tmp_path):
        labels_dir = tmp_path / "w" / "labels"
        self._sidecar(labels_dir, "oz1", "mine")
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        server = catalog_server(location="grpc://localhost:0", write_dir=tmp_path / "w")
        try:
            registered = server.sources.register("oz1", _adapter(image))
            server.metadata_db.sync_source_added("oz1", registered)
            ids = (
                server.metadata_db.query(
                    "SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t)"
                )
                .column(0)
                .to_pylist()
            )
            assert "oz1/labels/mine" in ids
        finally:
            server.shutdown()
            CacheManager.reset()
