"""Label sets are tensors of their image (biopb/biopb#1059 step 2).

A source answers for them through the base API -- ``label_sets``,
``resolve_tensor``, ``resolve_chunk_adapter`` -- so the format's own listing
and routing stay untouched, the catalog lists them after the image tensors, and
the serve path reaches them by ``array_id`` like any tensor. Two origins here:
an OME-Zarr's NGFF ``labels/`` group (read by the format) and a finished
sidecar under ``<write_dir>/labels/<source_id>/`` (attached at registration).
Whatever the origin, a set is listed only if it spans the image it binds to.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from biopb_tensor_server.adapters.labels import (
    LabelSetAdapter,
    open_label_set,
    sidecar_attacher,
    sidecar_attrs,
    sidecar_dir,
)
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, with_upload_state
from biopb_tensor_server.core.adapter_base import catalog_tensors
from biopb_tensor_server.core.config import PyramidConfig, SourceConfig
from biopb_tensor_server.core.errors import TensorNotFound, WriteNotSupportedError
from biopb_tensor_server.core.labels import (
    extent_mismatch,
    label_field,
    label_image_axes,
    split_label_field,
)
from biopb_tensor_server.core.normalize import NormalizingAdapter
from biopb_tensor_server.core.source_registry import SourceRegistry
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr

from tests import register_and_catalog

SHAPE = (64, 64)
CHUNK = (32, 32)


def _write_label_group(
    group: Path,
    *,
    dtype="uint32",
    axes=("y", "x"),
    levels=2,
    fill=7,
    extra=None,
    shape=SHAPE,
    array_attrs=None,
):
    """An NGFF label image at *group*: ``levels`` arrays plus its ``.zattrs``."""
    g = zarr.open_group(str(group), mode="w")
    for i in range(levels):
        arr = g.create_dataset(
            str(i), shape=tuple(s >> i for s in shape), chunks=CHUNK, dtype=dtype
        )
        arr[:] = fill if i == 0 else fill + i
        if array_attrs:
            arr.attrs.update(array_attrs)
    attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [{"name": a, "type": "space"} for a in axes],
                "datasets": [
                    {
                        "path": str(i),
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [2**i] * len(axes)}
                        ],
                    }
                    for i in range(levels)
                ],
            }
        ],
        "image-label": {
            "version": "0.4",
            "colors": [{"label-value": 7, "rgba": [255, 0, 0, 255]}],
        },
    }
    attrs.update(extra or {})
    (group / ".zattrs").write_text(json.dumps(attrs))
    return group


@pytest.fixture
def image(tmp_path):
    """An OME-Zarr image with a ``labels/`` group: ``nuclei`` (two levels),
    ``flat`` (one level), ``xy`` (non-canonical axes), ``bad`` (float) and
    ``small`` (does not span the image)."""
    zarr_path, _, _ = create_multiresolution_ome_zarr(
        str(tmp_path / "img"), n_levels=2, base_shape=SHAPE, chunk_size=CHUNK
    )
    root = Path(zarr_path)
    labels = root / "labels"
    labels.mkdir()
    (labels / ".zattrs").write_text(
        json.dumps({"labels": ["nuclei", "flat", "xy", "bad", "small"]})
    )
    _write_label_group(labels / "nuclei", array_attrs={"_ARRAY_DIMENSIONS": ["y", "x"]})
    _write_label_group(labels / "flat", levels=1, fill=5)
    _write_label_group(labels / "xy", axes=("x", "y"), levels=1, fill=9)
    _write_label_group(labels / "bad", dtype="float32")
    _write_label_group(labels / "small", levels=1, shape=(32, 32))
    return root


def _adapter(root, source_id="oz1"):
    return OmeZarrAdapter.create_from_config(
        SourceConfig(url=str(root), type="ome-zarr", source_id=source_id)
    )


@pytest.fixture
def registered(image):
    return SourceRegistry().register("oz1", _adapter(image))


NATIVE = {"labels/nuclei", "labels/flat", "labels/xy"}


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

    def test_extent_mismatch_says_why(self):
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

    def test_image_axes_states_what_the_extent_rule_implies(self):
        # The mapping a client would otherwise re-derive. Matching by NAME gets
        # t/z right and an unnamed axis wrong, which is the whole reason the
        # server says it out loud (biopb/biopb#1059).
        assert label_image_axes(["t", "z", "y", "x"], ["t", "c", "z", "y", "x"]) == [
            0,
            2,
            3,
            4,
        ]
        assert label_image_axes(["y", "x"], ["y", "x"]) == [0, 1]
        # An interleaved samples axis is NOT dropped -- `label_extent` drops only
        # the channel -- so it maps like any other axis.
        assert label_image_axes(["t", "y", "x", "s"], ["t", "c", "y", "x", "s"]) == [
            0,
            2,
            3,
            4,
        ]
        # An unnamed axis keeps its place; the set's `a0` is the image's axis 0
        # even though the image spells the same slider `a1`.
        assert label_image_axes(["", "y", "x"], ["", "c", "y", "x"]) == [0, 2, 3]

    def test_image_axes_is_none_for_a_set_that_does_not_span(self):
        # Nothing to state, and the same set `label_sets` drops.
        assert label_image_axes(["z", "y", "x"], ["t", "c", "z", "y", "x"]) is None


class TestANativeSetIsATensorOfItsImage:
    def test_listed_after_the_image_and_readable_by_id(self, registered):
        ids = [t.array_id for t in catalog_tensors(registered)]
        assert ids[0] == "oz1"
        assert set(ids[1:]) == {f"oz1/{f}" for f in NATIVE}

        nuclei = registered.resolve_tensor("oz1/labels/nuclei")
        assert isinstance(nuclei, LabelSetAdapter)
        assert nuclei.array_id == "oz1/labels/nuclei"
        assert registered.resolve_tensor("labels/nuclei") is nuclei
        desc = nuclei.get_tensor_descriptor()
        assert (list(desc.shape), desc.dtype) == ([64, 64], "<u4")

    def test_a_set_that_cannot_open_or_does_not_span_is_dropped(self, registered):
        """``bad`` is a float (the reader skips it); ``small`` is 32x32 on a
        64x64 image (the merge drops it). The image registers either way."""
        assert set(registered.label_sets) == NATIVE
        assert registered.resolve_tensor("oz1").array_id == "oz1"

    def test_an_unknown_set_is_the_formats_miss(self, registered):
        with pytest.raises(TensorNotFound):
            registered.resolve_tensor("labels/nope")

    def test_the_formats_own_routing_is_untouched(self, image):
        raw = _adapter(image)
        adapter = SourceRegistry().register("oz1", raw)
        assert [t.array_id for t in raw.list_tensor_descriptors()] == ["oz1"]
        assert adapter.resolve_tensor(None).array_id == "oz1"
        assert adapter.resolve_chunk_adapter("1").array_id == "oz1/1"

    def test_a_native_level_routes_under_the_set(self, registered):
        """``nuclei``'s arrays carry their own ``.zattrs``, which used to stop
        the level-opening walk one directory too early."""
        nuclei = registered.resolve_tensor("labels/nuclei")

        level = registered.resolve_chunk_adapter("labels/nuclei/1")
        assert level.array_id == "oz1/labels/nuclei/1"
        assert list(level.get_tensor_descriptor().shape) == [32, 32]
        assert level.content_version == nuclei.content_version
        assert registered.resolve_chunk_adapter("labels/nuclei") is nuclei

    def test_content_version_is_the_images_for_set_and_levels_alike(self, registered):
        nuclei = registered.resolve_tensor("labels/nuclei")
        assert nuclei.content_version == registered.content_version is not None
        # The image's own native levels share it too: one token per source.
        assert (
            registered.resolve_chunk_adapter("1").content_version
            == registered.content_version
        )

    def test_the_ladder_is_native_or_nearest_never_area(self, registered):
        cfg = PyramidConfig(reduction_method="area")
        nuclei = registered.resolve_tensor("labels/nuclei")
        native = nuclei._advertised_pyramid(nuclei.get_tensor_descriptor(), cfg)
        assert all(
            lvl.reduction_method == "precompute" and lvl.native for lvl in native
        )

        flat = registered.resolve_tensor("labels/flat")
        computed = flat._advertised_pyramid(flat.get_tensor_descriptor(), cfg)
        assert computed and all(lvl.reduction_method == "nearest" for lvl in computed)

    def test_metadata_names_the_image(self, registered):
        meta = registered.resolve_tensor("labels/nuclei").get_tensor_metadata()
        assert meta["image-label"]["source"] == {"image": "oz1"}
        assert meta["image-label"]["colors"][0]["label-value"] == 7
        assert meta["multiscales"][0]["datasets"][1]["path"] == "1"

    def test_a_set_is_read_only(self, registered):
        with pytest.raises(WriteNotSupportedError):
            registered.resolve_tensor("labels/nuclei").put_chunk(None, None, (), None)

    def test_a_non_canonical_set_is_normalized_like_any_tensor(self, registered):
        xy = registered.resolve_tensor("labels/xy")
        assert isinstance(xy, NormalizingAdapter)
        assert list(xy.get_tensor_descriptor().dim_labels) == ["y", "x"]
        assert [
            t.dim_labels[0]
            for t in catalog_tensors(registered)
            if t.array_id.endswith("/xy")
        ] == ["y"]


class TestOverTheWire:
    @pytest.fixture
    def served(self, writable_server, image):
        register_and_catalog(writable_server, "oz1", _adapter(image))
        return writable_server

    def test_catalog_lists_the_sets_after_the_image(self, served, client):
        rows = (
            served.metadata_db.query(
                "SELECT list_transform(tensors, t -> t.array_id) FROM sources "
                "WHERE source_id = 'oz1'"
            )
            .column(0)
            .to_pylist()[0]
        )
        assert rows[0] == "oz1"
        assert "oz1/labels/nuclei" in rows

    def test_read_the_set_and_its_native_level(self, served, client):
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
        # Stated, not left to the client: which of the image's axes each of the
        # set's own indexes. The NGFF block is a spec block and is left alone,
        # so this rides beside it under `biopb`.
        assert meta["biopb"]["labels"]["image_axes"] == [0, 1]

    def test_an_image_is_not_given_a_label_axis_block(self, served, client):
        desc = client.get_descriptor("oz1", with_metadata=True)
        meta = json.loads(desc.metadata_json)["metadata"]
        assert "labels" not in (meta.get("biopb") or {})

        coarse = client.get_tensor(
            "oz1/labels/nuclei", scale_hint=[2, 2], reduction_method="precompute"
        )
        assert coarse[:4, :4].compute().max() == 8


def _sidecar(
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
    return _write_label_group(
        group, levels=1, fill=4, extra=extra, shape=shape, axes=axes
    )


class TestASidecarIsAttachedAtRegistration:
    def test_a_ready_sidecar_is_a_set_a_pending_one_is_not(self, image, tmp_path):
        labels_dir = tmp_path / "labels"
        _sidecar(labels_dir, "oz1", "mine")
        _sidecar(labels_dir, "oz1", "half", state=UPLOAD_PENDING)
        _sidecar(labels_dir, "other", "theirs")

        adapter = SourceRegistry(on_register=sidecar_attacher(labels_dir)).register(
            "oz1", _adapter(image)
        )

        assert set(adapter.label_sets) == NATIVE | {"labels/mine"}
        mine = adapter.resolve_tensor("oz1/labels/mine")
        assert mine.content_version == b"\x01\x02"
        assert mine.get_tensor_metadata()["image-label"]["source"] == {"image": "oz1"}
        assert "biopb" not in mine.get_tensor_metadata()

    def test_one_that_does_not_span_or_bind_is_dropped(self, image, tmp_path):
        labels_dir = tmp_path / "labels"
        _sidecar(labels_dir, "oz1", "small", shape=(32, 32))
        _sidecar(labels_dir, "oz1", "extra", shape=(2, 64, 64), axes=("z", "y", "x"))
        _sidecar(labels_dir, "oz1", "orphan", image_field="Image:9")
        _sidecar(labels_dir, "oz1", "fits")

        adapter = SourceRegistry(on_register=sidecar_attacher(labels_dir)).register(
            "oz1", _adapter(image)
        )

        assert set(adapter.label_sets) == NATIVE | {"labels/fits"}

    def test_a_store_without_a_token_is_corrupt_and_skipped(self, image, tmp_path):
        labels_dir = tmp_path / "labels"
        group = _sidecar(labels_dir, "oz1", "untokened")
        attrs = json.loads((group / ".zattrs").read_text())
        del attrs["biopb"]["labels"]["content_version"]
        (group / ".zattrs").write_text(json.dumps(attrs))

        adapter = SourceRegistry(on_register=sidecar_attacher(labels_dir)).register(
            "oz1", _adapter(image)
        )
        assert "labels/untokened" not in adapter.label_sets

    def test_no_hook_means_no_sidecars(self, image, tmp_path):
        _sidecar(tmp_path / "labels", "oz1", "mine")
        adapter = SourceRegistry().register("oz1", _adapter(image))
        assert "labels/mine" not in adapter.label_sets

    def test_attach_and_detach_by_hand(self, registered, tmp_path):
        group = _sidecar(tmp_path / "labels", "oz1", "late")
        late = open_label_set(
            group, source_id="oz1", image_field="", name="late", content_version=b"x"
        )
        registered.attach_label_set("labels/late", late)
        assert registered.resolve_tensor("labels/late").array_id == "oz1/labels/late"
        assert registered.detach_label_set("labels/late") is not None
        assert registered.detach_label_set("labels/nuclei") is None  # the file's
        with pytest.raises(TensorNotFound):
            registered.resolve_tensor("labels/late")

    def test_the_server_wires_its_write_dir(self, writable_server, image, tmp_path):
        _sidecar(tmp_path / "labels", "oz1", "mine")
        register_and_catalog(writable_server, "oz1", _adapter(image))
        ids = (
            writable_server.metadata_db.query(
                "SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t)"
            )
            .column(0)
            .to_pylist()
        )
        assert "oz1/labels/mine" in ids
