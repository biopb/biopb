"""Label sets are tensors of their image (biopb/biopb#1059 step 2).

The registry answers for them -- ``attached_to(...).label_sets``,
``resolve_tensor`` -- so the format's own listing
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
    sidecar_attrs,
    sidecar_dir,
    sidecar_label_sets,
)
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.adapters.zarr import UPLOAD_PENDING, with_upload_state
from biopb_tensor_server.core.config import PyramidConfig, SourceConfig
from biopb_tensor_server.core.errors import (
    AttachedTensorMismatch,
    TensorNotFound,
    WriteNotSupportedError,
)
from biopb_tensor_server.core.labels import (
    extent_mismatch,
    label_extent,
    label_field,
    split_label_field,
)
from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr
from biopb_tensor_server.sources.source_registry import SourceRegistry

from tests import catalog_server, register_and_catalog

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
def reg(image):
    registry = SourceRegistry()
    registry.register("oz1", _adapter(image))
    return registry


@pytest.fixture
def registered(reg):
    return reg.get("oz1")


NATIVE = {"@labels/nuclei", "@labels/flat", "@labels/xy"}
# Also in the fixture, listed but 32x32 on a 64x64 image: it refuses to be read.
STALE_NATIVE = {"@labels/small"}


class TestTheFieldShape:
    def test_split(self):
        assert split_label_field("@labels/nuclei") == ("", "nuclei", None)
        assert split_label_field("A/1/@labels/nuclei/2") == ("A/1", "nuclei", "2")
        assert split_label_field("@labels/nuclei").set_field == "@labels/nuclei"
        assert split_label_field("labels") is None
        assert split_label_field("A/1") is None
        assert split_label_field(None) is None

    def test_compose(self):
        assert label_field("", "n") == "@labels/n"
        assert label_field("A/1", "n") == "A/1/@labels/n"

    def test_extent_mismatch_says_why(self):
        assert (
            extent_mismatch(["c", "y", "x"], [1, 64, 64], ["c", "y", "x"], [3, 64, 64])
            is None
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
        # The channel axis is a singleton in a set, not the image's length.
        assert "shape" in extent_mismatch(
            ["c", "y", "x"], [3, 64, 64], ["c", "y", "x"], [3, 64, 64]
        )

    def test_a_set_without_the_channel_axis_does_not_span(self):
        # A native group may omit `c` and an older server wrote sidecars without
        # it; neither is the image's rank.
        image = (["t", "c", "z", "y", "x"], [5, 3, 4, 64, 64])
        assert "axes" in extent_mismatch(["t", "z", "y", "x"], [5, 4, 64, 64], *image)

    def test_a_set_has_the_images_rank_with_a_singleton_channel(self):
        image = (["t", "c", "z", "y", "x"], [5, 3, 4, 64, 64])
        assert (
            extent_mismatch(["t", "c", "z", "y", "x"], [5, 1, 4, 64, 64], *image)
            is None
        )

    def test_an_rgb_image_has_a_set_without_the_samples_axis(self):
        # A mask indexes pixels, not a pixel's colour components, and a trailing
        # singleton would shift the spatial axes of a right-aligned viewer.
        rgb = (["y", "x", "s"], [64, 64, 3])
        assert extent_mismatch(["y", "x"], [64, 64], *rgb) is None
        assert "axes" in extent_mismatch(["y", "x", "s"], [64, 64, 3], *rgb)
        assert label_extent(*rgb) == (["y", "x"], [64, 64])
        # The aics shape, a singleton C beside the samples axis.
        aics = (["t", "c", "z", "y", "x", "s"], [2, 1, 4, 64, 64, 3])
        assert label_extent(*aics) == (["t", "c", "z", "y", "x"], [2, 1, 4, 64, 64])
        # Only the size-3/4 axis is samples; another `s` is an ordinary axis.
        assert label_extent(["y", "x", "s"], [64, 64, 5]) == (
            ["y", "x", "s"],
            [64, 64, 5],
        )


class TestANativeSetIsATensorOfItsImage:
    def test_listed_after_the_image_and_readable_by_id(self, registered, reg):
        ids = [t.array_id for t in reg.catalog_tensors("oz1", registered)]
        assert ids[0] == "oz1"
        assert set(ids[1:]) == {f"oz1/{f}" for f in NATIVE | STALE_NATIVE}

        nuclei = reg.resolve_tensor("oz1", "oz1/@labels/nuclei")
        assert isinstance(nuclei, LabelSetAdapter)
        assert nuclei.array_id == "oz1/@labels/nuclei"
        assert reg.resolve_tensor("oz1", "@labels/nuclei") is nuclei
        desc = nuclei.get_tensor_descriptor()
        assert (list(desc.shape), desc.dtype) == ([64, 64], "<u4")

    def test_a_set_that_does_not_span_is_listed_and_refuses_to_be_read(
        self, registered, reg
    ):
        """``bad`` is a float (the reader skips it); ``small`` is 32x32 on a
        64x64 image: listed, but a read says why instead of serving it
        misaligned. The image registers either way."""
        sets = reg.attached_to("oz1").label_sets(registered)
        assert set(sets) == NATIVE | STALE_NATIVE
        with pytest.raises(AttachedTensorMismatch, match="does not span"):
            reg.resolve_tensor("oz1", "@labels/small")
        assert reg.resolve_tensor("oz1", "oz1").array_id == "oz1"

    def test_an_unknown_set_is_the_formats_miss(self, registered, reg):
        with pytest.raises(TensorNotFound):
            reg.resolve_tensor("oz1", "@labels/nope")

    def test_the_formats_own_routing_is_untouched(self, image):
        raw = _adapter(image)
        reg = SourceRegistry()
        reg.register("oz1", raw)
        assert [t.array_id for t in raw.list_tensors()] == ["oz1"]
        assert reg.resolve_tensor("oz1", None).array_id == "oz1"
        assert reg.resolve_tensor("oz1", "1").array_id == "oz1/1"

    def test_a_native_level_routes_under_the_set(self, registered, reg):
        """``nuclei``'s arrays carry their own ``.zattrs``, which used to stop
        the level-opening walk one directory too early."""
        nuclei = reg.resolve_tensor("oz1", "@labels/nuclei")

        level = reg.resolve_tensor("oz1", "@labels/nuclei/1")
        assert level.array_id == "oz1/@labels/nuclei/1"
        assert list(level.get_tensor_descriptor().shape) == [32, 32]
        assert level.content_version == nuclei.content_version
        assert reg.resolve_tensor("oz1", "@labels/nuclei") is nuclei

    def test_content_version_is_the_images_for_set_and_levels_alike(
        self, registered, reg
    ):
        nuclei = reg.resolve_tensor("oz1", "@labels/nuclei")
        assert nuclei.content_version == registered.content_version is not None
        # The image's own native levels share it too: one token per source.
        assert (
            reg.resolve_tensor("oz1", "1").content_version == registered.content_version
        )

    def test_the_ladder_is_native_or_nearest_never_area(self, registered, reg):
        cfg = PyramidConfig(reduction_method="area")
        nuclei = reg.resolve_tensor("oz1", "@labels/nuclei")
        native = nuclei._advertised_pyramid(nuclei.get_tensor_descriptor(), cfg)
        assert all(
            lvl.reduction_method == "precompute" and lvl.native for lvl in native
        )

        flat = reg.resolve_tensor("oz1", "@labels/flat")
        computed = flat._advertised_pyramid(flat.get_tensor_descriptor(), cfg)
        assert computed and all(lvl.reduction_method == "nearest" for lvl in computed)

    def test_metadata_names_the_image(self, registered, reg):
        meta = reg.resolve_tensor("oz1", "@labels/nuclei").get_tensor_metadata()
        assert meta["image-label"]["source"] == {"image": "oz1"}
        assert meta["image-label"]["colors"][0]["label-value"] == 7
        assert meta["multiscales"][0]["datasets"][1]["path"] == "1"

    def test_a_set_is_read_only(self, registered, reg):
        with pytest.raises(WriteNotSupportedError):
            reg.resolve_tensor("oz1", "@labels/nuclei").put_chunk(None, None, (), None)

    def test_a_non_canonical_set_is_normalized_like_any_tensor(self, registered, reg):
        xy = reg.resolve_tensor("oz1", "@labels/xy")
        assert xy._axis_perm() is not None
        assert list(xy.get_tensor_descriptor().dim_labels) == ["y", "x"]
        assert [
            t.dim_labels[0]
            for t in reg.catalog_tensors("oz1", registered)
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
        assert "oz1/@labels/nuclei" in rows

    def test_read_the_set_and_its_native_level(self, served, client):
        full = client.get_tensor("oz1/@labels/nuclei")
        assert full.shape == SHAPE and full.dtype == np.uint32
        assert full[:8, :8].compute().tolist() == np.full((8, 8), 7).tolist()

        desc = client.get_descriptor(
            "oz1/@labels/nuclei", with_metadata=True, with_pyramid=True
        )
        assert [lvl.reduction_method for lvl in desc.pyramid] == [
            "precompute",
            "precompute",
        ]
        meta = json.loads(desc.metadata_json)["metadata"]
        assert meta["image-label"]["source"] == {"image": "oz1"}

    def test_a_computed_level_is_read_with_the_precompute_method(self, served, client):
        coarse = client.get_tensor(
            "oz1/@labels/nuclei", scale_hint=[2, 2], reduction_method="precompute"
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


def _adopting(labels_dir):
    """A registry that took the finished sidecars of ``oz1`` at boot."""
    registry = SourceRegistry()
    registry.adopt({"oz1": sidecar_label_sets("oz1", labels_dir)})
    return registry


class TestASidecarIsAttachedAtRegistration:
    def test_a_ready_sidecar_is_a_set_a_pending_one_is_not(self, image, tmp_path):
        labels_dir = tmp_path / "labels"
        _sidecar(labels_dir, "oz1", "mine")
        _sidecar(labels_dir, "oz1", "half", state=UPLOAD_PENDING)
        _sidecar(labels_dir, "other", "theirs")

        reg = _adopting(labels_dir)
        adapter = reg.register("oz1", _adapter(image))

        assert set(
            reg.attached_to("oz1").label_sets(adapter)
        ) == NATIVE | STALE_NATIVE | {"@labels/mine"}
        mine = reg.resolve_tensor("oz1", "oz1/@labels/mine")
        assert mine.content_version == b"\x01\x02"
        assert mine.get_tensor_metadata()["image-label"]["source"] == {"image": "oz1"}
        assert "biopb" not in mine.get_tensor_metadata()

    def test_one_that_does_not_span_or_bind_is_listed_and_refuses_to_be_read(
        self, image, tmp_path
    ):
        labels_dir = tmp_path / "labels"
        _sidecar(labels_dir, "oz1", "small", shape=(32, 32))
        _sidecar(labels_dir, "oz1", "extra", shape=(2, 64, 64), axes=("z", "y", "x"))
        _sidecar(labels_dir, "oz1", "orphan", image_field="Image:9")
        _sidecar(labels_dir, "oz1", "fits")

        reg = _adopting(labels_dir)
        adapter = reg.register("oz1", _adapter(image))

        stale = {"@labels/small", "@labels/extra", "Image:9/@labels/orphan"}
        assert set(reg.attached_to("oz1").label_sets(adapter)) == (
            NATIVE | {"@labels/fits"} | stale
        )
        for name in sorted(stale):
            with pytest.raises(AttachedTensorMismatch):
                reg.resolve_tensor("oz1", name)
        assert reg.resolve_tensor("oz1", "@labels/fits") is not None

    def test_a_store_without_a_token_is_corrupt_and_skipped(self, image, tmp_path):
        labels_dir = tmp_path / "labels"
        group = _sidecar(labels_dir, "oz1", "untokened")
        attrs = json.loads((group / ".zattrs").read_text())
        del attrs["biopb"]["labels"]["content_version"]
        (group / ".zattrs").write_text(json.dumps(attrs))

        reg = _adopting(labels_dir)
        adapter = reg.register("oz1", _adapter(image))
        assert "@labels/untokened" not in reg.attached_to("oz1").label_sets(adapter)

    def test_nothing_adopted_means_no_sidecars(self, image, tmp_path):
        _sidecar(tmp_path / "labels", "oz1", "mine")
        reg = SourceRegistry()
        adapter = reg.register("oz1", _adapter(image))
        assert "@labels/mine" not in reg.attached_to("oz1").label_sets(adapter)

    def test_attach_and_detach_by_hand(self, reg, tmp_path):
        group = _sidecar(tmp_path / "labels", "oz1", "late")
        late = open_label_set(
            group, source_id="oz1", image_field="", name="late", content_version=b"x"
        )
        reg.attach("oz1", "@labels/late", late)
        assert reg.resolve_tensor("oz1", "@labels/late").array_id == "oz1/@labels/late"
        assert reg.detach("oz1", "@labels/late") is not None
        assert reg.detach("oz1", "@labels/nuclei") is None  # the file's
        with pytest.raises(TensorNotFound):
            reg.resolve_tensor("oz1", "@labels/late")

    def test_the_server_adopts_its_write_dir_at_boot(self, image, tmp_path):
        _sidecar(tmp_path / "labels", "oz1", "mine")
        server = catalog_server("localhost:0", writable=True, write_dir=tmp_path)
        register_and_catalog(server, "oz1", _adapter(image))
        ids = (
            server.metadata_db.query(
                "SELECT t.array_id FROM sources, UNNEST(tensors) AS u(t)"
            )
            .column(0)
            .to_pylist()
        )
        assert "oz1/@labels/mine" in ids


class TestOwningField:
    """Attached fields route by longest prefix, so a level rides with its tensor."""

    def test_the_longest_key_wins_and_a_prefix_must_end_at_a_slash(self):
        from biopb_tensor_server.core.attachments import _owning_field

        keys = ["@fields/raw", "@fields/raw/@labels/nuclei", "@fields/ra"]
        assert _owning_field("@fields/raw", keys) == "@fields/raw"
        assert _owning_field("@fields/raw/@labels/nuclei/1", keys) == keys[1]
        assert _owning_field("@fields/raw/2", keys) == "@fields/raw"
        assert _owning_field("@fields/rawx", keys) is None
