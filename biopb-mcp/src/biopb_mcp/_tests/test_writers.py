"""GUI-free tests for the OME-Zarr napari writers (``biopb_mcp._writers``).

No napari/Qt: the writers are called exactly as napari calls a single-layer
writer -- ``func(path, data, meta) -> [path]`` -- and the output is reopened with
``zarr`` and asserted to be a valid OME-Zarr that round-trips.
"""

import numpy as np
import zarr

from biopb_mcp._writers import (
    _axis_dict,
    _default_dim_labels,
    write_image_ome_zarr,
    write_labels_ome_zarr,
)


def test_axis_typing_matches_server_convention():
    assert _axis_dict("x") == {"name": "x", "type": "space"}
    assert _axis_dict("Z") == {"name": "Z", "type": "space"}
    assert _axis_dict("C") == {"name": "C", "type": "channel"}
    assert _axis_dict("t") == {"name": "t", "type": "time"}
    assert _axis_dict("foo") == {"name": "foo"}


def test_default_dim_labels_canonical_zyx():
    assert _default_dim_labels(2) == ["y", "x"]
    assert _default_dim_labels(3) == ["z", "y", "x"]
    assert _default_dim_labels(4) == ["c", "z", "y", "x"]
    assert _default_dim_labels(5) == ["c", "t", "z", "y", "x"]


def _read_multiscale(path):
    g = zarr.open_group(str(path), mode="r")
    ms = g.attrs["multiscales"][0]
    lvl0 = g[ms["datasets"][0]["path"]]
    return ms, lvl0


def test_write_image_roundtrips(tmp_path):
    arr = (np.random.rand(3, 16, 24) * 255).astype("uint8")  # z, y, x
    out = tmp_path / "img.ome.zarr"

    paths = write_image_ome_zarr(str(out), arr, {"name": "im", "metadata": {}})
    assert paths == [str(out)]

    ms, lvl0 = _read_multiscale(out)
    # single resolution + axis names/types round-trip (server convention).
    assert [d["path"] for d in ms["datasets"]] == ["0"]
    assert [a["name"] for a in ms["axes"]] == ["z", "y", "x"]
    assert all(a["type"] == "space" for a in ms["axes"])
    assert lvl0.shape == arr.shape
    assert lvl0.dtype == arr.dtype
    np.testing.assert_array_equal(lvl0[:], arr)


def test_write_image_honors_scale_transform(tmp_path):
    arr = np.zeros((4, 8), dtype="uint16")  # y, x
    out = tmp_path / "scaled.ome.zarr"

    write_image_ome_zarr(str(out), arr, {"scale": [0.5, 0.25], "metadata": {}})

    ms, _ = _read_multiscale(out)
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [0.5, 0.25]


def test_write_image_honors_explicit_dim_labels(tmp_path):
    arr = np.zeros((2, 5, 6), dtype="uint8")  # c, y, x
    out = tmp_path / "labeled_axes.ome.zarr"

    write_image_ome_zarr(
        str(out), arr, {"metadata": {"dim_labels": ["c", "y", "x"]}}
    )

    ms, _ = _read_multiscale(out)
    assert [a["name"] for a in ms["axes"]] == ["c", "y", "x"]
    assert ms["axes"][0] == {"name": "c", "type": "channel"}


def test_write_labels_roundtrips_dtype_exact(tmp_path):
    mask = np.zeros((8, 8), dtype="int32")
    mask[2:5, 2:5] = 7
    out = tmp_path / "seg.ome.zarr"

    paths = write_labels_ome_zarr(str(out), mask, {"name": "seg", "metadata": {}})
    assert paths == [str(out)]

    _, lvl0 = _read_multiscale(out)
    assert lvl0.dtype == mask.dtype
    np.testing.assert_array_equal(lvl0[:], mask)


# --- multiscale ------------------------------------------------------------
# A real napari multiscale layer passes ``data`` as a
# ``napari.layers._multiscale_data.MultiScaleData`` (a ``Sequence``, NOT a list)
# and sets ``meta['multiscale'] = True``. The writer keys off that flag, so these
# GUI-free tests pass a plain list of arrays (which ``list()`` handles identically)
# together with ``meta={'multiscale': True, ...}``.


def _scales(ms):
    """The per-level coordinateTransformations scale vectors, level 0 first."""
    return [d["coordinateTransformations"][0]["scale"] for d in ms["datasets"]]


def test_write_multiscale_image_persists_all_levels(tmp_path):
    levels = [
        np.arange(16 * 24, dtype="uint8").reshape(16, 24),  # y, x
        np.zeros((8, 12), dtype="uint8"),
    ]
    out = tmp_path / "ms.ome.zarr"

    paths = write_image_ome_zarr(str(out), levels, {"multiscale": True, "metadata": {}})
    assert paths == [str(out)]

    g = zarr.open_group(str(out), mode="r")
    ms = g.attrs["multiscales"][0]
    assert [d["path"] for d in ms["datasets"]] == ["0", "1"]
    assert g["0"].shape == (16, 24)
    assert g["1"].shape == (8, 12)
    # factor derived from shape ratios: level0 [1,1], level1 [2,2].
    assert _scales(ms) == [[1.0, 1.0], [2.0, 2.0]]
    np.testing.assert_array_equal(g["0"][:], levels[0])


def test_write_multiscale_honors_physical_scale(tmp_path):
    levels = [np.zeros((16, 24), dtype="uint8"), np.zeros((8, 12), dtype="uint8")]
    out = tmp_path / "ms_scaled.ome.zarr"

    write_image_ome_zarr(
        str(out), levels, {"multiscale": True, "scale": [0.5, 0.25], "metadata": {}}
    )

    g = zarr.open_group(str(out), mode="r")
    ms = g.attrs["multiscales"][0]
    # level 0 == physical size; deeper levels == physical size * downsample factor.
    assert _scales(ms) == [[0.5, 0.25], [1.0, 0.5]]


def test_write_multiscale_labels_dtype_exact_each_level(tmp_path):
    l0 = np.zeros((8, 8), dtype="int32")
    l0[2:6, 2:6] = 9
    l1 = np.zeros((4, 4), dtype="int32")
    l1[1:3, 1:3] = 9
    out = tmp_path / "ms_seg.ome.zarr"

    write_labels_ome_zarr(str(out), [l0, l1], {"multiscale": True, "metadata": {}})

    g = zarr.open_group(str(out), mode="r")
    assert g["0"].dtype == l0.dtype and g["1"].dtype == l1.dtype
    np.testing.assert_array_equal(g["0"][:], l0)
    np.testing.assert_array_equal(g["1"][:], l1)


def test_multiscale_flag_with_single_level_stays_single(tmp_path):
    arr = np.zeros((8, 8), dtype="uint8")
    out = tmp_path / "ms_one.ome.zarr"

    write_image_ome_zarr(str(out), [arr], {"multiscale": True, "metadata": {}})

    ms, lvl0 = _read_multiscale(out)
    assert [d["path"] for d in ms["datasets"]] == ["0"]
    assert lvl0.shape == arr.shape
