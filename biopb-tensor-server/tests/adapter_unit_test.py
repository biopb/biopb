"""Server-side unit tests for adapters, config, and downsampling.

Tests for adapter behavior, configuration parsing, reduction-method
normalisation, and scaled read planning.
"""

import os
import tempfile

import numpy as np
import pytest
from biopb.tensor import (
    SliceHint,
    TensorDescriptor,
)
from biopb.tensor.ticket_pb2 import ChunkBounds
from biopb_tensor_server import (
    OmeZarrAdapter,
    ZarrAdapter,
    downsample as _ds,
)
from biopb_tensor_server.config import parse_config


def _zarr_available() -> bool:
    """Check if zarr is available with working numcodecs."""
    try:
        import zarr

        zarr.open_array
        return True
    except ImportError:
        return False


class TestZarrAdapter:
    """Tests for ZarrAdapter."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_tensor_descriptor(self):
        """Test descriptor generation."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 200), chunks=(50, 100), dtype="uint16"
            )

            adapter = ZarrAdapter(arr, "test-array", ["y", "x"])
            desc = adapter.get_tensor_descriptor()

            assert desc.array_id == "test-array"
            assert list(desc.shape) == [100, 200]
            assert list(desc.chunk_shape) == [50, 100]
            assert desc.dtype == "<u2"
            assert list(desc.dim_labels) == ["y", "x"]


class TestTensorConfig:
    """Tests for tensor server config parsing."""

    def test_legacy_compute_section_ignored_with_warning(self, caplog):
        """The removed [compute] section is warn-and-ignore, never an error."""
        import logging

        with caplog.at_level(logging.WARNING, logger="biopb_tensor_server.config"):
            config = parse_config(
                {
                    "server": {
                        "host": "127.0.0.1",
                        "port": 9000,
                    },
                    "compute": {
                        "backend": "gpu",
                        "gpu_min_input_mb": 8,
                    },
                    "sources": [],
                }
            )

        assert not hasattr(config, "compute_backend")
        assert not hasattr(config, "gpu_min_input_mb")
        assert any(
            "deprecated" in rec.getMessage() and "compute" in rec.getMessage()
            for rec in caplog.records
        )

    def test_parse_periodic_monitor_settings(self):
        config = parse_config(
            {
                "server": {
                    "monitor_mode": "periodic",
                    "rescan_interval": 12,
                    "full_rescan_interval": 120,
                    "stability_window": 45,
                    "stable_rescans_required": 2,
                    "probe_open_files": False,
                    "aggressive_dir_pruning": True,
                },
                "sources": [],
            }
        )

        assert config.monitor_mode == "periodic"
        assert config.rescan_interval == 12.0
        assert config.full_rescan_interval == 120.0
        assert config.stability_window == 45.0
        assert config.stable_rescans_required == 2
        assert config.probe_open_files is False
        assert config.aggressive_dir_pruning is True

    def test_parse_legacy_monitor_aliases(self):
        config = parse_config(
            {
                "server": {
                    "watcher_type": "off",
                    "poll_interval": 9,
                },
                "sources": [],
            }
        )

        assert config.monitor_mode == "off"
        assert config.rescan_interval == 9.0
        assert config.full_rescan_interval == 3600.0
        assert config.stability_window == 30.0
        assert config.stable_rescans_required == 0
        assert config.probe_open_files is True
        assert config.aggressive_dir_pruning is False

    def test_claim_generic_images_defaults_off(self):
        config = parse_config({"server": {}, "sources": []})
        assert config.claim_generic_images is False

    def test_parse_claim_generic_images(self):
        config = parse_config(
            {
                "server": {"claim_generic_images": True},
                "sources": [],
            }
        )
        assert config.claim_generic_images is True


class TestReductionMethodNormalization:
    """Tests for normalize_reduction_method and the deprecated linear alias."""

    def test_supported_methods_and_aliases(self):
        assert _ds.normalize_reduction_method("nearest") == "nearest"
        assert _ds.normalize_reduction_method("area") == "area"
        assert _ds.normalize_reduction_method("precompute") == "precompute"
        assert _ds.normalize_reduction_method("stride") == "nearest"
        assert _ds.normalize_reduction_method("decimate") == "nearest"
        assert _ds.normalize_reduction_method("mean") == "area"
        assert _ds.normalize_reduction_method("precomputed") == "precompute"

    def test_default_is_area(self):
        """Unspecified method resolves to area, matching PyramidConfig and
        the advertised pyramid levels (descriptor consistency, biopb#76)."""
        assert _ds.normalize_reduction_method("") == "area"
        assert _ds.normalize_reduction_method(None) == "area"

    def test_linear_aliases_to_area_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="biopb_tensor_server.downsample"):
            assert _ds.normalize_reduction_method("linear") == "area"
        assert any("deprecated" in rec.getMessage() for rec in caplog.records)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported reduction method"):
            _ds.normalize_reduction_method("cubic")

    def test_pyramid_config_accepts_linear_alias(self):
        from biopb_tensor_server.config import PyramidConfig

        # Tolerated deprecated alias: old configs must keep validating.
        PyramidConfig(reduction_method="linear")


class TestAdvisoryReductionCacheKey:
    """reduction_method is advisory: excluded from the cache key (biopb#76)."""

    def test_cache_key_strips_method(self):
        from biopb_tensor_server.chunk import (
            cache_key_for_chunk_id,
            encode_chunk_id,
            encode_chunk_id_with_scale,
        )

        bounds = ChunkBounds(start=[0, 0], stop=[64, 64])
        nearest_id = encode_chunk_id_with_scale("arr", bounds, (4, 4), "nearest")
        area_id = encode_chunk_id_with_scale("arr", bounds, (4, 4), "area")
        other_scale = encode_chunk_id_with_scale("arr", bounds, (2, 2), "area")

        assert cache_key_for_chunk_id(nearest_id) == cache_key_for_chunk_id(area_id)
        assert cache_key_for_chunk_id(nearest_id) != cache_key_for_chunk_id(other_scale)

        raw_id = encode_chunk_id("arr", bounds)
        assert cache_key_for_chunk_id(raw_id) == raw_id

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_cache_hit_across_reduction_methods(self):
        """A chunk warmed under one method serves a request for another."""
        import zarr
        from biopb_tensor_server.cache import CacheManager
        from biopb_tensor_server.chunk import encode_chunk_id_with_scale
        from biopb_tensor_server.config import CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(64, 64), chunks=(64, 64), dtype="uint16"
            )
            arr[:] = np.arange(64 * 64, dtype="uint16").reshape(64, 64)

            adapter = ZarrAdapter(arr, "test-array", ["y", "x"])
            cache_manager = CacheManager(CacheConfig(backend="memory"))

            bounds = ChunkBounds(start=[0, 0], stop=[64, 64])
            area_id = encode_chunk_id_with_scale("test-array", bounds, (4, 4), "area")
            nearest_id = encode_chunk_id_with_scale(
                "test-array", bounds, (4, 4), "nearest"
            )

            first = adapter.resolve_chunk_data(area_id, cache_manager)
            stats = cache_manager.stats()
            assert stats.misses == 1

            second = adapter.resolve_chunk_data(nearest_id, cache_manager)
            stats = cache_manager.stats()
            assert stats.misses == 1
            assert stats.hits == 1

            # The advisory method means the warmed (area) data is served.
            assert first.column("data").to_pylist() == second.column("data").to_pylist()


class TestGetScaledReadPlan:
    """Tests for BackendAdapter.get_scaled_read_plan default implementation."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_default_uses_virtual_scaling(self):
        """Test that default get_scaled_read_plan uses virtual scaling."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )
            arr[:] = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "test", ["y", "x"]
            )

            # Test that default implementation produces virtual scaling plan
            request_desc = TensorDescriptor(
                array_id="test",
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype="uint8",
                scale_hint=[2, 2],
                reduction_method="nearest",
            )
            plan = adapter.get_read_plan(request_desc)

            # Should have virtual chunks (2x2 grid = 4 chunks)
            assert len(plan.chunk_endpoints) == 4
            assert list(plan.descriptor.shape) == [50, 50]
            assert list(plan.descriptor.chunk_shape) == [25, 25]  # Virtual chunk shape

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_virtual_scaling_with_slice(self):
        """Test virtual scaling with slice hint."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )
            arr[:] = 1

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "test", ["y", "x"]
            )

            request_desc = TensorDescriptor(
                array_id="test",
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype="uint8",
                slice_hint=SliceHint(start=[10, 10], stop=[50, 50]),
                scale_hint=[2, 2],
                reduction_method="nearest",
            )
            plan = adapter.get_read_plan(request_desc)

            # Slice [10,10]->[50,50] intersects chunk [0,50]x[0,50].
            # Realized (snapped) source bounds = [0,0]->[50,50] -> shape (50/2, 50/2) = (25, 25)
            assert list(plan.descriptor.shape) == [25, 25]


class TestEmptyChunkShapeFallback:
    """get_chunk_size() must tolerate an empty/partial chunk_shape.

    A descriptor may legitimately omit chunk_shape (documented as "can be empty
    []"; the lean ListFlights form drops it, and the bulk-seeded remote proxy
    mirrors that lean catalog -- biopb/biopb#266). Before the fix, get_chunk_size
    returned a too-short tuple and _get_read_plan indexed it out of range against
    the full-rank shape, so every read of such a source raised IndexError.
    """

    from biopb_tensor_server.base import TensorAdapter

    class _StubTensorAdapter(TensorAdapter):
        """Minimal tensor adapter whose descriptor carries no chunk_shape."""

        def __init__(self, shape, dtype, dim_labels, chunk_shape=None):
            self._desc = TensorDescriptor(
                array_id="stub",
                shape=list(shape),
                chunk_shape=list(chunk_shape or []),
                dtype=dtype,
                dim_labels=list(dim_labels),
            )

        def get_tensor_descriptor(self):
            return self._desc

        def get_data(self, bounds):  # pragma: no cover - not exercised here
            raise NotImplementedError

    def test_empty_chunk_shape_derives_default_grid(self):
        # The reproducer: a 5-D FITS-like tensor (>i2) with no chunk_shape, as a
        # remote-proxy source seeded from the lean upstream catalog.
        shape = [1, 1, 1000, 512, 512]
        adapter = self._StubTensorAdapter(shape, ">i2", ["T", "C", "Z", "Y", "X"])

        from biopb_tensor_server.chunk import compute_safe_chunk_size

        chunk = adapter.get_chunk_size()
        # A full-rank grid (no longer a 0-length tuple).
        assert len(chunk) == len(shape)
        # Matches the server's default transfer-grid policy exactly.
        assert chunk == compute_safe_chunk_size(
            tuple(shape), ">i2", ["T", "C", "Z", "Y", "X"]
        )
        # Y-X plane kept whole; a non-spatial axis was split to fit the cap.
        assert chunk[-2:] == (512, 512)
        assert chunk[2] < 1000

    def test_read_plan_no_longer_raises_on_empty_chunk_shape(self):
        shape = [1, 1, 1000, 512, 512]
        adapter = self._StubTensorAdapter(shape, ">i2", ["T", "C", "Z", "Y", "X"])
        request_desc = TensorDescriptor(array_id="stub", shape=shape, dtype=">i2")
        plan = adapter.get_read_plan(request_desc)  # was IndexError
        assert plan.chunk_endpoints
        assert list(plan.descriptor.shape) == shape

    def test_normal_chunk_shape_is_untouched(self):
        adapter = self._StubTensorAdapter(
            [100, 100], "uint8", ["y", "x"], chunk_shape=[50, 50]
        )
        assert adapter.get_chunk_size() == (50, 50)


class TestGetPhysicalScale:
    """Per-dim physical-scale summary folded onto the descriptor (issue #31)."""

    def _make_ome_zarr(self, tmpdir, zattrs, shape=(100, 100), chunks=(50, 50)):
        import json

        import zarr

        zarr_path = os.path.join(tmpdir, "test.ome.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_dataset("0", shape=shape, chunks=chunks, dtype="uint8")
        with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
            json.dump(zattrs, f)
        root = zarr.open_group(zarr_path, mode="r")
        return OmeZarrAdapter(root["0"], "test")

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_ome_zarr_physical_scale_from_units(self):
        """Axes with units yield physical sizes aligned to dim order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zattrs = {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [0.325, 0.325]}
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [0.65, 0.65]}
                                ],
                            },
                        ],
                    }
                ]
            }
            adapter = self._make_ome_zarr(tmpdir, zattrs)
            scale, unit = adapter.get_physical_scale()
            assert scale == [0.325, 0.325]
            assert unit == ["micrometer", "micrometer"]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_ome_zarr_global_scale_composes(self):
        """A multiscales-level scale transform composes with the level-0 scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zattrs = {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [0.5, 0.5]}
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                        ],
                    }
                ]
            }
            adapter = self._make_ome_zarr(tmpdir, zattrs)
            scale, unit = adapter.get_physical_scale()
            assert scale == [0.5, 0.5]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_ome_zarr_no_unit_returns_none(self):
        """Unit-less (relative) scales carry no physical info -> None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zattrs = {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                        ],
                    }
                ]
            }
            adapter = self._make_ome_zarr(tmpdir, zattrs)
            assert adapter.get_physical_scale() is None

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_ome_zarr_channel_axis_zeroed(self):
        """A channel axis (no unit) gets 0.0/'' while space axes carry sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zattrs = {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 0.5, 0.5]}
                                ],
                            },
                        ],
                    }
                ]
            }
            adapter = self._make_ome_zarr(
                tmpdir, zattrs, shape=(2, 100, 100), chunks=(1, 50, 50)
            )
            scale, unit = adapter.get_physical_scale()
            assert scale == [0.0, 0.5, 0.5]
            assert unit == ["", "micrometer", "micrometer"]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_adapter_has_no_physical_scale(self):
        """A plain ZarrAdapter inherits the base None (no physical sizes)."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "plain.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(64, 64), chunks=(32, 32), dtype="uint8"
            )
            adapter = ZarrAdapter(arr, "plain", ["y", "x"])
            assert adapter.get_physical_scale() is None

    def _make_aics_adapter(self, dim_labels, scene_index, scenes, images):
        """Build an AicsImageIoAdapterBase without __init__ (no real file)."""
        from unittest.mock import MagicMock

        from biopb_tensor_server.adapters.aicsimageio import (
            _AicsImageIoAdapterBase,
        )

        a = _AicsImageIoAdapterBase.__new__(_AicsImageIoAdapterBase)
        a.dim_labels = dim_labels
        a.scene_index = scene_index
        a._aics_image = MagicMock()
        a._aics_image.scenes = scenes
        a._aics_image.ome_metadata.images = images
        return a

    @staticmethod
    def _ome_pixels(psx, psy, psz, unit="µm"):
        from unittest.mock import MagicMock

        u = MagicMock()
        u.value = unit
        px = MagicMock()
        px.physical_size_x, px.physical_size_y, px.physical_size_z = psx, psy, psz
        px.physical_size_x_unit = px.physical_size_y_unit = px.physical_size_z_unit = u
        img = MagicMock()
        img.pixels = px
        return img

    def test_aics_physical_scale_maps_by_label_and_zeros_tc(self):
        """Aics OME pixel sizes map onto dim_labels; T/C get 0.0/''."""
        img = self._ome_pixels(0.325, 0.325, 2.0)
        adapter = self._make_aics_adapter(
            ["T", "C", "Z", "Y", "X"], scene_index=0, scenes=["s0"], images=[img]
        )
        scale, unit = adapter.get_physical_scale()
        assert scale == [0.0, 0.0, 2.0, 0.325, 0.325]
        assert unit == ["", "", "µm", "µm", "µm"]

    def test_aics_physical_scale_reads_bound_scene(self):
        """A scene-bound tensor adapter reads the right OME image by its index.

        Scene selection is via the bound ``scene_index`` (set by
        ``get_tensor_adapter``), not a ``tensor_id`` argument -- the method is
        tensor-scoped on ``TensorAdapter``.
        """
        img0 = self._ome_pixels(1.0, 1.0, 0.0)
        img1 = self._ome_pixels(0.5, 0.5, 0.0)
        adapter = self._make_aics_adapter(
            ["C", "Y", "X"],
            scene_index=1,
            scenes=["s0", "s1"],
            images=[img0, img1],
        )
        scale, _ = adapter.get_physical_scale()
        assert scale == [0.0, 0.5, 0.5]

    def test_aics_physical_scale_none_when_no_sizes(self):
        """No positive physical size anywhere -> None."""
        img = self._ome_pixels(None, None, None)
        adapter = self._make_aics_adapter(
            ["Y", "X"], scene_index=0, scenes=["s0"], images=[img]
        )
        assert adapter.get_physical_scale() is None


class TestOmeZarrPrecompute:
    """Tests for OmeZarrAdapter precomputed level support."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_find_level_for_scale(self):
        """Test _find_level_for_scale parses multiscales correctly."""
        import json

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create OME-Zarr structure with multiscales
            zarr_path = os.path.join(tmpdir, "test.ome.zarr")
            root = zarr.open_group(zarr_path, mode="w")

            # Create level arrays
            root.create_dataset("0", shape=(100, 100), chunks=(50, 50), dtype="uint8")
            root.create_dataset("1", shape=(50, 50), chunks=(25, 25), dtype="uint8")
            root.create_dataset("2", shape=(25, 25), chunks=(12, 12), dtype="uint8")

            # Create .zattrs with multiscales
            zattrs = {
                "multiscales": [
                    {
                        "name": "test",
                        "axes": [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2, 2]}
                                ],
                            },
                            {
                                "path": "2",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [4, 4]}
                                ],
                            },
                        ],
                    }
                ]
            }
            with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
                json.dump(zattrs, f)

            # Open base level and create adapter
            root = zarr.open_group(zarr_path, mode="r")
            base_arr = root["0"]
            adapter = OmeZarrAdapter(base_arr, "test")

            # Test finding levels
            assert adapter._find_level_for_scale((1, 1)) == "0"
            assert adapter._find_level_for_scale((2, 2)) == "1"
            assert adapter._find_level_for_scale((4, 4)) == "2"
            assert adapter._find_level_for_scale((3, 3)) is None  # No match

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_get_scaled_read_plan_precompute(self):
        """Test get_scaled_read_plan with precompute method."""
        import json

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.ome.zarr")
            root = zarr.open_group(zarr_path, mode="w")

            # Create and populate level arrays
            arr0 = root.create_dataset(
                "0", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )
            arr1 = root.create_dataset(
                "1", shape=(50, 50), chunks=(25, 25), dtype="uint8"
            )
            arr0[:] = 1
            arr1[:] = 2

            zattrs = {
                "multiscales": [
                    {
                        "name": "test",
                        "axes": [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2, 2]}
                                ],
                            },
                        ],
                    }
                ]
            }
            with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
                json.dump(zattrs, f)

            root = zarr.open_group(zarr_path, mode="r")
            base_arr = root["0"]
            adapter = OmeZarrAdapter(base_arr, "test")

            # Request with precompute method
            request_desc = TensorDescriptor(
                array_id="test",
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype="uint8",
                scale_hint=[2, 2],
                reduction_method="precompute",
            )
            plan = adapter.get_read_plan(request_desc)

            # Should return level 1's shape
            assert list(plan.descriptor.shape) == [50, 50]
            assert list(plan.descriptor.chunk_shape) == [25, 25]

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_precompute_no_match_raises(self):
        """Test that precompute raises error when no matching level."""
        import json

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.ome.zarr")
            root = zarr.open_group(zarr_path, mode="w")

            root.create_dataset("0", shape=(100, 100), chunks=(50, 50), dtype="uint8")
            root.create_dataset("1", shape=(50, 50), chunks=(25, 25), dtype="uint8")

            zattrs = {
                "multiscales": [
                    {
                        "name": "test",
                        "axes": [{"name": "y"}, {"name": "x"}],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2, 2]}
                                ],
                            },
                        ],
                    }
                ]
            }
            with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
                json.dump(zattrs, f)

            root = zarr.open_group(zarr_path, mode="r")
            base_arr = root["0"]
            adapter = OmeZarrAdapter(base_arr, "test")

            # Request with non-matching scale
            request_desc = TensorDescriptor(
                array_id="test",
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype="uint8",
                scale_hint=[3, 3],
                reduction_method="precompute",
            )
            with pytest.raises(ValueError, match="No precomputed level matching"):
                adapter.get_read_plan(request_desc)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_precompute_falls_back_to_virtual_for_other_methods(self):
        """Test that non-precompute methods use virtual scaling."""
        import json

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.ome.zarr")
            root = zarr.open_group(zarr_path, mode="w")

            root.create_dataset("0", shape=(100, 100), chunks=(50, 50), dtype="uint8")
            root.create_dataset("1", shape=(50, 50), chunks=(25, 25), dtype="uint8")

            zattrs = {
                "multiscales": [
                    {
                        "name": "test",
                        "axes": [{"name": "y"}, {"name": "x"}],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [2, 2]}
                                ],
                            },
                        ],
                    }
                ]
            }
            with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
                json.dump(zattrs, f)

            root = zarr.open_group(zarr_path, mode="r")
            base_arr = root["0"]
            adapter = OmeZarrAdapter(base_arr, "test")

            # Request with nearest method (not precompute)
            request_desc = TensorDescriptor(
                array_id="test",
                shape=[100, 100],
                chunk_shape=[50, 50],
                dtype="uint8",
                scale_hint=[2, 2],
                reduction_method="nearest",
            )
            plan = adapter.get_read_plan(request_desc)

            # Should use virtual scaling (shape based on base / scale)
            assert list(plan.descriptor.shape) == [50, 50]
            # Virtual chunk shape is base_chunk / scale = 50/2 = 25
            assert list(plan.descriptor.chunk_shape) == [25, 25]


class TestSliceConversion:
    """Tests for slice coordinate conversion."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_convert_slice_to_level(self):
        """Test slice conversion from base to level coordinates."""
        import json

        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.ome.zarr")
            root = zarr.open_group(zarr_path, mode="w")

            root.create_dataset("0", shape=(100, 100), chunks=(50, 50), dtype="uint8")
            root.create_dataset("1", shape=(25, 50), chunks=(12, 25), dtype="uint8")

            zattrs = {
                "multiscales": [
                    {
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1]}
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [4, 2]}
                                ],
                            },
                        ]
                    }
                ]
            }
            with open(os.path.join(zarr_path, ".zattrs"), "w") as f:
                json.dump(zattrs, f)

            root = zarr.open_group(zarr_path, mode="r")
            base_arr = root["0"]
            adapter = OmeZarrAdapter(base_arr, "test")

            # Test slice conversion
            level_slice = adapter._convert_slice_to_level(
                SliceHint(start=[10, 20], stop=[50, 60]), (4, 2)
            )

            assert list(level_slice.start) == [2, 10]  # 10//4=2, 20//2=10
            assert list(level_slice.stop) == [12, 30]  # 50//4=12, 60//2=30


class TestGetData:
    """Tests for get_data method across adapters."""

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_aligned(self):
        """Test ZarrAdapter get_data with aligned bounds."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )
            arr[:] = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "test", ["y", "x"]
            )

            # Aligned bounds
            bounds = ChunkBounds(start=[0, 0], stop=[50, 50])
            data = adapter.get_data(bounds)

            assert data.shape == (50, 50)
            assert data.dtype == np.uint8
            # Verify by slicing the full array directly
            full_arr = zarr.open_array(zarr_path, mode="r")
            expected = full_arr[0:50, 0:50]
            np.testing.assert_array_equal(data, expected)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_non_aligned(self):
        """Test ZarrAdapter get_data with non-aligned bounds."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )
            arr[:] = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

            adapter = ZarrAdapter(
                zarr.open_array(zarr_path, mode="r"), "test", ["y", "x"]
            )

            # Non-aligned bounds
            bounds = ChunkBounds(start=[10, 20], stop=[30, 40])
            data = adapter.get_data(bounds)

            assert data.shape == (20, 20)
            # Verify by slicing the full array
            full_arr = zarr.open_array(zarr_path, mode="r")
            expected = full_arr[10:30, 20:40]
            np.testing.assert_array_equal(data, expected)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_out_of_bounds(self):
        """Test ZarrAdapter get_data raises on out-of-bounds."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )

            adapter = ZarrAdapter(arr, "test", ["y", "x"])

            # Out of bounds
            bounds = ChunkBounds(start=[0, 0], stop=[150, 100])
            with pytest.raises(ValueError, match="exceeds shape"):
                adapter.get_data(bounds)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_negative_start(self):
        """Test ZarrAdapter get_data raises on negative start."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )

            adapter = ZarrAdapter(arr, "test", ["y", "x"])

            # Negative start
            bounds = ChunkBounds(start=[-10, 0], stop=[50, 50])
            with pytest.raises(ValueError, match="is negative"):
                adapter.get_data(bounds)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_zarr_get_data_dimensionality_mismatch(self):
        """Test ZarrAdapter get_data raises on wrong dimensionality."""
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test.zarr")
            arr = zarr.open_array(
                zarr_path, mode="w", shape=(100, 100), chunks=(50, 50), dtype="uint8"
            )

            adapter = ZarrAdapter(arr, "test", ["y", "x"])

            # Wrong dimensionality
            bounds = ChunkBounds(start=[0], stop=[50])
            with pytest.raises(ValueError, match="dimensionality mismatch"):
                adapter.get_data(bounds)
