"""Scaled read benchmarks.

Measures downsampling computation performance.
Virtual scaling tests are skipped pending implementation.
"""

import time

import pytest
import numpy as np

from benchmarks.utils import measure_read_time
from benchmarks.conftest import get_all_source_ids


class TestScaledRead:
    """Benchmarks for virtual scaling/downsampling.

    Note: Virtual scaling (scale_hint parameter) is not yet implemented.
    These tests are placeholders.
    """

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_scaled_read_first(self, benchmark, data_source, bench_client_flight):
        """First scaled read (compute + cache) - placeholder."""
        pytest.skip("Virtual scaling not yet implemented")

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_scaled_read_cached(self, benchmark, data_source, bench_server, bench_client_flight):
        """Second scaled read (cache hit) - placeholder."""
        pytest.skip("Virtual scaling not yet implemented")

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_native_level_comparison(self, data_source, bench_client_flight):
        """Compare native resolution - placeholder for pyramid tests."""
        pytest.skip("Multi-level pyramid tests require ome-zarr HCS support")


class TestScaleComputation:
    """Benchmarks for downsampling computation."""

    def test_bench_downsample_nearest(self, benchmark):
        """Downsampling with nearest neighbor method."""
        data = np.random.randint(0, 1000, size=(512, 512), dtype=np.uint16)

        def downsample_nearest():
            return data[::4, ::4]

        result = benchmark(downsample_nearest)
        assert result.shape == (128, 128)

    def test_bench_downsample_mean(self, benchmark):
        """Downsampling with mean aggregation."""
        data = np.random.randint(0, 1000, size=(512, 512), dtype=np.uint16)

        def downsample_mean():
            reshaped = data.reshape(128, 4, 128, 4)
            return reshaped.mean(axis=(1, 3)).astype(np.uint16)

        result = benchmark(downsample_mean)
        assert result.shape == (128, 128)

    def test_bench_downsample_max(self, benchmark):
        """Downsampling with max projection."""
        data = np.random.randint(0, 1000, size=(512, 512), dtype=np.uint16)

        def downsample_max():
            reshaped = data.reshape(128, 4, 128, 4)
            return reshaped.max(axis=(1, 3))

        result = benchmark(downsample_max)
        assert result.shape == (128, 128)


class TestScaleVsNativeThroughput:
    """Compare throughput at different scales - placeholder."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_multilevel_throughput(self, data_source, bench_client_flight):
        """Throughput at different pyramid levels - placeholder."""
        pytest.skip("Multi-level pyramid tests require ome-zarr HCS support")


class TestMultiScaleNavigation:
    """Zooming in/out - placeholder."""

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_zoom_in_sequence(self, benchmark, data_source, bench_client_flight):
        """Zooming in - placeholder."""
        pytest.skip("Multi-scale navigation requires ome-zarr HCS support")

    @pytest.mark.parametrize("data_source", get_all_source_ids(), indirect=True)
    def test_bench_zoom_out_sequence(self, benchmark, data_source, bench_client_flight):
        """Zooming out - placeholder."""
        pytest.skip("Multi-scale navigation requires ome-zarr HCS support")