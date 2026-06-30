"""Tests for the RemoteTensorAdapter caching passthrough proxy (biopb/biopb#178).

Covers the §2 adapter slice of docs/remote-tensor-cache.md:
- the pure array_id byte-splice on a chunk_id (chunk.rewrite_chunk_id_array_id),
- the grpc:// url split,
- end-to-end proxying: a local server fronting an in-process *upstream* server
  mirrors its catalog and serves identical pixels through the proxy,
- the inherited segment cache (a second read is served without re-hitting upstream).
"""

import importlib.util
import threading
import time

import numpy as np
import pytest


def _zarr_available() -> bool:
    return importlib.util.find_spec("zarr") is not None


# --------------------------------------------------------------------------- unit


class TestArrayIdByteSplice:
    def test_rewrite_preserves_bounds_tail(self):
        from biopb.tensor.ticket_pb2 import ChunkBounds
        from biopb_tensor_server.chunk import (
            decode_chunk_id,
            encode_chunk_id,
            rewrite_chunk_id_array_id,
        )

        bounds = ChunkBounds(start=[0, 64], stop=[64, 128])
        cid = encode_chunk_id("lab__img/Image:0", bounds)
        out = rewrite_chunk_id_array_id(cid, "img/Image:0")

        array_id, decoded = decode_chunk_id(out)
        assert array_id == "img/Image:0"
        assert list(decoded.start) == [0, 64]
        assert list(decoded.stop) == [64, 128]
        # round-trips back
        assert rewrite_chunk_id_array_id(out, "lab__img/Image:0") == cid

    def test_rewrite_preserves_scale_suffix(self):
        from biopb.tensor.ticket_pb2 import ChunkBounds
        from biopb_tensor_server.chunk import (
            decode_scale_info,
            encode_chunk_id_with_scale,
            is_scaled_chunk,
            rewrite_chunk_id_array_id,
        )

        bounds = ChunkBounds(start=[0, 0], stop=[128, 128])
        scaled = encode_chunk_id_with_scale("lab__img", bounds, [2, 2], "stride")
        assert is_scaled_chunk(scaled)

        out = rewrite_chunk_id_array_id(scaled, "img")
        assert is_scaled_chunk(out)  # suffix preserved
        scale_hint, method = decode_scale_info(out)
        assert list(scale_hint) == [2, 2]
        assert method == "stride"


class TestSplitGrpcUrl:
    def test_single_source_form(self):
        from biopb_tensor_server.adapters.remote_tensor import _split_grpc_url

        endpoint, source_id = _split_grpc_url("grpc://lab-store:8815/experiment1")
        assert endpoint == "grpc://lab-store:8815"
        assert source_id == "experiment1"

    def test_bare_host_form(self):
        from biopb_tensor_server.adapters.remote_tensor import _split_grpc_url

        endpoint, source_id = _split_grpc_url("grpc://lab-store:8815")
        assert endpoint == "grpc://lab-store:8815"
        assert source_id is None


# -------------------------------------------------------------------- end-to-end


def _serve(server):
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    time.sleep(1)
    return t


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestRemoteTensorProxy:
    def _upstream(self, zarr_path):
        import zarr
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter

        arr = zarr.open_array(zarr_path, mode="r")
        upstream = TensorFlightServer("grpc://localhost:0")
        upstream.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
        _serve(upstream)
        return upstream

    def _proxy(
        self, upstream_port, local_source_id="lab__img", upstream_source_id="img"
    ):
        from biopb_tensor_server import TensorFlightServer
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        adapter = RemoteTensorAdapter(
            source_id=local_source_id,
            upstream_location=f"grpc://localhost:{upstream_port}",
            upstream_source_id=upstream_source_id,
        )
        proxy = TensorFlightServer("grpc://localhost:0")
        proxy.register_source(local_source_id, adapter)
        _serve(proxy)
        return proxy

    def test_mirrors_catalog_and_pixels(self, simple_zarr_array):
        import zarr
        from biopb.tensor import TensorFlightClient

        zarr_path, shape, _ = simple_zarr_array
        expected = zarr.open_array(zarr_path, mode="r")[:]

        upstream = self._upstream(zarr_path)
        try:
            proxy = self._proxy(upstream.port)
            try:
                client = TensorFlightClient(f"grpc://localhost:{proxy.port}")

                # catalog mirrored under the local (namespaced) source_id
                sources = client.list_sources()
                assert "lab__img" in sources
                assert "img" not in sources  # upstream id is not leaked

                # descriptor mirrored (shape/dtype), array_id is the LOCAL one
                darr = client.get_tensor("lab__img")
                assert darr.shape == shape
                assert darr.dtype == np.uint8

                # pixels are byte-identical to the upstream source
                np.testing.assert_array_equal(darr.compute(), expected)

                client.close()
            finally:
                proxy.shutdown()
        finally:
            upstream.shutdown()

    def test_scaled_read_downsamples_via_upstream(self, simple_zarr_array):
        from biopb.tensor import TensorFlightClient

        zarr_path, shape, _ = simple_zarr_array
        upstream = self._upstream(zarr_path)
        try:
            proxy = self._proxy(upstream.port)
            try:
                client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
                darr = client.get_tensor(
                    "lab__img", scale_hint=[2, 2], reduction_method="stride"
                )
                assert darr.shape == (shape[0] // 2, shape[1] // 2)
                assert darr.compute().shape == (shape[0] // 2, shape[1] // 2)
                client.close()
            finally:
                proxy.shutdown()
        finally:
            upstream.shutdown()

    def test_no_alias_passthrough_ids(self, simple_zarr_array):
        # A lone upstream may use the verbatim id (no alias): local == upstream.
        from biopb.tensor import TensorFlightClient

        zarr_path, shape, _ = simple_zarr_array
        upstream = self._upstream(zarr_path)
        try:
            proxy = self._proxy(
                upstream.port, local_source_id="img", upstream_source_id="img"
            )
            try:
                client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
                assert "img" in client.list_sources()
                assert client.get_tensor("img").shape == shape
                client.close()
            finally:
                proxy.shutdown()
        finally:
            upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_inherited_segment_cache(simple_zarr_array, tmp_path):
    """A second resolve_chunk_data for the same chunk is served from the file
    cache without a second upstream fetch -- the proxy inherits the segment cache."""
    import zarr
    from biopb.tensor.ticket_pb2 import ChunkBounds
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.chunk import encode_chunk_id
    from biopb_tensor_server.config import CacheConfig

    zarr_path, shape, chunks = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = TensorFlightServer("grpc://localhost:0")
    upstream.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    _serve(upstream)
    try:
        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{upstream.port}",
            upstream_source_id="img",
        )

        cache_manager = CacheManager(
            CacheConfig(backend="file", file_cache_dir=str(tmp_path / "cache"))
        )

        # one chunk (top-left 64x64) addressed by the LOCAL array_id
        bounds = ChunkBounds(start=[0, 0], stop=[chunks[0], chunks[1]])
        chunk_id = encode_chunk_id("lab__img", bounds)

        calls = {"n": 0}
        real = adapter._upstream_record_batch

        def counting(upstream_chunk_id):
            calls["n"] += 1
            return real(upstream_chunk_id)

        adapter._upstream_record_batch = counting

        first = adapter.resolve_chunk_data(chunk_id, cache_manager)
        second = adapter.resolve_chunk_data(chunk_id, cache_manager)

        assert calls["n"] == 1  # upstream hit once; second read came from the cache
        assert first.num_rows == 1 and second.num_rows == 1
        # the cache entry is locatable (powers the localhost mmap fast path)
        assert cache_manager.locate_entry(chunk_id) is not None

        cache_manager.close()
    finally:
        upstream.shutdown()
