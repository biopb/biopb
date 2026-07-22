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
        from biopb_tensor_server.core.chunk import (
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
        from biopb_tensor_server.core.chunk import (
            decode_scale_info,
            encode_chunk_id_with_scale,
            is_scaled_chunk,
            rewrite_chunk_id_array_id,
        )

        bounds = ChunkBounds(start=[0, 0], stop=[128, 128])
        scaled = encode_chunk_id_with_scale("lab__img", bounds, [2, 2])
        assert is_scaled_chunk(scaled)

        out = rewrite_chunk_id_array_id(scaled, "img")
        assert is_scaled_chunk(out)  # scale suffix preserved
        # decode_scale_info returns the scale_hint only (method left the chunk_id, #178).
        assert list(decode_scale_info(out)) == [2, 2]


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


class TestMirrorSourceUrlTree:
    """The mirror's display source_url carries the upstream path (biopb/biopb#297).

    So a browser trees a mirrored source by its remote filepath under an endpoint
    root, instead of collapsing every source of an upstream into a flat node.
    """

    def _adapter(self, alias=None):
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        return RemoteTensorAdapter(
            source_id="lab__img" if alias else "img",
            upstream_location="grpc://store:8815",
            upstream_source_id="img",
            alias=alias,
        )

    def test_seed_folds_upstream_path_into_display_url(self):
        a = self._adapter()
        a.seed_catalog([], {}, True, "file:///labs/Yu/exp1/img.tif")
        assert a._source_url == "grpc://store:8815/labs/Yu/exp1/img.tif"

    def test_alias_is_the_authority(self):
        a = self._adapter(alias="lab")
        a.seed_catalog([], {}, True, "file:///data/x.tif")
        assert a._source_url == "grpc://lab/data/x.tif"

    def test_remote_scheme_upstream_keeps_bucket(self):
        a = self._adapter()
        a.seed_catalog([], {}, True, "s3://bucket/key/img.zarr")
        assert a._source_url == "grpc://store:8815/bucket/key/img.zarr"

    def test_empty_upstream_url_falls_back_to_source_id_form(self):
        a = self._adapter()
        a.seed_catalog([], {}, True, None)
        assert a._source_url == "grpc://store:8815:img"

    def test_url_change_flags_changed_for_resync(self):
        a = self._adapter()
        assert a.seed_catalog([], {}, True, "file:///a/x.tif") is True
        assert a.seed_catalog([], {}, True, "file:///a/x.tif") is False
        assert a.seed_catalog([], {}, True, "file:///a/y.tif") is True

    def test_descriptor_carries_the_tree_url(self):
        # get_source_descriptor() is what the metadata-DB catalog stores.
        a = self._adapter()
        a.seed_catalog([], {}, True, "file:///labs/exp/img.tif")
        assert (
            a.get_source_descriptor().source_url == "grpc://store:8815/labs/exp/img.tif"
        )


# -------------------------------------------------------------------- end-to-end


def _serve(server):
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    time.sleep(1)
    return t


def _db_upstream(zarr_path, source_ids, max_list_flights_results=None):
    """An upstream with a populated metadata DB (so query_sources is complete).

    Returns ``(upstream, register, unregister)``; register/unregister keep the
    DuckDB catalog in sync so a re-list's ``query_sources`` reflects the change.
    Not yet served -- the caller decides when (e.g. after configuring a cap).
    """
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    arr = zarr.open_array(zarr_path, mode="r")
    db = MetadataDatabase()
    kwargs = {}
    if max_list_flights_results is not None:
        kwargs["max_list_flights_results"] = max_list_flights_results
    upstream = TensorFlightServer("grpc://localhost:0", metadata_db=db, **kwargs)

    def register(sid):
        adapter = ZarrAdapter(arr, sid, ["y", "x"])
        upstream.register_source(sid, adapter)
        db.sync_source_added(sid, adapter)

    def unregister(sid):
        upstream.unregister_source(sid)
        db.sync_source_removed(sid)

    for sid in source_ids:
        register(sid)
    return upstream, register, unregister


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

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_proxy_sliced_read_preserves_slice_hint(self):
        """A sliced read through the proxy keeps slice_hint on the forwarded plan,
        so the client can crop the outward-snapped result.

        Regression guard: the forward path must not strip slice_hint. The client
        needs it on the wire to crop; losing it silently returns the snapped
        (oversized) region instead of the requested slice.
        """
        import tempfile

        import zarr
        from biopb.tensor import TensorFlightClient
        from biopb.tensor.descriptor_pb2 import (
            FlightCmd,
            SliceHint,
            TensorDescriptor,
            TensorReadOption,
        )
        from pyarrow import flight

        with tempfile.TemporaryDirectory() as tmp:
            zpath = f"{tmp}/vol.zarr"
            za = zarr.open_array(
                zpath, mode="w", shape=(3, 40, 50), chunks=(1, 40, 50), dtype="<i2"
            )
            za[:] = np.arange(3 * 40 * 50, dtype="<i2").reshape(3, 40, 50)

            upstream = self._upstream_3d(zpath)
            try:
                proxy = self._proxy(
                    upstream.port,
                    local_source_id="hpc__aics",
                    upstream_source_id="aics",
                )
                try:
                    pc = TensorFlightClient(f"grpc://localhost:{proxy.port}")
                    # Request the middle z-plane; it snaps to exactly chunk 1.
                    sl = SliceHint(start=[1, 0, 0], stop=[2, 40, 50])
                    read_opt = TensorReadOption(tensor_id="hpc__aics", slice_hint=sl)
                    cmd = FlightCmd(source_id="hpc__aics", tensor_read=read_opt)
                    fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
                    info = pc._client.get_flight_info(fd, options=pc._call_options)
                    desc = TensorDescriptor.FromString(info.descriptor.command)
                    # The forwarded plan carries the realized-bounds crop signal.
                    assert desc.HasField("slice_hint")
                    assert list(desc.slice_hint.start) == [1, 0, 0]
                    assert list(desc.slice_hint.stop) == [2, 40, 50]
                    # The explicit endpoint list is shipped.
                    assert len(list(info.endpoints)) >= 1
                    pc.close()
                finally:
                    proxy.shutdown()
            finally:
                upstream.shutdown()

    def _upstream_3d(self, zarr_path):
        import zarr
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter

        arr = zarr.open_array(zarr_path, mode="r")
        upstream = TensorFlightServer("grpc://localhost:0")
        upstream.register_source("aics", ZarrAdapter(arr, "aics", ["z", "y", "x"]))
        _serve(upstream)
        return upstream

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_big_endian_source_with_empty_seeded_chunk_shape(self):
        """The exact reported failure: a big-endian upstream source mirrored with
        a lean (empty chunk_shape) catalog seed reads end-to-end through the proxy.

        Reproduces BOTH #292 (empty chunk_shape from the lean metadata-DB STRUCT
        -> read-plan IndexError) and #293 (big-endian '>i2' -> Arrow byte-swap) at
        once -- the way the mirrored AICS/FITS sources failed.
        """
        import tempfile

        import zarr
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        with tempfile.TemporaryDirectory() as tmp:
            zpath = f"{tmp}/be.zarr"
            src = (
                np.arange(3 * 40 * 50, dtype="<i2").reshape(3, 40, 50) - 1000
            ).astype(">i2")
            za = zarr.open_array(
                zpath, mode="w", shape=(3, 40, 50), chunks=(1, 40, 50), dtype=">i2"
            )
            za[:] = src

            upstream = TensorFlightServer("grpc://localhost:0")
            upstream.register_source(
                "aics",
                ZarrAdapter(zarr.open_array(zpath, mode="r"), "aics", ["z", "y", "x"]),
            )
            _serve(upstream)
            try:
                adapter = RemoteTensorAdapter(
                    source_id="hpc__aics",
                    upstream_location=f"grpc://localhost:{upstream.port}",
                    upstream_source_id="aics",
                )
                # Seed like the reconcile does from the metadata-DB STRUCT: an
                # EMPTY chunk_shape (the #292 trigger).
                adapter.seed_catalog(
                    upstream_tensors=[
                        {
                            "array_id": "aics",
                            "dim_labels": ["z", "y", "x"],
                            "shape": [3, 40, 50],
                            "chunk_shape": [],
                            "dtype": ">i2",
                        }
                    ],
                    metadata={},
                    data_resident=True,
                )
                proxy = TensorFlightServer("grpc://localhost:0")
                proxy.register_source("hpc__aics", adapter)
                _serve(proxy)
                try:
                    client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
                    darr = client.get_tensor("hpc__aics")  # was IndexError (#292)
                    assert darr.shape == (3, 40, 50)
                    # do_get used to raise ArrowNotImplementedError (#293).
                    np.testing.assert_array_equal(
                        darr.compute().astype("<i2"), src.astype("<i2")
                    )
                    client.close()
                finally:
                    proxy.shutdown()
            finally:
                upstream.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_forward_flight_info_returns_upstream_native_grid(self):
        """forward_flight_info forwards a whole GetFlightInfo to the upstream and
        returns ITS endpoints on the native grid + server-advertised pyramid --
        the proxy re-derives no grid or pyramid locally, so the advisory (empty)
        catalog seed never drives planning (biopb/biopb#295).
        """
        import tempfile

        import zarr
        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
        from biopb_tensor_server.core.chunk import (
            is_proxy_envelope,
            peel_proxy_envelope,
            routing_array_id,
        )

        with tempfile.TemporaryDirectory() as tmp:
            zpath = f"{tmp}/vol.zarr"
            za = zarr.open_array(
                zpath, mode="w", shape=(3, 40, 50), chunks=(1, 40, 50), dtype="<i2"
            )
            za[:] = np.zeros((3, 40, 50), dtype="<i2")

            upstream = TensorFlightServer("grpc://localhost:0")
            upstream.register_source(
                "aics",
                ZarrAdapter(zarr.open_array(zpath, mode="r"), "aics", ["z", "y", "x"]),
            )
            _serve(upstream)
            try:
                adapter = RemoteTensorAdapter(
                    source_id="hpc__aics",
                    upstream_location=f"grpc://localhost:{upstream.port}",
                    upstream_source_id="aics",
                )
                adapter.seed_catalog(
                    upstream_tensors=[
                        {
                            "array_id": "aics",
                            "dim_labels": ["z", "y", "x"],
                            "shape": [3, 40, 50],
                            "chunk_shape": [],  # advisory + empty; must be ignored
                            "dtype": "<i2",
                        }
                    ],
                    metadata={},
                    data_resident=True,
                    indexed_at="2026-07-19 00:00:00",
                )
                plan = adapter.forward_flight_info(
                    TensorReadOption(tensor_id="hpc__aics")
                )

                assert plan is not None
                # Native upstream grid (one z-plane per chunk), NOT the whole-volume
                # (3,40,50) default grid an empty seed would otherwise produce.
                assert list(plan.descriptor.chunk_shape) == [1, 40, 50]
                assert len(plan.chunk_endpoints) == 3  # one endpoint per plane
                # The upstream's server-advertised pyramid rode through the forward
                # (the lean catalog localizer would have stripped it).
                assert len(plan.descriptor.pyramid) >= 1
                # Endpoints carry LOCAL-routed proxy envelopes wrapping the
                # upstream chunk_id verbatim (upstream 'aics' -> route 'hpc__aics'),
                # and the seeded indexed_at rides as the envelope's content_version.
                for ce in plan.chunk_endpoints:
                    assert is_proxy_envelope(ce.chunk_id)
                    route, cv, _inner = peel_proxy_envelope(ce.chunk_id)
                    assert route == "hpc__aics"
                    assert cv == b"iat:2026-07-19 00:00:00"
                    assert routing_array_id(ce.chunk_id) == "hpc__aics"
            finally:
                upstream.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_forward_flight_info_surfaces_upstream_native_pyramid(self):
        """A pyramidal OME-Zarr upstream's NATIVE precompute levels are surfaced
        through the proxy -- the whole-call forward carries them (the proxy itself
        computes none: get_native_pyramid_levels is None on it, so the local
        planner would advertise a *computed* pyramid instead). biopb/biopb#295.
        """
        import tempfile

        import zarr
        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb_tensor_server import OmeZarrAdapter, TensorFlightServer
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
        from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr

        with tempfile.TemporaryDirectory() as tmp:
            zpath, _, _ = create_multiresolution_ome_zarr(
                tmp, n_levels=4, base_shape=(256, 256), chunk_size=(64, 64)
            )
            root = zarr.open_group(zpath, mode="r")

            upstream = TensorFlightServer("grpc://localhost:0")
            upstream.register_source("ome", OmeZarrAdapter(root["0"], "ome"))
            _serve(upstream)
            try:
                adapter = RemoteTensorAdapter(
                    source_id="hpc__ome",
                    upstream_location=f"grpc://localhost:{upstream.port}",
                    upstream_source_id="ome",
                )
                adapter.seed_catalog(
                    upstream_tensors=[
                        {
                            "array_id": "ome",
                            "dim_labels": ["y", "x"],
                            "shape": [256, 256],
                            "chunk_shape": [64, 64],
                            "dtype": "uint8",
                        }
                    ],
                    metadata={},
                    data_resident=True,
                )
                plan = adapter.forward_flight_info(
                    TensorReadOption(tensor_id="hpc__ome")
                )

                assert plan is not None
                # Multiple native levels, each a precomputed on-disk dataset.
                assert len(plan.descriptor.pyramid) >= 2
                assert all(lvl.native for lvl in plan.descriptor.pyramid)
                assert all(
                    lvl.reduction_method == "precompute"
                    for lvl in plan.descriptor.pyramid
                )
            finally:
                upstream.shutdown()

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_forward_flight_info_returns_none_when_upstream_unreachable(self, caplog):
        """A **transport** failure (upstream unreachable) returns None so the
        *server* falls back to its local planner. Categorized as an expected
        operational condition: logged at DEBUG, never escalated to WARNING."""
        import logging

        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        adapter = RemoteTensorAdapter(
            source_id="hpc__aics",
            upstream_location="grpc://localhost:1",  # nothing listening
            upstream_source_id="aics",
        )
        adapter.seed_catalog(
            upstream_tensors=[
                {
                    "array_id": "aics",
                    "dim_labels": ["z", "y", "x"],
                    "shape": [3, 40, 50],
                    "chunk_shape": [1, 40, 50],
                    "dtype": "<i2",
                }
            ],
            metadata={},
            data_resident=True,
        )
        with caplog.at_level(
            logging.DEBUG, logger="biopb_tensor_server.adapters.remote_tensor"
        ):
            plan = adapter.forward_flight_info(TensorReadOption(tensor_id="hpc__aics"))
        assert plan is None
        recs = [r for r in caplog.records if r.name.endswith("remote_tensor")]
        assert any("RPC failed" in r.getMessage() for r in recs)
        # a transport error is expected -- it must NOT be raised to WARNING
        assert not any(r.levelno >= logging.WARNING for r in recs)

    @pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
    def test_forward_flight_info_returns_none_on_unparseable_upstream_endpoint(
        self, caplog
    ):
        """A **logic** failure (a response received but not localizable) returns
        None so the server falls back -- but, unlike a transport error, it is
        unexpected and logged at WARNING so the fallback never silently masks a
        protocol mismatch or a proxy bug."""
        import logging
        from types import SimpleNamespace

        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        adapter = RemoteTensorAdapter(
            source_id="hpc__aics",
            upstream_location="grpc://localhost:1",
            upstream_source_id="aics",
        )
        adapter.seed_catalog(
            upstream_tensors=[
                {
                    "array_id": "aics",
                    "dim_labels": ["z", "y", "x"],
                    "shape": [3, 40, 50],
                    "chunk_shape": [1, 40, 50],
                    "dtype": "<i2",
                }
            ],
            metadata={},
            data_resident=True,
        )
        # A valid descriptor but a junk endpoint (no .ticket) -> the parse loop
        # raises inside the guarded LOGIC block (not the transport block).
        valid_cmd = adapter.get_tensor_descriptor().SerializeToString()
        fake_info = SimpleNamespace(
            descriptor=SimpleNamespace(command=valid_cmd),
            endpoints=[SimpleNamespace()],  # missing .ticket
        )
        adapter._upstream_flight_info = lambda read_opt: fake_info

        with caplog.at_level(
            logging.DEBUG, logger="biopb_tensor_server.adapters.remote_tensor"
        ):
            plan = adapter.forward_flight_info(TensorReadOption(tensor_id="hpc__aics"))
        assert plan is None
        recs = [r for r in caplog.records if r.name.endswith("remote_tensor")]
        assert any(
            r.levelno == logging.WARNING and "could not be localized" in r.getMessage()
            for r in recs
        )

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
class TestBareHostExpansion:
    """A bare grpc://host:port source mirrors *every* upstream source (§3)."""

    def test_expansion_mirrors_all_sources_namespaced(self, simple_zarr_array):
        from biopb_tensor_server import TensorFlightServer
        from biopb_tensor_server.core.config import SourceConfig, discover_sources

        zarr_path, shape, _ = simple_zarr_array
        upstream, _, _ = _db_upstream(zarr_path, ["img", "img2"])
        _serve(upstream)
        try:
            expanded = discover_sources(
                SourceConfig(url=f"grpc://localhost:{upstream.port}", alias="lab")
            )
            by_id = {s.source_id: s for s in expanded}
            assert set(by_id) == {"lab__img", "lab__img2"}
            assert by_id["lab__img"].url == f"grpc://localhost:{upstream.port}/img"
            assert all(s.type == "tensor-server" for s in expanded)

            # the expanded configs build working adapters that serve through a proxy
            from biopb.tensor import TensorFlightClient
            from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

            proxy = TensorFlightServer("grpc://localhost:0")
            for cfg in expanded:
                proxy.register_source(
                    cfg.source_id, RemoteTensorAdapter.create_from_config(cfg)
                )
            _serve(proxy)
            try:
                client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
                assert set(client.list_sources()) == {"lab__img", "lab__img2"}
                assert client.get_tensor("lab__img2").shape == shape
                client.close()
            finally:
                proxy.shutdown()
        finally:
            upstream.shutdown()

    def test_expansion_complete_despite_list_flights_truncation(
        self, simple_zarr_array
    ):
        """list_sources() is capped (max_list_flights_results), but the bare-host
        expansion must mirror EVERY upstream source via the complete server-side
        catalog -- otherwise a large upstream is silently under-mirrored."""
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server.core.config import SourceConfig, discover_sources

        zarr_path, _, _ = simple_zarr_array
        # cap list_flights at 1 while registering 3 sources
        upstream, _, _ = _db_upstream(
            zarr_path, ["a", "b", "c"], max_list_flights_results=1
        )
        _serve(upstream)
        try:
            # sanity: the capped list_sources IS truncated to 1...
            probe = TensorFlightClient(f"grpc://localhost:{upstream.port}")
            assert len(probe.list_sources()) == 1
            probe.close()

            # ...but the expansion mirrors all 3 (via the complete SQL catalog)
            expanded = discover_sources(
                SourceConfig(url=f"grpc://localhost:{upstream.port}", alias="lab")
            )
            assert {s.source_id for s in expanded} == {"lab__a", "lab__b", "lab__c"}
        finally:
            upstream.shutdown()


def test_create_from_config_resolves_token_from_credentials_profile():
    """A per-upstream credentials profile (storage_type=biopb-tensor) supplies the
    upstream bearer token to the adapter -- the multi-upstream auth path (§1/§3)."""
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.remote import CredentialProfile, CredentialsConfig

    creds = CredentialsConfig(
        default_profile=None,
        profiles=[
            CredentialProfile(
                name="lab-store", storage_type="biopb-tensor", token="s3cr3t"
            )
        ],
    )
    source = SourceConfig(url="grpc://lab:8815/img", credentials_profile="lab-store")
    adapter = RemoteTensorAdapter.create_from_config(source, creds)
    assert adapter._token == "s3cr3t"
    assert adapter._upstream_source_id == "img"
    assert adapter._upstream_location == "grpc://lab:8815"


def _meta_zarr_cls():
    from biopb_tensor_server import ZarrAdapter

    class _MetaZarr(ZarrAdapter):
        def get_metadata(self):
            return {"ome": {"channel": "DAPI"}}

    return _MetaZarr


# physical scale/unit for the 2D (y, x) fixture
_PHYS_SCALE = [0.5, 0.25]
_PHYS_UNIT = ["micrometer", "micrometer"]


def _phys_zarr_cls():
    from biopb_tensor_server import ZarrAdapter

    class _PhysZarr(ZarrAdapter):
        def _physical_scale(self):
            return list(_PHYS_SCALE), list(_PHYS_UNIT)

    return _PhysZarr


def _upstream_with_metadata(zarr_path):
    """An upstream server whose metadata DB holds one source ('img') with metadata."""
    import zarr
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    arr = zarr.open_array(zarr_path, mode="r")
    db = MetadataDatabase()  # in-memory, enabled
    upstream = TensorFlightServer("grpc://localhost:0", metadata_db=db)
    up_adapter = _meta_zarr_cls()(arr, "img", ["y", "x"])
    upstream.register_source("img", up_adapter)
    db.sync_source_added("img", up_adapter)  # populate the DuckDB sources row
    _serve(upstream)
    return upstream


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_get_metadata_reads_from_upstream_metadata_db(simple_zarr_array):
    """Source metadata is read from the upstream's metadata catalog via a SQL query
    (sources.metadata_json stores the raw dict) -- list_flights is lean and the
    buggy list_sources() path returned {}."""
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    zarr_path, _, _ = simple_zarr_array
    upstream = _upstream_with_metadata(zarr_path)
    try:
        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{upstream.port}",
            upstream_source_id="img",
        )
        assert adapter.get_metadata() == {"ome": {"channel": "DAPI"}}
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_metadata_flows_through_proxy_single_wrapped(simple_zarr_array):
    """End-to-end: a client GetFlightInfo through the proxy carries the upstream's
    metadata, wrapped exactly once (not empty, not double-wrapped).

    The proxy serves metadata from its own catalog, populated at registration
    from the mirrored ``get_metadata()`` -- the serve path never recomputes on
    the adapter (biopb/biopb#253), so a real proxy carries a metadata DB.
    """
    import json

    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    zarr_path, _, _ = simple_zarr_array
    upstream = _upstream_with_metadata(zarr_path)
    try:
        proxy_db = MetadataDatabase()
        proxy = TensorFlightServer("grpc://localhost:0", metadata_db=proxy_db)
        proxy_adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{upstream.port}",
            upstream_source_id="img",
        )
        proxy.register_source("lab__img", proxy_adapter)
        proxy_db.sync_source_added("lab__img", proxy_adapter)  # mirror -> catalog
        _serve(proxy)
        try:
            client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
            desc = client.get_descriptor("lab__img")  # GetFlightInfo(with_metadata)
            assert desc.metadata_json  # was empty under the bug
            wrapped = json.loads(desc.metadata_json)
            assert wrapped["metadata"] == {"ome": {"channel": "DAPI"}}
            # single wrap: the inner payload is the raw dict, not another envelope
            assert "metadata" not in wrapped["metadata"]
            client.close()
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_get_metadata_empty_when_upstream_has_no_metadata_db(simple_zarr_array):
    """No fallback: a reachable upstream whose metadata DB is absent yields {}
    (best-effort) -- the source still mirrors/serves, only metadata is empty."""
    import zarr
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = TensorFlightServer("grpc://localhost:0")  # no metadata_db
    upstream.register_source("img", _meta_zarr_cls()(arr, "img", ["y", "x"]))
    _serve(upstream)
    try:
        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{upstream.port}",
            upstream_source_id="img",
        )
        assert adapter.list_tensor_descriptors()  # reachable -> mirrored
        assert adapter.get_metadata() == {}  # query fails -> graceful empty
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_get_physical_scale_returns_none_unimplemented(simple_zarr_array):
    """The proxy's own ``_physical_scale()`` hook returns None (it does not
    override it) even when the upstream advertises a scale: physical scale is
    surfaced at a different layer -- it rides the whole-call forward
    (plan_flight_info -> forward_flight_info -> the upstream descriptor's
    physical_scale), not a per-open get_descriptor RPC from this hook
    (biopb/biopb#295). See test_physical_scale_surfaced_through_proxy for the
    end-to-end path."""
    import zarr
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = TensorFlightServer("grpc://localhost:0")
    upstream.register_source("img", _phys_zarr_cls()(arr, "img", ["y", "x"]))
    _serve(upstream)
    try:
        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{upstream.port}",
            upstream_source_id="img",
        )
        assert adapter._physical_scale() is None
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_physical_scale_surfaced_through_proxy(simple_zarr_array):
    """End-to-end: the upstream's physical_scale/physical_unit are surfaced to a
    client GetFlightInfo through the proxy. The whole-call forward
    (forward_flight_info) rides the upstream descriptor's physical scale straight
    through -- no per-open get_descriptor RPC of its own, since the same forwarded
    GetFlightInfo already carries the grid (biopb/biopb#295, closing #266's gap)."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = TensorFlightServer("grpc://localhost:0")
    upstream.register_source("img", _phys_zarr_cls()(arr, "img", ["y", "x"]))
    _serve(upstream)
    try:
        proxy = TensorFlightServer("grpc://localhost:0")
        proxy.register_source(
            "lab__img",
            RemoteTensorAdapter(
                source_id="lab__img",
                upstream_location=f"grpc://localhost:{upstream.port}",
                upstream_source_id="img",
            ),
        )
        _serve(proxy)
        try:
            client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
            # physical_scale is filled independent of with_metadata; fetch the
            # structural descriptor so this DB-less proxy is not asked for a
            # metadata catalog it does not have.
            desc = client.get_descriptor("lab__img", with_metadata=False)
            assert list(desc.physical_scale) == list(_PHYS_SCALE)
            assert list(desc.physical_unit) == list(_PHYS_UNIT)
            client.close()
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_server_get_flight_info_uses_proxy_forward():
    """A proxy tensor forwards to the upstream and returns ITS native pyramid,
    rather than running the local planner (which would advertise a *computed*,
    non-native pyramid). Calls server.get_flight_info directly to exercise the
    proxy's plan_flight_info forward (biopb/biopb#295).
    """
    import tempfile

    import zarr
    from biopb.tensor.descriptor_pb2 import (
        FlightCmd,
        TensorDescriptor,
        TensorReadOption,
    )
    from biopb_tensor_server import OmeZarrAdapter, TensorFlightServer
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from biopb_tensor_server.fixtures import create_multiresolution_ome_zarr
    from pyarrow import flight

    with tempfile.TemporaryDirectory() as tmp:
        zpath, _, _ = create_multiresolution_ome_zarr(
            tmp, n_levels=4, base_shape=(256, 256), chunk_size=(64, 64)
        )
        root = zarr.open_group(zpath, mode="r")

        upstream = TensorFlightServer("grpc://localhost:0")
        upstream.register_source("ome", OmeZarrAdapter(root["0"], "ome"))
        _serve(upstream)
        try:
            proxy = TensorFlightServer("grpc://localhost:0")
            proxy.register_source(
                "hpc__ome",
                RemoteTensorAdapter(
                    source_id="hpc__ome",
                    upstream_location=f"grpc://localhost:{upstream.port}",
                    upstream_source_id="ome",
                ),
            )
            _serve(proxy)
            try:
                cmd = FlightCmd(
                    source_id="hpc__ome",
                    tensor_read=TensorReadOption(tensor_id="hpc__ome"),
                )
                fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
                # Direct server-method call (no client): the proxy source carries no
                # token, so _authorize_source is a no-op and context can be None.
                info = proxy.get_flight_info(None, fd)

                out = TensorDescriptor.FromString(info.descriptor.command)
                # The upstream's NATIVE pyramid rode through the forward -- the local
                # planner would have produced a computed (native=False) pyramid.
                assert len(out.pyramid) >= 2
                assert all(lvl.native for lvl in out.pyramid)
                assert all(lvl.reduction_method == "precompute" for lvl in out.pyramid)
                assert len(info.endpoints) >= 1
            finally:
                proxy.shutdown()
        finally:
            upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_server_get_flight_info_falls_back_when_proxy_forward_none(simple_zarr_array):
    """When a proxy's forward_flight_info returns None (upstream unreachable /
    unparseable), server.get_flight_info falls through to the local planner and
    still returns a best-effort plan -- the branch degrades, it does not raise."""
    import zarr
    from biopb.tensor.descriptor_pb2 import FlightCmd, TensorReadOption
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from pyarrow import flight

    zarr_path, _, _ = simple_zarr_array
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
        adapter.seed_catalog(
            upstream_tensors=[
                {
                    "array_id": "img",
                    "dim_labels": ["y", "x"],
                    "shape": [128, 128],
                    "chunk_shape": [64, 64],
                    "dtype": "uint8",
                }
            ],
            metadata={},
            data_resident=True,
        )
        # Force the forward to yield nothing -> the server must use the local planner.
        adapter.forward_flight_info = lambda read_opt: None

        proxy = TensorFlightServer("grpc://localhost:0")
        proxy.register_source("lab__img", adapter)
        _serve(proxy)
        try:
            cmd = FlightCmd(
                source_id="lab__img",
                tensor_read=TensorReadOption(tensor_id="lab__img"),
            )
            fd = flight.FlightDescriptor.for_command(cmd.SerializeToString())
            info = proxy.get_flight_info(None, fd)  # must not raise
            assert len(info.endpoints) >= 1  # local planner produced a plan
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_monitored_upstream_relist_adds_and_removes(simple_zarr_array):
    """A monitored bare-host upstream is periodically re-listed: sources that
    appear/disappear on the upstream are mirrored/dropped on the proxy (§3 refresh).

    The upstream has a metadata DB so the re-list enumerates via the complete
    query_sources catalog -- removals are only applied on a complete list."""
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.sources.source_manager import SourceManager

    zarr_path, shape, _ = simple_zarr_array
    upstream, up_register, up_unregister = _db_upstream(zarr_path, ["img"])
    _serve(upstream)
    try:
        proxy = TensorFlightServer("grpc://localhost:0")
        _serve(proxy)
        try:
            manager = SourceManager(
                server=proxy,
                registry=get_default_registry(),
                discovery_state=DiscoveryState(),
                watcher=None,
                monitored_dirs=set(),
                metadata_db=None,
                monitored_upstreams=[
                    SourceConfig(url=f"grpc://localhost:{upstream.port}", alias="lab")
                ],
            )
            client = TensorFlightClient(f"grpc://localhost:{proxy.port}")

            # initial re-list mirrors the upstream's single source
            manager._reconcile_upstreams()
            assert set(client.list_sources()) == {"lab__img"}

            # a new upstream source appears -> mirrored on the next re-list
            up_register("img2")
            manager._reconcile_upstreams()
            assert set(client.list_sources()) == {"lab__img", "lab__img2"}
            assert client.get_tensor("lab__img2").shape == shape

            # an upstream source disappears -> dropped on the next re-list
            up_unregister("img")
            manager._reconcile_upstreams()
            assert set(client.list_sources()) == {"lab__img2"}

            client.close()
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_create_source_manager_captures_bare_host_monitored_upstream(simple_zarr_array):
    """create_source_manager records a monitored bare-host grpc:// source as a
    re-list upstream, and excludes the single-source grpc://host/<id> form."""
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.sources.source_manager import create_source_manager

    zarr_path, _, _ = simple_zarr_array
    server = TensorFlightServer("grpc://localhost:0")
    manager = create_source_manager(
        server=server,
        registry=get_default_registry(),
        watcher=None,
        # a local static source so there is something to serve (else it bails)
        static_sources=[
            SourceConfig(
                type="zarr", url=zarr_path, source_id="local", dim_labels=["y", "x"]
            )
        ],
        monitored_sources=[
            SourceConfig(url="grpc://lab:8815", alias="lab", monitor=True),
            SourceConfig(url="grpc://lab:8815/one", alias="lab", monitor=True),
        ],
        metadata_db=None,
    )
    urls = [u.url for u in manager._monitored_upstreams]
    assert "grpc://lab:8815" in urls
    assert (
        "grpc://lab:8815/one" not in urls
    )  # single-source form has nothing to re-list


def test_handle_rescan_walks_local_dirs_before_upstream_relist(tmp_path):
    """A rescan walks the monitored local dirs BEFORE re-listing upstreams.

    The upstream re-list registers one proxy per mirrored source, each a network
    round-trip; a large upstream can take minutes. Running the local walk first
    means local directory sources surface promptly instead of waiting behind the
    whole upstream mirror (biopb/biopb#178 ordering)."""
    from unittest.mock import MagicMock

    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.sources.source_manager import SourceManager

    manager = SourceManager(
        server=MagicMock(),
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        watcher=None,
        monitored_dirs={tmp_path},
        metadata_db=None,
        monitored_upstreams=[SourceConfig(url="grpc://lab:8815", alias="lab")],
    )

    order = []
    manager._rescan_monitored_dirs = lambda: order.append("local")
    manager._reconcile_due_upstreams = lambda: order.append("upstream")

    manager._handle_rescan()

    assert order == ["local", "upstream"]


def test_handle_rescan_suppresses_live_precache_for_boot_tick_upstream(tmp_path):
    """On the boot tick the local walk flips _initial_scan_done True before the
    upstream re-list; _handle_rescan suppresses the live-precache enqueue across
    that re-list so the startup upstream mirror routes to the slow backlog, not
    the un-idle-gated prompt tier. Steady-state ticks never suppress."""
    from unittest.mock import MagicMock

    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.sources.source_manager import SourceManager

    manager = SourceManager(
        server=MagicMock(),
        registry=get_default_registry(),
        discovery_state=DiscoveryState(),
        watcher=None,
        monitored_dirs={tmp_path},
        metadata_db=None,
        monitored_upstreams=[SourceConfig(url="grpc://lab:8815", alias="lab")],
    )

    seen = {}

    # The local walk flips the gate mid-tick, exactly as the first full scan does.
    def _walk():
        seen["during_local"] = manager._suppress_live_precache
        manager._initial_scan_done = True

    manager._rescan_monitored_dirs = _walk
    manager._reconcile_due_upstreams = lambda: seen.update(
        during_upstream=manager._suppress_live_precache
    )

    # Boot tick: initial scan not yet done at tick start.
    manager._initial_scan_done = False
    manager._handle_rescan()
    assert seen["during_local"] is False  # local walk is not suppressed
    assert seen["during_upstream"] is True  # startup upstream mirror is
    assert manager._suppress_live_precache is False  # reset after the re-list

    # Steady-state tick: initial scan already done at tick start -> no suppression.
    seen.clear()
    manager._handle_rescan()
    assert seen["during_upstream"] is False
    assert manager._suppress_live_precache is False


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_failed_upstream_retried_on_fast_incremental_cadence(simple_zarr_array):
    """An upstream down at boot is retried on the fast incremental cadence -- not
    only the slow force-full (1h) one: recovery happens on a rescan that is NOT a
    force-full pass, because the failed upstream is marked for prompt retry."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.core.metadata_db import MetadataDatabase
    from biopb_tensor_server.sources.source_manager import SourceManager

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")

    # reserve a port and leave it closed -- the upstream is "down at boot"
    seed = TensorFlightServer("grpc://localhost:0")
    port = seed.port
    seed.shutdown()
    url = f"grpc://localhost:{port}"

    proxy = TensorFlightServer("grpc://localhost:0")
    _serve(proxy)
    try:
        manager = SourceManager(
            server=proxy,
            registry=get_default_registry(),
            discovery_state=DiscoveryState(),
            watcher=None,
            monitored_dirs=set(),
            metadata_db=None,
            monitored_upstreams=[SourceConfig(url=url, alias="lab")],
        )
        client = TensorFlightClient(f"grpc://localhost:{proxy.port}")

        # first rescan is force-full (last-full = -inf) -> tries the dead upstream
        assert manager._should_force_full_rescan() is True
        manager._handle_rescan()
        assert url in manager._failed_upstreams  # recorded as failed
        # the force-full was consumed, so the next rescan is NOT force-full
        assert manager._should_force_full_rescan() is False
        assert set(client.list_sources()) == set()  # nothing mirrored yet

        # the upstream comes up on the SAME port with a populated metadata DB
        db = MetadataDatabase()
        up = TensorFlightServer(url, metadata_db=db)
        adapter = ZarrAdapter(arr, "img", ["y", "x"])
        up.register_source("img", adapter)
        db.sync_source_added("img", adapter)
        _serve(up)
        try:
            # this rescan is NOT a force-full pass, yet the failed upstream is
            # retried and recovers -- the whole point of the fast cadence
            assert manager._should_force_full_rescan() is False
            manager._handle_rescan()
            assert set(client.list_sources()) == {"lab__img"}
            assert manager._failed_upstreams == set()  # cleared on recovery
            client.close()
        finally:
            up.shutdown()
    finally:
        proxy.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_stable_upstream_backs_off_then_resets_on_change(simple_zarr_array):
    """A stable upstream is re-listed less often (period doubles in ticks while the
    source set is unchanged); a new source resets it to the fast every-tick cadence."""
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.sources.source_manager import SourceManager

    zarr_path, _, _ = simple_zarr_array
    upstream, up_register, _ = _db_upstream(zarr_path, ["img"])
    _serve(upstream)
    url = f"grpc://localhost:{upstream.port}"
    try:
        proxy = TensorFlightServer("grpc://localhost:0")
        _serve(proxy)
        try:
            manager = SourceManager(
                server=proxy,
                registry=get_default_registry(),
                discovery_state=DiscoveryState(),
                watcher=None,
                monitored_dirs=set(),
                metadata_db=None,
                monitored_upstreams=[SourceConfig(url=url, alias="lab")],
            )
            client = TensorFlightClient(f"grpc://localhost:{proxy.port}")

            def period():
                return manager._upstream_relist[url]["period"]

            manager._handle_rescan()  # mirrors lab__img (a change) -> stays fast
            assert period() == 1
            manager._handle_rescan()  # unchanged -> back off to every 2 ticks
            assert period() == 2
            manager._handle_rescan()  # skipped (not due this tick)
            manager._handle_rescan()  # due, unchanged -> every 4 ticks
            assert period() == 4

            # a new upstream source appears: the next DUE re-list mirrors it and
            # resets the cadence to fast (every tick)
            up_register("img2")
            for _ in range(4):  # advance past the skipped ticks to the next due
                manager._handle_rescan()
            assert period() == 1
            assert set(client.list_sources()) == {"lab__img", "lab__img2"}

            client.close()
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestUnreachableUpstream:
    """Unresolved/unreachable upstream policy (#178).

    Catalog surface degrades to an empty placeholder (no raise, no startup
    hard-fail); the serve surface stays live so the source recovers transparently
    once the upstream is back -- no consented resolve step (a proxy resolve is a
    cheap reconnect, unlike a cloud download).
    """

    def _dead_port(self):
        # Start then immediately stop a server to obtain a now-closed port.
        from biopb_tensor_server import TensorFlightServer

        s = TensorFlightServer("grpc://localhost:0")
        port = s.port
        s.shutdown()
        return port

    def test_catalog_degrades_to_placeholder_when_unreachable(self):
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{self._dead_port()}",
            upstream_source_id="img",
        )
        # no raise: empty placeholder catalog row, marked non-resident
        assert adapter.list_tensor_descriptors() == []
        assert adapter.is_resident() is False
        desc = adapter.get_source_descriptor()
        assert list(desc.tensors) == []
        assert desc.data_resident is False

    def test_serve_surface_still_raises_when_unreachable(self):
        from biopb.tensor.ticket_pb2 import ChunkBounds
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
        from pyarrow import flight

        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{self._dead_port()}",
            upstream_source_id="img",
        )
        # the serve path must NOT silently degrade -- a missing chunk is an error
        # (a dead upstream surfaces as a Flight transport / socket error).
        with pytest.raises((flight.FlightError, OSError)):
            adapter.get_data(ChunkBounds(start=[0, 0], stop=[8, 8]))

    def test_transparent_recovery_when_upstream_returns(self, simple_zarr_array):
        import zarr
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter
        from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

        zarr_path, shape, _ = simple_zarr_array
        arr = zarr.open_array(zarr_path, mode="r")

        # pick a port, leave it closed, point the adapter at it
        seed = TensorFlightServer("grpc://localhost:0")
        port = seed.port
        seed.shutdown()

        adapter = RemoteTensorAdapter(
            source_id="lab__img",
            upstream_location=f"grpc://localhost:{port}",
            upstream_source_id="img",
        )
        assert adapter.list_tensor_descriptors() == []  # down -> placeholder

        # bring an upstream up on that same port; the SAME adapter now serves live
        upstream = TensorFlightServer(f"grpc://localhost:{port}")
        upstream.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
        _serve(upstream)
        try:
            descs = adapter.list_tensor_descriptors()
            assert len(descs) == 1
            assert descs[0].array_id == "lab__img"  # localized
            assert adapter.is_resident() is True
            assert tuple(adapter.get_tensor_descriptor().shape) == shape
        finally:
            upstream.shutdown()


def test_unreachable_sole_monitored_upstream_does_not_block_startup():
    """A sole bare-host monitor=true upstream that is down at boot must not stop
    the server starting -- the re-list recovers it once reachable (#178)."""
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.sources.source_manager import create_source_manager

    server = TensorFlightServer("grpc://localhost:0")
    manager = create_source_manager(
        server=server,
        registry=get_default_registry(),
        watcher=None,
        static_sources=[],  # expansion of the down upstream yielded nothing
        monitored_sources=[
            SourceConfig(url="grpc://localhost:59599", alias="lab", monitor=True)
        ],
        metadata_db=None,
    )
    assert manager is not None  # would have been None (hard-fail) before the fix
    assert [u.url for u in manager._monitored_upstreams] == ["grpc://localhost:59599"]


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_display_friendly_proxied_source_url(simple_zarr_array):
    """A proxied source's catalog source_url is grpc://<alias-or-host:port>:<id>
    (keeps the grpc:// scheme, but far more legible than the bare endpoint that is
    identical for every source of an upstream)."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = TensorFlightServer("grpc://localhost:0")
    upstream.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    _serve(upstream)
    up = f"grpc://localhost:{upstream.port}"
    try:
        proxy = TensorFlightServer("grpc://localhost:0")
        proxy.register_source(  # aliased -> grpc://lab:img
            "lab__img",
            RemoteTensorAdapter(
                source_id="lab__img",
                upstream_location=up,
                upstream_source_id="img",
                alias="lab",
            ),
        )
        proxy.register_source(  # no alias -> grpc://<host:port>:img
            "img",
            RemoteTensorAdapter(
                source_id="img", upstream_location=up, upstream_source_id="img"
            ),
        )
        _serve(proxy)
        try:
            client = TensorFlightClient(f"grpc://localhost:{proxy.port}")
            sources = client.list_sources()
            assert sources["lab__img"].source_url == "grpc://lab:img"
            assert sources["img"].source_url == f"grpc://localhost:{upstream.port}:img"
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
    from biopb_tensor_server.core.chunk import encode_chunk_id, encode_proxy_envelope
    from biopb_tensor_server.core.config import CacheConfig

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

        # The client's chunk_id is a proxy envelope wrapping the UPSTREAM chunk_id
        # (routed to the local source); resolve peels it and forwards inner verbatim.
        bounds = ChunkBounds(start=[0, 0], stop=[chunks[0], chunks[1]])
        inner = encode_chunk_id("img", bounds)  # upstream array_id
        chunk_id = encode_proxy_envelope(inner, "lab__img", adapter.content_version)

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


def test_resolve_forwards_inner_verbatim():
    """The proxy forwards the envelope's opaque inner to the upstream byte-for-byte
    -- no rewrite, no proxy sentinel -- proving it never couples to the upstream
    codec (biopb/biopb#178 W1)."""
    import numpy as np
    from biopb.tensor.ticket_pb2 import ChunkBounds
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
    from biopb_tensor_server.core.base import pack_chunk_batch
    from biopb_tensor_server.core.chunk import (
        encode_chunk_id,
        encode_proxy_envelope,
        wrap_content_version,
    )

    adapter = RemoteTensorAdapter(
        source_id="lab__img",
        upstream_location="grpc://localhost:1",  # never dialed
        upstream_source_id="img",
    )
    bounds = ChunkBounds(start=[0, 0], stop=[4, 4])
    # An upstream-versioned inner exercises the opacity: the proxy must forward it
    # (0xFF version wrapper included) untouched, never peeling or re-wrapping it.
    inner = wrap_content_version(encode_chunk_id("img", bounds), b"iat:42")
    env = encode_proxy_envelope(inner, "lab__img", b"iat:99")

    captured = {}

    def fake(upstream_chunk_id):
        captured["id"] = upstream_chunk_id
        return pack_chunk_batch(np.zeros((4, 4), dtype=np.uint8))

    adapter._upstream_record_batch = fake
    adapter.resolve_chunk_data(env, cache_manager=None)

    assert captured["id"] == inner  # forwarded byte-for-byte
    assert captured["id"][0] != 0xFE  # no proxy envelope sentinel leaked upstream


def test_seed_catalog_sets_content_version_from_indexed_at():
    """The upstream's register timestamp becomes the mirror's content_version, so a
    re-register re-namespaces the proxy chunk cache (biopb/biopb#178 W1)."""
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    adapter = RemoteTensorAdapter(
        source_id="lab__img",
        upstream_location="grpc://localhost:1",  # never dialed
        upstream_source_id="img",
    )
    assert adapter.content_version is None  # unseeded -> unversioned

    adapter.seed_catalog([], {}, True, None, "2026-07-19 12:00:00")
    assert adapter.content_version == b"iat:2026-07-19 12:00:00"

    # A re-register (a new indexed_at) moves the version.
    adapter.seed_catalog([], {}, True, None, "2026-07-19 13:30:00")
    assert adapter.content_version == b"iat:2026-07-19 13:30:00"

    # An upstream with no indexed_at leaves the proxy unversioned.
    adapter.seed_catalog([], {}, True, None, None)
    assert adapter.content_version is None


# --- bulk-seed the mirror catalog (biopb/biopb#266-A) ------------------------


def test_seed_catalog_short_circuits_catalog_surface_without_dialing():
    """A seeded proxy answers list_tensor_descriptors/get_metadata from the seed,
    localizing array_ids, without ever dialing the upstream."""
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    adapter = RemoteTensorAdapter(
        source_id="lab__img",
        upstream_location="grpc://localhost:1",  # never dialed
        upstream_source_id="img",
    )
    adapter.seed_catalog(
        [
            {
                "array_id": "img",  # upstream id -> localized to lab__img
                "dim_labels": ["y", "x"],
                "shape": [4, 4],
                "chunk_shape": [4, 4],
                "dtype": "uint8",
            },
            {
                "array_id": "img/A2",  # multi-field: seeded too (live path can't)
                "dim_labels": ["y", "x"],
                "shape": [2, 2],
                "chunk_shape": [2, 2],
                "dtype": "uint16",
            },
        ],
        {"ome": "meta"},
    )

    descs = adapter.list_tensor_descriptors()
    assert [d.array_id for d in descs] == ["lab__img", "lab__img/A2"]
    assert list(descs[0].shape) == [4, 4]
    assert descs[1].dtype == "uint16"
    assert adapter.get_metadata() == {"ome": "meta"}
    # the whole point: no upstream RPC was made
    assert adapter._client is None


def test_upstream_clients_are_pooled_per_endpoint(monkeypatch):
    """B1 (biopb/biopb#266): mirrored sources of one upstream share a single
    client; distinct (endpoint, token) get distinct clients."""
    import biopb.tensor as bt
    from biopb_tensor_server.adapters import remote_tensor as rt

    built = []

    class _FakeClient:
        def __init__(self, location, cache_bytes=0, token=None):
            built.append((location, token))

        def close(self):
            pass

    monkeypatch.setattr(bt, "TensorFlightClient", _FakeClient)
    rt._clear_client_pool()

    a = rt.RemoteTensorAdapter("lab__a", "grpc://up:1", "a")
    b = rt.RemoteTensorAdapter("lab__b", "grpc://up:1", "b")  # same endpoint+token
    assert a.client is b.client  # one shared connection
    assert built == [("grpc://up:1", None)]  # built exactly once

    c = rt.RemoteTensorAdapter("lab__c", "grpc://up:2", "c")  # other endpoint
    assert c.client is not a.client
    d = rt.RemoteTensorAdapter("lab__d", "grpc://up:1", "d", token="t")  # other token
    assert d.client is not a.client
    assert len(built) == 3


def test_mark_unreachable_evicts_shared_client(monkeypatch):
    """A failure evicts (and closes) the pooled client so the next access rebuilds
    it; another adapter keeps its own reference until its own next failure."""
    import biopb.tensor as bt
    from biopb_tensor_server.adapters import remote_tensor as rt

    closed = []

    class _FakeClient:
        def __init__(self, location, cache_bytes=0, token=None):
            pass

        def close(self):
            closed.append(self)

    monkeypatch.setattr(bt, "TensorFlightClient", _FakeClient)
    rt._clear_client_pool()

    a = rt.RemoteTensorAdapter("lab__a", "grpc://up:1", "a")
    b = rt.RemoteTensorAdapter("lab__b", "grpc://up:1", "b")
    shared = a.client
    assert b.client is shared

    a._mark_unreachable(RuntimeError("down"))
    assert a._client is None  # dropped this adapter's reference
    assert shared in closed  # evicted from the pool and closed
    assert b._client is shared  # b keeps its stale ref until its own next failure

    # the next access rebuilds a fresh shared client (not the closed one)
    assert a.client is not shared


def test_get_tensor_descriptor_served_from_seed_without_rpc():
    """B2 (biopb/biopb#266): the serve-path get_tensor_descriptor reads the
    seeded structural descriptor -- no get_descriptor RPC -- so a bulk-mirrored
    GetFlightInfo makes no upstream call for the descriptor. Falls back to a live
    fetch only when a tensor is not in the seed (covered by the e2e proxy tests,
    which register without seeding)."""
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    adapter = RemoteTensorAdapter(
        source_id="lab__img",
        upstream_location="grpc://localhost:1",  # never dialed
        upstream_source_id="img",
    )
    adapter.seed_catalog(
        [
            {
                "array_id": "img",
                "dim_labels": ["y", "x"],
                "shape": [4, 4],
                "chunk_shape": [4, 4],
                "dtype": "uint8",
            },
            {
                "array_id": "img/A2",
                "dim_labels": ["y", "x"],
                "shape": [2, 2],
                "chunk_shape": [2, 2],
                "dtype": "uint16",
            },
        ],
        {"ome": "meta"},
    )

    # default (first) tensor
    desc = adapter.get_tensor_descriptor()
    assert desc.array_id == "lab__img"
    assert list(desc.shape) == [4, 4]
    assert desc.dtype == "uint8"
    # lean like a native descriptor -- the local server fills these itself
    assert not desc.metadata_json
    assert not desc.pyramid

    # a specific field via the tensor-layer view, also from the seed
    view = adapter.get_tensor_adapter("lab__img/A2")
    fdesc = view.get_tensor_descriptor()
    assert fdesc.array_id == "lab__img/A2"
    assert list(fdesc.shape) == [2, 2]
    assert fdesc.dtype == "uint16"

    # neither the source adapter nor the view ever dialed the upstream
    assert adapter._client is None
    assert view._client is None


def test_seed_catalog_empty_metadata_normalizes_to_dict():
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    adapter = RemoteTensorAdapter(
        source_id="lab__u",
        upstream_location="grpc://localhost:1",
        upstream_source_id="u",
    )
    adapter.seed_catalog([], None)  # unresolved upstream source: no tensors
    assert adapter.list_tensor_descriptors() == []
    assert adapter.get_metadata() == {}
    assert adapter._client is None


def test_fetch_upstream_catalog_returns_rows_and_complete():
    from biopb_tensor_server.adapters.remote_tensor import fetch_upstream_catalog

    class _FakeClient:
        def query_sources(self, sql, format="records"):  # noqa: A002 - fakes the real client's public `format` signature
            # source_url is now fetched so the mirror can be treed by path (#297).
            assert "tensors" in sql and "source_url" in sql and format == "records"
            return [
                {
                    "source_id": "a",
                    "source_url": "file:///d/a.zarr",
                    "tensors": [],
                    "metadata_json": None,
                }
            ]

    rows, complete = fetch_upstream_catalog(_FakeClient(), "grpc://fake")
    assert complete is True
    assert rows[0]["source_url"] == "file:///d/a.zarr"


def test_fetch_upstream_catalog_none_on_no_sql_catalog():
    from biopb_tensor_server.adapters.remote_tensor import fetch_upstream_catalog

    class _FakeClient:
        def query_sources(self, sql, format="records"):  # noqa: A002 - fakes the real client's public `format` signature
            raise RuntimeError("no metadata DB")

    rows, complete = fetch_upstream_catalog(_FakeClient(), "grpc://fake")
    assert rows is None
    assert complete is False


def test_fallback_warning_names_the_upstream_from_its_location_argument(caplog):
    """The upstream is named from the caller's endpoint, not from the SDK client's
    private ``_location`` (biopb/biopb#529). The fake declares no such attribute,
    so a reintroduced probe degrades the warning to "?" and fails this test."""
    import logging

    from biopb_tensor_server.adapters.remote_tensor import list_upstream_source_ids

    class _FakeClient:
        def query_sources(self, sql, format="records"):  # noqa: A002 - fakes the real client's public `format` signature
            raise RuntimeError("no metadata DB")

        def list_sources(self):
            return {"a": object()}

    with caplog.at_level(logging.WARNING):
        ids, complete = list_upstream_source_ids(_FakeClient(), "grpc://lab:8815")

    assert (ids, complete) == (["a"], False)
    assert "grpc://lab:8815" in caplog.text


def test_upstream_expansion_does_not_mask_its_error_with_a_failing_close(monkeypatch):
    """A broken channel fails BOTH the upstream call and the cleanup ``close()``.

    The ``finally:`` used to call ``close()`` unguarded (behind a dead
    ``getattr(client, "close", None)`` probe that only covered an SDK too old to
    install), so the close error replaced the upstream one on the way out --
    losing the diagnosis exactly when it matters (biopb/biopb#529).
    """
    import biopb.tensor as biopb_tensor
    from biopb_tensor_server.core.config import SourceConfig, _discover_tensor_server

    class _BrokenClient:
        def __init__(self, *_a, **_k):
            pass

        def query_sources(self, *_a, **_k):
            raise RuntimeError("channel is dead")

        def list_sources(self):
            raise RuntimeError("upstream unreachable")

        def close(self):
            raise RuntimeError("close on a broken channel")

    monkeypatch.setattr(biopb_tensor, "TensorFlightClient", _BrokenClient)

    with pytest.raises(RuntimeError, match="upstream unreachable"):
        _discover_tensor_server(
            SourceConfig(url="grpc://host:8815", type="tensor-server"), None
        )


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_reconcile_bulk_seeds_adapters_without_per_source_rpc(simple_zarr_array):
    """A re-list mirrors every upstream source from ONE query_sources: the
    adapters are bulk-seeded (their live per-source fetch never runs) and the
    local catalog is populated from the same result."""
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.core.metadata_db import MetadataDatabase
    from biopb_tensor_server.sources.source_manager import SourceManager

    zarr_path, _, _ = simple_zarr_array
    upstream, _, _ = _db_upstream(zarr_path, ["img", "img2"])
    _serve(upstream)
    try:
        local_db = MetadataDatabase()
        proxy = TensorFlightServer("grpc://localhost:0", metadata_db=local_db)
        _serve(proxy)
        try:
            manager = SourceManager(
                server=proxy,
                registry=get_default_registry(),
                discovery_state=DiscoveryState(),
                watcher=None,
                monitored_dirs=set(),
                metadata_db=local_db,
                monitored_upstreams=[
                    SourceConfig(url=f"grpc://localhost:{upstream.port}", alias="lab")
                ],
            )

            manager._reconcile_upstreams()

            assert set(proxy.sources) == {"lab__img", "lab__img2"}
            for sid in ("lab__img", "lab__img2"):
                adapter = proxy.sources.get(sid)
                assert adapter._descriptors_cache is not None  # seeded, not live
                assert adapter._metadata_cache is not None
                assert adapter._client is None  # no per-source upstream dial
                # source_url mirrors the upstream path under the endpoint root, so
                # a browser trees it by filepath instead of a flat node (#297).
                assert adapter._source_url.startswith(
                    f"grpc://localhost:{upstream.port}/"
                )
                assert adapter._source_url.endswith(".zarr")

            # local catalog populated from the bulk seed
            rows = (
                local_db._get_connection()
                .execute("SELECT source_id FROM sources ORDER BY source_id")
                .fetchall()
            )
            assert [r[0] for r in rows] == ["lab__img", "lab__img2"]
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()


# --- unresolved-upstream interaction + resolve refresh (biopb/biopb#266-A) ---


class _CatalogRowAdapter:
    """Minimal adapter to seed a controllable upstream catalog row
    (data_resident / tensors / metadata)."""

    def __init__(self, source_id, tensors, resident, metadata=None):
        self.source_id = source_id
        self._source_url = f"/data/{source_id}"
        self._source_type = "zarr"
        self._tensors = tensors
        self._resident = resident
        self._metadata = metadata or {}

    def get_source_descriptor(self):
        from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor

        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=self._source_url,
            source_type=self._source_type,
            data_resident=self._resident,
            tensors=[TensorDescriptor(**t) for t in self._tensors],
        )

    def get_metadata(self):
        return self._metadata


def test_seed_catalog_carries_residency_and_detects_change():
    """An unresolved upstream source (data_resident=false, empty tensors) mirrors
    as non-resident; re-seeding reports change only when something differs, and an
    in-place resolution flips residency + tensors."""
    from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter

    adapter = RemoteTensorAdapter(
        source_id="lab__cloud",
        upstream_location="grpc://localhost:1",  # never dialed
        upstream_source_id="cloud",
    )

    changed = adapter.seed_catalog([], None, data_resident=False)
    assert changed is True
    assert adapter.list_tensor_descriptors() == []
    assert adapter.is_resident() is False  # unresolved upstream -> not resident
    assert adapter._client is None

    # identical re-seed -> no change (so the caller skips a redundant re-sync)
    assert adapter.seed_catalog([], None, data_resident=False) is False

    # in-place resolution upstream -> change detected, now resident with tensors
    changed = adapter.seed_catalog(
        [
            {
                "array_id": "cloud",
                "dim_labels": ["y", "x"],
                "shape": [4, 4],
                "chunk_shape": [4, 4],
                "dtype": "uint8",
            }
        ],
        {"ome": "m"},
        data_resident=True,
    )
    assert changed is True
    assert adapter.is_resident() is True
    assert [d.array_id for d in adapter.list_tensor_descriptors()] == ["lab__cloud"]
    assert adapter._client is None


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_reconcile_mirrors_unresolved_then_refreshes_on_resolve():
    """A mirror of an unresolved upstream source is non-resident with no tensors;
    when the upstream resolves it in place, the next re-list refreshes the local
    catalog row (residency + tensors) without a per-source RPC."""
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.adapters import get_default_registry
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState
    from biopb_tensor_server.core.metadata_db import MetadataDatabase
    from biopb_tensor_server.sources.source_manager import SourceManager

    up_db = MetadataDatabase()
    upstream = TensorFlightServer("grpc://localhost:0", metadata_db=up_db)
    up_db.sync_source_added(
        "cloud", _CatalogRowAdapter("cloud", tensors=[], resident=False)
    )
    _serve(upstream)
    try:
        local_db = MetadataDatabase()
        proxy = TensorFlightServer("grpc://localhost:0", metadata_db=local_db)
        _serve(proxy)
        try:
            manager = SourceManager(
                server=proxy,
                registry=get_default_registry(),
                discovery_state=DiscoveryState(),
                watcher=None,
                monitored_dirs=set(),
                metadata_db=local_db,
                monitored_upstreams=[
                    SourceConfig(url=f"grpc://localhost:{upstream.port}", alias="lab")
                ],
            )

            manager._reconcile_upstreams()

            def _row():
                return (
                    local_db._get_connection()
                    .execute(
                        "SELECT data_resident, tensors FROM sources "
                        "WHERE source_id='lab__cloud'"
                    )
                    .fetchone()
                )

            resident, tensors = _row()
            assert resident is False  # unresolved mirror, not advertised resident
            assert tensors == []
            assert proxy.sources.get("lab__cloud").is_resident() is False

            # upstream resolves the source in place (same source_id)
            up_db.sync_source_added(
                "cloud",
                _CatalogRowAdapter(
                    "cloud",
                    tensors=[
                        {
                            "array_id": "cloud",
                            "dim_labels": ["y", "x"],
                            "shape": [8, 8],
                            "chunk_shape": [8, 8],
                            "dtype": "uint8",
                        }
                    ],
                    resident=True,
                    metadata={"ome": "m"},
                ),
            )

            manager._reconcile_upstreams()

            resident, tensors = _row()
            assert resident is True  # refreshed from the bulk re-list
            assert len(tensors) == 1
            assert tensors[0]["array_id"] == "lab__cloud"  # localized
            assert proxy.sources.get("lab__cloud").is_resident() is True
        finally:
            proxy.shutdown()
    finally:
        upstream.shutdown()
