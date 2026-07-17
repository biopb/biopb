"""Client-side compact-grid read path (biopb/biopb#346, PR-C).

The client opts into the compact GetFlightInfo response and, when the server
omits the endpoint list, regenerates the (chunk_id, bounds) grid via
``expand_compact_grid`` instead of parsing endpoints. These tests pin the two
branches over a real Flight socket:

- a compact-capable server -> the reconstruction path runs and yields correct
  data (byte-identical to the source);
- a server that returns explicit endpoints (old server, or any non-regular /
  proxied plan) -> the client still parses them and never touches the compact
  path.
"""

import threading
import time

import biopb.tensor._session as session
import numpy as np
import pytest


def _serve(source_id, adapter):
    from biopb_tensor_server import TensorFlightServer

    server = TensorFlightServer("grpc://localhost:0")
    server.register_source(source_id, adapter)
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1)
    return server


def _chunked_zarr(tmp_path):
    import zarr

    path = str(tmp_path / "a.zarr")
    za = zarr.open_array(
        path, mode="w", shape=(200, 150), chunks=(64, 64), dtype="uint16"
    )
    # Unique value per chunk, so a misplaced chunk fails the equality check.
    for i in range(4):
        for j in range(3):
            za[i * 64 : min((i + 1) * 64, 200), j * 64 : min((j + 1) * 64, 150)] = (
                i * 10 + j + 1
            )
    return path, zarr.open_array(path, mode="r")[:]


class TestCompactClientRead:
    def test_compact_path_used_and_data_correct(self, monkeypatch, tmp_path):
        zarr = pytest.importorskip("zarr")
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server import ZarrAdapter

        path, expected = _chunked_zarr(tmp_path)
        server = _serve(
            "s", ZarrAdapter(zarr.open_array(path, mode="r"), "s", ["y", "x"])
        )

        calls = []
        real = session._build_dask_array_from_compact_grid
        monkeypatch.setattr(
            session,
            "_build_dask_array_from_compact_grid",
            lambda *a, **k: (calls.append(1), real(*a, **k))[1],
        )
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
            out = client.get_tensor("s").compute()
            assert np.array_equal(out, expected)
            assert calls, "expected the compact reconstruction path to be exercised"
            client.close()
        finally:
            server.shutdown()

    def test_sliced_compact_read_correct(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server import ZarrAdapter

        path, expected = _chunked_zarr(tmp_path)
        server = _serve(
            "s", ZarrAdapter(zarr.open_array(path, mode="r"), "s", ["y", "x"])
        )
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
            out = client.get_tensor(
                "s", slice_hint=(slice(10, 150), slice(20, 140))
            ).compute()
            assert np.array_equal(out, expected[10:150, 20:140])
            client.close()
        finally:
            server.shutdown()

    def test_explicit_endpoints_still_work(self, monkeypatch, tmp_path):
        """An old server (or any non-regular / proxied plan) returns the explicit
        endpoint list; the client must parse it and not reconstruct. Simulated by
        forcing the plan non-regular so the server keeps sending endpoints."""
        zarr = pytest.importorskip("zarr")
        import biopb_tensor_server.core.base as base
        from biopb.tensor.client import TensorFlightClient
        from biopb_tensor_server import ZarrAdapter

        real_plan = base._get_read_plan

        def non_regular(*args, **kwargs):
            plan = real_plan(*args, **kwargs)
            plan.regular_grid = False  # -> server sends explicit endpoints
            return plan

        monkeypatch.setattr(base, "_get_read_plan", non_regular)

        calls = []
        real = session._build_dask_array_from_compact_grid
        monkeypatch.setattr(
            session,
            "_build_dask_array_from_compact_grid",
            lambda *a, **k: (calls.append(1), real(*a, **k))[1],
        )

        path, expected = _chunked_zarr(tmp_path)
        server = _serve(
            "s", ZarrAdapter(zarr.open_array(path, mode="r"), "s", ["y", "x"])
        )
        try:
            client = TensorFlightClient(
                f"grpc://localhost:{server.port}", cache_bytes=10_000_000
            )
            out = client.get_tensor("s").compute()
            assert np.array_equal(out, expected)
            assert not calls, "compact path must not run when the server sent endpoints"
            client.close()
        finally:
            server.shutdown()
