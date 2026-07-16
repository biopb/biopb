"""Unit and integration tests for the HTTP sidecar (http_server.py).

Unit tests use FastAPI TestClient with a mocked TensorFlightClient.
Integration tests spin up a real TensorFlightServer + ZarrAdapter.
"""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from biopb_tensor_server.serving.http_server import _request_array_id, create_app
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOKEN = "test-token-valid-1234"
_WRONG = "totally-wrong-token-xy"

# ---------------------------------------------------------------------------
# Stand-ins for protobuf descriptor objects
# ---------------------------------------------------------------------------


def _make_tensor_desc(
    array_id: str = "t0",
    shape=(4, 8, 16),
    dtype: str = "uint16",
    dim_labels=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        array_id=array_id,
        shape=list(shape),
        chunk_shape=[max(1, s // 2) for s in shape],
        dtype=dtype,
        dim_labels=list(dim_labels or ["z", "y", "x"]),
    )


def _make_source_desc(
    source_id: str = "src0",
    source_url: str = "/data/src0",
    tensors=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        source_id=source_id,
        source_url=source_url,
        source_type="zarr",
        metadata_json=None,
        tensors=tensors or [_make_tensor_desc()],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_mock_client(src_desc=None) -> MagicMock:
    """Return a MagicMock that satisfies the TensorFlightClient interface."""
    mc = MagicMock()
    src = src_desc or _make_source_desc()

    mc.list_sources.return_value = {src.source_id: src}
    mc.get_source_metadata.return_value = {"ome_ngff": {"version": "0.4"}}
    mc.cache_info.return_value = {"hits": 3, "misses": 1}

    # get_tensor → lazy array whose .compute() returns a numpy array
    arr = np.zeros(src.tensors[0].shape, dtype=src.tensors[0].dtype)
    lazy = MagicMock()
    lazy.compute.return_value = arr
    mc.get_tensor.return_value = lazy

    # _sources is accessed directly in the slice route for dim_labels
    mc._sources = {src.source_id: src}

    return mc


@pytest.fixture()
def auth_client():
    """TestClient backed by a mocked Flight client, token auth enabled."""
    mock_fc = _build_mock_client()
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(token=_TOKEN)
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, mock_fc


@pytest.fixture()
def dev_client():
    """TestClient in dev_mode (no auth required)."""
    mock_fc = _build_mock_client()
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(token=None)
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, mock_fc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _xbiopb(token: str) -> dict:
    return {"X-Biopb-Token": token}


# ===========================================================================
# Unit tests — health endpoints (unauthenticated)
# ===========================================================================


class TestHealthEndpoints:
    def test_livez_returns_200_without_auth(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/livez")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_readyz_returns_200_without_auth(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert "ready" in body
        assert "service" in body
        assert "version" in body

    def test_healthz_alias_matches_readyz(self, auth_client):
        tc, _ = auth_client
        r_health = tc.get("/healthz")
        r_ready = tc.get("/readyz")
        assert r_health.status_code == r_ready.status_code
        # Both should have same keys
        assert set(r_health.json().keys()) == set(r_ready.json().keys())

    def test_readyz_reports_dev_mode_false(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/readyz")
        assert r.json()["dev_mode"] is False

    def test_readyz_reports_dev_mode_true(self, dev_client):
        tc, _ = dev_client
        r = tc.get("/readyz")
        assert r.json()["dev_mode"] is True


# ===========================================================================
# Unit tests — authentication
# ===========================================================================


class TestAuthentication:
    def test_sources_requires_token(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources")
        assert r.status_code == 401

    def test_sources_wrong_token_bearer(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources", headers=_bearer(_WRONG))
        assert r.status_code == 401

    def test_sources_wrong_token_xbiopb(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources", headers=_xbiopb(_WRONG))
        assert r.status_code == 401

    def test_sources_valid_bearer_token(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources", headers=_bearer(_TOKEN))
        assert r.status_code == 200

    def test_sources_valid_xbiopb_token(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources", headers=_xbiopb(_TOKEN))
        assert r.status_code == 200

    def test_dev_mode_bypasses_auth(self, dev_client):
        tc, _ = dev_client
        r = tc.get("/api/sources")
        assert r.status_code == 200

    def test_diagnostics_requires_token(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/diagnostics")
        assert r.status_code == 401

    def test_slice_requires_token(self, auth_client):
        tc, _ = auth_client
        payload = {"source_id": "src0", "tensor_id": "t0"}
        r = tc.post("/api/slice", json=payload)
        assert r.status_code == 401


class TestCorsHeaders:
    def test_slice_exposes_shape_dtype_headers_for_browser(self, dev_client):
        tc, _ = dev_client
        payload = {"source_id": "src0", "tensor_id": "t0"}
        r = tc.post(
            "/api/slice",
            json=payload,
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        assert r.status_code == 200
        exposed = r.headers.get("access-control-expose-headers", "")
        exposed_lc = exposed.lower()
        assert "x-shape" in exposed_lc
        assert "x-dtype" in exposed_lc
        assert "x-dim-labels" in exposed_lc


# ===========================================================================
# Unit tests — sources endpoints
# ===========================================================================


class TestSourcesEndpoints:
    def test_list_sources_returns_list(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources", headers=_bearer(_TOKEN))
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["source_id"] == "src0"
        assert body[0]["source_url"] == "/data/src0"
        assert isinstance(body[0]["tensors"], list)

    def test_list_sources_tensor_fields(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources", headers=_bearer(_TOKEN))
        tensor = r.json()[0]["tensors"][0]
        assert tensor["array_id"] == "t0"
        assert tensor["shape"] == [4, 8, 16]
        assert tensor["dtype"] == "uint16"
        assert tensor["dim_labels"] == ["z", "y", "x"]

    def test_get_single_source(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources/src0", headers=_bearer(_TOKEN))
        assert r.status_code == 200
        assert r.json()["source_id"] == "src0"

    def test_get_missing_source_returns_404(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources/does-not-exist", headers=_bearer(_TOKEN))
        assert r.status_code == 404

    def test_get_source_metadata(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/sources/src0/metadata", headers=_bearer(_TOKEN))
        assert r.status_code == 200
        assert "ome_ngff" in r.json()

    def test_list_sources_calls_flight_client(self, auth_client):
        tc, mock_fc = auth_client
        tc.get("/api/sources", headers=_bearer(_TOKEN))
        mock_fc.list_sources.assert_called()


# ===========================================================================
# Unit tests — slice endpoint
# ===========================================================================


class TestSliceEndpoint:
    def _post_slice(self, tc, extra_headers=None, **kwargs):
        payload = {"source_id": "src0", "tensor_id": "t0", **kwargs}
        headers = {**_bearer(_TOKEN), **(extra_headers or {})}
        return tc.post("/api/slice", json=payload, headers=headers)

    def test_slice_returns_octet_stream(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc)
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]

    def test_slice_xshape_header_matches_array(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc)
        shape = [int(x) for x in r.headers["x-shape"].split(",")]
        assert shape == [4, 8, 16]

    def test_slice_xdtype_header(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc)
        assert r.headers["x-dtype"] == "uint16"

    def test_slice_xdimlabels_header(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc)
        labels = r.headers["x-dim-labels"].split(",")
        assert labels == ["z", "y", "x"]

    def test_slice_xdimlabels_qualified_catalog_bare_request(self):
        """Identity policy: the catalog descriptor carries the qualified
        array_id ("src0/t0"), but a browser may address the tensor by the bare
        field ("t0"). The best-effort dim-label lookup must still match and
        attach X-Dim-Labels (it is tolerant of both forms)."""
        qualified = _make_source_desc(
            tensors=[_make_tensor_desc(array_id="src0/t0", dim_labels=["z", "y", "x"])]
        )
        mock_fc = _build_mock_client(qualified)
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=_TOKEN)
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.post(
                    "/api/slice",
                    json={"source_id": "src0", "tensor_id": "t0"},
                    headers=_bearer(_TOKEN),
                )
        assert r.status_code == 200
        assert r.headers["x-dim-labels"].split(",") == ["z", "y", "x"]

    def test_slice_body_bytesize_matches_shape(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc)
        expected_bytes = 4 * 8 * 16 * 2  # shape * itemsize(uint16=2)
        assert len(r.content) == expected_bytes

    def test_slice_body_roundtrip_numpy(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc)
        arr = np.frombuffer(r.content, dtype="uint16").reshape(4, 8, 16)
        assert arr.shape == (4, 8, 16)
        assert np.all(arr == 0)  # fixture returns zeros

    def test_slice_normalizes_big_endian_uint16_bytes(self, auth_client):
        tc, mock_fc = auth_client

        expected = np.array(
            [[1, 256, 1024, 4095], [42, 512, 2048, 65535]],
            dtype=np.uint16,
        )
        # Simulate an adapter that returns big-endian uint16 payloads.
        be_arr = expected.astype(">u2", copy=False)
        lazy = MagicMock()
        lazy.compute.return_value = be_arr
        mock_fc.get_tensor.return_value = lazy

        r = self._post_slice(tc)
        assert r.status_code == 200
        assert r.headers["x-dtype"] == "uint16"

        arr = np.frombuffer(r.content, dtype=np.uint16).reshape(expected.shape)
        np.testing.assert_array_equal(arr, expected)

    def test_slice_with_range(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc, slice_start=[0, 0, 0], slice_stop=[2, 4, 8])
        assert r.status_code == 200
        # get_tensor should have been called with slice_hint
        _, mock_fc = auth_client
        call_kwargs = mock_fc.get_tensor.call_args
        assert call_kwargs is not None

    def test_slice_mismatched_start_stop_returns_422(self, auth_client):
        tc, _ = auth_client
        r = self._post_slice(tc, slice_start=[0, 0], slice_stop=[1, 2, 3])
        assert r.status_code == 422

    def test_slice_flight_error_returns_502(self, auth_client):
        tc, mock_fc = auth_client
        mock_fc.get_tensor.side_effect = RuntimeError("Flight connection lost")
        payload = {"source_id": "src0", "tensor_id": "t0"}
        r = tc.post("/api/slice", json=payload, headers=_bearer(_TOKEN))
        assert r.status_code == 502
        # Reset side effect for subsequent tests
        mock_fc.get_tensor.side_effect = None

    def test_slice_without_auth_returns_401(self, auth_client):
        tc, _ = auth_client
        r = tc.post("/api/slice", json={"source_id": "src0", "tensor_id": "t0"})
        assert r.status_code == 401

    def test_slice_passes_slice_hint_to_backend(self, auth_client):
        """Verify slice_hint IS passed to backend for server-side slicing."""
        tc, mock_fc = auth_client

        # Create a mock dask array
        mock_dask = MagicMock()
        mock_dask.compute.return_value = np.zeros((2, 4, 8), dtype="uint16")

        mock_fc.get_tensor.return_value = mock_dask

        r = self._post_slice(tc, slice_start=[0, 0, 0], slice_stop=[2, 4, 8])
        assert r.status_code == 200

        # Verify get_tensor was called with slice_hint (server-side slicing)
        call_kwargs = mock_fc.get_tensor.call_args.kwargs
        assert call_kwargs.get("slice_hint") is not None
        assert call_kwargs["slice_hint"] == (slice(0, 2), slice(0, 4), slice(0, 8))


# ===========================================================================
# Unit tests — chunk endpoint
# ===========================================================================


class TestChunkEndpoint:
    def _make_ticket_hex(self, chunk_id: bytes = b"test-chunk") -> str:
        """Create a hex-encoded TensorTicket string."""
        from biopb.tensor.ticket_pb2 import TensorTicket

        ticket = TensorTicket(chunk_id=chunk_id)
        return ticket.SerializeToString().hex()

    def test_chunk_returns_octet_stream(self, auth_client):
        tc, mock_fc = auth_client
        ticket_hex = self._make_ticket_hex()

        # Mock do_get to return a table
        import pyarrow as pa
        from biopb_tensor_server.core.base import pack_chunk_batch

        # do_get returns the unified binary chunk batch (biopb/biopb#293).
        batch = pack_chunk_batch(np.zeros((16, 16), dtype="uint16"))
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = pa.Table.from_batches([batch])

        mock_fc._client.do_get.return_value = mock_reader
        mock_fc._call_options = MagicMock()

        r = tc.get(
            f"/api/sources/src0/ticket/{ticket_hex}",
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]

    def test_chunk_xshape_header(self, auth_client):
        tc, mock_fc = auth_client
        ticket_hex = self._make_ticket_hex()

        import pyarrow as pa
        from biopb_tensor_server.core.base import pack_chunk_batch

        # do_get returns the unified binary chunk batch (biopb/biopb#293).
        batch = pack_chunk_batch(np.zeros((16, 16), dtype="uint16"))
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = pa.Table.from_batches([batch])

        mock_fc._client.do_get.return_value = mock_reader
        mock_fc._call_options = MagicMock()

        r = tc.get(
            f"/api/sources/src0/ticket/{ticket_hex}",
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 200
        shape = [int(x) for x in r.headers["x-shape"].split(",")]
        assert shape == [16, 16]

    def test_chunk_xdtype_header(self, auth_client):
        tc, mock_fc = auth_client
        ticket_hex = self._make_ticket_hex()

        import pyarrow as pa
        from biopb_tensor_server.core.base import pack_chunk_batch

        # do_get returns the unified binary chunk batch (biopb/biopb#293).
        batch = pack_chunk_batch(np.zeros((16, 16), dtype="uint16"))
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = pa.Table.from_batches([batch])

        mock_fc._client.do_get.return_value = mock_reader
        mock_fc._call_options = MagicMock()

        r = tc.get(
            f"/api/sources/src0/ticket/{ticket_hex}",
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 200
        assert r.headers["x-dtype"] == "uint16"

    def test_chunk_invalid_hex_returns_400(self, auth_client):
        tc, _ = auth_client
        r = tc.get(
            "/api/sources/src0/ticket/invalid_hex!",
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 400
        assert "Invalid ticket" in r.json()["detail"]

    def test_chunk_without_auth_returns_401(self, auth_client):
        tc, _ = auth_client
        ticket_hex = self._make_ticket_hex()
        r = tc.get(f"/api/sources/src0/ticket/{ticket_hex}")
        assert r.status_code == 401

    def test_chunk_flight_error_returns_502(self, auth_client):
        tc, mock_fc = auth_client
        ticket_hex = self._make_ticket_hex()

        import pyarrow.flight as flight

        mock_fc._client.do_get.side_effect = flight.FlightServerError("Chunk not found")
        mock_fc._call_options = MagicMock()

        r = tc.get(
            f"/api/sources/src0/ticket/{ticket_hex}",
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 502


# ===========================================================================
# Unit tests — diagnostics
# ===========================================================================


class TestDiagnostics:
    def test_diagnostics_returns_snapshot(self, auth_client):
        tc, _ = auth_client
        r = tc.get("/api/diagnostics", headers=_bearer(_TOKEN))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "connection_state" in body
        assert "dev_mode" in body

    def test_diagnostics_rate_limit(self, auth_client):
        tc, _ = auth_client
        h = _bearer(_TOKEN)
        r1 = tc.get("/api/diagnostics", headers=h)
        assert r1.status_code == 200
        r2 = tc.get("/api/diagnostics", headers=h)
        # Second request within 1s window should be rate-limited
        assert r2.status_code == 429

    def test_diagnostics_dev_mode_flag(self, dev_client):
        tc, _ = dev_client
        r = tc.get("/api/diagnostics")
        assert r.status_code == 200
        assert r.json()["dev_mode"] is True


# ===========================================================================
# Unit tests — _redact helper (indirect via diagnostics errors)
# ===========================================================================


class TestRedact:
    def test_redact_path_in_error(self, auth_client):
        tc, mock_fc = auth_client
        # Trigger an error containing a file path
        mock_fc.list_sources.side_effect = RuntimeError(
            "failed to open /home/user/secret/data.zarr"
        )
        tc.get("/api/sources", headers=_bearer(_TOKEN))
        mock_fc.list_sources.side_effect = None

        # The error should be recorded and redacted in diagnostics
        # (reset rate limit by using different session key)
        r = tc.get("/api/diagnostics", headers=_xbiopb(_TOKEN))
        assert r.status_code == 200
        body = r.json()
        last_msg = body.get("last_error_message", "")
        if last_msg:
            assert "/home/user/secret/data.zarr" not in last_msg
            assert "[REDACTED]" in last_msg


class TestRequestArrayId:
    """The sidecar must address tensors by array_id alone (identity policy), not
    the deprecated ``(source_id, tensor_id)`` pair. See biopb/biopb#75."""

    def test_qualified_array_id_passthrough(self):
        # TS client sends descriptor.array_id verbatim in tensor_id.
        assert _request_array_id("src", "src/fieldA") == "src/fieldA"

    def test_bare_field_is_qualified(self):
        # A browser/HTTP caller may send only the within-source field.
        assert _request_array_id("src", "fieldA") == "src/fieldA"

    def test_single_tensor_collapses_to_source_id(self):
        # Single-tensor source: array_id == source_id (no sentinel field).
        assert _request_array_id("src", "src") == "src"
        assert _request_array_id("src", "") == "src"
        assert _request_array_id("src", None) == "src"

    def test_slice_endpoint_uses_array_id_first_form(self, auth_client):
        """The slice endpoint passes array_id positionally, never the deprecated
        source_id=/tensor_id= keywords."""
        tc, mock_fc = auth_client
        lazy = MagicMock()
        lazy.compute.return_value = np.zeros((2, 4, 8), dtype="uint16")
        mock_fc.get_tensor.return_value = lazy

        r = tc.post(
            "/api/slice",
            json={"source_id": "src", "tensor_id": "src/fieldA"},
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 200
        call = mock_fc.get_tensor.call_args
        assert call.args[0] == "src/fieldA"
        assert "source_id" not in call.kwargs
        assert "tensor_id" not in call.kwargs


# ===========================================================================
# Integration tests — real TensorFlightServer + ZarrAdapter
# ===========================================================================


def _zarr_available() -> bool:
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestIntegration:
    """Integration tests: real Flight server ↔ HTTP sidecar ↔ TestClient."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        import zarr
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter

        # Create a small Zarr array
        zarr_path = str(tmp_path / "test.zarr")
        shape = (3, 32, 32)
        chunks = (1, 16, 16)
        rng = np.random.default_rng(0)
        data = rng.integers(0, 1000, shape, dtype="uint16")
        z = zarr.open_array(
            zarr_path, mode="w", shape=shape, chunks=chunks, dtype="uint16"
        )
        z[:] = data

        adapter = ZarrAdapter(z, "int-tensor", ["z", "y", "x"])

        # Bind to port 0 so the OS assigns a free port, avoiding flaky
        # "Address already in use" collisions when the suite runs back-to-back.
        server = TensorFlightServer("grpc://127.0.0.1:0")
        # Register under the same name as the adapter's array_id so that
        # the source_id returned by the server matches the tensor_id.
        server.register_source("int-tensor", adapter)

        t = threading.Thread(target=server.serve, daemon=True)
        t.start()
        time.sleep(0.5)  # allow server to bind

        self._flight_loc = f"grpc://localhost:{server.port}"
        self._shape = shape
        self._data = data
        self._server = server

        yield

        try:
            self._server.shutdown()
        except Exception:
            pass

    def _make_tc(self):
        app = create_app(
            flight_location=self._flight_loc,
            token=_TOKEN,
        )
        return TestClient(app, raise_server_exceptions=True)

    def test_integration_list_sources(self):
        with self._make_tc() as tc:
            r = tc.get("/api/sources", headers=_bearer(_TOKEN))
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        # source_id comes from DataSourceDescriptor returned by the server
        assert body[0]["source_id"] is not None
        assert body[0]["tensors"][0]["shape"] == list(self._shape)

    def test_integration_slice_roundtrip(self):
        with self._make_tc() as tc:
            # Discover the actual source_id/tensor_id from the server
            src_r = tc.get("/api/sources", headers=_bearer(_TOKEN))
            assert src_r.status_code == 200
            src = src_r.json()[0]
            source_id = src["source_id"]
            tensor_id = src["tensors"][0]["array_id"]
            r = tc.post(
                "/api/slice",
                json={"source_id": source_id, "tensor_id": tensor_id},
                headers=_bearer(_TOKEN),
            )
        assert r.status_code == 200
        shape_hdr = [int(x) for x in r.headers["x-shape"].split(",")]
        assert shape_hdr == list(self._shape)
        assert r.headers["x-dtype"] == "uint16"
        arr = np.frombuffer(r.content, dtype="uint16").reshape(self._shape)
        np.testing.assert_array_equal(arr, self._data)

    def test_integration_slice_subregion(self):
        with self._make_tc() as tc:
            src_r = tc.get("/api/sources", headers=_bearer(_TOKEN))
            src = src_r.json()[0]
            source_id = src["source_id"]
            tensor_id = src["tensors"][0]["array_id"]
            r = tc.post(
                "/api/slice",
                json={
                    "source_id": source_id,
                    "tensor_id": tensor_id,
                    "slice_start": [0, 0, 0],
                    "slice_stop": [1, 16, 16],
                },
                headers=_bearer(_TOKEN),
            )
        assert r.status_code == 200
        shape_hdr = [int(x) for x in r.headers["x-shape"].split(",")]
        assert shape_hdr == [1, 16, 16]

    def test_integration_health_no_auth(self):
        with self._make_tc() as tc:
            r = tc.get("/livez")
        assert r.status_code == 200


# ===========================================================================
# Unit tests — query_sources endpoint
# ===========================================================================


class TestQuerySourcesEndpoint:
    def test_query_sources_requires_token(self, auth_client):
        tc, _ = auth_client
        r = tc.post("/api/sources/query", json={"sql": "SELECT * FROM sources"})
        assert r.status_code == 401

    def test_query_sources_valid_request(self, auth_client):
        tc, mock_fc = auth_client

        # Mock query_sources to return an Arrow table
        import pyarrow as pa

        mock_table = pa.table(
            {
                "source_id": ["src0", "src1"],
                "source_type": ["zarr", "zarr"],
            }
        )
        mock_table = mock_table.replace_schema_metadata(
            {
                b"total_sources": "2",
                b"returned_sources": "2",
            }
        )
        mock_fc.query_sources.return_value = mock_table

        r = tc.post(
            "/api/sources/query",
            json={"sql": "SELECT source_id, source_type FROM sources"},
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 2
        assert body[0]["source_id"] == "src0"

    def test_query_sources_truncation_headers(self, auth_client):
        tc, mock_fc = auth_client

        import pyarrow as pa

        mock_table = pa.table(
            {
                "source_id": ["src0", "src1"],
            }
        )
        mock_table = mock_table.replace_schema_metadata(
            {
                b"total_sources": "100",
                b"returned_sources": "2",
            }
        )
        mock_fc.query_sources.return_value = mock_table

        r = tc.post(
            "/api/sources/query",
            json={"sql": "SELECT source_id FROM sources"},
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 200
        assert r.headers["X-Total-Sources"] == "100"
        assert r.headers["X-Returned-Sources"] == "2"
        assert r.headers["X-Truncated"] == "true"

    def test_query_sources_validation_error(self, auth_client):
        tc, mock_fc = auth_client
        mock_fc.query_sources.side_effect = ValueError("forbidden keyword: INSERT")

        r = tc.post(
            "/api/sources/query",
            json={"sql": "INSERT INTO sources VALUES ('evil')"},
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 400
        assert "forbidden keyword" in r.json()["detail"]

    def test_query_sources_flight_error(self, auth_client):
        tc, mock_fc = auth_client
        mock_fc.query_sources.side_effect = RuntimeError("Flight connection lost")

        r = tc.post(
            "/api/sources/query",
            json={"sql": "SELECT * FROM sources"},
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 502


class TestWindowsShutdownListener:
    """The graceful-stop listener the control supervisor drives on Windows."""

    def test_sentinel_path_matches_stop_side_contract(self):
        from pathlib import Path

        from biopb_tensor_server.serving.http_server import shutdown_sentinel_path

        # Must match DataPlaneSupervisor._win_stop_sentinel (the control writes it).
        # Fixed name (not pid-keyed) so stop and the daemon always agree.
        expected = Path.home() / ".local" / "share" / "biopb" / "tensor-server.stop"
        assert shutdown_sentinel_path() == expected

    def test_noop_off_windows(self):
        from biopb_tensor_server.serving.http_server import (
            _install_windows_shutdown_listener,
        )

        server = SimpleNamespace(should_exit=False)
        before = threading.active_count()
        with patch("biopb_tensor_server.serving.http_server.sys") as mock_sys:
            mock_sys.platform = "linux"
            _install_windows_shutdown_listener(server)  # must not raise
        assert threading.active_count() == before  # no watcher thread started
        assert server.should_exit is False


# ===========================================================================
# Admin routes — config read/write, status, restart (biopb/biopb#237)
# ===========================================================================


@pytest.fixture()
def admin_client(tmp_path):
    """Dev-mode TestClient wired with a config path and a health-reporting mock."""
    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        '{"server": {"host": "127.0.0.1", "port": 8815}, "keep_me": {"x": 1}}'
    )
    mock_fc = _build_mock_client()
    mock_fc.health_check.return_value = {
        "status": "SERVING",
        "source_count": 7,
        "writable": True,
        "uptime_seconds": 42,
        "full_scan_in_progress": True,
        "last_full_scan_finished_at": None,
    }
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(
            token=None,
            config_path=str(config_path),
            web_host="127.0.0.1",
            web_port=8814,
        )
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, config_path


@pytest.fixture()
def supervised_admin_client(tmp_path):
    """Like ``admin_client`` but control-owned (supervised=True): the admin
    self-restart must be refused so it can't race the control (biopb/biopb#418).
    """
    config_path = tmp_path / "biopb.json"
    config_path.write_text('{"server": {"host": "127.0.0.1", "port": 8815}}')
    mock_fc = _build_mock_client()
    mock_fc.health_check.return_value = {"status": "SERVING", "source_count": 1}
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(
            token=None,
            config_path=str(config_path),
            web_host="127.0.0.1",
            web_port=8814,
            supervised=True,
        )
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, config_path


class TestAdminConfigRoutes:
    def test_get_config_returns_path_config_and_schema(self, admin_client):
        tc, config_path = admin_client
        r = tc.get("/api/config")
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == str(config_path)
        assert body["config"]["server"]["port"] == 8815
        assert "properties" in body["schema"]

    def test_put_rejects_invalid_value_with_422_and_field_path(self, admin_client):
        tc, config_path = admin_client
        before = config_path.read_text()
        # downscale_factor of 1 is out of range (#34) -> schema rejects it.
        r = tc.put(
            "/api/config",
            json={"pyramid": {"downscale_factor": 1}},
            headers={"Sec-Fetch-Site": "same-origin"},
        )
        assert r.status_code == 422
        body = r.json()
        assert body["errors"]
        assert any("downscale_factor" in err["path"] for err in body["errors"])
        # Nothing written: disk untouched.
        assert config_path.read_text() == before

    def test_put_rejects_bad_case_insensitive_enum_the_schema_cannot_express(
        self, admin_client
    ):
        # log_level is a case-insensitive enum, so the published JSON Schema
        # emits no hard `enum` (it would reject valid differently-cased values).
        # The endpoint's semantic pass (validate_config_dict) must still reject a
        # value the server would refuse at load, so "the form accepted it" always
        # implies "the server will load it" (biopb/biopb#34).
        tc, config_path = admin_client
        before = config_path.read_text()
        r = tc.put(
            "/api/config",
            json={"server": {"log_level": "VERBOSE"}},
            headers={"Sec-Fetch-Site": "same-origin"},
        )
        assert r.status_code == 422
        body = r.json()
        assert any(err["path"] == ["server", "log_level"] for err in body["errors"])
        assert config_path.read_text() == before  # nothing written

    def test_put_malformed_section_returns_422_not_500(self, admin_client):
        # A wrong-typed section (a string where an object is expected) makes the
        # server's semantic validator's parse step raise while walking a non-dict.
        # The endpoint must degrade to a clean 422 (the JSON Schema's precise type
        # error), never a 500, and write nothing. Regression guard: the semantic
        # pass used to let that exception escape.
        tc, config_path = admin_client
        before = config_path.read_text()
        r = tc.put(
            "/api/config",
            json={"server": "not-a-dict"},
            headers={"Sec-Fetch-Site": "same-origin"},
        )
        assert r.status_code == 422
        body = r.json()
        assert body["errors"]
        # Schema's per-field error only -- no redundant root-level ([]) duplicate
        # from the semantic pass's structural fallback.
        assert not any(err["path"] == [] for err in body["errors"])
        assert config_path.read_text() == before  # nothing written

    def test_put_valid_saves_and_preserves_unsurfaced_keys(self, admin_client):
        import json

        tc, config_path = admin_client
        r = tc.put(
            "/api/config",
            json={"server": {"host": "127.0.0.1", "port": 9000}, "keep_me": {"x": 1}},
            headers={"Sec-Fetch-Site": "same-origin"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["saved"] is True and body["restart_required"] is True
        on_disk = json.loads(config_path.read_text())
        assert on_disk["server"]["port"] == 9000
        assert on_disk["keep_me"] == {"x": 1}  # unsurfaced key survived
        assert on_disk["$schema"] == "./biopb.schema.json"

    def test_put_blocks_cross_origin_without_token_header(self, admin_client):
        tc, config_path = admin_client
        before = config_path.read_text()
        r = tc.put(
            "/api/config",
            json={"server": {"port": 9000}},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert r.status_code == 403
        assert config_path.read_text() == before  # guarded before any write

    def test_put_cross_origin_allowed_with_xbiopb_token_header(self, admin_client):
        # A custom header a cross-origin browser fetch cannot set without a
        # (failing) CORS preflight is the same-origin proof, so it bypasses the
        # Sec-Fetch-Site check even on a cross-site request.
        tc, config_path = admin_client
        r = tc.put(
            "/api/config",
            json={"server": {"host": "127.0.0.1", "port": 9000}},
            headers={"Sec-Fetch-Site": "cross-site", "X-Biopb-Token": "anything"},
        )
        assert r.status_code == 200

    def test_put_cross_origin_allowed_with_authorization_header(self, admin_client):
        tc, config_path = admin_client
        r = tc.put(
            "/api/config",
            json={"server": {"host": "127.0.0.1", "port": 9000}},
            headers={"Sec-Fetch-Site": "cross-site", "Authorization": "Bearer x"},
        )
        assert r.status_code == 200


@pytest.fixture()
def admin_client_with_creds(tmp_path):
    """Admin TestClient whose config carries a credentials profile with secrets."""
    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 8815},
                "credentials": {
                    "profiles": [
                        {
                            "name": "aws-prod",
                            "storage_type": "s3",
                            "key": "AKIA-REAL",
                            "secret": "REAL-SECRET",
                            "region": "us-east-1",
                        }
                    ]
                },
            }
        )
    )
    mock_fc = _build_mock_client()
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(
            token=None,
            config_path=str(config_path),
            web_host="127.0.0.1",
            web_port=8814,
        )
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, config_path


class TestAdminConfigSecretRedaction:
    def test_get_masks_credential_secrets(self, admin_client_with_creds):
        from biopb_tensor_server.core.config import REDACTED_SENTINEL

        tc, _ = admin_client_with_creds
        prof = tc.get("/api/config").json()["config"]["credentials"]["profiles"][0]
        assert prof["key"] == REDACTED_SENTINEL
        assert prof["secret"] == REDACTED_SENTINEL
        assert prof["region"] == "us-east-1"  # non-secret passes through

    def test_put_with_redacted_sentinels_preserves_real_secret_on_disk(
        self, admin_client_with_creds
    ):
        import json as _json

        from biopb_tensor_server.core.config import REDACTED_SENTINEL

        tc, config_path = admin_client_with_creds
        # Round-trip the masked GET body back, editing only a non-secret field.
        body = tc.get("/api/config").json()["config"]
        body["credentials"]["profiles"][0]["region"] = "eu-west-1"
        r = tc.put("/api/config", json=body, headers={"Sec-Fetch-Site": "same-origin"})
        assert r.status_code == 200
        prof = _json.loads(config_path.read_text())["credentials"]["profiles"][0]
        assert prof["secret"] == "REAL-SECRET"  # not clobbered by the sentinel
        assert prof["key"] == "AKIA-REAL"
        assert prof["region"] == "eu-west-1"  # the genuine edit landed
        assert REDACTED_SENTINEL not in _json.dumps(prof)


class TestAdminStatusRoute:
    def test_status_merges_health_and_process_facts(self, admin_client):
        tc, config_path = admin_client
        r = tc.get("/api/admin/status")
        assert r.status_code == 200
        body = r.json()
        assert body["running"] is True
        assert body["health"] == "SERVING"
        assert body["source_count"] == 7
        assert body["full_scan_in_progress"] is True
        assert body["last_full_scan_finished_at"] is None
        assert body["config_path"] == str(config_path)
        assert isinstance(body["pid"], int)
        assert body["version"]
        # No token enforced ⇒ local mode; the admin UI keys the file chooser off
        # this (biopb/biopb#244).
        assert body["local"] is True

    def test_status_reports_not_supervised_by_default(self, admin_client):
        # A directly-launched `biopb server start` is not control-owned, so its
        # admin UI keeps the self-restart path (biopb/biopb#418).
        tc, _ = admin_client
        assert tc.get("/api/admin/status").json()["supervised"] is False

    def test_status_reports_supervised_when_control_owned(
        self, supervised_admin_client
    ):
        tc, _ = supervised_admin_client
        assert tc.get("/api/admin/status").json()["supervised"] is True

    def test_status_reports_not_local_when_token_enforced(self, tmp_path):
        # A token means remote mode; the admin UI then hides the file chooser.
        config_path = tmp_path / "biopb.json"
        config_path.write_text('{"server": {"host": "0.0.0.0", "port": 8815}}')
        mock_fc = _build_mock_client()
        mock_fc.health_check.return_value = {"status": "SERVING", "source_count": 1}
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=_TOKEN, config_path=str(config_path))
            with TestClient(app, raise_server_exceptions=True) as tc:
                assert (
                    tc.get("/api/admin/status", headers=_bearer(_TOKEN)).json()["local"]
                    is False
                )


class TestAdminBrowseRoute:
    def test_browse_lists_dirs_first_then_files(self, admin_client, tmp_path):
        tc, _ = admin_client
        base = tmp_path / "data"
        (base / "sub_b").mkdir(parents=True)
        (base / "sub_a").mkdir()
        (base / "img.tif").write_text("x")
        (base / "notes.txt").write_text("y")
        r = tc.get("/api/admin/browse", params={"path": str(base)})
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == str(base.resolve())
        assert body["parent"] == str(base.resolve().parent)
        assert body["truncated"] is False
        names = [(e["name"], e["is_dir"]) for e in body["entries"]]
        # Directories first (case-insensitive), then files.
        assert names == [
            ("sub_a", True),
            ("sub_b", True),
            ("img.tif", False),
            ("notes.txt", False),
        ]

    def test_browse_file_path_resolves_to_parent_dir(self, admin_client, tmp_path):
        tc, _ = admin_client
        f = tmp_path / "experiment.zarr"
        f.mkdir()
        (tmp_path / "peer.txt").write_text("z")
        # A typed *file* selection lists its containing directory so the chooser
        # keeps navigating instead of erroring.
        target = tmp_path / "peer.txt"
        r = tc.get("/api/admin/browse", params={"path": str(target)})
        assert r.status_code == 200
        assert r.json()["path"] == str(tmp_path.resolve())
        assert "experiment.zarr" in {e["name"] for e in r.json()["entries"]}

    def test_browse_defaults_to_home_when_no_path(self, admin_client):
        from pathlib import Path

        tc, _ = admin_client
        r = tc.get("/api/admin/browse")
        assert r.status_code == 200
        assert r.json()["path"] == str(Path.home().resolve())

    def test_browse_missing_dir_404(self, admin_client, tmp_path):
        tc, _ = admin_client
        r = tc.get(
            "/api/admin/browse",
            params={"path": str(tmp_path / "does" / "not" / "exist")},
        )
        # A non-existent path resolves to a non-existent parent → not a directory.
        assert r.status_code == 404

    def test_browse_unavailable_in_remote_mode(self, tmp_path):
        # Remote mode (token enforced): the FS-listing surface must not exist.
        config_path = tmp_path / "biopb.json"
        config_path.write_text('{"server": {"host": "0.0.0.0", "port": 8815}}')
        mock_fc = _build_mock_client()
        mock_fc.health_check.return_value = {"status": "SERVING"}
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=_TOKEN, config_path=str(config_path))
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.get(
                    "/api/admin/browse",
                    params={"path": str(tmp_path)},
                    headers=_bearer(_TOKEN),
                )
                assert r.status_code == 404


class TestCreateAppSupervisedFromEnv:
    def test_reads_supervised_from_env(self, monkeypatch):
        # The control marks the plane control-owned via BIOPB_DATA_PLANE_SUPERVISED
        # in the child env; create_app picks it up when not passed explicitly.
        monkeypatch.setenv("BIOPB_DATA_PLANE_SUPERVISED", "1")
        app = create_app(token=None)
        assert app.state.sidecar.supervised is True

    def test_defaults_unsupervised_without_env(self, monkeypatch):
        monkeypatch.delenv("BIOPB_DATA_PLANE_SUPERVISED", raising=False)
        app = create_app(token=None)
        assert app.state.sidecar.supervised is False


# ===========================================================================
# WebSocket render endpoint (/ws/render)
#
# The render pipeline itself is exercised by the HTTP /api/render path; these
# lock in the auth + message-dispatch behaviour that the refactor split out of
# the monolithic websocket handler (biopb/biopb#181).
# ===========================================================================


class TestWebSocketRender:
    def test_unknown_action_returns_error(self, dev_client):
        tc, _ = dev_client
        with tc.websocket_connect("/ws/render") as ws:
            ws.send_json({"action": "nope"})
            msg = ws.receive_json()
        assert msg["action"] == "error"
        assert msg["message"] == "Unknown action: nope"

    def test_invalid_params_returns_error(self, dev_client):
        tc, _ = dev_client
        with tc.websocket_connect("/ws/render") as ws:
            # RenderRequest requires source_id/tensor_id → validation error
            ws.send_json({"action": "render", "params": {}})
            msg = ws.receive_json()
        assert msg["action"] == "error"
        assert msg["message"].startswith("Invalid params")

    def test_missing_token_rejected(self, auth_client):
        from fastapi import WebSocketDisconnect

        tc, _ = auth_client
        # Token enforced (auth_client) and none supplied → server closes 4001.
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with tc.websocket_connect("/ws/render") as ws:
                ws.receive_json()
        assert excinfo.value.code == 4001

    def test_valid_query_token_accepts(self, auth_client):
        tc, _ = auth_client
        # Token via query param (browsers can't set WS headers).
        with tc.websocket_connect(f"/ws/render?token={_TOKEN}") as ws:
            ws.send_json({"action": "nope"})
            msg = ws.receive_json()
        assert msg["action"] == "error"
        assert msg["message"] == "Unknown action: nope"
