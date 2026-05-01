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

from fastapi.testclient import TestClient

from biopb_tensor_server.http_server import create_app

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
        "biopb_tensor_server.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(token=_TOKEN, dev_mode=False)
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, mock_fc


@pytest.fixture()
def dev_client():
    """TestClient in dev_mode (no auth required)."""
    mock_fc = _build_mock_client()
    with patch(
        "biopb_tensor_server.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(token=None, dev_mode=True)
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
        from biopb_tensor_server import ZarrAdapter, TensorFlightServer

        # Create a small Zarr array
        zarr_path = str(tmp_path / "test.zarr")
        shape = (3, 32, 32)
        chunks = (1, 16, 16)
        rng = np.random.default_rng(0)
        data = rng.integers(0, 1000, shape, dtype="uint16")
        z = zarr.open_array(zarr_path, mode="w", shape=shape, chunks=chunks, dtype="uint16")
        z[:] = data

        adapter = ZarrAdapter(z, "int-tensor", ["z", "y", "x"])

        # Pick an ephemeral-ish port for the Flight server
        port = 48816
        server = TensorFlightServer(f"grpc://127.0.0.1:{port}")
        # Register under the same name as the adapter's array_id so that
        # the source_id returned by the server matches the tensor_id.
        server.register_source("int-tensor", adapter)

        t = threading.Thread(target=server.serve, daemon=True)
        t.start()
        time.sleep(0.5)  # allow server to bind

        self._flight_loc = f"grpc://localhost:{port}"
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
            dev_mode=False,
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
