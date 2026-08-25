"""Unit and integration tests for the HTTP sidecar (http_server.py).

Unit tests use FastAPI TestClient with a mocked TensorFlightClient.
Integration tests spin up a real TensorFlightServer + ZarrAdapter.
"""

import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow.flight as flight
import pytest
from biopb_tensor_server.serving.http_server import (
    _split_array_version,
    _tile_edge,
    _tile_levels,
    _versioned_array_id,
    create_app,
)
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
    array_id: str = "src0",
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

    def get_descriptor(array_id, **_kwargs):
        """What GetFlightInfo actually does, including the parts that bite.

        A bare source_id resolves to the source's DEFAULT tensor whatever the
        count -- the server defaults rather than refusing, so the refusal is the
        caller's job -- and an unknown field is a terminal ``FlightServerError``
        (pyarrow has no NOT_FOUND class).
        """
        for tensor in src.tensors:
            if tensor.array_id == array_id:
                return tensor
        if array_id == src.source_id:
            return src.tensors[0]
        raise flight.FlightServerError(f"Tensor not found: {array_id}")

    mc.get_descriptor.side_effect = get_descriptor
    mc.get_source_metadata.return_value = {"ome_ngff": {"version": "0.4"}}
    mc.cache_info.return_value = {"hits": 3, "misses": 1}
    # /readyz reports whatever Flight says, so the mock has to say something a
    # dict-shaped reader can parse -- a bare MagicMock's .get() returns another
    # MagicMock, which is neither SERVING nor a number.
    mc.health_check.return_value = {
        "status": "SERVING",
        "source_count": 1,
        "metadata_db_enabled": True,
        "full_scan_in_progress": False,
    }

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


class TestReadyzTracksBackend:
    """Readiness must follow the backend, not the traffic (biopb/biopb#755).

    Every case here failed before the fix: readiness peeked at a client only the
    token-protected data routes ever created, and the response was 200 whatever
    the verdict.
    """

    def test_readyz_connects_instead_of_waiting_for_traffic(self, auth_client):
        """A probe alone -- no prior data request -- must reach the backend."""
        tc, mock_fc = auth_client
        r = tc.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["backend_health"]["status"] == "SERVING"
        assert body["backend_error"] is None
        # It asked Flight rather than reporting from a cached connection state.
        assert mock_fc.health_check.called

    def test_readyz_503_when_backend_not_serving(self, auth_client):
        tc, mock_fc = auth_client
        mock_fc.health_check.return_value = {"status": "NOT_SERVING"}
        r = tc.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        assert body["status"] == "degraded"

    def test_readyz_503_and_names_the_reason_when_connect_fails(self):
        """``backend_health: null`` must no longer be ambiguous."""
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            side_effect=OSError("connection refused"),
        ):
            app = create_app(token=_TOKEN)
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        assert body["backend_health"] is None
        assert "connect failed" in body["backend_error"]
        assert "connection refused" in body["backend_error"]

    def test_readyz_goes_unready_when_a_live_backend_dies(self, auth_client):
        """The stale-``connected`` case: a past connect must not vouch for now.

        This is the false *positive* -- the one that hits during a data-plane
        restart, which is exactly when the admin page polls this endpoint.
        """
        tc, mock_fc = auth_client
        assert tc.get("/readyz").status_code == 200  # connected, healthy

        mock_fc.health_check.side_effect = OSError("backend went away")
        r = tc.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False
        assert body["backend_health"] is None
        assert "health check failed" in body["backend_error"]

    def test_livez_stays_traffic_free(self):
        """Liveness answers for the sidecar process alone -- no backend contact."""
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            side_effect=AssertionError("/livez must not touch the backend"),
        ):
            app = create_app(token=_TOKEN)
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.get("/livez")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


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
        payload = {"array_id": "src0"}
        r = tc.post("/api/slice", json=payload)
        assert r.status_code == 401


class TestCorsHeaders:
    def test_slice_exposes_shape_dtype_headers_for_browser(self, dev_client):
        tc, _ = dev_client
        payload = {"array_id": "src0"}
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
        assert tensor["array_id"] == "src0"
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
        payload = {"array_id": "src0", **kwargs}
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

    def test_slice_addresses_a_qualified_tensor_by_its_whole_array_id(self):
        """The catalog descriptor's array_id is the address, whole.

        The bare within-source field used to be tolerated and rejoined; it is
        now a 404, because a name that resolves one way for the read and another
        for the geometry is how the two came to disagree.
        """
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
                ok = tc.post(
                    "/api/slice",
                    json={"array_id": "src0/t0"},
                    headers=_bearer(_TOKEN),
                )
                bare = tc.post(
                    "/api/slice",
                    json={"array_id": "t0"},
                    headers=_bearer(_TOKEN),
                )
        assert ok.status_code == 200
        assert ok.headers["x-dim-labels"].split(",") == ["z", "y", "x"]
        assert bare.status_code == 404

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
        payload = {"array_id": "src0"}
        r = tc.post("/api/slice", json=payload, headers=_bearer(_TOKEN))
        assert r.status_code == 502
        # Reset side effect for subsequent tests
        mock_fc.get_tensor.side_effect = None

    def test_slice_without_auth_returns_401(self, auth_client):
        tc, _ = auth_client
        r = tc.post("/api/slice", json={"array_id": "src0"})
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
        from biopb_tensor_server.core.adapter_base import pack_chunk_batch

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
        from biopb_tensor_server.core.adapter_base import pack_chunk_batch

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
        from biopb_tensor_server.core.adapter_base import pack_chunk_batch

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


class TestSliceAddressing:
    """The slice route resolves an array_id the way the tile routes do.

    One resolution point, so a single id cannot mean two tensors depending on
    which route asked. See biopb/biopb#766, #75.
    """

    def test_it_reads_the_array_id_the_descriptor_came_from(self, auth_client):
        tc, mock_fc = auth_client
        lazy = MagicMock()
        lazy.compute.return_value = np.zeros((2, 4, 8), dtype="uint16")
        mock_fc.get_tensor.return_value = lazy

        r = tc.post("/api/slice", json={"array_id": "src0"}, headers=_bearer(_TOKEN))

        assert r.status_code == 200
        call = mock_fc.get_tensor.call_args
        assert call.args[0] == "src0"
        assert "source_id" not in call.kwargs
        assert "tensor_id" not in call.kwargs

    def test_a_bare_source_id_reads_the_tensor_the_server_bound(self):
        """Not a sidecar refusal: array_id policy belongs to the Flight server.

        It defaults a bare source_id to the source's default tensor, and the
        answer says which. What #75 was actually about is that the read must go
        to *that* tensor rather than re-deriving one -- so the assertion is on
        the id the read carries, not on a status code.
        """
        tensors = [
            _make_tensor_desc(array_id="multi/a"),
            _make_tensor_desc(array_id="multi/b"),
        ]
        mock_fc = _build_mock_client(
            _make_source_desc(source_id="multi", tensors=tensors)
        )
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=None)
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.post("/api/slice", json={"array_id": "multi"})

        assert r.status_code == 200
        assert mock_fc.get_tensor.call_args.args[0] == "multi/a"

    def test_the_old_pair_is_rejected_rather_than_guessed_at(self, auth_client):
        # 422 from the model: a body without array_id names no tensor, and
        # inventing one from source_id/tensor_id is the split this route left.
        tc, mock_fc = auth_client
        r = tc.post(
            "/api/slice",
            json={"source_id": "src0", "tensor_id": "t0"},
            headers=_bearer(_TOKEN),
        )
        assert r.status_code == 422
        mock_fc.get_tensor.assert_not_called()


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
            # The array_id a real server advertises is the whole address.
            src_r = tc.get("/api/sources", headers=_bearer(_TOKEN))
            assert src_r.status_code == 200
            array_id = src_r.json()[0]["tensors"][0]["array_id"]
            r = tc.post(
                "/api/slice",
                json={"array_id": array_id},
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
            array_id = src_r.json()[0]["tensors"][0]["array_id"]
            r = tc.post(
                "/api/slice",
                json={
                    "array_id": array_id,
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

        # Deliberately UNtagged. A mock that fabricates the truncation keys tests
        # a contract nothing guarantees -- `schema.metadata` is None on any table
        # nobody tagged -- and it hid a handler that dereferenced them blindly.
        # Serving the rows must not depend on the tags being there.
        import pyarrow as pa

        mock_table = pa.table(
            {
                "source_id": ["src0", "src1"],
                "source_type": ["zarr", "zarr"],
            }
        )
        assert mock_table.schema.metadata is None
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
        # No tags -> report what was actually returned, not a failure.
        assert r.headers["X-Total-Sources"] == "2"
        assert r.headers["X-Returned-Sources"] == "2"
        assert r.headers["X-Truncated"] == "false"

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
        from biopb import _locations
        from biopb_tensor_server.serving.http_server import shutdown_sentinel_path

        # Both this watcher and DataPlaneSupervisor._win_stop_sentinel (the control
        # writes it) bind to the one shared definition, so they cannot drift. Fixed
        # name (not pid-keyed) under the biopb state dir so stop and the daemon agree.
        assert shutdown_sentinel_path() == _locations.tensor_stop_sentinel()
        assert shutdown_sentinel_path().name == "tensor-server.stop"

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
# Unit tests — tile addressing (GET /api/tile_info, GET /api/tile)
# ===========================================================================


def _tile_source_desc(content_version: bytes | None = None) -> SimpleNamespace:
    """A realistic tiled tensor: TCZYX uint16, 1024x1024 plane, 512x512 chunks.

    ``content_version`` rides the TENSOR descriptor: it is a serving field,
    filled by GetFlightInfo and empty on a catalog listing entry.
    """
    td = SimpleNamespace(
        array_id="tiled/Image:0",
        shape=[1, 3, 16, 1024, 1024],
        chunk_shape=[1, 1, 1, 512, 512],
        dtype="uint16",
        dim_labels=["t", "c", "z", "y", "x"],
        content_version=content_version,
    )
    return SimpleNamespace(
        source_id="tiled",
        source_url="/data/tiled.zarr",
        source_type="zarr",
        metadata_json=None,
        tensors=[td],
    )


@pytest.fixture()
def tile_client():
    """TestClient over a tiled tensor; compute() yields one 512x512 plane."""
    src = _tile_source_desc()
    mock_fc = _build_mock_client(src)
    lazy = MagicMock()
    lazy.compute.return_value = np.zeros((1, 1, 1, 512, 512), dtype=np.uint16)
    mock_fc.get_tensor.return_value = lazy
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        app = create_app(token=None)
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc, mock_fc


class TestTileGeometry:
    """The grid maths, independent of any request."""

    def test_edge_equals_chunk_when_chunk_is_at_target(self):
        edge = _tile_edge([1, 1024, 1024], [1, 512, 512], 1, 2)
        assert edge == 512

    def test_edge_halves_a_large_chunk_until_it_nests(self):
        # 2048 chunk -> 512 tile, i.e. chunk / 2**2: still nests, never straddles.
        edge = _tile_edge([1, 4096, 4096], [1, 2048, 2048], 1, 2)
        assert edge == 512
        assert 2048 % edge == 0

    def test_edge_keeps_a_small_chunk_whole(self):
        assert _tile_edge([1, 256, 256], [1, 256, 256], 1, 2) == 256

    def test_edge_never_exceeds_the_plane(self):
        # An unchunked adapter advertises zeros; the tile must still fit.
        assert _tile_edge([1, 180, 183], [0, 0, 0], 1, 2) == 183

    def test_edge_stops_halving_on_an_odd_chunk(self):
        # 183 has no power-of-two factor; halving would break nesting.
        assert _tile_edge([1, 4000, 4000], [1, 183, 183], 1, 2) == 183

    def test_single_chunk_plane_still_gets_tiled(self):
        # A whole-plane chunk with an odd extent has no interior boundary to
        # straddle, so the transport target wins instead of yielding one 1411px
        # tile (which is tiling switched off for the largest images).
        assert _tile_edge([1, 1411, 1411, 3], [1, 1411, 1411, 3], 1, 2) == 512

    def test_levels_index_zero_is_full_resolution(self):
        levels = _tile_levels([1, 1024, 1024], 1, 2, 512)
        assert levels[0]["scale"] == 1
        assert (levels[0]["width"], levels[0]["height"]) == (1024, 1024)

    def test_levels_stop_once_the_plane_fits_one_tile(self):
        levels = _tile_levels([1, 1024, 1024], 1, 2, 512)
        assert len(levels) == 2
        assert levels[-1]["cols"] == 1 and levels[-1]["rows"] == 1

    def test_level_grid_covers_a_non_multiple_plane(self):
        # 1000 / 512 -> 2 tiles, the second one short. Must not drop the remainder.
        levels = _tile_levels([1, 1000, 1000], 1, 2, 512)
        assert levels[0]["cols"] == 2 and levels[0]["rows"] == 2


class TestTileInfoEndpoint:
    def test_reports_grid_and_levels(self, tile_client):
        tc, _ = tile_client
        r = tc.get("/api/tile_info/tiled")
        assert r.status_code == 200
        body = r.json()
        assert body["array_id"] == "tiled/Image:0"
        assert body["tile_size"] == 512
        assert body["dtype"] == "uint16"
        assert body["plane"] == {"y": 3, "x": 4, "s": None}
        assert [lv["level"] for lv in body["levels"]] == [0, 1]

    def test_reports_selectable_axes_by_label(self, tile_client):
        tc, _ = tile_client
        body = tc.get("/api/tile_info/tiled").json()
        assert body["selectable"] == {"t": 0, "c": 1, "z": 2}

    def test_tile_size_nests_in_chunk_shape(self, tile_client):
        tc, _ = tile_client
        body = tc.get("/api/tile_info/tiled").json()
        chunk_x = body["chunk_shape"][body["plane"]["x"]]
        assert chunk_x % body["tile_size"] == 0

    def test_unknown_source_is_404(self, tile_client):
        tc, _ = tile_client
        assert tc.get("/api/tile_info/nope").status_code == 404

    def test_requires_token_when_one_is_set(self, auth_client):
        tc, _ = auth_client
        assert tc.get("/api/tile_info/src0").status_code == 401


class TestTileEndpoint:
    def test_raw_tile_returns_bytes_and_shape_headers(self, tile_client):
        tc, _ = tile_client
        r = tc.get("/api/tile/tiled", params={"level": 0, "col": 0, "row": 0})
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert r.headers["X-Dtype"] == "uint16"
        assert r.headers["X-Tile-Size"] == "512"
        assert len(r.content) == 512 * 512 * 2

    def test_neighbouring_tiles_ask_for_adjacent_world_bounds(self, tile_client):
        tc, mock_fc = tile_client
        tc.get("/api/tile/tiled", params={"level": 0, "col": 1, "row": 0})
        kwargs = mock_fc.get_tensor.call_args.kwargs
        # Bounds are full-resolution world coords: col 1 starts one tile in.
        assert kwargs["slice_hint"][4] == slice(512, 1024)
        assert kwargs["slice_hint"][3] == slice(0, 512)
        assert kwargs["scale_hint"][3] == 1 and kwargs["scale_hint"][4] == 1
        # Leading axes collapse to a single index so the payload is one plane.
        assert kwargs["slice_hint"][0] == slice(0, 1)

    def test_a_coarser_level_covers_more_world_at_a_higher_scale(self, tile_client):
        tc, mock_fc = tile_client
        tc.get("/api/tile/tiled", params={"level": 1, "col": 0, "row": 0})
        kwargs = mock_fc.get_tensor.call_args.kwargs
        # One level-1 tile spans 2x the world and downsamples it back to 512px.
        assert kwargs["slice_hint"][4] == slice(0, 1024)
        assert kwargs["scale_hint"][3] == 2 and kwargs["scale_hint"][4] == 2

    def test_edge_tile_is_clipped_to_the_plane(self, tile_client):
        tc, mock_fc = tile_client
        tc.get("/api/tile/tiled", params={"level": 0, "col": 1, "row": 1})
        hint = mock_fc.get_tensor.call_args.kwargs["slice_hint"]
        assert hint[3].stop == 1024 and hint[4].stop == 1024

    def test_selection_indexes_the_labelled_axis(self, tile_client):
        tc, mock_fc = tile_client
        tc.get("/api/tile/tiled", params={"c": 2, "z": 7})
        hint = mock_fc.get_tensor.call_args.kwargs["slice_hint"]
        assert hint[1] == slice(2, 3)  # c
        assert hint[2] == slice(7, 8)  # z

    def test_tile_outside_the_level_grid_is_404(self, tile_client):
        tc, _ = tile_client
        r = tc.get("/api/tile/tiled", params={"level": 0, "col": 99, "row": 0})
        assert r.status_code == 404

    def test_row_outside_the_grid_is_404_too(self, tile_client):
        tc, _ = tile_client
        assert tc.get("/api/tile/tiled", params={"row": 99}).status_code == 404

    def test_a_level_the_tensor_does_not_have_is_404(self, tile_client):
        """An unadvertised level must not be served, at any grid position.

        `col`/`row` cannot catch this: the old grid check was
        `row * edge * 2**level >= height`, which at (0,0) is `0 >= height` --
        false for every level. So tile (0,0) was reachable at any depth.
        """
        tc, mock_fc = tile_client
        n_levels = len(tc.get("/api/tile_info/tiled").json()["levels"])
        for level in (n_levels, n_levels + 3, 17, 24):
            r = tc.get("/api/tile/tiled", params={"level": level})
            assert r.status_code == 404, f"level {level} was served"
            assert "does not exist" in r.json()["detail"]

    def test_an_over_deep_level_never_reaches_the_backend(self, tile_client):
        """The reason this is a 404 and not a curiosity.

        `scale_hint` is honoured down into `downsample_block`, which edge-pads
        its input up to a multiple of the scale factor. Level 17 on a 512px
        plane therefore asks the data plane to allocate and write a 65536x65536
        array -- measured: level 13 already pads to 8192x8192, and the cost
        scales with the square. Rejecting before `get_tensor` is what keeps one
        query parameter from sizing an allocation in a shared backend process.
        """
        tc, mock_fc = tile_client
        before = mock_fc.get_tensor.call_count
        assert tc.get("/api/tile/tiled", params={"level": 17}).status_code == 404
        assert mock_fc.get_tensor.call_count == before

    def test_the_advertised_grid_is_exactly_what_is_servable(self, tile_client):
        """tile_info and /api/tile must agree; they used to derive it twice."""
        tc, _ = tile_client
        info = tc.get("/api/tile_info/tiled").json()
        for lv in info["levels"]:
            args = {"level": lv["level"], "col": lv["cols"] - 1, "row": lv["rows"] - 1}
            assert tc.get("/api/tile/tiled", params=args).status_code == 200
            # One past each edge of the advertised grid is gone.
            assert (
                tc.get(
                    "/api/tile/tiled", params={**args, "col": lv["cols"]}
                ).status_code
                == 404
            )
            assert (
                tc.get(
                    "/api/tile/tiled", params={**args, "row": lv["rows"]}
                ).status_code
                == 404
            )

    def test_a_bad_tile_404s_even_holding_a_matching_etag(self, tile_client):
        # Validation runs before the revalidation short-circuit, so a stale or
        # forged ETag cannot turn a nonexistent tile into a cheap 304.
        tc, _ = tile_client
        etag = tc.get("/api/tile/tiled").headers["ETag"]
        r = tc.get(
            "/api/tile/tiled",
            params={"level": 17},
            headers={"If-None-Match": etag},
        )
        assert r.status_code == 404

    def test_selection_out_of_range_is_422(self, tile_client):
        tc, _ = tile_client
        r = tc.get("/api/tile/tiled", params={"c": 99})
        assert r.status_code == 422

    def test_bad_format_is_rejected(self, tile_client):
        tc, _ = tile_client
        assert tc.get("/api/tile/tiled", params={"fmt": "gif"}).status_code == 422

    def test_response_is_cacheable(self, tile_client):
        tc, _ = tile_client
        r = tc.get("/api/tile/tiled")
        assert "max-age" in r.headers["Cache-Control"]
        assert r.headers["ETag"].startswith('"')

    def test_cache_is_private_never_public(self, tile_client):
        """`public` on a token-authenticated response is a shared-cache bypass.

        The URL carries no token (auth is a header, so rotation does not bust the
        cache), and RFC 9111 §3.5 lets a shared cache reuse a response to an
        authenticated request for *another* request when the response says
        `public` / `s-maxage` / `must-revalidate`. With no token in the cache key
        that other request can be an unauthenticated one, so an nginx
        proxy_cache or CDN in front of a `--remote` deployment would serve tiles
        having checked the token once, for someone else.
        """
        tc, _ = tile_client
        cc = tc.get("/api/tile/tiled").headers["Cache-Control"]
        assert "private" in cc
        for shared in ("public", "s-maxage", "must-revalidate"):
            assert shared not in cc, f"{shared!r} re-opens the shared-cache bypass"

    def test_varies_on_the_credential(self, tile_client):
        tc, _ = tile_client
        # Belt-and-braces: a cache that stores it anyway keys on the token
        # rather than colliding entries across users.
        assert "Authorization" in tc.get("/api/tile/tiled").headers["Vary"]

    def test_a_304_is_no_more_shareable_than_a_200(self, tile_client):
        # The revalidation path reuses the same header dict; if it ever stops
        # doing so, the cheap response is the one that leaks.
        tc, _ = tile_client
        etag = tc.get("/api/tile/tiled").headers["ETag"]
        r = tc.get("/api/tile/tiled", headers={"If-None-Match": etag})
        assert r.status_code == 304
        assert "private" in r.headers["Cache-Control"]
        assert "public" not in r.headers["Cache-Control"]
        assert "Authorization" in r.headers["Vary"]

    def test_matching_etag_revalidates_to_304_without_reading(self, tile_client):
        tc, mock_fc = tile_client
        etag = tc.get("/api/tile/tiled").headers["ETag"]
        before = mock_fc.get_tensor.call_count
        r = tc.get("/api/tile/tiled", headers={"If-None-Match": etag})
        assert r.status_code == 304
        assert r.content == b""
        assert mock_fc.get_tensor.call_count == before  # no backend read

    def test_etag_distinguishes_tiles(self, tile_client):
        tc, _ = tile_client
        a = tc.get("/api/tile/tiled", params={"col": 0}).headers["ETag"]
        b = tc.get("/api/tile/tiled", params={"col": 1}).headers["ETag"]
        assert a != b

    def test_etag_ignores_appearance_parameters(self, tile_client):
        # Contrast is applied client-side, so it must not fragment the cache --
        # and these parameters are no longer declared at all, so Starlette drops
        # them before the handler sees them.
        tc, _ = tile_client
        a = tc.get("/api/tile/tiled", params={"lo": 1}).headers["ETag"]
        b = tc.get("/api/tile/tiled", params={"lo": 5, "color": "green"}).headers[
            "ETag"
        ]
        assert a == b

    def test_a_rendered_tile_is_refused_not_silently_served_raw(self, tile_client):
        # The withdrawn form has to fail loudly: answering raw bytes to a caller
        # that asked for a PNG is the silent-wrong-content failure `sel` exists
        # to prevent.
        tc, mock_fc = tile_client
        before = mock_fc.get_tensor.call_count
        for bad in ("png", "jpeg"):
            r = tc.get("/api/tile/tiled", params={"fmt": bad})
            assert r.status_code == 410, r.text
            assert "fmt=raw" in r.json()["detail"]
        assert mock_fc.get_tensor.call_count == before


# ===========================================================================
# Unit tests — cancellation (client hung up before the read ran)
# ===========================================================================


async def _disconnected(self) -> bool:
    return True


class TestCancellation:
    """A disconnected caller must cost no backend read."""

    def test_tile_skips_the_read_and_answers_499(self, tile_client):
        tc, mock_fc = tile_client
        before = mock_fc.get_tensor.call_count
        with patch("starlette.requests.Request.is_disconnected", _disconnected):
            r = tc.get("/api/tile/tiled")
        assert r.status_code == 499
        assert mock_fc.get_tensor.call_count == before

    def test_slice_skips_the_read_and_answers_499(self, dev_client):
        tc, mock_fc = dev_client
        before = mock_fc.get_tensor.call_count
        with patch("starlette.requests.Request.is_disconnected", _disconnected):
            r = tc.post("/api/slice", json={"array_id": "src0"})
        assert r.status_code == 499
        assert mock_fc.get_tensor.call_count == before

    def test_cancellations_are_counted_in_diagnostics(self, tile_client):
        tc, _ = tile_client
        # Read the counter off the context, not /api/diagnostics: that route is
        # rate limited to 1 req/s per session, so a before/after pair 429s.
        diag = tc.app.state.sidecar.diag
        before = diag.cancelled
        with patch("starlette.requests.Request.is_disconnected", _disconnected):
            tc.get("/api/tile/tiled")
        assert diag.cancelled == before + 1
        assert diag.snapshot(dev_mode=True)["cancelled_reads"] == before + 1

    def test_a_connected_client_is_unaffected(self, tile_client):
        tc, mock_fc = tile_client
        before = mock_fc.get_tensor.call_count
        assert tc.get("/api/tile/tiled").status_code == 200
        assert mock_fc.get_tensor.call_count == before + 1


# ===========================================================================
# Unit tests — tile selection is checked against the axes that exist
# ===========================================================================


@contextmanager
def _tile_client_for(dim_labels, shape):
    """A tile client over a tensor with the given axis labels."""
    td = _make_tensor_desc(
        array_id="s/Image:0", shape=shape, dtype="uint16", dim_labels=dim_labels
    )
    src = _make_source_desc(source_id="s", tensors=[td])
    mock_fc = _build_mock_client(src)
    plane = np.zeros([1] * (len(shape) - 2) + [8, 8], dtype=np.uint16)
    lazy = MagicMock()
    lazy.compute.return_value = plane
    mock_fc.get_tensor.return_value = lazy
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        with TestClient(create_app(token=None), raise_server_exceptions=True) as tc:
            yield tc, mock_fc


def _slice_hint_bounds(mock_fc):
    """The (start, stop) vectors of the last backend read, per axis."""
    hint = mock_fc.get_tensor.call_args.kwargs["slice_hint"]
    return [sl.start for sl in hint], [sl.stop for sl in hint]


class TestTileSelectionValidation:
    """`t`/`z`/`c` must be checked against the axis they name, not just `ge=0`.

    The loop that built the slices iterated *axes*, so a parameter naming an
    axis the tensor does not have was never visited and never validated.
    """

    def test_nonexistent_axis_with_a_nonzero_index_is_422(self):
        with _tile_client_for(["y", "x"], [512, 512]) as (tc, _):
            for params in ({"t": 7}, {"z": 99}, {"c": 12345}):
                r = tc.get("/api/tile/s", params=params)
                assert r.status_code == 422, f"{params} was served"
                assert "no" in r.json()["detail"]

    def test_nonexistent_axis_at_index_zero_is_fine(self):
        # Every client sends a full selection; index 0 is the correct default
        # and must not become an error.
        with _tile_client_for(["y", "x"], [512, 512]) as (tc, _):
            assert (
                tc.get("/api/tile/s", params={"t": 0, "z": 0, "c": 0}).status_code
                == 200
            )

    def test_an_ignored_parameter_cannot_fragment_the_cache(self):
        """Identical bytes must not sit under unbounded distinct ETags.

        `t` is an unbounded non-negative int, so when it was silently dropped
        every value minted a fresh cache entry for the very same tile.
        """
        with _tile_client_for(["y", "x"], [512, 512]) as (tc, _):
            base = tc.get("/api/tile/s").headers["ETag"]
            assert tc.get("/api/tile/s", params={"t": 0}).headers["ETag"] == base
            assert tc.get("/api/tile/s", params={"t": 9}).status_code == 422

    def test_axes_are_validated_independently(self):
        # The nastiest shape: one selection half-checked, because `c` exists
        # and `t` does not.
        with _tile_client_for(["c", "y", "x"], [3, 512, 512]) as (tc, _):
            assert tc.get("/api/tile/s", params={"c": 2}).status_code == 200
            assert tc.get("/api/tile/s", params={"c": 3}).status_code == 422
            assert tc.get("/api/tile/s", params={"t": 5}).status_code == 422
            assert tc.get("/api/tile/s", params={"z": 5}).status_code == 422

    def test_existing_axes_keep_their_extent_check(self):
        with _tile_client_for(["t", "c", "z", "y", "x"], [1, 3, 16, 512, 512]) as (
            tc,
            _,
        ):
            assert tc.get("/api/tile/s", params={"z": 15}).status_code == 200
            assert tc.get("/api/tile/s", params={"z": 16}).status_code == 422
            # Extents are full-resolution, which holds at every level because
            # scale_hint is 1 on non-plane axes.
            assert (
                tc.get("/api/tile/s", params={"z": 15, "level": 1}).status_code == 200
            )

    def test_a_rejected_selection_never_reaches_the_backend(self):
        with _tile_client_for(["y", "x"], [512, 512]) as (tc, mock_fc):
            before = mock_fc.get_tensor.call_count
            assert tc.get("/api/tile/s", params={"c": 4}).status_code == 422
            assert mock_fc.get_tensor.call_count == before


class TestTileInfoUnnamedAxes:
    """Axes t/z/c cannot name are published, so a client knows to use `sel`."""

    def test_no_unnamed_axes_on_a_fully_named_tensor(self):
        with _tile_client_for(["t", "c", "z", "y", "x"], [1, 3, 16, 512, 512]) as (
            tc,
            _,
        ):
            assert tc.get("/api/tile_info/s").json()["sel_axes"] == []

    def test_an_unlabelled_axis_is_reported(self):
        # Reachable through `sel`, but with no semantic title -- which is what
        # this list tells the client, so it shows "pos" rather than inventing Z.
        with _tile_client_for(["pos", "c", "y", "x"], [5, 3, 512, 512]) as (tc, _):
            assert tc.get("/api/tile_info/s").json()["sel_axes"] == [
                {"axis": 0, "label": "pos", "extent": 5}
            ]

    def test_a_singleton_axis_is_not_reported(self):
        # Extent 1 is not navigable: index 0 is the only index.
        with _tile_client_for(["pos", "c", "y", "x"], [1, 3, 512, 512]) as (tc, _):
            assert tc.get("/api/tile_info/s").json()["sel_axes"] == []

    def test_a_duplicate_label_is_reported(self):
        # labeled_axis_index takes the first match, so the second C has no name
        # of its own even though it is labelled.
        with _tile_client_for(["c", "c", "y", "x"], [2, 3, 512, 512]) as (tc, _):
            assert tc.get("/api/tile_info/s").json()["sel_axes"] == [
                {"axis": 1, "label": "c", "extent": 3}
            ]

    def test_a_tiff_sequence_axis_is_reported(self):
        # The case this was built for: 155 single-page TIFFs stacked on an
        # opaque file axis. Before `sel` this was a one-frame tensor to every
        # tiled client.
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            info = tc.get("/api/tile_info/s").json()
            assert info["selectable"] == {"t": None, "z": None, "c": None}
            assert info["sel_axes"] == [{"axis": 0, "label": "i", "extent": 155}]


class TestTilePositionalSelection:
    """`sel=<axis>:<index>` reaches an axis t/z/c cannot name."""

    def test_an_unnamed_axis_is_selectable(self):
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, mock_fc):
            assert tc.get("/api/tile/s", params={"sel": "0:154"}).status_code == 200
            start, stop = _slice_hint_bounds(mock_fc)
            assert (start[0], stop[0]) == (154, 155)

    def test_the_index_is_range_checked(self):
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            assert tc.get("/api/tile/s", params={"sel": "0:155"}).status_code == 422

    def test_an_axis_the_tensor_lacks_is_refused(self):
        # No index-0 exemption: `sel` is never a default, so naming a
        # nonexistent axis is always a client mistake worth reporting.
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            assert tc.get("/api/tile/s", params={"sel": "9:0"}).status_code == 422

    def test_a_plane_axis_is_refused(self):
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            assert tc.get("/api/tile/s", params={"sel": "1:0"}).status_code == 422
            assert tc.get("/api/tile/s", params={"sel": "2:0"}).status_code == 422

    def test_a_named_axis_must_use_its_name(self):
        # Refused even though the two agree: one axis, two spellings, two cache
        # keys for one tile.
        with _tile_client_for(["t", "c", "z", "y", "x"], [4, 3, 16, 512, 512]) as (
            tc,
            _,
        ):
            assert tc.get("/api/tile/s", params={"sel": "0:2"}).status_code == 422

    def test_a_malformed_sel_is_refused(self):
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            for bad in ("0", "z:1", "0:", "-1:0", "0:1:2", ""):
                assert tc.get("/api/tile/s", params={"sel": bad}).status_code == 422, (
                    bad
                )

    def test_the_same_axis_twice_is_refused(self):
        with _tile_client_for(["pos", "c", "y", "x"], [5, 3, 512, 512]) as (tc, _):
            resp = tc.get("/api/tile/s", params=[("sel", "0:1"), ("sel", "0:2")])
            assert resp.status_code == 422

    def test_sel_and_a_named_axis_compose(self):
        with _tile_client_for(["pos", "c", "y", "x"], [5, 3, 512, 512]) as (
            tc,
            mock_fc,
        ):
            resp = tc.get("/api/tile/s", params={"sel": "0:4", "c": 2})
            assert resp.status_code == 200
            start, stop = _slice_hint_bounds(mock_fc)
            assert (start[0], stop[0]) == (4, 5)
            assert (start[1], stop[1]) == (2, 3)

    def test_a_rejected_sel_never_reaches_the_backend(self):
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, mock_fc):
            before = mock_fc.get_tensor.call_count
            assert tc.get("/api/tile/s", params={"sel": "0:999"}).status_code == 422
            assert mock_fc.get_tensor.call_count == before

    def test_the_etag_follows_the_plane_not_the_spelling(self):
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            first = tc.get("/api/tile/s", params={"sel": "0:7"}).headers["ETag"]
            same = tc.get("/api/tile/s", params={"sel": "0:7"}).headers["ETag"]
            other = tc.get("/api/tile/s", params={"sel": "0:8"}).headers["ETag"]
            assert first == same
            assert first != other

    def test_an_ignored_parameter_does_not_vary_the_etag(self):
        # z=0 on a tensor with no z axis resolves to nothing; it must not mint a
        # second cache entry for the same tile.
        with _tile_client_for(["i", "y", "x"], [155, 1024, 1344]) as (tc, _):
            plain = tc.get("/api/tile/s", params={"sel": "0:7"}).headers["ETag"]
            noisy = tc.get(
                "/api/tile/s", params={"sel": "0:7", "z": 0, "t": 0}
            ).headers["ETag"]
            assert plain == noisy


# ===========================================================================
# Unit tests — tiles are addressed by array_id alone
# ===========================================================================


@contextmanager
def _multi_tensor_client():
    """A source with two tensors, so a bare source_id is ambiguous."""
    tensors = [
        _make_tensor_desc(
            array_id="multi/Image:0",
            shape=[1, 1, 1, 512, 512],
            dtype="uint16",
            dim_labels=["T", "C", "Z", "Y", "X"],
        ),
        _make_tensor_desc(
            array_id="multi/Image:1",
            shape=[1, 1, 1, 2048, 2048],
            dtype="uint16",
            dim_labels=["T", "C", "Z", "Y", "X"],
        ),
    ]
    src = _make_source_desc(source_id="multi", tensors=tensors)
    mock_fc = _build_mock_client(src)
    lazy = MagicMock()
    lazy.compute.return_value = np.zeros((1, 1, 1, 512, 512), dtype=np.uint16)
    mock_fc.get_tensor.return_value = lazy
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        with TestClient(create_app(token=None), raise_server_exceptions=True) as tc:
            yield tc, mock_fc


class TestTileGridComesFromTheDescribedTensor:
    """The tile grid is read from GetFlightInfo, never from the source listing.

    A source listing is structural and carries an empty ``chunk_shape``
    (biopb/biopb#812); the tile edge nests inside the transfer grid, so it has to
    come from describing the tensor. Here the two surfaces disagree on purpose:
    the listed entry has no grid at all, and only ``get_descriptor`` fills one.
    """

    @staticmethod
    def _client():
        listed = _make_tensor_desc(
            array_id="s",
            shape=[1, 1, 1, 512, 512],
            dtype="uint16",
            dim_labels=["T", "C", "Z", "Y", "X"],
        )
        listed.chunk_shape = []  # what ListFlights actually publishes
        described = _make_tensor_desc(
            array_id="s",
            shape=[1, 1, 1, 512, 512],
            dtype="uint16",
            dim_labels=["T", "C", "Z", "Y", "X"],
        )
        described.chunk_shape = [1, 1, 1, 256, 256]

        mock_fc = _build_mock_client(_make_source_desc("s", tensors=[listed]))
        mock_fc.get_descriptor.side_effect = lambda array_id, **_k: described
        return mock_fc

    def test_tile_info_reports_the_described_grid(self):
        mock_fc = self._client()
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            with TestClient(create_app(token=None)) as tc:
                body = tc.get("/api/tile_info/s").json()

        assert body["chunk_shape"] == [1, 1, 1, 256, 256]
        assert body["tile_size"] == 256
        assert mock_fc.get_descriptor.called

    def test_the_source_listing_publishes_no_grid(self):
        mock_fc = self._client()
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            with TestClient(create_app(token=None)) as tc:
                (source,) = tc.get("/api/sources").json()

        assert source["tensors"][0]["shape"] == [1, 1, 1, 512, 512]
        assert source["tensors"][0]["chunk_shape"] == []


class TestTileArrayIdAddressing:
    """array_id is the whole address (identity policy in descriptor.proto).

    The pair form these routes used to take was split and immediately rejoined
    before the read, and the two derivations could disagree.
    """

    def test_a_qualified_array_id_selects_its_tensor(self):
        with _multi_tensor_client() as (tc, _):
            info = tc.get("/api/tile_info/multi/Image:1").json()
            assert info["array_id"] == "multi/Image:1"
            assert info["shape"] == [1, 1, 1, 2048, 2048]

    def test_a_field_containing_slashes_survives_the_route(self):
        # The policy allows '/' inside the field (HCS "plate/A01/0"); only the
        # source_id prefix is slash-free. {array_id:path} must capture it whole.
        td = _make_tensor_desc(
            array_id="plate/A01/0",
            shape=[1, 1, 1, 512, 512],
            dtype="uint16",
            dim_labels=["T", "C", "Z", "Y", "X"],
        )
        src = _make_source_desc(source_id="plate", tensors=[td])
        mock_fc = _build_mock_client(src)
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            with TestClient(create_app(token=None)) as tc:
                assert (
                    tc.get("/api/tile_info/plate/A01/0").json()["array_id"]
                    == "plate/A01/0"
                )

    def test_a_bare_id_publishes_the_qualified_id_the_server_chose(self):
        # The server defaults; the sidecar reports which tensor that was rather
        # than refusing. First contact is where the ambiguity ends: the viewer
        # threads the qualified id back through every tile after.
        with _multi_tensor_client() as (tc, _):
            info = tc.get("/api/tile_info/multi").json()
            assert info["array_id"] == "multi/Image:0"
            assert info["shape"] == [1, 1, 1, 512, 512]

    def test_a_bare_id_still_works_for_a_single_tensor_source(self):
        # For a single-tensor source the bare source_id *is* the array_id.
        with _tile_client_for(["t", "c", "z", "y", "x"], [1, 3, 16, 512, 512]) as (
            tc,
            _,
        ):
            assert tc.get("/api/tile_info/s").status_code == 200

    def test_an_unknown_field_404s_and_names_the_alternatives(self):
        with _multi_tensor_client() as (tc, _):
            r = tc.get("/api/tile/multi/Nope")
            assert r.status_code == 404
            assert "multi/Image:0" in r.json()["detail"]

    def test_the_read_addresses_the_tensor_the_geometry_came_from(self):
        """The bug the split allowed: geometry from one tensor, read of another.

        A bare multi-tensor id gave tensor[0]'s shape while the read was issued
        for the bare source_id, whose resolution is caller-dependent.
        """
        with _multi_tensor_client() as (tc, mock_fc):
            assert tc.get("/api/tile/multi/Image:1").status_code == 200
            assert mock_fc.get_tensor.call_args.args[0] == "multi/Image:1"

    def test_no_tensor_id_parameter_is_accepted_any_more(self):
        # A stale caller passing the old pair must not silently address
        # something else; the ignored query param cannot change the tensor.
        with _multi_tensor_client() as (tc, _):
            r = tc.get("/api/tile_info/multi", params={"tensor_id": "Image:1"})
            assert r.json()["array_id"] == "multi/Image:0"


# ===========================================================================
# Unit tests — the content version rides the array_id namespace (#780)
# ===========================================================================


@contextmanager
def _versioned_tile_client(content_version):
    """A tile client whose source publishes *content_version* (or None)."""
    mock_fc = _build_mock_client(_tile_source_desc(content_version))
    lazy = MagicMock()
    lazy.compute.return_value = np.zeros((1, 1, 1, 512, 512), dtype=np.uint16)
    mock_fc.get_tensor.return_value = lazy
    with patch(
        "biopb_tensor_server.serving.http_server.TensorFlightClient",
        return_value=mock_fc,
    ):
        with TestClient(create_app(token=None), raise_server_exceptions=True) as tc:
            yield tc, mock_fc


def _published_array_id(tc, requested="tiled"):
    r = tc.get(f"/api/tile_info/{requested}")
    assert r.status_code == 200, r.text
    return r.json()["array_id"]


class TestVersionTokenCodec:
    """Splicing a version into an array_id must not disturb the field half."""

    def test_it_round_trips(self):
        versioned = _versioned_array_id("plate_a1b2/A01/0", "9f1c4e2b")
        assert versioned == "plate_a1b2@9f1c4e2b/A01/0"
        assert _split_array_version(versioned) == ("plate_a1b2/A01/0", "9f1c4e2b")

    def test_an_unversioned_id_is_unchanged(self):
        assert _split_array_version("plate_a1b2/A01/0") == ("plate_a1b2/A01/0", None)

    def test_an_at_sign_in_a_field_is_not_a_version(self):
        # Only the source half is parsed: a source_id is `<type>_<hex>` and can
        # never contain "@", but a field may.
        assert _split_array_version("src_a1/weird@name") == ("src_a1/weird@name", None)

    def test_no_token_means_no_splice(self):
        assert _versioned_array_id("src_a1/f", None) == "src_a1/f"


class TestTileInfoPublishesTheVersion:
    def test_a_versioned_source_gets_a_versioned_array_id(self):
        with _versioned_tile_client(b"1700000000:4096") as (tc, _):
            array_id = _published_array_id(tc)
        assert array_id.startswith("tiled@")
        assert _split_array_version(array_id) == (
            "tiled/Image:0",
            array_id.split("@")[1][:8],
        )

    def test_an_unversioned_source_is_published_bare(self):
        # A URL that cannot be stat'd publishes no claim about its content.
        with _versioned_tile_client(None) as (tc, _):
            assert _published_array_id(tc) == "tiled/Image:0"


class TestVersionedTileRequests:
    def test_the_published_id_serves_tiles_and_goes_immutable(self):
        with _versioned_tile_client(b"1700000000:4096") as (tc, _):
            array_id = _published_array_id(tc)
            r = tc.get(f"/api/tile/{array_id}")
        assert r.status_code == 200
        assert "immutable" in r.headers["cache-control"]

    def test_an_unversioned_request_keeps_the_hour_hedge(self):
        with _versioned_tile_client(b"1700000000:4096") as (tc, _):
            r = tc.get("/api/tile/tiled/Image:0")
        assert r.status_code == 200
        assert "immutable" not in r.headers["cache-control"]
        assert "max-age=3600" in r.headers["cache-control"]

    def test_a_superseded_version_is_a_404_and_costs_no_read(self):
        with _versioned_tile_client(b"1700000000:4096") as (tc, mock_fc):
            stale = _published_array_id(tc)
        # Same source, re-indexed: the token it published is no longer current.
        with _versioned_tile_client(b"1700009999:5120") as (tc, mock_fc):
            before = mock_fc.get_tensor.call_count
            r = tc.get(f"/api/tile/{stale}")
            assert r.status_code == 404, r.text
            assert mock_fc.get_tensor.call_count == before
            # The 404 still names what does exist, as every other one does.
            assert "tiled/Image:0" in r.json()["detail"]

    def test_a_re_index_mints_a_different_id(self):
        with _versioned_tile_client(b"1700000000:4096") as (tc, _):
            before = _published_array_id(tc)
        with _versioned_tile_client(b"1700009999:5120") as (tc, _):
            after = _published_array_id(tc)
        assert before != after

    def test_an_unchanged_re_index_keeps_the_id(self):
        # The token is content, not a timestamp: re-registering identical
        # content must not evict every browser's cache.
        ids = []
        for _ in range(2):
            with _versioned_tile_client(b"1700000000:4096") as (tc, _):
                ids.append(_published_array_id(tc))
        assert ids[0] == ids[1]

    def test_the_etag_follows_the_content_not_just_the_url(self):
        # The unversioned URL is stable across a re-index, so only the ETag can
        # stop a revalidation answering 304 for bytes that changed.
        etags = []
        for cv in (b"1700000000:4096", b"1700009999:5120"):
            with _versioned_tile_client(cv) as (tc, _):
                etags.append(tc.get("/api/tile/tiled/Image:0").headers["ETag"])
        assert etags[0] != etags[1]

    def test_a_stale_source_listing_cannot_weaken_the_check(self):
        """The token is read off the *bound descriptor*, never off the listing.

        Here the listing is frozen at the old version and the check still holds.
        That independence is what let biopb/biopb#834 take the listing off the
        resolution path entirely.
        """
        old, new = b"1700000000:4096", b"1700009999:5120"

        def client(listing_cv, descriptor_cv):
            mock_fc = _build_mock_client(_tile_source_desc(listing_cv))
            fresh = _tile_source_desc(descriptor_cv).tensors[0]
            mock_fc.get_descriptor.side_effect = lambda aid, **k: fresh
            lazy = MagicMock()
            lazy.compute.return_value = np.zeros((1, 1, 1, 512, 512), dtype=np.uint16)
            mock_fc.get_tensor.return_value = lazy
            return mock_fc

        def serve(mock_fc, fn):
            with patch(
                "biopb_tensor_server.serving.http_server.TensorFlightClient",
                return_value=mock_fc,
            ):
                with TestClient(
                    create_app(token=None), raise_server_exceptions=True
                ) as tc:
                    return fn(tc)

        published = serve(client(old, old), _published_array_id)
        # Content moved on. The listing is stale (memoized at `old`); only the
        # descriptor knows.
        stale_listing = client(old, new)
        assert (
            serve(
                stale_listing, lambda tc: tc.get(f"/api/tile/{published}").status_code
            )
            == 404
        )
        assert serve(client(old, new), _published_array_id) != published

    def test_the_clients_percent_encoded_spelling_resolves_identically(self):
        # `encodeArrayId` in the TS client encodes per segment, so "@" arrives as
        # %40. Both spellings must be ONE cache entry, not two -- same ETag, same
        # policy -- or the versioned id would fragment the browser cache it
        # exists to protect.
        from urllib.parse import quote

        with _versioned_tile_client(b"1700000000:4096") as (tc, _):
            array_id = _published_array_id(tc)
            encoded = "/".join(quote(seg, safe="") for seg in array_id.split("/"))
            assert "%40" in encoded
            raw = tc.get(f"/api/tile/{array_id}")
            enc = tc.get(f"/api/tile/{encoded}")
        assert raw.status_code == enc.status_code == 200
        assert raw.headers["ETag"] == enc.headers["ETag"]
        assert raw.headers["cache-control"] == enc.headers["cache-control"]

    def test_slice_resolves_a_versioned_id_too(self):
        # One resolution point (#766), so the version works on every route.
        with _versioned_tile_client(b"1700000000:4096") as (tc, _):
            array_id = _published_array_id(tc)
            ok = tc.post("/api/slice", json={"array_id": array_id})
            stale = tc.post("/api/slice", json={"array_id": "tiled@deadbeef/Image:0"})
        assert ok.status_code == 200
        assert stale.status_code == 404


# ===========================================================================
# Unit tests — a tile resolves by targeted fetch, not a catalog scan (#834)
# ===========================================================================


class TestTileResolutionCostsOneFetch:
    """Listing every source to look up the one id the request already carries
    was a second RPC on every tile, and the catalog is the expensive half."""

    def test_a_qualified_tile_never_lists_the_catalog(self):
        with _multi_tensor_client() as (tc, mock_fc):
            mock_fc.reset_mock()
            assert tc.get("/api/tile/multi/Image:1").status_code == 200
            assert mock_fc.get_descriptor.call_count == 1
            mock_fc.list_sources.assert_not_called()

    def test_a_bare_id_needs_no_listing_either(self):
        """bioio names a lone scene `<source>/Image:0`, so the server answers a
        bare source_id with an array_id that differs from it. Deciding whether
        that was legitimate would take a tensor count, i.e. a listing -- and the
        sidecar does not decide it. The server's answer is the answer.
        """
        with _versioned_tile_client(b"1700000000:4096") as (tc, mock_fc):
            mock_fc.reset_mock()
            body = tc.get("/api/tile_info/tiled").json()
            assert body["array_id"].startswith("tiled@")
            mock_fc.list_sources.assert_not_called()

    def test_the_published_id_carries_the_qualified_tensor(self):
        # First contact ends the ambiguity: tile_info hands back the qualified
        # id, and every tile after it addresses that tensor by name.
        with _versioned_tile_client(b"1700000000:4096") as (tc, mock_fc):
            array_id = _published_array_id(tc)
            mock_fc.reset_mock()
            assert tc.get(f"/api/tile/{array_id}").status_code == 200
            mock_fc.list_sources.assert_not_called()

    def test_a_dead_backend_is_a_502_not_a_404(self):
        # The fetch now owns the 404, so its except clause must not swallow a
        # transport failure -- "this tensor is gone" and "the server is down"
        # are opposite answers for a viewer holding a tile URL.
        with _multi_tensor_client() as (tc, mock_fc):
            mock_fc.get_descriptor.side_effect = flight.FlightUnavailableError(
                "connection refused"
            )
            assert tc.get("/api/tile/multi/Image:1").status_code == 502

    def test_an_unresolved_cloud_source_stays_a_404(self):
        # The client raises its resolve-directive (a ValueError) where the
        # listing used to return a source with no tensors. Same 404 either way.
        with _multi_tensor_client() as (tc, mock_fc):
            mock_fc.get_descriptor.side_effect = ValueError(
                "Source 'multi' is unresolved (no tensors listed yet)."
            )
            assert tc.get("/api/tile/multi/Image:1").status_code == 404


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
class TestIntegrationLoneQualifiedTensor:
    """A real server whose single tensor is named `<source>/Image:0`.

    The shape the targeted fetch cannot decide on its own (biopb/biopb#834), and
    the common one: bioio names a lone scene, so most non-Zarr sources look like
    this. Worth a real Flight server rather than a mocked client, because what
    is being checked is precisely what GetFlightInfo answers.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        import zarr
        from biopb_tensor_server import TensorFlightServer, ZarrAdapter

        z = zarr.open_array(
            str(tmp_path / "lone.zarr"),
            mode="w",
            shape=(3, 32, 32),
            chunks=(1, 16, 16),
            dtype="uint16",
        )
        z[:] = np.zeros((3, 32, 32), dtype="uint16")
        adapter = ZarrAdapter(z, "lone", ["z", "y", "x"])
        # One tensor, carrying a name: array_id becomes "lone/Image:0".
        adapter._tensor_name = "Image:0"

        server = TensorFlightServer("grpc://127.0.0.1:0")
        server.register_source("lone", adapter)
        threading.Thread(target=server.serve, daemon=True).start()
        time.sleep(0.5)
        self._loc = f"grpc://localhost:{server.port}"
        self._server = server
        yield
        try:
            self._server.shutdown()
        except Exception:
            pass

    @contextmanager
    def _tc(self):
        with TestClient(create_app(flight_location=self._loc, token=None)) as tc:
            yield tc

    def test_the_bare_source_id_resolves_to_the_named_tensor(self):
        with self._tc() as tc:
            r = tc.get("/api/tile_info/lone")
        assert r.status_code == 200, r.text
        assert _split_array_version(r.json()["array_id"])[0] == "lone/Image:0"

    def test_the_qualified_id_resolves_too(self):
        with self._tc() as tc:
            r = tc.get("/api/tile_info/lone/Image:0")
        assert r.status_code == 200, r.text

    def test_a_tile_comes_back_for_the_qualified_id(self):
        with self._tc() as tc:
            array_id = tc.get("/api/tile_info/lone").json()["array_id"]
            r = tc.get(f"/api/tile/{array_id}")
        assert r.status_code == 200, r.text

    def test_an_unknown_field_is_still_a_404(self):
        with self._tc() as tc:
            assert tc.get("/api/tile_info/lone/Nope").status_code == 404
