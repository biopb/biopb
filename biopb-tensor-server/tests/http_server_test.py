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
    _ADVERTISED,
    _advertised_levels,
    _Level,
    _split_array_version,
    _tile_edge,
    _tile_levels,
    _tile_read,
    _versioned_array_id,
    _volume_plan,
    create_app,
)
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_advertised_ladder():
    """The ladder memo is process-global and keyed on (array_id, version).

    Defensive: unversioned tensors are not memoized at all, and real ids are
    unique, so a collision needs two fixtures sharing one array_id *and* one
    version. Cheap enough to rule out rather than reason about per test, given
    pytest-randomly decides the order.
    """
    _ADVERTISED.clear()
    yield
    _ADVERTISED.clear()


_TOKEN = "test-token-valid-1234"
_WRONG = "totally-wrong-token-xy"

# ---------------------------------------------------------------------------
# Stand-ins for protobuf descriptor objects
# ---------------------------------------------------------------------------


def _planned_pyramid(shape, dim_labels):
    """The pyramid GetFlightInfo would advertise for a tensor of this shape."""
    from biopb_tensor_server.core.chunk import build_pyramid_plan

    return build_pyramid_plan(list(shape), list(dim_labels))


def _make_tensor_desc(
    array_id: str = "src0",
    shape=(4, 8, 16),
    dtype: str = "uint16",
    dim_labels=None,
    physical_scale=None,
    physical_unit=None,
    pyramid=None,
) -> SimpleNamespace:
    # physical_scale / physical_unit / pyramid are repeated proto fields: always
    # present on a real descriptor, empty when the source declares none.
    # Spelled out here so a fake cannot be missing what the routes may read.
    labels = list(dim_labels or ["z", "y", "x"])
    return SimpleNamespace(
        array_id=array_id,
        shape=list(shape),
        chunk_shape=[max(1, s // 2) for s in shape],
        dtype=dtype,
        dim_labels=labels,
        physical_scale=list(physical_scale or []),
        physical_unit=list(physical_unit or []),
        # Defaults to what the Flight server would advertise for this shape --
        # the sidecar reads the ladder off the descriptor rather than deriving
        # it, so a fake without one is a tensor with no pyramid, which is a
        # different tensor. Pass an explicit list (`[]` included) to say
        # otherwise.
        pyramid=_planned_pyramid(shape, labels) if pyramid is None else list(pyramid),
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


def _pyramid_level(scale, shape, method="nearest", native=False) -> SimpleNamespace:
    """A ``PyramidLevel`` as the descriptor carries it."""
    return SimpleNamespace(
        scale_hint=list(scale),
        shape=list(shape),
        reduction_method=method,
        native=native,
    )


def _planned_levels(shape, labels):
    """The ladder the Flight server advertises for *shape*, via the real reader.

    The sidecar reads the ladder off the descriptor now instead of recomputing
    it, so these hand it exactly what ``build_pyramid_plan`` puts on the wire.
    """
    from biopb_tensor_server.core.chunk import build_pyramid_plan

    td = SimpleNamespace(array_id=f"planned/{list(shape)}", shape=list(shape))
    client = MagicMock()
    client.get_descriptor.return_value = SimpleNamespace(
        pyramid=build_pyramid_plan(shape, list(labels))
    )
    return _advertised_levels(client, td, None)


def _tile_source_desc(
    content_version: bytes | None = None, pyramid=None
) -> SimpleNamespace:
    """A realistic tiled tensor: TCZYX uint16, 1024x1024 plane, 512x512 chunks.

    ``content_version`` rides the TENSOR descriptor: it is a serving field,
    filled by GetFlightInfo and empty on a catalog listing entry.

    ``pyramid`` is empty by default, which is what the real server advertises
    for this tensor: a 1 Mpx plane is already under ``plane_max_pixels``, so the
    planner emits full resolution and nothing else.
    """
    td = SimpleNamespace(
        array_id="tiled/Image:0",
        shape=[1, 3, 16, 1024, 1024],
        chunk_shape=[1, 1, 1, 512, 512],
        dtype="uint16",
        dim_labels=["t", "c", "z", "y", "x"],
        physical_scale=[],
        physical_unit=[],
        content_version=content_version,
        pyramid=(
            _planned_pyramid([1, 3, 16, 1024, 1024], ["t", "c", "z", "y", "x"])
            if pyramid is None
            else list(pyramid)
        ),
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

    @pytest.mark.parametrize(
        "shape,expected",
        [
            # 14234**2 ND2 scene: 202 Mpx -> scale 8 is the first rung under 4 Mpx.
            ([1, 14234, 14234], 3),
            # 100000**2 WSI.
            ([1, 100000, 100000], 6),
        ],
    )
    def test_the_anchor_the_server_advertises_is_a_rung_this_ladder_has(
        self, shape, expected
    ):
        """The anchor is read in its own right, so it must be addressable.

        No longer recomputed here from PRECACHE_PLANE_MAX_PIXELS -- the sidecar
        can address a plane whose config it does not own, so the ladder comes
        from the server. This pins the two together: what the planner emits is a
        power of two on Y/X, hence a rung of the tile grid.
        """
        levels = _planned_levels(shape, ["z", "y", "x"])
        anchor = levels[-1]  # the 2-D target: finest non-identity rung
        assert anchor.scale[1] == anchor.scale[2] == 1 << expected
        assert expected < len(_tile_levels(shape, 1, 2, 512))


class TestTileSynthesisIsExact:
    """Reducing the warm level must equal a direct read at the coarser level.

    The property docs/precache-policy.md 4.2 rests on: `nearest` is a strided
    pick, so `data[::32]` and `data[::8][::4]` select the same elements and
    `ceil(ceil(n/a)/b) == ceil(n/(a*b))` gives the same count. Ragged extents
    included -- those are where a padding reducer would diverge.
    """

    @pytest.mark.parametrize("extent", [4096, 4000, 3999, 1001])
    @pytest.mark.parametrize("warm,level", [(0, 1), (1, 2), (3, 5), (2, 5)])
    def test_composition_is_bit_identical(self, extent, warm, level):
        from biopb_tensor_server.core.downsample import downsample_block

        rng = np.random.default_rng(0)
        block = rng.integers(0, 65535, size=(extent, 7), dtype=np.uint16)

        direct = downsample_block(block, (1 << level, 1), "nearest")
        staged = downsample_block(
            downsample_block(block, (1 << warm, 1), "nearest"),
            (1 << (level - warm), 1),
            "nearest",
        )
        assert staged.shape == direct.shape
        assert np.array_equal(staged, direct)

    def test_area_does_not_compose(self):
        # Recorded deliberately: this is the coupling that makes `nearest` load
        # bearing for the tile route, not merely the default (policy 6).
        from biopb_tensor_server.core.downsample import downsample_block

        rng = np.random.default_rng(0)
        block = rng.integers(0, 65535, size=(4000, 7), dtype=np.uint16)

        direct = downsample_block(block, (8, 1), "area")
        staged = downsample_block(
            downsample_block(block, (2, 1), "area"), (4, 1), "area"
        )
        assert staged.shape == direct.shape
        assert not np.array_equal(staged, direct)


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

    def test_a_coarser_level_covers_more_world(self, tile_client):
        tc, mock_fc = tile_client
        tc.get("/api/tile/tiled", params={"level": 1, "col": 0, "row": 0})
        kwargs = mock_fc.get_tensor.call_args.kwargs
        # World bounds come from the level addressed, whatever level is read.
        assert kwargs["slice_hint"][4] == slice(0, 1024)
        assert kwargs["slice_hint"][3] == slice(0, 1024)

    def test_a_level_below_the_warm_one_is_read_at_the_warm_scale(self, tile_client):
        # 1024x1024 fits plane_max_pixels, so the warm level is 0 and level 1 is
        # synthesized: the data plane is asked for scale 1 and the reduction to
        # scale 2 happens here, off the one warmed level (precache-policy.md 4.2).
        tc, mock_fc = tile_client
        r = tc.get("/api/tile/tiled", params={"level": 1, "col": 0, "row": 0})
        kwargs = mock_fc.get_tensor.call_args.kwargs
        assert kwargs["scale_hint"][3] == 1 and kwargs["scale_hint"][4] == 1
        # The mock answers 512x512 whatever it is asked; halving it is the proof
        # the in-process reduction ran.
        assert r.headers["X-Shape"] == "1,1,1,256,256"

    def test_a_kernel_choice_on_a_tile_is_410(self, tile_client):
        # Withdrawn, not ignored: silently serving different pixels is what a
        # caller pinned to an older server cannot detect. A tile is the display
        # path -- what the parameter selected was a store, not a kernel.
        tc, mock_fc = tile_client
        mock_fc.reset_mock()
        r = tc.get(
            "/api/tile/tiled",
            params={"level": 1, "col": 0, "row": 0, "reduction_method": "area"},
        )
        assert r.status_code == 410
        assert "POST /api/slice" in r.json()["detail"]
        # Refused before any backend call, like `fmt`.
        mock_fc.get_tensor.assert_not_called()

    @pytest.mark.parametrize("spelling", ["nearest", "decimate", "stride", "NEAREST"])
    def test_the_kernel_that_tiles_already_use_is_still_accepted(
        self, spelling, tile_client
    ):
        # Aliases resolve before the refusal, so a caller naming the behaviour it
        # was already getting is not broken by the withdrawal.
        tc, _ = tile_client
        r = tc.get("/api/tile/tiled", params={"reduction_method": spelling})
        assert r.status_code == 200

    def test_full_resolution_is_read_directly(self, tile_client):
        tc, mock_fc = tile_client
        r = tc.get("/api/tile/tiled", params={"level": 0, "col": 0, "row": 0})
        kwargs = mock_fc.get_tensor.call_args.kwargs
        assert kwargs["scale_hint"][3] == 1 and kwargs["scale_hint"][4] == 1
        assert r.headers["X-Shape"] == "1,1,1,512,512"

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
            mock_fc.list_sources.assert_not_called()
            # Two describes, no listing: the structural fetch this route makes
            # per request by contract, and the pyramid ladder -- which this
            # source pays for on every tile because it publishes no content
            # version to memoize on. The catalog, the expensive half and the
            # thing this class is about, is never touched.
            assert mock_fc.get_descriptor.call_count == 2

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


# ===========================================================================
# Unit tests — the volume plan (3-D reads)
# ===========================================================================


class TestVolumePlan:
    """The scale a 3-D read resolves to, and the tensors that have none.

    The expected scales are docs/precache-policy.md 5.1's table: this plan IS
    the Flight ladder's coarsest level, so a divergence here means the sidecar
    and the precache worker have stopped agreeing about what is warm.
    """

    @pytest.mark.parametrize(
        "shape,labels,scale,extent",
        [
            # cube: 2-D gate skipped (1.05 Mpx plane), 3-D gate fires alone.
            ([1024, 1024, 1024], ["z", "y", "x"], [4, 4, 4], (256, 256, 256)),
            # confocal: both gates, but Z is already under the 448 floor.
            ([200, 2048, 2048], ["z", "y", "x"], [1, 4, 4], (200, 512, 512)),
            ([1200, 2048, 2048], ["z", "y", "x"], [4, 8, 8], (300, 256, 256)),
            ([1000, 6000, 6000], ["z", "y", "x"], [4, 16, 16], (250, 375, 375)),
            # thin stack: neither gate fires, so the volume IS full resolution.
            ([40, 1024, 1024], ["z", "y", "x"], [1, 1, 1], (40, 1024, 1024)),
            # TCZYX: the non-volume axes are never scaled.
            (
                [3, 2, 181, 1024, 1024],
                ["t", "c", "z", "y", "x"],
                [1, 1, 1, 2, 2],
                (181, 512, 512),
            ),
        ],
    )
    def test_the_plan_is_the_flight_ladders_coarsest_level(
        self, shape, labels, scale, extent
    ):
        plan, reason = _volume_plan(
            shape, labels, "uint16", _planned_levels(shape, labels)
        )
        assert reason is None
        assert plan["scale_hint"] == scale
        assert (plan["depth"], plan["height"], plan["width"]) == extent

    def test_the_byte_count_is_the_wire_size_at_the_source_dtype(self):
        shape, labels = [1024, 1024, 1024], ["z", "y", "x"]
        plan, _ = _volume_plan(shape, labels, "uint16", _planned_levels(shape, labels))
        assert plan["bytes"] == 256 * 256 * 256 * 2

    @pytest.mark.parametrize(
        "shape,labels,expect",
        [
            # A plane has no depth.
            ([100000, 100000], ["y", "x"], "at least 3"),
            # An unlabeled or non-z third axis is NOT promoted to depth: a
            # timelapse rendered as a solid block is worse than no 3-D at all.
            ([4, 14234, 14234], ["c", "y", "x"], "no z axis"),
            ([500, 2, 1024, 1024], ["t", "c", "y", "x"], "no z axis"),
            # One plane is not a volume.
            ([1, 2048, 2048], ["z", "y", "x"], "extent 1"),
            # Interleaved RGB is a per-voxel tuple, not a scalar field.
            ([10, 512, 512, 3], ["z", "y", "x", "s"], "interleaved samples"),
        ],
    )
    def test_a_tensor_with_no_volume_is_refused_with_the_reason(
        self, shape, labels, expect
    ):
        # The refusals are facts about the tensor, so the ladder is irrelevant.
        plan, reason = _volume_plan(shape, labels, "uint16", ())
        assert plan is None
        assert expect in reason

    def test_the_volume_stays_within_the_pyramids_voxel_budget(self):
        # 448**3, the measured GPU budget (9.1). The read is unbounded in
        # principle -- slice_hint spans three whole axes -- so this is the only
        # thing standing between a 3-D toggle and a multi-gigabyte response.
        shape, labels = [4000, 8000, 8000], ["z", "y", "x"]
        plan, _ = _volume_plan(shape, labels, "uint16", _planned_levels(shape, labels))
        voxels = plan["depth"] * plan["height"] * plan["width"]
        assert voxels <= 448**3


class TestVolumeOnTileInfo:
    """`/api/tile_info` carries the plan, so a viewer needs no second call."""

    def test_a_volumetric_tensor_publishes_its_plan(self, tile_client):
        tc, _ = tile_client
        volume = tc.get("/api/tile_info/tiled").json()["volume"]
        assert volume["available"] is True
        # TCZYX [1, 3, 16, 1024, 1024]: 16 Mvox, under every gate.
        assert volume["axes"] == {"z": 2, "y": 3, "x": 4}
        assert volume["scale_hint"] == [1, 1, 1, 1, 1]
        assert (volume["depth"], volume["height"], volume["width"]) == (16, 1024, 1024)

    def test_a_flat_tensor_says_why_instead_of_omitting_the_field(self):
        # Always present: a viewer decides whether to offer 3-D from this, and
        # an absent field would be indistinguishable from an older server.
        td = _make_tensor_desc(
            array_id="flat", shape=(4096, 4096), dim_labels=["y", "x"]
        )
        mock_fc = _build_mock_client(_make_source_desc(source_id="flat", tensors=[td]))
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=None)
            with TestClient(app, raise_server_exceptions=True) as tc:
                volume = tc.get("/api/tile_info/flat").json()["volume"]
        assert volume["available"] is False
        assert "at least 3" in volume["reason"]

    def test_spacing_is_the_source_scale_times_the_plans_own(self):
        # Published as the product because that is what a renderer needs; the
        # two factors are in different orders and mixing them up stretches the
        # volume silently.
        td = _make_tensor_desc(
            array_id="phys",
            shape=(200, 2048, 2048),
            dim_labels=["z", "y", "x"],
            physical_scale=[0.5, 0.1, 0.1],
            # A spelling unit_to_um places, so it comes back canonicalised.
            physical_unit=["micrometer"] * 3,
        )
        mock_fc = _build_mock_client(_make_source_desc(source_id="phys", tensors=[td]))
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=None)
            with TestClient(app, raise_server_exceptions=True) as tc:
                volume = tc.get("/api/tile_info/phys").json()["volume"]
        # scale [1, 4, 4]: z keeps its 0.5 um step, x/y are four voxels wide.
        assert volume["scale_hint"] == [1, 4, 4]
        assert volume["spacing"] == {"z": 0.5, "y": 0.4, "x": 0.4}
        assert volume["unit"] == "µm"

    @pytest.mark.parametrize(
        "units,expect_spacing,expect_unit",
        [
            # Mixed but placeable: the EM / OME case. A z in nm beside an x/y in
            # um is a stack a thousand times too deep if the ratio is taken raw.
            (["nm", "µm", "µm"], {"z": 0.0005, "y": 0.4, "x": 0.4}, "µm"),
            # Uniform and placeable: converted, so the wire unit is always one
            # thing when it is named at all.
            (["mm", "mm", "mm"], {"z": 500.0, "y": 400.0, "x": 400.0}, "µm"),
            # Uniform and NOT placeable: kept. A ratio of like-for-like is valid
            # whether or not the unit can be named -- this is a NIfTI with
            # xyzt_units unset, which would otherwise lose its anisotropy.
            (["hogshead"] * 3, {"z": 0.5, "y": 0.4, "x": 0.4}, "hogshead"),
            (["", "", ""], {"z": 0.5, "y": 0.4, "x": 0.4}, None),
            # Mixed and not all placeable: refused. Guessing which axis the odd
            # unit belongs to is exactly the silent stretch this avoids.
            (["pixel", "µm", "µm"], None, None),
        ],
    )
    def test_the_three_axes_are_reduced_to_one_unit(
        self, units, expect_spacing, expect_unit
    ):
        # physical_unit is per-axis and adapters do not all normalise, but the
        # ratio compares the three against each other.
        td = _make_tensor_desc(
            array_id="units",
            shape=(200, 2048, 2048),
            dim_labels=["z", "y", "x"],
            physical_scale=[0.5, 0.1, 0.1],
            physical_unit=units,
        )
        mock_fc = _build_mock_client(_make_source_desc(source_id="units", tensors=[td]))
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=None)
            with TestClient(app, raise_server_exceptions=True) as tc:
                volume = tc.get("/api/tile_info/units").json()["volume"]
        if expect_spacing is None:
            assert volume["spacing"] is None
        else:
            assert volume["spacing"] == pytest.approx(expect_spacing)
        assert volume["unit"] == expect_unit

    def test_spacing_is_null_when_the_source_declares_none(self, tile_client):
        tc, _ = tile_client
        volume = tc.get("/api/tile_info/tiled").json()["volume"]
        assert volume["spacing"] is None
        assert volume["unit"] is None


class TestScalePolicyOnSlice:
    """`scale_policy` hands the scale decision to the server."""

    def _post(self, tc, **body):
        return tc.post("/api/slice", json={"array_id": "tiled/Image:0", **body})

    def test_the_server_reads_at_the_volume_plans_scale(self):
        # A tensor whose plan actually scales something, so "the server chose"
        # is distinguishable from "nothing was passed".
        td = _make_tensor_desc(
            array_id="big/Image:0",
            shape=(1024, 1024, 1024),
            dim_labels=["z", "y", "x"],
        )
        mock_fc = _build_mock_client(_make_source_desc(source_id="big", tensors=[td]))
        lazy = MagicMock()
        lazy.compute.return_value = np.zeros((256, 256, 256), dtype=np.uint16)
        mock_fc.get_tensor.return_value = lazy
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=None)
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.post(
                    "/api/slice",
                    json={"array_id": "big/Image:0", "scale_policy": "volume"},
                )
        assert r.status_code == 200, r.text
        assert mock_fc.get_tensor.call_args.kwargs["scale_hint"] == [4, 4, 4]
        # Echoed, because the caller did not choose it: this header is the only
        # statement of what it got.
        assert r.headers["X-Scale-Hint"] == "4,4,4"

    def test_a_caller_that_chose_gets_its_own_scale_echoed(self, tile_client):
        tc, mock_fc = tile_client
        r = self._post(tc, scale_hint=[1, 1, 1, 2, 2])
        assert r.status_code == 200, r.text
        assert mock_fc.get_tensor.call_args.kwargs["scale_hint"] == [1, 1, 1, 2, 2]
        assert r.headers["X-Scale-Hint"] == "1,1,1,2,2"

    def test_an_unscaled_read_still_says_so(self, tile_client):
        tc, _ = tile_client
        r = self._post(tc)
        assert r.headers["X-Scale-Hint"] == "1,1,1,1,1"

    def test_naming_a_scale_and_delegating_it_is_refused(self, tile_client):
        # Not last-wins: one read has one scale, and letting the two disagree
        # would make which one applies a silent policy.
        tc, mock_fc = tile_client
        r = self._post(tc, scale_policy="volume", scale_hint=[1, 1, 1, 2, 2])
        assert r.status_code == 422
        assert "one scale" in r.json()["detail"]
        mock_fc.get_tensor.assert_not_called()

    def test_an_unknown_policy_names_the_ones_that_exist(self, tile_client):
        tc, mock_fc = tile_client
        r = self._post(tc, scale_policy="coarsest")
        assert r.status_code == 422
        assert "volume" in r.json()["detail"]
        mock_fc.get_tensor.assert_not_called()

    def test_a_tensor_with_no_volume_gets_the_same_reason_tile_info_gives(self):
        td = _make_tensor_desc(
            array_id="flat", shape=(4096, 4096), dim_labels=["y", "x"]
        )
        mock_fc = _build_mock_client(_make_source_desc(source_id="flat", tensors=[td]))
        with patch(
            "biopb_tensor_server.serving.http_server.TensorFlightClient",
            return_value=mock_fc,
        ):
            app = create_app(token=None)
            with TestClient(app, raise_server_exceptions=True) as tc:
                r = tc.post(
                    "/api/slice", json={"array_id": "flat", "scale_policy": "volume"}
                )
        assert r.status_code == 422
        assert "at least 3" in r.json()["detail"]
        mock_fc.get_tensor.assert_not_called()


# ---------------------------------------------------------------------------
# Native pyramid levels (biopb/biopb#889)
# ---------------------------------------------------------------------------


def _native_source_desc(plane=4096, scales=(2, 4)):
    """A tiled tensor shipping a real on-disk pyramid over its Y/X plane.

    Bigger than the plain tile fixture on purpose: a 1024 plane at edge 512 has
    only two rungs, too few to tell "read the level" from "read the level and
    reduce the rest".
    """
    td = SimpleNamespace(
        array_id="native/Image:0",
        shape=[1, 3, 16, plane, plane],
        chunk_shape=[1, 1, 1, 512, 512],
        dtype="uint16",
        dim_labels=["t", "c", "z", "y", "x"],
        physical_scale=[],
        physical_unit=[],
        # Versioned, so the ladder is memoizable -- an unversioned tensor is
        # deliberately refetched every tile (see _advertised_levels).
        content_version=b"native-v1",
        pyramid=[
            _pyramid_level(
                [1, 1, 1, s, s],
                [1, 3, 16, plane // s, plane // s],
                method="precompute",
                native=True,
            )
            for s in (1, *scales)
        ],
    )
    return SimpleNamespace(
        source_id="native",
        source_url="/data/native.zarr",
        source_type="zarr",
        metadata_json=None,
        tensors=[td],
    )


@pytest.fixture()
def native_tile_client():
    """The tile fixture, but shipping a real 3-level on-disk pyramid."""
    src = _native_source_desc()
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


class TestTileReadPicksALevel:
    """Which advertised level backs a rung, and what is left to reduce."""

    SHAPE = [1, 3, 16, 1024, 1024]

    def _levels(self, *scales, native=True, method="precompute"):
        return [
            _Level(
                scale=(1, 1, 1, s, s),
                shape=(1, 3, 16, 1024 // s, 1024 // s),
                method=method,
                native=native,
            )
            for s in scales
        ]

    def test_an_exact_match_is_read_with_no_residual(self):
        plan = _tile_read(self.SHAPE, 3, 4, 2, self._levels(4, 2))
        assert plan.scale_hint == [1, 1, 1, 4, 4]
        assert plan.method == "precompute"
        assert plan.residual is None

    def test_a_dividing_level_is_read_and_the_rest_decimated(self):
        # Rung 3 wants scale 8; the coarsest native level is 4, so 4 is read and
        # the remaining 2 is done here. This is the case exact matching missed.
        plan = _tile_read(self.SHAPE, 3, 4, 3, self._levels(4, 2))
        assert plan.scale_hint == [1, 1, 1, 4, 4]
        assert plan.residual == [1, 1, 1, 2, 2]

    def test_the_coarsest_dividing_level_wins(self):
        # Both 2 and 4 divide 8; picking 2 would read four times the bytes.
        plan = _tile_read(self.SHAPE, 3, 4, 3, self._levels(2, 4))
        assert plan.scale_hint == [1, 1, 1, 4, 4]

    def test_a_rung_finer_than_every_level_reads_its_own_scale(self):
        # Rung 0 is full resolution: no coarser level divides 1.
        plan = _tile_read(self.SHAPE, 3, 4, 0, self._levels(4, 2))
        assert plan.scale_hint is None
        assert plan.read_level == 0
        assert plan.residual is None

    def test_a_level_that_scales_z_is_never_picked_for_a_plane(self):
        """The 3-D target must not serve a 2-D tile.

        A tile carries 1 on z, and 1 % 2 != 0 -- so the divisibility test that
        makes the residual whole also keeps the volumetric level out, with no
        separate gate to keep in sync.
        """
        volumetric = [
            _Level(
                scale=(1, 1, 4, 4, 4),
                shape=(1, 3, 4, 256, 256),
                method="",
                native=False,
            )
        ]
        plan = _tile_read(self.SHAPE, 3, 4, 2, volumetric)
        assert plan.scale_hint is None
        assert plan.read_level == 2

    def test_an_empty_ladder_still_anchors_on_full_resolution(self):
        # No coarser level exists, so full resolution is the anchor and the tail
        # is reduced from it -- what warm_level == 0 used to mean.
        plan = _tile_read(self.SHAPE, 3, 4, 2, [])
        assert plan.scale_hint == [1, 1, 1, 1, 1]
        assert plan.residual == [1, 1, 1, 4, 4]

    def test_a_computed_level_carries_its_own_method(self):
        plan = _tile_read(
            self.SHAPE, 3, 4, 3, self._levels(4, native=False, method="nearest")
        )
        assert plan.method == "nearest"


class TestNativeLevelsOnTheTileRoute:
    """End to end: a pyramidal source stops decimating full resolution."""

    def test_the_read_addresses_the_native_level(self, native_tile_client):
        tc, mock_fc = native_tile_client
        r = tc.get("/api/tile/native", params={"level": 2, "col": 0, "row": 0})
        assert r.status_code == 200
        kwargs = mock_fc.get_tensor.call_args.kwargs
        # Both halves of the address: an exact scale AND `precompute`. Either
        # one alone lands on a computed read of level 0.
        assert kwargs["scale_hint"] == [1, 1, 1, 4, 4]
        assert kwargs["reduction_method"] == "precompute"

    def test_a_rung_past_the_ladder_reads_the_coarsest_level_and_reduces(
        self, native_tile_client
    ):
        tc, mock_fc = native_tile_client
        r = tc.get("/api/tile/native", params={"level": 3, "col": 0, "row": 0})
        kwargs = mock_fc.get_tensor.call_args.kwargs
        assert kwargs["scale_hint"] == [1, 1, 1, 4, 4]
        # The mock answers 512x512 whatever it is asked; halving it is the proof
        # the in-process residual ran.
        assert r.headers["X-Shape"] == "1,1,1,256,256"

    def test_world_bounds_still_come_from_the_rung_addressed(self, native_tile_client):
        tc, mock_fc = native_tile_client
        tc.get("/api/tile/native", params={"level": 2, "col": 0, "row": 0})
        hint = mock_fc.get_tensor.call_args.kwargs["slice_hint"]
        # Level 2 at edge 512 spans 2048 world units, whatever level is read.
        assert hint[3] == slice(0, 2048) and hint[4] == slice(0, 2048)

    def test_the_ladder_is_published_for_diagnosis(self, native_tile_client):
        tc, _ = native_tile_client
        pyramid = tc.get("/api/tile_info/native").json()["pyramid"]
        # Coarsest first, level 0 dropped: it names full resolution, which is
        # what a caller gets without asking for a level.
        assert [entry["scale_hint"] for entry in pyramid] == [
            [1, 1, 1, 4, 4],
            [1, 1, 1, 2, 2],
        ]
        assert all(entry["native"] for entry in pyramid)

    @staticmethod
    def _ladder_fetches(mock_fc):
        return sum(
            1
            for call in mock_fc.get_descriptor.call_args_list
            if call.kwargs.get("with_pyramid")
        )

    def test_the_ladder_is_fetched_once_per_version(self, native_tile_client):
        tc, mock_fc = native_tile_client
        before = self._ladder_fetches(mock_fc)
        for level in (1, 2, 3):
            tc.get("/api/tile/native", params={"level": level})
        # Memoized on the content version, so a tile burst pays for one.
        assert self._ladder_fetches(mock_fc) - before == 1

    def test_an_unversioned_tensor_refetches_the_ladder_every_tile(self, tile_client):
        """No token means nothing changes when the file does.

        Memoizing there would let a ladder outlive the content it describes for
        the life of the process, and a stale ladder is not a slow tile -- it is
        a read addressed to a level that may no longer exist. One descriptor
        call per tile is the safe direction to err.
        """
        tc, mock_fc = tile_client
        before = self._ladder_fetches(mock_fc)
        for level in (0, 1):
            tc.get("/api/tile/tiled", params={"level": level})
        assert self._ladder_fetches(mock_fc) - before == 2


class TestNativeLevelsOnTheVolumePath:
    """`scale_policy="volume"` resolves to a native level where one exists."""

    def test_the_plan_takes_the_coarsest_native_level_and_its_method(self):
        levels = (
            _Level((1, 1, 1, 4, 4), (1, 3, 16, 256, 256), "precompute", True),
            _Level((1, 1, 1, 2, 2), (1, 3, 16, 512, 512), "precompute", True),
        )
        plan, reason = _volume_plan(
            [1, 3, 16, 1024, 1024], ["t", "c", "z", "y", "x"], "uint16", levels
        )
        assert reason is None
        assert plan["scale_hint"] == [1, 1, 1, 4, 4]
        assert plan["reduction_method"] == "precompute"
        # The level's OWN extent, not ceil_div(base, scale): a writer that floors
        # a level shape is still telling the truth about what the read returns.
        assert (plan["depth"], plan["height"], plan["width"]) == (16, 256, 256)

    def test_the_slice_read_carries_both_halves_of_the_address(
        self, native_tile_client
    ):
        tc, mock_fc = native_tile_client
        r = tc.post(
            "/api/slice",
            json={"array_id": "native/Image:0", "scale_policy": "volume"},
        )
        assert r.status_code == 200, r.text
        kwargs = mock_fc.get_tensor.call_args.kwargs
        assert kwargs["scale_hint"] == [1, 1, 1, 4, 4]
        assert kwargs["reduction_method"] == "precompute"
        assert r.headers["X-Scale-Hint"] == "1,1,1,4,4"


class TestRaggedPlanesComposeExactly:
    """Both read paths must return the extent the published grid promises.

    A rung served from a level and a rung read directly differ in where the bytes
    come from, never in how many. On a ragged plane that is not automatic: the
    grid is ceil, decimation is ceil, so the level slice has to be too. It
    floored until biopb/biopb#889, which made the last tile one short -- and
    where the last tile is a single pixel wide, empty.
    """

    EDGE = 512

    @pytest.mark.parametrize(
        "width", [2048, 2047, 1001, 1000, 999, 4095, 4097, 3000, 513]
    )
    @pytest.mark.parametrize("factor", [2, 4])
    def test_the_last_tile_of_a_row_is_the_width_the_grid_published(
        self, width, factor
    ):
        from biopb.tensor.descriptor_pb2 import SliceHint
        from biopb_tensor_server.core.adapter_base import _convert_slice_to_level

        grid = _tile_levels([1, 1, 1, width, width], 3, 4, self.EDGE)
        for rung in grid:
            scale = rung["scale"]
            if scale % factor:
                continue
            col = rung["cols"] - 1
            start = col * self.EDGE * scale
            stop = min(start + self.EDGE * scale, width)
            published = rung["width"] - col * self.EDGE

            # Read the rung directly: the data plane decimates.
            computed = len(range(0, stop - start, scale))
            assert computed == published, f"computed, level {rung['level']}"

            # Read a level and decimate the remainder here.
            level = _convert_slice_to_level(
                SliceHint(start=[start], stop=[stop]), [factor]
            )
            span = level.stop[0] - level.start[0]
            assert len(range(0, span, scale // factor)) == published, (
                f"via level, rung {rung['level']} factor {factor}"
            )

    def test_a_level_whose_writer_floored_its_shape_is_not_used(self):
        """A floored store holds fewer pixels than the grid promises.

        4097 at factor 2 is a 2049-wide rung; a level floored to 2048 cannot
        serve its last tile. Left to the Flight clients that address it by name.
        """
        shape = [1, 1, 1, 4097, 4097]
        floored = _Level((1, 1, 1, 2, 2), (1, 1, 1, 2048, 2048), "precompute", True)
        assert _tile_read(shape, 3, 4, 1, [floored]).scale_hint is None

        exact = _Level((1, 1, 1, 2, 2), (1, 1, 1, 2049, 2049), "precompute", True)
        assert _tile_read(shape, 3, 4, 1, [exact]).scale_hint == [1, 1, 1, 2, 2]

    def test_a_level_that_states_no_shape_is_still_usable(self):
        # Computed levels are ceil by construction; the check exists for on-disk
        # ones, and an unstated extent is not evidence of disagreement.
        shape = [1, 1, 1, 4097, 4097]
        unstated = _Level((1, 1, 1, 2, 2), (), "nearest", False)
        assert _tile_read(shape, 3, 4, 1, [unstated]).scale_hint == [1, 1, 1, 2, 2]


class TestVolumeStaysWithinTheBudget:
    """A native ladder is advertised instead of the computed plan, so nothing
    upstream applies the 3-D voxel budget to it (biopb/biopb#891)."""

    BUDGET = 448**3

    @staticmethod
    def _native(shape, scales):
        """A Y/X-only native ladder, which is what NGFF writers usually emit."""
        return tuple(
            _Level(
                scale=(1, s, s),
                shape=(shape[0], -(-shape[1] // s), -(-shape[2] // s)),
                method="precompute",
                native=True,
            )
            for s in scales
        )

    def _voxels(self, plan):
        return plan["depth"] * plan["height"] * plan["width"]

    @pytest.mark.parametrize(
        "shape,scales",
        [
            ([1500, 2048, 2048], (2, 4)),  # lightsheet: 4.4x budget unbounded
            ([4000, 4096, 4096], (2, 4, 8)),  # EM: 11.7x
            ([1800, 2048, 2048], (2, 4)),  # 5.2x, and under the client byte cap
        ],
    )
    def test_a_yx_only_native_ladder_does_not_escape_the_budget(self, shape, scales):
        plan, reason = _volume_plan(
            shape, ["z", "y", "x"], "uint16", self._native(shape, scales)
        )
        assert reason is None
        assert self._voxels(plan) <= self.BUDGET
        # Fell back to the computed plan, so the read is no longer addressed to
        # a stored level -- the method has to come off with the scale.
        assert plan["reduction_method"] is None
        assert plan["scale_hint"][0] > 1, "z must be scaled to fit a deep stack"

    def test_a_ladder_that_already_fits_is_used_as_advertised(self):
        # Deep enough in Y/X that the volume is bounded incidentally: 800x256x256
        # is 0.6x the budget, so the native level stands and stays addressable.
        shape = [800, 8192, 8192]
        plan, reason = _volume_plan(
            shape, ["z", "y", "x"], "uint16", self._native(shape, (2, 4, 8, 16, 32))
        )
        assert reason is None
        assert plan["scale_hint"] == [1, 32, 32]
        assert plan["reduction_method"] == "precompute"
        assert self._voxels(plan) <= self.BUDGET

    @pytest.mark.parametrize(
        "shape", [[4000, 4000, 4000], [20000, 20000, 20000], [100000, 8192, 8192]]
    )
    def test_the_computed_fallback_cannot_come_back_over_budget(self, shape):
        """Phase 2 stops each axis at the 448 floor, and 448**3 is the budget.

        So the fallback always fits, however extreme the source. The refusal
        beneath it in `_volume_plan` is therefore defensive -- kept because that
        invariant lives in another module and nothing here would notice it
        changing.
        """
        plan, reason = _volume_plan(
            shape, ["z", "y", "x"], "uint16", self._native(shape, (2, 4, 8))
        )
        assert reason is None
        assert self._voxels(plan) <= self.BUDGET
