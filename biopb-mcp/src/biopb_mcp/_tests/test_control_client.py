"""Unit tests for the control client's credential-carrying behavior (#470).

``_control_client`` reads the control's local credential file and carries the
token on its gated ``/api/data_plane/ensure`` POST, so an agent-spawned biopb-mcp
(which never inherited ``BIOPB_TENSOR_TOKEN``) can drive a token-gated control.
"""

from __future__ import annotations

import io
import json

import pytest

from biopb_mcp import _control_client


@pytest.fixture(autouse=True)
def _no_real_credential(monkeypatch):
    """Default the credential to absent so a test never reads the dev machine's."""
    monkeypatch.setattr("biopb._credentials.read_credential", lambda: None)


class TestAuthHeaders:
    def test_empty_when_no_credential(self):
        assert _control_client._auth_headers() == {}

    def test_carries_token_when_present(self, monkeypatch):
        monkeypatch.setattr("biopb._credentials.read_credential", lambda: "tok-123")
        assert _control_client._auth_headers() == {"X-Biopb-Token": "tok-123"}


class TestEnsureCarriesToken:
    def _capture_request(self, monkeypatch):
        """Stub urlopen to record the Request and return a minimal snapshot reply."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["req"] = req
            body = json.dumps({"data_plane": {"state": "serving"}}).encode()
            resp = io.BytesIO(body)
            resp.__enter__ = lambda: resp
            resp.__exit__ = lambda *a: None
            return resp

        monkeypatch.setattr(_control_client.urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_posts_with_token_header(self, monkeypatch):
        monkeypatch.setattr("biopb._credentials.read_credential", lambda: "tok-xyz")
        captured = self._capture_request(monkeypatch)
        snap = _control_client.ensure_data_plane(timeout=5.0)
        assert snap == {"state": "serving"}
        # urllib lower-cases header keys on the Request.
        assert captured["req"].get_header("X-biopb-token") == "tok-xyz"
        assert captured["req"].get_method() == "POST"

    def test_posts_without_header_when_tokenless(self, monkeypatch):
        captured = self._capture_request(monkeypatch)
        _control_client.ensure_data_plane(timeout=5.0)
        assert captured["req"].get_header("X-biopb-token") is None
