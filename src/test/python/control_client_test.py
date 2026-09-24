"""biopb.control: the plane the control names, and the credential that goes with it."""

from __future__ import annotations

import io
import json

import pytest
from biopb import control
from biopb.control import _client


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("BIOPB_TENSOR_URL", raising=False)
    monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)
    monkeypatch.setattr("biopb._credentials.read_credential", lambda: None)


def _serve(monkeypatch, payload, status=200):
    """Answer every urlopen with *payload*; return the requests made."""
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        resp = io.BytesIO(json.dumps(payload).encode())
        resp.status = status
        resp.__enter__ = lambda: resp
        resp.__exit__ = lambda *a: None
        return resp

    monkeypatch.setattr(_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        "biopb.control._data_plane.urllib.request.urlopen", fake_urlopen
    )
    return seen


def _refuse(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(_client.urllib.request, "urlopen", boom)
    monkeypatch.setattr("biopb.control._data_plane.urllib.request.urlopen", boom)


class TestDataPlane:
    def test_names_the_plane_and_its_credential(self, monkeypatch):
        monkeypatch.setattr("biopb._credentials.read_credential", lambda: "cred")
        _serve(monkeypatch, {"data_plane": {"grpc_url": "grpc://x:5"}})
        assert control.data_plane() == {"url": "grpc://x:5", "token": "cred"}

    def test_the_env_token_wins_over_the_file(self, monkeypatch):
        monkeypatch.setattr("biopb._credentials.read_credential", lambda: "cred")
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env")
        _serve(monkeypatch, {"data_plane": {"grpc_url": "grpc://x:5"}})
        assert control.data_plane()["token"] == "env"

    def test_no_control_is_none(self, monkeypatch):
        _refuse(monkeypatch)
        assert control.data_plane() is None

    def test_a_control_naming_no_plane_is_none(self, monkeypatch):
        _serve(monkeypatch, {"data_plane": {}})
        assert control.data_plane() is None


class TestEnsureDataPlane:
    def test_posts_with_the_token_header(self, monkeypatch):
        monkeypatch.setattr("biopb._credentials.read_credential", lambda: "tok")
        seen = _serve(monkeypatch, {"data_plane": {"grpc_url": "grpc://x:5"}})

        assert control.ensure_data_plane(timeout=5.0) == {
            "url": "grpc://x:5",
            "token": "tok",
        }
        (req,) = seen
        assert req.get_method() == "POST"
        assert "client_timeout=5.0" in req.full_url
        # urllib lower-cases header keys on the Request.
        assert req.get_header("X-biopb-token") == "tok"

    def test_no_header_when_tokenless(self, monkeypatch):
        seen = _serve(monkeypatch, {"data_plane": {"grpc_url": "grpc://x:5"}})
        control.ensure_data_plane(timeout=5.0)
        assert seen[0].get_header("X-biopb-token") is None

    def test_no_control_is_none(self, monkeypatch):
        _refuse(monkeypatch)
        assert control.ensure_data_plane(timeout=1.0) is None

    def test_an_answer_without_a_url_is_none(self, monkeypatch):
        _serve(monkeypatch, {"data_plane": {"state": "failed"}})
        assert control.ensure_data_plane(timeout=1.0) is None


def test_importing_it_does_not_import_pyarrow():
    import subprocess
    import sys

    code = "import sys, biopb.control; print('pyarrow' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
