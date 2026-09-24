"""biopb.tensor.Connection: where it dials, with what, and what it says when it cannot.

The Flight client is faked; no server runs.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from biopb.tensor import _connection
from biopb.tensor._connection import Connection, connect_error_message


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("BIOPB_TENSOR_URL", raising=False)
    monkeypatch.delenv("BIOPB_TENSOR_TOKEN", raising=False)
    # A credential is ON DISK throughout, so the rule that it goes only to the
    # control's own address is tested against a file that exists.
    monkeypatch.setattr("biopb._credentials.read_credential", lambda: "cred-tok")
    monkeypatch.setattr(_connection, "ensure_data_plane", lambda **_: None)
    monkeypatch.setattr(_connection.time, "sleep", lambda _s: None)


def _plane(monkeypatch, url="grpc://control:9", token="cred-tok"):
    ensure = MagicMock(return_value={"url": url, "token": token})
    monkeypatch.setattr(_connection, "ensure_data_plane", ensure)
    return ensure


def _clients(monkeypatch, *behaviours):
    """Fake Flight clients, one per dial; return the ``(url, token)`` dialed.

    Each behaviour is a health dict, or an exception the constructor raises.
    The last one repeats.
    """
    dials = []

    def factory(url, token=None, tls_fingerprint=None):
        dials.append((url, token))
        b = behaviours[min(len(dials), len(behaviours)) - 1]
        if isinstance(b, Exception):
            raise b
        client = MagicMock()
        client.health_check.return_value = b
        return client

    monkeypatch.setattr(_connection, "TensorFlightClient", factory)
    return dials


SERVING = {"status": "SERVING"}


class TestWhereItDials:
    def test_the_control_names_the_plane_and_its_credential_goes_with_it(
        self, monkeypatch
    ):
        _plane(monkeypatch)
        dials = _clients(monkeypatch, SERVING)

        conn = Connection()
        assert conn.connect() is True

        assert dials == [("grpc://control:9", "cred-tok")]
        assert conn.url == "grpc://control:9"
        assert conn.client is not None
        assert conn.last_message == ""

    def test_no_control_dials_nothing(self, monkeypatch):
        dials = _clients(monkeypatch, SERVING)

        conn = Connection()
        assert conn.connect() is False

        assert dials == []
        assert conn.client is None
        assert "biopb control start" in conn.last_message

    def test_the_env_url_bypasses_the_control_and_the_credential_file(
        self, monkeypatch
    ):
        monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://elsewhere:7")
        ensure = _plane(monkeypatch)
        dials = _clients(monkeypatch, SERVING)

        assert Connection().connect() is True

        ensure.assert_not_called()
        assert dials == [("grpc://elsewhere:7", None)]

    def test_the_env_url_carries_the_env_token(self, monkeypatch):
        monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://elsewhere:7")
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-tok")
        dials = _clients(monkeypatch, SERVING)

        Connection().connect()

        assert dials == [("grpc://elsewhere:7", "env-tok")]

    def test_a_given_url_is_dialed_with_the_given_token_only(self, monkeypatch):
        ensure = _plane(monkeypatch)
        dials = _clients(monkeypatch, SERVING)

        Connection().connect("grpc://typed:1")

        ensure.assert_not_called()
        assert dials == [("grpc://typed:1", None)]


class TestWaiting:
    def test_the_control_s_plane_is_waited_through_refusal_and_scan(self, monkeypatch):
        _plane(monkeypatch)
        dials = _clients(
            monkeypatch,
            RuntimeError("failed to connect"),
            {"status": "STARTING", "source_count": 3},
            SERVING,
        )

        conn = Connection()
        assert conn.connect() is True
        assert len(dials) == 3

    def test_a_plane_named_elsewhere_that_refuses_is_not_retried(self, monkeypatch):
        dials = _clients(monkeypatch, RuntimeError("Connection refused"))

        conn = Connection()
        assert conn.connect("grpc://typed:1") is False

        assert len(dials) == 1
        assert "Cannot reach" in conn.last_message

    def test_a_plane_named_elsewhere_is_waited_through_its_scan(self, monkeypatch):
        dials = _clients(monkeypatch, {"status": "STARTING"}, SERVING)

        assert Connection().connect("grpc://typed:1") is True
        assert len(dials) == 2

    def test_gives_up_at_the_deadline_saying_it_is_still_starting(self, monkeypatch):
        _plane(monkeypatch)
        _clients(monkeypatch, {"status": "STARTING", "source_count": 0})
        clock = iter(range(0, 1000, 10))
        monkeypatch.setattr(_connection.time, "monotonic", lambda: next(clock))

        conn = Connection()
        assert conn.connect(timeout=30) is False

        assert conn.client is None
        assert "starting" in conn.last_message
        assert "0 sources registered so far" in conn.last_message

    def test_a_token_the_plane_rejects_fails_although_health_answers(self, monkeypatch):
        """health is ungated, so a connect has to make one call that is not."""
        dials = []

        def factory(url, token=None, tls_fingerprint=None):
            dials.append(url)
            client = MagicMock()
            client.health_check.return_value = SERVING
            client.query_sources.side_effect = RuntimeError("Unauthenticated")
            return client

        monkeypatch.setattr(_connection, "TensorFlightClient", factory)

        conn = Connection()
        assert conn.connect("grpc://typed:1", token="bad") is False
        assert "rejected the token" in conn.last_message


class TestConnectErrorMessage:
    URL = "grpc://host:9"
    AUTH = RuntimeError("FlightUnauthenticatedError: token required")

    def test_the_control_s_address_points_at_the_control(self):
        msg = connect_error_message(self.AUTH, self.URL, None)
        assert "credential file held none" in msg
        assert "biopb control start" in msg

    def test_the_env_address_says_the_file_was_not_used(self):
        msg = connect_error_message(self.AUTH, self.URL, None, origin="env")
        assert "BIOPB_TENSOR_TOKEN" in msg
        assert "credential file was not used" in msg

    def test_a_typed_address_just_needs_a_token(self):
        msg = connect_error_message(self.AUTH, self.URL, None, origin="manual")
        assert msg.endswith("needs a token.")

    def test_a_rejected_token(self):
        msg = connect_error_message(self.AUTH, self.URL, "bad")
        assert "rejected the token" in msg

    def test_unreachable(self):
        msg = connect_error_message(
            RuntimeError("failed to connect to all addresses"), self.URL, None
        )
        assert "Cannot reach" in msg

    def test_anything_else_is_echoed(self):
        msg = connect_error_message(ValueError("odd"), self.URL, None)
        assert "odd" in msg


_FAKE_CERT = b"-----BEGIN CERTIFICATE-----\nZmFrZQ==\n-----END CERTIFICATE-----\n"


def _seed_cert(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))
    from biopb._locations import tls_server_cert

    path = tls_server_cert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_FAKE_CERT)
    return path


def test_a_local_tls_plane_is_dialed_with_its_own_anchor(monkeypatch, tmp_path):
    from biopb import _tls_material

    _seed_cert(monkeypatch, tmp_path)
    seen = {}

    def factory(url, token=None, tls_fingerprint=None):
        seen["fp"] = tls_fingerprint
        client = MagicMock()
        client.health_check.return_value = SERVING
        return client

    monkeypatch.setattr(_connection, "TensorFlightClient", factory)
    Connection().connect("grpcs://127.0.0.1:8815")
    assert seen["fp"] == _tls_material.fingerprint(_tls_material.leaf_pem(_FAKE_CERT))


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="chmod 000 blocks neither root nor Windows, so the read would succeed",
)
def test_an_unreadable_cert_is_not_reported_as_an_auth_problem(monkeypatch, tmp_path):
    """An unreadable cert says "Permission denied", which is an auth marker."""
    cert = _seed_cert(monkeypatch, tmp_path)
    cert.chmod(0o000)
    try:
        conn = Connection()
        assert conn.connect("grpcs://127.0.0.1:8815") is False
    finally:
        cert.chmod(0o600)
    assert "Authentication" not in conn.last_message
    assert str(cert) in conn.last_message
