"""TOFU certificate pinning for the tensor client (biopb/biopb#604).

Client-side unit tests, no live server: the TOFU state machine is driven by
monkeypatching the cert fetch, and the pin store / location parsing are checked
directly. The server-backed round-trip (a real ``grpcs://`` TensorFlightServer)
lives in ``biopb-tensor-server/tests/tls_test.py``, alongside the server package
whose CI installs it (biopb/biopb#579).
"""

import datetime
import ipaddress

import pytest
from biopb.tensor import _tls


@pytest.fixture(autouse=True)
def _isolate_pin_store(tmp_path, monkeypatch):
    """Point the pin store at a tmp state tree so tests never touch the real one.

    ``state_dir()`` honors ``$XDG_STATE_HOME`` first, and CI may set it, so we
    set it explicitly rather than relying on ``Path.home()``.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _make_cert(cn: str = "localhost") -> bytes:
    """A throwaway self-signed cert as PEM bytes (distinct per call via CN)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(cn), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


# --- location parsing (no crypto, no network) --------------------------------


def test_host_port_ignores_non_tls_schemes():
    assert _tls._host_port("grpc://host:8815") is None
    assert _tls._host_port("host:8815") is None
    assert _tls._host_port("grpc+tls://host:8815") == ("host", 8815)


def test_resolve_returns_none_for_plaintext():
    # Plaintext location needs no cert and must never touch the network.
    assert _tls.resolve_tls_root_certs("grpc://localhost:8815") is None


# --- pin store (no crypto) ---------------------------------------------------


def test_pin_store_roundtrip(tmp_path):
    store = tmp_path / "known.json"
    assert _tls._load_pins(store) == {}
    _tls._save_pin(store, "a:1", b"pem-a")
    _tls._save_pin(store, "b:2", b"pem-b")
    pins = _tls._load_pins(store)
    assert pins == {"a:1": "pem-a", "b:2": "pem-b"}


def test_load_pins_tolerates_missing_and_garbage(tmp_path):
    assert _tls._load_pins(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _tls._load_pins(bad) == {}


# --- TOFU state machine (crypto, fetch monkeypatched) ------------------------


def test_first_connect_pins_then_reuses(monkeypatch):
    pytest.importorskip("cryptography")
    cert = _make_cert()
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: cert)

    loc = "grpc+tls://host:8815"
    first = _tls.resolve_tls_root_certs(loc)
    assert first == cert
    # The pin was persisted, so a second resolve returns the same cert.
    assert _tls.resolve_tls_root_certs(loc) == cert


def test_changed_cert_raises_pin_mismatch(monkeypatch):
    pytest.importorskip("cryptography")
    cert_a = _make_cert("localhost")
    cert_b = _make_cert("localhost")
    assert cert_a != cert_b

    loc = "grpc+tls://host:8815"
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: cert_a)
    assert _tls.resolve_tls_root_certs(loc) == cert_a

    # Same host now presents a different cert -> refuse with a clear error.
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: cert_b)
    with pytest.raises(_tls.TlsPinMismatchError, match="does not match"):
        _tls.resolve_tls_root_certs(loc)


def test_distinct_hosts_pinned_independently(monkeypatch):
    pytest.importorskip("cryptography")
    cert_a = _make_cert()
    cert_b = _make_cert()
    certs = {"h1:1": cert_a, "h2:2": cert_b}
    monkeypatch.setattr(
        _tls, "_fetch_server_cert", lambda host, port: certs[f"{host}:{port}"]
    )
    assert _tls.resolve_tls_root_certs("grpc+tls://h1:1") == cert_a
    assert _tls.resolve_tls_root_certs("grpc+tls://h2:2") == cert_b
