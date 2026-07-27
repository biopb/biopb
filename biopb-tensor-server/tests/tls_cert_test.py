"""Self-signed cert generation + the `cert init` / `serve --tls` surface (#604 item 2).

The tensor server mints its own self-signed cert (no CA) so a headless install
can stand up TLS on its own; clients pin it via TOFU. These tests isolate the
state tree to a tmp dir so they never touch the real cert store.
"""

import importlib.util
import ipaddress
import os
import threading
import time

import pytest
from typer.testing import CliRunner


def _crypto_available() -> bool:
    return importlib.util.find_spec("cryptography") is not None


def _zarr_available() -> bool:
    return importlib.util.find_spec("zarr") is not None


pytestmark = pytest.mark.skipif(
    not _crypto_available(), reason="cryptography not available"
)


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


# --- cert module ------------------------------------------------------------


def test_collect_san_includes_loopback():
    from biopb_tensor_server.core.tls import collect_san_hosts

    dns, ips = collect_san_hosts()
    assert "localhost" in dns
    assert "127.0.0.1" in ips


def test_generated_cert_is_self_signed_with_sans():
    from biopb_tensor_server.core.tls import generate_self_signed_cert
    from cryptography import x509

    cert_pem, key_pem = generate_self_signed_cert(
        ["localhost", "myhost"], ["127.0.0.1", "10.0.0.5"]
    )
    assert b"BEGIN CERTIFICATE" in cert_pem
    assert b"PRIVATE KEY" in key_pem

    cert = x509.load_pem_x509_certificate(cert_pem)
    # Self-signed: issuer == subject.
    assert cert.issuer == cert.subject
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    assert "myhost" in san.get_values_for_type(x509.DNSName)
    ips = san.get_values_for_type(x509.IPAddress)
    assert ipaddress.ip_address("127.0.0.1") in ips
    assert ipaddress.ip_address("10.0.0.5") in ips


def test_ensure_server_cert_generates_then_reuses():
    from biopb._locations import tls_server_cert, tls_server_key
    from biopb_tensor_server.core.tls import ensure_server_cert

    assert not tls_server_cert().exists()
    cert1, key1 = ensure_server_cert()
    assert tls_server_cert().exists() and tls_server_key().exists()
    # A second call reuses the on-disk pair rather than minting a new one.
    cert2, key2 = ensure_server_cert()
    assert cert1 == cert2 and key1 == key2

    if os.name == "posix":
        # The key is a secret -> owner-only; the cert is public.
        assert (tls_server_key().stat().st_mode & 0o777) == 0o600


def test_ensure_regenerate_mints_new_cert():
    from biopb_tensor_server.core.tls import cert_fingerprint, ensure_server_cert

    cert1, _ = ensure_server_cert()
    cert2, _ = ensure_server_cert(regenerate=True)
    assert cert_fingerprint(cert1) != cert_fingerprint(cert2)


def test_generate_without_cryptography_raises_actionable(monkeypatch):
    """With the opt-in [tls] extra absent, cert gen fails with an install hint.

    cryptography is deliberately not in the default closure (biopb/biopb#355);
    hide it and confirm the TLS path degrades to an actionable error, not an
    opaque ImportError.
    """
    import sys

    from biopb_tensor_server.core.tls import generate_self_signed_cert

    monkeypatch.setitem(sys.modules, "cryptography", None)
    with pytest.raises(RuntimeError, match=r"biopb-tensor-server\[tls\]"):
        generate_self_signed_cert(["localhost"], ["127.0.0.1"])


def test_cert_init_without_cryptography_advises_cleanly(monkeypatch):
    """`cert init` without the extra exits non-zero with advice, not a traceback."""
    import sys

    from biopb_tensor_server.cli import app

    monkeypatch.setitem(sys.modules, "cryptography", None)
    result = CliRunner().invoke(app, ["cert", "init"])
    assert result.exit_code == 2
    assert "biopb-tensor-server[tls]" in result.output
    assert "Traceback" not in result.output


def test_serve_tls_without_cryptography_exits_cleanly(monkeypatch):
    """`serve --tls` (auto-gen) without the extra raises a clean typer.Exit."""
    import sys

    import typer
    from biopb_tensor_server.cli import _resolve_tls_material

    monkeypatch.setitem(sys.modules, "cryptography", None)
    with pytest.raises(typer.Exit):
        _resolve_tls_material(True, None, None)


def test_byo_cert_needs_no_cryptography(tmp_path, monkeypatch):
    """--tls-cert/--tls-key read files directly -- the crypto-free escape hatch."""
    import sys

    from biopb_tensor_server.cli import _resolve_tls_material
    from biopb_tensor_server.core.tls import generate_self_signed_cert

    cert_pem, key_pem = generate_self_signed_cert(["localhost"], ["127.0.0.1"])
    cf, kf = tmp_path / "c.pem", tmp_path / "k.pem"
    cf.write_bytes(cert_pem)
    kf.write_bytes(key_pem)

    monkeypatch.setitem(sys.modules, "cryptography", None)  # extra absent
    cert, key = _resolve_tls_material(False, cf, kf)
    assert cert == cert_pem and key == key_pem


# --- CLI: cert init ---------------------------------------------------------


def test_cert_init_generates_and_prints_fingerprint():
    from biopb._locations import tls_server_cert
    from biopb_tensor_server.cli import app
    from biopb_tensor_server.core.tls import cert_fingerprint

    result = CliRunner().invoke(app, ["cert", "init"])
    assert result.exit_code == 0, result.output
    assert tls_server_cert().exists()
    fp = cert_fingerprint(tls_server_cert().read_bytes())
    assert fp in result.output


def test_cert_init_idempotent_without_force():
    from biopb._locations import tls_server_cert
    from biopb_tensor_server.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["cert", "init"]).exit_code == 0
    before = tls_server_cert().read_bytes()
    out = runner.invoke(app, ["cert", "init"])
    assert out.exit_code == 0
    assert "already present" in out.output
    assert tls_server_cert().read_bytes() == before  # unchanged


def test_cert_init_force_rotates():
    from biopb._locations import tls_server_cert
    from biopb_tensor_server.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["cert", "init"]).exit_code == 0
    before = tls_server_cert().read_bytes()
    out = runner.invoke(app, ["cert", "init", "--force"])
    assert out.exit_code == 0
    assert tls_server_cert().read_bytes() != before  # rotated


# --- end to end: generated cert actually serves + a TOFU client reads -------


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_generated_cert_serves_and_tofu_client_reads(simple_zarr_array):
    import numpy as np
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.tls import ensure_server_cert

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = ensure_server_cert()  # into the isolated state tree

    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    time.sleep(1)
    try:
        client = TensorFlightClient(f"grpcs://localhost:{server.port}")
        got = client.get_tensor("img").compute()
        np.testing.assert_array_equal(got, arr[:])
        client.close()
    finally:
        server.shutdown()
