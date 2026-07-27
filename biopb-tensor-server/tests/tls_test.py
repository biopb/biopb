"""Server-side TLS for TensorFlightServer (biopb/biopb#604, item 1 -- server half).

The server serves ``grpc+tls://`` when handed a cert chain + key; a client that
trusts the cert connects over TLS, and a plaintext client is refused. The client
SDK's own TLS trust plumbing (``TensorFlightClient(grpcs://...)``) lands
separately -- these tests drive a raw ``pyarrow.flight.FlightClient`` so they
exercise only the server half.
"""

import datetime
import importlib.util
import ipaddress
import threading
import time

import pyarrow.flight as flight
import pytest


def _zarr_available() -> bool:
    return importlib.util.find_spec("zarr") is not None


def _crypto_available() -> bool:
    return importlib.util.find_spec("cryptography") is not None


def _self_signed_cert() -> tuple[bytes, bytes]:
    """A throwaway self-signed cert (PEM) + key (PEM), SANs localhost/127.0.0.1."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
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
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def _serve(server):
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    time.sleep(1)
    return t


def test_ensure_tls_scheme_normalizes_shorthands():
    from biopb_tensor_server.serving.server import _ensure_tls_scheme

    assert _ensure_tls_scheme("grpc://0.0.0.0:8815") == "grpc+tls://0.0.0.0:8815"
    assert _ensure_tls_scheme("grpcs://host:8815") == "grpc+tls://host:8815"
    # Already the TLS form -- left untouched.
    assert _ensure_tls_scheme("grpc+tls://host:8815") == "grpc+tls://host:8815"


def test_tls_cert_without_key_is_rejected():
    from biopb_tensor_server import TensorFlightServer

    with pytest.raises(ValueError, match="together"):
        TensorFlightServer("grpc://localhost:0", tls_cert_chain=b"cert")
    with pytest.raises(ValueError, match="together"):
        TensorFlightServer("grpc://localhost:0", tls_private_key=b"key")


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_trusting_client_reads_over_tls(simple_zarr_array):
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert()

    server = TensorFlightServer(
        "grpc://localhost:0",
        tls_cert_chain=cert_pem,
        tls_private_key=key_pem,
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        # A client that trusts the (self-signed) cert as its root connects.
        client = flight.FlightClient(
            f"grpc+tls://localhost:{server.port}", tls_root_certs=cert_pem
        )
        flights = list(client.list_flights())
        assert any(b"img" in fi.descriptor.command for fi in flights)
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_plaintext_client_is_refused_by_tls_server(simple_zarr_array):
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert()

    server = TensorFlightServer(
        "grpc://localhost:0",
        tls_cert_chain=cert_pem,
        tls_private_key=key_pem,
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        # Plaintext gRPC against a TLS port: the TLS handshake can't complete,
        # so the transport surfaces FlightUnavailableError (a FlightError).
        client = flight.FlightClient(f"grpc://localhost:{server.port}")
        with pytest.raises(flight.FlightError):
            list(client.list_flights(options=flight.FlightCallOptions(timeout=5.0)))
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_sdk_client_tofu_roundtrip(simple_zarr_array, tmp_path, monkeypatch):
    """TensorFlightClient over grpcs:// pins the server cert (TOFU) and reads it.

    End-to-end for biopb/biopb#604 item 1: the SDK client resolves TLS trust by
    pinning the presented cert on first connect, threads the pinned PEM through
    the chunk-fetch pool, and a lazy dask read succeeds over TLS.
    """
    import numpy as np
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    # Isolate the TOFU pin store (state/biopb/tls-known-hosts.json) to a tmp tree.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert()

    server = TensorFlightServer(
        "grpc://localhost:0",
        tls_cert_chain=cert_pem,
        tls_private_key=key_pem,
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        # grpcs:// -> TOFU fetch+pin the self-signed cert, then read over TLS.
        client = TensorFlightClient(f"grpcs://localhost:{server.port}")
        got = client.get_tensor("img").compute()
        np.testing.assert_array_equal(got, arr[:])
        # The cert was actually pinned for this host:port.
        from biopb._locations import tls_known_hosts

        assert f"localhost:{server.port}" in tls_known_hosts().read_text()
        client.close()
    finally:
        server.shutdown()
