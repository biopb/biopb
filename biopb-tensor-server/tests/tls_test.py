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


def _self_signed_cert(
    valid_days: int = 3650,
    dns_names: tuple[str, ...] = ("localhost",),
    ip_addresses: tuple[str, ...] = ("127.0.0.1", "::1"),
) -> tuple[bytes, bytes]:
    """A throwaway self-signed cert (PEM) + key (PEM), SANs localhost/127.0.0.1.

    A negative *valid_days* backdates notAfter, minting an already-expired cert.
    The SAN arguments exist for the certificates an operator supplies, which
    carry the names clients dial and nothing else -- notably no loopback.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cn = dns_names[0] if dns_names else "localhost"
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = datetime.datetime.now(datetime.timezone.utc)
    not_after = now + datetime.timedelta(days=valid_days)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Backdated a day against clock skew -- and further still when the span
        # itself is in the past, so an expired cert is orderly rather than absurd.
        .not_valid_before(min(now, not_after) - datetime.timedelta(days=1))
        .not_valid_after(not_after)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(d) for d in dns_names]
                + [x509.IPAddress(ipaddress.ip_address(i)) for i in ip_addresses]
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
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

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


@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_expired_cert_fails_with_its_actual_reason(tmp_path, monkeypatch):
    """An expired cert is refused, and the client says so (biopb/biopb#913).

    The pin is the trust anchor, not an exemption from validity: gRPC checks
    notAfter on the anchor even when the anchor is the presented leaf. Left to the
    transport this surfaces as `FlightUnavailableError: failed to connect to all
    addresses`, with `certificate has expired` only in gRPC's stderr log -- an
    outage on a date nobody was watching, presenting as a network fault.
    """
    from biopb.tensor import TensorFlightClient
    from biopb.tensor._tls import TlsCertExpiredError
    from biopb_tensor_server import TensorFlightServer

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

    cert_pem, key_pem = _self_signed_cert(valid_days=-30)
    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.mark_ready()
    _serve(server)
    try:
        with pytest.raises(TlsCertExpiredError) as excinfo:
            TensorFlightClient(f"grpcs://localhost:{server.port}")
        assert "cert init --force" in str(excinfo.value)
    finally:
        server.shutdown()


# --- the co-located HTTP sidecar over a TLS flight plane --------------------
# `launch` runs the Flight server and the FastAPI sidecar in one process, and the
# sidecar reaches Flight over loopback as an ordinary TensorFlightClient. That
# used to make TLS and the sidecar mutually exclusive (the entrypoint refused the
# combination). The sidecar now trusts the very cert that plane serves, read off
# local disk -- an explicit anchor, not a trust-on-first-use pin.


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_sidecar_reads_over_tls_without_pinning(
    simple_zarr_array, tmp_path, monkeypatch
):
    """The sidecar serves data off a TLS plane, and never touches the pin store.

    The pin-store assertion is what distinguishes the explicit-anchor path from a
    TOFU fallback: both would return 200, but only TOFU records a pin -- which
    would then break the sidecar whenever the cert is rotated.
    """
    import zarr
    from biopb._locations import tls_known_hosts
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.tls import cert_fingerprint
    from biopb_tensor_server.serving.http_server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert()

    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        app = create_app(
            flight_location=f"grpcs://localhost:{server.port}",
            token=None,
            tls_fingerprint=cert_fingerprint(cert_pem),
        )
        with TestClient(app, raise_server_exceptions=True) as tc:
            resp = tc.get("/api/sources")
            assert resp.status_code == 200
            assert any(s["source_id"] == "img" for s in resp.json())

        assert not tls_known_hosts().exists(), (
            "the sidecar holds the cert already -- it must not TOFU-pin it"
        )
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_sidecar_reads_over_a_cert_that_does_not_name_loopback(
    simple_zarr_array, tmp_path, monkeypatch
):
    """The shape of an operator's own certificate: the public name, and nothing else.

    The sidecar dials the plane over loopback, and gRPC checks the dialed name
    against the SANs however trust was established -- so this used to fail every
    request with "Peer name 127.0.0.1 is not in peer certificate", surfacing to
    the caller as a bare 502 (biopb/biopb#916). Anchoring on the fingerprint
    instead of the PEM is what earns the hostname override that fixes it.
    """
    import zarr
    from biopb._locations import tls_known_hosts
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.tls import cert_fingerprint
    from biopb_tensor_server.serving.http_server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert(
        dns_names=("gpu-051.hpc.example",), ip_addresses=()
    )

    server = TensorFlightServer(
        "grpc://127.0.0.1:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        app = create_app(
            flight_location=f"grpcs://127.0.0.1:{server.port}",
            token=None,
            tls_fingerprint=cert_fingerprint(cert_pem),
        )
        with TestClient(app, raise_server_exceptions=True) as tc:
            resp = tc.get("/api/sources")
            assert resp.status_code == 200, resp.text
            assert any(s["source_id"] == "img" for s in resp.json())

        assert not tls_known_hosts().exists(), (
            "verified-first-use, not TOFU: nothing may be pinned"
        )
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_sidecar_refuses_a_plane_presenting_a_different_certificate(
    simple_zarr_array, tmp_path, monkeypatch
):
    """The other half of the trade: the fingerprint is checked on every connect.

    Reaching the wire to resolve trust is only acceptable because a mismatch is
    refused -- this is verified-first-use, not the trust-on-first-use the pin
    store does for a plane whose certificate the client cannot already know.
    """
    import zarr
    from biopb.tensor._tls import clear_pin_cache
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.tls import cert_fingerprint
    from biopb_tensor_server.serving.http_server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))
    clear_pin_cache()

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    served, key_pem = _self_signed_cert()
    other, _ = _self_signed_cert()  # a different cert with the same names

    server = TensorFlightServer(
        "grpc://127.0.0.1:0", tls_cert_chain=served, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        app = create_app(
            flight_location=f"grpcs://127.0.0.1:{server.port}",
            token=None,
            tls_fingerprint=cert_fingerprint(other),
        )
        with TestClient(app, raise_server_exceptions=False) as tc:
            assert tc.get("/api/sources").status_code >= 500
    finally:
        clear_pin_cache()
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_a_local_sdk_client_reads_a_plane_serving_a_byo_cert(
    simple_zarr_array, tmp_path, monkeypatch
):
    """The other local consumer of #916: `biopb tensor` and biopb-mcp.

    `local_fingerprint` used to anchor on state/biopb/tls/server-cert.pem, which
    a `--tls-cert` never writes -- so a plane serving an operator's own cert was
    unreachable from this machine, with remediation advice (`cert init`) that
    would have produced a certificate the plane was not serving. The plane now
    publishes what it serves, keyed by port.
    """
    import numpy as np
    import zarr
    from biopb import _data_plane, _tls_material, _tls_record
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    # An operator's own certificate: the names clients dial, no loopback, and
    # nothing of it in the state tree.
    cert_pem, key_pem = _self_signed_cert(
        dns_names=("gpu-051.hpc.example",), ip_addresses=()
    )

    server = TensorFlightServer(
        "grpc://127.0.0.1:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        url = f"grpcs://127.0.0.1:{server.port}"
        # Without the record there is nothing on this disk that identifies the
        # plane, and that is an error rather than a silent fall back to TOFU.
        with pytest.raises(_data_plane.LocalTrustError):
            _data_plane.local_fingerprint(url)

        _tls_record.publish(
            server.port, _tls_material.fingerprint(_tls_material.leaf_pem(cert_pem))
        )
        client = TensorFlightClient(
            url, tls_fingerprint=_data_plane.local_fingerprint(url)
        )
        np.testing.assert_array_equal(client.get_tensor("img").compute(), arr[:])
        client.close()

        from biopb._locations import tls_known_hosts

        assert not tls_known_hosts().exists(), "verified, not pinned"
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_sidecar_dialing_plaintext_at_a_tls_plane_fails(simple_zarr_array):
    """A grpc:// sidecar against a TLS plane must fail, not silently degrade."""
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.serving.http_server import create_app
    from fastapi.testclient import TestClient

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert()

    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        app = create_app(flight_location=f"grpc://localhost:{server.port}", token=None)
        with TestClient(app, raise_server_exceptions=True) as tc:
            assert tc.get("/api/sources").status_code == 502
    finally:
        server.shutdown()


# --- hostname verification vs. the pin (biopb/biopb#606) ---------------------
# Pinning supplies the trust anchor; gRPC still matches the *dialed* name against
# the cert's SANs. A container that minted its cert before anyone knew what name
# clients would dial it by therefore pins fine and then fails every handshake.
# Where the anchor is the presented leaf, that second check is redundant and the
# client substitutes a name the cert does list.


def _cert_with_sans(*dns_names: str) -> tuple[bytes, bytes]:
    """A self-signed leaf whose SANs are exactly *dns_names* -- no localhost."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, dns_names[0])])
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
            x509.SubjectAlternativeName([x509.DNSName(n) for n in dns_names]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return (
        cert.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ),
    )


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_override_hostname_is_what_makes_a_mismatched_cert_connect(simple_zarr_array):
    """The raw mechanism: a pinned cert alone is not enough to dial `localhost`.

    The garbage-override row is the important one -- verification is still
    running, it is just matching a different name. (`disable_server_verification`
    would skip the chain entirely, making the pin decorative; deliberately not
    used anywhere.)
    """
    import zarr
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _cert_with_sans("wrong.example", "alt.example")

    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    loc = f"grpc+tls://localhost:{server.port}"
    opts = flight.FlightCallOptions(timeout=10.0)
    try:
        # Pinned as the trust anchor, but `localhost` is not in the SANs.
        pinned_only = flight.FlightClient(loc, tls_root_certs=cert_pem)
        with pytest.raises(flight.FlightError):
            list(pinned_only.list_flights(options=opts))
        pinned_only.close()

        # Same anchor, verifying against a name the cert does carry.
        overridden = flight.FlightClient(
            loc, tls_root_certs=cert_pem, override_hostname="wrong.example"
        )
        assert any(b"img" in fi.descriptor.command for fi in overridden.list_flights())
        overridden.close()

        # A name the cert does *not* carry still fails: this substitutes the name
        # being matched, it does not switch verification off.
        bogus = flight.FlightClient(
            loc, tls_root_certs=cert_pem, override_hostname="nope.invalid"
        )
        with pytest.raises(flight.FlightError):
            list(bogus.list_flights(options=opts))
        bogus.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_sdk_client_derives_the_override_and_reads(
    simple_zarr_array, tmp_path, monkeypatch
):
    """End-to-end #606: no operator ceremony for a cert that omits the dialed name.

    Without the derivation this exact setup pins successfully and then fails every
    handshake -- the headless-container shape #604 targets. The ``.compute()``
    matters as much as the connect: it proves the override reaches the chunk-fetch
    pool as ordinary graph data, like the anchor does.
    """
    import numpy as np
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _cert_with_sans("wrong.example", "alt.example")

    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        client = TensorFlightClient(f"grpcs://localhost:{server.port}")
        assert client._tls_trust.override_hostname == "wrong.example"
        np.testing.assert_array_equal(client.get_tensor("img").compute(), arr[:])
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
@pytest.mark.skipif(not _crypto_available(), reason="cryptography not available")
def test_a_cert_that_lists_the_dialed_name_gets_no_override(
    simple_zarr_array, tmp_path, monkeypatch
):
    """The normal case is untouched: nothing is substituted when nothing is wrong."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    cert_pem, key_pem = _self_signed_cert()  # SANs: localhost + 127.0.0.1

    server = TensorFlightServer(
        "grpc://localhost:0", tls_cert_chain=cert_pem, tls_private_key=key_pem
    )
    server.register_source("img", ZarrAdapter(arr, "img", ["y", "x"]))
    server.mark_ready()
    _serve(server)
    try:
        client = TensorFlightClient(f"grpcs://localhost:{server.port}")
        assert client._tls_trust.override_hostname is None
        client.close()
    finally:
        server.shutdown()
