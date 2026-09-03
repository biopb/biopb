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
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))


# --- cert module ------------------------------------------------------------


def test_collect_san_includes_loopback():
    from biopb_tensor_server.core.tls import collect_san_hosts

    dns, ips = collect_san_hosts()
    assert "localhost" in dns
    assert "127.0.0.1" in ips


def test_host_identity_is_resolved_once_per_process():
    """The SAN probes are cached: a stalling resolver is paid for once, not per cert."""
    from biopb_tensor_server.core import tls as tls_mod

    tls_mod._host_identity.cache_clear()
    assert tls_mod._host_identity.cache_info().misses == 0
    for _ in range(3):
        tls_mod.collect_san_hosts()
    assert tls_mod._host_identity.cache_info().misses == 1


def test_collect_san_hosts_returns_a_fresh_list_each_call():
    """The cached value is shared, so callers must not be able to mutate it."""
    from biopb_tensor_server.core.tls import collect_san_hosts

    dns, _ = collect_san_hosts()
    dns.append("mutated.example")
    again, _ = collect_san_hosts()
    assert "mutated.example" not in again


def test_a_stalling_name_probe_is_abandoned_not_awaited(monkeypatch):
    """A wedged resolver must cost a bounded blip, not block server startup."""
    from biopb_tensor_server.core import tls as tls_mod

    monkeypatch.setattr(tls_mod, "_NAME_PROBE_TIMEOUT_S", 0.1)
    started = threading.Event()

    def _never_returns():
        started.set()
        time.sleep(30)
        return "too.late.example"

    began = time.monotonic()
    assert tls_mod._bounded("stub", _never_returns, "fallback") == "fallback"
    assert time.monotonic() - began < 5
    assert started.is_set()  # it really ran; we abandoned it, not skipped it


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


def test_generated_cert_is_an_end_entity_server_cert():
    """BasicConstraints/KeyUsage/EKU, so a strict TLS stack accepts the leaf."""
    from biopb_tensor_server.core.tls import generate_self_signed_cert
    from cryptography import x509

    cert_pem, _ = generate_self_signed_cert(["localhost"], ["127.0.0.1"])
    cert = x509.load_pem_x509_certificate(cert_pem)

    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca is False
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku
    ku = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    assert ku.digital_signature and not ku.key_cert_sign


def test_validity_span_matches_the_documented_days():
    from biopb_tensor_server.core import tls as tls_mod
    from cryptography import x509

    cert_pem, _ = tls_mod.generate_self_signed_cert(["localhost"], [], days=10)
    cert = x509.load_pem_x509_certificate(cert_pem)
    span = cert.not_valid_after_utc - cert.not_valid_before_utc
    assert span.days == 10


def test_expiry_warning_fires_only_near_the_end_of_the_span():
    """Nothing else watches notAfter, and a pin does not excuse it (biopb/biopb#913).

    `days=0` mints a cert whose span ends at its backdated notBefore, i.e. one
    that is already expired -- the state an operator reaches by leaving a cert in
    place for its full validity.
    """
    from biopb_tensor_server.core import tls as tls_mod

    fresh, _ = tls_mod.generate_self_signed_cert(["localhost"], [])
    assert tls_mod.cert_expiry_warning(fresh) is None

    soon, _ = tls_mod.generate_self_signed_cert(["localhost"], [], days=10)
    assert "expires on" in tls_mod.cert_expiry_warning(soon)

    gone, _ = tls_mod.generate_self_signed_cert(["localhost"], [], days=0)
    assert "expired on" in tls_mod.cert_expiry_warning(gone)


def test_expiry_warning_is_advisory_not_fatal():
    """Unreadable material must not take down the TLS path it only annotates."""
    from biopb_tensor_server.core.tls import cert_expiry_warning

    assert cert_expiry_warning(b"not a certificate") is None


def test_cert_init_reports_an_expired_cert_it_reuses(monkeypatch):
    """The one command an operator runs to inspect the cert has to say it is dead."""
    from biopb._locations import tls_server_cert, tls_server_key
    from biopb_tensor_server.cli import app
    from biopb_tensor_server.core.tls import generate_self_signed_cert

    cert_pem, key_pem = generate_self_signed_cert(["localhost"], ["127.0.0.1"], days=0)
    tls_server_cert().parent.mkdir(parents=True, exist_ok=True)
    tls_server_cert().write_bytes(cert_pem)
    tls_server_key().write_bytes(key_pem)

    result = CliRunner().invoke(app, ["cert", "init"])
    assert result.exit_code == 0, result.output
    assert "expired on" in result.output
    assert "--force" in result.output


def test_serve_tls_warns_on_an_expired_byo_cert(tmp_path, capsys):
    """A BYO cert gets the same notice -- and still without `cryptography` required."""
    import sys

    from biopb_tensor_server.cli import _resolve_tls_material
    from biopb_tensor_server.core.tls import generate_self_signed_cert

    cert_pem, key_pem = generate_self_signed_cert(["localhost"], [], days=0)
    cf, kf = tmp_path / "c.pem", tmp_path / "k.pem"
    cf.write_bytes(cert_pem)
    kf.write_bytes(key_pem)

    cert, _ = _resolve_tls_material(False, cf, kf)
    assert cert == cert_pem  # advisory only: the cert is still served
    assert "expired" in capsys.readouterr().out

    # Without the extra there is nothing to parse with, and that is a no-op --
    # not a crash on the very path that exists to avoid needing it.
    sys.modules["cryptography"] = None
    try:
        assert _resolve_tls_material(False, cf, kf)[0] == cert_pem
    finally:
        del sys.modules["cryptography"]


def test_split_san_values_partitions_names_and_ips():
    from biopb_tensor_server.core.tls import split_san_values

    dns, ips = split_san_values(["lab.example", " 10.0.0.5 ", "::1", "", "vpn-host"])
    assert dns == ["lab.example", "vpn-host"]
    assert ips == ["10.0.0.5", "::1"]


def test_extra_sans_land_in_the_generated_cert():
    """`--san` covers a name this host cannot discover about itself."""
    from biopb_tensor_server.core.tls import ensure_server_cert
    from cryptography import x509

    cert_pem, _ = ensure_server_cert(extra_sans=["vpn.lab.example", "10.8.0.4"])
    san = (
        x509.load_pem_x509_certificate(cert_pem)
        .extensions.get_extension_for_class(x509.SubjectAlternativeName)
        .value
    )
    assert "vpn.lab.example" in san.get_values_for_type(x509.DNSName)
    assert ipaddress.ip_address("10.8.0.4") in san.get_values_for_type(x509.IPAddress)


def test_extra_sans_ignored_when_reusing_an_existing_cert():
    from biopb_tensor_server.core.tls import ensure_server_cert
    from cryptography import x509

    first, _ = ensure_server_cert()
    again, _ = ensure_server_cert(extra_sans=["vpn.lab.example"])
    assert again == first  # reuse wins; widening requires --force
    san = (
        x509.load_pem_x509_certificate(again)
        .extensions.get_extension_for_class(x509.SubjectAlternativeName)
        .value
    )
    assert "vpn.lab.example" not in san.get_values_for_type(x509.DNSName)

    rotated, _ = ensure_server_cert(regenerate=True, extra_sans=["vpn.lab.example"])
    san = (
        x509.load_pem_x509_certificate(rotated)
        .extensions.get_extension_for_class(x509.SubjectAlternativeName)
        .value
    )
    assert "vpn.lab.example" in san.get_values_for_type(x509.DNSName)


def test_cert_init_san_requires_force_to_widen():
    from biopb_tensor_server.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["cert", "init"]).exit_code == 0
    out = runner.invoke(app, ["cert", "init", "--san", "vpn.lab.example"])
    assert out.exit_code == 0
    assert "--san ignored" in out.output


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
    """The full digest is printed, colon-grouped and unwrapped (copy-pasteable)."""
    from biopb._locations import tls_server_cert
    from biopb_tensor_server.cli import app
    from biopb_tensor_server.core.tls import cert_fingerprint, format_fingerprint

    result = CliRunner().invoke(app, ["cert", "init"])
    assert result.exit_code == 0, result.output
    assert tls_server_cert().exists()
    fp = format_fingerprint(cert_fingerprint(tls_server_cert().read_bytes()))
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


def _labelled(output: str, label: str) -> str:
    """The one output line starting with `  <label>:`."""
    return next(ln for ln in output.splitlines() if ln.strip().startswith(label + ":"))


def test_cert_init_paths_survive_a_narrow_terminal(monkeypatch):
    """cert/key/fingerprint are copied verbatim, so none may be wrapped.

    A path split mid-component is unusable in the mount/scp/trust-config it gets
    pasted into, and the reader cannot tell a wrap from a real path.
    """
    from biopb._locations import tls_server_cert, tls_server_key
    from biopb_tensor_server.cli import app

    monkeypatch.setenv("COLUMNS", "60")
    out = CliRunner().invoke(app, ["cert", "init"])
    assert out.exit_code == 0, out.output
    for label, path in (("cert", tls_server_cert()), ("key", tls_server_key())):
        assert str(path) in _labelled(out.output, label)
    # The fingerprint is 95 chars -- well past the 60-column width it must defeat.
    assert len(_labelled(out.output, "fingerprint")) > 60


def test_cert_init_prints_a_bracketed_path_intact(tmp_path, monkeypatch):
    """Rich markup is off, so a `[...]` directory name is not eaten as a style tag.

    This is the failure that does not announce itself: with markup on, a state
    dir named `st[ate]dir` prints as `stdir` -- a *wrong* path, rendered as
    confidently as a right one.
    """
    from biopb._locations import tls_server_cert
    from biopb_tensor_server.cli import app

    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "st[ate]dir"))
    out = CliRunner().invoke(app, ["cert", "init"])
    assert out.exit_code == 0, out.output
    assert "st[ate]dir" in out.output
    assert str(tls_server_cert()) in _labelled(out.output, "cert")


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
