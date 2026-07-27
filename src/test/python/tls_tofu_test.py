"""TOFU certificate pinning for the tensor client (biopb/biopb#604).

Client-side unit tests, no live server: the TOFU state machine is driven by
monkeypatching the cert fetch, and the pin store / location parsing are checked
directly. The server-backed round-trip (a real ``grpcs://`` TensorFlightServer)
lives in ``biopb-tensor-server/tests/tls_test.py``, alongside the server package
whose CI installs it (biopb/biopb#579).

**No ``cryptography`` here, deliberately.** The client-side TOFU implementation is
stdlib-``ssl`` only, and the CI job that gates this directory installs
``.[test,tensor]`` — neither of which pulls ``cryptography`` (the root package is
crypto-free by design, biopb/biopb#355). Minting certs at runtime would therefore
have silently skipped the mismatch / MITM-refusal branch in the only job that
collects it, so the two certs below are static fixtures instead: throwaway
self-signed leaves generated once, valid 2020-2120, never used by anything real.
"""

import pytest
from biopb.tensor import _tls

# Two distinct self-signed certs. Only their *bytes* matter to these tests -- the
# code under test fingerprints the DER and compares, and never validates a chain.
CERT_A = b"""-----BEGIN CERTIFICATE-----
MIIC7TCCAdWgAwIBAgIUCvsn4P/GHYNv2AWe4MG9fmLXPO0wDQYJKoZIhvcNAQEL
BQAwFzEVMBMGA1UEAwwMYmlvcGItdGVzdC1hMCAXDTIwMDEwMTAwMDAwMFoYDzIx
MjAwMTAxMDAwMDAwWjAXMRUwEwYDVQQDDAxiaW9wYi10ZXN0LWEwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQDsmQF0903387d2PKb/Cxu2PcRX8N6V1zxU
XiTrIvjyFmcnIGlxoS7RTV7NZnXKU4/cd6WZgZ8x8pdsnI/cvBk2hKj0Ci1mtw3t
A3oBE3hUKLuYv1EAA6PnZR6ZEZihBtxUICQMK2IzXie9HT6ljqpZnK/6ZDAm7e5l
AS4wd49wgKI/LrhUXW/0eeFBKl2knkg1/y0VgRmsNdPe4ckUX9bJP9BP0iunsUlm
6suNeOLQdCBwuMI4qy4kuIfr/ranLnEIxq69IUIJYFoH1bUcvDj4zIj6xcHVb5Mf
9/YI9bDHUdcZ6t02GQdB4wfHwPlKHUOLacML5gpxvcdQwBDaKgwbAgMBAAGjLzAt
MB0GA1UdEQQWMBSCDGJpb3BiLXRlc3QtYYcEfwAAATAMBgNVHRMBAf8EAjAAMA0G
CSqGSIb3DQEBCwUAA4IBAQCiXjI6cabGaeuTXKnE/veduXi0LJ1qEQJhyh4AR7nC
BAl55ivIzVEgrGPTjqVKqnUxONhc32CWEcPGGS5u2NGeAXpNIq+UKuKHC+OOCf7X
cj5bHHLbUxlezQ3H+ksUCZJiemUJHcnydBlrZQilK0UNgyVfWJu0j43Xb+TyXhFm
vTUlZSCvDvtc6/TYFyeZQ7yjdLnfJ9mnqq13aqeHjPZbbBP2pG9tp5NcKw9n50H/
/PWGhyj5yGtxIDZZMq5AmMFPwvjZ745oxIKgBnI5g1/SJZvV0tv01/9G4SWJ2bh7
VrdH4eEI4A1WCEoCx3J/oqf+jUHqQd9PQd6HvEZP3Hrn
-----END CERTIFICATE-----
"""

CERT_B = b"""-----BEGIN CERTIFICATE-----
MIIC7TCCAdWgAwIBAgIULh8fvw/B+PQ4nOe2+j+JVp7J4c0wDQYJKoZIhvcNAQEL
BQAwFzEVMBMGA1UEAwwMYmlvcGItdGVzdC1iMCAXDTIwMDEwMTAwMDAwMFoYDzIx
MjAwMTAxMDAwMDAwWjAXMRUwEwYDVQQDDAxiaW9wYi10ZXN0LWIwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQDoZRNmqEiivDxEwk3MG+g6OrAnVW4iCEMX
tQ7DMLHqd+d4n6qnI8MUpU48YrJ0qrx3tdEeNqN9PBdSSi1TWUpJxGZh8G5fe7PT
n2Y2Z3PLZsjesktDjQ74zUQh0UAOxS45UgnPvUo3pNMsAz7SFseDYZkl46ObpLQS
wbTE/GAt7SD73czmZ5+pX4Who+DLiFHJwCtlacpFVvvHWhzEFC0hXUexNFgpuhLK
dp4NkT2/6ZJZJx+IFEgezGj5il3BbTddsnLH1C91oxvyJjAyRFWMZQvuAzraRebj
GFkGSRmTWWSQWgZsxhwARAXG+A6F6pCyMtLM7V95iExMJjHZ7Gv/AgMBAAGjLzAt
MB0GA1UdEQQWMBSCDGJpb3BiLXRlc3QtYocEfwAAATAMBgNVHRMBAf8EAjAAMA0G
CSqGSIb3DQEBCwUAA4IBAQBNhn479l0GUnDwePA1TUwoMkebHeMd9GLBcxBcoYai
/KWnLWV/VAFJ8gwW066YOvWzvLY7grL/BXA4ggdlucXgWoxW/VNOSuagFTxcZKCq
dzwlz2qkjdNf9I9f47HVdzN6PTgR6OvD1JjsyXrE8m6P0T3vGfiyim9QW8ZotcVM
/RI1dN0Uh0Ah+CvC28YelZkv0kPqevn4nXQDmb+4Q9woFWVyfUeEoACD6WfSHF9A
BL0jWigh+rY8QlLw96XBhR7yOc7KNYP0R5Ugvwk1Tifa/DXz2r9andvd5oCXweG4
+ePtSArd1LmmFIbUd1nb75WXgEHNgqROg7at1AQEQ7bc
-----END CERTIFICATE-----
"""


@pytest.fixture(autouse=True)
def _isolate_pin_state(tmp_path, monkeypatch):
    """Isolate everything TOFU resolution touches outside the process.

    The pin store goes to a tmp state tree (``state_dir()`` honors
    ``$XDG_STATE_HOME`` first, and CI may set it, so set it rather than relying on
    ``Path.home()``); the per-process memo is cleared so each test starts cold;
    and the SAN diagnostic is stubbed out, since it opens a real socket to the
    (fictional) host these tests name.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(_tls, "_warn_if_dialed_name_not_in_cert", lambda *a: None)
    _tls.clear_pin_cache()
    yield
    _tls.clear_pin_cache()


# --- location parsing (no network) -------------------------------------------


def test_host_port_ignores_non_tls_schemes():
    assert _tls._host_port("grpc://host:8815") is None
    assert _tls._host_port("host:8815") is None
    assert _tls._host_port("grpc+tls://host:8815") == ("host", 8815)


def test_resolve_returns_none_for_plaintext():
    # Plaintext location needs no cert and must never touch the network.
    assert _tls.resolve_tls_root_certs("grpc://localhost:8815") is None


# --- pin store ---------------------------------------------------------------


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


# --- TOFU state machine (fetch monkeypatched) --------------------------------


def test_first_connect_pins_then_reuses(monkeypatch):
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)

    loc = "grpc+tls://host:8815"
    assert _tls.resolve_tls_root_certs(loc) == CERT_A
    # The pin was persisted, so a cold resolve returns the same cert.
    _tls.clear_pin_cache()
    assert _tls.resolve_tls_root_certs(loc) == CERT_A


def test_changed_cert_raises_pin_mismatch(monkeypatch):
    loc = "grpc+tls://host:8815"
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    assert _tls.resolve_tls_root_certs(loc) == CERT_A

    # Same host now presents a different cert -> refuse with a clear error.
    # (Cold, i.e. a fresh client process: within one process the memo answers.)
    _tls.clear_pin_cache()
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_B)
    with pytest.raises(_tls.TlsPinMismatchError, match="does not match"):
        _tls.resolve_tls_root_certs(loc)


def test_distinct_hosts_pinned_independently(monkeypatch):
    certs = {"h1:1": CERT_A, "h2:2": CERT_B}
    monkeypatch.setattr(
        _tls, "_fetch_server_cert", lambda host, port: certs[f"{host}:{port}"]
    )
    assert _tls.resolve_tls_root_certs("grpc+tls://h1:1") == CERT_A
    assert _tls.resolve_tls_root_certs("grpc+tls://h2:2") == CERT_B


# --- per-process memo --------------------------------------------------------


def test_repeat_resolve_does_not_refetch(monkeypatch):
    """The eager call sites must not open a handshake per GetFlightInfo."""
    calls = []

    def _fetch(host, port):
        calls.append((host, port))
        return CERT_A

    monkeypatch.setattr(_tls, "_fetch_server_cert", _fetch)
    loc = "grpc+tls://host:8815"
    for _ in range(5):
        assert _tls.resolve_tls_root_certs(loc) == CERT_A
    assert len(calls) == 1


def test_memoized_resolve_survives_a_failing_fetch(monkeypatch):
    """A transient side-handshake failure can't break an already-resolved host."""
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    loc = "grpc+tls://host:8815"
    assert _tls.resolve_tls_root_certs(loc) == CERT_A

    def _boom(host, port):
        raise OSError("connection refused")

    monkeypatch.setattr(_tls, "_fetch_server_cert", _boom)
    assert _tls.resolve_tls_root_certs(loc) == CERT_A


def test_clear_pin_cache_forces_refetch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _tls, "_fetch_server_cert", lambda host, port: (calls.append(1), CERT_A)[1]
    )
    loc = "grpc+tls://host:8815"
    _tls.resolve_tls_root_certs(loc)
    _tls.clear_pin_cache()
    _tls.resolve_tls_root_certs(loc)
    assert len(calls) == 2


# --- explicit trust: configured CA / fingerprint (biopb/biopb#604 item 4) -----


def test_explicit_ca_is_used_verbatim_and_skips_the_network(monkeypatch):
    """A configured anchor is the answer: no probe, and nothing pinned."""

    def _never(host, port):
        raise AssertionError("a configured CA must not trigger a cert fetch")

    monkeypatch.setattr(_tls, "_fetch_server_cert", _never)
    resolved = _tls.resolve_tls_root_certs("grpc+tls://host:8815", ca_pem=CERT_A)
    assert resolved == CERT_A
    # The pin store stays untouched -- the config is the single source of truth.
    assert _tls._load_pins(_tls.tls_known_hosts()) == {}


def test_matching_fingerprint_accepts_the_presented_cert(monkeypatch):
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    resolved = _tls.resolve_tls_root_certs(
        "grpc+tls://host:8815", expected_fingerprint=_tls._fingerprint(CERT_A)
    )
    assert resolved == CERT_A
    # Verified-first-use is self-contained: no pin is written, so the config can
    # never end up disagreeing with a state file the operator never edited.
    assert _tls._load_pins(_tls.tls_known_hosts()) == {}


def test_wrong_fingerprint_is_refused_on_the_very_first_connect(monkeypatch):
    """The point of configuring one: an impostor present at first contact loses.

    Plain TOFU would have pinned CERT_B here and reported success.
    """
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_B)
    with pytest.raises(_tls.TlsPinMismatchError, match="configured fingerprint"):
        _tls.resolve_tls_root_certs(
            "grpc+tls://host:8815", expected_fingerprint=_tls._fingerprint(CERT_A)
        )


def test_fingerprint_spelling_is_forgiving(monkeypatch):
    """Whatever the operator pasted -- our colon-grouped display form or bare hex."""
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    digest = _tls._fingerprint(CERT_A)
    grouped = ":".join(digest[i : i + 2] for i in range(0, len(digest), 2)).upper()
    for spelling in (digest, digest.upper(), grouped, f"  {grouped}  "):
        _tls.clear_pin_cache()
        assert (
            _tls.resolve_tls_root_certs(
                "grpc+tls://host:8815", expected_fingerprint=spelling
            )
            == CERT_A
        )


def test_one_endpoint_can_carry_different_trust_per_caller(monkeypatch):
    """The memo keys on the trust material, not just host:port.

    A downstream server fronting two upstreams that happen to resolve to the same
    ``host:port`` must not serve one's configured anchor to the other.
    """
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    loc = "grpc+tls://host:8815"
    assert _tls.resolve_tls_root_certs(loc, ca_pem=CERT_B) == CERT_B
    assert _tls.resolve_tls_root_certs(loc) == CERT_A  # TOFU, unaffected
    assert _tls.resolve_tls_root_certs(loc, ca_pem=CERT_B) == CERT_B
