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

import ssl

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
    ``$BIOPB_STATE_HOME`` first, so set it rather than relying on
    ``Path.home()``); the per-process memo is cleared so each test starts cold;
    and the hostname-override probe is stubbed out, since it opens a real socket
    to the (fictional) host these tests name. The tests that exercise the probe
    itself override this stub.
    """
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(_tls, "_resolve_hostname_override", lambda *a, **k: None)
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
    assert _tls.resolve_tls_trust("grpc://localhost:8815") == _tls.NO_TLS


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
    assert _tls.resolve_tls_trust(loc).root_certs == CERT_A
    # The pin was persisted, so a cold resolve returns the same cert.
    _tls.clear_pin_cache()
    assert _tls.resolve_tls_trust(loc).root_certs == CERT_A


def test_changed_cert_raises_pin_mismatch(monkeypatch):
    loc = "grpc+tls://host:8815"
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    assert _tls.resolve_tls_trust(loc).root_certs == CERT_A

    # Same host now presents a different cert -> refuse with a clear error.
    # (Cold, i.e. a fresh client process: within one process the memo answers.)
    _tls.clear_pin_cache()
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_B)
    with pytest.raises(_tls.TlsPinMismatchError, match="does not match"):
        _tls.resolve_tls_trust(loc)


def test_distinct_hosts_pinned_independently(monkeypatch):
    certs = {"h1:1": CERT_A, "h2:2": CERT_B}
    monkeypatch.setattr(
        _tls, "_fetch_server_cert", lambda host, port: certs[f"{host}:{port}"]
    )
    assert _tls.resolve_tls_trust("grpc+tls://h1:1").root_certs == CERT_A
    assert _tls.resolve_tls_trust("grpc+tls://h2:2").root_certs == CERT_B


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
        assert _tls.resolve_tls_trust(loc).root_certs == CERT_A
    assert len(calls) == 1


def test_memoized_resolve_survives_a_failing_fetch(monkeypatch):
    """A transient side-handshake failure can't break an already-resolved host."""
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    loc = "grpc+tls://host:8815"
    assert _tls.resolve_tls_trust(loc).root_certs == CERT_A

    def _boom(host, port):
        raise OSError("connection refused")

    monkeypatch.setattr(_tls, "_fetch_server_cert", _boom)
    assert _tls.resolve_tls_trust(loc).root_certs == CERT_A


def test_clear_pin_cache_forces_refetch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _tls, "_fetch_server_cert", lambda host, port: (calls.append(1), CERT_A)[1]
    )
    loc = "grpc+tls://host:8815"
    _tls.resolve_tls_trust(loc)
    _tls.clear_pin_cache()
    _tls.resolve_tls_trust(loc)
    assert len(calls) == 2


# --- explicit trust: configured CA / fingerprint (biopb/biopb#604 item 4) -----


def test_explicit_ca_is_used_verbatim_and_skips_the_network(monkeypatch):
    """A configured anchor is the answer: no probe, and nothing pinned."""

    def _never(host, port):
        raise AssertionError("a configured CA must not trigger a cert fetch")

    monkeypatch.setattr(_tls, "_fetch_server_cert", _never)
    resolved = _tls.resolve_tls_trust("grpc+tls://host:8815", ca_pem=CERT_A)
    assert resolved.root_certs == CERT_A
    # The pin store stays untouched -- the config is the single source of truth.
    assert _tls._load_pins(_tls.tls_known_hosts()) == {}


def test_matching_fingerprint_accepts_the_presented_cert(monkeypatch):
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    resolved = _tls.resolve_tls_trust(
        "grpc+tls://host:8815", expected_fingerprint=_tls._fingerprint(CERT_A)
    )
    assert resolved.root_certs == CERT_A
    # Verified-first-use is self-contained: no pin is written, so the config can
    # never end up disagreeing with a state file the operator never edited.
    assert _tls._load_pins(_tls.tls_known_hosts()) == {}


def test_wrong_fingerprint_is_refused_on_the_very_first_connect(monkeypatch):
    """The point of configuring one: an impostor present at first contact loses.

    Plain TOFU would have pinned CERT_B here and reported success.
    """
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_B)
    with pytest.raises(_tls.TlsPinMismatchError, match="configured fingerprint"):
        _tls.resolve_tls_trust(
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
            _tls.resolve_tls_trust(
                "grpc+tls://host:8815", expected_fingerprint=spelling
            ).root_certs
            == CERT_A
        )


def test_one_endpoint_can_carry_different_trust_per_caller(monkeypatch):
    """The memo keys on the trust material, not just host:port.

    A downstream server fronting two upstreams that happen to resolve to the same
    ``host:port`` must not serve one's configured anchor to the other.
    """
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    loc = "grpc+tls://host:8815"
    assert _tls.resolve_tls_trust(loc, ca_pem=CERT_B).root_certs == CERT_B
    assert _tls.resolve_tls_trust(loc).root_certs == CERT_A  # TOFU, unaffected
    assert _tls.resolve_tls_trust(loc, ca_pem=CERT_B).root_certs == CERT_B


# --- hostname override (biopb/biopb#606) --------------------------------------
#
# A cert can pin fine and still fail every handshake, because gRPC matches the
# *dialed* name against the SANs separately from the trust anchor. Where the
# anchor is the presented leaf, that check is redundant and we substitute a name
# the cert does list. These drive the decision logic with the two probing
# handshakes stubbed; the real "does it actually complete a handshake" case needs
# a live TLS server and lives in biopb-tensor-server/tests/tls_test.py.

# The autouse fixture stubs the override resolver out (it opens a socket), so
# hold the real one from import time for the tests that are *about* it.
_REAL_OVERRIDE = _tls._resolve_hostname_override

DER_A = _tls._anchor_der(CERT_A)
DER_B = _tls._anchor_der(CERT_B)

SANS = (("DNS", "wrong.example"), ("DNS", "alt.example"), ("IP Address", "10.0.0.5"))


def _hostname_mismatch() -> ssl.SSLCertVerificationError:
    """The error OpenSSL raises when the dialed *name* isn't in the cert."""
    return _verify_error(
        "Hostname mismatch, certificate is not valid for 'lab-gpu.local'.", code=62
    )


def _ip_mismatch() -> ssl.SSLCertVerificationError:
    """The same verdict for a dialed *address* -- a different code and wording.

    Which of the two you get depends only on what was dialed, and the sidecar
    always dials an address, so anything that recognizes one must recognize both
    (biopb/biopb#916).
    """
    return _verify_error(
        "IP address mismatch, certificate is not valid for '127.0.0.1'.", code=64
    )


def _verify_error(message: str, *, code: int) -> ssl.SSLCertVerificationError:
    """An OpenSSL verification failure carrying *code* on ``verify_code``.

    Constructed rather than provoked: this suite mints no certs (see the module
    docstring), and ``verify_code`` is what the code under test reads.
    """
    err = ssl.SSLCertVerificationError(
        f"[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: {message}"
    )
    err.verify_code = code
    return err


def _expired() -> ssl.SSLCertVerificationError:
    """The error OpenSSL raises when the anchor is past its notAfter."""
    return _verify_error("certificate has expired", code=10)


def _stub_probe(monkeypatch, *, strict, lenient):
    """Stub the two probes: *strict* is the check_hostname=True verdict.

    Each is either an exception to raise or the ``(peercert, der)`` pair to
    return, so a test states only the handshake outcomes it cares about.
    """
    calls = []

    def _probe(host, port, pem, *, check_hostname):
        calls.append(check_hostname)
        outcome = strict if check_hostname else lenient
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(_tls, "_probe_peer", _probe)
    return calls


def test_pick_override_prefers_a_dns_name_over_an_ip():
    assert _tls._pick_override(SANS) == "wrong.example"
    assert _tls._pick_override((("IP Address", "10.0.0.5"),)) == "10.0.0.5"


def test_pick_override_skips_wildcards_and_empty_sans():
    # Substituting a wildcard would put a name that cannot exist on the wire as
    # SNI; there is nothing usable here, so the warning stands.
    assert _tls._pick_override((("DNS", "*.lab.local"),)) is None
    assert _tls._pick_override(()) is None
    # ... but a concrete name alongside a wildcard is fine.
    assert (
        _tls._pick_override((("DNS", "*.lab.local"), ("DNS", "gpu.lab.local")))
        == "gpu.lab.local"
    )


def test_no_override_when_the_dialed_name_is_covered(monkeypatch):
    """OpenSSL is the decider -- and a cert that matches is never touched.

    Notably this is what keeps a *wildcard* cert alone: a naive `host in sans`
    test would not see that `gpu.lab.local` matches `*.lab.local` and would
    override a connection that works.
    """
    calls = _stub_probe(monkeypatch, strict=({}, DER_A), lenient=({}, DER_A))
    assert _REAL_OVERRIDE("gpu.lab.local", 8815, CERT_A, tofu=True) is None
    assert calls == [True]  # no second handshake on the working path


def test_override_is_derived_when_the_dialed_name_is_absent(monkeypatch):
    calls = _stub_probe(
        monkeypatch,
        strict=_hostname_mismatch(),
        lenient=({"subjectAltName": SANS}, DER_A),
    )
    assert _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=True) == "wrong.example"
    assert calls == [True, False]


def test_an_address_mismatch_is_a_name_mismatch_too(monkeypatch):
    """The co-located sidecar's exact shape: dial 127.0.0.1, cert names a host.

    Read as a verify code rather than as message text -- OpenSSL words this one
    "IP address mismatch", so a check for "hostname mismatch" declined to help
    the one caller that always dials an address (biopb/biopb#916).
    """
    calls = _stub_probe(
        monkeypatch,
        strict=_ip_mismatch(),
        lenient=({"subjectAltName": (("DNS", "gpu-051.hpc.example"),)}, DER_A),
    )
    assert (
        _REAL_OVERRIDE("127.0.0.1", 8815, CERT_A, tofu=False) == "gpu-051.hpc.example"
    )
    assert calls == [True, False]


def test_no_override_when_the_anchor_is_not_the_presented_leaf(monkeypatch, caplog):
    """A private-CA anchor keeps the SAN check -- it is load-bearing there.

    Pinning the leaf is what makes the name check redundant; with a CA anchor any
    host in that PKI could impersonate any other, so this must not be applied
    blanket. The gate is cert identity rather than which mode we are in, so a
    server presenting something other than the anchor is refused an override too.
    """
    _stub_probe(
        monkeypatch,
        strict=_hostname_mismatch(),
        lenient=({"subjectAltName": SANS}, DER_B),  # presented != anchor
    )
    with caplog.at_level("WARNING"):
        assert _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=False) is None
    assert "load-bearing" in caplog.text


def test_no_usable_san_still_warns_with_the_mode_s_own_fix(monkeypatch, caplog):
    """A cert carrying no usable name: nothing to substitute, so name the fix.

    And name the *right* one -- a TOFU pin is cleared from the pin store, a
    configured fingerprint is updated in config (biopb/biopb#606).
    """
    for tofu, expected in ((True, "pin store"), (False, "fingerprint")):
        caplog.clear()
        _stub_probe(monkeypatch, strict=_hostname_mismatch(), lenient=({}, DER_A))
        with caplog.at_level("WARNING"):
            assert _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=tofu) is None
        assert "cert init --force --san" in caplog.text
        assert expected in caplog.text


def test_probe_failures_never_produce_an_override(monkeypatch):
    """A diagnostic must not turn a working connection into a broken one."""
    # A verification failure that is neither a name mismatch nor an expiry
    # -> not our business.
    _stub_probe(
        monkeypatch,
        strict=_verify_error("unable to get local issuer certificate", code=20),
        lenient=({"subjectAltName": SANS}, DER_A),
    )
    assert _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=True) is None

    # The probe itself can't connect -> silent.
    _stub_probe(
        monkeypatch, strict=OSError("connection refused"), lenient=OSError("nope")
    )
    assert _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=True) is None

    # The mismatch is real but the second handshake dies -> no cert to compare.
    _stub_probe(
        monkeypatch, strict=_hostname_mismatch(), lenient=OSError("connection reset")
    )
    assert _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=True) is None


def test_an_expired_certificate_is_raised_not_swallowed(monkeypatch):
    """Expiry is fatal and gRPC will not say why, so say it here (biopb/biopb#913).

    A pin is the trust anchor, not an exemption from validity: the handshake is
    refused, and the transport reports only "failed to connect to all addresses".
    The remediation names both halves -- re-mint, then drop the stale anchor --
    and which half is the client's depends on the mode.
    """
    for tofu, expected in ((True, "entry from"), (False, "configured TLS fingerprint")):
        calls = _stub_probe(monkeypatch, strict=_expired(), lenient=({}, DER_A))
        with pytest.raises(_tls.TlsCertExpiredError) as excinfo:
            _REAL_OVERRIDE("lab-gpu.local", 8815, CERT_A, tofu=tofu)
        assert "cert init --force" in str(excinfo.value)
        assert expected in str(excinfo.value)
        assert calls == [True]  # no point re-handshaking; the cert is done


def test_expiry_reaches_the_caller_of_resolve(monkeypatch):
    """The whole point: the reason surfaces at resolution, not in absl's stderr."""
    monkeypatch.setattr(_tls, "_resolve_hostname_override", _REAL_OVERRIDE)
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda h, p: CERT_A)
    _stub_probe(monkeypatch, strict=_expired(), lenient=({}, DER_A))
    with pytest.raises(_tls.TlsCertExpiredError):
        _tls.resolve_tls_trust("grpc+tls://lab-gpu.local:8815")


def test_anchor_der_rejects_a_bundle():
    """A CA chain is not a leaf, and is not silently treated as one."""
    assert _tls._anchor_der(CERT_A) is not None
    assert _tls._anchor_der(CERT_A + CERT_B) is None
    assert _tls._anchor_der(b"not a pem at all") is None


# --- the override on the resolved TlsTrust ------------------------------------


def test_resolved_trust_carries_the_override_and_memoizes_it(monkeypatch):
    """One probe per host:port -- the override rides the memo like the anchor."""
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    monkeypatch.setattr(_tls, "_resolve_hostname_override", _REAL_OVERRIDE)
    calls = _stub_probe(
        monkeypatch,
        strict=_hostname_mismatch(),
        lenient=({"subjectAltName": SANS}, DER_A),
    )

    loc = "grpc+tls://host:8815"
    first = _tls.resolve_tls_trust(loc)
    assert first.root_certs == CERT_A
    assert first.override_hostname == "wrong.example"

    assert _tls.resolve_tls_trust(loc) == first
    assert calls == [True, False]  # the memo answered the second time


def test_configured_ca_never_probes_and_never_overrides(monkeypatch):
    """`ca_pem` promises to stay offline -- and the usual one is a real CA."""

    def _never(*a, **k):
        raise AssertionError("a configured CA must not open a handshake")

    monkeypatch.setattr(_tls, "_resolve_hostname_override", _REAL_OVERRIDE)
    monkeypatch.setattr(_tls, "_probe_peer", _never)
    monkeypatch.setattr(_tls, "_fetch_server_cert", _never)
    trust = _tls.resolve_tls_trust("grpc+tls://host:8815", ca_pem=CERT_A)
    assert trust.root_certs == CERT_A
    assert trust.override_hostname is None


def test_key_id_separates_endpoints_and_anchors(monkeypatch):
    """The pool keys connections on this, so distinct trust must mean distinct id."""
    monkeypatch.setattr(_tls, "_fetch_server_cert", lambda host, port: CERT_A)
    loc = "grpc+tls://host:8815"
    tofu = _tls.resolve_tls_trust(loc).key_id
    configured = _tls.resolve_tls_trust(loc, ca_pem=CERT_B).key_id
    other_host = _tls.resolve_tls_trust("grpc+tls://other:8815").key_id
    assert len({tofu, configured, other_host}) == 3
    # Stable across resolves, and the full digest -- a collision would mean
    # handing one upstream's connection to another.
    assert _tls.resolve_tls_trust(loc, ca_pem=CERT_B).key_id == configured
    assert _tls._fingerprint(CERT_B) not in configured  # digests the PEM, not the DER
