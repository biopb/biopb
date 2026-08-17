"""The one data-plane resolver (biopb/biopb#615).

These tests exist because the bug they cover survived a green suite. Every
`cache-stats` test stubbed `_resolve_grpc_endpoint`, so the command could crash on
an unhandled ``TypeError`` for the whole of #618 with 91/91 passing — the one
caller of the one broken function was mocked out everywhere. So the seams here are
driven for real: a **real** loopback HTTP server stands in for the control, and
:func:`probe_scheme` is pointed at **real** plaintext and TLS listeners. What is
monkeypatched is the environment (``XDG_STATE_HOME``, ``BIOPB_*``), never the
function under test.
"""

import json
import os
import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from biopb import _data_plane

# A throwaway self-signed leaf + its key, generated once and valid 2026-2126.
# Static rather than minted at runtime because `cryptography` is deliberately not
# a dependency of this package (biopb/biopb#355) and the CI job that collects this
# directory installs `.[test,tensor]` — minting would silently skip the TLS branch
# in the only job that runs it. Never used by anything real.
_TEST_CERT = """-----BEGIN CERTIFICATE-----
MIIDNTCCAh2gAwIBAgIUJuk7aHUG4gt/2lYhivs8+GE2WHswDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQYmlvcGItdGVzdC1wcm9iZTAgFw0yNjA3MjgyMDExMzNa
GA8yMTI2MDcwNDIwMTEzM1owGzEZMBcGA1UEAwwQYmlvcGItdGVzdC1wcm9iZTCC
ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANm0i2ZrbTfVGpke2K9DvrJM
dSthIaPptwKSxX/QCt54j65J+aI1dwg6ujO2rgCJELXZ3QfGKF6QL52MGjeMGbYb
l/tuG8/Sfg0XcptnIJnAi4jvcWFuRSR16JMosWOIDjEO6r6TkLtcprudzAwzmYYb
o5fscuzJk1F/oBL2wVZIwFT1GPyaFMMBVgTEwSpdGy2R4FIyXB0ZwefUSZouayiQ
SEB0r6xxAdeW49L+KXk+gCjPtsybYDwLBitHqmzY6xi98zi8a4l8we4iGksr1jJA
GzG9qqx8atjQtVont5lRkX9fw6me/z3tV2e6TU4WdevkkQER5zHrjBuBlEgIbX8C
AwEAAaNvMG0wHQYDVR0OBBYEFDhfcjW5O53uEfq54jDJM0AM2ua3MB8GA1UdIwQY
MBaAFDhfcjW5O53uEfq54jDJM0AM2ua3MA8GA1UdEwEB/wQFMAMBAf8wGgYDVR0R
BBMwEYIJbG9jYWxob3N0hwR/AAABMA0GCSqGSIb3DQEBCwUAA4IBAQCM+Qx9SVRU
6P+LHe1sYN6zpSUp7Kt1ggDchXJL0MFcWt4+rNbVIQIZW6D7HdyRXFctinem9zu3
gMiF3cGTxnJ1f9o5Bs8K1uy82umvR53HW3RWk1VcRjkhz1pvCbao8ACO3OyuT+b2
e36Bykj5fBxGg5wGXoQz+BPX1b47SA3gnMhv2JGueDLfNu9yYrZCYoSX5YMobP4N
Xk/FMh4S+X0EEdAHFsKFVJc/7IdestADC1gYB1WK4HrErhfpR4UFIojd0M11X2L0
gV7Jm91081pDgxEWTJcwda9VWSSKKV0HHpaqCKpfeNUuyVWQqvGAr2iN2IZJwaGZ
y7qQ+eJnOaL9
-----END CERTIFICATE-----
"""

_TEST_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDZtItma2031RqZ
HtivQ76yTHUrYSGj6bcCksV/0AreeI+uSfmiNXcIOroztq4AiRC12d0HxihekC+d
jBo3jBm2G5f7bhvP0n4NF3KbZyCZwIuI73FhbkUkdeiTKLFjiA4xDuq+k5C7XKa7
ncwMM5mGG6OX7HLsyZNRf6AS9sFWSMBU9Rj8mhTDAVYExMEqXRstkeBSMlwdGcHn
1EmaLmsokEhAdK+scQHXluPS/il5PoAoz7bMm2A8CwYrR6ps2OsYvfM4vGuJfMHu
IhpLK9YyQBsxvaqsfGrY0LVaJ7eZUZF/X8Opnv897Vdnuk1OFnXr5JEBEecx64wb
gZRICG1/AgMBAAECggEAXd2zcSCGgdk3U6fyI3dhJH1E08RYfdUKXGiuERLBbPSs
dqhcouzMetbfa+arFX4Dn3TlETIGO+eNMC+1KhgVCejR2c263htS0BA5EPohG0ni
n9Mlnq2t0C+qbLDR8yk5fTuCSVNUxwQGu8Qos2YYHrOSELIZRzEOfMg7W5HbAHko
E3+wXuZlMimt0nlTGnAtWUzMZ95jeNeEEx/CsfOBYk6MTsOAwmxh1Ai4UV3tlrj4
+vvP+Xd69a/VUyheYupMi+ZBFEwrYUNMi/KK8kpzBL/+iZseh7VIcxPKWSyWeBhw
yhMK0olOAlxsvdhn2H5xAmDmBwl35BhA57jSlDfrEQKBgQDy10VZ4LevZ0bc0dGp
HEfuFw5Chqu78i469M1uhHQ8N9t2pPtLmwQ/5n9xpikcsajCSGBtB2RmyzWDuGBK
Z26zcdKDFLeWhGgQ2wDNcXmaMWopkc2AU0r0k7AkUW+dfKEecndkTVzwnJmLS9zT
z3Bys+6d/olRa5FR2WpgeIVhmQKBgQDlgJZ2Sm9rGy3d81lhv6/DhR0XJMivmCAz
dVfQn5+JJj/xbBNWcWSaALlCFd9qJJ0qDxpmLddTxgY25Bk3dVK9USvhQVDRmbVJ
I7xW/64SvAlGBP5AhOF7RIlBsIO0M80mzvDcxMKzMZJtFEWSyACWQAilEeKe+8IR
42tRqXTm1wKBgQDa3fCwd9u16DQy85yueUHPMdJ1XSFNLJJEKr04nYKRf5p6TWn8
E4P5/8nfaW3mYa0DJe5ade4kw4PA6x1GEgDxFGYyJCrvKvkMMAaCI4MA2Qag3rtD
rE6DLtTzdr5NR7WDVpGKwjtA1TOCG2a1NGJZzxgCKBYlXvjDt1usBRPaCQKBgAIb
rkYj5OYc98zkIVwOgLTREjVWNym1wgX2+/mEndiKq2eyUHMo032+p/T9cnHtKCxs
uxdZMHMqjIAQlFK4Fyx6BGcrTGzAdrPXSjGaY6T0aTllblh1YATb2k7qKiuLlkTW
/ctpW0h+GhQ6bXEtuSOoLuwlP+mp8lxrtF6pqdM9AoGAdqillmN+QBkug6V3ZRnq
O98ASRYn+ei4TQjl+vq+iK2q18CgvZfntYNuBjv44tuNMhvXsWEvOJ2M6lK5UmaB
dYHhV8V5elXUVsH+YQr3ooD0P+P6x/SlBIFVXLfw3dh+5nSmapPLVS2egJNsSwzB
DTisEXh80fn7dTQ97yUZJHc=
-----END PRIVATE KEY-----
"""


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Point every on-disk lookup (credential, cert, control record) at tmp_path.

    ``XDG_STATE_HOME`` alone is enough on POSIX, but CI also sets
    ``XDG_CONFIG_HOME``, and a stray ``BIOPB_*`` from the developer's shell would
    quietly win over everything these tests set up.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("BIOPB_TENSOR_URL", "BIOPB_TENSOR_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # A control the tests did not start must not be discovered: point the reader
    # at a port nothing is on, and let each test that wants one override it.
    monkeypatch.setenv("BIOPB_CONTROL_HOST", "127.0.0.1")
    monkeypatch.setenv("BIOPB_CONTROL_PORT", str(_free_port()))


def _free_port() -> int:
    """A port that was free a moment ago (nothing binds it)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def control(monkeypatch):
    """Start a real loopback HTTP control stub; return a `serve(payload, status)`.

    A real server, not a patched ``urlopen``: the resolver's job is to *ask
    something else* where the plane is, and a stubbed transport would not have
    caught #615's real defect either.
    """
    state = {"payload": {"control": "ok"}, "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
            body = json.dumps(state["payload"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # keep pytest output clean
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("BIOPB_CONTROL_PORT", str(httpd.server_address[1]))

    def serve(payload, status=200):
        state["payload"], state["status"] = payload, status

    try:
        yield serve
    finally:
        httpd.shutdown()
        httpd.server_close()


def _plaintext_listener() -> "tuple[socket.socket, int]":
    """A bare TCP listener that never speaks TLS."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)

    def accept_and_drop():
        try:
            conn, _ = sock.accept()
            conn.recv(4096)  # read the ClientHello, answer nothing
            conn.close()
        except OSError:
            pass

    threading.Thread(target=accept_and_drop, daemon=True).start()
    return sock, sock.getsockname()[1]


def _tls_listener(tmp_path) -> "tuple[socket.socket, int]":
    """A listener that completes a real TLS handshake with the test cert."""
    cert = tmp_path / "probe-cert.pem"
    key = tmp_path / "probe-key.pem"
    cert.write_text(_TEST_CERT)
    key.write_text(_TEST_KEY)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)

    def accept_and_handshake():
        try:
            conn, _ = sock.accept()
            try:
                ctx.wrap_socket(conn, server_side=True).close()
            except OSError:
                conn.close()
        except OSError:
            pass

    threading.Thread(target=accept_and_handshake, daemon=True).start()
    return sock, sock.getsockname()[1]


class TestResolutionOrder:
    """override -> $BIOPB_TENSOR_URL -> the control -> the default.

    The order is the whole point of #615: the control *knows* the endpoint (it
    chose the bind, the port and the scheme), so reconstructing one from defaults
    can only be a guess that goes stale. An explicit override still wins, because
    a plane launched outside any control is recorded nowhere and cannot be found.
    """

    def test_control_is_asked_when_nothing_is_explicit(self, control):
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})
        endpoint = _data_plane.resolve()
        assert endpoint.url == "grpc://127.0.0.1:9915"
        assert endpoint.origin == "control"

    def test_env_beats_the_control(self, control, monkeypatch):
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})
        monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://elsewhere:1234")
        endpoint = _data_plane.resolve()
        assert endpoint.url == "grpc://elsewhere:1234"
        assert endpoint.origin == "env"

    def test_override_beats_everything(self, control, monkeypatch):
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})
        monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://elsewhere:1234")
        endpoint = _data_plane.resolve("grpc://flag:5555")
        assert endpoint.url == "grpc://flag:5555"
        assert endpoint.origin == "flag"

    def test_falls_back_to_the_default_when_no_control_answers(self):
        # No control fixture here: BIOPB_CONTROL_PORT points at a free port.
        endpoint = _data_plane.resolve(probe=False)
        assert endpoint.url == _data_plane.default_url()
        assert endpoint.url.endswith(":8815")  # base 8810 + the flight offset
        assert endpoint.origin == "default"

    def test_the_default_port_is_derived_not_hardcoded(self):
        from biopb import _endpoints

        expected = _endpoints.flight_port_for(_endpoints.BASE_DEFAULT_PORT)
        assert _data_plane.default_url().endswith(f":{expected}")

    def test_the_fallback_scheme_comes_from_the_probe(self, monkeypatch):
        # #615 fault 1: the scheme was hardcoded `grpc://`, so a --tls plane was
        # dialed plaintext and reported as down. Nothing records a directly
        # launched plane's scheme, so it is asked of the socket instead. (A local
        # grpcs:// dial then needs the on-disk cert as its anchor, so seed one.)
        from biopb._locations import tls_server_cert

        cert = tls_server_cert()
        cert.parent.mkdir(parents=True, exist_ok=True)
        cert.write_bytes(b"PEMBYTES")
        monkeypatch.setattr(_data_plane, "probe_scheme", lambda *_a, **_k: "grpcs")
        assert _data_plane.resolve().url.startswith("grpcs://")

    def test_a_control_without_a_data_plane_url_is_no_answer(self, control):
        control({"control": "ok"})  # control up, plane never started
        assert _data_plane.resolve(probe=False).origin == "default"

    def test_a_failing_control_is_no_answer(self, control):
        control({"data_plane": {"grpc_url": "grpc://x:1"}}, status=500)
        assert _data_plane.resolve(probe=False).origin == "default"

    def test_every_origin_carries_a_readable_note(self, control):
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})
        assert "control" in _data_plane.resolve().origin_note
        assert "command line" in _data_plane.resolve("grpc://x:1").origin_note


class TestControlDiscovery:
    """`control_grpc_url` reads the supervisor snapshot off a live /health."""

    def test_reads_the_published_url(self, control):
        control({"data_plane": {"grpc_url": "grpcs://10.0.0.5:9000"}})
        assert _data_plane.control_grpc_url() == "grpcs://10.0.0.5:9000"

    def test_no_control_listening_is_none(self):
        assert _data_plane.control_grpc_url(timeout=0.5) is None

    def test_malformed_payload_is_none(self, control):
        control({"data_plane": "not-a-dict"})
        assert _data_plane.control_grpc_url() is None


class TestProbeScheme:
    """The scheme is asked of the listener, against real sockets."""

    def test_a_tls_listener_answers_grpcs(self, tmp_path):
        sock, port = _tls_listener(tmp_path)
        try:
            assert _data_plane.probe_scheme("127.0.0.1", port, timeout=5) == "grpcs"
        finally:
            sock.close()

    def test_a_plaintext_listener_answers_grpc(self):
        sock, port = _plaintext_listener()
        try:
            assert _data_plane.probe_scheme("127.0.0.1", port, timeout=5) == "grpc"
        finally:
            sock.close()

    def test_nothing_listening_answers_none(self):
        assert _data_plane.probe_scheme("127.0.0.1", _free_port(), timeout=0.5) is None


class TestTokenResolution:
    """explicit -> $BIOPB_TENSOR_TOKEN -> the control's credential file.

    The credential file was #615 fault 2: the core CLI read only the env var, so a
    token-gated local plane — the case biopb/biopb#470's handoff exists for —
    reported itself unreachable.
    """

    def _write_credential(self, token):
        from biopb._credentials import write_credential

        write_credential(token)

    def test_credential_file_is_read(self):
        self._write_credential("file-token")
        assert _data_plane.resolve_token() == "file-token"

    def test_env_beats_the_file(self, monkeypatch):
        self._write_credential("file-token")
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token")
        assert _data_plane.resolve_token() == "env-token"

    def test_explicit_beats_everything(self, monkeypatch):
        self._write_credential("file-token")
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token")
        assert _data_plane.resolve_token("flag-token") == "flag-token"

    def test_nothing_anywhere_is_none(self):
        # A tokenless local plane: unauthenticated is the correct answer.
        assert _data_plane.resolve_token() is None

    def test_blank_values_are_none_not_empty_string(self, monkeypatch):
        # "" would be sent as an empty Bearer header rather than omitted.
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "   ")
        assert _data_plane.resolve_token("") is None

    def test_the_endpoint_carries_the_resolved_token(self, control):
        self._write_credential("file-token")
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})
        assert _data_plane.resolve(probe=False).token == "file-token"

    def test_the_file_is_not_read_when_it_is_not_allowed(self):
        self._write_credential("file-token")
        assert _data_plane.resolve_token(allow_credential_file=False) is None

    def test_an_explicit_token_still_applies_without_the_file(self, monkeypatch):
        # Naming a server does not stop you naming its token.
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token")
        assert _data_plane.resolve_token(allow_credential_file=False) == "env-token"
        assert (
            _data_plane.resolve_token("flag-token", allow_credential_file=False)
            == "flag-token"
        )


class TestTheCredentialFollowsTheAddress:
    """The control's credential file rides only the endpoint the control named.

    That file is this machine's token for the plane the control owns. An address
    the user supplied went around the control on purpose -- it may be another
    user's server, or a store across the network -- so attaching the local
    credential to it would send a secret somewhere it was never issued for.
    """

    def _write_credential(self, token="file-token"):
        from biopb._credentials import write_credential

        write_credential(token)

    def test_a_control_named_endpoint_gets_the_file(self, control):
        self._write_credential()
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})

        endpoint = _data_plane.resolve(probe=False)

        assert endpoint.origin == "control"
        assert endpoint.token == "file-token"

    def test_a_server_flag_does_not(self, control):
        """Even with a control running and a credential on disk."""
        self._write_credential()
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})

        endpoint = _data_plane.resolve("grpc://data.mylab.example:8815", probe=False)

        assert endpoint.origin == "flag"
        assert endpoint.token is None

    def test_the_env_url_does_not_either(self, monkeypatch, control):
        self._write_credential()
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})
        monkeypatch.setenv("BIOPB_TENSOR_URL", "grpc://data.mylab.example:8815")

        endpoint = _data_plane.resolve(probe=False)

        assert endpoint.origin == "env"
        assert endpoint.token is None

    def test_a_loopback_flag_is_no_exception(self, control):
        """Locality is not the test -- the control's say-so is.

        A local address is still an address the user chose; the file belongs to
        the plane the control owns, which this may or may not be.
        """
        self._write_credential()
        control({"data_plane": {"grpc_url": "grpc://127.0.0.1:9915"}})

        assert _data_plane.resolve("grpc://127.0.0.1:9915", probe=False).token is None

    def test_the_guessed_default_does_not_get_it(self):
        """No control answered, so nothing vouches for what is on that port."""
        self._write_credential()

        endpoint = _data_plane.resolve(probe=False)

        assert endpoint.origin == "default"
        assert endpoint.token is None

    def test_an_explicit_token_reaches_a_flagged_endpoint(self):
        # The user names the server *and* its token -- the supported way to dial
        # something the control does not own.
        endpoint = _data_plane.resolve(
            "grpc://data.mylab.example:8815", "their-token", probe=False
        )
        assert endpoint.token == "their-token"

    def test_the_env_token_reaches_it_too(self, monkeypatch):
        monkeypatch.setenv("BIOPB_TENSOR_TOKEN", "env-token")
        endpoint = _data_plane.resolve("grpc://data.mylab.example:8815", probe=False)
        assert endpoint.token == "env-token"


class TestLocalTrustAnchor:
    """A local TLS plane is trusted from its cert on disk, never pinned."""

    def _seed_cert(self, body=b"PEMBYTES"):
        from biopb._locations import tls_server_cert

        path = tls_server_cert()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def test_plaintext_and_remote_planes_get_no_anchor(self):
        self._seed_cert()
        assert _data_plane.local_ca("grpc://localhost:8815") is None
        # A remote plane's cert is not on this disk and cannot be -- TOFU stays.
        assert _data_plane.local_ca("grpcs://data.mylab.example:8815") is None

    def test_a_local_tls_plane_is_anchored_on_the_on_disk_cert(self):
        self._seed_cert()
        for url in ("grpcs://localhost:8815", "grpcs://127.0.0.1:8815"):
            assert _data_plane.local_ca(url) == b"PEMBYTES"

    def test_a_missing_cert_errors_rather_than_falling_back_to_tofu(self):
        with pytest.raises(_data_plane.LocalTrustError, match="could not be read"):
            _data_plane.local_ca("grpcs://127.0.0.1:8815")

    def test_an_empty_cert_errors(self):
        self._seed_cert(b"  \n")
        with pytest.raises(_data_plane.LocalTrustError, match="empty"):
            _data_plane.local_ca("grpcs://127.0.0.1:8815")

    def test_resolve_attaches_the_anchor(self, control):
        self._seed_cert()
        control({"data_plane": {"grpc_url": "grpcs://127.0.0.1:8815"}})
        assert _data_plane.resolve().tls_ca_pem == b"PEMBYTES"

    @pytest.mark.skipif(
        os.name != "posix" or os.geteuid() == 0,
        reason="chmod 000 blocks neither root nor Windows, so the read would succeed",
    )
    def test_an_unreadable_cert_names_the_file(self):
        cert = self._seed_cert()
        cert.chmod(0o000)
        try:
            with pytest.raises(_data_plane.LocalTrustError) as exc:
                _data_plane.local_ca("grpcs://127.0.0.1:8815")
        finally:
            cert.chmod(0o600)
        assert str(cert) in str(exc.value)
