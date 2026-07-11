"""Tests for the Layer-3 single-origin front (``_control`` ASGI app).

Concerns beyond the health/ensure control API (covered in ``test_supervisor``):
(1) the control's own routes win, (2) the ``/data_plane`` namespace faithfully
reverse-proxies to the tensor web sidecar -- method, path, query, headers,
request/response bodies, and the ``/ws/render`` WebSocket -- with the mount
prefix stripped, and (3) ``/`` redirects to the dataviewer. A trivial stdlib
HTTP server and a ``websockets`` echo server stand in for the tensor sidecar so
no real tensor server is needed.
"""

import json
import os
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest
import websockets.sync.server
from biopb import _config_sessions
from websockets.sync.client import connect as ws_connect

from biopb_control._control import _loopback_url, serve_control_api
from biopb_control._supervisor import DataPlaneSpec, DataPlaneSupervisor


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _EchoHandler(BaseHTTPRequestHandler):
    """A stand-in tensor sidecar: echoes request line, query, a header, and body
    as JSON so the test can assert the proxy forwarded them verbatim."""

    def log_message(self, *_a):  # silence
        pass

    def _echo(self, body: bytes = b""):
        payload = json.dumps(
            {
                "path": self.path,
                "method": self.command,
                "auth": self.headers.get("Authorization"),
                "host": self.headers.get("Host"),
                "origin": self.headers.get("Origin"),
                "body": body.decode() if body else "",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-From-Upstream", "yes")
        # A latin-1 (non-ASCII) header value, e.g. a Content-Disposition filename.
        # The proxy must decode it as latin-1; UTF-8 would raise on the 0xE9 byte.
        self.send_header("Content-Disposition", 'attachment; filename="café.tiff"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        self._echo()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self._echo(self.rfile.read(length) if length else b"")


@pytest.fixture
def upstream():
    """A fake tensor web sidecar; yields its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


@pytest.fixture(autouse=True)
def _isolated_sessions(tmp_path, monkeypatch):
    """Point the session registry at a per-test dir (resolve() reads the env per
    request, so setting it here reaches the in-process uvicorn thread too)."""
    monkeypatch.setenv("BIOPB_SESSIONS_DIR", str(tmp_path / "sessions"))


@pytest.fixture
def control(upstream, tmp_path):
    """The control front, proxying to ``upstream``; yields its base URL."""
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),  # closed: no data plane needed for proxy tests
        server_log=tmp_path / "server.log",
    )
    sup = DataPlaneSupervisor(spec)
    api_port = _free_port()
    server, _thread = serve_control_api(
        "127.0.0.1", api_port, sup, ensure_timeout=8.0, data_web_url=upstream
    )
    try:
        yield f"http://127.0.0.1:{api_port}"
    finally:
        server.shutdown()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as resp:
        # resp.headers is a case-insensitive HTTPMessage; keep it (ASGI lowercases
        # header names on the wire, so a plain dict + cased .get would miss them).
        return resp.status, resp.headers, resp.read()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_a, **_k):
        return None  # surface the 3xx as an HTTPError instead of following it


def test_control_health_is_not_proxied(control):
    # /health is the control's own endpoint and must win over the proxy mounts.
    status, _headers, body = _get(f"{control}/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["control"] == "ok"
    assert "data_plane" in payload
    # It is the control answering, not the echo upstream.
    assert "path" not in payload


def test_root_redirects_to_dataviewer(control):
    # `/` sends the origin root to the dataviewer (until the dashboard lands).
    opener = urllib.request.build_opener(_NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as exc:
        opener.open(f"{control}/", timeout=5)
    assert exc.value.code in (302, 307)
    assert exc.value.headers["Location"] == "/data_plane/viewer/"


def test_data_api_is_proxied_with_prefix_stripped_query_and_auth(control):
    status, headers, body = _get(
        f"{control}/data_plane/api/sources?limit=5",
        headers={"Authorization": "Bearer sekret"},
    )
    assert status == 200
    assert headers.get("X-From-Upstream") == "yes"  # response came from upstream
    echoed = json.loads(body)
    # /data_plane stripped by the Mount; the sidecar sees a root-relative path.
    assert echoed["path"] == "/api/sources?limit=5"  # path + query preserved
    assert echoed["method"] == "GET"
    assert echoed["auth"] == "Bearer sekret"  # auth forwarded for re-validation


def test_dataviewer_static_is_proxied_with_prefix_stripped(control):
    # The /data_plane/viewer mount strips its (longer) prefix down to the sidecar
    # root, where the static app lives -- so assets resolve.
    _status, _headers, body = _get(f"{control}/data_plane/viewer/assets/app.js")
    assert json.loads(body)["path"] == "/assets/app.js"
    _status, _headers, root_body = _get(f"{control}/data_plane/viewer/")
    assert json.loads(root_body)["path"] == "/"  # index.html at the sidecar root


def test_latin1_response_header_is_proxied_not_500(control):
    # A non-ASCII (latin-1) upstream header must round-trip, not crash the proxy.
    # UTF-8-decoding the 0xE9 byte in "café.tiff" would raise -> 500.
    status, headers, _body = _get(f"{control}/data_plane/api/x")
    assert status == 200
    assert "café.tiff" in (headers.get("Content-Disposition") or "")


def test_loopback_url_maps_wildcards_and_brackets_ipv6():
    assert _loopback_url("0.0.0.0", 8814) == "http://127.0.0.1:8814"
    assert _loopback_url("", 8814) == "http://127.0.0.1:8814"
    assert _loopback_url("192.168.1.5", 8814) == "http://192.168.1.5:8814"
    # IPv6 must be bracketed so the :port suffix is unambiguous.
    assert _loopback_url("::", 8814) == "http://[::1]:8814"
    assert _loopback_url("::1", 8814) == "http://[::1]:8814"
    assert _loopback_url("fe80::1", 8814) == "http://[fe80::1]:8814"


def test_post_body_is_proxied(control):
    req = urllib.request.Request(
        f"{control}/data_plane/api/slice",
        data=b'{"z": 3}',
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        echoed = json.loads(resp.read())
    assert echoed["method"] == "POST"
    assert echoed["path"] == "/api/slice"
    assert echoed["body"] == '{"z": 3}'


def test_unknown_upstream_returns_502(tmp_path):
    # Proxy target down -> a clean 502, never a hang.
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_port=_free_port(),
        web_port=_free_port(),
        server_log=tmp_path / "server.log",
    )
    sup = DataPlaneSupervisor(spec)
    api_port = _free_port()
    server, _thread = serve_control_api(
        "127.0.0.1",
        api_port,
        sup,
        ensure_timeout=8.0,
        data_web_url=f"http://127.0.0.1:{_free_port()}",  # nothing listening
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            # Under the /data_plane namespace so it reaches the (down) proxy.
            urllib.request.urlopen(
                f"http://127.0.0.1:{api_port}/data_plane/x", timeout=5
            )
        assert exc.value.code == 502
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# WebSocket proxy (/ws/render)
# --------------------------------------------------------------------------- #
@pytest.fixture
def ws_upstream():
    """A websockets echo server standing in for the tensor /ws/render channel.

    On each text message it replies with a text frame then a binary frame,
    exercising both directions and both frame types through the proxy.
    """

    def handler(conn):
        for message in conn:
            conn.send(f"echo:{message}")
            conn.send(b"\x00\x01\x02")

    server = websockets.sync.server.serve(handler, "127.0.0.1", 0)
    host, port = server.socket.getsockname()[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# Per-session observe proxy (/session/<id>/*)
# --------------------------------------------------------------------------- #
def _register_session(session_id, upstream_url):
    """Register a live session whose loopback target is the echo ``upstream``."""
    u = urlparse(upstream_url)
    _config_sessions.register(session_id, host=u.hostname, port=u.port, pid=os.getpid())


def test_session_observe_is_proxied_prefix_stripped(control, upstream):
    _register_session("20260101-000000-42", upstream)
    _status, _headers, body = _get(f"{control}/session/20260101-000000-42/observe")
    # /session/<id> stripped by the Mount; the child sees a root-relative path.
    assert json.loads(body)["path"] == "/observe"


def test_session_api_is_proxied_with_query(control, upstream):
    _register_session("s1", upstream)
    _status, _headers, body = _get(f"{control}/session/s1/api/jobs?limit=5")
    echoed = json.loads(body)
    assert echoed["path"] == "/api/jobs?limit=5"
    assert echoed["method"] == "GET"


def test_session_proxy_drops_host_and_origin(control, upstream):
    # The child's loopback Host/Origin guard must pass on the trusted hop even
    # when the browser reached the control via some other host with an Origin.
    _register_session("s1", upstream)
    port = urlparse(upstream).port
    _status, _headers, body = _get(
        f"{control}/session/s1/api/status",
        headers={"Origin": "http://evil.example:8813", "Host": "evil.example:8813"},
    )
    echoed = json.loads(body)
    assert echoed["origin"] is None  # Origin stripped -> absent -> child allows it
    assert echoed["host"] == f"127.0.0.1:{port}"  # Host set to the loopback target


def test_session_proxy_allowlists_observe_surface(control, upstream):
    # Only /observe + /api/* are proxied. The child's /mcp agent transport (RCE,
    # and this hop strips its Host/Origin auth) must never be reachable -- not
    # directly, and not via a dot-traversal that httpx would collapse to /mcp.
    _register_session("s1", upstream)
    forbidden = [
        "mcp",  # direct
        "mcp/messages",  # under /mcp
        "",  # bare session root
        "healthz",  # any non-allowlisted root
        "api/../mcp",  # raw traversal
        "api/%2e%2e/mcp",  # encoded traversal (ASGI-decoded to `..` before the check)
    ]
    for path in forbidden:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"{control}/session/s1/{path}")
        assert exc.value.code == 404, path


def test_unknown_session_returns_404(control):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/session/nope/observe")
    assert exc.value.code == 404


def test_dead_session_returns_404_and_is_pruned(control, upstream):
    # A record whose pid is gone must not be proxied; resolve() prunes it.
    dead = subprocess_dead_pid()
    u = urlparse(upstream)
    _config_sessions.register("ghost", host=u.hostname, port=u.port, pid=dead)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/session/ghost/observe")
    assert exc.value.code == 404
    assert _config_sessions.read_session("ghost") is None  # pruned


def subprocess_dead_pid():
    """A pid that is (almost certainly) dead: spawn a child, reap it, reuse pid."""
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_websocket_render_is_proxied(ws_upstream, tmp_path):
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
    )
    sup = DataPlaneSupervisor(spec)
    api_port = _free_port()
    server, _thread = serve_control_api(
        "127.0.0.1", api_port, sup, ensure_timeout=8.0, data_web_url=ws_upstream
    )
    try:
        with ws_connect(
            f"ws://127.0.0.1:{api_port}/data_plane/ws/render?token=t"
        ) as ws:
            ws.send("hello")
            assert ws.recv() == "echo:hello"  # text both ways
            assert ws.recv() == b"\x00\x01\x02"  # binary upstream -> client
    finally:
        server.shutdown()
