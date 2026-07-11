"""Tests for the Layer-3 single-origin front (``_control`` ASGI app).

Two concerns beyond the health/ensure control API (covered in
``test_supervisor``): (1) the control's own routes win over the catch-all, and
(2) everything else is faithfully reverse-proxied to the tensor web sidecar --
method, path, query, headers, request/response bodies, and the ``/ws/render``
WebSocket. A trivial stdlib HTTP server and a ``websockets`` echo server stand
in for the tensor sidecar so no real tensor server is needed.
"""

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import websockets.sync.server
from websockets.sync.client import connect as ws_connect

from biopb_control._control import serve_control_api
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
                "body": body.decode() if body else "",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-From-Upstream", "yes")
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


def test_control_health_is_not_proxied(control):
    # /health is the control's own endpoint and must win over the catch-all.
    status, _headers, body = _get(f"{control}/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["control"] == "ok"
    assert "data_plane" in payload
    # It is the control answering, not the echo upstream.
    assert "path" not in payload


def test_get_is_proxied_with_path_query_and_auth(control):
    status, headers, body = _get(
        f"{control}/api/sources?limit=5",
        headers={"Authorization": "Bearer sekret"},
    )
    assert status == 200
    assert headers.get("X-From-Upstream") == "yes"  # response came from upstream
    echoed = json.loads(body)
    assert echoed["path"] == "/api/sources?limit=5"  # path + query preserved
    assert echoed["method"] == "GET"
    assert echoed["auth"] == "Bearer sekret"  # auth forwarded for re-validation


def test_post_body_is_proxied(control):
    req = urllib.request.Request(
        f"{control}/api/slice",
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
            urllib.request.urlopen(f"http://127.0.0.1:{api_port}/x", timeout=5)
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
        with ws_connect(f"ws://127.0.0.1:{api_port}/ws/render?token=t") as ws:
            ws.send("hello")
            assert ws.recv() == "echo:hello"  # text both ways
            assert ws.recv() == b"\x00\x01\x02"  # binary upstream -> client
    finally:
        server.shutdown()
