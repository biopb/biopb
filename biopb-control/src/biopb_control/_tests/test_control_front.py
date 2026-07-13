"""Tests for the Layer-3 single-origin front (``_control`` ASGI app).

Concerns beyond the health/ensure control API (covered in ``test_supervisor``):
(1) the control's own routes win, (2) the ``/data_plane`` namespace faithfully
reverse-proxies to the tensor web sidecar -- method, path, query, headers,
request/response bodies, and the ``/ws/render`` WebSocket -- with the mount
prefix stripped, and (3) ``/`` serves the control's own dashboard, which is
backed by the in-process ``/api/status`` + ``/api/sessions`` control API. A
trivial stdlib HTTP server and a ``websockets`` echo server stand in for the
tensor sidecar so no real tensor server is needed.
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


def test_control_health_is_not_proxied(control):
    # /health is the control's own endpoint and must win over the proxy mounts.
    status, _headers, body = _get(f"{control}/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["control"] == "ok"
    assert "data_plane" in payload
    # It is the control answering, not the echo upstream.
    assert "path" not in payload


def test_root_serves_the_control_dashboard(control):
    # `/` is the control's own buildless dashboard, not a redirect or the proxy.
    status, headers, body = _get(f"{control}/")
    assert status == 200
    assert "text/html" in (headers.get("Content-Type") or "")
    html = body.decode()
    assert "biopb · control" in html  # its title/header
    # It polls the in-process control API, not the data-plane proxy.
    assert "/api/status" in html
    assert "/api/sessions" in html
    assert "/api/algorithms" in html  # the algorithm-plane section polls it
    assert "path" not in html  # not the echo upstream's response


def test_api_status_reports_control_and_data_plane(control):
    status, _headers, body = _get(f"{control}/api/status")
    assert status == 200
    payload = json.loads(body)
    assert payload["control"] == "ok"
    assert "state" in payload["data_plane"]  # the supervisor snapshot
    assert payload["sessions"] == 0  # none registered in this test


def test_api_sessions_lists_live_sessions(control, upstream):
    # Empty to start, then a registered session shows up with its observe link.
    _status, _headers, body = _get(f"{control}/api/sessions")
    assert json.loads(body)["sessions"] == []

    _register_session("20260101-000000-42", upstream)
    _status, _headers, body = _get(f"{control}/api/sessions")
    sessions = json.loads(body)["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "20260101-000000-42"
    assert sessions[0]["observe_url"] == "/session/20260101-000000-42/observe"


def test_api_sessions_omits_dead_records(control, upstream):
    # A session whose pid is gone is pruned by list_sessions() on read.
    dead = subprocess_dead_pid()
    u = urlparse(upstream)
    _config_sessions.register("ghost", host=u.hostname, port=u.port, pid=dead)
    _status, _headers, body = _get(f"{control}/api/sessions")
    assert json.loads(body)["sessions"] == []
    assert _config_sessions.read_session("ghost") is None  # pruned


@pytest.mark.parametrize(
    "health, expected",
    [
        ({"alive": True, "ready": True, "busy": False}, "ready"),
        ({"alive": True, "ready": True, "busy": True}, "busy"),
        ({"alive": True, "ready": False}, "starting"),  # process up, not booted
        ({"alive": False}, "none"),  # never started this session
        ({"alive": False, "start_error": "boom"}, "error"),  # failed to start
        ({"alive": False, "dead": True}, "error"),  # died
        ({"alive": True, "ready": True, "start_error": "old"}, "ready"),  # recovered
        ({}, "none"),  # missing keys -> not attached
    ],
)
def test_kernel_state_mapping(health, expected):
    from biopb_control._control import _kernel_state

    assert _kernel_state(health) == expected


def test_api_sessions_includes_kernel_state(control, upstream):
    # A live session gets a probed "kernel" field. The echo upstream's /api/status
    # is not a real health endpoint (it echoes the request), so no kernel keys are
    # present -> "none"; the point is the field is present and the wiring probed.
    _register_session("20260101-000000-42", upstream)
    _status, _headers, body = _get(f"{control}/api/sessions")
    sessions = json.loads(body)["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["kernel"] == "none"


def test_api_sessions_kernel_unknown_when_child_unreachable(control):
    # A live-pid record whose port has no server: the probe fails fast and the
    # session still lists (kernel state is decorative, never drops the session).
    closed = _free_port()  # nothing listening
    _config_sessions.register(
        "s-unreach", host="127.0.0.1", port=closed, pid=os.getpid()
    )
    _status, _headers, body = _get(f"{control}/api/sessions")
    sessions = json.loads(body)["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s-unreach"
    assert sessions[0]["kernel"] == "unknown"


def test_probe_kernel_maps_child_health():
    # _probe_kernel over a real loopback GET to a stub child reporting a ready
    # kernel returns "ready" (the full HTTP + parse + map path).
    import asyncio

    import httpx

    from biopb_control._control import _probe_kernel

    class _Health(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):  # noqa: N802
            payload = json.dumps({"alive": True, "ready": True, "busy": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rec = {"host": "127.0.0.1", "port": server.server_address[1]}

        async def go():
            async with httpx.AsyncClient(timeout=None) as c:
                return await _probe_kernel(c, rec)

        assert asyncio.run(go()) == "ready"
    finally:
        server.shutdown()


def test_data_plane_stop_is_a_control_verb(control):
    # POST /api/data_plane/stop returns the snapshot; with no data plane wired in
    # this test it just reports a stopped plane rather than proxying anywhere.
    req = urllib.request.Request(f"{control}/api/data_plane/stop", method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read())
    assert resp.status == 200
    assert payload["data_plane"]["state"] == "stopped"


def test_data_plane_restart_stops_then_ensures(tmp_path):
    # /restart must stop() BEFORE ensure() (so a racing supervision tick sees
    # want=False and backs off, rather than mistaking the down port for a
    # conflict), then wait for the plane and return the snapshot. Spy the
    # supervisor so the endpoint's orchestration is checked without spawning a
    # real tensor server.
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_port=_free_port(),
        web_port=_free_port(),
        server_log=tmp_path / "server.log",
    )
    sup = DataPlaneSupervisor(spec)
    calls = []
    sup.stop = lambda: calls.append("stop")
    sup.ensure = lambda: calls.append("ensure")
    sup.wait_until_up = lambda w: (calls.append("wait"), True)[1]
    sup.snapshot = lambda: {"state": "serving", "restarts": 1, "last_error": None}

    api_port = _free_port()
    server, _thread = serve_control_api(
        "127.0.0.1",
        api_port,
        sup,
        ensure_timeout=8.0,
        data_web_url="http://127.0.0.1:1",
    )
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/api/data_plane/restart", method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read())
        assert resp.status == 200
        assert payload["data_plane"]["state"] == "serving"  # the returned snapshot
        assert calls == ["stop", "ensure", "wait"]  # order matters
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# /api/* auth gate (§6.1 — the control's own API on the single origin)
# --------------------------------------------------------------------------- #
_TOKEN = "test-control-token-123"


@pytest.fixture
def tokened_control(upstream, tmp_path):
    """A control configured with a data-plane token; its /api/* is gated by it."""
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
        token=_TOKEN,
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


def test_control_api_requires_token_when_configured(tokened_control):
    # Missing token -> 401.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{tokened_control}/api/status")
    assert exc.value.code == 401
    # Wrong token -> 401.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{tokened_control}/api/status", headers={"Authorization": "Bearer nope"})
    assert exc.value.code == 401
    # Correct Bearer -> 200; the X-Biopb-Token scheme is accepted too.
    status, _h, body = _get(
        f"{tokened_control}/api/status", headers={"Authorization": f"Bearer {_TOKEN}"}
    )
    assert status == 200 and json.loads(body)["control"] == "ok"
    status, _h, _b = _get(
        f"{tokened_control}/api/sessions", headers={"X-Biopb-Token": _TOKEN}
    )
    assert status == 200


def test_ensure_verb_is_exempt_from_token(tmp_path, upstream):
    # biopb-mcp's _control_client POSTs ensure with no token; it stays open even
    # on a tokened control (idempotent, spawns the already-owned plane). Spy the
    # supervisor so the gate is exercised without actually launching a plane.
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
        token=_TOKEN,
    )
    sup = DataPlaneSupervisor(spec)
    sup.ensure = lambda: None
    sup.wait_until_up = lambda w: True
    sup.snapshot = lambda: {"state": "serving"}
    api_port = _free_port()
    server, _thread = serve_control_api(
        "127.0.0.1", api_port, sup, ensure_timeout=8.0, data_web_url=upstream
    )
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{api_port}/api/data_plane/ensure",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # no token -> 200
            assert resp.status == 200
    finally:
        server.shutdown()


def test_health_and_dashboard_are_exempt_from_token(tokened_control):
    # The liveness probe and the dashboard shell must load without a token (the
    # shell then sends the token on its /api/* fetches).
    assert _get(f"{tokened_control}/health")[0] == 200
    assert _get(f"{tokened_control}/")[0] == 200


def test_data_plane_proxy_is_not_gated_by_the_control_token(tokened_control):
    # /data_plane/* is the sidecar's surface (it re-validates the forwarded
    # token); the control's /api/ gate must not touch it, so it still proxies.
    status, _h, body = _get(f"{tokened_control}/data_plane/api/sources")
    assert status == 200
    assert json.loads(body)["path"] == "/api/sources"


def test_token_less_api_rejects_non_loopback_host(control):
    # No token -> /api/* falls back to a loopback Host guard (rebinding backstop);
    # a forged external Host is refused.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/api/status", headers={"Host": "evil.example:8813"})
    assert exc.value.code == 421


def test_token_less_api_refuses_forgeable_cross_site_write(control):
    # A browser cross-site POST (Sec-Fetch-Site: cross-site, no token header) is
    # refused as CSRF before it can reach the stop side-effect -- even though the
    # loopback Host would otherwise pass.
    req = urllib.request.Request(
        f"{control}/api/data_plane/stop",
        data=b"",
        method="POST",
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 403


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
# Agent-client registration API (/api/agents)
# --------------------------------------------------------------------------- #
# The endpoints are a thin front over biopb._agents; we stub that core so the
# tests never touch the machine's real client configs, and assert the wiring:
# GET lists, POST register/unregister pass the path id through and return the
# fresh status, an AgentError is a 400, and the whole surface is token-gated.


def _post(url, headers=None):
    req = urllib.request.Request(url, data=b"", method="POST", headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.headers, resp.read()


def test_api_agents_lists_client_status(control, monkeypatch):
    fake = [
        {
            "id": "cursor",
            "name": "Cursor",
            "state": "installed",
            "drifted": False,
            "config_path": "/x/mcp.json",
        },
    ]
    monkeypatch.setattr("biopb._agents.statuses", lambda: fake)
    status, _h, body = _get(f"{control}/api/agents")
    assert status == 200
    assert json.loads(body)["agents"] == fake


def test_agent_register_passes_id_and_returns_status(control, monkeypatch):
    seen = {}

    def fake_register(agent_id):
        seen["id"] = agent_id
        return {
            "id": agent_id,
            "name": "Cursor",
            "state": "registered",
            "drifted": False,
            "config_path": "/x/mcp.json",
        }

    monkeypatch.setattr("biopb._agents.register", fake_register)
    status, _h, body = _post(f"{control}/api/agents/cursor/register")
    assert status == 200
    assert seen["id"] == "cursor"
    assert json.loads(body)["agent"]["state"] == "registered"


def test_agent_unregister_passes_id_and_returns_status(control, monkeypatch):
    monkeypatch.setattr(
        "biopb._agents.unregister",
        lambda agent_id: {
            "id": agent_id,
            "name": "Cursor",
            "state": "installed",
            "drifted": False,
            "config_path": "/x/mcp.json",
        },
    )
    status, _h, body = _post(f"{control}/api/agents/cursor/unregister")
    assert status == 200
    assert json.loads(body)["agent"]["state"] == "installed"


def test_agent_action_error_is_400(control, monkeypatch):
    from biopb._agents import AgentError

    def boom(agent_id):
        raise AgentError("unknown agent client 'nope'")

    monkeypatch.setattr("biopb._agents.register", boom)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{control}/api/agents/nope/register")
    assert exc.value.code == 400


def test_api_agents_is_token_gated(tokened_control, monkeypatch):
    monkeypatch.setattr("biopb._agents.statuses", list)
    # No token -> 401 (it is /api/*, not exempt like /api/data_plane/ensure).
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{tokened_control}/api/agents")
    assert exc.value.code == 401
    # Correct token -> 200.
    status, _h, _b = _get(
        f"{tokened_control}/api/agents", headers={"Authorization": f"Bearer {_TOKEN}"}
    )
    assert status == 200


# --------------------------------------------------------------------------- #
# Algorithm-plane inspection API (/api/algorithms)
# --------------------------------------------------------------------------- #
# A thin, read-only front over biopb._algorithms; we stub that core so the tests
# never dial a real gRPC server, and assert the wiring: GET returns the probed
# server rows, a core error is a clean 500, and the surface is token-gated.


def test_api_algorithms_lists_probed_servers(control, monkeypatch):
    fake = [
        {
            "url": "grpc://localhost:50051",
            "target": "localhost:50051",
            "scheme": "grpc",
            "state": "serving",
            "ops": ["threshold", "segment"],
            "op_count": 2,
            "error": None,
            "single_op": False,
        },
    ]
    monkeypatch.setattr("biopb._algorithms.statuses", lambda: fake)
    status, _h, body = _get(f"{control}/api/algorithms")
    assert status == 200
    assert json.loads(body)["servers"] == fake


def test_api_algorithms_core_error_is_500(control, monkeypatch):
    def boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("biopb._algorithms.statuses", boom)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/api/algorithms")
    assert exc.value.code == 500


def test_api_algorithms_is_token_gated(tokened_control, monkeypatch):
    monkeypatch.setattr("biopb._algorithms.statuses", list)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{tokened_control}/api/algorithms")
    assert exc.value.code == 401
    status, _h, _b = _get(
        f"{tokened_control}/api/algorithms",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert status == 200


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
