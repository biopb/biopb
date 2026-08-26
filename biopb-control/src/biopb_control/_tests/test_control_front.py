"""Tests for the single-origin web front (``_control`` ASGI app).

Concerns beyond the health/ensure control API (covered in ``test_supervisor``):
(1) the control's own routes win, (2) the ``/data_plane`` namespace faithfully
reverse-proxies to the tensor web sidecar -- method, path, query, headers,
request/response bodies -- with the mount prefix stripped, and (3) the control is the single web origin: it serves the
built ``web/`` SPA bundle at its root, falling back to ``index.html`` for deep
links (``/``, ``/viewer``, ``/session/<id>/observe``) and serving hashed assets
as real files, and (4) which session-child roots it will proxy at all — ``api``
always, the ``console`` (an RCE into that session's kernel) only on a
loopback-bound control, ``/mcp`` never. A trivial stdlib HTTP server stands in
for the tensor sidecar so no real tensor server is needed; a tmp bundle stands
in for ``web/packages/app/dist``.
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
from biopb import _sessions

from biopb_control._control import (
    _SESSION_ALLOWED_ROOTS,
    _is_proxied_session_path,
    _loopback_url,
    _rewrite_shell_html,
    _session_proxy_roots,
    build_app,
    normalize_url_prefix,
    serve_control_api,
)
from biopb_control._supervisor import DataPlaneSpec, DataPlaneSupervisor


@pytest.mark.parametrize(
    "path, guarded",
    [
        ("/session/s1/api/jobs", True),
        ("/session/20260101-000000-42/api/kernel/restart", True),
        # Bare /session/<id>/api (no further segment) is still forwarded to the
        # child by session_proxy (segments[0]=="api"), so it must be gated too --
        # the guard derives from the same _SESSION_ALLOWED_ROOTS and can't drift.
        ("/session/s1/api", True),
        ("/session/s1/observe", False),  # SPA shell, not the API
        ("/session/s1", False),  # bare session root
        ("/session/s1/mcp", False),  # non-API surface (proxy allowlist 404s it)
        ("/sessionfoo/api/x", False),  # not a /session/<id>/ path
        ("/api/status", False),  # control's own API, handled by the other branch
        # Not proxied by default, so not gated by default either -- the console
        # root only exists where _session_proxy_roots put it (test below).
        ("/session/s1/console/execute", False),
    ],
)
def test_is_proxied_session_path(path, guarded):
    assert _is_proxied_session_path(path) is guarded


def test_console_root_is_gated_wherever_it_is_proxied():
    # The auth gate reads the proxy's own root set, so enabling the console
    # cannot open an unauthenticated execute path: the same switch that makes it
    # reachable makes it guarded.
    roots = _session_proxy_roots(console_enabled=True)
    assert _is_proxied_session_path("/session/s1/console/execute", roots) is True
    assert _is_proxied_session_path("/session/s1/api/jobs", roots) is True
    assert _is_proxied_session_path("/session/s1/mcp", roots) is False


def test_console_root_is_off_by_default():
    assert "console" not in _session_proxy_roots(console_enabled=False)
    assert "console" in _session_proxy_roots(console_enabled=True)
    # Enabling the console only adds; it never displaces the always-on root.
    assert _session_proxy_roots(console_enabled=True) >= _SESSION_ALLOWED_ROOTS


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
def web_bundle(tmp_path):
    """A minimal built SPA bundle for the control to serve (index.html + one
    hashed asset). The real bundle is ``web/packages/app/dist``; here a stand-in
    is enough to exercise static-file serving + the SPA shell fallback."""
    d = tmp_path / "webapp"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(
        '<!doctype html><div id="root"></div><!-- biopb-spa-shell -->'
    )
    (d / "assets" / "app.js").write_text("console.log('biopb');")
    return d


@pytest.fixture
def control(upstream, tmp_path, web_bundle):
    """The control front, serving ``web_bundle`` and proxying the data plane to
    ``upstream``; yields its base URL."""
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),  # closed: no data plane needed for proxy tests
        server_log=tmp_path / "server.log",
        static_dir=web_bundle,
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


def test_health_advertises_the_console_gate(control):
    # The observe page must know before it renders an editor, and only the
    # control knows this half. Unauthenticated like `auth_required`, and for the
    # same reason: the bundle needs it before it holds a token.
    _status, _headers, body = _get(f"{control}/health")
    # The fixture binds 127.0.0.1, so the console is on here.
    assert json.loads(body)["console_enabled"] is True


def test_root_serves_the_spa_shell(control):
    # `/` serves the SPA shell (index.html) from the bundle, not a redirect or
    # the proxy. The React router then renders the dashboard for this path.
    status, headers, body = _get(f"{control}/")
    assert status == 200
    assert "text/html" in (headers.get("Content-Type") or "")
    html = body.decode()
    assert "biopb-spa-shell" in html  # the served index.html
    assert '<div id="root">' in html
    assert "path" not in html  # not the echo upstream's response


def test_deep_link_falls_back_to_the_spa_shell(control):
    # A client route that names no static file (/viewer) must serve index.html so
    # the SPA boots and routes client-side, not 404.
    for path in ("viewer", "admin", "unlock"):
        status, headers, body = _get(f"{control}/{path}")
        assert status == 200, path
        assert "text/html" in (headers.get("Content-Type") or "")
        assert "biopb-spa-shell" in body.decode(), path


def test_static_asset_is_served_from_the_bundle(control):
    # A real file in the bundle is served as itself (not the SPA shell), so the
    # hashed JS/CSS resolve at the control root.
    status, _headers, body = _get(f"{control}/assets/app.js")
    assert status == 200
    assert "console.log('biopb')" in body.decode()


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
    _sessions.register("ghost", host=u.hostname, port=u.port, pid=dead)
    _status, _headers, body = _get(f"{control}/api/sessions")
    assert json.loads(body)["sessions"] == []
    assert _sessions.read_session("ghost") is None  # pruned


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
    _sessions.register("s-unreach", host="127.0.0.1", port=closed, pid=os.getpid())
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
# /api/data_plane/logs — the dashboard log monitor's backing tail
# --------------------------------------------------------------------------- #


def test_data_plane_logs_tails_the_server_log(control, tmp_path):
    # The endpoint returns the tail of the file the supervisor writes the data
    # plane to (the same tmp_path/server.log the `control` fixture configures).
    (tmp_path / "server.log").write_text("".join(f"line {i}\n" for i in range(1, 51)))
    status, headers, body = _get(f"{control}/api/data_plane/logs?lines=10")
    assert status == 200
    payload = json.loads(body)
    assert payload["exists"] is True
    assert payload["lines"] == [f"line {i}" for i in range(41, 51)]  # last 10
    assert payload["truncated"] is True  # 50 lines exist, 10 returned
    assert payload["path"].endswith("server.log")
    # A config editor / log tail must never be cached (each poll wants fresh output).
    assert (headers.get("Cache-Control") or "").lower() == "no-store"


def test_data_plane_logs_caps_lines(control, tmp_path):
    # An absurd ?lines is clamped to _LOG_TAIL_MAX_LINES rather than honored.
    from biopb_control._control import _LOG_TAIL_MAX_LINES

    (tmp_path / "server.log").write_text(
        "".join(f"l{i}\n" for i in range(_LOG_TAIL_MAX_LINES + 500))
    )
    _status, _h, body = _get(f"{control}/api/data_plane/logs?lines=999999")
    payload = json.loads(body)
    assert len(payload["lines"]) == _LOG_TAIL_MAX_LINES


def test_data_plane_logs_missing_file(control):
    # No log written yet -> a clean exists:false, empty lines, not a 500.
    status, _h, body = _get(f"{control}/api/data_plane/logs")
    assert status == 200
    payload = json.loads(body)
    assert payload["exists"] is False
    assert payload["lines"] == []


def test_data_plane_logs_bad_lines_param_falls_back(control, tmp_path):
    # A non-integer ?lines degrades to the default rather than erroring.
    (tmp_path / "server.log").write_text("only line\n")
    status, _h, body = _get(f"{control}/api/data_plane/logs?lines=abc")
    assert status == 200
    assert json.loads(body)["lines"] == ["only line"]


def test_tail_file_bounds_bytes_and_drops_fragment(tmp_path):
    # The byte window bounds the read: with a small max_bytes we get only the tail,
    # truncated is True, and the (probably-partial) leading line is dropped so no
    # half-line is surfaced.
    from biopb_control._control import _tail_file

    p = tmp_path / "big.log"
    p.write_text("".join(f"{i:04d}-payload\n" for i in range(1000)))
    lines, truncated = _tail_file(p, max_lines=1000, max_bytes=200)
    assert truncated is True
    assert lines  # some tail lines survived
    assert lines[-1] == "0999-payload"  # last full line intact
    # Every returned line is a complete "NNNN-payload" record (no fragment).
    assert all(len(line) == len("0000-payload") for line in lines)


# --------------------------------------------------------------------------- #
# /api/* auth gate (the control's own API on the single origin)
# --------------------------------------------------------------------------- #
_TOKEN = "test-control-token-123"


@pytest.fixture
def tokened_control(upstream, tmp_path, web_bundle):
    """A control configured with a data-plane token; its /api/* is gated by it."""
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
        token=_TOKEN,
        static_dir=web_bundle,
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


def test_ensure_verb_is_token_gated(tmp_path, upstream):
    # /api/data_plane/ensure is gated like every other /api/* route now that the
    # credential handoff (biopb/biopb#470) lets _control_client carry the token: no
    # token -> 401, correct token -> 200. Spy the supervisor so the gate is
    # exercised without actually launching a plane.
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
    url = f"http://127.0.0.1:{api_port}/api/data_plane/ensure"
    try:
        # No token -> 401.
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                urllib.request.Request(url, data=b"", method="POST"), timeout=5
            )
        assert exc.value.code == 401
        # Correct token -> 200 (the header also clears the CSRF gate on the POST).
        req = urllib.request.Request(
            url, data=b"", method="POST", headers={"X-Biopb-Token": _TOKEN}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()


def test_health_and_spa_shell_are_exempt_from_token(tokened_control):
    # The liveness probe and the SPA shell must load without a token (the shell's
    # JS then sends the token on its /api/* fetches).
    assert _get(f"{tokened_control}/health")[0] == 200
    assert _get(f"{tokened_control}/")[0] == 200
    assert _get(f"{tokened_control}/viewer")[0] == 200  # deep link, still tokenless


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


def test_data_plane_only_proxies_the_api_not_static(control):
    # The sidecar no longer serves static assets (the control owns the whole UI),
    # so there is no /data_plane/viewer mount. Only the data API/ws/health proxy
    # under /data_plane; the viewer itself is a control-served SPA route (/viewer,
    # covered above). A /data_plane/api/* call still reaches the sidecar.
    status, _headers, body = _get(f"{control}/data_plane/api/sources")
    assert status == 200
    assert json.loads(body)["path"] == "/api/sources"


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


def test_upstream_dies_mid_response_returns_502(tmp_path):
    # An upstream that ACCEPTS the connection and then closes it without a valid
    # HTTP response raises httpx.RemoteProtocolError, not ConnectError. The
    # broadened catch (biopb#420) must still turn it into a clean 502, never let it
    # escape as a 500 traceback.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    dead_port = listener.getsockname()[1]

    def _accept_then_drop():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return  # listener closed by teardown
            with conn:
                try:
                    conn.recv(65536)  # consume the request, then answer nothing
                except OSError:
                    pass

    thread = threading.Thread(target=_accept_then_drop, daemon=True)
    thread.start()

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
        data_web_url=f"http://127.0.0.1:{dead_port}",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"http://127.0.0.1:{api_port}/data_plane/x", timeout=5
            )
        assert exc.value.code == 502
    finally:
        listener.close()
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
    # No token -> 401 (every /api/* route is gated).
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


def test_api_algorithms_folds_in_kernel_plugins(control, monkeypatch):
    # The kernel-plugin "bring your own tool" listing (biopb-mcp#92) rides in the
    # same payload under `plugins`, from the stubbed core inspector.
    monkeypatch.setattr("biopb._algorithms.statuses", list)
    fake_plugins = {
        "dir": "/home/u/.config/biopb/kernel",
        "files": [{"name": "tool.py", "summary": "My tool."}],
        "entry_points": [{"name": "lab", "dist": "lab-tools 1.2"}],
    }
    monkeypatch.setattr("biopb._kernel_plugins.summary", lambda: fake_plugins)
    status, _h, body = _get(f"{control}/api/algorithms")
    assert status == 200
    assert json.loads(body)["plugins"] == fake_plugins


def test_api_algorithms_plugin_error_degrades_not_500(control, monkeypatch):
    # A servers sweep succeeds but the plugin inspector blows up: the panel still
    # returns 200 with an empty plugin listing rather than 500-ing the whole card.
    monkeypatch.setattr("biopb._algorithms.statuses", list)

    def boom():
        raise RuntimeError("plugin dir unreadable")

    monkeypatch.setattr("biopb._kernel_plugins.summary", boom)
    status, _h, body = _get(f"{control}/api/algorithms")
    assert status == 200
    assert json.loads(body)["plugins"] == {"dir": "", "files": [], "entry_points": []}


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
# Per-session observe proxy (/session/<id>/*)
# --------------------------------------------------------------------------- #
def _register_session(session_id, upstream_url):
    """Register a live session whose loopback target is the echo ``upstream``."""
    u = urlparse(upstream_url)
    _sessions.register(session_id, host=u.hostname, port=u.port, pid=os.getpid())


def test_session_observe_serves_the_spa_shell(control, upstream):
    # The observe *page* is the control-served SPA shell (React ObservePage), NOT
    # proxied to the child -- only its /api/* calls are. A live session gets the
    # shell; the SPA then reads its id from the path and calls /session/<id>/api/*.
    # (An unknown/dead session -> 404, see test_unknown_session_returns_404.)
    _register_session("20260101-000000-42", upstream)
    status, headers, body = _get(f"{control}/session/20260101-000000-42/observe")
    assert status == 200
    assert "text/html" in (headers.get("Content-Type") or "")
    html = body.decode()
    assert "biopb-spa-shell" in html  # the control's index.html, not the child
    assert "path" not in html  # not the echo upstream's proxied response


def test_session_api_is_proxied_with_query(control, upstream):
    _register_session("s1", upstream)
    _status, _headers, body = _get(f"{control}/session/s1/api/jobs?limit=5")
    echoed = json.loads(body)
    assert echoed["path"] == "/api/jobs?limit=5"
    assert echoed["method"] == "GET"


def test_session_proxy_drops_host_and_origin(control, upstream):
    # On the trusted hop to the child the proxy replaces Host with the loopback
    # target and drops the browser's Origin, so the child's own loopback guard
    # passes. (The browser-facing Host must itself be loopback now -- the control
    # gates /session/<id>/api/* on it, see the rebinding test below -- so the
    # forged-Host variant is exercised there as a *rejection*, not here.)
    _register_session("s1", upstream)
    port = urlparse(upstream).port
    _status, _headers, body = _get(
        f"{control}/session/s1/api/status",
        headers={"Origin": "http://evil.example:8813", "Host": "127.0.0.1:8813"},
    )
    echoed = json.loads(body)
    assert echoed["origin"] is None  # Origin stripped -> absent -> child allows it
    assert echoed["host"] == f"127.0.0.1:{port}"  # Host set to the loopback target


def test_session_api_rejects_non_loopback_host(control, upstream):
    # biopb/biopb#424 item 1: /session/<id>/api/* is gated with the same
    # same-origin + loopback-Host policy as /api/*, because the proxy hop strips
    # the child's own Host/Origin guard and session ids are guessable. A forged
    # external Host (DNS-rebinding) is refused at the control before it can reach
    # the child -- even though the observe surface is otherwise unauthenticated.
    _register_session("s1", upstream)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(
            f"{control}/session/s1/api/status",
            headers={"Host": "evil.example:8813"},
        )
    assert exc.value.code == 421


def test_session_api_bare_root_is_gated(control, upstream):
    # The bare /session/<id>/api (no further segment) is still forwarded to the
    # child by session_proxy, so the gate must catch it too -- a forged Host is
    # refused here rather than reaching the child. Guards against the guard and
    # the proxy allowlist drifting on this boundary.
    _register_session("s1", upstream)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/session/s1/api", headers={"Host": "evil.example:8813"})
    assert exc.value.code == 421


def test_session_api_refuses_forgeable_cross_site_write(control, upstream):
    # A browser cross-site POST to a mutating kernel verb (Sec-Fetch-Site:
    # cross-site, no token header) is refused as CSRF -- a guessed session id must
    # not let a visited page restart someone's kernel.
    _register_session("s1", upstream)
    req = urllib.request.Request(
        f"{control}/session/s1/api/kernel/restart",
        data=b"",
        method="POST",
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 403


def test_session_observe_shell_stays_open(control, upstream):
    # Only the /api/* surface is gated; the observe SPA shell (a plain GET serving
    # the app bundle) must still load even from a forged Host, so the app can
    # bootstrap and then make its (gated) same-origin /api/* calls.
    _register_session("20260101-000000-42", upstream)
    status, _headers, _body = _get(
        f"{control}/session/20260101-000000-42/observe",
        headers={"Host": "evil.example:8813"},
    )
    assert status == 200


def test_session_proxy_allowlists_api_surface(control, upstream):
    # Only /api/* is proxied to the child (the /observe page is the control's own
    # SPA shell). The child's /mcp agent transport (RCE, and this hop strips its
    # Host/Origin auth) must never be reachable -- not directly, and not via a
    # dot-traversal that httpx would collapse to /mcp.
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


def test_session_console_is_proxied_on_a_loopback_control(control, upstream):
    # The user console (code into the session's kernel) rides the same hop as the
    # data API, under its own root. The `control` fixture binds 127.0.0.1, which
    # is what enables it.
    _register_session("s1", upstream)
    req = urllib.request.Request(
        f"{control}/session/s1/console/execute",
        data=b'{"code": "1 + 1"}',
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        echoed = json.loads(resp.read())
    assert echoed["path"] == "/console/execute"
    assert echoed["method"] == "POST"


def test_session_roots_are_reachable_with_a_token(tokened_control, upstream):
    # The other half of the gate, and the half the browser depends on: a caller
    # that *does* present the token gets through to the child.
    #
    # Pinned because the observe page went a long time sending no token at all
    # (biopb#730). It failed silently -- a 401 body is valid JSON, so the page
    # read through it and rendered a healthy session as an idle one with a dead
    # kernel -- and nothing on either side said the contract was broken. This
    # asserts the scheme the SPA sends is the scheme the control accepts.
    _register_session("s-auth", upstream)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{tokened_control}/session/s-auth/api/jobs")
    assert exc.value.code == 401

    status, _headers, body = _get(
        f"{tokened_control}/session/s-auth/api/jobs",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert status == 200
    assert json.loads(body)["path"] == "/api/jobs"


def test_session_console_is_gated_like_the_api(control, upstream):
    # The console is the one proxied path whose payload is arbitrary code, so it
    # must not be reachable by DNS-rebinding or as a cross-site write. Same gate,
    # asserted here because a new root that slipped past _guarded would be an
    # unauthenticated execute.
    _register_session("s1", upstream)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(
            f"{control}/session/s1/console/execute",
            headers={"Host": "evil.example:8813"},
        )
    assert exc.value.code == 421

    req = urllib.request.Request(
        f"{control}/session/s1/console/execute",
        data=b"{}",
        method="POST",
        headers={"Sec-Fetch-Site": "cross-site", "Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 403


@pytest.mark.parametrize("path", ["console/execute", "chat/turn"])
def test_the_execute_capable_roots_are_post_only(control, upstream, path):
    # The CSRF gate skips safe methods (correctly -- safe verbs must not change
    # state), so a cross-site GET is forwarded unchecked to whatever the child
    # serves. `<img src=".../console/execute?code=...">` is the shape. Pinning
    # POST here means that claim does not depend on the child's method list --
    # which is the whole point, and is why chat is checked too: it is the same
    # RCE behind the same gate, and a promise about another package's route
    # table is exactly what this refuses to rely on.
    _register_session("s1", upstream)
    for method in ("GET", "HEAD", "PUT", "DELETE"):
        req = urllib.request.Request(f"{control}/session/s1/{path}?x=1", method=method)
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 405, method
    # The data API keeps every verb: only the execute-capable roots are narrowed,
    # which is what lets chat's history be a readable GET under `api`.
    status, _headers, _body = _get(f"{control}/session/s1/api/jobs")
    assert status == 200


@pytest.mark.parametrize("console_enabled, expected", [(True, 502), (False, 404)])
def test_console_root_follows_the_switch(tmp_path, console_enabled, expected):
    # The behavioral half of the local-mode gate, asserted on build_app so the
    # "off" case needs no public listener. The session child is deliberately a
    # closed port, which separates the two outcomes cleanly: 404 means the root
    # is not a route at all (nothing was forwarded), 502 means it was routed and
    # only the child was absent. Answering the question with no upstream also
    # keeps this independent of what a child would reply.
    from starlette.testclient import TestClient

    _sessions.register("s1", host="127.0.0.1", port=_free_port(), pid=os.getpid())
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
    )
    app = build_app(
        DataPlaneSupervisor(spec),
        8.0,
        f"http://127.0.0.1:{_free_port()}",
        console_enabled=console_enabled,
    )
    # A loopback base_url: TestClient otherwise sends `Host: testserver`, which
    # the gate refuses (421) before the routing question under test is reached.
    with TestClient(app, base_url="http://127.0.0.1:8813") as client:
        resp = client.post("/session/s1/console/execute", json={"code": "1 + 1"})
        assert resp.status_code == expected
        # The data API is unaffected either way -- the switch narrows one root.
        assert client.get("/session/s1/api/jobs").status_code == 502


@pytest.mark.parametrize("console_enabled, expected", [(True, 502), (False, 404)])
def test_chat_root_follows_the_same_switch(tmp_path, console_enabled, expected):
    # The chat turn runs arbitrary code in the session kernel, so it is the same
    # RCE the allowlist exists to keep off this origin and rides the same gate.
    # The flag reads "console" but means "this control is loopback-bound".
    from starlette.testclient import TestClient

    _sessions.register("s1", host="127.0.0.1", port=_free_port(), pid=os.getpid())
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
    )
    app = build_app(
        DataPlaneSupervisor(spec),
        8.0,
        f"http://127.0.0.1:{_free_port()}",
        console_enabled=console_enabled,
    )
    with TestClient(app, base_url="http://127.0.0.1:8813") as client:
        resp = client.post("/session/s1/chat/turn", json={"text": "hi"})
        assert resp.status_code == expected
        # Chat's *reads* are not on this root: they are ordinary API calls, and
        # stay reachable either way. That split is deliberate -- the gate above
        # enforces POST-only, so a history GET here would be forwarded
        # cross-site unchecked.
        assert client.get("/session/s1/api/chat/history").status_code == 502


class _StopServe(Exception):
    """Abort serve_control_api once build_app has been called (nothing binds)."""


@pytest.mark.parametrize(
    "host, expected",
    [("127.0.0.1", True), ("localhost", True), ("0.0.0.0", False), ("::", False)],
)
def test_bind_address_decides_the_console(
    monkeypatch, tmp_path, upstream, host, expected
):
    # The gate reads *this* listener's bind through the shared predicate: a
    # loopback control enables the console, a network-reachable one does not --
    # regardless of --remote or the data plane's own bind. Asserted on the call
    # into build_app, so the "public" cases need no public listener.
    captured = {}

    def _capture(*_args, console_enabled, **_kwargs):
        captured["console"] = console_enabled
        raise _StopServe

    monkeypatch.setattr("biopb_control._control.build_app", _capture)
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
    )
    with pytest.raises(_StopServe):
        serve_control_api(
            host,
            _free_port(),
            DataPlaneSupervisor(spec),
            ensure_timeout=8.0,
            data_web_url=upstream,
        )
    assert captured["console"] is expected


def test_unknown_session_returns_404(control):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/session/nope/observe")
    assert exc.value.code == 404


def test_dead_session_returns_404_and_is_pruned(control, upstream):
    # A record whose pid is gone must not be proxied; resolve() prunes it.
    dead = subprocess_dead_pid()
    u = urlparse(upstream)
    _sessions.register("ghost", host=u.hostname, port=u.port, pid=dead)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{control}/session/ghost/observe")
    assert exc.value.code == 404
    assert _sessions.read_session("ghost") is None  # pruned


def subprocess_dead_pid():
    """A pid that is (almost certainly) dead: spawn a child, reap it, reuse pid."""
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# --------------------------------------------------------------------------- #
# --url-prefix: publishing this origin under a reverse-proxy path prefix
# (biopb/biopb#728 — an Open OnDemand /node/<host>/<port>/ route).
# --------------------------------------------------------------------------- #
_PREFIX = "/node/mantis-051/29847"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("/node/h/29847", "/node/h/29847"),
        ("node/h/29847", "/node/h/29847"),  # leading slash supplied
        ("/node/h/29847/", "/node/h/29847"),  # trailing slash dropped
        ("//node//h//", "/node/h"),  # empty segments collapsed
        ("  /node/h  ", "/node/h"),
        ("/", None),  # the root origin is "no prefix"
        ("", None),
        (None, None),
    ],
)
def test_normalize_url_prefix(raw, expected):
    assert normalize_url_prefix(raw) == expected


@pytest.mark.parametrize(
    "hostile",
    [
        # WHATWG URL parsing resolves <base href="/\evil.com/"> against the
        # document to http://evil.com/ -- a backslash right after the leading
        # slash enters the authority, so every relative URL in the SPA leaves the
        # origin. Browsers strip tabs/newlines *before* parsing, so those reach
        # the same position.
        "/\\evil.com",
        "\\\\evil.com",
        "/a\t\\evil.com",
        "/a\n\\evil.com",
        "/a /b",
        # A scheme-ish prefix must not survive as one.
        "https://evil.com/x",
        # Query/fragment in a <base href> silently rewrites what relatives mean.
        "/a?b",
        "/a#b",
        # Percent-encoding cannot be both decoded (what the middleware strips
        # from scope["path"]) and encoded (what the shell carries) at once.
        "/a%2fb",
        # Dot segments make the served <base href> and the stripped path disagree.
        "/a/../b",
        "/a/./b",
        # HTML/JS metacharacters: escaped at both injection sites too, so this is
        # the outer of two layers.
        '/a"onerror=x',
        "/a<script>",
        "/a</script><script>",
    ],
)
def test_normalize_url_prefix_rejects_non_paths(hostile):
    # Fail closed: a prefix that is not a plain same-origin path is a refusal to
    # start, not a document whose every relative URL points somewhere else.
    with pytest.raises(ValueError, match="invalid URL prefix segment"):
        normalize_url_prefix(hostile)


def test_build_app_rejects_a_hostile_prefix(tmp_path, web_bundle):
    # The rule binds at the single consumer, so no entry point can route around
    # it by constructing a spec directly.
    spec = DataPlaneSpec(config=tmp_path / "config.json", grpc_port=_free_port())
    with pytest.raises(ValueError):
        build_app(
            DataPlaneSupervisor(spec),
            8.0,
            "http://127.0.0.1:1",
            static_dir=web_bundle,
            url_prefix="/\\evil.com",
        )


def test_rewrite_shell_html_injects_base_and_runtime_global():
    out = _rewrite_shell_html(
        '<!doctype html><html><head><meta charset="utf-8" />'
        '<link rel="icon" href="/favicon.ico" />'
        '<script type="module" crossorigin src="/assets/index-abc.js"></script>'
        '<link rel="stylesheet" href="//cdn.example/x.css">'
        '<a href="relative/page">x</a>'
        "</head><body></body></html>",
        "/node/h/29847",
    )
    # <base> comes first in <head>, before anything that could resolve against it.
    assert out.index("<base href=") < out.index("<meta")
    assert '<base href="/node/h/29847/">' in out
    # The runtime hook the SPA reads instead of the build-time BASE_URL.
    assert 'window.__BIOPB_BASE__="/node/h/29847"' in out
    # Root-absolute refs are rewritten (<base> cannot touch them)...
    assert 'href="/node/h/29847/favicon.ico"' in out
    assert 'src="/node/h/29847/assets/index-abc.js"' in out
    # ...while protocol-relative and already-relative URLs are left alone.
    assert 'href="//cdn.example/x.css"' in out
    assert 'href="relative/page"' in out


def test_rewrite_shell_html_without_head_still_injects_first():
    out = _rewrite_shell_html('<div id="root"></div>', "/p")
    assert out.startswith('<base href="/p/">')


@pytest.fixture
def prefix_bundle(tmp_path):
    """A bundle whose index.html looks like the real Vite output: a <head> and
    root-absolute refs to the entry chunk and the icons."""
    d = tmp_path / "prefixed-webapp"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(
        "<!doctype html><html><head>"
        '<link rel="icon" href="/favicon.ico" />'
        '<script type="module" crossorigin src="/assets/app.js"></script>'
        '</head><body><div id="root"></div><!-- biopb-spa-shell --></body></html>'
    )
    (d / "assets" / "app.js").write_text("console.log('biopb');")
    return d


@pytest.fixture
def prefixed_control(upstream, tmp_path, prefix_bundle):
    """A control published under ``_PREFIX``; yields its base URL (no prefix)."""
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
        static_dir=prefix_bundle,
        url_prefix=_PREFIX,
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


def test_prefixed_shell_carries_the_prefix(prefixed_control):
    # The portal passes the full path through, so the shell must come back
    # pointing at itself under the prefix -- otherwise every asset ref leaves the
    # app's namespace and the page is a blank <div id="root">.
    status, headers, body = _get(f"{prefixed_control}{_PREFIX}/")
    assert status == 200
    assert "text/html" in (headers.get("Content-Type") or "")
    html = body.decode()
    assert "biopb-spa-shell" in html
    assert f'<base href="{_PREFIX}/">' in html
    assert f'window.__BIOPB_BASE__="{_PREFIX}"' in html
    assert f'src="{_PREFIX}/assets/app.js"' in html


def test_prefixed_deep_link_and_asset_resolve(prefixed_control):
    # A deep link under the prefix falls back to the shell, and the asset the
    # shell now points at is served as the real file.
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}/viewer")
    assert status == 200
    assert "biopb-spa-shell" in body.decode()
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}/assets/app.js")
    assert status == 200
    assert "console.log('biopb')" in body.decode()


def test_bare_prefix_with_no_trailing_slash_serves_the_shell(prefixed_control):
    # `/node/h/p` (the prefix itself) is the app root, not a 404.
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}")
    assert status == 200
    assert "biopb-spa-shell" in body.decode()


def test_unprefixed_requests_still_work(prefixed_control):
    # biopb-mcp's _control_client and the installer poll /health over loopback
    # with no prefix; configuring one for the portal must not break them.
    status, _headers, body = _get(f"{prefixed_control}/health")
    assert status == 200
    assert json.loads(body)["control"] == "ok"
    status, _headers, body = _get(f"{prefixed_control}/api/status")
    assert status == 200 and json.loads(body)["control"] == "ok"
    # And the shell served to a direct localhost hit still boots: the browser
    # asks for <prefix>/assets/... on this same origin, which strips back off.
    _status, _headers, body = _get(f"{prefixed_control}/")
    assert f'src="{_PREFIX}/assets/app.js"' in body.decode()


def test_prefix_matches_only_on_a_segment_boundary(prefixed_control):
    # `/node/mantis-051/29847x` shares a string prefix but is not under it, so it
    # is left alone and falls through to the SPA catch-all like any unknown path
    # -- never half-stripped into `x/...`.
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}x/api/status")
    assert status == 200
    assert "biopb-spa-shell" in body.decode()  # the shell, not the control API


def test_prefixed_control_api_is_reachable(prefixed_control):
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}/api/status")
    assert status == 200
    assert json.loads(body)["control"] == "ok"
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}/health")
    assert status == 200 and json.loads(body)["control"] == "ok"


def test_prefixed_data_plane_proxy_strips_both_prefixes(prefixed_control):
    # Prefix stripped by the middleware, /data_plane by the Mount: the sidecar
    # still sees a root-relative path and knows nothing of either namespace.
    status, _headers, body = _get(f"{prefixed_control}{_PREFIX}/data_plane/api/x?a=1")
    assert status == 200
    assert json.loads(body)["path"] == "/api/x?a=1"


def test_prefixed_session_api_is_proxied(prefixed_control, upstream):
    _register_session("s-prefixed", upstream)
    _status, _headers, body = _get(
        f"{prefixed_control}{_PREFIX}/session/s-prefixed/api/jobs?limit=5"
    )
    assert json.loads(body)["path"] == "/api/jobs?limit=5"


@pytest.fixture
def tokened_prefixed_control(upstream, tmp_path, prefix_bundle):
    spec = DataPlaneSpec(
        config=tmp_path / "config.json",
        grpc_host="127.0.0.1",
        grpc_port=_free_port(),
        server_log=tmp_path / "server.log",
        token=_TOKEN,
        static_dir=prefix_bundle,
        url_prefix=_PREFIX,
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


def test_prefixed_api_is_still_token_gated(tokened_prefixed_control):
    # The reason the prefix stripper has to be the OUTERMOST middleware: the auth
    # gate matches on scope["path"], so a still-prefixed /node/h/p/api/... would
    # not look like /api/ to it and would reach the verb ungated -- an auth
    # bypass, not a 404.
    for path in ("/api/status", "/api/data_plane/restart"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"{tokened_prefixed_control}{_PREFIX}{path}")
        assert exc.value.code == 401, path
    status, _headers, body = _get(
        f"{tokened_prefixed_control}{_PREFIX}/api/status",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert status == 200 and json.loads(body)["control"] == "ok"


def test_prefixed_session_api_is_still_token_gated(tokened_prefixed_control, upstream):
    _register_session("s-gated", upstream)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{tokened_prefixed_control}{_PREFIX}/session/s-gated/api/jobs")
    assert exc.value.code == 401


def test_no_prefix_serves_the_shell_verbatim(control):
    # The whole feature is a no-op when unconfigured: no <base>, no injected
    # global, index.html served as the file it is.
    _status, _headers, body = _get(f"{control}/")
    html = body.decode()
    assert "<base href=" not in html
    assert "__BIOPB_BASE__" not in html


# --------------------------------------------------------------------------- #
# GET/PUT /api/mcp_config — the biopb-mcp settings surface the control owns
# (global config; sessions are ephemeral). Needs biopb-mcp co-installed (it is in
# the workspace venv; the stdlib-only control CI skips these).
# --------------------------------------------------------------------------- #

pytest.importorskip("biopb_mcp")


def _put(url, payload, headers=None):
    body = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def mcp_home(monkeypatch, tmp_path):
    """Point Path.home() at a tmp dir so mcp_config_path() (resolved per-request
    inside the in-process uvicorn thread) reads/writes an isolated location."""
    import pathlib

    home = tmp_path / "mcp_home"
    home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    return home


def test_mcp_config_get_returns_config_path_and_schema(control, mcp_home):
    cfg_dir = mcp_home / ".config" / "biopb"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "mcp-config.json").write_text(json.dumps({"transport": {"port": 9000}}))
    status, _headers, body = _get(f"{control}/api/mcp_config")
    assert status == 200
    payload = json.loads(body)
    assert payload["path"].endswith("mcp-config.json")
    assert payload["config"] == {"transport": {"port": 9000}}
    # The schema carries the section that the editor renders (labels/help/bounds).
    assert "transport" in payload["schema"]["properties"]


def test_mcp_config_get_tolerates_missing_file(control, mcp_home):
    status, _headers, body = _get(f"{control}/api/mcp_config")
    assert status == 200
    assert json.loads(body)["config"] == {}


def test_mcp_config_put_writes_valid_config(control, mcp_home):
    status, payload = _put(
        f"{control}/api/mcp_config", {"transport": {"kind": "http", "port": 8080}}
    )
    assert status == 200, payload
    assert payload["saved"] is True
    on_disk = json.loads(
        (mcp_home / ".config" / "biopb" / "mcp-config.json").read_text()
    )
    assert on_disk["transport"]["port"] == 8080


def test_mcp_config_put_rejects_bad_enum_and_range(control, mcp_home):
    status, payload = _put(
        f"{control}/api/mcp_config",
        {"transport": {"kind": "websocket", "port": 99999}},
    )
    assert status == 422, payload
    paths = {tuple(e["path"]) for e in payload["errors"]}
    assert ("transport", "kind") in paths
    assert ("transport", "port") in paths
    # Nothing is written on a rejected PUT.
    assert not (mcp_home / ".config" / "biopb" / "mcp-config.json").exists()


def test_mcp_config_put_rejects_unhashable_enum_value_as_422_not_500(control, mcp_home):
    # A list/dict where an enum scalar is expected must be a clean 422, not a 500
    # (Enum.ok is total, so the PUT validator never raises on unhashable input).
    status, payload = _put(
        f"{control}/api/mcp_config", {"transport": {"kind": ["http"]}}
    )
    assert status == 422, payload
    assert ("transport", "kind") in {tuple(e["path"]) for e in payload["errors"]}


def test_mcp_config_put_rejects_inverted_health_poll(control, mcp_home):
    # The cross-field rule is declared once, in biopb-mcp, and reaches this
    # endpoint through the shared checker -- this handler restates nothing, so a
    # rule added there cannot silently pass here (biopb/biopb#34).
    status, payload = _put(
        f"{control}/api/mcp_config",
        {"tensor": {"health_poll_min_interval": 90, "health_poll_max_interval": 10}},
    )
    assert status == 422, payload
    paths = {tuple(e["path"]) for e in payload["errors"]}
    # Both ends of the range are flagged, so the form can highlight the pair the
    # user has to reconcile (and the load path clamps both).
    assert ("tensor", "health_poll_min_interval") in paths
    assert ("tensor", "health_poll_max_interval") in paths
