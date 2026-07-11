"""Tests for the observe web UI (_observe.py).

Routes are exercised over the standalone Starlette app via Starlette's
``TestClient`` — no real socket, no real kernel. The kernel is a ``MagicMock``
host returning canned ``execute`` results carrying the ``<<JOB_JSON>>`` envelope
the in-kernel job runner prints, exactly as ``test_mcp_server.py`` does.

The Host/Origin guard requires a loopback Host with a port, so the client's
``base_url`` is ``http://127.0.0.1:8766`` (matches the ``127.0.0.1:*`` allowlist
the real server enforces).
"""

import json
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from biopb_mcp.mcp import _observe, _server


def _reply(r, window_alive=True):
    """A kernel ``execute`` result whose stdout carries ``{"r": r, "w": ...}``."""
    env = {"r": r, "w": window_alive}
    return {
        "stdout": _server._JOB_DELIM + json.dumps(env) + "\n",
        "result_text": "",
        "error_text": "",
        "status": "ok",
    }


def _raw(status="ok", stdout="", error_text=""):
    return {
        "stdout": stdout,
        "result_text": "",
        "error_text": error_text,
        "status": status,
    }


@pytest.fixture
def host():
    h = MagicMock()
    h.health.return_value = {
        "alive": True,
        "ready": True,
        "start_error": None,
        "busy": False,
        "dead": False,
        "recent_respawns": 0,
        "watchdog_running": True,
    }
    h.execute.return_value = _reply([])  # jobs_summary() default: empty
    return h


@pytest.fixture(autouse=True)
def observe_state(host):
    """Install the mock host + snapshot/restore _observe + _server globals."""
    old_host = _server._kernel_host
    old_max = _observe._max_output_chars
    old_poll = _observe._poll_interval_ms
    old_mounted = _observe._mounted_http
    _server.set_kernel_host(host)
    _observe.configure(max_output_chars=20000, poll_interval_ms=3000)
    yield
    _server._kernel_host = old_host
    _observe._max_output_chars = old_max
    _observe._poll_interval_ms = old_poll
    _observe._mounted_http = old_mounted
    _observe._mw = None


@pytest.fixture
def client():
    return TestClient(
        _observe._build_standalone_app(), base_url="http://127.0.0.1:8766"
    )


# -- happy paths ------------------------------------------------------------


def test_observe_page_serves_html(client):
    r = client.get("/observe")
    assert r.status_code == 200
    assert "biopb-mcp" in r.text and "/api/jobs" in r.text
    # The poll interval is baked into the page (placeholder substituted).
    assert "__POLL_MS__" not in r.text
    assert "3000" in r.text


def test_observe_page_rebases_api_calls_for_fronting(client):
    # Behind the control front the page is served at /session/<id>/observe, so its
    # API calls must target /session/<id>/api/* -- not the control's own root
    # /api/*. It derives the base from window.location and prefixes every call, so
    # the same page works served directly (BASE "") or behind the prefix.
    html = client.get("/observe").text
    assert "const BASE = window.location.pathname" in html
    assert "BASE + '/api/jobs'" in html  # calls go through BASE
    # No bare root-absolute API call is left that would escape the session prefix.
    assert "fetch('/api" not in html
    assert "jpost('/api" not in html


def test_api_jobs_lists_summary(client, host):
    host.execute.return_value = _reply(
        [
            {
                "job_id": "job-1",
                "status": "running",
                "elapsed": 1.2,
                "stdout_len": 5,
                "code_preview": "print('hi')",
            }
        ]
    )
    r = client.get("/api/jobs")
    assert r.status_code == 200
    body = r.json()
    assert body["jobs"][0]["job_id"] == "job-1"
    assert body["jobs"][0]["code_preview"] == "print('hi')"
    assert body["headless"] is False


def test_api_job_detail(client, host):
    host.execute.return_value = _reply(
        {
            "job_id": "job-1",
            "code": "print('hi')",
            "status": "ok",
            "stdout": "hi",
            "result_text": "",
            "error_text": "",
            "cancel_reason": None,
            "elapsed": 0.1,
        },
        window_alive=True,
    )
    r = client.get("/api/jobs/job-1")
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == "print('hi')"
    assert body["stdout"] == "hi"
    assert body["truncated"] is False
    assert body["stdout_len"] == 2
    assert body["window_alive"] is True


def test_api_notebook_downloads_ipynb(client, host):
    host.execute.return_value = _reply(
        [
            {
                "job_id": "job-1",
                "code": "x = 1\nx",
                "status": "ok",
                "stdout": "",
                "result_text": "1",
                "error_text": "",
                "cancel_reason": None,
                "elapsed": 0.1,
                "created": 1_700_000_000.0,
            },
            {
                "job_id": "job-2",
                "code": "while True: pass",
                "status": "interrupted",
                "stdout": "",
                "result_text": "",
                "error_text": "Interrupted by user via the observe web UI.",
                "cancel_reason": "Interrupted by user via the observe web UI.",
                "elapsed": 4.0,
                "created": 1_700_000_004.0,
            },
        ]
    )
    r = client.get("/api/notebook")
    assert r.status_code == 200
    # The export round-trip asks the kernel for the full job history.
    assert "export()" in host.execute.call_args[0][0]
    assert r.headers["content-type"].startswith("application/x-ipynb+json")
    assert ".ipynb" in r.headers["content-disposition"]
    assert r.headers["x-filename"].endswith(".ipynb")

    nb = json.loads(r.text)
    assert nb["nbformat"] == 4
    # title (markdown) + bootstrap (code) + one cell per job.
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 3  # bootstrap + 2 jobs
    assert any("build_ops" in "".join(c["source"]) for c in code)  # bootstrap
    # The interrupted job is a live code cell flagged in its header + outputs.
    job2 = code[-1]
    assert "interrupted" in "".join(job2["source"])
    assert job2["metadata"]["biopb"]["status"] == "interrupted"
    assert any(o.get("name") == "stderr" for o in job2["outputs"])


def test_api_notebook_empty_session(client, host):
    host.execute.return_value = _reply([])
    r = client.get("/api/notebook")
    assert r.status_code == 200
    nb = json.loads(r.text)
    # Still a valid notebook: title + bootstrap, no job cells.
    assert nb["nbformat"] == 4
    assert any(c["cell_type"] == "code" for c in nb["cells"])


def test_api_interrupt_targets_running_job(client, host):
    host.execute.return_value = _reply({"job_id": "job-1", "interrupted": True})
    r = client.post("/api/kernel/interrupt")
    assert r.status_code == 200
    assert r.json()["interrupted"] is True
    # Forces the worker thread via interrupt_current (not a main-thread SIGINT).
    snippet = host.execute.call_args[0][0]
    assert "interrupt_current(" in snippet
    assert _observe._USER_INTERRUPT_MSG in snippet
    host.interrupt.assert_not_called()


def test_api_restart(client, host):
    r = client.post("/api/kernel/restart")
    assert r.status_code == 200 and r.json()["ok"] is True
    host.restart.assert_called_once()


def test_api_restart_failure_reports(client, host):
    host.restart.side_effect = RuntimeError("boom")
    r = client.post("/api/kernel/restart")
    assert r.status_code == 500
    assert r.json() == {"ok": False, "error": "boom"}


def test_api_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["alive"] is True and body["ready"] is True
    assert body["headless"] is False


# -- no kernel host ---------------------------------------------------------


def test_api_503_without_host(client):
    _server._kernel_host = None
    for method, path in [
        ("get", "/api/jobs"),
        ("get", "/api/jobs/job-1"),
        ("get", "/api/notebook"),
        ("post", "/api/kernel/interrupt"),
        ("post", "/api/kernel/restart"),
        ("get", "/api/status"),
    ]:
        r = getattr(client, method)(path)
        assert r.status_code == 503, path


def test_observe_page_without_host(client):
    _server._kernel_host = None
    assert client.get("/observe").status_code == 200


# -- truncation -------------------------------------------------------------


def test_detail_truncates_to_tail(client, host):
    _observe.configure(max_output_chars=50)
    big = "".join(str(i % 10) for i in range(200))
    host.execute.return_value = _reply(
        {
            "job_id": "job-1",
            "status": "ok",
            "stdout": big,
            "result_text": "",
            "error_text": "",
            "cancel_reason": None,
            "elapsed": 0.1,
        }
    )
    body = client.get("/api/jobs/job-1").json()
    assert body["truncated"] is True
    assert body["stdout_len"] == 200
    assert body["stdout"].startswith("…(truncated)…\n")
    assert body["stdout"].endswith(big[-50:])


# -- unknown / idle ---------------------------------------------------------


def test_detail_unknown_job_404(client, host):
    host.execute.return_value = _reply(
        {"job_id": "nope", "status": "unknown", "error_text": ""}
    )
    assert client.get("/api/jobs/nope").status_code == 404


# -- Host/Origin guard ------------------------------------------------------


def test_rejects_bad_origin(client):
    r = client.get("/api/jobs", headers={"origin": "http://evil.com"})
    assert r.status_code == 403


def test_rejects_bad_host(client):
    r = client.get("/api/jobs", headers={"host": "evil.com"})
    assert r.status_code == 421


def test_allows_loopback_origin(client):
    r = client.get("/api/jobs", headers={"origin": "http://127.0.0.1:8766"})
    assert r.status_code == 200


# -- content-type tolerance (we must NOT inherit the SDK's POST 400) ---------


def test_post_without_json_content_type_ok(client):
    r = client.post(
        "/api/kernel/interrupt",
        headers={"content-type": "text/plain"},
        content=b"",
    )
    assert r.status_code == 200


# -- busy kernel ------------------------------------------------------------


def test_busy_kernel_returns_200_marker(client, host):
    host.execute.return_value = _raw(status="busy")
    r = client.get("/api/jobs")
    assert r.status_code == 200
    body = r.json()
    assert body["busy"] is True and body["jobs"] == []


def test_kernel_error_returns_502(client, host):
    host.execute.return_value = _raw(status="error", error_text="kaboom")
    r = client.get("/api/jobs")
    assert r.status_code == 502


# -- describe() (server_status integration) ---------------------------------


def test_describe_not_running():
    _observe._mounted_http = False
    assert _observe.describe(8765) == {
        "running": False,
        "url": None,
        "mode": None,
    }


def test_describe_http_uses_mcp_port():
    _observe._mounted_http = True
    d = _observe.describe(8765)
    assert d["running"] is True
    assert d["url"] == "http://127.0.0.1:8765/observe"
    assert "http" in d["mode"]


# -- config defaults --------------------------------------------------------


def test_config_defaults():
    from biopb_mcp._config import get_setting

    assert get_setting({}, "mcp.observe.enabled") is True  # opt-out
    assert get_setting({}, "mcp.observe.max_output_chars") == 20000
    assert get_setting({}, "mcp.observe.poll_interval_ms") == 3000
